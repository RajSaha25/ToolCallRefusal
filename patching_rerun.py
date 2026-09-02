#!/usr/bin/env python3
"""Rerun activation patching, with a degeneracy screen.

Patching is the load-bearing evidence for the paper's "partial mediator" claim:
overwrite a tool run's refusal projection with its matched no-tool value and count
how many unsafe calls revert to safe. The published percentages (18/24/62/46/40)
come from runs that predate both the chat-template fix and the degeneracy audit,
and no generations were kept, so they cannot be checked without regenerating.

This does that, and additionally:
  - builds tool prompts through render_utils, so Gemma gets its schemas injected
    rather than silently losing them,
  - screens both baseline and patched output for degeneracy and reports the flip
    rate over clean responses only,
  - scores with the tool-call predicate (no classifier anywhere),
  - saves every generation so the result can be re-audited without another run.

  python patching_rerun.py --model Qwen3-14B --hf-id Qwen/Qwen3-14B --layer 33
"""
import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from tools.registry import get_domain          # noqa: E402
from rerun_against_relabels import load, OUT, TRIPLE_KEY   # noqa: E402
from rerun_steering_gen import parse_tool_calls, find_layers  # noqa: E402
from render_utils import make_renderer, DM     # noqa: E402
from degeneracy import is_degenerate           # noqa: E402


def log(*a):
    print(*a, flush=True)


def boot_ci(v, n=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    bs = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(n)]
    return (round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf-id", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n", type=int, default=300, help="harmful tool-normal baseline")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--load-4bit", action="store_true",
                    help="NF4 weights with bf16 compute, for a 70B on an 80GB card. "
                         "Activations are then not identical to the bf16 run the "
                         "directions were extracted from; the result is flagged.")
    a = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_grad_enabled(False)

    d = load(a.model, with_prompts=True)
    tok = AutoTokenizer.from_pretrained(a.hf_id, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16)
        net = AutoModelForCausalLM.from_pretrained(
            a.hf_id, quantization_config=q, device_map="cuda", trust_remote_code=True).eval()
        log("[load] 4-bit NF4 weights (bf16 compute)")
    else:
        net = AutoModelForCausalLM.from_pretrained(
            a.hf_id, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
    layers = find_layers(net)
    render = make_renderer(tok, get_domain, log=log)

    dirs = OUT / f"directions_{a.model}.pt"
    if not dirs.exists():
        sys.exit(f"missing {dirs}; run rerun_against_relabels.py --stage gpu first")
    r_cpu = torch.load(dirs, map_location="cpu")["r_text"].float()
    r_dev = r_cpu.to("cuda", torch.bfloat16).view(-1)
    LAYER = a.layer
    log(f"[load] {a.model} layer={LAYER} r_text from {dirs.name}")

    # ---- matched no-tool targets, keyed by the same triple as the divergence metric
    h = d[d["request_type"] == "Harmful"]
    notool = h[h["mode"] == "No-tool chat"].drop_duplicates(TRIPLE_KEY)
    toolh = h[h["mode"] == "Tool-enabled normal"]

    @torch.no_grad()
    def project(frame):
        out = []
        prompts = [render(r) for _, r in frame.iterrows()]
        for i in range(0, len(prompts), a.bs):
            enc = tok(prompts[i:i + a.bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=4096).to("cuda")
            hs = net(**enc, output_hidden_states=True).hidden_states[LAYER]
            out.append((hs[:, -1, :].float().cpu() @ r_cpu).numpy())
        return np.concatenate(out)

    t0 = time.time()
    nt_proj = project(notool)
    key_of = lambda row: tuple(row[k] for k in TRIPLE_KEY)  # noqa: E731
    target_by_key = {key_of(r): float(p) for (_, r), p in zip(notool.iterrows(), nt_proj)}
    global_target = float(np.mean(nt_proj))
    log(f"[proj] {len(notool)} no-tool targets, mean={global_target:.1f} ({time.time()-t0:.0f}s)")

    base = toolh[toolh.apply(lambda r: key_of(r) in target_by_key, axis=1)]
    base = base.sample(min(a.n, len(base)), random_state=4).reset_index(drop=True)
    prompts = [render(r) for _, r in base.iterrows()]
    targets = np.array([target_by_key.get(key_of(r), global_target) for _, r in base.iterrows()])
    log(f"[data] {len(base)} harmful tool-normal prompts with a matched no-tool target")

    class Hooks:
        def __init__(self):
            self.h = []

        def remove(self):
            for x in self.h:
                x.remove()
            self.h = []

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.remove()

    def set_proj(t_batch):
        """Overwrite the projection onto r with the matched no-tool value."""
        tt = torch.as_tensor(t_batch, device="cuda", dtype=torch.bfloat16).view(-1, 1)

        def hook(m, i, o):
            hid = o[0] if isinstance(o, tuple) else o
            cur = hid @ r_dev
            hid = hid + (tt - cur).unsqueeze(-1) * r_dev
            return (hid,) + tuple(o[1:]) if isinstance(o, tuple) else hid

        hk = Hooks()
        hk.h.append(layers[LAYER - 1].register_forward_hook(hook))
        return hk

    @torch.no_grad()
    def gen(idx, hook_targets=None):
        texts = []
        for i in range(0, len(idx), a.bs):
            chunk = [prompts[j] for j in idx[i:i + a.bs]]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=4096).to("cuda")
            Lp = enc["input_ids"].shape[1]
            hk = set_proj(hook_targets[i:i + a.bs]) if hook_targets is not None else None
            with (hk if hk is not None else nullcontext()):
                o = net.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            texts += [tok.decode(o[k, Lp:], skip_special_tokens=False) for k in range(len(chunk))]
        return texts

    def unsafe_of(rows_idx, texts):
        out = []
        for j, t in zip(rows_idx, texts):
            row = base.iloc[j]
            calls = parse_tool_calls(t or "")
            out.append(bool(calls) and
                       len(get_domain(DM[row["domain"]]).score_tool_calls_all(calls)) > 0)
        return np.array(out)

    def any_call_of(texts):
        # A "flip to safe" that is really "stopped calling" is engagement suppression,
        # not safer action. Report the call rate alongside so the two can be told apart.
        return np.array([bool(parse_tool_calls(t or "")) for t in texts])

    t0 = time.time()
    all_idx = list(range(len(base)))
    base_txt = gen(all_idx)
    base_unsafe = unsafe_of(all_idx, base_txt)
    log(f"[base] unsafe {base_unsafe.sum()}/{len(base)} ({time.time()-t0:.0f}s)")

    idx = [j for j in all_idx if base_unsafe[j]]
    if not idx:
        log("[patch] no unsafe baseline rows; nothing to patch")
        return
    t0 = time.time()
    patch_txt = gen(idx, hook_targets=targets[idx])
    patch_unsafe = unsafe_of(idx, patch_txt)
    log(f"[patch] {len(idx)} patched ({time.time()-t0:.0f}s)")

    deg = np.array([is_degenerate(t) for t in patch_txt])
    flips = ~patch_unsafe                       # unsafe -> safe
    clean = ~deg
    patch_call = any_call_of(patch_txt)
    # decompose the flips: the patched run still called a tool and it scored safe,
    # versus the patched run emitted no tool call at all
    flip_safe_call = flips & patch_call
    flip_no_call = flips & ~patch_call
    res = {
        "model": a.model, "layer": LAYER, "n_baseline": int(len(base)),
        "n_unsafe_baseline": int(base_unsafe.sum()),
        "baseline_unsafe_rate": round(float(base_unsafe.mean()), 3),
        "baseline_any_call_rate": round(float(any_call_of(base_txt).mean()), 3),
        "flip_rate_all": round(float(flips.mean()), 3),
        "flip_rate_all_ci": boot_ci(flips),
        "patched_any_call_rate": round(float(patch_call.mean()), 3),
        "flip_via_safe_call": round(float(flip_safe_call.mean()), 3),
        "flip_via_no_call": round(float(flip_no_call.mean()), 3),
        "unsafe_given_call_patched": (round(float(patch_unsafe[patch_call].mean()), 3)
                                      if patch_call.sum() else None),
        "patched_degenerate": round(float(deg.mean()), 3),
        "n_clean": int(clean.sum()),
        "flip_rate_clean": (round(float(flips[clean].mean()), 3) if clean.sum() else None),
        "flip_rate_clean_ci": (boot_ci(flips[clean]) if clean.sum() > 1 else None),
        "baseline_degenerate": round(float(np.mean([is_degenerate(t) for t in base_txt])), 3),
        "tools_native": bool(render.native_tools),
    }
    verdict = ("OK" if res["patched_degenerate"] < 0.05
               else "SUSPECT" if res["patched_degenerate"] < 0.5 else "ARTIFACT")
    res["verdict"] = verdict

    log(f"\n== {a.model} patching ==")
    log(f"  baseline unsafe      {res['n_unsafe_baseline']}/{res['n_baseline']}"
        f" ({res['baseline_unsafe_rate']:.3f}), {res['baseline_degenerate']:.1%} degenerate")
    log(f"  flipped to safe      {res['flip_rate_all']:.3f} {res['flip_rate_all_ci']}  (all patched)")
    log(f"    via a safe call    {res['flip_via_safe_call']:.3f}")
    log(f"    via no call        {res['flip_via_no_call']:.3f}")
    # every patched row had a call at baseline (that is how it was selected), so the
    # patched-subset call rate starts at 1.0 by construction
    log(f"  any-call, patched rows  1.000 -> {res['patched_any_call_rate']:.3f}"
        f"   unsafe|call after patch: {res['unsafe_given_call_patched']}"
        f"   (baseline any-call over all {res['n_baseline']}: {res['baseline_any_call_rate']:.3f})")
    res["quantized_4bit"] = bool(a.load_4bit)
    log(f"  patched degenerate   {res['patched_degenerate']:.1%}   -> {verdict}")
    if res["flip_rate_clean"] is not None:
        log(f"  flipped to safe      {res['flip_rate_clean']:.3f} {res['flip_rate_clean_ci']}"
            f"  (clean only, n={res['n_clean']})")

    OUT.mkdir(exist_ok=True)
    (OUT / f"patching_{a.model}.json").write_text(json.dumps(res, indent=2))
    raw = [{"id": base.iloc[j]["id"], "domain": base.iloc[j]["domain"],
            "target_proj": float(targets[j]), "baseline_text": base_txt[j],
            "patched_text": t, "patched_unsafe": bool(u), "patched_any_call": bool(k),
            "degenerate": bool(g)}
           for j, t, u, k, g in zip(idx, patch_txt, patch_unsafe, patch_call, deg)]
    (OUT / f"patching_raw_{a.model}.json").write_text(json.dumps(raw, indent=1))
    log(f"\nwrote {OUT / f'patching_{a.model}.json'}")


if __name__ == "__main__":
    main()
