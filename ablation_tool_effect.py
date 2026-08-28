#!/usr/bin/env python3
"""Does ablating the refusal direction change BEHAVIOUR, or only wording?

The published ablation result is measured on verbal refusal in no-tool chat, and
that measurement turned out to be classifier-dependent: for Qwen and Gemma the
apparent drop was ablation changing how the model refuses ("Certainly not…"
instead of "I can't…"), not whether it refuses. Under the three-way judge those
rows reverse.

That invites the obvious question, which this script answers: with tools on the
table, does ablation change the rate at which the model actually takes an unsafe
action? Unsafe-tool-call scoring is a predicate over the parsed calls, so it has
no classifier in it and cannot be confounded the same way.

Paired design — the same prompts under both conditions — so the flip counts in
each direction are reported, not just two marginal rates. Generations are saved so
the refusal-text side can be judged offline with judge_generations-style labelling.

  python ablation_tool_effect.py --model Qwen3-14B --hf-id Qwen/Qwen3-14B --layer 33 --n 200
"""
import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from tools.registry import get_domain  # noqa: E402
from rerun_against_relabels import load, OUT  # noqa: E402
from rerun_steering_gen import parse_tool_calls, find_layers  # noqa: E402
from render_utils import make_renderer, DM  # noqa: E402


def log(*a):
    print(*a, flush=True)


def boot_ci(v, n=2000, seed=0):
    v = np.asarray(v, float)
    if not len(v):
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    bs = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(n)]
    return (round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3))


def boot_ci_diff(a, b, n=2000, seed=0):
    """Paired CI on (b - a); same index is the same prompt."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.RandomState(seed)
    bs = []
    for _ in range(n):
        idx = rng.randint(0, len(a), len(a))
        bs.append(b[idx].mean() - a[idx].mean())
    return (round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf-id", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n", type=int, default=200, help="harmful tool-enabled prompts")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--mode", default="Tool-enabled normal")
    a = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_grad_enabled(False)

    d = load(a.model, with_prompts=True)
    sub = d[(d["mode"] == a.mode) & (d["request_type"] == "Harmful")]
    sub = sub.sample(min(a.n, len(sub)), random_state=11).reset_index(drop=True)
    log(f"[data] {len(sub)} harmful '{a.mode}' prompts")

    tok = AutoTokenizer.from_pretrained(a.hf_id, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    net = AutoModelForCausalLM.from_pretrained(
        a.hf_id, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
    layers = find_layers(net)
    render = make_renderer(tok, get_domain, log=log)

    dirs = OUT / f"directions_{a.model}.pt"
    if not dirs.exists():
        sys.exit(f"missing {dirs}; run rerun_against_relabels.py --stage gpu first")
    r_dev = torch.load(dirs, map_location="cpu")["r_text"].float().to("cuda", torch.bfloat16).view(-1)
    log(f"[load] {a.model} layer={a.layer} r_text from {dirs.name}")

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

    def ablate():
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            h = h - (h @ r_dev).unsqueeze(-1) * r_dev
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        hk = Hooks()
        for L in layers:
            hk.h.append(L.register_forward_hook(hook))
        return hk

    prompts = [render(r) for _, r in sub.iterrows()]

    @torch.no_grad()
    def gen(hook_fn=None):
        texts = []
        for i in range(0, len(prompts), a.bs):
            chunk = prompts[i:i + a.bs]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=4096).to("cuda")
            Lp = enc["input_ids"].shape[1]
            hk = hook_fn() if hook_fn else None
            with (hk if hk is not None else nullcontext()):
                o = net.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            texts += [tok.decode(o[j, Lp:], skip_special_tokens=False) for j in range(len(chunk))]
        return texts

    def score(texts):
        unsafe, called = [], []
        for (_, row), t in zip(sub.iterrows(), texts):
            calls = parse_tool_calls(t or "")
            called.append(bool(calls))
            unsafe.append(bool(calls) and
                          len(get_domain(DM[row["domain"]]).score_tool_calls_all(calls)) > 0)
        return np.array(unsafe), np.array(called)

    import time
    t0 = time.time(); base_txt = gen(); log(f"[gen] baseline ({time.time()-t0:.0f}s)")
    t0 = time.time(); abl_txt = gen(hook_fn=ablate); log(f"[gen] ablated ({time.time()-t0:.0f}s)")

    u0, c0 = score(base_txt)
    u1, c1 = score(abl_txt)

    to_unsafe = int((~u0 & u1).sum())     # ablation made it unsafe
    to_safe = int((u0 & ~u1).sum())       # ablation made it safe
    res = {
        "model": a.model, "layer": a.layer, "mode": a.mode, "n": int(len(sub)),
        "tools_native": bool(render.native_tools),
        "unsafe_baseline": round(float(u0.mean()), 3), "unsafe_baseline_ci": boot_ci(u0),
        "unsafe_ablated": round(float(u1.mean()), 3), "unsafe_ablated_ci": boot_ci(u1),
        "delta_unsafe": round(float(u1.mean() - u0.mean()), 3),
        "delta_ci_paired": boot_ci_diff(u0, u1),
        "flips_safe_to_unsafe": to_unsafe, "flips_unsafe_to_safe": to_safe,
        "called_tool_baseline": round(float(c0.mean()), 3),
        "called_tool_ablated": round(float(c1.mean()), 3),
    }
    log(f"\n== {a.model}: does ablation change the ACTION? ==")
    log(f"  unsafe rate  baseline {res['unsafe_baseline']:.3f} {res['unsafe_baseline_ci']}"
        f"  ->  ablated {res['unsafe_ablated']:.3f} {res['unsafe_ablated_ci']}")
    log(f"  paired delta {res['delta_unsafe']:+.3f}  95% CI {res['delta_ci_paired']}"
        f"   ({'significant' if not (res['delta_ci_paired'][0] <= 0 <= res['delta_ci_paired'][1]) else 'not significant'})")
    log(f"  flips: safe->unsafe {to_unsafe}, unsafe->safe {to_safe}")
    log(f"  made any tool call: {res['called_tool_baseline']:.3f} -> {res['called_tool_ablated']:.3f}")

    OUT.mkdir(exist_ok=True)
    (OUT / f"ablation_action_{a.model}.json").write_text(json.dumps(res, indent=2))
    raw = [{"id": r["id"], "domain": r["domain"], "user_prompt": r["user_prompt"],
            "base_text": bt, "abl_text": at,
            "base_unsafe": bool(x), "abl_unsafe": bool(y)}
           for (_, r), bt, at, x, y in zip(sub.iterrows(), base_txt, abl_txt, u0, u1)]
    (OUT / f"ablation_action_raw_{a.model}.json").write_text(json.dumps(raw, indent=1))
    log(f"\nwrote {OUT / f'ablation_action_{a.model}.json'}")


if __name__ == "__main__":
    main()
