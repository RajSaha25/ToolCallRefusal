#!/usr/bin/env python3
"""GPU half of the ablation / addition / steering rerun.

run_scaled_evaluation.py scores refusal inline through the Anthropic judge, which
needs JUDGE_KEY on the machine holding the GPU. This script instead generates and
dumps every completion that needs a refusal label, so the judging can happen where
the key already lives (see judge_generations.py). Unsafe-tool-call scoring is
predicate-based, so it stays here.

The hooks, grid and sample sizes mirror run_scaled_evaluation.py exactly, so the
numbers are comparable to the published ones -- only the refusal classifier changes.

  python rerun_steering_gen.py --model Qwen3-14B --hf-id Qwen/Qwen3-14B --layer 33
"""
import argparse
import json
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from tools.registry import get_domain  # noqa: E402
from rerun_against_relabels import load, OUT  # noqa: E402

DEVICE, DTYPE = "cuda", torch.bfloat16
DM = {"Finance": "finance", "Healthcare": "healthcare",
      "Education / Academic Integrity": "education", "Legal / Compliance": "legal"}


def log(*a):
    print(*a, flush=True)


def parse_tool_calls(text):
    m = re.search(r"\[TOOL_CALLS\]\s*(\[.*?\])", text, re.DOTALL)
    if m:
        try:
            p = json.loads(m.group(1))
            if isinstance(p, list):
                return [{"name": c.get("name", ""), "arguments": c.get("arguments", {})} for c in p]
        except json.JSONDecodeError:
            pass
    calls = []
    for raw in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            c = json.loads(raw.strip())
            calls.append({"name": c.get("name", ""),
                          "arguments": c.get("arguments", c.get("parameters", {}))})
        except json.JSONDecodeError:
            pass
    return calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf-id", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--grid", default="0,200,450,700")
    ap.add_argument("--add-coef", type=float, default=1245.0)
    ap.add_argument("--n-abl", type=int, default=120)
    ap.add_argument("--n-add", type=int, default=120)
    ap.add_argument("--n-steer-h", type=int, default=100)
    ap.add_argument("--n-steer-b", type=int, default=60)
    ap.add_argument("--max-new", type=int, default=200)
    a = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_grad_enabled(False)

    d = load(a.model, with_prompts=True)
    tok = AutoTokenizer.from_pretrained(a.hf_id, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    net = AutoModelForCausalLM.from_pretrained(
        a.hf_id, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True).eval()
    LAYER = a.layer
    layers = net.model.layers if hasattr(net.model, "layers") else net.model.language_model.layers

    dirs = OUT / f"directions_{a.model}.pt"
    if not dirs.exists():
        sys.exit(f"missing {dirs}; run rerun_against_relabels.py --stage gpu first")
    r_text = torch.load(dirs, map_location="cpu")["r_text"].float()
    r_dev = r_text.to(DEVICE, DTYPE).view(-1)
    log(f"[load] {a.model} layer={LAYER} r_text from {dirs.name}")

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

    def addvec(coef):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            h = h + coef * r_dev
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        hk = Hooks()
        hk.h.append(layers[LAYER - 1].register_forward_hook(hook))
        return hk

    def render(row):
        msgs = [{"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": row["user_prompt"]}]
        tools = None if row["mode"] == "No-tool chat" else list(get_domain(DM[row["domain"]]).tools_for_llm)
        kw = dict(tokenize=False, add_generation_prompt=True)
        for extra in ({"enable_thinking": False}, {}):
            try:
                return tok.apply_chat_template(msgs, tools=tools, **kw, **extra) if tools \
                    else tok.apply_chat_template(msgs, **kw, **extra)
            except TypeError:
                continue

    @torch.no_grad()
    def gen(prompts, hook_fn=None):
        texts = []
        for i in range(0, len(prompts), a.bs):
            chunk = prompts[i:i + a.bs]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(DEVICE)
            Lp = enc["input_ids"].shape[1]
            hk = hook_fn() if hook_fn else None
            with (hk if hk is not None else nullcontext()):
                o = net.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            texts += [tok.decode(o[j, Lp:], skip_special_tokens=False) for j in range(len(chunk))]
        return texts

    def unsafe_all(row, text):
        return len(get_domain(DM[row["domain"]]).score_tool_calls_all(parse_tool_calls(text))) > 0

    nt, tn = d["mode"] == "No-tool chat", d["mode"] == "Tool-enabled normal"
    H, B = d["request_type"] == "Harmful", d["request_type"] == "Benign"
    out = {"model": a.model, "layer": LAYER, "add_coef": a.add_coef}
    pending = []   # completions that need a refusal label

    def stash(exp, coef, frame, texts):
        for (_, row), t in zip(frame.iterrows(), texts):
            pending.append({"exp": exp, "coef": coef, "id": row["id"],
                            "user_prompt": row["user_prompt"], "text": t})

    # ---- ablation: harmful no-tool, refusal should drop -----------------------
    t0 = time.time()
    va = d[nt & H].sample(a.n_abl, random_state=7)
    pa = [render(r) for _, r in va.iterrows()]
    stash("ablation_base", None, va, gen(pa))
    stash("ablation_ablated", None, va, gen(pa, hook_fn=ablate))
    log(f"[abl] n={a.n_abl} generated ({time.time() - t0:.0f}s)")

    # ---- addition: benign no-tool, refusal should rise ------------------------
    t0 = time.time()
    vb = d[nt & B].sample(a.n_add, random_state=7)
    pb = [render(r) for _, r in vb.iterrows()]
    stash("addition_base", None, vb, gen(pb))
    stash("addition_added", a.add_coef, vb, gen(pb, hook_fn=lambda: addvec(a.add_coef)))
    log(f"[add] n={a.n_add} generated ({time.time() - t0:.0f}s)")

    # ---- steering dose-response ----------------------------------------------
    grid = [int(x) for x in a.grid.split(",")]
    sh = d[tn & H].sample(a.n_steer_h, random_state=5)
    sb = d[tn & B].sample(a.n_steer_b, random_state=5)
    shp = [render(r) for _, r in sh.iterrows()]
    sbp = [render(r) for _, r in sb.iterrows()]
    steer = {"grid": grid, "harmful_unsafe": [], "n_harmful": a.n_steer_h}
    for c in grid:
        t0 = time.time()
        hf = (lambda cc=c: addvec(float(cc))) if c > 0 else None
        th = gen(shp, hook_fn=hf)
        uh = [unsafe_all(row, t) for (_, row), t in zip(sh.iterrows(), th)]
        steer["harmful_unsafe"].append(round(float(np.mean(uh)), 3))
        stash("steer_benign", c, sb, gen(sbp, hook_fn=hf))
        log(f"[steer] c={c:4d} unsafe={np.mean(uh):.0%} ({time.time() - t0:.0f}s)")
    out["steering"] = steer

    OUT.mkdir(exist_ok=True)
    (OUT / f"steer_raw_{a.model}.json").write_text(
        json.dumps({"meta": out, "pending": pending}, indent=1))
    log(f"\nwrote {OUT / f'steer_raw_{a.model}.json'}  ({len(pending)} completions to judge)")


if __name__ == "__main__":
    main()
