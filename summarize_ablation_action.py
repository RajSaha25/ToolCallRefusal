#!/usr/bin/env python3
"""Does ablation make the model act *less*, or act *less safely*?

The headline unsafe rate conflates the two. A model that stops calling tools
altogether shows a lower unsafe rate without having become any safer in the sense
that matters. The conditional rate — unsafe given that a tool was called — is the
one that isolates safety from willingness to act.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "relabel_analysis"
ORDER = ["Qwen3-14B", "Mistral-7B-Instruct-v0.3", "c4ai-command-r7b-12-2024",
         "gemma-3-27b-it", "Meta-Llama-3.1-70B-Instruct"]
SHORT = {"Qwen3-14B": "Qwen3-14B", "Mistral-7B-Instruct-v0.3": "Mistral-7B",
         "c4ai-command-r7b-12-2024": "Command-R-7B", "gemma-3-27b-it": "Gemma-3-27B",
         "Meta-Llama-3.1-70B-Instruct": "Llama-3.1-70B"}


def boot_ci(v, n=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    bs = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(n)]
    return (round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3))


print("=" * 100)
print("ABLATION: effect on the ACTION (predicate-scored, no classifier involved)")
print("=" * 100)
print(f"{'model':<15}{'unsafe base':>12}{'unsafe abl':>12}{'delta':>9}{'sig':>5}"
      f"{'called base':>13}{'called abl':>12}{'unsafe|called base':>20}{'unsafe|called abl':>19}")

rows = []
for m in ORDER:
    p = OUT / f"ablation_action_{m}.json"
    raw_p = OUT / f"ablation_action_raw_{m}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    lo, hi = d["delta_ci_paired"]
    sig = "yes" if not (lo <= 0 <= hi) else "no"

    cb = ca = float("nan")
    if raw_p.exists():
        raw = json.loads(raw_p.read_text())
        # conditional rate needs the per-row "did it call a tool" flag, recovered
        # by re-parsing; base_unsafe implies a call was made
        import sys
        sys.path.insert(0, str(OUT.parent))
        from rerun_steering_gen import parse_tool_calls
        bc = [(bool(parse_tool_calls(r["base_text"] or "")), r["base_unsafe"]) for r in raw]
        ac = [(bool(parse_tool_calls(r["abl_text"] or "")), r["abl_unsafe"]) for r in raw]
        b_called = [u for c, u in bc if c]
        a_called = [u for c, u in ac if c]
        cb = float(np.mean(b_called)) if b_called else float("nan")
        ca = float(np.mean(a_called)) if a_called else float("nan")
        d["unsafe_given_called_base"] = None if not b_called else round(cb, 3)
        d["unsafe_given_called_abl"] = None if not a_called else round(ca, 3)
        d["n_called_base"], d["n_called_abl"] = len(b_called), len(a_called)
        p.write_text(json.dumps(d, indent=2))

    print(f"{SHORT[m]:<15}{d['unsafe_baseline']:>12.3f}{d['unsafe_ablated']:>12.3f}"
          f"{d['delta_unsafe']:>+9.3f}{sig:>5}"
          f"{d['called_tool_baseline']:>13.3f}{d['called_tool_ablated']:>12.3f}"
          f"{cb:>20.3f}{ca:>19.3f}")
    rows.append(d)

print()
print("Reading: 'called' is the share of prompts where the model emitted any tool call.")
print("'unsafe|called' is the unsafe rate among those. A model whose 'called' rate")
print("collapses under ablation has stopped acting, not become safer -- its headline")
print("unsafe rate falls for the wrong reason.")
