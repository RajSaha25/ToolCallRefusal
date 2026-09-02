#!/usr/bin/env python3
"""Steering dose-response with the call rate made explicit, on both sides.

"Unsafe -> 0" on harmful prompts is not safer calling if the model simply stops
calling tools, and text-judged over-refusal on benign prompts misses a model that
answers in prose instead of doing the task. This reports, per coefficient:

  harmful:  any-call, unsafe, unsafe-given-call, degenerate (v2)
  benign:   any-call (task completion), over-refusal (judge), degenerate (v2)

Reads relabel_analysis/steer_raw_<model>.json and steer_<model>.json; writes
relabel_analysis/steering_calls.json and prints a markdown table.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from rerun_steering_gen import parse_tool_calls      # noqa: E402
from tools.refusal import strip_tool_markup           # noqa: E402
from degeneracy import is_degenerate_v2               # noqa: E402

OUT = REPO / "relabel_analysis"
ORDER = ["Qwen3-14B", "Mistral-7B-Instruct-v0.3", "c4ai-command-r7b-12-2024",
         "gemma-3-27b-it", "Meta-Llama-3.1-70B-Instruct"]
SHORT = {"Qwen3-14B": "Qwen3-14B", "Mistral-7B-Instruct-v0.3": "Mistral-7B",
         "c4ai-command-r7b-12-2024": "Command-R-7B", "gemma-3-27b-it": "Gemma-3-27B",
         "Meta-Llama-3.1-70B-Instruct": "Llama-3.1-70B"}


def main():
    res = {}
    print("| Model | coef | H any-call | H unsafe | H unsafe given call | H degen | B any-call | B over-refusal | B degen |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in ORDER:
        p = OUT / f"steer_raw_{m}.json"
        if not p.exists():
            continue
        raw = json.loads(p.read_text())
        lab = json.loads((OUT / f"steer_{m}.json").read_text())["steering"]
        hg = raw.get("harmful_gens", [])
        bg = [r for r in raw["pending"] if r["exp"] == "steer_benign"]
        res[m] = {"grid": lab["grid"], "rows": []}
        for i, c in enumerate(lab["grid"]):
            h = [r for r in hg if r["coef"] == c]
            b = [r for r in bg if r["coef"] == c]
            if not h or not b:
                continue
            hc = [bool(parse_tool_calls(r["text"] or "")) for r in h]
            hu = [bool(r["unsafe"]) for r in h]
            n_hc = sum(hc)
            row = {
                "coef": c,
                "h_n": len(h),
                "h_any_call": round(sum(hc) / len(h), 3),
                "h_unsafe": round(sum(hu) / len(h), 3),
                "h_unsafe_given_call": (round(sum(u for u, k in zip(hu, hc) if k) / n_hc, 3) if n_hc else None),
                "h_n_calls": n_hc,
                "h_degen_v2": round(sum(is_degenerate_v2(strip_tool_markup(r["text"] or "")) for r in h) / len(h), 3),
                "b_n": len(b),
                "b_any_call": round(sum(bool(parse_tool_calls(r["text"] or "")) for r in b) / len(b), 3),
                "b_over_refusal": lab["benign_refuse_new"][i],
                "b_degen_v2": round(sum(is_degenerate_v2(strip_tool_markup(r["text"] or "")) for r in b) / len(b), 3),
            }
            res[m]["rows"].append(row)
            ugc = "--" if row["h_unsafe_given_call"] is None else f"{row['h_unsafe_given_call']:.2f} (n={n_hc})"
            print(f"| {SHORT[m]} | {c} | {row['h_any_call']:.2f} | {row['h_unsafe']:.2f} | {ugc} "
                  f"| {row['h_degen_v2']:.0%} | {row['b_any_call']:.2f} | {row['b_over_refusal']:.2f} | {row['b_degen_v2']:.0%} |")
    (OUT / "steering_calls.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'steering_calls.json'}")


if __name__ == "__main__":
    main()
