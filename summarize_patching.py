#!/usr/bin/env python3
"""Collect relabel_analysis/patching_<model>.json into one table.

The flip rate alone cannot distinguish "the patched model called the tool with safe
arguments" from "the patched model stopped calling tools". The two columns that
split it are the ones that matter for the mediator claim; unsafe-given-call after
patching says whether the direction carries any action-safety information at all.

  python summarize_patching.py            # markdown to stdout
  python summarize_patching.py --latex    # booktabs rows
"""
import argparse
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "relabel_analysis"
ORDER = ["Qwen3-14B", "Mistral-7B-Instruct-v0.3", "c4ai-command-r7b-12-2024",
         "gemma-3-27b-it", "Meta-Llama-3.1-70B-Instruct"]
SHORT = {"Qwen3-14B": "Qwen3-14B", "Mistral-7B-Instruct-v0.3": "Mistral-7B",
         "c4ai-command-r7b-12-2024": "Command-R-7B", "gemma-3-27b-it": "Gemma-3-27B",
         "Meta-Llama-3.1-70B-Instruct": "Llama-3.1-70B"}
# the draft's Table 2 patching row, same model order
PUBLISHED = {"Qwen3-14B": 0.18, "Mistral-7B-Instruct-v0.3": 0.24, "c4ai-command-r7b-12-2024": 0.62,
             "gemma-3-27b-it": 0.46, "Meta-Llama-3.1-70B-Instruct": 0.40}


def pct(x):
    return "--" if x is None else f"{100 * x:.0f}%"


def ci(c):
    return "" if not c else f" [{100 * c[0]:.0f}, {100 * c[1]:.0f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    rows = []
    for m in ORDER:
        p = OUT / f"patching_{m}.json"
        if not p.exists():
            rows.append((m, None))
            continue
        rows.append((m, json.loads(p.read_text())))

    if not a.latex:
        print("| Model | Baseline unsafe | Published flip | Rerun flip [95% CI] | via safe call | via no call "
              "| unsafe given call, after | Patched degenerate | Note |")
        print("|---|---|---|---|---|---|---|---|---|")
        for m, r in rows:
            if r is None:
                print(f"| {SHORT[m]} | -- | {pct(PUBLISHED[m])} | not run | | | | | |")
                continue
            note = []
            if r.get("quantized_4bit"):
                note.append("4-bit NF4 weights")
            if r["verdict"] != "OK":
                note.append(f"{r['verdict'].lower()}: clean-only flip {pct(r['flip_rate_clean'])}")
            if not r.get("tools_native", True):
                note.append("tools injected")
            print(f"| {SHORT[m]} | {r['n_unsafe_baseline']}/{r['n_baseline']} ({pct(r['baseline_unsafe_rate'])}) "
                  f"| {pct(PUBLISHED[m])} | {pct(r['flip_rate_all'])}{ci(r['flip_rate_all_ci'])} "
                  f"| {pct(r['flip_via_safe_call'])} | {pct(r['flip_via_no_call'])} "
                  f"| {pct(r['unsafe_given_call_patched'])} | {pct(r['patched_degenerate'])} | {'; '.join(note)} |")
        return

    print(r"\begin{tabular}{lrrrrrr}")
    print(r"\toprule")
    print(r"Model & Unsafe base & Flip (draft) & Flip (rerun) & via safe call & via no call & Unsafe$\mid$call after \\")
    print(r"\midrule")
    for m, r in rows:
        if r is None:
            print(f"{SHORT[m]} & -- & {100 * PUBLISHED[m]:.0f} & -- & -- & -- & -- \\\\")
            continue
        star = r"$^\dagger$" if r.get("quantized_4bit") else ""
        print(f"{SHORT[m]}{star} & {100 * r['baseline_unsafe_rate']:.0f} & {100 * PUBLISHED[m]:.0f} "
              f"& {100 * r['flip_rate_all']:.0f} [{100 * r['flip_rate_all_ci'][0]:.0f}, {100 * r['flip_rate_all_ci'][1]:.0f}] "
              f"& {100 * r['flip_via_safe_call']:.0f} & {100 * r['flip_via_no_call']:.0f} "
              f"& {100 * (r['unsafe_given_call_patched'] or 0):.0f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
