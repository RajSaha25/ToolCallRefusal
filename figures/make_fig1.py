"""Figure 1: text compliance vs tool-call compliance on the same harmful requests.

Both bars measure the same thing — the model DID the harmful thing — so they
share polarity (higher = worse), which is what reviewer RGib asked for.

  blue   = text compliance in no-tool chat (three-way judge label 'complied';
           strict — 'caveat' hedged compliance is NOT counted)
  red    = unsafe tool call in tool-enabled normal mode (any predicate fires,
           incl. fabricated-authorization values from relabel_out/fabricated_auth_audit.csv);
           faint extension = called a tool with no violation, so the full red bar
           height is the any-call rate and the dark share of it is unsafe|call.
           This keeps text/tool denominators matched (all harmful prompts) while
           showing tool-call propensity in the geometry itself.

Inputs (all regenerable):
  results/results_<model>.csv          — runs, tool_calls
  relabel_out/summary_three_way.csv    — per-model text label rates (new classifier)
  relabel_out/fabricated_auth_audit.csv — extra unsafe calls (fabricated auth)

Usage: python figures/make_fig1.py [plain|stacked]
  plain   -> figures/fig1_text_vs_tool_compliance.png          (two bars, matched denominators)
  stacked -> figures/fig1_text_vs_tool_compliance_stacked.png  (red bar split: unsafe + called-no-violation)
"""
import sys
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.metrics import add_call_columns, apply_fabricated_auth_overlay

MODELS = [  # (results model id, display name)
    ("Qwen3-14B", "Qwen3-14B"),
    ("Mistral-7B-Instruct-v0.3", "Mistral-7B"),
    ("Meta-Llama-3.1-70B-Instruct", "Llama-3.1-70B"),
    ("gemma-3-27b-it", "Gemma-3-27B"),
    ("c4ai-command-r7b-12-2024", "Command-R-7B"),
]
TOOL_MODE = "Tool-enabled normal"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "plain"   # "plain" | "stacked"


def load_runs():
    files = [f for f in glob(str(REPO / "results" / "results_*.csv")) if "cell_divergence" not in f]
    df = add_call_columns(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
    return apply_fabricated_auth_overlay(df)


def main():
    df = load_runs()
    text = pd.read_csv(REPO / "relabel_out" / "summary_three_way.csv").set_index("model")
    harm = df[df["request_type"] == "Harmful"]

    comply, caveat, unsafe, unsafe_given_call, any_call = [], [], [], [], []
    for mid, _ in MODELS:
        t = text.loc[mid]
        comply.append(100 * t["com_h"])
        caveat.append(100 * t["cav_h"])
        tn = harm[(harm["model"] == mid) & (harm["mode"] == TOOL_MODE)]
        unsafe.append(100 * tn["unsafe"].mean())
        any_call.append(100 * tn["any_call"].mean())
        unsafe_given_call.append(100 * tn.loc[tn["any_call"], "unsafe"].mean())

    x = np.arange(len(MODELS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(x - w / 2, comply, w, color="#e08214", label="Text: complied with harmful request (no-tool chat)")
    ax.bar(x + w / 2, unsafe, w, color="#d7301f", label="Tool: unsafe tool call (tool-enabled normal)")
    if VARIANT == "stacked":
        called_ok = [a - u for a, u in zip(any_call, unsafe)]
        ax.bar(x + w / 2, called_ok, w, bottom=unsafe, color="#d7301f", alpha=0.22, edgecolor="#d7301f",
               linewidth=0.8, label="Tool: called, no violation")

    for i in range(len(MODELS)):
        ax.text(x[i] - w / 2, comply[i] + 1, f"{comply[i]:.0f}", ha="center", va="bottom", fontsize=10)
        if VARIANT == "stacked":
            ax.text(x[i] + w / 2, max(unsafe[i] - 4, 2), f"{unsafe[i]:.0f}", ha="center", va="bottom",
                    fontsize=10, color="white", fontweight="bold")
            ax.text(x[i] + w / 2, any_call[i] + 1, f"{unsafe_given_call[i]:.0f}% unsafe | call",
                    ha="center", va="bottom", fontsize=8, color="#a02010")
        else:
            ax.text(x[i] + w / 2, unsafe[i] + 1, f"{unsafe[i]:.0f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([d for _, d in MODELS])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Harmful requests complied with (%)  ↑ worse")
    ax.set_title("Unsafe behavior in text vs. in tool calls on the same harmful requests")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = REPO / "figures" / ("fig1_text_vs_tool_compliance.png" if VARIANT == "plain"
                              else "fig1_text_vs_tool_compliance_stacked.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    for (mid, d), c, cv, u, a, ug in zip(MODELS, comply, caveat, unsafe, any_call, unsafe_given_call):
        print(f"{d:14} text comply {c:4.1f} (+{cv:4.1f} caveat) | tool unsafe {u:4.1f} | any-call {a:4.1f} | unsafe|call {ug:4.1f}")


if __name__ == "__main__":
    main()
