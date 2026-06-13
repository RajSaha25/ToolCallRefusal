# ToolCallRefusal

Why does a safety-tuned model refuse a harmful request in plain chat, but carry the same request out
through a tool call? This repo studies that gap two ways:

- a **behavioral eval** that measures refusal transfer (text refusal vs. unsafe tool call) across
  domains, modes, and system conditions, and
- a **mechanistic interpretability** study of *why* it happens inside the network — refusal is carried
  by a single linear direction in the residual stream, and tool context turns that direction down.

The headline result: refusal is governed by one linear direction; tool context suppresses it; the
direction strongly *predicts* the unsafe tool call (AUC ≈ 0.73 on Qwen3-14B) but only *partially
controls* it. It is a strong predictor and a partial mediator, not a single causal switch.

---

## Run the mechanistic interpretability workflow on your model

The two numbered notebooks are the main deliverable. They are written to run on **any** Hugging Face
causal LM whose chat template supports tool definitions — you set `MODEL_ID` and everything else adapts
(the layer is chosen from the data, and the steering/addition strengths scale to the model's own
activation magnitudes).

1. **`01_refusal_direction_and_suppression.ipynb`** — find the refusal direction, prove it is causal
   (ablation removes refusal, addition induces it), and measure the core result: the refusal signal is
   weaker when tools are in the prompt. Forward-pass heavy; a few minutes plus short generation checks.
   Saves the direction to `interp_artifacts/<model>/refusal_dirs.pt`.

2. **`02_causal_followups_and_scaling.ipynb`** — load that saved direction and run the harder questions
   with bootstrap confidence intervals: same-direction-under-tools (cosine), does the projection predict
   the unsafe action (AUC), patching, steering dose-response, and scaled ablation/addition.
   Generation heavy; budget 40+ minutes on an A100 for a 14B model.

Open `01` first, set `MODEL_ID` in the config cell, and run top to bottom. Then do the same in `02` with
the same `MODEL_ID`. Each model writes to its own `interp_artifacts/<model>/` folder, so runs don't
collide.

New to the methods? Read **`MECH_INTERP_GUIDE.md`** first — it explains residual streams, directions,
projections, ablation, and patching from scratch, then ties each idea to the exact cells here.
**`MECHANISTIC_INTERP.md`** is the terse results write-up with the Qwen3-14B numbers.

---

## Environment

Tested on an A100-80GB (RunPod), Python 3.12, with `torch 2.8.0+cu128` from the base image.

```bash
# torch is assumed preinstalled (CUDA build matched to your GPU). Install the rest:
python3.12 -m pip install --break-system-packages -r requirements.txt
```

Model weights are pulled from the Hugging Face hub on first use. Point the cache at a persistent disk so
you don't redownload after a restart:

```bash
export HF_HOME=/workspace/.cache/huggingface
```

A 14B model in bf16 needs roughly 30 GB of GPU memory. For smaller cards, lower the `N_*` sample-size
knobs in each notebook's config cell, or use a smaller model.

---

## Reproducing the Qwen3-14B numbers headless

The notebooks are the canonical, model-agnostic version. Two headless scripts reproduce the
`Qwen/Qwen3-14B` numbers without a notebook server (run the first before the second — it saves the
direction the second reuses):

```bash
python3.12 run_direction_and_suppression.py   # extract direction -> ablation/addition -> projection by mode
python3.12 run_scaled_evaluation.py           # batched re-run with bootstrap 95% CIs (authoritative numbers)
python3.12 rescore_results.py results/results_Qwen3-14B.csv   # re-score a behavioral CSV, no model needed
```

The earlier iterative rounds (the round-2 experiments, the steering-scale fix, and the tier-1
follow-ups) are kept in `legacy/` for provenance — they are Qwen-specific and superseded by the
notebooks. The behavioral eval lives in `run_qwen_eval.py` (and `Behavioral_eval.ipynb`).

---

## Layout

```
01_refusal_direction_and_suppression.ipynb   mech-interp part 1 (run this first)
02_causal_followups_and_scaling.ipynb         mech-interp part 2 (loads part 1's direction)
MECH_INTERP_GUIDE.md                          teaching guide (start here if the methods are new)
MECHANISTIC_INTERP.md                         terse results write-up
requirements.txt                              Python deps (torch assumed preinstalled)

run_direction_and_suppression.py              headless: extract + validate direction, projection by mode
run_scaled_evaluation.py                      headless: scaled re-run with bootstrap 95% CIs
measure_divergence.py, rescore_results.py     behavioral-metric helpers
run_qwen_eval.py                              behavioral eval

tools/        scenario + scoring package (domains, forbidden actions, tool definitions)
data/         the 2,304-row safety eval dataset
results/      behavioral eval outputs (CSV)
interp_artifacts/   figures, summary JSON, saved directions (per-model subfolders)
legacy/       superseded notebooks + the earlier iterative round-2 / tier-1 scripts (provenance)
```

---

## How the scoring works (and one bug worth knowing about)

A model tool call is scored against per-domain **forbidden actions** in `tools/`. The original
`DomainSpec.score_tool_calls(scenario_id, calls)` only checked the actions listed for that one scenario,
so a genuinely unsafe call that belonged to a *different* scenario's list was scored safe. The fix,
`DomainSpec.score_tool_calls_all(calls)`, checks every forbidden action in the domain. Re-scoring the
first 250 behavioral rows raised the harmful unsafe rate from 4.2% to 11.5%. The mech-interp notebooks
use the all-actions scorer.

---

## Honest limitations

- Results so far are on one model (Qwen3-14B). Cross-model replication is the point of making these
  notebooks model-agnostic — that is the open question.
- The refusal classifier used inside the generation experiments is a regex (fast, no API). It undercounts
  soft refusals, so read the *direction* of effects as solid and exact rates as approximate.
- Steering reduces unsafe calls but at a steep over-refusal cost; it is not a deployable defense yet.
- A length/format confound control (does irrelevant long context lower the projection?) is still owed.

*Reference:* Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda (2024), *Refusal in Language Models
Is Mediated by a Single Direction*, arXiv:2406.11717.
