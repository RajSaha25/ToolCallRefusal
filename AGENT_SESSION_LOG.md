# Coding-Agent Session Log — ToolCallRefusal

**Date:** 2026-06-06
**Environment:** remote A100-80GB VM (RunPod), Linux, headless background agent
**Model under study:** Qwen3-14B (behavioral eval in 4-bit; mechanistic interp in bf16)
**Agent:** Claude Code

> This is a reconstructed, structured summary of the session (not a verbatim keystroke transcript).
> It documents what was asked, what was done, the decisions and dead-ends, and the artifacts produced.
> Nulls, bugs, and honest caveats are included deliberately — they reflect the real research process.

---

## 1. Session overview

Starting from a cloned private research repo (`RajSaha25/ToolCallRefusal`), the session:
1. Pulled the latest code (with some GitHub-auth troubleshooting).
2. Stood up and ran a **behavioral safety eval** of Qwen3-14B (refusal in chat vs unsafe tool calls).
3. Designed and executed a full **mechanistic interpretability** study of *why* the model refuses in
   text but acts via tools — extracting a refusal direction, validating it causally, localizing the
   circuit, and testing interventions — across five iterative experiment rounds.
4. Found and fixed a **scoring bug** that was deflating the headline metric ~2–3×.
5. Scaled the key experiments with **batched generation + bootstrap 95% CIs**.
6. Wrote full documentation: a beginner's guide, a results doc, and a consolidated runnable notebook.

**One-line scientific result:** refusal is governed by a single linear direction; tool context
*suppresses* it; the direction strongly *predicts* the unsafe tool call (AUC 0.73) but only *partially
controls* it — a strong-predictor / partial-mediator dissociation.

---

## 2. Chronological work log

### Phase 1 — Repo access & setup
- Cloned repo was private; an initial **fine-grained PAT lacked repo scope** (404 / "metadata only").
  Diagnosed via the GitHub API, switched to a correctly-scoped token, pulled `main`.
- Surfaced exposed secrets committed in a notebook (HF token, Anthropic key) and flagged rotation.

### Phase 2 — Behavioral eval (Qwen3-14B)
- The eval was a **Google Colab** notebook (Drive mount, `userdata`, GPU assumed). Adapted it to run
  **headless on the VM**: rewrote the Colab cells into a local runner (`run_qwen_eval.py`), pointed
  paths at local storage, set `HF_HOME` to the large disk.
- Pulled the 2,304-row dataset **from the user's Google Drive via the Drive MCP connector** (used a
  subagent so the 200 KB base64 didn't flood context).
- Hit and fixed two dependency issues: `hf_transfer` missing, and **`transformers==4.47` predates
  Qwen3** (upgraded to 5.10 so the architecture is recognized).
- Smoke-tested (8 rows, clean), then launched the full 2,304-row run in the background, checkpointed
  every 25 rows, resumable.

### Phase 3 — Mechanistic interpretability (5 rounds)
**Round 1 (`run_interp.py`):** Extracted the refusal direction via **difference-in-means** (128 harmful
vs 128 harmless, No-tool), at layer 33/40. Validated causally — **ablation** dropped harmful refusal,
**activation addition** induced refusal on benign prompts. Measured projection onto the direction by
mode → suppression under tools (paired test significant).

**Round 2 (`run_interp2.py`):** System-condition interaction (safety-prompt lifts the signal;
tool-encouraging+stress is the floor); **component attribution** localizing the suppression to late
attention heads (L34–39) and MLPs (L29–32); patching and steering — **both came back null**.

**Diagnosis of the nulls (`run_interp3.py`):** the steering null was a **self-inflicted bug** — I swept
coefficients 50–200× too small for the direction's scale; corrected, steering then drove unsafe calls to
0%. The patching null was a **floor effect** — almost no unsafe calls in the sample to remove.

**Tier 0 — the scorer bug:** investigated why an obvious PHI-bypass tool call scored "safe." Root cause:
the scorer only checked each scenario's *narrow* forbidden-action list, missing cross-scenario
violations. Added `score_tool_calls_all` (`tools/core.py`); re-scoring raised the harmful unsafe rate
~2–3×.

**Tier 1 (`run_interp4.py`):** the validity check — re-extracted the direction *within* tool mode and
found **cosine 0.735** with the text direction (same feature, suppressed). AUC of projection→unsafe =
0.70. Patching redo on real divergence cases: only ~19% sufficient.

**Scaling (`run_interp5.py`):** re-ran the generation experiments **batched** at n=120–300 with
**bootstrap 95% CIs**. This tempered the small-n teasers (ablation 55→20% became 56→36%) and confirmed
the dissociation.

### Phase 4 — Documentation
- `MECH_INTERP_GUIDE.md` — a from-scratch explanation of the methodology for a non-expert, with code
  references and figure links.
- `MECHANISTIC_INTERP.md` — terse results doc with all numbers and CIs.
- `Tool_Refusal_Mechanistic_Interp.ipynb` — a **consolidated, deduplicated** notebook that shows every
  cached result + figure instantly and re-runs any experiment behind a flag.

### Phase 5 — Honest finding from the overnight data
Re-scoring the in-progress behavioral run surfaced a **configuration mismatch**: behavioral (4-bit,
thinking-mode ON) showed ~9.5% unsafe while interp (bf16, thinking OFF) showed ~31%. Flagged that the
two halves of the paper measure slightly different model configs and should be reconciled.

---

## 3. Key results (Qwen3-14B)

| Result | Value (95% CI where applicable) | Verdict |
|---|---|---|
| Refusal direction is causal | ablation 56→36% (n=120); addition 1→71% | real (modest ablation) |
| Tool context suppresses it | paired Δ +66.5, t=5.65 | strong |
| Same feature under tools | cosine 0.735 | validated |
| Suppression circuit | attn heads L34–39, MLPs L29–32 | localized |
| Projection predicts unsafe call | **AUC 0.73 [0.67–0.78]** | strong predictor |
| Patching restores safety | 19% [12–28%] | partial |
| Steering restores safety | unsafe→0% but 62% benign over-refusal | works, blunt |
| Scorer fix on behavioral data | unsafe 5.3%→9.5% (n=432) | important correction |

**Interpretation:** a prediction/intervention dissociation — the refusal direction is a strong predictor
and partial mediator of the unsafe action, not the sole causal switch; tool-refusal failure is driven
jointly by refusal-suppression *and* the tool affordance.

---

## 4. Artifacts produced

```
run_qwen_eval.py              # headless behavioral eval (Qwen3-14B)
run_interp.py                 # round 1: extract + validate + projection
run_interp2.py                # round 2: syscond, heads, patching, steering
run_interp3.py                # fixups: corrected steering scale + base-rate diagnostic
run_interp4.py                # tier-1: same-direction, patching redo, AUC
run_interp5.py                # scaled + batched + bootstrap CIs
rescore_results.py            # offline re-score with the fixed scorer
tools/core.py                 # + score_tool_calls_all (the fix)
Tool_Refusal_Mechanistic_Interp.ipynb   # consolidated runnable notebook
Mechanistic_Refusal_Direction.ipynb     # round-1 interactive notebook
MECH_INTERP_GUIDE.md          # beginner-friendly methodology guide
MECHANISTIC_INTERP.md         # terse results doc
interp_artifacts/             # refusal_dirs.pt, interp*_summary.json, fig_*.png (8 figures)
```

---

## 5. Notable engineering / problem-solving moments

- **GitHub auth:** diagnosed a fine-grained-PAT scope failure via the API before switching tokens.
- **Colab → headless port:** rewrote a Drive-dependent notebook into a local background runner.
- **Dependency archaeology:** identified that the pinned `transformers` predated the Qwen3 architecture.
- **Self-caught bug:** the steering null was my own coefficient-scaling error; diagnosed and corrected
  rather than reported as a real negative result.
- **Data-integrity bug:** found the scenario-scoped scorer was under-counting unsafe tool calls and fixed
  it (non-breaking), then quantified the impact.
- **Throughput:** moved from one-prompt-at-a-time to **batched generation** to make n=300 + CIs feasible
  on a shared GPU.
- **Statistical rigor:** added bootstrap 95% CIs, which tempered flashy small-n estimates.
- **Honesty:** reported null results, a config mismatch, and a partial-causation story rather than
  overclaiming a clean single-direction result.

---

## 6. Open questions / next steps

1. Reconcile the behavioral (4-bit, think-on) and interp (bf16, think-off) configs.
2. Stronger ablation (attention+MLP inputs, multiple directions) to test whether the modest causal effect
   is real or an artifact of crude intervention.
3. Affordance-isolation experiment (tools present, tool-format blocked) to complete the two-factor story.
4. Length/format confound control.
5. Cross-model replication (Qwen3-8B/4B, Llama-3.1, Mistral, Gemma).

---

*Generated by the coding agent at the end of the session as a record of the work.*
