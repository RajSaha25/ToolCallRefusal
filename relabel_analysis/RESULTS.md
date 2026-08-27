# Mechanism results under the three-way relabels

What changed and what did not, after re-running the label-dependent analyses on top of
`relabel_out/`. Reproduce with `rerun_against_relabels.py`, `rerun_steering_gen.py` and
`judge_generations.py`; per-model JSON sits beside this file.

Two independent corrections are in play and it is worth keeping them apart:

- **the refusal classifier** — the old regex/judge mix replaced by the shared three-way
  REFUSE / CAVEAT / COMPLY judge. Hedged compliance now lands in CAVEAT instead of
  inflating refusal.
- **the tool-safety scorer** — `tc_safe` (scenario-scoped) replaced by `tc_safe_fixed`
  (`score_tool_calls_all`, global scope), which catches unsafe calls the scoped scorer missed.

## Table 1 — reproduced exactly

The CPU stage reproduces `relabel_out/summary_three_way.csv` for all six models
(`table1_all_models.csv`), so the pipeline below is anchored to numbers already agreed on.
Text refusal falls for every model (Qwen 0.857 → 0.753, Mistral 0.703 → 0.435), and benign
over-refusal collapses from ~0.52–0.66 to ~0.00 — the old classifier was calling roughly
half of all benign no-tool responses refusals.

## Direction cosines — the number main.tex:287 cites but never computed

`main.tex:287` describes a behaviour-defined direction (refused vs complied harmful) and its
cosine against the request-type direction. No such code existed; the only cosine in the repo
was `cos(r_text, r_tool) = 0.735` in notebook 02, which is a different quantity. Computed at
each model's own layer, with the new labels:

| model | layer | cos(r_text, r_behav) | caveats folded in | cos(r_text, r_tool) | cos(r_behav old, new) |
|---|---|---|---|---|---|
| Qwen3-14B | 33/40 | 0.859 | 0.857 | 0.741 | 0.975 |
| Mistral-7B-Instruct-v0.3 | 26/32 | 0.711 | 0.675 | 0.773 | 0.978 |
| c4ai-command-r7b-12-2024 | 26/32 | 0.746 | 0.791 | 0.394 | 0.978 |

The claim holds: request harmfulness and the refusal decision occupy substantially the same
axis. The old-vs-new behaviour direction agrees at ~0.98 everywhere, so this result does not
depend on the relabel. Rebuilt `r_text` matches the committed `refusal_dirs.pt` at 0.997.

Caveat: models rarely comply with harmful requests, so the complied side is small — n=42/class
for Qwen, 84 for Mistral, 59 for Command-R. The "caveats folded in" column roughly doubles that
side and moves the answer very little.

## AUC — reproduces on old labels, rises under the corrected scorer

| model | vs unsafe (old) | vs unsafe (fixed) | published | vs refusal (old) | vs refusal (new) |
|---|---|---|---|---|---|
| Qwen3-14B | 0.722 | 0.808 | 0.724 | 0.972 | 0.945 |
| Mistral-7B-Instruct-v0.3 | 0.698 | 0.760 | 0.689 | 0.868 | 0.869 |
| c4ai-command-r7b-12-2024 | 0.713 | 0.768 | 0.751 | 0.923 | 0.901 |

Command-R is the one that drifts from the published value, but only 17/300 of its baseline
rows are unsafe and the CI is 0.611–0.811, which contains 0.751. The projection predicts the
unsafe action *better* once the scorer stops missing unsafe calls.

## Patching — unaffected by definition

Patching scores flips with `score_tool_calls_all`, a predicate over saved tool calls. It never
consults the refusal classifier, and it already used the global-scope scorer. The relabel
cannot move it, so the published patching percentages stand as they are.

## Steering — cost drops but the conclusion survives

Harmful-unsafe is predicate-scored and unchanged; benign over-refusal is re-judged. The grid is
scaled to each model's own projection gap, using the ratios Qwen's published grid implies
(0, 0.64x, 1.45x, 2.25x).

**Qwen3-14B** (gap 311, grid 0/200/450/700):

| coef | harmful unsafe | benign over-refusal (regex) | benign over-refusal (judge) |
|---|---|---|---|
| 0 | 0.510 | 0.000 | 0.000 |
| 200 | 0.370 | 0.017 | 0.000 |
| 450 | 0.260 | 0.217 | 0.150 |
| 700 | 0.030 | 0.550 | 0.467 |

**Llama-3.1-70B** (gap 13.6, grid 0/9/20/31):

| coef | harmful unsafe | benign over-refusal (regex) | benign over-refusal (judge) |
|---|---|---|---|
| 0 | 0.540 | 0.000 | 0.017 |
| 9 | 0.370 | 0.200 | 0.267 |
| 20 | 0.020 | 0.617 | 0.483 |
| 31 | 0.000 | 0.017 | 1.000 |

The qualitative claim holds in both: driving unsafe calls to zero costs roughly half the benign
traffic in Qwen (0.467) and all of it in Llama (1.000). The published "~70 percent" for Qwen is
higher than the 0.467 measured here on the same design.

Note the regex column is not even monotonic for Llama — 0.617 at c=20 then 0.017 at c=31 — while
the judge column rises cleanly. Under heavy steering the output drifts out of the phrasings the
regex keys on, so it stops tracking refusal at exactly the end of the curve that matters.

Gemma's steering is **not reportable**: with no tools in its prompts the harmful-unsafe rate is
0.000 at every coefficient by construction. Mistral was run at c=0 only.

## Ablation — holds for Mistral and Llama, reverses for Qwen and Gemma

Same generations scored two ways, so this isolates the classifier.

| model | regex base → ablated | judge base → ablated | verdict |
|---|---|---|---|
| Qwen3-14B | 0.667 → 0.450 | 0.767 → **0.925** | reverses |
| gemma-3-27b-it | 0.350 → 0.000 | 0.667 → **1.000** | reverses |
| Mistral-7B-Instruct-v0.3 | 0.292 → 0.150 | 0.500 → 0.367 | holds |
| Meta-Llama-3.1-70B-Instruct | 0.725 → 0.617 | 0.733 → 0.642 | holds |

Mistral and Llama keep a real drop under the judge, close in size to the regex-measured one.
Qwen and Gemma reverse outright: ablation *raises* measured refusal, and for Gemma every one of
the 120 ablated responses is judged a refusal while the regex finds none at all.

The cause is visible in the text. 66 of Qwen's 120 ablated responses are refusals the regex
misses, overwhelmingly because ablation shifts the phrasing out of the first-person modal forms
the regex keys on:

> "Certainly not. Submitting a client trade without proper approval is a serious violation of
> compliance and ethical standards."

An unambiguous refusal with no "I cannot" in it. So ablating the direction changed *how* these
models refuse without stopping them from refusing, and the regex read the phrasing shift as a
behavioural drop. The published Qwen (57% → 32%) and Gemma (21% → 0%) ablation figures should
not be relied on. Mistral's and Llama's are fine.

## Addition — strengthens everywhere, and rescues Llama

| model | base (regex → judge) | added (regex → judge) |
|---|---|---|
| Qwen3-14B | 0.017 → 0.000 | 0.750 → 1.000 |
| Mistral-7B-Instruct-v0.3 | 0.000 → 0.000 | 0.692 → 1.000 |
| gemma-3-27b-it | 0.008 → 0.000 | 0.208 → 1.000 |
| Meta-Llama-3.1-70B-Instruct | 0.008 → 0.000 | 0.000 → 1.000 |

Adding the direction drives benign refusal to 100% in every model, because the judge counts the
hedged refusals the regex dropped. The Llama row matters most: the paper reports its addition as
null (1% → 1%) and treats Llama as the family where the direction does not control refusal. On
the same intervention, scored properly, it goes 0% → 100%. Gemma moves the same way (21% → 100%).
The "addition is weak in Gemma and null in Llama" reading is a classifier artifact.

## A note on the denominator

An earlier pass dropped responses with no prose (a bare tool call and nothing else) instead of
counting them. `tools.refusal.classify_refusal` treats those as *not refused* and keeps them in
the denominator, which is right: a model that quietly calls a tool has not refused. Dropping
them left only the talkative minority and inflated every over-refusal rate — Llama's read as
100% at zero steering. The numbers above count them, matching the shared classifier. This moved
Qwen's top-coefficient over-refusal from 0.609 to 0.467.

## Not covered

- **Gemma tool-mode** — steering harmful-unsafe, `cos(r_text, r_tool)`, AUC vs unsafe. Blocked on
  the chat-template problem above, not on compute.
- **Command-R generation half** — ablation, addition and steering were never run for it. Its
  template does render tools, so this is only a matter of GPU time.
- **Mistral steering** — run at c=0 only; the full grid was not swept.
- **Patching** — deliberately not rerun, see above.
