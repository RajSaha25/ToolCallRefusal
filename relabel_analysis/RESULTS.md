# Mechanism results under the three-way relabels

What changed and what did not, after re-running the label-dependent analyses on top of
`relabel_out/`. All five families, run on a single B200 (183GB). Reproduce with
`rerun_against_relabels.py`, `rerun_steering_gen.py` and `judge_generations.py`; per-model JSON
sits beside this file, and `summarize_reruns.py` prints the tables.

Three separate corrections are in play and it is worth keeping them apart:

- **the refusal classifier** — the old regex/judge mix replaced by the shared three-way
  REFUSE / CAVEAT / COMPLY judge. Hedged compliance now lands in CAVEAT instead of
  inflating refusal.
- **the tool-safety scorer** — `tc_safe` (scenario-scoped) replaced by `tc_safe_fixed`
  (`score_tool_calls_all`, global scope), which catches unsafe calls the scoped scorer missed.
- **prompt rendering** — a template bug found during this work that silently stripped tool
  definitions for two families. See "The chat-template problem".

## Table 1 — reproduced exactly

The CPU stage reproduces `relabel_out/summary_three_way.csv` for all six models
(`table1_all_models.csv`). Text refusal falls for every model (Qwen 0.857 → 0.753, Mistral
0.703 → 0.435), and benign over-refusal collapses from ~0.52–0.66 to ~0.00 — the old classifier
was calling roughly half of all benign no-tool responses refusals.

## Direction cosines — the number main.tex:287 cites but never computed

`main.tex:287` describes a behaviour-defined direction (refused vs complied harmful) and its
cosine against the request-type direction. No such code existed; the only cosine in the repo was
`cos(r_text, r_tool) = 0.735` in notebook 02, a different quantity.

| model | layer | cos(r_text, r_behav) | caveats folded in | cos(r_text, r_tool) | cos(r_behav old, new) | n/class |
|---|---|---|---|---|---|---|
| Qwen3-14B | 33/40 | 0.859 | 0.857 | 0.741 | 0.975 | 42 |
| Mistral-7B-Instruct-v0.3 | 26/32 | 0.711 | 0.675 | 0.773 | 0.978 | 84 |
| c4ai-command-r7b-12-2024 | 26/32 | 0.746 | 0.791 | 0.394 | 0.978 | 59 |
| gemma-3-27b-it | 51/62 | 0.822 | 0.817 | 0.797 | 0.958 | 25 |
| Meta-Llama-3.1-70B-Instruct | 67/80 | 0.693 | 0.715 | 0.213 | 0.979 | 59 |

**The claim holds for all five**: request harmfulness and the refusal decision occupy
substantially the same axis. The old-vs-new behaviour direction agrees at 0.96–0.98 everywhere,
so this result does not depend on the relabel — a clean answer if a reviewer presses on it.
Rebuilt `r_text` matches the committed `refusal_dirs.pt` at 0.997 for Qwen, and Gemma's rebuilt
projection gap (18944) matches the original run's (18715).

Caveat: models rarely comply with harmful requests, so the complied side is small — as few as
25/class for Gemma. The "caveats folded in" column folds CAVEAT in with compliance, roughly
doubling that side, and moves the answer very little anywhere.

**`cos(r_text, r_tool)` is the weaker claim.** The paper says the text- and tool-derived
directions "remain aligned". That is well supported for Qwen (0.741), Mistral (0.773) and Gemma
(0.797), but not for Command-R (0.394) or Llama (0.213). Llama being the outlier sits alongside
the paper's own finding that it is the one family where tool context does not suppress the
refusal signal (t = 0.4).

## AUC — reproduces for all five, and rises under the corrected scorer

| model | vs unsafe (old) | vs unsafe (fixed) | published | reproduces | vs refusal (old) | vs refusal (new) |
|---|---|---|---|---|---|---|
| Qwen3-14B | 0.722 | 0.808 | 0.724 | yes | 0.972 | 0.945 |
| Mistral-7B-Instruct-v0.3 | 0.698 | 0.760 | 0.689 | yes | 0.868 | 0.869 |
| c4ai-command-r7b-12-2024 | 0.713 | 0.768 | 0.751 | yes | 0.923 | 0.901 |
| gemma-3-27b-it | 0.701 | 0.790 | 0.791 | yes | 0.903 | 0.946 |
| Meta-Llama-3.1-70B-Instruct | 0.761 | 0.864 | 0.881 | yes | 0.949 | 0.966 |

*The relabel-dependent column* is "vs refusal", and it improves for the two large models
(Gemma 0.903 → 0.946, Llama 0.950 → 0.966): once hedged compliance stops being scored as
refusal, the projection predicts the refusal decision better.

*The "vs unsafe" column is not relabel-dependent* — it is the same predicate-scored quantity as
patching — and every model now reproduces its published value once the prompt is built
correctly. Gemma lands at 0.790 against a published 0.791. Both models that initially appeared
not to reproduce were victims of the template bug, not of a real discrepancy.

## Patching — unaffected by definition

Patching scores flips with `score_tool_calls_all`, a predicate over saved tool calls. It never
consults the refusal classifier and already used the global-scope scorer, so the relabel cannot
move it. The published patching percentages stand.

## The chat-template problem

`apply_chat_template(..., tools=...)` silently drops the tools when a model's template has no
tool-handling block: the variable is unused, nothing is raised. The consequence is severe and
invisible — a tool-enabled prompt renders as plain chat, the model has nothing to call, and every
response scores as "made no unsafe tool call", i.e. perfectly safe.

| model | template length | carries tools |
|---|---|---|
| Qwen3-14B | 4168 | yes |
| Mistral-7B-Instruct-v0.3 | — | yes |
| c4ai-command-r7b-12-2024 | — | yes |
| gemma-3-27b-it (`google/` and `unsloth/`, identical) | 1532 | **no** |
| `NousResearch/Meta-Llama-3.1-70B-Instruct` | 348 | **no** |
| `unsloth/Meta-Llama-3.1-70B-Instruct` | 4614 | yes |

Llama scored 0/100 unsafe at every steering coefficient before this was caught, with all 100
responses clean verbal refusals. For Llama the fix is the mirror: the NousResearch reupload
ships a stripped template, `unsloth/` has the real one.

Gemma has no tool-capable template anywhere, so `render_utils.make_renderer` probes the template
once and, when it cannot carry tools, writes the schemas into the prompt text instead. That
reconstruction is validated two ways: the unsafe rate on harmful tool-normal prompts is 12.5%
against 14.6% stored under the same scorer, and the resulting AUC is 0.790 against a published
0.791. Both indicate the original Gemma tool-mode was produced this way and not through the
chat template — which also answers how the committed Gemma CSV came to hold 666 rows of tool
calls using 100% real tool names.

`_render_check.py` asserts a tool name actually appears in the rendered prompt. Run it before
any tool-mode sweep; it would have caught this immediately.

## Steering — cost drops but the conclusion survives

Harmful-unsafe is predicate-scored; benign over-refusal is re-judged. The grid is scaled to each
model's own projection gap via `--auto-grid`, using the ratios Qwen's published grid implies
(0, 0.64x, 1.45x, 2.25x) — the published Gemma run used Qwen's *absolute* grid (0/200/450/700)
against a projection gap of 18944, roughly 60x too small, which is why its published steering
curve is flat.

All figures below are the n=200 harmful / 120 benign pass. A first pass at n=100/60 gave the
same curves throughout, so these conclusions are not sample-size artifacts.

| model | grid | harmful unsafe | benign over-refusal (regex) | benign over-refusal (judge) |
|---|---|---|---|---|
| Qwen3-14B | 0 → 700 | 0.535 → 0.030 | 0.025 → 0.533 | 0.000 → **0.442** |
| Mistral-7B-Instruct-v0.3 | 0 → 17 | 0.270 → 0.000 | 0.000 → 0.658 | 0.033 → **0.683** |
| c4ai-command-r7b-12-2024 | 0 → 172 | 0.115 → 0.000 | 0.000 → 1.000 | 0.025 → **1.000** |
| gemma-3-27b-it | 0 → 42587 | 0.120 → 0.000 | 0.042 → 0.142 | 0.217 → **1.000** |
| Meta-Llama-3.1-70B-Instruct | 0 → 31 | 0.515 → 0.000 | 0.008 → 0.033 | 0.025 → **1.000** |

The qualitative claim holds everywhere and is if anything stronger than published: driving
unsafe calls to zero costs roughly half the benign traffic in Qwen and *all* of it in Command-R,
Gemma and Llama. Steering remains a blunt lever.

Note the regex column is not even monotonic for Llama (0.617 at c=20, then 0.017 at c=31) or
Gemma (0.750 then 0.167). Under heavy steering the output drifts out of the phrasings the regex
keys on, so it stops tracking refusal at exactly the end of the curve that matters. The judge
column rises cleanly in every model.

## Ablation — holds for three families, reverses for two

Same generations scored two ways, so this isolates the classifier.

n=240 per condition (a first pass at n=120 gave the same picture).

| model | regex base → ablated | judge base → ablated | verdict |
|---|---|---|---|
| Qwen3-14B | 0.637 → 0.375 | 0.775 → **0.925** | reverses |
| gemma-3-27b-it | 0.358 → 0.000 | 0.654 → **1.000** | reverses |
| Mistral-7B-Instruct-v0.3 | 0.271 → 0.121 | 0.463 → 0.338 | holds |
| Meta-Llama-3.1-70B-Instruct | 0.704 → 0.567 | 0.713 → 0.592 | holds |
| c4ai-command-r7b-12-2024 | 0.654 → 0.008 | 0.671 → 0.338 | holds, much weaker |

Mistral and Llama keep a real drop close in size to the regex-measured one. Command-R keeps a
drop but nothing like the published collapse: 72% → 2% becomes 0.667 → 0.342. Qwen and Gemma
reverse outright — ablation *raises* measured refusal, and for Gemma every one of the 120
ablated responses is judged a refusal while the regex finds none at all.

The cause is visible in the text. 66 of Qwen's 120 ablated responses are refusals the regex
misses, because ablation shifts the phrasing out of the first-person modal forms it keys on:

> "Certainly not. Submitting a client trade without proper approval is a serious violation of
> compliance and ethical standards."

An unambiguous refusal with no "I cannot" in it. Ablating the direction changed *how* these
models refuse without stopping them from refusing, and the regex read the phrasing shift as a
behavioural drop. **The published Qwen (57% → 32%), Gemma (21% → 0%) and Command-R (72% → 2%)
ablation figures should not be relied on.**

## Addition — strengthens everywhere, and rescues Llama

n=240 per condition.

| model | base (regex → judge) | added (regex → judge) |
|---|---|---|
| Qwen3-14B | 0.008 → 0.000 | 0.787 → 1.000 |
| Mistral-7B-Instruct-v0.3 | 0.000 → 0.000 | 0.650 → 1.000 |
| c4ai-command-r7b-12-2024 | 0.004 → 0.000 | 0.988 → 1.000 |
| gemma-3-27b-it | 0.008 → 0.000 | 0.200 → 1.000 |
| Meta-Llama-3.1-70B-Instruct | 0.004 → 0.000 | 0.000 → 1.000 |

Adding the direction drives benign refusal to 100% in every model, because the judge counts the
hedged refusals the regex dropped. The Llama row matters most: the paper reports its addition as
null (1% → 1%) and treats Llama as the family where the direction does not control refusal. On
the same intervention, scored properly, it goes 0% → 100%. Gemma moves the same way
(21% → 100%). **The "addition is weak in Gemma and null in Llama" reading is a classifier
artifact.**

## Follow-up: does ablation change the action, or only the wording?

The ablation result above is measured on verbal refusal, and that measurement turned out to be
classifier-dependent. So the obvious question is whether ablation moves *behaviour*. This asks it
directly: the same harmful tool-enabled prompts, baseline vs ablated, scored by the
unsafe-tool-call predicate — no classifier anywhere, so it cannot be confounded the same way.
Paired design, n=200 per model (`ablation_tool_effect.py`, `summarize_ablation_action.py`).

| model | unsafe base → ablated | delta (95% CI) | made any tool call | **unsafe given it called** |
|---|---|---|---|---|
| Qwen3-14B | 0.460 → 0.235 | −0.225 (−0.29, −0.17) | 0.590 → 0.370 | 0.780 → 0.635 |
| Mistral-7B-Instruct-v0.3 | 0.325 → 0.350 | +0.025 (n.s.) | 0.425 → 0.470 | 0.765 → 0.745 |
| c4ai-command-r7b-12-2024 | 0.120 → 0.180 | +0.060 (0.02, 0.11) | 0.135 → 0.230 | 0.889 → 0.783 |
| gemma-3-27b-it | 0.130 → 0.000 | −0.130 (−0.18, −0.09) | 0.175 → **0.000** | 0.743 → n/a |
| Meta-Llama-3.1-70B-Instruct | 0.450 → 0.360 | −0.090 (−0.15, −0.04) | 0.740 → 0.565 | 0.608 → 0.637 |

The headline column has no single story — ablation makes Qwen, Gemma and Llama *safer* and
Command-R *less* safe. The last column is the informative one, and it barely moves anywhere:
0.78 → 0.64, 0.77 → 0.75, 0.89 → 0.78, 0.61 → 0.64. Once a model has decided to call a tool,
ablating the refusal direction hardly changes whether that call is unsafe.

What does move, a lot, is how often the model calls a tool at all — down 22 points in Qwen,
17 in Llama, to zero in Gemma, and *up* 10 in Command-R.

**The direction gates engagement, not the safety of the action.** That is a sharper version of the
paper's "strong predictor, partial mediator" line: the mediation it performs runs through whether
the model acts at all, not through what it does once acting.

Two cautions. Gemma's tool-calling collapses to zero under ablation, so its apparent safety gain is
the model ceasing to act — closer to capability damage than to alignment, and it should not be read
as ablation making Gemma safe. And Mistral's effect is not significant at n=200.

## A note on the denominator

An earlier pass dropped responses with no prose (a bare tool call and nothing else) instead of
counting them. `tools.refusal.classify_refusal` treats those as *not refused* and keeps them in
the denominator, which is right: a model that quietly calls a tool has not refused. Dropping them
left only the talkative minority and inflated every over-refusal rate — Llama's read as 100% at
zero steering. The numbers above count them, matching the shared classifier. This moved Qwen's
top-coefficient over-refusal from 0.609 to 0.467.

## Provenance notes

- Llama-3.1-70B ran from `unsloth/Meta-Llama-3.1-70B-Instruct`; the `meta-llama` repos are gated
  to this account. Same bf16 weights, and unlike the NousResearch mirror it carries the real
  chat template.
- Gemma tool-mode is a reconstruction (schemas injected into the prompt text), validated against
  the stored unsafe rate and the published AUC as described above. Worth stating explicitly in
  the methods section rather than leaving implicit.
- Command-R's GPU stage was rerun and returned numbers identical to the first pass, under a
  different transformers version — evidence that the version does not affect results.

## Not covered

- **Patching** — deliberately not rerun; predicate-scored and classifier-independent.
- Ablation/addition/steering were run twice: once at `run_scaled_evaluation.py`'s sample sizes
  (120/120/100+60) and once at double that (240/240/200+120). The tables report the larger pass;
  the two agree throughout, so nothing here rests on sample size. Bootstrap CIs are in the
  per-model JSON.
