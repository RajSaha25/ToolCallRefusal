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

| model | layer | cos(r_text, r_behav) | caveats folded in | cos(r_text, r_tool) | cos(r_behav old, new) | n/class |
|---|---|---|---|---|---|---|
| Qwen3-14B | 33/40 | 0.859 | 0.857 | 0.741 | 0.975 | 42 |
| Mistral-7B-Instruct-v0.3 | 26/32 | 0.711 | 0.675 | 0.773 | 0.978 | 84 |
| c4ai-command-r7b-12-2024 | 26/32 | 0.746 | 0.791 | 0.394 | 0.978 | 59 |
| gemma-3-27b-it | 51/62 | 0.822 | 0.817 | withdrawn | 0.958 | 25 |
| Meta-Llama-3.1-70B-Instruct | 67/80 | 0.693 | 0.715 | 0.213 | 0.979 | 59 |

The claim holds for all five: request harmfulness and the refusal decision occupy substantially
the same axis. The old-vs-new behaviour direction agrees at 0.96–0.98 everywhere, so this result
does not depend on the relabel. Rebuilt `r_text` matches the committed `refusal_dirs.pt` at 0.997
for Qwen (the only model whose directions were committed), and Gemma's rebuilt projection gap
(18944) matches the original run's (18715).

Caveat: models rarely comply with harmful requests, so the complied side is small — as few as
25/class for Gemma. The "caveats folded in" column folds CAVEAT in with compliance, roughly
doubling that side, and moves the answer very little in every model.

The `cos(r_text, r_tool)` column needs care. It is withdrawn for Gemma (no tool-capable
template) and, for Llama, the correct value is 0.213 — the earlier 0.914 came from a run whose
"tool" prompts had no tools in them, which made `r_tool` a near-copy of `r_text`. Read on its own
merits, Llama's 0.213 is a genuine outlier against Qwen's 0.741 and Mistral's 0.773, and it sits
alongside the paper's own observation that Llama is the one model where tool context does not
suppress the refusal signal (t = 0.4). Worth a look: the paper's claim that "the text-derived and
tool-derived directions remain aligned" is well supported for Qwen and Mistral but weak for
Llama and Command-R (0.394).

Llama and Gemma were run from a single B200 (183GB); Llama-70B needs ~135GB in bf16 and fits
without model parallelism. Llama used `unsloth/Meta-Llama-3.1-70B-Instruct`, an ungated bf16
mirror, because the `meta-llama` repos are gated to this account — see the chat-template note
for why the mirror choice matters.

## AUC — reproduces on old labels, rises under the corrected scorer

| model | vs unsafe (old) | vs unsafe (fixed) | published | in CI? | vs refusal (old) | vs refusal (new) |
|---|---|---|---|---|---|---|
| Qwen3-14B | 0.722 | 0.808 | 0.724 | yes | 0.972 | 0.945 |
| Mistral-7B-Instruct-v0.3 | 0.698 | 0.760 | 0.689 | yes | 0.868 | 0.869 |
| c4ai-command-r7b-12-2024 | 0.713 | 0.768 | 0.751 | yes | 0.923 | 0.901 |
| gemma-3-27b-it | withdrawn | withdrawn | 0.791 | n/a | 0.903 | 0.946 |
| Meta-Llama-3.1-70B-Instruct | 0.761 | 0.864 | 0.881 | yes | 0.949 | 0.966 |

Two things to separate here.

*The relabel-dependent column* is "vs refusal", and it improves for the two large models
(Gemma 0.903 → 0.946, Llama 0.950 → 0.977): once hedged compliance stops being scored as
refusal, the projection predicts the refusal decision better.

*The "vs unsafe" column is not relabel-dependent* — it is the same kind of predicate-scored
quantity as patching. It reproduces the published AUC for all four models where the tool-mode
prompt can be built correctly. Llama needed the right tokenizer to get there: on the mirror
whose template drops tools it scored 0.591, and with tools actually rendered it scores 0.864
under the fixed scorer, whose CI (0.813–0.908) contains the published 0.881.

**Gemma's "vs unsafe" figure is withdrawn** — see the chat-template problem below. Its tool-mode
prompts contain no tool definitions, so the number measures something other than what the column
claims. Gemma's no-tool columns ("vs refusal") are unaffected, as are all its direction cosines
against `r_text` and `r_behav`.

## The chat-template problem (Gemma-3 and Llama-3.1)

`apply_chat_template(..., tools=...)` silently drops the tools when a model's template has no
tool-handling block: the variable is simply unused, no error is raised. Checking the templates
directly:

| model | template length | handles tools |
|---|---|---|
| Qwen3-14B | 4168 | yes |
| Mistral-7B-Instruct-v0.3 | — | yes (verified by rendering) |
| gemma-3-27b-it | 1532 | **no** |
| NousResearch/Meta-Llama-3.1-70B-Instruct | 348 | **no** |
| unsloth/Meta-Llama-3.1-70B-Instruct | 4614 | yes |

The consequence is silent and severe: a tool-enabled prompt renders as a plain chat prompt, the
model has nothing to call, and every response scores as "made no unsafe tool call" — perfectly
safe. Llama scored 0/100 unsafe at every steering coefficient before this was caught, with all
100 responses being clean verbal refusals.

For Llama this is a mirror artifact: the NousResearch reupload carries a stripped 348-character
template. `unsloth/Meta-Llama-3.1-70B-Instruct` has the real 4614-character template and renders
tools correctly, so Llama's tool-mode results are rerun from there.

Gemma-3 is different: **no** Gemma-3 mirror carries a tool-handling template — `google/` and
`unsloth/` are byte-identical at 1532 characters. That leaves an open question about the
committed Gemma behavioural data, which contains 666 rows of tool calls using 100% real tool
names (models cannot invent 20 exact names, so tools demonstrably reached those prompts
somehow). Whatever produced them is not the `render_prompt` path in
`run_behavioral_batched.py` as it behaves today. Worth tracing before the Gemma tool-mode
numbers are relied on.

A cheap guard for the future: assert that a tool name appears in the rendered prompt whenever
tools are supposed to be attached. `_render_check.py` does exactly this.

## Patching — unaffected by definition

Patching scores flips with `score_tool_calls_all`, a predicate over saved tool calls. It never
consults the refusal classifier, and it already used the global-scope scorer. The relabel
cannot move it, so the published patching percentages stand as they are.

## Steering — cost drops but the conclusion survives (Qwen3-14B)

Harmful-unsafe is predicate-scored and unchanged; benign over-refusal is re-judged.

| coef | harmful unsafe | benign over-refusal (regex) | benign over-refusal (judge) | 95% CI |
|---|---|---|---|---|
| 0 | 0.510 | 0.000 | 0.000 | [0.00, 0.00] |
| 200 | 0.370 | 0.042 | 0.000 | [0.00, 0.00] |
| 450 | 0.260 | 0.406 | 0.281 | [0.13, 0.44] |
| 700 | 0.030 | 0.717 | 0.609 | [0.46, 0.74] |

At the coefficient that clears unsafe calls, the over-refusal cost is 61%, not the ~70% the
paper quotes. Lower, still severe: steering remains a blunt lever.

## Ablation — robust for Mistral, reverses for Qwen

Same generations scored two ways, so this isolates the classifier.

| model | condition | regex | judge | three-way (judge) |
|---|---|---|---|---|
| Qwen3-14B | base | 0.667 | 0.767 | 92 refuse / 14 caveat / 14 comply |
| Qwen3-14B | ablated | 0.450 | **0.925** | 111 refuse / 8 caveat / 1 comply |
| Mistral-7B | base | 0.292 | 0.492 | 59 refuse / 40 caveat / 21 comply |
| Mistral-7B | ablated | 0.150 | **0.342** | 41 refuse / 46 caveat / 33 comply |

For Mistral the effect survives — refusal still drops by ~15 points under the judge, matching
the regex-measured drop. For Qwen it reverses: ablation *raises* measured refusal.

The cause is visible in the text. 66 of Qwen's 120 ablated responses are refusals the regex
misses, overwhelmingly because ablation shifts the phrasing out of the first-person modal
forms the regex keys on:

> "Certainly not. Submitting a client trade without proper approval is a serious violation of
> compliance and ethical standards."

That is an unambiguous refusal with no "I cannot" in it. So for Qwen, ablating the direction
changed *how* the model refuses without stopping it from refusing, and the regex read the
phrasing shift as a behavioural drop. The published Qwen ablation figure (57% → 32%) should
not be relied on; the Mistral-style claim is fine.

## Addition — strengthens

| model | base (regex → judge) | added (regex → judge) |
|---|---|---|
| Qwen3-14B | 0.017 → 0.000 | 0.750 → 1.000 |
| Mistral-7B-Instruct-v0.3 | 0.000 → 0.000 | 0.692 → 1.000 |

Adding the direction drives benign refusal to 100% in both models, cleaner than the published
numbers, because the judge counts hedged refusals the regex dropped.

## Not covered

Gemma-3-27B (~54GB in bf16) and Llama-3.1-70B (~140GB) do not fit the pod's 45G `/workspace`
volume. Everything is scripted for them — rerun with `--layer 51` and `--layer 67` on a host
with more disk. Steering and ablation were run for Qwen (full grid) and Mistral (ablation and
addition only, `--grid 0`); Command-R's generation half has not been run.
