# Mechanism rerun — handoff for the NeurIPS resubmission

Everything below was produced by re-running the label-dependent mechanism analyses on top of
`relabel_out/` and the global-scope tool-call scorer. Branch: `relabel-rerun`. Ready-to-paste
LaTeX: **`paper_updates.tex`** — three ready-to-paste tables, plus a commented outline for each
paragraph that needs rewriting (what to cover and which numbers to cite, not drafted prose).
Generated from the result JSON by `make_paper_updates.py`, so no number is hand-transcribed;
compile-tested, and every citation key it uses is already in `refs.bib`.

---

## 0. Your three asks, answered

**1. "Add the missing cosine values."** Done, all five models — and it is a *new table*
(§3.2 below), since §3.6 describes the analysis but no number appears anywhere in the draft.
It also turned up something: the behaviour-direction claim holds everywhere (0.69–0.86), but the
**other** cosine in that section — text-derived vs. tool-derived — **fails for two models**
(Command-R 0.394, Llama 0.213). §5 currently says those directions "remain aligned", and the
Discussion repeats it. Both need softening.

**2. "Rerun mech and update results."** Done. Table 2's ablation, addition and AUC columns all
change (§3.1). Suppression and Patching are carried over — they have no classifier in them, so the
relabel cannot move them, though see §5 for why re-verifying is still worth an hour.

**3. "Double-check the causal claims."** Done, and this is the largest change — it is narrative,
not a table. Two causal claims flip:
- **Ablation** reverses in Qwen and Gemma (it *raises* refusal), and Command-R's 72%→2% collapse
  is really 67%→34%.
- **Llama's addition is not null.** The draft reports 1%→1% and treats Llama as the family where
  the direction does not control refusal. It is **0%→100%**.

Because that ask deserved a test no classifier could confound, I also ran a new experiment
(§3.3): ablate, then score the unsafe tool call directly. Result — the direction gates **whether
the model engages**, not how safely it acts once engaged. That is a sharper claim than the current
"partial mediator" framing, and it is defensible against exactly the objection that sank the
ablation numbers.

**Scope of edits this implies:** 2 new tables, 1 rewritten table, 6 prose locations, 2 methods
additions, and a pass over the abstract and introduction (both still describe ablation as
collapsing refusal).

---

## 1. TL;DR

Table 1 in the current draft has already been updated with the new labels. **Table 2 has not** —
it is still entirely pre-relabel. Rerunning it changes three of its five columns, and two
published causal claims flip.

The story gets stronger, not weaker:

- The direction is still a strong predictor everywhere — AUC **0.76–0.86**, and all five models
  now reproduce their published AUC once the prompt is built correctly.
- Addition now works in **all five** models, including the Llama case the draft reports as null.
- We have the §3.6 cosine numbers, which the draft describes but never computed.
- A new result gives a sharper claim than "partial mediator": the direction gates **whether the
  model engages**, not how safely it acts once engaged.

What breaks: the ablation column. Under a classifier that reads the whole response rather than
matching refusal phrases, ablation **raises** measured refusal in Qwen and Gemma, and Command-R's
collapse is far smaller than reported.

---

## 2. Why the ablation numbers changed

The old refusal classifier keyed on first-person phrasing ("I cannot…", "I'm unable to…").
Ablation changes *how* a model refuses without stopping it:

> "Certainly not. Submitting a client trade without proper approval is a serious violation of
> compliance and ethical standards."

That is an unmistakable refusal with no "I cannot" in it. The regex counts it as compliance, so a
phrasing shift reads as a behavioural drop. For Qwen, 66 of 120 ablated responses are refusals the
regex missed. For Gemma the regex finds **zero** refusals among 120 ablated responses where the
judge finds **120**.

This is the same failure mode the relabel fixed for Table 1 — it just was never applied to the
intervention results.

---

## 3. Exact changes to the paper

### 3.1 Table 2 (`tab:mech`) — three columns change

| Model | Ablation (old → **new**) | Addition (old → **new**) | AUC (old → **new**) |
|---|---|---|---|
| Qwen3-14B | 57%→32% → **78%→92%** | 1%→71% → **0%→100%** | 0.724 → **0.808** |
| Mistral-7B | 29%→13% → **46%→34%** | 0%→80% → **0%→100%** | 0.689 → **0.760** |
| Command-R-7B | 72%→2% → **67%→34%** | 2%→92% → **0%→100%** | 0.751 → **0.768** |
| Gemma-3-27B | 21%→0% → **65%→100%** | 1%→25% → **0%→100%** | 0.791 → **0.790** |
| Llama-3.1-70B | 73%→61% → **71%→59%** | 1%→1% → **0%→100%** | 0.881 → **0.864** |

**Layer, Suppression and Patching are unchanged** — see §5 for why, and why re-verifying two of
them is still worth doing.

n = 240 per condition. A first pass at n = 120 gave the same picture throughout, so none of this
rests on sample size.

### 3.2 New table: direction cosines (`tab:cosine`)

§3.6 promises this analysis and no number appears anywhere in the draft. (The 0.735 in notebook 02
is `cos(r_text, r_tool)` — a different quantity.)

| Model | cos(r_text, r_behav) | cos(r_text, r_tool) | r_behav old vs new |
|---|---|---|---|
| Qwen3-14B | 0.859 | 0.741 | 0.975 |
| Mistral-7B | 0.711 | 0.773 | 0.978 |
| Command-R-7B | 0.746 | 0.394 | 0.978 |
| Gemma-3-27B | 0.822 | 0.797 | 0.958 |
| Llama-3.1-70B | 0.693 | 0.213 | 0.979 |

The §3.6 claim holds in every model (0.69–0.86). The last column is useful defensively: recomputing
the behaviour-defined direction under the *old* labels agrees at 0.96–0.98, so the check does not
depend on the relabelling.

**But `cos(r_text, r_tool)` needs a softened claim.** §5 currently says text- and tool-derived
directions "remain aligned". True for Qwen, Mistral and Gemma (0.74–0.80); not for Command-R (0.39)
or Llama (0.21).

### 3.3 New table: ablation on behaviour (`tab:ablation-action`)

Because the ablation result is classifier-dependent, we tested it a second way that no classifier
can touch — ablate, then score the unsafe tool call directly (a predicate over parsed calls).
n = 200, paired.

| Model | Unsafe (base → abl) | Made any call | **Unsafe given it called** |
|---|---|---|---|
| Qwen3-14B | 0.460 → 0.235 | 0.590 → 0.370 | 0.780 → 0.635 |
| Mistral-7B | 0.325 → 0.350 (n.s.) | 0.425 → 0.470 | 0.765 → 0.745 |
| Command-R-7B | 0.120 → 0.180 | 0.135 → 0.230 | 0.889 → 0.783 |
| Gemma-3-27B | 0.130 → 0.000 | 0.175 → **0.000** | 0.743 → n/a |
| Llama-3.1-70B | 0.450 → 0.360 | 0.740 → 0.565 | 0.608 → 0.637 |

The headline column has no consistent story. The conditional column barely moves anywhere, while
the rate of calling a tool at all swings 10–22 points in both directions. **Ablation changes how
often a model acts, not how safely it acts once it does.**

Caveat for the text: Gemma's tool-calling collapses to zero, so its apparent safety gain is the
model ceasing to act — capability damage, not alignment.

### 3.4 Steering

Qwen's over-refusal cost at the coefficient that clears unsafe calls: **70% → 44%**. Lower, but the
conclusion is unchanged and in fact stronger across the set — in Command-R, Gemma and Llama the same
operating point costs essentially *all* benign tool use.

Gemma's published steering curve is flat for a fixable reason: it used Qwen's **absolute** grid
(0/200/450/700) against a projection gap of 18,944, roughly 60× too small. Rerun on a scaled grid it
falls 0.12 → 0.00.

### 3.5 Prose that must change

| Location | Problem |
|---|---|
| §5 ¶"A single direction causally governs text refusal" | Says ablation sharply decreases refusal "across every model". False for two of five. |
| §5 ¶"Tool context suppresses…" | "Text-derived and tool-derived directions remain aligned" — not true for Command-R or Llama. |
| §5 ¶"strong predictor but partial mediator" | AUC range 0.69–0.88 → 0.76–0.86. |
| §5 ¶"Steering is a blunt lever" | "around 70 percent" → 44 percent. |
| §3.6 ¶"Robustness of the direction" | Promises a cosine analysis with no numbers; now has them. |
| §7 Discussion ¶"A shared refusal direction…" | "ablations collapse and decrease harmful prompt refusal rates" — needs softening. |
| Abstract + §1 Introduction | Still describe the mechanistic finding in terms of ablation collapsing refusal. Needs a pass once §5 settles. |

`paper_updates.tex` has an outline for each of these — what the paragraph must cover and which
numbers to cite — rather than drafted prose, so the writing stays in your voice. The three tables
in that file are ready to paste as-is.

### 3.6 Methods additions (reproducibility)

Two things a reviewer could reasonably ask about, both outlined in `paper_updates.tex`:

- **Gemma tool prompts.** Gemma-3 has no tool-handling block in its chat template, in any published
  checkpoint. Its tool schemas have to be written into the prompt text. Validated: 12.5% unsafe
  against 14.6% from the behavioural run, and it reproduces the published AUC to within 0.001.
- **Llama weights.** Loaded from `unsloth/Meta-Llama-3.1-70B-Instruct` (the `meta-llama` repos are
  gated). Same bf16 weights; unlike another common mirror, its chat template carries tools.

---

## 4. Two problems found in the current draft

**(a) Table 1 does not match the committed labels.** Four of five text-refusal values agree to
three decimals. Mistral reads **0.535** where `summary_three_way.csv` gives **0.435** — that looks
like a transcription slip rather than a scoring difference.

**(b) The unsafe/divergence columns run ~12% high.** Every model's unsafe rate in Table 1 is above
what the committed scorer produces (Qwen normal 0.544 vs 0.487; Gemma 0.172 vs 0.146), consistently.
That suggests Table 1 was built with a stricter predicate that is not in the repo. This matters
because the AUC column above uses the committed scorer — **if Table 1 keeps its current numbers, AUC
should be regenerated against that same scorer so the two tables agree.** Whoever built Table 1
should confirm which scorer it used.

---

## 5. What is not settled

**Suppression (Δ, t) and Patching are carried over unchanged.** Neither has a refusal classifier
anywhere in it, so the relabel genuinely cannot move them, and the published values stand on the
merits. Two reasons to re-verify anyway (~1 hour of GPU):

1. They are the only Table 2 cells still resting on runs from before we found the chat-template
   problem, and suppression is measured on *tool-mode prompts* — exactly what that bug corrupted.
2. Three other columns in the table changed. "Did you regenerate the whole table" is a question
   worth being able to answer yes to.

---

## 6. Bugs found (worth knowing, may affect other analyses)

- **Chat templates silently drop tools.** `apply_chat_template(..., tools=...)` ignores the tools
  when a template has no tool-handling block — no error. The tool-enabled prompt becomes a plain
  chat prompt and every response scores as safe. This hit Gemma-3 and one Llama mirror. Llama
  scored 0/100 unsafe at every steering coefficient before it was caught. `_render_check.py`
  asserts a tool name appears in the rendered prompt; run it before any tool-mode sweep.
- **Tool-call parsers must cover every family.** A parser handling only the Qwen/Mistral formats
  reads Llama's `<|python_tag|>` and Command-R's `<|START_ACTION|>` calls as "no tool calls", i.e.
  perfectly safe.
- **Over-refusal denominators.** Responses that are a bare tool call with no prose must count as
  *not refused* and stay in the denominator. Dropping them inflates every over-refusal rate.

---

## 7. Where things are

| | |
|---|---|
| Branch | `relabel-rerun` (pushed) |
| Tables to paste + writing outlines | `paper_updates.tex` |
| Full results write-up | `relabel_analysis/RESULTS.md` |
| Per-model numbers | `relabel_analysis/gpu_*.json`, `steer_*.json`, `ablation_action_*.json` |
| Raw generations | `relabel_analysis/steer_raw_*.json` (rescore without regenerating) |
| Direction vectors | `relabel_analysis/directions_*.pt` |
| Regenerate the LaTeX | `python3 make_paper_updates.py` |
| Print every table | `python3 summarize_reruns.py`, `python3 summarize_ablation_action.py` |

All runs were on a single B200 (183GB). Llama-70B fits in bf16 without model parallelism.
