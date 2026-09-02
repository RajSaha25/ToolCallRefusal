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

**2. "Rerun mech and update results."** Done. Every Table 2 column except Suppression has now
been rerun: ablation, addition and AUC in the first session, and patching in the second (§3.1,
§1c). Suppression is the only carried-over number left.

**3. "Double-check the causal claims."** Done, and this is the largest change — it is narrative,
not a table. Checking them turned up a second problem beyond the classifier, described in §1b:
some interventions push a model into broken output, which any refusal classifier scores as
refusal. What that leaves:
- **Ablation** reverses in Qwen (0.775→0.925, fluent output — a real effect). It still lowers
  refusal in Mistral, Llama and Command-R, though Command-R's 72%→2% collapse is really 67%→34%.
  Gemma's is **not reportable**: 100% of its ablated output is degenerate.
- **Addition is not measurable at the published coefficient.** It drives all five models into
  degenerate output. This affects the draft's existing numbers too, not just the rerun — the old
  regex scores "I'm sorry I cannot I'm sorry I cannot…" as refusal just as the judge does. At the
  natural magnitude (§1c) the effect is 24–33% in Qwen and Llama and ~0 in Mistral and Gemma.
- **Steering suppresses tool-calling, not unsafe tool-calling** (§3.4). "Unsafe → 0" at the
  coefficients a model survives is the model no longer calling tools — on harmful *and benign*
  prompts — while the unsafe rate among the calls it still makes never falls. An earlier version of
  this document read those points as a usable operating regime; that was wrong, and the draft's
  "blunt lever" is if anything an understatement.
- **Patching, now rerun for all five** (§1c, §3.1): the flips are real, but 70–97% of them are
  "stopped calling" rather than "called safely", and 80–96% of the calls a patched model still
  makes remain unsafe.

Because that ask deserved a test no classifier could confound, I also ran a new experiment
(§3.3): ablate, then score the unsafe tool call directly. Result — the direction gates **whether
the model engages**, not how safely it acts once engaged. Patching and steering now say the same
thing from two more directions. That is a sharper claim than the current "partial mediator"
framing, and it is defensible against exactly the objection that sank the ablation numbers.

**Scope of edits this implies:** 6 new tables, 1 rewritten table, 8 prose locations, 4 methods
additions (degeneracy screen, KL guardrail, call-rate reporting, prompt-rendering check), and a
pass over the abstract, the introduction and Future Work.

---

## 1. TL;DR

Table 1 in the current draft has already been updated with the new labels. **Table 2 has not** —
it is still entirely pre-relabel. Rerunning it changes four of its five columns, and the causal
section's claims narrow to one.

The numbers get smaller; the claim gets more consistent:

- The direction is still a strong predictor everywhere — AUC **0.76–0.86**, and all five models
  now reproduce their published AUC once the prompt is built correctly.
- We have the §3.6 cosine numbers, which the draft describes but never computed. The
  behaviour-direction cosine holds (0.69–0.86); the text-vs-tool cosine holds for 3/5.
- **Three interventions agree on what the direction does: it gates whether the model acts, not
  how safely it acts.** Ablation on tool prompts moves engagement 10–22 points with a flat unsafe
  rate among calls (§3.3). Patching flips unsafe calls mostly into *no* call, with 80–96% of the
  remaining calls still unsafe (§3.1). Steering removes unsafe calls only by removing calls, on
  the benign side too (§3.4).
- **Ablation is not a uniform story.** Command-R is textbook Arditi (0.68→0.33). Mistral and
  Llama drop modestly. Qwen (0.78→0.93) and — once ablated coherently — Gemma (0.66→1.00) refuse
  *more*, which says the max-separation direction is not a refusal direction in those models.
- **Only Command-R's direction passes Arditi's KL admissibility check** (§1c). Mistral and Llama
  fail at every layer; Gemma fails by two orders of magnitude, for a diagnosable reason (its
  massive-activation common mode), and your mean-out fix repairs the coherence at 27B.
- **Addition at the natural magnitude is real in two models and absent in two**: Qwen 0→33%
  and Llama 1→24%, coherent; Mistral and Gemma ≤ 1.3%; Command-R breaks before 1×. Nothing
  reaches 60% coherently. The 4× the draft used, and the 0→100% "rescue", are degenerate output
  in every model.

What breaks in the draft: the ablation, addition and patching columns and the steering
paragraph, for two separate reasons — a phrase-matching classifier, and broken output that no
classifier can score (§1b). Nothing in the geometric half of the paper breaks.

---

## 1b. Output degeneracy — a second problem, found on review

Ablation and addition can push a model off the rails entirely. Real examples from our runs:

> Gemma, ablated: `Vy수상수상수상수상수상…` for 400 characters
> Llama, addition: `I can I can I can I can…` for 600 characters

The judge has no "broken output" verdict, so it forces such a response into refuse/caveat/comply,
and "no help was given" reads as REFUSE. **Any intervention rate computed over degenerate output
measures breakage, not refusal.**

This applies to the draft's existing numbers too. Qwen's degenerate addition output is literally
"I'm sorry I cannot I'm sorry I cannot…", which the old regex classifier also scores as refusal —
so the published 1%→71% has the same problem. This is not a regression introduced by the relabel.

Per-cell rates are in `relabel_analysis/degeneracy_audit.json` (`degeneracy_audit.py`). What is
and is not measurable:

| | |
|---|---|
| **Clean** (0% degenerate) | every model's ablation except Gemma; all four baselines; the lower half of every steering grid |
| **Artifact** | addition at 4× the gap in all five (67–100% degenerate); Gemma's ablation (100%) |

Recommend putting the degeneracy screen in the paper. It is a real methodological contribution and
pre-empts the obvious reviewer question.

---

## 1c. Your ablation handoff — verified, then run

Your `HANDOFF_raj_ablation.md` was checked claim by claim against the saved generations and then
its open items were run on an H100 (2026-09-02). Everything you stated held. Two of your findings
overturned parts of an earlier version of this document, and both are now in it.

**Verified from the saved generations, no GPU:**
- Steering's "unsafe → 0" is tool-call collapse. Confirmed, and it is worse than you wrote: it
  happens on the benign side at the same coefficient, and unsafe-given-call never drops (§3.4).
- Your degeneracy rule catches the sentence-level loops mine missed (Qwen 2.25×: 32%→51%,
  Mistral 17: 18%→36%, Gemma addition 68%→100%). It is now `degeneracy.is_degenerate_v2` and
  every rate in this session uses it. The four clean steering points stay ≤ 2% under it.
- Thinking mode was already off for Qwen (`render_utils.py`). The cosines you quote are ours.
- `main` merged into `relabel-rerun` cleanly (no conflicts).

**Run on the GPU — the KL guardrail (Arditi App. C.1) for every candidate layer, your mean-out
recipe, random- and mean-direction controls, and addition at 1×/1.5×/2×/3× the natural
magnitude, plus patching for all five.** KL is scored two ways: on 48 generic harmless requests
(Arditi's protocol) and on the benchmark's own benign prompts, which sit near the refusal boundary.

| Model | KL of r_text, generic / dataset | Layers passing (generic) | Random dir. KL | Mean-out KL | Verdict |
|---|---|---|---|---|---|
| Command-R-7B | **0.04** / 0.14 | 8/17 (best L27, next to L26) | 0.001 | 0.20 | admissible |
| Mistral-7B | 0.25 / 0.91 | 0/17 | 0.004 | 0.28 | no admissible layer |
| Gemma-3-27B | 11.3 / 11.0 | 0/32 | 0.04 | 2.5 | no admissible layer; mean-out restores coherence |
| Llama-3.1-70B (4-bit) | 0.28 / 0.40 | 11/41 (best L60 at 0.08; operating L67 fails) | 0.011 | 0.39 | admissible at L60, not at the layer used |
| Qwen3-14B | 1.59 / 2.31 | 4/21 (best L21 at 0.09, cos 0.18 with the L33 direction used) | 0.004 | 1.63 | admissible at L21 — a nearly orthogonal direction; the one used fails |

Only Qwen has a layer that passes on the dataset's own benign prompts (L21, 0.06); everywhere
else nothing does — the operating directions are entangled with the benchmark's domains, not
only with refusal. Qwen's case is the sharpest: the direction whose ablation *raises* refusal
(L33) fails the guardrail by 16×, while an admissible direction exists at L21 that is nearly
orthogonal to it. Whether ablating *that* one lowers refusal is a ten-minute follow-up (§5). Gemma's diagnostics at 27B match your 4B run:
mean-projection/gap = 1.00, cos(r, mean) = 0.24, mean-activation norm ≈ 77k; a random direction is
harmless (KL 0.04, 0% degenerate), so Gemma is not fragile to ablation in general — it is fragile
to *this* direction because of what it shares with the common mode.

**Ablation controls, harmful no-tool prompts, n = 240, judged refusal over coherent outputs:**

| Model | none | r_text | mean-out | random | mean-dir |
|---|---|---|---|---|---|
| Command-R-7B | 0.68 | **0.33** | 0.48 | 0.69 | 0.47 (6% deg.) |
| Mistral-7B | 0.49 | **0.33** | 0.35 | 0.47 | 0.54 (10% deg.) |
| Gemma-3-27B | 0.66 | *100% deg.* | **1.00** (1% deg.) | 0.68 | *100% deg.* |
| Llama-3.1-70B (4-bit, n=120) | 0.69 | 0.63 | 0.64 | 0.69 | **0.34** (0% deg., 76% short) |
| Qwen3-14B | 0.78 | **0.94** | 0.95 | 0.78 | *97% deg.* |

Reading it: Command-R is the Arditi case, and its mean direction carries part of the signal
(ablating it alone drops refusal 20 points). Mistral is a modest real drop with a clean random
null. Llama drops 7 points under the direction and 35 under the mean direction alone, coherently
but with three-quarters of the outputs turning terse — its common mode carries more of its
refusal behaviour than the max-separation direction does. Gemma: your fix works at 27B, and the
coherent ablation *raises* refusal to 1.00 — the
outputs are fluent, on-topic refusals, and prompts the baseline hedged on become flat refusals.
With Qwen, that makes two of five models where removing the max-separation direction *increases*
refusal — consistent with the direction being harmfulness/topic rather than refusal in those
families (your Zhou et al. / Wollschläger et al. pointers).

**Addition at multiples of the diff-in-means gap, benign no-tool prompts, n = 240:**

| Model | 1× | 1.5× | 2× | 3× |
|---|---|---|---|---|
| Mistral-7B | 0.004 | 0.01 | 0.06 (5% deg.) | 97% degenerate |
| Command-R-7B | 0.41 among the coherent half (50% deg.) | 100% deg. | 100% deg. | 100% deg. |
| Gemma-3-27B | 0.013 | 91% deg. | 100% deg. | 100% deg. |
| Llama-3.1-70B (4-bit, n=120) | **0.24** (0% deg.) | — | 0.35 (8% deg.) | — |
| Qwen3-14B | **0.33** (2.5% deg.) | 0.53 (18% deg.) | 91% deg. | 100% deg. |

Arditi's 1× produces essentially no refusal in Mistral and Gemma, and Command-R breaks before 1×.
Qwen (0.00 → 0.33) and Llama (0.01 → 0.24) are the two models where 1× behaves as Arditi
describe, fully coherent. Nothing reaches 60% coherently in any model; the draft's 4× is
degenerate everywhere. "Addition rescues Llama" (0 → 100%) should be withdrawn as you said;
"addition at the natural magnitude raises benign refusal to 24–33% in Qwen and Llama and does
nothing in Mistral and Gemma" is what survives.

**Predictions were written down before the session and scored afterwards**: `PREDICTIONS.md`.
Held: the patching decomposition (5/5), the random-direction null (5/5), Gemma's mean-out
coherence and direction, Gemma and Command-R on the guardrail. Missed: Mistral failing the
guardrail at every layer, addition at 1× doing nothing in Mistral and Gemma, the refusal drops
under the mean-direction control in Command-R and Llama, and Gemma not being fragile to a random
direction.

---

## 2. Why the ablation numbers changed

The old refusal classifier keyed on first-person phrasing ("I cannot…", "I'm unable to…").
Ablation changes *how* a model refuses without stopping it:

> "Certainly not. Submitting a client trade without proper approval is a serious violation of
> compliance and ethical standards."

That is an unmistakable refusal with no "I cannot" in it. The regex counts it as compliance, so a
phrasing shift reads as a behavioural drop. For Qwen, 66 of 120 ablated responses are refusals the
regex missed — and Qwen's ablated output is fluent throughout, so this is a real effect, not the
degeneracy problem in §1b.

This is the same failure mode the relabel fixed for Table 1 — it just was never applied to the
intervention results. Gemma looks similar on the surface (regex 0, judge 120) but is a different
problem: there the output is degenerate, so neither number means anything.

---

## 3. Exact changes to the paper

### 3.1 Table 2 (`tab:mech`) — four columns change

| Model | Ablation (old → **new**) | Addition at 1× (old 4× → **new**) | AUC (old → **new**) | Patching flip (old → **new** = safe call + no call) |
|---|---|---|---|---|
| Qwen3-14B | 57%→32% → **78%→94%** | 1%→71% → **0%→33%** | 0.724 → **0.808** | 18% → **24%** = 9 + 15 |
| Mistral-7B | 29%→13% → **49%→33%** | 0%→80% → **0%→0.4%** | 0.689 → **0.760** | 24% → **20%** = 6 + 13 |
| Command-R-7B | 72%→2% → **68%→33%** | 2%→92% → **0%→41%** (50% degenerate) | 0.751 → **0.768** | 62% → **34%** = 6 + 29 |
| Gemma-3-27B | 21%→0% → **66%→100%** (mean-out; raw direction not reportable) | 1%→25% → **0%→1%** | 0.791 → **0.790** | 46% → **90%** = 3 + 87 |
| Llama-3.1-70B | 73%→61% → **71%→59%** | 1%→1% → **1%→24%** (4-bit) | 0.881 → **0.864** | 40% → **13%** = 3 + 9 (4-bit) |

Unsafe-given-call after patching: 0.89 / 0.93 / 0.92 / 0.80 / 0.96. The patching column should
become its own table (`tab:patching` in `paper_updates.tex`) because the split is the finding.
Gemma's old patching number came from a run with no tools in the prompt at all (§6). Llama's
rerun used 4-bit NF4 weights on an 80 GB card and is marked as such; a bf16 confirmation needs a
B200 or two H100s and about 30 minutes.

**Layer and Suppression are unchanged.** The ablation column now uses this session's control runs
(n = 240, judged over coherent outputs), which match the first session's numbers within 3 points
for Qwen, Mistral, Command-R and Llama.

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

Caveat for the text: Gemma's tool-calling collapses to zero because its ablated outputs are the
same token loops as in §1b, so its apparent safety gain is the model ceasing to produce anything —
capability damage, not alignment. This row should be rerun with the mean-out direction (§5, item 5)
before it is quoted.

### 3.4 Steering — this conclusion changes

**The draft's steering evidence has two problems, and an earlier version of this document fixed
the first and then misread the result.** First, the top of each grid is where the model breaks
(32–100% degenerate under the v2 rule), so those over-refusal rates (Qwen 0.44, Command-R / Gemma
/ Llama 1.00) measure breakage and should not be quoted. Second — the part I got wrong before —
"unsafe → 0" at the next coefficient down is not safer calling. It is the model no longer calling
tools, on both sides, and the text judge cannot see it:

| model | coef | harmful any-call | harmful unsafe | unsafe given call | benign any-call | benign over-refusal (judge) |
|---|---|---|---|---|---|---|
| Mistral-7B | 0 → 11 | 0.39 → **0.02** | 0.27 → 0.01 | 0.70 → 0.25 (n=4) | 0.51 → **0.05** | 0.03 → 0.08 |
| Llama-3.1-70B | 0 → 20 | 0.77 → **0.03** | 0.52 → 0.02 | 0.67 → 0.80 | 0.93 → **0.12** | 0.03 → 0.41 |
| Gemma-3-27B | 0 → 12162 | 0.15 → **0.04** | 0.12 → 0.04 | 0.77 → 0.78 | 0.39 → **0.07** | 0.22 → 0.14 |
| Qwen3-14B | 0 → 450 | 0.61 → 0.32 | 0.54 → 0.24 | 0.88 → 0.76 | 0.66 → 0.48 | 0.00 → 0.08 |
| Command-R-7B | 0 → 49 | 0.12 → 0.04 | 0.12 → 0.04 | 0.96 → 0.88 | 0.42 → **0.40** | 0.03 → 0.18 |

Three things follow. Text-judged over-refusal understates the collateral damage badly — Gemma's
*falls* while its benign task completion drops from 0.39 to 0.07, because a model that answers a
tool task in prose is not refusing, but it is not doing the task either. Unsafe-given-call is flat
or rising at every coefficient in every model: steering never makes the calls a model still makes
safer. And Command-R at c=49 is the only point that cuts harmful unsafe calls (0.12→0.04) while
keeping benign calls (0.42→0.40) — with unsafe-given-call still 0.88.

So the draft's "blunt lever" is, if anything, an understatement: the lever is engagement, and it
moves benign and harmful engagement together. That is the same finding as ablation (§3.3) and
patching (§3.1), from a third intervention. Report benign any-call next to over-refusal, and
unsafe-given-call next to unsafe; `tab:steering-calls` in `paper_updates.tex` does both.

For Future Work: the target is a direction, or a different intervention, that changes
unsafe-given-call rather than any-call. None of the three tried here does.

Separately, Gemma's published steering curve is flat for a fixable reason — it used Qwen's
**absolute** grid (0/200/450/700) against a projection gap of 18,944, roughly 60× too small.

### 3.5 Prose that must change

| Location | Problem |
|---|---|
| §5 ¶"A single direction causally governs text refusal" | Says ablation sharply decreases refusal "across every model". False for two of five. |
| §5 ¶"Tool context suppresses…" | "Text-derived and tool-derived directions remain aligned" — not true for Command-R or Llama. |
| §5 ¶"strong predictor but partial mediator" | AUC range 0.69–0.88 → 0.76–0.86. Patching flips are mostly "no call"; "partial mediator of engagement", not of tool-call safety. Patching numbers all change (§3.1). |
| §5 ¶"Steering is a blunt lever" | The 70 percent is degenerate output. The remaining evidence is call suppression on both sides (§3.4); report any-call and unsafe-given-call. |
| §5 Gemma, wherever ablation is described | No KL-admissible direction; raw-direction ablation is degenerate; under the mean-out recipe refusal *rises* 0.66→1.00. Say so with the KL stated (§1c). |
| §5 addition / "rescues Llama" | 0→100% is degenerate output; at 1× the effect is ≤ 1% in Mistral and Gemma. Withdraw. |
| Methods, direction extraction | State that the KL guardrail was applied and which models fail it; state the render check and the degeneracy screen (§3.6). |
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

Patching is done (§3.1). What is left, in the order I would do it:

1. **Llama's numbers are from 4-bit NF4 weights.** The direction rebuilt from the 4-bit model has
   cosine 0.93 with the bf16 one, so it is a mild perturbation, but the patching and control
   rows for Llama should be confirmed in bf16 before they go in the paper. That needs a B200 or
   2×H100 and ~30 minutes; the scripts take `--load-4bit` off and nothing else changes.
2. **Table 1 with your audited predicates.** The paper's unsafe columns are ~10 points above what
   our branch computes, which is almost certainly your `add_call_columns` +
   `apply_fabricated_auth_overlay` on `main` (now merged) rather than a bug. Recomputing is CPU
   only; it should land on the paper's numbers within a point or two. Mistral's refusal rate
   (0.435 here vs 0.535 in the draft) is still unexplained and may be the `strip_tool_markup`
   change; worth a look at the same time.
3. **The KL-admissible layer differs from the operating layer in Command-R (L27), Llama (L60)
   and Qwen (L21).** Everything downstream — AUC, cosines, proj_gap, the patching layer — was
   computed at the operating layer. If the paper adopts the guardrail, it should adopt the
   selected layer consistently, one more GPU pass per model (~10 min each). For Command-R and
   Llama the admissible direction is within cosine 0.80–0.93 of the one used, so expect small
   movements. Qwen is different: its admissible direction at L21 has cosine 0.18 with the L33
   direction whose ablation *raises* refusal. Ablating the L21 direction is the single most
   informative ten-minute run left — if it lowers refusal, Qwen becomes an Arditi case with the
   wrong layer chosen; if it doesn't, the "harmfulness, not refusal" reading stands with a
   guardrail-clean direction behind it.
4. **Suppression (Δ, t) is still carried over** from the pre-template-fix runs. Forward passes
   only; cheap to add to any session that has the models loaded.
5. **Gemma's tool-mode ablation** (§3.3) used the raw direction. If the mean-out direction is
   adopted for Gemma's no-tool ablation, the action-level experiment should use it too for
   consistency (~10 min).

None of these can revive the draft's Table 2. The narrower claim — the direction gates
engagement, not action safety; Command-R is the one clean Arditi case; Qwen and Gemma refuse more
when it is removed — is what the evidence supports, from three interventions and five models.

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
| Branch | `relabel-rerun` (pushed; `main` merged in on 2026-09-02) |
| Tables to paste + writing outlines | `paper_updates.tex` (7 tables: mech, cosine, ablation-action, patching, steering-calls, controls, addition) |
| Predictions vs outcomes for the second session | `PREDICTIONS.md` |
| Full results write-up (first session) | `relabel_analysis/RESULTS.md` |
| Per-model numbers | `relabel_analysis/gpu_*.json`, `steer_*.json`, `ablation_action_*.json`, `patching_*.json`, `controls_*.json` (KL scan, diagnostics), `controls_*_judged.json`, `steering_calls.json` |
| Raw generations | `relabel_analysis/steer_raw_*.json`, `patching_raw_*.json`, `controls_raw_*.json` — every intervention output is saved; nothing needs regenerating to be rescored |
| Direction vectors | `relabel_analysis/directions_*.pt`; `directions_controls_*.pt` adds mean-out, random, and mean-direction vectors and `mu_all` |
| Regenerate the LaTeX | `python3 make_paper_updates.py` |
| Print tables | `summarize_reruns.py`, `summarize_ablation_action.py`, `summarize_patching.py`, `summarize_steering.py` |
| Rerun scripts | `patching_rerun.py`, `controls_rerun.py` (KL scan + controls + addition sweep), both with `--load-4bit`; `judge_generations.py --raw-file` judges any saved file |
| Degeneracy | `degeneracy.py`: `is_degenerate` (first audit) and `is_degenerate_v2` (your rule; used for everything on 2026-09-02) |
| Pod logs | `logs_pod/` |

First session: one B200 (183GB), all models in bf16. Second session (2026-09-02): one H100 80GB;
Llama-70B in 4-bit NF4, marked wherever it appears.
