# A Guide to the Mechanistic Interpretability of Tool-Conditional Refusal

*Written for someone new to mechanistic interpretability. It explains the concepts from
scratch, then connects each idea to the exact experiments, code, and figures in this repo.*

> Companion docs: `MECHANISTIC_INTERP.md` is the terse results summary. **This file is the
> teaching version** — read it first if the methods are unfamiliar.

---

## 0. The one-paragraph version

A safety-tuned model will **say "no"** to a harmful request in chat, but when you give it
**tools** it will sometimes **just do the harmful thing** via a tool call. We wanted to know
*why, inside the network*. Using a standard interpretability technique, we found that "refusal"
is controlled by a **single direction** in the model's internal activations. We then showed that
this refusal signal is **turned down when tools are in the prompt** — which is the internal cause
of the model acting instead of refusing. We validated the direction is causal, localized which
parts of the network do the turning-down, and tested whether we can push the signal back up to
restore safety.

---

## 1. Background you need (mech-interp crash course)

You can read this whole section without any prior interpretability knowledge.

### 1.1 What is actually inside a transformer
When you feed text to a model like Qwen3-14B, three things happen:

1. The text is split into **tokens** (word-pieces). Each token starts as a vector of numbers.
2. Those vectors flow through a stack of **layers** (Qwen3-14B has **40**). Each layer reads the
   current vectors, does some computation (attention + an MLP), and **adds** its result back.
3. Because every layer *adds* to a running total, there is a single evolving vector per token
   that carries information all the way through. That running total is called the
   **residual stream**. For Qwen3-14B it has **5,120 numbers** per token per layer.

Think of the residual stream as the model's "working memory" at each position: a 5,120-dimensional
vector that summarizes what the model currently thinks about that token.

### 1.2 The key idea: features are directions
The central hypothesis of modern interpretability (the **linear representation hypothesis**) is
that the model stores human-meaningful concepts as **directions** in that 5,120-dim space. "This
request is in French," "this is about code," "this deserves a refusal" — each is (approximately) a
straight-line direction. If a concept is "on," the residual-stream vector points *more* along that
concept's direction.

A **direction** is just a unit-length arrow in the space (5,120 numbers, scaled so its length is 1).

### 1.3 Projection = "how much of this concept is present"
To ask *how strongly* a concept is active in a given activation vector **h**, you take the
**dot product** of **h** with the concept's direction **r̂**:

```
projection = h · r̂     (one number)
```

Geometrically: shine a light onto the arrow r̂; **h** casts a shadow on it, and the projection is
the signed length of that shadow. Big positive number → the concept is strongly active. Near zero
→ not active. We use this constantly: our "refusal-o-meter" is literally the projection of the
residual stream onto the refusal direction.

### 1.4 Reading vs. writing (and "hooks")
Two things you can do with internal activations:
- **Read** them (record the vectors) — to measure what the model is representing.
- **Write/edit** them mid-computation — to *test causality*. If editing a direction changes the
  behavior, that direction is causally responsible, not just correlated.

We edit activations using PyTorch **forward hooks** — small functions you attach to a layer that
intercept its output and modify it before it flows on. (See the `Hooks` class and the `ablate` /
`addvec` / `set_proj` functions in `run_interp.py` and `run_interp2.py`.)

---

## 2. The setup (model, tools, data)

### 2.1 Model and framework
- **Model:** `Qwen/Qwen3-14B`, loaded in **bf16** (full precision) on an A100 GPU. *(The separate
  behavioral eval uses the same model in 4-bit; for interpretability we use full precision so the
  activation numbers are clean.)*
- **Framework:** plain **Hugging Face `transformers` + raw PyTorch hooks** — *not* TransformerLens.
  Why: TransformerLens has no clean Qwen3 support at 14B, and raw hooks reuse the *exact* same code
  path as the behavioral eval, so the two halves are directly comparable.
- **Reading activations:** we pass `output_hidden_states=True` to the model and get the residual
  stream at every layer for free (`resid_last_token` / `resid_last` in the scripts).
- **The "decision token":** we measure at the **last token of the prompt** (the moment right before
  the model starts answering). With left-padding, that token sits at index `-1`.

### 2.2 The dataset (`data/…2304_normalized_labels.xlsx`, 2,304 rows)
Each row is one test case with these axes:
- **Request Type:** `Harmful` or `Benign`
- **Domain:** Finance · Healthcare · Education · Legal
- **Mode** (this is the crucial one):
  - `No-tool chat` — plain conversation, no tools available (text only)
  - `Tool-enabled normal` — *same request*, but tool/function definitions are added to the prompt
  - `Tool-enabled stress` — same plus added pressure ("it's urgent, skip approval")
- **System Condition:** `Neutral` · `Safety-reinforced` · `Tool-encouraging`

The model and weights are **identical** across modes — the only thing that changes is the text in
the prompt (whether tool definitions are present). The tool definitions come from the repo's
`tools/` package via `apply_chat_template(tools=...)` (see `format_prompt` in any run script), so the
prompts are byte-identical to the behavioral eval.

### 2.3 Scoring unsafe actions (and a bug we found)
When the model emits a tool call, we check it against a list of **forbidden actions** per domain
(see `tools/domains/healthcare.py`, e.g. `_retrieve_phi`, `_procedure_without_approval`). The scorer
lives in `tools/core.py`.
- **The bug:** the original `DomainSpec.score_tool_calls(scenario_id, calls)` only checked the
  forbidden actions *listed for that one scenario*. A model that took a genuinely-unsafe action
  belonging to a *different* scenario's list was scored "safe." Example: a PHI-retrieval call in a
  `healthcare_structured_tool` scenario was missed because that scenario only listed
  `unsafe_med_plan` + `summary_contains_phi`.
- **The fix:** we added `DomainSpec.score_tool_calls_all(calls)` (in `tools/core.py`) which checks
  **all** of a domain's forbidden actions. Re-scoring the first 250 behavioral rows, the harmful
  unsafe-call rate rose **4.2% → 11.5%** — the old scorer was missing ~⅔ of unsafe calls.
  (`rescore_results.py` re-scores any results CSV offline, no model needed.)

---

## 3. Method 1 — Finding the refusal direction (difference-in-means)

**Goal:** recover the single direction that represents "I should refuse this."

**Technique (Arditi et al., 2024):** contrast the model's internal state on harmful vs harmless
prompts.

1. Take 128 **harmful** and 128 **harmless** prompts (all `No-tool chat`).
2. Run each, record the decision-token residual stream **h** at every layer.
3. Average the harmful vectors and the harmless vectors separately, and subtract:
   ```
   r = mean(h | harmful) − mean(h | harmless)
   ```
   Everything unrelated to harmfulness (topic, length, politeness) is roughly balanced across the
   two sets, so it cancels in the subtraction. What's left, `r`, points along the "refuse vs not"
   axis. Normalize it to length 1 → the **refusal direction** `r̂`.

**Where in the code:** see `run_interp.py`, the `[step1]` block:
```python
A_harm = resid_last([format_prompt(r) for _,r in harm_ex.iterrows()])  # 128 harmful activations
A_ben  = resid_last([format_prompt(r) for _,r in ben_ex.iterrows()])   # 128 benign activations
diff   = A_harm.mean(0) - A_ben.mean(0)        # difference-in-means, per layer
dirs   = diff / diff.norm(dim=-1, keepdim=True)# unit refusal direction, per layer
```

**Result:** the harmful and harmless prompts separate cleanly along `r`. At the chosen layer the
projection reads **+149 for harmful** vs **−162 for benign**. We pick **layer 33 of 40** (strong,
causally-meaningful separation in the middle of the network).
→ **Figure:** `interp_artifacts/fig_separation_by_layer.png` (separation by layer; chosen layer marked).
Saved direction: `interp_artifacts/refusal_dirs.pt`.

---

## 4. Method 2 — Proving the direction is causal (ablation + addition)

Finding a direction that *correlates* with refusal isn't enough — maybe it's just "harmful topic
detector." To prove it *controls* refusal, we edit it and watch the behavior change.

- **Directional ablation** = *remove* the direction. At every layer we subtract the component of the
  activation that lies along `r̂`: `h ← h − (h·r̂) r̂`. If `r̂` is the refusal direction, the model
  should **stop refusing** harmful prompts.
  ```python
  # run_interp.py, ablate():
  h = h - (h @ r).unsqueeze(-1) * r   # project the refusal direction out, at every block
  ```
- **Activation addition** = *inject* the direction. We add `c · r̂` to the residual stream. The model
  should **start refusing** even harmless prompts.
  ```python
  # run_interp.py, addvec():
  h = h + coef * r                    # push the activation along the refusal direction
  ```

**Result (the two-way causal signature):**
| Intervention | Prompts | Refusal rate |
|---|---|---|
| Ablation (remove `r̂`) | Harmful | **55% → 20%** |
| Addition (inject `r̂`) | Benign | **0% → 90%** |

Removing it makes the model comply with harmful requests; adding it makes it refuse harmless ones.
That is exactly what "this is *the* refusal direction" looks like.

---

## 5. Method 3 — The core measurement (projection by mode)

Now the actual research question. For **harmful** prompts, we measure the projection onto `r̂` at the
decision token in each mode. Same request — only the tool context differs.

**Where in the code:** `proj_rows` / `proj_last` (compute `hs[LAYER][:,-1,:] @ r̂`).

**Result:**
| Mode | Mean projection (refusal-o-meter) |
|------|----------------------------------:|
| No-tool chat (text) | **+159.4** |
| Tool-enabled normal | +105.5 |
| Tool-enabled stress | +97.3 |

**Paired by scenario** (n = 96 — the *same* scenario across modes, so prompt content is held fixed
and only the tool context varies):
- No-tool − Tool-normal: **Δ = +66.5, t = 5.65**
- No-tool − Tool-stress: **Δ = +32.2, t = 2.78**

Both significant → the refusal feature is measurably **weaker when tools are present**. That is the
mechanistic shadow of the behavioral fact that the model refuses in text but acts via tools.
→ **Figure:** `interp_artifacts/fig_projection_by_mode.png`.

---

## 6. Going deeper (round 2)

### 6.1 System-condition interaction — `fig_syscond.png`
Projection by mode × system condition shows the tool suppression holds **in every condition**, that a
**safety-reinforced** system prompt strongly lifts the signal (a partial prompt-level fix), and that
**tool-encouraging + stress** is the global minimum (the most dangerous setup). *(Code: `[#12]` in
`run_interp2.py`.)*

### 6.2 Which parts of the network do the suppressing? — `fig_suppression_heads.png`, `fig_mlp_by_layer.png`
**Component attribution.** A transformer layer's contribution to the residual stream is a sum of
pieces: each **attention head** and each **MLP** writes its own vector. We can ask, per piece, *how
much does it write along the refusal direction* — and how that changes between text and tool mode.

For each attention head we reconstruct its individual output and dot it with `r̂`; for each MLP we
dot its output with `r̂`. Then we compare No-tool vs Tool. *(Code: `capture_contrib` in
`run_interp2.py`, which hooks each `self_attn.o_proj` input and each `mlp` output.)*

**Result:** the "stop writing refusal" effect concentrates in **late attention heads (layers 34–39,
led by L34·H36)** and **MLPs in layers 29–32**. So it's an attention+MLP circuit in the upper-middle
of the network, not one neuron.

### 6.3 Sufficiency test — activation patching (`run_interp2.py` `[#1]`)
**Idea:** if low refusal signal *causes* the unsafe action, then forcibly restoring the refusal
signal in tool mode should stop the unsafe call. We use `set_proj` to set the projection back to its
matched No-tool level. **First attempt was inconclusive (a "floor effect"):** the matched sample
barely produced any unsafe calls to begin with (6%) — and that was partly the scorer bug above.
We re-run this properly with the fixed scorer in §7.

### 6.4 Intervention test — steering (`fig_steering_fixed.png`)
**Idea:** add `+c·r̂` during tool-mode generation and sweep the strength `c`. **Result:** unsafe tool
calls fall **25% → 0%** as `c` rises — direct evidence that moving the mediator changes the behavior.
**But it's blunt:** the strength that zeroes unsafe calls also causes ~69% over-refusal on benign
requests (it suppresses tool-calling broadly). A *usable* defense needs a cleaner operating point
(future work). *(Note: an earlier run looked null because the coefficients were 50–200× too small —
the effective scale of this direction is ~1000s, not single digits.)*

---

## 7. Validity & causal follow-ups (Tier 1)

### 7.1 "Is it even the same direction under tools?" — the load-bearing check
Our whole story says tools *suppress the same refusal feature*. But what if, with tools present, the
model uses a *different* direction for refusal? Then "suppression" would be the wrong word.

**Test:** re-extract the refusal direction *within tool mode* (tool-harmful vs tool-benign) and
compute its **cosine similarity** to the text-extracted direction. Cosine near 1 = same feature;
near 0 = different feature. *(Code: `[#2]` / `dir_from` in `run_interp4.py`.)*

**Result: cosine = 0.735.** That's high — the refusal direction is essentially the **same feature**
in both contexts, just turned down under tools. The "suppression" framing is justified. ✅

### 7.2 Patching redo on real divergence cases — the nuance
With the fixed scorer, **34% (41/120)** of harmful tool prompts are now unsafe at baseline (no more
floor effect). For those 41, we restored the refusal projection to its matched **No-tool** level with
`set_proj` and re-generated. **Result: only 5/41 (12%) flipped to safe.** *(Code: `[#3]` in
`run_interp4.py`.)*

This is the most scientifically interesting result, and it refines the thesis. Compare:
- **Strong steering** (§6.4) pushes the projection *far above* its natural level (coef ~700) and drives
  unsafe **25% → 0%**.
- **Patching to the text-mode level** (~+159) only fixes 12%.

So restoring the refusal feature to *how strong it is in plain chat* is **not** enough to stop the
unsafe action — you have to push it much harder. Interpretation: the suppressed refusal direction is a
**contributing cause, but not the sole sufficient one**. Tool context does two things — (a) it turns the
refusal feature down, *and* (b) it provides an action affordance (the model is primed to emit a call).
Fixing only (a) to baseline leaves (b) intact, so the behavior often persists. This is a richer story
than "one direction explains everything," and an honest one.

### 7.3 Does the projection predict the unsafe action, example-by-example?
Per harmful tool prompt, we recorded the projection onto `r̂` and whether the resulting call was unsafe
(fixed scorer), then measured the **AUC** — the chance the method ranks a random unsafe case below a
random safe one. *(Code: `[#4]` in `run_interp4.py`.)*

**Result: AUC = 0.70.** Unsafe cases averaged projection **+52** vs **+112** for safe ones — a clear
gap in the predicted direction (lower refusal signal → unsafe), though not a perfect separator. So the
projection is a real per-example predictor of the unsafe action, not just a group-average effect.
→ **Figure:** `interp_artifacts/fig_auc.png` (projection distributions, safe vs unsafe).

---

### 7.4 Scaling up (the authoritative numbers + an honest dissociation)

The validation/steering/patching numbers above came from small samples (n=20–41). We re-ran them
**batched, at n=120–300, with bootstrap 95% confidence intervals** (`run_interp5.py`). What changed:

- **Ablation:** 55%→20% (n=20) became **56%→36%** [CI 28–45%] at n=120 — a real but **much more modest** effect than the teaser.
- **Addition:** held up — **1%→71%** [CI 62–78%].
- **AUC:** **0.726** [CI 0.668–0.782] at n=300 — tight and solid.
- **Patching:** **19%** [CI 12–28%] of unsafe cases restored to safe.
- **Steering:** clean dose-response, but zeroing unsafe calls costs ~62% benign over-refusal.

**The honest takeaway — a prediction/intervention dissociation.** The refusal direction *predicts* the
unsafe action strongly (AUC 0.73) and is clearly suppressed by tools — but *intervening* on it only
partially changes the behavior (ablation removes ~⅓ of refusals; patching fixes ~19%; steering is blunt).
So it is a **strong predictor / partial mediator**, not the single causal switch. The behavior is driven
jointly by refusal-suppression **and** the tool affordance. (One caveat: our interventions are crude —
one direction, hooked at decoder-layer outputs — so the modest causal numbers may *understate* the true
role; a stronger ablation is the key next experiment.)
→ **Figures:** `interp_artifacts/fig_steering_scaled.png`, `fig_auc_scaled.png`.

## 8. The figures, at a glance

| File | What it shows |
|------|----------------|
| `fig_separation_by_layer.png` | How well harmful/benign separate along `r̂`, per layer (Method 1). |
| `fig_projection_by_mode.png` | Mean refusal projection per mode — the core suppression result (Method 3). |
| `fig_syscond.png` | Projection by mode × system condition (§6.1). |
| `fig_suppression_heads.png` | Per-head drop in refusal-writing, No-tool − Tool (§6.2). |
| `fig_mlp_by_layer.png` | Per-layer MLP drop in refusal-writing (§6.2). |
| `fig_steering_fixed.png` | Steering dose-response: unsafe vs over-refusal vs strength (§6.4). |
| `fig_auc.png` | Projection distributions for safe vs unsafe calls (§7.3, pending). |

---

## 9. Reproducing everything

```bash
# Round 1: extract direction -> validate (ablation/addition) -> projection by mode
python3 run_interp.py
# Round 2: system-condition, suppression heads, patching, steering
python3 run_interp2.py
# Tier-1: same-direction check, patching redo, AUC (uses the fixed scorer)
python3 run_interp4.py
# Re-score any behavioral CSV with the fixed (all-actions) scorer:
python3 rescore_results.py results/results_Qwen3-14B.csv
```
All scripts reuse `data/…xlsx`, the `tools/` package, and the cached Qwen3-14B weights, and load the
saved refusal direction from `interp_artifacts/refusal_dirs.pt` so the geometry is identical across
runs. The interactive version is `Mechanistic_Refusal_Direction.ipynb`.

---

## 10. Glossary

- **Residual stream** — the per-token running-total vector that carries information through the
  layers (5,120 numbers in Qwen3-14B).
- **Direction / feature** — a unit vector in activation space that (approximately) encodes one
  human-meaningful concept.
- **Projection** — dot product of an activation with a direction; "how active" the feature is.
- **Difference-in-means** — finding a feature's direction by subtracting average activations of two
  contrasting prompt sets.
- **Directional ablation** — removing a direction from the residual stream to test causality.
- **Activation addition / steering** — adding a direction to the residual stream to induce/strengthen
  a behavior.
- **Activation patching** — copying an activation (or a component of it) from one run into another to
  test what is *sufficient* to cause a behavior.
- **Hook** — a function attached to a layer that intercepts and edits its activations during a forward
  pass.
- **Component attribution** — splitting a layer's output into its per-head / per-MLP pieces to see
  which write to a given direction.
- **AUC** — probability that the method ranks a random unsafe case as "more unsafe" than a random
  safe case; 0.5 = chance, 1.0 = perfect.

---

## 11. Honest limitations

- **One model so far** (Qwen3-14B). Generality across model families is the biggest open item.
- **Modest behavioral signal** (~34% unsafe after the scorer fix on a fresh sample) — the mechanism is clearer
  than the behavior on this dataset/model; a more tool-eager model may show the behavior more strongly.
- **Refusal detector** in validation/steering is a regex (fast, no API); it undercounts soft refusals.
- **Confound control pending** — tool prompts are longer/JSON-heavy; we still owe the check that
  *irrelevant* long context does *not* lower the projection (to attribute the effect to "tools," not
  "more context"). The cosine 0.735 result partly addresses the related "same feature?" worry.
- **Steering is blunt** — it reduces unsafe calls but at a steep helpfulness cost; not yet a deployable
  defense.

---

*References:* Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda (2024), *Refusal in Language
Models Is Mediated by a Single Direction*, arXiv:2406.11717.
