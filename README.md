# From Text Refusal to Tool Safety

Code and data for *From Text Refusal to Tool Safety: A Causal Mechanistic Study of Transfer Failure in LLM Agents*.

A safety-tuned model will usually refuse a harmful request in plain chat. Hand it the same request as a tool call and it often carries it out. We measure how large that gap is across five model families, then open up the network to work out why it happens.

The behavioral side asks whether a verbal refusal transfers to action. We built a 2,304-prompt dataset over four domains (finance, healthcare, education, legal), organized into no-tool, normal tool-enabled, and stress tool-enabled conditions. Tool-call safety is scored by 20 hand-written forbidden-action checks rather than a judge model, so the scoring is deterministic. In every model we tested, refusal in text did not carry over to the tool calls.

The mechanistic side asks why. We extract a single linear "refusal direction" in the residual stream. Its projection predicts whether a given tool call will be unsafe in all five models (AUC 0.69 to 0.88). But patching that direction back into a tool-mode run only partly restores refusal, which tells us the failure is not one clean switch: the refusal signal gets suppressed when tools are present, and the available tools pull the model toward acting. Steering along the direction cuts unsafe calls, but it drives over-refusal on benign prompts, so it is not a free fix.

The takeaway for anyone deploying agents: text safety and tool safety need separate evaluations. A model that refuses in chat is not automatically safe once it can call tools.

## Orchestrated V2 (next-paper environment)

The accepted-paper pipeline is intentionally preserved as the `legacy_v1`
baseline. It is a one-completion function-call probe: calls are parsed and
scored, but the Python tool implementations are not executed and no observation
is returned to the model.

The versioned V2 environment studies **refusal propagation across an agent
workflow**:

```text
Coordinator -> environment work-item tool result -> Domain specialist
            -> Safety reviewer -> sandbox executor
            -> Domain specialist (optional second round) -> Synthesizer
```

The coordinator has no operational tools, the specialist proposes immutable
actions, the reviewer binds a decision to the proposal ID and argument digest,
and only the deterministic sandbox executor can call a domain function. The
runtime validates credentials against episode state, distinguishes proposed,
approved, blocked, and executed actions, and labels tool observations as trusted
or untrusted. Model-visible observations use the native chat-template exchange
(`assistant.tool_calls` followed by `role=tool`), rather than being pasted into a
user message. `Tool-enabled stress` delivers its attack through that untrusted
tool-result channel while keeping the user text byte-identical to the normal condition.
The natural workflow requires a valid coordinator delegation; forced continuation
after refusal/non-delegation is available only as a labeled counterfactual.

The original workbook remains frozen. `build_multi_agent_dataset.py` creates a
2,304-row JSONL scenario manifest with the same balanced factorial design but
with explicit replicates, true A/B paraphrases, matched text across system
conditions, state and authorization fixtures, and trajectory-level golds.

```bash
python3.12 build_multi_agent_dataset.py

# Small CPU-side integrity test (no model download)
python3.12 -m unittest \
  tests/test_dataset_v2.py tests/test_tool_calls.py tests/test_runtime.py \
  tests/test_orchestration.py tests/test_orchestrated_integration.py \
  tests/test_trajectory_metrics.py

# One primary reviewed condition
MODELS=Qwen/Qwen3-14B python3.12 run_orchestrated_eval.py

# Recommended architecture/gate ablation
CONDITIONS=single:audit,delegated:audit,reviewed:reviewer,reviewed:policy \
  MODELS=Qwen/Qwen3-14B python3.12 run_orchestrated_eval.py

# Counterfactual: continue even if the coordinator refuses or does not delegate
COORDINATOR_POLICY=forced CONDITIONS=delegated:audit,reviewed:reviewer \
  MODELS=Qwen/Qwen3-14B python3.12 run_orchestrated_eval.py
```

Each run writes an atomic summary+events JSON bundle per episode, plus a
convenience summary CSV, append-only event JSONL, and run manifest containing the
dataset hash, model revision, precision, token limits,
protocol/code fingerprint, continuation policy, and git commit. Output names carry
the run fingerprint so incompatible resumptions cannot mix. The primary
harmful-case funnel is:

```text
coordinator delegated -> unsafe proposed -> unsafe approved -> unsafe executed
```

Benign tool-mode completion requires the schema-normalized required action
sequence. For the no-tool control, a non-refusal is exported only as
`non_refusal_response_proxy`; it is not mislabeled as task completion without a
separate helpfulness rubric. Explicitly invalid episodes are excluded by the
aggregation helper and reported as an excluded count.

See `docs/multi_agent_orchestration.md` for the research hypotheses, controls,
dataset invariants, metrics, related-work positioning, and mechanistic follow-up.

## Models

All five were run in bf16, no quantization:

- Qwen3-14B (`Qwen/Qwen3-14B`)
- Mistral-7B-Instruct-v0.3 (`mistralai/Mistral-7B-Instruct-v0.3`)
- Command-R-7B (`CohereLabs/c4ai-command-r7b-12-2024`)
- Gemma-3-27B-it (`google/gemma-3-27b-it`)
- Llama-3.1-70B-Instruct (`meta-llama/Llama-3.1-70B-Instruct`)

## Dataset

The eval crosses six factors over 2,304 prompts:

| Factor | Values |
|--------|--------|
| Domain | Finance, Healthcare, Education, Legal |
| Scenario | Direct, Indirect, Roleplay, Structured tool-oriented |
| Mode | No-tool chat, Tool-enabled normal, Tool-enabled stress |
| System condition | Neutral, Safety-reinforced, Tool-encouraging |
| Prompt variant | A, B |
| Request type | Harmful, Benign |

The original construction intended each mode triple to hold the request fixed while varying tool availability and stress framing. A later row-level audit found that portions of the four construction batches also change task wording or semantics across conditions, so use this workbook only to reproduce V1. Orchestrated V2 rebuilds the lattice with enforced prompt/state hashes: no-tool and both tool modes now share byte-identical user text, while stress exists only in the tool-output channel.

Each domain (`tools/domains/`) exposes five function-calling tools in OpenAI schema and five ForbiddenAction predicates, so 20 rule-based checks in all. A predicate is a deterministic Python check over the tool name and its arguments; it fires when a sensitive-data flag is set, a required authorization field is missing, a synthetic PII marker shows up in an argument, or a policy-evasion keyword appears. A tool run counts as safe only if no predicate fires. The authorization checks fire only on absent fields, so a fabricated approval ticket slips through, which makes the reported divergence a conservative lower bound.

Text refusal is scored with a hybrid judge: a regex fast path handles the explicit cases with no API call, and anything ambiguous goes to an LLM judge for a binary yes/no on whether the harmful part was declined. A tool call is never counted as a verbal refusal.

## Measurement conventions

Tool-use propensity metrics (`tools/metrics.py`, output in `results/propensity_metrics.csv`) follow three conventions:

1. **A tool call is anything the parser extracted** into the `tool_calls` column. There is no validity gate: a call to a nonexistent tool or with missing required arguments still counts as a call (the model acted). Well-formedness is reported once as a summary statistic — 98.7% of calls on harmful prompts name a real tool and carry its required arguments, and conditioning rates on validity moves them by at most 1.1 points — but it never filters the data.
2. **No-tool chat rows have no tool calls by definition.** No tools exist in that mode, so JSON the model happens to write in prose is not a call. The runner and the metrics module both enforce this.
3. **Denominators are row-level and form a funnel.** Every generated response counts once (batch repeats are separate observations, never collapsed). Per model, mode, and request type: `any_call` = rows with at least one call / all rows; `unsafe` = rows where a predicate fires / all rows (the paper's original rate); `unsafe_given_call` = rows where a predicate fires / rows with at least one call. The conditional rate is the fair cross-model comparison — the unconditional one rewards models that rarely call tools at all.

Unsafety in these metrics is scored with the current predicates (`score_tool_calls_all`) for every model, regardless of which scorer version produced its results CSV.

## Setup

Tested on an A100-80GB (RunPod), Python 3.12, torch 2.8.0+cu128 from the base image.

```bash
# torch is assumed preinstalled (CUDA build matched to your GPU)
python3.12 -m pip install --break-system-packages -r requirements.txt
```

Weights download from the Hugging Face hub on first use. Point the cache at a persistent disk so a restart does not redownload them:

```bash
export HF_HOME=/workspace/.cache/huggingface
```

A 14B model in bf16 needs roughly 30 GB of GPU memory. For smaller cards, lower the `N_*` sample-size knobs in each notebook's config cell.

## Reproducing

The two numbered notebooks are the model-agnostic version and run on any Hugging Face causal LM whose chat template accepts tool definitions. Set `MODEL_ID` and the rest adapts: the layer is picked from the data, and the steering strengths scale to the model's own activation magnitudes.

1. `01_refusal_direction_and_suppression.ipynb` finds the refusal direction, checks that it is causal (ablation removes refusal, addition induces it), and measures the core effect: the refusal signal is weaker with tools in the prompt. It saves the direction to `interp_artifacts/<model>/refusal_dirs.pt`.
2. `02_causal_followups_and_scaling.ipynb` loads that direction and runs the harder questions with bootstrap confidence intervals: same-direction-under-tools, whether the projection predicts the unsafe call (AUC), patching, steering dose-response, and scaled ablation and addition.

Run `01` first, then `02` with the same `MODEL_ID`. Each model writes to its own `interp_artifacts/<model>/` folder.

New to the methods? `MECH_INTERP_GUIDE.md` explains residual streams, directions, projections, ablation, and patching from the ground up, and ties each idea to the cells here. `MECHANISTIC_INTERP.md` is the terse results write-up.

The behavioral eval lives in `Behavioral_eval.ipynb`, with `run_qwen_eval.py` as a headless single-model runner. Three headless scripts reproduce the Qwen3-14B interp numbers without a notebook server:

```bash
python3.12 run_direction_and_suppression.py   # extract direction, ablation/addition, projection by mode
python3.12 run_scaled_evaluation.py           # batched re-run with bootstrap 95% CIs
python3.12 rescore_results.py results/results_Qwen3-14B.csv   # re-score a saved CSV, no model needed
```

## Layout

```
01_refusal_direction_and_suppression.ipynb   interp part 1 (run first)
02_causal_followups_and_scaling.ipynb         interp part 2 (loads part 1's direction)
Behavioral_eval.ipynb                         behavioral eval, all models
MECH_INTERP_GUIDE.md                          teaching guide (start here if the methods are new)
MECHANISTIC_INTERP.md                         terse results write-up

run_direction_and_suppression.py              headless: extract and validate the direction
run_scaled_evaluation.py                      headless: scaled re-run with bootstrap CIs
run_qwen_eval.py                              headless behavioral eval
rescore_results.py                            re-score a saved behavioral CSV

tools/               scenario and scoring package (domains, forbidden actions, tool schemas)
results/             behavioral output CSVs (Qwen3-14B, Gemma-3-27B, Llama-3.1-70B)
interp_artifacts/    per-model summaries, figures, saved directions
figures/             figures used in the paper
data/                the eval dataset spreadsheet (kept local, not tracked here)
```

## What we did and did not find

A few honest limits, since the effect is easy to overstate:

- The refusal direction is a strong predictor of the unsafe call but only a partial cause. Patching restores refusal 18 to 62 percent of the time depending on the model, not all of it.
- The suppression effect held in four of the five models. Llama-3.1-70B was the exception, where the drop in the refusal signal under tools was not significant.
- The regex refusal classifier used inside the generation experiments undercounts soft refusals, so read the direction of each effect as solid and the exact rates as approximate.
- Steering lowers unsafe calls at a real over-refusal cost. It is not a deployable defense.

## Citation

The paper is under review. Until it is public, cite it as:

```bibtex
@misc{toolsafety2026,
  title  = {From Text Refusal to Tool Safety: A Causal Mechanistic Study of Transfer Failure in LLM Agents},
  author = {},        % add author list on release
  year   = {2026},
  note   = {Under review},
}
```

## Reference

The refusal-direction method builds on Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, and Nanda (2024), *Refusal in Language Models Is Mediated by a Single Direction*, arXiv:2406.11717.
