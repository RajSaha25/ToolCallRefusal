# Multi-turn scaffold: Colab handoff

This scaffold (`run_behavioral_multiturn.py`, `tools/mock_results.py`,
`tests/test_multiturn.py`) was built and unit-tested on a laptop with **no
GPU** and **no `transformers` installed** — every test drives the loop with a
scripted fake model, so none of it has run against a real model yet. This doc
is what to do on Colab to close that gap.

## 1. Pull the branch

```bash
git clone <repo-url> ToolCallRefusal   # or !git clone in a Colab cell
cd ToolCallRefusal
git checkout aniekan/reviewer-response-items
```

If the notebook already has the repo cloned from earlier work, just:

```bash
git fetch origin
git checkout aniekan/reviewer-response-items
git pull
```

## 2. Environment

Same base image as the rest of this repo's headless scripts (see main
`README.md` → Setup):

- Python 3.12, torch 2.8.0+cu128 (preinstalled on the RunPod/Colab image this
  repo targets — do not `pip install torch` over it)
- A GPU runtime. **Mistral-7B-Instruct-v0.3 in bf16 needs ~15 GB** — a T4
  (16 GB) is tight but workable for the smoke test below; an A100 or L4 is
  safer if available. Gemma-3-27B and Llama-3.1-70B need substantially more
  and should NOT be the first thing you try.

```bash
python3.12 -m pip install --break-system-packages -r requirements.txt
export HF_HOME=/workspace/.cache/huggingface   # or /content/drive/... on Colab, so weights survive a restart
export HF_TOKEN=...       # only needed for gated models
export JUDGE_KEY=...      # optional: Anthropic key for the LLM judge on the final turn's refusal label; omit for regex-only labels
```

## 3. Run the 20-prompt validation sample

Start with **Mistral-7B-Instruct-v0.3** — it's the smallest and fastest model
in `MODELS`, and the one this scaffold's dry-run trajectory was modeled on.

```bash
python3.12 run_behavioral_multiturn.py \
  --models mistralai/Mistral-7B-Instruct-v0.3 \
  --sample-n 20 \
  --max-turns 10 \
  --out results/multiturn
```

Equivalent via env vars (matches `run_behavioral_batched.py`'s convention):

```bash
MODELS=mistralai/Mistral-7B-Instruct-v0.3 SAMPLE_N=20 MAX_TURNS=10 \
  python3.12 run_behavioral_multiturn.py
```

This should take a few minutes on a single GPU — multi-turn generation is one
row at a time (not batched across rows the way `run_behavioral_batched.py`
is; see the design-decision note in the PR / report about why), and each row
can involve up to 10 sequential generate() calls if the model keeps calling
tools.

## 4. What to check in the output

`results/multiturn/results_Mistral-7B-Instruct-v0.3_multiturn.csv` should
have 20 rows and all the columns listed in the scaffold spec:
`n_turns`, `termination_reason`, `all_tool_calls_json`, `all_tool_results_json`,
`full_trajectory_json`, `first_refusal_turn`, `first_unsafe_call_turn`,
`turn_of_first_forbidden_call`, `turn_of_first_refusal`,
`delayed_capitulation`, `tool_result_induced_compliance`, plus the same base
columns as `results/results_Mistral-7B-Instruct-v0.3.csv` (single-turn).

Specifically verify:

- `n_turns` is never 0 and never exceeds 10.
- `termination_reason` is one of `final_answer` / `max_turns` / `error`. If
  you see a lot of `error`, check the printed `[ERROR] row <id>: ...`
  messages — that's almost certainly a chat-template incompatibility with
  multi-turn `role: "tool"` messages (see the open question below), not a bug
  in the loop itself (that part is unit-tested).
- `full_trajectory_json` for a few rows, to eyeball whether the model's
  turn-2+ completions look like they're actually responding to the mock tool
  result (vs. ignoring it or repeating turn 1).
- At least a few rows with `n_turns > 1` — if every row terminates at turn 1,
  the model isn't chaining tool calls in this setup and multi-turn dynamics
  aren't being exercised, which is the whole point of this scaffold.
- Any rows where `delayed_capitulation` or `tool_result_induced_compliance`
  is `True` — these are the specific patterns the multi-turn work exists to
  surface, so read a couple of their `full_trajectory_json` values by hand to
  confirm they're real and not an artifact of the tool-call parser
  misreading the model's output.

## 5. Known open question to validate on Colab

`run_behavioral_multiturn.py` appends mock tool results as
`{"role": "tool", "name": ..., "content": ...}` messages and re-renders the
full history through `tokenizer.apply_chat_template(..., tools=tools)` every
turn (see `render_conversation()`). Mistral's template supports this role;
**Command-R and Gemma have not been checked** — if `apply_chat_template`
raises or silently drops the tool message for one of them, rows for that
model will show up as `termination_reason: "error"` (the loop catches
`generate_fn` exceptions per-row rather than crashing the whole run) or as
garbled turn-2 completions. If that happens, that model's turn-2+ rendering
needs a per-model fallback, similar to how `render_prompt()` in
`run_behavioral_batched.py` already tries several `apply_chat_template` kwarg
combinations for the *first* turn.

## 6. Hermes-3-Llama-3.1-8B

`Hermes-3-Llama-3.1-8B` is in `MODELS` but every row for it is tagged
`is_flagged_model=True` in the output CSV — its tool-call markup hasn't been
checked against `parse_tool_calls()` in a multi-turn setting. Run it, but
filter `is_flagged_model` before drawing conclusions from it until someone
has manually verified a handful of its trajectories.

## 7. Single-turn vs. multi-turn comparison

`run_behavioral_batched.py` is untouched, so `results/results_<model>.csv`
(single-turn) and `results/multiturn/results_<model>_multiturn.csv`
(multi-turn) can be diffed row-for-row by `id` once both exist for the same
model.
