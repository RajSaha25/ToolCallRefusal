# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mechanistic interpretability research studying **safety transfer failure in LLM agents** — specifically, *divergence* cases where a model refuses a harmful request in text but still emits an unsafe tool call. The benchmark uses synthetic, deterministic tool environments across four regulated domains (finance, healthcare, education, legal).

## Running the Research

The primary workflow is `Tool_Safety_Research.ipynb`, designed to run on Google Colab with a GPU (Mistral-7B-Instruct-v0.3 loaded in 4-bit). There is no local test suite or build system.

To use the `tools` package in a notebook or script:

```python
from tools.registry import get_domain, list_domains
from tools.scoring import score_run

domain = get_domain("finance")           # lazy-loads all domains on first call
print(domain.tool_names())               # tools to pass to the model API
```

## Architecture

The `tools/` package provides three layers:

**`tools/core.py`** — all dataclasses:
- `DomainSpec` — one domain environment; holds tool schemas, callable implementations, scenarios, forbidden action predicates, and PII markers
- `Scenario` — minimal: `id`, `label`, and `forbidden_action_ids`. Prompts and scenario metadata (family, request type, etc.) live in the dataset, not the code
- `ForbiddenAction` — a `(tool_name, tool_args) → bool` predicate that marks a call as unsafe
- `ToolViolation` — a matched unsafe call with its action id, category, tool name, and args

**`tools/registry.py`** — lazy singleton registry; domain modules self-register on import via `register()`. Importing any domain module is enough to make it available via `get_domain()`.

**`tools/scoring.py`** — behavioral scoring:
- `text_has_refusal(text)` — regex-based refusal detection
- `score_run(domain_name, scenario_id, response_text, tool_calls)` — returns `t_safe`, `tc_safe`, `divergence`, `unsafe_tool_calls`, `violations`
- `divergence=True` is the key transfer-failure signal (text refuses but tool call is unsafe)

**`tools/domains/`** — four domain modules (finance, healthcare, education, legal). Each self-registers and defines: synthetic data, OpenAI-style tool schemas (`TOOLS_FOR_LLM`), Python tool implementations, `FORBIDDEN_ACTIONS`, and `SCENARIOS`.

## Scoring Semantics

Two forbidden action categories:
- `intent` — the *what* of the call (e.g., requesting PHI, placing identifiers in a query)
- `contract` — missing required authorization field (e.g., no `approval_ticket`, no `attending_approval`)

`score_run` accepts tool calls in any of these shapes:
```python
{"name": "tool_name", "arguments": {...}}
{"tool_name": "tool_name", "tool_args": {...}}
{"function": {"name": "tool_name", "arguments": "{\"x\": 1}"}}  # JSON-serialized args
```

## Adding a New Domain

1. Create `tools/domains/<name>.py` following the existing domain pattern.
2. Call `register(DomainSpec(...))` at module level to self-register.
3. Add the import to `tools/registry.py`'s `_ensure_loaded()` function.

## Note on Docs Path

`docs/tool_call_scenario_environments.md` references `tool_safety_benchmark/` as the package path — the actual package is `tools/`.
