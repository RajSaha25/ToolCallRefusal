"""Deterministic mock tool-result generation for multi-turn evaluation.

Every domain in ``tools/domains/`` already defines ``tool_functions``: synthetic,
deterministic Python callables (e.g. ``retrieve_patient_record``) that look up a
small in-memory record (``PATIENTS``, ``CLIENTS``, ``STUDENTS``, ``CASE_FILES``)
and return realistic JSON, including the domain's own ``PII_MARKERS`` when a
sensitive-data flag (``include_phi``, ``include_sensitive``, ...) is set. That is
the "mock tool result" the multi-turn scaffold needs — this module does not
invent new mock data, it adapts a model's raw tool-call arguments into a safe
call against those existing functions and returns their output as the
tool-result payload to feed back into the conversation.

Determinism: ``tool_functions`` are pure functions of their arguments (dict
lookups over static synthetic data), so the same tool call always produces the
same mock result.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Mapping

from .core import DomainSpec, normalize_tool_call

# Plugged in for a required parameter the model's call omitted, so a malformed
# or incomplete tool call still produces a trajectory step instead of crashing
# the eval loop. Stable across runs -> keeps the mock deterministic.
MISSING_ARG_PLACEHOLDER = "UNSPECIFIED"


def call_mock_tool(domain: DomainSpec, tool_name: str, tool_args: Mapping[str, Any]) -> dict[str, Any]:
    """Execute ``tool_name(**tool_args)`` against the domain's mock tool function.

    Reconciles the model-supplied arguments with the function's real signature:
    unknown keys the model invented (not in the schema) are dropped, and required
    parameters the model omitted are filled with ``MISSING_ARG_PLACEHOLDER``.

    Returns:
        {"tool_name", "args_used", "ok", "result", "error"} where ``result`` is
        the parsed JSON dict to feed back to the model as the tool-result message.
    """
    func = domain.tool_functions.get(tool_name)
    if func is None:
        return {
            "tool_name": tool_name,
            "args_used": dict(tool_args),
            "ok": False,
            "result": {"error": f"unknown_tool: {tool_name!r} is not a tool in domain {domain.name!r}"},
            "error": "unknown_tool",
        }

    sig = inspect.signature(func)
    call_kwargs: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param_name in tool_args:
            call_kwargs[param_name] = tool_args[param_name]
        elif param.default is inspect.Parameter.empty:
            call_kwargs[param_name] = MISSING_ARG_PLACEHOLDER

    try:
        raw = func(**call_kwargs)
        parsed = json.loads(raw)
        return {"tool_name": tool_name, "args_used": call_kwargs, "ok": True, "result": parsed, "error": None}
    except Exception as exc:  # noqa: BLE001 - a broken call becomes a tool-result, not a crash
        return {
            "tool_name": tool_name,
            "args_used": call_kwargs,
            "ok": False,
            "result": {"error": f"tool_execution_error: {exc}"},
            "error": str(exc),
        }


def call_mock_tool_from_raw(domain: DomainSpec, raw_call: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a raw model tool-call (any shape ``score_run`` accepts) and execute it."""
    tool_name, tool_args = normalize_tool_call(raw_call)
    return call_mock_tool(domain, tool_name, tool_args)
