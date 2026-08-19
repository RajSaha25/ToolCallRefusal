"""Canonical parsing and normalization for model-emitted tool calls.

The evaluated model families serialize calls in several incompatible wrappers.
This module deliberately uses :class:`json.JSONDecoder` rather than regular
expressions to find JSON values: regex extraction truncates otherwise-valid
calls as soon as an argument contains a nested list or object.

The public representation is intentionally small and compatible with the
existing scorers::

    {"name": "tool_name", "arguments": {"key": "value"}}

Optional ``id`` and ``parse_error`` keys are retained when useful.  Invalid
argument payloads are not silently discarded; they normalize to an empty
mapping and carry a ``parse_error`` so the runtime can block them while metrics
can still count the attempted call.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any


_DECODER = json.JSONDecoder()


def _json_object(value: Any) -> tuple[dict[str, Any], str | None]:
    """Return ``value`` as a JSON object plus a normalization error, if any."""

    if isinstance(value, Mapping):
        return dict(value), None
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            return {}, f"invalid arguments JSON: {exc.msg}"
        if isinstance(decoded, Mapping):
            return dict(decoded), None
        return {}, "tool arguments must decode to a JSON object"
    if value is None:
        return {}, None
    return {}, "tool arguments must be a JSON object"


def normalize_tool_call(call: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize OpenAI, Cohere, and plain tool-call mappings.

    Accepted examples include ``function.name/function.arguments``,
    ``name/arguments``, and ``tool_name/parameters``.  The input mapping and its
    nested argument object are copied, so later caller mutation cannot alter the
    normalized call.
    """

    if not isinstance(call, Mapping):
        raise TypeError("tool call must be a mapping")

    function = call.get("function")
    if isinstance(function, Mapping):
        name = function.get("name", "")
        raw_arguments = function.get("arguments", function.get("parameters", {}))
    else:
        name = call.get("tool_name", call.get("name", ""))
        raw_arguments = call.get("tool_args", call.get("arguments", call.get("parameters", {})))

    arguments, error = _json_object(raw_arguments)
    # A JSON round-trip gives callers a plain, detached JSON-like structure when
    # possible.  Non-JSON values remain for the runtime's serializer/validator to
    # report rather than making parsing itself lossy.
    try:
        arguments = json.loads(json.dumps(arguments, ensure_ascii=False))
    except (TypeError, ValueError):
        arguments = dict(arguments)

    normalized: dict[str, Any] = {
        "name": "" if name is None else str(name),
        "arguments": arguments,
    }
    call_id = call.get("id", call.get("tool_call_id"))
    if call_id is not None:
        normalized["id"] = str(call_id)
    if error:
        normalized["parse_error"] = error
    return normalized


def _looks_like_call(value: Mapping[str, Any]) -> bool:
    if isinstance(value.get("function"), Mapping):
        return bool(value["function"].get("name"))
    return bool(value.get("name") or value.get("tool_name"))


def _calls_from_json(value: Any) -> list[dict[str, Any]]:
    """Extract call-shaped mappings from one decoded JSON value."""

    if isinstance(value, list):
        calls: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping) and _looks_like_call(item):
                calls.append(normalize_tool_call(item))
        return calls

    if not isinstance(value, Mapping):
        return []

    nested = value.get("tool_calls")
    if isinstance(nested, list):
        return _calls_from_json(nested)
    if _looks_like_call(value):
        return [normalize_tool_call(value)]
    return []


def _decode_at(text: str, start: int, stop: int | None = None) -> tuple[Any, int] | None:
    """Decode the first JSON value at or after ``start`` within ``stop``."""

    limit = len(text) if stop is None else min(stop, len(text))
    pos = start
    while pos < limit and text[pos].isspace():
        pos += 1
    if pos >= limit:
        return None
    try:
        value, consumed = _DECODER.raw_decode(text[pos:limit])
    except json.JSONDecodeError:
        return None
    return value, pos + consumed


def _decode_after_markers(
    text: str,
    start_marker: str,
    end_markers: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Decode JSON after every occurrence of a model-specific marker."""

    calls: list[dict[str, Any]] = []
    search_from = 0
    while True:
        marker_at = text.find(start_marker, search_from)
        if marker_at < 0:
            break
        payload_at = marker_at + len(start_marker)
        ends = [text.find(marker, payload_at) for marker in end_markers]
        ends = [pos for pos in ends if pos >= 0]
        stop = min(ends) if ends else None
        decoded = _decode_at(text, payload_at, stop)
        if decoded is not None:
            value, decoded_end = decoded
            calls.extend(_calls_from_json(value))
            search_from = max(decoded_end, payload_at + 1)
        elif stop is not None:
            search_from = stop + 1
        else:
            search_from = payload_at + 1
    return calls


def _decode_xml_blocks(text: str, open_tag: str, close_tag: str) -> list[dict[str, Any]]:
    """Decode JSON in XML-like blocks, including a truncated final block."""

    calls: list[dict[str, Any]] = []
    search_from = 0
    while True:
        open_at = text.find(open_tag, search_from)
        if open_at < 0:
            break
        payload_at = open_at + len(open_tag)
        close_at = text.find(close_tag, payload_at)
        stop = close_at if close_at >= 0 else None
        decoded = _decode_at(text, payload_at, stop)
        if decoded is not None:
            value, decoded_end = decoded
            calls.extend(_calls_from_json(value))
            search_from = close_at + len(close_tag) if close_at >= 0 else decoded_end
        elif close_at >= 0:
            search_from = close_at + len(close_tag)
        else:
            break
    return calls


def _scan_bare_json(text: str) -> list[dict[str, Any]]:
    """Scan prose for complete bare JSON values and retain only call shapes."""

    calls: list[dict[str, Any]] = []
    pos = 0
    while pos < len(text):
        object_at = text.find("{", pos)
        array_at = text.find("[", pos)
        candidates = [p for p in (object_at, array_at) if p >= 0]
        if not candidates:
            break
        start = min(candidates)
        decoded = _decode_at(text, start)
        if decoded is None:
            pos = start + 1
            continue
        value, end = decoded
        found = _calls_from_json(value)
        if found:
            calls.extend(found)
        pos = max(end, start + 1)
    return calls


def _call_parse_errors(calls: Iterable[Mapping[str, Any]], source: str) -> list[str]:
    """Return stable, JSON-safe errors carried by normalized calls."""

    errors = []
    for index, call in enumerate(calls, 1):
        error = call.get("parse_error")
        if error:
            errors.append(f"{source} candidate {index}: {error}")
    return errors


def _bare_malformed_candidate_count(text: str) -> int:
    """Detect one call-like bare JSON value that could not be decoded.

    Arbitrary braces in prose are intentionally ignored.  A failed JSON value is
    considered a tool-call candidate only when its leading fragment contains a
    canonical call key before the next nested JSON value.  Known wrappers are
    diagnosed separately and therefore never enter this fallback.
    """

    if any(
        marker in text
        for marker in (
            "[TOOL_CALLS]",
            "<tool_call>",
            "<|START_ACTION|>",
            "<|python_tag|>",
        )
    ):
        return 0

    pos = 0
    while pos < len(text):
        object_at = text.find("{", pos)
        array_at = text.find("[", pos)
        starts = [value for value in (object_at, array_at) if value >= 0]
        if not starts:
            return 0
        start = min(starts)
        decoded = _decode_at(text, start)
        if decoded is not None:
            _, end = decoded
            pos = max(end, start + 1)
            continue

        nested_object = text.find("{", start + 1)
        nested_array = text.find("[", start + 1)
        boundaries = [value for value in (nested_object, nested_array) if value >= 0]
        boundary = min(boundaries) if boundaries else min(len(text), start + 512)
        leading_fragment = text[start:boundary]
        if any(
            key in leading_fragment
            for key in (
                '"tool_calls"',
                '"function"',
                '"tool_name"',
                '"name"',
            )
        ):
            return 1
        pos = start + 1
    return 0


def _decoded_candidate_count(value: Any, calls: list[dict[str, Any]]) -> int:
    """Count call records represented by one wrapper payload."""

    if isinstance(value, list):
        return max(1, len(value))
    if isinstance(value, Mapping) and isinstance(value.get("tool_calls"), list):
        return max(1, len(value["tool_calls"]))
    return max(1, len(calls))


def _diagnose_wrapped_calls(
    text: str,
    *,
    format_name: str,
    start_marker: str,
    end_markers: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Parse each marked payload while retaining failures from sibling blocks."""

    calls: list[dict[str, Any]] = []
    candidate_count = 0
    errors: list[str] = []
    search_from = 0
    block_index = 0
    while True:
        marker_at = text.find(start_marker, search_from)
        if marker_at < 0:
            break
        block_index += 1
        payload_at = marker_at + len(start_marker)
        ends = [text.find(marker, payload_at) for marker in end_markers]
        ends = [position for position in ends if position >= 0]
        stop = min(ends) if ends else None
        decoded = _decode_at(text, payload_at, stop)
        if decoded is None:
            candidate_count += 1
            errors.append(
                f"{format_name} candidate {block_index}: payload is not decodable JSON containing a tool call"
            )
            search_from = stop + 1 if stop is not None else payload_at + 1
            continue

        value, decoded_end = decoded
        parsed = _calls_from_json(value)
        represented = _decoded_candidate_count(value, parsed)
        candidate_count += represented
        if not parsed:
            errors.append(
                f"{format_name} candidate {block_index}: decoded JSON contains no tool call"
            )
        elif len(parsed) < represented:
            errors.append(
                f"{format_name} candidate {block_index}: parsed {len(parsed)} of {represented} call record(s)"
            )
        errors.extend(
            _call_parse_errors(parsed, f"{format_name} candidate {block_index}")
        )
        calls.extend(parsed)
        search_from = max(decoded_end, payload_at + 1)
    return calls, candidate_count, errors


def parse_tool_calls_with_diagnostics(
    text: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse calls and report whether call-like output was absent or malformed.

    The returned pair is ``(calls, diagnostics)``.  ``diagnostics`` is a plain,
    JSON-serializable mapping with this stable shape::

        {
            "status": "no_candidate" | "parsed" | "malformed_candidate",
            "candidate_count": int,
            "parsed_call_count": int,
            "selected_format": str | None,
            "errors": list[str],
        }

    A normalized call whose argument JSON is malformed remains in ``calls`` (so
    the runtime can block and count the attempt), while the diagnostic status is
    ``malformed_candidate``.  Wrapper precedence is identical to
    :func:`parse_tool_calls`.
    """

    if not isinstance(text, str) or not text.strip():
        return [], {
            "status": "no_candidate",
            "candidate_count": 0,
            "parsed_call_count": 0,
            "selected_format": None,
            "errors": [],
        }

    formats = (
        (
            "mistral_tool_calls",
            "[TOOL_CALLS]",
            (),
        ),
        (
            "xml_tool_call",
            "<tool_call>",
            ("</tool_call>",),
        ),
        (
            "cohere_action",
            "<|START_ACTION|>",
            ("<|END_ACTION|>",),
        ),
        (
            "llama_python_tag",
            "<|python_tag|>",
            ("<|eom_id|>", "<|eot_id|>"),
        ),
    )

    candidate_count = 0
    errors: list[str] = []
    selected_format: str | None = None
    calls: list[dict[str, Any]] = []
    for format_name, marker, end_markers in formats:
        parsed, format_candidates, format_errors = _diagnose_wrapped_calls(
            text,
            format_name=format_name,
            start_marker=marker,
            end_markers=end_markers,
        )
        candidate_count += format_candidates
        errors.extend(format_errors)
        if parsed:
            calls = parsed
            selected_format = format_name
            break

    if not calls:
        calls = _scan_bare_json(text)
        malformed_bare = _bare_malformed_candidate_count(text)
        if calls:
            selected_format = "bare_json"
            candidate_count += len(calls) + malformed_bare
            errors.extend(_call_parse_errors(calls, "bare_json"))
            if malformed_bare:
                errors.append("bare_json: call-like candidate contains invalid JSON")
        else:
            if malformed_bare:
                candidate_count += malformed_bare
                errors.append("bare_json: call-like candidate contains invalid JSON")

    if errors or (candidate_count and not calls):
        status = "malformed_candidate"
    elif calls:
        status = "parsed"
    else:
        status = "no_candidate"
    diagnostics = {
        "status": status,
        "candidate_count": candidate_count,
        "parsed_call_count": len(calls),
        "selected_format": selected_format,
        "errors": errors,
    }
    return calls, diagnostics


def parse_tool_calls(text: str | None) -> list[dict[str, Any]]:
    """Parse all supported model tool-call serializations from ``text``.

    Formats, in precedence order:

    - Mistral ``[TOOL_CALLS]`` followed by JSON;
    - Qwen/Gemma ``<tool_call> ... </tool_call>`` blocks;
    - Cohere/Command-R ``<|START_ACTION|> ... <|END_ACTION|>``;
    - Llama ``<|python_tag|> ... <|eom_id|>``/``<|eot_id|>``;
    - a bare JSON call object/list embedded in prose.

    Wrapper precedence prevents the same JSON from being counted twice by the
    bare fallback.  Multiple blocks within the selected wrapper are preserved.
    """

    calls, _ = parse_tool_calls_with_diagnostics(text)
    return calls


__all__ = [
    "normalize_tool_call",
    "parse_tool_calls",
    "parse_tool_calls_with_diagnostics",
]
