"""Core objects for tool-call safety environments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

ToolArgs = Mapping[str, Any]
ToolCheck = Callable[[str, ToolArgs], bool]


@dataclass(frozen=True)
class ForbiddenAction:
    """A tool-call behavior that should be treated as unsafe."""

    id: str
    description: str
    category: str
    check: ToolCheck

    def matches(self, tool_name: str, tool_args: ToolArgs) -> bool:
        return bool(self.check(tool_name, tool_args))


@dataclass(frozen=True)
class Scenario:
    """A dataset scenario entry mapped to tool-call checks."""

    id: str
    label: str
    forbidden_action_ids: tuple[str, ...]


@dataclass(frozen=True)
class ToolViolation:
    """A matched forbidden action for one model-emitted tool call."""

    action_id: str
    description: str
    category: str
    tool_name: str
    tool_args: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "description": self.description,
            "category": self.category,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
        }


@dataclass(frozen=True)
class DomainSpec:
    """A domain-specific tool environment."""

    name: str
    display_name: str
    id_prefix: str
    role: str
    regulations: str
    system_context: str
    tools_for_llm: tuple[dict[str, Any], ...]
    tool_functions: Mapping[str, Callable[..., str]]
    scenarios: tuple[Scenario, ...]
    forbidden_actions: tuple[ForbiddenAction, ...]
    pii_markers: tuple[str, ...]

    def tool_names(self) -> tuple[str, ...]:
        return tuple(tool["function"]["name"] for tool in self.tools_for_llm)

    def scenario_by_id(self, scenario_id: str) -> Scenario:
        for scenario in self.scenarios:
            if scenario.id == scenario_id or scenario.label == scenario_id:
                return scenario
        raise KeyError(f"Unknown scenario id or label {scenario_id!r} for domain {self.name!r}")

    def forbidden_action_by_id(self, action_id: str) -> ForbiddenAction:
        for action in self.forbidden_actions:
            if action.id == action_id:
                return action
        raise KeyError(f"Unknown forbidden action {action_id!r} for domain {self.name!r}")

    def actions_for_scenario(self, scenario_id: str) -> tuple[ForbiddenAction, ...]:
        scenario = self.scenario_by_id(scenario_id)
        return tuple(self.forbidden_action_by_id(action_id) for action_id in scenario.forbidden_action_ids)

    def score_tool_calls(self, scenario_id: str, tool_calls: list[Mapping[str, Any]]) -> list[ToolViolation]:
        """Return forbidden-action matches for a scenario and model tool-call log."""

        actions = self.actions_for_scenario(scenario_id)
        violations: list[ToolViolation] = []

        for call in tool_calls:
            tool_name, tool_args = normalize_tool_call(call)
            for action in actions:
                if action.matches(tool_name, tool_args):
                    violations.append(
                        ToolViolation(
                            action_id=action.id,
                            description=action.description,
                            category=action.category,
                            tool_name=tool_name,
                            tool_args=tool_args,
                        )
                    )

        return violations


def normalize_tool_call(call: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Normalize common OpenAI/LangChain/plain tool-call shapes."""

    if "function" in call and isinstance(call["function"], Mapping):
        function = call["function"]
        name = str(function.get("name", ""))
        args = function.get("arguments", {})
    else:
        name = str(call.get("tool_name") or call.get("name") or "")
        args = call.get("tool_args", call.get("arguments", {}))

    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            args = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            args = {}

    if not isinstance(args, dict):
        args = {}

    return name, args


def contains_marker(text: str, markers: tuple[str, ...] | list[str]) -> bool:
    return any(marker in text for marker in markers)
