"""Tool-safety benchmark scaffold for refusal-transfer experiments."""

from .core import DomainSpec, ForbiddenAction, PromptSet, Scenario, ToolViolation
from .registry import get_domain, list_domains

__all__ = [
    "DomainSpec",
    "ForbiddenAction",
    "PromptSet",
    "Scenario",
    "ToolViolation",
    "get_domain",
    "list_domains",
]
