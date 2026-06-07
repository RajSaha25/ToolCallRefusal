"""Domain registry."""

from __future__ import annotations

from .core import DomainSpec

_REGISTRY: dict[str, DomainSpec] = {}
_LOADED = False


def register(domain: DomainSpec) -> DomainSpec:
    _REGISTRY[domain.name] = domain
    return domain


def get_domain(name: str) -> DomainSpec:
    _ensure_loaded()
    if name not in _REGISTRY:
        available = ", ".join(list_domains()) or "(none)"
        raise KeyError(f"Unknown domain {name!r}. Available: {available}")
    return _REGISTRY[name]


def list_domains() -> list[str]:
    _ensure_loaded()
    return sorted(_REGISTRY)


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return

    from .domains import education, finance, healthcare, legal  # noqa: F401

    _LOADED = True
