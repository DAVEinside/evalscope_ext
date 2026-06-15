"""Tiny registry for pruning strategies (mirrors evalscope's own Registry idea)."""
from __future__ import annotations

from typing import Dict, List, Type

from .base import PruningStrategy

_REGISTRY: Dict[str, Type[PruningStrategy]] = {}


def register_pruning_strategy(name: str):
    """Decorator registering a PruningStrategy subclass under ``name``."""

    def wrap(cls: Type[PruningStrategy]) -> Type[PruningStrategy]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"Pruning strategy {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return wrap


def get_pruning_strategy(name: str) -> PruningStrategy:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown pruning strategy {name!r}. Available: {list_strategies()}")
    return _REGISTRY[name]()


def list_strategies() -> List[str]:
    return sorted(_REGISTRY)
