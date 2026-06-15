"""Base class for pruning strategies.

A strategy operates on a list of *load-time feature dicts* (one per item) and
returns the kept items with per-item weights. It deliberately never sees model
scores, so the same strategy generalizes to an unseen model — this is what keeps
the pruner defensible against the "overfit to the shipped models" rubric trap.
"""
from __future__ import annotations

import abc
from typing import Dict, List


class PruningStrategy(abc.ABC):
    """Select a subset of items from their load-time features."""

    #: set by the registry decorator
    name: str = "base"

    @abc.abstractmethod
    def select(
        self,
        features: List[dict],
        ratio: float,
        *,
        seed: int = 0,
        **params,
    ) -> Dict[int, float]:
        """Return ``{kept_index: weight}``.

        Args:
            features: one feature dict per item (load-time only).
            ratio: target keep fraction in (0, 1].
            seed: determinism for tie-breaking / shuffles.
            **params: strategy-specific knobs supplied by the adapter
                (e.g. ``stratify_keys``, ``embed_keys``).

        The weight is ``stratum_size / selected_in_stratum`` so a weighted mean
        of the kept items is an unbiased estimate of the full-set metric.
        """
        raise NotImplementedError

    @staticmethod
    def budget(n: int, ratio: float) -> int:
        return max(1, min(n, round(ratio * n)))
