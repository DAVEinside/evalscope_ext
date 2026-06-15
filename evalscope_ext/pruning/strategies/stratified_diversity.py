"""``stratified_diversity`` — the primary, universal pruning strategy.

Pipeline (all model-free, runs on load-time features only):
  1. Stratify by a coarse difficulty x category grid (numeric keys -> terciles).
  2. Allocate the keep-budget across strata proportionally, optionally
     over-weighting the *discriminating middle band* via a difficulty label.
  3. Within each stratum, pick maximally representative+diverse items by
     farthest-point sampling on standardized embedding features.
  4. Weight each kept item by stratum_size / selected so the weighted mean is an
     unbiased estimate of the full-set metric.

Same code serves LCB (difficulty x platform), AA-LCR (token x #sources tertiles)
and MMMU (topic_difficulty x img_type) — the adapter only supplies the keys.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..base import PruningStrategy
from ..registry import register_pruning_strategy
from ..selection import first_of, k_center_greedy, largest_remainder, standardize, tertile_bucket


def _stratum_labels(features: List[dict], stratify_keys: List[str], numeric_keys: set) -> List[tuple]:
    cols = {}
    for key in stratify_keys:
        vals = [f.get(key) for f in features]
        if key in numeric_keys:
            cols[key] = [f"{key}={b}" for b in tertile_bucket([_num(v) for v in vals])]
        else:
            cols[key] = [f"{key}={_cat(v)}" for v in vals]
    return [tuple(cols[k][i] for k in stratify_keys) for i in range(len(features))]


def _cat(v):
    primary = first_of(v)  # primary category; handles list + stringified-list (e.g. img_type)
    return "none" if primary is None else str(primary)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _embed_matrix(features: List[dict], embed_keys: List[str]) -> np.ndarray:
    """Build a numeric matrix; impute missing with column mean."""
    cols = []
    for key in embed_keys:
        col = np.array([_num(f.get(key)) if _num(f.get(key)) is not None else np.nan for f in features], float)
        if np.all(np.isnan(col)):
            col[:] = 0.0
        else:
            col[np.isnan(col)] = np.nanmean(col)
        cols.append(col)
    return np.column_stack(cols) if cols else np.zeros((len(features), 1))


@register_pruning_strategy("stratified_diversity")
class StratifiedDiversity(PruningStrategy):
    def select(
        self,
        features: List[dict],
        ratio: float,
        *,
        seed: int = 0,
        stratify_keys: List[str] | None = None,
        numeric_stratify: List[str] | None = None,
        embed_keys: List[str] | None = None,
        midband_key: str | None = None,
        midband_values: List[str] | None = None,
        midband_boost: float = 1.5,
        **_ignore,
    ) -> Dict[int, float]:
        n = len(features)
        if n == 0:
            return {}
        budget = self.budget(n, ratio)
        stratify_keys = stratify_keys or []
        numeric_keys = set(numeric_stratify or [])
        embed_keys = embed_keys or []

        if stratify_keys:
            labels = _stratum_labels(features, stratify_keys, numeric_keys)
        else:
            labels = [("all",)] * n

        groups: Dict[tuple, List[int]] = {}
        for i, lab in enumerate(labels):
            groups.setdefault(lab, []).append(i)

        # stratum weight = size, up-weighted if it sits in the discriminating mid-band
        midvals = {str(v).lower() for v in (midband_values or ["medium"])}
        weights = {}
        for lab, idxs in groups.items():
            w = float(len(idxs))
            if midband_key is not None:
                labs = {str(features[i].get(midband_key)).lower() for i in idxs}
                if labs & midvals:
                    w *= midband_boost
            weights[lab] = w

        alloc = largest_remainder(weights, budget, min_each=1)

        emb = standardize(_embed_matrix(features, embed_keys)) if embed_keys else None
        kept: Dict[int, float] = {}
        for lab, idxs in groups.items():
            k = min(alloc.get(lab, 0), len(idxs))
            if k <= 0:
                continue
            if emb is not None:
                local = k_center_greedy(emb[idxs], k, seed=seed)
                chosen = [idxs[j] for j in local]
            else:
                # deterministic even stride fallback
                step = max(1, len(idxs) // k)
                chosen = idxs[::step][:k]
            w = len(idxs) / len(chosen)
            for c in chosen:
                kept[c] = w
        return kept
