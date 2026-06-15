"""Numpy-only selection primitives shared by strategies (no sklearn/torch)."""
from __future__ import annotations

import ast
from typing import Dict, List, Sequence

import numpy as np


def first_of(v):
    """Primary element of a value that may be a list OR a stringified list.

    MMMU's ``img_type`` arrives at eval time as a stringified list, e.g.
    ``"['Tables']"``; offline it is a real list. Both must normalize the same way
    or the stress-table lookup silently misses and every item gets the default.
    """
    if isinstance(v, (list, tuple)):
        return v[0] if v else None
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return parsed[0] if parsed else None
            except (ValueError, SyntaxError):
                pass
        return v
    return v


def standardize(X: np.ndarray) -> np.ndarray:
    """Z-score columns; zero-variance columns become 0."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def k_center_greedy(X: np.ndarray, k: int, seed: int = 0) -> List[int]:
    """Farthest-point sampling: maximally representative + diverse subset.

    Deterministic given ``seed``. Starts from the point closest to the centroid
    (a stable, non-outlier anchor), then greedily adds the point that is
    farthest from the current selection.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if k >= n:
        return list(range(n))
    if k <= 0:
        return []
    centroid = X.mean(axis=0)
    start = int(np.argmin(((X - centroid) ** 2).sum(axis=1)))
    selected = [start]
    min_d = ((X - X[start]) ** 2).sum(axis=1)
    rng = np.random.default_rng(seed)
    while len(selected) < k:
        far = float(min_d.max())
        # break ties deterministically among the farthest points
        cands = np.flatnonzero(min_d >= far - 1e-12)
        nxt = int(cands[rng.integers(len(cands))]) if len(cands) > 1 else int(cands[0])
        selected.append(nxt)
        d = ((X - X[nxt]) ** 2).sum(axis=1)
        min_d = np.minimum(min_d, d)
    return selected


def tertile_bucket(values: Sequence[float]) -> List[int]:
    """Bucket numeric values into 0/1/2 by terciles (robust to ties)."""
    v = np.asarray([np.nan if x is None else float(x) for x in values], dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return [0] * len(values)
    q1, q2 = np.nanquantile(finite, [1 / 3, 2 / 3])
    # ordinal labels: low=0, mid=1, high=2; non-finite -> mid
    out = []
    for x in v:
        if not np.isfinite(x):
            out.append(1)
        elif x <= q1:
            out.append(0)
        elif x <= q2:
            out.append(1)
        else:
            out.append(2)
    return out


def largest_remainder(weights: Dict, total: int, min_each: int = 1) -> Dict:
    """Apportion ``total`` units across keys ∝ weights (Hamilton's method).

    Seeds ``min_each`` per key first (so every stratum is represented whenever
    the budget allows), then distributes the leftover by largest fractional
    part. When ``total < min_each * n_keys`` the budget can't cover everyone, so
    the ``total`` highest-weight strata each get one unit (best-effort coverage
    of the most informative strata). Always sums to exactly ``total``.
    """
    keys = list(weights)
    if not keys or total <= 0:
        return {k: 0 for k in keys}

    # Not enough budget to give min_each to all -> one unit to the top-`total` strata.
    if min_each * len(keys) > total:
        ranked = sorted(keys, key=lambda k: weights[k], reverse=True)
        return {k: (1 if k in set(ranked[:total]) else 0) for k in keys}

    base = min_each
    alloc = {k: base for k in keys}
    remaining = total - base * len(keys)
    if remaining > 0:
        wsum = sum(weights.values()) or 1.0
        raw = {k: weights[k] / wsum * remaining for k in keys}
        for k in keys:
            alloc[k] += int(np.floor(raw[k]))
        used = sum(alloc.values())
        rema = sorted(keys, key=lambda k: (raw[k] - int(np.floor(raw[k])), weights[k]), reverse=True)
        i = 0
        while used < total and rema:
            alloc[rema[i % len(rema)]] += 1
            used += 1
            i += 1
    return alloc
