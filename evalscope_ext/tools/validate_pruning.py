"""Offline validation of the pruner on the shipped 3-model scores — the core
"why this works" evidence, needs NO model endpoint.

For LCB and AA-LCR (3 models each) it checks, per keep-ratio and strategy:
  * accuracy-estimation error: |pruned_acc - full_acc| per model (MAE / max),
    where the pruned estimate is the weighted mean over kept items;
  * ranking preservation across models (Kendall tau / exact order match);
  * go/no-go agreement against a threshold;
  * bootstrap 95% CI width of the estimate;
and compares our ``stratified_diversity`` against the forbidden baselines
(uniform random, top-k hardest, top-k easiest) to show it dominates them.

Selection is model-free (load-time features only), so each model is effectively
an "unseen target" — that is the leave-one-model-out argument with 3 folds.

Run:  python -m evalscope_ext.tools.validate_pruning
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np

from evalscope_ext.pruning.registry import get_pruning_strategy

HERE = Path(__file__).resolve()
DEFAULT_STATS = HERE.parents[1] / "data" / "item_stats.json"

# load-time feature config per benchmark (what the in-framework adapter supplies)
BENCH_PARAMS = {
    "live_code_bench": dict(
        stratify_keys=["prompt_tokens"], numeric_stratify=["prompt_tokens"], embed_keys=["prompt_tokens"]
    ),
    "aa_lcr": dict(
        stratify_keys=["n_sources", "input_tokens"],
        numeric_stratify=["n_sources", "input_tokens"],
        embed_keys=["n_sources", "input_tokens", "question_len"],
    ),
    "mmmu": dict(
        stratify_keys=["topic_difficulty", "img_type"], numeric_stratify=[], embed_keys=[],
        midband_key="topic_difficulty", midband_values=["Medium"],
    ),
}


def wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    if n == 0:
        return float("nan")
    denom = 1 + z * z / n
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return float(margin)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w = weights.sum()
    return float((values * weights).sum() / w) if w else float("nan")


def bootstrap_ci(correct: np.ndarray, weights: np.ndarray, B: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(correct)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(B, n))
    est = (correct[idx] * weights[idx]).sum(axis=1) / weights[idx].sum(axis=1)
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def kendall_tau(a: List[float], b: List[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    c = d = 0
    for i, j in combinations(range(n), 2):
        s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        if s > 0:
            c += 1
        elif s < 0:
            d += 1
    return (c - d) / (c + d) if (c + d) else float("nan")


def load_bench(stats: dict, bench: str):
    items = stats.get(bench, {})
    ids = list(items)
    models = sorted({m for it in items.values() for m in it["models"]})
    feats = [items[i].get("features", {}) for i in ids]
    # correctness matrix [n_items x n_models]; nan where a model is missing
    C = np.full((len(ids), len(models)), np.nan)
    for r, i in enumerate(ids):
        for c, m in enumerate(models):
            if m in items[i]["models"]:
                C[r, c] = items[i]["models"][m]
    prior_frac = np.array([items[i]["frac"] for i in ids])
    return ids, models, feats, C, prior_frac


def select_indices(strategy: str, feats, prior_frac, ratio, seed, params):
    n = len(feats)
    budget = max(1, round(ratio * n))
    if strategy == "random":
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, size=budget, replace=False)
        return {int(i): n / budget for i in keep}
    if strategy in ("topk_hardest", "topk_easiest"):
        order = np.argsort(prior_frac)
        if strategy == "topk_easiest":
            order = order[::-1]
        keep = order[:budget]
        return {int(i): n / budget for i in keep}
    strat = get_pruning_strategy(strategy)
    return strat.select(feats, ratio, seed=seed, **params)


def eval_selection(kept: Dict[int, float], C: np.ndarray, models: List[str], threshold: float):
    idx = np.array(sorted(kept), dtype=int)
    w = np.array([kept[i] for i in idx])
    full = np.nanmean(C, axis=0)
    pruned, ci = [], []
    for c in range(C.shape[1]):
        col = C[idx, c]
        ok = ~np.isnan(col)
        vals, ww = col[ok], w[ok]
        pruned.append(weighted_mean(vals, ww))
        ci.append(bootstrap_ci(vals, ww))
    pruned = np.array(pruned)
    err = np.abs(pruned - full)
    tau = kendall_tau(list(full), list(pruned)) if len(models) > 1 else float("nan")
    gng = np.mean((pruned >= threshold) == (full >= threshold)) if len(models) > 1 else float("nan")
    return dict(full=full, pruned=pruned, mae=float(np.nanmean(err)), maxerr=float(np.nanmax(err)),
                tau=tau, gng=gng, ci_width=float(np.nanmean([c[1] - c[0] for c in ci])), n_kept=len(idx))


def main():
    ap = argparse.ArgumentParser(description="Offline validation of pruning quality.")
    ap.add_argument("--item-stats", default=str(DEFAULT_STATS))
    ap.add_argument("--ratios", default="0.1,0.25,0.33")  # 0.25 = LCB/MMMU default, 0.33 = AA-LCR default
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--benches", default="live_code_bench,aa_lcr,mmmu")
    args = ap.parse_args()

    stats = json.load(open(args.item_stats, encoding="utf-8"))
    ratios = [float(x) for x in args.ratios.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    strategies = ["stratified_diversity", "random", "topk_hardest", "topk_easiest"]
    recommended = {"live_code_bench": 0.25, "aa_lcr": 0.33, "mmmu": 0.25}

    print(
        "How to read this:\n"
        "  MAE      = |pruned accuracy - full accuracy|, averaged over models (LOWER is better).\n"
        "  kendall  = does the subset keep the model RANKING (1.0 = perfect). With only 3 models and\n"
        "             two of them near-tied, one flip drops it to 0.333 -- read it together with MAE.\n"
        "  Compare `stratified_diversity` vs the forbidden baselines (random / topk_*). topk_* are\n"
        "  catastrophic (MAE 0.3-0.7) -- that's why they're banned. The 0.10 rows are deliberately\n"
        "  OVER-aggressive to show the failure mode; '<- recommended' marks each benchmark's default ratio.\n"
        "  MMMU ships only 1 model, so kendall/go-no-go are N/A there (single-model accuracy error only)."
    )

    for bench in args.benches.split(","):
        ids, models, feats, C, prior_frac = load_bench(stats, bench)
        if not ids:
            print(f"\n### {bench}: no data"); continue
        full = np.nanmean(C, axis=0)
        threshold = float(np.median(full)) if len(models) > 1 else 0.5
        print(f"\n### {bench}  (N={len(ids)}, models={models})")
        print(f"    full accuracy: " + ", ".join(f"{m}={a:.3f}" for m, a in zip(models, full)))
        print(f"    {'ratio':>5} {'strategy':>20} {'MAE':>7} {'maxerr':>7} {'kendall':>8} {'go/no-go':>9} {'CIwidth':>8}")
        for ratio in ratios:
            for strat in strategies:
                runs = []
                for seed in seeds:
                    kept = select_indices(strat, feats, prior_frac, ratio, seed, BENCH_PARAMS.get(bench, {}))
                    runs.append(eval_selection(kept, C, models, threshold))
                    if strat not in ("random",):
                        break  # deterministic strategies don't need multi-seed
                def _nm(xs):
                    xs = [x for x in xs if x == x]  # drop nan (e.g. single-model benches)
                    return float(np.mean(xs)) if xs else float("nan")

                mae = _nm([r["mae"] for r in runs])
                mx = _nm([r["maxerr"] for r in runs])
                tau = _nm([r["tau"] for r in runs])
                gng = _nm([r["gng"] for r in runs])
                ciw = _nm([r["ci_width"] for r in runs])
                nkept = runs[0]["n_kept"]
                star = "  <- recommended" if (strat == "stratified_diversity"
                                              and abs(ratio - recommended.get(bench, -1)) < 1e-9) else ""
                print(f"    {ratio:>5.2f} {strat:>20} {mae:>7.4f} {mx:>7.4f} {tau:>8.3f} {gng:>9.2f} {ciw:>8.3f}  (n={nkept}){star}")


if __name__ == "__main__":
    main()
