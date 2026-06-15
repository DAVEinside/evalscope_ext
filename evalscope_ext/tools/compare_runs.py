"""Compare a FULL evalscope run against a PRUNED run (the Task-2 run contract):

    python -m evalscope_ext.tools.compare_runs --full ./results_full/ --pruned ./results_pruned/

Reports, per benchmark: full vs pruned accuracy (pruned uses ``prune_weight`` for
an unbiased estimate), the gap with 95% CIs (Wilson for full, bootstrap for
pruned), a CI-overlap flag, per-stratum deltas, and go/no-go agreement vs a
threshold. For AA-LCR it prints the LLM-judge-noise caveat. With ``--perturb`` it
instead reads one run and reports per-perturbation accuracy + encoder sensitivity
(the Part B measurement). Tolerant to evalscope's review layout and to the shipped
``Evals`` layout (recursively globs ``*.jsonl``; joins predictions↔reviews on index).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


def _score_value(row: dict) -> Optional[float]:
    score = row.get("sample_score", {}).get("score", row.get("score", {}))
    value = score.get("value") if isinstance(score, dict) else None
    if not isinstance(value, dict):
        return None
    name = score.get("main_score_name")
    for key in ([name] if name else []) + ["acc", "pass"]:
        if key in value:
            try:
                return float(value[key])
            except (TypeError, ValueError):
                return None
    nums = [v for v in value.values() if isinstance(v, (int, float))]
    return float(nums[0]) if nums else None


def _find(row: dict, key: str):
    if key in row:
        return row[key]
    md = row.get("metadata") or row.get("sample_score", {}).get("metadata") or {}
    return md.get(key)


def _bench_name(path: str) -> str:
    base = os.path.basename(path).rsplit(".jsonl", 1)[0]
    base = base.split("__")[0]  # shipped layout: <bench>__<model>
    return base


def load_reviews(root: str) -> Dict[str, Dict[int, dict]]:
    out: Dict[str, Dict[int, dict]] = defaultdict(dict)
    files = glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)
    for f in files:
        # skip prediction files; review files carry the scores we join on
        if (os.sep + "predictions" + os.sep) in f:
            continue
        bench = _bench_name(f)
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sc = _score_value(row)
            if sc is None:
                continue
            idx = row.get("index", len(out[bench]))
            out[bench][idx] = {
                "score": sc,
                "weight": float(_find(row, "prune_weight") or 1.0),
                "difficulty": _find(row, "difficulty"),
                "img_type": _find(row, "img_type"),
                "perturb": _find(row, "perturb"),
                "perturb_base_id": _find(row, "perturb_base_id"),
            }
    return out


def wilson(p: float, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (c - m, c + m)


def boot_ci(scores, weights, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    s, w = np.array(scores, float), np.array(weights, float)
    n = len(s)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, (B, n))
    est = (s[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return (float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)))


def wmean(scores, weights):
    s, w = np.array(scores, float), np.array(weights, float)
    return float((s * w).sum() / w.sum()) if w.sum() else float("nan")


def compare(full_root, pruned_root, threshold, seed):
    full, pruned = load_reviews(full_root), load_reviews(pruned_root)
    benches = sorted(set(full) & set(pruned)) or sorted(set(full) | set(pruned))
    print(f"\n{'benchmark':>22} {'full_acc':>9} {'pruned_acc':>11} {'gap':>7} {'n_full':>7} {'n_kept':>7} {'CI_overlap':>11} {'go/no-go':>9}")
    for b in benches:
        f, p = full.get(b, {}), pruned.get(b, {})
        if not f or not p:
            print(f"{b:>22}  (missing in one run: full={len(f)} pruned={len(p)})")
            continue
        fs = [v["score"] for v in f.values()]
        full_acc = float(np.mean(fs))
        ps = [v["score"] for v in p.values()]
        pw = [v["weight"] for v in p.values()]
        pruned_acc = wmean(ps, pw)
        f_lo, f_hi = wilson(full_acc, len(fs))
        p_lo, p_hi = boot_ci(ps, pw, seed=seed)
        overlap = not (p_hi < f_lo or p_lo > f_hi)
        # 3-way go/no-go: if the decision bar sits INSIDE either run's confidence
        # interval, the model is statistically on the line — don't pretend the
        # binary verdict is meaningful; report TOO-CLOSE. Otherwise AGREE/DIFFER.
        near_bar = (f_lo <= threshold <= f_hi) or (p_lo <= threshold <= p_hi)
        same_side = (pruned_acc >= threshold) == (full_acc >= threshold)
        verdict = "TOO-CLOSE" if near_bar else ("AGREE" if same_side else "DIFFER")
        print(f"{b:>22} {full_acc:>9.3f} {pruned_acc:>11.3f} {pruned_acc-full_acc:>+7.3f} "
              f"{len(fs):>7} {len(ps):>7} {str(overlap):>11} {verdict:>9}")
        # per-stratum deltas
        strat_key = "difficulty" if any(v["difficulty"] for v in f.values()) else (
            "img_type" if any(v["img_type"] for v in f.values()) else None)
        if strat_key:
            def acc_by(d):
                g = defaultdict(list)
                for v in d.values():
                    k = v[strat_key]
                    k = k[0] if isinstance(k, list) and k else k
                    g[str(k)].append(v["score"])
                return {k: float(np.mean(x)) for k, x in g.items()}
            fa, pa = acc_by(f), acc_by(p)
            cells = ", ".join(f"{k}:{fa.get(k,float('nan')):.2f}->{pa.get(k,float('nan')):.2f}" for k in sorted(fa))
            print(f"{'':>22}   by {strat_key}: {cells}")
        if b.startswith("aa_lcr"):
            print(f"{'':>22}   note: AA-LCR is LLM-judge graded; ~part of any gap is judge noise, not sample variance.")


def perturb_report(root, seed):
    runs = load_reviews(root)
    for b, items in runs.items():
        if not any(v["perturb"] for v in items.values()):
            continue
        groups: Dict = defaultdict(dict)
        by_kind = defaultdict(list)
        for v in items.values():
            groups[v["perturb_base_id"]][v["perturb"]] = v["score"]
            if v["perturb"]:
                by_kind[v["perturb"]].append(v["score"])
        print(f"\n{b}: per-perturbation accuracy (n_base={len(groups)})")
        for kind in sorted(by_kind):
            print(f"   {kind:>10}: acc={np.mean(by_kind[kind]):.3f}")
        sens = [g["orig"] - np.mean([g[k] for k in g if k != "orig"])
                for g in groups.values() if "orig" in g and len(g) > 1]
        if sens:
            print(f"   mean encoder_sensitivity (acc_orig - mean_perturbed) = {np.mean(sens):+.3f} "
                  f"(higher => more encoder-bound)")


def main():
    ap = argparse.ArgumentParser(description="Compare full vs pruned evalscope runs.")
    ap.add_argument("--full", help="results dir of the full run")
    ap.add_argument("--pruned", help="results dir of the pruned run")
    ap.add_argument("--threshold", type=float, default=0.5, help="go/no-go accuracy bar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturb", action="store_true", help="encoder-sensitivity report over one run (use --pruned)")
    args = ap.parse_args()
    if args.perturb:
        perturb_report(args.pruned or args.full, args.seed)
    else:
        if not (args.full and args.pruned):
            ap.error("--full and --pruned are required (or use --perturb)")
        compare(args.full, args.pruned, args.threshold, args.seed)


if __name__ == "__main__":
    main()
