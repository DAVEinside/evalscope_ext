"""Build FULL + PRUNED result dirs from the shipped item_stats so the run
contract (`compare_runs`) can be demonstrated WITHOUT a live model. Uses one
shipped model's recorded correctness as the "scores" and applies the real
pruning strategy to select the pruned subset (with prune_weight).

    python -m evalscope_ext.tools.make_demo_runs --bench live_code_bench --model gpt-oss-120b --ratio 0.25
    python -m evalscope_ext.tools.compare_runs --full ./demo_runs/results_full --pruned ./demo_runs/results_pruned
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evalscope_ext.pruning.registry import get_pruning_strategy

HERE = Path(__file__).resolve()
DEFAULT_STATS = HERE.parents[1] / "data" / "item_stats.json"

PARAMS = {
    "live_code_bench": ("stratified_diversity",
                        dict(stratify_keys=["prompt_tokens"], numeric_stratify=["prompt_tokens"], embed_keys=["prompt_tokens"])),
    "aa_lcr": ("stratified_diversity",
               dict(stratify_keys=["n_sources", "input_tokens"], numeric_stratify=["n_sources", "input_tokens"],
                    embed_keys=["n_sources", "input_tokens", "question_len"])),
    # The go/no-go demo measures ACCURACY, so MMMU uses the accuracy-estimator
    # (stratified_diversity), NOT the encoder probe. The probe is a *targeted*
    # measurement (encoder sensitivity via perturbation), not an accuracy estimator,
    # so demoing it through an accuracy gap would be the wrong tool for the metric.
    "mmmu": ("stratified_diversity",
             dict(stratify_keys=["topic_difficulty", "img_type"], numeric_stratify=[], embed_keys=[],
                  midband_key="topic_difficulty", midband_values=["Medium"])),
}


def _row(idx, score, weight=1.0, meta=None):
    r = {"index": idx, "sample_score": {"score": {"value": {"acc": float(score)}, "main_score_name": "acc"}}}
    md = {"prune_weight": weight}
    md.update(meta or {})
    r["metadata"] = md
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="live_code_bench")
    ap.add_argument("--model", default=None, help="which shipped model's scores to use (default: first)")
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--item-stats", default=str(DEFAULT_STATS))
    ap.add_argument("--out", default="./demo_runs")
    args = ap.parse_args()

    stats = json.load(open(args.item_stats, encoding="utf-8"))[args.bench]
    ids = list(stats)
    model = args.model or sorted({m for it in stats.values() for m in it["models"]})[0]
    feats = [stats[i]["features"] for i in ids]
    scores = [stats[i]["models"].get(model) for i in ids]
    keep_meta = ["difficulty", "img_type", "topic_difficulty"]

    out = Path(args.out)
    full_dir = out / "results_full" / "reviews"
    pruned_dir = out / "results_pruned" / "reviews"
    full_dir.mkdir(parents=True, exist_ok=True)
    pruned_dir.mkdir(parents=True, exist_ok=True)

    with open(full_dir / f"{args.bench}.jsonl", "w", encoding="utf-8") as f:
        for r, idx in enumerate(ids):
            if scores[r] is None:
                continue
            meta = {k: feats[r].get(k) for k in keep_meta if feats[r].get(k) is not None}
            f.write(json.dumps(_row(r, scores[r], 1.0, meta)) + "\n")

    strat_name, params = PARAMS[args.bench]
    kept = get_pruning_strategy(strat_name).select(feats, args.ratio, seed=args.seed, **params)
    with open(pruned_dir / f"{args.bench}.jsonl", "w", encoding="utf-8") as f:
        for r, w in sorted(kept.items()):
            if scores[r] is None:
                continue
            meta = {k: feats[r].get(k) for k in keep_meta if feats[r].get(k) is not None}
            f.write(json.dumps(_row(r, scores[r], w, meta)) + "\n")

    print(f"bench={args.bench} model={model} strategy={strat_name} ratio={args.ratio}: "
          f"full={sum(s is not None for s in scores)} pruned={len(kept)} -> {out}")


if __name__ == "__main__":
    main()
