"""Build the offline per-item difficulty/discrimination prior from the shipped
model outputs, used ONLY to (a) weight features and stratum allocation and
(b) validate pruning. It is never consumed at eval-time selection, which keeps
the pruner defensible for an unseen model.

Reads ``Evals/Part 1`` (LiveCodeBench v5, AA-LCR) and ``Evals/MMMU`` predictions
+ reviews (joined on ``index``) and writes ``evalscope_ext/data/item_stats.json``:

    {
      "live_code_bench": {"<index>": {"models": {m: 0/1}, "k": int, "n": int,
                                       "frac": float, "features": {...}}},
      "aa_lcr":          {"<index>": {...}},
      "mmmu":            {"<id>":    {...}}   # keyed by metadata id (subject-safe)
    }

Run:  python -m evalscope_ext.tools.build_item_stats
      python -m evalscope_ext.tools.build_item_stats --evals-dir ../Evals --out evalscope_ext/data/item_stats.json
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

HERE = Path(__file__).resolve()
# build_item_stats.py -> tools -> evalscope_ext -> task2-evalscope -> ai-model-quality-challenge
DEFAULT_EVALS = HERE.parents[3] / "Evals"
DEFAULT_OUT = HERE.parents[1] / "data" / "item_stats.json"


def _read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _score_value(row: Dict[str, Any]) -> Optional[float]:
    """Pull the main scalar score (pass/acc) from a review row."""
    score = row.get("sample_score", {}).get("score", {})
    value = score.get("value")
    if not isinstance(value, dict):
        return None
    name = score.get("main_score_name")
    if name and name in value:
        v = value[name]
    elif "pass" in value:
        v = value["pass"]
    elif "acc" in value:
        v = value["acc"]
    else:
        nums = [x for x in value.values() if isinstance(x, (int, float))]
        v = nums[0] if nums else None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _correct(v: Optional[float]) -> Optional[int]:
    return None if v is None else int(v >= 0.5)


def _parse_img_type(raw) -> list:
    """img_type ships as a stringified list e.g. "['Tables']"."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        parsed = ast.literal_eval(str(raw))
        return [str(x) for x in parsed] if isinstance(parsed, (list, tuple)) else [str(parsed)]
    except (ValueError, SyntaxError):
        return [str(raw)]


def _lcb_features(pred: Dict[str, Any]) -> Dict[str, Any]:
    usage = (pred.get("model_output") or {}).get("usage") or {}
    return {"prompt_tokens": usage.get("input_tokens")}


def _aalcr_features(pred: Dict[str, Any]) -> Dict[str, Any]:
    md = pred.get("metadata") or {}
    urls = md.get("data_source_urls") or ""
    n_sources = len([u for u in str(urls).split(";") if u.strip()])
    q = md.get("question") or ""
    return {"n_sources": n_sources, "input_tokens": md.get("input_tokens"), "question_len": len(q)}


def _mmmu_features(pred: Dict[str, Any]) -> Dict[str, Any]:
    md = pred.get("metadata") or {}
    return {
        "id": md.get("id"),
        "question_type": md.get("question_type"),
        "subfield": md.get("subfield"),
        "img_type": _parse_img_type(md.get("img_type")),
        "topic_difficulty": md.get("topic_difficulty"),
    }


def _aggregate(records: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for key, rec in records.items():
        models = {m: c for m, c in rec["models"].items() if c is not None}
        if not models:
            continue
        n = len(models)
        k = sum(models.values())
        out[str(key)] = {
            "models": models,
            "k": k,
            "n": n,
            "frac": round(k / n, 4),
            "features": rec.get("features", {}),
        }
    return out


def build_part1(evals_dir: Path, bench_prefix: str, feature_fn) -> Dict[str, Any]:
    records: Dict[Any, Dict[str, Any]] = defaultdict(lambda: {"models": {}, "features": {}})
    review_dir = evals_dir / "Part 1" / "reviews"
    pred_dir = evals_dir / "Part 1" / "predictions"
    for rpath in glob.glob(str(review_dir / f"{bench_prefix}__*.jsonl")):
        model = os.path.basename(rpath).split("__", 1)[1].rsplit(".jsonl", 1)[0]
        for row in _read_jsonl(rpath):
            records[row["index"]]["models"][model] = _correct(_score_value(row))
    # features from one prediction file (items shared across models)
    pred_files = sorted(glob.glob(str(pred_dir / f"{bench_prefix}__*.jsonl")))
    if pred_files:
        for pred in _read_jsonl(pred_files[0]):
            idx = pred.get("index")
            if idx in records:
                records[idx]["features"] = feature_fn(pred)
    return _aggregate(records)


def build_mmmu(evals_dir: Path) -> Dict[str, Any]:
    records: Dict[Any, Dict[str, Any]] = defaultdict(lambda: {"models": {}, "features": {}})
    pred_root = evals_dir / "MMMU" / "predictions"
    review_root = evals_dir / "MMMU" / "reviews"
    for model_dir in sorted(glob.glob(str(pred_root / "*"))):
        model = os.path.basename(model_dir)
        for ppath in sorted(glob.glob(os.path.join(model_dir, "*.jsonl"))):
            subject_file = os.path.basename(ppath)
            # build index->id+features for this subject from predictions
            idx_to_id: Dict[Any, str] = {}
            for pred in _read_jsonl(ppath):
                feats = _mmmu_features(pred)
                item_id = feats.get("id") or f"{subject_file}:{pred.get('index')}"
                idx_to_id[pred.get("index")] = item_id
                if not records[item_id]["features"]:
                    feats = dict(feats)
                    feats["subject"] = subject_file.replace("mmmu_", "").replace(".jsonl", "")
                    records[item_id]["features"] = feats
            rpath = os.path.join(str(review_root), model, subject_file)
            if os.path.exists(rpath):
                for row in _read_jsonl(rpath):
                    item_id = idx_to_id.get(row.get("index"))
                    if item_id is not None:
                        records[item_id]["models"][model] = _correct(_score_value(row))
    return _aggregate(records)


def main():
    ap = argparse.ArgumentParser(description="Build offline item-stats prior from shipped Evals.")
    ap.add_argument("--evals-dir", default=str(DEFAULT_EVALS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    evals_dir = Path(args.evals_dir)
    if not evals_dir.exists():
        raise SystemExit(f"Evals dir not found: {evals_dir}")

    stats = {
        "live_code_bench": build_part1(evals_dir, "live_code_bench_v5", _lcb_features),
        "aa_lcr": build_part1(evals_dir, "aa_lcr", _aalcr_features),
        "mmmu": build_mmmu(evals_dir),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f)

    for bench, items in stats.items():
        if not items:
            print(f"  {bench}: (no data)")
            continue
        ks = [v["k"] for v in items.values()]
        n_models = max((v["n"] for v in items.values()), default=0)
        dist = {i: sum(1 for k in ks if k == i) for i in range(n_models + 1)}
        print(f"  {bench}: {len(items)} items, {n_models} models, models-correct dist {dist}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
