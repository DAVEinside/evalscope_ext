"""``mmmu_encoder_probe`` — Part B selection that targets image-encoder stress.

Selects a tiny probe from MMMU that concentrates on items whose answer depends
on *perception* (dense detail / OCR / scientific imagery), so a drop in accuracy
points at the encoder rather than generic reasoning. Model-free: it ranks by a
frozen per-img_type stress prior + offline image features (Laplacian variance,
low entropy, near-monochrome, optional OCR), stratifies by (subject, img_type)
for coverage, and diversifies within strata by farthest-point sampling. Pair it
with the perturbation eval (downscale/blur/JPEG) to *measure* encoder sensitivity.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

from ..base import PruningStrategy
from ..registry import register_pruning_strategy
from ..selection import first_of, k_center_greedy, largest_remainder, standardize

_STRESS_PATH = Path(__file__).resolve().parents[2] / "data" / "img_type_stress.json"
# image features that (independent of category) indicate encoder stress
STRESS_FEATURES = ["laplacian_var", "neg_entropy", "near_monochrome_frac", "ocr_token_count"]


@lru_cache(maxsize=1)
def _stress_table() -> dict:
    if _STRESS_PATH.exists():
        return json.loads(_STRESS_PATH.read_text(encoding="utf-8"))
    return {"_default": 0.55, "weights": {}}


def _primary_img_type(v):
    primary = first_of(v)  # handles real lists AND stringified lists ("['Tables']")
    return "Other" if primary is None else str(primary)


def _img_type_stress(v) -> float:
    tbl = _stress_table()
    return float(tbl.get("weights", {}).get(_primary_img_type(v), tbl.get("_default", 0.55)))


@register_pruning_strategy("mmmu_encoder_probe")
class MMMUEncoderProbe(PruningStrategy):
    def select(
        self,
        features: List[dict],
        ratio: float,
        *,
        seed: int = 0,
        subject_key: str = "subject",
        img_type_key: str = "img_type",
        question_type_key: str = "question_type",
        drop_open_ended: bool = True,
        type_weight: float = 1.5,
        **_ignore,
    ) -> Dict[int, float]:
        n = len(features)
        if n == 0:
            return {}

        # 1) candidate filter: drop open-ended (format noise, not perception)
        cand = [
            i for i in range(n)
            if not (drop_open_ended and str(features[i].get(question_type_key)).lower() == "open")
        ]
        if not cand:
            cand = list(range(n))

        # 2) encoder-stress score = category prior + standardized image-feature stress
        cat = np.array([_img_type_stress(features[i].get(img_type_key)) for i in cand])
        feat_cols = []
        for key in STRESS_FEATURES:
            if key == "neg_entropy":
                col = [-float(features[i].get("entropy", 0) or 0) for i in cand]
            else:
                col = [float(features[i].get(key, 0) or 0) for i in cand]
            if any(col):
                feat_cols.append(col)
        feat_stress = standardize(np.array(feat_cols).T).mean(axis=1) if feat_cols else np.zeros(len(cand))
        stress = type_weight * cat + feat_stress

        # 3) stratify by (subject, primary img_type) for coverage of all categories
        strata: Dict[tuple, List[int]] = {}
        for pos, i in enumerate(cand):
            key = (str(features[i].get(subject_key)), _primary_img_type(features[i].get(img_type_key)))
            strata.setdefault(key, []).append(pos)  # positions into `cand`

        # Keep ratio×N items, drawn only from the visually-meaningful candidates.
        budget = min(self.budget(n, ratio), len(cand))
        # Allocate ∝ stratum size × (category-stress)² so the probe concentrates on
        # encoder-stressing categories. largest_remainder seeds 1 per stratum when
        # budget ≥ #strata; below that it keeps the highest-stress strata.
        weights = {k: len(v) * float(np.mean(cat[v])) ** 2 for k, v in strata.items()}
        if budget < len(strata):
            logger.info(f"[mmmu_encoder_probe] budget {budget} < {len(strata)} strata; "
                        f"keeping the highest-stress strata only.")
        alloc = largest_remainder(weights, budget, min_each=1)

        # embedding for within-stratum diversity (image features if present, else stress)
        emb_keys = [k for k in ["laplacian_var", "entropy", "near_monochrome_frac", "edge_density", "megapixels"]
                    if any(features[i].get(k) for i in cand)]
        if emb_keys:
            emb = standardize(np.array([[float(features[i].get(k, 0) or 0) for k in emb_keys] for i in cand]))
        else:
            emb = stress.reshape(-1, 1)

        kept: Dict[int, float] = {}
        for key, positions in strata.items():
            k = min(alloc.get(key, 0), len(positions))
            if k <= 0:
                continue
            # restrict to the most encoder-stressing candidates, then diversify
            ranked = sorted(positions, key=lambda p: stress[p], reverse=True)
            pool = ranked[: max(k * 2, k)]
            local = k_center_greedy(emb[pool], k, seed=seed)
            chosen_pos = [pool[j] for j in local]
            w = len(positions) / len(chosen_pos)
            for p in chosen_pos:
                kept[cand[p]] = w
        return kept
