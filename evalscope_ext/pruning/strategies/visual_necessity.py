"""``visual_necessity`` — keep only MMMU items that actually NEED the image, then
select an encoder-stress probe within them.

Why: if an item is answerable from the question + options alone, an accuracy drop
on it cannot be attributed to the image encoder. The gold filter is MMMU-Pro's
text-only ablation (run N cheap LLMs with the image removed; an item is
"visually necessary" iff models can't answer it blind). That needs a live
endpoint, so this strategy consumes an optional per-item ``text_only_correct``
feature when present (produced by ``evalscope_ext.perturb`` / an offline run) and
otherwise falls back to a metadata heuristic (has image, not open-ended). It then
delegates selection to ``mmmu_encoder_probe``.
"""
from __future__ import annotations

from typing import Dict, List

from ..base import PruningStrategy
from ..registry import get_pruning_strategy, register_pruning_strategy


def _needs_image(f: dict, text_only_key: str, blind_threshold: float) -> bool:
    if str(f.get("question_type")).lower() == "open":
        return False
    if f.get("n_images", 1) in (0, None):
        return False
    to = f.get(text_only_key)
    if to is not None:  # answerable blind => not encoder-dependent
        try:
            return float(to) < blind_threshold
        except (TypeError, ValueError):
            return True
    return True


@register_pruning_strategy("visual_necessity")
class VisualNecessity(PruningStrategy):
    def select(
        self,
        features: List[dict],
        ratio: float,
        *,
        seed: int = 0,
        text_only_key: str = "text_only_correct",
        blind_threshold: float = 0.5,
        **params,
    ) -> Dict[int, float]:
        n = len(features)
        if n == 0:
            return {}
        cand = [i for i in range(n) if _needs_image(features[i], text_only_key, blind_threshold)]
        if not cand:
            cand = list(range(n))
        budget = self.budget(n, ratio)
        sub_ratio = min(1.0, budget / len(cand))
        probe = get_pruning_strategy("mmmu_encoder_probe")
        kept_sub = probe.select([features[i] for i in cand], sub_ratio, seed=seed, **params)
        return {cand[pos]: w for pos, w in kept_sub.items()}
