"""``mmmu_pruned`` — MMMU with the universal pruner, defaulting to the Part B
image-encoder probe, and an optional image-perturbation expansion.

- ``pruning_strategy`` (default ``mmmu_encoder_probe``): select a probe
  concentrated on encoder-stressing image categories + high-detail images,
  covering all subjects. Other strategies (``stratified_diversity``,
  ``visual_necessity``) also work here.
- ``perturb`` (default off): comma list of ``downscale,blur,jpeg``. When set,
  each probe item is expanded into the original + degraded copies so
  ``compare_runs --perturb`` can measure per-item encoder sensitivity.

MRO: PerturbationProbeMixin before PruningAdapterMixin ⇒ load → prune → perturb.
"""
from __future__ import annotations

from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.mmmu.mmmu_adapter import MMMUAdapter
from evalscope_ext.perturb.perturb_mixin import PerturbationProbeMixin
from evalscope_ext.pruning.mixin import PruningAdapterMixin
from evalscope_ext.pruning.registration import make_pruned_meta


def _subject_from_id(item_id: str) -> str:
    # e.g. 'validation_Art_Theory_5' -> 'Art_Theory'
    parts = str(item_id or "").split("_")
    return "_".join(parts[1:-1]) if len(parts) >= 3 else "Unknown"


_META = make_pruned_meta("mmmu", "mmmu_pruned", "mmmu_encoder_probe", 0.05)
_META.extra_params["perturb"] = {
    "type": "str",
    "description": "Comma list of image perturbations to expand the probe for an encoder-sensitivity "
    "measurement (downscale,blur,jpeg). Empty = off.",
    "value": "",
}


@register_benchmark(_META)
class MMMUPrunedAdapter(PerturbationProbeMixin, PruningAdapterMixin, MMMUAdapter):
    PRUNE_DEFAULT_STRATEGY = "mmmu_encoder_probe"
    PRUNE_DEFAULT_RATIO = 0.05
    PRUNE_EXTRACT_IMAGES = True
    PRUNE_FEATURE_KEYS = ["question_type", "subfield", "img_type", "topic_difficulty"]
    PRUNE_PARAMS = dict(subject_key="subject", img_type_key="img_type", question_type_key="question_type")

    def prune_features(self, sample) -> dict:
        feats = super().prune_features(sample)  # metadata keys + image features
        feats["subject"] = _subject_from_id((sample.metadata or {}).get("id"))
        return feats
