"""``PruningAdapterMixin`` — the universal evalscope hook.

Mixed into any ``DefaultDataAdapter`` subclass, it prunes the loaded dataset
*after* samples are built (so metadata/images are available) but *before* prompt
templating, by overriding ``load_subsets``. The benchmark keeps its own loading,
scoring and judge untouched; only the sample set shrinks. A ``<base>_pruned``
adapter is therefore a 3–10 line shim: declare the pruning ``extra_params`` and
which load-time features to feed the strategy.

Reads ``pruning_strategy`` / ``prune_ratio`` / ``prune_seed`` from
``self.extra_params`` (so ``--dataset-args '{"<name>":{"extra_params":{...}}}'``
works with zero core changes). Stamps ``prune_weight`` on each kept sample.

NOTE on ``prune_weight``: the weighted mean is an unbiased estimate of the
full-set metric only for a *representative* strategy (``stratified_diversity``).
The Part B probe strategies (``mmmu_encoder_probe`` / ``visual_necessity``) are
deliberately *targeted* measurements — their kept set over-represents
encoder-stressing items, so their weighted mean is an estimate of the probe
subset, not the full set. Use the perturbation eval, not the accuracy gap, to
read those.
"""
from __future__ import annotations

from typing import Callable, Dict, List

from evalscope.api.dataset import Dataset, DatasetDict, MemoryDataset, Sample
from evalscope.utils import get_logger

from .registry import get_pruning_strategy

logger = get_logger()


class PruningAdapterMixin:
    # Defaults; a pruned adapter overrides these class attributes.
    PRUNE_DEFAULT_STRATEGY: str = "stratified_diversity"
    PRUNE_DEFAULT_RATIO: float = 0.25
    PRUNE_FEATURE_KEYS: List[str] = []          # metadata keys copied into the feature dict
    PRUNE_PARAMS: Dict = {}                      # strategy params (stratify_keys, embed_keys, ...)
    PRUNE_EXTRACT_IMAGES: bool = False           # compute image features from sample.input

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ep = self.extra_params or {}
        self.pruning_strategy = ep.get("pruning_strategy", self.PRUNE_DEFAULT_STRATEGY)
        self.prune_ratio = float(ep.get("prune_ratio", self.PRUNE_DEFAULT_RATIO))
        self.prune_seed = int(ep.get("prune_seed", 0))

    # -- feature extraction (overridable per benchmark) --------------------
    def prune_features(self, sample: "Sample") -> dict:
        """Build a load-time feature dict for one sample (no model scores)."""
        md = sample.metadata or {}
        feats = {k: md.get(k) for k in self.PRUNE_FEATURE_KEYS}
        if self.PRUNE_EXTRACT_IMAGES:
            feats.update(self._image_features(sample))
        return feats

    def _image_features(self, sample: "Sample") -> dict:
        from evalscope_ext.features.image_features import features_for_images

        imgs = []
        content = getattr(sample, "input", None)
        msgs = content if isinstance(content, list) else []
        for m in msgs:
            parts = getattr(m, "content", None)
            if isinstance(parts, list):
                for p in parts:
                    img = getattr(p, "image", None)
                    if img is not None:
                        imgs.append(img)
        # Cache only on a stable id (test 'is not None', not truthiness, so id==0 works).
        _id = getattr(sample, "id", None)
        if _id is None:
            _id = (sample.metadata or {}).get("id")
        cache_key = f"{self.name}:{_id}" if _id is not None else None
        return features_for_images(imgs, cache_key=cache_key)

    # -- the hook ----------------------------------------------------------
    def load_subsets(self, load_func: Callable[[str], Dataset], is_fewshot: bool = False) -> DatasetDict:
        ds = super().load_subsets(load_func, is_fewshot)
        if is_fewshot:
            return ds  # never prune few-shot demonstrations
        return prune_dataset_dict(
            ds,
            strategy_name=self.pruning_strategy,
            ratio=self.prune_ratio,
            seed=self.prune_seed,
            feature_fn=self.prune_features,
            params=self.PRUNE_PARAMS,
            label=self.name,
        )


def prune_dataset_dict(ds, *, strategy_name, ratio, seed, feature_fn, params=None, label="") -> "DatasetDict":
    """Prune every subset of a DatasetDict in place. Extracted from the mixin so
    it is unit-testable on synthetic samples without any dataset download."""
    strategy = get_pruning_strategy(strategy_name)
    params = params or {}
    for subset, dataset in list(ds.items()):
        samples = list(dataset)
        if not samples:
            continue
        features = [feature_fn(s) for s in samples]
        kept = strategy.select(features, ratio, seed=seed, **params)
        new_samples = []
        for i in sorted(kept):
            s = samples[i]
            s.metadata = {**(s.metadata or {}), "prune_weight": round(float(kept[i]), 6)}
            new_samples.append(s)
        pruned = MemoryDataset(samples=new_samples, name=getattr(dataset, "name", subset))
        if hasattr(pruned, "reindex"):
            pruned.reindex()
        ds[subset] = pruned
        logger.info(f"[prune] {label}/{subset}: {len(samples)} -> {len(new_samples)} "
                    f"(strategy={strategy_name}, ratio={ratio})")
    return ds


# shared extra_params spec for every pruned benchmark
def pruning_extra_params(default_strategy: str, default_ratio: float) -> dict:
    return {
        "pruning_strategy": {
            "type": "str",
            "description": "Pruning strategy name (see evalscope_ext.pruning.list_strategies()).",
            "value": default_strategy,
        },
        "prune_ratio": {
            "type": "float",
            "description": "Fraction of items to KEEP, in (0, 1].",
            "value": default_ratio,
        },
        "prune_seed": {
            "type": "int",
            "description": "Random seed for deterministic within-stratum selection.",
            "value": 0,
        },
    }
