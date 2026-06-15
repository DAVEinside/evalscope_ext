"""Build a ``<base>_pruned`` BenchmarkMeta by cloning the parent benchmark's
metadata (same dataset_id / subset_list / metrics / judge) and adding the
pruning ``extra_params``. Keeps each pruned adapter a tiny shim and guarantees
the pruned run scores identically to the full run."""
from __future__ import annotations

import copy
import dataclasses

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import BENCHMARK_REGISTRY

from .mixin import pruning_extra_params


def make_pruned_meta(base_name: str, pruned_name: str, default_strategy: str, default_ratio: float) -> BenchmarkMeta:
    parent = BENCHMARK_REGISTRY.get(base_name)
    if parent is None:
        raise ValueError(
            f"Parent benchmark {base_name!r} not registered. Import its adapter module before {pruned_name!r}."
        )
    # Deep-copy so the pruned meta shares NO mutable state (subset_list, metric_list,
    # extra_params spec dicts, ...) with the parent in the registry. (data_adapter is a
    # class -> atomic under deepcopy; reset to None for @register_benchmark to fill.)
    base = copy.deepcopy(parent)
    merged_extra = {**(base.extra_params or {}), **pruning_extra_params(default_strategy, default_ratio)}
    return dataclasses.replace(
        base,
        name=pruned_name,
        pretty_name=f"{parent.pretty_name or base_name} (pruned)",
        description=(parent.description or "") + "\n\n*Pruned variant: selects a principled subset via "
        "evalscope_ext (see `pruning_strategy` / `prune_ratio`).*",
        extra_params=merged_extra,
        data_adapter=None,  # set by @register_benchmark
    )
