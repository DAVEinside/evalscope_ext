"""``live_code_bench_pruned`` — LiveCodeBench v5 with universal pruning.

Inherits all of LiveCodeBench's loading + sandbox scoring; only the sample set
shrinks. Re-injects ``difficulty``/``platform`` from the raw record (the parent
drops them) so the strategy can stratify on real coding-difficulty axes.
"""
from __future__ import annotations

from typing import Any, Dict

from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter
from evalscope_ext.pruning.mixin import PruningAdapterMixin
from evalscope_ext.pruning.registration import make_pruned_meta


def _input_chars(sample) -> int:
    total = 0
    for m in getattr(sample, "input", []) or []:
        c = getattr(m, "content", "")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            total += sum(len(getattr(p, "text", "") or "") for p in c)
    return total


@register_benchmark(make_pruned_meta("live_code_bench", "live_code_bench_pruned", "stratified_diversity", 0.25))
class LiveCodeBenchPrunedAdapter(PruningAdapterMixin, LiveCodeBenchAdapter):
    PRUNE_DEFAULT_STRATEGY = "stratified_diversity"
    PRUNE_DEFAULT_RATIO = 0.25
    PRUNE_FEATURE_KEYS = ["difficulty", "platform"]
    PRUNE_PARAMS = dict(
        stratify_keys=["difficulty", "platform"],
        embed_keys=["prompt_chars"],
        midband_key="difficulty",
        midband_values=["medium"],
    )

    def record_to_sample(self, record: Dict[str, Any]):
        sample = super().record_to_sample(record)
        md = dict(sample.metadata or {})
        md["difficulty"] = record.get("difficulty")
        md["platform"] = record.get("platform")
        sample.metadata = md
        return sample

    def prune_features(self, sample) -> dict:
        feats = super().prune_features(sample)
        feats["prompt_chars"] = _input_chars(sample)
        return feats
