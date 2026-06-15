"""``aa_lcr_pruned`` — AA-LCR (long-context) with universal pruning.

Stratifies on long-context complexity available at load time: number of source
documents (``data_source_urls``) and total ``input_tokens``. Default keep is
higher (0.33) than coding because AA-LCR is LLM-judge graded on only 100 items,
so very aggressive pruning is dominated by judge noise (validated offline).
"""
from __future__ import annotations

from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter
from evalscope_ext.pruning.mixin import PruningAdapterMixin
from evalscope_ext.pruning.registration import make_pruned_meta


@register_benchmark(make_pruned_meta("aa_lcr", "aa_lcr_pruned", "stratified_diversity", 0.33))
class AALCRPrunedAdapter(PruningAdapterMixin, AALCRAdapter):
    PRUNE_DEFAULT_STRATEGY = "stratified_diversity"
    PRUNE_DEFAULT_RATIO = 0.33
    PRUNE_FEATURE_KEYS = ["input_tokens"]
    PRUNE_PARAMS = dict(
        stratify_keys=["n_sources", "input_tokens"],
        numeric_stratify=["n_sources", "input_tokens"],
        embed_keys=["n_sources", "input_tokens", "question_len"],
    )

    def prune_features(self, sample) -> dict:
        md = sample.metadata or {}
        urls = md.get("data_source_urls") or ""
        return {
            "n_sources": len([u for u in str(urls).split(";") if u.strip()]),
            "input_tokens": md.get("input_tokens"),
            "question_len": len(md.get("question") or ""),
        }
