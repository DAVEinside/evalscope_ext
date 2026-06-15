"""Tests for the universal pruner. Runnable with pytest OR directly:
    python -m evalscope_ext.tests.test_pruning
"""
from __future__ import annotations

import copy

import numpy as np

import evalscope  # noqa: F401  (triggers benchmark auto-discovery / shim registration)
from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope_ext.pruning.mixin import prune_dataset_dict
from evalscope_ext.pruning.registry import get_pruning_strategy, list_strategies


def _synthetic_lcb(n_per=8):
    samples, idx = [], 0
    for diff in ["easy", "medium", "hard"]:
        for plat in ["leetcode", "codeforces", "atcoder"]:
            for j in range(n_per):
                samples.append(Sample(
                    input=f"problem {idx} " + "x" * (50 + idx % 200),
                    target="",
                    id=idx,
                    metadata={"difficulty": diff, "platform": plat},
                ))
                idx += 1
    return samples


def _feat(sample):
    md = sample.metadata or {}
    return {"difficulty": md.get("difficulty"), "platform": md.get("platform"),
            "prompt_chars": len(sample.input) if isinstance(sample.input, str) else 0}


def test_strategies_registered():
    for s in ["stratified_diversity", "mmmu_encoder_probe", "visual_necessity"]:
        assert s in list_strategies(), f"{s} not registered"


def test_pruned_benchmarks_registered():
    for n in ["live_code_bench_pruned", "aa_lcr_pruned", "mmmu_pruned"]:
        meta = BENCHMARK_REGISTRY.get(n)
        assert meta is not None
        assert "pruning_strategy" in (meta.extra_params or {})


def test_prune_dataset_dict_count_and_coverage():
    samples = _synthetic_lcb()
    n = len(samples)
    ds = DatasetDict({"default": MemoryDataset(samples=samples, name="default")})
    params = dict(stratify_keys=["difficulty", "platform"], embed_keys=["prompt_chars"],
                  midband_key="difficulty", midband_values=["medium"])
    out = prune_dataset_dict(copy.deepcopy(ds), strategy_name="stratified_diversity",
                             ratio=0.25, seed=0, feature_fn=_feat, params=params)
    kept = list(out["default"])
    assert abs(len(kept) - round(0.25 * n)) <= 1, f"kept {len(kept)} vs target {round(0.25*n)}"
    # every difficulty represented (coverage), and weights stamped
    diffs = {s.metadata["difficulty"] for s in kept}
    assert diffs == {"easy", "medium", "hard"}, diffs
    assert all("prune_weight" in s.metadata for s in kept)
    # weighted count reconstructs the full size (unbiased)
    wsum = sum(s.metadata["prune_weight"] for s in kept)
    assert abs(wsum - n) <= n * 0.25, f"weight sum {wsum} far from {n}"


def test_determinism():
    samples = _synthetic_lcb()
    ds = DatasetDict({"default": MemoryDataset(samples=samples, name="default")})
    params = dict(stratify_keys=["difficulty", "platform"], embed_keys=["prompt_chars"])
    a = prune_dataset_dict(copy.deepcopy(ds), strategy_name="stratified_diversity", ratio=0.3, seed=7,
                           feature_fn=_feat, params=params)
    b = prune_dataset_dict(copy.deepcopy(ds), strategy_name="stratified_diversity", ratio=0.3, seed=7,
                           feature_fn=_feat, params=params)
    assert [s.id for s in a["default"]] == [s.id for s in b["default"]]


def test_dataset_args_roundtrip():
    # faithful to get_benchmark: deep-copy meta, _update with dataset_args, instantiate.
    meta = copy.deepcopy(BENCHMARK_REGISTRY.get("live_code_bench_pruned"))
    meta._update({"extra_params": {"pruning_strategy": "visual_necessity", "prune_ratio": 0.5}})
    adapter = meta.data_adapter(benchmark_meta=meta, task_config=None)
    assert adapter.pruning_strategy == "visual_necessity"
    assert adapter.prune_ratio == 0.5


def test_image_features_and_perturbations():
    from PIL import Image, ImageDraw
    from evalscope_ext.features.image_features import image_features
    from evalscope_ext.perturb.image_ops import apply_perturbation, encoder_sensitivity, PERTURBATIONS

    chart = Image.new("RGB", (400, 300), "white")
    d = ImageDraw.Draw(chart)
    for x in range(0, 400, 10):
        d.line([(x, 0), (x, 300)], fill="black")
    photo = Image.fromarray((np.add.outer(np.linspace(0, 255, 300), np.linspace(0, 40, 400)) % 255).astype("uint8")).convert("RGB")
    cf, pf = image_features(chart), image_features(photo)
    assert cf["laplacian_var"] > pf["laplacian_var"]          # charts have more fine detail
    assert cf["near_monochrome_frac"] > pf["near_monochrome_frac"]
    for k in PERTURBATIONS:
        assert apply_perturbation(chart, k).size  # produces a valid image
    assert encoder_sensitivity(0.8, [0.6, 0.5]) > 0


def test_perturb_sample_replaces_image():
    import base64
    import io

    from PIL import Image, ImageDraw
    from evalscope.api.messages import ChatMessageUser
    from evalscope.api.messages.content import ContentImage, ContentText
    from evalscope_ext.perturb.perturb_mixin import expand_with_perturbations, perturb_sample

    img = Image.new("RGB", (64, 64), "white")
    dr = ImageDraw.Draw(img)
    for x in range(0, 64, 4):
        dr.line([(x, 0), (x, 64)], fill="black")  # fine detail so blur/downscale alter it
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    sample = Sample(
        input=[ChatMessageUser(content=[ContentText(text="<image 1> what is shown?"), ContentImage(image=b64)])],
        target="A", id=5, metadata={"img_type": ["Tables"]},
    )
    out = perturb_sample(sample, "blur", new_id=999)
    assert out.id == 999 and out.metadata["perturb"] == "blur" and out.metadata["perturb_base_id"] == 5
    new_img = out.input[0].content[1].image
    assert new_img != b64 and new_img.startswith("data:image")  # image actually changed
    expanded = expand_with_perturbations([sample], ["downscale", "blur"])
    kinds = {s.metadata["perturb"] for s in expanded}
    assert kinds == {"orig", "downscale", "blur"} and len(expanded) == 3


def test_largest_remainder_coverage_and_sum():
    from evalscope_ext.pruning.selection import largest_remainder
    # every stratum represented when budget >= #strata; sums to total
    a = largest_remainder({"a": 100, "b": 1, "c": 1, "d": 1}, 4, min_each=1)
    assert sum(a.values()) == 4 and all(v >= 1 for v in a.values()), a
    # budget < #strata: keep the top-`budget` by weight; still sums to total
    b = largest_remainder({"a": 5, "b": 3, "c": 1}, 2, min_each=1)
    assert sum(b.values()) == 2 and b["a"] == 1 and b["b"] == 1 and b["c"] == 0, b
    # proportional leftover goes to the heaviest
    c = largest_remainder({"a": 90, "b": 5, "c": 5}, 12, min_each=1)
    assert sum(c.values()) == 12 and c["a"] > c["b"] >= 1, c


def test_img_type_stringified_list_parsed():
    from evalscope_ext.pruning.strategies.mmmu_encoder_probe import _img_type_stress, _primary_img_type
    # runtime form is a STRINGIFIED list; must map to the real category, not the default
    assert _primary_img_type("['Tables']") == "Tables"
    assert _primary_img_type(["Tables"]) == "Tables"
    assert _img_type_stress("['Chemical Structures']") == _img_type_stress(["Chemical Structures"]) > 0.9
    assert _img_type_stress("['Photographs']") < 0.4  # easy category, low stress


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
    print("ALL TESTS PASSED" if not failed else f"{failed} TEST(S) FAILED")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(_run_all())
