"""Perturbation probe wiring for evalscope (Part B measurement).

When ``perturb`` is set in dataset-args, every kept sample is expanded into the
ORIGINAL plus degraded copies (downscale / blur / JPEG), tagged in metadata. The
model is then evaluated on all variants and ``compare_runs --perturb`` computes
per-item ``encoder_sensitivity = acc_orig - mean(acc_perturbed)``: large drop ⇒
the answer relied on fine visual signal the encoder must preserve.

MRO note: place this mixin BEFORE ``PruningAdapterMixin`` so the order is
load → prune → perturb (we only perturb the kept probe items).
"""
from __future__ import annotations

import base64
import io
from typing import Callable, List

from evalscope.api.dataset import Dataset, DatasetDict, MemoryDataset, Sample
from evalscope.utils import get_logger

from .image_ops import PERTURBATIONS, apply_perturbation
from ..features.image_features import load_image

logger = get_logger()


def _reencode(original: str, pil) -> str:
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}" if str(original).startswith("data:") else b64


def _base_id(sample: "Sample"):
    """A grouping key that survives MemoryDataset.reindex() (which overwrites .id).
    Prefer the dataset's own stable id (e.g. MMMU 'validation_Accounting_23')."""
    md = sample.metadata or {}
    return md.get("id", sample.id)


def perturb_sample(sample: "Sample", kind: str, new_id: int, base_id=None) -> "Sample":
    """Deep-copy a sample and replace every image with its perturbed version."""
    if base_id is None:
        base_id = _base_id(sample)
    s = sample.model_copy(deep=True)
    content = s.input if isinstance(s.input, list) else []
    for msg in content:
        parts = getattr(msg, "content", None)
        if not isinstance(parts, list):
            continue
        for part in parts:
            img_str = getattr(part, "image", None)
            if img_str is None:
                continue
            pil = load_image(img_str)
            if pil is None:
                continue
            part.image = _reencode(img_str, apply_perturbation(pil, kind))
    s.id = new_id
    s.metadata = {**(s.metadata or {}), "perturb": kind, "perturb_base_id": base_id}
    return s


def expand_with_perturbations(samples: List["Sample"], kinds: List[str], start_id: int = 0) -> List["Sample"]:
    """Return originals (tagged perturb='orig') + one perturbed copy per kind.

    Variants are grouped at compare time by ``perturb_base_id`` (a stable id), not
    by ``.id`` — the latter is overwritten when the dataset is reindexed.
    """
    out: List[Sample] = []
    nid = start_id
    for s in samples:
        base = _base_id(s)
        s.metadata = {**(s.metadata or {}), "perturb": "orig", "perturb_base_id": base}
        out.append(s)
        for kind in kinds:
            out.append(perturb_sample(s, kind, new_id=10_000_000 + nid, base_id=base))
            nid += 1
    return out


class PerturbationProbeMixin:
    PERTURB_DEFAULT: str = ""  # comma list e.g. "downscale,blur,jpeg"; empty = off

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ep = self.extra_params or {}
        raw = ep.get("perturb", self.PERTURB_DEFAULT) or ""
        self.perturb_kinds = [k.strip() for k in str(raw).split(",") if k.strip() in PERTURBATIONS]

    def load_subsets(self, load_func: Callable[[str], Dataset], is_fewshot: bool = False) -> DatasetDict:
        ds = super().load_subsets(load_func, is_fewshot)  # load (+ prune, if PruningAdapterMixin is downstream)
        if is_fewshot or not self.perturb_kinds:
            return ds
        for subset, dataset in list(ds.items()):
            expanded = expand_with_perturbations(list(dataset), self.perturb_kinds)
            pruned = MemoryDataset(samples=expanded, name=getattr(dataset, "name", subset))
            if hasattr(pruned, "reindex"):
                pruned.reindex()
            ds[subset] = pruned
            logger.info(f"[perturb] {getattr(self, 'name', '?')}/{subset}: x{1 + len(self.perturb_kinds)} "
                        f"variants ({', '.join(self.perturb_kinds)})")
        return ds
