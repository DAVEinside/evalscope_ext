"""PIL-only image perturbations to probe vision-encoder robustness.

Each perturbation degrades the *image signal* while leaving the question/options
untouched, so a paired accuracy drop isolates the encoder. Chosen to be the ones
the literature flags as encoder-discriminative (resolution loss biggest, blur
medium, compression milder).
"""
from __future__ import annotations

import io
from typing import Callable, Dict, List


def downscale(img, factor: float = 0.5):
    """Lose resolution: shrink to ``factor`` then restore size (detail is gone)."""
    w, h = img.size
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))))
    return small.resize((w, h))


def blur(img, radius: float = 2.0):
    from PIL import ImageFilter

    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def jpeg(img, quality: int = 20):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    from PIL import Image

    return Image.open(buf).convert("RGB")


PERTURBATIONS: Dict[str, Callable] = {
    "downscale": downscale,
    "blur": blur,
    "jpeg": jpeg,
}


def apply_perturbation(img, kind: str):
    if kind in ("orig", None):
        return img
    if kind not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation {kind!r}; choices: {sorted(PERTURBATIONS)}")
    return PERTURBATIONS[kind](img)


def encoder_sensitivity(acc_orig: float, acc_perturbed: List[float]) -> float:
    """Per-item/aggregate encoder sensitivity = clean accuracy minus mean perturbed.

    Larger ⇒ the encoder is the bottleneck; near-zero ⇒ the answer didn't rely on
    fine visual signal (or the encoder is robust).
    """
    if not acc_perturbed:
        return float("nan")
    return float(acc_orig - sum(acc_perturbed) / len(acc_perturbed))
