"""Offline image features for the MMMU encoder probe.

Core path uses ONLY Pillow + numpy (the environment has no cv2/torch/CLIP/
tesseract). Heavy extras (OCR token count, CLIP embedding) are gated behind
``importlib.util.find_spec`` and degrade gracefully so the benchmark still
registers and runs without them.

Rationale: dense-line / small-text / low-entropy images (charts, chemical
structures, tables, medical scans) stress a vision encoder far more than smooth
natural photos. ``laplacian_var`` (fine detail), ``entropy`` (information
density), ``edge_density`` and ``near_monochrome_frac`` (chart/table look)
capture that, and ``ocr_token_count`` captures text-in-image when OCR is present.
"""
from __future__ import annotations

import base64
import binascii
import importlib.util
import io
from typing import Dict, List, Optional

import numpy as np

_HAS_TESSERACT = importlib.util.find_spec("pytesseract") is not None
_FEATURE_CACHE: Dict[str, Dict[str, float]] = {}

CORE_KEYS = ["megapixels", "aspect_ratio", "entropy", "edge_density", "laplacian_var", "near_monochrome_frac"]


def load_image(src) -> Optional["object"]:
    """Decode base64 / data-URI / bytes / PIL image to an RGB PIL image."""
    from PIL import Image

    try:
        if hasattr(src, "convert"):  # already a PIL image
            img = src
        elif isinstance(src, (bytes, bytearray)):
            img = Image.open(io.BytesIO(bytes(src)))
        elif isinstance(src, str):
            s = src.split(",", 1)[1] if src.startswith("data:") else src
            img = Image.open(io.BytesIO(base64.b64decode(s)))
        else:
            return None
        return img.convert("RGB")
    except (binascii.Error, OSError, ValueError, IndexError):
        return None


def _gray_array(img, max_side: int = 256) -> np.ndarray:
    w, h = img.size
    scale = max_side / max(w, h) if max(w, h) > max_side else 1.0
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return np.asarray(img.convert("L"), dtype=float)


def _entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))  # bin width = 1.0
    p = hist / max(1, hist.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _laplacian_var(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    g = gray
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def _edge_density(gray: np.ndarray, thresh: float = 15.0) -> float:
    gx, gy = np.gradient(gray)
    mag = np.hypot(gx, gy)
    return float((mag > thresh).mean())


def _near_monochrome_frac(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=32, range=(0, 255))
    return float(hist.max() / max(1, hist.sum()))


def image_features(img, n_images: int = 1) -> Dict[str, float]:
    """Core PIL+numpy features for a single (already-loaded) PIL image."""
    w, h = img.size
    gray = _gray_array(img)
    feats = {
        "megapixels": round(w * h / 1e6, 4),
        "aspect_ratio": round(w / max(1, h), 3),
        "entropy": round(_entropy(gray), 4),
        "edge_density": round(_edge_density(gray), 4),
        "laplacian_var": round(_laplacian_var(gray), 2),
        "near_monochrome_frac": round(_near_monochrome_frac(gray), 4),
        "n_images": n_images,
    }
    if _HAS_TESSERACT:
        feats["ocr_token_count"] = _ocr_token_count(img)
    return feats


def _ocr_token_count(img) -> int:
    try:
        import pytesseract

        text = pytesseract.image_to_string(img)
        return len(text.split())
    except Exception:  # pragma: no cover - environment-dependent
        return 0


def features_for_images(images: List, cache_key: Optional[str] = None) -> Dict[str, float]:
    """Aggregate features over a sample's images (max over stress-relevant fields)."""
    if cache_key is not None and cache_key in _FEATURE_CACHE:
        return _FEATURE_CACHE[cache_key]
    pils = [im for im in (load_image(s) for s in images) if im is not None]
    if not pils:
        out = {k: 0.0 for k in CORE_KEYS}
        out["n_images"] = 0
        return out
    per = [image_features(im, n_images=len(pils)) for im in pils]
    # stress is dominated by the most demanding image; aggregate by max (entropy by mean)
    agg = {
        "megapixels": max(p["megapixels"] for p in per),
        "aspect_ratio": float(np.mean([p["aspect_ratio"] for p in per])),
        "entropy": float(np.mean([p["entropy"] for p in per])),
        "edge_density": max(p["edge_density"] for p in per),
        "laplacian_var": max(p["laplacian_var"] for p in per),
        "near_monochrome_frac": max(p["near_monochrome_frac"] for p in per),
        "n_images": len(pils),
    }
    if _HAS_TESSERACT:
        agg["ocr_token_count"] = max(p.get("ocr_token_count", 0) for p in per)
    if cache_key is not None:
        _FEATURE_CACHE[cache_key] = agg
    return agg
