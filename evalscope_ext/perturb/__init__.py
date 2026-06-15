"""Image-perturbation eval for the MMMU encoder probe.

``image_ops`` holds dependency-light (PIL-only) perturbations used to *measure*
encoder sensitivity: a weak encoder loses accuracy sharply under downscale/blur/
JPEG, a strong one degrades gracefully. ``perturb_mixin`` wires this into an
evalscope adapter (emit original + perturbed copies of each selected sample).
"""
from .image_ops import PERTURBATIONS, apply_perturbation, encoder_sensitivity  # noqa: F401

__all__ = ["PERTURBATIONS", "apply_perturbation", "encoder_sensitivity"]
