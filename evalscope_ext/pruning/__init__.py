"""Universal benchmark pruning for evalscope.

Public surface:
    register_pruning_strategy, get_pruning_strategy, list_strategies  (registry)
    PruningStrategy                                                   (base class)
    PruningAdapterMixin                                               (the evalscope hook)

Importing this package registers the built-in strategies.
"""
from .registry import get_pruning_strategy, list_strategies, register_pruning_strategy  # noqa: F401
from .base import PruningStrategy  # noqa: F401

# Register built-in strategies (import side-effect).
from .strategies import stratified_diversity  # noqa: F401,E402
from .strategies import mmmu_encoder_probe  # noqa: F401,E402
from .strategies import visual_necessity  # noqa: F401,E402

__all__ = [
    "register_pruning_strategy",
    "get_pruning_strategy",
    "list_strategies",
    "PruningStrategy",
]
