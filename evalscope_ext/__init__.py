"""evalscope_ext — benchmark-pruning extension for evalscope.

A *universal* pruning layer that selects the smallest principled subset of any
benchmark that still answers "is this model good enough?". Built against
modelscope/evalscope @ c14dbaf94e9129f7054ad4a184c2ff0cae2e6a5d.

Importing this package registers the pruning-strategy registry. The pruned
benchmark adapters live in-tree under ``evalscope/benchmarks/<base>_pruned/`` so
the stock ``evalscope eval`` CLI auto-discovers them; they import the reusable
machinery from here.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
