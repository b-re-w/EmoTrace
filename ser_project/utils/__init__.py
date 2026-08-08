# coding=utf-8
"""Losses, metrics, streaming inference helpers and training entry points.

The public names are the ones each submodule lists in its own `__all__`, and they are imported
lazily so that reading a metric helper does not pull in the training stack.
"""

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule
from transformers.utils.import_utils import define_import_structure


_AMBIGUOUS_NAMES = frozenset({"concordance_correlation_coefficient"})
r"""Names defined by more than one submodule, kept out of the flattened package namespace.

`losses` scores `torch` tensors inside the objective while `evaluate` scores `numpy` arrays for
`Trainer.compute_metrics`. Flattening both would let the directory order decide which one wins,
so callers import the coefficient from the submodule matching the tensor type at hand.
"""


if TYPE_CHECKING:
    from .evaluate import *
    from .losses import *
    from .streaming import *
    from .train import *
else:
    import sys

    _file = globals()["__file__"]
    _import_structure = {
        backends: {module: names - _AMBIGUOUS_NAMES for module, names in modules.items()}
        for backends, modules in define_import_structure(_file).items()
    }
    sys.modules[__name__] = _LazyModule(__name__, _file, _import_structure, module_spec=__spec__)
