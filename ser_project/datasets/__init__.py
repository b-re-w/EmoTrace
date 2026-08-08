# coding=utf-8
"""Corpus access, target construction and batching for the MSP-Conversation experiments.

The public names are the ones each submodule lists in its own `__all__`, and they are imported
lazily: reaching for a corpus path helper does not pull in `torch` (needed by the collator) or
the transcription stack (needed by the pseudo-labeller).
"""

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule
from transformers.utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .base import *
    from .collator import *
    from .msp_conversation import *
    from .pseudo_label import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
