# coding=utf-8
"""Online prediction of valence, arousal and dominance from conversational speech.

The package is organised in three subpackages, all reachable lazily from here:

- `ser_project.models` holds the causal WavLM backbone and the EmoTrace multi-task model.
- `ser_project.datasets` holds MSP-Conversation corpus access, target construction and batching.
- `ser_project.utils` holds the losses, the metrics and the training entry points.

Only the headline classes are re-exported at this level; everything else keeps its home in the
subpackage that defines it.
"""

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule


__version__ = "0.1.0"

_import_structure = {
    "datasets": [
        "CorpusPaths",
        "DataCollatorForEmoTrace",
        "build_msp_conversation_dataset",
    ],
    "models": [
        "CausalWavLMConfig",
        "CausalWavLMModel",
        "EmoTraceConfig",
        "EmoTraceFeatureExtractor",
        "EmoTraceForVADPrediction",
        "EmoTraceOutput",
        "EmoTraceProcessor",
    ],
    "utils": [],
}

if TYPE_CHECKING:
    from .datasets import CorpusPaths, DataCollatorForEmoTrace, build_msp_conversation_dataset
    from .models import (
        CausalWavLMConfig,
        CausalWavLMModel,
        EmoTraceConfig,
        EmoTraceFeatureExtractor,
        EmoTraceForVADPrediction,
        EmoTraceOutput,
        EmoTraceProcessor,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
