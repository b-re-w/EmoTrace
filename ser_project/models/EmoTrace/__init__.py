# coding=utf-8
"""EmoTrace: a causal speech encoder jointly supervised with emotion traces, transcripts and speaker activity."""

from transformers.models.auto import AutoConfig, AutoFeatureExtractor, AutoProcessor

from .configuration_emotrace import (
    BACKBONE_MODEL_TYPES,
    DEFAULT_BACKBONE_MODEL_TYPE,
    EmoTraceConfig,
    get_backbone_config_class,
)
from .modeling_emotrace import (
    EmoTraceForVADPrediction,
    EmoTraceOutput,
    EmoTracePreTrainedModel,
    get_backbone_model_class,
    get_ccc_loss,
)
from .processing_emotrace import (
    EMOTRACE_CHARACTERS,
    EMOTRACE_VOCAL_EVENT_TOKENS,
    EmoTraceFeatureExtractor,
    EmoTraceProcessor,
    EmoTraceProcessorKwargs,
    build_emotrace_ctc_tokenizer,
    build_emotrace_ctc_vocab,
)


def register_emotrace_auto_classes(exist_ok: bool = True) -> None:
    r"""
    Register the EmoTrace classes with the `transformers` auto classes.

    The registration is what lets `EmoTraceProcessor.save_pretrained` and `from_pretrained` round-trip, since the
    processor resolves its feature extractor through `AutoFeatureExtractor`. It is idempotent and runs when this
    package is imported.

    Args:
        exist_ok (`bool`, *optional*, defaults to `True`):
            Whether re-registering an already registered class is allowed.
    """
    AutoConfig.register(EmoTraceConfig.model_type, EmoTraceConfig, exist_ok=exist_ok)
    AutoFeatureExtractor.register(EmoTraceConfig, EmoTraceFeatureExtractor, exist_ok=exist_ok)
    AutoProcessor.register(EmoTraceConfig, EmoTraceProcessor, exist_ok=exist_ok)


register_emotrace_auto_classes()


__all__ = [
    "BACKBONE_MODEL_TYPES",
    "DEFAULT_BACKBONE_MODEL_TYPE",
    "EMOTRACE_CHARACTERS",
    "EMOTRACE_VOCAL_EVENT_TOKENS",
    "EmoTraceConfig",
    "EmoTraceFeatureExtractor",
    "EmoTraceForVADPrediction",
    "EmoTraceOutput",
    "EmoTracePreTrainedModel",
    "EmoTraceProcessor",
    "EmoTraceProcessorKwargs",
    "build_emotrace_ctc_tokenizer",
    "build_emotrace_ctc_vocab",
    "get_backbone_config_class",
    "get_backbone_model_class",
    "get_ccc_loss",
    "register_emotrace_auto_classes",
]
