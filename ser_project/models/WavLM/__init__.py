# coding=utf-8
"""Re-exports of the upstream `transformers` WavLM implementation.

WavLM ships with `transformers`, so nothing is defined here: this package only gives the
upstream classes a home inside `ser_project.models`, next to the causal variant derived from
them, and makes the encoder building blocks importable under a single name.

WavLM has no dedicated feature extractor or processor; the auto classes resolve the
`Wav2Vec2` ones for the `microsoft/wavlm-*` checkpoints, and those are re-exported as well.
"""

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    WavLMConfig,
    WavLMForAudioFrameClassification,
    WavLMForCTC,
    WavLMForSequenceClassification,
    WavLMForXVector,
    WavLMModel,
    WavLMPreTrainedModel,
)
from transformers.models.wavlm.modeling_wavlm import (
    WavLMAdapter,
    WavLMAdapterLayer,
    WavLMAttention,
    WavLMBaseModelOutput,
    WavLMEncoder,
    WavLMEncoderLayer,
    WavLMEncoderLayerStableLayerNorm,
    WavLMEncoderStableLayerNorm,
    WavLMFeatureEncoder,
    WavLMFeatureProjection,
    WavLMFeedForward,
    WavLMPositionalConvEmbedding,
    WavLMSamePadLayer,
)


__all__ = [
    "Wav2Vec2FeatureExtractor",
    "Wav2Vec2Processor",
    "WavLMAdapter",
    "WavLMAdapterLayer",
    "WavLMAttention",
    "WavLMBaseModelOutput",
    "WavLMConfig",
    "WavLMEncoder",
    "WavLMEncoderLayer",
    "WavLMEncoderLayerStableLayerNorm",
    "WavLMEncoderStableLayerNorm",
    "WavLMFeatureEncoder",
    "WavLMFeatureProjection",
    "WavLMFeedForward",
    "WavLMForAudioFrameClassification",
    "WavLMForCTC",
    "WavLMForSequenceClassification",
    "WavLMForXVector",
    "WavLMModel",
    "WavLMPositionalConvEmbedding",
    "WavLMPreTrainedModel",
    "WavLMSamePadLayer",
]
