# coding=utf-8
"""CausalWavLM: WavLM-large with every path from a frame to its own future removed.

The two components that read ahead are replaced and nothing else is: the positional convolution pads on the left
only, and self-attention is restricted to a causal sliding window. Parameter names and shapes are WavLM's, so a
released checkpoint loads as it is.
"""

from .configuration_causal_wavlm import CausalWavLMConfig
from .modeling_causal_wavlm import (
    CausalWavLMAttention,
    CausalWavLMEncoder,
    CausalWavLMEncoderLayer,
    CausalWavLMEncoderLayerStableLayerNorm,
    CausalWavLMEncoderStableLayerNorm,
    CausalWavLMModel,
    CausalWavLMPositionalConvCache,
    CausalWavLMPositionalConvEmbedding,
    CausalWavLMPreTrainedModel,
)


__all__ = [
    "CausalWavLMAttention",
    "CausalWavLMConfig",
    "CausalWavLMEncoder",
    "CausalWavLMEncoderLayer",
    "CausalWavLMEncoderLayerStableLayerNorm",
    "CausalWavLMEncoderStableLayerNorm",
    "CausalWavLMModel",
    "CausalWavLMPositionalConvCache",
    "CausalWavLMPositionalConvEmbedding",
    "CausalWavLMPreTrainedModel",
]
