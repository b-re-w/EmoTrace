"""PyTorch CausalWavLM model."""

from collections.abc import Callable

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.wavlm.modeling_wavlm import (
    WavLMAdapter,
    WavLMAttention,
    WavLMBaseModelOutput,
    WavLMEncoder,
    WavLMEncoderLayer,
    WavLMEncoderLayerStableLayerNorm,
    WavLMEncoderStableLayerNorm,
    WavLMFeatureEncoder,
    WavLMFeatureProjection,
    WavLMFeedForward,
    WavLMModel,
    WavLMPositionalConvEmbedding,
    WavLMPreTrainedModel,
)

from .configuration_causal_wavlm import CausalWavLMConfig


class CausalWavLMPositionalConvCache:
    r"""
    Left-context cache for [`CausalWavLMPositionalConvEmbedding`].

    The causal positional convolution needs the `num_conv_pos_embeddings - 1` frames that precede the current chunk.
    When a long recording is processed chunk by chunk, this object carries those frames across calls so that the
    result is identical to running the convolution over the concatenated input. A fresh cache behaves like the zero
    padding used at the start of a recording.
    """

    def __init__(self):
        self.states: torch.Tensor | None = None

    def reset(self) -> None:
        r"""Drop the retained frames, so the next chunk is treated as the start of a new recording."""
        self.states = None

    def update(self, hidden_states: torch.Tensor, left_padding: int) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, hidden_size, chunk_length)`):
                Channel-first hidden states of the current chunk.
            left_padding (`int`):
                Number of context frames the convolution consumes on the left.

        Returns:
            `torch.Tensor` of shape `(batch_size, hidden_size, left_padding + chunk_length)`: the current chunk
            prefixed with the retained frames of the previous chunks.
        """
        if self.states is None:
            self.states = hidden_states.new_zeros(hidden_states.shape[0], hidden_states.shape[1], left_padding)
        elif self.states.shape[-1] != left_padding:
            raise ValueError(
                f"This cache holds {self.states.shape[-1]} context frames but {left_padding} were requested. Call "
                "`reset()` before reusing a cache with a different configuration."
            )
        padded_states = torch.cat([self.states, hidden_states], dim=-1)
        # Sliced from the front rather than with `[-left_padding:]`, which would keep everything when there is no
        # left context at all. The copy releases the concatenated tensor once the caller is done with it.
        self.states = padded_states[:, :, padded_states.shape[-1] - left_padding :].clone()
        return padded_states


class CausalWavLMPositionalConvEmbedding(WavLMPositionalConvEmbedding):
    r"""
    Causal variant of [`WavLMPositionalConvEmbedding`].

    WavLM convolves the hidden states with a depthwise kernel of `num_conv_pos_embeddings` taps that is centred on the
    current frame, so every output frame depends on `num_conv_pos_embeddings // 2` future frames (1.28 s for the
    released checkpoints). Here the input is instead padded with `num_conv_pos_embeddings - 1` frames on the left
    only and the convolution itself runs without padding, which keeps the output length equal to the input length
    while making frame `t` a function of frames `<= t`. The trailing frame that `WavLMSamePadLayer` removes for even
    kernels no longer exists, so that layer is dropped. Parameter names and shapes are untouched, hence WavLM weights
    load as they are.
    """

    def __init__(self, config: CausalWavLMConfig):
        super().__init__(config)
        self.left_padding = config.num_conv_pos_embeddings - 1
        # Padding is applied explicitly in `forward`, on the left side only.
        self.conv.padding = (0,)
        del self.padding

    def forward(
        self, hidden_states: torch.Tensor, padding_cache: CausalWavLMPositionalConvCache | None = None
    ) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Projected features of the current chunk.
            padding_cache (`CausalWavLMPositionalConvCache`, *optional*):
                Left-context cache used when a recording is processed chunk by chunk. Updated in place.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`: the positional embeddings.
        """
        hidden_states = hidden_states.transpose(1, 2)

        if padding_cache is not None:
            hidden_states = padding_cache.update(hidden_states, self.left_padding)
        else:
            hidden_states = nn.functional.pad(hidden_states, (self.left_padding, 0))

        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)

        return hidden_states.transpose(1, 2)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    position_bias: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if position_bias is not None:
        attn_weights = attn_weights + position_bias
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class CausalWavLMAttention(WavLMAttention):
    r"""
    Causal sliding-window variant of [`WavLMAttention`].

    [`WavLMAttention`] routes its computation through `torch.nn.functional.multi_head_attention_forward` and smuggles
    the gated relative position bias in through the `attn_mask` argument, which rules out both key-value caching and
    the fused attention kernels. This class performs the same computation with the existing `q_proj`/`k_proj`/
    `v_proj`/`out_proj` submodules and the standard attention interface, so the bias travels as a first-class
    `position_bias` argument, a [`~cache_utils.Cache`] can be supplied, and the attention pattern is restricted to
    the `config.sliding_window` most recent frames, current frame included.

    Args:
        config ([`CausalWavLMConfig`]):
            Model configuration.
        layer_idx (`int`):
            Index of the layer this module belongs to, used to address the key-value cache.
        has_relative_position_bias (`bool`, *optional*, defaults to `True`):
            Whether this layer owns the bucket embedding table. Only the first layer does; the remaining layers reuse
            the bias it produces.
    """

    def __init__(self, config: CausalWavLMConfig, layer_idx: int, has_relative_position_bias: bool = True):
        super().__init__(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.config = config
        self.layer_idx = layer_idx
        self.sliding_window = config.sliding_window
        self.num_key_value_groups = 1
        self.is_causal = True

    def compute_bias(
        self, query_length: int, key_length: int, query_offset: int = 0, key_offset: int = 0
    ) -> torch.FloatTensor:
        r"""
        Args:
            query_length (`int`):
                Number of query frames.
            key_length (`int`):
                Number of key frames.
            query_offset (`int`, *optional*, defaults to 0):
                Absolute position of the first query frame, i.e. the number of frames already in the cache.
            key_offset (`int`, *optional*, defaults to 0):
                Absolute position of the first key frame, which is non-zero once a sliding cache has evicted frames.

        Returns:
            `torch.FloatTensor` of shape `(num_heads, query_length, key_length)`: the ungated relative position bias.
        """
        device = self.rel_attn_embed.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + query_offset
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :] + key_offset
        relative_position_bucket = self._relative_positions_bucket(memory_position - context_position)
        values = self.rel_attn_embed(relative_position_bucket)
        return values.permute([2, 0, 1])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, query_length, hidden_size)`):
                Input frames.
            attention_mask (`torch.Tensor`, *optional*):
                Prepared 4D causal sliding-window mask, as returned by
                [`~masking_utils.create_sliding_window_causal_mask`].
            position_bias (`torch.Tensor` of shape `(1, num_heads, query_length, key_length)`, *optional*):
                Ungated relative position bias computed by the first layer. Recomputed here when not provided.
            past_key_values (`~cache_utils.Cache`, *optional*):
                Key-value cache, updated in place.

        Returns:
            `tuple`: the attended states, the attention probabilities (`None` unless the eager attention
            implementation is used) and the ungated relative position bias to hand to the next layer.
        """
        batch_size, query_length, _ = hidden_states.size()
        hidden_shape = (batch_size, query_length, self.num_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            # The offsets have to be read before the cache grows, so that they describe the same window as the
            # attention mask, which the encoder builds before the first layer runs.
            query_offset = past_key_values.get_query_offset(self.layer_idx)
            key_length, key_offset = past_key_values.get_mask_sizes(query_length, self.layer_idx)
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        else:
            query_offset, key_offset, key_length = 0, 0, query_length

        if position_bias is None:
            position_bias = self.compute_bias(query_length, key_length, query_offset, key_offset).unsqueeze(0)

        gated_hidden_states = hidden_states.view(hidden_shape).transpose(1, 2)
        relative_position_proj = self.gru_rel_pos_linear(gated_hidden_states)
        relative_position_proj = relative_position_proj.view(batch_size, self.num_heads, query_length, 2, 4).sum(-1)
        gate_a, gate_b = torch.sigmoid(relative_position_proj).chunk(2, dim=-1)
        gate = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0
        gated_position_bias = gate * position_bias

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            position_bias=gated_position_bias,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, query_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, position_bias


class CausalWavLMEncoderLayer(WavLMEncoderLayer):
    r"""Post-layer-norm encoder layer built on [`CausalWavLMAttention`]."""

    def __init__(self, config: CausalWavLMConfig, layer_idx: int, has_relative_position_bias: bool = True):
        # The parent constructor is bypassed so that the non-causal attention is never allocated.
        nn.Module.__init__(self)
        self.attention = CausalWavLMAttention(config, layer_idx, has_relative_position_bias=has_relative_position_bias)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        attn_residual = hidden_states
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states, position_bias)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class CausalWavLMEncoderLayerStableLayerNorm(WavLMEncoderLayerStableLayerNorm):
    r"""Pre-layer-norm encoder layer built on [`CausalWavLMAttention`]. This is the variant WavLM-large uses."""

    def __init__(self, config: CausalWavLMConfig, layer_idx: int, has_relative_position_bias: bool = True):
        # The parent constructor is bypassed so that the non-causal attention is never allocated.
        nn.Module.__init__(self)
        self.attention = CausalWavLMAttention(config, layer_idx, has_relative_position_bias=has_relative_position_bias)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        outputs = (hidden_states, position_bias)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class CausalWavLMEncoderMixin:
    r"""
    Encoder body shared by the two causal encoder stacks.

    Both WavLM encoder variants run the same loop over the same submodule names and differ only in what
    `layer_norm` is applied to: the positional embeddings on the way in for the post-layer-norm architecture, the
    encoder output on the way out for the pre-layer-norm one. Subclasses declare which with `normalizes_output`.
    """

    normalizes_output: bool
    layer_class: type[nn.Module]

    def __init__(self, config: CausalWavLMConfig):
        # The WavLM constructor is bypassed so that the non-causal submodules are never allocated.
        nn.Module.__init__(self)
        self.config = config
        self.pos_conv_embed = CausalWavLMPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [
                self.layer_class(config, layer_idx=i, has_relative_position_bias=(i == 0))
                for i in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        padding_cache: CausalWavLMPositionalConvCache | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> tuple | BaseModelOutput:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, num_frames, hidden_size)`):
                Projected features of the current chunk.
            attention_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
                Frame-level mask, 1 for real frames and 0 for padding.
            past_key_values (`~cache_utils.Cache`, *optional*):
                Key-value cache holding the frames of earlier chunks, updated in place.
            padding_cache (`CausalWavLMPositionalConvCache`, *optional*):
                Left-context cache for the positional convolution, updated in place.

        Returns:
            [`~modeling_outputs.BaseModelOutput`] or `tuple`: the encoded frames.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

        hidden_states = hidden_states + self.pos_conv_embed(hidden_states, padding_cache=padding_cache)
        if not self.normalizes_output:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        causal_mask = create_sliding_window_causal_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)
        position_bias = None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            dropout_probability = torch.rand([])
            skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer or synced_gpus:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_bias=position_bias,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                )
                hidden_states, position_bias = layer_outputs[:2]

            if skip_the_layer:
                layer_outputs = (None, None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        if self.normalizes_output:
            hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class CausalWavLMEncoder(CausalWavLMEncoderMixin, WavLMEncoder):
    r"""Post-layer-norm encoder stack. Layer drop follows [`WavLMEncoder`] and never skips the first layer."""

    normalizes_output = False
    layer_class = CausalWavLMEncoderLayer


class CausalWavLMEncoderStableLayerNorm(CausalWavLMEncoderMixin, WavLMEncoderStableLayerNorm):
    r"""Pre-layer-norm encoder stack. This is the variant WavLM-large uses."""

    normalizes_output = True
    layer_class = CausalWavLMEncoderLayerStableLayerNorm


class CausalWavLMPreTrainedModel(WavLMPreTrainedModel):
    r"""
    Base class for causal WavLM models. Weight initialization, the frame-length helpers and the checkpoint plumbing
    all come from [`WavLMPreTrainedModel`]; what changes is the set of usable attention implementations. Scaled dot
    product attention becomes available because [`CausalWavLMAttention`] goes through the attention interface. Flash
    attention and flex attention stay out of reach: every query needs its own gated relative position bias added to
    the scores, which neither of them accepts.
    """

    config: CausalWavLMConfig
    _supports_sdpa = True
    _supports_flash_attn = False
    _supports_flex_attn = False


class CausalWavLMModel(WavLMModel, CausalWavLMPreTrainedModel):
    r"""
    WavLM-large with every path from a frame to its own future removed, so the representation of frame `t` is a
    function of the input up to time `t` only.

    Two components of [`WavLMModel`] are replaced: the positional convolution becomes
    [`CausalWavLMPositionalConvEmbedding`], and self-attention becomes [`CausalWavLMAttention`], restricted to the
    `config.sliding_window` most recent frames. Everything else, including the gated relative position bias, is the
    stock WavLM computation, and the parameters are exactly WavLM's, so
    `CausalWavLMModel.from_pretrained("microsoft/wavlm-large")` loads a released checkpoint as it is.

    One boundary effect is left in place. The convolutional feature extractor uses a 400-sample window with a
    320-sample stride and no padding, so frame `t` covers samples `[320 * t, 320 * t + 400)` and reaches 5 ms past the
    end of its own 20 ms step. This is inherent to WavLM's front end and is not corrected here.

    `config.feat_extract_norm` must be `"layer"`, which is what every `do_stable_layer_norm=True` checkpoint uses.
    The `"group"` feature encoder normalizes each channel over the whole time axis and is therefore not causal at
    all; it is rejected rather than silently accepted.

    For chunk-by-chunk inference, pass a [`~cache_utils.Cache`] built from this config, for instance
    `DynamicCache(config=config)`, together with a [`CausalWavLMPositionalConvCache`]; both are updated in place, so
    the same objects are reused across calls. Note that the feature extractor itself keeps no state across chunks.
    """

    config: CausalWavLMConfig

    def __init__(self, config: CausalWavLMConfig):
        if config.feat_extract_norm != "layer":
            raise ValueError(
                "CausalWavLMModel requires `feat_extract_norm='layer'` but the config asks for "
                f"'{config.feat_extract_norm}'. The alternative feature encoder normalizes every channel over the "
                "whole time axis with a group norm, so each frame would depend on the entire waveform. Checkpoints "
                "with `do_stable_layer_norm=True`, microsoft/wavlm-large among them, already use 'layer'."
            )

        # `WavLMModel.__init__` is bypassed so that the non-causal encoder is never allocated.
        CausalWavLMPreTrainedModel.__init__(self, config)
        self.config = config
        self.feature_extractor = WavLMFeatureEncoder(config)
        self.feature_projection = WavLMFeatureProjection(config)

        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.Tensor(config.hidden_size).uniform_())

        if config.do_stable_layer_norm:
            self.encoder = CausalWavLMEncoderStableLayerNorm(config)
        else:
            self.encoder = CausalWavLMEncoder(config)

        self.adapter = WavLMAdapter(config) if config.add_adapter else None

        self.post_init()

    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        mask_time_indices: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        padding_cache: CausalWavLMPositionalConvCache | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | WavLMBaseModelOutput:
        r"""
        Args:
            input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Raw waveform sampled at 16 kHz.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Sample-level mask, 1 for real samples and 0 for padding.
            mask_time_indices (`torch.BoolTensor` of shape `(batch_size, num_frames)`, *optional*):
                Frames to replace with the learned mask embedding, for masked-prediction objectives.
            past_key_values (`~cache_utils.Cache`, *optional*):
                Key-value cache holding the frames seen in earlier chunks. Updated in place and not returned; keep a
                reference to the object that is passed in. When it is used, `attention_mask` must cover the cached
                frames as well as the current ones.
            padding_cache (`CausalWavLMPositionalConvCache`, *optional*):
                Left-context cache for the positional convolution. Updated in place.

        Returns:
            [`~modeling_outputs.Wav2Vec2BaseModelOutput`] or `tuple`: one 1024-dimensional vector per 20 ms frame.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            padding_cache=padding_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return WavLMBaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


__all__ = [
    "CausalWavLMAttention",
    "CausalWavLMEncoder",
    "CausalWavLMEncoderLayer",
    "CausalWavLMEncoderLayerStableLayerNorm",
    "CausalWavLMEncoderStableLayerNorm",
    "CausalWavLMModel",
    "CausalWavLMPositionalConvCache",
    "CausalWavLMPositionalConvEmbedding",
    "CausalWavLMPreTrainedModel",
]
