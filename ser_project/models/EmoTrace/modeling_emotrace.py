# coding=utf-8
"""PyTorch EmoTrace model."""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_emotrace import BACKBONE_MODEL_TYPES, DEFAULT_BACKBONE_MODEL_TYPE, EmoTraceConfig


logger = logging.get_logger(__name__)


def get_backbone_model_class(model_type: str = DEFAULT_BACKBONE_MODEL_TYPE) -> type[PreTrainedModel]:
    r"""
    Return the model class of an EmoTrace backbone.

    The backbones live in sibling packages and are resolved on first use rather than at import time, so that this
    module can be imported and unit tested independently of them.

    Args:
        model_type (`str`, *optional*, defaults to `"causal_wavlm"`):
            One of `BACKBONE_MODEL_TYPES`.

    Returns:
        `type[PreTrainedModel]`: the encoder class of that `model_type`.
    """
    if model_type not in BACKBONE_MODEL_TYPES:
        raise ValueError(f"`model_type` must be one of {BACKBONE_MODEL_TYPES}, but is {model_type!r}.")

    try:
        if model_type == "wavlm":
            from ..WavLM import WavLMModel

            return WavLMModel

        from ..CausalWavLM import CausalWavLMModel
    except ImportError as error:
        raise ImportError(
            f"EmoTrace could not import the model class of the {model_type!r} backbone from `ser_project.models`."
        ) from error

    return CausalWavLMModel


def get_ccc_loss() -> Callable:
    r"""
    Return the concordance correlation coefficient loss used to train the VAD head.

    Returns:
        `Callable`: `ccc_loss(predictions, targets, mask=None)` from `ser_project.utils.losses`.
    """
    try:
        from ...utils.losses import ccc_loss
    except ImportError as error:
        raise ImportError(
            "The VAD loss requires `ccc_loss` from `ser_project.utils.losses`, which could not be imported."
        ) from error

    return ccc_loss


@dataclass
class EmoTraceOutput(ModelOutput):
    r"""
    Output of [`EmoTraceForVADPrediction`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Weighted sum of the losses of every task whose labels were provided. `None` when no labels are given.
        loss_dict (`dict[str, torch.FloatTensor]`, *optional*):
            Detached, *unweighted* per-task losses under the keys `loss_vad`, `loss_ctc` and `loss_spk`. Only the
            tasks that contributed to `loss` are present.
        vad_predictions (`torch.FloatTensor` of shape `(batch_size, num_frames, num_vad_targets)`):
            Valence, arousal and dominance predicted at every frame.
        ctc_logits (`torch.FloatTensor` of shape `(batch_size, num_frames, vocab_size)`, *optional*):
            Unnormalised scores of the ASR head. `None` when the head is disabled.
        spk_logits (`torch.FloatTensor` of shape `(batch_size, num_frames, num_speaker_labels)`, *optional*):
            Unnormalised scores of the speaker head, ordered as (speech, overlap, change). `None` when the head is
            disabled.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_frames, hidden_size)`):
            Frame representations produced by the backbone, before head dropout.
        hidden_states (`tuple[torch.FloatTensor]`, *optional*):
            Backbone hidden states, returned when `output_hidden_states=True`.
        attentions (`tuple[torch.FloatTensor]`, *optional*):
            Backbone attention weights, returned when `output_attentions=True`.
    """

    loss: torch.FloatTensor | None = None
    loss_dict: dict[str, torch.FloatTensor] | None = None
    vad_predictions: torch.FloatTensor | None = None
    ctc_logits: torch.FloatTensor | None = None
    spk_logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class EmoTracePreTrainedModel(PreTrainedModel):
    r"""
    Base class handling weight initialisation and the interface for loading pretrained EmoTrace checkpoints.
    """

    config: EmoTraceConfig
    base_model_prefix = "backbone"
    main_input_name = "input_values"
    input_modalities = "audio"
    supports_gradient_checkpointing = True


class EmoTraceForVADPrediction(EmoTracePreTrainedModel):
    r"""
    EmoTrace: a causal speech encoder with three thin linear heads sharing a single 20 ms frame grid.

    The encoder is a `CausalWavLMModel`; on top of it sit a VAD regression head trained with a concordance
    correlation coefficient loss, an ASR head trained with CTC over a character vocabulary extended with atomic
    vocal-event tokens, and a speaker head trained with binary cross-entropy over (speech, overlap, change).
    There is no sequence model and no temporal pooling above the encoder: every head is a single projection, so
    an ablation measures the quality of the representation rather than the capacity of a head.

    The total loss is `vad_loss_weight * L_CCC + ctc_loss_weight * L_CTC + spk_loss_weight * L_BCE`. A task whose
    labels are absent from the batch is skipped entirely, so the same class covers every ablation of the loss
    composition.

    The encoder class follows `config.backbone_config.model_type`, so the bidirectional `WavLMModel` the causal
    encoder is derived from can carry the identical heads and be trained by the identical trainer; that is what
    makes the cost of causality measurable at equal head capacity. Such a model reads the whole utterance and
    rejects the streaming caches, which `config.is_causal` reports.
    """

    def __init__(self, config: EmoTraceConfig):
        super().__init__(config)

        self.backbone = get_backbone_model_class(config.backbone_model_type)(config.backbone_config)
        hidden_size = config.backbone_config.hidden_size

        self.dropout = nn.Dropout(config.head_dropout)
        self.vad_head = nn.Linear(hidden_size, config.num_vad_targets)
        self.ctc_head = nn.Linear(hidden_size, config.vocab_size) if config.use_ctc_head else None
        self.spk_head = nn.Linear(hidden_size, config.num_speaker_labels) if config.use_spk_head else None

        # Kept out of the state dict: it is a plain restatement of `config.spk_pos_weight` that only needs to
        # follow the model across devices.
        self.register_buffer(
            "spk_pos_weight", torch.tensor(config.spk_pos_weight, dtype=torch.float32), persistent=False
        )

        self.post_init()

    @classmethod
    def from_backbone_pretrained(
        cls,
        backbone_name_or_path: str,
        backbone_model_type: str = DEFAULT_BACKBONE_MODEL_TYPE,
        **kwargs,
    ) -> "EmoTraceForVADPrediction":
        r"""
        Build an EmoTrace model whose encoder is initialised from a pretrained encoder checkpoint and whose heads
        are freshly initialised.

        Args:
            backbone_name_or_path (`str`):
                Identifier or local path of the encoder checkpoint, typically the `CausalWavLMModel` obtained at the
                end of bidirectional-to-causal distillation.
            backbone_model_type (`str`, *optional*, defaults to `"causal_wavlm"`):
                Encoder class the checkpoint is loaded with, one of `BACKBONE_MODEL_TYPES`. It is never inferred
                from the checkpoint: released WavLM weights are the same tensors for both encoders, so reading
                `model_type` would silently decide whether the resulting model is causal.
            kwargs:
                Forwarded to [`EmoTraceConfig`], e.g. `vocab_size` or the loss weights. `dtype`, `attn_implementation`
                and `device_map` are forwarded to the backbone loader instead.

        Returns:
            [`EmoTraceForVADPrediction`]: the assembled model.
        """
        backbone_kwargs = {
            key: kwargs.pop(key) for key in ("dtype", "attn_implementation", "device_map") if key in kwargs
        }
        backbone = get_backbone_model_class(backbone_model_type).from_pretrained(
            backbone_name_or_path, **backbone_kwargs
        )

        config = EmoTraceConfig(backbone_config=backbone.config, **kwargs)
        model = cls(config)
        model.backbone.load_state_dict(backbone.state_dict())
        return model

    def freeze_feature_encoder(self):
        """Disable gradient computation for the convolutional feature encoder of the backbone."""
        self.backbone.freeze_feature_encoder()

    def freeze_backbone(self):
        """Disable gradient computation for the whole backbone, leaving only the heads trainable."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor | int) -> torch.LongTensor:
        r"""
        Convert waveform lengths into the number of frames produced by the backbone.

        Args:
            input_lengths (`torch.LongTensor` or `int`):
                Number of waveform samples.

        Returns:
            `torch.LongTensor`: number of 20 ms frames.
        """
        return self.backbone._get_feat_extract_output_lengths(input_lengths)

    def get_frame_attention_mask(self, num_frames: int, attention_mask: torch.LongTensor) -> torch.BoolTensor:
        r"""
        Convert a waveform-level attention mask into a frame-level one.

        Args:
            num_frames (`int`):
                Number of frames produced by the backbone for the batch.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Waveform-level attention mask.

        Returns:
            `torch.BoolTensor` of shape `(batch_size, num_frames)`: `True` on frames covered by real audio.
        """
        return self.backbone._get_feature_vector_attention_mask(num_frames, attention_mask)

    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        vad_labels: torch.FloatTensor | None = None,
        vad_eval_mask: torch.BoolTensor | None = None,
        ctc_labels: torch.LongTensor | None = None,
        ctc_label_lengths: torch.LongTensor | None = None,
        ctc_loss_mask: torch.BoolTensor | None = None,
        spk_labels: torch.FloatTensor | None = None,
        past_key_values=None,
        padding_cache=None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | EmoTraceOutput:
        r"""
        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Raw 16 kHz waveform, normalised with dataset-level constants.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask of the valid waveform samples.
            vad_labels (`torch.FloatTensor` of shape `(batch_size, num_frames, num_vad_targets)`, *optional*):
                Valence, arousal and dominance on the 20 ms grid. `float("nan")` marks positions without a label
                and is excluded from the loss, per attribute.
            vad_eval_mask (`torch.BoolTensor` of shape `(batch_size, num_frames)`, *optional*):
                `True` at the frames that coincide with the original 100 ms annotation grid. Used by the loss only
                when `config.vad_loss_eval_frames_only` is set; it is otherwise carried for evaluation.
            ctc_labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
                Transcript token ids padded with `-100`.
            ctc_label_lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Number of valid tokens per transcript. Derived from `ctc_labels` when omitted.
            ctc_loss_mask (`torch.BoolTensor` of shape `(batch_size,)`, *optional*):
                `True` for the samples whose transcript supervision is usable. Samples flagged `False`, for
                instance overlapped speech with unreliable pseudo labels, do not contribute to the CTC loss.
            spk_labels (`torch.FloatTensor` of shape `(batch_size, num_frames, num_speaker_labels)`, *optional*):
                Binary speaker targets ordered as (speech, overlap, change). `float("nan")` marks unlabelled
                positions.
            past_key_values (`Cache`, *optional*):
                Attention cache of the backbone, holding the frames of the chunks already streamed. Updated in place
                and not returned, so the caller keeps the reference it passed in.
            padding_cache (`CausalWavLMPositionalConvCache`, *optional*):
                Left-context cache of the causal positional convolution, also updated in place. It has to travel with
                `past_key_values`: without it the first frames of every chunk would be convolved against zeros
                instead of against the end of the previous chunk.

        Returns:
            [`EmoTraceOutput`] or `tuple`.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # A bidirectional backbone declares neither cache, so they are forwarded only when a caller supplies one
        # and are rejected with a readable message rather than a `TypeError` raised inside the encoder.
        streaming_caches = {"past_key_values": past_key_values, "padding_cache": padding_cache}
        streaming_caches = {key: value for key, value in streaming_caches.items() if value is not None}
        if streaming_caches and not self.config.is_causal:
            raise ValueError(
                f"{sorted(streaming_caches)} were passed, but this model was built on the "
                f"'{self.config.backbone_model_type}' encoder, which reads the whole utterance at once and keeps "
                "no state between chunks. Streaming inference requires a 'causal_wavlm' backbone."
            )

        outputs = self.backbone(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **streaming_caches,
        )

        last_hidden_state = outputs.last_hidden_state
        hidden_states = self.dropout(last_hidden_state)
        num_frames = hidden_states.shape[1]

        if attention_mask is not None:
            frame_mask = self.get_frame_attention_mask(num_frames, attention_mask)
        else:
            frame_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)

        vad_predictions = self.vad_head(hidden_states)
        ctc_logits = self.ctc_head(hidden_states) if self.ctc_head is not None else None
        spk_logits = self.spk_head(hidden_states) if self.spk_head is not None else None

        loss = None
        loss_dict = {}

        if vad_labels is not None:
            task_loss = self._vad_loss(vad_predictions, vad_labels, frame_mask, vad_eval_mask)
            loss_dict["loss_vad"] = task_loss.detach()
            loss = self.config.vad_loss_weight * task_loss

        if ctc_labels is not None:
            if ctc_logits is None:
                logger.warning_once(
                    "`ctc_labels` were passed but the ASR head is disabled (`config.use_ctc_head=False`); the CTC "
                    "loss is skipped."
                )
            else:
                task_loss = self._ctc_loss(ctc_logits, ctc_labels, ctc_label_lengths, ctc_loss_mask, attention_mask)
                loss_dict["loss_ctc"] = task_loss.detach()
                weighted = self.config.ctc_loss_weight * task_loss
                loss = weighted if loss is None else loss + weighted

        if spk_labels is not None:
            if spk_logits is None:
                logger.warning_once(
                    "`spk_labels` were passed but the speaker head is disabled (`config.use_spk_head=False`); the "
                    "speaker loss is skipped."
                )
            else:
                task_loss = self._spk_loss(spk_logits, spk_labels, frame_mask)
                loss_dict["loss_spk"] = task_loss.detach()
                weighted = self.config.spk_loss_weight * task_loss
                loss = weighted if loss is None else loss + weighted

        if loss is None:
            loss_dict = None

        output = EmoTraceOutput(
            loss=loss,
            loss_dict=loss_dict,
            vad_predictions=vad_predictions,
            ctc_logits=ctc_logits,
            spk_logits=spk_logits,
            last_hidden_state=last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return output if return_dict else output.to_tuple()

    def _vad_loss(
        self,
        predictions: torch.FloatTensor,
        labels: torch.FloatTensor,
        frame_mask: torch.BoolTensor,
        eval_mask: torch.BoolTensor | None,
    ) -> torch.Tensor:
        r"""
        Concordance correlation coefficient loss over the labelled frames.

        Args:
            predictions (`torch.FloatTensor` of shape `(batch_size, num_frames, num_vad_targets)`):
                Output of the VAD head.
            labels (`torch.FloatTensor` of shape `(batch_size, num_frames, num_vad_targets)`):
                Targets, with `float("nan")` at unlabelled positions.
            frame_mask (`torch.BoolTensor` of shape `(batch_size, num_frames)`):
                `True` on frames covered by real audio.
            eval_mask (`torch.BoolTensor` of shape `(batch_size, num_frames)`, *optional*):
                Positions of the original 100 ms annotation grid.

        Returns:
            `torch.Tensor`: scalar loss.
        """
        mask = torch.isfinite(labels) & frame_mask.unsqueeze(-1)
        if eval_mask is not None and self.config.vad_loss_eval_frames_only:
            mask = mask & eval_mask.unsqueeze(-1)

        if not mask.any():
            return predictions.sum() * 0.0

        targets = torch.nan_to_num(labels)
        return get_ccc_loss()(predictions, targets, mask=mask)

    def _ctc_loss(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_lengths: torch.LongTensor | None,
        loss_mask: torch.BoolTensor | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        r"""
        CTC loss over the samples selected by `loss_mask`.

        Args:
            logits (`torch.FloatTensor` of shape `(batch_size, num_frames, vocab_size)`):
                Output of the ASR head.
            labels (`torch.LongTensor` of shape `(batch_size, target_length)`):
                Transcript token ids padded with `-100`.
            label_lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Number of valid tokens per transcript.
            loss_mask (`torch.BoolTensor` of shape `(batch_size,)`, *optional*):
                Samples whose transcript supervision is usable.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Waveform-level attention mask, used to derive the number of valid frames per sample.

        Returns:
            `torch.Tensor`: scalar loss.
        """
        batch_size, num_frames, _ = logits.shape
        labels_mask = labels >= 0

        if labels_mask.any() and labels[labels_mask].max() >= self.config.vocab_size:
            raise ValueError(
                f"Label values must be < vocab_size = {self.config.vocab_size}, but the batch contains "
                f"{int(labels[labels_mask].max())}."
            )

        if attention_mask is None:
            input_lengths = torch.full((batch_size,), num_frames, dtype=torch.long, device=logits.device)
        else:
            input_lengths = self.get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        input_lengths = input_lengths.clamp(max=num_frames)

        target_lengths = label_lengths if label_lengths is not None else labels_mask.sum(-1)
        target_lengths = target_lengths.to(torch.long)

        if loss_mask is None:
            keep = torch.ones(batch_size, dtype=torch.bool, device=logits.device)
        else:
            keep = loss_mask.bool()
        keep = keep & (target_lengths > 0)

        if not keep.any():
            return logits.sum() * 0.0

        flattened_targets = labels[keep].masked_select(labels_mask[keep])

        # `ctc_loss` expects (num_frames, batch_size, vocab_size) log probabilities and does not support fp16.
        log_probs = nn.functional.log_softmax(logits[keep], dim=-1, dtype=torch.float32).transpose(0, 1)

        with torch.backends.cudnn.flags(enabled=False):
            return nn.functional.ctc_loss(
                log_probs,
                flattened_targets,
                input_lengths[keep],
                target_lengths[keep],
                blank=self.config.ctc_blank_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )

    def _spk_loss(
        self,
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
        frame_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        r"""
        Masked binary cross-entropy over the frame-level speaker targets.

        Args:
            logits (`torch.FloatTensor` of shape `(batch_size, num_frames, num_speaker_labels)`):
                Output of the speaker head.
            labels (`torch.FloatTensor` of shape `(batch_size, num_frames, num_speaker_labels)`):
                Binary targets, with `float("nan")` at unlabelled positions.
            frame_mask (`torch.BoolTensor` of shape `(batch_size, num_frames)`):
                `True` on frames covered by real audio.

        Returns:
            `torch.Tensor`: scalar loss.
        """
        mask = torch.isfinite(labels) & frame_mask.unsqueeze(-1)
        if not mask.any():
            return logits.sum() * 0.0

        targets = torch.nan_to_num(labels)
        elementwise = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.spk_pos_weight.to(logits.dtype),
            reduction="none",
        )
        return (elementwise * mask).sum() / mask.sum()


__all__ = [
    "EmoTraceForVADPrediction",
    "EmoTraceOutput",
    "EmoTracePreTrainedModel",
    "get_backbone_model_class",
    "get_ccc_loss",
]
