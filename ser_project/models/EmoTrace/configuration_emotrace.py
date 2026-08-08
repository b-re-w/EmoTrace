# coding=utf-8
"""EmoTrace model configuration."""

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


DEFAULT_BACKBONE_MODEL_TYPE = "causal_wavlm"
r"""`model_type` of the encoder EmoTrace is trained on."""

BACKBONE_MODEL_TYPES = ("causal_wavlm", "wavlm")
r"""
Encoders the heads may be attached to, by `model_type`.

`causal_wavlm` is the online encoder. `wavlm` is the bidirectional model it is derived from, kept reachable so
that the two can be compared under the same heads, the same trainer and the same checkpoint format; a model built
on it is not causal and cannot be streamed.
"""


def get_backbone_config_class(model_type: str = DEFAULT_BACKBONE_MODEL_TYPE) -> type[PreTrainedConfig]:
    r"""
    Return the configuration class of an EmoTrace backbone.

    The backbones live in sibling packages and are resolved on first use rather than at import time, so that this
    module can be imported and unit tested independently of them.

    Args:
        model_type (`str`, *optional*, defaults to `"causal_wavlm"`):
            One of `BACKBONE_MODEL_TYPES`.

    Returns:
        `type[PreTrainedConfig]`: the configuration class of that encoder.
    """
    if model_type not in BACKBONE_MODEL_TYPES:
        raise ValueError(f"`model_type` must be one of {BACKBONE_MODEL_TYPES}, but is {model_type!r}.")

    try:
        if model_type == "wavlm":
            from ..WavLM import WavLMConfig

            return WavLMConfig

        from ..CausalWavLM import CausalWavLMConfig
    except ImportError as error:
        raise ImportError(
            f"EmoTrace could not import the configuration class of the {model_type!r} backbone from "
            "`ser_project.models`."
        ) from error

    return CausalWavLMConfig


@strict
class EmoTraceConfig(PreTrainedConfig):
    r"""
    Configuration class for [`EmoTraceForVADPrediction`], a causal speech encoder with three thin linear heads
    predicting continuous emotion traces, a CTC transcript and frame-level speaker activity.

    Following the composite-configuration pattern of multimodal models, the encoder is described by a nested
    `CausalWavLMConfig` stored under `backbone_config`; every other field configures the heads and the loss
    composition. All three heads operate at the backbone frame rate (one frame per 20 ms).

    Args:
        backbone_config (`dict`, `CausalWavLMConfig` or `WavLMConfig`, *optional*):
            Configuration of the speech encoder. A dictionary is converted to the configuration class its
            `model_type` entry names, defaulting to `CausalWavLMConfig`; if omitted, a default `CausalWavLMConfig`
            is created. A bidirectional `WavLMConfig` is accepted so that the same heads can be measured on the
            encoder the causal one is derived from, and the resulting model is then neither causal nor streamable.
        vocab_size (`int`, *optional*, defaults to 52):
            Size of the CTC vocabulary, i.e. the output dimension of the ASR head. The default matches the
            vocabulary produced by `build_emotrace_ctc_vocab` (characters plus atomic vocal-event tokens).
        num_vad_targets (`int`, *optional*, defaults to 3):
            Output dimension of the VAD head, one value per emotional attribute (valence, arousal, dominance).
        num_speaker_labels (`int`, *optional*, defaults to 3):
            Output dimension of the speaker head. The targets are binary and ordered as (speech, overlap, change).
        use_ctc_head (`bool`, *optional*, defaults to `True`):
            Whether to instantiate the ASR head. Set to `False` for ablations that train without transcript
            supervision so that the unused projection is not carried in the checkpoint.
        use_spk_head (`bool`, *optional*, defaults to `True`):
            Whether to instantiate the speaker head.
        head_dropout (`float`, *optional*, defaults to 0.0):
            Dropout applied to the backbone output before every head.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer used for the head projections.
        vad_loss_weight (`float`, *optional*, defaults to 1.0):
            Weight of the concordance correlation coefficient loss in the total loss.
        ctc_loss_weight (`float`, *optional*, defaults to 1.0):
            Weight of the CTC loss in the total loss.
        spk_loss_weight (`float`, *optional*, defaults to 1.0):
            Weight of the speaker binary cross-entropy loss in the total loss.
        vad_loss_eval_frames_only (`bool`, *optional*, defaults to `False`):
            Whether to restrict the VAD loss to the frames flagged by `vad_eval_mask`, i.e. the positions of the
            original 100 ms annotation grid. When `False` the loss is computed on every labelled 20 ms frame.
        ctc_blank_token_id (`int`, *optional*, defaults to 0):
            Index of the CTC blank symbol in the ASR vocabulary.
        ctc_loss_reduction (`str`, *optional*, defaults to `"mean"`):
            Reduction passed to `torch.nn.functional.ctc_loss`. One of `"mean"`, `"sum"` or `"none"`.
        ctc_zero_infinity (`bool`, *optional*, defaults to `True`):
            Whether to zero infinite CTC losses, which occur when the target is longer than the frame sequence.
        spk_pos_weight (`list[float]`, *optional*, defaults to `(1.0, 1.0, 10.0)`):
            Positive-class weights of the speaker head, ordered as (speech, overlap, change). The `change` target
            is active on a single frame per speaker transition and is up-weighted accordingly.

    Example:

    ```python
    >>> from ser_project.models.EmoTrace import EmoTraceConfig, EmoTraceForVADPrediction

    >>> configuration = EmoTraceConfig()
    >>> model = EmoTraceForVADPrediction(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "emotrace"
    sub_configs = {"backbone_config": PreTrainedConfig}
    # `Trainer` reads this to decide which forward outputs to accumulate during evaluation; the per-task loss
    # dictionary holds scalars and the frame representations are far too large to gather over a full split.
    keys_to_ignore_at_inference = ["loss_dict", "last_hidden_state", "hidden_states", "attentions"]

    backbone_config: dict | PreTrainedConfig | None = None
    vocab_size: int = 52
    num_vad_targets: int = 3
    num_speaker_labels: int = 3
    use_ctc_head: bool = True
    use_spk_head: bool = True
    head_dropout: float | int = 0.0
    initializer_range: float = 0.02
    vad_loss_weight: float | int = 1.0
    ctc_loss_weight: float | int = 1.0
    spk_loss_weight: float | int = 1.0
    vad_loss_eval_frames_only: bool = False
    ctc_blank_token_id: int = 0
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = True
    spk_pos_weight: list[float | int] | tuple[float | int, ...] = (1.0, 1.0, 10.0)

    def __post_init__(self, **kwargs):
        if self.backbone_config is None:
            self.backbone_config = get_backbone_config_class()()
        elif isinstance(self.backbone_config, dict):
            settings = dict(self.backbone_config)
            model_type = settings.pop("model_type", DEFAULT_BACKBONE_MODEL_TYPE)
            self.backbone_config = get_backbone_config_class(model_type)(**settings)
        super().__post_init__(**kwargs)

    @property
    def backbone_model_type(self) -> str:
        """`model_type` of the encoder, which selects the class the heads are built on."""
        return self.backbone_config.model_type

    @property
    def is_causal(self) -> bool:
        """Whether the encoder only reads the past, which is what makes streaming inference meaningful."""
        return self.backbone_model_type == DEFAULT_BACKBONE_MODEL_TYPE

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if len(self.spk_pos_weight) != self.num_speaker_labels:
            raise ValueError(
                "`spk_pos_weight` must provide one weight per speaker target, but got "
                f"`len(config.spk_pos_weight) = {len(self.spk_pos_weight)}` for "
                f"`config.num_speaker_labels = {self.num_speaker_labels}`."
            )
        if not 0 <= self.ctc_blank_token_id < self.vocab_size:
            raise ValueError(
                f"`ctc_blank_token_id = {self.ctc_blank_token_id}` is outside the ASR vocabulary of size "
                f"{self.vocab_size}."
            )
        if self.ctc_loss_reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f'`ctc_loss_reduction` must be one of "mean", "sum" or "none", but is "{self.ctc_loss_reduction}".'
            )

    @property
    def hidden_size(self) -> int:
        """Dimensionality of the backbone frame representations the heads project from."""
        return self.backbone_config.hidden_size

    @property
    def inputs_to_logits_ratio(self) -> int:
        """Number of waveform samples consumed per output frame."""
        return self.backbone_config.inputs_to_logits_ratio


__all__ = ["BACKBONE_MODEL_TYPES", "DEFAULT_BACKBONE_MODEL_TYPE", "EmoTraceConfig", "get_backbone_config_class"]
