r"""Training entry points for the two stages that produce an EmoTrace model.

The first stage turns a bidirectional [`WavLMModel`] into a causal encoder: the student is a
`CausalWavLMModel` initialised from the same checkpoint as the teacher, and it is trained to reproduce the
teacher's per-frame hidden states from past context only. No corpus label is involved, so the stage runs on
any 16 kHz audio.

The second stage trains [`EmoTraceForVADPrediction`] with its three heads at once. The weighted sum of the
task losses is formed inside the model, from the weights carried by its configuration; this module owns the
weights, switches the individual tasks on and off, and reports every term separately.

Both stages use the stock `transformers.Trainer` loop with a standard `TrainingArguments`. The two
subclasses below override `compute_loss` and `log`, and nothing else.

Example:

```python
>>> from ser_project.utils.train import parse_joint_arguments, EmoTraceJointTrainer

>>> model_args, joint_args, training_args = parse_joint_arguments()  # doctest: +SKIP
>>> model = model_args.load_model(joint_args)  # doctest: +SKIP
>>> trainer = EmoTraceJointTrainer(  # doctest: +SKIP
...     model=model,
...     args=training_args,
...     joint_args=joint_args,
...     train_dataset=train_dataset,
...     data_collator=collator,
... )
>>> trainer.train()  # doctest: +SKIP
```
"""

import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from transformers import HfArgumentParser, Trainer, TrainingArguments, WavLMModel
from transformers.utils import logging

from ..models.EmoTrace import BACKBONE_MODEL_TYPES, DEFAULT_BACKBONE_MODEL_TYPE
from .evaluate import DEFAULT_LABEL_NAMES
from .losses import hidden_state_distillation_loss


logger = logging.get_logger(__name__)


DEFAULT_WAVLM_NAME_OR_PATH = "microsoft/wavlm-large"
r"""Checkpoint that provides both the distillation teacher and the student initialisation."""

VAD_LABEL_NAMES = DEFAULT_LABEL_NAMES
r"""Batch keys the evaluation loop must forward to `compute_metrics`, in that order.

It is the list [`~ser_project.utils.evaluate.VADMetricComputer`] reads positionally, so the two are the same
object rather than two tuples that happen to agree.
"""

CTC_BATCH_KEYS = ("ctc_labels", "ctc_label_lengths", "ctc_loss_mask")
r"""Batch keys carrying transcript supervision."""

SPEAKER_BATCH_KEYS = ("spk_labels",)
r"""Batch keys carrying frame-level speaker supervision."""

TRANSCRIPT_MODES = ("verbatim", "intended")
r"""Transcript flavours a batch may provide, as `<key>_<mode>` columns."""

TRANSCRIPT_SUPERVISION_CHOICES = ("off", *TRANSCRIPT_MODES)
r"""Accepted values of `EmoTraceJointArguments.transcript_supervision`."""

DISTILLATION_OBJECTIVES = ("cosine", "mse")
r"""Accepted values of `DistillationArguments.objective`."""


def _mode_key(key: str, mode: str) -> str:
    r"""Name of the transcript column of `key` in transcription mode `mode`."""
    return f"{key}_{mode}"


class _ScalarAccumulator:
    r"""Running mean of named scalars over a logging interval.

    Values are kept as device tensors until they are read, so accumulating a loss term costs no
    host/device synchronisation; the single synchronisation happens when `consume` is called, once per
    logging step. Keys that appear in only some of the steps are averaged over the steps that provided
    them.
    """

    def __init__(self) -> None:
        self._sums: dict[str, Any] = {}
        self._counts: dict[str, int] = {}

    def __bool__(self) -> bool:
        return bool(self._counts)

    def update(self, values: Mapping[str, Any]) -> None:
        r"""Add one observation per entry of `values`.

        Args:
            values (`Mapping[str, torch.Tensor | float]`):
                Scalar values, either Python floats or zero-dimensional tensors.
        """
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.detach()
            if key in self._sums:
                self._sums[key] = self._sums[key] + value
                self._counts[key] += 1
            else:
                self._sums[key] = value
                self._counts[key] = 1

    def consume(self) -> dict[str, float]:
        r"""Return the mean of every accumulated key and start a new interval.

        Returns:
            `dict[str, float]`: mean value per key, empty when nothing was accumulated.
        """
        means = {key: float(value) / self._counts[key] for key, value in self._sums.items()}
        self._sums.clear()
        self._counts.clear()
        return means


def keep_vad_predictions(logits: Any, labels: Any = None) -> Any:
    r"""`preprocess_logits_for_metrics` callable keeping only the VAD predictions.

    [`EmoTraceForVADPrediction`] also returns the CTC and speaker logits, which are far larger than the
    quantity the metrics need and would be accumulated over the whole evaluation split. Passing this
    function to `Trainer(preprocess_logits_for_metrics=...)` drops them right after the forward pass.

    Args:
        logits (`torch.Tensor` or `tuple[torch.Tensor, ...]`):
            Model outputs kept by the evaluation loop; the VAD predictions come first.
        labels (`torch.Tensor`, *optional*):
            Unused, present because `Trainer` calls this with two arguments.

    Returns:
        `torch.Tensor`: the VAD predictions of shape `(batch_size, num_frames, num_vad_targets)`.
    """
    if isinstance(logits, (tuple, list)):
        return logits[0]
    return logits


# ---------------------------------------------------------------------------
# Stage one: bidirectional-to-causal distillation
# ---------------------------------------------------------------------------


@dataclass
class DistillationArguments:
    r"""Arguments of the bidirectional-to-causal distillation stage.

    Args:
        teacher_name_or_path (`str`, *optional*, defaults to `"microsoft/wavlm-large"`):
            Checkpoint of the frozen bidirectional teacher, loaded as a `WavLMModel`.
        student_name_or_path (`str`, *optional*, defaults to `"microsoft/wavlm-large"`):
            Checkpoint the causal student is initialised from. `CausalWavLMModel` is weight compatible
            with released WavLM checkpoints, so the student normally starts from the teacher itself and
            only has to learn to do without future context.
        sliding_window (`int`, *optional*, defaults to 800):
            Number of frames the student may attend to, itself included. At one frame per 20 ms the
            default is 16 s of left context.
        objective (`str`, *optional*, defaults to `"cosine"`):
            `"cosine"` matches the direction of the frame representations and is insensitive to the norm
            drift between the two models; `"mse"` matches them outright.
        layer_indices (`list[int]`, *optional*):
            Student hidden states to supervise, as indices into the `hidden_states` tuple; index `0` is
            the feature projection and index `i` the output of encoder layer `i`. Defaults to every
            encoder layer, which excludes index `0`: the convolutional front end and the feature
            projection are shared with the teacher and already produce identical frames there.
        teacher_layer_indices (`list[int]`, *optional*):
            Teacher layers paired with `layer_indices`, in the same order. Defaults to `layer_indices`,
            the correct pairing when student and teacher have the same depth.
        layer_weights (`list[float]`, *optional*):
            One weight per supervised layer, defaults to uniform weights.
        layer_reduction (`str`, *optional*, defaults to `"mean"`):
            `"mean"` or `"sum"` over the supervised layers.
        log_layer_residuals (`bool`, *optional*, defaults to `True`):
            Log the per-layer distillation residual under `distill_layer_<index>`. After convergence the
            residual is what the teacher representation holds and past context cannot supply, so its
            profile over depth is the measurement this stage exists to produce.
        disable_spec_augment (`bool`, *optional*, defaults to `True`):
            Turn `config.apply_spec_augment` off on the student. The teacher sees the unmasked waveform,
            so masked student frames would be asked to reproduce states they no longer have the input for.
        freeze_feature_encoder (`bool`, *optional*, defaults to `True`):
            Freeze the student's convolutional feature encoder, which is identical to the teacher's and
            has nothing to learn from this objective.
        dtype (`str`, *optional*):
            Weight dtype passed to both loaders, for instance `"bfloat16"`.
        attn_implementation (`str`, *optional*):
            Attention implementation passed to both loaders. The student supports `"sdpa"` and `"eager"`.
    """

    teacher_name_or_path: str = field(
        default=DEFAULT_WAVLM_NAME_OR_PATH,
        metadata={"help": "Checkpoint of the frozen bidirectional WavLM teacher."},
    )
    student_name_or_path: str = field(
        default=DEFAULT_WAVLM_NAME_OR_PATH,
        metadata={"help": "Checkpoint the CausalWavLM student is initialised from."},
    )
    sliding_window: int = field(
        default=800,
        metadata={"help": "Frames of left context the student attends to, itself included."},
    )
    objective: str = field(
        default="cosine",
        metadata={"help": "Hidden state matching objective.", "choices": list(DISTILLATION_OBJECTIVES)},
    )
    layer_indices: list[int] | None = field(
        default=None,
        metadata={"help": "Student layers to supervise. Defaults to every encoder layer."},
    )
    teacher_layer_indices: list[int] | None = field(
        default=None,
        metadata={"help": "Teacher layers paired with --layer_indices, in the same order."},
    )
    layer_weights: list[float] | None = field(
        default=None,
        metadata={"help": "One weight per supervised layer. Defaults to uniform weights."},
    )
    layer_reduction: str = field(
        default="mean",
        metadata={"help": "Reduction over the supervised layers.", "choices": ["mean", "sum"]},
    )
    log_layer_residuals: bool = field(
        default=True,
        metadata={"help": "Log the residual of every supervised layer under distill_layer_<index>."},
    )
    disable_spec_augment: bool = field(
        default=True,
        metadata={"help": "Disable SpecAugment on the student, which the teacher does not see."},
    )
    freeze_feature_encoder: bool = field(
        default=True,
        metadata={"help": "Freeze the student's convolutional feature encoder."},
    )
    dtype: str | None = field(
        default=None,
        metadata={"help": "Weight dtype of both models, for instance bfloat16."},
    )
    attn_implementation: str | None = field(
        default=None,
        metadata={"help": "Attention implementation of both models, for instance sdpa."},
    )

    def __post_init__(self):
        if self.objective not in DISTILLATION_OBJECTIVES:
            raise ValueError(f"`objective` must be one of {DISTILLATION_OBJECTIVES}, got {self.objective!r}.")
        if self.layer_reduction not in ("mean", "sum"):
            raise ValueError(f"`layer_reduction` must be 'mean' or 'sum', got {self.layer_reduction!r}.")
        if self.sliding_window < 1:
            raise ValueError(f"`sliding_window` must be a positive number of frames, got {self.sliding_window}.")
        if (
            self.layer_weights is not None
            and self.layer_indices is not None
            and len(self.layer_weights) != len(self.layer_indices)
        ):
            raise ValueError(
                f"`layer_weights` must provide one weight per supervised layer ({len(self.layer_indices)}), got "
                f"{len(self.layer_weights)}."
            )

    def _loader_kwargs(self) -> dict[str, Any]:
        r"""Keyword arguments shared by the teacher and the student loader."""
        kwargs: dict[str, Any] = {}
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype
        if self.attn_implementation is not None:
            kwargs["attn_implementation"] = self.attn_implementation
        return kwargs

    def load_teacher(self) -> WavLMModel:
        r"""Load the teacher, frozen and in evaluation mode.

        Returns:
            `WavLMModel`: the bidirectional teacher; its parameters require no gradient and its dropout
            and SpecAugment are inactive.
        """
        teacher = WavLMModel.from_pretrained(self.teacher_name_or_path, **self._loader_kwargs())
        teacher.eval()
        teacher.requires_grad_(False)
        return teacher

    def load_student(self):
        r"""Load the causal student.

        Returns:
            `CausalWavLMModel`: the student, with `config.sliding_window` set from these arguments and
            the feature encoder frozen when `freeze_feature_encoder` is set.
        """
        from ..models.CausalWavLM import CausalWavLMModel

        student = CausalWavLMModel.from_pretrained(
            self.student_name_or_path, sliding_window=self.sliding_window, **self._loader_kwargs()
        )
        if self.freeze_feature_encoder:
            student.freeze_feature_encoder()
        return student


class DistillationTrainer(Trainer):
    r"""Trains a causal student to reproduce the frame representations of a frozen bidirectional teacher.

    The two models run on the same waveform and emit one hidden state per 20 ms frame, so the layers are
    compared frame by frame with no alignment step; the loss is
    [`~ser_project.utils.losses.hidden_state_distillation_loss`]. Nothing but audio is required, and any
    label column a batch happens to carry is ignored.

    Padded frames are excluded from the comparison, using the frame-level mask the student derives from
    the sample-level `attention_mask`.

    The per-layer residuals are averaged over the logging interval on the local process and emitted
    alongside the loss, under `distill_layer_<index>` during training and `eval_distill_layer_<index>`
    during evaluation.

    Args:
        model (`CausalWavLMModel`):
            The student. Its `config.sliding_window` decides how much past context it may use.
        args (`TrainingArguments`, *optional*):
            Standard training arguments.
        teacher_model (`WavLMModel`):
            The frozen bidirectional teacher. It is put in evaluation mode, detached from autograd and
            moved next to the student; it is never saved by `save_model`.
        distillation_args ([`DistillationArguments`], *optional*):
            Objective, layer selection and logging switches. Defaults to [`DistillationArguments`].
        kwargs:
            Forwarded to `transformers.Trainer`.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        args: TrainingArguments | None = None,
        *,
        teacher_model: nn.Module | None = None,
        distillation_args: DistillationArguments | None = None,
        **kwargs,
    ):
        if teacher_model is None:
            raise ValueError(
                "`teacher_model` is required. Load it with `DistillationArguments.load_teacher()` or pass any "
                "bidirectional `WavLMModel` sharing the student's frame rate and hidden size."
            )

        super().__init__(model=model, args=args, **kwargs)

        self.distillation_args = distillation_args if distillation_args is not None else DistillationArguments()

        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        self.teacher_model.to(self.args.device)

        student_config = self.accelerator.unwrap_model(self.model).config
        teacher_config = self.teacher_model.config
        if student_config.hidden_size != teacher_config.hidden_size:
            raise ValueError(
                f"The student produces {student_config.hidden_size}-dimensional frames while the teacher produces "
                f"{teacher_config.hidden_size}-dimensional ones; hidden states can only be matched frame by frame "
                "when the two models share their hidden size."
            )
        if self.distillation_args.disable_spec_augment:
            student_config.apply_spec_augment = False

        # This objective has no notion of tokens, so the gradient-accumulation rescaling `Trainer` applies
        # when a model accepts loss kwargs would be wrong here.
        self.model_accepts_loss_kwargs = False
        # No column of the batch is a target; declaring that makes the evaluation loop call `compute_loss`
        # instead of looking for labels it will not find.
        self.label_names = []
        self.can_return_loss = True

        self._train_residuals = _ScalarAccumulator()
        self._eval_residuals = _ScalarAccumulator()

    def _resolve_layer_indices(self, num_hidden_states: int) -> list[int]:
        r"""Student layers to supervise, defaulting to every encoder layer.

        Args:
            num_hidden_states (`int`):
                Length of the student `hidden_states` tuple.

        Returns:
            `list[int]`: indices into that tuple.
        """
        if self.distillation_args.layer_indices is not None:
            return list(self.distillation_args.layer_indices)
        return list(range(1, num_hidden_states))

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        r"""Distillation loss between the student and teacher hidden states.

        Args:
            model (`nn.Module`):
                The student, possibly wrapped for distributed training.
            inputs (`dict[str, Any]`):
                Collated batch; only `input_values` and `attention_mask` are read.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Also return the student outputs.
            num_items_in_batch (`torch.Tensor` or `int`, *optional*):
                Unused; this loss averages over frames and layers, not over items.

        Returns:
            `torch.Tensor` or `tuple[torch.Tensor, Wav2Vec2BaseModelOutput]`: the loss, and the student
            outputs when `return_outputs` is set.
        """
        input_values = inputs.get("input_values")
        if input_values is None:
            raise KeyError(
                f"The batch carries no 'input_values'; got keys {sorted(inputs)}. Distillation reads the raw "
                "waveform and nothing else."
            )
        attention_mask = inputs.get("attention_mask")

        student_outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        self._move_teacher_to(input_values.device)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        student_hidden_states = student_outputs.hidden_states
        if student_hidden_states is None or teacher_outputs.hidden_states is None:
            raise ValueError(
                "Both models must return their hidden states. Check that `config.output_hidden_states` was not "
                "forced to `False`."
            )

        frame_mask = None
        if attention_mask is not None:
            num_frames = student_hidden_states[-1].shape[1]
            frame_mask = self.accelerator.unwrap_model(model)._get_feature_vector_attention_mask(
                num_frames, attention_mask
            )

        output = hidden_state_distillation_loss(
            student_hidden_states,
            teacher_outputs.hidden_states,
            layer_indices=self._resolve_layer_indices(len(student_hidden_states)),
            teacher_layer_indices=self.distillation_args.teacher_layer_indices,
            attention_mask=frame_mask,
            objective=self.distillation_args.objective,
            layer_weights=self.distillation_args.layer_weights,
            reduction=self.distillation_args.layer_reduction,
        )

        if self.distillation_args.log_layer_residuals:
            accumulator = self._train_residuals if model.training else self._eval_residuals
            accumulator.update({f"distill_layer_{index}": value for index, value in output.layer_losses.items()})

        return (output.loss, student_outputs) if return_outputs else output.loss

    def _move_teacher_to(self, device: torch.device) -> None:
        r"""Keep the teacher on the device the batch was placed on."""
        if next(self.teacher_model.parameters()).device != device:
            self.teacher_model.to(device)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        r"""Add the per-layer residuals of the interval that just ended to the log entry.

        Args:
            logs (`dict[str, float]`):
                Values `Trainer` is about to report.
            start_time (`float`, *optional*):
                Start of the interval, forwarded unchanged.
        """
        is_evaluation = any(key.startswith("eval_") for key in logs)
        accumulator = self._eval_residuals if is_evaluation else self._train_residuals
        if accumulator:
            prefix = "eval_" if is_evaluation else ""
            logs = {**logs, **{f"{prefix}{key}": value for key, value in accumulator.consume().items()}}
        super().log(logs, start_time)


# ---------------------------------------------------------------------------
# Stage two: the EmoTrace joint objective
# ---------------------------------------------------------------------------


@dataclass
class EmoTraceJointArguments:
    r"""Loss weights and ablation switches of the EmoTrace joint objective.

    The three tasks are switched independently, which is what the ablation table of the joint objective
    needs: the emotion head alone, plus transcripts, plus speaker activity, and the transcript task with
    either the verbatim or the intended targets.

    Args:
        vad_loss_weight (`float`, *optional*, defaults to 1.0):
            Weight of the concordance correlation coefficient loss.
        ctc_loss_weight (`float`, *optional*, defaults to 1.0):
            Weight of the CTC loss. Forced to `0.0` when transcript supervision is off.
        spk_loss_weight (`float`, *optional*, defaults to 1.0):
            Weight of the speaker binary cross-entropy loss. Forced to `0.0` when speaker supervision is
            off.
        transcript_supervision (`str`, *optional*, defaults to `"verbatim"`):
            `"off"` to train without transcripts, `"verbatim"` to use the transcript that keeps
            disfluencies and vocal events, `"intended"` to use the cleaned one.
        speaker_supervision (`bool`, *optional*, defaults to `True`):
            Whether the per-frame (speech, overlap, change) targets take part in the loss.
        vad_loss_eval_frames_only (`bool`, *optional*, defaults to `False`):
            Restrict the emotion loss to the frames of the original 100 ms annotation grid instead of
            every labelled 20 ms frame.
        log_task_losses (`bool`, *optional*, defaults to `True`):
            Log the unweighted per-task losses under `loss_vad`, `loss_ctc` and `loss_spk`.
    """

    vad_loss_weight: float = field(default=1.0, metadata={"help": "Weight of the CCC loss."})
    ctc_loss_weight: float = field(default=1.0, metadata={"help": "Weight of the CTC loss."})
    spk_loss_weight: float = field(default=1.0, metadata={"help": "Weight of the speaker BCE loss."})
    transcript_supervision: str = field(
        default="verbatim",
        metadata={
            "help": "Transcript targets of the CTC task, or 'off' to train without transcripts.",
            "choices": list(TRANSCRIPT_SUPERVISION_CHOICES),
        },
    )
    speaker_supervision: bool = field(
        default=True,
        metadata={"help": "Whether the frame-level speaker targets take part in the loss."},
    )
    vad_loss_eval_frames_only: bool = field(
        default=False,
        metadata={"help": "Compute the emotion loss on the 100 ms evaluation grid only."},
    )
    log_task_losses: bool = field(
        default=True,
        metadata={"help": "Log the unweighted per-task losses next to the total loss."},
    )

    def __post_init__(self):
        if self.transcript_supervision not in TRANSCRIPT_SUPERVISION_CHOICES:
            raise ValueError(
                f"`transcript_supervision` must be one of {TRANSCRIPT_SUPERVISION_CHOICES}, got "
                f"{self.transcript_supervision!r}."
            )

    @property
    def use_transcript_supervision(self) -> bool:
        r"""Whether the CTC task is part of the objective."""
        return self.transcript_supervision != "off"

    @property
    def transcript_mode(self) -> str | None:
        r"""Transcription mode of the CTC targets, `None` when the task is off."""
        return self.transcript_supervision if self.use_transcript_supervision else None

    def config_overrides(self) -> dict[str, Any]:
        r"""Configuration fields implied by these arguments.

        Returns:
            `dict[str, Any]`: keyword arguments for [`EmoTraceConfig`]. A task that is switched off gets
            both its head disabled and its weight zeroed, so a checkpoint's configuration records the
            ablation it was trained under.
        """
        return {
            "use_ctc_head": self.use_transcript_supervision,
            "use_spk_head": self.speaker_supervision,
            "vad_loss_weight": self.vad_loss_weight,
            "ctc_loss_weight": self.ctc_loss_weight if self.use_transcript_supervision else 0.0,
            "spk_loss_weight": self.spk_loss_weight if self.speaker_supervision else 0.0,
            "vad_loss_eval_frames_only": self.vad_loss_eval_frames_only,
        }

    def apply_to_config(self, config):
        r"""Write every implied field onto a configuration that has not been instantiated yet.

        Args:
            config ([`EmoTraceConfig`]):
                Configuration to update in place.

        Returns:
            [`EmoTraceConfig`]: the same object.
        """
        for key, value in self.config_overrides().items():
            setattr(config, key, value)
        return config

    def apply_to_model(self, model: nn.Module) -> None:
        r"""Write the loss weights onto an existing model and check that the required heads exist.

        The head switches are deliberately not applied here: the heads of an instantiated model are
        already allocated and are part of its checkpoint, so only the weights and the emotion-loss grid
        are updated.

        Args:
            model ([`EmoTraceForVADPrediction`]):
                Model whose configuration is updated in place.

        Raises:
            ValueError: when a task is supervised but its head is missing.
        """
        config = model.config
        overrides = self.config_overrides()
        for key in ("vad_loss_weight", "ctc_loss_weight", "spk_loss_weight", "vad_loss_eval_frames_only"):
            setattr(config, key, overrides[key])

        if self.use_transcript_supervision and getattr(model, "ctc_head", None) is None:
            raise ValueError(
                f"`transcript_supervision={self.transcript_supervision!r}` needs the ASR head, but the model was "
                "built with `config.use_ctc_head=False`. Build the model through "
                "`EmoTraceModelArguments.load_model(joint_args)` so the heads follow the ablation."
            )
        if self.speaker_supervision and getattr(model, "spk_head", None) is None:
            raise ValueError(
                "`speaker_supervision=True` needs the speaker head, but the model was built with "
                "`config.use_spk_head=False`. Build the model through "
                "`EmoTraceModelArguments.load_model(joint_args)` so the heads follow the ablation."
            )

        unsupervised = [
            name
            for name, supervised in (
                ("ctc_head", self.use_transcript_supervision),
                ("spk_head", self.speaker_supervision),
            )
            if not supervised and getattr(model, name, None) is not None
        ]
        if unsupervised:
            logger.warning(
                "The model carries %s while the corresponding task is switched off; the projection receives no "
                "gradient and is saved with its initial values.",
                " and ".join(unsupervised),
            )


@dataclass
class EmoTraceModelArguments:
    r"""Description of the EmoTrace model to train.

    Args:
        model_name_or_path (`str`, *optional*):
            Checkpoint of a complete EmoTrace model to continue training from. When omitted the heads are
            initialised from scratch on top of `backbone_name_or_path`.
        backbone_name_or_path (`str`, *optional*, defaults to `"microsoft/wavlm-large"`):
            Checkpoint of the encoder, normally the `CausalWavLMModel` produced by the distillation stage.
            The causal window is a property of that checkpoint and is read from its configuration.
        backbone_model_type (`str`, *optional*, defaults to `"causal_wavlm"`):
            Encoder class the checkpoint is loaded with. `"wavlm"` builds the bidirectional arm of the
            comparison: the same heads, the same trainer and the same data on the encoder the causal one
            is derived from. Released WavLM weights load into either class, so the choice is explicit and
            is never inferred from the checkpoint.
        vocab_size (`int`, *optional*, defaults to 52):
            Size of the CTC vocabulary, which must match the tokenizer that produced `ctc_labels`.
        head_dropout (`float`, *optional*, defaults to 0.0):
            Dropout applied to the frame representations before every head.
        spk_pos_weight (`list[float]`, *optional*):
            Positive-class weights of the speaker head, ordered as (speech, overlap, change).
        freeze_feature_encoder (`bool`, *optional*, defaults to `True`):
            Freeze the convolutional feature encoder of the backbone.
        freeze_backbone (`bool`, *optional*, defaults to `False`):
            Train the heads only, which measures the frozen representation rather than the whole model.
        dtype (`str`, *optional*):
            Weight dtype passed to the loader, for instance `"bfloat16"`.
        attn_implementation (`str`, *optional*):
            Attention implementation passed to the loader.
    """

    model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "EmoTrace checkpoint to continue training from."},
    )
    backbone_name_or_path: str = field(
        default=DEFAULT_WAVLM_NAME_OR_PATH,
        metadata={"help": "Checkpoint the encoder is initialised from."},
    )
    backbone_model_type: str = field(
        default=DEFAULT_BACKBONE_MODEL_TYPE,
        metadata={
            "help": "Encoder class the checkpoint is loaded with; 'wavlm' builds the bidirectional arm.",
            "choices": list(BACKBONE_MODEL_TYPES),
        },
    )
    vocab_size: int = field(default=52, metadata={"help": "Size of the CTC vocabulary."})
    head_dropout: float = field(default=0.0, metadata={"help": "Dropout applied before every head."})
    spk_pos_weight: list[float] | None = field(
        default=None,
        metadata={"help": "Positive-class weights of the speaker head: speech, overlap, change."},
    )
    freeze_feature_encoder: bool = field(
        default=True,
        metadata={"help": "Freeze the convolutional feature encoder of the backbone."},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={"help": "Freeze the whole backbone and train the heads only."},
    )
    dtype: str | None = field(default=None, metadata={"help": "Weight dtype, for instance bfloat16."})
    attn_implementation: str | None = field(
        default=None,
        metadata={"help": "Attention implementation, for instance sdpa."},
    )

    def load_model(self, joint_args: EmoTraceJointArguments | None = None):
        r"""Build the model these arguments describe.

        Args:
            joint_args ([`EmoTraceJointArguments`], *optional*):
                Ablation switches and loss weights, applied to the configuration before the model is
                instantiated so that the head of a disabled task is never allocated.

        Returns:
            [`EmoTraceForVADPrediction`]: the assembled model.
        """
        from ..models.EmoTrace import EmoTraceForVADPrediction

        config_kwargs: dict[str, Any] = {
            "vocab_size": self.vocab_size,
            "head_dropout": self.head_dropout,
        }
        if self.spk_pos_weight is not None:
            config_kwargs["spk_pos_weight"] = list(self.spk_pos_weight)
        if joint_args is not None:
            config_kwargs.update(joint_args.config_overrides())

        loader_kwargs: dict[str, Any] = {}
        if self.dtype is not None:
            loader_kwargs["dtype"] = self.dtype
        if self.attn_implementation is not None:
            loader_kwargs["attn_implementation"] = self.attn_implementation

        if self.model_name_or_path is not None:
            model = EmoTraceForVADPrediction.from_pretrained(self.model_name_or_path, **config_kwargs, **loader_kwargs)
        else:
            model = EmoTraceForVADPrediction.from_backbone_pretrained(
                self.backbone_name_or_path,
                backbone_model_type=self.backbone_model_type,
                **config_kwargs,
                **loader_kwargs,
            )

        if self.freeze_backbone:
            model.freeze_backbone()
        elif self.freeze_feature_encoder:
            model.freeze_feature_encoder()
        return model


class EmoTraceJointTrainer(Trainer):
    r"""Trains [`EmoTraceForVADPrediction`] on the emotion, transcript and speaker tasks at once.

    The weighted sum `vad_loss_weight * L_CCC + ctc_loss_weight * L_CTC + spk_loss_weight * L_BCE` is
    formed by the model, from the weights [`EmoTraceJointArguments`] writes onto its configuration; this
    trainer decides which terms exist. A task is switched off by removing its labels from the batch, so
    the corresponding term disappears from the loss instead of being multiplied by zero, and the ablation
    is enforced whatever the data collator emits.

    The unweighted value of every term is averaged over the logging interval on the local process and
    reported next to the total loss, under `loss_vad`, `loss_ctc` and `loss_spk` during training and with
    the `eval_` prefix during evaluation.

    Evaluation is set up for [`~ser_project.utils.evaluate.VADMetricComputer`]: unless
    `TrainingArguments.label_names` says otherwise, the labels forwarded to `compute_metrics` are
    `("vad_labels", "vad_eval_mask")`. Pass [`keep_vad_predictions`] as
    `preprocess_logits_for_metrics` to avoid accumulating the CTC and speaker logits over the split, and
    build the metric as `build_compute_metrics(label_pad_value=-100)`: the evaluation loop pads the
    gathered arrays with `-100` when the batches of a split do not all hold the same number of frames,
    and the padded positions have to be kept out of the coefficient.

    Args:
        model ([`EmoTraceForVADPrediction`]):
            The model to train.
        args (`TrainingArguments`, *optional*):
            Standard training arguments.
        joint_args ([`EmoTraceJointArguments`], *optional*):
            Loss weights and ablation switches. Defaults to [`EmoTraceJointArguments`].
        kwargs:
            Forwarded to `transformers.Trainer`.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        args: TrainingArguments | None = None,
        *,
        joint_args: EmoTraceJointArguments | None = None,
        **kwargs,
    ):
        super().__init__(model=model, args=args, **kwargs)

        self.joint_args = joint_args if joint_args is not None else EmoTraceJointArguments()
        self.joint_args.apply_to_model(self.accelerator.unwrap_model(self.model))

        # The three losses are already means over their own supervised positions, so the token-count
        # rescaling `Trainer` applies to models accepting loss kwargs does not apply.
        self.model_accepts_loss_kwargs = False
        if self.args.label_names is None:
            self.label_names = list(VAD_LABEL_NAMES)

        self._train_terms = _ScalarAccumulator()
        self._eval_terms = _ScalarAccumulator()

    def apply_ablation(self, inputs: dict[str, Any]) -> dict[str, Any]:
        r"""Return a copy of the batch holding only the supervision the ablation asks for.

        A batch may carry the two transcript flavours side by side, as `ctc_labels_verbatim` and
        `ctc_labels_intended` together with their `ctc_label_lengths_<mode>` and `ctc_loss_mask_<mode>`
        companions. The requested flavour is promoted to the plain keys the model reads and the others
        are dropped. A batch that carries only the plain keys is used as it is, since the collator was
        then configured for a single flavour.

        Args:
            inputs (`dict[str, Any]`):
                Collated batch.

        Returns:
            `dict[str, Any]`: the batch to pass to the model.
        """
        ablated = dict(inputs)
        mode_keys = [_mode_key(key, mode) for key in CTC_BATCH_KEYS for mode in TRANSCRIPT_MODES]

        if not self.joint_args.use_transcript_supervision:
            for key in (*CTC_BATCH_KEYS, *mode_keys):
                ablated.pop(key, None)
        else:
            mode = self.joint_args.transcript_mode
            if any(key in ablated for key in mode_keys) and _mode_key("ctc_labels", mode) not in ablated:
                raise KeyError(
                    f"The batch carries per-mode transcript columns but not {_mode_key('ctc_labels', mode)!r}, "
                    f"which `transcript_supervision={self.joint_args.transcript_supervision!r}` asks for; "
                    f"available keys: {sorted(ablated)}."
                )
            for key in CTC_BATCH_KEYS:
                source = _mode_key(key, mode)
                if source in ablated:
                    ablated[key] = ablated[source]
            for key in mode_keys:
                ablated.pop(key, None)

        if not self.joint_args.speaker_supervision:
            for key in SPEAKER_BATCH_KEYS:
                ablated.pop(key, None)

        return ablated

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        r"""Total loss of the tasks the ablation keeps.

        Args:
            model (`nn.Module`):
                The model, possibly wrapped for distributed training.
            inputs (`dict[str, Any]`):
                Collated batch.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Also return the model outputs.
            num_items_in_batch (`torch.Tensor` or `int`, *optional*):
                Unused; each task averages over its own supervised positions.

        Returns:
            `torch.Tensor` or `tuple[torch.Tensor, EmoTraceOutput]`: the loss, and the model outputs when
            `return_outputs` is set.
        """
        outputs = model(**self.apply_ablation(inputs))

        # A `ModelOutput` drops its unset fields, so the loss is fetched with `get` rather than indexed.
        loss = outputs.get("loss") if isinstance(outputs, Mapping) else outputs[0]
        if loss is None:
            raise ValueError(
                f"The model returned no loss for a batch with keys {sorted(inputs)}. At least one of "
                f"'vad_labels', 'ctc_labels' and 'spk_labels' must survive the ablation "
                f"(transcript_supervision={self.joint_args.transcript_supervision!r}, "
                f"speaker_supervision={self.joint_args.speaker_supervision})."
            )

        if self.joint_args.log_task_losses and isinstance(outputs, Mapping):
            loss_dict = outputs.get("loss_dict")
            if loss_dict:
                accumulator = self._train_terms if model.training else self._eval_terms
                accumulator.update(loss_dict)

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        r"""Add the per-task losses of the interval that just ended to the log entry.

        Args:
            logs (`dict[str, float]`):
                Values `Trainer` is about to report.
            start_time (`float`, *optional*):
                Start of the interval, forwarded unchanged.
        """
        is_evaluation = any(key.startswith("eval_") for key in logs)
        accumulator = self._eval_terms if is_evaluation else self._train_terms
        if accumulator:
            prefix = "eval_" if is_evaluation else ""
            logs = {**logs, **{f"{prefix}{key}": value for key, value in accumulator.consume().items()}}
        super().log(logs, start_time)


# ---------------------------------------------------------------------------
# Command line parsing
# ---------------------------------------------------------------------------


def _parse_dataclasses(parser: HfArgumentParser, args: Sequence[str] | None) -> tuple[Any, ...]:
    r"""Fill a parser's dataclasses from a command line or from a single configuration file.

    Args:
        parser (`HfArgumentParser`):
            Parser built from the dataclasses to fill.
        args (`Sequence[str]`, *optional*):
            Command line to read, defaulting to `sys.argv[1:]`. A single `.json`, `.yaml` or `.yml`
            argument is read as a configuration file instead.

    Returns:
        `tuple`: one instance per dataclass, in the order they were given to the parser.
    """
    argv = list(sys.argv[1:] if args is None else args)
    if len(argv) == 1:
        path = os.path.abspath(argv[0])
        if argv[0].endswith(".json"):
            return tuple(parser.parse_json_file(json_file=path))
        if argv[0].endswith((".yaml", ".yml")):
            return tuple(parser.parse_yaml_file(yaml_file=path))
    return tuple(parser.parse_args_into_dataclasses(args=argv))


def parse_distillation_arguments(
    args: Sequence[str] | None = None,
) -> tuple[DistillationArguments, TrainingArguments]:
    r"""Parse the arguments of the distillation stage.

    Args:
        args (`Sequence[str]`, *optional*):
            Command line to read, defaulting to `sys.argv[1:]`. A single `.json` or `.yaml` argument is
            read as a configuration file.

    Returns:
        `tuple[DistillationArguments, TrainingArguments]`
    """
    parser = HfArgumentParser((DistillationArguments, TrainingArguments))
    distillation_args, training_args = _parse_dataclasses(parser, args)
    return distillation_args, training_args


def parse_joint_arguments(
    args: Sequence[str] | None = None,
) -> tuple[EmoTraceModelArguments, EmoTraceJointArguments, TrainingArguments]:
    r"""Parse the arguments of the joint training stage.

    Args:
        args (`Sequence[str]`, *optional*):
            Command line to read, defaulting to `sys.argv[1:]`. A single `.json` or `.yaml` argument is
            read as a configuration file.

    Returns:
        `tuple[EmoTraceModelArguments, EmoTraceJointArguments, TrainingArguments]`
    """
    parser = HfArgumentParser((EmoTraceModelArguments, EmoTraceJointArguments, TrainingArguments))
    model_args, joint_args, training_args = _parse_dataclasses(parser, args)
    return model_args, joint_args, training_args


__all__ = [
    "CTC_BATCH_KEYS",
    "DEFAULT_WAVLM_NAME_OR_PATH",
    "DISTILLATION_OBJECTIVES",
    "SPEAKER_BATCH_KEYS",
    "TRANSCRIPT_MODES",
    "TRANSCRIPT_SUPERVISION_CHOICES",
    "VAD_LABEL_NAMES",
    "DistillationArguments",
    "DistillationTrainer",
    "EmoTraceJointArguments",
    "EmoTraceJointTrainer",
    "EmoTraceModelArguments",
    "keep_vad_predictions",
    "parse_distillation_arguments",
    "parse_joint_arguments",
]
