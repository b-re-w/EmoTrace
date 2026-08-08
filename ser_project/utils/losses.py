r"""Loss functions for the EmoTrace joint objective and for bidirectional-to-causal distillation.

Every loss in this module follows the masking convention of the collated batch: target
positions that carry no supervision are marked with `float("nan")` and may additionally be
removed through an explicit boolean mask. Invalid positions are neutralised with
`torch.where` before any arithmetic reaches them, so a `nan` target never contaminates the
forward value nor the backward pass.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor


__all__ = [
    "DistillationLossOutput",
    "ccc_loss",
    "concordance_correlation_coefficient",
    "hidden_state_distillation_loss",
    "masked_bce_loss",
    "masked_ctc_loss",
]


DEFAULT_EPS = 1e-8


def _check_reduction(reduction: str, allowed: Sequence[str]) -> None:
    if reduction not in allowed:
        raise ValueError(f"`reduction` must be one of {tuple(allowed)}, got {reduction!r}.")


def _accumulation_dtype(*tensors: Tensor) -> torch.dtype:
    r"""Return a dtype wide enough to accumulate sums over long sequences.

    Half precision cannot hold the running sums of a five minute conversation part, so
    `float16`/`bfloat16` inputs are promoted to `float32` while `float64` is kept.
    """
    dtype = torch.float32
    for tensor in tensors:
        dtype = torch.promote_types(dtype, tensor.dtype)
    return dtype


def _flatten_channels(tensor: Tensor) -> Tensor:
    r"""Flatten every leading dimension, keeping the last one as the channel dimension."""
    if tensor.dim() == 0:
        raise ValueError("Expected a tensor with at least one dimension.")
    if tensor.dim() == 1:
        return tensor.reshape(-1, 1)
    return tensor.reshape(-1, tensor.shape[-1])


def _valid_mask(targets: Tensor, mask: Optional[Tensor]) -> Tensor:
    r"""Combine the `nan` marking of `targets` with an optional external mask.

    Args:
        targets (`torch.Tensor`):
            Target tensor of shape `(..., num_channels)` where unlabeled entries are `nan`.
        mask (`torch.Tensor`, *optional*):
            Boolean tensor whose shape either matches `targets` or matches it without the
            channel dimension, in which case it is broadcast over the channels.

    Returns:
        `torch.BoolTensor`: mask of the same shape as `targets`, `True` where supervised.
    """
    valid = torch.isfinite(targets)
    if mask is None:
        return valid
    if mask.dim() == targets.dim() - 1:
        mask = mask.unsqueeze(-1)
    if mask.dim() != targets.dim():
        raise ValueError(
            f"`mask` with shape {tuple(mask.shape)} is not compatible with targets of shape "
            f"{tuple(targets.shape)}."
        )
    return valid & mask.to(torch.bool)


def _ccc_with_support(
    predictions: Tensor,
    targets: Tensor,
    mask: Optional[Tensor],
    eps: float,
) -> tuple[Tensor, Tensor]:
    r"""Per-channel concordance correlation coefficient together with its support size."""
    if predictions.shape != targets.shape:
        raise ValueError(
            f"`predictions` and `targets` must have the same shape, got "
            f"{tuple(predictions.shape)} and {tuple(targets.shape)}."
        )

    valid = _flatten_channels(_valid_mask(targets, mask))
    dtype = _accumulation_dtype(predictions, targets)
    flat_predictions = _flatten_channels(predictions).to(dtype)
    flat_targets = _flatten_channels(targets).to(dtype)

    zero = torch.zeros((), dtype=dtype, device=flat_predictions.device)
    x = torch.where(valid, flat_predictions, zero)
    y = torch.where(valid, flat_targets, zero)

    count = valid.sum(dim=0).to(dtype)
    safe_count = count.clamp(min=1.0)
    mean_x = x.sum(dim=0) / safe_count
    mean_y = y.sum(dim=0) / safe_count

    # Centering is applied after masking so that padded rows keep contributing exactly zero.
    dx = torch.where(valid, x - mean_x, zero)
    dy = torch.where(valid, y - mean_y, zero)
    var_x = dx.square().sum(dim=0) / safe_count
    var_y = dy.square().sum(dim=0) / safe_count
    covariance = (dx * dy).sum(dim=0) / safe_count

    ccc = 2.0 * covariance / (var_x + var_y + (mean_x - mean_y).square() + eps)
    ccc = torch.where(count >= 2.0, ccc, torch.zeros_like(ccc))
    return ccc, count


def concordance_correlation_coefficient(
    predictions: Tensor,
    targets: Tensor,
    mask: Optional[Tensor] = None,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    r"""Differentiable per-channel concordance correlation coefficient.

    The coefficient is `2 * cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y)) ** 2)` with
    population (biased) moments, pooled over every valid position of the tensor.

    Args:
        predictions (`torch.Tensor`):
            Predicted values of shape `(batch_size, sequence_length, num_channels)`. Tensors
            with more or fewer leading dimensions are accepted; the **last** dimension is
            always the channel dimension, so a single attribute must be passed as
            `(batch_size, sequence_length, 1)`.
        targets (`torch.Tensor`):
            Targets with the same shape as `predictions`, `nan` where unlabeled.
        mask (`torch.Tensor`, *optional*):
            Boolean mask selecting the positions to score, broadcast over channels when the
            channel dimension is omitted.
        eps (`float`, *optional*, defaults to `1e-8`):
            Constant added to the denominator for numerical stability.

    Returns:
        `torch.Tensor`: coefficients of shape `(num_channels,)`. Channels with fewer than two
        valid positions are reported as `0.0`.
    """
    ccc, _ = _ccc_with_support(predictions, targets, mask, eps)
    return ccc


def ccc_loss(
    predictions: Tensor,
    targets: Tensor,
    mask: Optional[Tensor] = None,
    channel_weights: Optional[Union[Sequence[float], Tensor]] = None,
    reduction: str = "mean",
    eps: float = DEFAULT_EPS,
) -> Tensor:
    r"""Concordance correlation coefficient loss, `1 - CCC`, computed over valid positions.

    The coefficient is pooled over the batch and the time axis of the whole mini-batch, one
    value per channel, which is the aggregation used when reporting CCC on time-continuous
    emotion traces. Channels whose support is smaller than two positions cannot define a
    coefficient and are dropped from the reduction instead of contributing a constant.

    Args:
        predictions (`torch.Tensor`):
            Predicted values of shape `(batch_size, sequence_length, num_channels)`; the last
            dimension is the channel dimension.
        targets (`torch.Tensor`):
            Targets with the same shape as `predictions`, `nan` at unlabeled positions.
        mask (`torch.Tensor`, *optional*):
            Boolean mask of shape `(batch_size, sequence_length)` or the full target shape,
            selecting the positions that take part in the loss.
        channel_weights (`Sequence[float]` or `torch.Tensor`, *optional*):
            Per-channel weights of shape `(num_channels,)` applied when reducing.
        reduction (`str`, *optional*, defaults to `"mean"`):
            One of `"mean"` (weighted mean over supported channels), `"sum"` or `"none"`
            (per-channel losses, zero for unsupported channels).
        eps (`float`, *optional*, defaults to `1e-8`):
            Constant added to the denominator of the coefficient.

    Returns:
        `torch.Tensor`: scalar loss, or a `(num_channels,)` tensor when `reduction="none"`.
            The returned tensor always stays connected to `predictions`, so a batch without
            any valid position yields a zero loss with a zero gradient rather than a
            detached constant.
    """
    _check_reduction(reduction, ("mean", "sum", "none"))
    ccc, count = _ccc_with_support(predictions, targets, mask, eps)

    supported = count >= 2.0
    per_channel = torch.where(supported, 1.0 - ccc, torch.zeros_like(ccc))
    if reduction == "none":
        return per_channel

    if channel_weights is None:
        weights = torch.ones_like(per_channel)
    else:
        weights = torch.as_tensor(channel_weights, dtype=per_channel.dtype, device=per_channel.device)
        if weights.shape != per_channel.shape:
            raise ValueError(
                f"`channel_weights` must have shape {tuple(per_channel.shape)}, got "
                f"{tuple(weights.shape)}."
            )
    weights = torch.where(supported, weights, torch.zeros_like(weights))

    total = (weights * per_channel).sum()
    if reduction == "sum":
        return total
    weight_sum = weights.sum()
    return total / torch.where(weight_sum > 0.0, weight_sum, torch.ones_like(weight_sum))


def masked_bce_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Optional[Tensor] = None,
    pos_weight: Optional[Union[float, Sequence[float], Tensor]] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""Binary cross entropy over supervised frames, with a configurable positive weight.

    This is the objective of the speaker head, whose three per-frame targets `(speech,
    overlap, change)` have very different densities: `change` is sparse enough that it needs a
    positive weight larger than one to avoid collapsing to the all-negative solution.

    Args:
        logits (`torch.Tensor`):
            Raw scores of shape `(batch_size, sequence_length, num_channels)`.
        targets (`torch.Tensor`):
            Binary targets with the same shape as `logits`, `nan` where unlabeled.
        mask (`torch.Tensor`, *optional*):
            Boolean mask of shape `(batch_size, sequence_length)` or the full target shape.
        pos_weight (`float`, `Sequence[float]` or `torch.Tensor`, *optional*):
            Weight of the positive class, either shared or one value per channel.
        reduction (`str`, *optional*, defaults to `"mean"`):
            One of `"mean"` (average over supervised entries, channels pooled), `"sum"` or
            `"none"` (per-entry losses, zero at unsupervised entries).

    Returns:
        `torch.Tensor`: scalar loss, or a tensor shaped like `logits` when `reduction="none"`.
    """
    _check_reduction(reduction, ("mean", "sum", "none"))
    if logits.shape != targets.shape:
        raise ValueError(
            f"`logits` and `targets` must have the same shape, got {tuple(logits.shape)} and "
            f"{tuple(targets.shape)}."
        )

    valid = _valid_mask(targets, mask)
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets)).to(logits.dtype)

    weight = None
    if pos_weight is not None:
        weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
        if weight.dim() > 1 or (weight.dim() == 1 and weight.shape[0] not in (1, logits.shape[-1])):
            raise ValueError(
                f"`pos_weight` must be a scalar or have shape ({logits.shape[-1]},), got "
                f"{tuple(weight.shape)}."
            )

    elementwise = F.binary_cross_entropy_with_logits(
        logits, safe_targets, reduction="none", pos_weight=weight
    )
    elementwise = torch.where(valid, elementwise, torch.zeros_like(elementwise))
    if reduction == "none":
        return elementwise

    dtype = _accumulation_dtype(elementwise)
    total = elementwise.to(dtype).sum()
    if reduction == "sum":
        return total
    return total / valid.sum().to(dtype).clamp(min=1.0)


def masked_ctc_loss(
    logits: Tensor,
    labels: Tensor,
    input_lengths: Tensor,
    label_lengths: Tensor,
    loss_mask: Optional[Tensor] = None,
    blank: int = 0,
    zero_infinity: bool = True,
    reduction: str = "mean",
) -> Tensor:
    r"""CTC loss restricted to the samples whose transcript supervision is usable.

    Per-sample losses are normalised by the target length, matching the semantics of
    `torch.nn.functional.ctc_loss(reduction="mean")`, and then averaged over the samples kept
    by `loss_mask`. Samples covered by unreliable pseudo transcripts (overlapped speech, for
    instance) are removed by the mask rather than by shrinking the batch, so the batch layout
    stays static.

    Args:
        logits (`torch.Tensor`):
            Head outputs of shape `(batch_size, sequence_length, vocab_size)`; log softmax is
            applied internally.
        labels (`torch.LongTensor`):
            Padded target sequences of shape `(batch_size, max_label_length)`. Padding may be
            any negative value, it is never read.
        input_lengths (`torch.LongTensor`):
            Number of valid frames per sample, shape `(batch_size,)`.
        label_lengths (`torch.LongTensor`):
            Number of valid target tokens per sample, shape `(batch_size,)`.
        loss_mask (`torch.BoolTensor`, *optional*):
            Shape `(batch_size,)`, `True` for the samples that contribute to the loss.
        blank (`int`, *optional*, defaults to 0):
            Index of the blank symbol.
        zero_infinity (`bool`, *optional*, defaults to `True`):
            Zero out infinite losses, which occur when a target is longer than the frame
            sequence that should emit it.
        reduction (`str`, *optional*, defaults to `"mean"`):
            One of `"mean"`, `"sum"` or `"none"` (per-sample losses, zero where masked out).

    Returns:
        `torch.Tensor`: scalar loss, or a `(batch_size,)` tensor when `reduction="none"`.
    """
    _check_reduction(reduction, ("mean", "sum", "none"))
    log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
    targets = labels.masked_fill(labels < 0, blank).to(torch.long)

    per_sample = F.ctc_loss(
        log_probs,
        targets,
        input_lengths.to(torch.long),
        label_lengths.to(torch.long),
        blank=blank,
        reduction="none",
        zero_infinity=zero_infinity,
    )
    per_sample = per_sample / label_lengths.to(per_sample.dtype).clamp(min=1.0)

    if loss_mask is None:
        keep = torch.ones_like(per_sample, dtype=torch.bool)
    else:
        keep = loss_mask.to(torch.bool)
    keep = keep & (label_lengths > 0)
    per_sample = torch.where(keep, per_sample, torch.zeros_like(per_sample))

    if reduction == "none":
        return per_sample
    total = per_sample.sum()
    if reduction == "sum":
        return total
    return total / keep.sum().to(per_sample.dtype).clamp(min=1.0)


class DistillationLossOutput(NamedTuple):
    r"""Output of [`hidden_state_distillation_loss`].

    Args:
        loss (`torch.Tensor`):
            Scalar loss aggregated over the selected layers.
        layer_losses (`Dict[int, torch.Tensor]`):
            Detached scalar loss per selected student layer, keyed by the layer index in the
            student `hidden_states` tuple (`0` is the output of the feature projection). The
            per-layer values are the distillation residual: what remains after convergence is
            the part of the teacher representation that cannot be recovered from past context.
    """

    loss: Tensor
    layer_losses: Dict[int, Tensor]


def _masked_frame_mean(per_frame: Tensor, mask: Optional[Tensor]) -> Tensor:
    r"""Average a `(batch_size, sequence_length)` tensor over the frames selected by `mask`."""
    if mask is None:
        return per_frame.mean()
    mask = mask.to(torch.bool)
    if mask.shape != per_frame.shape:
        raise ValueError(
            f"`attention_mask` must have shape {tuple(per_frame.shape)}, got {tuple(mask.shape)}."
        )
    per_frame = torch.where(mask, per_frame, torch.zeros_like(per_frame))
    return per_frame.sum() / mask.sum().to(per_frame.dtype).clamp(min=1.0)


def _normalize_layer_indices(indices: Sequence[int], num_layers: int, name: str) -> list[int]:
    normalized = []
    for index in indices:
        resolved = index + num_layers if index < 0 else index
        if not 0 <= resolved < num_layers:
            raise IndexError(
                f"{name} {index} is out of range for a hidden state tuple of length {num_layers}."
            )
        normalized.append(resolved)
    return normalized


def hidden_state_distillation_loss(
    student_hidden_states: Sequence[Tensor],
    teacher_hidden_states: Sequence[Tensor],
    layer_indices: Optional[Sequence[int]] = None,
    teacher_layer_indices: Optional[Sequence[int]] = None,
    attention_mask: Optional[Tensor] = None,
    objective: str = "cosine",
    layer_weights: Optional[Sequence[float]] = None,
    reduction: str = "mean",
) -> DistillationLossOutput:
    r"""Match the per-frame hidden states of a causal student to a bidirectional teacher.

    Both models run on the same audio and produce one hidden state per 20 ms frame, so the
    layers are compared frame by frame without any alignment step. The teacher states are
    detached inside this function; the caller does not have to do it.

    Args:
        student_hidden_states (`Sequence[torch.Tensor]`):
            Hidden states of the student as returned with `output_hidden_states=True`, each of
            shape `(batch_size, sequence_length, hidden_size)`. Index `0` is the output of the
            feature projection, index `i > 0` the output of encoder layer `i`.
        teacher_hidden_states (`Sequence[torch.Tensor]`):
            Hidden states of the teacher, same layout and same frame rate.
        layer_indices (`Sequence[int]`, *optional*):
            Student layers to supervise; negative indices count from the end. Defaults to every
            layer of the tuple.
        teacher_layer_indices (`Sequence[int]`, *optional*):
            Teacher layers paired with `layer_indices`, in the same order. Defaults to
            `layer_indices`, which is the correct pairing when student and teacher share the
            same depth.
        attention_mask (`torch.Tensor`, *optional*):
            Frame-level mask of shape `(batch_size, sequence_length)`, `True`/`1` on real
            frames. Padded frames must be excluded or they dominate the residual.
        objective (`str`, *optional*, defaults to `"cosine"`):
            `"cosine"` for `1 - cos(student, teacher)`, which is invariant to the norm drift
            between the two models, or `"mse"` for the mean squared error over hidden units.
        layer_weights (`Sequence[float]`, *optional*):
            One weight per selected layer, defaults to uniform weights.
        reduction (`str`, *optional*, defaults to `"mean"`):
            `"mean"` for the weighted mean over the selected layers, `"sum"` for the weighted
            sum.

    Returns:
        [`DistillationLossOutput`]: the aggregated loss and the detached per-layer residuals.
    """
    if objective not in ("cosine", "mse"):
        raise ValueError(f"`objective` must be 'cosine' or 'mse', got {objective!r}.")
    _check_reduction(reduction, ("mean", "sum"))
    if len(student_hidden_states) == 0 or len(teacher_hidden_states) == 0:
        raise ValueError("`student_hidden_states` and `teacher_hidden_states` must not be empty.")

    if layer_indices is None:
        layer_indices = range(len(student_hidden_states))
    student_layers = _normalize_layer_indices(
        list(layer_indices), len(student_hidden_states), "Student layer index"
    )
    if teacher_layer_indices is None:
        teacher_layers = _normalize_layer_indices(
            student_layers, len(teacher_hidden_states), "Teacher layer index"
        )
    else:
        teacher_layers = _normalize_layer_indices(
            list(teacher_layer_indices), len(teacher_hidden_states), "Teacher layer index"
        )
    if len(student_layers) != len(teacher_layers):
        raise ValueError(
            f"`layer_indices` and `teacher_layer_indices` must have the same length, got "
            f"{len(student_layers)} and {len(teacher_layers)}."
        )

    if layer_weights is None:
        weights = [1.0] * len(student_layers)
    else:
        weights = list(layer_weights)
        if len(weights) != len(student_layers):
            raise ValueError(
                f"`layer_weights` must provide one weight per selected layer "
                f"({len(student_layers)}), got {len(weights)}."
            )

    total = None
    layer_losses: Dict[int, Tensor] = {}
    for student_layer, teacher_layer, weight in zip(student_layers, teacher_layers, weights):
        student = student_hidden_states[student_layer]
        teacher = teacher_hidden_states[teacher_layer].detach()
        if student.shape != teacher.shape:
            raise ValueError(
                f"Student layer {student_layer} has shape {tuple(student.shape)} while teacher "
                f"layer {teacher_layer} has shape {tuple(teacher.shape)}."
            )
        dtype = _accumulation_dtype(student, teacher)
        student = student.to(dtype)
        teacher = teacher.to(dtype)

        if objective == "cosine":
            per_frame = 1.0 - F.cosine_similarity(student, teacher, dim=-1, eps=DEFAULT_EPS)
        else:
            per_frame = (student - teacher).square().mean(dim=-1)

        layer_loss = _masked_frame_mean(per_frame, attention_mask)
        layer_losses[student_layer] = layer_loss.detach()
        contribution = weight * layer_loss
        total = contribution if total is None else total + contribution

    weight_sum = sum(weights)
    if reduction == "mean" and weight_sum != 0.0:
        total = total / weight_sum
    return DistillationLossOutput(loss=total, layer_losses=layer_losses)


def _reference_ccc(predictions, targets) -> float:
    r"""Direct textbook CCC on two flat sequences, used only by the self-check."""
    import statistics

    n = len(predictions)
    mean_x = statistics.fmean(predictions)
    mean_y = statistics.fmean(targets)
    var_x = statistics.fmean([(value - mean_x) ** 2 for value in predictions])
    var_y = statistics.fmean([(value - mean_y) ** 2 for value in targets])
    covariance = statistics.fmean(
        [(predictions[i] - mean_x) * (targets[i] - mean_y) for i in range(n)]
    )
    return 2.0 * covariance / (var_x + var_y + (mean_x - mean_y) ** 2)


def _self_check() -> None:
    r"""Validate the losses against direct formulas on random tensors."""
    torch.manual_seed(0)

    predictions = torch.randn(4, 50, 3, dtype=torch.float64)
    targets = 0.6 * predictions + 0.4 * torch.randn(4, 50, 3, dtype=torch.float64)
    mask = torch.rand(4, 50) > 0.3
    targets = targets.masked_fill(torch.rand(4, 50, 3) > 0.8, float("nan"))

    # `eps` is switched off so that the comparison against the textbook formula is exact
    # rather than accurate to the size of the stabiliser.
    ccc = concordance_correlation_coefficient(predictions, targets, mask, eps=0.0)
    for channel in range(3):
        valid = (mask.unsqueeze(-1) & torch.isfinite(targets))[..., channel].reshape(-1)
        flat_predictions = predictions[..., channel].reshape(-1)[valid].tolist()
        flat_targets = targets[..., channel].reshape(-1)[valid].tolist()
        expected = _reference_ccc(flat_predictions, flat_targets)
        assert abs(ccc[channel].item() - expected) < 1e-12, (channel, ccc[channel].item(), expected)

    loss = ccc_loss(predictions, targets, mask, eps=0.0)
    assert abs(loss.item() - (1.0 - ccc).mean().item()) < 1e-12
    stabilised = ccc_loss(predictions, targets, mask, reduction="none")
    assert torch.allclose(stabilised, 1.0 - ccc, atol=1e-7)
    weighted = ccc_loss(predictions, targets, mask, channel_weights=[1.0, 0.0, 0.0], eps=0.0)
    assert abs(weighted.item() - (1.0 - ccc[0]).item()) < 1e-12

    # A perfect prediction must reach a zero loss, and the loss must stay differentiable and
    # finite even when every target of a channel is unlabeled.
    perfect = torch.randn(2, 20, 3)
    assert ccc_loss(perfect, perfect.clone()).abs().item() < 1e-5
    leaky = torch.randn(2, 20, 3, requires_grad=True)
    holed = torch.randn(2, 20, 3)
    holed[..., 1] = float("nan")
    holed[0, :5] = float("nan")
    value = ccc_loss(leaky, holed)
    value.backward()
    assert torch.isfinite(value).all() and torch.isfinite(leaky.grad).all()
    empty = torch.randn(2, 20, 3, requires_grad=True)
    all_nan = torch.full((2, 20, 3), float("nan"))
    zero_loss = ccc_loss(empty, all_nan)
    zero_loss.backward()
    assert zero_loss.item() == 0.0 and torch.isfinite(empty.grad).all()

    logits = torch.randn(3, 12, 3, dtype=torch.float64)
    binary_targets = (torch.rand(3, 12, 3) > 0.7).to(torch.float64)
    binary_targets[0, 0, 0] = float("nan")
    frame_mask = torch.rand(3, 12) > 0.2
    weight = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float64)
    bce = masked_bce_loss(logits, binary_targets, frame_mask, pos_weight=weight)
    valid = frame_mask.unsqueeze(-1) & torch.isfinite(binary_targets)
    reference_total, reference_count = 0.0, 0
    for b in range(3):
        for t in range(12):
            for c in range(3):
                if not valid[b, t, c]:
                    continue
                z = logits[b, t, c].item()
                y = binary_targets[b, t, c].item()
                log_sigmoid = -torch.nn.functional.softplus(torch.tensor(-z, dtype=torch.float64))
                log_sigmoid_neg = -torch.nn.functional.softplus(torch.tensor(z, dtype=torch.float64))
                reference_total += -(
                    weight[c].item() * y * log_sigmoid.item() + (1.0 - y) * log_sigmoid_neg.item()
                )
                reference_count += 1
    assert abs(bce.item() - reference_total / reference_count) < 1e-9

    student = [torch.randn(2, 15, 8) for _ in range(4)]
    teacher = [torch.randn(2, 15, 8) for _ in range(4)]
    padding = torch.ones(2, 15, dtype=torch.bool)
    padding[1, 10:] = False
    output = hidden_state_distillation_loss(
        student, teacher, layer_indices=[1, -1], attention_mask=padding, objective="cosine"
    )
    assert set(output.layer_losses) == {1, 3}
    manual = []
    for index in (1, 3):
        similarity = F.cosine_similarity(student[index], teacher[index], dim=-1)
        residual = (1.0 - similarity)[padding].mean()
        manual.append(residual)
        assert abs(output.layer_losses[index].item() - residual.item()) < 1e-6
    assert abs(output.loss.item() - torch.stack(manual).mean().item()) < 1e-6

    mse_output = hidden_state_distillation_loss(
        student, teacher, layer_indices=[2], objective="mse", reduction="sum"
    )
    expected_mse = (student[2] - teacher[2]).square().mean()
    assert abs(mse_output.loss.item() - expected_mse.item()) < 1e-6

    vocab = 7
    ctc_logits = torch.randn(3, 40, vocab, requires_grad=True)
    ctc_labels = torch.full((3, 6), -100, dtype=torch.long)
    ctc_labels[:, :4] = torch.randint(1, vocab, (3, 4))
    label_lengths = torch.tensor([4, 4, 4])
    input_lengths = torch.tensor([40, 40, 40])
    keep = torch.tensor([True, False, True])
    masked = masked_ctc_loss(ctc_logits, ctc_labels, input_lengths, label_lengths, keep)
    per_sample = masked_ctc_loss(
        ctc_logits, ctc_labels, input_lengths, label_lengths, keep, reduction="none"
    )
    assert per_sample[1].item() == 0.0
    assert abs(masked.item() - (per_sample[0] + per_sample[2]).item() / 2.0) < 1e-6
    masked.backward()
    assert torch.isfinite(ctc_logits.grad).all()

    print("ser_project.utils.losses self-check passed")


if __name__ == "__main__":
    _self_check()
