r"""The concordance correlation coefficient and the loss built on it.

The coefficient is the reported metric of the whole project, so it is checked against a direct
transcription of its definition rather than against another implementation:

    CCC = 2 cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y)) ** 2)

with population moments, pooled over every supervised position and computed one channel at a time.
The tests also pin down what "supervised" means, since the batch marks unlabeled positions with
`float('nan')` and may narrow them further with an evaluation mask: an ignored position must leave
the coefficient exactly where it was, whatever value the prediction takes there.
"""

import math

import numpy as np
import pytest
import torch

from ser_project.utils.evaluate import concordance_correlation_coefficient as numpy_ccc
from ser_project.utils.losses import ccc_loss, concordance_correlation_coefficient


def reference_ccc(predictions, targets) -> float:
    r"""Textbook concordance correlation coefficient of two equally long sequences.

    Args:
        predictions (`Sequence[float]`), targets (`Sequence[float]`):
            Paired observations.

    Returns:
        `float`: the coefficient, with population (biased) moments.
    """
    count = len(predictions)
    mean_x = math.fsum(predictions) / count
    mean_y = math.fsum(targets) / count
    var_x = math.fsum((value - mean_x) ** 2 for value in predictions) / count
    var_y = math.fsum((value - mean_y) ** 2 for value in targets) / count
    covariance = math.fsum(
        (predictions[index] - mean_x) * (targets[index] - mean_y) for index in range(count)
    ) / count
    return 2.0 * covariance / (var_x + var_y + (mean_x - mean_y) ** 2)


def reference_pearson(predictions, targets) -> float:
    r"""Textbook Pearson correlation of two equally long sequences."""
    count = len(predictions)
    mean_x = math.fsum(predictions) / count
    mean_y = math.fsum(targets) / count
    var_x = math.fsum((value - mean_x) ** 2 for value in predictions) / count
    var_y = math.fsum((value - mean_y) ** 2 for value in targets) / count
    covariance = math.fsum(
        (predictions[index] - mean_x) * (targets[index] - mean_y) for index in range(count)
    ) / count
    return covariance / math.sqrt(var_x * var_y)


@pytest.fixture
def traces(generator):
    r"""Returns correlated predictions and targets with holes, plus an evaluation mask.

    Returns:
        `tuple`: predictions `(4, 60, 3)`, targets `(4, 60, 3)` with `nan` holes, and a frame mask
        of shape `(4, 60)`, all in double precision so the comparison against the reference formula
        is limited by the formula and not by the accumulation.
    """
    predictions = torch.randn(4, 60, 3, generator=generator, dtype=torch.float64)
    targets = 0.6 * predictions + 0.4 * torch.randn(4, 60, 3, generator=generator, dtype=torch.float64)
    targets = targets.masked_fill(torch.rand(4, 60, 3, generator=generator) > 0.85, float("nan"))
    mask = torch.rand(4, 60, generator=generator) > 0.25
    return predictions, targets, mask


def valid_pairs(predictions, targets, mask, channel):
    r"""Returns the supervised `(prediction, target)` pairs of one channel as Python lists."""
    valid = (mask.unsqueeze(-1) & torch.isfinite(targets))[..., channel].reshape(-1)
    return (
        predictions[..., channel].reshape(-1)[valid].tolist(),
        targets[..., channel].reshape(-1)[valid].tolist(),
    )


# ---------------------------------------------------------------------------
# The definition
# ---------------------------------------------------------------------------


def test_matches_the_textbook_formula(traces):
    r"""Each channel reproduces the definition on the positions the mask and the holes leave."""
    predictions, targets, mask = traces
    coefficients = concordance_correlation_coefficient(predictions, targets, mask, eps=0.0)

    assert coefficients.shape == (3,)
    for channel in range(3):
        expected = reference_ccc(*valid_pairs(predictions, targets, mask, channel))
        assert coefficients[channel].item() == pytest.approx(expected, abs=1e-12)


def test_loss_is_one_minus_the_coefficient(traces):
    r"""`ccc_loss` is `1 - CCC`, reduced over the channels."""
    predictions, targets, mask = traces
    coefficients = concordance_correlation_coefficient(predictions, targets, mask, eps=0.0)

    per_channel = ccc_loss(predictions, targets, mask, reduction="none", eps=0.0)
    torch.testing.assert_close(per_channel, 1.0 - coefficients, rtol=0, atol=1e-12)

    mean_loss = ccc_loss(predictions, targets, mask, eps=0.0)
    assert mean_loss.item() == pytest.approx((1.0 - coefficients).mean().item(), abs=1e-12)

    sum_loss = ccc_loss(predictions, targets, mask, reduction="sum", eps=0.0)
    assert sum_loss.item() == pytest.approx((1.0 - coefficients).sum().item(), abs=1e-12)


def test_channel_weights_select_and_weight_channels(traces):
    r"""Channel weights reduce a weighted mean, and a zero weight removes a channel entirely."""
    predictions, targets, mask = traces
    per_channel = ccc_loss(predictions, targets, mask, reduction="none", eps=0.0)

    valence_only = ccc_loss(
        predictions, targets, mask, channel_weights=[1.0, 0.0, 0.0], eps=0.0
    )
    assert valence_only.item() == pytest.approx(per_channel[0].item(), abs=1e-12)

    weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    weighted = ccc_loss(predictions, targets, mask, channel_weights=weights, eps=0.0)
    expected = (weights * per_channel).sum() / weights.sum()
    assert weighted.item() == pytest.approx(expected.item(), abs=1e-12)


def test_perfect_prediction_scores_one(generator):
    r"""A prediction equal to the target reaches the maximum of the coefficient."""
    targets = torch.randn(2, 40, 3, generator=generator, dtype=torch.float64)
    coefficients = concordance_correlation_coefficient(targets.clone(), targets, eps=0.0)

    torch.testing.assert_close(coefficients, torch.ones(3, dtype=torch.float64), rtol=0, atol=1e-12)
    assert ccc_loss(targets.clone(), targets, eps=0.0).item() == pytest.approx(0.0, abs=1e-12)


def test_bias_and_scale_are_penalised_but_correlation_is_not(generator):
    r"""The coefficient never exceeds the Pearson correlation, and only agreement reaches it.

    A prediction that is a shifted or rescaled copy of the target correlates perfectly yet does not
    agree with it, which is the entire reason the traces are scored with CCC and not with `r`.
    """
    targets = torch.randn(1, 200, 1, generator=generator, dtype=torch.float64)
    shifted = targets + 0.5
    scaled = 2.0 * targets

    flat_targets = targets.reshape(-1).tolist()
    for prediction in (shifted, scaled):
        coefficient = concordance_correlation_coefficient(prediction, targets, eps=0.0)[0].item()
        pearson = reference_pearson(prediction.reshape(-1).tolist(), flat_targets)
        assert pearson == pytest.approx(1.0, abs=1e-12)
        assert coefficient < pearson
        assert coefficient == pytest.approx(
            reference_ccc(prediction.reshape(-1).tolist(), flat_targets), abs=1e-12
        )


def test_pooling_is_invariant_to_the_frame_order(traces, generator):
    r"""Frames are pooled, so shuffling them cannot move the coefficient."""
    predictions, targets, mask = traces
    permutation = torch.randperm(predictions.shape[1], generator=generator)

    original = concordance_correlation_coefficient(predictions, targets, mask, eps=0.0)
    shuffled = concordance_correlation_coefficient(
        predictions[:, permutation], targets[:, permutation], mask[:, permutation], eps=0.0
    )
    torch.testing.assert_close(original, shuffled, rtol=0, atol=1e-12)


def test_channels_are_scored_independently(traces, generator):
    r"""Corrupting one attribute leaves the coefficients of the other two untouched."""
    predictions, targets, mask = traces
    corrupted = predictions.clone()
    corrupted[..., 1] = 1000.0 * torch.randn(4, 60, generator=generator, dtype=torch.float64)

    original = concordance_correlation_coefficient(predictions, targets, mask, eps=0.0)
    changed = concordance_correlation_coefficient(corrupted, targets, mask, eps=0.0)

    torch.testing.assert_close(original[[0, 2]], changed[[0, 2]], rtol=0, atol=1e-12)
    assert changed[1].item() != original[1].item()


# ---------------------------------------------------------------------------
# What counts as a supervised position
# ---------------------------------------------------------------------------


def test_unlabeled_positions_are_ignored_whatever_is_predicted_there(traces):
    r"""Frames whose target is `nan` contribute nothing, even with an absurd prediction on them."""
    predictions, targets, mask = traces
    holes = ~torch.isfinite(targets)
    assert holes.any()

    poisoned = predictions.clone()
    poisoned[holes] = 1e6

    original = concordance_correlation_coefficient(predictions, targets, mask, eps=0.0)
    with_poison = concordance_correlation_coefficient(poisoned, targets, mask, eps=0.0)
    torch.testing.assert_close(original, with_poison, rtol=0, atol=1e-12)
    assert torch.isfinite(with_poison).all()


def test_masked_positions_are_ignored_whatever_is_predicted_there(traces):
    r"""The evaluation mask removes positions as firmly as a `nan` target does."""
    predictions, targets, mask = traces
    poisoned = predictions.clone()
    poisoned[~mask] = -1e6

    original = concordance_correlation_coefficient(predictions, targets, mask, eps=0.0)
    with_poison = concordance_correlation_coefficient(poisoned, targets, mask, eps=0.0)
    torch.testing.assert_close(original, with_poison, rtol=0, atol=1e-12)


def test_frame_mask_broadcasts_over_the_channels(traces):
    r"""A `(batch, frames)` mask means the same as the `(batch, frames, 3)` mask it expands to."""
    predictions, targets, mask = traces
    broadcast = concordance_correlation_coefficient(predictions, targets, mask, eps=0.0)
    expanded = concordance_correlation_coefficient(
        predictions, targets, mask.unsqueeze(-1).expand_as(targets), eps=0.0
    )
    integer = concordance_correlation_coefficient(predictions, targets, mask.long(), eps=0.0)

    torch.testing.assert_close(broadcast, expanded, rtol=0, atol=1e-12)
    torch.testing.assert_close(broadcast, integer, rtol=0, atol=1e-12)


def test_channel_without_support_is_dropped_from_the_reduction(generator):
    r"""A channel with fewer than two supervised frames cannot define a coefficient."""
    predictions = torch.randn(1, 30, 3, generator=generator, dtype=torch.float64)
    targets = predictions + 0.1
    targets[..., 2] = float("nan")
    targets[0, 0, 2] = 0.0

    per_channel = ccc_loss(predictions, targets, reduction="none", eps=0.0)
    assert per_channel[2].item() == 0.0

    mean_loss = ccc_loss(predictions, targets, eps=0.0)
    assert mean_loss.item() == pytest.approx(per_channel[:2].mean().item(), abs=1e-12)


def test_a_batch_without_any_label_yields_a_differentiable_zero(generator):
    r"""An all-`nan` batch must not produce `nan` gradients that would poison the optimizer."""
    predictions = torch.randn(2, 20, 3, generator=generator, requires_grad=True)
    targets = torch.full((2, 20, 3), float("nan"))

    loss = ccc_loss(predictions, targets)
    loss.backward()

    assert loss.item() == 0.0
    assert loss.requires_grad
    assert torch.isfinite(predictions.grad).all()


def test_gradients_flow_through_the_supervised_positions_only(generator):
    r"""Holes in the targets neither block nor corrupt the gradient of the other frames."""
    predictions = torch.randn(2, 30, 3, generator=generator, requires_grad=True)
    targets = torch.randn(2, 30, 3, generator=generator)
    targets[0, :5] = float("nan")
    targets[..., 1] = float("nan")

    loss = ccc_loss(predictions, targets)
    loss.backward()

    assert torch.isfinite(loss).all()
    assert torch.isfinite(predictions.grad).all()
    assert torch.equal(predictions.grad[..., 1], torch.zeros(2, 30))
    assert torch.equal(predictions.grad[0, :5, 0], torch.zeros(5))
    assert predictions.grad[1, :, 0].abs().sum() > 0.0


# ---------------------------------------------------------------------------
# Numerics and interface
# ---------------------------------------------------------------------------


def test_accumulation_survives_half_precision_inputs(generator):
    r"""Long half-precision sequences are accumulated in a wider dtype, not in `float16`."""
    predictions = torch.randn(1, 8000, 3, generator=generator).half()
    targets = (0.7 * predictions.float() + 0.3 * torch.randn(1, 8000, 3, generator=generator)).half()

    half = concordance_correlation_coefficient(predictions, targets, eps=0.0)
    exact = concordance_correlation_coefficient(predictions.double(), targets.double(), eps=0.0)

    assert torch.isfinite(half).all()
    torch.testing.assert_close(half.double(), exact, rtol=0, atol=1e-3)


def test_agrees_with_the_numpy_evaluation_metric(traces):
    r"""The training loss and the reported metric must score a batch identically."""
    predictions, targets, mask = traces
    torch_side = concordance_correlation_coefficient(predictions, targets, mask, eps=0.0)
    numpy_side = numpy_ccc(predictions.numpy(), targets.numpy(), mask.numpy())

    np.testing.assert_allclose(torch_side.numpy(), numpy_side, rtol=0, atol=1e-12)


def test_invalid_arguments_are_rejected(traces):
    r"""Shape and option mistakes fail loudly instead of scoring the wrong positions."""
    predictions, targets, mask = traces

    with pytest.raises(ValueError, match="same shape"):
        concordance_correlation_coefficient(predictions[:, :10], targets)
    with pytest.raises(ValueError, match="not compatible"):
        concordance_correlation_coefficient(predictions, targets, mask[..., None, None])
    with pytest.raises(ValueError, match="reduction"):
        ccc_loss(predictions, targets, reduction="median")
    with pytest.raises(ValueError, match="channel_weights"):
        ccc_loss(predictions, targets, channel_weights=[1.0, 1.0])
