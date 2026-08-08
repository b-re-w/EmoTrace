r"""Evaluation metrics for time-continuous valence/arousal/dominance prediction.

The reported quantity is the concordance correlation coefficient (CCC) between the predicted
traces and the annotator consensus traces. Two aggregations are produced side by side:

- **micro**: every evaluation frame of the corpus is pooled into a single coefficient per
  attribute. This is the number that is comparable with published MSP-Conversation results.
- **macro**: one coefficient per conversation part, averaged over parts without weighting, so
  that long parts cannot hide a poor fit on short ones.

The model predicts on the 20 ms grid, but only the frames flagged by `vad_eval_mask` are
scored. Those frames coincide with the positions of the released 100 ms traces, which keeps
the metric on the same grid as the published baselines instead of rewarding the model for the
extra temporal density it produces.

The module is deliberately free of any deep learning dependency: it consumes NumPy arrays and
accumulates in `float64`, so evaluation is unaffected by the mixed precision used in training.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np


__all__ = [
    "DEFAULT_ATTRIBUTE_NAMES",
    "PartwiseCCC",
    "build_compute_metrics",
    "ccc_by_part",
    "compute_metrics",
    "compute_vad_metrics",
    "concordance_correlation_coefficient",
    "VADMetricComputer",
]


DEFAULT_ATTRIBUTE_NAMES = ("valence", "arousal", "dominance")

DEFAULT_LABEL_NAMES = ("vad_labels", "vad_eval_mask")

VAD_LABELS_KEY = "vad_labels"
VAD_EVAL_MASK_KEY = "vad_eval_mask"
PART_IDS_KEY = "part_ids"


def _as_bool_mask(mask: Any) -> np.ndarray:
    r"""Interpret a mask array as booleans.

    `Trainer` pads gathered arrays with `-100` when batches have different lengths, so a mask
    that arrives as an integer array carries `-100` on padded frames. Comparing against zero
    maps both `False` and that padding to `False`.
    """
    mask = np.asarray(mask)
    if mask.dtype == np.bool_:
        return mask
    return mask > 0


def concordance_correlation_coefficient(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Per-attribute concordance correlation coefficient over the selected positions.

    The coefficient is `2 * cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y)) ** 2)` with
    population moments. It is invariant to the order of the frames, so predictions coming from
    several segments of the same conversation part can simply be pooled.

    Args:
        predictions (`np.ndarray`):
            Predicted values of shape `(..., num_attributes)`.
        targets (`np.ndarray`):
            Reference values with the same shape as `predictions`, `nan` where unlabeled.
        mask (`np.ndarray`, *optional*):
            Mask of shape `(...)` or `(..., num_attributes)` selecting the positions to score.

    A position is scored when its target is finite and the mask selects it. Predictions are
    deliberately not screened for finiteness: a `nan` produced at a scored position propagates
    to the coefficient, exactly as it would to the training loss, instead of being quietly
    dropped from the average.

    Returns:
        `np.ndarray`: coefficients of shape `(num_attributes,)`, `nan` for attributes with
        fewer than two valid positions or with a degenerate (zero variance on both sides)
        support.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"`predictions` and `targets` must have the same shape, got {predictions.shape} and "
            f"{targets.shape}."
        )

    valid = np.isfinite(targets)
    if mask is not None:
        mask = _as_bool_mask(mask)
        if mask.shape == targets.shape[:-1]:
            mask = mask[..., None]
        if mask.shape != targets.shape and mask.shape[-1] != 1:
            raise ValueError(
                f"`mask` with shape {mask.shape} is not compatible with targets of shape "
                f"{targets.shape}."
            )
        valid = valid & mask

    num_attributes = predictions.shape[-1]
    valid = valid.reshape(-1, num_attributes)
    flat_predictions = np.where(valid, predictions.reshape(-1, num_attributes), 0.0)
    flat_targets = np.where(valid, targets.reshape(-1, num_attributes), 0.0)

    count = valid.sum(axis=0).astype(np.float64)
    safe_count = np.maximum(count, 1.0)
    mean_x = flat_predictions.sum(axis=0) / safe_count
    mean_y = flat_targets.sum(axis=0) / safe_count
    dx = np.where(valid, flat_predictions - mean_x, 0.0)
    dy = np.where(valid, flat_targets - mean_y, 0.0)
    var_x = np.square(dx).sum(axis=0) / safe_count
    var_y = np.square(dy).sum(axis=0) / safe_count
    covariance = (dx * dy).sum(axis=0) / safe_count

    denominator = var_x + var_y + np.square(mean_x - mean_y)
    with np.errstate(invalid="ignore", divide="ignore"):
        ccc = 2.0 * covariance / denominator
    return np.where((count >= 2.0) & (denominator > 0.0), ccc, np.nan)


class PartwiseCCC(NamedTuple):
    r"""Per-conversation-part coefficients returned by [`ccc_by_part`].

    Args:
        part_ids (`np.ndarray`):
            Identifiers of the parts, sorted, shape `(num_parts,)`.
        ccc (`np.ndarray`):
            Coefficients of shape `(num_parts, num_attributes)`, `nan` where a part has too
            few evaluation frames for an attribute.
        counts (`np.ndarray`):
            Number of scored frames per part and attribute, shape `(num_parts, num_attributes)`.
    """

    part_ids: np.ndarray
    ccc: np.ndarray
    counts: np.ndarray


def _broadcast_part_ids(part_ids: Any, frame_shape: tuple[int, ...]) -> np.ndarray:
    r"""Expand per-sample identifiers to one identifier per frame."""
    part_ids = np.asarray(part_ids)
    if part_ids.shape == frame_shape:
        return part_ids
    if part_ids.shape == frame_shape[:1]:
        expanded_shape = frame_shape[:1] + (1,) * (len(frame_shape) - 1)
        return np.broadcast_to(part_ids.reshape(expanded_shape), frame_shape)
    raise ValueError(
        f"`part_ids` must have shape {frame_shape} or {frame_shape[:1]}, got {part_ids.shape}."
    )


def ccc_by_part(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
    part_ids: Optional[Any] = None,
) -> PartwiseCCC:
    r"""Concordance correlation coefficient computed separately for each conversation part.

    Args:
        predictions (`np.ndarray`):
            Predicted traces of shape `(num_samples, num_frames, num_attributes)`.
        targets (`np.ndarray`):
            Reference traces with the same shape, `nan` where unlabeled.
        mask (`np.ndarray`, *optional*):
            Evaluation mask of shape `(num_samples, num_frames)`.
        part_ids (array-like, *optional*):
            One identifier per sample, shape `(num_samples,)`, or per frame, shape
            `(num_samples, num_frames)`. Samples cut out of the same conversation part must
            share an identifier. When omitted the whole input is treated as a single part.

    Returns:
        [`PartwiseCCC`]: the sorted part identifiers with their coefficients and supports.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    frame_shape = targets.shape[:-1]
    num_attributes = targets.shape[-1]

    if part_ids is None:
        part_ids = np.zeros(frame_shape, dtype=np.int64)
    frame_part_ids = _broadcast_part_ids(part_ids, frame_shape).reshape(-1)

    valid = np.isfinite(targets)
    if mask is not None:
        expanded_mask = _as_bool_mask(mask)
        if expanded_mask.shape == frame_shape:
            expanded_mask = expanded_mask[..., None]
        valid = valid & expanded_mask
    valid = valid.reshape(-1, num_attributes)

    flat_predictions = predictions.reshape(-1, num_attributes)
    flat_targets = targets.reshape(-1, num_attributes)

    unique_ids, inverse = np.unique(frame_part_ids, return_inverse=True)
    ccc = np.full((len(unique_ids), num_attributes), np.nan, dtype=np.float64)
    counts = np.zeros((len(unique_ids), num_attributes), dtype=np.int64)
    for position in range(len(unique_ids)):
        selection = inverse == position
        part_valid = valid[selection]
        counts[position] = part_valid.sum(axis=0)
        ccc[position] = concordance_correlation_coefficient(
            flat_predictions[selection], flat_targets[selection], part_valid
        )
    return PartwiseCCC(part_ids=unique_ids, ccc=ccc, counts=counts)


def compute_vad_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    eval_mask: Optional[np.ndarray] = None,
    part_ids: Optional[Any] = None,
    attribute_names: Sequence[str] = DEFAULT_ATTRIBUTE_NAMES,
    label_pad_value: Optional[float] = None,
) -> Dict[str, float]:
    r"""Corpus-level CCC metrics for a set of predicted traces.

    Args:
        predictions (`np.ndarray`):
            Predicted traces of shape `(num_samples, num_frames, num_attributes)`.
        targets (`np.ndarray`):
            Reference traces with the same shape, `nan` where unlabeled.
        eval_mask (`np.ndarray`, *optional*):
            Mask of shape `(num_samples, num_frames)`, `True` on the frames that coincide with
            the original 100 ms label positions. Strongly recommended: without it the metric is
            computed on the dense 20 ms grid and is no longer comparable with published numbers.
        part_ids (array-like, *optional*):
            Conversation part identifier per sample or per frame, used for the macro average.
        attribute_names (`Sequence[str]`, *optional*, defaults to `("valence", "arousal",
            "dominance")`):
            Names of the trailing dimension, used to build the metric keys.
        label_pad_value (`float`, *optional*):
            Extra sentinel value marking padded target positions. Only needed when no
            `eval_mask` is supplied, since the mask already excludes padding.

    Returns:
        `Dict[str, float]`: `"<attribute>_ccc"` and `"mean_ccc"` for the micro aggregation,
        `"<attribute>_ccc_macro"` and `"mean_ccc_macro"` for the per-part aggregation, plus
        `"num_parts"` (parts that received a coefficient) and `"num_frames"` (scored positions
        per attribute, averaged over attributes).
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"`predictions` and `targets` must have the same shape, got {predictions.shape} and "
            f"{targets.shape}. Check that the model frame rate matches the label grid."
        )
    if predictions.shape[-1] != len(attribute_names):
        raise ValueError(
            f"Expected {len(attribute_names)} attributes in the last dimension, got "
            f"{predictions.shape[-1]}."
        )
    if label_pad_value is not None:
        targets = np.where(targets == label_pad_value, np.nan, targets)

    micro = concordance_correlation_coefficient(predictions, targets, eval_mask)
    partwise = ccc_by_part(predictions, targets, eval_mask, part_ids)

    metrics: Dict[str, float] = {}
    for index, name in enumerate(attribute_names):
        metrics[f"{name}_ccc"] = float(micro[index])
        column = partwise.ccc[:, index]
        finite = column[np.isfinite(column)]
        metrics[f"{name}_ccc_macro"] = float(finite.mean()) if finite.size else float("nan")

    metrics["mean_ccc"] = float(np.nanmean(micro)) if np.isfinite(micro).any() else float("nan")
    macro_values = [metrics[f"{name}_ccc_macro"] for name in attribute_names]
    finite_macro = [value for value in macro_values if np.isfinite(value)]
    metrics["mean_ccc_macro"] = float(np.mean(finite_macro)) if finite_macro else float("nan")

    metrics["num_parts"] = float(np.isfinite(partwise.ccc).any(axis=1).sum())
    metrics["num_frames"] = float(partwise.counts.sum(axis=0).mean())
    return metrics


def _select(container: Any, key: Union[int, str], role: str) -> Any:
    r"""Pick one array out of the container `Trainer` hands to `compute_metrics`."""
    if isinstance(container, Mapping):
        if isinstance(key, str) and key in container:
            return container[key]
        raise KeyError(f"{role} mapping has no entry {key!r}; available keys: {sorted(container)}.")
    if isinstance(container, (tuple, list)):
        if isinstance(key, int):
            if not -len(container) <= key < len(container):
                raise IndexError(
                    f"{role} tuple has {len(container)} entries, cannot select index {key}."
                )
            return container[key]
        raise TypeError(
            f"{role} is a tuple, so the selector must be an integer index, got {key!r}."
        )
    return container


class VADMetricComputer:
    r"""`compute_metrics` callable for `transformers.Trainer`.

    `Trainer` concatenates the model outputs listed in `TrainingArguments.label_names` into
    `EvalPrediction.label_ids`, in that exact order, and the remaining model outputs into
    `EvalPrediction.predictions`. Since [`EmoTraceForVADPrediction`] returns several tensors,
    the evaluation loop must be told which of them to keep, either with
    `ignore_keys_for_eval=["loss_dict", "last_hidden_state", ...]` so that `vad_predictions`
    stays first, or with a `preprocess_logits_for_metrics` function that returns it alone.

    Args:
        attribute_names (`Sequence[str]`, *optional*, defaults to `("valence", "arousal",
            "dominance")`):
            Names of the predicted attributes, in the order of the last dimension.
        label_names (`Sequence[str]`, *optional*, defaults to `("vad_labels", "vad_eval_mask")`):
            The same list that is passed to `TrainingArguments.label_names`; it maps the
            positions of `EvalPrediction.label_ids` back to names.
        prediction_key (`int` or `str`, *optional*, defaults to 0):
            Index into `EvalPrediction.predictions` (or key, when it is a mapping) holding the
            VAD predictions.
        part_ids (array-like, *optional*):
            Conversation part identifier per evaluation sample, in dataset order. Use it when
            the identifier is not part of the batch; otherwise add `"part_ids"` to
            `label_names` and it is picked up automatically.
        label_pad_value (`float`, *optional*):
            Sentinel marking padded target frames, only useful when no evaluation mask is
            available.
    """

    def __init__(
        self,
        attribute_names: Sequence[str] = DEFAULT_ATTRIBUTE_NAMES,
        label_names: Sequence[str] = DEFAULT_LABEL_NAMES,
        prediction_key: Union[int, str] = 0,
        part_ids: Optional[Any] = None,
        label_pad_value: Optional[float] = None,
    ) -> None:
        self.attribute_names = tuple(attribute_names)
        self.label_names = tuple(label_names)
        self.prediction_key = prediction_key
        self.part_ids = part_ids
        self.label_pad_value = label_pad_value

    def _named_labels(self, label_ids: Any, inputs: Any) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}
        if isinstance(inputs, Mapping):
            fields.update(inputs)
        if isinstance(label_ids, Mapping):
            fields.update(label_ids)
        elif isinstance(label_ids, (tuple, list)):
            if len(label_ids) > len(self.label_names):
                raise ValueError(
                    f"Received {len(label_ids)} label tensors but only {len(self.label_names)} "
                    f"names {self.label_names}; pass the same `label_names` as "
                    f"`TrainingArguments.label_names`."
                )
            fields.update(dict(zip(self.label_names, label_ids)))
        elif label_ids is not None:
            fields[self.label_names[0]] = label_ids
        return fields

    def __call__(self, eval_prediction: Any) -> Dict[str, float]:
        r"""Compute the metrics for one evaluation pass.

        Args:
            eval_prediction (`transformers.EvalPrediction` or `tuple`):
                Predictions and labels gathered over the evaluation dataset.

        Returns:
            `Dict[str, float]`: the metrics of [`compute_vad_metrics`].
        """
        predictions = getattr(eval_prediction, "predictions", None)
        label_ids = getattr(eval_prediction, "label_ids", None)
        inputs = getattr(eval_prediction, "inputs", None)
        if predictions is None and isinstance(eval_prediction, (tuple, list)):
            predictions, label_ids = eval_prediction[0], eval_prediction[1]

        vad_predictions = np.asarray(_select(predictions, self.prediction_key, "predictions"))
        fields = self._named_labels(label_ids, inputs)
        if VAD_LABELS_KEY not in fields:
            raise KeyError(
                f"{VAD_LABELS_KEY!r} was not found among the evaluation tensors "
                f"{sorted(fields)}; add it to `TrainingArguments.label_names`."
            )

        targets = np.asarray(fields[VAD_LABELS_KEY])
        eval_mask = fields.get(VAD_EVAL_MASK_KEY)
        part_ids = self.part_ids if self.part_ids is not None else fields.get(PART_IDS_KEY)
        if part_ids is not None and np.asarray(part_ids).shape[:1] != targets.shape[:1]:
            raise ValueError(
                f"`part_ids` covers {np.asarray(part_ids).shape[:1]} samples but the evaluation "
                f"output covers {targets.shape[:1]}; the dataset order must be preserved."
            )

        return compute_vad_metrics(
            vad_predictions,
            targets,
            eval_mask=eval_mask,
            part_ids=part_ids,
            attribute_names=self.attribute_names,
            label_pad_value=self.label_pad_value,
        )


def build_compute_metrics(
    attribute_names: Sequence[str] = DEFAULT_ATTRIBUTE_NAMES,
    label_names: Sequence[str] = DEFAULT_LABEL_NAMES,
    prediction_key: Union[int, str] = 0,
    part_ids: Optional[Any] = None,
    label_pad_value: Optional[float] = None,
) -> VADMetricComputer:
    r"""Build a configured [`VADMetricComputer`]; see that class for the arguments."""
    return VADMetricComputer(
        attribute_names=attribute_names,
        label_names=label_names,
        prediction_key=prediction_key,
        part_ids=part_ids,
        label_pad_value=label_pad_value,
    )


_DEFAULT_COMPUTER = VADMetricComputer()


def compute_metrics(eval_prediction: Any) -> Dict[str, float]:
    r"""Default `compute_metrics` callable, ready to be passed to `transformers.Trainer`.

    It assumes the default attribute order, `label_names=("vad_labels", "vad_eval_mask")` and
    `vad_predictions` as the first entry of `EvalPrediction.predictions`. Use
    [`build_compute_metrics`] to change any of those, in particular to supply the conversation
    part identifiers of the evaluation split.
    """
    return _DEFAULT_COMPUTER(eval_prediction)


def _reference_ccc(predictions: Sequence[float], targets: Sequence[float]) -> float:
    r"""Direct textbook CCC on two flat sequences, used only by the self-check."""
    import statistics

    mean_x = statistics.fmean(predictions)
    mean_y = statistics.fmean(targets)
    var_x = statistics.fmean([(value - mean_x) ** 2 for value in predictions])
    var_y = statistics.fmean([(value - mean_y) ** 2 for value in targets])
    covariance = statistics.fmean(
        [(predictions[i] - mean_x) * (targets[i] - mean_y) for i in range(len(predictions))]
    )
    return 2.0 * covariance / (var_x + var_y + (mean_x - mean_y) ** 2)


def _self_check() -> None:
    r"""Validate the metric against a direct formula and against the training loss."""
    rng = np.random.default_rng(0)
    num_samples, num_frames, num_attributes = 6, 40, 3

    predictions = rng.normal(size=(num_samples, num_frames, num_attributes))
    targets = 0.7 * predictions + 0.5 * rng.normal(size=predictions.shape)
    targets[rng.random(targets.shape) > 0.9] = np.nan
    eval_mask = np.zeros((num_samples, num_frames), dtype=bool)
    eval_mask[:, ::5] = True
    part_ids = np.array([0, 0, 1, 1, 1, 2])

    micro = concordance_correlation_coefficient(predictions, targets, eval_mask)
    for attribute in range(num_attributes):
        pairs = [
            (predictions[sample, frame, attribute], targets[sample, frame, attribute])
            for sample in range(num_samples)
            for frame in range(num_frames)
            if eval_mask[sample, frame] and np.isfinite(targets[sample, frame, attribute])
        ]
        expected = _reference_ccc([pair[0] for pair in pairs], [pair[1] for pair in pairs])
        assert abs(micro[attribute] - expected) < 1e-12, (attribute, micro[attribute], expected)

    # Frames outside the evaluation mask must not move the metric, whatever is predicted there.
    perturbed = predictions.copy()
    perturbed[:, 1::5] += 100.0
    assert np.allclose(
        micro, concordance_correlation_coefficient(perturbed, targets, eval_mask), atol=1e-12
    )
    assert np.allclose(
        concordance_correlation_coefficient(targets, targets, eval_mask), 1.0, atol=1e-12
    )

    partwise = ccc_by_part(predictions, targets, eval_mask, part_ids)
    assert partwise.part_ids.tolist() == [0, 1, 2]
    for position, part in enumerate(partwise.part_ids):
        selection = part_ids == part
        expected = concordance_correlation_coefficient(
            predictions[selection], targets[selection], eval_mask[selection]
        )
        assert np.allclose(partwise.ccc[position], expected, atol=1e-12)

    metrics = compute_vad_metrics(predictions, targets, eval_mask, part_ids)
    assert abs(metrics["valence_ccc"] - micro[0]) < 1e-12
    assert abs(metrics["mean_ccc"] - float(np.mean(micro))) < 1e-12
    assert abs(metrics["arousal_ccc_macro"] - float(np.mean(partwise.ccc[:, 1]))) < 1e-12
    assert metrics["num_parts"] == 3.0
    scored = (eval_mask[..., None] & np.isfinite(targets)).sum(axis=(0, 1))
    assert metrics["num_frames"] == float(scored.mean())

    # A part shorter than two evaluation frames has no coefficient and must be skipped by the
    # macro average instead of poisoning it.
    sparse_targets = targets.copy()
    sparse_targets[5] = np.nan
    sparse_metrics = compute_vad_metrics(predictions, sparse_targets, eval_mask, part_ids)
    assert sparse_metrics["num_parts"] == 2.0
    assert np.isfinite(sparse_metrics["mean_ccc_macro"])

    trainer_metrics = compute_metrics(
        ((predictions, rng.normal(size=(num_samples, num_frames, 5))), (targets, eval_mask))
    )
    assert abs(trainer_metrics["dominance_ccc"] - micro[2]) < 1e-12

    padded_mask = np.where(eval_mask, 1, -100)
    assert np.allclose(
        micro, concordance_correlation_coefficient(predictions, targets, padded_mask), atol=1e-12
    )

    scored_positions = eval_mask & np.isfinite(targets[..., 0])
    first_scored = tuple(int(index) for index in np.argwhere(scored_positions)[0])
    first_ignored = tuple(int(index) for index in np.argwhere(~scored_positions)[0])
    broken = predictions.copy()
    broken[first_ignored + (0,)] = np.nan
    assert np.allclose(
        micro, concordance_correlation_coefficient(broken, targets, eval_mask), atol=1e-12
    )
    broken[first_scored + (0,)] = np.nan
    assert np.isnan(concordance_correlation_coefficient(broken, targets, eval_mask)[0])

    try:
        import sys
        from pathlib import Path

        # Running this file as a script puts its own directory on `sys.path`, not the repository
        # root, so the package root is added explicitly before importing the training losses.
        repository_root = str(Path(__file__).resolve().parents[2])
        if repository_root not in sys.path:
            sys.path.insert(0, repository_root)
        import torch

        from ser_project.utils.losses import ccc_loss
    except ImportError:
        print("ser_project.utils.evaluate self-check passed (torch cross-check skipped)")
        return

    loss = ccc_loss(
        torch.from_numpy(predictions),
        torch.from_numpy(targets),
        torch.from_numpy(eval_mask),
        reduction="none",
        eps=0.0,
    )
    assert np.allclose(1.0 - loss.numpy(), micro, atol=1e-12)
    print("ser_project.utils.evaluate self-check passed")


if __name__ == "__main__":
    _self_check()
