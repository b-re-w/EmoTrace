r"""Multi-task data collator for the EmoTrace joint objective.

The collator turns dataset rows into the batch dictionary consumed by
[`EmoTraceForVADPrediction`]. It is responsible for the three alignments that
the model itself cannot perform:

1. Waveform padding, together with the sample-level attention mask.
2. Conversion of every sample's waveform length into the number of 20 ms
   backbone frames, using the same convolution arithmetic as
   `WavLMPreTrainedModel._get_feat_extract_output_lengths`, so that every label
   tensor has exactly the time dimension of the model outputs.
3. Projection of the continuous attribute traces, the speaker diarization and
   the transcript onto that frame grid.

Positions without supervision are filled with `float('nan')` instead of zero so
that a loss can mask them; a zero would be a valid attribute value and would
silently bias the regression towards the centre of the scale.

Frame targets are built by the helpers of `ser_project.datasets.msp_conversation`
and `ser_project.datasets.base` whenever those modules expose them, so that the
dataset builder and the collator cannot drift apart. Local equivalents keep this
module usable, importable and testable on its own; `resolved_helpers` reports
which implementation each role was bound to.
"""

from __future__ import annotations

import csv
import importlib
import inspect
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np
import torch

__all__ = [
    "DataCollatorForEmoTrace",
    "RESOLVED_HELPERS",
    "SPEAKER_TARGET_NAMES",
    "VAD_ATTRIBUTE_NAMES",
    "resolved_helpers",
]


VAD_ATTRIBUTE_NAMES: tuple[str, str, str] = ("valence", "arousal", "dominance")
r"""Attribute order of the last axis of `vad_labels`, matching the head output."""

SPEAKER_TARGET_NAMES: tuple[str, str, str] = ("speech", "overlap", "change")
r"""Channel order of the last axis of `spk_labels`."""

ATTRIBUTE_DIRECTORIES: dict[str, str] = {
    "valence": "Valence",
    "arousal": "Arousal",
    "dominance": "Dominance",
}

_DEFAULT_CONV_KERNEL: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
_DEFAULT_CONV_STRIDE: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)

_WAVEFORM_KEYS = ("input_values", "waveform", "speech", "raw_speech")
_PART_ID_KEYS = ("part_id", "conversation_part", "trace_id", "id", "name")
_CTC_LABEL_KEYS = ("ctc_labels", "labels", "input_ids", "transcript_ids")
_CTC_MASK_KEYS = ("ctc_loss_mask", "has_transcript", "transcript_usable")
_SPEAKER_TARGET_KEYS = ("spk_labels", "speaker_frames", "speaker_labels")
_SPEAKER_SEGMENT_KEYS = ("speaker_segments", "diarization", "segments")
_VAD_LABEL_KEYS = ("vad_labels", "vad_trace", "traces")
_FRAME_OFFSET_KEYS = ("vad_trace_offset", "trace_offset", "frame_offset")
_TIME_OFFSET_KEYS = ("start_time", "offset_seconds", "segment_start")


# ---------------------------------------------------------------------------
# Helper resolution against the sibling dataset modules
# ---------------------------------------------------------------------------

_HELPER_MODULES = (
    "ser_project.datasets.msp_conversation",
    "ser_project.datasets.base",
)

_HELPER_CANDIDATES: dict[str, tuple[str, ...]] = {
    "frame_count": (
        "num_frames_for_samples",
        "num_backbone_frames",
        "num_frames_from_samples",
    ),
    "frame_traces": (
        "load_frame_traces",
        "load_trace_set",
        "load_traces",
    ),
    "eval_mask": (
        "eval_grid_mask",
        "evaluation_grid_mask",
    ),
    "align_trace": (
        "align_trace_to_frames",
        "trace_to_frames",
        "resample_trace_to_frames",
    ),
    "speaker_frame_targets": (
        "speaker_frame_targets",
        "build_speaker_frame_targets",
        "make_speaker_targets",
    ),
    "diarization_segment": ("DiarizationSegment",),
    "corpus_paths": ("CorpusPaths",),
}

RESOLVED_HELPERS: dict[str, str] = {}
r"""Maps a helper role to the qualified name of the callable actually used.

Populated lazily on first use. A value ending in `(fallback)` means neither
sibling module exposed any of the candidate names and the local implementation
in this module was used instead.
"""

_HELPER_CACHE: dict[str, Callable[..., Any]] = {}
_MODULE_CACHE: dict[str, Any] = {}


def resolved_helpers() -> dict[str, str]:
    r"""Returns a copy of the helper resolution table.

    Returns:
        `dict[str, str]`: Helper role mapped to the qualified name of the
        callable that the collator bound to it. Roles are added as they are
        first used, so call this after collating a batch.
    """
    return dict(RESOLVED_HELPERS)


def _load_module(name: str) -> Any | None:
    r"""Imports a sibling module lazily, remembering failures.

    The import is deferred to first use so that this module stays importable
    while its siblings are still being written, and so that a package `__init__`
    importing both cannot deadlock on a circular import.
    """
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]
    try:
        module = importlib.import_module(name)
    except Exception:
        module = None
    _MODULE_CACHE[name] = module
    return module


def _resolve_helper(role: str, fallback: Callable[..., Any] | None) -> Callable[..., Any] | None:
    r"""Binds a helper role to a callable, preferring the sibling modules.

    Args:
        role (`str`): Key of `_HELPER_CANDIDATES`.
        fallback (`Callable`, *optional*): Local implementation used when no
            candidate name is exported by any sibling module.

    Returns:
        `Callable` or `None`: The bound helper, `None` when no candidate matched
        and no fallback was supplied.
    """
    if role in _HELPER_CACHE:
        return _HELPER_CACHE[role]

    helper: Callable[..., Any] | None = fallback
    source = (
        f"{__name__}.{getattr(fallback, '__name__', fallback)} (fallback)"
        if fallback is not None
        else "unavailable"
    )
    for module_name in _HELPER_MODULES:
        module = _load_module(module_name)
        if module is None:
            continue
        for name in _HELPER_CANDIDATES[role]:
            candidate = getattr(module, name, None)
            if callable(candidate):
                helper = candidate
                source = f"{module_name}.{name}"
                break
        if not source.endswith("(fallback)") and source != "unavailable":
            break

    RESOLVED_HELPERS[role] = source
    _HELPER_CACHE[role] = helper
    return helper


def _call_helper(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    r"""Calls `fn` passing only the keyword arguments its signature accepts.

    Sibling helpers may name an argument differently or not offer it at all;
    dropping unsupported keywords keeps the call working as long as the
    positional contract matches. Aliases for the same concept may therefore be
    passed together, since at most one of them will be forwarded.
    """
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return fn(*args, **kwargs)
    supported = {key: value for key, value in kwargs.items() if key in parameters}
    return fn(*args, **supported)


class _Segment(NamedTuple):
    r"""Fallback stand-in for `DiarizationSegment`, with the same field order."""

    speaker: str
    start: float
    end: float


# ---------------------------------------------------------------------------
# Local fallback implementations
# ---------------------------------------------------------------------------


def _num_frames_for_samples(
    num_samples: int,
    conv_kernel: Sequence[int] = _DEFAULT_CONV_KERNEL,
    conv_stride: Sequence[int] = _DEFAULT_CONV_STRIDE,
) -> int:
    r"""Number of backbone frames produced from `num_samples` input samples.

    Mirrors `WavLMPreTrainedModel._get_feat_extract_output_lengths`: the feature
    extractor uses no padding, so each layer contributes
    `floor((length - kernel_size) / stride) + 1`. With the WavLM-large geometry
    this yields one frame per 320 samples (20 ms) with a 400 sample (25 ms)
    receptive field, and the trailing incomplete window is dropped.

    Args:
        num_samples (`int`): Number of waveform samples at 16 kHz.
        conv_kernel (`Sequence[int]`): Kernel size of every extractor layer.
        conv_stride (`Sequence[int]`): Stride of every extractor layer.

    Returns:
        `int`: Frame count, clamped at zero for inputs shorter than the
        receptive field.
    """
    length = int(num_samples)
    for kernel_size, stride in zip(conv_kernel, conv_stride):
        length = (length - int(kernel_size)) // int(stride) + 1
    return max(length, 0)


def _align_trace_to_frames(
    times: np.ndarray,
    values: np.ndarray,
    num_frames: int,
    frame_stride_seconds: float = 0.02,
    start_time: float = 0.0,
    tolerance_seconds: float | None = None,
) -> np.ndarray:
    r"""Places a trace on the backbone frame grid by nearest central time.

    This is a placement, not a resampling: a frame that no trace sample falls on
    keeps `float('nan')`. A trace regenerated at the frame hop therefore fills
    every frame, while a trace left on the 100 ms evaluation hop fills only
    every fifth one and the loss simply has nothing to say about the rest. The
    released 100 ms values are never spread over neighbouring frames, since
    interpolating them would invent supervision the annotators did not give.

    Args:
        times (`np.ndarray`): Ascending trace central times in seconds,
            relative to the start of the conversation part.
        values (`np.ndarray`): Trace values of shape `(num_samples, channels)`.
        num_frames (`int`): Number of frames to produce.
        frame_stride_seconds (`float`): Backbone frame stride.
        start_time (`float`): Time of frame 0 within the conversation part.
        tolerance_seconds (`float`, *optional*): Largest accepted gap between a
            frame and the trace sample placed on it. Defaults to half a frame.

    Returns:
        `np.ndarray`: Array of shape `(num_frames, channels)` with `nan` at
        every frame that no trace sample matched.
    """
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]

    aligned = np.full((num_frames, values.shape[1]), np.nan, dtype=np.float32)
    if num_frames == 0 or times.size == 0:
        return aligned

    if tolerance_seconds is None:
        tolerance_seconds = frame_stride_seconds / 2.0

    usable = min(times.shape[0], values.shape[0])
    times, values = times[:usable], values[:usable]

    targets = start_time + np.arange(num_frames, dtype=np.float64) * frame_stride_seconds
    right = np.searchsorted(times, targets)
    left = np.clip(right - 1, 0, usable - 1)
    right = np.clip(right, 0, usable - 1)
    nearest = np.where(np.abs(times[right] - targets) < np.abs(times[left] - targets), right, left)

    matched = np.abs(times[nearest] - targets) <= tolerance_seconds
    aligned[matched] = values[nearest[matched]]
    return aligned


def _eval_grid_mask(
    times: np.ndarray,
    eval_hop_seconds: float = 0.1,
    frame_stride_seconds: float = 0.02,
) -> np.ndarray:
    r"""Marks the frames coinciding with an original 100 ms label position.

    The regenerated 20 ms trace and the released 100 ms trace come from the same
    median filter with the same reaction lag and the same first window, so the
    100 ms samples are exactly the 20 ms samples whose central time falls on the
    coarse grid; the values there are identical rather than merely close.

    Args:
        times (`np.ndarray`): Central time of every frame, in seconds.
        eval_hop_seconds (`float`): Hop of the evaluation grid.
        frame_stride_seconds (`float`): Hop of the frame grid, which sets the
            matching tolerance.

    Returns:
        `np.ndarray`: Boolean array shaped like `times`.
    """
    times = np.asarray(times, dtype=np.float64)
    tolerance = 0.25 * frame_stride_seconds
    offsets = times / eval_hop_seconds
    residual = np.abs(offsets - np.round(offsets)) * eval_hop_seconds
    return np.asarray(residual <= tolerance) & np.isfinite(times)


def _speaker_frame_targets(
    segments: Sequence[Any],
    num_frames: int,
    frame_stride_seconds: float = 0.02,
    time_offset: float = 0.0,
    initial_speakers: Sequence[str] | None = None,
) -> np.ndarray:
    r"""Rasterizes a diarization onto the frame grid.

    The three channels follow `SPEAKER_TARGET_NAMES`:

    - `speech`: at least one speaker is active in the frame,
    - `overlap`: two or more speakers are simultaneously active in the frame,
    - `change`: the set of active speakers differs from that of the previous
      frame.

    A frame is assigned to a turn as soon as its nominal support intersects the
    turn, so no turn is dropped however short it is.

    Args:
        segments (`Sequence`): Speaker turns exposing `speaker`, `start` and
            `end`. Overlapping turns are expected and are what makes the
            `overlap` channel fire.
        num_frames (`int`): Number of frames to produce.
        frame_stride_seconds (`float`): Backbone frame stride.
        time_offset (`float`): Time of frame 0 in the reference of `segments`.
        initial_speakers (`Sequence[str]`, *optional*): Speakers active in the
            frame preceding the window, so that `change` is correct at frame 0
            of a crop.

    Returns:
        `np.ndarray`: Float array of shape `(num_frames, 3)` holding zeros and
        ones.
    """
    if num_frames < 0:
        raise ValueError(f"num_frames must be non-negative, got {num_frames}.")
    targets = np.zeros((num_frames, 3), dtype=np.float32)
    if num_frames == 0:
        return targets

    speakers = sorted({str(segment.speaker) for segment in segments} | set(initial_speakers or ()))
    if not speakers:
        return targets
    speaker_index = {speaker: index for index, speaker in enumerate(speakers)}

    activity = np.zeros((num_frames, len(speakers)), dtype=bool)
    for segment in segments:
        start = float(segment.start) - time_offset
        end = float(segment.end) - time_offset
        first = max(int(math.floor(start / frame_stride_seconds + 1e-9)), 0)
        last = min(int(math.ceil(end / frame_stride_seconds - 1e-9)), num_frames)
        if last > first:
            activity[first:last, speaker_index[str(segment.speaker)]] = True

    previous = np.empty_like(activity)
    previous[1:] = activity[:-1]
    previous[0] = False
    for speaker in initial_speakers or ():
        if speaker in speaker_index:
            previous[0, speaker_index[speaker]] = True

    active_count = activity.sum(axis=1)
    targets[:, 0] = active_count >= 1
    targets[:, 1] = active_count >= 2
    targets[:, 2] = (activity != previous).any(axis=1)
    return targets


def _segment_fields(segment: Mapping[str, Any]) -> tuple[float | None, float | None, Any]:
    r"""Extracts start, end and speaker from a diarization mapping."""
    start = _first_present(segment, ("Start_Time", "start_time", "start", "begin"))
    end = _first_present(segment, ("End_Time", "end_time", "end", "stop"))
    speaker = _first_present(segment, ("Speaker", "speaker", "speaker_id", "label"))
    start = None if start is None else float(start)
    end = None if end is None else float(end)
    return start, end, speaker


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


@lru_cache(maxsize=32)
def _read_trace_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    r"""Reads a MSP-Conversation trace CSV.

    Results are cached because the rows of one batch often share a conversation
    part. The cache is deliberately small: at the 20 ms hop a single part holds
    around fourteen thousand rows per attribute, and a data loader replicates
    this cache in every worker process. The arrays are returned read-only so
    that callers cannot corrupt a cached entry.

    Args:
        path (`str`): Path to a `<Attribute>/<part>.csv` file with the header
            `Central_Time,Start_Time,End_Time,Mean,<annotator ids>`.

    Returns:
        `tuple[np.ndarray, np.ndarray]`: Central times in seconds and the mean
        trace value at each of them.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Trace file not found: {path}. Regenerate the 20 ms traces with "
            "msp_conv_utils.make_traces(VersionFldr, SaveFldr, window=120, overlap=100) "
            "and point `trace_dir` at the folder holding the Valence, Arousal and "
            "Dominance subfolders."
        )
    times: list[float] = []
    values: list[float] = []
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "Central_Time" not in reader.fieldnames:
            raise ValueError(
                f"{path} does not look like a trace file: expected a "
                "'Central_Time,Start_Time,End_Time,Mean,...' header, got "
                f"{reader.fieldnames}."
            )
        for row in reader:
            times.append(float(row["Central_Time"]))
            values.append(float(row["Mean"]))

    time_array = np.asarray(times, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float32)
    time_array.flags.writeable = False
    value_array.flags.writeable = False
    return time_array, value_array


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


@dataclass
class DataCollatorForEmoTrace:
    r"""Collates MSP-Conversation rows into the EmoTrace batch dictionary.

    The produced batch always carries the waveform and the attribute targets;
    the transcript and speaker targets are emitted only when the corresponding
    task is enabled, which is how the joint-loss ablations are configured. A
    disabled task contributes no keys at all rather than empty tensors, so a
    model reading `"ctc_labels" in batch` sees the ablation directly.

    Keys of the returned batch, with `B` the batch size, `S` the padded number
    of waveform samples and `T` the number of backbone frames produced from `S`:

    | key                 | dtype   | shape       |
    |---------------------|---------|-------------|
    | `input_values`      | float32 | `(B, S)`    |
    | `attention_mask`    | int64   | `(B, S)`    |
    | `vad_labels`        | float32 | `(B, T, 3)` |
    | `vad_eval_mask`     | bool    | `(B, T)`    |
    | `ctc_labels`        | int64   | `(B, L)`    |
    | `ctc_label_lengths` | int64   | `(B,)`      |
    | `ctc_loss_mask`     | bool    | `(B,)`      |
    | `spk_labels`        | float32 | `(B, T, 3)` |

    Every frame beyond a sample's own frame count, and every frame for which the
    corpus provides no annotation, holds `nan` in `vad_labels` and `spk_labels`.
    `ctc_labels` is padded with `ctc_label_pad_value` (`-100` by default).

    Accepted row fields, all optional unless stated:

    - waveform (required): `input_values`, `waveform`, `speech`, `raw_speech`,
      an `audio` mapping with an `array` entry as produced by the `datasets`
      audio feature, or an `audio_path` decoded here together with the optional
      `audio_offset_seconds` and `duration_seconds`.
    - `vad_labels`: either an array of shape `(num_frames, 3)` already on the
      20 ms grid and ordered as `VAD_ATTRIBUTE_NAMES`, or a mapping from
      attribute name to a one dimensional trace. Equivalently, one column per
      attribute named `valence`, `arousal` and `dominance`.
    - `part_id`: conversation part identifier, used to read the traces from
      `corpus_root` or `trace_dir` when the row carries no `vad_labels`.
    - `vad_trace_offset` / `start_time`: position of the row's first frame
      inside the conversation part, as a frame index or in seconds. Required for
      crops so that the traces are sliced at the right place and the 100 ms
      evaluation grid stays correctly phased.
    - `ctc_labels`: sequence of vocabulary ids.
    - `ctc_loss_mask`: bool overriding the automatic usability decision.
    - `spk_labels`: array of shape `(num_frames, 3)`, or `speaker_segments`: a
      diarization list, either `DiarizationSegment` tuples or mappings with
      `Start_Time` / `End_Time` / `Speaker` keys. Segment times are read in the
      same reference as the row's frame offset, that is relative to the
      conversation part, unless `speaker_segments_rebased` is set.

    Args:
        sampling_rate (`int`, *optional*, defaults to 16000):
            Expected waveform sampling rate. Rows carrying a different rate in
            an `audio` mapping are rejected.
        conv_kernel (`tuple[int, ...]`, *optional*):
            Kernel sizes of the backbone feature extractor.
        conv_stride (`tuple[int, ...]`, *optional*):
            Strides of the backbone feature extractor.
        frame_stride_seconds (`float`, *optional*, defaults to 0.02):
            Duration of one backbone frame.
        waveform_mean (`float`, *optional*, defaults to 0.0):
            Fixed dataset-level mean subtracted from the waveform.
        waveform_std (`float`, *optional*, defaults to 1.0):
            Fixed dataset-level standard deviation the waveform is divided by.
            These two are deliberately constants: per-utterance statistics leak
            information about the whole utterance into every frame and are not
            available to a streaming model.
        normalize_waveform (`bool`, *optional*, defaults to `True`):
            Whether to apply the fixed normalization above.
        decode_audio (`bool`, *optional*, defaults to `True`):
            Whether a row carrying only an `audio_path` may be decoded here.
            Turn it off to require materialized waveforms.
        pad_to_multiple_of (`int`, *optional*):
            Rounds the padded waveform length up to a multiple of this value.
        max_length (`int`, *optional*):
            Truncates waveforms to this many samples.
        corpus_root (`str`, *optional*):
            MSP-Conversation release directory. When set, rows without
            `vad_labels` have their traces read through the loader of
            `ser_project.datasets.msp_conversation`.
        frame_traces_dirname (`str`, *optional*, defaults to `"Traces_20ms"`):
            Name of the directory inside `corpus_root` holding the regenerated
            20 ms traces.
        trace_dir (`str`, *optional*):
            Direct path to a folder holding `Valence/`, `Arousal/` and
            `Dominance/` subfolders, read with this module's own CSV reader.
            An alternative to `corpus_root` for ad hoc trace folders.
        trace_hop_seconds (`float`, *optional*, defaults to 0.02):
            Hop of the traces referenced by the rows.
        eval_hop_seconds (`float`, *optional*, defaults to 0.1):
            Hop of the released traces the evaluation is reported on. Frames
            coinciding with that grid are flagged in `vad_eval_mask`.
        trace_scale (`float`, *optional*, defaults to 0.01):
            Factor applied to traces read from disk, mapping the corpus range
            `[-100, 100]` onto `[-1, 1]`. Not applied to values already present
            in the row.
        include_ctc (`bool`, *optional*, defaults to `True`):
            Emit the transcript keys.
        include_speaker (`bool`, *optional*, defaults to `True`):
            Emit the speaker key.
        ctc_label_pad_value (`int`, *optional*, defaults to -100):
            Padding value of `ctc_labels`.
        speaker_segments_rebased (`bool`, *optional*, defaults to `False`):
            Set when a row's segment times are already relative to its own first
            frame rather than to the conversation part, which suppresses the
            frame offset when rasterizing them.

    Example:

    ```python
    >>> from ser_project.datasets.collator import DataCollatorForEmoTrace
    >>> collator = DataCollatorForEmoTrace(include_ctc=False, include_speaker=False)
    >>> batch = collator([{"input_values": [0.0] * 16000}])
    >>> batch["input_values"].shape, batch["vad_labels"].shape
    (torch.Size([1, 16000]), torch.Size([1, 49, 3]))
    ```
    """

    sampling_rate: int = 16000
    conv_kernel: tuple[int, ...] = _DEFAULT_CONV_KERNEL
    conv_stride: tuple[int, ...] = _DEFAULT_CONV_STRIDE
    frame_stride_seconds: float = 0.02

    waveform_mean: float = 0.0
    waveform_std: float = 1.0
    normalize_waveform: bool = True
    decode_audio: bool = True
    pad_to_multiple_of: int | None = None
    max_length: int | None = None

    corpus_root: str | os.PathLike[str] | None = None
    frame_traces_dirname: str = "Traces_20ms"
    trace_dir: str | os.PathLike[str] | None = None
    trace_hop_seconds: float = 0.02
    eval_hop_seconds: float = 0.1
    trace_scale: float = 0.01

    include_ctc: bool = True
    include_speaker: bool = True
    ctc_label_pad_value: int = -100
    speaker_segments_rebased: bool = False

    def __post_init__(self):
        self.conv_kernel = tuple(int(value) for value in self.conv_kernel)
        self.conv_stride = tuple(int(value) for value in self.conv_stride)
        if len(self.conv_kernel) != len(self.conv_stride):
            raise ValueError(
                "conv_kernel and conv_stride must have the same length, got "
                f"{len(self.conv_kernel)} and {len(self.conv_stride)}."
            )
        if self.normalize_waveform and self.waveform_std == 0.0:
            raise ValueError("waveform_std must be non-zero when normalize_waveform is True.")
        ratio = self.eval_hop_seconds / self.trace_hop_seconds
        if ratio < 1 or abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(
                "eval_hop_seconds must be an integer multiple of trace_hop_seconds, got "
                f"{self.eval_hop_seconds} and {self.trace_hop_seconds}."
            )
        # A sibling frame-count helper is parameterized by frame geometry rather
        # than by the convolution stack, so it may only be trusted for the
        # stock extractor.
        self._standard_geometry = (
            self.conv_kernel == _DEFAULT_CONV_KERNEL and self.conv_stride == _DEFAULT_CONV_STRIDE
        )

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> "DataCollatorForEmoTrace":
        r"""Builds a collator whose frame arithmetic matches a model config.

        Args:
            config: A config exposing `conv_kernel` and `conv_stride`, either
                directly or through a nested `backbone_config` / `wavlm_config`
                / `encoder_config`.
            **kwargs: Forwarded to the constructor, overriding the values read
                from the config.

        Returns:
            [`DataCollatorForEmoTrace`]
        """
        source = config
        for attribute in ("backbone_config", "wavlm_config", "encoder_config"):
            nested = getattr(config, attribute, None)
            if nested is not None and hasattr(nested, "conv_kernel"):
                source = nested
                break
        if not hasattr(source, "conv_kernel"):
            raise ValueError(
                f"{type(config).__name__} exposes no conv_kernel/conv_stride; pass the "
                "backbone configuration instead."
            )
        settings: dict[str, Any] = {
            "conv_kernel": tuple(source.conv_kernel),
            "conv_stride": tuple(source.conv_stride),
        }
        sampling_rate = getattr(source, "sampling_rate", None)
        if sampling_rate is not None:
            settings["sampling_rate"] = int(sampling_rate)
        settings.update(kwargs)
        return cls(**settings)

    # -- frame arithmetic ---------------------------------------------------

    def num_frames(self, num_samples: int) -> int:
        r"""Number of backbone frames produced from `num_samples` samples.

        Args:
            num_samples (`int`): Waveform length in samples.

        Returns:
            `int`: Frame count, matching the time dimension of the backbone
            output for an input of that length.
        """
        if self._standard_geometry:
            helper = _resolve_helper("frame_count", _num_frames_for_samples)
        else:
            helper = _num_frames_for_samples
        value = _call_helper(
            helper,
            int(num_samples),
            sampling_rate=self.sampling_rate,
            frame_stride_seconds=self.frame_stride_seconds,
            conv_kernel=self.conv_kernel,
            conv_stride=self.conv_stride,
        )
        if isinstance(value, torch.Tensor):
            value = int(value.item())
        return max(int(value), 0)

    # -- entry point --------------------------------------------------------

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        r"""Collates a list of dataset rows.

        Args:
            features (`Sequence[Mapping[str, Any]]`): Dataset rows.

        Returns:
            `dict[str, torch.Tensor]`: The batch dictionary documented on the
            class.
        """
        if len(features) == 0:
            raise ValueError("Cannot collate an empty list of features.")

        waveforms = [self._extract_waveform(feature) for feature in features]
        if self.max_length is not None:
            waveforms = [waveform[: self.max_length] for waveform in waveforms]

        sample_lengths = [waveform.shape[0] for waveform in waveforms]
        padded_length = max(sample_lengths)
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 1:
            multiple = self.pad_to_multiple_of
            padded_length = int(math.ceil(padded_length / multiple) * multiple)

        batch_size = len(features)
        input_values = np.zeros((batch_size, padded_length), dtype=np.float32)
        attention_mask = np.zeros((batch_size, padded_length), dtype=np.int64)
        for index, waveform in enumerate(waveforms):
            length = waveform.shape[0]
            input_values[index, :length] = waveform
            attention_mask[index, :length] = 1

        if self.normalize_waveform:
            input_values -= np.float32(self.waveform_mean)
            input_values /= np.float32(self.waveform_std)
            # Padding must stay exactly zero: the backbone masks it out only
            # after the convolutional extractor has run over it.
            input_values *= attention_mask.astype(np.float32)

        total_frames = self.num_frames(padded_length)
        frame_counts = [min(self.num_frames(length), total_frames) for length in sample_lengths]

        batch: dict[str, torch.Tensor] = {
            "input_values": torch.from_numpy(input_values),
            "attention_mask": torch.from_numpy(attention_mask),
        }

        vad_labels, vad_eval_mask = self._collate_vad(features, frame_counts, total_frames)
        batch["vad_labels"] = torch.from_numpy(vad_labels)
        batch["vad_eval_mask"] = torch.from_numpy(vad_eval_mask)

        if self.include_ctc:
            ctc_labels, ctc_label_lengths, ctc_loss_mask = self._collate_ctc(features, frame_counts)
            batch["ctc_labels"] = torch.from_numpy(ctc_labels)
            batch["ctc_label_lengths"] = torch.from_numpy(ctc_label_lengths)
            batch["ctc_loss_mask"] = torch.from_numpy(ctc_loss_mask)

        if self.include_speaker:
            batch["spk_labels"] = torch.from_numpy(
                self._collate_speaker(features, frame_counts, total_frames)
            )

        return batch

    # -- waveform -----------------------------------------------------------

    def _extract_waveform(self, feature: Mapping[str, Any]) -> np.ndarray:
        r"""Pulls a mono 16 kHz waveform out of a dataset row."""
        audio = feature.get("audio")
        if isinstance(audio, Mapping) and "array" in audio:
            rate = audio.get("sampling_rate")
            if rate is not None and int(rate) != self.sampling_rate:
                raise ValueError(
                    f"Row provides audio at {rate} Hz but the collator expects "
                    f"{self.sampling_rate} Hz. Cast the dataset column with "
                    "datasets.Audio(sampling_rate=16000) before collating."
                )
            return self._as_mono(audio["array"])

        for key in _WAVEFORM_KEYS:
            if key in feature and feature[key] is not None:
                return self._as_mono(feature[key])

        if self.decode_audio and feature.get("audio_path"):
            return self._decode_audio(feature)

        raise KeyError(
            "No waveform in row. Expected one of "
            f"{_WAVEFORM_KEYS}, an 'audio' mapping with an 'array' entry, or an "
            f"'audio_path' to decode, got keys {sorted(feature.keys())}."
        )

    def _decode_audio(self, feature: Mapping[str, Any]) -> np.ndarray:
        r"""Decodes the window of an audio file that a row refers to.

        Rows built by `ser_project.datasets.msp_conversation` carry the audio
        lazily, as a path into either a conversation part or the whole
        conversation plus the offset of the part inside it. Only the requested
        window is read from disk.
        """
        try:
            import soundfile
        except ImportError as error:
            raise ImportError(
                "Decoding 'audio_path' requires the 'soundfile' package. Install it, or "
                "materialize the waveform in the dataset (for example with "
                "datasets.Audio(sampling_rate=16000)) before collating."
            ) from error

        path = os.fspath(feature["audio_path"])
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Audio file referenced by the row does not exist: {path}. The corpus "
                "'Audio' archive may still be extracting."
            )

        offset_seconds = float(feature.get("audio_offset_seconds") or 0.0)
        offset_seconds += self._frame_offset(feature) * self.frame_stride_seconds
        duration = feature.get("duration_seconds")

        with soundfile.SoundFile(path) as handle:
            if handle.samplerate != self.sampling_rate:
                raise ValueError(
                    f"{path} is sampled at {handle.samplerate} Hz but the collator expects "
                    f"{self.sampling_rate} Hz."
                )
            handle.seek(max(int(round(offset_seconds * handle.samplerate)), 0))
            frames = -1
            if duration is not None:
                remaining = float(duration) - self._frame_offset(feature) * self.frame_stride_seconds
                frames = max(int(round(remaining * handle.samplerate)), 0)
            if self.max_length is not None:
                frames = self.max_length if frames < 0 else min(frames, self.max_length)
            data = handle.read(frames=frames, dtype="float32", always_2d=False)

        return self._as_mono(data)

    @staticmethod
    def _as_mono(array: Any) -> np.ndarray:
        r"""Converts a waveform to a contiguous 1-D float32 array."""
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        waveform = np.asarray(array, dtype=np.float32)
        if waveform.ndim == 2:
            # Average the channels, whichever axis carries them.
            axis = 0 if waveform.shape[0] < waveform.shape[1] else 1
            waveform = waveform.mean(axis=axis)
        elif waveform.ndim != 1:
            raise ValueError(f"Expected a mono or 2-D waveform, got shape {waveform.shape}.")
        return np.ascontiguousarray(waveform, dtype=np.float32)

    # -- attribute traces ---------------------------------------------------

    def _collate_vad(
        self,
        features: Sequence[Mapping[str, Any]],
        frame_counts: Sequence[int],
        total_frames: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Builds `vad_labels` and `vad_eval_mask`."""
        labels = np.full((len(features), total_frames, 3), np.nan, dtype=np.float32)
        eval_mask = np.zeros((len(features), total_frames), dtype=bool)

        for index, feature in enumerate(features):
            count = frame_counts[index]
            if count == 0:
                continue
            values, mask = self._sample_trace(feature, count, self._frame_offset(feature))
            if values is None:
                continue
            labels[index, :count] = values
            # A frame is only usable for evaluation if it also carries a value.
            eval_mask[index, :count] = mask & np.isfinite(values).all(axis=-1)

        return labels, eval_mask

    def _frame_offset(self, feature: Mapping[str, Any]) -> int:
        r"""Index of the row's first frame inside its conversation part."""
        for key in _FRAME_OFFSET_KEYS:
            if key in feature and feature[key] is not None:
                return int(feature[key])
        for key in _TIME_OFFSET_KEYS:
            if key in feature and feature[key] is not None:
                return int(round(float(feature[key]) / self.frame_stride_seconds))
        return 0

    def _sample_trace(
        self,
        feature: Mapping[str, Any],
        num_frames: int,
        offset: int,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        r"""Returns the per-frame attribute values and the evaluation mask."""
        empty_mask = np.zeros(num_frames, dtype=bool)

        provided = _first_present(feature, _VAD_LABEL_KEYS)
        if provided is None and all(name in feature for name in VAD_ATTRIBUTE_NAMES):
            # The dataset stores one column per attribute rather than a stacked
            # array; assemble them in the canonical order.
            provided = {name: feature[name] for name in VAD_ATTRIBUTE_NAMES}
        if provided is not None:
            values = self._stack_attributes(provided)
            times = feature.get("trace_times")
            if times is not None and len(times) > 0:
                # The row states where its samples sit in time, which is the
                # only safe reading when the trace hop is not the frame stride.
                helper = _resolve_helper("align_trace", _align_trace_to_frames)
                aligned = _call_helper(
                    helper,
                    times,
                    values,
                    num_frames,
                    self.frame_stride_seconds,
                    start_time=offset * self.frame_stride_seconds,
                    time_offset=offset * self.frame_stride_seconds,
                    tolerance_seconds=self.frame_stride_seconds / 2.0,
                )
                aligned = np.asarray(aligned, dtype=np.float32).reshape(num_frames, -1)
            else:
                # No times given, so the values are taken to be this row's own
                # frames, one per backbone frame.
                aligned = self._pad_to(values[:num_frames], num_frames)
            return aligned, self._eval_mask_for(num_frames, offset)

        part_id = _first_present(feature, _PART_ID_KEYS)
        if self.corpus_root is not None:
            if part_id is None:
                raise KeyError(
                    "corpus_root is configured but the row carries neither 'vad_labels' nor a "
                    f"conversation part identifier (one of {_PART_ID_KEYS})."
                )
            return self._trace_from_corpus(str(part_id), num_frames, offset)

        if self.trace_dir is not None:
            if part_id is None:
                raise KeyError(
                    "trace_dir is configured but the row carries neither 'vad_labels' nor a "
                    f"conversation part identifier (one of {_PART_ID_KEYS})."
                )
            return self._trace_from_directory(str(part_id), num_frames, offset)

        return None, empty_mask

    def _eval_mask_for(self, num_frames: int, offset: int) -> np.ndarray:
        r"""Evaluation mask of a window of `num_frames` frames starting at `offset`.

        The mask is built for the whole conversation part and then sliced, since
        a grid helper is entitled to expect central times that start at zero and
        would otherwise reject every frame of a crop.
        """
        helper = _resolve_helper("eval_mask", _eval_grid_mask)
        total = offset + num_frames
        times = np.arange(total, dtype=np.float64) * self.frame_stride_seconds
        mask = _call_helper(
            helper,
            times,
            eval_hop_seconds=self.eval_hop_seconds,
            frame_stride_seconds=self.frame_stride_seconds,
        )
        mask = np.asarray(mask, dtype=bool).reshape(total)
        return mask[offset:total]

    def _corpus_paths(self) -> Any:
        r"""Resolves `corpus_root` into whatever the sibling loader expects.

        Passing a typed paths object rather than a bare directory is what lets
        `frame_traces_dirname` reach the loader.
        """
        paths_type = _resolve_helper("corpus_paths", None)
        if paths_type is None or not hasattr(paths_type, "from_root"):
            return self.corpus_root
        return paths_type.from_root(
            self.corpus_root, frame_traces_dirname=self.frame_traces_dirname
        )

    def _trace_from_corpus(
        self, part_id: str, num_frames: int, offset: int
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Reads the traces through the sibling corpus loader."""
        loader = _resolve_helper("frame_traces", None)
        if loader is None:
            raise RuntimeError(
                "corpus_root is set but ser_project.datasets.msp_conversation exposes no "
                f"trace loader (looked for {_HELPER_CANDIDATES['frame_traces']}). Provide "
                "'trace_dir' instead to use this module's own CSV reader."
            )
        # The loader aligns from the start of the conversation part, so a crop
        # is obtained by loading up to its end and slicing off the lead-in.
        trace_set = _call_helper(
            loader,
            part_id,
            self._corpus_paths(),
            num_frames=offset + num_frames,
            value_scale=self.trace_scale,
        )
        values = np.asarray(getattr(trace_set, "values", trace_set), dtype=np.float32)
        values = values[offset : offset + num_frames]

        mask = getattr(trace_set, "eval_mask", None)
        if mask is None:
            mask = self._eval_mask_for(num_frames, offset)
        else:
            mask = np.asarray(mask, dtype=bool)[offset : offset + num_frames]

        return self._pad_to(values, num_frames), self._pad_mask_to(mask, num_frames)

    def _trace_from_directory(
        self, part_id: str, num_frames: int, offset: int
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Reads the traces from a plain attribute-folder layout."""
        times: np.ndarray | None = None
        columns = []
        for attribute in VAD_ATTRIBUTE_NAMES:
            directory = ATTRIBUTE_DIRECTORIES[attribute]
            path = os.path.join(os.fspath(self.trace_dir), directory, f"{part_id}.csv")
            attribute_times, values = _read_trace_csv(path)
            if times is None:
                times = np.asarray(attribute_times, dtype=np.float64)
            columns.append(np.asarray(values, dtype=np.float32))

        width = min(int(times.shape[0]), min(column.shape[0] for column in columns))
        stacked = np.stack([column[:width] for column in columns], axis=-1)
        stacked = stacked * np.float32(self.trace_scale)

        # Placing by central time rather than by index keeps this correct
        # whether the folder holds the 20 ms traces or the released 100 ms ones.
        helper = _resolve_helper("align_trace", _align_trace_to_frames)
        aligned = _call_helper(
            helper,
            times[:width],
            stacked,
            num_frames,
            self.frame_stride_seconds,
            start_time=offset * self.frame_stride_seconds,
            time_offset=offset * self.frame_stride_seconds,
            tolerance_seconds=self.frame_stride_seconds / 2.0,
        )
        aligned = np.asarray(aligned, dtype=np.float32).reshape(num_frames, -1)
        return aligned, self._eval_mask_for(num_frames, offset)

    @staticmethod
    def _pad_to(values: np.ndarray, num_frames: int) -> np.ndarray:
        r"""Pads a `(n, 3)` array to `(num_frames, 3)` with `nan`."""
        if values.shape[0] == num_frames:
            return values.astype(np.float32, copy=False)
        padded = np.full((num_frames, values.shape[1]), np.nan, dtype=np.float32)
        usable = min(num_frames, values.shape[0])
        padded[:usable] = values[:usable]
        return padded

    @staticmethod
    def _pad_mask_to(mask: np.ndarray, num_frames: int) -> np.ndarray:
        r"""Pads a boolean mask to `num_frames` with `False`."""
        if mask.shape[0] == num_frames:
            return mask
        padded = np.zeros(num_frames, dtype=bool)
        usable = min(num_frames, mask.shape[0])
        padded[:usable] = mask[:usable]
        return padded

    @staticmethod
    def _stack_attributes(provided: Any) -> np.ndarray:
        r"""Normalizes a row's attribute labels to a `(num_frames, 3)` array."""
        if isinstance(provided, Mapping):
            lowered = {str(key).lower(): value for key, value in provided.items()}
            columns = []
            for attribute in VAD_ATTRIBUTE_NAMES:
                if attribute not in lowered:
                    raise KeyError(
                        f"Attribute '{attribute}' missing from the row's trace mapping; "
                        f"got {sorted(lowered)}."
                    )
                columns.append(np.asarray(lowered[attribute], dtype=np.float32).reshape(-1))
            width = min(column.shape[0] for column in columns)
            return np.stack([column[:width] for column in columns], axis=-1)

        if isinstance(provided, torch.Tensor):
            provided = provided.detach().cpu().numpy()
        values = np.asarray(provided, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if values.shape[-1] != 3:
            raise ValueError(
                "Attribute labels must have three channels ordered as "
                f"{VAD_ATTRIBUTE_NAMES}, got shape {values.shape}."
            )
        return values

    # -- transcript ---------------------------------------------------------

    def _collate_ctc(
        self,
        features: Sequence[Mapping[str, Any]],
        frame_counts: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Builds `ctc_labels`, `ctc_label_lengths` and `ctc_loss_mask`."""
        sequences: list[np.ndarray] = []
        for feature in features:
            provided = _first_present(feature, _CTC_LABEL_KEYS)
            if provided is None:
                sequences.append(np.zeros(0, dtype=np.int64))
                continue
            if isinstance(provided, torch.Tensor):
                provided = provided.detach().cpu().numpy()
            tokens = np.asarray(provided, dtype=np.int64).reshape(-1)
            # Rows coming back from an already padded column may carry the
            # ignore value; it is not part of the target sequence.
            sequences.append(tokens[tokens != self.ctc_label_pad_value])

        lengths = np.array([sequence.shape[0] for sequence in sequences], dtype=np.int64)
        width = int(lengths.max()) if lengths.size else 0
        labels = np.full((len(features), width), self.ctc_label_pad_value, dtype=np.int64)
        for index, sequence in enumerate(sequences):
            labels[index, : sequence.shape[0]] = sequence

        loss_mask = np.zeros(len(features), dtype=bool)
        for index, feature in enumerate(features):
            override = _first_present(feature, _CTC_MASK_KEYS)
            # A target longer than the input makes the CTC alignment impossible
            # and the loss infinite, so such a sample is dropped from the term.
            usable = bool(lengths[index] > 0 and lengths[index] <= frame_counts[index])
            if override is not None:
                usable = bool(override) and usable
            loss_mask[index] = usable

        return labels, lengths, loss_mask

    # -- speaker ------------------------------------------------------------

    def _collate_speaker(
        self,
        features: Sequence[Mapping[str, Any]],
        frame_counts: Sequence[int],
        total_frames: int,
    ) -> np.ndarray:
        r"""Builds `spk_labels`."""
        labels = np.full((len(features), total_frames, 3), np.nan, dtype=np.float32)

        for index, feature in enumerate(features):
            count = frame_counts[index]
            if count == 0:
                continue

            provided = _first_present(feature, _SPEAKER_TARGET_KEYS)
            if provided is not None:
                if isinstance(provided, torch.Tensor):
                    provided = provided.detach().cpu().numpy()
                values = np.asarray(provided, dtype=np.float32)
                if values.ndim != 2 or values.shape[-1] != 3:
                    raise ValueError(
                        "Speaker targets must have shape (num_frames, 3) ordered as "
                        f"{SPEAKER_TARGET_NAMES}, got shape {values.shape}."
                    )
                usable = min(count, values.shape[0])
                labels[index, :usable] = values[:usable]
                continue

            raw_segments = _first_present(feature, _SPEAKER_SEGMENT_KEYS)
            if raw_segments is None:
                continue

            segments = self._as_segments(raw_segments)
            offset_seconds = (
                0.0
                if self.speaker_segments_rebased
                else self._frame_offset(feature) * self.frame_stride_seconds
            )
            helper = _resolve_helper("speaker_frame_targets", _speaker_frame_targets)
            targets = _call_helper(
                helper,
                segments,
                count,
                frame_stride_seconds=self.frame_stride_seconds,
                time_offset=offset_seconds,
                start_time=offset_seconds,
                initial_speakers=self._initial_speakers(segments, offset_seconds),
            )
            if isinstance(targets, torch.Tensor):
                targets = targets.detach().cpu().numpy()
            targets = np.asarray(targets, dtype=np.float32).reshape(-1, 3)
            usable = min(count, targets.shape[0])
            labels[index, :usable] = targets[:usable]

        return labels

    def _as_segments(self, segments: Any) -> list[Any]:
        r"""Normalizes diarization entries to `(speaker, start, end)` tuples."""
        segment_type = _resolve_helper("diarization_segment", _Segment)
        normalized: list[Any] = []
        for segment in segments:
            if (
                hasattr(segment, "speaker")
                and hasattr(segment, "start")
                and hasattr(segment, "end")
            ):
                normalized.append(segment)
                continue
            if isinstance(segment, Mapping):
                start, end, speaker = _segment_fields(segment)
            elif isinstance(segment, Sequence) and not isinstance(segment, (str, bytes)):
                if len(segment) < 3:
                    raise ValueError(
                        "A diarization tuple must hold (speaker, start, end), got "
                        f"{segment!r}."
                    )
                speaker, start, end = segment[0], float(segment[1]), float(segment[2])
            else:
                raise TypeError(f"Unsupported diarization entry: {segment!r}.")
            if start is None or end is None:
                continue
            normalized.append(segment_type(str(speaker), float(start), float(end)))
        return normalized

    def _initial_speakers(self, segments: Sequence[Any], offset_seconds: float) -> list[str]:
        r"""Speakers active in the frame preceding a crop.

        Without this, the `change` channel would fire on the first frame of
        every crop that starts in the middle of a turn.
        """
        if offset_seconds <= 0.0:
            return []
        previous_start = offset_seconds - self.frame_stride_seconds
        return sorted(
            {
                str(segment.speaker)
                for segment in segments
                if float(segment.start) < offset_seconds and float(segment.end) > previous_start
            }
        )
