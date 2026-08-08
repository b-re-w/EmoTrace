r"""Shared primitives for the corpus and pseudo-label dataset modules.

This module owns the constants, lightweight containers and path resolution
helpers that every dataset builder in `ser_project.datasets` relies on, so
that frame arithmetic and directory expectations are defined exactly once.

The frame grid used throughout the project is the one produced by the WavLM
convolutional feature extractor: one frame every 20 ms, each frame covering a
25 ms receptive field.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol, Sequence, runtime_checkable

import numpy as np

TARGET_SAMPLING_RATE = 16000
r"""Sampling rate every audio backbone in this project expects."""

FRAME_STRIDE_SECONDS = 0.02
r"""Hop between two consecutive backbone frames, in seconds."""

FRAME_LENGTH_SECONDS = 0.025
r"""Receptive field of a single backbone frame, in seconds."""

EVAL_HOP_SECONDS = 0.1
r"""Hop of the released MSP-Conversation traces, i.e. the evaluation grid."""

TRACE_VALUE_RANGE = 100.0
r"""Absolute bound of the raw annotation scale; traces live in [-100, 100]."""

DEFAULT_VALUE_SCALE = 1.0 / TRACE_VALUE_RANGE
r"""Factor mapping raw trace values onto the [-1, 1] range used by the heads."""

ATTRIBUTES = ("valence", "arousal", "dominance")
r"""Emotional attributes, in the channel order of every `(T, 3)` VAD tensor."""

ATTRIBUTE_DIRECTORIES = {
    "valence": "Valence",
    "arousal": "Arousal",
    "dominance": "Dominance",
}
r"""Mapping from attribute name to its directory name inside the release."""

SPEAKER_TARGETS = ("speech", "overlap", "change")
r"""Per-frame speaker targets, in the channel order of every `(T, 3)` tensor."""

PARTITIONS = ("Train", "Development", "Test1", "Test2")
r"""Partition names as they appear in `partitions.txt`."""

CORPUS_ROOT_ENV_VAR = "MSP_CONVERSATION_ROOT"
r"""Environment variable overriding the default corpus location."""

DEFAULT_CORPUS_DIRNAME = "MSP_CONVERSATION_2.0"
DEFAULT_FRAME_TRACES_DIRNAME = "Traces_20ms"

EXPECTED_LAYOUT = """\
<corpus root>/
    partitions.txt
    msp_conv_utils.py
    Annotations/{Valence,Arousal,Dominance}/MSP-Conversation_<conv>_<part>_<worker>.csv
    Traces/{Valence,Arousal,Dominance}/MSP-Conversation_<conv>_<part>.csv
    Time_Labels/conversations.txt
    Time_Labels/conversation_parts.txt
    Time_Labels/test2_conversation_parts.txt
    Time_Labels/segments.json
    Speaker_Diarization/MSP-Conversation_<conv>.json
    Audio/MSP-Conversation_<conv>.wav\
"""
r"""Human readable description of the MSP-Conversation release layout."""


class CorpusLayoutError(FileNotFoundError):
    r"""Raised when a required corpus file or directory is missing.

    The message always repeats the expected layout so that the caller can tell
    at a glance which part of the release has not been extracted yet.
    """

    def __init__(self, message: str, root: os.PathLike[str] | str | None = None):
        details = [message]
        if root is not None:
            details.append(f"Corpus root: {os.fspath(root)}")
        details.append("Expected layout:\n" + EXPECTED_LAYOUT)
        super().__init__("\n".join(details))


def repository_root() -> Path:
    r"""Return the repository root, i.e. the directory holding `ser_project`."""
    return Path(__file__).resolve().parents[2]


def resolve_corpus_root(
    root: os.PathLike[str] | str | None = None,
    *,
    missing_ok: bool = False,
) -> Path:
    r"""Resolve the MSP-Conversation release directory.

    Args:
        root (`os.PathLike` or `str`, *optional*):
            Explicit release directory. When omitted the value of the
            `MSP_CONVERSATION_ROOT` environment variable is used, falling back
            to `<repository>/data/MSP_CONVERSATION_2.0`.
        missing_ok (`bool`, *optional*, defaults to `False`):
            Return the resolved path even when it does not exist instead of
            raising.

    Returns:
        `pathlib.Path`: The resolved release directory.

    Raises:
        CorpusLayoutError: The resolved directory does not exist and
            `missing_ok` is `False`.
    """
    if root is not None:
        candidate = Path(root).expanduser()
    elif os.environ.get(CORPUS_ROOT_ENV_VAR):
        candidate = Path(os.environ[CORPUS_ROOT_ENV_VAR]).expanduser()
    else:
        candidate = repository_root() / "data" / DEFAULT_CORPUS_DIRNAME

    candidate = candidate.resolve() if candidate.exists() else candidate.absolute()
    if not missing_ok and not candidate.is_dir():
        raise CorpusLayoutError(
            "MSP-Conversation release directory not found. Pass an explicit "
            f"root, or set ${CORPUS_ROOT_ENV_VAR}.",
            candidate,
        )
    return candidate


def require_path(
    path: os.PathLike[str] | str,
    description: str,
    *,
    root: os.PathLike[str] | str | None = None,
    non_empty: bool = False,
) -> Path:
    r"""Check that a corpus path exists and return it.

    Args:
        path (`os.PathLike` or `str`):
            Path to check.
        description (`str`):
            What the path holds, used to build an actionable error message.
        root (`os.PathLike` or `str`, *optional*):
            Release directory quoted in the error message.
        non_empty (`bool`, *optional*, defaults to `False`):
            Also require the directory to contain at least one entry, which is
            how a still-extracting archive is detected.

    Returns:
        `pathlib.Path`: The checked path.

    Raises:
        CorpusLayoutError: The path is missing or, with `non_empty`, empty.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise CorpusLayoutError(f"Missing {description}: {resolved}", root)
    if non_empty and resolved.is_dir() and not any(resolved.iterdir()):
        raise CorpusLayoutError(
            f"Empty {description}: {resolved}. The archive may still be extracting.",
            root,
        )
    return resolved


@dataclass(frozen=True)
class CorpusPaths:
    r"""Resolved locations of every file the MSP-Conversation loaders read.

    Args:
        root (`pathlib.Path`):
            Release directory.
        frame_traces_dirname (`str`, *optional*, defaults to `"Traces_20ms"`):
            Name of the directory holding the regenerated 20 ms traces.
    """

    root: Path
    frame_traces_dirname: str = DEFAULT_FRAME_TRACES_DIRNAME

    @classmethod
    def from_root(
        cls,
        root: "os.PathLike[str] | str | CorpusPaths | None" = None,
        *,
        frame_traces_dirname: str = DEFAULT_FRAME_TRACES_DIRNAME,
        missing_ok: bool = False,
    ) -> "CorpusPaths":
        r"""Build a `CorpusPaths` from a release directory.

        Passing an existing `CorpusPaths` returns it unchanged, which lets every
        public loader accept either form for its `paths` argument.
        """
        if isinstance(root, CorpusPaths):
            return root
        return cls(
            root=resolve_corpus_root(root, missing_ok=missing_ok),
            frame_traces_dirname=frame_traces_dirname,
        )

    @property
    def audio_dir(self) -> Path:
        return self.root / "Audio"

    @property
    def audio_parts_dir(self) -> Path:
        return self.root / "Audio_Parts"

    @property
    def annotations_dir(self) -> Path:
        return self.root / "Annotations"

    @property
    def traces_dir(self) -> Path:
        return self.root / "Traces"

    @property
    def frame_traces_dir(self) -> Path:
        return self.root / self.frame_traces_dirname

    @property
    def time_labels_dir(self) -> Path:
        return self.root / "Time_Labels"

    @property
    def diarization_dir(self) -> Path:
        return self.root / "Speaker_Diarization"

    @property
    def partitions_file(self) -> Path:
        return self.root / "partitions.txt"

    @property
    def utils_file(self) -> Path:
        return self.root / "msp_conv_utils.py"

    @property
    def conversations_file(self) -> Path:
        return self.time_labels_dir / "conversations.txt"

    @property
    def conversation_parts_file(self) -> Path:
        return self.time_labels_dir / "conversation_parts.txt"

    @property
    def test2_conversation_parts_file(self) -> Path:
        return self.time_labels_dir / "test2_conversation_parts.txt"

    @property
    def segments_file(self) -> Path:
        return self.time_labels_dir / "segments.json"

    def attribute_dir(self, attribute: str, *, frame_rate: str = "eval") -> Path:
        r"""Return the trace directory of one attribute.

        Args:
            attribute (`str`):
                One of `"valence"`, `"arousal"` or `"dominance"`.
            frame_rate (`str`, *optional*, defaults to `"eval"`):
                `"eval"` selects the released 100 ms traces, `"frame"` the
                regenerated 20 ms traces.
        """
        if attribute not in ATTRIBUTE_DIRECTORIES:
            raise ValueError(
                f"Unknown attribute {attribute!r}; expected one of {ATTRIBUTES}."
            )
        if frame_rate == "eval":
            parent = self.traces_dir
        elif frame_rate == "frame":
            parent = self.frame_traces_dir
        else:
            raise ValueError(
                f"Unknown frame_rate {frame_rate!r}; expected 'eval' or 'frame'."
            )
        return parent / ATTRIBUTE_DIRECTORIES[attribute]

    def conversation_audio(self, conversation_id: str) -> Path:
        return self.audio_dir / f"{conversation_id}.wav"

    def part_audio(self, part_id: str) -> Path:
        return self.audio_parts_dir / f"{part_id}.wav"

    def diarization_file(self, conversation_id: str) -> Path:
        return self.diarization_dir / f"{conversation_id}.json"

    def require_root(self) -> Path:
        r"""Check that the release directory itself exists."""
        return require_path(self.root, "MSP-Conversation release directory", root=self.root)


class DiarizationSegment(NamedTuple):
    r"""One speaker turn, expressed as `(speaker_label, start_seconds, end_seconds)`.

    Times are relative to whichever audio the segment list is attached to: the
    release stores them relative to the conversation, while dataset rows store
    them relative to the conversation part.
    """

    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def shifted(self, offset: float) -> "DiarizationSegment":
        r"""Return the same turn with `offset` seconds subtracted from both ends."""
        return DiarizationSegment(self.speaker, self.start - offset, self.end - offset)


@runtime_checkable
class TimeSpan(Protocol):
    r"""Anything carrying a start and an end time in seconds.

    Both `DiarizationSegment` and the utterance spans produced by the
    pseudo-label module satisfy this protocol, so the frame helpers below apply
    unchanged to either.
    """

    start: float
    end: float


@dataclass
class TraceSet:
    r"""Emotional traces resampled onto the backbone frame grid.

    Args:
        values (`np.ndarray`):
            Array of shape `(T, 3)` holding valence, arousal and dominance in
            that order. `float('nan')` marks frames without a label.
        times (`np.ndarray`):
            Array of shape `(T,)` with the central time of every frame in
            seconds, `float('nan')` for padded frames.
        eval_mask (`np.ndarray`):
            Boolean array of shape `(T,)`, `True` only at frames coinciding with
            an original 100 ms label position.
        attributes (`tuple` of `str`, *optional*):
            Channel order of `values`.
    """

    values: np.ndarray
    times: np.ndarray
    eval_mask: np.ndarray
    attributes: tuple[str, ...] = ATTRIBUTES

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float32)
        self.times = np.asarray(self.times, dtype=np.float64)
        self.eval_mask = np.asarray(self.eval_mask, dtype=bool)
        if self.values.ndim != 2 or self.values.shape[1] != len(self.attributes):
            raise ValueError(
                f"values must have shape (T, {len(self.attributes)}), got "
                f"{self.values.shape}."
            )
        if self.times.shape != (self.values.shape[0],):
            raise ValueError(
                f"times must have shape ({self.values.shape[0]},), got {self.times.shape}."
            )
        if self.eval_mask.shape != (self.values.shape[0],):
            raise ValueError(
                f"eval_mask must have shape ({self.values.shape[0]},), got "
                f"{self.eval_mask.shape}."
            )

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def channel(self, attribute: str) -> np.ndarray:
        r"""Return the trace of a single attribute as a `(T,)` array."""
        return self.values[:, self.attributes.index(attribute)]


def num_frames_for_samples(
    num_samples: int,
    *,
    sampling_rate: int = TARGET_SAMPLING_RATE,
    frame_stride_seconds: float = FRAME_STRIDE_SECONDS,
    frame_length_seconds: float = FRAME_LENGTH_SECONDS,
) -> int:
    r"""Return the number of backbone frames produced from a waveform.

    This reproduces the WavLM convolutional feature extractor exactly: it uses
    no padding, so frame `t` covers samples `[stride * t, stride * t + length)`
    and the last incomplete window is dropped.

    Args:
        num_samples (`int`):
            Length of the waveform in samples.

    Returns:
        `int`: Number of frames, zero when the waveform is shorter than one
        receptive field.
    """
    stride = int(round(frame_stride_seconds * sampling_rate))
    length = int(round(frame_length_seconds * sampling_rate))
    if num_samples < length:
        return 0
    return (num_samples - length) // stride + 1


def num_frames_for_duration(
    duration_seconds: float,
    *,
    sampling_rate: int = TARGET_SAMPLING_RATE,
    **kwargs,
) -> int:
    r"""Return the number of backbone frames produced from a duration in seconds."""
    return num_frames_for_samples(
        int(round(duration_seconds * sampling_rate)),
        sampling_rate=sampling_rate,
        **kwargs,
    )


def frame_span(
    start: float,
    end: float,
    num_frames: int,
    *,
    frame_stride_seconds: float = FRAME_STRIDE_SECONDS,
    tolerance: float = 1e-9,
) -> tuple[int, int]:
    r"""Convert a time interval into a half-open range of frame indices.

    A frame belongs to the span when its nominal `[t * stride, (t + 1) * stride)`
    support intersects `[start, end)`, so a span shorter than one frame still
    marks the single frame it falls into. The result is clipped to
    `[0, num_frames]`.

    Args:
        start (`float`), end (`float`):
            Interval bounds in seconds, relative to the first frame.
        num_frames (`int`):
            Total number of frames available.

    Returns:
        `tuple` of `int`: `(first_frame, last_frame_exclusive)`; an empty range
        when the interval falls outside the frame grid.

    Raises:
        ValueError: `end` is smaller than `start`.
    """
    if end < start:
        raise ValueError(f"Interval end {end} precedes start {start}.")
    first = int(math.floor(start / frame_stride_seconds + tolerance))
    last = int(math.ceil(end / frame_stride_seconds - tolerance))
    last = max(last, first + 1)
    first = min(max(first, 0), num_frames)
    last = min(max(last, 0), num_frames)
    return first, max(last, first)


def frame_center_times(
    num_frames: int,
    *,
    frame_stride_seconds: float = FRAME_STRIDE_SECONDS,
) -> np.ndarray:
    r"""Return the start time of every frame, in seconds, as a `(T,)` array."""
    return np.arange(num_frames, dtype=np.float64) * frame_stride_seconds


def synthetic_diarization_segments() -> list[DiarizationSegment]:
    r"""Return a small deterministic turn list covering every speaker pattern.

    The returned turns contain silence, a single active speaker, simultaneous
    speech and a back-to-back speaker change, which is what the per-frame
    speaker targets have to distinguish.
    """
    return [
        DiarizationSegment("1", 0.10, 0.50),
        DiarizationSegment("2", 0.40, 0.80),
        DiarizationSegment("1", 1.00, 1.20),
        DiarizationSegment("2", 1.20, 1.60),
    ]


def synthetic_trace_values(
    num_frames: int,
    *,
    seed: int = 0,
    attributes: Sequence[str] = ATTRIBUTES,
) -> np.ndarray:
    r"""Return smooth deterministic trace values of shape `(T, len(attributes))`.

    Values stay inside the raw annotation range `[-100, 100]` so that the helper
    can stand in for a released trace file.
    """
    time = np.arange(num_frames, dtype=np.float64) * FRAME_STRIDE_SECONDS
    generator = np.random.default_rng(seed)
    phases = generator.uniform(0.0, 2.0 * np.pi, size=len(attributes))
    values = np.stack(
        [80.0 * np.sin(0.7 * time + phase) for phase in phases],
        axis=-1,
    )
    return values.astype(np.float32)


__all__ = [
    "ATTRIBUTES",
    "ATTRIBUTE_DIRECTORIES",
    "CORPUS_ROOT_ENV_VAR",
    "CorpusLayoutError",
    "CorpusPaths",
    "DEFAULT_CORPUS_DIRNAME",
    "DEFAULT_FRAME_TRACES_DIRNAME",
    "DEFAULT_VALUE_SCALE",
    "DiarizationSegment",
    "EVAL_HOP_SECONDS",
    "EXPECTED_LAYOUT",
    "FRAME_LENGTH_SECONDS",
    "FRAME_STRIDE_SECONDS",
    "PARTITIONS",
    "SPEAKER_TARGETS",
    "TARGET_SAMPLING_RATE",
    "TRACE_VALUE_RANGE",
    "TimeSpan",
    "TraceSet",
    "frame_center_times",
    "frame_span",
    "num_frames_for_duration",
    "num_frames_for_samples",
    "repository_root",
    "require_path",
    "resolve_corpus_root",
    "synthetic_diarization_segments",
    "synthetic_trace_values",
]
