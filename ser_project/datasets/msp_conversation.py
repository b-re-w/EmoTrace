r"""Loaders for the MSP-Conversation release.

The module turns the released corpus into a [`datasets.Dataset`] with one row
per conversation part, regenerates the emotional traces on the 20 ms backbone
frame grid with the official utilities shipped inside the release, and derives
the per-frame speaker targets from the manual diarization.

Every function accepts either a release directory or a
[`~ser_project.datasets.base.CorpusPaths`] and raises
[`~ser_project.datasets.base.CorpusLayoutError`] with the expected layout when a
required file is absent, so a partially extracted release fails with an
actionable message instead of a `KeyError` deep inside a parser.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .base import (
    ATTRIBUTE_DIRECTORIES,
    ATTRIBUTES,
    DEFAULT_VALUE_SCALE,
    EVAL_HOP_SECONDS,
    FRAME_STRIDE_SECONDS,
    PARTITIONS,
    SPEAKER_TARGETS,
    TARGET_SAMPLING_RATE,
    CorpusLayoutError,
    CorpusPaths,
    DiarizationSegment,
    TraceSet,
    frame_span,
    num_frames_for_duration,
    require_path,
    resolve_corpus_root,
    synthetic_trace_values,
)

logger = logging.getLogger(__name__)

TRACE_WINDOW_MS = 120
TRACE_TIME_DELAY_MS = 3000
EVAL_TRACE_OVERLAP_MS = 20
FRAME_TRACE_OVERLAP_MS = 100
TRACE_MANIFEST_NAME = ".make_traces.json"

_UTILS_OPTIONAL_DEPENDENCIES = ("pydub", "librosa")
_UTILS_MODULE_CACHE: dict[str, types.ModuleType] = {}


class _MissingDependency:
    r"""Placeholder standing in for an attribute of an uninstalled package.

    `msp_conv_utils` imports `pydub` and `librosa` at module level even though
    only its audio splitting helpers need them. Substituting this object lets the
    trace helpers run in an environment without those packages while any call
    that genuinely needs them fails with an install hint.
    """

    def __init__(self, module_name: str, attribute: str):
        self._module_name = module_name
        self._attribute = attribute

    def _fail(self, used_as: str) -> None:
        raise ImportError(
            f"'{self._module_name}.{self._attribute}' is required by this "
            f"msp_conv_utils function ({used_as}), but '{self._module_name}' is "
            f"not installed. Install it with 'uv add {self._module_name}'."
        )

    def __getattr__(self, name: str) -> Any:
        self._fail(f"attribute '{name}'")

    def __call__(self, *args, **kwargs) -> Any:
        self._fail("called")


def _stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__getattr__ = lambda attribute: _MissingDependency(name, attribute)
    return module


def load_msp_conv_utils(paths: Any = None) -> types.ModuleType:
    r"""Import the official `msp_conv_utils.py` shipped inside the release.

    The file is loaded from disk rather than vendored so that the released
    median filter, trace grid and time conversions stay authoritative.

    Args:
        paths (`str`, `os.PathLike` or `CorpusPaths`, *optional*):
            Release directory or resolved paths.

    Returns:
        `types.ModuleType`: The loaded module, cached per file location.

    Raises:
        CorpusLayoutError: `msp_conv_utils.py` is not in the release directory.
    """
    corpus = CorpusPaths.from_root(paths)
    utils_file = require_path(
        corpus.utils_file, "official utilities 'msp_conv_utils.py'", root=corpus.root
    )
    key = str(utils_file)
    if key in _UTILS_MODULE_CACHE:
        return _UTILS_MODULE_CACHE[key]

    installed_stubs = []
    for dependency in _UTILS_OPTIONAL_DEPENDENCIES:
        if dependency in sys.modules or importlib.util.find_spec(dependency) is not None:
            continue
        sys.modules[dependency] = _stub_module(dependency)
        installed_stubs.append(dependency)

    spec = importlib.util.spec_from_file_location("msp_conv_utils", utils_file)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for dependency in installed_stubs:
            sys.modules.pop(dependency, None)

    _UTILS_MODULE_CACHE[key] = module
    return module


def read_partitions(paths: Any = None) -> dict[str, str]:
    r"""Read `partitions.txt`.

    Returns:
        `dict`: Conversation identifier to partition name, one of
        `("Train", "Development", "Test1", "Test2")`.
    """
    corpus = CorpusPaths.from_root(paths)
    partitions_file = require_path(
        corpus.partitions_file, "partition list 'partitions.txt'", root=corpus.root
    )
    partitions: dict[str, str] = {}
    for line in partitions_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        conversation_id, partition = line.split(";")
        if partition not in PARTITIONS:
            raise ValueError(
                f"Unknown partition {partition!r} for {conversation_id!r}; "
                f"expected one of {PARTITIONS}."
            )
        partitions[conversation_id] = partition
    return partitions


def _read_time_labels(path: Path) -> dict[str, tuple[float, float]]:
    times: dict[str, tuple[float, float]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        name, start, end = line.split(";")
        times[name] = (float(start), float(end))
    return times


def read_part_times(paths: Any = None) -> dict[str, tuple[float, float]]:
    r"""Read the start and end time of every conversation part.

    The release expresses regular parts relative to the source podcast and Test2
    parts relative to their conversation; both are converted here so that all
    returned times are relative to the start of the conversation audio, which is
    also the reference of the diarization files.

    Returns:
        `dict`: Part identifier to `(start_seconds, end_seconds)` relative to
        the conversation.
    """
    corpus = CorpusPaths.from_root(paths)
    conversations = _read_time_labels(
        require_path(
            corpus.conversations_file,
            "conversation timing 'Time_Labels/conversations.txt'",
            root=corpus.root,
        )
    )
    parts = _read_time_labels(
        require_path(
            corpus.conversation_parts_file,
            "part timing 'Time_Labels/conversation_parts.txt'",
            root=corpus.root,
        )
    )

    part_times: dict[str, tuple[float, float]] = {}
    for part_id, (start, end) in parts.items():
        conversation_id = conversation_of(part_id)
        if conversation_id not in conversations:
            raise CorpusLayoutError(
                f"Conversation {conversation_id!r} of part {part_id!r} is absent "
                "from 'Time_Labels/conversations.txt'.",
                corpus.root,
            )
        offset = conversations[conversation_id][0]
        part_times[part_id] = (start - offset, end - offset)

    if corpus.test2_conversation_parts_file.exists():
        for part_id, (start, end) in _read_time_labels(
            corpus.test2_conversation_parts_file
        ).items():
            part_times[part_id] = (start, end)
    return part_times


def conversation_of(part_id: str) -> str:
    r"""Return the conversation identifier of a conversation part identifier."""
    return part_id.rsplit("_", 1)[0]


def part_index_of(part_id: str) -> int:
    r"""Return the trailing part number of a conversation part identifier."""
    return int(part_id.rsplit("_", 1)[1])


def load_diarization(conversation_id: str, paths: Any = None) -> list[DiarizationSegment]:
    r"""Load the manual speaker diarization of one conversation.

    Args:
        conversation_id (`str`):
            Identifier such as `"MSP-Conversation_0002"`.

    Returns:
        `list` of `DiarizationSegment`: Turns sorted by start time, with times in
        seconds relative to the conversation. Turns may overlap, which is what
        the `overlap` target is derived from.
    """
    corpus = CorpusPaths.from_root(paths)
    diarization_file = require_path(
        corpus.diarization_file(conversation_id),
        f"speaker diarization of {conversation_id}",
        root=corpus.root,
    )
    with diarization_file.open() as handle:
        entries = json.load(handle)

    segments = [
        DiarizationSegment(
            speaker=str(entry["Speaker"]),
            start=float(entry["Start_Time"]),
            end=float(entry["End_Time"]),
        )
        for entry in entries
    ]
    return sorted(segments, key=lambda segment: (segment.start, segment.end))


def segments_within(
    segments: Iterable[DiarizationSegment],
    start: float,
    end: float,
    *,
    rebase: bool = True,
) -> list[DiarizationSegment]:
    r"""Clip turns to a time window and express them relative to its start.

    Args:
        segments (`Iterable` of `DiarizationSegment`):
            Turns to clip, in the same reference as `start` and `end`.
        start (`float`), end (`float`):
            Window bounds in seconds.
        rebase (`bool`, *optional*, defaults to `True`):
            Subtract `start` from the surviving turns.

    Returns:
        `list` of `DiarizationSegment`: Turns intersecting the window, truncated
        to it. Turns touching the window only at a boundary are dropped.
    """
    clipped: list[DiarizationSegment] = []
    for segment in segments:
        lower = max(segment.start, start)
        upper = min(segment.end, end)
        if upper <= lower:
            continue
        turn = DiarizationSegment(segment.speaker, lower, upper)
        clipped.append(turn.shifted(start) if rebase else turn)
    return clipped


def speaker_frame_targets(
    segments: Sequence[DiarizationSegment],
    num_frames: int,
    *,
    frame_stride_seconds: float = FRAME_STRIDE_SECONDS,
    time_offset: float = 0.0,
    initial_speakers: Sequence[str] | None = None,
) -> np.ndarray:
    r"""Derive the per-frame binary speaker targets from diarization turns.

    A frame is assigned to a turn as soon as its nominal
    `[t * stride, (t + 1) * stride)` support intersects the turn, so no turn is
    ever dropped, however short.

    The three channels are, in order:

    - `speech`: at least one speaker is active in the frame,
    - `overlap`: two or more speakers are simultaneously active in the frame,
    - `change`: the set of active speakers differs from that of the previous
      frame.

    Args:
        segments (`Sequence` of `DiarizationSegment`):
            Speaker turns; overlapping turns are expected and required for the
            `overlap` channel to fire.
        num_frames (`int`):
            Number of frames `T` to produce.
        frame_stride_seconds (`float`, *optional*, defaults to `0.02`):
            Hop of the frame grid.
        time_offset (`float`, *optional*, defaults to `0.0`):
            Time of the first frame in the reference of `segments`, which lets a
            window be cut out of a longer turn list without rebasing it first.
        initial_speakers (`Sequence` of `str`, *optional*):
            Speakers active in the frame preceding the window. Defaults to
            silence, so `change` fires on the very first frame when somebody is
            already speaking there.

    Returns:
        `np.ndarray`: Array of shape `(T, 3)` and dtype `float32` holding zeros
        and ones, ordered as `("speech", "overlap", "change")`.

    Raises:
        ValueError: `num_frames` is negative or a turn ends before it starts.
    """
    if num_frames < 0:
        raise ValueError(f"num_frames must be non-negative, got {num_frames}.")
    targets = np.zeros((num_frames, len(SPEAKER_TARGETS)), dtype=np.float32)
    if num_frames == 0:
        return targets

    speakers = sorted({segment.speaker for segment in segments} | set(initial_speakers or ()))
    if not speakers:
        return targets
    speaker_index = {speaker: index for index, speaker in enumerate(speakers)}

    activity = np.zeros((num_frames, len(speakers)), dtype=bool)
    for segment in segments:
        first, last = frame_span(
            segment.start - time_offset,
            segment.end - time_offset,
            num_frames,
            frame_stride_seconds=frame_stride_seconds,
        )
        if last > first:
            activity[first:last, speaker_index[segment.speaker]] = True

    previous = np.empty_like(activity)
    previous[1:] = activity[:-1]
    previous[0] = False
    for speaker in initial_speakers or ():
        previous[0, speaker_index[speaker]] = True

    active_count = activity.sum(axis=1)
    targets[:, SPEAKER_TARGETS.index("speech")] = active_count >= 1
    targets[:, SPEAKER_TARGETS.index("overlap")] = active_count >= 2
    targets[:, SPEAKER_TARGETS.index("change")] = (activity != previous).any(axis=1)
    return targets


def read_trace_csv(path: os.PathLike[str] | str) -> tuple[np.ndarray, np.ndarray]:
    r"""Read one trace CSV produced by the official `make_traces`.

    Args:
        path (`os.PathLike` or `str`):
            CSV file with a `Central_Time,Start_Time,End_Time,Mean,<workers...>`
            header.

    Returns:
        `tuple` of `np.ndarray`: The central times in seconds and the mean trace,
        both of shape `(N,)` and on the raw `[-100, 100]` scale.
    """
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"Empty trace file: {path}") from error
        if "Central_Time" not in header or "Mean" not in header:
            raise ValueError(
                f"Trace file {path} is missing the 'Central_Time'/'Mean' header."
            )
        time_column = header.index("Central_Time")
        mean_column = header.index("Mean")
        times, values = [], []
        for row in reader:
            if not row:
                continue
            times.append(float(row[time_column]))
            values.append(float(row[mean_column]))
    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64)


def load_eval_traces(
    part_id: str,
    paths: Any = None,
    *,
    value_scale: float = DEFAULT_VALUE_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Load the released 100 ms traces of one conversation part.

    Args:
        part_id (`str`):
            Identifier such as `"MSP-Conversation_0002_1"`.
        value_scale (`float`, *optional*, defaults to `0.01`):
            Factor applied to the raw `[-100, 100]` values; the default maps
            them onto `[-1, 1]`.

    Returns:
        `tuple` of `np.ndarray`: Central times of shape `(N,)` and values of
        shape `(N, 3)` ordered as `("valence", "arousal", "dominance")`.
    """
    return _load_traces(part_id, paths, frame_rate="eval", value_scale=value_scale)


def _load_traces(
    part_id: str,
    paths: Any,
    *,
    frame_rate: str,
    value_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    corpus = CorpusPaths.from_root(paths)
    times: np.ndarray | None = None
    columns = []
    for attribute in ATTRIBUTES:
        directory = corpus.attribute_dir(attribute, frame_rate=frame_rate)
        path = directory / f"{part_id}.csv"
        if not path.exists():
            hint = (
                "Run 'ensure_frame_traces' to regenerate them."
                if frame_rate == "frame"
                else "The 'Traces' directory ships with the release."
            )
            raise CorpusLayoutError(
                f"Missing {attribute} trace of {part_id!r}: {path}. {hint}",
                corpus.root,
            )
        attribute_times, attribute_values = read_trace_csv(path)
        if times is None:
            times = attribute_times
        elif attribute_times.shape != times.shape:
            raise ValueError(
                f"Traces of {part_id!r} disagree on their length: {attribute} has "
                f"{attribute_times.shape[0]} rows against {times.shape[0]}."
            )
        columns.append(attribute_values)
    values = np.stack(columns, axis=-1) * value_scale
    return times, values.astype(np.float32)


def eval_grid_mask(
    times: np.ndarray,
    *,
    eval_hop_seconds: float = EVAL_HOP_SECONDS,
    frame_stride_seconds: float = FRAME_STRIDE_SECONDS,
    tolerance: float = 1e-6,
) -> np.ndarray:
    r"""Mark the frames coinciding with an original 100 ms label position.

    The official `make_traces` applies the same median window and the same
    reaction lag at both hops, and the 100 ms filter positions are a subset of
    the 20 ms ones, so the frames selected here carry exactly the values of the
    released 100 ms trace.

    A frame qualifies when its central time is both an exact multiple of the
    evaluation hop and still on the regular 20 ms grid. The second condition
    rejects the trailing frames whose median window is truncated by the end of
    the audio: `make_traces` reports their clipped centre, which no longer
    matches the position of the corresponding 100 ms frame.

    Args:
        times (`np.ndarray`):
            Central times of the 20 ms trace, in seconds, as written by
            `make_traces`, i.e. starting at zero.

    Returns:
        `np.ndarray`: Boolean array of the same shape as `times`.
    """
    times = np.asarray(times, dtype=np.float64)
    regular = np.arange(times.shape[0], dtype=np.float64) * frame_stride_seconds
    on_frame_grid = np.abs(times - regular) <= tolerance
    offsets = times / eval_hop_seconds
    on_eval_grid = np.abs(offsets - np.round(offsets)) * eval_hop_seconds <= tolerance
    return on_frame_grid & on_eval_grid


def load_frame_traces(
    part_id: str,
    paths: Any = None,
    *,
    num_frames: int | None = None,
    value_scale: float = DEFAULT_VALUE_SCALE,
    use_released_eval_values: bool = False,
) -> TraceSet:
    r"""Load the 20 ms traces of one conversation part as a `TraceSet`.

    Args:
        part_id (`str`):
            Identifier such as `"MSP-Conversation_0002_1"`.
        num_frames (`int`, *optional*):
            Frame count to align on. The trace is truncated when it is longer
            and padded with `float('nan')` when it is shorter, so the result can
            be concatenated with backbone outputs without further checks.
        value_scale (`float`, *optional*, defaults to `0.01`):
            Factor applied to the raw `[-100, 100]` values.
        use_released_eval_values (`bool`, *optional*, defaults to `False`):
            Overwrite the frames selected by the evaluation mask with the values
            of the released 100 ms trace. The two disagree on roughly one
            evaluation frame in a hundred because `movingMedianFilter` repeats
            the previous output whenever a filter window contains no annotation
            sample, and that previous output is 20 ms old at the training hop
            against 100 ms old at the released hop. Enable this when the
            reported metric must be computed against the released labels
            themselves.

    Returns:
        `TraceSet`: Values of shape `(T, 3)`, central times of shape `(T,)` and
        the boolean evaluation mask of shape `(T,)`.
    """
    times, values = _load_traces(
        part_id, paths, frame_rate="frame", value_scale=value_scale
    )
    mask = eval_grid_mask(times)

    if use_released_eval_values:
        _, released = _load_traces(
            part_id, paths, frame_rate="eval", value_scale=value_scale
        )
        selected = np.flatnonzero(mask)
        available = min(selected.shape[0], released.shape[0])
        values[selected[:available]] = released[:available]
        if available < selected.shape[0]:
            mask[selected[available:]] = False

    if num_frames is not None:
        available = min(num_frames, values.shape[0])
        padded_values = np.full((num_frames, values.shape[1]), np.nan, dtype=np.float32)
        padded_times = np.full((num_frames,), np.nan, dtype=np.float64)
        padded_mask = np.zeros((num_frames,), dtype=bool)
        padded_values[:available] = values[:available]
        padded_times[:available] = times[:available]
        padded_mask[:available] = mask[:available]
        values, times, mask = padded_values, padded_times, padded_mask

    return TraceSet(values=values, times=times, eval_mask=mask)


def _annotated_part_ids(paths: Any = None) -> list[str]:
    corpus = CorpusPaths.from_root(paths)
    directory = require_path(
        corpus.annotations_dir / "Valence",
        "raw annotations 'Annotations/Valence'",
        root=corpus.root,
        non_empty=True,
    )
    part_ids = {
        "_".join(path.stem.split("_")[:3]) for path in directory.glob("*.csv")
    }
    return sorted(part_ids)


def trace_hop_seconds(path: os.PathLike[str] | str, *, num_rows: int = 8) -> float:
    r"""Infer the hop of a trace file from its first central times.

    Args:
        path (`os.PathLike` or `str`):
            Trace CSV to inspect.
        num_rows (`int`, *optional*, defaults to `8`):
            Number of leading rows used for the estimate.

    Returns:
        `float`: The hop in seconds, `float('nan')` when the file has fewer than
        two rows.
    """
    times, _ = read_trace_csv(path)
    if times.shape[0] < 2:
        return float("nan")
    return float(np.median(np.diff(times[:num_rows])))


def _trace_set_is_complete(
    directory: Path, num_parts: int, hop_seconds: float, *, tolerance: float = 1e-6
) -> bool:
    r"""Check that a trace directory holds one file per part at the wanted hop."""
    for attribute in ATTRIBUTES:
        attribute_dir = directory / ATTRIBUTE_DIRECTORIES[attribute]
        if not attribute_dir.is_dir():
            return False
        files = sorted(attribute_dir.glob("*.csv"))
        if len(files) != num_parts:
            return False
        if abs(trace_hop_seconds(files[0]) - hop_seconds) > tolerance:
            return False
    return True


def _trace_manifest(directory: Path) -> dict[str, Any] | None:
    manifest = directory / TRACE_MANIFEST_NAME
    if not manifest.exists():
        return None
    try:
        with manifest.open() as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def ensure_traces(
    paths: Any = None,
    *,
    save_dirname: str,
    window: int = TRACE_WINDOW_MS,
    overlap: int,
    time_delay: int = TRACE_TIME_DELAY_MS,
    force: bool = False,
) -> Path:
    r"""Generate a trace set with the official `make_traces`, once.

    A manifest file recording the filter settings and the number of parts is
    written next to the traces; a later call with the same settings and a
    complete directory returns immediately. A directory that already holds one
    file per annotated part at the requested hop is adopted even without a
    manifest, so traces generated by a direct `make_traces` call are reused.

    The traces are written to a staging directory and moved into place only once
    `make_traces` has returned, so an interrupted or concurrent run can never be
    mistaken for a complete trace set.

    Args:
        save_dirname (`str`):
            Directory name inside the release that receives the traces.
        window (`int`, *optional*, defaults to `120`):
            Median filter window in milliseconds.
        overlap (`int`):
            Median filter overlap in milliseconds. The hop is `window - overlap`,
            so `overlap=100` with the default window yields the 20 ms grid and
            `overlap=20` reproduces the released 100 ms grid.
        time_delay (`int`, *optional*, defaults to `3000`):
            Reaction lag compensation in milliseconds.
        force (`bool`, *optional*, defaults to `False`):
            Regenerate even when the manifest already matches.

    Returns:
        `pathlib.Path`: The directory holding the generated traces.
    """
    corpus = CorpusPaths.from_root(paths)
    directory = corpus.root / save_dirname
    expected = {
        "window": window,
        "overlap": overlap,
        "time_delay": time_delay,
        "attributes": list(ATTRIBUTES),
    }

    manifest = _trace_manifest(directory)
    if not force and manifest is not None:
        settings_match = all(manifest.get(key) == value for key, value in expected.items())
        counts = [
            len(list((directory / ATTRIBUTE_DIRECTORIES[attribute]).glob("*.csv")))
            for attribute in ATTRIBUTES
        ]
        if settings_match and all(count == manifest.get("num_parts") for count in counts):
            return directory

    part_ids = _annotated_part_ids(corpus)
    hop_seconds = (window - overlap) / 1000.0
    if not force and _trace_set_is_complete(directory, len(part_ids), hop_seconds):
        # A complete trace set produced by a direct make_traces call carries no
        # manifest; adopt it rather than spending an hour recomputing it. Only the
        # hop is verifiable from the files, so the reaction lag is taken on trust.
        logger.info(
            "Adopting the existing trace set in %s (hop=%d ms, %d parts).",
            directory,
            round(hop_seconds * 1000),
            len(part_ids),
        )
        with (directory / TRACE_MANIFEST_NAME).open("w") as handle:
            json.dump({**expected, "num_parts": len(part_ids)}, handle, indent=2)
        return directory

    utils = load_msp_conv_utils(corpus)
    _prune_stale_staging(corpus.root, save_dirname)
    staging_name = f"{save_dirname}.staging-{os.getpid()}"
    staging = corpus.root / staging_name
    logger.info(
        "Generating traces in %s for %d parts (window=%d ms, hop=%d ms, delay=%d ms).",
        directory,
        len(part_ids),
        window,
        window - overlap,
        time_delay,
    )
    try:
        utils.make_traces(
            str(corpus.root),
            SaveFldr=staging_name,
            window=window,
            overlap=overlap,
            time_delay=time_delay,
        )
        with (staging / TRACE_MANIFEST_NAME).open("w") as handle:
            json.dump({**expected, "num_parts": len(part_ids)}, handle, indent=2)
        _replace_directory(staging, directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return directory


def _prune_stale_staging(root: Path, save_dirname: str) -> None:
    r"""Delete staging directories left behind by a killed generation run.

    Directories whose owning process is still alive are kept, so that a
    concurrent generation is never destroyed.
    """
    for directory in root.glob(f"{save_dirname}.staging-*"):
        suffix = directory.name.rsplit("-", 1)[-1]
        if not suffix.isdigit():
            continue
        try:
            os.kill(int(suffix), 0)
        except ProcessLookupError:
            shutil.rmtree(directory, ignore_errors=True)
        except (PermissionError, OSError):
            continue


def _replace_directory(source: Path, destination: Path) -> None:
    r"""Move `source` onto `destination`, replacing it as atomically as possible."""
    retired = destination.with_name(f"{destination.name}.retired-{os.getpid()}")
    if destination.exists():
        os.replace(destination, retired)
    try:
        os.replace(source, destination)
    except OSError:
        if retired.exists():
            os.replace(retired, destination)
        raise
    shutil.rmtree(retired, ignore_errors=True)


def ensure_frame_traces(paths: Any = None, *, force: bool = False, **kwargs) -> Path:
    r"""Generate the 20 ms trace set used as training targets.

    This reruns the official median filter with `overlap=100` instead of
    interpolating the released 100 ms traces, so every 20 ms value is a genuine
    median over the raw annotations. Generating the full release takes on the
    order of an hour and is cached afterwards.

    Returns:
        `pathlib.Path`: The directory holding the 20 ms traces.
    """
    corpus = CorpusPaths.from_root(paths)
    return ensure_traces(
        corpus,
        save_dirname=corpus.frame_traces_dirname,
        overlap=FRAME_TRACE_OVERLAP_MS,
        force=force,
        **kwargs,
    )


def ensure_eval_traces(paths: Any = None, *, force: bool = False, **kwargs) -> Path:
    r"""Ensure the released 100 ms trace set is present, regenerating if needed.

    Returns:
        `pathlib.Path`: The directory holding the 100 ms traces.
    """
    corpus = CorpusPaths.from_root(paths)
    directory = corpus.traces_dir
    if not force and directory.is_dir() and any(directory.glob("*/*.csv")):
        return directory
    return ensure_traces(
        corpus,
        save_dirname=directory.name,
        overlap=EVAL_TRACE_OVERLAP_MS,
        force=force,
        **kwargs,
    )


def ensure_part_audio(paths: Any = None, *, force: bool = False) -> Path:
    r"""Split the conversation audio into conversation part audio.

    Wraps the official `split_conv_audio`, which needs `pydub` and `ffmpeg`.

    Returns:
        `pathlib.Path`: The directory holding the part audio files.
    """
    corpus = CorpusPaths.from_root(paths)
    if not force and corpus.audio_parts_dir.is_dir() and any(
        corpus.audio_parts_dir.glob("*.wav")
    ):
        return corpus.audio_parts_dir
    require_path(corpus.audio_dir, "conversation audio 'Audio'", root=corpus.root, non_empty=True)
    utils = load_msp_conv_utils(corpus)
    utils.split_conv_audio(str(corpus.root), SaveFldr=corpus.audio_parts_dir.name)
    return corpus.audio_parts_dir


def _resolve_audio(
    corpus: CorpusPaths,
    part_id: str,
    conversation_id: str,
    part_start: float,
    require_audio: bool,
) -> tuple[str, float]:
    part_audio = corpus.part_audio(part_id)
    if part_audio.exists():
        return str(part_audio), 0.0
    conversation_audio = corpus.conversation_audio(conversation_id)
    if conversation_audio.exists():
        return str(conversation_audio), part_start
    if require_audio:
        raise CorpusLayoutError(
            f"No audio for part {part_id!r}: neither {part_audio} nor "
            f"{conversation_audio} exists. The 'Audio' archive may still be "
            "extracting.",
            corpus.root,
        )
    return "", part_start


def build_msp_conversation_dataset(
    paths: Any = None,
    *,
    partitions: Sequence[str] | None = None,
    part_ids: Sequence[str] | None = None,
    require_audio: bool = False,
    require_traces: bool = True,
    value_scale: float = DEFAULT_VALUE_SCALE,
    sampling_rate: int = TARGET_SAMPLING_RATE,
):
    r"""Build a [`datasets.Dataset`] with one row per conversation part.

    Args:
        paths (`str`, `os.PathLike` or `CorpusPaths`, *optional*):
            Release directory or resolved paths.
        partitions (`Sequence` of `str`, *optional*):
            Keep only these partitions, e.g. `("Train",)`. Defaults to all.
        part_ids (`Sequence` of `str`, *optional*):
            Keep only these conversation parts, which is how a small debugging
            subset is built.
        require_audio (`bool`, *optional*, defaults to `False`):
            Raise instead of emitting an empty `audio_path` when neither the part
            audio nor the conversation audio is on disk. The default tolerates an
            archive that is still extracting.
        require_traces (`bool`, *optional*, defaults to `True`):
            Skip parts without released traces, which is every Test2 part.
        value_scale (`float`, *optional*, defaults to `0.01`):
            Factor applied to the raw `[-100, 100]` trace values.
        sampling_rate (`int`, *optional*, defaults to `16000`):
            Value stored in the `sampling_rate` column.

    Returns:
        `datasets.Dataset`: Rows sorted by part identifier with the columns
        `part_id`, `conversation_id`, `part_index`, `partition`, `audio_path`,
        `audio_offset_seconds`, `sampling_rate`, `part_start_seconds`,
        `part_end_seconds`, `duration_seconds`, `num_frames`, `trace_times`,
        `valence`, `arousal`, `dominance` and `diarization`.

        `audio_offset_seconds` is where the part starts inside `audio_path`: zero
        when the part audio exists, the offset inside the conversation audio
        otherwise. `num_frames` is the number of 20 ms backbone frames the part
        audio yields. `diarization` holds `(speaker, start, end)` turns relative
        to the part.
    """
    import datasets

    corpus = CorpusPaths.from_root(paths)
    corpus.require_root()
    partition_of = read_partitions(corpus)
    part_times = read_part_times(corpus)

    selected = sorted(part_times) if part_ids is None else sorted(part_ids)
    if partitions is not None:
        unknown = set(partitions) - set(PARTITIONS)
        if unknown:
            raise ValueError(f"Unknown partitions {sorted(unknown)}; expected {PARTITIONS}.")
        partitions = set(partitions)

    diarization_cache: dict[str, list[DiarizationSegment]] = {}
    rows: list[dict[str, Any]] = []
    for part_id in selected:
        if part_id not in part_times:
            raise KeyError(f"{part_id!r} is not a conversation part of the release.")
        conversation_id = conversation_of(part_id)
        partition = partition_of.get(conversation_id)
        if partition is None:
            raise CorpusLayoutError(
                f"Conversation {conversation_id!r} is absent from 'partitions.txt'.",
                corpus.root,
            )
        if partitions is not None and partition not in partitions:
            continue

        start, end = part_times[part_id]
        try:
            trace_times, trace_values = load_eval_traces(
                part_id, corpus, value_scale=value_scale
            )
        except CorpusLayoutError:
            if require_traces:
                continue
            trace_times = np.zeros((0,), dtype=np.float64)
            trace_values = np.zeros((0, len(ATTRIBUTES)), dtype=np.float32)

        if conversation_id not in diarization_cache:
            diarization_cache[conversation_id] = load_diarization(conversation_id, corpus)
        turns = segments_within(diarization_cache[conversation_id], start, end)

        audio_path, audio_offset = _resolve_audio(
            corpus, part_id, conversation_id, start, require_audio
        )
        duration = end - start
        rows.append(
            {
                "part_id": part_id,
                "conversation_id": conversation_id,
                "part_index": part_index_of(part_id),
                "partition": partition,
                "audio_path": audio_path,
                "audio_offset_seconds": audio_offset,
                "sampling_rate": sampling_rate,
                "part_start_seconds": start,
                "part_end_seconds": end,
                "duration_seconds": duration,
                "num_frames": num_frames_for_duration(duration, sampling_rate=sampling_rate),
                "trace_times": trace_times.tolist(),
                "valence": trace_values[:, 0].tolist(),
                "arousal": trace_values[:, 1].tolist(),
                "dominance": trace_values[:, 2].tolist(),
                "diarization": [
                    {"speaker": turn.speaker, "start": turn.start, "end": turn.end}
                    for turn in turns
                ],
            }
        )

    if not rows:
        raise ValueError(
            "No conversation part matched the requested selection "
            f"(partitions={partitions}, part_ids={part_ids})."
        )

    missing_audio = sum(1 for row in rows if not row["audio_path"])
    if missing_audio:
        logger.warning(
            "%d of %d parts have no audio on disk yet; 'audio_path' is empty for them.",
            missing_audio,
            len(rows),
        )

    features = datasets.Features(
        {
            "part_id": datasets.Value("string"),
            "conversation_id": datasets.Value("string"),
            "part_index": datasets.Value("int32"),
            "partition": datasets.Value("string"),
            "audio_path": datasets.Value("string"),
            "audio_offset_seconds": datasets.Value("float64"),
            "sampling_rate": datasets.Value("int32"),
            "part_start_seconds": datasets.Value("float64"),
            "part_end_seconds": datasets.Value("float64"),
            "duration_seconds": datasets.Value("float64"),
            "num_frames": datasets.Value("int32"),
            "trace_times": datasets.List(datasets.Value("float32")),
            "valence": datasets.List(datasets.Value("float32")),
            "arousal": datasets.List(datasets.Value("float32")),
            "dominance": datasets.List(datasets.Value("float32")),
            "diarization": datasets.List(
                {
                    "speaker": datasets.Value("string"),
                    "start": datasets.Value("float64"),
                    "end": datasets.Value("float64"),
                }
            ),
        }
    )
    return datasets.Dataset.from_list(rows, features=features)


def row_diarization(row: dict[str, Any]) -> list[DiarizationSegment]:
    r"""Convert the `diarization` column of a dataset row into segments.

    Args:
        row (`dict`):
            A row of the dataset produced by `build_msp_conversation_dataset`.

    Returns:
        `list` of `DiarizationSegment`: `(speaker, start, end)` turns relative to
        the conversation part.
    """
    return [
        DiarizationSegment(turn["speaker"], float(turn["start"]), float(turn["end"]))
        for turn in row["diarization"]
    ]


def check_eval_grid_alignment(
    part_id: str,
    paths: Any = None,
    *,
    atol: float = 1e-6,
) -> dict[str, Any]:
    r"""Compare the 20 ms traces of one part against the released 100 ms traces.

    The central times selected by the evaluation mask must reproduce the
    released ones exactly; a mismatch means the two trace sets were generated
    with different filter settings and raises.

    The values are only reported, not asserted: `movingMedianFilter` repeats its
    previous output whenever a window contains no annotation sample, so on the
    sparsely annotated parts a small fraction of the evaluation frames carries a
    value that is 20 ms fresher than the released one.

    Args:
        part_id (`str`):
            Conversation part to check.
        atol (`float`, *optional*, defaults to `1e-6`):
            Absolute tolerance below which two values count as equal, on the raw
            `[-100, 100]` scale.

    Returns:
        `dict`: `num_positions`, `num_mismatched` and `max_abs_diff`, the latter
        two aggregated over the three attributes.

    Raises:
        AssertionError: The evaluation grid does not line up with the released
            trace.
    """
    corpus = CorpusPaths.from_root(paths)
    eval_times, eval_values = load_eval_traces(part_id, corpus, value_scale=1.0)
    traces = load_frame_traces(part_id, corpus, value_scale=1.0)

    selected_times = traces.times[traces.eval_mask]
    selected_values = traces.values[traces.eval_mask]
    count = min(selected_times.shape[0], eval_times.shape[0])
    if not np.allclose(selected_times[:count], eval_times[:count], atol=atol):
        raise AssertionError(
            f"Evaluation grid of {part_id!r} does not line up with the released "
            "trace; the 20 ms and 100 ms trace sets disagree on their filter settings."
        )
    deviation = np.abs(selected_values[:count] - eval_values[:count])
    return {
        "num_positions": count,
        "num_mismatched": int((deviation > atol).sum()),
        "max_abs_diff": float(deviation.max()) if count else 0.0,
    }


def build_synthetic_corpus(
    root: os.PathLike[str] | str,
    *,
    num_workers: int = 2,
    part_duration: float = 6.0,
    annotation_rate: float = 20.0,
    write_audio: bool = True,
    write_traces: bool = True,
    utils_source: os.PathLike[str] | str | None = None,
    seed: int = 0,
) -> CorpusPaths:
    r"""Write a miniature corpus with the exact layout of the release.

    The synthetic corpus contains two annotated conversations and one Test2
    conversation, and is small enough for the official `make_traces` to run on it
    in a fraction of a second, which makes every loader in this module testable
    without the real corpora.

    Args:
        root (`os.PathLike` or `str`):
            Directory to create the corpus in.
        num_workers (`int`, *optional*, defaults to `2`):
            Number of annotators per conversation part.
        part_duration (`float`, *optional*, defaults to `6.0`):
            Duration of each conversation part in seconds.
        annotation_rate (`float`, *optional*, defaults to `20.0`):
            Sampling rate of the raw annotations in hertz.
        write_audio (`bool`, *optional*, defaults to `True`):
            Write silent conversation wav files.
        write_traces (`bool`, *optional*, defaults to `True`):
            Write the 100 ms traces directly, so the dataset builder works even
            when `msp_conv_utils.py` is unavailable.
        utils_source (`os.PathLike` or `str`, *optional*):
            `msp_conv_utils.py` to copy into the synthetic corpus. Defaults to
            the file of the real release when it can be located.

    Returns:
        `CorpusPaths`: Paths of the synthetic corpus.
    """
    root = Path(root)
    corpus = CorpusPaths(root=root)
    generator = np.random.default_rng(seed)

    conversations = {
        "MSP-Conversation_0001": ("Train", 10.0, 2),
        "MSP-Conversation_0002": ("Development", 100.0, 1),
    }
    test2_conversations = {"MSP-Conversation_test2_0001": 1}

    for directory in (
        corpus.time_labels_dir,
        corpus.diarization_dir,
        corpus.audio_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for attribute in ATTRIBUTES:
        corpus.attribute_dir(attribute, frame_rate="eval").mkdir(parents=True, exist_ok=True)
        (corpus.annotations_dir / ATTRIBUTE_DIRECTORIES[attribute]).mkdir(
            parents=True, exist_ok=True
        )

    partition_lines, conversation_lines, part_lines, test2_lines = [], [], [], []
    part_ids: list[str] = []
    for conversation_id, (partition, podcast_start, num_parts) in conversations.items():
        partition_lines.append(f"{conversation_id};{partition}")
        conversation_lines.append(
            f"{conversation_id};{podcast_start:.3f};"
            f"{podcast_start + num_parts * part_duration:.3f}"
        )
        for index in range(1, num_parts + 1):
            part_id = f"{conversation_id}_{index}"
            part_ids.append(part_id)
            start = podcast_start + (index - 1) * part_duration
            part_lines.append(f"{part_id};{start:.3f};{start + part_duration:.3f}")
    for conversation_id, num_parts in test2_conversations.items():
        partition_lines.append(f"{conversation_id};Test2")
        for index in range(1, num_parts + 1):
            start = (index - 1) * part_duration
            test2_lines.append(
                f"{conversation_id}_{index};{start:.3f};{start + part_duration:.3f}"
            )

    corpus.partitions_file.write_text("\n".join(partition_lines) + "\n")
    corpus.conversations_file.write_text("\n".join(conversation_lines) + "\n")
    corpus.conversation_parts_file.write_text("\n".join(part_lines) + "\n")
    corpus.test2_conversation_parts_file.write_text("\n".join(test2_lines) + "\n")
    corpus.segments_file.write_text("{}\n")

    num_samples = int(round(part_duration * annotation_rate))
    for part_id in part_ids:
        for attribute in ATTRIBUTES:
            for worker in range(num_workers):
                worker_id = f"{worker + 1:03d}"
                path = (
                    corpus.annotations_dir
                    / ATTRIBUTE_DIRECTORIES[attribute]
                    / f"{part_id}_{worker_id}.csv"
                )
                # The reaction lag of make_traces shifts annotations back by three
                # seconds, so the raw stamps have to extend past the audio.
                stamps = np.arange(num_samples, dtype=np.float64) / annotation_rate
                stamps = stamps + 3.0
                values = np.round(
                    80.0 * np.sin(2.0 * stamps + generator.uniform(0.0, 3.0))
                )
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["Time at Media End", "01-Jan-2020 00:00:00"])
                    writer.writerow(["Multimedia File", f"{part_id}.wav"])
                    writer.writerow(["Lower Label", "Negative Affect"])
                    writer.writerow(["Upper Label", "Positive Affect"])
                    writer.writerow(["Minimum Value", -100])
                    writer.writerow(["Maximum Value", 100])
                    writer.writerow(["Number of Steps", 21])
                    writer.writerow(["Second", "Rating"])
                    writer.writerow(["%%%%%%", "%%%%%%"])
                    for stamp, value in zip(stamps, values):
                        writer.writerow([f"{stamp:.6f}", f"{value:.2f}"])

    if write_traces:
        eval_times = np.arange(0.0, part_duration, EVAL_HOP_SECONDS)
        for part_id in part_ids:
            values = synthetic_trace_values(eval_times.shape[0], seed=seed)
            for column, attribute in enumerate(ATTRIBUTES):
                path = corpus.attribute_dir(attribute, frame_rate="eval") / f"{part_id}.csv"
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["Central_Time", "Start_Time", "End_Time", "Mean", "001"])
                    for row, time in enumerate(eval_times):
                        value = float(values[row, column])
                        writer.writerow(
                            [time, max(time - 0.06, 0.0), time + 0.06, value, value]
                        )

    for conversation_id, (_, _, num_parts) in conversations.items():
        _write_synthetic_diarization(corpus, conversation_id, num_parts * part_duration)
    for conversation_id, num_parts in test2_conversations.items():
        _write_synthetic_diarization(corpus, conversation_id, num_parts * part_duration)

    if write_audio:
        import soundfile

        for conversation_id, (_, _, num_parts) in conversations.items():
            duration = num_parts * part_duration
            waveform = np.zeros(int(duration * TARGET_SAMPLING_RATE), dtype=np.float32)
            soundfile.write(
                corpus.conversation_audio(conversation_id), waveform, TARGET_SAMPLING_RATE
            )
        for conversation_id, num_parts in test2_conversations.items():
            duration = num_parts * part_duration
            waveform = np.zeros(int(duration * TARGET_SAMPLING_RATE), dtype=np.float32)
            soundfile.write(
                corpus.conversation_audio(conversation_id), waveform, TARGET_SAMPLING_RATE
            )

    if utils_source is None:
        candidate = resolve_corpus_root(missing_ok=True) / "msp_conv_utils.py"
        utils_source = candidate if candidate.exists() else None
    if utils_source is not None:
        shutil.copyfile(utils_source, corpus.utils_file)

    return corpus


def _write_synthetic_diarization(
    corpus: CorpusPaths, conversation_id: str, duration: float
) -> None:
    entries = []
    speaker = 1
    start = 0.5
    while start + 1.0 < duration:
        entries.append(
            {
                "Start_Time": f"{start:.3f}",
                "End_Time": f"{start + 1.0:.3f}",
                "Speaker": str(speaker),
                "MSP_Podcast_Speaker": "Unknown",
                "Gender": "Unknown",
            }
        )
        # Every other turn starts before the previous one ends, which produces
        # the simultaneous speech the overlap target is derived from.
        start += 0.8 if speaker == 1 else 1.2
        speaker = 2 if speaker == 1 else 1
    with corpus.diarization_file(conversation_id).open("w") as handle:
        json.dump(entries, handle, indent=4)


__all__ = [
    "EVAL_TRACE_OVERLAP_MS",
    "FRAME_TRACE_OVERLAP_MS",
    "TRACE_TIME_DELAY_MS",
    "TRACE_WINDOW_MS",
    "build_msp_conversation_dataset",
    "build_synthetic_corpus",
    "check_eval_grid_alignment",
    "conversation_of",
    "ensure_eval_traces",
    "ensure_frame_traces",
    "ensure_part_audio",
    "ensure_traces",
    "eval_grid_mask",
    "load_diarization",
    "load_eval_traces",
    "load_frame_traces",
    "load_msp_conv_utils",
    "part_index_of",
    "read_part_times",
    "read_partitions",
    "read_trace_csv",
    "row_diarization",
    "segments_within",
    "speaker_frame_targets",
    "trace_hop_seconds",
]
