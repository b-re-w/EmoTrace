r"""Derivation of the per-frame speaker targets from a diarization.

The corpus stores speaker turns as time intervals that may overlap, follow each other back to back,
or leave a gap. `speaker_frame_targets` turns them into the three binary channels the speaker head
is trained on: `speech` when anybody is active, `overlap` when at least two speakers are, and
`change` when the set of active speakers differs from the previous frame.

Every expectation below is written out by hand or produced by an independent brute-force reference
that decides frame membership from the intersection of the frame support `[t * stride, (t + 1) *
stride)` with the turn, which is the rule the docstring of the helper states.
"""

import numpy as np
import pytest

from ser_project.datasets.base import (
    DiarizationSegment,
    frame_span,
    synthetic_diarization_segments,
)
from ser_project.datasets.collator import SPEAKER_TARGET_NAMES, DataCollatorForEmoTrace
from ser_project.datasets.msp_conversation import segments_within, speaker_frame_targets


STRIDE = 0.02
SPEECH = SPEAKER_TARGET_NAMES.index("speech")
OVERLAP = SPEAKER_TARGET_NAMES.index("overlap")
CHANGE = SPEAKER_TARGET_NAMES.index("change")


def active_speakers(segments, frame, stride=STRIDE, time_offset=0.0):
    r"""Speakers whose turn intersects the nominal support of one frame.

    Args:
        segments (`Sequence` of `DiarizationSegment`): Turns, in their own time reference.
        frame (`int`): Frame index.
        stride (`float`): Frame hop in seconds.
        time_offset (`float`): Time of frame 0 in the reference of `segments`.

    Returns:
        `set[str]`: the active speaker labels.
    """
    start = time_offset + frame * stride
    end = time_offset + (frame + 1) * stride
    return {segment.speaker for segment in segments if segment.start < end and segment.end > start}


def reference_targets(segments, num_frames, stride=STRIDE, time_offset=0.0, initial_speakers=()):
    r"""Brute-force `(num_frames, 3)` speaker targets, written from the definition."""
    previous = set(initial_speakers)
    rows = []
    for frame in range(num_frames):
        current = active_speakers(segments, frame, stride, time_offset)
        rows.append(
            [float(len(current) >= 1), float(len(current) >= 2), float(current != previous)]
        )
        previous = current
    return np.asarray(rows, dtype=np.float32).reshape(num_frames, 3)


def frames_where(targets, channel):
    r"""Indices at which one channel of a `(num_frames, 3)` target array is active."""
    return np.flatnonzero(targets[:, channel]).tolist()


# ---------------------------------------------------------------------------
# A hand-built conversation
# ---------------------------------------------------------------------------


def test_hand_built_conversation_covers_every_pattern():
    r"""Silence, a single speaker, simultaneous speech and a back-to-back turn change.

    The turns are `A [0.10, 0.50)`, `B [0.40, 0.80)`, `A [1.00, 1.20)`, `B [1.20, 1.60)`, so at a
    20 ms hop `A` holds frames 5 to 24, the two speakers share frames 20 to 24, `B` runs on to frame
    39, frames 40 to 49 are silent, and the last two turns touch at frame 60 without overlapping.
    """
    targets = speaker_frame_targets(synthetic_diarization_segments(), 90)

    assert frames_where(targets, SPEECH) == list(range(5, 40)) + list(range(50, 80))
    assert frames_where(targets, OVERLAP) == list(range(20, 25))
    assert frames_where(targets, CHANGE) == [5, 20, 25, 40, 50, 60, 80]
    assert targets.dtype == np.float32
    assert set(np.unique(targets).tolist()) <= {0.0, 1.0}


def test_simultaneous_speech_raises_overlap_only_where_both_speak():
    r"""Two turns sharing 100 ms raise `overlap` on exactly those five frames."""
    segments = [DiarizationSegment("A", 0.0, 0.20), DiarizationSegment("B", 0.10, 0.30)]
    targets = speaker_frame_targets(segments, 20)

    assert frames_where(targets, OVERLAP) == list(range(5, 10))
    assert frames_where(targets, SPEECH) == list(range(0, 15))
    assert frames_where(targets, CHANGE) == [0, 5, 10, 15]


def test_adjacent_turns_leave_no_gap_and_no_overlap():
    r"""A turn starting where the previous one ends changes the speaker without a silent frame."""
    segments = [DiarizationSegment("A", 0.0, 0.10), DiarizationSegment("B", 0.10, 0.20)]
    targets = speaker_frame_targets(segments, 12)

    assert frames_where(targets, SPEECH) == list(range(0, 10))
    assert frames_where(targets, OVERLAP) == []
    assert frames_where(targets, CHANGE) == [0, 5, 10]


def test_gap_between_turns_is_silence():
    r"""A pause between two turns is reported as silence and bounded by two changes."""
    segments = [DiarizationSegment("A", 0.0, 0.10), DiarizationSegment("B", 0.20, 0.30)]
    targets = speaker_frame_targets(segments, 20)

    assert frames_where(targets, SPEECH) == list(range(0, 5)) + list(range(10, 15))
    assert frames_where(targets, OVERLAP) == []
    assert frames_where(targets, CHANGE) == [0, 5, 10, 15]


def test_same_speaker_speaking_twice_still_changes_on_the_pause():
    r"""`change` follows the active set, so a speaker resuming after a pause fires it twice."""
    segments = [DiarizationSegment("A", 0.0, 0.10), DiarizationSegment("A", 0.20, 0.30)]
    targets = speaker_frame_targets(segments, 20)

    assert frames_where(targets, CHANGE) == [0, 5, 10, 15]
    assert frames_where(targets, OVERLAP) == []


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (0.031, 0.035, [1]),
        (0.039, 0.041, [1, 2]),
        (0.031, 0.031, [1]),
    ],
)
def test_a_turn_shorter_than_a_frame_is_never_dropped(start, end, expected):
    r"""Any turn touching the grid marks at least one frame, however short it is."""
    targets = speaker_frame_targets([DiarizationSegment("A", start, end)], 5)

    assert frames_where(targets, SPEECH) == expected


def test_turns_outside_the_window_are_clipped_away():
    r"""Turns that end before the first frame or start after the last one contribute nothing."""
    segments = [DiarizationSegment("A", -1.0, -0.5), DiarizationSegment("B", 5.0, 6.0)]
    targets = speaker_frame_targets(segments, 10)

    assert not targets.any()


# ---------------------------------------------------------------------------
# Crops
# ---------------------------------------------------------------------------


def test_change_on_the_first_frame_depends_on_the_previous_frame():
    r"""A crop starting inside a turn does not report a speaker change at its first frame."""
    segments = synthetic_diarization_segments()

    without_context = speaker_frame_targets(segments, 10, time_offset=0.20)
    with_context = speaker_frame_targets(segments, 10, time_offset=0.20, initial_speakers=["1"])

    assert with_context[0, SPEECH] == 1.0
    assert without_context[0, CHANGE] == 1.0
    assert with_context[0, CHANGE] == 0.0
    np.testing.assert_array_equal(without_context[1:], with_context[1:])


def test_a_windowed_crop_matches_a_slice_of_the_whole_part():
    r"""Reading a window through `time_offset` gives the frames of the full rasterization."""
    segments = synthetic_diarization_segments()
    offset_frames, length = 22, 30

    whole = speaker_frame_targets(segments, 90)
    previous = active_speakers(segments, offset_frames - 1)
    crop = speaker_frame_targets(
        segments,
        length,
        time_offset=offset_frames * STRIDE,
        initial_speakers=sorted(previous),
    )

    np.testing.assert_array_equal(crop, whole[offset_frames : offset_frames + length])


def test_segments_within_clips_and_rebases_the_turn_list():
    r"""A conversation part is cut out of the conversation before its frames are rasterized."""
    segments = [
        DiarizationSegment("A", 0.0, 2.0),
        DiarizationSegment("B", 1.5, 3.5),
        DiarizationSegment("C", 3.0, 3.0 + 1e-9),
        DiarizationSegment("D", 5.0, 6.0),
    ]
    clipped = segments_within(segments, 1.0, 3.0)

    assert [segment.speaker for segment in clipped] == ["A", "B"]
    assert clipped[0] == DiarizationSegment("A", 0.0, 1.0)
    assert clipped[1] == DiarizationSegment("B", 0.5, 2.0)
    # A turn touching the window only at its boundary carries no frame and is dropped.
    assert all(segment.speaker != "D" for segment in clipped)


# ---------------------------------------------------------------------------
# Degenerate input and invariants
# ---------------------------------------------------------------------------


def test_empty_diarization_and_empty_window():
    r"""No turn means no target, and a zero length window is an empty array, not an error."""
    assert speaker_frame_targets([], 4).shape == (4, 3)
    assert not speaker_frame_targets([], 4).any()
    assert speaker_frame_targets(synthetic_diarization_segments(), 0).shape == (0, 3)

    with pytest.raises(ValueError, match="non-negative"):
        speaker_frame_targets([], -1)


@pytest.mark.parametrize("seed", range(6))
def test_matches_the_brute_force_reference(seed):
    r"""Random overlapping turns reproduce the definition frame by frame."""
    rng = np.random.default_rng(seed)
    num_frames = 60
    segments = []
    for index in range(6):
        start = int(rng.integers(0, num_frames - 1))
        end = start + int(rng.integers(1, 12))
        segments.append(DiarizationSegment(f"S{index % 3}", start * STRIDE, end * STRIDE))
    initial = sorted({segment.speaker for segment in segments if rng.random() < 0.3})

    targets = speaker_frame_targets(segments, num_frames, initial_speakers=initial)
    expected = reference_targets(segments, num_frames, initial_speakers=initial)

    np.testing.assert_array_equal(targets, expected)


@pytest.mark.parametrize("seed", range(6))
def test_overlap_implies_speech(seed):
    r"""The channels are nested: a frame with simultaneous speech is a frame with speech."""
    rng = np.random.default_rng(seed)
    segments = [
        DiarizationSegment(
            f"S{index % 4}", float(rng.uniform(0.0, 1.0)), float(rng.uniform(1.0, 2.0))
        )
        for index in range(8)
    ]
    targets = speaker_frame_targets(segments, 100)

    assert np.all(targets[:, OVERLAP] <= targets[:, SPEECH])


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (0.0, 0.02, (0, 1)),
        (0.0, 0.021, (0, 2)),
        (0.019, 0.02, (0, 1)),
        (0.02, 0.06, (1, 3)),
        (-1.0, 0.04, (0, 2)),
        (10.0, 11.0, (5, 5)),
    ],
)
def test_frame_span_boundaries(start, end, expected):
    r"""The interval to frame conversion is half open and clipped to the available frames."""
    assert frame_span(start, end, 5) == expected


def test_frame_span_rejects_a_reversed_interval():
    r"""A turn ending before it starts is a corrupt annotation, not an empty span."""
    with pytest.raises(ValueError, match="precedes"):
        frame_span(1.0, 0.5, 10)


# ---------------------------------------------------------------------------
# The path the collator takes
# ---------------------------------------------------------------------------


def test_collator_rasterizes_mapping_style_segments():
    r"""Diarization entries straight out of the release JSON reach the same targets."""
    collator = DataCollatorForEmoTrace(include_ctc=False)
    release_style = [
        {"Start_Time": "0.10", "End_Time": "0.50", "Speaker": "1", "Gender": "Female"},
        {"Start_Time": "0.40", "End_Time": "0.80", "Speaker": "2", "Gender": "Male"},
    ]
    batch = collator([{"input_values": np.zeros(16000, np.float32), "speaker_segments": release_style}])

    frames = batch["spk_labels"].shape[1]
    expected = reference_targets(
        [DiarizationSegment("1", 0.10, 0.50), DiarizationSegment("2", 0.40, 0.80)], frames
    )
    np.testing.assert_array_equal(batch["spk_labels"][0].numpy(), expected)


def test_collator_carries_the_speaker_of_the_previous_frame_into_a_crop():
    r"""The collator supplies the initial speakers itself, from the segments of the whole part."""
    collator = DataCollatorForEmoTrace(include_ctc=False)
    segments = [DiarizationSegment("1", 0.0, 1.0)]
    offset_frames = 10
    batch = collator(
        [
            {
                "input_values": np.zeros(16000, np.float32),
                "speaker_segments": segments,
                "vad_trace_offset": offset_frames,
            }
        ]
    )

    frames = batch["spk_labels"].shape[1]
    expected = reference_targets(
        segments,
        frames,
        time_offset=offset_frames * STRIDE,
        initial_speakers=active_speakers(segments, offset_frames - 1),
    )
    np.testing.assert_array_equal(batch["spk_labels"][0].numpy(), expected)
    assert batch["spk_labels"][0, 0, CHANGE] == 0.0
