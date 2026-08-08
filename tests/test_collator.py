r"""The batch dictionary produced by [`DataCollatorForEmoTrace`].

The collator owns the alignment between the waveform and every label tensor, so these tests pin
down three things:

- the keys, dtypes and shapes of the batch, and the fact that its time dimension is the number of
  frames the backbone actually returns for that waveform;
- the marking of everything that carries no supervision with `float('nan')`, padding frames
  included, since a zero is a valid attribute value and would be learned as one;
- the phase of the evaluation mask, which must select the frames of the released 100 ms grid and
  nothing else, whether the row is a whole conversation part or a crop out of one.

The rows are built by hand and no corpus file is read, apart from the trace CSVs one test writes
into a temporary directory.
"""

import numpy as np
import pytest
import torch

from ser_project.datasets.base import DiarizationSegment, num_frames_for_samples
from ser_project.datasets.collator import (
    SPEAKER_TARGET_NAMES,
    VAD_ATTRIBUTE_NAMES,
    DataCollatorForEmoTrace,
    resolved_helpers,
)
from ser_project.datasets.msp_conversation import speaker_frame_targets
from ser_project.utils.losses import ccc_loss, masked_bce_loss, masked_ctc_loss


SAMPLING_RATE = 16000
EVAL_STEP = 5
r"""Number of 20 ms frames between two positions of the 100 ms evaluation grid."""


def noise(num_samples: int, seed: int = 0) -> np.ndarray:
    r"""Returns a deterministic waveform of `num_samples` samples."""
    return np.random.default_rng(seed).standard_normal(num_samples).astype(np.float32)


@pytest.fixture
def collator() -> DataCollatorForEmoTrace:
    r"""Returns a collator with every task enabled and no corpus attached."""
    return DataCollatorForEmoTrace()


# ---------------------------------------------------------------------------
# The batch contract
# ---------------------------------------------------------------------------


def test_batch_keys_shapes_and_dtypes(collator):
    r"""The batch carries exactly the documented keys, dtypes and shapes."""
    lengths = (SAMPLING_RATE, SAMPLING_RATE // 2)
    rows = [
        {
            "input_values": noise(lengths[0]),
            "vad_labels": np.zeros((num_frames_for_samples(lengths[0]), 3), np.float32),
            "ctc_labels": [1, 2, 3],
            "speaker_segments": [{"Start_Time": "0.0", "End_Time": "0.5", "Speaker": "A"}],
        },
        {"input_values": noise(lengths[1], seed=1), "ctc_labels": [4]},
    ]
    batch = collator(rows)

    frames = num_frames_for_samples(max(lengths))
    assert set(batch) == {
        "input_values",
        "attention_mask",
        "vad_labels",
        "vad_eval_mask",
        "ctc_labels",
        "ctc_label_lengths",
        "ctc_loss_mask",
        "spk_labels",
    }
    assert batch["input_values"].shape == (2, max(lengths))
    assert batch["attention_mask"].shape == (2, max(lengths))
    assert batch["vad_labels"].shape == (2, frames, len(VAD_ATTRIBUTE_NAMES))
    assert batch["vad_eval_mask"].shape == (2, frames)
    assert batch["ctc_labels"].shape == (2, 3)
    assert batch["ctc_label_lengths"].shape == (2,)
    assert batch["ctc_loss_mask"].shape == (2,)
    assert batch["spk_labels"].shape == (2, frames, len(SPEAKER_TARGET_NAMES))

    assert batch["input_values"].dtype == torch.float32
    assert batch["attention_mask"].dtype == torch.int64
    assert batch["vad_labels"].dtype == torch.float32
    assert batch["vad_eval_mask"].dtype == torch.bool
    assert batch["ctc_labels"].dtype == torch.int64
    assert batch["ctc_label_lengths"].dtype == torch.int64
    assert batch["ctc_loss_mask"].dtype == torch.bool
    assert batch["spk_labels"].dtype == torch.float32


@pytest.mark.parametrize("num_samples", [6400, 16000, 16137])
def test_label_time_axis_matches_the_backbone_output(collator, make_causal_model, num_samples):
    r"""The time dimension of the labels is the frame count the backbone really produces."""
    model = make_causal_model()
    batch = collator([{"input_values": noise(num_samples)}])

    with torch.no_grad():
        frames = model(batch["input_values"], attention_mask=batch["attention_mask"])
        frames = frames.last_hidden_state.shape[1]

    assert batch["vad_labels"].shape[1] == frames
    assert batch["vad_eval_mask"].shape[1] == frames
    assert batch["spk_labels"].shape[1] == frames
    assert collator.num_frames(num_samples) == frames


def test_disabled_tasks_emit_no_keys():
    r"""An ablation that drops a task drops its keys rather than emitting empty tensors."""
    collator = DataCollatorForEmoTrace(include_ctc=False, include_speaker=False)
    batch = collator([{"input_values": noise(3200)}])

    assert set(batch) == {"input_values", "attention_mask", "vad_labels", "vad_eval_mask"}


def test_from_config_follows_the_backbone_geometry(make_causal_config, make_causal_model):
    r"""A collator built from a model config counts frames the way that model does."""
    config = make_causal_config()
    collator = DataCollatorForEmoTrace.from_config(config)
    model = make_causal_model(config=config)

    assert collator.conv_kernel == tuple(config.conv_kernel)
    assert collator.conv_stride == tuple(config.conv_stride)
    with torch.no_grad():
        frames = model(torch.zeros(1, 9600)).last_hidden_state.shape[1]
    assert collator.num_frames(9600) == frames


def test_collator_binds_to_the_dataset_helpers(collator):
    r"""Frame counting and speaker rasterization come from the dataset modules, not from copies."""
    collator(
        [
            {
                "input_values": noise(3200),
                "speaker_segments": [("A", 0.0, 0.1)],
            }
        ]
    )
    helpers = resolved_helpers()

    assert helpers["frame_count"] == "ser_project.datasets.base.num_frames_for_samples"
    assert helpers["eval_mask"] == "ser_project.datasets.msp_conversation.eval_grid_mask"
    assert (
        helpers["speaker_frame_targets"]
        == "ser_project.datasets.msp_conversation.speaker_frame_targets"
    )


# ---------------------------------------------------------------------------
# Waveform padding and normalization
# ---------------------------------------------------------------------------


def test_padding_is_masked_and_left_at_exactly_zero():
    r"""Padded samples are zero after normalization, so the extractor convolves over silence."""
    collator = DataCollatorForEmoTrace(waveform_mean=0.5, waveform_std=2.0)
    lengths = (16000, 9600)
    waveforms = [noise(lengths[0]), noise(lengths[1], seed=1)]
    batch = collator([{"input_values": waveform} for waveform in waveforms])

    padding = lengths[0] - lengths[1]
    for index, length in enumerate(lengths):
        expected = (torch.from_numpy(waveforms[index]) - 0.5) / 2.0
        torch.testing.assert_close(batch["input_values"][index, :length], expected)
        assert torch.equal(
            batch["attention_mask"][index, :length], torch.ones(length, dtype=torch.int64)
        )
    assert torch.equal(batch["input_values"][1, lengths[1] :], torch.zeros(padding))
    assert torch.equal(
        batch["attention_mask"][1, lengths[1] :], torch.zeros(padding, dtype=torch.int64)
    )


def test_normalization_uses_fixed_constants_not_utterance_statistics():
    r"""Two rows differing only by a gain keep that difference: no per-utterance rescaling."""
    collator = DataCollatorForEmoTrace(waveform_mean=0.0, waveform_std=4.0)
    waveform = noise(3200)
    batch = collator([{"input_values": waveform}, {"input_values": 10.0 * waveform}])

    torch.testing.assert_close(batch["input_values"][1], 10.0 * batch["input_values"][0])
    torch.testing.assert_close(batch["input_values"][0], torch.from_numpy(waveform) / 4.0)


def test_max_length_truncates_and_pad_to_multiple_of_rounds_up():
    r"""Truncation and padding are applied before the frame count is derived."""
    collator = DataCollatorForEmoTrace(max_length=8000, pad_to_multiple_of=1600, include_ctc=False)
    batch = collator([{"input_values": noise(16000)}, {"input_values": noise(3000, seed=1)}])

    assert batch["input_values"].shape == (2, 8000)
    assert int(batch["attention_mask"][0].sum()) == 8000
    assert int(batch["attention_mask"][1].sum()) == 3000
    assert batch["vad_labels"].shape[1] == num_frames_for_samples(8000)


# ---------------------------------------------------------------------------
# Attribute traces, holes and the evaluation grid
# ---------------------------------------------------------------------------


def test_frames_without_supervision_are_nan_not_zero(collator):
    r"""Padding frames and rows without labels are `nan` in both label tensors."""
    lengths = (16000, 6400)
    batch = collator(
        [
            {
                "input_values": noise(lengths[0]),
                "vad_labels": np.zeros((num_frames_for_samples(lengths[0]), 3), np.float32),
                "speaker_segments": [("A", 0.0, 1.0)],
            },
            {"input_values": noise(lengths[1], seed=1)},
        ]
    )

    own_frames = num_frames_for_samples(lengths[1])
    assert torch.isfinite(batch["vad_labels"][0]).all()
    assert torch.isnan(batch["vad_labels"][1]).all()
    assert torch.isnan(batch["spk_labels"][1]).all()
    assert not batch["vad_eval_mask"][1].any()

    padded = collator(
        [
            {"input_values": noise(lengths[0]), "speaker_segments": [("A", 0.0, 1.0)]},
            {"input_values": noise(lengths[1], seed=1), "speaker_segments": [("A", 0.0, 1.0)]},
        ]
    )
    assert torch.isfinite(padded["spk_labels"][1, :own_frames]).all()
    assert torch.isnan(padded["spk_labels"][1, own_frames:]).all()


def test_evaluation_mask_selects_the_hundred_millisecond_grid(collator):
    r"""One frame in five is an evaluation position, and only where a label is present."""
    frames = num_frames_for_samples(16000)
    labels = np.zeros((frames, 3), np.float32)
    labels[7] = np.nan
    batch = collator([{"input_values": noise(16000), "vad_labels": labels}])

    selected = torch.nonzero(batch["vad_eval_mask"][0]).flatten().tolist()
    assert selected == list(range(0, frames, EVAL_STEP))
    assert torch.isfinite(batch["vad_labels"][0][batch["vad_eval_mask"][0]]).all()

    labels[5] = np.nan
    holed = collator([{"input_values": noise(16000), "vad_labels": labels}])
    assert 5 not in torch.nonzero(holed["vad_eval_mask"][0]).flatten().tolist()


@pytest.mark.parametrize("offset", [0, 3, 5])
def test_evaluation_mask_keeps_its_phase_in_a_crop(collator, offset):
    r"""A crop starting mid-part keeps the grid of the part instead of restarting it."""
    frames = num_frames_for_samples(16000)
    batch = collator(
        [
            {
                "input_values": noise(16000),
                "vad_labels": np.zeros((frames, 3), np.float32),
                "vad_trace_offset": offset,
            }
        ]
    )

    selected = torch.nonzero(batch["vad_eval_mask"][0]).flatten().tolist()
    assert selected == [index for index in range(frames) if (index + offset) % EVAL_STEP == 0]


def test_hundred_millisecond_traces_are_placed_and_never_interpolated(collator):
    r"""A trace given on the coarse grid fills one frame in five; the rest stays unlabeled."""
    times = np.arange(10) * 0.1
    values = np.stack([np.arange(10, dtype=np.float32)] * 3, axis=-1)
    batch = collator([{"input_values": noise(16000), "vad_labels": values, "trace_times": times}])

    labels = batch["vad_labels"][0]
    labelled = torch.nonzero(torch.isfinite(labels).all(dim=-1)).flatten()
    assert labelled.tolist() == list(range(0, 10 * EVAL_STEP, EVAL_STEP))
    torch.testing.assert_close(labels[labelled, 0], torch.arange(10, dtype=torch.float32))
    assert torch.isnan(labels[1:5]).all()


def test_traces_are_read_from_a_trace_directory(tmp_path, collator):
    r"""A folder of `<Attribute>/<part>.csv` files is read and rescaled to `[-1, 1]`."""
    frames = num_frames_for_samples(16000)
    for attribute in ("Valence", "Arousal", "Dominance"):
        directory = tmp_path / attribute
        directory.mkdir()
        rows = ["Central_Time,Start_Time,End_Time,Mean"]
        rows += [f"{index * 0.02:.6f},0,0,{index}" for index in range(frames)]
        (directory / "MSP-Conversation_0002_1.csv").write_text("\n".join(rows) + "\n")

    reader = DataCollatorForEmoTrace(trace_dir=tmp_path, include_ctc=False, include_speaker=False)
    batch = reader([{"input_values": noise(16000), "part_id": "MSP-Conversation_0002_1"}])

    expected = torch.arange(frames, dtype=torch.float32) * 0.01
    torch.testing.assert_close(batch["vad_labels"][0, :, 0], expected)
    assert batch["vad_labels"].shape[1] == frames


def test_missing_trace_file_raises_an_actionable_error(tmp_path):
    r"""A trace folder that has not been regenerated yet must say so."""
    reader = DataCollatorForEmoTrace(trace_dir=tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="make_traces"):
        reader([{"input_values": noise(3200), "part_id": "MSP-Conversation_0002_1"}])


def test_attribute_columns_are_stacked_in_the_canonical_order(collator):
    r"""Rows holding one column per attribute are stacked as valence, arousal, dominance."""
    frames = num_frames_for_samples(16000)
    row = {"input_values": noise(16000)}
    for index, attribute in enumerate(VAD_ATTRIBUTE_NAMES):
        row[attribute] = np.full(frames, float(index), np.float32)
    batch = collator([row])

    for index in range(3):
        torch.testing.assert_close(
            batch["vad_labels"][0, :, index], torch.full((frames,), float(index))
        )


# ---------------------------------------------------------------------------
# Transcript targets
# ---------------------------------------------------------------------------


def test_ctc_labels_are_padded_and_measured(collator):
    r"""Targets are padded with the ignore value, and the lengths describe the real tokens."""
    batch = collator(
        [
            {"input_values": noise(16000), "ctc_labels": [3, 4, 5, 6]},
            {"input_values": noise(16000, seed=1), "ctc_labels": [7]},
        ]
    )

    assert batch["ctc_labels"].tolist() == [[3, 4, 5, 6], [7, -100, -100, -100]]
    assert batch["ctc_label_lengths"].tolist() == [4, 1]
    assert batch["ctc_loss_mask"].tolist() == [True, True]


def test_ctc_loss_mask_rejects_unusable_supervision(collator):
    r"""Empty, over-long and explicitly disabled transcripts are dropped from the CTC term."""
    short = 3200
    batch = collator(
        [
            {"input_values": noise(16000), "ctc_labels": [1, 2]},
            {"input_values": noise(16000, seed=1)},
            {"input_values": noise(short, seed=2), "ctc_labels": list(range(1, 40))},
            {"input_values": noise(16000, seed=3), "ctc_labels": [1, 2], "ctc_loss_mask": False},
        ]
    )

    assert num_frames_for_samples(short) < 39
    assert batch["ctc_loss_mask"].tolist() == [True, False, False, False]
    assert batch["ctc_label_lengths"].tolist() == [2, 0, 39, 2]


def test_padded_label_columns_are_stripped(collator):
    r"""A row coming back from an already padded column keeps only its real tokens."""
    batch = collator([{"input_values": noise(16000), "ctc_labels": [3, 4, -100, -100]}])

    assert batch["ctc_label_lengths"].tolist() == [2]
    assert batch["ctc_labels"].tolist() == [[3, 4]]


# ---------------------------------------------------------------------------
# Speaker targets
# ---------------------------------------------------------------------------


def test_speaker_targets_are_rasterized_from_the_diarization(collator):
    r"""Diarization mappings give the same targets as the corpus helper called directly."""
    segments = [
        {"Start_Time": "0.00", "End_Time": "0.20", "Speaker": "A"},
        {"Start_Time": "0.10", "End_Time": "0.40", "Speaker": "B"},
    ]
    batch = collator([{"input_values": noise(16000), "speaker_segments": segments}])

    frames = num_frames_for_samples(16000)
    expected = speaker_frame_targets(
        [DiarizationSegment("A", 0.0, 0.2), DiarizationSegment("B", 0.1, 0.4)],
        frames,
        frame_stride_seconds=0.02,
    )
    torch.testing.assert_close(batch["spk_labels"][0], torch.from_numpy(expected))


def test_speaker_targets_of_a_crop_are_windowed(collator):
    r"""A crop reads its segments in the reference of the conversation part it was cut from."""
    segments = [("A", 0.0, 0.6), ("B", 0.5, 1.2)]
    offset = 15
    batch = collator(
        [
            {
                "input_values": noise(16000),
                "speaker_segments": segments,
                "vad_trace_offset": offset,
            }
        ]
    )

    frames = num_frames_for_samples(16000)
    expected = speaker_frame_targets(
        [DiarizationSegment("A", 0.0, 0.6), DiarizationSegment("B", 0.5, 1.2)],
        frames,
        frame_stride_seconds=0.02,
        time_offset=offset * 0.02,
        initial_speakers=["A"],
    )
    torch.testing.assert_close(batch["spk_labels"][0], torch.from_numpy(expected))
    # The crop starts inside A's turn, so nothing changes on its first frame.
    assert batch["spk_labels"][0, 0, SPEAKER_TARGET_NAMES.index("change")] == 0.0


def test_precomputed_speaker_targets_are_passed_through(collator):
    r"""A row that already holds frame targets is used as it is."""
    frames = num_frames_for_samples(16000)
    targets = np.zeros((frames, 3), np.float32)
    targets[10:20, 0] = 1.0
    batch = collator([{"input_values": noise(16000), "spk_labels": targets}])

    torch.testing.assert_close(batch["spk_labels"][0], torch.from_numpy(targets))


# ---------------------------------------------------------------------------
# The batch as the losses consume it
# ---------------------------------------------------------------------------


def test_batch_feeds_the_joint_loss(collator, generator):
    r"""Every label tensor is accepted by the loss it was built for, holes included."""
    frames = num_frames_for_samples(16000)
    batch = collator(
        [
            {
                "input_values": noise(16000),
                "vad_labels": np.zeros((frames, 3), np.float32),
                "ctc_labels": [1, 2, 3],
                "speaker_segments": [("A", 0.0, 0.5)],
            },
            {"input_values": noise(6400, seed=1), "ctc_labels": [4, 5]},
        ]
    )

    time_axis = batch["vad_labels"].shape[1]
    vad_predictions = torch.randn(2, time_axis, 3, generator=generator)
    speaker_logits = torch.randn(2, time_axis, 3, generator=generator)
    ctc_logits = torch.randn(2, time_axis, 8, generator=generator)
    input_lengths = torch.tensor(
        [collator.num_frames(int(length)) for length in batch["attention_mask"].sum(dim=-1)]
    )

    vad = ccc_loss(vad_predictions, batch["vad_labels"], batch["vad_eval_mask"])
    speaker = masked_bce_loss(speaker_logits, batch["spk_labels"])
    transcript = masked_ctc_loss(
        ctc_logits,
        batch["ctc_labels"],
        input_lengths,
        batch["ctc_label_lengths"],
        batch["ctc_loss_mask"],
    )

    for loss in (vad, speaker, transcript):
        assert torch.isfinite(loss).all()


# ---------------------------------------------------------------------------
# Rejected input
# ---------------------------------------------------------------------------


def test_invalid_rows_are_rejected(collator):
    r"""Malformed rows fail loudly rather than silently producing unaligned targets."""
    with pytest.raises(ValueError, match="empty list"):
        collator([])
    with pytest.raises(KeyError, match="No waveform"):
        collator([{"part_id": "MSP-Conversation_0002_1"}])
    with pytest.raises(ValueError, match="16000 Hz"):
        collator([{"audio": {"array": noise(3200), "sampling_rate": 8000}}])
    with pytest.raises(ValueError, match="num_frames, 3"):
        collator([{"input_values": noise(3200), "spk_labels": np.zeros((5, 2), np.float32)}])
    with pytest.raises(ValueError, match="three channels"):
        collator([{"input_values": noise(3200), "vad_labels": np.zeros((5, 2), np.float32)}])
