r"""Contracts that span two modules, and that no module can check on its own.

Each test here pins a place where one agent's output is another's input: the batch dictionary the
collator emits against the arguments the model declares, the CTC vocabulary of the ASR head against
the transcript normalizer that produces its targets, and the streaming caches of the backbone
against the model that has to forward them. A unit test of either side passes while these drift.

Everything runs on a miniature randomly initialized model and on hand-built rows, so no corpus file
and no checkpoint is read.
"""

import inspect

import numpy as np
import pytest
import torch

from ser_project.datasets.collator import DataCollatorForEmoTrace
from ser_project.datasets.pseudo_label import VOCAL_EVENT_TAGS, normalize_transcript
from ser_project.models.CausalWavLM import CausalWavLMConfig, CausalWavLMPositionalConvCache
from ser_project.models.EmoTrace import (
    EMOTRACE_VOCAL_EVENT_TOKENS,
    EmoTraceConfig,
    EmoTraceForVADPrediction,
    build_emotrace_ctc_tokenizer,
    build_emotrace_ctc_vocab,
)
from ser_project.models.WavLM import WavLMConfig


SAMPLING_RATE = 16000

TINY_BACKBONE_KWARGS: dict[str, object] = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "conv_dim": (16,) * 7,
    "num_conv_pos_embeddings": 16,
    "num_conv_pos_embedding_groups": 4,
    "num_buckets": 8,
    "max_bucket_distance": 32,
    "feat_extract_norm": "layer",
    "do_stable_layer_norm": True,
    "layerdrop": 0.0,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
    "activation_dropout": 0.0,
    "feat_proj_dropout": 0.0,
    "apply_spec_augment": False,
}
r"""A backbone small enough to instantiate in a test, with the real convolutional front end."""


@pytest.fixture
def emotrace_model():
    r"""Returns a factory building a miniature [`EmoTraceForVADPrediction`] in eval mode."""

    def _make(backbone_model_type: str = "causal_wavlm", **overrides):
        if backbone_model_type == "wavlm":
            backbone_config = WavLMConfig(**TINY_BACKBONE_KWARGS)
        else:
            backbone_config = CausalWavLMConfig(sliding_window=64, **TINY_BACKBONE_KWARGS)
        torch.manual_seed(0)
        config = EmoTraceConfig(backbone_config=backbone_config, **overrides)
        return EmoTraceForVADPrediction(config).eval()

    return _make


def rows(num_rows: int = 2, num_samples: int = 8000) -> list[dict]:
    r"""Returns hand-built dataset rows carrying every kind of supervision."""
    generator = np.random.default_rng(0)
    num_frames = (num_samples - 400) // 320 + 1
    return [
        {
            "part_id": f"MSP-Conversation_000{index}_1",
            "input_values": generator.standard_normal(num_samples).astype(np.float32) * 0.1,
            "vad_labels": generator.standard_normal((num_frames, 3)).astype(np.float32) * 0.3,
            "speaker_segments": [("A", 0.0, 2.0), ("B", 1.5, 4.0)],
            "ctc_labels": list(generator.integers(1, 40, size=8)),
        }
        for index in range(num_rows)
    ]


# ---------------------------------------------------------------------------
# The batch dictionary against the model signature
# ---------------------------------------------------------------------------


def test_every_batch_key_is_a_model_argument():
    r"""The collator emits no key the model would have to swallow in `**kwargs`.

    A key the model does not declare is silently ignored, so a renamed label would train a model on
    the tasks that are left instead of failing.
    """
    batch = DataCollatorForEmoTrace()(rows())
    parameters = inspect.signature(EmoTraceForVADPrediction.forward).parameters
    undeclared = [key for key in batch if key not in parameters]
    assert not undeclared, f"batch keys the model does not declare: {undeclared}"


def test_collated_batch_trains_every_head(emotrace_model):
    r"""A batch of the collator drives all three losses and produces gradients."""
    model = emotrace_model().train()
    outputs = model(**DataCollatorForEmoTrace()(rows()))

    assert set(outputs.loss_dict) == {"loss_vad", "loss_ctc", "loss_spk"}
    assert outputs.vad_predictions.shape[-1] == model.config.num_vad_targets
    assert outputs.ctc_logits.shape[-1] == model.config.vocab_size
    assert outputs.spk_logits.shape[-1] == model.config.num_speaker_labels

    outputs.loss.backward()
    assert torch.isfinite(model.vad_head.weight.grad).all()


def test_label_time_dimension_matches_the_frame_count(emotrace_model):
    r"""The frame arithmetic of the collator is the one the backbone actually applies."""
    batch = DataCollatorForEmoTrace()(rows(num_rows=2, num_samples=8000))
    with torch.no_grad():
        predictions = emotrace_model()(batch["input_values"], attention_mask=batch["attention_mask"])
    assert batch["vad_labels"].shape[1] == predictions.vad_predictions.shape[1]
    assert batch["spk_labels"].shape[1] == predictions.vad_predictions.shape[1]


# ---------------------------------------------------------------------------
# The CTC vocabulary against the transcript normalizer
# ---------------------------------------------------------------------------


def test_the_channel_orders_agree_across_modules():
    r"""Every module that names a channel of a target tensor names them in the same order.

    The tensors carry no names, so a module that ordered its own constant differently would swap two
    attributes silently: the loss would still fall and the metric would still be reported.
    """
    from ser_project.datasets.base import ATTRIBUTES
    from ser_project.datasets.collator import SPEAKER_TARGET_NAMES, VAD_ATTRIBUTE_NAMES
    from ser_project.datasets.msp_conversation import SPEAKER_TARGETS
    from ser_project.utils.evaluate import DEFAULT_ATTRIBUTE_NAMES
    from ser_project.utils.train import VAD_LABEL_NAMES

    assert ATTRIBUTES == VAD_ATTRIBUTE_NAMES == DEFAULT_ATTRIBUTE_NAMES
    assert SPEAKER_TARGETS == SPEAKER_TARGET_NAMES
    assert tuple(VAD_LABEL_NAMES) == ("vad_labels", "vad_eval_mask")


def test_vocabulary_holds_exactly_the_verbatim_tag_set():
    r"""The atomic tokens of the head are the tags the normalizer keeps, neither more nor fewer."""
    assert set(EMOTRACE_VOCAL_EVENT_TOKENS) == set(VOCAL_EVENT_TAGS)


def test_default_vocab_size_matches_the_built_vocabulary():
    r"""The ASR head is as wide as the tokenizer that produces its targets."""
    assert EmoTraceConfig().vocab_size == len(build_emotrace_ctc_vocab())


@pytest.mark.parametrize(
    "transcript",
    [
        "I was, uh, [laughter] going to say th- 25 dollars",
        "he [throatclearing] said [sniff] okay, don't you think?",
        "Well, [UH] I think so -- [laughs]",
    ],
)
def test_normalized_transcripts_tokenize_without_unknowns(transcript):
    r"""Nothing the normalizer emits falls outside the vocabulary.

    A symbol the vocabulary does not hold becomes `<unk>`, which the CTC loss then trains the head
    to emit; the transcript task would keep converging while the targets no longer say what the
    speaker said.
    """
    vocab = build_emotrace_ctc_vocab()
    tokenizer = build_emotrace_ctc_tokenizer(vocab)

    normalized = normalize_transcript(transcript)
    token_ids = tokenizer(normalized)["input_ids"]

    assert vocab["<unk>"] not in token_ids
    assert tokenizer.decode(token_ids, group_tokens=False) == normalized


# ---------------------------------------------------------------------------
# The backbone selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("backbone_model_type", "expected_class", "expected_causal"),
    [("causal_wavlm", "CausalWavLMModel", True), ("wavlm", "WavLMModel", False)],
)
def test_the_heads_run_on_either_encoder(
    emotrace_model, tmp_path, backbone_model_type, expected_class, expected_causal
):
    r"""The bidirectional arm of the comparison is the same object with another encoder.

    Both arms have to be ordinary checkpoints, so the encoder class has to survive a save and a
    reload: a model that came back as the wrong one would report the wrong arm's numbers.
    """
    model = emotrace_model(backbone_model_type)
    assert type(model.backbone).__name__ == expected_class
    assert model.config.is_causal is expected_causal

    waveform = torch.zeros(1, 8000)
    with torch.no_grad():
        reference = model(waveform).vad_predictions

    model.save_pretrained(tmp_path)
    reloaded = EmoTraceForVADPrediction.from_pretrained(tmp_path).eval()
    assert type(reloaded.backbone).__name__ == expected_class
    with torch.no_grad():
        assert torch.equal(reloaded(waveform).vad_predictions, reference)


def test_a_bidirectional_model_refuses_the_streaming_caches(emotrace_model):
    r"""Streaming state handed to a model that cannot use it is an error, not a silent full pass."""
    model = emotrace_model("wavlm")
    with pytest.raises(ValueError, match="streaming|causal_wavlm"):
        model(torch.zeros(1, 8000), padding_cache=CausalWavLMPositionalConvCache())


# ---------------------------------------------------------------------------
# The streaming caches through the heads
# ---------------------------------------------------------------------------


def test_streaming_through_the_model_reproduces_a_single_pass(emotrace_model):
    r"""The caches reach the encoder through the model's own `forward`, not around it.

    `StreamingSession` calls whichever module declares `padding_cache`. If the model dropped the
    argument, the session would fall back to driving the encoder directly and the heads would be
    applied by the session instead of by the model, which is the same computation only for as long
    as the two implementations agree.
    """
    from ser_project.utils.streaming import _resolve_encoder, check_streaming_equivalence

    model = emotrace_model()
    streamed_module, heads_applied_outside = _resolve_encoder(model)
    assert streamed_module is model and not heads_applied_outside

    report = check_streaming_equivalence(model, num_samples=12000, chunk_schedules=(1600, (128, 700, 2500, 61)))
    assert report.passed, report.summary()
