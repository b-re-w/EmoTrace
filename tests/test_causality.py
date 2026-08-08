r"""Causality of the CausalWavLM backbone.

The representation of frame `t` must be a function of the waveform up to time `t` only. Every test
here perturbs a part of the input and asserts that the outputs which precede it are *bit* identical,
because a leak of any size invalidates every online result the backbone is used for.

The only tolerated dependency on the future is the one the convolutional feature extractor of WavLM
carries: it has no padding, so frame `t` covers samples `[320 t, 320 t + 400)` and reaches 5 ms past
its own 20 ms step. The tests therefore expect the first frame that may change when the waveform is
perturbed from sample `p` onwards to be exactly `num_frames_for_samples(p)`, and they assert that
this frame does change, so that a test can never pass by comparing two identical forward passes.

Several tests come in pairs: one asserts the property on the causal model, its companion asserts
that the unmodified bidirectional WavLM violates the very same property. Without the companion, a
causality test that silently stopped exercising the model would still pass.
"""

import pytest
import torch
from transformers.cache_utils import DynamicCache

from ser_project.datasets.base import num_frames_for_samples
from ser_project.models import (
    CausalWavLMModel,
    CausalWavLMPositionalConvCache,
    CausalWavLMPositionalConvEmbedding,
    WavLMPositionalConvEmbedding,
)
from ser_project.models.CausalWavLM import modeling_causal_wavlm


ATTENTION_IMPLEMENTATIONS = ("sdpa", "eager")


class SymmetricPositionalConvEmbedding(WavLMPositionalConvEmbedding):
    r"""The unmodified WavLM positional convolution, plugged into a causal encoder.

    The causal encoder hands its positional convolution a streaming cache, which the stock module
    knows nothing about; accepting and ignoring it is all that is needed to reinstate the leak.
    """

    def forward(self, hidden_states: torch.Tensor, padding_cache=None) -> torch.Tensor:
        return super().forward(hidden_states)


def perturb_tail(waveform: torch.Tensor, first_sample: int, generator: torch.Generator) -> torch.Tensor:
    r"""Returns a copy of `waveform` whose samples from `first_sample` on are replaced by noise.

    Args:
        waveform (`torch.Tensor` of shape `(..., num_samples)`):
            Waveform to perturb.
        first_sample (`int`):
            Index of the first replaced sample.
        generator (`torch.Generator`):
            Source of the replacement noise.

    Returns:
        `torch.Tensor`: the perturbed copy.
    """
    perturbed = waveform.clone()
    tail_shape = perturbed[..., first_sample:].shape
    perturbed[..., first_sample:] = torch.randn(
        tail_shape, generator=generator, dtype=waveform.dtype
    )
    return perturbed


def first_changed_frame(reference: torch.Tensor, perturbed: torch.Tensor) -> int | None:
    r"""Index of the earliest frame at which two `(num_frames, hidden_size)` outputs differ.

    Returns:
        `int` or `None`: the frame index, `None` when the two tensors are bit identical.
    """
    changed = torch.nonzero((reference != perturbed).any(dim=-1)).flatten()
    return None if changed.numel() == 0 else int(changed[0])


def last_changed_frame(reference: torch.Tensor, perturbed: torch.Tensor) -> int | None:
    r"""Index of the latest frame at which two `(num_frames, hidden_size)` outputs differ."""
    changed = torch.nonzero((reference != perturbed).any(dim=-1)).flatten()
    return None if changed.numel() == 0 else int(changed[-1])


# ---------------------------------------------------------------------------
# Frame geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_samples", [3200, 16000, 16137, 32000])
def test_output_length_matches_the_bidirectional_model(
    make_causal_model, make_bidirectional_twin, num_samples
):
    r"""Making the backbone causal must not change how many frames it emits."""
    causal = make_causal_model()
    bidirectional = make_bidirectional_twin(causal)
    waveform = torch.randn(1, num_samples)

    with torch.no_grad():
        causal_frames = causal(waveform).last_hidden_state.shape[1]
        bidirectional_frames = bidirectional(waveform).last_hidden_state.shape[1]

    assert causal_frames == bidirectional_frames
    assert causal_frames == num_frames_for_samples(num_samples)


# ---------------------------------------------------------------------------
# The waveform-level guarantee
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attn_implementation", ATTENTION_IMPLEMENTATIONS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("perturbation", [400, 5011, 16000, 31680])
def test_future_samples_do_not_change_earlier_frames(
    make_causal_config, make_causal_model, generator, attn_implementation, dtype, perturbation
):
    r"""Every frame that ends before the perturbation is bit identical, and the next one changes.

    The perturbation points are deliberately not all aligned on a frame boundary: the guarantee is
    about samples, not about frames, and an implementation that rounded the boundary the wrong way
    would pass an aligned test and fail this one.
    """
    config = make_causal_config(sliding_window=64)
    config._attn_implementation = attn_implementation
    model = make_causal_model(config=config).to(dtype)

    num_samples = 32000
    waveform = torch.randn(1, num_samples, generator=generator, dtype=dtype)
    perturbed = perturb_tail(waveform, perturbation, generator)

    with torch.no_grad():
        reference = model(waveform).last_hidden_state[0]
        changed = model(perturbed).last_hidden_state[0]

    unaffected = num_frames_for_samples(perturbation)
    assert torch.equal(reference[:unaffected], changed[:unaffected])
    assert first_changed_frame(reference, changed) == unaffected


def test_bidirectional_wavlm_fails_the_future_check(
    make_causal_model, make_bidirectional_twin, generator
):
    r"""The same check on the unmodified WavLM fails, which is what gives the check its power."""
    causal = make_causal_model(sliding_window=64)
    bidirectional = make_bidirectional_twin(causal)

    num_samples, perturbation = 32000, 16000
    waveform = torch.randn(1, num_samples, generator=generator)
    perturbed = perturb_tail(waveform, perturbation, generator)

    with torch.no_grad():
        reference = bidirectional(waveform).last_hidden_state[0]
        changed = bidirectional(perturbed).last_hidden_state[0]

    unaffected = num_frames_for_samples(perturbation)
    assert not torch.equal(reference[:unaffected], changed[:unaffected])
    assert first_changed_frame(reference, changed) == 0


def test_the_check_catches_a_symmetric_positional_convolution(make_causal_model, generator):
    r"""Putting the leaking convolution back into an otherwise causal model breaks the check.

    The stock WavLM used above differs from the causal model in two places at once. Here a single
    component is reverted, so the check is shown to detect a leak that lives exactly where the fix
    lives, and cannot be passing for some unrelated reason.
    """
    model = make_causal_model(sliding_window=64)
    symmetric = SymmetricPositionalConvEmbedding(model.config).eval()
    symmetric.load_state_dict(model.encoder.pos_conv_embed.state_dict())
    model.encoder.pos_conv_embed = symmetric

    num_samples, perturbation = 32000, 16000
    waveform = torch.randn(1, num_samples, generator=generator)
    perturbed = perturb_tail(waveform, perturbation, generator)

    with torch.no_grad():
        reference = model(waveform).last_hidden_state[0]
        changed = model(perturbed).last_hidden_state[0]

    unaffected = num_frames_for_samples(perturbation)
    assert not torch.equal(reference[:unaffected], changed[:unaffected])


def test_the_check_catches_unmasked_attention(
    monkeypatch, make_causal_config, make_causal_model, generator
):
    r"""Dropping the attention mask makes the same model leak, and the check says so.

    This is the second half of the mutation argument: with the positional convolution left causal
    and only the attention mask removed, the waveform-level check must still fail.
    """
    config = make_causal_config(sliding_window=64)
    config._attn_implementation = "eager"
    model = make_causal_model(config=config)
    monkeypatch.setattr(
        modeling_causal_wavlm, "create_sliding_window_causal_mask", lambda **kwargs: None
    )

    num_samples, perturbation = 32000, 16000
    waveform = torch.randn(1, num_samples, generator=generator)
    perturbed = perturb_tail(waveform, perturbation, generator)

    with torch.no_grad():
        reference = model(waveform).last_hidden_state[0]
        changed = model(perturbed).last_hidden_state[0]

    unaffected = num_frames_for_samples(perturbation)
    assert not torch.equal(reference[:unaffected], changed[:unaffected])


def test_causality_holds_in_a_padded_batch(make_causal_model, generator):
    r"""A padding mask must not open a path from a later sample to an earlier frame."""
    model = make_causal_model(sliding_window=64)

    lengths = (32000, 24000)
    waveform = torch.zeros(2, max(lengths))
    attention_mask = torch.zeros(2, max(lengths), dtype=torch.long)
    for index, length in enumerate(lengths):
        waveform[index, :length] = torch.randn(length, generator=generator)
        attention_mask[index, :length] = 1

    perturbation = 12800
    perturbed = perturb_tail(waveform, perturbation, generator)
    perturbed *= attention_mask

    with torch.no_grad():
        reference = model(waveform, attention_mask=attention_mask).last_hidden_state
        changed = model(perturbed, attention_mask=attention_mask).last_hidden_state

    unaffected = num_frames_for_samples(perturbation)
    assert torch.equal(reference[:, :unaffected], changed[:, :unaffected])
    for row in range(len(lengths)):
        assert first_changed_frame(reference[row], changed[row]) == unaffected


def test_right_padding_does_not_change_the_valid_frames(make_causal_model, generator):
    r"""A short sample gets the same frames whether or not it is padded to a longer neighbour."""
    model = make_causal_model(sliding_window=64)

    short = torch.randn(16000, generator=generator)
    batch = torch.zeros(2, 32000)
    batch[0] = torch.randn(32000, generator=generator)
    batch[1, : short.shape[0]] = short
    attention_mask = torch.zeros(2, 32000, dtype=torch.long)
    attention_mask[0] = 1
    attention_mask[1, : short.shape[0]] = 1

    with torch.no_grad():
        batched = model(batch, attention_mask=attention_mask).last_hidden_state
        alone = model(
            short.unsqueeze(0), attention_mask=torch.ones(1, short.shape[0], dtype=torch.long)
        ).last_hidden_state

    frames = num_frames_for_samples(short.shape[0])
    torch.testing.assert_close(batched[1, :frames], alone[0], rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# The positional convolution, the first of the two leaks
# ---------------------------------------------------------------------------


def test_causal_positional_convolution_only_looks_backwards(make_causal_config, generator):
    r"""Perturbing frame `j` changes the outputs `j ... j + kernel_size - 1` and nothing earlier."""
    config = make_causal_config()
    torch.manual_seed(0)
    convolution = CausalWavLMPositionalConvEmbedding(config).eval()

    num_frames, perturbed_frame = 300, 150
    hidden_states = torch.randn(1, num_frames, config.hidden_size, generator=generator)
    perturbed = hidden_states.clone()
    perturbed[:, perturbed_frame] += 5.0

    with torch.no_grad():
        reference = convolution(hidden_states)[0]
        changed = convolution(perturbed)[0]

    assert first_changed_frame(reference, changed) == perturbed_frame
    assert last_changed_frame(reference, changed) == perturbed_frame + config.num_conv_pos_embeddings - 1


def test_stock_positional_convolution_looks_into_the_future(make_causal_config, generator):
    r"""The unmodified positional convolution moves a frame into the outputs that precede it."""
    config = make_causal_config()
    torch.manual_seed(0)
    convolution = WavLMPositionalConvEmbedding(config).eval()

    num_frames, perturbed_frame = 300, 150
    hidden_states = torch.randn(1, num_frames, config.hidden_size, generator=generator)
    perturbed = hidden_states.clone()
    perturbed[:, perturbed_frame] += 5.0

    with torch.no_grad():
        reference = convolution(hidden_states)[0]
        changed = convolution(perturbed)[0]

    # The symmetric kernel is centred on the current frame, so `num_conv_pos_embeddings // 2 - 1`
    # earlier outputs already carry it, which is 1.26 s at the 20 ms frame rate.
    leaked_frames = perturbed_frame - first_changed_frame(reference, changed)
    assert leaked_frames == config.num_conv_pos_embeddings // 2 - 1


@pytest.mark.parametrize("num_frames", [1, 17, 300])
def test_positional_convolution_preserves_the_sequence_length(make_causal_config, num_frames):
    r"""Left padding replaces the symmetric padding without changing the output length."""
    config = make_causal_config()
    torch.manual_seed(0)
    convolution = CausalWavLMPositionalConvEmbedding(config).eval()

    hidden_states = torch.randn(2, num_frames, config.hidden_size)
    with torch.no_grad():
        output = convolution(hidden_states)

    assert output.shape == hidden_states.shape


def test_positional_convolution_keeps_the_wavlm_parameter_paths(make_causal_config):
    r"""The convolution stays the same module, so released weights keep loading into it."""
    config = make_causal_config()
    torch.manual_seed(0)
    causal = CausalWavLMPositionalConvEmbedding(config)
    torch.manual_seed(0)
    stock = WavLMPositionalConvEmbedding(config)

    assert set(causal.state_dict()) == set(stock.state_dict())
    assert causal.conv.padding == (0,)
    assert causal.left_padding == config.num_conv_pos_embeddings - 1
    # `WavLMSamePadLayer` exists only to trim the frame the symmetric padding adds.
    assert not hasattr(causal, "padding")
    assert torch.nn.utils.parametrize.is_parametrized(causal.conv, "weight")


def test_state_dict_is_interchangeable_with_stock_wavlm(make_causal_model, make_bidirectional_twin):
    r"""A WavLM checkpoint loads into the causal model without a single missing or extra key."""
    causal = make_causal_model()
    bidirectional = make_bidirectional_twin(causal)

    assert set(causal.state_dict()) == set(bidirectional.state_dict())
    result = causal.load_state_dict(bidirectional.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys


def test_group_normalized_feature_encoder_is_rejected(make_causal_config):
    r"""The group-norm feature encoder normalizes over the whole time axis and cannot be causal."""
    config = make_causal_config(feat_extract_norm="group", do_stable_layer_norm=False)
    with pytest.raises(ValueError, match="feat_extract_norm"):
        CausalWavLMModel(config)


# ---------------------------------------------------------------------------
# The sliding window, the second of the two leaks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", [0, 3, 7, 8, 100, 319])
def test_attention_probabilities_stay_inside_the_sliding_window(
    make_causal_config, make_causal_model, generator, query
):
    r"""A query attends the `sliding_window` most recent frames, itself included, and nothing else."""
    window = 8
    config = make_causal_config(sliding_window=window, num_hidden_layers=1)
    config._attn_implementation = "eager"
    model = make_causal_model(config=config)

    num_frames = 320
    hidden_states = torch.randn(1, num_frames, config.hidden_size, generator=generator)
    with torch.no_grad():
        attentions = model.encoder(hidden_states, output_attentions=True).attentions

    probabilities = attentions[0][0, :, query]
    support = torch.nonzero(probabilities.abs().sum(dim=0)).flatten()
    assert int(support[0]) == max(query - window + 1, 0)
    assert int(support[-1]) == query
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(config.num_attention_heads), rtol=0, atol=1e-6
    )


def test_frames_before_the_sliding_window_cannot_reach_the_output(
    make_causal_config, make_causal_model, generator
):
    r"""A perturbed frame stops propagating once the window and the positional kernel are exhausted.

    One encoder layer widens the dependency by `sliding_window - 1` frames and the causal positional
    convolution by `num_conv_pos_embeddings - 1` frames, which bounds the reach exactly.
    """
    window, num_layers = 8, 1
    config = make_causal_config(sliding_window=window, num_hidden_layers=num_layers)
    model = make_causal_model(config=config)

    num_frames, perturbed_frame = 320, 10
    hidden_states = torch.randn(1, num_frames, config.hidden_size, generator=generator)
    perturbed = hidden_states.clone()
    perturbed[:, perturbed_frame] += 3.0

    with torch.no_grad():
        reference = model.encoder(hidden_states).last_hidden_state[0]
        changed = model.encoder(perturbed).last_hidden_state[0]

    reach = perturbed_frame + num_layers * (window - 1) + config.num_conv_pos_embeddings - 1
    assert reach < num_frames - 1
    assert torch.equal(reference[reach + 1 :], changed[reach + 1 :])
    assert last_changed_frame(reference, changed) == reach


def test_unbounded_attention_reaches_beyond_the_window(
    make_causal_config, make_causal_model, generator
):
    r"""The same model with a window wider than the input propagates to the very last frame."""
    num_frames, perturbed_frame = 320, 10
    config = make_causal_config(sliding_window=num_frames, num_hidden_layers=1)
    model = make_causal_model(config=config)

    hidden_states = torch.randn(1, num_frames, config.hidden_size, generator=generator)
    perturbed = hidden_states.clone()
    perturbed[:, perturbed_frame] += 3.0

    with torch.no_grad():
        reference = model.encoder(hidden_states).last_hidden_state[0]
        changed = model.encoder(perturbed).last_hidden_state[0]

    assert last_changed_frame(reference, changed) == num_frames - 1


def test_chunked_streaming_reproduces_the_full_pass(make_causal_config, make_causal_model, generator):
    r"""Feeding the encoder in chunks with the two caches gives back the single-pass result.

    A model whose frames depended on their future could not be reproduced this way, so this is the
    same guarantee seen from the deployment side.
    """
    config = make_causal_config(sliding_window=32)
    model = make_causal_model(config=config)

    num_frames, chunk = 200, 50
    hidden_states = torch.randn(1, num_frames, config.hidden_size, generator=generator)

    with torch.no_grad():
        expected = model.encoder(hidden_states).last_hidden_state
        past_key_values = DynamicCache(config=config)
        padding_cache = CausalWavLMPositionalConvCache()
        chunks = [
            model.encoder(
                hidden_states[:, start : start + chunk],
                past_key_values=past_key_values,
                padding_cache=padding_cache,
            ).last_hidden_state
            for start in range(0, num_frames, chunk)
        ]

    streamed = torch.cat(chunks, dim=1)
    assert streamed.shape == expected.shape
    torch.testing.assert_close(streamed, expected, rtol=1e-4, atol=1e-5)
