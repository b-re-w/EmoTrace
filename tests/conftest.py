r"""Shared fixtures for the test suite.

No test reads the corpora and no test downloads a checkpoint. The backbone every test uses is a
randomly initialized miniature of WavLM-large that keeps the two geometries the suite is about:

- the convolutional feature extractor of the released checkpoints, one frame per 320 samples with a
  400 sample receptive field, so frame counts are the real ones;
- the real positional convolution kernel (`num_conv_pos_embeddings=128`), so the 1.28 s of future
  context a bidirectional WavLM mixes into every frame is reachable and a leak is observable.

Everything else is shrunk: 32 hidden units, two layers, four heads.
"""

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # The package is used from the checkout rather than installed into the environment.
    sys.path.insert(0, str(REPO_ROOT))


TINY_BACKBONE_KWARGS: dict[str, object] = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "conv_dim": (16,) * 7,
    "num_conv_pos_embeddings": 128,
    "num_conv_pos_embedding_groups": 16,
    "num_buckets": 32,
    "max_bucket_distance": 128,
    "feat_extract_norm": "layer",
    "do_stable_layer_norm": True,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
    "activation_dropout": 0.0,
    "feat_proj_dropout": 0.0,
    "final_dropout": 0.0,
    "layerdrop": 0.0,
}
r"""Configuration overrides shared by the causal and the bidirectional miniature backbones."""

CAUSAL_ONLY_CONFIG_KEYS = frozenset(
    {"sliding_window", "model_type", "architectures", "transformers_version"}
)
r"""Fields of a `CausalWavLMConfig` that must not be carried over to a `WavLMConfig`."""


@pytest.fixture(autouse=True)
def deterministic_torch():
    r"""Seed the global generator and keep every test single threaded.

    Bit-exact comparisons are the point of the causality tests, and the number of threads decides
    how a reduction is split, hence the order in which floating point terms are summed.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    yield
    torch.set_num_threads(threads)


@pytest.fixture
def make_causal_config():
    r"""Returns a factory building a `CausalWavLMConfig` from the miniature settings.

    Returns:
        `Callable[..., CausalWavLMConfig]`: keyword arguments override the miniature settings.
    """
    from ser_project.models import CausalWavLMConfig

    def _make(**overrides) -> "CausalWavLMConfig":
        settings = dict(TINY_BACKBONE_KWARGS)
        settings.update(overrides)
        return CausalWavLMConfig(**settings)

    return _make


@pytest.fixture
def make_causal_model(make_causal_config):
    r"""Returns a factory building a randomly initialized `CausalWavLMModel` in eval mode.

    Returns:
        `Callable[..., CausalWavLMModel]`: accepts an explicit `config`, a `seed` for the
        initialization, and any configuration override otherwise.
    """
    from ser_project.models import CausalWavLMModel

    def _make(config=None, seed: int = 0, **overrides) -> "CausalWavLMModel":
        if config is None:
            config = make_causal_config(**overrides)
        torch.manual_seed(seed)
        return CausalWavLMModel(config).eval()

    return _make


@pytest.fixture
def make_bidirectional_twin():
    r"""Returns a factory building the stock `WavLMModel` that a causal model was derived from.

    The returned model has the same configuration and the very same weights as the causal model
    passed in, so any behavioural difference between the two comes from the causal machinery alone.

    Returns:
        `Callable[[CausalWavLMModel], WavLMModel]`
    """
    from ser_project.models import WavLMConfig, WavLMModel

    def _make(causal_model) -> "WavLMModel":
        settings = {
            key: value
            for key, value in causal_model.config.to_dict().items()
            if key not in CAUSAL_ONLY_CONFIG_KEYS
        }
        model = WavLMModel(WavLMConfig(**settings))
        model.load_state_dict(causal_model.state_dict(), strict=True)
        return model.eval()

    return _make


@pytest.fixture
def generator() -> torch.Generator:
    r"""Returns a seeded generator, so a failing test can be replayed exactly."""
    return torch.Generator().manual_seed(1234)
