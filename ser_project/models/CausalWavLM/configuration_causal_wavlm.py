"""CausalWavLM model configuration"""

from huggingface_hub.dataclasses import strict
from transformers.models.wavlm.configuration_wavlm import WavLMConfig


@strict
class CausalWavLMConfig(WavLMConfig):
    r"""
    Configuration class for [`CausalWavLMModel`].

    It inherits every field of [`WavLMConfig`] unchanged, so a released WavLM configuration converts field for field
    (`CausalWavLMConfig(**WavLMConfig.from_pretrained(...).to_dict())`) and the resulting model is weight compatible
    with the corresponding WavLM checkpoint. The only added field controls the width of the causal attention window.

    Args:
        sliding_window (`int`, *optional*, defaults to 800):
            Number of frames each query may attend to, itself included. The encoder runs at one frame per 20 ms, so
            the default corresponds to 16 seconds of left context. Setting it larger than the longest input turns
            the attention into plain causal attention.

    Example:

    ```python
    >>> from ser_project.models import CausalWavLMConfig, CausalWavLMModel

    >>> configuration = CausalWavLMConfig(sliding_window=800)
    >>> model = CausalWavLMModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "causal_wavlm"

    sliding_window: int = 800

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.sliding_window < 1:
            raise ValueError(f"`sliding_window` must be a positive number of frames, got {self.sliding_window}.")


__all__ = ["CausalWavLMConfig"]
