# coding=utf-8
"""Feature extraction, CTC vocabulary and processor classes for EmoTrace."""

import json
import os
import tempfile

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.tokenization_wav2vec2 import Wav2Vec2CTCTokenizer
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import AddedToken, AudioInput, PreTokenizedInput, TextInput
from transformers.utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)


EMOTRACE_CHARACTERS: tuple[str, ...] = tuple("abcdefghijklmnopqrstuvwxyz'*0123456789")
r"""
Single-character symbols of the CTC vocabulary.

Transcripts reach the head lower-cased and free of punctuation, which leaves the letters, the apostrophe of
contractions, the `*` marking an interrupted word and the digits. Every one of them is emitted by
`ser_project.datasets.pseudo_label.normalize_transcript` under its default settings; a symbol the normalizer can
produce but the vocabulary does not hold would silently become `<unk>`.
"""

EMOTRACE_VOCAL_EVENT_TOKENS: tuple[str, ...] = (
    "[uh]",
    "[um]",
    "[laughter]",
    "[breath]",
    "[lipsmack]",
    "[throatclearing]",
    "[sigh]",
    "[sniff]",
    "[cough]",
)
r"""
Atomic vocal-event tokens of the CTC vocabulary.

Each entry is a single unit of the target sequence rather than a run of characters, so that the ASR head is
supervised on paralinguistic events with the same granularity as on words. `Wav2Vec2CTCTokenizer` registers every
multi-character vocabulary entry as an added token, which keeps these tokens intact through tokenisation.

The set is the one the verbatim transcription convention defines, and it matches
`ser_project.datasets.pseudo_label.VOCAL_EVENT_TAGS` tag for tag: a tag kept by the normalizer but absent here
would be split into characters, and a token listed here but never produced would be an output unit that no target
ever selects.
"""


def build_emotrace_ctc_vocab(
    characters: tuple[str, ...] | list[str] = EMOTRACE_CHARACTERS,
    vocal_event_tokens: tuple[str, ...] | list[str] = EMOTRACE_VOCAL_EVENT_TOKENS,
    pad_token: str = "<pad>",
    bos_token: str = "<s>",
    eos_token: str = "</s>",
    unk_token: str = "<unk>",
    word_delimiter_token: str = "|",
) -> dict[str, int]:
    r"""
    Build the CTC vocabulary of the ASR head: characters plus atomic vocal-event tokens.

    The padding token is placed at index 0 because it doubles as the CTC blank, matching
    `EmoTraceConfig.ctc_blank_token_id`.

    Args:
        characters (`tuple[str]`, *optional*):
            Single-character symbols of the vocabulary.
        vocal_event_tokens (`tuple[str]`, *optional*):
            Multi-character tokens standing for paralinguistic events.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            Padding token, used as the CTC blank.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            Beginning-of-sequence token, unused by CTC but kept for vocabulary compatibility.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            End-of-sequence token, unused by CTC but kept for vocabulary compatibility.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            Token substituted for symbols outside the vocabulary.
        word_delimiter_token (`str`, *optional*, defaults to `"|"`):
            Token substituted for whitespace.

    Returns:
        `dict[str, int]`: mapping from token to index.

    Raises:
        ValueError: if the same token is requested twice.
    """
    tokens = [pad_token, bos_token, eos_token, unk_token, word_delimiter_token]
    tokens += list(characters)
    tokens += list(vocal_event_tokens)

    if len(set(tokens)) != len(tokens):
        duplicates = sorted({token for token in tokens if tokens.count(token) > 1})
        raise ValueError(f"The CTC vocabulary contains duplicate tokens: {duplicates}.")

    return {token: index for index, token in enumerate(tokens)}


def build_emotrace_ctc_tokenizer(
    vocab: dict[str, int] | None = None,
    pad_token: str = "<pad>",
    bos_token: str = "<s>",
    eos_token: str = "</s>",
    unk_token: str = "<unk>",
    word_delimiter_token: str = "|",
    **kwargs,
) -> Wav2Vec2CTCTokenizer:
    r"""
    Instantiate the CTC tokenizer of EmoTrace from an in-memory vocabulary.

    Args:
        vocab (`dict[str, int]`, *optional*):
            Vocabulary to use. Defaults to the output of `build_emotrace_ctc_vocab`.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            Padding token, used as the CTC blank.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            Beginning-of-sequence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            End-of-sequence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            Token substituted for symbols outside the vocabulary.
        word_delimiter_token (`str`, *optional*, defaults to `"|"`):
            Token substituted for whitespace.
        kwargs:
            Forwarded to [`Wav2Vec2CTCTokenizer`].

    Returns:
        [`Wav2Vec2CTCTokenizer`]: a tokenizer over the EmoTrace vocabulary.
    """
    vocab = build_emotrace_ctc_vocab() if vocab is None else vocab

    # `Wav2Vec2CTCTokenizer` reads its vocabulary from disk; the file is only needed during construction.
    with tempfile.TemporaryDirectory() as directory:
        vocab_file = os.path.join(directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as handle:
            json.dump(vocab, handle, ensure_ascii=False)

        tokenizer = Wav2Vec2CTCTokenizer(
            vocab_file,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            word_delimiter_token=word_delimiter_token,
            do_lower_case=False,
            **kwargs,
        )

    # `Wav2Vec2CTCTokenizer` registers every multi-character vocabulary entry with `lstrip`/`rstrip` enabled, which
    # swallows the spaces surrounding a vocal-event token and drops the word boundary from the target sequence.
    # Re-adding those entries without stripping keeps `[laughter]` atomic while preserving the delimiter.
    special_tokens = set(tokenizer.all_special_tokens)
    atomic_tokens = [token for token in vocab if len(token) > 1 and token not in special_tokens]
    if atomic_tokens:
        tokenizer.add_tokens(
            [AddedToken(token, lstrip=False, rstrip=False, normalized=False) for token in atomic_tokens]
        )

    return tokenizer


class EmoTraceFeatureExtractor(Wav2Vec2FeatureExtractor):
    r"""
    Feature extractor for EmoTrace: 16 kHz waveforms normalised with fixed, dataset-level constants.

    [`Wav2Vec2FeatureExtractor`] normalises every utterance to zero mean and unit variance, which makes the input
    scale depend on the surrounding utterance. Online prediction has no utterance boundary to compute statistics
    over, and the loudness of a conversation carries information about arousal and dominance, so EmoTrace applies
    the affine transform `(waveform - mean_value) / std_value` with constants measured once on the training
    partition. Padded positions are restored to `padding_value` afterwards, which requires an attention mask.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            Feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            Sampling rate the audio is expected to be digitalised at, expressed in hertz.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value used to fill padded positions.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether `__call__` should return an attention mask. The backbone derives frame padding from it, so it
            defaults to `True` unlike in [`Wav2Vec2FeatureExtractor`].
        mean_value (`float`, *optional*, defaults to 0.0):
            Constant subtracted from the waveform.
        std_value (`float`, *optional*, defaults to 0.1):
            Constant the waveform is divided by. The defaults describe speech at roughly -20 dBFS and must be
            replaced by the statistics of the training partition.
    """

    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        return_attention_mask=True,
        mean_value=0.0,
        std_value=0.1,
        **kwargs,
    ):
        if std_value <= 0:
            raise ValueError(f"`std_value` must be strictly positive, but is {std_value}.")

        kwargs.pop("do_normalize", None)
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            do_normalize=False,
            **kwargs,
        )
        self.mean_value = mean_value
        self.std_value = std_value

    def fixed_norm(self, input_values: list[np.ndarray], attention_mask: list[np.ndarray] | None) -> list[np.ndarray]:
        r"""
        Apply the fixed affine normalisation and restore the padded positions.

        Args:
            input_values (`list[np.ndarray]`):
                Waveforms, possibly padded.
            attention_mask (`list[np.ndarray]`, *optional*):
                Mask of the valid samples. When omitted, padded positions cannot be restored.

        Returns:
            `list[np.ndarray]`: normalised waveforms.
        """
        normed_input_values = [
            ((np.asarray(array, dtype=np.float32) - self.mean_value) / self.std_value).astype(np.float32)
            for array in input_values
        ]

        if attention_mask is not None:
            lengths = np.asarray(attention_mask, dtype=np.int32).sum(-1)
            for array, length in zip(normed_input_values, lengths):
                if length < array.shape[0]:
                    array[length:] = self.padding_value

        return normed_input_values

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        padding: bool | str | PaddingStrategy = False,
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool | None = None,
        return_tensors: str | TensorType | None = None,
        sampling_rate: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        r"""
        Featurise one or several waveforms.

        Args:
            raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                Mono-channel waveform(s) to pad and normalise.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Padding strategy.
            max_length (`int`, *optional*):
                Maximum length of the returned sequences.
            truncation (`bool`, *optional*, defaults to `False`):
                Whether to truncate sequences longer than `max_length`.
            pad_to_multiple_of (`int`, *optional*):
                Pad the sequences to a multiple of this value.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. Defaults to the value set on the feature extractor.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                Framework of the returned tensors.
            sampling_rate (`int`, *optional*):
                Sampling rate of `raw_speech`, checked against the one the extractor is configured for.

        Returns:
            [`BatchFeature`]: with `input_values` and, unless disabled, `attention_mask`.
        """
        # Normalisation happens on numpy arrays before any framework conversion, so `return_tensors` is applied
        # once at the end instead of being forwarded to the parent call.
        padded_inputs = super().__call__(
            raw_speech,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_tensors=None,
            sampling_rate=sampling_rate,
            **kwargs,
        )

        padded_inputs["input_values"] = self.fixed_norm(
            padded_inputs["input_values"], attention_mask=padded_inputs.get("attention_mask")
        )

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs


class EmoTraceProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 16000,
            "return_attention_mask": True,
        },
    }


class EmoTraceProcessor(ProcessorMixin):
    r"""
    Processor for EmoTrace, wrapping an [`EmoTraceFeatureExtractor`] and a [`Wav2Vec2CTCTokenizer`].

    Transcripts are returned under `ctc_labels` rather than `labels`, matching the batch dictionary consumed by
    [`EmoTraceForVADPrediction`], whose ASR head is one of three heads. The frame-rate targets of the other two
    heads (`vad_labels`, `spk_labels`) are produced by the dataset collator, not by this processor.

    Args:
        feature_extractor ([`EmoTraceFeatureExtractor`]):
            Waveform featuriser.
        tokenizer ([`Wav2Vec2CTCTokenizer`]):
            Tokenizer over the CTC vocabulary of characters and atomic vocal-event tokens.
    """

    valid_processor_kwargs = EmoTraceProcessorKwargs

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    @classmethod
    def from_defaults(cls, **kwargs) -> "EmoTraceProcessor":
        r"""
        Build a processor from the default EmoTrace vocabulary and feature-extraction settings.

        Args:
            kwargs:
                Forwarded to [`EmoTraceFeatureExtractor`], e.g. `mean_value` and `std_value`.

        Returns:
            [`EmoTraceProcessor`]: the assembled processor.
        """
        return cls(feature_extractor=EmoTraceFeatureExtractor(**kwargs), tokenizer=build_emotrace_ctc_tokenizer())

    def __call__(
        self,
        audio: AudioInput | None = None,
        text: str | list[str] | TextInput | PreTokenizedInput | None = None,
        **kwargs: Unpack[EmoTraceProcessorKwargs],
    ):
        r"""
        Featurise audio, tokenise transcripts, or both.

        Args:
            audio (`AudioInput`, *optional*):
                Waveform(s) sampled at 16 kHz, forwarded to the feature extractor.
            text (`str` or `list[str]`, *optional*):
                Transcript(s), forwarded to the tokenizer.

        Returns:
            [`BatchFeature`]: the feature-extractor output, with the token ids added under `ctc_labels` when both
            modalities are given, or the tokenizer output alone when only `text` is given.
        """
        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        output_kwargs = self._merge_kwargs(
            EmoTraceProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if audio is not None:
            inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])

        if text is None:
            return inputs
        if audio is None:
            return encodings

        inputs["ctc_labels"] = encodings["input_ids"]
        return inputs

    def batch_decode(self, *args, **kwargs):
        """Forward all arguments to [`Wav2Vec2CTCTokenizer.batch_decode`]."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Forward all arguments to [`Wav2Vec2CTCTokenizer.decode`]."""
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return self.feature_extractor.model_input_names + ["ctc_labels"]


__all__ = [
    "EMOTRACE_CHARACTERS",
    "EMOTRACE_VOCAL_EVENT_TOKENS",
    "EmoTraceFeatureExtractor",
    "EmoTraceProcessor",
    "EmoTraceProcessorKwargs",
    "build_emotrace_ctc_tokenizer",
    "build_emotrace_ctc_vocab",
]
