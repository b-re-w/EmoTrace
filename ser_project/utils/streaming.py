r"""Chunk-by-chunk streaming inference for the causal EmoTrace stack.

A causal model is only useful online if feeding it a recording in pieces reproduces, frame for frame, what a single
forward pass over the whole recording produces. Three pieces of state have to travel across the chunk boundary for
that to hold, and this module carries all three:

- **Waveform tail.** The convolutional feature encoder is stateless and uses no padding, so frame `t` is a function
  of the samples `[t * samples_per_frame, t * samples_per_frame + receptive_field)`. Consecutive chunks therefore
  have to overlap by `receptive_field - samples_per_frame` samples (80 samples, 5 ms, for WavLM). The session keeps
  that tail internally: callers hand over disjoint pieces of audio and never think about it.
- **Positional convolution left context.** [`CausalWavLMPositionalConvEmbedding`] pads its input with
  `num_conv_pos_embeddings - 1` frames on the left. Across chunks those frames come from a
  [`CausalWavLMPositionalConvCache`] instead of from zeros.
- **Attention keys and values.** A [`~cache_utils.Cache`] holds the keys and values of the frames already seen.

Because attention is restricted to a sliding window of `config.sliding_window` frames, the cache is built so that it
drops everything outside that window, which is what keeps the per-chunk cost independent of how long the stream has
been running; [`enforce_cache_window`] states and checks that invariant explicitly.

The public entry points are [`StreamingSession`] for running a stream, [`check_streaming_equivalence`] for the
numerical proof that streaming matches a single pass, and [`measure_streaming_latency`] for per-chunk latency
percentiles and the real-time factor. Both checks are runnable without a checkpoint or a corpus: with no model given
they build a small randomly initialised one with [`build_streaming_test_model`].
"""

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PreTrainedConfig
from transformers.utils import logging

from ..models.CausalWavLM import CausalWavLMPositionalConvCache


logger = logging.get_logger(__name__)


__all__ = [
    "CacheState",
    "FrameGeometry",
    "LatencyReport",
    "StreamingChunkOutput",
    "StreamingEquivalenceCase",
    "StreamingEquivalenceReport",
    "StreamingResult",
    "StreamingSession",
    "build_streaming_cache",
    "build_streaming_test_model",
    "check_streaming_equivalence",
    "describe_cache",
    "enforce_cache_window",
    "iter_waveform_chunks",
    "measure_streaming_latency",
    "resolve_backbone_config",
    "run_stream",
]


DEFAULT_SAMPLING_RATE = 16000

DEFAULT_CHUNK_SCHEDULES: tuple[int | tuple[int, ...], ...] = (
    400,
    1600,
    5120,
    5333,
    (128, 700, 2500, 313, 8000, 61),
)

_PREDICTION_FIELDS = ("vad_predictions", "ctc_logits", "spk_logits", "last_hidden_state")


def resolve_backbone_config(model_or_config: nn.Module | PreTrainedConfig) -> PreTrainedConfig:
    r"""
    Return the configuration of the causal speech encoder.

    Accepts an [`EmoTraceForVADPrediction`], a `CausalWavLMModel`, or either of their configurations, so that every
    helper in this module takes whichever of the two the caller happens to hold.

    Args:
        model_or_config (`torch.nn.Module` or `~configuration_utils.PreTrainedConfig`):
            Model or configuration to resolve.

    Returns:
        `~configuration_utils.PreTrainedConfig`: the encoder configuration, i.e. the one carrying `sliding_window`
        and the convolutional front-end description.
    """
    config = model_or_config.config if isinstance(model_or_config, nn.Module) else model_or_config
    if not isinstance(config, PreTrainedConfig):
        raise TypeError(f"Expected a model or a `PreTrainedConfig`, got {type(model_or_config).__name__}.")

    backbone_config = getattr(config, "backbone_config", None)
    return backbone_config if isinstance(backbone_config, PreTrainedConfig) else config


@dataclass(frozen=True)
class FrameGeometry:
    r"""
    Sample-to-frame arithmetic of the convolutional front end.

    The feature encoder is a stack of strided convolutions without padding or dilation, so output frame `t` is a
    function of exactly the samples `[t * samples_per_frame, t * samples_per_frame + receptive_field)`. Every
    quantity a streaming session needs follows from that single statement: how many frames a buffer yields, how many
    samples a chunk consumes, and how many samples must be retained for the next chunk.

    Args:
        samples_per_frame (`int`):
            Stride of the front end, i.e. the product of `config.conv_stride` (320 samples for WavLM).
        receptive_field (`int`):
            Number of samples a single frame reads (400 samples for WavLM).
        sampling_rate (`int`, *optional*, defaults to 16000):
            Sampling rate the waveform is expected at, used only to convert frames to seconds.
    """

    samples_per_frame: int
    receptive_field: int
    sampling_rate: int = DEFAULT_SAMPLING_RATE

    def __post_init__(self):
        if self.samples_per_frame < 1:
            raise ValueError(f"`samples_per_frame` must be positive, got {self.samples_per_frame}.")
        if self.receptive_field < self.samples_per_frame:
            raise ValueError(
                f"`receptive_field` ({self.receptive_field}) must be at least `samples_per_frame` "
                f"({self.samples_per_frame}); a shorter receptive field would leave gaps between frames."
            )
        if self.sampling_rate < 1:
            raise ValueError(f"`sampling_rate` must be positive, got {self.sampling_rate}.")

    @classmethod
    def from_config(cls, model_or_config, sampling_rate: int = DEFAULT_SAMPLING_RATE) -> FrameGeometry:
        r"""
        Derive the geometry from a model or its configuration.

        Args:
            model_or_config (`torch.nn.Module` or `~configuration_utils.PreTrainedConfig`):
                Model or configuration describing the convolutional front end.
            sampling_rate (`int`, *optional*, defaults to 16000):
                Sampling rate of the waveform.

        Returns:
            [`FrameGeometry`]: the geometry of the front end.
        """
        config = resolve_backbone_config(model_or_config)
        if getattr(config, "add_adapter", False):
            raise ValueError(
                "Streaming inference assumes one output frame per `conv_stride` samples, but the configuration adds "
                "an adapter on top of the encoder, which resamples the frame grid. Set `add_adapter=False`."
            )

        receptive_field = 1
        for kernel, stride in zip(reversed(config.conv_kernel), reversed(config.conv_stride), strict=True):
            receptive_field = (receptive_field - 1) * stride + kernel

        samples_per_frame = 1
        for stride in config.conv_stride:
            samples_per_frame *= stride

        return cls(samples_per_frame=samples_per_frame, receptive_field=receptive_field, sampling_rate=sampling_rate)

    @property
    def lookahead_samples(self) -> int:
        r"""Samples a frame reads past the end of its own step, and the overlap consecutive chunks must share."""
        return self.receptive_field - self.samples_per_frame

    @property
    def frame_duration(self) -> float:
        r"""Duration of one frame step in seconds."""
        return self.samples_per_frame / self.sampling_rate

    @property
    def lookahead_duration(self) -> float:
        r"""Duration in seconds of the front-end look-ahead, 5 ms for WavLM."""
        return self.lookahead_samples / self.sampling_rate

    def num_frames(self, num_samples: int) -> int:
        r"""
        Args:
            num_samples (`int`):
                Length of a waveform buffer.

        Returns:
            `int`: the number of frames the front end produces for it.
        """
        if num_samples < self.receptive_field:
            return 0
        return (num_samples - self.receptive_field) // self.samples_per_frame + 1

    def span_samples(self, num_frames: int) -> int:
        r"""
        Args:
            num_frames (`int`):
                A number of frames.

        Returns:
            `int`: the number of samples they span, from the first sample of the first frame to the last sample read
            by the last one.
        """
        if num_frames < 1:
            return 0
        return (num_frames - 1) * self.samples_per_frame + self.receptive_field

    def frame_time(self, frame_index: int) -> float:
        r"""
        Args:
            frame_index (`int`):
                Absolute index of a frame in the stream.

        Returns:
            `float`: the onset of that frame in seconds.
        """
        return frame_index * self.frame_duration


@dataclass(frozen=True)
class CacheState:
    r"""
    Occupancy of the attention cache of a stream.

    Args:
        seen_frames (`int`):
            Frames the cache has been fed since it was created, i.e. the position of the next query.
        retained_frames (`int`):
            Frames actually held in memory, which plateaus at `sliding_window - 1`.
        sliding_window (`int`):
            Attention window in frames, current frame included.
        num_bytes (`int`):
            Memory held by the cached keys and values, summed over layers.
    """

    seen_frames: int
    retained_frames: int
    sliding_window: int
    num_bytes: int

    def summary(self) -> str:
        r"""Return a one-line human-readable description of the cache occupancy."""
        return (
            f"cache: {self.retained_frames} frames retained of {self.seen_frames} seen "
            f"(window {self.sliding_window}), {self.num_bytes / 2**20:.1f} MiB"
        )


def build_streaming_cache(model_or_config: nn.Module | PreTrainedConfig) -> DynamicCache:
    r"""
    Build the attention cache a streaming session runs on.

    The cache is built from the encoder configuration, which carries `sliding_window`; every layer therefore becomes
    a sliding-window layer that keeps only the `sliding_window - 1` most recent frames and reports the offset of the
    oldest one, so both the attention mask and the relative position bias stay aligned after eviction. This is what
    bounds the per-chunk cost of a stream. A cache built without the configuration would grow without limit and
    would additionally report a wrong key offset.

    Args:
        model_or_config (`torch.nn.Module` or `~configuration_utils.PreTrainedConfig`):
            Model or configuration of the stream.

    Returns:
        [`~cache_utils.DynamicCache`]: an empty cache, to be passed to the model and updated in place.
    """
    config = resolve_backbone_config(model_or_config)
    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        raise ValueError(
            "The encoder configuration has no `sliding_window`, so an attention cache built from it would grow "
            "linearly with the length of the stream. Streaming inference requires a causal sliding-window model "
            "such as `CausalWavLMModel`."
        )

    cache = DynamicCache(config=config)
    enforce_cache_window(cache, sliding_window)
    return cache


def enforce_cache_window(past_key_values: Cache, sliding_window: int | None = None) -> int:
    r"""
    Assert that the attention cache evicts everything outside the sliding window, and report its occupancy.

    Eviction itself happens inside the cache: a sliding-window layer keeps the last `sliding_window - 1` keys and
    values on every update. What this function adds is the checkable statement of that invariant. It raises rather
    than silently truncating a cache that grows, because dropping frames from a layer that reports `kv_offset = 0`
    would misalign the attention mask and the relative position bias instead of merely costing memory.

    Args:
        past_key_values (`~cache_utils.Cache`):
            Cache of the stream.
        sliding_window (`int`, *optional*):
            Expected window, in frames. Checked against the window each layer was built with when given.

    Returns:
        `int`: the largest number of frames retained by any layer, which is bounded by `sliding_window - 1` however
        long the stream runs.
    """
    # A `DynamicCache` built without a configuration holds no layer yet and names the class it will replicate on the
    # first update, which is where a cache that would never evict is caught before it has cost anything.
    lazy_layer_class = getattr(past_key_values, "layer_class_to_replicate", None)
    if lazy_layer_class is not None and not getattr(lazy_layer_class, "is_sliding", False):
        raise ValueError(
            f"The attention cache creates {lazy_layer_class.__name__} layers, which keep every frame of the stream, "
            "so the per-chunk cost would grow without limit. Build the cache with `build_streaming_cache(model)`."
        )

    layers = getattr(past_key_values, "layers", None)
    if not layers:
        return 0

    retained = 0
    for layer_idx, layer in enumerate(layers):
        layer_window = getattr(layer, "sliding_window", None)
        if not getattr(layer, "is_sliding", False) or layer_window is None:
            raise ValueError(
                f"Layer {layer_idx} of the attention cache is not a sliding-window layer, so it would keep every "
                "frame of the stream and the per-chunk cost would grow without limit. Build the cache with "
                "`build_streaming_cache(model)`."
            )
        if sliding_window is not None and layer_window != sliding_window:
            raise ValueError(
                f"Layer {layer_idx} of the attention cache was built for a window of {layer_window} frames but the "
                f"model attends over {sliding_window}."
            )

        length = layer.keys.shape[-2] if getattr(layer, "is_initialized", False) else 0
        if length > layer_window - 1:
            raise RuntimeError(
                f"Layer {layer_idx} of the attention cache retains {length} frames, more than the "
                f"{layer_window - 1} its window allows."
            )
        retained = max(retained, length)

    return retained


def describe_cache(past_key_values: Cache) -> CacheState:
    r"""
    Args:
        past_key_values (`~cache_utils.Cache`):
            Cache of the stream.

    Returns:
        [`CacheState`]: how many frames the cache has seen, how many it still holds and how much memory that costs.
    """
    layers = getattr(past_key_values, "layers", None) or []
    sliding_window = max((getattr(layer, "sliding_window", 0) or 0) for layer in layers) if layers else 0
    seen_frames = max((layer.get_seq_length() for layer in layers), default=0)

    num_bytes = 0
    for layer in layers:
        if getattr(layer, "is_initialized", False):
            num_bytes += layer.keys.numel() * layer.keys.element_size()
            num_bytes += layer.values.numel() * layer.values.element_size()

    return CacheState(
        seen_frames=seen_frames,
        retained_frames=enforce_cache_window(past_key_values),
        sliding_window=sliding_window,
        num_bytes=num_bytes,
    )


@dataclass
class StreamingChunkOutput:
    r"""
    Predictions emitted for one chunk of a stream.

    Every tensor covers the frames `[start_frame, start_frame + num_frames)` of the stream. A chunk that does not
    complete a single frame, because fewer than `receptive_field` samples have accumulated, returns tensors with a
    frame dimension of zero rather than `None`, so a caller can concatenate outputs unconditionally.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_frames, hidden_size)`):
            Frame representations produced by the encoder.
        vad_predictions (`torch.FloatTensor` of shape `(batch_size, num_frames, num_vad_targets)`, *optional*):
            Valence, arousal and dominance. `None` when the model is a bare encoder.
        ctc_logits (`torch.FloatTensor` of shape `(batch_size, num_frames, vocab_size)`, *optional*):
            Scores of the ASR head, `None` when the head is absent.
        spk_logits (`torch.FloatTensor` of shape `(batch_size, num_frames, num_speaker_labels)`, *optional*):
            Scores of the speaker head, `None` when the head is absent.
        start_frame (`int`):
            Absolute index of the first frame emitted here.
        num_frames (`int`):
            Number of frames emitted here.
        num_input_samples (`int`):
            Number of waveform samples the caller handed over for this chunk.
        elapsed_seconds (`float`):
            Wall-clock time spent computing this chunk, excluding the time spent waiting for the audio.
    """

    last_hidden_state: torch.FloatTensor
    vad_predictions: torch.FloatTensor | None
    ctc_logits: torch.FloatTensor | None
    spk_logits: torch.FloatTensor | None
    start_frame: int
    num_frames: int
    num_input_samples: int
    elapsed_seconds: float

    @property
    def end_frame(self) -> int:
        r"""Absolute index one past the last frame emitted here."""
        return self.start_frame + self.num_frames

    @property
    def is_empty(self) -> bool:
        r"""Whether this chunk completed no frame at all."""
        return self.num_frames == 0


@dataclass
class StreamingResult:
    r"""
    Predictions of a whole stream, concatenated over chunks.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_frames, hidden_size)`, *optional*):
            Frame representations, `None` when the run was asked not to collect them.
        vad_predictions (`torch.FloatTensor` of shape `(batch_size, num_frames, num_vad_targets)`, *optional*):
            Valence, arousal and dominance over the whole stream.
        ctc_logits (`torch.FloatTensor` of shape `(batch_size, num_frames, vocab_size)`, *optional*):
            Scores of the ASR head over the whole stream.
        spk_logits (`torch.FloatTensor` of shape `(batch_size, num_frames, num_speaker_labels)`, *optional*):
            Scores of the speaker head over the whole stream.
        num_frames (`int`):
            Frames emitted in total.
        num_chunks (`int`):
            Chunks the stream was fed in.
        num_samples (`int`):
            Waveform samples fed in total.
        chunk_latencies (`tuple[float, ...]`):
            Wall-clock seconds spent on each chunk, in order.
        peak_retained_cache_frames (`int`):
            Largest occupancy the attention cache reached, bounded by `sliding_window - 1`.
    """

    last_hidden_state: torch.FloatTensor | None
    vad_predictions: torch.FloatTensor | None
    ctc_logits: torch.FloatTensor | None
    spk_logits: torch.FloatTensor | None
    num_frames: int
    num_chunks: int
    num_samples: int
    chunk_latencies: tuple[float, ...]
    peak_retained_cache_frames: int

    @property
    def total_seconds(self) -> float:
        r"""Total wall-clock compute time of the stream, in seconds."""
        return float(sum(self.chunk_latencies))

    def predictions(self) -> dict[str, torch.Tensor]:
        r"""
        Returns:
            `dict[str, torch.Tensor]`: the prediction tensors that are present, keyed by field name.
        """
        return {name: getattr(self, name) for name in _PREDICTION_FIELDS if getattr(self, name) is not None}


class StreamingSession:
    r"""
    Runs a causal model over a waveform that arrives in pieces.

    The session owns everything that has to survive a chunk boundary: the waveform tail the convolutional front end
    still needs, the left context of the causal positional convolution and the attention key-value cache. Chunks
    handed to [`process`] are treated as consecutive, disjoint pieces of one recording, and the frames emitted over
    the whole stream are the frames a single forward pass over the concatenated audio would produce, up to floating
    point reassociation. [`check_streaming_equivalence`] verifies exactly that.

    Chunks may have any length, including lengths that are not multiples of the frame stride and lengths too short
    to complete a frame; the session buffers what it cannot use yet. The residual of fewer than `receptive_field`
    samples left at the end of a recording never produces a frame, which is also true of a single forward pass.

    Every sample of the batch must carry real audio: a stream has no padding to mask, and a batch of unrelated
    recordings of different lengths must be run as separate sessions.

    Args:
        model (`torch.nn.Module`):
            An [`EmoTraceForVADPrediction`] or a bare `CausalWavLMModel`. It must be in evaluation mode.
        feature_extractor (`~feature_extraction_sequence_utils.SequenceFeatureExtractor`, *optional*):
            Feature extractor whose fixed `mean_value` and `std_value` normalise the incoming chunks. A feature
            extractor that normalises with per-utterance statistics is rejected: those statistics are not available
            online and would make the result depend on the chunking.
        sampling_rate (`int`, *optional*, defaults to 16000):
            Sampling rate of the incoming waveform.
        past_key_values (`~cache_utils.Cache`, *optional*):
            Attention cache to run on. Built with [`build_streaming_cache`] when omitted.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_extractor=None,
        sampling_rate: int = DEFAULT_SAMPLING_RATE,
        past_key_values: Cache | None = None,
    ):
        self.model = model
        self.encoder, self._applies_heads = _resolve_encoder(model)
        self.config = resolve_backbone_config(self.encoder)
        self.geometry = FrameGeometry.from_config(self.config, sampling_rate=sampling_rate)
        self.sliding_window = self.config.sliding_window

        parameter = next(model.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype

        self._mean, self._std = _normalization_constants(feature_extractor)
        self._external_cache = past_key_values
        if past_key_values is None:
            past_key_values = build_streaming_cache(self.config)
        enforce_cache_window(past_key_values, self.sliding_window)
        self.past_key_values: Cache = past_key_values
        self.padding_cache = CausalWavLMPositionalConvCache()
        self._tail: torch.Tensor | None = None
        self._batch_size: int | None = None
        self.num_frames_emitted = 0
        self.num_samples_consumed = 0

        if model.training:
            logger.warning_once(
                "The model is in training mode; dropout and layer drop make the streamed frames non-deterministic "
                "and different from a single forward pass. Call `model.eval()` before streaming."
            )

    def reset(self) -> None:
        r"""
        Forget the stream and start a new recording.

        A fresh attention cache is allocated rather than the existing one being cleared: clearing a sliding-window
        cache zeroes the retained keys and values without shrinking them, so the next chunk would attend over a
        window of zeros. When a cache was supplied by the caller, that same object is reused only if it is empty.
        """
        if self._external_cache is not None and _cache_length(self._external_cache) == 0:
            self.past_key_values = self._external_cache
        else:
            self.past_key_values = build_streaming_cache(self.config)
        self.padding_cache = CausalWavLMPositionalConvCache()
        self._tail = None
        self._batch_size = None
        self.num_frames_emitted = 0
        self.num_samples_consumed = 0

    @property
    def pending_samples(self) -> int:
        r"""Samples held back because they do not complete a frame yet."""
        return 0 if self._tail is None else int(self._tail.shape[-1])

    @property
    def cache_state(self) -> CacheState:
        r"""Occupancy of the attention cache, as a [`CacheState`]."""
        return describe_cache(self.past_key_values)

    def process(self, waveform) -> StreamingChunkOutput:
        r"""
        Consume the next piece of the recording and emit the frames it completes.

        Args:
            waveform (`torch.Tensor`, `numpy.ndarray` or nested `list`):
                Waveform samples of shape `(batch_size, num_samples)`, or `(num_samples,)` for a single stream. The
                chunk continues the audio handed over so far and must not repeat any of it.

        Returns:
            [`StreamingChunkOutput`]: the frames completed by this chunk, possibly none.
        """
        waveform = self._prepare(waveform)
        batch_size, num_input_samples = waveform.shape

        if self._batch_size is None:
            self._batch_size = batch_size
        elif batch_size != self._batch_size:
            raise ValueError(
                f"This session streams {self._batch_size} recordings in parallel but the chunk holds {batch_size}. "
                "Call `reset()` before streaming a different batch."
            )

        buffer = waveform if self._tail is None else torch.cat([self._tail, waveform], dim=-1)
        num_frames = self.geometry.num_frames(buffer.shape[-1])
        start_frame = self.num_frames_emitted

        if num_frames == 0:
            self._tail = buffer
            return self._empty_output(batch_size, start_frame, num_input_samples)

        consumed = num_frames * self.geometry.samples_per_frame
        inputs = buffer[:, : self.geometry.span_samples(num_frames)]
        # Copied so that the buffer of the current chunk can be released instead of being kept alive by a view.
        self._tail = buffer[:, consumed:].clone()

        started_at = time.perf_counter()
        outputs = self._forward(inputs, past_key_values=self.past_key_values, padding_cache=self.padding_cache)
        _synchronize(self.device)
        elapsed = time.perf_counter() - started_at

        if outputs["last_hidden_state"].shape[1] != num_frames:
            raise RuntimeError(
                f"Expected the encoder to emit {num_frames} frames for {inputs.shape[-1]} samples but it emitted "
                f"{outputs['last_hidden_state'].shape[1]}. The frame geometry does not describe this model."
            )

        self.num_frames_emitted += num_frames
        self.num_samples_consumed += consumed

        return StreamingChunkOutput(
            last_hidden_state=outputs["last_hidden_state"],
            vad_predictions=outputs["vad_predictions"],
            ctc_logits=outputs["ctc_logits"],
            spk_logits=outputs["spk_logits"],
            start_frame=start_frame,
            num_frames=num_frames,
            num_input_samples=num_input_samples,
            elapsed_seconds=elapsed,
        )

    def stream(self, chunks: Iterable) -> Iterator[StreamingChunkOutput]:
        r"""
        Args:
            chunks (`Iterable`):
                Consecutive pieces of one recording.

        Yields:
            [`StreamingChunkOutput`]: the output of every chunk, including the ones that complete no frame.
        """
        for chunk in chunks:
            yield self.process(chunk)

    def forward_offline(self, waveform) -> StreamingResult:
        r"""
        Run a single forward pass over a whole waveform, with no cache and no chunking.

        This is the reference a stream is compared against: it goes through the same encoder and the same heads as
        [`process`], so any difference measured against it comes from the chunking alone. The session state is left
        untouched.

        Args:
            waveform (`torch.Tensor`, `numpy.ndarray` or nested `list`):
                Waveform of shape `(batch_size, num_samples)` or `(num_samples,)`.

        Returns:
            [`StreamingResult`]: the predictions over the whole waveform.
        """
        waveform = self._prepare(waveform)
        num_frames = self.geometry.num_frames(waveform.shape[-1])
        if num_frames == 0:
            raise ValueError(
                f"A waveform of {waveform.shape[-1]} samples completes no frame; the convolutional front end needs "
                f"at least {self.geometry.receptive_field} samples."
            )
        inputs = waveform[:, : self.geometry.span_samples(num_frames)]

        started_at = time.perf_counter()
        outputs = self._forward(inputs, past_key_values=None, padding_cache=None)
        _synchronize(self.device)
        elapsed = time.perf_counter() - started_at

        return StreamingResult(
            last_hidden_state=outputs["last_hidden_state"],
            vad_predictions=outputs["vad_predictions"],
            ctc_logits=outputs["ctc_logits"],
            spk_logits=outputs["spk_logits"],
            num_frames=num_frames,
            num_chunks=1,
            num_samples=int(waveform.shape[-1]),
            chunk_latencies=(elapsed,),
            peak_retained_cache_frames=0,
        )

    def _forward(self, inputs, past_key_values, padding_cache) -> dict[str, torch.Tensor | None]:
        r"""Run the encoder and the heads on one already aligned span of waveform."""
        with torch.inference_mode():
            outputs = self.encoder(
                inputs,
                attention_mask=None,
                past_key_values=past_key_values,
                padding_cache=padding_cache,
                return_dict=True,
            )
            if self._applies_heads:
                return self._run_heads(outputs.last_hidden_state)
            return {
                "last_hidden_state": outputs.last_hidden_state,
                "vad_predictions": getattr(outputs, "vad_predictions", None),
                "ctc_logits": getattr(outputs, "ctc_logits", None),
                "spk_logits": getattr(outputs, "spk_logits", None),
            }

    def _run_heads(self, last_hidden_state: torch.Tensor) -> dict[str, torch.Tensor | None]:
        r"""Apply the thin prediction heads of the model to the encoder output."""
        hidden_states = last_hidden_state
        head_dropout = getattr(self.model, "dropout", None)
        if head_dropout is not None:
            hidden_states = head_dropout(hidden_states)

        outputs: dict[str, torch.Tensor | None] = {"last_hidden_state": last_hidden_state}
        for field_name, attribute in (
            ("vad_predictions", "vad_head"),
            ("ctc_logits", "ctc_head"),
            ("spk_logits", "spk_head"),
        ):
            head = getattr(self.model, attribute, None)
            outputs[field_name] = head(hidden_states) if head is not None else None
        return outputs

    def _empty_output(self, batch_size: int, start_frame: int, num_input_samples: int) -> StreamingChunkOutput:
        r"""Build the zero-frame output of a chunk that does not complete a frame yet."""

        def empty(size: int | None) -> torch.Tensor | None:
            if size is None:
                return None
            return torch.zeros((batch_size, 0, size), dtype=self.dtype, device=self.device)

        return StreamingChunkOutput(
            last_hidden_state=empty(self.config.hidden_size),
            vad_predictions=empty(_head_width(self.model, "vad_head")),
            ctc_logits=empty(_head_width(self.model, "ctc_head")),
            spk_logits=empty(_head_width(self.model, "spk_head")),
            start_frame=start_frame,
            num_frames=0,
            num_input_samples=num_input_samples,
            elapsed_seconds=0.0,
        )

    def _prepare(self, waveform) -> torch.Tensor:
        r"""Turn an incoming chunk into a `(batch_size, num_samples)` tensor on the model device, normalised."""
        tensor = torch.as_tensor(waveform)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 2:
            raise ValueError(
                f"Expected a chunk of shape (batch_size, num_samples) or (num_samples,), got {tuple(tensor.shape)}."
            )

        tensor = tensor.to(device=self.device, dtype=self.dtype)
        if self._mean is not None:
            tensor = (tensor - self._mean) / self._std
        return tensor


def iter_waveform_chunks(waveform, chunk_sizes: int | Sequence[int]) -> Iterator[torch.Tensor]:
    r"""
    Split a waveform into consecutive chunks, the way an audio stream would deliver it.

    Args:
        waveform (`torch.Tensor`, `numpy.ndarray` or nested `list`):
            Waveform of shape `(batch_size, num_samples)` or `(num_samples,)`.
        chunk_sizes (`int` or `Sequence[int]`):
            A constant chunk length in samples, or an explicit schedule of lengths that is cycled through. A varying
            schedule is the interesting case: it exercises chunk boundaries that fall inside a frame.

    Yields:
        `torch.Tensor`: the chunks, of shape `(batch_size, chunk_length)`.
    """
    tensor = torch.as_tensor(waveform)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)

    schedule = (chunk_sizes,) if isinstance(chunk_sizes, int) else tuple(chunk_sizes)
    if not schedule or any(size < 1 for size in schedule):
        raise ValueError(f"`chunk_sizes` must contain positive sample counts, got {chunk_sizes}.")

    position, step = 0, 0
    num_samples = tensor.shape[-1]
    while position < num_samples:
        size = schedule[step % len(schedule)]
        yield tensor[:, position : position + size]
        position += size
        step += 1


def run_stream(
    session: StreamingSession,
    chunks: Iterable,
    collect: bool = True,
    reset: bool = True,
) -> StreamingResult:
    r"""
    Feed a whole sequence of chunks to a session and gather the result.

    Args:
        session ([`StreamingSession`]):
            Session to run.
        chunks (`Iterable`):
            Consecutive pieces of one recording.
        collect (`bool`, *optional*, defaults to `True`):
            Whether to concatenate the per-chunk predictions. Set to `False` for long benchmark streams, where only
            the timings matter and holding every frame in memory is wasteful.
        reset (`bool`, *optional*, defaults to `True`):
            Whether to reset the session before the first chunk.

    Returns:
        [`StreamingResult`]: the concatenated predictions, the per-chunk timings and the peak cache occupancy.
    """
    if reset:
        session.reset()

    collected: dict[str, list[torch.Tensor]] = {name: [] for name in _PREDICTION_FIELDS}
    latencies: list[float] = []
    num_chunks = 0
    num_samples = 0
    num_frames = 0
    peak_retained = 0

    for output in session.stream(chunks):
        num_chunks += 1
        num_samples += output.num_input_samples
        num_frames += output.num_frames
        latencies.append(output.elapsed_seconds)
        peak_retained = max(peak_retained, enforce_cache_window(session.past_key_values, session.sliding_window))
        if collect:
            for name in _PREDICTION_FIELDS:
                tensor = getattr(output, name)
                if tensor is not None:
                    collected[name].append(tensor)

    merged: dict[str, torch.Tensor | None] = {}
    for name in _PREDICTION_FIELDS:
        parts = collected[name]
        merged[name] = torch.cat(parts, dim=1) if parts else None

    return StreamingResult(
        last_hidden_state=merged["last_hidden_state"],
        vad_predictions=merged["vad_predictions"],
        ctc_logits=merged["ctc_logits"],
        spk_logits=merged["spk_logits"],
        num_frames=num_frames,
        num_chunks=num_chunks,
        num_samples=num_samples,
        chunk_latencies=tuple(latencies),
        peak_retained_cache_frames=peak_retained,
    )


@dataclass(frozen=True)
class StreamingEquivalenceCase:
    r"""
    Result of comparing one chunking schedule against a single forward pass.

    Args:
        chunk_sizes (`tuple[int, ...]`):
            Schedule of chunk lengths in samples that was streamed.
        num_chunks (`int`):
            Chunks the waveform was split into.
        num_frames (`int`):
            Frames emitted by the stream.
        reference_frames (`int`):
            Frames emitted by the single forward pass.
        max_abs_diff (`dict[str, float]`):
            Largest absolute deviation from the single forward pass, per output tensor.
        peak_retained_cache_frames (`int`):
            Largest attention cache occupancy reached during the stream.
        passed (`bool`):
            Whether the frame counts agree and every deviation is within tolerance.
    """

    chunk_sizes: tuple[int, ...]
    num_chunks: int
    num_frames: int
    reference_frames: int
    max_abs_diff: dict[str, float]
    peak_retained_cache_frames: int
    passed: bool

    @property
    def worst_abs_diff(self) -> float:
        r"""Largest absolute deviation over all output tensors."""
        return max(self.max_abs_diff.values(), default=0.0)


@dataclass(frozen=True)
class StreamingEquivalenceReport:
    r"""
    Outcome of [`check_streaming_equivalence`].

    Args:
        passed (`bool`):
            Whether every chunking schedule reproduced the single forward pass.
        atol (`float`):
            Absolute tolerance the comparison used.
        num_samples (`int`):
            Length of the waveform that was streamed.
        sliding_window (`int`):
            Attention window of the model, in frames.
        cases (`tuple[StreamingEquivalenceCase, ...]`):
            One entry per chunking schedule.
    """

    passed: bool
    atol: float
    num_samples: int
    sliding_window: int
    cases: tuple[StreamingEquivalenceCase, ...]

    def summary(self) -> str:
        r"""Return a human-readable table of the per-schedule results."""
        lines = [
            f"streaming equivalence: {'PASS' if self.passed else 'FAIL'} "
            f"(atol {self.atol:g}, {self.num_samples} samples, sliding window {self.sliding_window} frames)",
            f"{'chunk sizes':>34}  {'chunks':>6}  {'frames':>6}  {'max|diff|':>10}  {'cache':>6}  result",
        ]
        for case in self.cases:
            sizes = ",".join(str(size) for size in case.chunk_sizes)
            if len(sizes) > 34:
                sizes = sizes[:31] + "..."
            lines.append(
                f"{sizes:>34}  {case.num_chunks:6d}  {case.num_frames:6d}  {case.worst_abs_diff:10.3e}  "
                f"{case.peak_retained_cache_frames:6d}  {'ok' if case.passed else 'MISMATCH'}"
            )
        return "\n".join(lines)


def check_streaming_equivalence(
    model: nn.Module | None = None,
    waveform=None,
    num_samples: int = 16000,
    batch_size: int = 1,
    chunk_schedules: Sequence[int | Sequence[int]] = DEFAULT_CHUNK_SCHEDULES,
    atol: float = 1e-5,
    device: torch.device | str | None = None,
    seed: int = 0,
) -> StreamingEquivalenceReport:
    r"""
    Check that chunked streaming reproduces a single forward pass, frame for frame.

    For every chunking schedule the same waveform is streamed and compared against
    [`StreamingSession.forward_offline`] on the concatenated audio. The schedules deliberately include chunks
    shorter than one frame, chunk lengths that are not multiples of the frame stride, and a varying schedule, since
    those are the cases where a naive implementation silently loses or duplicates a frame. The waveform is long
    enough for the attention cache to start evicting, so the sliding window and the relative position bias offsets
    are exercised too.

    Runnable on its own: with no model it builds a small randomly initialised one, needing neither a checkpoint nor
    a corpus.

    Args:
        model (`torch.nn.Module`, *optional*):
            Model to check. A small test model is built when omitted.
        waveform (`torch.Tensor`, *optional*):
            Waveform to stream. Random noise of `num_samples` samples is used when omitted.
        num_samples (`int`, *optional*, defaults to 16000):
            Length of the generated waveform.
        batch_size (`int`, *optional*, defaults to 1):
            Number of parallel streams in the generated waveform.
        chunk_schedules (`Sequence[int | Sequence[int]]`, *optional*):
            Chunking schedules to test. An `int` is a constant chunk length in samples; a sequence is a cycle of
            lengths.
        atol (`float`, *optional*, defaults to 1e-5):
            Absolute tolerance. Streaming and a single pass differ only by floating point reassociation, since the
            tensors handed to the kernels have different shapes.
        device (`torch.device` or `str`, *optional*):
            Device to run on. Defaults to the device of the model.
        seed (`int`, *optional*, defaults to 0):
            Seed of the generated model weights and waveform.

    Returns:
        [`StreamingEquivalenceReport`]: the per-schedule deviations and the overall verdict.
    """
    if model is None:
        model = build_streaming_test_model(seed=seed, device=device)
    elif device is not None:
        model = model.to(device)
    model.eval()

    session = StreamingSession(model)
    if waveform is None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        waveform = torch.randn(batch_size, num_samples, generator=generator)
    waveform = torch.as_tensor(waveform)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    reference = session.forward_offline(waveform)
    reference_tensors = reference.predictions()

    cases = []
    for schedule in chunk_schedules:
        sizes = (schedule,) if isinstance(schedule, int) else tuple(schedule)
        streamed = run_stream(session, iter_waveform_chunks(waveform, sizes))
        streamed_tensors = streamed.predictions()

        deviations = {}
        for name, expected in reference_tensors.items():
            actual = streamed_tensors.get(name)
            if actual is None or actual.shape != expected.shape:
                deviations[name] = float("inf")
            else:
                deviations[name] = float((actual.float() - expected.float()).abs().max())

        passed = streamed.num_frames == reference.num_frames and all(value <= atol for value in deviations.values())
        cases.append(
            StreamingEquivalenceCase(
                chunk_sizes=sizes,
                num_chunks=streamed.num_chunks,
                num_frames=streamed.num_frames,
                reference_frames=reference.num_frames,
                max_abs_diff=deviations,
                peak_retained_cache_frames=streamed.peak_retained_cache_frames,
                passed=passed,
            )
        )

    return StreamingEquivalenceReport(
        passed=all(case.passed for case in cases),
        atol=atol,
        num_samples=int(waveform.shape[-1]),
        sliding_window=session.sliding_window,
        cases=tuple(cases),
    )


@dataclass(frozen=True)
class LatencyReport:
    r"""
    Timing of a streaming run.

    Args:
        chunk_samples (`int`):
            Chunk length in samples.
        chunk_duration (`float`):
            Chunk length in seconds of audio.
        num_chunks (`int`):
            Chunks measured, warm-up excluded.
        num_frames (`int`):
            Frames emitted over the whole stream, warm-up included.
        audio_seconds (`float`):
            Audio duration of the measured chunks.
        compute_seconds (`float`):
            Wall-clock time spent on them.
        real_time_factor (`float`):
            `compute_seconds / audio_seconds`. Below 1 the model keeps up with the stream.
        latencies_ms (`tuple[float, ...]`):
            Per-chunk wall-clock time in milliseconds.
        algorithmic_latency_ms (`float`):
            Delay a frame suffers before its prediction can exist at all, independent of how fast the model is: a
            frame is only emitted once the chunk that completes its receptive field has arrived, so in the worst
            case one full chunk plus the front-end look-ahead.
        peak_retained_cache_frames (`int`):
            Largest attention cache occupancy, which plateaus at `sliding_window - 1`.
        drift_ms (`float`):
            Median chunk time over the last quarter of the stream minus the median over the first quarter. A value
            near zero is the evidence that the sliding-window eviction keeps the per-chunk cost constant.
    """

    chunk_samples: int
    chunk_duration: float
    num_chunks: int
    num_frames: int
    audio_seconds: float
    compute_seconds: float
    real_time_factor: float
    latencies_ms: tuple[float, ...]
    algorithmic_latency_ms: float
    peak_retained_cache_frames: int
    drift_ms: float

    def percentile(self, fraction: float) -> float:
        r"""
        Args:
            fraction (`float`):
                Quantile in `[0, 1]`, e.g. `0.95`.

        Returns:
            `float`: the per-chunk latency at that quantile, in milliseconds.
        """
        return _percentile(self.latencies_ms, fraction)

    @property
    def percentiles_ms(self) -> dict[str, float]:
        r"""Per-chunk latency at the usual quantiles, in milliseconds."""
        return {f"p{int(fraction * 100)}": self.percentile(fraction) for fraction in (0.5, 0.9, 0.95, 0.99)}

    @property
    def mean_latency_ms(self) -> float:
        r"""Mean per-chunk latency in milliseconds."""
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else float("nan")

    @property
    def max_latency_ms(self) -> float:
        r"""Largest per-chunk latency in milliseconds."""
        return max(self.latencies_ms, default=float("nan"))

    def summary(self) -> str:
        r"""Return a human-readable multi-line summary of the timings."""
        percentiles = self.percentiles_ms
        return "\n".join(
            [
                f"streaming latency over {self.num_chunks} chunks of {self.chunk_duration * 1000:.0f} ms "
                f"({self.audio_seconds:.1f} s of audio, {self.num_frames} frames)",
                f"  per-chunk compute  mean {self.mean_latency_ms:7.2f} ms   "
                f"p50 {percentiles['p50']:7.2f}   p90 {percentiles['p90']:7.2f}   "
                f"p95 {percentiles['p95']:7.2f}   p99 {percentiles['p99']:7.2f}   max {self.max_latency_ms:7.2f}",
                f"  real-time factor   {self.real_time_factor:.4f} "
                f"({self.compute_seconds:.2f} s compute / {self.audio_seconds:.2f} s audio)",
                f"  algorithmic delay  {self.algorithmic_latency_ms:.1f} ms (chunk + front-end look-ahead)",
                f"  cache plateau      {self.peak_retained_cache_frames} frames, drift {self.drift_ms:+.2f} ms",
            ]
        )


def measure_streaming_latency(
    model: nn.Module | None = None,
    chunk_duration: float = 0.32,
    stream_duration: float = 20.0,
    warmup_chunks: int = 3,
    num_threads: int | None = None,
    device: torch.device | str | None = None,
    seed: int = 0,
    waveform=None,
) -> LatencyReport:
    r"""
    Measure how long a chunk takes to process and how that compares to the audio it covers.

    A stream of white noise is pushed through a [`StreamingSession`] in fixed-size chunks and every chunk is timed
    individually, so the report carries the distribution rather than an average: an online system is limited by its
    tail, not by its mean. The first `warmup_chunks` chunks are dropped, since they pay for lazy allocations and for
    a cache that has not reached its steady state yet.

    Runnable on its own: with no model it builds a small randomly initialised one.

    Args:
        model (`torch.nn.Module`, *optional*):
            Model to measure. A small test model is built when omitted.
        chunk_duration (`float`, *optional*, defaults to 0.32):
            Audio duration of one chunk, in seconds.
        stream_duration (`float`, *optional*, defaults to 20.0):
            Total audio duration to stream, in seconds.
        warmup_chunks (`int`, *optional*, defaults to 3):
            Leading chunks to exclude from the statistics.
        num_threads (`int`, *optional*):
            Intra-op thread count to run under, restored afterwards. Worth pinning on a shared machine: a chunk of
            audio is a small tensor, so with many threads the measurement reports thread launch overhead rather
            than the cost of the model, and a small model can look two orders of magnitude slower than it is.
        device (`torch.device` or `str`, *optional*):
            Device to run on. Defaults to the device of the model.
        seed (`int`, *optional*, defaults to 0):
            Seed of the generated model weights and waveform.
        waveform (`torch.Tensor`, *optional*):
            Waveform to stream instead of generated noise.

    Returns:
        [`LatencyReport`]: per-chunk latency percentiles, the real-time factor and the cache plateau.
    """
    if model is None:
        model = build_streaming_test_model(seed=seed, device=device)
    elif device is not None:
        model = model.to(device)
    model.eval()

    session = StreamingSession(model)
    geometry = session.geometry
    chunk_samples = max(1, int(round(chunk_duration * geometry.sampling_rate)))

    if waveform is None:
        num_samples = max(chunk_samples, int(round(stream_duration * geometry.sampling_rate)))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        waveform = torch.randn(1, num_samples, generator=generator)

    chunks = list(iter_waveform_chunks(waveform, chunk_samples))
    previous_threads = torch.get_num_threads()
    try:
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        result = run_stream(session, chunks, collect=False)
    finally:
        if num_threads is not None:
            torch.set_num_threads(previous_threads)

    latencies_ms = [latency * 1000.0 for latency in result.chunk_latencies[warmup_chunks:]]
    if not latencies_ms:
        raise ValueError(
            f"The stream produced {result.num_chunks} chunks, which is not more than `warmup_chunks` "
            f"({warmup_chunks}); lengthen `stream_duration` or shorten `chunk_duration`."
        )

    measured_samples = sum(int(chunk.shape[-1]) for chunk in chunks[warmup_chunks:])
    audio_seconds = measured_samples / geometry.sampling_rate
    compute_seconds = sum(latencies_ms) / 1000.0

    quarter = max(1, len(latencies_ms) // 4)
    drift_ms = _percentile(latencies_ms[-quarter:], 0.5) - _percentile(latencies_ms[:quarter], 0.5)

    return LatencyReport(
        chunk_samples=chunk_samples,
        chunk_duration=chunk_samples / geometry.sampling_rate,
        num_chunks=len(latencies_ms),
        num_frames=result.num_frames,
        audio_seconds=audio_seconds,
        compute_seconds=compute_seconds,
        real_time_factor=compute_seconds / audio_seconds if audio_seconds > 0 else float("nan"),
        latencies_ms=tuple(latencies_ms),
        algorithmic_latency_ms=(chunk_samples / geometry.sampling_rate + geometry.lookahead_duration) * 1000.0,
        peak_retained_cache_frames=result.peak_retained_cache_frames,
        drift_ms=drift_ms,
    )


def build_streaming_test_model(
    hidden_size: int = 32,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 2,
    sliding_window: int = 16,
    with_heads: bool = True,
    seed: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    r"""
    Build a small randomly initialised model with the architecture of the real one.

    It keeps the convolutional front end of WavLM, so the frame geometry and every boundary effect are the real
    ones, and shrinks only the transformer. This is what makes the checks in this module runnable in seconds without
    a checkpoint or a corpus.

    Args:
        hidden_size (`int`, *optional*, defaults to 32):
            Width of the encoder.
        num_hidden_layers (`int`, *optional*, defaults to 2):
            Number of encoder layers.
        num_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads.
        sliding_window (`int`, *optional*, defaults to 16):
            Attention window in frames. The default is small on purpose so that a short test stream still overflows
            it and exercises cache eviction.
        with_heads (`bool`, *optional*, defaults to `True`):
            Whether to build the full [`EmoTraceForVADPrediction`] or only the bare causal encoder.
        seed (`int`, *optional*, defaults to 0):
            Seed of the weight initialisation.
        device (`torch.device` or `str`, *optional*):
            Device to place the model on.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Parameter dtype.

    Returns:
        `torch.nn.Module`: the model, in evaluation mode.
    """
    from ..models.CausalWavLM import CausalWavLMConfig, CausalWavLMModel
    from ..models.EmoTrace import EmoTraceConfig, EmoTraceForVADPrediction

    backbone_config = CausalWavLMConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=2 * hidden_size,
        conv_dim=(hidden_size // 2,) * 7,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=4,
        num_buckets=8,
        max_bucket_distance=32,
        sliding_window=sliding_window,
        feat_extract_norm="layer",
        do_stable_layer_norm=True,
        layerdrop=0.0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        feat_proj_dropout=0.0,
        final_dropout=0.0,
        mask_time_prob=0.0,
        mask_feature_prob=0.0,
        apply_spec_augment=False,
    )

    torch.manual_seed(seed)
    if with_heads:
        model = EmoTraceForVADPrediction(EmoTraceConfig(backbone_config=backbone_config, vocab_size=16))
    else:
        model = CausalWavLMModel(backbone_config)

    return model.to(device=device, dtype=dtype).eval()


def _resolve_encoder(model: nn.Module) -> tuple[nn.Module, bool]:
    r"""
    Find the module of `model` that accepts the streaming caches.

    Returns the module to call and whether the heads still have to be applied on top of its output. A model whose
    own `forward` takes `padding_cache` is called as it is; otherwise its `backbone` runs the encoder and the heads
    are applied afterwards, since the positional convolution state cannot reach the encoder any other way.
    """
    if _forward_accepts(model, "padding_cache"):
        return model, False

    backbone = getattr(model, "backbone", None)
    if backbone is not None and _forward_accepts(backbone, "padding_cache"):
        return backbone, True

    raise TypeError(
        f"{type(model).__name__} cannot be streamed: neither it nor its `backbone` accepts a `padding_cache` "
        "argument, so the left context of the causal positional convolution would be lost at every chunk boundary. "
        "Streaming requires a `CausalWavLMModel` or a model built on top of one."
    )


def _forward_accepts(module: nn.Module, name: str) -> bool:
    r"""Whether `module.forward` declares `name` as an explicit parameter rather than swallowing it in `**kwargs`."""
    try:
        parameters = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return False
    parameter = parameters.get(name)
    return parameter is not None and parameter.kind is not inspect.Parameter.VAR_KEYWORD


def _head_width(model: nn.Module, attribute: str) -> int | None:
    r"""Output width of a prediction head, or `None` when the model does not have it."""
    head = getattr(model, attribute, None)
    return head.out_features if head is not None else None


def _normalization_constants(feature_extractor) -> tuple[float | None, float | None]:
    r"""Read the fixed waveform normalisation constants of a feature extractor, rejecting per-utterance ones."""
    if feature_extractor is None:
        return None, None

    if getattr(feature_extractor, "do_normalize", False):
        raise ValueError(
            "The feature extractor normalises every utterance with its own statistics, which do not exist while a "
            "stream is still running and would make the predictions depend on the chunking. Streaming requires the "
            "fixed dataset-level constants of `EmoTraceFeatureExtractor`."
        )

    mean = getattr(feature_extractor, "mean_value", None)
    std = getattr(feature_extractor, "std_value", None)
    if mean is None or std is None:
        return None, None
    if std <= 0:
        raise ValueError(f"`std_value` must be strictly positive, got {std}.")
    return float(mean), float(std)


def _cache_length(past_key_values: Cache) -> int:
    r"""Number of frames a cache has already seen, tolerating a cache whose layers are not initialised yet."""
    layers = getattr(past_key_values, "layers", None) or []
    return max((layer.get_seq_length() for layer in layers), default=0)


def _synchronize(device: torch.device) -> None:
    r"""Wait for the queued kernels of an accelerator, so that a timing measures compute and not queueing."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _percentile(values: Sequence[float], fraction: float) -> float:
    r"""Linear-interpolation quantile of an unsorted sequence, with no dependency on a numerical library."""
    if not values:
        return float("nan")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"`fraction` must lie in [0, 1], got {fraction}.")

    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


if __name__ == "__main__":
    print(check_streaming_equivalence().summary())
    print()
    print(measure_streaming_latency(num_threads=1).summary())
