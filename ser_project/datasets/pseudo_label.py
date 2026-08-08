r"""Transcript pseudo-labelling pipeline for the MSP-Conversation release.

Every conversation part is transcribed twice with CrisperWhisper 2.0: once in `verbatim`
mode, which writes down disfluencies and atomic vocal-event tokens such as `[uh]`, `[um]`
or `[laughter]`, and once in `intended` mode, which writes the cleaned transcript. The
verbatim transcript is the CTC training target and the intended transcript is the ablation
condition, as well as the hypothesis of the quality-control pass that scores the
pseudo-labels against the human MSP-Podcast transcripts.

The transcription pass is expensive, so its results are serialized as JSON lines next to
the corpus and every later stage reads that file. Only [`run_pseudo_labelling`] and the
audio reader touch the corpus or the checkpoint; the normalization, consistency, masking
and scoring helpers work on plain strings and can be exercised with fixtures.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import string
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .base import CorpusLayoutError, CorpusPaths, DiarizationSegment, require_path
from .msp_conversation import (
    conversation_of,
    load_diarization,
    read_part_times,
    read_partitions,
    segments_within,
)


DEFAULT_MODEL_ID = "nyralabs/CrisperWhisper2.0_large"

MODEL_IDS: dict[str, str] = {
    "large": "nyralabs/CrisperWhisper2.0_large",
    "turbo": "nyralabs/CrisperWhisper2.0_turbo",
    "medium": "nyralabs/CrisperWhisper2.0_medium",
    "small": "nyralabs/CrisperWhisper2.0_small",
}

VERBATIM_MODE = "verbatim"
INTENDED_MODE = "intended"

FILLER_TAGS: tuple[str, ...] = ("[uh]", "[um]")
r"""Filled-pause tags of the verbatim transcription convention."""

SOUND_TAGS: tuple[str, ...] = (
    "[laughter]",
    "[breath]",
    "[lipsmack]",
    "[throatclearing]",
    "[sigh]",
    "[sniff]",
    "[cough]",
)
r"""Vocal-event tags of the verbatim transcription convention."""

VOCAL_EVENT_TAGS: tuple[str, ...] = FILLER_TAGS + SOUND_TAGS
r"""Every tag that stays a single atomic unit in the CTC vocabulary."""

CUTOFF_MARKER = "*"
r"""Marker appended to an interrupted word, as in `th*`."""

TAG_ALIASES: dict[str, str] = {
    "er": "uh",
    "erm": "um",
    "uhm": "um",
    "laugh": "laughter",
    "laughs": "laughter",
    "laughing": "laughter",
    "chuckle": "laughter",
    "breathing": "breath",
    "inhale": "breath",
    "exhale": "breath",
    "smack": "lipsmack",
    "throatclear": "throatclearing",
    "coughing": "cough",
}
r"""Spellings folded onto a canonical vocal-event tag."""

NON_LEXICAL_TAGS: frozenset[str] = frozenset(
    {
        "inaudible",
        "unintelligible",
        "crosstalk",
        "crosstalking",
        "foreign",
        "foreignlanguage",
        "silence",
        "noise",
        "background",
        "music",
        "applause",
        "static",
        "redacted",
    }
)
r"""Bracketed transcriber annotations that mark the absence of intelligible words."""

_SENTINEL = "\x00"
_PLACEHOLDER_RE = re.compile(rf"{_SENTINEL}([a-z]+){_SENTINEL}")
_TAG_RE = re.compile(r"\[[^\[\]]*\]")
_TIMECODE_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_CUTOFF_RE = re.compile(r"(?<=[a-zA-Z0-9])[-–—](?=\s|$)")
_TAG_KEY_RE = re.compile(r"[^a-z0-9]+")

_UNICODE_PUNCTUATION = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "′": "'",
    "´": "'",
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "…": " ",
    " ": " ",
}
_UNICODE_TABLE = str.maketrans(_UNICODE_PUNCTUATION)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizationConfig:
    r"""Rules turning a raw transcript into the canonical training form.

    The canonical form is lower case and free of punctuation, and keeps vocal-event tags as
    single whitespace-delimited units so that the CTC vocabulary can hold one symbol per
    tag.

    Args:
        lowercase (`bool`, *optional*, defaults to `True`):
            Lower case the whole transcript.
        event_tags (`tuple` of `str`, *optional*, defaults to `VOCAL_EVENT_TAGS`):
            Bracketed tags that survive normalization as atomic units.
        keep_event_tags (`bool`, *optional*, defaults to `True`):
            Keep the tags listed in `event_tags`. Set to `False` for text compared against
            a content-only reference.
        drop_non_lexical_tags (`bool`, *optional*, defaults to `True`):
            Delete bracketed transcriber annotations such as `[inaudible]` or
            `[crosstalk]`, which stand for words that were never recovered.
        unwrap_unknown_tags (`bool`, *optional*, defaults to `True`):
            Keep the words of an unrecognised bracketed group. The MSP-Podcast transcripts
            use one to mark an uncertain passage, `[hein courts council 00:04:09]`; the
            embedded time code is always removed.
        keep_cutoff_marker (`bool`, *optional*, defaults to `True`):
            Keep the `*` marking an interrupted word. A trailing hyphen, the marker used by
            the MSP-Podcast transcripts, is rewritten to `*` beforehand.
        keep_digits (`bool`, *optional*, defaults to `True`):
            Keep digits. Verbatim transcripts spell numbers out, so digits reach the
            canonical form only through intended text.
        ascii_only (`bool`, *optional*, defaults to `True`):
            Strip combining marks and drop the remaining non-ASCII letters.
        apostrophe (`str`, *optional*, defaults to `"'"`):
            The one character kept inside contractions.
    """

    lowercase: bool = True
    event_tags: tuple[str, ...] = VOCAL_EVENT_TAGS
    keep_event_tags: bool = True
    drop_non_lexical_tags: bool = True
    unwrap_unknown_tags: bool = True
    keep_cutoff_marker: bool = True
    keep_digits: bool = True
    ascii_only: bool = True
    apostrophe: str = "'"


DEFAULT_NORMALIZATION = NormalizationConfig()
r"""Normalization of the verbatim target: tags and cutoff markers are kept."""

CONTENT_NORMALIZATION = NormalizationConfig(keep_event_tags=False, keep_cutoff_marker=False)
r"""Normalization keeping lexical content only, used to compare two transcripts."""


def _tag_key(text: str) -> str:
    return _TAG_KEY_RE.sub("", text.lower())


@lru_cache(maxsize=16)
def _event_tag_map(event_tags: tuple[str, ...]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for tag in event_tags:
        key = _tag_key(tag)
        mapping[key] = f"[{key}]"
    for alias, target in TAG_ALIASES.items():
        target_key = _tag_key(target)
        if target_key in mapping:
            mapping.setdefault(_tag_key(alias), mapping[target_key])
    return mapping


def _encode_placeholder(index: int) -> str:
    # Placeholders are alphabetic so that they survive the character filter below whatever
    # the configuration does with digits.
    letters: list[str] = []
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters.append(string.ascii_lowercase[remainder])
    return f"{_SENTINEL}{''.join(reversed(letters))}{_SENTINEL}"


def _decode_placeholder(letters: str) -> int:
    value = 0
    for char in letters:
        value = value * 26 + string.ascii_lowercase.index(char) + 1
    return value - 1


def normalize_transcript(
    text: str, config: NormalizationConfig = DEFAULT_NORMALIZATION
) -> str:
    r"""Normalize a transcript into the canonical training form.

    Args:
        text (`str`):
            Raw transcript, either a model output or a human reference.
        config (`NormalizationConfig`, *optional*):
            Rules to apply.

    Returns:
        `str`: Lower case, punctuation-free text whose vocal-event tags are single
        whitespace-delimited units, for example `[uh] so we we need to reschedule`.

    Example:

    ```python
    >>> from ser_project.datasets.pseudo_label import normalize_transcript
    >>> normalize_transcript("[UM] So we, we need to th- think about it. [laughter]")
    '[um] so we we need to th* think about it [laughter]'
    ```
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text).translate(_UNICODE_TABLE)
    if config.lowercase:
        text = text.lower()
    if config.ascii_only:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(char for char in text if not unicodedata.combining(char))

    tag_map = _event_tag_map(tuple(config.event_tags))
    placeholders: list[str] = []

    def _substitute_tag(match: re.Match[str]) -> str:
        content = _TIMECODE_RE.sub(" ", match.group(0)[1:-1])
        key = _tag_key(content)
        canonical = tag_map.get(key)
        if canonical is not None:
            if not config.keep_event_tags:
                return " "
            placeholders.append(canonical)
            return f" {_encode_placeholder(len(placeholders) - 1)} "
        if key in NON_LEXICAL_TAGS and config.drop_non_lexical_tags:
            return " "
        if config.unwrap_unknown_tags:
            return f" {content} "
        return " "

    text = _TAG_RE.sub(_substitute_tag, text)
    text = _TRAILING_CUTOFF_RE.sub(CUTOFF_MARKER, text)
    text = text.replace("-", " ")

    allowed = set(string.ascii_lowercase) | {" ", _SENTINEL, config.apostrophe}
    if not config.lowercase:
        allowed |= set(string.ascii_uppercase)
    if config.keep_digits:
        allowed |= set(string.digits)
    if config.keep_cutoff_marker:
        allowed.add(CUTOFF_MARKER)
    text = "".join(char if char in allowed else " " for char in text)

    tokens = [token.strip(config.apostrophe) for token in text.split()]
    text = " ".join(token for token in tokens if token)

    if placeholders:
        text = _PLACEHOLDER_RE.sub(
            lambda match: placeholders[_decode_placeholder(match.group(1))], text
        )
        text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize_transcript(text: str) -> list[str]:
    r"""Split a canonical transcript into words, a vocal-event tag counting as one word.

    Args:
        text (`str`):
            Transcript in canonical form, as returned by [`normalize_transcript`].

    Returns:
        `list` of `str`: Whitespace-delimited units.
    """
    return text.split()


def split_into_ctc_units(
    text: str, event_tags: Sequence[str] = VOCAL_EVENT_TAGS
) -> list[str]:
    r"""Split a canonical transcript into the units of a character CTC vocabulary.

    Ordinary words are exploded into single characters, vocal-event tags stay atomic, and
    consecutive words are separated by one `" "` unit.

    Args:
        text (`str`):
            Transcript in canonical form, as returned by [`normalize_transcript`].
        event_tags (`Sequence` of `str`, *optional*, defaults to `VOCAL_EVENT_TAGS`):
            Tags that must not be split into characters.

    Returns:
        `list` of `str`: Vocabulary units in transcript order.

    Example:

    ```python
    >>> from ser_project.datasets.pseudo_label import split_into_ctc_units
    >>> split_into_ctc_units("[uh] ok")
    ['[uh]', ' ', 'o', 'k']
    ```
    """
    tags = set(event_tags)
    units: list[str] = []
    for index, token in enumerate(text.split()):
        if index:
            units.append(" ")
        if token in tags:
            units.append(token)
        else:
            units.extend(token)
    return units


# ---------------------------------------------------------------------------
# Verbatim and intended consistency
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsequenceCheck:
    r"""Outcome of comparing an intended transcript with its verbatim counterpart.

    Args:
        is_subsequence (`bool`):
            Whether every intended word occurs in the verbatim transcript, in order.
        coverage (`float`):
            Share of intended words covered by an order-preserving match. `1.0` when
            `is_subsequence` holds and for an empty intended transcript.
        matched (`int`):
            Number of intended words that were matched.
        intended_length (`int`):
            Number of intended words.
        verbatim_length (`int`):
            Number of verbatim words, tags excluded when they were ignored.
        missing_words (`tuple` of `str`):
            Unmatched intended words, truncated to the first few.
        passed (`bool`):
            Whether `coverage` reached the requested threshold.
    """

    is_subsequence: bool
    coverage: float
    matched: int
    intended_length: int
    verbatim_length: int
    missing_words: tuple[str, ...] = ()
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        r"""Return a JSON-serializable view of the check."""
        return {
            "is_subsequence": self.is_subsequence,
            "coverage": self.coverage,
            "matched": self.matched,
            "intended_length": self.intended_length,
            "verbatim_length": self.verbatim_length,
            "missing_words": list(self.missing_words),
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SubsequenceCheck":
        r"""Rebuild a check from the mapping produced by `to_dict`."""
        return cls(
            is_subsequence=bool(payload["is_subsequence"]),
            coverage=float(payload["coverage"]),
            matched=int(payload["matched"]),
            intended_length=int(payload["intended_length"]),
            verbatim_length=int(payload["verbatim_length"]),
            missing_words=tuple(payload.get("missing_words", ())),
            passed=bool(payload.get("passed", True)),
        )


def _is_subsequence(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    iterator = iter(haystack)
    return all(word in iterator for word in needle)


def check_intended_subsequence(
    verbatim: str,
    intended: str,
    *,
    min_coverage: float = 0.85,
    ignore_event_tags: bool = True,
    normalizer: Callable[[str], str] | None = None,
    max_missing_words: int = 16,
) -> SubsequenceCheck:
    r"""Check that the intended transcript is a subsequence of the verbatim transcript.

    Intended mode only removes material from what was said, so its words must reappear in
    order inside the verbatim transcript. Number and date formatting is the one systematic
    exception, `march third at nine thirty` against `March 3 at 9:30`, which is why the
    check reports a coverage ratio and flags rows below `min_coverage` instead of demanding
    an exact subsequence.

    Args:
        verbatim (`str`):
            Verbatim transcript, raw or canonical.
        intended (`str`):
            Intended transcript, raw or canonical.
        min_coverage (`float`, *optional*, defaults to `0.85`):
            Coverage below which the row is flagged.
        ignore_event_tags (`bool`, *optional*, defaults to `True`):
            Drop vocal-event tags from the verbatim side before comparing.
        normalizer (`Callable`, *optional*):
            Normalizer applied to both sides. Defaults to [`normalize_transcript`] with
            content-only settings.
        max_missing_words (`int`, *optional*, defaults to 16):
            Upper bound on the number of unmatched words kept in the result.

    Returns:
        `SubsequenceCheck`: The comparison outcome.
    """
    if normalizer is None:
        config = replace(CONTENT_NORMALIZATION, keep_event_tags=not ignore_event_tags)

        def normalizer(value: str) -> str:
            return normalize_transcript(value, config)

    verbatim_words = tokenize_transcript(normalizer(verbatim))
    intended_words = tokenize_transcript(normalizer(intended))
    if ignore_event_tags:
        tags = set(VOCAL_EVENT_TAGS)
        verbatim_words = [word for word in verbatim_words if word not in tags]

    if not intended_words:
        return SubsequenceCheck(
            is_subsequence=True,
            coverage=1.0,
            matched=0,
            intended_length=0,
            verbatim_length=len(verbatim_words),
            passed=True,
        )

    if _is_subsequence(intended_words, verbatim_words):
        return SubsequenceCheck(
            is_subsequence=True,
            coverage=1.0,
            matched=len(intended_words),
            intended_length=len(intended_words),
            verbatim_length=len(verbatim_words),
            passed=True,
        )

    matcher = difflib.SequenceMatcher(a=verbatim_words, b=intended_words, autojunk=False)
    matched: set[int] = set()
    for block in matcher.get_matching_blocks():
        matched.update(range(block.b, block.b + block.size))
    missing = [word for index, word in enumerate(intended_words) if index not in matched]
    coverage = len(matched) / len(intended_words)
    return SubsequenceCheck(
        is_subsequence=False,
        coverage=coverage,
        matched=len(matched),
        intended_length=len(intended_words),
        verbatim_length=len(verbatim_words),
        missing_words=tuple(missing[:max_missing_words]),
        passed=coverage >= min_coverage,
    )


# ---------------------------------------------------------------------------
# Word error rate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WordErrorRate:
    r"""Word-level edit statistics of one or more scored utterances.

    Args:
        hits (`int`):
            Correctly recognised reference words.
        substitutions (`int`):
            Reference words replaced by a different hypothesis word.
        deletions (`int`):
            Reference words missing from the hypothesis.
        insertions (`int`):
            Hypothesis words with no reference counterpart.
    """

    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def reference_length(self) -> int:
        r"""Number of reference words."""
        return self.hits + self.substitutions + self.deletions

    @property
    def errors(self) -> int:
        r"""Total number of edit operations."""
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        r"""Word error rate, `nan` when the reference is empty."""
        if self.reference_length == 0:
            return float("nan")
        return self.errors / self.reference_length

    def __add__(self, other: "WordErrorRate") -> "WordErrorRate":
        return WordErrorRate(
            hits=self.hits + other.hits,
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
        )

    def to_dict(self) -> dict[str, Any]:
        r"""Return a JSON-serializable view of the statistics."""
        return {
            "hits": self.hits,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_length": self.reference_length,
            "wer": self.wer,
        }


def word_error_rate(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> WordErrorRate:
    r"""Score one hypothesis against one reference with a Levenshtein alignment.

    Args:
        reference (`Sequence` of `str`):
            Reference words.
        hypothesis (`Sequence` of `str`):
            Hypothesis words.

    Returns:
        `WordErrorRate`: Hit, substitution, deletion and insertion counts.
    """
    reference = list(reference)
    hypothesis = list(hypothesis)
    if not reference:
        return WordErrorRate(insertions=len(hypothesis))
    if not hypothesis:
        return WordErrorRate(deletions=len(reference))

    # Cells are (cost, hits, substitutions, deletions, insertions); the leading cost keeps
    # `min` on the plain Levenshtein recursion while the operation counts ride along.
    previous: list[tuple[int, int, int, int, int]] = [
        (column, 0, 0, 0, column) for column in range(len(hypothesis) + 1)
    ]
    for row in range(1, len(reference) + 1):
        current: list[tuple[int, int, int, int, int]] = [(row, 0, 0, row, 0)]
        for column in range(1, len(hypothesis) + 1):
            cost, hits, subs, dels, ins = previous[column - 1]
            if reference[row - 1] == hypothesis[column - 1]:
                diagonal = (cost, hits + 1, subs, dels, ins)
            else:
                diagonal = (cost + 1, hits, subs + 1, dels, ins)
            cost, hits, subs, dels, ins = previous[column]
            deletion = (cost + 1, hits, subs, dels + 1, ins)
            cost, hits, subs, dels, ins = current[column - 1]
            insertion = (cost + 1, hits, subs, dels, ins + 1)
            current.append(min(diagonal, deletion, insertion, key=lambda cell: cell[0]))
        previous = current

    _, hits, subs, dels, ins = previous[-1]
    return WordErrorRate(hits=hits, substitutions=subs, deletions=dels, insertions=ins)


def english_wer_normalizer(
    spelling_mapping: Mapping[str, str] | None = None,
) -> Callable[[str], str]:
    r"""Return the text normalizer used when scoring pseudo-labels.

    Whisper's English normalizer resolves the number and date formatting differences that
    would otherwise dominate the error rate between a written reference and an intended
    transcript, `fifteen hundred` against `1500`. It is applied on top of
    [`normalize_transcript`], which has already resolved the bracketed annotations of both
    conventions.

    Args:
        spelling_mapping (`Mapping`, *optional*):
            British to American spelling map handed to the Whisper normalizer.

    Returns:
        `Callable`: A text normalizer.
    """
    try:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    except ImportError:
        return lambda text: normalize_transcript(text, CONTENT_NORMALIZATION)

    whisper_normalizer = EnglishTextNormalizer(dict(spelling_mapping or {}))
    return lambda text: whisper_normalizer(normalize_transcript(text, CONTENT_NORMALIZATION))


# ---------------------------------------------------------------------------
# Corpus access
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversationPart:
    r"""One conversation part and the audio it is read from.

    Args:
        part_id (`str`):
            Part identifier such as `"MSP-Conversation_0002_1"`.
        conversation_id (`str`):
            Parent conversation identifier.
        partition (`str` or `None`):
            Partition of the conversation, `None` when it is absent from the release.
        start (`float`), end (`float`):
            Part bounds in seconds relative to the conversation, the same reference as the
            diarization files.
        audio_path (`pathlib.Path`):
            Audio file the waveform is read from.
        audio_offset (`float`):
            Where the part starts inside `audio_path`: zero when the split part audio
            exists, `start` when the conversation audio is read instead.
    """

    part_id: str
    conversation_id: str
    partition: str | None
    start: float
    end: float
    audio_path: Path
    audio_offset: float

    @property
    def duration(self) -> float:
        r"""Part duration in seconds."""
        return self.end - self.start


def load_conversation_parts(
    paths: Any = None,
    *,
    partitions: Sequence[str] | None = None,
    part_ids: Sequence[str] | None = None,
) -> list[ConversationPart]:
    r"""Enumerate the conversation parts of a release.

    Args:
        paths (`str`, `os.PathLike` or `CorpusPaths`, *optional*):
            Release directory or resolved paths.
        partitions (`Sequence` of `str`, *optional*):
            Keep only parts of these partitions.
        part_ids (`Sequence` of `str`, *optional*):
            Keep only these parts, returned in the given order.

    Returns:
        `list` of `ConversationPart`: Parts sorted by identifier, or in `part_ids` order.
    """
    corpus = CorpusPaths.from_root(paths)
    corpus.require_root()
    part_times = read_part_times(corpus)
    partition_of = read_partitions(corpus)
    wanted = set(partitions) if partitions is not None else None

    parts: dict[str, ConversationPart] = {}
    for part_id, (start, end) in part_times.items():
        conversation_id = conversation_of(part_id)
        partition = partition_of.get(conversation_id)
        if wanted is not None and partition not in wanted:
            continue
        part_audio = corpus.part_audio(part_id)
        if part_audio.exists():
            audio_path, offset = part_audio, 0.0
        else:
            audio_path, offset = corpus.conversation_audio(conversation_id), start
        parts[part_id] = ConversationPart(
            part_id=part_id,
            conversation_id=conversation_id,
            partition=partition,
            start=start,
            end=end,
            audio_path=audio_path,
            audio_offset=offset,
        )

    if part_ids is None:
        return [parts[part_id] for part_id in sorted(parts)]
    missing = [part_id for part_id in part_ids if part_id not in parts]
    if missing:
        raise KeyError(f"Unknown conversation parts: {missing[:5]}")
    return [parts[part_id] for part_id in part_ids]


def load_part_audio(part: ConversationPart) -> tuple[np.ndarray, int]:
    r"""Read the waveform of a conversation part.

    Args:
        part (`ConversationPart`):
            Part to read.

    Returns:
        `tuple`: The mono waveform as `float32` and its sampling rate. The audio is
        returned at its native rate; nothing is resampled here.

    Raises:
        CorpusLayoutError: The audio file is absent, which is the state of a release whose
            `Audio` archive is still extracting.
    """
    import soundfile as sf

    if not part.audio_path.exists():
        raise CorpusLayoutError(
            f"No audio for part {part.part_id!r}: {part.audio_path} does not exist. "
            "The 'Audio' archive may still be extracting.",
            part.audio_path.parent.parent,
        )

    with sf.SoundFile(part.audio_path) as handle:
        sampling_rate = int(handle.samplerate)
        handle.seek(max(0, int(round(part.audio_offset * sampling_rate))))
        audio = handle.read(
            frames=int(round(part.duration * sampling_rate)),
            dtype="float32",
            always_2d=False,
        )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), sampling_rate


# ---------------------------------------------------------------------------
# Overlapped speech and transcript supervision masking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpanSpeechStats:
    r"""How much of a span carries speech and how much of it carries simultaneous speech.

    Args:
        duration (`float`):
            Span duration in seconds.
        speech_seconds (`float`):
            Seconds covered by at least one speaker.
        overlap_seconds (`float`):
            Seconds covered by two or more distinct speakers.
        speech_ratio (`float`):
            `speech_seconds` divided by `duration`.
        overlap_ratio (`float`):
            `overlap_seconds` divided by `speech_seconds`, `0.0` for a silent span. The
            ratio is taken against speech rather than against the span so that a mostly
            silent span whose speech is entirely overlapped still counts as unreliable.
    """

    duration: float
    speech_seconds: float
    overlap_seconds: float
    speech_ratio: float
    overlap_ratio: float

    def to_dict(self) -> dict[str, Any]:
        r"""Return a JSON-serializable view of the statistics."""
        return {
            "duration": self.duration,
            "speech_seconds": self.speech_seconds,
            "overlap_seconds": self.overlap_seconds,
            "speech_ratio": self.speech_ratio,
            "overlap_ratio": self.overlap_ratio,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpanSpeechStats":
        r"""Rebuild statistics from the mapping produced by `to_dict`."""
        return cls(
            duration=float(payload["duration"]),
            speech_seconds=float(payload["speech_seconds"]),
            overlap_seconds=float(payload["overlap_seconds"]),
            speech_ratio=float(payload["speech_ratio"]),
            overlap_ratio=float(payload["overlap_ratio"]),
        )


def _merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def span_speech_stats(
    segments: Iterable[DiarizationSegment], start: float, end: float
) -> SpanSpeechStats:
    r"""Measure speech and simultaneous speech inside a time span.

    The turns of one speaker are merged before the sweep, so two adjacent turns of the same
    person never count as overlap.

    Args:
        segments (`Iterable` of `DiarizationSegment`):
            Diarization turns, in the same reference as `start` and `end`.
        start (`float`), end (`float`):
            Span bounds in seconds.

    Returns:
        `SpanSpeechStats`: Speech and overlap statistics of the span.
    """
    duration = max(0.0, float(end) - float(start))
    per_speaker: dict[str, list[tuple[float, float]]] = {}
    for segment in segments:
        lower = max(segment.start, start)
        upper = min(segment.end, end)
        if upper > lower:
            per_speaker.setdefault(segment.speaker, []).append((lower, upper))

    events: list[tuple[float, int]] = []
    for intervals in per_speaker.values():
        for lower, upper in _merge_intervals(intervals):
            events.append((lower, 1))
            events.append((upper, -1))
    events.sort()

    speech_seconds = 0.0
    overlap_seconds = 0.0
    active = 0
    previous = start
    for position, delta in events:
        if position > previous:
            width = position - previous
            if active >= 1:
                speech_seconds += width
            if active >= 2:
                overlap_seconds += width
            previous = position
        active += delta

    return SpanSpeechStats(
        duration=duration,
        speech_seconds=speech_seconds,
        overlap_seconds=overlap_seconds,
        speech_ratio=speech_seconds / duration if duration > 0 else 0.0,
        overlap_ratio=overlap_seconds / speech_seconds if speech_seconds > 0 else 0.0,
    )


@dataclass(frozen=True)
class SupervisionPolicy:
    r"""Conditions under which a sample carries usable transcript supervision.

    Args:
        max_overlap_ratio (`float`, *optional*, defaults to 0.5):
            Largest share of the speech time that may be simultaneous speech. A sample
            dominated by overlapped speech is transcribed unreliably by a single-stream
            recogniser, so its CTC loss is switched off.
        min_speech_ratio (`float`, *optional*, defaults to 0.05):
            Smallest share of the span that has to carry speech.
        min_words (`int`, *optional*, defaults to 1):
            Smallest number of transcript words.
        require_subsequence_check (`bool`, *optional*, defaults to `True`):
            Require that the verbatim and intended transcripts passed the consistency
            check of the part the sample is taken from.
    """

    max_overlap_ratio: float = 0.5
    min_speech_ratio: float = 0.05
    min_words: int = 1
    require_subsequence_check: bool = True


DEFAULT_SUPERVISION_POLICY = SupervisionPolicy()


# ---------------------------------------------------------------------------
# Pseudo-label records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Word:
    r"""One transcribed word with its audio span.

    Args:
        word (`str`):
            Word as emitted by the model, vocal-event tags included.
        start (`float`), end (`float`):
            Word bounds in seconds relative to the conversation part.
    """

    word: str
    start: float
    end: float

    @property
    def midpoint(self) -> float:
        r"""Centre of the word in seconds."""
        return 0.5 * (self.start + self.end)


@dataclass(frozen=True)
class ModeTranscript:
    r"""The transcript of one conversation part in one transcription mode.

    Args:
        mode (`str`):
            `"verbatim"` or `"intended"`.
        text (`str`):
            Raw model output.
        normalized_text (`str`):
            Canonical form produced by [`normalize_transcript`].
        words (`tuple` of `Word`):
            Word-level timestamps, empty when they were not requested.
    """

    mode: str
    text: str
    normalized_text: str = ""
    words: tuple[Word, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        r"""Return a JSON-serializable view of the transcript."""
        return {
            "mode": self.mode,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "words": [[word.word, word.start, word.end] for word in self.words],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModeTranscript":
        r"""Rebuild a transcript from the mapping produced by `to_dict`."""
        return cls(
            mode=str(payload["mode"]),
            text=str(payload.get("text", "")),
            normalized_text=str(payload.get("normalized_text", "")),
            words=tuple(
                Word(word=str(item[0]), start=float(item[1]), end=float(item[2]))
                for item in payload.get("words", ())
            ),
        )


@dataclass(frozen=True)
class PseudoLabelRecord:
    r"""Everything the pipeline produced for one conversation part.

    Args:
        part_id (`str`):
            Conversation part identifier.
        conversation_id (`str`):
            Parent conversation identifier.
        partition (`str` or `None`):
            Partition of the conversation.
        start (`float`):
            Part start in seconds relative to the conversation.
        duration (`float`):
            Part duration in seconds. Word timestamps are relative to the part, so they run
            from `0.0` to `duration`.
        model_id (`str`):
            Model that produced the transcripts.
        language (`str`):
            Language code passed to the model.
        transcripts (`dict`):
            One `ModeTranscript` per transcription mode.
        subsequence (`SubsequenceCheck` or `None`):
            Consistency of the intended transcript with the verbatim transcript.
        speech (`SpanSpeechStats` or `None`):
            Speech and overlap statistics of the whole part.
    """

    part_id: str
    conversation_id: str
    partition: str | None
    start: float
    duration: float
    model_id: str
    language: str
    transcripts: dict[str, ModeTranscript] = field(default_factory=dict)
    subsequence: SubsequenceCheck | None = None
    speech: SpanSpeechStats | None = None

    @property
    def end(self) -> float:
        r"""Part end in seconds relative to the conversation."""
        return self.start + self.duration

    @property
    def verbatim(self) -> ModeTranscript | None:
        r"""Verbatim transcript, `None` when that mode was not run."""
        return self.transcripts.get(VERBATIM_MODE)

    @property
    def intended(self) -> ModeTranscript | None:
        r"""Intended transcript, `None` when that mode was not run."""
        return self.transcripts.get(INTENDED_MODE)

    def to_dict(self) -> dict[str, Any]:
        r"""Return a JSON-serializable view of the record."""
        return {
            "part_id": self.part_id,
            "conversation_id": self.conversation_id,
            "partition": self.partition,
            "start": self.start,
            "duration": self.duration,
            "model_id": self.model_id,
            "language": self.language,
            "transcripts": {
                mode: transcript.to_dict() for mode, transcript in self.transcripts.items()
            },
            "subsequence": None if self.subsequence is None else self.subsequence.to_dict(),
            "speech": None if self.speech is None else self.speech.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PseudoLabelRecord":
        r"""Rebuild a record from the mapping produced by `to_dict`."""
        subsequence = payload.get("subsequence")
        speech = payload.get("speech")
        return cls(
            part_id=str(payload["part_id"]),
            conversation_id=str(payload["conversation_id"]),
            partition=payload.get("partition"),
            start=float(payload.get("start", 0.0)),
            duration=float(payload.get("duration", 0.0)),
            model_id=str(payload.get("model_id", "")),
            language=str(payload.get("language", "")),
            transcripts={
                mode: ModeTranscript.from_dict(value)
                for mode, value in payload.get("transcripts", {}).items()
            },
            subsequence=None if not subsequence else SubsequenceCheck.from_dict(subsequence),
            speech=None if not speech else SpanSpeechStats.from_dict(speech),
        )


def words_in_span(
    words: Sequence[Word],
    start: float,
    end: float,
    *,
    policy: str = "midpoint",
    min_overlap: float = 0.5,
) -> list[Word]:
    r"""Select the words belonging to a time span of a conversation part.

    Args:
        words (`Sequence` of `Word`):
            Timestamped words of the part.
        start (`float`), end (`float`):
            Span bounds in seconds relative to the part.
        policy (`str`, *optional*, defaults to `"midpoint"`):
            `"midpoint"` keeps a word whose centre falls inside the span, `"overlap"` keeps
            a word of which at least `min_overlap` lies inside it.
        min_overlap (`float`, *optional*, defaults to 0.5):
            Overlap share required by the `"overlap"` policy.

    Returns:
        `list` of `Word`: Selected words in transcript order.
    """
    if policy not in {"midpoint", "overlap"}:
        raise ValueError(f"Unknown word selection policy {policy!r}; expected 'midpoint' or 'overlap'.")

    selected: list[Word] = []
    for word in words:
        if policy == "midpoint":
            if start <= word.midpoint < end:
                selected.append(word)
            continue
        covered = min(word.end, end) - max(word.start, start)
        if covered <= 0:
            continue
        width = word.end - word.start
        if width <= 0 or covered / width >= min_overlap:
            selected.append(word)
    return selected


def transcript_for_span(
    record: PseudoLabelRecord,
    start: float,
    end: float,
    *,
    mode: str = VERBATIM_MODE,
    config: NormalizationConfig = DEFAULT_NORMALIZATION,
    policy: str = "midpoint",
) -> str:
    r"""Return the canonical transcript of a crop of a conversation part.

    Args:
        record (`PseudoLabelRecord`):
            Record of the part the crop is taken from.
        start (`float`), end (`float`):
            Crop bounds in seconds relative to the part.
        mode (`str`, *optional*, defaults to `"verbatim"`):
            Transcription mode to read.
        config (`NormalizationConfig`, *optional*):
            Normalization rules.
        policy (`str`, *optional*, defaults to `"midpoint"`):
            Word selection policy, see [`words_in_span`].

    Returns:
        `str`: Canonical transcript of the crop; empty when the mode carries no word
        timestamps or when no word falls inside the crop.
    """
    transcript = record.transcripts.get(mode)
    if transcript is None or not transcript.words:
        return ""
    selected = words_in_span(transcript.words, start, end, policy=policy)
    return normalize_transcript(" ".join(word.word for word in selected), config)


def transcript_supervision_mask(
    record: PseudoLabelRecord,
    *,
    start: float | None = None,
    end: float | None = None,
    segments: Sequence[DiarizationSegment] | None = None,
    mode: str = VERBATIM_MODE,
    policy: SupervisionPolicy = DEFAULT_SUPERVISION_POLICY,
    config: NormalizationConfig = DEFAULT_NORMALIZATION,
) -> bool:
    r"""Decide whether a sample carries usable transcript supervision.

    A sample dominated by overlapped speech is excluded because a single-stream recogniser
    cannot transcribe simultaneous speakers reliably, and so is a sample whose part failed
    the verbatim and intended consistency check or whose crop holds no words. The result is
    the `ctc_loss_mask` entry of the sample.

    Args:
        record (`PseudoLabelRecord`):
            Record of the conversation part the sample is taken from.
        start (`float`, *optional*), end (`float`, *optional*):
            Crop bounds in seconds relative to the part. Both default to the whole part.
        segments (`Sequence` of `DiarizationSegment`, *optional*):
            Diarization turns relative to the part, as produced by
            [`~ser_project.datasets.msp_conversation.segments_within`]. Required to
            recompute the overlap of a crop; without them the part-level statistics stored
            on the record are used.
        mode (`str`, *optional*, defaults to `"verbatim"`):
            Transcription mode the supervision is drawn from.
        policy (`SupervisionPolicy`, *optional*):
            Thresholds to apply.
        config (`NormalizationConfig`, *optional*):
            Normalization rules used to count the transcript words.

    Returns:
        `bool`: `True` when the CTC loss should be applied to the sample.
    """
    transcript = record.transcripts.get(mode)
    if transcript is None:
        return False
    if (
        policy.require_subsequence_check
        and record.subsequence is not None
        and not record.subsequence.passed
    ):
        return False

    crop_start = 0.0 if start is None else start
    crop_end = record.duration if end is None else end

    if segments is not None:
        stats = span_speech_stats(segments, crop_start, crop_end)
    else:
        stats = record.speech
    if stats is not None:
        if stats.overlap_ratio > policy.max_overlap_ratio:
            return False
        if stats.speech_ratio < policy.min_speech_ratio:
            return False

    if start is None and end is None and transcript.normalized_text:
        text = transcript.normalized_text
    else:
        text = transcript_for_span(record, crop_start, crop_end, mode=mode, config=config)
    return len(tokenize_transcript(text)) >= policy.min_words


# ---------------------------------------------------------------------------
# Transcription backend
# ---------------------------------------------------------------------------


@runtime_checkable
class Transcriber(Protocol):
    r"""The interface the pipeline needs from a transcription backend."""

    def transcribe(
        self, audio: np.ndarray, sampling_rate: int, modes: Sequence[str]
    ) -> dict[str, ModeTranscript]:
        r"""Transcribe one conversation part in every requested mode.

        Args:
            audio (`np.ndarray`):
                Mono waveform of the part.
            sampling_rate (`int`):
                Sampling rate of `audio`.
            modes (`Sequence` of `str`):
                Transcription modes to produce.

        Returns:
            `dict`: One `ModeTranscript` per requested mode.
        """
        ...


@dataclass(frozen=True)
class PseudoLabelConfig:
    r"""Settings of one pseudo-labelling run.

    Args:
        model_id (`str`, *optional*, defaults to `"nyralabs/CrisperWhisper2.0_large"`):
            Model repository, or a key of `MODEL_IDS` such as `"turbo"`.
        backend (`str`, *optional*, defaults to `"auto"`):
            `"auto"`, `"ct2"` or `"transformers"`. Only the CTranslate2 runtime decodes
            both modes in a single batched pass.
        device (`str`, *optional*, defaults to `"auto"`):
            Device the model runs on.
        compute_type (`str`, *optional*, defaults to `"float16"`):
            Weight and activation precision of the backend.
        draft_model (`str` or `None`, *optional*):
            Draft model of the speculative decoder.
        speculative_decoding (`bool`, *optional*, defaults to `False`):
            Enable speculative decoding, which needs `draft_model`.
        language (`str`, *optional*, defaults to `"en"`):
            Language code passed to the model.
        modes (`tuple` of `str`, *optional*, defaults to `("verbatim", "intended")`):
            Transcription modes produced for every part.
        word_timestamps (`bool`, *optional*, defaults to `True`):
            Keep word-level timings, which every crop of a part depends on.
        longform_strategy (`str`, *optional*, defaults to `"continuation"`):
            Strategy stitching the 30 s decoder windows of a multi-minute part.
        chunk_duration (`float`, *optional*, defaults to 30.0):
            Length of one decoder window in seconds.
        stride (`float`, *optional*, defaults to 26.0):
            Advance between decoder windows in seconds.
        max_new_tokens (`int`, *optional*, defaults to 256):
            Token budget of one decoder window.
        hallucination_mitigation (`bool`, *optional*, defaults to `True`):
            Detect and repair the looping-repetition failure mode.
        batch_size (`int`, *optional*, defaults to 8):
            Number of parts transcribed between two flushes of the output file.
        min_subsequence_coverage (`float`, *optional*, defaults to 0.85):
            Coverage below which the consistency check flags a part.
        normalization (`NormalizationConfig`, *optional*):
            Rules producing the canonical transcript stored beside the raw one.
    """

    model_id: str = DEFAULT_MODEL_ID
    backend: str = "auto"
    device: str = "auto"
    compute_type: str = "float16"
    draft_model: str | None = None
    speculative_decoding: bool = False
    language: str = "en"
    modes: tuple[str, ...] = (VERBATIM_MODE, INTENDED_MODE)
    word_timestamps: bool = True
    longform_strategy: str = "continuation"
    chunk_duration: float = 30.0
    stride: float = 26.0
    max_new_tokens: int = 256
    hallucination_mitigation: bool = True
    batch_size: int = 8
    min_subsequence_coverage: float = 0.85
    normalization: NormalizationConfig = DEFAULT_NORMALIZATION

    @property
    def resolved_model_id(self) -> str:
        r"""Model repository, resolving the size shorthands of `MODEL_IDS`."""
        return MODEL_IDS.get(self.model_id, self.model_id)

    def to_dict(self) -> dict[str, Any]:
        r"""Return a JSON-serializable view of the configuration."""
        return {
            "model_id": self.resolved_model_id,
            "backend": self.backend,
            "device": self.device,
            "compute_type": self.compute_type,
            "draft_model": self.draft_model,
            "speculative_decoding": self.speculative_decoding,
            "language": self.language,
            "modes": list(self.modes),
            "word_timestamps": self.word_timestamps,
            "longform_strategy": self.longform_strategy,
            "chunk_duration": self.chunk_duration,
            "stride": self.stride,
            "max_new_tokens": self.max_new_tokens,
            "hallucination_mitigation": self.hallucination_mitigation,
            "batch_size": self.batch_size,
            "min_subsequence_coverage": self.min_subsequence_coverage,
            "normalization": {
                "lowercase": self.normalization.lowercase,
                "event_tags": list(self.normalization.event_tags),
                "keep_event_tags": self.normalization.keep_event_tags,
                "drop_non_lexical_tags": self.normalization.drop_non_lexical_tags,
                "unwrap_unknown_tags": self.normalization.unwrap_unknown_tags,
                "keep_cutoff_marker": self.normalization.keep_cutoff_marker,
                "keep_digits": self.normalization.keep_digits,
                "ascii_only": self.normalization.ascii_only,
            },
        }


class CrisperWhisperTranscriber:
    r"""CrisperWhisper 2.0 backend loading its checkpoint on first use.

    The model is instantiated inside [`CrisperWhisperTranscriber.transcribe`], so the
    module stays importable and testable on a machine that has no checkpoint. Both modes
    differ only by the decoder prompt prefix and share one encoder pass, which the
    CTranslate2 runtime exploits by decoding them as two rows of one batch; on a runtime
    without that entry point the modes are decoded one after another instead.

    Args:
        config (`PseudoLabelConfig`, *optional*):
            Settings of the run.
    """

    def __init__(self, config: PseudoLabelConfig | None = None) -> None:
        self.config = config or PseudoLabelConfig()
        self._model: Any | None = None
        self._supports_dual = True

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from crisperwhisper import CrisperWhisperModel
        except ImportError as error:
            raise ImportError(
                "The 'crisperwhisper' package is required to generate transcript "
                "pseudo-labels. Install it with `uv add 'crisperwhisper[ct2]'` for the "
                "CTranslate2 runtime, or `uv add 'crisperwhisper[transformers]'` for the "
                "pure torch runtime."
            ) from error

        config = self.config
        self._model = CrisperWhisperModel(
            config.resolved_model_id,
            backend=config.backend,
            compute_type=config.compute_type,
            device=config.device,
            draft_model=config.draft_model,
        )
        return self._model

    def _decode_kwargs(self) -> dict[str, Any]:
        config = self.config
        return {
            "language": config.language,
            "longform_strategy": config.longform_strategy,
            "chunk_duration": config.chunk_duration,
            "stride": config.stride,
            "max_new_tokens": config.max_new_tokens,
            "hallucination_mitigation": config.hallucination_mitigation,
            "word_timestamps": config.word_timestamps,
        }

    def transcribe(
        self, audio: np.ndarray, sampling_rate: int, modes: Sequence[str]
    ) -> dict[str, ModeTranscript]:
        r"""Transcribe one conversation part in every requested mode.

        Args:
            audio (`np.ndarray`):
                Mono waveform of the part.
            sampling_rate (`int`):
                Sampling rate of `audio`.
            modes (`Sequence` of `str`):
                Transcription modes to produce.

        Returns:
            `dict`: One `ModeTranscript` per requested mode.
        """
        model = self._load_model()
        modes = tuple(modes)
        kwargs = self._decode_kwargs()

        if len(modes) > 1 and self._supports_dual:
            try:
                results = model.transcribe_dual(audio, sr=sampling_rate, modes=modes, **kwargs)
            except (NotImplementedError, AttributeError):
                self._supports_dual = False
            else:
                return {
                    mode: self._to_transcript(mode, result)
                    for mode, result in zip(modes, results)
                }

        if self.config.speculative_decoding:
            kwargs["speculative_decoding"] = True
        return {
            mode: self._to_transcript(
                mode, model.transcribe(audio, sr=sampling_rate, mode=mode, **kwargs)
            )
            for mode in modes
        }

    def _to_transcript(self, mode: str, result: Any) -> ModeTranscript:
        text = getattr(result, "text", "") or ""
        words = tuple(
            Word(word=str(item.word).strip(), start=float(item.start), end=float(item.end))
            for item in (getattr(result, "words", None) or ())
            if str(item.word).strip()
        )
        return ModeTranscript(
            mode=mode,
            text=text.strip(),
            normalized_text=normalize_transcript(text, self.config.normalization),
            words=words,
        )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def default_output_dir(paths: Any = None, model_id: str = DEFAULT_MODEL_ID) -> Path:
    r"""Return the directory holding the pseudo-labels of one model.

    Args:
        paths (`str`, `os.PathLike` or `CorpusPaths`, *optional*):
            Release directory or resolved paths.
        model_id (`str`, *optional*):
            Model repository or size shorthand.

    Returns:
        `pathlib.Path`: `<release>/PseudoLabels/<model name>`.
    """
    corpus = CorpusPaths.from_root(paths, missing_ok=True)
    resolved = MODEL_IDS.get(model_id, model_id)
    return corpus.root / "PseudoLabels" / resolved.split("/")[-1]


def write_pseudo_labels(
    records: Iterable[PseudoLabelRecord], path: os.PathLike[str] | str, *, append: bool = False
) -> Path:
    r"""Serialize records as JSON lines.

    Args:
        records (`Iterable` of `PseudoLabelRecord`):
            Records to write.
        path (`os.PathLike` or `str`):
            Destination file.
        append (`bool`, *optional*, defaults to `False`):
            Append instead of truncating.

    Returns:
        `pathlib.Path`: The destination file.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a" if append else "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return destination


def iter_pseudo_labels(path: os.PathLike[str] | str) -> Iterator[PseudoLabelRecord]:
    r"""Iterate over the records of a JSON lines file.

    Args:
        path (`os.PathLike` or `str`):
            File written by [`write_pseudo_labels`].

    Yields:
        `PseudoLabelRecord`: One record per line, in file order.
    """
    source = require_path(path, "pseudo-label file")
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield PseudoLabelRecord.from_dict(json.loads(line))


def load_pseudo_labels(path: os.PathLike[str] | str) -> dict[str, PseudoLabelRecord]:
    r"""Read a pseudo-label file into a mapping keyed by conversation part.

    A part written more than once, which happens when an interrupted run is resumed, keeps
    its last occurrence.

    Args:
        path (`os.PathLike` or `str`):
            File written by [`write_pseudo_labels`].

    Returns:
        `dict`: Records keyed by part identifier.
    """
    return {record.part_id: record for record in iter_pseudo_labels(path)}


# ---------------------------------------------------------------------------
# Transcription pass
# ---------------------------------------------------------------------------


def build_record(
    part: ConversationPart,
    transcripts: Mapping[str, ModeTranscript],
    *,
    config: PseudoLabelConfig | None = None,
    segments: Sequence[DiarizationSegment] | None = None,
) -> PseudoLabelRecord:
    r"""Assemble the record of one conversation part.

    The canonical transcripts, the verbatim and intended consistency check and the overlap
    statistics the masking rule reads are all computed here.

    Args:
        part (`ConversationPart`):
            Part the transcripts belong to.
        transcripts (`Mapping`):
            One `ModeTranscript` per transcription mode.
        config (`PseudoLabelConfig`, *optional*):
            Settings of the run.
        segments (`Sequence` of `DiarizationSegment`, *optional*):
            Diarization turns relative to the part.

    Returns:
        `PseudoLabelRecord`: The assembled record.
    """
    config = config or PseudoLabelConfig()
    normalized = {
        mode: transcript
        if transcript.normalized_text
        else replace(
            transcript,
            normalized_text=normalize_transcript(transcript.text, config.normalization),
        )
        for mode, transcript in transcripts.items()
    }

    subsequence = None
    verbatim = normalized.get(VERBATIM_MODE)
    intended = normalized.get(INTENDED_MODE)
    if verbatim is not None and intended is not None:
        subsequence = check_intended_subsequence(
            verbatim.text,
            intended.text,
            min_coverage=config.min_subsequence_coverage,
        )

    return PseudoLabelRecord(
        part_id=part.part_id,
        conversation_id=part.conversation_id,
        partition=part.partition,
        start=part.start,
        duration=part.duration,
        model_id=config.resolved_model_id,
        language=config.language,
        transcripts=dict(normalized),
        subsequence=subsequence,
        speech=None if segments is None else span_speech_stats(segments, 0.0, part.duration),
    )


def transcribe_parts(
    parts: Sequence[ConversationPart],
    transcriber: Transcriber,
    *,
    config: PseudoLabelConfig | None = None,
    paths: Any = None,
) -> Iterator[PseudoLabelRecord]:
    r"""Transcribe conversation parts and yield their records.

    Args:
        parts (`Sequence` of `ConversationPart`):
            Parts to transcribe.
        transcriber (`Transcriber`):
            Backend producing the transcripts.
        config (`PseudoLabelConfig`, *optional*):
            Settings of the run.
        paths (`str`, `os.PathLike` or `CorpusPaths`, *optional*):
            Release the diarization is read from. Without it the overlap statistics stay
            unset and the masking rule falls back to the transcript conditions alone.

    Yields:
        `PseudoLabelRecord`: One record per part, in input order.
    """
    config = config or PseudoLabelConfig()
    corpus = None if paths is None else CorpusPaths.from_root(paths)
    cache: dict[str, list[DiarizationSegment] | None] = {}

    for part in parts:
        audio, sampling_rate = load_part_audio(part)
        transcripts = transcriber.transcribe(audio, sampling_rate, config.modes)

        segments: list[DiarizationSegment] | None = None
        if corpus is not None:
            if part.conversation_id not in cache:
                try:
                    cache[part.conversation_id] = load_diarization(part.conversation_id, corpus)
                except CorpusLayoutError:
                    cache[part.conversation_id] = None
            turns = cache[part.conversation_id]
            if turns is not None:
                segments = segments_within(turns, part.start, part.end)

        yield build_record(part, transcripts, config=config, segments=segments)


def run_pseudo_labelling(
    paths: Any = None,
    *,
    output_dir: os.PathLike[str] | str | None = None,
    config: PseudoLabelConfig | None = None,
    partitions: Sequence[str] | None = None,
    part_ids: Sequence[str] | None = None,
    transcriber: Transcriber | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    r"""Transcribe every conversation part and store the pseudo-labels next to the corpus.

    The pass is resumable: parts already present in the output file are skipped unless
    `overwrite` is set, and the file is flushed every `config.batch_size` parts, so an
    interrupted run loses at most one batch of transcription work.

    Args:
        paths (`str`, `os.PathLike` or `CorpusPaths`, *optional*):
            Release directory or resolved paths.
        output_dir (`os.PathLike` or `str`, *optional*):
            Destination directory, defaults to [`default_output_dir`].
        config (`PseudoLabelConfig`, *optional*):
            Settings of the run.
        partitions (`Sequence` of `str`, *optional*):
            Restrict the run to these partitions.
        part_ids (`Sequence` of `str`, *optional*):
            Restrict the run to these conversation parts.
        transcriber (`Transcriber`, *optional*):
            Backend to use. Defaults to a [`CrisperWhisperTranscriber`], whose checkpoint is
            loaded when the first part is transcribed.
        overwrite (`bool`, *optional*, defaults to `False`):
            Restart from an empty output file.
        progress (`bool`, *optional*, defaults to `True`):
            Display a progress bar when `tqdm` is installed.

    Returns:
        `pathlib.Path`: The JSON lines file holding the records.
    """
    corpus = CorpusPaths.from_root(paths)
    config = config or PseudoLabelConfig()
    destination = (
        Path(output_dir)
        if output_dir is not None
        else default_output_dir(corpus, config.model_id)
    )
    destination.mkdir(parents=True, exist_ok=True)
    if not os.access(destination, os.W_OK):
        raise PermissionError(
            f"Pseudo-labels cannot be written to '{destination}'. Point `output_dir` at a "
            "writable working copy of the release."
        )

    labels_path = destination / "transcripts.jsonl"
    if overwrite and labels_path.exists():
        labels_path.unlink()
    done = set(load_pseudo_labels(labels_path)) if labels_path.exists() else set()

    parts = [
        part
        for part in load_conversation_parts(corpus, partitions=partitions, part_ids=part_ids)
        if part.part_id not in done
    ]

    if transcriber is None:
        transcriber = CrisperWhisperTranscriber(config)

    records: Iterable[PseudoLabelRecord] = transcribe_parts(
        parts, transcriber, config=config, paths=corpus
    )
    if progress:
        try:
            from tqdm.auto import tqdm

            records = tqdm(records, total=len(parts), desc="Pseudo-labelling")
        except ImportError:
            pass

    written = 0
    with labels_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            written += 1
            if written % max(1, config.batch_size) == 0:
                handle.flush()

    manifest = {
        "corpus": str(corpus.root),
        "labels": labels_path.name,
        "config": config.to_dict(),
        "num_parts": len(done) + written,
        "num_parts_written": written,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return labels_path


# ---------------------------------------------------------------------------
# Quality control against the MSP-Podcast transcripts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PodcastSegmentSpan:
    r"""An MSP-Podcast speaking turn located inside a conversation part.

    Args:
        segment_id (`str`):
            MSP-Podcast segment identifier such as `"MSP-PODCAST_0002_0033"`.
        conversation_id (`str`):
            Conversation the turn falls into.
        part_id (`str`):
            Conversation part the turn falls into.
        start (`float`), end (`float`):
            Turn bounds in seconds relative to the conversation part, the same reference as
            the word timestamps of a pseudo-label.
    """

    segment_id: str
    conversation_id: str
    part_id: str
    start: float
    end: float


def map_podcast_segments_to_parts(paths: Any = None) -> dict[str, PodcastSegmentSpan]:
    r"""Locate the MSP-Podcast speaking turns inside the conversation parts.

    `Time_Labels/segments.json` places the shared speaking turns relative to their
    conversation; the turns are rebased onto the conversation part that contains them.
    Turns straddling a part boundary are dropped.

    Args:
        paths (`str`, `os.PathLike` or `CorpusPaths`, *optional*):
            Release directory or resolved paths.

    Returns:
        `dict`: `PodcastSegmentSpan` keyed by segment identifier.
    """
    corpus = CorpusPaths.from_root(paths)
    segments_file = require_path(
        corpus.segments_file,
        "MSP-Podcast segment timing 'Time_Labels/segments.json'",
        root=corpus.root,
    )
    part_times = read_part_times(corpus)

    parts_of: dict[str, list[tuple[str, float, float]]] = {}
    for part_id, (start, end) in part_times.items():
        parts_of.setdefault(conversation_of(part_id), []).append((part_id, start, end))
    for entries in parts_of.values():
        entries.sort(key=lambda entry: entry[1])

    with segments_file.open(encoding="utf-8") as handle:
        segments = json.load(handle)

    spans: dict[str, PodcastSegmentSpan] = {}
    for segment_id, entry in segments.items():
        conversation_id = entry.get("Conversation") or entry.get("Conversation_Part")
        if conversation_id is None:
            continue
        start = float(entry["Start_Time"])
        end = float(entry["End_Time"])
        for part_id, part_start, part_end in parts_of.get(conversation_id, ()):
            if start >= part_start and end <= part_end:
                spans[segment_id] = PodcastSegmentSpan(
                    segment_id=segment_id,
                    conversation_id=conversation_id,
                    part_id=part_id,
                    start=start - part_start,
                    end=end - part_start,
                )
                break
    return spans


def load_podcast_transcripts(
    source: os.PathLike[str] | str, segment_ids: Iterable[str] | None = None
) -> dict[str, str]:
    r"""Read the human MSP-Podcast transcripts.

    Args:
        source (`os.PathLike` or `str`):
            Directory of `MSP-PODCAST_<podcast>_<segment>.txt` files, or the `Transcripts`
            archive of the MSP-Podcast release, which is read without being extracted.
        segment_ids (`Iterable` of `str`, *optional*):
            Read only these segments. The archive holds every MSP-Podcast turn, so passing
            the turns shared with MSP-Conversation is strongly preferred.

    Returns:
        `dict`: Raw transcript text keyed by segment identifier.
    """
    origin = require_path(source, "MSP-Podcast transcripts")
    wanted = None if segment_ids is None else set(segment_ids)
    transcripts: dict[str, str] = {}

    if origin.is_dir():
        for path in sorted(origin.glob("*.txt")):
            if wanted is None or path.stem in wanted:
                transcripts[path.stem] = path.read_text(
                    encoding="utf-8", errors="replace"
                ).strip()
        return transcripts

    if origin.suffix.lower() != ".zip":
        raise ValueError(
            f"'{origin}' is neither a directory of transcripts nor a zip archive."
        )

    with zipfile.ZipFile(origin) as archive:
        for info in archive.infolist():
            name = info.filename
            if info.is_dir() or not name.endswith(".txt") or "__MACOSX" in name:
                continue
            segment_id = Path(name).stem
            if wanted is not None and segment_id not in wanted:
                continue
            with archive.open(info) as handle:
                transcripts[segment_id] = handle.read().decode("utf-8", errors="replace").strip()
    return transcripts


@dataclass(frozen=True)
class SegmentScore:
    r"""The word error rate of one MSP-Podcast speaking turn.

    Args:
        segment_id (`str`):
            MSP-Podcast segment identifier.
        part_id (`str`):
            Conversation part the hypothesis was cut from.
        reference (`str`):
            Normalized human transcript.
        hypothesis (`str`):
            Normalized pseudo-label restricted to the turn.
        stats (`WordErrorRate`):
            Edit statistics of the pair.
    """

    segment_id: str
    part_id: str
    reference: str
    hypothesis: str
    stats: WordErrorRate

    def to_dict(self) -> dict[str, Any]:
        r"""Return a JSON-serializable view of the score."""
        return {
            "segment_id": self.segment_id,
            "part_id": self.part_id,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
            **self.stats.to_dict(),
        }


@dataclass(frozen=True)
class PseudoLabelQualityReport:
    r"""Word error rate of the pseudo-labels on the shared MSP-Podcast speaking turns.

    Args:
        mode (`str`):
            Transcription mode that was scored.
        stats (`WordErrorRate`):
            Edit statistics accumulated over every scored turn.
        num_segments (`int`):
            Number of scored turns.
        skipped (`collections.Counter`):
            Number of turns dropped, per reason.
        scores (`tuple` of `SegmentScore`):
            Per-turn scores, empty when they were not kept.
    """

    mode: str
    stats: WordErrorRate
    num_segments: int
    skipped: Counter
    scores: tuple[SegmentScore, ...] = ()

    @property
    def wer(self) -> float:
        r"""Aggregate word error rate over every scored turn."""
        return self.stats.wer

    def to_dict(self) -> dict[str, Any]:
        r"""Return a JSON-serializable view of the report."""
        return {
            "mode": self.mode,
            "num_segments": self.num_segments,
            "skipped": dict(self.skipped),
            **self.stats.to_dict(),
            "scores": [score.to_dict() for score in self.scores],
        }


def evaluate_pseudo_labels(
    records: Mapping[str, PseudoLabelRecord] | os.PathLike[str] | str,
    transcripts: Mapping[str, str] | os.PathLike[str] | str,
    paths: Any = None,
    *,
    mode: str = INTENDED_MODE,
    normalizer: Callable[[str], str] | None = None,
    word_policy: str = "midpoint",
    min_reference_words: int = 1,
    keep_scores: bool = True,
    max_segments: int | None = None,
) -> PseudoLabelQualityReport:
    r"""Score the pseudo-labels against the human MSP-Podcast transcripts.

    The MSP-Podcast speaking turns lying inside MSP-Conversation are located with
    `Time_Labels/segments.json`, the pseudo-label words falling into each turn are gathered
    through their timestamps, and every pair is scored with a word-level Levenshtein
    alignment. The intended mode is the meaningful comparison because the human transcripts
    are cleaned; scoring the verbatim mode charges every correctly transcribed filler as an
    insertion.

    Args:
        records (`Mapping` or `os.PathLike` or `str`):
            Pseudo-labels keyed by part, or the file they were written to.
        transcripts (`Mapping` or `os.PathLike` or `str`):
            Human transcripts keyed by segment, or the directory or archive holding them.
        paths (`str`, `os.PathLike` or `CorpusPaths`, *optional*):
            Release directory or resolved paths.
        mode (`str`, *optional*, defaults to `"intended"`):
            Transcription mode to score.
        normalizer (`Callable`, *optional*):
            Normalizer applied to both sides, defaults to [`english_wer_normalizer`].
        word_policy (`str`, *optional*, defaults to `"midpoint"`):
            Word selection policy, see [`words_in_span`].
        min_reference_words (`int`, *optional*, defaults to 1):
            Turns with fewer reference words are skipped.
        keep_scores (`bool`, *optional*, defaults to `True`):
            Keep the per-turn scores in the report.
        max_segments (`int`, *optional*):
            Score at most this many turns, which is how a quick sanity pass is run.

    Returns:
        `PseudoLabelQualityReport`: Aggregate and per-turn word error rates.
    """
    if isinstance(records, (str, os.PathLike)):
        records = load_pseudo_labels(records)
    spans = {
        segment_id: span
        for segment_id, span in map_podcast_segments_to_parts(paths).items()
        if span.part_id in records
    }
    if isinstance(transcripts, (str, os.PathLike)):
        transcripts = load_podcast_transcripts(transcripts, segment_ids=spans.keys())
    if normalizer is None:
        normalizer = english_wer_normalizer()

    total = WordErrorRate()
    skipped: Counter = Counter()
    scores: list[SegmentScore] = []

    for segment_id in sorted(spans):
        if max_segments is not None and len(scores) >= max_segments:
            break
        span = spans[segment_id]
        transcript = records[span.part_id].transcripts.get(mode)
        if transcript is None:
            skipped["missing_mode"] += 1
            continue
        if not transcript.words:
            skipped["missing_word_timestamps"] += 1
            continue
        reference_text = transcripts.get(segment_id)
        if reference_text is None:
            skipped["missing_reference"] += 1
            continue

        reference = tokenize_transcript(normalizer(reference_text))
        if len(reference) < min_reference_words:
            skipped["short_reference"] += 1
            continue
        selected = words_in_span(transcript.words, span.start, span.end, policy=word_policy)
        hypothesis = tokenize_transcript(normalizer(" ".join(word.word for word in selected)))

        stats = word_error_rate(reference, hypothesis)
        total = total + stats
        scores.append(
            SegmentScore(
                segment_id=segment_id,
                part_id=span.part_id,
                reference=" ".join(reference) if keep_scores else "",
                hypothesis=" ".join(hypothesis) if keep_scores else "",
                stats=stats,
            )
        )

    return PseudoLabelQualityReport(
        mode=mode,
        stats=total,
        num_segments=len(scores),
        skipped=skipped,
        scores=tuple(scores) if keep_scores else (),
    )


__all__ = [
    "CUTOFF_MARKER",
    "CONTENT_NORMALIZATION",
    "DEFAULT_MODEL_ID",
    "DEFAULT_NORMALIZATION",
    "DEFAULT_SUPERVISION_POLICY",
    "FILLER_TAGS",
    "INTENDED_MODE",
    "MODEL_IDS",
    "NON_LEXICAL_TAGS",
    "SOUND_TAGS",
    "TAG_ALIASES",
    "VERBATIM_MODE",
    "VOCAL_EVENT_TAGS",
    "ConversationPart",
    "CrisperWhisperTranscriber",
    "ModeTranscript",
    "NormalizationConfig",
    "PodcastSegmentSpan",
    "PseudoLabelConfig",
    "PseudoLabelQualityReport",
    "PseudoLabelRecord",
    "SegmentScore",
    "SpanSpeechStats",
    "SubsequenceCheck",
    "SupervisionPolicy",
    "Transcriber",
    "Word",
    "WordErrorRate",
    "build_record",
    "check_intended_subsequence",
    "default_output_dir",
    "english_wer_normalizer",
    "evaluate_pseudo_labels",
    "iter_pseudo_labels",
    "load_conversation_parts",
    "load_part_audio",
    "load_podcast_transcripts",
    "load_pseudo_labels",
    "map_podcast_segments_to_parts",
    "normalize_transcript",
    "run_pseudo_labelling",
    "span_speech_stats",
    "split_into_ctc_units",
    "tokenize_transcript",
    "transcribe_parts",
    "transcript_for_span",
    "transcript_supervision_mask",
    "word_error_rate",
    "words_in_span",
    "write_pseudo_labels",
]
