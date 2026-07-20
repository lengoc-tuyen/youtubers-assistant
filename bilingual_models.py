"""Immutable data contracts shared by the bilingual subtitle pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CanonicalLyricLine:
    """A cleaned source lyric line with its stable, one-based identity."""

    id: int
    source: str


@dataclass(frozen=True)
class LyricsSourceMetadata:
    """Provenance for plain lyrics used to construct canonical lines."""

    provider: str
    title: str
    artist: str
    duration: Optional[float]
    record_id: Optional[int]


@dataclass(frozen=True)
class AlignedLyricLine:
    """A canonical source line with immutable audio timing."""

    id: int
    source: str
    start: float
    end: float


@dataclass(frozen=True)
class TranslationLine:
    """A validated Vietnamese translation for one canonical line."""

    id: int
    translation: str
    needs_review: bool = False


@dataclass(frozen=True)
class BilingualCue:
    """A complete subtitle cue joined solely by canonical line ID."""

    id: int
    source: str
    translation: str
    start: float
    end: float


@dataclass(frozen=True)
class CueQualityWarning:
    """A non-blocking quality warning for a valid subtitle cue."""

    line_id: int
    code: str
    message: str
