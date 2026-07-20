"""The single canonicalization boundary for bilingual lyric processing."""

from __future__ import annotations

import re
from typing import Tuple

from bilingual_models import CanonicalLyricLine


SECTION_TAG = re.compile(r"^\[[^\]\r\n]+\]$")


def canonicalize_lyrics(raw_lyrics: str) -> Tuple[CanonicalLyricLine, ...]:
    """Drop only blank/section lines while preserving every source lyric line and order."""
    canonical_sources = tuple(
        stripped
        for raw_line in raw_lyrics.splitlines()
        for stripped in (raw_line.strip(),)
        if stripped and not SECTION_TAG.fullmatch(stripped)
    )
    if not canonical_sources:
        raise ValueError("Lyrics are empty after canonicalization.")
    return tuple(
        CanonicalLyricLine(id=index, source=source)
        for index, source in enumerate(canonical_sources, start=1)
    )
