"""The single canonicalization boundary for bilingual lyric processing."""

from __future__ import annotations

import re
from typing import Tuple

from bilingual_models import CanonicalLyricLine


SECTION_TAG = re.compile(r"^\[.*?\]$")
GENIUS_RECOMMENDATION_MARKER = "you might also like"


def canonicalize_lyrics(raw_lyrics: str) -> Tuple[CanonicalLyricLine, ...]:
    """Create the one clean LRCLIB lyric source used by translation and alignment.

    Only presentation/boilerplate is removed: blank lines, section tags, and a
    Genius-style ``You might also like`` recommendation block. All remaining
    lyric text and its order are preserved except surrounding space.
    """
    cleaned_sources = []
    skip_recommendation_block = False

    for raw_line in raw_lyrics.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # A new section ends the recommendation block, but is not a lyric line.
        if SECTION_TAG.fullmatch(line):
            skip_recommendation_block = False
            continue

        if line.casefold() == GENIUS_RECOMMENDATION_MARKER:
            skip_recommendation_block = True
            continue

        if skip_recommendation_block:
            continue

        cleaned_sources.append(line)

    canonical_sources = tuple(cleaned_sources)
    if not canonical_sources:
        raise ValueError("Lyrics are empty after canonicalization.")
    return tuple(
        CanonicalLyricLine(id=index, source=source)
        for index, source in enumerate(canonical_sources, start=1)
    )
