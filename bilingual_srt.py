"""Pure SRT renderers for English and English--Vietnamese subtitle files."""

from __future__ import annotations

from typing import Sequence

from bilingual_models import AlignedLyricLine, BilingualCue
from srt_maker import srt_time


def render_english_srt(lines: Sequence[AlignedLyricLine]) -> str:
    """Render English-only, line-level SRT in canonical ID order."""
    return "\n".join(
        f"{line.id}\n{srt_time(line.start)} --> {srt_time(line.end)}\n{line.source}\n"
        for line in lines
    )


def render_bilingual_srt(cues: Sequence[BilingualCue]) -> str:
    """Render English source followed by parenthesized Vietnamese translation."""
    return "\n".join(
        f"{cue.id}\n{srt_time(cue.start)} --> {srt_time(cue.end)}\n"
        f"{cue.source}\n({cue.translation})\n"
        for cue in cues
    )
