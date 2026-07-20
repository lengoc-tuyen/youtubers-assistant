"""Post-merge timing and Vietnamese reading-speed checks."""

from __future__ import annotations

from typing import Sequence, Tuple

from bilingual_models import BilingualCue, CueQualityWarning


DEFAULT_MIN_CUE_DURATION = 0.25
DEFAULT_MAX_VIETNAMESE_CPS = 17.0


class CueQualityError(ValueError):
    """Raised when a cue cannot safely be written to an SRT file."""


def evaluate_cue_quality(
    cues: Sequence[BilingualCue],
    *,
    min_duration: float = DEFAULT_MIN_CUE_DURATION,
    max_vietnamese_cps: float = DEFAULT_MAX_VIETNAMESE_CPS,
) -> Tuple[CueQualityWarning, ...]:
    """Reject invalid timing and warn when Vietnamese text exceeds a readable speed."""
    if min_duration <= 0:
        raise ValueError("min_duration must be positive.")
    if max_vietnamese_cps <= 0:
        raise ValueError("max_vietnamese_cps must be positive.")

    warnings = []
    for cue in cues:
        duration = cue.end - cue.start
        if duration <= 0:
            raise CueQualityError(f"Line {cue.id} has non-positive duration.")
        if duration < min_duration:
            raise CueQualityError(
                f"Line {cue.id} is shorter than minimum duration {min_duration:.2f}s.",
            )

        visible_characters = len("".join(cue.translation.split()))
        characters_per_second = visible_characters / duration
        if characters_per_second > max_vietnamese_cps:
            warnings.append(CueQualityWarning(
                line_id=cue.id,
                code="reading_speed_high",
                message=(
                    f"Vietnamese reading speed is {characters_per_second:.1f} CPS "
                    f"(limit {max_vietnamese_cps:.1f})."
                ),
            ))
    return tuple(warnings)
