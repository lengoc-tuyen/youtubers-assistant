"""Translation payload validation and ID-only bilingual cue merging."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from bilingual_models import (
    AlignedLyricLine,
    BilingualCue,
    CanonicalLyricLine,
    TranslationLine,
)


@dataclass(frozen=True)
class TranslationValidationError(ValueError):
    """Details of a strict JSON translation contract violation."""

    message: str
    invalid_ids: Tuple[int, ...] = ()

    def __str__(self) -> str:
        return self.message


def get_translation_items(payload: Mapping[str, Any]) -> Any:
    """Return the provider's translation array, including the legacy alias."""
    return payload.get("lines", payload.get("translations"))


def validate_translation_response(
    payload: Mapping[str, Any],
    canonical_lines: Sequence[CanonicalLyricLine],
) -> Tuple[TranslationLine, ...]:
    """Validate strict Groq JSON and return translations ordered by canonical ID."""
    response_lines = get_translation_items(payload)
    if not isinstance(response_lines, list):
        raise TranslationValidationError("Translation response must contain a lines array.")

    expected_ids = tuple(line.id for line in canonical_lines)
    expected_id_set = set(expected_ids)
    parsed = []
    invalid_ids = set()
    seen_ids = []

    for item in response_lines:
        if not isinstance(item, Mapping):
            raise TranslationValidationError("Every translation item must be an object.")
        line_id = item.get("id")
        translation = item.get("translation")
        if not isinstance(line_id, int):
            raise TranslationValidationError("Every translation ID must be an integer.")
        seen_ids.append(line_id)
        if line_id not in expected_id_set:
            invalid_ids.add(line_id)
            continue
        if not isinstance(translation, str) or not translation.strip():
            invalid_ids.add(line_id)
            continue
        needs_review = item.get("needs_review", False)
        if not isinstance(needs_review, bool):
            invalid_ids.add(line_id)
            continue
        parsed.append(TranslationLine(
            id=line_id,
            translation=translation.strip(),
            needs_review=needs_review,
        ))

    duplicate_ids = {
        line_id for line_id, count in Counter(seen_ids).items() if count > 1
    }
    known_duplicates = duplicate_ids & expected_id_set
    valid_ids = {line.id for line in parsed} - known_duplicates
    missing_ids = expected_id_set - valid_ids
    all_invalid_ids = tuple(sorted(invalid_ids | missing_ids | known_duplicates))
    if (
        len(response_lines) != len(expected_ids)
        or invalid_ids
        or duplicate_ids
        or missing_ids
    ):
        raise TranslationValidationError(
            "Translation IDs must match canonical IDs exactly once with non-empty strings.",
            all_invalid_ids,
        )

    by_id = {line.id: line for line in parsed}
    return tuple(by_id[line_id] for line_id in expected_ids)


def merge_by_line_id(
    aligned_lines: Sequence[AlignedLyricLine],
    translations: Sequence[TranslationLine],
) -> Tuple[BilingualCue, ...]:
    """Merge matching IDs; repeated source text is never used as a join key."""
    translations_by_id = {line.id: line for line in translations}
    if len(translations_by_id) != len(translations):
        raise ValueError("Translation IDs must be unique before merge.")

    aligned_ids = tuple(line.id for line in aligned_lines)
    if len(set(aligned_ids)) != len(aligned_ids):
        raise ValueError("Alignment IDs must be unique before merge.")
    if set(aligned_ids) != set(translations_by_id):
        raise ValueError("Alignment and translation IDs must match exactly.")

    return tuple(
        BilingualCue(
            id=line.id,
            source=line.source,
            translation=translations_by_id[line.id].translation,
            start=line.start,
            end=line.end,
        )
        for line in aligned_lines
    )
