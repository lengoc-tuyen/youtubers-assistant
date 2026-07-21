"""Provider-neutral literal/Groq translation workflow and targeted repair logic."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, Mapping, Protocol, Sequence, Tuple

from bilingual_models import BilingualCue, CanonicalLyricLine, TranslationLine
from translation_workflow import (
    TranslationValidationError,
    get_translation_items,
    validate_translation_response,
)


GROQ_PROMPT_VERSION = "2026-07-english-vietnamese-lyrics-v2"
GROQ_SYSTEM_PROMPT = """You are an expert English-to-Vietnamese lyric translator and subtitle adapter.

Read the entire song before translating so that every line is interpreted using the full-song context.

The English source is authoritative. The literal Vietnamese draft is only a reference and may contain mistranslations, unnatural wording, or incorrect literal interpretations.

Produce natural, expressive Vietnamese that preserves the original meaning, imagery, emotional tone, humor, irony, wordplay, cultural references, and the user-requested vibe.

Translate the intended effect rather than individual words when a literal translation would sound unnatural. Preserve proper nouns and terms without a natural Vietnamese equivalent. For coined, ambiguous, or culturally specific terms, prefer a concise contextual rendering or keep the original term instead of inventing an unsupported meaning.

Keep repeated source lines translated consistently unless their context clearly changes.

Return exactly one Vietnamese translation for every input ID. Do not merge, split, omit, duplicate, reorder, or create lines. Do not add parentheses, timestamps, English text, explanations, markdown, or commentary.

Return valid JSON only, with this exact root shape:
{"lines":[{"id":1,"translation":"Vietnamese translation"}]}

The root array key must be named "lines", never "translations". Every output item must contain exactly one integer input ID and one non-empty Vietnamese "translation" string."""

class JsonTranslationClient(Protocol):
    """An injected OpenAI-compatible JSON client; implementations own HTTP concerns."""

    def request_json(self, system_prompt: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the parsed JSON object from a provider response."""


def build_groq_payload(
    canonical_lines: Sequence[CanonicalLyricLine],
    literal_drafts: Mapping[int, str],
    *,
    title: str,
    artist: str,
    vibe: str,
) -> Dict[str, Any]:
    """Build the full-song, timestamp-free Groq input contract."""
    return {
        "song": {"title": title, "artist": artist},
        "vibe": vibe,
        "lines": [
            {
                "id": line.id,
                "source": line.source,
                "literal_draft": literal_drafts[line.id],
            }
            for line in canonical_lines
        ],
    }


def _extract_valid_translations(
    payload: Mapping[str, Any],
    canonical_lines: Sequence[CanonicalLyricLine],
) -> Dict[int, str]:
    """Keep only unambiguous valid responses while preparing a partial retry."""
    response_lines = get_translation_items(payload)
    if not isinstance(response_lines, list):
        return {}
    expected_ids = {line.id for line in canonical_lines}
    candidate_ids = [
        item.get("id")
        for item in response_lines
        if isinstance(item, Mapping) and isinstance(item.get("id"), int)
    ]
    duplicate_ids = {
        line_id for line_id, count in Counter(candidate_ids).items() if count > 1
    }
    valid = {}
    for item in response_lines:
        if not isinstance(item, Mapping):
            continue
        line_id = item.get("id")
        translation = item.get("translation")
        if (
            isinstance(line_id, int)
            and line_id in expected_ids
            and line_id not in duplicate_ids
            and isinstance(translation, str)
            and translation.strip()
        ):
            valid[line_id] = translation.strip()
    return valid


def build_partial_repair_context(
    canonical_lines: Sequence[CanonicalLyricLine],
    invalid_ids: Sequence[int],
    current_translations: Mapping[int, str],
) -> Tuple[Dict[str, Any], ...]:
    """Create minimal retry inputs with only neighboring source context."""
    by_id = {line.id: index for index, line in enumerate(canonical_lines)}
    repair_context = []
    for line_id in invalid_ids:
        index = by_id[line_id]
        line = canonical_lines[index]
        repair_context.append({
            "id": line.id,
            "source": line.source,
            "current_translation": current_translations.get(line.id, ""),
            "previous_source": canonical_lines[index - 1].source if index else None,
            "next_source": (
                canonical_lines[index + 1].source
                if index + 1 < len(canonical_lines)
                else None
            ),
        })
    return tuple(repair_context)


def validate_with_partial_repair(
    initial_payload: Mapping[str, Any],
    canonical_lines: Sequence[CanonicalLyricLine],
    repair: Callable[[Tuple[Dict[str, Any], ...]], Mapping[str, Any]],
) -> Tuple[TranslationLine, ...]:
    """Validate once, then retry only known bad IDs while retaining valid outputs."""
    try:
        return validate_translation_response(initial_payload, canonical_lines)
    except TranslationValidationError as error:
        valid_translations = _extract_valid_translations(initial_payload, canonical_lines)
        expected_ids = {line.id for line in canonical_lines}
        retry_ids = tuple(
            line.id
            for line in canonical_lines
            if line.id in error.invalid_ids and line.id in expected_ids
        )
        if not retry_ids:
            raise

        repair_context = build_partial_repair_context(
            canonical_lines,
            retry_ids,
            valid_translations,
        )
        repair_payload = repair(repair_context)
        repair_lines = tuple(line for line in canonical_lines if line.id in set(retry_ids))
        repaired = validate_translation_response(repair_payload, repair_lines)
        combined_payload = {
            "lines": [
                {
                    "id": line.id,
                    "translation": (
                        valid_translations[line.id]
                        if line.id in valid_translations
                        else next(item.translation for item in repaired if item.id == line.id)
                    ),
                }
                for line in canonical_lines
            ],
        }
        return validate_translation_response(combined_payload, canonical_lines)


class TranslateLibraryLiteralTranslator:
    """Use the ``translate`` Python package for English-to-Vietnamese draft lines."""

    cache_identity = "translate-python-en-vi"

    def __init__(self, translator_factory: Callable[..., Any] | None = None) -> None:
        self._translator_factory = translator_factory

    def translate(self, canonical_lines: Sequence[CanonicalLyricLine]) -> Dict[int, str]:
        """Translate every immutable canonical source line using one library translator."""
        factory = self._translator_factory or self._load_translator_factory()
        translator = factory(to_lang="vi", from_lang="en")
        try:
            drafts = {
                line.id: translator.translate(line.source).strip()
                for line in canonical_lines
            }
        except Exception as error:
            raise RuntimeError("The translate library could not create literal Vietnamese drafts.") from error
        if any(not isinstance(draft, str) or not draft for draft in drafts.values()):
            raise RuntimeError("The translate library returned an empty literal draft.")
        return drafts

    @staticmethod
    def _load_translator_factory() -> Callable[..., Any]:
        try:
            from translate import Translator
        except ImportError as error:
            raise RuntimeError("translate is not installed. Run: pip install -r requirements.txt") from error
        return Translator


class GroqTranslationService:
    """Translate full-song lyrics with strict IDs and bounded, targeted repair hooks."""

    def __init__(self, client: JsonTranslationClient) -> None:
        self._client = client
        self.cache_identity = "groq"

    def translate(
        self,
        canonical_lines: Sequence[CanonicalLyricLine],
        literal_drafts: Mapping[int, str],
        *,
        title: str,
        artist: str,
        vibe: str,
    ) -> Tuple[TranslationLine, ...]:
        payload = build_groq_payload(
            canonical_lines,
            literal_drafts,
            title=title,
            artist=artist,
            vibe=vibe,
        )
        response = self._client.request_json(GROQ_SYSTEM_PROMPT, payload)
        return validate_with_partial_repair(
            response,
            canonical_lines,
            lambda repair_context: self._repair(
                repair_context,
                title=title,
                artist=artist,
                vibe=vibe,
            ),
        )

    def _repair(
        self,
        repair_context: Tuple[Dict[str, Any], ...],
        *,
        title: str,
        artist: str,
        vibe: str,
    ) -> Mapping[str, Any]:
        """Repair only broken IDs while retaining the full-song interpretation context."""
        return self._client.request_json(
            GROQ_SYSTEM_PROMPT + "\nRepair only the supplied IDs and return strict JSON.",
            {
                "song": {"title": title, "artist": artist},
                "vibe": vibe,
                "lines": list(repair_context),
            },
        )

    def shorten(
        self,
        cues: Sequence[BilingualCue],
        line_ids: Sequence[int],
        max_vietnamese_cps: float,
    ) -> Tuple[TranslationLine, ...]:
        """Request concise repairs only for cues that exceed the reading-speed limit."""
        cue_by_id = {cue.id: index for index, cue in enumerate(cues)}
        selected_cues = []
        for line_id in line_ids:
            index = cue_by_id[line_id]
            cue = cues[index]
            duration = cue.end - cue.start
            selected_cues.append({
                "id": cue.id,
                "source": cue.source,
                "current_translation": cue.translation,
                "previous_line": cues[index - 1].source if index else None,
                "next_line": cues[index + 1].source if index + 1 < len(cues) else None,
                "duration_seconds": round(duration, 3),
                "target_max_characters": max(1, int(duration * max_vietnamese_cps)),
            })
        response = self._client.request_json(
            GROQ_SYSTEM_PROMPT + (
                "\nShorten only the supplied Vietnamese translations to fit duration. "
                "Do not change meaning, vibe, IDs, source text, or timestamps. "
                "Return strict JSON only."
            ),
            {"lines": selected_cues},
        )
        canonical_subset = tuple(
            CanonicalLyricLine(id=cue["id"], source=cue["source"])
            for cue in selected_cues
        )
        return validate_translation_response(response, canonical_subset)
