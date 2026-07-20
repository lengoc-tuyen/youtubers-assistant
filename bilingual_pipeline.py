"""CLI orchestrator for cached English--Vietnamese lyric subtitle generation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Tuple

from bilingual_models import (
    AlignedLyricLine,
    BilingualCue,
    CanonicalLyricLine,
    TranslationLine,
)
from bilingual_srt import render_bilingual_srt, render_english_srt
from canonical_lyrics import canonicalize_lyrics
from lrclib_lyrics import LRCLIBLyricsRecord, lookup_lyrics_record
from pipeline_cache import JsonFileCache, atomic_write_json, atomic_write_text
from provider_clients import OpenAICompatibleJsonClient, RetryPolicy
from quality_guard import DEFAULT_MAX_VIETNAMESE_CPS, evaluate_cue_quality
from srt_maker import align_canonical_lines
from translation_service import (
    GROQ_PROMPT_VERSION,
    GroqTranslationService,
    TranslateLibraryLiteralTranslator,
)
from translation_workflow import merge_by_line_id


class LiteralTranslator(Protocol):
    cache_identity: str

    def translate(self, lines: Sequence[CanonicalLyricLine]) -> Mapping[int, str]:
        """Return a validated literal draft per canonical ID."""


class GroqTranslator(Protocol):
    cache_identity: str

    def translate(
        self,
        lines: Sequence[CanonicalLyricLine],
        literal_drafts: Mapping[int, str],
        *,
        title: str,
        artist: str,
        vibe: str,
    ) -> Sequence[TranslationLine]:
        """Return a validated stylistic translation per canonical ID."""


Aligner = Callable[
    ...,
    Tuple[Tuple[AlignedLyricLine, ...], dict[str, Any], Tuple[tuple, ...]],
]
LyricsLookup = Callable[[str, str], Optional[LRCLIBLyricsRecord]]


@dataclass(frozen=True)
class PipelineRequest:
    youtube_url: str
    title: str
    artist: str
    vibe: str
    output_path: Path
    data_dir: Path
    max_vietnamese_cps: float = DEFAULT_MAX_VIETNAMESE_CPS


@dataclass(frozen=True)
class PipelineResult:
    cues: Tuple[BilingualCue, ...]
    output_en_path: Path
    output_bilingual_path: Path
    translation_path: Path
    metadata_path: Path


class BilingualPipeline:
    """Coordinates canonical lyric lookup, parallel alignment/translation, and artifact writes."""

    def __init__(
        self,
        *,
        lyrics_lookup: LyricsLookup,
        aligner: Aligner,
        literal_translator: LiteralTranslator,
        groq_translator: GroqTranslator,
        cache: JsonFileCache,
    ) -> None:
        self._lyrics_lookup = lyrics_lookup
        self._aligner = aligner
        self._literal_translator = literal_translator
        self._groq_translator = groq_translator
        self._cache = cache

    async def run(self, request: PipelineRequest) -> PipelineResult:
        """Run the lyric-dependent stages, then await alignment and translation concurrently."""
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        record, lyrics_cache_hit = await self._load_lyrics(request)
        canonical_lines = canonicalize_lyrics(record.plain_lyrics)
        canonical_payload = self._canonical_payload(canonical_lines)
        canonical_hash = self._cache.key_for(canonical_payload)

        alignment_task = asyncio.to_thread(
            self._load_or_align,
            request,
            canonical_lines,
            canonical_hash,
        )
        translation_task = self._load_or_translate(
            request,
            canonical_lines,
            canonical_hash,
        )
        (aligned_lines, raw_alignment, alignment_cache_hit), (
            translations,
            literal_cache_hit,
            groq_cache_hit,
        ) = await asyncio.gather(alignment_task, translation_task)

        cues = merge_by_line_id(aligned_lines, translations)
        warnings = evaluate_cue_quality(
            cues,
            max_vietnamese_cps=request.max_vietnamese_cps,
        )
        cues, warnings = await self._shorten_overlong_cues(
            cues,
            warnings,
            request.max_vietnamese_cps,
        )
        self._cache.put("groq", self._groq_cache_key(request, canonical_hash), {
            "lines": [
                {"id": cue.id, "translation": cue.translation}
                for cue in cues
            ],
        })
        artifact_paths = self._write_artifacts(
            request,
            aligned_lines,
            cues,
            raw_alignment,
            record,
            canonical_hash,
            warnings,
            {
                "lyrics": lyrics_cache_hit,
                "alignment": alignment_cache_hit,
                "literal": literal_cache_hit,
                "groq": groq_cache_hit,
            },
        )
        return PipelineResult(cues=cues, **artifact_paths)

    async def _shorten_overlong_cues(
        self,
        cues: Tuple[BilingualCue, ...],
        warnings: Sequence[Any],
        max_vietnamese_cps: float,
    ) -> Tuple[Tuple[BilingualCue, ...], Tuple[Any, ...]]:
        """Run one ID-scoped shortening repair when the configured Groq provider supports it."""
        line_ids = tuple(
            warning.line_id
            for warning in warnings
            if getattr(warning, "code", None) == "reading_speed_high"
        )
        shorten = getattr(self._groq_translator, "shorten", None)
        if not line_ids or not callable(shorten):
            return cues, tuple(warnings)

        shortened = tuple(await asyncio.to_thread(
            shorten,
            cues,
            line_ids,
            max_vietnamese_cps,
        ))
        if [line.id for line in shortened] != list(line_ids):
            raise RuntimeError("Length repair IDs did not match the overlong cues exactly.")
        replacements = {line.id: line.translation for line in shortened}
        repaired_cues = tuple(
            BilingualCue(
                id=cue.id,
                source=cue.source,
                translation=replacements.get(cue.id, cue.translation),
                start=cue.start,
                end=cue.end,
            )
            for cue in cues
        )
        return repaired_cues, evaluate_cue_quality(
            repaired_cues,
            max_vietnamese_cps=max_vietnamese_cps,
        )

    async def _load_lyrics(
        self,
        request: PipelineRequest,
    ) -> Tuple[LRCLIBLyricsRecord, bool]:
        key = self._cache.key_for({
            "provider": "lrclib",
            "title": request.title,
            "artist": request.artist,
        })
        cached = self._cache.get("lyrics", key)
        if cached:
            try:
                return self._deserialize_lyrics_record(cached), True
            except (RuntimeError, TypeError, ValueError):
                pass

        record = await asyncio.to_thread(self._lyrics_lookup, request.artist, request.title)
        if record is None:
            raise RuntimeError("Không tìm thấy plain lyrics trên LRCLIB; sẽ không chạy alignment.")
        self._cache.put("lyrics", key, self._serialize_lyrics_record(record))
        return record, False

    async def _load_or_translate(
        self,
        request: PipelineRequest,
        canonical_lines: Sequence[CanonicalLyricLine],
        canonical_hash: str,
    ) -> Tuple[Tuple[TranslationLine, ...], bool, bool]:
        literal_key = self._cache.key_for({
            "canonical_hash": canonical_hash,
            "provider": self._cache_identity(self._literal_translator),
        })
        cached_literal = self._cache.get("literal", literal_key)
        if cached_literal:
            try:
                raw_drafts = cached_literal["drafts"]
                if not isinstance(raw_drafts, Mapping):
                    raise ValueError("Cached literal drafts are invalid.")
                literal_drafts = {
                    int(line_id): draft
                    for line_id, draft in raw_drafts.items()
                    if isinstance(draft, str)
                }
                self._ensure_drafts_cover(canonical_lines, literal_drafts)
                literal_cache_hit = True
            except (KeyError, TypeError, ValueError, RuntimeError):
                cached_literal = None
        if not cached_literal:
            literal_drafts = dict(await asyncio.to_thread(
                self._literal_translator.translate,
                canonical_lines,
            ))
            self._ensure_drafts_cover(canonical_lines, literal_drafts)
            self._cache.put("literal", literal_key, {
                "drafts": {str(line_id): text for line_id, text in literal_drafts.items()},
            })
            literal_cache_hit = False

        groq_key = self._groq_cache_key(request, canonical_hash)
        cached_groq = self._cache.get("groq", groq_key)
        if cached_groq:
            try:
                raw_lines = cached_groq["lines"]
                if not isinstance(raw_lines, list):
                    raise ValueError("Cached Groq translations are invalid.")
                translations = tuple(
                    TranslationLine(
                        id=int(line["id"]),
                        translation=str(line["translation"]),
                        needs_review=bool(line.get("needs_review", False)),
                    )
                    for line in raw_lines
                    if isinstance(line, Mapping)
                )
                self._ensure_translation_ids(canonical_lines, translations)
                return translations, literal_cache_hit, True
            except (KeyError, TypeError, ValueError, RuntimeError):
                cached_groq = None

        translations = tuple(await asyncio.to_thread(
            self._groq_translator.translate,
            canonical_lines,
            literal_drafts,
            title=request.title,
            artist=request.artist,
            vibe=request.vibe,
        ))
        self._ensure_translation_ids(canonical_lines, translations)
        self._cache.put("groq", groq_key, {
            "lines": [asdict(line) for line in translations],
        })
        return translations, literal_cache_hit, False

    def _load_or_align(
        self,
        request: PipelineRequest,
        canonical_lines: Sequence[CanonicalLyricLine],
        canonical_hash: str,
    ) -> Tuple[Tuple[AlignedLyricLine, ...], dict[str, Any], bool]:
        cache_key = self._cache.key_for({
            "youtube_url": request.youtube_url,
            "canonical_hash": canonical_hash,
            "aligner": self._cache_identity(self._aligner),
        })
        cached = self._cache.get("alignment", cache_key)
        if cached:
            try:
                raw_lines = cached["lines"]
                if not isinstance(raw_lines, list):
                    raise ValueError("Cached alignment lines are invalid.")
                lines = tuple(
                    AlignedLyricLine(
                        id=int(line["id"]),
                        source=str(line["source"]),
                        start=float(line["start"]),
                        end=float(line["end"]),
                    )
                    for line in raw_lines
                    if isinstance(line, Mapping)
                )
                self._ensure_alignment_ids(canonical_lines, lines)
                raw = cached.get("raw_alignment")
                if not isinstance(raw, dict):
                    raise ValueError("Cached alignment is missing raw alignment JSON.")
                return lines, raw, True
            except (KeyError, TypeError, ValueError, RuntimeError):
                cached = None

        run_key = cache_key[:16]
        lines, raw_alignment, _ = self._aligner(
            request.youtube_url,
            canonical_lines,
            data_dir=request.data_dir / run_key,
            raw_json_path=request.output_path.parent / "alignment_raw.json",
        )
        self._ensure_alignment_ids(canonical_lines, lines)
        self._cache.put("alignment", cache_key, {
            "lines": [asdict(line) for line in lines],
            "raw_alignment": raw_alignment,
        })
        return lines, raw_alignment, False

    def _write_artifacts(
        self,
        request: PipelineRequest,
        aligned_lines: Sequence[AlignedLyricLine],
        cues: Sequence[BilingualCue],
        raw_alignment: Mapping[str, Any],
        record: LRCLIBLyricsRecord,
        canonical_hash: str,
        warnings: Sequence[Any],
        cache_hits: Mapping[str, bool],
    ) -> Mapping[str, Path]:
        output_dir = request.output_path.parent
        output_en_path = output_dir / "output_en.srt"
        raw_path = output_dir / "alignment_raw.json"
        translation_path = output_dir / "translation_result.json"
        metadata_path = output_dir / "pipeline_metadata.json"

        atomic_write_text(output_en_path, render_english_srt(aligned_lines))
        atomic_write_text(request.output_path, render_bilingual_srt(cues))
        atomic_write_json(raw_path, dict(raw_alignment))
        atomic_write_json(translation_path, {
            "lines": [
                {"id": cue.id, "translation": cue.translation}
                for cue in cues
            ],
        })
        atomic_write_json(metadata_path, {
            "lyrics": {
                "provider": "lrclib",
                "record_id": record.record_id,
                "title": record.title,
                "artist": record.artist,
                "duration": record.duration,
            },
            "canonical_hash": canonical_hash,
            "groq_prompt_version": GROQ_PROMPT_VERSION,
            "cache_hits": dict(cache_hits),
            "warnings": [asdict(warning) for warning in warnings],
        })
        return {
            "output_en_path": output_en_path,
            "output_bilingual_path": request.output_path,
            "translation_path": translation_path,
            "metadata_path": metadata_path,
        }

    @staticmethod
    def _canonical_payload(lines: Sequence[CanonicalLyricLine]) -> Mapping[str, Any]:
        return {"lines": [asdict(line) for line in lines]}

    def _groq_cache_key(self, request: PipelineRequest, canonical_hash: str) -> str:
        return self._cache.key_for({
            "canonical_hash": canonical_hash,
            "title": request.title,
            "artist": request.artist,
            "vibe": request.vibe,
            "max_vietnamese_cps": request.max_vietnamese_cps,
            "provider": self._cache_identity(self._groq_translator),
            "prompt_version": GROQ_PROMPT_VERSION,
        })

    @staticmethod
    def _cache_identity(value: Any) -> str:
        return str(getattr(value, "cache_identity", type(value).__name__))

    @staticmethod
    def _serialize_lyrics_record(record: LRCLIBLyricsRecord) -> Mapping[str, Any]:
        return {
            "record_id": record.record_id,
            "title": record.title,
            "artist": record.artist,
            "album": record.album,
            "duration": record.duration,
            "plain_lyrics": record.plain_lyrics,
        }

    @staticmethod
    def _deserialize_lyrics_record(value: Mapping[str, Any]) -> LRCLIBLyricsRecord:
        required_text = ("title", "artist", "plain_lyrics")
        if not all(isinstance(value.get(field), str) for field in required_text):
            raise RuntimeError("Cached lyrics record is invalid.")
        record_id = value.get("record_id")
        duration = value.get("duration")
        album = value.get("album")
        return LRCLIBLyricsRecord(
            record_id=record_id if isinstance(record_id, int) else None,
            title=str(value["title"]),
            artist=str(value["artist"]),
            album=album if isinstance(album, str) else None,
            duration=float(duration) if isinstance(duration, (int, float)) else None,
            plain_lyrics=str(value["plain_lyrics"]),
        )

    @staticmethod
    def _ensure_drafts_cover(
        lines: Sequence[CanonicalLyricLine],
        drafts: Mapping[int, str],
    ) -> None:
        if set(drafts) != {line.id for line in lines} or any(
            not isinstance(text, str) or not text.strip() for text in drafts.values()
        ):
            raise RuntimeError("Literal translation did not cover canonical IDs exactly.")

    @staticmethod
    def _ensure_translation_ids(
        lines: Sequence[CanonicalLyricLine],
        translations: Sequence[TranslationLine],
    ) -> None:
        expected_ids = [line.id for line in lines]
        actual_ids = [line.id for line in translations]
        if actual_ids != expected_ids or any(not line.translation.strip() for line in translations):
            raise RuntimeError("Groq translation IDs did not match canonical IDs exactly.")

    @staticmethod
    def _ensure_alignment_ids(
        lines: Sequence[CanonicalLyricLine],
        aligned: Sequence[AlignedLyricLine],
    ) -> None:
        expected_ids = [line.id for line in lines]
        actual_ids = [line.id for line in aligned]
        if actual_ids != expected_ids or any(line.end <= line.start for line in aligned):
            raise RuntimeError("Alignment IDs or timing did not match canonical lines exactly.")


def load_dotenv_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs without overriding explicit environment variables."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_pipeline_from_environment(cache_dir: Path) -> BilingualPipeline:
    """Construct concrete providers only after environment configuration has been validated."""
    load_dotenv_file(Path(".env"))
    timeout_seconds = float(os.environ.get("PROVIDER_TIMEOUT_SECONDS", "30"))
    retry_policy = RetryPolicy(attempts=int(os.environ.get("PROVIDER_MAX_RETRIES", "3")))
    groq_client = OpenAICompatibleJsonClient(
        endpoint=os.environ.get("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions"),
        api_key=_required_environment("GROQ_API_KEY"),
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        timeout_seconds=timeout_seconds,
        retry_policy=retry_policy,
    )
    literal_translator = TranslateLibraryLiteralTranslator()
    groq_translator = GroqTranslationService(groq_client)
    groq_translator.cache_identity = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    return BilingualPipeline(
        lyrics_lookup=lookup_lyrics_record,
        aligner=align_canonical_lines,
        literal_translator=literal_translator,
        groq_translator=groq_translator,
        cache=JsonFileCache(cache_dir),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create English--Vietnamese lyric subtitles from a YouTube song.",
    )
    parser.add_argument("--youtube-url", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--artist", required=True)
    parser.add_argument("--vibe", required=True)
    parser.add_argument("--output", type=Path, default=Path("output_bilingual.srt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".bilingual_cache"))
    parser.add_argument("--max-vietnamese-cps", type=float, default=DEFAULT_MAX_VIETNAMESE_CPS)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the configured pipeline without printing API credentials or raw provider payloads."""
    args = parse_args(argv)
    try:
        pipeline = build_pipeline_from_environment(args.cache_dir)
        result = asyncio.run(pipeline.run(PipelineRequest(
            youtube_url=args.youtube_url,
            title=args.title,
            artist=args.artist,
            vibe=args.vibe,
            output_path=args.output,
            data_dir=args.data_dir,
            max_vietnamese_cps=args.max_vietnamese_cps,
        )))
    except (RuntimeError, ValueError) as error:
        print(f"❌ {error}")
        return 1

    print(f"✅ English SRT: {result.output_en_path}")
    print(f"✅ Bilingual SRT: {result.output_bilingual_path}")
    print(f"✅ Metadata: {result.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
