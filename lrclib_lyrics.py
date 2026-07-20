"""Fetch plain song lyrics from LRCLIB with an interactive command-line flow."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"
DEFAULT_TIMEOUT_SECONDS = 10
USER_AGENT = "youtubers-assistant/1.0"

SEARCH_TEXT_TRANSLATION = str.maketrans({
    "\u00a0": " ",
    "\u2018": "'",
    "\u2019": "'",
    "\u201b": "'",
    "\u2032": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
})
MATCHING_TEXT_TRANSLATION = str.maketrans({
    "đ": "d",
    "Đ": "d",
    "ø": "o",
    "Ø": "o",
    "ł": "l",
    "Ł": "l",
    "ß": "ss",
    "æ": "ae",
    "Æ": "ae",
    "œ": "oe",
    "Œ": "oe",
})

ResponseOpener = Callable[..., object]


@dataclass(frozen=True)
class LRCLIBLyricsRecord:
    """The plain-lyrics metadata needed by the bilingual pipeline."""

    record_id: Optional[int]
    title: str
    artist: str
    album: Optional[str]
    duration: Optional[float]
    plain_lyrics: str


def sanitize_search_text(value: str) -> str:
    """Clean text for LRCLIB search without changing intentional capitalization."""
    normalized = unicodedata.normalize("NFKC", value).translate(SEARCH_TEXT_TRANSLATION)
    return " ".join(normalized.split())


def normalize_metadata(value: str) -> str:
    """Create a case-, accent-, whitespace-, and punctuation-insensitive match key."""
    matching_text = build_matching_text(value)
    words_only = matching_text.replace("&", " and ").replace("'", "")
    return re.sub(r"[_\W]+", " ", words_only).strip()


def build_matching_text(value: str) -> str:
    """Convert text to a case- and accent-insensitive representation."""
    normalized = unicodedata.normalize("NFKD", sanitize_search_text(value))
    translated = normalized.translate(MATCHING_TEXT_TRANSLATION)
    without_accents = "".join(
        character
        for character in translated
        if unicodedata.category(character) != "Mn"
    )
    return without_accents.casefold()


def normalize_compact_metadata(value: str) -> str:
    """Normalize stylized spelling such as P!nk, Ke$ha, and AC/DC."""
    matching_text = build_matching_text(value)
    with_stylized_letters = re.sub(
        r"(?<=[a-z0-9])!(?=[a-z0-9])",
        "i",
        matching_text,
    )
    with_stylized_letters = re.sub(
        r"(?<=[a-z0-9])\$(?=[a-z0-9])",
        "s",
        with_stylized_letters,
    )
    return re.sub(r"[^a-z0-9]+", "", with_stylized_letters)


def metadata_matches(record_value: str, expected_value: str) -> bool:
    """Match an artist name, including guarded fallback for stylized spellings."""
    if normalize_metadata(record_value) == normalize_metadata(expected_value):
        return True

    record_compact = normalize_compact_metadata(record_value)
    expected_compact = normalize_compact_metadata(expected_value)
    return (
        len(record_compact) >= 3
        and record_compact == expected_compact
    )


def validate_song_details(artist: str, title: str) -> tuple[str, str]:
    """Return clean artist and title values, or reject incomplete input."""
    cleaned_artist = sanitize_search_text(artist)
    cleaned_title = sanitize_search_text(title)
    if not cleaned_artist or not cleaned_title:
        raise ValueError("Tên bài hát và tên nghệ sĩ không được để trống.")
    return cleaned_artist, cleaned_title


def build_search_request(artist: str, title: str) -> Request:
    """Create the documented LRCLIB search request for a song and artist."""
    cleaned_artist, cleaned_title = validate_song_details(artist, title)
    query = urlencode({
        "track_name": cleaned_title,
        "artist_name": cleaned_artist,
    })
    return Request(
        f"{LRCLIB_SEARCH_URL}?{query}",
        headers={"User-Agent": USER_AGENT},
    )


def select_exact_record(
    records: Sequence[object],
    artist: str,
    title: str,
) -> Optional[LRCLIBLyricsRecord]:
    """Return metadata and plain lyrics for the exact title/artist match."""
    expected_title = normalize_metadata(title)

    for candidate in records:
        if not isinstance(candidate, Mapping):
            continue

        record_artist = candidate.get("artistName")
        record_title = candidate.get("trackName")
        plain_lyrics = candidate.get("plainLyrics")
        if not all(isinstance(value, str) for value in (
            record_artist,
            record_title,
            plain_lyrics,
        )):
            continue
        if not plain_lyrics.strip():
            continue
        if (
            metadata_matches(record_artist, artist)
            and normalize_metadata(record_title) == expected_title
        ):
            record_id = candidate.get("id")
            duration = candidate.get("duration")
            album = candidate.get("albumName")
            return LRCLIBLyricsRecord(
                record_id=record_id if isinstance(record_id, int) else None,
                title=record_title.strip(),
                artist=record_artist.strip(),
                album=album.strip() if isinstance(album, str) and album.strip() else None,
                duration=(
                    float(duration)
                    if isinstance(duration, (int, float)) and not isinstance(duration, bool)
                    else None
                ),
                plain_lyrics=plain_lyrics.strip(),
            )

    return None


def select_exact_match(
    records: Sequence[object],
    artist: str,
    title: str,
) -> Optional[str]:
    """Return plain lyrics only; retained for the original CLI API."""
    record = select_exact_record(records, artist, title)
    return record.plain_lyrics if record else None


def lookup_lyrics_record(
    artist: str,
    title: str,
    *,
    opener: ResponseOpener = urlopen,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Optional[LRCLIBLyricsRecord]:
    """Look up an exact song match in LRCLIB and return lyrics provenance."""
    if timeout <= 0:
        raise ValueError("Timeout phải lớn hơn 0 giây.")

    cleaned_artist, cleaned_title = validate_song_details(artist, title)
    request = build_search_request(cleaned_artist, cleaned_title)
    try:
        with opener(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
        records = json.loads(payload)
    except HTTPError as error:
        raise RuntimeError(f"LRCLIB trả về HTTP {error.code}.") from error
    except URLError as error:
        raise RuntimeError(f"Không thể kết nối đến LRCLIB: {error.reason}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("LRCLIB trả về dữ liệu không hợp lệ.") from error

    if not isinstance(records, list):
        raise RuntimeError("LRCLIB trả về dữ liệu không hợp lệ.")
    return select_exact_record(records, cleaned_artist, cleaned_title)


def lookup_plain_lyrics(
    artist: str,
    title: str,
    *,
    opener: ResponseOpener = urlopen,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Optional[str]:
    """Look up plain lyrics only; retained for the original CLI API."""
    record = lookup_lyrics_record(artist, title, opener=opener, timeout=timeout)
    return record.plain_lyrics if record else None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tìm plain lyrics theo tên bài hát và nghệ sĩ trên LRCLIB.",
    )
    parser.add_argument("--title", help="Tên bài hát; bỏ trống để nhập khi chạy.")
    parser.add_argument("--artist", help="Tên nghệ sĩ/tác giả; bỏ trống để nhập khi chạy.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Tùy chọn: lưu plain lyrics vào file UTF-8.",
    )
    return parser.parse_args(argv)


def prompt_for_value(label: str, input_func: Callable[[], str]) -> str:
    """Read a CLI value while keeping prompts out of the lyrics output stream."""
    print(label, end="", file=sys.stderr, flush=True)
    return input_func().strip()


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    input_func: Callable[[], str] = input,
) -> int:
    """Prompt for a song and artist, then print its plain lyrics."""
    args = parse_args(argv)
    try:
        title = args.title or prompt_for_value("Tên bài hát: ", input_func)
        artist = args.artist or prompt_for_value("Tên nghệ sĩ/tác giả: ", input_func)
    except (EOFError, KeyboardInterrupt):
        print("\n❌ Đã hủy nhập liệu.", file=sys.stderr)
        return 1

    try:
        lyrics = lookup_plain_lyrics(artist, title)
    except (RuntimeError, ValueError) as error:
        print(f"❌ {error}", file=sys.stderr)
        return 1

    if lyrics is None:
        print("❌ Không tìm thấy plain lyrics khớp chính xác với bài hát và nghệ sĩ này.", file=sys.stderr)
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{lyrics}\n", encoding="utf-8")

    print(lyrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
