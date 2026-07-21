from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

from bilingual_models import AlignedLyricLine, CanonicalLyricLine

# ------------------------------------------------------------
# Defaults
# ------------------------------------------------------------
DEFAULT_LYRICS = "lyrics.txt"
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_SRT = "output.srt"
DEFAULT_RAW_JSON = "alignment_raw.json"
DEFAULT_WHISPER_MODEL = "large"
DEFAULT_LANGUAGE = "en"
DEFAULT_DEMUCS_MODEL = "htdemucs"
DEFAULT_DEMUCS_SEGMENT = 7
DEFAULT_DEMUCS_OVERLAP = 0.25
DEFAULT_YOUTUBE_CLIENT = "WEB"
LOCAL_VIDEO_FALLBACK_NAME = "video.mp4"

# Activity / subtitle clamping (from notebook)
DEFAULT_TOP_DB = 30
DEFAULT_MAX_PHRASE_GAP = 0.55
DEFAULT_PAD = 0.06
DEFAULT_MIN_ACTIVITY = 0.04
DEFAULT_SUSPECT_DURATION = 5.0

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}


class AlignmentError(RuntimeError):
    """Raised when canonical lyrics cannot be mapped to safe audio timings."""


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def run_command(command: Sequence[str], error_label: str) -> subprocess.CompletedProcess:
    """Run a command; raise RuntimeError with stderr/stdout on failure."""
    try:
        return subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip() or (exc.stdout or "").strip() or str(exc)
        raise RuntimeError(f"{error_label}: {detail}") from exc


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        sys.exit("❌ ffmpeg not found in PATH. Install ffmpeg first.")
    if not shutil.which("ffprobe"):
        sys.exit("❌ ffprobe not found in PATH. Install ffmpeg first.")


def clean_lyrics(raw: str) -> str:
    """Remove empty lines, section tags, and Genius boilerplate."""
    lines = raw.splitlines()
    cleaned: List[str] = []
    skip_block = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Section tags always end a Genius recommendation block, then are dropped.
        if re.fullmatch(r"\[.*?\]", line):
            skip_block = False
            continue

        if line.lower() == "you might also like":
            skip_block = True
            continue

        if skip_block:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def norm_words(text: str) -> List[str]:
    """Normalize text to alphanumeric word tokens (stable-ts style)."""
    return re.findall(r"'?[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())


def srt_time(sec: float) -> str:
    """Convert seconds to SRT timestamp HH:MM:SS,mmm."""
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
        if s == 60:
            s = 0
            m += 1
            if m == 60:
                m = 0
                h += 1
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def overlap_len(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-16")


# ------------------------------------------------------------
# YouTube input
# ------------------------------------------------------------

def is_youtube_url(value: str) -> bool:
    """Return whether value is an HTTP(S) URL for a single YouTube video."""
    parsed = urlparse(value.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in YOUTUBE_HOSTS:
        return False

    path = parsed.path.rstrip("/")
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = path.lstrip("/")
        return bool(video_id) and "/" not in video_id

    if path == "/watch":
        return bool(parse_qs(parsed.query).get("v", [""])[0])

    if path.startswith("/shorts/"):
        video_id = path.removeprefix("/shorts/")
        return bool(video_id) and "/" not in video_id

    return False

def find_local_video_fallback() -> Optional[Path]:
    """Find a user-uploaded ``video.mp4`` beside the runner or this script."""
    candidates = (Path.cwd() / LOCAL_VIDEO_FALLBACK_NAME, Path(__file__).parent / LOCAL_VIDEO_FALLBACK_NAME)
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def use_local_video_fallback(out_path: Path, reason: str) -> Path:
    """Transcode a user-provided MP4 after remote YouTube media is unavailable."""
    local_video = find_local_video_fallback()
    if local_video:
        print(f"⚠️  {reason} Using local fallback: {local_video}")
        return convert_to_mp3(local_video, out_path)
    locations = ", ".join(str(path) for path in (
        Path.cwd() / LOCAL_VIDEO_FALLBACK_NAME,
        Path(__file__).parent / LOCAL_VIDEO_FALLBACK_NAME,
    ))
    raise RuntimeError(f"{reason} No uploaded video.mp4 was found at: {locations}")


def download_youtube(url: str, out_path: Path) -> Path:
    """Use audio stream, remote MP4, then an uploaded ``video.mp4`` in that order."""
    try:
        from pytubefix import YouTube
        from pytubefix.exceptions import BotDetection, PytubeFixError
    except ImportError:
        raise RuntimeError("pytubefix is not installed. Run: pip install -r requirements.txt")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"⏳ Downloading audio from YouTube → {out_path}")
    try:
        # The WEB client lets pytubefix generate a PO token automatically when needed.
        yt = YouTube(url, DEFAULT_YOUTUBE_CLIENT)
        audio_stream = yt.streams.filter(only_audio=True).first()
    except BotDetection as error:
        return use_local_video_fallback(
            out_path,
            "YouTube blocked this runtime as automated traffic before stream lookup.",
        )
    except (PytubeFixError, HTTPError, URLError) as error:
        return use_local_video_fallback(out_path, f"YouTube stream lookup failed: {error}.")

    if audio_stream:
        try:
            downloaded_audio = audio_stream.download(
                output_path=str(out_path.parent),
                filename="downloaded_audio",
            )
            if not downloaded_audio:
                raise RuntimeError("Audio-only stream produced an empty file.")
            raw_audio_path = Path(downloaded_audio)
            if not raw_audio_path.is_file() or raw_audio_path.stat().st_size == 0:
                raise RuntimeError("Audio-only stream produced an empty file.")
            return convert_to_mp3(raw_audio_path, out_path)
        except (PytubeFixError, HTTPError, URLError) as error:
            print(f"⚠️  Audio-only download failed ({error}); trying video fallback.")
        except RuntimeError as error:
            if "Audio-only stream produced" not in str(error):
                raise
            print(f"⚠️  {error} Trying video fallback.")
    else:
        print("⚠️  No audio-only stream was available; trying video fallback.")

    try:
        video_stream = (
            yt.streams.filter(progressive=True, file_extension="mp4")
            .order_by("resolution")
            .asc()
            .first()
        )
    except BotDetection as error:
        return use_local_video_fallback(
            out_path,
            "YouTube blocked this runtime as automated traffic before the video fallback.",
        )
    except (PytubeFixError, HTTPError, URLError) as error:
        return use_local_video_fallback(out_path, f"YouTube video fallback lookup failed: {error}.")
    if not video_stream:
        return use_local_video_fallback(
            out_path,
            "YouTube did not offer an audio-only stream or a progressive MP4 fallback.",
        )

    try:
        downloaded_video = video_stream.download(
            output_path=str(out_path.parent),
            filename="downloaded_video.mp4",
        )
    except (PytubeFixError, HTTPError, URLError) as error:
        return use_local_video_fallback(out_path, f"YouTube video fallback download failed: {error}.")
    if not downloaded_video:
        return use_local_video_fallback(out_path, "YouTube video fallback produced an empty file.")
    video_path = Path(downloaded_video)
    if not video_path.is_file() or video_path.stat().st_size == 0:
        return use_local_video_fallback(out_path, "YouTube video fallback produced an empty file.")
    return convert_to_mp3(video_path, out_path)


def resolve_youtube_source(youtube_url: str, data_dir: Path) -> Path:
    """Download a validated YouTube URL and return a normalized MP3 source."""
    if not is_youtube_url(youtube_url):
        sys.exit("❌ Input must be a valid YouTube URL for a single video.")
    return download_youtube(youtube_url, data_dir / "yt_source.mp3")


def convert_to_mp3(source: Path, output: Path) -> Path:
    """Extract or transcode a downloaded media file into MP3 for the rest of the pipeline."""
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(source),
        "-vn",
        "-codec:a", "libmp3lame",
        "-q:a", "2",
        str(output),
    ]
    print(f"⏳ Converting {source.name} → {output.name} (MP3)…")
    run_command(command, "FFmpeg MP3 conversion failed")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"MP3 conversion failed: empty output {output}")
    print(f"✅ MP3 source: {output}")
    return output


def convert_audio(
    source: Path,
    output: Path,
    *,
    channels: int,
    sample_rate: int,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(source),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-f", "wav",
        str(output),
    ]
    print(f"⏳ Converting {source.name} → {output.name} ({channels}ch / {sample_rate} Hz)…")
    run_command(cmd, "FFmpeg conversion failed")
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"Conversion failed: empty output {output}")
    print(f"✅ Converted: {output}")
    return output


# ------------------------------------------------------------
# Demucs
# ------------------------------------------------------------

def separate_vocals(
    demucs_input: Path,
    vocal_out: Path,
    *,
    out_dir: Path,
    model_name: str = DEFAULT_DEMUCS_MODEL,
    segment: int = DEFAULT_DEMUCS_SEGMENT,
    overlap: float = DEFAULT_DEMUCS_OVERLAP,
) -> Path:
    """Run Demucs two-stem vocal separation and convert to 16 kHz mono."""
    if not demucs_input.exists():
        raise FileNotFoundError(f"Demucs input missing: {demucs_input}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print("⏳ Running Demucs (may take a few minutes)…")
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", model_name,
        "--segment", str(segment),
        "--overlap", str(overlap),
        "-o", str(out_dir),
        str(demucs_input),
    ]
    run_command(cmd, "Demucs failed")

    src = out_dir / model_name / demucs_input.stem / "vocals.wav"
    if not src.exists():
        found = list(out_dir.glob("**/vocals.wav"))
        if not found:
            raise FileNotFoundError("Demucs finished but no vocals.wav was found.")
        src = found[0]
        print(f"ℹ️  Found vocals at: {src}")

    vocal_out.parent.mkdir(parents=True, exist_ok=True)
    print("⏳ Converting Demucs vocals → 16 kHz mono…")
    convert_audio(src, vocal_out, channels=1, sample_rate=16000)
    print(f"✅ Vocals: {vocal_out}")
    return vocal_out


# ------------------------------------------------------------
# Alignment
# ------------------------------------------------------------

def load_lyrics_lines(lyrics_path: Path) -> List[str]:
    if not lyrics_path.is_file():
        sys.exit(f"❌ Lyrics file not found: {lyrics_path.resolve()}")
    raw = read_text_file(lyrics_path)
    cleaned = clean_lyrics(raw)
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        sys.exit("❌ Lyrics file is empty after cleaning.")
    return lines


def run_alignment(
    audio_path: Path,
    lyrics_text: str,
    *,
    model_size: str,
    language: str,
    raw_json_path: Path,
) -> dict:
    try:
        import stable_whisper
    except ImportError:
        sys.exit("❌ stable-whisper not installed. Run: pip install -U stable-ts")

    print(f"⏳ Loading Whisper model ({model_size})…")
    model = stable_whisper.load_model(model_size)

    print("⏳ Forced alignment…")
    result = model.align(
        str(audio_path),
        lyrics_text,
        language=language,
        stream=False,
    )

    raw_json_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_as_json(str(raw_json_path))
    print(f"✅ Raw alignment JSON: {raw_json_path}")
    return result.to_dict()


def extract_words(raw_result: dict) -> List[Dict[str, float | str]]:
    all_words: List[Dict[str, float | str]] = []
    for seg in raw_result.get("segments", []):
        for w in seg.get("words", []) or []:
            word = (w.get("word") or "").strip()
            if not word:
                continue
            if "start" not in w or "end" not in w:
                continue
            all_words.append({
                "word": word,
                "start": float(w["start"]),
                "end": float(w["end"]),
            })
    return all_words


# ------------------------------------------------------------
# Activity-based timing clamp
# ------------------------------------------------------------

def build_activity_intervals(
    audio_path: Path,
    *,
    top_db: float,
    min_activity: float,
) -> List[Tuple[float, float]]:
    import librosa

    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    raw = librosa.effects.split(
        y,
        top_db=top_db,
        frame_length=2048,
        hop_length=256,
    )
    intervals: List[Tuple[float, float]] = []
    for a, b in raw:
        start = a / sr
        end = b / sr
        if end - start >= min_activity:
            intervals.append((start, end))
    return intervals


def pick_best_activity_cluster(
    start: float,
    end: float,
    activity_intervals: Sequence[Tuple[float, float]],
    *,
    max_phrase_gap: float,
    pad: float,
    min_activity: float,
) -> Tuple[float, float, bool]:
    """Within lyric window, pick the densest vocal cluster and pad lightly."""
    clipped: List[Tuple[float, float]] = []
    for a_start, a_end in activity_intervals:
        if overlap_len(start, end, a_start, a_end) <= 0:
            continue
        c_start = max(start, a_start)
        c_end = min(end, a_end)
        if c_end - c_start >= min_activity:
            clipped.append((c_start, c_end))

    if not clipped:
        return start, end, False

    clipped.sort()
    clusters: List[Tuple[float, float]] = []
    cur_start, cur_end = clipped[0]
    for a_start, a_end in clipped[1:]:
        if a_start - cur_end <= max_phrase_gap:
            cur_end = max(cur_end, a_end)
        else:
            clusters.append((cur_start, cur_end))
            cur_start, cur_end = a_start, a_end
    clusters.append((cur_start, cur_end))

    def cluster_score(cluster: Tuple[float, float]) -> float:
        c_start, c_end = cluster
        vocal_sum = sum(
            overlap_len(c_start, c_end, a_start, a_end)
            for a_start, a_end in clipped
        )
        duration = c_end - c_start
        return vocal_sum - 0.10 * duration

    best_start, best_end = max(clusters, key=cluster_score)
    best_start = max(start, best_start - pad)
    best_end = min(end, best_end + pad)

    if best_end <= best_start or best_end - best_start < 0.12:
        return start, end, False
    return best_start, best_end, True


def build_line_items(
    lyrics_lines: Sequence[str],
    all_words: Sequence[Dict[str, float | str]],
) -> Tuple[List[dict], int, List[tuple]]:
    line_items: List[dict] = []
    word_i = 0
    bad_lines: List[tuple] = []

    for line_i, line in enumerate(lyrics_lines, start=1):
        n = len(norm_words(line))
        if n == 0:
            continue
        line_words = all_words[word_i : word_i + n]
        if len(line_words) < n:
            bad_lines.append((line_i, line, "Not enough words"))
            break
        raw_start = min(float(w["start"]) for w in line_words)
        raw_end = max(float(w["end"]) for w in line_words)
        line_items.append({
            "line_i": line_i,
            "text": line,
            "raw_start": raw_start,
            "raw_end": raw_end,
            "word_start": word_i,
            "word_end": word_i + n,
        })
        word_i += n

    return line_items, word_i, bad_lines


def build_srt_blocks(
    line_items: Sequence[dict],
    activity_intervals: Sequence[Tuple[float, float]],
    *,
    max_phrase_gap: float,
    pad: float,
    min_activity: float,
    suspect_duration: float,
    clamp: bool,
) -> Tuple[List[str], List[tuple]]:
    timed_items, suspects = build_timed_line_items(
        line_items,
        activity_intervals,
        max_phrase_gap=max_phrase_gap,
        pad=pad,
        min_activity=min_activity,
        suspect_duration=suspect_duration,
        clamp=clamp,
    )
    srt_blocks = [
        f"{index}\n"
        f"{srt_time(float(item['start']))} --> {srt_time(float(item['end']))}\n"
        f"{item['text']}\n"
        for index, item in enumerate(timed_items, start=1)
    ]
    return srt_blocks, suspects


def build_timed_line_items(
    line_items: Sequence[dict],
    activity_intervals: Sequence[Tuple[float, float]],
    *,
    max_phrase_gap: float,
    pad: float,
    min_activity: float,
    suspect_duration: float,
    clamp: bool,
) -> Tuple[List[dict], List[tuple]]:
    """Apply existing overlap/clamp logic and retain precise timestamps per line."""
    timed_items: List[dict] = []
    suspects: List[tuple] = []

    for idx, item in enumerate(line_items):
        line = item["text"]
        start = float(item["raw_start"])
        end = float(item["raw_end"])

        if idx + 1 < len(line_items):
            next_start = float(line_items[idx + 1]["raw_start"])
            if next_start > start:
                end = min(end, next_start - 0.03)
        if idx > 0:
            prev_end = float(line_items[idx - 1]["raw_end"])
            if prev_end < end:
                start = max(start, prev_end + 0.01)

        if end <= start:
            start = float(item["raw_start"])
            end = float(item["raw_end"])

        if clamp and activity_intervals:
            new_start, new_end, _ = pick_best_activity_cluster(
                start,
                end,
                activity_intervals,
                max_phrase_gap=max_phrase_gap,
                pad=pad,
                min_activity=min_activity,
            )
        else:
            new_start, new_end = start, end

        duration = new_end - new_start
        if duration > suspect_duration:
            suspects.append((
                item["line_i"],
                srt_time(new_start),
                srt_time(new_end),
                round(duration, 2),
                line,
            ))

        timed_items.append({
            **item,
            "start": new_start,
            "end": new_end,
        })

    return timed_items, suspects


def map_timed_items_to_canonical(
    canonical_lines: Sequence[CanonicalLyricLine],
    timed_items: Sequence[Mapping[str, object]],
) -> Tuple[AlignedLyricLine, ...]:
    """Assign timings positionally to canonical IDs and reject any partial alignment."""
    if len(canonical_lines) != len(timed_items):
        raise AlignmentError(
            "Alignment did not produce a timing window for every canonical lyric line.",
        )

    aligned_lines = []
    for canonical_line, timed_item in zip(canonical_lines, timed_items):
        text = timed_item.get("text")
        start = timed_item.get("start")
        end = timed_item.get("end")
        if text != canonical_line.source:
            raise AlignmentError(
                f"Alignment text drifted at canonical line {canonical_line.id}.",
            )
        if isinstance(start, bool) or isinstance(end, bool):
            raise AlignmentError(f"Alignment timing is invalid at line {canonical_line.id}.")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise AlignmentError(f"Alignment timing is invalid at line {canonical_line.id}.")
        if float(end) <= float(start):
            raise AlignmentError(f"Alignment has non-positive duration at line {canonical_line.id}.")
        aligned_lines.append(AlignedLyricLine(
            id=canonical_line.id,
            source=canonical_line.source,
            start=float(start),
            end=float(end),
        ))
    return tuple(aligned_lines)


def ensure_ffmpeg_available() -> None:
    """Raise an import-safe error when the runtime media dependency is unavailable."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise AlignmentError("ffmpeg and ffprobe must be available on PATH.")


def align_canonical_lines(
    youtube_url: str,
    canonical_lines: Sequence[CanonicalLyricLine],
    *,
    data_dir: Path,
    raw_json_path: Path,
    model_size: str = DEFAULT_WHISPER_MODEL,
    language: str = DEFAULT_LANGUAGE,
    use_demucs: bool = True,
    clamp: bool = True,
    top_db: float = DEFAULT_TOP_DB,
    max_phrase_gap: float = DEFAULT_MAX_PHRASE_GAP,
    pad: float = DEFAULT_PAD,
    min_activity: float = DEFAULT_MIN_ACTIVITY,
    suspect_duration: float = DEFAULT_SUSPECT_DURATION,
    demucs_segment: int = DEFAULT_DEMUCS_SEGMENT,
) -> Tuple[Tuple[AlignedLyricLine, ...], dict, Tuple[tuple, ...]]:
    """Run the existing media/alignment pipeline against immutable canonical lyrics."""
    if not canonical_lines:
        raise AlignmentError("At least one canonical lyric line is required.")
    canonical_ids = tuple(line.id for line in canonical_lines)
    if len(set(canonical_ids)) != len(canonical_ids):
        raise AlignmentError("Canonical lyric IDs must be unique.")
    if not is_youtube_url(youtube_url):
        raise AlignmentError("Input must be a valid YouTube URL for a single video.")

    ensure_ffmpeg_available()
    data_dir.mkdir(parents=True, exist_ok=True)
    source = download_youtube(youtube_url, data_dir / "yt_source.mp3")
    whisper_wav = convert_audio(
        source,
        data_dir / "yt_input.wav",
        channels=1,
        sample_rate=16000,
    )

    activity_audio = whisper_wav
    vocal_wav = data_dir / "yt_vocals.wav"
    if use_demucs:
        demucs_wav = convert_audio(
            source,
            data_dir / "yt_demucs.wav",
            channels=2,
            sample_rate=44100,
        )
        try:
            activity_audio = separate_vocals(
                demucs_wav,
                vocal_wav,
                out_dir=data_dir / "demucs_separated",
                segment=demucs_segment,
            )
        except Exception as exc:
            print(f"⚠️  Demucs failed ({exc}); falling back to full mix for activity.")

    lyrics_text = "\n".join(line.source for line in canonical_lines)
    raw_result = run_alignment(
        whisper_wav,
        lyrics_text,
        model_size=model_size,
        language=language,
        raw_json_path=raw_json_path,
    )
    line_items, _, bad_lines = build_line_items(
        tuple(line.source for line in canonical_lines),
        extract_words(raw_result),
    )
    if bad_lines or len(line_items) != len(canonical_lines):
        raise AlignmentError("Forced alignment could not cover every canonical lyric line.")

    activity_intervals: List[Tuple[float, float]] = []
    active_clamp = clamp
    if active_clamp:
        try:
            activity_intervals = build_activity_intervals(
                activity_audio,
                top_db=top_db,
                min_activity=min_activity,
            )
        except ImportError:
            active_clamp = False
        except Exception as exc:
            print(f"⚠️  Activity detection failed ({exc}); using raw windows.")
            active_clamp = False

    timed_items, suspects = build_timed_line_items(
        line_items,
        activity_intervals,
        max_phrase_gap=max_phrase_gap,
        pad=pad,
        min_activity=min_activity,
        suspect_duration=suspect_duration,
        clamp=active_clamp,
    )
    return map_timed_items_to_canonical(canonical_lines, timed_items), raw_result, tuple(suspects)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Forced-align lyrics from a YouTube video and write line-by-line SRT.",
    )
    p.add_argument(
        "youtube_url",
        nargs="?",
        metavar="YOUTUBE_URL",
        help="YouTube video URL. If omitted, you will be prompted.",
    )
    p.add_argument(
        "-l", "--lyrics",
        default=DEFAULT_LYRICS,
        help=f"Lyrics text file (default: {DEFAULT_LYRICS})",
    )
    p.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT_SRT,
        help=f"Output SRT path (default: {DEFAULT_OUTPUT_SRT})",
    )
    p.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Working directory for intermediate audio (default: {DEFAULT_DATA_DIR})",
    )
    p.add_argument(
        "--raw-json",
        default=DEFAULT_RAW_JSON,
        help=f"Path to save raw alignment JSON (default: {DEFAULT_RAW_JSON})",
    )
    p.add_argument(
        "-m", "--model",
        default=DEFAULT_WHISPER_MODEL,
        help=f"Whisper model size (default: {DEFAULT_WHISPER_MODEL})",
    )
    p.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Language code for alignment (default: {DEFAULT_LANGUAGE})",
    )
    p.add_argument(
        "--no-demucs",
        action="store_true",
        help="Skip Demucs vocal separation (activity uses full mix).",
    )
    p.add_argument(
        "--no-clamp",
        action="store_true",
        help="Skip librosa activity clamping; use raw alignment windows.",
    )
    p.add_argument("--top-db", type=float, default=DEFAULT_TOP_DB)
    p.add_argument("--max-phrase-gap", type=float, default=DEFAULT_MAX_PHRASE_GAP)
    p.add_argument("--pad", type=float, default=DEFAULT_PAD)
    p.add_argument("--min-activity", type=float, default=DEFAULT_MIN_ACTIVITY)
    p.add_argument("--suspect-duration", type=float, default=DEFAULT_SUSPECT_DURATION)
    p.add_argument(
        "--demucs-segment",
        type=int,
        default=DEFAULT_DEMUCS_SEGMENT,
        help="Demucs --segment (default 7; higher may OOM on some GPUs).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    require_ffmpeg()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    youtube_url = (args.youtube_url or "").strip()
    if not youtube_url:
        youtube_url = input("Enter YouTube video URL: ").strip()
    if not youtube_url:
        sys.exit("❌ No YouTube video URL provided.")

    try:
        source = resolve_youtube_source(youtube_url, data_dir)
    except RuntimeError as error:
        sys.exit(f"❌ {error}")

    # Whisper / alignment audio: mono 16 kHz
    whisper_wav = data_dir / "yt_input.wav"
    convert_audio(source, whisper_wav, channels=1, sample_rate=16000)

    # Lyrics
    lyrics_lines = load_lyrics_lines(Path(args.lyrics))
    lyrics_text = "\n".join(lyrics_lines)
    clean_path = data_dir / "lyrics_clean.txt"
    clean_path.write_text(lyrics_text + "\n", encoding="utf-8")
    print(f"✅ Lyrics lines: {len(lyrics_lines)} (saved {clean_path})")

    # Optional Demucs for activity audio
    vocal_wav = data_dir / "yt_vocals.wav"
    activity_audio = whisper_wav
    if not args.no_demucs:
        demucs_wav = data_dir / "yt_demucs.wav"
        convert_audio(source, demucs_wav, channels=2, sample_rate=44100)
        try:
            separate_vocals(
                demucs_wav,
                vocal_wav,
                out_dir=data_dir / "demucs_separated",
                segment=args.demucs_segment,
            )
            activity_audio = vocal_wav
        except Exception as exc:
            print(f"⚠️  Demucs failed ({exc}); falling back to full mix for activity.")
            activity_audio = whisper_wav
    elif vocal_wav.exists():
        activity_audio = vocal_wav
        print(f"ℹ️  Using existing vocals: {vocal_wav}")

    # Alignment
    raw_result = run_alignment(
        whisper_wav,
        lyrics_text,
        model_size=args.model,
        language=args.language,
        raw_json_path=Path(args.raw_json),
    )
    all_words = extract_words(raw_result)
    expected_words = sum(len(norm_words(line)) for line in lyrics_lines)
    print(f"Words from alignment: {len(all_words)}")
    print(f"Words expected from lyrics: {expected_words}")
    if abs(expected_words - len(all_words)) > 5:
        print("⚠️  Word count mismatch is large; SRT line mapping may drift.")

    # Activity + SRT
    activity_intervals: List[Tuple[float, float]] = []
    clamp = not args.no_clamp
    if clamp:
        print(f"⏳ Building activity intervals from {activity_audio}…")
        try:
            activity_intervals = build_activity_intervals(
                activity_audio,
                top_db=args.top_db,
                min_activity=args.min_activity,
            )
            print(f"Activity intervals: {len(activity_intervals)}")
        except ImportError:
            print("⚠️  librosa not installed; skipping activity clamp. "
                  "Install with: pip install librosa")
            clamp = False
        except Exception as exc:
            print(f"⚠️  Activity detection failed ({exc}); using raw windows.")
            clamp = False

    line_items, word_i, bad_lines = build_line_items(lyrics_lines, all_words)
    srt_blocks, suspects = build_srt_blocks(
        line_items,
        activity_intervals,
        max_phrase_gap=args.max_phrase_gap,
        pad=args.pad,
        min_activity=args.min_activity,
        suspect_duration=args.suspect_duration,
        clamp=clamp,
    )

    out_path = Path(args.output)
    out_path.write_text("\n".join(srt_blocks) + "\n", encoding="utf-8")
    print(f"✅ SRT written: {out_path.resolve()}")
    print(f"Used lyric lines: {len(srt_blocks)} / {len(lyrics_lines)}")
    print(f"Used words: {word_i} / {len(all_words)}")
    print(f"TOP_DB={args.top_db} MAX_PHRASE_GAP={args.max_phrase_gap} PAD={args.pad}")

    if bad_lines:
        print("\nBad lines:")
        for item in bad_lines[:10]:
            print(" ", item)
    if suspects:
        print("\nSuspect long lines:")
        for s in suspects[:20]:
            print(" ", s)


if __name__ == "__main__":
    main()
