"""
Download audio from YouTube and transcribe to English using faster-whisper (GPU).

Usage:
    Single video:  python 1_transcribe.py --url "https://www.youtube.com/watch?v=..."
    Playlist:      python 1_transcribe.py --url "https://www.youtube.com/playlist?list=..."
    From file:     python 1_transcribe.py --file urls.txt
    Cleanup audio: python 1_transcribe.py --file urls.txt --cleanup-audio
"""

import os
# Must be set before any OpenMP-linked library loads (PyTorch + CTranslate2 both bundle libiomp5md.dll on Windows).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ctranslate2 calls LoadLibrary() which searches PATH (not AddDllDirectory entries).
# pip-installed nvidia packages land in site-packages/nvidia/*/bin/ — prepend them to PATH.
import glob as _glob, sys as _sys
_nvidia_bins = set(
    os.path.dirname(p)
    for p in _glob.glob(
        os.path.join(_sys.prefix, "Lib", "site-packages", "nvidia", "**", "*.dll"),
        recursive=True,
    )
)
if _nvidia_bins:
    os.environ["PATH"] = os.pathsep.join(_nvidia_bins) + os.pathsep + os.environ.get("PATH", "")
del _glob, _sys, _nvidia_bins

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yt_dlp
from faster_whisper import BatchedInferencePipeline, WhisperModel
from tqdm import tqdm

# yt-dlp's CLI reads --js-runtimes and --remote-components from a config file, but
# the Python YoutubeDL class doesn't — it needs these as ydl_opts. Use yt-dlp's own
# option parser to convert CLI flags into the correct internal dict format, then
# diff against default opts to extract only the keys these flags actually changed.
def _build_js_runtime_opts() -> dict:
    flagged = yt_dlp.parse_options([
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
    ]).ydl_opts
    default = yt_dlp.parse_options([]).ydl_opts
    return {k: v for k, v in flagged.items() if default.get(k) != v}

_JS_RUNTIME_OPTS = _build_js_runtime_opts()

AUDIO_DIR     = Path("audio")
TRANSCRIPT_DIR = Path("transcripts")
WHISPER_MODEL = "large-v3"
DEVICE        = "cuda"
COMPUTE_TYPE  = "int8_float16"
BATCH_SIZE    = 8     # audio chunks processed in parallel on GPU; reduce to 4 if OOM errors occur

DOWNLOAD_RETRIES = 3

# Cookie source: reads live from Edge's cookie store each run so it never goes stale.
# Edge must be installed and you must be logged into YouTube in it.
# To fall back to a static file instead, change to: COOKIES_FROM_BROWSER = None
# and set COOKIES_FILE = Path("cookies.txt").
COOKIES_FROM_BROWSER = "firefox"

# Primes Whisper's decoder with domain vocabulary — reduces errors on proper nouns and Sanskrit terms.
# When new mis-transcribed terms surface from real usage, add them here so future runs are consistent.
INITIAL_PROMPT = (
    "This is a spiritual discourse (satsang) by Premanand Maharajji in Hindi. "
    "Places and deities: Radha Rani, Shri Krishna, Vrindavan, Braj, Yamuna, Golok, Bhagavan, Hari. "
    "Saints and lineages: Kabir Das, Tulsidas, Surdas, Mirabai, Chaitanya, Hit Harivansh. "
    "Core practice terms: Satsang, Bhakti, Prema, Japa, Kirtan, Naam, Seva, Sharanagati, Sumiran. "
    "Doctrinal terms: Dharma, Moksha, Karma, Maya, Samsara, Vairagya, Viveka, Ananda, Atma, Paramatma, "
    "Brahma, Brahman, Brahmaswaroopam, Paramananda, Sat-Chit-Ananda, Bhagavath. "
    "Roles: Guru, Shishya, Sadhak, Bhakt, Maharaj, Mantra, Lila."
)


def setup_dirs():
    AUDIO_DIR.mkdir(exist_ok=True)
    TRANSCRIPT_DIR.mkdir(exist_ok=True)


def already_transcribed(video_id: str) -> bool:
    return (TRANSCRIPT_DIR / f"{video_id}.json").exists()


def normalize_url(url: str) -> str:
    """
    Normalize to a canonical form so yt-dlp behaves predictably:
    - URL with &list= → pure playlist URL  (process whole playlist)
    - Plain video URL  → unchanged         (process single video)
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    if "list" in params:
        return f"https://www.youtube.com/playlist?list={params['list'][0]}"
    return url


def _video_id_from_url(url: str) -> str | None:
    """Extract the 11-character video ID from a watch URL, or None if not a single-video URL."""
    params = parse_qs(urlparse(url).query)
    vid = params.get("v", [None])[0]
    return vid if vid and len(vid) == 11 else None


def enumerate_entries(url: str) -> list[dict]:
    """
    Cheap metadata-only pass: returns [{id, title, url}] for a single video or all
    videos in a playlist, without downloading anything.
    """
    target = normalize_url(url)

    # For single video URLs, parse the ID directly — no yt-dlp call needed and no
    # risk of format-resolution errors (which occur even with extract_flat for single videos).
    vid = _video_id_from_url(target)
    if vid:
        return [{"id": vid, "title": "untitled", "url": f"https://www.youtube.com/watch?v={vid}"}]

    # Playlist: yt-dlp with extract_flat only fetches IDs/titles, no format resolution.
    ydl_opts = {
        "extract_flat": True,
        "ignoreerrors": True,   # skip unavailable videos instead of aborting
        "quiet": True,
        "no_warnings": True,
        **({"cookiesfrombrowser": (COOKIES_FROM_BROWSER,)} if COOKIES_FROM_BROWSER else {}),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(target, download=False)

    if not info:
        return []

    out = []
    for e in (info.get("entries") or []):
        if not e or not e.get("id"):
            continue
        vid = e["id"]
        if len(vid) != 11:
            continue
        out.append({
            "id":    vid,
            "title": e.get("title", "untitled"),
            "url":   f"https://www.youtube.com/watch?v={vid}",
        })
    return out


def _build_download_opts(nocookie_fallback: bool = False) -> dict:
    """
    Build ydl_opts for audio download.

    Normal mode: passes browser cookies, no extractor_args — uses yt-dlp's default web
    client. Requires a JS runtime (Node.js) for YouTube's n-challenge decryption.
    Install Node.js to make this the only path needed.

    nocookie_fallback: unauthenticated web_creator client; bypasses n-challenge (YouTube
    skips it for anonymous sessions) and yields format 18 (360p mp4). Sufficient for
    audio transcription; won't work for members-only videos.
    """
    base = {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best[acodec!=none]/best",
        "outtmpl": str(AUDIO_DIR / "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "quiet": True,
        "no_warnings": True,
        **_JS_RUNTIME_OPTS,
    }
    if nocookie_fallback:
        base["extractor_args"] = {"youtube": {"player_client": ["web_creator"]}}
    if COOKIES_FROM_BROWSER:
        base["cookiesfrombrowser"] = (COOKIES_FROM_BROWSER,)
    return base


def download_one(video_url: str) -> dict | None:
    """Download a single video's audio (with retries). Returns full metadata dict."""

    def _try_download(ydl_opts: dict, hard_skip_on_format_error: bool = False) -> dict | None | bool:
        """Returns metadata dict on success, False on hard skip, None on retriable failure."""
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                vid = info["id"]
                return {
                    "title":       info.get("title", "untitled"),
                    "video_id":    vid,
                    "audio_path":  str(AUDIO_DIR / f"{vid}.mp3"),
                    "url":         f"https://www.youtube.com/watch?v={vid}",
                    "channel":     info.get("uploader", ""),
                    "upload_date": info.get("upload_date", ""),
                }
            except Exception as e:
                err = str(e)
                if "Video unavailable" in err or "Please sign in" in err:
                    return False  # hard skip regardless of pass
                if "Requested format is not available" in err:
                    # In Pass 1 this likely means no JS runtime — fall through to Pass 2.
                    # In Pass 2 the video genuinely has no downloadable format.
                    return False if hard_skip_on_format_error else None
                wait = 2 ** attempt
                print(f"    Attempt {attempt}/{DOWNLOAD_RETRIES} failed ({e}); retrying in {wait}s...")
                time.sleep(wait)
        return None  # retriable failure exhausted

    # Pass 1: normal download (with cookies; requires JS runtime for n-challenge).
    result = _try_download(_build_download_opts(), hard_skip_on_format_error=False)
    if result is False:
        print(f"    Skipping (not available): {video_url}")
        return None
    if result is not None:
        return result

    # Pass 2: unauthenticated web_creator — bypasses n-challenge, yields format 18.
    # Reached when the normal path fails due to missing JS runtime.
    print(f"    Retrying without cookies (no-JS fallback)...")
    result = _try_download(_build_download_opts(nocookie_fallback=True), hard_skip_on_format_error=True)
    if result is False:
        print(f"    Skipping (not available): {video_url}")
        return None
    if result is not None:
        return result

    print(f"    Giving up on {video_url}")
    return None


def transcribe_audio(audio_path: str, pipeline: BatchedInferencePipeline) -> dict:
    # BatchedInferencePipeline processes BATCH_SIZE audio chunks in parallel on the GPU,
    # raising utilization from ~20-40% to ~60-80%. Chunks are independent so
    # condition_on_previous_text is not applicable (and not needed — batched mode
    # inherently avoids the repetition loop that flag was guarding against).
    segments_iter, info = pipeline.transcribe(
        audio_path,
        language="hi",
        task="translate",
        beam_size=5,
        batch_size=BATCH_SIZE,
        initial_prompt=INITIAL_PROMPT,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    # Consume the iterator with a progress bar driven by segment timestamps.
    segments = []
    last_end = 0.0
    total    = max(info.duration, 1.0)
    bar_fmt  = "{l_bar}{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]"
    with tqdm(total=total, unit="s", desc="    Transcribing", bar_format=bar_fmt, leave=False) as pbar:
        for seg in segments_iter:
            pbar.update(max(0.0, seg.end - last_end))
            last_end = seg.end
            segments.append(seg)
        pbar.update(max(0.0, total - last_end))

    full_text = " ".join(s.text.strip() for s in segments)
    return {
        "full_text": full_text,
        "language_detected":     info.language,
        "language_probability":  round(info.language_probability, 3),
        "duration_seconds":      round(info.duration, 2),
        "segments": [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments
        ],
    }


def save_transcript_atomic(video_info: dict, whisper_result: dict) -> str:
    """Write to a .tmp file then rename, so a crash mid-write doesn't corrupt the JSON."""
    video_id  = video_info["video_id"]
    final     = TRANSCRIPT_DIR / f"{video_id}.json"
    tmp       = final.with_suffix(".json.tmp")
    data = {
        **video_info,
        "transcribed_at": datetime.now().isoformat(),
        "task": "translate",
        **whisper_result,
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, final)
    print(f"    Saved: {final.name}")
    return str(final)


def process_urls(urls: list[str], cleanup_audio: bool = False):
    setup_dirs()
    seen_ids: set[str] = set()

    # ── Pass 1: enumerate every URL (cheap, no audio download) ────────────────
    print("\nEnumerating videos from input URLs...")
    pending: list[dict] = []
    for url in urls:
        url = url.strip()
        if not url or url.startswith("#"):
            continue
        try:
            entries = enumerate_entries(url)
        except Exception as e:
            print(f"  Failed to enumerate {url}: {e}")
            continue
        for entry in entries:
            vid = entry["id"]
            if vid in seen_ids:
                continue
            seen_ids.add(vid)
            if already_transcribed(vid):
                print(f"  Skip (already transcribed): {entry['title']}")
                continue
            pending.append(entry)

    if not pending:
        print("Nothing new to transcribe.")
        return

    print(f"\n{len(pending)} new video(s) to process\n")

    # ── Pass 2: load model once, then download+transcribe each pending video ──
    print(f"Loading faster-whisper {WHISPER_MODEL} on {DEVICE} ({COMPUTE_TYPE})...")
    model    = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    pipeline = BatchedInferencePipeline(model=model)
    print(f"Model ready (batch_size={BATCH_SIZE})\n")

    for i, entry in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] {entry['title']}")
        video = download_one(entry["url"])
        if video is None:
            continue

        try:
            result = transcribe_audio(video["audio_path"], pipeline)
            save_transcript_atomic(video, result)
            if cleanup_audio:
                try:
                    os.remove(video["audio_path"])
                    print(f"    Removed audio: {Path(video['audio_path']).name}")
                except OSError as e:
                    print(f"    Could not remove audio: {e}")
        except Exception as e:
            print(f"    Transcription failed: {e}")

    print(f"\nDone. Transcripts in: {TRANSCRIPT_DIR.resolve()}")


def cleanup_orphaned_audio():
    """Delete audio/<id>.mp3 files whose corresponding transcripts/<id>.json already exists.

    Run after any session where --cleanup-audio was forgotten. Only audio files with a
    matching transcript are removed — audio for which transcription never succeeded
    is preserved so it can be retried.
    """
    if not AUDIO_DIR.exists():
        print("No audio/ directory — nothing to clean up.")
        return
    audio_files = list(AUDIO_DIR.glob("*.mp3"))
    if not audio_files:
        print("No .mp3 files in audio/ — already clean.")
        return

    removed = 0
    kept    = 0
    freed_bytes = 0
    for audio in audio_files:
        video_id = audio.stem
        if (TRANSCRIPT_DIR / f"{video_id}.json").exists():
            size = audio.stat().st_size
            audio.unlink()
            removed += 1
            freed_bytes += size
        else:
            kept += 1

    freed_mb = freed_bytes / (1024 * 1024)
    print(f"Removed {removed} orphan .mp3 file(s) ({freed_mb:.1f} MB freed)")
    if kept:
        print(f"Kept {kept} .mp3 file(s) with no transcript (retry transcription to clear these)")


def main():
    parser = argparse.ArgumentParser(description="Transcribe satsang videos to English")
    parser.add_argument("--url",  type=str, help="YouTube video or playlist URL")
    parser.add_argument("--file", type=str, help="Text file with one URL per line (# = comment)")
    parser.add_argument("--cleanup-audio", action="store_true",
                        help="Delete the .mp3 after a successful transcription")
    parser.add_argument("--cleanup-orphans", action="store_true",
                        help="Delete audio/*.mp3 files whose transcript already exists, then exit")
    args = parser.parse_args()

    if args.cleanup_orphans:
        cleanup_orphaned_audio()
        return

    if not (args.url or args.file):
        parser.error("one of --url, --file, or --cleanup-orphans is required")
    if args.url and args.file:
        parser.error("--url and --file are mutually exclusive")

    if args.url:
        urls = [args.url]
    else:
        with open(args.file, encoding="utf-8") as f:
            urls = f.readlines()
    process_urls(urls, cleanup_audio=args.cleanup_audio)


if __name__ == "__main__":
    main()
