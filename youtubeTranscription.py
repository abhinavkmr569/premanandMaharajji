"""
YouTube Transcriber for Premanand Maharajji Satsangs
-----------------------------------------------------
Downloads audio from YouTube and transcribes (+ translates to English) using Whisper.

Requirements:
    pip install yt-dlp openai-whisper

Also needs ffmpeg on PATH:
    Windows: download from https://ffmpeg.org/download.html and add to PATH
    (You mentioned you already have ffmpeg manually pathed — should be fine)

Usage:
    Single video:   python transcribe_youtube.py --url "https://www.youtube.com/watch?v=..."
    Playlist:       python transcribe_youtube.py --url "https://www.youtube.com/playlist?list=..."
    From file:      python transcribe_youtube.py --file urls.txt
"""

import os
import json
import argparse
import whisper
import yt_dlp
from pathlib import Path
from datetime import datetime

import os
os.environ["PATH"] += r";C:\Users\abhin\anaconda3\Library\bin"

import subprocess                                                                                
subprocess.run(["ffmpeg", "-version"])


# ── Config ────────────────────────────────────────────────────────────────────

AUDIO_DIR       = Path("audio")          # downloaded audio files go here
TRANSCRIPT_DIR  = Path("transcripts")    # output .txt and .json files go here
WHISPER_MODEL   = "large-v2"               # options: tiny, base, small, medium, large
                                         # medium is the sweet spot for Hindi→English accuracy
TASK            = "translate"            # "translate" = Hindi→English in one shot
                                         # "transcribe" = keep original language

# ──────────────────────────────────────────────────────────────────────────────


def setup_dirs():
    AUDIO_DIR.mkdir(exist_ok=True)
    TRANSCRIPT_DIR.mkdir(exist_ok=True)


def download_audio(url: str) -> list[dict]:
    """
    Downloads audio from a YouTube URL (single video or playlist).
    Returns list of dicts: [{title, video_id, audio_path, url}]
    """
    downloaded = []

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(AUDIO_DIR / "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",   # 128kbps is enough for speech
        }],
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        # Handle both single videos and playlists
        if "entries" in info:
            entries = info["entries"]
        else:
            entries = [info]

        for entry in entries:
            if entry is None:
                continue
            video_id   = entry.get("id", "unknown")
            title      = entry.get("title", "untitled")
            audio_path = AUDIO_DIR / f"{video_id}.mp3"

            downloaded.append({
                "title":      title,
                "video_id":   video_id,
                "audio_path": str(audio_path),
                "url":        f"https://www.youtube.com/watch?v={video_id}",
            })
            print(f"  ✓ Downloaded: {title}")

    return downloaded


def transcribe_audio(audio_path: str, model) -> dict:
    """
    Transcribes (and translates to English) an audio file using Whisper.
    Returns dict with 'text' and 'segments'.
    """
    result = model.transcribe(
        audio_path,
        task=TASK,                  # "translate" converts Hindi → English
        language="hi",              # tell Whisper to expect Hindi
        verbose=False,
    )
    return result


def save_transcript(video_info: dict, whisper_result: dict):
    """
    Saves two files per video:
      - transcripts/<video_id>.txt   → plain text (easy to read)
      - transcripts/<video_id>.json  → structured with segments + timestamps
    """
    video_id = video_info["video_id"]
    title    = video_info["title"]
    url      = video_info["url"]
    text     = whisper_result["text"].strip()
    segments = whisper_result.get("segments", [])

    # ── Plain text ────────────────────────────────────────────────────────────
    txt_path = TRANSCRIPT_DIR / f"{video_id}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Title: {title}\n")
        f.write(f"URL:   {url}\n")
        f.write(f"Date:  {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(text)

    # ── Structured JSON (useful for chunking later in Lessons 9–12) ───────────
    json_path = TRANSCRIPT_DIR / f"{video_id}.json"
    structured = {
        "title":      title,
        "video_id":   video_id,
        "url":        url,
        "language":   "hi",
        "task":       TASK,
        "transcribed_at": datetime.now().isoformat(),
        "full_text":  text,
        "segments": [
            {
                "start": round(s["start"], 2),
                "end":   round(s["end"],   2),
                "text":  s["text"].strip(),
            }
            for s in segments
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Saved: {txt_path.name}  +  {json_path.name}")
    return str(txt_path), str(json_path)


def already_transcribed(video_id: str) -> bool:
    """Skip videos we've already processed — useful when running on a playlist repeatedly."""
    return (TRANSCRIPT_DIR / f"{video_id}.json").exists()


def process_urls(urls: list[str]):
    setup_dirs()

    print(f"\n🔄 Loading Whisper model: {WHISPER_MODEL}  (task={TASK})")
    model = whisper.load_model(WHISPER_MODEL)
    print("✓ Model loaded\n")

    for url in urls:
        url = url.strip()
        if not url or url.startswith("#"):
            continue

        print(f"\n{'='*60}")
        print(f"📥 Downloading: {url}")

        try:
            videos = download_audio(url)
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            continue

        for video in videos:
            vid_id = video["video_id"]

            if already_transcribed(vid_id):
                print(f"  ⏭  Already transcribed: {video['title']} — skipping")
                continue

            print(f"\n🎙  Transcribing: {video['title']}")
            try:
                result = transcribe_audio(video["audio_path"], model)
                save_transcript(video, result)

                # Optional: delete audio after transcription to save disk space
                # os.remove(video["audio_path"])

            except Exception as e:
                print(f"  ✗ Transcription failed: {e}")

    print(f"\n✅ Done. Transcripts saved to: {TRANSCRIPT_DIR.resolve()}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transcribe YouTube videos to English text")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url",  type=str, help="Single YouTube video or playlist URL")
    group.add_argument("--file", type=str, help="Text file with one YouTube URL per line")
    args = parser.parse_args()

    if args.url:
        urls = [args.url]
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            urls = f.readlines()

    process_urls(urls)


if __name__ == "__main__":
    main()