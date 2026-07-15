# Premanand Maharajji Satsang Chatbot

A local, privacy-first **RAG (Retrieval-Augmented Generation) chatbot** that answers spiritual questions grounded in the YouTube satsangs of **Premanand Maharajji**. Source material is primarily Hindi with Sanskrit; it is transcribed and translated to English via Whisper, indexed for hybrid retrieval, and served through a chat UI that **cites the original video at the exact timestamp** for every answer.

The model speaks *as* Maharajji to the seeker — warmly, in persona — but only from what the retrieved passages actually contain. Nothing is invented; every claim traces back to a citation.

---

## Why this exists

Hundreds of hours of teaching live across ~700 YouTube videos, un-searchable and un-navigable. This project turns that corpus into something you can *ask a question of* — and always land on the exact moment in the exact video where the teaching was given.

---

## How it works

Three sequential, fully idempotent stages. Re-running any stage skips already-processed items.

```
YouTube URLs
    │
    ▼
┌─────────────────────┐   faster-whisper large-v3, GPU
│ 1_transcribe.py     │   Hindi/Sanskrit audio → English transcript JSON
└─────────────────────┘
    │  transcripts/*.json
    ▼
┌─────────────────────┐   semantic chunking + all-mpnet-base-v2 embeddings
│ 2_build_index.py    │   → ChromaDB (vectors) + BM25 (keywords)
└─────────────────────┘
    │  chroma_db/
    ▼
┌─────────────────────┐   hybrid retrieval + cross-encoder rerank
│ 3_chatbot.py        │   → Gradio chat UI, streaming, timestamped citations
└─────────────────────┘
```

### Retrieval pipeline (per query)

1. **Hybrid candidate gathering** — semantic search (ChromaDB cosine) *and* BM25 keyword search run in parallel, then merge. Semantic catches conceptual matches; BM25 catches proper nouns ("Radha Rani", "Yamuna", scripture names).
2. **Cross-encoder reranking** — `BAAI/bge-reranker-base` reads query + chunk *together* for a far more accurate 0–1 relevance score than cosine alone. Chunks below threshold are dropped.
3. **Per-video deduplication** — only the top chunk per video survives, so the final results span 5 different satsangs instead of near-duplicates.

If nothing clears the relevance bar, the bot says so in character rather than hallucinating.

---

## Features

- **Timestamped citations** — every answer links to `[Video Title @ MM:SS](youtube_url&t=...s)`.
- **Two backends** — local **Qwen3 8B** via Ollama (free, private) or **Claude API** (optional upgrade).
- **Streaming responses** — tokens appear within ~1 second.
- **Thinking mode** — deeper reasoning for nuanced philosophical questions (Qwen3).
- **Persona grounding** — speaks as Maharajji; never breaks character into assistant-voice.
- **Seeker memory** — quietly extracts and persists personal context across turns to personalize guidance.
- **Situational reformulation** — "I lost my job and feel like a failure" is rewritten into satsang vocabulary for retrieval, while the LLM still responds to the real situation.
- **In-UI transcript correction** — edit a chunk and re-embed it live; corrections are logged and applied immediately.
- **Full instrumentation** — every query and thumbs up/down logged as JSONL for diagnosis and future fine-tuning.

---

## Tech stack

| Component | Choice |
|---|---|
| Speech-to-text | `faster-whisper` large-v3 (batched, CTranslate2) |
| Download | `yt-dlp` (playlist support) |
| Embeddings | `all-mpnet-base-v2` (English) |
| Vector store | ChromaDB (local, persistent) |
| Keyword search | `rank-bm25` |
| Reranker | `BAAI/bge-reranker-base` cross-encoder |
| LLM | Qwen3 8B via Ollama · Claude API (optional) |
| UI | Gradio |

---

## Hardware target

- GPU: RTX 4070 laptop, 8 GB VRAM
- faster-whisper large-v3 ≈ 4 GB · Qwen3 8B Q4_K_M ≈ 5 GB
- Run **Stage 1** and **Stage 3** at different times — they cannot share VRAM. Stage 2 runs on CPU.

---

## Setup

```powershell
pip install -r requirements.txt
```

Local inference (free, private):

```powershell
# Install Ollama from https://ollama.com/download, then:
ollama pull qwen3:8b
```

Optional Claude backend:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Requirements:
- **ffmpeg** on PATH (yt-dlp audio extraction).
- **CUDA Toolkit** for GPU transcription — the GPU driver alone is not enough (faster-whisper needs cuBLAS). Download from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads). Set `DEVICE = "cpu"` in `1_transcribe.py` to run without it.
- **Node.js** recommended — lets yt-dlp solve YouTube's n-challenge on the primary download path.

---

## Usage

### Stage 1 — Transcribe

```powershell
# Single video
python 1_transcribe.py --url "https://www.youtube.com/watch?v=VIDEO_ID"

# Playlist
python 1_transcribe.py --url "https://www.youtube.com/playlist?list=PLAYLIST_ID"

# Batch from file — paste playlist URLs into urls.txt, one per line
python 1_transcribe.py --file urls.txt --cleanup-audio
```

> **Always use `--cleanup-audio` at scale.** 700 videos × ~50 MB ≈ 35 GB of audio otherwise. Audio is only needed during transcription and is deleted immediately after.

Close Stage 1 (free VRAM) before launching the chatbot.

### Stage 2 — Build index

```powershell
python 2_build_index.py
```

Chunks, embeds, and upserts all transcripts into ChromaDB + BM25. Safe to re-run after adding new transcripts — existing chunks are skipped.

### Stage 3 — Chat

```powershell
python 3_chatbot.py
```

Opens a Gradio UI in the browser. Pick a backend, optionally enable Thinking Mode, and ask.

---

## Project structure

```
premanandMaharajji/
├── 1_transcribe.py     # download audio + transcribe to English
├── 2_build_index.py    # chunk + embed + ChromaDB + BM25
├── 3_chatbot.py        # RAG chatbot with Gradio UI
├── requirements.txt
├── urls.txt            # one YouTube URL per line (playlists OK)
├── audio/              # downloaded .mp3 (gitignored)
├── transcripts/        # one .json per video (gitignored)
├── logs/               # query + feedback JSONL (gitignored)
└── chroma_db/          # vector index + bm25_index.json (gitignored)
```

---

## Design notes

- **Semantic chunking, not fixed-size** — satsangs are Q&A and parables; splitting mid-story wrecks retrieval. Chunks split on topic shifts (cosine-distance spikes) with guardrails on min/max length.
- **Translate-in-one-pass** — Whisper does Hindi+Sanskrit → English directly (`task="translate"`); the whole system operates in English.
- **BGE reranker over MS-MARCO** — MS-MARCO was trained on web-search queries and systematically under-scores devotional passages; BGE handles the vocabulary gap between colloquial English questions and translated satsang text.
- **RAG over fine-tuning** — the LLM's job is *application* of retrieved teaching, not memorization. Fine-tuning on satsang text would teach the style without the grounding, inviting hallucination. Citations keep it honest.

---

## License & attribution

Personal, non-commercial project. All teachings belong to **Premanand Maharajji**; this tool only indexes and points back to his publicly available videos. Please watch the source satsangs — the citations link straight to them.
