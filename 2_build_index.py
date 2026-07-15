"""
Chunk transcripts and build a searchable ChromaDB vector index.

Run after 1_transcribe.py has populated the transcripts/ directory.
    python 2_build_index.py

Re-running is safe — already-indexed videos are skipped automatically.
"""

import json
from pathlib import Path

import chromadb
import numpy as np
import torch
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

TRANSCRIPT_DIR   = Path("transcripts")
CHROMA_DIR       = Path("chroma_db")
BM25_FILE        = CHROMA_DIR / "bm25_index.json"
COLLECTION_NAME  = "satsangs"
EMBED_MODEL      = "all-mpnet-base-v2"
SEMANTIC_THRESH  = 0.35
MIN_CHUNK_WORDS  = 80
MAX_CHUNK_WORDS  = 300


def load_transcripts() -> list[dict]:
    paths = sorted(TRANSCRIPT_DIR.glob("*.json"))
    transcripts = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            transcripts.append(json.load(f))
    print(f"Loaded {len(transcripts)} transcript(s)")
    return transcripts


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def chunk_segments_semantic(segments: list[dict], segment_embeddings: np.ndarray) -> list[list[dict]]:
    """Split segments at topic boundaries using precomputed embeddings."""
    if len(segments) <= 1:
        return [segments] if segments else []

    chunks: list[list[dict]] = []
    current: list[dict] = [segments[0]]
    current_words = len(segments[0]["text"].split())

    for i in range(1, len(segments)):
        seg       = segments[i]
        seg_words = len(seg["text"].split())
        dist      = cosine_distance(segment_embeddings[i - 1], segment_embeddings[i])

        topic_shift = dist > SEMANTIC_THRESH and current_words >= MIN_CHUNK_WORDS
        too_long    = current_words + seg_words > MAX_CHUNK_WORDS

        if topic_shift or too_long:
            chunks.append(current)
            # carry last segment as overlap for context continuity
            current       = [current[-1], seg]
            current_words = len(current[-2]["text"].split()) + seg_words
        else:
            current.append(seg)
            current_words += seg_words

    if current:
        chunks.append(current)
    return chunks


def make_chunk_docs(transcript: dict, embed_model: SentenceTransformer) -> list[dict]:
    segments = transcript.get("segments", [])
    if not segments:
        return []

    video_id = transcript["video_id"]
    url      = transcript.get("url", "")
    title    = transcript.get("title", "")
    channel  = transcript.get("channel", "")

    seg_texts      = [s["text"] for s in segments]
    seg_embeddings = embed_model.encode(seg_texts, batch_size=64, show_progress_bar=False)

    docs = []
    for i, group in enumerate(chunk_segments_semantic(segments, seg_embeddings)):
        text       = " ".join(s["text"] for s in group)
        start_time = group[0]["start"]
        end_time   = group[-1]["end"]
        ts_url     = f"{url}&t={int(start_time)}s" if url else ""

        docs.append({
            "id":   f"{video_id}_chunk{i:04d}",
            "text": text,
            "metadata": {
                "video_id":    video_id,
                "title":       title,
                "url":         url,
                "ts_url":      ts_url,
                "channel":     channel,
                "start":       start_time,
                "end":         end_time,
                "chunk_index": i,
            },
        })
    return docs


def build_index():
    CHROMA_DIR.mkdir(exist_ok=True)
    transcripts = load_transcripts()
    if not transcripts:
        print("No transcripts found. Run 1_transcribe.py first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model ({EMBED_MODEL}) on {device} for chunking...")
    # GPU model used only for segment-level cosine distance during chunking (fast, runs alone).
    # ChromaDB uses its own SentenceTransformerEmbeddingFunction instance (CPU) for adds/queries,
    # keeping the collection's stored EF name consistent between Stage 2 and Stage 3.
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    embed_fn    = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Find which video_ids are already fully indexed so we skip them entirely
    # (instead of embedding all segments and discovering nothing is new).
    existing = collection.get(include=["documents", "metadatas"])
    existing_video_ids = {m["video_id"] for m in existing["metadatas"]} if existing["metadatas"] else set()

    # One-time migration: build BM25 index from existing ChromaDB content if the file is missing.
    if not BM25_FILE.exists() and existing["ids"]:
        print("Building BM25 index from existing ChromaDB content (one-time migration)...")
        corpus = [{"id": cid, "text": doc} for cid, doc in zip(existing["ids"], existing["documents"])]
        with open(BM25_FILE, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False)
        print(f"  BM25 index built: {len(corpus)} entries")

    pending = [t for t in transcripts if t["video_id"] not in existing_video_ids]
    print(f"Videos to process: {len(pending)} new, {len(transcripts) - len(pending)} already indexed")

    if not pending:
        print("Index is already up to date.")
        return

    all_docs = []
    for t in tqdm(pending, desc="Chunking + embedding"):
        all_docs.extend(make_chunk_docs(t, embed_model))
    print(f"Total new chunks: {len(all_docs)}")

    BATCH = 100
    for i in tqdm(range(0, len(all_docs), BATCH), desc="Indexing"):
        batch = all_docs[i : i + BATCH]
        collection.add(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )

    # Append new chunks to BM25 corpus file.
    bm25_corpus = []
    if BM25_FILE.exists():
        with open(BM25_FILE, encoding="utf-8") as f:
            bm25_corpus = json.load(f)
    existing_bm25_ids = {e["id"] for e in bm25_corpus}
    new_bm25 = [{"id": d["id"], "text": d["text"]} for d in all_docs if d["id"] not in existing_bm25_ids]
    if new_bm25:
        bm25_corpus.extend(new_bm25)
        with open(BM25_FILE, "w", encoding="utf-8") as f:
            json.dump(bm25_corpus, f, ensure_ascii=False)
        print(f"BM25 index updated: {len(bm25_corpus)} total entries")

    print(f"\nDone. Index has {collection.count()} total chunks in {CHROMA_DIR.resolve()}")


if __name__ == "__main__":
    build_index()
