"""
RAG chatbot for Premanand Maharajji satsangs.

Backends (select in UI):
  - Local (Ollama): free, private, runs on your GPU
      Install: https://ollama.com/download
      Pull model: ollama pull qwen3:8b
  - Claude API: higher quality, requires ANTHROPIC_API_KEY env var

Run:
    python 3_chatbot.py
"""

import json
import math
import os
import re
import threading
from datetime import datetime
from pathlib import Path

import chromadb
import gradio as gr
import numpy as np
import requests
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

CHROMA_DIR      = Path("chroma_db")
BM25_FILE       = CHROMA_DIR / "bm25_index.json"
COLLECTION_NAME = "satsangs"
EMBED_MODEL     = "all-mpnet-base-v2"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"

TOP_K           = 5        # final chunks shown to LLM and cited in sources
RERANK_POOL     = 25       # candidates fetched from each source before reranking (sized for 50K+ chunks)
MIN_SCORE       = 0.30     # discard chunks whose reranker sigmoid score is below this
MAX_HISTORY     = 4        # keep at most this many (user, assistant) turns in LLM context

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:8b"

CLAUDE_MODEL = "claude-sonnet-4-6"

# Unique delimiter so strip_sources() never collides with `---` the model might use itself.
SOURCES_DELIMITER = "\n\n<!--sources-->\n\n---\n"

USER_CONTEXT_FILE = Path("user_context.txt")

LOGS_DIR          = Path("logs")
QUERY_LOG_FILE    = LOGS_DIR / "queries.jsonl"
FEEDBACK_LOG_FILE = LOGS_DIR / "feedback.jsonl"


def _log_jsonl(path: Path, record: dict):
    """Append a single JSON object to a JSONL file. Best-effort — never raises."""
    try:
        LOGS_DIR.mkdir(exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  Failed to write log to {path}: {e}")

_user_context = ""
_context_lock = threading.Lock()


def load_user_context() -> str:
    if USER_CONTEXT_FILE.exists():
        return USER_CONTEXT_FILE.read_text(encoding="utf-8").strip()
    return ""


def save_user_context(text: str):
    USER_CONTEXT_FILE.write_text(text, encoding="utf-8")


_BASE_SYSTEM_PROMPT = (
    "You are Premanand Maharajji, speaking directly to a seeker who has come to you in satsang.\n\n"
    "You will be given passages from your own past satsangs. Speak ONLY from what is present "
    "in those passages — they are the boundary of what you may say. "
    "Do not add any specific fact, number, name, story, instruction, or spiritual claim "
    "that does not appear in the provided passages, even if you feel it would fit naturally. "
    "If the passages do not contain enough to answer fully, say so with warmth: "
    "\"What you ask, I have not spoken of in the words given here — come again, and we shall see.\"\n\n"
    "Speak in first person, as Maharajji. Address the seeker directly and personally. "
    "Use 'beta' or 'bacche' when it feels natural. Let Sanskrit and Hindi terms arise naturally "
    "(Bhakti, Prema, Japa, Vairagya, Maya, Naam) — explain their depth as you would in satsang, "
    "not as dictionary definitions.\n\n"
    "Do not say 'Maharajji says' — you ARE Maharajji speaking. "
    "Let your words flow as they do in satsang: unhurried, warm, from the heart. "
    "No bullet points, no numbered steps, no bold headers. "
    "Speak as to a beloved child sitting before you.\n\n"
    "No matter how long the conversation goes, never slip into the role of an AI assistant "
    "or a narrator summarising teachings. You are always Maharajji, speaking directly. "
    "When in doubt about a specific detail — a number, a name, a practice — stay silent on it "
    "rather than risk saying something not in the passages.\n\n"
    "Respond in the language the seeker uses — Hindi or English, whichever they write in. "
    "When speaking Hindi, use the simple, warm, conversational Hindi of satsang — "
    "not formal or bookish. Short sentences. The words of a loving elder, not a lecturer."
)


def build_system_prompt() -> str:
    with _context_lock:
        ctx = _user_context
    if ctx:
        return _BASE_SYSTEM_PROMPT + f"\n\nWhat you know about this seeker:\n{ctx}"
    return _BASE_SYSTEM_PROMPT


_CONTEXT_EXTRACT_PROMPT = (
    "Read the conversation exchange below. Identify any new personal facts the seeker "
    "revealed — their location, family, spiritual background, struggles, or anything that "
    "would help you know them better as a person.\n\n"
    "If there is genuinely new personal information not already captured, write 1-3 plain sentences.\n"
    "If there is nothing new or personal, reply with exactly: NOTHING_NEW\n\n"
    "Known facts so far:\n{existing}\n\n"
    "New exchange:\n"
    "Seeker: {message}\n"
    "Response: {response}"
)


def _extract_context_update(message: str, response: str, existing: str) -> str | None:
    prompt = _CONTEXT_EXTRACT_PROMPT.format(
        existing=existing or "(none yet)",
        message=message,
        response=response[:600],
    )
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model":    OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "think":    False,
                "options":  {"temperature": 0.1},
            },
            timeout=60,
        )
        r.raise_for_status()
        result = r.json().get("message", {}).get("content", "").strip()
        if result and result != "NOTHING_NEW":
            return result
    except Exception:
        pass
    return None


_CONTEXT_MAX_WORDS = 500

_CONTEXT_COMPRESS_PROMPT = (
    "The following are accumulated personal facts about a seeker, gathered across many conversations. "
    "Compress this into the essential, durable facts about the person — location, family, work or study, "
    "spiritual background, recurring struggles, important relationships. "
    "Keep specifics where they matter. Drop duplication, transient details, and outdated information. "
    "Output 5-10 plain sentences. No headers, no bullet points, no preamble.\n\n"
    "Accumulated facts:\n{context}"
)


def _compress_user_context(context: str) -> str | None:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model":    OLLAMA_MODEL,
                "messages": [{"role": "user", "content": _CONTEXT_COMPRESS_PROMPT.format(context=context)}],
                "stream":   False,
                "think":    False,
                "options":  {"temperature": 0.1},
            },
            timeout=120,
        )
        r.raise_for_status()
        result = r.json().get("message", {}).get("content", "").strip()
        return result or None
    except Exception:
        return None


def _update_context_job(message: str, response: str):
    global _user_context
    with _context_lock:
        existing = _user_context
    new_info = _extract_context_update(message, response, existing)
    if not new_info:
        return
    with _context_lock:
        updated = ((_user_context + "\n" + new_info).strip() if _user_context else new_info)
        if len(updated.split()) > _CONTEXT_MAX_WORDS:
            compressed = _compress_user_context(updated)
            if compressed and len(compressed.split()) < len(updated.split()):
                print(f"  Compressed user context: {len(updated.split())} → {len(compressed.split())} words")
                updated = compressed
        _user_context = updated
    save_user_context(updated)


EMPTY_RETRIEVAL_MESSAGE = (
    "Beta, what you ask of me I have not spoken of in the words given here. "
    "Come again with your question, perhaps in different words, and we shall see."
)

# Used when retrieval finds nothing — allows natural in-character responses to
# greetings, personal updates, and conversational messages without the passage constraint.
_CONVERSATIONAL_SYSTEM = (
    "You are Premanand Maharajji, sitting in satsang. The seeker has said something "
    "conversational — a greeting, a personal update, or casual words — rather than asking "
    "a specific spiritual question. Respond naturally in character: warm, brief, unhurried, "
    "as you would between teachings or at the start of satsang. Address them directly. "
    "Use 'beta' or 'bacche' if it feels right. No citations needed. "
    "Respond in the language the seeker uses."
)


# Personal pronouns and reference words that signal a situational/contextual message
# needing reformulation. Clean spiritual queries ("what is bhakti?") don't trigger this.
_SITUATIONAL_MARKERS = re.compile(
    r"\b(i|i'm|i've|i'd|i'll|my|mine|myself|"
    r"remember|earlier|before|previously|yesterday|today|"
    r"happened|happening|feeling|felt|struggling|struggle|"
    r"situation|case|problem|issue)\b",
    re.IGNORECASE,
)


def is_situational(message: str) -> bool:
    """True if the message references the seeker's own life or prior context."""
    return bool(_SITUATIONAL_MARKERS.search(message))


def reformulate_query(message: str, history: list[dict]) -> str:
    """Rewrite a seeker's message as a focused spiritual question for better retrieval.

    Situational messages ("I lost my job and feel hopeless") don't match satsang chunks
    well because the vocabulary differs. This rewrites them into spiritual terms
    (surrender, vairagya, detachment, maya...) that appear in the transcripts.
    Recent history is included so references like "the attendance issue I mentioned"
    can be resolved into concrete spiritual themes.
    The original message is still used for LLM generation — only retrieval changes.
    Falls back to the original on any error so retrieval never breaks.
    """
    recent = ""
    for msg in history[-4:]:
        role = "Seeker" if msg["role"] == "user" else "Maharajji"
        text = strip_sources(_content_text(msg["content"]))[:300]
        if text:
            recent += f"{role}: {text}\n"
    context_block = f"\nRecent conversation for context:\n{recent}\n" if recent else ""

    prompt = (
        f"A seeker has come to Premanand Maharajji.{context_block}\n"
        "Their latest message:\n"
        f"\"{message}\"\n\n"
        "Rewrite this as a concise spiritual question capturing the core themes the seeker "
        "needs guidance on. Use vocabulary and Sanskrit/Hindi terms that would appear in satsang "
        "transcripts — for example: satya (truth), dharma (righteous duty), karma, sharanagati "
        "(surrender), vairagya (detachment), bhakti (devotion), maya (illusion), equanimity, "
        "Naam, Seva, grief, worldly attachment, fruits of action, God's will, acceptance.\n\n"
        "Write only the rewritten question. Nothing else."
    )
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model":    OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "think":    False,
                "options":  {"temperature": 0.1},
            },
            timeout=30,
        )
        r.raise_for_status()
        result = r.json().get("message", {}).get("content", "").strip()
        return result if result else message
    except Exception:
        return message


# ── Index loaders ─────────────────────────────────────────────────────────────

_collection = None
_bm25: BM25Okapi | None = None
_bm25_ids: list[str] | None = None
_reranker: CrossEncoder | None = None


def get_collection():
    global _collection
    if _collection is None:
        embed_fn    = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        client      = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    return _collection


def get_bm25() -> tuple[BM25Okapi | None, list[str] | None]:
    global _bm25, _bm25_ids
    if _bm25 is None and BM25_FILE.exists():
        with open(BM25_FILE, encoding="utf-8") as f:
            corpus = json.load(f)
        _bm25_ids = [e["id"] for e in corpus]
        _bm25 = BM25Okapi([e["text"].lower().split() for e in corpus])
    return _bm25, _bm25_ids


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(CROSS_ENCODER_MODEL)
    return _reranker


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, x))))


# Audit log for transcript corrections — append-only, useful for review and recovery.
CORRECTIONS_LOG = CHROMA_DIR / "chunk_corrections.jsonl"


def update_chunk(chunk_id: str, new_text: str) -> tuple[bool, str]:
    """Replace a chunk's text in ChromaDB and BM25. Returns (success, status_msg).

    ChromaDB re-embeds the document automatically because the collection has an
    embedding function attached. BM25 is updated in-place and the in-memory index
    is invalidated so the next query rebuilds it. An audit line is appended to
    chunk_corrections.jsonl so corrections are traceable.
    """
    global _bm25, _bm25_ids

    new_text = new_text.strip()
    if not new_text:
        return False, "Empty text — refusing to save."

    try:
        collection = get_collection()
        existing = collection.get(ids=[chunk_id], include=["metadatas", "documents"])
        if not existing["ids"]:
            return False, f"Chunk `{chunk_id}` not found in index."

        old_text = existing["documents"][0]
        metadata = existing["metadatas"][0]

        if new_text == old_text:
            return False, "No change — text is identical."

        # Re-embed via update (collection's embedding function recomputes the vector).
        collection.update(ids=[chunk_id], documents=[new_text])

        # Update BM25 corpus file in place.
        if BM25_FILE.exists():
            with open(BM25_FILE, encoding="utf-8") as f:
                corpus = json.load(f)
            for entry in corpus:
                if entry["id"] == chunk_id:
                    entry["text"] = new_text
                    break
            with open(BM25_FILE, "w", encoding="utf-8") as f:
                json.dump(corpus, f, ensure_ascii=False)
            # Force BM25 to rebuild on next query.
            _bm25 = None
            _bm25_ids = None

        # Audit log.
        CHROMA_DIR.mkdir(exist_ok=True)
        with open(CORRECTIONS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts":        datetime.now().isoformat(),
                "chunk_id":  chunk_id,
                "video_id":  metadata.get("video_id"),
                "title":     metadata.get("title"),
                "old_text":  old_text,
                "new_text":  new_text,
            }, ensure_ascii=False) + "\n")

        return True, f"Saved & re-embedded ({len(new_text)} chars). Next query will use the corrected text."
    except Exception as e:
        return False, f"Failed: {e}"


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve(query: str) -> tuple[list[dict], dict]:
    """Hybrid BM25 + semantic retrieval, reranked by a cross-encoder.

    Returns (chunks, meta) where meta has diagnostic fields suitable for logging
    and for showing the user in the retrieval-details panel.
    """
    meta: dict = {"candidate_count": 0, "top_scores": [], "min_score": MIN_SCORE}

    # 1. Semantic candidates via ChromaDB
    raw = get_collection().query(
        query_texts=[query],
        n_results=RERANK_POOL,
        include=["documents", "metadatas", "distances"],
    )
    candidates: dict[str, dict] = {}
    for cid, doc, m, dist in zip(
        raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        candidates[cid] = {"text": doc, "metadata": m}

    # 2. BM25 keyword candidates — catches exact proper noun matches semantic search misses
    bm25, bm25_ids = get_bm25()
    if bm25 is not None:
        bm25_scores = bm25.get_scores(query.lower().split())
        top_idx = np.argsort(bm25_scores)[::-1][:RERANK_POOL]
        new_ids = [bm25_ids[i] for i in top_idx if bm25_scores[i] > 0 and bm25_ids[i] not in candidates]
        if new_ids:
            fetched = get_collection().get(ids=new_ids, include=["documents", "metadatas"])
            for cid, doc, m in zip(fetched["ids"], fetched["documents"], fetched["metadatas"]):
                candidates[cid] = {"text": doc, "metadata": m}

    meta["candidate_count"] = len(candidates)
    if not candidates:
        return [], meta

    # 3. Cross-encoder rerank — scores query vs each candidate together (more accurate than cosine)
    reranker = get_reranker()
    cids = list(candidates.keys())
    ce_scores = reranker.predict([(query, candidates[cid]["text"]) for cid in cids])

    # 4. Sort by reranker score, convert to interpretable 0-1 via sigmoid, filter, return TOP_K.
    top_scores = sorted(ce_scores.tolist(), reverse=True)[:5]
    meta["top_scores"] = [round(sigmoid(s), 3) for s in top_scores]
    print(f"  Reranker top-5 sigmoid scores: {meta['top_scores']}")
    #    One chunk per video maximum — at 700+ videos, the same theme recurs across hundreds of
    #    videos and without this the top 5 would often be near-identical passages.
    ranked = sorted(zip(cids, ce_scores.tolist()), key=lambda x: x[1], reverse=True)
    results = []
    seen_videos: set[str] = set()
    for cid, ce_score in ranked:
        score = round(sigmoid(ce_score), 3)
        if score < MIN_SCORE:
            break  # sorted descending — nothing below this will pass either
        c = candidates[cid]
        vid = c["metadata"].get("video_id", cid)
        if vid in seen_videos:
            continue
        seen_videos.add(vid)
        results.append({"id": cid, "text": c["text"], "metadata": c["metadata"], "score": score})
        if len(results) >= TOP_K:
            break
    return results, meta


def fmt_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def build_context_block(chunks: list[dict]) -> str:
    # Deliberately omit the video title here — the LLM would cite episode numbers
    # (e.g. "#1187") in-text, breaking the first-person Maharajji persona.
    # Full titles with links appear in the sources block shown to the user separately.
    parts = []
    for c in chunks:
        ts = fmt_timestamp(c["metadata"].get("start", 0))
        parts.append(f"[Your words from satsang — {ts}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def format_sources_md(chunks: list[dict]) -> str:
    lines = ["**Sources**"]
    seen  = set()
    for c in chunks:
        meta   = c["metadata"]
        ts_url = meta.get("ts_url") or meta.get("url", "")
        title  = meta.get("title", "Unknown")
        ts     = fmt_timestamp(meta.get("start", 0))
        key    = f"{meta.get('video_id', '')}_{int(meta.get('start', 0))}"
        if key not in seen:
            seen.add(key)
            lines.append(f"- [{title} @ {ts}]({ts_url})  *(relevance: {c['score']:.2f})*")
    return "\n".join(lines)


def _content_text(content) -> str:
    """Gradio 6 may return content as a list of multimodal blocks — extract plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b["text"] if isinstance(b, dict) and "text" in b else str(b)
            for b in content
        )
    return str(content)


def strip_sources(text: str) -> str:
    """Remove the appended sources block before feeding an old response back to the LLM."""
    return text.split(SOURCES_DELIMITER, 1)[0].rstrip()


def detect_language(text: str) -> str:
    """Return 'Hindi' if text contains significant Devanagari script, else 'English'."""
    devanagari = sum(1 for c in text if 'ऀ' <= c <= 'ॿ')
    return "Hindi" if devanagari > len(text) * 0.15 else "English"


# ── LLM backends (streaming) ───────────────────────────────────────────────────



def _has_repetition_loop(buffer: str) -> bool:
    """Detect when the model is stuck repeating a phrase.

    Checks whether the most recent 80 chars of output have already appeared earlier
    in the buffer (with a 200-char gap to avoid matching itself). An 80-char exact
    repetition essentially never occurs in legitimate prose, so this is a high-precision
    signal that the model is looping.
    """
    if len(buffer) < 400:
        return False
    tail = buffer[-80:]
    head = buffer[:-200]
    return tail in head


def stream_ollama(messages: list[dict], thinking: bool):
    """Yield incremental answer text from Ollama.

    Ollama 0.23+ separates Qwen3 thinking tokens into message.thinking (hidden)
    vs message.content (shown). We pass think=True/False natively so Ollama
    controls the mode; we never need to inject /think into the prompt.
    """
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":    OLLAMA_MODEL,
                "messages": messages,
                "stream":   True,
                "think":    thinking,   # Ollama native thinking control (Qwen3 / deepseek-r1)
                "options":  {"temperature": 1.0 if thinking else 0.3, "repeat_penalty": 1.1, "repeat_last_n": 512, "top_p": 0.9, "num_predict": 2048},
            },
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        yield (
            "**Ollama is not running.**\n\n"
            "Start it with `ollama serve`, then pull the model:\n"
            "```\nollama pull qwen3:8b\n```"
        )
        return
    except Exception as e:
        yield f"Ollama error: {e}"
        return

    buffer = ""

    for line in resp.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue

        if chunk.get("error"):
            yield f"Ollama error: {chunk['error']}"
            return

        if chunk.get("done"):
            break

        # message.thinking contains the hidden reasoning; message.content is the answer.
        piece = chunk.get("message", {}).get("content", "")
        if not piece:
            continue

        buffer += piece

        if _has_repetition_loop(buffer):
            print("  Repetition loop detected — truncating stream")
            # Cut at the start of the repeat so the visible answer ends cleanly.
            buffer = buffer[:-80].rstrip()
            yield buffer
            return

        yield buffer


def stream_claude(messages: list[dict], system: str | None = None):
    """Yield incremental answer text from Claude."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        yield (
            "**ANTHROPIC_API_KEY not set.**\n\n"
            "Set it in your environment, e.g. in PowerShell:\n"
            "```\n$env:ANTHROPIC_API_KEY = \"sk-ant-...\"\n```"
        )
        return

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        accumulated = ""
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=system or build_system_prompt(),
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in messages if m["role"] != "system"
            ],
            temperature=0.1,
        ) as stream:
            for text in stream.text_stream:
                accumulated += text
                yield accumulated
    except Exception as e:
        yield f"Claude API error: {e}"


# ── Chat function ──────────────────────────────────────────────────────────────

def _format_debug_md(message: str, retrieval_query: str, meta: dict, chunks: list[dict], path: str) -> str:
    """Render the retrieval-details panel as markdown."""
    lines = [f"**Path:** {path}"]
    if retrieval_query != message:
        lines.append(f"**Original:** {message}")
        lines.append(f"**Reformulated:** {retrieval_query}")
    else:
        lines.append(f"**Query (used as-is):** {message}")
    lines.append(f"**Candidates considered:** {meta.get('candidate_count', 0)}")
    if meta.get("top_scores"):
        lines.append(f"**Top-5 reranker scores:** {meta['top_scores']}  *(MIN_SCORE = {meta.get('min_score', MIN_SCORE)})*")
    lines.append(f"**Chunks returned to LLM:** {len(chunks)}")
    return "\n\n".join(lines)


def _build_chunk_state(chunks: list[dict]) -> tuple[dict, list[tuple[str, str]]]:
    """Return (state_dict, dropdown_choices) for the chunk-edit accordion."""
    state = {c["id"]: c for c in chunks}
    choices = []
    for i, c in enumerate(chunks):
        title = (c["metadata"].get("title") or "Unknown")[:60]
        ts    = fmt_timestamp(c["metadata"].get("start", 0))
        label = f"{i+1}. {title} @ {ts}  (score {c['score']:.2f})"
        choices.append((label, c["id"]))
    return state, choices


def chat(message: str, history: list[dict], backend: str, thinking: bool):
    empty_chunk_state = {}
    empty_chunk_choices = gr.update(choices=[], value=None)

    if not message.strip():
        yield history, "", "", empty_chunk_state, empty_chunk_choices
        return

    # Skip reformulation for short, clean spiritual queries — they already use the right
    # vocabulary and don't reference the seeker's situation. Saves 1–3s per query.
    if len(message.split()) <= 10 and not is_situational(message):
        retrieval_query = message
        was_reformulated = False
    else:
        retrieval_query = reformulate_query(message, history)
        was_reformulated = (retrieval_query != message)

    chunks, retr_meta = retrieve(retrieval_query)

    # Empty retrieval — two paths depending on message length:
    # Short (≤12 words): likely a greeting or social remark — respond in character freely.
    # Long: the seeker is sharing a personal situation needing spiritual guidance;
    #        do NOT let the LLM improvise teachings without passages to ground them.
    if not chunks:
        path = "conversational fallback" if len(message.split()) <= 12 else "honest refusal (long message)"
        debug_md = _format_debug_md(message, retrieval_query, retr_meta, chunks, path)

        if len(message.split()) <= 12:
            max_msgs = MAX_HISTORY * 2 if MAX_HISTORY > 0 else len(history)
            conv_messages = [{"role": "system", "content": _CONVERSATIONAL_SYSTEM}]
            for msg in history[-max_msgs:]:
                raw = _content_text(msg["content"])
                content = raw if msg["role"] == "user" else strip_sources(raw)
                conv_messages.append({"role": msg["role"], "content": content})
            conv_messages.append({"role": "user", "content": message})

            history = history + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": ""},
            ]
            streamer = stream_ollama(conv_messages, thinking=False) if backend == "Local (Ollama)" \
                else stream_claude(conv_messages, system=_CONVERSATIONAL_SYSTEM)
            partial = ""
            for partial in streamer:
                history[-1]["content"] = partial
                yield history, "", debug_md, empty_chunk_state, empty_chunk_choices
            if not partial:
                history[-1]["content"] = EMPTY_RETRIEVAL_MESSAGE
                yield history, "", debug_md, empty_chunk_state, empty_chunk_choices
        else:
            partial = EMPTY_RETRIEVAL_MESSAGE
            history = history + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": EMPTY_RETRIEVAL_MESSAGE},
            ]
            yield history, "", debug_md, empty_chunk_state, empty_chunk_choices

        _log_jsonl(QUERY_LOG_FILE, {
            "ts":             datetime.now().isoformat(),
            "backend":        backend,
            "query":          message,
            "reformulated":   retrieval_query if was_reformulated else None,
            "is_situational": is_situational(message),
            "candidate_count": retr_meta.get("candidate_count", 0),
            "top_scores":     retr_meta.get("top_scores", []),
            "chunks_returned": 0,
            "chunks_used":    [],
            "response_length": len(partial),
            "path":           path,
        })
        return

    debug_md     = _format_debug_md(message, retrieval_query, retr_meta, chunks, "grounded response")
    context      = build_context_block(chunks)
    sources_md   = format_sources_md(chunks)
    sources_tail = SOURCES_DELIMITER + sources_md
    chunk_state_value, chunk_choices = _build_chunk_state(chunks)
    chunk_selector_update = gr.update(choices=chunk_choices, value=None)

    # Build LLM messages: system + trimmed history (sources stripped) + new query.
    # Each turn = 2 messages (user + assistant), so trim by MAX_HISTORY * 2.
    max_msgs = MAX_HISTORY * 2 if MAX_HISTORY > 0 else len(history)
    trimmed_history = history[-max_msgs:] if MAX_HISTORY > 0 else history
    messages = [{"role": "system", "content": build_system_prompt()}]
    for msg in trimmed_history:
        raw = _content_text(msg["content"])
        content = raw if msg["role"] == "user" else strip_sources(raw)
        messages.append({"role": msg["role"], "content": content})
    messages.append({
        "role": "user",
        "content": (
            f"These are your own words from past satsangs:\n\n"
            f"{context}\n\n"
            f"A seeker comes to you and says: {message}\n\n"
            f"Draw on the wisdom in these passages and apply it to what the seeker has shared. "
            f"The passages may speak in general terms — bring out what is relevant to this seeker's specific situation. "
            f"Do not introduce new facts, names, numbers, or stories that are not present in the passages. "
            f"Only say you have not spoken of this if the passages are genuinely unrelated to the seeker's question.\n\n"
            f"Speak in first person throughout — you ARE Maharajji addressing this seeker directly. "
            f"Never refer to Maharajji in the third person. No headers, no bullet points, no numbered lists. "
            f"Do not cite video titles or episode numbers — sources are shown separately. "
            f"Let the words flow as they would in satsang: unhurried, warm, from the heart.\n\n"
            f"The seeker wrote in {detect_language(message)} — respond in that same language."
        ),
    })

    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": ""},
    ]

    streamer = stream_ollama(messages, thinking=thinking) if backend == "Local (Ollama)" \
        else stream_claude(messages)

    partial = ""
    for partial in streamer:
        history[-1]["content"] = partial + sources_tail
        yield history, "", debug_md, chunk_state_value, chunk_selector_update

    if not partial:
        history[-1]["content"] = (
            "*(No response received from the model. "
            "If using Ollama, check that `ollama serve` is running and `qwen3:8b` is pulled.)*"
        ) + sources_tail
        yield history, "", debug_md, chunk_state_value, chunk_selector_update
    else:
        # Update user context file in background — no latency impact on the UI.
        threading.Thread(
            target=_update_context_job,
            args=(message, strip_sources(partial)),
            daemon=True,
        ).start()

    _log_jsonl(QUERY_LOG_FILE, {
        "ts":             datetime.now().isoformat(),
        "backend":        backend,
        "query":          message,
        "reformulated":   retrieval_query if was_reformulated else None,
        "is_situational": is_situational(message),
        "candidate_count": retr_meta.get("candidate_count", 0),
        "top_scores":     retr_meta.get("top_scores", []),
        "chunks_returned": len(chunks),
        "chunks_used": [
            {
                "video_id": c["metadata"].get("video_id"),
                "title":    c["metadata"].get("title"),
                "start":    c["metadata"].get("start"),
                "score":    c["score"],
            }
            for c in chunks
        ],
        "response_length": len(partial),
        "path":           "grounded response",
    })


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def main():
    global _user_context
    _user_context = load_user_context()
    if _user_context:
        print(f"Loaded user context ({len(_user_context.splitlines())} lines)")

    print("Loading vector index...")
    try:
        count = get_collection().count()
        print(f"Ready — {count} chunks indexed")
    except Exception as e:
        print(f"\nFailed to load index: {e}")
        print("Run 2_build_index.py first to build the index.")
        return

    bm25, bm25_ids = get_bm25()
    if bm25 is not None:
        print(f"BM25 index loaded — {len(bm25_ids)} entries")
    else:
        print("BM25 index not found — re-run 2_build_index.py to generate it (keyword search disabled)")

    print(f"Loading reranker ({CROSS_ENCODER_MODEL})...")
    get_reranker()
    print("Reranker ready")

    if count == 0:
        print("Index is empty — run 2_build_index.py after transcribing some videos.")
        return

    with gr.Blocks(title="Premanand Maharajji Satsang Chatbot") as demo:
        gr.Markdown(
            "# Premanand Maharajji Satsang Chatbot\n"
            "Ask questions about his teachings. Every answer is grounded in actual satsang transcripts."
        )

        with gr.Row():
            backend_radio = gr.Radio(
                choices=["Local (Ollama)", "Claude API"],
                value="Local (Ollama)",
                label="LLM Backend",
                scale=2,
            )
            thinking_toggle = gr.Checkbox(
                value=False,
                label="Thinking mode (Qwen3 only — slower, deeper philosophical reasoning)",
                scale=2,
            )
            with gr.Column(scale=1):
                gr.Markdown(f"*{count} chunks indexed*")

        chatbot = gr.Chatbot(elem_classes=["chatbot"])

        with gr.Accordion("Retrieval details", open=False):
            debug_panel = gr.Markdown("*(ask a question to see retrieval details)*")

        with gr.Accordion("Source chunks (view & edit transcripts)", open=False):
            gr.Markdown(
                "*Pick a source from the last response to view its full text. "
                "If the transcript is wrong, edit and save — the correction is re-embedded immediately "
                "and used by all future queries. Audit log in `chroma_db/chunk_corrections.jsonl`.*"
            )
            chunk_state    = gr.State({})
            chunk_selector = gr.Dropdown(label="Select chunk", choices=[], interactive=True)
            chunk_text_box = gr.Textbox(
                label="Chunk text (editable)",
                lines=10, max_lines=25, interactive=True,
            )
            with gr.Row():
                save_chunk_btn = gr.Button("Save & re-embed", variant="primary", scale=1)
                chunk_status   = gr.Markdown("", elem_classes=["chunk-status"])

        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Ask about Maharajji's teachings...",
                label="Your question",
                lines=2,
                scale=7,
            )
            submit_btn = gr.Button("Ask", variant="primary", scale=1)
            stop_btn   = gr.Button("Stop", variant="stop", scale=1)

        clear_btn = gr.Button("Clear conversation")

        gr.Examples(
            examples=[
                "What does Maharajji say about the nature of the mind?",
                "How should one practice bhakti according to Maharajji?",
                "What does Maharajji teach about liberation (moksha)?",
                "What is the importance of satsang in spiritual life?",
            ],
            inputs=msg_box,
        )

        # Disable thinking toggle when Claude is selected — it only applies to Qwen3/Ollama.
        backend_radio.change(
            fn=lambda b: gr.update(interactive=(b == "Local (Ollama)"), value=False),
            inputs=backend_radio,
            outputs=thinking_toggle,
        )

        chat_outputs = [chatbot, msg_box, debug_panel, chunk_state, chunk_selector]
        submit_event = submit_btn.click(
            chat, [msg_box, chatbot, backend_radio, thinking_toggle], chat_outputs
        )
        submit_event_kb = msg_box.submit(
            chat, [msg_box, chatbot, backend_radio, thinking_toggle], chat_outputs
        )
        clear_btn.click(
            lambda: ([], "", "*(ask a question to see retrieval details)*", {},
                    gr.update(choices=[], value=None), "", ""),
            outputs=[chatbot, msg_box, debug_panel, chunk_state,
                     chunk_selector, chunk_text_box, chunk_status],
        )

        # When user picks a chunk from the dropdown, load its text into the textbox.
        def on_select_chunk(chunk_id, state):
            if not chunk_id or chunk_id not in state:
                return "", ""
            return state[chunk_id]["text"], ""

        chunk_selector.change(
            on_select_chunk,
            inputs=[chunk_selector, chunk_state],
            outputs=[chunk_text_box, chunk_status],
        )

        # Save the edited text — updates ChromaDB, BM25, and the audit log.
        def on_save_chunk(chunk_id, new_text, state):
            if not chunk_id:
                return state, "*Select a chunk first.*"
            ok, msg = update_chunk(chunk_id, new_text)
            if ok and chunk_id in state:
                state[chunk_id] = {**state[chunk_id], "text": new_text}
            prefix = "✓ " if ok else "⚠️ "
            return state, prefix + msg

        save_chunk_btn.click(
            on_save_chunk,
            inputs=[chunk_selector, chunk_text_box, chunk_state],
            outputs=[chunk_state, chunk_status],
        )

        # Stop button cancels the in-flight chat generator — the streamer's for-loop
        # exits, no further yields happen, partial response stays in the chat as-is.
        stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[submit_event, submit_event_kb])

        def on_like(data: gr.LikeData, hist):
            """Append a feedback record to logs/feedback.jsonl when user clicks thumbs."""
            idx = data.index[0] if isinstance(data.index, (list, tuple)) else data.index
            if idx is None or idx >= len(hist):
                return
            assistant_msg = hist[idx]
            user_msg = hist[idx - 1] if idx > 0 else None
            _log_jsonl(FEEDBACK_LOG_FILE, {
                "ts":       datetime.now().isoformat(),
                "query":    _content_text(user_msg["content"]) if user_msg else None,
                "response": strip_sources(_content_text(assistant_msg["content"])),
                "liked":    data.liked,
            })

        chatbot.like(on_like, inputs=chatbot, outputs=None)

    demo.launch(inbrowser=True, theme=gr.themes.Soft(), css=".chatbot { min-height: 460px; }")


if __name__ == "__main__":
    main()
