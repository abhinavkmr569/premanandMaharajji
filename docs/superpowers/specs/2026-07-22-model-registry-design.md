# Model Registry & Launch-Time Model Selection

**Date:** 2026-07-22
**File touched:** `3_chatbot.py` (plus `CLAUDE.md`, `README.md`)
**Status:** approved design, not yet implemented

## Problem

`3_chatbot.py` hard-codes `OLLAMA_MODEL = "qwen3:8b"` at line 42 and reads it from four
call sites. Evaluating a second model (Gemma 4 E4B) currently means editing the constant,
which loses the qwen3 configuration and makes it impossible to tell from the logs which
model produced which answer.

Generation parameters are also model-specific in a way the current code does not express.
`repeat_penalty = 1.1` and `repeat_last_n = 512` were tuned against a Qwen3 repetition
failure mode. Applying them unchanged to a different architecture is inheritance, not
tuning.

A separate defect surfaced while designing this: `num_ctx` is never set, so Ollama applies
its 4096-token default and truncates the prompt **from the beginning, silently**. The
grounded-path prompt is roughly 5000–6000 tokens (system prompt and user context ~500–800;
five chunks at up to 300 words ~2000; four history turns ~2000–3000; closing instruction
block ~250). The content dropped first is the system prompt — the persona definition and
the "speak only from the passages" constraint. This affects the current qwen3 setup today,
independent of any model swap.

## Goals

1. Choose the Ollama model at launch, consciously, on every run.
2. Preserve the current qwen3 configuration byte-for-byte as one registry entry.
3. Carry per-model generation parameters, context length, and thinking mechanism.
4. Make `logs/queries.jsonl` and `logs/feedback.jsonl` splittable by model.
5. Set `num_ctx` explicitly for both models.

## Non-goals

- No live model switching inside the UI. One model per process.
- No second model resident in VRAM. The 8188 MiB card cannot hold qwen3:8b (~5 GB) and
  gemma4:e4b-it-qat (6.1 GB) together; Ollama would evict and reload on every query.
- No raising Gemma to its full 128K context. That would blow the VRAM budget and make the
  comparison against qwen3 meaningless.
- No changes to retrieval, chunking, or the Claude API backend.

## Design

### 1. Registry

Replace `OLLAMA_MODEL` with a `MODELS` dict keyed by short name.

```python
MODELS = {
    "qwen3": {
        "label":     "Qwen3 8B  (~5 GB VRAM)",
        "tag":       "qwen3:8b",
        "num_ctx":   8192,
        "thinking_mode": "api",
        "thinking_temperature": 1.0,
        "chat_options": {
            "temperature":   0.3,
            "repeat_penalty": 1.1,
            "repeat_last_n":  512,
            "top_p":          0.9,
            "num_predict":    2048,
        },
        "helper_options": {"temperature": 0.1},
    },
    "gemma4": {
        "label":     "Gemma 4 E4B QAT  (~6.1 GB VRAM)",
        "tag":       "gemma4:e4b-it-qat",
        "num_ctx":   8192,
        "thinking_mode": "prompt_token",
        "thinking_temperature": 1.0,
        "chat_options": {
            "temperature":   0.3,
            "top_p":         0.95,
            "num_predict":   2048,
        },
        "helper_options": {"temperature": 0.1},
    },
}

DEFAULT_MODEL = "qwen3"
ACTIVE: dict = MODELS[DEFAULT_MODEL]   # rebound once at startup
```

`qwen3.chat_options` is copied verbatim from the current `stream_ollama()` call at line 542.
Nothing is re-tuned.

`gemma4` deliberately omits `repeat_penalty` and `repeat_last_n`, falling back to Ollama
defaults. `_has_repetition_loop()` still guards the failure case at the stream level. If
Gemma loops in practice, they get added back with evidence.

`num_ctx: 8192` on both — identical for a fair comparison, and it fits: qwen3:8b leaves
~2.2 GB headroom for KV cache, gemma4 e4b-qat leaves ~1.1 GB, against ~7.2 GB usable after
the Windows desktop reserve.

### 2. Launch-time selection

New `select_model()`, called at the top of `main()` **before** the index and reranker load,
so the decision is made before the ~20-second startup wait.

Behavior:

- Shell out to `ollama list`. Mark each registry entry as pulled or not pulled. A failure
  to run `ollama` is non-fatal — print a note and show every entry unmarked.
- Print a numbered menu of `label` values with pulled status.
- Read a choice from stdin. Empty input selects `DEFAULT_MODEL`. Invalid input re-prompts.
- If `sys.stdin.isatty()` is false, skip the prompt entirely, print a loud warning naming
  the fallback, and use `DEFAULT_MODEL`. This prevents `EOFError` killing the process when
  launched from a wrapper or with piped stdin.
- Rebind the module-level `ACTIVE` to the chosen config and print the active tag.

Selecting a model that is not pulled is allowed — Ollama will pull it on first request, and
blocking on it would be more annoying than the wait.

### 3. Call sites

All four existing Ollama calls read from `ACTIVE`:

| Function | Line (current) | Options used |
|---|---|---|
| `stream_ollama()` | 527 | `ACTIVE["chat_options"]`, plus `num_ctx` |
| `reformulate_query()` | 236 | `ACTIVE["helper_options"]`, plus `num_ctx` |
| `_extract_context_update()` | 126 | `ACTIVE["helper_options"]`, plus `num_ctx` |
| `_compress_user_context()` | 165 | `ACTIVE["helper_options"]`, plus `num_ctx` |

Helpers use the same model as chat. Only one model is ever resident, so there is no
eviction and no reload cost.

### 4. Thinking mode

Two mechanisms, dispatched on `ACTIVE["thinking_mode"]`, entirely inside `stream_ollama()`.
Keeping the branch there means `chat()` is unchanged and the Claude path
(`stream_claude()`) is untouched.

**`"api"` (qwen3):** current behavior. Send `"think": thinking`. Ollama routes reasoning to
`message.thinking` and it never reaches `message.content`.

**`"prompt_token"` (gemma4):** when `thinking` is true, prepend `<|think|>` to the content
of `messages[0]` if that message has `role == "system"`; if it does not, insert a new system
message carrying only the token. Do **not** send the `think` key at all — passing it to a
model whose template does not declare the thinking capability makes Ollama return an error,
which the stream surfaces as `Ollama error: ...` in place of the answer.

**`None`:** never send `think`, never prepend. Reserved for future entries.

Temperature in both cases is `ACTIVE["thinking_temperature"]` when thinking is on, otherwise
`chat_options["temperature"]`.

#### Reasoning leakage — the one unverified piece

With `"prompt_token"`, reasoning arrives inside `message.content` as structured tags rather
than in the separate `message.thinking` field. The existing content/thinking split in
`stream_ollama()` does not catch it, so reasoning would stream straight into the visible
answer.

A content-side filter is required. The exact tag strings Gemma 4 emits are not documented in
any source found during design, so the filter must be verified empirically on first run.

**Fail-safe requirement:** the filter suppresses output only after a recognized opening tag
is seen. If no recognized tag appears, it streams normally. It must never be possible for an
unrecognized format to swallow the entire response. Recognized tags live in one module-level
tuple so they can be corrected in one place once observed.

Until the tag format is confirmed against a real response, treat Gemma thinking mode as
unverified. Standard (non-thinking) mode has no such dependency and works regardless.

### 5. UI

No dropdown — selection already happened at launch.

- Header line shows the active model label alongside the existing chunk count.
- Thinking checkbox label is built from the active model rather than hard-coding "Qwen3
  only".
- Checkbox is non-interactive when `ACTIVE["thinking_mode"]` is `None`, and — as today —
  when the Claude API backend is selected. The existing `backend_radio.change` handler at
  line 906 keeps its Claude behavior and gains the `thinking_mode is None` condition.

### 6. Logging

Add `"model": ACTIVE["tag"]` to:

- both `_log_jsonl(QUERY_LOG_FILE, ...)` records in `chat()` (lines 711 and 787)
- the feedback record written by `on_like`

Without this the accumulated thumbs data cannot be split by model and the comparison the
whole feature exists to enable is not possible.

### 7. Stale references

Four hard-coded `qwen3:8b` strings become the active tag or a neutral phrasing:

- line 7 — module docstring
- line 552 — "Ollama is not running" message
- line 776 — "no response received" message
- `README.md` and `CLAUDE.md` — model rows, VRAM table, setup instructions

## Error handling

| Case | Behavior |
|---|---|
| `ollama list` unavailable | Menu still shows, entries unmarked, note printed |
| Non-TTY stdin | Loud warning, fall back to `DEFAULT_MODEL`, no raise |
| Invalid menu input | Re-prompt |
| Selected model not pulled | Allowed; Ollama pulls on first request |
| `think` sent to non-thinking model | Prevented by construction — `"prompt_token"` and `None` never send the key |
| Unrecognized reasoning tags | Stream normally; never swallow the response |
| Ollama not running | Existing `ConnectionError` handler, message now names the active tag |

## Verification

1. `python 3_chatbot.py`, press Enter — qwen3 selected, generation parameters identical to
   pre-change. Confirm by diffing a `queries.jsonl` row against a pre-change row.
2. Confirm `num_ctx` is actually applied: run a long multi-turn conversation and check the
   persona holds where it previously would have been truncated.
3. Select gemma4, thinking off, ask a grounded question. Confirm answer streams, sources
   render, `model` appears in the log line.
4. Select gemma4, thinking on. Inspect raw `message.content` in the terminal for the tag
   format. Record the actual strings and correct the recognized-tag tuple.
5. Non-TTY check: pipe empty stdin, confirm the warning prints and the process starts on
   qwen3 rather than raising.
6. Thumbs up/down on one response per model; confirm both `feedback.jsonl` rows carry
   distinct `model` values.

## Open items

- Gemma 4 reasoning tag strings — resolved by verification step 4.
- Whether 8192 `num_ctx` is enough, or whether the prompt still overflows. Measurable once
  logging is in place.
