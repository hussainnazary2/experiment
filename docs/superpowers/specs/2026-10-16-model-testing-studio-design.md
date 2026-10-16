# Local model speed chatbot — design

**Date:** 2026-10-16
**Status:** implemented

## Goal

Show a viewer how fast a local GGUF model runs on CPU. The app must put
everything needed on screen:

1. the reply **streaming token by token**,
2. **tokens/sec** (decode speed),
3. **how many tokens have been generated**,
4. time-to-first-token and elapsed time for context.

It is a **plain chatbot** — no agent, no planner, no tools — and the app does
**not** record the screen; the user captures that with their own tool.

## Why new code was needed

- The app's chat is agent-driven: `frontend/src/components/chat/ChatPanel.tsx`
  talks to `/ws`, which runs the LangGraph planner/agent. Even a trivial
  "hello" goes through a planner.
- `POST /api/chat/stream` exists but streams from whatever model is loaded and
  reports `tokens_used = len(content.split())` — a word count, not tokens, and
  no timing at all.
- Loading a model is gated to the single pinned GGUF
  (`api/routes/model.py::_require_pinned_target`), so you cannot test other
  models.

## Architecture

```
frontend/src/components/chat/PlaygroundChat.tsx     (center view, default)
  |  POST /api/playground/chat/stream   (SSE)
  v
ggufloader/api/routes/playground.py
  |  get_model_backend()  ->  core/llm/model_backend.py (the only Llama)
  |  router.load(...)     ->  ungated model load
  v
llama.cpp
```

### Backend

`ggufloader/api/routes/playground.py`, mounted at `/api/playground` in
`api/app.py`:

| Method | Path | Purpose |
|---|---|---|
| GET | `/models` | GGUF files on disk (reuses the model catalog scan) |
| POST | `/model/load` | load **any** GGUF, replacing the current one |
| GET | `/model` | what is loaded now |
| POST | `/chat/stream` | SSE: `token` / `stats` / `done` / `error` |
| POST | `/chat` | non-streaming fallback with the same stats |

`/model/load` frees the current backend first (`set_model_backend(None)`) and
then loads through `core/router.py`, so **at most one `Llama` exists** — the
repo's invariant 1 still holds. It deliberately does not call the pinned-target
gate.

`/chat/stream` measures on the server, from the real stream:

- `ttft_ms` = request start → first streamed piece;
- `decode_ms` = first piece → last piece;
- `tokens_generated` = `ModelBackend.count_tokens(text)` (the model's own
  tokenizer);
- `tokens_per_second` = tokens / decode time;
- `total_tokens_per_second` = tokens / wall time (includes prefill).

A `stats` event is pushed about every 0.25 s while generating, so the numbers
move live; `done` carries the final figures plus the full text.

### Frontend

`frontend/src/components/chat/PlaygroundChat.tsx`:

- model `<select>` + **Load**, and a chip showing the loaded file, quant,
  context size and CPU/GPU;
- four stat cards: **Tokens/sec** (accented, the headline number),
  **Tokens generated**, **First token**, **Elapsed**;
- the conversation, with the assistant reply rendered through the existing
  `MarkdownRenderer` and a blinking cursor while streaming;
- composer with Enter-to-send, **Stop** (aborts the `fetch`), and **Clear**,
  plus a max-tokens field.

It is wired as the **default center view** in
`frontend/src/components/layout/AppLayout.tsx`, with a small Chatbot / Agent
switch (`uiStore.centerView`). The agent chat is untouched and still one click
away.

## Removed

In-app screen recording: `frontend/src/hooks/useScreenRecorder.ts`, the
recording HUD, and the Electron `save-recording` IPC in `electron/main.ts` and
`electron/preload.ts`.

## Left in place (unwired)

The multi-model benchmark engine `ggufloader/core/testing/runner.py`, its
`/api/benchmark/suite/*` routes, and their tests still exist and pass, but no
UI surfaces them any more. Delete them if the suite is not wanted later.

## Testing

- `tests/unit/test_playground.py` — headless: exact TTFT/decode-rate math,
  SSE token/stats/done parsing, history + system prompt forwarding, the 400
  with no model, error reporting inside the stream, the non-streaming
  endpoint, model status, and route mounting.
- `python -m pytest tests/unit` stays green (aside from the pre-existing
  llama-cpp-python/PySide6-dependent failures).
- `cd frontend && npx tsc -b && npm run build`; `cd electron && npx tsc --noEmit`.
