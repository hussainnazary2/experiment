"""Playground routes - a plain streaming chatbot for showing model speed.

Deliberately **not** an agent: no planner, no tools, no plan steps. Load a
GGUF (any GGUF - the pinned-model gate is bypassed here on purpose), send a
message, and stream the reply back token by token together with live speed
stats so a viewer can watch:

* the tokens as they stream,
* how many tokens have been generated and the running prompt/total counts,
* tokens/sec (decode speed) and time-to-first-token.

Endpoints
---------
GET  /api/playground/models        candidate GGUFs on disk
POST /api/playground/model/load    load a GGUF (ungated) and make it the backend
GET  /api/playground/model         what is loaded right now
POST /api/playground/chat/stream   SSE: token / stats / done / error
POST /api/playground/chat          non-streaming fallback
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Iterator, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ggufloader.api.deps import get_model_backend, set_model_backend, get_router
from ggufloader.core.testing.runner import _count_tokens

router = APIRouter()
logger = logging.getLogger(__name__)

# How often (seconds) to push a live `stats` event while tokens stream.
_STATS_INTERVAL_S = 0.25


class PlaygroundMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatStreamRequest(BaseModel):
    message: str
    history: Optional[List[PlaygroundMessage]] = None
    system_prompt: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 512


class LoadModelRequest(BaseModel):
    path: str
    use_gpu: bool = False          # CPU by default: this is a CPU speed demo
    n_ctx: Optional[int] = None    # None = router picks


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stats(backend: Any, parts: List[str], t0: float, first: Optional[float],
           last: Optional[float], prompt_tokens: int = 0) -> dict:
    """Compute the live speed readout for the text generated so far.

    ``tokens_per_second`` is the *decode* rate (tokens divided by the time
    since the first token), which is what people mean by "speed" for a local
    model. ``total_tokens_per_second`` includes the prompt/prefill time.

    Token counts: ``prompt_tokens`` is what went in (system + history +
    message, counted with the model's own tokenizer), ``tokens_generated``
    is what came out so far, and ``total_tokens`` is their sum.
    """
    text = "".join(parts)
    generated = _count_tokens(backend, text)
    now = time.perf_counter()
    elapsed_ms = (now - t0) * 1000.0
    if first is not None:
        ttft_ms = (first - t0) * 1000.0
        decode_ms = max(0.0, ((last or first) - first) * 1000.0)
    else:
        ttft_ms = elapsed_ms
        decode_ms = 0.0
    tps = (generated / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
    total_tps = (generated / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0
    return {
        "prompt_tokens": prompt_tokens,
        "tokens_generated": generated,
        "total_tokens": prompt_tokens + generated,
        "elapsed_ms": round(elapsed_ms, 1),
        "ttft_ms": round(ttft_ms, 1),
        "decode_ms": round(decode_ms, 1),
        "tokens_per_second": round(tps, 2),
        "total_tokens_per_second": round(total_tps, 2),
    }


def _count_prompt_tokens(backend: Any, messages: List[dict]) -> int:
    """Token count of the prompt parts, using the model's own tokenizer.

    Counted from the message contents (plus a small per-message allowance
    for role/turn markers the chat template adds) - close enough for a
    live prompt/total readout without rendering the template twice.
    """
    blob = "\n".join(str(m.get("content") or "") for m in messages)
    if not blob:
        return 0
    return _count_tokens(backend, blob) + 4 * len(messages)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@router.get("/models")
async def list_models(directory: Optional[str] = None) -> dict:
    """GGUF files available to chat with (reuses the model catalog scan)."""
    from ggufloader.api.routes.model import model_catalog

    catalog = await model_catalog(directory=directory, recursive=True)
    models = [
        {
            "path": m["path"],
            "filename": m["filename"],
            "size_gb": m.get("size_gb", 0),
            "size_human": m.get("size_human", ""),
            "architecture": m.get("architecture", ""),
            "quantization": m.get("quantization", ""),
            "family": m.get("family", ""),
        }
        for m in catalog.get("models", [])
    ]
    return {"directory": catalog.get("directory"), "count": len(models), "models": models}


@router.post("/model/load")
async def load_model(req: LoadModelRequest) -> dict:
    """Load a GGUF for chatting, replacing whatever was loaded.

    This bypasses the pinned-model gate on purpose: the point of the
    playground is to run *different* models and watch how fast they are.
    """
    if not os.path.exists(req.path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.path}")

    from ggufloader.core.router import ModelRole

    # Free the current model first so only one Llama ever exists.
    current = get_model_backend()
    if current is not None:
        try:
            current.unload()
        except Exception as e:  # noqa: BLE001 - freeing memory is best-effort
            logger.warning("Failed to unload previous model: %s", e)
        set_model_backend(None)

    try:
        router_ = get_router()
        backend, profile, strategy, _config = router_.load(
            req.path, ModelRole.CHAT, n_ctx=req.n_ctx, use_gpu=req.use_gpu
        )
        set_model_backend(backend)
        return {
            "status": "loaded",
            "path": req.path,
            "filename": profile.filename,
            "architecture": profile.architecture,
            "quantization": profile.quantization,
            "parameters": profile.param_count_estimate,
            "size_gb": round(float(profile.file_size_gb or 0), 2),
            "n_ctx": strategy.n_ctx,
            "use_gpu": strategy.use_gpu,
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - surface load failures to the UI
        logger.error("Playground model load failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model")
async def model_status() -> dict:
    """What the playground is currently chatting with."""
    backend = get_model_backend()
    if backend is None:
        return {"loaded": False}
    path = getattr(backend, "model_path", None)
    return {
        "loaded": True,
        "path": path,
        "filename": os.path.basename(path) if path else None,
        "architecture": getattr(backend, "architecture", None),
        "quantization": getattr(backend, "quantization", None),
        "n_ctx": getattr(backend, "n_ctx", None),
        "use_gpu": bool(getattr(backend, "use_gpu", False)),
    }


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def _build_messages(req: ChatStreamRequest) -> List[dict]:
    messages: List[dict] = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    for m in (req.history or []):
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": req.message})
    return messages


@router.post("/chat/stream")
async def chat_stream(req: ChatStreamRequest):
    """Stream the reply as SSE: token / stats / done / error.

    ``stats`` events are pushed about 4x/second while generating so the UI can
    show a live tokens/sec and a running token count.
    """
    backend = get_model_backend()
    if backend is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    messages = _build_messages(req)
    temperature = req.temperature
    max_tokens = req.max_tokens
    prompt_tokens = _count_prompt_tokens(backend, messages)

    def event_stream() -> Iterator[str]:
        parts: List[str] = []
        t0 = time.perf_counter()
        first: Optional[float] = None
        last: Optional[float] = None
        last_emit = 0.0
        try:
            for piece in backend.chat_stream(
                messages=messages, temperature=temperature, max_tokens=max_tokens
            ):
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
                parts.append(piece)
                yield _sse("token", {"token": piece})
                if now - last_emit >= _STATS_INTERVAL_S:
                    last_emit = now
                    yield _sse("stats", _stats(
                        backend, parts, t0, first, last, prompt_tokens))

            done = _stats(backend, parts, t0, first, last, prompt_tokens)
            done["content"] = "".join(parts)
            yield _sse("done", done)
        except Exception as e:  # noqa: BLE001 - report into the stream
            logger.error("Playground chat failed: %s", e)
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat")
async def chat(req: ChatStreamRequest) -> dict:
    """Non-streaming fallback: the whole reply plus its final stats."""
    backend = get_model_backend()
    if backend is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    messages = _build_messages(req)
    parts: List[str] = []
    t0 = time.perf_counter()
    first: Optional[float] = None
    last: Optional[float] = None
    try:
        for piece in backend.chat_stream(
            messages=messages, temperature=req.temperature, max_tokens=req.max_tokens
        ):
            now = time.perf_counter()
            if first is None:
                first = now
            last = now
            parts.append(piece)
    except Exception as e:  # noqa: BLE001
        logger.error("Playground chat failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    out = _stats(backend, parts, t0, first, last, _count_prompt_tokens(backend, messages))
    out["content"] = "".join(parts)
    return out
