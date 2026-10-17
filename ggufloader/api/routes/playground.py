"""Playground routes - a plain streaming chatbot for showing model speed.

Deliberately **not** an agent: no planner, no tools, no plan steps. Load a
GGUF (any GGUF - the pinned-model gate is bypassed here on purpose), send a
message, and stream the reply back token by token together with live speed
stats so a viewer can watch:

* the tokens as they stream,
* how many tokens have been generated,
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

from ggufloader.api.deps import (
    get_model_backend,
    model_lifecycle_lock,
    set_model_backend,
    get_router,
)
from ggufloader.core.defaults import STOP_TOKENS_UNIFIED
from ggufloader.core.testing.runner import _count_tokens

router = APIRouter()
logger = logging.getLogger(__name__)

# How often (seconds) to push a live `stats` event while tokens stream.
_STATS_INTERVAL_S = 0.25


def _is_standalone_chat_model(filename: str) -> bool:
    """Companion files like ``*-mmproj-*.gguf`` are the vision projector
    that belongs to a parent model (e.g. gemma4-12b-q4_k_m.gguf). They are
    only ~100-800 MB and cannot be loaded as a standalone LLM - offering
    them in the chat dropdown just produces a confusing load error, so the
    playground catalog hides them.
    """
    return "mmproj" not in filename.lower()


class PlaygroundMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatStreamRequest(BaseModel):
    message: str
    history: Optional[List[PlaygroundMessage]] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None  # None = family default (model_families.json)
    max_tokens: int = 512


class LoadModelRequest(BaseModel):
    path: str
    use_gpu: Optional[bool] = None   # None = auto-detect via router (GPU when available)
    n_ctx: Optional[int] = None      # None = router picks


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stats(backend: Any, parts: List[str], t0: float, first: Optional[float],
           last: Optional[float]) -> dict:
    """Compute the live speed readout for the text generated so far.

    ``tokens_per_second`` is the *decode* rate (tokens divided by the time
    since the first token), which is what people mean by "speed" for a local
    model. ``total_tokens_per_second`` includes the prompt/prefill time.
    """
    text = "".join(parts)
    tokens = _count_tokens(backend, text)
    now = time.perf_counter()
    elapsed_ms = (now - t0) * 1000.0
    if first is not None:
        ttft_ms = (first - t0) * 1000.0
        decode_ms = max(0.0, ((last or first) - first) * 1000.0)
    else:
        ttft_ms = elapsed_ms
        decode_ms = 0.0
    tps = (tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
    total_tps = (tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0
    return {
        "tokens_generated": tokens,
        "elapsed_ms": round(elapsed_ms, 1),
        "ttft_ms": round(ttft_ms, 1),
        "decode_ms": round(decode_ms, 1),
        "tokens_per_second": round(tps, 2),
        "total_tokens_per_second": round(total_tps, 2),
    }


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
        if _is_standalone_chat_model(m.get("filename", ""))
    ]
    return {"directory": catalog.get("directory"), "count": len(models), "models": models}


@router.post("/model/load")
async def load_model(req: LoadModelRequest) -> dict:
    """Load a GGUF for chatting, replacing whatever was loaded.

    This bypasses the pinned-model gate on purpose: the point of the
    playground is to run *different* models and watch how fast they are.

    The swap mirrors ``/api/model/load``: one global lifecycle lock, old
    model unloaded deterministically BEFORE the new one is constructed.
    The previous drop-the-reference-only flow left the C-level llama_free
    to a random GC pass - which frequently fired inside the very next
    Llama() construction and segfaulted the whole server, making chat-tab
    model loading die silently. use_gpu now defaults to the router's
    auto-detection (GPU when the build supports it) instead of forcing CPU.
    """
    if not os.path.exists(req.path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.path}")

    # A mmproj file is a vision-projector companion, not a chat model - fail
    # with a self-explanatory message instead of the generic fallback-ladder
    # error ("tried full_gpu and cpu") that made it look like a settings
    # problem ("switch to Q4 / smaller ctx / CPU").
    if not _is_standalone_chat_model(os.path.basename(req.path)):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{os.path.basename(req.path)} is a vision companion file "
                "(mmproj), not a standalone model. Load the main model it "
                "belongs to instead."
            ),
        )

    # An explicit device choice (CPU/GPU toggle in the chat bar) is
    # remembered so the startup auto-load uses the same device next time.
    if req.use_gpu is not None:
        from ggufloader.api.auto_load import remember_device_preference
        remember_device_preference(req.use_gpu)

    from ggufloader.core.router import ModelRole

    try:
        with model_lifecycle_lock():
            # Free the current model first so only one Llama ever exists -
            # deterministically, before the new constructor runs.
            current = get_model_backend()
            if current is not None:
                try:
                    current.unload()
                except Exception as e:  # noqa: BLE001 - freeing memory is best-effort
                    logger.warning("Failed to unload previous model: %s", e)
                set_model_backend(None)

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
    """What the playground is currently chatting with.

    architecture/quantization come from the router's (cached) metadata
    inspect of the loaded file - ModelBackend itself does not carry them,
    which is why this endpoint used to always return null/null.
    """
    backend = get_model_backend()
    if backend is None:
        return {"loaded": False}
    path = getattr(backend, "model_path", None)
    # Backends may carry their own metadata (test fakes, future fields);
    # the real ModelBackend does not, so fall back to the router's cached
    # metadata-only inspect of the loaded file.
    architecture = getattr(backend, "architecture", None)
    quantization = getattr(backend, "quantization", None)
    if path and (architecture is None or quantization is None):
        try:
            profile = get_router().inspect(path)  # cached, metadata-only
            architecture = architecture or profile.architecture
            quantization = quantization or profile.quantization
        except Exception:  # noqa: BLE001 - cosmetic fields only
            pass
    return {
        "loaded": True,
        "path": path,
        "filename": os.path.basename(path) if path else None,
        "architecture": architecture,
        "quantization": quantization,
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


def _generation_kwargs(backend: Any, req: ChatStreamRequest) -> dict:
    """Sampling defaults from the loaded model's family + request overrides.

    The playground used to forward ONLY temperature/max_tokens: no
    repeat_penalty, no top_k - and fatally, no stop strings. A model that
    missed its EOS (or was rendered through a wrong chat template) then
    repeated garbage until max_tokens. Family defaults
    (model_families.json, overridable per model via model_params.json)
    now fill in the rest; an explicit request temperature still wins.
    """
    kwargs: dict[str, Any] = {
        "max_tokens": req.max_tokens,
        "stop": list(STOP_TOKENS_UNIFIED),
    }
    try:
        from ggufloader.core.llm.model_profiles import resolve_chat_config
        config = resolve_chat_config(getattr(backend, "model_path", "") or "")
        for k in ("temperature", "top_k", "top_p", "repeat_penalty", "min_p"):
            v = config.get("params", {}).get(k)
            if v is not None:
                kwargs[k] = v
    except Exception as e:  # noqa: BLE001 - defaults only
        logger.debug("Family sampling defaults unavailable: %s", e)
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature
    return kwargs


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
    gen_kwargs = _generation_kwargs(backend, req)

    def event_stream() -> Iterator[str]:
        parts: List[str] = []
        t0 = time.perf_counter()
        first: Optional[float] = None
        last: Optional[float] = None
        last_emit = 0.0
        try:
            for piece in backend.chat_stream(messages=messages, **gen_kwargs):
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
                parts.append(piece)
                yield _sse("token", {"token": piece})
                if now - last_emit >= _STATS_INTERVAL_S:
                    last_emit = now
                    yield _sse("stats", _stats(backend, parts, t0, first, last))

            done = _stats(backend, parts, t0, first, last)
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

    parts: List[str] = []
    t0 = time.perf_counter()
    first: Optional[float] = None
    last: Optional[float] = None
    try:
        for piece in backend.chat_stream(
            messages=_build_messages(req), **_generation_kwargs(backend, req)
        ):
            now = time.perf_counter()
            if first is None:
                first = now
            last = now
            parts.append(piece)
    except Exception as e:  # noqa: BLE001
        logger.error("Playground chat failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    out = _stats(backend, parts, t0, first, last)
    out["content"] = "".join(parts)
    return out
