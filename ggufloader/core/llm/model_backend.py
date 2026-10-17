"""
ModelBackend - Thread-safe wrapper around the llama-cpp-python runtime.

This is the *only* module in the application that talks directly to
llama_cpp. It exposes a callable interface compatible with the raw
``Llama`` object (``model(prompt, stream=True, ...)``), so UI code and
addons that historically called the raw object keep working, while all
access is serialized through a lock to prevent concurrent-call crashes.

This module is intentionally Qt-free so it can be unit tested in
isolation and reused from any worker thread.
"""

from __future__ import annotations

import gc
import logging
import threading
from typing import Any, Dict, Iterator, List, Optional
logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    Llama = None
    LLAMA_AVAILABLE = False


def create_llama(**kwargs: Any) -> Any:  # pragma: no cover - thin factory
    """Construct a llama-cpp ``Llama`` instance.

    ModelBackend is the single owner of the Llama constructor; legacy
    engines that need a raw instance (e.g. the LangChain engine) go
    through here instead of importing llama_cpp themselves.
    """
    if not LLAMA_AVAILABLE:
        raise RuntimeError(
            "llama-cpp-python is required but not installed. "
            "Install it with: pip install llama-cpp-python"
        )
    return Llama(**kwargs)


def holdback_stream(token_iter: Iterator[str], stops: List[str]) -> Iterator[str]:
    """Yield tokens while withholding a tail that could complete a stop.

    Mirrors GPT4All's partial-match rule (llmodel_shared.cpp:123-140): if
    the emitted text currently ends with a proper prefix of any stop
    sequence, those characters are withheld until more text resolves them.
    llama.cpp already strips complete stops engine-side; this is the
    safety net for splits it cannot see across chunk boundaries.
    """
    stops = [s for s in (stops or []) if s]
    if not stops:
        yield from token_iter
        return
    max_len = max(len(s) for s in stops)
    pending = ""
    for chunk in token_iter:
        pending += chunk
        if not pending:
            continue
        # 1. A complete stop inside the buffer: emit up to it, drop it.
        cut = None
        cut_len = 0
        for s in stops:
            idx = pending.find(s)
            if idx != -1 and (cut is None or idx < cut):
                cut = idx
                cut_len = len(s)
        if cut is not None:
            if cut > 0:
                yield pending[:cut]
            pending = pending[cut + cut_len:]
            continue
        # 2. Release everything except a possible stop-prefix tail.
        keep = 0
        for k in range(min(max_len - 1, len(pending) - 1), 0, -1):
            if any(s.startswith(pending[-k:]) for s in stops):
                keep = k
                break
        release = len(pending) - keep
        if release > 0:
            yield pending[:release]
            pending = pending[release:]
    if pending:
        yield pending


class ModelBackend:
    """Owns a loaded GGUF model and serializes access to it."""

    def __init__(
        self,
        model_path: str,
        use_gpu: bool = False,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        n_keep: int = 512,
        flash_attn: bool = True,
        use_mmap: bool = True,
        rope_freq_base: Optional[float] = None,
    ) -> None:
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.n_gpu_layers_requested = n_gpu_layers
        self._n_ctx = n_ctx
        self._n_batch = n_batch
        self._n_threads = n_threads
        self._n_keep = n_keep
        self._flash_attn = flash_attn
        self._use_mmap = use_mmap
        self._rope_freq_base = rope_freq_base
        self._llama: Any = None
        # RLock (reentrant): chat_stream holds the lock for the whole stream
        # lifetime, and the playground's live `stats` events count tokens
        # (count_tokens re-acquires) from the SAME thread mid-stream. A plain
        # Lock deadlocked there - the stream froze after ~0.25s and every
        # later model operation (including loading) hung forever. Cross-thread
        # serialization is preserved: other threads still block until the
        # stream finishes.
        self._lock = threading.RLock()
        # A3: KV prefix cache — stores last prompt tokens + state snapshot
        # for reusing the longest common prefix across turns.
        self._kv_cache_tokens: Optional[List[int]] = None
        self._kv_cache_state: Optional[bytes] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self) -> "ModelBackend":
        """Execute exactly one load attempt at the requested gpu_layers.

        The backend executes; it does NOT decide. The retry ladder
        (full → partial → CPU-small-ctx) lives in ``ModelRouter.load``,
        which records each attempt on ``strategy.attempts``.
        """
        if not LLAMA_AVAILABLE:
            raise RuntimeError(
                "llama-cpp-python is required but not installed.\n"
                "Install it with: pip install llama-cpp-python"
            )
        gpu_layers = (self.n_gpu_layers_requested if self.use_gpu else 0)
        logger.info(
            "Loading model %s (gpu_layers=%d, ctx=%d)",
            self.model_path, gpu_layers, self._n_ctx,
        )
        llama_kwargs = dict(
            model_path=self.model_path,
            n_ctx=self._n_ctx,
            n_gpu_layers=gpu_layers,
            n_batch=self._n_batch,
            n_threads=self._n_threads or 8,
            n_keep=self._n_keep,
            flash_attn=self._flash_attn,
            use_mmap=self._use_mmap,
            n_gpu_layers_k=gpu_layers,
            n_gpu_layers_v=gpu_layers,
            verbose=True,
        )
        if self._rope_freq_base:
            llama_kwargs["rope_freq_base"] = self._rope_freq_base
        # Chat-format hint for TEMPLATE-LESS GGUFs. llama-cpp-python's
        # fallback for files without ``tokenizer.chat_template`` is the
        # llama-2 format (``<s>[INST] ... [/INST]``) - catastrophically
        # wrong for ChatML-style models: an LFM2.5 file that way degenerated
        # into echoing ``<s>hi</s> <s>world</s>`` loops. When the GGUF has
        # no embedded template but the detected family knows its format
        # (see model_families.json "chat_format"), pass it explicitly.
        # Files WITH a template (or unknown families) are untouched.
        try:
            from ggufloader.core.llm.model_profiles import (
                detect_family,
                read_gguf_general_metadata,
            )
            meta = read_gguf_general_metadata(self.model_path)
            if not meta.get("chat_template"):
                family_profile, _how = detect_family(meta, self.model_path)
                chat_format = family_profile.get("chat_format") or ""
                if chat_format:
                    llama_kwargs["chat_format"] = chat_format
                    logger.info(
                        "GGUF has no embedded chat template; using family "
                        "chat format %r for %s", chat_format, self.model_path,
                    )
        except Exception as e:  # noqa: BLE001 - never block a load on this
            logger.debug("Chat-format hint lookup failed: %s", e)
        self._llama = Llama(**llama_kwargs)
        return self

    def unload(self) -> None:
        """Release the model and free CPU/GPU memory - deterministically.

        Simply dropping the reference (the old behaviour) leaves the actual
        ``llama_free`` to a *random* later GC pass. When that pass runs inside
        the next ``Llama(...)`` construction, or on a background thread while
        another context exists, llama.cpp can segfault and take the whole
        server down - which is exactly what killed model swaps in the chat
        (playground) tab. Instead: close explicitly and collect right here,
        at a controlled point where nothing else is being constructed.
        """
        with self._lock:
            llama = self._llama
            self._llama = None
        if llama is None:
            return
        try:
            close = getattr(llama, "close", None)
            if callable(close):
                close()
        except Exception:  # noqa: BLE001 - freeing is best-effort
            logger.debug("Llama.close() raised during unload", exc_info=True)
        del llama
        # Run the destructor NOW (this thread, nothing else constructing)
        # instead of at an arbitrary allocation site later.
        gc.collect()

    @property
    def is_loaded(self) -> bool:
        return self._llama is not None

    @property
    def n_ctx(self) -> Optional[int]:
        """Configured context window (None when no model loaded)."""
        return self._n_ctx if self._llama is not None else None

    def count_tokens(self, text: str) -> int:
        """Tokenize *text* with the loaded model; falls back to chars/4."""
        if self._llama is None or not text:
            return max(1, len(text) // 4) if text else 0
        try:
            with self._lock:
                tokens = self._llama.tokenize(
                    text.encode("utf-8"), add_bos=False, special=False
                )
            return len(tokens)
        except Exception:  # noqa: BLE001 - estimate beats crashing the UI
            return max(1, len(text) // 4)

    @property
    def n_ctx_train(self) -> Optional[int]:
        """Tokens the model was trained for (None when unavailable).

        Running far beyond this degrades quality even though llama.cpp
        happily accepts a larger ``n_ctx``.
        """
        if self._llama is None:
            return None
        try:
            return int(self._llama.n_ctx_train())
        except Exception:  # noqa: BLE001 - depends on llama-cpp version
            return None

    @property
    def gpu_status(self) -> Dict[str, Any]:
        """Honest GPU offload report for the loaded model.

        ``build_supports_gpu`` reflects the installed wheel's compiled
        backends; ``requested`` is the user's toggle. When the two
        disagree, the model IS running on CPU no matter what the UI
        promised - exactly what users need to be told.
        """
        build_supports_gpu = False
        try:
            from llama_cpp import llama_cpp as _lc

            probe = getattr(_lc, "llama_supports_gpu_offload", None)
            build_supports_gpu = bool(probe()) if callable(probe) else False
        except Exception:  # noqa: BLE001 - probe is best-effort
            pass
        requested = bool(self.use_gpu and self.n_gpu_layers_requested != 0)
        if requested and not build_supports_gpu:
            state = "cpu_fallback"
            reason = "this Python environment has a CPU-only llama.cpp build"
        elif requested:
            state = "gpu"
            reason = ""
        else:
            state = "cpu_by_choice"
            reason = "GPU Acceleration is toggled off"
        return {
            "state": state,
            "reason": reason,
            "requested": requested,
            "build_supports_gpu": build_supports_gpu,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def chat_stream(self, messages: List[Dict[str, str]], **kwargs: Any) -> Iterator[str]:
        """Stream assistant content for a chat *messages* list.

        Uses ``llama.create_chat_completion`` with the model's embedded
        chat template (Llama-3, Qwen, Gemma, etc.).
        """
        kwargs = dict(kwargs)
        kwargs["stream"] = True
        kwargs["messages"] = messages
        stops = list(kwargs.get("stop") or [])
        with self._lock:
            llama = self._require_llama()
            stream = llama.create_chat_completion(**kwargs)

            def _deltas() -> Iterator[str]:
                for chunk in stream:
                    choices = chunk.get("choices") or [{}]
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content")
                    if text:
                        yield text

            for token in holdback_stream(_deltas(), stops):
                yield token

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Non-streaming :meth:`chat_stream`; returns the full reply."""
        parts: List[str] = []
        for piece in self.chat_stream(messages, **kwargs):
            parts.append(piece)
        return "".join(parts)

    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream raw text tokens for *prompt*.

        ``kwargs`` are passed straight through to llama_cpp (max_tokens,
        temperature, top_p, repeat_penalty, top_k, stop, ...).
        """
        kwargs = dict(kwargs)
        kwargs["stream"] = True
        for token_data in self._stream(prompt, kwargs):
            yield token_data.get("choices", [{}])[0].get("text", "")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a complete (non-streamed) text response."""
        with self._lock:
            llama = self._require_llama()
            response = llama(prompt, **kwargs)
        return response["choices"][0]["text"]

    def __call__(self, prompt: str, **kwargs: Any) -> Any:
        """Llama-compatible callable used by the UI and addons.

        ``stream=True`` returns a generator of token dicts (same shape as
        llama_cpp); otherwise a full response dict is returned. The lock
        is held for the lifetime of a stream so two threads never call
        the runtime concurrently.
        """
        if kwargs.get("stream"):
            return self._stream(prompt, kwargs)
        with self._lock:
            llama = self._require_llama()
            return llama(prompt, **kwargs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_llama(self) -> Any:
        if not LLAMA_AVAILABLE:
            raise RuntimeError("llama-cpp-python is not installed")
        if self._llama is None:
            raise RuntimeError("No model is loaded")
        return self._llama

    def _stream(self, prompt: str, kwargs: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Yield raw token dicts while holding the lock for the stream lifetime."""
        with self._lock:
            llama = self._require_llama()
            stream = llama(prompt, **kwargs)
            for token_data in stream:
                yield token_data
