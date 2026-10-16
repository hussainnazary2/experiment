"""Model Testing Studio - run a suite of GGUF models back-to-back and measure speed.

This is the engine behind the "Testing" panel. For every selected GGUF it:

1. loads the model (CPU by default) and times the load,
2. runs a fixed prompt workload through it,
3. measures **real** time-to-first-token (TTFT) and decode throughput
   (tokens/sec) from llama.cpp's own stream,
4. unloads the model before moving to the next one,
5. records everything into a :class:`SuiteRun` report.

Design notes
------------
* The runner never touches the app's pinned chat model. The API layer frees
  the global backend before a run and re-triggers the startup auto-loader
  afterwards, so at most one ``Llama`` instance exists at any time (see
  AGENTS.md, invariant 1).
* All model I/O goes through injected callables (``load_fn`` / ``unload_fn`` /
  ``clock``), so the whole engine is unit-testable headless with a fake
  backend - no model, no GPU, no llama.cpp.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Fixed workload
# ============================================================================


@dataclass
class SuitePrompt:
    """One prompt in the fixed benchmark workload."""

    id: str
    label: str
    prompt: str
    max_tokens: int = 128

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
        }


# Kept intentionally small and fixed so every model does exactly the same
# work and the speed numbers are directly comparable.
SUITE_PROMPTS: List[SuitePrompt] = [
    SuitePrompt(
        id="short",
        label="Short answer",
        prompt="In one sentence, explain why the sky is blue.",
        max_tokens=64,
    ),
    SuitePrompt(
        id="medium",
        label="Medium answer",
        prompt="List five benefits of regular exercise. One short line each.",
        max_tokens=128,
    ),
    SuitePrompt(
        id="code",
        label="Code answer",
        prompt=(
            "Write a Python function that returns the nth Fibonacci number. "
            "Include a short docstring and one usage example."
        ),
        max_tokens=160,
    ),
]


# ============================================================================
# Result dataclasses
# ============================================================================


@dataclass
class PromptResult:
    """Measured performance of one prompt on one model."""

    prompt_id: str
    label: str
    tokens_generated: int
    ttft_ms: float
    gen_time_ms: float
    tokens_per_second: float
    text: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "label": self.label,
            "tokens_generated": self.tokens_generated,
            "ttft_ms": round(self.ttft_ms, 1),
            "gen_time_ms": round(self.gen_time_ms, 1),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "text": self.text,
            "error": self.error,
        }


@dataclass
class ModelResult:
    """Aggregate result for one model across the whole workload."""

    path: str
    name: str
    status: str = "pending"  # pending|loading|running|ok|error|cancelled
    error: Optional[str] = None
    size_gb: float = 0.0
    architecture: str = ""
    quantization: str = ""
    parameters: str = ""
    n_ctx: Optional[int] = None
    device: str = "cpu"
    load_ms: float = 0.0
    prompts: List[PromptResult] = field(default_factory=list)

    @property
    def avg_tps(self) -> float:
        vals = [p.tokens_per_second for p in self.prompts if p.error is None]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_ttft_ms(self) -> float:
        vals = [p.ttft_ms for p in self.prompts if p.error is None]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(p.tokens_generated for p in self.prompts)

    @property
    def gen_time_ms(self) -> float:
        return sum(p.gen_time_ms for p in self.prompts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "status": self.status,
            "error": self.error,
            "size_gb": self.size_gb,
            "architecture": self.architecture,
            "quantization": self.quantization,
            "parameters": self.parameters,
            "n_ctx": self.n_ctx,
            "device": self.device,
            "load_ms": round(self.load_ms, 1),
            "prompts": [p.to_dict() for p in self.prompts],
            "prompt_count": len(self.prompts),
            "avg_tps": round(self.avg_tps, 2),
            "avg_ttft_ms": round(self.avg_ttft_ms, 1),
            "total_tokens": self.total_tokens,
            "gen_time_ms": round(self.gen_time_ms, 1),
        }


@dataclass
class SuiteRun:
    """A complete multi-model test run."""

    run_id: str
    device: str = "cpu"
    status: str = "queued"  # queued|running|done|cancelled|error
    started_at: float = 0.0
    finished_at: float = 0.0
    error: Optional[str] = None
    current_model: str = ""
    current_prompt: str = ""
    progress: float = 0.0
    models: List[ModelResult] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or time.time()
        return (end - self.started_at) * 1000

    def leaderboard(self) -> List[Dict[str, Any]]:
        """Rank models that produced numbers, fastest first."""
        ranked = [m for m in self.models if m.status == "ok" and m.prompts]
        ranked.sort(key=lambda m: m.avg_tps, reverse=True)
        return [
            {"rank": i + 1, "name": m.name, "avg_tps": round(m.avg_tps, 2)}
            for i, m in enumerate(ranked)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "device": self.device,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": round(self.duration_ms, 1),
            "error": self.error,
            "current_model": self.current_model,
            "current_prompt": self.current_prompt,
            "progress": round(self.progress, 3),
            "models": [m.to_dict() for m in self.models],
            "leaderboard": self.leaderboard(),
        }


# ============================================================================
# Token / stream helpers
# ============================================================================


def _count_tokens(backend: Any, text: str) -> int:
    """Count tokens with the model's tokenizer, falling back to chars/4."""
    if not text:
        return 0
    fn = getattr(backend, "count_tokens", None)
    if callable(fn):
        try:
            return int(fn(text))
        except Exception:  # noqa: BLE001 - a bad tokenizer must not kill a run
            pass
    return max(1, len(text) // 4)


def _iter_text(backend: Any, prompt: str, max_tokens: int):
    """Yield plain text pieces from a backend stream (accepts token dicts)."""
    stream = backend.generate_stream(prompt, max_tokens=max_tokens, temperature=0.0)
    for piece in stream:
        if isinstance(piece, dict):
            choices = piece.get("choices") or [{}]
            piece = choices[0].get("text", "")
        if piece:
            yield piece


def _default_load(path: str, *, use_gpu: bool, n_ctx: Optional[int]) -> Tuple[Any, Any, Any]:
    """Default loader: router-driven load (inspect -> plan -> load ladder).

    Imported lazily so importing this module never drags in llama.cpp or the
    API layer.
    """
    from ggufloader.api.deps import get_router
    from ggufloader.core.router import ModelRole

    router = get_router()
    backend, profile, strategy, _config = router.load(
        path, ModelRole.CHAT, n_ctx=n_ctx, use_gpu=use_gpu
    )
    return backend, profile, strategy


def _default_unload(backend: Any) -> None:
    try:
        backend.unload()
    except Exception as e:  # noqa: BLE001 - unloading must never mask the next model
        logger.warning("Failed to unload benchmark model: %s", e)


# ============================================================================
# Runner
# ============================================================================


class SuiteRunner:
    """Loads each model in turn, benchmarks it, unloads it, and reports.

    All I/O is injected so the engine is fully headless-testable:

    * ``load_fn(path, *, use_gpu, n_ctx) -> (backend, profile, strategy)``
    * ``unload_fn(backend) -> None``
    * ``clock() -> float`` (monotonic seconds)
    """

    def __init__(
        self,
        load_fn: Optional[Callable[..., Tuple[Any, Any, Any]]] = None,
        unload_fn: Optional[Callable[[Any], None]] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._load_fn = load_fn or _default_load
        self._unload_fn = unload_fn or _default_unload
        self._clock = clock
        self._cancel = threading.Event()
        # The live run object, mutated in place, so an API layer can poll it.
        self.last_run: Optional[SuiteRun] = None

    def cancel(self) -> None:
        """Request cancellation; the run loop checks between pieces."""
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(
        self,
        paths: List[str],
        *,
        device: str = "cpu",
        prompt_ids: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        n_ctx: Optional[int] = None,
        run_id: Optional[str] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> SuiteRun:
        """Run the whole suite and return the completed :class:`SuiteRun`.

        ``device`` is ``"cpu"`` (use_gpu=False) or ``"gpu"`` (use_gpu=True).
        ``prompt_ids`` selects a subset of :data:`SUITE_PROMPTS`.
        ``max_tokens`` overrides each prompt's own cap when provided.
        """
        prompts = self._select_prompts(prompt_ids)
        use_gpu = device == "gpu"
        run = SuiteRun(
            run_id=run_id or uuid.uuid4().hex[:12],
            device=device,
            status="running",
            started_at=time.time(),
        )
        self.last_run = run
        for path in paths:
            run.models.append(
                ModelResult(
                    path=path,
                    name=Path(path).stem or path,
                    device=device,
                )
            )
        self._emit(on_event, {"type": "run_start", "run": run.to_dict()})

        total_units = max(1, len(run.models) * max(1, len(prompts)))
        done_units = 0

        for model in run.models:
            if self._cancel.is_set():
                model.status = "cancelled"
                continue

            run.current_model = model.name
            model.status = "loading"
            self._emit(on_event, {"type": "model_start", "model": model.to_dict()})

            backend: Any = None
            try:
                load_start = self._clock()
                backend, profile, strategy = self._load_fn(
                    model.path, use_gpu=use_gpu, n_ctx=n_ctx
                )
                model.load_ms = (self._clock() - load_start) * 1000
                self._apply_profile(model, profile, strategy)
                model.status = "running"
            except Exception as e:  # noqa: BLE001 - one bad model must not end the run
                model.status = "error"
                model.error = str(e)
                logger.warning("Benchmark load failed for %s: %s", model.path, e)
                done_units += len(prompts)
                run.progress = done_units / total_units
                self._emit(on_event, {"type": "model_done", "model": model.to_dict()})
                continue

            try:
                for prompt in prompts:
                    if self._cancel.is_set():
                        model.status = "cancelled"
                        break
                    run.current_prompt = prompt.id
                    result = self._benchmark_prompt(backend, prompt, max_tokens)
                    model.prompts.append(result)
                    done_units += 1
                    run.progress = done_units / total_units
                    self._emit(
                        on_event,
                        {
                            "type": "prompt_done",
                            "model": model.to_dict(),
                            "prompt": result.to_dict(),
                        },
                    )
                if model.status == "running":
                    model.status = "ok"
            finally:
                self._unload_fn(backend)

            run.current_prompt = ""
            self._emit(on_event, {"type": "model_done", "model": model.to_dict()})

        run.current_model = ""
        run.finished_at = time.time()
        run.progress = 1.0
        if self._cancel.is_set():
            run.status = "cancelled"
        else:
            run.status = "done"
        self._emit(on_event, {"type": "run_done", "run": run.to_dict()})
        return run

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_prompts(self, prompt_ids: Optional[List[str]]) -> List[SuitePrompt]:
        if not prompt_ids:
            return list(SUITE_PROMPTS)
        wanted = set(prompt_ids)
        selected = [p for p in SUITE_PROMPTS if p.id in wanted]
        return selected or list(SUITE_PROMPTS)

    def _benchmark_prompt(
        self, backend: Any, prompt: SuitePrompt, max_tokens: Optional[int]
    ) -> PromptResult:
        cap = int(max_tokens) if max_tokens else prompt.max_tokens
        start = self._clock()
        first_at: Optional[float] = None
        last_at: Optional[float] = None
        pieces: List[str] = []
        error: Optional[str] = None
        try:
            for piece in _iter_text(backend, prompt.prompt, cap):
                now = self._clock()
                if first_at is None:
                    first_at = now
                last_at = now
                pieces.append(piece)
        except Exception as e:  # noqa: BLE001 - record and keep going
            error = str(e)
            logger.warning("Benchmark generation failed (%s): %s", prompt.id, e)

        text = "".join(pieces)
        tokens = _count_tokens(backend, text)

        if first_at is not None:
            ttft_ms = (first_at - start) * 1000
            gen_ms = max(0.0, ((last_at or first_at) - first_at) * 1000)
        else:
            ttft_ms = (self._clock() - start) * 1000
            gen_ms = 0.0
        tps = (tokens / (gen_ms / 1000.0)) if gen_ms > 0 else 0.0

        return PromptResult(
            prompt_id=prompt.id,
            label=prompt.label,
            tokens_generated=tokens,
            ttft_ms=ttft_ms,
            gen_time_ms=gen_ms,
            tokens_per_second=tps,
            text=text[:400],
            error=error,
        )

    @staticmethod
    def _apply_profile(model: ModelResult, profile: Any, strategy: Any) -> None:
        """Copy architecture/quant/ctx from the router profile onto the result."""
        if profile is not None:
            model.architecture = getattr(profile, "architecture", "") or ""
            model.quantization = getattr(profile, "quantization", "") or ""
            model.parameters = getattr(profile, "param_count_estimate", "") or ""
            size_gb = getattr(profile, "file_size_gb", 0.0) or 0.0
            model.size_gb = round(float(size_gb), 2)
        if strategy is not None:
            model.n_ctx = getattr(strategy, "n_ctx", None) or model.n_ctx

    @staticmethod
    def _emit(on_event: Optional[Callable[[Dict[str, Any]], None]], event: Dict[str, Any]) -> None:
        if on_event is None:
            return
        try:
            on_event(event)
        except Exception as e:  # noqa: BLE001 - a UI callback must not break a run
            logger.debug("Suite event callback failed: %s", e)


# ============================================================================
# Persistence
# ============================================================================


class SuiteStore:
    """Save, list, and load suite runs as JSON files."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else Path.home() / ".ggufloader" / "benchmarks" / "suites"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # noqa: BLE001 - read-only home must not crash the app
            logger.warning("Could not create suite store dir %s: %s", self.root, e)

    def save(self, run: SuiteRun) -> Path:
        path = self.root / f"{run.run_id}.json"
        path.write_text(
            json.dumps(run.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        path = self.root / f"{run_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 - corrupt file should not raise
            logger.warning("Failed to read suite run %s: %s", run_id, e)
            return None

    def list(self, limit: int = 20) -> List[Dict[str, Any]]:
        runs: List[Dict[str, Any]] = []
        try:
            files = sorted(self.root.glob("*.json"), reverse=True)
        except Exception:  # noqa: BLE001
            return runs
        for f in files[:limit]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            models = data.get("models", [])
            runs.append(
                {
                    "run_id": data.get("run_id", f.stem),
                    "status": data.get("status", ""),
                    "device": data.get("device", ""),
                    "started_at": data.get("started_at", 0),
                    "duration_ms": data.get("duration_ms", 0),
                    "model_count": len(models),
                    "best_tps": max((m.get("avg_tps", 0) for m in models), default=0),
                    "leaderboard": data.get("leaderboard", []),
                }
            )
        return runs


# ============================================================================
# Exports
# ============================================================================


def export_markdown(run: SuiteRun) -> str:
    """Render a run as a shareable Markdown report."""
    lines: List[str] = []
    lines.append(f"# Model speed test ({run.device.upper()})")
    lines.append("")
    lines.append(f"- Run: `{run.run_id}`")
    lines.append(f"- Status: {run.status}")
    lines.append(f"- Duration: {run.duration_ms / 1000:.1f}s")
    lines.append("")
    lines.append("## Leaderboard")
    lines.append("")
    lines.append("| Rank | Model | Tokens/sec | TTFT (ms) | Load (ms) |")
    lines.append("| ---: | :---- | ---------: | --------: | --------: |")
    ranked = sorted(
        [m for m in run.models if m.prompts],
        key=lambda m: m.avg_tps,
        reverse=True,
    )
    for i, m in enumerate(ranked):
        lines.append(
            f"| {i + 1} | {m.name} | {m.avg_tps:.2f} | {m.avg_ttft_ms:.0f} | {m.load_ms:.0f} |"
        )
    failed = [m for m in run.models if m.status == "error"]
    if failed:
        lines.append("")
        lines.append("## Failed models")
        lines.append("")
        for m in failed:
            lines.append(f"- {m.name}: {m.error}")
    lines.append("")
    return "\n".join(lines)


def export_csv(run: SuiteRun) -> str:
    """Render a run as CSV, one row per model/prompt measurement."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "run_id",
            "device",
            "model",
            "architecture",
            "quantization",
            "load_ms",
            "prompt_id",
            "tokens_generated",
            "ttft_ms",
            "gen_time_ms",
            "tokens_per_second",
            "status",
            "error",
        ]
    )
    for m in run.models:
        if m.prompts:
            for p in m.prompts:
                writer.writerow(
                    [
                        run.run_id,
                        run.device,
                        m.name,
                        m.architecture,
                        m.quantization,
                        f"{m.load_ms:.1f}",
                        p.prompt_id,
                        p.tokens_generated,
                        f"{p.ttft_ms:.1f}",
                        f"{p.gen_time_ms:.1f}",
                        f"{p.tokens_per_second:.2f}",
                        m.status,
                        m.error or p.error or "",
                    ]
                )
        else:
            writer.writerow(
                [
                    run.run_id,
                    run.device,
                    m.name,
                    m.architecture,
                    m.quantization,
                    f"{m.load_ms:.1f}",
                    "",
                    "",
                    "",
                    "",
                    "",
                    m.status,
                    m.error or "",
                ]
            )
    return buf.getvalue()
