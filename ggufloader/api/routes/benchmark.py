"""Benchmark routes - model performance benchmarking + the Model Testing Studio.

The Studio endpoints (``/suite/*``) run a *suite* of GGUF models back-to-back
on CPU, measure real tokens/sec + TTFT, and expose live progress for the UI.
They deliberately bypass the pinned-model load gate (see
``ggufloader/core/testing/runner.py``); the app's global chat backend is freed
before a run and reloaded afterwards so only one ``Llama`` exists at a time.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


# ============================================================================
# Existing benchmark report/history routes
# ============================================================================


@router.get("/prompts")
async def list_prompts() -> list:
    """List available benchmark prompts."""
    from ggufloader.core.agent.benchmark import ModelBenchmark
    bm = ModelBenchmark()
    return bm.list_prompts()


@router.get("/history")
async def benchmark_history(limit: int = 20) -> list:
    """Get benchmark history."""
    from ggufloader.core.agent.benchmark import ModelBenchmark
    bm = ModelBenchmark()
    return bm.get_history(limit)


@router.get("/reports")
async def list_reports() -> list:
    """List saved benchmark reports."""
    from ggufloader.core.agent.benchmark import ModelBenchmark
    bm = ModelBenchmark()
    return bm.list_reports()


@router.get("/reports/{report_id}")
async def get_report(report_id: str) -> dict:
    """Get a benchmark report."""
    from ggufloader.core.agent.benchmark import ModelBenchmark
    bm = ModelBenchmark()
    report = bm.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


# ============================================================================
# Model Testing Studio
# ============================================================================


class SuiteRunRequest(BaseModel):
    """Request body for starting a multi-model test run."""

    paths: List[str]
    device: str = "cpu"                     # "cpu" (default) or "gpu"
    prompt_ids: Optional[List[str]] = None  # subset of SUITE_PROMPTS; None = all
    max_tokens: Optional[int] = None        # override per-prompt cap
    n_ctx: Optional[int] = None             # None = router picks


class _RunState:
    """Module-level holder for the single active/last test run."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.runner = None            # SuiteRunner of the active/last run
        self.thread: Optional[threading.Thread] = None
        self.snapshot: Optional[Dict[str, Any]] = None

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


_state = _RunState()


def _free_global_backend() -> None:
    """Unload the pinned chat model so the suite owns the only Llama."""
    from ggufloader.api.deps import get_model_backend, set_model_backend

    backend = get_model_backend()
    if backend is None:
        return
    try:
        backend.unload()
    except Exception as e:  # noqa: BLE001 - freeing memory is best-effort
        logger.warning("Failed to unload chat backend before suite run: %s", e)
    set_model_backend(None)


def _reload_pinned_async() -> None:
    """Ask the startup auto-loader to reload the pinned model in the background."""
    try:
        from ggufloader.api.auto_load import start_auto_load

        start_auto_load(force=True)
    except Exception as e:  # noqa: BLE001 - reload is best-effort
        logger.warning("Failed to re-trigger auto-load after suite run: %s", e)


@router.get("/suite/prompts")
async def suite_prompts() -> list:
    """The fixed prompt workload used by every model in a test run."""
    from ggufloader.core.testing.runner import SUITE_PROMPTS

    return [p.to_dict() for p in SUITE_PROMPTS]


@router.get("/suite/models")
async def suite_models(directory: Optional[str] = None) -> dict:
    """Candidate GGUF models on disk (reuses the model catalog scan)."""
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


@router.post("/suite/run")
async def suite_run(req: SuiteRunRequest) -> dict:
    """Start a test run over the given models (background thread)."""
    from ggufloader.core.testing.runner import SuiteRunner, SuiteStore

    if not req.paths:
        raise HTTPException(status_code=400, detail="No models selected")

    device = "gpu" if str(req.device).lower() == "gpu" else "cpu"

    with _state.lock:
        if _state.is_running():
            raise HTTPException(status_code=409, detail="A test run is already in progress")

        _free_global_backend()

        runner = SuiteRunner()
        _state.runner = runner
        _state.snapshot = None

        def _worker() -> None:
            store = SuiteStore()
            try:
                run = runner.run(
                    req.paths,
                    device=device,
                    prompt_ids=req.prompt_ids,
                    max_tokens=req.max_tokens,
                    n_ctx=req.n_ctx,
                )
                store.save(run)
                with _state.lock:
                    _state.snapshot = run.to_dict()
                logger.info(
                    "Test run %s finished: %d model(s), %.1fs",
                    run.run_id, len(run.models), run.duration_ms / 1000,
                )
            except Exception as e:  # noqa: BLE001 - surface failures to the UI
                logger.error("Test run failed: %s", e)
                with _state.lock:
                    _state.snapshot = {
                        "run_id": None, "status": "error", "error": str(e),
                        "device": device, "models": [], "leaderboard": [],
                    }
            finally:
                _reload_pinned_async()

        thread = threading.Thread(target=_worker, daemon=True, name="model-suite-run")
        _state.thread = thread
        thread.start()

    return {"status": "running", "device": device, "model_count": len(req.paths)}


@router.get("/suite/status")
async def suite_status() -> dict:
    """Live status of the current run (or the last finished one)."""
    with _state.lock:
        runner = _state.runner
        if runner is not None and runner.last_run is not None and _state.is_running():
            return {"running": True, **runner.last_run.to_dict()}
        if _state.snapshot is not None:
            return {"running": _state.is_running(), **_state.snapshot}
        if runner is not None and runner.last_run is not None:
            return {"running": False, **runner.last_run.to_dict()}
    return {"running": False, "status": "idle", "models": [], "leaderboard": []}


@router.post("/suite/cancel")
async def suite_cancel() -> dict:
    """Cooperatively cancel the active run."""
    with _state.lock:
        runner = _state.runner
        if runner is None or not _state.is_running():
            return {"status": "idle", "message": "No run in progress"}
        runner.cancel()
    return {"status": "cancelling"}


@router.get("/suite/runs")
async def suite_runs(limit: int = 20) -> list:
    """List saved test runs (newest first)."""
    from ggufloader.core.testing.runner import SuiteStore

    return SuiteStore().list(limit)


@router.get("/suite/runs/{run_id}")
async def suite_run_detail(run_id: str) -> dict:
    """Fetch one saved test run."""
    from ggufloader.core.testing.runner import SuiteStore

    data = SuiteStore().get(run_id)
    if not data:
        raise HTTPException(status_code=404, detail="Run not found")
    return data


@router.get("/suite/runs/{run_id}/export")
async def suite_export(run_id: str, format: str = "json"):
    """Export a saved run as json | markdown | csv."""
    from ggufloader.core.testing.runner import SuiteStore, export_csv, export_markdown

    data = SuiteStore().get(run_id)
    if not data:
        raise HTTPException(status_code=404, detail="Run not found")

    fmt = (format or "json").lower()
    if fmt == "json":
        import json

        return PlainTextResponse(
            json.dumps(data, indent=2, ensure_ascii=False), media_type="application/json"
        )

    run = _run_from_dict(data)
    if fmt == "markdown":
        return PlainTextResponse(export_markdown(run), media_type="text/markdown")
    if fmt == "csv":
        return PlainTextResponse(export_csv(run), media_type="text/csv")
    raise HTTPException(status_code=400, detail=f"Unknown format: {format}")


@router.post("/run")
async def run_single(req: SuiteRunRequest) -> dict:
    """Single-model convenience run: starts a one-model suite.

    Kept for the existing ``benchmarkApi.run`` client call.
    """
    return await suite_run(req)


def _run_from_dict(data: Dict[str, Any]) -> Any:
    """Rebuild a SuiteRun from its serialized dict (for exports)."""
    from ggufloader.core.testing.runner import ModelResult, PromptResult, SuiteRun

    run = SuiteRun(
        run_id=data.get("run_id", ""),
        device=data.get("device", "cpu"),
        status=data.get("status", "done"),
        started_at=data.get("started_at", 0.0),
        finished_at=data.get("finished_at", 0.0),
        error=data.get("error"),
        progress=data.get("progress", 1.0),
    )
    for m in data.get("models", []):
        model = ModelResult(
            path=m.get("path", ""),
            name=m.get("name", ""),
            status=m.get("status", "pending"),
            error=m.get("error"),
            size_gb=m.get("size_gb", 0.0),
            architecture=m.get("architecture", ""),
            quantization=m.get("quantization", ""),
            parameters=m.get("parameters", ""),
            n_ctx=m.get("n_ctx"),
            device=m.get("device", "cpu"),
            load_ms=m.get("load_ms", 0.0),
        )
        for p in m.get("prompts", []):
            model.prompts.append(
                PromptResult(
                    prompt_id=p.get("prompt_id", ""),
                    label=p.get("label", ""),
                    tokens_generated=p.get("tokens_generated", 0),
                    ttft_ms=p.get("ttft_ms", 0.0),
                    gen_time_ms=p.get("gen_time_ms", 0.0),
                    tokens_per_second=p.get("tokens_per_second", 0.0),
                    text=p.get("text", ""),
                    error=p.get("error"),
                )
            )
        run.models.append(model)
    return run
