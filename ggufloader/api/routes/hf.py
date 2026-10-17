"""HuggingFace model browser routes.

Exposes the existing HFService (search / download with resume + SHA256
verification) to the React frontend so users can discover and download GGUF
models without leaving the app:

- ``GET  /api/hf/search``        — search GGUF repos (empty query = trending)
- ``GET  /api/hf/file-info``     — size/etag for one repo file
- ``POST /api/hf/download``      — start a background download
- ``GET  /api/hf/download/status`` — live progress (bytes, speed, ETA)
- ``POST /api/hf/download/cancel`` — cooperative cancel
- ``GET  /api/hf/local``         — GGUF files already in the models folder

Downloads land in the default models folder so the auto-loader and every
model picker (Playground, Testing Studio, Agent) find them immediately.
Only one download runs at a time; a second request while one is active is
rejected with 409.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ggufloader.config import get_paths
from ggufloader.services.hf_service import HFService

router = APIRouter()
logger = logging.getLogger(__name__)


def _models_dir() -> Path:
    paths = get_paths()
    return Path(paths.get("models") or (Path.home() / ".ggufloader" / "models"))


def _service() -> HFService:
    return HFService(_models_dir())


# ---------------------------------------------------------------------------
# Background download state (one at a time)
# ---------------------------------------------------------------------------

_dl_lock = threading.Lock()
_dl_state: Dict[str, Any] = {
    "status": "idle",        # idle|downloading|verifying|done|error|cancelled
    "repo_id": None,
    "file_path": None,
    "bytes_downloaded": 0,
    "total_bytes": 0,
    "speed_bps": 0.0,
    "elapsed_secs": 0.0,
    "path": None,
    "error": None,
}
_dl_thread: Optional[threading.Thread] = None
_dl_cancel = threading.Event()


class DownloadRequest(BaseModel):
    repo_id: str
    file_path: str
    expected_hash: Optional[str] = None


def _download_status_snapshot() -> dict:
    with _dl_lock:
        return dict(_dl_state)


def _set_dl(**kw) -> None:
    with _dl_lock:
        _dl_state.update(kw)


def _worker_download(repo_id: str, file_path: str, expected_hash: Optional[str]) -> None:
    try:
        service = _service()

        def on_progress(p) -> None:
            _set_dl(
                status=p.status or "downloading",
                bytes_downloaded=p.bytes_downloaded,
                total_bytes=p.total_bytes,
                speed_bps=p.speed_bps,
                elapsed_secs=p.elapsed_secs,
            )

        def should_cancel() -> bool:
            return _dl_cancel.is_set()

        dest = service.download(
            repo_id, file_path,
            expected_hash=expected_hash,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
        _set_dl(status="done", path=str(dest), error=None)
        logger.info("HF download complete: %s", dest)
        # If nothing is loaded yet, make the new model ready immediately
        # (auto-load prefers the starter model, else the newest GGUF —
        # which is now the file we just wrote). No-op when a model is
        # already loaded or under pytest.
        try:
            from ggufloader.api.auto_load import start_auto_load
            start_auto_load(force=False)
        except Exception:  # noqa: BLE001 - never mask a good download
            logger.exception("Auto-load after HF download failed")
    except RuntimeError as e:
        if _dl_cancel.is_set():
            _set_dl(status="cancelled", error=str(e))
        else:
            _set_dl(status="error", error=str(e))
    except Exception as e:  # noqa: BLE001 - surface, never crash the thread
        logger.exception("HF download worker failed")
        _set_dl(status="error", error=str(e))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/search")
async def hf_search(q: str = "", sort: str = "likes", limit: int = 20) -> dict:
    """Search HuggingFace for GGUF repos. Empty q returns trending repos."""
    if sort not in ("likes", "downloads", "lastModified"):
        sort = "likes"
    limit = max(1, min(limit, 50))
    service = _service()
    models = service.search(q.strip(), sort=sort, limit=limit)
    return {"models": [m.to_dict() for m in models], "query": q, "sort": sort}


@router.get("/file-info")
async def hf_file_info(repo_id: str, file_path: str) -> dict:
    """Size / ETag / direct URL for one repo file (HEAD request)."""
    if not repo_id or not file_path:
        raise HTTPException(status_code=400, detail="repo_id and file_path are required")
    service = _service()
    info = service.get_file_info(repo_id, file_path)
    if info.get("size", 0) == 0:
        raise HTTPException(status_code=404, detail=f"File not found on HuggingFace: {repo_id}/{file_path}")
    return info


class FilesInfoRequest(BaseModel):
    repo_id: str
    file_paths: list[str]


@router.post("/files-info")
async def hf_files_info(req: FilesInfoRequest) -> dict:
    """Batch size lookup for a repo's GGUF files (concurrent HEADs).

    The search API no longer includes per-file sizes, so the UI calls this
    once when a repo card is expanded. Missing files come back with size 0
    instead of failing the whole batch.
    """
    if not req.repo_id or not req.file_paths:
        raise HTTPException(status_code=400, detail="repo_id and file_paths are required")
    service = _service()

    results: Dict[str, dict] = {}

    def _head(fp: str) -> None:
        try:
            results[fp] = service.get_file_info(req.repo_id, fp)
        except Exception as e:  # noqa: BLE001 - one bad file must not fail the batch
            logger.debug("files-info failed for %s/%s: %s", req.repo_id, fp, e)
            results[fp] = {"size": 0, "etag": "", "url": ""}

    threads = [
        threading.Thread(target=_head, args=(fp,), daemon=True)
        for fp in req.file_paths[:30]  # cap: giant repos stay fast
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    return {"repo_id": req.repo_id, "files": results}


@router.post("/download")
async def hf_download(req: DownloadRequest) -> dict:
    """Start a background download into the models folder (resume-capable)."""
    global _dl_thread
    if not req.repo_id or not req.file_path:
        raise HTTPException(status_code=400, detail="repo_id and file_path are required")
    if not req.file_path.endswith(".gguf"):
        raise HTTPException(status_code=400, detail="Only .gguf files can be downloaded")

    with _dl_lock:
        if _dl_thread is not None and _dl_thread.is_alive():
            raise HTTPException(
                status_code=409,
                detail="A download is already running — cancel it first.",
            )
        _dl_cancel.clear()
        _dl_state.update({
            "status": "downloading",
            "repo_id": req.repo_id,
            "file_path": req.file_path,
            "bytes_downloaded": 0,
            "total_bytes": 0,
            "speed_bps": 0.0,
            "elapsed_secs": 0.0,
            "path": str(_models_dir() / req.file_path),
            "error": None,
        })
        _dl_thread = threading.Thread(
            target=_worker_download,
            args=(req.repo_id, req.file_path, req.expected_hash),
            daemon=True,
            name="hf-download",
        )
        _dl_thread.start()
    return {"status": "started", "repo_id": req.repo_id, "file_path": req.file_path}


@router.get("/download/status")
async def hf_download_status() -> dict:
    """Live download progress snapshot (poll ~1 Hz from the UI)."""
    snap = _download_status_snapshot()
    snap["models_dir"] = str(_models_dir())
    return snap


@router.post("/download/cancel")
async def hf_download_cancel() -> dict:
    """Cooperatively cancel the running download (partial file is kept)."""
    with _dl_lock:
        running = _dl_thread is not None and _dl_thread.is_alive()
    if not running:
        return {"status": "idle", "message": "No download is running."}
    _dl_cancel.set()
    return {"status": "cancelling"}


@router.get("/local")
async def hf_local() -> dict:
    """GGUF files already present in the models folder."""
    service = _service()
    files = []
    for p in service.list_local_models():
        try:
            st = p.stat()
        except OSError:
            continue
        files.append({
            "path": str(p),
            "filename": p.name,
            "size_bytes": st.st_size,
            "size_gb": round(st.st_size / (1024 ** 3), 2),
        })
    return {"models": files, "models_dir": str(_models_dir())}
