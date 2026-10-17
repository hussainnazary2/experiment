"""GPU routes - install, status check, and live hardware telemetry."""

from __future__ import annotations

import logging
import subprocess
import time

from fastapi import APIRouter

from ggufloader.api.deps import refresh_router_system
from ggufloader.core.system_probe import llama_cpp_available, llama_supports_gpu_offload

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/status")
async def gpu_status() -> dict:
    """Check GPU support status."""
    if not llama_cpp_available():
        return {"gpu_available": False, "status": "not_installed"}
    has_gpu = llama_supports_gpu_offload()
    return {
        "gpu_available": has_gpu,
        "status": "installed" if has_gpu else "cpu_only",
    }


@router.post("/install")
async def install_gpu() -> dict:
    """Install GPU support (triggers background installation)."""
    # This would trigger the GPU install service
    # For now, return a placeholder
    # A re-detection lets the router's SystemProfile see the new GPU.
    refresh_router_system()
    return {"status": "installation_started", "message": "GPU installation initiated"}


# ---------------------------------------------------------------------------
# Live hardware telemetry
# ---------------------------------------------------------------------------

# nvidia-smi takes ~30-100 ms per call; cache briefly so several UI widgets
# polling at 1 Hz don't spawn duplicate subprocesses.
_SMI_TTL = 0.5
_smi_cache: dict = {"ts": 0.0, "data": None}


def _query_nvidia_smi() -> dict | None:
    """One nvidia-smi snapshot: utilization, memory, name. None without NVIDIA."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        parts = [p.strip() for p in result.stdout.strip().split("\n")[0].split(",")]
        if len(parts) < 4:
            return None
        return {
            "name": parts[0],
            "utilization_pct": float(parts[1]),
            "memory_used_mb": float(parts[2]),
            "memory_total_mb": float(parts[3]),
            "temperature_c": float(parts[4]) if len(parts) > 4 and parts[4] else None,
        }
    except Exception:  # noqa: BLE001 - no NVIDIA / timeout / parse issues
        return None


def _gpu_snapshot() -> dict | None:
    now = time.monotonic()
    if _smi_cache["data"] is not None and now - _smi_cache["ts"] < _SMI_TTL:
        return _smi_cache["data"]
    data = _query_nvidia_smi()
    _smi_cache["ts"] = now
    _smi_cache["data"] = data
    return data


def _ram_gb(value_bytes: float) -> float:
    return round(value_bytes / (1024 ** 3), 2)


@router.get("/live")
async def gpu_live() -> dict:
    """Live hardware telemetry for the HUD / recordings (~poll 1 Hz).

    Returns RAM usage, CPU load, this backend process's RSS (a good proxy
    for how much memory the loaded model occupies on CPU), and — when an
    NVIDIA GPU is present — utilization, VRAM and temperature from
    nvidia-smi. Everything degrades gracefully: fields are null (never an
    error) when a source is unavailable.
    """
    payload: dict = {
        "ram_used_gb": None,
        "ram_total_gb": None,
        "ram_pct": None,
        "cpu_pct": None,
        "process_rss_gb": None,
        "gpu": None,
        "llama_gpu_available": False,
    }

    try:
        import psutil
        vm = psutil.virtual_memory()
        payload["ram_used_gb"] = _ram_gb(vm.total - vm.available)
        payload["ram_total_gb"] = _ram_gb(vm.total)
        payload["ram_pct"] = round(vm.percent, 1)
        try:
            payload["cpu_pct"] = round(psutil.cpu_percent(interval=None), 1)
        except Exception:  # noqa: BLE001
            pass
        try:
            proc = psutil.Process()
            payload["process_rss_gb"] = _ram_gb(proc.memory_info().rss)
        except Exception:  # noqa: BLE001
            pass
    except ImportError:
        logger.debug("psutil unavailable - RAM/CPU telemetry disabled")

    try:
        payload["llama_gpu_available"] = bool(
            llama_cpp_available() and llama_supports_gpu_offload()
        )
    except Exception:  # noqa: BLE001
        pass

    gpu = _gpu_snapshot()
    if gpu is not None:
        payload["gpu"] = gpu

    return payload
