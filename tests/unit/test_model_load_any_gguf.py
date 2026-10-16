"""Regression tests: any GGUF loads, not just the pinned Gemma 4 12B Q4_K_M.

``POST /api/model/load`` used to hard-reject every other architecture /
quantization with HTTP 400 ("Unsupported model: architecture 'lfm2'"), and
the UI swallowed that error - so the models actually used for testing
(LFM2.5, Qwen3.5, Ornith, ...) appeared to "not load" with no explanation.

The pinned Gemma 4 12B Q4_K_M is still the tuned default (it is what the
advisory ``compatible`` verdict in /api/model/info reports) but it is no
longer a gate on loading.

Headless: ``ModelRouter.load`` is stubbed, so no llama.cpp, GPU or
multi-GB model is needed.
"""

from __future__ import annotations

import pytest

import ggufloader.api.deps as deps
import ggufloader.api.routes.model as model_routes
from ggufloader.core.router import ModelRole, ModelRouter, SystemProfile

from tests.unit.test_router import _create_model_file, _kv_str, _kv_u32

# The models from the bug report, plus the pinned target for contrast.
# (filename, gguf architecture, block_count, is_pinned_target)
USER_MODELS = [
    ("gemma4-12b-q4_k_m.gguf", "gemma4", 48, True),
    ("lfm2.5-8b-q4_k_m.gguf", "lfm2", 32, False),
    ("ornith-9b-q4_k_m.gguf", "ornith", 40, False),
    ("qwen3.5-2b-q8_0.gguf", "qwen3", 28, False),
    ("qwen3.5-9.7b-q4_k_m.gguf", "qwen3", 36, False),
]


class StubBackend:
    """Stands in for ModelBackend - /model/info only reads attributes."""

    def __init__(self, path: str, profile) -> None:
        self.model_path = path
        self.architecture = profile.architecture
        self.quantization = profile.quantization
        self.parameters = profile.param_count_estimate
        self.n_ctx = 4096
        self.use_gpu = False

    def unload(self) -> None:
        pass


@pytest.fixture
def models_dir(tmp_path):
    """The reported models, written as small (metadata-only) GGUF files."""
    paths = {}
    for name, arch, layers, _pinned in USER_MODELS:
        kvs = _kv_str("general.architecture", arch) + _kv_u32(f"{arch}.block_count", layers)
        paths[name] = _create_model_file(tmp_path, name, kvs, kv_count=2)
    return paths


@pytest.fixture
def client(monkeypatch, models_dir):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    router = ModelRouter(system=SystemProfile(ram_gb=16.0, vram_gb=0.0, has_gpu_support=False))
    loaded: list[str] = []
    remembered: list[str] = []

    def fake_load(self, path, role=ModelRole.CHAT, **kwargs):
        profile = self.inspect(path)
        strategy = self.plan(profile)
        loaded.append(path)
        return StubBackend(path, profile), profile, strategy, self.route(profile, role)

    monkeypatch.setattr(ModelRouter, "load", fake_load)
    # Never touch the real (user-level) config dir from a test.
    monkeypatch.setattr(
        model_routes, "remember_model_dir", lambda d: (remembered.append(d), d)[1])
    monkeypatch.setattr(deps, "_router", router)

    app = FastAPI()
    app.include_router(model_routes.router, prefix="/api/model")
    with TestClient(app) as c:
        c.loaded = loaded
        c.remembered = remembered
        yield c
    deps.set_model_backend(None)


# ── The reported bug ─────────────────────────────────────────────────────


@pytest.mark.parametrize("name,arch,layers,pinned", USER_MODELS)
def test_every_reported_model_loads(client, models_dir, name, arch, layers, pinned):
    """None of these may be rejected any more - that was the bug."""
    res = client.post("/api/model/load", json={"path": str(models_dir[name])})
    assert res.status_code == 200, res.text

    body = res.json()
    assert body["status"] == "loaded"
    assert body["filename"] == name
    assert body["strategy"]["n_ctx"] > 0      # router planned a real context
    assert str(models_dir[name]) in client.loaded
    # The folder is remembered so the next startup scans it too.
    assert client.remembered[-1] == str(models_dir[name].parent)


def test_non_pinned_model_is_advisory_not_fatal(client, models_dir):
    """It loads, and /model/info reports it as not-the-tuned-target."""
    res = client.post("/api/model/load", json={"path": str(models_dir["lfm2.5-8b-q4_k_m.gguf"])})
    assert res.status_code == 200, res.text

    info = client.get("/api/model/info").json()
    assert info["loaded"] is True
    assert info["architecture"] == "lfm2"
    assert info["compatible"] is False
    assert "not the tuned target" in (info["compatible_error"] or "")


def test_pinned_model_is_reported_compatible(client, models_dir):
    res = client.post("/api/model/load", json={"path": str(models_dir["gemma4-12b-q4_k_m.gguf"])})
    assert res.status_code == 200, res.text

    info = client.get("/api/model/info").json()
    assert info["compatible"] is True
    assert info["compatible_error"] is None


def test_missing_file_is_still_a_404(client):
    res = client.post("/api/model/load", json={"path": "/nope/missing.gguf"})
    assert res.status_code == 404


# ── Startup auto-load ────────────────────────────────────────────────────


def test_auto_load_prefers_pinned_then_falls_back_to_any_gguf(tmp_path, monkeypatch):
    """With only non-pinned GGUFs on disk, startup must still load one."""
    import ggufloader.api.auto_load as auto_load

    def write(name: str, arch: str):
        kvs = _kv_str("general.architecture", arch) + _kv_u32(f"{arch}.block_count", 32)
        return _create_model_file(tmp_path, name, kvs, kv_count=2)

    qwen = write("qwen3.5-9.7b-q4_k_m.gguf", "qwen3")

    monkeypatch.setattr(auto_load, "get_router", lambda: ModelRouter(
        system=SystemProfile(ram_gb=16.0, vram_gb=0.0, has_gpu_support=False)))

    path, profile = auto_load._find_model_file([str(tmp_path)])
    assert path == qwen
    assert profile.architecture == "qwen3"

    gemma = write("gemma4-12b-q4_k_m.gguf", "gemma4")
    path, profile = auto_load._find_model_file([str(tmp_path)])
    assert path == gemma            # the tuned target wins when present
    assert profile.architecture == "gemma4"


def test_auto_load_finds_nothing_in_an_empty_dir(tmp_path, monkeypatch):
    import ggufloader.api.auto_load as auto_load

    monkeypatch.setattr(auto_load, "get_router", lambda: ModelRouter(
        system=SystemProfile(ram_gb=16.0, vram_gb=0.0, has_gpu_support=False)))
    path, profile = auto_load._find_model_file([str(tmp_path)])
    assert path is None and profile is None
