"""Tests for the Playground chatbot routes (plain streaming chat + live stats).

Headless: a fake backend stands in for llama.cpp, so no model or GPU is needed.
"""

from __future__ import annotations

import json
import time

import pytest

import ggufloader.api.routes.playground as playground


# ── Test doubles ─────────────────────────────────────────────────────────


class FakeBackend:
    def __init__(self, pieces=("Hello ", "world"), tokens=7, fail=False):
        self.pieces = list(pieces)
        self._tokens = tokens
        self.fail = fail
        self.model_path = "/models/fake-model-Q4_K_M.gguf"
        self.architecture = "llama"
        self.quantization = "Q4_K_M"
        self.n_ctx = 2048
        self.use_gpu = False
        self.seen: list = []

    def chat_stream(self, messages=None, **kwargs):
        if self.fail:
            raise RuntimeError("boom")
        self.seen.append({"messages": messages, "kwargs": kwargs})
        for piece in self.pieces:
            yield piece

    def count_tokens(self, text: str) -> int:
        return self._tokens

    def unload(self) -> None:
        pass


def parse_sse(text: str):
    """Turn an SSE body into a list of (event, data) tuples."""
    out = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event:
            out.append((event, data))
    return out


@pytest.fixture
def client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(playground.router, prefix="/api/playground")
    with TestClient(app) as c:
        yield c


# ── Stats math ───────────────────────────────────────────────────────────


class TestStats:
    def test_decode_rate_and_ttft_are_exact(self):
        backend = FakeBackend(tokens=10)
        base = time.perf_counter()
        t0 = base - 1.0        # started 1s ago
        first = base - 0.5     # first token 0.5s in
        last = base - 0.1      # last token 0.4s later
        stats = playground._stats(backend, ["some generated text"], t0, first, last)

        assert stats["tokens_generated"] == 10
        assert stats["ttft_ms"] == 500.0
        assert stats["decode_ms"] == 400.0
        assert stats["tokens_per_second"] == 25.0   # 10 tokens / 0.4s
        assert stats["elapsed_ms"] == pytest.approx(1000.0, abs=50)
        assert stats["total_tokens_per_second"] == pytest.approx(10.0, abs=0.6)

    def test_no_tokens_yet_is_zero(self):
        backend = FakeBackend(tokens=0)
        base = time.perf_counter()
        stats = playground._stats(backend, [], base, None, None)
        assert stats["tokens_generated"] == 0
        assert stats["decode_ms"] == 0.0
        assert stats["tokens_per_second"] == 0.0


# ── Streaming chat ───────────────────────────────────────────────────────


class TestChatStream:
    def test_streams_tokens_and_final_stats(self, client, monkeypatch):
        backend = FakeBackend(pieces=("Hel", "lo ", "there"), tokens=5)
        monkeypatch.setattr(playground, "get_model_backend", lambda: backend)

        r = client.post("/api/playground/chat/stream", json={"message": "hi"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")

        events = parse_sse(r.text)
        kinds = [e for e, _ in events]
        tokens = [d["token"] for e, d in events if e == "token"]
        assert tokens == ["Hel", "lo ", "there"]

        done = [d for e, d in events if e == "done"]
        assert len(done) == 1
        assert done[0]["content"] == "Hello there"
        assert done[0]["tokens_generated"] == 5
        assert "tokens_per_second" in done[0]
        assert kinds[-1] == "done"

    def test_live_stats_events_are_emitted(self, client, monkeypatch):
        backend = FakeBackend(pieces=("a", "b", "c"), tokens=3)
        monkeypatch.setattr(playground, "get_model_backend", lambda: backend)
        monkeypatch.setattr(playground, "_STATS_INTERVAL_S", 0.0)  # emit every token

        r = client.post("/api/playground/chat/stream", json={"message": "hi"})
        events = parse_sse(r.text)
        stats = [d for e, d in events if e == "stats"]
        assert stats, "expected live stats events"
        assert all("tokens_per_second" in s and "tokens_generated" in s for s in stats)

    def test_history_and_system_prompt_are_forwarded(self, client, monkeypatch):
        backend = FakeBackend()
        monkeypatch.setattr(playground, "get_model_backend", lambda: backend)

        client.post(
            "/api/playground/chat/stream",
            json={
                "message": "third",
                "system_prompt": "be brief",
                "history": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "second"},
                ],
            },
        )
        sent = backend.seen[0]["messages"]
        assert sent[0] == {"role": "system", "content": "be brief"}
        assert sent[1]["content"] == "first"
        assert sent[2]["content"] == "second"
        assert sent[-1] == {"role": "user", "content": "third"}

    def test_no_model_loaded_returns_400(self, client, monkeypatch):
        monkeypatch.setattr(playground, "get_model_backend", lambda: None)
        r = client.post("/api/playground/chat/stream", json={"message": "hi"})
        assert r.status_code == 400

    def test_generation_error_is_reported_in_stream(self, client, monkeypatch):
        backend = FakeBackend(fail=True)
        monkeypatch.setattr(playground, "get_model_backend", lambda: backend)
        r = client.post("/api/playground/chat/stream", json={"message": "hi"})
        events = parse_sse(r.text)
        assert events[-1][0] == "error"
        assert "boom" in events[-1][1]["message"]

    def test_non_streaming_endpoint(self, client, monkeypatch):
        backend = FakeBackend(pieces=("all ", "at ", "once"), tokens=3)
        monkeypatch.setattr(playground, "get_model_backend", lambda: backend)
        r = client.post("/api/playground/chat", json={"message": "hi"})
        assert r.status_code == 200
        body = r.json()
        assert body["content"] == "all at once"
        assert body["tokens_generated"] == 3


# ── Model status / routes ────────────────────────────────────────────────


class TestModelStatus:
    def test_status_when_loaded(self, client, monkeypatch):
        monkeypatch.setattr(playground, "get_model_backend", lambda: FakeBackend())
        body = client.get("/api/playground/model").json()
        assert body["loaded"] is True
        assert body["filename"] == "fake-model-Q4_K_M.gguf"
        assert body["quantization"] == "Q4_K_M"

    def test_status_when_empty(self, client, monkeypatch):
        monkeypatch.setattr(playground, "get_model_backend", lambda: None)
        assert client.get("/api/playground/model").json() == {"loaded": False}

    def test_routes_registered(self):
        paths = {r.path for r in playground.router.routes if hasattr(r, "path")}
        assert {
            "/models",
            "/model",
            "/model/load",
            "/chat",
            "/chat/stream",
        } <= paths

    def test_router_mounted_in_app(self):
        from ggufloader.api.app import create_app

        app = create_app()
        paths = [str(getattr(r, "path", r)) for r in app.routes]
        assert any("/api/playground" in p for p in paths)


# ── mmproj companion files ───────────────────────────────────────────────


class TestMmprojCompanionFiles:
    """mmproj GGUFs are vision projectors for a parent model, not chat LLMs:
    they must not appear in the chat dropdown, and loading one directly must
    fail with a message that says WHY (not the generic fallback-ladder error
    that reads like a quantization/settings problem).
    """

    def test_is_standalone_chat_model(self):
        assert playground._is_standalone_chat_model("gemma4-12b-q4_k_m.gguf")
        assert playground._is_standalone_chat_model("Gemma4-12B-MMProj-F16.gguf") is False
        assert playground._is_standalone_chat_model("qwen3-vl-mmproj.gguf") is False

    def test_models_endpoint_hides_mmproj(self, client, monkeypatch):
        async def fake_catalog(directory=None, recursive=True):
            return {
                "directory": directory,
                "models": [
                    {"path": "/m/gemma4-12b-q4_k_m.gguf", "filename": "gemma4-12b-q4_k_m.gguf",
                     "size_human": "8.1 GB", "quantization": "Q4_K_M"},
                    {"path": "/m/gemma4-12b-mmproj-f16.gguf", "filename": "gemma4-12b-mmproj-f16.gguf",
                     "size_human": "167 MB", "quantization": "F16"},
                ],
            }

        monkeypatch.setattr(
            "ggufloader.api.routes.model.model_catalog", fake_catalog
        )
        body = client.get("/api/playground/models").json()
        names = [m["filename"] for m in body["models"]]
        assert names == ["gemma4-12b-q4_k_m.gguf"]
        assert body["count"] == 1

    def test_load_mmproj_returns_clear_400(self, client, tmp_path):
        mmproj = tmp_path / "gemma4-12b-mmproj-f16.gguf"
        mmproj.write_bytes(b"\x00" * 16)
        res = client.post("/api/playground/model/load", json={"path": str(mmproj)})
        assert res.status_code == 400
        assert "vision companion file" in res.json()["detail"]


# ── Device preference (CPU / GPU) ─────────────────────────────────────────


class TestDevicePreference:
    """An explicit CPU/GPU choice must be persisted so the startup
    auto-load honors it on the next run (user asked for CPU).
    """

    class _FakeStrategy:
        def __init__(self, use_gpu):
            self.n_ctx = 2048
            self.use_gpu = use_gpu

    class _FakeProfile:
        filename = "model-q4_k_m.gguf"
        architecture = "llama"
        quantization = "Q4_K_M"
        param_count_estimate = "7B"
        file_size_gb = 4.0

    class _FakeRouter:
        def __init__(self, use_gpu=None):
            self.seen_use_gpu = "unset"
            self._use_gpu = use_gpu

        def load(self, path, role, n_ctx=None, use_gpu=None):
            self.seen_use_gpu = use_gpu
            return object(), TestDevicePreference._FakeProfile(), \
                TestDevicePreference._FakeStrategy(use_gpu=use_gpu), {}

    def test_roundtrip(self, tmp_path, monkeypatch):
        import ggufloader.api.auto_load as auto_load

        monkeypatch.setattr(auto_load, "find_config_dir", lambda: str(tmp_path))
        assert auto_load.get_remembered_device_preference() is None  # auto
        auto_load.remember_device_preference(False)
        assert auto_load.get_remembered_device_preference() is False  # CPU
        auto_load.remember_device_preference(True)
        assert auto_load.get_remembered_device_preference() is True  # GPU

    def test_playground_load_persists_explicit_device(self, client, tmp_path,
                                                      monkeypatch):
        import ggufloader.api.auto_load as auto_load

        monkeypatch.setattr(auto_load, "find_config_dir", lambda: str(tmp_path))
        model = tmp_path / "model-q4_k_m.gguf"
        model.write_bytes(b"\x00" * 16)

        fake_router = self._FakeRouter()
        monkeypatch.setattr(playground, "get_router", lambda: fake_router)
        monkeypatch.setattr(playground, "get_model_backend", lambda: None)
        monkeypatch.setattr(playground, "set_model_backend", lambda b: None)

        res = client.post(
            "/api/playground/model/load",
            json={"path": str(model), "use_gpu": False},
        )
        assert res.status_code == 200
        assert res.json()["use_gpu"] is False
        # The explicit device reached the router load call
        assert fake_router.seen_use_gpu is False
        # CPU choice persisted for the next startup auto-load
        assert auto_load.get_remembered_device_preference() is False

    def test_auto_device_preference_not_overwritten_by_auto_load(
            self, client, tmp_path, monkeypatch):
        """A load without an explicit device (use_gpu=None) must NOT touch
        the saved preference - 'auto' never overwrites a deliberate choice.
        """
        import ggufloader.api.auto_load as auto_load

        monkeypatch.setattr(auto_load, "find_config_dir", lambda: str(tmp_path))
        auto_load.remember_device_preference(False)

        model = tmp_path / "model-q4_k_m.gguf"
        model.write_bytes(b"\x00" * 16)

        fake_router = self._FakeRouter()
        monkeypatch.setattr(playground, "get_router", lambda: fake_router)
        monkeypatch.setattr(playground, "get_model_backend", lambda: None)
        monkeypatch.setattr(playground, "set_model_backend", lambda b: None)

        res = client.post("/api/playground/model/load", json={"path": str(model)})
        assert res.status_code == 200
        assert auto_load.get_remembered_device_preference() is False  # untouched


# ── Family sampling defaults + stop tokens (LFM <s>-echo fix) ────────────


class TestFamilySamplingDefaults:
    """The playground used to forward ONLY temperature/max_tokens - no
    repeat_penalty, no stop strings - so a model that missed its EOS
    repeated garbage until max_tokens. It now applies the loaded model's
    family sampling (model_families.json) + unified stop tokens; an
    explicit request temperature still wins.
    """

    def _fake_backend(self):
        return FakeBackend()

    def _patch_resolve(self, monkeypatch, params=None, error=False):
        import ggufloader.core.llm.model_profiles as mp

        if error:
            def boom(path):
                raise RuntimeError("no metadata")
            monkeypatch.setattr(mp, "resolve_chat_config", boom)
        else:
            monkeypatch.setattr(
                mp, "resolve_chat_config",
                lambda path: {"params": dict(params or {})})

    def test_family_params_and_stops_reach_backend(self, client, monkeypatch):
        fake = self._fake_backend()
        monkeypatch.setattr(playground, "get_model_backend", lambda: fake)
        self._patch_resolve(monkeypatch, {
            "temperature": 0.2, "top_k": 80, "repeat_penalty": 1.05,
        })
        client.post("/api/playground/chat",
                    json={"message": "hi", "max_tokens": 64})
        kw = fake.seen[0]["kwargs"]
        assert kw["temperature"] == 0.2      # family default (request sent none)
        assert kw["top_k"] == 80
        assert kw["repeat_penalty"] == 1.05
        assert kw["max_tokens"] == 64        # request value
        assert "<|im_end|>" in kw["stop"]    # unified stop tokens
        assert "</s>" in kw["stop"]

    def test_explicit_request_temperature_wins(self, client, monkeypatch):
        fake = self._fake_backend()
        monkeypatch.setattr(playground, "get_model_backend", lambda: fake)
        self._patch_resolve(monkeypatch, {"temperature": 0.2})
        client.post("/api/playground/chat",
                    json={"message": "hi", "temperature": 0.9})
        assert fake.seen[0]["kwargs"]["temperature"] == 0.9

    def test_family_resolution_failure_still_sends_stops(self, client, monkeypatch):
        fake = self._fake_backend()
        monkeypatch.setattr(playground, "get_model_backend", lambda: fake)
        self._patch_resolve(monkeypatch, error=True)
        client.post("/api/playground/chat", json={"message": "hi"})
        kw = fake.seen[0]["kwargs"]
        assert "<|im_end|>" in kw["stop"]    # graceful degradation
        assert kw["max_tokens"] == 512

    def test_stream_endpoint_applies_defaults_too(self, client, monkeypatch):
        fake = self._fake_backend()
        monkeypatch.setattr(playground, "get_model_backend", lambda: fake)
        self._patch_resolve(monkeypatch, {"temperature": 0.2, "repeat_penalty": 1.05})
        body = client.post(
            "/api/playground/chat/stream",
            json={"message": "hi", "max_tokens": 32},
        ).text
        assert "event: token" in body
        kw = fake.seen[0]["kwargs"]
        assert kw["repeat_penalty"] == 1.05
        assert "<|im_end|>" in kw["stop"]
