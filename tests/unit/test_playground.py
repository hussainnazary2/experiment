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

    def test_prompt_and_total_token_counts(self):
        """The readouts above the prompt box need in + out = total."""
        backend = FakeBackend(tokens=7)
        base = time.perf_counter()
        stats = playground._stats(
            backend, ["some generated text"], base - 1.0, base - 0.5, base - 0.1,
            prompt_tokens=11)
        assert stats["prompt_tokens"] == 11
        assert stats["tokens_generated"] == 7
        assert stats["total_tokens"] == 18


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

    def test_stream_reports_prompt_and_total_tokens(self, client, monkeypatch):
        """Every stats/done frame carries the prompt + total counts the UI
        renders above the prompt box."""
        backend = FakeBackend(pieces=("hey"), tokens=4)
        monkeypatch.setattr(playground, "get_model_backend", lambda: backend)
        monkeypatch.setattr(playground, "_STATS_INTERVAL_S", 0.0)

        r = client.post("/api/playground/chat/stream", json={"message": "hi"})
        events = parse_sse(r.text)
        frames = [d for e, d in events if e in ("stats", "done")]
        assert frames, "expected stats/done frames"
        for f in frames:
            assert f["prompt_tokens"] > 0
            assert f["tokens_generated"] == 4
            assert f["total_tokens"] == f["prompt_tokens"] + 4

    def test_prompt_tokens_count_system_and_history(self, client, monkeypatch):
        backend = FakeBackend(tokens=4)
        monkeypatch.setattr(playground, "get_model_backend", lambda: backend)
        honest = client.post(
            "/api/playground/chat/stream",
            json={"message": "hi"},
        )
        long = client.post(
            "/api/playground/chat/stream",
            json={
                "message": "hi",
                "system_prompt": "be brief",
                "history": [{"role": "user", "content": "first"}],
            },
        )
        short_prompt = [d for e, d in parse_sse(honest.text) if e == "done"][0]["prompt_tokens"]
        long_prompt = [d for e, d in parse_sse(long.text) if e == "done"][0]["prompt_tokens"]
        assert long_prompt > short_prompt


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
