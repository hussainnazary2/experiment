"""Regression: a reply must not stall after its first token.

Bug report: a model loads fine, the user sends a message, and the answer never
arrives - the Chatbot panel sits on "0 tok/s / 0 total tokens" with the Stop
button lit and only the first token ever reaches the browser.

Cause: ``ModelBackend.chat_stream`` holds ``self._lock`` for the whole stream
(correct - the runtime is single-model and must not be entered twice), while
the playground's live ``stats`` event calls ``backend.count_tokens()`` from the
very thread that is streaming. A non-reentrant ``threading.Lock`` therefore
blocked the thread that already owned it, permanently.

These tests drive a fake llama.cpp runtime, so they need no model, GPU or
``llama-cpp-python`` install.
"""

from __future__ import annotations

import threading

import pytest

import ggufloader.core.llm.model_backend as model_backend
from ggufloader.core.llm.model_backend import ModelBackend


class FakeLlama:
    """Minimal stand-in for the llama.cpp runtime of a loaded model."""

    def __init__(self, chunks=("Hel", "lo"), hold_after_first=None):
        self._chunks = chunks
        self._hold = hold_after_first
        self.tokenize_calls = 0

    def create_chat_completion(self, **kwargs):
        for index, piece in enumerate(self._chunks):
            if index == 1 and self._hold is not None:
                self._hold.wait(timeout=5)
            yield {"choices": [{"delta": {"content": piece}}]}

    def tokenize(self, text, **kwargs):
        self.tokenize_calls += 1
        return [0] * max(1, len(text) // 4)


@pytest.fixture(autouse=True)
def _pretend_runtime_is_installed(monkeypatch):
    """The lock behaviour under test is independent of the llama.cpp wheel."""
    monkeypatch.setattr(model_backend, "LLAMA_AVAILABLE", True)


def _backend(fake: FakeLlama) -> ModelBackend:
    backend = ModelBackend("fake.gguf")
    backend._llama = fake
    return backend


def test_counting_tokens_while_streaming_does_not_stall_the_stream():
    """The playground's stats path re-enters the lock from the streaming thread."""
    backend = _backend(FakeLlama())
    seen: list[str] = []

    def stream_and_count():
        for piece in backend.chat_stream(
            [{"role": "user", "content": "hi"}], max_tokens=8
        ):
            seen.append(piece)
            backend.count_tokens("".join(seen))  # what playground._stats() does
        seen.append("<done>")

    worker = threading.Thread(target=stream_and_count, daemon=True)
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive(), (
        "chat_stream stalled: the streaming thread deadlocked on its own lock "
        "while counting tokens (the reported 'answer stops there' bug)"
    )
    assert seen == ["Hel", "lo", "<done>"]


def test_reentrancy_does_not_allow_a_second_generation_on_one_thread():
    """Read-only re-entry is fine; a nested generation is not (it would
    interleave two decodes into one llama.cpp context)."""
    backend = _backend(FakeLlama())
    seen: list[str] = []

    def nested_generation():
        for piece in backend.chat_stream([{"role": "user", "content": "hi"}]):
            seen.append(piece)
            with pytest.raises(RuntimeError, match="Nested generation"):
                list(backend.chat_stream([{"role": "user", "content": "again"}]))

    nested_generation()
    assert seen == ["Hel", "lo"]


def test_other_threads_still_wait_for_the_running_stream():
    """Re-entrancy must not turn the single-model runtime into a free-for-all."""
    hold = threading.Event()
    backend = _backend(FakeLlama(hold_after_first=hold))
    first_token = threading.Event()
    stream_finished = threading.Event()
    other_thread_done = threading.Event()

    def stream_thread():
        for _piece in backend.chat_stream([{"role": "user", "content": "hi"}]):
            first_token.set()  # we are inside the locked stream
        stream_finished.set()

    def other_thread():
        first_token.wait(timeout=5)
        backend.count_tokens("let me in")
        other_thread_done.set()

    streamer = threading.Thread(target=stream_thread, daemon=True)
    outsider = threading.Thread(target=other_thread, daemon=True)
    streamer.start()
    outsider.start()

    assert first_token.wait(timeout=5)
    assert not other_thread_done.wait(0.3), (
        "another thread read the runtime while a stream was still running - "
        "two generations would interleave in one KV cache"
    )

    hold.set()
    streamer.join(timeout=5)
    assert stream_finished.is_set()
    assert other_thread_done.wait(2.0), (
        "the waiting thread must proceed once the stream has released the runtime"
    )
