"""Tests for the Model Testing Studio (multi-model CPU speed benchmarking).

Fully headless: a fake backend + injected clock stand in for llama.cpp, so
no model, GPU, or real inference is required.
"""

from __future__ import annotations

import types

import pytest

from ggufloader.core.testing.runner import (
    SUITE_PROMPTS,
    ModelResult,
    PromptResult,
    SuitePrompt,
    SuiteRun,
    SuiteRunner,
    SuiteStore,
    export_csv,
    export_markdown,
)


# ── Test doubles ─────────────────────────────────────────────────────────


class FakeBackend:
    """Minimal backend: yields a fixed number of text pieces."""

    def __init__(self, pieces=("Hello ", "world"), tokens=6, fail_load=False):
        self.pieces = list(pieces)
        self._tokens = tokens
        self.fail_load = fail_load
        self.unloaded = False
        self.calls = []

    def generate_stream(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        for piece in self.pieces:
            yield piece

    def count_tokens(self, text: str) -> int:
        return self._tokens

    def unload(self) -> None:
        self.unloaded = True


class StepClock:
    """Monotonic clock that advances a fixed delta on every call."""

    def __init__(self, step: float = 0.01):
        self.step = step
        self.t = 0.0
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        value = self.t
        self.t += self.step
        return value


def make_loader(backends):
    """Build a load_fn that returns backends keyed by path."""
    loaded = []

    def load_fn(path, *, use_gpu=False, n_ctx=None):
        backend = backends[path]
        if backend.fail_load:
            raise RuntimeError(f"cannot load {path}")
        loaded.append(path)
        profile = types.SimpleNamespace(
            architecture="llama", quantization="Q4_K_M",
            param_count_estimate="7B", file_size_gb=4.2,
        )
        strategy = types.SimpleNamespace(n_ctx=2048, use_gpu=use_gpu)
        return backend, profile, strategy

    load_fn.loaded = loaded
    return load_fn


# ── Measurement math ─────────────────────────────────────────────────────


class TestMeasurement:
    def test_ttft_and_tps_are_measured(self):
        backend = FakeBackend(pieces=("Hello ", "world"), tokens=6)
        load_fn = make_loader({"/m/a.gguf": backend})
        clock = StepClock(step=0.01)

        runner = SuiteRunner(load_fn=load_fn, unload_fn=lambda b: b.unload(), clock=clock)
        run = runner.run(["/m/a.gguf"], prompt_ids=["short"])

        assert run.status == "done"
        model = run.models[0]
        assert model.status == "ok"
        assert len(model.prompts) == 1
        result = model.prompts[0]
        # 2 pieces -> ttft = one step (10ms), decode = one step (10ms).
        assert result.tokens_generated == 6
        assert result.ttft_ms == pytest.approx(10.0, rel=1e-6)
        assert result.gen_time_ms == pytest.approx(10.0, rel=1e-6)
        assert result.tokens_per_second == pytest.approx(600.0, rel=1e-3)

    def test_max_tokens_override_is_passed_through(self):
        backend = FakeBackend()
        load_fn = make_loader({"/m/a.gguf": backend})
        runner = SuiteRunner(load_fn=load_fn, clock=StepClock())
        runner.run(["/m/a.gguf"], prompt_ids=["short"], max_tokens=32)

        assert backend.calls[0]["kwargs"]["max_tokens"] == 32
        assert backend.calls[0]["kwargs"]["temperature"] == 0.0

    def test_prompt_cap_used_when_no_override(self):
        backend = FakeBackend()
        load_fn = make_loader({"/m/a.gguf": backend})
        runner = SuiteRunner(load_fn=load_fn, clock=StepClock())
        runner.run(["/m/a.gguf"], prompt_ids=["short"])

        expected = next(p for p in SUITE_PROMPTS if p.id == "short").max_tokens
        assert backend.calls[0]["kwargs"]["max_tokens"] == expected

    def test_empty_stream_gives_zero_tps(self):
        backend = FakeBackend(pieces=(), tokens=0)
        load_fn = make_loader({"/m/a.gguf": backend})
        runner = SuiteRunner(load_fn=load_fn, clock=StepClock())
        run = runner.run(["/m/a.gguf"], prompt_ids=["short"])
        result = run.models[0].prompts[0]
        assert result.tokens_generated == 0
        assert result.tokens_per_second == 0.0


# ── Run behaviour ────────────────────────────────────────────────────────


class TestRunBehaviour:
    def test_models_run_in_order_and_all_prompts_execute(self):
        a, b = FakeBackend(), FakeBackend()
        load_fn = make_loader({"/m/a.gguf": a, "/m/b.gguf": b})

        runner = SuiteRunner(load_fn=load_fn, unload_fn=lambda x: x.unload(), clock=StepClock())
        run = runner.run(["/m/a.gguf", "/m/b.gguf"])

        assert load_fn.loaded == ["/m/a.gguf", "/m/b.gguf"]
        assert [m.name for m in run.models] == ["a", "b"]
        assert all(len(m.prompts) == len(SUITE_PROMPTS) for m in run.models)
        assert a.unloaded and b.unloaded
        assert run.progress == 1.0

    def test_load_failure_is_isolated(self):
        bad = FakeBackend(fail_load=True)
        good = FakeBackend()
        load_fn = make_loader({"/m/bad.gguf": bad, "/m/good.gguf": good})

        runner = SuiteRunner(load_fn=load_fn, unload_fn=lambda x: x.unload(), clock=StepClock())
        run = runner.run(["/m/bad.gguf", "/m/good.gguf"])

        assert run.status == "done"
        assert run.models[0].status == "error"
        assert "cannot load" in (run.models[0].error or "")
        assert run.models[1].status == "ok"
        assert run.models[1].avg_tps > 0

    def test_cancel_before_run_marks_all_cancelled(self):
        load_fn = make_loader({"/m/a.gguf": FakeBackend()})
        runner = SuiteRunner(load_fn=load_fn, clock=StepClock())
        runner.cancel()
        run = runner.run(["/m/a.gguf"])
        assert run.status == "cancelled"
        assert run.models[0].status == "cancelled"

    def test_leaderboard_sorted_fastest_first(self):
        slow = FakeBackend(tokens=1)
        fast = FakeBackend(tokens=10)
        load_fn = make_loader({"/m/slow.gguf": slow, "/m/fast.gguf": fast})
        runner = SuiteRunner(load_fn=load_fn, clock=StepClock())
        run = runner.run(["/m/slow.gguf", "/m/fast.gguf"])

        board = run.leaderboard()
        assert [b["rank"] for b in board] == [1, 2]
        assert board[0]["name"] == "fast"
        assert board[0]["avg_tps"] >= board[1]["avg_tps"]

    def test_last_run_is_exposed_for_polling(self):
        load_fn = make_loader({"/m/a.gguf": FakeBackend()})
        runner = SuiteRunner(load_fn=load_fn, clock=StepClock())
        assert runner.last_run is None
        run = runner.run(["/m/a.gguf"])
        assert runner.last_run is run

    def test_events_are_emitted(self):
        load_fn = make_loader({"/m/a.gguf": FakeBackend()})
        runner = SuiteRunner(load_fn=load_fn, clock=StepClock())
        events = []
        runner.run(["/m/a.gguf"], on_event=events.append)
        kinds = [e["type"] for e in events]
        assert "run_start" in kinds
        assert "model_start" in kinds
        assert kinds.count("prompt_done") == len(SUITE_PROMPTS)
        assert kinds[-1] == "run_done"

    def test_gpu_device_sets_use_gpu(self):
        seen = {}

        def load_fn(path, *, use_gpu=False, n_ctx=None):
            seen["use_gpu"] = use_gpu
            profile = types.SimpleNamespace(architecture="x", quantization="", param_count_estimate="", file_size_gb=0.0)
            return FakeBackend(), profile, types.SimpleNamespace(n_ctx=1024)

        runner = SuiteRunner(load_fn=load_fn, clock=StepClock())
        runner.run(["/m/a.gguf"], device="gpu", prompt_ids=["short"])
        assert seen["use_gpu"] is True


# ── Store + exports ──────────────────────────────────────────────────────


class TestStoreAndExports:
    def _run(self):
        load_fn = make_loader({"/m/a.gguf": FakeBackend()})
        runner = SuiteRunner(load_fn=load_fn, clock=StepClock())
        return runner.run(["/m/a.gguf"])

    def test_store_round_trip(self, tmp_path):
        run = self._run()
        store = SuiteStore(tmp_path / "suites")
        store.save(run)

        assert store.get(run.run_id)["run_id"] == run.run_id
        listed = store.list()
        assert listed and listed[0]["run_id"] == run.run_id
        assert listed[0]["model_count"] == 1

    def test_get_missing_returns_none(self, tmp_path):
        assert SuiteStore(tmp_path / "suites").get("nope") is None

    def test_export_markdown_has_leaderboard(self):
        md = export_markdown(self._run())
        assert "Leaderboard" in md
        assert "Tokens/sec" in md
        assert "| a |" in md

    def test_export_csv_has_header_and_rows(self):
        csv_text = export_csv(self._run())
        lines = [ln for ln in csv_text.splitlines() if ln.strip()]
        assert lines[0].startswith("run_id,device,model")
        assert len(lines) == len(SUITE_PROMPTS) + 1  # header + one row per prompt

    def test_export_markdown_lists_failures(self):
        bad = FakeBackend(fail_load=True)
        load_fn = make_loader({"/m/bad.gguf": bad})
        runner = SuiteRunner(load_fn=load_fn, clock=StepClock())
        run = runner.run(["/m/bad.gguf"])
        assert "Failed models" in export_markdown(run)


# ── Route wiring ─────────────────────────────────────────────────────────


class TestRoutes:
    def test_suite_routes_registered(self):
        from ggufloader.api.routes.benchmark import router as bench_router

        paths = {r.path for r in bench_router.routes if hasattr(r, "path")}
        for expected in (
            "/suite/prompts",
            "/suite/models",
            "/suite/run",
            "/suite/status",
            "/suite/cancel",
            "/suite/runs",
            "/suite/runs/{run_id}",
            "/suite/runs/{run_id}/export",
            "/run",
        ):
            assert expected in paths, f"missing route {expected}"

    def test_router_still_has_legacy_routes(self):
        from ggufloader.api.routes.benchmark import router as bench_router

        paths = {r.path for r in bench_router.routes if hasattr(r, "path")}
        assert {"/prompts", "/history", "/reports"} <= paths

    def test_suite_prompts_defined(self):
        assert len(SUITE_PROMPTS) >= 3
        assert all(isinstance(p, SuitePrompt) for p in SUITE_PROMPTS)

    def test_dataclass_helpers(self):
        r = PromptResult("short", "Short", 10, 20.0, 30.0, 100.0)
        assert r.to_dict()["tokens_per_second"] == 100.0
        m = ModelResult(path="/x/a.gguf", name="a")
        m.prompts.append(r)
        assert m.avg_tps == 100.0 and m.total_tokens == 10
        run = SuiteRun(run_id="r1", models=[m])
        assert run.to_dict()["models"][0]["name"] == "a"
