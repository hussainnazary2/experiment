"""Load failures must say what actually went wrong.

Bug report: two Qwen GGUFs refused to load with

    Could not load qwen3.5-2b-q8_0.gguf after 1 attempt(s) (cpu). Last error:
    Failed to load model from file: F:\\...\\qwen3.5-2b-q8_0.gguf.
    The model file may be corrupt or unsupported.

llama-cpp-python reports every refusal with that one opaque sentence; the real
reason (unknown architecture, unknown quantization, truncated file, out of
memory) only reaches stderr, which the windowed app discards. These tests pin
the three pieces that now surface it:

1. ``ModelBackend.load`` folds llama.cpp's captured log into the exception,
2. ``ModelRouter`` maps the reason to an actionable hint,
3. ``check_integrity`` catches an incomplete download before llama.cpp sees it.
"""

from __future__ import annotations

import struct
import sys

import pytest

import ggufloader.core.llm.model_backend as model_backend
from ggufloader.core.llm.gguf_meta import check_integrity
from ggufloader.core.llm.model_backend import ModelBackend, summarize_llama_log
from ggufloader.core.router import ModelRole, ModelRouter, SystemProfile, _load_failure_hint

from tests.unit.test_router import _create_model_file, _kv_str, _kv_u32


# ---------------------------------------------------------------------------
# summarize_llama_log
# ---------------------------------------------------------------------------

def test_summarize_llama_log_extracts_the_reason():
    log = (
        "llama_model_loader: loaded meta data with 33 key-value pairs\n"
        "llama_model_loader: error loading model: unknown model architecture: 'qwen3.5'\n"
        "llama_load_model_from_file: failed to load model\n"
    )
    summary = summarize_llama_log(log)
    assert "unknown model architecture: 'qwen3.5'" in summary
    assert len(summary) < 400


def test_summarize_llama_log_is_empty_for_an_empty_log():
    assert summarize_llama_log("") == ""
    assert isinstance(summarize_llama_log("\n\n"), str)  # never raises


# ---------------------------------------------------------------------------
# ModelBackend.load folds the captured log into the error
# ---------------------------------------------------------------------------

class ReasonedFailingLlama:
    """Prints llama.cpp's real diagnosis, then fails the way llama-cpp does."""

    def __init__(self, **kwargs):
        print(
            "llama_model_loader: error loading model: unknown model architecture: 'qwen3.5'",
            file=sys.stderr,
        )
        raise ValueError(f"Failed to load model from file: {kwargs.get('model_path')}")


class SilentFailingLlama:
    def __init__(self, **kwargs):
        raise ValueError(f"Failed to load model from file: {kwargs.get('model_path')}")


def test_load_failure_carries_the_llama_cpp_reason(monkeypatch):
    monkeypatch.setattr(model_backend, "LLAMA_AVAILABLE", True)
    monkeypatch.setattr(model_backend, "Llama", ReasonedFailingLlama)

    backend = ModelBackend("qwen3.5-2b-q8_0.gguf")
    with pytest.raises(RuntimeError) as excinfo:
        backend.load()

    message = str(excinfo.value)
    assert "Failed to load model from file" in message      # original error kept
    assert "unknown model architecture: 'qwen3.5'" in message  # real reason added


def test_load_failure_without_a_log_keeps_the_original_error(monkeypatch):
    monkeypatch.setattr(model_backend, "LLAMA_AVAILABLE", True)
    monkeypatch.setattr(model_backend, "Llama", SilentFailingLlama)

    with pytest.raises(ValueError):
        ModelBackend("plain.gguf").load()


# ---------------------------------------------------------------------------
# _load_failure_hint
# ---------------------------------------------------------------------------

def _profile(name="qwen3.5-2b-q8_0.gguf", arch="qwen3", size=2.1):
    from ggufloader.core.router import ModelProfile
    return ModelProfile(path=name, filename=name, architecture=arch, file_size_gb=size)


def test_hint_names_the_architecture_and_the_upgrade():
    hint = _load_failure_hint(
        _profile(), "ValueError: Failed to load model from file: x - llama.cpp "
                    "reported: unknown model architecture: 'qwen3.5'", use_gpu=False)
    assert "qwen3" in hint
    assert "llama-cpp-python" in hint


def test_hint_for_a_metadata_shape_mismatch_points_at_the_repair():
    """The reported Qwen3.5 file: complete, but 3 sections where llama.cpp reads 4."""
    hint = _load_failure_hint(
        _profile(),
        "Failed to load model from file: x - llama.cpp reported: error loading model: "
        "error loading model hyperparameters: key qwen35.rope.dimension_sections has "
        "wrong array length; expected 4, got 3; failed to load model",
        use_gpu=False)
    assert "metadata is not the shape" in hint
    assert "Repair metadata" in hint
    # It must not tell people to re-download a file that is perfectly complete.
    assert "re-download" not in hint.lower()


def test_hint_for_a_truncated_file_says_redownload():
    hint = _load_failure_hint(
        _profile(), "llama.cpp reported: tensor data is truncated", use_gpu=False)
    assert "truncated" in hint.lower()
    assert "download" in hint.lower()


def test_hint_for_out_of_memory_suggests_a_smaller_quant():
    hint = _load_failure_hint(
        _profile(), "llama.cpp reported: failed to allocate 4096 MB", use_gpu=False)
    assert "memory" in hint.lower()


def test_hint_stays_honest_when_the_reason_is_unknown():
    """No reason captured: say so instead of asserting the file is corrupt."""
    hint = _load_failure_hint(_profile(), "some unexplained failure", use_gpu=False)
    assert "may be corrupt" not in hint.lower()
    assert "no specific reason" in hint.lower()


# ---------------------------------------------------------------------------
# check_integrity: incomplete downloads
# ---------------------------------------------------------------------------

def _gguf_bytes(data_bytes: int) -> bytes:
    """A one-tensor F32 [32, 32] GGUF: its tensor table needs 4096 data bytes."""
    name = b"blk.0.weight"
    body = (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 1)                       # tensor count
        + struct.pack("<Q", 0)                       # kv count
        + struct.pack("<Q", len(name)) + name
        + struct.pack("<I", 2)                       # n_dims
        + struct.pack("<Q", 32) + struct.pack("<Q", 32)
        + struct.pack("<I", 0)                       # F32
        + struct.pack("<Q", 0)                       # offset
    )
    data_start = (len(body) + 31) // 32 * 32
    return body + b"\x00" * (data_start - len(body) + data_bytes)


def test_check_integrity_accepts_a_complete_file(tmp_path):
    p = tmp_path / "complete.gguf"
    p.write_bytes(_gguf_bytes(32 * 32 * 4))
    result = check_integrity(p)
    assert result.ok and result.code == "ok"


def test_check_integrity_flags_an_incomplete_download(tmp_path):
    p = tmp_path / "partial.gguf"
    p.write_bytes(_gguf_bytes(64))                   # 4024 bytes short
    result = check_integrity(p)
    assert not result.ok
    assert result.code == "truncated"
    assert "tensor table" in result.detail


def test_check_integrity_flags_a_non_gguf_file(tmp_path):
    p = tmp_path / "renamed.bin"
    p.write_bytes(b"NOPE" + b"\x00" * 4096)
    result = check_integrity(p)
    assert not result.ok and result.code == "not_gguf"


def test_check_integrity_never_blocks_what_it_cannot_parse(tmp_path):
    """Unparseable or newer-format files are llama.cpp's call, not ours."""
    broken = tmp_path / "broken-table.gguf"
    broken.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 2 ** 40) + struct.pack("<Q", 0)
    )
    assert check_integrity(broken).ok

    metadata_only = _create_model_file(
        tmp_path, "metadata-only.gguf",
        _kv_str("general.architecture", "qwen3") + _kv_u32("qwen3.block_count", 28))
    assert check_integrity(metadata_only).ok


# ---------------------------------------------------------------------------
# Router integration: refuse early, and name the reason
# ---------------------------------------------------------------------------

def _cpu_router() -> ModelRouter:
    return ModelRouter(system=SystemProfile(
        ram_gb=16.0, vram_gb=0.0, has_gpu_support=False))


def test_router_refuses_a_truncated_file_without_touching_llama_cpp(tmp_path, monkeypatch):
    p = tmp_path / "qwen3.5-9.7b-q4_k_m.gguf"
    p.write_bytes(_gguf_bytes(64))

    attempts: list[dict] = []

    def must_not_run(*args, **kwargs):
        attempts.append(kwargs)
        raise AssertionError("llama.cpp must not be asked to load a truncated file")

    monkeypatch.setattr(model_backend, "ModelBackend", must_not_run)

    with pytest.raises(RuntimeError) as excinfo:
        _cpu_router().load(str(p), ModelRole.CHAT)

    assert "truncated" in str(excinfo.value)
    assert attempts == []


def test_router_error_names_the_reason_and_the_fix(tmp_path, monkeypatch):
    p = _create_model_file(
        tmp_path, "qwen3.5-2b-q8_0.gguf",
        _kv_str("general.architecture", "qwen3") + _kv_u32("qwen3.block_count", 28))

    class UnsupportedArchBackend:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            raise RuntimeError(
                "Failed to load model from file: qwen3.5-2b-q8_0.gguf - "
                "llama.cpp reported: unknown model architecture: 'qwen3.5'")

        def unload(self):
            pass

    monkeypatch.setattr(model_backend, "ModelBackend", UnsupportedArchBackend)

    with pytest.raises(RuntimeError) as excinfo:
        _cpu_router().load(str(p), ModelRole.CHAT)

    message = str(excinfo.value)
    assert "unknown model architecture" in message     # the real reason survives
    assert "after 1 attempt" in message                # the ladder report is intact
    assert "llama-cpp-python" in message               # actionable next step
    assert "may be corrupt or unsupported" not in message
