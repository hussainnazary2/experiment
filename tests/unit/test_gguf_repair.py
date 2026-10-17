"""Tests for in-place GGUF metadata repair (core/llm/gguf_repair.py).

The reported failure these cover:

    error loading model: error loading model hyperparameters: key
    qwen35.rope.dimension_sections has wrong array length; expected 4, got 3

llama.cpp reads that key into ``std::array<int, 4>`` for qwen2vl / qwen3vl /
qwen35 models and refuses any other length; some exporters (Ollama publishes
both shapes) write only the three sections that carry data. The repair appends
the missing zero *without* moving any tensor data, because tensor offsets are
relative to the aligned start of the tensor data: the bytes come from the
alignment padding when there is enough of it, and otherwise from shortening a
provenance-only string by exactly the shortfall.

Headless: synthetic GGUF fixtures only - no model, GPU or llama-cpp-python.
"""

from __future__ import annotations

import struct

from ggufloader.core.llm.gguf_meta import (
    read_tensor_names,
    read_tensor_table,
    scan_layout,
)
from ggufloader.core.llm.gguf_repair import diagnose, repair, undo

# A recognizable pattern so "the tensor data was not touched" is checkable.
TENSOR_DATA = bytes(range(256)) * 16      # 4096 bytes: one F32 [32, 32] tensor
KEY = "qwen35.rope.dimension_sections"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _kv_str(key: str, value: str) -> bytes:
    k, v = key.encode(), value.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v


def _kv_int_array(key: str, values) -> bytes:
    k = key.encode()
    return (
        struct.pack("<Q", len(k)) + k
        + struct.pack("<I", 9)                       # GGUF_TYPE_ARRAY
        + struct.pack("<I", 5)                       # element type INT32
        + struct.pack("<Q", len(values))
        + b"".join(struct.pack("<i", v) for v in values)
    )


def _assemble(arch, sections, values, filler, filler_key, extra_kv, tensor_data,
              tensor_name="blk.0.weight", tensor_dims=(32, 32)):
    pairs = [
        _kv_str("general.architecture", arch),
        _kv_str("general.name", "Test model"),
        _kv_int_array(f"{arch}.rope.dimension_sections", values[:sections]),
    ]
    if filler_key:
        pairs.append(_kv_str(filler_key, filler))
    pairs.extend(_kv_str(k, v) for k, v in extra_kv)
    kv = b"".join(pairs)
    header = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1)
              + struct.pack("<Q", len(pairs)))
    name = tensor_name.encode()
    tensor = (struct.pack("<Q", len(name)) + name
              + struct.pack("<I", len(tensor_dims))
              + b"".join(struct.pack("<Q", d) for d in tensor_dims)
              + struct.pack("<I", 0) + struct.pack("<Q", 0))     # F32 at offset 0
    body = header + kv + tensor
    table_end = len(body)
    data_start = (table_end + 31) // 32 * 32
    blob = body + b"\x00" * (data_start - table_end) + tensor_data
    return blob, table_end


def write_gguf(tmp_path, name="model.gguf", *, arch="qwen35", sections=3,
               values=(11, 11, 10, 0), target_padding=8,
               filler_key="general.description", filler="x" * 64,
               extra_kv=(), tensor_data=TENSOR_DATA,
               tensor_name="blk.0.weight", tensor_dims=(32, 32)):
    """Write a fixture whose header padding is exactly *target_padding*."""
    blob, table_end = _assemble(arch, sections, values, filler, filler_key,
                               extra_kv, tensor_data, tensor_name, tensor_dims)
    if target_padding is not None:
        # Growing the filler value by N shrinks the padding by N (mod 32).
        grown = (table_end % 32 and 32 - table_end % 32) % 32
        padding = (32 - table_end % 32) % 32
        if padding != target_padding:
            blob, table_end = _assemble(arch, sections, values,
                                        filler + "x" * ((padding - target_padding) % 32),
                                        filler_key, extra_kv, tensor_data, tensor_name,
                                        tensor_dims)
    assert (32 - table_end % 32) % 32 == target_padding
    path = tmp_path / name
    path.write_bytes(blob)
    return path


def _read_int_array(path, key=KEY):
    layout = scan_layout(path)
    arr = layout.arrays[key]
    with open(path, "rb") as f:
        f.seek(arr.data)
        return [struct.unpack("<i", f.read(4))[0] for _ in range(arr.count)]


def _value_length(path, key) -> int:
    layout = scan_layout(path)
    start, _end = layout.kv_ranges[key]
    with open(path, "rb") as f:
        f.seek(start)
        (name_len,) = struct.unpack("<Q", f.read(8))
        f.seek(start + 8 + name_len + 4)
        return struct.unpack("<Q", f.read(8))[0]


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def test_diagnose_finds_the_short_sections_array(tmp_path):
    path = write_gguf(tmp_path)
    findings = diagnose(path)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.code == "rope_sections_short"
    assert finding.repairable
    assert KEY in finding.summary and "requires 4" in finding.summary
    assert finding.plan is not None and finding.plan.mode == "padding"


def test_diagnose_is_quiet_for_a_healthy_file(tmp_path):
    path = write_gguf(tmp_path, sections=4)
    assert diagnose(path) == []
    report = repair(path)
    assert report.applied is False
    assert "no known metadata problem" in report.message


def test_diagnose_will_not_guess_when_there_are_too_many_entries(tmp_path):
    path = write_gguf(tmp_path, sections=5, values=(1, 2, 3, 4, 5))
    before = path.read_bytes()

    findings = diagnose(path)
    assert findings and not findings[0].repairable
    assert "exactly 4" in findings[0].summary
    assert repair(path).applied is False
    assert path.read_bytes() == before


def test_diagnose_leaves_unknown_architectures_alone(tmp_path):
    path = write_gguf(tmp_path, arch="llama", sections=3)
    findings = diagnose(path)
    assert findings and not findings[0].repairable
    assert "not one the app knows" in findings[0].detail


def test_diagnose_reports_nothing_for_a_file_it_cannot_parse(tmp_path):
    path = tmp_path / "junk.gguf"
    path.write_bytes(b"not a gguf at all")
    assert diagnose(path) == []
    assert repair(path).applied is False


# ---------------------------------------------------------------------------
# Repair: padding mode
# ---------------------------------------------------------------------------

def test_repair_appends_the_missing_zero_in_place(tmp_path):
    path = write_gguf(tmp_path, target_padding=8)
    before_size = path.stat().st_size
    before_data_start = scan_layout(path).data_start
    before_tail = path.read_bytes()[before_data_start:]

    report = repair(path)

    assert report.applied is True
    assert report.mode == "padding"
    assert report.trimmed_keys == []
    assert path.stat().st_size == before_size                    # size unchanged
    after = scan_layout(path)
    assert after is not None and after.data_start == before_data_start
    assert _read_int_array(path) == [11, 11, 10, 0]
    assert path.read_bytes()[before_data_start:] == before_tail  # data untouched
    assert report.file_size_before == report.file_size_after == before_size


def test_repair_writes_a_header_backup_and_can_be_undone(tmp_path):
    path = write_gguf(tmp_path, target_padding=8)
    original = path.read_bytes()

    report = repair(path)
    assert report.backup_path and path.with_name(path.name + ".header-backup").is_file()
    assert _read_int_array(path) == [11, 11, 10, 0]

    restored = undo(path)
    assert restored.applied is True
    assert path.read_bytes() == original              # byte-for-byte original
    assert _read_int_array(path) == [11, 11, 10]
    # and it can be repaired again afterwards
    assert repair(path).applied is True
    assert _read_int_array(path) == [11, 11, 10, 0]


def test_undo_without_a_backup_says_so(tmp_path):
    path = write_gguf(tmp_path)
    report = undo(path)
    assert report.applied is False
    assert "no header backup" in report.message


# ---------------------------------------------------------------------------
# Repair: exact-space trim mode
# ---------------------------------------------------------------------------

def test_repair_trims_a_provenance_string_when_there_is_no_padding(tmp_path):
    """No alignment padding? Shorten a provenance string by exactly the shortfall."""
    path = write_gguf(tmp_path, target_padding=1, filler_key="general.license.link",
                      filler="L" * 64)
    before = scan_layout(path)
    before_bytes = path.stat().st_size
    before_tail = path.read_bytes()[before.data_start:]
    before_len = _value_length(path, "general.license.link")
    assert before.padding == 1 < 4                     # the reason for trim mode

    report = repair(path)

    assert report.applied is True
    assert report.mode == "trim-value"
    assert report.trimmed_keys == ["general.license.link"]
    after = scan_layout(path)
    assert after is not None
    assert after.data_start == before.data_start       # tensor data did not move
    assert after.kv_count == before.kv_count           # no keys were dropped
    assert path.stat().st_size == before_bytes
    assert _read_int_array(path) == [11, 11, 10, 0]
    # exactly 3 bytes (needed 4 - padding 1) came off that one value
    assert _value_length(path, "general.license.link") == before_len - 3
    assert _value_length(path, "general.name") == len("Test model")
    assert path.read_bytes()[before.data_start:] == before_tail


def test_repair_prefers_the_least_meaningful_key(tmp_path):
    path = write_gguf(tmp_path, target_padding=1, filler_key="general.license.link",
                      filler="L" * 64, extra_kv=(("general.description", "d" * 64),))
    report = repair(path)
    assert report.trimmed_keys == ["general.license.link"]
    assert _value_length(path, "general.description") == 64      # untouched


def test_repair_never_trims_a_non_string(tmp_path):
    """An array like general.tags is left alone - only strings can be shortened."""
    path = write_gguf(tmp_path, target_padding=1, filler_key="tokenizer.ggml.model",
                      filler="gpt2" + "x" * 60)
    before = path.read_bytes()

    findings = diagnose(path)
    assert findings and not findings[0].repairable
    assert "re-converted" in findings[0].detail
    assert repair(path).applied is False
    assert path.read_bytes() == before


def test_repair_refuses_when_the_only_string_is_too_short(tmp_path):
    # One droppable string, but shorter than the 3 bytes needed.
    path = write_gguf(tmp_path, target_padding=1, filler_key="general.size_label",
                      filler="1B")
    assert repair(path).applied is False


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

def test_repair_verifies_its_own_work_and_rolls_back(tmp_path, monkeypatch):
    """A repair that does not verify must leave the file exactly as it was."""
    path = write_gguf(tmp_path, target_padding=8)
    original = path.read_bytes()

    import ggufloader.core.llm.gguf_repair as repair_mod

    monkeypatch.setattr(repair_mod, "_verify",
                        lambda *a, **k: (False, "simulated verification failure"))
    report = repair(path)

    assert report.applied is False
    assert "put back" in report.message
    assert path.read_bytes() == original


def test_a_write_failure_rolls_back(tmp_path, monkeypatch):
    path = write_gguf(tmp_path, target_padding=8)
    original = path.read_bytes()

    import ggufloader.core.llm.gguf_repair as repair_mod

    def boom(*args, **kwargs):
        raise ValueError("simulated write failure")

    monkeypatch.setattr(repair_mod, "_rewrite_region", boom)
    report = repair(path)

    assert report.applied is False
    assert "put back" in report.message
    assert path.read_bytes() == original


def test_missing_file_is_reported_not_raised(tmp_path):
    report = repair(tmp_path / "nope.gguf")
    assert report.applied is False
    assert "no such file" in report.message


# ---------------------------------------------------------------------------
# API: diagnose / repair / undo, and the loaded-model guard
# ---------------------------------------------------------------------------

def _client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import ggufloader.api.routes.model as model_routes

    app = FastAPI()
    app.include_router(model_routes.router, prefix="/api/model")
    return TestClient(app)


def test_api_repair_needs_confirm_and_can_be_undone(tmp_path):
    from ggufloader.api import deps

    path = write_gguf(tmp_path, target_padding=8)
    with _client() as client:
        diagnosed = client.post("/api/model/diagnose", json={"path": str(path)}).json()
        assert diagnosed["repairable"] is True
        assert diagnosed["findings"][0]["code"] == "rope_sections_short"
        assert diagnosed["findings"][0]["repairable"] is True

        planned = client.post("/api/model/repair", json={"path": str(path)}).json()
        assert planned["planned"] is True and planned["applied"] is False
        assert planned["confirm_required"] is True
        assert _read_int_array(path) == [11, 11, 10]          # nothing written yet

        done = client.post("/api/model/repair",
                           json={"path": str(path), "confirm": True}).json()
        assert done["applied"] is True
        assert _read_int_array(path) == [11, 11, 10, 0]

        undone = client.post("/api/model/repair/undo", json={"path": str(path)}).json()
        assert undone["applied"] is True
        assert _read_int_array(path) == [11, 11, 10]
    deps.set_model_backend(None)


def test_api_refuses_to_repair_the_loaded_model(tmp_path):
    from ggufloader.api import deps

    path = write_gguf(tmp_path, target_padding=8)

    class Loaded:
        model_path = str(path)

    deps.set_model_backend(Loaded())
    try:
        with _client() as client:
            res = client.post("/api/model/repair",
                              json={"path": str(path), "confirm": True})
        assert res.status_code == 409
        assert "unload it first" in res.json()["detail"]
        assert _read_int_array(path) == [11, 11, 10]          # untouched
    finally:
        deps.set_model_backend(None)


# ---------------------------------------------------------------------------
# Tensor-name failures - what a repaired file hit next on the real machine
# ---------------------------------------------------------------------------

#: llama.cpp's message from the reported machine, verbatim in shape.
LOAD_ERROR = (
    "Failed to load model from file: F:\\AIDeployMasterModels\\gguf\\"
    "qwen3.5-9.7b-q4_k_m.gguf - llama.cpp reported: error loading model: missing "
    "tensor 'blk.0.ssm_dt.bias'; failed to load model"
)


def test_read_tensor_names_lists_the_tensor_table(tmp_path):
    path = write_gguf(tmp_path, tensor_name="blk.0.ssm_dt.weight")
    assert read_tensor_names(path) == ["blk.0.ssm_dt.weight"]
    assert read_tensor_names(tmp_path / "missing.gguf") == []   # never raises
    # Shapes come along for the ride: they decide naming variant vs absent.
    assert read_tensor_table(path) == [("blk.0.ssm_dt.weight", [32, 32])]


def test_same_parameter_under_another_name_is_a_naming_mismatch(tmp_path):
    # sections=4 keeps the metadata clean so the tensor finding is the only one.
    # 1-D dims: the shape a dt bias has, so this is the same parameter respelled.
    path = write_gguf(tmp_path, sections=4, tensor_name="blk.0.ssm_dt.weight",
                      tensor_dims=(32,))
    findings = [f for f in diagnose(path, error_text=LOAD_ERROR)
                if f.code == "tensor_naming_mismatch"]
    assert len(findings) == 1
    finding = findings[0]
    assert "'blk.0.ssm_dt.weight' (32)" in finding.summary  # what the file holds
    assert "blk.0.ssm_dt.bias" in finding.summary          # what llama.cpp wants
    assert "1-D" in finding.detail                          # shape checked
    assert "pip install -U llama-cpp-python" in finding.detail
    assert finding.repairable is False                      # never a silent rename
    assert "rename" in finding.detail                       # says why not


def test_a_2d_sibling_is_a_different_parameter_not_a_respelling(tmp_path):
    # A dt *projection* matrix is not the dt *bias*: nothing may rename it, and
    # no runtime upgrade can conjure the bias the loader asks for.
    path = write_gguf(tmp_path, sections=4, tensor_name="blk.0.ssm_dt.weight",
                      tensor_dims=(2048, 32))
    finding = next(f for f in diagnose(path, error_text=LOAD_ERROR)
                   if f.code == "tensor_naming_mismatch")
    assert "'blk.0.ssm_dt.weight' (2048x32)" in finding.summary
    assert "not in the file at all" in finding.detail
    assert "pip install -U llama-cpp-python" not in finding.detail  # upgrade is not the fix


def test_the_bare_spelling_counts_as_the_same_parameter(tmp_path):
    # Some writers store it with no suffix at all.
    path = write_gguf(tmp_path, sections=4, tensor_name="blk.0.ssm_dt",
                      tensor_dims=(32,))
    findings = [f for f in diagnose(path, error_text=LOAD_ERROR)
                if f.code == "tensor_naming_mismatch"]
    assert len(findings) == 1
    assert "blk.0.ssm_dt" in findings[0].summary


def test_absent_tensor_is_reported_as_missing_not_as_a_naming_issue(tmp_path):
    path = write_gguf(tmp_path, sections=4, tensor_name="blk.0.weight")
    findings = diagnose(path, error_text=LOAD_ERROR)
    codes = [f.code for f in findings]
    assert "tensor_missing" in codes and "tensor_naming_mismatch" not in codes
    missing = next(f for f in findings if f.code == "tensor_missing")
    assert "does not contain" in missing.summary
    assert "no runtime upgrade can supply it" in missing.detail
    assert missing.repairable is False


def test_tensor_that_is_present_is_reported_as_present(tmp_path):
    path = write_gguf(tmp_path, sections=4, tensor_name="blk.0.ssm_dt.bias")
    findings = diagnose(path, error_text=LOAD_ERROR)
    assert [f.code for f in findings] == ["tensor_present"]
    assert "does list it" in findings[0].summary
    assert findings[0].repairable is False


def test_metadata_and_tensor_findings_are_both_reported(tmp_path):
    # The real file has both problems: 3 M-RoPE sections and a tensor the
    # loader names differently.
    path = write_gguf(tmp_path, sections=3, tensor_name="blk.0.ssm_dt.weight")
    codes = {f.code for f in diagnose(path, error_text=LOAD_ERROR)}
    assert {"rope_sections_short", "tensor_naming_mismatch"} <= codes


def test_no_error_text_means_no_tensor_findings(tmp_path):
    path = write_gguf(tmp_path, sections=4, tensor_name="blk.0.ssm_dt.weight")
    assert [f.code for f in diagnose(path)] == []


def _profile():
    from ggufloader.core.router import ModelProfile
    return ModelProfile(path="x.gguf", filename="qwen3.5-9.7b-q4_k_m.gguf",
                        architecture="qwen35", file_size_gb=6.0)


def test_hint_for_a_tensor_mismatch_names_the_tensor_and_the_upgrade():
    from ggufloader.core.router import _load_failure_hint
    hint = _load_failure_hint(_profile(), LOAD_ERROR, use_gpu=False)
    assert "blk.0.ssm_dt.bias" in hint
    assert "pip install -U llama-cpp-python" in hint
    assert "/api/model/diagnose" in hint
    assert "gave no specific reason" not in hint


def test_hint_never_claims_llama_cpp_said_nothing_when_it_did():
    from ggufloader.core.router import _load_failure_hint
    unmapped = ("Failed to load model from file: x.gguf - llama.cpp reported: "
                "error loading model: something brand new; failed to load model")
    hint = _load_failure_hint(_profile(), unmapped, use_gpu=False)
    assert "gave no specific reason" not in hint
    assert "/api/model/diagnose" in hint
    # ...but the old wording still stands when there is genuinely no reason.
    bare = _load_failure_hint(_profile(), "Failed to load model from file: x.gguf",
                              use_gpu=False)
    assert "gave no specific reason" in bare


def test_api_diagnose_takes_the_load_error_and_reports_the_backup(tmp_path):
    path = write_gguf(tmp_path, sections=4, tensor_name="blk.0.ssm_dt.weight")
    fixable = write_gguf(tmp_path, name="fixable.gguf", target_padding=8)
    with _client() as client:
        res = client.post("/api/model/diagnose",
                          json={"path": str(path), "error": LOAD_ERROR})
        assert res.status_code == 200
        body = res.json()
        assert body["has_header_backup"] is False
        assert body["repairable"] is False
        assert any(f["code"] == "tensor_naming_mismatch" for f in body["findings"])

        # Once a repair has been applied, the UI can offer to undo it - which is
        # exactly what matters when the repair did not fix the load.
        assert client.post("/api/model/repair",
                           json={"path": str(fixable), "confirm": True}
                           ).json()["applied"] is True
        after = client.post("/api/model/diagnose",
                            json={"path": str(fixable)}).json()
        assert after["has_header_backup"] is True
        assert after["repairable"] is False      # already fixed
