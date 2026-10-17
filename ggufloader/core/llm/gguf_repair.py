"""In-place repair of GGUF metadata that llama.cpp refuses to read.

Why this exists
---------------
llama.cpp reads the M-RoPE section sizes into ``std::array<int, 4>`` for every
multi-axis architecture it knows (``qwen2vl``, ``qwen3vl``, ``qwen35``,
``qwen35moe``) and refuses a file that declares any other length::

    error loading model: error loading model hyperparameters: key
    qwen35.rope.dimension_sections has wrong array length; expected 4, got 3

The fourth section has no axis, so it is always 0 - llama.cpp's own log of a
working Qwen3.5 prints ``[11, 11, 10, 0]`` - but some exporters write only the
three that carry data. Ollama publishes both shapes: its ``qwen3.5:2b-q8_0``
blob has the four-entry form, while other tags carry three. A model that is
otherwise perfect is then unloadable over one missing zero, and llama.cpp
reports that as a generic load failure.

Repairing it does not need a re-quantize
----------------------------------------
Tensor offsets are *relative* to the aligned start of the tensor data, so the
file is consistent as long as that start does not move. The metadata section is
followed by the tensor table and then alignment padding, which makes the length
budget explicit: with padding ``p`` to spare, the repair may insert at most
``p`` bytes, and any bytes removed have to be offset by a filler of the same
size - so what the repair needs is not "room" but *exact* arithmetic.

* Padding mode: the file has at least as much padding as the missing entries
  need. Append them and let the padding shrink by the same amount.
* Trim mode: no padding to spare. Shrink a provenance-only string value
  (``general.license.link``, ``general.base_model.*``, ...) by exactly the
  shortfall. llama.cpp never reads those to load or run a model, and because
  the amount is exact, the tensor data still does not move.

Either way the file keeps its size, none of the multi-gigabyte tensor data is
touched, the original header is written to ``<model>.header-backup`` first, and
the result is re-read and verified - a repair that fails its own check rolls
itself back.
"""

from __future__ import annotations

import logging
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ggufloader.core.llm.gguf_meta import (
    ArrayMeta,
    GgufLayout,
    read as read_metadata,
    read_tensor_table,
    scan_layout,
)

logger = logging.getLogger(__name__)

#: llama.cpp resolves tensors by exact name (qwen35.cpp / qwen3next.cpp build
#: the recurring-block names themselves), so its load error quotes the name it
#: was looking for. That name drives the tensor findings below.
MISSING_TENSOR_RE = re.compile(r"missing tensor ['\"]([^'\"]+)['\"]")
UNEXPECTED_TENSOR_RE = re.compile(r"unexpected tensor ['\"]([^'\"]+)['\"]")

#: llama.cpp's LLM_KV_ROPE_DIMENSION_SECTIONS suffix for M-RoPE models.
ROPE_SECTIONS_SUFFIX = ".rope.dimension_sections"

#: The length every loader above requires (std::array<int, 4>).
ROPE_SECTIONS_REQUIRED = 4

#: Architectures whose loader we know reads that key as four ints.
ROPE_SECTIONS_ARCHS = ("qwen2vl", "qwen3vl", "qwen35", "qwen35moe")

_INT32 = 5                       # GGUF_TYPE_INT32, the element type of the array
_STRING = 8                      # GGUF_TYPE_STRING

#: Provenance metadata, least meaningful first. llama.cpp never reads any of
#: these to load or run a model, so they are what a repair trades for space.
#: Deliberately narrow: nothing the loader or this app consumes is listed.
DROPPABLE_STRINGS = (
    "general.license.link",
    "general.base_model",
    "general.license",
    "general.author",
    "general.url",
    "general.doi",
    "general.repo_url",
    "general.size_label",
    "general.description",
)

#: Rewriting the header means holding it in memory. Metadata sections are a
#: few MB at most; anything beyond this is not a GGUF we should be editing.
MAX_HEADER_BYTES = 64 * 1024 * 1024

_BACKUP_SUFFIX = ".header-backup"


@dataclass
class RoomPlan:
    """How the repair will pay for the bytes it needs to insert."""

    mode: str                                   # "padding" | "trim-value"
    needed: int = 0                             # bytes to insert
    padding: int = 0                            # alignment padding available
    freed: int = 0                              # bytes removed elsewhere
    deletion: Optional[Tuple[int, int]] = None  # byte range to remove
    patch: Optional[Tuple[int, bytes]] = None   # extra write (the shrunk length)
    trim_key: str = ""
    trim_from: int = 0
    trim_to: int = 0
    modified_from: int = 0                      # first byte the plan touches

    def describe(self) -> str:
        if self.mode == "padding":
            return (f"append the missing entries into the {self.padding} bytes of "
                    f"alignment padding before the tensor data")
        return (f"shorten the provenance value '{self.trim_key}' from "
                f"{self.trim_from} to {self.trim_to} bytes to free exactly "
                f"{self.freed} byte(s)")


@dataclass
class Finding:
    """One known metadata problem, and whether it can be fixed in place."""

    code: str
    summary: str
    detail: str
    repairable: bool = False
    fix: str = ""
    target_key: str = ""
    plan: Optional[RoomPlan] = None


@dataclass
class RepairReport:
    """What a diagnose/repair/undo call found or did."""

    path: str
    applied: bool = False
    mode: str = ""
    findings: List[Finding] = field(default_factory=list)
    trimmed_keys: List[str] = field(default_factory=list)
    backup_path: Optional[str] = None
    file_size_before: int = 0
    file_size_after: int = 0
    message: str = ""


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _architecture(path: Path) -> str:
    return str(read_metadata(path).get("architecture") or "").lower()


def _string_value_span(path: Path, key: str, start: int, end: int
                       ) -> Optional[Tuple[int, int, int]]:
    """``(value_length_offset, value_offset, value_length)`` for a string KV.

    Returns None when that key is not a string (arrays and numbers are left
    alone: shrinking them would mean dropping whole elements or bits).
    """
    try:
        with open(path, "rb") as f:
            f.seek(start)
            (name_len,) = struct.unpack("<Q", f.read(8))
            if name_len != len(key.encode()):
                return None
            f.seek(start + 8 + name_len)
            (value_type,) = struct.unpack("<I", f.read(4))
            if value_type != _STRING:
                return None
            value_length_offset = f.tell()
            (value_length,) = struct.unpack("<Q", f.read(8))
            if value_length > end - (value_length_offset + 8):
                return None                     # inconsistent with the scan
            return value_length_offset, value_length_offset + 8, value_length
    except (struct.error, OSError):
        return None


def _plan_room(path: Path, layout: GgufLayout, needed: int, target_key: str
               ) -> Optional[RoomPlan]:
    """Find exactly ``needed`` bytes without moving the tensor data.

    ``data_start`` is computed as ``align_up(tensor_table_end, alignment)``, so
    writing the right *number* of bytes is what keeps it still: inserting
    ``needed`` into padding is fine while ``padding >= needed``, and any other
    source of bytes must free exactly ``needed - padding``.
    """
    if layout.padding >= needed:
        return RoomPlan(mode="padding", needed=needed, padding=layout.padding,
                        modified_from=0)

    shortfall = needed - layout.padding
    order = {prefix: rank for rank, prefix in enumerate(DROPPABLE_STRINGS)}
    candidates = []
    for key, (start, end) in layout.kv_ranges.items():
        if key == target_key:
            continue
        rank = next((order[p] for p in DROPPABLE_STRINGS if key.startswith(p)), None)
        if rank is None:
            continue
        span = _string_value_span(path, key, start, end)
        if span is None:
            continue
        value_length_offset, value_offset, value_length = span
        candidates.append((rank, -value_length, key, value_length_offset,
                           value_offset, value_length, start))
    candidates.sort()

    for _rank, _neg, key, value_length_offset, value_offset, value_length, start in candidates:
        if value_length < shortfall:
            continue
        return RoomPlan(
            mode="trim-value",
            needed=needed, padding=layout.padding, freed=shortfall,
            deletion=(value_offset + shortfall, value_offset + value_length),
            patch=(value_length_offset, struct.pack("<Q", value_length - shortfall)),
            trim_key=key, trim_from=value_length, trim_to=value_length - shortfall,
            modified_from=start,
        )
    return None


def _tensor_finding(p: Path, name: str, *, unexpected: bool) -> Finding:
    """Explain a llama.cpp tensor-name failure against the file's inventory.

    Two outcomes that look identical in llama.cpp's message but need opposite
    responses:

    * the file holds the parameter under a different name (a converter naming
      difference - e.g. Unsloth GGUFs write ``ssm_dt.bias`` where Ollama's own
      write ``ssm_dt``). A runtime that accepts both spellings fixes it, so the
      advice is to upgrade first.
    * the file does not hold it at all. No runtime can supply a missing tensor,
      so the file was written for a different layout and needs re-converting.

    A rename is deliberately *not* offered: two names that look similar can be
    two different parameters (``ssm_dt`` vs ``ssm_dt.bias`` differ in shape), and
    rewriting the tensor table to guess is how a file gets silently corrupted.
    """
    table = read_tensor_table(p)
    names = [n for n, _dims in table]
    verb = "unexpected" if unexpected else "missing"
    if name in names:
        return Finding(
            code="tensor_present",
            summary=f"llama.cpp reported the {verb} tensor '{name}', but the file does list it",
            detail=("the whole tensor table reads cleanly here, so the load stopped on a "
                    "mismatch llama.cpp did not name - most often a second tensor in the "
                    "same block, or a shape change. Note the key llama.cpp printed right "
                    "before this line and report it with the model name."),
            target_key=name,
        )

    # blk.0.ssm_dt.bias -> base 'blk.0.ssm_dt' in block 'blk.0'. The file may
    # hold that base bare or with another suffix, so both count as candidates.
    base, _, _suffix = name.rpartition(".")
    block = base.rpartition(".")[0] or base
    siblings = [(n, d) for n, d in table
                if base and (n == base or n.startswith(base + "."))]

    def label(n: str, dims) -> str:
        shape = "x".join(str(d) for d in dims) if dims else ""
        return f"'{n}'" + (f" ({shape})" if shape else "")

    if siblings:
        shown = ", ".join(label(n, d) for n, d in siblings[:3])
        more = f" (+{len(siblings) - 3} more)" if len(siblings) > 3 else ""
        flat = [d for _n, d in siblings if len(d) == 1]
        # Shape decides whether this is the same parameter under another name or
        # a genuinely different one - which is what pick the advice:
        # 1-D per-head tensor = a dt bias spelled differently, 2-D = a projection.
        shape_note = (
            f"One of them is 1-D ({'x'.join(str(x) for x in flat[0])}), the shape a dt "
            f"bias has, so this is most likely the same parameter under a different "
            f"spelling - some converters write the dt bias as a plain weight. Newer "
            f"llama.cpp/Ollama accepted more of these spellings over time, so try an "
            f"upgrade first: 'pip install -U llama-cpp-python'. If the load still fails, "
            f"convert the original weights with the current converter."
            if flat else
            f"None of them is 1-D, so the 1-D dt bias this loader requires is not in the "
            f"file at all - those are different parameters (a 2-D dt projection is not a "
            f"dt bias). No runtime upgrade can supply a tensor the file does not contain: "
            f"use a GGUF converted for llama.cpp, or convert the original weights with "
            f"the current converter."
        )
        return Finding(
            code="tensor_naming_mismatch",
            summary=(f"the file names this parameter {shown}{more}, but this build's "
                     f"llama.cpp asks for '{name}'"),
            detail=(
                "The file is complete and its tensor table parses, so this is a "
                "converter naming difference rather than a truncated download. Which "
                "side is old depends on the writer: Ollama's own GGUFs use "
                "'blk.N.ssm_dt.bias', while other exporters write the same parameter "
                "under a different name. " + shape_note + " The app will not rename "
                "tensors itself: two similar names can be two different parameters, and "
                "a wrong rename would corrupt the model silently instead of failing."
            ),
            target_key=name,
        )

    return Finding(
        code="tensor_missing",
        summary=(f"this file does not contain '{name}' (no tensor in block "
                 f"'{block or '?'}' starts with '{base}')"),
        detail=(
            "The file simply does not hold what this architecture's loader requires, so "
            "no runtime upgrade can supply it - this GGUF came from a converter whose "
            "tensor layout differs from the installed llama.cpp. Use a GGUF that was "
            "converted for llama.cpp (or convert the original weights with the current "
            "converter) rather than re-downloading the same file."
        ),
        target_key=name,
    )


def diagnose(path: str | Path, error_text: str = "") -> List[Finding]:
    """Report known metadata AND tensor problems; never writes to the file.

    ``error_text`` is llama.cpp's load message when there is one. It selects the
    tensor checks, because which tensor to look for is only known from the error
    the loader printed.
    """
    p = Path(path)
    layout = scan_layout(p)
    if layout is None:
        return []
    arch = _architecture(p)
    findings: List[Finding] = []

    for key, arr in layout.arrays.items():
        if not key.endswith(ROPE_SECTIONS_SUFFIX):
            continue
        if arr.count == ROPE_SECTIONS_REQUIRED:
            continue
        required = ROPE_SECTIONS_REQUIRED
        if arch not in ROPE_SECTIONS_ARCHS:
            findings.append(Finding(
                code="rope_sections_unknown_arch",
                summary=f"{key} has {arr.count} entries where llama.cpp reads {required}",
                detail=(f"architecture '{arch or 'unknown'}' is not one the app knows "
                        f"llama.cpp's expected section count for, so it will not touch it"),
            ))
            continue
        if arr.element_type != _INT32:
            findings.append(Finding(
                code="rope_sections_type",
                summary=f"{key} is not an array of 32-bit ints",
                detail=(f"the array's element type is {arr.element_type}; llama.cpp reads "
                        f"this key as int32, so the app will not rewrite it"),
            ))
            continue
        if arr.count > required:
            findings.append(Finding(
                code="rope_sections_too_many",
                summary=f"{key} has {arr.count} entries where llama.cpp allows exactly {required}",
                detail=("dropping entries would need a guess about which section is which, "
                        "so the app will not do it automatically"),
            ))
            continue

        missing = required - arr.count
        needed = missing * 4
        plan = _plan_room(p, layout, needed, key)
        summary = (f"{key} has {arr.count} entries but llama.cpp requires {required} "
                   f"(the last one is always 0)")
        if plan is None:
            findings.append(Finding(
                code="rope_sections_short",
                summary=summary,
                detail=(f"repairing in place needs {needed} spare bytes before the tensor "
                        f"data; this file has {layout.padding} bytes of padding and no "
                        f"provenance string long enough to shorten, so it would have to "
                        f"be re-converted instead"),
                target_key=key,
            ))
            continue
        findings.append(Finding(
            code="rope_sections_short",
            summary=summary,
            detail=(f"the exporter wrote {missing} fewer section(s) than llama.cpp's "
                    f"loader accepts, so loading stops at the hyperparameters"),
            repairable=True,
            fix=f"append {missing} zero(s) to {key} ({plan.describe()})",
            target_key=key,
            plan=plan,
        ))

    # Tensor-name failures only exist in llama.cpp's message, so they are
    # reported when the caller passes that message in.
    text = error_text or ""
    missing = MISSING_TENSOR_RE.search(text)
    unexpected = UNEXPECTED_TENSOR_RE.search(text)
    for match, is_unexpected in ((missing, False), (unexpected, True)):
        if match and not any(f.target_key == match.group(1) for f in findings):
            findings.append(_tensor_finding(p, match.group(1), unexpected=is_unexpected))
    return findings


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------


def _backup_path(path: Path) -> Path:
    return path.with_name(path.name + _BACKUP_SUFFIX)


def _copy_prefix(src: Path, dst: Path, nbytes: int) -> int:
    """Copy the first ``nbytes`` of *src* into *dst*; returns bytes written."""
    written = 0
    with open(src, "rb") as s, open(dst, "wb") as d:
        while written < nbytes:
            chunk = s.read(min(4 * 1024 * 1024, nbytes - written))
            if not chunk:
                break
            d.write(chunk)
            written += len(chunk)
        d.flush()
        os.fsync(d.fileno())
    return written


def _deleted_before(abs_offset: int, deletions: List[Tuple[int, int]]) -> int:
    return sum(end - start for start, end in deletions if end <= abs_offset)


def _rewrite_region(
    region: bytes,
    region_start: int,
    deletions: List[Tuple[int, int]],
    patches: List[Tuple[int, bytes]],
    insertions: List[Tuple[int, bytes]],
    padding: int,
) -> bytes:
    """Rebuild [region_start, data_start): remove, patch, insert.

    Inserting pushes everything after it to the right, so the result is trimmed
    back to the region's original length by eating into the alignment padding at
    the region's tail. That shrink is exactly what keeps ``data_start`` (and so
    every tensor offset) where it was.
    """
    buf = bytearray(region)
    for start, end in sorted(deletions, reverse=True):
        del buf[start - region_start:end - region_start]

    def at(abs_offset: int) -> int:
        return abs_offset - region_start - _deleted_before(abs_offset, deletions)

    for abs_offset, data in patches:
        offset = at(abs_offset)
        if offset < 0 or offset + len(data) > len(buf):
            raise ValueError("the repair region does not cover the bytes it edits")
        buf[offset:offset + len(data)] = data
    for abs_offset, data in insertions:
        offset = at(abs_offset)
        if offset < 0 or offset > len(buf):
            raise ValueError("the repair region does not cover the bytes it edits")
        buf[offset:offset] = data

    grown = len(buf) - len(region)
    if grown > 0:
        if grown > padding:
            raise ValueError(
                f"the repair needs {grown} bytes of alignment padding but only "
                f"{padding} are available")
        del buf[len(region):]                  # the tail of the region is padding
    return bytes(buf)


def repair(path: str | Path, *, dry_run: bool = False) -> RepairReport:
    """Fix repairable metadata in place (or plan it with ``dry_run``)."""
    p = Path(path)
    report = RepairReport(path=str(p))
    if not p.is_file():
        report.message = f"no such file: {path}"
        return report

    layout = scan_layout(p)
    if layout is None:
        report.message = ("this file's GGUF tables could not be read, so nothing "
                          "can be changed safely")
        return report

    findings = diagnose(p)
    report.findings = findings
    target = next((f for f in findings if f.repairable and f.plan), None)
    if target is None:
        report.message = (findings[0].detail if findings
                          else "no known metadata problem was found in this file")
        return report

    arr = layout.arrays[target.target_key]
    plan = target.plan
    assert plan is not None                     # guaranteed by the filter above
    needed = (ROPE_SECTIONS_REQUIRED - arr.count) * 4
    deletions = [plan.deletion] if plan.deletion else []
    patches = [(arr.count_field, struct.pack("<Q", ROPE_SECTIONS_REQUIRED))]
    if plan.patch:
        patches.append(plan.patch)
    insertions = [(arr.data_end, b"\x00" * needed)]
    region_start = min([off for off, _ in patches] +
                       [start for start, _ in deletions] +
                       [off for off, _ in insertions] +
                       [plan.modified_from or arr.count_field])
    region_end = layout.data_start
    if region_end - region_start > MAX_HEADER_BYTES:
        report.message = ("the metadata section is larger than this app will rewrite "
                          f"({(region_end - region_start) / (1024 ** 2):.0f} MB)")
        return report

    report.mode = plan.mode
    report.trimmed_keys = [plan.trim_key] if plan.trim_key else []
    report.backup_path = str(_backup_path(p))

    if dry_run:
        report.message = f"ready to repair: {target.fix}"
        return report

    # 1. Keep the original header (everything before the tensor data) so this is
    #    always reversible, even if the process dies mid-write.
    _copy_prefix(p, _backup_path(p), layout.data_start)

    size_before = layout.file_size
    before_values = _read_int_array(p, arr, arr.count)
    try:
        with open(p, "r+b") as f:
            f.seek(region_start)
            region = f.read(region_end - region_start)
            if len(region) != region_end - region_start:
                raise ValueError("could not read the metadata region")
            new_region = _rewrite_region(region, region_start, deletions, patches,
                                         insertions, layout.padding)
            f.seek(region_start)
            f.write(new_region)
            if len(new_region) < len(region):       # park the freed bytes as padding
                f.write(b"\x00" * (len(region) - len(new_region)))
            f.flush()
            os.fsync(f.fileno())
    except (OSError, ValueError) as e:
        _restore(p, layout.data_start)
        report.message = (f"could not write the repair ({e}) - the original header "
                          f"was put back")
        return report

    # 2. Believe nothing: re-read the file and check the invariants that make
    #    this safe (same size, tensor data still where the offsets say).
    try:
        ok, problem = _verify(p, layout, target.target_key, before_values, size_before)
    except Exception as e:  # noqa: BLE001 - a broken result must not escape
        ok, problem = False, f"unexpected error while checking the result ({e})"
    report.file_size_before = size_before
    report.file_size_after = p.stat().st_size
    if not ok:
        _restore(p, layout.data_start)
        report.message = (f"repair failed its own verification ({problem}) - the "
                          f"original header was put back, the file is unchanged")
        return report

    logger.info("Repaired %s: %s", p.name, target.fix)
    report.applied = True
    report.message = f"repaired: {target.fix}"
    return report


def _read_int_array(path: Path, arr: ArrayMeta, count: int) -> List[int]:
    with open(path, "rb") as f:
        f.seek(arr.data)
        return [struct.unpack("<i", f.read(4))[0] for _ in range(min(count, arr.count))]


def _verify(path: Path, before: GgufLayout, target_key: str,
            before_values: List[int], size_before: int) -> Tuple[bool, str]:
    after = scan_layout(path)
    if after is None:
        return False, "the file could not be re-read"
    if after.file_size != size_before:
        return False, "the file size changed"
    if after.data_start != before.data_start:
        return False, "the tensor data would have moved"
    if after.tensor_count != before.tensor_count:
        return False, "the tensor count changed"
    arr = after.arrays.get(target_key)
    if arr is None:
        return False, f"{target_key} disappeared"
    if arr.count != ROPE_SECTIONS_REQUIRED:
        return False, f"{target_key} still has {arr.count} entries"
    values = _read_int_array(path, arr, ROPE_SECTIONS_REQUIRED)
    if values[:len(before_values)] != before_values:
        return False, "the existing entries changed"
    if any(v != 0 for v in values[len(before_values):]):
        return False, "the appended entries are not zero"
    return True, ""


def _restore(path: Path, data_start: int) -> bool:
    """Put the backed-up header back, leaving the tensor data alone."""
    backup = _backup_path(path)
    if not backup.is_file():
        return False
    try:
        with open(backup, "rb") as src, open(path, "r+b") as dst:
            original = src.read(data_start)
            if len(original) != data_start:
                return False
            dst.seek(0)
            dst.write(original)
            dst.flush()
            os.fsync(dst.fileno())
        return True
    except OSError as e:  # noqa: BLE001 - reported to the caller
        logger.error("Restore failed for %s: %s", path, e)
        return False


def undo(path: str | Path) -> RepairReport:
    """Restore a model's header from its ``.header-backup`` sidecar.

    Deliberately does not need to parse the model first: the whole point is to
    work even when the header is damaged, so the backup's own length decides how
    much is written back.
    """
    p = Path(path)
    backup = _backup_path(p)
    report = RepairReport(path=str(p), backup_path=str(backup))
    if not p.is_file():
        report.message = f"no such file: {path}"
        return report
    if not backup.is_file():
        report.message = (f"no header backup next to this model "
                          f"(expected {backup.name})")
        return report
    try:
        with open(backup, "rb") as src, open(p, "r+b") as dst:
            original = src.read()
            dst.seek(0)
            dst.write(original)
            dst.flush()
            os.fsync(dst.fileno())
    except OSError as e:  # noqa: BLE001 - reported to the caller
        report.message = f"the header backup could not be written back ({e})"
        return report
    report.applied = True
    report.message = f"the original header was restored ({len(original):,} bytes)"
    return report
