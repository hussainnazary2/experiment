from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelProfile:
    max_tokens: int
    max_steps: int
    temperature: float
    n_ctx_target: int
    n_batch: int
    n_threads: int
    n_keep: int
    use_mmap: bool
    flash_attn: bool
    json_retries: int


# Per-family agent profiles (universal mode). Sampling stays conservative
# for reliable structured output on every family; resource knobs (context,
# token budget, step budget) scale with what the family's models typically
# need. Unknown files get the GENERIC profile.
_FAMILY_PROFILES: dict[str, dict] = {
    # Starter model: tuned in v2.3, unchanged.
    "gemma4": dict(
        max_tokens=4096, max_steps=20, temperature=0.1,
        n_ctx_target=8192,
    ),
    "gemma3": dict(
        max_tokens=4096, max_steps=20, temperature=0.1,
        n_ctx_target=8192,
    ),
    "llama": dict(
        max_tokens=4096, max_steps=20, temperature=0.1,
        n_ctx_target=8192,
    ),
    "qwen": dict(
        max_tokens=4096, max_steps=22, temperature=0.1,
        n_ctx_target=8192,
    ),
    "mistral": dict(
        max_tokens=4096, max_steps=20, temperature=0.1,
        n_ctx_target=8192,
    ),
    "phi": dict(
        max_tokens=3072, max_steps=16, temperature=0.1,
        n_ctx_target=8192,
    ),
    # Reasoning models: longer chains need room.
    "deepseek": dict(
        max_tokens=6144, max_steps=24, temperature=0.15,
        n_ctx_target=16384,
    ),
    # Tiny models: keep everything small so they stay responsive.
    "tinyllama": dict(
        max_tokens=2048, max_steps=12, temperature=0.15,
        n_ctx_target=4096,
    ),
    "smollm": dict(
        max_tokens=2048, max_steps=12, temperature=0.15,
        n_ctx_target=4096,
    ),
}

_GENERIC = dict(
    max_tokens=4096, max_steps=16, temperature=0.1, n_ctx_target=4096,
)


def _detect_family(model_path: str | Path) -> str:
    """Family id for a model file (router-side detector, metadata-only)."""
    try:
        from ggufloader.core.llm.model_profiles import (
            read_gguf_general_metadata as _read_meta,
            detect_family as _detect,
        )
        meta = _read_meta(model_path)
        profile, _via = _detect(meta, model_path)
        return profile.get("family", "generic")
    except Exception:  # noqa: BLE001 - profile selection must never fail
        return "generic"


def _clamp_ctx(n_ctx_train: int, target: int) -> int:
    if n_ctx_train <= 0:
        return target
    return min(target, n_ctx_train)


def _make_profile(family: str, n_ctx_train: int) -> ModelProfile:
    """Agent profile for a model family (sampling tuned for JSON output)."""
    cfg = _FAMILY_PROFILES.get(family, _GENERIC)
    n_ctx = _clamp_ctx(n_ctx_train, cfg["n_ctx_target"])
    return ModelProfile(
        max_tokens=cfg["max_tokens"],
        max_steps=cfg["max_steps"],
        temperature=cfg["temperature"],
        n_ctx_target=n_ctx,
        n_batch=512,
        n_threads=8,
        n_keep=512,
        use_mmap=True,
        flash_attn=True,
        json_retries=2,
    )


def get_profile(model_path: str | Path, n_ctx_train: int = 0) -> ModelProfile:
    """Agent profile for a model path, detected per family.

    ``n_ctx_train`` (from GGUF metadata) clamps the context target when the
    file's trained context is smaller than the family's agent target.
    """
    family = _detect_family(model_path)
    return _make_profile(family, n_ctx_train)
