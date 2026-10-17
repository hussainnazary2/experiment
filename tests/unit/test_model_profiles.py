"""Universal model tests: any GGUF loads; families are detected per file.

Multi-family detection is active again (model_families.json). The
agent-side profile is picked per family, and the router-side detector
recognizes gemma4 / llama / qwen / mistral / ... files (everything else
resolves to generic with conservative defaults).
"""

from ggufloader.core.agent.model_profiles import get_profile


def test_gemma4_profile_has_safe_defaults():
    p = get_profile("gemma-4-12b-it-Q4_K_M.gguf", n_ctx_train=8192)
    assert p.max_tokens <= 4096
    assert p.max_steps >= 16
    assert p.temperature <= 0.2
    assert p.n_ctx_target == 8192
    assert p.n_batch >= 256
    assert p.flash_attn is True


def test_profiles_are_detected_per_family():
    """Each family gets its own agent profile (context/token budgets)."""
    p = get_profile("qwen2.5-7b-instruct-q4_k_m.gguf", n_ctx_train=32768)
    assert p.n_ctx_target == 8192  # qwen agent target, not the file's 32k
    t = get_profile("tinyllama-1.1b-chat-q4_k_m.gguf")
    assert t.n_ctx_target == 4096  # tiny models get a small window
    assert t.max_tokens <= 2048


def test_unknown_model_gets_conservative_generic_profile():
    """Mystery files fall back to generic (small ctx, bounded budget)."""
    p = get_profile("mystery-99b-F16.gguf")
    assert p.max_tokens == 4096
    assert p.n_ctx_target == 4096
    assert p.temperature <= 0.2


def test_ctx_clamped_by_trained_context():
    p = get_profile("gemma-4-12b-it-Q4_K_M.gguf", n_ctx_train=4096)
    assert p.n_ctx_target == 4096  # clamped to the file's trained context
    p2 = get_profile("gemma-4-12b-it-Q4_K_M.gguf")  # no ctx info -> target
    assert p2.n_ctx_target == 8192


# ---------------------------------------------------------------------------
# Router-side detection: families come from model_families.json
# ---------------------------------------------------------------------------

def test_gemma4_arch_resolves_to_pinned_family():
    from ggufloader.core.llm.model_profiles import detect_family
    prof, via = detect_family(
        {"architecture": "gemma4", "name": "gemma-4-12b-it"},
        "gemma-4-12b-it-Q4_K_M.gguf")
    assert prof["family"] == "gemma4", prof["family"]
    assert "gemma4" in via or "arch" in via, via


def test_gemma4_name_match_without_arch():
    from ggufloader.core.llm.model_profiles import detect_family
    prof, via = detect_family({}, "gemma-4-12b-it-Q4_K_M.gguf")
    assert prof["family"] == "gemma4", prof["family"]


def test_known_families_resolve_to_their_own_tuning():
    """llama / gemma3 / qwen files resolve to their own family profiles."""
    from ggufloader.core.llm.model_profiles import detect_family
    cases = [
        ({"architecture": "llama", "name": "Meta Llama 3 8B Instruct"}, "llama"),
        ({"architecture": "gemma3", "name": "gemma-3-8b-it"}, "gemma3"),
        ({"architecture": "qwen2", "name": "Qwen2.5-7B-Instruct"}, "qwen"),
        ({"architecture": "mistral", "name": "Mistral-7B-Instruct"}, "mistral"),
    ]
    for meta, expected in cases:
        prof, via = detect_family(meta, "model.gguf")
        assert prof["family"] == expected, (meta, prof["family"])
        assert via  # non-empty explanation


def test_unknown_file_never_crashes_returns_generic():
    from ggufloader.core.llm.model_profiles import detect_family
    prof, via = detect_family({}, "mystery-model-9b-q8_0.gguf")
    assert prof["family"] == "generic"
    assert via  # non-empty explanation


# ---------------------------------------------------------------------------
# LFM family (Liquid AI LFM2 / LFM2.5) - ChatML-like template + official
# sampling (temp 0.2, top_k 80, repeat 1.05)
# ---------------------------------------------------------------------------

def test_lfm_family_detected_from_arch_with_chatml():
    from ggufloader.core.llm.model_profiles import detect_family
    prof, via = detect_family(
        {"architecture": "lfm2moe"}, "lfm2.5-8b-a1b-q4.gguf")
    assert prof["family"] == "lfm", prof["family"]
    assert prof["chat_format"] == "chatml"
    assert prof["temperature"] == 0.2
    assert prof["top_k"] == 80
    assert prof["repeat_penalty"] == 1.05
    assert prof["supports_system_prompt"] is True


def test_lfm_family_detected_from_filename_without_arch():
    from ggufloader.core.llm.model_profiles import detect_family
    prof, via = detect_family({}, "/downloads/lfm2.5-8b.gguf")
    assert prof["family"] == "lfm"
    assert prof["chat_format"] == "chatml"


def test_lfm_not_confused_with_llama():
    """'lfm2' contains no 'llama' substring - make sure ordering keeps it."""
    from ggufloader.core.llm.model_profiles import detect_family
    prof, _ = detect_family({"architecture": "lfm2"}, "lfm2-8b.gguf")
    assert prof["family"] == "lfm"
