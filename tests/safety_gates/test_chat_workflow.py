"""HIGH Gate: Chat Workflow - 95%+ pass rate."""
import pytest
from ggufloader.core.llm.model_profiles import FAMILY_PROFILES, _template_supports_system
from ggufloader.core.llm.prompt_builder import PromptBuilder

def test_family_profiles_loaded():
    assert len(FAMILY_PROFILES) > 0

def test_family_has_required_keys():
    for profile in FAMILY_PROFILES:
        assert "family" in profile
        assert "archs" in profile
        assert isinstance(profile["archs"], tuple)

def test_multi_family_profiles_loaded():
    # Universal mode: real families shipped in model_families.json.
    families = {p["family"] for p in FAMILY_PROFILES}
    assert "gemma4" in families
    assert {"llama", "qwen", "mistral", "gemma3"} <= families
    gemma4 = next(p for p in FAMILY_PROFILES if p["family"] == "gemma4")
    assert gemma4["supports_system_prompt"] is True
    # Every family carries anti-degeneration guidance for the agent path.
    for p in FAMILY_PROFILES:
        assert "never repeat yourself" in p.get("system_prompt", "").lower()

def test_prompt_builder_produces_messages():
    builder = PromptBuilder()
    msgs = builder.build_messages([], "hello", system_prompt="test")
    assert isinstance(msgs, list)
    assert len(msgs) >= 1
