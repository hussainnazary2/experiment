from ggufloader.core.llm.model_backend import ModelBackend


def test_model_backend_accepts_tuning_params():
    backend = ModelBackend(
        model_path="dummy.gguf",
        n_ctx=8192,
        n_batch=512,
        n_threads=8,
        n_keep=512,
        flash_attn=True,
    )
    assert backend._n_ctx == 8192
    assert backend._n_batch == 512
    assert backend._n_threads == 8
    assert backend._n_keep == 512
    assert backend._flash_attn is True


def test_model_backend_defaults_are_safe_for_8gb_vram():
    backend = ModelBackend(model_path="dummy.gguf")
    assert backend._n_ctx == 8192
    assert backend._n_batch in (256, 512)
    assert backend._n_keep >= 128

# ---------------------------------------------------------------------------
# Chat-format hint for template-less GGUFs (the LFM2.5 <s>-echo fix)
# ---------------------------------------------------------------------------


class TestChatFormatHint:
    """A GGUF without tokenizer.chat_template must not silently fall back
    to llama-cpp-python's llama-2 format (<s>[INST]...) when its family
    knows better: an LFM2.5 file rendered that way degenerated into
    echoing ``<s>hi</s> <s>world</s>`` loops. The family's chat_format
    (model_families.json) is passed to Llama() instead.
    """

    def _load_with_meta(self, monkeypatch, meta, path="/models/lfm2.5-8b.gguf"):
        import ggufloader.core.llm.model_backend as mb
        import ggufloader.core.llm.model_profiles as mp

        recorded = {}

        class FakeLlama:
            def __init__(self, **kwargs):
                recorded.update(kwargs)

        monkeypatch.setattr(mb, "Llama", FakeLlama)
        monkeypatch.setattr(
            mp, "read_gguf_general_metadata", lambda p, **k: dict(meta))
        backend = ModelBackend(model_path=path)
        backend.load()
        return recorded

    def test_template_less_lfm_gets_chatml(self, monkeypatch):
        recorded = self._load_with_meta(monkeypatch, {"architecture": "lfm2moe"})
        assert recorded["chat_format"] == "chatml"

    def test_template_less_lfm_by_filename_gets_chatml(self, monkeypatch):
        recorded = self._load_with_meta(monkeypatch, {})
        assert recorded["chat_format"] == "chatml"  # "lfm2.5" in the path

    def test_gguf_with_embedded_template_untouched(self, monkeypatch):
        recorded = self._load_with_meta(
            monkeypatch,
            {"architecture": "lfm2moe", "chat_template": "<|im_start|>..."},
        )
        assert "chat_format" not in recorded  # embedded template wins

    def test_unknown_family_keeps_default_behavior(self, monkeypatch):
        recorded = self._load_with_meta(
            monkeypatch, {"architecture": "mystery99"},
            path="/models/mystery-99b-f16.gguf",
        )
        assert "chat_format" not in recorded
