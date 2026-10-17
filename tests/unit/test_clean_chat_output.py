"""Chat output must never reach the UI as raw special-token soup.

The failure these tests pin down: a GGUF that ships without
``tokenizer.chat_template`` is rendered by llama-cpp-python through its
llama-2 fallback formatter, and a model that was never trained on
``[INST] ... [/INST]`` answers with ``<s>hi</s> <s>world</s>`` repeated until
the token budget runs out - with no stop strings, so the UI shows all of it.
"""

from ggufloader.core.agent.token_cleaner import (
    SpecialTokenScrubber,
    get_cleaner,
    strip_special_tokens,
)
from ggufloader.core.defaults import STOP_TOKENS_UNIFIED
from ggufloader.core.llm.model_backend import ModelBackend, merge_stop_tokens
from ggufloader.core.llm.prompt_builder import render_transcript


class FakeLlama:
    """The two llama-cpp entry points chat uses, plus its metadata view."""

    def __init__(self, chunks=(), *, metadata=None, chat_format="chat_template.default"):
        self.metadata = (
            {"tokenizer.chat_template": "jinja here"} if metadata is None else metadata
        )
        self.chat_format = chat_format
        self.chat_calls = []
        self.completion_calls = []
        self._chunks = list(chunks)

    def create_chat_completion(self, **kwargs):
        self.chat_calls.append(kwargs)
        return iter(self._chunks)

    def create_completion(self, **kwargs):
        self.completion_calls.append(kwargs)
        return iter(self._chunks)

    def __call__(self, prompt, **kwargs):      # raw completion path
        self.completion_calls.append({"prompt": prompt, **kwargs})
        return iter({"choices": [{"text": c}]} for c in self._chunks)


def _backend(fake):
    b = ModelBackend("fake.gguf")
    b._llama = fake
    return b


MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "hi"},
]


# ---------------------------------------------------------------------------
# Which prompt path a model gets
# ---------------------------------------------------------------------------

def test_template_less_model_answers_from_a_transcript():
    # lfm2.5-style file: real architecture, no embedded chat template, and
    # llama-cpp-python's llama-2 fallback selected - the user's exact case.
    fake = FakeLlama(
        metadata={"general.architecture": "lfm2"}, chat_format="llama-2")
    list(_backend(fake).chat_stream(MESSAGES))
    assert fake.chat_calls == []               # llama-2 format was NOT used
    call = fake.completion_calls[0]
    assert call["prompt"].startswith("Be brief.")
    assert "User: hi" in call["prompt"]
    assert call["prompt"].rstrip().endswith("Assistant:")
    # Transcript path adds newline-anchored turn cues to the unified set.
    assert call["stop"] == list(STOP_TOKENS_UNIFIED) + ["\nUser:", "\nAssistant:"]


def test_transcript_path_stops_at_the_next_turn_cue():
    # A transcript has no turn token, so a model that rambles into the next
    # turn (and the leading space it answers "Assistant:" with) is cut here.
    fake = FakeLlama(
        chunks=[
            {"choices": [{"text": " First answer."}]},
            {"choices": [{"text": "\n\nAssistant:"}]},
            {"choices": [{"text": " rambling on"}]},
        ],
        metadata={"general.architecture": "lfm2"}, chat_format="llama-2")
    assert "".join(_backend(fake).chat_stream(MESSAGES)) == "First answer."
    stops = fake.completion_calls[0]["stop"]
    assert "\nUser:" in stops and "\nAssistant:" in stops


def test_model_with_embedded_template_keeps_chat_completion():
    fake = FakeLlama()
    list(_backend(fake).chat_stream(MESSAGES))
    assert fake.completion_calls == []
    assert fake.chat_calls[0]["messages"] is MESSAGES


def test_llama_family_keeps_the_llama2_formatter():
    # For llama-2 family models the fallback really is the native format.
    fake = FakeLlama(metadata={"general.architecture": "llama"}, chat_format="llama-2")
    list(_backend(fake).chat_stream(MESSAGES))
    assert fake.chat_calls and not fake.completion_calls


def test_llama2_format_would_have_been_the_bug():
    # Same file, but the GGUF *does* ship a template: whatever llama-cpp
    # picked stays, including an explicitly requested llama-2 format.
    fake = FakeLlama(chat_format="llama-2")
    list(_backend(fake).chat_stream(MESSAGES))
    assert fake.chat_calls and not fake.completion_calls


# ---------------------------------------------------------------------------
# Special tokens never reach the output
# ---------------------------------------------------------------------------

def test_raw_special_tokens_are_stripped_from_the_stream():
    fake = FakeLlama(chunks=[
        {"choices": [{"delta": {"content": "<s>Hi"}}]},
        {"choices": [{"delta": {"content": " there"}}]},
        {"choices": [{"delta": {"content": "</s>"}}]},   # cut by the stop set
        {"choices": [{"delta": {"content": "</s>"}}]},
    ])
    out = "".join(_backend(fake).chat_stream(MESSAGES))
    assert out == "Hi there"
    assert "<s>" not in out and "</s>" not in out


def test_marker_split_across_chunks_is_still_removed():
    scrubber = SpecialTokenScrubber()
    out = "".join(scrubber.feed(p) for p in ("<sta", "rt_of_turn>ok ", "<|chann", "el|>done"))
    out += scrubber.flush()
    assert out == "ok done"


def test_ordinary_text_with_angles_is_untouched():
    scrubber = SpecialTokenScrubber()
    out = "".join(scrubber.feed(p) for p in ("x < ", "y and <div>z</div>"))
    out += scrubber.flush()
    assert out == "x < y and <div>z</div>"


def test_incomplete_marker_at_end_of_stream_is_dropped_not_rendered():
    scrubber = SpecialTokenScrubber()
    assert scrubber.feed("</s") == ""
    assert scrubber.flush() == ""


def test_generation_stops_at_the_first_eos_rather_than_looping():
    # The regression itself: `hi</s></s></s>...` to max_tokens.
    fake = FakeLlama(chunks=[
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [{"delta": {"content": "</s>"}}]},
        {"choices": [{"delta": {"content": "</s>"}}]},
    ])
    assert "".join(_backend(fake).chat_stream(MESSAGES)) == "hi"


def test_generate_stream_also_stops_and_scrubs():
    fake = FakeLlama(chunks=["Answer", "</s>", "</s>"])
    out = "".join(_backend(fake).generate_stream("User: hi\n\nAssistant:"))
    assert out == "Answer"
    assert fake.completion_calls[0]["stop"] == list(STOP_TOKENS_UNIFIED)


def test_no_text_traps_in_the_stop_set():
    stops = merge_stop_tokens(None)
    assert stops == list(STOP_TOKENS_UNIFIED)
    assert "user:" not in stops and "###" not in stops


# ---------------------------------------------------------------------------
# Transcript rendering + cleaners
# ---------------------------------------------------------------------------

def test_render_transcript_labels_roles_and_leads_the_model_to_answer():
    text = render_transcript(MESSAGES + [
        {"role": "user", "content": "and now?"},
        {"role": "assistant", "content": "this"},
    ])
    assert text.startswith("Be brief.")
    assert "User: hi" in text and "User: and now?" in text
    assert "Assistant: this" in text and text.endswith("Assistant:")


def test_render_transcript_skips_empty_messages():
    text = render_transcript([{"role": "user", "content": "only"}])
    assert text == "User: only\n\nAssistant:"


def test_cleaners_remove_bos_eos_too():
    for architecture in (None, "gemma4", "lfm2"):
        cleaned = get_cleaner(architecture).clean("</s>answer<s>")
        assert "<s>" not in cleaned and "</s>" not in cleaned
    assert strip_special_tokens("<|eot_id|>ok<start_of_turn>") == "ok"


def test_cleaners_keep_role_names_attached_to_channel_markers():
    # The generic special-token strip runs *after* the role-carrying patterns,
    # so a Gemma stream (`<|channel|>thought<|message|>answer`) still loses the
    # channel name with its marker instead of leaving a stray "thought".
    stream = "<|channel|>thought<|message|>the real answer"
    for architecture in (None, "gemma4"):
        cleaned = get_cleaner(architecture).clean(stream)
        assert "thought" not in cleaned
        assert "the real answer" in cleaned
