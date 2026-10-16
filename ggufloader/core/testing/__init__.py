"""Model Testing Studio - run a suite of GGUF models and measure real CPU speed.

See ``runner.py`` for the engine and
``docs/superpowers/specs/2026-10-16-model-testing-studio-design.md`` for the
design.
"""

from ggufloader.core.testing.runner import (
    ModelResult,
    PromptResult,
    SuitePrompt,
    SuiteRun,
    SuiteRunner,
    SuiteStore,
    SUITE_PROMPTS,
    export_csv,
    export_markdown,
)

__all__ = [
    "ModelResult",
    "PromptResult",
    "SuitePrompt",
    "SuiteRun",
    "SuiteRunner",
    "SuiteStore",
    "SUITE_PROMPTS",
    "export_csv",
    "export_markdown",
]
