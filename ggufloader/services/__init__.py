"""
services - application layer bridging the UI to the pure core layer.

- ModelService: background model loading/unloading (PySide6, legacy Qt UI).
- ChatService: streaming chat generation (PySide6, legacy Qt UI).
- AgentService: background agent turns (PySide6, legacy Qt UI).
- HFService: HuggingFace GGUF discovery + download (pure Python, REST).

The Qt services are only importable when PySide6 is installed (the React
frontend talks to the FastAPI backend instead, so server-only environments
do not need them). HFService is pure Python and always available.
"""

from .hf_service import HFService

__all__ = ["HFService"]

try:  # legacy Qt services — optional since the React UI replaced them
    from .model_service import ModelService  # noqa: F401
    from .chat_service import ChatService  # noqa: F401
    from .agent_service import AgentService  # noqa: F401

    __all__ += ["ModelService", "ChatService", "AgentService"]
except ImportError:  # PySide6 not installed (server-only / headless)
    pass
