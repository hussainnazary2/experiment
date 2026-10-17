# Session Progress Log

## Session: 2026-10-16 - Local model speed chatbot

**Goal (after direction change):** a plain chatbot that shows everything a
viewer needs - the tokens streaming, tokens/sec, and tokens generated. No
agent, no tools, and **no in-app recording** (the user records externally).

- `ggufloader/api/routes/playground.py` - `/api/playground/models`,
  `/model/load` (ungated: loads any GGUF), `/model`, `/chat/stream`
  (SSE `token` / `stats` / `done` / `error`) and `/chat`. `stats` is pushed
  ~4x/second with the running token count and decode tokens/sec; token counts
  come from the model's own tokenizer.
- `frontend/src/components/chat/PlaygroundChat.tsx` - the chatbot center view:
  model picker + Load, streaming replies, and four live readouts (tokens/sec,
  tokens generated, first token, elapsed).
- `frontend/src/components/layout/AppLayout.tsx` + `stores/uiStore.ts` -
  Chatbot is the default center view, with a Chatbot / Agent switch.

**Removed:** `useScreenRecorder.ts`, the recording HUD, and the Electron
`save-recording` IPC.

**Left in place but unwired:** the multi-model suite
(`ggufloader/core/testing/`) and its `/api/benchmark/suite/*` routes.

**Verified:** `pytest tests/unit` - 611 passed, including 12 new playground
tests (the only failures are the pre-existing `llama-cpp-python`/`PySide6`
dependent ones, identical on a clean clone); `tsc -b` + `vite build` clean;
`electron` `tsc --noEmit` clean; and a TestClient run against the real
`create_app()` exercised `/api/playground/model`, `/models`, `/chat/stream`
and `/chat` end to end.

**Next:** real-model validation on a machine with llama.cpp + GGUFs.

## Current State

**Last Updated:** 2026-08-27
**Session ID:** harness-setup
**Active Feature:** feat-011 - Agent Harness: Self-Verification

## Status

### What's Done

- [x] feat-001: Project Setup & Build
- [x] feat-002: Chat Panel & Streaming
- [x] feat-003: Model Loading & Parameters
- [x] feat-004: Agent Mode (LangGraph)
- [x] feat-005: LocalDocs RAG
- [x] feat-006: HuggingFace Model Catalog
- [x] feat-007: Model Family Auto-Detection (30+ families)
- [x] feat-008: Advanced Settings Dialog
- [x] feat-009: Agent Harness: Workspace Context (AGENTS.md, git, prefix cache)
- [x] feat-010: Agent Harness: Approval & Clipping (approval workflow, output clipping, working memory)

### What's In Progress

- [ ] feat-011: Agent Harness: Self-Verification
  - Details: Agent runs typecheck/tests before declaring success
  - Blockers: none

### What's Next

1. feat-011: Self-Verification hooks
2. feat-012: Subagent Delegation
3. feat-013: Context Compaction

## Blockers / Risks

- [ ] Qwen3.5 GGUF compatibility: Ollama-quantized models have metadata fields not yet in upstream llama.cpp

## Decisions Made

- **Model family detection via JSON**: Uses `model_families.json` with 30+ profiles instead of fragile template parsing
- **Approval workflow in AgentEngine**: Added `on_approval` callback to match GraphAgent's LangGraph interrupt pattern
- **Output clipping by age**: Recent results (last 3) get 2000 chars; older get 400 chars

## Files Modified This Session

- `ggufloader/core/agent/agent_engine.py` - Added approval workflow, output clipping, working memory
- `ggufloader/core/agent/workspace_context.py` - Added WorkingMemory class
- `ggufloader/core/agent/graph_agent.py` - Wired WorkingMemory import
- `ggufloader/core/agent/__init__.py` - Exported WorkingMemory, ApprovalCallback

## Evidence of Completion

- [x] Tests pass: `python -m pytest tests/ -x -q` (190+ tests)
- [x] Type check: `python -m py_compile` on all modified files

## Notes for Next Session

- Next feature to implement: feat-011 (Self-Verification)
- The comparison doc at `docs/agent-harness-deep-comparison.md` has the full gap analysis
- Harness validation score: currently 28/100, bottleneck is state (now being fixed)
