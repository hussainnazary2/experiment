"""End-to-end check of the chatbot path with real GGUF weights.

Usage:
    python scripts/verify_playground_chat.py MODEL.gguf [MODEL2.gguf ...]

Loads each GGUF through `POST /api/playground/model/load` and streams a reply
from `POST /api/playground/chat/stream` exactly like the Chatbot panel does,
reporting every event it receives (token / stats / done / error) plus the
load diagnostics for any model that refuses to load.

Run it with a wall-clock timeout: before 2.4.1 the stream deadlocked after its
first token (the live `stats` event re-entered the model lock), so this script
would print the load result and then never finish.
"""

import os
import sys
import time

os.environ.setdefault("GGUFLOADER_SKIP_AUTOLOAD", "1")
sys.path.insert(0, ".")

from fastapi.testclient import TestClient

DEFAULT_MODELS = (
    "/tmp/gguf/LFM2-350M-Q4_K_M.gguf",
    "/tmp/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf",
)


def stream_once(client, path):
    t = time.perf_counter()
    r = client.post("/api/playground/model/load",
                    json={"path": path, "use_gpu": False})
    print(f"[load] {os.path.basename(path)} -> HTTP {r.status_code} "
          f"in {time.perf_counter() - t:.1f}s :: {r.text[:400]}", flush=True)
    if r.status_code != 200:
        return

    counts = {"token": 0, "stats": 0, "done": 0, "error": 0}
    text, last_stats, errors = "", {}, []
    event = None
    t = time.perf_counter()
    with client.stream(
        "POST", "/api/playground/chat/stream",
        json={"message": "Say hi in three words.", "max_tokens": 24, "temperature": 0.2},
    ) as s:
        for line in s.iter_lines():
            if line.startswith("event: "):
                event = line[7:].strip()
                counts[event] = counts.get(event, 0) + 1
            elif line.startswith("data: ") and event:
                import json as _json
                data = _json.loads(line[6:])
                if event == "token":
                    text += data.get("token", "")
                elif event == "stats":
                    last_stats = data
                elif event == "done":
                    last_stats = data
                    text = data.get("content", text)
                elif event == "error":
                    errors.append(data.get("message"))
    elapsed = time.perf_counter() - t
    print(f"[chat] {len(text)} chars, events={counts}, {elapsed:.1f}s", flush=True)
    print(f"[chat] reply={text[:120]!r}", flush=True)
    print(f"[chat] stats: tok/s={last_stats.get('tokens_per_second')} "
          f"generated={last_stats.get('tokens_generated')} "
          f"total={last_stats.get('total_tokens')} ttft={last_stats.get('ttft_ms')}ms",
          flush=True)
    if errors:
        print(f"[chat] ERRORS: {errors}", flush=True)


def main():
    paths = tuple(sys.argv[1:]) or DEFAULT_MODELS

    from ggufloader.core.llm.gguf_meta import check_integrity, read
    for p in paths:
        meta = read(p)
        print(f"[integrity] {os.path.basename(p)}: arch={meta.get('architecture')!r} "
              f"ctx={meta.get('qwen2.context_length') or meta.get('lfm2.context_length')} "
              f"-> {check_integrity(p)}", flush=True)

    from ggufloader.api.app import create_app
    with TestClient(create_app()) as client:
        for path in paths:
            try:
                stream_once(client, path)
            except Exception as e:  # noqa: BLE001
                print(f"[chat] {os.path.basename(path)} FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
