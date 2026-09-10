#!/usr/bin/env python3
"""Deep-decode probe: the P2 persistent_topk gate must keep the engine alive
past ~24K context. Pre-fix this is a deterministic EngineDeadError on GB10.

Usage: probe_deep_decode.py [base_url] [model] [target_tokens] [max_tokens]
Exit 0 only if the request succeeds AND /health still answers.
"""
import json
import sys
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "glm-5.3-flash-nvfp4-sm121"
TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 26000
MAX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 512

# ~3.5 chars/token English filler; repeat to target.
UNIT = ("The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. ")
prompt = (UNIT * (TARGET * 4 // len(UNIT) + 1))[: TARGET * 4]
prompt += "\n\nReply with exactly: DEEP_DECODE_OK"

body = json.dumps({
    "model": MODEL, "temperature": 0, "max_tokens": MAX_TOKENS,
    "messages": [{"role": "user", "content": prompt}],
}).encode()
req = urllib.request.Request(
    BASE + "/v1/chat/completions", data=body,
    headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        res = json.load(r)
except Exception as e:  # noqa: BLE001
    print(json.dumps({"verdict": "FAIL", "stage": "request", "error": repr(e)}))
    sys.exit(1)
dt = time.time() - t0
content = (res["choices"][0]["message"].get("content") or "")
usage = res.get("usage", {})
try:
    with urllib.request.urlopen(BASE + "/health", timeout=30) as r:
        healthy = r.status == 200
except Exception as e:  # noqa: BLE001
    healthy = False
    print(f"health error: {e!r}")
out = {
    "verdict": "PASS" if (
        healthy and usage.get("completion_tokens") and "DEEP_DECODE_OK" in content
    ) else ("PASS_ENGINE_ALIVE" if (
        healthy and usage.get("completion_tokens")
    ) else "FAIL"),
    "prompt_tokens": usage.get("prompt_tokens"),
    "completion_tokens": usage.get("completion_tokens"),
    "wall_s": round(dt, 1), "healthy_after": healthy,
    "content_head": content[:120],
}
print(json.dumps(out, indent=1))
sys.exit(0 if out["verdict"].startswith("PASS") else 1)
