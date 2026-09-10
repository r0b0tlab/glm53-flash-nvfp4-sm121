#!/usr/bin/env python3
"""Decode throughput for a serve candidate. Usage: bench_decode.py TAG [runs] [max_tokens]."""
import json
import sys
import time
import urllib.request
from pathlib import Path

TAG = sys.argv[1] if len(sys.argv) > 1 else "base"
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 512
BASE = __import__("os").environ.get("GLM53_BASE", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = __import__("os").environ.get("GLM53_MODEL", "glm-5.3-flash-nvfp4-sm121")
PROMPT = "Write a detailed technical explanation of rotary position embeddings."

rows = []
for i in range(RUNS):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": MAXTOK,
                       "messages": [{"role": "user", "content": PROMPT}]}).encode()
    req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        res = json.load(r)
    dt = time.time() - t0
    ct = res["usage"]["completion_tokens"]
    rows.append({"run": i, "completion_tokens": ct, "wall_s": dt, "tok_s": ct / dt})
    print(f"run {i}: {ct} tok / {dt:.2f}s = {ct/dt:.2f} tok/s", flush=True)

out = {"tag": TAG, "max_tokens": MAXTOK, "runs": rows,
       "mean_tok_s": sum(r["tok_s"] for r in rows) / len(rows)}
p = Path.home() / "ling30vl-nvfp4/evidence/perf"
p.mkdir(parents=True, exist_ok=True)
json.dump(out, open(p / f"{TAG}.json", "w"), indent=1)
print("MEAN", round(out["mean_tok_s"], 2), "tok/s")
