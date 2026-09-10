#!/usr/bin/env python3
"""Aggregate throughput vs concurrency for a single-GB10 serve.
Usage: bench_concurrency.py TAG [levels] [max_tokens]"""
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

TAG = sys.argv[1] if len(sys.argv) > 1 else "base"
LEVELS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else "1,2,4,8,16,32".split(","))]
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 256
BASE = __import__("os").environ.get("GLM53_BASE", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = __import__("os").environ.get("GLM53_MODEL", "glm-5.3-flash-nvfp4-sm121")
PROMPT = "Write a detailed technical explanation of rotary position embeddings."

results = []
for c in LEVELS:
    lock = threading.Lock()
    toks = [0] * c
    errs = [None] * c

    def worker(i):
        body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": MAXTOK,
                           "messages": [{"role": "user", "content": PROMPT}]}).encode()
        req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                res = json.load(r)
            with lock:
                toks[i] = res["usage"]["completion_tokens"]
        except Exception as e:
            with lock:
                errs[i] = str(e)[:120]

    t0 = time.time()
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(c)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    total = sum(toks)
    agg = total / wall if wall > 0 else 0.0
    ok = all(e is None for e in errs)
    results.append({"concurrency": c, "total_tokens": total, "wall_s": wall,
                    "aggregate_tok_s": agg, "per_stream_tok_s": agg / c, "ok": ok,
                    "errors": [e for e in errs if e][:2]})
    print(f"c={c:2d}  aggregate={agg:6.2f} tok/s  per-stream={agg/c:5.2f}  wall={wall:6.1f}s  ok={ok}", flush=True)

p = Path.home() / "ling30vl-nvfp4/evidence/perf"
p.mkdir(parents=True, exist_ok=True)
json.dump({"tag": TAG, "max_tokens": MAXTOK, "levels": results},
          open(p / f"{TAG}_concurrency.json", "w"), indent=1)
print("CONCURRENCY_DONE", TAG)
