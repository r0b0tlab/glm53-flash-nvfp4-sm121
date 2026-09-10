#!/usr/bin/env python3
"""Acceptance matrix on a live DFlash2 serve.

For each prompt type (code / prose / reasoning), measures wall tok/s and the
spec-decode acceptance delta (pooled ratio, mean accepted length, per-position
acceptance) from /metrics counters.

Usage: acceptance_matrix.py [base_url] [model] [max_tokens]
"""
import json
import sys
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "glm-5.3-flash-nvfp4-sm121"
MT = int(sys.argv[3]) if len(sys.argv) > 3 else 512

PROMPTS = [
    ("code", "Write a Python function that merges two sorted lists into one "
             "sorted list. Include a docstring and type hints."),
    ("prose", "Write a detailed technical explanation of rotary position embeddings."),
    ("reasoning", "A bat and a ball cost $1.10. The bat costs $1.00 more than the "
                  "ball. How much does the ball cost? Show your reasoning."),
]


def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        txt = r.read().decode()
    out = {"per_pos": {}}
    for line in txt.splitlines():
        if not line.startswith("vllm:spec_decode") or "{" not in line:
            continue
        name = line.split("{")[0].split(":")[-1]
        if "per_pos" in line and "total" in name:
            pos = int(line.split('position="')[1].split('"')[0])
            out["per_pos"][pos] = float(line.split()[-1])
        elif name.endswith("_total") and "per_pos" not in name:
            out[name] = float(line.split()[-1])
    return out


def gen(prompt):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": MT,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.load(r)
    dt = time.time() - t0
    return j["usage"]["completion_tokens"], dt


rows = []
for name, prompt in PROMPTS:
    m0 = metrics()
    ct, dt = gen(prompt)
    m1 = metrics()
    acc = m1.get("num_accepted_tokens_total", 0) - m0.get("num_accepted_tokens_total", 0)
    dr = m1.get("num_draft_tokens_total", 0) - m0.get("num_draft_tokens_total", 0)
    de = m1.get("num_drafts_total", 0) - m0.get("num_drafts_total", 0)
    per_pos = {
        p: round((m1["per_pos"].get(p, 0) - m0["per_pos"].get(p, 0)) / de, 3)
        if de else None for p in sorted(m1["per_pos"])
    }
    row = {
        "case": name, "tok": ct, "wall_s": round(dt, 1),
        "tok_s": round(ct / dt, 2),
        "pooled_accept": round(acc / dr, 4) if dr else None,
        "mean_accepted_len": round(acc / de, 2) if de else None,
        "per_pos_accept": per_pos,
    }
    rows.append(row)
    print(json.dumps(row))

print(json.dumps({"summary": rows}))
