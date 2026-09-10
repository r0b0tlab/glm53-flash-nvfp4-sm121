#!/usr/bin/env python3
"""ModelOpt NVFP4 corruption canary (vLLM #54150): U+FFFD + repetition-lock probe.
Usage: canary_corruption.py <base_url> <model> [passes]   exit 0 only if clean.
"""
import json
import re
import sys
import urllib.request

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2]
PASSES = int(sys.argv[3]) if len(sys.argv) > 3 else 3

PROMPTS = [
    ("hangul", "다음 문장을 한국어로 정확히 따라 쓰세요: 안녕하세요, 오늘 날씨가 좋네요."),
    ("mixed", "Write exactly: café — naïve — 東京 — Ωmega — 42%."),
    ("toolish", 'Call the tool: get_weather(city="Austin"). Output only the call.'),
    ("longprose", "Write 400 words of English prose about lighthouses."),
    ("code", "def fib(n):\n    # complete this function\n"),
    ("math", "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. "
             "How much does the ball cost? Show the final number."),
]


def ask(prompt, max_tokens=512):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.load(r)
    m = j["choices"][0]["message"]
    return ((m.get("content") or "") + (m.get("reasoning") or "")
            + (m.get("reasoning_content") or ""))


def locked(text):
    toks = re.findall(r"\S+", text)
    for i in range(max(0, len(toks) - 6)):
        g = " ".join(toks[i:i + 6])
        if len(g) > 12 and text.count(g) >= 8:
            return g
    return None


bad = 0
for p in range(PASSES):
    for name, prompt in PROMPTS:
        out = ask(prompt)
        rep = locked(out)
        fffd = out.count("\ufffd")
        status = "ok"
        if fffd or rep or not out.strip():
            status = "FAIL"
            bad += 1
        print(f"pass={p} case={name} chars={len(out)} U+FFFD={fffd} "
              f"lock={rep!r} empty={not out.strip()} -> {status}")
print(json.dumps({"verdict": "PASS" if bad == 0 else "FAIL", "bad_cases": bad}))
sys.exit(0 if bad == 0 else 1)
