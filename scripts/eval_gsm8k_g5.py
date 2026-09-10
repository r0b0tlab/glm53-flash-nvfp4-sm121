#!/usr/bin/env python3
"""GSM8K 200-sample quality row (stdlib; reads evidence/gsm8k_test.jsonl)."""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = __import__("os").environ.get("GLM53_BASE", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = __import__("os").environ.get("GLM53_MODEL", "glm-5.3-flash-nvfp4-sm121")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
DATA = Path.home() / "ling30vl-nvfp4" / "evidence" / "gsm8k_test.jsonl"
OUT = Path.home() / "ling30vl-nvfp4" / "evidence" / "g5_gsm8k.json"


def chat(prompt):
    body = json.dumps({"model": MODEL, "temperature": 0.0, "max_tokens": 1024,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def extract(text):
    m = re.findall(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if m:
        return m[-1].replace(",", "")
    m = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return m[-1].replace(",", "") if m else None


def main():
    rows = [json.loads(line) for line in open(DATA)][:N]
    correct = total = fail = 0
    times = []
    for row in rows:
        ref = re.findall(r"####\s*(-?\d[\d,]*\.?\d*)", row["answer"])[-1].replace(",", "")
        t0 = time.time()
        try:
            out = chat("Solve the following math problem step by step, and end your answer "
                       f'with "#### <number>".\n\n{row["question"]}')
        except Exception as e:
            print("ERR:", e)
            fail += 1
            continue
        times.append(time.time() - t0)
        total += 1
        ok = extract(out) == ref
        correct += int(ok)
        if not ok and total <= 5:
            print(f"MISS ref={ref} pred={extract(out)} out={out[:160]!r}")
    acc = correct / total if total else 0.0
    res = {"samples": total, "correct": correct, "accuracy": acc, "model_failures": fail,
           "avg_latency_s": (sum(times) / len(times)) if times else 0.0,
           "extract": "flexible (#### last, else last number)",
           "dataset_sha256": __import__("hashlib").sha256(open(DATA, "rb").read()).hexdigest()}
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT, "w"), indent=1)
    sys.exit(0 if acc >= 0.90 else 1)


if __name__ == "__main__":
    main()
