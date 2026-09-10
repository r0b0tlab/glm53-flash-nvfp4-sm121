#!/usr/bin/env python3
"""Spec-decode losslessness: greedy outputs must be byte-identical between the
no-drafter and drafter serves (same prompts, temperature 0).

Record mode:   test_spec_lossless.py record <tag> [base_url] [model]
Compare mode:  test_spec_lossless.py compare <tag_a> <tag_b>
Exit 0 only when every prompt matches exactly.
"""
import json
import sys
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "evidence" / "spec"
PROMPTS = [
    "Explain in exactly one sentence why the sky is blue.",
    "Write a Python one-liner that reverses a string.",
    "What is 17 * 23? Answer with the number only.",
    "List the first five prime numbers, comma-separated, nothing else.",
    "Complete: def fib(n): return n if n < 2 else",
    "Name the capital of France in one word.",
]


def ask(base, model, prompt, max_tokens=2048):
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.load(r)
    m = j["choices"][0]["message"]
    return ((m.get("content") or "") + "|" + (m.get("reasoning") or "")
            + (m.get("reasoning_content") or ""))


def record(tag, base, model):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(PROMPTS):
        out = ask(base, model, p)
        rows.append({"i": i, "prompt": p, "out": out})
        print(f"[{tag}] {i}: {len(out)} chars")
    (OUT / f"{tag}.json").write_text(json.dumps(rows, indent=1))
    print(f"RECORDED {OUT / (tag + '.json')}")
    return 0


def compare(a, b):
    ra = json.loads((OUT / f"{a}.json").read_text())
    rb = json.loads((OUT / f"{b}.json").read_text())
    bad = [i for i, (x, y) in enumerate(zip(ra, rb)) if x["out"] != y["out"]]
    verdict = "PASS" if not bad and len(ra) == len(rb) else "FAIL"
    print(json.dumps({"verdict": verdict, "mismatched_prompts": bad,
                      "n": len(ra)}, indent=1))
    return 0 if verdict == "PASS" else 1


def main() -> int:
    mode = sys.argv[1]
    if mode == "record":
        tag = sys.argv[2]
        base = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8000"
        model = sys.argv[4] if len(sys.argv) > 4 else "glm-5.3-flash-nvfp4-sm121"
        return record(tag, base, model)
    if mode == "compare":
        return compare(sys.argv[2], sys.argv[3])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
