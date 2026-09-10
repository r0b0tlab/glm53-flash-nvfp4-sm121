#!/usr/bin/env python3
"""Vision gate: 6 image probes + 2 frame-sequence probes against the local server."""
import base64
import json
import re
import sys
import urllib.request
from pathlib import Path

BASE = __import__("os").environ.get("GLM53_BASE", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = __import__("os").environ.get("GLM53_MODEL", "glm-5.3-flash-nvfp4-sm121")
import os as _os
MEDIA = Path(_os.environ.get("GLM53_MEDIA", Path.home() / "glm53-flash-nvfp4" / "media"))
OUT = Path(_os.environ.get("GLM53_EVIDENCE", Path.home() / "glm53-flash-nvfp4" / "evidence")) / "g8_vision.json"


def data_url(p):
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def chat(parts, max_tokens=8192):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
                       "chat_template_kwargs": {"reasoning_effort": "max"},
                       "messages": [{"role": "user", "content": parts}]}).encode()
    req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def ask(text, files):
    parts = [{"type": "image_url", "image_url": {"url": data_url(MEDIA / f)}} for f in files]
    parts.append({"type": "text", "text": text})
    return chat(parts)


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def main():
    probes = []
    for n in (3, 4, 5, 6):
        out = ask("How many red squares are in this image? Answer with a single number.", [f"count_{n}.png"])
        probes.append({"id": f"count_{n}", "expect": str(n), "pass": str(n) in re.findall(r"\d+", out), "out": out})
    out = ask("What four-digit number is shown? Answer with digits only.", ["digits_4271.png"])
    probes.append({"id": "digits", "expect": "4271", "pass": "4271" in out, "out": out})
    out = ask("In this bar chart, which colored bar is the tallest? Answer with one word.", ["bars.png"])
    probes.append({"id": "bars", "expect": "green", "pass": "green" in norm(out), "out": out})
    out = ask("I am sending 4 frames in order. List their solid colors in order, comma-separated.",
              [f"frame_{i}.png" for i in range(4)])
    seq = norm(out)
    ok = all(w in seq for w in ("red", "green", "blue", "yellow")) and \
        seq.index("red") < seq.index("green") < seq.index("blue") < seq.index("yellow")
    probes.append({"id": "frame_order", "expect": "red,green,blue,yellow", "pass": ok, "out": out})
    out = ask("In the 4 frames I sent, what is the color of the third frame? Answer with one word.",
              [f"frame_{i}.png" for i in range(4)])
    probes.append({"id": "frame3", "expect": "blue", "pass": "blue" in norm(out), "out": out})

    passed = sum(p["pass"] for p in probes)
    res = {"probes": probes, "passed": passed, "total": len(probes),
           "verdict": {"G8_PASS": passed == len(probes)},
           "video_note": "native video_url unsupported by vLLM fork impl at e2e5751; "
                         "video content probed as ordered multi-image frames"}
    json.dump(res, open(OUT, "w"), indent=1)
    for p in probes:
        print(("PASS " if p["pass"] else "FAIL ") + p["id"], "->", p["out"][:120].replace("\n", " "))
    print("VERDICT:", res["verdict"], f"{passed}/{len(probes)}")
    sys.exit(0 if passed == len(probes) else 1)


if __name__ == "__main__":
    main()
