#!/usr/bin/env python3
"""Text gate probes for the MP artifact (stdlib). 17*23=391, fib(10)=55, JSON, code, tool."""
import json
import re
import sys
import urllib.request

BASE = __import__("os").environ.get("GLM53_BASE", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = __import__("os").environ.get("GLM53_MODEL", "glm-5.3-flash-nvfp4-sm121")


def chat(messages, tools=None, max_tokens=4096):
    body = {"model": MODEL, "temperature": 0, "max_tokens": max_tokens, "messages": messages}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]


def main():
    results = {}
    c = chat([{"role": "user", "content": "What is 17*23? Think briefly."}], max_tokens=2048)
    txt = c["message"]["content"]
    results["mul"] = {"expect": "391", "pass": "391" in txt, "out": txt[-200:]}
    c = chat([{"role": "user", "content": "What is the 10th Fibonacci number (F(10), with F(0)=0, F(1)=1)?"}])
    results["fib"] = {"expect": "55", "pass": "55" in c["message"]["content"], "out": c["message"]["content"][-200:]}
    c = chat([{"role": "user", "content": 'Reply with exactly {"ok":true} and nothing else.'}])
    results["json"] = {"expect": '{"ok":true}', "pass": bool(re.search(r'\{\s*"ok"\s*:\s*true\s*\}', c["message"]["content"])),
                       "out": c["message"]["content"][-200:]}
    c = chat([{"role": "user", "content": "Write a Python one-liner that reverses a string s. Code only."}])
    results["code"] = {"expect": "[::-1]", "pass": "[::-1]" in c["message"]["content"],
                       "out": c["message"]["content"][-200:]}
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    c = chat([{"role": "user", "content": "What is the weather in Austin, Texas? Use the tool."}], tools=tools)
    tc = c["message"].get("tool_calls") or []
    ok = bool(tc) and tc[0]["function"]["name"] == "get_weather"
    try:
        args = json.loads(tc[0]["function"]["arguments"]) if ok else {}
        ok = ok and "Austin" in args.get("city", "")
    except Exception:
        ok = False
    results["tool"] = {"expect": "tool_call get_weather(city=Austin...)", "pass": ok,
                       "out": json.dumps(tc)[:200]}
    for k, v in results.items():
        print(("PASS " if v["pass"] else "FAIL ") + k, "->", v["out"][:120].replace("\n", " "))
    passed = sum(v["pass"] for v in results.values())
    print(f"TEXT_GATES {passed}/{len(results)}")
    import os as _os
    with open(_os.environ.get("GLM53_EVIDENCE", _os.path.expanduser("~/glm53-flash-nvfp4/evidence")) + "/g7_text.json", "w") as fh:
        json.dump(results, fh, indent=1)
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
