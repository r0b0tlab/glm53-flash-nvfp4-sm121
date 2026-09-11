#!/usr/bin/env bash
# Post-boot gate bundle for the DFlash2 serve (run on rank0).
# canary -> acceptance matrix -> lossless record/compare -> benches.
set -uo pipefail
cd ~/glm53-flash-nvfp4 || exit 1
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="evidence/gates/$TS"
mkdir -p "$OUT"
M=glm-5.3-flash-nvfp4-sm121
B=http://127.0.0.1:8000

echo "== canary (3 passes) =="
python3 scripts/canary_corruption.py "$B" "$M" 3 > "$OUT/canary.json" 2>&1 || true
tail -1 "$OUT/canary.json"

echo "== acceptance matrix (512 tok) =="
python3 scripts/acceptance_matrix.py "$B" "$M" 512 > "$OUT/acceptance_matrix.json" 2>&1 || true
grep -E '"case"|"summary"' "$OUT/acceptance_matrix.json" | tail -4

echo "== lossless record (dflash2) =="
python3 scripts/test_spec_lossless.py record dflash2 "$B" "$M" 2>&1 | tail -2

echo "== lossless compare (ar vs dflash2) =="
python3 scripts/test_spec_lossless.py compare ar dflash2 2>&1 | tail -8 | tee "$OUT/lossless_compare.json"

echo "== perf: code prompt, 2x512 =="
python3 - <<'EOF' | tee "$OUT/bench_code.txt"
import json, time, urllib.request
M = "glm-5.3-flash-nvfp4-sm121"
def ask(p, mt=512):
    body = json.dumps({"model": M, "temperature": 0, "max_tokens": mt,
                       "messages": [{"role": "user", "content": p}]}).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    j = json.load(urllib.request.urlopen(req, timeout=1800))
    dt = time.time() - t0
    return j["usage"]["completion_tokens"], dt
p = ("Write a Python function that merges two sorted lists into one sorted list. "
     "Include a docstring and type hints.")
for i in range(2):
    ct, dt = ask(p)
    print(f"code run {i}: {ct} tok / {dt:.1f}s = {ct/dt:.2f} tok/s")
EOF

echo "POST_BOOT_GATES_DONE $TS"
