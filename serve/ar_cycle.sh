#!/usr/bin/env bash
# AR (no-drafter) cycle: boot SPEC_METHOD=none, wait for health, then record the
# spec-decode losslessness reference + AR baseline perf.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source config.env
pkill -f "[b]oot_when_ready.sh" 2>/dev/null || true
for H in "$RANK0" "$RANK1"; do
  ssh -o BatchMode=yes "$H" 'docker rm -f glm53_vllm 2>/dev/null || true'
done
IMAGE=glm53-flash-vllm-sm121:overlay-v2 MAX_MODEL_LEN=131072 MAX_NUM_SEQS=4 \
  GPU_MEMORY_UTILIZATION=0.86 SPEC_METHOD=none SPEC_TOKENS=0 KV_CACHE_DTYPE=bfloat16 \
  COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  bash serve/launch-glm53-tp2.sh
echo "== waiting for /health =="
until ssh -o BatchMode=yes "$RANK0" 'curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1'; do sleep 30; done
echo "AR_HEALTHY $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ssh -o BatchMode=yes "$RANK0" '
  cd ~/glm53-flash-nvfp4 &&
  echo "== parser sanity (truncation case) ==" &&
  python3 - <<EOF
import json, urllib.request
body = json.dumps({"model":"glm-5.3-flash-nvfp4-sm121","temperature":0,"max_tokens":64,
  "messages":[{"role":"user","content":"Count from 1 to 200 slowly, one number per line."}]}).encode()
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
j = json.load(urllib.request.urlopen(req, timeout=300))
m = j["choices"][0]["message"]
print("content_len:", len(m.get("content") or ""), "reasoning_len:", len(m.get("reasoning_content") or ""))
EOF
  echo "== lossless AR record ==" &&
  python3 scripts/test_spec_lossless.py record ar http://127.0.0.1:8000 glm-5.3-flash-nvfp4-sm121 2>&1 | tail -3 &&
  echo "== AR bench (prose prompt, 3x512) ==" &&
  GLM53_BASE=http://127.0.0.1:8000/v1/chat/completions GLM53_MODEL=glm-5.3-flash-nvfp4-sm121 python3 scripts/bench_decode.py ar 3 512 2>&1 | tail -5
'
