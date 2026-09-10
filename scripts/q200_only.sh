#!/usr/bin/env bash
# Q200v2-only lane (run ON rank0 after the serve is healthy).
set -uo pipefail
cd ~/glm53-flash-nvfp4
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="evidence/q200v2/$TS"
mkdir -p "$OUT"
cd "$OUT"
Q200V2_RUNNER="$HOME/qwen38-flash-next-w4a16/q200v2ar-20260829T141533Z-runner" \
python3 "$HOME/glm53-flash-nvfp4/scripts/run_q200v2_glm53.py" \
  --base-url http://127.0.0.1:8000 \
  --run-id "glm53-q200v2-$TS" \
  --model glm-5.3-flash-nvfp4-sm121 \
  --image-id sha256:3fceaf69a626569e6063bcaf63277e68b2b954aa099ecf2b5cb782943d446c17 \
  --profile-id "dflash2-k7-bf16kv-126k-max" \
  --candidate-id "glm53-dflash2" \
  --workers "${WORKERS:-2}" --max-tokens 8192 > run.log 2>&1
echo "Q200_RC=$?" >> run.log
tail -20 run.log
