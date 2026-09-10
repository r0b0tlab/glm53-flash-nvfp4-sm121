#!/usr/bin/env bash
# Q200v2 text180 quality lane on the GLM-5.3-Flash-NVFP4 DFlash2 serve.
# Run ON rank0 (the API host). reasoning_effort=max (model cannot disable thinking).
set -euo pipefail
cd ~/glm53-flash-nvfp4
export Q200V2_RUNNER="${Q200V2_RUNNER:-$HOME/qwen38-flash-next-w4a16/q200v2ar-20260829T141533Z-runner}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="evidence/q200v2/$TS"
mkdir -p "$OUT"
cd "$OUT"
python3 ~/glm53-flash-nvfp4/scripts/run_q200v2_glm53.py \
  --base-url http://127.0.0.1:8000/v1 \
  --run-id "glm53-q200v2-$TS" \
  --model glm-5.3-flash-nvfp4-sm121 \
  --image-id "${IMAGE_ID:-overlay-v2}" \
  --profile-id "${PROFILE_ID:-dflash2-k7-bf16kv-262k-max}" \
  --candidate-id "glm53-dflash2" \
  --workers "${WORKERS:-2}" \
  --max-tokens 8192 \
  2>&1 | tee -a run.log
