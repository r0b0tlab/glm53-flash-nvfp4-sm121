#!/usr/bin/env bash
# BFCL v4 multi_turn_base structural-hard20 lane for GLM-5.3-Flash-NVFP4 (DFlash2 serve).
# Run ON rank0 inside tmux/background. thinking=max via the wrapper's REQUIRED_CHAT_KWARGS.
set -euo pipefail
EV=$HOME/glm53-flash-nvfp4/evidence/bfcl-hard20
export Q200V2_RUNNER=$HOME/qwen38-flash-next-w4a16/q200v2ar-20260829T141533Z-runner
export BFCL_PROJECT_ROOT=${BFCL_PROJECT_ROOT:-$EV/bfcl-run}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}
export OPENAI_API_KEY=EMPTY
export Q200_SERVED_MODEL=glm-5.3-flash-nvfp4-sm121
export Q200_IMAGE_ID=sha256:3fceaf69a626569e6063bcaf63277e68b2b954aa099ecf2b5cb782943d446c17
export Q200_PROFILE_ID=034a73df69f708b709dd75c4ea6adf3c22b19ba1832a65b69c88f9f990f690fb
export Q200_CANDIDATE_ID=glm53-dflash2
export Q200_BFCL_REGISTRY=glm53-hard20-FC
export Q200_BFCL_TIMING_PATH=$EV/bfcl-hard20-timing.json
export BFCL_NUM_THREADS=1
export BFCL_HTTP_TIMEOUT=3600
export BFCL_MAX_RETRIES=2
export BFCL_MAX_TOKENS=8192
mkdir -p "$EV"
rm -rf "$BFCL_PROJECT_ROOT"
mkdir -p "$BFCL_PROJECT_ROOT"
rm -f "$Q200_BFCL_TIMING_PATH" 2>/dev/null || true
echo "=== BFCL hard20 start $(date -Is) base=$OPENAI_BASE_URL root=$BFCL_PROJECT_ROOT ==="
~/r0b0bench-venv/bin/python "$HOME/glm53-flash-nvfp4/scripts/run_bfcl_hard20_glm53.py" run \
  > "$EV/bfcl-hard20.stdout" 2> "$EV/bfcl-hard20.stderr"
rc=$?
echo "BFCL_RC=$rc"
tail -20 "$EV/bfcl-hard20.stdout" | cut -c1-200
exit $rc
