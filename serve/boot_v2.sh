#!/usr/bin/env bash
# First-boot profile for the GLM-5.3-Flash-NVFP4 overlay image (no drafter).
# FULL_DECODE_ONLY: the model is not torch-compiled (piecewise graphs
# unavailable with breakable off), and sparse MLA admits FULL_DECODE_ONLY.
set -euo pipefail
cd "$(dirname "$0")/.."
export IMAGE="${IMAGE:-glm53-flash-vllm-sm121:overlay-v2}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export SPEC_METHOD="${SPEC_METHOD:-none}"
export SPEC_TOKENS="${SPEC_TOKENS:-0}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"
export COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}}"
exec bash serve/launch-glm53-tp2.sh
