#!/usr/bin/env bash
# DFlash2 spec-decode profile (primary campaign lane).
# method=dflash, num_speculative_tokens = block_size - 1 = 7.
# Thinking is always on for this model; reasoning_effort=max everywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
export IMAGE="${IMAGE:-glm53-flash-vllm-sm121:overlay-v2}"
export SPEC_METHOD="${SPEC_METHOD:-dflash2}"
export SPEC_TOKENS="${SPEC_TOKENS:-7}"
export SPEC_MODEL_DIR="${SPEC_MODEL_DIR:-$HOME/models/incoai/GLM-5.3-Flash-DFlash2}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-126720}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.86}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"
export COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}}"
# FlashInfer autotune's distributed profiling hung once on rank0 during a
# spec-decode warmup (rank1 idle, rank0 spinning). The tuned configs are
# cached; skipping the sweep boots reliably and measured within noise of the
# autotuned baseline (see evidence/perf).
export KERNEL_CONFIG="${KERNEL_CONFIG:-{\"enable_flashinfer_autotune\":false\}}"
exec bash serve/launch-glm53-tp2.sh
