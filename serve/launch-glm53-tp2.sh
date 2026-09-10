#!/usr/bin/env bash
# GLM-5.3-Flash-NVFP4 dual-GB10 TP=2 SM121 serve launcher. Run on the HEAD.
# Rank1 (node2) starts first, then rank0 (node3) which binds the API.
# Usage: [ENV=...] bash serve/launch-glm53-tp2.sh          (SG_DRYRUN=1 to preview)
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_HERE}/../config.env"

IMAGE="${IMAGE:?set IMAGE}"
NAME="${NAME:-glm53_vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-}"
SPEC_METHOD="${SPEC_METHOD:-mtp}"          # mtp | dflash2 | none
SPEC_TOKENS="${SPEC_TOKENS:-5}"
SPEC_MODEL_DIR="${SPEC_MODEL_DIR:-}"       # drafter dir (dflash2 only)
REASONING_PARSER="${REASONING_PARSER:-deepseek_r1}"
TOOL_PARSER="${TOOL_PARSER:-glm47}"
MOE_BACKEND="${MOE_BACKEND:-auto}"
LINEAR_BACKEND="${LINEAR_BACKEND:-auto}"
ATTN_BACKEND="${ATTN_BACKEND:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-}"
THINKING_DEFAULT="${THINKING_DEFAULT:-off}"
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [[ "${THINKING_DEFAULT}" == "off" ]]; then
  # GLM-5.3's template cannot disable thinking; "off" means the model's default
  # effort, which we pin to max per the user's standing rule for this model.
  CHAT_KWARGS='{"reasoning_effort":"max"}'
else
  CHAT_KWARGS='{"reasoning_effort":"max"}'
fi

serve_cmd() {
  local rank="$1" headless="${2:-}"
  local -a c=(
    exec vllm serve /model
    --served-model-name "$SERVED"
    --host 0.0.0.0 --port "$API_PORT"
    --tensor-parallel-size 2 --pipeline-parallel-size 1
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --enable-chunked-prefill
    --long-prefill-token-threshold 1024
    --tool-call-parser "$TOOL_PARSER" --enable-auto-tool-choice
    --reasoning-parser "$REASONING_PARSER"
    --default-chat-template-kwargs "$CHAT_KWARGS"
    --distributed-executor-backend mp
    --nnodes 2 --node-rank "$rank"
    --master-addr "$RANK0_FABRIC" --master-port "$MASTER_PORT"
    --moe-backend "$MOE_BACKEND" --linear-backend "$LINEAR_BACKEND"
  )
  case "$SPEC_METHOD" in
    mtp)     [[ "$SPEC_TOKENS" != "0" ]] && c+=(--speculative-config "$(printf '{"method":"mtp","num_speculative_tokens":%s}' "$SPEC_TOKENS")") ;;
    dflash2) c+=(--speculative-config "$(printf '{"method":"dflash","model":"/drafter","num_speculative_tokens":%s}' "$SPEC_TOKENS")") ;;
    none|"") : ;;
    *) echo "unknown SPEC_METHOD=$SPEC_METHOD" >&2; exit 2 ;;
  esac
  [[ -n "$KV_CACHE_MEMORY_BYTES" ]] && c+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
  [[ -n "$ATTN_BACKEND" ]] && c+=(--attention-backend "$ATTN_BACKEND")
  [[ "$ENFORCE_EAGER" == "1" ]] && c+=(--enforce-eager)
  [[ -n "$COMPILATION_CONFIG" ]] && c+=(--compilation-config "$COMPILATION_CONFIG")
  [[ -n "$KERNEL_CONFIG" ]] && c+=(--kernel-config "$KERNEL_CONFIG")
  [[ -n "$headless" ]] && c+=(--headless)
  printf '%q ' "${c[@]}"
}

common=( --gpus all --ipc=host --network host --entrypoint /bin/bash
  --shm-size=64g --ulimit memlock=-1:-1 --ulimit stack=67108864
  --cap-add=IPC_LOCK --device=/dev/infiniband
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="$VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"
  -e TRITON_CACHE_DIR=/root/.cache/triton
  -e VLLM_CACHE_ROOT=/root/.cache/vllm
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX"
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_NVLS_ENABLE=0
  -e NCCL_DEBUG="$NCCL_DEBUG"
  -e FLASHINFER_DISABLE_VERSION_CHECK=1
  -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
  -e FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
  -e MASTER_ADDR="$RANK0_FABRIC" -e MASTER_PORT="$MASTER_PORT"
)

if [[ "${SG_DRYRUN:-0}" == "1" ]]; then
  echo "RANK1: docker run ... $(printf '%q' "$(serve_cmd 1 x)")"
  echo "RANK0: docker run ... $(printf '%q' "$(serve_cmd 0 '')")"
  exit 0
fi

for H in "$RANK0" "$RANK1"; do
  ssh -o BatchMode=yes "$H" 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; mkdir -p ~/.cache/triton ~/.cache/vllm ~/.cache/flashinfer'
  ssh -o BatchMode=yes "$H" "docker rm -f $NAME 2>/dev/null || true"
done
LOCAL_ID="$(ssh -o BatchMode=yes "$RANK0" "docker image inspect '$IMAGE' --format '{{.Id}}'")"
REMOTE_ID="$(ssh -o BatchMode=yes "$RANK1" "docker image inspect '$IMAGE' --format '{{.Id}}'")"
if [[ "$LOCAL_ID" != "$REMOTE_ID" ]]; then
  echo "image parity FAIL: $LOCAL_ID != $REMOTE_ID" >&2; exit 1
fi

mounts_rank() {
  local rank="$1"
  echo "-v $MODEL_DIR:/model:ro -v \$HOME/.cache/triton:/root/.cache/triton -v \$HOME/.cache/vllm:/root/.cache/vllm -v \$HOME/.cache/flashinfer:/root/.cache/flashinfer"
  if [[ "$SPEC_METHOD" == "dflash2" ]]; then echo "-v $SPEC_MODEL_DIR:/drafter:ro"; fi
}

echo "== rank1 (worker first) =="
# shellcheck disable=SC2046,SC2086
ssh -o BatchMode=yes "$RANK1" docker run -d --name "$NAME" "${common[@]}" \
  $(mounts_rank 1) -e NCCL_IB_HCA="$RANK1_HCA" -e NCCL_SOCKET_IFNAME="$RANK1_ETH" \
  "$LOCAL_ID" -lc "$(printf '%q' "$(serve_cmd 1 x)")"
sleep 25
echo "== rank0 (API) =="
# shellcheck disable=SC2046,SC2086
ssh -o BatchMode=yes "$RANK0" docker run -d --name "$NAME" "${common[@]}" \
  $(mounts_rank 0) -e NCCL_IB_HCA="$RANK0_HCA" -e NCCL_SOCKET_IFNAME="$RANK0_ETH" \
  "$LOCAL_ID" -lc "$(printf '%q' "$(serve_cmd 0 '')")"
echo "API: http://$RANK0_FABRIC:$API_PORT/v1  (poll /health, never /v1/models)"
echo "poll: ssh $RANK0 'until curl -sf http://127.0.0.1:$API_PORT/health >/dev/null; do sleep 20; done; echo HEALTHY'"
