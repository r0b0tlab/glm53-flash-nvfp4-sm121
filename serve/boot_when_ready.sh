#!/usr/bin/env bash
# Wait for both ranks to be ready (model tree + overlay-v2 image), then boot
# and block until /health. Run on the head, in tmux.
set -euo pipefail
cd "$(dirname "$0")/.."
source config.env

echo "== waiting for node2 model copy (33 shards) =="
until ssh -o BatchMode=yes "$RANK1" 'ls ~/models/nvidia/GLM-5.3-Flash-NVFP4/*.safetensors 2>/dev/null | wc -l' | grep -q '^33$'; do
  sleep 60
done
echo "model copy complete on node2"

echo "== waiting for overlay-v2 on node2 =="
until ssh -o BatchMode=yes "$RANK1" 'docker image inspect glm53-flash-vllm-sm121:overlay-v2 --format "{{.Id}}"' >/dev/null 2>&1; do
  sleep 20
done
echo "image present on node2"

echo "== booting =="
bash serve/boot_v2.sh

echo "== waiting for /health =="
until ssh -o BatchMode=yes "$RANK0" 'curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1'; do
  sleep 30
done
echo "HEALTHY $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ssh -o BatchMode=yes "$RANK0" 'docker logs glm53_vllm 2>&1 | grep -iE "Using .*MoE backend|attention backend|MLA|KV cache|graph|Marlin|emulation|index_topk|Eagle3" | tail -25' | tee evidence/gates/g1_boot_grep.txt
