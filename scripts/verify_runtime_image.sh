#!/bin/bash
# Post-build verification gates for the SM121 vLLM runtime image.
# Usage: bash verify_runtime_image.sh [image]
set -uo pipefail
IMG=${1:-ling30vl-vllm-sm121:e2e5751}
PROBE=$HOME/ling30vl-nvfp4/convert
OUT=$HOME/ling30vl-nvfp4/evidence/image-verify.txt
{
  echo "=== image inspect ==="
  docker image inspect "$IMG" --format '{{.Id}} arch={{.Architecture}}/{{.Os}} size={{.Size}}'
  docker image inspect "$IMG" --format '{{json .Config.Labels}}'
  echo
  echo "=== arch gate: plain sm_120 present, sm_120a absent ==="
  docker run --rm --entrypoint bash "$IMG" -c '
V=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
cd "$V" || exit 1
for so in *.so; do
  [ -f "$so" ] || continue
  echo -n "$so: "
  cuobjdump --list-elf "$so" 2>/dev/null | grep -oE "(sm|compute)_[0-9]+[a-z]?" | sort -u | tr "\n" " "
  echo
done'
  echo
  echo "=== identity + patch checks ==="
  docker run --rm --entrypoint bash -v "$PROBE:/probe:ro" "$IMG" -c 'python3 /probe/verify_identity.py'
  echo
  echo "=== GPU microtests ==="
  docker run --rm --runtime nvidia --gpus all --entrypoint bash -v "$PROBE:/probe:ro" "$IMG" -c 'python3 /probe/microtest_sm121.py'
  echo
  echo "=== IMAGE_VERIFY_DONE ==="
} 2>&1 | tee "$OUT"
