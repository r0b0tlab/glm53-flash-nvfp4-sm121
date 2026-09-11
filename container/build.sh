#!/usr/bin/env bash
# Reproducible build of the GLM-5.3-Flash-NVFP4 SM121 overlay image.
#
#   bash container/build.sh [--push] [--tag TAG]
#
# Pulls the pinned base by digest, applies the anchored overlay patches in-build
# (each fails closed on anchor drift), verifies the result, then — when RANK0/RANK1
# are configured — distributes the image to both ranks and asserts identical image IDs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
# config.env carries node/fabric specifics and is intentionally not committed
# (see config.env.example). Without it the image is still built and verified
# locally; distribution to the two ranks is skipped.
if [[ -f "${ROOT}/config.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/config.env"
else
  echo "note: ${ROOT}/config.env not found — building locally only (copy config.env.example to enable distribution)" >&2
fi
RANK0="${RANK0:-}"
RANK1="${RANK1:-}"

BASE_DIGEST="sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
BASE="vllm/vllm-openai@${BASE_DIGEST}"
TAG="${TAG:-glm53-flash-vllm-sm121:overlay-v2}"
PUSH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --push) PUSH=1 ;;
    --tag) TAG="$2"; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
  shift
done

echo "== pull pinned base ${BASE} =="
docker pull "${BASE}"

echo "== build overlay ${TAG} =="
docker build -f "${ROOT}/container/Dockerfile.overlay" -t "${TAG}" "${ROOT}"

echo "== in-image verify =="
docker run --rm --entrypoint python3 "${TAG}" /opt/glm53/patches/verify_overlay.py

if [[ -n "${RANK0}" && -n "${RANK1}" ]]; then
  echo "== distribute to ranks =="
  ID0="$(ssh -o BatchMode=yes "${RANK0}" "docker image inspect '${TAG}' --format '{{.Id}}'")"
  if ! ssh -o BatchMode=yes "${RANK1}" "docker image inspect '${TAG}' --format '{{.Id}}'" 2>/dev/null | grep -q "${ID0#sha256:}"; then
    docker save "${TAG}" | ssh -o BatchMode=yes "${RANK1}" docker load
  fi
  ID1="$(ssh -o BatchMode=yes "${RANK1}" "docker image inspect '${TAG}' --format '{{.Id}}'")"
  [[ "${ID0}" == "${ID1}" ]] || { echo "image parity FAIL: ${ID0} != ${ID1}" >&2; exit 1; }
  echo "image parity OK: ${ID0}"
else
  echo "== distribute skipped (RANK0/RANK1 unset — see config.env.example) =="
fi

if [[ "${PUSH}" == "1" ]]; then
  echo "== push to GHCR =="
  docker tag "${TAG}" "ghcr.io/r0b0tlab/glm53-flash-nvfp4-sm121:${TAG##*:}"
  docker push "ghcr.io/r0b0tlab/glm53-flash-nvfp4-sm121:${TAG##*:}"
fi
