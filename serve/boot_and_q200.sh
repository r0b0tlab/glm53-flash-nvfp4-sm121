#!/usr/bin/env bash
# Boot the DFlash2 profile, wait for health, then run the Q200v2 lane.
set -euo pipefail
cd "$(dirname "$0")/.."
# config.env carries node/fabric specifics and is not committed (see config.env.example).
if [[ -f config.env ]]; then
  # shellcheck disable=SC1091
  source config.env
fi
[[ -n "${RANK0:-}" ]] || { echo "error: RANK0 is unset — set it in config.env (see config.env.example)" >&2; exit 2; }
bash serve/boot_dflash2.sh
echo "== waiting for /health =="
until ssh -o BatchMode=yes "$RANK0" 'curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1'; do sleep 30; done
echo "HEALTHY $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ssh -o BatchMode=yes "$RANK0" 'cd ~/glm53-flash-nvfp4 && bash scripts/q200_only.sh'
echo "Q200_LANE_DONE"
