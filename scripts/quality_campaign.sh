#!/usr/bin/env bash
# Quality campaign on the live DFlash2 serve (run ON rank0/node3).
# Order: canary -> text gates -> Q200v2 text180 -> NIAH core ladder.
set -uo pipefail
cd ~/glm53-flash-nvfp4 || exit 1
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="evidence/campaign/$TS"
mkdir -p "$OUT"
echo "CAMPAIGN $TS" | tee "$OUT/README.txt"

run() { echo "== $1 ==" | tee -a "$OUT/README.txt"; shift; "$@" 2>&1 | tee -a "$OUT/README.txt"; }

echo "== canary (3 passes) =="
python3 scripts/canary_corruption.py http://127.0.0.1:8000 glm-5.3-flash-nvfp4-sm121 3 \
  > "$OUT/canary.json" 2>&1 || true
tail -1 "$OUT/canary.json" | tee -a "$OUT/README.txt"

echo "== text gates =="
GLM53_BASE=http://127.0.0.1:8000/v1/chat/completions GLM53_MODEL=glm-5.3-flash-nvfp4-sm121 \
  python3 scripts/probe_text.py > "$OUT/text_gates.txt" 2>&1 || true
tail -1 "$OUT/text_gates.txt" | tee -a "$OUT/README.txt"

echo "== Q200v2 text180 (workers=2, max_tokens=8192, effort=max) =="
( cd "$OUT" || exit 1
  # Kit provenance: the frozen Q200v2 set is published at
  # https://github.com/r0b0tlab/r0b0bench/tree/main/subsets/q200v2
  # (quality-text-180-v2.jsonl sha256 74623ab9b075120cd6f7a93059cc16d8817a6039dd20118b8f0350279f8b1ed6).
  Q200V2_RUNNER="${Q200V2_RUNNER:-$HOME/qwen38-flash-next-w4a16/q200v2ar-20260829T141533Z-runner}" \
  python3 "$HOME/glm53-flash-nvfp4/scripts/run_q200v2_glm53.py" \
    --base-url http://127.0.0.1:8000 \
    --run-id "glm53-q200v2-$TS" \
    --model glm-5.3-flash-nvfp4-sm121 \
    --image-id sha256:3fceaf69a626569e6063bcaf63277e68b2b954aa099ecf2b5cb782943d446c17 \
    --profile-id "dflash2-k7-bf16kv-131k-max" \
    --candidate-id glm53-dflash2 \
    --workers 2 --max-tokens 8192 > q200v2.log 2>&1 )
tail -20 "$OUT/q200v2.log" | tee -a "$OUT/README.txt"

echo "== NIAH core ladder (window 131072; 25/50/90 + multi-key) =="
python3 scripts/run_niah_glm53.py --base-url http://127.0.0.1:8000 \
  --target-tokens 126656 --output "$OUT/niah_131k.json" > "$OUT/niah.log" 2>&1 || true
tail -12 "$OUT/niah.log" | tee -a "$OUT/README.txt"

echo "CAMPAIGN_DONE $TS" | tee -a "$OUT/README.txt"
