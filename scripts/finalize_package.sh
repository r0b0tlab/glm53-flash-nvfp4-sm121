#!/usr/bin/env bash
# Finalize the publication package: sync evidence -> sanitize -> manifest -> pub/.
# Run on the HEAD after all lanes complete.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
# shellcheck disable=SC1091
source config.env

echo "== sync evidence from node3 =="
ssh -o BatchMode=yes "$RANK0" 'tar -C ~/glm53-flash-nvfp4/evidence -cf - . 2>/dev/null' | tar -C evidence -xf - 2>/dev/null || true

echo "== sync vision-bench evidence from node2 =="
ssh -o BatchMode=yes "$RANK1" 'tar -C ~/glm53-flash-nvfp4/evidence -cf - vision-bench VISION_BENCH.log 2>/dev/null' | tar -C evidence -xf - 2>/dev/null || true

echo "== sync repo files into pub/ =="
cp "$ROOT/README.md" pub/README.md
rm -rf pub/container pub/patches pub/serve pub/scripts
cp -r container patches serve scripts pub/
cp .github/workflows/validate.yml pub/.github/workflows/validate.yml

echo "== sanitize evidence =="
python3 scripts/package_evidence.py "$ROOT/evidence" "$ROOT/pub/evidence"

echo "== residue check =="
if grep -rInE "/home/[a-z0-9_]+|100\.[0-9]+\.[0-9]+\.[0-9]+|192\.168\.[0-9]+\.[0-9]+" \
    --include='*.json' --include='*.md' --include='*.txt' --include='*.log' \
    --include='*.py' --include='*.sh' pub/ ; then
  echo "::error::private residue found in pub/"
  exit 1
fi
echo "no private residue"

echo "== manifest verify =="
(cd pub/evidence && sha256sum -c MANIFEST.sha256 > /dev/null && echo "MANIFEST OK ($(wc -l < MANIFEST.sha256) files)")

echo "== script syntax =="
for f in pub/scripts/*.py pub/patches/*.py; do python3 -m py_compile "$f"; done
for f in pub/serve/*.sh pub/scripts/*.sh pub/container/*.sh; do bash -n "$f"; done
echo "syntax OK"

echo "PACKAGE_FINALIZED"
