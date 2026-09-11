#!/usr/bin/env bash
# Regression guard for the reported failure:
#   ./container/build.sh: line 13: config.env: No such file or directory
# config.env is intentionally not committed (see config.env.example), so a fresh
# clone must still be able to build. docker is stubbed here so the check needs no
# daemon and does not pull the ~22 GB pinned base.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

mkdir -p "${TMP}/container" "${TMP}/stub"
cp "${ROOT}/container/build.sh" "${TMP}/container/build.sh"
cp "${ROOT}/container/Dockerfile.overlay" "${TMP}/container/Dockerfile.overlay"
if [[ -e "${TMP}/config.env" ]]; then
  echo "FAIL: fixture unexpectedly contains config.env" >&2
  exit 1
fi

cat > "${TMP}/stub/docker" <<'STUB'
#!/usr/bin/env bash
echo "stub-docker: $*"
exit 0
STUB
chmod +x "${TMP}/stub/docker"

set +e
out="$(PATH="${TMP}/stub:${PATH}" bash "${TMP}/container/build.sh" 2>&1)"
rc=$?
set -e

if [[ ${rc} -ne 0 ]]; then
  echo "FAIL: container/build.sh exited ${rc} without config.env" >&2
  echo "${out}" >&2
  exit 1
fi

grep -q 'config.env not found' <<<"${out}" || {
  echo "FAIL: no notice that config.env is missing" >&2; echo "${out}" >&2; exit 1; }
grep -q 'in-image verify' <<<"${out}" || {
  echo "FAIL: build did not reach the in-image verify step" >&2; echo "${out}" >&2; exit 1; }
grep -q 'distribute skipped' <<<"${out}" || {
  echo "FAIL: distribution was not skipped while RANK0/RANK1 are unset" >&2; echo "${out}" >&2; exit 1; }
if grep -q 'config.env: No such file or directory' <<<"${out}"; then
  echo "FAIL: the originally reported error text is still produced" >&2
  exit 1
fi

# And the same fixture with config.env copied from the example must source it and
# still skip distribution (the example ships placeholders, no real ranks).
cp "${ROOT}/config.env.example" "${TMP}/config.env"
set +e
out2="$(PATH="${TMP}/stub:${PATH}" bash "${TMP}/container/build.sh" 2>&1)"
rc2=$?
set -e
if [[ ${rc2} -ne 0 ]]; then
  echo "FAIL: build.sh exited ${rc2} with config.env copied from the example" >&2
  echo "${out2}" >&2
  exit 1
fi
if grep -q 'config.env not found' <<<"${out2}"; then
  echo "FAIL: present config.env was not sourced" >&2
  exit 1
fi
grep -q 'distribute skipped' <<<"${out2}" || {
  echo "FAIL: a copied config.env.example triggered distribution (ranks must ship commented)" >&2
  echo "${out2}" >&2; exit 1; }

# Phase 3: the sibling class — a script that genuinely needs node values must fail
# closed with an actionable message, not a shell "No such file or directory".
set +e
out3="$(cd "${ROOT}" && bash serve/launch-glm53-tp2.sh 2>&1)"
rc3=$?
set -e
if [[ ${rc3} -ne 2 ]]; then
  echo "FAIL: serve/launch-glm53-tp2.sh exited ${rc3} (want 2) without config.env" >&2
  echo "${out3}" >&2; exit 1
fi
grep -q 'RANK0 is unset' <<<"${out3}" || {
  echo "FAIL: serve launcher did not name the missing variable" >&2; echo "${out3}" >&2; exit 1; }
if grep -q 'No such file or directory' <<<"${out3}"; then
  echo "FAIL: serve launcher still leaks a raw shell error" >&2
  exit 1
fi

echo "PASS: build.sh runs without config.env, sources it when present, skips distribution; serve launcher fails closed with a clear message"
