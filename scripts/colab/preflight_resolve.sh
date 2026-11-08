#!/usr/bin/env bash
# Pre-flight lock drift check for uv.lock, bounded by uv.lock.exclude-newer.
# Usage: bash scripts/colab/preflight_resolve.sh
#
# Refusals are one `REFUSED: <CODE>: <detail>` line on stderr, exit 1:
# 1. Inputs: pyproject.toml and uv.lock must exist (`LOCK_INPUT_MISSING`); the exclude-newer
#    timestamp file uv.lock.exclude-newer must exist (`LOCK_TIMESTAMP_MISSING`) and hold one
#    valid RFC 3339 UTC instant not in the future (`LOCK_TIMESTAMP_MALFORMED`).
# 2. Tools: uv must be on PATH or in UV_BIN (`PREFLIGHT_TOOL_MISSING`) and must match the
#    pinned version 0.13.5 (`UV_VERSION_MISMATCH`).
# 3. Resolution: `uv lock --exclude-newer <timestamp>` is run in a clean staging directory
#    without uv.lock; a failed compile is `LOCK_COMPILE_FAILED`, and a compile whose normalised
#    output differs from the committed uv.lock is `LOCK_DRIFT`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=scripts/colab/lock_common.sh
source "${SCRIPT_DIR}/lock_common.sh"

refuse() {
  local code="$1"
  shift
  echo "REFUSED: ${code}: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    uv|default)
      shift
      ;;
    *)
      echo "Usage: $0" >&2
      exit 2
      ;;
  esac
done

PYPROJECT_FILE="${ROLESET_PREFLIGHT_PYPROJECT_FILE:-${REPO_ROOT}/${PYPROJECT_REL}}"
LOCK_FILE="${ROLESET_PREFLIGHT_LOCK_FILE:-${REPO_ROOT}/${LOCK_REL}}"
TS_FILE="${ROLESET_PREFLIGHT_EXCLUDE_NEWER_FILE:-${REPO_ROOT}/${TS_REL}}"

[[ -f "${PYPROJECT_FILE}" ]] || refuse LOCK_INPUT_MISSING "${PYPROJECT_FILE} not found"
[[ -f "${LOCK_FILE}" ]] || refuse LOCK_INPUT_MISSING "${LOCK_FILE} not found"
EXCLUDE_NEWER="$(lock_read_exclude_newer "${TS_FILE}")" || exit 1

UV_BIN="${UV_BIN:-$(command -v uv || true)}"
[[ -n "${UV_BIN}" && -x "${UV_BIN}" ]] || refuse PREFLIGHT_TOOL_MISSING "uv not found (PATH or UV_BIN)"
lock_check_uv "${UV_BIN}"

TMP_LOCK="$(mktemp)"
COMPILE_LOG="$(mktemp)"
NORM_COMMITTED="$(mktemp)"
NORM_COMPILED="$(mktemp)"
trap 'rm -f "${TMP_LOCK}" "${COMPILE_LOG}" "${NORM_COMMITTED}" "${NORM_COMPILED}"' EXIT

if ! lock_run_compile "${UV_BIN}" "${PYPROJECT_FILE}" "${TMP_LOCK}" "${EXCLUDE_NEWER}" >"${COMPILE_LOG}" 2>&1; then
  echo "REFUSED: LOCK_COMPILE_FAILED: uv lock failed (exclude-newer ${EXCLUDE_NEWER}); last lines:" >&2
  tail -n 15 "${COMPILE_LOG}" >&2 || true
  exit 1
fi

lock_normalise "${LOCK_FILE}" > "${NORM_COMMITTED}"
lock_normalise "${TMP_LOCK}" > "${NORM_COMPILED}"

if ! diff -u "${NORM_COMMITTED}" "${NORM_COMPILED}" >/dev/null 2>&1; then
  echo "REFUSED: LOCK_DRIFT: compiled output differs from committed ${LOCK_FILE} (exclude-newer ${EXCLUDE_NEWER} from ${TS_FILE})" >&2
  diff -u "${NORM_COMMITTED}" "${NORM_COMPILED}" | head -n 40 >&2 || true
  exit 1
fi

PKG_COUNT="$(grep -c '^\[\[package\]\]' "${NORM_COMMITTED}" || true)"
echo "PREFLIGHT_OK lock=uv.lock exclude_newer=${EXCLUDE_NEWER} packages=${PKG_COUNT}"
