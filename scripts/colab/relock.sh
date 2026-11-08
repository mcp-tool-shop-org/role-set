#!/usr/bin/env bash
# Regenerate uv.lock and its exclude-newer timestamp file (uv.lock.exclude-newer) together.
# Usage: bash scripts/colab/relock.sh [uv] --exclude-newer <YYYY-MM-DDTHH:MM:SSZ|now> [--locks-dir <dir>]
#
# Runs the same bounded `uv lock` compile that preflight_resolve.sh runs (scripts/colab/lock_common.sh),
# bounded by `--exclude-newer`, and writes uv.lock and uv.lock.exclude-newer only after the compile
# succeeds. `now` takes the current UTC second. A bump is a deliberate PR that moves the timestamp
# and re-locks (docs/colab-locks.md). --locks-dir writes both files to another directory instead.
#
# Refusals are one `REFUSED: <CODE>: <detail>` line on stderr, exit 1: `LOCK_TIMESTAMP_MISSING`,
# `LOCK_TIMESTAMP_MALFORMED`, `PREFLIGHT_TOOL_MISSING`, `LOCK_INPUT_MISSING`, `UV_VERSION_MISMATCH`,
# `LOCK_COMPILE_FAILED`. It makes no `colab` call.
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

usage() {
  echo "Usage: $0 [uv] --exclude-newer <YYYY-MM-DDTHH:MM:SSZ|now> [--locks-dir <dir>]" >&2
  exit 2
}

[[ $# -gt 0 ]] || usage
EXCLUDE_NEWER=""
LOCKS_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    uv|default)
      shift
      ;;
    --exclude-newer)
      [[ $# -ge 2 ]] || usage
      EXCLUDE_NEWER="$2"
      shift 2
      ;;
    --locks-dir)
      [[ $# -ge 2 ]] || usage
      LOCKS_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      ;;
  esac
done

[[ -n "${EXCLUDE_NEWER}" ]] || refuse LOCK_TIMESTAMP_MISSING "relock needs --exclude-newer <YYYY-MM-DDTHH:MM:SSZ|now>"
if [[ "${EXCLUDE_NEWER}" == "now" ]]; then
  EXCLUDE_NEWER="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi
lock_check_exclude_newer "${EXCLUDE_NEWER}" || refuse LOCK_TIMESTAMP_MALFORMED "--exclude-newer ${LOCK_TS_REASON}"

UV_BIN="${UV_BIN:-$(command -v uv || true)}"
[[ -n "${UV_BIN}" && -x "${UV_BIN}" ]] || refuse PREFLIGHT_TOOL_MISSING "uv not found (PATH or UV_BIN)"
PYPROJECT_FILE="${ROLESET_PREFLIGHT_PYPROJECT_FILE:-${REPO_ROOT}/${PYPROJECT_REL}}"
[[ -f "${PYPROJECT_FILE}" ]] || refuse LOCK_INPUT_MISSING "${PYPROJECT_FILE} not found"
lock_check_uv "${UV_BIN}"

OUT_DIR="${LOCKS_DIR:-${REPO_ROOT}}"
[[ -d "${OUT_DIR}" ]] || refuse LOCK_INPUT_MISSING "locks directory ${OUT_DIR} not found"
LOCK_OUT="${OUT_DIR}/${LOCK_REL}"
TS_OUT="${OUT_DIR}/${TS_REL}"

TMP_LOCK="$(mktemp "${OUT_DIR}/.uv.lock.XXXXXX")"
TMP_TS="$(mktemp "${OUT_DIR}/.uv.lock.exclude-newer.XXXXXX")"
COMPILE_LOG="$(mktemp)"
trap 'rm -f "${TMP_LOCK}" "${TMP_TS}" "${COMPILE_LOG}"' EXIT

if ! lock_run_compile "${UV_BIN}" "${PYPROJECT_FILE}" "${TMP_LOCK}" "${EXCLUDE_NEWER}" >"${COMPILE_LOG}" 2>&1; then
  echo "REFUSED: LOCK_COMPILE_FAILED: uv lock failed (exclude-newer ${EXCLUDE_NEWER}); last lines:" >&2
  tail -n 15 "${COMPILE_LOG}" >&2 || true
  exit 1
fi

{
  echo "# uv --exclude-newer bound for uv.lock: the resolver sees only files uploaded before this"
  echo "# instant (RFC 3339, UTC, whole seconds). Written with the lock by scripts/colab/relock.sh; it"
  echo "# moves only in a deliberate re-lock PR (docs/colab-locks.md)."
  echo "${EXCLUDE_NEWER}"
} > "${TMP_TS}"

chmod 644 "${TMP_LOCK}" "${TMP_TS}"
mv -f "${TMP_LOCK}" "${LOCK_OUT}"
mv -f "${TMP_TS}" "${TS_OUT}"

LOCK_SHA="$(sha256sum "${LOCK_OUT}" | awk '{print $1}')"
PKG_COUNT="$(grep -c '^\[\[package\]\]' "${LOCK_OUT}" || true)"
echo "RELOCK_OK lock=uv.lock exclude_newer=${EXCLUDE_NEWER} packages=${PKG_COUNT} lock_sha256=${LOCK_SHA}"
