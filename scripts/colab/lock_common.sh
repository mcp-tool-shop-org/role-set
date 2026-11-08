# shellcheck shell=bash
# Shared by preflight_resolve.sh (the drift check) and relock.sh (lock regeneration): the
# exclude-newer timestamp file beside uv.lock, the uv version pin (0.13.5), and the bounded
# `uv lock` compile command both run. Sourced, never executed. Callers define `refuse <CODE> <detail>`.
#
# The timestamp file (uv.lock.exclude-newer) holds the instant passed to uv's `--exclude-newer`:
# the resolver sees only files uploaded before it, so a re-resolve of a lock made under it cannot
# pick up a release published later. Format: `#` comment lines and blank lines are ignored, and
# exactly one other line must be an RFC 3339 UTC timestamp with whole seconds,
# `YYYY-MM-DDTHH:MM:SSZ`, that is a real instant and not in the future. A missing file is
# `LOCK_TIMESTAMP_MISSING`, anything else wrong is `LOCK_TIMESTAMP_MALFORMED`. See docs/colab-locks.md.

PINNED_UV_VERSION="0.13.5"
LOCK_REL="uv.lock"
TS_REL="uv.lock.exclude-newer"
PYPROJECT_REL="pyproject.toml"
# First-party runtime pin `aspire-si-gates==1.7.6` in pyproject.toml was uploaded to PyPI on
# 2026-10-29T19:56:03.97Z (#129) while the transitive dev tree in uv.lock stays bounded at
# 2026-10-07T00:00:00Z. When the global bound is earlier than 2026-10-30T00:00:00Z, pass this
# package-specific bound so a clean compile from pyproject.toml reproduces uv.lock byte for byte.
ASPIRESI_PIN_BOUND="2026-10-30T00:00:00Z"

LOCK_EXCLUDE_NEWER_RE='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$'

# lock_check_exclude_newer <text>: returns 0 for a valid bound; otherwise sets LOCK_TS_REASON, returns 1.
lock_check_exclude_newer() {
  local ts="$1" round_trip now
  LOCK_TS_REASON=""
  if [[ ! "${ts}" =~ ${LOCK_EXCLUDE_NEWER_RE} ]]; then
    LOCK_TS_REASON="'${ts}' is not an RFC 3339 UTC timestamp of the form YYYY-MM-DDTHH:MM:SSZ"
    return 1
  fi
  round_trip="$(date -u -d "${ts}" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || true)"
  if [[ "${round_trip}" != "${ts}" ]]; then
    LOCK_TS_REASON="'${ts}' is not a real date and time"
    return 1
  fi
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "${ts}" > "${now}" ]]; then
    LOCK_TS_REASON="'${ts}' is in the future (now ${now}), so a later upload would still be visible"
    return 1
  fi
  return 0
}

# lock_read_exclude_newer <file>: prints the file's timestamp, or refuses LOCK_TIMESTAMP_MISSING /
# LOCK_TIMESTAMP_MALFORMED.
lock_read_exclude_newer() {
  local file="$1"
  [[ -f "${file}" ]] || refuse LOCK_TIMESTAMP_MISSING "${file} not found; each lock needs its exclude-newer file (docs/colab-locks.md)"
  local lines=()
  mapfile -t lines < <(tr -d '\r' < "${file}" | grep -vE '^[[:space:]]*(#|$)' || true)
  if [[ "${#lines[@]}" -ne 1 ]]; then
    refuse LOCK_TIMESTAMP_MALFORMED "${file} must hold exactly one timestamp line, found ${#lines[@]}"
  fi
  lock_check_exclude_newer "${lines[0]}" || refuse LOCK_TIMESTAMP_MALFORMED "${file}: ${LOCK_TS_REASON}"
  printf '%s\n' "${lines[0]}"
}

# lock_check_uv <uv_bin>: refuses UV_VERSION_MISMATCH unless uv is the pinned version (0.13.5).
lock_check_uv() {
  local uv_bin="$1" ver_line ver
  ver_line="$("${uv_bin}" --version 2>&1 || true)"
  ver="$(awk '{print $2}' <<<"${ver_line}")"
  if [[ "${ver}" != "${PINNED_UV_VERSION}" ]]; then
    refuse UV_VERSION_MISMATCH "${uv_bin} reports '${ver_line}', role-set pins uv ${PINNED_UV_VERSION}; install uv ${PINNED_UV_VERSION} or point UV_BIN at it"
  fi
}

# lock_normalise <file>: strips CRs and strips uv's [options] / [options.exclude-newer-package]
# metadata block so the lock body matches the committed uv.lock byte for byte.
lock_normalise() {
  tr -d '\r' < "$1" | awk '
    /^\[options(\.exclude-newer-package)?\]$/ { skip = 1; next }
    skip && /^$/ { skip = 0; next }
    skip { next }
    { print }
  '
}

# lock_run_compile <uv_bin> <pyproject_file> <out_file> <exclude_newer>: runs `uv lock` from scratch
# in a clean staging directory (without any pre-existing uv.lock) bounded by `--exclude-newer`, and
# writes the normalised lock to <out_file>.
lock_run_compile() {
  local uv_bin="$1" pyproject_file="$2" out_file="$3" exclude_newer="$4"
  local stage_dir rc=0
  stage_dir="$(mktemp -d)"
  cp "${pyproject_file}" "${stage_dir}/pyproject.toml"
  if [[ -f "${REPO_ROOT}/README.md" ]]; then
    cp "${REPO_ROOT}/README.md" "${stage_dir}/README.md"
  else
    : > "${stage_dir}/README.md"
  fi
  mkdir -p "${stage_dir}/src/role_set"
  cp "${REPO_ROOT}/src/role_set/__init__.py" "${stage_dir}/src/role_set/__init__.py"
  local pkg_args=()
  if [[ "${exclude_newer}" < "${ASPIRESI_PIN_BOUND}" ]]; then
    pkg_args=(--exclude-newer-package "aspire-si-gates=${ASPIRESI_PIN_BOUND}")
  fi
  (
    cd "${stage_dir}"
    env -u UV_EXCLUDE_NEWER -u UV_EXCLUDE_NEWER_PACKAGE "${uv_bin}" lock \
      --exclude-newer "${exclude_newer}" \
      "${pkg_args[@]}"
  ) || rc=$?
  if [[ "${rc}" -eq 0 && -f "${stage_dir}/uv.lock" ]]; then
    lock_normalise "${stage_dir}/uv.lock" > "${out_file}"
  else
    rc="${rc:-1}"
    [[ "${rc}" -ne 0 ]] || rc=1
  fi
  rm -rf "${stage_dir}"
  return "${rc}"
}
