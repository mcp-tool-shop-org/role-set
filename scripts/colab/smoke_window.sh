#!/usr/bin/env bash
# Window 2 (Mike's go, 2026-10-17), from the rig in WSL Ubuntu:
#   MSYS_NO_PATHCONV=1 wsl -d Ubuntu -e bash -l scripts/colab/smoke_window.sh <window-1 llama-server> <out dir>
# One A100: upload window 1's llama-server (checked by sha256 on the VM), install the CUDA 13.4 runtime wheels,
# pin the libraries it loads, convert both Colab models to Q8_0 at pinned revisions (Lightning must reproduce window 1's
# hash), smoke-serve each with logprobs, then fetch the results. Unit cap and wall clock enforced; the VM is stopped in
# a trap and the sessions list checked empty.
# The live-rate cap guard (R&D, 2026-11-06): Colab's balance moves only after a session ends, so the cap runs on the
# session's own live rate (the account rate after colab new less the rate before). `colab usage` is account-wide, so
# the own delta is taken only when it is stable: two consecutive reads PIN_RATE_POLL_S apart that agree within 0.01 and
# sit at or above GPU_RATE_FLOOR, within PIN_RATE_POLLS polls (plus the one extra read stability needs). The watcher
# re-reads it every cap check (at most 60 s apart on a GPU window): a rise moves the deadline to the highest rate seen,
# and a rate above the max ends the window as a cap trip (RATE_ABOVE_MAX). Every reading goes to rate.json.
# A change in the "Active assignments" count (a neighbour opening or closing) re-baselines once two reads at the new
# count agree; a count missing or 0 for 3 reads in a row ends the window (ASSIGNMENTS_UNREADABLE). Known limit: a
# neighbour opening in the same read our own rate rises can fold that rise into the baseline; pricing at
# max(max rate, highest seen) bounds it, and a window's rate is set at assignment.
# Known limit: a Colab CLI outage that fails 3 consecutive usage reads (about 3 min at 60 s a cycle) ends the window as
# ASSIGNMENTS_UNREADABLE; it stops early and adds no spend.
# Known limit: a neighbour whose assignment count leads its rate by 2 or more reads (a registration lag of 2+ minutes)
# can still falsely trip RATE_ABOVE_MAX; unlikely, and it adds no spend.
# A GPU window names its worst-case rate with --max-rate or PIN_MAX_RATE (no default); an own rate that stays unread
# refuses RATE_UNREAD and one above the max refuses RATE_ABOVE_MAX, each right after the open, and the trap stops the
# session. There is no fallback to a constant rate on a GPU card. Each window's rate.json records the reading.
# Retired: the window-2 draft of this script (a balance-only cap, never committed) ran windows 2 to 7 on 2026-10-17,
# and balance-only copies of it ran 7b and 8 that day; nothing since. It must not be run again; this file replaces it.
set -uo pipefail
SHAPE_FLAG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --max-rate) PIN_MAX_RATE="${2:-}"; shift; [ $# -gt 0 ] && shift;;
    --max-rate=*) PIN_MAX_RATE="${1#--max-rate=}"; shift;;
    --shape) SHAPE_FLAG="${2:-}"; shift; [ $# -gt 0 ] && shift;;
    --shape=*) SHAPE_FLAG="${1#--shape=}"; shift;;
    --check-only) CHECK_ONLY=1; shift;;  # run the checks that come before any colab call, then exit 0
    *) break;;
  esac
done
BIN="${1:?usage: smoke_window.sh <llama-server> <out dir>}"
OUT="${2:?usage: smoke_window.sh <llama-server> <out dir>}"
if [ -n "$SHAPE_FLAG" ]; then
  case "$SHAPE_FLAG" in
    standard) PIN_HIGH_MEM=0;;
    high-mem) PIN_HIGH_MEM=1;;
    *) echo "REFUSED SHAPE_INVALID: --shape=$SHAPE_FLAG must be standard or high-mem (or set PIN_HIGH_MEM=0|1)" >&2; exit 2;;
  esac
fi
case "${PIN_GPU:-A100}" in
  A100|H100|G4|L4|T4|V5E1|V6E1|CPU) ;;
  *) echo "REFUSED: PIN_GPU=${PIN_GPU} must be A100, H100, G4, L4, T4 or CPU (or V5E1, V6E1)" >&2; exit 2;;
esac
if [ "${PIN_STUB:-0}" = 1 ] && [ "${PIN_GPU:-A100}" != CPU ]; then
  echo "REFUSED: PIN_STUB=1 may only run on PIN_GPU=CPU, never on ${PIN_GPU:-A100}" >&2
  exit 2
fi
# PIN_STUB_CALL_S slows the stub's calls for a forced mid-wave deadline (R&D on #91): a test knob, never on a GPU window.
if [ -n "${PIN_STUB_CALL_S:-}" ] && { [ "${PIN_GPU:-A100}" != CPU ] || [ "${PIN_STUB:-0}" != 1 ]; }; then
  echo "REFUSED: PIN_STUB_CALL_S is a stub rung's setting; never on ${PIN_GPU:-A100} (requires PIN_STUB=1 PIN_GPU=CPU)" >&2
  exit 2
fi
if [ -n "${PIN_STUB_DROP_AFTER_CALLS:-}" ] && { [ "${PIN_GPU:-A100}" != CPU ] || [ "${PIN_STUB:-0}" != 1 ]; }; then
  echo "REFUSED: PIN_STUB_DROP_AFTER_CALLS is a stub rung's setting; never on ${PIN_GPU:-A100} (requires PIN_STUB=1 PIN_GPU=CPU)" >&2
  exit 2
fi
if [ "${PIN_GPU:-A100}" != CPU ] && [ "${PIN_GATE:-1}" != 0 ] && [ -z "${PIN_GATE_MIN_DISK_GB:-}" ]; then
  # The gate's disk floor is the window's computed peak + 20 GB; without it a too-small VM is found only after the token
  # and the paid time (R&D on 346ea2d). A CPU window may leave it at its default.
  echo "REFUSED: a ${PIN_GPU:-A100} window with the VM gate on needs PIN_GATE_MIN_DISK_GB (its peak + 20 GB)" >&2
  exit 2
fi
if [ -n "${PIN_HF_TOKEN_FROM:-}" ]; then
  # Scoped tokens only (R&D, 2026-10-18 / 2026-10-23): a pilot/critic-eval window reads with HF_TOKEN_GGUF_READ, a
  # prebuild writes with HF_TOKEN_GGUF_WRITE, and an offrig-bench mirror window reads with HF_TOKEN_OLLAMA_PINS_READ,
  # each fine-grained to the one private repo. The account-wide HF_TOKEN never goes to a VM. Checked before any VM is
  # asked for; presence only, never the value.
  case "${PIN_ROLE:-}:$PIN_HF_TOKEN_FROM" in
    pilot:HF_TOKEN_GGUF_READ|critic-eval:HF_TOKEN_GGUF_READ|batching-drift:HF_TOKEN_GGUF_READ|toolcount:HF_TOKEN_GGUF_READ|prebuild:HF_TOKEN_GGUF_WRITE|offrig-bench:HF_TOKEN_OLLAMA_PINS_READ) ;;
    *) echo "REFUSED: PIN_ROLE=${PIN_ROLE:-unset} may not send $PIN_HF_TOKEN_FROM (pilot/critic-eval/batching-drift/toolcount: HF_TOKEN_GGUF_READ; prebuild: HF_TOKEN_GGUF_WRITE; offrig-bench: HF_TOKEN_OLLAMA_PINS_READ)"; exit 2;;
  esac
  tokvar=$PIN_HF_TOKEN_FROM
  [ -n "${!tokvar:-}" ] || { echo "PIN_HF_TOKEN_FROM names $tokvar, which is empty on the rig"; exit 3; }
fi
CAP_UNITS="${PIN_CAP_UNITS:-8}"
WALL_S="${PIN_WALL_S:-5400}"
S=${PIN_SESSION:-pinw2}
OPEN_LOCK="${PIN_OPEN_LOCK:-/tmp/colab-window-open.lock}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"
rm -f "$OUT/.exec.pid" "$OUT/.upload.pid" "$OUT/.end" "$OUT/.end.ts" "$OUT/.final_pulled" "$OUT/job.log" \
  "$OUT/.retry" "$OUT/.retried" "$OUT/.session_open" "$OUT/.stop_completed" "$OUT/.stop_result" "$OUT/.stop_path" "$OUT/stop.log"
JOB="$here/${PIN_JOB_SCRIPT:-pin_window.py}"  # mock_job.py for a test of this script on a cheap runtime
MARGIN="${PIN_CAP_MARGIN:-0.3}"  # units kept back at the cap to fetch the results and stop the VM (R&D, 2026-10-17 / 2026-11-11)
DL_S="${PIN_DL_TIMEOUT:-300}"
COPYBACK_DL_S="${PIN_COPYBACK_DL_S:-60}"
# Every colab call runs under `timeout -k "$KILL_GRACE_S" <limit>` so a client ignoring SIGTERM is killed after the grace.
KILL_GRACE_S="${PIN_KILL_GRACE_S:-10}"
OPEN_LOCK_S="${PIN_OPEN_LOCK_TIMEOUT_S:-180}"
STOP_LOCK_S="${PIN_STOP_LOCK_TIMEOUT_S:-60}"
NEW_S="${PIN_NEW_TIMEOUT_S:-300}"
STOP_S="${PIN_STOP_TIMEOUT_S:-60}"
STOP_ATTEMPTS=2
STOP_SETTLE_S="${PIN_STOP_SETTLE_S:-5}"
GATE_SETTLE_S="${PIN_GATE_SETTLE_S:-15}"
SESSIONS_S="${PIN_SESSIONS_TIMEOUT_S:-30}"
STATUS_S="${PIN_STATUS_TIMEOUT_S:-60}"
UPLOAD_PART_S="${PIN_UPLOAD_TIMEOUT_S:-600}"
UPLOAD_TOKEN_S="${PIN_TOKEN_UPLOAD_TIMEOUT_S:-120}"
# The margin's budget (R&D on #91, extended with bounded stop): the job's deadline is cap - MARGIN; it gets GRACE_S past
# it to cut its wave, tar and print its result, the exit fetch needs FETCH_ALLOW_S after that, and stopping the session
# (STOP_ATTEMPTS tries of STOP_S + KILL_GRACE_S = STOP_BOUND_S) must also fit before the ceiling at the cap. So
# CLEANUP_BOUND_S = GRACE_S + FETCH_ALLOW_S + STOP_BOUND_S must fit in MARGIN at the window's worst-case rate, and CAP_UNITS
# must exceed MARGIN, or the window refuses before colab new: PIN_MAX_RATE on a GPU card (rates move: the A100 High-RAM
# shape read 8.88/hr and the A100 6.73/hr on 2026-11-03), the configured stub rate on a CPU one.
# Defaults from the measurements (R&D, 2026-10-18 / sdlab #66): 60 s grace + 120 s fetch + 140 s bounded stop (2 x 70 s)
# = 320 s. A GPU window derives its margin from them at its max rate (below); an explicit PIN_CAP_MARGIN may raise it,
# never lower it.
GRACE_S="${PIN_DEADLINE_GRACE_S:-60}"
FETCH_ALLOW_S="${PIN_FETCH_ALLOW_S:-120}"
# PIN_HIGH_MEM is required on every GPU card (no default), and cross-checked against single-shape high-mem cards
# (L4, V5E1, V6E1 are always High-RAM). Only CPU stubs default PIN_HIGH_MEM to 0.
if [ "${PIN_GPU:-A100}" = CPU ]; then
  PIN_HIGH_MEM="${PIN_HIGH_MEM:-0}"
  if [ "$PIN_HIGH_MEM" != 0 ] && [ "$PIN_HIGH_MEM" != 1 ]; then
    echo "REFUSED SHAPE_INVALID: PIN_HIGH_MEM=$PIN_HIGH_MEM must be 0 (Standard) or 1 (High-RAM)" >&2
    exit 2
  fi
else
  if [ -z "${PIN_HIGH_MEM:-}" ]; then
    echo "REFUSED SHAPE_REQUIRED: a ${PIN_GPU:-A100} window requires PIN_HIGH_MEM=0 (Standard) or PIN_HIGH_MEM=1 (High-RAM) (or --shape standard|high-mem); there is no default" >&2
    exit 2
  fi
  if [ "$PIN_HIGH_MEM" != 0 ] && [ "$PIN_HIGH_MEM" != 1 ]; then
    echo "REFUSED SHAPE_INVALID: PIN_HIGH_MEM=$PIN_HIGH_MEM must be 0 (Standard) or 1 (High-RAM)" >&2
    exit 2
  fi
  case "${PIN_GPU:-A100}:$PIN_HIGH_MEM" in
    L4:0|V5E1:0|V6E1:0)
      echo "REFUSED SHAPE_STANDARD_UNAVAILABLE: PIN_GPU=${PIN_GPU:-A100} is always high-mem (High-RAM); Standard (PIN_HIGH_MEM=0) is unavailable; set PIN_HIGH_MEM=1" >&2
      exit 2;;
  esac
fi
SHAPE_EXPECTED="Standard"; [ "$PIN_HIGH_MEM" = 1 ] && SHAPE_EXPECTED="High-RAM"
SHAPE_OBS=""
# The cards a rung has run (G4, A100 in both shapes). Only the CPU stubs keep a configured rate (measured 0.08, --high-mem
# 0.26), used when their own rate is unread and always labelled "configured_cpu" where it is recorded.
case "${PIN_GPU:-A100}:$PIN_HIGH_MEM" in
  G4:*|A100:*) CPU_RATE="";;
  CPU:1) CPU_RATE=0.26;;
  CPU:0) CPU_RATE=0.08;;
  *) echo "REFUSED: PIN_GPU=${PIN_GPU:-A100} (PIN_HIGH_MEM=$PIN_HIGH_MEM) has no measured rate yet; refuse until one is measured" >&2; exit 2;;
esac
DEFAULT_SERVER_SHA256="b4659b3fc74037deb96b2148c455753121fa21dde174d7124547e2bf9ffd37c3"
if [ "${PIN_SERVER_SHA256+set}" = set ]; then
  if ! printf '%s' "$PIN_SERVER_SHA256" | grep -qE '^[0-9a-f]{64}$'; then
    echo "REFUSED: PIN_SERVER_SHA256=${PIN_SERVER_SHA256} must be 64 lowercase hex characters" >&2
    exit 2
  fi
fi
WANT_SERVER_SHA256="${PIN_SERVER_SHA256:-$DEFAULT_SERVER_SHA256}"
if [ "${PIN_SKIP_BINARY:-0}" != 1 ] && [ "${PIN_ROLE:-}" != offrig-bench ]; then
  if [ ! -f "$BIN" ]; then
    echo "REFUSED: binary $BIN does not exist" >&2
    exit 2
  fi
  GOT_SERVER_SHA256=$(sha256sum "$BIN" | awk '{print $1}')
  if [ "$GOT_SERVER_SHA256" != "$WANT_SERVER_SHA256" ]; then
    echo "REFUSED: llama-server $BIN sha256 $GOT_SERVER_SHA256 != expected $WANT_SERVER_SHA256" >&2
    exit 2
  fi
fi
MAX_RATE="${PIN_MAX_RATE:-}"
if [ -n "$MAX_RATE" ] && ! awk -v r="$MAX_RATE" 'BEGIN{exit !(r ~ /^[0-9]+(\.[0-9]+)?$/ && r + 0 > 0)}'; then
  echo "REFUSED: PIN_MAX_RATE=$MAX_RATE must be a positive number of units/hr" >&2
  exit 2
fi
if [ "${PIN_GPU:-A100}" != CPU ] && [ -z "$MAX_RATE" ]; then
  echo "REFUSED: a ${PIN_GPU:-A100} window needs --max-rate or PIN_MAX_RATE (the most units/hr it may run at; no default)" >&2
  exit 2
fi
USAGE_T="${PIN_USAGE_TIMEOUT_S:-30}"
GATE_TRIES="${PIN_GATE_TRIES:-3}"
GATE_S="${PIN_GATE_TIMEOUT_S:-180}"
for _bound_var in WALL_S DL_S COPYBACK_DL_S GRACE_S FETCH_ALLOW_S KILL_GRACE_S OPEN_LOCK_S STOP_LOCK_S NEW_S STOP_S \
    SESSIONS_S STATUS_S UPLOAD_PART_S UPLOAD_TOKEN_S USAGE_T GATE_TRIES GATE_S; do
  case "${!_bound_var}" in
    ''|*[!0-9]*|0) echo "REFUSED: $_bound_var=${!_bound_var} must be a whole number of 1 or more" >&2; exit 2;;
  esac
done
for _nonneg_var in STOP_SETTLE_S GATE_SETTLE_S; do
  case "${!_nonneg_var}" in
    ''|*[!0-9]*) echo "REFUSED: $_nonneg_var=${!_nonneg_var} must be a non-negative whole number" >&2; exit 2;;
  esac
done
if ! awk -v c="$CAP_UNITS" 'BEGIN{exit !(c ~ /^[0-9]*\.?[0-9]+$/ && c + 0 > 0)}'; then
  echo "REFUSED: PIN_CAP_UNITS=$CAP_UNITS must be a positive number of units" >&2
  exit 2
fi
STOP_BOUND_S=$(( STOP_ATTEMPTS * (STOP_S + KILL_GRACE_S) ))
CLEANUP_BOUND_S=$(( GRACE_S + FETCH_ALLOW_S + STOP_BOUND_S ))
MARGIN_RATE="${MAX_RATE:-$CPU_RATE}"
# A GPU window's margin is derived (Publisher, 2026-11-06; extended with bounded stop, sdlab #66): the units the grace,
# the fetch and the bounded `colab stop` (with its retry) cost at the max rate, rounded up to 0.01. An explicit
# PIN_CAP_MARGIN may raise it, never lower it. A CPU stub keeps its margin as before.
MARGIN_DERIVED=""
if [ "${PIN_GPU:-A100}" != CPU ]; then
  MARGIN_DERIVED=$(awk -v b="$CLEANUP_BOUND_S" -v r="$MAX_RATE" \
    'BEGIN{x = b * r / 36; c = int(x); if (x - c > 1e-9) c++; printf "%.2f", c / 100}')
  if [ -n "${PIN_CAP_MARGIN:-}" ]; then
    if ! awk -v m="$PIN_CAP_MARGIN" 'BEGIN{exit !(m ~ /^[0-9]*\.?[0-9]+$/)}'; then
      echo "REFUSED: PIN_CAP_MARGIN=$PIN_CAP_MARGIN must be a number of units" >&2; exit 2
    fi
    if awk -v m="$PIN_CAP_MARGIN" -v d="$MARGIN_DERIVED" 'BEGIN{exit !(m + 0 < d - 1e-9)}'; then
      echo "REFUSED MARGIN_TOO_SMALL: PIN_CAP_MARGIN=$PIN_CAP_MARGIN is below the derived margin $MARGIN_DERIVED (grace ${GRACE_S}s + fetch ${FETCH_ALLOW_S}s + stop ${STOP_BOUND_S}s = ${CLEANUP_BOUND_S}s at the max rate $MAX_RATE/hr)" >&2
      exit 2
    fi
    MARGIN=$PIN_CAP_MARGIN
  else
    MARGIN=$MARGIN_DERIVED
  fi
else
  if ! awk -v m="$MARGIN" 'BEGIN{exit !(m ~ /^[0-9]*\.?[0-9]+$/)}'; then
    echo "REFUSED: PIN_CAP_MARGIN=$MARGIN must be a number of units" >&2; exit 2
  fi
fi
# Rounded, not truncated: 0.30 units at 6/hr is 180 s, which %d made 179 (R&D review of #132). A margin derived above
# fits by construction, so the check binds only an explicit or a CPU margin.
BUDGET_S=$(awk -v m="$MARGIN" -v r="$MARGIN_RATE" 'BEGIN{printf "%d", m/r*3600 + 0.5}')
if [ "$MARGIN" != "$MARGIN_DERIVED" ] && [ "$CLEANUP_BOUND_S" -gt "$BUDGET_S" ]; then
  echo "REFUSED: grace ${GRACE_S}s + fetch ${FETCH_ALLOW_S}s + stop ${STOP_BOUND_S}s don't fit the margin: $MARGIN units at $MARGIN_RATE/hr is ${BUDGET_S}s; raise PIN_CAP_MARGIN or shorten the grace" >&2
  exit 2
fi
if awk -v c="$CAP_UNITS" -v m="$MARGIN" 'BEGIN{exit !(c + 0 <= m + 0 + 1e-9)}'; then
  echo "REFUSED CAP_TOO_SMALL_FOR_CLEANUP: PIN_CAP_UNITS=$CAP_UNITS cannot hold the margin $MARGIN (grace ${GRACE_S}s + fetch ${FETCH_ALLOW_S}s + stop ${STOP_BOUND_S}s = ${CLEANUP_BOUND_S}s at ${MARGIN_RATE}/hr)" >&2
  exit 2
fi
# 4 polls x 5 s = 20 s, as rnd's rates.sh; stability takes one read more, so at most RATE_POLLS + 1 reads.
RATE_POLLS="${PIN_RATE_POLLS:-4}"; RATE_POLL_S="${PIN_RATE_POLL_S:-5}"
case "$RATE_POLLS" in ''|*[!0-9]*|0) echo "REFUSED: PIN_RATE_POLLS=$RATE_POLLS must be a whole number of 1 or more" >&2; exit 2;; esac
RATE_READS_MAX=$(( RATE_POLLS + 1 ))
[ "${PIN_GPU:-A100}" = CPU ] && RATE_READS_MAX=1  # a CPU stub reads once, as before, and keeps its labelled configured rate
# Between the cheapest GPU card measured, the T4 at 1.07/hr (16 reads in window windowC2-12-t4-s1a-20261027), and the
# highest CPU rate, 0.26/hr (--high-mem): a smaller own delta is a session not yet registered, or a neighbour's change
# (a parallel CPU session), not this card's rate.
GPU_RATE_FLOOR=1.0
if [ "${CHECK_ONLY:-0}" = 1 ]; then
  echo "CHECK OK: margin $MARGIN (derived ${MARGIN_DERIVED:-n/a}) at ${MARGIN_RATE}/hr is ${BUDGET_S}s for grace ${GRACE_S}s + fetch ${FETCH_ALLOW_S}s + stop ${STOP_BOUND_S}s (${CLEANUP_BOUND_S}s)"
  exit 0
fi
export COLUMNS=150
# google-colab-cli 0.7.4 has no CLI flag or env var to disable its PyPI update banner (colab_cli/auto_update.py
# announce_upgrade / run_background_check read SettingsStore at ~/.config/colab-cli/settings.json). Set
# "enable_update_check": false in settings.json before any colab call, and strip any banner lines before cutting status.
COLAB_SETTINGS="${COLAB_SETTINGS_FILE:-$HOME/.config/colab-cli/settings.json}"
mkdir -p "$(dirname "$COLAB_SETTINGS")"
python3 -c '
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
try:
    d = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    if not isinstance(d, dict): d = {}
except Exception:
    d = {}
if d.get("enable_update_check") is not False:
    d["enable_update_check"] = False
    p.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
' "$COLAB_SETTINGS"
strip_colab_banner() {
  grep -v -E '^\[colab\] (A new version|Run .* to update|You can run .*colab update|To silence this check)' || true
}
scrub_log() {
  sed -u -E \
    -e 's/(colab-runtime-proxy-token|access_token)=[^&"'\''[:space:])]+/\1=[REDACTED]/g' \
    -e 's/([Bb]earer[[:space:]]+)[A-Za-z0-9._-]+/\1REDACTED/g' \
    -e 's/eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/[REDACTED]/g' \
    -e 's/hf_[A-Za-z0-9]{6,}/[REDACTED]/g'
}
usage() { ( trap - TERM INT; exec timeout -k "$KILL_GRACE_S" "$USAGE_T" colab usage </dev/null 2>&1 ); }  # every usage call is bounded
bal() { usage | strip_colab_banner | grep -oP 'balance: \K[0-9.]+' || true; }
# PIN_COLAB_CONFIG is optional (used by the orphan helper's pattern), with no default private copy:
# `colab download` and `colab sessions` may write ~/.config/colab-cli/sessions.json through the CLI
# (google-colab-cli 0.7.4 common.py:86-112), which holds its own exclusive filelock.ReadWriteLock
# (state.py:59, 135-136). A private copy would let token state drift between the copy and the shared
# store. Studio rule: our code never writes or edits sessions.json directly; the CLI may.
cli() {
  local cfg_args=()
  [ -n "${PIN_COLAB_CONFIG:-}" ] && cfg_args=(--config "$PIN_COLAB_CONFIG")
  ( trap - TERM INT; exec timeout -k "$KILL_GRACE_S" "${CLI_T:-$DL_S}" colab "${cfg_args[@]}" "$@" </dev/null 2>&1 ) | strip_colab_banner | scrub_log
  return "${PIPESTATUS[0]}"
}
# Single bounded `colab stop` helper: every `colab stop` in this file goes through here, bounded by
# `timeout -k "$KILL_GRACE_S" "$STOP_S"` with one retry (`STOP_ATTEMPTS=2`).
stop_session() {  # $1: caller (trap, gate, watcher, stub_drop)
  local caller="${1:-trap}" attempt=1 out="" rc=0
  while [ "$attempt" -le "$STOP_ATTEMPTS" ]; do
    out=$( ( trap - TERM INT; exec timeout -k "$KILL_GRACE_S" "$STOP_S" colab stop -s "$S" </dev/null 2>&1 ) ) && rc=0 || rc=$?
    printf '%s %s\n' "$attempt" "$rc" > "$OUT/.stop_result"
    [ "$caller" != stub_drop ] && printf '%s\n' "$caller" > "$OUT/.stop_path"
    if [ -n "$out" ]; then
      printf '%s\n' "$out" | strip_colab_banner | tail -1 | scrub_log || true
    fi
    if [ "$rc" -eq 0 ]; then
      echo "STOP_COMPLETED caller=$caller attempt=$attempt/$STOP_ATTEMPTS session=$S $(date -u +%FT%TZ)" >> "$OUT/stop.log"
      [ "$caller" != stub_drop ] && touch "$OUT/.stop_completed"
      return 0
    fi
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      echo "STOP_TIMEOUT caller=$caller attempt=$attempt/$STOP_ATTEMPTS session=$S timeout_s=$STOP_S kill_grace_s=$KILL_GRACE_S rc=$rc $(date -u +%FT%TZ)" | scrub_log | tee -a "$OUT/stop.log" >&2
    else
      echo "STOP_FAILED caller=$caller attempt=$attempt/$STOP_ATTEMPTS session=$S rc=$rc $(date -u +%FT%TZ)" | scrub_log | tee -a "$OUT/stop.log" >&2
    fi
    if [ "$attempt" -lt "$STOP_ATTEMPTS" ]; then
      echo "STOP_RETRY caller=$caller attempt=$(( attempt + 1 ))/$STOP_ATTEMPTS session=$S $(date -u +%FT%TZ)" | scrub_log | tee -a "$OUT/stop.log" >&2
    fi
    attempt=$(( attempt + 1 ))
  done
  return "$rc"
}
# Stops this window's own session only: another window (or a probe) may be running beside it.
# Holds the open lock while stopping, so a neighbour's rate reads never straddle a stop (R&D, from Argon's review).
TOKF=""  # the rig-side token file, if this window sends one: removed in the trap too, so a crash can't leave it
ended=""
SESSION_OPEN=0
OPEN_LOCK_HELD=0
IN_MAIN_EXEC_WAIT=0
INTERRUPT_EXIT_RC=143
SLEEP_PID=""
WATCH=""
EXEC=""
SPENT_FINAL="null"
MAIN_PID="$$"
RATE_LIVE=""; RATE_SCOPE=""; OWN_RATE=""; T_OPEN=""; T_NEW=""; PIN_DEADLINE=""; RATE_POLLS_USED=0; RATE_WHY=""
RATE_STABLE_S=""; RATE_MAX_SEEN=""; ASSIGN0=""; RATE0=""; RATE_AFTER=""
READINGS="$OUT/rate-readings.jsonl"; rm -f "$READINGS" "$OUT/.rate_max" "$OUT/.rate_verdict" "$OUT/rate.json"
WATCHER_CHILD_PID_FILE="$OUT/.watcher_child.pid"
rm -f "$WATCHER_CHILD_PID_FILE" "$OUT/.watcher_reread.out"
KILLED_TREE_PIDS=""
# Kill a process and every descendant found by parent PID (pgrep -P), never by name: TERM, then KILL.
kill_pid_tree() {
  local root="$1" p c alive i
  local -a pids=() frontier=() next=() live_pids=()
  KILLED_TREE_PIDS=""
  [[ "$root" =~ ^[0-9]+$ ]] || return 1
  frontier=("$root")
  while [ "${#frontier[@]}" -gt 0 ]; do
    next=()
    for p in "${frontier[@]}"; do
      pids+=("$p")
      while read -r c; do
        [ -n "$c" ] && next+=("$c")
      done < <(pgrep -P "$p" 2>/dev/null || true)
    done
    frontier=("${next[@]+"${next[@]}"}")
  done
  for p in "${pids[@]}"; do
    if kill -0 "$p" 2>/dev/null && [ "$(ps -o stat= -p "$p" 2>/dev/null | tr -d ' ')" != Z ]; then
      live_pids+=("$p")
    fi
  done
  [ "${#live_pids[@]}" -gt 0 ] || return 1
  KILLED_TREE_PIDS=$(IFS=,; echo "${live_pids[*]}")
  kill -TERM "${live_pids[@]}" 2>/dev/null || true
  for i in 1 2 3 4 5 6 7 8 9 10; do
    alive=0
    for p in "${live_pids[@]}"; do
      if kill -0 "$p" 2>/dev/null && [ "$(ps -o stat= -p "$p" 2>/dev/null | tr -d ' ')" != Z ]; then
        alive=1
      fi
    done
    [ "$alive" -eq 0 ] && return 0
    sleep 0.1
  done
  kill -KILL "${live_pids[@]}" 2>/dev/null || true
  return 0
}
end_window() {
  [ -f "$OUT/.end" ] || echo "$1" > "$OUT/.end"
  ended=$1
  if [ -f "$OUT/.upload.pid" ]; then
    local upid
    upid=$(cat "$OUT/.upload.pid" 2>/dev/null || true)
    [ -n "$upid" ] && kill_pid_tree "$upid" || true
  fi
  if [ -f "$OUT/.exec.pid" ]; then
    local epid c
    epid=$(cat "$OUT/.exec.pid" 2>/dev/null || true)
    if [ -n "$epid" ]; then
      for c in $(pgrep -P "$epid" 2>/dev/null || true); do pkill -TERM -P "$c" 2>/dev/null || true; done
      pkill -TERM -P "$epid" 2>/dev/null || true
      kill -TERM "$epid" 2>/dev/null || true
      (
        sleep "${PIN_KILL_ESCALATE_S:-5}"
        for c in $(pgrep -P "$epid" 2>/dev/null || true); do pkill -KILL -P "$c" 2>/dev/null || true; done
        pkill -KILL -P "$epid" 2>/dev/null || true
        kill -KILL "$epid" 2>/dev/null || true
      ) </dev/null >/dev/null 2>&1 &
    fi
  fi
}
interruptible_sleep() {
  sleep "$1" &
  SLEEP_PID=$!
  wait "$SLEEP_PID" 2>/dev/null || true
  SLEEP_PID=""
}
on_interrupt() {
  local sig="${1:-TERM}"
  if [ "$sig" = INT ]; then INTERRUPT_EXIT_RC=130; else INTERRUPT_EXIT_RC=143; fi
  [ -f "$OUT/.end.ts" ] || date +%s > "$OUT/.end.ts"
  end_window term
  [ -n "${SLEEP_PID:-}" ] && kill -TERM "$SLEEP_PID" 2>/dev/null || true
  if [ "${IN_MAIN_EXEC_WAIT:-0}" = 1 ]; then
    return 0
  fi
  exit "$INTERRUPT_EXIT_RC"
}
reap_watcher_orphans() {
  local cpid="" ckind="" w_child="" w_comm=""
  if [ -f "$WATCHER_CHILD_PID_FILE" ]; then
    read -r cpid ckind < "$WATCHER_CHILD_PID_FILE" 2>/dev/null || true
    if [ -n "$cpid" ] && kill_pid_tree "$cpid"; then
      echo "WATCHER ORPHAN KILLED: pid=$cpid kind=${ckind:-unknown} pids=$KILLED_TREE_PIDS $(date -u +%T)" | tee -a "$OUT/smoke-window.log"
    fi
    rm -f "$WATCHER_CHILD_PID_FILE" 2>/dev/null || true
  fi
  if [ -n "${WATCH:-}" ]; then
    for w_child in $(pgrep -P "$WATCH" 2>/dev/null || true); do
      w_comm=$(cat "/proc/$w_child/comm" 2>/dev/null || echo "")
      if [ "$w_comm" = sleep ]; then
        kill -TERM "$w_child" 2>/dev/null || true
      elif kill_pid_tree "$w_child"; then
        echo "WATCHER ORPHAN KILLED: pid=$w_child kind=${w_comm:-child} pids=$KILLED_TREE_PIDS $(date -u +%T)" | tee -a "$OUT/smoke-window.log"
      fi
    done
  fi
}
jnum() { if [ -n "${1:-}" ]; then printf '%s' "$1"; else printf null; fi; }
jstr() { if [ -n "${1:-}" ]; then printf '"%s"' "$1"; else printf null; fi; }
# The window receipt for the rate (rate.json) and the outcome draft both carry the reading the cap ran on.
rate_fields() {
  printf '"shape_expected": "%s", "shape_observed": %s, "rate_live": %s, "rate_scope": "%s", "rate_used": %s, "rate_max_seen": %s, "max_rate": %s, "margin_derived": %s, "margin": %s, "stop_bound_s": %s, "cleanup_bound_s": %s, "deadline": %s' \
    "$SHAPE_EXPECTED" "$(jstr "$SHAPE_OBS")" "$(jnum "$RATE_LIVE")" "$RATE_SCOPE" "$(jnum "$OWN_RATE")" "$(jnum "$RATE_MAX_SEEN")" "$(jnum "$MAX_RATE")" \
    "$(jnum "$MARGIN_DERIVED")" "$MARGIN" "$STOP_BOUND_S" "$CLEANUP_BOUND_S" "$(jnum "$PIN_DEADLINE")"
}
write_rate_receipt() {  # $1: the verdict: ok, RATE_UNREAD or RATE_ABOVE_MAX
  printf '{"verdict": "%s", "why": "%s", "gpu": "%s", "high_mem": "%s", %s, "cap": %s, "account_rate_before": %s, "account_rate_after": %s, "polls": %s, "floor": %s, "t_new": %s, "t_open": %s, "rate_stable_s": %s, "assign0": %s, "readings": [%s]}\n' \
    "$1" "$RATE_WHY" "${PIN_GPU:-A100}" "${PIN_HIGH_MEM:-0}" "$(rate_fields)" "$CAP_UNITS" "$(jnum "$RATE0")" \
    "$(jnum "$RATE_AFTER")" "$RATE_POLLS_USED" "$GPU_RATE_FLOOR" "$(jnum "$T_NEW")" "$(jnum "$T_OPEN")" \
    "$(jnum "$RATE_STABLE_S")" "$(jnum "$ASSIGN0")" "$( [ -f "$READINGS" ] && paste -sd, "$READINGS")" | scrub_log > "$OUT/rate.json"
}
outcome_draft() {  # $1: end, $2: spent (a number, or null)
  local sp="none" s_att=0 s_rc="null" draft_tmp="$OUT/.outcome.draft.tmp.$$"
  [ -f "$OUT/.stop_path" ] && sp=$(head -1 "$OUT/.stop_path" 2>/dev/null || echo "none")
  [ -n "$sp" ] || sp="none"
  if [ -f "$OUT/.stop_result" ]; then
    read -r s_att s_rc < "$OUT/.stop_result" 2>/dev/null || { s_att=0; s_rc="null"; }
    case "$s_att" in ''|*[!0-9]*) s_att=0;; esac
    case "$s_rc" in ''|*[!0-9]*) s_rc="null";; esac
  fi
  printf '{"window": "%s", "date": "%s", "commit": "%s", "card": "%s", "cap": %s, "spent": %s, "basis": "rate_x_runtime", %s, "stop_path": "%s", "stop_attempts": %s, "stop_rc": %s, "end": "%s", "outcome": null, "why": null}\n' \
    "$(basename "$OUT")" "$(date -u +%F)" "${PIN_ROLESET_COMMIT:-unset}" "${PIN_GPU:-A100}" "$CAP_UNITS" "$2" \
    "$(rate_fields)" "$sp" "$s_att" "$s_rc" "$1" | scrub_log > "$draft_tmp"
  mv -f "$draft_tmp" "$OUT/outcome.draft.json"
}
stop_all() {
  local exit_rc=$?
  trap - EXIT
  trap 'echo "CLEANUP_SIGNAL during stop_all $(date -u +%FT%TZ)" >> "$OUT/stop.log"' TERM INT
  if [ "${OPEN_LOCK_HELD:-0}" = 1 ]; then
    flock -u 9 2>/dev/null || true
    exec 9>&- 2>/dev/null || true
    OPEN_LOCK_HELD=0
  fi
  [ -n "${TOKF:-}" ] && rm -f "$TOKF"
  if [ -n "${WATCH:-}" ]; then
    reap_watcher_orphans
    kill "$WATCH" 2>/dev/null || true
    wait "$WATCH" 2>/dev/null || true
    reap_watcher_orphans
    WATCH=""
  fi
  if [ -f "$OUT/.exec.pid" ] && [ ! -f "$OUT/.final_pulled" ]; then
    touch "$OUT/.final_pulled"
    if declare -F fetch_log >/dev/null 2>&1; then
      CLI_T=30 fetch_log final || true
    fi
    if [ "${COPYBACK_S:-0}" -gt 0 ] 2>/dev/null && declare -F copyback_journals >/dev/null 2>&1; then
      DL_S=30 copyback_journals || true
    fi
  fi
  if [ "${SESSION_OPEN:-0}" = 1 ] && [ ! -f "$OUT/.stop_completed" ]; then
    (
      # Bounded: the EXIT trap can fire while another window holds the open lock (R&D on #89). After STOP_LOCK_S it stops anyway.
      flock -w "$STOP_LOCK_S" 9 || echo "open lock not free after ${STOP_LOCK_S} s; stopping $S anyway"
      echo "stopping $S"; stop_session trap || true
      sleep "$STOP_SETTLE_S"; echo "sessions after stop:"; CLI_T="$SESSIONS_S" cli sessions | tail -3 || true
    ) 9>"$OPEN_LOCK"
  fi
  local end_tag
  end_tag=$(cat "$OUT/.end" 2>/dev/null || echo "")
  if [ -n "$end_tag" ] || [ -f "$OUT/outcome.draft.json" ]; then
    [ -n "$end_tag" ] || end_tag="exec"
    if [ "$SPENT_FINAL" = "null" ] && [ -n "$OWN_RATE" ] && [ -n "$T_OPEN" ] && [ "$end_tag" != "RATE_UNREAD" ] && [ "$end_tag" != "SHAPE_MISMATCH" ]; then
      if [ "$end_tag" != "RATE_ABOVE_MAX" ] || [ -f "$OUT/.exec.pid" ] || [ -f "$OUT/.rate_verdict" ]; then
        local t_end_now
        t_end_now=$(cat "$OUT/.end.ts" 2>/dev/null || date +%s)
        SPENT_FINAL=$(awk -v r="$OWN_RATE" -v t="$(( t_end_now - T_OPEN ))" 'BEGIN{printf "%.2f", r*(t>0?t:0)/3600}')
      fi
    fi
    outcome_draft "$end_tag" "$SPENT_FINAL" || true
  fi
  return "$exit_rc"
}
trap stop_all EXIT
trap 'on_interrupt TERM' TERM
trap 'on_interrupt INT' INT
rate() { usage | strip_colab_banner | grep -oP 'rate: \K[0-9]+(\.[0-9]+)?' || true; }
# One usage read as "<rate> <assignments>", either field "-" when unread: the rate and the assignment count from the same
# call, so a neighbour's open or close is seen in the same read as the rate it moves.
rate_assign() {
  local u r n
  u=$(usage | strip_colab_banner)
  r=$(printf '%s\n' "$u" | grep -oP 'rate: \K[0-9]+(\.[0-9]+)?' | head -1 || true)
  n=$(printf '%s\n' "$u" | grep -oP 'Active assignments: \K[0-9]+' | head -1 || true)
  echo "${r:--} ${n:--}"
}
# This session's own rate: the account rate after colab new less the rate before; empty when either is unread or the
# difference is 0 or less (a neighbour closing between the reads), never a constant.
own_rate_of() { awk -v a="$1" -v b="$2" 'BEGIN{ if (a == "" || b == "") exit; r = a - b; if (r > 0) print r + 0 }'; }
record_reading() {  # $1: phase (open, run or rebaseline), $2: account rate, $3: own delta, $4: assignments, $5: baseline
  printf '{"t": %s, "phase": "%s", "account_rate": %s, "own": %s, "assignments": %s, "baseline": %s}\n' "$(date +%s)" "$1" \
    "$(jnum "$2")" "$(jnum "$3")" "$(jnum "${4:-}")" "$(jnum "${5:-}")" >> "$READINGS"
}
# The own delta of an account read, or empty: signed here, so the stability check sees a neighbour's change too.
delta_of() { awk -v a="$1" -v b="$2" 'BEGIN{ if (a == "" || b == "") exit; print a - b + 0 }'; }
rate_refuse() {  # $1: RATE_UNREAD, RATE_ABOVE_MAX or SHAPE_MISMATCH, $2: why. The EXIT trap stops the session.
  RATE_WHY=$2
  echo "$1" > "$OUT/.end"
  echo "REFUSED $1: $2; stopping $S $(date -u +%T)"
  write_rate_receipt "$1"
  outcome_draft "$1" null
  exit 5
}
START=$(bal)
sha256sum "$BIN"
GPU_ARGS=(--gpu "${PIN_GPU:-A100}"); [ "${PIN_GPU:-A100}" = CPU ] && GPU_ARGS=()
# PIN_HIGH_MEM=1 asks for the high-memory shape (a CPU conversion needs it: 8 cores and 52 GB against the default's 2 and
# 12 GB, measured 2026-10-18).
[ "${PIN_HIGH_MEM:-0}" = 1 ] && GPU_ARGS+=(--high-mem)
# The VM health gate (Mike and R&D, 2026-10-18): before anything is uploaded, vm_gate.py reads the card, disk, RAM and
# the pipe to Hugging Face; a VM that fails, or whose first exec dies or times out (windows 11c and 12e lost 5.7 units
# to first-exec deaths after the upload), is stopped and a fresh one asked for, up to PIN_GATE_TRIES. Each attempt is a
# row in gate.jsonl (readings only, no identity data). It catches VMs that are bad at the start; drops and hangs later
# in the run are the resume path's and checkpoints.py's. PIN_GATE=0 skips it.
# The Hugging Face floor scales to the window (R&D, 2026-10-19): set PIN_GATE_HF_BYTES_WINDOW (the bytes the window will
# download) and PIN_GATE_PROJECTED_S (its projected seconds); every PIN_GATE_* below reaches the gate. Without them the
# floor stays 129 MB/s; PIN_GATE_MIN_HF_MBPS overrides either.
GATE_ENV=(--env "PIN_GATE_GPU=$([ "${PIN_GPU:-A100}" = CPU ] && echo 0 || echo 1)")
for v in $(compgen -e | grep '^PIN_GATE_' | grep -v -E '^PIN_GATE_(TRIES|TIMEOUT_S|SETTLE_S|GPU)$' || true); do GATE_ENV+=(--env "$v=${!v}"); done
GATE_REJECTED_S=0
RATE0=""; RATE_AFTER=""
for try in $(seq 1 "$GATE_TRIES"); do
  t_try=$(date +%s)
  # Windows on this rig open one at a time: the rate read before, colab new and the rate read after hold one lock, so a
  # neighbour opening in between can't land in this window's rate (windows 11a and 11b each counted both sessions).
  # Bounded by OPEN_LOCK_S (default 180 s): refuses OPEN_LOCK_TIMEOUT before any colab new if held past the bound.
  # Released before GATE_SETTLE_S and vm_gate.py so vm_gate's HF download never blocks another window's open.
  exec 9>"$OPEN_LOCK"
  flock -w "$OPEN_LOCK_S" 9 &
  SLEEP_PID=$!
  lock_rc=0
  wait "$SLEEP_PID" 2>/dev/null || lock_rc=$?
  SLEEP_PID=""
  if [ "$lock_rc" -ne 0 ]; then
    exec 9>&-
    echo "REFUSED OPEN_LOCK_TIMEOUT: open lock $OPEN_LOCK not free after ${OPEN_LOCK_S}s" >&2
    echo OPEN_LOCK_TIMEOUT > "$OUT/.end"
    exit 2
  fi
  OPEN_LOCK_HELD=1
  RATE0=$(rate)
  [ "$try" = 1 ] && echo "start balance $START, account rate $RATE0/hr $(date -u +%FT%TZ)"
  SESSION_OPEN=1; touch "$OUT/.session_open"; rm -f "$OUT/.stop_completed" "$OUT/.stop_result" "$OUT/.stop_path"
  new_out=$( ( trap - TERM INT; exec timeout -k "$KILL_GRACE_S" "$NEW_S" colab new "${GPU_ARGS[@]}" -s "$S" </dev/null 2>&1 ) ) && new_rc=0 || new_rc=$?
  if [ "$new_rc" -eq 0 ]; then
    T_NEW=$(date +%s)  # billing starts here: the cap's clock does too (R&D review of #132)
    printf '%s\n' "$new_out" | strip_colab_banner | tail -2 | scrub_log
  else
    printf '%s\n' "$new_out" | strip_colab_banner | tail -2 | scrub_log
    if [ "$new_rc" -eq 124 ] || [ "$new_rc" -eq 137 ]; then
      echo "REFUSED NEW_TIMEOUT: colab new -s $S timed out after ${NEW_S}s (+${KILL_GRACE_S}s kill grace, rc=$new_rc)" >&2
      echo NEW_TIMEOUT > "$OUT/.end"
    else
      SESSION_OPEN=0; rm -f "$OUT/.session_open"
    fi
    flock -u 9 2>/dev/null || true; exec 9>&-; OPEN_LOCK_HELD=0
    echo "no ${PIN_GPU:-A100} assigned"; exit 2
  fi
  sess_out=$(CLI_T="$SESSIONS_S" cli sessions || true)
  sess_row=$(printf '%s\n' "$sess_out" | awk -v p="[$S]" '$1 == p {print; exit}')
  SHAPE_OBS=$(printf '%s\n' "$sess_row" | grep -oP 'Shape:\s*\K[^|]+' | sed 's/[[:space:]]*$//' | head -1 || true)
  if [ "$SHAPE_OBS" != "$SHAPE_EXPECTED" ]; then
    flock -u 9 2>/dev/null || true; exec 9>&-; OPEN_LOCK_HELD=0
    rate_refuse SHAPE_MISMATCH "requested PIN_HIGH_MEM=$PIN_HIGH_MEM (Shape: $SHAPE_EXPECTED) on ${PIN_GPU:-A100}, but session [$S] reports Shape: ${SHAPE_OBS:-missing}"
  fi
  # The live rate guard, still under the open lock: up to RATE_READS_MAX reads, RATE_POLL_S apart. A GPU card's own
  # delta counts only once two consecutive reads agree within 0.01 at or above GPU_RATE_FLOOR; a CPU stub reads once.
  RATE_AFTER=""; RATE_LIVE=""; RATE_POLLS_USED=0; RATE_STABLE_S=""; prev=""
  for poll in $(seq 1 "$RATE_READS_MAX"); do
    RATE_POLLS_USED=$poll
    read -r RATE_AFTER n_open < <(rate_assign)
    [ "$RATE_AFTER" = - ] && RATE_AFTER=""; [ "$n_open" = - ] && n_open=""
    d=$(delta_of "$RATE_AFTER" "$RATE0")
    record_reading open "$RATE_AFTER" "$d" "$n_open" "$RATE0"
    ASSIGN0=$n_open  # the count with our session in it, as of the read the rate was taken from
    if [ "${PIN_GPU:-A100}" = CPU ]; then
      RATE_LIVE=$(own_rate_of "$RATE_AFTER" "$RATE0"); break
    fi
    if [ -n "$prev" ] && [ -n "$d" ] && awk -v a="$prev" -v b="$d" -v f="$GPU_RATE_FLOOR" \
        'BEGIN{x = a - b; if (x < 0) x = -x; exit !(x <= 0.01 + 1e-9 && a >= f && b >= f)}'; then
      RATE_LIVE=$(awk -v a="$prev" -v b="$d" 'BEGIN{print (a > b ? a : b) + 0}')
      RATE_STABLE_S=$(( $(date +%s) - T_NEW ))
      break
    fi
    prev=$d
    [ "$poll" -lt "$RATE_READS_MAX" ] && interruptible_sleep "$RATE_POLL_S"
  done
  flock -u 9 2>/dev/null || true; exec 9>&-; OPEN_LOCK_HELD=0
  if [ -n "$RATE_LIVE" ] && [ "${PIN_GPU:-A100}" != CPU ]; then
    # Priced on max(max rate, highest stable rate seen), never on the window rate alone (Publisher, 2026-11-06): an
    # under-read rate (a late registration, a neighbour closing) can then only end the window early, never stretch it.
    RATE_SCOPE=own_delta; RATE_MAX_SEEN=$RATE_LIVE
    OWN_RATE=$(awk -v m="$MAX_RATE" -v r="$RATE_LIVE" 'BEGIN{print (r > m ? r : m) + 0}')
  elif [ -n "$RATE_LIVE" ]; then
    OWN_RATE=$RATE_LIVE; RATE_SCOPE=own_delta
  elif [ "${PIN_GPU:-A100}" = CPU ]; then
    OWN_RATE=$CPU_RATE; RATE_SCOPE=configured_cpu
  else
    RATE_SCOPE=unread
    rate_refuse RATE_UNREAD "own rate not stable at or above ${GPU_RATE_FLOOR}/hr after $RATE_POLLS_USED polls (account rate ${RATE0:-unread}/hr before the open, ${RATE_AFTER:-unread}/hr after)"
  fi
  if [ -n "$MAX_RATE" ] && [ "$RATE_SCOPE" = own_delta ] && awk -v r="$RATE_LIVE" -v m="$MAX_RATE" 'BEGIN{exit !(r > m)}'; then
    rate_refuse RATE_ABOVE_MAX "own rate $RATE_LIVE/hr is above the max rate $MAX_RATE/hr"
  fi
  sess_out=$(CLI_T="$SESSIONS_S" cli sessions || true)
  vm_name=$(printf '%s\n' "$sess_out" | awk -v p="[$S]" '$1 == p {print $2; exit}')
  vm_pool=""; vm_zone=""
  if [[ "$vm_name" =~ ^(gpu-[a-z0-9]+|m)-(s|hm)-([a-z0-9]+)-([a-z]{2,}[0-9][a-z][0-9]+)- ]]; then
    vm_pool="${BASH_REMATCH[3]}"; vm_zone="${BASH_REMATCH[4]}"
  elif [[ "$vm_name" =~ ^(gpu-[a-z0-9]+|m)-(s|hm)-([a-z]{2,}[0-9][a-z][0-9]+)- ]]; then
    vm_zone="${BASH_REMATCH[3]}"
  fi
  interruptible_sleep "$GATE_SETTLE_S"
  [ "${PIN_GATE:-1}" = 0 ] && break
  if gate_out=$( ( trap - TERM INT; exec timeout -k "$KILL_GRACE_S" "$GATE_S" colab exec -s "$S" "${GATE_ENV[@]}" -f "$here/vm_gate.py" --timeout "$GATE_S" </dev/null 2>&1 ) ); then
    line=$(printf '%s\n' "$gate_out" | grep -m1 '^VM_GATE ' || true)
  else
    line=""
  fi
  row=${line#VM_GATE }; [ -n "$line" ] || row='{"ok": false, "why": ["the gate exec died or timed out before it printed"]}'
  printf '{"try": %s, "gpu": "%s", "vm": "%s", "zone": "%s", "pool": "%s", "gate": %s}\n' \
    "$try" "${PIN_GPU:-A100}" "$vm_name" "$vm_zone" "$vm_pool" "$row" | scrub_log >> "$OUT/gate.jsonl"
  if printf '%s' "$row" | grep -q '"ok": true'; then echo "VM gate passed (try $try)"; break; fi
  echo "VM gate rejected try $try: $(printf '%s' "$row" | grep -o '"why": \[[^]]*\]' || true)" | scrub_log
  if [ "$try" = "$GATE_TRIES" ]; then
    GATE_REJECTED_S=$(( GATE_REJECTED_S + $(date +%s) - t_try ))
    echo gate_failed > "$OUT/.end"
    echo "VM GATE: no healthy ${PIN_GPU:-A100} in $GATE_TRIES tries"
    exit 4
  fi
  (
    flock -w "$STOP_LOCK_S" 9 || echo "open lock not free after ${STOP_LOCK_S} s; stopping rejected $S anyway"
    stop_session gate || true
  ) 9>"$OPEN_LOCK"
  SESSION_OPEN=0; rm -f "$OUT/.session_open" "$OUT/.stop_completed"
  interruptible_sleep "$STOP_SETTLE_S"
  GATE_REJECTED_S=$(( GATE_REJECTED_S + $(date +%s) - t_try ))
done
usage | strip_colab_banner | grep -E "rate|assign" | scrub_log || true
# The cap counts this window's own spend: its rate (the account rate after its session opened, less the rate before) times
# its own runtime (including any VMs the gate rejected). The account balance would count a neighbour's spend too (window 8
# was cut off that way).
# OWN_RATE was read live after the open (the guard above): a GPU window that couldn't read it, or read it above its max
# rate, never got here. A neighbour closing between the reads makes the difference 0 or less; that is unread, not a
# constant (R&D, 2026-11-06). Only a CPU stub falls back, to its configured rate, labelled configured_cpu.
T_OPEN=$(( T_NEW - GATE_REJECTED_S ))  # from the accepted VM's open, not after the gate and the polls
[ -n "$RATE_MAX_SEEN" ] && echo "$RATE_MAX_SEEN" > "$OUT/.rate_max"
echo "own rate ${RATE_LIVE:-$OWN_RATE}/hr ($RATE_SCOPE; max ${MAX_RATE:-none}; priced at $OWN_RATE/hr; rejected gate VMs: ${GATE_REJECTED_S}s)"
# The job's deadline (R&D, 2026-10-18): own spend at the cap less MARGIN. Past it the job starts nothing new, cuts a wave
# under way (SIGTERM, then SIGKILL) and writes pilot-out.tar and pin-result.json, so a cap stop returns results, not only
# job.log. The watcher waits PIN_DEADLINE_GRACE_S past it for the PIN_RESULT line before it kills the exec.
PIN_DEADLINE=$(awk -v t="$T_OPEN" -v c="$CAP_UNITS" -v m="$MARGIN" -v r="$OWN_RATE" 'BEGIN{printf "%d", t + (c-m)/r*3600}')
export PIN_DEADLINE
echo "deadline $(date -u -d "@$PIN_DEADLINE" +%T) (cap $CAP_UNITS less $MARGIN at $OWN_RATE/hr, $RATE_SCOPE); grace ${GRACE_S}s"
write_rate_receipt ok
# This window's token: the job writes it into pin-result.json, and a second exec stops only on this window's result.
export PIN_JOB_TOKEN="$S-$T_OPEN"
# The watcher reads the job's own log, /content/pin/job.log, fetched every PIN_WATCH_S seconds, never the exec's output
# stream: that stream stopped reaching the rig in windows 10d and 12a while the job ran on (mistakes #31). It ends the
# window, through the main script's exit sequence (fetch the results, then stop the session), when:
# - job.log carries the PIN_RESULT line: the job is done, even if the exec never returns;
# - job.log hasn't grown: nothing within 10 minutes of the exec starting, or nothing new for PIN_STALL_S (600 s) after
#   that. Every long step writes a heartbeat at most 300 s apart: run_proc's, and the in-process downloads' (written only
#   while their bytes grow, so a hung download goes silent). 1200 s until 2026-10-18 (R&D, after window 13q)
#   (R&D, 2026-10-17; the slowest silent step, a conversion, takes about 12 minutes);
# - own spend reaches the cap less MARGIN, the units kept back to fetch the results.
# Colab dropping the session ends the exec at once, with nothing to fetch. At the cap itself the session stops whatever
# is under way: the cap is the ceiling.
# job.log comes down to job.log.part and replaces the local copy only when the fetch exited 0 and the new copy is no
# smaller (R&D, 2026-10-17): a fetch that dies partway, a VM reclaimed mid-transfer or the timeout, can't truncate the
# one local copy of the checkpoint lines, which is the case they are kept for.
COPYBACK_S="${PIN_COPYBACK_S:-600}"
COPYBACK_DIR="$OUT/copyback"
COPYBACK_LOCK="${PIN_COPYBACK_LOCK:-$COPYBACK_DIR/.copyback.lock}"
COPYBACK_STATE="${PIN_COPYBACK_STATE:-$COPYBACK_DIR/.copyback.state.json}"
fetch_log() {  # $1: a name for this fetcher's part file, so the watcher's and the exit sequence's never collide
  local part="$OUT/job.log.part.${1:-x}" raw_out="$OUT/.fetch.raw.${1:-x}" cfg_args=() fpid new old dl_out dl_rc
  [ -n "${PIN_COLAB_CONFIG:-}" ] && cfg_args=(--config "$PIN_COLAB_CONFIG")
  rm -f "$part" "$raw_out"
  if [ "${1:-}" = watcher ]; then
    timeout -k "$KILL_GRACE_S" "${CLI_T:-$DL_S}" colab "${cfg_args[@]}" download -s "$S" /content/pin/job.log "$part" </dev/null >"$raw_out" 2>&1 &
    fpid=$!
    printf '%s fetch\n' "$fpid" > "$WATCHER_CHILD_PID_FILE"
    wait "$fpid" 2>/dev/null && dl_rc=0 || dl_rc=$?
    rm -f "$WATCHER_CHILD_PID_FILE"
  else
    ( trap - TERM INT; exec timeout -k "$KILL_GRACE_S" "${CLI_T:-$DL_S}" colab "${cfg_args[@]}" download -s "$S" /content/pin/job.log "$part" </dev/null >"$raw_out" 2>&1 ) && dl_rc=0 || dl_rc=$?
  fi
  dl_out=$(strip_colab_banner < "$raw_out" 2>/dev/null | scrub_log || true)
  rm -f "$raw_out"
  printf '%s\n' "$dl_out" > "$OUT/.fetch.out"
  [ "$dl_rc" -eq 0 ] || { rm -f "$part"; return 1; }
  grep -Eqi "^\[colab\] (Download failed|Error|Session .* not found)" "$OUT/.fetch.out" && { rm -f "$part"; return 1; }
  [ -f "$part" ] || return 1
  new=$(stat -c %s "$part"); old=$(stat -c %s "$OUT/job.log" 2>/dev/null || echo 0)
  if [ "$new" -ge "$old" ]; then mv -f "$part" "$OUT/job.log"; else rm -f "$part"; return 1; fi
}
# Journal copy-back (R&D, Card nh / nv): every PIN_COPYBACK_S (default 600s), pull live calls.jsonl and generations.jsonl
# under /content/pin/toolcount/<served_as>/ plus the latest heartbeat into $OUT/copyback/<served_as>/.
# Uses a private lock and state file under $OUT/copyback/ (our code never writes shared ~/.config/colab-cli/sessions.json).
copyback_journals() {
  [ "$COPYBACK_S" -gt 0 ] 2>/dev/null || return 0
  local models m jf remote dest_dir dest part dl_out dl_rc ts hb
  models=$(
    {
      grep -oE 'HEARTBEAT toolcount model=[^ ]+' "$OUT/job.log" 2>/dev/null | sed 's/.*model=//' || true
      sed -n 's/^.*=== \([^ ]*\) ===$/\1/p' "$OUT/job.log" 2>/dev/null || true
    } | sed '/^$/d' | awk '!seen[$0]++'
  )
  [ -n "$models" ] || return 0
  mkdir -p "$COPYBACK_DIR"
  (
    flock -w 30 8 || return 0
    local mypid="${BASHPID:-$$}"
    for m in $models; do
      dest_dir="$COPYBACK_DIR/$m"
      mkdir -p "$dest_dir"
      hb=$(grep -E "(^| )HEARTBEAT toolcount model=${m}( |$)" "$OUT/job.log" 2>/dev/null | tail -1 | scrub_log || true)
      [ -n "$hb" ] || hb=$(grep -E "(^| )HEARTBEAT " "$OUT/job.log" 2>/dev/null | tail -1 | scrub_log || true)
      if [ -n "$hb" ]; then
        printf '%s\n' "$hb" > "$dest_dir/heartbeat.txt.part.$mypid"
        mv -f "$dest_dir/heartbeat.txt.part.$mypid" "$dest_dir/heartbeat.txt"
        printf '%s\n' "$hb" > "$COPYBACK_DIR/heartbeat.txt.part.$mypid"
        mv -f "$COPYBACK_DIR/heartbeat.txt.part.$mypid" "$COPYBACK_DIR/heartbeat.txt"
      fi
      for jf in calls.jsonl generations.jsonl; do
        remote="/content/pin/toolcount/$m/$jf"
        dest="$dest_dir/$jf"
        part="$dest.part.$mypid"
        rm -f "$part"
        ts=$(date -u +%FT%TZ)
        dl_out=$(CLI_T="${CLI_T:-$DL_S}" cli download -s "$S" "$remote" "$part") && dl_rc=0 || dl_rc=$?
        printf '%s\n' "$dl_out" > "$COPYBACK_DIR/.copyback.fetch.out"
        if [ ! -f "$dest" ] && printf '%s\n' "$dl_out" | grep -Fq "[colab] Download failed: $remote"; then
          rm -f "$part"
          continue
        fi
        if [ "$dl_rc" -eq 0 ] && printf '%s\n' "$dl_out" | grep -Eqi "^\[colab\] (Download failed|Error|Session .* not found)"; then
          dl_rc=1
        fi
        python3 - "$part" "$dest" "$COPYBACK_STATE" "$COPYBACK_DIR/copyback.jsonl" "$OUT/copyback.jsonl" "$ts" "$m/$jf" "$dl_rc" "${hb:-}" <<'PY' || true
import json, os, pathlib, sys
part_p = pathlib.Path(sys.argv[1])
dest_p = pathlib.Path(sys.argv[2])
state_p = pathlib.Path(sys.argv[3])
log1_p = pathlib.Path(sys.argv[4])
log2_p = pathlib.Path(sys.argv[5])
ts, rel_file, dl_rc, hb = sys.argv[6], sys.argv[7], int(sys.argv[8]), sys.argv[9]

try:
    state = json.loads(state_p.read_text(encoding="utf-8")) if state_p.is_file() else {}
    if not isinstance(state, dict):
        state = {}
except Exception:
    state = {}
files_state = state.setdefault("files", {})
prev_info = files_state.get(rel_file, {}) if isinstance(files_state.get(rel_file), dict) else {}

old_lines = 0
old_bytes = 0
if dest_p.is_file():
    try:
        old_raw = dest_p.read_bytes()
        old_bytes = len(old_raw)
        old_lines = len([ln for ln in old_raw.decode("utf-8").splitlines() if ln.strip()])
    except Exception:
        old_lines = int(prev_info.get("lines", 0) or 0)
        old_bytes = int(prev_info.get("bytes", 0) or 0)
else:
    old_lines = int(prev_info.get("lines", 0) or 0)
    old_bytes = int(prev_info.get("bytes", 0) or 0)

rc = dl_rc
new_bytes = 0
new_lines = 0
if rc == 0 and part_p.is_file():
    try:
        raw = part_p.read_bytes()
        if len(raw) == 0 and old_lines == 0 and old_bytes == 0:
            try:
                part_p.unlink()
            except OSError:
                pass
            sys.exit(0)
        if len(raw) == 0:
            rc = 2
        else:
            salvaged = False
            if not raw.endswith(b"\n"):
                last_nl = raw.rfind(b"\n")
                if last_nl < 0:
                    rc = 2
                else:
                    raw = raw[: last_nl + 1]
                    salvaged = True
            if rc == 0:
                text = raw.decode("utf-8")
                non_empty = [ln for ln in text.splitlines() if ln.strip()]
                if not non_empty:
                    rc = 2
                else:
                    try:
                        last_obj = json.loads(non_empty[-1])
                        if not isinstance(last_obj, dict) or not last_obj.get("call_id"):
                            raise ValueError("bad last line")
                    except Exception:
                        if not salvaged and len(non_empty) > 1:
                            non_empty = non_empty[:-1]
                            raw = ("\n".join(non_empty) + "\n").encode("utf-8")
                            salvaged = True
                        else:
                            rc = 2
                    if rc == 0:
                        new_lines = len(non_empty)
                        new_bytes = len(raw)
                        if salvaged:
                            if new_lines <= old_lines:
                                rc = 2
                        else:
                            if new_lines == 0 or new_lines < old_lines or (new_lines == old_lines and new_bytes < old_bytes):
                                rc = 2
                    if rc == 0:
                        for ln in non_empty:
                            obj = json.loads(ln)
                            if not isinstance(obj, dict) or not obj.get("call_id"):
                                raise ValueError("missing call_id")
                        if salvaged:
                            part_p.write_bytes(raw)
                        os.replace(part_p, dest_p)
                        files_state[rel_file] = {
                            "time": ts,
                            "bytes": new_bytes,
                            "lines": new_lines,
                            "heartbeat": hb,
                        }
                        state_part = state_p.with_name(f"{state_p.name}.part.{os.getpid()}")
                        state_part.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                        os.replace(state_part, state_p)
    except Exception:
        rc = 2
elif rc == 0:
    rc = 2

if part_p.exists():
    try:
        part_p.unlink()
    except OSError:
        pass

if rc != 0:
    rep_bytes = old_bytes
    rep_lines = old_lines
else:
    rep_bytes = new_bytes
    rep_lines = new_lines

entry = json.dumps(
    {"time": ts, "file": rel_file, "bytes": rep_bytes, "lines": rep_lines, "rc": rc}
) + "\n"
for lp in (log1_p, log2_p):
    with lp.open("a", encoding="utf-8", newline="\n") as f:
        f.write(entry)
PY
      done
    done
  ) 8>"$COPYBACK_LOCK"
}
job_done_in_log() {
  if [ -n "${PIN_JOB_TOKEN:-}" ]; then
    grep "^PIN_RESULT" "$OUT/job.log" 2>/dev/null | grep -Fq "\"job_token\": \"$PIN_JOB_TOKEN\""
  else
    grep -q "^PIN_RESULT" "$OUT/job.log" 2>/dev/null
  fi
}
WATCH_S="${PIN_WATCH_S:-120}"
TICK_S="$WATCH_S"
if [ "$COPYBACK_S" -gt 0 ] 2>/dev/null && [ "$COPYBACK_S" -lt "$TICK_S" ]; then
  TICK_S="$COPYBACK_S"
fi
# A GPU window re-reads its rate every cap check, so the check runs at least once a minute.
if [ "${PIN_GPU:-A100}" != CPU ] && [ "$TICK_S" -gt 60 ]; then TICK_S=60; fi
( trap - EXIT TERM INT
  seen=0; size0=-1; grew=0; ended=""; noted=""; ceil_tripped=""; last_copyback=0; sess_fail=0; run_prev=""
  base="$RATE0"; last_assign="$ASSIGN0"; last_own="${RATE_LIVE:-}"; pend=""; pend_prev=""; pend_from=""; bad_n=0
  watcher_handle_launcher_gone() {
    if [ -f "$OUT/.session_open" ] && [ ! -f "$OUT/.stop_completed" ]; then
      [ -f "$OUT/.end.ts" ] || date +%s > "$OUT/.end.ts"
      [ -f "$OUT/.end" ] || echo watcher > "$OUT/.end"
      if [ -f "$OUT/.upload.pid" ]; then
        kill_pid_tree "$(cat "$OUT/.upload.pid" 2>/dev/null || true)" || true
      fi
      if [ -f "$OUT/.exec.pid" ]; then
        kill_pid_tree "$(cat "$OUT/.exec.pid" 2>/dev/null || true)" || true
      fi
      (
        flock -w "$STOP_LOCK_S" 9 || echo "open lock not free after ${STOP_LOCK_S} s; watcher stopping $S anyway"
        echo "watcher stopping $S (launcher gone)"; stop_session watcher || true
        sleep "$STOP_SETTLE_S"; echo "sessions after watcher stop:"; CLI_T="$SESSIONS_S" cli sessions | tail -3 || true
      ) 9>"$OPEN_LOCK"
      local end_w t_end_w spent_w="null"
      end_w=$(cat "$OUT/.end" 2>/dev/null || echo watcher)
      if [ -n "$OWN_RATE" ] && [ -n "$T_OPEN" ]; then
        t_end_w=$(cat "$OUT/.end.ts" 2>/dev/null || date +%s)
        spent_w=$(awk -v r="$OWN_RATE" -v t="$(( t_end_w - T_OPEN ))" 'BEGIN{printf "%.2f", r*(t>0?t:0)/3600}')
      fi
      outcome_draft "$end_w" "$spent_w" || true
    fi
    exit 0
  }
  launcher_alive() {
    kill -0 "$MAIN_PID" 2>/dev/null || return 1
    if [ -r "/proc/$MAIN_PID/stat" ]; then
      local st_line="" st_rest=""
      IFS= read -r st_line < "/proc/$MAIN_PID/stat" 2>/dev/null || return 1
      st_rest="${st_line##*) }"
      case "${st_rest%% *}" in Z|X|x) return 1;; esac
    fi
    return 0
  }
  watcher_sleep() {
    local rem="$1"
    case "$rem" in
      ''|*[!0-9]*)
        sleep "$rem" || return 1
        launcher_alive || watcher_handle_launcher_gone
        return 0
        ;;
    esac
    while [ "$rem" -gt 0 ]; do
      launcher_alive || watcher_handle_launcher_gone
      sleep 1 || return 1
      rem=$(( rem - 1 ))
    done
    launcher_alive || watcher_handle_launcher_gone
    return 0
  }
  while watcher_sleep "$TICK_S"; do
    # The mid-run rate (R&D review of #132): every reading is recorded; a stable reading at or above the floor raises the
    # highest rate seen, and one above the max rate ends the window as a cap trip (RATE_ABOVE_MAX), fetch then stop.
    if [ "${PIN_GPU:-A100}" != CPU ] && [ -z "$ended" ] && [ ! -f "$OUT/.end" ]; then
      # `colab usage` is account-wide: when the assignment count moves, a neighbour opened or closed, and the delta
      # against the old baseline is theirs too. Nothing is judged until two consecutive reads at the new count agree
      # within 0.01 (a neighbour's rate can register a read after its count); then the baseline becomes the account
      # rate less our last stable rate. Pricing stays max(max rate, highest seen) throughout. A missing count, or 0 while
      # we are open, is recorded; three in a row end the window as ASSIGNMENTS_UNREADABLE (fetch, then stop).
      # Known limit: a neighbour opening in the same read our own rate rises folds that rise into the baseline.
      : > "$OUT/.watcher_reread.out"
      timeout -k "$KILL_GRACE_S" "$USAGE_T" colab usage </dev/null >"$OUT/.watcher_reread.out" 2>&1 &
      reread_pid=$!
      printf '%s reread\n' "$reread_pid" > "$WATCHER_CHILD_PID_FILE"
      wait "$reread_pid" 2>/dev/null || true
      rm -f "$WATCHER_CHILD_PID_FILE"
      u_run=$(strip_colab_banner < "$OUT/.watcher_reread.out" 2>/dev/null || true)
      rm -f "$OUT/.watcher_reread.out"
      a_run=$(printf '%s\n' "$u_run" | grep -oP 'rate: \K[0-9]+(\.[0-9]+)?' | head -1 || true)
      n_run=$(printf '%s\n' "$u_run" | grep -oP 'Active assignments: \K[0-9]+' | head -1 || true)
      if [ -z "$n_run" ] || [ "$n_run" = 0 ]; then
        bad_n=$(( bad_n + 1 )); run_prev=""; pend=""
        record_reading bad_count "$a_run" "" "$n_run" "$base"
        if [ "$bad_n" -ge 3 ]; then
          [ -f "$OUT/.end.ts" ] || date +%s > "$OUT/.end.ts"
          echo "ASSIGNMENTS_UNREADABLE|assignment count missing or 0 for $bad_n consecutive reads (mid-run)" > "$OUT/.rate_verdict"
          echo "REFUSED ASSIGNMENTS_UNREADABLE (mid-run): assignment count missing or 0 for $bad_n consecutive reads; fetching, then stopping $S $(date -u +%T)"
          end_window ASSIGNMENTS_UNREADABLE; continue
        fi
      elif [ -z "$a_run" ]; then
        bad_n=$(( bad_n + 1 )); run_prev=""; pend=""
        record_reading bad_rate "" "" "$n_run" "$base"
        if [ "$bad_n" -ge 3 ]; then
          [ -f "$OUT/.end.ts" ] || date +%s > "$OUT/.end.ts"
          echo "ASSIGNMENTS_UNREADABLE|usage rate unreadable for $bad_n consecutive reads (mid-run)" > "$OUT/.rate_verdict"
          echo "REFUSED ASSIGNMENTS_UNREADABLE (mid-run): usage rate unreadable for $bad_n consecutive reads; fetching, then stopping $S $(date -u +%T)"
          end_window ASSIGNMENTS_UNREADABLE; continue
        fi
      elif [ "$n_run" != "$last_assign" ]; then
        bad_n=0
        [ -n "$pend" ] || pend_from=$last_assign
        pend=1; pend_prev=$a_run; last_assign=$n_run; run_prev=""
        record_reading rebaseline_pending "$a_run" "" "$n_run" "$base"
      elif [ -n "$pend" ]; then
        bad_n=0
        if awk -v a="$pend_prev" -v b="$a_run" 'BEGIN{x = a - b; if (x < 0) x = -x; exit !(x <= 0.01 + 1e-9)}'; then
          [ -n "$RATE_MAX_SEEN" ] && base=$(awk -v a="$a_run" -v o="$RATE_MAX_SEEN" 'BEGIN{b = a - o; print (b > 0 ? b : 0) + 0}')
          record_reading rebaseline "$a_run" "$RATE_MAX_SEEN" "$n_run" "$base"
          echo "RE-BASELINE: assignments $pend_from -> $n_run, neighbour baseline $base/hr $(date -u +%T)"
          pend=""
        else
          pend_prev=$a_run
          record_reading rebaseline_pending "$a_run" "" "$n_run" "$base"
        fi
      else
        d_run=$(delta_of "$a_run" "$base")
        if [ -z "$d_run" ] || awk -v d="$d_run" -v f="$GPU_RATE_FLOOR" 'BEGIN{exit !(d < f - 1e-9)}'; then
          bad_n=$(( bad_n + 1 )); run_prev=""
          record_reading below_floor "$a_run" "$d_run" "$n_run" "$base"
          if [ "$bad_n" -ge 3 ]; then
            [ -f "$OUT/.end.ts" ] || date +%s > "$OUT/.end.ts"
            echo "ASSIGNMENTS_UNREADABLE|own rate below floor ($GPU_RATE_FLOOR/hr) for $bad_n consecutive reads (mid-run)" > "$OUT/.rate_verdict"
            echo "REFUSED ASSIGNMENTS_UNREADABLE (mid-run): own rate below floor ($GPU_RATE_FLOOR/hr) for $bad_n consecutive reads; fetching, then stopping $S $(date -u +%T)"
            end_window ASSIGNMENTS_UNREADABLE; continue
          fi
        else
          bad_n=0
          record_reading run "$a_run" "$d_run" "$n_run" "$base"
          if [ -n "$run_prev" ] && [ -n "$d_run" ] && awk -v a="$run_prev" -v b="$d_run" -v f="$GPU_RATE_FLOOR" \
              'BEGIN{x = a - b; if (x < 0) x = -x; exit !(x <= 0.01 + 1e-9 && a >= f && b >= f)}'; then
            if awk -v r="$d_run" -v m="$RATE_MAX_SEEN" 'BEGIN{exit !(r > m)}'; then
              RATE_MAX_SEEN=$(awk -v r="$d_run" 'BEGIN{print r + 0}'); echo "$RATE_MAX_SEEN" > "$OUT/.rate_max"
              OWN_RATE=$(awk -v m="$MAX_RATE" -v r="$RATE_MAX_SEEN" 'BEGIN{print (r > m ? r : m) + 0}')
              PIN_DEADLINE=$(awk -v t="$T_OPEN" -v c="$CAP_UNITS" -v m="$MARGIN" -v r="$OWN_RATE" 'BEGIN{printf "%d", t + (c-m)/r*3600}')
              echo "RATE ROSE: own rate $RATE_MAX_SEEN/hr, priced at $OWN_RATE/hr $(date -u +%T)"
            fi
            if awk -v r="$d_run" -v m="$MAX_RATE" 'BEGIN{exit !(r > m)}'; then
              [ -f "$OUT/.end.ts" ] || date +%s > "$OUT/.end.ts"
              echo "RATE_ABOVE_MAX|own rate $d_run/hr is above the max rate $MAX_RATE/hr (mid-run)" > "$OUT/.rate_verdict"
              echo "REFUSED RATE_ABOVE_MAX (mid-run): own rate $d_run/hr is above the max rate $MAX_RATE/hr; fetching, then stopping $S $(date -u +%T)"
              end_window RATE_ABOVE_MAX; continue
            fi
            last_own=$d_run
          fi
          run_prev=$d_run
        fi
      fi
    fi
    if [ -z "$ended" ] && [ ! -f "$OUT/.end" ]; then
      sess_w=$(CLI_T="$SESSIONS_S" cli sessions 2>&1) && sess_rc=0 || sess_rc=$?
      if [ "$sess_rc" -eq 0 ]; then
        sess_fail=0
        if ! printf '%s\n' "$sess_w" | grep -q "^\[$S\]"; then
          date +%s > "$OUT/.end.ts"
          echo "SESSION GONE ($S) $(date -u +%T)"
          CLI_T=20 fetch_log gone || true
          DL_S=20 copyback_journals || true
          touch "$OUT/.final_pulled"
          end_window gone; continue
        fi
      else
        sess_fail=$(( sess_fail + 1 ))
        if [ "$sess_fail" -ge 2 ]; then
          date +%s > "$OUT/.end.ts"
          echo "SESSION GONE ($S) after $sess_fail failed session checks $(date -u +%T)"
          CLI_T=20 fetch_log gone || true
          DL_S=20 copyback_journals || true
          touch "$OUT/.final_pulled"
          end_window gone; continue
        fi
      fi
    fi
    now=$(date +%s)
    t_ref="$now"
    [ -f "$OUT/.end.ts" ] && t_ref=$(cat "$OUT/.end.ts" 2>/dev/null || echo "$now")
    spent=$(awk -v r="$OWN_RATE" -v t="$(( t_ref - T_OPEN ))" 'BEGIN{printf "%.3f", r*(t>0?t:0)/3600}')
    if [ -z "$ceil_tripped" ] && awk -v s="$spent" -v c="$CAP_UNITS" 'BEGIN{exit !(s>=c)}'; then
      ceil_tripped=1
      [ -f "$OUT/.end.ts" ] || date +%s > "$OUT/.end.ts"
      echo "CAP CEILING (own spend $spent of $CAP_UNITS): stopping now $(date -u +%T)"; end_window cap
    fi
    # The exit sequence is under way in the main script: only the ceiling (and launcher liveness) is watched now.
    [ -n "$ended" ] && continue
    [ -f "$OUT/.end" ] && continue
    if [ -f "$OUT/.exec.pid" ]; then
      CLI_T="$COPYBACK_DL_S" fetch_log watcher || true
      # The kernel's IDLE/BUSY beside each fetch: whether a busy session with regular file traffic still gets reclaimed.
      raw_st=$( ( trap - TERM INT; exec timeout -k "$KILL_GRACE_S" "$STATUS_S" colab status -s "$S" </dev/null 2>&1 ) || true)
      st=$(printf '%s\n' "$raw_st" | strip_colab_banner | scrub_log | tr -s ' \n' ' ' | sed 's/^ //; s/ $//' | cut -c1-300)
      echo "$(date -u +%T) $st" | scrub_log >> "$OUT/status.log"
      size=$(stat -c %s "$OUT/job.log" 2>/dev/null || echo 0)
      # The exec never reached the kernel (windows 11c, 12e: "Connection was lost" at the first command): the kernel
      # reads IDLE and job.log is absent or empty at the first look. Re-run the exec once on the same session, keeping
      # its uploads (R&D, 2026-10-17). A second time stalls as before; the job's lock stops a late first exec.
      if [ ! -f "$OUT/.retried" ] && [ "$size" = 0 ] && echo "$st" | grep -q "Status: IDLE"; then
        echo "EXEC NEVER STARTED ($S): kernel IDLE, job.log empty; re-running the exec once $(date -u +%T)"
        touch "$OUT/.retried" "$OUT/.retry"; kill "$(cat "$OUT/.exec.pid")" 2>/dev/null || true; continue
      fi
      if [ "$size" -gt 0 ] && [ "$size" != "$size0" ]; then seen=1; grew=$now; size0=$size; fi
      if job_done_in_log; then
        echo "JOB DONE (PIN_RESULT in job.log) $(date -u +%T)"; end_window done; continue
      fi
      if [ "$COPYBACK_S" -gt 0 ] 2>/dev/null && [ "$size" -gt 0 ] && [ $(( now - last_copyback )) -ge "$COPYBACK_S" ]; then
        DL_S="$COPYBACK_DL_S" copyback_journals || true
        last_copyback=$now
        if [ "${PIN_STUB:-0}" = 1 ] && [ -n "${PIN_STUB_DROP_AFTER_CALLS:-}" ]; then
          cb_lines=$(wc -l "$COPYBACK_DIR"/*/calls.jsonl 2>/dev/null | awk 'END{print $1+0}')
          if [ "${cb_lines:-0}" -ge "$PIN_STUB_DROP_AFTER_CALLS" ]; then
            (
              flock -w "$STOP_LOCK_S" 9 || true
              stop_session stub_drop >/dev/null 2>&1 || true
            ) 9>"$OPEN_LOCK"
            date +%s > "$OUT/.end.ts"
            touch "$OUT/.final_pulled"
            echo "SESSION GONE ($S) $(date -u +%T)"; end_window gone; continue
          fi
        fi
      fi
      # A job that dies with its kernel (an OOM kill) is caught only by the stall below. Window 13q (2026-10-18): the job
      # killed its kernel, but `colab status` kept reading "BUSY (exec(...))" for 22 minutes and the exec never returned;
      # status follows the CLI's exec, not the kernel. A rule on IDLE could never fire, so there is none (R&D). A child
      # killed for memory (a server, a converter) raises in the job instead, and its finally writes the result.
      started=$(stat -c %Y "$OUT/.exec.pid")
      stalled=""
      [ "$seen" = 0 ] && [ $(( now - started )) -gt "${PIN_STALL_FIRST_S:-600}" ] && stalled="job.log empty 10 minutes after the exec started"
      [ "$seen" = 1 ] && [ $(( now - grew )) -gt "${PIN_STALL_S:-600}" ] && stalled="job.log unchanged for ${PIN_STALL_S:-600}s"
      if [ -n "$stalled" ]; then
        [ -f "$OUT/.end.ts" ] || date +%s > "$OUT/.end.ts"
        echo "STALL ($S): $stalled $(date -u +%T)" | scrub_log | tee "$OUT/stall.txt"; end_window stall; continue
      fi
    fi
    if awk -v s="$spent" -v c="$CAP_UNITS" -v m="$MARGIN" 'BEGIN{exit !(s>=c-m)}'; then
      if [ "$now" -lt $(( PIN_DEADLINE + GRACE_S )) ]; then
        # The job cuts its own wave at the deadline and writes its results; its PIN_RESULT line ends the window above.
        [ -n "$noted" ] || echo "DEADLINE (own spend $spent of $CAP_UNITS): the job cuts its wave and writes its results; up to ${GRACE_S}s $(date -u +%T)"
        noted=1
      else
        [ -f "$OUT/.end.ts" ] || date +%s > "$OUT/.end.ts"
        echo "CAP REACHED (own spend $spent of $CAP_UNITS; $MARGIN kept back to fetch the results) $(date -u +%T)"; end_window cap
      fi
    fi
  done ) &
WATCH=$!
# The binary goes up gzipped in 40 MB parts, each retried; a part that never lands ends the window (the trap stops the VM).
# PIN_SKIP_BINARY=1 (a test of this script with mock_job.py) sends none.
UPLOAD_RETRY_SLEEP_S="${PIN_UPLOAD_RETRY_SLEEP_S:-10}"
run_upload() {  # $1: timeout_s, $2: local path, $3: remote path, $4: quiet (1 = suppress error output)
  local upid rc out="" raw_up=""
  if [ "${4:-0}" = 1 ]; then
    ( trap - TERM INT; exec timeout -k "$KILL_GRACE_S" "$1" colab upload -s "$S" "$2" "$3" </dev/null >/dev/null 2>&1 ) &
  else
    raw_up="$OUT/.upload.raw.$$"
    rm -f "$raw_up"
    ( trap - TERM INT; exec timeout -k "$KILL_GRACE_S" "$1" colab upload -s "$S" "$2" "$3" </dev/null >"$raw_up" 2>&1 ) &
  fi
  upid=$!
  echo "$upid" > "$OUT/.upload.pid"
  wait "$upid" && rc=0 || rc=$?
  rm -f "$OUT/.upload.pid"
  if [ -n "$raw_up" ]; then
    out=$(cat "$raw_up" 2>/dev/null || true)
    rm -f "$raw_up"
  fi
  if [ "$rc" -ne 0 ] && [ "${4:-0}" != 1 ]; then
    printf '%s\n' "$out" | strip_colab_banner | tail -1 | scrub_log
  fi
  return "$rc"
}
UP="$OUT/upload"; mkdir -p "$UP"; rm -f "$UP"/llama-server.gz*
if [ "${PIN_SKIP_BINARY:-0}" != 1 ] && [ "${PIN_ROLE:-}" = offrig-bench ]; then
  PIN_UPLOADS="$BIN ${PIN_UPLOADS:-}"
elif [ "${PIN_SKIP_BINARY:-0}" != 1 ]; then
  gzip -9 -c "$BIN" > "$UP/llama-server.gz" && split -b 40M -d "$UP/llama-server.gz" "$UP/llama-server.gz.part"
fi
for part in "$UP"/llama-server.gz.part*; do
  [ -e "$part" ] || continue
  [ -f "$OUT/.end" ] && break
  ok=0
  for try in 1 2 3 4; do
    [ -f "$OUT/.end" ] && break
    if run_upload "$UPLOAD_PART_S" "$part" "/content/$(basename "$part")"; then ok=1; break; fi
    [ -f "$OUT/.end" ] && break
    echo "upload of $(basename "$part") failed (try $try)"; interruptible_sleep "$UPLOAD_RETRY_SLEEP_S"
  done
  if [ "$ok" != 1 ]; then
    [ -f "$OUT/.end" ] && break
    echo "UPLOAD FAILED: $(basename "$part")"; exit 3
  fi
  echo "uploaded $(basename "$part")"
done
if [ "${PIN_ROLE:-}" = toolcount ]; then
  case " ${PIN_UPLOADS:-} " in
    */toolcount-pkg.tar\ *|*toolcount-pkg.tar\ *) ;;
    *)
      tar -C "$here/../.." -c \
        scripts/toolcount/__init__.py \
        scripts/toolcount/compass_lists.py \
        scripts/toolcount/gen_registry.py \
        scripts/toolcount/oracle.py \
        scripts/toolcount/runner.py \
        scripts/toolcount/scorer.py > "$UP/toolcount-pkg.tar"
      export PIN_TOOLCOUNT_PKG_SHA256="${PIN_TOOLCOUNT_PKG_SHA256:-$(sha256sum "$UP/toolcount-pkg.tar" | awk '{print $1}')}"
      PIN_UPLOADS="${PIN_UPLOADS:+$PIN_UPLOADS }$UP/toolcount-pkg.tar"
      ;;
  esac
fi
if [ -z "${PIN_RESUME_CALLS_DIR:-}" ] && [ -n "${PIN_RESUME_FROM:-}" ]; then
  export PIN_RESUME_CALLS_DIR="$PIN_RESUME_FROM"
fi
if [ -n "${PIN_RESUME_CALLS_DIR:-}" ]; then
  case " ${PIN_UPLOADS:-} " in
    */resume.tar\ *|*resume.tar\ *) ;;
    *)
      if [ -d "$PIN_RESUME_CALLS_DIR" ]; then
        tar -C "$PIN_RESUME_CALLS_DIR" --exclude='copyback.jsonl' --exclude='.copyback*' --exclude='*.part.*' -cf "$UP/resume.tar" .
        PIN_UPLOADS="${PIN_UPLOADS:+$PIN_UPLOADS }$UP/resume.tar"
      elif [ -f "$PIN_RESUME_CALLS_DIR" ]; then
        tar -C "$(dirname "$PIN_RESUME_CALLS_DIR")" -cf "$UP/resume.tar" "$(basename "$PIN_RESUME_CALLS_DIR")"
        PIN_UPLOADS="${PIN_UPLOADS:+$PIN_UPLOADS }$UP/resume.tar"
      fi
      ;;
  esac
fi
# Inputs for the window (PIN_UPLOADS: space-separated local files), each to /content/<basename>, retried like the parts.
for f in ${PIN_UPLOADS:-}; do
  [ -f "$OUT/.end" ] && break
  ok=0
  for try in 1 2 3 4; do
    [ -f "$OUT/.end" ] && break
    if run_upload "$UPLOAD_PART_S" "$f" "/content/$(basename "$f")"; then ok=1; break; fi
    [ -f "$OUT/.end" ] && break
    echo "upload of $(basename "$f") failed (try $try)"; interruptible_sleep "$UPLOAD_RETRY_SLEEP_S"
  done
  if [ "$ok" != 1 ]; then
    [ -f "$OUT/.end" ] && break
    echo "UPLOAD FAILED: $(basename "$f")"; exit 3
  fi
  echo "uploaded $(basename "$f") $(sha256sum "$f" | cut -c1-16)"
done
# The private GGUF repo's token (R&D, 2026-10-18), only for a window that names one: PIN_HF_TOKEN_FROM is the name of
# the rig's environment variable Mike set (the read token for a pilot window, the write token for a prebuild). It goes
# over as a file, written owner-only and deleted here at once: never on a command line, never through --env, never in a
# log. The job reads it once, deletes it and drops it before any wave. It goes up only after the VM gate passed, so a
# rejected VM never receives it. If the job dies before reading it, the file stays at /dev/shm/hf.token on a VM that
# this script is stopping anyway (the EXIT trap); nothing fetches it back. On the VM it goes to RAM-backed /dev/shm,
# outside /content and every tar.
if [ -n "${PIN_HF_TOKEN_FROM:-}" ] && [ ! -f "$OUT/.end" ]; then
  tokvar=$PIN_HF_TOKEN_FROM
  TOKF=$(mktemp); chmod 600 "$TOKF"; printf '%s' "${!tokvar}" > "$TOKF"; tf=$TOKF
  ok=0
  for try in 1 2 3; do
    [ -f "$OUT/.end" ] && break
    if run_upload "$UPLOAD_TOKEN_S" "$tf" /dev/shm/hf.token 1; then ok=1; break; fi
    [ -f "$OUT/.end" ] && break
    interruptible_sleep 5
  done
  rm -f "$tf"; TOKF=""
  if [ "$ok" != 1 ]; then
    if [ ! -f "$OUT/.end" ]; then
      echo "UPLOAD FAILED: the token file"; exit 3
    fi
  else
    echo "uploaded the token file (owner-only, deleted on the rig)"
  fi
fi
t0=$(date +%s)
# Every PIN_* variable reaches the VM (window 10 failed when PIN_RND_COMMIT wasn't on a hand-kept list), except the
# window script's own settings, which only matter here.
ENV_ARGS=()
for v in $(compgen -e | grep '^PIN_' | grep -v -E '^PIN_(CAP_UNITS|WALL_S|SESSION|GPU|UPLOADS|DOWNLOADS|WATCH_S|STALL_FIRST_S|STALL_S|CAP_MARGIN|DL_TIMEOUT|SKIP_BINARY|JOB_SCRIPT|OPEN_LOCK|HF_TOKEN_FROM|HIGH_MEM|DEADLINE_GRACE_S|FETCH_ALLOW_S|UPLOAD_RETRY_SLEEP_S|COPYBACK_S|COPYBACK_DL_S|COPYBACK_LOCK|COPYBACK_STATE|COLAB_CONFIG|MAX_RATE|RATE_POLLS|RATE_POLL_S|USAGE_TIMEOUT_S|STUB_COPYBACK_RESUME|KILL_ESCALATE_S|C2_WATCH_S|C2_KILL_GRACE_S|KILL_GRACE_S|OPEN_LOCK_TIMEOUT_S|STOP_LOCK_TIMEOUT_S|NEW_TIMEOUT_S|STOP_TIMEOUT_S|STOP_SETTLE_S|GATE_SETTLE_S|SESSIONS_TIMEOUT_S|STATUS_TIMEOUT_S|UPLOAD_TIMEOUT_S|TOKEN_UPLOAD_TIMEOUT_S)$' || true); do
  ENV_ARGS+=(--env "$v=${!v}")
done
[ -n "${PIN_ROLE:-}" ] || ENV_ARGS+=(--env "PIN_ROLE=smoke")
run_exec() {
  ( trap - TERM INT; exec timeout --signal=SIGINT -k "$KILL_GRACE_S" "$WALL_S" colab exec -s "$S" "${ENV_ARGS[@]}" -f "$JOB" --timeout "$WALL_S" </dev/null ) \
    > >(scrub_log >> "$OUT/smoke-window.log") 2>&1 &
  EXEC=$!
  echo $EXEC > "$OUT/.exec.pid"
}
rc=0
if [ ! -f "$OUT/.end" ]; then
  run_exec
  IN_MAIN_EXEC_WAIT=1
  wait $EXEC && rc=0 || rc=$?
  IN_MAIN_EXEC_WAIT=0
  if [ -f "$OUT/.retry" ] && [ ! -f "$OUT/.end" ]; then  # the watcher saw the exec never start: once more, same session
    rm -f "$OUT/.retry"
    run_exec
    IN_MAIN_EXEC_WAIT=1
    wait $EXEC && rc=0 || rc=$?
    IN_MAIN_EXEC_WAIT=0
  fi
  if [ ! -f "$OUT/.end" ] && { [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; }; then
    [ -f "$OUT/.end.ts" ] || date +%s > "$OUT/.end.ts"
    echo "WALL REACHED (wall ${WALL_S}s, exec exit=$rc): fetching, then stopping $S $(date -u +%T)"
    echo wall > "$OUT/.end"
  fi
fi
echo "exec exit=$rc wall=$(( $(date +%s)-t0 ))s end=$(cat "$OUT/.end" 2>/dev/null || echo exec)"
# The exit sequence: fetch, then the trap stops the session. The watcher stays up as the cap's ceiling meanwhile.
[ -f "$OUT/.end" ] || echo exec > "$OUT/.end"
if [ "$(cat "$OUT/.end" 2>/dev/null)" != gone ]; then
  f0=$(date +%s)
  fetch_log final && echo "fetched job.log ($(stat -c %s "$OUT/job.log") bytes)" || echo "job.log not refreshed; the last good copy stays"
  if [ "$COPYBACK_S" -gt 0 ] 2>/dev/null; then
    copyback_journals || true
  fi
  touch "$OUT/.final_pulled"
  DEFAULT_DL="pin-result.json pilot-out.tar lib.tar generations.jsonl"
  [ "${PIN_ROLE:-}" = offrig-bench ] && DEFAULT_DL="pin-result.json offrig-bench-out.tar ps.json"
  [ "${PIN_ROLE:-}" = toolcount ] && DEFAULT_DL="pin-result.json toolcount-out.tar lib.tar generations.jsonl"
  for f in ${PIN_DOWNLOADS:-$DEFAULT_DL}; do
    dl_out=$( ( trap - TERM INT; exec timeout -k "$KILL_GRACE_S" "$DL_S" colab download -s "$S" "/content/pin/$f" "$OUT/$f" </dev/null 2>&1 ) ) && dl_rc=0 || dl_rc=$?
    printf '%s\n' "$dl_out" | strip_colab_banner | tail -1 | scrub_log
  done
  # The exit sequence's fetch time, the measure for PIN_CAP_MARGIN.
  echo "exit fetch took $(( $(date +%s) - f0 ))s"
fi
reap_watcher_orphans
kill "$WATCH" 2>/dev/null || true
rm -f "$OUT"/job.log.part.* 2>/dev/null || true
if [ -d "$COPYBACK_DIR" ]; then
  ( flock -w 5 8 || true; rm -f "$OUT"/copyback/*.part.* "$OUT"/copyback/*/*.part.* 2>/dev/null || true ) 8>"$COPYBACK_LOCK"
fi
grep -h "^PIN_RESULT" "$OUT/job.log" "$OUT/smoke-window.log" 2>/dev/null | head -1 | cut -c1-600 | scrub_log || true
echo "end balance $(bal) (start $START)"
# The window's outcome record (R&D, 2026-10-18: every outcome.json carries its date and commit). The rig writes what it
# measured; the outcome class and its why are filled in by whoever reads the window, in outcome.json.
T_END=$(cat "$OUT/.end.ts" 2>/dev/null || date +%s)
if [ "${PIN_GPU:-A100}" != CPU ] && [ -f "$OUT/.rate_max" ]; then
  RATE_MAX_SEEN=$(cat "$OUT/.rate_max")
  OWN_RATE=$(awk -v m="$MAX_RATE" -v r="$RATE_MAX_SEEN" 'BEGIN{print (r > m ? r : m) + 0}')
  PIN_DEADLINE=$(awk -v t="$T_OPEN" -v c="$CAP_UNITS" -v m="$MARGIN" -v r="$OWN_RATE" 'BEGIN{printf "%d", t + (c-m)/r*3600}')
fi
if [ -f "$OUT/.rate_verdict" ]; then
  v=$(cat "$OUT/.rate_verdict"); RATE_WHY=${v#*|}; write_rate_receipt "${v%%|*}"
else
  write_rate_receipt ok
fi
SPENT_FINAL=$(awk -v r="$OWN_RATE" -v t="$(( T_END - T_OPEN ))" 'BEGIN{printf "%.2f", r*(t>0?t:0)/3600}')
outcome_draft "$(cat "$OUT/.end" 2>/dev/null || echo exec)" "$SPENT_FINAL"
if [ "$(cat "$OUT/.end" 2>/dev/null)" = term ]; then
  exit "$INTERRUPT_EXIT_RC"
fi
