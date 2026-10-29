#!/usr/bin/env bash
# Tool-count cliff study — Window 1 (W1: Granite 4.2 8B, 2,400 calls) & R1 CPU stub launcher.
# Follows the `colab` skill window path and C2-era guards (live wall, live cap, vitals/DEAD HARNESS watcher).
#
# Usage:
#   Local R1 CPU stub (no Colab, no GPU, verifies all 9 branches):
#     TOOLCOUNT_MODE=local-stub bash scripts/toolcount/launch-w1.sh /dev/null /tmp/toolcount-r1
#   Colab R1 CPU stub window (cap 0.05, wall 3000s):
#     TOOLCOUNT_MODE=stub bash scripts/toolcount/launch-w1.sh /dev/null /tmp/toolcount-r1-vm
#   Colab W1 paid window (Granite 4.2 8B, A100, cap 13, wall 9000s = 2.5h):
#     TOOLCOUNT_MODE=run bash scripts/toolcount/launch-w1.sh /path/to/llama-server /path/to/w1-out

set -euo pipefail

BIN="${1:?usage: launch-w1.sh <llama-server-or-/dev/null> <out-dir>}"
OUT="${2:?usage: launch-w1.sh <llama-server-or-/dev/null> <out-dir>}"
MODE="${TOOLCOUNT_MODE:-local-stub}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
SMOKE_SH="$REPO_ROOT/scripts/colab/smoke_window.sh"

# ---- Pinned study constants (copied from read files; unpinned items marked "to pin") ----
export PIN_ROLESET_COMMIT="${PIN_ROLESET_COMMIT:-to pin}"
export PIN_RND_COMMIT="${PIN_RND_COMMIT:-84008467b2cb21b4f648262bf242495c49eed828}"
export PIN_TOOL_COMPASS_COMMIT="6f5a233"
export PIN_SERVER_SHA256="${PIN_SERVER_SHA256:-b4659b3fc74037deb96b2148c455753121fa21dde174d7124547e2bf9ffd37c3}"
export PIN_LLAMA_CPP_COMMIT="620d2225cc436ab0b001e3200e590e3b286ef9ab"

# Model pins (from writers.json & pin_window.py)
export GRANITE_8B_REPO="ibm-granite/granite-4.2-8b"
export GRANITE_8B_REVISION="a7042144125ce5ae64f89610e8289647fc000b9a"
export GRANITE_8B_GGUF_SHA256="${GRANITE_8B_GGUF_SHA256:-to pin}"
export QWEN_27B_GGUF_SHA256="2ab268e0bd3f8e5daf1280e2243ac19abd5e3bb34e8444f5cfff5147fab977d4"
export GEMMA_31B_GGUF_SHA256="e357a45071c2cbbc44b97506dfadc34bac0d5f6592f1d500edfaa9647f01e7bf"

# Input artifact pins (filled after R0 and real compass embedding run)
export R0_TAR="${R0_TAR:-$OUT/toolcount-r0.tar}"
export R0_TAR_SHA256="${R0_TAR_SHA256:-to pin}"
export TOOLCOUNT_PKG_TAR="${TOOLCOUNT_PKG_TAR:-$OUT/toolcount-pkg.tar}"

# Per-request HTTP timeout guard (C2-era standard)
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-60}"

# Always enforce the tokenizer hard stop for the tool-count cliff study (Publisher's condition, Card kh item 8)
export PIN_ROLE="toolcount"
export PIN_TOKENIZER_HARD_STOP="1"
export PIN_MODELS="${PIN_MODELS:-granite-4.2-8b}"
export PIN_ONLY="${PIN_ONLY:-$PIN_MODELS}"
export PIN_TOOLCOUNT_HALF="${PIN_TOOLCOUNT_HALF:-ALL}"
export PIN_TOOLCOUNT_SEED="${PIN_TOOLCOUNT_SEED:-20261028}"

mkdir -p "$OUT"

scrub_colab_log() {
  sed -u -E \
    -e 's/(colab-runtime-proxy-token|access_token)=[^&"'\''[:space:])]+/\1=[REDACTED]/g' \
    -e 's/([Bb]earer[[:space:]]+)[A-Za-z0-9._-]+/\1REDACTED/g' \
    -e 's/eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/[REDACTED]/g' \
    -e 's/hf_[A-Za-z0-9]{6,}/[REDACTED]/g'
}

# ---- Mode 1: Local CPU stub rung R1 (zero GPU, zero Colab, reaches all 9 branches) ----
if [ "$MODE" = "local-stub" ]; then
  export PIN_STUB=1
  export PIN_GPU=CPU
  export PIN_SKIP_BINARY=1
  python3 -m scripts.toolcount.runner --mode r1-stub --out-dir "$OUT" --seed "$PIN_TOOLCOUNT_SEED"
  exit 0
fi

# ---- Mode 2 (Colab CPU stub R1) & Mode 3 (Colab A100 paid W1) ----
case "$MODE" in
  stub)
    export PIN_STUB=1
    export PIN_GPU=CPU
    export PIN_SKIP_BINARY=1
    export PIN_CAP_UNITS="${PIN_CAP_UNITS:-0.05}"
    export PIN_CAP_MARGIN="${PIN_CAP_MARGIN:-0.01}"
    export PIN_WALL_S="${PIN_WALL_S:-3000}"
    export PIN_DEADLINE_GRACE_S="${PIN_DEADLINE_GRACE_S:-60}"
    export PIN_FETCH_ALLOW_S="${PIN_FETCH_ALLOW_S:-120}"
    export PIN_GATE_MIN_RAM_GB="${PIN_GATE_MIN_RAM_GB:-8}"
    export PIN_GATE_MIN_HF_MBPS="${PIN_GATE_MIN_HF_MBPS:-20}"
    export PIN_COPYBACK_S="${PIN_COPYBACK_S:-10}"
    export PIN_SESSION="${PIN_SESSION:-tc-r1-stub}"
    CARD_RATE="0.08"
    if [ "$PIN_ROLESET_COMMIT" = "to pin" ] || [ -z "$PIN_ROLESET_COMMIT" ]; then
      PIN_ROLESET_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || git.exe -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null | tr -d '\r' || true)"
      export PIN_ROLESET_COMMIT
    fi
    for commit_var in PIN_RND_COMMIT PIN_ROLESET_COMMIT; do
      if ! printf '%s' "${!commit_var}" | grep -qE '^[0-9a-f]{40}$'; then
        echo "REFUSED: $commit_var='${!commit_var}' must be 40 hex characters" >&2
        exit 2
      fi
    done
    if [ ! -f "$R0_TAR" ] && [ -d "$REPO_ROOT/artifacts/toolcount/r0" ]; then
      tar -C "$REPO_ROOT/artifacts/toolcount/r0" -cf "$R0_TAR" .
    fi
    if [ -f "$R0_TAR" ] && { [ "$R0_TAR_SHA256" = "to pin" ] || [ -z "$R0_TAR_SHA256" ]; }; then
      export R0_TAR_SHA256="$(sha256sum "$R0_TAR" | awk '{print $1}')"
    fi
    if [ -f "$R0_TAR" ]; then
      export PIN_UPLOADS="${PIN_UPLOADS:-$R0_TAR}"
    fi
    ;;
  run)
    export PIN_STUB=0
    export PIN_GPU="${PIN_GPU:-A100}"
    export PIN_SKIP_BINARY=0
    # Design v1 §5: W1 Granite 4.2 8B — wall 2.5 h (9000 s), cap 13 units (binding ceiling = cap 13)
    export PIN_CAP_UNITS="${PIN_CAP_UNITS:-13}"
    export PIN_CAP_MARGIN="${PIN_CAP_MARGIN:-0.3}"
    export PIN_WALL_S="${PIN_WALL_S:-9000}"
    export PIN_DEADLINE_GRACE_S="${PIN_DEADLINE_GRACE_S:-60}"
    export PIN_FETCH_ALLOW_S="${PIN_FETCH_ALLOW_S:-120}"
    export PIN_GATE_MIN_DISK_GB="${PIN_GATE_MIN_DISK_GB:-60}"
    export PIN_GATE_MIN_RAM_GB="${PIN_GATE_MIN_RAM_GB:-50}"
    export PIN_GATE_MIN_HF_MBPS="${PIN_GATE_MIN_HF_MBPS:-60}"
    export PIN_SESSION="${PIN_SESSION:-tc-w1-granite8b}"
    CARD_RATE="5.30"

    for commit_var in PIN_RND_COMMIT PIN_ROLESET_COMMIT; do
      if ! printf '%s' "${!commit_var}" | grep -qE '^[0-9a-f]{40}$'; then
        echo "REFUSED: $commit_var='${!commit_var}' must be 40 hex characters" >&2
        exit 2
      fi
    done
    # Refuse paid run if any required pin is still "to pin"
    for req_var in PIN_ROLESET_COMMIT R0_TAR_SHA256; do
      if [ "${!req_var}" = "to pin" ] || [ -z "${!req_var}" ]; then
        echo "REFUSED: $req_var is still 'to pin'; pin it before launching a paid GPU window" >&2
        exit 2
      fi
    done
    if [ ! -f "$BIN" ]; then
      echo "REFUSED: llama-server binary '$BIN' not found" >&2
      exit 2
    fi
    GOT_BIN_SHA="$(sha256sum "$BIN" | awk '{print $1}')"
    if [ "$GOT_BIN_SHA" != "$PIN_SERVER_SHA256" ]; then
      echo "REFUSED: llama-server sha256 $GOT_BIN_SHA != pinned $PIN_SERVER_SHA256" >&2
      exit 2
    fi
    if [ ! -f "$R0_TAR" ]; then
      echo "REFUSED: R0 archive '$R0_TAR' not found" >&2
      exit 2
    fi
    GOT_R0_SHA="$(sha256sum "$R0_TAR" | awk '{print $1}')"
    if [ "$GOT_R0_SHA" != "$R0_TAR_SHA256" ]; then
      echo "REFUSED: R0 archive sha256 $GOT_R0_SHA != pinned $R0_TAR_SHA256" >&2
      exit 2
    fi
    export PIN_UPLOADS="${PIN_UPLOADS:-$R0_TAR}"
    ;;
  *)
    echo "REFUSED: unknown TOOLCOUNT_MODE='$MODE' (expected local-stub, stub, or run)" >&2
    exit 2
    ;;
esac

tar -C "$REPO_ROOT" -c \
  scripts/toolcount/__init__.py \
  scripts/toolcount/compass_lists.py \
  scripts/toolcount/gen_registry.py \
  scripts/toolcount/oracle.py \
  scripts/toolcount/runner.py \
  scripts/toolcount/scorer.py > "$TOOLCOUNT_PKG_TAR"
export PIN_TOOLCOUNT_PKG_SHA256="${PIN_TOOLCOUNT_PKG_SHA256:-$(sha256sum "$TOOLCOUNT_PKG_TAR" | awk '{print $1}')}"
case " ${PIN_UPLOADS:-} " in
  *" $TOOLCOUNT_PKG_TAR "*) ;;
  *) export PIN_UPLOADS="${PIN_UPLOADS:+$PIN_UPLOADS }$TOOLCOUNT_PKG_TAR" ;;
esac

export PIN_DOWNLOADS="${PIN_DOWNLOADS:-pin-result.json toolcount-out.tar lib.tar generations.jsonl}"

# ---- C2-era Guards: Live Wall, Live Cap, and Vitals / DEAD HARNESS Watcher ----
VITALS_LOG="$OUT/vitals.log"
T_LAUNCH="$(date +%s)"
HARD_WALL_S=$(( PIN_WALL_S + PIN_DEADLINE_GRACE_S + PIN_FETCH_ALLOW_S ))
KILL_GRACE_S="${PIN_C2_KILL_GRACE_S:-35}"

c2_guard_loop() {
  local win_pid="$1"
  local win_out="${2:-$OUT}"
  local vitals_log="$win_out/vitals.log"
  while kill -0 "$win_pid" 2>/dev/null; do
    sleep "${PIN_C2_WATCH_S:-30}" || true
    kill -0 "$win_pid" 2>/dev/null || break
    local now t_spend elapsed spent_elapsed spent
    now="$(date +%s)"
    elapsed=$(( now - T_LAUNCH ))
    if [ -f "$win_out/.end" ]; then
      t_spend="$(cat "$win_out/.end.ts" 2>/dev/null || stat -c %Y "$win_out/.end" 2>/dev/null || echo "$now")"
    else
      t_spend="$now"
    fi
    spent_elapsed=$(( t_spend - T_LAUNCH ))
    [ "$spent_elapsed" -ge 0 ] || spent_elapsed=0
    spent="$(awk -v r="$CARD_RATE" -v t="$spent_elapsed" 'BEGIN{printf "%.4f", r*t/3600}')"
    printf '%s elapsed=%ds spent=%s/%s session=%s\n' \
      "$(date -u +%FT%TZ)" "$elapsed" "$spent" "$PIN_CAP_UNITS" "$PIN_SESSION" >> "$vitals_log"

    # 1. Live Wall Guard
    if [ "$elapsed" -ge "$HARD_WALL_S" ]; then
      printf '%s LIVE WALL EXCEEDED (%ds >= %ds): stopping window and session %s\n' \
        "$(date -u +%FT%TZ)" "$elapsed" "$HARD_WALL_S" "$PIN_SESSION" | tee -a "$vitals_log"
      kill -TERM "$win_pid" 2>/dev/null || true
      sleep "$KILL_GRACE_S"
      kill -KILL "$win_pid" 2>/dev/null || true
      colab stop -s "$PIN_SESSION" 2>&1 | scrub_colab_log >> "$vitals_log" || true
      break
    fi

    # 2. Live Cap Guard (frozen once .end exists, especially on .end == gone)
    if [ ! -f "$win_out/.end" ] && awk -v s="$spent" -v c="$PIN_CAP_UNITS" 'BEGIN{exit !(s>=c)}'; then
      printf '%s LIVE CAP CEILING REACHED (%s >= %s): stopping window and session %s\n' \
        "$(date -u +%FT%TZ)" "$spent" "$PIN_CAP_UNITS" "$PIN_SESSION" | tee -a "$vitals_log"
      kill -TERM "$win_pid" 2>/dev/null || true
      sleep "$KILL_GRACE_S"
      kill -KILL "$win_pid" 2>/dev/null || true
      colab stop -s "$PIN_SESSION" 2>&1 | scrub_colab_log >> "$vitals_log" || true
      break
    fi

    # 3. Dead-Harness Watcher: if .exec.pid exists in $win_out and that process is gone without .end or PIN_RESULT
    if [ -f "$win_out/.exec.pid" ] && [ ! -f "$win_out/.end" ]; then
      local epid
      epid="$(cat "$win_out/.exec.pid" 2>/dev/null || true)"
      if [ -n "$epid" ] && ! kill -0 "$epid" 2>/dev/null; then
        if ! grep -q "^PIN_RESULT" "$win_out/job.log" 2>/dev/null; then
          printf '%s DEAD HARNESS: exec pid %s exited without PIN_RESULT: stopping window and session %s\n' \
            "$(date -u +%FT%TZ)" "$epid" "$PIN_SESSION" | tee -a "$vitals_log"
          kill -TERM "$win_pid" 2>/dev/null || true
          sleep "$KILL_GRACE_S"
          kill -KILL "$win_pid" 2>/dev/null || true
          colab stop -s "$PIN_SESSION" 2>&1 | scrub_colab_log >> "$vitals_log" || true
          break
        fi
      fi
    fi
  done
}

run_one_window() {
  local win_out="${1:-$OUT}"
  mkdir -p "$win_out"
  bash "$SMOKE_SH" "$BIN" "$win_out" &
  WIN_PID=$!
  c2_guard_loop "$WIN_PID" "$win_out" &
  GUARD_PID=$!

  wait "$WIN_PID" && WIN_RC=0 || WIN_RC=$?
  kill "$GUARD_PID" 2>/dev/null || true
  return "$WIN_RC"
}

if [ "$MODE" = "stub" ] && [ "${PIN_STUB_COPYBACK_RESUME:-0}" = "1" ]; then
  DROP_OUT="${OUT}-drop"
  export PIN_STUB_DROP_AFTER_CALLS="${PIN_STUB_DROP_AFTER_CALLS:-3}"
  export PIN_COPYBACK_S="${PIN_COPYBACK_S:-2}"
  T_LAUNCH="$(date +%s)"
  run_one_window "$DROP_OUT" || true
  [ -f "$DROP_OUT/outcome.draft.json" ] && cp -f "$DROP_OUT/outcome.draft.json" "$OUT/outcome.pass1.draft.json"
  unset PIN_STUB_DROP_AFTER_CALLS
  export PIN_RESUME_FROM="${PIN_RESUME_FROM:-${DROP_OUT}/copyback}"
  export PIN_RESUME_CALLS_DIR="${PIN_RESUME_CALLS_DIR:-$PIN_RESUME_FROM}"
  T_LAUNCH="$(date +%s)"
  run_one_window "$OUT"
  exit $?
fi

run_one_window "$OUT"
exit $?
