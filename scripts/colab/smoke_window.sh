#!/usr/bin/env bash
# Window 2 (Mike's go, 2026-10-17), from the rig in WSL Ubuntu:
#   MSYS_NO_PATHCONV=1 wsl -d Ubuntu -e bash -l scripts/colab/smoke_window.sh <window-1 llama-server> <out dir>
# One A100: upload window 1's llama-server (checked by sha256 on the VM), install the CUDA 13.4 runtime wheels,
# pin the libraries it loads, convert both Colab models to Q8_0 at pinned revisions (Lightning must reproduce window 1's
# hash), smoke-serve each with logprobs, then fetch the results. Unit cap and wall clock enforced; the VM is stopped in
# a trap and the sessions list checked empty.
set -u
BIN="${1:?usage: smoke_window.sh <llama-server> <out dir>}"
OUT="${2:?usage: smoke_window.sh <llama-server> <out dir>}"
CAP_UNITS="${PIN_CAP_UNITS:-8}"
WALL_S="${PIN_WALL_S:-5400}"
S=${PIN_SESSION:-pinw2}
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"; rm -f "$OUT/.exec.pid" "$OUT/.end" "$OUT/job.log"
JOB="$here/${PIN_JOB_SCRIPT:-pin_window.py}"  # mock_job.py for a test of this script on a cheap runtime
MARGIN="${PIN_CAP_MARGIN:-0.3}"  # units kept back at the cap to fetch the results (R&D, 2026-10-17)
DL_S="${PIN_DL_TIMEOUT:-300}"
export COLUMNS=150
bal() { colab usage 2>&1 | grep -oP 'balance: \K[0-9.]+'; }
# Stops this window's own session only: another window (or a probe) may be running beside it.
stop_all() {
  echo "stopping $S"; colab stop -s "$S" 2>&1 | tail -1
  sleep 5; echo "sessions after stop:"; colab sessions 2>&1 | tail -3
}
trap stop_all EXIT
rate() { colab usage 2>&1 | grep -oP 'rate: \K[0-9.]+'; }
# Windows on this rig open one at a time: the rate read before, the open and the rate read after hold one lock, so a
# neighbour opening in between can't land in this window's rate (windows 11a and 11b each counted both sessions).
exec 9>/tmp/colab-window-open.lock; flock 9
START=$(bal); RATE0=$(rate); echo "start balance $START, account rate $RATE0/hr $(date -u +%FT%TZ)"
sha256sum "$BIN"
GPU_ARGS=(--gpu "${PIN_GPU:-A100}"); [ "${PIN_GPU:-A100}" = CPU ] && GPU_ARGS=()
timeout 300 colab new "${GPU_ARGS[@]}" -s "$S" 2>&1 | tail -2 || { echo "no ${PIN_GPU:-A100} assigned"; exit 2; }
sleep 15; colab usage 2>&1 | grep -E "rate|assign"
# The cap counts this window's own spend: its rate (the account rate after its session opened, less the rate before) times
# its own runtime. The account balance would count a neighbour's spend too (window 8 was cut off that way).
OWN_RATE=$(awk -v a="$(rate)" -v b="$RATE0" 'BEGIN{r=a-b; if (r<=0) r=a; print r}'); T_OPEN=$(date +%s)
echo "own rate $OWN_RATE/hr"
flock -u 9
# The watcher reads the job's own log, /content/pin/job.log, fetched every PIN_WATCH_S seconds, never the exec's output
# stream: that stream stopped reaching the rig in windows 10d and 12a while the job ran on (mistakes #31). It ends the
# window, through the main script's exit sequence (fetch the results, then stop the session), when:
# - job.log carries the PIN_RESULT line: the job is done, even if the exec never returns;
# - job.log hasn't grown: nothing within 10 minutes of the exec starting, or nothing new for 20 minutes after that
#   (R&D, 2026-10-17; the slowest silent step, a conversion, takes about 12 minutes);
# - own spend reaches the cap less MARGIN, the units kept back to fetch the results.
# Colab dropping the session ends the exec at once, with nothing to fetch. At the cap itself the session stops whatever
# is under way: the cap is the ceiling.
# job.log comes down to job.log.part and replaces the local copy only when the fetch exited 0 and the new copy is no
# smaller (R&D, 2026-10-17): a fetch that dies partway, a VM reclaimed mid-transfer or the timeout, can't truncate the
# one local copy of the checkpoint lines, which is the case they are kept for.
fetch_log() {  # $1: a name for this fetcher's part file, so the watcher's and the exit sequence's never collide
  local part="$OUT/job.log.part.${1:-x}" new old
  rm -f "$part"
  timeout "$DL_S" colab download -s "$S" /content/pin/job.log "$part" > "$OUT/.fetch.out" 2>&1 || { rm -f "$part"; return 1; }
  grep -qi "fail\|error" "$OUT/.fetch.out" && { rm -f "$part"; return 1; }
  [ -f "$part" ] || return 1
  new=$(stat -c %s "$part"); old=$(stat -c %s "$OUT/job.log" 2>/dev/null || echo 0)
  if [ "$new" -ge "$old" ]; then mv -f "$part" "$OUT/job.log"; else rm -f "$part"; return 1; fi
}
( seen=0; size0=-1; grew=0; ended=""
  end_window() { echo "$1" > "$OUT/.end"; ended=$1; [ -f "$OUT/.exec.pid" ] && kill "$(cat "$OUT/.exec.pid")" 2>/dev/null; }
  while sleep "${PIN_WATCH_S:-120}"; do
    now=$(date +%s)
    spent=$(awk -v r="$OWN_RATE" -v t="$(( now - T_OPEN ))" 'BEGIN{printf "%.3f", r*t/3600}')
    if awk -v s="$spent" -v c="$CAP_UNITS" 'BEGIN{exit !(s>=c)}'; then
      echo "CAP CEILING (own spend $spent of $CAP_UNITS): stopping now $(date -u +%T)"; end_window cap; stop_all; exit 0
    fi
    # The exit sequence is under way in the main script: only the ceiling is watched now.
    [ -n "$ended" ] && continue
    [ -f "$OUT/.end" ] && continue
    if ! colab sessions 2>&1 | grep -q "^\[$S\]"; then
      echo "SESSION GONE ($S) $(date -u +%T)"; end_window gone; exit 0
    fi
    if [ -f "$OUT/.exec.pid" ]; then
      fetch_log watcher
      # The kernel's IDLE/BUSY beside each fetch: whether a busy session with regular file traffic still gets reclaimed.
      echo "$(date -u +%T) $(timeout 60 colab status -s "$S" 2>&1 | tr -s ' \n' ' ' | cut -c1-300)" >> "$OUT/status.log"
      size=$(stat -c %s "$OUT/job.log" 2>/dev/null || echo 0)
      if [ "$size" -gt 0 ] && [ "$size" != "$size0" ]; then seen=1; grew=$now; size0=$size; fi
      if grep -q "^PIN_RESULT" "$OUT/job.log" 2>/dev/null; then
        echo "JOB DONE (PIN_RESULT in job.log) $(date -u +%T)"; end_window done; continue
      fi
      started=$(stat -c %Y "$OUT/.exec.pid")
      stalled=""
      [ "$seen" = 0 ] && [ $(( now - started )) -gt "${PIN_STALL_FIRST_S:-600}" ] && stalled="job.log empty 10 minutes after the exec started"
      [ "$seen" = 1 ] && [ $(( now - grew )) -gt "${PIN_STALL_S:-1200}" ] && stalled="job.log unchanged for 20 minutes"
      if [ -n "$stalled" ]; then
        echo "STALL ($S): $stalled $(date -u +%T)" | tee "$OUT/stall.txt"; end_window stall; continue
      fi
    fi
    if awk -v s="$spent" -v c="$CAP_UNITS" -v m="$MARGIN" 'BEGIN{exit !(s>=c-m)}'; then
      echo "CAP REACHED (own spend $spent of $CAP_UNITS; $MARGIN kept back to fetch the results) $(date -u +%T)"; end_window cap
    fi
  done ) &
WATCH=$!
# The binary goes up gzipped in 40 MB parts, each retried; a part that never lands ends the window (the trap stops the VM).
# PIN_SKIP_BINARY=1 (a test of this script with mock_job.py) sends none.
UP="$OUT/upload"; mkdir -p "$UP"; rm -f "$UP"/llama-server.gz*
[ "${PIN_SKIP_BINARY:-0}" = 1 ] || { gzip -9 -c "$BIN" > "$UP/llama-server.gz" && split -b 40M -d "$UP/llama-server.gz" "$UP/llama-server.gz.part"; }
for part in "$UP"/llama-server.gz.part*; do
  [ -e "$part" ] || continue
  ok=0
  for try in 1 2 3 4; do
    if timeout 600 colab upload -s "$S" "$part" "/content/$(basename "$part")" 2>&1 | tail -1 | grep -qv -i "fail\|error"; then ok=1; break; fi
    echo "upload of $(basename "$part") failed (try $try)"; sleep 10
  done
  [ "$ok" = 1 ] || { echo "UPLOAD FAILED: $(basename "$part")"; exit 3; }
  echo "uploaded $(basename "$part")"
done
# Inputs for the window (PIN_UPLOADS: space-separated local files), each to /content/<basename>, retried like the parts.
for f in ${PIN_UPLOADS:-}; do
  ok=0
  for try in 1 2 3 4; do
    if timeout 600 colab upload -s "$S" "$f" "/content/$(basename "$f")" 2>&1 | tail -1 | grep -qv -i "fail\|error"; then ok=1; break; fi
    echo "upload of $(basename "$f") failed (try $try)"; sleep 10
  done
  [ "$ok" = 1 ] || { echo "UPLOAD FAILED: $(basename "$f")"; exit 3; }
  echo "uploaded $(basename "$f") $(sha256sum "$f" | cut -c1-16)"
done
t0=$(date +%s)
# Every PIN_* variable reaches the VM (window 10 failed when PIN_RND_COMMIT wasn't on a hand-kept list), except the
# window script's own settings, which only matter here.
ENV_ARGS=()
for v in $(compgen -e | grep '^PIN_' | grep -v -E '^PIN_(CAP_UNITS|WALL_S|SESSION|GPU|UPLOADS|DOWNLOADS|WATCH_S|STALL_FIRST_S|STALL_S|CAP_MARGIN|DL_TIMEOUT|SKIP_BINARY|JOB_SCRIPT)$'); do
  ENV_ARGS+=(--env "$v=${!v}")
done
[ -n "${PIN_ROLE:-}" ] || ENV_ARGS+=(--env "PIN_ROLE=smoke")
timeout --signal=SIGINT --kill-after=30s "$WALL_S" colab exec -s "$S" "${ENV_ARGS[@]}" -f "$JOB" --timeout "$WALL_S" \
  > "$OUT/smoke-window.log" 2>&1 &
EXEC=$!
echo $EXEC > "$OUT/.exec.pid"
wait $EXEC
echo "exec exit=$? wall=$(( $(date +%s)-t0 ))s end=$(cat "$OUT/.end" 2>/dev/null || echo exec)"
# The exit sequence: fetch, then the trap stops the session. The watcher stays up as the cap's ceiling meanwhile.
[ -f "$OUT/.end" ] || echo exec > "$OUT/.end"
if [ "$(cat "$OUT/.end" 2>/dev/null)" != gone ]; then
  fetch_log final && echo "fetched job.log ($(stat -c %s "$OUT/job.log") bytes)" || echo "job.log not refreshed; the last good copy stays"
  for f in ${PIN_DOWNLOADS:-pin-result.json lib.tar generations.jsonl}; do
    timeout "$DL_S" colab download -s "$S" "/content/pin/$f" "$OUT/$f" 2>&1 | tail -1
  done
fi
kill "$WATCH" 2>/dev/null
rm -f "$OUT"/job.log.part.*
grep -h "^PIN_RESULT" "$OUT/job.log" "$OUT/smoke-window.log" 2>/dev/null | head -1 | cut -c1-600
echo "end balance $(bal) (start $START)"
