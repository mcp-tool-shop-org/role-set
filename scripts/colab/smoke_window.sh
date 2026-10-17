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
mkdir -p "$OUT"; rm -f "$OUT/.exec.pid"
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
timeout 300 colab new --gpu "${PIN_GPU:-A100}" -s "$S" 2>&1 | tail -2 || { echo "no A100 assigned"; exit 2; }
sleep 15; colab usage 2>&1 | grep -E "rate|assign"
# The cap counts this window's own spend: its rate (the account rate after its session opened, less the rate before) times
# its own runtime. The account balance would count a neighbour's spend too (window 8 was cut off that way).
OWN_RATE=$(awk -v a="$(rate)" -v b="$RATE0" 'BEGIN{r=a-b; if (r<=0) r=a; print r}'); T_OPEN=$(date +%s)
echo "own rate $OWN_RATE/hr"
flock -u 9
# The watcher also ends the exec when Colab has dropped the session: window 10b's VM vanished mid-download and the exec
# would have waited out its wall time on nothing.
# It also watches the job make progress (R&D, 2026-10-17, after window 11c's exec hung on a lost connection with nothing
# running on the VM): no line in the exec's log within 10 minutes of the exec starting, or none for 20 minutes after
# that, closes the window's own session and records a stall. Every finished model call prints a line, and the slowest
# silent step (a conversion) takes about 12 minutes.
( seen=0; size0=-1; grew=0
  while sleep "${PIN_WATCH_S:-120}"; do
    if ! colab sessions 2>&1 | grep -q "^\\[$S\\]"; then
      echo "SESSION GONE ($S) $(date -u +%T)"; [ -f "$OUT/.exec.pid" ] && kill "$(cat "$OUT/.exec.pid")" 2>/dev/null; exit 0
    fi
    if [ -f "$OUT/.exec.pid" ]; then
      now=$(date +%s); size=$(stat -c %s "$OUT/smoke-window.log" 2>/dev/null || echo 0)
      if [ "$size" -gt 0 ] && [ "$size" != "$size0" ]; then seen=1; grew=$now; size0=$size; fi
      started=$(stat -c %Y "$OUT/.exec.pid")
      stalled=""
      [ "$seen" = 0 ] && [ $(( now - started )) -gt "${PIN_STALL_FIRST_S:-600}" ] && stalled="no line in the 10 minutes after the exec started"
      [ "$seen" = 1 ] && [ $(( now - grew )) -gt "${PIN_STALL_S:-1200}" ] && stalled="no new line for 20 minutes"
      if [ -n "$stalled" ]; then
        echo "STALL ($S): $stalled $(date -u +%T)" | tee "$OUT/stall.txt"
        kill "$(cat "$OUT/.exec.pid")" 2>/dev/null; stop_all; exit 0
      fi
    fi
    spent=$(awk -v r="$OWN_RATE" -v t="$(( $(date +%s) - T_OPEN ))" 'BEGIN{printf "%.2f", r*t/3600}')
    awk -v s="$spent" -v c="$CAP_UNITS" 'BEGIN{exit !(s>c)}' && { echo "CAP REACHED (own spend $spent of $CAP_UNITS)"; stop_all; exit 0; }; done ) &
WATCH=$!
# The binary goes up gzipped in 40 MB parts, each retried; a part that never lands ends the window (the trap stops the VM).
UP="$OUT/upload"; mkdir -p "$UP"; rm -f "$UP"/llama-server.gz*
gzip -9 -c "$BIN" > "$UP/llama-server.gz" && split -b 40M -d "$UP/llama-server.gz" "$UP/llama-server.gz.part"
for part in "$UP"/llama-server.gz.part*; do
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
for v in $(compgen -e | grep '^PIN_' | grep -v -E '^PIN_(CAP_UNITS|WALL_S|SESSION|GPU|UPLOADS|DOWNLOADS|WATCH_S|STALL_FIRST_S|STALL_S)$'); do
  ENV_ARGS+=(--env "$v=${!v}")
done
[ -n "${PIN_ROLE:-}" ] || ENV_ARGS+=(--env "PIN_ROLE=smoke")
timeout --signal=SIGINT --kill-after=30s "$WALL_S" colab exec -s "$S" "${ENV_ARGS[@]}" -f "$here/pin_window.py" --timeout "$WALL_S" \
  > "$OUT/smoke-window.log" 2>&1 &
EXEC=$!
echo $EXEC > "$OUT/.exec.pid"
wait $EXEC
echo "exec exit=$? wall=$(( $(date +%s)-t0 ))s"
kill "$WATCH" 2>/dev/null
for f in ${PIN_DOWNLOADS:-pin-result.json lib.tar generations.jsonl}; do
  timeout 900 colab download -s "$S" "/content/pin/$f" "$OUT/$f" 2>&1 | tail -1
done
grep -h "PIN_RESULT" "$OUT/smoke-window.log" | cut -c1-600
echo "end balance $(bal) (start $START)"
