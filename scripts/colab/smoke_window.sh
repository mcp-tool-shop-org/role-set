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
if [ "${PIN_STUB:-0}" = 1 ] && [ "${PIN_GPU:-A100}" != CPU ]; then
  echo "REFUSED: PIN_STUB=1 may only run on PIN_GPU=CPU, never on ${PIN_GPU:-A100}" >&2
  exit 2
fi
if [ "${PIN_GPU:-A100}" != CPU ] && [ "${PIN_GATE:-1}" != 0 ] && [ -z "${PIN_GATE_MIN_DISK_GB:-}" ]; then
  # The gate's disk floor is the window's computed peak + 20 GB; without it a too-small VM is found only after the token
  # and the paid time (R&D on 346ea2d). A CPU window may leave it at its default.
  echo "REFUSED: a ${PIN_GPU:-A100} window with the VM gate on needs PIN_GATE_MIN_DISK_GB (its peak + 20 GB)" >&2
  exit 2
fi
if [ -n "${PIN_HF_TOKEN_FROM:-}" ]; then
  # Scoped tokens only (R&D, 2026-10-18): a pilot window reads with HF_TOKEN_GGUF_READ, a prebuild writes with
  # HF_TOKEN_GGUF_WRITE, each fine-grained to the one private repo. The account-wide HF_TOKEN never goes to a VM.
  # Checked before any VM is asked for; presence only, never the value.
  case "${PIN_ROLE:-}:$PIN_HF_TOKEN_FROM" in
    pilot:HF_TOKEN_GGUF_READ|prebuild:HF_TOKEN_GGUF_WRITE) ;;
    *) echo "REFUSED: PIN_ROLE=${PIN_ROLE:-unset} may not send $PIN_HF_TOKEN_FROM (pilot: HF_TOKEN_GGUF_READ; prebuild: HF_TOKEN_GGUF_WRITE)"; exit 2;;
  esac
  tokvar=$PIN_HF_TOKEN_FROM
  [ -n "${!tokvar:-}" ] || { echo "PIN_HF_TOKEN_FROM names $tokvar, which is empty on the rig"; exit 3; }
fi
CAP_UNITS="${PIN_CAP_UNITS:-8}"
WALL_S="${PIN_WALL_S:-5400}"
S=${PIN_SESSION:-pinw2}
OPEN_LOCK="${PIN_OPEN_LOCK:-/tmp/colab-window-open.lock}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"; rm -f "$OUT/.exec.pid" "$OUT/.end" "$OUT/job.log" "$OUT/.retry" "$OUT/.retried"
JOB="$here/${PIN_JOB_SCRIPT:-pin_window.py}"  # mock_job.py for a test of this script on a cheap runtime
MARGIN="${PIN_CAP_MARGIN:-0.3}"  # units kept back at the cap to fetch the results (R&D, 2026-10-17)
DL_S="${PIN_DL_TIMEOUT:-300}"
export COLUMNS=150
bal() { colab usage 2>&1 | grep -oP 'balance: \K[0-9.]+'; }
# Stops this window's own session only: another window (or a probe) may be running beside it.
# Holds the open lock while stopping, so a neighbour's rate reads never straddle a stop (R&D, from Argon's review).
TOKF=""  # the rig-side token file, if this window sends one: removed in the trap too, so a crash can't leave it
stop_all() {
  [ -n "$TOKF" ] && rm -f "$TOKF"
  (
    # Bounded: the EXIT trap can fire while this script still holds the open lock, and an unbounded wait would never
    # stop the session (R&D on #89). After 60 s it stops the session anyway.
    flock -w 60 9 || echo "open lock not free after 60 s; stopping $S anyway"
    echo "stopping $S"; colab stop -s "$S" 2>&1 | tail -1
    sleep 5; echo "sessions after stop:"; colab sessions 2>&1 | tail -3
  ) 9>"$OPEN_LOCK"
}
trap stop_all EXIT
rate() { colab usage 2>&1 | grep -oP 'rate: \K[0-9.]+'; }
# Windows on this rig open one at a time: the rate read before, the open and the rate read after hold one lock, so a
# neighbour opening in between can't land in this window's rate (windows 11a and 11b each counted both sessions).
exec 9>"$OPEN_LOCK"; flock 9
START=$(bal); RATE0=$(rate); echo "start balance $START, account rate $RATE0/hr $(date -u +%FT%TZ)"
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
GATE_TRIES="${PIN_GATE_TRIES:-3}"; GATE_S="${PIN_GATE_TIMEOUT_S:-180}"
GATE_ENV=(--env "PIN_GATE_GPU=$([ "${PIN_GPU:-A100}" = CPU ] && echo 0 || echo 1)")
for v in $(compgen -e | grep '^PIN_GATE_' | grep -v -E '^PIN_GATE_(TRIES|TIMEOUT_S|GPU)$'); do GATE_ENV+=(--env "$v=${!v}"); done
for try in $(seq 1 "$GATE_TRIES"); do
  timeout 300 colab new "${GPU_ARGS[@]}" -s "$S" 2>&1 | tail -2 || { flock -u 9; exec 9>&-; echo "no ${PIN_GPU:-A100} assigned"; exit 2; }
  sleep 15
  [ "${PIN_GATE:-1}" = 0 ] && break
  line=$(timeout "$GATE_S" colab exec -s "$S" "${GATE_ENV[@]}" -f "$here/vm_gate.py" --timeout "$GATE_S" 2>&1 | grep -m1 '^VM_GATE ')
  row=${line#VM_GATE }; [ -n "$line" ] || row='{"ok": false, "why": ["the gate exec died or timed out before it printed"]}'
  printf '{"try": %s, "gpu": "%s", "gate": %s}\n' "$try" "${PIN_GPU:-A100}" "$row" >> "$OUT/gate.jsonl"
  if printf '%s' "$row" | grep -q '"ok": true'; then echo "VM gate passed (try $try)"; break; fi
  echo "VM gate rejected try $try: $(printf '%s' "$row" | grep -o '"why": \[[^]]*\]')"
  colab stop -s "$S" 2>&1 | tail -1; sleep 5
  if [ "$try" = "$GATE_TRIES" ]; then flock -u 9; exec 9>&-; echo "VM GATE: no healthy ${PIN_GPU:-A100} in $GATE_TRIES tries"; exit 4; fi
done
colab usage 2>&1 | grep -E "rate|assign"
# The cap counts this window's own spend: its rate (the account rate after its session opened, less the rate before) times
# its own runtime. The account balance would count a neighbour's spend too (window 8 was cut off that way).
# A neighbour closing between the two reads can make the difference 0 or less: then take the runtime's own known rate,
# not the whole account's (R&D, from Argon's review; the account rate would stop this window early).
case "${PIN_GPU:-A100}" in A100) GPU_RATE=5.3;; CPU) GPU_RATE=0.08;; *) GPU_RATE="";; esac
OWN_RATE=$(awk -v a="$(rate)" -v b="$RATE0" -v g="$GPU_RATE" 'BEGIN{r=a-b; if (r<=0) r=(g!="" ? g : a); print r}'); T_OPEN=$(date +%s)
echo "own rate $OWN_RATE/hr"
# The job's deadline (R&D, 2026-10-18): own spend at the cap less MARGIN. Past it the job starts nothing new, cuts a wave
# under way (SIGTERM, then SIGKILL) and writes pilot-out.tar and pin-result.json, so a cap stop returns results, not only
# job.log. The watcher waits PIN_DEADLINE_GRACE_S past it for the PIN_RESULT line before it kills the exec.
PIN_DEADLINE=$(awk -v t="$T_OPEN" -v c="$CAP_UNITS" -v m="$MARGIN" -v r="$OWN_RATE" 'BEGIN{printf "%d", t + (c-m)/r*3600}')
export PIN_DEADLINE
GRACE_S="${PIN_DEADLINE_GRACE_S:-180}"
echo "deadline $(date -u -d "@$PIN_DEADLINE" +%T) (cap $CAP_UNITS less $MARGIN at $OWN_RATE/hr); grace ${GRACE_S}s"
# This window's token: the job writes it into pin-result.json, and a second exec stops only on this window's result.
export PIN_JOB_TOKEN="$S-$T_OPEN"
flock -u 9; exec 9>&-
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
( seen=0; size0=-1; grew=0; ended=""; idle=0; noted=""
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
      st=$(timeout 60 colab status -s "$S" 2>&1 | tr -s ' \n' ' ' | cut -c1-300)
      echo "$(date -u +%T) $st" >> "$OUT/status.log"
      size=$(stat -c %s "$OUT/job.log" 2>/dev/null || echo 0)
      # The exec never reached the kernel (windows 11c, 12e: "Connection was lost" at the first command): the kernel
      # reads IDLE and job.log is absent or empty at the first look. Re-run the exec once on the same session, keeping
      # its uploads (R&D, 2026-10-17). A second time stalls as before; the job's lock stops a late first exec.
      if [ ! -f "$OUT/.retried" ] && [ "$size" = 0 ] && echo "$st" | grep -q "Status: IDLE"; then
        echo "EXEC NEVER STARTED ($S): kernel IDLE, job.log empty; re-running the exec once $(date -u +%T)"
        touch "$OUT/.retried" "$OUT/.retry"; kill "$(cat "$OUT/.exec.pid")" 2>/dev/null; continue
      fi
      if [ "$size" -gt 0 ] && [ "$size" != "$size0" ]; then seen=1; grew=$now; size0=$size; fi
      if grep -q "^PIN_RESULT" "$OUT/job.log" 2>/dev/null; then
        echo "JOB DONE (PIN_RESULT in job.log) $(date -u +%T)"; end_window done; continue
      fi
      # The job died with its kernel (R&D, 2026-10-18; most likely an OOM kill): the kernel reads IDLE, job.log has lines
      # and no PIN_RESULT, at two checks running. A child killed by OOM (a server, a converter) raises in the job instead,
      # and its finally writes the result. Two checks, so one stale status read can't end a window.
      if [ "$size" -gt 0 ] && echo "$st" | grep -q "Status: IDLE"; then idle=$(( idle + 1 )); else idle=0; fi
      if [ "$idle" -ge 2 ]; then
        echo "JOB DIED ($S): kernel IDLE at two checks, job.log without PIN_RESULT (likely OOM) $(date -u +%T)" | tee "$OUT/died.txt"
        end_window died; continue
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
      if [ "$now" -lt $(( PIN_DEADLINE + GRACE_S )) ]; then
        # The job cuts its own wave at the deadline and writes its results; its PIN_RESULT line ends the window above.
        [ -n "$noted" ] || echo "DEADLINE (own spend $spent of $CAP_UNITS): the job cuts its wave and writes its results; up to ${GRACE_S}s $(date -u +%T)"
        noted=1
      else
        echo "CAP REACHED (own spend $spent of $CAP_UNITS; $MARGIN kept back to fetch the results) $(date -u +%T)"; end_window cap
      fi
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
# The private GGUF repo's token (R&D, 2026-10-18), only for a window that names one: PIN_HF_TOKEN_FROM is the name of
# the rig's environment variable Mike set (the read token for a pilot window, the write token for a prebuild). It goes
# over as a file, written owner-only and deleted here at once: never on a command line, never through --env, never in a
# log. The job reads it once, deletes it and drops it before any wave. It goes up only after the VM gate passed, so a
# rejected VM never receives it. If the job dies before reading it, the file stays at /dev/shm/hf.token on a VM that
# this script is stopping anyway (the EXIT trap); nothing fetches it back. On the VM it goes to RAM-backed /dev/shm,
# outside /content and every tar.
if [ -n "${PIN_HF_TOKEN_FROM:-}" ]; then
  tokvar=$PIN_HF_TOKEN_FROM
  TOKF=$(mktemp); chmod 600 "$TOKF"; printf '%s' "${!tokvar}" > "$TOKF"; tf=$TOKF
  ok=0
  for try in 1 2 3; do
    if timeout 120 colab upload -s "$S" "$tf" /dev/shm/hf.token 2>&1 | tail -1 | grep -qv -i "fail\|error"; then ok=1; break; fi
    sleep 5
  done
  rm -f "$tf"; TOKF=""
  [ "$ok" = 1 ] || { echo "UPLOAD FAILED: the token file"; exit 3; }
  echo "uploaded the token file (owner-only, deleted on the rig)"
fi
t0=$(date +%s)
# Every PIN_* variable reaches the VM (window 10 failed when PIN_RND_COMMIT wasn't on a hand-kept list), except the
# window script's own settings, which only matter here.
ENV_ARGS=()
for v in $(compgen -e | grep '^PIN_' | grep -v -E '^PIN_(CAP_UNITS|WALL_S|SESSION|GPU|UPLOADS|DOWNLOADS|WATCH_S|STALL_FIRST_S|STALL_S|CAP_MARGIN|DL_TIMEOUT|SKIP_BINARY|JOB_SCRIPT|OPEN_LOCK|HF_TOKEN_FROM|HIGH_MEM|DEADLINE_GRACE_S)$'); do
  ENV_ARGS+=(--env "$v=${!v}")
done
[ -n "${PIN_ROLE:-}" ] || ENV_ARGS+=(--env "PIN_ROLE=smoke")
run_exec() {
  timeout --signal=SIGINT --kill-after=30s "$WALL_S" colab exec -s "$S" "${ENV_ARGS[@]}" -f "$JOB" --timeout "$WALL_S" \
    >> "$OUT/smoke-window.log" 2>&1 &
  EXEC=$!
  echo $EXEC > "$OUT/.exec.pid"
}
run_exec
wait $EXEC; rc=$?
if [ -f "$OUT/.retry" ]; then  # the watcher saw the exec never start: once more, same session
  rm -f "$OUT/.retry"; run_exec; wait $EXEC; rc=$?
fi
echo "exec exit=$rc wall=$(( $(date +%s)-t0 ))s end=$(cat "$OUT/.end" 2>/dev/null || echo exec)"
# The exit sequence: fetch, then the trap stops the session. The watcher stays up as the cap's ceiling meanwhile.
[ -f "$OUT/.end" ] || echo exec > "$OUT/.end"
if [ "$(cat "$OUT/.end" 2>/dev/null)" != gone ]; then
  f0=$(date +%s)
  fetch_log final && echo "fetched job.log ($(stat -c %s "$OUT/job.log") bytes)" || echo "job.log not refreshed; the last good copy stays"
  for f in ${PIN_DOWNLOADS:-pin-result.json pilot-out.tar lib.tar generations.jsonl}; do
    timeout "$DL_S" colab download -s "$S" "/content/pin/$f" "$OUT/$f" 2>&1 | tail -1
  done
  # The exit sequence's fetch time, the measure for PIN_CAP_MARGIN.
  echo "exit fetch took $(( $(date +%s) - f0 ))s"
fi
kill "$WATCH" 2>/dev/null
rm -f "$OUT"/job.log.part.*
grep -h "^PIN_RESULT" "$OUT/job.log" "$OUT/smoke-window.log" 2>/dev/null | head -1 | cut -c1-600
echo "end balance $(bal) (start $START)"
# The window's outcome record (R&D, 2026-10-18: every outcome.json carries its date and commit). The rig writes what it
# measured; the outcome class and its why are filled in by whoever reads the window, in outcome.json.
spent_end=$(awk -v r="$OWN_RATE" -v t="$(( $(date +%s) - T_OPEN ))" 'BEGIN{printf "%.2f", r*t/3600}')
printf '{"window": "%s", "date": "%s", "commit": "%s", "card": "%s", "cap": %s, "spent": %s, "basis": "rate_x_runtime", "end": "%s", "outcome": null, "why": null}\n' \
  "$(basename "$OUT")" "$(date -u +%F)" "${PIN_ROLESET_COMMIT:-unset}" "${PIN_GPU:-A100}" "$CAP_UNITS" "$spent_end" \
  "$(cat "$OUT/.end" 2>/dev/null || echo exec)" > "$OUT/outcome.draft.json"
