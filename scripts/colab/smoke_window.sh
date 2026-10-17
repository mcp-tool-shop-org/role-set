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
mkdir -p "$OUT"
export COLUMNS=150
bal() { colab usage 2>&1 | grep -oP 'balance: \K[0-9.]+'; }
stop_all() {
  for s in $(colab sessions 2>&1 | grep -oP '^\[\K[^]]+'); do echo "stopping $s"; colab stop -s "$s" 2>&1 | tail -1; done
  sleep 5; echo "sessions after stop:"; colab sessions 2>&1 | tail -3
}
trap stop_all EXIT
START=$(bal); echo "start balance $START $(date -u +%FT%TZ)"
sha256sum "$BIN"
timeout 300 colab new --gpu "${PIN_GPU:-A100}" -s "$S" 2>&1 | tail -2 || { echo "no A100 assigned"; exit 2; }
sleep 15; colab usage 2>&1 | grep -E "rate|assign"
( while sleep 120; do now=$(bal); awk -v s="$START" -v n="$now" -v c="$CAP_UNITS" 'BEGIN{exit !(s-n>c)}' && { echo "CAP REACHED ($START -> $now)"; stop_all; exit 0; }; done ) &
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
timeout --signal=SIGINT --kill-after=30s "$WALL_S" colab exec -s "$S" --env "PIN_ROLE=${PIN_ROLE:-smoke}" --env "PIN_ROLESET_COMMIT=${PIN_ROLESET_COMMIT:-}" --env "PIN_ONLY=${PIN_ONLY:-}" --env "PIN_SMOKE_CTX=${PIN_SMOKE_CTX:-}" --env "PIN_SMOKE_MAX_TOKENS=${PIN_SMOKE_MAX_TOKENS:-}" --env "PIN_CARDS_JSON=${PIN_CARDS_JSON:-}" -f "$here/pin_window.py" --timeout "$WALL_S" \
  > "$OUT/smoke-window.log" 2>&1
echo "exec exit=$? wall=$(( $(date +%s)-t0 ))s"
kill "$WATCH" 2>/dev/null
for f in pin-result.json lib.tar generations.jsonl; do
  timeout 900 colab download -s "$S" "/content/pin/$f" "$OUT/$f" 2>&1 | tail -1
done
grep -h "PIN_RESULT" "$OUT/smoke-window.log" | cut -c1-600
echo "end balance $(bal) (start $START)"
