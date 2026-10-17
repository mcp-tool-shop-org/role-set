#!/usr/bin/env bash
# Window 1 split across two VMs side by side (Mike, 2026-10-16), from the rig in WSL Ubuntu:
#   MSYS_NO_PATHCONV=1 wsl -d Ubuntu -e bash -l scripts/colab/pin_window_split.sh <out dir>
# VM "pinb" (A100): build llama-server, pin and smoke the maker GGUFs (PIN_ROLE=build).
# VM "pinc" (L4):   convert the BF16 models to Q8_0 and hash them (PIN_ROLE=convert); CPU and disk only.
# One unit cap across both; both stopped in a trap; the sessions list checked empty at the end.
set -u
OUT="${1:?usage: pin_window_split.sh <out dir>}"
CAP_UNITS="${PIN_CAP_UNITS:-7}"
WALL_S="${PIN_WALL_S:-4800}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT/build" "$OUT/convert"
export COLUMNS=150
bal() { colab usage 2>&1 | grep -oP 'balance: \K[0-9.]+'; }
stop_all() {
  for s in $(colab sessions 2>&1 | grep -oP '^\[\K[^]]+'); do echo "stopping $s"; colab stop -s "$s" 2>&1 | tail -1; done
  sleep 5; echo "sessions after stop:"; colab sessions 2>&1 | tail -3
}
trap stop_all EXIT
START=$(bal); echo "start balance $START $(date -u +%FT%TZ)"
timeout 300 colab new --gpu A100 -s pinb 2>&1 | tail -2 || { echo "no A100 assigned"; exit 2; }
if ! timeout 300 colab new --gpu L4 --high-mem -s pinc 2>&1 | tail -2; then
  echo "no second VM: the convert step runs after the build on pinb"; SECOND=0
else SECOND=1; fi
sleep 20; colab usage 2>&1 | grep -E "rate|assign"
( while sleep 120; do now=$(bal); awk -v s="$START" -v n="$now" -v c="$CAP_UNITS" 'BEGIN{exit !(s-n>c)}' && { echo "CAP REACHED ($START -> $now)"; stop_all; exit 0; }; done ) &
WATCH=$!
run() {  # session role outdir
  timeout --signal=SIGINT --kill-after=30s "$WALL_S" colab exec -s "$1" --env "PIN_ROLE=$2" -f "$here/pin_window.py" --timeout "$WALL_S" \
    > "$3/pin-window.log" 2>&1
  echo "$1 ($2) exit=$? $(date -u +%T)"
  timeout 600 colab download -s "$1" /content/pin/pin-result.json "$3/pin-result.json" 2>&1 | tail -1
}
t0=$(date +%s)
if [ "$SECOND" = 1 ]; then
  run pinb build "$OUT/build" & B=$!
  run pinc convert "$OUT/convert" & C=$!
  wait "$B" "$C"
else
  run pinb build "$OUT/build"; run pinb convert "$OUT/convert"
fi
echo "both done, wall=$(( $(date +%s)-t0 ))s"
kill "$WATCH" 2>/dev/null
timeout 900 colab download -s pinb /content/pin/llama-server "$OUT/llama-server" 2>&1 | tail -1
timeout 900 colab download -s pinb /content/pin/lib.tar "$OUT/lib.tar" 2>&1 | tail -1
[ -f "$OUT/llama-server" ] && sha256sum "$OUT/llama-server"
grep -h "PIN_RESULT" "$OUT"/*/pin-window.log | cut -c1-400
echo "end balance $(bal) (start $START)"
