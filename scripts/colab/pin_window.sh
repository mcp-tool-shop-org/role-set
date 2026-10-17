#!/usr/bin/env bash
# Window 1 (pinning), run from the rig in WSL Ubuntu, where the colab CLI is installed:
#   MSYS_NO_PATHCONV=1 wsl -d Ubuntu -e bash -l scripts/colab/pin_window.sh <out dir>
# One A100 40 GB. A unit cap stops the window early; every step is bounded; the VM is stopped in a trap and the
# sessions list is checked empty at the end. Writes <out dir>/pin-result.json and <out dir>/llama-server.
set -u
OUT="${1:?usage: pin_window.sh <out dir>}"
CAP_UNITS="${PIN_CAP_UNITS:-7}"   # about 60 min at 5.30 units/h, with margin
WALL_S="${PIN_WALL_S:-4800}"      # 80 min outer timeout for the VM-side script
S=pin
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
timeout 300 colab new --gpu A100 -s "$S" 2>&1 | tail -2 || { echo "no A100 assigned"; exit 2; }
sleep 20
colab usage 2>&1 | grep -E "rate|assign"
# The cap: a background watcher stops the VM if the balance falls CAP_UNITS below the start.
( while sleep 120; do now=$(bal); awk -v s="$START" -v n="$now" -v c="$CAP_UNITS" 'BEGIN{exit !(s-n>c)}' && { echo "CAP REACHED ($START -> $now)"; colab stop -s "$S"; exit 0; }; done ) &
WATCH=$!
t0=$(date +%s)
timeout --signal=SIGINT --kill-after=30s "$WALL_S" colab exec -s "$S" -f "$here/pin_window.py" --timeout "$WALL_S" 2>&1 | tee "$OUT/pin-window.log" | grep -E "^\[|PIN_RESULT" 
echo "exec exit=${PIPESTATUS[0]} wall=$(( $(date +%s)-t0 ))s"
kill "$WATCH" 2>/dev/null
timeout 600 colab download -s "$S" /content/pin/pin-result.json "$OUT/pin-result.json" 2>&1 | tail -1
timeout 900 colab download -s "$S" /content/pin/llama-server "$OUT/llama-server" 2>&1 | tail -1
timeout 900 colab download -s "$S" /content/pin/lib.tar "$OUT/lib.tar" 2>&1 | tail -1
[ -f "$OUT/llama-server" ] && sha256sum "$OUT/llama-server"
[ -f "$OUT/lib.tar" ] && tar -tf "$OUT/lib.tar"
echo "end balance $(bal) (start $START)"
