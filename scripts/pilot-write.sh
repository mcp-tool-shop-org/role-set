#!/bin/sh
# A paid pilot's write, launched from a merged, pinned checkout of role-set.
#
#   RS_HEARTBEAT=<watchdog heartbeat file> RS_EXEMPLAR_REPO=<aspire-si clone> \
#     sh scripts/pilot-write.sh <lane port> <critic grant> <pod deadline HH:MM UTC> <work dir>
#
# The work dir holds plan.json, pack/ (a synced pack) and pool.jsonl (harvested excerpts); the run writes to
# <work dir>/out with budget.json beside it. The budget is the minutes left to the deadline minus 8 (in-flight
# slots and shutdown), capped at 40; the process is killed 2 minutes before the deadline whatever happens. On every
# exit the critic is unloaded and budget.json records how the run ended (kill, systemic stop, budget or queue)
# and whether the card is free. Standards: the runner's andon and pins; this script adds the time box and the unload.
set -u
PORT="$1"; GRANT="$2"; DEADLINE="$3"; WORK="$4"
: "${RS_HEARTBEAT:?set RS_HEARTBEAT to the watchdog heartbeat file}"
: "${RS_EXEMPLAR_REPO:?set RS_EXEMPLAR_REPO to an aspire-si clone}"
CRITIC="${RS_CRITIC:-gemma4:31b}"
EXEMPLAR_COMMIT="${RS_EXEMPLAR_COMMIT:-a7ec78393a82681278f73a7a7e6ccbda20a1ac73}"
CONCURRENCY="${RS_CONCURRENCY:-8}"
BIN=.venv/Scripts/role-set.exe; [ -x "$BIN" ] || BIN=.venv/bin/role-set
mkdir -p "$WORK/out"

DL=$(date -u -d "$(date -u +%Y-%m-%d) $DEADLINE:00" +%s); NOW=$(date -u +%s)
MIN=$(( (DL - NOW) / 60 - 8 )); [ "$MIN" -gt 40 ] && MIN=40
[ "$MIN" -lt 5 ] && { echo "refused: only $MIN budget minutes left before $DEADLINE UTC"; exit 3; }
KILL=$(( DL - NOW - 120 ))
echo "{\"deadline_utc\": \"$DEADLINE\", \"start_epoch\": $NOW, \"stop_after_minutes\": $MIN, \"kill_after_s\": $KILL, \"commit\": \"$(git rev-parse HEAD)\"}" > "$WORK/budget.json"

timeout -k 15 "$KILL" "$BIN" write --plan "$WORK/plan.json" --pack "$WORK/pack" --out "$WORK/out" --stop-after-minutes "$MIN" \
  --endpoint "http://127.0.0.1:$PORT/v1" --model qwen3-235b-instruct-2507 --critic-endpoint http://127.0.0.1:11434 --critic-model "$CRITIC" \
  --grant "$GRANT" --heartbeat "$RS_HEARTBEAT" --concurrency "$CONCURRENCY" --pool "$WORK/pool.jsonl" \
  --exemplar-repo "$RS_EXEMPLAR_REPO" --exemplar-commit "$EXEMPLAR_COMMIT" --exemplar-files pilot-t04.jsonl batch2-t01.jsonl \
  batch2-t14.jsonl batch2-t16.jsonl batch3-t05.jsonl batch3-t13.jsonl batch3-t17.jsonl batch4-t02.jsonl batch4-t03.jsonl batch4-t06.jsonl \
  > "$WORK/run.log" 2>&1
RC=$?; END=$(date -u +%s)

curl -s -m 30 http://127.0.0.1:11434/api/generate -d "{\"model\": \"$CRITIC\", \"keep_alive\": 0}" > /dev/null
PS=$(ollama ps 2>&1 | tail -n +2)
python - "$WORK" "$RC" "$END" "$PS" <<'PY'
import json, sys, pathlib
d, rc, end, ps = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].strip()
b = json.loads((d / "budget.json").read_text())
reports = sorted((d / "out").rglob("batch-report.json"))
st = json.loads(reports[0].read_text()).get("status", {}) if reports else {}
ended = ("kill" if rc == 124 or not reports else "systemic stop" if st.get("halted")
         else "budget" if st.get("skipped", {}).get("time budget") else "queue")
b.update(end_epoch=end, elapsed_s=end - b["start_epoch"], exit_code=rc, ended_by=ended, halted=st.get("halted"),
         critic_unloaded=not ps, ollama_ps_after=ps or "empty")
(d / "budget.json").write_text(json.dumps(b, indent=1))
print(json.dumps(b))
PY
echo "rc=$RC"; tail -3 "$WORK/run.log"
