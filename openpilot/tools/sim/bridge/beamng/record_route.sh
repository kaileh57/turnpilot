#!/usr/bin/env bash
# record_route.sh <name> <dir> <route_drive args...>: a recorded routed drive, start to composed demo video. On a fresh
# stack without its UI, record_ui.py records the UI while route_drive.py drives (no chase camera: it costs the driving
# cameras 6 to 9 pairs a second, and the turns fail at that rate). Then route_stats.py scores the runs, and for each run
# chase_render.py renders the chase view from the pose log and demo_video.py composes <dir>/<name>_run<N>.mp4.
# Count a run only if its stats line says model >= 18 Hz and detector up.
#   bash openpilot/tools/sim/bridge/beamng/record_route.sh town15b2 ~/turnpilot-artifacts/demo/final --route town15b2 --runs 1
set -u
NAME=$1; OUT=$2; shift 2
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$( cd "$DIR/../../../../.." && pwd )"
PY=$ROOT/.venv/bin/python
mkdir -p "$OUT"
bash "$DIR/restart_stack.sh" "$OUT/${NAME}_stack.log" ui > /dev/null 2>&1
sleep 5
cd "$ROOT"
export OPENPILOT_PREFIX=${OPENPILOT_PREFIX:-tph1} PYTHONPATH=$ROOT
$PY -m openpilot.tools.turnpilot.record_ui "$OUT/${NAME}_ui.mp4" > "$OUT/${NAME}_ui.log" 2>&1 &
UI=$!
$PY -m openpilot.tools.sim.bridge.beamng.route_drive --out "$OUT/$NAME.json" "$@" --no-keyboard < /dev/null > "$OUT/$NAME.log" 2>&1
echo "route_drive exit $?"
kill -INT $UI; wait $UI
$PY -m openpilot.tools.sim.bridge.beamng.route_stats "$OUT/$NAME.json"
RUNS=$($PY -c "import json, sys; print(len(json.load(open(sys.argv[1]))['runs']))" "$OUT/$NAME.json" 2>/dev/null || echo 0)
for k in $(seq 1 "$RUNS"); do
  $PY -m openpilot.tools.sim.bridge.beamng.chase_render "$OUT/$NAME.json" --run "$k" -o "$OUT/${NAME}_run${k}_chase.mp4" \
    > "$OUT/${NAME}_run${k}_chase.log" 2>&1 || { echo "chase render of run $k failed, see ${NAME}_run${k}_chase.log"; continue; }
  $PY -m openpilot.tools.turnpilot.demo_video --chase "$OUT/${NAME}_run${k}_chase.mp4" --ui "$OUT/${NAME}_ui.mp4" \
    --result "$OUT/${NAME}_stats.json" --run "$k" -o "$OUT/${NAME}_run${k}.mp4"
done
