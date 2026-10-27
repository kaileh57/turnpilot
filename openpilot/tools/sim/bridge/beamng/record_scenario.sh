#!/usr/bin/env bash
# record_scenario.sh <scenario> <dir> <scenarios/run.py args...>: a recorded scenario, start to composed demo videos. On a
# fresh stack without its UI, scenarios/run.py --record records the UI while the runs drive and renders each run's chase
# view after (chase_render.py); demo_video.py then composes <dir>/<scenario>-<time>/demo_run<N>.mp4 for every run.
# TURNPILOT_TILES picks navd's tiles (the lanes tiles for lane_change).
#   bash openpilot/tools/sim/bridge/beamng/record_scenario.sh ped_in_path ~/turnpilot-artifacts/demo/final --runs 2
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
$PY -m openpilot.tools.sim.bridge.beamng.scenarios.run "$NAME" --record "$OUT" "$@" --no-keyboard < /dev/null > "$OUT/$NAME.log" 2>&1
echo "scenario run exit $?"
RUN_DIR=$(sed -n 's/^results: \(.*\)\/result.json$/\1/p' "$OUT/$NAME.log" | tail -1)
[ -n "$RUN_DIR" ] || { echo "no results, see $OUT/$NAME.log"; exit 1; }
RUNS=$($PY -c "import json, sys; print(len(json.load(open(sys.argv[1]))['runs']))" "$RUN_DIR/result.json" 2>/dev/null || echo 0)
for k in $(seq 1 "$RUNS"); do
  [ -f "$RUN_DIR/chase_run$k.mp4" ] || { echo "run $k: no chase video"; continue; }
  $PY -m openpilot.tools.turnpilot.demo_video --chase "$RUN_DIR/chase_run$k.mp4" --ui "$RUN_DIR/ui.mp4" \
    --result "$RUN_DIR/result.json" --run "$k" -o "$RUN_DIR/demo_run$k.mp4"
done
$PY -m openpilot.tools.sim.bridge.beamng.scenarios.score "$RUN_DIR/result.json" | tail -20
