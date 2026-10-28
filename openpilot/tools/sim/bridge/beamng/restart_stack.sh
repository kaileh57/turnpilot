#!/usr/bin/env bash
# restart_stack.sh <log> [BLOCK]: a fresh openpilot for BeamNG drives (route_drive.py, scenarios/run.py): stop the manager
# and the children a killed manager orphans, wait until no process holds the chestnut (the eGPU modeld runs on), set the
# TurnPilot params, and start launch_openpilot.sh in the background. A fresh manager also clears the
# Offroad_ExcessiveActuation that a collision latches in selfdrived. BLOCK adds processes to leave out, e.g. "ui" when the
# UI is recorded on its own (record_ui.py). navd routes on TURNPILOT_TILES, by default the west_coast_usa_signals tiles
# (sim_tiles.sh): with them not built it stops here, before touching the running stack.
set -u
LOG=$1
BLK=${2:-}
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$( cd "$DIR/../../../../.." && pwd )"
TILES=$("$DIR/sim_tiles.sh") || exit 1
export TURNPILOT_TILES=$TILES
echo "tiles: $TURNPILOT_TILES"
cd "$ROOT"
source .venv/bin/activate
export OPENPILOT_PREFIX=${OPENPILOT_PREFIX:-tph1} PYTHONPATH=$ROOT
pkill -INT -f "system/manager/manager.py" 2>/dev/null
for i in $(seq 1 30); do
  pgrep -f "system/manager/manager.py|selfdrive.modeld.modeld" >/dev/null || break
  sleep 1
done
pkill -9 -f "system/manager/manager.py|selfdrive.modeld.modeld" 2>/dev/null
for p in $(pgrep -f "openpilot\.(system|selfdrive)\.|selfdrive/ui|system/ui"); do
  [ "$(ps -o ppid= -p "$p" | tr -d ' ')" = "1" ] && kill "$p"
done
# the chestnut's driver lists one user client per process that has it open; a modeld that is still dying holds it
for i in $(seq 1 20); do
  n=$(ioreg -p IOService -l -w0 -r -n "custom 8516b768-CLEAN" | grep -c UserClient)
  [ "$n" = "0" ] && break
  sleep 1
done
echo "chestnut user clients: $n"
mkdir -p "/tmp/msgq_$OPENPILOT_PREFIX"
python -c "
from openpilot.common.params import Params
p = Params(); p.remove('Offroad_ExcessiveActuation')
for k in ('NavigationEnabled', 'TurnPilotEnabled'): p.put_bool(k, True)
p.put_bool('TurnPilotShadowMode', False)
p.put('TurnPilotRouteSource', 'valhalla')
p.remove('NavDestination')
"
BLOCK=$BLK nohup openpilot/tools/sim/bridge/beamng/launch_openpilot.sh > "$LOG" 2>&1 &
sleep 3
pgrep -fl "system/manager/manager.py" | head -1
