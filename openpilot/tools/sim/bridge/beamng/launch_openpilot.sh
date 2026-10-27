#!/usr/bin/env bash
# openpilot as a comma four in a Tesla Model 3, fed by the BeamNG bridge (run_bridge.py). Like tools/sim/launch_openpilot.sh,
# but a Model 3 with alpha longitudinal instead of a Civic, and loggerd left on so every run leaves an rlog.
# OPENPILOT_ROOT picks the (built) checkout to run; extra processes to block go in BLOCK, e.g. BLOCK=turnpilotd for gate1.py.

export PASSIVE="0"
export NOBOARD="1"
export SIMULATION="1"
export SKIP_FW_QUERY="1"
export FINGERPRINT="TESLA_MODEL_3"

# the bridge is camerad; encoderd writes lossless video on PC, too heavy for the Mac
export BLOCK="${BLOCK},camerad,encoderd,micd,logmessaged,manage_athenad"
if [[ "$CI" ]]; then
  export BLOCK="${BLOCK},ui"
fi

# msgq needs the prefix directory to exist (macOS keeps msgq in /tmp)
if [[ -n "$OPENPILOT_PREFIX" ]]; then
  mkdir -p "/tmp/msgq_$OPENPILOT_PREFIX"
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
OPENPILOT_ROOT="${OPENPILOT_ROOT:-$DIR/../../../../..}"
cd "$OPENPILOT_ROOT"
export PYTHONPATH="$OPENPILOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# navd routes on the sim map's own tiles, never on real roads (the default 'auto' falls back to Mapbox);
# TURNPILOT_TILES picks the tiles, e.g. ~/turnpilot_data/valhalla/west_coast_usa_lanes for the turn-lane demos
python3 -c "
import os
from openpilot.common.params import Params
from openpilot.selfdrive.test.helpers import set_params_enabled
set_params_enabled()
p = Params()
p.put_bool('AlphaLongitudinalEnabled', True)
p.put('TurnPilotRouteSource', 'valhalla')
tiles = os.environ.get('TURNPILOT_TILES') or p.get('TurnPilotValhallaTiles') or os.path.expanduser('~/turnpilot_data/valhalla/west_coast_usa')
p.put('TurnPilotValhallaTiles', tiles)
"

cd openpilot/system/manager && exec ./manager.py
