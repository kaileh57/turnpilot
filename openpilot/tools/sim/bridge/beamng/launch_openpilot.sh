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

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
OPENPILOT_ROOT="${OPENPILOT_ROOT:-$DIR/../../../../..}"
cd "$OPENPILOT_ROOT"
export PYTHONPATH="$OPENPILOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python3 -c "
from openpilot.common.params import Params
from openpilot.selfdrive.test.helpers import set_params_enabled
set_params_enabled()
Params().put_bool('AlphaLongitudinalEnabled', True)
"

cd openpilot/system/manager && exec ./manager.py
