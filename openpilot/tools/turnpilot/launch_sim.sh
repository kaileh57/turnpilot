#!/usr/bin/env bash
# openpilot/tools/sim/launch_openpilot.sh for the TurnPilot bench, with two changes:
#   TURNPILOT_LOG=1 keeps loggerd, encoderd and logmessaged running, so a run leaves an rlog
#     in ~/.comma${OPENPILOT_PREFIX}/media/0/realdata
#   FINGERPRINT can be overridden, e.g. FINGERPRINT=TESLA_MODEL_3 for the fake Tesla
# Run it from the repo venv.

export PASSIVE="0"
export NOBOARD="1"
export SIMULATION="1"
export SKIP_FW_QUERY="1"
export FINGERPRINT="${FINGERPRINT:-HONDA_CIVIC_2022}"

export BLOCK="${BLOCK},camerad,micd,manage_athenad"
if [[ -z "$TURNPILOT_LOG" ]]; then
  export BLOCK="${BLOCK},loggerd,encoderd,logmessaged"
fi

# msgq needs the prefix directory to exist (macOS keeps msgq in /tmp)
if [[ -n "$OPENPILOT_PREFIX" ]]; then
  mkdir -p "/tmp/msgq_$OPENPILOT_PREFIX"
fi

python3 -c "from openpilot.selfdrive.test.helpers import set_params_enabled; set_params_enabled()"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd "$DIR/../../system/manager" && exec ./manager.py
