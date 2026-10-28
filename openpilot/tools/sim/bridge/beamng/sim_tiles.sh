#!/usr/bin/env bash
# sim_tiles.sh: prints the tiles navd routes on for a BeamNG drive (restart_stack.sh). TURNPILOT_TILES when set (e.g. the
# west_coast_usa_lanes tiles for the turn-lane demos), said on stderr when they carry no controls (no controls_built
# marker: navd reports them unsupported and TurnPilot shows "Map stops unavailable"). Otherwise the signals tiles, the
# level's stop signs and signals; when those aren't built it fails, rather than route on tiles with no stops.
set -u
if [ -n "${TURNPILOT_TILES:-}" ]; then
  [ -f "$TURNPILOT_TILES/controls_built" ] || \
    echo "warning: $TURNPILOT_TILES has no controls (no controls_built): no stop control, Map stops unavailable" >&2
  echo "$TURNPILOT_TILES"
  exit 0
fi
SIGNALS=$HOME/turnpilot_data/valhalla/west_coast_usa_signals
if [ ! -f "$SIGNALS/controls_built" ]; then
  echo "error: $SIGNALS isn't built (no controls_built): build it with tools/turnpilot/build_sim_tiles.sh, or set TURNPILOT_TILES" >&2
  exit 1
fi
echo "$SIGNALS"
