#!/usr/bin/env bash
# Valhalla tiles for the sim's stock map, with lanes: the west_coast_usa road-graph export from the PC, plus turn lanes
# (openpilot/tools/sim/bridge/beamng/maplanes.py), built into tiles with build_valhalla_tiles.sh.
#
#   tools/turnpilot/build_sim_tiles.sh [export.osm [out_dir]]
#
# The export defaults to ~/pc_relay/incoming/osm_export/west_coast_usa.osm and out_dir to
# $TURNPILOT_DATA/valhalla/west_coast_usa_lanes (TURNPILOT_DATA defaults to ~/turnpilot_data). The tiles built from the export
# alone stay where they are, in west_coast_usa. Point TurnPilotValhallaTiles at out_dir, and set TurnPilotRouteSource to
# valhalla so that a destination outside the map is an error instead of a route on Indiana's real roads.
# $VALHALLA_PYTHON is the Python with pyvalhalla and openpilot's dependencies, the repo venv by default.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DATA="${TURNPILOT_DATA:-$HOME/turnpilot_data}"
EXPORT="${1:-$HOME/pc_relay/incoming/osm_export/west_coast_usa.osm}"
OUT="${2:-$DATA/valhalla/west_coast_usa_lanes}"
PY="${VALHALLA_PYTHON:-$ROOT/.venv/bin/python}"
LANES_OSM="$DATA/osm/$(basename "$OUT").osm"

[ -f "$EXPORT" ] || { echo "no export at $EXPORT" >&2; exit 2; }
[ -x "$PY" ] || { echo "no Python at $PY: set VALHALLA_PYTHON" >&2; exit 2; }

PYTHONPATH="$ROOT" "$PY" -m openpilot.tools.sim.bridge.beamng.maplanes "$EXPORT" "$LANES_OSM"
VALHALLA_PYTHON="$PY" "$HERE/build_valhalla_tiles.sh" "$LANES_OSM" "$OUT"
