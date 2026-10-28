#!/usr/bin/env python3
"""Re-dump the level's signal data that mapsignals.py tags the sim tiles from: core_trafficSignals' mapNodeSignals (the
map link each signal and stop instance controls, with its stop line) into wcusa_map_node_signals.json, and the stop-sign
props into wcusa_stop_props.json. The committed files came from a one-off Lua call on 2026-10-27 (the provenance gap in
the json's header); this is that call written down. Unverified against the game since (no game access when it was written):
check the instance count (334 on west_coast_usa) before rebuilding the tiles from a new dump.

  python -m openpilot.tools.sim.bridge.beamng.probe_signals [--host H --port P]
then tools/turnpilot/build_sim_tiles.sh.
"""
import argparse
import json
import os
import time

HERE = os.path.dirname(__file__)
LUA_MAP_NODE_SIGNALS = "return jsonEncode(core_trafficSignals.getMapNodeSignals and core_trafficSignals.getMapNodeSignals() or {})"
LUA_STOP_PROPS = " ".join([
  "local out = {} local objs = getObjectsByClass and getObjectsByClass('TSStatic') or {}",
  "for _, o in ipairs(objs) do local s = o:getField('shapeName', 0) or ''",
  "if string.find(string.lower(s), 'stop') then local p = o:getPosition() local r = o:getRotation()",
  "table.insert(out, {shape = s, pos = {p.x, p.y, p.z}, rot = {r.x, r.y, r.z, r.w}}) end end",
  "return jsonEncode(out)",
])


def main():
  from beamngpy import BeamNGpy
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument('--host', default='192.168.137.1')
  ap.add_argument('--port', type=int, default=25252)
  args = ap.parse_args()
  bng = BeamNGpy(args.host, args.port).open(launch=False)
  try:
    mns = json.loads(str(bng.control.queue_lua_command(LUA_MAP_NODE_SIGNALS, response=True)))
    props = json.loads(str(bng.control.queue_lua_command(LUA_STOP_PROPS, response=True)))
  finally:
    bng.disconnect()
  instances = {s['instance'] for inner in mns.values() for lst in inner.values() for s in lst}
  print(f"{len(instances)} signal and stop instances on {sum(len(v) for v in mns.values())} links; {len(props)} stop props")
  stamp = time.strftime('%Y-%m-%d %H:%M')
  with open(os.path.join(HERE, 'wcusa_map_node_signals.json'), 'w') as f:
    json.dump({"_provenance": f"core_trafficSignals mapNodeSignals, probe_signals.py {stamp}", "mapNodeSignals": mns}, f, indent=1,
              sort_keys=True)
  with open(os.path.join(HERE, 'wcusa_stop_props.json'), 'w') as f:
    json.dump({"_provenance": f"TSStatic props with 'stop' in the shape, probe_signals.py {stamp}", "props": props}, f, indent=1)


if __name__ == "__main__":
  main()
