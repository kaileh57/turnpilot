#!/usr/bin/env python3
"""The sim's traffic light oracle: the game's own light states ahead of the ego, published as turnpilotLights (simOracle).

SIM ONLY. turnpilotd takes these lights only with SIMULATION=1 and TurnPilotSimLightOracle set (launch_openpilot.sh sets
the param with TURNPILOT_LIGHT_ORACLE=1); anything else that publishes them is rejected out loud and the lights are
unknown, so on the car every mapped signal is held for the driver's go (map_controls.py). The scenario drivers also read
it as the ground truth for their go at a signal (driver.GasTaps), with or without turnpilotd taking it.

It runs on its own BeamNGpy connection (never the bridge's world lock), at RATE_HZ: the ego's pose, then the signal
instances whose stop line (wcusa_map_node_signals.json) is 0 to MAX_AHEAD_M ahead and within MAX_LATERAL_M of the ego's
axis, facing it (the instance's direction, from the game, dotted with the ego's heading, over FACING_DOT), nearest first,
at most MAX_LIGHTS of them, each read with get_instance_state. A failed poll publishes an empty list: unknown, a hold. The
lights go out --light-latency s late (default 0.3), as a camera's would.

  python -m openpilot.tools.sim.bridge.beamng.light_oracle [--host H --port P --light-latency 0.3]
Unverified against the game since it was written (2026-10-27, no game access): run it once and check the published
states against the HUD before trusting a batch on it.
"""
import argparse
import collections
import json
import math
import os
import time

import openpilot.cereal.messaging as messaging

SIGNALS = os.path.join(os.path.dirname(__file__), "wcusa_map_node_signals.json")
RATE_HZ = 4.
MAX_AHEAD_M = 200.
MAX_LATERAL_M = 25.
FACING_DOT = 0.7
MAX_LIGHTS = 4


def light_state(raw) -> str:
  """A game signal state ('greenTrafficLight', 'red', 2, ...) as TurnpilotLights.State; anything else is unknown."""
  s = str(raw).lower()
  for name in ('red', 'yellow', 'green'):
    if name in s:
      return name
  if 'amber' in s:
    return 'yellow'
  return 'unknown'


def stop_lines(path: str = SIGNALS) -> dict[str, tuple[float, float]]:
  """{traffic light instance: its stop line (x, y)} from the mapNodeSignals dump (the first link it controls)."""
  with open(path) as f:
    mns = json.load(f)
  mns = mns.get("mapNodeSignals", mns)
  out = {}
  for inner in mns.values():
    for lst in inner.values():
      for s in lst:
        if s["instance"].startswith("trafficLight") and s["instance"] not in out:
          out[s["instance"]] = (s["pos"]["x"], s["pos"]["y"])
  return out


def lights_ahead(ego_xy, ego_fwd, lines: dict, dirs: dict) -> list[tuple[str, float]]:
  """[(instance, m ahead to its stop line)] facing the ego, nearest first, at most MAX_LIGHTS."""
  fx, fy = ego_fwd
  n = math.hypot(fx, fy) or 1.
  fx, fy = fx / n, fy / n
  out = []
  for name, (x, y) in lines.items():
    dx, dy = x - ego_xy[0], y - ego_xy[1]
    ahead, lateral = dx * fx + dy * fy, -dx * fy + dy * fx
    d = dirs.get(name)
    if d is None or not (0. <= ahead <= MAX_AHEAD_M) or abs(lateral) > MAX_LATERAL_M:
      continue
    dn = math.hypot(d[0], d[1]) or 1.
    if (d[0] * fx + d[1] * fy) / dn > FACING_DOT:
      out.append((name, ahead))
  return sorted(out, key=lambda e: e[1])[:MAX_LIGHTS]


def make_msg(lights: list[tuple[str, float, str]]):
  """turnpilotLights from [(instance, distance, state)]."""
  msg = messaging.new_message('turnpilotLights')
  msg.turnpilotLights.source = 'simOracle'
  msg.turnpilotLights.lights = [{'state': state, 'distance': float(dist), 'instance': name} for name, dist, state in lights]
  return msg


def main():
  from beamngpy import BeamNGpy
  from openpilot.tools.sim.bridge.beamng.scenario import ego_pose

  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument('--host', default='192.168.137.1')
  ap.add_argument('--port', type=int, default=25252)
  ap.add_argument('--light-latency', type=float, default=0.3)
  args = ap.parse_args()

  pm = messaging.PubMaster(['turnpilotLights'])
  lines = stop_lines()
  bng = BeamNGpy(args.host, args.port).open(launch=False)
  try:
    dirs = {}
    for name in lines:
      st = bng.traffic_signals.get_instance_state(name)
      d = st.get('dir') if isinstance(st, dict) else None
      if d is not None:
        dirs[name] = (d[0], d[1]) if isinstance(d, (list, tuple)) else (d['x'], d['y'])
    print(f"light_oracle: {len(dirs)} of {len(lines)} lights with a direction", flush=True)
    queue: collections.deque = collections.deque()
    while True:
      t = time.monotonic()
      try:
        pos, heading = ego_pose(bng)
        ahead = lights_ahead(pos[:2], (math.cos(heading), math.sin(heading)), lines, dirs)
        lights = [(name, dist, light_state(bng.traffic_signals.get_instance_state(name).get('state'))) for name, dist in ahead]
      except Exception as e:  # unknown: an empty list, which holds every signal
        print(f"light_oracle: poll failed: {e!r}", flush=True)
        lights = []
      queue.append((t + args.light_latency, lights))
      while queue and queue[0][0] <= time.monotonic():
        pm.send('turnpilotLights', make_msg(queue.popleft()[1]))
      time.sleep(max(1. / RATE_HZ - (time.monotonic() - t), 0.))
  finally:
    bng.disconnect()


if __name__ == "__main__":
  main()
