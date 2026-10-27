"""Scenario hooks for BeamNG.tech: move the ego, spawn traffic and a pedestrian prop, force traffic signals.

Every function takes a BeamNGpy instance. In the bridge process pass world.bng under world.bng_lock; a separate script can
open its own client (BeamNG.tech accepts several). Positions are BeamNG world meters (x east, y north, z up), headings are
radians counterclockwise from east.
"""
import json
import math
import os

PED_SHAPE = "/levels/italy/art/shapes/buildings/human_scale_190.dae"  # the only human-shaped stock asset (1.90 m, static)
SIGNALS_JSON = os.path.join(os.path.dirname(__file__), "wcusa_signals.json")  # signalized junctions on west_coast_usa
GREEN, YELLOW, RED = 1, 2, 3  # lightsBasic controller state indices
LEVEL_TOD = 0.87  # west_coast_usa starts at 08:52 (BeamNG time of day: 0 is noon, 0.5 midnight)


def lua(bng, chunk: str, response: bool = False):
  return bng.control.queue_lua_command(chunk, response=response)


def heading_quat(heading: float) -> tuple:
  """Rotation quaternion that points an object whose front is its local -y (vehicles, the mannequin) at `heading`.
  BeamNG turns a positive z quaternion clockwise: the default spawn (0, 0, 0.383, 0.924) faces south-west."""
  yaw = -(heading + math.pi / 2)
  return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def ego_pose(bng, vid: str = 'ego'):
  """((x, y, z), heading rad) of a vehicle, read in the game engine."""
  r = lua(bng, f"local o = scenetree.findObject('{vid}') local p = o:getPosition() local d = o:getDirectionVector() " +
               "return string.format('%f,%f,%f,%f,%f', p.x, p.y, p.z, d.x, d.y)", response=True)
  x, y, z, dx, dy = (float(v) for v in str(r).split(','))
  return (x, y, z), math.atan2(dy, dx)


def ground_z(bng, x: float, y: float, z_hint: float = 1000.0) -> float:
  """Height of the static ground or road under (x, y), by a ray cast down from above z_hint."""
  r = lua(bng, f"local h = castRayStatic(vec3({x}, {y}, {z_hint}), vec3(0, 0, -1), 2000) return tostring({z_hint} - h)", response=True)
  return float(r)


def teleport_ego(bng, pos, heading: float | None = None, rot_quat=None, vid: str = 'ego'):
  rot = rot_quat if rot_quat is not None else (heading_quat(heading) if heading is not None else None)
  return bng.vehicles.teleport(vid, pos, rot)


def spawn_traffic(bng, amount: int = 8, parked: int = 0):
  """BeamNG's own traffic AI around the player vehicle."""
  bng.traffic.spawn(max_amount=amount, police_ratio=0.0, extra_amount=0, parked_amount=parked)


def remove_traffic(bng, keep=('ego',)):
  """Stop the traffic AI and delete every vehicle but ours (a stopped traffic vehicle stays where it is)."""
  bng.traffic.stop(stop=True)
  for vid in bng.vehicles.get_current_info(include_config=False):
    if vid not in keep:
      lua(bng, f"local o = scenetree.findObject('{vid}') if o then o:delete() end")


def _ped_pose(x, y, z, heading):
  qx, qy, qz, qw = heading_quat(heading)  # the mesh faces its -Y
  return f"{x}, {y}, {z}, {qx}, {qy}, {qz}, {qw}"


def spawn_pedestrian(bng, name: str, x: float, y: float, z: float | None = None, heading: float = 0.0, z_hint: float = 1000.0):
  """The Italy mannequin as a static prop (a TSStatic with collision, invisible to BeamNG's traffic AI).
  heading: the way the figure faces. z defaults to the first static surface below z_hint at (x, y): pass a z_hint a few
  meters above the road (the ego's z + 3), or the ray stops on a roof, a tree or a bridge."""
  z = ground_z(bng, x, y, z_hint) if z is None else z
  lua(bng, f"local o = scenetree.findObject('{name}') if o then o:delete() end " +
           f"o = createObject('TSStatic') o:setField('shapeName', 0, '{PED_SHAPE}') o:setField('collisionType', 0, 'Visible Mesh') " +
           f"o:registerObject('{name}') scenetree.MissionGroup:addObject(o) o:setPosRot({_ped_pose(x, y, z, heading)})")
  return z


def move_pedestrian(bng, name: str, x: float, y: float, z: float, heading: float):
  """Re-pose the prop; call every step to make it walk."""
  lua(bng, f"local o = scenetree.findObject('{name}') if o then o:setPosRot({_ped_pose(x, y, z, heading)}) end")


def remove_pedestrian(bng, name: str):
  lua(bng, f"local o = scenetree.findObject('{name}') if o then o:delete() end")


def load_signals(path: str = SIGNALS_JSON) -> list[dict]:
  with open(path) as f:
    return json.load(f)


def nearest_junction(x: float, y: float, signals: list[dict] | None = None) -> dict:
  signals = load_signals() if signals is None else signals
  return min(signals, key=lambda j: math.hypot(j['x'] - x, j['y'] - y))


def hold_signals(bng):
  """Stop the signal sequences so forced states stay (the sequence timer overwrites strict states otherwise)."""
  lua(bng, "core_trafficSignals.setActive(false)")


def release_signals(bng, instances=()):
  for name in instances:
    bng.traffic_signals.set_instance_strict_state(name, None)
  lua(bng, "core_trafficSignals.setActive(true)")


def force_signals(bng, instances, state: int = GREEN, hold: bool = True):
  """Force signal instances (e.g. a junction's 'instances' from wcusa_signals.json) to one state."""
  for name in instances:
    bng.traffic_signals.set_instance_strict_state(name, state)
  if hold:
    hold_signals(bng)


def all_green(bng, signals: list[dict] | None = None):
  """Every signalized junction on west_coast_usa green and held: nav scenarios don't test the model on lights."""
  signals = load_signals() if signals is None else signals
  force_signals(bng, [i for j in signals for i in j['instances']], GREEN)


def restore_game(bng, tod: float = LEVEL_TOD, signals: list[dict] | None = None):
  """Undo the bridge's changes to the shared game: every signal back on its sequence, HUD and particles on, the light
  at `tod` and running on. The loaded scenario and the ego stay."""
  signals = load_signals() if signals is None else signals
  release_signals(bng, [i for j in signals for i in j['instances']])
  bng.ui.show_hud()
  bng.settings.set_particles_enabled(True)
  bng.env.set_tod(tod=tod, play=False)


if __name__ == "__main__":
  import argparse
  from beamngpy import BeamNGpy
  ap = argparse.ArgumentParser(description="Restore the shared game after a bridge run that didn't (crash, or no --restore-on-exit).")
  ap.add_argument('--host', default='192.168.137.1')
  ap.add_argument('--port', type=int, default=25252)
  ap.add_argument('--tod', type=float, default=LEVEL_TOD)
  args = ap.parse_args()
  game = BeamNGpy(args.host, args.port).open(launch=False)
  try:
    restore_game(game, args.tod)
  finally:
    game.disconnect()
