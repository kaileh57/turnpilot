#!/usr/bin/env python3
"""Gate 3: does the car stop for a pedestrian standing in its lane?

Spawns the ego on the approach to a junction from the road-graph export, stands the Italy mannequin in the ego's lane
--ped-ahead m in front of it, facing the car, engages at --speed and logs at 10 Hz until the car stops, hits the
mannequin or times out. It records what each layer did: the driving model (modelV2.action.shouldStop and
desiredAcceleration, its lead), the detector (person boxes in turnpilotDetections, their confidence and range), the scene
(turnpilotScene pedestrians and inPath) and turnpilotd's plan (accelCap, speedCap, shouldStop, reason), plus the final gap.
The mannequin is removed after every run and on exit.

The model alone: TurnPilotEnabled=1 (so modeld runs the detector) with turnpilotd blocked, as for gate1.py; nothing then
publishes turnpilotPlan, so the planner runs as stock. With TurnPilot: turnpilotd running, no route.
  python -m openpilot.tools.sim.bridge.beamng.gate3 --runs 2 --speed 35 --out gate3.json
"""
import argparse
import json
import math
import time
from multiprocessing import Queue

import numpy as np

import openpilot.cereal.messaging as messaging
from openpilot.tools.sim.bridge.beamng.bridge import BeamNGBridge
from openpilot.tools.sim.bridge.beamng.cameras import MOUNT
from openpilot.tools.sim.bridge.beamng.gate1 import pick_turn
from openpilot.tools.sim.bridge.beamng.roadgraph import RoadGraph
from openpilot.tools.sim.bridge.beamng.run_bridge import config_from_args, parse_args
from openpilot.tools.sim.bridge.beamng.scenario import ego_pose, heading_quat, remove_pedestrian, spawn_pedestrian
from openpilot.tools.sim.bridge.common import control_cmd_gen

PED = "tp_gate3_ped"
SETTLE = 4.0
ENGAGE_TIMEOUT = 20.0
RUN_TIMEOUT = 60.0
STOPPED_SPEED = 0.2  # m/s
STOPPED_HOLD = 3.0   # s at standstill ends a run
PERSON = 0
DEFAULT_FRONT_M = 2.3  # ref node to the ETK 800's front bumper, if the bounding box isn't available


class Gate3:
  def __init__(self, turn, runs: int, approach: float, ped_ahead: float, speed_kph: float, out: str):
    self.turn, self.runs, self.approach, self.ped_ahead, self.speed_kph, self.out = turn, runs, approach, ped_ahead, speed_kph, out
    self.results, self.run, self.phase = [], -1, "next"
    self.sm = None

  def start_run(self, b):
    self.run += 1
    pos, heading = self.turn.spawn(self.approach)
    b.car.cancel()
    b.world.teleport(pos, heading_quat(heading))
    ped_xy, _ = self.turn.spawn(self.approach - self.ped_ahead)
    with b.world.bng_lock:
      z = spawn_pedestrian(b.world.bng, PED, ped_xy[0], ped_xy[1], heading=heading + math.pi, z_hint=pos[2] + 3.0)
    self.ped = np.array([ped_xy[0], ped_xy[1], z])
    self.heading = heading
    self.t0, self.phase, self.engaged_t, self.stopped_since = time.monotonic(), "settle", None, None
    self.rows, self.dets, self.disengaged, self.offsets = [], [], False, None
    print(f"gate3 run {self.run + 1}/{self.runs}: mannequin {self.ped_ahead:.0f} m ahead at {self.ped.round(1).tolist()}", flush=True)

  def measure_offsets(self, b):
    """Camera and front bumper, m ahead of the bridge's position (the IMU), from the ref node and bounding box."""
    s = b.simulator_state
    p = np.array([s.position.x, s.position.y])
    with b.world.bng_lock:
      (rx, ry, _), hd = ego_pose(b.world.bng)
    h = np.array([math.cos(hd), math.sin(hd)])
    cam = float(np.dot(np.array([rx, ry]) + -MOUNT[1] * h - p, h))
    front = float(np.dot(np.array([rx, ry]) - p, h)) + DEFAULT_FRONT_M
    try:
      with b.world.bng_lock:  # a game engine request, not the vehicle's link
        bbox = b.world.ego.get_bbox()
      corners = np.array([v[:2] for v in bbox.values()])
      front = float(np.max((corners - p) @ h))
    except Exception as e:
      print(f"gate3: no bounding box ({e!r}), front bumper {DEFAULT_FRONT_M} m ahead of the ref node", flush=True)
    self.offsets = (cam, front)

  def __call__(self, b) -> bool:
    try:
      return self.step(b)
    except Exception:
      with b.world.bng_lock:  # never leave the mannequin in the shared game
        remove_pedestrian(b.world.bng, PED)
      raise

  def step(self, b) -> bool:
    if self.sm is None:
      self.sm = messaging.SubMaster(['modelV2', 'turnpilotPlan', 'turnpilotScene', 'carState'])
      self.det_sock = messaging.sub_sock('turnpilotDetections', conflate=False)
    s = b.simulator_state
    now = time.monotonic()
    if self.phase == "next":
      if self.run + 1 >= self.runs:
        self.finish(b)
        return True
      self.start_run(b)
      return False

    b.car.sm.update(0)
    enabled = b.car.sm['selfdriveState'].enabled
    if self.phase == "settle":
      if now - self.t0 > SETTLE:
        self.measure_offsets(b)
        b.set_speed_kph, b.rolling_start_kph, b.auto_engage, self.phase = self.speed_kph, self.speed_kph * 0.8, True, "engage"
      return False

    pos = np.array([s.position.x, s.position.y])
    h = np.array([math.cos(self.heading), math.sin(self.heading)])
    along = float(np.dot(self.ped[:2] - pos, h))
    cam_gap, bumper_gap = along - self.offsets[0], along - self.offsets[1]
    for m in messaging.drain_sock(self.det_sock):
      td = m.turnpilotDetections
      people = [d for d in td.detections if d.cls == PERSON]
      best = max(people, key=lambda d: d.conf, default=None)
      self.dets.append((round(now - self.t0, 2), round(cam_gap, 1), str(td.camera), round(best.conf, 3) if best else 0.0,
                        [round(best.x0), round(best.y0), round(best.x1), round(best.y1)] if best else None))

    if self.phase == "engage":
      if enabled:
        self.phase, self.engaged_t = "drive", now
      elif now - self.t0 > SETTLE + ENGAGE_TIMEOUT:
        self.end_run(b, "no engagement", cam_gap, bumper_gap)
      return False

    if not enabled:
      self.disengaged = True
    if b.rk_frame % 10 == 0:
      self.sm.update(0)
      mv, plan, scene = self.sm['modelV2'], self.sm['turnpilotPlan'], self.sm['turnpilotScene']
      lead = mv.leadsV3[0] if len(mv.leadsV3) else None
      peds = [o for o in scene.objects if str(o.cls) == 'pedestrian']
      ped = min(peds, key=lambda o: o.x, default=None)
      self.rows.append({
        't': round(now - self.t0, 2), 'v': round(s.wheel_speed, 2), 'cam_gap': round(cam_gap, 2), 'bumper_gap': round(bumper_gap, 2),
        'enabled': bool(enabled), 'model_stop': bool(mv.action.shouldStop), 'model_accel': round(mv.action.desiredAcceleration, 2),
        'lead_prob': round(lead.prob, 2) if lead else None, 'lead_x': round(lead.x[0], 1) if lead and len(lead.x) else None,
        'plan_alive': bool(self.sm.alive['turnpilotPlan']), 'plan_accel_cap': round(plan.accelCap, 2), 'plan_speed_cap': round(plan.speedCap, 2),
        'plan_stop': bool(plan.shouldStop), 'plan_reason': plan.reason, 'scene_available': bool(scene.available),
        'scene_ped': None if ped is None else {'x': round(ped.x, 1), 'y': round(ped.y, 2), 'vx': round(ped.vx, 2), 'in_path': bool(ped.inPath),
                                               'id': ped.trackId, 'conf': round(ped.conf, 2)},
      })
    if s.wheel_speed < STOPPED_SPEED:
      self.stopped_since = self.stopped_since or now
    else:
      self.stopped_since = None
    if along < 0.5 or bumper_gap < 0.0:
      self.end_run(b, "reached the mannequin", cam_gap, bumper_gap)
    elif self.stopped_since is not None and now - self.stopped_since > STOPPED_HOLD:
      self.end_run(b, "stopped", cam_gap, bumper_gap)
    elif now - self.engaged_t > RUN_TIMEOUT:
      self.end_run(b, "timeout", cam_gap, bumper_gap)
    return False

  def end_run(self, b, reason, cam_gap, bumper_gap):
    b.car.cancel()
    b.auto_engage = False
    with b.world.bng_lock:
      remove_pedestrian(b.world.bng, PED)
    rows, dets = self.rows, self.dets
    seen = [d for d in dets if d[3] > 0]
    def first(key):
      r = next((r for r in rows if key(r)), None)
      return None if r is None else {'t': r['t'], 'cam_gap': r['cam_gap'], 'v': r['v']}
    res = {
      'run': self.run + 1, 'end': reason, 'disengaged': self.disengaged, 'speed_kph': self.speed_kph,
      'final_cam_gap_m': round(cam_gap, 2), 'final_bumper_gap_m': round(bumper_gap, 2),
      'max_v': max((r['v'] for r in rows), default=None), 'offsets_cam_front_m': [round(x, 2) for x in self.offsets or (0, 0)],
      'model_should_stop_first': first(lambda r: r['model_stop']),
      'model_brake_first': first(lambda r: r['model_accel'] < -1.0),
      'model_min_accel': min((r['model_accel'] for r in rows), default=None),
      'plan_stop_first': first(lambda r: r['plan_alive'] and (r['plan_speed_cap'] < r['v'] + 0.5 or r['plan_accel_cap'] < 10)),
      'scene_ped_first': first(lambda r: r['scene_ped'] is not None),
      'scene_ped_in_path_first': first(lambda r: r['scene_ped'] is not None and r['scene_ped']['in_path']),
      'detections': len(dets), 'person_detections': len(seen),
      'person_first_cam_gap': seen[0][1] if seen else None,
      'person_conf_by_range': conf_by_range(dets),
      'rows': rows, 'dets': dets,
    }
    self.results.append(res)
    print(f"gate3 run {res['run']}: {json.dumps({k: v for k, v in res.items() if k not in ('rows', 'dets')})}", flush=True)
    self.phase = "next"

  def finish(self, b):
    with open(self.out, 'w') as f:
      json.dump({'junction': self.turn.junction, 'ped_ahead': self.ped_ahead, 'speed_kph': self.speed_kph, 'runs': self.results}, f, indent=1)
    print(f"gate3: {len(self.results)} runs, results in {self.out}", flush=True)
    b.shutdown()


def conf_by_range(dets, step: float = 10.0) -> dict:
  """Per 10 m of camera gap: detector runs, runs that saw a person, and the median person conf."""
  out = {}
  for lo in np.arange(0.0, 150.0, step):
    d = [x for x in dets if lo <= x[1] < lo + step]
    if d:
      seen = [x[3] for x in d if x[3] > 0]
      out[f"{lo:.0f}-{lo + step:.0f}"] = {'runs': len(d), 'seen': len(seen), 'conf_median': round(float(np.median(seen)), 3) if seen else 0.0}
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--junction', default='group_80_81_82_83', help='its approach is the straight (signal group or road-graph node)')
  ap.add_argument('--turn', choices=('left', 'right'), default='right', help='picks the approach and the lane (right: the curb lane)')
  ap.add_argument('--approach', type=float, default=140.0, help='spawn this far before the junction, m')
  ap.add_argument('--ped-ahead', type=float, default=100.0, help='mannequin this far ahead of the spawn, m')
  ap.add_argument('--speed', type=float, default=35.0, help='cruise set speed, km/h')
  ap.add_argument('--runs', type=int, default=1)
  ap.add_argument('--out', default='gate3_results.json')
  args, rest = ap.parse_known_args()
  turn = pick_turn(RoadGraph(), args.junction, args.turn)
  gate = Gate3(turn, args.runs, args.approach, args.ped_ahead, args.speed, args.out)
  q: Queue = Queue()
  bridge = BeamNGBridge(config_from_args(parse_args(rest)), scenario=gate)
  p = bridge.run(q)
  try:
    p.join()
  except KeyboardInterrupt:
    q.put(control_cmd_gen("quit"))
    p.join()


if __name__ == "__main__":
  main()
