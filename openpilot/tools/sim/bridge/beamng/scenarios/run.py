#!/usr/bin/env python3
"""Run a scenario on the BeamNG bridge and score it.

Each run: teleport the car to the scenario's start, place its mannequin or cars, send the destination to turnpilot_web
the way the phone does, engage, then watch turnpilotPlan, selfdriveState, carState, navInstruction and modelV2 until
the car arrives, disengages, stops (where the scenario ends on a stop) or times out. A result JSON per invocation holds
every run: outcome, maneuvers taken of attempted, disengagements with cause, holds with reason and duration, the
closest the car came to the mannequin or another car, the rlog segments, and the 20 Hz rows it was scored from.
Everything a run spawns is removed after it, and again from here after the bridge exits (a sweep by name).

Start openpilot first (navd, turnpilotd and turnpilot_web need NavigationEnabled and TurnPilotEnabled, which this sets):
  OPENPILOT_PREFIX=tpsim openpilot/tools/sim/bridge/beamng/launch_openpilot.sh
  OPENPILOT_PREFIX=tpsim python -m openpilot.tools.sim.bridge.beamng.scenarios.run ped_crosswalk_right --runs 3 --record ~/turnpilot-artifacts/demo
Bridge options (--host, --capture, --signals, ...) pass through; --record sets --capture and records the UI as well.
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
from multiprocessing import Queue
from pathlib import Path

import openpilot.cereal.messaging as messaging
from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params
from openpilot.selfdrive.turnpilot.web.pairing import PORT
from openpilot.selfdrive.turnpilot.web.store import ensure_token
from openpilot.tools.sim.bridge.beamng import frame
from openpilot.tools.sim.bridge.beamng.scenario import heading_quat
from openpilot.tools.sim.bridge.beamng.scenarios import score
from openpilot.tools.sim.bridge.beamng.scenarios.catalog import DATA_JSON, SCENARIOS, Plan, build, hold_distance
from openpilot.tools.sim.bridge.beamng.scenarios.director import Director, sweep
from openpilot.tools.sim.bridge.beamng.scenarios.timing import oncoming_should_release, walker_should_start

SETTLE_S = 4.         # after the teleport: the car is back in D and the GPS has caught up
CLEAR_TIMEOUT_S = 5.  # wait this long for navd to drop the old route
ROUTE_TIMEOUT_S = 30.
ENGAGE_TIMEOUT_S = 30.
ROLL_FRACTION = 0.8   # the driver rolls up to this much of the set speed before engaging, as gate1 does
SERVICES = ['turnpilotPlan', 'selfdriveState', 'carState', 'navInstruction', 'navRoute', 'modelV2']
OUT_ROOT = os.path.expanduser("~/turnpilot-artifacts/scenarios")


def wall() -> float:
  return time.time()  # noqa: TID251  the recordings are aligned on wall-clock time


def web_call(url: str, token: str, method: str, body: dict | None = None) -> dict:
  data = None if body is None else json.dumps(body).encode()
  req = urllib.request.Request(url, data=data, method=method, headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'})
  with urllib.request.urlopen(req, timeout=5) as r:
    return json.loads(r.read() or b'{}')


def rlog_segments(since: float) -> list[str]:
  root = Path(Paths.log_root())
  if not root.is_dir():
    return []
  return sorted(str(d) for d in root.iterdir() if d.is_dir() and d.stat().st_mtime >= since)


def write_sidecar(video: str, start: float, fps: float, source: str):
  """`<video>.start.json`: the wall-clock time of the video's first frame, which demo_video aligns on."""
  with open(f"{video}.start.json", 'w') as f:
    json.dump({'start': start, 'fps': fps, 'source': source}, f)


class Runner:
  """The scenario, run in the bridge process at 100 Hz (BeamNGBridge scenario hook). Built in the parent and pickled into
  the bridge process, so everything live (SubMaster, the director thread and its client) starts on the first call."""

  def __init__(self, plan: Plan, runs: int, out_dir: str, web_url: str, token: str, host: str, port: int):
    self.plan, self.runs, self.out_dir, self.web_url, self.token = plan, runs, out_dir, web_url, token
    self.host, self.port = host, port
    self.sm = self.director = None
    self.results: list[dict] = []
    self.run_idx = -1
    self.phase = 'next'
    self.sidecar_done = False
    self.started_wall = wall()

  def __call__(self, b) -> bool:
    try:
      if self.sm is None:
        self.sm = messaging.SubMaster(SERVICES)
        self.director = Director(self.host, self.port)
        self.director.start()
      self.sidecar(b)
      return self.step(b)
    except Exception:
      traceback.print_exc()
      if self.phase not in ('next', 'done'):
        self.end_run(b, 'error', traceback.format_exc(limit=3))
      self.finish(b)
      return True

  def sidecar(self, b):
    cap = getattr(b.world, 'capture', None)
    if cap is not None and not self.sidecar_done and cap.frames >= 10:  # the capture writes one frame a tick from its first
      from openpilot.tools.sim.bridge.beamng.capture import FPS
      write_sidecar(b.cfg.capture, wall() - cap.frames / FPS, FPS, 'beamng chase')
      self.sidecar_done = True

  def step(self, b) -> bool:
    now = time.monotonic()
    self.sm.update(0)
    if self.sm.updated['navRoute']:
      c = self.sm['navRoute'].coordinates
      self.route_pts, self.route_id = len(c), score.route_id(c)
    if self.sm.updated['navInstruction'] and not self.sm.valid['navInstruction']:
      self.nav_cleared = True
    plan = self.plan
    if not self.director.is_alive() and self.phase not in ('next', 'done'):
      self.end_run(b, 'error', "the scenario director stopped (is the game reachable?)")
      self.finish(b)
      return True
    if self.phase == 'next':
      if self.run_idx + 1 >= self.runs:
        self.finish(b)
        return True
      self.start_run(b)
    elif self.phase == 'settle':
      if now - self.t0 > SETTLE_S and self.setup.done():
        if self.setup.exception() is not None:
          self.end_run(b, 'error', f"setup: {self.setup.exception()!r}")
        elif plan.dest is None:
          self.engage(b)
        else:
          self.web = self.director.submit(web_call, f"{self.web_url}/api/destination", self.token, 'DELETE')
          self.nav_cleared, self.phase, self.t_phase = False, 'clear', now
    elif self.phase == 'clear':
      # navd drops the old route within its 1 Hz tick; posting before it does could leave navd on the old one
      if self.web.done() and (self.nav_cleared or now - self.t_phase > CLEAR_TIMEOUT_S):
        lat, lon, _ = frame.to_latlon(*plan.dest)
        self.web = self.director.submit(web_call, f"{self.web_url}/api/destination", self.token, 'POST',
                                        {'latitude': lat, 'longitude': lon, 'name': f"scenario {plan.name}"})
        self.route_pts, self.phase, self.t_phase = None, 'route', now
    elif self.phase == 'route':
      if self.web.done() and self.web.exception() is not None:
        self.end_run(b, 'no destination', f"turnpilot_web at {self.web_url}: {self.web.exception()!r}")
      elif self.web.done() and self.route_pts and self.sm.valid['navInstruction'] and self.sm['navInstruction'].distanceRemaining > 0:
        self.engage(b)
      elif now - self.t_phase > ROUTE_TIMEOUT_S:
        self.end_run(b, 'no route', "navd sent no route (is navd running, with a route source for the sim map?)")
    elif self.phase == 'engage':
      if self.sm['selfdriveState'].enabled:
        self.phase, self.t_engaged, self.t_engaged_wall = 'run', now, wall()
      elif now - self.t_phase > ENGAGE_TIMEOUT_S:
        self.end_run(b, 'no engagement', "openpilot never engaged")
    elif self.phase == 'run' and b.rk_frame % 5 == 0:  # 20 Hz
      r = self.row(b, now)
      self.rows.append(r)
      self.triggers(b, r)
      self.driver(b)
      end = self.end_check.update(r)
      if end is not None:
        self.end_run(b, end)
    return False

  def start_run(self, b):
    self.run_idx += 1
    plan = self.plan
    print(f"scenario {plan.name} run {self.run_idx + 1}/{self.runs}", flush=True)
    b.auto_engage = False
    b.car.cancel()
    b.blinker = (False, False)
    b.world.teleport(plan.spawn, heading_quat(plan.heading))
    self.setup = self.director.submit(self.director.setup, plan.actors)
    self.t0, self.t0_wall, self.phase = time.monotonic(), wall(), 'settle'
    self.rows: list[dict] = []
    self.fired: set[str] = set()
    self.route_pts, self.route_id, self.nav_cleared = None, 0, False
    self.t_engaged = self.t_engaged_wall = None
    self.end_check = score.EndCheck(plan.timeout_s, plan.end_on_stop_s)

  def engage(self, b):
    b.set_speed_kph, b.rolling_start_kph, b.auto_engage = self.plan.speed_kph, self.plan.speed_kph * ROLL_FRACTION, True
    self.phase, self.t_phase = 'engage', time.monotonic()

  def row(self, b, now) -> dict:
    s, sm = b.simulator_state, self.sm
    x, y = s.position.x, s.position.y
    h = math.radians(90.0 - s.bearing)  # compass to CCW from east
    r = {
      't': round(now - self.t_engaged, 3), 'wall': round(wall(), 3), 'x': round(x, 2), 'y': round(y, 2), 'heading': round(h, 3),
      'v': round(s.wheel_speed, 2), 'route': self.route_id,
      'dest_dist': None if self.plan.dest is None else round(math.hypot(self.plan.dest[0] - x, self.plan.dest[1] - y), 1),
      'blinker': 'left' if b.blinker[0] else 'right' if b.blinker[1] else '',
      'ped_gap': None, 'car_gap': None, 'pacer_dx': None,
      **score.msg_fields(sm['selfdriveState'], sm['carState'], sm['turnpilotPlan'] if sm.seen['turnpilotPlan'] else None,
                         sm['navInstruction'] if sm.seen['navInstruction'] and sm.valid['navInstruction'] else None,
                         sm['modelV2']),
    }
    for a in self.plan.actors:
      pose = self.director.poses.get(a.name)
      if pose is None:
        continue
      d = math.hypot(pose[0] - x, pose[1] - y)
      key = 'ped_gap' if a.kind == 'walker' else 'car_gap'
      r[key] = round(d if r[key] is None else min(r[key], d), 2)
      if a.trigger == 'pace':
        r['pacer_dx'] = round((pose[0] - x) * math.cos(h) + (pose[1] - y) * math.sin(h), 2)
    return r

  def triggers(self, b, r):
    """Set the actors going when the car gets to them: the walker steps out as the car arrives, the oncoming cars go
    to reach the junction just after the car reaches its hold line, the pacer goes once the car is engaged."""
    plan, d = self.plan, self.director
    for a in plan.actors:
      if a.name in self.fired or a.trigger == 'setup':
        continue
      go = False
      if a.trigger == 'walk':
        go = walker_should_start(hold_distance(plan, r['x'], r['y']), r['v'], a.params['step_in_m'], a.params['lead_s'], a.speed)
        job = (d.walk, a.name)
      elif a.trigger == 'oncoming':
        go = oncoming_should_release(hold_distance(plan, r['x'], r['y']), r['v'], a.params['start_m'], a.speed, a.accel,
                                     a.params['arrive_after_s'])
        job = (d.release, a.name)
      elif a.trigger == 'pace':
        go, job = True, (d.release_pacer, a.name, (r['x'], r['y']), r['v'])
      if go:
        self.fired.add(a.name)
        d.submit(*job)
        print(f"scenario: {a.name} goes at t={r['t']:.1f} s, {hold_distance(plan, r['x'], r['y']):.0f} m from the hold line", flush=True)

  def driver(self, b):
    """The driver taps the stalk when TurnPilot asks for a signal (PLAN: Tesla can't signal for openpilot), and lets go
    when it stops asking, which it does as soon as the lane change starts."""
    p = str(self.sm['turnpilotPlan'].signalPrompt)
    b.blinker = (p == 'left', p == 'right')

  def end_run(self, b, outcome: str, detail: str = ''):
    b.auto_engage = False
    b.car.cancel()
    b.blinker = (False, False)
    end_wall = wall()
    res = score.summarize(self.plan.name, self.rows, outcome)
    res.update(run=self.run_idx + 1, detail=detail, t_start_wall=self.t0_wall, t_engaged_wall=self.t_engaged_wall, t_end_wall=end_wall,
               duration_s=round(end_wall - (self.t_engaged_wall or self.t0_wall), 1), rlog_segments=rlog_segments(self.t0_wall - 60.),
               rows=self.rows)
    self.results.append(res)
    print(score.report(f"scenario {self.plan.name} run {res['run']}", res) + (f"\n  {detail}" if detail else ''), flush=True)
    self.director.submit(self.director.cleanup)
    self.save()
    self.phase = 'next'

  def save(self):
    os.makedirs(self.out_dir, exist_ok=True)
    plan = self.plan
    out = {'scenario': plan.name, 'started_wall': self.started_wall, 'spawn': plan.spawn, 'heading': plan.heading, 'dest': plan.dest,
           'junction': plan.junction, 'speed_kph': plan.speed_kph, 'notes': plan.notes,
           'actors': [{'name': a.name, 'kind': a.kind, 'trigger': a.trigger, 'path': a.path, 'params': a.params} for a in plan.actors],
           'passed': sum(r['passed'] for r in self.results), 'runs': self.results}
    with open(os.path.join(self.out_dir, 'result.json'), 'w') as f:
      json.dump(out, f, indent=1)

  def finish(self, b):
    self.phase = 'done'
    if self.director is not None:
      self.director.close()
    self.save()
    passed = sum(r['passed'] for r in self.results)
    print(f"scenario {self.plan.name}: {passed}/{len(self.results)} passed, results in {os.path.join(self.out_dir, 'result.json')}", flush=True)
    b.shutdown()


def sweep_game(host: str, port: int):
  from beamngpy import BeamNGpy
  bng = BeamNGpy(host, port).open(launch=False)
  try:
    n = sweep(bng)
    if n:
      print(f"cleanup: removed {n} leftover scenario actor(s)", flush=True)
  finally:
    bng.disconnect()


def start_ui_recording(path: str):
  env = dict(os.environ, RECORD_OUTPUT=path)
  return subprocess.Popen([sys.executable, '-m', 'openpilot.tools.turnpilot.record_ui', path], env=env)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('name', nargs='?', choices=sorted(SCENARIOS), help='scenario')
  ap.add_argument('--runs', type=int, default=1)
  ap.add_argument('--record', default='', help='record the chase view and the UI into DIR/<scenario>-<time>/')
  ap.add_argument('--no-ui', action='store_true', help='with --record, record the chase view only')
  ap.add_argument('--out', default=OUT_ROOT, help='results go to OUT/<scenario>-<time>/result.json (or the --record directory)')
  ap.add_argument('--data', default=DATA_JSON, help='junction list for lane_change and occupied_lane')
  ap.add_argument('--entry', default='', help='entry of --data by name (default: the first)')
  ap.add_argument('--web', default=f'http://127.0.0.1:{PORT}', help='turnpilot_web, where the destination goes')
  ap.add_argument('--cleanup', action='store_true', help='only remove leftover scenario actors from the game')
  args, rest = ap.parse_known_args()

  from openpilot.tools.sim.bridge.beamng.run_bridge import config_from_args, parse_args
  bargs = parse_args(rest)
  if args.cleanup:
    sweep_game(bargs.host, bargs.port)
    return
  if args.name is None:
    ap.error("name a scenario")

  plan = build(args.name, data_path=args.data, entry=args.entry)
  run_dir = os.path.join(os.path.expanduser(args.record or args.out), f"{args.name}-{time.strftime('%Y%m%d-%H%M%S')}")
  os.makedirs(run_dir, exist_ok=True)

  from openpilot.tools.sim.bridge.beamng.bridge import BeamNGBridge
  from openpilot.tools.sim.bridge.common import control_cmd_gen
  cfg = config_from_args(bargs)
  cfg.spawn_pos, cfg.spawn_rot = plan.spawn, heading_quat(plan.heading)
  if args.record:
    cfg.capture = os.path.join(run_dir, 'chase.mp4')

  params = Params()
  token = ensure_token(params)
  for key in ('NavigationEnabled', 'TurnPilotEnabled'):
    if not params.get_bool(key):
      params.put_bool(key, True)
      print(f"set {key}: navd, turnpilotd and turnpilot_web run on it", flush=True)

  sweep_game(bargs.host, bargs.port)
  ui = start_ui_recording(os.path.join(run_dir, 'ui.mp4')) if args.record and not args.no_ui else None
  q: Queue = Queue()
  bridge = BeamNGBridge(cfg, scenario=Runner(plan, args.runs, run_dir, args.web.rstrip('/'), token, bargs.host, bargs.port))
  p = bridge.run(q)
  try:
    p.join()
  except KeyboardInterrupt:
    q.put(control_cmd_gen("quit"))
    p.join()
  finally:
    if ui is not None:
      ui.send_signal(signal.SIGINT)  # SIGINT lets ffmpeg finish the mp4
      try:
        ui.wait(timeout=30)
      except subprocess.TimeoutExpired:
        ui.kill()
    sweep_game(bargs.host, bargs.port)  # the bridge process may have died before its own cleanup
  print(f"results: {os.path.join(run_dir, 'result.json')}")


if __name__ == "__main__":
  main()
