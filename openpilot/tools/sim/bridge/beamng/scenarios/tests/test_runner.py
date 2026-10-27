"""The runner's state machine, dry: a fake bridge, director, SubMaster and clock around a car that drives the scenario's
approach, turns at the junction and stops at the destination. Catches the sequencing the live game can't be used for."""
import json
import math
import os
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

import openpilot.cereal.messaging as messaging
from openpilot.cereal import custom
from openpilot.tools.sim.bridge.beamng.roadgraph import GRAPH_JSON
from openpilot.tools.sim.bridge.beamng.scenarios import catalog, run

pytestmark = pytest.mark.skipif(not os.path.exists(GRAPH_JSON), reason="needs piston's road-graph export")
DT = 0.01


class Clock:
  def __init__(self):
    self.t = 1000.0

  def monotonic(self):
    return self.t

  def wall(self):
    return 1.7e9 + self.t


class FakeDirector:
  def __init__(self, *args, **kwargs):
    self.calls, self.poses = [], {}

  def start(self):
    pass

  def is_alive(self):
    return True

  def submit(self, fn, *args):
    self.calls.append((getattr(fn, '__name__', str(fn)), args))
    f: Future = Future()
    f.set_result(fn(*args))
    return f

  def setup(self, actors):
    for a in actors:
      self.poses[a.name] = (a.path[0][0], a.path[0][1], a.heading)

  def walk(self, name):
    pass

  def release(self, name):
    pass

  def release_pacer(self, name, xy, v):
    pass

  def cleanup(self):
    self.poses.clear()

  def close(self):
    self.calls.append(('close', ()))


class World:
  """The car and navd: drives spawn -> junction -> destination at 8 m/s once engaged."""
  def __init__(self, plan, clock):
    self.plan, self.clock = plan, clock
    self.path = np.array([plan.spawn[:2], plan.junction, plan.dest[:2]])
    self.seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))]
    self.s, self.enabled, self.dest_set, self.route_t = 0.0, False, False, None
    self.msgs, self.updated, self.next_nav = {}, set(), 0.0

  def pos(self):
    i = min(int(np.searchsorted(self.seg, self.s, side='right')), len(self.path) - 1)
    a, b = self.path[i - 1], self.path[i]
    f = (self.s - self.seg[i - 1]) / (self.seg[i] - self.seg[i - 1])
    return a + min(f, 1.0) * (b - a), math.atan2(b[1] - a[1], b[0] - a[0])

  def web(self, url, token, method, body=None):
    assert token == 'tok' and url.endswith('/api/destination')
    self.dest_set = method == 'POST'
    self.route_t = self.clock.t + 1.0 if self.dest_set else None
    return {'ok': True}

  def step(self, b):
    v = 8.0 if self.enabled else 0.0
    self.s = min(self.s + v * DT, self.seg[-1])
    (x, y), h = self.pos()
    b.simulator_state.position = SimpleNamespace(x=float(x), y=float(y), z=0.0)
    b.simulator_state.bearing = 90.0 - math.degrees(h)
    b.simulator_state.wheel_speed = v
    if b.auto_engage and not self.enabled:
      self.enabled = True
      b.auto_engage = False
    if b.car.cancelled:
      self.enabled, b.car.cancelled = False, False
    self.updated = {'selfdriveState', 'carState', 'turnpilotPlan', 'modelV2'}
    ss = messaging.new_message('selfdriveState', valid=True)
    ss.selfdriveState.enabled = self.enabled
    self.msgs['selfdriveState'] = ss
    self.msgs['carState'] = messaging.new_message('carState', valid=True)
    tp = messaging.new_message('turnpilotPlan', valid=True)
    tp.turnpilotPlan.state = custom.TurnpilotPlan.State.following
    self.msgs['turnpilotPlan'] = tp
    self.msgs['modelV2'] = messaging.new_message('modelV2', valid=True)
    if self.clock.t >= self.next_nav:  # navd at 1 Hz
      self.next_nav = self.clock.t + 1.0
      routed = self.route_t is not None and self.clock.t >= self.route_t
      nav = messaging.new_message('navInstruction', valid=routed)
      if routed:
        to_turn, left = float(self.seg[1] - self.s), float(self.seg[-1] - self.s)
        n = nav.navInstruction
        n.maneuverType, n.maneuverModifier = ('turn', 'right') if to_turn > -10 else ('arrive', '')
        n.maneuverDistance = to_turn if to_turn > -10 else left
        n.distanceRemaining = left
      route = messaging.new_message('navRoute', valid=True)
      route.navRoute.coordinates = [{'latitude': 39.0, 'longitude': -86.0}, {'latitude': 39.1, 'longitude': -86.1}] if routed else []
      self.msgs['navInstruction'], self.msgs['navRoute'] = nav, route
      self.updated |= {'navInstruction', 'navRoute'}


class FakeSM:
  def __init__(self, world):
    self.world = world
    self.seen = dict.fromkeys(run.SERVICES, False)
    self.updated = dict.fromkeys(run.SERVICES, False)
    self.valid = dict.fromkeys(run.SERVICES, True)

  def update(self, timeout=0):
    for s in run.SERVICES:
      self.updated[s] = s in self.world.updated
      if self.updated[s]:
        self.seen[s] = True
        self.valid[s] = self.world.msgs[s].valid
    self.world.updated = set()

  def __getitem__(self, s):
    m = self.world.msgs.get(s) or messaging.new_message(s)
    return getattr(m.as_reader(), s)


def bridge():
  car = SimpleNamespace(cancelled=False)
  car.cancel = lambda: setattr(car, 'cancelled', True)
  b = SimpleNamespace(simulator_state=SimpleNamespace(), car=car, blinker=(False, False), auto_engage=False, rk_frame=0,
                      set_speed_kph=0.0, rolling_start_kph=0.0, cfg=SimpleNamespace(capture=''), teleports=[], done=False)
  b.world = SimpleNamespace(capture=None, teleport=lambda pos, rot: b.teleports.append(pos))
  b.shutdown = lambda: setattr(b, 'done', True)
  return b


# the fake world plays no turnpilotd and no detector, so each scenario fails on what only they would do
CASES = [
  ('ped_crosswalk_right', 2, 'walk', "never held to a stop for the pedestrian in the crosswalk"),
  ('unprotected_left', 1, 'release', "never waited for oncoming traffic"),
  ('occupied_lane', 1, 'release_pacer', "the planner never saw the target lane as occupied"),
]


@pytest.mark.parametrize("name,runs,trigger,failure", CASES)
def test_runs_end_to_end(tmp_path, monkeypatch, name, runs, trigger, failure):
  clock = Clock()
  plan = catalog.build(name)
  world = World(plan, clock)
  monkeypatch.setattr(run, 'time', SimpleNamespace(monotonic=clock.monotonic, strftime=lambda *a: 'now'))
  monkeypatch.setattr(run, 'wall', clock.wall)
  monkeypatch.setattr(run, 'Director', FakeDirector)
  monkeypatch.setattr(run, 'web_call', world.web)
  monkeypatch.setattr(run.messaging, 'SubMaster', lambda services: FakeSM(world))
  runner = run.Runner(plan, runs, str(tmp_path), 'http://127.0.0.1:8088', 'tok', 'host', 0)
  b = bridge()
  b.world.teleport = lambda pos, rot: (b.teleports.append(pos), setattr(world, 's', 0.0))
  b.simulator_state.position, b.simulator_state.bearing, b.simulator_state.wheel_speed = SimpleNamespace(x=0, y=0, z=0), 0.0, 0.0
  for _ in range(int(200 / DT)):
    world.step(b)
    if runner(b):
      break
    b.rk_frame += 1
    clock.t += DT
  assert b.done and len(b.teleports) == runs
  out = json.load(open(tmp_path / 'result.json'))
  assert [r['outcome'] for r in out['runs']] == ['arrived'] * runs
  names = [c[0] for c in runner.director.calls]
  actors = sum(a.trigger == {'walk': 'walk', 'release': 'oncoming', 'release_pacer': 'pace'}[trigger] for a in plan.actors)
  assert names.count('setup') == runs and names.count(trigger) == runs * actors and names.count('cleanup') >= runs and names[-1] == 'close'
  assert [c[1][2] for c in runner.director.calls if c[0] == 'web'] == ['DELETE', 'POST'] * runs
  for r in out['runs']:
    assert r['maneuvers_attempted'] == 1 and not r['disengagements']
    assert r['failures'][0] == failure
    gap = 'ped_gap' if name.startswith('ped') else 'car_gap'
    assert any(row[gap] is not None for row in r['rows'])
