"""turnpilotd closed loop: synthetic navd, model and scene around a point-mass car that follows the plan's limits."""
from opendbc.car.structs import car
from openpilot.cereal import custom, log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.hooks import CLEAR_DESIRE, plan_accel
from openpilot.selfdrive.turnpilot.lane_planner import lane_change_window
from openpilot.selfdrive.turnpilot.scene_policy import HOLD_LINE_M, NO_ACCEL_CAP, ONCOMING_TTA_S, PED_STOP_GAP_M
from openpilot.selfdrive.turnpilot.tests.synthetic import FakeNavd, car_control, car_state, model_v2, nav_instruction, scene
from openpilot.selfdrive.turnpilot.desire_timing import TURN_DESIRE_LEAD_M
from openpilot.selfdrive.turnpilot.turn_speed import NO_SPEED_CAP, V_TURN_MIN
from openpilot.selfdrive.turnpilot.turnpilotd import NO_LONG_REASON, TurnPilot, long_control_for

State = custom.TurnpilotPlan.State
LaneChange = custom.TurnpilotPlan.LaneChange
Desire = log.Desire
THREE_LANES_LEFT_TURN = ((("left",), True), (("straight",), False), (("straight", "right"), False))


class FakeSM:
  def __init__(self):
    self.data = {'carState': car_state(0.), 'carControl': car_control(), 'navInstruction': nav_instruction(0., valid=False),
                 'modelV2': model_v2()}
    self.updated = dict.fromkeys(self.data, False)
    self.valid = dict.fromkeys(self.data, True)
    self.alive = dict.fromkeys(self.data, True)

  def __getitem__(self, s):
    return self.data[s]

  def put(self, s, msg, valid=True):
    self.data[s], self.updated[s], self.valid[s] = msg, True, valid


class Sim:
  def __init__(self, maneuvers, cruise=13., nav_lanes=None, model_lanes=(0, 0), shadow=False, driver_signals=False, long_control=True):
    self.navd, self.tp, self.sm = FakeNavd(maneuvers, nav_lanes), TurnPilot(), FakeSM()
    self.s, self.v, self.cruise, self.t = 0., cruise, cruise, 0.
    self.model_lanes, self.shadow, self.driver_signals, self.long_control = model_lanes, shadow, driver_signals, long_control
    self.trace = []

  def step(self, objects=(), available=True, steer=0., blinker=False, lat_active=True):
    self.sm.updated = dict.fromkeys(self.sm.data, False)
    instr = self.navd.update(self.s)
    if instr is not None:
      self.sm.put('navInstruction', instr)
    self.sm.put('carState', car_state(self.v, steer, left_blinker=blinker))
    self.sm.put('carControl', car_control(lat_active))
    self.sm.put('modelV2', model_v2(*self.model_lanes))
    plan = self.tp.plan(self.sm, scene(objects, available), True, self.shadow, self.driver_signals,
                        long_control=self.long_control).as_reader().turnpilotPlan
    a = min(plan_accel(plan.accelCap, plan.speedCap, self.v), (self.cruise - self.v) / 2., 1.5)
    if plan.shouldStop:
      a = min(a, -1.)
    self.v = max(self.v + max(a, -3.5) * DT_MDL, 0.)
    self.s += self.v * DT_MDL
    self.t += DT_MDL
    self.trace.append((self.s, self.v, plan))
    return plan

  def distance(self):
    return self.tp.follower.distance

  def run(self, seconds, objects=lambda sim: (), turn_at=None):
    for _ in range(int(seconds / DT_MDL)):
      steer = 90. if turn_at is not None and abs(self.s - turn_at) < 8. else 0.
      self.step(objects(self), steer=steer)


def rising(trace, field, value):
  prev, edges = None, []
  for s, _, plan in trace:
    cur = getattr(plan, field)
    if cur == value and prev != value:
      edges.append(s)
    prev = cur
  return edges


def test_disabled_and_no_route():
  sm = FakeSM()
  plan = TurnPilot().update(sm, scene(available=False), False, False).as_reader().turnpilotPlan
  assert plan.state == State.disabled and plan.desire == Desire.none
  assert (plan.accelCap, plan.speedCap, plan.shouldStop) == (NO_ACCEL_CAP, NO_SPEED_CAP, False)
  sm.put('navInstruction', nav_instruction(100., valid=False), valid=False)
  plan = TurnPilot().update(sm, scene(available=False), True, False).as_reader().turnpilotPlan
  assert plan.state == State.noRoute and not plan.routeValid


def test_nothing_without_openpilot_longitudinal():
  # stock ACC holds the speed: the caps would do nothing, and a turn desire would go out at road speed
  sim = Sim([(300., "turn", "left"), (1500., "turn", "right")], cruise=15., nav_lanes={0: THREE_LANES_LEFT_TURN},
            model_lanes=(2, 0), long_control=False)
  sim.run(30., turn_at=300.)
  assert sim.s > 320.
  for _, _, plan in sim.trace:
    assert plan.state == State.disabled and plan.reason == NO_LONG_REASON
    assert (plan.desire, plan.laneChange, plan.signalPrompt) == (Desire.none, LaneChange.none, LaneChange.none)
    assert (plan.accelCap, plan.speedCap, plan.shouldStop) == (NO_ACCEL_CAP, NO_SPEED_CAP, False)


def test_left_turn_drive():
  sim = Sim([(300., "turn", "left"), (1500., "turn", "right")])
  sim.run(45., turn_at=300.)
  states = [str(plan.state) for _, _, plan in sim.trace]
  assert states[0] == 'following' and 'approaching' in states and 'turning' in states
  assert 1 <= len(rising(sim.trace, 'desire', Desire.turnLeft)) <= 4  # repeats until turned; this sim has no yaw rate
  at_desire = [v for s, v, _ in sim.trace if 300. - TURN_DESIRE_LEAD_M < s < 300.]  # gate 1's speed from the pulse on
  assert max(at_desire) < V_TURN_MIN + 0.3
  after = [plan for s, _, plan in sim.trace if s > 320.]
  assert after and all(plan.maneuverModifier == "right" for plan in after)
  assert all(plan.desire == Desire.none for plan in after)


def test_consecutive_turns_drive():
  sim = Sim([(300., "turn", "right"), (420., "turn", "left"), (2000., "turn", "right")])
  for _ in range(int(60. / DT_MDL)):
    near = [m for m in (300., 420.) if abs(sim.s - m) < 8.]
    sim.step(available=True, steer=90. if near else 0.)
  # a pulse each, and more on the REPULSE_S clock while the car hasn't turned (this sim has no yaw rate)
  assert 1 <= len(rising(sim.trace, 'desire', Desire.turnRight)) <= 4
  assert 1 <= len(rising(sim.trace, 'desire', Desire.turnLeft)) <= 4
  assert sim.s > 440.


def test_highway_bends_are_left_alone():
  # "continue slight right" and "new name slight left" at 30 m/s: no keep desire, no braking
  sim = Sim([(400., "continue", "slight right"), (900., "new name", "slight left"), (5000., "turn", "right")], cruise=30.)
  sim.run(40.)
  assert sim.s > 1000. and min(v for _, v, _ in sim.trace) > 29.9
  assert all(plan.desire == Desire.none and plan.speedCap == NO_SPEED_CAP for _, _, plan in sim.trace)


def test_turn_desire_waits_for_openpilot_to_steer():
  # the driver engages 20 m before the left: no desire before, a fresh pulse then
  sim = Sim([(300., "turn", "left"), (1500., "turn", "right")])
  for _ in range(int(45. / DT_MDL)):
    sim.step(lat_active=sim.s > 280.)
  edges = rising(sim.trace, 'desire', Desire.turnLeft)
  assert edges and 280. < edges[0] < 281.
  assert all(plan.desire == Desire.none for s, _, plan in sim.trace if s < 280.)


def test_destination_change_drops_the_old_turn():
  # a new destination from the phone 25 m before a left, whose route starts with a left 284 m on: the old pulse is
  # cleared at once, the car speeds up again, and the next pulse waits for the new left's lead
  sim = Sim([(300., "turn", "left"), (1500., "turn", "right")])
  switch = None
  for _ in range(int(80. / DT_MDL)):
    if switch is None and sim.distance() < 25.:
      switch = sim.s
      sim.navd = FakeNavd([(switch + 284., "turn", "left"), (switch + 1500., "turn", "right")])
    sim.step()
  after = [(s, v, plan) for s, v, plan in sim.trace if s > switch]
  assert next(s for s, _, plan in after if plan.desire == CLEAR_DESIRE) < switch + 2.
  assert all(plan.desire != Desire.turnLeft for _, _, plan in after if plan.maneuverDistance > TURN_DESIRE_LEAD_M)
  assert any(plan.desire == Desire.turnLeft for _, _, plan in after) and max(v for _, v, _ in after) > 12.


def test_oncoming_car_holds_unprotected_left():
  spawn = {}

  def oncoming(sim):
    if 'x' not in spawn and sim.distance() < 40.:
      spawn['x'], spawn['t'] = 300. + 72., sim.t  # reaches the junction 8 s later, just as the car would
    if 'x' not in spawn:
      return ()
    x_world = spawn['x'] - 9. * (sim.t - spawn['t'])
    return ({'cls': 'car', 'x': x_world - sim.s, 'y': 5., 'vx': -9.},)

  sim = Sim([(300., "turn", "left"), (1500., "turn", "right")])
  sim.run(60., objects=oncoming, turn_at=300.)
  assert any(plan.reason == "waiting for oncoming" and plan.state == State.holding for _, _, plan in sim.trace)
  # the car stopped at the hold line while the oncoming car was less than 5 s out, then took the turn
  arrive = spawn['t'] + 72. / 9.
  for i, (s, _, _) in enumerate(sim.trace):
    t = (i + 1) * DT_MDL
    if arrive - ONCOMING_TTA_S < t < arrive - 1.:
      assert s < 300. - HOLD_LINE_M + 1.5
  assert any(v < 0.1 and plan.shouldStop for _, v, plan in sim.trace)
  assert sim.s > 320.


def test_detector_unavailable_holds_unprotected_left():
  sim = Sim([(300., "turn", "left"), (1500., "turn", "right")])
  for _ in range(int(45. / DT_MDL)):
    plan = sim.step(available=False)
  assert plan.reason == "detector unavailable" and plan.state == State.holding
  assert 300. - HOLD_LINE_M - 1.5 < sim.s < 300. - HOLD_LINE_M + 1.5 and sim.v < 0.1


def test_pedestrian_in_path_stops_then_goes():
  def pedestrian(sim):
    return ({'cls': 'pedestrian', 'x': 200. - sim.s, 'y': 0.3, 'inPath': True},) if sim.t < 30. else ()

  sim = Sim([(1000., "turn", "right"), (2000., "turn", "left")])
  sim.run(25., objects=pedestrian)
  assert sim.v < 0.1 and 200. - PED_STOP_GAP_M - 2. < sim.s < 200.
  assert sim.trace[-1][2].reason == "pedestrian in path" and sim.trace[-1][2].shouldStop
  sim.run(10., objects=pedestrian)
  assert sim.s > 205.


def test_lane_change_request_and_driver_prompt():
  sim = Sim([(500., "turn", "left"), (2000., "turn", "right")], cruise=15., nav_lanes={0: THREE_LANES_LEFT_TURN}, model_lanes=(2, 0))
  sim.run(15.)
  requests = [(s, plan) for s, _, plan in sim.trace if plan.laneChange == LaneChange.left]
  assert requests and all(plan.state == State.laneChange for _, plan in requests)
  start, end = lane_change_window(15., 2)
  assert 500. - start < requests[0][0] < 500. - end

  sim = Sim([(500., "turn", "left"), (2000., "turn", "right")], cruise=15., nav_lanes={0: THREE_LANES_LEFT_TURN}, model_lanes=(2, 0),
            driver_signals=True)
  sim.run(15.)
  assert any(plan.signalPrompt == LaneChange.left and plan.state == State.laneChange for _, _, plan in sim.trace)
  assert all(plan.laneChange == LaneChange.none for _, _, plan in sim.trace)
  plan = sim.step(blinker=True)
  assert plan.laneChange == LaneChange.left


def test_shadow_computes_the_same_plan():
  def oncoming(sim):
    return ({'cls': 'car', 'x': 330. - sim.s, 'y': 5., 'vx': -9.},) if sim.t > 18. else ()

  runs = []
  for shadow in (False, True):
    sim = Sim([(300., "turn", "left"), (1500., "turn", "right")], nav_lanes={0: THREE_LANES_LEFT_TURN}, model_lanes=(1, 0), shadow=shadow)
    sim.run(40., objects=oncoming, turn_at=300.)
    runs.append(sim.trace)
  for (_, _, p0), (_, _, p1) in zip(*runs, strict=True):
    assert p1.shadow and not p0.shadow
    d0, d1 = p0.to_dict(), p1.to_dict()
    d0.pop('shadow', None)
    d1.pop('shadow', None)
    assert d0 == d1


class FakeParams(dict):
  def get(self, key):
    return super().get(key)


def test_long_control_from_car_params():
  assert not long_control_for(FakeParams())  # card hasn't written CarParams yet
  for long_control in (False, True):
    cp = car.CarParams.new_message(openpilotLongitudinalControl=long_control)
    assert long_control_for(FakeParams(CarParams=cp.to_bytes())) == long_control
