"""turnpilotd closed loop: synthetic navd, model and scene around a point-mass car that follows the plan's limits."""
from openpilot.cereal import custom, log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.hooks import plan_accel
from openpilot.selfdrive.turnpilot.lane_planner import LC_WARN_S, lane_change_window
from openpilot.selfdrive.turnpilot.scene.lane_paint import UNKNOWN, LinePaint
from openpilot.selfdrive.turnpilot.scene_policy import HOLD_LINE_M, NO_ACCEL_CAP, ONCOMING_TTA_S, PED_STOP_GAP_M
from openpilot.selfdrive.turnpilot.tests.synthetic import FakeNavd, car_state, model_v2, nav_instruction, scene
from openpilot.selfdrive.turnpilot.desire_timing import TURN_DESIRE_LEAD_M
from openpilot.selfdrive.turnpilot.turn_speed import NO_SPEED_CAP, V_TURN_MIN
from openpilot.selfdrive.turnpilot.turnpilotd import TurnPilot

State = custom.TurnpilotPlan.State
LaneChange = custom.TurnpilotPlan.LaneChange
Desire = log.Desire
THREE_LANES_LEFT_TURN = ((("left",), True), (("straight",), False), (("straight", "right"), False))


class FakeSM:
  def __init__(self):
    self.data = {'carState': car_state(0.), 'navInstruction': nav_instruction(0., valid=False), 'modelV2': model_v2()}
    self.updated = dict.fromkeys(self.data, False)
    self.valid = dict.fromkeys(self.data, True)
    self.alive = dict.fromkeys(self.data, True)

  def __getitem__(self, s):
    return self.data[s]

  def put(self, s, msg, valid=True):
    self.data[s], self.updated[s], self.valid[s] = msg, True, valid


class Sim:
  def __init__(self, maneuvers, cruise=13., nav_lanes=None, model_lanes=(0, 0), shadow=False, driver_signals=False, paint=(),
               paint_rule=True):
    self.navd, self.tp, self.sm = FakeNavd(maneuvers, nav_lanes), TurnPilot(), FakeSM()
    self.paint, self.paint_rule, self.inputs_valid = paint, paint_rule, True
    self.s, self.v, self.cruise, self.t = 0., cruise, cruise, 0.
    self.model_lanes, self.shadow, self.driver_signals = model_lanes, shadow, driver_signals
    self.trace = []

  def step(self, objects=(), available=True, steer=0., blinker=False):
    self.sm.updated = dict.fromkeys(self.sm.data, False)
    instr = self.navd.update(self.s)
    if instr is not None:
      self.sm.put('navInstruction', instr)
    self.sm.put('carState', car_state(self.v, steer, left_blinker=blinker))
    self.sm.put('modelV2', model_v2(*self.model_lanes))
    plan = self.tp.update(self.sm, scene(objects, available), True, self.shadow, self.driver_signals,
                          paint=self.paint, paint_rule=self.paint_rule, inputs_valid=self.inputs_valid).as_reader().turnpilotPlan
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


BW, SW = LinePaint('broken', 'white', 0.9, 0.3), LinePaint('solid', 'white', 0.9, 0.95)
BROKEN_WHITE_LEFT = (UNKNOWN, BW, SW, UNKNOWN)


def test_lane_change_warned_then_requested():
  sim = Sim([(500., "turn", "left"), (2000., "turn", "right")], cruise=15., nav_lanes={0: THREE_LANES_LEFT_TURN}, model_lanes=(2, 0),
            paint=BROKEN_WHITE_LEFT)
  sim.run(25.)
  offers = rising(sim.trace, 'laneChangeOffer', LaneChange.left)
  requests = [(s, plan) for s, _, plan in sim.trace if plan.laneChange == LaneChange.left]
  assert offers and requests and all(plan.state == State.laneChange for _, plan in requests)
  start, end = lane_change_window(15., 2)
  assert 500. - start < offers[0] < 500. - end
  assert abs((requests[0][0] - offers[0]) - LC_WARN_S * 15.) < 2.  # the warning's 3 s at 15 m/s
  counting = [plan.laneChangeCountdown for s, _, plan in sim.trace if offers[0] <= s < requests[0][0]]
  assert counting[0] > LC_WARN_S - 0.1 and counting[-1] < 0.1 and all(a >= b for a, b in zip(counting, counting[1:], strict=False))
  assert all(plan.laneChangeAbort == 'none' for _, _, plan in sim.trace)

  # the Tesla: the prompt and countdown, nothing until the driver's blinker
  sim = Sim([(500., "turn", "left"), (2000., "turn", "right")], cruise=15., nav_lanes={0: THREE_LANES_LEFT_TURN}, model_lanes=(2, 0),
            driver_signals=True, paint=BROKEN_WHITE_LEFT)
  sim.run(10.)
  assert any(plan.signalPrompt == LaneChange.left and plan.laneChangeOffer == LaneChange.left and plan.state == State.laneChange
             for _, _, plan in sim.trace)
  assert all(plan.laneChange == LaneChange.none for _, _, plan in sim.trace)
  plan = sim.step(blinker=True)
  assert plan.laneChange == LaneChange.left


def test_no_paint_no_lane_change_and_aborts_reach_the_plan():
  # the map's lanes alone, with no paint read, no longer move the car
  sim = Sim([(500., "turn", "left"), (2000., "turn", "right")], cruise=15., nav_lanes={0: THREE_LANES_LEFT_TURN}, model_lanes=(2, 0))
  sim.run(25.)
  assert all(plan.laneChange == LaneChange.none and plan.laneChangeOffer == LaneChange.none for _, _, plan in sim.trace)
  # paint lost under an offer: the plan says why, for selfdrived's alert
  sim = Sim([(500., "turn", "left"), (2000., "turn", "right")], cruise=15., model_lanes=(1, 0), paint=BROKEN_WHITE_LEFT)
  while not sim.trace or sim.trace[-1][2].laneChangeOffer == LaneChange.none:
    sim.step()
  sim.paint = ()
  plan = sim.step()
  assert plan.laneChangeAbort == 'lineNotBroken' and plan.laneChangeOffer == LaneChange.none and plan.laneChange == LaneChange.none
  # inputs failing their checks abort too
  sim = Sim([(500., "turn", "left"), (2000., "turn", "right")], cruise=15., model_lanes=(1, 0), paint=BROKEN_WHITE_LEFT)
  while not sim.trace or sim.trace[-1][2].laneChangeOffer == LaneChange.none:
    sim.step()
  sim.inputs_valid = False
  assert sim.step().laneChangeAbort == 'badInputs'


def test_lane_change_from_lane_paint():
  # no map lanes: the broken white line on the left lets the car over toward a left turn; none read, no change
  sim = Sim([(500., "turn", "left"), (2000., "turn", "right")], cruise=15., model_lanes=(1, 0), paint=BROKEN_WHITE_LEFT)
  sim.run(25.)
  requests = [s for s, _, plan in sim.trace if plan.laneChange == LaneChange.left]
  start, end = lane_change_window(15.)
  assert requests and 500. - start < requests[0] < 500. - end
  assert all(plan.laneChange != LaneChange.right for _, _, plan in sim.trace)
  assert all(plan.laneChangePaintRule == plan.laneChange for _, _, plan in sim.trace)
  sim = Sim([(500., "turn", "left"), (2000., "turn", "right")], cruise=15., model_lanes=(1, 0))
  sim.run(25.)
  assert all(plan.laneChange == LaneChange.none for _, _, plan in sim.trace)
  # TurnPilotLanePaintActive off: nothing acts or is offered, and laneChangePaintRule says what would have been asked for
  sim = Sim([(500., "turn", "left"), (2000., "turn", "right")], cruise=15., model_lanes=(1, 0), paint=BROKEN_WHITE_LEFT, paint_rule=False)
  sim.run(25.)
  assert all(plan.laneChange == LaneChange.none and plan.laneChangeOffer == LaneChange.none for _, _, plan in sim.trace)
  assert any(plan.laneChangePaintRule == LaneChange.left for _, _, plan in sim.trace)


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
