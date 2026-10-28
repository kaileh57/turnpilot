"""turnpilotd closed loop: synthetic navd, model and scene around a point-mass car that follows the plan's limits."""
from unittest import mock

from opendbc.car.structs import car
from openpilot.cereal import custom, log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.hooks import ABORT_HOLD_S, CLEAR_DESIRE, lane_change_events, plan_accel
from openpilot.selfdrive.turnpilot.lane_planner import LC_WARN_S, lane_change_window
from openpilot.selfdrive.turnpilot.scene.lane_paint import UNKNOWN, LinePaint
from openpilot.selfdrive.turnpilot.scene_policy import HOLD_LINE_M, NO_ACCEL_CAP, ONCOMING_TTA_S, PED_STOP_GAP_M
from openpilot.selfdrive.turnpilot.tests.synthetic import FakeNavd, car_control, car_state, junction_of, model_v2, nav_instruction, scene
from openpilot.selfdrive.turnpilot.desire_timing import TURN_DESIRE_LEAD_M
from openpilot.selfdrive.turnpilot.turn_skip import MIN_FIRST_PULSE_M
from openpilot.selfdrive.turnpilot.turn_speed import NO_SPEED_CAP, V_TURN_MIN
from openpilot.selfdrive.turnpilot.turnpilotd import (CAR_UNKNOWN_REASON, DETECTOR_DEADLINE_S, DETECTOR_START_DEADLINE_S, NO_LONG_REASON,
                                                       WAITING_REASON, TurnPilot, long_control_for)
from openpilot.selfdrive.turnpilot.handback import Handback
from openpilot.selfdrive.turnpilot.scene.tracker import STALE_S

State = custom.TurnpilotPlan.State
Fault = custom.TurnpilotPlan.Fault
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
  def __init__(self, maneuvers, cruise=13., nav_lanes=None, model_lanes=(0, 0), shadow=False, driver_signals=False, paint=(),
               paint_rule=True, long_control=True):
    self.navd, self.tp, self.sm = FakeNavd(maneuvers, nav_lanes), TurnPilot(), FakeSM()
    self.paint, self.paint_rule, self.inputs_valid = paint, paint_rule, True
    self.s, self.v, self.cruise, self.t = 0., cruise, cruise, 0.
    self.model_lanes, self.shadow, self.driver_signals, self.long_control = model_lanes, shadow, driver_signals, long_control
    self.big = False
    self.model_running = True  # modeld publishing modelV2 (it doesn't while it loads the detector)
    self.nav_filter = None  # (sim, instruction) -> (instruction, valid): what navd sends instead
    self.trace = []

  def step(self, objects=(), available=True, steer=0., blinker=False, lat_active=True, yaw_rate=0., gas=False,
           engaged=False):
    self.sm.updated = dict.fromkeys(self.sm.data, False)
    instr = self.navd.update(self.s)
    if instr is not None:
      instr, valid = self.nav_filter(self, instr) if self.nav_filter is not None else (instr, True)
      self.sm.put('navInstruction', instr, valid)
    self.sm.put('carState', car_state(self.v, steer, left_blinker=blinker, yaw_rate=yaw_rate, gas_pressed=gas))
    self.sm.put('carControl', car_control(lat_active, engaged))
    if self.model_running:
      self.sm.put('modelV2', model_v2(*self.model_lanes, big=self.big))
    plan = self.tp.plan(self.sm, scene(objects, available), True, self.shadow, self.driver_signals, paint=self.paint,
                        paint_rule=self.paint_rule, inputs_valid=self.inputs_valid, long_control=self.long_control).as_reader().turnpilotPlan
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
  plan = TurnPilot().update(sm, scene(), True, False).as_reader().turnpilotPlan
  assert plan.state == State.noRoute and not plan.routeValid and plan.handback == Handback.none


def test_nothing_without_openpilot_longitudinal():
  # stock ACC holds the speed: the caps would do nothing, and a turn desire would go out at road speed
  sim = Sim([(300., "turn", "left"), (1500., "turn", "right")], cruise=15., nav_lanes={0: THREE_LANES_LEFT_TURN},
            model_lanes=(2, 0), long_control=False)
  sim.run(30., turn_at=300.)
  assert sim.s > 320.
  for _, _, plan in sim.trace:
    assert plan.state == State.disabled and plan.reason == NO_LONG_REASON and plan.fault == Fault.noLongitudinal
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
  # the driver engages 30 m before the left: no desire before, a fresh pulse then
  sim = Sim([(300., "turn", "left"), (1500., "turn", "right")])
  for _ in range(int(45. / DT_MDL)):
    sim.step(lat_active=sim.s > 270.)
  edges = rising(sim.trace, 'desire', Desire.turnLeft)
  assert edges and 270. < edges[0] < 271.
  assert all(plan.desire == Desire.none for s, _, plan in sim.trace if s < 270.)
  assert not any(plan.tooFastToTurn for _, _, plan in sim.trace)
  # engaged 20 m before it, too late for a first pulse to be taken: skipped out loud. Driving it with openpilot off
  # isn't skipped under the driver
  for engage in (280., None):
    sim = Sim([(300., "turn", "left"), (1500., "turn", "right")])
    for _ in range(int(45. / DT_MDL)):
      sim.step(lat_active=engage is not None and sim.s > engage)
    assert all(plan.desire == Desire.none for _, _, plan in sim.trace), engage
    assert any(plan.tooFastToTurn for _, _, plan in sim.trace) == (engage is not None), engage


def test_destination_change_keeps_a_committed_turn():
  # a new destination from the phone 25 m before a left whose desire is out, its route starting with a left 284 m on
  # (navd's next route, routeCount 1): a committed turn is never cancelled. The old left is held to its end, its desire
  # never cleared and its cap kept until the car is past it, then the new route is taken up and its left pulses at its lead
  sim = Sim([(300., "turn", "left"), (1500., "turn", "right")])
  switch = None
  for _ in range(int(80. / DT_MDL)):
    if switch is None and sim.distance() < 25.:
      switch = sim.s
      assert sim.trace[-1][2].desire == Desire.turnLeft
      sim.navd = FakeNavd([(switch + 284., "turn", "left"), (switch + 1500., "turn", "right")], route_count=1)
    sim.step()
  after = [(s, v, plan) for s, v, plan in sim.trace if s > switch]
  before = [(v, plan) for s, v, plan in after if s < 300.]
  assert all(plan.desire != CLEAR_DESIRE and plan.speedCap < 6. for _, plan in before)
  assert all(b[0] <= a[0] + 1e-6 for a, b in zip(before, before[1:], strict=False))  # never speeding up into it
  assert all(plan.desire != Desire.turnLeft for s, _, plan in after if s > 316. and plan.maneuverDistance > TURN_DESIRE_LEAD_M)
  new_left = [s for s in rising(sim.trace, 'desire', Desire.turnLeft) if s > 316.]
  assert new_left and switch + 284. - new_left[0] > MIN_FIRST_PULSE_M and max(v for _, v, _ in after) > 12.
  # not yet committed (60 m out, no desire out): the old left is dropped at once, and the car speeds up again
  sim = Sim([(300., "turn", "left"), (1500., "turn", "right")])
  sim.cruise = 13.
  while sim.distance() > 60. or sim.s < 10.:
    sim.step()
  switch = sim.s
  sim.navd = FakeNavd([(switch + 284., "turn", "left"), (switch + 1500., "turn", "right")], route_count=1)
  for _ in range(int(10. / DT_MDL)):
    sim.step()
  assert not any(p.desire == Desire.turnLeft for s, _, p in sim.trace if switch < s < switch + 200.)
  assert sim.v > 12.


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


class FakeParams(dict):
  def get(self, key):
    return super().get(key)


def test_long_control_from_car_params():
  assert long_control_for(FakeParams()) is None  # card hasn't written CarParams yet: not known
  for long_control in (False, True):
    cp = car.CarParams.new_message(openpilotLongitudinalControl=long_control)
    assert long_control_for(FakeParams(CarParams=cp.to_bytes())) == long_control


def test_unknown_longitudinal_is_disabled_without_a_fault():
  # every drive starts before card has written CarParams: no "TurnPilot unavailable" flash, and nothing acted on
  sim = Sim([(300., "turn", "left")], long_control=None)
  sim.run(2.)
  for _, _, plan in sim.trace:
    assert plan.state == State.disabled and plan.reason == CAR_UNKNOWN_REASON and plan.fault == Fault.none
    assert plan.desire == Desire.none and plan.speedCap == NO_SPEED_CAP


def test_enabled_plan_has_no_fault():
  sim = Sim([(300., "turn", "left")])
  sim.run(2.)
  assert all(plan.fault == Fault.none for _, _, plan in sim.trace)


def test_waits_for_the_detector_before_acting():
  # TurnPilot turned on with no detections yet (modeld loads the detector only while disengaged): no desire, no lane
  # change, no turn cap, and no handback either, since nothing has stopped
  sim = Sim([(150., "turn", "left"), (1500., "turn", "right")])
  for _ in range(int(DETECTOR_START_DEADLINE_S / DT_MDL) - 1):
    plan = sim.step(available=False)
    assert plan.state == State.unavailable and plan.reason == WAITING_REASON and plan.handback == Handback.none
    assert plan.desire == Desire.none and plan.laneChange == LaneChange.none and plan.speedCap == NO_SPEED_CAP
  assert sim.s > 170.  # drove past the turn, nothing held it
  plan = sim.step()  # first detections: acting
  assert plan.state != State.unavailable and plan.handback == Handback.none


def test_detector_stops_hand_back_without_a_hold():
  sim = Sim([(400., "turn", "left"), (1500., "turn", "right")])
  sim.run(5.)
  assert sim.trace[-1][2].handback == Handback.none
  # a slow spell: modeld's forced run comes 1 s after the last detection, the scene unavailable from STALE_S: no handback
  for _ in range(int((1.0 - STALE_S) / DT_MDL)):
    assert sim.step(available=False).handback == Handback.none
  sim.step()
  # stopped: handed back once the deadline passes, with no desire, no lane change and no stop-and-wait at the turn
  plans = [sim.step(available=False) for _ in range(int(60. / DT_MDL))]
  first = next(i for i, p in enumerate(plans) if p.handback != Handback.none)
  assert abs((first + 1) * DT_MDL - (DETECTOR_DEADLINE_S - STALE_S)) <= 1.5 * DT_MDL
  assert all(p.handback == Handback.detectorStopped and p.state == State.unavailable for p in plans[first:])
  assert all(p.desire == Desire.none and p.laneChange == LaneChange.none and not p.shouldStop for p in plans)
  assert plans[-1].reason == "detector stopped: take control"
  assert sim.s > 420. and min(v for _, v, _ in sim.trace[-len(plans):]) > 5.
  # the detector comes back: acting again
  assert sim.step().handback == Handback.none and sim.trace[-1][2].state != State.unavailable


def test_handback_mid_turn_keeps_the_turn_cap_and_sends_no_new_pulse():
  # the turn desire went out and the car is slowing for the turn when the detector stops: the cap stays (a handback must
  # not speed the car up), the desire out is never cancelled, and no new pulse goes out
  sim = Sim([(300., "turn", "left"), (1500., "turn", "right")])
  while sim.distance() > 15.:
    sim.step()
  assert any(p.desire == Desire.turnLeft for _, _, p in sim.trace)
  capped = sim.trace[-1][2].speedCap
  assert capped < NO_SPEED_CAP
  n = len(sim.trace)
  plans = [sim.step(available=False) for _ in range(int(1.5 / DT_MDL))]
  assert plans[-1].handback == Handback.detectorStopped and plans[-1].speedCap <= capped + 1.
  first = next(i for i, p in enumerate(plans) if p.handback != Handback.none)
  assert all(s < sim.trace[n + first][0] for s in rising(sim.trace, 'desire', Desire.turnLeft))
  assert plans[-1].desire == Desire.turnLeft  # held, not cancelled
  assert all(p.desire in (Desire.none, Desire.turnLeft, CLEAR_DESIRE) for p in plans)


def test_handback_in_the_turn_holds_its_desire_to_the_clear():
  # never cancel a turn under way: handed back as the car starts turning, the desire stays out until the turn's own
  # clear (its angle less CLEAR_MARGIN_DEG), as when TurnPilot keeps acting, with no new rising edge
  def drive(stop_detector):
    sim = Sim([(300., "turn", "left"), (1500., "turn", "right")])
    while sim.distance() > 5.:
      sim.step()
    for _ in range(int(10. / DT_MDL)):
      sim.step(available=not stop_detector, yaw_rate=0.3)  # 17 deg/s
    return sim
  kept, handed = drive(False), drive(True)
  assert handed.trace[-1][2].handback == Handback.detectorStopped
  def desires(sim):
    return [p.desire for _, _, p in sim.trace]
  assert CLEAR_DESIRE in desires(handed)
  assert desires(handed).index(CLEAR_DESIRE) == desires(kept).index(CLEAR_DESIRE)
  assert len(rising(handed.trace, 'desire', Desire.turnLeft)) == len(rising(kept.trace, 'desire', Desire.turnLeft))


def test_fault_carries_a_model_fallback():
  # a fault is a disabled plan, but a model fallback's handback still goes out on it: the handback outranks the fault
  sim = Sim([(400., "turn", "left")], long_control=False)
  sim.big = True
  sim.run(1.)
  assert sim.trace[-1][2].handback == Handback.none and sim.trace[-1][2].fault == Fault.noLongitudinal
  sim.big = False
  plan = sim.step()
  assert plan.state == State.disabled and plan.fault == Fault.noLongitudinal and plan.handback == Handback.modelFellBack
  assert sim.tp.cancel_route


def test_skip_and_wrong_lane_unset_while_not_acting():
  # too fast for a near left with the detector stopped: no skip (navd would reroute) and no wrong-lane flag, since the
  # handback owns the alert slot and TurnPilot isn't driving the route
  sim = Sim([(400., "turn", "left"), (1500., "turn", "right")], cruise=20., nav_lanes={0: THREE_LANES_LEFT_TURN}, model_lanes=(0, 2))
  sim.run(2.)
  for _ in range(int(30. / DT_MDL)):
    p = sim.step(available=False)
    if p.handback != Handback.none:
      assert not p.tooFastToTurn and p.laneMismatch == custom.TurnpilotPlan.LaneMismatch.none


def test_model_fallback_hands_back_and_cancels_the_route():
  sim = Sim([(400., "turn", "left"), (1500., "turn", "right")])
  sim.big = True
  sim.run(5.)
  assert sim.trace[-1][2].handback == Handback.none and not sim.tp.cancel_route
  sim.big = False  # modeld fell back to the small model
  plan = sim.step()
  assert plan.handback == Handback.modelFellBack and plan.state == State.unavailable
  assert sim.tp.cancel_route  # main() removes NavDestination once, which ends navd's route
  sim.tp.cancel_route = False
  plans = [sim.step() for _ in range(int(60. / DT_MDL))]
  assert all(p.handback == Handback.modelFellBack and p.desire == Desire.none and p.laneChange == LaneChange.none for p in plans)
  assert not sim.tp.cancel_route  # once
  # for the rest of the drive: turning TurnPilot off and on doesn't bring it back on the small model
  sim.tp.update(sim.sm, scene(), False, False)
  assert sim.step().handback == Handback.modelFellBack


def test_small_model_from_the_start_is_not_a_fallback():
  # no chestnut: modeld runs the small model from the start. That is no fallback; with no detector pkl for the small
  # model there are no detections: TurnPilot waits for them, then hands back as a detector that never started
  sim = Sim([(400., "turn", "left")])
  for _ in range(int(5. / DT_MDL)):
    plan = sim.step(available=False)
  assert plan.handback == Handback.none and plan.reason == WAITING_REASON and not sim.tp.cancel_route
  for _ in range(int(DETECTOR_START_DEADLINE_S / DT_MDL)):
    plan = sim.step(available=False)
  assert plan.handback == Handback.noDetector and not sim.tp.cancel_route


def offered_sim(**kwargs):
  sim = Sim([(500., "turn", "left"), (2000., "turn", "right")], cruise=15., model_lanes=(1, 0), paint=BROKEN_WHITE_LEFT, **kwargs)
  while not sim.trace or sim.trace[-1][2].laneChangeOffer == LaneChange.none:
    sim.step()
  return sim


def test_handback_cancels_an_offered_lane_change_out_loud():
  sim = offered_sim()
  plans = [sim.step(available=False) for _ in range(int(3. / DT_MDL))]
  first = next(i for i, p in enumerate(plans) if p.handback != Handback.none)
  assert all(p.laneChangeOffer == LaneChange.left for p in plans[:first])
  held = [p for p in plans[first:] if p.laneChangeAbort == 'handedBack']
  assert plans[first].laneChangeAbort == 'handedBack' and abs(len(held) * DT_MDL - ABORT_HOLD_S) < 2 * DT_MDL
  assert all(p.laneChangeOffer == LaneChange.none and p.laneChange == LaneChange.none and p.signalPrompt == LaneChange.none
             for p in plans[first:])


def test_fault_or_turning_off_cancels_an_offered_lane_change_out_loud():
  for stop in ('no_long', 'error', 'off'):
    sim = offered_sim()
    if stop == 'no_long':
      sim.long_control = False
      plans = [sim.step() for _ in range(int(3. / DT_MDL))]
    elif stop == 'error':
      sim.tp.update = mock.Mock(side_effect=RuntimeError("synthetic failure"))
      plans = [sim.step() for _ in range(int(3. / DT_MDL))]
    else:
      plans = [sim.tp.plan(sim.sm, scene(), False, False, long_control=True).as_reader().turnpilotPlan for _ in range(int(3. / DT_MDL))]
    reason = 'turnedOff' if stop == 'off' else 'fault'
    assert all(p.state == State.disabled and p.laneChangeOffer == LaneChange.none for p in plans), stop
    held = [p for p in plans if p.laneChangeAbort == reason]
    assert plans[0].laneChangeAbort == reason and abs(len(held) * DT_MDL - ABORT_HOLD_S) < 2 * DT_MDL, stop
    # selfdrived raises the abort from a disabled plan turned off; a fault's own alert (turnpilotUnavailable, with its
    # chime) outranks it and says what happened, so the abort isn't raised under it (hooks.plan_events)
    if stop == 'off':
      assert lane_change_events(plans[0]) == ('turnpilotLaneChangeAborted',), stop
    else:
      assert plans[0].fault != Fault.none and lane_change_events(plans[0]) == (), stop


def invalid_nav(sim):
  """What navd sends off the route or rerouting: an invalid instruction for the same route."""
  return nav_instruction(0., valid=False, route_count=sim.navd.route_count), False


def approach_with_desire_out(cruise=13., to=25.):
  sim = Sim([(300., "turn", "left", junction_of("left")), (1500., "turn", "right", junction_of("right"))], cruise=cruise)
  while sim.distance() > to or sim.s < 10.:
    sim.step()
  assert Desire.turnLeft in [p.desire for _, _, p in sim.trace] and sim.trace[-1][2].speedCap < NO_SPEED_CAP
  return sim


def assert_turn_held(sim, n):
  """From trace index n on: the turn's desire never cancelled or skipped, its cap kept, the car not sped up before the
  junction at s=300."""
  before = sim.trace[n - 1][2]
  after = [(s, v, p) for s, v, p in sim.trace[n:] if s < 300.]
  assert after
  assert all(p.desire != CLEAR_DESIRE and not p.tooFastToTurn for _, _, p in after)
  assert all(p.speedCap <= before.speedCap + 0.5 for _, _, p in after)
  assert max(v for _, v, _ in after) <= sim.trace[n - 1][1] + 0.05


def test_one_off_route_update_keeps_the_committed_turn():
  # the desire is out 12 m before the left when one navd update has a fix 20 m off the step: no CLEAR, no skip of the
  # same turn coming back under a new id, the cap stays, and no new pulse goes out from 10 m
  sim = approach_with_desire_out(cruise=13., to=12.)
  n, glitched = len(sim.trace), []
  def once(sim, instr):
    if not glitched:
      glitched.append(sim.s)
      return invalid_nav(sim)
    return instr, True
  sim.nav_filter = once
  while sim.s < 330.:
    sim.step()
  assert glitched
  assert_turn_held(sim, n)
  assert all(p.desire != CLEAR_DESIRE and not p.tooFastToTurn for _, _, p in sim.trace)


def test_model_fallback_on_the_approach_holds_the_turn_and_defers_the_route_cancel():
  # the big model falls back 25 m before the left, its desire out: the handback holds the desire and the cap, and the
  # route isn't cancelled until the turn is over. Were navd's route gone at once (every instruction invalid), the turn
  # is still held: no CLEAR, the cap stays, the car doesn't speed up into the junction
  for route_gone in (False, True):
    sim = approach_with_desire_out()
    sim.big = True
    sim.step()
    n = len(sim.trace)
    sim.big = False
    if route_gone:
      sim.nav_filter = lambda sim, instr: invalid_nav(sim)
    held = []
    while sim.s < 330.:
      plan = sim.step()
      assert plan.handback == Handback.modelFellBack
      if sim.s < 300.:
        held.append(sim.tp.cancel_route and sim.tp.turn_committed)
    assert all(held) and sim.tp.cancel_route and not sim.tp.turn_committed, route_gone
    assert_turn_held(sim, n)


def test_reroute_onto_the_same_turn_keeps_it():
  # a same-destination reroute starts 25 m before the left (GPS drift): two invalid updates while it runs, one between
  # the routes, then the new route with the same left. The turn is held throughout, under the same maneuver
  sim = approach_with_desire_out()
  n, first, frames = len(sim.trace), sim.tp.follower.maneuver_id, []
  def reroute(sim, instr):
    frames.append(sim.s)
    if len(frames) <= 3:
      if len(frames) == 3:
        sim.navd.route_count = 1  # the new route is in: one invalid instruction marks it, then its own
      return invalid_nav(sim)
    return instr, True
  sim.nav_filter = reroute
  while sim.s < 330.:
    sim.step()
    if sim.s < 290.:
      assert sim.tp.follower.maneuver_id == first
  assert_turn_held(sim, n)


def test_navd_going_quiet_on_the_approach_keeps_the_turn_cap():
  sim = approach_with_desire_out()
  n = len(sim.trace)
  sim.nav_filter = lambda sim, instr: (instr, True)
  update = sim.navd.update
  sim.navd.update = lambda s: None if s < 310. else update(s)
  while sim.s < 330.:
    sim.step()
  assert_turn_held(sim, n)


def test_turn_just_after_a_bend_or_a_new_name_is_slowed_for_not_skipped():
  # a left junction turn 40-50 m after a step that is NONE (a bend no other road meets, a straight-on name change): the
  # turn cap brakes for the left before navd moves past that step, so the left isn't skipped as too fast, and its pulse
  # goes once the car is where navd moves past the step, not up to a second later
  for first in (("turn", "right", junction_of("right", 0, 60.)), ("new name", "straight", (0., 0., 2))):
    for gap in (40., 50.):
      sim = Sim([(300.,) + first, (300. + gap, "turn", "left", junction_of("left")), (1500., "turn", "right", junction_of("right"))])
      while sim.s < 305. + gap:
        sim.step()
      plans = [p for _, _, p in sim.trace]
      assert not any(p.tooFastToTurn for p in plans), (first, gap)
      edges = rising(sim.trace, 'desire', Desire.turnLeft)
      assert 300. + 9. < edges[0] < 300. + gap - MIN_FIRST_PULSE_M, (first, gap)


def test_turn_too_close_after_a_bend_for_its_first_pulse_is_skipped_out_loud():
  # navd moves past a NONE step only 10 m beyond it, so a left 5-30 m after it comes up inside the lead gate 1's model
  # takes a turn from (and under 10 m, past the junction). Its first pulse never goes that late: the turn is skipped out
  # loud (tooFastToTurn, navd reroutes), not pulsed into a missed turn, or into the curb past the junction
  for first in (("turn", "right", junction_of("right", 0, 60.)), ("new name", "straight", (0., 0., 2))):
    for gap in (5., 8., 12., 15., 20., 25., 30.):
      sim = Sim([(300.,) + first, (300. + gap, "turn", "left", junction_of("left", 2)), (1500., "turn", "right", junction_of("right", 2))])
      while sim.s < 300. + gap + 20.:
        sim.step()
      assert not rising(sim.trace, 'desire', Desire.turnLeft), (first, gap)
      assert any(p.tooFastToTurn for _, _, p in sim.trace), (first, gap)


SOFT_DISABLE_S = 3.  # selfdrived/state.py SOFT_DISABLE_TIME: lat stays active this long after the handback


def recover_after_outage(cruise, outage, reengage=1.):
  """The turn desire goes out, then the detector stops for outage s: handback, soft disable (lat active SOFT_DISABLE_S),
  then openpilot disengaged until the driver engages again reengage s after the detector comes back. Returns the sim
  and the trace index at which TurnPilot acts again."""
  sim = Sim([(300., "turn", "left", junction_of("left")), (1500., "turn", "right", junction_of("right"))], cruise=cruise)
  while sim.distance() > TURN_DESIRE_LEAD_M - 1. or sim.s < 10.:
    sim.step()
  assert sim.trace[-1][2].desire == Desire.turnLeft
  t, handback_t, back_t, back_i = 0., None, None, None
  while sim.s < 310.:
    available = t >= outage
    if handback_t is None and not sim.tp.acting:
      handback_t = t
    if available and back_t is None:
      back_t, back_i = t, len(sim.trace)
    disengaged = handback_t is not None and t - handback_t >= SOFT_DISABLE_S
    lat = not disengaged or (back_t is not None and t - back_t >= reengage)
    sim.step(available=available, lat_active=lat)
    t += DT_MDL
  return sim, back_i


def test_turn_dropped_while_disengaged_is_not_left_half_driven():
  # the detector is back after a long outage and the driver engages again before the left, whose desire was dropped
  # while openpilot was off: once openpilot steers again, TurnPilot never drives on toward it capped, 'approaching' and
  # with no desire and no alert (a fresh pulse, or the turn skipped out loud)
  for cruise in (13., 3.5):
    sim, back = recover_after_outage(cruise, outage=5.)
    engaged = back + int(round(1. / DT_MDL))
    assert any(p.desire == Desire.turnLeft or p.tooFastToTurn for _, _, p in sim.trace[engaged:engaged + 5]), cruise
    after = [p for s, _, p in sim.trace[engaged + 5:] if s < 300.]
    assert after, cruise
    silent = [p for p in after if p.desire == Desire.none and p.speedCap < NO_SPEED_CAP and not p.tooFastToTurn and
              p.state == State.approaching]
    assert not silent, cruise


def test_dropped_turn_taken_up_or_skipped_with_its_pulse_cleared():
  # the left's desire went out 38 m before it, then the detector stopped and openpilot stopped steering lat_off s in;
  # the detector is back loss s in, with openpilot steering again. The pulse may still be in the model's queue. Room to
  # make the turn and for a pulse to be taken (ENTRY_M out): a fresh pulse, and the turn driven. Too close: skipped out
  # loud, its pulse cleared from the model, and the turn cap kept until the car is past the junction (never turning
  # from the old pulse at speed, never speeding up into the junction)
  for lat_off, loss, pulse in ((1.5, 2.5, True), (3.0, 4.0, True), (3.2, 3.6, True), (4.4, 5.0, False)):
    sim = Sim([(300., "turn", "left", junction_of("left")), (1500., "turn", "right", junction_of("right"))], cruise=13.)
    while sim.distance() > 38. or sim.s < 10.:
      sim.step()
    for i in range(int(round(loss / DT_MDL))):
      sim.step(available=False, lat_active=i * DT_MDL < lat_off)
    back = len(sim.trace)
    while sim.s < 310.:
      sim.step()
    after = [(s, v, p) for s, v, p in sim.trace[back:]]
    to_junction = [(v, p) for s, v, p in after if s < 300.]
    assert all(b[0] <= a[0] + 1e-6 for a, b in zip(to_junction, to_junction[1:], strict=False)), (lat_off, loss)
    if pulse:
      assert after[0][2].desire == Desire.turnLeft and not any(p.tooFastToTurn for _, _, p in after), (lat_off, loss)
    else:
      assert after[0][2].tooFastToTurn and after[0][2].desire == CLEAR_DESIRE, (lat_off, loss)
      assert not any(p.desire == Desire.turnLeft for _, _, p in after), (lat_off, loss)
      assert all(p.speedCap < 5. for _, p in to_junction), (lat_off, loss)


def test_detector_back_within_the_soft_disable_keeps_the_turn():
  # back before openpilot disengaged: the held desire stays out with no new rising edge, and TurnPilot drives the turn on
  sim, back = recover_after_outage(13., outage=2.)
  after = [p for s, _, p in sim.trace[back:] if s < 300.]
  assert after and all(p.desire in (Desire.turnLeft, Desire.none) and not p.tooFastToTurn for p in after)
  assert after[0].desire == Desire.turnLeft and all(p.state == State.turning for p in after[:5])



def test_detector_back_after_a_stop_on_the_approach_keeps_the_turn():
  # stopped at a light 20 m before a committed left, the detector drops for 2.2 s and comes back with openpilot still
  # engaged (inside the soft disable): the standstill re-armed the desire, it wasn't dropped, so the turn isn't skipped
  # and its pulse goes again as the car pulls away
  sim = Sim([(300., "turn", "left", junction_of("left")), (1500., "turn", "right", junction_of("right"))], cruise=13.)
  while sim.distance() > 20. or sim.s < 10.:
    sim.step()
  assert Desire.turnLeft in [p.desire for _, _, p in sim.trace]
  sim.cruise = 0.
  for _ in range(int(3. / DT_MDL)):  # brake to a stop
    sim.step()
  for i in range(int(3.7 / DT_MDL)):  # stopped past STANDSTILL_REARM_S, the detector down from 0.5 s to 2.7 s
    sim.v = 0.
    sim.step(available=not 0.5 <= i * DT_MDL < 2.7)
  assert any(p.handback == Handback.detectorStopped for _, _, p in sim.trace)
  assert sim.trace[-1][2].handback == Handback.none
  stop = len(sim.trace)
  sim.cruise = 13.
  for _ in range(int(10. / DT_MDL)):
    sim.step()
  assert sim.s > 300.
  assert not any(p.tooFastToTurn for _, _, p in sim.trace)
  assert rising(sim.trace[stop:], 'desire', Desire.turnLeft)


def test_detector_that_never_starts_hands_back():
  # no pkl, a detector that fails to load, or a driving model with no detector: never a fresh detection. Past
  # DETECTOR_START_DEADLINE_S of modeld running, that is a handback (selfdrived: soft disable while engaged, no engaging)
  sim = Sim([(400., "turn", "left"), (1500., "turn", "right")])
  sim.model_running = False  # modeld loading the detector: not counted
  for _ in range(int(2 * DETECTOR_START_DEADLINE_S / DT_MDL)):
    assert sim.step(available=False).handback == Handback.none
  sim.model_running = True
  plans = [sim.step(available=False) for _ in range(int((DETECTOR_START_DEADLINE_S + 5.) / DT_MDL))]
  first = next(i for i, p in enumerate(plans) if p.handback != Handback.none)
  assert abs((first + 1) * DT_MDL - DETECTOR_START_DEADLINE_S) <= 1.5 * DT_MDL
  assert all(p.handback == Handback.noDetector and p.state == State.unavailable for p in plans[first:])
  assert plans[-1].reason == "detector not running: take control"
  assert all(p.desire == Desire.none and p.laneChange == LaneChange.none for p in plans)
  # it starts after all: acting
  assert sim.step().handback == Handback.none and sim.trace[-1][2].state != State.unavailable


def test_fault_in_a_detector_gap_keeps_the_stop_and_the_handback():
  # stopped short of a pedestrian, the detector goes quiet and the stop is carried. One failed frame in the gap must not
  # drop the carried stop, nor forget that the detector ran (its handback would never come)
  x_ped = 200.
  sim = Sim([(1500., "turn", "right")])
  def pedestrian(sim):
    return ({'cls': 'pedestrian', 'x': x_ped - sim.s, 'y': 0.1, 'vx': 0., 'vy': 0., 'inPath': True},)
  for _ in range(int(40. / DT_MDL)):
    sim.step(pedestrian(sim))
  assert sim.trace[-1][2].shouldStop and sim.v == 0.
  for _ in range(6):
    assert sim.step(available=False).shouldStop
  sim.tp.update = mock.Mock(side_effect=RuntimeError("synthetic failure"))
  assert sim.step(available=False).fault == Fault.error
  del sim.tp.update
  plans = [sim.step(available=False) for _ in range(int(10. / DT_MDL))]
  assert all(p.shouldStop for p in plans) and sim.s < x_ped - PED_STOP_GAP_M + 1.
  assert plans[-1].handback == Handback.detectorStopped
  # failing on every frame: the disabled plans still carry the stopped detector's handback, by its deadline
  sim = Sim([(1500., "turn", "right")])
  sim.run(3.)
  sim.tp.update = mock.Mock(side_effect=RuntimeError("synthetic failure"))
  plans = [sim.step(available=False) for _ in range(int(3. / DT_MDL))]
  assert all(p.fault == Fault.error and p.state == State.disabled for p in plans)
  first = next(i for i, p in enumerate(plans) if p.handback != Handback.none)
  assert abs((first + 1) * DT_MDL - (DETECTOR_DEADLINE_S - STALE_S)) <= 1.5 * DT_MDL
  assert all(p.handback == Handback.detectorStopped for p in plans[first:])
  # with the detector running, a failing frame is only the fault
  assert sim.step().handback == Handback.none


def test_model_fallback_outlives_turnpilotd():
  # TurnPilot turned off and on runs a new turnpilotd, which never saw the big model: modeld's ChestnutActive (False:
  # the big model isn't running this drive) still hands back, with no route cancel (the first process did that)
  sim = Sim([(400., "turn", "left"), (1500., "turn", "right")])
  sim.tp.big_model_failed(enabled=True)
  plans = [sim.step() for _ in range(int(5. / DT_MDL))]
  assert all(p.handback == Handback.modelFellBack and p.desire == Desire.none and p.laneChange == LaneChange.none for p in plans)
  assert not sim.tp.cancel_route
  # in the process that drove the route, the param read first is the fallback as much as modelV2 is: the route is cancelled
  sim = Sim([(400., "turn", "left"), (1500., "turn", "right")])
  sim.big = True
  sim.run(2.)
  sim.tp.big_model_failed(enabled=True)
  assert sim.step().handback == Handback.modelFellBack and sim.tp.cancel_route


def oncoming_at(junction_s: float, after_s: float):
  """An oncoming car that reaches the left at junction_s 8 s after the car is 40 m from it (none before after_s)."""
  spawn = {}
  def objects(sim):
    if sim.s < after_s:
      return ()
    if 'x' not in spawn and junction_s - sim.s < 40.:
      spawn['x'], spawn['t'] = junction_s + 72., sim.t
    if 'x' not in spawn:
      return ()
    return ({'cls': 'car', 'x': spawn['x'] - 9. * (sim.t - spawn['t']) - sim.s, 'y': 5., 'vx': -9.},)
  return objects


def test_fault_keeps_the_next_turns_junction_hold():
  # a left is committed (the car passed its hold line faster than COMMIT_SPEED), then one frame faults. The next left,
  # with an oncoming car, still holds for it: maneuver ids never repeat within a drive, so the first left's commitment
  # (or a gas release) can't be read as the next one's
  for fault in (False, True):
    sim = Sim([(300., "turn", "left"), (700., "turn", "left"), (2500., "turn", "right")])
    oncoming = oncoming_at(700., 400.)
    for _ in range(int(100. / DT_MDL)):
      if fault and 500. < sim.s and not getattr(sim, 'faulted', False):
        sim.faulted = True
        sim.tp.update = mock.Mock(side_effect=RuntimeError("synthetic failure"))
        assert sim.step().fault == Fault.error
        del sim.tp.update
      near = any(abs(sim.s - m) < 8. for m in (300., 700.))
      sim.step(oncoming(sim), steer=90. if near else 0.)
      if sim.s > 730.:
        break
    assert sim.tp.scene.committed_id is not None, fault
    assert any(p.reason == "waiting for oncoming" for s, _, p in sim.trace if s > 400.), fault
    assert any(p.shouldStop for s, v, p in sim.trace if 400. < s < 700. and v < 0.1), fault


def held_for_pedestrian(x_ped=200.):
  sim = Sim([(1500., "turn", "right")])
  def pedestrian(sim):
    return ({'cls': 'pedestrian', 'x': x_ped - sim.s, 'y': 0.1, 'vx': 0., 'vy': 0., 'inPath': True},)
  for _ in range(int(40. / DT_MDL)):
    sim.step(pedestrian(sim))
  assert sim.trace[-1][2].shouldStop and sim.v == 0.
  return sim, pedestrian


def test_persistent_fault_keeps_the_scenes_stops():
  # turnpilotd failing on every frame: the faulted plans still carry the pedestrian stop (live, or carried through a
  # detector gap, which then hands back). The car never drives on through the pedestrian
  for gap in (False, True):
    sim, pedestrian = held_for_pedestrian()
    sim.tp.update = mock.Mock(side_effect=RuntimeError("synthetic failure"))
    plans = [sim.step(() if gap else pedestrian(sim), available=not gap) for _ in range(int(12. / DT_MDL))]
    assert all(p.fault == Fault.error and p.state == State.disabled for p in plans), gap
    assert all(p.shouldStop and p.speedCap < 1. for p in plans), gap
    assert sim.s < 200. - PED_STOP_GAP_M + 1. and sim.v < 0.1, gap
    assert (plans[-1].handback == Handback.detectorStopped) == gap


def test_fault_keeps_a_junction_hold():
  # waiting at the hold line for an oncoming car, turnpilotd faults (once in a detector gap, or on every frame): the
  # route is gone with the fault, but the hold in force stays where it was until the scene says otherwise or the
  # driver presses the gas
  for mode in ('every frame', 'in a gap'):
    sim = Sim([(300., "turn", "left"), (1500., "turn", "right")])
    def car(sim):  # an oncoming car edging up to it
      return ({'cls': 'car', 'x': 305. - sim.s, 'y': 5., 'vx': -1.5},)
    while not (sim.v < 0.05 and sim.s > 250.):
      sim.step(car(sim))
      assert sim.t < 120.
    assert sim.trace[-1][2].reason == "waiting for oncoming" and sim.trace[-1][2].shouldStop
    stopped = sim.s
    if mode == 'every frame':
      sim.tp.update = mock.Mock(side_effect=RuntimeError("synthetic failure"))
      plans = [sim.step(car(sim)) for _ in range(int(5. / DT_MDL))]
    else:
      plans = [sim.step(available=False) for _ in range(4)]
      sim.tp.update = mock.Mock(side_effect=RuntimeError("synthetic failure"))
      plans.append(sim.step(available=False))
      del sim.tp.update
      plans += [sim.step(available=False) for _ in range(int(5. / DT_MDL))]
    assert all(p.shouldStop for p in plans), mode
    assert sim.s < stopped + 0.5, mode
    # the driver looks and goes: the gas releases it
    released = sim.step(car(sim) if mode == 'every frame' else (), available=mode == 'every frame', gas=True)
    assert not released.shouldStop and released.accelCap == NO_ACCEL_CAP, mode
    # the turn cap at most; handed back (the gap outlasted the detector's deadline), no faster than the driver takes it
    # (a committed turn faulting, or handed back: held to the speed the driver's gas takes it to)
    assert released.speedCap > 2. or released.speedCap >= sim.v - 1e-3, mode


def test_detector_not_loaded_while_engaged_isnt_a_detector_that_never_started():
  # TurnPilot turned on while disengaged, and openpilot engaged before modeld read TurnPilotEnabled: modeld loads the
  # detector only while disengaged, so it hasn't tried yet. Engaged frames don't count against its modelV2 start
  # deadline until modeld had its chance; TurnPilot waits (not acting). But it never waits silently: from the first
  # engaged frame with no detection yet, DETECTOR_START_DEADLINE_S on (modeld loading, or not publishing at all: a load
  # that failed while the driver engaged during it) it hands back noDetector, out loud
  for model_running in (True, False):
    sim = Sim([(4000., "turn", "left"), (5000., "turn", "right")])
    for _ in range(int(0.5 / DT_MDL)):
      sim.step(available=False)
    sim.model_running = model_running
    plans = [sim.step(available=False, engaged=True) for _ in range(int(2 * DETECTOR_START_DEADLINE_S / DT_MDL))]
    first = next(i for i, p in enumerate(plans) if p.handback == Handback.noDetector)
    assert abs((first + 1) * DT_MDL - DETECTOR_START_DEADLINE_S) <= 1.5 * DT_MDL, model_running
    assert all(p.handback == Handback.none and p.state == State.unavailable and p.reason == WAITING_REASON
               for p in plans[:first]), model_running
    assert all(p.handback == Handback.noDetector for p in plans[first:]), model_running
  # ...whether or not openpilot stays engaged
  sim = Sim([(4000., "turn", "left"), (5000., "turn", "right")])
  sim.model_running = False
  plans = [sim.step(available=False, engaged=True)] + [sim.step(available=False) for _ in range(int(20. / DT_MDL))]
  first = next(i for i, p in enumerate(plans) if p.handback == Handback.noDetector)
  assert abs((first + 1) * DT_MDL - DETECTOR_START_DEADLINE_S) <= 1.5 * DT_MDL
  # disengaged long enough for modeld to have tried: engaged frames count from then on
  sim = Sim([(4000., "turn", "left"), (5000., "turn", "right")])
  for _ in range(int(3. / DT_MDL)):
    sim.step(available=False)
  plans = [sim.step(available=False, engaged=True) for _ in range(int(DETECTOR_START_DEADLINE_S / DT_MDL))]
  first = next(i for i, p in enumerate(plans) if p.handback == Handback.noDetector)
  assert abs((first + 1) * DT_MDL - (DETECTOR_START_DEADLINE_S - 3.)) <= 1.5 * DT_MDL


def test_handback_never_speeds_the_car_up():
  # the driving model falls back 60 m before a left, its desire not out yet: TurnPilot hands back and the route is
  # cancelled, but through the soft disable (openpilot still steering) the car never speeds up. Once openpilot is
  # disengaged the hold goes with it
  sim = Sim([(300., "turn", "left", junction_of("left")), (1500., "turn", "right", junction_of("right"))], cruise=13.)
  sim.big = True
  while sim.distance() > 60. or sim.s < 10.:
    sim.step()
  assert sim.trace[-1][2].desire == Desire.none and sim.trace[-1][2].speedCap < 12.
  sim.big = False
  sim.nav_filter = lambda sim, instr: (nav_instruction(0., valid=False), False)  # the route cancelled
  start = len(sim.trace)
  for _ in range(int(SOFT_DISABLE_S / DT_MDL)):
    sim.step()
  held = sim.trace[start:]
  assert all(p.handback == Handback.modelFellBack for _, _, p in held)
  assert all(b[1] <= a[1] + 1e-6 for a, b in zip(held, held[1:], strict=False))
  assert sim.step(lat_active=False).speedCap == NO_SPEED_CAP
  assert sim.step().speedCap == NO_SPEED_CAP  # engaged again later (on the small model): no hold left over


def test_fault_keeps_the_commitment_to_the_turn_under_way():
  # past the hold line committed to a left (or released into it by the gas), one frame faults and navd brings the same
  # left back under a new id. Slow in the junction with an oncoming car in range, the car is never held there again
  def oncoming(sim):
    return ({'cls': 'car', 'x': 306. - sim.s, 'y': 5., 'vx': -1.5},)
  for how in ('committed', 'released'):
    sim = Sim([(300., "turn", "left"), (1500., "turn", "right")])
    if how == 'committed':
      while sim.tp.scene.committed_id is None:
        sim.step()
        assert sim.s < 300.
    else:
      while not (sim.v < 0.05 and sim.s > 250.):
        sim.step(oncoming(sim))
        assert sim.t < 120.
      sim.cruise = 1.
      sim.step(oncoming(sim), gas=True)
      assert sim.tp.scene.released_id is not None
      while sim.s < 300. - HOLD_LINE_M + 1.:
        sim.step(oncoming(sim))
    for _ in range(14):  # 20 degrees round the corner
      sim.step(oncoming(sim), yaw_rate=0.5)
    pulses, turn = sim.tp.desires.pulses, sim.tp.follower.maneuver_id
    sim.tp.update = mock.Mock(side_effect=RuntimeError("synthetic failure"))
    assert sim.step(oncoming(sim)).fault == Fault.error
    del sim.tp.update
    sim.cruise = sim.v = 1.
    start = len(sim.trace)
    while sim.s < 305.:
      sim.step(oncoming(sim), yaw_rate=0.05)
    after = [p for _, _, p in sim.trace[start:]]
    assert not any(p.shouldStop or p.reason == "waiting for oncoming" for p in after), how
    # nor is the turn under way dropped: not skipped, its cap kept, its desire not pulsed afresh
    assert not any(p.tooFastToTurn for p in after) and all(p.speedCap < 5. for p in after), how
    assert sim.tp.desires.pulses == pulses and sim.tp.follower.maneuver_id == turn, how


def drive_through_the_left(sim, n):
  """Drive the committed left at s=300 from trace index n, turning from 2 m before it, and check it was held: the same
  maneuver until the junction, its cap kept and the car not sped up into it, no CLEAR until the car is round it, no
  skip. The left is round (80 of its 90 degrees) about 10 m past the junction."""
  turn = sim.tp.follower.maneuver_id
  while sim.s < 325.:
    sim.step(yaw_rate=0.5 if 298. < sim.s < 312. else 0.)
    if sim.s < 300.:
      assert sim.tp.follower.maneuver_id == turn
  assert_turn_held(sim, n)
  v_n = sim.trace[n - 1][1]
  assert max(v for s, v, _ in sim.trace[n:] if s < 305.) <= v_n + 0.05
  assert all(p.desire != CLEAR_DESIRE for s, _, p in sim.trace[n:] if s < 305.)
  assert not any(p.tooFastToTurn for _, _, p in sim.trace)


def test_reroute_onto_the_same_turn_keeps_its_junction_where_it_was():
  # a reroute near m before the committed left (one invalid update, then routeCount 1), whose same left reads off m
  # further on (its origin snapped back) or closer (snapped forward) in every update after: the junction is where it
  # was dead-reckoned, so the cap doesn't lift and the turn isn't read as passed before the car gets there
  for off, near in ((25., 14.), (-20., 14.), (-25., 14.), (-20., 25.), (25., 6.)):
    sim = approach_with_desire_out(to=near + 5.)
    while sim.distance() > near:
      sim.step()
    n = len(sim.trace)
    def snapped(sim, instr, off=off, k=[0], J=300., NEXT=1500.):  # noqa: B006 (k counts the updates of this sim)
      k[0] += 1
      if k[0] == 1:
        return invalid_nav(sim)
      if instr.maneuverDistance > -10.:
        return nav_instruction(J - sim.s + off, "turn", "left", remaining=NEXT - sim.s, junction=junction_of("left"),
                               route_count=1, ahead=[(NEXT - sim.s + off, "turn", "right", junction_of("right"))]), True
      return nav_instruction(NEXT - sim.s + off, "turn", "right", remaining=NEXT - sim.s, junction=junction_of("right"),
                             route_count=1), True
    sim.nav_filter = snapped
    drive_through_the_left(sim, n)


def test_navd_moving_on_early_keeps_the_committed_turn():
  # navd moves on to the next step (other bearings) at m before the committed left: a sharp turn's next step, or GPS
  # noise forward. The left isn't passed until the car has turned it or is dead-reckoned past it: its id and cap stay,
  # and no CLEAR goes before the car is round it
  for at, nxt in ((10., 1200.), (6., 1200.), (15., 1200.), (20., 1200.), (8., 60.)):
    sim = approach_with_desire_out(to=at + 5.)
    sim.navd.maneuvers[1] = (300. + nxt, "turn", "right", junction_of("right", 2))
    n = len(sim.trace)
    def early(sim, instr, at=at):
      if sim.s >= 300. - at:
        pos, mtype, modifier, junction = sim.navd.maneuvers[1]
        return nav_instruction(pos - sim.s, mtype, modifier, remaining=pos - sim.s, junction=junction), True
      return instr, True
    sim.nav_filter = early
    drive_through_the_left(sim, n)
    # once the left is over, the next step is taken up
    while sim.s < 340.:
      sim.step()
    assert sim.tp.follower.maneuver.side == -1, (at, nxt)
