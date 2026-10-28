"""turnpilotd closed loop: synthetic navd, model and scene around a point-mass car that follows the plan's limits."""
from unittest import mock

from opendbc.car.structs import car
from openpilot.cereal import custom, log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.hooks import ABORT_HOLD_S, CLEAR_DESIRE, lane_change_events, plan_accel
from openpilot.selfdrive.turnpilot.lane_planner import LC_WARN_S, lane_change_window
from openpilot.selfdrive.turnpilot.scene.lane_paint import UNKNOWN, LinePaint
from openpilot.selfdrive.turnpilot.scene_policy import HOLD_LINE_M, NO_ACCEL_CAP, ONCOMING_TTA_S, PED_STOP_GAP_M
from openpilot.selfdrive.turnpilot.tests.synthetic import FakeNavd, car_control, car_state, model_v2, nav_instruction, scene
from openpilot.selfdrive.turnpilot.desire_timing import TURN_DESIRE_LEAD_M
from openpilot.selfdrive.turnpilot.turn_speed import NO_SPEED_CAP, V_TURN_MIN
from openpilot.selfdrive.turnpilot.turnpilotd import (CAR_UNKNOWN_REASON, DETECTOR_DEADLINE_S, NO_LONG_REASON, WAITING_REASON,
                                                       TurnPilot, long_control_for)
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
    self.trace = []

  def step(self, objects=(), available=True, steer=0., blinker=False, lat_active=True, yaw_rate=0.):
    self.sm.updated = dict.fromkeys(self.sm.data, False)
    instr = self.navd.update(self.s)
    if instr is not None:
      self.sm.put('navInstruction', instr)
    self.sm.put('carState', car_state(self.v, steer, left_blinker=blinker, yaw_rate=yaw_rate))
    self.sm.put('carControl', car_control(lat_active))
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
  sim = Sim([(200., "turn", "left"), (1500., "turn", "right")])
  for _ in range(int(20. / DT_MDL)):
    plan = sim.step(available=False)
    assert plan.state == State.unavailable and plan.reason == WAITING_REASON and plan.handback == Handback.none
    assert plan.desire == Desire.none and plan.laneChange == LaneChange.none and plan.speedCap == NO_SPEED_CAP
  assert sim.s > 220.  # drove past the turn, nothing held it
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
  # model there are no detections, so TurnPilot waits for them
  sim = Sim([(400., "turn", "left")])
  for _ in range(int(5. / DT_MDL)):
    plan = sim.step(available=False)
  assert plan.handback == Handback.none and plan.reason == WAITING_REASON and not sim.tp.cancel_route


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
    # selfdrived raises the abort from the disabled plan
    assert lane_change_events(plans[0]) == ('turnpilotLaneChangeAborted',), stop
