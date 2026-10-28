import openpilot.cereal.messaging as messaging
from openpilot.cereal import custom, log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.turnpilot.hooks import ABORT_HOLD_S
from openpilot.selfdrive.turnpilot.lane_planner import (LC_ABORT_COOLDOWN_S, LC_CONFIRM_FRAMES, LC_REQUEST_TIMEOUT_S,
                                                        LC_SIGNAL_TIMEOUT_S, LC_WARN_S, YELLOW_HOLD_S, LanePlanner, ModelLanes, Phase,
                                                        YellowHold, change_blocker, desired_changes, lane_change_window,
                                                        lane_changes_needed, model_lanes, model_lanes_from_msg, stopped_in_lane,
                                                        target_lane_blocked)
from openpilot.selfdrive.turnpilot.route_follower import LEFT, NOMINAL_ANGLE_DEG, RIGHT, Maneuver, classify
from openpilot.selfdrive.turnpilot.scene.lane_paint import UNKNOWN, LinePaint
from openpilot.selfdrive.turnpilot.tests.synthetic import car_state, model_v2, road, scene

LaneChange = custom.TurnpilotPlan.LaneChange
THREE_LANES_LEFT_TURN = ((("left",), True), (("straight",), False), (("straight", "right"), False))
TWO_LANES_RIGHT_TURN = ((("straight",), False), (("right",), True))
BW, SW = LinePaint('broken', 'white', 0.9, 0.3), LinePaint('solid', 'white', 0.9, 0.95)
SY, BY = LinePaint('solid', 'yellow', 0.9, 0.95), LinePaint('broken', 'yellow', 0.5, 0.3)
NO_LINE = LinePaint('none', 'unknown', 1., 0.)


def paint(left=UNKNOWN, right=UNKNOWN, far_left=UNKNOWN, far_right=UNKNOWN):
  return far_left, left, right, far_right


def counts(lanes):
  return lanes.left, lanes.right


def maneuver(mtype, modifier, lanes=()):
  return Maneuver(*classify(mtype, modifier), mtype, modifier, NOMINAL_ANGLE_DEG.get(modifier, 0.), lanes)


def test_model_lanes_counts_room_over_lane_width():
  assert model_lanes(*road(2, 0)) == ModelLanes(2, 0, True, False)
  assert counts(model_lanes(*road(1, 1))) == (1, 1)
  # a wide shoulder with no painted line beyond it is not a lane, but leaves room to keep toward
  lanes = model_lanes(*road(0, 0, shoulder=3.))
  assert (lanes.left, lanes.right, lanes.keep_left_ok, lanes.keep_right_ok) == (0, 0, True, True)
  # an unsure road edge: paint alone says one lane
  assert counts(model_lanes(*road(2, 0, edge_std=2.))) == (1, 0)
  # no ego line on a side: no keep that way
  line_ys, probs, edges, stds = road(1, 1)
  assert not model_lanes(line_ys, [probs[0], 0.1, probs[2], probs[3]], edges, stds).keep_left_ok


def test_model_lanes_from_msg():
  assert counts(model_lanes_from_msg(model_v2(2, 1))) == (2, 1)
  assert model_lanes_from_msg(messaging.new_message('modelV2').modelV2) == ModelLanes()  # nothing from modeld yet


UNSURE_BW = LinePaint('broken', 'white', 0.4, 0.3)
COLORLESS = LinePaint('broken', 'unknown', 0.9, 0.3)
NOT_BW = (SW, SY, BY, NO_LINE, UNKNOWN, UNSURE_BW, COLORLESS, LinePaint('unknown', 'white', 0.9, 0.3))
LaneChangeState = log.LaneChangeState


def striped(lanes_left, lanes_right):
  """A multi-lane road: broken white between the travel lanes, solid white at the edges."""
  return paint(left=BW if lanes_left else SW, right=BW if lanes_right else SW)


def test_desired_changes():
  left = maneuver("turn", "left", THREE_LANES_LEFT_TURN)
  assert desired_changes(left, model_lanes(*road(2, 0))) == 2
  assert desired_changes(left, model_lanes(*road(1, 1))) == 1
  assert desired_changes(left, model_lanes(*road(0, 2))) == 0
  assert desired_changes(maneuver("turn", "right", TWO_LANES_RIGHT_TURN), model_lanes(*road(0, 1))) == -1
  # no map lanes: one toward a turn's side at a time; an exit to the curb by the model's count, not across solid white
  assert desired_changes(maneuver("turn", "left"), model_lanes(*road(2, 0))) == 1
  assert desired_changes(maneuver("turn", "right"), model_lanes(*road(0, 2))) == -1
  assert desired_changes(maneuver("off ramp", "slight right"), model_lanes(*road(0, 2))) == -2
  assert desired_changes(maneuver("off ramp", "slight right"), model_lanes(*road(0, 2)), paint=paint(right=SW)) == 0
  assert desired_changes(maneuver("off ramp", "slight right"), model_lanes(*road(0, 2)), paint=paint(right=BW, far_right=SW)) == -1
  assert desired_changes(maneuver("fork", "slight right"), model_lanes(*road(0, 1))) == 0
  assert desired_changes(None, model_lanes(*road(1, 1))) == 0


def test_every_change_needs_broken_white_into_a_travel_lane():
  lanes = model_lanes(*road(1, 1))
  assert change_blocker(LEFT, lanes, paint(left=BW)) == '' and change_blocker(RIGHT, lanes, paint(right=BW)) == ''
  for p in NOT_BW:  # solid, yellow, no line, unknown, unsure, colorless: no change
    assert change_blocker(LEFT, lanes, paint(left=p)) in ('lineNotBroken', 'noLane'), p
  assert change_blocker(LEFT, lanes, ()) == 'lineNotBroken'  # no paint read at all (no detector, wide camera, stale scene)
  # the paint the other way doesn't count: no left/right mixup
  assert change_blocker(RIGHT, lanes, paint(left=BW, right=SW)) == 'lineNotBroken'
  assert change_blocker(LEFT, lanes, paint(left=SW, right=BW)) == 'lineNotBroken'
  # broken white, but the model sees no lane that way (a bike lane's line turns broken before a junction)
  assert change_blocker(RIGHT, model_lanes(*road(1, 0)), paint(right=BW)) == 'noLane'
  # a worn solid edge line read broken white beside a shoulder or a bike lane: the model sees no lane there
  assert change_blocker(RIGHT, model_lanes(*road(0, 0, shoulder=2.)), paint(right=BW)) == 'noLane'
  line_ys, probs, edges, stds = road(0, 1)
  line_ys[3] = line_ys[2] + 0.7 * 3.6  # the next line out is a bike lane's width away, not a lane's
  assert change_blocker(RIGHT, model_lanes(line_ys, probs, edges, stds), paint(right=BW)) == 'noLane'
  # parked cars ahead in that lane: a parking strip
  parked = scene([{'cls': 'car', 'x': 30., 'y': -3.6, 'vx': 0.}]).objects
  assert change_blocker(RIGHT, lanes, paint(right=BW), parked) == 'laneParked'
  assert change_blocker(LEFT, lanes, paint(left=BW), parked) == ''
  # left-hand traffic: never away from the curb (the center line is broken white there)
  assert change_blocker(RIGHT, lanes, paint(right=BW), right_hand_traffic=False) == 'noLane'
  assert change_blocker(LEFT, lanes, paint(left=BW), right_hand_traffic=False) == ''


def test_lanes_needed_every_path_gated():
  left = maneuver("turn", "left", THREE_LANES_LEFT_TURN)
  # the map's lanes say two over, but only across broken white
  assert lane_changes_needed(left, model_lanes(*road(2, 0)), paint=paint(left=BW)) == 2
  for p in NOT_BW + ((),):
    pt = paint(left=p) if p != () else ()
    assert lane_changes_needed(left, model_lanes(*road(2, 0)), paint=pt) == 0, p
  # an exit by the model's count: also only across broken white (it used to go on unknown paint)
  ramp = maneuver("off ramp", "slight right")
  assert lane_changes_needed(ramp, model_lanes(*road(0, 2)), paint=paint(right=BW)) == -2
  assert lane_changes_needed(ramp, model_lanes(*road(0, 2)), paint=()) == 0
  assert lane_changes_needed(ramp, model_lanes(*road(0, 2)), paint=paint()) == 0
  # turns without the map's lanes, either side; left-hand traffic only toward its curb
  assert lane_changes_needed(maneuver("turn", "left"), model_lanes(*road(2, 0)), paint=paint(left=BW)) == 1
  assert lane_changes_needed(maneuver("turn", "right"), model_lanes(*road(0, 2)), paint=paint(right=BW)) == -1
  assert lane_changes_needed(maneuver("turn", "left"), model_lanes(*road(1, 0)), False, paint=paint(left=BW)) == 1
  assert lane_changes_needed(maneuver("turn", "right"), model_lanes(*road(0, 1)), False, paint=paint(right=BW)) == 0


def test_yellow_veto_and_lane_count():
  left = maneuver("turn", "left", THREE_LANES_LEFT_TURN)
  assert lane_changes_needed(left, model_lanes(*road(2, 0)), paint=paint(left=SY)) == 0
  assert lane_changes_needed(left, model_lanes(*road(2, 0)), paint=paint(left=BY)) == 0
  # lanes beyond a yellow line don't count toward where the car is: two lanes seen to the right, the far one beyond yellow,
  # puts the car in the middle of the map's three lanes, not the left one
  assert desired_changes(left, model_lanes(*road(1, 2)), paint=paint(far_right=SY)) == 1
  # broken white at the ego line, yellow at the far one: still one lane over
  assert lane_changes_needed(maneuver("turn", "left"), model_lanes(*road(2, 0)), paint=paint(left=BW, far_left=SY)) == 1


def test_yellow_hold():
  hold = YellowHold()
  out = hold.update(paint(left=SY), 15., 0.05, False)
  assert out[1] == SY
  # worn for a stretch: the line reads unknown, but stays yellow for the hold
  for _ in range(int((YELLOW_HOLD_S - 0.5) / 0.05)):
    out = hold.update(paint(left=UNKNOWN), 30., 0.05, False)
  assert out[1].color == 'yellow' and change_blocker(LEFT, model_lanes(*road(1, 1)), out) == 'lineNotBroken'
  # and when nothing is read at all
  assert hold.update((), 30., 0.05, False)[1].color == 'yellow'
  for _ in range(40):
    out = hold.update(paint(left=BW), 30., 0.05, False)
  assert out[1] == BW  # held long and far enough
  # a change moves the holds with the lines; a change of unknown direction holds all four
  hold = YellowHold()
  hold.update(paint(far_left=SY), 15., 0.05, False)
  hold.shift(LEFT)
  assert hold.update(paint(), 15., 0.05, False)[1].color == 'yellow'
  hold.shift(RIGHT)
  assert hold.update(paint(), 15., 0.05, False)[0].color == 'yellow'
  hold.shift(0)
  assert all(p.color == 'yellow' for p in hold.update(paint(), 15., 0.05, False))
  hold = YellowHold()
  hold.shift(0)  # nothing held, nothing to spread
  assert hold.update(paint(), 15., 0.05, False) == paint()


def test_target_lane_blocked_and_stopped_sides():
  sc = scene([{'cls': 'car', 'x': 5., 'y': 3.6, 'vx': 15.}])
  assert target_lane_blocked(sc.objects, LEFT, 15.)
  assert not target_lane_blocked(sc.objects, RIGHT, 15.)
  # well ahead and pulling away vs closing fast from behind
  assert not target_lane_blocked(scene([{'cls': 'car', 'x': 40., 'y': 3.6, 'vx': 20.}]).objects, LEFT, 15.)
  assert target_lane_blocked(scene([{'cls': 'car', 'x': -40., 'y': 3.6, 'vx': 25.}]).objects, LEFT, 15.)
  assert not target_lane_blocked(scene([{'cls': 'car', 'x': -40., 'y': 3.6, 'vx': 16.}]).objects, LEFT, 15.)
  # two lanes over doesn't count
  assert not target_lane_blocked(scene([{'cls': 'car', 'x': 0., 'y': 7.5, 'vx': 15.}]).objects, LEFT, 15.)
  # TrackedObject y is left: a parked car on the left is LEFT's, never RIGHT's
  parked_left = scene([{'cls': 'car', 'x': 20., 'y': 3.6, 'vx': 0.}]).objects
  assert stopped_in_lane(parked_left, LEFT) and not stopped_in_lane(parked_left, RIGHT)


class Drive:
  """A car on a road of lanes, with the real DesireHelper executing nav lane change requests."""
  def __init__(self, lanes_left, lanes_right, v=15., driver_signals=False, paint=striped, paint_rule=True):
    """paint: (lanes left, lanes right) -> the four lines' paint where the car is now, or None for none read."""
    self.lanes_left, self.lanes_right, self.v = lanes_left, lanes_right, v
    self.paint = paint
    self.lp, self.dh = LanePlanner(paint_rule), DesireHelper()
    self.driver_signals = driver_signals
    self.lc_t = self.t = 0.
    self.requests, self.prompts, self.offers, self.aborts, self.decisions = [], [], [], [], []

  def step(self, m, distance, objects=(), blinker=None, blindspot=None, maneuver_id=0, inputs_valid=True):
    lanes = model_lanes(*road(self.lanes_left, self.lanes_right))
    left_blinker, right_blinker = blinker == LaneChange.left, blinker == LaneChange.right
    d = self.lp.update(m, distance, lanes, self.dh.lane_change_state, self.v, left_blinker, right_blinker, objects,
                       self.driver_signals, True, DT_MDL, self.paint(self.lanes_left, self.lanes_right) if self.paint else (),
                       blindspot == LaneChange.left, blindspot == LaneChange.right, maneuver_id, inputs_valid)
    prev = self.decisions[-1] if self.decisions else None
    if d.lane_change != LaneChange.none and (prev is None or prev.lane_change == LaneChange.none):
      self.requests.append((d.lane_change, distance, self.t))
    if d.offer != LaneChange.none and (prev is None or prev.offer == LaneChange.none):
      self.offers.append((d.offer, distance, self.t))
    if d.abort != 'none' and (prev is None or prev.abort == 'none'):
      self.aborts.append((d.abort, self.t))
    self.decisions.append(d)
    self.prompts.append(d.prompt)
    # the model takes 3 s to change lanes once DesireHelper asks it to
    starting = self.dh.lane_change_state == LaneChangeState.laneChangeStarting
    self.lc_t = self.lc_t + DT_MDL if starting else 0.
    prob = 1. if starting and self.lc_t < 3. else 0.
    if starting and self.lc_t >= 3. and self.lc_t - DT_MDL < 3.:
      k = 1 if self.dh.lane_change_direction == log.LaneChangeDirection.left else -1
      self.lanes_left, self.lanes_right = self.lanes_left - k, self.lanes_right + k
    cs = car_state(self.v, left_blinker=left_blinker, right_blinker=right_blinker)
    self.dh.update(cs, True, prob, d.lane_change)
    self.t += DT_MDL
    return d

  def run(self, m, distance, seconds, moving=True, **kw):
    for _ in range(int(round(seconds / DT_MDL))):
      self.step(m, distance, **kw)
      distance -= self.v * DT_MDL if moving else 0.
    return distance


def test_left_turn_on_three_lane_road_warns_then_changes_twice():
  drive = Drive(lanes_left=2, lanes_right=0)
  m = maneuver("turn", "left", THREE_LANES_LEFT_TURN)
  distance = drive.run(m, 500., 40.)
  assert [r for r, _, _ in drive.requests] == [LaneChange.left] * 2 and drive.lp.changes == 2
  assert (drive.lanes_left, drive.lanes_right) == (0, 2)
  # each request comes the full warning after its offer, inside the window
  assert len(drive.offers) == 2
  for (_, _, t_offer), (_, _, t_req) in zip(drive.offers, drive.requests, strict=True):
    assert abs((t_req - t_offer) - LC_WARN_S) < 2 * DT_MDL
  start, end = lane_change_window(drive.v, 2)
  assert all(end < d < start for _, d, _ in drive.offers)
  assert drive.offers[0][1] > lane_change_window(drive.v)[0]  # two changes to make: the first starts earlier
  # the countdown runs down during the warning
  counting = [d.countdown for d in drive.decisions if d.offer != LaneChange.none and d.lane_change == LaneChange.none]
  assert counting[0] > LC_WARN_S - 0.1 and min(counting) < 0.1
  assert not drive.aborts and distance < 100.


def test_never_without_broken_white():
  m = maneuver("turn", "left", THREE_LANES_LEFT_TURN)
  for p in NOT_BW:
    drive = Drive(lanes_left=2, lanes_right=0, paint=lambda left, right, p=p: paint(left=p, right=SW))
    drive.run(m, 500., 40.)
    assert not drive.offers and not drive.requests, p
  drive = Drive(lanes_left=2, lanes_right=0, paint=None)  # the detector's paint never arrives
  drive.run(m, 500., 40.)
  assert not drive.offers
  drive = Drive(lanes_left=0, lanes_right=2, paint=None)
  drive.run(maneuver("off ramp", "slight right"), 500., 40.)
  assert not drive.offers  # an exit by the model's lane count no longer goes on unknown paint


def test_turned_off_plans_nothing():
  drive = Drive(lanes_left=2, lanes_right=0, paint_rule=False)
  drive.run(maneuver("turn", "left", THREE_LANES_LEFT_TURN), 500., 40.)
  assert not drive.offers and not drive.requests


def test_tesla_driver_confirms_with_the_stalk():
  drive = Drive(lanes_left=0, lanes_right=1, driver_signals=True)
  m = maneuver("turn", "right", TWO_LANES_RIGHT_TURN)
  distance = drive.run(m, 300., 2.)
  assert drive.offers and drive.prompts[-1] == LaneChange.right and not drive.requests
  # the countdown ends: still nothing until the driver signals
  distance = drive.run(m, distance, LC_WARN_S)
  d = drive.decisions[-1]
  assert d.offer == LaneChange.right and d.countdown == 0. and d.lane_change == LaneChange.none and drive.lp.phase == Phase.WAIT
  drive.run(m, distance, 5., blinker=LaneChange.right)
  assert [r for r, _, _ in drive.requests] == [LaneChange.right] and drive.lp.changes == 1 and not drive.aborts


def test_tesla_stalk_during_the_countdown_is_the_go_ahead():
  drive = Drive(lanes_left=0, lanes_right=1, driver_signals=True)
  m = maneuver("turn", "right", TWO_LANES_RIGHT_TURN)
  distance = drive.run(m, 300., 1.5)
  assert drive.lp.phase == Phase.WARN
  drive.run(m, distance, 5., blinker=LaneChange.right)
  assert drive.requests and drive.requests[0][2] - drive.offers[0][2] < LC_WARN_S and drive.lp.changes == 1


def test_tesla_no_signal_aborts_and_offers_again_later():
  drive = Drive(lanes_left=0, lanes_right=1, driver_signals=True)
  m = maneuver("turn", "right", TWO_LANES_RIGHT_TURN)
  distance = drive.run(m, 250., LC_CONFIRM_FRAMES * DT_MDL + LC_WARN_S + LC_SIGNAL_TIMEOUT_S + 0.5, moving=False)
  assert [a for a, _ in drive.aborts] == ['noSignal'] and not drive.requests
  t_abort = drive.aborts[0][1]
  drive.run(m, distance, LC_ABORT_COOLDOWN_S + 1., moving=False)
  held = [d.abort for d, t in zip(drive.decisions, np_times(drive), strict=True) if t >= t_abort - 1e-6]
  n = int(round(ABORT_HOLD_S / DT_MDL))
  assert held[:n] == ['noSignal'] * n and held[n + 1] == 'none'  # held for the alert, then gone
  assert len(drive.offers) == 2 and drive.offers[1][2] - t_abort >= LC_ABORT_COOLDOWN_S


def np_times(drive):
  return [k * DT_MDL for k in range(len(drive.decisions))]


def test_tesla_blinker_off_before_the_change_starts_aborts():
  drive = Drive(lanes_left=0, lanes_right=1, driver_signals=True)
  m = maneuver("turn", "right", TWO_LANES_RIGHT_TURN)
  distance = drive.run(m, 290., 1.5)
  # the driver flicks the stalk; the blinker runs out before DesireHelper starts (as when the blind spot holds it)
  drive.dh.lane_change_state = LaneChangeState.preLaneChange
  drive.lp.update(m, distance, model_lanes(*road(0, 1)), LaneChangeState.preLaneChange, 15., False, True, (), True, True, DT_MDL,
                  paint(right=BW), maneuver_id=0)
  assert drive.lp.phase == Phase.REQUEST
  drive.dh.lane_change_state = LaneChangeState.preLaneChange
  drive.step(m, distance)
  assert drive.aborts[-1][0] == 'noSignal' and drive.decisions[-1].lane_change == LaneChange.none


def test_aborts_before_the_change_starts():
  m = maneuver("turn", "right", TWO_LANES_RIGHT_TURN)

  def offered(**kw):
    drive = Drive(lanes_left=0, lanes_right=1, **kw)
    distance = drive.run(m, 290., 1.5)
    assert drive.lp.phase == Phase.WARN
    return drive, distance

  # the line turns solid, unknown, or stops being read
  for p in (SW, UNKNOWN, None):
    drive, distance = offered()
    drive.paint = (lambda left, right, p=p: paint(right=p)) if p is not None else None
    drive.run(m, distance, 1.)
    assert [a for a, _ in drive.aborts] == ['lineNotBroken'] and not drive.requests, p
  # the blind spot fills
  drive, distance = offered()
  drive.run(m, distance, 1., blindspot=LaneChange.right)
  assert [a for a, _ in drive.aborts] == ['blindspot'] and not drive.requests
  # a car in the target lane
  drive, distance = offered()
  drive.run(m, distance, 1., objects=scene([{'cls': 'car', 'x': 2., 'y': -3.6, 'vx': 15.}]).objects)
  assert [a for a, _ in drive.aborts] == ['carInLane']
  # parked cars come into view in that lane
  drive, distance = offered()
  drive.run(m, distance, 1., objects=scene([{'cls': 'car', 'x': 50., 'y': -3.6, 'vx': 0.}]).objects)
  assert [a for a, _ in drive.aborts] == ['laneParked']
  # the route changes, or goes
  drive, distance = offered()
  drive.run(m, distance, 1., maneuver_id=1)
  assert [a for a, _ in drive.aborts] == ['routeChanged']
  drive, distance = offered()
  drive.run(None, distance, 1.)
  assert [a for a, _ in drive.aborts] == ['routeChanged']
  # the car slows below DesireHelper's speed
  drive, distance = offered()
  drive.v = 7.
  drive.run(m, distance, 1.)
  assert [a for a, _ in drive.aborts] == ['tooSlow']
  # inputs go invalid, the planner is turned off
  drive, distance = offered()
  drive.run(m, distance, 1., inputs_valid=False)
  assert [a for a, _ in drive.aborts] == ['badInputs']
  drive, distance = offered()
  drive.lp.paint_rule = False
  drive.run(m, distance, 1.)
  assert [a for a, _ in drive.aborts] == ['turnedOff']
  # already in a lane the turn leaves from (the model's count changed under the offer)
  two_right = maneuver("turn", "right", ((("straight",), False), (("straight", "right"), True), (("right",), True)))
  drive = Drive(lanes_left=0, lanes_right=2)
  distance = drive.run(two_right, 290., 1.5)
  assert drive.lp.phase == Phase.WARN and drive.lp.offer == LaneChange.right
  drive.lanes_left, drive.lanes_right = 1, 1
  drive.run(two_right, distance, 1.)
  assert [a for a, _ in drive.aborts] == ['notNeeded']


def test_too_close_to_the_turn_aborts():
  drive = Drive(lanes_left=0, lanes_right=1, driver_signals=True)
  m = maneuver("turn", "right", TWO_LANES_RIGHT_TURN)
  _, end = lane_change_window(drive.v)
  drive.run(m, end + 3.5 * drive.v, 6.)  # offered just before the end of the window, never signaled
  assert drive.offers and [a for a, _ in drive.aborts] == ['tooLate']


def test_desire_helper_waiting_too_long_aborts():
  lp = LanePlanner()
  m = maneuver("turn", "right", TWO_LANES_RIGHT_TURN)
  lanes = model_lanes(*road(0, 1))
  args = {'lanes': lanes, 'v_ego': 15., 'left_blinker': False, 'right_blinker': False, 'objects': (), 'driver_signals': False,
          'right_hand_traffic': True, 'dt': DT_MDL, 'paint': paint(right=BW)}
  distance, d = 290., None
  for _ in range(int((LC_CONFIRM_FRAMES * DT_MDL + LC_WARN_S + LC_REQUEST_TIMEOUT_S + 1.) / DT_MDL)):
    d = lp.update(m, distance, lc_state=LaneChangeState.preLaneChange, **args)
    if d.abort != 'none':
      break
    distance -= 15. * DT_MDL
  assert d.abort == 'timedOut' and d.lane_change == LaneChange.none


def test_never_cancelled_once_started():
  drive = Drive(lanes_left=0, lanes_right=1)
  m = maneuver("turn", "right", TWO_LANES_RIGHT_TURN)
  distance = drive.run(m, 290., LC_CONFIRM_FRAMES * DT_MDL + LC_WARN_S + 0.3)
  assert drive.dh.lane_change_state == LaneChangeState.laneChangeStarting
  # mid-change everything goes wrong: the paint, the blind spot, the route
  drive.paint = lambda left, right: paint(right=SY)
  drive.run(None, distance, 1., blindspot=LaneChange.right, maneuver_id=7)
  assert not drive.aborts and drive.dh.lane_change_state in (LaneChangeState.laneChangeStarting, LaneChangeState.laneChangeFinishing)


def test_blind_spot_or_car_at_the_offer_holds_it_back():
  drive = Drive(lanes_left=0, lanes_right=1)
  m = maneuver("turn", "right", TWO_LANES_RIGHT_TURN)
  distance = drive.run(m, 300., 3., blindspot=LaneChange.right)
  assert not drive.offers and drive.decisions[-1].reason == "car in blind spot"
  beside = scene([{'cls': 'car', 'x': -3., 'y': -3.6, 'vx': 15.}]).objects
  distance = drive.run(m, distance, 2., objects=beside)
  assert not drive.offers and drive.decisions[-1].reason == "car in target lane"
  drive.run(m, distance, 6.)
  assert [r for r, _, _ in drive.requests] == [LaneChange.right]


def test_driver_lane_change_gets_a_cooldown_too():
  lp = LanePlanner()
  m = maneuver("turn", "left")
  lanes = model_lanes(*road(1, 1))
  kw = {'v_ego': 15., 'objects': (), 'driver_signals': True, 'right_hand_traffic': True, 'dt': DT_MDL, 'paint': paint(left=BW, right=BW)}
  lp.update(m, 300., lanes, LaneChangeState.laneChangeStarting, left_blinker=False, right_blinker=False, **kw)  # direction unknown
  d = None
  for k in range(LC_CONFIRM_FRAMES + 2):
    d = lp.update(m, 300. - k, lanes, LaneChangeState.off, left_blinker=False, right_blinker=False, **kw)
  assert d.offer == LaneChange.none  # still cooling down: the lines may have renumbered


def test_no_request_below_lane_change_speed():
  drive = Drive(lanes_left=0, lanes_right=2, v=7.)
  drive.run(maneuver("turn", "right", TWO_LANES_RIGHT_TURN), 200., 10.)
  assert not drive.offers and not drive.requests


def three_lanes_and_a_bike_lane(lanes_left, lanes_right):
  """Double yellow on the left, broken white between the three lanes, a solid white bike lane line on the right."""
  return paint(left=BW if lanes_left else SY, right=BW if lanes_right else SW)


def test_turns_move_over_while_the_line_is_broken_white():
  # left turn, no map lanes: over twice, then the line on the left is yellow
  drive = Drive(lanes_left=2, lanes_right=0, paint=three_lanes_and_a_bike_lane)
  drive.run(maneuver("turn", "left"), 500., 40.)
  assert [r for r, _, _ in drive.requests] == [LaneChange.left] * 2 and (drive.lanes_left, drive.lanes_right) == (0, 2)
  # right turn: over to the right lane, never into the bike lane
  drive = Drive(lanes_left=0, lanes_right=2, paint=three_lanes_and_a_bike_lane)
  drive.run(maneuver("turn", "right"), 500., 40.)
  assert [r for r, _, _ in drive.requests] == [LaneChange.right] * 2 and (drive.lanes_left, drive.lanes_right) == (2, 0)


def test_yellow_line_stops_map_lane_changes():
  # the map says the leftmost of three lanes, but the model's two lanes on the left are beyond a yellow line
  drive = Drive(lanes_left=2, lanes_right=0, paint=lambda left, right: paint(left=SY, right=SW))
  drive.run(maneuver("turn", "left", THREE_LANES_LEFT_TURN), 500., 30.)
  assert not drive.offers
