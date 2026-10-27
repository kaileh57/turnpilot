import openpilot.cereal.messaging as messaging
from openpilot.cereal import custom, log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.turnpilot.lane_planner import (LC_BLOCK_X_M, LanePlanner, ModelLanes, lane_change_window, lane_changes_needed,
                                                        model_lanes, model_lanes_from_msg, target_lane_blocked)
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


def test_lanes_needed():
  left = maneuver("turn", "left", THREE_LANES_LEFT_TURN)
  assert lane_changes_needed(left, model_lanes(*road(2, 0))) == 2
  assert lane_changes_needed(left, model_lanes(*road(1, 1))) == 1
  assert lane_changes_needed(left, model_lanes(*road(0, 2))) == 0
  # without the map's lanes, never toward oncoming traffic, and toward the curb only for an exit (a junction's curb
  # strip is often parking)
  assert lane_changes_needed(maneuver("turn", "left"), model_lanes(*road(2, 0))) == 0
  assert lane_changes_needed(maneuver("turn", "right"), model_lanes(*road(0, 2))) == 0
  assert lane_changes_needed(maneuver("turn", "right", TWO_LANES_RIGHT_TURN), model_lanes(*road(0, 1))) == -1
  assert lane_changes_needed(maneuver("off ramp", "slight right"), model_lanes(*road(0, 1))) == -1
  assert lane_changes_needed(maneuver("fork", "slight right"), model_lanes(*road(0, 1))) == 0
  # the map says left, but the model sees no lane there
  assert lane_changes_needed(left, model_lanes(*road(0, 0))) == 0


def test_lanes_needed_from_paint():
  # without the map's lanes: toward a turn while the ego line that way reads broken white, one change at a time
  assert lane_changes_needed(maneuver("turn", "left"), model_lanes(*road(2, 0)), paint=paint(left=BW)) == 1
  assert lane_changes_needed(maneuver("turn", "right"), model_lanes(*road(0, 2)), paint=paint(right=BW)) == -1
  assert lane_changes_needed(maneuver("turn", "left"), model_lanes(*road(1, 0)), False, paint=paint(left=BW)) == 1  # left-hand traffic
  # solid, yellow, no line, unknown or unsure that way: no change, as before the paint was read
  for p in (SW, SY, BY, NO_LINE, UNKNOWN, LinePaint('broken', 'white', 0.4, 0.3), LinePaint('broken', 'unknown', 0.9, 0.3)):
    assert lane_changes_needed(maneuver("turn", "right"), model_lanes(*road(0, 2)), paint=paint(right=p)) == 0, p
  # the paint the other way doesn't count, and the model must see a lane there too (a bike lane's line turns broken
  # before a junction)
  assert lane_changes_needed(maneuver("turn", "right"), model_lanes(*road(1, 0)), paint=paint(left=BW)) == 0
  assert lane_changes_needed(maneuver("turn", "right"), model_lanes(*road(0, 0)), paint=paint(right=BW)) == 0
  # an exit still counts the model's lanes to the curb, whatever the paint, unless the line is yellow
  assert lane_changes_needed(maneuver("off ramp", "slight right"), model_lanes(*road(0, 2)), paint=paint(right=BW)) == -2
  assert lane_changes_needed(maneuver("off ramp", "slight right"), model_lanes(*road(0, 2)), paint=paint(right=SW)) == -2
  assert lane_changes_needed(maneuver("off ramp", "slight right"), model_lanes(*road(0, 2)), paint=paint(right=SY)) == 0
  # no paint at all (old logs, no detector): exactly the old answers
  assert lane_changes_needed(maneuver("turn", "left"), model_lanes(*road(2, 0)), paint=()) == 0
  assert lane_changes_needed(maneuver("off ramp", "slight right"), model_lanes(*road(0, 1)), paint=()) == -1


def test_yellow_veto_and_lane_count():
  left = maneuver("turn", "left", THREE_LANES_LEFT_TURN)
  # the map wants two lanes over to the left, but the line there is yellow: oncoming traffic
  assert lane_changes_needed(left, model_lanes(*road(2, 0)), paint=paint(left=SY)) == 0
  assert lane_changes_needed(left, model_lanes(*road(2, 0)), paint=paint(left=BY)) == 0
  assert lane_changes_needed(left, model_lanes(*road(2, 0)), paint=paint(left=LinePaint('solid', 'yellow', 0.2, 0.9))) == 2  # unsure
  # lanes beyond a yellow line don't count toward where the car is: two lanes seen to the right, the far one beyond yellow,
  # puts the car in the middle of the map's three lanes, not the left one
  assert lane_changes_needed(left, model_lanes(*road(0, 2))) == 0
  assert lane_changes_needed(left, model_lanes(*road(1, 2)), paint=paint(far_right=SY)) == 1
  # without the map's lanes too: broken white at the ego line, yellow at the far one, is still one lane over
  assert lane_changes_needed(maneuver("turn", "left"), model_lanes(*road(2, 0)), paint=paint(left=BW, far_left=SY)) == 1


def test_target_lane_blocked():
  sc = scene([{'cls': 'car', 'x': 5., 'y': 3.6, 'vx': 15.}])
  assert target_lane_blocked(sc.objects, LEFT, 15.)
  assert not target_lane_blocked(sc.objects, RIGHT, 15.)
  # well ahead and pulling away vs closing fast from behind
  assert not target_lane_blocked(scene([{'cls': 'car', 'x': 40., 'y': 3.6, 'vx': 20.}]).objects, LEFT, 15.)
  assert target_lane_blocked(scene([{'cls': 'car', 'x': -40., 'y': 3.6, 'vx': 25.}]).objects, LEFT, 15.)
  assert not target_lane_blocked(scene([{'cls': 'car', 'x': -40., 'y': 3.6, 'vx': 16.}]).objects, LEFT, 15.)
  # two lanes over doesn't count
  assert not target_lane_blocked(scene([{'cls': 'car', 'x': 0., 'y': 7.5, 'vx': 15.}]).objects, LEFT, 15.)


class Drive:
  """A car on a road of lanes, with the real DesireHelper executing nav lane change requests."""
  def __init__(self, lanes_left, lanes_right, v=15., driver_signals=False, paint=None):
    """paint: (lanes left, lanes right) -> the four lines' paint where the car is now, or None for none read."""
    self.lanes_left, self.lanes_right, self.v = lanes_left, lanes_right, v
    self.paint = paint
    self.lp, self.dh = LanePlanner(), DesireHelper()
    self.driver_signals = driver_signals
    self.lc_t = 0.
    self.requests, self.prompts = [], []

  def run(self, m, distance, seconds, objects=(), blinker=None):
    prev = LaneChange.none
    for _ in range(int(seconds / DT_MDL)):
      lanes = model_lanes(*road(self.lanes_left, self.lanes_right))
      left_blinker = blinker == LaneChange.left
      right_blinker = blinker == LaneChange.right
      request, prompt, reason = self.lp.update(m, distance, lanes, self.dh.lane_change_state, self.v, left_blinker, right_blinker,
                                               objects, self.driver_signals, True, DT_MDL,
                                               self.paint(self.lanes_left, self.lanes_right) if self.paint else ())
      if request != LaneChange.none and prev == LaneChange.none:
        self.requests.append((request, distance))
      prev = request
      self.prompts.append(prompt)
      # the model takes 3 s to change lanes once DesireHelper asks it to
      starting = self.dh.lane_change_state == log.LaneChangeState.laneChangeStarting
      self.lc_t = self.lc_t + DT_MDL if starting else 0.
      prob = 1. if starting and self.lc_t < 3. else 0.
      if starting and self.lc_t >= 3. and self.lc_t - DT_MDL < 3.:
        d = 1 if self.dh.lane_change_direction == log.LaneChangeDirection.left else -1
        self.lanes_left, self.lanes_right = self.lanes_left - d, self.lanes_right + d
      self.dh.update(car_state(self.v, left_blinker=left_blinker, right_blinker=right_blinker), True, prob, request)
      distance -= self.v * DT_MDL
    return distance


def test_left_turn_on_three_lane_road_requests_two_lane_changes():
  drive = Drive(lanes_left=2, lanes_right=0)
  m = maneuver("turn", "left", THREE_LANES_LEFT_TURN)
  distance = drive.run(m, 500., 30.)
  assert drive.requests and all(r == LaneChange.left for r, _ in drive.requests)
  assert len(drive.requests) == 2 and drive.lp.changes == 2
  assert (drive.lanes_left, drive.lanes_right) == (0, 2)
  start, end = lane_change_window(drive.v, 2)
  assert all(end < d < start for _, d in drive.requests)
  assert drive.requests[0][1] > lane_change_window(drive.v)[0]  # two changes to make: the first starts earlier
  assert distance < 100.


def test_car_in_target_lane_blocks_the_request():
  drive = Drive(lanes_left=0, lanes_right=1)
  m = maneuver("turn", "right", TWO_LANES_RIGHT_TURN)
  beside = scene([{'cls': 'car', 'x': -3., 'y': -3.6, 'vx': 15.}]).objects
  drive.run(m, 300., 5., objects=beside)
  assert not drive.requests and drive.lp.update(m, 200., model_lanes(*road(0, 1)), 0, 15., False, False, beside,
                                                 False, True, DT_MDL)[2] == "car in target lane"
  drive.run(m, 225., 5.)
  assert [r for r, _ in drive.requests] == [LaneChange.right]
  assert abs(drive.requests[0][1] - 225.) < LC_BLOCK_X_M


def test_driver_signals_on_cars_openpilot_cant_signal_for():
  drive = Drive(lanes_left=0, lanes_right=1, driver_signals=True)
  m = maneuver("turn", "right", TWO_LANES_RIGHT_TURN)
  distance = drive.run(m, 300., 3.)
  assert not drive.requests and drive.prompts[-1] == LaneChange.right
  drive.run(m, distance, 5., blinker=LaneChange.right)
  assert [r for r, _ in drive.requests] == [LaneChange.right] and drive.lp.changes == 1


def test_no_request_below_lane_change_speed():
  drive = Drive(lanes_left=0, lanes_right=2, v=7.)
  drive.run(maneuver("turn", "right", TWO_LANES_RIGHT_TURN), 200., 10.)
  assert not drive.requests


def three_lanes_and_a_bike_lane(lanes_left, lanes_right):
  """Double yellow on the left, broken white between the three lanes, a solid white bike lane line on the right."""
  return paint(left=BW if lanes_left else SY, right=BW if lanes_right else SW)


def test_turns_move_over_while_the_line_is_broken_white():
  # left turn, no map lanes: over twice, then the line on the left is yellow
  drive = Drive(lanes_left=2, lanes_right=0, paint=three_lanes_and_a_bike_lane)
  drive.run(maneuver("turn", "left"), 500., 30.)
  assert [r for r, _ in drive.requests] == [LaneChange.left] * 2 and (drive.lanes_left, drive.lanes_right) == (0, 2)
  start, end = lane_change_window(drive.v)
  assert all(end < d < start for _, d in drive.requests)
  # right turn: over to the right lane, never into the bike lane
  drive = Drive(lanes_left=0, lanes_right=2, paint=three_lanes_and_a_bike_lane)
  drive.run(maneuver("turn", "right"), 500., 30.)
  assert [r for r, _ in drive.requests] == [LaneChange.right] * 2 and (drive.lanes_left, drive.lanes_right) == (2, 0)
  # nothing read: a turn without the map's lanes changes nothing, as before
  drive = Drive(lanes_left=2, lanes_right=0)
  drive.run(maneuver("turn", "left"), 500., 30.)
  assert not drive.requests
  drive = Drive(lanes_left=2, lanes_right=0, paint=lambda left, right: paint())
  drive.run(maneuver("turn", "left"), 500., 30.)
  assert not drive.requests


def test_yellow_line_stops_map_lane_changes():
  # the map says the leftmost of three lanes, but the model's two lanes on the left are beyond a yellow line
  drive = Drive(lanes_left=2, lanes_right=0, paint=lambda left, right: paint(left=SY, right=SW))
  drive.run(maneuver("turn", "left", THREE_LANES_LEFT_TURN), 500., 30.)
  assert not drive.requests
