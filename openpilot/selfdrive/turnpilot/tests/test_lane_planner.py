import openpilot.cereal.messaging as messaging
from openpilot.cereal import custom, log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.turnpilot.lane_planner import (LC_BLOCK_X_M, LanePlanner, ModelLanes, lane_change_window, lane_changes_needed, lane_mismatch,
                                                        model_lanes, model_lanes_from_msg, target_lane_blocked)
from openpilot.selfdrive.turnpilot.route_follower import LEFT, NOMINAL_ANGLE_DEG, RIGHT, Maneuver, classify
from openpilot.selfdrive.turnpilot.tests.synthetic import car_state, model_v2, road, scene

LaneChange = custom.TurnpilotPlan.LaneChange
THREE_LANES_LEFT_TURN = ((("left",), True), (("straight",), False), (("straight", "right"), False))
TWO_LANES_RIGHT_TURN = ((("straight",), False), (("right",), True))


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
  def __init__(self, lanes_left, lanes_right, v=15., driver_signals=False):
    self.lanes_left, self.lanes_right, self.v = lanes_left, lanes_right, v
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
                                               objects, self.driver_signals, True, DT_MDL)
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


LaneMismatch = custom.TurnpilotPlan.LaneMismatch
LEFT_ONLY_AND_THROUGH = ((("left",), False), (("straight",), True), (("straight", "right"), True))


def test_turn_only_lane_where_the_route_goes_straight():
  # three lanes at an intersection the route goes straight through, 40 m ahead at 10 m/s (the planner changes no lane
  # closer than 60 m): in the left-turn-only lane it is flagged, in the through lanes not
  in_left = model_lanes(*road(0, 2))
  assert lane_mismatch(None, 500., LEFT_ONLY_AND_THROUGH, 40., in_left, 10.) == LaneMismatch.turnOnlyGoingStraight
  for lanes in (model_lanes(*road(1, 1)), model_lanes(*road(2, 0))):
    assert lane_mismatch(None, 500., LEFT_ONLY_AND_THROUGH, 40., lanes, 10.) == LaneMismatch.none
  # further out the lane planner still has time; past it, nothing; a turn before it comes first
  assert lane_mismatch(None, 500., LEFT_ONLY_AND_THROUGH, 100., in_left, 10.) == LaneMismatch.none
  assert lane_mismatch(None, 500., LEFT_ONLY_AND_THROUGH, -5., in_left, 10.) == LaneMismatch.none
  assert lane_mismatch(maneuver("turn", "left", THREE_LANES_LEFT_TURN), 20., LEFT_ONLY_AND_THROUGH, 40., in_left, 10.) == \
         LaneMismatch.none
  # without the model reading the ego lane's lines the lane count means nothing
  unsure = model_lanes(*road(0, 2, line_prob=0.3))
  assert lane_mismatch(None, 500., LEFT_ONLY_AND_THROUGH, 40., unsure, 10.) == LaneMismatch.none


def test_wrong_lane_for_the_turn():
  # the route turns left from the left-turn lane; the car is still in a through lane 40 m out
  left_turn = maneuver("turn", "left", THREE_LANES_LEFT_TURN)
  assert lane_mismatch(left_turn, 40., (), 0., model_lanes(*road(1, 1)), 10.) == LaneMismatch.wrongLaneForTurn
  assert lane_mismatch(left_turn, 40., (), 0., model_lanes(*road(0, 2)), 10.) == LaneMismatch.none
  assert lane_mismatch(left_turn, 150., (), 0., model_lanes(*road(1, 1)), 10.) == LaneMismatch.none  # the planner's turn
  # a right from a lane marked straight-and-right is fine
  right_turn = maneuver("turn", "right", ((("straight",), False), (("straight", "right"), False)))
  assert lane_mismatch(right_turn, 40., (), 0., model_lanes(*road(1, 0)), 10.) == LaneMismatch.none
  assert lane_mismatch(right_turn, 40., (), 0., model_lanes(*road(0, 1)), 10.) == LaneMismatch.wrongLaneForTurn
