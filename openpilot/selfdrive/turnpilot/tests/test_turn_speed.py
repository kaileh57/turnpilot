import math

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.desire_timing import TURN_DESIRE_LEAD_M
from openpilot.selfdrive.turnpilot.route_follower import NOMINAL_ANGLE_DEG, Maneuver, classify
from openpilot.selfdrive.turnpilot.hooks import plan_accel
from openpilot.selfdrive.turnpilot.turn_speed import NO_SPEED_CAP, V_TURN_MAX, V_TURN_MIN, turn_speed_cap, turn_target_speed


def maneuver(mtype, modifier):
  kind, side = classify(mtype, modifier)
  return Maneuver(kind, side, mtype, modifier, NOMINAL_ANGLE_DEG.get(modifier, 0.))


def test_target_speeds():
  # every junction turn goes at gate 1's speed; only a slight turn across the road is wider and faster
  for angle in (90., 135.):
    for curb_side in (True, False):
      assert math.isclose(turn_target_speed(angle, curb_side), V_TURN_MIN)
  assert V_TURN_MIN < turn_target_speed(45., curb_side=False) < V_TURN_MAX
  assert turn_target_speed(5., curb_side=True) == V_TURN_MAX


def test_which_maneuvers_are_capped():
  assert turn_speed_cap(maneuver("turn", "right"), 0., 4.) == V_TURN_MIN
  assert turn_speed_cap(maneuver("turn", "slight left"), 0., 4.) < 10.
  assert turn_speed_cap(maneuver("off ramp", "slight right"), 0., 25.) == NO_SPEED_CAP
  assert turn_speed_cap(maneuver("fork", "left"), 0., 25.) == NO_SPEED_CAP
  assert turn_speed_cap(maneuver("turn", "uturn"), 0., 4.) == NO_SPEED_CAP
  assert turn_speed_cap(None, 0., 4.) == NO_SPEED_CAP
  # traffic keeping left: the right turn is the wide one
  assert turn_speed_cap(maneuver("turn", "slight right"), 0., 4., right_hand_traffic=False) > \
         turn_speed_cap(maneuver("turn", "slight right"), 0., 4.)


def test_braking_curve():
  m = maneuver("turn", "right")
  v_turn = turn_speed_cap(m, 0., 0.)
  assert math.isclose(turn_speed_cap(m, 100. + TURN_DESIRE_LEAD_M, 0.), math.sqrt(v_turn ** 2 + 2 * 1.5 * 100.))
  assert turn_speed_cap(m, 300., 15.) > 25.  # far out it doesn't bite


def test_car_reaches_turn_speed_by_the_desire():
  for modifier in ("right", "left"):
    m = maneuver("turn", modifier)
    v_turn = turn_speed_cap(m, 0., 0.)
    d, v, min_a = 300., 15., 0.
    while d > TURN_DESIRE_LEAD_M:
      a = min(plan_accel(10., turn_speed_cap(m, d, v), v), 0.)  # the car would hold its speed otherwise
      min_a = min(min_a, a)
      v = max(v + a * DT_MDL, 0.)
      d -= v * DT_MDL
    assert v < v_turn + 0.25, modifier
    assert min_a > -1.6, modifier
