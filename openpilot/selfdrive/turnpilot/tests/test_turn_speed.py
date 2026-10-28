import math

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.desire_timing import TURN_DESIRE_LEAD_M
from openpilot.selfdrive.turnpilot.route_follower import NOMINAL_ANGLE_DEG, Maneuver, classify
from openpilot.selfdrive.turnpilot.hooks import plan_accel
from openpilot.selfdrive.turnpilot.turn_speed import (CORNER_M, NO_SPEED_CAP, V_CORNER, V_TURN_MAX, V_TURN_MIN, turn_speed_cap,
                                                      turn_target_speed)


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
  assert turn_speed_cap(maneuver("turn", "right"), 0., 4.) == V_CORNER
  assert turn_speed_cap(maneuver("turn", "slight left"), 0., 4.) < 10.
  assert turn_speed_cap(maneuver("off ramp", "slight right"), 0., 25.) == NO_SPEED_CAP
  assert turn_speed_cap(maneuver("fork", "left"), 0., 25.) == NO_SPEED_CAP
  assert turn_speed_cap(maneuver("turn", "uturn"), 0., 4.) == NO_SPEED_CAP
  assert turn_speed_cap(None, 0., 4.) == NO_SPEED_CAP
  # traffic keeping left: the right turn is the wide one
  assert turn_speed_cap(maneuver("turn", "slight right"), 0., 4., right_hand_traffic=False) > \
         turn_speed_cap(maneuver("turn", "slight right"), 0., 4.)


def test_no_cap_where_the_highway_bends():
  # "continue slight right" at 30 m/s braked the car at 2 m/s^2 toward 6.9 m/s from 300 m out
  for mtype, modifier in (("continue", "slight right"), ("new name", "slight left"), ("turn", "slight right")):
    for d in (300., 100., 20., 0.):
      assert turn_speed_cap(maneuver(mtype, modifier), d, 30.) == NO_SPEED_CAP, (mtype, modifier, d)
  # a slight turn in town is still capped, and a junction turn is capped at any speed
  assert turn_speed_cap(maneuver("turn", "slight right"), 20., 14.) < 10.
  assert turn_speed_cap(maneuver("turn", "right"), 100., 30.) < 15.


def test_braking_curve():
  m = maneuver("turn", "right")
  assert math.isclose(turn_speed_cap(m, 100. + TURN_DESIRE_LEAD_M, 0.), math.sqrt(V_TURN_MIN ** 2 + 2 * 1.5 * 100.))
  assert turn_speed_cap(m, 300., 15.) > 25.  # far out it doesn't bite


def test_car_reaches_turn_speed_by_the_desire():
  for modifier in ("right", "left"):
    m = maneuver("turn", modifier)
    d, v, min_a = 300., 15., 0.
    at_desire = None
    while d > CORNER_M:
      a = min(plan_accel(10., turn_speed_cap(m, d, v), v), 0.)  # the car would hold its speed otherwise
      min_a = min(min_a, a)
      v = max(v + a * DT_MDL, 0.)
      d -= v * DT_MDL
      if at_desire is None and d < TURN_DESIRE_LEAD_M:
        at_desire = v
    assert V_TURN_MIN - 0.5 < at_desire < V_TURN_MIN + 0.25, modifier  # gate 1's speed when the desire goes out
    assert v < V_CORNER + 0.5, modifier                                  # and its own turn speed by the corner
    assert min_a > -1.6, modifier


def test_corner_curve_at_real_angles():
  # a right whose bearings turn 70 degrees still slows to V_CORNER by CORNER_M, as the nominal 90 did; a slight keep doesn't
  for angle in (70., 85., 90., 120.):
    real = Maneuver(*classify("turn", "right"), "turn", "right", angle, geometry=True)
    assert turn_speed_cap(real, CORNER_M, 4.) == V_CORNER, angle
  keep = Maneuver(*classify("turn", "slight right"), "turn", "slight right", 30., geometry=True)
  assert turn_speed_cap(keep, CORNER_M, 4.) > V_CORNER
