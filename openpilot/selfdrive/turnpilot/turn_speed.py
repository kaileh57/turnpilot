"""Turn speed: a speed cap that slows the car before the route's turns.

At a junction nothing in openpilot knows which way the route goes, so the car would reach a turn at road
speed. The target speed comes from a lateral accel limit on the corner's radius, estimated from the turn angle
(navInstruction has no bearings, so the angle is the modifier's nominal one unless a caller has better), with a
braking curve back from the junction. The planner turns the cap into an accel with a ~1 s time constant, so the
curve leads the car by that much.
"""
import math

from openpilot.selfdrive.turnpilot.route_follower import KEEP, LEFT, TURN

NO_SPEED_CAP = 100.
A_LAT_MAX = 2.0         # m/s^2 through the corner
A_DECEL = 1.5           # m/s^2 braking curve into the corner
V_TURN_MIN = 4.0        # m/s, what a 90-degree right comes to
V_TURN_MAX = 15.0
TURN_START_M = 5.0      # m before the maneuver point where the car should be at the turn speed
LAG_S = 1.0             # s the curve leads the car, the planner's speed-cap time constant
CORNER_TANGENT_M = 8.0  # m from the corner's start to the junction centre, turning to the curb side
FAR_SIDE_TANGENT_M = 12.0  # turning across the junction (left where traffic keeps right) is a wider arc


def turn_target_speed(angle_deg: float, curb_side: bool) -> float:
  """m/s to take a turn of this unsigned angle."""
  half = math.radians(min(max(angle_deg, 1.), 179.)) / 2
  radius = (CORNER_TANGENT_M if curb_side else FAR_SIDE_TANGENT_M) / math.tan(half)
  return min(max(math.sqrt(A_LAT_MAX * radius), V_TURN_MIN), V_TURN_MAX)


def turn_speed_cap(maneuver, distance: float, v_ego: float, right_hand_traffic: bool = True) -> float:
  """Speed cap for now, from the next maneuver (RouteFollower.upcoming) and the distance to it."""
  if maneuver is None or not (maneuver.kind == TURN or (maneuver.kind == KEEP and not maneuver.ramp)):
    return NO_SPEED_CAP  # ramps and forks: braking on the main road is wrong, and the model follows the ramp's curve
  curb_side = (maneuver.side == LEFT) != right_hand_traffic
  v_turn = turn_target_speed(maneuver.angle_deg, curb_side)
  d = max(distance - TURN_START_M - max(v_ego, 0.) * LAG_S, 0.)
  return min(math.sqrt(v_turn ** 2 + 2. * A_DECEL * d), NO_SPEED_CAP)
