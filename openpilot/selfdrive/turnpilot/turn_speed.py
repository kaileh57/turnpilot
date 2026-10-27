"""Turn speed: a speed cap that slows the car before the route's turns.

At a junction nothing in openpilot knows which way the route goes, so the car would reach a turn at road
speed. The target speed comes from a lateral accel limit on the corner's radius, estimated from the turn angle
(navInstruction has no bearings, so the angle is the modifier's nominal one unless a caller has better), but never
under 25 km/h, with a braking curve back to where the turn desire goes out. That is the speed the big model took
gate 1's turns from (a pulse 40 m out at 6.9 m/s, after which the model slows itself to about 4.5 m/s for the turn).
Braking on to a geometric 4-5 m/s before the pulse put the car at the junction 6 s after it, by which time the
model had dropped the desire and drove straight on (four junctions of four, 2026-10-27). Past the pulse a second
curve brings a junction turn down to the 4.3 m/s gate 1's model chose by itself 15 m out: at junctions where the
model planned the turn but held 7 m/s, it gave the turn up or stopped at the corner. The planner turns the cap into
an accel with a 1 s time constant, so the car trails the curve; the curve leads the car by 2 s.
A keep (a slight turn at a junction) is capped only from town speeds: above KEEP_CAP_SPEED_MAX it is a road bearing
off, and the cap would brake the car at 2 m/s^2 from highway speed for a bend it takes as it is. Ramps and forks are
never capped.
"""
import math

from openpilot.selfdrive.turnpilot.desire_timing import TURN_DESIRE_LEAD_M
from openpilot.selfdrive.turnpilot.route_follower import KEEP, LEFT, TURN

NO_SPEED_CAP = 100.
A_LAT_MAX = 2.0         # m/s^2 through the corner
A_DECEL = 1.5           # m/s^2 braking curve into the corner
V_TURN_MIN = 6.9        # m/s (25 km/h), gate 1's speed at the pulse
V_TURN_MAX = 15.0
KEEP_CAP_SPEED_MAX = 15.0  # m/s: a keep is capped only below this
LAG_S = 2.0             # s the curve leads the car
V_CORNER = 4.3          # m/s a junction turn slows to by CORNER_M before the junction (gate 1's model, by itself)
CORNER_M = 15.
CORNER_TANGENT_M = 8.0  # m from the corner's start to the junction center, turning to the curb side
FAR_SIDE_TANGENT_M = 12.0  # turning across the junction (left where traffic keeps right) is a wider arc


def corner_tangent(curb_side: bool) -> float:
  return CORNER_TANGENT_M if curb_side else FAR_SIDE_TANGENT_M


def turn_target_speed(angle_deg: float, curb_side: bool) -> float:
  """m/s to take a turn of this unsigned angle."""
  half = math.radians(min(max(angle_deg, 1.), 179.)) / 2
  radius = corner_tangent(curb_side) / math.tan(half)
  return min(max(math.sqrt(A_LAT_MAX * radius), V_TURN_MIN), V_TURN_MAX)


def turn_speed_cap(maneuver, distance: float, v_ego: float, right_hand_traffic: bool = True) -> float:
  """Speed cap for now, from the next maneuver (RouteFollower.upcoming) and the distance to it."""
  town_keep = maneuver is not None and maneuver.kind == KEEP and not maneuver.ramp and v_ego < KEEP_CAP_SPEED_MAX
  if maneuver is None or not (maneuver.kind == TURN or town_keep):
    return NO_SPEED_CAP  # ramps and forks: braking on the main road is wrong, and the model follows the ramp's curve
  curb_side = (maneuver.side == LEFT) != right_hand_traffic
  v_turn = turn_target_speed(maneuver.angle_deg, curb_side)
  lag = max(v_ego, 0.) * LAG_S
  cap = math.sqrt(v_turn ** 2 + 2. * A_DECEL * max(distance - TURN_DESIRE_LEAD_M - lag, 0.))
  if maneuver.kind == TURN and maneuver.angle_deg >= 90.:
    cap = min(cap, math.sqrt(V_CORNER ** 2 + 2. * A_DECEL * max(distance - CORNER_M - lag / 2., 0.)))
  return min(cap, NO_SPEED_CAP)
