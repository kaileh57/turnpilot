"""Scene policy: tracked pedestrians and cars into longitudinal limits.

The driving model keeps lane keeping, leads, lights and the turn itself. This covers what it doesn't: a
pedestrian in or entering the path, a pedestrian at the crosswalk a curb-side turn crosses, and oncoming
traffic at a turn across it (every such turn is treated as unprotected). Each becomes a stop point; the
nearest one wins and is turned into a speed cap on a comfortable braking curve, plus an accel cap when that
curve can't make it. shouldStop is only set at the stop point near standstill, because longcontrol's stopping
state ignores the accel target and ramps the brakes on its own.

Objects are TurnpilotScene.TrackedObject (or anything with the same fields): x m forward, y m LEFT in the
calibrated frame, vx/vy over-ground velocity in that frame (ego motion removed), cls compared by name, inPath
set by the tracker. If a tracker publishes velocity relative to the car instead, a parked car ahead reads as
oncoming and the junction hold is conservative, not missed.
"""
import math
from dataclasses import dataclass

from openpilot.selfdrive.turnpilot.route_follower import LEFT, TURN

NO_ACCEL_CAP = 10.
NO_SPEED_CAP = 100.
VEHICLES = ("car", "truck", "bus", "motorcycle", "cyclist")

STOP_DECEL = 1.5            # m/s^2, the braking curve to a stop point
STOP_LAG_S = 1.0            # s the curve leads the car, the planner's speed-cap time constant
STOP_SPEED = 0.3            # m/s: slower than this...
STOP_NEAR_M = 1.5           # ...and this close to the stop point, shouldStop holds the car
HARD_STOP_MIN_M = 0.5

PATH_HALF_WIDTH_M = 1.5     # corridor around the car's axis for objects the tracker hasn't flagged inPath
NEAR_PATH_M = 15.           # ...used for anything this close ahead
ENTER_T_S = 1.5             # a pedestrian walking into the corridor within this counts as in it
PED_STOP_GAP_M = 5.0        # stop this far (camera to pedestrian) short of a pedestrian in the path

HOLD_LINE_M = 8.0           # m before the maneuver point where the car waits at a junction
HOLD_ENGAGE_MARGIN_M = 15.  # the junction hold engages within braking distance of the hold line plus this
COMMIT_SPEED = 2.0          # m/s: past the hold line and faster than this, the car is committed to its turn
TURN_ZONE_PAST_M = 5.       # m past the maneuver point the junction rules still apply

CROSSWALK_X_BEFORE_M = 6.   # crosswalk zone on the turn side, along the car's axis relative to the maneuver point
CROSSWALK_X_AFTER_M = 15.
CROSSWALK_Y_NEAR_M = 1.5    # and across it, to the turn side
CROSSWALK_Y_FAR_M = 15.

ONCOMING_TTA_S = 5.         # hold while an oncoming vehicle reaches the junction sooner than this
ONCOMING_MIN_SPEED = 1.     # m/s toward the car
ONCOMING_Y_NEAR_M = 1.5     # oncoming lanes, to the turn side
ONCOMING_Y_FAR_M = 20.
ONCOMING_PAST_M = 5.        # a vehicle this far short of the maneuver point has cleared it


@dataclass
class SceneDecision:
  accel_cap: float = NO_ACCEL_CAP
  speed_cap: float = NO_SPEED_CAP
  should_stop: bool = False
  holding: bool = False   # the limits bite now
  reason: str = ""


def stop_command(v_ego: float, d_stop: float) -> tuple[float, float, bool]:
  """(accel cap, speed cap, should stop) to stop d_stop m ahead."""
  d = max(d_stop, 0.)
  speed_cap = math.sqrt(2. * STOP_DECEL * max(d - v_ego * STOP_LAG_S, 0.))
  a_need = -v_ego ** 2 / (2. * max(d, HARD_STOP_MIN_M))
  accel_cap = a_need if a_need < -STOP_DECEL else NO_ACCEL_CAP
  return accel_cap, speed_cap, v_ego < STOP_SPEED and d < STOP_NEAR_M


def in_path(obj) -> bool:
  """The tracker's inPath follows the planned path; the straight-ahead corridor here only backs it up close in
  and for pedestrians walking into it."""
  if obj.inPath or (obj.x < NEAR_PATH_M and abs(obj.y) < PATH_HALF_WIDTH_M):
    return True
  return obj.y * obj.vy < 0 and abs(obj.y + obj.vy * ENTER_T_S) < PATH_HALF_WIDTH_M


def pedestrian_stops(objects):
  return [(obj.x - PED_STOP_GAP_M, "pedestrian in path") for obj in objects
          if str(obj.cls) == "pedestrian" and obj.x > 0. and in_path(obj)]


def junction_stops(objects, available, maneuver, distance, v_ego, right_hand_traffic):
  """Holds at the hold line before a turn: the crosswalk of a curb-side turn, oncoming traffic across a far-side one."""
  if maneuver is None or maneuver.kind != TURN or distance < -TURN_ZONE_PAST_M:
    return []
  d_hold = distance - HOLD_LINE_M
  if d_hold < 0. and v_ego > COMMIT_SPEED:
    return []  # committed: stopping in the junction is worse than finishing the turn
  if d_hold > v_ego ** 2 / (2. * STOP_DECEL) + HOLD_ENGAGE_MARGIN_M:
    return []
  side = maneuver.side  # the crosswalk and the oncoming lanes are both on the turn side; y is positive left
  curb_side = (side == LEFT) != right_hand_traffic
  if curb_side:
    walkers = [obj for obj in objects if str(obj.cls) == "pedestrian" and
               distance - CROSSWALK_X_BEFORE_M < obj.x < distance + CROSSWALK_X_AFTER_M and
               CROSSWALK_Y_NEAR_M < obj.y * side < CROSSWALK_Y_FAR_M]
    return [(d_hold, "pedestrian in crosswalk")] if walkers else []
  if not available:
    return [(d_hold, "detector unavailable")]
  for obj in objects:
    toward = -obj.vx
    if str(obj.cls) in VEHICLES and ONCOMING_Y_NEAR_M < obj.y * side < ONCOMING_Y_FAR_M and \
       obj.x > distance - ONCOMING_PAST_M and toward > ONCOMING_MIN_SPEED and \
       max(obj.x - distance, 0.) / toward < ONCOMING_TTA_S:
      return [(d_hold, "waiting for oncoming")]
  return []


def scene_policy(objects, available: bool, maneuver, distance: float, v_ego: float,
                 right_hand_traffic: bool = True) -> SceneDecision:
  """objects: tracked objects (ignored when not available); maneuver: RouteFollower.upcoming or None."""
  objects = list(objects) if available else []
  stops = pedestrian_stops(objects) + junction_stops(objects, available, maneuver, distance, v_ego, right_hand_traffic)
  if not stops:
    return SceneDecision(reason="" if available else "detector unavailable")
  d_stop, reason = min(stops)
  accel_cap, speed_cap, should_stop = stop_command(v_ego, d_stop)
  holding = should_stop or accel_cap < NO_ACCEL_CAP or speed_cap < v_ego + 0.5
  return SceneDecision(accel_cap, speed_cap, should_stop, holding, reason)
