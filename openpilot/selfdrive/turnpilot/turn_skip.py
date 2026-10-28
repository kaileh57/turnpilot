"""Too fast to turn: skip the turn rather than brake hard for it or take it at speed.

Kellen's rule (2026-10-27): during the approach to a junction turn, keep checking that the car can still get down to
a safe speed for the turn's real angle by the junction, braking comfortably. Once it can't, the turn is skipped for
good: its desire never goes out, its speed cap and the scene and lane logic for it are dropped (no hard braking for a
turn), the driver gets the turnpilotTooFastToTurn alert, and navd reroutes once the car is past the junction. The model
drives on as it would with no route.

The check, from the turn's angle (bearing_after less bearing_before) and the side of the road it turns across:
  v_safe = sqrt(A_LAT_SAFE * R),  R = corner tangent / tan(angle / 2)    (turn_speed.corner_tangent: 8 m to the curb side,
                                                                           12 m across the road)
  feasible  <=>  v_ego^2 <= v_safe^2 + 2 * A_COMFORT * max(d - v_ego * T_LAG, 0)
A_COMFORT is the hardest the turn speed cap may brake (hooks.SPEED_CAP_ACCEL_MIN, 2 m/s^2), so "feasible" means the
cap can get there within its own limit, and T_LAG is the planner's time constant from cap to accel (hooks.SPEED_CAP_TAU,
1 s), the distance covered before braking builds. A_LAT_SAFE is the turn cap's 2 m/s^2. At 90 degrees to the curb side
that is v_safe 4.0 m/s, and the latest feasible speeds are 11.4 m/s 40 m out, 18.5 m/s 100 m out and 22.9 m/s 150 m out.
The cap's own curve (1.5 m/s^2, leading the car by 2 s) stays under this one, so a car that follows the cap never skips.

The check only gates the desire's pulses. A turn is skipped after SKIP_CONFIRM_FRAMES infeasible frames in a row, so one
dead-reckoning jump doesn't skip it; meanwhile no pulse starts (feasible_now gates it). Once the desire is out the turn
is never skipped and nothing is cancelled: a turn the model has started planning is finished (cancelling mid-turn is how
you crash). Near the junction the check fails for a car rolling at the model's own corner speed, which is one more
reason it can't cancel. It still gates the repeat pulses and a fresh pulse on re-engaging (the old 10 m/s gate's job):
a car the driver sped up isn't sent into the turn again from road speed.
Only junction turns (route_follower.TURN) are checked. A keep (slight turns, forks, ramps) is taken at speed by the
model, and its desire goes at any speed.
"""
import math
from dataclasses import dataclass

from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.turnpilot.hooks import SPEED_CAP_ACCEL_MIN, SPEED_CAP_TAU
from openpilot.selfdrive.turnpilot.route_follower import LEFT, TURN
from openpilot.selfdrive.turnpilot.turn_speed import A_LAT_MAX, V_TURN_MAX, corner_tangent

A_LAT_SAFE = A_LAT_MAX           # m/s^2 through the corner
A_COMFORT = -SPEED_CAP_ACCEL_MIN  # m/s^2: 2.0, the hardest the turn speed cap brakes
T_LAG = SPEED_CAP_TAU            # s before the braking builds
SKIP_CONFIRM_FRAMES = 5          # 0.25 s at 20 Hz


def safe_turn_speed(angle_deg: float, curb_side: bool) -> float:
  """m/s to take a junction turn of this unsigned angle, from the corner's radius."""
  half = math.radians(min(max(angle_deg, 1.), 179.)) / 2
  return min(math.sqrt(A_LAT_SAFE * corner_tangent(curb_side) / math.tan(half)), V_TURN_MAX)


def can_make_turn(v_ego: float, distance: float, angle_deg: float, curb_side: bool) -> bool:
  """Whether braking at A_COMFORT after T_LAG gets the car to safe_turn_speed by the junction, distance m ahead."""
  v_ego = max(v_ego, 0.)
  v_safe = safe_turn_speed(angle_deg, curb_side)
  return v_ego ** 2 <= v_safe ** 2 + 2. * A_COMFORT * max(distance - v_ego * T_LAG, 0.)


@dataclass
class SkipState:
  skipped: bool       # this maneuver is skipped for good
  feasible_now: bool  # a desire may start this frame


class TurnSkip:
  def __init__(self):
    self.maneuver_id = -1
    self.skipped = False
    self._infeasible = 0

  def skip(self, maneuver, distance: float, maneuver_id: int, v_ego: float, reason: str) -> None:
    """Skip this maneuver for good now, for reason: as when it is too fast to make (turnpilotd: a turn whose desire was
    dropped while TurnPilot had handed back, with too little road left for a fresh pulse to be taken)."""
    self.maneuver_id, self.skipped, self._infeasible = maneuver_id, True, 0
    cloudlog.event("turnpilot_turn_skipped", error=True, reason=reason, type=maneuver.type, modifier=maneuver.modifier,
                   angle_deg=round(maneuver.angle_deg, 1), geometry=maneuver.geometry, distance=round(distance, 1),
                   v_ego=round(v_ego, 2))

  def update(self, maneuver, distance: float, maneuver_id: int, v_ego: float, desire_out: bool,
             right_hand_traffic: bool = True) -> SkipState:
    """maneuver: RouteFollower.upcoming; desire_out: this maneuver's desire has gone out (DesireTiming.started)."""
    if maneuver_id != self.maneuver_id:
      self.maneuver_id, self.skipped, self._infeasible = maneuver_id, False, 0
    if self.skipped:
      return SkipState(True, False)
    if maneuver is None or maneuver.kind != TURN:
      self._infeasible = 0
      return SkipState(False, True)
    curb_side = (maneuver.side == LEFT) != right_hand_traffic
    if can_make_turn(v_ego, distance, maneuver.angle_deg, curb_side):
      self._infeasible = 0
      return SkipState(False, True)
    if desire_out:  # committed: never skipped now, only no new pulse
      return SkipState(False, False)
    self._infeasible += 1
    if self._infeasible >= SKIP_CONFIRM_FRAMES:
      self.skipped = True
      cloudlog.event("turnpilot_turn_skipped", error=True, type=maneuver.type, modifier=maneuver.modifier,
                     angle_deg=round(maneuver.angle_deg, 1), geometry=maneuver.geometry, distance=round(distance, 1),
                     v_ego=round(v_ego, 2), v_safe=round(safe_turn_speed(maneuver.angle_deg, curb_side), 2))
    return SkipState(self.skipped, False)
