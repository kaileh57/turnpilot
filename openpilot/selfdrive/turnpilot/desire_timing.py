"""Desire timing: which desire to send the driving model for the next maneuver, and when.

modeld feeds the model only the rising edge of a desire, so the desire is held through the maneuver and the
lead distance is the timing lever. Gate 1 (big model, 25 km/h, 2026-10-27) took 90-degree turns from a pulse
40 m before the junction and not from 10-20 m; a keep desire goes within max(2.5 s * v, 20 m). Once a desire
starts it is latched until the maneuver is passed or the timeout: navd's 1 Hz distance carries a few meters of
noise, and dropping the desire whenever it read past the lead again sent the model a new rising edge each time.
A keep toward a side where the model sees no lane line drove the car off the road there, so it is dropped.
The model forgets a pulse about 5 s after it (the length of its desire history), and when it does its plan drops
the turn: at g62 it planned the left, dropped it 10 m out and went straight, and a late retry pulse brought the turn
back at once. Pulses on a timer either came too early (forgotten halfway round a slow turn) or too late (still
remembered after the turn, which turned the car on past the exit road into the curb). So the plan decides: inside
PLAN_GATE_M a pulse goes out whenever the model's plan stops showing the turn, until the car has turned
TURN_DONE_DEG by its yaw rate; further out, every REPULSE_S. A repeat pulse is a short drop to none, long enough
that modeld (unsynchronized, conflated subscriber) can't miss it.
"""
import math

from openpilot.cereal import log
from openpilot.selfdrive.turnpilot.route_follower import KEEP, LEFT, TURN

Desire = log.Desire

TURN_DESIRE_LEAD_M = 40.      # gate 1's best lead on the big model
KEEP_DESIRE_LEAD_S = 2.5
KEEP_DESIRE_LEAD_MIN_M = 20.
KEEP_DESIRE_PAST_M = 5.       # a keep desire lasts this far past the maneuver point
DESIRE_TIMEOUT_S = 15.        # moving time a desire may be held before it is dropped for good
REPULSE_S = 3.5               # a turn desire pulses again this often further out than PLAN_GATE_M...
PLAN_GATE_M = 30.             # ...and inside it (to PLAN_PAST_M past) whenever the plan's end is less than...
PLAN_TURN_M = 5.              # ...this far toward the turn, at most every PLAN_GAP_S...
PLAN_PAST_M = 15.
PLAN_GAP_S = 1.5
TURN_DONE_DEG = 60.           # ...until the car has turned this far toward the maneuver
MAX_PULSES = 6
PULSE_GAP_FRAMES = 3          # frames of none before a repeat pulse
STANDSTILL_SPEED = 0.3        # m/s
STANDSTILL_REARM_S = 2.       # stopped this long (a queue, a red light), the pulse goes again once the car moves

DESIRES = {(TURN, LEFT): Desire.turnLeft, (TURN, -LEFT): Desire.turnRight,
           (KEEP, LEFT): Desire.keepLeft, (KEEP, -LEFT): Desire.keepRight}


def keep_lead(v_ego: float) -> float:
  return max(KEEP_DESIRE_LEAD_S * v_ego, KEEP_DESIRE_LEAD_MIN_M)


class DesireTiming:
  def __init__(self):
    self.maneuver_id = -1
    self._reset()

  def _reset(self):
    self.active = False       # a desire is being held for the current maneuver
    self.started = False      # the desire went out for this maneuver: latched until passed or timed out
    self.done = False         # passed or timed out: nothing more for this maneuver
    self.pulses = 0
    self._since_pulse = 0.
    self.turned_deg = 0.      # heading change toward the turn since the first pulse
    self._gap = 0
    self._moving_t = 0.
    self._stopped_t = 0.

  def update(self, maneuver, distance: float, maneuver_id: int, passed: bool, v_ego: float,
             steering_angle_deg: float, keep_left_ok: bool, keep_right_ok: bool, dt: float, yaw_rate: float = 0.,
             plan_end_y: float | None = None) -> int:
    """The desire to publish. maneuver is RouteFollower.upcoming (None when there is none);
    keep_left_ok/keep_right_ok: the model sees a lane line and road that way (lane_planner.ModelLanes);
    yaw_rate: rad/s, left positive; plan_end_y: m, the lateral end of modelV2's plan (right positive), None if unknown."""
    if maneuver_id != self.maneuver_id:
      self.maneuver_id = maneuver_id
      self._reset()
      passed = False  # that was the maneuver before this one
    if passed or maneuver is None:
      self.done = self.done or passed
      self.active = False
      return Desire.none
    desire = DESIRES.get((maneuver.kind, maneuver.side), Desire.none)
    if desire == Desire.none or self.done:
      return Desire.none

    if maneuver.kind == TURN:
      want = self.started or distance < TURN_DESIRE_LEAD_M
    else:
      if distance < -KEEP_DESIRE_PAST_M:
        self.done, self.active = True, False
        return Desire.none
      want = self.started or (distance < keep_lead(v_ego) and (keep_left_ok if maneuver.side == LEFT else keep_right_ok))
    if not want:
      self.active = False
      return Desire.none

    if v_ego < STANDSTILL_SPEED:
      self._stopped_t += dt
      if self._stopped_t > STANDSTILL_REARM_S:
        self.active = False  # the model has forgotten the pulse by now; send it again once the car moves
        return Desire.none
    else:
      self._stopped_t = 0.
      self._moving_t += dt
      if self._moving_t > DESIRE_TIMEOUT_S:
        self.done, self.active = True, False
        return Desire.none

    if self.active and maneuver.kind == TURN:
      self._since_pulse += dt
      self.turned_deg += math.degrees(yaw_rate * dt) * maneuver.side
      if distance >= PLAN_GATE_M:
        due = self._since_pulse >= REPULSE_S
      else:
        dropped = plan_end_y is not None and -plan_end_y * maneuver.side < PLAN_TURN_M
        due = dropped and self._since_pulse >= PLAN_GAP_S and distance > -PLAN_PAST_M
      if due and self.turned_deg < TURN_DONE_DEG and self.pulses < MAX_PULSES:
        self._gap, self._since_pulse = PULSE_GAP_FRAMES, 0.
        self.pulses += 1
    if self._gap > 0:
      self._gap -= 1
      return Desire.none

    if not self.active:
      self.active = self.started = True
      self.pulses += 1
      self._since_pulse = 0.
    return desire
