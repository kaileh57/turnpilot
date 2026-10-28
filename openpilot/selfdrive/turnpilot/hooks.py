"""What upstream processes take from turnpilotPlan: modeld its desire and lane change, plannerd its limits, selfdrived
its lane change events."""
import math

SPEED_CAP_TAU = 1.0           # s, from a speed cap to an accel
SPEED_CAP_ACCEL_MIN = -2.0    # m/s^2: a speed cap alone brakes no harder than this; accelCap carries hard stops
# turnpilotPlan.desire past log.Desire: the turn is done, modeld drops the turn desires from the model's desire history
CLEAR_DESIRE = 255


def active_plan(sm):
  """The turnpilotPlan to act on, or None while turnpilotd is stale or in shadow mode.
  sm is a SubMaster subscribed to turnpilotPlan; anything without an alive dict (a test's plain dict) gives None."""
  if not getattr(sm, 'alive', {}).get('turnpilotPlan', False):
    return None
  plan = sm['turnpilotPlan']
  return None if plan.shadow else plan


ABORT_HOLD_S = 2.             # turnpilotPlan.laneChangeAbort holds its reason this long, for selfdrived's alert
# turnpilotPlan.laneChangeAbort -> what the driver is told
ABORT_TEXT = {
  'lineNotBroken': 'Lane line not broken white', 'noLane': 'No lane there', 'laneParked': 'Cars parked in that lane',
  'blindspot': 'Car in blind spot', 'carInLane': 'Car in target lane', 'noSignal': 'No signal from driver',
  'routeChanged': 'Route changed', 'notNeeded': 'Already in the lane', 'tooLate': 'Too close to the turn',
  'tooSlow': 'Too slow to change lanes', 'timedOut': 'Lane change timed out', 'turnedOff': 'Lane changes turned off',
  'badInputs': 'openpilot inputs invalid', 'stale': 'TurnPilot not responding',
}


def lane_change_events(plan) -> tuple[str, ...]:
  """The onroad event names (log.OnroadEvent.EventName) a turnpilotPlan raises: the coming lane change, left or right,
  from its warning until it starts, and an abort while laneChangeAbort holds its reason. plan: active_plan's, or None."""
  if plan is None:
    return ()
  offer = str(plan.laneChangeOffer)
  events = ('turnpilotLaneChangeLeft',) if offer == 'left' else ('turnpilotLaneChangeRight',) if offer == 'right' else ()
  return events + (('turnpilotLaneChangeAborted',) if str(plan.laneChangeAbort) != 'none' else ())


class LaneChangeEvents:
  """selfdrived's lane change events: the plan's, and an abort when the plan goes away (turnpilotd stopped, shadow mode
  turned on) while a change was offered, so a warned-of change never just disappears."""
  def __init__(self):
    self.offered = False
    self._stale_t = 0.

  def update(self, plan, dt: float) -> tuple[str, ...]:
    """plan: active_plan's, None when stale or in shadow mode."""
    if plan is None:
      if self.offered:
        self.offered, self._stale_t = False, ABORT_HOLD_S
      if self._stale_t > 0.:
        self._stale_t -= dt
        return ('turnpilotLaneChangeAborted',)
      return ()
    self._stale_t = 0.
    self.offered = str(plan.laneChangeOffer) != 'none'
    return lane_change_events(plan)


def lane_change_text(plan, side: str) -> str:
  """The coming lane change's alert line: the countdown, then the prompt to signal, then waiting for the blind spot."""
  if plan is None:
    return f'Lane Change {side.capitalize()}'
  if plan.laneChangeCountdown > 0.:
    return f'Lane Change {side.capitalize()} in {math.ceil(plan.laneChangeCountdown)}'
  if str(plan.laneChange) == side:
    return f'Changing Lanes {side.capitalize()} Once Clear'
  return f'Signal {side.capitalize()} to Change Lanes'


def plan_desire(desire: int) -> tuple[int, bool]:
  """(desire to feed the model, clear the turn desire history) from turnpilotPlan.desire."""
  return (0, True) if desire == CLEAR_DESIRE else (desire, False)


def plan_accel(accel_cap: float, speed_cap: float, v_ego: float) -> float:
  """The longitudinal planner's accel candidate from a plan's caps."""
  return min(accel_cap, max((speed_cap - v_ego) / SPEED_CAP_TAU, SPEED_CAP_ACCEL_MIN))
