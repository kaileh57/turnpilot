"""What upstream processes take from turnpilotPlan: modeld its desire and lane change, plannerd its limits, selfdrived
its alerts and lane change events.

A plan is acted on only while it is fresh by its receive time. SubMaster.alive can't say: under SIMULATION it stays
true for good once a service was seen, so a dead turnpilotd's last speed cap or desire would hold for the rest of a
sim drive.
"""
import math
import time

from openpilot.cereal import custom

SPEED_CAP_TAU = 1.0           # s, from a speed cap to an accel
SPEED_CAP_ACCEL_MIN = -2.0    # m/s^2: a speed cap alone brakes no harder than this; accelCap carries hard stops
PLAN_MAX_AGE_S = 0.5          # s since the last turnpilotPlan (20 Hz): SubMaster's own 10 periods, outside simulation too
# turnpilotPlan.desire past log.Desire: the turn is done, modeld drops the turn desires from the model's desire history
CLEAR_DESIRE = 255


def fresh_plan(sm, now: float | None = None):
  """The turnpilotPlan while it is fresh and valid, whatever its state (a disabled plan may carry a handback), or None.
  sm is a SubMaster subscribed to turnpilotPlan; anything without its recv_time and valid dicts (a test's plain dict)
  gives None."""
  recv_t = getattr(sm, 'recv_time', {}).get('turnpilotPlan')
  if recv_t is None or not getattr(sm, 'valid', {}).get('turnpilotPlan', False):
    return None
  if (time.monotonic() if now is None else now) - recv_t > PLAN_MAX_AGE_S:
    return None
  return sm['turnpilotPlan']


def shown_plan(sm, now: float | None = None):
  """The fresh turnpilotPlan selfdrived alerts from, disabled ones too (a fault's plan carries its abort), or None while
  it is stale, invalid or in shadow mode."""
  plan = fresh_plan(sm, now)
  return None if plan is None or plan.shadow else plan


def active_plan(sm, now: float | None = None):
  """The turnpilotPlan to act on, or None while it is stale, invalid, disabled or in shadow mode."""
  plan = fresh_plan(sm, now)
  return None if plan is None or plan.shadow or plan.state == custom.TurnpilotPlan.State.disabled else plan


def plan_events(sm, now: float | None = None) -> list[str]:
  """selfdrived's onroad event names from a fresh, valid turnpilotPlan (none while it is stale: a dead turnpilotd is
  processNotRunning already). sm is a SubMaster subscribed to turnpilotPlan.
  TurnPilot's alerts share one slot, at most one of them a frame, by this priority:
  1. a handback (turnpilotDetectorStopped, turnpilotModelFellBack; handback.py): nothing here while one is set;
  2. a fault (turnpilotUnavailable);
  3. too fast to turn (turnpilotTooFastToTurn);
  4. a planned lane change (turnpilotLaneChangeLeft/Right, LaneChangeEvents);
  5. the wrong lane (turnpilotWrongLane).
  turnpilotd keeps the lower ones unset while it isn't acting or is faulted (turnpilotd.py): no skip, no wrong lane and
  no lane change offer then, so this order holds without filtering. A handback or a fault cancels an offered lane change
  with its abort (turnpilotLaneChangeAborted, reason handedBack or fault), which is raised with them. On screen the
  engaged handback's soft disable (Priority.MID) outranks the abort (LOW); a fault's turnpilotUnavailable and a
  disengaged handback's alert are PERMANENT at Priority.LOWER, so there the abort shows its ABORT_HOLD_S first and the
  status line returns after it."""
  if not getattr(sm, 'recv_time', {}).get('turnpilotPlan'):
    return []
  plan = fresh_plan(sm, now)
  if plan is None or plan.handback != custom.TurnpilotPlan.Handback.none:
    return []
  if plan.fault != custom.TurnpilotPlan.Fault.none:
    return ['turnpilotUnavailable']
  if plan.shadow:  # in shadow mode the driver drives, and TurnPilot acts on nothing
    return []
  if plan.tooFastToTurn:
    return ['turnpilotTooFastToTurn']
  if plan.laneMismatch != custom.TurnpilotPlan.LaneMismatch.none:
    return ['turnpilotWrongLane']
  return []


ABORT_HOLD_S = 2.             # turnpilotPlan.laneChangeAbort holds its reason this long, for selfdrived's alert
# turnpilotPlan.laneChangeAbort -> what the driver is told
ABORT_TEXT = {
  'lineNotBroken': 'Lane line not broken white', 'noLane': 'No lane there', 'laneParked': 'Cars parked in that lane',
  'blindspot': 'Car in blind spot', 'carInLane': 'Car in target lane', 'noSignal': 'No signal from driver',
  'routeChanged': 'Route changed', 'notNeeded': 'Already in the lane', 'tooLate': 'Too close to the turn',
  'tooSlow': 'Too slow to change lanes', 'timedOut': 'Lane change timed out', 'turnedOff': 'Lane changes turned off',
  'badInputs': 'openpilot inputs invalid', 'stale': 'TurnPilot not responding',
  'handedBack': 'TurnPilot handed back', 'fault': 'TurnPilot unavailable',
}


def lane_change_events(plan) -> tuple[str, ...]:
  """The onroad event names (log.OnroadEvent.EventName) a turnpilotPlan raises: the coming lane change, left or right,
  from its warning until it starts, and an abort while laneChangeAbort holds its reason. plan: shown_plan's, or None."""
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
    """plan: shown_plan's, None when stale or in shadow mode."""
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


def plan_desire(desire: int, lat_active: bool) -> tuple[int, bool]:
  """(desire to feed the model, clear the turn desire history) from turnpilotPlan.desire. No desire while openpilot
  isn't steering (carControl.latActive), so the model never plans a turn the driver didn't engage for; turnpilotd
  pulses afresh once it is. A clear goes through regardless: it only takes turn desires out of the model's queue."""
  if desire == CLEAR_DESIRE:
    return 0, True
  return (desire if lat_active else 0), False


def plan_accel(accel_cap: float, speed_cap: float, v_ego: float) -> float:
  """The longitudinal planner's accel candidate from a plan's caps."""
  return min(accel_cap, max((speed_cap - v_ego) / SPEED_CAP_TAU, SPEED_CAP_ACCEL_MIN))
