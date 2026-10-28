"""What upstream processes take from turnpilotPlan: modeld its desire and lane change, plannerd its limits.

A plan is acted on only while it is fresh by its receive time. SubMaster.alive can't say: under SIMULATION it stays
true for good once a service was seen, so a dead turnpilotd's last speed cap or desire would hold for the rest of a
sim drive.
"""
import time

from openpilot.cereal import custom

SPEED_CAP_TAU = 1.0           # s, from a speed cap to an accel
SPEED_CAP_ACCEL_MIN = -2.0    # m/s^2: a speed cap alone brakes no harder than this; accelCap carries hard stops
PLAN_MAX_AGE_S = 0.5          # s since the last turnpilotPlan (20 Hz): SubMaster's own 10 periods, outside simulation too
# turnpilotPlan.desire past log.Desire: the turn is done, modeld drops the turn desires from the model's desire history
CLEAR_DESIRE = 255


def active_plan(sm, now: float | None = None):
  """The turnpilotPlan to act on, or None while it is stale, invalid, disabled or in shadow mode.
  sm is a SubMaster subscribed to turnpilotPlan; anything without its recv_time and valid dicts (a test's plain dict)
  gives None."""
  recv_t = getattr(sm, 'recv_time', {}).get('turnpilotPlan')
  if recv_t is None or not getattr(sm, 'valid', {}).get('turnpilotPlan', False):
    return None
  if (time.monotonic() if now is None else now) - recv_t > PLAN_MAX_AGE_S:
    return None
  plan = sm['turnpilotPlan']
  return None if plan.shadow or plan.state == custom.TurnpilotPlan.State.disabled else plan


def plan_events(sm, now: float | None = None) -> list[str]:
  """selfdrived's onroad event names from a fresh, valid turnpilotPlan (none while it is stale: a dead turnpilotd is
  processNotRunning already). sm is a SubMaster subscribed to turnpilotPlan."""
  recv_t = getattr(sm, 'recv_time', {}).get('turnpilotPlan')
  if recv_t is None or not recv_t or not getattr(sm, 'valid', {}).get('turnpilotPlan', False):
    return []
  if (time.monotonic() if now is None else now) - recv_t > PLAN_MAX_AGE_S:
    return []
  plan = sm['turnpilotPlan']
  events = []
  if plan.fault != custom.TurnpilotPlan.Fault.none:
    events.append('turnpilotUnavailable')
  if not plan.shadow:  # in shadow mode the driver drives, and TurnPilot acts on nothing
    if plan.tooFastToTurn:
      events.append('turnpilotTooFastToTurn')
    if plan.laneMismatch != custom.TurnpilotPlan.LaneMismatch.none:
      events.append('turnpilotWrongLane')
  return events


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
