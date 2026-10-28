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
  4. the wrong lane (turnpilotWrongLane).
  turnpilotd keeps the lower ones unset while it isn't acting (turnpilotd.py), so this order holds there too."""
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
