"""What upstream processes take from turnpilotPlan: modeld its desire and lane change, plannerd its limits."""
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


def plan_desire(desire: int) -> tuple[int, bool]:
  """(desire to feed the model, clear the turn desire history) from turnpilotPlan.desire."""
  return (0, True) if desire == CLEAR_DESIRE else (desire, False)


def plan_accel(accel_cap: float, speed_cap: float, v_ego: float) -> float:
  """The longitudinal planner's accel candidate from a plan's caps."""
  return min(accel_cap, max((speed_cap - v_ego) / SPEED_CAP_TAU, SPEED_CAP_ACCEL_MIN))
