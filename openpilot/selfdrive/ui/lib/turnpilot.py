"""What TurnPilot shows, from turnpilotPlan and carState. No raylib here: soundd uses it for the prompt's chime."""
from openpilot.cereal import custom, log

State = custom.TurnpilotPlan.State
Desire = log.Desire

DESIRE_TEXT = {
  Desire.turnLeft: 'Turning left',
  Desire.turnRight: 'Turning right',
  Desire.keepLeft: 'Keeping left',
  Desire.keepRight: 'Keeping right',
}


def active_plan(sm, enabled: bool):
  """The turnpilotPlan reader when TurnPilot is on and turnpilotd is publishing, else None."""
  if not (enabled and sm.alive['turnpilotPlan'] and sm.valid['turnpilotPlan']):
    return None
  plan = sm['turnpilotPlan']
  return None if plan.state == State.disabled else plan


def signal_prompt(plan, car_state) -> str:
  """'left' or 'right' while the plan waits for the driver's blinker that way, else ''.

  turnpilotd keeps signalPrompt set after the driver signals, so the blinker check is here."""
  if plan is None or plan.state != State.laneChange:
    return ''
  side = str(plan.signalPrompt)
  if (side == 'left' and not car_state.leftBlinker) or (side == 'right' and not car_state.rightBlinker):
    return side
  return ''


def status_text(plan, engaged: bool) -> tuple[str, str]:
  """(title, detail) for the status line; ('', '') while TurnPilot is just following or has no route.

  It says what the car is doing, so it needs openpilot engaged, except in shadow mode, where the driver steers."""
  if plan is None or not (engaged or plan.shadow):
    return '', ''
  detail = ''
  if plan.state == State.turning:
    modifier = plan.maneuverModifier
    side = 'left' if 'left' in modifier else 'right' if 'right' in modifier else ''
    title = DESIRE_TEXT.get(plan.desire) or (f'Turning {side}' if side else 'Turning')
  elif plan.state == State.laneChange:
    side = next((str(s) for s in (plan.laneChange, plan.signalPrompt) if str(s) != 'none'), '')
    title = f'Lane change {side}' if side else 'Lane change'
  elif plan.state == State.holding:
    title, detail = 'Holding', plan.reason
  elif plan.state == State.arrived:
    title = 'Arrived'
  else:
    return '', ''
  if plan.shadow:
    detail = f'shadow: {detail}' if detail else 'shadow'
  return title, detail


def map_status_text(status: dict | None, destination_set: bool) -> tuple[str, str]:
  """(title, detail) for navd's TurnPilotMapStatus while a destination is set: the reason there is no route, or the
  drive's map downloading. Shown whether or not openpilot is engaged, since it is about the route, not the car."""
  if not destination_set or not status:
    return '', ''
  if status.get('state') == 'error':
    return 'No route', status.get('short') or status.get('message') or 'routing failed'
  if status.get('state') == 'downloading':
    progress = status.get('progress')
    return 'Downloading map', f'for this drive, {progress * 100:.0f}%' if isinstance(progress, (int, float)) else 'for this drive'
  return '', ''


def mapped_area_text(coverage_cache, position, override: str = '') -> str:
  """Which mapped area position ((latitude, longitude) or None) is in, for the TurnPilot settings screen."""
  from openpilot.selfdrive.turnpilot.maps.client import MapError
  if override:
    return 'local tiles (developer), no geofence'
  try:
    coverage = coverage_cache.get()
  except MapError:
    return 'saved mapped area is unreadable'
  if coverage is None:
    return 'no mapped area downloaded'
  if position is None:
    return f'car position unknown; mapped: {coverage.names()}'
  region = coverage.region_at(position[1], position[0])
  return region.name if region else "Outside TurnPilot's mapped area"
