"""TurnPilot on the onroad screen: a short status while it acts, and the prompt to signal for a nav lane change.

The maneuver itself (arrow, distance, street) is in the navigation panel. Everything here is gated on
TurnPilotEnabled and a live, valid turnpilotPlan, so nothing shows when TurnPilot is off or turnpilotd is down.
"""
import pyray as rl

from openpilot.cereal import custom, log
from openpilot.selfdrive.ui.mici.onroad.alert_renderer import ALERT_MARGIN, Alert, AlertRenderer
from openpilot.system.ui.lib.application import FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel

State = custom.TurnpilotPlan.State
Desire = log.Desire
AlertSize = log.SelfdriveState.AlertSize
AlertStatus = log.SelfdriveState.AlertStatus

DESIRE_TEXT = {
  Desire.turnLeft: 'Turning left',
  Desire.turnRight: 'Turning right',
  Desire.keepLeft: 'Keeping left',
  Desire.keepRight: 'Keeping right',
}

TEXT_COLOR = rl.Color(255, 255, 255, int(255 * 0.9))  # alert text
DETAIL_COLOR = rl.Color(255, 255, 255, int(255 * 0.65))  # alert subtext


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


def status_text(plan) -> tuple[str, str]:
  """(title, detail) for the status line; ('', '') while TurnPilot is just following or has no route."""
  if plan is None:
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


def prompt_alert(plan, car_state) -> Alert | None:
  side = signal_prompt(plan, car_state)
  if not side:
    return None
  # preLaneChange* gets upstream's turn signal icon and small alert background
  return Alert(text1=f'Signal {side}', text2='shadow mode' if plan.shadow else 'for the route', size=AlertSize.small,
               status=AlertStatus.userPrompt, alert_type=f'preLaneChange{side.capitalize()}/turnpilot')


class SignalPrompt(AlertRenderer):
  """The signal prompt, drawn as a userPrompt alert. The owner sets it each frame, only while openpilot has no alert."""

  def __init__(self):
    super().__init__()
    self._alert: Alert | None = None

  def set_alert(self, alert: Alert | None):
    self._alert = alert

  def get_alert(self, sm) -> Alert | None:
    if self._alert is not None:
      self._prev_alert = self._alert
    return self._alert


class TurnPilotStatus(Widget):
  """One status line at the top of the camera view, over the same fade as a normal alert."""

  def __init__(self):
    super().__init__()
    self._title = UnifiedLabel('', 30, FontWeight.BOLD, TEXT_COLOR)
    self._detail = UnifiedLabel('', 26, FontWeight.MEDIUM, DETAIL_COLOR)

  def set_status(self, title: str, detail: str):
    self._title.set_text(title)
    self._detail.set_text(detail)

  @property
  def showing(self) -> bool:
    return bool(self._title.text)

  def _render(self, rect: rl.Rectangle):
    if not self.showing:
      return
    height = round(rect.height * 0.583)  # upstream's small alert background
    solid = round(height * 0.2)
    color = rl.Color(0, 0, 0, int(255 * 0.9))
    rl.draw_rectangle(int(rect.x), int(rect.y), int(rect.width), solid, color)
    rl.draw_rectangle_gradient_v(int(rect.x), int(rect.y + solid), int(rect.width), height - solid, color, rl.Color(0, 0, 0, 0))

    # beside the nav panel the view is 275 px wide, so long lines wrap rather than elide
    width = rect.width - ALERT_MARGIN
    title_height = self._title.get_content_height(int(width))
    self._title.render(rl.Rectangle(rect.x + ALERT_MARGIN, rect.y + 10, width, title_height))
    if self._detail.text:
      self._detail.render(rl.Rectangle(rect.x + ALERT_MARGIN, rect.y + 12 + title_height, width, height - title_height - 22))
