"""TurnPilot on the onroad screen: a short status while it acts, and the countdown and prompt to signal for a nav lane change.

The maneuver itself (arrow, distance, street) is in the navigation panel. Everything here is gated on
TurnPilotEnabled and a live, valid turnpilotPlan, so nothing shows when TurnPilot is off or turnpilotd is down.
The text logic is in selfdrive/ui/lib/turnpilot.py.
"""
import pyray as rl

from openpilot.cereal import log
from openpilot.selfdrive.turnpilot.hooks import lane_change_text
from openpilot.selfdrive.ui.lib.turnpilot import signal_prompt
from openpilot.selfdrive.ui.mici.onroad.alert_renderer import ALERT_MARGIN, Alert, AlertRenderer
from openpilot.system.ui.lib.application import FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel

AlertSize = log.SelfdriveState.AlertSize
AlertStatus = log.SelfdriveState.AlertStatus

TEXT_COLOR = rl.Color(255, 255, 255, int(255 * 0.9))  # alert text
DETAIL_COLOR = rl.Color(255, 255, 255, int(255 * 0.65))  # alert subtext


def prompt_alert(plan, car_state) -> Alert | None:
  side = signal_prompt(plan, car_state)
  if not side:
    return None
  # preLaneChange* gets upstream's turn signal icon and small alert background. Engaged, selfdrived's
  # turnpilotLaneChange* alert says the same and chimes; this shows in shadow mode, and while openpilot isn't engaged
  return Alert(text1=lane_change_text(plan, side), text2='shadow mode' if plan.shadow else 'for the route', size=AlertSize.small,
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
