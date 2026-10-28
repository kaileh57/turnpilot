"""navd's map and route status in TurnPilotMapStatus, for the web app and the comma UI.

{"state": "idle" | "checking" | "downloading" | "ready" | "error", "message": for the driver, "short": for the
onroad status line, "code": MapError code on errors, plus "region", "source" ("server" or "local"), "progress"
(0-1 or null), "bytes", "total" while downloading, and "at" (epoch seconds).
"""
from openpilot.selfdrive.turnpilot.maps.paths import wall_time

STATUS_PARAM = 'TurnPilotMapStatus'
STATES = ('idle', 'checking', 'downloading', 'ready', 'error')


class NavStatus:
  """Writes the status param only when it changes."""

  def __init__(self, params):
    self.params = params
    self.last: dict | None = None

  def set(self, state: str, message: str = '', short: str = '', code: str = '', **extra):
    assert state in STATES, state
    status = {'state': state, 'message': message, 'short': short or message, 'code': code, **extra}
    if status != self.last:
      self.params.put(STATUS_PARAM, {**status, 'at': wall_time()})
      self.last = status

  def error(self, e, **extra):
    self.set('error', e.message, e.short, e.code, **extra)


def read_status(params) -> dict | None:
  status = params.get(STATUS_PARAM)
  return status if isinstance(status, dict) and status.get('state') in STATES else None
