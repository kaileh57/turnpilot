"""Status for the phone: a thread reads cereal into a plain snapshot, and handlers format the latest one."""
import threading
import time

from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.navigation.helpers import latest_gps_source

SERVICES = ["deviceState", "navInstruction", "navRoute", "turnpilotPlan", "gpsLocation", "gpsLocationExternal"]
STALE_AFTER = 5.  # s without a new snapshot means the reader thread is stuck or dead

STATE_LABELS = {
  "disabled": "Off", "noRoute": "No route", "following": "Following route", "approaching": "Approaching maneuver",
  "turning": "Turning", "laneChange": "Changing lanes", "holding": "Holding", "arrived": "Arrived",
  "offline": "Not running",
}
MANEUVER_NAMES = {"new name": "continue", "end of road": "turn", "on ramp": "take ramp", "off ramp": "take exit",
                  "rotary": "roundabout", "roundabout turn": "roundabout"}


def read_services(sm) -> dict:
  """Plain values out of a SubMaster. Nothing else in the web app touches cereal."""
  def up(s):
    return sm.seen[s] and sm.alive[s]

  # navd sends navInstruction with valid=False when it has no route, and turnpilotd doesn't set valid at all
  ni, plan = sm["navInstruction"], sm["turnpilotPlan"]
  route = None
  if up("navInstruction") and sm.valid["navInstruction"]:
    route = {"text": ni.maneuverPrimaryText, "type": ni.maneuverType, "modifier": ni.maneuverModifier,
             "maneuver_m": ni.maneuverDistance, "remaining_m": ni.distanceRemaining, "remaining_s": ni.timeRemaining,
             "points": len(sm["navRoute"].coordinates) if sm.seen["navRoute"] else 0}
  source = latest_gps_source(sm)
  return {
    "at": time.monotonic(),
    "onroad": bool(up("deviceState") and sm["deviceState"].started),
    "route": route,
    "plan": {"state": str(plan.state), "reason": plan.reason, "shadow": plan.shadow} if up("turnpilotPlan") else None,
    "gps": (sm[source].latitude, sm[source].longitude) if source else None,
  }


def format_distance(meters: float, metric: bool) -> str:
  if metric:
    if meters < 1000:
      return f"{round(meters, -1):.0f} m"
    return f"{meters / 1000:.1f} km" if meters < 10000 else f"{meters / 1000:.0f} km"
  miles = meters / 1609.344
  if miles < 0.2:
    return f"{round(meters * 3.28084, -1):.0f} ft"
  return f"{miles:.1f} mi" if miles < 10 else f"{miles:.0f} mi"


def format_duration(seconds: float) -> str:
  minutes = round(seconds / 60)
  if minutes < 1:
    return "<1 min"
  return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h {minutes % 60:02d} min"


def maneuver_text(kind: str, modifier: str, road: str) -> str:
  """'turn', 'left', 'Main St' -> 'Turn left onto Main St'."""
  if kind == "arrive":
    return f"Arrive at {road}" if road else "Arrive"
  words = " ".join(w for w in (MANEUVER_NAMES.get(kind, kind), modifier) if w).capitalize()
  return f"{words} onto {road}" if words and road else words or road


def format_status(snap: dict | None, destination, enabled: bool, metric: bool, now: float | None = None) -> dict:
  """The /api/status body. snap comes from read_services, destination from Store.destination."""
  now = time.monotonic() if now is None else now
  fresh = snap is not None and now - snap["at"] < STALE_AFTER
  if snap is None or not fresh:
    snap = {"onroad": False, "route": None, "plan": None, "gps": None}
  route, plan = snap["route"], snap["plan"]
  return {
    "reading": fresh,
    "onroad": snap["onroad"],
    "destination": destination,
    "navigating": route is not None,
    "maneuver": None if route is None else {
      "text": maneuver_text(route["type"], route["modifier"], route["text"]),
      "distance": format_distance(route["maneuver_m"], metric),
    },
    "remaining": None if route is None else {
      "distance": format_distance(route["remaining_m"], metric),
      "duration": format_duration(route["remaining_s"]),
      "seconds": route["remaining_s"],  # the phone adds this to its own clock to get an arrival time
    },
    "turnpilot": {
      "enabled": enabled,
      "state": plan["state"] if plan else "offline",
      "label": STATE_LABELS.get(plan["state"], plan["state"]) if plan else STATE_LABELS["offline"],
      "reason": plan["reason"] if plan else "turnpilotd is not running",
      "shadow": plan["shadow"] if plan else False,
    },
    "gps": snap["gps"] and {"latitude": snap["gps"][0], "longitude": snap["gps"][1]},
  }


class StatusReader(threading.Thread):
  def __init__(self):
    super().__init__(name="status", daemon=True)
    self.lock = threading.Lock()
    self.snapshot = None
    self.stopped = threading.Event()

  def stop(self):
    self.stopped.set()

  def get(self):
    with self.lock:
      return self.snapshot

  def run(self):
    try:
      # imported here so a missing messaging build leaves the search and destination working
      from openpilot.cereal import messaging
      sm = messaging.SubMaster(SERVICES)
    except Exception:
      cloudlog.exception("turnpilot_web.status_unavailable")
      return
    while not self.stopped.is_set():
      try:
        sm.update(500)
        snap = read_services(sm)
        with self.lock:
          self.snapshot = snap
      except Exception:
        cloudlog.exception("turnpilot_web.status_failed")
        time.sleep(1)
