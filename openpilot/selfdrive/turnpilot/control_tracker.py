"""The route's stop signs, yields and traffic signals between navd's 1 Hz fixes (navInstruction.controls).

navd sends each control's distance once a second from the GPS projection onto the route. A projection at standstill is
pure noise (2 m of it walked the old junction hold 4 m over its line), and one a second jumps by the GPS error, so the
distance here is dead-reckoned with vEgo every frame, and a fresh fix pulls it at most CORRECTION_M (none while slower
than FREEZE_SPEED). A fix more than JUMP_M off is a re-initialisation, logged (turnpilot.control_jump), and the policy
re-checks whether the stop is still feasible (map_controls.py).

A control's identity is (routeCount, id), never the maneuver: a skipped turn or a straight-through junction keeps its
stop. A new route (a reroute) drops the controls the car wasn't stopping for yet, and keeps an engaged or held one: a
same-kind control of the new route within ADOPT_M of it takes over its state (it is the same one), and one the new route
doesn't list is carried under its old key until passed. A held control is never dropped and its distance is frozen:
the latch is the policy's, position-free.
A gap in the route (an invalid navInstruction, or none for NAV_STALE_S) is dead-reckoned through, for GAP_MAX_M of
travel; then every control but a held one is dropped, and listed in lost for the policy to say so.
"""
from dataclasses import dataclass
from typing import Any

from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.turnpilot.route_follower import NAV_STALE_S

CORRECTION_M = 0.5    # a fix moves the dead-reckoned distance at most this far
FREEZE_SPEED = 0.5    # m/s: slower than this, fixes are ignored
JUMP_M = 20.          # a fix this far off re-initialises the distance
ADOPT_M = 20.         # a new route's control this close to an engaged one of the same kind is that one
DROP_PAST_M = 15.     # a control this far behind the car is passed
GAP_MAX_M = 150.      # m dead-reckoned without navd before the controls are dropped


@dataclass
class TrackedControl:
  key: tuple[int, int]       # (routeCount, id)
  kind: str                  # 'stop', 'yield' or 'signal'
  d: float                   # m to the stop point, dead-reckoned
  line_mapped: bool = False
  junction: bool = True
  cross_suspect: bool = False
  engaged: bool = False      # the policy is stopping (or deciding) for it: carried through a reroute
  held: bool = False         # the policy's latch holds the car at it: never dropped, its distance frozen
  jumped: bool = False       # re-initialised by a jump this frame
  state: Any = None          # the policy's own state for it


def from_msg(route_count: int, c) -> TrackedControl:
  return TrackedControl((route_count, int(c.id)), str(c.kind), float(c.distance), bool(c.lineMapped), bool(c.junction),
                        bool(c.crossSuspect))


class ControlTracker:
  def __init__(self):
    self.controls: dict[tuple[int, int], TrackedControl] = {}
    self.route_count: int | None = None
    self.status = 'none'     # navInstruction.controlsStatus of the last valid message
    self.route = False       # on a route now: the last navInstruction was valid and fresh
    self.invalid = False     # the last navInstruction was invalid (off the route, rerouting)
    self.stale_t = 0.
    self.gap_m = 0.          # m travelled in a gap
    self.lost: list[TrackedControl] = []  # dropped this frame by a gap too long

  def update(self, nav, updated: bool, valid: bool, v_ego: float, dt: float) -> list[TrackedControl]:
    """nav: the latest navInstruction; updated: it arrived this frame; valid: its valid flag. The controls nearest first."""
    v = max(v_ego, 0.)
    self.lost = []
    for c in self.controls.values():
      c.jumped = False
      if not c.held:
        c.d -= v * dt
    if updated and valid:
      self.stale_t, self.gap_m, self.route, self.invalid = 0., 0., True, False
      self._fix(nav, v)
    else:
      self.stale_t = 0. if updated else self.stale_t + dt
      self.invalid = self.invalid or updated
      gap = self.invalid or self.stale_t > NAV_STALE_S
      if gap:
        self.route = False
        self.gap_m += v * dt
        if self.gap_m > GAP_MAX_M:
          self.lost = [c for c in self.controls.values() if not c.held]
          for c in self.lost:
            cloudlog.event("turnpilot.control_lost", kind=c.kind, key=list(c.key), engaged=c.engaged, gap_m=round(self.gap_m, 1))
          self.controls = {k: c for k, c in self.controls.items() if c.held}
    self.controls = {k: c for k, c in self.controls.items() if c.held or c.d >= -DROP_PAST_M}
    return sorted(self.controls.values(), key=lambda c: c.d)

  def _fix(self, nav, v: float) -> None:
    route_count = int(nav.routeCount)
    self.status = str(nav.controlsStatus)
    measured = {(route_count, int(c.id)): c for c in nav.controls}
    if route_count != self.route_count:
      old, self.controls = self.controls, {k: from_msg(route_count, c) for k, c in measured.items()}
      self.route_count = route_count
      for o in old.values():
        if not (o.engaged or o.held):
          continue
        same = [n for n in self.controls.values() if n.kind == o.kind and n.state is None and not n.engaged and
                abs(n.d - o.d) < ADOPT_M]
        if same:
          n = min(same, key=lambda n: abs(n.d - o.d))
          n.d, n.engaged, n.held, n.state = o.d, o.engaged, o.held, o.state
        else:
          self.controls[o.key] = o
      return
    for key, c in measured.items():
      t = self.controls.get(key)
      if t is None:
        self.controls[key] = from_msg(route_count, c)
        continue
      if t.held:
        continue
      err = float(c.distance) - t.d
      if abs(err) > JUMP_M:
        cloudlog.event("turnpilot.control_jump", kind=t.kind, key=list(key), d=round(t.d, 1), measured=round(float(c.distance), 1))
        t.d, t.jumped = float(c.distance), True
      elif v >= FREEZE_SPEED:
        t.d += min(max(err, -CORRECTION_M), CORRECTION_M)
    # out of navd's window (passed, beyond 400 m, past the 8 nearest): only the ones the car is stopping for stay
    self.controls = {k: c for k, c in self.controls.items() if k in measured or c.engaged or c.held}
