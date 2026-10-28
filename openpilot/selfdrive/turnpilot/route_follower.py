"""Route following: the next maneuver and the distance to it, between navd's 1 Hz updates.

navd publishes navInstruction once a second: maneuverDistance is the distance to the end of the current step,
and it runs down to about -10 m before navd advances to the next step. Between updates the distance is
dead-reckoned from carState.vEgo, which at highway speed is up to 30 m better than the last message.
Maneuver type and modifier are Mapbox/OSRM strings, copied through by navd unchanged.
What counts as a turn: only a junction maneuver, where the route leaves the road at an intersection that other roads
meet (navInstruction.maneuverBranches > 0, from Valhalla's intersection bearings). Where the road just bends and no
other road meets it, the maneuver is NONE: the driving model follows the bend (0.11.3's Autospeed slows for curves), and
TurnPilot sends no desire and no speed cap there. navd moves past such a step only 10 m beyond its end, so the turn or
keep after it (RouteFollower.ahead, from allManeuvers with its junction) is looked ahead to for its speed cap: a junction
turn 30-50 m past a bend was otherwise seen at road speed, too late to slow for, and skipped. Its desire waits until the
car is past the step: a turn desire before a straight-on junction would turn the car there. That is when the step is
dead-reckoned NAVD_ADVANCE_M past, where navd moves on, not up to a second later at navd's next update: the turn after
it becomes the maneuver then, under its own id, which navd's update moving on to it keeps (an update still on the step
only brings the turn's distance). Even so a turn under turn_skip.MIN_FIRST_PULSE_M past the step is skipped out loud.
The angle is the real one, bearing_after less bearing_before. A route
without intersection data (maneuverGeometryValid false; Valhalla always has it) leaves every turn a junction at the
modifier's nominal angle, which is logged once per maneuver and shown in the plan's reason.
A new route (a destination set from the phone, a reroute) can start with a maneuver of the same type and modifier as
the old one, so it is told apart by navd marking it (one invalid navInstruction between the two routes, and a new
routeCount) or by the distance jumping up by more than GPS noise or dead reckoning explain. Either way the old maneuver
was dropped, not passed, and RouteFollower.replaced says so for the one update: its turn pulse must not carry over to the
new route.
A committed turn (turnpilotd: its turn desire is out) is never dropped that way (Kellen: never cancel a turn once
committed, and a car slowing for a turn must not speed up into it). The junction is where it was whatever navd says:
- an invalid navInstruction for the same route (navd off the step for a fix, rerouting, the destination removed) or navd
  going quiet past NAV_STALE_S is a gap: the maneuver stays, dead-reckoned, under the same id, until navd's next valid
  update carries on with it or the turn is over (cleared at its angle, or dead-reckoned PASSED_DIST_M past it);
- navd on another step (not same_junction) is a gap too, however close in: navd moving on early to the next step (a
  sharp turn's, up to 10 m before its end, or a GPS jump forward) doesn't mean the car has made the turn. It is over
  only on real evidence: the car has turned it (DesireTiming clears it at its angle) or is dead-reckoned PASSED_DIST_M
  past it. Until then its id and its cap stay and no CLEAR goes; navd's next update after that takes up the next step;
- a jump in the distance to the same junction is GPS noise: the dead-reckoned distance stands;
- a new route (a new routeCount) whose first maneuver is the same junction (same_junction) within SAME_TURN_M of the
  dead-reckoned one is the same maneuver (a reroute onto the road the car is on, its origin snapped up to a reroute's
  25 m off), still where it was dead-reckoned. A new route that doesn't turn there (a reroute whose origin snapped past
  the junction, a destination changed from the phone) is a gap: the turn is held to its end, and the new route taken up
  after.
After a reroute or a jump navd's distance to the junction can stay off by that much: the turn's cap would lift, or the
turn read as passed, before the car gets there. So from then on that junction is dead-reckoned, while the turn is
committed and after it while navd is still on its step (by its bearings), and never taken up again as a new maneuver.
"""
from dataclasses import dataclass

from openpilot.common.swaglog import cloudlog

LEFT, RIGHT = 1, -1

# kinds of maneuver, by what TurnPilot does about them
TURN = "turn"      # a junction turn: turn desire, turn speed cap, lane choice
KEEP = "keep"      # slight turns, forks, ramps and exits: keep desire
UTURN = "uturn"    # the driving model can't U-turn; nothing is sent
ARRIVE = "arrive"
NONE = "none"

TURN_MODIFIERS = {"left": LEFT, "sharp left": LEFT, "right": RIGHT, "sharp right": RIGHT}
SLIGHT_MODIFIERS = {"slight left": LEFT, "slight right": RIGHT}
KEEP_TYPES = ("fork", "off ramp")  # a keep whatever the modifier says: their branches part gently
# with a slight modifier, nothing: the road itself bends that way. A keep desire there swerves the car, at highway speed
CONTINUE_TYPES = ("continue", "new name")
RAMP_TYPES = ("fork", "off ramp", "on ramp", "merge")
NO_ACTION_TYPES = ("depart", "notification", "merge", "roundabout", "rotary", "roundabout turn",
                   "exit roundabout", "exit rotary")
NOMINAL_ANGLE_DEG = {"slight left": 45., "slight right": 45., "left": 90., "right": 90.,
                     "sharp left": 135., "sharp right": 135., "uturn": 180.}

SIDE_MIN_DEG = 20.         # a heading change under this has no side to check the modifier against

NEW_MANEUVER_JUMP_M = 10.  # navd's distance jumping up by more than this...
ADVANCE_NEAR_M = 5.        # ...from this close means it moved on to the next maneuver (it advances past -10 m),
                           # while a jump from further out is GPS noise
ROUTE_JUMP_M = 50.         # a jump up by more than this is a new maneuver from anywhere...
REPLACED_FROM_M = 15.      # ...and from further out than this, where navd never moves on (it does on a sharp turn's
                           # next step up to 10 m early), a new route
PASSED_NEAR_M = 30.        # a new maneuver means the last one was passed if it was this close
PASSED_DIST_M = 15.        # dead-reckoned this far past a maneuver navd hasn't advanced from, it counts as passed
NAVD_ADVANCE_M = 10.       # navd moves past a step this far beyond its end (navd.py MANEUVER_TRANSITION_THRESHOLD)
NAV_STALE_S = 3.           # no navInstruction for this long, the route is lost
SAME_TURN_M = 40.          # a committed turn is the first maneuver of a new route this close to where it was dead-reckoned
                           # (navd reroutes 25 m off the route; the next junction the same way is a block on)
SAME_BEARING_DEG = 15.     # the same junction's bearings, from two routes or two fixes, are this close
ARRIVE_DIST_M = 15.


@dataclass(frozen=True)
class Maneuver:
  kind: str
  side: int            # LEFT, RIGHT, or 0
  type: str
  modifier: str
  angle_deg: float     # unsigned turn angle: from the junction's bearings, else the modifier's nominal one
  lanes: tuple = ()    # ((directions, active), ...) left to right, from NavInstruction.lanes; empty when navd has none
  geometry: bool = False  # angle_deg and the junction test came from the route's bearings
  bend: bool = False      # the road bends here and no other road meets it: left to the model (kind NONE)
  bearing_side: int = 0   # LEFT or RIGHT as the bearings turn, 0 without them or under SIDE_MIN_DEG
  bearings: tuple = ()    # (bearing_before, bearing_after) with geometry
  opposing: str = 'unknown'  # NavInstruction.Opposing: a road straight ahead whose traffic a far-side turn crosses

  @property
  def ramp(self) -> bool:
    return self.type in RAMP_TYPES


def classify(mtype: str, modifier: str) -> tuple[str, int]:
  """(kind, side) of an OSRM maneuver type and modifier."""
  if mtype == "arrive":
    return ARRIVE, TURN_MODIFIERS.get(modifier, 0)
  if mtype in NO_ACTION_TYPES or modifier in ("", "straight"):
    return NONE, 0
  if modifier == "uturn":
    return UTURN, 0
  if mtype in CONTINUE_TYPES and modifier in SLIGHT_MODIFIERS:
    return NONE, 0
  side = TURN_MODIFIERS.get(modifier) or SLIGHT_MODIFIERS.get(modifier, 0)
  if side == 0:
    return NONE, 0
  if mtype in KEEP_TYPES or modifier in SLIGHT_MODIFIERS:
    return KEEP, side
  return TURN, side


def lanes_from_instruction(instruction) -> tuple:
  return lanes_from_list(instruction.lanes)


def lanes_from_list(lanes) -> tuple:
  return tuple((tuple(str(d) for d in lane.directions), bool(lane.active)) for lane in lanes)


def turn_angle(bearing_before: float, bearing_after: float) -> float:
  """Signed heading change in degrees, (-180, 180]: positive to the right (bearings run clockwise)."""
  return -((bearing_before - bearing_after + 180.) % 360. - 180.)


def maneuver_from_instruction(instruction) -> Maneuver:
  return make_maneuver(instruction.maneuverType, instruction.maneuverModifier, lanes_from_instruction(instruction),
                       instruction.maneuverGeometryValid, instruction.maneuverBearingBefore, instruction.maneuverBearingAfter,
                       instruction.maneuverBranches, str(instruction.maneuverOpposing))


def maneuver_from_ahead(ahead) -> Maneuver:
  """A Maneuver from a NavInstruction.Maneuver (allManeuvers), which has no lanes."""
  return make_maneuver(ahead.type, ahead.modifier, (), ahead.geometryValid, ahead.bearingBefore, ahead.bearingAfter, ahead.branches,
                       str(ahead.opposing))


def make_maneuver(mtype: str, modifier: str, lanes: tuple, geometry_valid: bool, bearing_before: float, bearing_after: float,
                  branches: int, opposing: str = 'unknown') -> Maneuver:
  kind, side = classify(mtype, modifier)
  if not geometry_valid:
    return Maneuver(kind, side, mtype, modifier, NOMINAL_ANGLE_DEG.get(modifier, 0.), lanes)  # opposing unknown
  angle = turn_angle(bearing_before, bearing_after)
  bearing_side = 0 if abs(angle) < SIDE_MIN_DEG else RIGHT if angle > 0 else LEFT
  bearings = (float(bearing_before), float(bearing_after))
  if kind in (TURN, KEEP) and branches == 0:
    return Maneuver(NONE, 0, mtype, modifier, abs(angle), lanes, geometry=True, bend=True, bearing_side=bearing_side, bearings=bearings,
                    opposing=opposing)
  return Maneuver(kind, side, mtype, modifier, abs(angle), lanes, geometry=True, bearing_side=bearing_side, bearings=bearings,
                  opposing=opposing)


def same_bearings(a: Maneuver, b: Maneuver) -> bool:
  """Both have junction geometry, and it is the same junction's: bearings before and after within SAME_BEARING_DEG.
  Two junctions in a row the same way can't share them: the heading changes at the first."""
  return a.geometry and b.geometry and len(a.bearings) == len(b.bearings) == 2 and \
    all(abs(turn_angle(x, y)) <= SAME_BEARING_DEG for x, y in zip(a.bearings, b.bearings, strict=True))


def same_junction(a: Maneuver, b: Maneuver) -> bool:
  """The same maneuver at the same junction, as far as the route says: type and modifier, and the bearings when both
  have them."""
  return (a.type, a.modifier) == (b.type, b.modifier) and (same_bearings(a, b) or not (a.geometry and b.geometry))


class RouteFollower:
  def __init__(self, maneuver_id: int = 0):
    """maneuver_id: where the ids start. A follower that replaces another one (turnpilotd after a fault) starts past the
    old one's, so no id repeats: what was latched for an old maneuver (ScenePolicy's commitment) can't match a new one."""
    self.maneuver_id = maneuver_id
    self.route_count = 0  # navInstruction.routeCount of the last valid update
    self.reset()

  def reset(self):
    self.valid = False
    self.maneuver: Maneuver | None = None
    self.distance = float('inf')            # m to the next maneuver, dead-reckoned
    self.distance_remaining = float('inf')  # m to the destination
    self.passed = False                     # true for the one update in which the last maneuver was passed
    self.replaced = False                   # true for the one update in which a new route dropped the last maneuver
    self.through_lanes: tuple = ()          # NavInstruction.throughLanes: the next intersection the route goes straight through
    self.through_distance = float('inf')    # m to it, dead-reckoned
    self.ahead: Maneuver | None = None      # past a NONE maneuver, the next turn or keep (allManeuvers), for its speed cap
    self.ahead_distance = float('inf')      # m to it, dead-reckoned
    self._passed_id = -1
    self._stale_t = 0.
    self._moved_on_from: tuple[str, str] | None = None  # (type, modifier) of the NONE step left for the turn after it
    self._held_id = -1  # a committed maneuver navd's distance was off for (a reroute, GPS noise): dead-reckoned to its end

  @property
  def upcoming(self) -> Maneuver | None:
    """The next maneuver, or None once it is passed and navd hasn't moved on yet."""
    return self.maneuver if self.valid and self._passed_id != self.maneuver_id else None

  @property
  def arrived(self) -> bool:
    return self.valid and self.maneuver is not None and self.maneuver.kind == ARRIVE and self.distance < ARRIVE_DIST_M

  def update(self, instruction, updated: bool, valid: bool, v_ego: float, dt: float, committed: bool = False):
    """instruction: the latest navInstruction; updated: it arrived since the last call; valid: its valid flag;
    committed: the current maneuver's turn desire is out, so it is held through a gap in the route (module docstring)."""
    self.passed = self.replaced = False
    if updated and not valid:
      if committed and self.valid:
        updated = False  # a gap: dead-reckoned on as between updates
      else:
        dropped = self.valid
        self.reset()
        if dropped:  # navd dropped the route: a new one comes next, or the car is off it
          self.maneuver_id += 1
          self.replaced = True
        return
    if updated:
      self._stale_t = 0.
      self._on_instruction(instruction, committed, max(v_ego, 0.) * dt)
    elif self.valid:
      self._stale_t += dt
      if self._stale_t > NAV_STALE_S and not committed:  # navd went quiet: the maneuver is over, as when navd drops the
        self.reset()  # route, but not replaced (a turn under way finishes and its desire is cleared as for a pass)
        self.maneuver_id += 1
        return
      travelled = max(v_ego, 0.) * dt
      self.distance -= travelled
      self.distance_remaining -= travelled
      self.through_distance -= travelled
      self.ahead_distance -= travelled

    if self.valid and self.maneuver is not None and self.maneuver.kind == NONE and self.ahead is not None and \
       self.distance < -NAVD_ADVANCE_M:
      self._move_on()
    if self.valid and self.distance < -PASSED_DIST_M and self._passed_id != self.maneuver_id:
      self._passed_id = self.maneuver_id
      self.passed = True

  @staticmethod
  def _log_new(maneuver: Maneuver, distance: float):
    """A log line for each maneuver TurnPilot would act on but has no junction data for, and for a turn's bearings
    that disagree with its modifier (the side comes from the modifier)."""
    if maneuver.kind not in (TURN, KEEP):
      return
    if not maneuver.geometry:
      cloudlog.warning(f"turnpilot: no junction data for {maneuver.type} {maneuver.modifier} {distance:.0f} m ahead, " +
                       f"taken as a junction at the nominal {maneuver.angle_deg:.0f} deg")
    elif maneuver.bearing_side and maneuver.bearing_side != maneuver.side:
      cloudlog.warning(f"turnpilot: {maneuver.type} {maneuver.modifier} {distance:.0f} m ahead, but its bearings turn " +
                       f"{maneuver.angle_deg:.0f} deg the other way")

  def _move_on(self):
    """Past a NONE step where navd moves on from it: the turn or keep after it is the maneuver now (module docstring)."""
    self._moved_on_from = (self.maneuver.type, self.maneuver.modifier)
    self.maneuver, self.distance = self.ahead, self.ahead_distance
    self.ahead, self.ahead_distance = None, float('inf')
    self.maneuver_id += 1

  @staticmethod
  def _following(instruction) -> tuple[Maneuver | None, float]:
    """Past a NONE step, the next turn or keep in allManeuvers and its distance."""
    for ahead in list(instruction.allManeuvers)[1:]:
      following = maneuver_from_ahead(ahead)
      if following.kind != NONE:
        return (following, float(ahead.distance)) if following.kind in (TURN, KEEP) else (None, float('inf'))
    return None, float('inf')

  def _on_instruction(self, instruction, committed: bool, travelled: float):
    distance = float(instruction.maneuverDistance)
    maneuver = maneuver_from_instruction(instruction)
    prev = self.maneuver
    route_count, new_route = int(instruction.routeCount), self.valid and int(instruction.routeCount) != self.route_count
    if self._moved_on_from is not None:
      if not new_route and maneuver.kind == NONE and (maneuver.type, maneuver.modifier) == self._moved_on_from:
        # navd still on the step the car is past: the turn after it, by navd's distance to it when it's that one
        following, following_distance = self._following(instruction)
        same = following is not None and prev is not None and (following.type, following.modifier) == (prev.type, prev.modifier)
        self.distance = following_distance if same else self.distance - travelled
        self.distance_remaining = float(instruction.distanceRemaining)
        return
      self._moved_on_from = None
    advanced = self.distance < ADVANCE_NEAR_M and distance > self.distance + NEW_MANEUVER_JUMP_M
    jumped = distance > self.distance + ROUTE_JUMP_M
    same = self.valid and prev is not None and (maneuver.type, maneuver.modifier) == (prev.type, prev.modifier)
    here = self.valid and prev is not None and same_junction(maneuver, prev)
    if committed and self.valid and prev is not None:
      if not here or (new_route and abs(distance - self.distance) > SAME_TURN_M):
        # navd on another step (moved on early, a new route that doesn't turn there): a gap, the turn held to its end
        # (module docstring), navd's updates new until then
        self.distance -= travelled
        return
      if new_route or advanced or jumped:
        self._held_id = self.maneuver_id  # a reroute onto the road the car is on, or GPS noise: the same turn
    if here and self._held_id == self.maneuver_id and (committed or (not new_route and same_bearings(maneuver, prev))):
      # the junction is where it was: while the turn is committed, and after it to its pass while navd is still on its
      # step (by its bearings: without them the next step may be another turn the same way)
      advanced, jumped, distance = False, False, self.distance - travelled
    self.route_count = route_count
    new = not same or advanced or jumped or (new_route and not committed)
    if new:
      self._log_new(maneuver, distance)
      self.replaced = self.valid and (new_route or (jumped and self.distance > REPLACED_FROM_M))
      if self.valid and not self.replaced and self.distance < PASSED_NEAR_M and self._passed_id != self.maneuver_id:
        self.passed = True
      self.maneuver_id += 1
    self.valid = True
    self.maneuver = maneuver
    self.distance = distance
    self.distance_remaining = float(instruction.distanceRemaining)
    self.through_lanes = lanes_from_list(instruction.throughLanes)
    self.through_distance = float(instruction.throughLanesDistance) if self.through_lanes else float('inf')
    self.ahead, self.ahead_distance = None, float('inf')
    if maneuver.kind == NONE:  # a bend or a road going on: the turn after it may be close
      self.ahead, self.ahead_distance = self._following(instruction)
