"""Route following: the next maneuver and the distance to it, between navd's 1 Hz updates.

navd publishes navInstruction once a second: maneuverDistance is the distance to the end of the current step,
and it runs down to about -10 m before navd advances to the next step. Between updates the distance is
dead-reckoned from carState.vEgo, which at highway speed is up to 30 m better than the last message.
Maneuver type and modifier are Mapbox/OSRM strings, copied through by navd unchanged.
A new route (a destination set from the phone, a reroute) can start with a maneuver of the same type and modifier as
the old one, so it is told apart by navd marking it (one invalid navInstruction between the two routes) or by the
distance jumping up by more than GPS noise or dead reckoning explain. Either way the old maneuver was dropped, not
passed, and RouteFollower.replaced says so for the one update: its turn pulse must not carry over to the new route.
"""
from dataclasses import dataclass

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

NEW_MANEUVER_JUMP_M = 10.  # navd's distance jumping up by more than this...
ADVANCE_NEAR_M = 5.        # ...from this close means it moved on to the next maneuver (it advances past -10 m),
                           # while a jump from further out is GPS noise
ROUTE_JUMP_M = 50.         # a jump up by more than this is a new maneuver from anywhere...
REPLACED_FROM_M = 15.      # ...and from further out than this, where navd never moves on (it does on a sharp turn's
                           # next step up to 10 m early), a new route
PASSED_NEAR_M = 30.        # a new maneuver means the last one was passed if it was this close
PASSED_DIST_M = 15.        # dead-reckoned this far past a maneuver navd hasn't advanced from, it counts as passed
NAV_STALE_S = 3.           # no navInstruction for this long, the route is lost
ARRIVE_DIST_M = 15.


@dataclass(frozen=True)
class Maneuver:
  kind: str
  side: int            # LEFT, RIGHT, or 0
  type: str
  modifier: str
  angle_deg: float     # unsigned nominal turn angle from the modifier; navInstruction carries no bearings
  lanes: tuple = ()    # ((directions, active), ...) left to right, from NavInstruction.lanes; empty when navd has none

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
  return tuple((tuple(str(d) for d in lane.directions), bool(lane.active)) for lane in instruction.lanes)


def maneuver_from_instruction(instruction) -> Maneuver:
  mtype, modifier = instruction.maneuverType, instruction.maneuverModifier
  kind, side = classify(mtype, modifier)
  return Maneuver(kind, side, mtype, modifier, NOMINAL_ANGLE_DEG.get(modifier, 0.), lanes_from_instruction(instruction))


class RouteFollower:
  def __init__(self):
    self.maneuver_id = 0
    self.reset()

  def reset(self):
    self.valid = False
    self.maneuver: Maneuver | None = None
    self.distance = float('inf')            # m to the next maneuver, dead-reckoned
    self.distance_remaining = float('inf')  # m to the destination
    self.passed = False                     # true for the one update in which the last maneuver was passed
    self.replaced = False                   # true for the one update in which a new route dropped the last maneuver
    self._passed_id = -1
    self._stale_t = 0.

  @property
  def upcoming(self) -> Maneuver | None:
    """The next maneuver, or None once it is passed and navd hasn't moved on yet."""
    return self.maneuver if self.valid and self._passed_id != self.maneuver_id else None

  @property
  def arrived(self) -> bool:
    return self.valid and self.maneuver is not None and self.maneuver.kind == ARRIVE and self.distance < ARRIVE_DIST_M

  def update(self, instruction, updated: bool, valid: bool, v_ego: float, dt: float):
    """instruction: the latest navInstruction; updated: it arrived since the last call; valid: its valid flag."""
    self.passed = self.replaced = False
    if updated and not valid:
      dropped = self.valid
      self.reset()
      if dropped:  # navd dropped the route: a new one comes next, or the car is off it
        self.maneuver_id += 1
        self.replaced = True
      return
    if updated:
      self._stale_t = 0.
      self._on_instruction(instruction)
    elif self.valid:
      self._stale_t += dt
      if self._stale_t > NAV_STALE_S:
        self.reset()
        return
      travelled = max(v_ego, 0.) * dt
      self.distance -= travelled
      self.distance_remaining -= travelled

    if self.valid and self.distance < -PASSED_DIST_M and self._passed_id != self.maneuver_id:
      self._passed_id = self.maneuver_id
      self.passed = True

  def _on_instruction(self, instruction):
    distance = float(instruction.maneuverDistance)
    maneuver = maneuver_from_instruction(instruction)
    prev = self.maneuver
    advanced = self.distance < ADVANCE_NEAR_M and distance > self.distance + NEW_MANEUVER_JUMP_M
    jumped = distance > self.distance + ROUTE_JUMP_M
    new = not self.valid or prev is None or advanced or jumped or (maneuver.type, maneuver.modifier) != (prev.type, prev.modifier)
    if new:
      self.replaced = self.valid and jumped and self.distance > REPLACED_FROM_M
      if self.valid and not self.replaced and self.distance < PASSED_NEAR_M and self._passed_id != self.maneuver_id:
        self.passed = True
      self.maneuver_id += 1
    self.valid = True
    self.maneuver = maneuver
    self.distance = distance
    self.distance_remaining = float(instruction.distanceRemaining)
