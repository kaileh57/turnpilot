"""Lane planner: get into the lane a turn leaves from, one nav lane change at a time, each warned of, confirmed and
checked until it starts.

Which lane the car is in comes from what the model sees: lanes either side counted from the room between the
ego lane's line and the road edge, at about 3.6 m a lane, but only where the next line out is painted (room
alone counts shoulders and parking strips as lanes). modelV2 can't tell a yellow line from a white one, and on
an undivided road its far road edge lies beyond the oncoming lanes; the lane paint read off the road camera
(scene/lane_paint.py, one entry per modelV2 lane line) can. US markings say what is beyond a line: broken white
is a lane going the same way, solid white an edge, shoulder, bike or parking lane, yellow oncoming traffic or a
divided road's left edge.

Where to go (desired_changes): with the map's lane list (NavInstruction.lanes) the car's lane is counted from the curb
side, which is always its own direction's, and lanes beyond a yellow line count 0. That count is known only where the model
can make it: a curb lane often has no edge line, and one lane in from it the model sees room for a lane but no line, so it
can't tell the curb lane from the one beside it. The changes made for the maneuver are counted too (LanePlanner.done), and
the lane dead reckoned from where the first of them started: no more changes than the lane list needs from there, and none
from a lane it can't place, which is said out loud (lane_unknown: turnpilotPlan.laneMismatch laneUnknown) instead. On the
far side from the curb only the list's own lanes are lanes (nav_room): on an undivided road the model's far road edge lies
beyond the oncoming lanes (BeamNG, 2026-10-27: a second change left across a double yellow into the oncoming lane). Without it (the usual case: OSM tags
lanes on few roads) an exit moves toward the curb by the model's lane count, but not across a curb-side line read solid
white, and a turn, or an exit away from the curb, moves one lane toward its side at a time, looked at again after each
change.

Whether the next change may go (change_blocker), on every path, map lanes or not: only into a valid travel lane, which
means all of
- the ego line that way reads broken white at BROKEN_WHITE_CONF_MIN or better: solid, yellow, no line, unknown, or no
  paint read at all (no detector, the wide camera, a stale scene) is no change. A yellow read keeps vetoing for
  YELLOW_HOLD_S and YELLOW_HOLD_M after the last one, so a worn stretch doesn't open a gap; the hold moves with the lines
  in a lane change, and covers all four lines when the direction of one is unknown;
- the model sees a lane that way, its next line out at least LANE_MIN lane widths beyond the ego line (an 8 ft parking
  strip beside an 11 ft lane is 0.73, a 6 ft bike lane beside a 10 ft one 0.61), not beyond a yellow line;
- nothing stopped in that lane ahead (parked cars: a parking strip);
- under US markings (MUTCD) only: in left-hand traffic the center line is broken white, so nothing goes away from the curb
  there, and nothing tells right-hand Europe (white center lines too) from the US.
Before the paint, a change toward the curb before a junction turn put the car in a parking strip (BeamNG, 2026-10-27).

How a change goes (LanePlanner): once the same change is wanted for LC_CONFIRM_FRAMES, the driver is warned LC_WARN_S
ahead: laneChangeOffer says the side (selfdrived raises turnpilotLaneChangeLeft/Right, a chime) and laneChangeCountdown the
seconds left. On cars where openpilot can't signal (the Tesla: opendbc has no blinker command), the driver's stalk is the
go-ahead, at any point in the countdown or up to LC_SIGNAL_TIMEOUT_S after it; elsewhere the request goes when the
countdown ends. The request stands in for the nudge (and, where openpilot signals, the blinker) and DesireHelper still
waits for the car's own blind spot signal (Tesla DAS_status). Until DesireHelper starts the change, every frame checks it
again, and any failure aborts it, with laneChangeAbort saying why for ABORT_HOLD_S (selfdrived raises
turnpilotLaneChangeAborted): the line no longer broken white or the lane no longer valid, the blind spot occupied, a car in
the target lane, no signal in time (or the blinker off before the change started), the route changed, the change no
longer needed, the car too slow or too close to the maneuver, DesireHelper waiting too long. Once the change has started
nothing cancels it: the car is already moving over. A change the driver makes alone gets the same cooldown before the next
offer, for the lines to settle. TurnPilotLanePaintActive off plans no change at all; turnpilotd then publishes what this
would have asked for in laneChangePaintRule.
"""
import math
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import NamedTuple

from openpilot.cereal import custom, log
from openpilot.selfdrive.controls.lib.desire_helper import LANE_CHANGE_SPEED_MIN
from openpilot.selfdrive.turnpilot.hooks import ABORT_HOLD_S, ABORT_TEXT
from openpilot.selfdrive.turnpilot.route_follower import KEEP, LEFT, RIGHT, TURN
from openpilot.selfdrive.turnpilot.scene.lane_paint import UNKNOWN, LinePaint

LaneChange = custom.TurnpilotPlan.LaneChange
LaneMismatch = custom.TurnpilotPlan.LaneMismatch
LaneChangeState = log.LaneChangeState

LANE_WIDTH = 3.6
SAMPLE_IDX = 8              # modelV2 point ~12 m ahead (X_IDXS[8])
LINE_PROB_MIN = 0.5
EDGE_STD_MAX = 1.2          # a road edge is trusted when its std is under this
SHOULDER_LANES = 0.75       # room narrower than this many lane widths beyond the ego line is a shoulder
LANE_MIN = 0.85             # a lane beside the ego lane has its next line out this many lane widths away or more (to 1.5)
KEEP_MIN_ROOM_M = 1.0       # a keep desire toward a trusted edge closer than this beyond the ego line drives off the road
BROKEN_WHITE_CONF_MIN = 0.6  # the ego line that way read broken white at this confidence or better allows a change
SOLID_WHITE_CONF_MIN = 0.6   # a curb-side line read solid white at this confidence stops an exit's changes across it
YELLOW_CONF_MIN = 0.3        # a line read yellow at this confidence vetoes changes across it (the cautious answer, so a lower bar)
YELLOW_HOLD_S, YELLOW_HOLD_M = 5., 100.  # and keeps vetoing until this long and this far after its last yellow read
STOPPED_V = 2.               # m/s: an object this slow in the target lane, up to STOPPED_X_M ahead, makes it no travel lane
STOPPED_X_M = 80.

LC_START_S, LC_START_MIN_M, LC_START_MAX_M = 20., 150., 400.  # lane changes start this far before the maneuver...
LC_END_S, LC_END_MIN_M, LC_END_MAX_M = 6., 60., 120.          # ...and none start closer than this
LC_EXTRA_LANE_S = 8.        # each change past the first starts this much earlier (an exit across a highway)
LC_SPEED_MARGIN = 0.5       # m/s above DesireHelper's minimum, so a request never waits below it
LC_CONFIRM_FRAMES = 10      # the same need this many frames in a row before the warning
LC_WARN_S = 3.              # the warning (chime, countdown) comes this long before the change may go
LC_SIGNAL_TIMEOUT_S = 8.    # after the countdown, the driver has this long to signal before the change is aborted
LC_COOLDOWN_S = 3.          # after any lane change, ours or the driver's, for the model's lane lines to settle
LC_ABORT_COOLDOWN_S = 6.    # after an abort, before the same change is offered again
LC_REQUEST_TIMEOUT_S = 8.   # DesireHelper waiting this long (blindspot) aborts the request
# an abort once the request is out waits this long, the request withdrawn, to see whether DesireHelper started the change
# already: modeld runs it on the plan it last had, and turnpilotd sees its state a modelV2 or two later
LC_ABORT_SETTLE_S = 0.2
LC_BLOCK_X_M = 15.          # a vehicle in the target lane within this, ahead or behind, blocks a request...
LC_CLOSING_T_S = 3.         # ...as does one that closes to that distance within this
LC_LANE_MARGIN_M = 0.5

LANE_UNKNOWN_REASON = "can't confirm lane"

SIDE_DIRECTIONS = {LEFT: ("left", "slightLeft"), RIGHT: ("right", "slightRight")}
OPENPILOT_SIGNALS_BRANDS = ("hyundai",)  # ports that forward CarControl blinkers (CAN-FD); everyone else is prompted


@dataclass(frozen=True)
class ModelLanes:
  left: int = 0               # lanes the model sees beside the ego lane
  right: int = 0
  keep_left_ok: bool = False  # a keep desire that way has a lane line and road beside it
  keep_right_ok: bool = False
  width: float = LANE_WIDTH
  measured: bool = field(default=False, compare=False)  # the model reads both ego lane lines: the counts mean something
  # the count that way is what is there: every lane counted from a painted line and a trusted edge, or no room for one. Not
  # where the next line out is unpainted but the trusted edge leaves room for a lane (a curb lane with no edge line, seen
  # from the lane beside it), nor where the edge is unsure (the paint alone says at least one)
  left_known: bool = field(default=True, compare=False)
  right_known: bool = field(default=True, compare=False)


def _side(ego_y, ego_prob, outer_y, outer_prob, edge_y, edge_std, sign, width):
  """(lanes counted on one side, keep ok that way, the count known: ModelLanes); sign is +1 for the right, -1 for the left
  (modelV2 y is right)."""
  room = sign * (edge_y - ego_y)
  painted = outer_prob > LINE_PROB_MIN and LANE_MIN * width <= sign * (outer_y - ego_y) <= 1.5 * width
  edge_ok = edge_std < EDGE_STD_MAX
  if not painted or (edge_ok and room < SHOULDER_LANES * width):
    lanes = 0
  else:
    lanes = max(int(math.floor(room / width + 0.25)), 1) if edge_ok else 1
  keep_ok = ego_prob > LINE_PROB_MIN and not (lanes == 0 and edge_ok and room < KEEP_MIN_ROOM_M)
  known = edge_ok and (painted or room < SHOULDER_LANES * width)
  return lanes, keep_ok, known


def model_lanes(line_ys, line_probs, edge_ys, edge_stds) -> ModelLanes:
  """From modelV2 laneLines [far left, left, right, far right] and roadEdges [left, right] at one distance ahead."""
  measured = line_ys[2] - line_ys[1]
  width = measured if min(line_probs[1], line_probs[2]) > LINE_PROB_MIN and 2.5 <= measured <= 4.8 else LANE_WIDTH
  left, keep_left, left_known = _side(line_ys[1], line_probs[1], line_ys[0], line_probs[0], edge_ys[0], edge_stds[0], -1., width)
  right, keep_right, right_known = _side(line_ys[2], line_probs[2], line_ys[3], line_probs[3], edge_ys[1], edge_stds[1], 1., width)
  measured = min(line_probs[1], line_probs[2]) > LINE_PROB_MIN
  return ModelLanes(left, right, keep_left, keep_right, width, measured, left_known, right_known)


def model_lanes_from_msg(model) -> ModelLanes:
  lines, edges = model.laneLines, model.roadEdges
  if len(lines) < 4 or len(edges) < 2 or len(model.laneLineProbs) < 4 or len(model.roadEdgeStds) < 2 or \
     any(len(e.y) <= SAMPLE_IDX for e in list(lines) + list(edges)):
    return ModelLanes(left_known=False, right_known=False)
  return model_lanes([ln.y[SAMPLE_IDX] for ln in lines], list(model.laneLineProbs),
                     [e.y[SAMPLE_IDX] for e in edges], list(model.roadEdgeStds))


def ego_lane_index(lane_count: int, lanes: ModelLanes, right_hand_traffic: bool) -> int | None:
  """The car's lane in a nav lane list of lane_count, from the left, counted from the curb side (always its own
  direction's): modelV2's far road edge on an undivided road lies beyond the oncoming lanes. None when the model's count on
  the curb side isn't known (ModelLanes.right_known)."""
  if not (lanes.right_known if right_hand_traffic else lanes.left_known):
    return None
  idx = lane_count - 1 - lanes.right if right_hand_traffic else lanes.left
  return min(max(idx, 0), lane_count - 1)


def _nav_lanes(maneuver) -> tuple:
  """The maneuver's lane list when the lane planner goes by it: a turn's or an exit's, of two lanes or more."""
  if maneuver is None or not (maneuver.kind == TURN or (maneuver.kind == KEEP and maneuver.type == "off ramp")):
    return ()
  return maneuver.lanes if len(maneuver.lanes) >= 2 else ()


def _valid_lanes(maneuver, nav) -> list[int]:
  return ([i for i, (_, active) in enumerate(nav) if active] or
          [i for i, (dirs, _) in enumerate(nav) if any(d in SIDE_DIRECTIONS[maneuver.side] for d in dirs)] or
          [0 if maneuver.side == LEFT else len(nav) - 1])


def nav_lane_span(lane_count: int, lanes: ModelLanes, right_hand_traffic: bool, done: int | None = 0, start=None) -> tuple[int, int]:
  """(first, last): the lanes of a nav lane list of lane_count, from the left, the car may be in. done: lane changes
  completed for this maneuver, positive left, None when one went a way that wasn't seen; start: (lane, done) where the
  first change of ours for it started, None before. From the model's count (ego_lane_index) and the lane dead reckoned
  from start, each within where done changes can have taken the car from anywhere in the list; where they disagree the car
  may be in either, and with neither, anywhere done allows. lanes: paint_lanes' already."""
  lo, hi = (max(-done, 0), min(lane_count - 1 - done, lane_count - 1)) if done is not None else (0, lane_count - 1)
  if lo > hi:  # more changes one way than the list has lanes: it doesn't describe the road here
    lo, hi = 0, lane_count - 1
  found = []
  idx = ego_lane_index(lane_count, lanes, right_hand_traffic)
  if idx is not None:
    found.append(idx)
  if start is not None and done is not None:
    found.append(start[0] - (done - start[1]))
  found = [min(max(i, lo), hi) for i in found]
  return (min(found), max(found)) if found else (lo, hi)


def nav_room(maneuver, lanes: ModelLanes, direction: int, right_hand_traffic: bool = True, paint=(), done: int | None = 0,
             start=None) -> int | None:
  """Lanes of the maneuver's lane list beyond the car toward direction (LEFT or RIGHT), wherever in it the car may be
  (nav_lane_span): the only lanes that way known to go its way. None without a lane list."""
  nav = _nav_lanes(maneuver)
  if not nav or direction not in (LEFT, RIGHT):
    return None
  lo, hi = nav_lane_span(len(nav), paint_lanes(lanes, paint), right_hand_traffic, done, start)
  return lo if direction == LEFT else len(nav) - 1 - hi


def lane_unconfirmed(maneuver, lanes: ModelLanes, right_hand_traffic: bool = True, paint=(), done: int | None = 0, start=None) -> bool:
  """The maneuver has a lane list, and the car may be in a lane it doesn't leave from or in one it does, which can't be
  told apart (nav_lane_span): no change is planned, and the driver is told (LanePlanner.lane_unknown)."""
  nav = _nav_lanes(maneuver)
  if not nav:
    return False
  lo, hi = nav_lane_span(len(nav), paint_lanes(lanes, paint), right_hand_traffic, done, start)
  valid = _valid_lanes(maneuver, nav)
  return lo < hi and not all(i in valid for i in range(lo, hi + 1))


def lane_mismatch(maneuver, distance: float, through_lanes: tuple, through_distance: float, lanes: ModelLanes, v_ego: float,
                  right_hand_traffic: bool = True):
  """The car is in a turn-only lane where the route goes straight through, or not in a lane that turns where the route
  turns, closer than the lane planner still changes lanes (lane_change_window's end). Only flagged, never acted on:
  LaneMismatch.none without the map's lane list, without the model reading the ego lane's lines or counting the lanes on
  the curb side."""
  if not lanes.measured:
    return LaneMismatch.none
  near = lane_change_window(v_ego)[1]
  turn = maneuver is not None and maneuver.kind == TURN
  if len(through_lanes) >= 2 and 0. < through_distance < near and not (turn and distance < through_distance):
    idx = ego_lane_index(len(through_lanes), lanes, right_hand_traffic)
    directions, active = through_lanes[idx] if idx is not None else ((), True)
    if not active and directions and "straight" not in directions:
      return LaneMismatch.turnOnlyGoingStraight
  if turn and len(maneuver.lanes) >= 2 and 0. < distance < near:
    idx = ego_lane_index(len(maneuver.lanes), lanes, right_hand_traffic)
    directions, active = maneuver.lanes[idx] if idx is not None else ((), True)
    if not active and not any(d in SIDE_DIRECTIONS[maneuver.side] for d in directions):
      return LaneMismatch.wrongLaneForTurn
  return LaneMismatch.none


def _yellow(p: LinePaint) -> bool:
  return p.kind != 'none' and p.color == 'yellow' and p.confidence >= YELLOW_CONF_MIN


LATCHED_YELLOW = LinePaint('unknown', 'yellow', 1., 0.)


def paint_lanes(lanes: ModelLanes, paint) -> ModelLanes:
  """The model's lane counts with every lane beyond a yellow line taken out. paint: one LinePaint per modelV2 lane line,
  [far left, left, right, far right], or empty when it wasn't read."""
  if len(paint) != 4:
    return lanes
  def side(n, ego, outer):
    return 0 if _yellow(ego) else min(n, 1) if _yellow(outer) else n
  return replace(lanes, left=side(lanes.left, paint[1], paint[0]), right=side(lanes.right, paint[2], paint[3]))


def broken_white_toward(paint, side) -> bool:
  """The ego lane's line toward side (LEFT or RIGHT) reads broken white: a lane going the same way is beyond it."""
  if len(paint) != 4 or side not in (LEFT, RIGHT):
    return False
  return paint[1 if side == LEFT else 2].is_('broken', 'white', BROKEN_WHITE_CONF_MIN)


def stopped_in_lane(objects, direction: int, width: float = LANE_WIDTH) -> bool:
  """Something stopped up to STOPPED_X_M ahead in the lane one over toward direction (LEFT or RIGHT): a parked car says
  that lane is a parking strip. objects: TurnpilotScene.TrackedObject-like, y positive left, vx over ground."""
  return any(0. < obj.x < STOPPED_X_M and abs(obj.vx) < STOPPED_V and 0.5 * width < obj.y * direction < 1.5 * width + LC_LANE_MARGIN_M
             for obj in objects)


def _toward(direction: int):
  return LaneChange.left if direction == LEFT else LaneChange.right if direction == RIGHT else LaneChange.none


def _direction(lc) -> int:
  return LEFT if lc == LaneChange.left else RIGHT if lc == LaneChange.right else 0


def desired_changes(maneuver, lanes: ModelLanes, right_hand_traffic: bool = True, paint=(), done: int | None = 0, start=None) -> int:
  """Lane changes the route wants before the maneuver, positive left, negative right, before change_blocker has a say.
  paint: as for paint_lanes; done, start: as for nav_lane_span. With a lane list, the fewest changes any lane the car may
  be in needs, and none when it may already be in one the maneuver leaves from."""
  if maneuver is None or not (maneuver.kind == TURN or (maneuver.kind == KEEP and maneuver.type == "off ramp")):
    return 0
  lanes = paint_lanes(lanes, paint)
  curb = RIGHT if right_hand_traffic else LEFT
  nav = _nav_lanes(maneuver)
  if nav:
    valid = _valid_lanes(maneuver, nav)
    lo, hi = nav_lane_span(len(nav), lanes, right_hand_traffic, done, start)
    return lo - max(valid) if lo > max(valid) else hi - min(valid) if hi < min(valid) else 0
  if maneuver.side == curb and maneuver.kind == KEEP:  # an exit: over to the curb lane by the model's count
    n = lanes.left if curb == LEFT else lanes.right
    if len(paint) == 4:  # not across a curb-side line read solid white: a shoulder or a bike lane is beyond it
      ego, outer = (paint[1], paint[0]) if curb == LEFT else (paint[2], paint[3])
      n = 0 if ego.is_('solid', 'white', SOLID_WHITE_CONF_MIN) else min(n, 1) if outer.is_('solid', 'white', SOLID_WHITE_CONF_MIN) else n
    return n if curb == LEFT else -n
  return 1 if maneuver.side == LEFT else -1  # a turn, or an exit away from the curb: one toward its side at a time


def change_blocker(direction: int, lanes: ModelLanes, paint=(), objects=(), right_hand_traffic: bool = True,
                   nav_room: int | None = None) -> str:
  """'' when one lane change toward direction (LEFT or RIGHT) is into a valid travel lane, else the abort reason why not
  (a TurnpilotPlan.LaneChangeAbort name). paint: as for paint_lanes, held yellow included; objects: as for stopped_in_lane;
  nav_room: as nav_room() returns it, None without the map's lane list."""
  if direction not in (LEFT, RIGHT):
    return 'noLane'
  if not right_hand_traffic and direction == RIGHT:
    return 'noLane'  # away from the curb in left-hand traffic: the center line is broken white there
  if not broken_white_toward(paint, direction):
    return 'lineNotBroken'
  if (paint_lanes(lanes, paint).left if direction == LEFT else paint_lanes(lanes, paint).right) == 0:
    return 'noLane'  # the model sees no lane that way, or only beyond a yellow line
  if nav_room is not None and nav_room <= 0:
    return 'noLane'  # the map's lane list has no lane that way: the model's are oncoming, or not where it can place the car
  if stopped_in_lane(objects, direction, lanes.width):
    return 'laneParked'
  return ''


def lane_changes_needed(maneuver, lanes: ModelLanes, right_hand_traffic: bool = True, paint=(), objects=(), done: int | None = 0,
                        start=None) -> int:
  """Lane changes the maneuver still needs, positive left, negative right, or 0 when the next one may not go
  (change_blocker). paint: as for paint_lanes; objects: as for stopped_in_lane, empty when unknown; done, start: as for
  nav_lane_span."""
  need = desired_changes(maneuver, lanes, right_hand_traffic, paint, done, start)
  direction = LEFT if need > 0 else RIGHT
  room = nav_room(maneuver, lanes, direction, right_hand_traffic, paint, done, start)
  if need and change_blocker(direction, lanes, paint, objects, right_hand_traffic, room):
    return 0
  return need


def lane_change_window(v_ego: float, changes: int = 1) -> tuple[float, float]:
  """(start, end) m before the maneuver, for this many lane changes still to make."""
  start = min(max(LC_START_S * v_ego, LC_START_MIN_M), LC_START_MAX_M) + max(changes - 1, 0) * LC_EXTRA_LANE_S * v_ego
  end = min(max(LC_END_S * v_ego, LC_END_MIN_M), LC_END_MAX_M)
  return start, end


def target_lane_blocked(objects, direction: int, v_ego: float, width: float = LANE_WIDTH) -> bool:
  """A vehicle in the lane one over toward direction (LEFT or RIGHT) within LC_BLOCK_X_M, or closing to it fast.
  objects: TurnpilotScene.TrackedObject-like, y positive left, vx over ground."""
  for obj in objects:
    lateral = obj.y * direction
    if not (0.5 * width < lateral < 1.5 * width + LC_LANE_MARGIN_M):
      continue
    rel = obj.vx - v_ego
    if abs(obj.x) < LC_BLOCK_X_M:
      return True
    if obj.x > LC_BLOCK_X_M and rel < 0 and (obj.x - LC_BLOCK_X_M) / -rel < LC_CLOSING_T_S:
      return True
    if obj.x < -LC_BLOCK_X_M and rel > 0 and (-LC_BLOCK_X_M - obj.x) / rel < LC_CLOSING_T_S:
      return True
  return False


class YellowHold:
  """Per modelV2 lane line, how long and how far since it was last read yellow; it reads yellow until both pass
  YELLOW_HOLD_S and YELLOW_HOLD_M. Lane changes move the holds with the lines."""
  def __init__(self):
    self.since = [(math.inf, math.inf)] * 4

  def update(self, paint, v_ego: float, dt: float, frozen: bool) -> tuple:
    """paint as for paint_lanes; frozen: a lane change is under way, so the lines may be renumbering. Returns the paint with
    every held line read yellow (four entries whenever anything is held)."""
    self.since = [(t + dt, d + v_ego * dt) for t, d in self.since]
    if len(paint) == 4 and not frozen:
      self.since = [(0., 0.) if _yellow(p) else hold for p, hold in zip(paint, self.since, strict=True)]
    held = [t < YELLOW_HOLD_S or d < YELLOW_HOLD_M for t, d in self.since]
    if not any(held):
      return tuple(paint)
    paint = tuple(paint) if len(paint) == 4 else (UNKNOWN,) * 4
    return tuple(p if _yellow(p) or not h else LATCHED_YELLOW for p, h in zip(paint, held, strict=True))

  def shift(self, direction: int) -> None:
    """The car moved one lane toward direction (LEFT or RIGHT): each line is now one index further the other way. With the
    direction unknown, any hold covers all four lines, since it could be any of them now."""
    none = (math.inf, math.inf)
    if direction == LEFT:
      self.since = [none] + self.since[:3]
    elif direction == RIGHT:
      self.since = self.since[1:] + [none]
    else:
      latest = min(self.since)
      self.since = [latest] * 4


class Phase(IntEnum):
  IDLE = 0
  WARN = 1      # the countdown: the driver has been told which way and when
  WAIT = 2      # the countdown is over, waiting for the driver's blinker (cars where openpilot can't signal)
  REQUEST = 3   # DesireHelper has the request; it starts the change when the blind spot is clear


class LaneDecision(NamedTuple):
  lane_change: int   # TurnpilotPlan.laneChange: the request for DesireHelper
  prompt: int        # TurnpilotPlan.signalPrompt: ask the driver to signal
  reason: str        # one short line for the status line
  offer: int         # TurnpilotPlan.laneChangeOffer: the side of the change warned of, until it starts or aborts
  countdown: float   # TurnpilotPlan.laneChangeCountdown: s of the warning left
  abort: str         # TurnpilotPlan.laneChangeAbort: why the last change was aborted, for ABORT_HOLD_S after


class LanePlanner:
  def __init__(self, paint_rule: bool = True):
    self.paint_rule = paint_rule      # TurnPilotLanePaintActive: plan lane changes at all
    self.phase = Phase.IDLE
    self.offer = LaneChange.none      # the change warned of
    self.countdown = 0.
    self.abort = 'none'               # the last abort's reason, held for ABORT_HOLD_S
    self.ours = False                 # a DesireHelper lane change we asked for is under way
    self.changes = 0                  # nav lane changes completed
    self.done: int | None = 0         # lane changes completed for this maneuver, ours or the driver's, positive left; None
                                      # once one went a way that wasn't seen
    self.lane_unknown = False         # in or past the window, the car's lane in the maneuver's lane list can't be confirmed
    self._done_id = None              # the maneuver done counts for
    self._start = None                # (lane, done) where our first change for it started: nav_lane_span
    self._from_lane: int | None = None  # the lane in the lane list the change wanted now is counted from
    self.yellow = YellowHold()
    self._lc_dir: int | None = None   # direction of the DesireHelper lane change under way, 0 when unknown
    self._maneuver_id = None          # the maneuver the offer is for
    self._cooldown = 0.
    self._abort_t = 0.
    self._candidate = LaneChange.none
    self._confirm = 0
    self._wait_t = 0.
    self._pre_t = 0.
    self._pending = ''                # an abort in REQUEST, said once DesireHelper is seen not to have started
    self._pending_t = 0.

  @property
  def active(self) -> bool:
    return self.phase != Phase.IDLE or self.ours

  def update(self, maneuver, distance: float, lanes: ModelLanes, lc_state, v_ego: float, left_blinker: bool,
             right_blinker: bool, objects, driver_signals: bool, right_hand_traffic: bool, dt: float, paint=(),
             left_blindspot: bool = False, right_blindspot: bool = False, maneuver_id=None, inputs_valid: bool = True) -> LaneDecision:
    """lc_state: modelV2.meta.laneChangeState; objects: tracked scene objects, empty when the scene is unavailable
    (DesireHelper's blindspot check still applies); paint: as for paint_lanes, empty when not read; blindspots: carState's;
    maneuver_id: RouteFollower.maneuver_id, a new one is a changed route; inputs_valid: carState and modelV2 pass their
    checks, else nothing is offered and an offer aborts."""
    if self._abort_t > 0.:
      self._abort_t -= dt
      if self._abort_t <= 0.:
        self.abort = 'none'
    if maneuver_id != self._done_id:  # a new maneuver: its changes are counted from here
      self._done_id, self.done, self._start = maneuver_id, 0, None
    changing = lc_state in (LaneChangeState.laneChangeStarting, LaneChangeState.laneChangeFinishing)
    paint = self.yellow.update(paint, v_ego, dt, changing)
    if changing:  # never cancelled once started: the car is already moving over
      if self._lc_dir is None:  # ours, or the driver's by the blinker
        req = self.offer if self.phase == Phase.REQUEST else \
          LaneChange.left if left_blinker and not right_blinker else LaneChange.right if right_blinker and not left_blinker else LaneChange.none
        self._lc_dir = _direction(req)
        if self.phase == Phase.REQUEST and self._start is None and self._from_lane is not None and self.done is not None:
          self._start = (self._from_lane, self.done)  # the first of ours for this maneuver: dead reckoned from here
      self.ours = self.ours or self.phase == Phase.REQUEST
      self.lane_unknown = False
      self._reset()
      return self._decision()
    if self._lc_dir is not None:  # a change just ended, ours or the driver's: the lines have renumbered
      self.yellow.shift(self._lc_dir)
      if self._lc_dir and self.done is not None:
        self.done += self._lc_dir
      else:  # a way not seen: nothing to dead reckon from for the rest of this maneuver
        self.done, self._start = None, None
      self._lc_dir = None
      self._cooldown = max(self._cooldown, LC_COOLDOWN_S)
    if self.ours:  # DesireHelper is done with it, whether or not a held blinker keeps it in preLaneChange
      self.ours = False
      self.changes += 1

    fast = v_ego > LANE_CHANGE_SPEED_MIN + LC_SPEED_MARGIN
    want = desired_changes(maneuver, lanes, right_hand_traffic, paint, self.done, self._start) if fast else 0
    start, end = lane_change_window(v_ego, abs(want))
    want_dir = LEFT if want > 0 else RIGHT if want < 0 else 0
    nav = _nav_lanes(maneuver)
    if nav and want_dir:
      lo, hi = nav_lane_span(len(nav), paint_lanes(lanes, paint), right_hand_traffic, self.done, self._start)
      self._from_lane = lo if want_dir == LEFT else hi
    else:
      self._from_lane = None
    self.lane_unknown = self.paint_rule and maneuver is not None and 0. < distance < start and \
      lane_unconfirmed(maneuver, lanes, right_hand_traffic, paint, self.done, self._start)
    idle_reason = LANE_UNKNOWN_REASON if self.lane_unknown else ""

    if self._pending:  # the request is withdrawn; DesireHelper didn't start the change (else `changing` above): abort
      self._pending_t -= dt
      if self._pending_t > 1e-6:
        return self._decision(ABORT_TEXT[self._pending])
      reason = self._pending
      self._reset()
      return self._abort(reason)
    if self.phase != Phase.IDLE:
      return self._follow_up(maneuver, maneuver_id, distance, end, want_dir, fast, lanes, paint, objects, lc_state, v_ego,
                             left_blinker, right_blinker, left_blindspot, right_blindspot, driver_signals, right_hand_traffic, dt,
                             inputs_valid)

    if self._cooldown > 0.:
      self._cooldown -= dt
      self._confirm = 0
      return self._decision(idle_reason)
    if not self.paint_rule or not inputs_valid or not (end < distance < start):
      want_dir = 0
    if want_dir and change_blocker(want_dir, lanes, paint, objects, right_hand_traffic,
                                   nav_room(maneuver, lanes, want_dir, right_hand_traffic, paint, self.done, self._start)):
      want_dir = 0
    direction = _toward(want_dir)
    self._confirm = self._confirm + 1 if direction == self._candidate else 1
    self._candidate = direction
    if direction == LaneChange.none or self._confirm < LC_CONFIRM_FRAMES:
      return self._decision(idle_reason)
    if left_blindspot if want_dir == LEFT else right_blindspot:
      return self._decision("car in blind spot")
    if target_lane_blocked(objects, want_dir, v_ego, lanes.width):
      return self._decision("car in target lane")
    self.phase, self.offer, self.countdown = Phase.WARN, direction, LC_WARN_S
    self._maneuver_id, self._wait_t, self._pre_t = maneuver_id, 0., 0.
    self.abort, self._abort_t = 'none', 0.
    return self._decision(prompt=driver_signals)

  def _follow_up(self, maneuver, maneuver_id, distance, end, want_dir, fast, lanes, paint, objects, lc_state, v_ego,
                 left_blinker, right_blinker, left_blindspot, right_blindspot, driver_signals, right_hand_traffic, dt,
                 inputs_valid) -> LaneDecision:
    """An offered change, every frame until DesireHelper starts it: still wanted, still valid, still clear, still signaled."""
    d = _direction(self.offer)
    if not self.paint_rule:
      return self._abort('turnedOff')
    if not inputs_valid:
      return self._abort('badInputs')
    if maneuver is None or maneuver_id != self._maneuver_id:
      return self._abort('routeChanged')
    if not fast:
      return self._abort('tooSlow')
    if distance <= end:
      return self._abort('tooLate')
    reason = change_blocker(d, lanes, paint, objects, right_hand_traffic,
                            nav_room(maneuver, lanes, d, right_hand_traffic, paint, self.done, self._start))
    if reason:
      return self._abort(reason)
    if want_dir != d:
      return self._abort('notNeeded')  # the model's lane count says the car is in the lane it wants
    if left_blindspot if d == LEFT else right_blindspot:
      return self._abort('blindspot')
    if target_lane_blocked(objects, d, v_ego, lanes.width):
      return self._abort('carInLane')

    signaled = left_blinker if d == LEFT else right_blinker
    if self.phase == Phase.WARN:
      self.countdown = max(self.countdown - dt, 0.)
      if driver_signals and signaled:  # the driver's go-ahead, any time in the countdown
        self.phase = Phase.REQUEST
      elif self.countdown <= 0.:
        self.phase = Phase.WAIT if driver_signals else Phase.REQUEST
    elif self.phase == Phase.WAIT:
      self._wait_t += dt
      if signaled:
        self.phase = Phase.REQUEST
      elif self._wait_t > LC_SIGNAL_TIMEOUT_S:
        return self._abort('noSignal')
    elif self.phase == Phase.REQUEST:
      if driver_signals and not signaled:
        return self._abort('noSignal')  # the blinker went off before the change started: never change unsignaled
      if lc_state == LaneChangeState.preLaneChange:
        self._pre_t += dt
        if self._pre_t > LC_REQUEST_TIMEOUT_S:
          return self._abort('timedOut')
    if self.phase == Phase.REQUEST:
      self.countdown = 0.
      return self._decision(request=True, prompt=driver_signals)
    return self._decision(prompt=driver_signals)

  def cancel(self, reason: str) -> bool:
    """TurnPilot stopped acting (reason 'handedBack' or 'fault'): an offered change that DesireHelper hasn't started is
    aborted with reason, held ABORT_HOLD_S as any abort. A change under way goes on: nothing cancels it once started, and
    one DesireHelper may have started on the request already isn't announced as aborted (_abort).
    True when there was an offer."""
    if self.phase == Phase.IDLE:
      return False
    self._abort(reason)
    return True

  def _abort(self, reason: str) -> LaneDecision:
    """Abort the offered change. Once the request is out, DesireHelper may have started it in modeld's last step, which
    turnpilotd sees a modelV2 or two later: the request is withdrawn at once, and the abort said only after
    LC_ABORT_SETTLE_S with no change starting. One that does start is ours, and goes on: no abort contradicts it."""
    if self.phase == Phase.REQUEST:  # (a second abort while one waits, a handback's say, keeps the first reason)
      if not self._pending:
        self._pending, self._pending_t = reason, LC_ABORT_SETTLE_S
      return self._decision(ABORT_TEXT[self._pending])
    self._reset()
    self.abort, self._abort_t = reason, ABORT_HOLD_S
    self._cooldown = max(self._cooldown, LC_ABORT_COOLDOWN_S)
    return self._decision(ABORT_TEXT[reason])

  def _reset(self) -> None:
    self.phase, self.offer, self.countdown = Phase.IDLE, LaneChange.none, 0.
    self._confirm, self._candidate, self._wait_t, self._pre_t = 0, LaneChange.none, 0., 0.
    self._pending, self._pending_t = '', 0.

  def _decision(self, reason: str = "", request: bool = False, prompt: bool = False) -> LaneDecision:
    return LaneDecision(self.offer if request else LaneChange.none, self.offer if prompt else LaneChange.none,
                        reason, self.offer, self.countdown, self.abort)
