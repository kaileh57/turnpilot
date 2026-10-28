"""Lane planner: get into the lane a turn leaves from, one nav lane change at a time.

Which lane the car is in comes from what the model sees: lanes either side counted from the room between the
ego lane's line and the road edge, at about 3.6 m a lane, but only where the next line out is painted (room
alone counts shoulders and parking strips as lanes). modelV2 can't tell a yellow line from a white one, and on
an undivided road its far road edge lies beyond the oncoming lanes; the lane paint read off the road camera
(scene/lane_paint.py, one entry per modelV2 lane line) can. US markings say what is beyond a line: broken white
is a lane going the same way, solid white an edge, shoulder, bike or parking lane, yellow oncoming traffic or a
divided road's left edge.

- Always: no change is requested across a line read yellow, and lanes beyond a yellow line count 0, also in the
  count used with the map's lane list (NavInstruction.lanes). A yellow read keeps vetoing for YELLOW_HOLD_S and
  YELLOW_HOLD_M after the last one, so a worn stretch the classifier can't read doesn't open a gap; the hold moves
  with the lines when the car changes lanes.
- With the list, the car's lane is counted from the curb side, which is always its own direction's.
- Without it (the usual case: OSM tags lanes on few roads), an exit moves toward the curb by the model's lane count,
  but not across a curb-side line read solid white (a shoulder or bike lane edge), as before the paint was read.
- The broken white rule, only with TurnPilotLanePaintActive set (turnpilotd publishes what it would have asked for
  either way, in laneChangePaintRule): without the list, the car also moves toward the maneuver's side, turns as well
  as exits, while the ego lane's line on that side reads broken white with good confidence: one change at a time,
  looked at again after each change's cooldown, so it keeps moving over while that line stays broken white inside the
  lane change window. Solid, yellow, no line or unknown on that side means no change. It holds only under US markings
  (MUTCD): in left-hand traffic the center line is broken white, so it never acts away from the curb there, and
  nothing tells right-hand Europe (white center lines too) from the US, so the param is for US and Canadian drives.
  Paint alone isn't enough: the model must also see a lane that way, its next line out at least LANE_MIN lane
  widths beyond the ego line (an 8 ft parking strip beside an 11 ft lane is 0.73, a 6 ft bike lane beside a 10 ft one
  0.61), and nothing stopped in that lane ahead (parked cars). Before the paint, a change toward the curb before a
  junction turn put the car in a parking strip (BeamNG, 2026-10-27), which is why a turn needs all three.

DesireHelper executes the change: a request stands in for the blinker and the nudge, and is dropped the moment
the change starts, or DesireHelper would chain a second one. On cars where openpilot can't signal, the driver
is prompted to signal and the request waits for that blinker.
"""
import math
from dataclasses import dataclass, replace

from openpilot.cereal import custom, log
from openpilot.selfdrive.controls.lib.desire_helper import LANE_CHANGE_SPEED_MIN
from openpilot.selfdrive.turnpilot.route_follower import KEEP, LEFT, RIGHT, TURN
from openpilot.selfdrive.turnpilot.scene.lane_paint import UNKNOWN, LinePaint

LaneChange = custom.TurnpilotPlan.LaneChange
LaneChangeState = log.LaneChangeState

LANE_WIDTH = 3.6
SAMPLE_IDX = 8              # modelV2 point ~12 m ahead (X_IDXS[8])
LINE_PROB_MIN = 0.5
EDGE_STD_MAX = 1.2          # a road edge is trusted when its std is under this
SHOULDER_LANES = 0.75       # room narrower than this many lane widths beyond the ego line is a shoulder
LANE_MIN = 0.85             # a lane beside the ego lane has its next line out this many lane widths away or more (to 1.5)
KEEP_MIN_ROOM_M = 1.0       # a keep desire toward a trusted edge closer than this beyond the ego line drives off the road
BROKEN_WHITE_CONF_MIN = 0.6  # the ego line toward a turn read broken white at this confidence allows a change without map lanes
SOLID_WHITE_CONF_MIN = 0.6   # a curb-side line read solid white at this confidence stops an exit's changes across it
YELLOW_CONF_MIN = 0.3        # a line read yellow at this confidence vetoes changes across it (the cautious answer, so a lower bar)
YELLOW_HOLD_S, YELLOW_HOLD_M = 5., 100.  # and keeps vetoing until this long and this far after its last yellow read
STOPPED_V = 2.               # m/s: an object this slow in the target lane, up to STOPPED_X_M ahead, keeps the broken white rule out
STOPPED_X_M = 80.

LC_START_S, LC_START_MIN_M, LC_START_MAX_M = 20., 150., 400.  # lane changes start this far before the maneuver...
LC_END_S, LC_END_MIN_M, LC_END_MAX_M = 6., 60., 120.          # ...and none start closer than this
LC_EXTRA_LANE_S = 8.        # each change past the first starts this much earlier (an exit across a highway)
LC_SPEED_MARGIN = 0.5       # m/s above DesireHelper's minimum, so a request never waits below it
LC_CONFIRM_FRAMES = 10      # the same need this many frames in a row before a request
LC_COOLDOWN_S = 3.          # after a change, for the model's lane lines to settle
LC_REQUEST_TIMEOUT_S = 8.   # DesireHelper waiting this long (blindspot, no driver blinker) drops the request
LC_BLOCK_X_M = 15.          # a vehicle in the target lane within this, ahead or behind, blocks a request...
LC_CLOSING_T_S = 3.         # ...as does one that closes to that distance within this
LC_LANE_MARGIN_M = 0.5

SIDE_DIRECTIONS = {LEFT: ("left", "slightLeft"), RIGHT: ("right", "slightRight")}
OPENPILOT_SIGNALS_BRANDS = ("hyundai",)  # ports that forward CarControl blinkers (CAN-FD); everyone else is prompted


@dataclass(frozen=True)
class ModelLanes:
  left: int = 0               # lanes the model sees beside the ego lane
  right: int = 0
  keep_left_ok: bool = False  # a keep desire that way has a lane line and road beside it
  keep_right_ok: bool = False
  width: float = LANE_WIDTH


def _side(ego_y, ego_prob, outer_y, outer_prob, edge_y, edge_std, sign, width):
  """(lanes counted on one side, keep ok that way); sign is +1 for the right, -1 for the left (modelV2 y is right)."""
  room = sign * (edge_y - ego_y)
  painted = outer_prob > LINE_PROB_MIN and LANE_MIN * width <= sign * (outer_y - ego_y) <= 1.5 * width
  edge_ok = edge_std < EDGE_STD_MAX
  if not painted or (edge_ok and room < SHOULDER_LANES * width):
    lanes = 0
  else:
    lanes = max(int(math.floor(room / width + 0.25)), 1) if edge_ok else 1
  keep_ok = ego_prob > LINE_PROB_MIN and not (lanes == 0 and edge_ok and room < KEEP_MIN_ROOM_M)
  return lanes, keep_ok


def model_lanes(line_ys, line_probs, edge_ys, edge_stds) -> ModelLanes:
  """From modelV2 laneLines [far left, left, right, far right] and roadEdges [left, right] at one distance ahead."""
  measured = line_ys[2] - line_ys[1]
  width = measured if min(line_probs[1], line_probs[2]) > LINE_PROB_MIN and 2.5 <= measured <= 4.8 else LANE_WIDTH
  left, keep_left = _side(line_ys[1], line_probs[1], line_ys[0], line_probs[0], edge_ys[0], edge_stds[0], -1., width)
  right, keep_right = _side(line_ys[2], line_probs[2], line_ys[3], line_probs[3], edge_ys[1], edge_stds[1], 1., width)
  return ModelLanes(left, right, keep_left, keep_right, width)


def model_lanes_from_msg(model) -> ModelLanes:
  lines, edges = model.laneLines, model.roadEdges
  if len(lines) < 4 or len(edges) < 2 or len(model.laneLineProbs) < 4 or len(model.roadEdgeStds) < 2 or \
     any(len(e.y) <= SAMPLE_IDX for e in list(lines) + list(edges)):
    return ModelLanes()
  return model_lanes([ln.y[SAMPLE_IDX] for ln in lines], list(model.laneLineProbs),
                     [e.y[SAMPLE_IDX] for e in edges], list(model.roadEdgeStds))


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


def lane_changes_needed(maneuver, lanes: ModelLanes, right_hand_traffic: bool = True, paint=(), paint_rule: bool = True,
                        objects=()) -> int:
  """Lane changes the maneuver still needs: positive left, negative right. paint: as for paint_lanes; paint_rule: the
  broken white rule (TurnPilotLanePaintActive); objects: as for stopped_in_lane, empty when unknown."""
  if maneuver is None or not (maneuver.kind == TURN or (maneuver.kind == KEEP and maneuver.type == "off ramp")):
    return 0
  lanes = paint_lanes(lanes, paint)
  curb = RIGHT if right_hand_traffic else LEFT
  nav = maneuver.lanes
  if len(nav) >= 2:
    valid = ([i for i, (_, active) in enumerate(nav) if active] or
             [i for i, (dirs, _) in enumerate(nav) if any(d in SIDE_DIRECTIONS[maneuver.side] for d in dirs)] or
             [0 if maneuver.side == LEFT else len(nav) - 1])
    idx = len(nav) - 1 - lanes.right if curb == RIGHT else lanes.left  # from the left, counted from the curb
    idx = min(max(idx, 0), len(nav) - 1)
    need = idx - max(valid) if idx > max(valid) else idx - min(valid) if idx < min(valid) else 0
  else:
    exit_need = 0
    if maneuver.side == curb and maneuver.kind == KEEP:
      n = lanes.left if curb == LEFT else lanes.right
      if len(paint) == 4:  # not across a curb-side line read solid white: a shoulder or a bike lane is beyond it
        ego, outer = (paint[1], paint[0]) if curb == LEFT else (paint[2], paint[3])
        n = 0 if ego.is_('solid', 'white', SOLID_WHITE_CONF_MIN) else min(n, 1) if outer.is_('solid', 'white', SOLID_WHITE_CONF_MIN) else n
      exit_need = n if curb == LEFT else -n
    paint_need = 0
    if paint_rule and (right_hand_traffic or maneuver.side == curb) and broken_white_toward(paint, maneuver.side) and \
       not stopped_in_lane(objects, maneuver.side, lanes.width):
      paint_need = 1 if maneuver.side == LEFT else -1
    need = exit_need if abs(exit_need) >= abs(paint_need) else paint_need
  if (need > 0 and lanes.left == 0) or (need < 0 and lanes.right == 0):
    return 0  # never toward a side where the model sees no lane
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
    """The car moved one lane toward direction (LEFT or RIGHT): each line is now one index further the other way."""
    none = (math.inf, math.inf)
    if direction == LEFT:
      self.since = [none] + self.since[:3]
    elif direction == RIGHT:
      self.since = self.since[1:] + [none]


class LanePlanner:
  def __init__(self, paint_rule: bool = True):
    self.paint_rule = paint_rule      # the broken white rule (TurnPilotLanePaintActive)
    self.requested = LaneChange.none  # what was last asked of DesireHelper
    self.prompt = LaneChange.none     # what the driver was last asked to signal
    self.ours = False                 # a DesireHelper lane change we asked for is under way
    self.changes = 0                  # nav lane changes completed
    self.yellow = YellowHold()
    self._lc_dir: int | None = None   # direction of the DesireHelper lane change under way, 0 when unknown
    self._cooldown = 0.
    self._candidate = LaneChange.none
    self._confirm = 0
    self._pre_t = 0.

  @property
  def active(self) -> bool:
    return self.requested != LaneChange.none or self.prompt != LaneChange.none or self.ours

  def update(self, maneuver, distance: float, lanes: ModelLanes, lc_state, v_ego: float, left_blinker: bool,
             right_blinker: bool, objects, driver_signals: bool, right_hand_traffic: bool, dt: float, paint=()):
    """(laneChange, signalPrompt, reason). lc_state: modelV2.meta.laneChangeState; objects: tracked scene objects,
    empty when the scene is unavailable (DesireHelper's blindspot check still applies); paint: as for paint_lanes."""
    changing = lc_state in (LaneChangeState.laneChangeStarting, LaneChangeState.laneChangeFinishing)
    paint = self.yellow.update(paint, v_ego, dt, changing)
    if changing:
      if self._lc_dir is None:  # ours, or the driver's by the blinker
        req = self.requested if self.requested != LaneChange.none else \
          LaneChange.left if left_blinker and not right_blinker else LaneChange.right if right_blinker and not left_blinker else LaneChange.none
        self._lc_dir = LEFT if req == LaneChange.left else RIGHT if req == LaneChange.right else 0
      self.ours = self.ours or self.requested != LaneChange.none
      return self._none()
    if self._lc_dir is not None:
      self.yellow.shift(self._lc_dir)
      self._lc_dir = None
    if self.ours:  # DesireHelper is done with it, whether or not a held blinker keeps it in preLaneChange
      self.ours = False
      self.changes += 1
      self._cooldown = LC_COOLDOWN_S
    if self._cooldown > 0.:
      self._cooldown -= dt
      self._confirm = 0
      return self._none()

    need = lane_changes_needed(maneuver, lanes, right_hand_traffic, paint, self.paint_rule, objects) \
      if v_ego > LANE_CHANGE_SPEED_MIN + LC_SPEED_MARGIN else 0
    start, end = lane_change_window(v_ego, abs(need))
    if not end < distance < start:
      need = 0
    direction = LaneChange.left if need > 0 else LaneChange.right if need < 0 else LaneChange.none
    self._confirm = self._confirm + 1 if direction == self._candidate else 1
    self._candidate = direction
    if direction == LaneChange.none or self._confirm < LC_CONFIRM_FRAMES:
      return self._none()
    if target_lane_blocked(objects, LEFT if direction == LaneChange.left else RIGHT, v_ego, lanes.width):
      return self._none("car in target lane")

    if lc_state == LaneChangeState.preLaneChange and self.requested != LaneChange.none:
      self._pre_t += dt
      if self._pre_t > LC_REQUEST_TIMEOUT_S:
        self._cooldown = LC_COOLDOWN_S
        return self._none("lane change timed out")
    else:
      self._pre_t = 0.
    signaled = left_blinker if direction == LaneChange.left else right_blinker
    self.requested = direction if signaled or not driver_signals else LaneChange.none
    self.prompt = direction if driver_signals else LaneChange.none
    return self.requested, self.prompt, ""

  def _none(self, reason=""):
    self.requested = self.prompt = LaneChange.none
    self._pre_t = 0.
    return LaneChange.none, LaneChange.none, reason
