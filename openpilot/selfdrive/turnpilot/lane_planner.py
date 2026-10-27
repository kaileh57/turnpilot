"""Lane planner: get into the lane a turn leaves from, one nav lane change at a time.

Which lane the car is in comes from what the model sees: lanes either side counted from the room between the
ego lane's line and the road edge, at about 3.6 m a lane, but only where the next line out is painted (room
alone counts shoulders and parking strips as lanes). modelV2 can't tell a yellow line from a white one, and on
an undivided road its far road edge lies beyond the oncoming lanes, so without the map's lane list
(NavInstruction.lanes) the car only ever changes toward the curb, and only for an exit: before a junction turn
the painted strip at the curb is as often a parking lane as a lane (BeamNG, 2026-10-27: the change put the car
in it and the right turn from there ended against the corner). With the list, the car's lane is counted from
the curb side, which is always its own direction's.

DesireHelper executes the change: a request stands in for the blinker and the nudge, and is dropped the moment
the change starts, or DesireHelper would chain a second one. On cars where openpilot can't signal, the driver
is prompted to signal and the request waits for that blinker.
"""
import math
from dataclasses import dataclass

from openpilot.cereal import custom, log
from openpilot.selfdrive.controls.lib.desire_helper import LANE_CHANGE_SPEED_MIN
from openpilot.selfdrive.turnpilot.route_follower import KEEP, LEFT, RIGHT, TURN

LaneChange = custom.TurnpilotPlan.LaneChange
LaneChangeState = log.LaneChangeState

LANE_WIDTH = 3.6
SAMPLE_IDX = 8              # modelV2 point ~12 m ahead (X_IDXS[8])
LINE_PROB_MIN = 0.5
EDGE_STD_MAX = 1.2          # a road edge is trusted when its std is under this
SHOULDER_LANES = 0.75       # room narrower than this many lane widths beyond the ego line is a shoulder
KEEP_MIN_ROOM_M = 1.0       # a keep desire toward a trusted edge closer than this beyond the ego line drives off the road

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
  painted = outer_prob > LINE_PROB_MIN and 0.6 * width <= sign * (outer_y - ego_y) <= 1.5 * width
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


def lane_changes_needed(maneuver, lanes: ModelLanes, right_hand_traffic: bool = True) -> int:
  """Lane changes the maneuver still needs: positive left, negative right."""
  if maneuver is None or not (maneuver.kind == TURN or (maneuver.kind == KEEP and maneuver.type == "off ramp")):
    return 0
  curb = RIGHT if right_hand_traffic else LEFT
  nav = maneuver.lanes
  if len(nav) >= 2:
    valid = ([i for i, (_, active) in enumerate(nav) if active] or
             [i for i, (dirs, _) in enumerate(nav) if any(d in SIDE_DIRECTIONS[maneuver.side] for d in dirs)] or
             [0 if maneuver.side == LEFT else len(nav) - 1])
    idx = len(nav) - 1 - lanes.right if curb == RIGHT else lanes.left  # from the left, counted from the curb
    idx = min(max(idx, 0), len(nav) - 1)
    need = idx - max(valid) if idx > max(valid) else idx - min(valid) if idx < min(valid) else 0
  elif maneuver.side == curb and maneuver.kind == KEEP:
    need = lanes.left if curb == LEFT else -lanes.right
  else:
    return 0
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


class LanePlanner:
  def __init__(self):
    self.requested = LaneChange.none  # what was last asked of DesireHelper
    self.prompt = LaneChange.none     # what the driver was last asked to signal
    self.ours = False                 # a DesireHelper lane change we asked for is under way
    self.changes = 0                  # nav lane changes completed
    self._cooldown = 0.
    self._candidate = LaneChange.none
    self._confirm = 0
    self._pre_t = 0.

  @property
  def active(self) -> bool:
    return self.requested != LaneChange.none or self.prompt != LaneChange.none or self.ours

  def update(self, maneuver, distance: float, lanes: ModelLanes, lc_state, v_ego: float, left_blinker: bool,
             right_blinker: bool, objects, driver_signals: bool, right_hand_traffic: bool, dt: float):
    """(laneChange, signalPrompt, reason). lc_state: modelV2.meta.laneChangeState; objects: tracked scene objects,
    empty when the scene is unavailable (DesireHelper's blindspot check still applies)."""
    if lc_state in (LaneChangeState.laneChangeStarting, LaneChangeState.laneChangeFinishing):
      self.ours = self.ours or self.requested != LaneChange.none
      return self._none()
    if self.ours:  # DesireHelper is done with it, whether or not a held blinker keeps it in preLaneChange
      self.ours = False
      self.changes += 1
      self._cooldown = LC_COOLDOWN_S
    if self._cooldown > 0.:
      self._cooldown -= dt
      self._confirm = 0
      return self._none()

    need = lane_changes_needed(maneuver, lanes, right_hand_traffic) if v_ego > LANE_CHANGE_SPEED_MIN + LC_SPEED_MARGIN else 0
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
