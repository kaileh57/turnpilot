"""Synthetic cereal messages for TurnPilot's tests."""
import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.modeld.constants import ModelConstants

LANE_WIDTH = 3.6


def nav_instruction(distance, mtype="turn", modifier="left", remaining=None, lanes=(), valid=True):
  """A navInstruction reader. lanes: ((directions, active), ...) left to right."""
  msg = messaging.new_message('navInstruction', valid=valid)
  ni = msg.navInstruction
  ni.maneuverDistance = float(distance)
  ni.maneuverType = mtype
  ni.maneuverModifier = modifier
  ni.distanceRemaining = float(distance + 500. if remaining is None else remaining)
  if lanes:
    ni.init('lanes', len(lanes))
    for lane, (directions, active) in zip(ni.lanes, lanes, strict=True):
      lane.directions = list(directions)
      lane.active = active
  return msg.as_reader().navInstruction


class FakeNavd:
  """navd's 1 Hz instruction stream for a car driving along a route of maneuvers [(s, type, modifier), ...]:
  the distance to the end of the current step, which runs to -10 m before navd moves to the next step."""
  def __init__(self, maneuvers, lanes=None, period=20):
    self.maneuvers = sorted(maneuvers)
    self.lanes = lanes or {}
    self.period = period
    self.step = 0
    self.frame = 0

  def update(self, s):
    """(instruction or None when not due this frame) for a car at s m along the route."""
    while self.step + 1 < len(self.maneuvers) and self.maneuvers[self.step][0] - s < -10.:
      self.step += 1
    due = self.frame % self.period == 0
    self.frame += 1
    if not due:
      return None
    pos, mtype, modifier = self.maneuvers[self.step]
    return nav_instruction(pos - s, mtype, modifier, remaining=self.maneuvers[-1][0] - s, lanes=self.lanes.get(self.step, ()))


def road(lanes_left, lanes_right, line_prob=0.9, edge_std=0.3, lane_width=LANE_WIDTH, shoulder=0.5):
  """(line_ys, line_probs, edge_ys, edge_stds) as modelV2 sees a straight road: the car centered in its lane with
  lanes_left and lanes_right more lanes beside it; y positive right, as modelV2 publishes it."""
  half = lane_width / 2
  line_ys = [-half - lane_width, -half, half, half + lane_width]
  line_probs = [line_prob if lanes_left else 0.05, line_prob, line_prob, line_prob if lanes_right else 0.05]
  edge_ys = [-half - lanes_left * lane_width - shoulder, half + lanes_right * lane_width + shoulder]
  return line_ys, line_probs, edge_ys, [edge_std, edge_std]


def model_v2(lanes_left=0, lanes_right=0, lane_change_state='off', **kwargs):
  """A modelV2 reader with flat lane lines and road edges from road()."""
  line_ys, line_probs, edge_ys, edge_stds = road(lanes_left, lanes_right, **kwargs)
  msg = messaging.new_message('modelV2')
  m = msg.modelV2
  x = [float(v) for v in ModelConstants.X_IDXS]
  m.init('laneLines', 4)
  for line, y in zip(m.laneLines, line_ys, strict=True):
    line.x = x
    line.y = [float(y)] * len(x)
  m.laneLineProbs = [float(p) for p in line_probs]
  m.init('roadEdges', 2)
  for edge, y in zip(m.roadEdges, edge_ys, strict=True):
    edge.x = x
    edge.y = [float(y)] * len(x)
  m.roadEdgeStds = [float(s) for s in edge_stds]
  m.meta.laneChangeState = lane_change_state
  return msg.as_reader().modelV2


def car_state(v_ego=10., steering_angle_deg=0., left_blinker=False, right_blinker=False):
  msg = messaging.new_message('carState')
  cs = msg.carState
  cs.vEgo = float(v_ego)
  cs.steeringAngleDeg = float(steering_angle_deg)
  cs.leftBlinker = left_blinker
  cs.rightBlinker = right_blinker
  return msg.as_reader().carState


def scene(objects=(), available=True):
  """A turnpilotScene reader. objects: dicts of TrackedObject fields (cls, x, y, vx, vy, inPath)."""
  msg = messaging.new_message('turnpilotScene')
  sc = msg.turnpilotScene
  sc.available = available
  sc.init('objects', len(objects))
  for i, (obj, fields) in enumerate(zip(sc.objects, objects, strict=True)):
    obj.trackId = i
    obj.conf = 0.9
    obj.age = 10
    for k, v in fields.items():
      setattr(obj, k, v)
  return msg.as_reader().turnpilotScene
