"""BeamNG's AI road graph (map.getMap(), exported by piston to west_coast_usa_graph.json): junctions, their arms, and
legal 90-degree turns, on public paved roads. Positions are BeamNG world metres (x east, y north), headings radians
counterclockwise from east."""
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

GRAPH_JSON = os.path.expanduser("~/pc_relay/incoming/osm_export/west_coast_usa_graph.json")
ARM_HEADING_DIST = 15.0  # m along an arm to measure its direction


def wrap(a: float) -> float:
  return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Turn:
  junction: str
  center: np.ndarray
  approach: np.ndarray  # points from the junction back along the incoming arm
  exit: np.ndarray  # points from the junction along the outgoing arm
  angle: float  # rad, left positive
  lanes_in: str  # BeamNG lane string of the approach link, oriented to our travel ('+' = our direction, left to right)
  radius_in: float  # half the road width at the approach

  @property
  def direction(self) -> str:
    return "left" if self.angle > 0 else "right"

  def spawn(self, dist: float):
    """(x, y, z), heading on the approach, `dist` m before the junction, in the lane for the turn (right-hand traffic)."""
    pts = self.approach
    seg = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)])
    dist = min(dist, seg[-1])
    i = max(1, int(np.searchsorted(seg, dist)))
    f = (dist - seg[i - 1]) / max(seg[i] - seg[i - 1], 1e-6)
    p = pts[i - 1] + f * (pts[i] - pts[i - 1])
    d = pts[i - 1] - pts[i]  # travel direction: toward the junction
    heading = math.atan2(d[1], d[0])
    ours = [k for k, c in enumerate(self.lanes_in) if c == '+'] or [len(self.lanes_in) - 1]
    lane = ours[0] if self.direction == "left" else ours[-1]
    w = 2 * self.radius_in / max(len(self.lanes_in), 1)
    offset = -self.radius_in + w * (lane + 0.5)  # to the right of the centreline
    x, y = p[0] + offset * math.sin(heading), p[1] - offset * math.cos(heading)
    return (float(x), float(y), float(p[2]) + 0.3), heading


class RoadGraph:
  def __init__(self, path: str = GRAPH_JSON):
    with open(path) as f:
      g = json.load(f)
    self.pos = {k: np.array(v['p'], dtype=float) for k, v in g['nodes'].items()}
    self.radius = {k: float(v['r']) for k, v in g['nodes'].items()}
    self.adj: dict[str, set] = defaultdict(set)
    self.out: dict[str, set] = defaultdict(set)
    self.links: dict[tuple, dict] = {}
    for link in g['links']:
      if link.get('type') == 'private' or link.get('drivability', 1) < 1:
        continue
      a, b = link['inNode'], link['outNode']
      self.adj[a].add(b)
      self.adj[b].add(a)
      self.out[a].add(b)
      if not link.get('oneWay'):
        self.out[b].add(a)
      self.links[(a, b)] = self.links[(b, a)] = link

  def junctions(self):
    return [n for n, nb in self.adj.items() if len(nb) >= 3]

  def nearest_junction(self, x: float, y: float) -> str:
    return min(self.junctions(), key=lambda n: math.hypot(self.pos[n][0] - x, self.pos[n][1] - y))

  def walk(self, j: str, first: str, length: float) -> np.ndarray:
    """Points from junction j along the arm through `first`, until `length` m or the next junction."""
    pts, prev, cur, d = [self.pos[j]], j, first, 0.0
    while True:
      d += float(np.linalg.norm(self.pos[cur][:2] - self.pos[prev][:2]))
      pts.append(self.pos[cur])
      nxt = self.adj[cur] - {prev}
      if d >= length or len(nxt) != 1:
        return np.array(pts)
      prev, cur = cur, next(iter(nxt))

  @staticmethod
  def arm_heading(pts: np.ndarray) -> float:
    d = np.linalg.norm(pts[:, :2] - pts[0, :2], axis=1)
    k = int(np.argmax(d >= ARM_HEADING_DIST)) if (d >= ARM_HEADING_DIST).any() else len(pts) - 1
    v = pts[k] - pts[0]
    return math.atan2(v[1], v[0])

  def lanes_toward(self, j: str, first: str) -> str:
    """Lane string of the link between `first` and junction j, oriented for travel first -> j."""
    link = self.links[(first, j)]
    lanes = link.get('lanes') or ('+' if link.get('oneWay') else '-+')
    if link['inNode'] == first:
      return lanes
    return ''.join('+' if c == '-' else '-' for c in reversed(lanes))

  def turns(self, j: str, approach_len: float = 150.0, exit_len: float = 80.0, tolerance_deg: float = 15.0) -> list[Turn]:
    """Legal turns of about 90 degrees at junction j."""
    out = []
    for a in self.adj[j]:
      if j not in self.out[a]:
        continue
      approach = self.walk(j, a, approach_len)
      h_in = wrap(self.arm_heading(approach) + math.pi)  # travelling toward the junction
      for b in self.adj[j]:
        if b == a or b not in self.out[j]:
          continue
        exit_pts = self.walk(j, b, exit_len)
        angle = wrap(self.arm_heading(exit_pts) - h_in)
        if abs(abs(math.degrees(angle)) - 90) <= tolerance_deg:
          out.append(Turn(j, self.pos[j], approach, exit_pts, angle, self.lanes_toward(j, a), self.radius[a]))
    return out
