"""One short drive per maneuver of a route, to find the junctions the model takes before a whole route depends on them.

A full routed drive stops telling anything at its first missed turn, since the reroute goes elsewhere. Here every turn
and keep of the route (as Valhalla gives it from the route's start) becomes its own route: a start APPROACH_M before it
on the route, in the lane for the maneuver, and a destination AFTER_M past it.
"""
import math
import os

import numpy as np

from openpilot.selfdrive.navigation.route import cumulative_distance
from openpilot.selfdrive.turnpilot.route_follower import KEEP, LEFT, TURN, classify
from openpilot.selfdrive.turnpilot.routing.valhalla import ValhallaSource
from openpilot.tools.sim.bridge.beamng.frame import from_latlon, to_latlon
from openpilot.tools.sim.bridge.beamng.roadgraph import RoadGraph
from openpilot.tools.sim.bridge.beamng.routes import Route

TILES = os.path.expanduser("~/turnpilot_data/valhalla/west_coast_usa")
APPROACH_M = 130.
MIN_APPROACH_M = 60.
AFTER_M = 70.


def lane_offset(graph: RoadGraph, p, heading: float, side: int, z_ref: float) -> tuple[float, float]:
  """(meters right of the road graph's centreline, z) for the lane a maneuver to `side` leaves from. Of the roads
  along the heading near p, the one nearest z_ref (the junction's height): tunnels and bridges cross the town."""
  def seg_dist(kv):
    a, b = graph.pos[kv[0][0]][:2], graph.pos[kv[0][1]][:2]
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-9), 0., 1.))
    return float(np.linalg.norm(p - (a + t * ab)))
  near = [kv for kv in graph.links.items() if seg_dist(kv) < 12.]
  near.sort(key=lambda kv: abs(float(graph.pos[kv[0][0]][2]) - z_ref))
  for (a, b), _link in near:
    d = graph.pos[b][:2] - graph.pos[a][:2]
    if abs(math.cos(math.atan2(d[1], d[0]) - heading)) < 0.8:
      continue  # a crossing road
    first, j = (a, b) if math.cos(math.atan2(d[1], d[0]) - heading) > 0 else (b, a)
    lanes = graph.lanes_toward(j, first)
    ours = [k for k, c in enumerate(lanes) if c == '+'] or [len(lanes) - 1]
    lane = ours[0] if side == LEFT else ours[-1]
    radius = graph.radius[first]
    w = 2 * radius / max(len(lanes), 1)
    z = float(graph.pos[first][2])
    return -radius + w * (lane + 0.5), z
  return 1.8, float(graph.pos[min(graph.pos, key=lambda n: float(np.linalg.norm(graph.pos[n][:2] - p)))][2])


def screen_routes(route: Route, graph: RoadGraph | None = None, source: ValhallaSource | None = None) -> list[Route]:
  graph = graph or RoadGraph()
  source = source or ValhallaSource(TILES)
  out = []
  start, heading = route.start, route.heading
  for leg_i, (lat, lon) in enumerate(route.legs):
    slat, slon, _ = to_latlon(start[0], start[1])
    steps = source.route((slon, slat), (lon, lat), (90. - math.degrees(heading)) % 360, None)
    coords = np.concatenate([np.array(s['geometry']['coordinates']) for s in steps])
    xy = np.array([from_latlon(c[1], c[0])[:2] for c in coords])
    along = cumulative_distance(coords)
    s_at, prev_s = 0., 0.
    for i, step in enumerate(steps):
      m = step['maneuver']
      kind, side = classify(m.get('type', ''), m.get('modifier', ''))
      if i and kind in (TURN, KEEP):
        s0 = max(s_at - APPROACH_M, prev_s + 15.)
        if s_at - s0 >= MIN_APPROACH_M:
          p = np.array([np.interp(s0, along, xy[:, 0]), np.interp(s0, along, xy[:, 1])])
          q = np.array([np.interp(s0 + 5., along, xy[:, 0]), np.interp(s0 + 5., along, xy[:, 1])])
          h = math.atan2(q[1] - p[1], q[0] - p[0])
          jxy = xy[min(np.searchsorted(along, s_at), len(xy) - 1)]
          z_j = float(graph.pos[min(graph.pos, key=lambda n: float(np.linalg.norm(graph.pos[n][:2] - jxy)))][2])
          offset, z = lane_offset(graph, p, h, side, z_j)
          x, y = p[0] + offset * math.sin(h), p[1] - offset * math.cos(h)
          s1 = min(s_at + AFTER_M, along[-1])
          e = (float(np.interp(s1, along, coords[:, 1])), float(np.interp(s1, along, coords[:, 0])))
          name = f"{route.name}-L{leg_i + 1}-m{i}-{m.get('type')}-{m.get('modifier', '').replace(' ', '_')}"
          out.append(Route(name, (round(float(x), 2), round(float(y), 2), round(z + 0.5, 2)), round(h, 4), (e,),
                           f"{m.get('type')} {m.get('modifier', '')} at ({xy[np.searchsorted(along, s_at)][0]:.0f}, " +
                           f"{xy[np.searchsorted(along, s_at)][1]:.0f}), {s_at - s0:.0f} m approach"))
        prev_s = s_at
      s_at += step['distance']
    end = from_latlon(lat, lon)
    start = (end[0], end[1], start[2])
    heading = math.atan2(xy[-1][1] - xy[-6][1], xy[-1][0] - xy[-6][0])
  return out


if __name__ == "__main__":
  import sys
  from openpilot.tools.sim.bridge.beamng.routes import ROUTES
  for r in screen_routes(ROUTES[sys.argv[1] if len(sys.argv) > 1 else 'loop']):
    print(r.name, r.start, r.heading, r.maneuvers)
