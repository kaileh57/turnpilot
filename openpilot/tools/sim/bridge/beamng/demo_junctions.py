"""Junctions on the stock west_coast_usa map for a lane change before a turn: a multi-lane approach, a left (or right)
turn at its end, and a destination that forces it. Each one comes with a start pose, in the lane the car has to leave,
that can be teleported to, and is checked against Valhalla tiles built from maplanes.py's export: the route from the pose
must go straight to the junction, turn there, and carry the approach lanes with the turn's lane active.

  python -m openpilot.tools.sim.bridge.beamng.demo_junctions [--kind left|right] [--tiles DIR] [--top N]

Positions are BeamNG world meters (x east, y north, z up) and headings are radians counterclockwise from east, as in
scenario.teleport_ego; the bearing is the same heading as a compass direction, the one navd's GPS bearing uses. Lane
positions assume the lanes divide the road evenly, as roadgraph.Turn.spawn does.
"""
import argparse
import json
import math
import os
from dataclasses import dataclass

from openpilot.selfdrive.turnpilot.routing.guidance import approach_lanes
from openpilot.selfdrive.turnpilot.routing.valhalla_source import ValhallaSource
from openpilot.tools.sim.bridge.beamng import frame, maplanes
from openpilot.tools.sim.bridge.beamng.maplanes import OsmMap, Way, heading_along, wrap

GRAPH_JSON = os.path.expanduser("~/pc_relay/incoming/osm_export/west_coast_usa_graph.json")
TILES = os.path.expanduser("~/turnpilot_data/valhalla/west_coast_usa_lanes")
MIN_RUN_M = 300.0  # multi-lane road in the travel direction up to the junction
START_M = (400.0, 350.0, 300.0)  # the start pose is this far back (or 10 m short of the start of the lanes), the first that routes
EXIT_M = (150.0, 80.0, 250.0, 400.0)  # and the destination this far along the road the car turns onto
CONTINUE_DEG = 20.0  # a way continues a road if it leaves within this of its heading
TURN_DEG = (60.0, 120.0)  # the turn is about 90 degrees
MAX_START_ERR_M = 25.0  # the route's first leg must be this close to the start's distance from the junction


@dataclass
class Demo:
  kind: str  # 'left' or 'right'
  junction: int  # OSM node id
  node: str  # BeamNG map node
  center: tuple  # BeamNG x, y, z of the junction
  highway: str
  maxspeed: str
  oneway: bool
  lanes: int  # in the direction of travel
  run_m: float  # multi-lane road up to the junction
  turn_deg: float  # counterclockwise positive
  start: tuple  # x, y, z of the start pose
  heading: float  # rad counterclockwise from east
  start_lane: int  # from the left among the lanes in the direction of travel, 0-based
  start_m: float  # from the junction
  destination: tuple  # BeamNG x, y, z
  onto: str = ""  # the road the turn leaves on
  route_m: float = 0.0
  verified: str = ""  # empty if the route checks out, else why not

  @property
  def bearing(self) -> float:
    return round(90.0 - math.degrees(self.heading), 4) % 360.0

  def latlon(self, p) -> tuple:
    return frame.to_latlon(*p)[:2]


def run_up_to(osm: OsmMap, way: Way, forward: bool, limit: float = 1500.0):
  """The multi-lane road leading to the end of `way`, as [(way, forward)] in driving order, and its length."""
  chain, length, seen = [(way, forward)], osm.length(way), {(way.id, forward)}
  while length < limit:
    start = chain[0][0].nodes[0] if chain[0][1] else chain[0][0].nodes[-1]
    heading = heading_along(osm.points(*chain[0]))
    best = None
    for other, other_forward in maplanes.entering(osm, start):
      if (other.id, other_forward) in seen or other.lanes(other_forward) < 2:
        continue
      turn = abs(math.degrees(wrap(heading - heading_along(osm.points(other, other_forward)[::-1]) - math.pi)))
      if turn < CONTINUE_DEG and (best is None or turn < best[0]):
        best = (turn, other, other_forward)
    if best is None:
      break
    chain.insert(0, best[1:])
    seen.add((best[1].id, best[2]))
    length += osm.length(best[1])
  return chain, length


def road_points(osm: OsmMap, chain):
  """(x, y, z, way, forward) of every node along a chain of ways, without repeating the nodes where they meet."""
  out = []
  for way, forward in chain:
    nodes = way.nodes if forward else way.nodes[::-1]
    out += [(*osm.xy[n], osm.z[n], n, way, forward) for n in (nodes if not out else nodes[1:])]
  return out


def point_before_end(pts, distance: float):
  """(x, y, z), heading, node index of the segment, for the point `distance` m before the end of a polyline."""
  target = sum(math.dist(a[:2], b[:2]) for a, b in zip(pts, pts[1:], strict=False)) - distance
  walked = 0.0
  for i, (a, b) in enumerate(zip(pts, pts[1:], strict=False)):
    d = math.dist(a[:2], b[:2])
    if walked + d >= target and d > 0:
      f = (target - walked) / d
      return tuple(a[k] + f * (b[k] - a[k]) for k in range(3)), math.atan2(b[1] - a[1], b[0] - a[0]), i
    walked += d
  return pts[-1][:3], 0.0, len(pts) - 2


def lane_position(p, heading: float, half_width: float, total: int, forward_lanes: int, lane: int):
  """The centre of a lane, counted from the left among the lanes in the direction of travel."""
  width = 2 * half_width / total
  offset = -half_width + width * (total - forward_lanes + lane + 0.5)  # to the right of the road's centreline
  return p[0] + offset * math.sin(heading), p[1] - offset * math.cos(heading), p[2] + 0.3


def exit_road(osm: OsmMap, way: Way, forward: bool, length: float):
  """A point `length` m along the road that leaves the junction through (way, forward), staying as straight as it can."""
  chain, total = [(way, forward)], osm.length(way)
  while total < length:
    node = way.nodes[-1] if forward else way.nodes[0]
    heading = heading_along(osm.points(way, forward)[::-1]) + math.pi
    nxt = min(((abs(wrap(heading_along(osm.points(w, f)) - heading)), w, f) for w, f in maplanes.exits(osm, node)
               if w is not way), default=None, key=lambda r: r[0])
    if nxt is None or math.degrees(nxt[0]) > 60:
      break
    way, forward = nxt[1:]
    chain.append((way, forward))
    total += osm.length(way)
  pts = road_points(osm, chain)
  walked, last = 0.0, pts[0]
  for a, b in zip(pts, pts[1:], strict=False):
    d = math.dist(a[:2], b[:2])
    if walked + d >= length and d > 0:
      f = (length - walked) / d
      return tuple(a[k] + f * (b[k] - a[k]) for k in range(3))
    walked += d
    last = b
  return last[:3]


def approaches(osm: OsmMap, kind: str):
  """(way, forward, turn angle in degrees, the road that is turned onto, chain, length) for every approach with a multi-lane
  road of MIN_RUN_M up to a turn of this kind at its end."""
  for way in osm.ways:
    if not way.public():
      continue
    for forward in (True, False):
      kinds = maplanes.exit_kinds(osm, way, forward) if way.lanes(forward) >= 2 else {}
      if kind not in kinds or len(kinds) < 2:
        continue
      heading_in = wrap(heading_along(osm.points(way, forward)[::-1]) + math.pi)
      turns = [(math.degrees(wrap(heading_along(osm.points(w, f)) - heading_in)), (w, f)) for w, f in kinds[kind]]
      angle, onto = min(turns, key=lambda r: abs(abs(r[0]) - 90))
      chain, length = run_up_to(osm, way, forward)
      if TURN_DEG[0] <= abs(angle) <= TURN_DEG[1] and length >= MIN_RUN_M + 10:
        yield way, forward, angle, onto, chain, length


def make_demo(osm: OsmMap, radius: dict, kind: str, approach, start_m: float, exit_m: float) -> Demo:
  way, forward, angle, onto, chain, length = approach
  p, heading, i = point_before_end(road_points(osm, chain), start_m)
  pts = road_points(osm, chain)
  seg_way, seg_forward = pts[i + 1][4], pts[i + 1][5]  # the way the segment between the two nodes belongs to
  half_width = (radius[osm.names[pts[i][3]]] + radius[osm.names[pts[i + 1][3]]]) / 2
  total, ahead = int(seg_way.tags["lanes"]), seg_way.lanes(seg_forward)
  lane = ahead - 1 if kind == "left" else 0  # the curb lane for a left turn, the lane beside the oncoming traffic for a right
  end = way.nodes[-1] if forward else way.nodes[0]
  return Demo(kind, end, osm.names[end], (*osm.xy[end], osm.z[end]), way.tags["highway"], way.tags.get("maxspeed", ""),
              way.oneway, ahead, length, angle, lane_position(p, heading, half_width, total, ahead, lane), heading, lane, start_m,
              exit_road(osm, *onto, exit_m), f"{onto[0].tags['highway']} {onto[0].tags.get('maxspeed', '')}".strip())


def find(osm: OsmMap, radius: dict, source: ValhallaSource, kind: str):
  """A Demo for every approach: the first start and destination that check out, else the first that was tried."""
  for approach in approaches(osm, kind):
    first = None
    for start_m in dict.fromkeys(min(approach[5] - 10, m) for m in START_M):
      for exit_m in EXIT_M:
        demo = verify(make_demo(osm, radius, kind, approach, start_m, exit_m), source)
        first = first or demo
        if not demo.verified:
          break
      if not demo.verified:
        break
    yield demo if not demo.verified else first


def verify(demo: Demo, source: ValhallaSource) -> Demo:
  """Route from the start pose to the destination and say what is wrong with it, if anything."""
  try:
    steps = source.route(demo.latlon(demo.start)[::-1], demo.latlon(demo.destination)[::-1], demo.bearing, "en")
  except Exception as e:
    demo.verified = f"no route: {e}"
    return demo
  demo.route_m = sum(s["distance"] for s in steps)
  turn = steps[1]["maneuver"] if len(steps) > 1 else {}
  lon, lat = (turn.get("location") or [0, 0])[:2]
  where = math.dist(frame.from_latlon(lat, lon)[:2], demo.center[:2])
  lanes = approach_lanes(steps, 0, demo.start_m)
  side = demo.kind
  if where > 20 or turn.get("type") != "turn" or side not in turn.get("modifier", ""):
    demo.verified = f"first maneuver {turn.get('type')} {turn.get('modifier')} is {where:.0f} m from the junction"
  elif abs(steps[0]["distance"] - demo.start_m) > MAX_START_ERR_M:
    demo.verified = f"first leg is {steps[0]['distance']:.0f} m, not {demo.start_m:.0f}"
  elif len(lanes) != demo.lanes:
    demo.verified = f"{len(lanes)} lanes in the route, not {demo.lanes}"
  elif not lanes[0 if side == "left" else -1]["active"] or sum(lane["active"] for lane in lanes) != 1:
    demo.verified = f"active lanes {[lane['active'] for lane in lanes]}"
  return demo


def describe(demo: Demo) -> str:
  s, d, c = demo.start, demo.destination, demo.center
  (dlat, dlon), (clat, clon) = demo.latlon(d), demo.latlon(c)
  road = f"{demo.highway} {demo.maxspeed}, {demo.lanes} lanes {'one-way' if demo.oneway else 'each way'}, {demo.run_m:.0f} m of them"
  return "\n".join([
    f"{demo.kind} turn at {demo.node} ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}), {road}, turn {demo.turn_deg:+.0f} deg onto {demo.onto}",
    f"  start x={s[0]:.1f} y={s[1]:.1f} z={s[2]:.2f} heading={demo.heading:.4f} rad (bearing {demo.bearing:.1f}), "
    + f"{demo.start_m:.0f} m before the junction, in lane {demo.start_lane + 1} of {demo.lanes} from the left",
    f"  destination lat={dlat:.7f} lon={dlon:.7f} (x={d[0]:.1f} y={d[1]:.1f}), junction lat={clat:.7f} lon={clon:.7f}",
    f"  route {demo.route_m:.0f} m, {'OK' if not demo.verified else 'NOT OK: ' + demo.verified}"])


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--kind", choices=("left", "right"), default="left")
  ap.add_argument("--tiles", default=TILES, help="Valhalla tiles built from maplanes.py's output")
  ap.add_argument("--export", default=maplanes.EXPORT)
  ap.add_argument("--graph", default=GRAPH_JSON)
  ap.add_argument("--top", type=int, default=8)
  ap.add_argument("--all", action="store_true", help="also list the ones that don't check out")
  args = ap.parse_args()
  osm = OsmMap(args.export)
  with open(args.graph) as f:
    radius = {k: v["r"] for k, v in json.load(f)["nodes"].items()}
  source = ValhallaSource(args.tiles)
  demos = list(find(osm, radius, source, args.kind))
  demos.sort(key=lambda d: (bool(d.verified), d.oneway, int(d.maxspeed.split()[0] or 99), abs(abs(d.turn_deg) - 90), -d.run_m))
  for demo in [d for d in demos if args.all or not d.verified][:args.top]:
    print(describe(demo))


if __name__ == "__main__":
  main()
