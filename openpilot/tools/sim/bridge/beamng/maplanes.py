"""Turn lanes for the west_coast_usa road-graph export, so that sim routes carry navInstruction.lanes.

export_osm.py (on the PC) tags every way with lanes, and lanes:forward / lanes:backward on two-way ways, from the road
graph's per-link lane strings. Nothing says which lane goes where, and Valhalla reports lanes only from turn:lanes. This adds
turn:lanes (turn:lanes:forward or :backward on a two-way way) to every public way with two or more lanes in its direction
of travel, where it ends at a junction with at least two kinds of exit. The exits come from the geometry: roads a car can
leave on, public and paved, that turn left or right by more than 30 degrees or go on within 30 degrees (and not back).

The leftmost lane takes the left turn, the rightmost takes the right turn and goes through, and the rest go through. Where
the junction has no through road (a T) the left half turns left and the right half right:

    left | through | through;right      left | through;right      left | through      through | through;right
    left | right      left | left;right | right

Private roads, tracks and unpaved roads get nothing, and neither does a way whose junction offers one kind of exit. The
export has no motorway_link, so a freeway diverging by more than 30 degrees counts as a left or right exit, which puts the
exit lane where the geometry says it is. It only reads the export: the lanes tags are never changed.

  python -m openpilot.tools.sim.bridge.beamng.maplanes [west_coast_usa.osm [west_coast_usa_lanes.osm]]
"""
import argparse
import math
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from openpilot.tools.sim.bridge.beamng import frame

EXPORT = os.path.expanduser("~/pc_relay/incoming/osm_export/west_coast_usa.osm")
OUT = os.path.expanduser("~/turnpilot_data/osm/west_coast_usa_lanes.osm")
THROUGH_DEG = 30.0  # an exit within this of straight on goes through
UTURN_DEG = 150.0  # and one past this turns back on itself
REACH_M = 15.0  # how far along a road its direction is measured at a junction


@dataclass
class Way:
  id: int
  nodes: list[int]
  tags: dict[str, str]
  element: ET.Element | None = field(default=None, repr=False)

  @property
  def oneway(self) -> bool:
    return self.tags.get("oneway") == "yes"

  def lanes(self, forward: bool) -> int:
    """Lanes going along the way's node order (forward) or against it."""
    if self.oneway:
      return int(self.tags.get("lanes", 1)) if forward else 0
    return int(self.tags.get("lanes:forward" if forward else "lanes:backward", 0))

  def public(self) -> bool:
    t = self.tags
    return t.get("access") != "private" and t.get("surface") == "paved" and t.get("highway") != "track"


class OsmMap:
  """The export's nodes in BeamNG meters (x east, y north), z from the ele tag, and its ways."""

  def __init__(self, path: str):
    self.tree = ET.parse(path)
    self.xy: dict[int, tuple[float, float]] = {}
    self.z: dict[int, float] = {}
    self.names: dict[int, str] = {}
    self.ways: list[Way] = []
    for el in self.tree.getroot():
      tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
      if el.tag == "node":
        i = int(el.get("id"))
        x, y, _ = frame.from_latlon(float(el.get("lat")), float(el.get("lon")))
        self.xy[i], self.z[i], self.names[i] = (x, y), float(tags.get("ele", 0.0)), tags.get("beamng:node", "")
      elif el.tag == "way":
        self.ways.append(Way(int(el.get("id")), [int(nd.get("ref")) for nd in el.findall("nd")], tags, el))
    self.starts: dict[int, list[Way]] = defaultdict(list)  # ways beginning at a node
    self.ends: dict[int, list[Way]] = defaultdict(list)  # and ways ending there
    for way in self.ways:
      self.starts[way.nodes[0]].append(way)
      self.ends[way.nodes[-1]].append(way)

  def points(self, way: Way, forward: bool = True) -> list[tuple[float, float]]:
    pts = [self.xy[n] for n in way.nodes]
    return pts if forward else pts[::-1]

  def length(self, way: Way) -> float:
    return sum(math.dist(a, b) for a, b in zip(self.points(way), self.points(way)[1:], strict=False))


def heading_along(pts, reach: float = REACH_M) -> float:
  """Heading of the first `reach` m of a polyline, radians counterclockwise from east."""
  d = 0.0
  end = pts[-1]
  for a, b in zip(pts, pts[1:], strict=False):
    d += math.dist(a, b)
    end = b
    if d >= reach:
      break
  return math.atan2(end[1] - pts[0][1], end[0] - pts[0][0])


def wrap(a: float) -> float:
  return (a + math.pi) % (2 * math.pi) - math.pi


def turn_kind(delta: float) -> str | None:
  """'left', 'through' or 'right' for an exit leaving at delta radians (counterclockwise) from the way in, None for a U-turn."""
  deg = math.degrees(wrap(delta))
  if abs(deg) <= THROUGH_DEG:
    return "through"
  if abs(deg) >= UTURN_DEG:
    return None
  return "left" if deg > 0 else "right"


def exits(osm: OsmMap, node: int):
  """(way, forward) for every public way a car can leave `node` on."""
  for way in osm.starts[node]:
    if way.public() and way.lanes(True):
      yield way, True
  for way in osm.ends[node]:
    if way.public() and way.lanes(False):
      yield way, False


def entering(osm: OsmMap, node: int):
  """(way, forward) for every public way a car can arrive at `node` on."""
  for way in osm.ends[node]:
    if way.public() and way.lanes(True):
      yield way, True
  for way in osm.starts[node]:
    if way.public() and way.lanes(False):
      yield way, False


def exit_kinds(osm: OsmMap, way: Way, forward: bool) -> dict[str, list[tuple[Way, bool]]]:
  """The exits at the end of `way` (in its direction of travel) by kind, with the way and direction of each."""
  node = way.nodes[-1] if forward else way.nodes[0]
  approach = osm.points(way, forward)
  heading_in = wrap(heading_along(approach[::-1]) + math.pi)  # the way we are driving, into the junction
  kinds = defaultdict(list)
  for other, other_forward in exits(osm, node):
    if other is way and other_forward != forward:
      continue
    kind = turn_kind(heading_along(osm.points(other, other_forward)) - heading_in)
    if kind:
      kinds[kind].append((other, other_forward))
  return kinds


def lane_string(n: int, kinds) -> str | None:
  """turn:lanes for n lanes into a junction with exits of these kinds, None if the lanes have no choice to show."""
  left, through, right = (k in kinds for k in ("left", "through", "right"))
  if n < 2 or left + through + right < 2:
    return None
  lanes: list[set[str]] = [set() for _ in range(n)]
  if through:
    for lane in lanes[1 if left else 0:]:
      lane.add("through")
    if left:
      lanes[0].add("left")
    if right:
      lanes[-1].add("right")
  else:  # a T junction: the left half turns left, the right half right, and an odd middle lane either
    for lane in lanes[:(n + 1) // 2]:
      lane.add("left")
    for lane in lanes[n // 2:]:
      lane.add("right")
  return "|".join(";".join(k for k in ("left", "through", "right") if k in lane) for lane in lanes)


def way_turn_lanes(osm: OsmMap, way: Way) -> dict[str, str]:
  """The turn:lanes tags this way gets, if any."""
  tags = {}
  if not way.public():
    return tags
  for forward in (True, False):
    n = way.lanes(forward)
    value = lane_string(n, exit_kinds(osm, way, forward)) if n >= 2 else None
    if value:
      key = "turn:lanes" if way.oneway else "turn:lanes:" + ("forward" if forward else "backward")
      tags[key] = value
  return tags


def add_turn_lanes(osm: OsmMap) -> Counter:
  """Add the turn lanes to the ways' elements and say how many of each lane string there are."""
  counts = Counter()
  for way in osm.ways:
    for key, value in way_turn_lanes(osm, way).items():
      ET.SubElement(way.element, "tag", k=key, v=value)
      way.tags[key] = value
      counts[value] += 1
  return counts


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("export", nargs="?", default=EXPORT, help="west_coast_usa.osm from export_osm.py")
  ap.add_argument("out", nargs="?", default=OUT)
  args = ap.parse_args()
  osm = OsmMap(args.export)
  counts = add_turn_lanes(osm)
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  osm.tree.write(args.out, encoding="utf-8", xml_declaration=True)
  print(f"{sum(counts.values())} of {len(osm.ways)} ways tagged with turn lanes -> {args.out}")
  for value, n in counts.most_common():
    print(f"  {n:4d}  {value}")


if __name__ == "__main__":
  main()
