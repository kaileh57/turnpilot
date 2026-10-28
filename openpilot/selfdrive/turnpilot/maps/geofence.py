"""The drivable geofence: the map server's coverage GeoJSON, and which region a point is in.

Coordinates are (longitude, latitude) as in GeoJSON. The regions are a few hundred km across, so plain planar
ray casting on degrees is exact enough; a point on a ring's edge or vertex counts as inside.
"""
from dataclasses import dataclass

EDGE_EPSILON = 1e-12  # degrees squared; a cross product this small puts the point on the edge


def on_segment(x, y, a, b) -> bool:
  (x0, y0), (x1, y1) = a, b
  cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
  if abs(cross) > EDGE_EPSILON:
    return False
  return min(x0, x1) - EDGE_EPSILON <= x <= max(x0, x1) + EDGE_EPSILON and min(y0, y1) - EDGE_EPSILON <= y <= max(y0, y1) + EDGE_EPSILON


def in_ring(x, y, ring) -> bool | None:
  """True inside, False outside, None on the boundary. The ring may or may not repeat its first point."""
  inside = False
  n = len(ring)
  for i in range(n):
    a, b = ring[i], ring[(i + 1) % n]
    if on_segment(x, y, a, b):
      return None
    (x0, y0), (x1, y1) = a, b
    if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
      inside = not inside
  return inside


def in_polygon(x, y, rings) -> bool:
  """A GeoJSON Polygon's coordinates: an outer ring, then holes. A hole's edge is still drivable."""
  if not rings:
    return False
  outer = in_ring(x, y, rings[0])
  if outer is False:
    return False
  if outer is None:
    return True
  return not any(in_ring(x, y, hole) is True for hole in rings[1:])


def in_geometry(lon, lat, geometry) -> bool:
  kind = geometry.get('type')
  if kind == 'Polygon':
    return in_polygon(lon, lat, geometry['coordinates'])
  if kind == 'MultiPolygon':
    return any(in_polygon(lon, lat, rings) for rings in geometry['coordinates'])
  raise ValueError(f"coverage geometry must be a Polygon or MultiPolygon, not {kind!r}")


def bounds(geometry) -> tuple[float, float, float, float]:
  """(west, south, east, north) of a Polygon or MultiPolygon."""
  polygons = [geometry['coordinates']] if geometry['type'] == 'Polygon' else geometry['coordinates']
  xs = [p[0] for rings in polygons for ring in rings for p in ring]
  ys = [p[1] for rings in polygons for ring in rings for p in ring]
  return min(xs), min(ys), max(xs), max(ys)


@dataclass(frozen=True)
class Region:
  id: str
  name: str
  geometry: dict

  def contains(self, lon, lat) -> bool:
    return in_geometry(lon, lat, self.geometry)


def place_name(region_name: str) -> str:
  """'Indianapolis area' -> 'Indianapolis', for "around Indianapolis and Seattle"."""
  name = region_name.strip()
  return name[:-len(' area')] if name.lower().endswith(' area') else name


def join_names(names: list[str]) -> str:
  if len(names) <= 1:
    return ''.join(names)
  return ', '.join(names[:-1]) + ' and ' + names[-1]


class Coverage:
  """The parsed /v1/coverage FeatureCollection. Raises ValueError on anything malformed, so a bad cache is loud."""

  def __init__(self, data: dict):
    if not isinstance(data, dict) or data.get('type') != 'FeatureCollection' or not isinstance(data.get('features'), list):
      raise ValueError("coverage is not a GeoJSON FeatureCollection")
    self.version = str(data.get('version') or '')
    if not self.version:
      raise ValueError("coverage has no version")
    regions = []
    for feature in data['features']:
      props, geometry = feature.get('properties') or {}, feature.get('geometry') or {}
      if not props.get('id'):
        raise ValueError("a coverage region has no id")
      in_geometry(0., 0., geometry)  # raises for a geometry type we don't handle
      regions.append(Region(str(props['id']), str(props.get('name') or props['id']), geometry))
    if not regions:
      raise ValueError("coverage has no regions")
    self.regions = regions
    self.data = data

  def region_at(self, lon, lat) -> Region | None:
    return next((r for r in self.regions if r.contains(lon, lat)), None)

  def names(self) -> str:
    """'Indianapolis and Seattle'."""
    return join_names([place_name(r.name) for r in self.regions])

  def outside_message(self) -> str:
    return f"Outside TurnPilot's mapped area. TurnPilot can only navigate around {self.names()}."

  def summary(self) -> list[dict]:
    return [{'id': r.id, 'name': r.name} for r in self.regions]
