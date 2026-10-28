"""Point-in-polygon on GeoJSON Polygons and MultiPolygons, in lon/lat. No dependencies."""


def in_ring(lon, lat, ring) -> bool:
  inside = False
  for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1], strict=True):
    if (y0 > lat) != (y1 > lat) and lon < x0 + (lat - y0) * (x1 - x0) / (y1 - y0):
      inside = not inside
  return inside


def in_polygon(lon, lat, rings) -> bool:
  """Inside the outer ring and outside every hole."""
  return bool(rings) and in_ring(lon, lat, rings[0]) and not any(in_ring(lon, lat, hole) for hole in rings[1:])


def in_geometry(lon, lat, geometry) -> bool:
  if geometry['type'] == 'Polygon':
    return in_polygon(lon, lat, geometry['coordinates'])
  if geometry['type'] == 'MultiPolygon':
    return any(in_polygon(lon, lat, rings) for rings in geometry['coordinates'])
  raise ValueError(f"unsupported geometry {geometry['type']}")
