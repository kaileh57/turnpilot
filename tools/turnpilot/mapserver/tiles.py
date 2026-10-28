"""Valhalla's tile grid, and which tiles a route needs.

Valhalla stores level 0 (highways) in 4 degree squares, level 1 (arterials) in 1 degree squares and level 2
(local roads) in 0.25 degree squares. A tile's path is its level and its index in that grid, with the index
zero-padded to a multiple of three digits and split into directories: tiles/2/000/756/425.gph.
"""
import math
from pathlib import Path

LEVEL_SIZE = {0: 4.0, 1: 1.0, 2: 0.25}
EARTH_KM = 6371.0


def tile_bbox(level: int, index: int) -> tuple[float, float, float, float]:
  """(west, south, east, north) of a tile."""
  size = LEVEL_SIZE[level]
  ncols = round(360 / size)
  row, col = divmod(index, ncols)
  west, south = -180 + col * size, -90 + row * size
  return west, south, west + size, south + size


def parse_path(rel: str) -> tuple[int, int] | None:
  """(level, index) from a path like '2/000/756/425.gph', or None for anything else."""
  parts = Path(rel).with_suffix('').parts
  if len(parts) < 2 or Path(rel).suffix != '.gph' or not all(p.isdigit() for p in parts):
    return None
  level = int(parts[0])
  if level not in LEVEL_SIZE:
    return None
  return level, int(''.join(parts[1:]))


def densify(points, step_km=1.0):
  """The polyline's points with extra points so no two neighbours are more than step_km apart."""
  out = []
  for (lon0, lat0), (lon1, lat1) in zip(points, points[1:], strict=False):
    n = max(1, math.ceil(haversine_km(lon0, lat0, lon1, lat1) / step_km))
    out.extend((lon0 + (lon1 - lon0) * i / n, lat0 + (lat1 - lat0) * i / n) for i in range(n))
  if points:
    out.append(tuple(points[-1]))
  return out


def haversine_km(lon0, lat0, lon1, lat1) -> float:
  p0, p1 = math.radians(lat0), math.radians(lat1)
  a = math.sin((p1 - p0) / 2) ** 2 + math.cos(p0) * math.cos(p1) * math.sin(math.radians(lon1 - lon0) / 2) ** 2
  return 2 * EARTH_KM * math.asin(math.sqrt(a))


def needed(tiles: dict[str, tuple[int, int]], route, buffer_km: float) -> list[str]:
  """The paths of the tiles that come within buffer_km of the route. tiles maps path to (level, index)."""
  points = densify(route)
  out = []
  for rel, (level, index) in tiles.items():
    west, south, east, north = tile_bbox(level, index)
    for lon, lat in points:
      dlat = buffer_km / 111.0
      dlon = buffer_km / (111.0 * max(math.cos(math.radians(lat)), 0.05))
      if west - dlon <= lon <= east + dlon and south - dlat <= lat <= north + dlat:
        out.append(rel)
        break
  return sorted(out)
