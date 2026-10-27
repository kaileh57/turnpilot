# Copied from ~/pc_relay/incoming/osm_export/frame.py (written by piston, the agent on the BeamNG PC, for the
# west_coast_usa road-graph export), re-indented to two spaces. Keep it in step with that export: the routing graph
# and the bridge GPS must use the same projection.
"""BeamNG world <-> WGS84 for the sim's road export and the bridge's GPS.

BeamNG x = east, y = north, z = up (metres) are local ENU coordinates on the tangent plane at ORIGIN (h = 0 on
the WGS84 ellipsoid). to_latlon: ENU -> ECEF -> geodetic (Bowring start, three Newton steps; sub-mm for
|x|, |y| < 50 km). from_latlon is its exact inverse: geodetic -> ECEF -> ENU. Pass z / h to round-trip exactly;
with z = 0 the horizontal error from dropping h is below 1 mm over a 10 km map.
"""
import math

ORIGIN = (39.7684, -86.1581)       # placeholder (Indianapolis), so US / right-hand defaults apply
A = 6378137.0                       # WGS84 semi-major axis, m
F = 1 / 298.257223563
E2 = F * (2 - F)                    # first eccentricity squared
B = A * (1 - F)
EP2 = E2 / (1 - E2)


def _ecef(lat, lon, h):
  p, l = math.radians(lat), math.radians(lon)
  n = A / math.sqrt(1 - E2 * math.sin(p) ** 2)
  return ((n + h) * math.cos(p) * math.cos(l), (n + h) * math.cos(p) * math.sin(l), (n * (1 - E2) + h) * math.sin(p))


_P0, _L0 = math.radians(ORIGIN[0]), math.radians(ORIGIN[1])
_X0 = _ecef(ORIGIN[0], ORIGIN[1], 0.0)
_SP, _CP, _SL, _CL = math.sin(_P0), math.cos(_P0), math.sin(_L0), math.cos(_L0)


def to_latlon(x, y, z=0.0):
  """BeamNG (x east, y north, z up) m -> (lat deg, lon deg, h m above the ellipsoid)."""
  X = _X0[0] - _SL * x - _SP * _CL * y + _CP * _CL * z
  Y = _X0[1] + _CL * x - _SP * _SL * y + _CP * _SL * z
  Z = _X0[2] + _CP * y + _SP * z
  p = math.hypot(X, Y)
  th = math.atan2(Z * A, p * B)
  lat = math.atan2(Z + EP2 * B * math.sin(th) ** 3, p - E2 * A * math.cos(th) ** 3)
  for _ in range(3):
    n = A / math.sqrt(1 - E2 * math.sin(lat) ** 2)
    h = p / math.cos(lat) - n
    lat = math.atan2(Z, p * (1 - E2 * n / (n + h)))
  n = A / math.sqrt(1 - E2 * math.sin(lat) ** 2)
  h = p / math.cos(lat) - n
  return math.degrees(lat), math.degrees(math.atan2(Y, X)), h


def from_latlon(lat, lon, h=0.0):
  """(lat deg, lon deg, h m above the ellipsoid) -> BeamNG (x east, y north, z up) m."""
  X, Y, Z = _ecef(lat, lon, h)
  dx, dy, dz = X - _X0[0], Y - _X0[1], Z - _X0[2]
  return (-_SL * dx + _CL * dy,
      -_SP * _CL * dx - _SP * _SL * dy + _CP * dz,
      _CP * _CL * dx + _CP * _SL * dy + _SP * dz)


if __name__ == "__main__":
  worst = 0.0
  for x in (-5000.0, -1234.5, 0.0, 777.0, 5000.0):
    for y in (-5000.0, 0.0, 2500.0, 5000.0):
      for z in (0.0, 150.0):
        xx, yy, zz = from_latlon(*to_latlon(x, y, z))
        worst = max(worst, abs(xx - x), abs(yy - y), abs(zz - z))
  print(f"round trip worst error {worst:.2e} m")
  assert worst < 1e-6
