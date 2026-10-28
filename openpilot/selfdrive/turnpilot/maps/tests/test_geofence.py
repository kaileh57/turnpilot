import json
import os
import time

import pytest

from openpilot.selfdrive.turnpilot.maps.client import MapError
from openpilot.selfdrive.turnpilot.maps.coverage import CoverageCache, check_trip, coverage_path, save
from openpilot.selfdrive.turnpilot.maps.geofence import Coverage, in_geometry, in_polygon, in_ring
from openpilot.selfdrive.turnpilot.maps.tests.fakes import CHICAGO, INDY, INDY_EAST, SEATTLE, coverage_doc

SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
HOLE = [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]]


def test_ring_inside_outside_and_edges():
  assert in_ring(5, 5, SQUARE) is True
  assert in_ring(11, 5, SQUARE) is False
  assert in_ring(-0.0001, 5, SQUARE) is False
  for edge in [(0, 5), (10, 5), (5, 0), (5, 10)]:  # on a side
    assert in_ring(*edge, SQUARE) is None, edge
  for vertex in [(0, 0), (10, 10), (0, 10)]:
    assert in_ring(*vertex, SQUARE) is None, vertex
  assert in_ring(5, 5, SQUARE[:-1]) is True  # an open ring works too
  # a ray through a vertex counts it once: a diamond, with the point level with its side vertices
  diamond = [[0, -5], [5, 0], [0, 5], [-5, 0]]
  assert in_ring(0, 0, diamond) is True
  assert in_ring(-6, 0, diamond) is False
  assert in_ring(6, 0, diamond) is False


def test_concave_polygon():
  u = [[0, 0], [9, 0], [9, 9], [6, 9], [6, 3], [3, 3], [3, 9], [0, 9]]  # a U, open at the top middle
  assert in_ring(1, 8, u) and in_ring(8, 8, u) and in_ring(4.5, 1, u)
  assert in_ring(4.5, 6, u) is False


def test_polygon_with_hole():
  assert in_polygon(2, 2, [SQUARE, HOLE])
  assert not in_polygon(5, 5, [SQUARE, HOLE])
  assert in_polygon(4, 5, [SQUARE, HOLE])  # the hole's edge is still drivable
  assert in_polygon(0, 0, [SQUARE, HOLE])  # the outer edge counts as inside
  assert not in_polygon(1, 1, [])


def test_multipolygon():
  multi = {'type': 'MultiPolygon', 'coordinates': [[SQUARE], [[[20, 0], [30, 0], [30, 10], [20, 10], [20, 0]]]]}
  assert in_geometry(5, 5, multi)
  assert in_geometry(25, 5, multi)
  assert in_geometry(20, 5, multi)  # an edge of the second part
  assert not in_geometry(15, 5, multi)  # between the parts
  with pytest.raises(ValueError):
    in_geometry(0, 0, {'type': 'Point', 'coordinates': [0, 0]})


def test_coverage_regions_and_messages():
  coverage = Coverage(coverage_doc())
  assert coverage.region_at(*INDY).id == 'indianapolis'
  assert coverage.region_at(*SEATTLE).name == 'Seattle area'
  assert coverage.region_at(*CHICAGO) is None
  assert coverage.names() == 'Indianapolis and Seattle'
  assert coverage.outside_message() == "Outside TurnPilot's mapped area. TurnPilot can only navigate around Indianapolis and Seattle."


@pytest.mark.parametrize('bad', [
  {}, {'type': 'FeatureCollection', 'features': []}, {'type': 'FeatureCollection', 'version': 'v', 'features': []},
  {'type': 'FeatureCollection', 'version': 'v', 'features': [{'properties': {}, 'geometry': {'type': 'Polygon', 'coordinates': [SQUARE]}}]},
  {'type': 'FeatureCollection', 'version': 'v', 'features': [{'properties': {'id': 'x'}, 'geometry': {'type': 'LineString'}}]},
])
def test_malformed_coverage_is_refused(bad):
  with pytest.raises(ValueError):
    Coverage(bad)


def test_check_trip():
  coverage = Coverage(coverage_doc())
  assert check_trip(coverage, INDY, INDY_EAST).id == 'indianapolis'
  assert check_trip(coverage, None, SEATTLE).id == 'seattle'  # the car's position unknown: the destination decides
  with pytest.raises(MapError) as e:
    check_trip(coverage, INDY, CHICAGO)
  assert e.value.code == 'outside_coverage' and 'Indianapolis and Seattle' in e.value.message and e.value.message.startswith('Outside')
  with pytest.raises(MapError) as e:
    check_trip(coverage, CHICAGO, INDY)
  assert e.value.code == 'outside_coverage' and e.value.message.startswith('The car is outside')
  with pytest.raises(MapError) as e:
    check_trip(coverage, SEATTLE, INDY)
  assert e.value.code == 'different_regions'


def test_cache_roundtrip_and_corruption(tmp_path):
  cache = CoverageCache(tmp_path)
  assert cache.get() is None
  with pytest.raises(MapError) as e:
    cache.require()
  assert e.value.code == 'no_coverage'
  save(coverage_doc(version='v1'), tmp_path)
  assert cache.get().version == 'v1'
  assert abs(cache.fetched_at - time.time()) < 60  # noqa: TID251
  save(coverage_doc(version='v2'), tmp_path)
  stat = coverage_path(tmp_path).stat()
  os.utime(coverage_path(tmp_path), ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))  # same-second writes still reload
  assert cache.get().version == 'v2'
  with pytest.raises(ValueError):
    save({'type': 'nope'}, tmp_path)  # a malformed answer never replaces a good cache
  assert json.loads(coverage_path(tmp_path).read_text())['coverage']['version'] == 'v2'
  coverage_path(tmp_path).write_text('{not json')
  with pytest.raises(MapError) as e:
    cache.get()
  assert e.value.code == 'bad_coverage'
