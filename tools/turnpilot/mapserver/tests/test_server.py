import hashlib
import io
import json
import tarfile
import threading
from http.server import ThreadingHTTPServer

import pytest
import requests

from tools.turnpilot.mapserver import geo, tiles
from tools.turnpilot.mapserver.server import Maps, Refusal, Region, make_handler

KEY = 'k' * 40
SQUARE = {'type': 'Polygon', 'coordinates': [[[-87, 39], [-85, 39], [-85, 41], [-87, 41], [-87, 39]]]}


class FakeRouter:
  def __init__(self, line, fail=False):
    self.line, self.fail = line, fail

  def route(self, request):
    if self.fail:
      raise RuntimeError('No path could be found for input')
    return {'routes': [{'distance': 12345.0, 'geometry': {'type': 'LineString', 'coordinates': self.line}}]}


def make_region(root, rid, geometry, files, router):
  d = root / rid
  d.mkdir()
  index = {}
  for rel, data in files.items():
    p = d / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    level, idx = tiles.parse_path(rel.removeprefix('tiles/'))
    index[rel] = [level, idx, hashlib.sha256(data).hexdigest(), len(data)]
  (d / 'fence.geojson').write_text(json.dumps({'type': 'Feature', 'geometry': geometry, 'properties': {}}))
  (d / 'meta.json').write_text(json.dumps({'id': rid, 'name': rid.title(), 'osm_timestamp': 't', 'built_at': 'b'}))
  (d / 'index.json').write_text(json.dumps(index))
  return Region(d, router=router)


def tile_rel(level, lon, lat):
  size = tiles.LEVEL_SIZE[level]
  index = int((lat + 90) // size) * round(360 / size) + int((lon + 180) // size)
  digits = str(index).zfill(3 * -(-len(str(round(360 / size) * round(180 / size) - 1)) // 3))
  return f"tiles/{level}/" + '/'.join(digits[i:i + 3] for i in range(0, len(digits), 3)) + '.gph'


@pytest.fixture
def server(tmp_path):
  near = tile_rel(2, -86.1, 39.8)
  far = tile_rel(2, -80.1, 45.8)
  indy = make_region(tmp_path, 'indianapolis', SQUARE, {near: b'near', far: b'far', tile_rel(0, -86, 40): b'hw'},
                     FakeRouter([[-86.2, 39.7], [-86.0, 39.9]]))
  sea_geom = {'type': 'MultiPolygon', 'coordinates': [[[[-123, 47], [-121, 47], [-121, 48], [-123, 48], [-123, 47]]]]}
  sea = make_region(tmp_path, 'seattle', sea_geom, {}, FakeRouter([], fail=True))
  maps = Maps([indy, sea], KEY)
  httpd = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(maps))
  threading.Thread(target=httpd.serve_forever, daemon=True).start()
  yield f"http://127.0.0.1:{httpd.server_address[1]}", maps, near, far
  httpd.shutdown()


def auth():
  return {'Authorization': f'Bearer {KEY}'}


def test_tile_bbox_matches_real_indiana_tile():
  assert tiles.parse_path('0/002/813.gph') == (0, 2813)
  assert tiles.tile_bbox(0, 2813) == (-88.0, 34.0, -84.0, 38.0)
  assert tiles.parse_path('2/000/756/425.gph') == (2, 756425)
  assert tiles.parse_path('2/000/756/425.txt') is None
  assert tiles.parse_path('../2/000.gph') is None


def test_point_in_polygon_with_hole_and_multipolygon():
  holed = {'type': 'Polygon', 'coordinates': [SQUARE['coordinates'][0], [[-86.5, 39.5], [-85.5, 39.5], [-85.5, 40.5], [-86.5, 40.5]]]}
  assert geo.in_geometry(-86.9, 39.1, holed)
  assert not geo.in_geometry(-86.0, 40.0, holed)
  assert not geo.in_geometry(-84.9, 40.0, holed)
  multi = {'type': 'MultiPolygon', 'coordinates': [SQUARE['coordinates'], [[[0, 0], [1, 0], [1, 1], [0, 1]]]]}
  assert geo.in_geometry(0.5, 0.5, multi)
  assert not geo.in_geometry(2, 2, multi)


def test_health_needs_no_key(server):
  base, *_ = server
  r = requests.get(base + '/v1/health', timeout=5)
  assert r.json() == {'ok': True, 'regions': ['indianapolis', 'seattle']}


def test_everything_else_needs_the_key(server):
  base, *_ = server
  assert requests.get(base + '/v1/coverage', timeout=5).status_code == 401
  bad = {'Authorization': 'Bearer ' + 'x' * 40}
  assert requests.get(base + '/v1/coverage', headers=bad, timeout=5).status_code == 401
  assert requests.post(base + '/v1/corridor', json={}, timeout=5).status_code == 401


def test_coverage(server):
  base, maps, *_ = server
  c = requests.get(base + '/v1/coverage', headers=auth(), timeout=5).json()
  assert c['version'] == maps.version
  assert [f['properties']['id'] for f in c['features']] == ['indianapolis', 'seattle']
  assert c['features'][1]['geometry']['type'] == 'MultiPolygon'


def test_corridor_returns_nearby_tiles_with_checksums(server):
  base, _, near, far = server
  r = requests.post(base + '/v1/corridor', json={'start': [-86.2, 39.7], 'destination': [-86.0, 39.9]},
                    headers=auth(), timeout=5)
  assert r.status_code == 200
  assert r.headers['X-TurnPilot-Region'] == 'indianapolis'
  assert r.headers['Content-Encoding'] == 'gzip'  # requests asks for it and decodes it
  with tarfile.open(fileobj=io.BytesIO(r.content)) as tar:
    names = tar.getnames()
    manifest = json.load(tar.extractfile('manifest.json'))
    assert names[0] == 'manifest.json'
    assert near in names and far not in names
    for rel, sha in manifest['files'].items():
      assert hashlib.sha256(tar.extractfile(rel).read()).hexdigest() == sha
  assert set(manifest['files']) == set(names[1:])
  assert manifest['region'] == 'indianapolis' and manifest['route_km'] == 12.35


@pytest.mark.parametrize('start,dest,which', [
  ([-90, 30], [-86, 40], 'start'),
  ([-86, 40], [-90, 30], 'destination'),
  ([-90, 30], [-91, 31], 'both'),
])
def test_corridor_refuses_outside_the_fence(server, start, dest, which):
  base, *_ = server
  r = requests.post(base + '/v1/corridor', json={'start': start, 'destination': dest}, headers=auth(), timeout=5)
  assert r.status_code == 422
  assert r.json() == {'error': 'outside_coverage', 'which': which}


def test_corridor_refuses_between_regions(server):
  base, *_ = server
  r = requests.post(base + '/v1/corridor', json={'start': [-86, 40], 'destination': [-122, 47.5]}, headers=auth(), timeout=5)
  assert r.status_code == 422 and r.json()['error'] == 'different_regions'


def test_corridor_no_route(server):
  base, *_ = server
  r = requests.post(base + '/v1/corridor', json={'start': [-122.5, 47.5], 'destination': [-122, 47.6]}, headers=auth(), timeout=5)
  assert r.status_code == 422 and r.json()['error'] == 'no_route'


@pytest.mark.parametrize('body', [{'start': 'x', 'destination': [0, 0]}, {'start': [0, 99], 'destination': [0, 0]}, [1, 2]])
def test_corridor_bad_requests(server, body):
  base, *_ = server
  r = requests.post(base + '/v1/corridor', json=body, headers=auth(), timeout=5)
  assert r.status_code == 400


def test_region_for_prefers_a_shared_region(server):
  _, maps, *_ = server
  with pytest.raises(Refusal):
    maps.region_for((-86, 40), (-122, 47.5))
  assert maps.region_for((-86, 40), (-86.5, 40.5)).id == 'indianapolis'
