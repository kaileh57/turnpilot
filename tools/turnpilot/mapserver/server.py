#!/usr/bin/env python3
"""TurnPilot's map server: the drivable areas, and the Valhalla tiles a drive needs.

  GET  /v1/health     {"ok": true, "regions": [...]}, no key needed
  GET  /v1/coverage   the drivable areas as a GeoJSON FeatureCollection, with a "version"
  POST /v1/corridor   {"start": [lon, lat], "destination": [lon, lat]} -> a tar of manifest.json and tiles/

Every /v1 call but health needs "Authorization: Bearer <key>". A corridor is every tile within BUFFER_KM of
the route, so the car can reroute locally without a connection. Start and destination must both be inside
one region's fence, or the server refuses with a 422 that says why.

Each region is a directory under --regions, written by build_region.py:
  fence.geojson  the drivable area, a Feature with a Polygon or MultiPolygon
  meta.json      {"id", "name", "osm_timestamp", "built_at"}
  index.json     {"tiles/2/000/756/425.gph": [level, index, sha256, size], ...}
  tiles/         the tile hierarchy
  tiles.tar      the same tiles as one extract, which Valhalla routes from
"""
import argparse
import gzip
import hashlib
import hmac
import io
import json
import logging
import ssl
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
  from tools.turnpilot.mapserver import geo, tiles
except ImportError:  # run as a plain script on the server
  import geo  # type: ignore[no-redef]
  import tiles  # type: ignore[no-redef]

BUFFER_KM = 15.0  # with 28 km level-2 tiles this keeps every road within 15-40 km of the route
GZIP_LEVEL = 3
MAX_BODY = 4096
CACHE_BYTES = 150 * 1024 * 1024  # Valhalla's tile cache; the server has 1 GB

log = logging.getLogger("turnpilot-maps")


class Refusal(Exception):
  def __init__(self, code: int, body: dict):
    super().__init__(body.get('error', ''))
    self.code, self.body = code, body


class Region:
  def __init__(self, path: Path, router=None):
    self.path = path
    self.fence = json.loads((path / 'fence.geojson').read_text())
    self.meta = json.loads((path / 'meta.json').read_text())
    self.id = self.meta['id']
    self.index = {rel: tuple(v) for rel, v in json.loads((path / 'index.json').read_text()).items()}
    self.router = router
    self.lock = threading.Lock()

  def contains(self, lon, lat) -> bool:
    return geo.in_geometry(lon, lat, self.fence['geometry'])

  def feature(self) -> dict:
    return {'type': 'Feature', 'geometry': self.fence['geometry'],
            'properties': {k: self.meta.get(k) for k in ('id', 'name', 'osm_timestamp', 'built_at')}}

  def route_line(self, start, destination) -> tuple[list[tuple[float, float]], float]:
    """The route's line and its length in km."""
    if self.router is None:
      import valhalla
      config = valhalla.get_config(tile_extract=str(self.path / 'tiles.tar'), tile_dir=str(self.path / 'tiles'))
      config['mjolnir']['max_cache_size'] = CACHE_BYTES
      self.router = valhalla.Actor(config)
    request = {
      'locations': [{'lon': start[0], 'lat': start[1]}, {'lon': destination[0], 'lat': destination[1]}],
      'costing': 'auto', 'format': 'osrm', 'shape_format': 'geojson',
    }
    with self.lock:
      try:
        r = self.router.route(request)
      except RuntimeError as e:  # Valhalla raises for "no path" and "no suitable edges near location"
        raise Refusal(422, {'error': 'no_route', 'detail': str(e)[:200]}) from e
    if isinstance(r, str):
      r = json.loads(r)
    routes = r.get('routes') or []
    if not routes:
      raise Refusal(422, {'error': 'no_route'})
    line = [tuple(c) for c in routes[0]['geometry']['coordinates']]
    km = routes[0].get('distance', 0) / 1000
    return line, km


class Maps:
  def __init__(self, regions: list[Region], key: str):
    self.regions = regions
    self.key = key
    blob = json.dumps([[r.meta, r.fence] for r in regions], sort_keys=True).encode()
    self.version = hashlib.sha256(blob).hexdigest()[:16]

  def coverage(self) -> dict:
    return {'type': 'FeatureCollection', 'version': self.version, 'features': [r.feature() for r in self.regions]}

  def region_for(self, start, destination) -> Region:
    start_in = [r for r in self.regions if r.contains(*start)]
    dest_in = [r for r in self.regions if r.contains(*destination)]
    if not start_in and not dest_in:
      raise Refusal(422, {'error': 'outside_coverage', 'which': 'both'})
    if not start_in:
      raise Refusal(422, {'error': 'outside_coverage', 'which': 'start'})
    if not dest_in:
      raise Refusal(422, {'error': 'outside_coverage', 'which': 'destination'})
    shared = [r for r in start_in if r in dest_in]
    if not shared:
      raise Refusal(422, {'error': 'different_regions'})
    return shared[0]

  def corridor(self, start, destination) -> tuple[Region, dict, list[str]]:
    region = self.region_for(start, destination)
    line, km = region.route_line(start, destination)
    grid = {rel: (v[0], v[1]) for rel, v in region.index.items()}
    paths = tiles.needed(grid, line, BUFFER_KM)
    manifest = {'region': region.id, 'version': self.version, 'route_km': round(km, 2), 'buffer_km': BUFFER_KM,
                'files': {rel: region.index[rel][2] for rel in paths}}
    return region, manifest, paths


def point(value) -> tuple[float, float]:
  """(lon, lat) from a JSON [lon, lat], or a 400."""
  try:
    lon, lat = (float(v) for v in value)
  except (TypeError, ValueError) as e:
    raise Refusal(400, {'error': 'bad_request', 'detail': 'points are [lon, lat]'}) from e
  if not (-180 <= lon <= 180 and -90 <= lat <= 90):
    raise Refusal(400, {'error': 'bad_request', 'detail': 'point out of range'})
  return lon, lat


def make_handler(maps: Maps):
  class Handler(BaseHTTPRequestHandler):
    server_version = 'turnpilot-maps'
    sys_version = ''

    def log_message(self, format, *args):  # noqa: A002
      log.info("%s %s", self.address_string(), format % args)

    def send_json(self, code, obj):
      body = json.dumps(obj).encode()
      self.send_response(code)
      self.send_header('Content-Type', 'application/json')
      self.send_header('Content-Length', str(len(body)))
      self.end_headers()
      self.wfile.write(body)

    def authorized(self) -> bool:
      header = self.headers.get('Authorization', '')
      given = header[7:] if header.startswith('Bearer ') else ''
      return bool(maps.key) and hmac.compare_digest(given.encode(), maps.key.encode())

    def handle_request(self, method):
      try:
        if method == 'GET' and self.path == '/v1/health':
          return self.send_json(200, {'ok': True, 'regions': [r.id for r in maps.regions]})
        if not self.path.startswith('/v1/'):
          raise Refusal(404, {'error': 'not_found'})
        if not self.authorized():
          raise Refusal(401, {'error': 'unauthorized'})
        if method == 'GET' and self.path == '/v1/coverage':
          return self.send_json(200, maps.coverage())
        if method == 'POST' and self.path == '/v1/corridor':
          return self.corridor()
        raise Refusal(404, {'error': 'not_found'})
      except Refusal as e:
        self.send_json(e.code, e.body)
      except Exception:
        log.exception("request failed")
        self.send_json(500, {'error': 'server_error'})

    def corridor(self):
      length = int(self.headers.get('Content-Length') or 0)
      if not 0 < length <= MAX_BODY:
        raise Refusal(400, {'error': 'bad_request', 'detail': 'body missing or too large'})
      try:
        body = json.loads(self.rfile.read(length))
      except ValueError as e:
        raise Refusal(400, {'error': 'bad_request', 'detail': 'body is not JSON'}) from e
      if not isinstance(body, dict):
        raise Refusal(400, {'error': 'bad_request'})
      region, manifest, paths = maps.corridor(point(body.get('start')), point(body.get('destination')))
      size = sum(region.index[rel][3] for rel in paths)
      log.info("corridor %s: %.1f km, %d tiles, %.1f MB", region.id, manifest['route_km'], len(paths), size / 1e6)

      gz = 'gzip' in self.headers.get('Accept-Encoding', '')
      self.send_response(200)
      self.send_header('Content-Type', 'application/x-tar')
      if gz:  # tiles compress about 2.8 to 1, and the car downloads them over cellular
        self.send_header('Content-Encoding', 'gzip')
      self.send_header('X-TurnPilot-Region', region.id)
      self.send_header('Connection', 'close')
      self.end_headers()
      self.close_connection = True
      out = gzip.GzipFile(fileobj=self.wfile, mode='wb', compresslevel=GZIP_LEVEL) if gz else self.wfile
      with tarfile.open(fileobj=out, mode='w|') as tar:
        data = json.dumps(manifest).encode()
        info = tarfile.TarInfo('manifest.json')
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        for rel in paths:
          tar.add(region.path / rel, arcname=rel, recursive=False)
      if gz:
        out.close()

    def do_GET(self):
      self.handle_request('GET')

    def do_POST(self):
      self.handle_request('POST')

  return Handler


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--regions', type=Path, default=Path('/srv/turnpilot-maps/regions'))
  ap.add_argument('--key-file', type=Path, default=Path('/etc/turnpilot-maps/key'))
  ap.add_argument('--cert', type=Path, default=Path('/etc/turnpilot-maps/cert.pem'))
  ap.add_argument('--cert-key', type=Path, default=Path('/etc/turnpilot-maps/cert.key'))
  ap.add_argument('--host', default='0.0.0.0')
  ap.add_argument('--port', type=int, default=443)
  args = ap.parse_args()
  logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

  key = args.key_file.read_text().strip()
  if len(key) < 32:
    raise SystemExit(f"{args.key_file} must hold a key of at least 32 characters")
  regions = [Region(p) for p in sorted(args.regions.iterdir()) if (p / 'meta.json').is_file()]
  if not regions:
    raise SystemExit(f"no regions under {args.regions}")
  maps = Maps(regions, key)
  log.info("serving %s, coverage version %s", [r.id for r in regions], maps.version)

  httpd = ThreadingHTTPServer((args.host, args.port), make_handler(maps))
  context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
  context.minimum_version = ssl.TLSVersion.TLSv1_2
  context.load_cert_chain(args.cert, args.cert_key)
  httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
  httpd.serve_forever()


if __name__ == '__main__':
  main()
