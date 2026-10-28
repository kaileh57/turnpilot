"""Where navd's route comes from: Valhalla on a corridor of tiles from TurnPilot's map server, or on local tiles.

There is no Mapbox and no fallback. Before routing, the car and the destination are checked against the cached
coverage; then the corridor for the drive is downloaded, verified and unpacked, and every route and reroute for that
destination runs locally on it. Any failure is a MapError, reported in TurnPilotMapStatus, and navd has no route.

TurnPilotValhallaTiles is the developer and sim override (BeamNG sets it): when set, its tiles are used as they are,
with no geofence and no download.
"""
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.turnpilot.maps import corridor
from openpilot.selfdrive.turnpilot.maps.client import MapClient, MapError, Throttle
from openpilot.selfdrive.turnpilot.maps.coverage import NO_COVERAGE, CoverageCache, check_trip, refresh
from openpilot.selfdrive.turnpilot.maps.geofence import Region
from openpilot.selfdrive.turnpilot.maps.paths import maps_dir
from openpilot.selfdrive.turnpilot.maps.status import NavStatus
from openpilot.selfdrive.turnpilot.routing.valhalla_source import ValhallaSource

PACKS_DIR = 'corridors'
CONFIG_KEYS = ('TurnPilotMapHost', 'TurnPilotMapKey', 'TurnPilotMapCert')
RETRY_SECONDS = 30.  # between downloads after one that could work on a retry (a network blip)
REFUSED_RETRY_SECONDS = 300.  # after a refusal (422, bad key, bad TLS, bad pack), unless the destination or settings change


def local_override(params) -> str:
  return (params.get('TurnPilotValhallaTiles') or '').strip()


def check_route_source(params):
  """TurnPilotRouteSource is left over from Mapbox routing. 'mapbox' fails loudly, so nobody thinks they're on it."""
  mode = (params.get('TurnPilotRouteSource') or 'auto').strip().lower()
  if mode == 'mapbox':
    raise MapError('mapbox_removed', "Mapbox routing was removed from TurnPilot. Set TurnPilotRouteSource back to auto.",
                   'Mapbox routing was removed')


@dataclass
class Corridor:
  destination: tuple
  pack: corridor.Pack
  source: ValhallaSource
  region: Region


@dataclass
class Failure:
  destination: tuple
  config: tuple
  error: MapError
  at: float


class MapRouter:
  """navd's route source. route() returns the steps or raises a MapError; the status param says what happened."""

  def __init__(self, params, directory: Path | None = None, client_factory=MapClient.from_params, valhalla=ValhallaSource):
    self.params = params
    self.directory = directory
    self.client_factory = client_factory
    self.valhalla = valhalla
    self.status = NavStatus(params)
    self.coverage = CoverageCache(directory)
    self.local: tuple[str, ValhallaSource] | None = None
    self.corridor: Corridor | None = None
    self.failure: Failure | None = None

  @property
  def packs_dir(self) -> Path:
    return (self.directory or maps_dir()) / PACKS_DIR

  def config(self) -> tuple:
    return tuple(self.params.get(k) for k in CONFIG_KEYS)

  def clear(self):
    """No destination."""
    self.failure = None
    self.status.set('idle')

  def route(self, start, destination, bearing, language) -> list[dict]:
    try:
      check_route_source(self.params)
      tiles = local_override(self.params)
      if tiles:
        return self.route_local(tiles, start, destination, bearing, language)
      return self.route_server(start, destination, bearing, language)
    except MapError as e:
      cloudlog.error(f"turnpilot route refused: {e.code}: {e.message}")
      self.status.error(e, destination=list(destination))
      raise

  def route_local(self, tiles, start, destination, bearing, language):
    if self.local is None or self.local[0] != tiles:
      try:
        self.local = (tiles, self.valhalla(tiles))
      except Exception as e:
        raise MapError('local_tiles', f"TurnPilotValhallaTiles ({tiles}) doesn't load: {e}", 'Local tiles do not load') from None
    steps = self.routed(self.local[1], start, destination, bearing, language, 'the local tiles')
    if not steps:
      raise MapError('no_route', f"Valhalla found no route on the local tiles ({tiles}).", 'No route found')
    self.status.set('ready', 'Routing on local tiles (developer override, no geofence).', 'Local tiles', source='local')
    return steps

  def coverage_or_fetch(self):
    """The cached coverage. With none cached yet, one fetch now, since navd may be the first to run online."""
    coverage = self.coverage.get()
    if coverage is None:
      try:
        return refresh(self.params, self.directory)
      except MapError as e:
        raise MapError(NO_COVERAGE.code, f"{NO_COVERAGE.message} Fetching it now failed: {e.message}", NO_COVERAGE.short) from None
    return coverage

  def route_server(self, start, destination, bearing, language):
    current = self.corridor
    if current is not None and current.destination == destination and current.pack.path.is_dir():
      # a reroute: the corridor is wider than the route, and the geofence sits inside the tiles, so try it first
      try:
        steps = current.source.route(start, destination, bearing, language)
      except Exception as e:  # the car left the corridor: Valhalla finds no road near it
        cloudlog.warning(f"turnpilot: routing on the corridor failed ({e}), downloading a new corridor")
        steps = []
      if steps:
        self.ready(current.region)
        return steps
    elif (self.status.last or {}).get('destination') != list(destination):  # a retry keeps showing its error meanwhile
      self.status.set('checking', 'Checking the mapped area', destination=list(destination))

    coverage = self.coverage_or_fetch()
    region = check_trip(coverage, start, destination)
    self.download(start, destination, region, coverage.names())
    assert self.corridor is not None
    steps = self.routed(self.corridor.source, start, destination, bearing, language, 'the downloaded map')
    if not steps:
      raise MapError('no_route', "No route on the downloaded map for this drive.", 'No route found')
    self.ready(region)
    return steps

  def ready(self, region: Region):
    self.status.set('ready', f"Ready: map for this drive in the {region.name}", 'Ready', source='server', region=region.name)

  def routed(self, source, start, destination, bearing, language, what):
    try:
      return source.route(start, destination, bearing, language)
    except Exception as e:
      raise MapError('routing_failed', f"Routing on {what} failed: {e}", 'Routing failed') from None

  def download(self, start, destination, region: Region, names: str):
    config = self.config()
    failure = self.failure
    if failure is not None and failure.destination == destination and failure.config == config:
      wait = RETRY_SECONDS if failure.error.retry else REFUSED_RETRY_SECONDS
      if time.monotonic() - failure.at < wait:
        raise failure.error
    try:
      self.corridor = self.fetch(start, destination, region, names)
      self.failure = None
    except MapError as e:
      self.failure = Failure(destination, config, e, time.monotonic())
      raise

  def fetch(self, start, destination, region: Region, names: str) -> Corridor:
    client = self.client_factory(self.params, self.directory)
    packs_dir = self.packs_dir
    packs_dir.mkdir(parents=True, exist_ok=True)
    download = packs_dir / f".download-{uuid.uuid4().hex[:8]}.tar"
    base = {'destination': list(destination), 'region': region.name, 'source': 'server'}

    def progress(done, total):
      self.status.set('downloading', 'Downloading map for this drive…', 'Downloading map', progress=round(done / total, 3) if total else None,
                      bytes=done, total=total, **base)

    report = Throttle(progress)
    report(0, None, final=True)
    try:
      size = client.corridor(start, destination, download, report, names)
      report(size, size, final=True)
      pack = corridor.unpack(download, packs_dir)
    finally:
      download.unlink(missing_ok=True)
    if pack.manifest['region'] != region.id:
      cloudlog.warning(f"turnpilot: corridor is for region {pack.manifest['region']}, the geofence says {region.id}")
    try:
      source = self.valhalla(str(pack.path))
    except Exception as e:
      raise MapError('routing_failed', f"The downloaded map doesn't load in Valhalla: {e}", 'Downloaded map does not load') from None
    corridor.prune(packs_dir, in_use=pack.path)
    cloudlog.info(f"turnpilot: corridor {pack.path.name} for {destination}, {size / 1e6:.1f} MB, {pack.manifest.get('route_km')} km")
    return Corridor(destination, pack, source, region)
