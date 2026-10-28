"""The cached coverage (drivable geofence) from /v1/coverage, and the geofence check before routing.

turnpilot_web refreshes the cache while the comma is online; navd, the web endpoint and the UI read it. With no cache
the check fails loudly: TurnPilot never assumes a place is covered.
"""
import json
import threading
from pathlib import Path

from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.turnpilot.maps.client import MapClient, MapError, missing_config, write_atomic
from openpilot.selfdrive.turnpilot.maps.geofence import Coverage, Region
from openpilot.selfdrive.turnpilot.maps.paths import maps_dir, wall_time

COVERAGE_FILE = 'coverage.json'
REFRESH_PERIOD = 6 * 3600.  # s between refreshes once one worked; the server rebuilds monthly
RETRY_PERIOD = 60.  # s between attempts while offline or misconfigured
STATUS_PARAM = 'TurnPilotMapCoverageStatus'

NO_COVERAGE_MESSAGE = ("TurnPilot has never downloaded its mapped area from the map server, so it can't tell where it " +
                       "can drive. Set up the map server in the TurnPilot web app with the comma online.")


def no_coverage(detail: str = '') -> MapError:
  return MapError('no_coverage', NO_COVERAGE_MESSAGE + (f" {detail}" if detail else ''), 'No mapped area downloaded')


def coverage_path(directory: Path | None = None) -> Path:
  return (directory or maps_dir()) / COVERAGE_FILE


def save(data: dict, directory: Path | None = None) -> Coverage:
  coverage = Coverage(data)  # a malformed answer never replaces a good cache
  write_atomic(coverage_path(directory), json.dumps({'fetched_at': wall_time(), 'coverage': data}).encode())
  return coverage


class CoverageCache:
  """The coverage file, parsed again only when it changes."""

  def __init__(self, directory: Path | None = None):
    self.directory = directory
    self.stamp = None
    self.coverage: Coverage | None = None
    self.fetched_at: float | None = None

  def get(self) -> Coverage | None:
    """The cached coverage, None if none was ever fetched. A cache that doesn't parse raises a MapError."""
    path = coverage_path(self.directory)
    try:
      st = path.stat()
    except FileNotFoundError:
      self.stamp, self.coverage, self.fetched_at = None, None, None
      return None
    stamp = (st.st_mtime_ns, st.st_size)
    if stamp != self.stamp:
      try:
        cached = json.loads(path.read_text())
        coverage, fetched_at = Coverage(cached['coverage']), float(cached['fetched_at'])
      except (ValueError, KeyError, TypeError) as e:
        raise MapError('bad_coverage', f"TurnPilot's saved mapped area at {path} is unreadable ({e}). It is downloaded again when the " +
                       "comma is online.", 'Saved mapped area unreadable') from None
      self.stamp, self.coverage, self.fetched_at = stamp, coverage, fetched_at
    return self.coverage

  def require(self) -> Coverage:
    coverage = self.get()
    if coverage is None:
      raise no_coverage()
    return coverage


def check_trip(coverage: Coverage, start, destination) -> Region:
  """The region a drive from start (or None if unknown) to destination is in, both (lon, lat), or a MapError."""
  around = f"TurnPilot can only navigate around {coverage.names()}."
  dest = coverage.region_at(*destination)
  here = coverage.region_at(*start) if start is not None else None
  if dest is None:
    raise MapError('outside_coverage', coverage.outside_message(), 'Destination outside mapped area')
  if start is not None and here is None:
    raise MapError('outside_coverage', f"The car is outside TurnPilot's mapped area. {around}", 'Car outside mapped area')
  if here is not None and here.id != dest.id:
    raise MapError('different_regions', f"The destination is in the {dest.name}, but the car is in the {here.name}. " +
                   "TurnPilot can't route between mapped areas.", 'Destination in another mapped area')
  return dest


def refresh(params, directory: Path | None = None) -> Coverage:
  """Fetch /v1/coverage into the cache, and record how that went in TurnPilotMapCoverageStatus."""
  try:
    data = MapClient.from_params(params, directory).coverage()
    try:
      coverage = save(data, directory)
    except ValueError as e:
      raise MapError('bad_response', f"The map server's coverage is not valid: {e}", 'Map server sent bad coverage') from None
  except MapError as e:
    params.put(STATUS_PARAM, {'ok': False, 'code': e.code, 'error': e.message, 'at': wall_time()})
    raise
  params.put(STATUS_PARAM, {'ok': True, 'version': coverage.version, 'regions': coverage.summary(), 'at': wall_time()})
  return coverage


class CoverageRefresher(threading.Thread):
  """Keeps the cache fresh while the comma is online. wake() retries now, as after the map server settings change."""

  def __init__(self, params, directory: Path | None = None):
    super().__init__(name='coverage', daemon=True)
    self.params, self.directory = params, directory
    self.woken = threading.Event()
    self.stopped = threading.Event()

  def wake(self):
    self.woken.set()

  def stop(self):
    self.stopped.set()
    self.woken.set()

  def run(self):
    last_error = None
    while not self.stopped.is_set():
      wait = REFRESH_PERIOD
      try:
        if missing_config(self.params) and last_error == 'not_configured':
          wait = RETRY_PERIOD  # not set up (a fresh install, the sim): said once, not every minute
        else:
          refresh(self.params, self.directory)
          last_error = None
      except MapError as e:
        if e.code != last_error:
          cloudlog.warning(f"turnpilot coverage refresh failed: {e.code}: {e.message}")
        last_error = e.code
        wait = RETRY_PERIOD
      except Exception:
        cloudlog.exception("turnpilot coverage refresh failed")
        wait = RETRY_PERIOD
      self.woken.wait(wait)
      self.woken.clear()
