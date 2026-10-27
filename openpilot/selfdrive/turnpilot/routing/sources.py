"""Where navd gets its route from: Mapbox online, or Valhalla on local tiles."""
import math
from typing import Protocol

import requests

from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.turnpilot.routing.valhalla_source import ValhallaSource, tiles_usable


class RouteSource(Protocol):
  def route(self, start, destination, bearing, language) -> list[dict]:
    """The steps of the route from start to destination, both (longitude, latitude), in Mapbox's shape with
    geojson geometry. The ego's bearing is in degrees, or None. Empty if there is no route; raises on failure."""


class MapboxSource:
  """Mapbox Directions v5, through the comma API proxy or with MAPBOX_TOKEN."""

  def __init__(self, host, token, api):
    self.host, self.token, self.api = host, token, api

  def route(self, start, destination, bearing, language):
    token = self.token
    if token is None:
      token = self.api.get_token()

    params = {
      'access_token': token,
      'geometries': 'geojson',
      'overview': 'full',
      'steps': 'true',
      'banner_instructions': 'true',
      'alternatives': 'false',
      'language': language,
    }
    if bearing is not None and math.isfinite(bearing):  # a GPS fix without a heading can report NaN
      params['bearings'] = f"{(bearing + 360) % 360:.0f},90;"

    coords_str = ';'.join(f'{lon},{lat}' for lon, lat in (start, destination))
    url = self.host + '/directions/v5/mapbox/driving-traffic/' + coords_str
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
      cloudlog.event("API request failed", status_code=resp.status_code, text=resp.text, error=True)
    resp.raise_for_status()

    r = resp.json()
    return r['routes'][0]['legs'][0]['steps'] if len(r['routes']) else []


def source_mode(mode) -> str:
  """'auto', 'mapbox' or 'valhalla' from the TurnPilotRouteSource param; anything else is auto."""
  mode = (mode or 'auto').strip().lower()
  if mode not in ('auto', 'mapbox', 'valhalla'):
    cloudlog.warning(f"unknown TurnPilotRouteSource {mode!r}, using auto")
    return 'auto'
  return mode


def select_source(mode, tiles) -> str:
  """'mapbox' or 'valhalla' from the TurnPilotRouteSource and TurnPilotValhallaTiles params."""
  mode = source_mode(mode)
  if mode != 'auto':
    return mode
  return 'valhalla' if tiles_usable(tiles) else 'mapbox'


class FallbackSource:
  """In auto mode: Valhalla's route, or Mapbox's when Valhalla fails or finds none, as for a destination outside its tiles."""

  def __init__(self, valhalla: ValhallaSource, mapbox: MapboxSource):
    self.valhalla, self.mapbox = valhalla, mapbox

  def route(self, start, destination, bearing, language):
    try:
      steps = self.valhalla.route(start, destination, bearing, language)
      if len(steps):
        return steps
      cloudlog.warning("valhalla found no route, trying mapbox")
    except Exception:
      cloudlog.exception("valhalla failed to route, trying mapbox")
    return self.mapbox.route(start, destination, bearing, language)


class RouteSources:
  """The route source the params ask for, with the Valhalla source kept between requests.
  An explicit 'valhalla' is strict: it fails where Valhalla can't route, so a sim never quietly drives Mapbox's real roads."""

  def __init__(self, mapbox: MapboxSource):
    self.mapbox = mapbox
    self.valhalla = None
    self.fallback = None
    self.tiles = None

  def get(self, params) -> RouteSource:
    tiles = params.get('TurnPilotValhallaTiles')
    mode = source_mode(params.get('TurnPilotRouteSource'))
    if select_source(mode, tiles) == 'mapbox':
      return self.mapbox
    if self.valhalla is None or tiles != self.tiles:
      try:
        self.valhalla, self.tiles = ValhallaSource(tiles), tiles
        self.fallback = FallbackSource(self.valhalla, self.mapbox)
      except Exception:
        if mode == 'valhalla':
          raise
        cloudlog.exception("valhalla tiles do not load, using mapbox")
        return self.mapbox
    return self.valhalla if mode == 'valhalla' else self.fallback
