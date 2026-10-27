"""Where navd gets its route from: Mapbox online, or Valhalla on local tiles."""
from typing import Protocol

import requests

from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.turnpilot.routing.valhalla import ValhallaSource, tiles_usable


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
    if bearing is not None:
      params['bearings'] = f"{(bearing + 360) % 360:.0f},90;"

    coords_str = ';'.join(f'{lon},{lat}' for lon, lat in (start, destination))
    url = self.host + '/directions/v5/mapbox/driving-traffic/' + coords_str
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
      cloudlog.event("API request failed", status_code=resp.status_code, text=resp.text, error=True)
    resp.raise_for_status()

    r = resp.json()
    return r['routes'][0]['legs'][0]['steps'] if len(r['routes']) else []


def select_source(mode, tiles) -> str:
  """'mapbox' or 'valhalla' from the TurnPilotRouteSource and TurnPilotValhallaTiles params."""
  mode = (mode or 'auto').strip().lower()
  if mode in ('mapbox', 'valhalla'):
    return mode
  if mode != 'auto':
    cloudlog.warning(f"unknown TurnPilotRouteSource {mode!r}, using auto")
  return 'valhalla' if tiles_usable(tiles) else 'mapbox'


class RouteSources:
  """The route source the params ask for, with the Valhalla source kept between requests."""

  def __init__(self, mapbox: MapboxSource):
    self.mapbox = mapbox
    self.valhalla = None
    self.tiles = None

  def get(self, params) -> RouteSource:
    tiles = params.get('TurnPilotValhallaTiles')
    if select_source(params.get('TurnPilotRouteSource'), tiles) == 'mapbox':
      return self.mapbox
    if self.valhalla is None or tiles != self.tiles:
      self.valhalla, self.tiles = ValhallaSource(tiles), tiles
    return self.valhalla
