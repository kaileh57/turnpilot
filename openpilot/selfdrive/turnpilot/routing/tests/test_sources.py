import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

from openpilot.selfdrive.navigation.navd import RouteEngine
from openpilot.selfdrive.turnpilot.routing import valhalla
from openpilot.selfdrive.turnpilot.routing.sources import MapboxSource, RouteSources, select_source
from openpilot.selfdrive.turnpilot.routing.valhalla import ValhallaSource, locate, tiles_usable

FIXTURES = Path(__file__).parent / 'fixtures'
RESPONSE = json.loads((FIXTURES / 'valhalla_left_then_right.json').read_text())
STEPS = RESPONSE['routes'][0]['legs'][0]['steps']
START, DESTINATION = (-86.1699, 39.77), (-86.16, 39.774)


class FakeActor:
  def __init__(self, config):
    self.config, self.requests = config, []

  def route(self, request):
    self.requests.append(request)
    return RESPONSE


def fake_valhalla():
  return SimpleNamespace(Actor=FakeActor, get_config=lambda **kwargs: kwargs)


def params_of(values):
  params = MagicMock()
  params.get.side_effect = values.get
  return params


class TestSelection(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.tiles = Path(self.directory.name)
    (self.tiles / 'tiles').mkdir()
    (self.tiles / 'tiles.tar').write_bytes(b'')

  def test_select_source(self):
    tiles, missing = str(self.tiles), str(self.tiles / 'missing')
    cases = [
      (None, None, 'mapbox'),  # unset params, as in navd's tests
      ('auto', None, 'mapbox'),
      ('auto', '', 'mapbox'),
      ('auto', missing, 'mapbox'),  # tiles that aren't there aren't usable
      ('auto', tiles, 'valhalla'),
      (None, tiles, 'valhalla'),
      ('auto', 'http://127.0.0.1:8002', 'valhalla'),
      ('mapbox', tiles, 'mapbox'),
      ('Mapbox ', tiles, 'mapbox'),
      ('valhalla', None, 'valhalla'),  # asked for, so it fails loudly instead of quietly using Mapbox
      ('valhalla', tiles, 'valhalla'),
      ('bogus', tiles, 'valhalla'),
      ('bogus', None, 'mapbox'),
    ]
    for mode, tiles_param, expected in cases:
      self.assertEqual(select_source(mode, tiles_param), expected, (mode, tiles_param))

  def test_locate(self):
    root = self.tiles
    self.assertEqual(locate(str(root)), {'extract': str(root / 'tiles.tar'), 'dir': str(root / 'tiles')})
    (root / 'tiles.tar').unlink()
    self.assertEqual(locate(str(root)), {'extract': '', 'dir': str(root / 'tiles')})
    (root / 'tiles').rmdir()
    self.assertIsNone(locate(str(root)))
    (root / 'valhalla.json').write_text('{}')
    self.assertEqual(locate(str(root)), {'config': str(root / 'valhalla.json')})
    self.assertEqual(locate(str(root / 'valhalla.json')), {'config': str(root / 'valhalla.json')})
    (root / 'valhalla.json').unlink()
    (root / '2').mkdir()  # a bare tile hierarchy
    self.assertEqual(locate(str(root)), {'extract': '', 'dir': str(root)})
    (root / 'extract.tar').write_bytes(b'')
    self.assertEqual(locate(str(root / 'extract.tar')), {'extract': str(root / 'extract.tar'), 'dir': ''})
    (root / 'notes.txt').write_text('')
    self.assertIsNone(locate(str(root / 'notes.txt')))
    self.assertEqual(locate('http://127.0.0.1:8002/'), {'url': 'http://127.0.0.1:8002'})
    self.assertFalse(tiles_usable(None) or tiles_usable(''))

  def test_route_sources(self):
    mapbox = MapboxSource('https://example.test', 'token', None)
    sources = RouteSources(mapbox)
    values = {}
    self.assertIs(sources.get(params_of(values)), mapbox)
    values['TurnPilotValhallaTiles'] = str(self.tiles)
    with patch.object(valhalla, 'import_valhalla', fake_valhalla):
      source = sources.get(params_of(values))
      self.assertIsInstance(source, ValhallaSource)
      self.assertEqual(source.actor.config, {'tile_extract': str(self.tiles / 'tiles.tar'), 'tile_dir': str(self.tiles / 'tiles')})
      self.assertIs(sources.get(params_of(values)), source)  # the actor is kept between requests
      values['TurnPilotRouteSource'] = 'mapbox'
      self.assertIs(sources.get(params_of(values)), mapbox)
      values['TurnPilotRouteSource'] = 'auto'
      self.assertIs(sources.get(params_of(values)), source)
      values['TurnPilotValhallaTiles'] = 'http://127.0.0.1:9000'  # new tiles, new source
      self.assertIsNot(sources.get(params_of(values)), source)
      values['TurnPilotRouteSource'] = 'valhalla'
      values['TurnPilotValhallaTiles'] = str(self.tiles / 'missing')
      with self.assertRaises(FileNotFoundError):
        sources.get(params_of(values))


class TestMapbox(unittest.TestCase):
  def response(self, routes):
    response = MagicMock(status_code=200)
    response.json.return_value = {'routes': routes}
    return response

  def test_request(self):
    api = MagicMock()
    api.get_token.return_value = 'device-token'
    source = MapboxSource('https://api.commadotai.com/v1/nav', None, api)
    with patch('requests.get', return_value=self.response([{'legs': [{'steps': STEPS}]}])) as get:
      self.assertEqual(source.route(START, DESTINATION, -10, 'en'), STEPS)
    self.assertEqual(get.call_args.args[0], 'https://api.commadotai.com/v1/nav/directions/v5/mapbox/driving-traffic/-86.1699,39.77;-86.16,39.774')
    options = get.call_args.kwargs['params']
    self.assertEqual((options['access_token'], options['bearings'], options['language'], options['geometries']), ('device-token', '350,90;', 'en', 'geojson'))
    with patch('requests.get', return_value=self.response([])) as get:
      self.assertEqual(MapboxSource('https://api.mapbox.com', 'mapbox-token', None).route(START, DESTINATION, None, None), [])
    self.assertNotIn('bearings', get.call_args.kwargs['params'])
    self.assertEqual(get.call_args.kwargs['params']['access_token'], 'mapbox-token')

  def test_failure(self):
    response = self.response([])
    response.raise_for_status.side_effect = requests.HTTPError('test failure')
    with patch('requests.get', return_value=response), self.assertRaises(requests.HTTPError):
      MapboxSource('https://api.mapbox.com', 'mapbox-token', None).route(START, DESTINATION, None, None)


class TestValhalla(unittest.TestCase):
  def test_request(self):
    with patch.object(valhalla, 'import_valhalla', fake_valhalla), tempfile.TemporaryDirectory() as tiles:
      (Path(tiles) / 'tiles').mkdir()
      source = ValhallaSource(tiles)
      self.assertEqual(source.route(START, DESTINATION, 92.4, 'en'), STEPS)
    request = source.actor.requests[0]
    self.assertEqual(request['locations'], [{'lon': -86.1699, 'lat': 39.77, 'heading': 92, 'heading_tolerance': 90}, {'lon': -86.16, 'lat': 39.774}])
    expected = {'costing': 'auto', 'format': 'osrm', 'shape_format': 'geojson', 'banner_instructions': True, 'voice_instructions': True, 'language': 'en'}
    self.assertEqual({k: request[k] for k in expected}, expected)
    request = source.request(START, DESTINATION, -0.4, None)  # no language without one, and the heading wraps
    self.assertEqual(request['locations'][0]['heading'], 0)
    self.assertNotIn('language', request)
    self.assertNotIn('heading', source.request(START, DESTINATION, None, None)['locations'][0])
    self.assertEqual(source.request(START, DESTINATION, 300.2, None)['locations'][0]['heading'], 300)

  def test_config_file(self):
    with patch.object(valhalla, 'import_valhalla', fake_valhalla), tempfile.TemporaryDirectory() as directory:
      config = Path(directory) / 'valhalla.json'
      config.write_text('{}')
      self.assertEqual(ValhallaSource(str(config)).actor.config, str(config))

  def test_http_fallback(self):
    response = MagicMock(status_code=200)
    response.json.return_value = RESPONSE
    with patch.object(valhalla, 'import_valhalla', return_value=None), patch('requests.post', return_value=response) as post:
      source = ValhallaSource('http://localhost:8002/')
      self.assertEqual(source.route(START, DESTINATION, None, None), STEPS)
      self.assertEqual(post.call_args.args[0], 'http://localhost:8002/route')
      self.assertEqual(post.call_args.kwargs['json']['format'], 'osrm')
      with tempfile.TemporaryDirectory() as tiles:  # tiles but no pyvalhalla: a local valhalla_service
        (Path(tiles) / 'tiles').mkdir()
        source = ValhallaSource(tiles)
        self.assertIsNone(source.actor)
        self.assertEqual(source.route(START, DESTINATION, None, None), STEPS)
        self.assertEqual(post.call_args.args[0], 'http://127.0.0.1:8002/route')
        with patch.dict('os.environ', {'TURNPILOT_VALHALLA_URL': 'http://127.0.0.1:9999'}):
          ValhallaSource(tiles).route(START, DESTINATION, None, None)
          self.assertEqual(post.call_args.args[0], 'http://127.0.0.1:9999/route')
      response.raise_for_status.side_effect = requests.HTTPError('no route')
      with self.assertRaises(requests.HTTPError):
        source.route(START, DESTINATION, None, None)
      response.raise_for_status.side_effect = None
      response.json.return_value = {'routes': []}
      self.assertEqual(source.route(START, DESTINATION, None, None), [])

  def test_device_install_on_path(self):
    marker = object()
    with tempfile.TemporaryDirectory() as device, patch.dict('os.environ', {'TURNPILOT_PYVALHALLA': device}):
      with patch('importlib.import_module', side_effect=[ImportError, marker]), patch.object(sys, 'path', list(sys.path)):
        self.assertIs(valhalla.import_valhalla(), marker)
        self.assertEqual(sys.path[-1], device)
      with patch('importlib.import_module', side_effect=ImportError), patch.object(sys, 'path', list(sys.path)):
        self.assertIsNone(valhalla.import_valhalla())  # an install that doesn't import
    with patch.dict('os.environ', {'TURNPILOT_PYVALHALLA': '/nonexistent/pyvalhalla'}), patch('importlib.import_module', side_effect=ImportError):
      self.assertIsNone(valhalla.import_valhalla())
    with patch('importlib.import_module', return_value=marker), patch.object(sys, 'path', list(sys.path)) as path:
      self.assertIs(valhalla.import_valhalla(), marker)  # a system install wins and the path is left alone
      self.assertEqual(path, sys.path)


class TestRouteEngineSources(unittest.TestCase):
  def engine(self, values):
    pm, sm = MagicMock(), MagicMock()
    with patch('openpilot.selfdrive.navigation.navd.Api'):
      engine = RouteEngine(sm, pm, params=params_of(values))
    engine.last_position, engine.last_bearing = START, 90
    return engine, pm

  def test_source_used_and_guidance_sent(self):
    values = {'LanguageSetting': 'main_en'}
    engine, pm = self.engine(values)
    source = MagicMock()
    source.route.return_value = STEPS
    with patch.object(engine.sources, 'get', return_value=source) as get:
      engine.calculate_route(DESTINATION)
    self.assertIs(get.call_args.args[0], engine.params)
    self.assertEqual(source.route.call_args.args, (START, DESTINATION, 90, 'en'))
    self.assertEqual((engine.step_idx, len(engine.route_geometry)), (0, 4))
    engine.last_position = (-86.1675, 39.77)  # 205 m along the 590 m step
    engine.send_instruction()
    message = pm.send.call_args.args[1]
    instruction = message.navInstruction.to_dict()
    self.assertAlmostEqual(instruction['maneuverDistance'], 385, delta=3)
    self.assertEqual((instruction['maneuverType'], instruction['maneuverModifier']), ('turn', 'left'))
    self.assertEqual(instruction['lanes'][0], {'directions': ['left'], 'active': True, 'activeDirection': 'left'})
    self.assertEqual([m['type'] for m in instruction['allManeuvers']], ['turn', 'new name', 'arrive'])
    self.assertAlmostEqual(instruction['allManeuvers'][0]['distance'], instruction['maneuverDistance'], places=3)

  def test_any_failure_clears_the_route(self):
    engine, _ = self.engine({})
    source = MagicMock()
    source.route.return_value = STEPS
    with patch.object(engine.sources, 'get', return_value=source):
      engine.calculate_route(DESTINATION)
      self.assertIsNotNone(engine.route)
      source.route.side_effect = RuntimeError('No path could be found for input')
      engine.calculate_route(DESTINATION)
    self.assertIsNone(engine.route)
    self.assertIsNone(engine.nav_destination)  # so that it is requested again


if __name__ == '__main__':
  unittest.main()
