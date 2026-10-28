"""The route's stop signs, yields and signals from Valhalla (routing/controls.py): on a stub Valhalla, and on the sim's
tiles with controls when they're built (tools/turnpilot/build_sim_tiles.sh into ~/turnpilot_data/valhalla/west_coast_usa_signals)."""
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from openpilot.selfdrive.turnpilot.routing.controls import (CONTROL_LINE_M, MAP_MARGIN_M, MARKER, SIDECAR, actor_call, controls_ahead,
                                                            haversine, http_call, route_controls)

M_PER_DEG = 111195.
LAT0, LON0 = 39.77, -86.16


def pt(m):
  return LAT0 + m / M_PER_DEG, LON0


def steps(lengths=(200., 200.), every=10.):
  """Steps due north, back to back, their geojson geometry a point every `every` m."""
  out, start = [], 0.
  for length in lengths:
    n = int(length / every)
    out.append({'geometry': {'coordinates': [[pt(start + k * every)[1], pt(start + k * every)[0]] for k in range(n + 1)]},
                'distance': length})
    start += length
  return out


class StubValhalla:
  """trace_attributes and locate over a straight route: edges [(begin m, end m, fields...)], a shape point every 10 m.
  locate answers, for a point on an edge, that edge's candidate with its stop and yield signs."""
  def __init__(self, edges, total=400., fail_edge_walk=False, unmatched=()):
    self.total, self.fail_edge_walk, self.unmatched = total, fail_edge_walk, set(unmatched)
    self.edges = []
    for i, (b, e, f) in enumerate(edges):
      d = {'id': 1000 + i, 'way_id': f.get('way_id', 50 + i), 'forward': f.get('forward', True), 'begin_shape_index': int(b / 10.),
           'end_shape_index': int(e / 10.), 'traffic_signal': f.get('edge_signal', False), 'road_class': f.get('road_class', 'residential'),
           'internal_intersection': f.get('internal', False), 'length': (e - b) / 1000.,
           'end_node': {'traffic_signal': f.get('node_signal', False),
                        'intersecting_edges': [{'road_class': c} for c in f.get('crossing', ['residential'])]}}
      self.edges.append((d, f.get('stop', False), f.get('yield', False)))
    self.requests = []

  def __call__(self, endpoint, req):
    self.requests.append((endpoint, req))
    if endpoint == 'trace_attributes':
      if self.fail_edge_walk and req['shape_match'] == 'edge_walk':
        raise RuntimeError("no edge_walk path")
      return {'shape': [{'lat': pt(m)[0], 'lon': pt(m)[1]} for m in range(0, int(self.total) + 1, 10)],
              'edges': [d for d, _, _ in self.edges]}
    out = []
    for loc in req['locations']:
      m = (loc['lat'] - LAT0) * M_PER_DEG
      hit = next(((d, s, y) for d, s, y in self.edges if d['begin_shape_index'] * 10. < m < d['end_shape_index'] * 10.), None)
      if hit is None or hit[0]['id'] in self.unmatched:
        out.append({'edges': [{'edge_id': {'value': 1}, 'edge': {'stop_sign': True}}]})
      else:
        out.append({'edges': [{'edge_id': {'value': hit[0]['id']}, 'edge': {'stop_sign': hit[1], 'yield_sign': hit[2]}}]})
    return out


class StubRoad(StubValhalla):
  """StubValhalla, and locate by heading (controls_ahead) on its road due north: road edges [(begin m, end m)]. At a
  point inside an edge, that edge; at a node (radius 1), the edge ending there and the one starting there."""
  def __init__(self, edges, road, **kw):
    super().__init__(edges, **kw)
    self.road = road

  def __call__(self, endpoint, req):
    if endpoint != 'locate' or 'heading' not in req['locations'][0]:
      return super().__call__(endpoint, req)
    self.requests.append((endpoint, req))
    loc = req['locations'][0]
    if abs(loc['heading'] % 360) > 45 and abs(loc['heading'] % 360) < 315:
      return [{'edges': []}]  # nothing heads that way
    m = (loc['lat'] - LAT0) * M_PER_DEG
    out = []
    for b, e in self.road:
      shape = [{'lat': pt(x)[0], 'lon': pt(x)[1]} for x in (b, (b + e) / 2, e)]
      if loc['radius'] == 1 and (abs(m - b) < 0.5 or abs(m - e) < 0.5):
        out.append({'edge_info': {'shape': shape}, 'edge': {'forward': True}, 'percent_along': 0. if abs(m - b) < 0.5 else 1.,
                    'distance': 0., 'heading': 0.})
      elif loc['radius'] > 1 and b <= m < e:
        out.append({'edge_info': {'shape': shape}, 'edge': {'forward': True}, 'percent_along': (m - b) / (e - b), 'distance': 2.,
                    'heading': 0.})
    return [{'edges': out}]


def test_controls_ahead_follow_the_road_from_the_car_edge_after_edge():
  """Off the route (navd: offRouteControls): the road the car is on, from where it is, by its heading, then on from each
  node by the edge carrying on, walked like a route. Its end isn't a destination: a control there counts."""
  v = StubRoad([(0., 100., {'stop': True}), (100., 250., {'node_signal': True}), (250., 400., {'stop': True})],
               road=[(0., 100.), (100., 250.), (250., 400.)])
  controls, status, road = controls_ahead(v, pt(0.), 3.)
  assert status == 'ok'
  assert [(c.kind, round(c.route_m, 1)) for c in controls] == \
    [('stop', 100. - CONTROL_LINE_M - MAP_MARGIN_M), ('signal', 250. - CONTROL_LINE_M - MAP_MARGIN_M),
     ('stop', 400. - CONTROL_LINE_M - MAP_MARGIN_M)]
  assert abs(haversine(road[0], road[-1]) - 400.) < 0.5
  assert all(abs(haversine(c.point, pt(c.route_m))) < 0.5 for c in controls)
  heads = [r['locations'][0] for e, r in v.requests if e == 'locate' and 'heading' in r['locations'][0]]
  assert heads[0]['radius'] > 1 and all(h['radius'] == 1 for h in heads[1:]) and len(heads) == 3


def test_controls_ahead_unknown_is_failed_never_none():
  v = StubRoad([(0., 400., {'node_signal': True})], road=[(0., 400.)])
  assert controls_ahead(v, pt(0.), 180.)[:2] == ([], 'failed')  # no road heading that way
  assert controls_ahead(v, pt(0.), float('nan'))[:2] == ([], 'failed')  # no heading
  assert controls_ahead(lambda *_: 1 / 0, pt(0.), 0.)[:2] == ([], 'failed')


def test_controls_ahead_unsupported_without_the_marker(tmp_path):
  v = StubRoad([(0., 400., {'node_signal': True})], road=[(0., 400.)])
  assert controls_ahead(v, pt(0.), 0., tmp_path)[:2] == ([], 'unsupported')
  assert controls_ahead(v, pt(0.), 0., marked(tmp_path))[1] == 'ok'


def marked(tmp_path, sidecar=None):
  (tmp_path / MARKER).touch()
  if sidecar is not None:
    (tmp_path / SIDECAR).write_text(json.dumps({'controls': sidecar}))
  return tmp_path


def summary(controls):
  return [(c.kind, round(c.route_m, 1), c.step_idx, round(c.along_step_m, 1), c.line_mapped, c.junction, c.cross_suspect) for c in controls]


def test_kinds_at_the_edge_end_less_the_line_offset():
  v = StubValhalla([(0., 100., {'stop': True}), (100., 250., {'node_signal': True}), (250., 330., {'edge_signal': True}),
                    (330., 400., {})])
  controls, status = route_controls(v, steps())
  assert status == 'ok'
  off = CONTROL_LINE_M + MAP_MARGIN_M
  assert summary(controls) == [('stop', 100. - off, 0, 100. - off, False, True, False),
                               ('signal', 250. - off, 1, 50. - off, False, True, False),
                               ('signal', 330. - off, 1, 130. - off, False, True, False)]
  # one batched verbose locate, a metre back from each edge's end
  locates = [r for e, r in v.requests if e == 'locate']
  assert len(locates) == 1 and locates[0]['verbose'] and len(locates[0]['locations']) == 4
  assert abs((locates[0]['locations'][0]['lat'] - LAT0) * M_PER_DEG - 99.) < 0.05


def test_yield_and_the_edge_at_the_destination():
  # the last edge ends at the route's end (the destination): a control there isn't driven to
  v = StubValhalla([(0., 150., {'yield': True}), (150., 400., {'stop': True})])
  controls, _ = route_controls(v, steps())
  assert [c.kind for c in controls] == ['yield']


def test_internal_intersection_skipped():
  v = StubValhalla([(0., 100., {'node_signal': True, 'internal': True}), (100., 200., {'stop': True}), (200., 400., {})])
  assert [c.kind for c in route_controls(v, steps())[0]] == ['stop']


def test_merge_keeps_the_signal():
  v = StubValhalla([(0., 100., {'stop': True}), (100., 110., {'node_signal': True}), (110., 300., {'yield': True}), (300., 400., {})])
  controls, _ = route_controls(v, steps())
  assert [(c.kind, c.id) for c in controls] == [('signal', 0), ('yield', 1)]


def test_cross_suspect_and_junction():
  v = StubValhalla([(0., 100., {'stop': True, 'road_class': 'primary', 'crossing': ['residential', 'service_other']}),
                    (100., 200., {'yield': True, 'road_class': 'residential', 'crossing': ['residential']}),
                    (200., 300., {'stop': True, 'crossing': []}), (300., 400., {})])
  controls, _ = route_controls(v, steps())
  assert [(c.kind, c.junction, c.cross_suspect) for c in controls] == [('stop', True, True), ('yield', True, False), ('stop', False, False)]


def test_walk_or_snap_and_partial_said():
  v = StubValhalla([(0., 100., {'stop': True}), (100., 400., {})], fail_edge_walk=True)
  controls, status = route_controls(v, steps())
  assert status == 'walkOrSnap' and [c.kind for c in controls] == ['stop']
  v = StubValhalla([(0., 100., {'stop': True}), (100., 200., {}), (200., 400., {})], unmatched=(1001,))
  assert route_controls(v, steps())[1] == 'partial'


def test_failures_are_failed_and_unsupported_without_the_marker(tmp_path):
  def broken(endpoint, req):
    raise ConnectionError("no valhalla_service")  # pyvalhalla missing, and no service either: loud, never empty-and-ok
  assert route_controls(broken, steps()) == ([], 'failed')
  assert route_controls(StubValhalla([(0., 400., {})]), [{'geometry': {'coordinates': []}, 'distance': 0.}]) == ([], 'failed')
  assert route_controls(StubValhalla([(0., 100., {'stop': True}), (100., 400., {})]), steps(), tiles_dir=tmp_path) == ([], 'unsupported')
  assert route_controls(StubValhalla([(0., 100., {'stop': True}), (100., 400., {})]), steps(), tiles_dir=marked(tmp_path))[1] == 'ok'


def test_sidecar_line_is_projected(tmp_path):
  line = pt(93.5)
  side = {'77:f': {'kind': 'stop', 'instance': 'stop 7', 'line': list(line)}, '77:b': {'kind': 'stop', 'line': list(pt(20.))}}
  v = StubValhalla([(0., 100., {'stop': True, 'way_id': 77}), (100., 200., {'stop': True, 'way_id': 78}), (200., 400., {})])
  controls, status = route_controls(v, steps(), tiles_dir=marked(tmp_path, side))
  assert status == 'ok'
  assert abs(controls[0].route_m - (93.5 - MAP_MARGIN_M)) < 1. and controls[0].line_mapped and controls[0].instance == 'stop 7'
  assert not controls[1].line_mapped and abs(controls[1].route_m - (200. - CONTROL_LINE_M - MAP_MARGIN_M)) < 0.1


def test_http_equals_actor():
  v = StubValhalla([(0., 100., {'stop': True}), (100., 250., {'node_signal': True}), (250., 400., {})])

  class Resp:
    def __init__(self, body):
      self.body = body

    def raise_for_status(self):
      pass

    def json(self):
      return json.loads(json.dumps(self.body))

  posted = []
  def post(url, json=None, timeout=None):
    posted.append((url, timeout))
    return Resp(v(url.rsplit('/', 1)[1], json))

  class Actor:
    def trace_attributes(self, req):
      return json.dumps(v('trace_attributes', req))  # some pyvalhalla versions answer a JSON string

    def locate(self, req):
      return v('locate', req)
  via_http = route_controls(http_call('http://127.0.0.1:8002', post), steps())
  via_actor = route_controls(actor_call(Actor()), steps())
  assert via_http == via_actor and via_http[0]
  assert posted[0] == ('http://127.0.0.1:8002/trace_attributes', 5.) and posted[1][0].endswith('/locate')


# --- the sim's tiles, with the level's stop signs and signals ---

SIGNALS_TILES = Path('~/turnpilot_data/valhalla/west_coast_usa_signals').expanduser()
SIGNALS_OSM = Path('~/turnpilot_data/osm/west_coast_usa_signals.osm').expanduser()
HAVE_TILES = importlib.util.find_spec('valhalla') is not None and (SIGNALS_TILES / MARKER).exists() and SIGNALS_OSM.exists()


@pytest.fixture(scope='module')
def sim_map():
  from openpilot.selfdrive.turnpilot.routing.valhalla_source import ValhallaSource
  root = ET.parse(SIGNALS_OSM).getroot()
  ll, ctrl = {}, {}
  for el in root.iter('node'):
    ll[el.get('id')] = (float(el.get('lat')), float(el.get('lon')))
    tags = {t.get('k'): t.get('v') for t in el.findall('tag')}
    if 'beamng:signal' in tags:
      ctrl[tags['beamng:signal']] = el.get('id')
  ways = [[nd.get('ref') for nd in w.findall('nd')] for w in root.iter('way')]
  return ValhallaSource(str(SIGNALS_TILES)), ll, ctrl, ways


def through(sim_map, instance):
  """{'along', 'against'}: the controls' instances on a route over the control's way each way, on past its far end."""
  src, ll, ctrl, ways = sim_map
  w = next(w for w in ways if ctrl[instance] in w)
  out = {}
  for label, (a, b) in (('along', (w[0], w[-1])), ('against', (w[-1], w[0]))):
    far = next((x[-1] if x[0] == b else x[0]) for x in ways if x is not w and b in (x[0], x[-1]))
    steps_ = src.route(ll[a][::-1], ll[far][::-1], None, None)
    controls, status = src.controls(steps_, tiles_dir=src.tiles_dir)
    assert status == 'ok'
    out[label] = [c.instance for c in controls if c.line_mapped]
  return out


@pytest.mark.skipif(not HAVE_TILES, reason="no west_coast_usa_signals tiles (tools/turnpilot/build_sim_tiles.sh)")
@pytest.mark.parametrize("instance, along", [('stop 19', True), ('stop 35', False), ('stop 43', True), ('trafficLight 13', False)])
def test_sim_controls_only_on_their_approach(sim_map, instance, along):
  got = through(sim_map, instance)
  assert (instance in got['along']) == along and (instance in got['against']) == (not along), got


@pytest.mark.skipif(not HAVE_TILES, reason="no west_coast_usa_signals tiles (tools/turnpilot/build_sim_tiles.sh)")
def test_sim_controls_ahead_of_the_car_off_the_route(sim_map):
  """BeamNG route_demo, 2026-10-28: rerouting at (-332, 495) heading southwest, the car drove through the signal at
  (-382, 445). The lookup ahead of the car finds it on its mapped line; heading the other way, it isn't ahead."""
  src = sim_map[0]
  controls, status, _ = src.controls_ahead((39.7728582, -86.1619751), 225., tiles_dir=src.tiles_dir)
  assert status == 'ok'
  first = controls[0]
  assert (first.kind, first.line_mapped) == ('signal', True) and 60. < first.route_m < 75., controls
  assert haversine(first.point, (39.7724078, -86.1625586)) < 6.
  controls, status, _ = src.controls_ahead((39.7724078, -86.1625586), 45., tiles_dir=src.tiles_dir)
  assert status == 'ok' and all(c.kind != 'signal' or c.route_m > 200. for c in controls), controls
