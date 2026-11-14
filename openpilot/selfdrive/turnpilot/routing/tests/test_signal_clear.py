"""A signal's clear road straight on (routing/controls.py: signal_clear_m, MapControl.signalClearM), on a graph of nodes and
two-way roads answering verbose locate as Valhalla does, with the sidecar's signal layer (osm_sidecar.signal_layer) in a
shard beside it."""
import json
import math

import pytest

from openpilot.selfdrive.turnpilot.maps.corridor import level2_index
from openpilot.selfdrive.turnpilot.routing import controls as controls_module
from openpilot.selfdrive.turnpilot.routing.controls import (CLEAR_GOVERN_M, CLEAR_LOOK_M, CLEAR_OWN_M, CONTROL_LINE_M, MAP_MARGIN_M,
                                                            signal_clear_m)

LAT0, LON0 = 39.77, -86.16
M_PER_DEG = 111195.
STAMP = '2026-10-26T21:24:46Z'
FROM = CONTROL_LINE_M + MAP_MARGIN_M  # the stop point short of its junction's node, no line mapped
ROAD = 50                             # the approach road's way id, south to north in its node order


def ll(x: float, y: float) -> tuple[float, float]:
  """(lat, lon) of a point x m east and y m north of the origin."""
  return LAT0 + y / M_PER_DEG, LON0 + x / (M_PER_DEG * math.cos(math.radians(LAT0)))


class Graph:
  """Nodes {name: (x, y)} and two-way ways [(a, b, {way, internal})] (way: its id, its node order a to b). locate at a node:
  every edge leaving it (percent_along 0) and arriving at it (1), its heading there, its shape in its own direction."""
  def __init__(self, nodes, ways, holes=()):
    self.nodes, self.edges, self.holes, self.calls = nodes, [], set(holes), 0
    for a, b, f in ways:
      self.edges.append((a, b, f.get('internal', False), f.get('way', ROAD), True))
      self.edges.append((b, a, f.get('internal', False), f.get('way', ROAD), False))

  def __call__(self, endpoint, req):
    assert endpoint == 'locate'
    self.calls += 1
    loc = req['locations'][0]
    name = min(self.nodes, key=lambda n: math.dist(ll(*self.nodes[n]), (loc['lat'], loc['lon'])))
    if name in self.holes or math.dist(ll(*self.nodes[name]), (loc['lat'], loc['lon'])) > 1e-6:
      return [{'edges': [], 'nodes': []}]
    out = []
    for a, b, internal, way, forward in self.edges:
      if name not in (a, b):
        continue
      (xa, ya), (xb, yb) = self.nodes[a], self.nodes[b]
      hdg = math.degrees(math.atan2(xb - xa, yb - ya)) % 360.
      shape = [{'lat': p[0], 'lon': p[1]} for p in ((ll(xa, ya), ll(xb, yb)) if forward else (ll(xb, yb), ll(xa, ya)))]
      out.append({'percent_along': 0. if a == name else 1., 'heading': hdg, 'distance': 0.,
                  'edge': {'forward': forward, 'traffic_signal': False, 'classification': {'use': 'road', 'internal': internal}},
                  'edge_info': {'shape': shape, 'way_id': way}})
    lat, lon = ll(*self.nodes[name])
    return [{'edges': out, 'nodes': [{'lat': lat, 'lon': lon, 'traffic_signal': False}]}]


def sig(y, kind='signal', x=0., ways=(ROAD,), direction=''):
  """A signal-layer node y m north on the road (x m east of it)."""
  lat, lon = ll(x, y)
  return {'node': int(1000 + y), 'kind': kind, 'lat': lat, 'lon': lon, 'ways': list(ways), 'direction': direction}


def layer_dir(path, signals=(), stamp=STAMP, key=True):
  """A sidecar directory: the tiles' stamp and the shard of the test's level-2 tile, with the signal layer (key) or not."""
  path.mkdir(parents=True, exist_ok=True)
  (path / 'meta.json').write_text(json.dumps({'id': 'test', 'osm_timestamp': STAMP}))
  (path / 'osm_controls').mkdir(exist_ok=True)
  cell = level2_index(LON0, LAT0)
  shard = {'version': 1, 'osm_timestamp': stamp, 'tile': cell, 'controls': {}, 'junctions': {}, 'rail': [], 'unplaced': []}
  if key:
    shard['signals'] = list(signals)
  (path / 'osm_controls' / f'{cell}.json').write_text(json.dumps(shard))
  return path


def road(*ys, junctions=(), end=400.):
  """The approach road north through J (0 m, a signal's junction with a cross road) and nodes at ys, to an end node; a cross
  road at each of junctions too."""
  names = {'S': -100., 'J': 0., **{f'n{y:g}': y for y in ys}, **{f'x{y:g}': y for y in junctions}, 'E': end}
  order = sorted(names, key=names.get)
  nodes = {n: (0., names[n]) for n in order}
  ways = [(a, b, {}) for a, b in zip(order, order[1:], strict=False)]
  for i, n in enumerate(['J'] + [f'x{y:g}' for y in junctions]):
    nodes[f'{n}w'], nodes[f'{n}e'] = (-40., names[n]), (40., names[n])
    ways += [(f'{n}w', n, {'way': 70 + i}), (n, f'{n}e', {'way': 70 + i})]
  return nodes, ways


def clear(tmp_path, nodes, ways, signals=(), node='J', junction=True, from_m=FROM, holes=(), **kw):
  side = layer_dir(tmp_path / 'side', signals, **kw)
  return signal_clear_m(Graph(nodes, ways, holes), ll(*nodes[node]), 0., from_m, junction, side)


@pytest.mark.parametrize('next_m', [30., 50., 60., 80., 150.])
def test_the_next_junction_straight_on_is_where_another_signal_could_be(tmp_path, next_m):
  nodes, ways = road(junctions=(next_m,))
  assert clear(tmp_path, nodes, ways) == pytest.approx(FROM + next_m, abs=0.5)
  # a node no other road meets is no place, and with nothing for CLEAR_LOOK_M it is that far at least
  nodes, ways = road(next_m)
  assert clear(tmp_path, nodes, ways) == CLEAR_LOOK_M


def test_a_mid_block_signalled_crossing_is_a_place_though_the_tiles_have_nothing_there(tmp_path):
  nodes, ways = road(45., junctions=(150.,))
  assert clear(tmp_path, nodes, ways, [sig(45., 'crossing')]) == pytest.approx(FROM + 45., abs=0.5)
  # on no drivable way, mapped beside the road: still there; and a signal node at its own place
  assert clear(tmp_path, nodes, ways, [sig(45., 'crossing', x=8., ways=())]) == pytest.approx(FROM + 45., abs=0.5)
  assert clear(tmp_path, nodes, ways, [sig(80., x=2.)]) == pytest.approx(FROM + 80., abs=0.5)
  # without the layer the graph alone says 150 m: so the layer must be there, or nothing is known
  assert clear(tmp_path, nodes, ways, [sig(45., 'crossing')], key=False) == 0.


def test_the_junctions_own_heads_and_crosswalks_and_the_oncoming_stop_line_are_no_place(tmp_path):
  nodes, ways = road(12., 25., junctions=(150.,))
  own = [sig(0.), sig(12., direction='b'), sig(25., 'crossing'), sig(20., 'signal', x=10., ways=())]
  assert clear(tmp_path, nodes, ways, own) == pytest.approx(FROM + 150., abs=0.5)
  # an undirected signal on the road past the node, or one facing our way, is another signal's heads
  assert clear(tmp_path, nodes, ways, [sig(12.)]) == pytest.approx(FROM + 12., abs=0.5)
  assert clear(tmp_path, nodes, ways, [sig(12., direction='f')]) == pytest.approx(FROM + 12., abs=0.5)
  # a crossing past the junction's own reach is a mid-block one
  nodes, ways = road(CLEAR_OWN_M + 5., junctions=(150.,))
  assert clear(tmp_path, nodes, ways, [sig(CLEAR_OWN_M + 5., 'crossing')]) == pytest.approx(FROM + CLEAR_OWN_M + 5., abs=0.5)


def test_a_divided_roads_far_carriageway_is_the_junctions_only_where_valhalla_flags_it_internal(tmp_path):
  nodes, ways = road(junctions=(20., 150.))
  on = [sig(20.)]  # the far carriageway's own signal node
  assert clear(tmp_path, nodes, ways, on) == pytest.approx(FROM + 20., abs=0.5)  # not flagged: the next place, gate shut
  ways = [(a, b, {'internal': True}) if (a, b) == ('J', 'x20') else (a, b, f) for a, b, f in ways]
  assert clear(tmp_path, nodes, ways, on) == pytest.approx(FROM + 150., abs=0.5)


def test_a_fork_takes_its_nearest_branch_and_a_road_that_does_not_go_on_is_unknown(tmp_path):
  nodes, ways = road(40., end=60.)
  nodes.update({'L': (-20., 140.), 'R': (20., 140.)})
  ways += [('E', 'L', {'way': 60}), ('E', 'R', {'way': 61})]
  assert clear(tmp_path, nodes, ways) == pytest.approx(FROM + 60., abs=0.5)  # the fork's node
  nodes, ways = road(40., end=60.)
  nodes['T'] = (60., 80.)  # bends 72 deg right
  assert clear(tmp_path, nodes, ways + [('E', 'T', {'way': 62})]) == 0.
  nodes, ways = road(end=60.)  # the road ends
  assert clear(tmp_path, nodes, ways) == 0.


def test_a_tile_edge_a_stale_or_missing_layer_is_unknown(tmp_path):
  nodes, ways = road(40., 80., junctions=(150.,))
  assert clear(tmp_path, nodes, ways, holes={'n80'}) == 0.
  assert clear(tmp_path / 'a', nodes, ways, stamp='2026-01-01T00:00:00Z') == 0.
  assert clear(tmp_path / 'b', nodes, ways, key=False) == 0.
  assert signal_clear_m(Graph(nodes, ways), ll(*nodes['J']), 0., FROM, True, None) == 0.
  # an edge walked with no way id: the layer's nodes on it can't be matched, so nothing is known
  ways = [(a, b, {'way': 0}) if (a, b) == ('n40', 'n80') else (a, b, f) for a, b, f in ways]
  assert clear(tmp_path / 'c', nodes, ways, [sig(60.)]) == 0.


def test_a_signal_on_its_approachs_own_node_governs_the_first_road_junction_on(tmp_path):
  nodes, ways = road(-10., junctions=(120.,))
  assert clear(tmp_path, nodes, ways, [sig(-10.)], node='n-10', junction=False, from_m=5.) == pytest.approx(135., abs=0.5)
  # no road junction within CLEAR_GOVERN_M: a mid-block signal, its own node its junction
  nodes = {'S': (0., -100.), 'L': (0., -10.), 'A': (0., CLEAR_GOVERN_M + 20.), 'N': (0., 100.), 'Nw': (-40., 100.),
           'Ne': (40., 100.), 'E': (0., 400.)}
  ways = [(a, b, {}) for a, b in (('S', 'L'), ('L', 'A'), ('A', 'N'), ('N', 'E'))] + [('Nw', 'N', {'way': 70}), ('N', 'Ne', {'way': 70})]
  nodes['n-10'] = nodes.pop('L')
  ways = [(a.replace('L', 'n-10'), b.replace('L', 'n-10'), f) for a, b, f in ways]
  assert clear(tmp_path, nodes, ways, [sig(-10.)], node='n-10', junction=False, from_m=5.) == pytest.approx(115., abs=0.5)


def test_the_walk_costs_a_locate_a_node_once_a_route(tmp_path):
  nodes, ways = road(40., 80., junctions=(150.,))
  g, memo = Graph(nodes, ways), {}
  side = layer_dir(tmp_path, [])
  first = signal_clear_m(g, ll(*nodes['J']), 0., FROM, True, side, memo)
  n = g.calls
  assert signal_clear_m(g, ll(*nodes['J']), 0., FROM, True, side, memo) == first and g.calls == n
  assert n <= 5  # J, n40, n80, x150 and no more


def test_navd_publishes_it_for_a_route_signal_and_nothing_else(tmp_path):
  from unittest.mock import patch
  from openpilot.cereal import messaging
  from openpilot.selfdrive.turnpilot.control_tracker import from_msg
  from openpilot.selfdrive.turnpilot.routing.guidance import control_msg
  from openpilot.selfdrive.turnpilot.routing.tests.test_route_controls import StubValhalla, osm_dir, steps
  v = StubValhalla([(0., 100., {'stop': True}), (100., 250., {'node_signal': True}), (250., 400., {})])
  side = osm_dir(tmp_path / 'side')
  # a sidecar before the layer: no walk at all
  with patch.object(controls_module, 'signal_clear_m', lambda *a: 42.):
    controls, _ = controls_module.route_controls(v, steps(), sidecar_dir=side)
  assert [(c.kind, c.signal_clear_m) for c in controls] == [('stop', 0.), ('signal', 0.)]
  layer_dir(side, [])
  with patch.object(controls_module, 'signal_clear_m', lambda *a: 42.):
    controls, _ = controls_module.route_controls(v, steps(), sidecar_dir=side)
  assert [(c.kind, c.signal_clear_m) for c in controls] == [('stop', 0.), ('signal', 42.)]
  msg = messaging.new_message('navInstruction')
  msg.navInstruction.controls = [control_msg(c, 10., c.id) for c in controls]
  read = msg.as_reader().navInstruction.controls
  assert [c.signalClearM for c in read] == [0., 42.]
  assert [from_msg(1, c).clear_m for c in read] == [0., 42.]
