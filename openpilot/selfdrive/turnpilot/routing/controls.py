"""The route's stop signs, yields and traffic signals, from Valhalla (navInstruction.controls, map_controls.py).

route_controls() takes navd's route steps and asks Valhalla two things. trace_attributes (edge_walk over the route's own
shape) gives the directed edges the route drives, with each edge's traffic_signal, its end node's traffic_signal and the
roads meeting there. Locate gives what trace_attributes doesn't: a directed edge's stop_sign and yield_sign. One batched
verbose locate, radius 1, a metre back from each edge's end, picks the candidate with the edge's own id. Valhalla puts a
control on the edge that ends at the junction, so every control is placed at the END of its edge:
  signal = the edge's traffic_signal or its end node's; stop = locate's stop_sign; yield = locate's yield_sign.
Internal intersection edges (inside a divided road's junction) are skipped.
junction: other roads meet at the end node. crossSuspect (stop and yield only): the route's road outranks every road
crossing there, so the sign is likely the cross road's (real OSM: undirected stops are flagged on every approach).
The stop line: a sim sidecar (tiles_dir/controls.json, {"controls": {"<way_id>:<f|b>": {kind, line: [lat, lon], ...}}} from
tools/sim/bridge/beamng/mapsignals.py) gives it for its controls, projected onto the route within the edge (lineMapped);
otherwise it is the edge's end less CONTROL_LINE_M. Either way less MAP_MARGIN_M. Controls within MERGE_M merge, the
signal over the stop over the yield.
Status: ok; walkOrSnap (edge_walk failed, walk_or_snap matched it, logged); partial (more than PARTIAL_UNMATCHED of the
edges unmatched by locate); failed (nothing: both matches failed, or any exception); unsupported (tiles_dir given and it
has no controls_built marker: tiles built from data with no controls, the old sim exports). The route publishes whatever
the status: navd sends it on every message, and turnpilotd says so out loud.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

from openpilot.common.swaglog import cloudlog

CONTROL_LINE_M = 10.    # m before the junction node a stop line is taken to be, without a mapped one
MAP_MARGIN_M = 1.       # m short of the line the car stops
MERGE_M = 15.           # controls this close are one
PARTIAL_UNMATCHED = 0.10
LOCATE_BACK_M = 1.      # m back from an edge's end the locate point goes, so it is on the edge, not the junction node
END_OF_ROUTE_M = 1.     # an edge ending this close to the route's end ends at the destination: its control isn't driven to
SIDECAR = 'controls.json'
MARKER = 'controls_built'
KIND_RANK = {'signal': 0, 'stop': 1, 'yield': 2}
ROAD_CLASS_RANK = {'motorway': 0, 'trunk': 1, 'primary': 2, 'secondary': 3, 'tertiary': 4, 'unclassified': 5, 'residential': 6,
                   'service_other': 7}
HTTP_TIMEOUT_S = 5.
EARTH_R = 6371000.

ATTRIBUTES = ['edge.id', 'edge.way_id', 'edge.forward', 'edge.length', 'edge.begin_shape_index', 'edge.end_shape_index',
              'edge.traffic_signal', 'edge.internal_intersection', 'edge.road_class', 'node.traffic_signal', 'node.type',
              'node.intersecting_edge.road_class', 'shape']


@dataclass
class Control:
  id: int              # index along the route
  kind: str            # 'stop', 'yield' or 'signal'
  route_m: float       # m along the route (navd's step frame: the steps' distances, then along the step) to the stop point
  step_idx: int
  along_step_m: float
  line_mapped: bool = False
  junction: bool = True
  cross_suspect: bool = False
  instance: str = ''   # the sim's signal instance its line came from (the sidecar), for logs and tests


def decode6(s: str) -> list[tuple[float, float]]:
  """Valhalla's encoded polyline (precision 6) as [(lat, lon), ...]."""
  out, i, lat, lon = [], 0, 0, 0
  while i < len(s):
    vals = []
    for _ in range(2):
      shift = res = 0
      while True:
        b = ord(s[i]) - 63
        i += 1
        res |= (b & 0x1f) << shift
        shift += 5
        if b < 0x20:
          break
      vals.append(~(res >> 1) if res & 1 else res >> 1)
    lat, lon = lat + vals[0], lon + vals[1]
    out.append((lat / 1e6, lon / 1e6))
  return out


def haversine(a, b) -> float:
  (la1, lo1), (la2, lo2) = a, b
  x = math.radians(lo2 - lo1) * math.cos(math.radians((la1 + la2) / 2))
  return EARTH_R * math.hypot(x, math.radians(la2 - la1))


def cumulative(pts) -> list[float]:
  cum = [0.]
  for a, b in zip(pts, pts[1:], strict=False):
    cum.append(cum[-1] + haversine(a, b))
  return cum


def point_back(pts, cum, idx: int, back_m: float):
  """The point back_m along pts before index idx."""
  target = cum[idx] - back_m
  j = idx
  while j > 0 and cum[j - 1] > target:
    j -= 1
  if j == 0:
    return pts[0]
  a, b = pts[j - 1], pts[j]
  seg = cum[j] - cum[j - 1]
  f = 0. if seg <= 0. else (target - cum[j - 1]) / seg
  return a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])


def project(pts, cum, p, lo_m: float, hi_m: float) -> float | None:
  """m along pts of the closest spot to p between lo_m and hi_m along, or None."""
  best = None
  for j in range(len(pts) - 1):
    if cum[j + 1] < lo_m or cum[j] > hi_m:
      continue
    a, b = pts[j], pts[j + 1]
    kx = math.cos(math.radians(a[0]))
    ax, ay, bx, by, px, py = a[1] * kx, a[0], b[1] * kx, b[0], p[1] * kx, p[0]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0. if L2 <= 0. else min(max(((px - ax) * dx + (py - ay) * dy) / L2, 0.), 1.)
    q = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    d = haversine(q, p)
    if best is None or d < best[0]:
      best = (d, cum[j] + t * (cum[j + 1] - cum[j]))
  return None if best is None else best[1]


def route_shape(steps) -> list[tuple[float, float]]:
  """The steps' geojson geometry as one [(lat, lon), ...], consecutive duplicates dropped."""
  pts = []
  for step in steps:
    for lon, lat in (step.get('geometry') or {}).get('coordinates') or []:
      if not pts or pts[-1] != (lat, lon):
        pts.append((lat, lon))
  return pts


def step_frame(steps, pts, cum):
  """[(start along the shape, start in navd's frame)] per step: navd measures a step by its distance, then along its
  geometry, so a control's route distance is the steps before it plus its distance along its own step's geometry."""
  starts, along, frame, k = [], 0., 0., 0
  for step in steps:
    coords = (step.get('geometry') or {}).get('coordinates') or []
    starts.append((along, frame))
    if coords:
      first = (coords[0][1], coords[0][0])
      while k < len(pts) - 1 and pts[k] != first:
        k += 1
      along = cum[k]
      starts[-1] = (along, frame)
      last = (coords[-1][1], coords[-1][0])
      j = k
      while j < len(pts) - 1 and pts[j] != last:
        j += 1
      along, k = cum[j], j
    frame += float(step.get('distance', 0.))
  return starts


def to_step(starts, s: float) -> tuple[int, float, float]:
  """(step index, m along it, m in navd's route frame) of shape distance s."""
  idx = 0
  for i, (start, _) in enumerate(starts):
    if start <= s + 1e-6:
      idx = i
  start, frame = starts[idx]
  return idx, s - start, frame + s - start


def load_sidecar(tiles_dir) -> dict:
  if tiles_dir is None:
    return {}
  path = Path(tiles_dir) / SIDECAR
  if not path.is_file():
    return {}
  return json.loads(path.read_text()).get('controls', {})


def http_call(url: str, post):
  """call() for a valhalla_service at url; post is requests.post."""
  def call(endpoint, req):
    resp = post(f"{url}/{endpoint}", json=req, timeout=HTTP_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()
  return call


def actor_call(actor):
  """call() for a pyvalhalla Actor (its methods take and return JSON as str or dict, by version)."""
  def call(endpoint, req):
    out = getattr(actor, endpoint)(req)
    return json.loads(out) if isinstance(out, str) else out
  return call


def trace(call, pts):
  """trace_attributes over the shape: (response, status), edge_walk first, then walk_or_snap."""
  req = {'shape': [{'lat': la, 'lon': lo} for la, lo in pts], 'costing': 'auto', 'shape_match': 'edge_walk',
         'filters': {'attributes': ATTRIBUTES, 'action': 'include'}}
  try:
    t = call('trace_attributes', req)
    if t.get('edges'):
      return t, 'ok'
  except Exception as e:  # seen on 1 of 81 Valhalla-made routes: surfaced in the status, never hidden
    cloudlog.warning(f"turnpilot controls: edge_walk failed ({type(e).__name__}: {e}), trying walk_or_snap")
  req['shape_match'] = 'walk_or_snap'
  t = call('trace_attributes', req)
  if not t.get('edges'):
    raise ValueError("trace_attributes matched no edges")
  cloudlog.warning("turnpilot controls: the route matched only by walk_or_snap")
  return t, 'walkOrSnap'


def route_controls(call, steps, tiles_dir=None) -> tuple[list[Control], str]:
  """(controls along the route nearest first, status). call(endpoint, request) -> response: actor_call or http_call.
  tiles_dir: the local tiles' directory, for the controls_built marker and the sim sidecar; None for tiles built from
  real OSM, which carry their own controls."""
  try:
    if tiles_dir is not None and not (Path(tiles_dir) / MARKER).exists():
      return [], 'unsupported'
    return _route_controls(call, steps, tiles_dir)
  except Exception:
    cloudlog.exception("turnpilot controls: failed")
    return [], 'failed'


def _route_controls(call, steps, tiles_dir) -> tuple[list[Control], str]:
  pts = route_shape(steps)
  if len(pts) < 2:
    return [], 'failed'
  cum = cumulative(pts)
  starts = step_frame(steps, pts, cum)
  t, status = trace(call, pts)
  tpts = decode6(t['shape']) if isinstance(t.get('shape'), str) else [(p['lat'], p['lon']) for p in t.get('shape', [])]
  tcum = cumulative(tpts)
  scale = cum[-1] / tcum[-1] if tcum and tcum[-1] > 0. else 1.
  edges = [e for e in t['edges'] if not e.get('internal_intersection')]
  locs = [dict(zip(('lat', 'lon'), point_back(tpts, tcum, e['end_shape_index'], LOCATE_BACK_M), strict=True)) | {'radius': 1}
          for e in edges]
  located = call('locate', {'locations': locs, 'costing': 'auto', 'verbose': True}) if locs else []
  sidecar = load_sidecar(tiles_dir)
  found, unmatched = [], 0
  for e, loc in zip(edges, located, strict=False):
    candidate = next((c.get('edge') or {} for c in (loc or {}).get('edges') or []
                      if (c.get('edge_id') or {}).get('value') == e['id']), None)
    if candidate is None:
      unmatched += 1
      candidate = {}
    node = e.get('end_node') or {}
    kinds = []
    if e.get('traffic_signal') or node.get('traffic_signal'):
      kinds.append('signal')
    if candidate.get('stop_sign'):
      kinds.append('stop')
    if candidate.get('yield_sign'):
      kinds.append('yield')
    if not kinds:
      continue
    end_i, begin_i = e['end_shape_index'], e['begin_shape_index']
    span = (tcum[end_i] - tcum[begin_i]) * scale
    # where the edge ends on the route's own shape, near where the trace says
    expect = tcum[end_i] * scale
    end_m = project(pts, cum, tpts[end_i], expect - 30., expect + 30.)
    end_m = expect if end_m is None else end_m
    if end_m > cum[-1] - END_OF_ROUTE_M:
      continue  # the route ends on this edge (a destination mid-edge snaps onto either direction): no junction ahead
    crossing = [ROAD_CLASS_RANK.get(x.get('road_class'), 8) for x in node.get('intersecting_edges') or []]
    junction = bool(crossing)
    ours = ROAD_CLASS_RANK.get(e.get('road_class'), 8)
    cross_suspect = junction and all(ours < c for c in crossing)
    entry = sidecar.get(f"{e.get('way_id')}:{'f' if e.get('forward') else 'b'}")
    for kind in kinds:
      line_m, mapped, instance = end_m - CONTROL_LINE_M, False, ''
      if entry is not None and entry.get('kind') == kind and entry.get('line'):
        p = project(pts, cum, tuple(entry['line']), end_m - span - 1., end_m + 1.)
        if p is not None:
          line_m, mapped, instance = p, True, entry.get('instance', '')
      found.append((line_m - MAP_MARGIN_M, kind, mapped, junction, cross_suspect and kind in ('stop', 'yield'), instance))
  if edges and unmatched > PARTIAL_UNMATCHED * len(edges) and status == 'ok':
    status = 'partial'
  merged = []
  for c in sorted(found, key=lambda c: c[0]):
    if merged and c[0] - merged[-1][0] < MERGE_M:
      if KIND_RANK[c[1]] < KIND_RANK[merged[-1][1]]:
        merged[-1] = c
      continue
    merged.append(c)
  controls = []
  for i, (s, kind, mapped, junction, suspect, instance) in enumerate(merged):
    step_idx, along, route_m = to_step(starts, max(s, 0.))
    controls.append(Control(i, kind, route_m, step_idx, along, mapped, junction, suspect, instance))
  cloudlog.info(f"turnpilot controls: {len(controls)} on the route ({status}), {unmatched}/{len(edges)} edges unmatched")
  return controls, status
