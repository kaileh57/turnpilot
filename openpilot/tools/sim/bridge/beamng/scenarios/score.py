#!/usr/bin/env python3
"""Scenario scoring from recorded rows, one row per 50 ms of a run (run.py records them from msgq and the game).

A row is a dict:
  t, wall            s since the run started, and wall-clock time
  x, y, heading, v   the ego's camera mount in BeamNG world meters, heading rad CCW from east, wheel speed m/s
  enabled            selfdriveState.enabled; alert, alert_text: its alertType and alertText1
  steer, gas, brake  carState steeringPressed, gasPressed, brakePressed
  state, reason      turnpilotPlan.state and .reason; lc, prompt: its laneChange and signalPrompt
  control_prompt     turnpilotPlan.controlPrompt: the go TurnPilot asks the driver for ('none', 'stopSignGo', ...)
  lc_state           modelV2.meta.laneChangeState
  nav_valid, nav_type, nav_mod, nav_dist, nav_remaining   navInstruction; route: an id of the current navRoute
  dest_dist          m from the ego to the destination (None without one)
  ped_gap, car_gap   m from the camera mount to the nearest mannequin / scenario car (None when there is none)
  pacer_dx           m the pacer car is ahead of the ego along its heading (None without one)
  blinker            the driver's stalk: '', 'left' or 'right'

Rescore a saved result: python -m openpilot.tools.sim.bridge.beamng.scenarios.score result.json
Score a recorded drive: python -m openpilot.tools.sim.bridge.beamng.scenarios.score --scenario route_demo rlog...
"""
import argparse
import json

import numpy as np

from openpilot.selfdrive.turnpilot.lane_planner import LC_BLOCK_X_M
from openpilot.selfdrive.turnpilot.route_follower import KEEP, TURN, classify

STOPPED_V = 0.3        # m/s: standing
RESUMED_V = 2.0        # m/s: moving again after a hold
RESUME_WITHIN_S = 15.  # a hold counts as resumed if the car is moving again this soon after it ends
HOLD_MERGE_S = 1.0     # holds with the same reason this close together are one
ATTEMPT_M = 30.        # reaching this close to a maneuver on the route is an attempt
STEP_CHANGE_M = 2.     # the distance left after the current maneuver changing by more than this is a new step
REROUTE_WINDOW_S = 1.5 # a new navRoute this close to a step change means navd rerouted (it runs at 1 Hz)
CONFIRM_M = 20.        # after navd moves on, the car stays on the route this far along the next step: the step was taken
TURN_TAKEN_DEG = 45.   # a turn is taken only if the car's heading turned this far its way (the junctions are ~90 degrees)
ARRIVE_M = 15.         # route_follower.ARRIVE_DIST_M, measured to the destination in the game
CAUSE_WINDOW_S = 5.    # a disengagement's cause is the last alert this soon before it
COLLISION_GAP_M = 2.5  # camera mount to object: about where the bumper touches (the mount sits ~2 m behind it)
PED_STOP_GAP = (3., 8.)
CAR_MIN_GAP_M = 4.     # camera mount to the other car's centre
# the camera mount (the car's centreline) past this far left of the road's divider (negative: short of it): the car is in the
# oncoming lane. Half the car's width (0.95 m) less the road model's error (clean runs keep the centre 1.45-1.65 m short of
# it, p99): straddling the divider with the left wheels over it counts, not only a car wholly across
WRONG_SIDE_PAST_M = -0.5
WRONG_SIDE_MIN_M = 3.  # driving this far in the oncoming lane, outside every junction box, fails the run
ON_ROAD_M = 1.5        # within the road's half width plus this of a link's centreline: on that road
ALIGNED_DEG = 35.      # the car's heading within this of a link's direction (either way): driving along it


def msg_fields(ss, cs, tp, nav, model) -> dict:
  """The message part of a row from selfdriveState, carState, turnpilotPlan (None before the first), navInstruction (None
  while navd has no valid route) and modelV2 readers."""
  return {
    'enabled': bool(ss.enabled), 'alert': str(ss.alertType), 'alert_text': str(ss.alertText1),
    'steer': bool(cs.steeringPressed), 'gas': bool(cs.gasPressed), 'brake': bool(cs.brakePressed),
    'state': '' if tp is None else str(tp.state), 'reason': '' if tp is None else str(tp.reason),
    'lc': '' if tp is None else str(tp.laneChange), 'prompt': '' if tp is None else str(tp.signalPrompt),
    'control_prompt': '' if tp is None else str(tp.controlPrompt),
    'lc_state': str(model.meta.laneChangeState),
    'nav_valid': nav is not None, 'nav_type': '' if nav is None else str(nav.maneuverType),
    'nav_mod': '' if nav is None else str(nav.maneuverModifier),
    'nav_dist': 0. if nav is None else round(float(nav.maneuverDistance), 1),
    'nav_remaining': 0. if nav is None else round(float(nav.distanceRemaining), 1),
  }


def route_id(coords) -> int:
  """An id for a navRoute: navd resends the same route every 10 s, a reroute changes it."""
  n = len(coords)
  return hash((n, round(coords[0].latitude, 6), round(coords[n - 1].longitude, 6))) if n else 0


LOG_SERVICES = ('selfdriveState', 'carState', 'turnpilotPlan', 'navInstruction', 'navRoute', 'modelV2')


def rows_from_log(msgs) -> list[dict]:
  """Rows from a recorded message stream (log events in time order, e.g. an rlog's LogReader): one per turnpilotPlan,
  with the latest of the others. A log has no game, so positions, gaps and the destination distance are None."""
  latest: dict = {}
  rows: list[dict] = []
  t0, route = None, 0
  for m in msgs:
    w = m.which()
    if w not in LOG_SERVICES:
      continue
    latest[w] = m
    if w == 'navRoute':
      route = route_id(m.navRoute.coordinates)
    if w != 'turnpilotPlan' or not all(s in latest for s in ('selfdriveState', 'carState', 'modelV2')):
      continue
    t = m.logMonoTime * 1e-9
    t0 = t if t0 is None else t0
    cs = latest['carState'].carState
    nav = latest.get('navInstruction')
    rows.append({'t': round(t - t0, 3), 'wall': None, 'x': None, 'y': None, 'heading': None, 'v': round(cs.vEgo, 2), 'route': route,
                 'dest_dist': None, 'blinker': 'left' if cs.leftBlinker else 'right' if cs.rightBlinker else '',
                 'ped_gap': None, 'car_gap': None, 'pacer_dx': None,
                 **msg_fields(latest['selfdriveState'].selfdriveState, cs, m.turnpilotPlan,
                              nav.navInstruction if nav is not None and nav.valid else None, latest['modelV2'].modelV2)})
  return rows


def disengagements(rows) -> list[dict]:
  """Every enabled to disabled transition, with the driver override or the last alert before it as the cause."""
  out, engaged, last_alert = [], False, None
  for r in rows:
    if r.get('alert'):
      last_alert = r
    if r['enabled']:
      engaged = True
      continue
    if not engaged:
      continue
    engaged = False
    override = next((k for k in ('steer', 'gas', 'brake') if r.get(k)), None)
    if override:
      cause = f"driver {override}"
    elif last_alert is not None and r['t'] - last_alert['t'] <= CAUSE_WINDOW_S:
      cause = f"{last_alert['alert']}: {last_alert.get('alert_text', '')}".rstrip(': ')
    else:
      cause = "unknown"
    out.append({'t': r['t'], 'cause': cause, 'x': r.get('x'), 'y': r.get('y'), 'v': r.get('v')})
  return out


def holds(rows) -> list[dict]:
  """Spans of turnpilotPlan.state holding, with reason, duration, the slowest speed in them and whether the car moved on."""
  spans: list[dict] = []
  for r in rows:
    if r.get('state') != 'holding':
      continue
    cur = spans[-1] if spans else None
    if cur is not None and r['t'] - cur['end'] <= HOLD_MERGE_S and r.get('reason') == cur['reason']:
      cur['end'] = r['t']
      cur['min_v'] = min(cur['min_v'], r['v'])
    else:
      spans.append({'reason': r.get('reason', ''), 'start': r['t'], 'end': r['t'], 'min_v': r['v']})
  for h in spans:
    after = [r for r in rows if h['end'] < r['t'] <= h['end'] + RESUME_WITHIN_S]
    h['duration_s'] = round(h['end'] - h['start'], 2)
    h['stopped'] = h['min_v'] < STOPPED_V
    h['resumed'] = any(r['v'] > RESUMED_V for r in after)
  return spans


def gas_taps(rows) -> list[dict]:
  """Each press of the gas while engaged (an override: the driver's go at a hold), with the go TurnPilot asked for and
  the hold's reason just before it, and how long the car had stood. Pressing it to engage is not one."""
  out, stopped_t = [], None
  for a, b in zip(rows, rows[1:], strict=False):
    stopped_t = (a['t'] if stopped_t is None else stopped_t) if a['v'] < STOPPED_V else None
    if b['enabled'] and b.get('gas') and not (a['enabled'] and a.get('gas')):
      out.append({'t': b['t'], 'prompt': a.get('control_prompt', ''), 'reason': a.get('reason', ''),
                  'stopped_s': 0. if stopped_t is None else round(b['t'] - stopped_t, 2)})
  return out


def _after_advance(rows, i: int) -> tuple[bool, int]:
  """From row i, where navd moved on to a new step: (whether the car left the route, the row where that was settled).
  On the route, the new step's instruction stays valid on the same route until the car has driven CONFIRM_M along it.
  Off it (it went straight on past a turn: navd projects it past the step's end and moves on all the same), navd sends
  no instruction (OFF_ROUTE_DISTANCE) and reroutes seconds later, outside REROUTE_WINDOW_S."""
  base_route, base_left = rows[i].get('route'), rows[i]['nav_remaining']
  for j in range(i, len(rows)):
    r = rows[j]
    if not r.get('nav_valid') or r.get('route') != base_route:
      return True, j
    if base_left - r['nav_remaining'] >= CONFIRM_M:
      return False, j
  return False, len(rows) - 1


def _turn_deg(rows, a: int, b: int, side: int) -> float | None:
  """The most the car's heading turned toward `side` (route_follower.LEFT/RIGHT) over rows a..b, in degrees; None
  without headings (a recorded drive's rows)."""
  hs = [r['heading'] for r in rows[a:b + 1] if r.get('heading') is not None]
  if len(hs) < 2:
    return None
  turned = (np.unwrap(np.array(hs)) - hs[0]) * side
  return round(float(np.degrees(turned.max())), 1)


def maneuvers(rows) -> list[dict]:
  """The turns and keeps of the route in the order navd gave them: attempted once the car came within ATTEMPT_M while
  engaged. Taken if navd then moved on to the next step of the same route and the car stayed on it (_after_advance), and
  for a turn, where the rows have headings, if the car's heading turned TURN_TAKEN_DEG its way. A missed turn makes navd
  reroute, which shows as a new navRoute around the step change (navd sends the instruction and the route in either
  order), or as the car off the route just after it: a straight-through at a turn moves navd on, then off the route."""
  route_changes = []
  prev = None
  for r in rows:
    if r.get('nav_valid'):
      if prev is not None and prev.get('route') is not None and r.get('route') != prev.get('route'):
        route_changes.append(r['t'])
      prev = r
  steps: list[dict] = []
  cur = None
  for i, r in enumerate(rows):
    if not r.get('nav_valid'):
      continue
    future = r['nav_remaining'] - r['nav_dist']  # navd: the route left after this step's maneuver, fixed per step
    if cur is None or abs(future - cur['future']) > STEP_CHANGE_M:
      if cur is not None:
        rerouted = any(abs(t - r['t']) <= REROUTE_WINDOW_S for t in route_changes)
        cur['end'] = 'advanced' if future < cur['future'] and not rerouted else 'reroute'
        cur['end_i'] = i
      cur = {'future': future, 'type': r.get('nav_type', ''), 'modifier': r.get('nav_mod', ''), 'attempted': False, 't': None,
             'attempt_i': None}
      steps.append(cur)
    cur['type'], cur['modifier'] = r.get('nav_type', ''), r.get('nav_mod', '')
    if r['nav_dist'] < ATTEMPT_M and r['enabled'] and not cur['attempted']:
      cur['attempted'], cur['t'], cur['attempt_i'] = True, r['t'], i
  out = []
  for s in steps:
    kind, side = classify(s['type'], s['modifier'])
    if kind not in (TURN, KEEP):
      continue
    advanced = s['attempted'] and s.get('end') == 'advanced'
    left_route, turn_deg = False, None
    if advanced:
      left_route, settled = _after_advance(rows, s['end_i'])
      if kind == TURN:
        turn_deg = _turn_deg(rows, s['attempt_i'], settled, side)
    taken = advanced and not left_route and (turn_deg is None or turn_deg >= TURN_TAKEN_DEG)
    out.append({'type': s['type'], 'modifier': s['modifier'], 'attempted': s['attempted'], 't': s['t'], 'taken': taken,
                'left_route_after': left_route, 'turn_deg': turn_deg})
  return out


def min_gap(rows, key):
  vals = [r[key] for r in rows if r.get(key) is not None]
  return round(min(vals), 2) if vals else None


def oncoming_offsets(rows, road) -> list:
  """For each row, m the car's centreline is left of the divider between the two directions of the road it drives along
  (negative: on its own side); None inside a junction box (a turn crosses there), off the roads, across a road, or with
  no position. road: catalog.road_model's {'links': [[ax, ay, bx, by, ra, rb, lanes], ...], 'junctions': [[x, y, box], ...]},
  lanes as BeamNG's lane string for travel a -> b, left to right ('+' that way, '-' the other). A road with no '-' lane
  our way has no oncoming lane; one with only '-' lanes is all oncoming (the wrong way down a one-way road)."""
  links = road['links']
  if not links:
    return [None] * len(rows)
  L = np.array([lk[:6] for lk in links], dtype=float)
  lanes = [lk[6] for lk in links]
  a, d = L[:, 0:2], L[:, 2:4] - L[:, 0:2]
  length = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
  u = d / length[:, None]
  J = np.array([j[:3] for j in road['junctions']], dtype=float).reshape(-1, 3)
  cos_aligned = np.cos(np.radians(ALIGNED_DEG))
  out = []
  for r in rows:
    if r.get('x') is None or r.get('y') is None or r.get('heading') is None:
      out.append(None)
      continue
    p = np.array([r['x'], r['y']])
    if len(J) and (np.hypot(J[:, 0] - p[0], J[:, 1] - p[1]) < J[:, 2]).any():
      out.append(None)
      continue
    h = np.array([np.cos(r['heading']), np.sin(r['heading'])])
    f = np.clip(((p - a) * u).sum(axis=1) / length, 0., 1.)
    radius = L[:, 4] + f * (L[:, 5] - L[:, 4])
    rel = p - (a + f[:, None] * d)
    dist = np.linalg.norm(rel, axis=1)
    along = u @ h
    ok = (np.abs(along) >= cos_aligned) & (dist <= radius + ON_ROAD_M)
    if not ok.any():
      out.append(None)
      continue
    k = int(np.argmin(np.where(ok, dist, np.inf)))
    fwd = 1. if along[k] > 0 else -1.
    lane_str = lanes[k] if fwd > 0 else ''.join('+' if c == '-' else '-' for c in reversed(lanes[k]))
    n_against = lane_str.count('-')
    if n_against == 0:
      out.append(None)
      continue
    ux, uy = u[k] * fwd
    right = rel[k, 0] * uy - rel[k, 1] * ux  # m right of the centreline, for our travel
    divider = -radius[k] + 2 * radius[k] / len(lane_str) * n_against  # right of the centreline, for our travel
    out.append(round(float(divider - right), 2))
  return out


def wrong_side(rows, road) -> tuple[float | None, list[dict]]:
  """m driven in an oncoming lane (WRONG_SIDE_PAST_M past the divider, outside the junction boxes), and each stretch of
  it; None without positions (a recorded drive) or without a road model (or one with no roads in it)."""
  if road is None or not road.get('links') or not any(r.get('x') is not None for r in rows):
    return None, []  # with positions, check() fails the run for it: an empty road model checks nothing
  past = oncoming_offsets(rows, road)
  total, spans, cur = 0., [], None
  for i, r in enumerate(rows):
    if past[i] is not None and past[i] > WRONG_SIDE_PAST_M:
      dm = r['v'] * (r['t'] - rows[i - 1]['t']) if i else 0.
      total += dm
      if cur is None:
        cur = {'t': r['t'], 'x': r['x'], 'y': r['y'], 'm': 0., 'max_past_m': past[i]}
        spans.append(cur)
      cur['m'] = round(cur['m'] + dm, 1)
      cur['max_past_m'] = max(cur['max_past_m'], past[i])
    else:
      cur = None
  return round(total, 1), spans


def lane_changes(rows) -> list[dict]:
  """Each start of a DesireHelper lane change, with where the pacer car was then."""
  out, prev = [], None
  for r in rows:
    s = r.get('lc_state')
    if s == 'laneChangeStarting' and prev != s:
      out.append({'t': r['t'], 'pacer_dx': r.get('pacer_dx'), 'v': r['v']})
    prev = s
  return out


def stop_gap(rows, outcome: str = ''):
  """Camera-to-mannequin gap where the car stopped for one: where the run ended if it ended stopped, else where the car
  first stood after moving (a pause at the rolling start is not a stop)."""
  if outcome == 'stopped' and rows and rows[-1].get('ped_gap') is not None:
    return round(rows[-1]['ped_gap'], 2)
  moved = False
  for r in rows:
    moved = moved or r['v'] > RESUMED_V
    if moved and r['v'] < STOPPED_V and r.get('ped_gap') is not None:
      return round(r['ped_gap'], 2)
  return None


class EndCheck:
  """Decides live when a run is over: arrived, disengaged, stopped (standing end_on_stop_s after moving), or timeout."""
  def __init__(self, timeout_s: float, end_on_stop_s: float = 0.):
    self.timeout_s, self.end_on_stop_s = timeout_s, end_on_stop_s
    self.engaged = self.moved = False
    self.stopped_t = None

  def update(self, r) -> str | None:
    self.engaged = self.engaged or r['enabled']
    if self.engaged and not r['enabled']:
      return 'disengaged'
    if (r.get('dest_dist') is not None and r['dest_dist'] < ARRIVE_M) or r.get('state') == 'arrived':
      return 'arrived'
    self.moved = self.moved or r['v'] > RESUMED_V
    if self.end_on_stop_s > 0 and self.moved and r['v'] < STOPPED_V:
      self.stopped_t = r['t'] if self.stopped_t is None else self.stopped_t
      if r['t'] - self.stopped_t >= self.end_on_stop_s:
        return 'stopped'
    else:
      self.stopped_t = None
    if r['t'] >= self.timeout_s:
      return 'timeout'
    return None


def _hold_with(hs, word):
  return [h for h in hs if word in h['reason']]


GO_PROMPTS = ('stopSignGo', 'signalGo', 'unprotectedTurnGo')  # turnpilotPlan.controlPrompt values that ask for the driver's go


def check(name: str, res: dict) -> list[str]:
  """Why a scored run fails its scenario (empty: it passes)."""
  fails = []
  if res['disengagements']:
    fails.append(f"{len(res['disengagements'])} disengagement(s): {res['disengagements'][0]['cause']}")
  if res['positions'] and res['wrong_side_m'] is None:
    fails.append("no road model: the oncoming lane was not checked")
  elif res['wrong_side_m'] is not None and res['wrong_side_m'] >= WRONG_SIDE_MIN_M:
    w = res['wrong_side'][0]
    fails.append(f"drove {res['wrong_side_m']:.0f} m in the oncoming lane outside the junctions, first at t={w['t']:.1f} s " +
                 f"({w['x']:.0f}, {w['y']:.0f}), up to {max(x['max_past_m'] for x in res['wrong_side']):.1f} m past the divider")
  asked = [x for x in res.get('gas_taps', []) if x['prompt'] not in GO_PROMPTS]
  if asked:
    x = asked[0]
    fails.append(f"{len(asked)} gas press(es) with no go asked for, first at t={x['t']:.1f} s ({x['reason'] or 'no hold'}, " +
                 f"stood {x['stopped_s']} s)")
  taken, attempted = res['maneuvers_taken'], res['maneuvers_attempted']
  hs = res['holds']
  if name == 'ped_crosswalk_right':
    h = _hold_with(hs, 'crosswalk')
    if not any(x['stopped'] for x in h):
      fails.append("never held to a stop for the pedestrian in the crosswalk")
    elif not any(x['resumed'] for x in h if x['stopped']):
      fails.append("did not move on after the crosswalk cleared")
  elif name.startswith('unprotected_left'):  # and its variants at other approaches
    h = _hold_with(hs, 'oncoming')
    if not h:
      fails.append("never waited for oncoming traffic")
    elif not any(x['resumed'] for x in h):
      fails.append("did not go after the oncoming cars passed")
    if res['min_gap_car_m'] is not None and res['min_gap_car_m'] < CAR_MIN_GAP_M:
      fails.append(f"came within {res['min_gap_car_m']} m of an oncoming car")
    if 'unprotectedTurnGo' not in res['stopped_for_go']:
      fails.append("never stopped at the unprotected left's hold line for the driver's go")
    if not any(x['prompt'] == 'unprotectedTurnGo' for x in res['gas_taps']):
      fails.append("the driver never gave the go at the unprotected left's hold line")
  elif name == 'ped_in_path':
    g = res['stop_gap_m']
    if res['outcome'] != 'stopped' or g is None:
      fails.append("did not stop for the pedestrian")
    elif not PED_STOP_GAP[0] <= g <= PED_STOP_GAP[1]:
      fails.append(f"stopped {g} m from the pedestrian, outside {PED_STOP_GAP[0]}-{PED_STOP_GAP[1]} m")
  elif name == 'occupied_lane':
    if not res['lane_blocked']:
      fails.append("the planner never saw the target lane as occupied")
    if not res['lane_changes']:
      fails.append("never changed lanes")
    elif (dx := res['lane_changes'][0]['pacer_dx']) is not None and abs(dx) < LC_BLOCK_X_M:
      fails.append(f"changed lanes with the pacer {dx:.1f} m alongside")
  elif name == 'lane_change':
    if not res['lane_changes']:
      fails.append("never changed lanes")
  if name in ('ped_crosswalk_right', 'lane_change', 'route_demo') or name.startswith('unprotected_left'):
    if res['outcome'] != 'arrived':
      fails.append(f"ended {res['outcome']}, not arrived")
    if attempted == 0 or taken < attempted:
      fails.append(f"took {taken} of {attempted} maneuvers")
  if name in ('ped_crosswalk_right', 'ped_in_path') and res['min_gap_ped_m'] is not None and res['min_gap_ped_m'] < COLLISION_GAP_M:
    fails.append(f"came within {res['min_gap_ped_m']} m of the pedestrian")
  return fails


def summarize(name: str, rows, outcome: str, road: dict | None = None) -> dict:
  """road: catalog.road_model's roads around the run, for the oncoming-lane check (rows with positions need one)."""
  ms = maneuvers(rows)
  wrong_m, wrong_spans = wrong_side(rows, road)
  res = {
    'outcome': outcome,
    'maneuvers_taken': sum(m['taken'] for m in ms),
    'maneuvers_attempted': sum(m['attempted'] for m in ms),
    'maneuvers': ms,
    'disengagements': disengagements(rows),
    'holds': holds(rows),
    'min_gap_ped_m': min_gap(rows, 'ped_gap'),
    'min_gap_car_m': min_gap(rows, 'car_gap'),
    'stop_gap_m': stop_gap(rows, outcome),
    'gap_reference': "camera mount (top of the windshield) to the mannequin, or to the other car's centre",
    'lane_changes': lane_changes(rows),
    'gas_taps': gas_taps(rows),
    'positions': any(r.get('x') is not None for r in rows),
    'wrong_side_m': wrong_m,
    'wrong_side': wrong_spans,
    # the gos TurnPilot asked for with the car standing (stopSignGo, unprotectedTurnGo, signalGo)
    'stopped_for_go': sorted({r['control_prompt'] for r in rows if r.get('control_prompt', 'none') not in ('', 'none') and r['v'] < STOPPED_V}),
    'lane_blocked': any(r.get('reason') == 'car in target lane' for r in rows),
    'signal_prompts': sum(1 for a, b in zip(rows, rows[1:], strict=False) if b.get('blinker') and b['blinker'] != a.get('blinker')),
  }
  res['failures'] = check(name, res)
  res['passed'] = not res['failures']
  return res


def report(label: str, res: dict) -> str:
  holds = [(h['reason'], h['duration_s']) for h in res['holds']]
  taps = [(x['prompt'], x['stopped_s']) for x in res.get('gas_taps', [])]
  line = f"{label}: {'PASS' if res['passed'] else 'FAIL'} {res['outcome']}, {res['maneuvers_taken']}/{res['maneuvers_attempted']} maneuvers, " + \
         f"{len(res['disengagements'])} disengagements, holds {holds}, gas taps {taps}, min gap ped {res['min_gap_ped_m']} car {res['min_gap_car_m']}, " + \
         f"oncoming lane {res.get('wrong_side_m')} m"
  return line + ''.join(f"\n  {f}" for f in res['failures'])


def main():
  ap = argparse.ArgumentParser(description="Rescore scenario results, or score a recorded drive's rlogs.")
  ap.add_argument('paths', nargs='+', help='result.json files, or with --scenario, rlog segments in order')
  ap.add_argument('--scenario', default='', help='score these rlogs as this scenario (route_demo: arrival and maneuvers)')
  args = ap.parse_args()
  if args.scenario:
    from openpilot.tools.lib.logreader import LogReader
    rows = [r for p in args.paths for r in rows_from_log(LogReader(p, sort_by_time=True))]
    print(report(f"{args.scenario} from {len(args.paths)} log(s)", summarize(args.scenario, rows, 'arrived' if any(
      r['state'] == 'arrived' for r in rows) else 'log ended')))
    return
  for path in args.paths:
    with open(path) as f:
      result = json.load(f)
    for run in result['runs']:
      print(report(f"{result['scenario']} run {run['run']}", summarize(result['scenario'], run['rows'], run['outcome'], road=result.get('road'))))


if __name__ == "__main__":
  main()
