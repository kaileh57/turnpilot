#!/usr/bin/env python3
"""Scenario scoring from recorded rows, one row per 50 ms of a run (run.py records them from msgq and the game).

A row is a dict:
  t, wall            s since the run started, and wall-clock time
  x, y, heading, v   the ego's camera mount in BeamNG world meters, heading rad CCW from east, wheel speed m/s
  enabled            selfdriveState.enabled; alert, alert_text: its alertType and alertText1
  steer, gas, brake  carState steeringPressed, gasPressed, brakePressed
  state, reason      turnpilotPlan.state and .reason; lc, prompt: its laneChange and signalPrompt
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

from openpilot.selfdrive.turnpilot.lane_planner import LC_BLOCK_X_M
from openpilot.selfdrive.turnpilot.route_follower import KEEP, TURN, classify

STOPPED_V = 0.3        # m/s: standing
RESUMED_V = 2.0        # m/s: moving again after a hold
RESUME_WITHIN_S = 15.  # a hold counts as resumed if the car is moving again this soon after it ends
HOLD_MERGE_S = 1.0     # holds with the same reason this close together are one
ATTEMPT_M = 30.        # reaching this close to a maneuver on the route is an attempt
STEP_CHANGE_M = 2.     # the distance left after the current maneuver changing by more than this is a new step
REROUTE_WINDOW_S = 1.5 # a new navRoute this close to a step change means navd rerouted (it runs at 1 Hz)
ARRIVE_M = 15.         # route_follower.ARRIVE_DIST_M, measured to the destination in the game
CAUSE_WINDOW_S = 5.    # a disengagement's cause is the last alert this soon before it
COLLISION_GAP_M = 2.5  # camera mount to object: about where the bumper touches (the mount sits ~2 m behind it)
PED_STOP_GAP = (3., 8.)
CAR_MIN_GAP_M = 4.     # camera mount to the other car's centre


def msg_fields(ss, cs, tp, nav, model) -> dict:
  """The message part of a row from selfdriveState, carState, turnpilotPlan (None before the first), navInstruction (None
  while navd has no valid route) and modelV2 readers."""
  return {
    'enabled': bool(ss.enabled), 'alert': str(ss.alertType), 'alert_text': str(ss.alertText1),
    'steer': bool(cs.steeringPressed), 'gas': bool(cs.gasPressed), 'brake': bool(cs.brakePressed),
    'state': '' if tp is None else str(tp.state), 'reason': '' if tp is None else str(tp.reason),
    'lc': '' if tp is None else str(tp.laneChange), 'prompt': '' if tp is None else str(tp.signalPrompt),
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


def maneuvers(rows) -> list[dict]:
  """The turns and keeps of the route in the order navd gave them: attempted once the car came within ATTEMPT_M while
  engaged, taken if navd then moved on to the next step of the same route. A missed turn makes navd reroute instead,
  which shows as a new navRoute around the step change (navd sends the instruction and the route in either order)."""
  valid = [r for r in rows if r.get('nav_valid')]
  route_changes = [b['t'] for a, b in zip(valid, valid[1:], strict=False) if a.get('route') is not None and b.get('route') != a.get('route')]
  steps: list[dict] = []
  cur = None
  for r in valid:
    future = r['nav_remaining'] - r['nav_dist']  # navd: the route left after this step's maneuver, fixed per step
    if cur is None or abs(future - cur['future']) > STEP_CHANGE_M:
      if cur is not None:
        rerouted = any(abs(t - r['t']) <= REROUTE_WINDOW_S for t in route_changes)
        cur['end'] = 'advanced' if future < cur['future'] and not rerouted else 'reroute'
      cur = {'future': future, 'type': r.get('nav_type', ''), 'modifier': r.get('nav_mod', ''), 'attempted': False, 't': None}
      steps.append(cur)
    cur['type'], cur['modifier'] = r.get('nav_type', ''), r.get('nav_mod', '')
    if r['nav_dist'] < ATTEMPT_M and r['enabled'] and not cur['attempted']:
      cur['attempted'], cur['t'] = True, r['t']
  out = []
  for s in steps:
    kind, _ = classify(s['type'], s['modifier'])
    if kind in (TURN, KEEP):
      out.append({'type': s['type'], 'modifier': s['modifier'], 'attempted': s['attempted'], 't': s['t'],
                  'taken': s['attempted'] and s.get('end') == 'advanced'})
  return out


def min_gap(rows, key):
  vals = [r[key] for r in rows if r.get(key) is not None]
  return round(min(vals), 2) if vals else None


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


def check(name: str, res: dict) -> list[str]:
  """Why a scored run fails its scenario (empty: it passes)."""
  fails = []
  if res['disengagements']:
    fails.append(f"{len(res['disengagements'])} disengagement(s): {res['disengagements'][0]['cause']}")
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


def summarize(name: str, rows, outcome: str) -> dict:
  ms = maneuvers(rows)
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
    'lane_blocked': any(r.get('reason') == 'car in target lane' for r in rows),
    'signal_prompts': sum(1 for a, b in zip(rows, rows[1:], strict=False) if b.get('blinker') and b['blinker'] != a.get('blinker')),
  }
  res['failures'] = check(name, res)
  res['passed'] = not res['failures']
  return res


def report(label: str, res: dict) -> str:
  holds = [(h['reason'], h['duration_s']) for h in res['holds']]
  line = f"{label}: {'PASS' if res['passed'] else 'FAIL'} {res['outcome']}, {res['maneuvers_taken']}/{res['maneuvers_attempted']} maneuvers, " + \
         f"{len(res['disengagements'])} disengagements, holds {holds}, min gap ped {res['min_gap_ped_m']} car {res['min_gap_car_m']}"
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
      print(report(f"{result['scenario']} run {run['run']}", summarize(result['scenario'], run['rows'], run['outcome'])))


if __name__ == "__main__":
  main()
