#!/usr/bin/env python3
"""Stats of routed drives (route_drive.py) from their rlogs: maneuvers taken out of attempted, disengagements and the
events behind them, TurnPilot holds, reroutes, desire pulses, lane changes, arrival, and whether the detector in modeld
was still running at the end (it turns itself off for the drive after a few slow frames).

Every route navd publishes (navRoute) is a version; one whose end moved is a new leg, any other a reroute. A version's
turn and keep maneuvers come from the first navInstruction after it (allManeuvers, placed along the route from the car's
own progress). The car attempted one once its progress along that version reached the maneuver, and took it once it went
on TAKEN_PAST_M past it on the route; one attempted and not taken before the next version is missed.

  python -m openpilot.tools.sim.bridge.beamng.route_stats loop.json --out loop_stats.json
"""
import argparse
import glob
import json
import os

import numpy as np

from openpilot.selfdrive.navigation.route import RouteTracker, local_meters
from openpilot.selfdrive.turnpilot.hooks import CLEAR_DESIRE
from openpilot.selfdrive.turnpilot.route_follower import KEEP, TURN, classify
from openpilot.tools.lib.logreader import LogReader

REALDATA = os.path.expanduser(f"~/.comma{os.environ.get('OPENPILOT_PREFIX', 'tph1')}/media/0/realdata")
SERVICES = ('gpsLocationExternal', 'navRoute', 'navInstruction', 'selfdriveState', 'onroadEvents', 'turnpilotPlan', 'modelV2', 'carState',
            'turnpilotDetections')
NEW_LEG_M = 30.     # a route whose end moved this far is a new destination
REACHED_M = 5.      # progress within this of a maneuver: attempted
TAKEN_PAST_M = 30.  # progress this far past it, on the route: taken
ON_ROUTE_M = 10.    # lateral distance from the route that still counts as on it
DETECTOR_GAP_S = 5.  # no turnpilotDetections for this long before the run's end: the detector was off by then


def segments_for(wall_start: float, wall_end: float, realdata: str = REALDATA) -> list[str]:
  """rlogs that may hold [wall_start, wall_end]: written to after the start, created before the end."""
  out = []
  for f in glob.glob(os.path.join(realdata, '*', 'rlog.zst')):
    st = os.stat(f)
    if st.st_mtime >= wall_start and st.st_birthtime <= wall_end + 5:
      out.append(f)
  return sorted(out, key=lambda f: os.stat(f).st_birthtime)


def read_window(files, t0_ns: int, t1_ns: int) -> dict:
  msgs: dict[str, list] = {s: [] for s in SERVICES}
  for f in files:
    for m in LogReader(f, sort_by_time=True):
      w = m.which()
      if w in msgs and t0_ns <= m.logMonoTime <= t1_ns:
        msgs[w].append((m.logMonoTime * 1e-9, m.valid, getattr(m, w).as_builder().to_dict() if w == 'navRoute' else getattr(m, w)))
  return msgs


def route_versions(msgs):
  """[(t, coords (n, 2) lon/lat, kind 'first'/'leg'/'reroute')] in time order."""
  versions, last_key = [], None
  for t, _, nr in msgs['navRoute']:
    coords = np.array([[c['longitude'], c['latitude']] for c in nr.get('coordinates', [])])
    if len(coords) < 2:
      continue
    key = (len(coords), round(float(coords[0, 0]), 6), round(float(coords[-1, 0]), 6), round(float(coords[len(coords) // 2, 1]), 6))
    if key == last_key:
      continue
    kind = 'first'
    if versions:
      moved = float(np.linalg.norm(local_meters(coords[-1], versions[-1][1][-1])))
      kind = 'leg' if moved > NEW_LEG_M else 'reroute'
    versions.append((t, coords, kind))
    last_key = key
  return versions


def maneuvers_of(version, msgs, gps):
  """[(along m, type, modifier, kind, side)] of a version's turns and keeps, from the first navInstruction after it."""
  t_v, coords, _ = version
  for t, valid, ni in msgs['navInstruction']:
    if t < t_v + 0.5 or not valid or not len(ni.allManeuvers):
      continue
    fix = gps_at(gps, t)
    if fix is None:
      return []
    progress = RouteTracker(coords).project(*fix)
    out = []
    for m in ni.allManeuvers:
      kind, side = classify(m.type, m.modifier)
      if kind in (TURN, KEEP):
        out.append((progress + m.distance, m.type, m.modifier, kind, side))
    return out
  return []


def gps_at(gps, t):
  i = int(np.searchsorted(gps[:, 0], t))
  i = min(max(i, 0), len(gps) - 1)
  return (gps[i, 1], gps[i, 2]) if abs(gps[i, 0] - t) < 1.0 else None


def judge(version, end_t, maneuvers, gps, arrived_end: bool):
  """Each maneuver: 'taken', 'missed' or None (not reached), over the car's track while this version was the route."""
  t_v, coords, _ = version
  tracker = RouteTracker(coords)
  track = []
  for t, lon, lat in gps:
    if t_v <= t < end_t:
      progress = tracker.project(lon, lat)
      _, lateral = lateral_distance(coords, lon, lat)
      track.append((t, progress, lateral))
  out = []
  for along, *_ in maneuvers:
    reached = [r for r in track if r[1] >= along - REACHED_M]
    if not reached:
      out.append(None)
      continue
    past = [r for r in reached if r[1] >= along + TAKEN_PAST_M and r[2] < ON_ROUTE_M]
    out.append('taken' if past or (arrived_end and reached[-1][2] < ON_ROUTE_M) else 'missed')
  return out


def lateral_distance(coords, lon, lat):
  from openpilot.selfdrive.navigation.route import along_path
  return along_path(coords, (lon, lat))


def episodes(times, flags, labels=None):
  """[(start, end, label)] of runs where flags is true."""
  out, start = [], None
  for i, (t, f) in enumerate(zip(times, flags, strict=True)):
    if f and start is None:
      start = i
    elif not f and start is not None:
      out.append((times[start], t, labels[start] if labels else ''))
      start = None
  if start is not None:
    out.append((times[start], times[-1], labels[start] if labels else ''))
  return out


def run_stats(run: dict, realdata: str = REALDATA) -> dict:
  wall0 = run['wall_start']
  wall1 = wall0 + run['mono_end'] - run['mono_start']
  files = segments_for(wall0, wall1, realdata)
  msgs = read_window(files, int(run['mono_start'] * 1e9), int(run['mono_end'] * 1e9))
  gps = np.array([(t, g.longitude, g.latitude) for t, _, g in msgs['gpsLocationExternal'] if g.hasFix]).reshape(-1, 3)

  versions = route_versions(msgs)
  rows, taken, attempted = [], 0, 0
  for i, v in enumerate(versions):
    end_t = versions[i + 1][0] if i + 1 < len(versions) else run['mono_end'] + 1
    next_kind = versions[i + 1][2] if i + 1 < len(versions) else ('arrived' if run['end'] == 'arrived' else run['end'])
    mans = maneuvers_of(v, msgs, gps)
    verdicts = judge(v, end_t, mans, gps, next_kind in ('leg', 'arrived'))
    for (along, mtype, modifier, kind, _side), verdict in zip(mans, verdicts, strict=True):
      rows.append({"route_version": i, "version_kind": v[2], "along_m": round(along, 1), "type": mtype, "modifier": modifier,
                   "kind": kind, "result": verdict})
      attempted += verdict is not None
      taken += verdict == 'taken'

  sd = msgs['selfdriveState']
  t_sd = [t for t, _, _ in sd]
  enabled = [s.enabled for _, _, s in sd]
  first_on = next((t for t, e in zip(t_sd, enabled, strict=True) if e), None)
  dis = []
  for i in range(1, len(sd)):
    if enabled[i - 1] and not enabled[i] and first_on is not None:
      t = t_sd[i]
      ev = [(te, [str(e.name) for e in evs]) for te, _, evs in msgs['onroadEvents'] if t - 0.5 <= te <= t + 0.1]
      names = sorted({n for _, ns in ev for n in ns})
      after = next((s for ts, _, s in sd if ts >= t), sd[i][2])
      dis.append({"t_s": round(t - run['mono_start'], 1), "events": names, "alert": str(after.alertType),
                  "alert_text": f"{after.alertText1} {after.alertText2}".strip()})

  plan = msgs['turnpilotPlan']
  t_p = [t for t, _, _ in plan]
  holds = episodes(t_p, [str(p.state) == 'holding' for _, _, p in plan], [p.reason for _, _, p in plan])
  desires = [int(p.desire) for _, _, p in plan]
  pulses = sum(1 for a, b in zip([0] + desires[:-1], desires, strict=True) if 0 < b < CLEAR_DESIRE and not a)
  clears = sum(1 for a, b in zip([0] + desires[:-1], desires, strict=True) if b == CLEAR_DESIRE and a != b)
  t_m = [t for t, _, _ in msgs['modelV2']]
  model_hz = (len(t_m) - 1) / (t_m[-1] - t_m[0]) if len(t_m) > 1 and t_m[-1] > t_m[0] else 0.  # one model run per camera pair
  prompts = episodes(t_p, [str(p.signalPrompt) != 'none' for _, _, p in plan], [str(p.signalPrompt) for _, _, p in plan])
  lc_state = [str(m.meta.laneChangeState) for _, _, m in msgs['modelV2']]
  lane_changes = sum(1 for a, b in zip(['off'] + lc_state[:-1], lc_state, strict=True) if b == 'laneChangeStarting' and a != b)
  states = {}
  for _, _, p in plan:
    states[str(p.state)] = states.get(str(p.state), 0) + 1

  # the driver's gas while engaged: an override (DisengageOnAccelerator off), e.g. going on after a stop sign
  cs = msgs['carState']
  en_at = lambda t: enabled[min(max(int(np.searchsorted(t_sd, t)) - 1, 0), len(enabled) - 1)] if enabled else False  # noqa: E731
  overrides = episodes([t for t, _, _ in cs], [c.gasPressed and en_at(t) for t, _, c in cs])

  nav = [(t, ni.distanceRemaining) for t, valid, ni in msgs['navInstruction'] if valid]
  t_det = [t for t, _, _ in msgs['turnpilotDetections']]
  detector_up = bool(t_det) and run['mono_end'] - t_det[-1] < DETECTOR_GAP_S
  return {
    "run": run['run'], "route": run['route'], "end": run['end'], "arrived": run['end'] == 'arrived', "wall_start": wall0,
    "duration_s": run['duration_s'], "distance_m": run['distance_m'], "traffic": run.get('traffic', 0),
    "maneuvers_taken": taken, "maneuvers_attempted": attempted, "maneuvers_missed": attempted - taken,
    "disengagements": dis, "disengagement_count": len(dis),
    "gas_overrides": [round(a - run['mono_start'], 1) for a, _, _ in overrides],
    "holds": [{"t_s": round(a - run['mono_start'], 1), "dur_s": round(b - a, 1), "reason": r} for a, b, r in holds],
    "reroutes": sum(v[2] == 'reroute' for v in versions), "legs_routed": sum(v[2] in ('first', 'leg') for v in versions),
    "desire_pulses": pulses, "desire_clears": clears, "model_hz": round(model_hz, 1),
    "detector_up": detector_up, "detector_runs": len(t_det),
    "detector_last_s": round(t_det[-1] - run['mono_start'], 1) if t_det else None,
    "lane_change_prompts": len(prompts), "lane_changes": lane_changes,
    "plan_state_frames": states, "min_distance_remaining_m": round(min((d for _, d in nav), default=float('nan')), 1),
    "maneuvers": rows, "rlogs": files,
  }


def summary_line(s: dict) -> str:
  causes = "; ".join(f"{d['t_s']}s {','.join(d['events']) or d['alert']}" for d in s['disengagements']) or "-"
  holds = f"{len(s['holds'])} ({sum(h['dur_s'] for h in s['holds']):.0f} s)"
  return (f"{s['route']} run {s['run']} | traffic {s['traffic']} | end {s['end']} | maneuvers {s['maneuvers_taken']}/{s['maneuvers_attempted']} | " +
          f"disengagements {s['disengagement_count']}: {causes} | gas overrides {len(s['gas_overrides'])} | holds {holds} | " +
          f"reroutes {s['reroutes']} | " +
          f"desire pulses {s['desire_pulses']} | clears {s.get('desire_clears', 0)} | model {s.get('model_hz', 0)} Hz | " +
          f"detector {'up' if s.get('detector_up') else 'OFF from ' + str(s.get('detector_last_s')) + ' s'} | " +
          f"lane changes {s['lane_changes']}/{s['lane_change_prompts']} prompted | " +
          f"{s['distance_m']:.0f} m in {s['duration_s']:.0f} s")


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('runs_json', help="route_drive.py's --out")
  ap.add_argument('--out', default='')
  ap.add_argument('--realdata', default=REALDATA)
  args = ap.parse_args()
  with open(args.runs_json) as f:
    runs = json.load(f)['runs']
  stats = [run_stats(r, args.realdata) for r in runs]
  for s in stats:
    print(summary_line(s))
  with open(args.out or args.runs_json.replace('.json', '_stats.json'), 'w') as f:
    json.dump(stats, f, indent=1)


if __name__ == "__main__":
  main()
