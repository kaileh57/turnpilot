"""The side views at a stand on the car's budget (modeld_detector's stand schedule), on test_far_first's model of the car:
the big model's time as on 11-06, the full pass's run and each step's extra, and a side run's cost as the car logged it
(route 26, its only sweep: 11.3 ms the run, 7.1 ms its sectors; the load's warm runs 11.0-12.7 ms), its sectors at the
faster judge's share of that (test_sideview_cost.py: 0.53 of the plain one's time on this Mac). Until the stand
schedule, no side view ran past the first at a stand on this car: the held cycle left every odd step 2-4 ms of debt, and a
side run's step left more than DEBT_CAP_S from any step (route 26: one side run in 9.6 s of sweep, noRoom the rest; route 29
never asked for one). Each stand is a minute of steps; per pass the runs a second, the longest gap between runs, and every
step's lateness, which must stay under a third of a frame (no frame dropped)."""
import math
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.turnpilot.detector import modeld_detector
from openpilot.selfdrive.turnpilot.detector.modeld_detector import CYCLE_S, FORCE_SKIP_LATE_S, REST_S, STAND_VIEW_DUE_S, TIME_WINDOW
from openpilot.selfdrive.turnpilot.detector.sideview import SIDE_VIEWS
from openpilot.selfdrive.turnpilot.detector.tests.test_far_first import EXTRA_SD_S, FULL_S, LIGHT_S, PAINT_S, ROAD_EXTRA_S, RUN_SD_S, \
  TAIL_P, TAIL_S, WIDE_EXTRA_S, model_times
from openpilot.selfdrive.turnpilot.detector.tests.test_modeld_detector import FakeLights, Meta, side_detector
from openpilot.selfdrive.turnpilot.stop_rules import DET_FRESH_S

SIDE_S, SIDE_EXTRA_S = 0.0113, WIDE_EXTRA_S  # route 26's side run; its message as a wide run's
LOGGED_SECTOR_S = 0.0071   # route 26's sectors (the plain judge, a first run)
SECTOR_S = 0.0038          # ...at the faster judge's 0.53 of its time
JUNCTION_LIGHT_S = 0.0025  # the light read on a road run reading two heads or more, on the car's scale (light_replay.py: median)
SECONDS = 60.


class SectorCost:
  """Stands in for SectorQuality: costs SECTOR_S on the sim clock (spread), reads no sector."""
  def __init__(self, clock, s, spread):
    self.clock, self.s, self.spread = clock, s, spread

  def update(self, geom, name, frame, t):
    self.clock.t += self.spread(self.s, RUN_SD_S)
    return []


def stand(monkeypatch, view: str, v_ego: float, signal_near: bool, seed: int = 0, sector_s: float = SECTOR_S,
          light_s: float = LIGHT_S, shift: float = 0., noise: bool = False):
  """SECONDS of steps at a stand on the car's numbers: per frame what ran ('road', 'wide', 'sideLeft', 'sideRight' or None)
  and the lateness the next frame starts with; noise: every timing spread as test_far_first's car_drive spreads them (11-06's
  overheads, a 1% tail of +3 ms on each)."""
  clock = SimpleNamespace(t=1000.)
  det = side_detector(side_time=SIDE_S)  # its maps ready on the real clock, then the sim's
  monkeypatch.setattr(modeld_detector, 'time', SimpleNamespace(monotonic=lambda: clock.t, perf_counter=lambda: clock.t))
  rng = np.random.default_rng(seed + 100)

  def spread(s, sd):
    if not noise:
      return s
    return max(0., s + rng.normal(0., sd)) + (TAIL_S if rng.random() < TAIL_P else 0.)

  def costs(fn, s):
    def run(*args, **kwargs):
      clock.t += spread(s, RUN_SD_S)
      return fn(*args, **kwargs)
    return run
  det.light_mode, det.lights = 'addOnly', FakeLights()
  det.times = deque([FULL_S] * TIME_WINDOW, maxlen=TIME_WINDOW)
  det.paint_times.extend([PAINT_S] * TIME_WINDOW)
  det.light_times.extend([light_s] * TIME_WINDOW)
  det.run = costs(det.run, FULL_S)
  det.lane_paint.read = costs(det.lane_paint.read, PAINT_S)
  det.lights.read = costs(det.lights.read, light_s)
  side = det.side
  side.run = costs(side.run, SIDE_S)
  side.quality = SectorCost(clock, sector_s, spread)
  ran = []

  class PM:
    def send(self, name, msg):
      td = msg.as_reader().turnpilotDetections
      view_ = str(td.view)
      if view_ in SIDE_VIEWS:
        clock.t += spread(SIDE_EXTRA_S, EXTRA_SD_S)
        ran.append(view_)
      else:
        road = str(td.camera) == 'road'
        clock.t += spread(ROAD_EXTRA_S if road else WIDE_EXTRA_S, EXTRA_SD_S)
        ran.append(str(td.camera))
  pm = PM()
  frames = round(SECONDS / CYCLE_S)
  model = model_times(frames, seed, shift)
  t0, late, what = clock.t + REST_S, [], []
  for k in range(frames):
    start = max(clock.t + REST_S, t0 + k * CYCLE_S)
    clock.t = start + float(model[k])
    n = len(ran)
    det.step(pm, det.model_state, float(model[k]), 0., log.Desire.none, [Meta(2 * k), Meta(2 * k + 1)], view, 0., v_ego, signal_near)
    what.append(ran[n] if len(ran) > n else None)
    late.append(max(0., clock.t + REST_S - (t0 + (k + 1) * CYCLE_S)))
  return det, what, np.array(late)


def fresh(what, name: str, within: float = DET_FRESH_S) -> float:
  """The share of the stand's frames, from its second second on (the stand schedule's start behind it), on which the pass's
  last run is within `within` (stop_rules.DET_FRESH_S: past it a view's reach is gone)."""
  last, n, ok = -np.inf, 0, 0
  for k, w in enumerate(what):
    if w == name:
      last = k
    if k * CYCLE_S >= 2.:
      n, ok = n + 1, ok + ((k - last) * CYCLE_S <= within + 1e-9)
  return ok / n


def rates(what) -> dict:
  """Per pass: (runs a second, longest gap between runs in s) over the stand, from its first second (the held cycle's
  STAND_STARVED_S before the stand schedule)."""
  out = {}
  for name in ('road', 'wide') + SIDE_VIEWS:
    at = [k for k, w in enumerate(what) if w == name and k * CYCLE_S >= 1.]
    out[name] = (len(at) / (len(what) * CYCLE_S - 1.), max(np.diff(at)) * CYCLE_S if len(at) > 1 else np.inf)
  return out


STANDS = [
  # (what, view, v_ego, signal mapped near)
  ('stop sign, standing', 'stopSweep', 0., False),
  ('stop sign, creeping', 'stopSweep', 1.5, False),
  ('right on a red, standing', 'sideLook', 0., True),
  ('uncontrolled junction, waiting', 'sideLook', 0., False),
]


@pytest.mark.parametrize("what_, view, v_ego, signal_near", STANDS)
def test_at_a_stand_every_side_view_runs_inside_the_deciders_freshness_on_the_cars_budget(monkeypatch, what_, view, v_ego, signal_near):
  light_s = JUNCTION_LIGHT_S if signal_near else LIGHT_S
  det, what, late = stand(monkeypatch, view, v_ego, signal_near, light_s=light_s)
  r = rates(what)
  print(what_, {k: (round(a, 2), round(b, 2)) for k, (a, b) in r.items()}, {n: round(fresh(what, n), 3) for n in ('wide',) + SIDE_VIEWS},
        f"latest {1e3 * late.max():.1f} ms")
  assert det.stand and not det.side.disabled
  # each side view and the wide pass (the inner band's) inside DET_FRESH_S; at a right on a red the road camera's light read
  # (2.5 ms) leaves the frames too little for all four, the side views fresh on 94-97% of them
  within = 0.9 if signal_near else 0.95
  for name in SIDE_VIEWS + ('wide',):
    assert fresh(what, name) >= within and r[name][0] >= 0.8 / STAND_VIEW_DUE_S, (name, r[name], fresh(what, name))
  assert r['road'][0] >= 4.3 and r['road'][1] <= 0.35  # the lights and the oncoming band near 5 a second (4.4-4.7)
  assert late.max() < FORCE_SKIP_LATE_S + 0.004  # never a frame dropped (a third of a frame, and a run's spread)


def test_crossing_from_the_commit_point_keeps_the_wide_pass_and_the_side_views(monkeypatch):
  det, what, late = stand(monkeypatch, 'stopSweepGo', 1., False)
  r = rates(what)
  print('crossing', {k: (round(a, 2), round(b, 2)) for k, (a, b) in r.items()}, f"latest {1e3 * late.max():.1f} ms")
  assert r['wide'][0] >= 4.8 and r['road'][0] == 0.
  for name in SIDE_VIEWS:
    assert fresh(what, name) >= 0.95, (name, r[name])
  assert late.max() < FORCE_SKIP_LATE_S + 0.004


def test_the_held_cycle_alone_gave_the_side_views_no_frame_on_this_budget(monkeypatch):
  # the repro: the stand schedule held off (its starvation clock never runs out), the car's budget gives the side views
  # nothing past the first, as route 26's sweep logged
  monkeypatch.setattr(modeld_detector, 'STAND_STARVED_S', 1e9)
  det, what, late = stand(monkeypatch, 'stopSweep', 0., False, sector_s=LOGGED_SECTOR_S)
  r = rates(what)
  assert not det.stand and all(r[name][0] < 0.2 for name in SIDE_VIEWS), r


def test_a_dearer_side_view_or_a_slower_model_runs_fewer_never_drops_a_frame_nor_turns_the_side_views_off(monkeypatch):
  # the plain judge's logged 7.1 ms of sectors, or the model 1 ms slower: the side views fall behind DET_FRESH_S (the decider
  # then waits), the road camera keeps its 5 a second near enough, no frame is dropped and no overrun turns them off
  for sector_s, shift in ((LOGGED_SECTOR_S, 0.), (SECTOR_S, 0.001), (0.009, 0.)):
    det, what, late = stand(monkeypatch, 'stopSweep', 0., False, sector_s=sector_s, shift=shift)
    r = rates(what)
    print(sector_s, shift, {k: (round(a, 2), round(b, 2)) for k, (a, b) in r.items()}, f"latest {1e3 * late.max():.1f} ms")
    assert late.max() < FORCE_SKIP_LATE_S + 0.004
    assert r['road'][0] >= 4.3
    assert not det.side.disabled and all(r[name][0] > 0.4 for name in SIDE_VIEWS)


def test_a_look_reads_the_lights_with_no_signal_mapped_and_paints_nothing_at_the_stand(monkeypatch):
  # a look reads the lights on every road run as always (a right on a red or unread light, the signal unmapped: the red
  # rail), where a sweep reads them only near a mapped signal; the paint gives way at the stand (nothing reads it under
  # PAINT_SKIP_V)
  det, what, late = stand(monkeypatch, 'sideLook', 0., False, light_s=JUNCTION_LIGHT_S)
  assert what.count('road') > 200 and len(det.lights.calls) == what.count('road')
  assert det.lane_paint.calls == []
  det, what, late = stand(monkeypatch, 'stopSweep', 0., False)
  assert what.count('road') > 200 and det.lights.calls == []  # the sweep, unmapped: no colour read (lightStatus sweep)


def test_a_look_above_the_side_views_gate_is_auto(monkeypatch):
  # at 5 m/s a look is auto: the road camera ten times a second, the paint read, no side view, the far slot its own
  det, what, late = stand(monkeypatch, 'sideLook', 5., False)
  _, what_auto, _ = stand(monkeypatch, 'auto', 5., False)
  assert what == what_auto and what.count('road') > 590 and not any(w in SIDE_VIEWS for w in what)
  assert not det.stand and det.lane_paint.calls


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_with_the_cars_timing_spread_and_a_slower_model_the_side_views_stay_on(monkeypatch, seed):
  # side runs sit near SIDE_STAND_CAP_S by design, and SIDE_OVERRUNS of TIME_WINDOW past it turn them off for the drive:
  # with every timing spread as on 11-06 and its tails, and the model 1 ms slower, they never are
  for shift in (0., 0.001):
    det, what, late = stand(monkeypatch, 'stopSweep', 0., False, seed=seed, shift=shift, noise=True)
    r = rates(what)
    print(seed, shift, {k: (round(a, 2), round(b, 2)) for k, (a, b) in r.items()}, {n: round(fresh(what, n), 3) for n in SIDE_VIEWS},
          f"latest {1e3 * late.max():.1f} ms")
    assert det.stand and not det.side.disabled and all(r[n][0] > 1.5 for n in SIDE_VIEWS)
    assert late.max() < FORCE_SKIP_LATE_S + 0.006 and r['road'][0] >= 4.3


# modeld's own clock (modeld.py): frames arrive every CYCLE_S and the main camera's vipc client is conflated, so a step that
# ends a whole frame late finds a newer frame and the ones between are dropped (vipc_dropped_frames). One dropped frame
# makes that frame's cameraOdometry invalid (fill_model_msg.fill_pose_msg), locationd's inputsOK false and selfdrived raise
# locationdTemporaryError (soft disable); three within a few seconds put frameDropPerc (FirstOrderFilter, 10 s) past 1:
# modeldLagging. The harness above counts lateness as the time budget does; this one plays the frames as camerad sends them.
# modeld's time per frame outside the model and the detector, measured on the 11-06/11-08 rlogs where a frame started at
# the last one's end: 3.5-3.9 ms (REST_S, the time budget's count of it, is 3), the detector's own step on a frame where
# nothing runs about 0.05 ms (this Mac's 5 us at 8.8x)
REAL_REST_S, STEP_S, UPLOAD_S = 0.0037, 0.00005, 0.042  # an upload: route 26's 38.1-38.6 ms a view, and 10% more


class EofMeta:
  def __init__(self, frame_id, eof_s):
    self.frame_id, self.timestamp_eof = frame_id, int(eof_s * 1e9)


def real_clock(monkeypatch, plan, seed=0, scale=1., ready=True):
  """plan: (seconds, view, v_ego, signal_near) spells, each on modeld's real clock. The desktop-scaled costs (the sectors, the
  junction light read) times scale. Per processed frame: its view, the lateness it started with, the frames dropped before it,
  and whether the side views' maps were on the device."""
  clock = SimpleNamespace(t=1000.)
  det = side_detector(side_time=SIDE_S, ready=ready)
  monkeypatch.setattr(modeld_detector, 'time', SimpleNamespace(monotonic=lambda: clock.t, perf_counter=lambda: clock.t))
  rng = np.random.default_rng(seed + 100)

  def spread(s, sd):
    return max(0., s + rng.normal(0., sd)) + (TAIL_S if rng.random() < TAIL_P else 0.)

  def costs(fn, s, sd=RUN_SD_S):
    def run(*args, **kwargs):
      clock.t += spread(s, sd)
      return fn(*args, **kwargs)
    return run
  det.light_mode, det.lights = 'addOnly', FakeLights()
  det.times = deque([FULL_S] * TIME_WINDOW, maxlen=TIME_WINDOW)
  det.paint_times.extend([PAINT_S] * TIME_WINDOW)
  det.light_times.extend([LIGHT_S] * TIME_WINDOW)
  det.run = costs(det.run, FULL_S)
  det.lane_paint.read = costs(det.lane_paint.read, PAINT_S)
  light = {'s': LIGHT_S}
  read = det.lights.read
  det.lights.read = lambda *a, **kw: (setattr(clock, 't', clock.t + spread(light['s'], RUN_SD_S)), read(*a, **kw))[1]
  det.stand_update = costs(det.stand_update, STEP_S, EXTRA_SD_S / 4)
  side = det.side
  side.run = costs(side.run, SIDE_S)
  side.upload = costs(side.upload, UPLOAD_S)
  side.quality = SectorCost(clock, SECTOR_S * scale, spread)

  class PM:
    def send(self, name, msg):
      td = msg.as_reader().turnpilotDetections
      clock.t += spread(ROAD_EXTRA_S if str(td.camera) == 'road' and str(td.view) not in SIDE_VIEWS else WIDE_EXTRA_S, EXTRA_SD_S / 2)
  spells, t = [], 0.
  for secs, view, v_ego, signal_near in plan:
    spells.append((t + secs, view, v_ego, signal_near))
    t += secs
  model = model_times(round(t / CYCLE_S) + 1, seed)
  t0, end, last = clock.t, clock.t, -1
  out = []
  while True:
    arrived = math.floor((end - t0) / CYCLE_S + 1e-9)  # the newest frame camerad has sent by now (conflated)
    j = arrived if arrived > last else last + 1
    start = end if arrived > last else t0 + j * CYCLE_S
    if j * CYCLE_S >= t:
      return det, out
    _, view, v_ego, signal_near = next(sp for sp in spells if j * CYCLE_S < sp[0])
    light['s'] = (JUNCTION_LIGHT_S if signal_near or view == 'sideLook' else LIGHT_S) * scale
    clock.t = start + max(0., REAL_REST_S + rng.normal(0., 0.00035)) + float(model[j])
    eof = t0 + j * CYCLE_S
    det.step(PM(), det.model_state, float(model[j]), 0., log.Desire.none, [EofMeta(j, eof), EofMeta(j, eof)], view, 0., v_ego, signal_near)
    out.append((view, start - eof, j - last - 1 if last >= 0 else 0, side.geometry is not None))
    end, last = clock.t, j


def frame_drop_perc(dropped) -> float:
  """modeld's frameDropPerc, its highest over the frames (modeld.py: FirstOrderFilter(0., 10., 1 / 20), x / (1 + x))."""
  x, alpha, top = 0., (1. / 20) / (10. + 1. / 20), 0.
  for d in dropped:
    x = (1 - alpha) * x + alpha * min(d, 10)
    top = max(top, 100 * x / (1 + x))
  return top


@pytest.mark.parametrize("what_, view, v_ego, signal_near", STANDS + [('crossing', 'stopSweepGo', 1., False)])
@pytest.mark.parametrize("scale", [1., 1.6])
def test_on_modeld_clock_a_stand_drops_no_frame_so_no_stock_lateness_alert(monkeypatch, what_, view, v_ego, signal_near, scale):
  # the stand schedule's side runs may leave a third of a frame of lateness; a frame is dropped only at a whole one. The time
  # budget counts modeld's own time as REST_S, 0.7 ms a frame under the car's: admitted on the budget's debt alone, the side
  # runs filled the stand until it ran a whole frame late (3-19 frames dropped in 90 s at a stop sign, modeldLagging past
  # 3.7 ms). Admitted on the frame's measured lateness (measure_late), at the car's timing (11-06 model times and spreads,
  # its sectors and light read at 1.6x the desktop scale too) no frame of the stand starts more than 25 ms late and none is
  # dropped: neither locationdTemporaryError nor modeldLagging
  det, out = real_clock(monkeypatch, [(2., 'auto', 8., False), (20., view, v_ego, signal_near)], scale=scale)
  late = np.array([o[1] for o in out if o[0] == view][20:])  # from a second in: the drive's lateness paid back
  dropped = [o[2] for o in out]
  print(what_, scale, f"worst {1e3 * late.max():.1f} ms p99 {1e3 * np.percentile(late, 99):.1f}", det.stand)
  assert det.stand and not det.side.disabled
  assert sum(dropped) == 0 and frame_drop_perc(dropped) == 0. and late.max() < 0.025


@pytest.mark.parametrize("view, signal_near", [('stopSweep', False), ('sideLook', True), ('sideLook', False)])
@pytest.mark.parametrize("seed", range(4))
def test_the_first_stands_map_upload_takes_a_paid_back_frame(monkeypatch, seed, view, signal_near):
  # the side maps are recomputed at the first stand of a drive (a stop sweep, or a look at a right on red or a junction) and
  # uploaded a view a step (38.1-38.6 ms each on the car). From the held cycle's odd step after a full pass, 2-4 ms late,
  # the upload's step ended 41 ms late (median) and up to 48 at this cost, a frame from a drop; on a frame paid back as its
  # age says (measure_late) it stays under 40, and the maps are up within 2 s (0.35-0.45 s before)
  det, out = real_clock(monkeypatch, [(2., 'auto', 8., False), (6., view, 0., signal_near)], seed=seed, ready=False)
  stand = [o for o in out if o[0] == view]
  late = np.array([o[1] for o in stand])
  up = [k for k, o in enumerate(stand) if o[3]]
  print(seed, view, signal_near, f"worst {1e3 * late.max():.1f} ms, maps up after {up[0] * CYCLE_S if up else None} s")
  assert sum(o[2] for o in out) == 0 and late.max() < 0.040
  assert up and up[0] * CYCLE_S < 2.
