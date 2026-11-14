"""The side look (turnpilotPlan.detectorView sideLook, modeld_detector SIDE_LOOK): turnpilotd asks for the side views at walking
pace while a curb-side turn's signal stops or holds the car for its right-on-red look, and while a turn at a junction with no
mapped control may wait for cross traffic; never over the unprotected left's view or a stop sweep. What the side views deliver
at a stand (modeld's stand schedule: the road pass every STAND_ROAD_DUE_S, the full wide pass and each side every 0.40-0.47 s)
is fed here through the real SceneTracker, its coast set from turnpilotd's sweeping as main() sets it, to test_turnpilotd's
Sim: a car boxed only by a side view is a track a hit every 0.4 s or so, published at its second hit, gone SWEEP_COAST_S after
its last. The consumers (map_controls.right_red_blocked, scene_policy.cross_stops, stop_go's go) see it as crossing traffic."""
import math

import pytest

from openpilot.cereal import messaging
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot import map_controls as mc
from openpilot.selfdrive.turnpilot.detector.decoder import CAR
from openpilot.selfdrive.turnpilot.junction_watch import detector_view
from openpilot.selfdrive.turnpilot.scene.tests.test_tracker import CAMS, calibration, car_state
from openpilot.selfdrive.turnpilot.scene.tests.test_tracker_views import run as detector_run
from openpilot.selfdrive.turnpilot.scene.tracker import MAX_COAST_S, SWEEP_COAST_S, SceneTracker
from openpilot.selfdrive.turnpilot.tests import test_turnpilotd_cross_traffic as ct
from openpilot.selfdrive.turnpilot.tests.shipped import shipped
from openpilot.selfdrive.turnpilot.tests.synthetic import junction_of
from openpilot.selfdrive.turnpilot.tests.test_turnpilotd import Sim
from openpilot.selfdrive.turnpilot.tests.test_turnpilotd import T_LEFT
from openpilot.selfdrive.turnpilot.tests.test_turnpilotd_light_rails import camera
from openpilot.selfdrive.turnpilot.scene_policy import CROSS_REASON

STAND_V = 2.0          # m/s: modeld's SIDE_VIEW_V_MAX, the stand schedule's speed
STARVED_S = 0.3        # modeld's STAND_STARVED_S: the schedule starts once the side views have gone this long without a frame
ROAD_DUE_S = 0.2       # modeld's STAND_ROAD_DUE_S
SIDE_PERIODS = (0.40, 0.45, 0.47)  # s between one view's runs at a stand, in turn (the detector's 2.1-2.5 Hz)
FORWARD_DEG = 60.      # the forward passes box out to this bearing either side; the side views from there...
SIDE_DEG = 85.         # ...to this


class Feed:
  """modeld's runs of cars(sim) [(x ahead, y left[, forward[, class]]) m, car frame; a car by default] through a real
  SceneTracker, as turnpilotd's main() takes them:
  at or under STAND_V under the last plan's sideLook (STARVED_S on) the stand schedule, one run a frame; otherwise the road
  and the full wide pass on alternate frames. forward: whether the forward passes box the cars too (False: a car only a side
  view sees). Returns the scene's objects (TrackedObject fields) and whether it is available."""
  def __init__(self, cars, forward: bool = True):
    self.cars, self.forward = cars, forward
    self.trk, self.cal = SceneTracker(CAMS, 'mici'), calibration()
    self.frame, self.look_s, self.due, self.turn = 0, 0., {}, {}
    self.ran: list = []  # (t, view) of each run
    self.miss = lambda view, n: False  # (view, its run count) -> that run boxes nothing (a missed detection)
    self.coast = None  # the tracker's coast instead of main()'s (SWEEP_COAST_S while turnpilotd sweeps or looks)

  def sees(self, view: str, x: float, y: float, forward: bool) -> bool:
    b = math.degrees(math.atan2(y, x))
    if view in ('road', 'full'):
      return forward and abs(b) <= FORWARD_DEG
    return FORWARD_DEG < (b if view == 'sideLeft' else -b) <= SIDE_DEG

  def view_now(self, sim) -> str:
    look = bool(sim.trace) and str(sim.trace[-1][2].detectorView) == 'sideLook' and sim.v <= STAND_V
    self.look_s = self.look_s + DT_MDL if look else 0.
    if self.look_s < STARVED_S:
      self.due = {}
      return 'road' if self.frame % 2 == 0 else 'full'
    for view in ('road', 'full', 'sideLeft', 'sideRight'):
      if sim.t >= self.due.get(view, 0.) - 1e-6:
        k = self.turn.get(view, 0)
        self.turn[view] = k + 1
        self.due[view] = sim.t + (ROAD_DUE_S if view == 'road' else SIDE_PERIODS[k % len(SIDE_PERIODS)])
        return view
    return ''

  def __call__(self, sim):
    self.frame += 1
    view = self.view_now(sim)
    runs = []
    if view:
      cam = 'road' if view == 'road' else 'wide'
      cars = [(c[0], c[1], c[2] if len(c) > 2 else self.forward, c[3] if len(c) > 3 else CAR) for c in self.cars(sim)]
      n = sum(1 for _, v in self.ran if v == view)
      boxed = [] if self.miss(view, n) else [(cls, x, y, 0.7) for x, y, fwd, cls in cars if x > 1. and self.sees(view, x, y, fwd)]
      runs = [detector_run(self.trk, cam, 'full' if view == 'road' else view, sim.t, boxed, self.frame)]
      self.ran.append((sim.t, view))
    self.trk.coast_s = self.coast or (SWEEP_COAST_S if sim.tp.sweeping else MAX_COAST_S)
    self.trk.update_runs(runs, self.cal, car_state(sim.v), now=sim.t)
    sim.tp.observe_runs(self.trk)  # the runs and tracks of this tick, as main() hands them over
    msg = messaging.new_message('turnpilotScene')
    self.trk.fill_scene(msg.turnpilotScene)
    scene = msg.as_reader().turnpilotScene
    keep = ('trackId', 'cls', 'conf', 'x', 'y', 'vx', 'vy', 'age', 'inPath', 'camera', 'view', 'sideTrusted')
    return [{k: getattr(o, k) for k in keep} for o in scene.objects], scene.available


def step(sim, feed, **kw):
  objects, available = feed(sim)
  return sim.step(objects, available=available, engaged=True, **kw)


def test_the_side_look_is_asked_after_the_lefts_view_and_a_stop_sweep():
  assert detector_view('auto', False, True) == 'sideLook' and detector_view('none', False, False) == 'auto'
  assert detector_view('road', False, True) == 'road' and detector_view('roadAndWide', True, True) == 'roadAndWide'
  assert detector_view('auto', True, True) == 'stopSweep'


SIGNAL, JUNCTION, HEAD = 191., 200., 215.  # a right at a mapped signal: its line, the junction, the head the camera reads
V_CROSS = 9.  # m/s, 20 mph: a car on the cross road, in its near lane from the left (the lane the right turns into)


def right_on_red(cars, unread_at=5., first='red', feed_kw=None):
  """A right at a mapped signal stood at its read red, the red then unread (boxed, no colour) unread_at s into the stand,
  cars(sim, t since the red went unread or None) the cars the detector boxes. Returns (sim, feed, [(t since unread, plan)],
  the detector views asked at the stand)."""
  sim = shipped(Sim([(JUNCTION, "turn", "right", junction_of("right")), (1500., "turn", "left", junction_of("left"))],
                    controls=[(SIGNAL, 'signal')]))
  sim.answers_prompt = True
  read, st = {'state': first}, {'t': None}
  camera(sim, lambda s: [(read['state'], HEAD)], reach_m=400.)
  feed = Feed(lambda s: cars(s, None if st['t'] is None else s.t - st['t']))
  for k, v in (feed_kw or {}).items():
    setattr(feed, k, v)
  while not (sim.v < 0.05 and sim.s > SIGNAL - 20.):
    step(sim, feed)
    assert sim.t < 90.
  views, sweeping = set(), []
  for _ in range(int(unread_at / DT_MDL)):
    views.add(str(step(sim, feed).detectorView))
    sweeping.append(sim.tp.sweeping)
  assert all(sweeping[int(1. / DT_MDL):]) or 'sideLook' not in views or first != 'red'  # coasting through a missed side pass (SWEEP_COAST_S)
  read['state'], st['t'] = 'unknown', sim.t
  rows = []
  for _ in range(int(25. / DT_MDL)):
    rows.append((sim.t - st['t'], step(sim, feed)))
  return sim, feed, rows, views


def cross_from_left(appear_s: float, y0: float = 45.):
  """A car on the cross road coming from the left at V_CROSS, y0 m up it appear_s s after the red went unread, through our
  path and on; boxed only by a side view (beyond the forward passes' bearings)."""
  def cars(sim, t):
    if t is None or t < appear_s:
      return []
    y = y0 - V_CROSS * (t - appear_s)
    return [(JUNCTION - 1.8 - sim.s, y, False)] if y > -15. else []
  return cars, appear_s + y0 / V_CROSS  # ...and when it crosses our path


def went(sim, rows):
  """s after the red went unread that the car first moved."""
  return next((t for (t, _), (_, v, _) in zip(rows, sim.trace[-len(rows):], strict=True) if v > 0.05), None)


@pytest.mark.parametrize('appear_s', [0., 1.])
def test_a_right_on_red_asks_for_the_side_look_and_a_car_only_a_side_view_sees_holds_it_until_it_has_passed(appear_s):
  # without the look (until 2026-11-14) the side views never ran at a right on red: the right went 2.55 s after the red went
  # unread, the car 22 m up the cross road 2.5 s from our path, into the lane it was coming down
  cars, crosses = cross_from_left(appear_s)
  sim, feed, rows, views = right_on_red(cars)
  assert views == {'sideLook'}
  assert any(v.startswith('side') for _, v in feed.ran)
  assert any(p.reason == mc.RIGHT_RED_CROSS_REASON for _, p in rows)
  t_go = went(sim, rows)
  assert t_go is not None and crosses <= t_go < crosses + 3., (t_go, crosses)


def test_a_car_standing_where_only_a_side_view_sees_it_holds_a_right_on_red_only_until_its_track_is_trusted():
  # a car waiting on the cross road 20 m to the left: a new track is no telling which way it moves (right_red_blocked's
  # 'new') until its velocity is trusted: by SIDE_TRUST_HITS over SIDE_TRUST_SPAN_S of side hits (until 2026-11-14 the
  # 25 hits of VEL_TRUST_HITS, 12.3 s at the side views' cadence); then it stands, and the right goes
  sim, _, rows, _ = right_on_red(lambda s, t: [(JUNCTION - 1.8 - s.s, 20., False)] if t is not None else [])
  t_go = went(sim, rows)
  assert t_go is not None and t_go < 6., t_go


def crawl_turn(side: str, cars):
  """A turn out of a T's stem at a junction with no mapped control, the car crawling up to it (cruise 1.8 m/s: at walking
  pace from far out, so its cross-traffic hold's look is asked for), cars(sim) boxed as Feed takes them. [(t, s, v, plan)]."""
  sim = shipped(Sim([(ct.JUNCTION, "turn", side, T_LEFT if side == 'left' else ct.T_RIGHT), (1900., "turn", "right", junction_of("right"))],
                    controls_status='ok', cruise=1.8))
  sim.answers_prompt = True
  feed, rows = Feed(cars), []
  while sim.s < ct.JUNCTION + 20. and sim.t < 400.:
    plan = step(sim, feed)
    rows.append((sim.t, sim.s, sim.v, plan))
  return rows


@pytest.mark.parametrize('side', ['left', 'right'])
def test_a_turn_crawling_up_to_an_uncontrolled_junction_is_held_for_a_car_only_a_side_view_sees(side):
  # without the look the car crawled into the through road, its bumper 3 m past the near edge as the car crossed
  alone = crawl_turn(side, lambda sim: [])
  meet = next(t for t, s, _, _ in alone if s + ct.BUMPER_M >= ct.LANE['left']) + 0.4

  def cars(sim):
    t = sim.t - meet
    return [(ct.LANE['left'] - sim.s, -V_CROSS * t, False)] if -6. <= t <= 2. else []
  rows = crawl_turn(side, cars)
  assert ct.bumper_as_it_crosses(rows, meet, V_CROSS) < ct.JUNCTION - 3.6
  # held until it has crossed, its velocity trusted by hits over time (scene/tracker.py side_trusted): until 2026-11-14 a
  # side-only car never earned VEL_TRUST_HITS and the hold let go 1.25 s before it crossed, inside CROSS_Y_NEAR_M
  assert max(t for t, _, _, p in rows if str(p.reason) == CROSS_REASON) >= meet
  assert any(str(p.reason) == CROSS_REASON for _, _, _, p in rows) and any(str(p.detectorView) == 'sideLook' for *_, p in rows)
  t_go = next(t for t, _, v, _ in rows if t > meet and v > 1.)
  assert t_go < meet + 5.  # its crossing, then CROSS_CLEAR_S of the trusted track's tail
  assert rows[-1][1] >= ct.JUNCTION + 20.


class StandCadence:
  """The stand schedule's views at a stop sign, one run a frame: the road pass every 4 frames, the full wide pass and each side
  view every periods[k] frames in turn (8: 0.40 s, 9: 0.45 s = DET_FRESH_S, 10: 0.50 s)."""
  def __init__(self, periods):
    self.periods, self.due, self.turn, self.k, self.last = periods, {}, {}, 0, {}

  def view(self) -> str | None:
    self.k += 1
    for v in ('road', 'full', 'sideLeft', 'sideRight'):
      if self.k >= self.due.get(v, 0):
        n = self.turn.get(v, 0)
        self.turn[v] = n + 1
        self.due[v] = self.k + (4 if v == 'road' else self.periods[n % len(self.periods)])
        return v
    return None


def stand_step(self, *args, **kw):
  """test_turnpilotd_stop.StopSim.step at the stand cadence (self.cadence): no run on an idle frame, and a track on a view
  carries the camera time of that view's last run."""
  from openpilot.selfdrive.turnpilot.scene.tracker import RawFrame
  from openpilot.selfdrive.turnpilot.tests.test_turnpilotd_stop import FakeTracker, StopSim
  from openpilot.selfdrive.turnpilot.tests.test_visibility import sectors
  view = self.cadence.view()
  self.t_cam += DT_MDL
  runs = []
  if view is not None and (not view.startswith('side') or self.side_status(self) in ('running', 'noRoom')):
    side = view.startswith('side')
    raw = RawFrame(list(self.boxes(self).get(view, [])), 'road' if view == 'road' else 'wide', self.t_cam, view if side else 'full',
                   lights=() if side else tuple(self.heads(self)))
    runs = [(view, self.t_cam, raw, sectors(view) if side else None)]
    self.cadence.last[view] = self.t_cam
  tracks = []
  for trk in self.tracks(self):
    if trk.view in self.cadence.last:
      trk.t_seen = self.cadence.last[trk.view]
      tracks.append(trk)
  self.tp.side_status = self.side_status(self)
  self.tp.observe_runs(FakeTracker(runs, tracks))
  kw.setdefault('engaged', self.engaged)
  plan = super(StopSim, self).step(*args, **kw)
  if str(plan.stopGo.state) == 'committed' and self.committed_s is None:
    self.committed_s = self.s
  return plan


def stop_at_cadence(tmp_path, mode: str, periods, car: bool):
  """A stop sign (test_turnpilotd_stop's all-way under auto, test_turnpilotd_vision's under vision) at the stand cadence,
  standing, then 24 s on: a car crossing from the left (test_turnpilotd_stop_night.crossing, on its side view) or none.
  Returns (s from arrival to the go or None, s from standing to the go or None, s from standing until the car is gone or
  None, the take-control prompts)."""
  from openpilot.selfdrive.turnpilot.tests.test_turnpilotd_stop import StopSim, go_frame, run_to_line
  from openpilot.selfdrive.turnpilot.tests.test_turnpilotd_stop_night import crossing, handed_back
  from openpilot.selfdrive.turnpilot.tests.test_turnpilotd_vision import arrived, vsim
  tmp_path.mkdir(parents=True, exist_ok=True)
  sim = vsim(tmp_path) if mode == 'vision' else StopSim(tmp_path)
  sim.cadence = StandCadence(periods)
  sim.step = stand_step.__get__(sim)
  run_to_line(sim)
  n0, t0 = len(sim.plans()), sim.t_cam
  gone = None
  if car:
    sim.tracks, gone = crossing()
  sim.run(24.)
  plans = sim.plans()
  k, a = go_frame(sim), arrived(plans)
  return ((k - a) * DT_MDL if k is not None else None, (k - n0 + 1) * DT_MDL if k is not None else None,
          gone() - t0 if gone is not None else None, handed_back(plans))


@pytest.mark.parametrize('periods', [(8,), (8, 9, 9, 9, 10), (9, 9, 10), (9, 10), (12,), (16,)],
                         ids=['0.40', 'mixed', 'slow', 'everyOther0.50', '0.60', '0.80'])
@pytest.mark.parametrize('mode', ['auto', 'vision'])
def test_a_stop_signs_go_decides_on_side_views_at_the_stand_cadence(tmp_path, mode, periods):
  # each side view every 0.40-0.50 s (modeld's stand schedule, 2.1-2.5 Hz), not the 5 Hz of the held cycle the decider's
  # tests feed: an empty junction goes one window after the car arrives, as at 5 Hz; a car coming on a side view holds the
  # go until it has gone, never released by the gaps between its view's runs. Each wide camera view is fresh within 1.5
  # times its own cadence (visibility.fresh_s, to 0.9 s) and the window grows by that age past DET_FRESH_S: until
  # 2026-11-14 every other pass 0.50 s apart (2.1 Hz) never went, saying nothing but "views catching up"
  from openpilot.selfdrive.turnpilot import stop_rules as sr
  from openpilot.selfdrive.turnpilot.visibility import CADENCE_FRESH_MAX_S, CADENCE_SLACK
  gaps = sorted(periods * 5)
  extra = min(max(sr.DET_FRESH_S, CADENCE_SLACK * gaps[len(gaps) // 2] * DT_MDL), CADENCE_FRESH_MAX_S) - sr.DET_FRESH_S
  window = (sr.VISION_CLEAR_S if mode == 'vision' else sr.CLEAR_S) + extra
  after_arrival, _, _, handed = stop_at_cadence(tmp_path / 'empty', mode, periods, False)
  assert after_arrival == pytest.approx(window, abs=0.16) and not handed, (after_arrival, window)
  _, go_s, gone_s, handed = stop_at_cadence(tmp_path / 'car', mode, periods, True)
  assert go_s is not None and gone_s + window - 0.11 <= go_s < gone_s + window + 2. and not handed, (go_s, gone_s)


@pytest.mark.parametrize('appear_s', [0., 1.5, 1.9, 2.3])
def test_a_walker_in_the_rights_crosswalk_only_the_wide_pass_sees_still_holds_the_car_under_the_side_look(appear_s):
  # the person rail at the stand schedule's slower wide pass (about 2.3 a second): a walker 6 m to the right at the turn point,
  # seen only by the full wide pass, appearing as late as 0.2 s before the right-on-red go. Seen 0.7 s or more before it, the
  # look waits for them (RIGHT_RED_PERSON_REASON); later, the go is let go and the scene's crosswalk hold stands the car at
  # once, short of them
  from openpilot.selfdrive.turnpilot.detector.decoder import PERSON
  sim, _, rows, _ = right_on_red(lambda s, t: [(JUNCTION - s.s, -6., True, PERSON)] if t is not None and t >= appear_s else [])
  after = [(s, v) for (t, _), (s, v, _) in zip(rows, sim.trace[-len(rows):], strict=True) if t >= appear_s]
  assert max(v for _, v in after) < 0.5 and max(s for s, _ in after) < JUNCTION - 5.


def test_a_right_at_an_unmapped_red_asks_for_the_side_look_at_its_stand():
  # the camera's own signal at a junction the map has none for (map_controls.camera_signal) is a right's look too
  from openpilot.selfdrive.turnpilot.tests.test_turnpilotd_light_rails import right_at_an_unmapped_red
  sim, line, _ = right_at_an_unmapped_red(0.)
  while not (sim.v < 0.05 and sim.s > line - 20.):
    sim.step(engaged=True)
    assert sim.t < 90.
  plans = [sim.step(engaged=True) for _ in range(int(2. / DT_MDL))]
  assert all(str(p.detectorView) == 'sideLook' for p in plans) and sim.tp.sweeping
  # ...asked from RIGHT_LOOK_M short of its line, never in a queue further back
  views = [(line - s, str(p.detectorView)) for s, v, p in sim.trace if v <= STAND_V and line - s > mc.RIGHT_LOOK_M + 1.]
  assert all(view == 'auto' for _, view in views)


def test_a_side_view_missing_one_pass_in_three_still_holds_a_right_on_red_for_its_car():
  # the coast through a missed pass (turnpilotd's sweeping under the look: SWEEP_COAST_S): with MAX_COAST_S the track died
  # between hits 0.8-0.94 s apart and the right went 0.2 s before the car crossed its path
  cars, crosses = cross_from_left(0.)
  sim, _, rows, _ = right_on_red(cars, feed_kw={'miss': lambda view, n: view.startswith('side') and n % 3 == 2})
  t_go = went(sim, rows)
  assert t_go is not None and crosses <= t_go < crosses + 3., (t_go, crosses)


def test_a_car_standing_where_only_a_side_view_sees_it_is_trusted_standing_never_closing_from_box_jitter(monkeypatch):
  # 1 px of jitter on every box edge, a hit every 0.40-0.47 s: trusted by hits over time, and never faster than
  # ONCOMING_MIN_SPEED (cross_stops' closing) nor RIGHT_RED_CROSS_V (the look's) once trusted
  import numpy as np
  from openpilot.selfdrive.turnpilot.scene import tracker as tr
  from openpilot.selfdrive.turnpilot.scene.tests import test_tracker_views as tv
  from openpilot.selfdrive.turnpilot.scene_policy import ONCOMING_MIN_SPEED
  rng, box = np.random.default_rng(3), tv.box
  monkeypatch.setattr(tv, 'box', lambda *a: [b + rng.normal(0., 1.) for b in box(*a)])
  trk, cal = SceneTracker(CAMS, 'mici'), calibration()
  trk.coast_s = SWEEP_COAST_S
  t, speeds = 1., []
  for k in range(40):
    t += SIDE_PERIODS[k % len(SIDE_PERIODS)]
    tracks = trk.update(tv.run(trk, 'wide', 'sideLeft', t, [(CAR, 7., 20., 0.7)], k + 1), cal, car_state(), now=t)
    assert len(tracks) <= 1
    if tracks and tr.side_trusted(tracks[0]):
      speeds.append(float(np.hypot(*tracks[0].x[2:])))
  assert len(speeds) > 25 and max(speeds) < min(ONCOMING_MIN_SPEED, mc.RIGHT_RED_CROSS_V), max(speeds)


@pytest.mark.parametrize('miss', ['oneInThree', 'twoInARow'])
def test_a_right_on_red_counts_its_clear_in_side_runs_so_missed_passes_still_hold_it(miss):
  # RIGHT_RED_SIDE_RUNS runs in a row of each side view clear of the look's lanes as well as RIGHT_RED_CLEAR_S: until
  # 2026-11-14 two passes missed in a row let the right go with the car 20 m up, 2.2 s from our path
  pattern = {'oneInThree': lambda view, n: view.startswith('side') and n % 3 == 2,
             'twoInARow': lambda view, n: view.startswith('side') and n % 5 in (3, 4)}[miss]
  cars, crosses = cross_from_left(0.)
  sim, _, rows, _ = right_on_red(cars, feed_kw={'miss': pattern})
  t_go = went(sim, rows)
  assert t_go is not None and crosses <= t_go < crosses + 3., (t_go, crosses)


def test_a_car_first_boxed_just_after_the_stand_holds_a_right_at_an_unread_light():
  # boxed and unread from the approach: the right goes 1.2 s into its stand; a car first boxed 45 m up 0.5 s in was not
  # yet a published track at the go (until 2026-11-14), and now its box holds the side run count
  cars, crosses = cross_from_left(0.5)
  sim, _, rows, _ = right_on_red(cars, unread_at=0.05, first='unknown')
  t_go = went(sim, rows)
  assert t_go is not None and crosses <= t_go, (t_go, crosses)


def test_an_empty_junction_waits_no_longer_for_the_side_run_count():
  sim, _, rows, _ = right_on_red(lambda s, t: [])
  assert went(sim, rows) == pytest.approx(2.55, abs=0.06)  # RIGHT_RED_UNREAD_S and its reads, as before the count
