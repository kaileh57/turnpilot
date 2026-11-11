"""turnpilotd driving the rolling left (unprotected_left.ROLLING_GO) in closed loop: the autonomous left whose oncoming band has
been clear a whole window on the approach goes without stopping at its line; anything else brakes to the line as before.
The Sim, route and road runs of test_turnpilotd_left, the car in the turn's own lane with its lamp lit on the approach."""
import math

import pytest

from openpilot.cereal import log, messaging
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot import map_controls as mc
from openpilot.selfdrive.turnpilot.detector.light_state import LIGHT_MODE_DEFAULT
from openpilot.selfdrive.turnpilot.hooks import TAKE_CONTROL_PROMPTS
from openpilot.selfdrive.turnpilot.scene.tracker import RawBox, RawFrame
from openpilot.selfdrive.turnpilot.scene_policy import ONCOMING_ENTRY_M, STOP_DECEL
from openpilot.selfdrive.turnpilot.tests import test_map_controls as map_tests
from openpilot.selfdrive.turnpilot.tests.synthetic import junction_of
from openpilot.selfdrive.turnpilot.tests.test_camera_lights import detections as light_detections
from openpilot.selfdrive.turnpilot.tests.test_turnpilotd import THREE_LANES_LEFT_TURN, Sim
from openpilot.selfdrive.turnpilot.tests.test_turnpilotd_left import JUNCTION, LINE, STRAIGHT_M, TEST_RANGE_M, V_25, car_frame, raw_box
from openpilot.selfdrive.turnpilot.turn_speed import V_CORNER
from openpilot.selfdrive.turnpilot.unprotected_left import CLEAR_WINDOW_S, HELD_MIN_M

Desire = log.Desire
ROLL_MARGIN_V = 0.8  # m/s under the corner's speed a rolling left may dip (the car follows its caps about a second behind)


def lets_go_rolling(policy_cls=mc.MapControlPolicy) -> bool:
  """map_controls lets go of the unprotected turn's stop on the left's go with the latch not holding (the rolling go's
  release): a car 22 m from the turn on its stop curve, LeftCommand.go from there."""
  car = map_tests.left_car()
  for _ in range(int(60. / DT_MDL)):
    d = car.turn[0] - car.s
    out, _, _ = car.policy_step(map_tests.left_cmd(go=d < 22., reason="Turning"))
    if d < 20.:
      return not out.stops
  return False


def test_map_controls_lets_go_of_a_rolling_lefts_stop():
  # the one thing the rolling go needs of map_controls: without it a rolling go is refused (turnpilot.left_roll_refused)
  # and every left brakes to its line as before (the last test here)
  assert lets_go_rolling()


def box(sim, s, y, motion='unknown'):
  """A car's raw box at road position (s, y left), as the road camera sees it from the car's heading, its motion the
  tracker's verdict."""
  b = raw_box(*car_frame(sim, s - sim.s, y))
  return RawBox(b.camera, b.cls, b.conf, b.bearing_deg, b.span_deg, b.xy, b.near_m, b.box, motion=motion)


def runs(cars=lambda sim: ()):
  """A road camera run every other frame of cars(sim): [(s, y left, motion)]."""
  frame = [0]

  def raw(sim):
    frame[0] += 1
    if frame[0] % 2:
      return None
    return RawFrame([box(sim, s, y, m) for s, y, m in cars(sim) if -5. < s - sim.s < 400.], 'road', sim.t)
  return raw


def roll_sim(cars=lambda sim: (), lamp=True, **kw):
  """A 25 mph left at JUNCTION with one opposing lane, the car in the route's left-turn lane (no lane change wanted), the
  driver's lamp lit when the car asks for it (300 ft out), the full pass's range measured."""
  sim = Sim([(JUNCTION, "turn", "left", (0., 270., 2, 'present', V_25, 1, STRAIGHT_M)), (1500., "turn", "right", junction_of("right"))],
            nav_lanes={0: THREE_LANES_LEFT_TURN}, model_lanes=(0, 2), **kw)
  sim.tp.auto_left, sim.tp.left_range_m = 'auto', TEST_RANGE_M
  sim.go_taps, sim.answers_prompt, sim.raw = False, lamp, runs(cars)
  return sim


def drive(sim, until_s=JUNCTION + 30., limit_s=120.):
  """Engaged to until_s (or limit_s), the model turning at an 8 m radius through the junction once the turn is pulsed.
  [(t, s, v, plan, the left's decision)]."""
  rows = []
  while sim.s < until_s and sim.t < limit_s:
    turning = sim.tp.desires.started and abs(JUNCTION - sim.s) < 12. and sim.v > 0.3
    plan = sim.step(engaged=True, yaw_rate=sim.v / 8. if turning else 0.)
    rows.append((sim.t, sim.s, sim.v, plan, sim.tp.left_now))
  return rows


def stood(rows):
  """The car came to rest short of the junction."""
  return any(v < 0.3 and s < JUNCTION for _, s, v, _, _ in rows)


def rolled(rows):
  return any(d is not None and d.rolled for *_, d in rows)


def test_an_empty_road_is_turned_across_without_a_stop():
  sim = roll_sim()
  rows = drive(sim)
  assert sim.s >= JUNCTION + 30.
  assert rolled(rows) and not stood(rows) and not any(p.shouldStop for *_, p, _ in rows)
  # never under the corner's speed by more than the margin, from the approach through the junction
  slowest = min(v for _, s, v, _, _ in rows if s < JUNCTION + 5.)
  assert slowest >= V_CORNER - ROLL_MARGIN_V, slowest
  # the go where the stop for the line was about to take the car under that speed, the turn pulsed from it, not before
  go = next(i for i, r in enumerate(rows) if r[4].rolled)
  assert 15. < JUNCTION - rows[go][1] < 35., JUNCTION - rows[go][1]
  assert all(p.desire == Desire.none for *_, p, _ in rows[:go - 1])
  pulse = next(i for i, r in enumerate(rows) if r[3].desire == Desire.turnLeft)
  assert pulse - go <= 2 and rows[pulse][1] < LINE
  assert str(rows[go][3].unprotectedLeft.state) == 'go' and rows[go][3].controlPrompt == 'leftGo'
  assert any(str(p.unprotectedLeft.state) == 'committed' for *_, p, _ in rows[go:])
  assert not any(str(p.controlPrompt) in TAKE_CONTROL_PROMPTS or p.turnMissed for *_, p, _ in rows)
  assert sim.tp.go_cert.id == 0 or not sim.tp.go_cert.active  # no lift: the car never stood


def oncoming(v, s_at, t_at):
  """A car driving toward ours at v m/s in the oncoming lane, at road position s_at at time t_at, read closing."""
  return lambda sim: [(s_at - v * (sim.t - t_at), 3.5, 'closing')]


@pytest.mark.parametrize('lead_s', [12., 8., 5., 3.5])
def test_a_closing_car_seen_before_the_go_brings_the_car_to_its_line(lead_s):
  # an oncoming car that reaches the junction lead_s after the car would have rolled into it: in the band on the roll-in,
  # so no rolling go; the car stands at its line and goes once it has passed, a window after
  empty = drive(roll_sim())
  t_go = next(t for t, *_, d in empty if d.rolled)
  cars = oncoming(12., JUNCTION, t_go + lead_s)
  sim = roll_sim(cars)
  rows = drive(sim)
  assert sim.s >= JUNCTION + 30. and not rolled(rows)
  stand = [(t, s) for t, s, v, _, _ in rows if v < 0.3 and s < JUNCTION]
  assert stand and all(LINE - 1.5 < s <= LINE + 0.2 for _, s in stand), stand[:3]
  t_passed = t_go + lead_s + 5. / 12.  # out of the band: ONCOMING_PAST_M short of the maneuver point
  go = next(t for t, _, _, p, _ in rows if p.desire == Desire.turnLeft)
  assert go >= t_passed + CLEAR_WINDOW_S - 0.3, (go, t_passed)
  assert not any(str(p.controlPrompt) in TAKE_CONTROL_PROMPTS for *_, p, _ in rows)


def test_a_car_first_seen_after_the_rolling_go_withdraws_it_short_of_the_oncoming_lanes():
  # nothing in the band until the go; 0.5 s after it a car appears 60 m up the oncoming lane at 12 m/s: the go is withdrawn,
  # the car stops on the comfortable curve short of the oncoming lanes and goes after it has passed
  empty = drive(roll_sim())
  t_go = next(t for t, *_, d in empty if d.rolled)
  sim = roll_sim(lambda sim: [(JUNCTION + 60. - 12. * (sim.t - t_go - 0.5), 3.5, 'closing')] if sim.t >= t_go + 0.5 else [])
  rows = drive(sim)
  assert rolled(rows) and sim.s >= JUNCTION + 30.
  assert any(str(p.controlAlert) == 'oncomingHolding' for *_, p, _ in rows)
  stand = [s for _, s, v, _, _ in rows if v < 0.3 and s < JUNCTION]
  assert stand and max(stand) <= JUNCTION - HELD_MIN_M, max(stand, default=None)
  # braked no harder than the comfortable curve (the Sim's car follows the plan's caps)
  decel = min((b[2] - a[2]) / DT_MDL for a, b in zip(rows, rows[1:], strict=False) if b[1] < JUNCTION)
  assert decel >= -(STOP_DECEL + 0.6), decel
  # its pulse withdrawn with it, a fresh one at the go after the car has passed (5 s to the junction, a window after)
  t_w = next(t for t, *_, p, _ in rows if str(p.controlAlert) == 'oncomingHolding')
  later = [t for t, _, _, p, _ in rows if t > t_w + 0.5 and p.desire == Desire.turnLeft]
  assert later and later[0] >= t_go + 0.5 + 5. + CLEAR_WINDOW_S - 0.5, (later[:1], t_go)
  assert not any(str(p.controlPrompt) in TAKE_CONTROL_PROMPTS for *_, p, _ in rows)


@pytest.mark.parametrize('late_s', [1.2, 1.6, 2.0, 2.6])
def test_a_car_seen_late_in_a_rolling_go_never_stops_the_car_in_the_oncoming_lanes(late_s):
  # whenever in the go a car is first seen: either the go is withdrawn and the car stands short of the oncoming lanes, or it
  # could no longer stop short of them and the turn is finished; never a stand in them
  empty = drive(roll_sim())
  t_go = next(t for t, *_, d in empty if d.rolled)
  sim = roll_sim(lambda sim: [(JUNCTION + 80. - 12. * (sim.t - t_go - late_s), 3.5, 'closing')] if sim.t >= t_go + late_s else [])
  rows = drive(sim)
  assert sim.s >= JUNCTION + 30.
  stand = [s for _, s, v, _, _ in rows if v < 0.3 and JUNCTION - 15. < s < JUNCTION + 8.]
  assert all(s <= JUNCTION - ONCOMING_ENTRY_M for s in stand), max(stand)


def test_a_car_parked_at_the_far_curb_and_a_car_ahead_going_away_dont_stop_the_roll():
  # one lane each way: a car standing 8 m to the left, 30 m past the junction (beyond the oncoming lane's far edge), and a
  # car 40 m ahead in our own direction's lane to the right, going away: neither is oncoming traffic
  sim = roll_sim(lambda sim: [(JUNCTION + 30., 8., 'standing'), (sim.s + 40., -0.5, 'receding')])
  rows = drive(sim)
  assert rolled(rows) and not stood(rows) and sim.s >= JUNCTION + 30.


@pytest.mark.parametrize('motion', ['standing', 'unknown', 'closing'])
def test_a_car_standing_in_the_oncoming_lane_holds_the_left_whatever_its_motion_reads(motion):
  # in the oncoming lane itself (3.5 m left) 30 m past the junction for the whole approach, gone 12 s after the car is at its line
  empty = drive(roll_sim())
  t_go = next(t for t, *_, d in empty if d.rolled)
  sim = roll_sim(lambda sim: [(JUNCTION + 30., 3.5, motion)] if sim.t < t_go + 20. else [])
  rows = drive(sim)
  assert not rolled(rows) and stood(rows)
  assert not any(p.desire == Desire.turnLeft for t, *_, p, _ in rows if t < t_go + 20.)


def signal_roll_sim(state, cars=lambda sim: ()):
  """roll_sim with a signal mapped at the junction's line; state(sim) -> the light an accepted source (the sim oracle) reads
  on every frame, None for nothing read."""
  line = JUNCTION - 8.5
  sim = roll_sim(cars, controls=[(line, 'signal')])
  sim.tp.oracle_ok = True

  def lights(sim):
    st = state(sim)
    msg = messaging.new_message('turnpilotLights')
    msg.turnpilotLights.source = 'simOracle'
    if st is not None:
      msg.turnpilotLights.lights = [{'state': st, 'distance': line - sim.s, 'instance': 'trafficLight 1'}]
    return msg.as_reader().turnpilotLights
  sim.lights = lights
  return sim


def camera_roll_sim(read):
  """roll_sim with a signal mapped at the junction's line and no accepted light source, as on the car: the camera's reader
  (add-only) reading the head 15 m past the junction; read(sim) -> the state of a road run's read made this frame, None for
  no read this frame (turnpilotd carries the last one up to CAMERA_STALE_S)."""
  sim = roll_sim(controls=[(JUNCTION - 8.5, 'signal')])
  sim.tp.oracle_ok, sim.tp.camera_mode = False, LIGHT_MODE_DEFAULT
  none = messaging.new_message('turnpilotLights').as_reader().turnpilotLights

  def lights(sim):
    state = read(sim)
    if state is not None:
      sim.sm.put('turnpilotDetections', light_detections(state, JUNCTION + 15. - sim.s, shadow=False))
    return none
  sim.lights = lights
  return sim


def every(period_s, state, t0=0.):
  """A camera read of state every period_s from t0."""
  last = [-math.inf]

  def read(sim):
    if sim.t >= t0 and sim.t - last[0] >= period_s - 1e-6:
      last[0] = sim.t
      return state(sim) if callable(state) else state
    return None
  return read


def test_the_cameras_green_rolls_a_signal_on_three_road_runs_reading_it_never_on_one_read_carried():
  # read green on every road run of the approach (5 Hz): rolled
  rows = drive(camera_roll_sim(every(0.2, 'green')))
  assert rolled(rows) and not stood(rows)
  t_ask = next(t for t, *_, d in rows if d.roll == 'go')
  # one green read 0.3 s before that, and nothing else read: carried for a second it is still one read, never a steady green
  sim = camera_roll_sim(every(1e9, 'green', t0=t_ask - 0.3))
  rows = drive(sim)
  assert not rolled(rows) and stood(rows) and sim.s >= JUNCTION + 30.
  # two reads are not three
  hits = [t_ask - 1.2, t_ask - 0.2]
  rows = drive(camera_roll_sim(lambda sim: 'green' if any(abs(sim.t - h) < DT_MDL / 2. for h in hits) else None))
  assert not rolled(rows)


def test_a_red_read_between_the_cameras_greens_starts_them_over():
  rows = drive(camera_roll_sim(every(0.2, 'green')))
  t_ask = next(t for t, *_, d in rows if d.roll == 'go')
  # green on every run but one red read 0.4 s before the roll could be asked: no three greens since it by the last point
  seq = every(0.2, lambda sim: 'red' if abs(sim.t - (t_ask - 0.4)) < 0.1 else 'green')
  rows = drive(camera_roll_sim(seq), limit_s=60.)
  assert not rolled(rows)
  assert not any(p.desire == Desire.turnLeft for t, _, _, p, _ in rows if t < t_ask + 1.)


def test_a_signal_nobody_reads_is_never_rolled():
  # a mapped signal with no colour read: the left rolls up to the junction's mouth and is judged standing there, as before
  sim = signal_roll_sim(lambda sim: None)
  rows = drive(sim)
  assert not rolled(rows) and stood(rows)


def test_a_signal_read_green_is_rolled():
  sim = signal_roll_sim(lambda sim: 'green')
  rows = drive(sim)
  assert rolled(rows) and not stood(rows) and sim.s >= JUNCTION + 30.
  assert min(v for _, s, v, _, _ in rows if s < JUNCTION + 5.) >= V_CORNER - ROLL_MARGIN_V


def test_a_signal_read_red_is_never_rolled():
  sim = signal_roll_sim(lambda sim: 'red')
  rows = drive(sim, limit_s=60.)
  assert not rolled(rows) and stood(rows) and sim.s < JUNCTION
  assert not any(p.desire == Desire.turnLeft for *_, p, _ in rows)


def test_a_green_read_only_as_the_car_reaches_the_rolls_last_point_isnt_rolled():
  # nothing read until 0.5 s before a rolling go could last be given, green from there: not CAMERA_GREEN_STEADY_S of green
  # by then, so the car goes on to the junction's mouth and is judged there
  seen = []
  sim = signal_roll_sim(lambda sim: None)
  while sim.s < JUNCTION and sim.t < 60. and (not seen or seen[-1][1] != 'done'):
    sim.step(engaged=True)
    seen.append((sim.t, sim.tp.left_now.roll))
  t_done = seen[-1][0]
  sim = signal_roll_sim(lambda sim: 'green' if sim.t >= t_done - 0.5 else None)
  rows = drive(sim)
  assert not rolled(rows) and sim.s >= JUNCTION + 30.


def test_a_red_read_after_a_rolling_go_stops_the_car():
  # green through the go, red 0.4 s after it: the go is withdrawn while the car can stop, and it stands for the green
  first = drive(signal_roll_sim(lambda sim: 'green'))
  t_go = next(t for t, *_, d in first if d.rolled)
  sim = signal_roll_sim(lambda sim: 'green' if sim.t < t_go + 0.4 or sim.t > t_go + 20. else 'red')
  rows = drive(sim)
  assert rolled(rows) and stood(rows)
  stand = [s for t, s, v, _, _ in rows if v < 0.3 and s < JUNCTION and t < t_go + 20.]
  assert stand and max(stand) <= JUNCTION - ONCOMING_ENTRY_M
  assert not any(p.desire == Desire.turnLeft for t, _, _, p, _ in rows if t_go + 2. < t < t_go + 20.)


def test_no_lamp_no_roll():
  # the driver never lights the lamp: the left stops at its line first (and goes unsignalled after its wait, as before)
  sim = roll_sim(lamp=False)
  rows = drive(sim)
  assert not rolled(rows) and stood(rows)


def test_a_person_in_the_path_is_never_rolled_past():
  # a pedestrian standing in the car's path at the junction's near side: the scene's hold, no rolling go, the car stops
  def objects(sim):
    return [{'trackId': 7, 'cls': 'pedestrian', 'x': JUNCTION - 6. - sim.s, 'y': 0.2, 'vx': 0., 'vy': 0., 'age': 30, 'conf': 0.8,
             'inPath': True}] if 0. < JUNCTION - 6. - sim.s < 60. else []
  sim = roll_sim()
  rows = []
  while sim.s < JUNCTION + 30. and sim.t < 60.:
    plan = sim.step(objects=objects(sim), engaged=True)
    rows.append((sim.t, sim.s, sim.v, plan, sim.tp.left_now))
  assert not rolled(rows) and sim.s < JUNCTION - 6.
  assert not any(p.desire == Desire.turnLeft for *_, p, _ in rows)


def test_the_public_builds_window_from_the_stand_never_rolls(left_window_from_its_stand):
  sim = roll_sim()
  sim.tp.left.window_from_approach = False
  rows = drive(sim)
  assert not rolled(rows) and stood(rows) and sim.s >= JUNCTION + 30.


def test_a_rolling_go_the_lines_stop_doesnt_let_go_is_refused_and_the_left_stops_first(monkeypatch):
  # map_controls keeping its stop through the go (as it does without the release): the go is taken back the next frame, no
  # turn is pulsed on the approach, and the car stands at its line and goes from there as any left
  approach = mc.MapControlPolicy._approach

  def keeps(self, c, controls, v, gas_pressed, gas_lift, out, prompts):
    if c.kind == mc.UNPROTECTED and not c.held:
      saved, self._release_req = getattr(self, '_release_req', None), None
      try:
        return approach(self, c, controls, v, gas_pressed, gas_lift, out, prompts)
      finally:
        self._release_req = saved
    return approach(self, c, controls, v, gas_pressed, gas_lift, out, prompts)
  monkeypatch.setattr(mc.MapControlPolicy, '_approach', keeps)
  sim = roll_sim()
  rows = drive(sim)
  assert not rolled(rows) and stood(rows) and sim.s >= JUNCTION + 30.
  first = next(i for i, r in enumerate(rows) if r[3].desire == Desire.turnLeft)
  assert rows[first - 1][2] < 0.3 and math.isclose(rows[first][1], LINE, abs_tol=1.5)


def test_a_red_or_yellow_head_read_beside_the_greens_is_no_green_to_roll_on(monkeypatch):
  # a red left arrow beside green balls: any red or yellow read at the junction, by any source, and the roll has no green
  tp = camera_roll_sim(lambda sim: None).tp
  monkeypatch.setattr(tp.map_controls, 'signal_known', lambda maneuver_id, distance: True)
  for accepted, camera in ((set(), {'green', 'red'}), ({'green'}, {'red'}), ({'red'}, {'green'}), (set(), {'green', 'yellow'})):
    monkeypatch.setattr(tp.map_controls, 'junction_lights', lambda distance, a=accepted, c=camera: (a, c))
    assert tp.left_roll_light(1, 50.)[0] == 'unread', (accepted, camera)
  monkeypatch.setattr(tp.map_controls, 'junction_lights', lambda distance: ({'green'}, set()))
  assert tp.left_roll_light(1, 50.)[0] == 'green'
