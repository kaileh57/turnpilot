"""The lights' rail through turnpilotd on the shipped settings (the camera's reader add-only, no oracle): a governing red that
is read always stops the car. The Sim's car follows the plan's caps and has no driving model, so a stop here is TurnPilot's
own."""
import pytest

from openpilot.cereal import log, messaging
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot import map_controls as mc
from openpilot.selfdrive.turnpilot.hooks import plan_accel
from openpilot.selfdrive.turnpilot.tests.shipped import shipped
from openpilot.selfdrive.turnpilot.tests.synthetic import junction_of
from openpilot.selfdrive.turnpilot.tests.test_turnpilotd import Sim

NONE = messaging.new_message('turnpilotLights').as_reader().turnpilotLights  # no oracle: the car's own source alone
THROUGH = [(1000., "turn", "right"), (2000., "turn", "left")]   # the signal at 200 m is gone straight through
RIGHT = [(200., "turn", "right"), (1500., "turn", "left")]      # a curb-side turn at the junction at 200 m


def detections(heads, s_car: float):
  """A road run's turnpilotDetections: heads [(state, road position s)], all governing."""
  msg = messaging.new_message('turnpilotDetections')
  td = msg.turnpilotDetections
  td.camera, td.lightStatus = 'road', 'read'
  td.lights.source, td.lights.shadow = 'camera', False
  td.lights.lights = [{'state': st, 'distance': s - s_car, 'governs': True, 'votes': 4, 'conf': 0.7} for st, s in heads]
  return msg.as_reader().turnpilotDetections


def camera(sim, heads, scale: float = 1., reach_m: float = 140.):
  """sim with the camera's reader on every other frame (a road run): heads(sim) its [(state, road position s)], each read
  from reach_m off, its range scale times long."""
  def lights(s):
    if int(round(s.t / DT_MDL)) % 2 == 0:
      seen = [(st, s.s + scale * (at - s.s)) for st, at in (heads(s) if callable(heads) else heads) if 2. < at - s.s < reach_m]
      s.sm.put('turnpilotDetections', detections(seen, s.s))
    return NONE
  sim.lights = lights
  return sim


def run_to(sim, s_end: float, limit_s: float = 60.):
  """Engaged until s_end or limit_s. Returns (any plan with shouldStop, the lowest speed short of s_end)."""
  plans = []
  while sim.s < s_end and sim.t < limit_s:
    plans.append(sim.step(engaged=True))
  return any(p.shouldStop for p in plans), min(v for s, v, _ in sim.trace if s < s_end)


def crossing_speed(sim, line: float, seconds: float = 45.):
  """The speed the car is 4 m past line at, None if it never gets there, driving engaged."""
  for _ in range(int(seconds / DT_MDL)):
    s0 = sim.s
    sim.step(engaged=True)
    if s0 <= line + 4. < sim.s:
      return sim.v
  return None


@pytest.mark.parametrize("heads", [[('red', 225.)], [('red', 225.), ('green', 230.)], [('red', 225.), ('red', 227.), ('green', 230.)]],
                         ids=["redAlone", "redBesideGreen", "twoRedsBesideGreen"])
def test_a_read_red_at_a_mapped_signal_stops_the_car_whatever_reads_green_beside_it(heads):
  sim = camera(shipped(Sim(THROUGH, controls=[(200., 'signal')])), heads, reach_m=400.)
  stopped, v_min = run_to(sim, 215.)
  assert stopped and v_min < 0.3, f"no stop in the plan: lowest speed {v_min:.1f} m/s, through the line at s={sim.s:.0f}"


@pytest.mark.parametrize("heads", [[('red', 215.)], [('red', 215.), ('green', 220.)]], ids=["redAlone", "redBesideGreen"])
def test_a_right_at_a_mapped_signal_never_goes_while_a_red_is_read(heads):
  sim = camera(shipped(Sim(RIGHT, controls=[(192., 'signal')])), heads, reach_m=400.)
  stopped, v_min = run_to(sim, 205., limit_s=45.)
  assert stopped and v_min < 0.3 and sim.s < 196., f"went on to s={sim.s:.0f} with the red read on every run"


@pytest.mark.parametrize("mapped", [True, False], ids=["mappedSignal", "unmappedJunction"])
def test_a_right_at_a_signal_whose_heads_are_boxed_and_unread_stops_at_the_line(mapped):
  # a right flows only on a read green wherever a light is boxed at its junction, the map's signal or not
  sim = camera(shipped(Sim(RIGHT, controls=[(192., 'signal')] if mapped else [])), [('unknown', 215.)], reach_m=400.)
  stopped, v_min = run_to(sim, 205., limit_s=60.)
  assert stopped and v_min < 0.3, f"no stop in the plan: lowest speed {v_min:.1f} m/s before the junction"


FAR = [(2600., "turn", "right", junction_of("right"))]


@pytest.mark.parametrize("scale", [1.0, 1.4])
def test_a_red_whose_head_ranges_long_still_stops_the_car_at_its_mapped_signal(scale):
  # a far-side head 24 m past the stop line, read red from 100 m off, its range 1.4 times long (a head hung lower than the
  # reader assumes): past the line's window as read, and still the line's red
  line, head = 291., 315.
  sim = camera(shipped(Sim(FAR, controls=[(line, 'signal')])), [('red', head)], scale=scale, reach_m=100.)
  v = crossing_speed(sim, line)
  assert v is None and sim.v < 0.1, f"read red from 100 m out, crossed 4 m past the line at {v} m/s"


def right_at_an_unmapped_red(invalid_s: float, junction: float = 300.):
  """A right at a junction the map has no signal for, the camera reading its red; navd's message invalid for invalid_s from
  60 m before the junction, its content as it was."""
  state = {'light': 'red'}
  sim = camera(shipped(Sim([(junction, "turn", "right", junction_of("right")), (1900., "turn", "left", junction_of("left"))])),
               lambda s: [(state['light'], junction + 15.)])
  sim.answers_prompt = True
  gap = {}

  def nav(s, instr):
    if invalid_s and 't0' not in gap and junction - s.s < 60.:
      gap['t0'] = s.t
    return instr, not ('t0' in gap and s.t - gap['t0'] < invalid_s)
  sim.nav_filter = nav
  return sim, junction - mc.CONTROL_LINE_M, state


@pytest.mark.parametrize("invalid_s", [0., 2.5])
def test_a_nav_gap_never_lets_the_car_through_the_camera_red_it_reads(invalid_s):
  # the route gone for 2.5 s mid-approach, and the turn's corner cap with it: the camera's stop for the red stays in the
  # plan (its own curve from a few frames on, as it engages), and the car stands at the junction's line
  sim, line, state = right_at_an_unmapped_red(invalid_s)
  v = crossing_speed(sim, line, 40.)
  assert v is None and sim.v < 0.1 and sim.s < line + 4., f"crossed 4 m past the junction's line at {v} m/s"
  gap = [p for _, _, p in sim.trace if not p.routeValid]
  assert bool(gap) == bool(invalid_s) and all(p.speedCap < 10. and p.reason.startswith("Red light") for p in gap[5:])
  # ...and with the red gone unread at the stand the right goes by its own procedure, never left standing for good
  state['light'] = 'unknown'
  t0 = sim.t
  while sim.s < line + 4. and sim.t - t0 < 30.:
    sim.step(engaged=True)
  assert sim.s >= line + 4.


def right_on_red_go(signal: float, junction: float, head: float):
  """A right at a mapped signal stood at its red, the red then boxed with no colour read until its right-on-red go. Returns
  (sim, what the head reads from now on (a dict), the time of its go)."""
  sim = shipped(Sim([(junction, "turn", "right", junction_of("right")), (1500., "turn", "left", junction_of("left"))],
                    controls=[(signal, 'signal')]))
  sim.answers_prompt = True
  read = {'state': 'red'}
  camera(sim, lambda s: [(read['state'], head)], reach_m=400.)
  while not (sim.v < 0.05 and sim.s > signal - 20.):
    sim.step(engaged=True)
    assert sim.t < 90.
  assert signal - 3. < sim.s <= signal + 0.5
  for _ in range(int(5. / DT_MDL)):
    plan = sim.step(engaged=True)
  assert plan.shouldStop and plan.reason == mc.RIGHT_RED_WAIT_REASON
  read['state'] = 'unknown'
  for _ in range(int(25. / DT_MDL)):
    plan = sim.step(engaged=True)
    if not plan.shouldStop and sim.v > 0.05:
      return sim, read, sim.t
  raise AssertionError("never let go on the unread light")


def test_a_red_read_again_just_after_the_right_on_red_go_stops_the_car_short_of_the_junction():
  signal, junction = 191., 200.
  sim, read, t_go = right_on_red_go(signal, junction, 215.)
  while sim.t - t_go < 0.3:
    sim.step(engaged=True)
  read['state'] = 'red'  # read red again: the car just rolling, short of its line
  s_r, v_r = sim.s, sim.v
  assert s_r < signal and v_r < 1.5
  turned = []
  for _ in range(int(25. / DT_MDL)):
    plan = sim.step(engaged=True)
    turned.append(plan.desire in (log.Desire.turnLeft, log.Desire.turnRight))
  assert sim.s <= signal + mc.RED_PAST_M and sim.v < 0.1 and plan.shouldStop and not any(turned), \
    f"read red again at {s_r - signal:+.1f} m of the line at {v_r:.1f} m/s, the car went on to {sim.s - signal:+.1f} m"
  # the red gone unread again: the right goes on its procedure once more
  read['state'] = 'unknown'
  t0 = sim.t
  while sim.s < junction and sim.t - t0 < 30.:
    sim.step(engaged=True)
  assert sim.s >= junction


@pytest.mark.parametrize('past_m', [0.5, 1.0, 1.5])
def test_a_red_read_again_just_past_the_line_after_the_right_on_red_go_stops_the_car(past_m):
  # read red again past the line at about 2 m/s (1.5 m: 2.1 m/s, over COMMIT_SPEED): a firm stop still ends inside
  # RED_PAST_M, short of the first cross lane, so the right stops (the matrix: 1.0-1.7 m past at 2.1-2.2 m/s went on)
  signal, junction = 191., 200.
  sim, read, t_go = right_on_red_go(signal, junction, 215.)
  while sim.s < signal + past_m:
    sim.step(engaged=True)
    assert sim.t - t_go < 10.
  read['state'] = 'red'
  s_r, v_r = sim.s, sim.v
  assert mc.red_stop_fits(v_r, signal - s_r), (s_r - signal, v_r)
  for _ in range(int(15. / DT_MDL)):
    plan = sim.step(engaged=True)
  assert sim.s <= signal + mc.RED_PAST_M and sim.v < 0.1 and plan.shouldStop, \
    f"read red again at {s_r - signal:+.1f} m of the line at {v_r:.1f} m/s, the car went on to {sim.s - signal:+.1f} m"


def test_a_red_read_again_once_the_right_is_into_its_turn_changes_nothing():
  signal, junction = 191., 200.
  sim, read, t_go = right_on_red_go(signal, junction, 215.)
  while sim.s < signal + mc.RED_PAST_M + 1.:
    sim.step(engaged=True)
  read['state'] = 'red'
  for _ in range(int(10. / DT_MDL)):
    plan = sim.step(engaged=True)
    assert not plan.shouldStop
  assert sim.s > junction


# --- a green read only in bursts (a night lamp scoring just over the reader's floor) ---

def boxed_in_bursts(boxed_s: float, every_s: float, gap_red_s: float | None = None):
  """A mapped signal 9 m before a right 2.3 km on; the reader boxes its one head (reader id 0) in bursts of boxed_s every
  every_s, its vote published 0.2 s into each: red on the approach and the car's first 5 s at the line, green after. A red
  read gap_red_s into each gap, if given. Returns (sim, the state: 'green' once the light is)."""
  line, head = 291., 315.
  sim = shipped(Sim([(2600., "turn", "right", junction_of("right"))], controls=[(line, 'signal')]))
  st = {'stood': 0., 'green': False}

  def lights(s):
    if s.v < 0.1 and line - s.s < 30.:
      st['stood'] += DT_MDL
    st['green'] = st['stood'] >= 5.
    if int(round(s.t / DT_MDL)) % 2 == 0:
      phase = s.t % every_s
      heads = []
      if 2. < head - s.s < 140.:
        if 0.2 <= phase < boxed_s:
          heads = [('green' if st['green'] else 'red', head)]
        elif gap_red_s is not None and abs(phase - boxed_s - gap_red_s) < 0.06:
          heads = [('red', head)]
      s.sm.put('turnpilotDetections', with_ids(heads, s.s))
    return NONE
  sim.lights = lights
  return sim, st


def with_ids(heads, s_car: float):
  """detections, each head carrying the reader's id 0 and a bearing."""
  msg = messaging.new_message('turnpilotDetections')
  td = msg.turnpilotDetections
  td.camera, td.lightStatus = 'road', 'read'
  td.lights.source, td.lights.shadow = 'camera', False
  td.lights.lights = [{'state': st, 'read': st, 'distance': s - s_car, 'governs': True, 'votes': 4, 'conf': 0.7, 'bearing': 0.,
                       'headIdPlusOne': 1} for st, s in heads]
  return msg.as_reader().turnpilotDetections


def went_after_green(sim, st, seconds: float = 60.):
  t_green = None
  for _ in range(int(seconds / DT_MDL)):
    sim.step(engaged=True)
    if st['green'] and t_green is None:
      t_green = sim.t
    if t_green is not None and sim.v > 0.5:
      return sim.t - t_green
  return None


@pytest.mark.parametrize("boxed_s, every_s", [(1.0, 4.), (1.2, 4.), (1.0, 2.5)])
def test_a_green_boxed_only_in_short_bursts_releases_the_stand_within_a_few_seconds(boxed_s, every_s):
  # a head boxed in bursts shorter than CAMERA_GREEN_STEADY_S with gaps over CAMERA_GREEN_GAP_S: stood for good before.
  # Released once the bursts' own read spans sum to CAMERA_GREEN_STEADY_S (each publishes about 0.7 s of its boxed_s):
  # the second burst's or the third's
  sim, st = boxed_in_bursts(boxed_s, every_s)
  went = went_after_green(sim, st)
  assert went is not None and went < 2. * every_s + 1., f"stood {sim.t:.0f} s in, reason {sim.trace[-1][2].reason!r}"


def test_a_green_in_bursts_with_a_red_read_in_every_gap_never_releases_the_stand():
  sim, st = boxed_in_bursts(1.0, 4., gap_red_s=1.5)
  assert went_after_green(sim, st) is None and sim.trace[-1][2].shouldStop


# --- a red read by a near-side head, then only far-side greens past the match window (CAMERA_ASSOC_PAST_M) ---

def far_greens_only(far_heads=((334., -1.5), (338., 1.5)), clear_m: float | None = None, monkeypatch=None):
  """A mapped signal (line 291 m) whose stop a near-side pole's red raised (the head out of the road camera's top inside
  14 m), the far-side heads read only once the light is green (5 s into the stand), 43 and 47 m past the line. clear_m: the
  signal's signalClearM (the map's gate), None for none (closed)."""
  from openpilot.selfdrive.turnpilot import control_tracker
  line = 291.
  if clear_m is not None:
    real = control_tracker.from_msg

    def from_msg(route_count, c):
      t = real(route_count, c)
      if t.kind == 'signal':
        t.clear_m = clear_m
      return t
    monkeypatch.setattr(control_tracker, 'from_msg', from_msg)
  sim = shipped(Sim([(2600., "turn", "right", junction_of("right"))], controls=[(line, 'signal')]))
  st = {'stood': 0., 'green': False}

  def lights(s):
    if s.v < 0.1 and line - s.s < 30.:
      st['stood'] += DT_MDL
    st['green'] = st['stood'] >= 5.
    if int(round(s.t / DT_MDL)) % 2 == 0:
      heads = []
      if 14. < line + 1. - s.s < 140.:
        heads.append((0, 'green' if st['green'] else 'red', line + 1. - s.s, -4.))
      for k, (hs, b) in enumerate(far_heads):
        if st['green'] and 2. < hs - s.s < 140.:
          heads.append((k + 1, 'green', hs - s.s, b))
      msg = messaging.new_message('turnpilotDetections')
      td = msg.turnpilotDetections
      td.camera, td.lightStatus = 'road', 'read'
      td.lights.source, td.lights.shadow = 'camera', False
      td.lights.lights = [{'state': c, 'read': c, 'distance': d, 'governs': True, 'votes': 4, 'conf': 0.7, 'bearing': b,
                           'headIdPlusOne': hid + 1} for hid, c, d, b in heads]
      s.sm.put('turnpilotDetections', msg.as_reader().turnpilotDetections)
    return NONE
  sim.lights = lights
  rows = []
  for _ in range(int(60. / DT_MDL)):
    p = sim.step(engaged=True)
    rows.append((sim.t, sim.v, st['green'], p))
  return rows


def test_far_side_greens_past_the_window_release_the_stand_where_the_map_clears_the_road(monkeypatch):
  rows = far_greens_only(clear_m=150., monkeypatch=monkeypatch)
  t_green = next(t for t, v, g, p in rows if g)
  assert any(v > 0.5 for t, v, g, p in rows if t_green <= t <= t_green + 5.)


def test_far_side_greens_the_map_cant_tie_to_the_light_hold_it_and_say_so_for_good():
  # the map's gate closed (no signalClearM): no release, but the line says a green is read that it can't tie to this
  # light and that the gas goes, for as long as it is read; one chime (the unseen line's onset), never a take-control
  from openpilot.selfdrive.turnpilot.hooks import TAKE_CONTROL_PROMPTS
  rows = far_greens_only()
  t_green = next(t for t, v, g, p in rows if g)
  after = [(t, v, p) for t, v, g, p in rows if t >= t_green + 1.]
  assert all(v < 0.05 and p.shouldStop for t, v, p in after)
  assert all(p.reason == mc.FAR_GREEN_UNMATCHED_REASON for t, v, p in after), sorted({p.reason for t, v, p in after})
  assert not any(str(p.controlPrompt) in TAKE_CONTROL_PROMPTS for t, v, g, p in rows)
  alerts = [str(p.controlAlert) for t, v, g, p in rows if t >= t_green]
  onsets = [a for k, a in enumerate(alerts) if a != 'none' and (k == 0 or alerts[k - 1] != a)]
  assert onsets in ([], ['signalUnseen']), onsets


@pytest.mark.parametrize('before', ['yellow', 'green'])
def test_a_red_after_a_yellow_stop_is_stopped_as_a_red_when_the_driver_engages_again_near_the_line(before):
  """The driver cancels 25 m before the signal's line and holds 6.9 m/s; the light reads red from 22 m out (after a yellow,
  or straight from green); he engages again 7.7 m before the line on the read red, where a firm stop for it fits
  (red_stop_fits). The car stands within RED_PAST_M of the line. A stop decided on the yellow kept the yellow's -2.0 floor
  and no reach past the line through the red: it overran 2.2 m past the line and the car turned right through the red at
  3 m/s (the closed-loop matrix, gate2)."""
  line, head = 291., 315.
  sim = shipped(Sim([(300., "turn", "right", junction_of("right")), (1500., "turn", "left", junction_of("left"))],
                    controls=[(line, 'signal')], cruise=10.))
  sim.answers_prompt = True

  def light(s):
    d = line - s.s
    return 'green' if d > 70. else before if d > 22. else 'red'
  camera(sim, lambda s: [(light(s), head)])
  phase, engage, stood = 'on', None, False
  sim.accel = lambda s, plan: 0. if phase == 'off' else \
    min(plan_accel(plan.accelCap, plan.speedCap, s.v), (s.cruise - s.v) / 2., 1.5, -1. if plan.shouldStop else 1.5)
  for _ in range(int(60. / DT_MDL)):
    d = line - sim.s
    if phase == 'on' and d <= 25.:
      phase, sim.v = 'off', 6.9  # the driver's foot: openpilot off, the speed held
    elif phase == 'off' and light(sim) == 'red' and d <= 7.7:
      phase, engage = 'again', (d, sim.v)
    sim.step(engaged=phase != 'off', lat_active=phase != 'off')
    if phase == 'again' and sim.v < 0.05:
      stood = True
      break
    if sim.s > line + 12.:
      break
  assert engage is not None and mc.red_stop_fits(engage[1], engage[0]), engage
  went = f"went to {sim.s - line:+.1f} m past the line at {sim.v:.1f} m/s without standing"
  assert stood and sim.s <= line + mc.RED_PAST_M, \
    f"engaged again {engage[0]:.1f} m before the line at {engage[1]:.1f} m/s on a read red ({before} before it): the car {went}"


def test_a_lefts_yellow_stop_meeting_its_red_at_the_line_goes_as_a_late_red_never_a_stop_past_its_reach():
  """The matrix's gyr25|reengage lefts: decided on the yellow, the driver takes over 25 m before the signal's line and rolls
  to it at 4.8 m/s, handing back on the red he sees; the camera's red is read a run later at the line, where no red's stop
  fits (red_stop_fits). The signal goes through as a late red (unsaid where the left's own hold past it stops the car); it is a stop again only where
  the car slows to COMMIT_SPEED inside RED_PAST_M (a late red's rule), never one holding it beyond. The left's own
  handback past its hold here is HEAD's (the left's procedure, not the signal). Made a red's stop there, the signal stopped the
  car 4.4 m past its line, beyond RED_PAST_M (the matrix's left then stood there handed back for good)."""
  line, head = 291., 315.
  sim = shipped(Sim([(300., "turn", "left", junction_of("left")), (1500., "turn", "right", junction_of("right"))],
                    controls=[(line, 'signal')], cruise=10.))
  sim.answers_prompt = True
  camera(sim, lambda s: [('green' if line - s.s > 40. else 'yellow' if line - s.s > 0.3 else 'red', head)])
  phase = 'on'
  sim.accel = lambda s, plan: 0. if phase == 'off' else \
    min(plan_accel(plan.accelCap, plan.speedCap, s.v), (s.cruise - s.v) / 2., 1.5, -1. if plan.shouldStop else 1.5)
  modes = []
  for _ in range(int(45. / DT_MDL)):
    d = line - sim.s
    if phase == 'on' and d <= 25.:
      phase, sim.v = 'off', 4.8
    elif phase == 'off' and d <= 0.8:
      phase = 'again'
    sim.step(engaged=phase != 'off', lat_active=phase != 'off')
    sig = [c for c in sim.tp.map_controls.tracker.controls.values() if c.kind == 'signal']
    if sig and sig[0].state is not None and sim.s > line:
      modes.append((sig[0].state.mode, sig[0].held, sim.s - line))
  assert modes and modes[0][0] == mc.SIGNAL_COMMITTED, modes[:1]  # unsaid here: the left's own hold stops the car (turn_hold)
  held_past = [m[2] for m in modes if m[1] and m[2] > mc.RED_PAST_M + 0.05]
  assert not held_past, f"the signal held the car {max(held_past):.1f} m past its line"

