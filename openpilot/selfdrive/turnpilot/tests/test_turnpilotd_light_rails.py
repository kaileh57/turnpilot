"""The lights' rail through turnpilotd on the shipped settings (the camera's reader add-only, no oracle): a governing red that
is read always stops the car. The Sim's car follows the plan's caps and has no driving model, so a stop here is TurnPilot's
own."""
import pytest

from openpilot.cereal import log, messaging
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot import map_controls as mc
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
