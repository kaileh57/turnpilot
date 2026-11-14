"""A camera stop let go by the green of its junction's far-side heads that read long, where the map clears the road straight
on (map_controls.far_green, FAR_GREEN_*, NEXT_HEAD_*;
routing/controls.signal_clear_m, MapControl.signalClearM).

Route 29, the left at a signal (light_proxy.py: the drive's video through the detector and the reader, replayed through the
policy): a red read 80-92 m from the line raised the stop, the light turned green with the line 43 m ahead, and the only
governing heads read after it were the far-side ones, green, read 44-95 m out with the line 1-34 m ahead: 41-61 m past the
line as read, out of its window (CAMERA_ASSOC_PAST_M), their range 1.4 times long (CAMERA_RANGE_LONG). Nothing let the stop
go and the replayed plan braked from 5 s after the green to the line; closed loop the car would stand there on the green.
The local tiles and the OSM signal layer built locally from the region's extract put the next place straight on (a
drivable way meeting; no signal or signalled crossing nearer: the one 28.5 m past the junction's node is its own crosswalk,
CLEAR_OWN_M) 107.1 m past that stop point (signalClearM). FIXTURE is the drive's
governing reads, by the line's distance. Shipped settings throughout."""
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.turnpilot.routing.controls import CLEAR_LOOK_M

from openpilot.selfdrive.turnpilot.map_controls import (CAMERA_GREEN_STEADY_S, CAMERA_RANGE_LONG, CAMERA_STALE_S, NO_LIGHTS,
                                                         READER_CAM_M, READER_HEAD_M, CameraLights, Lights, next_head_read_m)
from openpilot.selfdrive.turnpilot.tests.test_map_controls import Car, held

LINE = 300.
ROUTE29_CLEAR_M = 107.1  # signalClearM of that left's signal from the local tiles and signal layer


FIXTURE = [
  (94.6, [('unknown', 99.8)]), (92.1, [('red', 103.7)]), (90.8, [('red', 103.5)]), (88.3, [('red', 103.0)]),
  (86.5, [('red', 96.2)]), (84.7, [('red', 102.3)]), (83.4, [('red', 96.3)]), (81.6, [('red', 100.1)]),
  (80.4, [('red', 96.7)]), (52.9, [('unknown', 101.5)]), (34.3, [('unknown', 94.6)]), (34.0, [('green', 95.3)]),
  (26.1, [('unknown', 87.2)]), (25.6, [('green', 83.8)]), (25.0, [('green', 83.5)]), (24.5, [('green', 83.1)]),
  (24.2, [('green', 82.1)]), (18.2, [('green', 77.4)]), (15.8, [('green', 62.6)]), (12.6, [('green', 57.9)]),
  (7.6, [('unknown', 55.7)]), (6.7, [('green', 56.2)]), (6.1, [('green', 50.7)]),
  (5.7, [('unknown', 51.6), ('green', 44.7), ('unknown', 55.4)]), (5.4, [('green', 52.0), ('green', 56.0)]),
  (5.0, [('green', 44.0), ('green', 56.0), ('green', 52.6)]), (4.7, [('green', 52.4)]), (3.8, [('green', 54.6)]),
  (3.2, [('green', 43.4), ('green', 52.0)]), (2.9, [('green', 43.6), ('green', 51.9)]), (2.2, [('green', 46.9)]),
  (1.9, [('green', 47.2)]), (1.7, [('green', 46.0)]), (1.5, [('green', 47.0), ('green', 46.8), ('green', 43.5)]),
  (1.2, [('green', 43.5)])]


def by_distance(rows, extra=None):
  """turnpilotd's camera input for runs read at the line's distances in rows: each run's read made once, when the car is that
  far from the line, its heads where the run put them against the line; carried to the next run, gone CAMERA_STALE_S after
  it. extra(car) adds heads [(state, m)] to every run."""
  state = {'i': 0, 'read': None, 't': -1.}

  def make(car):
    d = LINE - car.s
    due = state['i']
    while due < len(rows) and rows[due][0] >= d:
      due += 1
    if due > state['i']:
      state['i'] = due
      d_read, heads = rows[due - 1]
      heads = [(s, dist - d_read + d) for s, dist in heads] + (extra(car) if extra is not None else [])
      state['read'], state['t'] = Lights(camera=CameraLights(heads, False, 'read')), car.t
    if state['read'] is None or car.t - state['t'] > CAMERA_STALE_S + 1e-9:
      return NO_LIGHTS
    return state['read']
  return make


def paced(start_d, heads, hz=5.):
  """The reader's road runs hz a second from when the line is start_d m out: each run's heads [(state, m past the line, range
  factor)] read where they are then, scaled by their factor; the runs before it read nothing."""
  state = {'t': None, 'read': None}

  def make(car):
    d = LINE - car.s
    if d > start_d:
      return NO_LIGHTS
    if state['t'] is None or car.t - state['t'] >= 1. / hz - 1e-9:
      state['t'] = car.t
      state['read'] = Lights(camera=CameraLights([(s, (d + past) * k) for s, past, k in heads], False, 'read'))
    return state['read']
  return make


def red_then(rest):
  """A red read on the near-side head from 92 to 80 m out (the drive's), then rest(car)."""
  red = by_distance([r for r in FIXTURE if r[0] > 70.])

  def make(car):
    return red(car) if LINE - car.s > 75. else rest(car)
  return make


def approach(lights, clear_m, controls=((LINE, 'signal', True),), v=12.7, until_s=None):
  """The car from 96 m out, the signal's signalClearM clear_m (0 unknown), to 10 m past the line or held (or until_s)."""
  car = Car(list(controls), v=v)
  car.s = LINE - 96.

  def lit(car):
    c = car.policy.tracker.controls.get((1, 0))
    if c is not None:
      c.clear_m = clear_m
    return lights(car)
  stop = (lambda out, dec: car.t >= until_s) if until_s is not None else (lambda out, dec: car.s > LINE + 10. or held(out, dec))
  out, dec = car.run_until(stop, max_s=40., lights=lit)
  return car, out, dec


def released(car) -> bool:
  st = car.policy.tracker.controls[(1, 0)].state
  return bool(st.done) or st.mode == 'go'


def test_the_drives_far_side_greens_let_the_stop_go_where_the_map_clears_the_road():
  car, out, dec = approach(by_distance(FIXTURE), ROUTE29_CLEAR_M)
  assert not held(out, dec) and car.s > LINE + 10. and released(car)
  assert min(r[2] for r in car.rows) > 1.  # slowed by the red's curve, never stopped at the green
  # the map's distance unknown (0, an old navd's or one it could not follow): the stop stands to the line, as before
  car, out, dec = approach(by_distance(FIXTURE), 0.)
  assert held(out, dec) and abs(LINE - car.s) < 5.


def test_far_side_greens_with_the_road_clear_let_the_stop_go_a_second_after_they_can_be_taken():
  # the 364-379 shape at the reader's 5 runs a second: the green from 45 m out on the far-side heads 28 and 34 m past the
  # line, read 1.4 times long, the road straight on clear as far as the map looks (CLEAR_LOOK_M). Read 102 and 110 m out
  # there, they lie past the nearest read the next place's heads can show (next_head_read_m, 97 m) until the line is about
  # 31 m out; from there the stop goes within a steady green
  green = paced(45., [('green', 28., CAMERA_RANGE_LONG), ('green', 34., CAMERA_RANGE_LONG)])
  car, out, dec = approach(red_then(green), CLEAR_LOOK_M)
  takeable = next(r[0] for r in car.rows if LINE - r[1] <= 31.)
  let_go = next((r[0] for r in car.rows if not r[4].should_stop and not r[5].stops and LINE - r[1] < 45.), None)
  assert released(car) and not held(out, dec)
  assert let_go is not None and let_go - takeable <= CAMERA_GREEN_STEADY_S + 0.25, let_go - takeable


@pytest.mark.parametrize("past", [50., 60., 80.])
@pytest.mark.parametrize("k", [1.0, CAMERA_RANGE_LONG])
def test_the_next_junctions_green_never_lets_this_lines_red_go(past, k):
  """The rail: this line's heads unread after its red, the next junction's green past m past the line read k times its range
  (the elevation's range reads long for a low head). The map closes the gate (its next place there: signalClearM no further
  than the head) or keeps it shut by its margin (next_head_read_m) where it says the road is clear further on than the head
  (a map that missed a junction); unknown holds too. Held at the line. (Read short, 0.75 times, a head 50-60 m past the
  line lies inside the line's own window as read: test_a_next_junctions_green_read_short_is_inside_the_window_as_read_whatever_the_map_says.)"""
  for clear_m in (0., past + 1., past + 20.):
    car, out, dec = approach(red_then(paced(70., [('green', past, k)])), clear_m)
    assert held(out, dec) and abs(LINE - car.s) < 5., (past, k, clear_m, car.s - LINE)


def in_force_to(car, until_m) -> bool:
  """The stop never let go while the car was further than until_m from the line."""
  return all(r[5].stops for r in car.rows if LINE - r[1] > until_m and LINE - r[1] < 70.)


def test_a_red_read_anywhere_keeps_the_stop_beside_the_far_greens():
  # a governing red at any range (the next junction's, or this one's read past every window), a nearer red or yellow no line
  # takes, a low-confidence red: never let go by the far greens. They come inside the window as read only from 2 m out
  # ((d + 28) x 1.4 <= d + 40), where the movement rule takes them as before
  far = [('green', 28., CAMERA_RANGE_LONG), ('green', 34., CAMERA_RANGE_LONG)]
  car, out, dec = approach(red_then(paced(45., far)), CLEAR_LOOK_M)
  assert not in_force_to(car, 15.)  # the control: let go well short of the line
  for extra in ([('red', 140., 1.)], [('yellow', 100., 1.)], [('red', -15., 1.)], [('red', 2., 1.)]):
    car, out, dec = approach(red_then(paced(45., far + extra)), CLEAR_LOOK_M)
    assert in_force_to(car, 2.5), extra
  base = paced(45., far)

  def vetoed(car):
    out = base(car)
    if out.camera is not None:
      out.camera.vetoes = [('red', 50., None, 0.)]
    return out
  car, out, dec = approach(red_then(vetoed), CLEAR_LOOK_M)
  assert in_force_to(car, 2.5)


def test_a_next_junctions_green_read_short_is_inside_the_window_as_read_whatever_the_map_says():
  # the open case, unchanged: a head 50 or 60 m past the line read 0.75 times its range (hung higher than assumed, or the
  # pitch off) is inside the line's window as read (CAMERA_ASSOC_PAST_M) and its green is the line's, with the map's next
  # place there or not known. Narrowing the green window to next_head_read_m would hold these, and on route 29 it held a
  # left's near line through its green (CONFIDENT_BUILD.md)
  for past in (50., 60.):
    for clear_m in (past + 10., 0.):
      car, out, dec = approach(red_then(paced(70., [('green', past, 0.75)])), clear_m)
      assert not held(out, dec) and released(car), (past, clear_m)


def test_the_nearest_read_another_place_can_show_is_ranged_as_the_reader_ranges():
  # light_state.ASSUMED_HEAD_M and scene/ranging.DEFAULT_HEIGHT, which map_controls copies (it doesn't import the reader)
  from openpilot.selfdrive.turnpilot.detector.light_state import ASSUMED_HEAD_M
  from openpilot.selfdrive.turnpilot.scene.ranging import DEFAULT_HEIGHT
  assert (READER_HEAD_M, READER_CAM_M) == (ASSUMED_HEAD_M, DEFAULT_HEIGHT)
  c = SimpleNamespace(d=0., clear_m=0.)
  assert next_head_read_m(c) is None
  for t, ratio in ((50., 0.617), (100., 0.548), (150., 0.493)):
    c = SimpleNamespace(d=t + 15. - 100., clear_m=100.)
    assert next_head_read_m(c) / t == pytest.approx(ratio, abs=0.005)
  assert next_head_read_m(SimpleNamespace(d=0., clear_m=10.)) == 0.


def test_the_far_greens_count_only_toward_a_stop_the_cameras_red_raised():
  # with no stop in force they are no green of the line (a right's flow, a turn's go, a stale green): only a release changes
  car = Car([(LINE, 'signal', True)], v=12.7)
  car.s = LINE - 40.

  def lit(car):
    c = car.policy.tracker.controls.get((1, 0))
    if c is not None:
      c.clear_m = 150.
    return paced(40., [('green', 28., CAMERA_RANGE_LONG)])(car)
  car.run_until(lambda out, dec: car.s > LINE - 3., max_s=20., lights=lit)
  st = car.policy.tracker.controls[(1, 0)].state
  assert st.mode == 'go' and st.camera_green.n == 0 and not st.cam_green
