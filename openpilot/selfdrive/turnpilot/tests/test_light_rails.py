"""The lights' rail through map_controls and the camera's reader: a governing red that is read always stops the car, and
nothing that is not this movement's read green lets it go. The camera add-only and no accepted source, as the car ships."""
import math
from dataclasses import replace

import numpy as np
import pytest

from openpilot.cereal import messaging
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot import map_controls as mc
from openpilot.selfdrive.turnpilot.control_tracker import TrackedControl
from openpilot.selfdrive.turnpilot.detector import decoder, light_state
from openpilot.selfdrive.turnpilot.detector.light_state import LightReader, head_near, head_place
from openpilot.selfdrive.turnpilot.detector.modeld_detector import fill_heads, fill_lights
from openpilot.selfdrive.turnpilot.detector.tests.test_light_state import COLORS, H, LENS, LEVEL, STRIDE, W, Y_HEIGHT, nv12
from openpilot.selfdrive.turnpilot.map_controls import CameraLights, GreenReads, Lights, camera_light_input
from openpilot.selfdrive.turnpilot.tests.test_map_controls import Car, held
from openpilot.selfdrive.turnpilot.tests.test_signal_turns import HEAD, right_at_a_signal

JUNCTION, LINE = 200., 192.  # a mapped signal's line 8 m short of its junction
CAM_H = 1.22


def test_the_shipped_settings_these_tests_rest_on():
  assert (mc.CAMERA_ASSOC_BEFORE_M, mc.CAMERA_ASSOC_PAST_M, mc.CAMERA_RANGE_LONG) == (10., 40., 1.4)
  assert mc.CAMERA_RANGE_MAX == light_state.MAX_RANGE
  assert (mc.RED_GONE_S, mc.RED_GONE_READS, mc.CAMERA_GREEN_READS, mc.CAMERA_GREEN_STEADY_S) == (2.5, 6, 3, 1.0)
  assert (decoder.MIN_CONF, decoder.LIGHT_READ_CONF, decoder.LIGHT_GREEN_CONF) == (0.35, 0.2, 0.1)
  assert (light_state.VOTE_M, light_state.VOTE_N, light_state.GOVERN_BEARING_DEG, light_state.ASSUMED_HEAD_M) == (5, 2, 10., 5.5)
  assert light_state.LIGHT_MODE_DEFAULT == 'addOnly'


def cam_heads(*heads, dark=(), **kw):
  """Governing heads the camera reads (add-only, acting): (voted state, road position s)."""
  return lambda car: Lights(camera=CameraLights([(st, s - car.s) for st, s in heads], False, 'read',
                                                dark=[(st, s - car.s) for st, s in dark], **kw))


def drive(car, lights, seconds=40., until_s=None):
  """Where the car is first held, None if it never is, driving until until_s or for seconds."""
  held_at = None
  for _ in range(int(seconds / DT_MDL)):
    out, dec, _ = car.step(lights=lights)
    if held(out, dec) and held_at is None:
      held_at = car.s
    if until_s is not None and car.s > until_s:
      break
  return held_at


def through(v=13.4):
  return Car([(LINE, 'signal', True)], v=v)


def right(v=13.4, mapped=True):
  """A curb-side turn in hand at the junction, its look clear."""
  car = Car([(LINE, 'signal', True)] if mapped else [], v=v)
  car.junction = (JUNCTION, 7)
  car.right = True
  return car


def camera_of(reader, out):
  """turnpilotd's camera input from one run of the reader: its heads and its unpublished boxes as modeld fills them."""
  msg = messaging.new_message('turnpilotDetections')
  td = msg.turnpilotDetections
  fill_lights(td.lights, out, False)
  fill_heads(td.init('lightRejects', len(reader.rejects)), reader.rejects)
  td = msg.as_reader().turnpilotDetections
  return camera_light_input(td.lights, 'read', 0., 'addOnly', td.lightRejects)


def placed(cam, d: float):
  """cam with every head and veto d m ahead (a synthetic frame's boxes range anywhere), its ids and bearings kept."""
  return replace(cam, heads=[(s, d) for s, _ in cam.heads], dark=[(s, d) for s, _ in cam.dark], nears=None,
                 vetoes=[(v[0], d) for v in cam.vetoes])


# --- the signal's state is its movement's: the majority of its nearest heads, the left-turn head for a left ---

BALLS, LEFT_HEAD = (0., -2.5), 5.  # bearings, deg left: two through heads over the lanes, the left-turn head beside them


def junction_heads(balls: str, left: str | None = None, more=()):
  """The junction's far-side heads 25 m past the line: the two through heads reading balls, the left-turn head reading left
  (None: none there), each with the reader's id and its bearing; more: other heads [(state, road position s, bearing)]."""
  rows = [(balls, LINE + 25., b) for b in BALLS] + ([(left, LINE + 25., LEFT_HEAD)] if left is not None else []) + list(more)
  return lambda car: Lights(camera=CameraLights([(st, s - car.s) for st, s, _ in rows], False, 'read',
                                                ids=list(range(len(rows))), bearings=[b for _, _, b in rows]))


def left_turn(v=13.4):
  """The route's unprotected left at the junction, TurnPilot taking it, its signal mapped."""
  car = through(v)
  car.turn, car.junction = (JUNCTION, 7, 'left'), (JUNCTION, 7)
  return car


MOVEMENTS = {'through': through, 'right': right, 'left': left_turn}


@pytest.mark.parametrize("movement", MOVEMENTS)
def test_all_heads_red_stops_every_movement_at_the_line(movement):
  car = MOVEMENTS[movement]()
  held_at = drive(car, junction_heads('red', 'red'), until_s=LINE + 20.)
  assert held_at is not None and LINE - 5. < held_at < LINE + 2.


@pytest.mark.parametrize("movement", MOVEMENTS)
def test_red_balls_beside_a_green_left_arrow_stop_through_and_right_and_are_the_lefts_green(movement):
  car = MOVEMENTS[movement]()
  lit = junction_heads('red', 'green')
  held_at = drive(car, lit, until_s=LINE + 3.)
  st = car.policy.tracker.controls[(1, 0)].state
  if movement == 'left':  # no stop at its signal's line: its own head is green. Its own hold in the junction's mouth, where
    # its oncoming lanes are judged as ever, is what holds it
    assert st.camera == 'green' and st.mode == 'go' and not any(r[5].stops and r[5].shown.kind == 'signal' for r in car.rows)
    assert held_at is not None and held_at > LINE + 1. and car.rows[-1][5].shown.kind == mc.UNPROTECTED
    assert car.policy.turn_light(7, JUNCTION - car.s) == 'green'
  else:
    assert held_at is not None and LINE - 5. < held_at < LINE + 2. and st.camera == 'red'
    for _ in range(int(10. / DT_MDL)):
      out, dec, _ = car.step(lights=lit)
      assert held(out, dec)


@pytest.mark.parametrize("movement", MOVEMENTS)
def test_green_balls_beside_a_red_left_head_let_through_and_right_go_unslowed_and_hold_the_left(movement):
  car = MOVEMENTS[movement]()
  lit = junction_heads('green', 'red')
  if movement == 'left':
    held_at = drive(car, lit, until_s=LINE + 20.)
    assert held_at is not None and LINE - 5. < held_at < LINE + 2.
    assert car.policy.tracker.controls[(1, 0)].state.light == 'red' and car.policy.turn_light(7, JUNCTION - car.s) == 'red'
  else:
    for _ in range(int(18. / DT_MDL)):
      out, dec, _ = car.step(lights=lit)
      assert not out.stops and car.v > 13.3
    assert car.s > JUNCTION


def test_the_lefts_own_light_is_its_head_at_a_junction_with_no_signal_mapped_too():
  for left, state in (('red', 'red'), ('green', 'green')):
    car = Car([], v=5.)
    car.turn, car.junction = (JUNCTION, 7, 'left'), (JUNCTION, 7)
    car.s = JUNCTION - 60.
    car.run(1., lights=junction_heads('green' if left == 'red' else 'red', left))
    assert car.policy.junction_lights(JUNCTION - car.s)[1] == {state}
    assert (car.policy._left_red is not None) == (left == 'red')


def test_one_head_in_view_governs_and_a_tie_goes_to_the_head_ahead_or_to_red():
  assert mc.movement_state([('red', 40., 6.)]) == 'red' and mc.movement_state([('red', 40., 6.)], 1) == 'red'
  assert mc.movement_state([('green', 40., None), ('unknown', 41., None)]) == 'green'
  assert mc.movement_state([('green', 40., 0.5), ('red', 40., 5.)]) == 'green'    # a tie: the head nearest the car's heading
  assert mc.movement_state([('green', 40., 5.), ('red', 40., 0.5)]) == 'red'
  assert mc.movement_state([('green', 40., 2.), ('red', 40., -2.)]) == 'red'      # neither is: red
  assert mc.movement_state([('green', 40., None), ('red', 40., None)]) == 'red'   # no bearings: red
  assert mc.movement_state([('green', 40., None), ('yellow', 40., None)]) == 'yellow'
  assert mc.movement_state([('green', 40., 5.), ('red', 40., 0.), ('red', 40., -2.)], 1) == 'green'   # a left: its own head
  assert mc.movement_state([('red', 40., 5.), ('green', 40., 0.), ('green', 40., -2.)], 1) == 'red'
  assert mc.movement_state([('red', 40., 5.), ('green', 40., 0.), ('green', 40., -2.)], -1) == 'green'  # a far-side right
  assert mc.movement_state([('red', 40., None), ('green', 40., None), ('green', 40., None)], 1) == 'red'  # which is its head unknown
  assert mc.movement_state([('green', 40., 0.), ('red', 90., 0.), ('red', 92., 1.)]) == 'green'  # the next junction's have no say


def test_the_next_junctions_red_has_no_say_while_this_lines_heads_read_green():
  # a red 70 m past this line that no line of its own takes is this line's by its span, but not of its nearest heads: this
  # line's own read green, and the car goes through unslowed. With this line's heads unread, or none, it stops the car here
  far = [('red', LINE + 70., 0.)]
  car = through()
  for _ in range(int(18. / DT_MDL)):
    out, dec, _ = car.step(lights=junction_heads('green', more=far))
    assert not out.stops and car.v > 13.3
  assert car.s > LINE + 10.
  alone = lambda car: Lights(camera=CameraLights([('red', LINE + 70. - car.s)], False, 'read', ids=[9], bearings=[0.]))  # noqa: E731
  for lit in (junction_heads('unknown', more=far), alone):
    car = through()
    held_at = drive(car, lit, until_s=LINE + 10.)
    assert held_at is not None and LINE - 5. < held_at < LINE + 2.


@pytest.mark.parametrize("reds, greens", [(1, 0), (2, 0), (1, 1), (2, 1), (2, 2)])
def test_a_read_red_stops_a_through_car_where_green_heads_beside_it_are_no_majority(reds, greens):
  # the through heads red from far out; other heads matched to the same signal vote green (a protected-left green arrow
  # read as a ball, the next junction's green ranged into the window, a low-confidence green box), no bearing to tell them by
  car = through()
  heads = [('red', LINE + 25. + 2 * i) for i in range(reds)] + [('green', LINE + 30. + i) for i in range(greens)]
  held_at = drive(car, cam_heads(*heads), until_s=LINE + 20.)
  assert held_at is not None and LINE - 5. < held_at < LINE + 2., f"no stop: s={car.s:.1f} v={car.v:.1f}"
  st = car.policy.tracker.controls[(1, 0)].state
  assert st.light == 'red' and st.camera == 'red' and st.cam_green == 0
  # ...and holds it there for as long as the red is read: the greens beside it never make a steady green
  for _ in range(int(20. / DT_MDL)):
    out, dec, _ = car.step(lights=cam_heads(*heads))
    assert held(out, dec) and not st.camera_green.n


def test_a_right_never_goes_while_a_red_is_read_beside_a_green():
  car = right()
  lit = cam_heads(('red', LINE + 25.), ('green', LINE + 30.))
  assert drive(car, lit, until_s=LINE + 20.) is not None
  for _ in range(int(20. / DT_MDL)):  # its look clear throughout: the red read is what it waits on
    out, dec, _ = car.step(lights=lit)
    assert held(out, dec) and out.stops[0].reason == mc.RIGHT_RED_WAIT_REASON


def test_a_read_red_beside_a_green_at_an_unmapped_junction_makes_the_cameras_signal():
  car = right(mapped=False)
  held_at = drive(car, cam_heads(('red', JUNCTION + 15.), ('green', JUNCTION + 20.)), until_s=JUNCTION + 15.)
  assert held_at is not None and car.policy.camera_signal is not None


def arrow_layer(color: str, lamp: int, factor: int, gain: float = 1.0):
  """A lit left arrow's light as a far head shows it: test_light_state's 30x90 drawing six times larger, area-averaged down
  by factor as a lens and a sensor shrink it (gain over 1: the lamp brighter than the sensor's range)."""
  w, h, k = 30, 90, 6
  small = np.zeros((h, w, 3), np.float32)
  tip_x, cy = 4, 15 + 30 * lamp
  for dy in range(-9, 10):
    small[cy + dy, tip_x + abs(dy): tip_x + abs(dy) + 3] = COLORS[color]
  small[cy - 2:cy + 3, tip_x:w - 4] = COLORS[color]
  big = np.kron(small, np.ones((k, k, 1), np.float32)) * gain
  hh, ww = h * k // factor, w * k // factor
  return big[:hh * factor, :ww * factor].reshape(hh, factor, ww, factor, 3).mean(axis=(1, 3))


def arrow_frame(red_conf=0.30):
  """A night frame: two red balls (the through heads, 15x45 px) and, 60 px to their left, a head of the same size showing a
  lit green left arrow in its bottom lamp, which the reader's ball-or-arrow test takes for a ball at this size. Returns
  (nv12, decode rows): the reds at red_conf, the arrow's head at 0.15."""
  img = np.zeros((H, W, 3), np.float32)
  img[:] = (20, 17, 15)
  layer = arrow_layer('green', 2, 12, 2.0)
  h, w = layer.shape[:2]
  y0, boxes = 200, []
  yy, xx = np.mgrid[0:H, 0:W]
  for x0 in (660, 720):
    img[(xx + 0.5 - (x0 + w / 2)) ** 2 + (yy + 0.5 - (y0 + h / 6)) ** 2 <= (0.38 * w) ** 2] = COLORS['red']
    boxes.append([x0, y0, x0 + w, y0 + h, red_conf, 9])
  lit = layer.max(axis=-1) > 0
  region = img[y0:y0 + h, 600:600 + w]
  frac = np.clip(layer.max(axis=-1, keepdims=True) / 255., 0., 1.)
  region[:] = np.where(lit[..., None], np.clip(region * (1 - frac) + np.clip(layer, 0, 255), 0, 255), region)
  boxes.append([600, y0, 600 + w, y0 + h, 0.15, 9])
  return nv12(img.astype(np.uint8)), np.array(boxes, np.float32)


def test_red_balls_beside_a_green_arrow_the_reader_takes_for_a_ball_are_the_signals_red():
  frame, boxes = arrow_frame()
  reader = LightReader(LENS)
  for i in range(3):
    out = reader.read(i * 0.2, frame, STRIDE, Y_HEIGHT, LEVEL, CAM_H, boxes)
  cam = camera_of(reader, out)
  assert sorted(s for s, _ in cam.heads) == ['green', 'red', 'red']  # the premise: the arrow votes green beside the reds
  signal = TrackedControl((1, 0), 'signal', cam.heads[0][1] - 25.)  # the heads 25 m past its line
  assert mc.camera_state(signal, [signal], cam) == 'red'


# --- a red on a head ranged past its line's window is still that line's ---

def reader_range(true_m: float, centre_m: float, box_m: float = light_state.HEAD_M):
  """(range, nearer range) the reader gives a head true_m ahead whose centre hangs centre_m over the road, its box box_m tall."""
  v_c = H / 2 - LENS.f * (centre_m - CAM_H) / true_m
  half = LENS.f * box_m / true_m / 2
  box = np.array([660., v_c - half, 660. + 2 * half / 3, v_c + half, 0.5, 9], np.float32)
  d = head_place(LENS, LEVEL, CAM_H, box)[2]
  return d, head_near(LENS, CAM_H, box, d)


def test_a_heads_nearer_range_is_its_boxs_where_it_hangs_low():
  d, near = reader_range(50., 5.5)
  assert abs(d - 50.) < 0.5 and abs(near - 50.) < 0.5  # hung as assumed: the two agree
  for centre_m, long in ((4.3, 1.39), (3.7, 1.73), (3.0, 2.40)):
    d, near = reader_range(50., centre_m)
    assert abs(d / 50. - long) < 0.02 and abs(near - 50.) < 0.6, (centre_m, d, near)
  # a box drawn round the lamp's glow alone says a longer range: the elevation's stands
  d, near = reader_range(50., 5.5, box_m=0.5)
  assert near == d
  # a box larger than a head (its glow, a backplate) is never placed nearer than the lowest mounting allows
  d, near = reader_range(50., 5.5, box_m=4.)
  assert abs(near - d * (light_state.LOW_HEAD_M - CAM_H) / (light_state.ASSUMED_HEAD_M - CAM_H)) < 1e-6


FIRST_READ_M = 55.  # a head is read from about here in (HEAD_MIN_H_PX: a 1.07 m head at 50 m on this lens)


def ranged_red(past_line_m: float, centre_m: float, box_m: float = light_state.HEAD_M):
  """One far-side head past_line_m past the mapped line, its centre centre_m over the road, read red on every run once it
  is FIRST_READ_M off, at the ranges the reader gives it."""
  def lights(car):
    true_m = LINE + past_line_m - car.s
    if not 0. < true_m <= FIRST_READ_M:
      return Lights(camera=CameraLights([], False, 'read'))
    d, near = reader_range(true_m, centre_m, box_m)
    return Lights(camera=CameraLights([('red', d)], False, 'read', nears=[near]))
  return lights


@pytest.mark.parametrize("centre_m, box_m", [(5.5, 1.07), (4.3, 1.07), (3.7, 1.07), (3.0, 1.07), (5.5, 0.5), (4.3, 0.5), (3.7, 0.5)])
def test_a_read_red_on_a_low_far_side_head_stops_the_car(centre_m, box_m):
  # 25 mph; a far-side head 25 m past the mapped line reads red from 55 m off (30 m short of the line). Hung lower than
  # assumed it ranges 1.4 to 2.4 times long, past the line's window: its nearer range reaches it. Boxed by its glow alone
  # (0.5 m) it has none, and the range over CAMERA_RANGE_LONG reaches for a head 3.7 m up or higher
  car = through(v=11.2)
  held_at = drive(car, ranged_red(25., centre_m, box_m), until_s=LINE + 10.)
  assert held_at is not None and held_at < LINE + 2., f"no stop, s={car.s:.1f} v={car.v:.1f}"


def test_the_span_a_red_may_lie_in_has_its_bounds():
  line = TrackedControl((1, 0), 'signal', 30.)
  assert mc.camera_owner(69., [line], 'red') is line and mc.camera_owner(71., [line], 'green') is None  # the window as read
  assert mc.camera_owner(97., [line], 'red') is line and mc.camera_owner(97., [line], 'yellow') is line
  assert mc.camera_owner(99., [line], 'red') is None  # past the range over CAMERA_RANGE_LONG
  assert mc.camera_owner(140., [line], 'red', near=65.) is line and mc.camera_owner(140., [line], 'unknown', near=65.) is None
  assert mc.camera_owner(140., [line], 'red', near=140.) is None  # a head 3 m up boxed by its glow: no nearer range to say so
  assert mc.camera_owner(mc.CAMERA_RANGE_MAX, [TrackedControl((1, 1), 'signal', 80.)], 'red') is None  # a range the reader could not say
  near, far = TrackedControl((1, 1), 'signal', 30.), TrackedControl((1, 2), 'signal', 60.)
  assert mc.camera_owner(110., [near, far], 'red', near=50.) is near  # nobody's as read: the first line ahead its span reaches
  assert mc.camera_owner(95., [near, far], 'red') is far    # in a window as read: the last line before it, as any head


def test_a_right_never_goes_while_a_red_ranged_past_its_window_is_read():
  car = right(v=10.)
  lights = ranged_red(30., 4.3)
  car.run_until(held, lights=lights)
  for _ in range(int(20. / DT_MDL)):
    out, dec, _ = car.step(lights=lights)
    assert held(out, dec) and car.s < LINE + 3.


def test_the_next_junctions_red_on_a_short_block_is_its_own_lines_or_none_from_85_m_past_this_one():
  # two mapped signals 70 m apart; the far one's heads red, read from far out at their true range, this one's unread. The
  # red is its own line's (the last line before it): the car goes through this line unslowed and stops at the next
  nxt = LINE + 70.
  car = Car([(LINE, 'signal', True), (nxt, 'signal', True)], v=13.4)
  lit = cam_heads(('red', nxt + 25.))
  for _ in range(int(60. / DT_MDL)):
    out, dec, _ = car.step(lights=lit)
    if car.s < LINE + 2.:
      assert car.policy.tracker.controls[(1, 0)].state.mode == 'go'
    if held(out, dec):
      break
  assert held(out, dec) and nxt - 5. < car.s < nxt + 2.
  # ...and with no line of its own (the next junction unmapped), a red 85 m and more past this line is not this line's at
  # any range the reader says (under MAX_RANGE): the car is never slowed for it here
  car = through()
  lit = lambda car: Lights(camera=CameraLights([('red', LINE + 85. - car.s)] if LINE + 85. - car.s < mc.CAMERA_RANGE_MAX else [], False, 'read'))  # noqa: E731
  for _ in range(int(16. / DT_MDL)):
    out, dec, _ = car.step(lights=lit)
    assert not out.stops and car.v > 13.3
  assert car.s > LINE + 5.
  # the error the rule accepts: a red 41 to 83 m past this line that no line of its own takes (the next junction unmapped,
  # or its own heads ranged past its window) may be this line's, and stops the car one line early for it
  car = through()
  assert drive(car, cam_heads(('red', LINE + 60.)), until_s=LINE + 10.) is not None


def test_a_green_keeps_the_window_as_read_but_for_a_head_that_read_the_signals_red():
  # the stop a red raised on a head ranged long (60 m past the line as read) is let go by that same head's green, known by
  # the reader's id for it; another head's green out there (the next junction's) lets nothing go
  def lit(state, hid):
    return lambda car: Lights(camera=CameraLights([(state, LINE + 60. - car.s)], False, 'read', ids=[hid], bearings=[0.5 * hid]))
  car = through(v=11.2)
  car.s = LINE - 60.
  out, dec = car.run_until(held, lights=lit('red', 4))
  out, dec, _ = car.run(10., lights=lit('green', 9))
  assert held(out, dec)
  out, dec, _ = car.run(mc.CAMERA_GREEN_STEADY_S + 0.3, lights=lit('green', 4))
  assert not held(out, dec) and not out.stops


# --- a red is gone only on a head that read it, in view and reading its lamp ---

def doghouse_frame(red_conf: float):
  """A night frame at a stand: the through head (a red ball, top lamp, 14x42) boxed at red_conf, and beside it a five-lamp
  doghouse head (28x34) with its red ball lit, boxed at 0.6. Returns (nv12, decode rows)."""
  img = np.zeros((H, W, 3), np.uint8)
  img[:] = (20, 17, 15)
  yy, xx = np.mgrid[0:H, 0:W]
  for cx, cy, rad in ((667., 207., 5.3), (634., 206., 5.3)):
    img[(xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= rad ** 2] = COLORS['red']
  return nv12(img), np.array([[660, 200, 674, 242, red_conf, 9], [620, 200, 648, 234, 0.6, 9]], np.float32)


def standing_at_the_line(frames):
  """A car standing a metre short of its line, the reader reading frames(state) on a road run 5 times a second, every head
  it publishes 22 m past the line. Returns (car, lights, state): state['reads'] the last run's (read, reason) pairs."""
  reader = LightReader(LENS)
  car = through(v=0.)
  car.s, car.cruise = LINE - 1., 13.4
  state = {'t': 0., 'cam': None, 'frame': 0, 'reads': None}

  def lights(car):
    if state['frame'] % 4 == 0:
      frame, boxes = frames(state)
      out = reader.read(state['t'], frame, STRIDE, Y_HEIGHT, LEVEL, CAM_H, boxes)
      state['cam'] = placed(camera_of(reader, out), LINE + 22. - car.s)
      state['reads'] = sorted((h.read, h.reason) for h in out)
    state['frame'] += 1
    state['t'] += DT_MDL
    return Lights(camera=state['cam'])
  return car, lights, state


def test_a_red_still_lit_is_not_let_go_by_a_head_that_never_read_it():
  # stood at a camera red, the through head's box falls under the reader's floor while a doghouse head of the same signal,
  # its red ball lit, is boxed on every run and never read (aspect): no head that read the red says it has gone
  car, lights, state = standing_at_the_line(lambda state: doghouse_frame(state['conf']))
  state['conf'] = 0.30
  out, dec, _ = car.run(2., lights=lights)
  assert held(out, dec) and state['reads'] == [('red', 'read'), ('unknown', 'aspect')]
  state['conf'] = 0.18
  for _ in range(int(12. / DT_MDL)):
    out, dec, _ = car.step(lights=lights)
    assert held(out, dec), f"red let go {car.t - 2.:.1f} s after its last read, reads {state['reads']}"


def test_only_a_head_that_read_the_red_says_it_has_gone():
  def lit(*heads):  # (state, the reader's id), every head on one mast arm 22 m past the line, 3 degrees apart
    return lambda car: Lights(camera=CameraLights([(s, LINE + 22. - car.s) for s, _ in heads], False, 'read',
                                                  ids=[i for _, i in heads], bearings=[3. * i for _, i in heads]))
  car = right_at_a_signal()
  car.run_until(held, lights=lit(('red', 1), ('unknown', 2)))
  st = car.policy.tracker.controls[(1, 0)].state
  for _ in range(int(20. / DT_MDL)):  # the red's head out of the runs, the other head lit and unread on every one
    out, dec, _ = car.step(lights=lit(('unknown', 2)))
    assert held(out, dec) and st.cam_seen == 0
  out, dec, _ = car.run(mc.RED_GONE_S - 0.3, lights=lit(('unknown', 1), ('unknown', 2)))  # its own head in view, not red
  assert held(out, dec) and st.cam_seen == 1
  out, dec, _ = car.run(0.5, lights=lit(('unknown', 1), ('unknown', 2)))
  assert not held(out, dec)
  # the reader's id for the head changed (not boxed for a while): the same place and bearing is the same head
  car = right_at_a_signal()
  car.run_until(held, lights=lit(('red', 1)))
  car.run(3., lights=lit())
  out, dec, _ = car.run(mc.RED_GONE_S + 0.3, lights=lambda car: Lights(camera=CameraLights(
    [('unknown', LINE + 22. - car.s)], False, 'read', ids=[7], bearings=[3.])))
  assert not held(out, dec)


def test_a_head_whose_read_says_nothing_of_its_lamp_is_no_red_gone():
  tl = messaging.new_message('turnpilotDetections').turnpilotDetections.lights
  tl.source, tl.shadow = 'camera', False
  rows = [('unknown', 'dark'), ('unknown', 'aspect'), ('unknown', 'twoLit'), ('unknown', 'noColor'), ('red', 'aspect')]
  for i, (o, (state, reason)) in enumerate(zip(tl.init('lights', len(rows)), rows, strict=True)):
    o.state, o.read, o.reason, o.governs, o.distance, o.instance, o.bearing = state, 'unknown', reason, True, 20. + i, f"head{i} near{15 + i}.5 h40", 1.5
  cam = camera_light_input(tl.as_reader(), 'read', 0., 'addOnly')
  assert cam.dark == [('unknown', 20.), ('unknown', 21.), ('unknown', 22.)]  # a head still voting red is not one
  assert cam.ids == [0, 1, 2, 3, 4] and cam.nears == [15.5, 16.5, 17.5, 18.5, 19.5] and cam.bearings == [1.5] * 5


# --- a red read under the reader's confidence floor: no stop of its own, and no green beside it ---

def test_a_held_red_is_not_released_by_a_green_arrow_while_its_own_boxes_dip_under_the_floor():
  # stood at a camera red on two red balls boxed at 0.30. The protected-left green arrow comes on in the head beside them
  # (boxed at 0.15: read, a green is read down to LIGHT_GREEN_CONF) and the balls' boxes dip to 0.13: read too, red, no
  # heads, and the arrow's green counts for nothing beside them
  def frames(state):
    frame, boxes = arrow_frame(red_conf=state['red_conf'])
    boxes = boxes if state['arrow'] else boxes[:2]
    return frame, boxes[np.argsort(-boxes[:, 4], kind='stable')]  # decode_lights' order: the most confident first
  car, lights, state = standing_at_the_line(frames)
  state.update(red_conf=0.30, arrow=False)
  out, dec, _ = car.run(2., lights=lights)
  assert held(out, dec) and state['reads'] == [('red', 'read'), ('red', 'read')]
  state.update(red_conf=0.13, arrow=True)
  st = car.policy.tracker.controls[(1, 0)].state
  for _ in range(int(12. / DT_MDL)):
    out, dec, _ = car.step(lights=lights)
    assert held(out, dec), f"red let go {car.t - 2.:.2f} s after the arrow came on, reads {state['reads']}"
  assert state['reads'] == [('green', 'read')] and st.cam_veto == 1 and st.cam_green == 0 and len(state['cam'].vetoes) == 2


def vetoed(state, hid=1):
  """One governing head of the signal reading state, a low-confidence red read beside it."""
  return lambda car: Lights(camera=CameraLights([(state, HEAD - car.s)], False, 'read', ids=[hid], vetoes=[('red', HEAD + 1. - car.s, None)]))


def test_a_low_confidence_red_raises_no_stop_and_lets_none_go():
  # alone it stops nothing: a through car is not slowed for it
  car = through()
  for _ in range(int(18. / DT_MDL)):
    out, dec, _ = car.step(lights=lambda car: Lights(camera=CameraLights([], False, 'read', vetoes=[('red', LINE + 25. - car.s, None)])))
    assert not out.stops
  assert car.s > LINE + 10.
  # beside the signal's green it is no steady green, and beside its unread head no red gone: the stop a red raised stands
  for state in ('green', 'unknown'):
    car = through()
    car.run_until(held, lights=cam_heads(('red', HEAD), ids=[1]))
    out, dec, _ = car.run(10., lights=vetoed(state))
    assert held(out, dec), state
  # a right in hand flows on a read green, never on one a low-confidence red is read beside: it stops at its line
  car = right()
  assert drive(car, cam_heads(('green', HEAD)), until_s=JUNCTION + 5.) is None
  car = right()
  assert drive(car, vetoed('green'), until_s=JUNCTION + 5.) is not None


def test_a_low_confidence_red_at_a_junction_leaves_its_camera_green_out():
  # the rolling left's light and the left's own (turnpilotd.left_roll_light, turn_light) read the junction through
  # junction_lights: no camera green there beside a low-confidence red
  car = Car([], v=5.)
  car.step(lights=lambda car: Lights(camera=CameraLights([('green', 60.)], False, 'read')))
  assert car.policy.junction_lights(50.)[1] == {'green'}
  car.step(lights=lambda car: Lights(camera=CameraLights([('green', 60.)], False, 'read', vetoes=[('red', 62., None)])))
  assert car.policy.junction_lights(50.)[1] == {'unknown'}
  # ...but where the green's heads would still be the majority with it a head, it changes nothing
  car.step(lights=lambda car: Lights(camera=CameraLights([('green', 60.), ('green', 61.), ('green', 62.)], False, 'read',
                                                         vetoes=[('red', 62., None)])))
  assert car.policy.junction_lights(50.)[1] == {'green'}


# --- a right at a junction the map has no signal for, its heads boxed and unread ---

@pytest.mark.parametrize("heads", [[('unknown', JUNCTION + 15.)], [('unknown', JUNCTION + 15.), ('unknown', JUNCTION + 18.)]],
                         ids=["oneUnreadHead", "twoUnreadHeads"])
def test_a_right_at_an_unmapped_signal_whose_heads_are_boxed_and_unread_stops_at_the_line_looks_and_goes(heads):
  # a right flows only on a read green wherever a light is boxed at its junction: here the map has no signal, the junction's
  # governing heads are boxed on every run and read no colour. The camera's signal is made for it, and the right's
  # procedure follows: a stop at the junction's line, a stand, a look, its go
  car = right(mapped=False)
  out, dec = car.run_until(held, lights=cam_heads(*heads))
  line = JUNCTION - mc.CONTROL_LINE_M
  assert line - 5. < car.s < line + 2. and car.policy.camera_signal.state.turn_stop
  assert out.stops[0].reason == mc.RIGHT_RED_LOOKING_REASON and out.prompt == 'none'
  t_held = car.t
  car.run_until(lambda out, dec: not held(out, dec), max_s=10., lights=cam_heads(*heads))
  assert mc.RIGHT_RED_DWELL_S - 2 * DT_MDL <= car.t - t_held <= mc.RIGHT_RED_DWELL_S + 0.2
  car.run(8., lights=cam_heads(*heads))
  assert car.s > JUNCTION


def test_a_right_at_an_unmapped_junction_flows_on_a_read_green_and_on_no_head_at_all():
  for lit in (cam_heads(('green', JUNCTION + 15.)), cam_heads()):
    car = right(mapped=False)
    assert drive(car, lit, until_s=JUNCTION + 15.) is None and min(r[2] for r in car.rows) > 13.3


# --- the right-on-red go: its looks counted at any run rate, its stand ended when the turn is dropped, a red read again ---

class Reader:
  """turnpilotd.camera_lights: a road run every period_s reading state(t) on one governing head of the signal; the read is
  carried between runs (one CameraLights, one run number) and aged out to none after CAMERA_STALE_S."""
  def __init__(self, period_s, state):
    self.period, self.state, self.read, self.age, self.next_t, self.runs = period_s, state, None, 0., 0., []

  def __call__(self, car):
    self.age += DT_MDL
    if car.t >= self.next_t - 1e-9:
      self.next_t += self.period
      s = self.state(car.t)
      self.read, self.age = CameraLights([(s, HEAD - car.s)], False, 'read'), 0.
      self.runs.append((car.t, s))
    return Lights(camera=self.read if self.read is not None and self.age <= mc.CAMERA_STALE_S else None)


@pytest.mark.parametrize("period", [0.2, 0.9, 1.2, 1.6, 2.4])
def test_a_rights_go_on_an_unread_red_takes_its_looks_at_any_run_rate(period):
  # the red on its head goes unread (boxed, no colour) from t_end: the go needs RIGHT_RED_UNREAD_READS road runs since the
  # last red read, the carried read aged out between two runs or not
  car = right_at_a_signal()
  car.right = True
  t_end = [None]
  reader = Reader(period, lambda t: 'red' if t_end[0] is None or t < t_end[0] else 'unknown')
  car.run_until(held, lights=reader)
  car.run(5., lights=reader)
  t_end[0] = car.t
  last_red = max(t for t, s in reader.runs if s == 'red')
  out, dec = car.run_until(lambda out, dec: not held(out, dec), max_s=40., lights=reader)
  st = car.policy.tracker.controls[(1, 0)].state
  looks = sum(1 for t, s in reader.runs if last_red < t <= car.t)
  assert st.cause == 'rightOnRed' and looks >= mc.RIGHT_RED_UNREAD_READS and car.t - last_red >= mc.RIGHT_RED_UNREAD_S - 2 * DT_MDL


def test_a_reader_gone_quiet_leaves_the_rights_go_to_the_seconds():
  # no road run at all for RIGHT_RED_QUIET_S (the reader off, failing): the right is not left waiting on looks that never come
  car = right_at_a_signal()
  car.right = True
  car.run_until(held, lights=cam_heads(('red', HEAD)))
  car.run(mc.RIGHT_RED_DWELL_S + mc.RIGHT_RED_CLEAR_S, lights=cam_heads(('red', HEAD)))
  t0 = car.t
  car.run_until(lambda out, dec: not held(out, dec), max_s=30., lights=mc.NO_LIGHTS)
  assert mc.RIGHT_RED_QUIET_S - 2 * DT_MDL <= car.t - t0 <= mc.RIGHT_RED_QUIET_S + 0.2


def no_heads(car):
  return Lights(camera=CameraLights([], False, 'read'))


@pytest.mark.parametrize("gone", ["skipped", "noRoute"])
def test_a_stand_for_want_of_a_green_ends_once_its_turn_is_dropped(gone):
  # stood at an unread light for its right-on-red look, a car crossing; TurnPilot stops taking the turn (skipped, or the
  # route gone): no right, so no procedure to end the stand. A turn not taken goes straight, and going straight no red
  # read means go: let go TURN_DROPPED_S on
  car = right_at_a_signal()
  car.right = False
  out, dec = car.run_until(held, lights=no_heads)
  assert out.stops[0].reason == mc.RIGHT_RED_CROSS_REASON
  car.run(2., lights=no_heads)
  car.right = None
  junction = (lambda c: (7, JUNCTION - c.s, False)) if gone == "skipped" else None
  if junction is None:
    car.junction = None
  t0 = car.t
  kw = {'junction': junction} if junction is not None else {}
  car.run_until(lambda out, dec: not held(out, dec), max_s=10., lights=no_heads, **kw)
  st = car.policy.tracker.controls[(1, 0)].state
  assert mc.TURN_DROPPED_S - 2 * DT_MDL <= car.t - t0 <= mc.TURN_DROPPED_S + 0.2 and st.mode == 'go' and not st.turn_stop


def test_a_stand_for_want_of_a_green_stays_through_a_faults_frames_and_a_reds_stand_when_its_turn_is_dropped():
  car = right_at_a_signal()
  car.right = False
  car.run_until(held, lights=no_heads)
  car.right, car.junction = None, None
  for _ in range(int(5. / DT_MDL)):  # turnpilotd faulted before it knew the route: the stops in force stay
    out = car.policy.update(None, False, True, car.v, False, no_heads(car), DT_MDL, turn=mc.KEEP_TURN, engaged=True)
    assert out.stops and out.stops[0].should_stop
  car = right_at_a_signal()
  car.right = False
  car.run_until(held, lights=cam_heads(('red', HEAD)))
  car.right = None
  for _ in range(int(20. / DT_MDL)):  # a red was read for it: the light's stop, whatever becomes of the turn
    out, dec, _ = car.step(lights=no_heads, junction=lambda c: (7, JUNCTION - c.s, False))
    assert held(out, dec)


def right_gone_on_red(reread_after_s: float, state: str = 'red'):
  """A right let go from its stand at a red gone unread, the red (or state) read again reread_after_s after its go. Returns
  (car, the lights from then on, m past its line and m/s at the re-read)."""
  car = right_at_a_signal()
  car.right = True
  car.run_until(held, lights=cam_heads(('red', HEAD)))
  car.run_until(lambda out, dec: not held(out, dec), max_s=20., lights=cam_heads(('unknown', HEAD)))
  assert car.policy.tracker.controls[(1, 0)].state.cause == 'rightOnRed'
  car.run(reread_after_s, lights=cam_heads(('unknown', HEAD)))
  return car, cam_heads((state, HEAD)), car.s - LINE, car.v


@pytest.mark.parametrize("state", ['red', 'yellow'])
def test_a_red_read_again_just_after_a_rights_go_on_red_stops_it_again(state):
  car, lit, past, v = right_gone_on_red(0.1 if state == 'yellow' else 0.4, state)
  assert past < 0.5 and v < 1.
  for _ in range(int(15. / DT_MDL)):
    out, dec, _ = car.step(lights=lit)
  st = car.policy.tracker.controls[(1, 0)].state
  assert held(out, dec) and car.s < LINE + mc.RED_PAST_M and not st.done and not car.policy.right_on_red(JUNCTION - car.s)
  assert out.stops[0].reason in (mc.RIGHT_RED_WAIT_REASON, mc.RED_PAST_REASON)
  # ...and once it goes unread again the right's procedure is its own again: stood, looked, gone
  car.run_until(lambda out, dec: not held(out, dec), max_s=20., lights=cam_heads(('unknown', HEAD)))
  assert st.cause == 'rightOnRed'


def test_a_red_read_again_after_a_rights_go_stops_it_again_at_the_cameras_own_signal_too():
  # a junction the map has no signal for: the camera's signal, let go on the right's go, is kept to RED_PAST_M past its
  # line and is the stop again on a red read there; then the right's procedure once more, and no second signal after it
  head = JUNCTION + 15.
  car = right(mapped=False)
  car.run_until(held, lights=cam_heads(('red', head)))
  cs = car.policy.camera_signal
  car.run_until(lambda out, dec: not held(out, dec), max_s=20., lights=cam_heads(('unknown', head)))
  assert cs.state.cause == 'rightOnRed' and car.policy.camera_signal is cs
  car.run(0.4, lights=cam_heads(('unknown', head)))
  for _ in range(int(15. / DT_MDL)):
    out, dec, _ = car.step(lights=cam_heads(('red', head)))
  assert held(out, dec) and car.policy.camera_signal is cs and not cs.state.done
  assert car.s < JUNCTION - mc.CONTROL_LINE_M + mc.RED_PAST_M
  car.run_until(lambda out, dec: not held(out, dec), max_s=20., lights=cam_heads(('unknown', head)))
  assert cs.state.cause == 'rightOnRed'
  for _ in range(int(10. / DT_MDL)):  # into its turn, the red read again: nothing stops it, no new signal is made
    out, dec, _ = car.step(lights=cam_heads(('red', head)) if car.s > JUNCTION - mc.CONTROL_LINE_M + mc.RED_PAST_M else cam_heads(('unknown', head)))
    assert not out.stops
  assert car.s > JUNCTION + 5. and car.policy.camera_signal is None


def test_a_red_read_again_once_the_right_is_past_stopping_short_finishes_the_turn():
  car, lit, past, v = right_gone_on_red(3.)
  assert v > mc.COMMIT_SPEED and not mc.red_stop_fits(v, -past)
  alerts = set()
  for _ in range(int(10. / DT_MDL)):
    out, dec, _ = car.step(lights=lit)
    alerts.add(out.alert)
    assert not out.stops
  assert car.s > JUNCTION + 10. and alerts == {'none'}


# --- the camera's steady green ---

def test_three_greens_in_one_burst_are_no_steady_green():
  g = GreenReads()
  for run, t in enumerate((0., 0.1, 0.2)):
    g.update(True, False, run, 0.1 if t else 0.)
  for _ in range(8):  # carried, then no colour, for the rest of the second
    g.update(False, False, 2, 0.1)
    assert not g.steady
  g.update(True, False, 3, 0.1)  # a read a second after the first: the reads span the time
  assert g.steady and g.first_t - g.last_t >= mc.CAMERA_GREEN_STEADY_S - 1e-9
  for _ in range(int(mc.CAMERA_STALE_S / 0.1)):
    g.update(False, False, 3, 0.1)
  assert g.steady
  g.update(False, False, 3, 0.1)  # the last read no longer one in hand
  assert not g.steady


def test_a_camera_signal_a_red_raised_outlives_the_route():
  # a red read at the route turn's unmapped junction, its stop decided and not yet engaged; no route for a spell (navd's
  # message invalid): the signal stays, dead-reckoned, and the car stops at it. The route back under a new maneuver id at
  # the same junction is that signal's again
  car = right(mapped=False)
  car.run(3., lights=cam_heads(('red', JUNCTION + 15.)))
  cs = car.policy.camera_signal
  assert cs is not None and not cs.engaged and cs.state.mode == 'stop'
  car.junction, car.right = None, None
  car.run(2.5, lights=cam_heads(('red', JUNCTION + 15.)))
  assert car.policy.camera_signal is cs and abs(cs.d - (JUNCTION - mc.CONTROL_LINE_M - car.s)) < 1.
  car.junction, car.right = (JUNCTION, 9), True
  car.step(lights=cam_heads(('red', JUNCTION + 15.)))
  assert car.policy.camera_signal is cs and cs.key == (mc.CAMERA_ROUTE, 9) and cs.state.turn_id == 9
  held_at = drive(car, cam_heads(('red', JUNCTION + 15.)), until_s=JUNCTION)
  assert held_at is not None and math.isclose(held_at, JUNCTION - mc.CONTROL_LINE_M, abs_tol=5.)
