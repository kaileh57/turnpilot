"""Lane paint on synthetic road frames: a flat road rendered through the comma four road camera's real projection, in camerad's
padded NV12, with modelV2-style lane lines where the paint is."""
import time

import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.turnpilot.scene.lane_paint import (UNKNOWN, LanePaintClassifier, LinePaint, fill_lane_paint, lane_paint_from_msg,
                                                            project, runs, PAINT, ROAD, INVALID)
from openpilot.selfdrive.turnpilot.scene.ranging import calib_from_view, road_lens
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

LENS = road_lens(DEVICE_CAMERAS[('mici', 'os04c10')].narrow_road)
W, H = LENS.width, LENS.height
STRIDE, Y_HEIGHT, UV_HEIGHT, _ = get_nv12_info(W, H)
RPY = [0.0, 0.03, 0.0]
HEIGHT = 1.35
LINE_YS = (-5.4, -1.8, 1.8, 5.4)   # modelV2's four lines, y right
LINE_W = 0.12
ASPHALT, CONCRETE = (90., 128., 128.), (150., 122., 130.)
WHITE, YELLOW = (205., 128., 128.), (150., 96., 150.)
YELLOW_ON_CONCRETE = (138., 106., 142.)  # the demo route's freeway: darker than the concrete, low U, high V
V_EGO, DT = 15., 0.1


def _ground():
  """Ground (x forward, y right) under every pixel, nan above the horizon."""
  u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
  rays = np.stack([(u - LENS.center[0]) / LENS.f, (v - LENS.center[1]) / LENS.f, np.ones_like(u)], axis=-1) @ calib_from_view(RPY).T
  down = rays[..., 2] > 1e-3
  scale = np.where(down, HEIGHT / np.where(down, rays[..., 2], 1.), np.nan)
  return rays[..., 0] * scale, rays[..., 1] * scale


GX, GY = _ground()


def solid(color, y0=0., width=LINE_W):
  return lambda x, y: (np.abs(y - y0) < width / 2, color)


def dashed(color, dash=3., cycle=12., phase=0.):
  return lambda x, y: ((np.abs(y) < LINE_W / 2) & (np.mod(x + phase, cycle) < dash), color)


def double(color, gap=0.12):
  half = (LINE_W + gap) / 2
  return lambda x, y: ((np.abs(np.abs(y) - half) < LINE_W / 2), color)


def botts_dots():
  """California's Botts' dots: 10 cm white dots, groups of 4 at 0.91 m, a 7.3 m cycle."""
  def mark(x, y):
    k = np.mod(x, 7.3)
    near = np.min([np.hypot(k - 0.91 * j, y) for j in range(4)], axis=0)
    return near < 0.05, WHITE
  return mark


def render(marks, s, road=ASPHALT, gain=1., crosswalk=False, car=None, seed=0):
  """NV12 frame with the car at s m along the road. marks: per modelV2 line, f(x, y about the line) -> (mask, yuv) or None.
  car: (x0, x1, y0, y1) m of a dark box standing on the road, drawn as its pixel rectangle; returns (frame, its box)."""
  rng = np.random.default_rng(seed)
  img = np.empty((H, W, 3), np.float32)
  img[:] = road
  x = GX + s
  for y0, mark in zip(LINE_YS, marks, strict=True):
    if mark is not None:
      m, color = mark(x, GY - y0)
      img[m] = color
  if crosswalk:  # continental bars along the lanes, 0.6 m wide and apart, 3 m deep, 12-15 m ahead of where the car starts
    img[(np.mod(GY, 1.2) < 0.6) & (np.abs(x - 13.5) < 1.5)] = WHITE
  box = None
  if car is not None:
    pts = np.array([[cx, cy, HEIGHT - dz] for cx in car[:2] for cy in car[2:] for dz in (0., 1.5)])
    u, v, _ = project(LENS, calib_from_view(RPY).T, pts)
    box = np.array([[max(u.min(), 0), max(v.min(), 0), min(u.max(), W - 1), min(v.max(), H - 1)]])
    x0, y0, x1, y1 = box[0].astype(int)
    img[y0:y1, x0:x1] = (30., 128., 128.)
  img[np.isnan(GX)] = (170., 140., 120.)  # sky
  img[..., 0] *= gain
  img[..., 0] += rng.normal(0., 2., (H, W))
  img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
  out = np.zeros((Y_HEIGHT + UV_HEIGHT, STRIDE), np.uint8)
  out[:H, :W] = img[..., 0]
  uv = img[..., 1:].reshape(H // 2, 2, W // 2, 2, 2).astype(np.float32).mean(axis=(1, 3))
  out[Y_HEIGHT:Y_HEIGHT + H // 2, :W] = np.rint(uv).astype(np.uint8).reshape(H // 2, W)
  return out.reshape(-1), box


def lines_xyz(ys=LINE_YS):
  x = np.linspace(0., 100., 33)
  return [(x, np.full_like(x, y), np.full_like(x, HEIGHT)) for y in ys]


def drive(marks, seconds=2., probs=(0.9,) * 4, calibrated=True, boxes_at=None, **kw):
  clf = LanePaintClassifier(LENS)
  paint = [UNKNOWN] * 4
  for k in range(int(seconds / DT)):
    frame, box = render(marks, V_EGO * DT * k, seed=k, **kw)
    paint = clf.update(k * DT, frame, STRIDE, Y_HEIGHT, RPY, calibrated, lines_xyz(), list(probs), box if boxes_at is None else boxes_at)
  return paint


def kinds(paint):
  return [(p.kind, p.color) for p in paint]


class TestLanePaint(OpenpilotTestCase):
  def test_road_markings(self):
    paint = drive([double(YELLOW), dashed(WHITE), solid(WHITE), None])
    assert kinds(paint) == [('solid', 'yellow'), ('broken', 'white'), ('solid', 'white'), ('none', 'unknown')], paint
    assert paint[1].confidence > 0.8 and paint[2].confidence > 0.8 and paint[0].confidence > 0.6
    assert 0.2 < paint[1].coverage < 0.5 and paint[2].coverage > 0.9
    # on concrete: yellow no brighter than the concrete, Botts' dots, a dash
    paint = drive([solid(YELLOW_ON_CONCRETE), botts_dots(), dashed(WHITE, phase=5.), None], road=CONCRETE)
    assert kinds(paint)[:3] == [('solid', 'yellow'), ('broken', 'white'), ('broken', 'white')], paint

  def test_brightness(self):
    for gain in (0.45, 1.15):
      paint = drive([None, dashed(WHITE), solid(WHITE), double(YELLOW)], gain=gain)
      assert kinds(paint)[1:] == [('broken', 'white'), ('solid', 'white'), ('solid', 'yellow')], (gain, paint)

  def test_car_over_a_solid_line(self):
    car = (14., 18.5, 0.9, 2.7)  # a car astride the ego right line, 14-18.5 m ahead (it keeps pace)
    paint = drive([None, dashed(WHITE), solid(WHITE), None], car=car)
    assert kinds(paint)[1:3] == [('broken', 'white'), ('solid', 'white')], paint
    # without the detector's box the line under the car mustn't read broken
    paint = drive([None, dashed(WHITE), solid(WHITE), None], car=car, boxes_at=np.zeros((0, 6)))
    assert paint[2].kind != 'broken', paint

  def test_crosswalk_bars_are_not_a_broken_line(self):
    for s0 in (0., 6.):
      clf = LanePaintClassifier(LENS)
      for k in range(15):  # the car drives up to the crosswalk and over its near edge
        s = s0 + 1. * k
        frame, _ = render([None, None, solid(WHITE), None], s, crosswalk=True, seed=k)
        paint = clf.update(k * DT, frame, STRIDE, Y_HEIGHT, RPY, True, lines_xyz(), [0.9] * 4)
        assert paint[1].kind != 'broken' and paint[2].kind != 'broken', (s, paint)

  def test_unknown_when_unsure(self):
    marks = [None, dashed(WHITE), solid(WHITE), None]
    assert drive(marks, calibrated=False) == [UNKNOWN] * 4
    assert drive(marks, probs=(0.9, 0.2, 0.9, 0.9))[1] == UNKNOWN
    assert drive(marks, seconds=0.25) == [UNKNOWN] * 4  # too few frames yet
    # lines beyond the frame's edge
    clf = LanePaintClassifier(LENS)
    for k in range(20):
      frame, _ = render(marks, V_EGO * DT * k, seed=k)
      paint = clf.update(k * DT, frame, STRIDE, Y_HEIGHT, RPY, True, lines_xyz((-30., -1.8, 1.8, 30.)), [0.9] * 4)
    assert paint[0] == UNKNOWN and paint[3] == UNKNOWN and paint[2].kind == 'solid'
    # stale frames start over; so does a line that jumps sideways (the model renumbering lines in a lane change)
    assert clf.update(20 * DT + 1., frame, STRIDE, Y_HEIGHT, RPY, True, lines_xyz((-30., -1.8, 1.8, 30.)), [0.9] * 4)[2] == UNKNOWN
    clf = LanePaintClassifier(LENS)
    for k in range(20):
      frame, _ = render(marks, V_EGO * DT * k, seed=k)
      ys = (-30., -1.8, 1.8, 30.) if k < 19 else (-30., -1.8, -1.8 + 3.6, 30.)
      paint = clf.update(k * DT, frame, STRIDE, Y_HEIGHT, RPY, True, lines_xyz(), [0.9] * 4) if k < 19 else \
        clf.update(k * DT, frame, STRIDE, Y_HEIGHT, RPY, True, lines_xyz(ys), [0.9] * 4)
    assert paint[1].kind == 'broken'

  def test_runs(self):
    s = np.array([PAINT] * 12 + [ROAD] * 36 + [PAINT] * 12 + [ROAD] * 9)
    assert runs(s) == (3., True)
    assert runs(np.array([PAINT] * 12 + [ROAD] * 20 + [INVALID] + [ROAD] * 15 + [PAINT] * 12)) == (3., False)  # not closed
    assert runs(np.array([PAINT] * 10 + [ROAD] * 60)) == (2.5, False)
    assert runs(np.array([ROAD] * 69)) == (0., False)

  def test_message_round_trip_and_old_logs(self):
    from openpilot.cereal import messaging
    msg = messaging.new_message('turnpilotDetections')
    paint = [UNKNOWN, LinePaint('broken', 'white', 0.75, 0.3), LinePaint('solid', 'yellow', 0.5, 0.9), LinePaint('none', 'unknown', 1., 0.)]
    fill_lane_paint(msg.turnpilotDetections.init('lanePaint', 4), paint)
    back = lane_paint_from_msg(msg.as_reader().turnpilotDetections.lanePaint)
    assert [(p.kind, p.color) for p in back] == kinds(paint) and abs(back[1].confidence - 0.75) < 1e-6
    assert lane_paint_from_msg(messaging.new_message('turnpilotDetections').as_reader().turnpilotDetections.lanePaint) == (UNKNOWN,) * 4

  def test_cost(self):
    clf = LanePaintClassifier(LENS)
    frame, _ = render([double(YELLOW), dashed(WHITE), solid(WHITE), None], 0.)
    lines = lines_xyz()
    for _ in range(3):
      clf.update(0., frame, STRIDE, Y_HEIGHT, RPY, True, lines, [0.9] * 4)
    t0 = time.perf_counter()
    for k in range(20):
      clf.update(k * DT, frame, STRIDE, Y_HEIGHT, RPY, True, lines, [0.9] * 4)
    assert (time.perf_counter() - t0) / 20 < 0.01  # ~1 ms on an M-series Mac; generous for CI
