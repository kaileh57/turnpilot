"""RawBox.motion: the road camera's raw boxes (full pass and far crop) associated run to run, their over-ground motion fitted
with the car's own taken out (tracker.BoxMotion)."""
import math

import numpy as np

from openpilot.cereal import messaging
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.turnpilot.detector.decoder import CAR
from openpilot.selfdrive.turnpilot.scene.ranging import calib_from_view
from openpilot.selfdrive.turnpilot.scene.tracker import MOTION_MIN_SPAN_S, SceneTracker

CAMS = DEVICE_CAMERAS[('mici', 'os04c10')]
RPY, HEIGHT = [0.0001, -0.0554, 0.0112], 1.238
FAR_CROP = (256, 188, 896, 572)
CAR_W, CAR_H = 1.8, 1.5

# a held left at night (a drive on the test car): the car standing at its line, the far crop and the road camera's full pass
# alternating. Road camera pixels of the car boxes under the horizon's band, per run: (s, view, boxes). The 428-515 box is a car
# parked at the near end of the far curb, 536-576 and 504-547 two more parked along it, 704-737 a car standing up the road
# right of straight ahead; each read as a far blocker on every far run of that hold
HELD_LEFT = [
  (0.000, 'far', [(428, 422, 513, 480), (536, 428, 575, 466), (504, 431, 546, 469), (704, 438, 737, 456)]),
  (0.200, 'full', [(428, 419, 515, 479)]),
  (0.300, 'far', [(429, 422, 511, 479), (504, 431, 546, 469), (536, 429, 576, 465)]),
  (0.400, 'full', [(426, 419, 516, 480)]),
  (0.450, 'far', [(428, 422, 512, 480), (535, 428, 575, 465), (504, 431, 545, 469)]),
  (0.650, 'full', [(426, 419, 518, 479)]),
  (0.750, 'far', [(429, 422, 511, 480), (538, 428, 576, 464), (505, 430, 546, 468), (704, 437, 736, 455)]),
  (0.850, 'full', [(426, 419, 517, 480)]),
  (0.950, 'far', [(428, 422, 510, 480), (506, 431, 547, 469), (537, 428, 576, 464), (704, 439, 737, 456)]),
  (1.100, 'full', [(427, 419, 516, 479)]),
  (1.200, 'far', [(428, 422, 513, 479), (536, 428, 575, 465), (705, 438, 735, 456), (503, 431, 545, 469)]),
  (1.300, 'full', [(426, 419, 515, 480)]),
  (1.350, 'far', [(428, 422, 513, 479), (537, 428, 576, 465), (503, 431, 546, 469)]),
  (1.550, 'full', [(426, 419, 515, 480)]),
  (1.650, 'far', [(428, 422, 512, 479), (506, 430, 547, 468), (705, 438, 736, 455), (536, 428, 575, 466)]),
  (1.750, 'full', [(427, 419, 514, 479)]),
  (1.850, 'far', [(429, 422, 511, 480), (506, 430, 547, 468), (538, 428, 576, 464), (705, 438, 735, 455)]),
  (2.050, 'full', [(427, 419, 518, 479)]),
  (2.150, 'far', [(429, 422, 511, 479), (538, 428, 576, 464), (506, 430, 547, 468), (704, 439, 736, 455)]),
  (2.250, 'full', [(427, 419, 518, 479)]),
  (2.350, 'far', [(428, 422, 513, 481), (704, 437, 736, 456), (505, 431, 545, 469), (537, 428, 575, 464)]),
  (2.500, 'full', [(426, 419, 517, 479)]),
  (2.600, 'far', [(429, 422, 512, 479), (537, 428, 576, 465), (507, 430, 546, 467), (705, 439, 736, 455)]),
  (2.700, 'full', [(426, 419, 514, 480)]),
  (2.750, 'far', [(429, 421, 512, 480), (536, 428, 575, 466), (706, 438, 735, 455), (505, 430, 546, 469)]),
  (2.950, 'full', [(427, 419, 519, 479)]),
]


def calibration():
  msg = messaging.new_message('extrinsicsCalibration')
  cal = msg.extrinsicsCalibration
  cal.rpyCalib, cal.wideFromDeviceEuler, cal.height = RPY, [0., 0., 0.], [HEIGHT]
  cal.calStatus = 'calibrated'
  return msg.as_reader().extrinsicsCalibration


def car_state(v_ego=0.):
  msg = messaging.new_message('carState')
  msg.carState.vEgo = v_ego
  return msg.as_reader().carState


def run_msg(boxes, t, view='full', frame_id=0):
  msg = messaging.new_message('turnpilotDetections')
  td = msg.turnpilotDetections
  td.frameId, td.timestampEof, td.camera, td.view = frame_id, int(t * 1e9), 'road', view
  for d, (x0, y0, x1, y1) in zip(td.init('detections', len(boxes)), boxes, strict=True):
    d.cls, d.conf, d.x0, d.y0, d.x1, d.y1 = CAR, 0.6, float(x0), float(y0), float(x1), float(y1)
  return msg.as_reader().turnpilotDetections


def far_msg(boxes, t, frame_id=0):
  msg = messaging.new_message('turnpilotFarDetections')
  fd = msg.turnpilotFarDetections
  fd.crop = [float(v) for v in FAR_CROP]
  r = fd.run
  r.frameId, r.timestampEof, r.camera, r.view = frame_id, int(t * 1e9), 'road', 'far'
  for d, (x0, y0, x1, y1) in zip(r.init('detections', len(boxes)), boxes, strict=True):
    d.cls, d.conf, d.x0, d.y0, d.x1, d.y1 = CAR, 0.6, float(x0), float(y0), float(x1), float(y1)
  return msg.as_reader().turnpilotFarDetections


def box_of(x, y_left, lens, rot):
  """Road camera pixels of a car whose rear (or front) centre is on the ground at (x forward, y left) m from the camera."""
  u, v = lens.project(rot.T @ np.array([x, -y_left, HEIGHT]))
  half = lens.f * CAR_W / 2 / x
  top = lens.project(rot.T @ np.array([x, -y_left, HEIGHT - CAR_H]))[1]
  return (u - half, top, u + half, v)


def drive(tracker, cal, path, v_ego=0., hz=8., seconds=2.5, noise_px=0., seed=0):
  """Runs of one car on path(t) -> (x, y_left), the full pass and the far crop alternating at hz. Returns [(t, RawBox)]."""
  rng = np.random.default_rng(seed)
  lens, rot = tracker.lenses['road'], calib_from_view(RPY)
  out = []
  for k in range(int(seconds * hz) + 1):
    t = 100. + k / hz
    box = np.array(box_of(*path(t - 100.), lens, rot)) + rng.normal(0., noise_px, 4)
    if k % 2:
      tracker.update_far(far_msg([box], t, k), cal)
      frame = tracker.raw_far
    else:
      tracker.update(run_msg([box], t, frame_id=k), cal, car_state(v_ego), now=t, yaw_rate=0.)
      frame = tracker.raw
    out.append((t, frame.boxes[0]))
  return out


def test_held_left_far_curb_cars_read_standing():
  tracker, cal = SceneTracker(CAMS, 'mici'), calibration()
  last = {}
  for k, (t, view, boxes) in enumerate(HELD_LEFT):
    t += 100.
    if view == 'far':
      tracker.update_far(far_msg(boxes, t, k), cal)
      frame = tracker.raw_far
    else:
      tracker.update(run_msg(boxes, t, frame_id=k), cal, car_state(0.), now=t, yaw_rate=0.)
      frame = tracker.raw
    for b in frame.boxes:
      x0 = b.box[0] * CAMS.narrow_road.width
      last[min((428, 505, 536, 705), key=lambda a: abs(a - x0))] = (t - 100., b)
  # every box seen in the last second: the parked cars read standing; the car up the road (115 m out, its fit scatters a few
  # m/s) standing or not told, never closing, and never receding on that scatter (it stands: MOTION_AWAY_SIGMA)
  assert len(last) == 4
  for key, (t, b) in last.items():
    assert t > HELD_LEFT[-1][0] - 1., key
    want = ('standing', 'unknown') if key == 705 else ('standing',)
    assert b.motion in want, (key, b.motion, b.radial_mps, b.lateral_mps)


def test_unknown_until_enough_sightings():
  tracker, cal = SceneTracker(CAMS, 'mici'), calibration()
  seen = drive(tracker, cal, lambda t: (60., 7.), seconds=1.5)
  for t, b in seen:
    if t - 100. < MOTION_MIN_SPAN_S - 1e-6:
      assert b.motion == 'unknown'
  assert seen[-1][1].motion == 'standing'


def test_oncoming_car_reads_closing():
  tracker, cal = SceneTracker(CAMS, 'mici'), calibration()
  seen = drive(tracker, cal, lambda t: (150. - 15. * t, 5.), noise_px=0.5)
  b = seen[-1][1]
  assert b.motion == 'closing' and b.radial_mps < -10., (b.motion, b.radial_mps)


def test_same_direction_car_reads_receding():
  tracker, cal = SceneTracker(CAMS, 'mici'), calibration()
  seen = drive(tracker, cal, lambda t: (90. + 10. * t, -1.), noise_px=0.5)
  b = seen[-1][1]
  assert b.motion == 'receding' and b.radial_mps > 6., (b.motion, b.radial_mps)


def test_crossing_car_never_reads_standing():
  # no radial speed at all: only its lateral speed tells it from a parked car
  for y0, vy in ((-12., 10.), (12., -10.), (-6., 4.)):
    tracker, cal = SceneTracker(CAMS, 'mici'), calibration()
    seen = drive(tracker, cal, lambda t, y0=y0, vy=vy: (40., y0 + vy * t), seconds=2.)
    assert all(b.motion != 'standing' for _, b in seen), (y0, vy, [(b.motion, b.lateral_mps) for _, b in seen])


def test_moving_car_clears_nothing():
  # the car itself at 10 m/s passing a parked car: its own motion is taken out, but moving nothing reads standing or receding
  tracker, cal = SceneTracker(CAMS, 'mici'), calibration()
  seen = drive(tracker, cal, lambda t: (80. - 10. * t, 6.), v_ego=10.)
  assert all(b.motion in ('unknown', 'closing') for _, b in seen)


def test_one_noisy_fit_keeps_a_parked_car_standing():
  tracker, cal = SceneTracker(CAMS, 'mici'), calibration()
  seen = drive(tracker, cal, lambda t: (110., -3.5), seconds=6., noise_px=0.7, seed=3)
  late = [b.motion for t, b in seen if t - 100. >= 2.]
  assert late.count('standing') >= 0.9 * len(late), late


def test_wide_and_side_boxes_have_no_motion():
  tracker, cal = SceneTracker(CAMS, 'mici'), calibration()
  for k in range(12):
    t = 100. + k * 0.125
    msg = messaging.new_message('turnpilotDetections')
    td = msg.turnpilotDetections
    td.frameId, td.timestampEof, td.camera, td.view = k, int(t * 1e9), 'wide', 'full'
    d = td.init('detections', 1)[0]
    d.cls, d.conf, d.x0, d.y0, d.x1, d.y1 = CAR, 0.6, 600., 400., 700., 470.
    tracker.update(msg.as_reader().turnpilotDetections, cal, car_state(0.), now=t, yaw_rate=0.)
  assert tracker.raw.boxes[0].motion == 'unknown' and tracker.raw.boxes[0].radial_mps is None
  assert math.isfinite(tracker.raw.boxes[0].bearing_deg)


def launch_after_stand(t, x0=35., stand_s=3., accel=2.):
  """A car standing x0 m out in the oncoming lanes, then launching toward the car at accel m/s^2."""
  dt = max(0., t - stand_s)
  return x0 - 0.5 * accel * dt * dt, 5.


def test_a_stopped_oncoming_car_that_launches_leaves_standing_quickly():
  # an oncoming queue launching as the light turns: the window's slope alone kept it standing 1.8-2.1 s past its launch at
  # 29-37 m on the logged cars' noise, so a left consuming 'standing' would go into it. A step in range and width ends it
  # inside 2 s here (2.75 s on the slope alone)
  for seed in range(4):
    tracker, cal = SceneTracker(CAMS, 'mici'), calibration()
    seen = drive(tracker, cal, launch_after_stand, seconds=6., noise_px=0.5, seed=seed)
    assert any(b.motion == 'standing' for t, b in seen if t - 100. < 3.), seed
    left = next(t - 100. - 3. for t, b in seen if t - 100. > 3. and b.motion != 'standing')
    assert left <= 2.0, (seed, left)
    assert all(b.motion != 'standing' for t, b in seen if t - 100. >= 3. + left), seed


# --- the car itself moving (a left's roll-in): its own travel and turn out of every sighting ---

def roll_in(tracker, cal, world, v0=8., decel=0., yaw_rate=0., hz=8., seconds=3., noise_px=0.5, seed=0, short=1.):
  """Runs of one car at world(t) -> (x, y left) on the ground while ours drives from the origin at v0 (m/s, braking decel
  m/s^2, turning yaw_rate rad/s left), the full pass and the far crop alternating. short: the share of its true range the
  ground reads it at (the road's grade, the car's pitch: a box bottom lower or higher in the frame). Returns [(t, RawBox, v)]."""
  rng = np.random.default_rng(seed)
  lens, rot = tracker.lenses['road'], calib_from_view(RPY)
  x = y = yaw = 0.
  out, dt = [], 1. / hz
  for k in range(int(seconds * hz) + 1):
    t = 100. + k * dt
    v = max(v0 - decel * k * dt, 0.)
    wx, wy = world(k * dt)
    fx, fy = (wx - x) * math.cos(yaw) + (wy - y) * math.sin(yaw), -(wx - x) * math.sin(yaw) + (wy - y) * math.cos(yaw)
    box = np.array(box_of(fx, fy, lens, rot))
    box[3] = lens.project(rot.T @ np.array([fx * short, -fy * short, HEIGHT]))[1]  # its bottom as the ground range reads it
    box = box + rng.normal(0., noise_px, 4)
    if k % 2:
      tracker.update_far(far_msg([box], t, k), cal)
      frame = tracker.raw_far
    else:
      tracker.update(run_msg([box], t, frame_id=k), cal, car_state(v), now=t, yaw_rate=yaw_rate)
      frame = tracker.raw
    out.append((t, frame.boxes[0], v))
    v_next = max(v0 - decel * (k + 1) * dt, 0.)
    x, y, yaw = x + 0.5 * (v + v_next) * dt * math.cos(yaw), y + 0.5 * (v + v_next) * dt * math.sin(yaw), yaw + yaw_rate * dt
  return out


def test_a_car_parked_up_the_road_reads_standing_on_the_roll_in():
  # braking from 6 m/s toward a left's line, a car parked 60 m up the far curb: standing by the end of the window
  # (it read unknown on every run while the car had moved faster than 3 m/s at any sighting of the fit)
  tracker, cal = SceneTracker(CAMS, 'mici'), calibration()
  seen = roll_in(tracker, cal, lambda t: (60., 8.), v0=6., decel=1., seconds=3.)
  late = [b.motion for t, b, v in seen if t - 100. >= 2.]
  assert late.count('standing') >= 0.7 * len(late), [(b.motion, b.radial_mps) for _, b, _ in seen]
  assert all(abs(b.radial_mps) < 1.5 for t, b, v in seen if t - 100. >= 2.)


def test_a_car_ahead_keeping_pace_reads_receding_at_any_speed_of_ours():
  for v in (5., 8., 12.):
    tracker, cal = SceneTracker(CAMS, 'mici'), calibration()
    seen = roll_in(tracker, cal, lambda t, v=v: (45. + v * t, -0.5), v0=v, seconds=3.)
    late = [b.motion for t, b, _ in seen if t - 100. >= 2.]
    assert late.count('receding') >= 0.7 * len(late), (v, [(b.motion, b.radial_mps) for _, b, _ in seen])


def oncoming_fits(v0, u, short, x0=150., decel=0., yaw_rate=0., seed=0):
  tracker, cal = SceneTracker(CAMS, 'mici'), calibration()
  return roll_in(tracker, cal, lambda t: (x0 - u * t, 5.), v0=v0, decel=decel, yaw_rate=yaw_rate, short=short,
                 seconds=min(3., (x0 - 8.) / (u + v0)), seed=seed)


def test_an_oncoming_car_is_never_receding_whatever_our_speed():
  # receding clears a box anywhere in the band: never said of a car coming, from a crawl to 20 m/s, first seen 150 m or 80 m
  # out, ours at rest or at up to 12 m/s (steady or braking, straight or on a bend), the ground's range true, 20% long or
  # up to 30% short, three seeds of box noise each
  for x0 in (150., 80.):
    for v0 in (0., 2., 4., 6., 8., 10., 12.):
      for decel in (0., 1.5):
        for u in (0.5, 1., 2., 3., 5., 8., 12., 20.):
          for short in (0.7, 0.8, 1., 1.2):
            for yaw_rate in (0., 0.03):
              for seed in range(3):
                seen = oncoming_fits(v0, u, short, x0, decel, yaw_rate, seed)
                wrong = [(round(t - 100., 2), b.radial_mps) for t, b, _ in seen if b.motion == 'receding']
                assert not wrong, (x0, v0, decel, u, short, yaw_rate, seed, wrong)


def test_an_oncoming_car_is_never_standing_while_ours_moves():
  # ...nor standing, ours at 4-12 m/s: at 5 m/s or more with its range true, long or 20% short (the parked cars the drives
  # passed read 0.82-0.95 of their range at 20-50 m), at 8 m/s or more read 30% short. Slower than that it may read standing,
  # as it does with ours at rest, and a box standing in the oncoming lanes holds the left all the same (unprotected_left)
  for x0 in (150., 80.):
    for v0 in (4., 6., 8., 10., 12.):
      for decel in (0., 1.5):
        for u, shorts in ((5., (0.8, 1., 1.2)), (8., (0.7, 0.8, 1., 1.2)), (12., (0.7, 0.8, 1., 1.2)), (20., (0.7, 0.8, 1., 1.2))):
          for short in shorts:
            for yaw_rate in (0., 0.03):
              for seed in range(3):
                seen = oncoming_fits(v0, u, short, x0, decel, yaw_rate, seed)
                wrong = [(round(t - 100., 2), b.radial_mps) for t, b, _ in seen if b.motion == 'standing']
                assert not wrong, (x0, v0, decel, u, short, yaw_rate, seed, wrong)


def test_a_first_fit_of_a_far_car_is_never_receding():
  # one second of sightings of a car standing 150 m out fits its radial speed to several m/s either way: no receding on it
  for seed in range(8):
    tracker, cal = SceneTracker(CAMS, 'mici'), calibration()
    seen = drive(tracker, cal, lambda t: (150., 5.), seconds=3., noise_px=0.7, seed=seed)
    assert all(b.motion != 'receding' for _, b in seen), (seed, [(b.motion, b.radial_mps) for _, b in seen])


def test_the_cars_own_turn_is_taken_out():
  # a parked car while ours turns at 0.2 rad/s at 4 m/s: no lateral speed of its own, never closing nor receding
  tracker, cal = SceneTracker(CAMS, 'mici'), calibration()
  seen = roll_in(tracker, cal, lambda t: (50., 3.), v0=4., yaw_rate=0.1, seconds=3.)
  late = [b for t, b, _ in seen if t - 100. >= 2.]
  assert all(b.motion in ('standing', 'unknown') for b in late), [(b.motion, b.radial_mps, b.lateral_mps) for b in late]
  assert all(abs(b.lateral_mps) < 1.5 for b in late), [b.lateral_mps for b in late]
