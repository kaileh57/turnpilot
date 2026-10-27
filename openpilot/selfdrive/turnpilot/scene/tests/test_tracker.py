import numpy as np

from openpilot.cereal import messaging
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.turnpilot.detector.decoder import CAR, PERSON, fill_detections
from openpilot.selfdrive.turnpilot.scene.ranging import calib_from_view
from openpilot.selfdrive.turnpilot.scene.tracker import PATH_X, SceneTracker, predicted_path

CAMS = DEVICE_CAMERAS[('mici', 'os04c10')]
RPY, WIDE_EULER, HEIGHT = [0.0, 0.03, 0.01], [0.0, 0.005, -0.003], 1.3
DT = 0.15  # the detector's period at every 3rd model frame
SIZES = {PERSON: (0.6, 1.7), CAR: (1.8, 1.5)}  # width, height in m


def calibration():
  msg = messaging.new_message('extrinsicsCalibration')
  cal = msg.extrinsicsCalibration
  cal.rpyCalib, cal.wideFromDeviceEuler, cal.height = RPY, WIDE_EULER, [HEIGHT]
  return msg.as_reader().extrinsicsCalibration


def car_state(v_ego=0.0, yaw_rate=0.0):
  msg = messaging.new_message('carState')
  msg.carState.vEgo, msg.carState.yawRate = v_ego, yaw_rate
  return msg.as_reader().carState


def box(tracker: SceneTracker, camera: str, cls: int, x: float, y: float):
  """The box an object standing at ground point (x forward, y left) makes in a camera."""
  lens = tracker.lenses[camera]
  rot = calib_from_view(RPY, WIDE_EULER if camera == 'wide' else None).T
  u, v = lens.project(rot @ np.array([x, -y, HEIGHT]))
  w, h = SIZES[cls]
  scale = lens.f / np.hypot(x, y)
  return [u - w * scale / 2, v - h * scale, u + w * scale / 2, v]


def detections(tracker, camera: str, t: float, objects, frame_id=0):
  """objects: (cls, x, y, conf). Returns a TurnpilotDetections reader at camera time t."""
  rows = np.array([[*box(tracker, camera, cls, x, y), conf, cls] for cls, x, y, conf in objects], dtype=np.float32).reshape(-1, 6)
  msg = messaging.new_message('turnpilotDetections')
  fill_detections(msg.turnpilotDetections, rows, frame_id, int(t * 1e9), camera, 0.005)
  return msg.as_reader().turnpilotDetections


def nearest_track(tracks, x, y):
  """The track id whose position (in a frame's snapshot) is nearest to (x, y)."""
  return min(tracks, key=lambda tid: np.hypot(tracks[tid]['x'][0] - x, tracks[tid]['x'][1] - y))


class TestSceneTracker(OpenpilotTestCase):
  def setUp(self):
    self.tracker = SceneTracker(CAMS, 'mici')
    self.cal, self.cs = calibration(), car_state()

  def run_frames(self, n, objects_at, camera='road', v_ego=0.0, t0=10.0):
    """objects_at(t) -> [(cls, x, y, conf)]. Returns a snapshot of the published tracks after each frame: {id: state}."""
    out = []
    for i in range(n):
      t = t0 + i * DT
      det = detections(self.tracker, camera(i) if callable(camera) else camera, t, objects_at(t - t0), i)
      tracks = self.tracker.update(det, self.cal, car_state(v_ego), now=t)
      out.append({trk.track_id: {'x': trk.x.copy(), 'camera': trk.camera, 'in_path': trk.in_path} for trk in tracks})
    return out

  def test_crossing_pedestrians_keep_ids(self):
    # A walks right to left at 12 m, B left to right at 16 m; their boxes cross in the image mid-way
    def objects(t):
      return [(PERSON, 12.0, -4.0 + 1.5 * t, 0.8), (PERSON, 16.0, 4.0 - 1.5 * t, 0.7)]
    history = self.run_frames(40, objects)
    ids = set()
    for i, tracks in enumerate(history[1:], start=1):
      assert len(tracks) == 2
      (_, xa, ya, _), (_, xb, yb, _) = objects(i * DT)
      ids.add((nearest_track(tracks, xa, ya), nearest_track(tracks, xb, yb)))
    assert len(ids) == 1 and len(next(iter(ids))) == len(set(next(iter(ids))))
    a, b = next(iter(ids))
    # walking speed over the ground, sideways
    np.testing.assert_allclose(history[-1][a]['x'][3], 1.5, atol=0.3)
    np.testing.assert_allclose(history[-1][b]['x'][3], -1.5, atol=0.3)

  def test_ego_motion_is_removed(self):
    # a parked car ahead while we drive at 10 m/s: it closes in, but over the ground it stands still
    def objects(t):
      return [(CAR, 50.0 - 10.0 * t, 0.0, 0.9)]
    history = self.run_frames(20, objects, v_ego=10.0)
    assert len(history[-1]) == 1
    trk = next(iter(history[-1].values()))
    np.testing.assert_allclose(trk['x'][0], 50.0 - 10.0 * 19 * DT, rtol=0.05)
    assert abs(trk['x'][2]) < 1.0 and abs(trk['x'][3]) < 0.5
    assert trk['in_path']

  def test_camera_switch_keeps_id(self):
    def objects(t):
      return [(PERSON, 9.0, 3.0 - 0.5 * t, 0.8)]
    history = self.run_frames(12, objects, camera=lambda i: 'road' if i < 6 else 'wide')
    ids = {tid for tracks in history[1:] for tid in tracks}
    assert len(ids) == 1 and all(len(tracks) == 1 for tracks in history[1:])
    assert history[-1][ids.pop()]['camera'] == 'wide'

  def test_staleness(self):
    det = detections(self.tracker, 'road', 20.0, [(CAR, 20.0, 0.0, 0.9)])
    self.tracker.update(det, self.cal, self.cs, now=20.0)
    assert self.tracker.available
    self.tracker.update(det, self.cal, self.cs, now=20.4)  # same message again: no new detections
    assert self.tracker.available
    self.tracker.update(det, self.cal, self.cs, now=20.6)
    assert not self.tracker.available
    msg = messaging.new_message('turnpilotScene')
    self.tracker.fill_scene(msg.turnpilotScene)
    assert not msg.turnpilotScene.available and len(msg.turnpilotScene.objects) == 0

  def test_fill_scene(self):
    def objects(t):
      return [(PERSON, 15.0, 0.5, 0.8), (CAR, 25.0, 6.0, 0.9)]
    self.run_frames(3, objects)
    msg = messaging.new_message('turnpilotScene')
    self.tracker.fill_scene(msg.turnpilotScene)
    scene = msg.as_reader().turnpilotScene
    assert scene.available and len(scene.objects) == 2
    by_cls = {str(o.cls): o for o in scene.objects}
    ped, car = by_cls['pedestrian'], by_cls['car']
    np.testing.assert_allclose((ped.x, ped.y), (15.0, 0.5), atol=0.3)
    np.testing.assert_allclose((car.x, car.y), (25.0, 6.0), atol=0.5)
    assert ped.inPath and not car.inPath
    assert all(0 <= v <= 1 for v in ped.box) and ped.box[2] > ped.box[0] and str(ped.camera) == 'road'

  def test_no_tracks_above_horizon(self):
    # a box whose bottom is in the sky can't be ranged, so it never starts a track
    for i in range(3):
      msg = messaging.new_message('turnpilotDetections')
      fill_detections(msg.turnpilotDetections, np.array([[600, 10, 700, 80, 0.9, CAR]], dtype=np.float32), i, int((1 + i * DT) * 1e9), 'road', 0)
      tracks = self.tracker.update(msg.as_reader().turnpilotDetections, self.cal, self.cs, now=1 + i * DT)
    assert tracks == [] and self.tracker.tracks == []

  def test_predicted_path(self):
    def model(xs, ys_right):
      msg = messaging.new_message('modelV2')
      msg.modelV2.position.x, msg.modelV2.position.y = xs, ys_right
      return msg.as_reader().modelV2
    x, y = predicted_path(None)
    assert x[-1] == PATH_X and not y.any()
    # stopped: the model path is a few cm long, so straight ahead
    x, y = predicted_path(model([0.0, 0.01, 0.02, 0.02], [0.0, 0.0, 0.0, 0.0]))
    assert x[-1] == PATH_X and not y.any()
    # a right curve that ends at 30 m carries on along its last heading; y is to the left
    xs = [float(v) for v in np.linspace(0, 30, 16)]
    x, y = predicted_path(model(xs, [0.002 * v ** 2 for v in xs]))
    assert x[-1] == PATH_X
    np.testing.assert_allclose(np.interp(20.0, x, y), -0.8, atol=0.05)
    assert y[-1] < np.interp(30.0, x, y) < 0

  def test_turning_ego_stationary_pedestrian(self):
    # we turn left at 0.3 rad/s and 5 m/s past a pedestrian standing still; with the model's yaw rate removed it stays
    # still over the ground, where ignoring the turn would show it crossing at about 3 m/s
    omega, v, world = 0.3, 5.0, np.array([12.0, 4.0])
    msg = messaging.new_message('modelV2')
    msg.modelV2.orientationRate.z = [-omega]  # the model's z is down
    model = msg.as_reader().modelV2
    for i in range(12):
      t = i * DT
      psi = omega * t
      pos = v / omega * np.array([np.sin(psi), 1 - np.cos(psi)])
      c, s = np.cos(psi), np.sin(psi)
      x, y = np.array([[c, s], [-s, c]]) @ (world - pos)
      det = detections(self.tracker, 'road', 10 + t, [(PERSON, float(x), float(y), 0.8)], i)
      tracks = self.tracker.update(det, self.cal, car_state(v), model, now=10 + t)
    assert len(tracks) == 1 and tracks[0].x is not None
    assert np.hypot(*tracks[0].x[2:]) < 0.7
