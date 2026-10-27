from collections import deque

import numpy as np

from openpilot.cereal import log
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.turnpilot.detector.decoder import PERSON
from openpilot.selfdrive.turnpilot.detector.graph import letterbox
from openpilot.selfdrive.turnpilot.detector.modeld_detector import RUN_EVERY, STEP_BUDGET_S, WIDE_HOLD_S, ModeldDetector
from openpilot.selfdrive.turnpilot.scene.lane_paint import UNKNOWN, LinePaint


class FakeOutput:
  def __init__(self, raw):
    self.raw = raw

  def numpy(self):
    return self.raw


class FakeRun:
  """Stands in for the compiled jit: records which frame it got, returns one person."""
  def __init__(self, lb):
    self.calls, self.fail = [], False
    self.raw = np.zeros((1, 300, 6), np.float32)
    self.raw[0, 0] = [*lb.to_input(100, 200), *lb.to_input(160, 400), 0.9, PERSON]

  def __call__(self, input_frame):
    if self.fail:
      raise RuntimeError("gpu fell over")
    self.calls.append(input_frame)
    return FakeOutput(self.raw)


class Meta:
  def __init__(self, frame_id):
    self.frame_id, self.timestamp_eof = frame_id, frame_id * 50_000_000


class FakeLanePaint:
  """Stands in for LanePaintReader: records the frame and boxes it got."""
  def __init__(self):
    self.calls, self.fail = [], False

  def read(self, frame, meta, boxes):
    if self.fail:
      raise RuntimeError("bad frame")
    self.calls.append((frame, meta.frame_id, None if boxes is None else len(boxes)))
    return [UNKNOWN, LinePaint('broken', 'white', 0.9, 0.3), LinePaint('solid', 'yellow', 0.8, 0.95), UNKNOWN]


class FakeModelState:
  frames = ['host0', 'host1']


class PM:
  def __init__(self):
    self.sent = []

  def send(self, name, msg):
    self.sent.append(msg.as_reader().turnpilotDetections)


def make_detector(cameras=None, det_time=0.005):
  det = ModeldDetector.__new__(ModeldDetector)  # skips loading a pkl
  det.lb = letterbox(1344, 760)
  det.run = FakeRun(det.lb)
  det.model_state = FakeModelState()
  det.cameras = cameras or {'road': 0, 'wide': 1}
  det.lane_paint = FakeLanePaint() if 'road' in det.cameras else None
  det.frames = {cam: f'frame{i}' for cam, i in det.cameras.items()}
  det.times = deque([det_time])
  det.since_run, det.wide_until, det.enabled = RUN_EVERY, -1.0, True
  return det


class TestModeldDetector(OpenpilotTestCase):
  def test_camera_choice(self):
    det = make_detector()
    assert det.choose_camera(0.001, log.Desire.none, 0.0) == 'road'
    assert det.choose_camera(-0.02, log.Desire.none, 1.0) == 'wide'
    assert det.choose_camera(0.0, log.Desire.none, 1.0 + WIDE_HOLD_S - 0.1) == 'wide'  # held after the turn
    assert det.choose_camera(0.0, log.Desire.none, 1.0 + WIDE_HOLD_S + 0.1) == 'road'
    assert det.choose_camera(0.0, log.Desire.turnRight, 10.0) == 'wide'
    assert det.choose_camera(0.0, log.Desire.laneChangeLeft, 20.0) == 'road'
    assert make_detector({'wide': 0}).choose_camera(0.0, log.Desire.none, 0.0) == 'wide'
    assert make_detector({'road': 0}).choose_camera(0.05, log.Desire.turnLeft, 0.0) == 'road'

  def test_every_nth_frame_and_message(self):
    det, pm = make_detector(), PM()
    for k in range(3 * RUN_EVERY):
      det.step(pm, det.model_state, 0.010, 0.0, log.Desire.none, [Meta(100 + k), Meta(200 + k)])
    assert len(pm.sent) == 3 and det.run.calls == ['frame0'] * 3
    td = pm.sent[-1]
    assert str(td.camera) == 'road' and td.frameId == 100 + 2 * RUN_EVERY and len(td.detections) == 1
    np.testing.assert_allclose([td.detections[0].x0, td.detections[0].y1], [100, 400], atol=1e-3)

  def test_wide_uses_extra_frame(self):
    det, pm = make_detector(), PM()
    det.step(pm, det.model_state, 0.010, 0.03, log.Desire.none, [Meta(1), Meta(2)])
    assert det.run.calls == ['frame1'] and str(pm.sent[0].camera) == 'wide' and pm.sent[0].frameId == 2

  def test_budget(self):
    det, pm = make_detector(det_time=0.006), PM()
    for _ in range(10):
      det.step(pm, det.model_state, STEP_BUDGET_S - 0.005, 0.0, log.Desire.none, [Meta(1), Meta(2)])
    assert pm.sent == []  # no room this step, every step
    det.step(pm, det.model_state, STEP_BUDGET_S - 0.007, 0.0, log.Desire.none, [Meta(1), Meta(2)])
    assert len(pm.sent) == 1  # the first frame with room runs, even if it's not on the N-frame grid

  def test_failures_turn_it_off(self):
    det, pm = make_detector(), PM()
    det.run.fail = True
    det.step(pm, det.model_state, 0.010, 0.0, log.Desire.none, [Meta(1), Meta(2)])
    assert not det.enabled and pm.sent == []
    det = make_detector()
    det.step(pm, object(), 0.010, 0.0, log.Desire.none, [Meta(1), Meta(2)])  # modeld fell back to the other model
    assert not det.enabled and pm.sent == []

  def test_lane_paint_rides_along_on_the_road_frame(self):
    det, pm = make_detector(), PM()
    det.step(pm, det.model_state, 0.010, 0.0, log.Desire.none, [Meta(7), Meta(8)])
    det.step(pm, det.model_state, 0.010, 0.0, log.Desire.none, [Meta(9), Meta(10)])  # not on the N-frame grid
    det.since_run = RUN_EVERY
    det.step(pm, det.model_state, 0.010, 0.03, log.Desire.none, [Meta(11), Meta(12)])  # the detector looks through the wide
    # the road camera's host frame, behind the detector's boxes on it; nothing while it looks through the wide camera, where
    # there are no boxes for the road frame
    assert det.lane_paint.calls == [('host0', 7, 1)]
    assert len(pm.sent) == 2 and str(pm.sent[1].camera) == 'wide' and len(pm.sent[1].lanePaint) == 0
    paint = pm.sent[0].lanePaint
    assert [(str(p.kind), str(p.color)) for p in paint] == [('unknown', 'unknown'), ('broken', 'white'), ('solid', 'yellow'),
                                                           ('unknown', 'unknown')]
    assert abs(paint[1].confidence - 0.9) < 1e-6

  def test_lane_paint_failure_leaves_the_detector_on(self):
    det, pm = make_detector(), PM()
    det.lane_paint.fail = True
    det.step(pm, det.model_state, 0.010, 0.0, log.Desire.none, [Meta(1), Meta(2)])
    assert det.enabled and det.lane_paint is None and len(pm.sent) == 1 and len(pm.sent[0].detections) == 1
    # no road camera, no paint
    det, pm = make_detector({'wide': 0}), PM()
    det.step(pm, det.model_state, 0.010, 0.0, log.Desire.none, [Meta(1), Meta(1)])
    assert len(pm.sent) == 1 and len(pm.sent[0].lanePaint) == 0


class TestLanePaintReader(OpenpilotTestCase):
  def test_reads_this_frames_lines_off_the_bus(self):
    from openpilot.cereal import messaging
    from openpilot.selfdrive.turnpilot.detector.modeld_detector import LanePaintReader
    from openpilot.selfdrive.turnpilot.scene.tests.test_lane_paint import DT, HEIGHT, LINE_YS, RPY, V_EGO, dashed, render, solid, WHITE
    pm = messaging.PubMaster(['modelV2', 'extrinsicsCalibration', 'deviceState', 'narrowRoadCameraState'])
    reader = LanePaintReader(1344, 760)
    msg = messaging.new_message('deviceState')
    msg.deviceState.deviceType = 'mici'
    pm.send('deviceState', msg)
    msg = messaging.new_message('narrowRoadCameraState')
    msg.narrowRoadCameraState.sensor = 'os04c10'
    pm.send('narrowRoadCameraState', msg)
    msg = messaging.new_message('extrinsicsCalibration')
    msg.extrinsicsCalibration.rpyCalib, msg.extrinsicsCalibration.calStatus = RPY, 'calibrated'
    pm.send('extrinsicsCalibration', msg)
    x = np.linspace(0., 100., 33)
    paint = None
    for k in range(20):
      msg = messaging.new_message('modelV2')
      msg.modelV2.frameId = 100 + k
      for line, y in zip(msg.modelV2.init('laneLines', 4), LINE_YS, strict=True):
        line.x, line.y, line.z = x.tolist(), [y] * len(x), [HEIGHT] * len(x)
      msg.modelV2.laneLineProbs = [0.9] * 4
      pm.send('modelV2', msg)
      frame, _ = render([None, dashed(WHITE), solid(WHITE), None], V_EGO * DT * k, seed=k)
      meta = Meta(100 + k)
      meta.timestamp_eof = int(k * DT * 1e9)
      if k == 19:
        assert reader.read(frame, Meta(99), None) == [UNKNOWN] * 4  # another frame's lines
      paint = reader.read(frame, meta, None)
    assert [(p.kind, p.color) for p in paint[1:3]] == [('broken', 'white'), ('solid', 'white')], paint
