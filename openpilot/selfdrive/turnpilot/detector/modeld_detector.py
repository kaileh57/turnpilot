"""The detector inside modeld: after modelV2 is sent, on frames modeld already uploaded, it runs YOLO26 on one camera and
publishes turnpilotDetections. The road camera sees far ahead on straights; the wide one sees the crosswalk and cross
traffic in turns. It runs in modeld because the chestnut's USB interface is claimed by one process.

Nothing here runs unless TurnPilotEnabled is set and the compiled pkl for this device and camera size exists
(compile_detector.py builds it; the pkls are not in git).
"""
import time

import numpy as np

from openpilot.cereal import log, messaging
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.turnpilot.detector.decoder import decode, fill_detections
from openpilot.selfdrive.turnpilot.detector.graph import detector_pkl_path, letterbox

RUN_EVERY = 3              # at most every 3rd model frame (6.7 Hz)
STEP_BUDGET_S = 0.045      # driving model + detector must finish within this of the 50 ms cycle
WIDE_CURVATURE = 0.01      # 1/m (a 100 m radius): at or above this, look through the wide camera
WIDE_HOLD_S = 2.0          # and keep looking through it this long after the turn ends
TURN_DESIRES = (log.Desire.turnLeft, log.Desire.turnRight)
TIME_FILTER = 0.2          # weight of the latest run in the detector time estimate


def frame_view(model_state, index: int):
  """A uint8 device view of packed frame `index` (0 main camera, 1 extra) in the input modeld uploads every step."""
  from tinygrad import Tensor, dtypes
  from tinygrad.uop.ops import UOp
  size = model_state.frame_copy_size
  offset = model_state.packed_input.nbytes - 2 * size + index * size
  view = model_state.input_device.view(size, offset).ensure_allocated()
  return Tensor(UOp.from_buffer(view, dtypes.uint8))


class ModeldDetector:
  def __init__(self, model_state, cam_w: int, cam_h: int, cameras: dict[str, int]):
    """cameras: 'road' and/or 'wide' -> index of its frame in modeld's packed input."""
    from tinygrad.engine.realize import lower_and_compile
    from openpilot.selfdrive.modeld.helpers import load_oob
    art = load_oob(detector_pkl_path(cam_w, cam_h, model_state.chestnut), model_state.chestnut)
    if art['input_specs']['input_frame'][0] != (model_state.frame_copy_size,):
      raise ValueError(f"detector compiled for frames of {art['input_specs']['input_frame'][0]}, modeld has {model_state.frame_copy_size}")
    self.run = art['run']
    self.run.captured._linear = lower_and_compile(self.run.captured._linear)
    self.lb = letterbox(cam_w, cam_h, tuple(art['metadata']['input_size']))
    self.model_state = model_state
    self.cameras = cameras
    self.frames = {cam: frame_view(model_state, i) for cam, i in cameras.items()}
    self.det_time = 0.0
    self.since_run = RUN_EVERY
    self.wide_until = -1.0
    self.enabled = True

  def choose_camera(self, desired_curvature: float, desire: int, now: float) -> str:
    if abs(desired_curvature) >= WIDE_CURVATURE or desire in TURN_DESIRES:
      self.wide_until = now + WIDE_HOLD_S
    want = 'wide' if now < self.wide_until else 'road'
    return want if want in self.frames else next(iter(self.frames))

  def step(self, pm, model_state, model_execution_time: float, desired_curvature: float, desire: int, metas: list) -> None:
    """Called once per model frame after modelV2 is sent. metas: FrameMeta of packed frames 0 and 1."""
    if not self.enabled:
      return
    if model_state is not self.model_state:
      cloudlog.warning("turnpilot detector: driving model switched devices, detector off for this drive")
      self.enabled = False
      return
    self.since_run += 1
    if self.since_run < RUN_EVERY or model_execution_time + self.det_time > STEP_BUDGET_S:
      return
    try:
      cam = self.choose_camera(desired_curvature, desire, time.monotonic())
      t0 = time.perf_counter()
      raw = self.run(input_frame=self.frames[cam]).numpy()
      dt = time.perf_counter() - t0
      self.det_time = dt if self.det_time == 0 else (1 - TIME_FILTER) * self.det_time + TIME_FILTER * dt
      self.since_run = 0
      meta = metas[self.cameras[cam]]
      msg = messaging.new_message('turnpilotDetections', valid=bool(np.isfinite(raw).all()))
      fill_detections(msg.turnpilotDetections, decode(raw, self.lb), meta.frame_id, meta.timestamp_eof, cam, dt)
      pm.send('turnpilotDetections', msg)
    except Exception:
      cloudlog.exception("turnpilot detector failed, off for this drive")
      self.enabled = False


def load(model_state, cam_w: int, cam_h: int, main_wide_camera: bool, use_extra_client: bool) -> ModeldDetector | None:
  """The detector, or None (modeld then runs exactly as without TurnPilot). modeld only calls this with TurnPilotEnabled set."""
  try:
    path = detector_pkl_path(cam_w, cam_h, model_state.chestnut)
    if not path.is_file():
      cloudlog.warning(f"turnpilot detector: no {path.name}, detector off")
      return None
    cameras = {'wide': 0} if main_wide_camera else ({'road': 0, 'wide': 1} if use_extra_client else {'road': 0})
    return ModeldDetector(model_state, cam_w, cam_h, cameras)
  except Exception:
    cloudlog.exception("turnpilot detector failed to load, detector off")
    return None
