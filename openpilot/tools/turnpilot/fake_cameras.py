import time

import numpy as np

from msgq.visionipc import VisionIpcServer
from openpilot.cereal import messaging
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

# comma four road cameras
W, H = 1344, 760
SENSOR = "os04c10"
STREAMS = ((VisionStreamType.VISION_STREAM_NARROW_ROAD, 'narrowRoadCameraState'),
           (VisionStreamType.VISION_STREAM_WIDE_ROAD, 'wideRoadCameraState'))


def gray_nv12(w: int = W, h: int = H, seed: int = 0) -> bytes:
  stride, y_height, uv_height, size = get_nv12_info(w, h)
  buf = np.full(size, 128, dtype=np.uint8)
  rng = np.random.default_rng(seed)
  buf[:stride * y_height] = rng.integers(64, 192, stride * y_height, dtype=np.uint8)
  return buf.tobytes()


class FakeCameras:
  """camerad stand-in: road and wide NV12 frames on VisionIPC plus their camera states, both stamped with one sof"""
  def __init__(self, w: int = W, h: int = H, sensor: str = SENSOR):
    self.sensor = sensor
    self.frame_id = 0
    self.pm = messaging.PubMaster([s for _, s in STREAMS])
    self.server = VisionIpcServer("camerad")
    stride, y_height, _, size = get_nv12_info(w, h)
    for stream, _ in STREAMS:
      self.server.create_buffers_with_sizes(stream, 5, w, h, size, stride, stride * y_height)
    self.server.start_listener()

  def send(self, road: bytes, wide: bytes) -> None:
    ts = time.monotonic_ns()
    for (stream, service), yuv in zip(STREAMS, (road, wide), strict=True):
      self.server.send(stream, yuv, self.frame_id, ts, ts)
      msg = messaging.new_message(service, valid=True)
      cs = getattr(msg, service)
      cs.frameId = self.frame_id
      cs.timestampSof = ts
      cs.timestampEof = ts
      cs.sensor = self.sensor
      self.pm.send(service, msg)
    self.frame_id += 1
