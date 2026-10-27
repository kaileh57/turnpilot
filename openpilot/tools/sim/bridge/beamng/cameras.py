"""comma four road and wide cameras from BeamNG.tech camera sensors, published like camerad.

Both cameras sit at one mount, level and pointing straight ahead, at the os04c10's 1344x760 output. BeamNG renders a
camera when its requested update time has passed, on the game frame; with an update time below the game's frame time
both cameras render on every frame, the same frame. A poll returns the newest render, and the game answers it on its main
thread, so every poll (3 MB) costs game frame time: polling each camera every 10 ms on its own connection dropped the game
to 9 fps. So both cameras are polled in one pipelined burst on their own connection (one BeamNGpy connection is not
thread safe, so it has its own thread), which also puts both answers on the same game frame, a bit faster than 20 Hz.
A 20 Hz publisher drops a pair unless both frames are new (by hash) and stamps both with one time for modeld's sync check.
"""
import hashlib
import threading
import time
from collections import deque
from dataclasses import dataclass

import msgpack
import numpy as np
from beamngpy import BeamNGpy
from beamngpy.sensors import Camera

import openpilot.cereal.messaging as messaging
from msgq.visionipc import VisionIpcServer
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.common.realtime import Ratekeeper
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.tools.sim.bridge.beamng.yuv import NV12Converter

W, H = 1344, 760  # what a comma four's camerad outputs from the os04c10
ROAD_FOV_Y = 36.82  # deg, common/transformations/camera.py _os_config: 2 atan(380 / 1141.5)
WIDE_FOV_Y = 83.57  # deg, 2 atan(380 / 425.25); a pinhole stand-in for the fisheye, which is what modeld's warp assumes
MOUNT = (0.0, -0.55, 1.3)  # etk800 vehicle frame (-y forward): top center of the windshield, 1.34 m above the road
REPUBLISH_AFTER = 0.15  # s; at a standstill BeamNG can render identical frames, and modeld must not starve


def bng_send(conn, **msg) -> int:
  req_id, packed = conn._pack_data(msg)
  conn.skt.send(packed)
  return req_id


def bng_recv(conn, req_id: int | None = None):
  # beamngpy's own recv tries to utf-8 decode every bytes field: a 3 MB attempt per frame, and a black frame comes back as str
  msg = msgpack.unpackb(conn.skt.recv(), raw=False, strict_map_key=False)
  assert req_id is None or msg.get('_id') == req_id, f"out of order reply {msg.get('_id')} != {req_id}"
  return msg


def fingerprint(buf) -> bytes:
  return hashlib.blake2b(memoryview(buf)[::997].tobytes(), digest_size=8).digest()


@dataclass
class Frame:
  seq: int
  rgb: np.ndarray  # (h, w, 3 or 4)
  t_lo: float  # rendered after the poll before this one started...
  t_hi: float  # ...and before this poll returned (time.monotonic)

  @property
  def t(self) -> float:
    return (self.t_lo + self.t_hi) / 2


class CameraStream:
  """Polls BeamNG cameras in one burst on their own connection and keeps each camera's newest distinct frame."""

  def __init__(self, host: str, port: int, names, period: float, size=(W, H)):
    self.bng = BeamNGpy(host, port).open(launch=False)
    self.names, self.period, (self.w, self.h) = list(names), period, size
    self.latest: list[Frame | None] = [None] * len(self.names)
    self.polls = 0

  def run(self, exit_event: threading.Event):
    conn = self.bng.connection
    seq, prev_fp, t_prev = [0] * len(self.names), [None] * len(self.names), time.monotonic()
    while not exit_event.is_set():
      t0 = time.monotonic()
      ids = [bng_send(conn, type="PollCamera", name=n, isUsingSharedMemory=False) for n in self.names]
      bufs = [bng_recv(conn, i)["data"].get("colour") for i in ids]
      t1 = time.monotonic()
      self.polls += 1
      for i, buf in enumerate(bufs):
        # periodic polls come back RGB, ad-hoc ones RGBA
        if isinstance(buf, bytes) and len(buf) in (self.w * self.h * 3, self.w * self.h * 4):
          fp = fingerprint(buf)
          if fp != prev_fp[i]:
            seq[i] += 1
            self.latest[i] = Frame(seq[i], np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, -1), t_prev, t1)
            prev_fp[i] = fp
      t_prev = t0
      time.sleep(max(0.0, self.period - (time.monotonic() - t0)))

  def close(self):
    self.bng.disconnect()


class Camerad:
  """Simulates camerad for a comma four (os04c10) at W x H."""

  def __init__(self):
    self.pm = messaging.PubMaster(['narrowRoadCameraState', 'wideRoadCameraState'])
    self.server = VisionIpcServer("camerad")
    stride, y_height, _, size = get_nv12_info(W, H)
    for stream in (VisionStreamType.VISION_STREAM_NARROW_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD):
      self.server.create_buffers_with_sizes(stream, 5, W, H, size, stride, stride * y_height)
    self.server.start_listener()
    self.frame_id = 0

  def send(self, road_nv12: np.ndarray, wide_nv12: np.ndarray, t_sof: int, t_eof: int):
    for stream, service, yuv in ((VisionStreamType.VISION_STREAM_NARROW_ROAD, 'narrowRoadCameraState', road_nv12),
                                 (VisionStreamType.VISION_STREAM_WIDE_ROAD, 'wideRoadCameraState', wide_nv12)):
      self.server.send(stream, yuv, self.frame_id, t_sof, t_eof)
      dat = messaging.new_message(service, valid=True)
      setattr(dat, service, {
        "frameId": self.frame_id,
        "timestampSof": t_sof,
        "timestampEof": t_eof,
        "sensor": "os04c10",
        "transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
      })
      self.pm.send(service, dat)
    self.frame_id += 1


class CameraRig:
  """The two cameras in BeamNG, their pollers, and the 20 Hz pair publisher."""

  def __init__(self, bng: BeamNGpy, bng_lock: threading.Lock, vehicle, host: str, port: int, update_time: float = 0.02,
               poll_period: float = 0.04, mount=MOUNT, log=print):
    self.log = log
    self.names = ("tp_road", "tp_wide")
    self.bng, self.bng_lock = bng, bng_lock
    with bng_lock:
      for name in self.names:  # a camera left over from a run that didn't exit cleanly
        try:
          bng.connection.send({'type': "CloseCamera", 'name': name}).recv()
        except Exception:
          pass
      self.cams = [Camera(name, bng, vehicle, requested_update_time=-1.0, update_priority=1.0, pos=mount, dir=(0, -1, 0), up=(0, 0, 1),
                          resolution=(W, H), field_of_view_y=fov, near_far_planes=(0.1, 1000.0), is_using_shared_memory=False,
                          is_render_annotations=False, is_render_depth=False, is_render_instance=False)
                   for name, fov in zip(self.names, (ROAD_FOV_Y, WIDE_FOV_Y), strict=True)]
      # set both update times in one burst, so both cameras start their timers on the same game frame
      sent = [bng.connection.send({'type': "SetCameraRequestedUpdateTime", 'name': name, 'updateTime': update_time}) for name in self.names]
      for r in sent:
        r.ack("CompletedSetCameraRequestedUpdateTime")
    self.stream = CameraStream(host, port, self.names, poll_period)
    self.converters = [NV12Converter(W, H) for _ in self.names]
    self.camerad: Camerad | None = None
    self.stats = {'pairs': 0, 'republished': 0, 'skipped': 0, 'skew': deque(maxlen=200), 'age': deque(maxlen=200), 'convert': deque(maxlen=200)}
    self.threads: list[threading.Thread] = []
    self.last_report: dict = {}
    self.last_t_frame = 0.0

  def start(self, exit_event: threading.Event):
    self.threads.append(threading.Thread(target=self.stream.run, args=(exit_event,), name="cam_poll", daemon=True))
    self.threads.append(threading.Thread(target=self.publish_loop, args=(exit_event,), name="cam_publish", daemon=True))
    for t in self.threads:
      t.start()

  def publish_loop(self, exit_event: threading.Event):
    self.camerad = Camerad()
    rk = Ratekeeper(20, print_delay_threshold=None)
    last_seq, last_pub, last_report = (0, 0), 0.0, time.monotonic()
    while not exit_event.is_set():
      road, wide = self.stream.latest
      now = time.monotonic()
      if road is not None and wide is not None:
        fresh = road.seq != last_seq[0] and wide.seq != last_seq[1]
        if fresh or now - last_pub > REPUBLISH_AFTER:
          self.publish(road, wide, fresh)
          last_seq, last_pub = (road.seq, wide.seq), time.monotonic()
        else:
          self.stats['skipped'] += 1
      if now - last_report > 5.0:
        self.report(now - last_report)
        last_report = now
      rk.keep_time()

  def publish(self, road: Frame, wide: Frame, fresh: bool):
    t0 = time.monotonic()
    yuv = [c.convert(f.rgb) for c, f in zip(self.converters, (road, wide), strict=True)]
    t1 = time.monotonic()
    # a republished pair gets a later time than the last one, as a camera that kept seeing the same scene would
    t_frame = road.t if fresh else self.last_t_frame + 0.05
    t_frame = max(t_frame, self.last_t_frame + 0.001)
    self.last_t_frame = t_frame
    self.camerad.send(yuv[0], yuv[1], int(t_frame * 1e9), int(t_frame * 1e9))
    t2 = time.monotonic()
    st = self.stats
    st['pairs' if fresh else 'republished'] += 1
    st['skew'].append(abs(road.t - wide.t))
    st['age'].append(t2 - road.t)
    st['convert'].append((t1 - t0) / 2)

  def report(self, dt: float):
    st = self.stats
    if st['skew']:
      self.log(f"cameras: {st['pairs'] / dt:.1f} fresh pairs/s, {st['republished'] / dt:.1f} republished/s, " +
               f"skew med {np.median(st['skew']) * 1e3:.1f} p90 {np.percentile(st['skew'], 90) * 1e3:.1f} ms, " +
               f"age med {np.median(st['age']) * 1e3:.1f} p90 {np.percentile(st['age'], 90) * 1e3:.1f} ms, " +
               f"nv12 {np.median(st['convert']) * 1e3:.1f} ms/frame, {self.stream.polls / dt:.1f} polls/s")
    self.last_report = {'fresh_pairs_per_s': st['pairs'] / dt, 'republished_per_s': st['republished'] / dt,
                            'skew_med_ms': float(np.median(st['skew']) * 1e3) if st['skew'] else None,
                            'age_med_ms': float(np.median(st['age']) * 1e3) if st['age'] else None}
    st['pairs'] = st['republished'] = st['skipped'] = 0
    self.stream.polls = 0

  def close(self):
    for t in self.threads:
      t.join(timeout=2.0)
    self.stream.close()
    with self.bng_lock:
      for cam in self.cams:
        try:
          cam.remove()
        except Exception as e:
          self.log(f"camera remove failed: {e}")
