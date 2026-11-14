#!/usr/bin/env python3
"""The camera light reader replayed on a logged drive's road runs, for what it reads and what it costs.

  DEV=METAL light_replay.py SEGMENT_DIR... --raw RAW_DIR [--reps 5] [--save OUT.pkl] [--against BASE.pkl]

Each segment directory holds the segment's rlog.zst (or qlog.zst) and qcamera.ts. Its video is upscaled to the road camera's
size (bicubic), packed into camerad's padded NV12 and run through the detector (yolo26n through tinygrad on this host; the
raw outputs cached in RAW_DIR as raw_<segment name>.npy). On every frame a logged road run was on (turnpilotDetections
camera road, matched through qNarrowRoadEncodeIdx), modeld's own ModeldDetector.read_lights runs with the real
CameraLightReader, decode_lights and fill_heads, the frame's modelV2 and calibration taken from the log at the run's time,
in the drive's order, so the voter carries from run to run as on the car. Every run is read reps times from the same
reader state (its lightTime the minimum: a loaded host's one-off stalls drop out) and must publish the same each time.

--save writes every run's published lights and lightRejects (every field) and its times; --against compares them with a
saved replay, run by run, and prints each run that differs (exit 1 if any): a change to the reader's code that must not
change what it reads is proved on the drive's own frames by a replay before and after it.

The device cost: the car logs lightTime on every road run, so on the runs where this host's replay of a build and the car
published the same heads, the car's time over its base against this host's is the scale (runtime finder, 2026-11-12:
7.7x, quartiles 6.3-9.0, for the reader of f928161cc on route 29). --scale R --device-base S prints each run's device
estimate S + R (t - base on this host), base the median of the runs that read no box.

The video is the qcamera's 526x330 upscaled: its boxes are not the car's, nor its lamps' pixels. Only the comparison and
the relative cost mean anything. Prints counts and times only, never a path, a place or a position.
"""
import argparse
import collections
import copy
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np

from openpilot.cereal import messaging
from openpilot.selfdrive.turnpilot.detector import light_state
from openpilot.selfdrive.turnpilot.detector.graph import letterbox
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

W, H = 1344, 760  # the road camera on a comma four


class LogBus:
  """What CameraLightReader reads off its SubMaster, set from the log at each run's time."""
  def __init__(self):
    self.seen = collections.defaultdict(lambda: True)
    self.cur = {}

  def update(self, timeout=0):
    pass

  def __getitem__(self, k):
    return self.cur[k]


class Meta:
  def __init__(self, frame_id: int, timestamp_eof: int):
    self.frame_id, self.timestamp_eof = frame_id, timestamp_eof


class Frames:
  frames: list = [None, None]
  chestnut = True


def read_segment(seg: Path) -> dict:
  """The log's road runs (frameId -> (logMonoTime, lightTime, timestampEof, heads published)), its qcamera frames' ids (encode index ->
  (frameId, timestampEof)), modelV2 by frameId, the calibrations in time order and the device and camera states."""
  from openpilot.tools.lib.logreader import LogReader
  rlog = seg / 'rlog.zst' if (seg / 'rlog.zst').exists() else seg / 'qlog.zst'
  out = {'enc': {}, 'model': {}, 'cal': [], 'dev': None, 'cam': None, 'runs': {}}
  for e in LogReader(str(rlog), only_union_types=True):
    w = e.which()
    if w == 'qNarrowRoadEncodeIdx':
      out['enc'][e.qNarrowRoadEncodeIdx.segmentId] = (e.qNarrowRoadEncodeIdx.frameId, e.qNarrowRoadEncodeIdx.timestampEof)
    elif w == 'modelV2':
      out['model'][e.modelV2.frameId] = e.modelV2
    elif w == 'extrinsicsCalibration':
      out['cal'].append((e.logMonoTime, e.extrinsicsCalibration))
    elif w == 'deviceState' and out['dev'] is None:
      out['dev'] = e.deviceState
    elif w == 'narrowRoadCameraState' and out['cam'] is None:
      out['cam'] = e.narrowRoadCameraState
    elif w == 'turnpilotDetections' and str(e.turnpilotDetections.camera) == 'road' and str(e.turnpilotDetections.view) == 'full':
      d = e.turnpilotDetections
      out['runs'][d.frameId] = (e.logMonoTime, d.lightTime, d.timestampEof, len(d.lights.lights))
  return out


def qcamera_frames(path: Path):
  """The video's frames in order, each as padded NV12 at W x H, flat."""
  stride, y_height, uv_height, _ = get_nv12_info(W, H)
  ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', str(path), '-vf', f'scale={W}:{H}:flags=bicubic', '-fps_mode', 'passthrough',
                         '-f', 'rawvideo', '-pix_fmt', 'nv12', 'pipe:1'], stdout=subprocess.PIPE)
  n = W * H * 3 // 2
  assert ff.stdout is not None
  while True:
    buf = ff.stdout.read(n)
    if len(buf) < n:
      break
    img = np.frombuffer(buf, np.uint8)
    nv = np.zeros((y_height + uv_height, stride), np.uint8)
    nv[:H, :W] = img[:H * W].reshape(H, W)
    nv[y_height:y_height + H // 2, :W] = img[H * W:].reshape(H // 2, W)
    yield nv.reshape(-1)
  ff.wait()


def detector_raws(seg: Path, raw_dir: Path) -> np.ndarray:
  """The detector's (300, 6) output on every frame of the segment's video, cached in raw_dir."""
  cache = raw_dir / f'raw_{seg.name}.npy'
  if cache.exists():
    return np.load(cache)
  from tinygrad import Tensor, TinyJit
  from openpilot.selfdrive.turnpilot.detector.graph import make_detector
  _, detect = make_detector(W, H)
  jit = TinyJit(lambda x: detect(x).realize())
  raws = np.stack([jit(Tensor(nv).realize()).numpy().copy() for nv in qcamera_frames(seg / 'qcamera.ts')])
  raw_dir.mkdir(parents=True, exist_ok=True)
  np.save(cache, raws)
  return raws


def lights_record(lights) -> tuple:
  """Every field of a list of TurnpilotLights.Light, as a tuple to compare."""
  return tuple((str(o.state), round(float(o.distance), 4), str(o.instance), round(float(o.conf), 5), tuple(round(float(b), 5) for b in o.box),
                round(float(o.bearing), 4), int(o.votes), bool(o.governs), str(o.read), str(o.reason)) for o in lights)


def make_reader(bus: LogBus):
  """modeld's ModeldDetector as far as read_lights goes, its reader the real CameraLightReader on bus."""
  from openpilot.selfdrive.turnpilot.detector import modeld_detector as md
  det = md.ModeldDetector.__new__(md.ModeldDetector)
  det.lb, det.cameras = letterbox(W, H), {'road': 0, 'wide': 1}
  det.light_times = collections.deque(maxlen=md.TIME_WINDOW)
  det.light_mode, det.lights = 'addOnly', md.CameraLightReader(bus)
  det.model_state = Frames()
  return det


def replay_segment(seg: Path, raws: np.ndarray, det, bus: LogBus, reps: int, t_base: int | None) -> list[dict]:
  """Every logged road run's read on its video frame (module docstring), in the drive's order. t_base: the drive's first
  logged time (ns), runs' times are seconds from it."""
  log = read_segment(seg)
  bus.cur.update(deviceState=log['dev'], narrowRoadCameraState=log['cam'])
  cal, models = log['cal'], log['model']
  first_model = next(iter(models.values()))
  rows, n_frames = [], 0
  for i, nv in enumerate(qcamera_frames(seg / 'qcamera.ts')):
    n_frames += 1
    fid, _ = log['enc'].get(i, (None, None))
    if fid not in log['runs']:
      continue
    t_log, car_lt, eof, car_n = log['runs'][fid]
    bus.cur['extrinsicsCalibration'] = max((c for c in cal if c[0] <= t_log), default=cal[0], key=lambda c: c[0])[1]
    bus.cur['modelV2'] = models.get(fid, first_model)
    Frames.frames = [nv, nv]
    metas = [Meta(fid, eof), Meta(fid, eof)]
    start = copy.deepcopy(det.lights.reader) if det.lights.reader is not None else None
    reads, times, walls, first = [0], [], [], None
    orig = light_state.read_head

    def counted(*a, reads=reads, orig=orig, **k):
      reads[0] += 1
      return orig(*a, **k)
    light_state.read_head = counted
    try:
      for rep in range(reps):
        if rep:
          det.lights.reader = copy.deepcopy(start)
        reads[0] = 0
        msg = messaging.new_message('turnpilotDetections')
        td = msg.turnpilotDetections
        t0 = time.perf_counter()
        det.read_lights(td, det.model_state, metas, raws[i])
        walls.append(time.perf_counter() - t0)
        back = msg.as_reader().turnpilotDetections
        rec = (str(back.lightStatus), lights_record(back.lights.lights), lights_record(back.lightRejects))
        if first is None:
          first, n_reads = rec, reads[0]
        elif rec != first:
          raise AssertionError(f'run {fid}: a rep read differently from the same state')
        times.append(back.lightTime)
    finally:
      light_state.read_head = orig
    rows.append({'seg': seg.name.rsplit('--', 1)[-1], 'frame': int(fid), 't': (t_log - (t_base or t_log)) * 1e-9, 'status': first[0],
                 'lights': first[1], 'rejects': first[2], 'reads': n_reads, 'mac_s': min(times), 'wall_s': min(walls),
                 'car_s': float(car_lt), 'car_heads': car_n})
  if n_frames != len(log['enc']):
    print(f'  {seg.name.rsplit("--", 1)[-1]}: {n_frames} video frames against {len(log["enc"])} encoded: frames may not match their runs')
  return rows


def compare(base: list[dict], new: list[dict]) -> list[str]:
  """Each run whose published lights or lightRejects differ, or that is in one replay only."""
  key = {(r['seg'], r['frame']): r for r in base}
  out = []
  for r in new:
    b = key.pop((r['seg'], r['frame']), None)
    if b is None:
      out.append(f"segment {r['seg']} frame {r['frame']}: not in the base replay")
      continue
    for field in ('status', 'lights', 'rejects'):
      if b[field] != r[field]:
        out.append(f"segment {r['seg']} frame {r['frame']} ({r['t']:.2f} s): {field} {b[field]} -> {r[field]}")
  out += [f"segment {s} frame {f}: in the base replay only" for s, f in key]
  return out


def percentiles(a) -> str:
  a = np.asarray(a, float)
  if not len(a):
    return 'n=0'
  p50, p90, p99 = np.percentile(a, [50, 90, 99])
  return f'n={len(a)} p50={p50:.2f} p90={p90:.2f} p99={p99:.2f} max={a.max():.2f}'


def summary(rows: list[dict], scale: float | None, device_base: float | None) -> None:
  mac = np.array([r['mac_s'] for r in rows]) * 1e3
  print('runs', len(rows), 'reading a box', f"{100 * np.mean([r['reads'] > 0 for r in rows]):.0f}%",
        'reads per run', dict(sorted(collections.Counter(r['reads'] for r in rows).items())))
  print('this host, lightTime ms:', percentiles(mac))
  print('this host, read_lights with its rejects ms:', percentiles([r['wall_s'] * 1e3 for r in rows]))
  print('by reads, median ms:', {n: round(float(np.median([r['mac_s'] * 1e3 for r in rows if r['reads'] == n])), 3)
                                 for n in sorted({r['reads'] for r in rows})})
  if scale is not None and device_base is not None:
    base = float(np.median([r['mac_s'] for r in rows if r['reads'] == 0])) * 1e3
    dev = device_base + scale * np.maximum(mac - base, 0.)
    print(f'device estimate ({device_base:.3f} ms + {scale:.1f} x (t - {base:.3f})):', percentiles(dev),
          f'over 4 ms {100 * np.mean(dev > 4.):.0f}%')


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('segments', nargs='+', type=Path)
  ap.add_argument('--raw', type=Path, required=True)
  ap.add_argument('--reps', type=int, default=5)
  ap.add_argument('--save', type=Path)
  ap.add_argument('--against', type=Path)
  ap.add_argument('--scale', type=float)
  ap.add_argument('--device-base', type=float)
  args = ap.parse_args()
  segs = sorted(args.segments, key=lambda p: int(p.name.rsplit('--', 1)[-1]))
  bus = LogBus()
  det = make_reader(bus)
  rows: list[dict] = []
  t_base = None
  for seg in segs:
    raws = detector_raws(seg, args.raw)
    if t_base is None:
      t_base = min(v[0] for v in read_segment(seg)['runs'].values())
    rows += replay_segment(seg, raws, det, bus, args.reps, t_base)
  summary(rows, args.scale, args.device_base)
  if args.save:
    with open(args.save, 'wb') as f:
      pickle.dump(rows, f)
  if args.against:
    with open(args.against, 'rb') as f:
      diffs = compare(pickle.load(f), rows)
    print(f'against the base replay: {len(diffs)} runs differ')
    for d in diffs[:40]:
      print(' ', d)
    if diffs:
      raise SystemExit(1)


if __name__ == '__main__':
  main()
