#!/usr/bin/env python3
"""Replay one segment's camera videos through the compiled detector and the scene tracker, the way modeld and turnpilotd
would see them, and write annotated JPEGs and a summary.

  DEV=METAL replay_route.py RLOG FCAMERA.hevc ECAMERA.hevc OUT_DIR --cameras road|wide|alt --count 120

Frames are decoded with ffmpeg and repacked into camerad's padded NV12 layout (as process_replay does). Ranging is checked
against the radar-matched lead: the nearest in-path vehicle track against radarState.leadOne.dRel, which is measured from
the front bumper, so expect ours to read about the camera-to-bumper distance longer.
"""
import argparse
import subprocess
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from openpilot.cereal import messaging
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.turnpilot.detector.decoder import BUS, CAR, TRUCK, decode, fill_detections
from openpilot.selfdrive.turnpilot.detector.graph import detector_pkl_path, letterbox
from openpilot.selfdrive.turnpilot.scene.tracker import SceneTracker
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.tools.lib.logreader import LogReader

NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
COLORS = {0: (0, 0, 255), 1: (0, 160, 255), 2: (255, 160, 0), 3: (0, 255, 255), 5: (255, 0, 255), 7: (0, 255, 0)}


def video_frames(fn: str, start: int, count: int, step: int, pix_fmt: str) -> tuple[int, int, np.ndarray]:
  """Frames start, start+step, ... as tight NV12 rows or (h, w, 3) RGB. tools/lib's FrameReader passes -vsync, which
  ffmpeg 9 rejects, so this calls ffmpeg directly."""
  probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                                   '-of', 'csv=p=0', fn]).decode()
  w, h = map(int, probe.strip().split(',')[:2])
  sel = f"select='gte(n\\,{start})*not(mod(n-{start}\\,{step}))'"
  dat = subprocess.check_output(['ffmpeg', '-v', 'error', '-f', 'hevc', '-i', fn, '-vf', sel, '-fps_mode', 'passthrough',
                                 '-frames:v', str(count), '-f', 'rawvideo', '-pix_fmt', pix_fmt, 'pipe:1'])
  shape = (-1, h, w, 3) if pix_fmt == 'rgb24' else (-1, h * w * 3 // 2)
  return w, h, np.frombuffer(dat, np.uint8).reshape(shape)


def camerad_nv12(img: np.ndarray, w: int, h: int) -> np.ndarray:
  stride, y_height, uv_height, _ = get_nv12_info(w, h)
  out = np.zeros((y_height + uv_height, stride), dtype=np.uint8)
  out[:h, :w] = img[:h * w].reshape(h, w)
  out[y_height:y_height + h // 2, :w] = img[h * w:].reshape(h // 2, w)
  return out.reshape(-1)


def latest(msgs, t):
  return msgs[max(0, int(np.searchsorted([m.logMonoTime for m in msgs], t)) - 1)]


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('rlog')
  ap.add_argument('fcamera')
  ap.add_argument('ecamera')
  ap.add_argument('out', type=Path)
  ap.add_argument('--cameras', choices=('road', 'wide', 'alt'), default='road')
  ap.add_argument('--start', type=int, default=0)
  ap.add_argument('--count', type=int, default=120)
  ap.add_argument('--step', type=int, default=3, help='frames between detector runs (modeld runs it every 3rd)')
  ap.add_argument('--save-every', type=int, default=30)
  args = ap.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  from tinygrad import Device, Tensor
  from tinygrad.engine.realize import lower_and_compile
  from tinygrad_repo.examples.openpilot.helpers import load_pickle

  lr = list(LogReader(args.rlog))
  device_type = next(str(m.deviceState.deviceType) for m in lr if m.which() == 'deviceState')
  sensor = next(str(m.narrowRoadCameraState.sensor) for m in lr if m.which() == 'narrowRoadCameraState')
  device_type = 'tici' if device_type == 'unknown' else device_type  # logs older than deviceType
  by = {w: [m for m in lr if m.which() == w] for w in ('extrinsicsCalibration', 'carState', 'radarState')}
  models = {m.modelV2.frameId: m.modelV2 for m in lr if m.which() == 'modelV2'}
  frame_ids = {'road': {m.narrowRoadEncodeIdx.segmentId: m.narrowRoadEncodeIdx.frameId for m in lr if m.which() == 'narrowRoadEncodeIdx'},
               'wide': {m.wideRoadEncodeIdx.segmentId: m.wideRoadEncodeIdx.frameId for m in lr if m.which() == 'wideRoadEncodeIdx'}}
  eofs = {'road': {m.narrowRoadCameraState.frameId: m.narrowRoadCameraState.timestampEof for m in lr if m.which() == 'narrowRoadCameraState'},
          'wide': {m.wideRoadCameraState.frameId: m.wideRoadCameraState.timestampEof for m in lr if m.which() == 'wideRoadCameraState'}}

  frames, w, h = {}, 0, 0
  for cam, fn in (('road', args.fcamera), ('wide', args.ecamera)):
    if args.cameras in (cam, 'alt'):
      w, h, nv12 = video_frames(fn, args.start, args.count, args.step, 'nv12')
      frames[cam] = (nv12, video_frames(fn, args.start, args.count, args.step, 'rgb24')[2])
  art = load_pickle(detector_pkl_path(w, h, False), out_of_band=True)
  run = art['run']
  run.captured._linear = lower_and_compile(run.captured._linear)
  lb = letterbox(w, h, tuple(art['metadata']['input_size']))
  tracker = SceneTracker(DEVICE_CAMERAS[(device_type, sensor)], device_type)
  print(f"{device_type}/{sensor} {w}x{h} -> {lb.in_w}x{lb.in_h}, {art['metadata']['model']} on {Device.DEFAULT}")

  counts, times, ids, ranges = Counter(), [], Counter(), []
  n = min(len(f[0]) for f in frames.values())
  for k in range(n):
    cam = args.cameras if args.cameras != 'alt' else ('road', 'wide')[k % 2]
    seg_idx = args.start + k * args.step
    t0 = time.perf_counter()
    raw = run(input_frame=Tensor(camerad_nv12(frames[cam][0][k], w, h)).realize()).numpy()
    times.append(time.perf_counter() - t0)
    dets = decode(raw, lb)
    frame_id = frame_ids[cam].get(seg_idx, seg_idx)
    eof = eofs[cam].get(frame_id, 0)
    msg = messaging.new_message('turnpilotDetections')
    fill_detections(msg.turnpilotDetections, dets, frame_id, eof, cam, times[-1])
    tracks = tracker.update(msg.as_reader().turnpilotDetections, latest(by['extrinsicsCalibration'], eof).extrinsicsCalibration,
                            latest(by['carState'], eof).carState, models.get(frame_ids['road'].get(seg_idx, -1)), now=eof * 1e-9)
    counts.update(NAMES[int(d[5])] for d in dets)
    ids.update(trk.track_id for trk in tracks)
    lead = latest(by['radarState'], eof).radarState.leadOne
    cars = [float(trk.x[0]) for trk in tracks if trk.x is not None and trk.in_path and trk.cls in (CAR, BUS, TRUCK)]
    if lead.present and lead.radar and cars:
      ranges.append((lead.dRel, min(cars)))
    if k % args.save_every == 0 or k == n - 1:
      img = cv2.cvtColor(frames[cam][1][k], cv2.COLOR_RGB2BGR)
      for x0, y0, x1, y1, _, c in dets:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), COLORS[int(c)], 2)
      for trk in (trk for trk in tracks if trk.camera == cam):
        x, y = trk.x[:2] if trk.x is not None else (np.nan, np.nan)
        label = f"#{trk.track_id} {NAMES[trk.cls]} {x:.1f},{y:+.1f}m" + (' PATH' if trk.in_path else '')
        org = (int(trk.box_seen[0]), max(12, int(trk.box_seen[1]) - 4))
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[trk.cls], 1)
      cv2.putText(img, f"{cam} {seg_idx} {len(dets)} dets {times[-1] * 1e3:.1f} ms", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
      cv2.imwrite(str(args.out / f'{cam}_{seg_idx:04d}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 85])

  ms = np.array(times[2:]) * 1e3
  print(f"{n} frames, {sum(counts.values())} detections {dict(counts)}")
  print(f"detector: median {np.median(ms):.2f} ms, p90 {np.percentile(ms, 90):.2f} ms, with the readback")
  print(f"{len(ids)} published track ids; frames published per id, longest: {sorted(ids.values(), reverse=True)[:8]}")
  if ranges:
    r = np.array(ranges)
    d = r[:, 1] - r[:, 0]
    print(f"nearest in-path vehicle vs radar lead (n={len(r)}, {r[:, 0].min():.0f}-{r[:, 0].max():.0f} m): " +
          f"ours - radar median {np.median(d):+.2f} m, p10 {np.percentile(d, 10):+.2f}, p90 {np.percentile(d, 90):+.2f}")


if __name__ == '__main__':
  main()
