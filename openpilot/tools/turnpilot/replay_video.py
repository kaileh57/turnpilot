#!/usr/bin/env python3
"""Run TurnPilot's detector and scene tracker on a public comma route and write the road and wide cameras, side by side
with the boxes drawn on them, as an h264 mp4 at the cameras' real 20 fps.

  DEV=METAL replay_video.py 5beb9b58bd12b691/0000010a--a51155e496 --start 840 --seconds 84 -o replay_detections.mp4

That is the last 84 s of openpilot's demo route (a comma four, 1344x760): an urban stretch with signals, crosswalks and
pedestrians. The rest of the route has few: a 2 fps scan of its qcameras found a person in only segments 7, 11, 14 and 15.

Both cameras go through the detector as full frames (letterboxed, no crop, as in modeld) and each through its own
SceneTracker, so a track id belongs to one panel (one SceneTracker takes one camera per update, and an object in both views
would flip its camera on every update). Boxes are the tracker's published tracks, where this frame's detection put them:
`#id class range`, with the range in meters over the ground from the camera (a flat-ground estimate, shown as 100+ m
beyond that). A track whose position is inside the driving model's path (modelV2.position, 1.5 m either side) is yellow and
filled; the path itself is the green strip. Gray boxes are detections the tracker hasn't confirmed. The route's rlog gives
the calibration, car speed and driving model path, and its segment files are downloaded to --cache. Only a public route
works without a comma account.

tools/lib/framereader fails with this Mac's ffmpeg 9 (-vsync), so frames are streamed from ffmpeg directly, one NV12 frame at
a time: the detector gets camerad's padded NV12 layout, the drawing gets the same frame as BGR.
"""
import argparse
import json
import subprocess
import time
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from openpilot.cereal import messaging
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.turnpilot.detector.decoder import BICYCLE, BUS, CAR, MOTORCYCLE, PERSON, TRUCK, decode, fill_detections
from openpilot.selfdrive.turnpilot.detector.graph import DEFAULT_MODEL, letterbox, make_detector
from openpilot.selfdrive.turnpilot.scene.ranging import calib_from_view, camera_height
from openpilot.selfdrive.turnpilot.scene.tracker import CORRIDOR_HALF_WIDTH, PATH_X, SceneTracker, predicted_path
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route

FPS = 20
SEGMENT_FRAMES = 60 * FPS
LEGEND_H = 48
FONT = cv2.FONT_HERSHEY_SIMPLEX
# BGR
NAMES = {PERSON: 'person', BICYCLE: 'bicycle', CAR: 'car', MOTORCYCLE: 'motorcycle', BUS: 'bus', TRUCK: 'truck'}
COLORS = {PERSON: (70, 70, 255), BICYCLE: (255, 80, 200), CAR: (255, 170, 40), MOTORCYCLE: (255, 120, 255), BUS: (200, 255, 0), TRUCK: (80, 220, 80)}
PATH_COLOR, CORRIDOR_COLOR, UNCONFIRMED = (0, 235, 255), (90, 200, 60), (150, 150, 150)
CORRIDOR_MIN_X = 7.  # m: the hood hides the road nearer than this
LOG_TYPES = ('deviceState', 'carState', 'modelV2', 'extrinsicsCalibration', 'narrowRoadEncodeIdx', 'wideRoadEncodeIdx', 'narrowRoadCameraState')


def fetch(src: str | None, cache: Path) -> Path:
  """A route file's local path; downloads a URL once into cache/dongle/route/segment/file."""
  if src is None:
    raise SystemExit('the route is missing a file this replay needs (rlog, fcamera or ecamera)')
  if not src.startswith('http'):
    return Path(src)
  dest = cache.joinpath(*urlparse(src).path.split('/')[-4:])
  if not dest.exists():
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    print(f'downloading {dest.relative_to(cache)}', flush=True)
    with requests.get(src, stream=True, timeout=60) as r:
      r.raise_for_status()
      with open(part, 'wb') as f:
        for chunk in r.iter_content(1 << 20):
          f.write(chunk)
    if int(r.headers.get('content-length', part.stat().st_size)) != part.stat().st_size:
      raise SystemExit(f'{dest.name} came back truncated')
    part.rename(dest)
  return dest


class Log:
  """What the replay reads from the rlogs: messages by type in time order, and each camera frame's id and end-of-frame time."""
  def __init__(self, paths: list[Path]):
    self.series: dict[str, tuple[list[int], list]] = defaultdict(lambda: ([], []))
    self.stamps: dict[str, dict[tuple[int, int], tuple[int, int]]] = {'road': {}, 'wide': {}}
    self.models: dict[int, object] = {}
    self.device_type = self.sensor = ''
    for path in paths:
      for m in LogReader(str(path)):
        w = m.which()
        if w not in LOG_TYPES:
          continue
        if w in ('narrowRoadEncodeIdx', 'wideRoadEncodeIdx'):
          e = getattr(m, w)
          self.stamps['road' if w.startswith('narrow') else 'wide'][(e.segmentNum, e.segmentId)] = (e.frameId, e.timestampEof)
        elif w == 'deviceState':
          self.device_type = self.device_type or str(m.deviceState.deviceType)
        elif w == 'narrowRoadCameraState':
          self.sensor = self.sensor or str(m.narrowRoadCameraState.sensor)
        else:
          self.series[w][0].append(m.logMonoTime)
          self.series[w][1].append(m)
          if w == 'modelV2':
            self.models[m.modelV2.frameId] = m.modelV2
    for t, _ in self.series.values():
      assert t == sorted(t)
    if self.device_type == 'unknown':
      self.device_type = 'tici'  # logs older than deviceType

  def at(self, name: str, t: int):
    """The latest message of this type at or before t (ns), else the first."""
    times, msgs = self.series[name]
    return getattr(msgs[max(0, bisect_right(times, t) - 1)], name)

  def stamp(self, cam: str, segment: int, k: int, last: tuple[int, int]) -> tuple[int, int]:
    """(frame id, end-of-frame ns) of frame k of a segment's video; one 20 Hz frame after the last, for a frame the log lacks."""
    return self.stamps[cam].get((segment, k), (last[0] + 1, last[1] + 1_000_000_000 // FPS))


def video_size(fn: Path) -> tuple[int, int]:
  out = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                                 '-of', 'csv=p=0', str(fn)]).decode()
  w, h = map(int, out.strip().split(',')[:2])
  return w, h


def video_frames(fn: Path, w: int, h: int):
  """A segment video's frames as tight NV12 arrays, streamed. -fps_mode passthrough keeps ffmpeg from duplicating or dropping."""
  p = subprocess.Popen(['ffmpeg', '-v', 'error', '-f', 'hevc', '-i', str(fn), '-fps_mode', 'passthrough', '-f', 'rawvideo',
                        '-pix_fmt', 'nv12', 'pipe:1'], stdout=subprocess.PIPE)
  size = w * h * 3 // 2
  try:
    while len(buf := p.stdout.read(size)) == size:
      yield np.frombuffer(buf, np.uint8)
  finally:
    p.kill()
    p.wait()


def camerad_nv12(img: np.ndarray, w: int, h: int) -> np.ndarray:
  """Tight NV12 into camerad's padded layout, which is what the detector graph reads."""
  stride, y_height, uv_height, _ = get_nv12_info(w, h)
  out = np.zeros((y_height + uv_height, stride), dtype=np.uint8)
  out[:h, :w] = img[:h * w].reshape(h, w)
  out[y_height:y_height + h // 2, :w] = img[h * w:].reshape(h // 2, w)
  return out.reshape(-1)


def outlined(img, text, org, scale=0.6, color=(255, 255, 255)):
  """Text with a black outline. The outline is the same thickness shifted 4 ways, since this build's glyph spacing changes with
  thickness and a thicker black copy wouldn't line up."""
  for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, 2), (0, -2)):
    cv2.putText(img, text, (org[0] + dx, org[1] + dy), FONT, scale, (0, 0, 0), 1, cv2.LINE_AA)
  cv2.putText(img, text, org, FONT, scale, color, 1, cv2.LINE_AA)


def tag(img, text, x, y, bg, fg=(0, 0, 0)):
  """A text pill with its bottom-left at (x, y), or inside the frame's top edge when there's no room above, and kept in the frame."""
  (tw, th), base = cv2.getTextSize(text, FONT, 0.55, 1)
  top = y - th - base - 4 if y - th - base - 4 >= 0 else y
  x = max(0, min(x, img.shape[1] - tw - 7))
  cv2.rectangle(img, (x, top), (x + tw + 6, top + th + base + 4), bg, -1)
  cv2.putText(img, text, (x + 3, top + th + 1), FONT, 0.55, fg, 1, cv2.LINE_AA)


class Panel:
  """One camera: its tracker, the detector's last result, and the drawing of it."""
  def __init__(self, cam: str, tracker: SceneTracker, w: int, h: int):
    self.cam, self.tracker, self.w, self.h = cam, tracker, w, h
    self.lens = tracker.lenses[cam]
    self.dets = np.zeros((0, 6), np.float32)
    self.tracks: list = []
    self.ms = 0.
    self.calibration = None
    self.model_v2 = None
    self.confirmed = 0

  def corridor(self, img: np.ndarray) -> None:
    """The driving model's path, 1.5 m either side, projected onto the ground."""
    cal = self.calibration
    if cal is None or len(cal.rpyCalib) != 3:
      return
    wide_euler = list(cal.wideFromDeviceEuler) if self.cam == 'wide' and len(cal.wideFromDeviceEuler) == 3 else None
    view_from_calib = calib_from_view(list(cal.rpyCalib), wide_euler).T
    height = camera_height(cal)
    px, py = predicted_path(self.model_v2)
    xs = np.arange(CORRIDOR_MIN_X, min(PATH_X, float(px[-1])), 1.)
    ys = np.interp(xs, px, py)
    edges = []
    for side in (1., -1.):
      pts = []
      for x, y in zip(xs, ys + side * CORRIDOR_HALF_WIDTH, strict=True):
        ray = view_from_calib @ np.array([x, -y, height])  # calib frame is x forward, y right, z down
        if ray[2] > 0.1:
          pts.append(self.lens.project(ray))
      edges.append(pts)
    if len(edges[0]) > 1 and len(edges[1]) > 1:
      poly = np.array(edges[0] + edges[1][::-1], dtype=np.int32)
      tint = img.copy()
      cv2.fillPoly(tint, [poly], CORRIDOR_COLOR, cv2.LINE_AA)
      cv2.addWeighted(tint, 0.3, img, 0.7, 0, img)

  def draw(self, nv12: np.ndarray, header: str) -> np.ndarray:
    img = cv2.cvtColor(nv12.reshape(self.h * 3 // 2, self.w), cv2.COLOR_YUV2BGR_NV12)
    self.corridor(img)
    # every published track is drawn where it was last measured, not where the tracker has extrapolated it (that drifts well off
    # the object in a turn). One matched to this run's detections gets a label; one coasting through a missed detection is a
    # thin outline.
    fresh = [trk for trk in self.tracks if trk.t_seen == self.tracker.last_t]
    for trk in (trk for trk in self.tracks if trk.t_seen != self.tracker.last_t):
      cv2.rectangle(img, tuple(int(v) for v in trk.box_seen[:2]), tuple(int(v) for v in trk.box_seen[2:]), COLORS[trk.cls], 1, cv2.LINE_AA)
    seen = np.array([trk.box_seen for trk in fresh]).reshape(-1, 4)
    for x0, y0, x1, y1, _, _ in self.dets:
      if not (np.abs(seen - (x0, y0, x1, y1)).max(axis=1) < 0.01).any():
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), UNCONFIRMED, 1, cv2.LINE_AA)
    tint = img.copy()
    for trk in fresh:
      if trk.in_path:
        cv2.rectangle(tint, tuple(int(v) for v in trk.box_seen[:2]), tuple(int(v) for v in trk.box_seen[2:]), PATH_COLOR, -1)
    if any(trk.in_path for trk in fresh):
      cv2.addWeighted(tint, 0.25, img, 0.75, 0, img)
    for trk in sorted(fresh, key=lambda t: (t.in_path, t.cls == PERSON)):  # pedestrians on top
      x0, y0, x1, y1 = (int(v) for v in trk.box_seen)
      color = COLORS[trk.cls]
      rng = float(np.hypot(*trk.x[:2]))
      cv2.rectangle(img, (x0, y0), (x1, y1), PATH_COLOR if trk.in_path else color, 4 if trk.in_path else 2, cv2.LINE_AA)
      tag(img, f"#{trk.track_id} {NAMES[trk.cls]} " + (f"{rng:.1f} m" if rng < 100 else '100+ m') + ('  IN PATH' if trk.in_path else ''),
          x0, y0, PATH_COLOR if trk.in_path else color)
    outlined(img, header, (12, 28), 0.7)
    return img


def legend(width: int, title: str) -> np.ndarray:
  img = np.full((LEGEND_H, width, 3), 24, np.uint8)
  x = 16
  outlined(img, title, (x, 31), 0.62, (255, 255, 255))
  x += cv2.getTextSize(title, FONT, 0.62, 1)[0][0] + 40
  for cls, name in NAMES.items():
    cv2.rectangle(img, (x, 14), (x + 18, 32), COLORS[cls], -1)
    cv2.putText(img, name, (x + 26, 30), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    x += 26 + cv2.getTextSize(name, FONT, 0.55, 1)[0][0] + 22
  cv2.rectangle(img, (x, 14), (x + 18, 32), PATH_COLOR, -1)
  notes = 'in the driving path   |   #n track id   |   m = range over the ground   |   green strip = model path +-1.5 m   |   gray = not yet tracked'
  cv2.putText(img, notes, (x + 26, 30), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
  return img


def open_encoder(out: Path, width: int, height: int) -> subprocess.Popen:
  out.parent.mkdir(parents=True, exist_ok=True)
  return subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-framerate', str(FPS),
                           '-i', 'pipe:0', '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p', '-r', str(FPS),
                           '-movflags', '+faststart', str(out)], stdin=subprocess.PIPE)


def stats(values) -> dict:
  v = np.array(values[3:] if len(values) > 6 else values) * 1e3  # the first runs trace and compile the graph
  return {'median_ms': round(float(np.median(v)), 2), 'p90_ms': round(float(np.percentile(v, 90)), 2), 'max_ms': round(float(v.max()), 2)} if len(v) else {}


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  ap.add_argument('route', help='dongle_id/route or dongle_id|route, a public one')
  ap.add_argument('--start', type=float, default=0, help='seconds into the route')
  ap.add_argument('--seconds', type=float, default=75)
  ap.add_argument('-o', '--out', type=Path, default=Path('~/turnpilot-artifacts/demo/replay_detections.mp4').expanduser())
  ap.add_argument('--cache', type=Path, default=Path('~/.cache/turnpilot/replay').expanduser())
  ap.add_argument('--data-dir', help='a local copy of the route, as tools/lib/route.py reads it')
  ap.add_argument('--model', default=DEFAULT_MODEL)
  ap.add_argument('--step', type=int, default=1, help='frames between detector runs on each camera; the two cameras alternate when 2 or more')
  ap.add_argument('--scale', type=float, default=1.0, help='scale the written video')
  ap.add_argument('--raw-cache', type=Path, help="raw detector outputs: read from here if the file exists (no detector run), else saved here after the run")
  args = ap.parse_args()

  route = Route(args.route, data_dir=args.data_dir)
  first = int(args.start // 60)
  skip = round((args.start - first * 60) * FPS)
  n_frames = round(args.seconds * FPS)
  segs = list(range(first, min(route.max_seg_number, first + (skip + n_frames - 1) // SEGMENT_FRAMES) + 1))
  files = {s: [fetch(paths()[s], args.cache) for paths in (route.log_paths, route.camera_paths, route.ecamera_paths)] for s in segs}
  w, h = video_size(files[segs[0]][1])
  print(f'{route.name} segments {segs[0]}-{segs[-1]}: loading the logs', flush=True)
  log = Log([files[s][0] for s in segs])
  device_camera = DEVICE_CAMERAS[(log.device_type, log.sensor)]
  assert (device_camera.narrow_road.width, device_camera.narrow_road.height) == (w, h), f'{log.device_type}/{log.sensor} is not {w}x{h}'

  cached = dict(np.load(args.raw_cache)) if args.raw_cache and args.raw_cache.exists() else None
  fresh: dict[str, np.ndarray] = {}  # 'road_<frame in the route>' -> raw (1, 300, 6), 'dt_road_<frame>' -> detector seconds
  if cached is None:
    from tinygrad import Device, Tensor, TinyJit
    lb, detect = make_detector(w, h, args.model)
    run = TinyJit(detect, prune=True)
    backend = Device.DEFAULT
  else:
    lb, backend = letterbox(w, h), 'raw cache'
  panels = {cam: Panel(cam, SceneTracker(device_camera, log.device_type), w, h) for cam in ('road', 'wide')}
  print(f'{log.device_type}/{log.sensor} {w}x{h} -> {lb.in_w}x{lb.in_h}, {args.model} on {backend}', flush=True)

  out_w, out_h = 2 * w, h + LEGEND_H
  scaled = (round(out_w * args.scale) // 2 * 2, round(out_h * args.scale) // 2 * 2)
  canvas = np.zeros((out_h, out_w, 3), np.uint8)
  canvas[h:] = legend(out_w, f'TurnPilot  {args.model} on the full {w}x{h} frame')
  encoder = open_encoder(args.out, *scaled)

  times = {'road': [], 'wide': []}
  track_ms = []
  raw_counts = {cam: defaultdict(int) for cam in panels}
  track_ids: dict[str, dict[int, int]] = {cam: {} for cam in panels}  # track id -> class
  ped_frames, path_frames, max_peds, skew = defaultdict(int), defaultdict(int), defaultdict(int), []
  ped_spans: dict[tuple[str, int], list] = {}  # (camera, track id) -> first s, last s, nearest range m, detector runs in path
  last = dict.fromkeys(panels, (0, 0))
  frame, t_start = 0, time.perf_counter()
  for seg in segs:
    road_video, wide_video = (video_frames(files[seg][i], w, h) for i in (1, 2))
    for k, (road, wide) in enumerate(zip(road_video, wide_video, strict=False)):
      if seg == segs[0] and k < skip:
        continue
      if frame >= n_frames:
        break
      frames = {'road': road, 'wide': wide}
      t_route = seg * 60 + k / FPS
      for cam, panel in panels.items():
        last[cam] = frame_id, eof = log.stamp(cam, seg, k, last[cam])
        if (frame + (args.step // 2 if cam == 'wide' else 0)) % args.step == 0:
          key = f'{cam}_{seg * SEGMENT_FRAMES + k}'
          if cached is None:
            padded = camerad_nv12(frames[cam], w, h)
            t0 = time.perf_counter()
            raw = run(input_frame=Tensor(padded).realize()).numpy()
            dt = time.perf_counter() - t0
            fresh[key], fresh['dt_' + key] = raw, np.float32(dt)
          else:
            raw, dt = cached[key], float(cached['dt_' + key])
          times[cam].append(dt)
          dets = decode(raw, lb)
          msg = messaging.new_message('turnpilotDetections')
          fill_detections(msg.turnpilotDetections, dets, frame_id, eof, cam, dt)
          panel.calibration = log.at('extrinsicsCalibration', eof)
          panel.model_v2 = log.models.get(frame_id) or log.at('modelV2', eof)
          t1 = time.perf_counter()
          panel.tracks = panel.tracker.update(msg.as_reader().turnpilotDetections, panel.calibration, log.at('carState', eof),
                                              panel.model_v2, now=eof * 1e-9)
          track_ms.append(time.perf_counter() - t1)
          panel.dets, panel.ms = dets, dt * 1e3
          for c in dets[:, 5]:
            raw_counts[cam][NAMES[int(c)]] += 1
          for trk in panel.tracks:
            track_ids[cam][trk.track_id] = trk.cls
            if trk.cls == PERSON and trk.x is not None:
              span = ped_spans.setdefault((cam, trk.track_id), [t_route, t_route, np.inf, 0])
              span[1], span[2], span[3] = t_route, min(span[2], float(np.hypot(*trk.x[:2]))), span[3] + trk.in_path
          peds = [trk for trk in panel.tracks if trk.cls == PERSON]
          ped_frames[cam] += bool(peds)
          max_peds[cam] = max(max_peds[cam], len(peds))
          path_frames[cam] += any(trk.in_path for trk in panel.tracks)
      skew.append(abs(last['road'][1] - last['wide'][1]) * 1e-6)
      car = log.at('carState', last['road'][1])
      for i, (cam, panel) in enumerate(panels.items()):
        head = f"{cam.upper()}  {len(panel.tracks)} tracked  {panel.ms:.1f} ms"
        if cam == 'road':
          head += f"   {t_route:6.1f} s   {car.vEgo * 2.237:.0f} mph"
        canvas[:h, i * w:(i + 1) * w] = panel.draw(frames[cam], head)
      cv2.line(canvas, (w, 0), (w, h), (24, 24, 24), 2)
      encoder.stdin.write((canvas if args.scale == 1 else cv2.resize(canvas, scaled, interpolation=cv2.INTER_AREA)).tobytes())
      frame += 1
      if frame % 100 == 0:
        print(f'{frame}/{n_frames} frames, {frame / (time.perf_counter() - t_start):.1f} fps', flush=True)
    if frame >= n_frames:
      break
  encoder.stdin.close()
  if encoder.wait() != 0:
    raise SystemExit('ffmpeg failed to encode the video')
  if args.raw_cache and cached is None:
    args.raw_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.raw_cache, **fresh)

  summary = {
    'route': route.name.canonical_name, 'start_s': args.start, 'frames': frame, 'seconds': frame / FPS, 'device': f'{log.device_type}/{log.sensor}',
    'frame': f'{w}x{h}', 'model': args.model, 'input': f'{lb.in_w}x{lb.in_h}', 'backend': backend, 'step': args.step,
    'detections_summed_over_frames': {cam: dict(c) for cam, c in raw_counts.items()},
    'distinct_tracks': {cam: {NAMES[c]: sum(1 for v in ids.values() if v == c) for c in NAMES} for cam, ids in track_ids.items()},
    'detector_runs_with_pedestrian_track': dict(ped_frames), 'max_simultaneous_pedestrian_tracks': dict(max_peds),
    'detector_runs_with_in_path_track': dict(path_frames),
    'pedestrian_tracks': [{'camera': cam, 'id': i, 'from_s': round(a, 1), 'to_s': round(b, 1), 'nearest_m': round(r, 1),
                           'in_path_s': round(n * args.step / FPS, 1)} for (cam, i), (a, b, r, n) in sorted(ped_spans.items(), key=lambda kv: kv[1][0])],
    'processing_fps': round(frame / (time.perf_counter() - t_start), 1),
    'detector': {cam: stats(t) for cam, t in times.items()}, 'tracker_ms_median': round(float(np.median(track_ms)) * 1e3, 3),
    'road_wide_eof_skew_ms_max': round(float(max(skew)), 1),
  }
  args.out.with_suffix('.json').write_text(json.dumps(summary, indent=2))
  print(json.dumps(summary, indent=2))
  print(f'wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
  main()
