#!/usr/bin/env python3
"""Rebuild a drive's camera light reads from its qcamera video and replay them through the planner.

  DEV=METAL light_proxy.py detect ROUTE --out DIR [--segments 3,4]
  light_proxy.py score ROUTE --proxy DIR --out PREFIX [--pkg-root DIR]

detect: every frame of each segment's qcamera.ts through the full-frame detector (graph.make_detector, yolo26n from the
ONNX, the graph the car's pkl holds), scaled back to the road camera's size (the qcamera is the whole frame scaled to
526x330, each axis on its own). A frame is indexed by its place in the segment's video against the segment's own
qNarrowRoadEncodeIdx (its segmentId: the frame's frameId and timestampEof), never by an assumed rate: one segment's
container says 60000/1001 fps and decodes to three times its frames with a frame-rate conversion, so the frames are decoded
as they are stored (-fps_mode passthrough) and a segment whose count disagrees with its index is refused, loudly. Per frame
the class-9 boxes down to decoder.LIGHT_GREEN_CONF are kept with the NV12 pixels around each (all the light reader reads),
DIR/<route>--<seg>.pkl.

score: drive_scorecard's open-loop replay of ROUTE (the package from --pkg-root, or the tree), with every logged road
camera run's lights (turnpilotDetections lights, lightRejects, lightStatus) replaced by the package's own LightReader on
that run's frame of the proxy: the same reader state across runs, as modeld keeps it, the calibration and the driving
model's path from the log. A run whose frame the video lacks keeps no lights (lightStatus noCalibration, nothing read).
Writes the scorecard's PREFIX.md and PREFIX.json, the per-frame records (PREFIX_<route>.pkl) and every run's published
heads (PREFIX_<route>_reads.pkl: [(t, [(state, read, reason, distance, near, bearing, governs, conf)])]).

What it cannot show: the proxy is a 0.39x video upscaled, so the detector boxes less, later and with other confidences
than on the car's full frame; it reads at the logged build's road run times, and the car's motion is the logged one.
"""
import argparse
import glob
import os
import pickle
import subprocess
import sys

import numpy as np

CAM_W, CAM_H = 1344, 760     # the comma four's road camera (mici, os04c10): the frame the detector and the reader see
QCAM_W, QCAM_H = 526, 330
PATCH_MARGIN = 0.5           # of a box's height kept around it (the reader reads inside the box; a sample crops a sixth)


def segment_dirs(logs_dir: str, route: str) -> list[str]:
  dirs = glob.glob(os.path.join(logs_dir, '*', f'{route}--*'))
  return sorted((d for d in dirs if os.path.exists(os.path.join(d, 'qcamera.ts'))), key=lambda d: int(d.rsplit('--', 1)[1]))


def encode_index(qlog: str) -> dict:
  """segmentId -> (frameId, timestampEof ns) of the segment's qcamera frames."""
  from openpilot.tools.lib.logreader import LogReader
  out = {}
  for m in LogReader(qlog, only_union_types=True):
    if m.which() == 'qNarrowRoadEncodeIdx':
      e = m.qNarrowRoadEncodeIdx
      out[int(e.segmentId)] = (int(e.frameId), int(e.timestampEof))
  return out


def frames_nv12(fn: str):
  """The video's frames as stored, scaled to CAM_W x CAM_H, as tight NV12."""
  size = CAM_W * CAM_H * 3 // 2
  p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', fn, '-fps_mode', 'passthrough', '-vf', f'scale={CAM_W}:{CAM_H}',
                        '-f', 'rawvideo', '-pix_fmt', 'nv12', 'pipe:1'], stdout=subprocess.PIPE)
  try:
    while True:
      buf = p.stdout.read(size)
      if len(buf) < size:
        break
      yield np.frombuffer(buf, np.uint8)
  finally:
    p.stdout.close()
    p.wait()


def count_frames(fn: str) -> int:
  out = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
                                 'stream=nb_read_packets', '-of', 'csv=p=0', fn]).decode()
  return int(out.split()[0].split(',')[0])  # the transport stream lists its one video stream once per program


def camerad_layout(tight: np.ndarray) -> np.ndarray:
  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
  stride, y_height, uv_height, _ = get_nv12_info(CAM_W, CAM_H)
  out = np.zeros((y_height + uv_height, stride), dtype=np.uint8)
  out[:CAM_H, :CAM_W] = tight[:CAM_W * CAM_H].reshape(CAM_H, CAM_W)
  out[y_height:y_height + CAM_H // 2, :CAM_W] = tight[CAM_W * CAM_H:].reshape(CAM_H // 2, CAM_W)
  return out.reshape(-1)


def patch(tight: np.ndarray, box) -> tuple:
  """(x0, y0, luma rows, chroma rows) around a box, on even rows and columns."""
  x0, y0, x1, y1 = (float(b) for b in box[:4])
  m = max(2., (y1 - y0) * PATCH_MARGIN)
  px0, py0 = max(0, int(x0 - m) // 2 * 2), max(0, int(y0 - m) // 2 * 2)
  px1, py1 = min(CAM_W, (int(np.ceil(x1 + m)) + 1) // 2 * 2), min(CAM_H, (int(np.ceil(y1 + m)) + 1) // 2 * 2)
  luma = tight[:CAM_W * CAM_H].reshape(CAM_H, CAM_W)[py0:py1, px0:px1].copy()
  chroma = tight[CAM_W * CAM_H:].reshape(CAM_H // 2, CAM_W)[py0 // 2:py1 // 2, px0:px1].copy()
  return px0, py0, luma, chroma


def detect(args) -> None:
  from tinygrad import Tensor, TinyJit
  from openpilot.selfdrive.turnpilot.detector.decoder import LIGHT_GREEN_CONF, decode_lights
  from openpilot.selfdrive.turnpilot.detector.graph import make_detector
  os.makedirs(args.out, exist_ok=True)
  lb, run = make_detector(CAM_W, CAM_H)
  jit = TinyJit(run, prune=True)
  want = {int(s) for s in args.segments.split(',')} if args.segments else None
  for d in segment_dirs(args.logs_dir, args.route):
    seg = int(d.rsplit('--', 1)[1])
    out = os.path.join(args.out, f'{args.route}--{seg}.pkl')
    if (want is not None and seg not in want) or os.path.exists(out):
      continue
    idx = encode_index(os.path.join(d, 'qlog.zst'))
    video = os.path.join(d, 'qcamera.ts')
    n = count_frames(video)
    if n != len(idx):
      print(f'segment {seg}: {n} video frames against {len(idx)} encode index entries: refused', file=sys.stderr)
      continue
    rows = []
    for i, tight in enumerate(frames_nv12(video)):
      raw = jit(input_frame=Tensor(camerad_layout(tight)).realize()).numpy()
      boxes = decode_lights(raw, lb, LIGHT_GREEN_CONF)
      fid, eof = idx.get(i, (None, None))
      rows.append((i, fid, eof, np.asarray(boxes, dtype=np.float32), [patch(tight, b) for b in boxes]))
    if len(rows) != len(idx):
      print(f'segment {seg}: decoded {len(rows)} frames against {len(idx)}: refused', file=sys.stderr)
      continue
    with open(out + '.tmp', 'wb') as f:
      pickle.dump({'route': args.route, 'segment': seg, 'w': CAM_W, 'h': CAM_H, 'rows': rows}, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(out + '.tmp', out)
    print(f'segment {seg}: {len(rows)} frames, {sum(len(r[3]) for r in rows)} light boxes', flush=True)


def load_proxy(proxy_dir: str, route: str) -> dict:
  """frameId -> (eof ns, boxes, patches) over the route's segments."""
  out = {}
  for p in glob.glob(os.path.join(proxy_dir, f'{route}--*.pkl')):
    with open(p, 'rb') as f:
      data = pickle.load(f)
    for _, fid, eof, boxes, patches in data['rows']:
      if fid is not None:
        out[fid] = (eof, boxes, patches)
  return out


def sparse_frame(patches) -> np.ndarray:
  """A camerad-layout NV12 frame holding only the patches (the reader reads nothing outside its boxes)."""
  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
  stride, y_height, uv_height, _ = get_nv12_info(CAM_W, CAM_H)
  img = np.zeros((y_height + uv_height, stride), dtype=np.uint8)
  img[y_height:] = 128
  for x0, y0, luma, chroma in patches:
    img[y0:y0 + luma.shape[0], x0:x0 + luma.shape[1]] = luma
    img[y_height + y0 // 2:y_height + y0 // 2 + chroma.shape[0], x0:x0 + chroma.shape[1]] = chroma
  return img.reshape(-1)


class ProxyLights:
  """The package's LightReader over the proxy, one reader for the drive (modeld's CameraLightReader)."""
  def __init__(self, proxy: dict):
    from openpilot.common.transformations.camera import DEVICE_CAMERAS
    from openpilot.selfdrive.turnpilot.detector import light_state, modeld_detector
    from openpilot.selfdrive.turnpilot.scene import ranging, tracker
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
    self.proxy, self.md, self.ranging, self.tracker = proxy, modeld_detector, ranging, tracker
    self.reader = light_state.LightReader(ranging.road_lens(DEVICE_CAMERAS[('mici', 'os04c10')].narrow_road))
    self.stride, self.y_height, _, _ = get_nv12_info(CAM_W, CAM_H)
    self.cal = None
    self.paths: dict = {}         # frameId -> the driving model's path (tracker.predicted_path) on that frame
    self.reads: list = []
    self.missing = 0
    self.runs = 0
    self.pathless = 0
    self.eof_off: list = []       # the logged run's timestampEof less its proxy frame's: 0 when the frames line up

  def see_models(self, real, msgs) -> None:
    """Every logged modelV2's path by its frameId: a run reads its own frame's path, as modeld's reader does."""
    for _, w, b in msgs:
      if w == 'modelV2':
        m = real.log_from_bytes(b).modelV2
        self.paths[int(m.frameId)] = self.tracker.predicted_path(m)

  def see(self, w: str, ev) -> None:
    if w == 'extrinsicsCalibration':
      self.cal = ev.extrinsicsCalibration

  def reread(self, real, b: bytes, t: float) -> bytes:
    ev = real.log_from_bytes(b)
    d = ev.turnpilotDetections
    if str(d.camera) != 'road' or str(getattr(d, 'view', 'full')) != 'full' or str(d.lightStatus) == 'wideCamera':
      return b
    self.runs += 1
    mb = ev.as_builder()
    out = mb.turnpilotDetections
    hit = self.proxy.get(int(d.frameId))
    calibrated = self.cal is not None and str(self.cal.calStatus) == 'calibrated'
    if hit is None or not calibrated:
      self.missing += hit is None
      out.init('lights').init('lights', 0)
      out.init('lightRejects', 0)
      out.lightStatus = 'noCalibration'
      self.reads.append((t, None))
      return mb.to_bytes()
    eof, boxes, patches = hit
    path = self.paths.get(int(d.frameId))
    self.pathless += path is None
    self.eof_off.append(int(d.timestampEof) - int(eof))
    heads = self.reader.read(eof * 1e-9, sparse_frame(patches), self.stride, self.y_height, list(self.cal.rpyCalib),
                             self.ranging.camera_height(self.cal), boxes, path)
    self.md.fill_lights(out.lights, heads, False)
    rejects = list(getattr(self.reader, 'rejects', []) or [])
    if hasattr(self.md, 'fill_heads'):
      self.md.fill_heads(out.init('lightRejects', len(rejects)), rejects)
    out.lightStatus = 'read'
    self.reads.append((t, [(h.state, h.read, h.reason, h.distance, getattr(h, 'near', None), h.bearing, h.governs, h.conf)
                           for h in heads]))
    return mb.to_bytes()


def score(args) -> None:
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  import drive_scorecard as ds
  td, hooks = ds.load_package(args.pkg_root)
  from openpilot.cereal import messaging
  data = ds.load_route(args.route, args.logs_dir, args.cache_dir)
  frames = ds.build_frames(data['msgs'])
  px = ProxyLights(load_proxy(args.proxy, args.route))
  px.see_models(messaging, data['msgs'])
  for f in frames:
    t = f['t'] / 1e9
    if 'extrinsicsCalibration' in f['msgs']:
      px.see('extrinsicsCalibration', messaging.log_from_bytes(f['msgs']['extrinsicsCalibration']))
    f['dets'] = [px.reread(messaging, b, t) for b in f['dets']]
    if 'turnpilotDetections' in f['msgs']:
      f['msgs']['turnpilotDetections'] = f['dets'][-1] if f['dets'] else f['msgs']['turnpilotDetections']
  off = np.array(px.eof_off or [0])
  equal, worst = int((off == 0).sum()), abs(off).max() / 1e6
  print(f'{px.runs} road runs re-read, {px.missing} with no proxy frame, {px.pathless} with no model path; frame time against ' +
        f'the proxy frame: {equal} equal, max off {worst:.1f} ms', flush=True)
  h = ds.Harness(td, hooks, frames, data['params'], data['wall_off'])
  recs = h.run()
  name = args.route.split('--', maxsplit=1)[0]
  with open(f'{args.out}_{name}.pkl', 'wb') as f:
    pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
  with open(f'{args.out}_{name}_reads.pkl', 'wb') as f:
    pickle.dump(px.reads, f, protocol=pickle.HIGHEST_PROTOCOL)
  rows = []
  take_control = tuple(getattr(hooks, 'TAKE_CONTROL_PROMPTS', ()))
  for k, row in enumerate(ds.find_maneuvers(recs)):
    s = ds.score_row(recs, row, take_control)
    s['index'], s['route'] = k, name
    s['t_pass_rel'] = s['t_pass'] - recs[0]['t']
    rows.append(s)
  totals = ds.drive_totals(recs, rows, False, h.sim_takeovers)
  for s in rows:
    for key in ('t_start', 't_pass', 't_end'):
      s.pop(key)
  rt = {'route': name, 'frames': len(frames), 'replay_s': 0., 'rows': rows, 'totals': totals,
        'agreement': {}, 'agreement_n': 0, 'errors': dict(h.errors.most_common(8)), 'first_tracebacks': {}}
  label = args.label or (os.path.basename(os.path.normpath(args.pkg_root)) if args.pkg_root else 'working tree')
  ds.write_outputs(args.out, {'label': f'{label}, proxy light reads', 'logged_engagement': False,
                              'package': td.__file__.rsplit('/selfdrive/', 1)[-1], 'pkg_root_given': bool(args.pkg_root),
                              'cache_version': ds.CACHE_VERSION, 'thresholds': {}}, [rt])
  print(name, rt['totals']['verdicts'], 'errors', rt['errors'])


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  sub = ap.add_subparsers(dest='cmd', required=True)
  a = sub.add_parser('detect')
  a.add_argument('route')
  a.add_argument('--out', required=True)
  a.add_argument('--segments', help='comma-separated segment numbers (default all)')
  a.add_argument('--logs-dir', default=os.path.expanduser('~/turnpilot_data/logs'))
  b = sub.add_parser('score')
  b.add_argument('route')
  b.add_argument('--proxy', required=True)
  b.add_argument('--out', required=True)
  b.add_argument('--pkg-root')
  b.add_argument('--label')
  b.add_argument('--logs-dir', default=os.path.expanduser('~/turnpilot_data/logs'))
  b.add_argument('--cache-dir', default=os.environ.get('TP_SCORECARD_CACHE', os.path.expanduser('~/.cache/tp_scorecard')))
  args = ap.parse_args(argv)
  detect(args) if args.cmd == 'detect' else score(args)


if __name__ == '__main__':
  main()
