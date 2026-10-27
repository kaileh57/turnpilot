#!/usr/bin/env python3
"""Run TurnPilot's lane paint classifier on a public comma route and draw modelV2's four lane lines with what it read.

  DEV=METAL lane_paint_video.py 5beb9b58bd12b691/0000010a--a51155e496 --start 840 --seconds 60 -o lane_paint.mp4
  lane_paint_video.py <route> --start 60 --seconds 60 --frames-every 3 --frames-dir out/  # stills to label by hand

The classifier runs at the detector's rate (every 2nd frame, 10 Hz) on the road camera, in camerad's padded NV12 layout,
with the route's own modelV2 and calibration, and with YOLO26's boxes on the same frame masking what they cover (as the
detector does; --no-detect leaves them out). Each line is drawn over 8-25 m with a dot per sample: green paint, gray road,
red not readable. Its label is the classifier's smoothed output: kind, color, confidence, coverage.

--frames-every saves stills for labeling: <name>_blind.jpg marks only the ends of the two ego lines (1 left, 2 right), so the
paint can be judged without seeing the classifier's answer, and <name>.jpg is the same frame drawn in full. Their
predictions go to frames.json beside them, keyed by name. See replay_video.py for how this Mac reads the route.

  lane_paint_video.py --score labels.json frames/frames.json ...

prints the confusion table of hand labels ({name: {"left": "broken white", "right": "solid yellow"}}, "unclear" skipped)
against those predictions, and the gate for the lane planner's broken-white rule.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.turnpilot.detector.decoder import decode
from openpilot.selfdrive.turnpilot.detector.graph import DEFAULT_MODEL, make_detector
from openpilot.selfdrive.turnpilot.lane_planner import BROKEN_WHITE_CONF_MIN
from openpilot.selfdrive.turnpilot.scene.lane_paint import INVALID, PAINT, XS, LanePaintClassifier, project, sample_lines
from openpilot.selfdrive.turnpilot.scene.ranging import calib_from_view, road_lens
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.tools.lib.route import Route
from openpilot.tools.turnpilot.replay_video import FPS, SEGMENT_FRAMES, Log, camerad_nv12, fetch, open_encoder, outlined, tag, video_frames, video_size

LINE_COLORS = ((200, 120, 255), (255, 0, 255), (255, 255, 0), (120, 220, 255))  # BGR, laneLines 0-3
STATE_COLORS = {PAINT: (60, 230, 60), INVALID: (40, 40, 230)}
ROAD_COLOR = (150, 150, 150)
PAINT_COLORS = {'white': (255, 255, 255), 'yellow': (0, 220, 255), 'unknown': (180, 180, 180)}


def line_pixels(lens, rpy, xyz, xs) -> np.ndarray:
  x, y, z = xyz
  pts = np.stack([xs, np.interp(xs, x, y), np.interp(xs, x, z)], axis=1)
  u, v, front = project(lens, calib_from_view(rpy).T, pts)
  return np.stack([u, v], axis=1)[front]


def draw(img, lens, rpy, lines_xyz, probs, state, paint, blind: bool) -> None:
  for i, xyz in enumerate(lines_xyz):
    if probs[i] < 0.3 or (blind and i not in (1, 2)):
      continue
    px = line_pixels(lens, rpy, xyz, XS)
    if blind:  # the ends of the sampled stretch only, 0.5 m outboard, so the paint itself stays visible
      for x in (XS[0], XS[-1]):
        x0, y0, z0 = xyz
        y = float(np.interp(x, x0, y0)) + (0.5 if i == 2 else -0.5)
        u, v, _ = project(lens, calib_from_view(rpy).T, np.array([x, y, float(np.interp(x, x0, z0))]))
        cv2.circle(img, (int(u), int(v)), 5, LINE_COLORS[i], -1, cv2.LINE_AA)
        outlined(img, str(i), (int(u) + (8 if i == 2 else -20), int(v) - 6), 0.6, LINE_COLORS[i])
      continue
    for (u, v), s in zip(px, state[i], strict=True):
      if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
        cv2.circle(img, (int(u), int(v)), 2, STATE_COLORS.get(int(s), ROAD_COLOR), -1)
    p = paint[i]
    far = px[len(px) * 2 // 3] if len(px) else None
    if far is not None:
      label = f"{i} {p.kind} {p.color if p.kind in ('broken', 'solid') else ''} {p.confidence:.2f} c{p.coverage:.2f}"
      tag(img, label, int(far[0]) - (160 if i < 2 else -10), int(far[1]) - 10 - 26 * (i in (0, 3)), PAINT_COLORS[p.color], (0, 0, 0))


LABELS = ('broken white', 'solid white', 'solid yellow', 'broken yellow', 'none')
PREDICTED = ('broken white', 'broken white, low conf', 'solid white', 'solid yellow', 'broken yellow', 'broken, color unknown',
             'solid, color unknown', 'none', 'unknown')
SHORT = dict(zip(PREDICTED, ('BW', 'BW lowconf', 'SW', 'SY', 'BY', 'B ?color', 'S ?color', 'none', 'unknown'), strict=True))


def predicted(p: dict) -> str:
  if p['kind'] in ('unknown', 'none'):
    return p['kind']
  if p['color'] == 'unknown':
    return f"{p['kind']}, color unknown"
  if (p['kind'], p['color']) == ('broken', 'white') and p['confidence'] < BROKEN_WHITE_CONF_MIN:
    return 'broken white, low conf'
  return f"{p['kind']} {p['color']}"


def score(labels_path: Path, prediction_paths: list[Path]) -> dict:
  labels = json.loads(labels_path.read_text())
  preds = {}
  for path in prediction_paths:
    preds |= json.loads(path.read_text())
  table = {lab: dict.fromkeys(PREDICTED, 0) for lab in LABELS}
  frames = 0
  for name, lab in labels.items():
    if name not in preds:
      continue
    frames += 1
    for side, idx in (('left', 1), ('right', 2)):
      if lab[side] in table:
        table[lab[side]][predicted(preds[name]['paint'][idx])] += 1
  false_bw = sum(table[lab]['broken white'] for lab in LABELS if lab not in ('broken white', 'none'))
  bw = table['broken white']
  recall = bw['broken white'] / max(sum(bw.values()), 1)
  print(f"{frames} labeled frames, ego left and right lines. Rows: hand label; columns: predicted, B/S broken/solid, W/Y white/yellow, " +
        f"BW at confidence >= {BROKEN_WHITE_CONF_MIN} (the planner's bar)")
  print('label \\ predicted'.ljust(16) + ''.join(SHORT[c].rjust(11) for c in PREDICTED))
  for lab in LABELS:
    print(lab.ljust(16) + ''.join(str(table[lab][c]).rjust(11) for c in PREDICTED) + f"   ({sum(table[lab].values())})")
  gate = false_bw <= 1 and recall > 0.5
  print(f"solid or yellow read broken white: {false_bw}; broken white recognized: {recall:.0%}; gate {'passed' if gate else 'missed'}")
  return {'frames': frames, 'table': table, 'false_broken_white': false_bw, 'broken_white_recall': recall, 'gate_passed': gate}


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  ap.add_argument('route', nargs='?')
  ap.add_argument('--score', nargs='+', type=Path, help='labels.json, then one or more frames.json')
  ap.add_argument('--start', type=float, default=0)
  ap.add_argument('--seconds', type=float, default=60)
  ap.add_argument('-o', '--out', type=Path, help='an mp4 of every frame')
  ap.add_argument('--frames-every', type=float, help='save a still pair every this many seconds')
  ap.add_argument('--frames-dir', type=Path, default=Path('~/turnpilot-artifacts/phase3/lanepaint/frames').expanduser())
  ap.add_argument('--cache', type=Path, default=Path('~/.cache/turnpilot/replay').expanduser())
  ap.add_argument('--step', type=int, default=2, help='frames between classifier runs (the detector runs every 2nd)')
  ap.add_argument('--no-detect', action='store_true', help="don't mask samples behind YOLO26's boxes")
  ap.add_argument('--scale', type=float, default=0.5)
  args = ap.parse_args()
  if args.score:
    result = score(args.score[0], args.score[1:])
    args.score[0].with_name('confusion.json').write_text(json.dumps(result, indent=1))
    return

  route = Route(args.route)
  first = int(args.start // 60)
  skip = round((args.start - first * 60) * FPS)
  n_frames = round(args.seconds * FPS)
  segs = list(range(first, min(route.max_seg_number, first + (skip + n_frames - 1) // SEGMENT_FRAMES) + 1))
  files = {s: [fetch(paths()[s], args.cache) for paths in (route.log_paths, route.camera_paths)] for s in segs}
  w, h = video_size(files[segs[0]][1])
  log = Log([files[s][0] for s in segs])
  lens = road_lens(DEVICE_CAMERAS[(log.device_type, log.sensor)].narrow_road)
  stride, y_height, _, _ = get_nv12_info(w, h)
  run = lb = None
  if not args.no_detect:
    from tinygrad import Tensor, TinyJit
    lb, detect = make_detector(w, h, DEFAULT_MODEL)
    run = TinyJit(detect, prune=True)

  clf = LanePaintClassifier(lens)
  size = (round(w * args.scale) // 2 * 2, round(h * args.scale) // 2 * 2)
  encoder = open_encoder(args.out, *size) if args.out else None
  if args.frames_every:
    args.frames_dir.mkdir(parents=True, exist_ok=True)
  saved = {}
  times = []
  paint, state, boxes = None, None, np.zeros((0, 6))
  last = (0, 0)
  frame = 0
  next_still = args.start
  for seg in segs:
    for k, nv12 in enumerate(video_frames(files[seg][1], w, h)):
      if seg == segs[0] and k < skip:
        continue
      if frame >= n_frames:
        break
      t_route = seg * 60 + k / FPS
      last = frame_id, eof = log.stamp('road', seg, k, last)
      model = log.models.get(frame_id)
      cal = log.at('extrinsicsCalibration', eof)
      if frame % args.step == 0 and model is not None:
        padded = camerad_nv12(nv12, w, h)
        if run is not None:
          boxes = decode(run(input_frame=Tensor(padded).realize()).numpy(), lb)
        lines_xyz = [(np.array(ln.x), np.array(ln.y), np.array(ln.z)) for ln in model.laneLines]
        probs = list(model.laneLineProbs)
        rpy, calibrated = list(cal.rpyCalib), str(cal.calStatus) == 'calibrated'
        t0 = time.perf_counter()
        paint = clf.update(eof * 1e-9, padded, stride, y_height, rpy, calibrated, lines_xyz, probs, boxes)
        times.append(time.perf_counter() - t0)
        state, _ = sample_lines(padded, stride, y_height, lens, rpy, lines_xyz, boxes)  # again, only for the drawing
      want_still = args.frames_every and t_route >= next_still and paint is not None and frame % args.step == 0
      if (encoder or want_still) and paint is not None:
        img = cv2.cvtColor(nv12.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
        head = f"{t_route:6.1f} s  {log.at('carState', eof).vEgo * 2.237:.0f} mph"
        if want_still:
          name = f"s{seg:02d}_{k:04d}"
          blind = img.copy()
          draw(blind, lens, rpy, lines_xyz, probs, state, paint, True)
          outlined(blind, f"{name}  {head}", (12, 28), 0.7)
          cv2.imwrite(str(args.frames_dir / f"{name}_blind.jpg"), cv2.resize(blind, size, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
        for x0, y0, x1, y1 in boxes[:, :4].astype(int):
          cv2.rectangle(img, (x0, y0), (x1, y1), (255, 170, 40), 1)
        draw(img, lens, rpy, lines_xyz, probs, state, paint, False)
        outlined(img, head, (12, 28), 0.7)
        if want_still:
          cv2.imwrite(str(args.frames_dir / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
          saved[name] = {'t': round(t_route, 2), 'probs': [round(p, 2) for p in probs],
                         'paint': [{'kind': p.kind, 'color': p.color, 'confidence': round(p.confidence, 2), 'coverage': round(p.coverage, 2)} for p in paint]}
          next_still += args.frames_every
        if encoder:
          encoder.stdin.write(cv2.resize(img, size, interpolation=cv2.INTER_AREA).tobytes())
      frame += 1
    if frame >= n_frames:
      break
  if encoder:
    encoder.stdin.close()
    encoder.wait()
  if saved:
    path = args.frames_dir / 'frames.json'
    prev = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(prev | saved, indent=1))
  ms = np.array(times[3:] or times) * 1e3
  print(json.dumps({'frames': frame, 'runs': len(times), 'stills': len(saved), 'classifier_ms': {
    'median': round(float(np.median(ms)), 3), 'p90': round(float(np.percentile(ms, 90)), 3), 'max': round(float(ms.max()), 3)}}))


if __name__ == '__main__':
  main()
