#!/usr/bin/env python3
"""Export YOLO26 (Ultralytics, AGPL-3.0) to the end-to-end ONNX files the TurnPilot detector compiles.

ultralytics isn't an openpilot dependency, so run this from a throwaway venv:

  uv venv /tmp/yolo-venv && uv pip install --python /tmp/yolo-venv/bin/python ultralytics onnx
  /tmp/yolo-venv/bin/python export_yolo26.py --out /tmp/yolo-export

The input size is dynamic, so one file serves every camera: compile_detector.py fixes it per camera size
(640x384 for 1344x760 frames, 640x416 for 1928x1208). nms=False keeps YOLO26's NMS-free head: output0 is
(1, 300, 6) = x0, y0, x1, y1, conf, cls in input pixels.
"""
import argparse
import hashlib
import os
import shutil
from pathlib import Path


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--out', type=Path, required=True, help='work dir; the .pt weights download here')
  ap.add_argument('--models', nargs='+', default=['yolo26n', 'yolo26s'])
  args = ap.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  os.chdir(args.out)
  from ultralytics import YOLO

  for name in args.models:
    pt = args.out / f'{name}.pt'
    model = YOLO(str(pt))  # downloads github.com/ultralytics/assets/releases/download/v8.4.0/<name>.pt if missing
    exported = Path(model.export(format='onnx', imgsz=(384, 640), opset=17, simplify=False, device='cpu', nms=False, dynamic=True))
    dst = args.out / f'{name}.onnx'
    shutil.move(exported, dst)
    for p in (pt, dst):
      print(f'{p.name} {p.stat().st_size} sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}')


if __name__ == '__main__':
  main()
