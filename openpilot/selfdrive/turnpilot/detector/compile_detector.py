#!/usr/bin/env python3
"""Compile the TurnPilot detector into one TinyJit pickle per camera size, the way modeld's warp and driving pkls are built.

  DEV=USB+AMD:LLVM python3 compile_detector.py --retargetable             # chestnut attached: big_detector_{w}x{h}_tinygrad.pkl
  DEV=METAL HCQ_RUNTIME_DEV=CPU:CLANG JIT=2 python3 compile_detector.py   # mac (JIT=2 as in modeld's SConscript)

--retargetable drops the host-side compiled programs, which modeld rebuilds on load (it calls lower_and_compile), so a
pkl compiled on the Mac with the chestnut attached also loads on the comma four.

Run with the repo root and tinygrad_repo on PYTHONPATH. The jit takes one camera's NV12 buffer (input_frame, uint8
[stride * (y_height + uv_height)]) and returns YOLO26's (1, 300, 6). modeld calls it with views into the two frames it
already uploaded, so the self-test does the same: both frames in one device buffer, views alternating, each output
checked against an unjitted run, before and after the pickle round trip.
"""
import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from openpilot.selfdrive.turnpilot.detector.graph import DEFAULT_MODEL, INPUT_SIZES, MODELS_DIR, detector_pkl_path, make_detector, onnx_path
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


def frame_views(frames: np.ndarray):
  """Both frames in one device buffer, as modeld packs them, and a uint8 view of each."""
  from tinygrad import Tensor, dtypes
  from tinygrad.uop.ops import UOp
  packed = np.zeros(256 + frames.size, dtype=np.uint8)  # offset like modeld's packed npy inputs
  packed[256:] = frames.reshape(-1)
  buf = Tensor(packed).realize()._buffer()
  size = frames.shape[1]
  return [Tensor(UOp.from_buffer(buf.view(size, 256 + i * size).ensure_allocated(), dtypes.uint8)) for i in range(len(frames))]


def compile_one(cam_w: int, cam_h: int, model: str, out: Path, runs: int, retargetable: bool) -> None:
  from tinygrad import Device, Tensor, TinyJit
  from tinygrad.engine.realize import lower_and_compile
  from tinygrad_repo.examples.openpilot.helpers import dump_pickle, load_pickle, make_retargetable
  stride, y_height, uv_height, _ = get_nv12_info(cam_w, cam_h)
  frame_size = stride * (y_height + uv_height)
  lb, detect = make_detector(cam_w, cam_h, model)

  rng = np.random.default_rng(42)
  frames = rng.integers(0, 256, (2, frame_size), dtype=np.uint8)
  expected = [detect(Tensor(f).realize()).numpy() for f in frames]
  assert not np.array_equal(expected[0], expected[1]), "detector output doesn't depend on the frame"

  run = TinyJit(detect, prune=True)
  views = frame_views(frames)

  def check(fn, label):
    for i in (0, 1, 0, 1):
      np.testing.assert_allclose(fn(input_frame=views[i]).numpy(), expected[i], rtol=1e-4, atol=1e-3, err_msg=f'{label} frame {i}')

  check(run, 'jit')
  times = []
  for _ in range(runs):
    Device.default.synchronize()
    st = time.perf_counter()
    run(input_frame=views[len(times) % 2]).realize()
    Device.default.synchronize()
    times.append((time.perf_counter() - st) * 1e3)
  print(f'{model} {cam_w}x{cam_h} -> {lb.in_w}x{lb.in_h} on {Device.DEFAULT}: min {min(times):.2f} ms, median {np.median(times):.2f} ms')

  if retargetable:
    make_retargetable(run)
  specs = {'input_frame': ((frame_size,), np.dtype(np.uint8).str, Device.DEFAULT)}
  metadata = {'model': model, 'onnx_sha256': hashlib.sha256(onnx_path(model).read_bytes()).hexdigest(),
              'camera_size': (cam_w, cam_h), 'input_size': (lb.in_w, lb.in_h)}
  dump_pickle({'metadata': metadata, 'run': run, 'input_specs': specs,
               'output_specs': {'output': ((1, 300, 6), np.dtype(np.float32).str, Device.DEFAULT)}}, out)

  loaded = load_pickle(out, out_of_band=True)
  loaded['run'].captured._linear = lower_and_compile(loaded['run'].captured._linear)
  check(loaded['run'], 'pickle')
  print(f'wrote {out} ({out.stat().st_size / 1e6:.1f} MB)', flush=True)


def main():
  from tinygrad import Device
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--model', default=DEFAULT_MODEL, help='ONNX in selfdrive/turnpilot/models: yolo26n or yolo26s')
  ap.add_argument('--sizes', nargs='+', default=[f'{w}x{h}' for w, h in INPUT_SIZES], help='camera frame sizes, WxH')
  ap.add_argument('--out', type=Path, default=MODELS_DIR)
  ap.add_argument('--benchmark-runs', type=int, default=20)
  ap.add_argument('--retargetable', action='store_true', help='rebuild host programs on load (for a pkl built off the device)')
  args = ap.parse_args()
  chestnut = 'AMD' in Device.DEFAULT
  for size in args.sizes:
    w, h = map(int, size.split('x'))
    compile_one(w, h, args.model, detector_pkl_path(w, h, chestnut, args.out), args.benchmark_runs, args.retargetable)


if __name__ == '__main__':
  main()
