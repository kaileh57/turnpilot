#!/usr/bin/env python3
"""Hour-0 chestnut smoke test: load the big model the way modeld does, then time the upload and the model.

Run it with nothing else holding chestnut (modeld, the manager, a WebUSB tab):
  python -m openpilot.tools.turnpilot.chestnut_smoke [--width 1344 --height 760 --runs 200]
"""
import argparse
import time

import numpy as np

from openpilot.selfdrive.modeld.modeld import ModelState  # sets modeld's tinygrad env before tinygrad loads
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.helpers import chestnut_compiled, chestnut_present
from tinygrad.device import Device


def stats(ts: list[float]) -> str:
  ms = np.array(ts) * 1e3
  return f"median {np.median(ms):.2f} ms, p95 {np.percentile(ms, 95):.2f} ms, max {ms.max():.2f} ms (n={len(ms)})"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--width', type=int, default=1344)
  parser.add_argument('--height', type=int, default=760)
  parser.add_argument('--runs', type=int, default=200)
  parser.add_argument('--hz', type=float, default=0., help='pace model.run like modeld, 0 runs back to back')
  args = parser.parse_args()

  print(f"chestnut_present={chestnut_present()} chestnut_compiled={chestnut_compiled()}")
  assert chestnut_present() and chestnut_compiled()

  st = time.monotonic()
  model = ModelState(args.width, args.height, True)
  print(f"big model loaded on {model.model_device} in {time.monotonic() - st:.1f} s")
  st = time.monotonic()
  model.warmup()
  print(f"warmup in {time.monotonic() - st:.1f} s")
  dev = Device[model.model_device]

  # one packed upload: both NV12 frames plus the small inputs, as modeld sends it every frame
  rng = np.random.default_rng(0)
  ups = []
  for _ in range(args.runs):
    model.packed_input[:] = rng.integers(0, 255, model.packed_input.shape, dtype=np.uint8)
    dev.synchronize()
    t = time.perf_counter()
    model.input_device.copy_from(model.input_host)
    dev.synchronize()
    ups.append(time.perf_counter() - t)
  print(f"upload {model.packed_input.nbytes} B (2 x {model.frame_copy_size} B frames): {stats(ups)}")

  frames = {k: rng.integers(0, 255, model.frame_copy_size, dtype=np.uint8) for k in model.vision_input_names}
  eye = {k: np.eye(3, dtype=np.float32) for k in model.vision_input_names}
  inputs = {'desire_pulse': np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32),
            'traffic_convention': np.array([1, 0], dtype=np.float32),
            'action_t': np.array([0.3, 0.5], dtype=np.float32)}
  runs = []
  for _ in range(args.runs):
    t = time.perf_counter()
    out = model.run(frames, eye, inputs)
    runs.append(time.perf_counter() - t)
    if args.hz > 0:
      time.sleep(max(0., 1 / args.hz - runs[-1]))
  assert all(np.all(np.isfinite(v)) for v in out.values() if isinstance(v, np.ndarray))
  print(f"model.run (upload + warp + model + readback) at {args.hz or 'max'} Hz: {stats(runs)}")


if __name__ == "__main__":
  main()
