#!/usr/bin/env python3
"""A VisionIPC client like modeld's: reads the road and wide streams and reports rate, duplicates, frame id gaps, the
road-wide timestamp difference and frame age (receive time minus timestamp_sof)."""
import argparse
import hashlib
import time

import numpy as np

import openpilot.cereal.messaging as messaging
from msgq.visionipc import VisionIpcClient
from openpilot.cereal.visionipc import VisionStreamType

STREAMS = (VisionStreamType.VISION_STREAM_NARROW_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD)


def measure(duration: float, connect_timeout: float = 60.0) -> dict:
  clients = [VisionIpcClient("camerad", s, False) for s in STREAMS]
  t_end = time.monotonic() + connect_timeout
  while not all(c.connect(False) for c in clients):
    if time.monotonic() > t_end:
      raise TimeoutError("no camerad VisionIPC streams")
    time.sleep(0.1)
  sm = messaging.SubMaster(['narrowRoadCameraState', 'wideRoadCameraState'])
  road, wide = clients
  ids, ages, sofs_r, sofs_w, hashes, recv_t = [], [], {}, {}, [], []
  t_end = time.monotonic() + duration
  while time.monotonic() < t_end:
    buf = road.recv(100)
    if buf is None:
      continue
    now = time.monotonic_ns()
    ids.append(road.frame_id)
    sofs_r[road.frame_id] = road.timestamp_sof
    ages.append((now - road.timestamp_sof) / 1e6)
    recv_t.append(now / 1e9)
    hashes.append(hashlib.blake2b(bytes(buf.data[::4099]), digest_size=8).digest())
    while wide.recv(0) is not None:
      sofs_w[wide.frame_id] = wide.timestamp_sof
  sm.update(0)
  common = [i for i in sofs_r if i in sofs_w]
  dts = np.diff(recv_t) * 1e3 if len(recv_t) > 1 else np.array([0.0])
  out = {
    "frames": len(ids),
    "rate_hz": len(ids) / duration,
    "unique_rate_hz": len(set(hashes)) / duration,
    "frame_id_gaps": int(sum(max(0, b - a - 1) for a, b in zip(ids, ids[1:], strict=False))),
    "sof_mismatch_max_ms": max((abs(sofs_r[i] - sofs_w[i]) / 1e6 for i in common), default=None),
    "pairs_matched": len(common),
    "age_med_ms": float(np.median(ages)) if ages else None,
    "age_p90_ms": float(np.percentile(ages, 90)) if ages else None,
    "interval_med_ms": float(np.median(dts)),
    "interval_p90_ms": float(np.percentile(dts, 90)),
    "sensor": str(sm['narrowRoadCameraState'].sensor) if sm.seen['narrowRoadCameraState'] else None,
    "size": (road.width, road.height),
  }
  return out


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--duration', type=float, default=10.0)
  args = p.parse_args()
  for k, v in measure(args.duration).items():
    print(f"{k}: {v:.1f}" if isinstance(v, float) else f"{k}: {v}")
