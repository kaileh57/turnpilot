"""The pose log of a drive turned into the poses chase_render.py teleports the car to. Pure math, no game connection.

The log (route_drive's and the scenario runner's `poses`, 20 Hz) is the AdvancedIMU's pose at cameras.MOUNT. It holds
each IMU batch for one to three rows and trails the game by 150 to 210 ms, so the first row of each new value is kept,
moved back by POSE_LAG_S, fitted onto the render's 20 Hz grid, and turned into the teleport position by IMU_OFFSET.
"""
import math

import numpy as np

POSE_LAG_S = 0.16  # the held pose trails the game by 150-210 ms; the first row of a new value is ~20 ms fresher than the mean
# vehicle frame (x left, -y forward, z up): the teleport position to the IMU at cameras.MOUNT, the IMU read on the car
# standing after a teleport, 2026-10-27; with it a re-teleported IMU lands within 1 cm of the logged pose
IMU_OFFSET = (0.33, -0.332, 1.066)
SMOOTH_S = 0.08    # s, Gaussian sigma of the fit through the kept rows: their times jitter by one 50 ms row
WINDOW_S = 0.25    # s each side of a grid time that the fit looks at
HOLD_S = 0.15      # a value held longer than this is the car standing: its last row is kept too
JUMP_M = 2.0       # a step this long at more than JUMP_MPS is a teleport, which the fit must not smooth over
JUMP_MPS = 40.0


def rot_matrix(q) -> np.ndarray:
  """Columns: the vehicle's x, y and z axes in world space, for a BeamNG quaternion (x, y, z, w). BeamNG turns the other
  way round from the usual sense (pose.beamng_quat), so this is the usual matrix of the conjugate."""
  x, y, z, w = -q[0], -q[1], -q[2], q[3]
  return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                   [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def teleport_pos(imu_pos, q) -> np.ndarray:
  return np.asarray(imu_pos, dtype=float) - rot_matrix(q) @ np.array(IMU_OFFSET)


def kept_rows(rows: np.ndarray) -> np.ndarray:
  """Rows (wall s, x, y, z, qx, qy, qz, qw) where a new value came in, and the last row of every value held for HOLD_S."""
  keep = []
  start = 0
  for i in range(1, len(rows) + 1):
    if i == len(rows) or not np.array_equal(rows[i, 1:], rows[start, 1:]):
      keep.append(start)
      if rows[i - 1, 0] - rows[start, 0] > HOLD_S:
        keep.append(i - 1)
      start = i
  return rows[keep]


def segments(rows: np.ndarray) -> list[np.ndarray]:
  """Split at teleports."""
  if len(rows) < 2:
    return [rows]
  step = np.linalg.norm(np.diff(rows[:, 1:4], axis=0), axis=1)
  dt = np.maximum(np.diff(rows[:, 0]), 1e-3)
  cuts = np.flatnonzero((step > JUMP_M) & (step / dt > JUMP_MPS)) + 1
  return np.split(rows, cuts)


def fit(seg: np.ndarray, t: float) -> np.ndarray:
  """(x, y, z, qx, qy, qz, qw) at t: a Gaussian-weighted straight-line fit through the segment's rows near t, the
  quaternion normalized after (its rows put in one hemisphere first). Outside the rows, the nearest row."""
  ts = seg[:, 0]
  if t <= ts[0]:
    return seg[0, 1:].copy()
  if t >= ts[-1]:
    return seg[-1, 1:].copy()
  near = np.abs(ts - t) <= WINDOW_S
  if near.sum() < 2:
    k = int(np.searchsorted(ts, t))
    a = (t - ts[k - 1]) / (ts[k] - ts[k - 1])
    near = np.zeros(len(ts), dtype=bool)
    near[k - 1:k + 1] = True
    vals = seg[near, 1:].copy()
    vals[1, 3:] *= np.sign(vals[0, 3:] @ vals[1, 3:]) or 1.0
    out = vals[0] * (1 - a) + vals[1] * a
  else:
    tt, vals = ts[near] - t, seg[near, 1:].copy()
    ref = vals[np.argmin(np.abs(tt)), 3:]
    vals[:, 3:] *= np.where(vals[:, 3:] @ ref < 0, -1.0, 1.0)[:, None]
    w = np.exp(-0.5 * (tt / SMOOTH_S) ** 2)
    A = np.stack([np.ones_like(tt), tt], axis=1) * w[:, None]
    coef, *_ = np.linalg.lstsq(A, vals * w[:, None], rcond=None)
    out = coef[0]
  out[3:] /= np.linalg.norm(out[3:])
  return out


def ego_track(rows, lag: float = POSE_LAG_S):
  """The log's rows, cleaned: (segments, first and last game time). Times move back by lag, onto the game's clock."""
  rows = np.array([[r[0], *r[2:9]] for r in rows], dtype=float)
  rows[:, 0] -= lag
  rows = kept_rows(rows)
  return segments(rows), float(rows[0, 0]), float(rows[-1, 0])


def ego_at(segs: list[np.ndarray], t: float) -> np.ndarray:
  for k, seg in enumerate(segs):
    nxt = segs[k + 1][0, 0] if k + 1 < len(segs) else math.inf
    if t < nxt:
      return fit(seg, t)
  return fit(segs[-1], t)


def actor_at(rows: list, t: float) -> tuple | None:
  """(x, y, z, heading) of a scenario actor at wall time t, linear between its logged rows; None outside them."""
  if not rows or t < rows[0][0] or t > rows[-1][0] + 0.5:
    return None
  ts = [r[0] for r in rows]
  k = int(np.searchsorted(ts, t))
  if k == 0 or k >= len(rows):
    return tuple(rows[min(k, len(rows) - 1)][1:5])
  r0, r1 = rows[k - 1], rows[k]
  a = (t - r0[0]) / max(r1[0] - r0[0], 1e-3)
  dh = (r1[4] - r0[4] + math.pi) % (2 * math.pi) - math.pi
  return (r0[1] + a * (r1[1] - r0[1]), r0[2] + a * (r1[2] - r0[2]), r0[3] + a * (r1[3] - r0[3]), r0[4] + a * dh)
