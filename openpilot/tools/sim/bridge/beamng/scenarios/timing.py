"""Timing for scenario actors: where a walking mannequin is, when a walker starts and when oncoming cars go, and the
timestamps of a BeamNG AI script. Pure functions of positions (BeamNG world meters, x east, y north) and speeds.
"""
import math

import numpy as np

WALK_SPEED = 1.4  # m/s, a brisk pedestrian
PROFILE_DT = 0.02  # s, integration step for speed profiles
NODE_SPACING = 4.0  # m between AI script nodes


def arc_lengths(pts) -> np.ndarray:
  pts = np.asarray(pts, dtype=float)
  return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))]


def point_at(pts, s: float):
  """(point, heading) at arc length s along a polyline, clamped to its ends. Heading is radians counterclockwise from east."""
  pts = np.asarray(pts, dtype=float)
  seg = arc_lengths(pts)
  s = min(max(s, 0.0), float(seg[-1]))
  i = min(max(1, int(np.searchsorted(seg, s))), len(pts) - 1)
  f = (s - seg[i - 1]) / max(seg[i] - seg[i - 1], 1e-9)
  d = pts[i] - pts[i - 1]
  return pts[i - 1] + f * d, math.atan2(d[1], d[0])


def resample(pts, spacing: float = NODE_SPACING) -> np.ndarray:
  """The polyline with points every `spacing` m (and its last point)."""
  seg = arc_lengths(pts)
  s = np.r_[np.arange(0.0, seg[-1], spacing), seg[-1]]
  return np.array([point_at(pts, float(v))[0] for v in s])


def walker_pose(path, t: float, speed: float = WALK_SPEED):
  """(point, heading) of a walker t s after it set off along path; it stands at the end once there."""
  return point_at(path, speed * max(t, 0.0))


def eta(dist: float, v: float, v_min: float = 1.0) -> float:
  """Seconds to cover dist at the current speed (no slower than v_min, so a car rolling up isn't infinitely far)."""
  return max(dist, 0.0) / max(v, v_min)


def walker_should_start(d_ego: float, v_ego: float, step_in_m: float, lead_s: float, speed: float = WALK_SPEED) -> bool:
  """True once the walker must set off to step off the curb lead_s before the ego, at its current speed, reaches the
  crosswalk d_ego m ahead. step_in_m: from the walker's start to the road edge."""
  return eta(d_ego, v_ego) <= step_in_m / speed + lead_s


def profile_times(s_nodes, v: float, accel: float, s_change: float = math.inf, v_change: float = 0.0,
                  dt: float = PROFILE_DT) -> list[float]:
  """Seconds from a standing start to reach each arc length in s_nodes (increasing): speed up at `accel` to v, and from
  s_change on change to v_change at the same rate. A node the car never reaches (it stopped short) gets math.inf."""
  out = []
  t = s = vel = 0.0
  for target in s_nodes:
    while s < target:
      want = v if s < s_change else v_change
      vel = max(vel + float(np.clip(want - vel, -accel * dt, accel * dt)), 0.0)
      if vel < 1e-3 and want < 1e-3:
        return out + [math.inf] * (len(s_nodes) - len(out))
      s += vel * dt
      t += dt
    out.append(t - (s - target) / vel if vel > 1e-3 else t)
  return out


def script_nodes(path, v: float, accel: float, s_change: float = math.inf, v_change: float = 0.0,
                 spacing: float = NODE_SPACING) -> list[dict]:
  """A BeamNG AI script ({x, y, z, t} nodes, t from the release) along path from a standing start. Nodes the car would
  not reach are dropped; ai.set_script needs at least three."""
  pts = resample(path, spacing)
  times = profile_times(arc_lengths(pts), v, accel, s_change, v_change)
  return [{'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2]) if len(p) > 2 else 0.0, 't': float(t)}
          for p, t in zip(pts, times, strict=True) if math.isfinite(t)]


def travel_time(s: float, v: float, accel: float) -> float:
  """Seconds from a standing start to cover s m, speeding up at accel to v."""
  return profile_times([s], v, accel)[0]


def stop_eta(dist: float, v: float, v_min: float = 1.0) -> float:
  """Seconds to come to a stop dist m ahead, braking evenly from v to 0 there: 2 dist / v. Exact once the car is braking
  for it, whatever its rate; while it still cruises, later than it will be (the release waits for the braking)."""
  return 2.0 * max(dist, 0.0) / max(v, v_min)


def oncoming_should_release(d_ego_hold: float, v_ego: float, s_car: float, v_car: float, accel: float,
                            arrive_after_s: float) -> bool:
  """True once oncoming cars standing s_car m before the junction must go so the first arrives there arrive_after_s
  after the ego has stopped at its hold line d_ego_hold m ahead (an unprotected left stops there for the driver's go).
  At its current speed it would get there sooner: the cars arrived before the go, not after it. Already late releases at
  once."""
  return stop_eta(d_ego_hold, v_ego) + arrive_after_s <= travel_time(s_car, v_car, accel)


def pacer_drop_s(s_start: float, s_junction: float, window_start_m: float, v: float, block_s: float) -> float:
  """Meters into its script (which starts s_start along its lane) the pacer starts dropping back: it keeps the target
  lane occupied for block_s at v after the car, alongside, comes within window_start_m of the junction (s_junction
  along the lane), or for block_s from the start if that is already past."""
  return max(s_junction - window_start_m - s_start, 0.0) + v * block_s


def catch_up_lead(v_ego: float, v_car: float, accel: float) -> float:
  """Meters a car released from rest beside an ego holding v_ego falls behind by the time it reaches v_car (speeding
  up at accel), so a pacer placed this far ahead ends up alongside."""
  t_ramp = v_car / accel
  return max(v_ego * t_ramp - v_car ** 2 / (2.0 * accel), 0.0)
