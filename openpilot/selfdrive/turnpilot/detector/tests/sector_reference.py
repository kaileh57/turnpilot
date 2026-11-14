"""sideview.SectorQuality as plainly written (31581b3dd: each sector gathered, converted and judged on its own, numpy's
median and percentile), kept as the reference the faster one must match sector for sector (test_light_cost.py). A change
to what it judges changes both, the same way."""
from collections import deque

import numpy as np

from openpilot.selfdrive.turnpilot.detector.sideview import (FROZEN_MAD_DN, FROZEN_S, GAP_S, HISTORY_S, MIN_STATIC_SAMPLES, RATIO_USABLE,
                                                            SAT_DN, STATIC_FRAC, STATIC_MEAN_DN, STATIC_STD_DN, STUCK_STRIDE,
                                                            WASHED_LIFT_DN, WASHED_RATIO, WASHED_SAT, SideGeometry, dark_threshold,
                                                            full_range, gradient)


class ReferenceSectorQuality:
  """Each side-view sector's quality, from the wide frame's host copy on that view's runs. Frozen and blocked look back over
  the sector's own recent samples (the view's runs, 5 Hz while the sweep runs), so a gap in them starts those over."""
  def __init__(self, dark_dn: float | None = None):
    self.dark_dn = dark_threshold() if dark_dn is None else dark_dn
    self.history: dict[tuple, deque] = {}    # (view, inner) -> (t, luma) samples within HISTORY_S
    self.changed_t: dict[tuple, float] = {}  # (view, inner) -> when it last changed by FROZEN_MAD_DN or more
    self.feed: tuple | None = None           # (t, the luma plane's STUCK_STRIDE sample) of the last run, any view
    self.feed_changed_t = 0.                 # when that sample last differed from the run before

  def update(self, geom: SideGeometry, name: str, frame: np.ndarray, t: float) -> list[tuple[float, float, str]]:
    """frame: the wide camera's NV12 host copy (flat uint8); t: its camera time, s. Returns (inner, outer, quality) per
    sector of the view."""
    stride = geom.stride
    stuck = self._stuck(frame[:stride * geom.y_height], t)
    ref = full_range(frame[geom.ref_idx])
    ref_g = float(gradient(frame, geom.ref_idx, stride).mean()) if len(ref) else 0.
    ref_p5 = float(np.percentile(ref, 5)) if len(ref) else 0.
    out = []
    for s in geom.maps[name].sectors:
      key = (name, s.inner)
      vals = full_range(frame[s.idx])
      ratio = float(gradient(frame, s.idx, stride).mean()) / ref_g if ref_g > 1e-6 else 0.
      hist = self.history.setdefault(key, deque())
      if hist and t - hist[-1][0] > GAP_S:
        hist.clear()
      if not hist or float(np.abs(vals - hist[-1][1]).mean()) >= FROZEN_MAD_DN:
        self.changed_t[key] = t
      hist.append((t, vals))
      while hist and t - hist[0][0] > HISTORY_S + 1e-6:
        hist.popleft()
      out.append((s.inner, s.outer, 'frozen' if stuck else self._quality(vals, ratio, ref_p5, hist, t - self.changed_t[key])))
    return out

  def _stuck(self, luma: np.ndarray, t: float) -> bool:
    """The feed stuck: the luma plane's STUCK_STRIDE sample unchanged, bit for bit, for FROZEN_S of runs no further apart than
    GAP_S. A copy is kept: the host buffer is camerad's, refilled under us."""
    sample = luma[::STUCK_STRIDE].copy()
    last = self.feed
    if last is None or t - last[0] > GAP_S or not np.array_equal(sample, last[1]):
      self.feed_changed_t = t
    self.feed = (t, sample)
    return t - self.feed_changed_t >= FROZEN_S - 1e-6

  def _quality(self, vals: np.ndarray, ratio: float, ref_p5: float, hist: deque, still_s: float) -> str:
    if float((vals >= SAT_DN).mean()) >= WASHED_SAT:
      return 'washed'
    # frozen before blocked (review 2026-11-07, LS-3): under vision a blocked sector after dark is judged by the walk's other
    # rules (static and dark is every side view at a night stand), so a stuck feed, static too, must still read frozen: the
    # whole feed's is (_stuck, every sector). In a live feed a still band clipped black (an unlit night stand: under the dark
    # threshold, so no noise lifts it) is the dark it looks like, never frozen (the review of 2026-11-08: under vision a dark
    # band is judged on its boxes, where a frozen one waited with no bound): blocked or dark below. A still band that isn't
    # dark stays frozen
    if still_s >= FROZEN_S and float(np.median(vals)) >= self.dark_dn:
      return 'frozen'
    if len(hist) >= MIN_STATIC_SAMPLES:
      stack = np.stack([h[1] for h in hist])
      static = (stack.std(axis=0) < STATIC_STD_DN) & (stack.mean(axis=0) < STATIC_MEAN_DN)
      if float(static.mean()) >= STATIC_FRAC:
        return 'blocked'
    if float(np.median(vals)) < self.dark_dn:
      return 'dark'
    if float(np.percentile(vals, 5)) - ref_p5 >= WASHED_LIFT_DN and ratio < WASHED_RATIO:
      return 'washed'
    if ratio < RATIO_USABLE:
      return 'degraded'
    return 'ok'
