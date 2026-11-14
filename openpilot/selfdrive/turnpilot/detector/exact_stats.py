"""np.percentile and np.median of a 1-D float array, the same value bit for bit at a fraction of their time: both take the
order statistics they need by one partition, as numpy does, without its argument handling (about 10 us a call on an M-series
Mac, 80 on the comma; the light reader and the side views' sector quality call them per lamp and per sector on every run).
tests/test_light_cost.py checks them against numpy's on arrays of every size the readers see."""
import math

import numpy as np


def percentile(a: np.ndarray, q: float) -> float:
  """float(np.percentile(a, q)), numpy's default (linear) method: the order statistics either side of (n - 1) * q / 100,
  lerped in the array's dtype as numpy's _lerp does (from the far one where the weight is half or more)."""
  n = a.size
  if n < 2:
    return float(np.percentile(a, q))
  at = (n - 1) * (q / 100)
  lo = math.floor(at)
  hi = min(lo + 1, n - 1)
  part = np.partition(a, (lo, hi) if hi != lo else lo)
  below, above = part[lo], part[hi]
  t = at - lo
  diff = above - below
  return float(above - diff * (1 - t)) if t >= 0.5 else float(below + diff * t)


def median(a: np.ndarray) -> float:
  """float(np.median(a)): the middle order statistic, or the mean of the middle two in the array's dtype."""
  n = a.size
  if n < 1:
    return float(np.median(a))
  h = n // 2
  if n % 2:
    return float(np.partition(a, h)[h])
  part = np.partition(a, (h - 1, h))
  return float((part[h - 1] + part[h]) / 2)
