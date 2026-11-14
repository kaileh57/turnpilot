"""sideview.SectorQuality's faster judge (its view's sectors gathered and converted as one array, the temporal statistics
once a view, exact_stats' median and percentile) against the one as plainly written (sector_reference.py, 31581b3dd's):
the same quality for every sector of every run, through a stand's live, dark, washed, blocked, still and stuck frames, a
gap between sweeps, and less time (the stand schedule's side runs: test_stand_schedule.py)."""
import time

import numpy as np

from openpilot.selfdrive.turnpilot.detector import sideview as sv
from openpilot.selfdrive.turnpilot.detector.sideview import SectorQuality
from openpilot.selfdrive.turnpilot.detector.tests.sector_reference import ReferenceSectorQuality
from openpilot.selfdrive.turnpilot.detector.tests.test_sideview import geometry, textured_frame


def frames(geom, rng, n: int):
  """A stand's wide frames: live texture, then spells of a darkened, a lifted and a saturated band, a sector clipped black,
  one held still while lit, and the whole feed stuck."""
  left, right = geom.maps['sideLeft'].sectors, geom.maps['sideRight'].sectors
  still = prev = None
  for k in range(n):
    f = textured_frame(geom, rng)
    phase = (k // 15) % 6
    if phase == 1:
      f[:] = np.clip(f.astype(int) - 110, 16, 235).astype(np.uint8)
    elif phase == 2:
      f[:] = np.clip(f.astype(int) + 90, 16, 255).astype(np.uint8)
    elif phase == 3:
      f[left[1].idx] = 235
      f[right[2].idx] = 16
    elif phase == 4:
      if still is None:
        still = f[left[3].idx].copy()
      f[left[3].idx] = still
    if phase == 5 and k % 15 >= 5:
      f = prev.copy()  # a stuck feed: the last frame again, bit for bit
    prev = f
    yield f


def test_every_sectors_quality_is_the_plain_judges():
  rng = np.random.default_rng(11)
  geom = geometry()
  fast, plain = SectorQuality(dark_dn=45.), ReferenceSectorQuality(dark_dn=45.)
  seen = set()
  t = 0.
  for k, f in enumerate(frames(geom, rng, 240)):
    t += 0.2 if k % 50 else sv.GAP_S + 0.5  # a new sweep now and then
    for name in ('sideLeft', 'sideRight'):
      want = plain.update(geom, name, f, t)
      assert fast.update(geom, name, f, t) == want, (k, name)
      seen |= {q for _, _, q in want}
  assert {'ok', 'dark', 'washed', 'blocked', 'frozen'} <= seen, seen


def test_a_views_sectors_cost_about_half_the_plain_judges():
  rng = np.random.default_rng(3)
  geom = geometry()
  fs = list(frames(geom, rng, 30))

  def best(q) -> float:
    out = np.inf
    for rep in range(6):
      t0 = time.perf_counter()
      for k, f in enumerate(fs):
        q.update(geom, ('sideLeft', 'sideRight')[k % 2], f, 100. * rep + 0.2 * k)
      out = min(out, time.perf_counter() - t0)
    return out
  assert best(SectorQuality(dark_dn=45.)) < 0.75 * best(ReferenceSectorQuality(dark_dn=45.))
