"""The light reader as plainly written (light_state at 31581b3dd: every per-pixel quantity over each whole box,
each helper called afresh), kept as the reference the faster read in light_state.py must match field for field
(test_light_cost.py). A change to what the reader reads changes both, the same way."""
import math

import numpy as np

from openpilot.selfdrive.turnpilot.detector.decoder import LIGHT_READ_CONF, MIN_CONF
from openpilot.selfdrive.turnpilot.detector.light_state import (ARROW_MIN_PX, ASSUMED_HEAD_M, BLOWN_V, COLOR_SHARE, CORE_REL, GAP_REL,
                                                               GOVERN_BEARING_DEG, GREEN_LOW_MAX, HEAD_M, HEAD_MIN_H_PX, HUES, INNER_X,
                                                               LIT_FRAC_MIN, LIT_S, LIT_V, LOW_HEAD_M, MAX_ASPECT, MAX_RANGE, MAX_READ_HEADS,
                                                               MIN_ASPECT, MIN_COLOR_PX, MIN_ELEVATION_DEG, ROUND_MAX, ROW_REL, RUN_MAX_REL,
                                                               SAMPLE_S, SECOND_LIT_REL, STREAK_MAX, SYMMETRY_MIN, YELLOW_SPAN, HeadRead,
                                                               LightHead, LightReader)
from openpilot.selfdrive.turnpilot.scene.ranging import Lens, calib_from_view


def lamp_pixels(nv12: np.ndarray, stride: int, y_height: int, x0: int, y0: int, x1: int, y1: int):
  """(lamp mask, saturated hue or nan, HSV value 0-255) per pixel of the box, from NV12 (BT.601 full range)."""
  rows = np.arange(y0, y1)
  cols = np.arange(x0, x1)
  y = nv12.reshape(-1, stride)[rows[:, None], cols[None, :]].astype(np.float32)
  uv = nv12.reshape(-1, stride)[y_height + rows[:, None] // 2, (cols[None, :] // 2) * 2 + np.array([[0], [1]])[:, :, None]]
  u, v = uv[0].astype(np.float32) - 128., uv[1].astype(np.float32) - 128.
  r, g, b = y + 1.402 * v, y - 0.344136 * u - 0.714136 * v, y + 1.772 * u
  rgb = np.clip(np.stack([r, g, b], axis=-1), 0., 255.)
  mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
  s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.)
  hue = np.full(mx.shape, np.nan)
  d = np.maximum(mx - mn, 1e-6)
  rr, gg, bb = rgb[..., 0], rgb[..., 1], rgb[..., 2]
  h = np.where(mx == rr, ((gg - bb) / d) % 6., np.where(mx == gg, (bb - rr) / d + 2., (rr - gg) / d + 4.)) * 60.
  colored = (s >= LIT_S) & (mx >= LIT_V)
  hue[colored] = h[colored]
  in_lamp_hue = np.zeros(mx.shape, bool)
  for ranges in HUES.values():
    for lo, hi in ranges:
      in_lamp_hue |= (h >= lo) & (h < hi)
  lamp = (colored & in_lamp_hue) | (mx >= BLOWN_V)
  return lamp, hue, mx


def lamp_runs(lamp: np.ndarray, value: np.ndarray) -> list[tuple[int, int, int, float, int]]:
  """The box's lit lamps, top to bottom (module docstring): (first row, last row, lamp pixels, centre row, core rows) of
  each run of lit rows, its centre the mean row of its core and its core rows the core's first to last row. lamp, value:
  the box's inner columns (lamp_pixels). A lamp's glow joins its run, so a lamp is one run wherever the thirds' edges fall;
  two lit lamps are two runs, the dark housing between them."""
  prof = lamp.sum(axis=1)
  lit = np.nonzero(prof >= max(1., ROW_REL * float(prof.max())))[0]
  if not len(lit):
    return []
  gap = max(1, int(round(lamp.shape[0] * GAP_REL)))
  bounds, start = [], int(lit[0])
  for a, b in zip(lit[:-1], lit[1:], strict=True):
    if b - a - 1 > gap:
      bounds.append((start, int(a)))
      start = int(b)
  bounds.append((start, int(lit[-1])))
  out = []
  for r0, r1 in bounds:
    v = np.where(lamp[r0:r1 + 1], value[r0:r1 + 1], 0.)
    # its brightest: the 90th percentile of its lamp pixels, not one pixel (a blown glint on a visor by day would be the core)
    core = (v >= CORE_REL * float(np.percentile(v[v > 0], 90))).sum(axis=1)
    centre = float((np.arange(r0, r1 + 1) + 0.5) @ core / max(int(core.sum()), 1))
    cr = np.nonzero(core)[0]
    out.append((r0, r1, int(prof[r0:r1 + 1].sum()), centre, int(cr[-1] - cr[0] + 1) if len(cr) else r1 - r0 + 1))
  return out


def hue_counts(hues: np.ndarray) -> dict[str, int]:
  """The saturated pixels (finite hues) in each lamp colour's hue range."""
  hues = hues[np.isfinite(hues)]
  return {name: sum(int(((hues >= lo) & (hues < hi)).sum()) for lo, hi in ranges) for name, ranges in HUES.items()}


def hue_color(hues: np.ndarray) -> str | None:
  n = int(np.isfinite(hues).sum())
  if n < MIN_COLOR_PX:
    return None
  for name, k in hue_counts(hues).items():
    if k >= COLOR_SHARE * n:
      return name
  return None


def symmetric(mask: np.ndarray) -> bool:
  """The mask against its own mirror image left to right, about its centroid column."""
  ys, xs = np.nonzero(mask)
  cx = xs.mean()
  mirrored = np.rint(2 * cx - xs).astype(int)
  ok = (mirrored >= 0) & (mirrored < mask.shape[1])
  mirror = np.zeros_like(mask)
  mirror[ys[ok], mirrored[ok]] = True
  inter = (mask & mirror).sum()
  union = (mask | mirror).sum()
  return union > 0 and inter / union >= SYMMETRY_MIN


def head_place(lens: Lens, rpy_calib, cam_height: float, box) -> tuple[float, float, float]:
  """(deg its bottom edge lies above the calibrated horizon, bearing deg left of the forward axis, m to the head from its
  centre's elevation and ASSUMED_HEAD_M) of a class-9 box: no pixel is read."""
  x0, y0, x1, y1 = (float(b) for b in box[:4])
  rot = calib_from_view(rpy_calib)
  def angles(u, v):
    ray = rot @ lens.ray(u, v)  # calib frame: x forward, y right, z down
    return math.degrees(math.atan2(-ray[2], math.hypot(ray[0], ray[1]))), math.degrees(math.atan2(-ray[1], ray[0]))
  bottom_elev, bearing = angles((x0 + x1) / 2, y1)
  centre_elev, _ = angles((x0 + x1) / 2, (y0 + y1) / 2)
  rise = max(ASSUMED_HEAD_M - cam_height, 0.5)
  distance = min(rise / math.tan(math.radians(centre_elev)), MAX_RANGE) if centre_elev > 0. else MAX_RANGE
  return bottom_elev, bearing, distance


def head_near(lens: Lens, cam_height: float, box, distance: float) -> float:
  """m to the head of a class-9 box by its height against a standard head's (HEAD_M), where that is nearer than distance
  (head_place's, from its elevation over ASSUMED_HEAD_M), and never nearer than a head at the lowest mounting (LOW_HEAD_M)
  at that elevation would be. A box drawn round a lamp's glow alone is shorter than its head and says a longer range: the
  elevation's stands."""
  rise = max(ASSUMED_HEAD_M - cam_height, 0.5)
  floor = distance * max(LOW_HEAD_M - cam_height, 0.5) / rise
  by_size = lens.f * HEAD_M / max(float(box[3]) - float(box[1]), 1.)
  return min(distance, max(by_size, floor))


def read_head(nv12: np.ndarray, stride: int, y_height: int, lens: Lens, rpy_calib, cam_height: float, box) -> HeadRead:
  """One class-9 box's read (module docstring). box: a decode() row, x0 y0 x1 y1 in full-frame pixels."""
  x0, y0, x1, y1 = (float(b) for b in box[:4])
  bottom_elev, bearing, distance = head_place(lens, rpy_calib, cam_height, box)
  out = HeadRead('unknown', 'read', bearing=bearing, distance=distance, near=head_near(lens, cam_height, box, distance))
  bw, bh = x1 - x0, y1 - y0
  if max(bh, bw) < HEAD_MIN_H_PX:
    out.reason, out.gated = 'small', True
    return out
  if bottom_elev < MIN_ELEVATION_DEG:
    out.reason, out.gated = 'belowHorizon', True
    return out
  if not (MIN_ASPECT <= bh / max(bw, 1e-6) <= MAX_ASPECT):
    out.reason = 'aspect'
    return out
  ix0, iy0 = int(math.floor(x0)), int(math.floor(y0))
  ix1, iy1 = min(int(math.ceil(x1)), lens.width), min(int(math.ceil(y1)), lens.height)
  lamp, hue, value = lamp_pixels(nv12, stride, y_height, ix0, iy0, ix1, iy1)
  h, w = lamp.shape
  edges = np.rint(np.linspace(0, h, 4)).astype(int)
  inner = slice(int(round(w * INNER_X)), max(int(round(w * (1 - INNER_X))), int(round(w * INNER_X)) + 1))
  fracs = np.array([lamp[edges[k]:edges[k + 1], inner].mean() if edges[k + 1] > edges[k] else 0. for k in range(3)])
  out.detail = f"h{h} fr{fracs[0]:.2f}/{fracs[1]:.2f}/{fracs[2]:.2f}"
  if fracs.max() < LIT_FRAC_MIN:
    out.reason = 'dark'
    return out
  runs = lamp_runs(lamp[:, inner], value[:, inner])
  r0, r1, n, centre, core_rows = max(runs, key=lambda r: r[2])
  # the lamp's colour and shape are read on its run grown through the rows next to it that hold any lamp pixel, never into
  # another run: an arrow's thin head and its glow's thin edge are under ROW_REL, and are the lamp's
  prof = lamp[:, inner].sum(axis=1)
  e0, e1 = r0, r1
  lo, hi = max((r[1] + 1 for r in runs if r[1] < r0), default=0), min((r[0] - 1 for r in runs if r[0] > r1), default=h - 1)
  while e0 > lo and prof[e0 - 1] > 0:
    e0 -= 1
  while e1 < hi and prof[e1 + 1] > 0:
    e1 += 1
  rows = slice(e0, e1 + 1)
  peak = int(value[rows, inner][lamp[rows, inner]].max())
  widest = int(lamp[r0:r1 + 1].sum(axis=1).max())  # across the whole box: a narrow box's inner columns clip a glow
  out.detail += f" run{r0}-{r1} lamp{e0}-{e1} c{centre / h:.2f} cr{core_rows} n{n} wd{widest} runs{len(runs)} pk{peak}"
  third_px = h / 3 * lamp[:, inner].shape[1]
  if any(r[2] >= max(LIT_FRAC_MIN * third_px, SECOND_LIT_REL * n) for r in runs if r[0] != r0):
    out.reason = 'twoLit'
    return out
  # the whole run against the box (a glow's tail is the lamp's, and never past RUN_MAX_REL of a head, unless it is a round
  # red or green glow: ROUND_MAX), its core against its widest row (a lamp's face is round however far its glow trails; a
  # lit pole is its core end to end): STREAK_MAX, above
  out.detail += " hue " + " ".join(f"{k[0]}{v}" for k, v in hue_counts(hue[rows, inner]).items())
  color = hue_color(hue[rows, inner])
  round_glow = color in ('red', 'green') and r1 - r0 + 1 <= ROUND_MAX * widest  # the box drawn round the glow alone
  if core_rows > STREAK_MAX * widest or (r1 - r0 + 1 > RUN_MAX_REL * h and not round_glow):
    out.reason = 'streak'
    return out
  if color is None:
    out.reason = 'noColor'
    return out
  if color == 'yellow' and not YELLOW_SPAN[0] <= centre / h <= YELLOW_SPAN[1]:
    out.reason = 'positionMismatch'
    return out
  mask = lamp[rows]
  if mask.sum() < ARROW_MIN_PX:
    out.reason = 'tooFewLampPx'
    return out
  if not symmetric(mask):
    out.reason = 'arrow'
    if color != 'red':
      return out  # a green arrow is never a permission; a yellow one may be a flashing yellow arrow's permissive left
  out.state = color  # a red arrow keeps its colour (module docstring)
  return out


class ReferenceReader(LightReader):
  """LightReader with its read loop as plainly written (31581b3dd): each box's place and pixels worked out in its own
  read_head, the reference's."""
  def read(self, t: float, nv12: np.ndarray, stride: int, y_height: int, rpy_calib, cam_height: float, boxes,
           path=None) -> list[LightHead]:
    """boxes: decode_lights() rows; path: (x, y left) of the lane ahead (tracker.predicted_path), None for straight."""
    reads, rejects, low_read, low_left = [], [], 0, []

    def lane_bearing(distance: float) -> float:
      return math.degrees(math.atan2(float(np.interp(distance, path[0], path[1])), distance)) if path is not None else 0.
    self.frames += 1
    for box in boxes:
      norm = (float(box[0]) / self.lens.width, float(box[1]) / self.lens.height, float(box[2]) / self.lens.width,
              float(box[3]) / self.lens.height)
      low = float(box[4]) < LIGHT_READ_CONF
      if low:
        # under LIGHT_READ_CONF (down to decoder.LIGHT_GREEN_CONF) a box counts only if it reads green: read only within
        # the governing bearing and at most GREEN_LOW_MAX a frame, the rest left unread (lowConf)
        _, bearing, distance = head_place(self.lens, rpy_calib, cam_height, box)
        if abs(bearing - lane_bearing(distance)) > GOVERN_BEARING_DEG:
          rejects.append(LightHead(-1, norm, float(box[4]), 'unknown', 0, 'unknown', 'lowConf', bearing, distance, False))
          continue
        if low_read >= GREEN_LOW_MAX:
          low_left.append((box, norm, bearing, distance))
          continue
        low_read += 1
      if len(reads) >= MAX_READ_HEADS:
        self.gate_counts['overCap'] += 1
        rejects.append(LightHead(-1, norm, float(box[4]), 'unknown', 0, 'unknown', 'overCap', 0., 0., False))
        continue
      r = read_head(nv12, stride, y_height, self.lens, rpy_calib, cam_height, box)
      if r.gated:
        self.gate_counts[r.reason] += 1
        rejects.append(LightHead(-1, norm, float(box[4]), 'unknown', 0, 'unknown', r.reason, r.bearing, r.distance, False))
        continue
      if low and r.state != 'green':
        # no head: what it read is kept (a red or yellow read under LIGHT_READ_CONF says lowConf: map_controls' veto)
        rejects.append(LightHead(-1, norm, float(box[4]), 'unknown', 0, r.state, r.reason if r.state == 'unknown' else 'lowConf',
                                 r.bearing, r.distance, False, r.detail, r.near))
        continue
      self.head_counts[r.reason] += 1
      if r.state != 'unknown':
        self.colour_counts[r.state] += 1
      reads.append((norm, float(box[4]), r))
    heads = self.voter.update(t, [(norm, r.state) for norm, _, r in reads])
    # the low-confidence boxes within the governing bearing past GREEN_LOW_MAX: read on a run a head reads or votes green
    # (at any confidence: the green map_controls would count), for a red or yellow among them (never a head: a third green
    # there is left unpublished too); else left unread
    green = any(r.state == 'green' for _, _, r in reads) or any(hd.state == 'green' for hd in heads)
    more = len(reads)
    for box, norm, bearing, distance in low_left:
      if not green or more >= MAX_READ_HEADS:
        rejects.append(LightHead(-1, norm, float(box[4]), 'unknown', 0, 'unknown', 'lowConf', bearing, distance, False))
        continue
      more += 1
      r = read_head(nv12, stride, y_height, self.lens, rpy_calib, cam_height, box)
      self.gate_counts[r.reason] += r.gated
      reason = r.reason if r.gated or r.state == 'unknown' else 'lowConf'
      rejects.append(LightHead(-1, norm, float(box[4]), 'unknown', 0, r.state, reason, r.bearing, r.distance, False, r.detail, r.near))
    out = []
    for hd, (norm, conf, r) in zip(heads, reads, strict=True):
      out.append(LightHead(hd.id, norm, conf, hd.state, hd.votes, r.state, r.reason, r.bearing, r.distance,
                           abs(r.bearing - lane_bearing(r.distance)) <= GOVERN_BEARING_DEG, r.detail, r.near))
    if out and (self.sample_t is None or not 0. <= t - self.sample_t < SAMPLE_S):
      self.sample_t = t
      self.sample(t, nv12, stride, y_height, out)
    # a box under the detector's MIN_CONF (decoder.LIGHT_READ_CONF, read for an earlier red) is a head only once it reads a
    # colour or holds a vote: the boxes that read no lamp stay unpublished, as before, so nothing downstream counts them as
    # heads; they and the gated boxes are kept for the log (rejects)
    published = [h for h in out if h.conf >= MIN_CONF or h.read != 'unknown' or h.state != 'unknown']
    self.rejects = rejects + [h for h in out if not (h.conf >= MIN_CONF or h.read != 'unknown' or h.state != 'unknown')]
    return published
