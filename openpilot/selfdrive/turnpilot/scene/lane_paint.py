"""Lane paint: whether each of modelV2's four lane lines is painted broken or solid, white or yellow, read off the road camera.

modelV2 says where the lines are, not what they are, and the lane planner needs to know what is beside the car before it
moves over toward a turn (lane_planner.py). US markings (MUTCD) carry the answer:
- broken white: the lane beyond goes the same way, and crossing is allowed
- solid white: a road edge, shoulder, bike or parking lane, or a lane change discouraged near a junction
- yellow, solid or broken: oncoming traffic beyond it, or the left edge of a divided road
- California's Botts' dots (raised dots in place of painted dashes) read as broken

Each line is sampled every X_STEP_M from X_MIN_M to X_MAX_M ahead where modelV2 puts it: its own x, y and z in the
calibrated frame (z is the ground below the camera), projected through the road camera as the UI draws laneLines. At each
sample a short profile is read across the line, on the ground. Paint is a narrow ridge of luma above the road on both
sides of it, by a margin that scales with the road's brightness; a step (a shadow edge) or a wide bright band (a crosswalk
bar, sun on concrete) is neither paint nor road and doesn't count. Yellow on concrete can be darker than the concrete, so a
narrow dip in U (the NV12 UV plane) below the road on both sides is paint too, and yellow. Luma paint with U near the
road's is white.

Per frame a line has paint at some share of its samples. Over WINDOW_S that share and the gaps between painted runs decide
the kind: a solid line covers nearly every sample, a US broken line (3 m dashes on a 12 m cycle) a quarter to a half,
with closed gaps between dashes in most frames. A line is unknown, never a guess, when its samples fall outside the frame
or under a detected vehicle, its probability is low, calibration is invalid, the frames are stale, or there aren't
enough samples. A line jumping sideways (the model renumbering them in a lane change) starts its window over.

On 135 hand-labeled frames of openpilot's demo route (lane_paint_video.py), no solid or yellow ego line read broken white;
59% of broken white ones did, and the rest mostly read unknown. The miss that matters: on concrete freeways a light slab
joint runs along some dashed lines and fills their gaps, so they read solid, which only costs a lane change.

Only the road (narrow) camera, a pinhole, is read; 8 m is the nearest ground clear of a comma four's view of the hood.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from openpilot.selfdrive.turnpilot.scene.ranging import Lens, calib_from_view

X_MIN_M, X_MAX_M, X_STEP_M = 8., 25., 0.25
XS = np.arange(X_MIN_M, X_MAX_M + 1e-6, X_STEP_M)
PROFILE_STEP_M = 0.05
PROFILE_HALF_M = 1.1                  # the profile across a line runs this far either side of it
OFFSETS = np.arange(-PROFILE_HALF_M, PROFILE_HALF_M + 1e-6, PROFILE_STEP_M)
SEARCH_HALF_M = 0.4                   # paint is looked for this far either side of where the model puts the line
FLANK_M = (0.3, 0.7)                  # the road level is the median luma this far either side of the brightest point
WIDTH_HALF_M = 0.25                   # a ridge above half its height over more than WIDTH_MAX_M of the +-WIDTH_HALF_M around
WIDTH_MAX_M = 0.4                     # its peak isn't a line: a crosswalk bar, a patch of sun
LUMA_MIN = 12.                        # paint is at least this much brighter than the road on both sides,
LUMA_REL = 0.1                        # and this share of the road's own luma
LUMA_SATURATED = 235.                 # a road this bright can't show paint
# Yellow on concrete is often no brighter than the concrete (the demo route's freeway: luma 100-110 on 120-135), so a dip in
# U below the road on both sides is paint too. U and V at the paint minus the road's:
YELLOW_DU = -8.                       # yellow at or below this (the freeway's yellow edge: -12 to -19, its V +9 to +12),
YELLOW_DV_MIN = -3.                   # and V not below this (a green bike lane is low U and low V)
WHITE_DU = -4.                        # luma paint at or above this, and not yellow, is white
YELLOW_SHARE = 0.3                    # yellow takes this share of the painted samples (yellow is the cautious answer),
WHITE_SHARE, WHITE_MAX_YELLOW = 0.7, 0.1  # white this share, with no more than this share yellow
LINE_PROB_MIN = 0.5
WINDOW_S = 1.5                        # paint is accumulated over this long
STALE_S = 0.5                         # no frame for this long starts every window over
JUMP_M = 1.0                          # a line moving this much sideways between frames is a different line
MIN_FRAMES = 4                        # in the window, before anything but unknown
MIN_VALID = 60                        # valid samples in the window
FULL_FRAMES = 8                       # confidence is full from this many frames on
SOLID_COV = 0.8                       # paint at this share of the samples or more is solid
BROKEN_COV = (0.04, 0.6)              # broken in between (Botts' dots cover about 5%),
GAP_MIN_M, GAP_MAX_M = 1., 12.        # with a gap this long between painted runs
GAP_SHARE, GAP_SHARE_FULL = 0.4, 0.6  # in this share of the frames (a 3 m dash on a 12 m cycle shows a closed gap in at most 2/3),
DASH_MAX_M = 6.                       # and the longest painted run per frame no longer than this (median over the window)
NONE_COV = 0.02                       # paint at no more than this share is no line

KINDS = ('unknown', 'none', 'broken', 'solid')
COLORS = ('unknown', 'white', 'yellow')

_C = len(OFFSETS) // 2
_SEARCH = np.arange(_C - round(SEARCH_HALF_M / PROFILE_STEP_M), _C + round(SEARCH_HALF_M / PROFILE_STEP_M) + 1)
_FLANK = np.arange(round(FLANK_M[0] / PROFILE_STEP_M), round(FLANK_M[1] / PROFILE_STEP_M) + 1)
_WIDTH = np.arange(-round(WIDTH_HALF_M / PROFILE_STEP_M), round(WIDTH_HALF_M / PROFILE_STEP_M) + 1)
PAINT, ROAD, INVALID = 1, 0, -1
NO_COLOR, WHITE, YELLOW = 0, 1, 2


@dataclass(frozen=True)
class LinePaint:
  kind: str = 'unknown'     # KINDS
  color: str = 'unknown'    # COLORS
  confidence: float = 0.
  coverage: float = 0.      # share of the window's valid samples with paint

  def is_(self, kind: str, color: str, min_confidence: float) -> bool:
    return self.kind == kind and self.color == color and self.confidence >= min_confidence


UNKNOWN = LinePaint()


@dataclass
class _Frame:
  t: float
  valid: int
  paint: int
  gap: bool              # a closed gap of GAP_MIN_M to GAP_MAX_M between painted runs
  long_run_m: float      # the longest painted run
  white: int             # painted samples read white,
  yellow: int            # and yellow


@dataclass
class _Line:
  frames: deque = field(default_factory=deque)
  y: float | None = None


def project(lens: Lens, view_from_calib: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """(u, v, in front) of calibrated-frame points (..., 3), x forward, y right, z down, through a pinhole lens."""
  ray = pts @ view_from_calib.T
  z = ray[..., 2]
  front = z > 0.5
  zs = np.where(front, z, 1.)
  cx, cy = lens.center
  return cx + lens.f * ray[..., 0] / zs, cy + lens.f * ray[..., 1] / zs, front


def _median_flanks(profile: np.ndarray, at: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Median of the profile FLANK_M either side of index `at`, per sample: (left, right)."""
  return (np.median(np.take_along_axis(profile, at[..., None] - _FLANK, axis=2), axis=2),
          np.median(np.take_along_axis(profile, at[..., None] + _FLANK, axis=2), axis=2))


def _at(profile: np.ndarray, idx: np.ndarray) -> np.ndarray:
  return np.take_along_axis(profile, idx[..., None], axis=2)[..., 0]


def sample_lines(nv12: np.ndarray, stride: int, y_height: int, lens: Lens, rpy_calib, lines_xyz, boxes=None):
  """Per line and sample (len(XS)): its state (PAINT, ROAD, INVALID) and color vote (NO_COLOR, WHITE, YELLOW).
  nv12: camerad's padded NV12 frame, flat uint8. lines_xyz: per line (x, y, z) arrays as in modelV2.laneLines.
  boxes: (N, >=4) x0 y0 x1 y1 of detected objects in this frame; samples behind them are INVALID."""
  centers = np.empty((len(lines_xyz), len(XS), 3))
  for i, (x, y, z) in enumerate(lines_xyz):
    centers[i, :, 0] = XS
    centers[i, :, 1] = np.interp(XS, x, y)
    centers[i, :, 2] = np.interp(XS, x, z)
  pts = np.repeat(centers[:, :, None, :], len(OFFSETS), axis=2)
  pts[..., 1] += OFFSETS
  u, v, front = project(lens, calib_from_view(rpy_calib).T, pts)
  ui, vi = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
  inside = front & (ui >= 0) & (ui < lens.width) & (vi >= 0) & (vi < lens.height)
  ok = inside.all(axis=2)
  ui, vi = np.where(inside, ui, 0), np.where(inside, vi, 0)
  luma = nv12[vi * stride + ui].astype(np.float32)
  uv = (y_height + vi // 2) * stride + (ui // 2) * 2
  chroma_u, chroma_v = nv12[uv].astype(np.float32), nv12[uv + 1].astype(np.float32)

  # luma: a narrow ridge above the road on both sides
  peak_idx = _SEARCH[np.argmax(luma[..., _SEARCH], axis=2)]
  road = np.maximum(*_median_flanks(luma, peak_idx))
  ridge = _at(luma, peak_idx) - road
  above = np.take_along_axis(luma, peak_idx[..., None] + _WIDTH, axis=2) >= (road + ridge / 2)[..., None]
  bright = ridge >= np.maximum(LUMA_MIN, LUMA_REL * road)
  wide = bright & (above.sum(axis=2) * PROFILE_STEP_M > WIDTH_MAX_M)
  # chroma: a dip in U below the road on both sides, V not dipping with it
  dip_idx = _SEARCH[np.argmin(chroma_u[..., _SEARCH], axis=2)]
  dip = _at(chroma_u, dip_idx) - np.minimum(*_median_flanks(chroma_u, dip_idx))
  dv = _at(chroma_v, dip_idx) - 0.5 * np.add(*_median_flanks(chroma_v, dip_idx))
  yellow = (dip <= YELLOW_DU) & (dv >= YELLOW_DV_MIN)
  u_road = 0.5 * np.add(*_median_flanks(chroma_u, peak_idx))
  white = bright & ~wide & ~yellow & (_at(chroma_u, peak_idx) - u_road >= WHITE_DU)

  ok &= road < LUMA_SATURATED
  if boxes is not None and len(boxes):
    b = np.asarray(boxes, dtype=np.float32)[:, :4]
    row = vi[..., _C][..., None]
    u0, u1 = ui[..., 0][..., None], ui[..., -1][..., None]
    behind = (row >= b[:, 1]) & (row <= b[:, 3]) & (np.maximum(u0, u1) >= b[:, 0]) & (np.minimum(u0, u1) <= b[:, 2])
    ok &= ~behind.any(axis=-1)

  painted = (bright & ~wide) | yellow
  state = np.where(~ok | (wide & ~yellow), INVALID, np.where(painted, PAINT, ROAD))
  color = np.where(state != PAINT, NO_COLOR, np.where(yellow, YELLOW, np.where(white, WHITE, NO_COLOR)))
  return state, color


def runs(state: np.ndarray) -> tuple[float, bool]:
  """(longest painted run in m, whether a closed gap of GAP_MIN_M..GAP_MAX_M lies between painted runs) for one line's
  samples. An INVALID sample ends a run, and a gap with one in it isn't closed."""
  starts = np.r_[0, np.flatnonzero(np.diff(state)) + 1]
  lengths = np.diff(np.r_[starts, len(state)]) * X_STEP_M
  vals = state[starts]
  painted = lengths[vals == PAINT]
  closed = (vals[1:-1] == ROAD) & (vals[:-2] == PAINT) & (vals[2:] == PAINT)
  gap = bool((closed & (lengths[1:-1] >= GAP_MIN_M) & (lengths[1:-1] <= GAP_MAX_M)).any())
  return float(painted.max()) if len(painted) else 0., gap


def classify(frames) -> LinePaint:
  """One line's paint from its window of frames."""
  if len(frames) < MIN_FRAMES:
    return UNKNOWN
  valid = sum(f.valid for f in frames)
  if valid < MIN_VALID:
    return UNKNOWN
  coverage = sum(f.paint for f in frames) / valid
  evidence = min(1., len(frames) / FULL_FRAMES)
  seen = [f for f in frames if f.valid >= len(XS) // 2]
  if coverage >= SOLID_COV:
    kind, margin = 'solid', min(1., (coverage - SOLID_COV) / (1. - SOLID_COV) + 0.5)
  elif BROKEN_COV[0] <= coverage <= BROKEN_COV[1] and seen and \
       (gap_share := np.mean([f.gap for f in seen])) >= GAP_SHARE and np.median([f.long_run_m for f in seen]) <= DASH_MAX_M:
    kind, margin = 'broken', min(1., gap_share / GAP_SHARE_FULL)
  elif coverage <= NONE_COV:
    return LinePaint('none', 'unknown', evidence, coverage)
  else:
    return LinePaint('unknown', 'unknown', 0., coverage)
  painted = sum(f.paint for f in frames)
  yellow, white = sum(f.yellow for f in frames) / max(painted, 1), sum(f.white for f in frames) / max(painted, 1)
  if yellow >= YELLOW_SHARE:
    color, share = 'yellow', min(1., yellow / (2 * YELLOW_SHARE))
  elif white >= WHITE_SHARE and yellow <= WHITE_MAX_YELLOW:
    color, share = 'white', white
  else:
    color, share = 'unknown', 1.
  return LinePaint(kind, color, float(evidence * margin * share), coverage)


class LanePaintClassifier:
  """Accumulates sample_lines() over time into one LinePaint per modelV2 lane line."""
  def __init__(self, lens: Lens):
    assert not lens.fisheye, "the road camera is a pinhole"
    self.lens = lens
    self.lines = [_Line() for _ in range(4)]
    self.last_t: float | None = None

  def reset(self) -> None:
    self.lines = [_Line() for _ in range(4)]
    self.last_t = None

  def update(self, t: float, nv12: np.ndarray, stride: int, y_height: int, rpy_calib, calibrated: bool, lines_xyz, line_probs,
             boxes=None) -> list[LinePaint]:
    """t: s, the frame's time. lines_xyz: modelV2.laneLines as (x, y, z) per line, of this very frame."""
    if self.last_t is not None and not (0. <= t - self.last_t <= STALE_S):
      self.reset()
    self.last_t = t
    if not calibrated or len(lines_xyz) != 4 or len(line_probs) != 4:
      self.reset()
      return [UNKNOWN] * 4
    state, color = sample_lines(nv12, stride, y_height, self.lens, rpy_calib, lines_xyz, boxes)
    out = []
    for i, line in enumerate(self.lines):
      y = float(np.interp(X_MIN_M, lines_xyz[i][0], lines_xyz[i][1]))
      if line.y is not None and abs(y - line.y) > JUMP_M:
        line.frames.clear()
      line.y = y
      while line.frames and t - line.frames[0].t > WINDOW_S:
        line.frames.popleft()
      if line_probs[i] >= LINE_PROB_MIN:
        s = state[i]
        long_run, gap = runs(s)
        line.frames.append(_Frame(t, int((s != INVALID).sum()), int((s == PAINT).sum()), gap, long_run,
                                  int((color[i] == WHITE).sum()), int((color[i] == YELLOW).sum())))
      out.append(classify(line.frames) if line_probs[i] >= LINE_PROB_MIN else UNKNOWN)
    return out


def fill_lane_paint(builder_list, paint: list[LinePaint]) -> None:
  """Fill a List(TurnpilotDetections.LanePaint) builder, initialized to len(paint)."""
  for b, p in zip(builder_list, paint, strict=True):
    b.kind, b.color, b.confidence, b.coverage = p.kind, p.color, float(p.confidence), float(p.coverage)


def lane_paint_from_msg(lane_paint) -> tuple[LinePaint, ...]:
  """The four lines' paint from a turnpilotDetections.lanePaint reader; unknown for a log without it."""
  if len(lane_paint) != 4:
    return (UNKNOWN,) * 4
  return tuple(LinePaint(str(p.kind), str(p.color), float(p.confidence), float(p.coverage)) for p in lane_paint)
