"""The light reader's faster read (light_state.py: a frame's boxes converted together, each box cropped before its colour,
the hue only where it is kept, each quantity once, an exact quicker percentile) against the reader as plainly written
(light_reference.py, 31581b3dd's): the same heads, states, votes, reasons and details on every frame, and less time.
On the drive's own frames the same is proved by light_replay.py (route 29, every road run: 0 of 3,406 differ)."""
import time

import numpy as np

from openpilot.selfdrive.turnpilot.detector import light_state
from openpilot.selfdrive.turnpilot.detector.exact_stats import median, percentile
from openpilot.selfdrive.turnpilot.detector.light_state import GREEN_LOW_MAX, MAX_READ_HEADS, LightReader, lamp_pixels, lamp_pixels_many, read_head
from openpilot.selfdrive.turnpilot.detector.tests import light_reference as ref
from openpilot.selfdrive.turnpilot.detector.tests.test_light_state import COLORS, CORE, H, LENS, LEVEL, NIGHT_SKY, STRIDE, W, \
  Y_HEIGHT, night_lamp, nv12

RNG_SEED = 7


def read_fields(r) -> tuple:
  return (r.state, r.reason, r.gated, r.bearing, r.distance, r.detail, r.near)


def head_fields(h) -> tuple:
  return (h.id, h.box, h.conf, h.state, h.votes, h.read, h.reason, h.bearing, h.distance, h.governs, h.detail, h.near)


def junction_frame(rng, n_heads: int, night: bool) -> tuple[np.ndarray, np.ndarray]:
  """A frame of n_heads signal heads over the horizon, lit at random (day: solid lamps, some arrows and two-lit; night: a
  glowing lamp on the sky), a few lit poles, signs and noise boxes among them, every box a decode_lights row with a
  confidence spread from LIGHT_GREEN_CONF to 0.9, in confidence order."""
  img = np.zeros((H, W, 3), np.uint8)
  img[:] = NIGHT_SKY if night else (105, 175, 225)
  img += rng.integers(0, 12, img.shape, dtype=np.uint8)
  boxes = []
  for _ in range(n_heads):
    h = int(rng.integers(20, 90))
    w = max(4, int(h / rng.uniform(1.2, 3.2)))
    x0, y0 = int(rng.integers(0, W - w)), int(rng.integers(0, 330 - h))
    color = str(rng.choice(['red', 'yellow', 'green']))
    lamp = int(rng.integers(0, 3))
    cx, cy = x0 + w / 2, y0 + h / 6 + lamp * h / 3
    if rng.random() < 0.1:
      img[y0:y0 + h, x0:x0 + w] = (25, 25, 25)  # a head with no lamp lit toward the car
    elif night:
      night_lamp(img, cx, cy, color, r=max(2., w * 0.3))
    else:
      img[y0:y0 + h, x0:x0 + w] = (25, 25, 25)
      yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w]
      disc = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= (0.38 * min(w, h / 3)) ** 2
      if rng.random() < 0.15:
        disc &= xx < cx + 1  # half a ball: an arrow's shape
      img[y0:y0 + h, x0:x0 + w][disc] = COLORS[color]
      if rng.random() < 0.1:
        img[y0 + 2 * h // 3:y0 + h, x0:x0 + w] = COLORS['white']  # a second, blown lamp
    # the box as the detector draws it: off by a pixel or two, fractional
    jit = rng.uniform(-2., 2., 4)
    boxes.append([x0 + jit[0], y0 + jit[1], x0 + w + jit[2], y0 + h + jit[3], rng.uniform(0.1, 0.9), 9])
  for _ in range(int(rng.integers(0, 4))):  # a lit pole, a sign, a tail light below the horizon
    x0, y0 = int(rng.integers(0, W - 30)), int(rng.integers(0, H - 120))
    img[y0:y0 + 100, x0:x0 + 8] = COLORS['yellow'] if rng.random() < 0.5 else CORE['red']
    boxes.append([x0 - 1.5, y0 - 0.5, x0 + 9.3, y0 + 101.2, rng.uniform(0.1, 0.9), 9])
  b = np.array(boxes, np.float32).reshape(-1, 6)
  b[:, :4] = np.clip(b[:, :4], 0, [W, H, W, H])
  b = b[(b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])]
  return nv12(img), b[np.argsort(-b[:, 4], kind='stable')][:16]


def test_the_percentile_and_median_are_numpys_bit_for_bit():
  # the light reader's lamp core (90th) and the sector quality's 5th percentile and median, on the float32 values they take
  rng = np.random.default_rng(RNG_SEED)
  for k in range(20000):
    n = int(rng.integers(1, 300)) if k % 10 else int(rng.integers(300, 3000))
    a = [rng.uniform(150, 255, n), rng.integers(150, 256, n).astype(float), np.clip(rng.normal(230, 30, n), 0, 255),
         (rng.integers(0, 256, n) - 16.) * (255. / 219.)][k % 4].astype(np.float32)
    for q in (90, 5):
      assert percentile(a, q) == float(np.percentile(a, q)), (n, k, q)
    assert median(a) == float(np.median(a)), (n, k)


def test_a_boxs_pixels_are_the_plain_reads_pixel_for_pixel():
  rng = np.random.default_rng(RNG_SEED)
  frame = nv12(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))  # every colour, saturated and not, blown and dark
  rects = []
  for _ in range(400):
    w, h = int(rng.integers(1, 60)), int(rng.integers(1, 120))
    x0, y0 = int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1))
    rects.append((x0, y0, x0 + w, y0 + h))  # odd and even edges, the frame's own
  rects += [(0, 0, 37, 91), (W - 31, H - 77, W, H), (W - 1, 0, W, 1)]
  together = lamp_pixels_many(frame, STRIDE, Y_HEIGHT, rects)
  for rect, many in zip(rects, together, strict=True):
    plain = ref.lamp_pixels(frame, STRIDE, Y_HEIGHT, *rect)
    for got in (lamp_pixels(frame, STRIDE, Y_HEIGHT, *rect), many):
      for a, b in zip(plain, got, strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True), rect


def test_every_heads_read_is_the_plain_reads():
  rng = np.random.default_rng(RNG_SEED)
  reasons = set()
  for k in range(60):
    frame, boxes = junction_frame(rng, int(rng.integers(1, 10)), night=k % 2 == 1)
    rpy = [0., float(rng.uniform(-0.03, 0.03)), float(rng.uniform(-0.03, 0.03))]
    for box in boxes:
      plain = ref.read_head(frame, STRIDE, Y_HEIGHT, LENS, rpy, 1.22, box)
      assert read_fields(read_head(frame, STRIDE, Y_HEIGHT, LENS, rpy, 1.22, box)) == read_fields(plain), box
      reasons.add(plain.reason)
  # the frames reach every path the read has: gates, colour and shape
  assert {'small', 'belowHorizon', 'aspect', 'dark', 'read', 'streak', 'twoLit', 'noColor', 'arrow', 'positionMismatch'} <= reasons, reasons


def test_the_readers_heads_votes_and_rejects_are_the_plain_readers_frame_after_frame():
  rng = np.random.default_rng(RNG_SEED)
  fast, plain = LightReader(LENS), ref.ReferenceReader(LENS)
  path = (np.linspace(0., 120., 33), np.linspace(0., 6., 33))  # a lane bending left
  paths = seen = 0
  for k in range(240):
    if k % 40 == 0:
      frame, boxes = junction_frame(rng, int(rng.integers(3, 16)), night=k % 80 == 0)
    # the same heads over a few frames (the voter's matches), jittered; some frames with more boxes than the reader reads
    b = boxes.copy()
    b[:, :4] += rng.uniform(-1., 1., (len(b), 4)).astype(np.float32)
    b[:, :4] = np.clip(b[:, :4], 0, [W, H, W, H])
    b[:, 4] = np.clip(b[:, 4] + rng.uniform(-0.05, 0.05, len(b)), 0.1, 0.95).astype(np.float32)
    b = b[np.argsort(-b[:, 4], kind='stable')]
    p = path if k % 3 else None
    out_fast = fast.read(k * 0.1, frame, STRIDE, Y_HEIGHT, LEVEL, 1.22, b, p)
    out_plain = plain.read(k * 0.1, frame, STRIDE, Y_HEIGHT, LEVEL, 1.22, b, p)
    assert [head_fields(h) for h in out_fast] == [head_fields(h) for h in out_plain], k
    assert [head_fields(h) for h in fast.rejects] == [head_fields(h) for h in plain.rejects], k
    paths += p is not None
    seen += len(out_plain)
  assert (fast.gate_counts, fast.head_counts, fast.colour_counts) == (plain.gate_counts, plain.head_counts, plain.colour_counts)
  assert seen and paths
  # the frames take every branch of the loop: over the cap, low boxes read and left unread, a green's late reads
  assert fast.gate_counts['overCap'] > 0
  assert plain.head_counts and plain.colour_counts['green'] > 0


def test_a_low_boxs_late_read_on_a_green_run_is_the_plain_readers():
  # under LIGHT_READ_CONF past GREEN_LOW_MAX within the bearing: read only on a run that reads green (light_state's low_left)
  frame = np.zeros((H, W, 3), np.uint8)
  frame[:] = NIGHT_SKY
  boxes = []
  for i, (color, conf) in enumerate([('green', 0.8)] + [('red', 0.15)] * (GREEN_LOW_MAX + 2)):
    cx = 600 + 30 * i
    night_lamp(frame, cx, 200, color, r=5.)
    boxes.append([cx - 9, 186, cx + 9, 236, conf - 0.001 * i, 9])
  img, b = nv12(frame), np.array(boxes, np.float32)
  fast, plain = LightReader(LENS), ref.ReferenceReader(LENS)
  for t in (0., 0.1, 0.2):
    out_fast, out_plain = (r.read(t, img, STRIDE, Y_HEIGHT, LEVEL, 1.22, b) for r in (fast, plain))
    assert [head_fields(h) for h in out_fast] == [head_fields(h) for h in out_plain]
    assert [head_fields(h) for h in fast.rejects] == [head_fields(h) for h in plain.rejects]
  late = [h for h in plain.rejects if h.reason == 'lowConf' and h.read == 'red']
  assert len(late) == GREEN_LOW_MAX + 2, plain.rejects  # all of them read: the first two in the loop, the rest late


def test_a_junctions_read_costs_well_under_the_plain_one():
  # what the device's budget needs (light_replay.py: route 29's road runs, this tree's reader p90 about half the plain one's
  # on the car's scale): measured relative, the best of many reads each, so a loaded host slows both alike
  rng = np.random.default_rng(RNG_SEED)
  frame, boxes = junction_frame(rng, MAX_READ_HEADS, night=True)
  fast, plain = LightReader(LENS), ref.ReferenceReader(LENS)

  def best(reader) -> float:
    out = np.inf
    for k in range(25):
      t0 = time.perf_counter()
      reader.read(k * 0.1, frame, STRIDE, Y_HEIGHT, LEVEL, 1.22, boxes)
      out = min(out, time.perf_counter() - t0)
    return out
  assert best(fast) < 0.7 * best(plain)


def test_the_rotation_and_place_are_worked_out_once_a_frame():
  rng = np.random.default_rng(RNG_SEED)
  frame, boxes = junction_frame(rng, 10, night=False)
  calls = {'rot': 0}
  orig = light_state.calib_from_view

  def counted(*a, **k):
    calls['rot'] += 1
    return orig(*a, **k)
  light_state.calib_from_view = counted
  try:
    LightReader(LENS).read(0., frame, STRIDE, Y_HEIGHT, LEVEL, 1.22, boxes)
  finally:
    light_state.calib_from_view = orig
  assert calls['rot'] == 1
