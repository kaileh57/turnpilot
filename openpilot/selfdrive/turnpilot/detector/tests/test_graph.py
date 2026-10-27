import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.turnpilot.detector.graph import PAD_VALUE, Letterbox, letterbox, make_prepare
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


def solid_nv12(w: int, h: int, y: int, u: int, v: int) -> np.ndarray:
  """A camerad-layout NV12 buffer (as modeld copies it) of one color; the stride padding is garbage."""
  stride, y_height, uv_height, _ = get_nv12_info(w, h)
  buf = np.full((y_height + uv_height, stride), 255, dtype=np.uint8)
  buf[:h, :w] = y
  uv = buf[y_height:y_height + h // 2, :w].reshape(h // 2, w // 2, 2)
  uv[..., 0], uv[..., 1] = u, v
  return buf.reshape(-1)


class TestLetterbox(OpenpilotTestCase):
  def test_comma_cameras(self):
    # as Ultralytics' LetterBox(auto=False) pads them
    lb = letterbox(1344, 760)
    assert (lb.in_w, lb.in_h, lb.content, lb.pad) == (640, 384, (640, 362), (0, 11))
    lb = letterbox(1928, 1208)
    assert (lb.in_w, lb.in_h, lb.content, lb.pad) == (640, 416, (640, 401), (0, 7))

  def test_round_trip(self):
    rng = np.random.default_rng(0)
    for lb in (letterbox(1344, 760), letterbox(1928, 1208), Letterbox(1000, 1000, 640, 384)):
      x, y = rng.uniform(0, lb.cam_w, 100), rng.uniform(0, lb.cam_h, 100)
      np.testing.assert_allclose(lb.to_frame(*lb.to_input(x, y)), (x, y), atol=1e-9)
      # the frame's corners land on the content's corners
      (cw, ch), (px, py) = lb.content, lb.pad
      np.testing.assert_allclose(lb.to_input(0, 0), (px, py))
      np.testing.assert_allclose(lb.to_input(lb.cam_w, lb.cam_h), (px + cw, py + ch), atol=1.0)

  def test_portrait_frame_pads_sides(self):
    lb = Letterbox(760, 1344, 640, 384)
    assert lb.content[1] == 384 and lb.pad[1] == 0 and lb.pad[0] > 0


class TestPrepare(OpenpilotTestCase):
  def test_solid_red_and_padding(self):
    lb = Letterbox(256, 144, 64, 48)  # content 64x36, 6 rows of padding top and bottom
    from tinygrad import Tensor
    img = make_prepare(lb)(Tensor(solid_nv12(256, 144, 82, 90, 240))).numpy()
    assert img.shape == (1, 3, 48, 64)
    (_, ch), (_, py) = lb.content, lb.pad
    inside = img[0, :, py:py + ch].reshape(3, -1)
    np.testing.assert_allclose(inside.mean(axis=1), [1.0, 0.0, 0.0], atol=0.02)  # BT.601 limited (82, 90, 240) is red
    assert np.ptp(inside, axis=1).max() < 1e-5
    np.testing.assert_allclose(img[0, :, :py], PAD_VALUE / 255, atol=1e-6)
    np.testing.assert_allclose(img[0, :, py + ch:], PAD_VALUE / 255, atol=1e-6)

  def test_matches_numpy(self):
    # luma box-filtered 2x2, then nearest on the chroma grid, against a direct numpy version on a random frame
    w, h = 256, 144
    lb = Letterbox(w, h, 96, 64)
    stride, y_height, uv_height, _ = get_nv12_info(w, h)
    frame = np.random.default_rng(1).integers(0, 256, stride * (y_height + uv_height), dtype=np.uint8)
    from tinygrad import Tensor
    got = make_prepare(lb)(Tensor(frame)).numpy()[0]

    planes = frame.reshape(-1, stride).astype(np.float32)
    luma = planes[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    uv = planes[y_height:y_height + h // 2, :w].reshape(h // 2, w // 2, 2) - 128
    (cw, ch), (px, py) = lb.content, lb.pad
    expected = np.full((3, lb.in_h, lb.in_w), PAD_VALUE / 255, dtype=np.float32)
    for iy in range(py, py + ch):
      for ix in range(px, px + cw):
        sx = min(int((ix + 0.5 - px) / (2 * lb.scale)), w // 2 - 1)
        sy = min(int((iy + 0.5 - py) / (2 * lb.scale)), h // 2 - 1)
        c, (u, v) = (luma[sy, sx] - 16) * 1.164, uv[sy, sx]
        expected[:, iy, ix] = np.clip(np.array([c + 1.596 * v, c - 0.392 * u - 0.813 * v, c + 2.017 * u]) / 255, 0, 1)
    np.testing.assert_allclose(got, expected, atol=1e-4)
