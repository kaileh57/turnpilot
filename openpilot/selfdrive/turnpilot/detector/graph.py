"""YOLO26 on a full camera frame as one tinygrad graph: modeld's NV12 buffer in, the (1, 300, 6) end-to-end output out.

The whole frame is letterboxed (no crop, no warp), so a box maps back to sensor pixels with one scale and one offset.
Luma is box-filtered 2x2 onto the chroma grid and both are sampled nearest-neighbor onto the model input: for a
1344x760 frame into 640x384 that is close to an area resize. YOLO26 takes RGB in 0-1.
"""
from dataclasses import dataclass
from pathlib import Path

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'
DEFAULT_MODEL = 'yolo26n'
# model input (w, h) per camera frame (w, h); multiples of 32, the long side 640
INPUT_SIZES = {(1344, 760): (640, 384), (1928, 1208): (640, 416)}
PAD_VALUE = 114  # Ultralytics' letterbox gray


@dataclass(frozen=True)
class Letterbox:
  cam_w: int
  cam_h: int
  in_w: int
  in_h: int

  @property
  def scale(self) -> float:
    return min(self.in_w / self.cam_w, self.in_h / self.cam_h)

  @property
  def content(self) -> tuple[int, int]:
    """(w, h) of the scaled frame inside the model input."""
    return round(self.cam_w * self.scale), round(self.cam_h * self.scale)

  @property
  def pad(self) -> tuple[int, int]:
    """(left, top) padding, rounded as Ultralytics' LetterBox rounds it."""
    w, h = self.content
    return round((self.in_w - w) / 2 - 0.1), round((self.in_h - h) / 2 - 0.1)

  def to_frame(self, x, y):
    """Model input pixels to full-frame pixels (continuous coordinates, pixel i spans [i, i+1])."""
    px, py = self.pad
    return (x - px) / self.scale, (y - py) / self.scale

  def to_input(self, x, y):
    px, py = self.pad
    return x * self.scale + px, y * self.scale + py


def letterbox(cam_w: int, cam_h: int, in_size: tuple[int, int] | None = None) -> Letterbox:
  in_w, in_h = in_size or INPUT_SIZES[(cam_w, cam_h)]
  return Letterbox(cam_w, cam_h, in_w, in_h)


def source_index(n: int, pad: int, content: int, scale: float, n_half: int):
  """Per model-input column (or row): the half-resolution frame pixel holding its center, and whether it's inside the frame."""
  from tinygrad import Tensor, dtypes
  i = Tensor.arange(n, dtype=dtypes.float32)
  src = ((i + 0.5 - pad) / (2 * scale)).floor().clip(0, n_half - 1).cast(dtypes.int32)
  return src, (i >= pad) & (i < pad + content)


def make_prepare(lb: Letterbox):
  """NV12 buffer (uint8, flat, camerad's padded layout) -> (1, 3, in_h, in_w) RGB 0-1, letterboxed."""
  from tinygrad import Tensor, dtypes
  stride, y_height, _, _ = get_nv12_info(lb.cam_w, lb.cam_h)
  uv_offset = stride * y_height
  (cw, ch), (px, py) = lb.content, lb.pad

  def prepare(input_frame: Tensor) -> Tensor:
    sx, vx = source_index(lb.in_w, px, cw, lb.scale, lb.cam_w // 2)
    sy, vy = source_index(lb.in_h, py, ch, lb.scale, lb.cam_h // 2)
    sx, vx, sy, vy = sx.reshape(1, -1), vx.reshape(1, -1), sy.reshape(-1, 1), vy.reshape(-1, 1)
    frame = input_frame.cast(dtypes.float32)
    y_idx = 2 * sy * stride + 2 * sx
    luma = (frame[y_idx] + frame[y_idx + 1] + frame[y_idx + stride] + frame[y_idx + stride + 1]) * 0.25
    uv_idx = uv_offset + sy * stride + 2 * sx
    u, v = frame[uv_idx] - 128, frame[uv_idx + 1] - 128
    # BT.601 limited range, the inverse of camerad's RGB to NV12
    c = (luma - 16) * 1.164
    rgb = Tensor.stack(c + 1.596 * v, c - 0.392 * u - 0.813 * v, c + 2.017 * u).div(255).clip(0, 1)
    return (vx & vy).where(rgb, PAD_VALUE / 255).unsqueeze(0)
  return prepare


def onnx_path(model: str = DEFAULT_MODEL) -> Path:
  return MODELS_DIR / f'{model}.onnx'


def detector_pkl_path(cam_w: int, cam_h: int, chestnut: bool, models_dir: Path = MODELS_DIR) -> Path:
  """The compiled detector for a camera size, named like modeld's pkls: big_ for the chestnut."""
  return models_dir / f'{"big_" if chestnut else ""}detector_{cam_w}x{cam_h}_tinygrad.pkl'


def make_detector(cam_w: int, cam_h: int, model: str = DEFAULT_MODEL, in_size: tuple[int, int] | None = None):
  """(letterbox, detect): detect(input_frame) -> (1, 300, 6) Tensor, x0 y0 x1 y1 in model input pixels, conf, COCO class."""
  from tinygrad import Context
  from tinygrad.nn.onnx import OnnxRunner
  lb = letterbox(cam_w, cam_h, in_size)
  prepare = make_prepare(lb)
  runner = OnnxRunner(onnx_path(model))
  input_name = next(iter(runner.graph_inputs))

  def detect(input_frame):
    # as modeld's warp: no split reduce, so the gathers lower to plain indexed loads
    with Context(SPLIT_REDUCEOP=0):
      img = prepare(input_frame).realize()
    return next(iter(runner({input_name: img}).values()))
  return lb, detect
