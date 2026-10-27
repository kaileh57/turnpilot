import numpy as np

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


class NV12Converter:
  """RGB(A) to NV12 with the same BT.601 integer math as tools/sim/lib/camerad.rgb_to_nv12, in uint16 with preallocated
  buffers (about 4x faster at 1344x760). The alpha channel of BeamNG's RGBA frames is ignored."""

  def __init__(self, w: int, h: int):
    self.w, self.h = w, h
    self.stride, self.y_height, self.uv_height, self.size = get_nv12_info(w, h)
    self.buf = np.zeros(self.size, dtype=np.uint8)
    planes = self.buf[:self.stride * (self.y_height + self.uv_height)].reshape(-1, self.stride)
    self.y_plane = planes[:h, :w]
    self.uv_plane = planes[self.y_height:self.y_height + h // 2, :w].reshape(h // 2, w // 2, 2)
    self.acc = np.empty((h, w), dtype=np.uint16)
    self.tmp = np.empty((h, w), dtype=np.uint16)
    self.sub = [np.empty((h // 2, w // 2), dtype=np.uint16) for _ in range(3)]
    self.c_acc = np.empty((h // 2, w // 2), dtype=np.uint16)
    self.c_tmp = np.empty((h // 2, w // 2), dtype=np.uint16)

  def _sub2x2(self, ch, out):
    np.add(ch[0::2, 0::2], ch[0::2, 1::2], out=out, dtype=np.uint16)
    out += ch[1::2, 0::2]
    out += ch[1::2, 1::2]
    out += 2
    out >>= 2

  def convert(self, rgb: np.ndarray) -> np.ndarray:
    """rgb: (h, w, 3 or 4) uint8. Returns the NV12 buffer (owned by this converter, overwritten on the next call)."""
    assert rgb.shape[:2] == (self.h, self.w) and rgb.dtype == np.uint8, rgb.shape
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    acc, tmp = self.acc, self.tmp

    # Y = ((13 b + 65 g + 33 r + 64) >> 7) + 16, max 28369 fits uint16, result <= 237 so no clip needed
    np.multiply(b, 13, out=acc, dtype=np.uint16)
    np.multiply(g, 65, out=tmp, dtype=np.uint16)
    acc += tmp
    np.multiply(r, 33, out=tmp, dtype=np.uint16)
    acc += tmp
    acc += 64
    acc >>= 7
    acc += 16
    self.y_plane[:] = acc

    rs, gs, bs = self.sub
    self._sub2x2(r, rs)
    self._sub2x2(g, gs)
    self._sub2x2(b, bs)
    c, t = self.c_acc, self.c_tmp
    # U = (56 b - 37 g - 19 r + 0x8080) >> 8 and V = (56 r - 47 g - 9 b + 0x8080) >> 8; adding the offset first keeps every
    # intermediate in [18616, 47176], inside uint16, and the results in [72, 184]
    np.multiply(bs, 56, out=c)
    c += 0x8080
    np.multiply(gs, 37, out=t)
    c -= t
    np.multiply(rs, 19, out=t)
    c -= t
    c >>= 8
    self.uv_plane[:, :, 0] = c
    np.multiply(rs, 56, out=c)
    c += 0x8080
    np.multiply(gs, 47, out=t)
    c -= t
    np.multiply(bs, 9, out=t)
    c -= t
    c >>= 8
    self.uv_plane[:, :, 1] = c
    return self.buf
