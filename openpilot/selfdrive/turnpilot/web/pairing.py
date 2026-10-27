"""Things the phone needs to find and trust the comma: its address, the pairing URL, the QR code and the home-screen icon."""
import socket
import struct
import zlib

from openpilot.common.params import Params
from openpilot.selfdrive.turnpilot.web.store import ensure_token

PORT = 8088
HOTSPOT_GATEWAY = "172.20.10.1"  # an iPhone personal hotspot is always 172.20.10.0/28 with the phone at .1


def _source_ip(target: str) -> str | None:
  """The address the kernel would send from to reach target. A UDP connect sends nothing."""
  with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    try:
      s.connect((target, 80))
      ip = s.getsockname()[0]
    except OSError:
      return None
  return None if ip.startswith("127.") or ip == "0.0.0.0" else ip


def local_ip() -> str:
  """Our address on the hotspot, else on whatever network has the default route."""
  return _source_ip(HOTSPOT_GATEWAY) or _source_ip("8.8.8.8") or "127.0.0.1"


def pairing_url(params: Params | None = None) -> str:
  """http://<ip>:8088/?t=<token>, the URL for the QR code on the comma screen. Creates the token if there is none."""
  return f"http://{local_ip()}:{PORT}/?t={ensure_token(params or Params())}"


def qr_svg(text: str, border: int = 4) -> str:
  # imported here because openpilot.common.qrcode pulls in raylib, which the rest of the server doesn't need
  from openpilot.common.qrcode import _Qr, _capacity
  raw = text.encode()
  for version in range(1, 21):
    if 4 + (8 if version <= 9 else 16) + len(raw) * 8 <= _capacity(version) * 8:
      break
  else:
    raise ValueError("text is too long for a QR code")
  modules = _Qr(version, raw).modules
  side = len(modules) + 2 * border
  cells = "".join(f"M{x + border} {y + border}h1v1h-1z" for y, row in enumerate(modules) for x, dark in enumerate(row) if dark)
  svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {side} {side}" shape-rendering="crispEdges">'
  return svg + f'<rect width="{side}" height="{side}" fill="#fff"/><path d="{cells}"/></svg>'


def icon_png(size: int = 180) -> bytes:
  """A white arrow on black, as the 180x180 PNG iOS wants for the home screen. Drawn here so no binary lives in the repo."""
  rows = bytearray()
  for y in range(size):
    rows.append(0)  # PNG filter: none
    half = (y / size - 0.2) / 0.6 * 0.3  # half width of the arrow at this row, as a fraction of the size
    for x in range(size):
      inside = 0.2 <= y / size <= 0.8 and abs(x / size - 0.5) <= half
      rows += b"\xff\xff\xff" if inside else b"\x00\x00\x00"

  def chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

  header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB
  return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(rows), 9)) + chunk(b"IEND", b"")
