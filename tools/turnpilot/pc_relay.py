#!/usr/bin/env python3
"""File relay between the Mac and piston (the BeamNG PC) over the USB4 link.

GET serves everything under the relay root. PUT or POST writes into incoming/:
a file sent as "x" or "incoming/x" lands at incoming/x. Multipart form uploads
(curl -F file=@x) are accepted too.
"""
from __future__ import annotations

import argparse
import email.parser
import email.policy
import http.server
import os
from datetime import datetime
from pathlib import Path


def make_handler(root: Path):
  incoming = (root / "incoming").resolve()

  class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, directory=str(root), **kwargs)

    def end_headers(self):
      self.send_header("Cache-Control", "no-store")
      super().end_headers()

    def _target(self, name: str) -> Path | None:
      rel = name.split("?", 1)[0].lstrip("/")
      if rel.startswith("incoming/"):
        rel = rel[len("incoming/"):]
      path = (incoming / rel).resolve()
      if not rel or incoming not in path.parents:
        return None
      return path

    def _write(self, path: Path, data: bytes):
      path.parent.mkdir(parents=True, exist_ok=True)
      tmp = path.with_name(path.name + ".part")
      tmp.write_bytes(data)
      os.replace(tmp, path)
      print(f"{datetime.now().isoformat(timespec='seconds')} wrote {path.relative_to(root)} ({len(data)} bytes)", flush=True)

    def _upload(self):
      length = int(self.headers.get("Content-Length", 0))
      body = self.rfile.read(length)
      ctype = self.headers.get("Content-Type", "")
      written = []
      if ctype.startswith("multipart/form-data"):
        msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
          f"Content-Type: {ctype}\r\n\r\n".encode() + body)
        base = self.path.rstrip("/") if self.path.strip("/") not in ("", "incoming") else ""
        for part in msg.iter_parts():
          fname = part.get_filename()
          if not fname:
            continue
          path = self._target(f"{base}/{fname}" if base else fname)
          if path is None:
            return self.send_error(400, "bad path")
          self._write(path, part.get_payload(decode=True))
          written.append(path)
      else:
        path = self._target(self.path)
        if path is None:
          return self.send_error(400, "bad path")
        self._write(path, body)
        written.append(path)
      self.send_response(201)
      self.send_header("Content-Type", "text/plain")
      self.end_headers()
      self.wfile.write("".join(f"{p.relative_to(root)}\n" for p in written).encode())

    do_PUT = _upload
    do_POST = _upload

  return Handler


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", default=str(Path.home() / "pc_relay"))
  parser.add_argument("--host", default="192.168.137.74")
  parser.add_argument("--port", type=int, default=8765)
  args = parser.parse_args()
  root = Path(args.root).resolve()
  (root / "incoming").mkdir(parents=True, exist_ok=True)
  server = http.server.ThreadingHTTPServer((args.host, args.port), make_handler(root))
  print(f"serving {root} on {args.host}:{args.port}", flush=True)
  server.serve_forever()


if __name__ == "__main__":
  main()
