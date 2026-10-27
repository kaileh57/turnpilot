#!/usr/bin/env python3
"""turnpilot_web: the phone companion. One page and a small JSON API, standard library only.

Chase's phone joins nothing: the comma joins the phone's hotspot, and this serves one page on port 8088. Every request
needs the token in TurnPilotWebToken, as a Bearer header or as the cookie set by opening /?t=<token> once.
"""
import hmac
import json
import urllib.parse
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.turnpilot.web import places
from openpilot.selfdrive.turnpilot.web.pairing import PORT, icon_png, pairing_url, qr_svg
from openpilot.selfdrive.turnpilot.web.status import StatusReader, format_status
from openpilot.selfdrive.turnpilot.web.store import Store, ensure_token, valid_latlon

INDEX = Path(__file__).with_name("index.html")
COOKIE = "tp_token"
MAX_BODY = 16 * 1024
LOOPBACK = ("127.0.0.1", "::1")
UNPAIRED = b"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>TurnPilot</title><body style="font:20px system-ui;margin:24px">
<h1>Not paired</h1><p>Scan the QR code on the comma screen with the phone camera.</p>"""


class ApiError(Exception):
  def __init__(self, code: int, message: str):
    super().__init__(message)
    self.code = code


def number(value):
  """A coordinate from JSON, which Shortcuts may send as a string."""
  if isinstance(value, bool):
    return None
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


class App:
  """What the handlers share. search and resolve can be swapped for tests."""
  def __init__(self, params, reader=None, search=places.search, resolve=places.resolve_apple_maps):
    self.store = Store(params)
    self.reader = reader
    self.search = search
    self.resolve = resolve

  def token_ok(self, candidates) -> bool:
    expected = self.store.token().encode()
    return bool(expected) and any(hmac.compare_digest(c.encode(), expected) for c in candidates)

  def position(self):
    """(latitude, longitude) to bias searches towards: the live fix, else the last one saved."""
    snap = self.reader.get() if self.reader else None
    return snap["gps"] if snap and snap["gps"] else self.store.last_gps()

  def status(self) -> dict:
    snap = self.reader.get() if self.reader else None
    return format_status(snap, self.store.destination(), self.store.flag("TurnPilotEnabled"), self.store.flag("IsMetric"))


class Handler(BaseHTTPRequestHandler):
  app: App  # set on a subclass by make_handler
  server_version = "turnpilot-web"

  def log_message(self, format, *args):  # noqa: A002
    pass  # the first request line carries the pairing token

  def do_GET(self):
    self.dispatch("GET")

  def do_POST(self):
    self.dispatch("POST")

  def do_DELETE(self):
    self.dispatch("DELETE")

  def send(self, code, body: bytes, ctype: str, extra=()):
    self.send_response(code)
    self.send_header("Content-Type", ctype)
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-store")
    self.send_header("Referrer-Policy", "no-referrer")
    self.send_header("X-Content-Type-Options", "nosniff")
    for k, v in extra:
      self.send_header(k, v)
    self.end_headers()
    self.wfile.write(body)

  def send_json(self, code, obj):
    self.send(code, json.dumps(obj).encode(), "application/json")

  def body(self) -> dict:
    length = int(self.headers.get("Content-Length") or 0)
    if length > MAX_BODY:
      raise ApiError(413, "body too large")
    try:
      data = json.loads(self.rfile.read(length) or b"{}")
    except ValueError:
      raise ApiError(400, "body is not JSON") from None
    if not isinstance(data, dict):
      raise ApiError(400, "body must be a JSON object")
    return data

  def credentials(self, query, is_page) -> list[str]:
    found = []
    auth = self.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
      found.append(auth[7:].strip())
    if morsel := SimpleCookie(self.headers.get("Cookie", "")).get(COOKIE):
      found.append(morsel.value)
    if is_page:  # the token in the URL works on the page only, so it stays out of API request lines
      found += query.get("t", [])
    return found

  def dispatch(self, method):
    url = urllib.parse.urlsplit(self.path)
    path, query = url.path, urllib.parse.parse_qs(url.query)
    try:
      if method == "GET" and path == "/icon.png":  # iOS fetches this without our cookie; it holds nothing private
        return self.send(200, icon_png(), "image/png", [("Cache-Control", "max-age=86400")])
      is_page = method == "GET" and path == "/"
      authorized = self.app.token_ok(self.credentials(query, is_page))
      if path == "/pair.svg":  # shows the token, so only the token holder or the device itself may see it
        authorized = authorized or self.client_address[0] in LOOPBACK
      if not authorized:
        if is_page:
          return self.send(401, UNPAIRED, "text/html; charset=utf-8")
        raise ApiError(401, "unauthorized")
      if method == "POST" and not self.headers.get("Content-Type", "").lower().startswith("application/json"):
        raise ApiError(415, "send application/json")  # a cross-site form can't send this without a preflight
      if is_page:
        return self.page(query)
      if method == "GET" and path == "/pair.svg":
        with self.app.store.lock:
          url = pairing_url(self.app.store.params)
        return self.send(200, qr_svg(url).encode(), "image/svg+xml")
      self.send_json(200, self.api(method, path, query))
    except ApiError as e:
      self.send_json(e.code, {"error": str(e)})
    except ValueError as e:
      self.send_json(400, {"error": str(e)})
    except Exception:
      cloudlog.exception("turnpilot_web.request_failed")
      self.send_json(500, {"error": "internal error"})

  def page(self, query):
    extra = []
    if token := query.get("t"):  # visited with the token: remember it for the next visit
      extra = [("Set-Cookie", f"{COOKIE}={token[0]}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Strict")]
    self.send(200, INDEX.read_bytes(), "text/html; charset=utf-8", extra)

  def api(self, method, path, query):
    app, store = self.app, self.app.store
    if method == "GET" and path == "/api/status":
      return app.status()
    if method == "GET" and path == "/api/search":
      return {"results": self.search(query)}
    if method == "GET" and path == "/api/favorites":
      return store.places()
    if method == "GET" and path == "/api/pairing":
      host = self.headers.get("Host") or f"localhost:{PORT}"
      return {"url": f"http://{host}/?t={store.token()}", "endpoint": f"http://{host}/api/destination", "token": store.token()}
    if method == "POST" and path == "/api/destination":
      return {"ok": True, **self.navigate(self.body())}
    if (method, path) in (("DELETE", "/api/destination"), ("POST", "/api/stop")):
      store.stop()
      return {"ok": True}
    if method == "POST" and path == "/api/favorites":
      body = self.body()
      return {"ok": True, **store.add_favorite(number(body.get("latitude")), number(body.get("longitude")), body.get("name"))}
    if method == "DELETE" and path.startswith("/api/favorites/"):
      if not store.remove_favorite(path.rsplit("/", 1)[1]):
        raise ApiError(404, "no such favorite")
      return {"ok": True}
    raise ApiError(404, "not found")

  def search(self, query) -> list[dict]:
    text = (query.get("q") or [""])[0].strip()
    if len(text) < 2:
      return []
    try:
      return self.app.search(text, self.app.position())
    except Exception as e:
      cloudlog.warning(f"turnpilot_web.search_failed {e}")
      raise ApiError(502, "place search is not reachable") from None

  def navigate(self, body) -> dict:
    """Set the destination from coordinates, or from a shared Apple Maps link."""
    place = body
    if "url" in body:
      place = self.app.resolve(str(body["url"]))
      if place is None:
        raise ApiError(400, "that is not a maps link")
      if "latitude" not in place:  # the link only names or addresses a place
        found = self.search({"q": [place["query"]]})
        if not found:
          raise ApiError(404, f"no place found for {place['query']!r}")
        place = {**found[0], "name": place["name"] or found[0]["name"]}
    lat, lon = number(place.get("latitude")), number(place.get("longitude"))
    if lat is None or lon is None or not valid_latlon(lat, lon):
      raise ApiError(400, "need latitude and longitude, or a url")
    return self.app.store.navigate(lat, lon, place.get("name"))


def make_handler(app: App):
  return type("BoundHandler", (Handler,), {"app": app})


class Server(ThreadingHTTPServer):
  request_queue_size = 64  # the default of 5 resets connections when a page opens several at once


def main():
  params = Params()
  ensure_token(params)
  reader = StatusReader()
  reader.start()
  server = Server(("", PORT), make_handler(App(params, reader)))
  cloudlog.info(f"turnpilot_web listening on :{PORT}")
  server.serve_forever()


if __name__ == "__main__":
  main()
