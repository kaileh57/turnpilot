import http.client
import json
import re
import struct
import threading
import time
import zlib

import numpy as np

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.common.qrcode import decode_matrix
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.turnpilot.web import pairing, places, server, status
from openpilot.selfdrive.turnpilot.web.store import Store, ensure_token, last_gps_position

INDIANAPOLIS = (39.7684, -86.1581)


def photon_feature(name, lat, lon, **props):
  return {"geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": {"name": name, **props}}


class FakeSearch:
  """Stands in for places.search so no test touches the network."""
  def __init__(self, results=None, error=None):
    self.results, self.error, self.calls = results or [], error, []

  def __call__(self, query, near=None):
    self.calls.append((query, near))
    if self.error:
      raise self.error
    return self.results


class WebTestCase(OpenpilotTestCase):
  """A real server on a free port, backed by the per-test Params that OpenpilotTestCase isolates."""
  def setup_method(self):
    self.params = Params()
    self.search = FakeSearch([{"name": "Cafe", "detail": "1 Main St", "latitude": 39.1, "longitude": -86.2}])
    self.app = server.App(self.params, reader=None, search=self.search, resolve=places.resolve_apple_maps)
    self.token = ensure_token(self.params)
    self.httpd = server.Server(("127.0.0.1", 0), server.make_handler(self.app))
    self.port = self.httpd.server_address[1]
    threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

  def teardown_method(self):
    self.httpd.shutdown()
    self.httpd.server_close()

  def call(self, method, path, body=None, auth="bearer", headers=None, content_type="application/json"):
    """(status, parsed body or bytes, response headers)."""
    h = dict(headers or {})
    if auth == "bearer":
      h["Authorization"] = f"Bearer {self.token}"
    elif auth == "cookie":
      h["Cookie"] = f"tp_token={self.token}"
    data = None
    if body is not None:
      data = body if isinstance(body, bytes) else json.dumps(body).encode()
    if method == "POST":
      h["Content-Type"] = content_type
    conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
    conn.request(method, path, body=data, headers=h)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    is_json = resp.getheader("Content-Type", "").startswith("application/json")
    return resp.status, json.loads(raw) if is_json else raw, dict(resp.getheaders())


class TestToken(WebTestCase):
  def test_generated_once_and_persisted(self):
    assert len(self.token) >= 20
    assert self.params.get("TurnPilotWebToken") == self.token
    assert ensure_token(self.params) == self.token

  def test_token_ok(self):
    assert self.app.token_ok(["nope", self.token])
    assert not self.app.token_ok([])
    assert not self.app.token_ok(["", "nope", self.token[:-1], self.token + "x"])
    self.params.remove("TurnPilotWebToken")
    assert not self.app.token_ok([""])  # an unset token must not accept an empty credential

  def test_api_needs_the_token(self):
    for method, path in [("GET", "/api/status"), ("GET", "/api/favorites"), ("GET", "/api/search?q=cafe"), ("GET", "/api/pairing"),
                         ("POST", "/api/destination"), ("POST", "/api/stop"), ("DELETE", "/api/destination")]:
      for auth in ("none", "wrong"):
        headers = {"Authorization": "Bearer wrong"} if auth == "wrong" else None
        code, body, _ = self.call(method, path, body={}, auth="none", headers=headers)
        assert code == 401, (method, path, auth)
        assert body == {"error": "unauthorized"}
    assert self.params.get("NavDestination") is None

  def test_bearer_and_cookie(self):
    assert self.call("GET", "/api/status", auth="bearer")[0] == 200
    assert self.call("GET", "/api/status", auth="cookie")[0] == 200
    assert self.call("GET", "/api/status", auth="none", headers={"Authorization": f"Basic {self.token}"})[0] == 401

  def test_query_token_works_on_the_page_only(self):
    code, body, _ = self.call("GET", "/", auth="none")
    assert code == 401 and b"Not paired" in body

    code, body, headers = self.call("GET", f"/?t={self.token}", auth="none")
    assert code == 200 and body.startswith(b"<!doctype html>")
    cookie = headers["Set-Cookie"]
    assert cookie.startswith(f"tp_token={self.token};") and "HttpOnly" in cookie and "SameSite=Strict" in cookie

    assert self.call("GET", "/?t=wrong", auth="none")[0] == 401
    assert self.call("GET", f"/api/status?t={self.token}", auth="none")[0] == 401

  def test_icon_is_public_and_a_png(self):
    code, body, headers = self.call("GET", "/icon.png", auth="none")
    assert code == 200 and headers["Content-Type"] == "image/png"
    assert body[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", body[16:24])
    assert (width, height) == (180, 180)
    idat = body[body.index(b"IDAT") + 4:body.index(b"IEND") - 8]
    assert len(zlib.decompress(idat)) == 180 * (1 + 180 * 3)

  def test_pair_svg_is_for_the_token_holder_or_the_device(self):
    assert self.call("GET", "/pair.svg", auth="bearer")[0] == 200
    self.monkeypatch_loopback(())
    assert self.call("GET", "/pair.svg", auth="none")[0] == 401
    assert self.call("GET", "/pair.svg", auth="bearer")[0] == 200
    self.monkeypatch_loopback(("127.0.0.1", "::1"))
    code, body, headers = self.call("GET", "/pair.svg", auth="none")
    assert code == 200 and headers["Content-Type"] == "image/svg+xml" and body.startswith(b"<svg")

  def monkeypatch_loopback(self, value):
    old = server.LOOPBACK
    server.LOOPBACK = value
    self.addCleanup(setattr, server, "LOOPBACK", old)


class TestDestination(WebTestCase):
  def test_write_and_name(self):
    code, body, _ = self.call("POST", "/api/destination", {"latitude": 39.1, "longitude": -86.2, "name": "Cafe"})
    assert code == 200 and body == {"ok": True, "name": "Cafe", "latitude": 39.1, "longitude": -86.2}
    # navd and the UI read exactly these two keys
    assert self.params.get("NavDestination") == {"latitude": 39.1, "longitude": -86.2}
    assert self.call("GET", "/api/favorites")[1]["recents"] == [{"name": "Cafe", "latitude": 39.1, "longitude": -86.2}]
    assert self.call("GET", "/api/status")[1]["destination"] == {"name": "Cafe", "latitude": 39.1, "longitude": -86.2}

  def test_unnamed_destination_shows_coordinates(self):
    self.params.put("NavDestination", {"latitude": 1.5, "longitude": 2.5}, block=True)  # e.g. set over ssh
    assert self.call("GET", "/api/status")[1]["destination"]["name"] == "1.50000, 2.50000"

  def test_numbers_as_strings(self):
    assert self.call("POST", "/api/destination", {"latitude": "39.5", "longitude": "-86.5"})[0] == 200
    assert self.params.get("NavDestination") == {"latitude": 39.5, "longitude": -86.5}

  def test_bad_requests_leave_the_destination_alone(self):
    self.call("POST", "/api/destination", {"latitude": 39.1, "longitude": -86.2})
    good = self.params.get("NavDestination")
    for body in [{"latitude": 91, "longitude": 0}, {"latitude": 0, "longitude": -181}, {"latitude": "x", "longitude": 1},
                 {"latitude": True, "longitude": 1}, {"latitude": 1}, {}, {"url": "not a url"}, {"url": "https://example.com/"}]:
      assert self.call("POST", "/api/destination", body)[0] == 400, body
    assert self.call("POST", "/api/destination", b"{nope")[0] == 400
    assert self.call("POST", "/api/destination", b"[1]")[0] == 400
    assert self.call("POST", "/api/destination", {"latitude": 1, "longitude": 2}, content_type="text/plain")[0] == 415
    assert self.params.get("NavDestination") == good

  def test_stop(self):
    self.call("POST", "/api/destination", {"latitude": 39.1, "longitude": -86.2})
    assert self.call("POST", "/api/stop", {})[0] == 200
    assert self.params.get("NavDestination") is None
    assert self.call("GET", "/api/status")[1]["destination"] is None
    self.call("POST", "/api/destination", {"latitude": 39.1, "longitude": -86.2})
    assert self.call("DELETE", "/api/destination")[0] == 200
    assert self.params.get("NavDestination") is None
    assert self.call("POST", "/api/stop", {})[0] == 200  # clearing nothing is fine

  def test_apple_maps_link_with_coordinates(self):
    url = "https://maps.apple.com/?ll=39.7684%2C-86.1581&q=Monument%20Circle&t=m"
    code, body, _ = self.call("POST", "/api/destination", {"url": url})
    assert code == 200 and body["name"] == "Monument Circle"
    assert self.params.get("NavDestination") == {"latitude": 39.7684, "longitude": -86.1581}
    assert self.search.calls == []

  def test_apple_maps_link_with_only_an_address_is_geocoded(self):
    url = "https://maps.apple.com/?address=1%20Main%20St%2C%20Bloomington%2C%20IN&auid=1"
    code, body, _ = self.call("POST", "/api/destination", {"url": url})
    assert code == 200 and body["name"] == "1 Main St, Bloomington, IN"  # the shared address names it, the match only places it
    assert self.search.calls == [("1 Main St, Bloomington, IN", None)]
    assert self.params.get("NavDestination") == {"latitude": 39.1, "longitude": -86.2}

  def test_apple_maps_link_nothing_found(self):
    self.search.results = []
    assert self.call("POST", "/api/destination", {"url": "https://maps.apple.com/?q=zzzz"})[0] == 404
    assert self.params.get("NavDestination") is None

  def test_search_bias_uses_the_last_gps_fix(self):
    msg = messaging.new_message("gpsLocationExternal", valid=True)
    msg.gpsLocationExternal.latitude, msg.gpsLocationExternal.longitude = INDIANAPOLIS
    self.params.put("LastGPSPosition", msg.to_bytes(), block=True)
    assert last_gps_position(self.params) == INDIANAPOLIS
    code, body, _ = self.call("GET", "/api/search?q=coffee%20shop")
    assert code == 200 and body["results"][0]["name"] == "Cafe"
    assert self.search.calls == [("coffee shop", INDIANAPOLIS)]

  def test_search_edge_cases(self):
    code, body, _ = self.call("GET", "/api/search?q=a")
    assert (code, body) == (200, {"results": []}) and self.search.calls == []
    self.search.error = ConnectionError("offline")
    assert self.call("GET", "/api/search?q=cafe")[0] == 502


class TestFavorites(WebTestCase):
  def test_add_list_remove(self):
    code, fav, _ = self.call("POST", "/api/favorites", {"name": "Home", "latitude": 39.1, "longitude": -86.2})
    assert code == 200 and len(fav["id"]) == 8
    self.call("POST", "/api/favorites", {"name": "Work", "latitude": 39.2, "longitude": -86.3})
    names = [p["name"] for p in self.call("GET", "/api/favorites")[1]["favorites"]]
    assert names == ["Home", "Work"]
    assert self.params.get("TurnPilotFavorites")["favorites"][0]["id"] == fav["id"]

    assert self.call("DELETE", f"/api/favorites/{fav['id']}")[0] == 200
    assert [p["name"] for p in self.call("GET", "/api/favorites")[1]["favorites"]] == ["Work"]
    assert self.call("DELETE", f"/api/favorites/{fav['id']}")[0] == 404

  def test_saving_the_same_place_renames_it(self):
    first = self.call("POST", "/api/favorites", {"name": "Home", "latitude": 39.1, "longitude": -86.2})[1]
    again = self.call("POST", "/api/favorites", {"name": "Mom's", "latitude": 39.1, "longitude": -86.2})[1]
    assert again["id"] == first["id"]
    assert [p["name"] for p in self.call("GET", "/api/favorites")[1]["favorites"]] == ["Mom's"]

  def test_invalid_favorite(self):
    assert self.call("POST", "/api/favorites", {"name": "Nowhere", "latitude": 99, "longitude": 0})[0] == 400
    assert self.call("GET", "/api/favorites")[1] == {"favorites": [], "recents": []}

  def test_favorite_name_wins_for_the_destination(self):
    self.call("POST", "/api/favorites", {"name": "Home", "latitude": 39.1, "longitude": -86.2})
    self.call("POST", "/api/destination", {"name": "1 Main St", "latitude": 39.1, "longitude": -86.2})
    assert self.call("GET", "/api/status")[1]["destination"]["name"] == "Home"

  def test_recents_are_newest_first_deduped_and_capped(self):
    store = Store(self.params)
    for i in range(15):
      store.navigate(30 + i, -80, f"Place {i}")
    store.navigate(35, -80, "Place 5 again")
    recents = store.places()["recents"]
    assert len(recents) == 10
    assert recents[0]["name"] == "Place 5 again" and recents[1]["name"] == "Place 14"
    assert sum(p["latitude"] == 35 for p in recents) == 1

  def test_concurrent_requests_do_not_lose_writes(self):
    codes = []

    def add(i):
      body = {"name": f"Place {i}", "latitude": 30 + i, "longitude": -80}
      codes.append(self.call("POST", "/api/favorites", body)[0])
      codes.append(self.call("POST", "/api/destination", body)[0])
    threads = [threading.Thread(target=add, args=(i,)) for i in range(20)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()
    assert codes == [200] * 40
    places = self.call("GET", "/api/favorites")[1]
    assert sorted(p["latitude"] for p in places["favorites"]) == [30 + i for i in range(20)]
    assert len(places["recents"]) == 10

  def test_survives_garbage_in_the_param(self):
    self.params.put("TurnPilotFavorites", ["not", "a", "dict"], block=True)
    assert self.call("GET", "/api/favorites")[1] == {"favorites": [], "recents": []}
    self.params.put("TurnPilotFavorites", {"favorites": [1, {"name": "ok", "latitude": 1, "longitude": 2}]}, block=True)
    assert [p["name"] for p in self.call("GET", "/api/favorites")[1]["favorites"]] == ["ok"]


class TestPlaces(OpenpilotTestCase):
  APPLE = [
    # old share link
    ("https://maps.apple.com/?ll=39.7684,-86.1581&q=Monument%20Circle&t=m",
     {"latitude": 39.7684, "longitude": -86.1581, "name": "Monument Circle"}),
    # newer links use coordinate=
    ("https://maps.apple.com/place?coordinate=39.7684,-86.1581&name=Soldiers%20Monument&address=x",
     {"latitude": 39.7684, "longitude": -86.1581, "name": "Soldiers Monument"}),
    ("https://maps.apple.com/?auid=123&ll=39.1%2C-86.2&lsp=9902&q=Cafe%20Roma", {"latitude": 39.1, "longitude": -86.2, "name": "Cafe Roma"}),
    ("http://maps.apple.com/?daddr=40.0,-75.5", {"latitude": 40.0, "longitude": -75.5, "name": ""}),
    ("https://maps.apple.com/?q=40.5,-75.25", {"latitude": 40.5, "longitude": -75.25, "name": ""}),
    ("maps://?ll=-33.86,151.21&q=Sydney", {"latitude": -33.86, "longitude": 151.21, "name": "Sydney"}),
    ("https://maps.apple.com/?sll=39.5,-86.5&q=Pizza", {"latitude": 39.5, "longitude": -86.5, "name": "Pizza"}),
    # no coordinates: only something to geocode
    ("https://maps.apple.com/?q=Starbucks", {"query": "Starbucks", "name": "Starbucks"}),
    ("https://maps.apple.com/?address=1%20Main%20St&q=Cafe", {"query": "1 Main St", "name": "Cafe"}),
    # junk
    ("https://maps.apple.com/", None),
    ("ftp://maps.apple.com/?ll=1,2", None),
    ("hello", None),
    ("https://maps.apple.com/?ll=95,200", None),
  ]

  def test_parse_apple_maps(self):
    for url, expected in self.APPLE:
      assert places.parse_apple_maps(url) == expected, url

  def test_short_link_is_followed_only_on_apple_maps(self):
    short = "https://maps.apple.com/p/abc123"
    seen = []

    def get(url):
      seen.append(url)
      return "https://maps.apple.com/?ll=39.5%2C-86.5&q=Park", b""
    assert places.resolve_apple_maps(short, get) == {"latitude": 39.5, "longitude": -86.5, "name": "Park"}

    def page(url):
      return url, b'<a href="/place?foo=1&amp;coordinate=39.25%2C-86.75">'
    assert places.resolve_apple_maps(short, page) == {"latitude": 39.25, "longitude": -86.75, "name": ""}

    def boom(url):
      raise OSError("offline")
    assert places.resolve_apple_maps(short, boom) is None

    seen.clear()
    assert places.resolve_apple_maps("https://evil.example/p/abc?ll=1,2", get) == {"latitude": 1.0, "longitude": 2.0, "name": ""}
    assert places.resolve_apple_maps("https://evil.example/p/abc", get) is None
    assert seen == []  # never fetches a host other than maps.apple.com

  def test_photon_results(self):
    data = {"features": [
      photon_feature("Cafe Roma", 39.1, -86.2, street="Main St", housenumber="12", city="Bloomington", state="Indiana", country="USA"),
      {"geometry": {"coordinates": [-86.0, 39.0]}, "properties": {"street": "Oak Ave", "housenumber": "3", "city": "Carmel"}},
      photon_feature("Broken", 999, 0),
      {"properties": {"name": "No geometry"}},
    ]}
    assert places.photon_places(data) == [
      {"name": "Cafe Roma", "detail": "12 Main St, Bloomington, Indiana, USA", "latitude": 39.1, "longitude": -86.2},
      {"name": "3 Oak Ave", "detail": "Carmel", "latitude": 39.0, "longitude": -86.0},
    ]

  def test_nominatim_results(self):
    data = [{"lat": "39.1", "lon": "-86.2", "name": "Cafe Roma", "display_name": "Cafe Roma, Main St, Bloomington"},
            {"lat": "39.2", "lon": "-86.3", "name": "", "display_name": "Oak Ave, Carmel"}, {"lat": "x", "lon": "1"}]
    assert places.nominatim_places(data) == [
      {"name": "Cafe Roma", "detail": "Cafe Roma, Main St, Bloomington", "latitude": 39.1, "longitude": -86.2},
      {"name": "Oak Ave", "detail": "Oak Ave, Carmel", "latitude": 39.2, "longitude": -86.3},
    ]

  def test_search_uses_photon_with_the_bias(self):
    urls = []

    def get_json(url):
      urls.append(url)
      return {"features": [photon_feature("Cafe", 39.1, -86.2)]}
    found = places.search("cafe roma", INDIANAPOLIS, get_json)
    assert found[0]["name"] == "Cafe" and len(urls) == 1
    assert urls[0].startswith("https://photon.komoot.io/api/?q=cafe+roma&limit=8&lat=39.76840&lon=-86.15810")

  def test_search_falls_back_to_nominatim(self):
    urls = []

    def photon_down(url):
      urls.append(url)
      if "photon" in url:
        raise OSError("down")
      return [{"lat": "39.1", "lon": "-86.2", "display_name": "Cafe, Bloomington"}]
    assert places.search("cafe", INDIANAPOLIS, photon_down)[0]["name"] == "Cafe"
    assert "nominatim.openstreetmap.org/search?q=cafe&format=jsonv2&limit=8&viewbox=" in urls[1]

    def photon_empty(url):
      return {"features": []} if "photon" in url else [{"lat": "1", "lon": "2", "display_name": "Somewhere"}]
    assert places.search("somewhere", None, photon_empty)[0]["name"] == "Somewhere"

  def test_search_failures(self):
    def down(url):
      raise OSError("down")
    try:
      places.search("cafe", None, down)
      raise AssertionError("expected ConnectionError")
    except ConnectionError:
      pass

    def photon_only_empty(url):
      return {"features": []} if "photon" in url else []
    assert places.search("zzzz", None, photon_only_empty) == []

  def test_user_agent_identifies_the_app(self):
    assert places.USER_AGENT.startswith("turnpilot-web/") and "http" in places.USER_AGENT


class TestPairing(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()

  def test_pairing_url(self):
    url = pairing.pairing_url(self.params)
    token = self.params.get("TurnPilotWebToken")
    assert re.fullmatch(rf"http://\d+\.\d+\.\d+\.\d+:8088/\?t={token}", url), url

  def test_qr_round_trips(self):
    url = pairing.pairing_url(self.params)
    svg = pairing.qr_svg(url)
    border = 4
    side = int(re.findall(r'viewBox="0 0 (\d+) ', svg)[0])
    n = side - 2 * border
    modules = np.zeros((n, n), dtype=bool)
    for x, y in re.findall(r"M(\d+) (\d+)h1v1h-1z", svg):
      modules[int(y) - border, int(x) - border] = True
    assert decode_matrix(modules) == url


class TestStatus(OpenpilotTestCase):
  ROUTE = {"text": "Main St", "type": "turn", "modifier": "left", "maneuver_m": 310., "remaining_m": 6600., "remaining_s": 780.,
           "points": 120}
  PLAN = {"state": "following", "reason": "", "shadow": False}

  def snap(self, **kw):
    return {"at": 100., "onroad": True, "route": self.ROUTE, "plan": self.PLAN, "gps": (39.7, -86.1), **kw}

  def test_navigating(self):
    dest = {"name": "Home", "latitude": 1, "longitude": 2}
    s = status.format_status(self.snap(), dest, True, False, now=101.)
    assert s == {
      "reading": True, "onroad": True, "destination": dest, "navigating": True,
      "maneuver": {"text": "Turn left onto Main St", "distance": "1020 ft"},
      "remaining": {"distance": "4.1 mi", "duration": "13 min", "seconds": 780.},
      "turnpilot": {"enabled": True, "state": "following", "label": "Following route", "reason": "", "shadow": False},
      "gps": {"latitude": 39.7, "longitude": -86.1},
    }
    metric = status.format_status(self.snap(), dest, True, True, now=101.)
    assert metric["maneuver"]["distance"] == "310 m" and metric["remaining"]["distance"] == "6.6 km"

  def test_no_route(self):
    s = status.format_status(self.snap(route=None, onroad=False, gps=None), None, True, False, now=101.)
    assert s["navigating"] is False and s["maneuver"] is None and s["remaining"] is None and s["gps"] is None
    assert s["onroad"] is False and s["destination"] is None

  def test_turnpilot_states(self):
    plan = {"state": "holding", "reason": "pedestrian in crosswalk", "shadow": True}
    tp = status.format_status(self.snap(plan=plan), None, True, False, now=101.)["turnpilot"]
    assert (tp["state"], tp["label"], tp["reason"], tp["shadow"]) == ("holding", "Holding", "pedestrian in crosswalk", True)
    tp = status.format_status(self.snap(plan=None), None, False, False, now=101.)["turnpilot"]
    assert tp == {"enabled": False, "state": "offline", "label": "Not running", "reason": "turnpilotd is not running", "shadow": False}

  def test_stale_or_missing_snapshot_reads_as_unavailable(self):
    for snap, now in [(None, 0.), (self.snap(), 100. + status.STALE_AFTER + 1)]:
      s = status.format_status(snap, None, True, False, now=now)
      assert s["reading"] is False and s["onroad"] is False and s["navigating"] is False
      assert s["turnpilot"]["state"] == "offline"

  def test_text_helpers(self):
    assert [status.format_distance(m, False) for m in (0, 44, 120, 160, 400, 1609, 8000, 16100, 40000)] == \
           ["0 ft", "140 ft", "390 ft", "520 ft", "0.2 mi", "1.0 mi", "5.0 mi", "10 mi", "25 mi"]
    assert [status.format_distance(m, True) for m in (4, 120, 999, 1500, 9500, 12000)] == ["0 m", "120 m", "1000 m", "1.5 km", "9.5 km", "12 km"]
    assert [status.format_duration(s) for s in (10, 60, 125, 3540, 3600, 4500)] == ["<1 min", "1 min", "2 min", "59 min", "1 h 00 min", "1 h 15 min"]
    assert status.maneuver_text("turn", "slight right", "Oak Ave") == "Turn slight right onto Oak Ave"
    assert status.maneuver_text("arrive", "right", "12 Main St") == "Arrive at 12 Main St"
    assert status.maneuver_text("arrive", "", "") == "Arrive"
    assert status.maneuver_text("off ramp", "right", "I-65 N") == "Take exit right onto I-65 N"
    assert status.maneuver_text("", "", "Main St") == "Main St"

  def test_status_endpoint_without_a_reader(self):
    app = server.App(Params(), reader=None)
    assert app.status()["reading"] is False


class TestStatusReader(WebTestCase):
  """Real cereal messages through a real SubMaster, into the /api/status JSON."""
  def test_end_to_end(self):
    reader = status.StatusReader()
    self.app.reader = reader
    reader.start()
    self.addCleanup(reader.stop)

    pm = messaging.PubMaster(["deviceState", "navInstruction", "navRoute", "turnpilotPlan", "gpsLocation"])
    self.call("POST", "/api/destination", {"latitude": 39.1, "longitude": -86.2, "name": "Home"})
    self.params.put_bool("IsMetric", True)

    def publish():
      msg = messaging.new_message("deviceState", valid=True)
      msg.deviceState.started = True
      pm.send("deviceState", msg)
      msg = messaging.new_message("navInstruction", valid=True)
      ni = msg.navInstruction
      ni.maneuverPrimaryText, ni.maneuverType, ni.maneuverModifier = "Main Street", "turn", "left"
      ni.maneuverDistance, ni.distanceRemaining, ni.timeRemaining = 250., 5400., 600.
      pm.send("navInstruction", msg)
      msg = messaging.new_message("navRoute", valid=True)
      msg.navRoute.init("coordinates", 3)
      pm.send("navRoute", msg)
      msg = messaging.new_message("turnpilotPlan")  # turnpilotd leaves valid unset
      msg.turnpilotPlan.state = "approaching"
      msg.turnpilotPlan.reason = "waiting for oncoming"
      pm.send("turnpilotPlan", msg)
      msg = messaging.new_message("gpsLocation", valid=True)
      msg.gpsLocation.hasFix, msg.gpsLocation.latitude, msg.gpsLocation.longitude = True, 39.7, -86.1
      pm.send("gpsLocation", msg)

    body = {}
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
      publish()
      time.sleep(0.2)
      body = self.call("GET", "/api/status")[1]
      if body["navigating"] and body["gps"] and body["turnpilot"]["state"] != "offline":
        break
    assert body["reading"] and body["onroad"] and body["navigating"]
    assert body["destination"]["name"] == "Home"
    assert body["maneuver"] == {"text": "Turn left onto Main Street", "distance": "250 m"}
    assert body["remaining"] == {"distance": "5.4 km", "duration": "10 min", "seconds": 600.}
    assert (body["turnpilot"]["state"], body["turnpilot"]["reason"]) == ("approaching", "waiting for oncoming")
    assert body["gps"] == {"latitude": 39.7, "longitude": -86.1}
    assert self.app.position() == (39.7, -86.1)  # the live fix is what searches bias towards
