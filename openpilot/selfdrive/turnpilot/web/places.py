"""Place search (Photon, then Nominatim) and Apple Maps share links."""
import json
import re
import urllib.parse
import urllib.request

from openpilot.selfdrive.turnpilot.web.store import valid_latlon

# Nominatim's usage policy requires a User-Agent that identifies the app
USER_AGENT = "turnpilot-web/0.1 (openpilot fork; https://github.com/kaileh57/turnpilot)"
PHOTON_URL = "https://photon.komoot.io/api/"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
TIMEOUT = 8
LIMIT = 8
APPLE_HOST = "maps.apple.com"


def fetch(url: str, timeout: float = TIMEOUT, limit: int = 1 << 20) -> tuple[str, bytes]:
  """GET a URL, returns (final url after redirects, body)."""
  req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/html"})
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    return resp.geturl(), resp.read(limit)


def fetch_json(url: str):
  return json.loads(fetch(url)[1])


def _join(*parts):
  seen, out = set(), []
  for p in parts:
    p = str(p).strip() if p else ""
    if p and p.lower() not in seen:
      seen.add(p.lower())
      out.append(p)
  return out


def photon_places(data) -> list[dict]:
  places = []
  for f in data.get("features", []):
    try:
      lon, lat = f["geometry"]["coordinates"][:2]
    except (KeyError, TypeError, ValueError):
      continue
    p = f.get("properties") or {}
    street = " ".join(_join(p.get("housenumber"), p.get("street")))
    name = p.get("name") or street or p.get("city") or ""
    detail = ", ".join(_join(street if street != name else "", p.get("city") or p.get("town") or p.get("village"),
                             p.get("state"), p.get("country")))
    if name and valid_latlon(lat, lon):
      places.append({"name": name, "detail": detail, "latitude": lat, "longitude": lon})
  return places


def nominatim_places(data) -> list[dict]:
  places = []
  for r in data if isinstance(data, list) else []:
    try:
      lat, lon = float(r["lat"]), float(r["lon"])
    except (KeyError, TypeError, ValueError):
      continue
    full = r.get("display_name") or ""
    name = r.get("name") or full.split(",")[0].strip()
    if name and valid_latlon(lat, lon):
      places.append({"name": name, "detail": full, "latitude": lat, "longitude": lon})
  return places


def search(query: str, near=None, get_json=fetch_json) -> list[dict]:
  """Photon first, Nominatim if it fails or finds nothing. near is (latitude, longitude) to bias towards."""
  q = {"q": query, "limit": LIMIT}
  if near:
    q.update(lat=f"{near[0]:.5f}", lon=f"{near[1]:.5f}")
  errors = []
  try:
    if places := photon_places(get_json(PHOTON_URL + "?" + urllib.parse.urlencode(q))):
      return places
  except Exception as e:
    errors.append(e)
  q = {"q": query, "format": "jsonv2", "limit": LIMIT}
  if near:  # a soft preference for about 100 km around the car, not a limit
    q["viewbox"] = f"{near[1] - 1:.5f},{near[0] + 1:.5f},{near[1] + 1:.5f},{near[0] - 1:.5f}"
  try:
    return nominatim_places(get_json(NOMINATIM_URL + "?" + urllib.parse.urlencode(q)))
  except Exception as e:
    if errors:  # both failed, as opposed to Photon finding nothing
      raise ConnectionError(f"search failed: {errors[0]}; {e}") from e
    return []


_LATLON = re.compile(r"\s*(-?\d{1,3}(?:\.\d+)?)\s*[, ]\s*(-?\d{1,3}(?:\.\d+)?)\s*")


def parse_latlon(text):
  """'39.77,-86.16' -> (39.77, -86.16), or None."""
  m = _LATLON.fullmatch(text or "")
  if m and valid_latlon(float(m[1]), float(m[2])):
    return float(m[1]), float(m[2])
  return None


def parse_apple_maps(url: str) -> dict | None:
  """Pick what an Apple Maps link says out of its query string.

  Returns {latitude, longitude, name} when the link carries coordinates, {query, name} when it only names
  or addresses a place, and None when it is not a link we understand.
  """
  u = urllib.parse.urlsplit(url.strip())
  if u.scheme not in ("http", "https", "maps"):
    return None
  args = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items() if v}
  # coordinate= is the newer share link, ll= the older one, sll= the map's search center
  for key in ("coordinate", "ll", "daddr", "q", "sll"):
    if latlon := parse_latlon(args.get(key)):
      break
  text = [args.get(k, "") for k in ("name", "q", "address", "daddr")]
  name = next((t for t in text if t and not parse_latlon(t)), "")
  if latlon:
    return {"latitude": latlon[0], "longitude": latlon[1], "name": name}
  query = args.get("address") or name
  return {"query": query, "name": name} if query else None


def resolve_apple_maps(url: str, get=fetch) -> dict | None:
  """parse_apple_maps, following a short link (maps.apple.com/p/...) first, since it has no query string."""
  place = parse_apple_maps(url)
  u = urllib.parse.urlsplit(url.strip())
  has_coords = place is not None and "latitude" in place
  if has_coords or u.hostname != APPLE_HOST or not u.path.startswith("/p/"):
    return place
  try:
    final, body = get(url)
  except Exception:
    return place
  if urllib.parse.urlsplit(final).hostname == APPLE_HOST and (found := parse_apple_maps(final)) and "latitude" in found:
    return found
  m = re.search(rb"[?&;](?:ll|coordinate)=(-?\d+\.\d+)(?:,|%2C)(-?\d+\.\d+)", body)
  if m and valid_latlon(float(m[1]), float(m[2])):
    return {"latitude": float(m[1]), "longitude": float(m[2]), "name": place["name"] if place else ""}
  return place
