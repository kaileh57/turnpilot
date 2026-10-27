#!/usr/bin/env python3
"""Small steps of build_valhalla_tiles.sh that are easier in Python than in shell.

  config <out_dir>                    write <out_dir>/valhalla.json for a driving-only build
  has_country <admin.sqlite>          exit 0 if the admin database has a country (admin_level 2)
  country <input> <out.osm> <iso>     write one country polygon around the input's bounding box

Real extracts carry boundary relations, but a regional one has no polygon for the country itself. Without
one Valhalla can't tell the driving side and assumes left-hand traffic, which changes turn costs and
roundabout directions. So the build adds a polygon for the country around the data when it is missing.
"""
import json
import re
import sqlite3
import struct
import sys
import zlib

PAD = 0.05  # degrees


def varint(data, i):
  value = shift = 0
  while True:
    b = data[i]
    i += 1
    value |= (b & 0x7f) << shift
    if b < 0x80:
      return value, i
    shift += 7


def fields(data):
  """(field number, value) of every field in a protobuf message; length-delimited values are bytes."""
  i = 0
  while i < len(data):
    key, i = varint(data, i)
    kind = key & 7
    if kind == 0:
      value, i = varint(data, i)
    elif kind == 2:
      n, i = varint(data, i)
      value, i = data[i:i + n], i + n
    elif kind == 1:
      value, i = data[i:i + 8], i + 8
    elif kind == 5:
      value, i = data[i:i + 4], i + 4
    else:
      raise ValueError(f"unsupported wire type {kind}")
    yield key >> 3, value


def unzigzag(v):
  return (v >> 1) ^ -(v & 1)


def pbf_bbox(path):
  """(south, north, west, east) from the OSMHeader block of a .osm.pbf, or None if it has no bounding box."""
  with open(path, "rb") as f:
    header = f.read(struct.unpack(">I", f.read(4))[0])
    size = dict(fields(header))[3]
    blob = dict(fields(f.read(size)))
  data = zlib.decompress(blob[3]) if 3 in blob else blob[1]
  box = dict(fields(data)).get(1)
  if box is None:
    return None
  box = {k: unzigzag(v) * 1e-9 for k, v in fields(box)}  # 1 left, 2 right, 3 top, 4 bottom
  return box[4], box[3], box[1], box[2]


def osm_xml_bbox(path):
  lat, lon = [], []
  with open(path, encoding="utf-8") as f:
    for line in f:
      if "<node " in line:
        la = re.search(r"""\blat=['"]([-0-9.]+)['"]""", line)
        lo = re.search(r"""\blon=['"]([-0-9.]+)['"]""", line)
        if la and lo:
          lat.append(float(la.group(1)))
          lon.append(float(lo.group(1)))
  return min(lat), max(lat), min(lon), max(lon)


def write_config(out):
  from valhalla import get_config
  cfg = get_config(tile_extract="", tile_dir=out + "/tiles", verbose=True)
  # Driving only: the car and the sim never route on foot or by bike, and those layers roughly double the tiles.
  cfg["mjolnir"].update(tile_extract=out + "/tiles.tar", admin=out + "/admin.sqlite", timezone="", traffic_extract="",
                        landmarks="", transit_dir="", transit_feeds_dir="", include_pedestrian=False, include_bicycle=False)
  cfg["additional_data"]["elevation"] = ""
  with open(out + "/valhalla.json", "w") as f:
    json.dump(cfg, f, indent=2)


def has_country(path):
  db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
  return db.execute("select count(*) from admins where admin_level = 2").fetchone()[0] > 0


def write_country(src, dst, iso):
  s, n, w, e = pbf_bbox(src) if src.endswith(".pbf") else osm_xml_bbox(src)
  corners = [(s - PAD, w - PAD), (s - PAD, e + PAD), (n + PAD, e + PAD), (n + PAD, w - PAD)]
  tags = [("type", "boundary"), ("boundary", "administrative"), ("admin_level", "2"), ("name", iso),
          ("ISO3166-1", iso), ("ISO3166-1:alpha2", iso)]
  with open(dst, "w", encoding="utf-8") as f:
    f.write("<?xml version='1.0' encoding='UTF-8'?>\n<osm version='0.6' generator='turnpilot'>\n")
    for i, (la, lo) in enumerate(corners):
      f.write(f"<node id='{9000000001 + i}' version='1' lat='{la:.7f}' lon='{lo:.7f}'/>\n")
    f.write("<way id='9000000101' version='1'>" + "".join(f"<nd ref='{9000000001 + i}'/>" for i in (0, 1, 2, 3, 0)) + "</way>\n")
    f.write("<relation id='9000000201' version='1'><member type='way' ref='9000000101' role='outer'/>")
    f.write("".join(f"<tag k='{k}' v='{v}'/>" for k, v in tags))
    f.write("</relation>\n</osm>\n")
  print(f"{iso} polygon around lat {s:.3f}..{n:.3f} lon {w:.3f}..{e:.3f}")


if __name__ == "__main__":
  cmd, args = sys.argv[1], sys.argv[2:]
  if cmd == "config":
    write_config(*args)
  elif cmd == "has_country":
    sys.exit(0 if has_country(*args) else 1)
  elif cmd == "country":
    write_country(*args)
  else:
    sys.exit(__doc__)
