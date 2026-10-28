"""Stop signs and traffic signals for the west_coast_usa road-graph export, from the level's own signal system.

BeamNG's core_trafficSignals keeps, for every signal instance, the map link it controls: mapNodeSignals[a][b] is a list
of {instance, pos, state, action} for traffic going a -> b, with pos the stop line (near b). The probe dumped that as
map_node_signals.json. For each (a, b, instance) this inserts one new OSM node on the a -> b segment of the way that
holds it, at the stop line projected onto the segment, and tags it the way real OSM maps a one-direction control:

  stop instance         highway=stop             direction=forward|backward
  trafficLight instance highway=traffic_signals  traffic_signals:direction=forward|backward

forward / backward is relative to the way's node order. The node is interior to exactly one way, so the direction is
never ambiguous (a direction tag on a junction node shared by several ways is). beamng:signal=<instance> lets a test
map a Valhalla edge back to the live light. Every pair it can't place is printed and counted, never guessed.

It also writes the sidecar navd's routing/controls.py reads, controls.json next to out.osm (build_sim_tiles.sh copies it
into the tiles with the controls_built marker): {"controls": {"<way_id>:<f|b>": {kind, instance, line: [lat, lon],
sign_visible}}, "skipped": [[instance, a, b, why], ...]}, keyed by the way (after the splits) that holds the control and
the direction it applies in, so a route's edge finds its control's real stop line.

  python mapsignals.py lanes.osm map_node_signals.json out.osm [--stop-props stop_props.json --visible-only]
"""
import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from openpilot.tools.sim.bridge.beamng import frame

END_CLEAR_M = 1.0  # the new node stays at least this far inside the segment
SIGN_PROP = "sign_stop.dae"
VISIBLE_M = 15.0  # a stop instance counts as signed when a sign_stop prop is this close


def load(path):
  tree = ET.parse(path)
  root = tree.getroot()
  by_name, xy, ele, ways = {}, {}, {}, []
  for el in root:
    if el.tag == "node":
      i = int(el.get("id"))
      tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
      x, y, _ = frame.from_latlon(float(el.get("lat")), float(el.get("lon")))
      xy[i], ele[i] = (x, y), float(tags.get("ele", 0.0))
      if "beamng:node" in tags:
        by_name[tags["beamng:node"]] = i
    elif el.tag == "way":
      ways.append(el)
  seg = {}  # (u, v) consecutive in a way -> (way element, index of u)
  for w in ways:
    refs = [int(nd.get("ref")) for nd in w.findall("nd")]
    for k, (u, v) in enumerate(zip(refs, refs[1:], strict=False)):
      seg[(u, v)] = (w, k)
  return tree, by_name, xy, ele, seg


def project(p, a, b):
  """Fraction along a -> b of p's projection, clamped END_CLEAR_M inside the segment, and the segment length."""
  dx, dy = b[0] - a[0], b[1] - a[1]
  L = math.hypot(dx, dy)
  t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (L * L)
  lo = min(0.5, END_CLEAR_M / L)
  return min(max(t, lo), 1 - lo), L


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("osm")
  ap.add_argument("signals", help="map_node_signals.json")
  ap.add_argument("out")
  ap.add_argument("--stop-props", help="stop_props.json, to tag only stop instances with a visible sign")
  ap.add_argument("--visible-only", action="store_true")
  ap.add_argument("--sidecar", help="where to write controls.json (default: next to out)")
  a = ap.parse_args()

  tree, by_name, xy, ele, seg = load(a.osm)
  root = tree.getroot()
  signs = []
  if a.stop_props:
    props = json.load(open(a.stop_props))
    props = props["props"] if isinstance(props, dict) else props  # wcusa_stop_props.json wraps them with their provenance
    signs = [p["pos"][:2] for p in props if SIGN_PROP in p["shape"]]
  mns = json.load(open(a.signals))
  mns = mns.get("mapNodeSignals", mns)  # wcusa_map_node_signals.json wraps the dump with its provenance

  next_id = max(xy) + 1
  inserts = defaultdict(list)  # way element -> [(index of u, fraction, node element)]
  seen, counts, skips = set(), Counter(), []
  for na, inner in mns.items():
    for nb, lst in inner.items():
      for s in lst:
        inst = s["instance"]
        kind = "stop" if inst.startswith("stop") else "signal"
        if (na, nb, inst) in seen:
          continue
        seen.add((na, nb, inst))
        ia, ib = by_name.get(na), by_name.get(nb)
        if ia is None or ib is None:
          skips.append((inst, na, nb, "map node not in the export"))
          continue
        if (ia, ib) in seg:
          way, k = seg[(ia, ib)]
          u, v, direction = ia, ib, "forward"
        elif (ib, ia) in seg:
          way, k = seg[(ib, ia)]
          u, v, direction = ib, ia, "backward"
        else:
          skips.append((inst, na, nb, "a and b are not consecutive in any way"))
          continue
        p = (s["pos"]["x"], s["pos"]["y"])
        visible = None
        if signs and kind == "stop":
          visible = min(math.dist(p, q) for q in signs) <= VISIBLE_M
          counts[f"stop with sign prop {'<=' if visible else '>'} {VISIBLE_M:.0f} m"] += 1
          if a.visible_only and not visible:
            skips.append((inst, na, nb, "no sign_stop prop nearby (--visible-only)"))
            continue
        t, L = project(p, xy[u], xy[v])
        x = xy[u][0] + t * (xy[v][0] - xy[u][0])
        y = xy[u][1] + t * (xy[v][1] - xy[u][1])
        z = ele[u] + t * (ele[v] - ele[u])
        lat, lon, _ = frame.to_latlon(x, y, 0.0)
        node = ET.Element("node", id=str(next_id), version="1", lat=f"{lat:.8f}", lon=f"{lon:.8f}")
        tags = {"ele": f"{z:.2f}", "beamng:signal": inst, "beamng:link": f"{na}>{nb}"}
        node.set("tp_direction", direction)  # removed before writing: for the sidecar
        node.set("tp_kind", kind)
        if visible is not None:
          node.set("tp_visible", "yes" if visible else "no")
        if kind == "stop":
          tags.update({"highway": "stop", "direction": direction})
        else:
          tags.update({"highway": "traffic_signals", "traffic_signals:direction": direction})
        if visible is not None:
          tags["beamng:sign_visible"] = "yes" if visible else "no"
        for k2, v2 in tags.items():
          ET.SubElement(node, "tag", k=k2, v=v2)
        next_id += 1
        inserts[way].append((k, t if direction == "forward" else t, node))
        counts[f"{kind} {direction}"] += 1
        off = math.dist(p, (x, y))
        if off > 10:
          counts[f"{kind}: stop line > 10 m off its link"] += 1

  # Put each way's new control nodes in its node list, between u and v, in order along the way. Then split any way that
  # holds two controls: Valhalla flags a whole edge, so one edge with two controls loses one (a forward and a backward
  # control on one edge came out as the forward one only). The cut goes at an existing node between them, or a new one
  # midway. turn:lanes(:forward) stays on the piece that ends at the junction, turn:lanes:backward on the one that starts.
  extra_nodes, new_ways = [], []
  next_way = max(int(w.get("id")) for w in root.iter("way")) + 1
  ctrl_ids = {n.get("id") for lst in inserts.values() for _, _, n in lst}
  pos = {n.get("id"): (float(n.get("lat")), float(n.get("lon"))) for lst in inserts.values() for _, _, n in lst}
  for el in root.iter("node"):
    pos.setdefault(el.get("id"), (float(el.get("lat")), float(el.get("lon"))))
  for way, lst in inserts.items():
    nds = way.findall("nd")
    for nd in nds:
      way.remove(nd)
    by_seg = defaultdict(list)
    for k, t, n in lst:
      by_seg[k].append((t, n))
    refs = []
    for k, nd in enumerate(nds):
      refs.append(nd.get("ref"))
      refs += [n.get("id") for _, n in sorted(by_seg.get(k, []), key=lambda e: e[0])]
    cidx = [i for i, r in enumerate(refs) if r in ctrl_ids]
    cuts = []
    for ci, cj in zip(cidx, cidx[1:], strict=False):
      if cj - ci >= 2:
        cuts.append((ci + cj) // 2)
      else:  # adjacent controls: a new node midway between them
        (la1, lo1), (la2, lo2) = pos[refs[ci]], pos[refs[cj]]
        node = ET.Element("node", id=str(next_id), version="1", lat=f"{(la1 + la2) / 2:.8f}", lon=f"{(lo1 + lo2) / 2:.8f}")
        ET.SubElement(node, "tag", k="beamng:split", v="yes")
        extra_nodes.append(node)
        next_id += 1
        refs.insert(cj, node.get("id"))
        cidx = [i + (i >= cj) for i in cidx]
        cuts.append(cj)
    pieces, s = [], 0
    for c in cuts:
      pieces.append(refs[s:c + 1])
      s = c
    pieces.append(refs[s:])
    tags = [(t.get("k"), t.get("v")) for t in way.findall("tag")]
    for j, piece in enumerate(pieces):
      target = way if j == 0 else ET.Element("way", id=str(next_way), version="1")
      if j:
        next_way += 1
        new_ways.append(target)
        for k2, v2 in tags:
          if k2 in ("turn:lanes", "turn:lanes:forward") and j != len(pieces) - 1:
            continue
          if k2 == "turn:lanes:backward":
            continue
          ET.SubElement(target, "tag", k=k2, v=v2)
      elif len(pieces) > 1:
        for t in way.findall("tag"):
          if t.get("k") in ("turn:lanes", "turn:lanes:forward"):
            way.remove(t)
      for i2, r in enumerate(piece):
        target.insert(i2, ET.Element("nd", ref=r))
    if len(pieces) > 1:
      counts["ways split so that each edge holds one control"] += 1

  first_way = next(i for i, el in enumerate(root) if el.tag == "way")
  new_nodes = sorted([n for lst in inserts.values() for _, _, n in lst] + extra_nodes, key=lambda n: int(n.get("id")))
  for j, n in enumerate(new_nodes):  # after the export's nodes, ids ascending: Valhalla needs sorted input
    root.insert(first_way + j, n)
  for w in new_ways:
    root.append(w)

  # the sidecar: each control by the way (piece) that holds it and the direction it applies in
  sidecar = {}
  holder = {}
  for w in root.iter("way"):
    for nd in w.findall("nd"):
      if nd.get("ref") in ctrl_ids:
        holder[nd.get("ref")] = w.get("id")
  for lst in inserts.values():
    for _, _, n in lst:
      key = f"{holder[n.get('id')]}:{'f' if n.get('tp_direction') == 'forward' else 'b'}"
      tags = {t.get("k"): t.get("v") for t in n.findall("tag")}
      if key in sidecar:
        counts["sidecar: two controls for one way and direction (the first kept)"] += 1
        continue
      sidecar[key] = {"kind": n.get("tp_kind"), "instance": tags["beamng:signal"], "line": [float(n.get("lat")), float(n.get("lon"))],
                      "sign_visible": None if n.get("tp_visible") is None else n.get("tp_visible") == "yes"}
      for k in ("tp_direction", "tp_kind", "tp_visible"):
        n.attrib.pop(k, None)
  side_path = Path(a.sidecar) if a.sidecar else Path(a.out).with_name("controls.json")
  side_path.write_text(json.dumps({"controls": sidecar, "skipped": [list(s) for s in skips]}, indent=1, sort_keys=True))
  tree.write(a.out, encoding="utf-8", xml_declaration=True)

  print(f"{len(ctrl_ids)} control nodes added to {len(inserts)} ways, {len(new_ways)} ways from splits -> {a.out}, " +
        f"{len(sidecar)} in the sidecar {side_path}")
  for k, v in sorted(counts.items()):
    print(f"  {v:4d}  {k}")
  print(f"skipped {len(skips)}:", file=sys.stderr)
  for s in skips:
    print("  ", *s, file=sys.stderr)


if __name__ == "__main__":
  main()
