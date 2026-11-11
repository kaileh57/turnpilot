#!/usr/bin/env python3
"""What a vehicle-CAN tap achieved for the car's turn signals, from one route's rlogs.

  blinker_bus_check.py ROUTE [--logs-dir DIR]

Reads `can`, `sendcan` and `carState` from every segment of ROUTE (counter--hash) under DIR/<device>/ROUTE--<n>/rlog* and
prints:

1. frames and distinct ids per source: the buses 0 (party CAN, car side), 1 (vehicle CAN: the tap) and 2 (party CAN,
   Autopilot side), the panda's echoes of what it sent there (128-130) and the frames it refused (192-194);
2. for DAS_bodyControls (0x3E9), SCCM_leftStalk (0x249), VCFRONT_lighting (0x3F5) and UI_warning (0x311): the buses that
   carry each, its rate there between its first and last frame, and how long that was;
3. whether 0x3E9 on bus 1 is the bus-2 frame relayed (same counter, same bytes) or the Autopilot computer's own;
4. each time our own turn request went out on bus 1 (sendcan 0x3E9 to bus 1 with DAS_turnIndicatorRequest left or
   right): frames, seconds, echoes (src 129), refusals (src 193), and what the lamp on that side did from the first frame
   to LAMP_AFTER_S after the last, with a verdict: lit, three flashes, nothing;
5. one summary line: bus 1 live or not, the lamp path proven or not.

Decoding. Every signal is read through the DBCs in opendbc/dbc, nothing by hand:
- tesla_model3_party.dbc: DAS_bodyControls (BO_ 1001: DAS_turnIndicatorRequest 8|3, DAS_turnIndicatorRequestReason 33|5,
  DAS_bodyControlsCounter 52|4), UI_warning (BO_ 785: leftBlinkerOn 22|1, rightBlinkerOn 23|1, leftBlinkerBlinking 25|2,
  rightBlinkerBlinking 26|2; blinking 2 is the lit phase of a blink, 1 its dark phase), SCCM_leftStalk (BO_ 585:
  SCCM_turnIndicatorStalkStatus 16|3, 0 idle).
- tesla_model3_vehicle.dbc: ID3F5VCFRONT_lighting (BO_ 1013: VCFRONT_turnSignalLeftStatus 50|2,
  VCFRONT_turnSignalRightStatus 52|2; 1 is LIGHT_ON, the bulb itself).
The source offsets (128 + bus for a sent frame's echo, 192 + bus for a refused one) are pandad's, as
opendbc/car/tesla/car_signal.py reads them (SRC_ECHO, SRC_REJECTED).

The lamp is read from VCFRONT_lighting on bus 1 where the route has it, else UI_warning on bus 0, else carState's
blinkers (a level with no flashes to count). The output has route seconds, bus numbers, CAN ids and counts, and the bytes
of DAS_bodyControls only: no positions, no VIN, no device id and no log paths.
"""
import argparse
import bisect
import glob
import os
import re
import sys
from collections import Counter, defaultdict

from opendbc.can.dbc import DBC

BODY, STALK, LIGHTING, UI_WARNING = 0x3E9, 0x249, 0x3F5, 0x311
NAMES = {BODY: 'DAS_bodyControls', STALK: 'SCCM_leftStalk', LIGHTING: 'VCFRONT_lighting', UI_WARNING: 'UI_warning'}
PARTY, VEHICLE, AUTOPILOT = 0, 1, 2
SRC_ECHO, SRC_REFUSED = 128, 192
SOURCES = (0, 1, 2, 128, 129, 130, 192, 193, 194)
SOURCE_NAMES = {0: 'party, car side', 1: 'vehicle CAN (the tap)', 2: 'party, Autopilot side', 128: 'sent to 0', 129: 'sent to 1',
                130: 'sent to 2', 192: 'refused for 0', 193: 'refused for 1', 194: 'refused for 2'}
REQ_LEFT, REQ_RIGHT = 1, 2  # DAS_TURN_INDICATOR_LEFT, _RIGHT
SIDES = {REQ_LEFT: 'left', REQ_RIGHT: 'right'}
LIGHT_ON, BLINK_LIT = 1, 2

ATTEMPT_GAP_S = 1.5   # request frames of one side this close together are one attempt (the stock frame is 2 Hz)
LAMP_AFTER_S = 3.     # the lamp is watched from an attempt's first frame to this long after its last
TELL_MIN_S = 3.       # an attempt shorter than this is over before a fourth flash could show: lit, but which kind is unknown
DARK_S = 1.5          # a lamp dark this long before the attempt's last frame, the requests still going, has gone out
MATCH_S = 0.5         # an echo or a refusal within this of the frame it answers; a bus-2 frame within this of a bus-1 one
RELAYED_MIN = 0.9     # this share of bus-1 0x3E9 frames byte-identical to their bus-2 twin: relayed


def _signals():
  party, vehicle = DBC('tesla_model3_party'), DBC('tesla_model3_vehicle')
  body, ui, stalk, light = party.msgs[BODY].sigs, party.msgs[UI_WARNING].sigs, party.msgs[STALK].sigs, vehicle.msgs[LIGHTING].sigs
  return {
    'request': body['DAS_turnIndicatorRequest'], 'reason': body['DAS_turnIndicatorRequestReason'],
    'counter': body['DAS_bodyControlsCounter'],
    'ui_on': {'left': ui['leftBlinkerOn'], 'right': ui['rightBlinkerOn']},
    'ui_blink': {'left': ui['leftBlinkerBlinking'], 'right': ui['rightBlinkerBlinking']},
    'stalk': stalk['SCCM_turnIndicatorStalkStatus'],
    'bulb': {'left': light['VCFRONT_turnSignalLeftStatus'], 'right': light['VCFRONT_turnSignalRightStatus']},
  }


class Level:
  """A boolean over time, stored as its changes: [(t, value)]."""
  def __init__(self):
    self.changes: list[tuple[float, bool]] = []
    self.samples = 0

  def add(self, t: float, value: bool) -> None:
    self.samples += 1
    if not self.changes or self.changes[-1][1] != value:
      self.changes.append((t, value))

  def window(self, t0: float, t1: float) -> tuple[int, float, float | None, bool]:
    """(rising edges, seconds true, the last time true, any sample) in [t0, t1]. A level already true at t0 counts as
    its first edge."""
    if not self.changes or self.changes[0][0] > t1:
      return 0, 0., None, False
    i = bisect.bisect_right(self.changes, (t0, True)) - 1
    value = self.changes[i][1] if i >= 0 else False
    edges, lit, last, t = int(value), 0., t0 if value else None, t0
    for tc, v in self.changes[max(i + 1, 0):]:
      if tc > t1:
        break
      if value:
        lit, last = lit + tc - t, tc
      edges += int(v and not value)
      value, t = v, tc
    if value:
      lit, last = lit + t1 - t, t1
    return edges, lit, last, True


class BusCheck:
  """Feed it the route's messages in any order of kinds, then report(). Times are seconds on one clock."""
  def __init__(self):
    self.sig = _signals()
    self.count: Counter = Counter()                       # (src, address) -> frames
    self.seen: dict[tuple[int, int], list[float]] = {}    # (src, one of the four ids) -> [first, last] time
    self.t0: float | None = None
    self.t1 = 0.
    self.body: dict[int, list] = defaultdict(list)        # src -> [(t, dat)] of 0x3E9
    self.sent: list[tuple[float, int, bytes]] = []        # sendcan 0x3E9: (t, bus, dat)
    self.lamps = {src: {side: Level() for side in ('left', 'right')} for src in ('vcfront', 'ui', 'carState')}
    self.stalk = Level()                                  # the stalk off idle, any bus

  def _clock(self, t: float) -> None:
    self.t0 = t if self.t0 is None else min(self.t0, t)
    self.t1 = max(self.t1, t)

  def can(self, t: float, src: int, address: int, dat: bytes) -> None:
    self._clock(t)
    self.count[(src, address)] += 1
    if address in NAMES:
      span = self.seen.setdefault((src, address), [t, t])
      span[0], span[1] = min(span[0], t), max(span[1], t)
    if address == BODY and len(dat) == 8:
      self.body[src].append((t, bytes(dat)))
    elif address == LIGHTING and src == VEHICLE and len(dat) == 8:
      for side, sig in self.sig['bulb'].items():
        self.lamps['vcfront'][side].add(t, sig.get_raw_value(dat) == LIGHT_ON)
    elif address == UI_WARNING and src == PARTY and len(dat) >= 7:
      for side in ('left', 'right'):
        blink = self.sig['ui_blink'][side].get_raw_value(dat)
        self.lamps['ui'][side].add(t, blink == BLINK_LIT or (blink == 0 and self.sig['ui_on'][side].get_raw_value(dat) == 1))
    elif address == STALK and src in (PARTY, VEHICLE, AUTOPILOT) and len(dat) >= 3:
      self.stalk.add(t, self.sig['stalk'].get_raw_value(dat) != 0)

  def sendcan(self, t: float, bus: int, address: int, dat: bytes) -> None:
    self._clock(t)
    if address == BODY and len(dat) == 8:
      self.sent.append((t, bus, bytes(dat)))

  def car_state(self, t: float, left: bool, right: bool) -> None:
    self._clock(t)
    self.lamps['carState']['left'].add(t, bool(left))
    self.lamps['carState']['right'].add(t, bool(right))

  # the report

  def _span(self) -> float:
    return max(self.t1 - (self.t0 or 0.), 1e-9)

  def _buses(self) -> list[dict]:
    return [{'src': src, 'frames': sum(n for (s, _), n in self.count.items() if s == src),
             'ids': len({a for (s, a) in self.count if s == src})} for src in SOURCES]

  def _carriers(self) -> dict[int, list[tuple[int, int, float, float]]]:
    """id -> [(src, frames, Hz while present, seconds present)]: the rate between a source's first and last frame of it, so
    a burst reads as its own rate over a short presence, not as a slow message."""
    out = {}
    for addr in NAMES:
      out[addr] = []
      for src in SOURCES:
        n = self.count[(src, addr)]
        if n:
          present = self.seen[(src, addr)][1] - self.seen[(src, addr)][0]
          out[addr].append((src, n, (n - 1) / present if present > 0. else 0., present))
    return out

  def _relay(self) -> dict:
    """Bus 1's 0x3E9 against bus 2's: each bus-1 frame's twin is the bus-2 frame with its counter within MATCH_S."""
    one, two = self.body[VEHICLE], sorted(self.body[AUTOPILOT])
    times = [t for t, _ in two]
    counter = self.sig['counter'].get_raw_value
    same = differs = alone = 0
    example = None
    for t, dat in one:
      lo, hi = bisect.bisect_left(times, t - MATCH_S), bisect.bisect_right(times, t + MATCH_S)
      twins = [d for _, d in two[lo:hi] if counter(d) == counter(dat)]
      if not twins:
        alone += 1
      elif dat in twins:
        same += 1
      else:
        differs += 1
        example = example or (dat, twins[0])
    n = len(one)
    verdict = 'absent' if not n else 'relayed' if same >= RELAYED_MIN * n else 'own'
    return {'frames': n, 'same': same, 'differs': differs, 'alone': alone, 'verdict': verdict, 'example': example}

  def _lamp_source(self) -> str | None:
    return next((src for src in ('vcfront', 'ui', 'carState') if any(lv.samples for lv in self.lamps[src].values())), None)

  def _answers(self, src: int, frames: list[tuple[float, bytes]]) -> int:
    """How many of an attempt's frames came back on src (an echo, a refusal): the same bytes within MATCH_S after."""
    back = sorted(self.body[src])
    times, n = [t for t, _ in back], 0
    for t, dat in frames:
      lo, hi = bisect.bisect_left(times, t - 1e-6), bisect.bisect_right(times, t + MATCH_S)
      n += any(d == dat for _, d in back[lo:hi])
    return n

  def _attempts(self) -> list[dict]:
    request = self.sig['request'].get_raw_value
    groups: list[tuple[str, list]] = []
    for t, bus, dat in sorted(self.sent):
      side = SIDES.get(request(dat))
      if bus != VEHICLE or side is None:
        continue
      if groups and groups[-1][0] == side and t - groups[-1][1][-1][0] <= ATTEMPT_GAP_S:
        groups[-1][1].append((t, dat))
      else:
        groups.append((side, [(t, dat)]))
    source, out = self._lamp_source(), []
    for side, frames in groups:
      first, last = frames[0][0], frames[-1][0]
      a = {'side': side, 't': first - (self.t0 or 0.), 'frames': len(frames), 'seconds': last - first,
           'reasons': sorted({self.sig['reason'].get_raw_value(d) for _, d in frames}),
           'echoed': self._answers(SRC_ECHO + VEHICLE, frames), 'refused': self._answers(SRC_REFUSED + VEHICLE, frames),
           'lamp_source': source, 'flashes': 0, 'lit_s': 0., 'stalk': False}
      if source is not None:
        a['flashes'], a['lit_s'], last_lit, seen = self.lamps[source][side].window(first, last + LAMP_AFTER_S)
        a['stalk'] = self.stalk.window(first, last + LAMP_AFTER_S)[0] > 0
        a['verdict'] = lamp_verdict(source, a['flashes'], a['lit_s'], last_lit, first, last) if seen else 'no lamp reading'
      else:
        a['verdict'] = 'no lamp reading'
      out.append(a)
    return out

  def report(self) -> dict:
    request = self.sig['request'].get_raw_value
    attempts = self._attempts()
    bus1 = sum(n for (s, _), n in self.count.items() if s == VEHICLE)
    clean = [a for a in attempts if not a['stalk'] and a['echoed'] and a['lamp_source'] in ('vcfront', 'ui', 'carState')]
    held = any(a['verdict'] == 'lit' for a in clean)
    flashed = any(a['verdict'].startswith(('three flashes', 'lit', 'went out')) for a in clean)
    return {
      'seconds': self._span() if self.t0 is not None else 0., 'buses': self._buses(), 'carriers': self._carriers(),
      'relay': self._relay(), 'attempts': attempts,
      'party_requests': sum(1 for _, bus, dat in self.sent if bus == PARTY and request(dat) in SIDES),
      'lamp_source': self._lamp_source(), 'bus1_live': bus1 > 0, 'bus1_frames': bus1,
      'lamp_path': 'held' if held else 'flashes' if flashed else 'unproven',
    }


def lamp_verdict(source: str, flashes: int, lit_s: float, last_lit: float | None, first: float, last: float) -> str:
  """One attempt's lamp, from its first request frame to LAMP_AFTER_S after its last (first, last: those frames' times)."""
  if not flashes or last_lit is None:
    return 'nothing'
  if source == 'carState':  # a level, held through a blink's dark phase: no flashes to count
    return 'lit' if last - last_lit <= DARK_S else 'went out early (carState only: flashes not countable)'
  if last - first < TELL_MIN_S:
    return f'lit, too short to tell ({flashes} flashes in a {last - first:.1f} s attempt)'
  if last - last_lit > DARK_S:
    return 'three flashes' if flashes == 3 else f'went out after {flashes} flashes'
  return 'lit'


def render(route: str, r: dict) -> list[str]:
  out = [f'blinker bus check: route {route}, {r["seconds"]:.0f} s of CAN', '', 'frames per source:']
  for b in r['buses']:
    out.append(f'  src {b["src"]:>3} ({SOURCE_NAMES[b["src"]]}): {b["frames"]} frames, {b["ids"]} ids')
  out += ['', 'the four messages:']
  for addr, name in NAMES.items():
    on = ', '.join(f'src {src} at {hz:.1f} Hz ({n} frames over {present:.0f} s)' for src, n, hz, present in r['carriers'][addr]) \
      or 'on no bus'
    out.append(f'  0x{addr:03X} {name}: {on}')
  rel = r['relay']
  out.append('')
  if rel['verdict'] == 'absent':
    out.append('0x3E9 on bus 1: none, so nothing to compare with bus 2')
  else:
    what = 'relayed: byte-identical to the bus-2 frame of the same counter' if rel['verdict'] == 'relayed' else \
      "the Autopilot computer's own: it differs from bus 2"
    counts = f'{rel["same"]} identical, {rel["differs"]} differing, {rel["alone"]} with no bus-2 frame of that counter'
    out.append(f'0x3E9 on bus 1: {what} ({counts}, of {rel["frames"]})')
    if rel['example'] is not None:
      out.append(f'  first differing pair: bus 1 {rel["example"][0].hex()}  bus 2 {rel["example"][1].hex()}')
  out.append('')
  source = {'vcfront': 'VCFRONT_lighting 0x3F5 on bus 1', 'ui': 'UI_warning 0x311 on bus 0', 'carState': "carState's blinkers",
            None: 'nothing'}[r['lamp_source']]
  out.append(f'our turn requests on bus 1: {len(r["attempts"])} (lamp read from {source})')
  for i, a in enumerate(r['attempts'], 1):
    stalk = "; the driver's stalk moved in the window, so the lamp may be his" if a['stalk'] else ''
    sent = f'{a["frames"]} frames over {a["seconds"]:.1f} s, reason {"/".join(str(x) for x in a["reasons"])}'
    back = f'{a["echoed"]} echoed, {a["refused"]} refused'
    lamp = f'lamp {a["flashes"]} flashes, {a["lit_s"]:.1f} s lit: {a["verdict"]}{stalk}'
    out.append(f'  {i}. {a["t"]:.1f} s {a["side"]}: {sent}, {back}; {lamp}')
  if r['party_requests']:
    out.append(f'  ({r["party_requests"]} request frames went to bus 0, the party bus, which has not lit this car\'s lamps)')
  live = f'bus 1 live ({r["bus1_frames"]} frames)' if r['bus1_live'] else 'bus 1 silent (0 frames): no vehicle-CAN tap seen'
  path = {'held': 'lamp path proven: a request on bus 1 held the lamp',
          'flashes': 'lamp path proven for a tap only: a request on bus 1 flashed the lamp but did not hold it',
          'unproven': 'lamp path not proven'}[r['lamp_path']]
  out += ['', f'summary: {live}; {path}']
  return out


def segments(logs_dir: str, route: str) -> list[tuple[int, str]]:
  """[(segment, rlog path)] of route under logs_dir/<device>/, in order."""
  found = {}
  for path in glob.glob(os.path.join(logs_dir, '*', f'{route}--*', 'rlog*')):
    m = re.fullmatch(re.escape(route) + r'--(\d+)', os.path.basename(os.path.dirname(path)))
    if m:
      found.setdefault(int(m.group(1)), path)
  return sorted(found.items())


def read_route(logs_dir: str, route: str) -> tuple[BusCheck, list[str]]:
  from openpilot.tools.lib.logreader import LogReader
  check, notes = BusCheck(), []
  segs = segments(logs_dir, route)
  for seg, path in segs:
    try:
      for m in LogReader(path, only_union_types=True):
        which, t = m.which(), m.logMonoTime * 1e-9
        if which == 'can':
          for c in m.can:
            check.can(t, c.src, c.address, c.dat)
        elif which == 'sendcan':
          for c in m.sendcan:
            check.sendcan(t, c.src, c.address, c.dat)
        elif which == 'carState':
          check.car_state(t, m.carState.leftBlinker, m.carState.rightBlinker)
    except Exception as e:  # the path names the device: never printed
      notes.append(f'segment {seg}: unreadable ({type(e).__name__})')
  notes.insert(0, f'{len(segs)} segments' + (f' ({segs[0][0]}-{segs[-1][0]})' if segs else ''))
  return check, notes


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('route', help='route name: counter--hash')
  ap.add_argument('--logs-dir', default=os.path.expanduser('~/turnpilot_data/logs'))
  args = ap.parse_args(argv)
  if not re.fullmatch(r'[0-9a-f]{8}--[0-9a-f]{10}', args.route):
    print('route must be counter--hash, with no device id', file=sys.stderr)
    return 2
  check, notes = read_route(args.logs_dir, args.route)
  if check.t0 is None:
    print(f'no can, sendcan or carState read for route {args.route} ({"; ".join(notes)})', file=sys.stderr)
    return 1
  lines = render(args.route, check.report())
  lines.insert(1, f'read: {"; ".join(notes)}')
  print('\n'.join(lines))
  return 0


if __name__ == '__main__':
  sys.exit(main())
