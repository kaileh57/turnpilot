"""blinker_bus_check's verdicts, on synthetic CAN: a silent bus 1, 0x3E9 relayed or the Autopilot computer's own, and a
request on bus 1 that holds the lamp, gives three flashes, or lights nothing."""
import pytest
from opendbc.can.packer import CANPacker
from opendbc.car.tesla.teslacan import body_controls_checksum, create_body_controls
from openpilot.tools.turnpilot import blinker_bus_check as bc

PARTY_PACKER, VEHICLE_PACKER = CANPacker('tesla_model3_party'), CANPacker('tesla_model3_vehicle')
STOCK_HZ, FLASH_S, LIT_S = 2., 0.7, 0.35  # the stock frame's rate; the lamp's period and lit phase (route logs: 0.7 s, 0.3-0.4 s)
REASON = 8


def stock(counter, byte0=0x01):
  dat = bytearray(8)
  dat[0], dat[6] = byte0, (counter & 0xF) << 4
  dat[7] = body_controls_checksum(dat)
  return bytes(dat)


def lighting(left=False, right=False):
  return VEHICLE_PACKER.make_can_msg('ID3F5VCFRONT_lighting', 1, {'VCFRONT_turnSignalLeftStatus': int(left),
                                                                   'VCFRONT_turnSignalRightStatus': int(right)})[1]


def ui_warning(left_blink=0, right_blink=0):
  return PARTY_PACKER.make_can_msg('UI_warning', 0, {'leftBlinkerBlinking': left_blink, 'rightBlinkerBlinking': right_blink,
                                                     'leftBlinkerOn': int(left_blink == 2), 'rightBlinkerOn': int(right_blink == 2)})[1]


def flashing(t, start, flashes):
  """The lamp's lit phase at t for a blink that starts at start and makes this many flashes (None: until the end)."""
  n, phase = divmod(t - start, FLASH_S)
  return t >= start and phase < LIT_S and (flashes is None or n < flashes)


def drive(seconds=30., tap='relayed', request=(10., 20., bc.REQ_LEFT), flashes=None, lamp='vcfront', echo=True, refuse=False,
          stalk_at=None):
  """A BusCheck fed `seconds` of a car: party traffic on 0 and 2, vehicle CAN on 1 when tap ('relayed': 0x3E9 there is the
  bus-2 frame; 'own': its own bytes; None: no tap), our request copied onto bus 1 over request = (from, to, side), the
  side's lamp making `flashes` flashes from its first frame (None: for as long as asked; 0: never), reported by `lamp`."""
  check = bc.BusCheck()
  t0, steps = 1000., int(seconds * 100)
  for i in range(steps):
    t, rel = t0 + i / 100., i / 100.
    asked = request is not None and request[0] <= rel < request[1]
    side = 'left' if request is None or request[2] == bc.REQ_LEFT else 'right'
    lit = flashes != 0 and request is not None and flashing(rel, request[0], flashes) and (flashes is not None or rel < request[1] + 0.5)
    if i % 10 == 0:  # 10 Hz
      check.can(t, 0, 0x118, bytes(8))
      check.can(t, 130, 0x118, bytes(8))
      blink = (2 if lit else 1) if flashes != 0 and asked else 2 if lit else 0
      if lamp in ('vcfront', 'ui'):
        check.can(t, 0, bc.UI_WARNING, ui_warning(**{f'{side}_blink': blink}) if lamp == 'ui' else ui_warning())
      if tap:
        check.can(t, 1, bc.STALK, bytes([0, 0, 1 if stalk_at is not None and stalk_at <= rel < stalk_at + 1. else 0]))
        if lamp == 'vcfront':
          check.can(t, 1, bc.LIGHTING, lighting(**{side: lit}))
      check.car_state(t, lit and side == 'left' and lamp == 'carState', lit and side == 'right' and lamp == 'carState')
    if i % int(100 / STOCK_HZ) == 0:
      n = i // int(100 / STOCK_HZ)
      frame = stock(n)
      check.can(t, 2, bc.BODY, frame)
      check.can(t + 0.001, 128, bc.BODY, frame)  # the panda's forward to the car side
      if tap:
        check.can(t + 0.002, 1, bc.BODY, frame if tap == 'relayed' else stock(n, byte0=0x05))
        if asked:
          ours = create_body_controls(frame if tap == 'relayed' else stock(n, byte0=0x05), request[2], REASON)
          check.sendcan(t + 0.01, 1, bc.BODY, ours)
          if refuse:
            check.can(t + 0.012, 193, bc.BODY, ours)
          elif echo:
            check.can(t + 0.012, 129, bc.BODY, ours)
  return check


def text(check):
  return '\n'.join(bc.render('00000000--0000000000', check.report()))


def test_a_silent_bus_1_is_said_and_proves_nothing():
  r = drive(tap=None, request=None, lamp='ui').report()
  by_src = {b['src']: b for b in r['buses']}
  assert by_src[1] == {'src': 1, 'frames': 0, 'ids': 0} and by_src[0]['frames'] == 600 and by_src[0]['ids'] == 2
  assert not r['bus1_live'] and r['relay']['verdict'] == 'absent' and not r['attempts'] and r['lamp_path'] == 'unproven'
  carriers = {addr: [(src, round(hz, 1)) for src, _, hz, _ in on] for addr, on in r['carriers'].items()}
  assert carriers == {bc.BODY: [(2, 2.), (128, 2.)], bc.STALK: [], bc.LIGHTING: [], bc.UI_WARNING: [(0, 10.)]}
  # a burst is its own rate over a short presence, not a slow message
  check = drive(tap=None, request=None, lamp='ui')
  for i in range(5):
    check.can(1010. + i * 0.5, 0, bc.BODY, stock(i))
  (src, n, hz, present), *_ = check.report()['carriers'][bc.BODY]
  assert (src, n, round(hz, 1), round(present, 1)) == (0, 5, 2., 2.)
  out = text(drive(tap=None, request=None, lamp='ui'))
  assert 'summary: bus 1 silent (0 frames): no vehicle-CAN tap seen; lamp path not proven' in out
  assert '0x249 SCCM_leftStalk: on no bus' in out and '0x3E9 on bus 1: none' in out


def test_a_request_that_holds_the_lamp_is_lit_and_proves_the_path():
  check = drive()
  r = check.report()
  assert r['bus1_live'] and r['lamp_source'] == 'vcfront' and r['relay']['verdict'] == 'relayed' and r['relay']['differs'] == 0
  (a,) = r['attempts']
  assert (a['side'], a['frames'], a['echoed'], a['refused'], a['reasons']) == ('left', 20, 20, 0, [REASON])
  assert a['t'] == pytest.approx(10.01) and a['seconds'] == pytest.approx(9.5)
  assert a['flashes'] >= 13 and a['verdict'] == 'lit' and r['lamp_path'] == 'held'
  out = text(check)
  assert 'relayed: byte-identical to the bus-2 frame of the same counter' in out
  assert 'summary: bus 1 live (' in out and 'lamp path proven: a request on bus 1 held the lamp' in out
  # the right side, read from UI_warning where vehicle CAN has no lighting frame
  r = drive(request=(5., 12., bc.REQ_RIGHT), lamp='ui').report()
  assert r['lamp_source'] == 'ui' and [(a['side'], a['verdict']) for a in r['attempts']] == [('right', 'lit')]
  # ...and from carState's blinkers, a level, where neither is there
  r = drive(lamp='carState').report()
  assert r['lamp_source'] == 'carState' and r['attempts'][0]['verdict'] == 'lit' and r['lamp_path'] == 'held'


def test_three_flashes_then_dark_is_a_tap_not_a_hold():
  check = drive(flashes=3, tap='own')
  r = check.report()
  (a,) = r['attempts']
  assert a['flashes'] == 3 and a['verdict'] == 'three flashes' and r['lamp_path'] == 'flashes'
  assert r['relay']['verdict'] == 'own' and r['relay']['same'] == 0 and r['relay']['differs'] == r['relay']['frames'] == 60
  out = text(check)
  assert "the Autopilot computer's own: it differs from bus 2" in out and 'first differing pair: bus 1 05' in out
  assert 'lamp path proven for a tap only' in out
  assert drive(flashes=5).report()['attempts'][0]['verdict'] == 'went out after 5 flashes'


def test_no_lamp_is_nothing_whether_echoed_or_refused():
  r = drive(flashes=0).report()
  assert [(a['verdict'], a['echoed'], a['refused']) for a in r['attempts']] == [('nothing', 20, 0)] and r['lamp_path'] == 'unproven'
  r = drive(flashes=0, refuse=True).report()
  assert [(a['verdict'], a['echoed'], a['refused']) for a in r['attempts']] == [('nothing', 0, 20)] and r['lamp_path'] == 'unproven'
  assert 'lamp path not proven' in text(drive(flashes=0))


def test_what_is_no_proof():
  # an attempt over before a fourth flash could show: lit, but a tap and a hold look the same
  r = drive(request=(10., 11.6, bc.REQ_LEFT), flashes=3).report()
  assert r['attempts'][0]['verdict'].startswith('lit, too short to tell (3 flashes')
  assert r['lamp_path'] == 'flashes', "the lamp answered bus 1, held or not unknown"
  # the driver's stalk moved in the window: the lamp may be his
  check = drive(stalk_at=12.)
  r = check.report()
  assert r['attempts'][0]['stalk'] and r['attempts'][0]['verdict'] == 'lit' and r['lamp_path'] == 'unproven'
  assert "the driver's stalk moved in the window" in text(check)
  # frames that never reached the bus (no echo) prove nothing, whatever the lamp did
  assert drive(echo=False).report()['lamp_path'] == 'unproven'
  # two bursts of one side further apart than a missed stock frame are two attempts
  check = drive()
  check.sendcan(1000. + 25., 1, bc.BODY, create_body_controls(stock(50), bc.REQ_LEFT, REASON))
  assert [a['frames'] for a in check.report()['attempts']] == [20, 1]


def test_level_window():
  lv = bc.Level()
  for t, v in ((0., False), (1., True), (1.3, False), (2., True), (2.3, False), (5., True)):
    lv.add(t, v)
  edges, lit, last, seen = lv.window(0.5, 4.)
  assert (edges, round(lit, 6), round(last, 6), seen) == (2, 0.6, 2.3, True)
  assert lv.window(1.1, 1.2) == (1, pytest.approx(0.1), 1.2, True), "lit at the window's start counts as a flash"
  assert lv.window(3., 4.)[:1] == (0,) and bc.Level().window(0., 1.) == (0, 0., None, False)


def test_nothing_that_names_the_device_is_printed(tmp_path, capsys):
  device = 'f00dfacecafe1234'
  seg = tmp_path / device / '00000007--0123456789--0'
  seg.mkdir(parents=True)
  (seg / 'rlog.zst').write_bytes(b'not a log')
  assert bc.segments(str(tmp_path), '00000007--0123456789') == [(0, str(seg / 'rlog.zst'))]
  assert bc.main(['00000007--0123456789', '--logs-dir', str(tmp_path)]) == 1
  said = capsys.readouterr()
  assert device not in said.out + said.err and str(tmp_path) not in said.out + said.err
  assert 'segment 0: unreadable' in said.err
  assert bc.main([f'{device}|00000007--0123456789', '--logs-dir', str(tmp_path)]) == 2
  # a full report: route seconds, buses, ids, counts and 0x3E9 bytes only
  out = text(drive(flashes=3, tap='own'))
  assert device not in out and 'VIN' not in out and '/' not in out.replace('/s', '')
