"""ControlTracker: the route's stop signs, yields and signals between navd's 1 Hz fixes (control_tracker.py)."""
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.control_tracker import (CORRECTION_M, DROP_PAST_M, GAP_MAX_M, JUMP_M, ControlTracker)
from openpilot.selfdrive.turnpilot.tests.synthetic import nav_instruction


def nav(controls, route_count=1, status='ok', valid=True):
  return nav_instruction(500., route_count=route_count, controls=controls, controls_status=status, valid=valid)


def step(tr, msg=None, v=10., valid=True, n=1):
  """n frames, msg arriving on the first; the controls after the last. tr.lost_all: every control lost on the way."""
  out, tr.lost_all = None, []
  for i in range(n):
    out = tr.update(msg if i == 0 else None, msg is not None and i == 0, valid, v, DT_MDL)
    tr.lost_all += tr.lost
  return out


def test_dead_reckons_between_fixes():
  tr = ControlTracker()
  c, = step(tr, nav([(3, 'stop', 100.)]), v=0.)
  assert c.key == (1, 3) and c.kind == 'stop' and c.d == 100.
  step(tr, v=10., n=20)  # a second at 10 m/s, no fix
  assert abs(tr.controls[(1, 3)].d - 90.) < 1e-6


def test_fix_pulls_at_most_half_a_meter():
  tr = ControlTracker()
  step(tr, nav([(0, 'signal', 100.)]), v=0.)
  step(tr, nav([(0, 'signal', 104.)]), v=10.)  # dead-reckoned 99.5, measured 104: +0.5 only
  assert abs(tr.controls[(1, 0)].d - (100. - 10. * DT_MDL + CORRECTION_M)) < 1e-6
  step(tr, nav([(0, 'signal', 80.)]), v=10.)
  assert abs(tr.controls[(1, 0)].d - (100. - 20. * DT_MDL + CORRECTION_M - CORRECTION_M)) < 1e-6


def test_no_correction_while_slow():
  tr = ControlTracker()
  step(tr, nav([(0, 'stop', 4.)]), v=0.)
  step(tr, nav([(0, 'stop', 7.)]), v=0.2)  # standstill GPS noise moves nothing
  assert abs(tr.controls[(1, 0)].d - (4. - 0.2 * DT_MDL)) < 1e-6


def test_jump_reinitialises_and_flags():
  tr = ControlTracker()
  step(tr, nav([(0, 'stop', 100.)]), v=0.)
  c, = step(tr, nav([(0, 'stop', 100. + JUMP_M + 5.)]), v=10.)
  assert c.jumped and c.d == 100. + JUMP_M + 5.
  c, = step(tr, v=10.)
  assert not c.jumped


def test_held_ignores_fixes():
  tr = ControlTracker()
  step(tr, nav([(0, 'stop', 2.)]), v=0.)
  tr.controls[(1, 0)].held = True
  step(tr, nav([(0, 'stop', 60.)]), v=1.)
  assert tr.controls[(1, 0)].d == 2.  # frozen: the latch is position-free


def test_new_route_drops_unengaged_and_carries_engaged():
  tr = ControlTracker()
  step(tr, nav([(0, 'stop', 50.), (1, 'signal', 300.)]), v=0.)
  tr.controls[(1, 0)].engaged = True
  tr.controls[(1, 0)].state = 'mine'
  # a reroute: the same stop sign 12 m off (adopted, keeps the old state and distance), the signal gone
  out = step(tr, nav([(4, 'stop', 62.), (5, 'yield', 200.)], route_count=2), v=0.)
  assert [c.key for c in out] == [(2, 4), (2, 5)]
  assert out[0].engaged and out[0].state == 'mine' and out[0].d == 50.
  # a reroute that doesn't list it: carried under its old key
  out = step(tr, nav([(0, 'yield', 200.)], route_count=3), v=0.)
  assert [c.key for c in out] == [(2, 4), (3, 0)]


def test_dropped_once_past():
  tr = ControlTracker()
  step(tr, nav([(0, 'yield', 1.)]), v=0.)
  step(tr, v=10., n=int((DROP_PAST_M + 1.) / 10. / DT_MDL) + 1)
  assert not tr.controls


def test_unengaged_out_of_window_dropped_engaged_kept():
  tr = ControlTracker()
  step(tr, nav([(0, 'stop', 50.), (1, 'signal', 90.)]), v=0.)
  tr.controls[(1, 0)].engaged = True
  step(tr, nav([]), v=0.)
  assert list(tr.controls) == [(1, 0)]


def test_gap_dead_reckons_then_drops_but_never_the_held():
  tr = ControlTracker()
  step(tr, nav([(0, 'stop', 100.), (1, 'signal', 140.), (2, 'stop', 300.)]), v=0.)
  tr.controls[(1, 0)].held = True
  step(tr, nav([], valid=False), v=10., valid=False)
  step(tr, v=10., n=int(GAP_MAX_M / 10. / DT_MDL) - 5)
  assert len(tr.controls) == 3 and not tr.lost_all
  step(tr, v=10., n=10)
  assert list(tr.controls) == [(1, 0)] and {c.key for c in tr.lost_all} == {(1, 1), (1, 2)}


def test_stale_nav_is_a_gap():
  tr = ControlTracker()
  step(tr, nav([(0, 'stop', 300.)]), v=0.)
  step(tr, v=10., n=int(2.9 / DT_MDL))  # navd quiet for under 3 s: dead reckoning, not yet a gap
  assert tr.gap_m == 0.
  step(tr, v=10., n=int(GAP_MAX_M / 10. / DT_MDL) + 5)
  assert not tr.controls and [c.key for c in tr.lost_all] == [(1, 0)]


def test_status():
  tr = ControlTracker()
  assert tr.status == 'none' and not tr.route
  step(tr, nav([], status='unsupported'), v=0.)
  assert tr.status == 'unsupported' and tr.route
  step(tr, nav([], valid=False), v=0., valid=False)
  assert not tr.route  # off the route: no status shown for it
