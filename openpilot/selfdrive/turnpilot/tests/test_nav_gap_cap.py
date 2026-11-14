"""A route turn's corner cap through a gap in the route (turnpilotd.update: gap_cap). navInstruction invalid for a moment
(navd between two fixes of a reroute, a dropped step) dropped the follower's maneuver and the cap with it the same frame, so
the car sped up toward a junction it was slowing for until a stop's own curve met it. The cap in force is kept, dead-reckoned,
through a gap under the follower's own bound on a route gone quiet (route_follower.NAV_STALE_S), and no longer."""
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.route_follower import NAV_STALE_S
from openpilot.selfdrive.turnpilot.tests.test_turnpilotd_light_rails import right_at_an_unmapped_red


def approach(invalid_s: float):
  """The right at a red (test_turnpilotd_light_rails), navd invalid for invalid_s from 60 m out: [(t, s, v, speedCap,
  routeValid, the cap kept through the gap)] to the stand, and the time the gap began."""
  sim, line, _ = right_at_an_unmapped_red(invalid_s)
  rows = []
  while sim.t < 40. and not (sim.v < 0.05 and sim.s > line - 30.):
    p = sim.step(engaged=True)
    rows.append((sim.t, sim.s, sim.v, p.speedCap, p.routeValid, getattr(sim.tp, 'gap_cap', None)))
  gap = [r for r in rows if not r[4]]
  return rows, gap[0][0] if gap else None


def test_a_gap_in_the_route_keeps_the_corner_cap_and_the_car_never_speeds_up():
  rows, t0 = approach(2.5)
  before = [r for r in rows if r[0] < t0]
  during = [r for r in rows if t0 <= r[0] <= t0 + 2.5]
  assert during and all(r[3] <= before[-1][3] + 0.05 for r in during), [round(r[3], 2) for r in during[::10]]
  assert max(r[2] for r in during) <= before[-1][2] + 0.02  # 2026-11-13: +0.65 m/s under the lifted cap


def test_a_route_gone_longer_than_its_stale_bound_lets_the_cap_go():
  # navd invalid past NAV_STALE_S: the route is lost and the turn with it, its cap no longer kept (the camera's stop still
  # stands for the red)
  rows, t0 = approach(NAV_STALE_S + 3.)
  held = [r for r in rows if t0 + DT_MDL <= r[0] < t0 + NAV_STALE_S - 2 * DT_MDL]
  after = [r for r in rows if t0 + NAV_STALE_S + 2 * DT_MDL <= r[0] < t0 + NAV_STALE_S + 1.]
  assert held and all(r[5] is not None for r in held)
  assert after and all(r[5] is None or r[5][2] >= NAV_STALE_S for r in after)
