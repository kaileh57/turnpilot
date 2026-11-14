"""turnpilotd's autonomous left standing at its line through a gap in navd's route: navInstruction invalid for a while (navd off
its step for a fix, rerouting), navd's 1 Hz messages arriving invalid. Up to the follower's bound on a lost route (NAV_STALE_S
from the last valid message) the left rides it out with nothing said (route_follower: line_hold) and never goes while it
lasts (nav_gap: no go without a valid route); a longer loss, or navd back on a route that doesn't turn there, ends the left
with the handback said as before. The Sim, route and road runs of test_turnpilotd_left."""
import pytest

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.hooks import TAKE_CONTROL_PROMPTS
from openpilot.selfdrive.turnpilot.route_follower import NAV_STALE_S
from openpilot.selfdrive.turnpilot.scene.tracker import RawFrame
from openpilot.selfdrive.turnpilot.tests.synthetic import junction_of, nav_instruction
from openpilot.selfdrive.turnpilot.tests.test_turnpilotd_left import JUNCTION, car_frame, drive, left_sim, raw_box
from openpilot.selfdrive.turnpilot.unprotected_left import CLEAR_WINDOW_S

Desire = log.Desire


def gap_at_the_line(gap_s: float, stream_s: float, new_route: bool = False, gap_after_s: float = 0.):
  """LEFT_25 standing at its line, a car in the band; 2 s on (t0) the stream has stream_s more to run, and navInstruction comes
  invalid from gap_after_s after t0 for gap_s. new_route: navd comes back on a new route (routeCount 1) that turns right at the next junction
  instead. Returns (sim, plans from the gap on, [whether nav was invalid] per plan)."""
  st = {'t0': None, 'n': 0}

  def raw(s):
    st['n'] += 1
    if st['n'] % 2:
      return None
    xs = [JUNCTION + 40. - (s.t * 12.) % 45.] if st['t0'] is None or s.t < st['t0'] + stream_s else []
    return RawFrame([raw_box(*car_frame(s, x - s.s, 3.6)) for x in xs if -5. < x - s.s < 400.], 'road', s.t)

  def nav(s, instr):
    if st['t0'] is None or s.t < st['t0'] + gap_after_s:
      return instr, True
    if s.t < st['t0'] + gap_after_s + gap_s:
      return instr, False
    if new_route:
      return nav_instruction(1500. - s.s, "turn", "right", remaining=instr.distanceRemaining, junction=junction_of("right"),
                             route_count=1), True
    return instr, True
  sim = left_sim()
  sim.raw, sim.nav_filter = raw, nav
  while not (sim.trace and str(sim.trace[-1][2].unprotectedLeft.state) == 'waitGap' and sim.v < 0.05):
    drive(sim, DT_MDL)
    assert sim.t < 90.
  drive(sim, 2.)
  st['t0'] = sim.t
  plans, invalid = [], []
  for _ in range(int(30. / DT_MDL)):
    turning = sim.tp.desires.started and abs(JUNCTION - sim.s) < 12. and sim.v > 0.3
    plans.append(sim.step(engaged=True, yaw_rate=sim.v / 8. if turning else 0.))
    invalid.append(sim.sm.updated['navInstruction'] and not sim.sm.valid['navInstruction'] or
                   (invalid[-1] if invalid and not sim.sm.updated['navInstruction'] else False))
  return sim, plans, invalid


def said(plans):
  return sorted({(str(p.controlPrompt), p.reason) for p in plans if str(p.controlPrompt) in TAKE_CONTROL_PROMPTS})


@pytest.mark.parametrize('gap_after_s', [0., 0.25, 0.5, 0.75])
@pytest.mark.parametrize('gap_s', [1.0, 2.0, 2.5])
def test_a_gap_in_the_route_at_the_lefts_line_is_ridden_out_in_silence(gap_s, gap_after_s):
  # until 2026-11-14 the first invalid message dropped the maneuver: the latch handed back (unprotectedTurnTakeControl) and
  # the left came back on a new id at its line ('Junction position unsure: take control'), never to turn
  # at every phase of navd's 1 Hz messages: the gap is timed from its first invalid message
  sim, plans, _ = gap_at_the_line(gap_s, stream_s=6., gap_after_s=gap_after_s)
  assert not said(plans), said(plans)
  go = next(k for k, p in enumerate(plans) if p.desire == Desire.turnLeft)
  assert 6. < go * DT_MDL < 6. + CLEAR_WINDOW_S + 0.5, go * DT_MDL  # one window after the stream, as with no gap
  assert sim.s > JUNCTION + 10.


@pytest.mark.parametrize('gap_s', [1.8, 2.2])
def test_a_left_whose_window_fills_inside_the_gap_waits_for_the_route_to_come_back(gap_s):
  # the stream ends 1 s before the gap: with no gap the window would be full and the left gone inside it, 2.5 s on
  sim, plans, invalid = gap_at_the_line(gap_s, stream_s=0., gap_after_s=1.)
  assert not said(plans), said(plans)
  went = [k for k, p in enumerate(plans) if str(p.controlPrompt) == 'leftGo' or p.desire == Desire.turnLeft]
  assert went and not any(invalid[k] for k in went)
  assert went[0] * DT_MDL >= 1. + gap_s - 1e-6, went[0] * DT_MDL
  assert sim.s > JUNCTION + 10.


def test_a_route_lost_past_the_followers_bound_still_ends_the_left_out_loud():
  _, plans, invalid = gap_at_the_line(NAV_STALE_S + 1., stream_s=6.)
  assert ('unprotectedTurnTakeControl', 'Unprotected left: take control') in said(plans), said(plans)
  first = next(k for k, p in enumerate(plans) if str(p.controlPrompt) in TAKE_CONTROL_PROMPTS)
  assert first * DT_MDL < NAV_STALE_S + 1.  # at the follower's bound, inside the loss
  assert not any(p.desire == Desire.turnLeft or str(p.controlPrompt) == 'leftGo' for k, p in enumerate(plans) if invalid[k])


def test_navd_back_inside_the_bound_on_a_route_that_does_not_turn_there_ends_the_left_out_loud():
  sim, plans, invalid = gap_at_the_line(2., stream_s=6., new_route=True)
  assert ('unprotectedTurnTakeControl', 'Unprotected left: take control') in said(plans), said(plans)
  assert not any(p.desire == Desire.turnLeft for p in plans) and sim.s < JUNCTION


def test_a_left_on_its_own_hold_after_a_withdrawn_go_rides_a_gap_in_the_route_too():
  # the latch let go by the go, the car standing on the left's own hold just past its line (unprotected_left: hold_here)
  from openpilot.selfdrive.turnpilot.tests.test_turnpilotd_left import withdrawn_go
  sim = withdrawn_go()
  assert sim.tp.left.hold_here and not sim.tp.left.went
  t0 = sim.t
  sim.nav_filter = lambda s, instr: (instr, not (t0 + 0.5 <= s.t < t0 + 2.5))
  plans = drive(sim, 4.)
  assert not said(plans), said(plans)
  assert not any(p.desire == Desire.turnLeft or str(p.controlPrompt) == 'leftGo' for p in plans[int(0.5 / DT_MDL):int(2.6 / DT_MDL)])
