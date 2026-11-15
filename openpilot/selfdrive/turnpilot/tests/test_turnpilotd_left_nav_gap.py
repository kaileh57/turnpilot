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


def gap_on_the_approach(at_m: float, gap_s: float, stream: bool):
  """LEFT_25 driven from far out, navInstruction invalid for gap_s from the frame the car is at_m from the junction, with a car
  streaming through the left's band until 40 s (stream) or an empty road. Returns (sim, [(t, s, v, plan, nav invalid)])."""
  cars = (lambda sim: [(JUNCTION + 40. - (sim.t * 12.) % 45., 3.6)] if sim.t < 40. else []) if stream else (lambda sim: ())
  sim, st = left_sim(cars), {'t0': None}

  def nav(s, instr):
    if st['t0'] is None and JUNCTION - s.s < at_m:
      st['t0'] = s.t
    return instr, st['t0'] is None or not (st['t0'] <= s.t < st['t0'] + gap_s)
  sim.nav_filter = nav
  rows = []
  while sim.s < JUNCTION + 25. and sim.t < 70.:
    p = drive(sim, DT_MDL)[0]
    rows.append((sim.t, sim.s, sim.v, p, st['t0'] is not None and st['t0'] <= sim.t < st['t0'] + gap_s))
  return sim, rows


@pytest.mark.parametrize('at_m', [28., 20., 14.])
def test_a_gap_in_the_route_while_the_left_brakes_to_its_line_keeps_its_stop_and_its_hold(at_m):
  # until 2026-11-14 a left braking to its line (the latch not yet set) was dropped on the gap's first invalid message, its
  # stop with it: the car rolled over its line at 4-5 m/s with the band full, the left skipped tooLate, and sped up straight
  # across the junction (the matrix's left_unsig|crossL cells: into a crossing car)
  sim, rows = gap_on_the_approach(at_m, 2.5, stream=True)
  _, alone = gap_on_the_approach(0., 0., stream=True)
  assert not any(p.tooFastToTurn for *_, p, _ in rows)
  assert not said([p for *_, p, _ in rows]), said([p for *_, p, _ in rows])
  assert not any(p.desire == Desire.turnLeft or str(p.controlPrompt) == 'leftGo' for *_, p, gap in rows if gap)
  past = next(t for t, s, *_ in rows if s > JUNCTION - 5.)
  assert past > 40., past  # stood at its line while the band was full, as with the route valid
  assert abs(past - next(t for t, s, *_ in alone if s > JUNCTION - 5.)) < 0.2
  assert sim.s > JUNCTION + 10. and abs(sim.heading_deg) > 60.  # then turned


@pytest.mark.parametrize('at_m', [28., 14.])
def test_an_empty_roads_left_braking_into_a_gap_goes_only_on_a_window_after_it(at_m):
  sim, rows = gap_on_the_approach(at_m, 2.5, stream=False)
  assert not any(p.tooFastToTurn for *_, p, _ in rows) and not said([p for *_, p, _ in rows])
  end = max(t for t, *_, gap in rows if gap)
  go = next(t for t, *_, p, _ in rows if p.desire == Desire.turnLeft)
  assert go > end, (go, end)
  assert sim.s > JUNCTION + 10. and abs(sim.heading_deg) > 60.


@pytest.mark.parametrize('frm', ['left', 'right'])
def test_a_t_stems_left_through_a_gap_in_the_route_pulses_and_holds_for_cross_traffic_as_with_the_route_valid(frm):
  # the matrix's +3 R2 against 31581b3dd (left_nocontrol_T|crossL/crossR|navInvalid25): its turn desire's repeat and entry
  # edges with the cross car tracked are the route-valid run's, 0.6 s later (the first pulse waits for the route back); the
  # older tree skipped this turn in the gap, so it had none to flag. The cross hold stays in force
  from openpilot.selfdrive.turnpilot.scene_policy import CROSS_REASON, LOOK_REASON
  from openpilot.selfdrive.turnpilot.tests import test_turnpilotd_cross_traffic as ct
  cross_at = ct.meets('left', frm)

  def run(gap: bool):
    sim, st, rows = ct.turn('left'), {'t0': None}, []

    def nav(s, instr):
      if st['t0'] is None and ct.JUNCTION - s.s < 60.:
        st['t0'] = s.t
      return instr, not gap or st['t0'] is None or not (st['t0'] <= s.t < st['t0'] + 2.5)
    sim.nav_filter = nav
    while sim.s < ct.JUNCTION + 20. and sim.t < 60.:
      plan = sim.step(ct.cross_car(sim, cross_at, frm), engaged=True)
      rows.append((sim.t, sim.s, sim.v, plan, gap and st['t0'] is not None and st['t0'] <= sim.t < st['t0'] + 2.5))
    rises = [k for k in range(1, len(rows)) if rows[k][3].desire == Desire.turnLeft and rows[k - 1][3].desire != Desire.turnLeft]
    return sim, rows, rises
  _, valid, valid_rises = run(False)
  sim, rows, rises = run(True)
  assert not any(rows[k][4] for k in rises)  # nothing pulsed while the route is gone
  # the same edges but for the last: at the look's walking pace (scene_policy, 2026-11-14) the desire repeats on its own clock
  # four or five times on the approach, and the gap's 0.6 s shift can move the last repeat past the run's end
  assert abs(len(rises) - len(valid_rises)) <= 1, (len(rises), len(valid_rises))
  # short of the lane the cross car comes along: for one in the far lane its centre line (until 2026-11-14 the near edge: the
  # look now walks the left into the near lane before a car SEEN_S out is tracked, and stops it short of the far lane)
  edge = ct.JUNCTION - 3.6 if frm == 'left' else ct.JUNCTION
  for r in (valid, rows):
    assert ct.bumper_as_it_crosses([x[:4] for x in r], cross_at) < edge
    # held by the hold or, for a far-lane car past the hold's commit point, by the look's own stop ("Look both ways")
    held = [x for x in r if str(x[3].reason) == CROSS_REASON or (str(x[3].reason) == LOOK_REASON and float(x[3].speedCap) < 1.)]
    assert held and held[0][0] < cross_at - 2.5
  assert sim.s >= ct.JUNCTION + 20. and not said([x[3] for x in rows])
