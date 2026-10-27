from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.desire_timing import (ENTRY_M, PULSE_GAP_FRAMES, REPULSE_S, STANDSTILL_REARM_S, TURN_DESIRE_LEAD_M,
                                                         DesireTiming, keep_lead)
from openpilot.selfdrive.turnpilot.route_follower import RouteFollower
from openpilot.selfdrive.turnpilot.tests.synthetic import FakeNavd

Desire = log.Desire


def drive(maneuvers, v=8., seconds=40., lines=(True, True), turns_at=8., speed_profile=None, noise=None, steer_sign=1.):
  """Drive along a route; the car steers into a turn once it is turns_at m from it (None: never turns).
  noise(i) m is added to the position navd sees on frame i. Returns [(s, distance, desire)] per frame."""
  navd, rf, dt_ = FakeNavd(maneuvers), RouteFollower(), DesireTiming()
  s, out = 0., []
  for i in range(int(seconds / DT_MDL)):
    v_now = speed_profile(i * DT_MDL) if speed_profile else v
    instr = navd.update(s + (noise(i) if noise else 0.))
    rf.update(instr, instr is not None, True, v_now, DT_MDL)
    steer, yaw = 0., 0.
    if turns_at is not None and rf.upcoming is not None and rf.distance < turns_at:
      steer, yaw = 90. * steer_sign, v_now / 8. * steer_sign  # an 8 m radius
    desire = dt_.update(rf.upcoming, rf.distance, rf.maneuver_id, rf.passed, v_now, steer, *lines, DT_MDL, yaw)
    out.append((s, rf.distance, desire))
    s += v_now * DT_MDL
  return out


def rising_edges(trace, desire):
  prev, edges = Desire.none, []
  for s, d, cur in trace:
    if cur == desire and prev != desire:
      edges.append((s, d))
    prev = cur
  return edges


def test_left_turn_desire_from_the_lead():
  trace = drive([(200., "turn", "left"), (900., "turn", "right")], v=12.)
  edges = rising_edges(trace, Desire.turnLeft)
  assert len(edges) == 2 and 0. < edges[1][1] < ENTRY_M  # the pulse at the lead, and the one for the turn itself
  s, d = edges[0]
  assert TURN_DESIRE_LEAD_M - 12. * DT_MDL - 1. < d < TURN_DESIRE_LEAD_M  # within a frame of the lead, dead-reckoned
  assert all(desire == Desire.none for s, d, desire in trace if s < 200. - TURN_DESIRE_LEAD_M - 1.)
  # held through the junction, dropped once navd moves on
  held = [desire for s, d, desire in trace if 200. - TURN_DESIRE_LEAD_M + 1. < s < 205.]
  assert held.count(Desire.turnLeft) == len(held) - PULSE_GAP_FRAMES  # but for the drop before the entry pulse
  assert all(desire != Desire.turnLeft for s, d, desire in trace if s > 230.)
  assert not rising_edges(trace, Desire.turnRight)  # the next turn, 700 m on, isn't reached


def test_navd_distance_noise_adds_no_rising_edges():
  # navd's 1 Hz distance jumps by 3 m either way around the lead: the desire latches on the first crossing
  for maneuver, desire in (("left", Desire.turnLeft), ("slight right", Desire.keepRight)):
    route = [(200., "turn" if maneuver == "left" else "fork", maneuver), (900., "turn", "right")]
    noisy = rising_edges(drive(route, v=4., seconds=60., noise=lambda i: 3. if (i // 20) % 2 else -3.), desire)
    assert noisy, maneuver
    # repeats only on the REPULSE_S clock, never from the noise
    assert all((s1 - s0) / 4. > REPULSE_S for (s0, _), (s1, _) in zip(noisy[:-1], noisy[1:], strict=True)), maneuver
    assert len(noisy) == 1 or desire != Desire.keepRight


def test_consecutive_turns_each_get_a_desire():
  trace = drive([(200., "turn", "left"), (320., "turn", "right"), (2000., "turn", "left")], v=6., seconds=70.)
  left, right = rising_edges(trace, Desire.turnLeft), rising_edges(trace, Desire.turnRight)
  assert 2 <= len(left) <= 3 and 2 <= len(right) <= 3  # 40 m at 6 m/s is more than REPULSE_S: repeats until turned
  assert 320. - TURN_DESIRE_LEAD_M - 1. < right[0][0] < 320. - TURN_DESIRE_LEAD_M + 1.
  assert all(s < 325. for s, _ in right) and all(s < 205. for s, _ in left)  # none once the car has turned


def test_slight_right_exit_sends_keep_right():
  v = 25.
  trace = drive([(1500., "off ramp", "slight right"), (3000., "turn", "left")], v=v, seconds=70.)
  edges = rising_edges(trace, Desire.keepRight)
  assert len(edges) == 1
  assert keep_lead(v) - v * 1.1 < edges[0][1] <= keep_lead(v)
  assert not rising_edges(trace, Desire.turnRight)


def test_keep_suppressed_without_lane_line_on_that_side():
  trace = drive([(300., "fork", "slight left"), (3000., "turn", "left")], v=15., seconds=30., lines=(False, True))
  assert not rising_edges(trace, Desire.keepLeft)


def test_uturn_and_straight_send_nothing():
  trace = drive([(100., "turn", "uturn"), (300., "continue", "straight"), (3000., "turn", "left")], v=10., seconds=40.)
  assert all(desire == Desire.none for s, d, desire in trace if s < 2900.)


def test_turn_desire_pulses_again_until_through_the_junction():
  # the model forgets a pulse after about 5 s; a slow approach gets a new one every REPULSE_S up to the junction
  trace = drive([(200., "turn", "right"), (900., "turn", "left")], v=3., turns_at=None, seconds=80.)
  edges = rising_edges(trace, Desire.turnRight)
  assert len(edges) == 4 and 0. < edges[-1][1] < ENTRY_M  # every REPULSE_S on the approach, and one at the entry
  for (s0, _), (s1, _) in zip(edges[:-2], edges[1:-1], strict=True):
    assert abs((s1 - s0) / 3. - REPULSE_S) < 0.3  # plus the gap
  # the drop between two pulses is long enough for modeld to see it
  gap = [desire for s, d, desire in trace if edges[0][0] < s < edges[1][0] and desire == Desire.none]
  assert len(gap) == PULSE_GAP_FRAMES


def test_repeat_pulses_stop_once_the_car_has_turned():
  # turning at 3 m/s on an 8 m radius from 20 m out: 60 degrees take 2.8 s, after which a pulse would turn the car a
  # second time; before that, a slow turn still needs them
  trace = drive([(200., "turn", "right"), (900., "turn", "left")], v=3., turns_at=20., seconds=80., steer_sign=-1.)
  edges = rising_edges(trace, Desire.turnRight)
  assert len(edges) >= 2 and all(d > 20. - 2.8 * 3. for _, d in edges)


def test_pulse_again_after_a_standstill():
  # stopped at 15 m from a left turn (a red light) for 10 s, then on through the turn
  def speed(t):
    return 8. if t < 185. / 8. else (0. if t < 185. / 8. + 10. else 5.)
  trace = drive([(200., "turn", "left"), (900., "turn", "right")], seconds=50., speed_profile=speed)
  edges = rising_edges(trace, Desire.turnLeft)
  assert abs(edges[-1][0] - 185.) < 1.  # a pulse goes as the car pulls away
  stopped = [desire for s, d, desire in trace if abs(s - 185.) < 0.5]
  assert stopped.count(Desire.none) >= int((10. - STANDSTILL_REARM_S) / DT_MDL) - 2
