import math

import numpy as np
import pytest

# no beamngpy here: the scenario math runs without the game's client library
from openpilot.tools.sim.bridge.beamng.scenarios.timing import (GO_NEAR_M, WALK_SPEED, GoWatch, catch_up_lead, oncoming_release_delay,
                                                                pacer_drop_s, point_at, profile_times, script_nodes, travel_time,
                                                                walker_pose)

L_PATH = np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 12.0]])  # east 10 m, then north 10 m, climbing 2 m


def analytic_time(s, v, a):
  ramp = v * v / (2 * a)
  return math.sqrt(2 * s / a) if s <= ramp else v / a + (s - ramp) / v


class TestWalker:
  def test_point_at(self):
    p, h = point_at(L_PATH, 5.0)
    assert np.allclose(p, [5.0, 0.0, 10.0]) and h == pytest.approx(0.0)
    p, h = point_at(L_PATH, 15.0)
    assert np.allclose(p, [10.0, 5.0, 11.0]) and h == pytest.approx(math.pi / 2)
    assert np.allclose(point_at(L_PATH, -3.0)[0], L_PATH[0])
    assert np.allclose(point_at(L_PATH, 99.0)[0], L_PATH[-1])

  def test_walks_then_stands(self):
    assert np.allclose(walker_pose(L_PATH, -1.0)[0], L_PATH[0])  # not started yet
    p, h = walker_pose(L_PATH, 10.0 / WALK_SPEED)
    assert np.allclose(p, [10.0, 0.0, 10.0])
    p, h = walker_pose(L_PATH, 100.0)
    assert np.allclose(p, L_PATH[-1]) and h == pytest.approx(math.pi / 2)  # faces the way it last walked



def drive(watch, legs, dt=0.05):
  """Feed a GoWatch 20 Hz rows: legs of (seconds, m to the junction at the start, speed, holding, turn desire), the car
  moving toward the junction at speed. Returns the t of the first go and of the first held stand near the junction."""
  t, go, stood = 0.0, None, None
  for dur, d, v, holding, desire in legs:
    for k in range(int(round(dur / dt))):
      watch.update(t, d - v * k * dt, v, holding, desire)
      go = t if go is None and watch.went else go
      stood = t if stood is None and watch.held_t is not None else stood
      t += dt
  return go, stood


class TestGoWatch:
  def test_the_go_is_the_release_of_the_hold_at_the_junction_not_one_further_out(self):
    # BeamNG smoke 2026-10-28, ped_crosswalk_right: held at a signal 116 m out, then braking for the junction's own signal,
    # stood there 2 s, released: the go is the release at the junction, never the one 116 m out nor the braking to it
    go, stood = drive(GoWatch(), [(3., 117., 0., True, False), (3., 117., 0., False, False), (8., 116., 8., False, False),
                                  (5., 30., 3., True, True), (2., 12.3, 0., True, False), (0.5, 12.3, 0., False, False),
                                  (2., 12.3, 1., False, True)])
    assert stood == pytest.approx(19.0, abs=0.06) and go == pytest.approx(21.0, abs=0.06)

  def test_a_car_that_stood_and_moves_again_has_gone_whatever_the_plan_says(self):
    go, _ = drive(GoWatch(), [(2., 20., 0., True, False), (1., 20., 0.2, True, False), (1., 20., 1., True, False)])
    assert go == pytest.approx(3.0, abs=0.06)

  def test_on_a_green_no_stop_the_go_is_the_turn_desire_out_near_the_junction(self):
    go, stood = drive(GoWatch(), [(5., 70., 8., False, True)])
    assert stood is None and go == pytest.approx((70. - GO_NEAR_M) / 8., abs=0.06)

  def test_a_stand_or_a_desire_far_from_the_junction_is_no_go(self):
    go, stood = drive(GoWatch(), [(3., 60., 0., True, True), (3., 60., 0., False, True), (2., 60., 5., False, True)])
    assert go is None and stood is None


class TestOncomingRelease:
  @pytest.mark.parametrize("start_m", [30., 35.])
  def test_first_car_arrives_arrive_after_s_after_the_go(self, start_m):
    # BeamNG smoke 2026-10-28: released on the car braking to the hold line, the cars came after the car, stood at the
    # signal's line 22-26 m out, had turned. Released delay s after the car stands at its hold near the junction, the
    # first reaches the junction arrive_after s after the go: the sim driver's tap STOPPED_TAP_S after the stop
    from openpilot.tools.sim.bridge.beamng.driver import STOPPED_TAP_S
    delay = oncoming_release_delay(start_m, 11.0, 2.5, 3.5)
    assert delay >= 0. and delay + travel_time(start_m, 11.0, 2.5) == pytest.approx(STOPPED_TAP_S + 3.5, abs=0.01)

  def test_a_start_too_far_out_to_make_it_is_refused(self):
    with pytest.raises(ValueError, match="60"):
      oncoming_release_delay(60.0, 11.0, 2.5, 3.5)


class TestProfiles:
  @pytest.mark.parametrize("s", [5.0, 24.0, 60.0, 150.0])
  def test_matches_constant_accel(self, s):
    assert travel_time(s, 11.0, 2.5) == pytest.approx(analytic_time(s, 11.0, 2.5), abs=0.03)

  def test_stops_short(self):
    # cruise at 10 m/s, then from 50 m slow at 2 m/s^2 to a stop 25 m later: anything past 75 m is never reached
    t = profile_times([10.0, 60.0, 74.0, 76.0, 200.0], 10.0, 2.0, s_change=50.0, v_change=0.0)
    assert all(math.isfinite(x) for x in t[:3]) and t[3:] == [math.inf, math.inf]
    assert t == sorted(t)

  def test_slowing_takes_longer(self):
    fast = profile_times([100.0, 200.0], 10.0, 2.0)
    slow = profile_times([100.0, 200.0], 10.0, 2.0, s_change=100.0, v_change=4.0)
    assert slow[0] == pytest.approx(fast[0], abs=0.03) and slow[1] > fast[1] + 10.0

  def test_script_nodes(self):
    path = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    nodes = script_nodes(path, 10.0, 2.0)
    assert len(nodes) >= 3 and nodes[0]['t'] == 0.0 and nodes[0]['x'] == 0.0 and nodes[-1]['x'] == pytest.approx(100.0)
    assert all(b['t'] > a['t'] and b['x'] - a['x'] <= 4.0 + 1e-6 for a, b in zip(nodes, nodes[1:], strict=False))
    assert nodes[-1]['t'] == pytest.approx(analytic_time(100.0, 10.0, 2.0), abs=0.03)
    stopping = script_nodes(path, 10.0, 2.0, s_change=40.0, v_change=0.0)
    assert stopping[-1]['x'] < 70.0  # nodes past the stop are dropped


class TestPacer:
  def test_catch_up_lead_puts_it_alongside(self):
    v, a, dt = 12.0, 3.0, 0.01
    lead = catch_up_lead(v, v, a)
    ego, car, vel = 0.0, lead, 0.0
    for _ in range(int(10.0 / dt)):
      vel = min(vel + a * dt, v)
      car += vel * dt
      ego += v * dt
    assert abs(car - ego) < 0.2

  def test_drop_after_the_window_opens(self):
    # junction 380 m along the lane, window opens 250 m out, script starts 40 m along: alongside the car, the pacer reaches
    # the window 90 m in, then blocks 6 s at 12 m/s
    assert pacer_drop_s(40.0, 380.0, 250.0, 12.0, 6.0) == pytest.approx(90.0 + 72.0)
    assert pacer_drop_s(200.0, 380.0, 250.0, 12.0, 6.0) == pytest.approx(72.0)  # already inside the window
