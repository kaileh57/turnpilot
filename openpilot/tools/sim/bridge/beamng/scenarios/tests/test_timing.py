import math

import numpy as np
import pytest

# no beamngpy here: the scenario math runs without the game's client library
from openpilot.tools.sim.bridge.beamng.scenarios.timing import (WALK_SPEED, catch_up_lead, eta, oncoming_should_release, pacer_drop_s,
                                                                point_at, profile_times, script_nodes, travel_time, walker_pose,
                                                                walker_should_start)

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

  def test_starts_lead_before_the_car(self):
    # at 8 m/s, 4 s lead, 1.5 m to the road edge: leave when the car is 8 * (4 + 1.5 / 1.4) = 40.6 m out
    assert not walker_should_start(41.5, 8.0, 1.5, 4.0)
    assert walker_should_start(40.0, 8.0, 1.5, 4.0)

  def test_a_crawling_car_still_triggers(self):
    assert eta(4.0, 0.0) == pytest.approx(4.0)  # speed floored at 1 m/s
    assert walker_should_start(4.0, 0.0, 1.5, 4.0)
    assert not walker_should_start(40.0, 0.0, 1.5, 4.0)


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


class TestOncoming:
  @pytest.mark.parametrize("v_ego", [5.0, 8.0, 12.0])
  def test_first_car_arrives_just_after_the_car_reaches_the_hold_line(self, v_ego):
    # a car holding its speed toward the hold line, checked at 20 Hz like the runner; the oncoming car stands 60 m out
    s_car, v_car, a_car, after, dt = 60.0, 11.0, 2.5, 1.5, 0.05
    d, t = 150.0, 0.0
    while not oncoming_should_release(d, v_ego, s_car, v_car, a_car, after):
      d -= v_ego * dt
      t += dt
    ego_at_hold = t + d / v_ego
    car_at_junction = t + travel_time(s_car, v_car, a_car)
    assert after <= car_at_junction - ego_at_hold <= after + dt * 1.01

  def test_late_releases_at_once(self):
    assert oncoming_should_release(2.0, 8.0, 60.0, 11.0, 2.5, 1.5)


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
