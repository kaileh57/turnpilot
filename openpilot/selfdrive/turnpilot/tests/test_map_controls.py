"""MapControlPolicy: stop signs, yields and traffic signals from the map (map_controls.py), on a point-mass car."""
import math

import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.hooks import SPEED_CAP_ACCEL_MIN, plan_accel
from openpilot.selfdrive.turnpilot.map_controls import (DWELL_S, GREEN_STEADY_S, LIGHT_STALE_S, MAP_DECEL_MAX, NO_LIGHTS,
                                                         T_LAT, YIELD_SPEED, Lights, MapControlPolicy, engage_distance,
                                                         light_input)
from openpilot.selfdrive.turnpilot.scene_policy import NO_ACCEL_CAP, ScenePolicy, decide
from openpilot.selfdrive.turnpilot.tests.synthetic import nav_instruction


class Car:
  """A car on a straight road with map controls at s positions [(s, kind[, line_mapped, junction, cross_suspect])],
  navd's fixes once a second (offset by gps(t) m), following the policy's limits through ScenePolicy's composition."""
  def __init__(self, controls, v=13.4, cruise=None, route_count=1, status='ok', yellow_decel=2.0):
    self.controls = list(controls)
    self.policy = MapControlPolicy(yellow_decel)
    self.scene = ScenePolicy()
    self.s, self.v, self.t = 0., v, 0.
    self.cruise = v if cruise is None else cruise
    self.route_count, self.status = route_count, status
    self.frame = 0
    self.gps = lambda t: 0.
    self.nav_valid = True
    self.lead_stop: float | None = None  # a lead stopped here: the car stops 0.5 m short of it
    self.rows = []

  def nav(self):
    controls = [(i, c[1], c[0] - self.s + self.gps(self.t)) + tuple(c[2:]) for i, c in enumerate(self.controls)
                if -15. < c[0] - self.s < 400.][:8]
    return nav_instruction(500., route_count=self.route_count, controls=controls, controls_status=self.status,
                           valid=self.nav_valid)

  def step(self, gas=False, lights=NO_LIGHTS, accel=None):
    """lights: Lights, or a function of the car giving them."""
    lights = lights(self) if callable(lights) else lights
    due = self.frame % 20 == 0
    self.frame += 1
    out = self.policy.update(self.nav() if due else None, due, self.nav_valid, self.v, gas, lights, DT_MDL)
    dec = decide(out.stops, self.v)
    dec.accel_cap = self.scene.ramp(dec.accel_cap, DT_MDL)
    cmd = min(plan_accel(dec.accel_cap, dec.speed_cap, self.v), (self.cruise - self.v) / 2., 1.5)
    if dec.should_stop:
      cmd = min(cmd, -1.)
    if gas:
      cmd = 1.0 if accel is None else accel  # the driver's foot: openpilot's long plan is overridden
    elif accel is not None:
      cmd = accel
    if self.lead_stop is not None and self.s > self.lead_stop - 8. - self.v ** 2 / 3. and self.v > 0.:
      cmd = min(cmd, -self.v ** 2 / (2. * max(self.lead_stop - self.s - 0.5, 0.1)) - 0.3)
    self.v = max(self.v + cmd * DT_MDL, 0.)
    self.s += self.v * DT_MDL
    self.t += DT_MDL
    self.rows.append((self.t, self.s, self.v, cmd, dec, out))
    return out, dec, cmd

  def run(self, seconds, **kw):
    res = None
    for _ in range(int(round(seconds / DT_MDL))):
      res = self.step(**kw)
    return res

  def run_until(self, cond, max_s=60., **kw):
    for _ in range(int(max_s / DT_MDL)):
      out, dec, cmd = self.step(**kw)
      if cond(out, dec):
        return out, dec
    raise AssertionError("condition never met")


def held(out, dec):
  return dec.should_stop and out.shown is not None and out.shown.action in ('held', 'awaitingDriver')


# --- stop sign lifecycle ---

def test_stop_sign_lifecycle():
  car = Car([(250., 'stop')])
  engaged_at = None
  caps = []
  for _ in range(int(40. / DT_MDL)):
    out, dec, cmd = car.step()
    if engaged_at is None and out.stops:
      engaged_at = 250. - car.s
      assert engaged_at <= engage_distance(13.4) + 1.
      assert engaged_at > engage_distance(13.4) - 2.
    if out.stops and not dec.should_stop and car.v > 0.3:
      caps.append(out.stops[0].speed_cap)
    assert cmd >= MAP_DECEL_MAX - 1e-6  # never harder than -2.0
    if held(out, dec):
      break
  else:
    raise AssertionError("never held")
  assert all(b <= a + 1e-9 for a, b in zip(caps, caps[1:], strict=False))  # the cap never rises on the approach
  assert car.v < 0.3 and -2. < 250. - car.s < 5.
  assert out.reason == '' and 'stop sign' in dec.reason.lower()
  # the dwell: gas in it is ignored, and there's no prompt yet
  out, dec, _ = car.run(1.0)
  assert held(out, dec) and out.prompt == 'none'
  car.step(gas=True, accel=0.)
  out, dec, _ = car.step(accel=0.)
  assert held(out, dec)
  out, dec, _ = car.run(DWELL_S)
  assert held(out, dec) and out.prompt == 'stopSignGo' and 'tap gas to go' in dec.reason
  # then a gas tap goes: the hold releases at standstill (turnpilotd re-pulses the turn there)
  out, dec, _ = car.step(gas=True, accel=0.)
  assert not dec.should_stop and out.released_stopped and out.prompt == 'none'
  out, dec, _ = car.run(5.)
  assert not out.stops and car.s > 260.


def test_no_motion_before_the_tap():
  car = Car([(150., 'stop')])
  car.run_until(held)
  s_held = car.s
  car.run(30.)
  assert car.s - s_held < 0.05 and car.rows[-1][-1].prompt == 'stopSignGo'


@pytest.mark.parametrize("v0", [11.2, 15.6])  # 25 and 35 mph
def test_stop_error_and_creep_with_gps_noise(v0):
  for sign in (-1., 1.):
    car = Car([(300., 'stop')], v=v0)
    car.gps = lambda t, sign=sign: sign * 3. * math.cos(t)  # +-3 m GPS error, swinging
    car.run_until(held)
    s_held = car.s
    assert -2. < 300. - car.s < 5.
    car.run(10.)
    assert car.s - s_held < 0.3


# --- the latch ---

def test_latch_survives_jumps_gaps_new_routes_and_invalid_nav():
  car = Car([(120., 'stop')])
  car.run_until(held)
  s_held = car.s
  car.gps = lambda t: 3.
  car.run(1.5)
  car.gps = lambda t: -3.
  car.run(1.5)
  car.nav_valid = False
  car.run(5.)
  car.nav_valid, car.route_count, car.controls = True, 2, []  # a reroute that doesn't list it
  out, dec, _ = car.run(1.5)
  assert held(out, dec) and car.s - s_held < 0.05
  # it still releases only by its rule: the dwell's long past, a tap goes
  out, dec, _ = car.step(gas=True, accel=0.)
  assert not dec.should_stop


def test_overrun_with_gas_held_through_the_dwell():
  car = Car([(120., 'stop')])
  car.run_until(held)
  car.run(0.5)
  out, dec, _ = car.run(3., gas=True, accel=1.)  # the driver drives off in the dwell
  assert not dec.should_stop
  assert any(o.alert == 'stopOverrun' for *_, o in car.rows[-60:])
  out, dec, _ = car.run(3.)
  assert not dec.should_stop and car.v > 2.


# --- gas before the stop ---

def test_gas_before_the_stop_then_lift_past_the_line_is_an_overrun():
  car = Car([(150., 'stop')])
  car.run_until(lambda out, dec: out.stops)
  car.run_until(lambda out, dec: car.s > 151., gas=True, accel=0.5)
  out, dec, _ = car.step()
  assert not out.stops and out.alert == 'stopOverrun'
  out, dec, _ = car.run(1.)
  assert out.alert == 'stopOverrun' and not dec.should_stop


def test_gas_before_the_stop_then_lift_short_resumes_at_the_floor():
  car = Car([(150., 'stop')], v=10.)
  car.run_until(lambda out, dec: out.stops)
  car.run_until(lambda out, dec: car.s > 128., gas=True, accel=1.)  # 22 m short at about 14 m/s: needs about -4.4
  out, dec, cmd = car.step()
  assert out.alert == 'cannotStopComfortably' and out.stops[0].accel_cap == MAP_DECEL_MAX
  assert out.shown.action == 'infeasible'
  for _ in range(40):
    out, dec, cmd = car.step()
    assert cmd >= MAP_DECEL_MAX - 1e-6


def test_infeasible_at_first_sight_brakes_at_the_floor_and_alerts():
  car = Car([(20., 'stop')], v=15.)
  out, dec, cmd = car.step()
  assert out.stops[0].accel_cap == MAP_DECEL_MAX == SPEED_CAP_ACCEL_MIN
  assert out.alert == 'cannotStopComfortably' and out.shown.action == 'infeasible'
  cmds = [car.step()[2] for _ in range(30)]
  assert min(cmds) >= MAP_DECEL_MAX - 1e-6
  # it runs the line at the floor: an overrun, said out loud, never harder braking
  out, dec = car.run_until(lambda out, dec: out.alert == 'stopOverrun')
  assert not dec.should_stop


# --- queue ---

def test_queue_behind_a_lead_creeps_to_the_line():
  car = Car([(200., 'stop')])
  car.lead_stop = 180.  # stopped behind a car 20 m short of the line: no latch there
  car.run_until(lambda out, dec: car.v < 0.01 and car.s > 150.)
  assert 200. - car.s > 5.
  out, dec, _ = car.run(3.)
  assert not dec.should_stop and out.shown.action != 'held'
  car.lead_stop = None  # the lead leaves: the curve re-seeds and the car creeps up
  car.run_until(held, max_s=30.)
  assert -2. < 200. - car.s < 5.


# --- signals, no light source (the car tonight) ---

def test_unknown_signal_holds_and_prompts():
  car = Car([(200., 'signal')])
  out, dec = car.run_until(lambda out, dec: out.stops)
  assert out.prompt == 'signalGo' and 'light not checked' in dec.reason
  car.run_until(held)
  out, dec, _ = car.run(20.)
  assert held(out, dec) and out.prompt == 'signalGo' and 'tap gas to go' in dec.reason
  out, dec, _ = car.step(gas=True, accel=0.)
  assert not dec.should_stop and out.released_stopped


def test_unknown_signal_pre_confirm_while_rolling_and_release_span():
  car = Car([(200., 'signal'), (220., 'signal'), (300., 'signal')])
  car.run_until(lambda out, dec: out.stops)
  car.run(1.)
  car.step(gas=True, accel=0.)  # a tap while rolling: the driver saw the light
  out, dec = car.run_until(lambda out, dec: car.s > 205.)
  assert min(r[2] for r in car.rows) > 5.  # never stopped
  # the one 20 m on (the far side of a divided road) went with it; the one at 300 m is its own
  assert all(s.reason == '' or car.s > 220. for s in out.stops)
  out, dec = car.run_until(lambda out, dec: out.stops and out.shown.id == 2)
  car.run_until(held)


# --- signals with the sim oracle ---

def lights(state, distance):
  return Lights([(state, distance)], 'simOracle')


def oracle(state, offset=0.):
  """The sim oracle reporting the first control's light, as the car moves."""
  return lambda car: lights(state, car.controls[0][0] - car.s + offset)


def test_green_has_no_map_cap():
  car = Car([(200., 'signal')])
  for _ in range(int(25. / DT_MDL)):
    out, dec, cmd = car.step(lights=oracle('green'))
    assert dec.speed_cap >= car.v and not dec.should_stop
  assert car.s > 210. and out.reason in ('', 'Green (sim oracle)')


def test_red_holds_and_steady_green_releases():
  car = Car([(150., 'signal')])
  car.run_until(held, lights=oracle('red'))
  out, dec, _ = car.run(20., lights=oracle('red'))
  assert held(out, dec) and out.prompt == 'none' and dec.reason == 'Red light (sim oracle)'
  out, dec, _ = car.run(GREEN_STEADY_S - 0.1, lights=oracle('green'))  # 0.4 s of green: not yet
  assert held(out, dec)
  out, dec, _ = car.run(0.15, lights=oracle('green'))
  assert not dec.should_stop and any(r[-1].released_stopped for r in car.rows[-3:])
  out, dec, _ = car.run(5., lights=oracle('green'))
  assert car.s > 155. and not out.stops


def test_gas_on_red_releases_with_going_on_red():
  car = Car([(150., 'signal')])
  car.run_until(held, lights=oracle('red'))
  out, dec, _ = car.step(gas=True, accel=0., lights=oracle('red'))
  assert not dec.should_stop and out.alert == 'goingOnRed'


def test_stale_or_far_light_is_unknown():
  car = Car([(150., 'signal')])
  out, dec = car.run_until(lambda out, dec: out.stops, lights=lights('green', 100.))  # 25 m+ off the control: not its light
  assert out.prompt == 'signalGo'
  # two lights disagreeing about it: unknown
  out, dec, _ = car.step(lights=Lights([('green', car.controls[0][0] - car.s), ('red', car.controls[0][0] - car.s + 3.)], 'simOracle'))
  assert out.shown.light_state == 'unknown' and out.prompt == 'signalGo'


@pytest.mark.parametrize("mph", [25, 35, 45])
@pytest.mark.parametrize("side", [-1., 1.])
def test_yellow_boundary(mph, side):
  v = mph * 0.44704
  d_stop = v * T_LAT + v ** 2 / (2. * 2.0)
  car = Car([(400., 'signal')], v=v)
  car.run_until(lambda out, dec: 400. - car.s < d_stop + side * 1. + 5., lights=oracle('green'))
  car.run_until(lambda out, dec: 400. - car.s < d_stop + side * 1., lights=oracle('green'))
  out, dec, _ = car.step(lights=oracle('yellow'))
  if side > 0:  # 1 m beyond the stopping distance: stop, never past the yellow floor
    assert out.shown.action in ('braking', 'approaching') and out.alert == 'none'
    for _ in range(int(20. / DT_MDL)):
      out, dec, cmd = car.step(lights=oracle('red'))
      assert cmd >= -2.0 - 1e-6
      if held(out, dec):
        break
    assert held(out, dec) and 400. - car.s > -2.
  else:  # 1 m inside it: committed, never reversed by the red
    assert out.shown.action == 'proceedingOnYellow' and out.alert == 'proceedingOnYellow'
    for _ in range(int(10. / DT_MDL)):
      out, dec, cmd = car.step(lights=oracle('red'))
      assert not dec.should_stop and dec.speed_cap >= car.v
    assert car.s > 405.


def test_flicker_green_unknown_green():
  car = Car([(300., 'signal')])
  car.run_until(lambda out, dec: 300. - car.s < 80., lights=oracle('green'))  # engaged, and going on the green
  out, dec, _ = car.run(0.6, lights=oracle('green'))
  assert out.shown.action == 'released' and dec.speed_cap >= car.v
  out, dec, _ = car.run(0.4, lights=NO_LIGHTS)  # 0.4 s with no light: far out, that's a stop decision
  assert dec.speed_cap < car.v or dec.accel_cap < NO_ACCEL_CAP
  out, dec, _ = car.run(0.4, lights=oracle('green'))  # green back, not steady yet: still stopping
  assert dec.speed_cap < car.v + 0.5
  out, dec, _ = car.run(0.2, lights=oracle('green'))
  assert dec.speed_cap >= car.v  # steady: no cap


def test_light_input_acceptance():
  class Msg:
    source = 'simOracle'
    lights = [type('L', (), {'state': 'green', 'distance': 50., 'instance': 'trafficLight 1'})()]
  ok = light_input(Msg(), age_s=0.1, simulation=True, oracle_param=True)
  assert ok.lights == [('green', 50.)] and not ok.rejected
  for kw in ({'simulation': False, 'oracle_param': True}, {'simulation': True, 'oracle_param': False}):
    r = light_input(Msg(), age_s=0.1, **kw)
    assert r.lights is None and r.rejected
  stale = light_input(Msg(), age_s=LIGHT_STALE_S + 0.1, simulation=True, oracle_param=True)
  assert stale.lights is None and not stale.rejected
  assert light_input(None, age_s=0., simulation=True, oracle_param=True).lights is None


# --- yields ---

def test_yield_caps_speed_never_stops():
  car = Car([(150., 'yield')])
  for _ in range(int(20. / DT_MDL)):
    out, dec, cmd = car.step()
    assert not dec.should_stop and dec.accel_cap == NO_ACCEL_CAP or dec.accel_cap >= MAP_DECEL_MAX
    if 0. < 150. - car.s < 1.:
      assert car.v < YIELD_SPEED + 0.6
    if out.stops:
      assert out.prompt == 'yieldAhead'
  assert car.s > 160.


def test_cross_suspect_yield_dropped():
  car = Car([(150., 'yield', False, True, True)])
  out, dec, _ = car.run(15.)
  assert car.s > 160. and min(r[2] for r in car.rows) > 13.


def test_cross_suspect_stop_still_stops():
  car = Car([(150., 'stop', False, True, True)])
  out, dec = car.run_until(held)
  assert 'unverified' in dec.reason


# --- status ---

def test_controls_status_not_ok_is_loud():
  car = Car([], status='unsupported')
  out, dec, _ = car.step()
  assert out.alert == 'mapControlsUnavailable' and out.reason == 'Map stops/signals: unsupported'
  out, dec, _ = car.run(10.)
  assert out.reason == 'Map stops/signals: unsupported'
  ok = Car([], status='ok')
  out, dec, _ = ok.step()
  assert out.alert == 'none' and out.reason == ''


def test_light_rejected_alert():
  car = Car([(300., 'signal')])
  out, dec, _ = car.step(lights=Lights(None, 'simOracle', rejected=True))
  assert out.alert == 'lightSourceRejected'


def test_crossing_reason():
  car = Car([(200., 'signal', False, False, False)])
  out, dec = car.run_until(held)
  assert 'crossing' in dec.reason
