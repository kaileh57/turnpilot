import pytest

import openpilot.cereal.messaging as messaging
from openpilot.cereal import custom, log
from openpilot.tools.sim.bridge.beamng.scenarios import score

DT = 0.05


def row(t, **kw):
  r = {'t': t, 'x': 0.0, 'y': 0.0, 'v': 8.0, 'enabled': True, 'alert': '', 'alert_text': '', 'steer': False, 'gas': False, 'brake': False,
       'state': 'following', 'reason': '', 'lc': 'none', 'prompt': 'none', 'lc_state': 'off', 'nav_valid': True, 'nav_type': 'turn',
       'nav_mod': 'right', 'nav_dist': 100.0, 'nav_remaining': 300.0, 'route': 1, 'dest_dist': 200.0, 'ped_gap': None, 'car_gap': None,
       'pacer_dx': None, 'blinker': ''}
  r.update(kw)
  return r


def drive(segments):
  """Rows at 20 Hz from (duration, fields) segments; a callable field gets the time into its segment."""
  rows, t = [], 0.0
  for dur, fields in segments:
    for k in range(int(round(dur / DT))):
      tau = k * DT
      rows.append(row(round(t + tau, 3), **{f: (v(tau) if callable(v) else v) for f, v in fields.items()}))
    t += dur
  return rows


def route(steps, v=10.0, route_id=1):
  """navInstruction fields for driving a route at v: steps are (length m, type, modifier), the last one an arrive."""
  total = sum(s[0] for s in steps)
  segs, done = [], 0.0
  for i, (length, mtype, mod) in enumerate(steps):
    future = total - done - length
    # navd moves on 10 m past a maneuver, so a step's distance runs down to -10
    span = length + (10.0 if i + 1 < len(steps) else -14.0) - (10.0 if i else 0.0)
    start = length - (10.0 if i else 0.0)
    segs.append((span / v, {'nav_type': mtype, 'nav_mod': mod, 'route': route_id,
                            'nav_dist': lambda tau, s=start: round(s - v * tau, 1),
                            'nav_remaining': lambda tau, s=start, f=future: round(s - v * tau + f, 1)}))
    done += length
  return segs


class TestDisengagements:
  def test_cause_is_the_last_alert(self):
    rows = drive([(2.0, {}), (1.0, {'alert': 'steerTempUnavailable/softDisabling', 'alert_text': 'TAKE CONTROL'}),
                  (1.0, {'enabled': False, 'alert': ''})])
    d = score.disengagements(rows)
    assert len(d) == 1 and d[0]['t'] == pytest.approx(3.0) and d[0]['cause'].startswith('steerTempUnavailable')

  def test_driver_override(self):
    d = score.disengagements(drive([(1.0, {}), (1.0, {'enabled': False, 'brake': True})]))
    assert [x['cause'] for x in d] == ['driver brake']

  def test_not_before_the_first_engagement(self):
    assert score.disengagements(drive([(2.0, {'enabled': False}), (2.0, {})])) == []


class TestHolds:
  def test_merges_flicker_and_sees_the_car_stop_and_go(self):
    rows = drive([(2.0, {}),
                  (3.0, {'state': 'holding', 'reason': 'pedestrian in crosswalk', 'v': lambda tau: max(8.0 - 4.0 * tau, 0.0)}),
                  (0.1, {'state': 'following', 'v': 0.0}),
                  (4.0, {'state': 'holding', 'reason': 'pedestrian in crosswalk', 'v': 0.0}),
                  (3.0, {'v': lambda tau: 2.0 * tau})])
    hs = score.holds(rows)
    assert len(hs) == 1
    h = hs[0]
    assert h['duration_s'] == pytest.approx(7.05, abs=0.06) and h['stopped'] and h['resumed']

  def test_a_hold_that_never_lets_go(self):
    hs = score.holds(drive([(1.0, {}), (5.0, {'state': 'holding', 'reason': 'waiting for oncoming', 'v': 0.0})]))
    assert hs[0]['stopped'] and not hs[0]['resumed']


class TestManeuvers:
  def test_route_taken(self):
    rows = drive(route([(150.0, 'turn', 'right'), (200.0, 'turn', 'left'), (80.0, 'arrive', '')]))
    ms = score.maneuvers(rows)
    assert [(m['modifier'], m['attempted'], m['taken']) for m in ms] == [('right', True, True), ('left', True, True)]

  def test_missed_turn_reroutes(self):
    # the car goes straight past the right turn; navd reroutes: new route id, a longer way round
    rows = drive(route([(150.0, 'turn', 'right'), (100.0, 'arrive', '')])[:1] +
                 [(2.0, {'nav_type': 'turn', 'nav_mod': 'right', 'nav_dist': lambda tau: round(-10.0 * tau, 1),
                         'nav_remaining': lambda tau: round(100.0 - 10.0 * tau, 1)})] +
                 route([(120.0, 'turn', 'right'), (90.0, 'turn', 'right'), (100.0, 'arrive', '')], route_id=2))
    ms = score.maneuvers(rows)
    assert (ms[0]['attempted'], ms[0]['taken']) == (True, False)
    assert all(m['taken'] for m in ms[1:])

  def test_instruction_before_route_still_a_reroute(self):
    # navd sends the new route's first instruction a tick before the new navRoute: the step still ends as a reroute
    rows = drive([(3.0, {'nav_dist': lambda tau: round(25.0 - 10.0 * tau, 1), 'nav_remaining': lambda tau: round(325.0 - 10.0 * tau, 1)}),
                  (0.05, {'nav_dist': 80.0, 'nav_remaining': 400.0, 'nav_mod': 'left'}),
                  (2.0, {'nav_dist': 80.0, 'nav_remaining': 400.0, 'nav_mod': 'left', 'route': 2})])
    ms = score.maneuvers(rows)
    assert (ms[0]['attempted'], ms[0]['taken']) == (True, False)

  def test_not_attempted_when_disengaged(self):
    rows = drive([(3.0, {'enabled': False, 'nav_dist': lambda tau: round(40.0 - 10.0 * tau, 1),
                         'nav_remaining': lambda tau: round(340.0 - 10.0 * tau, 1)})])
    assert not score.maneuvers(rows)[0]['attempted']


class TestEnd:
  def test_arrival_disengagement_timeout(self):
    e = score.EndCheck(60.0)
    assert e.update(row(0.0)) is None
    assert e.update(row(1.0, dest_dist=14.0)) == 'arrived'
    e = score.EndCheck(60.0)
    e.update(row(0.0))
    assert e.update(row(1.0, enabled=False)) == 'disengaged'
    assert score.EndCheck(60.0).update(row(60.0)) == 'timeout'

  def test_stop(self):
    e = score.EndCheck(60.0, end_on_stop_s=4.0)
    ends = [e.update(r) for r in drive([(2.0, {'v': 0.0}), (3.0, {}), (5.0, {'v': 0.0})])]
    first = next(i for i, x in enumerate(ends) if x)
    assert ends[first] == 'stopped' and first * DT == pytest.approx(9.0, abs=DT)


class TestChecks:
  def crosswalk_run(self, hold=True):
    hold_fields = {'state': 'holding', 'reason': 'pedestrian in crosswalk'} if hold else {}
    rows = drive(route([(120.0, 'turn', 'right'), (60.0, 'arrive', '')])[:1] +
                 [(4.0, {**hold_fields, 'v': 0.0, 'ped_gap': 9.0, 'nav_dist': 9.0, 'nav_remaining': 69.0})] +
                 [(2.0, {'v': 5.0, 'nav_dist': lambda tau: round(9.0 - 10.0 * tau, 1), 'nav_remaining': lambda tau: round(69.0 - 10.0 * tau, 1)})] +
                 [(4.0, {'v': 8.0, 'nav_type': 'arrive', 'nav_mod': '', 'nav_dist': lambda tau: round(50.0 - 10.0 * tau, 1),
                         'nav_remaining': lambda tau: round(50.0 - 10.0 * tau, 1), 'dest_dist': lambda tau: 50.0 - 10.0 * tau})])
    return rows

  def test_crosswalk_pass_and_fail(self):
    res = score.summarize('ped_crosswalk_right', self.crosswalk_run(), 'arrived')
    assert res['passed'], res['failures']
    assert res['maneuvers_taken'] == res['maneuvers_attempted'] == 1 and res['min_gap_ped_m'] == 9.0
    res = score.summarize('ped_crosswalk_right', self.crosswalk_run(hold=False), 'arrived')
    assert not res['passed'] and 'never held' in res['failures'][0]

  @pytest.mark.parametrize("gap,ok", [(5.0, True), (2.0, False), (9.5, False)])
  def test_ped_in_path_gap(self, gap, ok):
    # a pause at the rolling start 80 m out is not the stop for the pedestrian
    rows = drive([(2.0, {'v': 0.0, 'ped_gap': 80.0}), (6.0, {'v': 8.0, 'ped_gap': lambda tau, g=gap: g + 64.0 - 8.0 * tau}),
                  (4.0, {'v': lambda tau: 8.0 - 2.0 * tau, 'ped_gap': lambda tau, g=gap: g + 16.0 - 8.0 * tau + tau * tau}),
                  (4.0, {'v': 0.0, 'ped_gap': gap})])
    res = score.summarize('ped_in_path', rows, 'stopped')
    assert res['stop_gap_m'] == pytest.approx(gap, abs=0.05) and res['passed'] == ok
    assert score.stop_gap(rows, 'timeout') == pytest.approx(gap, abs=0.05)  # without the end, the first stop after moving

  def test_occupied_lane_waits_for_the_pacer(self):
    blocked = drive([(4.0, {'reason': 'car in target lane', 'pacer_dx': 1.0, 'v': 12.0})])
    late = drive([(1.0, {'lc_state': 'laneChangeStarting', 'pacer_dx': -22.0, 'v': 12.0})])
    early = drive([(1.0, {'lc_state': 'laneChangeStarting', 'pacer_dx': 3.0, 'v': 12.0})])
    assert score.summarize('occupied_lane', blocked + late, 'arrived')['passed']
    res = score.summarize('occupied_lane', blocked + early, 'arrived')
    assert not res['passed'] and 'alongside' in res['failures'][0]
    assert not score.summarize('occupied_lane', drive([(1.0, {})]) + late, 'arrived')['passed']  # never blocked


def msg(service, t, **fields):
  m = messaging.new_message(service, valid=True)  # navd marks a navInstruction without a route invalid
  m.logMonoTime = int(t * 1e9)
  body = getattr(m, service)
  for k, v in fields.items():
    setattr(body, k, v)
  return m.as_reader()


class TestFromMessages:
  def stream(self):
    """A recorded stream: route from navd, TurnPilot holding at a crosswalk, then a disengagement."""
    out = []
    for i in range(200):  # 10 s at 20 Hz
      t = i * DT
      holding = 2.0 <= t < 5.0
      out.append(msg('selfdriveState', t, enabled=t < 8.0, alertType='' if t < 7.0 else 'steerSaturated/warning'))
      out.append(msg('carState', t, vEgo=0.0 if holding else 6.0, steeringPressed=False))
      model = messaging.new_message('modelV2', valid=True)
      model.logMonoTime = int(t * 1e9)
      model.modelV2.meta.laneChangeState = log.LaneChangeState.laneChangeStarting if 6.0 <= t < 6.5 else log.LaneChangeState.off
      out.append(model.as_reader())
      if i % 20 == 0:
        out.append(msg('navInstruction', t, maneuverType='turn', maneuverModifier='right', maneuverDistance=60.0 - 6.0 * t,
                       distanceRemaining=160.0 - 6.0 * t))
      plan = msg('turnpilotPlan', t, state=custom.TurnpilotPlan.State.holding if holding else custom.TurnpilotPlan.State.approaching,
                 reason='pedestrian in crosswalk' if holding else '')
      out.append(plan)
    return out

  def test_scores_a_log(self):
    rows = score.rows_from_log(self.stream())
    assert len(rows) == 200 and rows[0]['state'] == 'approaching' and rows[50]['state'] == 'holding'
    assert rows[0]['lc_state'] == 'off'
    assert [lc['t'] for lc in score.lane_changes(rows)] == [pytest.approx(6.0)]  # the scorer's enum names match the readers'
    res = score.summarize('ped_crosswalk_right', rows, 'log ended')
    assert [(h['reason'], h['stopped'], h['resumed']) for h in res['holds']] == [('pedestrian in crosswalk', True, True)]
    assert len(res['disengagements']) == 1 and res['disengagements'][0]['cause'].startswith('steerSaturated')
    assert res['maneuvers_attempted'] == 1 and res['maneuvers_taken'] == 0  # it disengaged before navd moved on
