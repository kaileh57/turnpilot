"""turnpilotd driving the rolling left (unprotected_left.ROLLING_GO) in closed loop: the autonomous left whose oncoming band has
been clear a whole window on the approach goes without stopping at its line; anything else brakes to the line as before.
The Sim, route and road runs of test_turnpilotd_left, the car in the turn's own lane with its lamp lit on the approach."""
import math
import types
from collections import deque

import pytest

from openpilot.cereal import log, messaging
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot import map_controls as mc
from openpilot.selfdrive.turnpilot.detector.light_state import LIGHT_MODE_DEFAULT
from openpilot.selfdrive.turnpilot.hooks import TAKE_CONTROL_PROMPTS, plan_accel
from openpilot.selfdrive.turnpilot.lane_planner import EDGE_STD_MAX
from openpilot.selfdrive.turnpilot.route_follower import LEFT, RIGHT
from openpilot.selfdrive.turnpilot.scene.tracker import RawBox, RawFrame
from openpilot.selfdrive.turnpilot.scene_policy import ONCOMING_ENTRY_M, STOP_DECEL
from openpilot.selfdrive.turnpilot.tests import test_map_controls as map_tests
from openpilot.selfdrive.turnpilot.tests.synthetic import junction_of, model_v2
from openpilot.selfdrive.turnpilot.tests.test_camera_lights import detections as light_detections
from openpilot.selfdrive.turnpilot.tests.test_turnpilotd import THREE_LANES_LEFT_TURN, Sim
from openpilot.selfdrive.turnpilot.tests.test_turnpilotd_left import JUNCTION, LINE, STRAIGHT_M, TEST_RANGE_M, V_25, car_frame, raw_box
from openpilot.selfdrive.turnpilot.turn_speed import V_CORNER
from openpilot.selfdrive.turnpilot.unprotected_left import (CLEAR_PATH_PAST_M, CLEAR_WINDOW_S, FAR_EDGE_AHEAD_M, FULL_PASS_RANGE_M, HELD_MIN_M,
                                                           LANE_W_M, ROLL_STOP_LAG_S, SPEED_MARGIN, VETO_MARGIN_M, WITHDRAWN_REASON,
                                                           red_stop_m)

Desire = log.Desire
ROLL_MARGIN_V = 0.8  # m/s under the corner's speed a rolling left may dip (the car follows its caps about a second behind)


def lets_go_rolling(policy_cls=mc.MapControlPolicy) -> bool:
  """map_controls lets go of the unprotected turn's stop on the left's go with the latch not holding (the rolling go's
  release): a car 22 m from the turn on its stop curve, LeftCommand.go from there."""
  car = map_tests.left_car()
  for _ in range(int(60. / DT_MDL)):
    d = car.turn[0] - car.s
    out, _, _ = car.policy_step(map_tests.left_cmd(go=d < 22., reason="Turning"))
    if d < 20.:
      return not out.stops
  return False


def test_map_controls_lets_go_of_a_rolling_lefts_stop():
  # the one thing the rolling go needs of map_controls: without it a rolling go is refused (turnpilot.left_roll_refused)
  # and every left brakes to its line as before (the last test here)
  assert lets_go_rolling()


def box(sim, s, y, motion='unknown'):
  """A car's raw box at road position (s, y left), as the road camera sees it from the car's heading, its motion the
  tracker's verdict."""
  b = raw_box(*car_frame(sim, s - sim.s, y))
  return RawBox(b.camera, b.cls, b.conf, b.bearing_deg, b.span_deg, b.xy, b.near_m, b.box, motion=motion)


def runs(cars=lambda sim: ()):
  """A road camera run every other frame of cars(sim): [(s, y left, motion)]."""
  frame = [0]

  def raw(sim):
    frame[0] += 1
    if frame[0] % 2:
      return None
    return RawFrame([box(sim, s, y, m) for s, y, m in cars(sim) if -5. < s - sim.s < 400.], 'road', sim.t)
  return raw


def roll_sim(cars=lambda sim: (), lamp=True, **kw):
  """A 25 mph left at JUNCTION with one opposing lane, the car in the route's left-turn lane (no lane change wanted), the
  driver's lamp lit when the car asks for it (300 ft out), the full pass's range measured."""
  sim = Sim([(JUNCTION, "turn", "left", (0., 270., 2, 'present', V_25, 1, STRAIGHT_M)), (1500., "turn", "right", junction_of("right"))],
            nav_lanes={0: THREE_LANES_LEFT_TURN}, model_lanes=(0, 2), **kw)
  sim.tp.auto_left, sim.tp.left_range_m = 'auto', TEST_RANGE_M
  sim.go_taps, sim.answers_prompt, sim.raw = False, lamp, runs(cars)
  return sim


def drive(sim, until_s=JUNCTION + 30., limit_s=120.):
  """Engaged to until_s (or limit_s), the model turning at an 8 m radius through the junction once the turn is pulsed.
  [(t, s, v, plan, the left's decision)]."""
  rows = []
  while sim.s < until_s and sim.t < limit_s:
    turning = sim.tp.desires.started and abs(JUNCTION - sim.s) < 12. and sim.v > 0.3
    plan = sim.step(engaged=True, yaw_rate=sim.v / 8. if turning else 0.)
    rows.append((sim.t, sim.s, sim.v, plan, sim.tp.left_now))
  return rows


def stood(rows):
  """The car came to rest short of the junction."""
  return any(v < 0.3 and s < JUNCTION for _, s, v, _, _ in rows)


def rolled(rows):
  return any(d is not None and d.rolled for *_, d in rows)


def test_an_empty_road_is_turned_across_without_a_stop():
  sim = roll_sim()
  rows = drive(sim)
  assert sim.s >= JUNCTION + 30.
  assert rolled(rows) and not stood(rows) and not any(p.shouldStop for *_, p, _ in rows)
  # never under the corner's speed by more than the margin, from the approach through the junction
  slowest = min(v for _, s, v, _, _ in rows if s < JUNCTION + 5.)
  assert slowest >= V_CORNER - ROLL_MARGIN_V, slowest
  # the go where the stop for the line was about to take the car under that speed, the turn pulsed from it, not before
  go = next(i for i, r in enumerate(rows) if r[4].rolled)
  assert 15. < JUNCTION - rows[go][1] < 35., JUNCTION - rows[go][1]
  assert all(p.desire == Desire.none for *_, p, _ in rows[:go - 1])
  pulse = next(i for i, r in enumerate(rows) if r[3].desire == Desire.turnLeft)
  assert pulse - go <= 2 and rows[pulse][1] < LINE
  assert str(rows[go][3].unprotectedLeft.state) == 'go' and rows[go][3].controlPrompt == 'leftGo'
  assert any(str(p.unprotectedLeft.state) == 'committed' for *_, p, _ in rows[go:])
  assert not any(str(p.controlPrompt) in TAKE_CONTROL_PROMPTS or p.turnMissed for *_, p, _ in rows)
  assert sim.tp.go_cert.id == 0 or not sim.tp.go_cert.active  # no lift: the car never stood


def oncoming(v, s_at, t_at):
  """A car driving toward ours at v m/s in the oncoming lane, at road position s_at at time t_at, read closing."""
  return lambda sim: [(s_at - v * (sim.t - t_at), 3.5, 'closing')]


@pytest.mark.parametrize('lead_s', [12., 8., 5., 3.5])
def test_a_closing_car_seen_before_the_go_brings_the_car_to_its_line(lead_s):
  # an oncoming car that reaches the junction lead_s after the car would have rolled into it: in the band on the roll-in,
  # so no rolling go; the car stands at its line and goes once it has passed, a window after
  empty = drive(roll_sim())
  t_go = next(t for t, *_, d in empty if d.rolled)
  cars = oncoming(12., JUNCTION, t_go + lead_s)
  sim = roll_sim(cars)
  rows = drive(sim)
  assert sim.s >= JUNCTION + 30. and not rolled(rows)
  stand = [(t, s) for t, s, v, _, _ in rows if v < 0.3 and s < JUNCTION]
  assert stand and all(LINE - 1.5 < s <= LINE + 0.2 for _, s in stand), stand[:3]
  t_passed = t_go + lead_s + 5. / 12.  # out of the band: ONCOMING_PAST_M short of the maneuver point
  go = next(t for t, _, _, p, _ in rows if p.desire == Desire.turnLeft)
  assert go >= t_passed + CLEAR_WINDOW_S - 0.3, (go, t_passed)
  assert not any(str(p.controlPrompt) in TAKE_CONTROL_PROMPTS for *_, p, _ in rows)


def test_a_car_first_seen_after_the_rolling_go_withdraws_it_short_of_the_oncoming_lanes():
  # nothing in the band until the go; 0.5 s after it a car appears 60 m up the oncoming lane at 12 m/s: the go is withdrawn,
  # the car stops on the comfortable curve short of the oncoming lanes and goes after it has passed
  empty = drive(roll_sim())
  t_go = next(t for t, *_, d in empty if d.rolled)
  sim = roll_sim(lambda sim: [(JUNCTION + 60. - 12. * (sim.t - t_go - 0.5), 3.5, 'closing')] if sim.t >= t_go + 0.5 else [])
  rows = drive(sim)
  assert rolled(rows) and sim.s >= JUNCTION + 30.
  assert any(str(p.controlAlert) == 'oncomingHolding' for *_, p, _ in rows)
  stand = [s for _, s, v, _, _ in rows if v < 0.3 and s < JUNCTION]
  assert stand and max(stand) <= JUNCTION - HELD_MIN_M, max(stand, default=None)
  # braked no harder than the comfortable curve (the Sim's car follows the plan's caps)
  decel = min((b[2] - a[2]) / DT_MDL for a, b in zip(rows, rows[1:], strict=False) if b[1] < JUNCTION)
  assert decel >= -(STOP_DECEL + 0.6), decel
  # its pulse withdrawn with it, a fresh one at the go after the car has passed (5 s to the junction, a window after)
  t_w = next(t for t, *_, p, _ in rows if str(p.controlAlert) == 'oncomingHolding')
  later = [t for t, _, _, p, _ in rows if t > t_w + 0.5 and p.desire == Desire.turnLeft]
  assert later and later[0] >= t_go + 0.5 + 5. + CLEAR_WINDOW_S - 0.5, (later[:1], t_go)
  assert not any(str(p.controlPrompt) in TAKE_CONTROL_PROMPTS for *_, p, _ in rows)


@pytest.mark.parametrize('late_s', [1.2, 1.6, 2.0, 2.6])
def test_a_car_seen_late_in_a_rolling_go_never_stops_the_car_in_the_oncoming_lanes(late_s):
  # whenever in the go a car is first seen: either the go is withdrawn and the car stands short of the oncoming lanes, or it
  # could no longer stop short of them and the turn is finished; never a stand in them
  empty = drive(roll_sim())
  t_go = next(t for t, *_, d in empty if d.rolled)
  sim = roll_sim(lambda sim: [(JUNCTION + 80. - 12. * (sim.t - t_go - late_s), 3.5, 'closing')] if sim.t >= t_go + late_s else [])
  rows = drive(sim)
  assert sim.s >= JUNCTION + 30.
  stand = [s for _, s, v, _, _ in rows if v < 0.3 and JUNCTION - 15. < s < JUNCTION + 8.]
  assert all(s <= JUNCTION - ONCOMING_ENTRY_M for s in stand), max(stand)


def lagged(lag_s):
  """The Sim's own follower (Sim.step) with each plan frame's command applied lag_s later: the actuator's lag."""
  q = deque()

  def accel(sim, plan):
    a = min(plan_accel(plan.accelCap, plan.speedCap, sim.v), (sim.cruise - sim.v) / 2., 1.5)
    q.append(min(a, -1.) if plan.shouldStop else a)
    return q.popleft() if len(q) > round(lag_s / DT_MDL) else 0.
  return accel


def test_a_rolling_go_withdrawn_on_a_car_that_answers_half_a_second_late_never_rests_in_the_oncoming_lanes():
  # the car follows each command ROLL_STOP_LAG_S late (the lag the go's own last point budgets): whenever after the go the
  # oncoming car is first seen, a withdrawn go rests the car where its next go is judged (HELD_MIN_M or more short of the
  # junction), and one seen past the last point such a stop fits is carried through, never a stand in the lanes
  base = roll_sim()
  base.accel = lagged(ROLL_STOP_LAG_S)
  t_go = next(t for t, *_, d in drive(base) if d is not None and d.rolled)
  rests, through, turned = {}, [], {}
  for late_s in (0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.4, 2.6, 2.7, 2.8, 2.9):
    sim = roll_sim(lambda sim: [(JUNCTION + 80. - 12. * (sim.t - t_go - late_s), 3.5, 'closing')] if sim.t >= t_go + late_s else [])  # noqa: B023
    sim.accel = lagged(ROLL_STOP_LAG_S)
    rows = drive(sim)
    stand = [s for _, s, v, _, _ in rows if v < 0.3 and JUNCTION - 15. < s < JUNCTION + 8.]
    if stand:
      rests[late_s] = round(JUNCTION - max(stand), 2)
    else:
      through.append(late_s)
    turned[late_s] = sim.s >= JUNCTION + 30.
  assert rests and through, (rests, through)
  assert all(d >= HELD_MIN_M for d in rests.values()), rests
  assert max(rests) < min(through), (rests, through)  # withdrawn while the stop fits, carried through from there on
  assert all(turned.values()), turned


def gapped(cars, t0, gap_s):
  """runs(cars) with no road run at all for gap_s from t0."""
  frame = [0]

  def raw(sim):
    frame[0] += 1
    if frame[0] % 2 or t0 <= sim.t < t0 + gap_s:
      return None
    return RawFrame([box(sim, s, y, m) for s, y, m in cars(sim) if -5. < s - sim.s < 400.], 'road', sim.t)
  return raw


@pytest.mark.parametrize('gap_s', [0., 0.9], ids=['everyRun', 'noRunFor0.9s'])
@pytest.mark.parametrize('v_car, r0', [(12., 60.), (17., 80.), (20., 100.)])
def test_a_rolling_go_whose_road_view_goes_stale_while_it_can_still_stop_is_withdrawn(v_car, r0, gap_s):
  # an oncoming car comes into view r0 m up the lane 0.75 s before the rolling go's last point to stop, and the detector
  # skips gap_s of road runs from that moment: the first run that shows it comes after that point. A view gone stale is no
  # clear band: the go is withdrawn while the car can stop, as it is for the car seen on every run, and the left is never
  # in the oncoming lane as that car reaches the junction
  empty = drive(roll_sim())
  t_go = next(t for t, *_, d in empty if d.rolled)
  t_seen = max(t for t, *_, d in empty if d is not None and d.rolled and d.state == 'go' and d.withdrawable) - 0.75
  assert t_seen > t_go + 0.5

  def pos(t):
    return r0 - v_car * (t - t_seen)

  def cars(sim):
    return [(JUNCTION + pos(sim.t), 3.5, 'closing')] if sim.t >= t_seen else []
  sim = roll_sim(cars)
  sim.raw = gapped(cars, t_seen, gap_s)
  rows = drive(sim)
  in_lane = [(round(t - t_go, 2), round(s - JUNCTION, 1)) for t, s, _, _, _ in rows
             if JUNCTION - ONCOMING_ENTRY_M < s < JUNCTION + CLEAR_PATH_PAST_M and t >= t_seen and -2. < pos(t) < 5.]
  assert not in_lane, in_lane[:3]
  assert rolled(rows) and stood(rows) and sim.s >= JUNCTION + 30.
  stand = [s for _, s, v, _, _ in rows if v < 0.3 and s < JUNCTION]
  assert max(stand) <= JUNCTION - HELD_MIN_M
  if gap_s:
    assert any(p.reason == WITHDRAWN_REASON['stale'] for *_, p, _ in rows)
  assert not any(str(p.controlPrompt) in TAKE_CONTROL_PROMPTS for *_, p, _ in rows)


def meets_in_lane(rows, pos, t_from):
  """The frames the left is in the oncoming lane (ONCOMING_ENTRY_M before the maneuver point to CLEAR_PATH_PAST_M past it)
  with the oncoming car pos(t) m past the junction within 5 m short of it to 2 m through."""
  return [(round(t, 2), round(s - JUNCTION, 1)) for t, s, _, _, _ in rows
          if JUNCTION - ONCOMING_ENTRY_M < s < JUNCTION + CLEAR_PATH_PAST_M and t >= t_from and -2. < pos(t) < 5.]


@pytest.mark.parametrize('gap_s', [0., 0.8], ids=['everyRun', 'noRunFor0.8s'])
@pytest.mark.parametrize('v_car, r0', [(12., 60.), (17., 80.), (20., 100.)])
def test_on_the_cars_full_pass_a_25_mph_left_never_rolls_so_a_car_from_past_its_reach_finds_it_at_its_line(v_car, r0, gap_s):
  # an oncoming car r0 m up the lane 2 s after where an empty road's roll would go (113-140 m from the car at that go: past
  # the 98 m the full pass boxes), first boxed then, with or without a detector gap from that moment. A roll's go comes 4-5 s
  # before the junction and can be withdrawn for under two of them, so that car would meet the left in its lane. The roll
  # needs its whole exposure's range seen (roll_why 'range': 179 m here): the car brakes to its line and waits for it
  t_go = next(t for t, *_, d in drive(roll_sim()) if d.rolled)

  def pos(t):
    return r0 - v_car * (t - t_go - 2.)

  def cars(sim):
    return [(JUNCTION + pos(sim.t), 3.5, 'closing')] if sim.t >= t_go + 2. else []
  sim = roll_sim(cars)
  sim.tp.full_pass_range_m = FULL_PASS_RANGE_M
  sim.raw = gapped(cars, t_go + 2., gap_s)
  rows = drive(sim)
  assert not meets_in_lane(rows, pos, t_go)
  assert not rolled(rows) and 'range' in {d.roll for *_, d in rows if d is not None}
  stand = [s for _, s, v, _, _ in rows if v < 0.3 and s < JUNCTION]
  assert stand and all(LINE - 1.5 < s <= LINE + 0.2 for s in stand)
  assert sim.s >= JUNCTION + 30. and not any(str(p.controlPrompt) in TAKE_CONTROL_PROMPTS for *_, p, _ in rows)


@pytest.mark.parametrize('ahead_m', [100., 150., 185., 200., 240., 300.])
def test_a_roll_on_a_pass_that_covers_its_range_is_never_met_in_its_lane_by_a_car_that_pass_shows(ahead_m):
  # the pass covering 210 m (past the 179 m the roll needs seen): a car at SPEED_MARGIN x the limit ahead_m from ours where
  # an empty road's roll would go, boxed on every run it is within the pass's range. Seen in the band it holds the roll, the
  # car at its line; past the range at the go it arrives after the left is through. Either way never with the left in its lane
  t_go, s_go = next((t, s) for t, s, _, _, d in drive(roll_sim()) if d.rolled)

  def pos(t):
    return s_go + ahead_m - SPEED_MARGIN * V_25 * (t - t_go) - JUNCTION

  def cars(sim):
    return [(JUNCTION + pos(sim.t), 3.5, 'closing')] if JUNCTION + pos(sim.t) - sim.s <= TEST_RANGE_M else []
  rows = drive(roll_sim(cars))
  assert not meets_in_lane(rows, pos, 0.)
  assert rolled(rows) == (ahead_m > 235.), "a car inside R_req at the go is in the band: no roll"


def test_the_late_seen_car_of_the_lagged_plant_finds_no_roll_on_the_cars_full_pass():
  # the car answering ROLL_STOP_LAG_S late and an oncoming car first boxed 80 m up the lane 2.6-2.9 s after where an empty
  # road's roll would go (past the full pass's 98 m at that go): no roll on that pass, so nothing to withdraw late: the car
  # stands at its line, never in the lanes, until the car has passed
  base = roll_sim()
  base.accel = lagged(ROLL_STOP_LAG_S)
  t_go = next(t for t, *_, d in drive(base) if d is not None and d.rolled)
  for late_s in (2.6, 2.7, 2.8, 2.9):
    def pos(t, late_s=late_s):
      return 80. - 12. * (t - t_go - late_s)
    sim = roll_sim(lambda sim, late_s=late_s, pos=pos: [(JUNCTION + pos(sim.t), 3.5, 'closing')] if sim.t >= t_go + late_s else [])
    sim.tp.full_pass_range_m = FULL_PASS_RANGE_M
    sim.accel = lagged(ROLL_STOP_LAG_S)
    rows = drive(sim)
    assert not rolled(rows) and not meets_in_lane(rows, pos, t_go), late_s
    stand = [JUNCTION - s for _, s, v, _, _ in rows if v < 0.3 and JUNCTION - 15. < s < JUNCTION + 8.]
    assert stand and min(stand) >= HELD_MIN_M and sim.s >= JUNCTION + 30., (late_s, min(stand, default=None))


def far_side(edge_m, line_m=None, edge_std=0.3, flare=None):
  """(sim) -> the Sim's synthetic modelV2 with the road's far side drawn to the car's left: the left road edge edge_m out
  (flare (s, m): m out from road position s on) at edge_std, and with line_m a lane line the model is sure of there (the
  synthetic road() draws its left edge at the car's own lane line: no oncoming side at all)."""
  def model(sim):
    m = model_v2(*sim.model_lanes, big=sim.big, holds=sim.model_holds, **sim.model_kw).as_builder()
    m.roadEdges[0].y = [-(flare[1] if flare is not None and sim.s + x >= flare[0] else edge_m) for x in m.roadEdges[0].x]
    m.roadEdgeStds = [edge_std, m.roadEdgeStds[1]]
    if line_m is not None:
      m.laneLines[0].y = [-line_m] * len(m.laneLines[0].x)
      m.laneLineProbs = [0.9] + list(m.laneLineProbs)[1:]
    return m.as_reader()
  return model


def test_a_car_parked_at_the_far_curb_and_a_car_ahead_going_away_dont_stop_the_roll():
  # one lane each way and a parking strip (the model's far road edge 7.4 m out): a car standing 8 m to the left, 30 m past
  # the junction (beyond the oncoming lane's far edge), and a car 40 m ahead in our own direction's lane to the right,
  # going away: neither is oncoming traffic
  sim = roll_sim(lambda sim: [(JUNCTION + 30., 8., 'standing'), (sim.s + 40., -0.5, 'receding')])
  sim.model = far_side(7.4)
  rows = drive(sim)
  assert rolled(rows) and not stood(rows) and sim.s >= JUNCTION + 30.


FAR_SIDES = {'twoLanesDrawn': far_side(9.0), 'flarePastTheJunction': far_side(7.4, flare=(JUNCTION + 5., 9.0)),
             'edgeNotTrusted': far_side(7.4, edge_std=2.), 'noFarSideDrawn': None}


@pytest.mark.parametrize('lag_s', [0.6, 1.5, 3.0])
@pytest.mark.parametrize('model', list(FAR_SIDES))
def test_a_car_standing_in_an_oncoming_lane_the_map_doesnt_count_holds_the_left(model, lag_s):
  # navd maps one opposing lane; the road has two. A car stands at the head of the second, 7.2 m left of our axis and 25 m
  # past the junction, in view on every run, launches at 2.5 m/s^2 two seconds after an empty road's rolling go, and the
  # tracker reads it standing lag_s longer. The map's one lane alone is no far edge: the cameras don't show a strip beyond
  # it (two lanes drawn, a flare past the junction, an edge the model doesn't trust, no far side at all), so the car
  # standing there holds the left from the start, and the left is never in that lane as it comes through
  t_go = next(t for t, *_, d in drive(roll_sim()) if d.rolled)

  def pos(t):
    return 25. - 0.5 * 2.5 * max(t - t_go - 2., 0.) ** 2

  def cars(sim):
    return [(JUNCTION + pos(sim.t), 7.2, 'standing' if sim.t < t_go + 2. + lag_s else 'closing')] if pos(sim.t) > -10. else []
  sim = roll_sim(cars)
  sim.model = FAR_SIDES[model]
  rows = drive(sim, limit_s=60.)
  lane = LANE_W_M  # the second lane lies a lane further along the turn's path
  in_lane = [(round(t - t_go, 2), round(s - JUNCTION, 1), round(v, 1)) for t, s, v, _, _ in rows
             if JUNCTION - ONCOMING_ENTRY_M + lane < s < JUNCTION + CLEAR_PATH_PAST_M + lane and -2. < pos(t) < 5.]
  assert not in_lane, in_lane[:2]
  assert not rolled(rows) and stood(rows) and sim.s >= JUNCTION + 30.
  assert not any(p.desire == Desire.turnLeft for t, _, _, p, _ in rows if pos(t) > 0.)


def test_the_models_far_side_is_read_past_the_junction_where_it_trusts_its_edge():
  tp = roll_sim().tp
  sim = types.SimpleNamespace(model_lanes=(0, 2), big=False, model_holds=False, model_kw={}, s=JUNCTION - 30.)
  # the road edge 7.4 m to the left all the way: read 7.4 from 30 m out, no line beyond the car's own
  assert tp.left_far_side(far_side(7.4)(sim), LEFT, 30.) == (pytest.approx(7.4), None)
  # a flare 5 m past the junction (9.0 m from there on): its furthest out past the maneuver point counts
  assert tp.left_far_side(far_side(7.4, flare=(JUNCTION + 5., 9.0))(sim), LEFT, 30.)[0] == pytest.approx(9.0)
  # a line the model is sure of beyond the car's own: its lateral too
  assert tp.left_far_side(far_side(7.5, line_m=4.5)(sim), LEFT, 30.) == (pytest.approx(7.5), pytest.approx(4.5))
  # an edge the model doesn't trust (the lane planner's EDGE_STD_MAX), no model, or the junction further than the edge is read
  assert tp.left_far_side(far_side(7.4, edge_std=EDGE_STD_MAX)(sim), LEFT, 30.) == (None, None)
  assert tp.left_far_side(None, LEFT, 30.) == (None, None)
  assert tp.left_far_side(far_side(7.4)(sim), LEFT, FAR_EDGE_AHEAD_M + 1.) == (None, None)
  # the other side's turn (left-hand traffic) reads the right edge: the synthetic road's, two lanes and a shoulder out
  assert tp.left_far_side(far_side(7.4)(sim), RIGHT, 30.)[0] == pytest.approx(1.8 + 2 * 3.6 + 0.5)


@pytest.mark.parametrize('motion', ['standing', 'unknown', 'closing'])
def test_a_car_standing_in_the_oncoming_lane_holds_the_left_whatever_its_motion_reads(motion):
  # in the oncoming lane itself (3.5 m left) 30 m past the junction for the whole approach, gone 12 s after the car is at its line
  empty = drive(roll_sim())
  t_go = next(t for t, *_, d in empty if d.rolled)
  sim = roll_sim(lambda sim: [(JUNCTION + 30., 3.5, motion)] if sim.t < t_go + 20. else [])
  rows = drive(sim)
  assert not rolled(rows) and stood(rows)
  assert not any(p.desire == Desire.turnLeft for t, *_, p, _ in rows if t < t_go + 20.)


def signal_roll_sim(state, cars=lambda sim: ()):
  """roll_sim with a signal mapped at the junction's line; state(sim) -> the light an accepted source (the sim oracle) reads
  on every frame, None for nothing read."""
  line = JUNCTION - 8.5
  sim = roll_sim(cars, controls=[(line, 'signal')])
  sim.tp.oracle_ok = True

  def lights(sim):
    st = state(sim)
    msg = messaging.new_message('turnpilotLights')
    msg.turnpilotLights.source = 'simOracle'
    if st is not None:
      msg.turnpilotLights.lights = [{'state': st, 'distance': line - sim.s, 'instance': 'trafficLight 1'}]
    return msg.as_reader().turnpilotLights
  sim.lights = lights
  return sim


def camera_roll_sim(read):
  """roll_sim with a signal mapped at the junction's line and no accepted light source, as on the car: the camera's reader
  (add-only) reading the head 15 m past the junction; read(sim) -> the state of a road run's read made this frame, None for
  no read this frame (turnpilotd carries the last one up to CAMERA_STALE_S)."""
  sim = roll_sim(controls=[(JUNCTION - 8.5, 'signal')])
  sim.tp.oracle_ok, sim.tp.camera_mode = False, LIGHT_MODE_DEFAULT
  none = messaging.new_message('turnpilotLights').as_reader().turnpilotLights

  def lights(sim):
    state = read(sim)
    if state is not None:
      sim.sm.put('turnpilotDetections', light_detections(state, JUNCTION + 15. - sim.s, shadow=False))
    return none
  sim.lights = lights
  return sim


def every(period_s, state, t0=0.):
  """A camera read of state every period_s from t0."""
  last = [-math.inf]

  def read(sim):
    if sim.t >= t0 and sim.t - last[0] >= period_s - 1e-6:
      last[0] = sim.t
      return state(sim) if callable(state) else state
    return None
  return read


def test_the_cameras_green_rolls_a_signal_on_three_road_runs_reading_it_never_on_one_read_carried():
  # read green on every road run of the approach (5 Hz): rolled
  rows = drive(camera_roll_sim(every(0.2, 'green')))
  assert rolled(rows) and not stood(rows)
  t_ask = next(t for t, *_, d in rows if d.roll == 'go')
  # one green read 0.3 s before that, and nothing else read: carried for a second it is still one read, never a steady green
  sim = camera_roll_sim(every(1e9, 'green', t0=t_ask - 0.3))
  rows = drive(sim)
  assert not rolled(rows) and stood(rows) and sim.s >= JUNCTION + 30.
  # two reads are not three
  hits = [t_ask - 1.2, t_ask - 0.2]
  rows = drive(camera_roll_sim(lambda sim: 'green' if any(abs(sim.t - h) < DT_MDL / 2. for h in hits) else None))
  assert not rolled(rows)


def test_a_red_read_between_the_cameras_greens_starts_them_over():
  rows = drive(camera_roll_sim(every(0.2, 'green')))
  t_ask = next(t for t, *_, d in rows if d.roll == 'go')
  # green on every run but one red read 0.4 s before the roll could be asked: no three greens since it by the last point
  seq = every(0.2, lambda sim: 'red' if abs(sim.t - (t_ask - 0.4)) < 0.1 else 'green')
  rows = drive(camera_roll_sim(seq), limit_s=60.)
  assert not rolled(rows)
  assert not any(p.desire == Desire.turnLeft for t, _, _, p, _ in rows if t < t_ask + 1.)


def test_a_signal_nobody_reads_is_never_rolled():
  # a mapped signal with no colour read: the left rolls up to the junction's mouth and is judged standing there, as before
  sim = signal_roll_sim(lambda sim: None)
  rows = drive(sim)
  assert not rolled(rows) and stood(rows)


def test_a_signal_read_green_is_rolled():
  sim = signal_roll_sim(lambda sim: 'green')
  rows = drive(sim)
  assert rolled(rows) and not stood(rows) and sim.s >= JUNCTION + 30.
  assert min(v for _, s, v, _, _ in rows if s < JUNCTION + 5.) >= V_CORNER - ROLL_MARGIN_V


def test_a_signal_read_red_is_never_rolled():
  sim = signal_roll_sim(lambda sim: 'red')
  rows = drive(sim, limit_s=60.)
  assert not rolled(rows) and stood(rows) and sim.s < JUNCTION
  assert not any(p.desire == Desire.turnLeft for *_, p, _ in rows)


def test_a_green_read_only_as_the_car_reaches_the_rolls_last_point_isnt_rolled():
  # nothing read until 0.5 s before a rolling go could last be given, green from there: not CAMERA_GREEN_STEADY_S of green
  # by then, so the car goes on to the junction's mouth and is judged there
  seen = []
  sim = signal_roll_sim(lambda sim: None)
  while sim.s < JUNCTION and sim.t < 60. and (not seen or seen[-1][1] != 'done'):
    sim.step(engaged=True)
    seen.append((sim.t, sim.tp.left_now.roll))
  t_done = seen[-1][0]
  sim = signal_roll_sim(lambda sim: 'green' if sim.t >= t_done - 0.5 else None)
  rows = drive(sim)
  assert not rolled(rows) and sim.s >= JUNCTION + 30.


def test_a_red_read_after_a_rolling_go_stops_the_car():
  # green through the go, red 0.4 s after it: the go is withdrawn while the car can stop, and it stands for the green
  first = drive(signal_roll_sim(lambda sim: 'green'))
  t_go = next(t for t, *_, d in first if d.rolled)
  sim = signal_roll_sim(lambda sim: 'green' if sim.t < t_go + 0.4 or sim.t > t_go + 20. else 'red')
  rows = drive(sim)
  assert rolled(rows) and stood(rows)
  stand = [s for t, s, v, _, _ in rows if v < 0.3 and s < JUNCTION and t < t_go + 20.]
  assert stand and max(stand) <= JUNCTION - ONCOMING_ENTRY_M
  assert not any(p.desire == Desire.turnLeft for t, _, _, p, _ in rows if t_go + 2. < t < t_go + 20.)


def late_red_sim(d_red, red_s=12.):
  """roll_sim (no signal mapped at the junction) with the camera's reader acting and no accepted source, as on the car: no
  head boxed until the car is d_red m from the maneuver point, from then a head 15 m past the junction read on every road
  run, red for red_s and green after."""
  sim = roll_sim()
  sim.tp.oracle_ok, sim.tp.camera_mode = False, LIGHT_MODE_DEFAULT
  none = messaging.new_message('turnpilotLights').as_reader().turnpilotLights
  sim.t_red = None

  def lights(sim):
    if sim.t_red is None and JUNCTION - sim.s <= d_red:
      sim.t_red = sim.t
    if sim.t_red is not None and int(round(sim.t / DT_MDL)) % 2 == 0:
      state = 'red' if sim.t < sim.t_red + red_s else 'green'
      sim.sm.put('turnpilotDetections', light_detections(state, JUNCTION + 15. - sim.s, shadow=False))
    return none
  sim.lights = lights
  return sim


@pytest.mark.parametrize('d_red', [12.6, 12.4, 12.3])
def test_a_red_first_read_late_in_a_rolling_go_stops_the_left_while_a_firm_stop_still_ends_short_of_the_oncoming_lanes(d_red):
  # 4.9 m/s, d_red m from the maneuver point: past where the comfortable curve stops the car short of the oncoming lanes
  # (the go's own withdrawal) and past where map_controls takes up a stop for a red at a junction with no signal mapped, but
  # the red's own stop (RED_LAG_S, then RED_DECEL) still ends HELD_MIN_M short of the junction: the red stops the left there
  sim = late_red_sim(d_red)
  rows = drive(sim, limit_s=80.)
  red = [(t, s, v, p) for t, s, v, p, _ in rows if sim.t_red is not None and sim.t_red <= t < sim.t_red + 12.]
  room = JUNCTION - red[0][1] - HELD_MIN_M
  assert red[0][2] ** 2 / (2. * STOP_DECEL) + red[0][2] * ROLL_STOP_LAG_S + VETO_MARGIN_M > room  # past the go's own withdrawal
  assert red_stop_m(red[0][2]) + VETO_MARGIN_M < room
  assert max(s for _, s, _, _ in red) <= JUNCTION - HELD_MIN_M, JUNCTION - max(s for _, s, _, _ in red)
  assert min(v for *_, v, _ in red) < 0.3 and not any(p.desire == Desire.turnLeft for t, _, _, p in red if t > sim.t_red + 1.)
  decel = min((b[2] - a[2]) / DT_MDL for a, b in zip(rows, rows[1:], strict=False) if b[1] < JUNCTION)
  assert decel >= -(mc.RED_DECEL + 0.05), decel
  # on the green it goes from there, a window after, never handed back
  assert sim.s >= JUNCTION + 30. and not any(str(p.controlPrompt) in TAKE_CONTROL_PROMPTS for *_, p, _ in rows)


@pytest.mark.parametrize('d_red', [11.9, 10.4, 9.0, 6.0])
def test_a_red_first_read_past_where_a_firm_stop_ends_short_of_the_oncoming_lanes_never_stops_the_left_in_them(d_red):
  # the same red first read d_red m out: no stop of the red's own ends HELD_MIN_M short of the junction with VETO_MARGIN_M
  # to spare any more, so the left finishes, never braked to a stand in the oncoming lanes
  sim = late_red_sim(d_red)
  rows = drive(sim)
  v0 = next(v for t, _, v, _, _ in rows if t >= sim.t_red)
  assert red_stop_m(v0) + VETO_MARGIN_M > d_red - HELD_MIN_M - 0.3
  assert sim.s >= JUNCTION + 30. and not any(v < 0.3 for _, s, v, _, _ in rows if JUNCTION - 15. < s < JUNCTION + 8.)


@pytest.mark.parametrize('lag_s', [0.3, ROLL_STOP_LAG_S])
def test_a_late_red_never_rests_a_car_that_answers_late_nearer_than_its_next_go_is_judged_from(lag_s):
  # the car answering each command lag_s late, the red first read anywhere from 13 m to 10.4 m out: stopped by map_controls'
  # stop, the go's comfortable withdrawal or the red's firm stop, it rests HELD_MIN_M or more short of the junction (nearer
  # is the position fault's handback, 0.5 m further the oncoming lanes); past the last of them it finishes
  rests, through = {}, []
  for k in range(14):
    d_red = 13. - 0.2 * k
    sim = late_red_sim(d_red)
    sim.accel = lagged(lag_s)
    rows = drive(sim, limit_s=80.)
    stand = [JUNCTION - s for t, s, v, _, _ in rows if sim.t_red <= t < sim.t_red + 12. and v < 0.3 and JUNCTION - 15. < s < JUNCTION + 8.]
    if stand:
      rests[round(d_red, 1)] = round(min(stand), 2)
    else:
      through.append(round(d_red, 1))
    assert not any(str(p.controlPrompt) in TAKE_CONTROL_PROMPTS for *_, p, _ in rows), d_red
  assert rests and through and min(rests) > max(through), (rests, through)
  assert all(d >= HELD_MIN_M for d in rests.values()), rests


def test_no_lamp_no_roll():
  # the driver never lights the lamp: the left stops at its line first (and goes unsignalled after its wait, as before)
  sim = roll_sim(lamp=False)
  rows = drive(sim)
  assert not rolled(rows) and stood(rows)


def test_a_person_in_the_path_is_never_rolled_past():
  # a pedestrian standing in the car's path at the junction's near side: the scene's hold, no rolling go, the car stops
  def objects(sim):
    return [{'trackId': 7, 'cls': 'pedestrian', 'x': JUNCTION - 6. - sim.s, 'y': 0.2, 'vx': 0., 'vy': 0., 'age': 30, 'conf': 0.8,
             'inPath': True}] if 0. < JUNCTION - 6. - sim.s < 60. else []
  sim = roll_sim()
  rows = []
  while sim.s < JUNCTION + 30. and sim.t < 60.:
    plan = sim.step(objects=objects(sim), engaged=True)
    rows.append((sim.t, sim.s, sim.v, plan, sim.tp.left_now))
  assert not rolled(rows) and sim.s < JUNCTION - 6.
  assert not any(p.desire == Desire.turnLeft for *_, p, _ in rows)


def test_the_public_builds_window_from_the_stand_never_rolls(left_window_from_its_stand):
  sim = roll_sim()
  sim.tp.left.window_from_approach = False
  rows = drive(sim)
  assert not rolled(rows) and stood(rows) and sim.s >= JUNCTION + 30.


def test_a_rolling_go_the_lines_stop_doesnt_let_go_is_refused_and_the_left_stops_first(monkeypatch):
  # map_controls keeping its stop through the go (as it does without the release): the go is taken back the next frame, no
  # turn is pulsed on the approach, and the car stands at its line and goes from there as any left
  approach = mc.MapControlPolicy._approach

  def keeps(self, c, controls, v, gas_pressed, gas_lift, out, prompts):
    if c.kind == mc.UNPROTECTED and not c.held:
      saved, self._release_req = getattr(self, '_release_req', None), None
      try:
        return approach(self, c, controls, v, gas_pressed, gas_lift, out, prompts)
      finally:
        self._release_req = saved
    return approach(self, c, controls, v, gas_pressed, gas_lift, out, prompts)
  monkeypatch.setattr(mc.MapControlPolicy, '_approach', keeps)
  sim = roll_sim()
  rows = drive(sim)
  assert not rolled(rows) and stood(rows) and sim.s >= JUNCTION + 30.
  first = next(i for i, r in enumerate(rows) if r[3].desire == Desire.turnLeft)
  assert rows[first - 1][2] < 0.3 and math.isclose(rows[first][1], LINE, abs_tol=1.5)


def test_a_red_or_yellow_head_read_beside_the_greens_is_no_green_to_roll_on(monkeypatch):
  # a red left arrow beside green balls: any red or yellow read at the junction, by any source, and the roll has no green
  tp = camera_roll_sim(lambda sim: None).tp
  monkeypatch.setattr(tp.map_controls, 'signal_known', lambda maneuver_id, distance: True)
  for accepted, camera in ((set(), {'green', 'red'}), ({'green'}, {'red'}), ({'red'}, {'green'}), (set(), {'green', 'yellow'})):
    monkeypatch.setattr(tp.map_controls, 'junction_lights', lambda distance, a=accepted, c=camera: (a, c))
    assert tp.left_roll_light(1, 50.)[0] == 'unread', (accepted, camera)
  monkeypatch.setattr(tp.map_controls, 'junction_lights', lambda distance: ({'green'}, set()))
  assert tp.left_roll_light(1, 50.)[0] == 'green'


def go_sounds(seen_s: float):
  """The rolling go withdrawn a frame after it for a car first boxed 98 m up and seen seen_s, the car passing, the left going
  again: [(s after the first go, banner, sound)] of each leftGo spell, as selfdrived builds the alert from the plan (None,
  None if they change while it is up)."""
  from openpilot.selfdrive.selfdrived.events import turnpilot_control_prompt_alert
  first = {'t': None}

  def cars(sim):
    left = sim.tp.left_now
    if first['t'] is None and left is not None and left.rolled:
      first['t'] = sim.t
    if first['t'] is None or sim.t - first['t'] > seen_s:
      return []
    return [(JUNCTION + 98. - V_25 * (sim.t - first['t']), 3.5, 'closing')]
  sim = roll_sim(cars)
  rows = drive(sim, limit_s=80.)
  assert sim.s >= JUNCTION + 30.
  out = []
  for (t, _, _, p, _), (_, _, _, q, _) in zip(rows[1:], rows, strict=False):
    if str(p.controlPrompt) != 'leftGo':
      continue
    alert = turnpilot_control_prompt_alert(None, None, _PlanSM(p), False, 0, 0)
    if str(q.controlPrompt) != 'leftGo':
      out.append((round(t - first['t'], 2), alert.alert_text_1, alert.audible_alert))
    elif (alert.alert_text_1, alert.audible_alert) != out[-1][1:]:
      out[-1] = (out[-1][0], None, None)  # the go's banner or sound changed while it was up
  return out


class _PlanSM(dict):
  """selfdrived's SubMaster as hooks.fresh_plan reads it: this plan, received now and valid."""
  def __init__(self, plan):
    import time
    super().__init__(turnpilotPlan=plan)
    self.valid, self.recv_time = {'turnpilotPlan': True}, {'turnpilotPlan': time.monotonic()}


@pytest.mark.parametrize("seen_s", [0.2, 0.5, 1.0, 3.0])
def test_a_go_again_soon_after_its_withdrawn_go_is_shown_without_a_second_chime(seen_s):
  # a rolling go withdrawn for a car boxed briefly goes again 2.7-5.5 s later: "Turning" both times, chimed only the first
  AudibleAlert = log.SelfdriveState.AudibleAlert
  sounds = go_sounds(seen_s)
  assert len(sounds) == 2 and sounds[1][0] < 8., sounds
  assert [b for _, b, _ in sounds] == ["Turning", "Turning"]
  assert [a for _, _, a in sounds] == [AudibleAlert.prompt, AudibleAlert.none], sounds


def test_a_go_after_the_repeat_window_or_of_another_left_chimes():
  from openpilot.selfdrive.turnpilot.hooks import PROMPT_REPEAT_S
  from openpilot.selfdrive.turnpilot.unprotected_left import UnprotectedLeft
  AudibleAlert = log.SelfdriveState.AudibleAlert
  sounds = go_sounds(PROMPT_REPEAT_S)  # the car boxed the whole window: the left goes again more than 8 s on
  assert len(sounds) == 2 and sounds[1][0] > PROMPT_REPEAT_S and [a for _, _, a in sounds] == [AudibleAlert.prompt] * 2, sounds
  # another maneuver's left starts with no withdrawn go behind it
  ul = UnprotectedLeft()
  ul.maneuver_id, ul.withdrawn_t = 1, 0.
  ul._reset()
  assert ul.withdrawn_t == math.inf and not ul.repeat_go
