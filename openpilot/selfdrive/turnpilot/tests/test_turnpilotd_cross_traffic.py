"""A turn at a junction with no mapped control yields to the cross traffic the cameras track (scene_policy.cross_stops,
turnpilotd.uncontrolled_turn), closed loop on test_turnpilotd's Sim with the shipped settings.

The route turns where its road meets a through road and navd's controls are ok with no stop, yield or signal there (a T's stem
with an unmapped sign, or an uncontrolled junction; navd suspects a stop only where the through road outranks the route's by
two classes). The cross car is the scene's track of it: x m ahead, y m left, moving along the through road."""
import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.hooks import TAKE_CONTROL_PROMPTS
from openpilot.selfdrive.turnpilot.scene_policy import CROSS_CLEAR_S, CROSS_REASON, HOLD_LINE_M
from openpilot.selfdrive.turnpilot.tests.shipped import shipped
from openpilot.selfdrive.turnpilot.tests.synthetic import junction_of
from openpilot.selfdrive.turnpilot.tests.test_turnpilotd import T_LEFT, Sim

JUNCTION = 300.
T_RIGHT = junction_of("right", opposing='none')
LANE = {'left': JUNCTION - 1.8, 'right': JUNCTION + 1.8}  # s of the through road's lane carrying traffic from that side
V_CROSS = 9.     # m/s, 20 mph
SEEN_S = 3.5     # s before it crosses the car's path a cross car is first tracked (32 m up the through road)
BUMPER_M = 1.9   # m the front bumper is ahead of the car's s


def turn(side: str, controls=()):
  sim = shipped(Sim([(JUNCTION, "turn", side, T_LEFT if side == 'left' else T_RIGHT), (1900., "turn", "right", junction_of("right"))],
                    controls_status='ok', controls=controls))
  sim.answers_prompt = True
  return sim


def cross_car(sim, cross_at: float, frm: str, track: int = 7, age: int = 40, v: float = V_CROSS):
  """The scene's track of a car on the through road that crosses the car's path at cross_at, coming from frm; none before
  SEEN_S of it or 1.5 s past it."""
  if not -SEEN_S <= sim.t - cross_at <= 1.5:
    return ()
  sign = 1. if frm == 'left' else -1.
  return ({'cls': 'car', 'x': LANE[frm] - sim.s, 'y': -sign * v * (sim.t - cross_at), 'vx': 0., 'vy': -sign * v,
           'age': age + int((sim.t - cross_at + SEEN_S) * 10.), 'trackId': track, 'conf': 0.8, 'camera': 'wide'},)


def drive(side: str, objects=lambda sim: (), s_end: float = JUNCTION + 20., t_max: float = 60., controls=()):
  """(sim, [(t, s, v, plan)]) of the turn to s_end (or t_max)."""
  sim, rows = turn(side, controls), []
  while sim.s < s_end and sim.t < t_max:
    plan = sim.step(objects(sim), engaged=True)
    rows.append((sim.t, sim.s, sim.v, plan))
  return sim, rows


def bumper_as_it_crosses(rows, cross_at: float, v: float = V_CROSS) -> float:
  """The furthest s of the front bumper while the cross car is within 2 m of the car's axis."""
  return max(s for t, s, _, _ in rows if abs(t - cross_at) <= 2. / v) + BUMPER_M


def meets(side: str, frm: str) -> float:
  """The time a cross car from frm must cross to meet the car in its lane, the car driving the turn with nobody about."""
  _, alone = drive(side)
  return next(t for t, s, _, _ in alone if s + BUMPER_M >= LANE[frm]) + 0.4


@pytest.mark.parametrize("side, frm", [('left', 'left'), ('left', 'right'), ('right', 'left')])
def test_a_tracked_car_on_the_through_road_holds_the_turn_out_of_a_t_stem(side, frm):
  cross_at = meets(side, frm)
  sim, rows = drive(side, lambda sim: cross_car(sim, cross_at, frm))
  edge = JUNCTION - 3.6  # the through road's near edge
  bumper = bumper_as_it_crosses(rows, cross_at)
  assert bumper < edge, f"front bumper {bumper - edge:+.1f} m into the through road as the tracked car crosses"
  held = [(t, s, v, p) for t, s, v, p in rows if str(p.reason) == CROSS_REASON]
  assert held and held[0][0] < cross_at - 2.5  # held from when it is tracked
  assert max(s for _, s, _, _ in held) <= JUNCTION - HOLD_LINE_M + 0.5
  assert min(float(p.accelCap) for _, _, _, p in held) > -2.5 and not any(str(p.controlAlert) == 'hardBrake' for _, _, _, p in rows)
  # it has crossed: the car makes its turn and nothing was handed back
  assert not any(str(p.controlPrompt) in TAKE_CONTROL_PROMPTS for _, _, _, p in rows)
  assert sim.s >= JUNCTION + 20. and sim.t < cross_at + 20., (sim.s, sim.t, str(rows[-1][3].reason))


def test_a_right_is_not_held_for_a_car_coming_along_the_far_lane_from_its_own_side():
  cross_at = meets('right', 'left')
  _, alone = drive('right')
  _, rows = drive('right', lambda sim: cross_car(sim, cross_at, 'right'))
  assert not any(str(p.reason) == CROSS_REASON for _, _, _, p in rows)
  assert [round(s, 3) for _, s, _, _ in rows] == [round(s, 3) for _, s, _, _ in alone]


@pytest.mark.parametrize("side", ['left', 'right'])
def test_a_parked_car_or_one_waiting_on_the_through_road_slows_nothing(side):
  _, alone = drive(side)

  def standing(sim):  # tracked from 40 m out, long enough for its speed to be the tracker's
    d = JUNCTION - sim.s
    return ({'cls': 'car', 'x': LANE['left'] - sim.s, 'y': 12., 'vx': 0., 'vy': 0., 'age': 30 + int(sim.t * 10.), 'trackId': 3, 'conf': 0.8,
             'camera': 'wide'},) if -5. < d < 40. else ()
  _, rows = drive(side, standing)
  assert not any(str(p.reason) == CROSS_REASON for _, _, _, p in rows)
  assert [round(s, 3) for _, s, _, _ in rows] == [round(s, 3) for _, s, _, _ in alone]


@pytest.mark.parametrize("side", ['left', 'right'])
def test_a_minute_of_cross_traffic_stands_the_turn_with_no_handback_then_it_goes_in_the_gap(side):
  t0 = meets(side, 'left')
  crossings = [t0 + 4. * k for k in range(15)]  # a car every 4 s for a minute, from the left

  def traffic(sim):
    return tuple(o for k, at in enumerate(crossings) for o in cross_car(sim, at, 'left', track=10 + k))
  sim, rows = drive(side, traffic, t_max=crossings[-1] + 40.)
  for at in crossings:
    assert bumper_as_it_crosses(rows, at) < JUNCTION - 3.6, at
  waiting = [(t, v, p) for t, _, v, p in rows if t0 <= t <= crossings[-1]]
  assert sum(DT_MDL for _, v, _ in waiting if v < 0.3) > 45.  # it stood through the stream
  assert not any(str(p.controlPrompt) in TAKE_CONTROL_PROMPTS for _, _, _, p in rows)
  assert all(str(p.handback) == 'none' and str(p.fault) == 'none' for _, _, _, p in rows)
  # the last car is past: the stand ends on its own and the turn is made
  t_moving = next(t for t, _, v, _ in rows if t > crossings[-1] and v > 1.)
  assert t_moving < crossings[-1] + 6., t_moving - crossings[-1]
  assert sim.s >= JUNCTION + 20.


def test_a_right_is_not_slowed_by_new_tracks_that_are_not_cross_traffic():
  _, alone = drive('right')

  def oncoming(sim):  # an oncoming car two lanes over, first tracked a second before it is level with the junction
    x = JUNCTION + 12. - 12. * (sim.t - t_pass)
    return ({'cls': 'car', 'x': x - sim.s, 'y': 7.4, 'vx': -12., 'vy': 0., 'age': 1 + int((sim.t - t_pass + 1.) * 10.), 'trackId': 4,
             'conf': 0.8},) if -1. <= sim.t - t_pass <= 2. else ()
  for t_pass in (meets('right', 'left') - 3., meets('right', 'left') - 1.5):
    _, rows = drive('right', oncoming)
    assert not any(str(p.reason) == CROSS_REASON for _, _, _, p in rows), t_pass
    assert [round(s, 3) for _, s, _, _ in rows] == [round(s, 3) for _, s, _, _ in alone], t_pass


def test_a_parked_car_first_tracked_at_the_corner_costs_a_right_a_moment_on_the_curve_and_no_stand():
  # on the through road's left arm, 16 m off, first tracked with the junction 22 m ahead: until its velocity is the tracker's
  # (VEL_TRUST_HITS, 2.5 s) it may be coming, so the right eases off on the comfortable curve; trusted and standing, the hold
  # is gone at once (no clear window after a new track's hold) and the car never stands for it
  _, alone = drive('right')

  def parked(sim):
    d = JUNCTION - sim.s
    if not -5. < d < 22.:
      parked.t0 = None
      return ()
    parked.t0 = sim.t if getattr(parked, 't0', None) is None else parked.t0
    return ({'cls': 'car', 'x': LANE['left'] - sim.s, 'y': 16., 'vx': 0., 'vy': 0., 'age': 1 + int((sim.t - parked.t0) * 10.), 'trackId': 3,
             'conf': 0.8, 'camera': 'wide'},)
  sim, rows = drive('right', parked)
  held = [(t, v, p) for t, _, v, p in rows if str(p.reason) == CROSS_REASON]
  assert held and held[-1][0] - held[0][0] <= 2.5 + 0.1
  assert all(float(p.accelCap) > 0. for _, _, p in held) and min(v for _, _, v, _ in rows) > 1.
  assert not any(p.shouldStop for _, _, _, p in rows)
  t_end = lambda rs: next(t for t, s, _, _ in rs if s >= JUNCTION + 20.)  # noqa: E731
  assert t_end(rows) - t_end(alone) < 4.


def test_a_new_track_on_the_through_road_is_taken_as_closing_and_held_for_on_the_curve():
  # first tracked SEEN_S before it crosses, its hits from one and its velocity read as nothing: not the tracker's until
  # VEL_TRUST_HITS, so it is taken as coming, and the hold is the comfortable curve's alone
  cross_at = meets('left', 'left')
  _, rows = drive('left', lambda sim: tuple(dict(o, vy=0.) if o['age'] < 25 else o for o in cross_car(sim, cross_at, 'left', age=1)))
  assert bumper_as_it_crosses(rows, cross_at) < JUNCTION - 3.6
  early = [p for t, _, _, p in rows if cross_at - SEEN_S + 0.1 < t < cross_at - 1.2]
  # (inside ONCOMING_Y_FAR_M the oncoming hold takes a new track the same way, under its own words)
  assert early and all(str(p.reason) in (CROSS_REASON, "waiting for oncoming") and float(p.accelCap) > 0. for p in early)
  assert str(early[0].reason) == CROSS_REASON


def test_the_hold_lets_go_a_clear_window_after_the_last_car_and_an_empty_road_never_holds():
  cross_at = meets('left', 'left')
  _, rows = drive('left', lambda sim: cross_car(sim, cross_at, 'left'))
  last = max(t for t, _, _, p in rows if str(p.reason) == CROSS_REASON)
  assert CROSS_CLEAR_S - 0.2 < last - cross_at < CROSS_CLEAR_S + 0.3  # tracked as closing until it is at the car's path
  _, alone = drive('left')
  assert not any(str(p.reason) == CROSS_REASON for _, _, _, p in alone)


def test_a_junction_with_a_mapped_control_or_a_light_head_read_is_not_this_rules():
  sim = turn('left')
  for _ in range(40):
    sim.step(engaged=True)
  m, d = sim.tp.follower.upcoming, sim.tp.follower.distance
  assert sim.tp.uncontrolled_turn(m, d)
  assert not sim.tp.uncontrolled_turn(None, d)
  sim.tp.head_seen_m = 10.  # a light head read 10 m back: its cross traffic is the light's to stop
  assert not sim.tp.uncontrolled_turn(m, d)
  # a stop sign mapped at the junction: the stop-sign decider judges its cross traffic
  sim = turn('left', controls=[(JUNCTION - 11., 'stop')])
  for _ in range(40):
    sim.step(engaged=True)
  assert not sim.tp.uncontrolled_turn(sim.tp.follower.upcoming, sim.tp.follower.distance)
