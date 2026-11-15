"""A turn at a junction with no mapped control yields to the cross traffic the cameras track (scene_policy.cross_stops,
turnpilotd.uncontrolled_turn), closed loop on test_turnpilotd's Sim with the shipped settings.

The route turns where its road meets a through road and navd's controls are ok with no stop, yield or signal there (a T's stem
with an unmapped sign, or an uncontrolled junction; navd suspects a stop only where the through road outranks the route's by
two classes). The cross car is the scene's track of it: x m ahead, y m left, moving along the through road."""
import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.hooks import TAKE_CONTROL_PROMPTS
from openpilot.selfdrive.turnpilot.scene_policy import CROSS_CLEAR_S, CROSS_REASON
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


def cross_car(sim, cross_at: float, frm: str, track: int = 7, age: int = 40, v: float = V_CROSS, seen: float = SEEN_S):
  """The scene's track of a car on the through road that crosses the car's path at cross_at, coming from frm; none before
  seen s of it (SEEN_S) or 1.5 s past it."""
  if not -seen <= sim.t - cross_at <= 1.5:
    return ()
  sign = 1. if frm == 'left' else -1.
  return ({'cls': 'car', 'x': LANE[frm] - sim.s, 'y': -sign * v * (sim.t - cross_at), 'vx': 0., 'vy': -sign * v,
           'age': age + int((sim.t - cross_at + seen) * 10.), 'trackId': track, 'conf': 0.8, 'camera': 'wide'},)


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
  # a far-lane car tracked SEEN_S out at the walking car's own arrival is first tracked with a left already in the near lane,
  # committed (committed_past_line): the look's own stop stands it short of the far lane (look_stop, CROSS_FAR_ENTRY_M); until
  # 2026-11-14 the left came at its corner speed and SEEN_S found it 12 m out
  cross_at = meets(side, frm)
  sim, rows = drive(side, lambda sim: cross_car(sim, cross_at, frm))
  # short of the lane the tracked car comes along: the through road's near edge for one from the left, its centre line for one
  # in the far lane from the right. The look (scene_policy) walks the car past the hold line before a car SEEN_S out at the
  # walking car's own arrival is tracked, so the hold stands it where the comfortable curve can, short of that lane (until
  # 2026-11-14 at its hold line, HOLD_LINE_M, reached at the corner speed)
  edge = JUNCTION - 3.6 if frm == 'left' else JUNCTION
  bumper = bumper_as_it_crosses(rows, cross_at)
  assert bumper < edge, f"front bumper {bumper - edge:+.1f} m into the through road as the tracked car crosses"
  # held from when it is tracked: by the hold, or by the look's own stop for it (scene_policy.look_tracks_block), "Look both ways"
  from openpilot.selfdrive.turnpilot.scene_policy import LOOK_REASON
  held = [(t, s, v, p) for t, s, v, p in rows if str(p.reason) == CROSS_REASON or (str(p.reason) == LOOK_REASON and float(p.speedCap) < 1.)]
  assert held and held[0][0] < cross_at - 2.5
  assert frm == 'right' or any(str(p.reason) == CROSS_REASON for *_, p in held)
  assert max(s for _, s, _, _ in held) + BUMPER_M <= edge - 0.25
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
  t_end = lambda rs: next(t for t, s, _, _ in rs if s >= JUNCTION + 20.)  # noqa: E731
  for t_pass in (meets('right', 'left') - 3., meets('right', 'left') - 1.5):
    _, rows = drive('right', oncoming)
    assert not any(str(p.reason) == CROSS_REASON for _, _, _, p in rows), t_pass
    # the look (scene_policy.look_tracks_block) takes a new track 6-13 m off our axis on the cross road's band as maybe crossing
    # until its velocity is the tracker's or it leaves the band: at walking pace 1.3-1.7 s here, at most a moment's stand short
    # of the lane (until 2026-11-14 the trace was the empty road's to the centimetre)
    assert t_end(rows) - t_end(alone) < 2.5, (t_pass, t_end(rows) - t_end(alone))
    assert not any(str(p.controlPrompt) in TAKE_CONTROL_PROMPTS for _, _, _, p in rows)


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


# ------------------------------------------------------------------------- a stop sign's line the driver crossed

NODE_M = 13.   # test_turnpilotd_stop's vinfo stopPointDistance: the junction's node past the line
V_CREEP_X = 7.


def creep_and_reengage(tmp_path, tta: float, creep_m: float = 5., y_new: float | None = None, age: int = 30, trusted: bool = True,
                       seconds: float = 4.):
  """A stop sign under the shipped vision decider (test_turnpilotd_vision.vsim), held at its line 2 s: the driver's gas, openpilot
  off, creeps the car creep_m on at up to 1.5 m/s (map_controls: the latch let go as his, his crossing never said), and he
  engages again with a car crossing from the left tracked on the near lane 1.8 m short of the node, tta s from our path when
  the creep starts (moving at V_CREEP_X; y_new: first published only once it is that near, a new track). Returns (the
  plans of the seconds after the re-engage, the crossing car's (x, y) then, where the car met it or None)."""
  from openpilot.selfdrive.turnpilot.tests.test_turnpilotd_stop import S_LINE, run_to_line
  from openpilot.selfdrive.turnpilot.tests.test_turnpilotd_vision import vsim
  sim = vsim(tmp_path)
  run_to_line(sim)
  for _ in range(int(2. / DT_MDL)):
    sim.step()
  t0, s0 = sim.t, sim.s
  first = {}

  def crossing():
    y = V_CREEP_X * (tta - (sim.t - t0))
    x = S_LINE + NODE_M - 1.8 - sim.s
    if y_new is not None and y > y_new and 'k' not in first:
      return [], (x, y)
    k = first.setdefault('k', sim.k)
    obj = {'trackId': 7, 'cls': 'car', 'x': float(x), 'y': float(y), 'vx': float(-sim.v), 'vy': -V_CREEP_X,
           'age': (age if trusted else 0) + (sim.k - k) // 2, 'inPath': bool(abs(y) < 1.5 and 0. < x < 50.), 'camera': 'wide'}
    return ([obj] if y > -30. else []), (x, y)
  sim.accel = lambda s, plan: 0.8 if s.v < 1.5 else 0.
  while sim.s < s0 + creep_m:
    sim.step(crossing()[0], engaged=False, gas=True)
  sim.accel = sim.planner
  at = crossing()[1]
  plans, contact = [], None
  for _ in range(int(seconds / DT_MDL)):
    objs, (x, y) = crossing()
    plans.append(sim.step(objs, engaged=True))
    if contact is None and -3.6 < x < 3.0 and abs(y) < 1.9 and sim.v > 0.5:
      contact = (round(x, 1), round(y, 1), round(sim.v, 1))
  return plans, at, contact


def test_re_engaged_past_a_stop_line_the_driver_crossed_the_car_holds_for_the_crossing_car_it_tracks(tmp_path):
  # the 4c matrix case: re-engaged 5 m past the line with a car from the left tracked 21 m up the cross road (3.0 s out).
  # The driver's crossing is silent by decision and the decider is done (stop_go._driver), so before 2026-11-14 nothing
  # held: the scene's cross hold was only an uncontrolled junction's (a route turn's), and the car pulled out in front of it
  # at 4 m/s. The junction is now, for its cross traffic, an uncontrolled one (turnpilotd.driven_stop_cross): held short
  # of the lanes, silent about the crossing itself, "waiting for cross traffic" while it holds
  plans, at, contact = creep_and_reengage(tmp_path, tta=7.3, seconds=12.)
  held = [p for p in plans[:40] if p.shouldStop or p.speedCap < 1.]
  assert held and contact is None, (at, contact)
  assert all(p.reason == CROSS_REASON for p in held)
  assert not any(str(p.controlPrompt) in TAKE_CONTROL_PROMPTS for p in plans)
  # and it goes once the car has passed (gone 30 m by, then the clear window): never a stand for good
  assert any(p.speedCap > 5. and not p.shouldStop for p in plans[-40:]), [(p.reason, round(p.speedCap, 1)) for p in plans[-40::10]]


def test_re_engaged_past_a_driven_stop_line_a_car_first_tracked_inside_the_near_band_still_holds(tmp_path):
  # with no side view running the wide pass first boxes a car crossing from the side about 12 m out: a new track the route
  # turn's hold takes for something on our own road (CROSS_Y_NEAR_M). In the junction the car is in, for DRIVEN_DOUBT_S, it
  # holds on the comfortable curve; contact at 3.5 m/s before (the matrix's noroom creep cells)
  plans, at, contact = creep_and_reengage(tmp_path, tta=6.4, y_new=12.5, trusted=False)
  assert any(p.shouldStop or p.speedCap < 1. for p in plans[:40]) and contact is None, (at, contact)


def test_re_engaged_already_at_the_cross_lanes_the_car_carries_on_never_stopping_in_them(tmp_path):
  # creeping 9 m past the line (2 m short of the near lane) at 1.5 m/s: it can no longer stand short of the lanes on the
  # comfortable curve, so it carries on (never a stop in the cross lanes); what gets through, said plainly in the report
  plans, _at, _contact = creep_and_reengage(tmp_path, tta=9., creep_m=9.5)
  assert not any(p.shouldStop for p in plans[:40])


# ------------------------------------------------------------- the look (scene_policy: an uncontrolled turn is a yield)

def test_with_no_side_view_arriving_an_uncontrolled_turn_walks_into_the_cross_lanes_and_says_so():
  # rails 2026-11-14: driven in at the corner speed (a right 3.0 m/s, a left 4.6) whatever its cross traffic, a car on a
  # collision course past the forward passes' view the whole way in. The plain Sim runs no side view (the test car's logged
  # state): the look asks for them, waits for none past CADENCE_FRESH_MAX_S, and walks the car to the cross lanes at LOOK_V on
  # the forward passes, saying so, never standing, nothing handed back
  from openpilot.selfdrive.turnpilot.scene_policy import CROSS_ENTRY_M, LOOK_REASON, LOOK_V, ONCOMING_ENTRY_M
  for side in ('left', 'right'):
    _, rows = drive(side)
    walk = [(s, v, p) for _, s, v, p in rows if JUNCTION - CROSS_ENTRY_M - 1. <= s <= JUNCTION - ONCOMING_ENTRY_M]
    assert walk and max(v for _, v, _ in walk) <= LOOK_V + 0.1 and min(v for _, v, _ in walk) > LOOK_V - 0.3, side
    assert all(str(p.reason) == LOOK_REASON for _, _, p in walk) and any(str(p.detectorView) == 'sideLook' for _, _, p in walk)
    assert not any(p.shouldStop for *_, p in rows) and not any(str(p.controlPrompt) in TAKE_CONTROL_PROMPTS for *_, p in rows)
    assert not any(str(p.controlAlert) not in ('none',) for *_, p in rows if JUNCTION - 30. <= p_s(rows, p) <= JUNCTION)
    assert rows[-1][1] >= JUNCTION + 20.


def p_s(rows, plan) -> float:
  return next(s for _, s, _, p in rows if p is plan)


def test_the_look_is_not_for_a_turn_off_the_road_into_a_side_street_nor_for_a_junction_with_a_control():
  # navd's junction data: the one other road is ours going on straight (opposing present, one branch), so the turn crosses or
  # enters no road with traffic: no look, the turn at its corner speed as before. A mapped stop sign: the stop-sign decider's
  from openpilot.selfdrive.turnpilot.scene_policy import LOOK_REASON
  from openpilot.selfdrive.turnpilot.turn_speed import V_CORNER_CURB
  sim = shipped(Sim([(JUNCTION, "turn", "right", junction_of("right", branches=1)), (1900., "turn", "right", junction_of("right"))],
                    controls_status='ok'))
  rows = []
  while sim.s < JUNCTION + 20. and sim.t < 60.:
    rows.append((sim.t, sim.s, sim.v, sim.step(engaged=True)))
  assert not any(str(p.reason) == LOOK_REASON for *_, p in rows)
  assert min(v for _, s, v, _ in rows if JUNCTION - 10. <= s <= JUNCTION) >= V_CORNER_CURB - 0.05
  _, rows = drive('right', controls=[(JUNCTION - 11., 'stop')])
  assert not any(str(p.reason) == LOOK_REASON for *_, p in rows)


def test_a_car_first_tracked_inside_the_near_band_holds_the_look_short_of_the_lane():
  # at walking pace with no side view the wide pass first boxes a car crossing from the left about 12 m up its road, a new
  # track inside CROSS_Y_NEAR_M that the hold takes for our own road's. The look waits for it (scene_policy.look_tracks_block,
  # from DRIVEN_NEAR_M), on the comfortable curve, and the car stands short of the lane until it has crossed: here first boxed
  # with the car at LOOK_STOP_M, where a stop from walking pace still ends short of the lane
  from openpilot.selfdrive.turnpilot.scene_policy import LOOK_REASON, LOOK_STOP_M, LOOK_V
  _, alone = drive('right')
  cross_at = next(t for t, s, _, _ in alone if s >= JUNCTION - LOOK_STOP_M) + 12. / V_CROSS

  def late(sim):  # boxed from 12 m up the cross road on, a new track
    return tuple(dict(o, age=1 + int((sim.t - cross_at + 12. / V_CROSS) * 10.)) for o in cross_car(sim, cross_at, 'left')
                 if o['y'] <= 12.)
  _, rows = drive('right', late)
  assert bumper_as_it_crosses(rows, cross_at) < JUNCTION - 3.6
  # the look neither ended nor let the car speed up with it tracked near: until 2026-11-14 the hold took no new track inside
  # CROSS_Y_NEAR_M and the look, its side runs clear, ended with the car crossing in front of ours (matrix stand10)
  assert max(v for t, _, v, _ in rows if cross_at - 12. / V_CROSS <= t <= cross_at) <= LOOK_V + 0.1
  assert any(float(p.speedCap) < LOOK_V - 0.2 for t, _, _, p in rows if cross_at - 12. / V_CROSS <= t <= cross_at)
  assert any(str(p.reason) == LOOK_REASON for *_, p in rows)
  assert rows[-1][1] >= JUNCTION + 20.
