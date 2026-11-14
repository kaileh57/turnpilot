"""A walker at a right turn's corner, first tracked (or tracked again) with the car past its hold line and committed
(scene_policy.late_walker_stops), closed loop on test_turnpilotd's Sim with the shipped settings.

She stands 7 m right of the car's axis, outside the road camera's half-view from about 12 m out, and the wide camera runs
only from the turn's pulse (8 m), so her track lapsing as the car crosses the hold line is ordinary. Tracked again she holds
the car while it can still stop short of her; once she is across, the car finishes its turn."""
import math

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.hooks import TAKE_CONTROL_PROMPTS
from openpilot.selfdrive.turnpilot.scene_policy import CROSSWALK_REASON
from openpilot.selfdrive.turnpilot.tests.shipped import shipped
from openpilot.selfdrive.turnpilot.tests.synthetic import junction_of
from openpilot.selfdrive.turnpilot.tests.test_turnpilotd import Sim

JUNCTION = 300.
CORNER = (JUNCTION - 4., -7.)  # where she waits: the near corner, 7 m right of the car's axis (in the turn's crosswalk zone)
WALK = 1.3                     # m/s
Desire = log.Desire


def right_turn():
  sim = shipped(Sim([(JUNCTION, "turn", "right", junction_of("right")), (1900., "turn", "left", junction_of("left"))], controls_status='ok'))
  sim.answers_prompt = True
  return sim


def drive(lapse: bool, t_max: float = 70.):
  """(sim, [(t, s, v, plan, she is within 3 m of the near corner)]) of a right with a walker who stands at the corner and
  steps into the crosswalk of the road the car turns into once the car is 9 m out, across it and gone 13 m on; lapse: her
  track is lost while the car is 12 m to 7.5 m from the turn, and is a new track after."""
  sim = right_turn()
  rows, step_t, seen_t, track_id = [], None, None, 4
  while sim.s < JUNCTION + 15. and sim.t < t_max:
    d = JUNCTION - sim.s
    if step_t is None and d < 9.:
      step_t = sim.t
    walked = WALK * (sim.t - step_t) if step_t is not None else 0.
    tracked = walked < 13. and d < 45. and not (lapse and 7.5 < d < 12.)
    if tracked and seen_t is None:
      seen_t, track_id = sim.t, track_id + 1  # a new track when she is found again, its hits from one (10 a second)
    elif not tracked:
      seen_t = None
    objects = ({'cls': 'pedestrian', 'x': CORNER[0] + walked - sim.s, 'y': CORNER[1], 'vx': WALK if step_t is not None else 0., 'vy': 0.,
                'age': 1 + int((sim.t - seen_t) * 10.), 'trackId': track_id, 'conf': 0.8},) if tracked else ()
    plan = sim.step(objects, engaged=True)
    rows.append((sim.t, sim.s, sim.v, plan, walked < 3.))
  return sim, rows


def into_the_junction_with_her_there(rows) -> float | None:
  """The car's speed entering the junction (2 m before the turn point) while she is within 3 m of the near corner, None if it waits."""
  return next((v for _t, s, v, _p, there in rows if there and s > JUNCTION - 2. and v > 1.), None)


def test_a_walker_tracked_all_the_way_holds_the_right():
  assert into_the_junction_with_her_there(drive(lapse=False)[1]) is None


def test_a_walker_whose_track_lapses_across_the_hold_line_holds_the_right_once_she_is_tracked_again():
  sim, rows = drive(lapse=True)
  back = next((JUNCTION - s, v) for _t, s, v, _p, _there in rows if JUNCTION - s <= 7.5)
  assert back[1] > 2., back  # the premise: over the line faster than COMMIT_SPEED with no hold in force, so committed
  v = into_the_junction_with_her_there(rows)
  assert v is None, f"tracked again {back[0]:.1f} m out at {back[1]:.1f} m/s, in the crosswalk; the car drove into the junction at {v:.1f} m/s"
  held = [(t, s, v, p) for t, s, v, p, _ in rows if JUNCTION - s <= 7.5 and str(p.reason) == CROSSWALK_REASON]
  assert held and min(v for _, _, v, _ in held) < 2.  # slowed for her on the way in
  # in the junction the hold is the comfortable curve's alone (no accel cap, no hard brake said), and nobody was asked to take over
  assert all(float(p.accelCap) > 0. and str(p.controlAlert) != 'hardBrake' for _, _, _, p in held)
  assert not any(str(p.controlPrompt) in TAKE_CONTROL_PROMPTS for _, _, _, p, _ in rows)
  assert sim.s >= JUNCTION + 15.  # she is across: the turn is finished


def test_the_car_stands_for_a_walker_first_tracked_after_its_commit_and_turns_once_she_is_gone():
  # she stands in the mouth of the road the car turns into, 2.5 m right of its axis, seen by no camera until the car is
  # 7.5 m out and committed; she stays 6 s after the car has stopped for her, then is gone
  sim, at, rows, seen_t, rest_t = right_turn(), (JUNCTION + 1., -2.5), [], None, None
  while sim.s < JUNCTION + 15. and sim.t < 70.:
    if rest_t is None and seen_t is not None and sim.v < 0.05:
      rest_t = sim.t
    there = JUNCTION - sim.s <= 7.5 and (rest_t is None or sim.t - rest_t < 6.)
    seen_t = sim.t if there and seen_t is None else seen_t
    objects = ({'cls': 'pedestrian', 'x': at[0] - sim.s, 'y': at[1], 'vx': 0., 'vy': 0., 'age': 1 + int((sim.t - seen_t) * 10.),
                'trackId': 5, 'conf': 0.8},) if there else ()
    plan = sim.step(objects, engaged=True)
    rows.append((sim.t, sim.s, sim.v, plan, there))
  assert sim.tp.scene.committed_id is not None and rest_t is not None
  stood = [(t, s, p) for t, s, v, p, there in rows if there and v < 0.05]
  assert 5.5 < len(stood) * DT_MDL < 6.5
  assert all(p.shouldStop and str(p.reason) == CROSSWALK_REASON for _, _, p in stood[2:])
  assert math.hypot(at[0] - stood[0][1], at[1]) - 1.9 > 2.5  # m from the front bumper to her
  assert not any(str(p.controlPrompt) in TAKE_CONTROL_PROMPTS for _, _, _, p, _ in rows)
  # she is gone: the hold's go pulses the turn again from the stand, and the car drives on
  after = [p for t, _, _, p, _ in rows if t > stood[-1][0]]
  assert any(p.desire == Desire.turnRight for p in after[:int(1. / DT_MDL)])
  assert sim.s >= JUNCTION + 15., (sim.s, sim.v, str(rows[-1][3].reason))


def test_a_walker_who_never_leaves_is_the_standoffs_to_bound_as_at_the_line():
  # she stands in the mouth of the road for good: the stand is said (pedestrianWaiting) and, everyone it is for standing
  # STAND_UNDECIDED_S, handed back held, as the crosswalk hold at the line is (crosswalk_standoff)
  from openpilot.selfdrive.turnpilot.unprotected_left import STAND_UNDECIDED_S
  sim, at, rows, seen_t = right_turn(), (JUNCTION + 1., -2.5), [], None
  while sim.t < 95.:
    there = JUNCTION - sim.s <= 7.5
    seen_t = sim.t if there and seen_t is None else seen_t
    objects = ({'cls': 'pedestrian', 'x': at[0] - sim.s, 'y': at[1], 'vx': 0., 'vy': 0., 'age': 1 + int((sim.t - seen_t) * 10.),
                'trackId': 5, 'conf': 0.8},) if there else ()
    rows.append((sim.t, sim.s, sim.v, sim.step(objects, engaged=True)))
  t_stood = next(t for t, _, v, p in rows if v < 0.05 and p.shouldStop and str(p.reason) == CROSSWALK_REASON)
  stood = [(t, p) for t, _, _, p in rows if t >= t_stood]
  assert all(p.shouldStop for _, p in stood) and rows[-1][2] < 0.05 and sim.s < JUNCTION  # held to the end, short of her
  assert any(str(p.controlAlert) == 'pedestrianWaiting' for _, p in stood)
  handed = [t for t, p in stood if str(p.controlPrompt) == 'pedestrianWaitingTakeControl']
  assert handed and STAND_UNDECIDED_S - 1. < handed[0] - t_stood < STAND_UNDECIDED_S + 1.
