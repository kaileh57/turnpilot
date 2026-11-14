"""turnpilotd's autonomous left, a go from a stand withdrawn by an oncoming car first seen after its first motion, on a car
that answers each command up to ROLL_STOP_LAG_S late: the go from its line, from its full creep for a view and from a
signal's mouth. The withdrawal is taken only where its stop, the lag in it, stands the car where its next go is judged
(unprotected_left.go_can_stop, the rolling go's own rule); past that point the left carries through at its post-go profile,
and the scene's junction holds for the car it tracks yield to it too (ScenePolicy.carry_id). The Sim, route and road runs of
test_turnpilotd_left; the oncoming car is boxed by the road camera and tracked (a confirmed track) from when it is seen."""
from collections import deque

import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.hooks import TAKE_CONTROL_PROMPTS, plan_accel
from openpilot.selfdrive.turnpilot.scene.tracker import RawFrame
from openpilot.selfdrive.turnpilot.scene_policy import ONCOMING_ENTRY_M
from openpilot.selfdrive.turnpilot.tests.test_turnpilotd_left import JUNCTION, car_frame, left_sim, raw_box, signal_sim
from openpilot.selfdrive.turnpilot.unprotected_left import CLEAR_PATH_PAST_M, HELD_MIN_M, ROLL_STOP_LAG_S

ONC_R0 = 60.   # m up the oncoming lane the car is first seen
ONC_V = 12.    # m/s it closes at
LATES = (0., 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6)  # s after the go's first motion it is first seen


def lagged(lag_s):
  """The Sim's own follower (Sim.step) with each plan frame's command applied lag_s later: the actuator's lag."""
  q = deque()

  def accel(sim, plan):
    a = min(plan_accel(plan.accelCap, plan.speedCap, sim.v), (sim.cruise - sim.v) / 2., 1.5)
    q.append(min(a, -1.) if plan.shouldStop else a)
    return q.popleft() if len(q) > round(lag_s / DT_MDL) else 0.
  return accel


def stand_go(kind, lag_s, late_s):
  """A left standing at its line ('line': a stream holds it), at its full creep ('creep': a car parked in the junction box
  hides the band), or in a signal's mouth on a green ('mouth': a stream holds it), until it has stood 3 s; then the road
  clears, it goes, and late_s after its first motion the oncoming car is seen ONC_R0 up the lane. The plant answers lag_s
  late from the stand on. Returns (rows (t, s, v, plan, the oncoming car's s or None), the go's distance, first motion t)."""
  sim = signal_sim(red_until=0.) if kind == 'mouth' else left_sim()
  st = {'go': None, 'move': None, 'n': 0, 'stood': None, 'go_d': None}

  def onc(s):
    return JUNCTION + ONC_R0 - ONC_V * (s.t - st['move'] - late_s) if st['move'] is not None and s.t >= st['move'] + late_s else None

  def blockers(s):
    if st['stood'] is not None and s.t > st['stood'] + 3.:
      return []
    return [('parked', JUNCTION + 6., 3.6)] if kind == 'creep' else [('stream', JUNCTION + 40. - (s.t * 12.) % 80., 3.6)]

  def objects(s):
    out = []
    for k, x, y in blockers(s):
      if k == 'parked':
        ox, oy = car_frame(s, x - s.s, y)
        out.append({'cls': 'car', 'x': ox, 'y': oy, 'vx': 0., 'vy': 0., 'age': 40, 'trackId': 7, 'conf': 0.8, 'camera': 'road'})
    x = onc(s)
    if x is not None:
      ox, oy = car_frame(s, x - s.s, 3.6)
      out.append({'cls': 'car', 'x': ox, 'y': oy, 'vx': -ONC_V, 'vy': 0., 'age': 40, 'trackId': 9, 'conf': 0.8, 'camera': 'road'})
    return out

  def raw(s):
    st['n'] += 1
    if st['n'] % 2:
      return None
    pts = [(x, y) for _, x, y in blockers(s)] + ([(onc(s), 3.6)] if onc(s) is not None else [])
    return RawFrame([raw_box(*car_frame(s, x - s.s, y)) for x, y in pts if -5. < x - s.s < 400.], 'road', s.t)
  sim.raw = raw
  rows = []
  while sim.s < JUNCTION + 30. and sim.t < 90.:
    if st['stood'] is None and sim.v < 0.02 and sim.trace and str(sim.trace[-1][2].unprotectedLeft.state) in ('waitGap', 'creep') \
       and (kind != 'creep' or JUNCTION - sim.s < 5.5) and (kind != 'mouth' or JUNCTION - sim.s < 7.):
      st['stood'] = sim.t
      sim.accel = lagged(lag_s)
    turning = sim.tp.desires.started and abs(JUNCTION - sim.s) < 12. and sim.v > 0.3
    p = sim.step(objects(sim), engaged=True, yaw_rate=sim.v / 8. if turning else 0.)
    if st['stood'] is not None and st['go'] is None and str(p.unprotectedLeft.state) == 'go':
      st['go'], st['go_d'] = sim.t, JUNCTION - sim.s
    if st['go'] is not None and st['move'] is None and sim.v >= 0.3:
      st['move'] = sim.t
    rows.append((sim.t, sim.s, sim.v, p, onc(sim)))
  assert st['move'] is not None, (kind, lag_s, late_s)
  return rows, st['go_d'], st['move']


@pytest.mark.parametrize('lag_s', [0., 0.3, ROLL_STOP_LAG_S])
@pytest.mark.parametrize('kind', ['line', 'creep', 'mouth'])
def test_a_go_from_a_stand_withdrawn_on_a_car_that_answers_late_never_rests_in_the_oncoming_lanes(kind, lag_s):
  """Whenever after the go's first motion the oncoming car is first seen, a withdrawn go rests the car where its next go is
  judged (HELD_MIN_M or more short of the junction), never handed back; one seen past the last point such a stop fits is
  carried through, and that car never meets it in the oncoming lanes. Until the go from a stand took the rolling go's rule
  it was withdrawn wherever v^2 / (2 STOP_DECEL) fitted short of the lanes, no lag, and the hold's stop runs longer than
  that: on the ideal plant it rested 3.69 m from the junction at the line (in the lane of the car it stopped for), 3.64 m
  at the creep and 4.15 m at the mouth, handed back ('position'), and with half a second of lag 1.0-3.5 m out."""
  rests, through = {}, []
  for late_s in LATES:
    rows, go_d, moved = stand_go(kind, lag_s, late_s)
    after = [(s, v) for t, s, v, _, _ in rows if t > moved + late_s + 0.3]
    stand = [JUNCTION - s for s, v in after if v < 0.05 and JUNCTION - 9.5 < s < JUNCTION + 8.]
    prompts = {str(p.controlPrompt) for _, _, _, p, _ in rows} & set(TAKE_CONTROL_PROMPTS)
    assert not prompts, (kind, lag_s, late_s, prompts)
    in_lanes = [(round(t - moved, 2), round(JUNCTION - s, 2)) for t, s, _, _, x in rows
                if x is not None and abs(x - JUNCTION) < 3. and JUNCTION - ONCOMING_ENTRY_M < s < JUNCTION + CLEAR_PATH_PAST_M]
    assert not in_lanes, (kind, lag_s, late_s, in_lanes[:3])
    assert rows[-1][1] > JUNCTION + 20., (kind, lag_s, late_s)  # turned, the withdrawn ones on their next go
    if stand:
      rests[late_s] = round(min(stand), 2)
    else:
      through.append(late_s)
  assert all(d >= HELD_MIN_M for d in rests.values()), (kind, lag_s, rests)
  assert through and (not rests or max(rests) < min(through)), (kind, lag_s, rests, through)  # withdrawn while it fits
  if kind == 'line':
    assert rests, (kind, lag_s)  # a car seen early in the launch from the line still withdraws it


def test_the_scene_yields_to_the_left_on_every_frame_it_carries_its_go_through():
  """turnpilotd hands the scene the left it carries (ScenePolicy.carry_id) on every frame its go is past where its withdrawal
  fits inside the hold line, committed or not yet: never a junction hold stopping the car where the left won't, and on no
  other frame."""
  sim = left_sim()
  rows = []
  while sim.s < JUNCTION + 20. and sim.t < 60.:
    turning = sim.tp.desires.started and abs(JUNCTION - sim.s) < 12. and sim.v > 0.3
    sim.step(engaged=True, yaw_rate=sim.v / 8. if turning else 0.)
    left, scene = sim.tp.left, sim.tp.scene
    rows.append((left.carrying, left.state, scene.carry_id, sim.tp.follower.maneuver_id))
  carried = [r for r in rows if r[0]]
  assert carried and all(cid == mid for _, _, cid, mid in carried)
  assert all(cid is None for c, _, cid, _ in rows if not c)
  assert {st for _, st, _, _ in carried} == {'go', 'committed'}  # from where its withdrawal no longer fits, through its commit


@pytest.mark.parametrize('at_m', [7.2, 6.8])
def test_a_committed_left_a_person_stands_short_of_the_lanes_is_judged_afresh_before_it_goes_on(at_m):
  """The go from the line committed (go_committed), a person steps into the path at_m from the junction and the car stops for
  her (the person rail), short of the oncoming lanes where its withdrawal fits again; an oncoming car comes into view while it
  stands. Committed no more: its own hold, judged afresh, so that car holds it and it goes once the car has passed, by itself,
  never a handback. Until it uncommitted at rest it drove on across the lanes once she had gone, on a band last judged before
  the stand."""
  walker = {'cls': 'pedestrian', 'x': 4., 'y': 0.1, 'vx': 0., 'vy': 0., 'inPath': True, 'age': 20, 'trackId': 31, 'conf': 0.8,
            'camera': 'road'}
  gone = dict(walker, y=4., vy=1.2, inPath=False, age=30)
  st = {'t': None, 'onc': None}

  def cars(sim):
    return [] if st['onc'] is None else [(JUNCTION + 60. - ONC_V * (sim.t - st['onc']), 3.6)]
  sim = left_sim()
  sim.raw = None
  frame = [0]

  def raw(s):
    frame[0] += 1
    return None if frame[0] % 2 else RawFrame([raw_box(*car_frame(s, x - s.s, y)) for x, y in cars(s) if -5. < x - s.s < 400.], 'road', s.t)
  sim.raw = raw
  rows = []
  while sim.s < JUNCTION + 20. and sim.t < 80.:
    if st['t'] is None and str(sim.tp.left.state) == 'committed' and JUNCTION - sim.s < at_m:
      st['t'] = sim.t
    objs = ()
    if st['t'] is not None:
      dt = sim.t - st['t']
      objs = (walker,) if dt < 3. else (gone,) if dt < 4. else ()
      if st['onc'] is None and dt > 1.5:
        st['onc'] = sim.t  # seen while the car stands for her
    turning = sim.tp.desires.started and abs(JUNCTION - sim.s) < 12. and sim.v > 0.3
    p = sim.step(objs, engaged=True, yaw_rate=sim.v / 8. if turning else 0.)
    rows.append((sim.t, sim.s, sim.v, p))
  assert st['t'] is not None and st['onc'] is not None
  stood = [JUNCTION - s for t, s, v, _ in rows if t > st['t'] and v < 0.05]
  assert stood and min(stood) > HELD_MIN_M, stood[:1]
  # never into the oncoming lanes while that car is within 30 m of the junction
  onc_at = [(t, JUNCTION + 60. - ONC_V * (t - st['onc'])) for t, *_ in rows if t >= st['onc']]
  met = [round(t - st['onc'], 2) for (t, x), (_, s, _, _) in zip(onc_at, rows[-len(onc_at):], strict=True)
         if JUNCTION - ONCOMING_ENTRY_M < s < JUNCTION + CLEAR_PATH_PAST_M and -5. < x - JUNCTION < 30.]
  assert not met, met[:3]
  assert not {str(p.controlPrompt) for *_, p in rows} & set(TAKE_CONTROL_PROMPTS)
  assert sim.s > JUNCTION + 10.
