"""A step in navd's distance on the approach to a stop sign (a GPS jump, as near a junction) is no position fault
(stop_go.settled_spread): the car stops at its sign, looks and goes; only readings that keep disagreeing hand it back (H2).
test_turnpilotd_vision's vsim (the shipped visibility file, no surveys, TurnPilotStopSignGo vision), shipped settings."""
import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot import stop_rules as sr
from openpilot.selfdrive.turnpilot.stop_go import H2_SPREAD_M, settled_spread
from openpilot.selfdrive.turnpilot.tests.shipped import shipped
from openpilot.selfdrive.turnpilot.tests.test_turnpilotd_stop import S_LINE, go_frame
from openpilot.selfdrive.turnpilot.tests.test_turnpilotd_vision import vsim


def drive(tmp_path, jump_m: float, from_m: float = 40., for_s: float = 4., noise=None):
  """The sim after 70 s: navd's place for the car jump_m ahead of where it is for for_s from from_m before the stop line;
  noise: (sim) -> m added to it on every reading instead."""
  sim = shipped(vsim(tmp_path))
  assert sim.tp.stop_sign_mode == 'vision'
  update, state = sim.navd.update, {'t0': None}

  def jumped(s):
    if noise is not None:
      return update(s + noise(sim))
    if jump_m and state['t0'] is None and S_LINE - s < from_m:
      state['t0'] = sim.t
    on = state['t0'] is not None and sim.t - state['t0'] < for_s
    return update(s + (jump_m if on else 0.))
  sim.navd.update = jumped
  for _ in range(int(70. / DT_MDL)):
    sim.step()
  return sim


def stood_at(sim) -> float:
  """m past the stop line (under 0 short of it) where the car first stood."""
  return next(s for s, v, _ in sim.trace if v < 0.05 and s > S_LINE - 30.) - S_LINE


def test_with_no_jump_the_stop_sign_is_stopped_at_and_crossed(tmp_path):
  sim = drive(tmp_path, 0.)
  assert go_frame(sim) is not None and sim.s > S_LINE + 20.
  assert not any(p.controlPrompt == 'stopSignTakeControl' for p in sim.plans())
  assert abs(stood_at(sim)) < 1.


@pytest.mark.parametrize("jump_m", [5., 7.5, 10., -5., -7.5, -10.])
@pytest.mark.parametrize("from_m, for_s", [(40., 4.), (40., 1.), (25., 4.), (25., 1.), (12., 1.), (40., 60.), (25., 60.), (12., 60.)])
def test_a_step_in_nav_distance_on_the_approach_is_no_handback(tmp_path, jump_m, from_m, for_s):
  sim = drive(tmp_path, jump_m, from_m, for_s)
  handed = [p for p in sim.plans() if p.controlPrompt == 'stopSignTakeControl']
  assert not handed, f"{len(handed) * DT_MDL:.0f} s of stopSignTakeControl ({handed[0].reason!r}), the car at {sim.s - S_LINE:+.1f} m of the line"
  assert not any(str(p.controlAlert) == 'junctionUncertain' for p in sim.plans())
  # it stood for its sign, then looked and went
  assert go_frame(sim) is not None and sim.s > S_LINE + 20.
  # where: within the step of the true line at the worst (the tracker moves its distance half a metre a fix toward the
  # readings), and for a jump that came back within the position error
  at = stood_at(sim)
  assert abs(at) <= (abs(jump_m) + 1. if for_s > 4. else sr.POS_ERR_M + 0.5), at


def test_readings_that_keep_disagreeing_are_still_the_position_fault(tmp_path):
  # every reading 6 m off the last one, to and fro: no step to anchor on, the junction's position isn't known (H2)
  sim = drive(tmp_path, 0., noise=lambda sim: 6. if int(sim.t) % 2 else 0.)
  plans = sim.plans()
  assert any(str(p.controlAlert) == 'junctionUncertain' for p in plans)
  handed = [p for p in plans if p.controlPrompt == 'stopSignTakeControl']
  assert handed and handed[0].reason == "Stop sign position uncertain: take control"
  assert go_frame(sim) is None


def fixes(*vals):
  return [(float(k), float(x)) for k, x in enumerate(vals)]


def test_settled_spread_takes_a_step_out_and_keeps_a_spread_that_persists():
  assert settled_spread([]) == 0. and settled_spread(fixes(3.)) == 0.
  assert settled_spread(fixes(0., 1., 3., 2.)) == 3.            # under the fault's spread: as it is
  assert settled_spread(fixes(0., 0.5, 0., 7.5)) == 0.          # a step, one fresh reading at the latch
  assert settled_spread(fixes(0., 0.5, 7.5, 8., 7.)) == 1.      # a step that stays: the fresh level's own spread
  assert settled_spread(fixes(0., 0.5, 7.5, 8., 0.2)) == 0.5    # a step and back: the level it came back to
  assert settled_spread(fixes(-6., 0., 0.5, 0.)) == 0.5         # either sign
  assert settled_spread(fixes(0., 2.6, 5.2, 5.)) == pytest.approx(0.2)  # a step navd's filter spread over two readings
  # readings that keep leaving the latest one's level: the whole spread
  assert settled_spread(fixes(0., 6., 0., 6., 0.)) == 6. and settled_spread(fixes(6., 0., 6., 0.)) == 6.
  # a drift with no step in it, or one that walks on more than a step a reading: the whole spread
  assert settled_spread(fixes(0., 1.5, 3., 4.5)) == 4.5 > H2_SPREAD_M
  assert settled_spread(fixes(0., 2.5, 5., 7.5)) == 7.5 and settled_spread(fixes(7.5, 5., 2.5, 0., 0.3)) == 7.5
