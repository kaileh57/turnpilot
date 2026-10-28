"""A map stop through openpilot's real longitudinal path, with navd's distance from noisy 1 Hz GPS.

BeamNG's GPS is exact, so this is the only coverage of GPS jitter at a stop line. The planner (20 Hz) takes the min of
the model's (easing to the set speed), the cruise and TurnPilot's (hooks.plan_accel of MapControlPolicy's stop through
ScenePolicy's composition and ramp) candidates, as plannerd does; LongControl (100 Hz) runs its real state machine; the
car follows the command through 0.15 s of dead time and a 0.3 s lag (about a Model 3's brakes and regen).
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import should_stop as op_should_stop
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.lib.longitudinal_planner import get_cruise_accel
from openpilot.selfdrive.turnpilot.hooks import plan_accel
from openpilot.selfdrive.turnpilot.map_controls import MapControlPolicy
from openpilot.selfdrive.turnpilot.scene_policy import ScenePolicy, decide
from openpilot.selfdrive.turnpilot.tests.synthetic import nav_instruction

CP = NS(longitudinalTuning=NS(kiBP=[0.], kiV=[0.]), stopAccel=-2.0)
LINE_M = 250.
V0 = 30 * 0.44704
SEEDS = 40


def run(sigma: float, seed: int, t_max: float = 40.):
  rng = np.random.default_rng(seed)
  policy, scene, lc = MapControlPolicy(), ScenePolicy(), LongControl(CP)
  v, s, a_act = V0, 0., 0.
  a_cmd_hist = [0.] * int(0.15 / DT_CTRL)
  a_cruise = a_target_prev = a_target = 0.
  stop_flag = False
  latched_s = None
  rows = []
  for k in range(int(t_max / DT_CTRL)):
    if k % 5 == 0:
      frame = k // 5
      due = frame % 20 == 0
      nav = None
      if due:
        meas = LINE_M - s + rng.normal(0., sigma)
        nav = nav_instruction(500., route_count=1, controls=[(0, 'stop', meas)] if meas < 400. else [])
      out = policy.update(nav, due, True, v, False, dt=DT_MDL)
      dec = decide(out.stops, v)
      dec.accel_cap = scene.ramp(dec.accel_cap, DT_MDL)
      a_e2e = float(np.clip(V0 - v, -0.5, 0.5))
      a_cruise = get_cruise_accel(V0, v, a_cruise, a_target_prev, DT_MDL)
      cands = [(a_e2e, False), (a_cruise, op_should_stop(v, a_cruise)), (plan_accel(dec.accel_cap, dec.speed_cap, v), dec.should_stop)]
      a_target = float(np.clip(min(c[0] for c in cands), -3.5, 2.0))
      stop_flag = any(c[1] for c in cands)
      a_target_prev = a_target
      if latched_s is None and dec.should_stop:
        latched_s = s
      rows.append((s, v, a_target, a_act))
    CS = NS(vEgo=v, aEgo=a_act, brakePressed=False, cruiseState=NS(standstill=False))
    a_cmd_hist.append(lc.update(True, CS, a_target, stop_flag, (-3.5, 2.0)))
    a_act += (a_cmd_hist.pop(0) - a_act) * DT_CTRL / 0.3
    v = max(v + (0. if v <= 0. and a_act < 0. else a_act) * DT_CTRL, 0.)
    s += v * DT_CTRL
  arr = np.array(rows)
  return arr, latched_s, s


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_stop_through_gps_jitter(sigma):
  harder, creeps, errors = 0, 0, []
  for seed in range(SEEDS):
    arr, latched_s, s_end = run(sigma, seed)
    assert latched_s is not None, f"seed {seed}: never latched"
    harder += arr[:, 2].min() < -2.05  # the commanded accel: never past the map stop's floor
    creeps += s_end - latched_s > 0.3  # no creep once latched
    errors.append(LINE_M - s_end)
  errors = np.array(errors)
  assert harder == 0 and creeps == 0
  assert np.abs(errors).max() < 2., errors  # stopped within 2 m of the stop point (a GPS bias would go straight through)
