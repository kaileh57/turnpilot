"""The upstream hunks: what modeld, DesireHelper and the longitudinal planner do with turnpilotPlan."""
import numpy as np

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.turnpilot.hooks import CLEAR_DESIRE, active_plan, plan_desire
from openpilot.selfdrive.turnpilot.tests.synthetic import car_state

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection


class FakeSM(dict):
  def __init__(self, msgs, alive=None):
    super().__init__(msgs)
    self.alive = dict.fromkeys(msgs, True) | (alive or {})


def turnpilot_plan(shadow, desire=log.Desire.turnLeft, lane_change='left', accel_cap=-3., speed_cap=2., should_stop=True):
  msg = messaging.new_message('turnpilotPlan')
  p = msg.turnpilotPlan
  p.desire, p.laneChange, p.accelCap, p.speedCap, p.shouldStop, p.shadow = desire, lane_change, accel_cap, speed_cap, should_stop, shadow
  return msg.as_reader().turnpilotPlan


def test_active_plan():
  assert active_plan(FakeSM({'turnpilotPlan': turnpilot_plan(False)})) is not None
  assert active_plan(FakeSM({'turnpilotPlan': turnpilot_plan(True)})) is None
  assert active_plan(FakeSM({'turnpilotPlan': turnpilot_plan(False)}, alive={'turnpilotPlan': False})) is None
  assert active_plan({'turnpilotPlan': turnpilot_plan(False)}) is None  # the maneuver tests' plain dict


def modeld_inputs(sm, dh_desire=log.Desire.none):
  """modeld's turnpilot reads, as in its main loop: the desire fed to the model and DesireHelper's nav lane change."""
  tp = active_plan(sm)
  desire = tp.desire if dh_desire == log.Desire.none and tp is not None else dh_desire
  return desire, tp.laneChange.raw if tp is not None else LaneChangeDirection.none


def test_modeld_reads():
  assert modeld_inputs(FakeSM({'turnpilotPlan': turnpilot_plan(False)})) == (log.Desire.turnLeft, LaneChangeDirection.left)
  assert modeld_inputs(FakeSM({'turnpilotPlan': turnpilot_plan(True)})) == (log.Desire.none, LaneChangeDirection.none)
  # DesireHelper's own lane change desire wins
  assert modeld_inputs(FakeSM({'turnpilotPlan': turnpilot_plan(False)}), log.Desire.laneChangeRight)[0] == log.Desire.laneChangeRight


def test_clear_desire():
  # the clear is fed to the model as no desire, and asks modeld to clear the turn desire history
  assert plan_desire(CLEAR_DESIRE) == (log.Desire.none, True)
  assert plan_desire(log.Desire.turnRight) == (log.Desire.turnRight, False)
  assert CLEAR_DESIRE >= ModelConstants.DESIRE_LEN and CLEAR_DESIRE not in log.Desire.schema.enumerants.values()


def torque_state(v, left=True):
  msg = messaging.new_message('carState')
  msg.carState.vEgo = v
  msg.carState.leftBlinker = left
  msg.carState.steeringPressed = True
  msg.carState.steeringTorque = 1. if left else -1.
  return msg.as_reader().carState


def blindspot_state(v):
  msg = messaging.new_message('carState')
  msg.carState.vEgo = v
  msg.carState.leftBlindspot = True
  return msg.as_reader().carState


def test_desire_helper_without_nav_is_upstream():
  dh = DesireHelper()
  for _ in range(3):
    dh.update(car_state(15.), True, 0.5)
  assert dh.lane_change_state == LaneChangeState.off
  dh.update(car_state(15., left_blinker=True), True, 0.5)
  assert dh.lane_change_state == LaneChangeState.preLaneChange
  dh.update(car_state(15., left_blinker=True), True, 0.5)
  assert dh.lane_change_state == LaneChangeState.preLaneChange  # waits for the nudge
  dh.update(torque_state(15.), True, 0.5)
  assert dh.lane_change_state == LaneChangeState.laneChangeStarting and dh.desire == log.Desire.laneChangeLeft


def test_nav_lane_change_is_blinker_plus_nudge():
  dh = DesireHelper()
  dh.update(car_state(15.), True, 0.5, LaneChangeDirection.right)
  assert (dh.lane_change_state, dh.lane_change_direction) == (LaneChangeState.preLaneChange, LaneChangeDirection.right)
  dh.update(car_state(15.), True, 0.5, LaneChangeDirection.right)
  assert dh.lane_change_state == LaneChangeState.laneChangeStarting and dh.desire == log.Desire.laneChangeRight


def test_nav_lane_change_keeps_blindspot_and_speed_checks():
  dh = DesireHelper()
  for _ in range(5):
    dh.update(blindspot_state(15.), True, 0.5, LaneChangeDirection.left)
  assert dh.lane_change_state == LaneChangeState.preLaneChange
  dh = DesireHelper()
  for _ in range(5):
    dh.update(car_state(7.), True, 0.5, LaneChangeDirection.left)
  assert dh.lane_change_state == LaneChangeState.off


def planner_sm(v=15., plan=None):
  msgs = {s: messaging.new_message(s) for s in ('radarState', 'controlsState', 'selfdriveState', 'carState', 'modelV2')}
  msgs['controlsState'].controlsState.longControlState = LongCtrlState.pid
  msgs['carState'].carState.vEgo = v
  msgs['carState'].carState.vCruise = 30. * 3.6
  model = msgs['modelV2'].modelV2
  model.action.desiredAcceleration = 0.5
  for field in ('velocity', 'acceleration', 'position'):
    getattr(model, field).x = [float(v)] * ModelConstants.IDX_N
  sm = FakeSM({s: m.as_reader().__getattr__(s) for s, m in msgs.items()})
  if plan is not None:
    sm['turnpilotPlan'] = plan
    sm.alive['turnpilotPlan'] = True
  return sm


def run_planner(plan):
  from opendbc.car.honda.interface import CarInterface
  from opendbc.car.honda.values import CAR
  lp = LongitudinalPlanner(CarInterface.get_non_essential_params(CAR.HONDA_CIVIC), init_v=15.)
  out = []
  for _ in range(20):
    lp.update(planner_sm(plan=plan))
    out.append((float(lp.output_a_target), lp.output_should_stop))
  return out


def test_shadow_never_changes_planner_output():
  baseline = run_planner(None)
  assert run_planner(turnpilot_plan(shadow=True)) == baseline
  active = run_planner(turnpilot_plan(shadow=False))
  assert all(a < b for (a, _), (b, _) in zip(active, baseline, strict=True))
  assert all(stop for _, stop in active) and not any(stop for _, stop in baseline)
  # an active plan with no caps changes nothing either
  assert np.allclose(run_planner(turnpilot_plan(shadow=False, accel_cap=10., speed_cap=100., should_stop=False)), baseline)
