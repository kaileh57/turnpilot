#!/usr/bin/env python3
"""turnpilotd: turns the active route and the scene into desire pulses, lane-change requests and longitudinal limits.

modeld takes the plan's desire and lane change, plannerd its accel and speed caps (see hooks.py). In shadow mode
(TurnPilotShadowMode) everything is computed and published with shadow=True, and nothing acts on it.
It runs onroad only while TurnPilotEnabled. The manager doesn't restart a process that died, and a dead one
soft-disables openpilot for the rest of the drive (processNotRunning), so an exception in the plan or the scene is
logged and a disabled plan or an empty scene goes out instead.
"""
import time

from opendbc.car.structs import car
from openpilot.cereal import custom, messaging
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL, Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.turnpilot.desire_timing import DesireTiming
from openpilot.selfdrive.turnpilot.lane_planner import (OPENPILOT_SIGNALS_BRANDS, LanePlanner, lane_change_window, lane_mismatch,
                                                         model_lanes_from_msg)
from openpilot.selfdrive.turnpilot.route_follower import KEEP, TURN, RouteFollower
from openpilot.selfdrive.turnpilot.scene.tracker import SceneTracker, default_yaw_rate
from openpilot.selfdrive.turnpilot.scene_policy import NO_ACCEL_CAP, scene_policy
from openpilot.selfdrive.turnpilot.turn_skip import TurnSkip
from openpilot.selfdrive.turnpilot.turn_speed import NO_SPEED_CAP, turn_speed_cap

State = custom.TurnpilotPlan.State
Fault = custom.TurnpilotPlan.Fault
LaneMismatch = custom.TurnpilotPlan.LaneMismatch

APPROACH_M = 150.  # a turn or keep this close (or inside its lane change window) is being approached
NO_LONG_REASON = "openpilot longitudinal is off"
ERROR_REASON = "TurnPilot error"
CAR_UNKNOWN_REASON = "waiting for the car's parameters"
ERROR_LOG_PERIOD_S = 10.  # s between logged exceptions from one place, so one that fails every frame doesn't flood the log
# the plan is invalid once carState or modelV2 have failed their checks this many frames in a row (0.5 s at 20 Hz, as
# hooks.PLAN_MAX_AGE_S), not at the first: a plan invalid for one frame drops its desire for that frame, and modeld takes
# the desire coming back as a new pulse
PLAN_INPUT_GRACE_FRAMES = 10
MISMATCH_CONFIRM_FRAMES = 20  # a lane mismatch this many frames in a row (1 s) is flagged: the model's lane count flickers
NO_JUNCTION_DATA_REASON = "no junction data: nominal turn angle"


class ErrorLog:
  """cloudlog.exception for the exception being handled, at most once per ERROR_LOG_PERIOD_S for each event."""
  def __init__(self):
    self.last: dict[str, float] = {}

  def __call__(self, event: str):
    now = time.monotonic()
    if now - self.last.get(event, -ERROR_LOG_PERIOD_S) >= ERROR_LOG_PERIOD_S:
      self.last[event] = now
      cloudlog.exception(event)


log_error = ErrorLog()


class TurnPilot:
  def __init__(self):
    self.reset()

  def reset(self):
    self.follower = RouteFollower()
    self.desires = DesireTiming()
    self.lanes = LanePlanner()
    self.skip = TurnSkip()
    self.mismatch = LaneMismatch.none
    self._mismatch_candidate, self._mismatch_frames = LaneMismatch.none, 0

  def _lane_mismatch(self, found) -> int:
    """found, once it has held MISMATCH_CONFIRM_FRAMES; logged when it changes."""
    self._mismatch_frames = self._mismatch_frames + 1 if found == self._mismatch_candidate else 1
    self._mismatch_candidate = found
    if (found == LaneMismatch.none or self._mismatch_frames >= MISMATCH_CONFIRM_FRAMES) and found != self.mismatch:
      if found != LaneMismatch.none:
        rf = self.follower
        cloudlog.event("turnpilot_lane_mismatch", error=True, mismatch=str(found), maneuver_distance=round(rf.distance, 1),
                       through_distance=round(rf.through_distance, 1))
      self.mismatch = found
    return self.mismatch

  def update(self, sm, scene, enabled: bool, shadow: bool, driver_signals: bool = True, right_hand_traffic: bool = True):
    """A turnpilotPlan message from sm's carState, navInstruction and modelV2, and a TurnpilotScene."""
    msg = messaging.new_message('turnpilotPlan')
    plan = msg.turnpilotPlan
    plan.shadow = shadow
    plan.accelCap = NO_ACCEL_CAP
    plan.speedCap = NO_SPEED_CAP
    if not enabled:
      self.reset()
      plan.state = State.disabled
      return msg

    cs, model = sm['carState'], sm['modelV2']
    v_ego = max(cs.vEgo, 0.)
    rf = self.follower
    rf.update(sm['navInstruction'], sm.updated['navInstruction'], sm.valid['navInstruction'], v_ego, DT_MDL)
    upcoming = None if rf.arrived else rf.upcoming
    # too fast to make the turn: skipped, so nothing below acts on it (no desire, no cap, no turn holds or lane changes)
    desire_out = self.desires.maneuver_id == rf.maneuver_id and self.desires.started
    skip = self.skip.update(upcoming, rf.distance, rf.maneuver_id, v_ego, desire_out, right_hand_traffic)
    maneuver = None if skip.skipped else upcoming
    plan.tooFastToTurn = skip.skipped

    available = scene.available
    objects = list(scene.objects) if available else []

    lanes = model_lanes_from_msg(model)
    plan.desire = self.desires.update(maneuver, rf.distance, rf.maneuver_id, rf.passed, v_ego, cs.steeringAngleDeg,
                                      lanes.keep_left_ok, lanes.keep_right_ok, DT_MDL, default_yaw_rate(cs, model),
                                      lat_active=sm['carControl'].latActive, replaced=rf.replaced, may_start=skip.feasible_now)
    plan.laneChange, plan.signalPrompt, lane_reason = self.lanes.update(
      maneuver, rf.distance, lanes, model.meta.laneChangeState, v_ego, cs.leftBlinker, cs.rightBlinker, objects,
      driver_signals, right_hand_traffic, DT_MDL)
    plan.laneMismatch = self._lane_mismatch(lane_mismatch(maneuver, rf.distance, rf.through_lanes, rf.through_distance, lanes,
                                                          v_ego, right_hand_traffic))
    decision = scene_policy(objects, available, maneuver, rf.distance, v_ego, right_hand_traffic)
    plan.accelCap = decision.accel_cap
    plan.speedCap = min(decision.speed_cap, turn_speed_cap(maneuver, rf.distance, v_ego, right_hand_traffic))
    plan.shouldStop = decision.should_stop
    no_junction_data = maneuver is not None and maneuver.kind in (TURN, KEEP) and not maneuver.geometry
    plan.reason = decision.reason or lane_reason or (NO_JUNCTION_DATA_REASON if no_junction_data else "")

    plan.routeValid = rf.valid
    if rf.valid and rf.maneuver is not None:
      plan.maneuverDistance = rf.distance
      plan.maneuverType = rf.maneuver.type
      plan.maneuverModifier = rf.maneuver.modifier
      plan.maneuverAngle = rf.maneuver.angle_deg
      plan.maneuverGeometryValid = rf.maneuver.geometry
      plan.maneuverBend = rf.maneuver.bend

    if decision.holding:
      plan.state = State.holding
    elif not rf.valid:
      plan.state = State.noRoute
    elif rf.arrived:
      plan.state = State.arrived
    elif self.lanes.active:
      plan.state = State.laneChange
    elif self.desires.active:
      plan.state = State.turning
    elif maneuver is not None and maneuver.kind in (TURN, KEEP) and \
         rf.distance < max(APPROACH_M, lane_change_window(v_ego)[0]):
      plan.state = State.approaching
    else:
      plan.state = State.following
    return msg

  def plan(self, sm, scene, enabled: bool, shadow: bool, *args, long_control: bool | None, **kwargs):
    """update(), but a disabled plan while openpilot doesn't control speed (on a Tesla, without alpha longitudinal):
    its speed caps would do nothing there, and the turn desires would go out at whatever speed stock ACC holds.
    long_control None: card hasn't written CarParams yet, so it isn't known. A disabled plan while TurnPilot is on
    carries its fault, which selfdrived shows (turnpilotUnavailable): TurnPilot never stops working unannounced."""
    if enabled and not long_control:
      self.reset()
      return disabled_plan(shadow, CAR_UNKNOWN_REASON) if long_control is None else \
        disabled_plan(shadow, NO_LONG_REASON, Fault.noLongitudinal)
    try:
      return self.update(sm, scene, enabled, shadow, *args, **kwargs)
    except Exception:
      log_error("turnpilotd.plan_failed")
      self.reset()
      return disabled_plan(shadow, ERROR_REASON, Fault.error)


def disabled_plan(shadow: bool, reason: str, fault=Fault.none):
  """A plan nothing acts on: no desire, no lane change, no limits."""
  msg = messaging.new_message('turnpilotPlan')
  plan = msg.turnpilotPlan
  plan.state, plan.shadow, plan.reason, plan.fault = State.disabled, shadow, reason, fault
  plan.accelCap, plan.speedCap = NO_ACCEL_CAP, NO_SPEED_CAP
  return msg


def car_params(params):
  """CarParams as card last wrote them, or None: not yet written, or unreadable."""
  cp_bytes = params.get("CarParams")
  if cp_bytes is None:
    return None
  try:
    return messaging.log_from_bytes(cp_bytes, car.CarParams)
  except Exception:
    log_error("turnpilotd.car_params_unreadable")
    return None


def driver_signals_for(params) -> bool:
  """True unless the car's port signals for openpilot's lane changes; unknown cars prompt the driver."""
  cp = car_params(params)
  return cp is None or cp.brand not in OPENPILOT_SIGNALS_BRANDS


def long_control_for(params) -> bool | None:
  """CarParams.openpilotLongitudinalControl; None until card has written CarParams, False if they don't read."""
  if params.get("CarParams") is None:
    return None
  cp = car_params(params)
  return cp is not None and cp.openpilotLongitudinalControl


def main():
  params = Params()
  pm = messaging.PubMaster(['turnpilotPlan', 'turnpilotScene'])
  optional = ['navInstruction', 'turnpilotDetections']  # no route, or no detector (TurnPilotEnabled off, no pkl)
  sm = messaging.SubMaster(['carState', 'carControl', 'navInstruction', 'modelV2', 'turnpilotDetections', 'extrinsicsCalibration',
                            'deviceState', 'narrowRoadCameraState'],
                           ignore_alive=optional, ignore_avg_freq=optional, ignore_valid=optional)
  rk = Ratekeeper(1. / DT_MDL, print_delay_threshold=None)
  tp = TurnPilot()
  tracker = None
  enabled = shadow = False
  driver_signals, right_hand_traffic = True, True
  long_control: bool | None = None
  invalid_frames = PLAN_INPUT_GRACE_FRAMES
  while True:
    sm.update(0)
    if sm.frame % int(1. / DT_MDL) == 0:
      enabled = params.get_bool("TurnPilotEnabled")
      shadow = params.get_bool("TurnPilotShadowMode")
      right_hand_traffic = not params.get_bool("IsRhdDetected")
      driver_signals = driver_signals_for(params)
      long_control = long_control_for(params)

    scene_msg = messaging.new_message('turnpilotScene')
    try:
      if tracker is None and sm.seen['deviceState'] and sm.seen['narrowRoadCameraState']:
        device_type = str(sm['deviceState'].deviceType)
        tracker = SceneTracker(DEVICE_CAMERAS[(device_type, str(sm['narrowRoadCameraState'].sensor))], device_type)
      if tracker is not None:
        tracker.update(sm['turnpilotDetections'] if sm.seen['turnpilotDetections'] else None, sm['extrinsicsCalibration'],
                       sm['carState'], sm['modelV2'])
        tracker.fill_scene(scene_msg.turnpilotScene)
    except Exception:
      log_error("turnpilotd.scene_failed")
      tracker, scene_msg = None, messaging.new_message('turnpilotScene')  # unavailable: as with no detector, lefts hold
    pm.send('turnpilotScene', scene_msg)

    msg = tp.plan(sm, scene_msg.as_reader().turnpilotScene, enabled, shadow, driver_signals, right_hand_traffic, long_control=long_control)
    invalid_frames = 0 if sm.all_checks(['carState', 'modelV2']) else invalid_frames + 1
    msg.valid = invalid_frames < PLAN_INPUT_GRACE_FRAMES
    pm.send('turnpilotPlan', msg)
    rk.keep_time()


if __name__ == "__main__":
  main()
