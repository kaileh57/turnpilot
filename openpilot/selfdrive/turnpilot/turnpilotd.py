#!/usr/bin/env python3
"""turnpilotd: turns the active route and the scene into desire pulses, lane-change requests and longitudinal limits.

modeld takes the plan's desire and lane change, plannerd its accel and speed caps (see hooks.py). In shadow mode
(TurnPilotShadowMode) everything is computed and published with shadow=True, and nothing acts on it.
"""
from opendbc.car.structs import car
from openpilot.cereal import custom, messaging
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL, Ratekeeper
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.turnpilot.desire_timing import DesireTiming
from openpilot.selfdrive.turnpilot.lane_planner import OPENPILOT_SIGNALS_BRANDS, LanePlanner, lane_change_window, model_lanes_from_msg
from openpilot.selfdrive.turnpilot.route_follower import KEEP, TURN, RouteFollower
from openpilot.selfdrive.turnpilot.scene.tracker import SceneTracker, default_yaw_rate
from openpilot.selfdrive.turnpilot.scene_policy import NO_ACCEL_CAP, scene_policy
from openpilot.selfdrive.turnpilot.turn_speed import NO_SPEED_CAP, turn_speed_cap

State = custom.TurnpilotPlan.State

APPROACH_M = 150.  # a turn or keep this close (or inside its lane change window) is being approached


class TurnPilot:
  def __init__(self):
    self.reset()

  def reset(self):
    self.follower = RouteFollower()
    self.desires = DesireTiming()
    self.lanes = LanePlanner()

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
    maneuver = None if rf.arrived else rf.upcoming

    available = scene.available
    objects = list(scene.objects) if available else []

    lanes = model_lanes_from_msg(model)
    plan.desire = self.desires.update(maneuver, rf.distance, rf.maneuver_id, rf.passed, v_ego, cs.steeringAngleDeg,
                                      lanes.keep_left_ok, lanes.keep_right_ok, DT_MDL, default_yaw_rate(cs, model),
                                      float(model.position.y[-1]) if len(model.position.y) else None)
    plan.laneChange, plan.signalPrompt, lane_reason = self.lanes.update(
      maneuver, rf.distance, lanes, model.meta.laneChangeState, v_ego, cs.leftBlinker, cs.rightBlinker, objects,
      driver_signals, right_hand_traffic, DT_MDL)
    decision = scene_policy(objects, available, maneuver, rf.distance, v_ego, right_hand_traffic)
    plan.accelCap = decision.accel_cap
    plan.speedCap = min(decision.speed_cap, turn_speed_cap(maneuver, rf.distance, v_ego, right_hand_traffic))
    plan.shouldStop = decision.should_stop
    plan.reason = decision.reason or lane_reason

    plan.routeValid = rf.valid
    if rf.valid and rf.maneuver is not None:
      plan.maneuverDistance = rf.distance
      plan.maneuverType = rf.maneuver.type
      plan.maneuverModifier = rf.maneuver.modifier

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


def driver_signals_for(params) -> bool:
  """True unless the car's port signals for openpilot's lane changes; unknown cars prompt the driver."""
  cp_bytes = params.get("CarParams")
  if cp_bytes is None:
    return True
  return messaging.log_from_bytes(cp_bytes, car.CarParams).brand not in OPENPILOT_SIGNALS_BRANDS


def main():
  params = Params()
  pm = messaging.PubMaster(['turnpilotPlan', 'turnpilotScene'])
  optional = ['navInstruction', 'turnpilotDetections']  # no route, or no detector (TurnPilotEnabled off, no pkl)
  sm = messaging.SubMaster(['carState', 'navInstruction', 'modelV2', 'turnpilotDetections', 'extrinsicsCalibration',
                            'deviceState', 'narrowRoadCameraState'],
                           ignore_alive=optional, ignore_avg_freq=optional, ignore_valid=optional)
  rk = Ratekeeper(1. / DT_MDL, print_delay_threshold=None)
  tp = TurnPilot()
  tracker = None
  enabled = shadow = False
  driver_signals, right_hand_traffic = True, True
  while True:
    sm.update(0)
    if sm.frame % int(1. / DT_MDL) == 0:
      enabled = params.get_bool("TurnPilotEnabled")
      shadow = params.get_bool("TurnPilotShadowMode")
      right_hand_traffic = not params.get_bool("IsRhdDetected")
      driver_signals = driver_signals_for(params)

    scene_msg = messaging.new_message('turnpilotScene')
    if tracker is None and sm.seen['deviceState'] and sm.seen['narrowRoadCameraState']:
      device_type = str(sm['deviceState'].deviceType)
      tracker = SceneTracker(DEVICE_CAMERAS[(device_type, str(sm['narrowRoadCameraState'].sensor))], device_type)
    if tracker is not None:
      tracker.update(sm['turnpilotDetections'] if sm.seen['turnpilotDetections'] else None, sm['extrinsicsCalibration'],
                     sm['carState'], sm['modelV2'])
      tracker.fill_scene(scene_msg.turnpilotScene)
    pm.send('turnpilotScene', scene_msg)

    msg = tp.update(sm, scene_msg.as_reader().turnpilotScene, enabled, shadow, driver_signals, right_hand_traffic)
    msg.valid = sm.all_checks(['carState', 'modelV2'])
    pm.send('turnpilotPlan', msg)
    rk.keep_time()


if __name__ == "__main__":
  main()
