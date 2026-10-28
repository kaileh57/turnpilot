#!/usr/bin/env python3
"""turnpilotd: turns the active route and the scene into desire pulses, lane-change requests and longitudinal limits.

modeld takes the plan's desire and lane change, plannerd its accel and speed caps (see hooks.py). In shadow mode
(TurnPilotShadowMode) everything is computed and published with shadow=True, and nothing acts on it.

TurnPilot acts only while it can do its job (Kellen, 2026-10-27: no degraded mode, and every failure loud):
- Not until the detector's first fresh results after TurnPilot is turned on (state unavailable, "waiting for the
  detector"): modeld loads it only while disengaged, so a mid-drive enable can't be handed back before it ever ran.
- Once it has run, DETECTOR_DEADLINE_S without fresh results is a stopped detector. modeld's own skips stay inside it: a
  slow spell forces a run after 1 s, a failed run retries after 1 s. A second forced wait (2 s) or failure in a row is
  past it.
- A modelV2 from the small model after one from the big model is a fallback (modelV2.big), for the rest of the drive.
On either, plan.handback is set and selfdrived soft-disables (handback.py). TurnPilot stops at once: no new desire,
lane change or signal prompt, the maneuver in hand is dropped (its turn desire still cleared from the model once the car
has turned), and on a fallback the destination is removed, which cancels navd's route. The scene's stops in force stay,
and so does the turn speed cap of a turn whose desire already went out: a handback must not speed the car up.
"""
from opendbc.car.structs import car
from openpilot.cereal import custom, messaging
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL, Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.turnpilot.desire_timing import DesireTiming
from openpilot.selfdrive.turnpilot.handback import Handback
from openpilot.selfdrive.turnpilot.lane_planner import OPENPILOT_SIGNALS_BRANDS, LanePlanner, lane_change_window, model_lanes_from_msg
from openpilot.selfdrive.turnpilot.route_follower import KEEP, TURN, RouteFollower
from openpilot.selfdrive.turnpilot.scene.tracker import STALE_S, SceneTracker, default_yaw_rate
from openpilot.selfdrive.turnpilot.scene_policy import NO_ACCEL_CAP, ScenePolicy
from openpilot.selfdrive.turnpilot.turn_speed import NO_SPEED_CAP, turn_speed_cap

State = custom.TurnpilotPlan.State

APPROACH_M = 150.  # a turn or keep this close (or inside its lane change window) is being approached
# no fresh detections for this long (the scene goes unavailable STALE_S after the last) and the detector has stopped:
# past modeld's first forced run or retry (1 s), short of its second (2 s)
DETECTOR_DEADLINE_S = 1.5
HANDBACK_REASON = {Handback.detectorStopped: "detector stopped: take control",
                   Handback.modelFellBack: "driving model fell back: take control"}
WAITING_REASON = "waiting for the detector"


class TurnPilot:
  def __init__(self):
    self.big_seen = False  # modelV2 came from the big model this drive (kept when TurnPilot is turned off and on)
    self.fell_back = False  # ...and then from the small one
    self.cancel_route = False  # set once on a fallback: main() removes NavDestination
    self.reset()

  def reset(self):
    self.follower = RouteFollower()
    self.desires = DesireTiming()
    self.lanes = LanePlanner()
    self.scene = ScenePolicy()
    self.detector_seen = False   # fresh detections since TurnPilot was turned on
    self.stale_s = 0.            # how long the scene has been unavailable
    self.acting = False
    self.dropped_id: int | None = None  # the maneuver in hand at a handback: not taken up again
    self.turn_cap_id: int | None = None  # ...whose turn speed cap stays, its desire having gone out

  def watch_model(self, model, enabled: bool) -> None:
    """Note a fallback from the big driving model, whether or not TurnPilot is on: it lasts the drive."""
    if model.big:
      self.big_seen = True
    elif self.big_seen and not self.fell_back:
      self.fell_back = True
      self.cancel_route = enabled  # the route TurnPilot was driving; with it off, navd's route isn't TurnPilot's

  def handback(self, available: bool) -> int:
    """Why TurnPilot can't act now, as a Handback: none while it can, and while the detector hasn't started yet."""
    self.detector_seen = self.detector_seen or available
    self.stale_s = 0. if available else self.stale_s + DT_MDL
    if self.fell_back:
      return Handback.modelFellBack
    if self.detector_seen and self.stale_s > DETECTOR_DEADLINE_S - STALE_S:
      return Handback.detectorStopped
    return Handback.none

  def update(self, sm, scene, enabled: bool, shadow: bool, driver_signals: bool = True, right_hand_traffic: bool = True):
    """A turnpilotPlan message from sm's carState, navInstruction and modelV2, and a TurnpilotScene."""
    msg = messaging.new_message('turnpilotPlan')
    plan = msg.turnpilotPlan
    plan.shadow = shadow
    plan.accelCap = NO_ACCEL_CAP
    plan.speedCap = NO_SPEED_CAP
    self.watch_model(sm['modelV2'], enabled)
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

    handback = self.handback(available)
    plan.handback = handback
    acting = self.detector_seen and handback == Handback.none
    if self.acting and not acting:
      # handing back: drop the maneuver in hand, keep its turn cap if its desire went out (the car may be in the turn)
      self.dropped_id = rf.maneuver_id
      self.turn_cap_id = rf.maneuver_id if self.desires.started else None
      self.desires.abandon()
      self.lanes = LanePlanner()
    self.acting = acting
    act_on = maneuver if acting and rf.maneuver_id != self.dropped_id else None
    cap_on = act_on if act_on is not None else (maneuver if rf.maneuver_id == self.turn_cap_id else None)

    lanes = model_lanes_from_msg(model)
    plan.desire = self.desires.update(act_on, rf.distance, rf.maneuver_id, rf.passed, v_ego, cs.steeringAngleDeg,
                                      lanes.keep_left_ok, lanes.keep_right_ok, DT_MDL, default_yaw_rate(cs, model))
    plan.laneChange, plan.signalPrompt, lane_reason = self.lanes.update(
      act_on, rf.distance, lanes, model.meta.laneChangeState, v_ego, cs.leftBlinker, cs.rightBlinker, objects,
      driver_signals, right_hand_traffic, DT_MDL)
    # the scene's holds only ever brake, so they keep the real maneuver through a handback
    decision = self.scene.update(objects, available, maneuver, rf.distance, v_ego, right_hand_traffic, rf.maneuver_id, cs.gasPressed)
    plan.accelCap = decision.accel_cap
    plan.speedCap = min(decision.speed_cap, turn_speed_cap(cap_on, rf.distance, v_ego, right_hand_traffic))
    plan.shouldStop = decision.should_stop
    plan.reason = decision.reason or lane_reason

    plan.routeValid = rf.valid
    if rf.valid and rf.maneuver is not None:
      plan.maneuverDistance = rf.distance
      plan.maneuverType = rf.maneuver.type
      plan.maneuverModifier = rf.maneuver.modifier

    if not acting:
      plan.state = State.unavailable
      plan.reason = HANDBACK_REASON.get(handback, WAITING_REASON)
    elif decision.holding:
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
    if tp.cancel_route:
      tp.cancel_route = False
      cloudlog.warning(f"turnpilot: driving model fell back to the small model, handing back{'' if shadow else ' and cancelling the route'}")
      if not shadow:
        params.remove("NavDestination")
    rk.keep_time()


if __name__ == "__main__":
  main()
