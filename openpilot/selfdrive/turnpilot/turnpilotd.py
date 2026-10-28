#!/usr/bin/env python3
"""turnpilotd: turns the active route and the scene into desire pulses, lane-change requests and longitudinal limits.

modeld takes the plan's desire and lane change, plannerd its accel and speed caps (see hooks.py). The lane paint the
detector reads comes in with its detections, and is used only while those are fresh (the scene is available); none read
is unknown paint, and no lane change. The lane planner plans lane changes only with TurnPilotLanePaintActive set; without
it a second planner publishes what it would have asked for in laneChangePaintRule, so drives can be scored. In shadow mode
(TurnPilotShadowMode) everything is computed and published with shadow=True, and nothing acts on it.
It runs onroad only while TurnPilotEnabled. The manager doesn't restart a process that died, and a dead one
soft-disables openpilot for the rest of the drive (processNotRunning), so an exception in the plan or the scene is
logged and a disabled plan or an empty scene goes out instead.

TurnPilot acts only while it can do its job (Kellen, 2026-10-27: no degraded mode, and every failure loud):
- Not until the detector's first fresh results after TurnPilot is turned on (state unavailable, "waiting for the
  detector"): modeld loads it only while disengaged, so a mid-drive enable can't be handed back before it ever ran.
- Once it has run, DETECTOR_DEADLINE_S without fresh results is a stopped detector. modeld's own skips stay inside it: a
  slow spell forces a run after 1 s, a failed run retries after 1 s. A second forced wait (2 s) or failure in a row is
  past it.
- A modelV2 from the small model after one from the big model is a fallback (modelV2.big), for the rest of the drive.
On either, plan.handback is set and selfdrived soft-disables (handback.py). TurnPilot stops at once: no new desire,
lane change, signal prompt, skipped-turn reroute or wrong-lane flag, and the maneuver in hand is dropped. A turn whose
desire already went out is never cancelled (desire_timing.py): its desire is held, with no new pulse, and cleared from
the model at the turn's angle as usual; a keep desire is dropped. On a fallback the destination is removed, which
cancels navd's route, once such a turn is over. The scene's stops in force stay, and so does the turn speed cap of a turn
whose desire already went out: a handback must not speed the car up. Nor does the route going away under a committed
turn cancel it (route_follower.py: gaps, navd going quiet, reroutes are dead-reckoned through).
A fault (plan()) is a disabled plan; it carries a model fallback's handback, so a fault never hides one.
A handback or a fault cancels a lane change that is offered and not yet started, with laneChangeAbort handedBack or
fault for ABORT_HOLD_S (selfdrived raises turnpilotLaneChangeAborted); TurnPilot turned off cancels it as turnedOff.
A change DesireHelper has started goes on.
Alert priority (hooks.plan_events): handback, then fault (turnpilotUnavailable), then too fast to turn, then a lane
change, then wrong lane. While TurnPilot isn't acting or is faulted, too fast to turn, wrong lane and lane change offers
are never set here.
"""
import time

from opendbc.car.structs import car
from openpilot.cereal import custom, log, messaging
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.realtime import DT_MDL, Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.turnpilot.desire_timing import DesireTiming
from openpilot.selfdrive.turnpilot.handback import Handback
from openpilot.selfdrive.turnpilot.hooks import ABORT_HOLD_S, CLEAR_DESIRE
from openpilot.selfdrive.turnpilot.lane_planner import (OPENPILOT_SIGNALS_BRANDS, LanePlanner, lane_change_window, lane_mismatch,
                                                         model_lanes_from_msg)
from openpilot.selfdrive.turnpilot.route_follower import KEEP, TURN, RouteFollower
from openpilot.selfdrive.turnpilot.scene.lane_paint import lane_paint_from_msg
from openpilot.selfdrive.turnpilot.scene.tracker import STALE_S, SceneTracker, default_yaw_rate
from openpilot.selfdrive.turnpilot.scene_policy import NO_ACCEL_CAP, ScenePolicy
from openpilot.selfdrive.turnpilot.turn_skip import TurnSkip
from openpilot.selfdrive.turnpilot.turn_speed import NO_SPEED_CAP, turn_speed_cap

Desire = log.Desire
State = custom.TurnpilotPlan.State
Fault = custom.TurnpilotPlan.Fault
LaneMismatch = custom.TurnpilotPlan.LaneMismatch
LaneChange = custom.TurnpilotPlan.LaneChange

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
# no fresh detections for this long (the scene goes unavailable STALE_S after the last) and the detector has stopped:
# past modeld's first forced run or retry (1 s), short of its second (2 s)
DETECTOR_DEADLINE_S = 1.5
HANDBACK_REASON = {Handback.detectorStopped: "detector stopped: take control",
                   Handback.modelFellBack: "driving model fell back: take control"}
WAITING_REASON = "waiting for the detector"


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
    self.big_seen = False  # modelV2 came from the big model this drive (kept when TurnPilot is turned off and on)
    self.fell_back = False  # ...and then from the small one
    self.cancel_route = False  # set once on a fallback: main() removes NavDestination, once no turn is committed
    self.stop_abort, self._stop_abort_t = 'none', 0.  # a lane change cancelled by a disabled plan: reset() can't hold it
    self.reset()

  def reset(self):
    self.follower = RouteFollower()
    self.desires = DesireTiming()
    self.lanes = LanePlanner()
    self.paint_rule_lanes = LanePlanner(paint_rule=True)  # what the lane planner would ask for, while it's turned off
    self.skip = TurnSkip()
    self.mismatch = LaneMismatch.none
    self._mismatch_candidate, self._mismatch_frames = LaneMismatch.none, 0
    self.scene = ScenePolicy()
    self.detector_seen = False   # fresh detections since TurnPilot was turned on
    self.stale_s = 0.            # how long the scene has been unavailable
    self.acting = False
    self.dropped_id: int | None = None  # the maneuver in hand at a handback: not taken up again
    self.turn_cap_id: int | None = None  # ...whose turn speed cap stays, its desire having gone out
    self.turn_held_id: int | None = None  # ...whose turn desire was out: held to its clear, never cancelled
    self.last_desire = Desire.none

  @property
  def turn_committed(self) -> bool:
    """The route's maneuver is a turn whose desire went out and isn't over: it is held through a gap in the route
    (route_follower.py), and a fallback's route cancel waits for it."""
    d = self.desires
    return d.maneuver_id == self.follower.maneuver_id and d.started and not d.done and d.side != 0

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

  def stop(self, reason: str) -> None:
    """TurnPilot goes disabled (turned off, or a fault: reason turnedOff or fault) and resets: an offered lane change is
    cancelled out loud, its abort held ABORT_HOLD_S on the disabled plans (fill_stop_abort), as any abort."""
    if self.lanes.cancel(reason):
      self.stop_abort, self._stop_abort_t = reason, ABORT_HOLD_S
    self.reset()

  def fill_stop_abort(self, plan) -> None:
    if self._stop_abort_t > 0.:
      self._stop_abort_t -= DT_MDL
      plan.laneChangeAbort = self.stop_abort

  def update(self, sm, scene, enabled: bool, shadow: bool, driver_signals: bool = True, right_hand_traffic: bool = True, paint=(),
             paint_rule: bool = True, inputs_valid: bool = True):
    """A turnpilotPlan message from sm's carState, navInstruction and modelV2, a TurnpilotScene, and the lane paint of
    modelV2's four lane lines (empty when unknown). paint_rule: TurnPilotLanePaintActive; inputs_valid: carState and
    modelV2 pass their checks."""
    msg = messaging.new_message('turnpilotPlan')
    plan = msg.turnpilotPlan
    plan.shadow = shadow
    plan.accelCap = NO_ACCEL_CAP
    plan.speedCap = NO_SPEED_CAP
    self.watch_model(sm['modelV2'], enabled)
    if not enabled:
      self.stop('turnedOff')
      plan.state = State.disabled
      self.fill_stop_abort(plan)
      return msg
    self._stop_abort_t = 0.

    cs, model = sm['carState'], sm['modelV2']
    v_ego = max(cs.vEgo, 0.)
    rf = self.follower
    rf.update(sm['navInstruction'], sm.updated['navInstruction'], sm.valid['navInstruction'], v_ego, DT_MDL,
              committed=self.turn_committed)
    upcoming = None if rf.arrived else rf.upcoming

    available = scene.available
    objects = list(scene.objects) if available else []

    handback = self.handback(available)
    plan.handback = handback
    acting = self.detector_seen and handback == Handback.none
    if self.acting and not acting:
      # handing back: drop the maneuver in hand, keep its turn cap if its desire went out (the car may be in the turn),
      # and a turn desire that went out is held to its clear, never cancelled; anything else is abandoned
      d = self.desires
      in_turn = d.maneuver_id == rf.maneuver_id and d.started and not d.done and d.side != 0
      self.dropped_id = rf.maneuver_id
      self.turn_cap_id = rf.maneuver_id if d.started else None
      self.turn_held_id = rf.maneuver_id if in_turn else None
      if not in_turn:
        d.abandon()
      # an offered lane change is cancelled out loud, its abort held on this planner; one under way goes on
      self.lanes.cancel('handedBack')
      self.paint_rule_lanes.cancel('handedBack')
    self.acting = acting

    # too fast to make the turn: skipped, so nothing below acts on it (no desire, no cap, no turn holds or lane changes)
    desire_out = self.desires.maneuver_id == rf.maneuver_id and self.desires.started
    skip = self.skip.update(upcoming, rf.distance, rf.maneuver_id, v_ego, desire_out, right_hand_traffic)
    maneuver = None if skip.skipped else upcoming
    # navd reroutes on it and selfdrived alerts: only while TurnPilot is driving the route (handback outranks it)
    plan.tooFastToTurn = skip.skipped and acting

    act_on = maneuver if acting and rf.maneuver_id != self.dropped_id else None
    held = act_on is None and rf.maneuver_id == self.turn_held_id
    desire_on = maneuver if held else act_on
    cap_on = act_on if act_on is not None else (maneuver if rf.maneuver_id == self.turn_cap_id else None)

    lanes = model_lanes_from_msg(model)
    desire = self.desires.update(desire_on, rf.distance, rf.maneuver_id, rf.passed, v_ego, cs.steeringAngleDeg,
                                 lanes.keep_left_ok, lanes.keep_right_ok, DT_MDL, default_yaw_rate(cs, model),
                                 lat_active=sm['carControl'].latActive, replaced=rf.replaced,
                                 may_start=skip.feasible_now and not held)
    if held and desire not in (Desire.none, CLEAR_DESIRE) and desire != self.last_desire:
      desire = Desire.none  # held: the desire out stays out, but never a new rising edge (a repeat pulse under way)
    plan.desire = self.last_desire = desire
    # carState or modelV2 failing their checks: no change is offered, and one offered aborts. Not acting: act_on is None,
    # so nothing is offered (an offer at the handback was cancelled above, as handedBack)
    lane_kw = {'left_blindspot': cs.leftBlindspot, 'right_blindspot': cs.rightBlindspot, 'maneuver_id': rf.maneuver_id,
               'inputs_valid': inputs_valid}
    lane_args = (act_on, rf.distance, lanes, model.meta.laneChangeState, v_ego, cs.leftBlinker, cs.rightBlinker, objects,
                 driver_signals, right_hand_traffic, DT_MDL, paint)
    self.lanes.paint_rule = paint_rule
    lc = self.lanes.update(*lane_args, **lane_kw)
    plan.laneChange, plan.signalPrompt, lane_reason = lc.lane_change, lc.prompt, lc.reason
    plan.laneChangeOffer, plan.laneChangeCountdown, plan.laneChangeAbort = lc.offer, lc.countdown, lc.abort
    rule = lc if paint_rule else self.paint_rule_lanes.update(*lane_args, **lane_kw)  # its offers and aborts are never raised
    plan.laneChangePaintRule = rule.lane_change if rule.lane_change != LaneChange.none else rule.prompt
    found = lane_mismatch(act_on, rf.distance, rf.through_lanes, rf.through_distance, lanes, v_ego, right_hand_traffic) \
      if acting else LaneMismatch.none  # handback outranks it, and with the maneuver dropped it would misread the lane
    plan.laneMismatch = self._lane_mismatch(found)
    # the scene's holds only ever brake, so they keep the real maneuver through a handback
    decision = self.scene.update(objects, available, maneuver, rf.distance, v_ego, right_hand_traffic, rf.maneuver_id, cs.gasPressed)
    plan.accelCap = decision.accel_cap
    # past a bend or a straight-on step, the turn after it is slowed for already (route_follower.py: ahead)
    ahead_cap = turn_speed_cap(rf.ahead, rf.ahead_distance, v_ego, right_hand_traffic) if act_on is not None else NO_SPEED_CAP
    plan.speedCap = min(decision.speed_cap, turn_speed_cap(cap_on, rf.distance, v_ego, right_hand_traffic), ahead_cap)
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

  def plan(self, sm, scene, enabled: bool, shadow: bool, *args, long_control: bool | None, **kwargs):
    """update(), but a disabled plan while openpilot doesn't control speed (on a Tesla, without alpha longitudinal):
    its speed caps would do nothing there, and the turn desires would go out at whatever speed stock ACC holds.
    long_control None: card hasn't written CarParams yet, so it isn't known. A disabled plan while TurnPilot is on
    carries its fault, which selfdrived shows (turnpilotUnavailable): TurnPilot never stops working unannounced.
    It also carries a model fallback's handback, which outranks the fault (see the module docstring): a fallback lasts
    the drive, so it is noted here too, where update() isn't reached."""
    if enabled and not long_control:
      self.stop('fault')
      self.watch_model(sm['modelV2'], enabled)
      handback = Handback.modelFellBack if self.fell_back else Handback.none
      msg = disabled_plan(shadow, CAR_UNKNOWN_REASON, handback=handback) if long_control is None else \
        disabled_plan(shadow, NO_LONG_REASON, Fault.noLongitudinal, handback)
      self.fill_stop_abort(msg.turnpilotPlan)
      return msg
    try:
      return self.update(sm, scene, enabled, shadow, *args, **kwargs)
    except Exception:
      log_error("turnpilotd.plan_failed")
      self.stop('fault')
      msg = disabled_plan(shadow, ERROR_REASON, Fault.error, Handback.modelFellBack if self.fell_back else Handback.none)
      self.fill_stop_abort(msg.turnpilotPlan)
      return msg


def disabled_plan(shadow: bool, reason: str, fault=Fault.none, handback=Handback.none):
  """A plan nothing acts on: no desire, no lane change, no limits. handback: a model fallback, which selfdrived still
  raises through a disabled plan (hooks.fresh_plan)."""
  msg = messaging.new_message('turnpilotPlan')
  plan = msg.turnpilotPlan
  plan.state, plan.shadow, plan.reason, plan.fault, plan.handback = State.disabled, shadow, reason, fault, handback
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


def lane_paint_active(params) -> bool:
  """TurnPilotLanePaintActive; off when this build's params library predates the key (rebuild common/ after a merge)."""
  try:
    return params.get_bool("TurnPilotLanePaintActive")
  except UnknownKeyName:
    cloudlog.error("turnpilotd: TurnPilotLanePaintActive unknown to this build's params, lane paint rule off; run scons")
    return False


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
  enabled = shadow = paint_rule = False
  driver_signals, right_hand_traffic = True, True
  long_control: bool | None = None
  invalid_frames = PLAN_INPUT_GRACE_FRAMES
  while True:
    sm.update(0)
    if sm.frame % int(1. / DT_MDL) == 0:
      enabled = params.get_bool("TurnPilotEnabled")
      shadow = params.get_bool("TurnPilotShadowMode")
      paint_rule = lane_paint_active(params)
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

    scene = scene_msg.as_reader().turnpilotScene
    paint = lane_paint_from_msg(sm['turnpilotDetections'].lanePaint) if scene.available and sm.seen['turnpilotDetections'] else ()
    # the lane planner takes this frame's checks (an offered change aborts at once); the plan's valid has a grace
    valid = sm.all_checks(['carState', 'modelV2'])
    msg = tp.plan(sm, scene, enabled, shadow, driver_signals, right_hand_traffic, paint, paint_rule, valid, long_control=long_control)
    invalid_frames = 0 if valid else invalid_frames + 1
    msg.valid = invalid_frames < PLAN_INPUT_GRACE_FRAMES
    pm.send('turnpilotPlan', msg)
    if tp.cancel_route and not tp.turn_committed:  # a turn under way finishes first, its cap and desire held
      tp.cancel_route = False
      cloudlog.warning(f"turnpilot: driving model fell back to the small model, handing back{'' if shadow else ' and cancelling the route'}")
      if not shadow:
        params.remove("NavDestination")
    rk.keep_time()


if __name__ == "__main__":
  main()
