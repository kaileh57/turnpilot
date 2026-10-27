"""Scene policy: tracked pedestrians and cars into longitudinal limits.

The driving model keeps lane keeping, leads, lights and the turn itself. This covers what it doesn't: a
pedestrian in or entering the path, a pedestrian at the crosswalk a curb-side turn crosses, and oncoming
traffic at a turn across it (every such turn is treated as unprotected). Each becomes a stop point; the
tightest one wins and is turned into a speed cap on a comfortable braking curve, plus an accel cap when that
curve can't make it. shouldStop is only set at the stop point near standstill, because longcontrol's stopping
state ignores the accel target and ramps the brakes on its own.

A track is evidence, not fact: a stop that needs harder braking than the curve only gets it from a confirmed track
(enough hits at a high enough confidence), so a two-hit false positive slows the car on the curve instead of
stopping it hard, and the accel cap ramps in at a bounded jerk rather than stepping. A person moving along the path is
followed like a lead, not stopped for, and one on a moving bicycle or motorcycle is its rider, left to the driving
model like any vehicle. A crosswalk hold needs a steadily detected track with enough hits.

With no scene, the car still waits at the hold line of an unprotected far-side turn (a blind turn across traffic is
worse), and the plan's reason asks the driver to check traffic and press the gas; the UI shows it as a prompt with a
chime. A gas press releases the junction hold for that maneuver, not just while it's held: otherwise lifting off brings
the hold back, inside the junction if the car crept past the line, and only the scene coming back or rolling past the
line faster than COMMIT_SPEED would release it. ScenePolicy keeps that and the rest of what one frame can't know;
scene_policy() is one frame on its own.

Objects are TurnpilotScene.TrackedObject (or anything with the same fields): x m forward, y m LEFT in the
calibrated frame, vx/vy over-ground velocity in that frame (ego motion removed), cls compared by name, inPath
set by the tracker, age its hits. If a tracker publishes velocity relative to the car instead, a parked car ahead
reads as oncoming and the junction hold is conservative, not missed.
"""
import math
from collections import deque
from dataclasses import dataclass, field

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.hooks import SPEED_CAP_TAU, plan_accel
from openpilot.selfdrive.turnpilot.route_follower import LEFT, TURN
from openpilot.selfdrive.turnpilot.turn_speed import NO_SPEED_CAP

NO_ACCEL_CAP = 10.
VEHICLES = ("car", "truck", "bus", "motorcycle", "cyclist")
TWO_WHEELERS = ("cyclist", "motorcycle")

STOP_DECEL = 1.5            # m/s^2, the braking curve to a stop point
STOP_LAG_S = 1.0            # s the curve leads the car, the planner's speed-cap time constant
STOP_SPEED = 0.3            # m/s: slower than this...
STOP_NEAR_M = 1.5           # ...and this close to the stop point, shouldStop holds the car
HARD_STOP_MIN_M = 0.5
ACCEL_CAP_JERK = 5.0        # m/s^3: an accel cap tightens from -STOP_DECEL no faster than this (-3.5 in 0.4 s)

CONFIRM_HITS = 5            # braking harder than STOP_DECEL needs a track that has had this many hits...
CONFIRM_CONF = 0.6          # ...at this confidence (once: it stays confirmed while it's tracked)

PATH_HALF_WIDTH_M = 1.5     # corridor around the car's axis for objects the tracker hasn't flagged inPath
NEAR_PATH_M = 15.           # ...used for anything this close ahead
ENTER_T_S = 1.5             # a pedestrian walking into the corridor within this counts as in it,
ENTER_MAX_X_M = 40.         # this close: farther out a mono track's lateral speed is mostly range noise
PED_STOP_GAP_M = 5.0        # stop this far (camera to pedestrian) short of a pedestrian in the path
MOVING_PED_VX = 1.0         # m/s: a person moving along the path faster than this is followed, not stopped for,
FOLLOW_GAP_S = 1.0          # with this many seconds of their speed added to the gap
RIDER_OVERLAP = 0.3         # a person box this much inside a cyclist or motorcycle box (of the smaller box) is its rider,
RIDER_MIN_SPEED = 2.0       # m/s, when the two-wheeler moves faster than this: someone walking a bike stays a pedestrian

HOLD_LINE_M = 8.0           # m before the maneuver point where the car waits at a junction
HOLD_ENGAGE_MARGIN_M = 15.  # the junction hold engages within braking distance of the hold line plus this
COMMIT_SPEED = 2.0          # m/s: past the hold line and faster than this, the car is committed to its turn
TURN_ZONE_PAST_M = 5.       # m past the maneuver point the junction rules still apply
SCENE_SETTLE_S = 1.0        # s the scene must be back before an empty one clears a far-side turn: new tracks need hits
                            # and a velocity, and a detector running only now and then must not clear it between runs
BLIND_TURN_REASON = "blind turn: check traffic, press gas"

CROSSWALK_X_BEFORE_M = 6.   # crosswalk zone on the turn side, along the car's axis relative to the maneuver point
CROSSWALK_X_AFTER_M = 15.
CROSSWALK_Y_NEAR_M = 1.5    # and across it, to the turn side
CROSSWALK_Y_FAR_M = 15.
# A crosswalk hold needs a track that has had this many hits at this confidence, and is steady: this many of them in the
# last this many seconds. BeamNG, (-311, 337): a wide camera false positive 10 m to the side held a right turn on 2-4
# hits at 0.42-0.63, one new hit in any second; a pedestrian there is detected on most of the detector's 10 Hz runs.
CROSSWALK_HITS = 5
CROSSWALK_CONF = 0.5
CROSSWALK_STEADY_HITS = 3
CROSSWALK_STEADY_S = 1.0

ONCOMING_TTA_S = 5.         # hold while an oncoming vehicle reaches the junction sooner than this
ONCOMING_MIN_SPEED = 1.     # m/s toward the car
ONCOMING_Y_NEAR_M = 1.5     # oncoming lanes, to the turn side
ONCOMING_Y_FAR_M = 20.
ONCOMING_PAST_M = 5.        # a vehicle this far short of the maneuver point has cleared it


@dataclass
class SceneDecision:
  accel_cap: float = NO_ACCEL_CAP
  speed_cap: float = NO_SPEED_CAP
  should_stop: bool = False
  holding: bool = False   # the limits bite now
  reason: str = ""


@dataclass
class Stop:
  accel_cap: float
  speed_cap: float
  should_stop: bool
  reason: str


def stop_command(v_ego: float, d_stop: float) -> tuple[float, float, bool]:
  """(accel cap, speed cap, should stop) to stop d_stop m ahead."""
  d = max(d_stop, 0.)
  speed_cap = math.sqrt(2. * STOP_DECEL * max(d - v_ego * STOP_LAG_S, 0.))
  a_need = -v_ego ** 2 / (2. * max(d, HARD_STOP_MIN_M))
  accel_cap = a_need if a_need < -STOP_DECEL else NO_ACCEL_CAP
  return accel_cap, speed_cap, v_ego < STOP_SPEED and d < STOP_NEAR_M


def stop(v_ego: float, d_stop: float, reason: str, confirmed: bool = True, v_lead: float = 0.) -> Stop:
  """A stop point d_stop m ahead, moving away at v_lead (then the car only has to slow to v_lead by there). Unconfirmed,
  it brakes no harder than the curve: no accel cap, and a speed cap no further below v_ego than STOP_DECEL takes."""
  accel_cap, speed_cap, should_stop = stop_command(max(v_ego - v_lead, 0.), d_stop)
  speed_cap += v_lead
  if not confirmed and accel_cap < NO_ACCEL_CAP:
    accel_cap, speed_cap = NO_ACCEL_CAP, max(speed_cap, v_ego - STOP_DECEL * SPEED_CAP_TAU)
  return Stop(accel_cap, speed_cap, should_stop and v_lead <= 0., reason)


def confirmed_track(obj) -> bool:
  return obj.age >= CONFIRM_HITS and obj.conf >= CONFIRM_CONF


def crosswalk_walker(obj) -> bool:
  return obj.age >= CROSSWALK_HITS and obj.conf >= CROSSWALK_CONF


def in_path(obj) -> bool:
  """The tracker's inPath follows the planned path; the straight-ahead corridor here only backs it up close in
  and for pedestrians walking into it."""
  if obj.inPath or (obj.x < NEAR_PATH_M and abs(obj.y) < PATH_HALF_WIDTH_M):
    return True
  return obj.x < ENTER_MAX_X_M and obj.y * obj.vy < 0 and abs(obj.y + obj.vy * ENTER_T_S) < PATH_HALF_WIDTH_M


def box_overlap(a, b) -> float:
  """Intersection of two boxes (x0, y0, x1, y1) over the smaller one's area."""
  w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
  smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
  return w * h / smaller if w > 0 and h > 0 and smaller > 0 else 0.


def is_rider(person, objects) -> bool:
  """YOLO calls a cyclist's rider a person, and the rider's box ranges badly (it ends above the ground): a person box
  inside a moving two-wheeler's box in the same camera is the two-wheeler."""
  return len(person.box) == 4 and any(
    str(o.cls) in TWO_WHEELERS and str(o.camera) == str(person.camera) and len(o.box) == 4 and math.hypot(o.vx, o.vy) > RIDER_MIN_SPEED and
    box_overlap(person.box, o.box) >= RIDER_OVERLAP for o in objects)


def pedestrian_stops(objects, v_ego: float, is_confirmed=confirmed_track) -> list[Stop]:
  """Pedestrians in the path: a stop short of a standing or crossing one, a following gap behind one moving along it."""
  stops = []
  for obj in objects:
    if str(obj.cls) != "pedestrian" or obj.x <= 0. or not in_path(obj) or is_rider(obj, objects):
      continue
    if obj.vx > MOVING_PED_VX:
      stops.append(stop(v_ego, obj.x - PED_STOP_GAP_M - obj.vx * FOLLOW_GAP_S, "following pedestrian", is_confirmed(obj), obj.vx))
    else:
      stops.append(stop(v_ego, obj.x - PED_STOP_GAP_M, "pedestrian in path", is_confirmed(obj)))
  return stops


def junction_stops(objects, settled: bool, maneuver, distance, v_ego, right_hand_traffic, is_walker=crosswalk_walker) -> list[Stop]:
  """Holds at the hold line before a turn: the crosswalk of a curb-side turn, oncoming traffic across a far-side one, and
  a far-side one the scene can't vouch for (settled: available for SCENE_SETTLE_S)."""
  if maneuver is None or maneuver.kind != TURN or distance < -TURN_ZONE_PAST_M:
    return []
  d_hold = distance - HOLD_LINE_M
  if d_hold < 0. and v_ego > COMMIT_SPEED:
    return []  # committed: stopping in the junction is worse than finishing the turn
  if d_hold > v_ego ** 2 / (2. * STOP_DECEL) + HOLD_ENGAGE_MARGIN_M:
    return []
  side = maneuver.side  # the crosswalk and the oncoming lanes are both on the turn side; y is positive left
  curb_side = (side == LEFT) != right_hand_traffic
  if curb_side:
    walkers = [obj for obj in objects if str(obj.cls) == "pedestrian" and is_walker(obj) and
               distance - CROSSWALK_X_BEFORE_M < obj.x < distance + CROSSWALK_X_AFTER_M and
               CROSSWALK_Y_NEAR_M < obj.y * side < CROSSWALK_Y_FAR_M]
    return [stop(v_ego, d_hold, "pedestrian in crosswalk")] if walkers else []
  for obj in objects:
    toward = -obj.vx
    if str(obj.cls) in VEHICLES and ONCOMING_Y_NEAR_M < obj.y * side < ONCOMING_Y_FAR_M and \
       obj.x > distance - ONCOMING_PAST_M and toward > ONCOMING_MIN_SPEED and \
       max(obj.x - distance, 0.) / toward < ONCOMING_TTA_S:
      return [stop(v_ego, d_hold, "waiting for oncoming")]
  return [] if settled else [stop(v_ego, d_hold, BLIND_TURN_REASON)]


def decide(stops: list[Stop], v_ego: float, available: bool) -> SceneDecision:
  if not stops:
    return SceneDecision(reason="" if available else "detector unavailable")
  accel_cap, speed_cap = min(s.accel_cap for s in stops), min(s.speed_cap for s in stops)
  should_stop = any(s.should_stop for s in stops)
  reason = min(stops, key=lambda s: (not s.should_stop, plan_accel(s.accel_cap, s.speed_cap, v_ego))).reason
  holding = should_stop or accel_cap < NO_ACCEL_CAP or speed_cap < v_ego + 0.5
  return SceneDecision(accel_cap, speed_cap, should_stop, holding, reason)


def scene_policy(objects, available: bool, maneuver, distance: float, v_ego: float,
                 right_hand_traffic: bool = True) -> SceneDecision:
  """One frame on its own: evidence from this frame's tracks alone, no ramp, no gas release, an available scene trusted
  at once. objects: tracked objects (ignored when not available); maneuver: RouteFollower.upcoming or None."""
  objects = list(objects) if available else []
  stops = pedestrian_stops(objects, v_ego) + junction_stops(objects, available, maneuver, distance, v_ego, right_hand_traffic)
  return decide(stops, v_ego, available)


@dataclass
class TrackEvidence:
  hits: int                                     # the track's age when last seen
  recent: deque = field(default_factory=deque)  # times of its hits in the last CROSSWALK_STEADY_S
  confirmed: bool = False                       # earned a hard stop
  walker: bool = False                          # earned a crosswalk hold


class ScenePolicy:
  """scene_policy() with what one frame can't know: which tracks earned a hard stop or a crosswalk hold (kept while they're
  tracked), how long the scene has been back, the accel cap ramping in, and a gas press that released a junction hold."""

  def __init__(self):
    self.t = 0.
    self.tracks: dict[int, TrackEvidence] = {}
    self.available_s = 0.
    self.accel_cap = NO_ACCEL_CAP
    self.released_id: int | None = None  # the maneuver whose junction hold a gas press released

  def update(self, objects, available: bool, maneuver, distance: float, v_ego: float, right_hand_traffic: bool = True,
             maneuver_id: int | None = None, gas_pressed: bool = False, dt: float = DT_MDL) -> SceneDecision:
    """As scene_policy(), plus maneuver_id (RouteFollower.maneuver_id) and carState.gasPressed, once per frame dt apart."""
    self.t += dt
    objects = list(objects) if available else []
    self.available_s = self.available_s + dt if available else 0.
    self.update_evidence(objects)

    junction = junction_stops(objects, self.available_s >= SCENE_SETTLE_S, maneuver, distance, v_ego, right_hand_traffic,
                              lambda obj: self.tracks[obj.trackId].walker)
    if junction and gas_pressed and maneuver_id is not None:
      self.released_id = maneuver_id  # the driver looked and goes: stopping again further in would be worse
    if maneuver_id is not None and maneuver_id == self.released_id:
      junction = []
    stops = pedestrian_stops(objects, v_ego, lambda obj: self.tracks[obj.trackId].confirmed) + junction
    decision = decide(stops, v_ego, available)
    decision.accel_cap = self.ramp(decision.accel_cap, dt)
    return decision

  def update_evidence(self, objects) -> None:
    tracks = {}
    for obj in objects:
      ev = self.tracks.get(obj.trackId) or TrackEvidence(obj.age)
      ev.recent.extend([self.t] * max(obj.age - ev.hits, 0))
      ev.hits = obj.age
      while ev.recent and ev.recent[0] <= self.t - CROSSWALK_STEADY_S:
        ev.recent.popleft()
      ev.confirmed = ev.confirmed or confirmed_track(obj)
      ev.walker = ev.walker or (crosswalk_walker(obj) and len(ev.recent) >= CROSSWALK_STEADY_HITS)
      tracks[obj.trackId] = ev
    self.tracks = tracks

  def ramp(self, target: float, dt: float) -> float:
    """The accel cap toward target: at once when it eases, from -STOP_DECEL at ACCEL_CAP_JERK when it tightens."""
    self.accel_cap = target if target >= self.accel_cap else max(target, min(self.accel_cap, -STOP_DECEL) - ACCEL_CAP_JERK * dt)
    return self.accel_cap
