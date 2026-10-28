"""Map stop control: stop signs, yields and traffic signals on the route (navInstruction.controls), turned into stops.

The driving model doesn't stop for them (BeamNG, 2026-10-27: 0 of 31 reds, full stops at 2 of 8 stop signs), so
TurnPilot does, from the map. Each control becomes a scene_policy.Stop, composed with the pedestrian, oncoming and turn
limits by ScenePolicy (its reason, holding and the accel cap's jerk ramp). It runs every frame, whatever the route's
maneuver is: a skipped turn or a straight-through junction keeps its stop.

The rules (Kellen, 2026-10-27; /private/tmp/claude-501/stop_design.md):
- The approach: a control engages within engage_distance (braking at MAP_STOP_DECEL from v, a second's lag, a margin).
  From then on a speed cap on a MAP_STOP_DECEL curve to the stop point, which only ever comes down (a cap that rises and
  falls with a GPS fix hunts), re-seeded only at standstill short of the latch (stopped in a queue: the lead leaves, the
  car creeps up to the line). An accel cap where the curve is late, never past MAP_DECEL_MAX (-2.0 m/s^2, the speed
  cap's own floor): a map stop never takes stop()'s hard-stop path. A stop that needs more than that, checked when the
  control is first seen, when the driver's gas comes off, and after a jump in its distance, brakes at -2.0 and says so
  (cannotStopComfortably, repeating while it lasts). Never harder, never silent.
- The latch (HELD): slower than LATCH_SPEED within LATCH_NEAR_M of the stop point (LATCH_PAST_M past it), the car is held:
  shouldStop, speed cap 0. The latch is the policy's and position-free: GPS fixes, nav gaps, reroutes, a skipped turn,
  a handback and a fault leave it be (turnpilotd keeps this policy through a fault). Only its release rule lets go, and
  only TurnPilot turned off clears it. A car stopped further out (a queue) isn't latched: the MPC holds it behind the lead.
- Positive go only. A stop sign is released by a gas tap after DWELL_S held (no auto-go: no cross traffic sensing), and
  the driver is prompted then (stopSignGo). A signal is held at whatever the light, unless an accepted light source says
  green, steadily, for GREEN_STEADY_S: unknown, stale, absent, conflicting or unassociated lights all hold. The driver's
  gas tap releases it any time after it engages (seen it green while rolling up), and it prompts for that at once
  (signalGo) while the light is unknown. On the car tonight there is no light source, so every mapped signal is held.
  A tap on a red (the sim oracle's) still goes, with goingOnRed: the driver's right on red. Releasing one releases the
  controls RELEASE_SPAN_M past it too (the far side of a divided road).
- A green that goes yellow, red or unknown is decided once: stop when the stopping distance at YELLOW_DECEL after T_LAT
  fits, braking no harder than YELLOW_DECEL; otherwise committed, with proceedingOnYellow, and a later red never reverses
  it. A stop so decided is released by the next steady green.
- A yield caps the speed to YIELD_SPEED at its line, never stops the car, and goes once passed or on a gas tap. One whose
  road outranks every crossing one (crossSuspect: likely mapped for the cross road) is dropped and logged. Such a stop
  sign still stops, and says the approach is unverified.
- The driver's gas before a stop sign is an override; when it comes off past the line moving, the stop is dropped
  (stopOverrun), else the curve resumes from there (re-seeded, checked for feasibility). Past the line moving with no
  release is an overrun: dropped, logged, alerted. Driven off a held stop under the driver's foot is the same.
- An unprotected turn (Kellen, 2026-10-27: the same rule as stop signs and lights) is a routed far-side turn (a left on the
  right) across an oncoming approach (navInstruction.maneuverOpposing present, or unknown: no junction data is never
  clear) with no stop sign or signal of its own at the junction (one within PROTECT_BEFORE_M before it to PROTECT_PAST_M
  past it, whatever its state: a signal already holds, a stop sign already stops). turnpilotd hands it in each frame
  (update()'s turn) and it becomes a control of its own, kind unprotectedTurn, at the junction's hold line less
  MAP_MARGIN_M: the curve, the latch and the stop sign's gas rules, with no dwell: held, the driver is prompted at once
  (unprotectedTurnGo), and only a gas tap goes. A failing or missing controls status leaves the turn unprotected. One the
  car isn't held at goes with its maneuver (skipped too fast, rerouted away, passed), never with a fault (it carries on,
  dead-reckoned); a held one is position-free like any latch. After the go, scene_policy.py's oncoming and pedestrian
  holds may still stop the car: they only add holds.
- navd's controlsStatus other than ok (navd always sets it; none is a navd from before controls) raises
  mapControlsUnavailable and keeps "Map stops/signals: <status>" in the reason: no controls on tiles without control
  data is never shown as clear.
"""
import math
from dataclasses import dataclass, field

from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.turnpilot.control_tracker import JUMP_M, ControlTracker, TrackedControl
from openpilot.selfdrive.turnpilot.hooks import ABORT_HOLD_S, SPEED_CAP_ACCEL_MIN
from openpilot.selfdrive.turnpilot.route_follower import LEFT, TURN
from openpilot.selfdrive.turnpilot.scene_policy import HARD_STOP_MIN_M, HOLD_LINE_M, NO_ACCEL_CAP, STOP_LAG_S, Stop
from openpilot.selfdrive.turnpilot.turn_speed import NO_SPEED_CAP

MAP_STOP_DECEL = 1.5          # m/s^2, the braking curve to a map stop point
MAP_DECEL_MAX = SPEED_CAP_ACCEL_MIN  # -2.0 m/s^2: a map stop never brakes harder
YELLOW_DECEL = 2.0            # m/s^2 for a stop decided on a yellow (TurnPilotYellowDecel; 2.5 needs Kellen's sign-off)
YELLOW_DECEL_MAX = 2.5
CONTROL_LINE_M = 10.          # without a mapped line, the stop line is the junction less this (routing/controls.py)
LATCH_SPEED = 0.3             # m/s: slower than this...
LATCH_NEAR_M = 5.             # ...this close to the stop point...
LATCH_PAST_M = 2.             # ...or this far past it, the latch holds the car
HOLD_ACCEL = -0.5             # m/s^2, the held car's accel cap
DWELL_S = 2.0                 # s held at a stop sign before the driver's go counts
GREEN_STEADY_S = 0.5          # s of green before it counts (the sim oracle; a camera would need 1.0)
T_LAT = 0.8                   # s from a yellow to braking
COMMIT_SPEED = 2.0            # m/s: past the line faster than this, the car went
YIELD_SPEED = 5.0             # m/s at a yield's line
RELEASE_SPAN_M = 30.          # releasing a control releases those this far past it
OVERRUN_PAST_M = 2.           # m past the line, moving faster than COMMIT_SPEED with no release: an overrun
DRIVE_OFF_M = 2.              # m a held car moved under the driver's gas: driven off
LIGHT_ASSOC_M = 25.           # a light belongs to the signal control whose stop point is this close to its own
LIGHT_STALE_S = 0.6           # s: an older light message is stale (unknown)

UNPROTECTED = 'unprotectedTurn'  # MapControlStatus.Kind's name for it
UNPROTECTED_ROUTE = -1        # its key is (UNPROTECTED_ROUTE, maneuver id): a maneuver's, not a route's control
KEEP_TURN = 'keep'            # update()'s turn when the route isn't known this frame (a fault): the turn's stop carries on
MAP_MARGIN_M = 1.             # m short of the hold line an unprotected turn stops, as navd's controls stop short of their line
PROTECT_BEFORE_M = 30.        # a stop sign or signal this far before the junction (its line, mapped or the junction less 11)...
PROTECT_PAST_M = 2.           # ...to this far past it is the junction's own: the turn isn't unprotected

GO_STOP_SIGN = "Stop sign: check cross traffic, press accelerator to go"
GO_SIGNAL = "Signal ahead: light not checked, press accelerator to proceed"


def engage_distance(v_ego: float) -> float:
  """m from the stop point a control engages: braking at MAP_STOP_DECEL, a second's lag, 15 m to spare."""
  return v_ego ** 2 / (2. * MAP_STOP_DECEL) + v_ego * STOP_LAG_S + 15.


def curve(v_ego: float, d: float) -> float:
  return math.sqrt(2. * MAP_STOP_DECEL * max(d - v_ego * STOP_LAG_S, 0.))


def needed(v_ego: float, d: float) -> float:
  return -v_ego ** 2 / (2. * max(d, HARD_STOP_MIN_M))


def turn_stop_feasible(v_ego: float, distance: float) -> bool:
  """An unprotected turn distance m ahead can be stopped for at its hold line at MAP_DECEL_MAX or gentler."""
  return needed(max(v_ego, 0.), distance - HOLD_LINE_M - MAP_MARGIN_M) >= MAP_DECEL_MAX - 1e-9


def unprotected_turn(maneuver, right_hand_traffic: bool = True) -> bool:
  """A routed turn across the oncoming traffic of the road straight ahead: a junction turn to the far side (a left on the
  right) whose junction has an oncoming approach, or might have (opposing unknown: no junction data is never clear). A
  keep, a bend and a T's stem (opposing none) aren't. The junction's own stop sign or signal is MapControlPolicy's to
  check (protected())."""
  return maneuver is not None and maneuver.kind == TURN and (maneuver.side == LEFT) == right_hand_traffic and \
    maneuver.opposing != 'none'


def protected(controls, distance: float) -> bool:
  """A stop sign or signal of the junction distance m ahead stops the car there already."""
  return any(c.kind in ('stop', 'signal') and distance - PROTECT_BEFORE_M <= c.d <= distance + PROTECT_PAST_M for c in controls)


@dataclass
class Lights:
  lights: list | None = None   # [(state, distance)] from an accepted, fresh source; None without one
  source: str = 'none'         # TurnpilotLights.Source of the last message
  rejected: bool = False       # a source that isn't accepted here sent lights (lightSourceRejected)


NO_LIGHTS = Lights()


def light_input(msg, age_s: float, simulation: bool, oracle_param: bool) -> Lights:
  """Lights from a turnpilotLights message age_s old. The sim oracle only with SIMULATION=1 and TurnPilotSimLightOracle
  set: anything else that sends lights is rejected, out loud, and its lights are unknown. Stale is unknown."""
  if msg is None or str(msg.source) == 'none':
    return Lights()
  source = str(msg.source)
  if not (source == 'simOracle' and simulation and oracle_param):
    return Lights(None, source, rejected=True)
  if age_s > LIGHT_STALE_S:
    return Lights(None, source)
  return Lights([(str(light.state), float(light.distance)) for light in msg.lights], source)


@dataclass
class Shown:
  id: int
  kind: str
  distance: float
  action: str
  light_state: str = 'unknown'
  light_source: str = 'none'
  controls_status: str = 'none'
  line_mapped: bool = False


@dataclass
class MapDecision:
  stops: list = field(default_factory=list)
  prompt: str = 'none'
  alert: str = 'none'
  shown: Shown | None = None
  reason: str = ''               # the controls' status to show when nothing holds
  released_stopped: bool = False  # a hold let go this frame with the car stopped: turnpilotd re-pulses the turn
  gas_go: bool = False            # a control was released by the driver's gas tap this frame: that press is the go


@dataclass
class ControlState:
  cap: float = NO_SPEED_CAP
  infeasible: bool = False
  override: bool = False        # the driver's gas on the approach
  mode: str = 'stop'            # signals: 'stop', 'go' (a steady green), 'committed' (on a yellow)
  yellow: bool = False          # a stop decided on a yellow
  green_t: float = 0.
  light: str = 'unknown'
  dwell_t: float = 0.
  moved_m: float = 0.
  gas_in_dwell: bool = False
  done: str = ''                # 'released', 'passed', 'overrun', 'dropped'
  action: str = 'none'


def description(c: TrackedControl, turn_name: str = 'left') -> str:
  if c.kind == UNPROTECTED:
    return f"unprotected {turn_name}"
  if c.kind == 'stop':
    return "map stop sign (approach unverified)" if c.cross_suspect else "stop sign"
  return "signal" if c.junction else "signal (crossing)"


class MapControlPolicy:
  def __init__(self, yellow_decel: float = YELLOW_DECEL):
    self.tracker = ControlTracker()
    self.yellow_decel = yellow_decel
    self.held: tuple[int, int] | None = None  # the latched control's key
    self.prev_gas = False
    self._alert, self._alert_t = 'none', 0.
    self.turn: TrackedControl | None = None  # the unprotected turn's control (module docstring)
    self.turn_name = 'left'

  def stop_ahead(self) -> float | None:
    """m to the nearest stop sign or signal the car is going to stop at (not released, not going on a green), or None:
    a turn just past one is taken from a stop, never too fast (turn_skip.py)."""
    ds = [0. if c.held else c.d for c in self.tracker.controls.values() if c.kind in ('stop', 'signal') and
          (c.held or c.d > -LATCH_PAST_M) and not (c.state is not None and (c.state.done or c.state.mode != 'stop'))]
    return min(ds) if ds else None

  def turn_engaged(self, maneuver_id: int) -> bool:
    """The unprotected turn of this maneuver is engaged (stopping for it, held at it, or released from it)."""
    tc = self.turn
    return tc is not None and tc.key == (UNPROTECTED_ROUTE, int(maneuver_id)) and tc.engaged

  def held_stops(self) -> list:
    """The latch alone, for a faulted plan whose map update failed: a held car stays held."""
    return [Stop(HOLD_ACCEL, 0., True, "Stopped at a map stop", 0.)] if self.held is not None else []

  def _transient(self, alert: str) -> None:
    self._alert, self._alert_t = alert, ABORT_HOLD_S

  def _light(self, c: TrackedControl, controls, lights: Lights) -> str:
    """The light state for signal control c: the lights within LIGHT_ASSOC_M of it that have no nearer signal control,
    if they agree."""
    if lights.lights is None:
      return 'unknown'
    signals = [o for o in controls if o.kind == 'signal' and not (o.state is not None and o.state.done)]
    states = {s for s, dist in lights.lights
              if abs(dist - c.d) < LIGHT_ASSOC_M and min(signals, key=lambda o: abs(dist - o.d), default=c) is c}
    return states.pop() if len(states) == 1 else 'unknown'

  def _turn_control(self, turn, controls, v: float, dt: float) -> TrackedControl | None:
    """The unprotected turn's control this frame. turn: (maneuver id, m to its junction, 'left' or 'right') for an
    unprotected_turn() the car is taking (not skipped), None, or KEEP_TURN: not known this frame (turnpilotd faulted
    before it knew the route), the one there carries on, dead-reckoned."""
    tc = self.turn
    if tc is not None and tc.held:
      return tc  # position-free: only its release lets go
    if isinstance(turn, str):
      if tc is not None:
        tc.d -= v * dt
      return tc
    want = turn is not None and not protected(controls, turn[1])
    key = (UNPROTECTED_ROUTE, int(turn[0])) if turn is not None else None
    if tc is not None and (not want or tc.key != key):
      if tc.engaged and not (tc.state is not None and tc.state.done):
        cloudlog.event("turnpilot.unprotected_turn_dropped", key=list(tc.key), d=round(tc.d, 1), v=round(v, 2),
                       why='protected' if turn is not None and key == tc.key else 'maneuver')
      tc = None
    if want:
      d = turn[1] - HOLD_LINE_M - MAP_MARGIN_M
      if tc is None:
        tc = TrackedControl(key, UNPROTECTED, d)
      else:
        tc.jumped = abs(d - (tc.d - v * dt)) > JUMP_M
        tc.d = d
      self.turn_name = turn[2]
    self.turn = tc
    return tc

  def _release(self, c: TrackedControl, controls, cause: str, v: float, out: MapDecision) -> None:
    st = c.state
    st.done = 'released'
    out.gas_go = out.gas_go or cause == 'gas'
    if self.held == c.key:
      self.held, c.held = None, False
      out.released_stopped = out.released_stopped or v < LATCH_SPEED
    for o in controls:
      if c.kind == UNPROTECTED:
        break  # past a turn's hold line the route is on another road: nothing there is the same junction's
      if o is not c and c.d <= o.d <= c.d + RELEASE_SPAN_M and o.kind not in ('yield', UNPROTECTED):
        o.state = o.state or ControlState()
        o.state.done = 'released'
    cloudlog.event("turnpilot.map_stop_released", kind=c.kind, key=list(c.key), cause=cause, d=round(c.d, 1), v=round(v, 2))

  def _overrun(self, c: TrackedControl, v: float) -> None:
    c.state.done = 'overrun'
    if self.held == c.key:
      self.held, c.held = None, False
    self._transient('stopOverrun')
    cloudlog.event("turnpilot.map_stop_overrun", error=True, kind=c.kind, key=list(c.key), d=round(c.d, 1), v=round(v, 2))

  def update(self, nav, updated: bool, valid: bool, v_ego: float, gas_pressed: bool, lights: Lights = NO_LIGHTS,
             dt: float = DT_MDL, turn=None) -> MapDecision:
    """nav: the latest navInstruction, updated: it arrived this frame, valid: its valid flag; carState's vEgo and
    gasPressed; lights: light_input()'s; turn: (maneuver id, m to its junction, 'left' or 'right') of the route's
    unprotected_turn() the car is taking, or None."""
    v = max(v_ego, 0.)
    controls = self.tracker.update(nav, updated, valid, v, dt)
    tc = self._turn_control(turn, controls, v, dt)
    if tc is not None:
      controls = sorted(controls + [tc], key=lambda c: c.d)
    gas_edge, gas_lift = gas_pressed and not self.prev_gas, self.prev_gas and not gas_pressed
    self.prev_gas = gas_pressed
    out = MapDecision()
    prompts = []
    infeasible = False
    if any(c.engaged for c in self.tracker.lost):
      self._transient('mapControlsUnavailable')
    for c in controls:
      st = c.state = c.state or ControlState()
      if st.done:
        continue
      if c.kind == 'signal':
        st.light = self._light(c, controls, lights)
        st.green_t = st.green_t + dt if st.light == 'green' else 0.
      if c.held:
        self._held(c, controls, v, gas_pressed, gas_edge, dt, out, prompts)
      elif c.kind == 'yield':
        self._yield(c, v, gas_edge, out, prompts)
      else:
        infeasible = self._approach(c, controls, v, gas_pressed, gas_edge, gas_lift, out, prompts) or infeasible

    # what the driver sees and hears
    out.prompt = prompts[0] if prompts else 'none'
    # 'none' is a navd that predates controls: nothing to say about a route it never looked at
    status_bad = self.tracker.route and self.tracker.status not in ('ok', 'none')
    if status_bad:
      out.reason = f"Map stops/signals: {self.tracker.status}"
    self._alert_t = max(self._alert_t - dt, 0.)
    if infeasible:
      out.alert = 'cannotStopComfortably'
    elif self._alert_t > 0.:
      out.alert = self._alert
    elif status_bad:
      out.alert = 'mapControlsUnavailable'
    elif lights.rejected:
      out.alert = 'lightSourceRejected'
    live = [c for c in controls if c.state is not None and not c.state.done]
    pick = next((c for c in live if c.held), None) or next((c for c in live if c.engaged), None) or \
      next((c for c in live if c.d > -OVERRUN_PAST_M), None)
    if pick is not None:
      st = pick.state
      out.shown = Shown(pick.key[1] & 0xFFFF, pick.kind, pick.d, st.action, st.light, lights.source, self.tracker.status, pick.line_mapped)
      if not out.reason and pick.kind == 'signal' and st.mode == 'go' and lights.lights is not None:
        out.reason = "Green (sim oracle)"
    elif self.tracker.route:
      out.shown = Shown(0, 'none', 0., 'none', 'unknown', lights.source, self.tracker.status)
    return out

  def _held(self, c, controls, v, gas_pressed, gas_edge, dt, out, prompts) -> None:
    st = c.state
    st.dwell_t += dt
    st.moved_m += v * dt
    what = description(c, self.turn_name)
    cause, prompt = None, None
    if c.kind == UNPROTECTED:
      # no dwell: the driver looks across and goes; until then nothing else releases it
      prompt = 'unprotectedTurnGo'
      if gas_edge:
        cause = 'gas'
      reason = f"Stopped at {what} — tap gas to go"
    elif c.kind == 'stop':
      ready = st.dwell_t >= DWELL_S - 1e-9
      st.gas_in_dwell = st.gas_in_dwell or (gas_edge and not ready)  # ignored, and a drive-off on it is an overrun
      prompt = 'stopSignGo' if ready else None
      if ready and gas_edge:
        cause = 'gas'
      reason = f"Stopped at {what} — tap gas to go" if ready else f"Stopped at {what}"
    else:
      prompt = 'signalGo' if st.light == 'unknown' else None
      if st.green_t >= GREEN_STEADY_S - 1e-9:
        cause = 'green'
      elif gas_edge:
        cause = 'gas'
        if st.light == 'red':
          self._transient('goingOnRed')
      reason = f"Red light (sim oracle){'' if c.junction else ' (crossing)'}" if st.light == 'red' else \
        f"Stopped at {what} — tap gas to go"
    if cause is None and gas_pressed and st.moved_m > DRIVE_OFF_M and v > COMMIT_SPEED:
      # driven off under the driver's foot: lifting must not stop the car in the junction
      if c.kind == 'stop' and st.gas_in_dwell:
        self._transient('stopOverrun')
      elif c.kind == 'signal' and st.light == 'red':
        self._transient('goingOnRed')
      cause = 'drivenOff'
    if cause == 'green':  # a stop decided on a yellow is undone by the next steady green: back to going on it
      st.mode, st.yellow, st.action = 'go', False, 'released'
      self.held, c.held = None, False
      out.released_stopped = out.released_stopped or v < LATCH_SPEED
      cloudlog.event("turnpilot.map_stop_released", kind=c.kind, key=list(c.key), cause=cause, v=round(v, 2))
      return
    if cause is not None:
      st.action = 'released'
      self._release(c, controls, cause, v, out)
      return
    if prompt is not None:
      prompts.insert(0, prompt)  # the held control's prompt first
    st.action = 'awaitingDriver' if prompt is not None else 'held'
    out.stops.append(Stop(HOLD_ACCEL, 0., True, reason, 0.))

  def _yield(self, c, v, gas_edge, out, prompts) -> None:
    st = c.state
    if c.cross_suspect:
      st.done = 'dropped'
      cloudlog.event("turnpilot.map_yield_dropped", kind=c.kind, key=list(c.key), reason="crossSuspect", d=round(c.d, 1))
      return
    if not c.engaged:
      if c.d > engage_distance(v):
        return
      c.engaged = True
    if gas_edge or c.d < 0.:
      st.done, st.action = 'passed', 'released'
      return
    # a speed at a fixed point, not a stop: the curve down to YIELD_SPEED, no accel cap, never shouldStop
    st.cap = min(st.cap, math.sqrt(YIELD_SPEED ** 2 + 2. * MAP_STOP_DECEL * max(c.d - v * STOP_LAG_S, 0.)))
    st.action = 'braking' if st.cap < v + 0.5 else 'approaching'
    out.stops.append(Stop(NO_ACCEL_CAP, st.cap, False, "Yield ahead", c.d))
    prompts.append('yieldAhead')

  def _approach(self, c, controls, v, gas_pressed, gas_edge, gas_lift, out, prompts) -> bool:
    """A stop sign or signal not held: the curve, the decisions, the latch. True while it is infeasible."""
    st = c.state
    if not c.engaged:
      if c.d > engage_distance(v) or c.d < -OVERRUN_PAST_M:
        return False
      c.engaged = True
      st.cap = curve(v, c.d)
      self._check(c, v, 'engaged')
    elif c.jumped:
      st.cap = curve(v, c.d)
      self._check(c, v, 'jumped')

    if c.kind == 'signal':
      oracle = st.light != 'unknown'
      if st.mode == 'stop' and st.green_t >= GREEN_STEADY_S - 1e-9:
        st.mode, st.yellow = 'go', False
      elif st.mode == 'go' and st.light != 'green':
        # decided once: a later red never reverses a commit, and only a steady green undoes a stop
        if c.d >= v * T_LAT + v ** 2 / (2. * self.yellow_decel):
          st.mode, st.yellow, st.cap = 'stop', True, curve(v, c.d)
          cloudlog.event("turnpilot.map_signal_decided", decision='stop', light=st.light, d=round(c.d, 1), v=round(v, 2))
        else:
          st.mode = 'committed'
          self._transient('proceedingOnYellow')
          cloudlog.event("turnpilot.map_signal_decided", decision='go', light=st.light, d=round(c.d, 1), v=round(v, 2))
      if st.mode in ('go', 'committed'):
        st.action = 'released' if st.mode == 'go' else 'proceedingOnYellow'
        if c.d < -OVERRUN_PAST_M:
          st.done = 'passed'
        return False
      if gas_edge:  # the driver's go: seen it green while rolling up, or going on the red
        if st.light == 'red':
          self._transient('goingOnRed')
        st.action = 'released'
        self._release(c, controls, 'gas', v, out)
        return False
      if not oracle:
        prompts.append('signalGo')
      lit = {'red': "Red light (sim oracle)", 'yellow': "Yellow light (sim oracle)"}.get(st.light)
      reason = (lit + ('' if c.junction else ' (crossing)')) if lit else f"Signal ahead: light not checked{'' if c.junction else ' (crossing)'}"
    else:
      if c.kind == UNPROTECTED:
        reason = f"Unprotected {self.turn_name} ahead"
      else:
        reason = "Map stop sign, approach unverified" if c.cross_suspect else "Stop sign ahead"
      if gas_pressed:
        st.override = True
      elif st.override and gas_lift:
        st.override = False
        if c.d < 0. and v > COMMIT_SPEED:
          st.action = 'overrun'
          self._overrun(c, v)
          return False
        st.cap = curve(v, c.d)
        self._check(c, v, 'gasLifted')

    if c.d < -OVERRUN_PAST_M and v > COMMIT_SPEED and not gas_pressed:
      st.action = 'overrun'
      self._overrun(c, v)
      return False

    cv = curve(v, c.d)
    st.cap = cv if v < LATCH_SPEED else min(st.cap, cv)  # re-seeded only at standstill (a queue)
    need = needed(v, c.d)
    floor = min(MAP_DECEL_MAX, -self.yellow_decel) if st.yellow else MAP_DECEL_MAX
    accel = max(need, floor) if need < -MAP_STOP_DECEL else NO_ACCEL_CAP
    if st.infeasible and need >= floor:
      st.infeasible = False
    if v < LATCH_SPEED and -LATCH_PAST_M < c.d < LATCH_NEAR_M and not gas_pressed and self.held is None:
      self.held, c.held = c.key, True
      st.dwell_t, st.moved_m, st.infeasible = 0., 0., False
      cloudlog.event("turnpilot.map_stop_held", kind=c.kind, key=list(c.key), d=round(c.d, 1), light=st.light)
      self._held(c, controls, v, gas_pressed, False, 0., out, prompts)
      return False
    st.action = 'infeasible' if st.infeasible else 'braking' if (st.cap < v + 0.5 or accel < NO_ACCEL_CAP) else 'approaching'
    out.stops.append(Stop(accel, st.cap, False, reason, c.d))
    return st.infeasible

  def _check(self, c, v, when: str) -> None:
    """Can the car stop at -MAP_DECEL_MAX from here? If not it brakes at the floor, alerted (cannotStopComfortably)."""
    floor = min(MAP_DECEL_MAX, -self.yellow_decel) if c.state.yellow else MAP_DECEL_MAX
    c.state.infeasible = needed(v, c.d) < floor - 1e-9
    if c.state.infeasible:
      cloudlog.event("turnpilot.map_stop_infeasible", error=True, kind=c.kind, key=list(c.key), when=when, d=round(c.d, 1),
                     v=round(v, 2), needed=round(needed(v, c.d), 2))
