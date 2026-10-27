import time
from multiprocessing import Queue

from openpilot.common.realtime import Ratekeeper
from openpilot.tools.sim.bridge.common import QueueMessageType, SimulatorBridge
from openpilot.tools.sim.bridge.beamng.control import AccelController
from openpilot.tools.sim.bridge.beamng.sensors import BeamNGSensors
from openpilot.tools.sim.bridge.beamng.tesla import SimulatedTesla
from openpilot.tools.sim.bridge.beamng.world import BeamNGConfig, BeamNGState, BeamNGWorld

MANUAL_STEER_DEG = 90.0  # steering wheel degrees for a full manual steer input
STEER_NM_PER_UNIT = 12.5  # manual steer input as torsion bar torque; the port reads > 1 Nm as the driver steering
SET_SPEED_STEP_KPH = 5.0
DT = 0.01


class BeamNGBridge(SimulatorBridge):
  """openpilot on this Mac driving a car in BeamNG.tech, as a comma four in a Tesla Model 3.

  Queue commands (tools/sim/bridge/common.py control_cmd_gen, from the keyboard, the CLI or a scenario script):
  cruise_up engages (or raises the set speed by 5 km/h), cruise_down engages at the current speed (or lowers it),
  cruise_cancel cancels, plus upstream's steer_/throttle_/brake_/blinker_ commands, reset and quit.
  `scenario` runs in the bridge process with the world after the first sensor data: scenario(bridge) -> None, called at
  100 Hz until it returns True (done)."""

  def __init__(self, cfg: BeamNGConfig, engage: bool = False, set_speed_kph: float = 0.0, scenario=None, test_run: bool = False):
    super().__init__(dual_camera=True, high_quality=False)
    self.cfg = cfg
    self.auto_engage = engage
    self.set_speed_kph = set_speed_kph
    self.scenario = scenario
    self.test_run = test_run
    self.engage_t = 0.0

  def spawn_world(self, q: Queue):
    return BeamNGWorld(self.cfg)

  def handle_command(self, cmd: str, state: BeamNGState):
    m = cmd.split('_')
    car = self.car
    if m[0] == "cruise":
      if m[1] == "up":
        car.adjust_set_speed(SET_SPEED_STEP_KPH) if car.engaged else car.engage(state.wheel_speed, self.set_speed_kph)
      elif m[1] == "down":
        car.adjust_set_speed(-SET_SPEED_STEP_KPH) if car.engaged else car.engage(state.wheel_speed)
      elif m[1] == "cancel":
        car.cancel()
    elif m[0] == "steer":
      self.manual[0] = float(m[1])
    elif m[0] == "throttle":
      self.manual[1] = float(m[1])
    elif m[0] == "brake":
      self.manual[2] = float(m[1])
    elif m[0] == "blinker":  # the stalk latches: the same side again turns it off
      want = {"left": (True, False), "right": (False, True)}.get(m[1], (False, False))
      self.blinker = (False, False) if want == self.blinker else want
    elif m[0] == "ignition":
      state.ignition = not state.ignition
    elif m[0] == "reset":
      self.world.reset()
    elif m[0] == "quit":
      self.shutdown()

  def _run(self, q: Queue):
    self.world = self.spawn_world(q)
    self.car = SimulatedTesla()
    self.sensors = BeamNGSensors()
    self.accel = AccelController()
    state = self.simulator_state = BeamNGState()
    self.manual = [0.0, 0.0, 0.0]  # steer -1..1 (right positive, like upstream's keyboard), throttle, brake
    self.blinker = (False, False)
    rk = Ratekeeper(100, print_delay_threshold=None)
    scenario_done = self.scenario is None
    last_print = 0.0

    while self._keep_alive:
      self.manual = [0.0, 0.0, 0.0]
      while not q.empty():
        msg = q.get()
        if getattr(msg, 'type', None) == QueueMessageType.CONTROL_COMMAND:
          self.handle_command(msg.info, state)

      self.world.read_sensors(state)
      steer_in, gas, brake = self.manual
      state.user_gas, state.user_brake = gas, brake
      state.user_torque_nm = -steer_in * STEER_NM_PER_UNIT
      state.left_blinker, state.right_blinker = self.blinker

      self.sensors.update(state, self.world)
      self.car.update(state)

      if self.auto_engage and state.valid:
        self.engage_step()

      # what the Model 3's EPS and drive inverter would do with openpilot's DAS_steeringControl and DAS_control
      if self.car.lat_active() and steer_in == 0.0:
        steer = self.car.steer_cmd
      else:
        steer = -steer_in * MANUAL_STEER_DEG
      if self.car.long_active() and gas == 0.0 and brake == 0.0:
        throttle, brake_out = self.accel.update(self.car.accel_cmd, state.a_long, state.wheel_speed, DT)
      else:
        self.accel.reset()
        throttle, brake_out = gas, brake
        if gas == 0.0 and brake == 0.0 and not self.car.engaged and abs(state.wheel_speed) < 1.0:
          brake_out = 0.5  # nobody driving: hold the car instead of letting it creep in D
      self.world.apply_controls(steer, throttle, brake_out)
      state.is_engaged = self.car.engaged

      if not scenario_done:
        scenario_done = bool(self.scenario(self))

      if self.world.exit_event.is_set():
        self.shutdown()

      now = time.monotonic()
      if not self.test_run and now - last_print > 1.0:
        last_print = now
        self.print_status()
      self.started.value = True
      rk.keep_time()

  def engage_step(self):
    """Engage once openpilot can: cruise on when engageable; if openpilot didn't take the rising edge, cancel and retry."""
    sm = self.car.sm
    sm.update(0)
    now = time.monotonic()
    if sm['selfdriveState'].enabled:
      self.auto_engage = False
    elif not self.car.engaged and sm['selfdriveState'].engageable and now - self.engage_t > 1.0:
      self.car.engage(self.simulator_state.wheel_speed, self.set_speed_kph)
      self.engage_t = now
    elif self.car.engaged and now - self.engage_t > 2.0:
      self.car.cancel()
      self.engage_t = now

  def print_status(self):
    s = self.simulator_state
    c = self.car
    print(f"v {s.wheel_speed:5.2f} m/s  angle {s.steering_angle:6.1f}  cruise {'ON ' if c.engaged else 'off'} set {c.set_speed_kph:5.1f} km/h  " +
          f"lat {'on ' if c.lat_active() else 'off'} cmd {c.steer_cmd:6.1f}  long {'on ' if c.long_active() else 'off'} " +
          f"accel {c.accel_cmd:5.2f} (meas {s.a_long:5.2f})  vehicle link {self.world.vehicle_rate:4.1f} Hz", flush=True)

