"""A Tesla Model 3 (HW3, 3-bit DAS_steeringControl firmware) for openpilot's car port, in place of upstream's simulated Honda.

Every message is the one opendbc/car/tesla/carstate.py reads, on the bus it reads it from (party = 0, autopilot party = 2),
at the rate panda's tesla.h RX checks expect, with the values that mean "a healthy car". DI_autonomyHealth (0x054) goes
out on bus 0 in every batch, or opendbc's interface fingerprints the car as dashcam-only. Tesla has no cruise buttons:
the car's own cruise state (DI_state) engages openpilot, so this car owns that state, like a real Model 3's TACC.

openpilot's commands come back the way the car would see them: DAS_steeringControl and DAS_control parsed from sendcan
(bus 0) with opendbc's DBC, so they carry openpilot's own Tesla limits. Ported from commahack/sim-device/tools/simulated_tesla.py.
"""
import time

import openpilot.cereal.messaging as messaging
from opendbc.can import CANParser
from opendbc.can.packer import CANPacker
from openpilot.common.params import Params
from openpilot.selfdrive.pandad.pandad_api_impl import can_capnp_to_list, can_list_to_can_capnp

DBC = "tesla_model3_party"
PARTY, AUTOPILOT_PARTY = 0, 2  # opendbc/car/tesla/values.py CANBUS

GEAR_P, GEAR_R, GEAR_N, GEAR_D = 1, 2, 3, 4  # DI_gear
CRUISE_STANDBY, CRUISE_ENABLED = 1, 2  # DI_cruiseState
EAC_AVAILABLE, EAC_ACTIVE = 1, 2  # EPAS3S_eacStatus
ACC_ON, ACC_CANCEL_GENERIC_SILENT = 4, 13  # DAS_accState
ANGLE_CONTROL = 1  # DAS_steeringControlType
SET_SPEED_MIN_KPH, SET_SPEED_MAX_KPH = 5.0, 255.5  # DI_digitalSpeed is 9 bits of 0.5
GAS_DEAD_BAND = 0.5  # percent
COMMAND_TIMEOUT = 0.2  # s without a DAS message and the command is stale
RATE_FILTER = 0.3

# ticks at 100 Hz between sends, from panda's tesla.h RX checks: EPAS3S_sysStatus and DI_systemStatus 100 Hz; DI_speed,
# ESP_B, ESP_status and DAS_steeringControl 50 Hz; DAS_control 25 Hz; DI_state and UI_warning 10 Hz
EVERY = {"EPAS3S_sysStatus": 1, "DI_systemStatus": 1, "DI_autonomyHealth": 1, "DI_speed": 2, "ESP_B": 2, "ESP_status": 2,
         "SCCM_steeringAngleSensor": 2, "DAS_steeringControl": 2, "DAS_control": 4, "DI_state": 10, "UI_warning": 10,
         "DAS_status": 10, "DAS_settings": 10}


def clip(x, lo, hi):
  return min(max(x, lo), hi)


class TeslaCan:
  """The car's side of the CAN bus, without IPC. State fields (openpilot convention, left positive):
  wheel_speed m/s, steering_angle deg (steering wheel, Tesla ratio), user_gas 0..1, user_brake 0..1, user_torque_nm,
  left_blinker, right_blinker, gear (DI_gear)."""

  def __init__(self):
    self.packer = CANPacker(DBC)
    self.parser = CANParser(DBC, [], PARTY)
    for name in ("DAS_steeringControl", "DAS_control"):
      self.parser.vl[name]  # registers the message
    self.blindspot = (False, False)

    self.cruise_state = CRUISE_STANDBY
    self.set_speed_kph = 0.0
    self.last_angle = None
    self.steering_rate = 0.0
    self.steer_t = self.accel_t = -1e9
    self.steer_cmd = 0.0  # deg, openpilot convention
    self.steer_type = 0
    self.accel_cmd = 0.0
    self.acc_state = 0

  # *** cruise: the driver's stalk ***
  @property
  def engaged(self) -> bool:
    return self.cruise_state == CRUISE_ENABLED

  def engage(self, speed_ms: float, set_speed_kph: float = 0.0):
    """Pull the stalk: cruise on at the higher of the car's speed and the wanted set speed."""
    if not self.engaged:
      self.cruise_state = CRUISE_ENABLED
      self.set_speed_kph = clip(max(speed_ms * 3.6, set_speed_kph), SET_SPEED_MIN_KPH, SET_SPEED_MAX_KPH)

  def cancel(self):
    self.cruise_state = CRUISE_STANDBY

  def adjust_set_speed(self, delta_kph: float):
    if self.engaged:
      self.set_speed_kph = clip(self.set_speed_kph + delta_kph, SET_SPEED_MIN_KPH, SET_SPEED_MAX_KPH)

  # *** openpilot's commands, as the car reads them ***
  def parse_sendcan(self, can_packets, now=None):
    """can_packets: [(nanos, [(address, dat, src), ...]), ...] from sendcan."""
    updated = self.parser.update(can_packets)
    now = time.monotonic() if now is None else now
    if 0x488 in updated:
      sc = self.parser.vl["DAS_steeringControl"]
      self.steer_cmd = -sc["DAS_steeringAngleRequest"]  # the port negates the angle
      self.steer_type = int(sc["DAS_steeringControlType"])
      self.steer_t = now
    if 0x2b9 in updated:
      dc = self.parser.vl["DAS_control"]
      self.acc_state = int(dc["DAS_accState"])
      self.accel_cmd = dc["DAS_accelMin"]  # openpilot sends accelMin = accel, accelMax = max(accel, 0)
      self.accel_t = now
      if self.acc_state == ACC_CANCEL_GENERIC_SILENT:
        self.cancel()

  def lat_active(self, now=None) -> bool:
    now = time.monotonic() if now is None else now
    return self.steer_type == ANGLE_CONTROL and now - self.steer_t < COMMAND_TIMEOUT

  def long_active(self, now=None) -> bool:
    now = time.monotonic() if now is None else now
    return self.engaged and self.acc_state == ACC_ON and now - self.accel_t < COMMAND_TIMEOUT

  # *** the car's CAN ***
  def step(self, state, tick: int, dt: float = 0.01):
    """One 100 Hz tick: the car's own reactions, then its CAN as [(address, data, bus)]."""
    if self.engaged and state.user_brake > 0:  # as in the car, the brake ends cruise
      self.cancel()
    if self.last_angle is not None:
      self.steering_rate += RATE_FILTER * ((state.steering_angle - self.last_angle) / dt - self.steering_rate)
    self.last_angle = state.steering_angle
    return self.frames(state, tick)

  def frames(self, state, tick: int):
    speed_kph = state.wheel_speed * 3.6
    angle = clip(state.steering_angle, -819.2, 819.0)
    gas = clip(state.user_gas * 100.0, 0.0, 100.0)
    brake = state.user_brake > 0
    torque = clip(state.user_torque_nm, -20.5, 20.45)
    left, right = bool(state.left_blinker), bool(state.right_blinker)
    lbs, rbs = self.blindspot
    sigs = {
      "DI_autonomyHealth": (PARTY, {"DI_ah_dGearAvailable": 1, "DI_ah_rGearAvailable": 1, "DI_ah_pGearAvailable": 1}),
      "DI_speed": (PARTY, {"DI_vehicleSpeed": clip(speed_kph, -40, 285), "DI_uiSpeed": clip(round(speed_kph), 0, 254), "DI_uiSpeedUnits": 1}),
      "DI_systemStatus": (PARTY, {"DI_gear": state.gear, "DI_systemState": 5,  # DI_SYS_ENABLE
                                  "DI_accelPedalPos": gas if gas > GAS_DEAD_BAND else 0.0, "DI_brakePedalState": int(brake)}),
      # ESP_driverBrakeApply: 1 not applied, 2 driver applying brakes (0 is "not initialised")
      "ESP_status": (PARTY, {"ESP_driverBrakeApply": 2 if brake else 1, "ESP_brakeApply": int(brake)}),
      # the port negates angle and torsion bar torque (openpilot: positive is left)
      "EPAS3S_sysStatus": (PARTY, {"EPAS3S_internalSAS": -angle, "EPAS3S_internalSASQF": 1,
                                   "EPAS3S_eacStatus": EAC_ACTIVE if self.lat_active() else EAC_AVAILABLE,
                                   "EPAS3S_eacErrorCode": 0, "EPAS3S_handsOnLevel": 0, "EPAS3S_torsionBarTorque": -torque}),
      "DI_state": (PARTY, {"DI_cruiseState": self.cruise_state, "DI_digitalSpeed": self.set_speed_kph, "DI_speedUnits": 1,
                           "DI_autoparkState": 1}),
      "ESP_B": (PARTY, {"ESP_vehicleSpeed": clip(speed_kph, 0, 511), "ESP_wheelSpeedsQF": 1,
                        "ESP_vehicleStandstillSts": int(abs(state.wheel_speed) < 0.1)}),
      "UI_warning": (PARTY, {"buckleStatus": 1, "anyDoorOpen": 0, "leftBlinkerOn": int(left), "rightBlinkerOn": int(right),
                             "leftBlinkerBlinking": 2 if left else 0, "rightBlinkerBlinking": 2 if right else 0}),
      # the stock Autopilot computer's side of the harness: its steering sensor, settings and idle commands
      "SCCM_steeringAngleSensor": (AUTOPILOT_PARTY, {"SCCM_steeringAngle": -angle,
                                                     "SCCM_steeringAngleSpeed": clip(-self.steering_rate, -4096, 4095.5)}),
      "DAS_status": (AUTOPILOT_PARTY, {"DAS_blindSpotRearLeft": int(lbs), "DAS_blindSpotRearRight": int(rbs)}),
      "DAS_settings": (AUTOPILOT_PARTY, {"DAS_autosteerEnabled": 0, "DAS_aebEnabled": 1, "DAS_fcwEnabled": 1}),
      "DAS_control": (AUTOPILOT_PARTY, {"DAS_aebEvent": 0, "DAS_accState": ACC_ON}),
      "DAS_steeringControl": (AUTOPILOT_PARTY, {"DAS_steeringControlType": 0}),
    }
    return [self.packer.make_can_msg(name, bus, values) for name, (bus, values) in sigs.items() if tick % EVERY[name] == 0]


class SimulatedTesla(TeslaCan):
  """Same role as tools/sim/lib/simulated_car.SimulatedCar: publishes can and pandaStates, reads sendcan."""

  def __init__(self):
    super().__init__()
    self.pm = messaging.PubMaster(['can', 'pandaStates'])
    self.sm = messaging.SubMaster(['carParams', 'selfdriveState'])
    self.sendcan = messaging.sub_sock('sendcan', conflate=False, timeout=0)
    self.params = Params()
    self.idx = 0
    self.obd_multiplexing = False

  def read_sendcan(self):
    msgs = messaging.drain_sock_raw(self.sendcan)
    if msgs:
      self.parse_sendcan(can_capnp_to_list(msgs, msgtype='sendcan'))

  def send_panda_state(self, state):
    self.sm.update(0)
    if self.params.get_bool("ObdMultiplexingEnabled") != self.obd_multiplexing:
      self.obd_multiplexing = not self.obd_multiplexing
      self.params.put_bool("ObdMultiplexingChanged", True, block=True)
    # echo the safety mode card chose (tesla, plus the long-control flag with alpha longitudinal), or selfdrived raises
    # controlsMismatch
    configs = self.sm['carParams'].safetyConfigs
    dat = messaging.new_message('pandaStates', 1)
    dat.valid = True
    dat.pandaStates[0] = {
      'ignitionLine': state.ignition,
      'pandaType': "blackPanda",
      'controlsAllowed': True,
      'safetyModel': str(configs[0].safetyModel) if len(configs) else "tesla",
      'safetyParam': int(configs[0].safetyParam) if len(configs) else 0,
      'alternativeExperience': self.sm['carParams'].alternativeExperience,
    }
    self.pm.send('pandaStates', dat)

  def update(self, state):
    self.read_sendcan()
    if state.valid:
      self.pm.send('can', can_list_to_can_capnp(self.step(state, self.idx)))
    if self.idx % 10 == 0:  # pandaStates at its 10 Hz service rate
      self.send_panda_state(state)
    self.idx += 1
