from types import SimpleNamespace

from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.structs import car
from opendbc.car.can_definitions import CanData
from opendbc.car.car_helpers import interfaces
from opendbc.car.tesla.values import CAR, TeslaFlags
from openpilot.tools.sim.bridge.beamng.tesla import TeslaCan, GEAR_D

CANDIDATE = CAR.TESLA_MODEL_3


def car_state(**kw):
  s = {"wheel_speed": 10.0, "steering_angle": 15.0, "user_gas": 0.0, "user_brake": 0.0, "user_torque_nm": 0.0, "left_blinker": False,
           "right_blinker": False, "gear": GEAR_D}
  s.update(kw)
  return SimpleNamespace(**s)


def packets(car, state, ticks, start=0):
  return [(int((start + t) * 1e7), [CanData(*m) for m in car.step(state, start + t)]) for t in range(ticks)]


def fingerprint(car, state):
  finger = gen_empty_fingerprint()
  for _, frames in packets(car, state, 200):
    for f in frames:
      finger[f.src][f.address] = len(f.dat)
  return finger


class TestTeslaCan:
  def setup_method(self):
    self.car = TeslaCan()
    finger = fingerprint(self.car, car_state())
    self.CP = interfaces[CANDIDATE].get_params(CANDIDATE, finger, [], alpha_long=True, is_release=False, docs=False)
    self.CI = interfaces[CANDIDATE](self.CP)

  def update(self, state, ticks=100, start=1000):
    cs = None
    for nanos, frames in packets(self.car, state, ticks, start):
      cs = self.CI.update([(nanos, frames)])
    return cs

  def test_fingerprint_not_dashcam(self):
    finger = fingerprint(TeslaCan(), car_state())
    assert 0x054 in finger[0], "DI_autonomyHealth must be on bus 0 from the first frames"
    assert 0x293 in finger[2]
    assert not self.CP.dashcamOnly
    assert not self.CP.flags & TeslaFlags.MISSING_DAS_SETTINGS
    assert self.CP.openpilotLongitudinalControl
    assert self.CP.steerControlType == structs.CarParams.SteerControlType.angle

  def test_carstate(self):
    cs = self.update(car_state(wheel_speed=10.0, steering_angle=15.0, left_blinker=True))
    assert cs.canValid and not cs.canTimeout
    assert abs(cs.vEgoRaw - 10.0) < 0.1
    assert abs(cs.steeringAngleDeg - 15.0) < 0.11
    assert cs.gearShifter == structs.CarState.GearShifter.drive
    assert cs.leftBlinker and not cs.rightBlinker
    assert not (cs.seatbeltUnlatched or cs.doorOpen or cs.brakePressed or cs.gasPressed or cs.steeringPressed)
    assert not (cs.steerFaultPermanent or cs.steerFaultTemporary or cs.accFaulted or cs.invalidLkasSetting or cs.stockAeb)
    assert cs.cruiseState.available and not cs.cruiseState.enabled

  def test_engage_and_brake_cancel(self):
    self.update(car_state())
    self.car.engage(10.0, set_speed_kph=50.0)
    cs = self.update(car_state())
    assert cs.cruiseState.enabled
    assert abs(cs.cruiseState.speed - 50.0 / 3.6) < 0.2
    cs = self.update(car_state(user_brake=0.5))
    assert cs.brakePressed and not cs.cruiseState.enabled

  def test_pedals(self):
    cs = self.update(car_state(user_gas=0.3))
    assert cs.gasPressed

  def test_commands_round_trip(self):
    self.car.engage(10.0)
    self.update(car_state(steering_angle=0.0))
    msg = car.CarControl.new_message(enabled=True, latActive=True, longActive=True)
    msg.actuators.steeringAngleDeg = 30.0
    msg.actuators.accel = 1.2
    CC = msg.as_reader()
    sent = []
    for i in range(20):
      new_actuators, can_sends = self.CI.CC.update(CC, self.CI.CS, int(i * 1e7))
      sent.append((int(i * 1e7), [CanData(*m) for m in can_sends]))
    self.car.parse_sendcan(sent, now=0.0)
    # the angle the car sees is openpilot's rate-limited one, not the raw request
    assert abs(self.car.steer_cmd - new_actuators.steeringAngleDeg) < 0.11
    assert 0.0 < self.car.steer_cmd < 30.0
    assert abs(self.car.accel_cmd - 1.2) < 0.05
    assert self.car.lat_active(now=0.05) and self.car.long_active(now=0.05)
    assert not self.car.lat_active(now=1.0)

    msg.cruiseControl.cancel = True
    _, can_sends = self.CI.CC.update(msg.as_reader(), self.CI.CS, int(30 * 1e7))
    self.car.parse_sendcan([(int(30 * 1e7), [CanData(*m) for m in can_sends])], now=0.3)
    assert not self.car.engaged
