#!/usr/bin/env python3
"""A bridge with no simulator: a parked car with ignition on, gray comma four camera frames, IMU, GPS and DM.

It takes the sim stack onroad without BeamNG, to check that every process starts and stays up:
  openpilot/tools/turnpilot/launch_sim.sh   # in one terminal
  python -m openpilot.tools.turnpilot.null_bridge   # in another
"""
import threading
import time

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.tools.sim.lib.common import SimulatorState
from openpilot.tools.sim.lib.simulated_car import SimulatedCar
from openpilot.tools.sim.lib.simulated_sensors import SimulatedSensors
from openpilot.tools.turnpilot.fake_cameras import FakeCameras, gray_nv12


class Sensors(SimulatedSensors):
  # SimulatedSensors without its 1928x1208 camerad, FakeCameras sends the frames
  def __init__(self):
    self.pm = messaging.PubMaster(['accelerometer', 'gyroscope', 'gpsLocationExternal', 'driverStateV2', 'driverMonitoringState', 'peripheralState'])
    self.last_perp_update = 0
    self.last_dmon_update = 0


def send_frames():
  # its own thread like the upstream bridge, so frame copies don't add jitter to the 100 Hz CAN
  cams = FakeCameras()
  road, wide = gray_nv12(seed=0), gray_nv12(seed=1)
  rk = Ratekeeper(20, print_delay_threshold=None)
  while True:
    cams.send(road, wide)
    rk.keep_time()


def main():
  Params().put_bool("AlphaLongitudinalEnabled", True)  # as the upstream bridge does, so the Honda safety param matches
  state = SimulatorState()
  state.valid = True
  car, sensors = SimulatedCar(), Sensors()
  threading.Thread(target=send_frames, daemon=True).start()

  rk = Ratekeeper(100, print_delay_threshold=None)
  st = time.monotonic()
  while True:
    car.update(state)
    sensors.update(state, None)
    if rk.frame % 1000 == 0:
      print(f"null bridge up {time.monotonic() - st:.0f} s", flush=True)
    rk.keep_time()


if __name__ == "__main__":
  main()
