import math
import time

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from openpilot.common.realtime import DT_DMON
from openpilot.tools.sim.lib.simulated_sensors import SimulatedSensors

GPS_PERIOD = 0.1  # s, a ublox at 10 Hz


class BeamNGSensors(SimulatedSensors):
  """Upstream's SimulatedSensors without its 1928x1208 camerad (cameras.py publishes the os04c10 frames): one IMU message
  per 100 Hz tick, and gpsLocationExternal at 10 Hz in frame.py's projection, with a fix, so navd uses it."""

  def __init__(self):
    self.pm = messaging.PubMaster(['accelerometer', 'gyroscope', 'gpsLocationExternal', 'driverStateV2', 'driverMonitoringState',
                                   'peripheralState'])
    self.last_perp_update = self.last_dmon_update = self.last_gps_update = 0.0

  def send_imu_message(self, simulator_state):
    state = simulator_state
    dat = messaging.new_message('accelerometer', valid=True)
    dat.accelerometer.timestamp = dat.logMonoTime
    dat.accelerometer.init('acceleration')
    dat.accelerometer.acceleration.v = list(state.imu.accelerometer)
    self.pm.send('accelerometer', dat)

    dat = messaging.new_message('gyroscope', valid=True)
    dat.gyroscope.timestamp = dat.logMonoTime
    dat.gyroscope.init('gyroUncalibrated')
    dat.gyroscope.gyroUncalibrated.v = list(state.imu.gyroscope)
    self.pm.send('gyroscope', dat)

  def send_gps_message(self, simulator_state):
    state = simulator_state
    v = state.velocity  # BeamNG world: x east, y north, z up
    dat = messaging.new_message('gpsLocationExternal', valid=True)
    dat.gpsLocationExternal = {
      "unixTimestampMillis": int(time.time() * 1000),  # noqa: TID251
      "flags": 1,
      "hasFix": True,
      "satelliteCount": 12,
      "horizontalAccuracy": 1.0,
      "verticalAccuracy": 1.0,
      "speedAccuracy": 0.1,
      "bearingAccuracyDeg": 0.1,
      "vNED": [v.y, v.x, -v.z],
      "bearingDeg": state.bearing,
      "latitude": state.gps.latitude,
      "longitude": state.gps.longitude,
      "altitude": state.gps.altitude,
      "speed": math.hypot(v.x, v.y),
      "source": log.GpsLocationData.SensorSource.ublox,
    }
    self.pm.send('gpsLocationExternal', dat)

  def update(self, simulator_state, world=None):
    state = simulator_state
    if not state.valid:
      return
    now = time.monotonic()
    self.send_imu_message(state)
    if now - self.last_gps_update >= GPS_PERIOD:
      self.send_gps_message(state)
      self.last_gps_update = now
    if now - self.last_dmon_update > DT_DMON / 2:
      self.send_fake_driver_monitoring()
      self.last_dmon_update = now
    if now - self.last_perp_update > 0.25:
      self.send_peripheral_state()
      self.last_perp_update = now
