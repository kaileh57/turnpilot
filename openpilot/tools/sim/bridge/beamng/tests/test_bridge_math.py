import math

import numpy as np

from openpilot.tools.sim.bridge.beamng import frame
from openpilot.tools.sim.bridge.beamng.cameras import W, H
from openpilot.tools.sim.bridge.beamng.control import AccelController
from openpilot.tools.sim.bridge.beamng.scenario import heading_quat
from openpilot.tools.sim.bridge.beamng.world import Pose
from openpilot.tools.sim.bridge.beamng.yuv import NV12Converter
from openpilot.tools.sim.lib.camerad import rgb_to_nv12


def imu_reading(t, pos, yaw, acc=(0.0, 0.0, 0.0)):
  """An AdvancedIMU reading: sensor X forward, Y up, Z right, for a level car heading `yaw` (counterclockwise from east)."""
  fwd = [math.cos(yaw), math.sin(yaw), 0.0]
  right = [math.sin(yaw), -math.cos(yaw), 0.0]
  return {'time': t, 'pos': list(pos), 'dirX': fwd, 'dirY': [0.0, 0.0, 1.0], 'dirZ': right, 'accSmooth': list(acc)}


class TestNV12:
  def test_matches_upstream(self):
    rgba = np.random.default_rng(1).integers(0, 256, size=(H, W, 4), dtype=np.uint8)
    ref = np.frombuffer(rgb_to_nv12(np.ascontiguousarray(rgba[:, :, :3])), dtype=np.uint8)
    assert np.array_equal(NV12Converter(W, H).convert(rgba), ref)
    assert np.array_equal(NV12Converter(W, H).convert(np.ascontiguousarray(rgba[:, :, :3])), ref)


class TestPose:
  def test_left_turn_and_rest(self):
    # 10 m/s north, turning left at 0.2 rad/s, accelerating forward at 1 m/s^2
    v, r, a = 10.0, 0.2, 1.0
    hist = []
    for i in range(10):
      t = i * 0.01
      yaw = math.pi / 2 + r * t
      hist.append(imu_reading(t, (0.0, v * t + a * t * t / 2, 0.0), yaw, acc=(a, 0.0, 0.0)))
    p = Pose(hist, hist[-2:])
    assert abs(p.yaw_rate - r) < 1e-3
    assert abs(p.vel[1] - (v + a * 0.065)) < 0.01  # average over the 50 ms window
    assert abs(p.a_long - a) < 0.05
    assert abs(p.bearing - (90.0 - math.degrees(math.pi / 2 + r * 0.09)) % 360.0) < 1e-6  # compass
    up, left, back = p.accel
    # comma four sensor axes (up, left, back): at rest the accelerometer reads +g up; locationd's meas = [-v2, -v1, -v0]
    assert abs(up - 9.81) < 1e-6 and abs(left) < 1e-6 and abs(back + 1.0) < 1e-6
    gyro_up, _, _ = p.gyro
    assert abs(gyro_up - r) < 1e-3  # device yaw rate (z down) = -gyro_up: negative in a left turn


class TestFrame:
  def test_round_trip(self):
    for x, y, z in ((-717.1, 101.0, 118.7), (1351.0, 2198.0, 0.0), (-1603.0, -1216.0, 50.0)):
      lat, lon, h = frame.to_latlon(x, y, z)
      assert np.allclose(frame.from_latlon(lat, lon, h), (x, y, z), atol=1e-6)


class TestScenario:
  def test_heading_quat_matches_known_spawn(self):
    # the known-good spawn (0, 0, 0.3826834, 0.9238795) faces south-west
    assert np.allclose(heading_quat(math.radians(-135)), (0.0, 0.0, 0.3826834, 0.9238795), atol=1e-6)


class TestAccelController:
  def test_signs(self):
    c = AccelController()
    th, br = c.update(1.0, 0.0, 10.0, 0.01)
    assert th > 0 and br == 0
    c.reset()
    th, br = c.update(-2.0, 0.0, 10.0, 0.01)
    assert br > 0 and th == 0
    th, br = c.update(-0.5, 0.0, 0.1, 0.01)
    assert th == 0 and br > 0  # standstill hold
