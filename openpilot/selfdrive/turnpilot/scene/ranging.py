"""Monocular ranging: project a box's bottom-center onto a flat ground plane.

Frames: openpilot's device and calib frames are x forward, y right, z down. Positions out of here are x forward and
y LEFT, in meters over the ground from the point under the road camera, in the calibrated frame.

The wide road camera on a real device is a fisheye. Matching road-camera features (pinhole, trusted focal length) to the
wide camera on the public demo routes fits an equidistant lens, r = f * theta, far better than a pinhole: rms 1.4 px vs
3.3 px over 0-33 deg on a comma four (os04c10, f 452), 1.1 px vs 1.4 px over 0-22 deg on the 1928x1208 camera (f 633).
Only the one point per box is undistorted. The sim renders a pinhole, so the lens follows the device type.
"""
from dataclasses import dataclass

import numpy as np

from openpilot.common.transformations.camera import CameraConfig, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler

DEFAULT_HEIGHT = 1.22   # m, calibrationd's initial camera height
MIN_HEIGHT = 0.5
MAX_RANGE = 200.        # m; farther ground points are noise
PINHOLE_DEVICE_TYPES = ('pc',)  # the sim's wide camera is a pinhole
# equidistant focal length of the wide road camera by its frame size, fitted on the demo routes (see above)
WIDE_FISHEYE_F = {(1344, 760): 452.0, (1928, 1208): 633.0}


@dataclass(frozen=True)
class Lens:
  width: int
  height: int
  f: float
  fisheye: bool = False  # equidistant: radius = f * angle off the optical axis

  @property
  def center(self) -> tuple[float, float]:
    return self.width / 2, self.height / 2

  def ray(self, u: float, v: float) -> np.ndarray:
    """Pixel -> ray in the camera's view frame (x right, y down, z forward)."""
    cx, cy = self.center
    dx, dy = (u - cx) / self.f, (v - cy) / self.f
    if not self.fisheye:
      return np.array([dx, dy, 1.0])
    theta = np.hypot(dx, dy)
    if theta < 1e-9:
      return np.array([0.0, 0.0, 1.0])
    s = np.sin(theta) / theta
    return np.array([dx * s, dy * s, np.cos(theta)])

  def project(self, ray: np.ndarray) -> tuple[float, float]:
    """Ray in the view frame -> pixel (the inverse of ray)."""
    cx, cy = self.center
    x, y, z = ray
    if not self.fisheye:
      return cx + self.f * x / z, cy + self.f * y / z
    r = np.hypot(x, y)
    theta = np.arctan2(r, z)
    k = self.f * theta / r if r > 1e-12 else 0.0
    return cx + k * x, cy + k * y


def road_lens(cam: CameraConfig) -> Lens:
  return Lens(cam.width, cam.height, cam.focal_length)


def wide_lens(cam: CameraConfig, device_type: str) -> Lens:
  if device_type in PINHOLE_DEVICE_TYPES or (cam.width, cam.height) not in WIDE_FISHEYE_F:
    return Lens(cam.width, cam.height, cam.focal_length)
  return Lens(cam.width, cam.height, WIDE_FISHEYE_F[(cam.width, cam.height)], fisheye=True)


def calib_from_view(rpy_calib, wide_from_device_euler=None) -> np.ndarray:
  """Rotation from a camera's view frame to the calibrated frame; the wide camera adds its offset from the road camera,
  as the UI's augmented_road_view does."""
  view_from_calib = view_frame_from_device_frame @ rot_from_euler(rpy_calib)
  if wide_from_device_euler is not None:
    view_from_calib = view_frame_from_device_frame @ rot_from_euler(wide_from_device_euler) @ rot_from_euler(rpy_calib)
  return view_from_calib.T


def ground_point(lens: Lens, calib_from_view_rot: np.ndarray, height: float, u: float, v: float) -> tuple[float, float] | None:
  """(x forward, y left) in m where the ray through pixel (u, v) meets flat ground `height` below the camera, or None when
  the ray doesn't go down (at or above the horizon) or the point is beyond MAX_RANGE."""
  ray = calib_from_view_rot @ lens.ray(u, v)   # calib frame: x fwd, y right, z down
  if ray[2] <= 1e-6:
    return None
  x, y = ray[0] * height / ray[2], ray[1] * height / ray[2]
  if x <= 0 or np.hypot(x, y) > MAX_RANGE:
    return None
  return float(x), float(-y)


def camera_height(calibration) -> float:
  h = list(calibration.height) if calibration is not None else []
  return float(h[0]) if len(h) and np.isfinite(h[0]) and h[0] > MIN_HEIGHT else DEFAULT_HEIGHT
