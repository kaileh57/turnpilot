import numpy as np

# etk800 in realistic_automatic, roughly: full throttle gives ~4 m/s^2 at low speed, full brake ~9 m/s^2
THROTTLE_PER_ACCEL = 1 / 4.0
BRAKE_PER_ACCEL = 1 / 9.0
DRAG_THROTTLE_PER_MS = 0.004  # throttle to hold speed on the flat, per m/s
KP, KI = 0.15, 0.25
I_LIMIT = 0.3
STOP_SPEED = 0.3  # m/s: below this with a request to slow down, hold the brake
STANDSTILL_BRAKE = 0.5


class AccelController:
  """Throttle and brake for a wanted acceleration, from feedforward plus a PI on the measured acceleration.

  It stands in for the Model 3's drive inverter and iBooster following DAS_control: the car's own job, not openpilot's."""

  def __init__(self):
    self.i = 0.0

  def reset(self):
    self.i = 0.0

  def update(self, a_target: float, a_meas: float, v_ego: float, dt: float) -> tuple[float, float]:
    if v_ego < STOP_SPEED and a_target <= 0.0:
      self.i = 0.0
      return 0.0, STANDSTILL_BRAKE
    err = a_target - a_meas
    self.i = float(np.clip(self.i + KI * err * dt, -I_LIMIT, I_LIMIT))
    u = KP * err + self.i
    if a_target >= 0.0:
      u += a_target * THROTTLE_PER_ACCEL + DRAG_THROTTLE_PER_MS * v_ego
    else:
      u += a_target * BRAKE_PER_ACCEL
    throttle = float(np.clip(u, 0.0, 1.0))
    brake = float(np.clip(-u, 0.0, 1.0))
    return throttle, brake
