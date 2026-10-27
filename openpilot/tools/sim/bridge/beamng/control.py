import numpy as np

# etk800 in realistic_automatic, roughly: 0.2 throttle gives ~1 m/s^2 at low speed, full brake ~9 m/s^2
THROTTLE_PER_ACCEL = 0.22
BRAKE_PER_ACCEL = 1 / 9.0
DRAG_THROTTLE_PER_MS = 0.004  # throttle to hold speed on the flat, per m/s
KP, KI = 0.25, 0.08  # throttle per m/s of speed error, per m of distance error
I_LIMIT = 0.3
V_REF_SLACK = 1.5  # m/s: the reference speed never runs further than this from the car's
STOP_SPEED = 0.3  # m/s: below this with a request to slow down, hold the brake
# m/s^2 the car may run ahead of the request before the PI lets go of what it built up: in a tight turn at 4 m/s the
# car slowed at 1 m/s^2 on a +0.7 request, the PI wound up, and out of the turn the car surged at 4.5-4.8 m/s^2
# (on a 1.0 request), which openpilot's excessive actuation check (over 4 m/s^2 for 0.25 s) latched as a fault
A_OVERSHOOT = 1.5
STANDSTILL_BRAKE = 0.5


class AccelController:
  """Throttle and brake for a wanted acceleration: feedforward, plus a PI that keeps the car's speed on the wanted
  acceleration's integral. It stands in for the Model 3's drive inverter and iBooster following DAS_control.

  The feedback is on speed, not acceleration: the game's acceleration arrives in 100 ms batches and bounces with the
  suspension and gear shifts, and a PI on it cycled between full throttle and braking."""

  def __init__(self):
    self.reset()

  def reset(self):
    self.i = 0.0
    self.v_ref = None

  def update(self, a_target: float, v_ego: float, dt: float, a_meas: float | None = None) -> tuple[float, float]:
    """a_meas: the car's measured acceleration, m/s^2, only to stop a surge past the request."""
    if v_ego < STOP_SPEED and a_target <= 0.0:
      self.reset()
      return 0.0, STANDSTILL_BRAKE
    if self.v_ref is None or (a_meas is not None and a_meas > a_target + A_OVERSHOOT):
      self.v_ref, self.i = v_ego, min(self.i, 0.0)
    self.v_ref = float(np.clip(self.v_ref + a_target * dt, v_ego - V_REF_SLACK, v_ego + V_REF_SLACK))
    err = self.v_ref - v_ego
    self.i = float(np.clip(self.i + KI * err * dt, -I_LIMIT, I_LIMIT))
    u = KP * err + self.i
    if a_target >= 0.0:
      u += a_target * THROTTLE_PER_ACCEL + DRAG_THROTTLE_PER_MS * v_ego
    else:
      u += a_target * BRAKE_PER_ACCEL
    return float(np.clip(u, 0.0, 1.0)), float(np.clip(-u, 0.0, 1.0))
