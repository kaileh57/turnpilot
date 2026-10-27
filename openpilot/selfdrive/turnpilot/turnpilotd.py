#!/usr/bin/env python3
"""turnpilotd: turns the active route and the scene into desire pulses, lane-change requests and longitudinal limits.

Skeleton: publishes a disabled plan until the route follower, lane planner and scene logic land.
"""
from openpilot.cereal import custom, messaging
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper

NO_ACCEL_CAP = 10.
NO_SPEED_CAP = 100.


def main():
  params = Params()
  pm = messaging.PubMaster(["turnpilotPlan"])
  sm = messaging.SubMaster(["carState", "navInstruction", "navRoute", "modelV2"], ignore_alive=["navInstruction", "navRoute"])
  rk = Ratekeeper(20, print_delay_threshold=None)
  enabled = shadow = False
  while True:
    sm.update(0)
    if sm.frame % 20 == 0:
      enabled = params.get_bool("TurnPilotEnabled")
      shadow = params.get_bool("TurnPilotShadowMode")
    msg = messaging.new_message("turnpilotPlan")
    plan = msg.turnpilotPlan
    plan.state = custom.TurnpilotPlan.State.noRoute if enabled else custom.TurnpilotPlan.State.disabled
    plan.accelCap = NO_ACCEL_CAP
    plan.speedCap = NO_SPEED_CAP
    plan.shadow = shadow
    pm.send("turnpilotPlan", msg)
    rk.keep_time()


if __name__ == "__main__":
  main()
