"""The sim's driver waits out TurnPilot's lane change countdown before working the stalk."""
import openpilot.cereal.messaging as messaging
from openpilot.cereal import custom, log
from openpilot.tools.sim.bridge.beamng.driver import PromptedDriver

LaneChange = custom.TurnpilotPlan.LaneChange


class FakeSM(dict):
  def __init__(self):
    super().__init__()
    self.alive = {'turnpilotPlan': True}
    self['modelV2'] = messaging.new_message('modelV2').modelV2

  def update(self, timeout=0):
    pass

  def plan(self, prompt, countdown):
    msg = messaging.new_message('turnpilotPlan')
    msg.turnpilotPlan.signalPrompt, msg.turnpilotPlan.laneChangeOffer = prompt, prompt
    msg.turnpilotPlan.laneChangeCountdown = countdown
    self['turnpilotPlan'] = msg.as_reader().turnpilotPlan


def test_stalk_waits_for_the_countdown():
  driver = PromptedDriver.__new__(PromptedDriver)
  driver.sm, driver.side, driver.started, driver.signals = FakeSM(), LaneChange.none, False, 0
  for countdown in (3., 2., 0.5):  # the warning: prompted, but the countdown is still running
    driver.sm.plan(LaneChange.left, countdown)
    assert driver.update() == (False, False)
  driver.sm.plan(LaneChange.left, 0.)
  assert driver.update() == (True, False) and driver.signals == 1
  # held until the change starts, then let go once the prompt is gone
  msg = messaging.new_message('modelV2')
  msg.modelV2.meta.laneChangeState = log.LaneChangeState.laneChangeStarting
  driver.sm['modelV2'] = msg.as_reader().modelV2
  assert driver.update() == (True, False)
  driver.sm['modelV2'] = messaging.new_message('modelV2').as_reader().modelV2  # the change is done
  driver.sm.plan(LaneChange.none, 0.)
  assert driver.update() == (False, False)
