"""The driver's part in a nav lane change on a Tesla. openpilot can't signal on that port, so turnpilotd prompts
"signal left" (turnpilotPlan.signalPrompt) and the driver works the stalk; DesireHelper then makes the change. The
stalk stays on until the change is made, then cancels as the wheel straightens, or goes off if the prompt does."""
import openpilot.cereal.messaging as messaging
from openpilot.cereal import custom, log

LaneChange = custom.TurnpilotPlan.LaneChange
STARTING = log.LaneChangeState.laneChangeStarting


class PromptedDriver:
  def __init__(self):
    self.sm = messaging.SubMaster(['turnpilotPlan', 'modelV2'])
    self.side = LaneChange.none
    self.started = False
    self.signals = 0  # prompts answered, for the logs

  def update(self) -> tuple[bool, bool]:
    """(left, right) blinker the driver holds now."""
    self.sm.update(0)
    prompt = self.sm['turnpilotPlan'].signalPrompt if self.sm.alive['turnpilotPlan'] else LaneChange.none
    lc_state = self.sm['modelV2'].meta.laneChangeState
    if self.side == LaneChange.none:
      if prompt != LaneChange.none:
        self.side, self.started = prompt, False
        self.signals += 1
    elif lc_state == STARTING:
      self.started = True
    elif self.started or prompt != self.side:
      self.side = LaneChange.none
    return self.side == LaneChange.left, self.side == LaneChange.right
