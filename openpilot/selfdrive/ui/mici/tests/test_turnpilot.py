from openpilot.cereal import log, messaging
from openpilot.common.test import OpenpilotTestCase


def plan(**fields):
  msg = messaging.new_message('turnpilotPlan', valid=True)
  for k, v in fields.items():
    setattr(msg.turnpilotPlan, k, v)
  return msg.as_reader().turnpilotPlan


def car_state(**fields):
  msg = messaging.new_message('carState', valid=True)
  for k, v in fields.items():
    setattr(msg.carState, k, v)
  return msg.as_reader().carState


class FakeSM:
  def __init__(self, p, alive=True, valid=True):
    self.alive = {'turnpilotPlan': alive}
    self.valid = {'turnpilotPlan': valid}
    self._plan = p

  def __getitem__(self, service):
    return self._plan


class TestTurnPilotOnroad(OpenpilotTestCase):
  def setup_method(self):
    # imported per test: ui_state reads Params at import, which needs the test's prefix
    from openpilot.selfdrive.ui.mici.onroad import turnpilot
    self.tp = turnpilot

  def test_nothing_while_following(self):
    for state in ('noRoute', 'following', 'approaching'):
      assert self.tp.status_text(plan(state=state, reason='detector unavailable')) == ('', '')
    assert self.tp.status_text(None) == ('', '')

  def test_turning(self):
    assert self.tp.status_text(plan(state='turning', desire=log.Desire.turnLeft)) == ('Turning left', '')
    assert self.tp.status_text(plan(state='turning', desire=log.Desire.keepRight)) == ('Keeping right', '')
    # between desire pulses the maneuver gives the side
    assert self.tp.status_text(plan(state='turning', maneuverModifier='sharp right')) == ('Turning right', '')
    assert self.tp.status_text(plan(state='turning', maneuverModifier='uturn')) == ('Turning', '')

  def test_lane_change(self):
    assert self.tp.status_text(plan(state='laneChange', laneChange='left')) == ('Lane change left', '')
    assert self.tp.status_text(plan(state='laneChange', signalPrompt='right')) == ('Lane change right', '')
    assert self.tp.status_text(plan(state='laneChange')) == ('Lane change', '')

  def test_holding_and_arrived(self):
    assert self.tp.status_text(plan(state='holding', reason='pedestrian in path')) == ('Holding', 'pedestrian in path')
    assert self.tp.status_text(plan(state='arrived', reason='car in target lane')) == ('Arrived', '')

  def test_shadow(self):
    assert self.tp.status_text(plan(state='turning', desire=log.Desire.turnRight, shadow=True)) == ('Turning right', 'shadow')
    assert self.tp.status_text(plan(state='holding', reason='waiting for oncoming', shadow=True)) == ('Holding', 'shadow: waiting for oncoming')
    assert self.tp.status_text(plan(state='following', shadow=True)) == ('', '')

  def test_signal_prompt(self):
    p = plan(state='laneChange', signalPrompt='left')
    assert self.tp.signal_prompt(p, car_state()) == 'left'
    assert self.tp.signal_prompt(p, car_state(rightBlinker=True)) == 'left'
    assert self.tp.signal_prompt(p, car_state(leftBlinker=True)) == ''  # signaled: now just "Lane change left"
    assert self.tp.signal_prompt(plan(state='laneChange', signalPrompt='right'), car_state(leftBlinker=True)) == 'right'
    assert self.tp.signal_prompt(plan(state='laneChange'), car_state()) == ''
    assert self.tp.signal_prompt(plan(state='holding', signalPrompt='left'), car_state()) == ''  # the hold is what to show
    assert self.tp.signal_prompt(None, car_state()) == ''

  def test_prompt_alert(self):
    assert self.tp.prompt_alert(plan(state='laneChange'), car_state()) is None
    alert = self.tp.prompt_alert(plan(state='laneChange', signalPrompt='right'), car_state())
    assert alert.text1 == 'Signal right'
    assert alert.status == log.SelfdriveState.AlertStatus.userPrompt
    assert alert.alert_type.startswith('preLaneChangeRight/')  # upstream's turn signal icon on that side
    shadow = self.tp.prompt_alert(plan(state='laneChange', signalPrompt='left', shadow=True), car_state())
    assert shadow.text2 == 'shadow mode'

  def test_active_plan(self):
    p = plan(state='turning')
    assert self.tp.active_plan(FakeSM(p), True) is p
    assert self.tp.active_plan(FakeSM(p), False) is None
    assert self.tp.active_plan(FakeSM(p, alive=False), True) is None
    assert self.tp.active_plan(FakeSM(p, valid=False), True) is None
    assert self.tp.active_plan(FakeSM(plan(state='disabled')), True) is None
