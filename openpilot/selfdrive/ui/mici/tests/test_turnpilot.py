from openpilot.cereal import log, messaging
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.lib.turnpilot import active_plan, signal_prompt, status_text


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
    from openpilot.selfdrive.ui.mici.layouts.settings.turnpilot import shown_url
    self.tp = turnpilot
    self.shown_url = shown_url

  def test_nothing_while_following(self):
    for state in ('noRoute', 'following', 'approaching'):
      assert status_text(plan(state=state, reason=''), True) == ('', '')
    assert status_text(None, True) == ('', '')

  def test_unavailable_shows_engaged_or_not(self):
    for engaged in (True, False):
      assert status_text(plan(state='unavailable', reason='detector stopped: take control'), engaged) == \
        ('TurnPilot unavailable', 'detector stopped: take control')

  def test_turning(self):
    assert status_text(plan(state='turning', desire=log.Desire.turnLeft), True) == ('Turning left', '')
    assert status_text(plan(state='turning', desire=log.Desire.keepRight), True) == ('Keeping right', '')
    # between desire pulses the maneuver gives the side
    assert status_text(plan(state='turning', maneuverModifier='sharp right'), True) == ('Turning right', '')
    assert status_text(plan(state='turning', maneuverModifier='uturn'), True) == ('Turning', '')

  def test_lane_change(self):
    assert status_text(plan(state='laneChange', laneChange='left'), True) == ('Lane change left', '')
    assert status_text(plan(state='laneChange', signalPrompt='right'), True) == ('Lane change right', '')
    assert status_text(plan(state='laneChange'), True) == ('Lane change', '')

  def test_holding_and_arrived(self):
    assert status_text(plan(state='holding', reason='pedestrian in path'), True) == ('Holding', 'pedestrian in path')
    assert status_text(plan(state='arrived', reason='car in target lane'), True) == ('Arrived', '')

  def test_shadow(self):
    assert status_text(plan(state='turning', desire=log.Desire.turnRight, shadow=True), True) == ('Turning right', 'shadow')
    assert status_text(plan(state='holding', reason='waiting for oncoming', shadow=True), True) == ('Holding', 'shadow: waiting for oncoming')
    assert status_text(plan(state='following', shadow=True), True) == ('', '')

  def test_signal_prompt(self):
    p = plan(state='laneChange', signalPrompt='left')
    assert signal_prompt(p, car_state()) == 'left'
    assert signal_prompt(p, car_state(rightBlinker=True)) == 'left'
    assert signal_prompt(p, car_state(leftBlinker=True)) == ''  # signaled: now just "Lane change left"
    assert signal_prompt(plan(state='laneChange', signalPrompt='right'), car_state(leftBlinker=True)) == 'right'
    assert signal_prompt(plan(state='laneChange'), car_state()) == ''
    assert signal_prompt(plan(state='holding', signalPrompt='left'), car_state()) == ''  # the hold is what to show
    assert signal_prompt(None, car_state()) == ''

  def test_prompt_alert(self):
    assert self.tp.prompt_alert(plan(state='laneChange'), car_state()) is None
    alert = self.tp.prompt_alert(plan(state='laneChange', signalPrompt='right'), car_state())
    assert alert.text1 == 'Signal Right to Change Lanes'
    # the warning's countdown, before the change may go
    counting = self.tp.prompt_alert(plan(state='laneChange', signalPrompt='right', laneChangeOffer='right', laneChangeCountdown=2.3),
                                    car_state())
    assert counting.text1 == 'Lane Change Right in 3'
    assert alert.status == log.SelfdriveState.AlertStatus.userPrompt
    assert alert.alert_type.startswith('preLaneChangeRight/')  # upstream's turn signal icon on that side
    shadow = self.tp.prompt_alert(plan(state='laneChange', signalPrompt='left', shadow=True), car_state())
    assert shadow.text2 == 'shadow mode'

  def test_active_plan(self):
    p = plan(state='turning')
    assert active_plan(FakeSM(p), True) is p
    assert active_plan(FakeSM(p), False) is None
    assert active_plan(FakeSM(p, alive=False), True) is None
    assert active_plan(FakeSM(p, valid=False), True) is None
    assert active_plan(FakeSM(plan(state='disabled')), True) is None

  def test_status_needs_engaged_unless_shadow(self):
    # the driver is steering: nothing is holding or turning the car
    assert status_text(plan(state='holding', reason='pedestrian in path'), False) == ('', '')
    assert status_text(plan(state='turning', desire=log.Desire.turnLeft), False) == ('', '')
    # shadow mode is for comparing with what the driver does, so it shows while disengaged
    assert status_text(plan(state='turning', desire=log.Desire.turnLeft, shadow=True), False) == ('Turning left', 'shadow')
    # the prompt is a navigation hint, shown either way
    assert signal_prompt(plan(state='laneChange', signalPrompt='left'), car_state()) == 'left'

  def test_pair_url_without_network(self):
    assert self.shown_url('http://127.0.0.1:8088/?t=abc') == 'no network'
    assert self.shown_url('http://172.20.10.2:8088/?t=abc') == 'http://172.20.10.2:8088/?t=abc'


class TestMapStatus:
  def test_route_refusal_and_download_on_the_status_line(self):
    from openpilot.selfdrive.ui.lib.turnpilot import map_status_text
    refused = {'state': 'error', 'code': 'outside_coverage', 'short': 'Destination outside mapped area',
               'message': "Outside TurnPilot's mapped area. TurnPilot can only navigate around Indianapolis and Seattle."}
    assert map_status_text(refused, True) == ('No route', 'Destination outside mapped area')
    assert map_status_text(refused, False) == ('', '')  # nothing once the destination is gone
    assert map_status_text({'state': 'error', 'message': 'Routing failed'}, True) == ('No route', 'Routing failed')
    assert map_status_text({'state': 'downloading', 'progress': 0.42}, True) == ('Downloading map', 'for this drive, 42%')
    assert map_status_text({'state': 'downloading', 'progress': None}, True) == ('Downloading map', 'for this drive')
    for state in ('idle', 'checking', 'ready'):
      assert map_status_text({'state': state}, True) == ('', '')
    assert map_status_text(None, True) == ('', '')

  def test_mapped_area_text(self, tmp_path):
    from openpilot.selfdrive.turnpilot.maps.coverage import CoverageCache, coverage_path, save
    from openpilot.selfdrive.turnpilot.maps.tests.fakes import CHICAGO, INDY, coverage_doc
    from openpilot.selfdrive.ui.lib.turnpilot import mapped_area_text
    cache = CoverageCache(tmp_path)
    assert mapped_area_text(cache, (INDY[1], INDY[0])) == 'no mapped area downloaded'
    save(coverage_doc(), tmp_path)
    assert mapped_area_text(cache, (INDY[1], INDY[0])) == 'Indianapolis area'
    assert mapped_area_text(cache, (CHICAGO[1], CHICAGO[0])) == "Outside TurnPilot's mapped area"
    assert mapped_area_text(cache, None) == 'car position unknown; mapped: Indianapolis and Seattle'
    assert mapped_area_text(cache, None, '/sim/tiles') == 'local tiles (developer), no geofence'
    coverage_path(tmp_path).write_text('garbage')
    assert mapped_area_text(cache, (INDY[1], INDY[0])) == 'saved mapped area is unreadable'
