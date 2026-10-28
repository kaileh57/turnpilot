import threading

from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal import log, messaging
from openpilot.cereal.messaging import SubMaster, PubMaster
from openpilot.selfdrive.ui.soundd import SELFDRIVE_STATE_TIMEOUT, Soundd, check_selfdrive_timeout_alert

AudibleAlert = log.SelfdriveState.AudibleAlert
AlertSize = log.SelfdriveState.AlertSize


class FakeSM(dict):
  """selfdriveState, carState and turnpilotPlan, all alive, valid and updated."""

  def __init__(self, prompt='right', state='laneChange', blinker=False, alert_size=AlertSize.none, sound=AudibleAlert.none):
    ss = messaging.new_message('selfdriveState', valid=True)
    ss.selfdriveState.alertSize = alert_size
    ss.selfdriveState.alertSound = sound
    cs = messaging.new_message('carState', valid=True)
    cs.carState.rightBlinker = blinker
    tp = messaging.new_message('turnpilotPlan', valid=True)
    tp.turnpilotPlan.state = state
    tp.turnpilotPlan.signalPrompt = prompt
    super().__init__(selfdriveState=ss.as_reader().selfdriveState, carState=cs.as_reader().carState,
                     turnpilotPlan=tp.as_reader().turnpilotPlan)
    self.alive = self.valid = self.updated = dict.fromkeys(self, True)


class TestSoundd(OpenpilotTestCase):
  def test_check_selfdrive_timeout_alert(self, mocker):
    sm = SubMaster(['selfdriveState'])
    pm = PubMaster(['selfdriveState'])

    cs = messaging.new_message('selfdriveState')
    cs.selfdriveState.enabled = True
    threading.Timer(0.01, pm.send, args=("selfdriveState", cs)).start()
    sm.update(100)
    assert sm.updated['selfdriveState']

    sm.recv_time['selfdriveState'] = 0
    clock = mocker.patch("openpilot.selfdrive.ui.soundd.time.monotonic", return_value=SELFDRIVE_STATE_TIMEOUT)
    assert not check_selfdrive_timeout_alert(sm)

    clock.return_value = SELFDRIVE_STATE_TIMEOUT + 0.1
    assert check_selfdrive_timeout_alert(sm)

    clock.return_value = SELFDRIVE_STATE_TIMEOUT + 10
    assert not check_selfdrive_timeout_alert(sm)

  def test_turnpilot_signal_prompt_chime(self):
    s = Soundd()

    def played():
      s.current_sound_frame = len(s.loaded_sounds[s.current_sound])

    s.get_audible_alert(FakeSM())
    assert s.current_alert == AudibleAlert.prompt
    played()
    s.get_audible_alert(FakeSM())
    assert s.current_alert == AudibleAlert.none  # once, not again while the prompt stays up

    s.get_audible_alert(FakeSM(blinker=True))  # signaled, so the prompt goes
    assert s.current_alert == AudibleAlert.none
    s.get_audible_alert(FakeSM())  # and a new prompt chimes again
    assert s.current_alert == AudibleAlert.prompt

    # none under an openpilot alert (the UI shows the alert, not the prompt), or with TurnPilot off
    for sm in (FakeSM(alert_size=AlertSize.small), FakeSM(state='disabled'), FakeSM(prompt='none')):
      s = Soundd()
      s.get_audible_alert(sm)
      assert s.current_alert == AudibleAlert.none

    # openpilot's own sound plays out first
    s = Soundd()
    s.get_audible_alert(FakeSM(sound=AudibleAlert.engage))
    assert s.current_alert == AudibleAlert.engage
    played()
    s.get_audible_alert(FakeSM())
    s.get_audible_alert(FakeSM())
    assert s.current_alert == AudibleAlert.prompt

  # TODO: add test with micd for checking that soundd actually outputs sounds
