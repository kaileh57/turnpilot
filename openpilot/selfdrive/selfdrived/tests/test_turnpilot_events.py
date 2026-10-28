"""TurnPilot's lane change events: the alerts selfdrived builds from turnpilotPlan."""
import time

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.selfdrive.selfdrived.events import ET, Events

EventName = log.OnroadEvent.EventName
AudibleAlert = log.SelfdriveState.AudibleAlert


class FakeSM(dict):
  """SubMaster's dicts: the plan received just now and valid, or never received (hooks.fresh_plan reads recv_time)."""
  def __init__(self, plan=None, alive=True):
    super().__init__(turnpilotPlan=plan)
    received = alive and plan is not None
    self.alive = {'turnpilotPlan': received}
    self.valid = {'turnpilotPlan': received}
    self.recv_time = {'turnpilotPlan': time.monotonic() if received else None}


def plan(**fields):
  msg = messaging.new_message('turnpilotPlan')
  for k, v in fields.items():
    setattr(msg.turnpilotPlan, k, v)
  return msg.as_reader().turnpilotPlan


def alerts(name, sm):
  events = Events()
  events.add(name)
  return events.create_alerts([ET.WARNING], [car.CarParams.new_message(), car.CarState.new_message(), sm, False, 0, 0])


def test_lane_change_warning_left_and_right():
  for name, side, chime in ((EventName.turnpilotLaneChangeLeft, 'left', AudibleAlert.tpLaneChangeLeft),
                            (EventName.turnpilotLaneChangeRight, 'right', AudibleAlert.tpLaneChangeRight)):
    a, = alerts(name, FakeSM(plan(laneChangeOffer=side, laneChangeCountdown=2.5)))
    assert a.alert_text_1 == f'Lane Change {side.capitalize()} in 3' and a.audible_alert == chime
    a, = alerts(name, FakeSM(plan(laneChangeOffer=side)))
    assert a.alert_text_1 == f'Signal {side.capitalize()} to Change Lanes'
    assert a.alert_type == f'turnpilotLaneChange{side.capitalize()}/warning'  # separate names: a chime per side later


def test_lane_change_aborted():
  a, = alerts(EventName.turnpilotLaneChangeAborted, FakeSM(plan(laneChangeAbort='blindspot')))
  assert a.alert_text_1 == 'Lane Change Cancelled' and a.alert_text_2 == 'Car in blind spot'
  assert a.audible_alert == AudibleAlert.tpLaneChangeAbort and a.duration >= 100  # 1 s at least, so it is seen
  a, = alerts(EventName.turnpilotLaneChangeAborted, FakeSM(None, alive=False))
  assert a.alert_text_2 == 'TurnPilot not responding'


def test_turnpilot_chimes_on_skip_and_handback():
  a, = alerts(EventName.turnpilotTooFastToTurn, FakeSM(plan()))
  assert a.audible_alert == AudibleAlert.tpTooFastTurn
  for name in (EventName.turnpilotDetectorStopped, EventName.turnpilotModelFellBack):
    events = Events()
    events.add(name)
    a, = events.create_alerts([ET.SOFT_DISABLE], [car.CarParams.new_message(), car.CarState.new_message(), FakeSM(plan()), False, 300, 0])
    assert a.alert_text_1 == 'TAKE CONTROL IMMEDIATELY' and a.audible_alert == AudibleAlert.tpHandback
