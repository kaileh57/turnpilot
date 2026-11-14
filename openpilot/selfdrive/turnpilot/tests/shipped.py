"""The shipped settings for test_turnpilotd's Sim: each checked against its params_keys.h default, so a test that says it
runs the shipped build does."""
import re
from pathlib import Path

from openpilot.selfdrive.turnpilot.unprotected_left import FULL_PASS_RANGE_M

HEADER = (Path(__file__).parents[3] / 'common' / 'params_keys.h').read_text()
# TurnPilot attribute: (param, its params_keys.h default as written, the value main() gives the attribute from it)
SHIPPED = {'always_drive': ('TurnPilotAlwaysDrive', '1', True), 'no_move_s': ('TurnPilotNoMoveS', '20.0', 20.),
           'stop_sign_mode': ('TurnPilotStopSignGo', 'vision', 'vision'), 'auto_left': ('TurnPilotAutoLeft', 'auto', 'auto'),
           'left_range_m': ('TurnPilotLeftRangeM', '210.0', 210.), 'left_far_range_m': ('TurnPilotLeftFarRangeM', '0.0', 0.),
           'go_override': ('TurnPilotGoOverride', '1', True), 'camera_mode': ('TurnPilotCameraLights', 'addOnly', 'addOnly'),
           'curb_ped_cap': ('TurnPilotCurbPedCap', '1', True), 'yellow_decel': ('TurnPilotYellowDecel', '2.0', 2.0)}


def shipped(sim):
  """sim (test_turnpilotd.Sim) set as main() sets a fresh install's TurnPilot. No sim oracle (the car has none), the full
  pass's range capped where the car's is, no gas taps."""
  for attr, (key, text, value) in SHIPPED.items():
    m = re.search(r'\{"' + re.escape(key) + r'", \{[^}]*?, "([^"]*)"\}\}', HEADER)
    assert m is not None and m.group(1) == text, (key, m and m.group(1), text)
    setattr(sim.tp, attr, value)
  sim.tp.stop_sign_known = True
  sim.tp.oracle_ok = False
  sim.tp.full_pass_range_m = FULL_PASS_RANGE_M
  sim.go_taps = False
  return sim
