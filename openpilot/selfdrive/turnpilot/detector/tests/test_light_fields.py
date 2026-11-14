"""A head's nearer range and the reader's id on fields of their own (TurnpilotLights.Light nearDistance, headIdPlusOne),
beside the instance text map_controls parses today ('head<id> near<m>'), which stays as it was so the parser keeps working;
both fields read 0 (none) in logs from before them."""
import os
from types import SimpleNamespace

import pytest

from openpilot.cereal import messaging
from openpilot.selfdrive.turnpilot.detector.light_state import LightHead
from openpilot.selfdrive.turnpilot.detector.modeld_detector import fill_heads, fill_lights
from openpilot.selfdrive.turnpilot.map_controls import CAMERA_ASSOC_PAST_M, CAMERA_ROUTE, camera_light_input, camera_owner, head_id, head_near


def test_the_nearer_range_and_the_id_have_fields_and_the_instance_is_as_before():
  td = messaging.new_message('turnpilotDetections').turnpilotDetections
  fill_lights(td.lights, [LightHead(6, (0.4, 0.1, 0.41, 0.15), 0.6, 'red', 3, 'red', 'read', 1., 70.7, True, 'h43 fr0.47/0.25/0.00', 29.36),
                          LightHead(0, (0.5, 0.1, 0.51, 0.15), 0.6, 'green', 3, 'green', 'read', 2., 40., True)], False)
  fill_heads(td.init('lightRejects', 1), [LightHead(-1, (0.6, 0.1, 0.61, 0.16), 0.13, 'unknown', 0, 'red', 'lowConf', -2., 60., False,
                                                    'h44', 31.2)])
  lights = td.as_reader().lights.lights
  assert [x.instance for x in lights] == ['head6 near29.4 h43 fr0.47/0.25/0.00', 'head0']
  assert [x.nearDistance for x in lights] == [pytest.approx(29.36), 0.] and [x.headIdPlusOne for x in lights] == [7, 1]
  (rej,) = td.as_reader().lightRejects
  assert rej.nearDistance == pytest.approx(31.2) and rej.headIdPlusOne == 0 and rej.instance == 'head-1 near31.2 h44'
  # map_controls reads the fields (the exact nearer range, not the instance's 0.1 m), the instance where they are 0
  cam = camera_light_input(td.as_reader().lights, 'read', 0., 'addOnly', td.as_reader().lightRejects)
  assert cam.ids == [6, 0] and cam.nears == [pytest.approx(29.36), None] and cam.vetoes == [('red', 60., pytest.approx(31.2), -2.)]


def test_a_light_from_before_the_fields_reads_none():
  light = messaging.new_message('turnpilotDetections').turnpilotDetections.lights.init('lights', 1)[0]
  light.instance = 'head3 near12.0'
  assert light.nearDistance == 0. and light.headIdPlusOne == 0


OLD_LOG = os.environ.get('TP_OLD_LIGHT_RLOG', '')  # an rlog.zst of a drive from before the fields


@pytest.mark.skipif(not os.path.exists(OLD_LOG), reason='TP_OLD_LIGHT_RLOG names no log on this host')
def test_a_drive_from_before_the_fields_reads_with_the_new_schema():
  from openpilot.tools.lib.logreader import LogReader
  path = OLD_LOG
  n = 0
  for e in LogReader(path, only_union_types=True):
    if e.which() == 'turnpilotDetections':
      for light in e.turnpilotDetections.lights.lights:
        n += 1
        assert light.nearDistance == 0. and light.headIdPlusOne == 0 and light.instance.startswith('head')
  assert n > 0


HEADS = [LightHead(6, (0.4, 0.1, 0.41, 0.15), 0.6, 'red', 3, 'red', 'read', 1., 70.7, True, 'h43', 29.4),
         LightHead(0, (0.5, 0.1, 0.51, 0.15), 0.6, 'green', 3, 'green', 'read', 2., 40., True),
         LightHead(11, (0.6, 0.1, 0.61, 0.15), 0.6, 'yellow', 3, 'yellow', 'read', -3., 125., True, '', 47.1)]
REJECTS = [LightHead(-1, (0.6, 0.1, 0.61, 0.16), 0.13, 'unknown', 0, 'red', 'lowConf', -2., 60., False, 'h44', 31.2)]


def encoded(fields: bool):
  """turnpilotDetections of HEADS and REJECTS as modeld fills them today (fields), or as before the fields (both 0: the
  instance text alone, as in the drive logs and the sim's lights)."""
  td = messaging.new_message('turnpilotDetections').turnpilotDetections
  fill_lights(td.lights, HEADS, False)
  fill_heads(td.init('lightRejects', len(REJECTS)), REJECTS)
  if not fields:
    for light in list(td.lights.lights) + list(td.lightRejects):
      light.nearDistance, light.headIdPlusOne = 0., 0
  r = td.as_reader()
  return r.lights, r.lightRejects


def test_map_controls_reads_the_same_heads_from_the_fields_and_from_the_instance():
  new, old = encoded(True), encoded(False)
  assert [x.headIdPlusOne for x in new[0].lights] == [7, 1, 12] and [x.headIdPlusOne for x in old[0].lights] == [0, 0, 0]
  assert [head_id(x) for x in new[0].lights] == [head_id(x) for x in old[0].lights] == [6, 0, 11]
  assert [head_near(x) for x in new[0].lights] == pytest.approx([head_near(x) for x in old[0].lights]) and \
    head_near(new[0].lights[1]) is None and head_near(old[0].lights[1]) is None
  a, b = (camera_light_input(m[0], 'read', 0., 'addOnly', m[1]) for m in (new, old))
  assert a.heads == b.heads and a.ids == b.ids == [6, 0, 11] and a.dark == b.dark and a.bearings == b.bearings
  assert [n is None for n in a.nears] == [n is None for n in b.nears] and \
    [x for x in a.nears if x is not None] == pytest.approx([x for x in b.nears if x is not None])
  assert [v[:2] + v[3:] for v in a.vetoes] == [v[:2] + v[3:] for v in b.vetoes] and \
    [v[2] for v in a.vetoes] == pytest.approx([v[2] for v in b.vetoes])
  # the decision that reads the nearer range: which signal line takes each head (a red ranged long past every window is
  # the first line its nearer range reaches), the same from either encoding
  lines = [SimpleNamespace(d=d, key=(k, 1)) for k, d in ((CAMERA_ROUTE, 18.), (4, 75.), (5, 140.))]
  for cam in (a, b):
    owners = [camera_owner(h[1], lines, h[0], cam.nears[i]) for i, h in enumerate(cam.heads)]
    assert [o.d if o is not None else None for o in owners] == [75., 18., 18.]


def test_a_head_whose_nearer_range_differs_in_the_fields_by_the_instances_rounding_only():
  # the field is exact where the instance rounds to 0.1 m: the only difference, a few cm of a head's span, which decides
  # only where the span ends within those cm of a line's window (here its far edge, CAMERA_ASSOC_PAST_M past the line)
  edge = 18. + CAMERA_ASSOC_PAST_M
  td = messaging.new_message('turnpilotDetections').turnpilotDetections
  fill_lights(td.lights, [LightHead(2, (0.4, 0.1, 0.41, 0.15), 0.6, 'red', 3, 'red', 'read', 0., 120., True, '', edge + 0.04)], False)
  light = td.as_reader().lights.lights[0]
  assert light.instance.startswith(f'head2 near{edge:.1f}') and head_near(light) == pytest.approx(edge + 0.04)
  line = [SimpleNamespace(d=18., key=(CAMERA_ROUTE, 1))]
  assert camera_owner(120., line, 'red', edge) is line[0] and camera_owner(120., line, 'red', head_near(light)) is None
