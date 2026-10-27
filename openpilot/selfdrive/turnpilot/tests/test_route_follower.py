from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.route_follower import ARRIVE, KEEP, LEFT, NONE, RIGHT, TURN, UTURN, RouteFollower, classify
from openpilot.selfdrive.turnpilot.tests.synthetic import FakeNavd, nav_instruction


def test_classify():
  assert classify("turn", "left") == (TURN, LEFT)
  assert classify("end of road", "sharp right") == (TURN, RIGHT)
  assert classify("on ramp", "right") == (TURN, RIGHT)
  assert classify("turn", "slight right") == (KEEP, RIGHT)
  # the road bending: no keep desire, which at highway speed swerves the car
  assert classify("continue", "slight right") == (NONE, 0)
  assert classify("new name", "slight left") == (NONE, 0)
  assert classify("continue", "left") == (TURN, LEFT)
  assert classify("off ramp", "right") == (KEEP, RIGHT)
  assert classify("fork", "left") == (KEEP, LEFT)
  assert classify("turn", "uturn") == (UTURN, 0)
  assert classify("turn", "straight") == (NONE, 0)
  assert classify("roundabout", "left") == (NONE, 0)
  assert classify("arrive", "right") == (ARRIVE, RIGHT)


def test_dead_reckoning_between_updates():
  rf = RouteFollower()
  rf.update(nav_instruction(200.), True, True, 10., DT_MDL)
  assert rf.valid and rf.distance == 200.
  for _ in range(10):
    rf.update(None, False, True, 10., DT_MDL)
  assert abs(rf.distance - 195.) < 1e-6
  # the next message wins over the dead-reckoned estimate
  rf.update(nav_instruction(193.), True, True, 10., DT_MDL)
  assert rf.distance == 193.
  assert rf.maneuver.kind == TURN and rf.maneuver.side == LEFT


def test_passed_when_navd_moves_on():
  navd = FakeNavd([(100., "turn", "left"), (400., "turn", "right")])
  rf = RouteFollower()
  s, v, passed_at, ids = 0., 10., [], set()
  for _ in range(int(30 / DT_MDL)):
    instr = navd.update(s)
    rf.update(instr, instr is not None, True, v, DT_MDL)
    ids.add(rf.maneuver_id)
    if rf.passed:
      passed_at.append(s)
    s += v * DT_MDL
  assert len(passed_at) == 1 and 100. < passed_at[0] < 125.
  assert len(ids) == 2
  assert rf.maneuver.side == RIGHT


def test_gps_jump_is_not_a_new_maneuver():
  rf = RouteFollower()
  rf.update(nav_instruction(100.), True, True, 10., DT_MDL)
  first = rf.maneuver_id
  rf.update(nav_instruction(115.), True, True, 10., DT_MDL)
  assert rf.maneuver_id == first and not rf.passed and rf.distance == 115.


def test_passed_by_dead_reckoning_when_navd_stalls():
  rf = RouteFollower()
  rf.update(nav_instruction(5.), True, True, 10., DT_MDL)
  passed = 0
  for _ in range(int(2.5 / DT_MDL)):
    rf.update(None, False, True, 10., DT_MDL)
    passed += rf.passed
  assert passed == 1 and rf.upcoming is None
  # navd catching up with the next step doesn't count the same maneuver twice
  rf.update(nav_instruction(300., modifier="right"), True, True, 10., DT_MDL)
  assert not rf.passed and rf.upcoming is not None


def test_route_lost():
  rf = RouteFollower()
  rf.update(nav_instruction(100.), True, True, 10., DT_MDL)
  rf.update(nav_instruction(100., valid=False), True, False, 10., DT_MDL)
  assert not rf.valid
  rf.update(nav_instruction(100.), True, True, 10., DT_MDL)
  for _ in range(int(4 / DT_MDL)):
    rf.update(None, False, True, 0., DT_MDL)
  assert not rf.valid


def test_arrived():
  rf = RouteFollower()
  rf.update(nav_instruction(40., "arrive", "right"), True, True, 5., DT_MDL)
  assert not rf.arrived
  rf.update(nav_instruction(10., "arrive", "right"), True, True, 5., DT_MDL)
  assert rf.arrived
