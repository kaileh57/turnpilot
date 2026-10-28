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


def test_new_route_with_the_same_maneuver_is_a_new_maneuver():
  # a destination set from the phone 30 m before a left, whose route starts with a left 284 m on
  rf = RouteFollower()
  rf.update(nav_instruction(30.), True, True, 7., DT_MDL)
  first = rf.maneuver_id
  rf.update(nav_instruction(284.), True, True, 7., DT_MDL)
  assert rf.maneuver_id != first and rf.replaced and not rf.passed
  rf.update(None, False, True, 7., DT_MDL)
  assert not rf.replaced  # for the one update
  # navd marking the new route: one invalid instruction between the two
  rf = RouteFollower()
  rf.update(nav_instruction(30.), True, True, 7., DT_MDL)
  first = rf.maneuver_id
  rf.update(nav_instruction(30., valid=False), True, False, 7., DT_MDL)
  assert rf.maneuver_id != first and rf.replaced and rf.upcoming is None
  rf.update(nav_instruction(35.), True, True, 7., DT_MDL)
  assert rf.upcoming is not None and not rf.replaced and not rf.passed


def test_moving_on_from_a_sharp_turn_is_not_a_new_route():
  # navd moves on to the next step's geometry up to 10 m before a sharp turn's end: the left was passed
  rf = RouteFollower()
  rf.update(nav_instruction(8.), True, True, 4., DT_MDL)
  first = rf.maneuver_id
  rf.update(nav_instruction(300.), True, True, 4., DT_MDL)
  assert rf.maneuver_id != first and rf.passed and not rf.replaced


def test_committed_turn_is_held_through_a_gap_in_the_route():
  # the turn desire is out 14 m before a left when one navd update has a fix 20 m off the step (valid False, the same
  # route): the maneuver stays, dead-reckoned, under the same id, and navd's next update carries on with it
  rf = RouteFollower()
  rf.update(nav_instruction(14., route_count=1), True, True, 6., DT_MDL, committed=True)
  first = rf.maneuver_id
  rf.update(nav_instruction(12., valid=False, route_count=1), True, False, 6., DT_MDL, committed=True)
  assert rf.maneuver_id == first and not rf.replaced and rf.upcoming is not None and abs(rf.distance - 13.7) < 1e-4
  for _ in range(19):
    rf.update(None, False, True, 6., DT_MDL, committed=True)
  rf.update(nav_instruction(8., route_count=1), True, True, 6., DT_MDL, committed=True)
  assert rf.maneuver_id == first and not rf.replaced and not rf.passed and rf.distance == 8.
  # not committed, the maneuver is dropped as before
  rf = RouteFollower()
  rf.update(nav_instruction(60., route_count=1), True, True, 6., DT_MDL)
  rf.update(nav_instruction(60., valid=False, route_count=1), True, False, 6., DT_MDL)
  assert rf.upcoming is None and not rf.valid


def test_committed_turn_is_held_while_navd_is_quiet_or_rerouting():
  # navd quiet (or rerouting, invalid) for longer than NAV_STALE_S with the turn desire out: the turn is dead-reckoned
  # to its pass, and the route is dropped once the turn is over
  for gap in ('quiet', 'invalid'):
    rf = RouteFollower()
    rf.update(nav_instruction(25., route_count=1), True, True, 5., DT_MDL, committed=True)
    first, passed = rf.maneuver_id, 0
    for i in range(int(10. / DT_MDL)):
      instr = nav_instruction(0., valid=False, route_count=1) if gap == 'invalid' and i % 20 == 0 else None
      rf.update(instr, instr is not None, False, 5., DT_MDL, committed=not passed)
      passed += rf.passed
      if not passed:
        assert rf.maneuver_id == first and rf.upcoming is not None and not rf.replaced, gap
    assert passed == 1, gap
    assert not rf.valid and rf.maneuver_id != first, gap  # lost once nothing holds it


def test_committed_turn_survives_a_distance_jump_and_a_reroute_onto_the_same_turn():
  # a 60 m jump in navd's distance from 30 m out (a GPS jump back along the road) isn't a new route for a committed turn
  rf = RouteFollower()
  rf.update(nav_instruction(30., route_count=1), True, True, 5., DT_MDL, committed=True)
  first = rf.maneuver_id
  rf.update(nav_instruction(90., route_count=1), True, True, 5., DT_MDL, committed=True)
  assert rf.maneuver_id == first and not rf.replaced and abs(rf.distance - 29.75) < 1e-4
  # a reroute (a new routeCount) that starts with the same turn where the car is: the same maneuver
  rf.update(nav_instruction(27., route_count=2), True, True, 5., DT_MDL, committed=True)
  assert rf.maneuver_id == first and not rf.replaced and rf.distance == 27.
  # a new route that doesn't turn there: replaced, and its turn is a new maneuver
  rf.update(nav_instruction(284., route_count=3), True, True, 5., DT_MDL, committed=True)
  assert rf.maneuver_id != first and rf.replaced and not rf.passed
