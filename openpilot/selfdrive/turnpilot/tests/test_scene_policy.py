from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.hooks import plan_accel
from openpilot.selfdrive.turnpilot.route_follower import NOMINAL_ANGLE_DEG, Maneuver, classify
from openpilot.selfdrive.turnpilot.scene_policy import HOLD_LINE_M, NO_ACCEL_CAP, NO_SPEED_CAP, PED_STOP_GAP_M, scene_policy
from openpilot.selfdrive.turnpilot.tests.synthetic import scene

LEFT_TURN = Maneuver(*classify("turn", "left"), "turn", "left", NOMINAL_ANGLE_DEG["left"])
RIGHT_TURN = Maneuver(*classify("turn", "right"), "turn", "right", NOMINAL_ANGLE_DEG["right"])


def ped(x, y, vx=0., vy=0., in_path=False):
  return {'cls': 'pedestrian', 'x': x, 'y': y, 'vx': vx, 'vy': vy, 'inPath': in_path}


def car(x, y, vx=0., vy=0.):
  return {'cls': 'car', 'x': x, 'y': y, 'vx': vx, 'vy': vy, 'inPath': False}


def policy(objects=(), available=True, maneuver=None, distance=1000., v_ego=10.):
  sc = scene(objects, available)
  return scene_policy(sc.objects, sc.available, maneuver, distance, v_ego)


def test_nothing_there():
  d = policy()
  assert (d.accel_cap, d.speed_cap, d.should_stop, d.holding, d.reason) == (NO_ACCEL_CAP, NO_SPEED_CAP, False, False, "")


def test_pedestrian_in_path_stops():
  d = policy([ped(30., 0.2, in_path=True)], v_ego=10.)
  assert d.holding and d.reason == "pedestrian in path" and d.speed_cap < 10.
  # too close for the comfortable curve: brake harder through accelCap
  d = policy([ped(14., 0.2, in_path=True)], v_ego=10.)
  assert d.accel_cap < -5.
  # at the stop point near standstill: hold
  d = policy([ped(PED_STOP_GAP_M + 0.5, 0.2, in_path=True)], v_ego=0.1)
  assert d.should_stop


def test_closed_loop_stop_short_of_pedestrian():
  x_ped, v, s, stopped = 60., 12., 0., False
  for _ in range(int(20 / DT_MDL)):
    d = policy([ped(x_ped - s, 0., in_path=True)], v_ego=v)
    a = max(min(plan_accel(d.accel_cap, d.speed_cap, v), 0.5), -3.5)  # planner min() with cruise, then ACCEL_MIN
    if d.should_stop:
      a = min(a, -1.)
      stopped = True
    v = max(v + a * DT_MDL, 0.)
    s += v * DT_MDL
  assert stopped and v == 0.
  assert 0. < x_ped - s < PED_STOP_GAP_M + 2.


def test_pedestrian_on_sidewalk_ignored_walking_in_counts():
  assert not policy([ped(20., -5.)]).holding
  d = policy([ped(20., -3., vy=2.5)])
  assert d.holding and d.reason == "pedestrian in path"


def test_crosswalk_hold_on_right_turn():
  d = policy([ped(23., -6.)], maneuver=RIGHT_TURN, distance=20., v_ego=5.)
  assert d.holding and d.reason == "pedestrian in crosswalk"
  assert d.speed_cap < 5.
  # same pedestrian, but the turn is a left: not on the turn side
  assert policy([ped(23., -6.)], maneuver=LEFT_TURN, distance=20., v_ego=5.).reason != "pedestrian in crosswalk"
  # committed: past the hold line and moving
  assert not policy([ped(8., -6.)], maneuver=RIGHT_TURN, distance=HOLD_LINE_M - 3., v_ego=4.).holding


def test_oncoming_car_holds_unprotected_left():
  d = policy([car(60., 5., vx=-12.)], maneuver=LEFT_TURN, distance=15., v_ego=4.)
  assert d.holding and d.reason == "waiting for oncoming"
  # a gap of more than 5 s, a receding car, a car in the cross street, or one already through: go
  assert not policy([car(60., 5., vx=-5.)], maneuver=LEFT_TURN, distance=15., v_ego=4.).holding
  assert not policy([car(60., 5., vx=8.)], maneuver=LEFT_TURN, distance=15., v_ego=4.).holding
  assert not policy([car(60., -5., vx=-12.)], maneuver=LEFT_TURN, distance=15., v_ego=4.).holding
  assert not policy([car(5., 5., vx=-12.)], maneuver=LEFT_TURN, distance=15., v_ego=4.).holding
  # far from the junction the hold isn't engaged yet
  assert not policy([car(160., 5., vx=-30.)], maneuver=LEFT_TURN, distance=120., v_ego=10.).holding
  # stopped at the hold line: stays stopped
  d = policy([car(60., 5., vx=-12.)], maneuver=LEFT_TURN, distance=HOLD_LINE_M + 0.5, v_ego=0.)
  assert d.should_stop
  # committed to the turn
  assert not policy([car(40., 5., vx=-12.)], maneuver=LEFT_TURN, distance=HOLD_LINE_M - 3., v_ego=4.).holding


def test_detector_unavailable():
  d = policy([car(60., 5., vx=-12.)], available=False, maneuver=LEFT_TURN, distance=15., v_ego=4.)
  assert d.holding and d.reason == "detector unavailable"
  d = policy(available=False, maneuver=RIGHT_TURN, distance=15., v_ego=4.)
  assert not d.holding and d.reason == "detector unavailable" and d.speed_cap == NO_SPEED_CAP
  d = policy(available=False)
  assert not d.holding and d.reason == "detector unavailable"
