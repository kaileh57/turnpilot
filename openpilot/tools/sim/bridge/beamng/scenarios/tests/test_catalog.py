import math
import os

import numpy as np
import pytest

from openpilot.selfdrive.turnpilot import scene_policy
from openpilot.selfdrive.turnpilot.route_follower import RIGHT
from openpilot.tools.sim.bridge.beamng.roadgraph import GRAPH_JSON, RoadGraph, wrap
from openpilot.tools.sim.bridge.beamng.scenarios import catalog
from openpilot.tools.sim.bridge.beamng.scenarios.timing import arc_lengths

pytestmark = pytest.mark.skipif(not os.path.exists(GRAPH_JSON), reason="needs piston's road-graph export")
CROSSWALK_JUNCTIONS = [(-768.6, 555.2), (-759.2, 376.7), (-650.1, 481.7), (-600.7, 220.9), (-445.3, 147.0), (-32.5, 403.8)]


@pytest.fixture(scope="module")
def graph():
  return RoadGraph()


def car_frame(p, ego, heading):
  """(x forward, y left) of world point p for a car at ego facing heading."""
  d = np.asarray(p[:2]) - np.asarray(ego[:2])
  return float(d[0] * math.cos(heading) + d[1] * math.sin(heading)), float(-d[0] * math.sin(heading) + d[1] * math.cos(heading))


def test_every_scenario_builds(graph):
  for name in catalog.SCENARIOS:
    plan = catalog.build(name, graph)
    assert plan.dest is not None, name
    near = min(math.hypot(*(graph.pos[j][:2] - np.asarray(plan.spawn[:2]))) for j in graph.junctions())
    assert near > 20.0, f"{name} spawns {near:.0f} m from a junction"
    if plan.junction is not None:  # the car starts on its approach, facing the junction
      bearing = math.atan2(plan.junction[1] - plan.spawn[1], plan.junction[0] - plan.spawn[0])
      assert abs(math.degrees(wrap(bearing - plan.heading))) < 30.0, name


def test_crosswalk_junctions(graph):
  for xy in CROSSWALK_JUNCTIONS[:4]:
    for direction in ('left', 'right'):
      jn = catalog.find_turn(graph, xy, direction)
      assert np.hypot(*(jn.center[:2] - np.asarray(xy))) < 5.0
      assert jn.straight is not None
  for xy in CROSSWALK_JUNCTIONS[4:]:  # no 90-degree turn in the road graph there
    with pytest.raises(ValueError):
      catalog.find_turn(graph, xy, 'right')


def test_walker_crosses_the_policy_crosswalk_zone(graph):
  plan = catalog.build('ped_crosswalk_right', graph)
  ped = plan.actors[0]
  path = np.array(ped.path)
  # the car waiting at the hold line, in its lane, looking down the approach
  ego = np.asarray(plan.junction) - (scene_policy.HOLD_LINE_M + 4.0) * np.array([math.cos(plan.heading), math.sin(plan.heading)])
  ego += 1.8 * np.array([math.sin(plan.heading), -math.cos(plan.heading)])
  dist = math.hypot(*(np.asarray(plan.junction) - ego))

  def in_zone(p):
    x, y = car_frame(p, ego, plan.heading)
    return dist - scene_policy.CROSSWALK_X_BEFORE_M < x < dist + scene_policy.CROSSWALK_X_AFTER_M and \
      scene_policy.CROSSWALK_Y_NEAR_M < y * RIGHT < scene_policy.CROSSWALK_Y_FAR_M

  walk = [path[0] + f * (path[1] - path[0]) for f in np.linspace(0.15, 0.85, 8)]
  assert all(in_zone(p) for p in walk), "the crossing is outside scene_policy's crosswalk zone"
  assert not in_zone(path[-1]), "the walker never leaves the zone, so the car would hold forever"
  assert car_frame(path[0], ego, plan.heading)[0] < car_frame(path[1], ego, plan.heading)[0]  # starts on the near side


def test_oncoming_cars(graph):
  plan = catalog.build('unprotected_left', graph)
  cars = [a for a in plan.actors if a.trigger == 'oncoming']
  assert len(cars) == 2
  ego_lane = car_frame(plan.spawn, plan.junction, plan.heading)[1]  # left of the junction centre, along the car's heading
  for k, car in enumerate(cars):
    path = np.array(car.path)
    start = math.hypot(*(path[0, :2] - np.asarray(plan.junction)))
    assert start == pytest.approx(60.0 + 25.0 * k, abs=3.0)
    assert abs(math.degrees(wrap(car.heading - plan.heading - math.pi))) < 25.0  # coming the other way
    lane = car_frame(path[0], plan.junction, plan.heading)[1]
    assert lane * ego_lane < 0 and abs(lane) > 1.0  # on the other side of the road
    assert arc_lengths(path)[-1] > start + 40.0  # through the junction and away


def test_standing_pedestrian_in_the_lane(graph):
  plan = catalog.build('ped_in_path', graph)
  ped = np.array(plan.actors[0].path[0])
  x, y = car_frame(ped, plan.spawn, plan.heading)
  assert 60.0 < x < 120.0 and abs(y) < 0.5


@pytest.mark.parametrize("entry", ['wpTown_5_left', 'autojunction_1311_left'])
def test_pacer_lane(graph, entry):
  plan = catalog.build('occupied_lane', graph, entry=entry)
  pacer = plan.actors[0]
  lane = np.array(pacer.path)
  s, off, h = catalog.project(lane, np.asarray(plan.spawn))
  assert off == pytest.approx(3.6, abs=0.1)  # the car is one lane right of the pacer: the lane a left turn changes into
  assert abs(math.degrees(wrap(h - plan.heading))) < 5.0
  end_to_junction = math.hypot(*(lane[-1, :2] - np.asarray(plan.junction)))
  assert end_to_junction == pytest.approx(catalog.PACER_END_BEFORE_M, abs=15.0)
  assert pacer.params['s_junction'] == pytest.approx(arc_lengths(lane)[-1] + catalog.PACER_END_BEFORE_M, abs=1.0)


def test_unprotected_lefts_give_the_car_3_to_4_s_before_the_oncoming_cars():
  # the car stops at its hold line and waits ~2 s for the driver's go: oncoming cars timed 1.5 s after it reaches the line
  # have passed before it may go, so the detector's oncoming hold is never exercised
  lefts = {n: e for n, e in catalog.SCENARIOS.items() if e.actor == 'oncoming'}
  assert set(lefts) >= {'unprotected_left', 'unprotected_left_visible'}
  for n, e in lefts.items():
    assert 3.0 <= e.arrive_after_s <= 4.0, n
