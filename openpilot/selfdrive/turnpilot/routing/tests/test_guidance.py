import json
import unittest
from pathlib import Path

import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.turnpilot.route_follower import KEEP, NONE, RIGHT, TURN, maneuver_from_instruction
from openpilot.selfdrive.turnpilot.routing.guidance import (MAX_MANEUVERS, approach_lanes, banner_lanes, fill_guidance,
                                                            intersection_lanes, maneuver_junction, through_lanes, to_direction,
                                                            upcoming_maneuvers)

FIXTURES = Path(__file__).parent / 'fixtures'


def steps_of(name):
  """The steps of a response recorded from Valhalla (format=osrm, shape_format=geojson, banner_instructions=true)."""
  return json.loads((FIXTURES / name).read_text())['routes'][0]['legs'][0]['steps']


def lane(directions, active, active_direction='none'):
  return {'directions': directions, 'active': active, 'activeDirection': active_direction}


class TestGuidance(unittest.TestCase):
  def test_directions(self):
    expected = {'left': 'left', 'sharp left': 'left', 'slight left': 'slightLeft', 'straight': 'straight', 'slight right': 'slightRight',
                'sharp right': 'right', 'right': 'right', 'uturn': 'none', 'none': 'none', '': 'none', None: 'none'}
    for indication, direction in expected.items():
      self.assertEqual(to_direction(indication), direction, indication)

  def test_valhalla_left_turn_lanes(self):
    steps = steps_of('valhalla_left_then_right.json')
    # Valhalla only adds the sub banner for the last 400 m of the step, but the first intersection of the next step has the lanes all along.
    left_lane = [lane(['left'], True, 'left'), lane(['straight'], False), lane(['straight', 'right'], False)]
    self.assertEqual(banner_lanes(steps[0], 590), [])
    self.assertEqual(banner_lanes(steps[0], 300), left_lane)
    self.assertEqual(intersection_lanes(steps[1]), left_lane)
    for distance in (590, 300, 20, -4):
      self.assertEqual(approach_lanes(steps, 0, distance), left_lane)
    # A step without lane data on its approach has none, and nor has the last one.
    self.assertEqual(approach_lanes(steps, 1, 200), [])
    self.assertEqual(approach_lanes(steps, 3, 0), [])

  def test_valhalla_right_turn_lanes(self):
    steps = steps_of('valhalla_right_turn.json')
    self.assertEqual(approach_lanes(steps, 0, 500), [lane(['left'], False), lane(['straight'], False), lane(['straight', 'right'], True, 'right')])

  def test_mapbox_sub_banner(self):
    # Mapbox's banner marks the recommended lane active, and gives a valid lane only its active_direction.
    components = [{'type': 'lane', 'directions': ['left', 'sharp left'], 'active': False},
                  {'type': 'lane', 'directions': ['straight', 'slight left'], 'active': False, 'active_direction': 'slight left'},
                  {'type': 'lane', 'directions': ['straight'], 'active': True, 'active_direction': 'straight'},
                  {'type': 'icon', 'text': 'not a lane'}]
    steps = [{'distance': 900, 'bannerInstructions': [{'distanceAlongGeometry': 900, 'primary': {'text': 'Exit'}},
                                                      {'distanceAlongGeometry': 400, 'primary': {'text': 'Exit'}, 'sub': {'components': components}}]},
             {'distance': 100, 'bannerInstructions': []}]
    self.assertEqual(approach_lanes(steps, 0, 700), [])
    self.assertEqual(approach_lanes(steps, 0, 350), [lane(['left'], False), lane(['straight', 'slightLeft'], True, 'slightLeft'),
                                                     lane(['straight'], True, 'straight')])

  def test_mapbox_style_intersection_lanes(self):
    steps = [{'distance': 100, 'bannerInstructions': []},
             {'distance': 100, 'bannerInstructions': [], 'intersections': [
               {'lanes': [{'valid': False, 'indications': ['left']}, {'valid': True, 'indications': ['straight', 'right'], 'valid_indication': 'right'}]},
               {'lanes': [{'valid': True, 'indications': ['straight']}]}]}]
    self.assertEqual(approach_lanes(steps, 0, 50), [lane(['left'], False), lane(['straight', 'right'], True, 'right')])  # the first intersection only

  def test_upcoming_maneuvers(self):
    steps = steps_of('valhalla_left_then_right.json')
    maneuvers = upcoming_maneuvers(steps, 0, 500)
    self.assertEqual([(m['type'], m['modifier']) for m in maneuvers], [('turn', 'left'), ('new name', 'right'), ('arrive', '')])
    # The first is where the current step ends, and each one after it is a step further.
    self.assertEqual([m['distance'] for m in maneuvers], [500, 500 + 445.0, 500 + 445.0 + 257.0])
    self.assertEqual([m['type'] for m in upcoming_maneuvers(steps, 2, 30)], ['arrive'])
    self.assertEqual(upcoming_maneuvers(steps, 3, 0), [])

  def test_maneuvers_are_capped(self):
    steps = [{'distance': 100, 'maneuver': {'type': 'turn', 'modifier': 'left'}} for _ in range(40)]
    maneuvers = upcoming_maneuvers(steps, 5, 30)
    self.assertEqual(len(maneuvers), MAX_MANEUVERS)
    self.assertEqual([m['distance'] for m in maneuvers], [30 + 100 * i for i in range(MAX_MANEUVERS)])  # the nearest ones
    self.assertEqual(len(upcoming_maneuvers(steps, 35, 30)), 4)
    instruction = messaging.new_message('navInstruction').navInstruction
    fill_guidance(instruction, steps, 0, 30)
    self.assertEqual(len(instruction.allManeuvers), MAX_MANEUVERS)

  def test_fill_navinstruction(self):
    steps = steps_of('valhalla_left_then_right.json')
    instruction = messaging.new_message('navInstruction').navInstruction
    fill_guidance(instruction, steps, 0, 300)
    lanes = instruction.to_dict()['lanes']
    self.assertEqual(lanes[0], {'directions': ['left'], 'active': True, 'activeDirection': 'left'})
    self.assertEqual(lanes[2], {'directions': ['straight', 'right'], 'active': False, 'activeDirection': 'none'})
    maneuvers = instruction.to_dict()['allManeuvers']
    self.assertEqual([(m['type'], m.get('modifier', '')) for m in maneuvers], [('turn', 'left'), ('new name', 'right'), ('arrive', '')])
    self.assertAlmostEqual(maneuvers[1]['distance'], 745, places=1)
    fill_guidance(instruction, steps, 3, 0)  # on the last step there is nothing ahead
    self.assertEqual(len(instruction.lanes), 0)
    self.assertEqual(len(instruction.allManeuvers), 0)


  def test_junction_or_bend(self):
    # recorded from Valhalla: a left at a four-way junction, then the road bending right where no other road meets it
    steps = steps_of('valhalla_left_then_right.json')
    self.assertEqual(maneuver_junction(steps, 0), {'before': 90., 'after': 0., 'branches': 2})
    self.assertEqual(maneuver_junction(steps, 1), {'before': 0., 'after': 90., 'branches': 0})
    self.assertIsNone(maneuver_junction(steps, 2))  # the arrival has no exit
    self.assertIsNone(maneuver_junction(steps, 3))  # nothing after the last step
    instruction = messaging.new_message('navInstruction').navInstruction
    fill_guidance(instruction, steps, 0, 300)
    self.assertTrue(instruction.maneuverGeometryValid)
    self.assertEqual((instruction.maneuverBearingBefore, instruction.maneuverBearingAfter, instruction.maneuverBranches), (90., 0., 2))
    instruction.maneuverType, instruction.maneuverModifier = 'turn', 'left'
    turn = maneuver_from_instruction(instruction)
    self.assertEqual((turn.kind, turn.angle_deg, turn.bend), (TURN, 90., False))
    fill_guidance(instruction, steps, 1, 300)
    instruction.maneuverType, instruction.maneuverModifier = 'new name', 'right'
    bend = maneuver_from_instruction(instruction)
    self.assertEqual((bend.kind, bend.angle_deg, bend.bend), (NONE, 90., True))  # the model's, not a turn
    fill_guidance(instruction, steps, 2, 0)
    self.assertFalse(instruction.maneuverGeometryValid)

  def test_one_way_road_in_counts_as_a_branch(self):
    # a T where the other road is one-way toward the junction (entry false): still a road the model could take
    step = {'maneuver': {'type': 'end of road', 'modifier': 'left', 'bearing_before': 114, 'bearing_after': 24},
            'intersections': [{'bearings': [24, 203, 294], 'entry': [True, False, False], 'in': 2, 'out': 0}]}
    self.assertEqual(maneuver_junction([{}, step], 0)['branches'], 1)
    bend = {'maneuver': {'type': 'turn', 'modifier': 'right', 'bearing_before': 20, 'bearing_after': 71},
            'intersections': [{'bearings': [71, 200], 'entry': [True, False], 'in': 1, 'out': 0}]}
    self.assertEqual(maneuver_junction([{}, bend], 0)['branches'], 0)
    self.assertIsNone(maneuver_junction([{}, {'maneuver': {'type': 'turn'}, 'intersections': []}], 0))

  def test_real_angle_and_side(self):
    instruction = messaging.new_message('navInstruction').navInstruction
    instruction.maneuverGeometryValid, instruction.maneuverBranches = True, 1
    for mtype, modifier, before, after, kind, angle in (('turn', 'left', 94., 42., TURN, 52.), ('turn', 'right', 350., 80., TURN, 90.),
                                                        ('turn', 'slight right', 92., 110., KEEP, 18.),
                                                        ('turn', 'sharp right', 10., 160., TURN, 150.)):
      instruction.maneuverType, instruction.maneuverModifier = mtype, modifier
      instruction.maneuverBearingBefore, instruction.maneuverBearingAfter = before, after
      m = maneuver_from_instruction(instruction)
      self.assertEqual(m.kind, kind)
      self.assertAlmostEqual(m.angle_deg, angle)
    self.assertEqual(m.bearing_side, RIGHT)

  def test_through_lanes(self):
    # a straight-through intersection with a left-turn-only lane, 200 m into a 500 m step
    step = {'distance': 500., 'geometry': {'coordinates': [[0., 0.], [0.0045, 0.]]},
            'intersections': [{'location': [0., 0.]},
                              {'location': [0.0018, 0.], 'lanes': [{'indications': ['left'], 'valid': False},
                                                                   {'indications': ['straight'], 'valid': True}]},
                              {'location': [0.003, 0.], 'lanes': [{'indications': ['straight'], 'valid': True}]}]}
    lanes, distance = through_lanes([step], 0, 450.)  # 50 m into the step
    self.assertEqual(lanes, [lane(['left'], False), lane(['straight'], True)])
    self.assertAlmostEqual(distance, 150., delta=1.)
    self.assertEqual(through_lanes([step], 0, 250.), ([], 0.))  # past it; the next has a single lane


if __name__ == '__main__':
  unittest.main()
