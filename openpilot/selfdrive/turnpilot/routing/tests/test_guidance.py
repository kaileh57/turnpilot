import json
import unittest
from pathlib import Path

import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.turnpilot.routing.guidance import (approach_lanes, banner_lanes, fill_guidance, intersection_lanes,
                                                            to_direction, upcoming_maneuvers)

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


if __name__ == '__main__':
  unittest.main()
