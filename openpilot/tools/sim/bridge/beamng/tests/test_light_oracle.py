"""The sim's light oracle (light_oracle.py): which lights are ahead, and what they say."""
from openpilot.tools.sim.bridge.beamng.light_oracle import MAX_LIGHTS, light_state, lights_ahead, make_msg, stop_lines


def test_states():
  assert [light_state(s) for s in ('greenTrafficLight', 'redTrafficLight', 'yellowTrafficLight', 'amber', None, 'off')] == \
    ['green', 'red', 'yellow', 'yellow', 'unknown', 'unknown']


def test_ahead_facing_nearest_first():
  lines = {'a': (0., 50.), 'b': (0., 120.), 'behind': (0., -10.), 'far': (0., 260.), 'side': (40., 60.), 'away': (0., 80.)}
  dirs = dict.fromkeys(lines, (0., 1.)) | {'away': (0., -1.)}
  assert lights_ahead((0., 0.), (0., 1.), lines, dirs) == [('a', 50.), ('b', 120.)]
  assert lights_ahead((0., 0.), (0., 1.), lines, {}) == []  # no direction from the game: not known to face the car
  many = {f'l{i}': (0., 10. * i + 5.) for i in range(10)}
  assert len(lights_ahead((0., 0.), (0., 2.), many, dict.fromkeys(many, (0., 1.)))) == MAX_LIGHTS


def test_message_and_the_committed_stop_lines():
  m = make_msg([('trafficLight 1', 42., 'red')]).turnpilotLights
  assert str(m.source) == 'simOracle' and (str(m.lights[0].state), m.lights[0].distance) == ('red', 42.)
  assert len(stop_lines()) > 100
