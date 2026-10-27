"""The lanes and the upcoming maneuvers for navInstruction, read from Mapbox-shaped route steps.

Mapbox and Valhalla (format=osrm) both put the lanes of a maneuver in two places: the sub banner of the step
before it, and the first intersection of the step that starts with it. Valhalla only adds the sub banner for
the last 400 m of the step, while the intersection has the lanes all along, so the banner comes first (it is
what Mapbox shows) and the intersection fills in the rest. A lane is active if it can be used for the maneuver,
and activeDirection is the direction it takes from there.
"""

MAX_MANEUVERS = 10  # allManeuvers is for the planner's lookahead; a long route would otherwise send every step each second


def to_direction(indication) -> str:
  """An OSRM lane indication as a NavInstruction.Direction. Sharp turns count as turns, and u-turns as none."""
  indication = indication or ''
  for direction in ('left', 'right', 'straight'):
    if direction in indication:
      return 'slight' + direction.capitalize() if 'slight' in indication and direction != 'straight' else direction
  return 'none'


def make_lane(indications, usable, active_direction) -> dict:
  return {'directions': list(dict.fromkeys(map(to_direction, indications or []))), 'active': bool(usable),
          'activeDirection': to_direction(active_direction)}


def banner_lanes(step, distance_to_maneuver) -> list[dict]:
  """The lanes in the sub banner navd would show for this step now. A lane is active if it can be used for the maneuver."""
  banners = step.get('bannerInstructions') or []
  if not banners:
    return []
  banner = next((b for b in reversed(banners) if distance_to_maneuver < b['distanceAlongGeometry']), banners[0])
  components = (banner.get('sub') or {}).get('components') or []
  return [make_lane(c.get('directions'), c.get('active') or c.get('active_direction'), c.get('active_direction'))
          for c in components if c.get('type') == 'lane']


def intersection_lanes(step) -> list[dict]:
  """The lanes of the intersection that starts the step, which are the approach lanes of its maneuver."""
  intersections = step.get('intersections') or []
  lanes = (intersections[0].get('lanes') if intersections else None) or []
  return [make_lane(lane.get('indications'), lane.get('valid') or lane.get('active'), lane.get('valid_indication'))
          for lane in lanes]


def approach_lanes(steps, step_idx, distance_to_maneuver) -> list[dict]:
  """The lanes on the approach to the maneuver that ends the current step, empty if the route has none there."""
  lanes = banner_lanes(steps[step_idx], distance_to_maneuver)
  if not lanes and step_idx + 1 < len(steps):
    lanes = intersection_lanes(steps[step_idx + 1])
  return lanes


def upcoming_maneuvers(steps, step_idx, distance_to_maneuver) -> list[dict]:
  """The next MAX_MANEUVERS maneuvers ahead, with their distance from the ego: the one that ends the current step first."""
  maneuvers = []
  distance = distance_to_maneuver
  for ahead in steps[step_idx + 1:step_idx + 1 + MAX_MANEUVERS]:
    maneuver = ahead.get('maneuver') or {}
    maneuvers.append({'distance': distance, 'type': maneuver.get('type') or '', 'modifier': maneuver.get('modifier') or ''})
    distance += ahead['distance']
  return maneuvers


def fill_guidance(instruction, steps, step_idx, distance_to_maneuver):
  """Fill NavInstruction.lanes and allManeuvers for the ego being distance_to_maneuver short of the end of the step."""
  instruction.lanes = approach_lanes(steps, step_idx, distance_to_maneuver)
  instruction.allManeuvers = upcoming_maneuvers(steps, step_idx, distance_to_maneuver)
