"""Routes for routed drives on west_coast_usa (route_drive.py).

Picked with Valhalla on the road-graph tiles (TurnPilotValhallaTiles, west_coast_usa) on 2026-10-27. A route is a
start pose and one or more legs: the destination of each leg is set, the way the phone app does it, once the one
before is reached. One leg from a start back near itself is never longer than a few turns on this map, so the loop
goes out to a far destination and back. Maneuvers count every turn and keep (a fork, a slight turn) on the way, as
Valhalla gave them from the start pose; the export has many forks with slight modifiers and few plain turns.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
  name: str
  start: tuple[float, float, float]  # BeamNG world m (x east, y north, z up), in the lane
  heading: float                     # rad counterclockwise from east
  legs: tuple[tuple[float, float], ...]  # (latitude, longitude) of each leg's destination
  maneuvers: str                     # what Valhalla gave, for the reader


# 150 m before the signalized junction group_62_63_64_65, heading south-west (bearing 224)
G62_START = ((-550.16, 586.97, 90.49), -2.3374)

ROUTES = {r.name: r for r in (
  Route("short", *G62_START, ((39.770761, -86.1626246),),
        "4 turns, about 750 m: left, left, right, right"),
  Route("short2", (-629.47, 459.21, 116.2), -0.7875, ((39.770761, -86.1626246),),
        "3 turns, about 450 m: left, right, right (short without its first junction, group_62, which the model won't turn at)"),
  Route("loop", *G62_START, ((39.7661729, -86.1647217), (39.7737206, -86.1659774)),
        "18 maneuvers, about 5.6 km: out (left, left, right, right, keep left, keep left, keep right, 1.5 km) and back " +
        "(keep right, keep left, left, keep left, left, keep right, keep left, right, keep left, right, left, 4.0 km), " +
        "ending about 110 m from the start"),
)}
