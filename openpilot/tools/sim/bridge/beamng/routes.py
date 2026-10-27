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
  Route("town18", *G62_START, ((39.7730509, -86.1600557), (39.7713386, -86.1617554), (39.7699151, -86.1657233),
                               (39.7739997, -86.16245)),
        "18 maneuvers, about 2.9 km in four legs, all in the town west of the highway: left, left, right, left, " +
        "slight right; left, left, right, left, right; keep right, right, left; right, right, left, right, keep right. " +
        "Ends about 180 m from the start"),
  Route("town15", *G62_START, ((39.7707207, -86.1626724), (39.7699151, -86.1657233), (39.7723673, -86.167217),
                               (39.773957, -86.1651128)),
        "15 maneuvers, about 2.2 km in four legs in the town: left, left, right, right; keep right, right, left; " +
        "right, right, slight left, left; right, right, left, right. Ends about 60 m from the start"),
  Route("town14", (-629.47, 459.21, 116.2), -0.7875, ((39.7707207, -86.1626724), (39.7699151, -86.1657233),
                                                     (39.7723673, -86.167217), (39.773957, -86.1651128)),
        "town15 from just past its first junction (g62), whose left the model takes in some runs and not in others"),
  Route("town15b", (-629.47, 459.21, 116.2), -0.7875, ((39.7694848, -86.1648298), (39.7680603, -86.1612328),
                                                      (39.7693629, -86.1603638), (39.7705062, -86.1649586)),
        "15 maneuvers, about 2.7 km in four legs from just past g62, with every leg ending on a two-way street (a leg " +
        "that ended on a divided road had Valhalla start the next one on the other carriageway) and no junction that " +
        "failed in screening: right, left, right; right, right, right, left, right, slight left; left, left, right; " +
        "left, right, right. Ends about 230 m from the start"),
  Route("town15b2", (-629.47, 459.21, 116.2), -0.7875, ((39.7694848, -86.1648298), (39.7698616, -86.1639357)),
        "town15b's first leg and its second up to 39 m past the left at (-539, 157), short of the fork at DR272_7: " +
        "7 turns, about 1.1 km: right, left, right; right, right, right, left"),
  Route("fork272", (-525.4, 161.7, 102.0), 0.2, ((39.7680603, -86.1612328),),
        "the rest of town15b's leg 2 from just past its left at (-539, 157), about 700 m: the fork at DR272_7 (-486, " +
        "159) that the route goes straight through (no maneuver; the model took the right branch), right, slight left"),
  Route("loop", *G62_START, ((39.7661729, -86.1647217), (39.7737206, -86.1659774)),
        "18 maneuvers, about 5.6 km: out (left, left, right, right, keep left, keep left, keep right, 1.5 km) and back " +
        "(keep right, keep left, left, keep left, left, keep right, keep left, right, keep left, right, left, 4.0 km), " +
        "ending about 110 m from the start"),
)}
