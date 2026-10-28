using Cxx = import "/include/c++.capnp";
$Cxx.namespace("cereal");

@0xb526ba661d550a59;

# custom.capnp: a home for empty structs reserved for custom forks
# These structs are guaranteed to remain reserved and empty in mainline
# cereal, so use these if you want custom events in your fork.

# DO rename the structs
# DON'T change the identifier (e.g. @0x81c2f05a394cf4af)

struct TurnpilotPlan @0x81c2f05a394cf4af {
  # Published by turnpilotd at 20 Hz. Everything the rest of openpilot takes from TurnPilot.
  enum State {
    disabled @0;     # TurnPilotEnabled off
    noRoute @1;      # enabled, no active route
    following @2;    # on route, nothing pending
    approaching @3;  # maneuver ahead, desire not yet sent
    turning @4;      # turn desire sent, maneuver in progress
    laneChange @5;   # nav lane change requested or in progress
    holding @6;      # scene logic is holding the car (pedestrian, oncoming, cross traffic)
    arrived @7;
    unavailable @8;  # TurnPilot isn't acting: the detector hasn't started yet, or it handed back (see handback)
  }
  enum Handback {
    none @0;
    detectorStopped @1;  # the detector produced nothing fresh within its deadline
    modelFellBack @2;    # modeld fell back from the big driving model to the small one
  }
  enum LaneChange {
    none @0;
    left @1;
    right @2;
  }
  enum Fault {
    none @0;
    error @1;           # turnpilotd failed this frame (logged); TurnPilot does nothing until it recovers
    noLongitudinal @2;  # openpilot doesn't control speed on this car, so TurnPilot does nothing
  }
  enum LaneMismatch {
    none @0;
    turnOnlyGoingStraight @1;  # the car is in a lane that only turns, at an intersection the route goes straight through
    wrongLaneForTurn @2;       # the route turns at the junction ahead and the car's lane doesn't turn that way
  }
  state @0 :State;
  desire @1 :UInt8;               # index into log.Desire; modeld uses it while DesireHelper is idle
  laneChange @2 :LaneChange;      # request for DesireHelper (acts like blinker plus nudge)
  signalPrompt @3 :LaneChange;    # ask the driver to signal (cars where openpilot can't)
  accelCap @4 :Float32;           # m/s^2, upper bound on planner accel; 10 means none
  speedCap @5 :Float32;           # m/s, turn speed cap; 100 means none
  shouldStop @6 :Bool;
  reason @7 :Text;                # one short line for the UI, e.g. "pedestrian in crosswalk"
  maneuverDistance @8 :Float32;   # m to the next maneuver
  maneuverType @9 :Text;          # OSRM/Mapbox maneuver type, e.g. "turn"
  maneuverModifier @10 :Text;     # e.g. "left", "slight right"
  routeValid @11 :Bool;
  shadow @12 :Bool;               # computed but not applied (TurnPilotShadowMode)
  fault @13 :Fault;               # why a disabled plan is disabled while TurnPilot is on; selfdrived shows it
  tooFastToTurn @14 :Bool;        # the next turn is skipped, too fast to make it: no desire, no cap; navd reroutes past it
  laneMismatch @15 :LaneMismatch; # shown and logged; the lane planner should have moved the car earlier
  maneuverAngle @16 :Float32;     # deg, unsigned: from the junction's bearings when maneuverGeometryValid, else nominal
  maneuverGeometryValid @17 :Bool;
  maneuverBend @18 :Bool;         # the road bends and no other road meets it there: the model's, not a turn
  handback @19 :Handback;         # why TurnPilot handed control back; selfdrived soft-disables on it (turnpilot/handback.py)
}

struct TurnpilotDetections @0xaedffd8f31e7b55d {
  # Raw YOLO26 detections on the full camera frame, published by modeld when the detector runs.
  enum Camera {
    road @0;
    wide @1;
  }
  struct Detection {
    cls @0 :UInt8;      # COCO class id (0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck)
    conf @1 :Float32;
    x0 @2 :Float32;     # pixels in the full sensor frame
    y0 @3 :Float32;
    x1 @4 :Float32;
    y1 @5 :Float32;
  }
  frameId @0 :UInt32;
  timestampEof @1 :UInt64;
  camera @2 :Camera;
  executionTime @3 :Float32;   # s
  detections @4 :List(Detection);
}

struct TurnpilotScene @0xf35cc4560bbf6ec2 {
  # Tracked, ranged objects, published by turnpilotd's scene module.
  enum ObjectClass {
    pedestrian @0;
    cyclist @1;
    car @2;
    truck @3;
    bus @4;
    motorcycle @5;
  }
  struct TrackedObject {
    trackId @0 :UInt32;
    cls @1 :ObjectClass;
    conf @2 :Float32;
    x @3 :Float32;      # m forward, calibrated (road) frame
    y @4 :Float32;      # m left
    vx @5 :Float32;     # m/s
    vy @6 :Float32;
    age @7 :UInt32;     # frames tracked
    inPath @8 :Bool;    # inside the planned driving corridor
    camera @9 :TurnpilotDetections.Camera;
    box @10 :List(Float32);  # x0, y0, x1, y1 normalized 0-1 in that camera's frame, for the UI
  }
  available @0 :Bool;          # false when detections are stale; never read as "nothing there"
  objects @1 :List(TrackedObject);
}

struct CustomReserved3 @0xda96579883444c35 {
}

struct CustomReserved4 @0x80ae746ee2596b11 {
}

struct CustomReserved5 @0xa5cd762cd951a455 {
}

struct CustomReserved6 @0xf98d843bfd7004a3 {
}

struct CustomReserved7 @0xb86e6369214c01c8 {
}

struct CustomReserved8 @0xf416ec09499d9d19 {
}

struct CustomReserved9 @0xa1680744031fdb2d {
}

struct CustomReserved10 @0xcb9fd56c7057593a {
}

struct CustomReserved11 @0xc2243c65e0340384 {
}

struct CustomReserved12 @0x9ccdc8676701b412 {
}

struct CustomReserved13 @0xcd96dafb67a082d0 {
}

struct CustomReserved14 @0xb057204d7deadf3f {
}

struct CustomReserved15 @0xbd443b539493bc68 {
}

struct CustomReserved16 @0xfc6241ed8877b611 {
}

struct CustomReserved17 @0xa30662f84033036c {
}

struct CustomReserved18 @0xc86a3d38d13eb3ef {
}

struct CustomReserved19 @0xa4f1eb3323f5f582 {
}
