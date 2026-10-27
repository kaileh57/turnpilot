# BeamNG.tech bridge

openpilot on the Mac drives a car in stock BeamNG.tech on the Windows PC, as a comma four in a Tesla Model 3. The bridge
runs on the Mac and talks to the game over BeamNGpy 1.36 at `192.168.137.1:25252`. It loads `west_coast_usa` with one
ETK 800 (the stock sedan closest to a Model 3: 4.67 m long and 1.39 m tall, against 4.69 m and 1.44 m) and publishes:

- `can` and `pandaStates` for a Model 3 on 3-bit DAS firmware, including `DI_autonomyHealth` (0x054) on bus 0, so card
  fingerprints `TESLA_MODEL_3` with `dashcamOnly` false and alpha longitudinal
- road and wide frames on VisionIPC at 1344x760 with `narrowRoadCameraState` and `wideRoadCameraState` (`sensor` os04c10)
- `accelerometer` and `gyroscope` at 100 Hz, `gpsLocationExternal` at 10 Hz (with a fix, in `frame.py`'s projection),
  and upstream's fake driver monitoring and peripheral state

openpilot's own `DAS_steeringControl` and `DAS_control` come back from `sendcan`, after openpilot's stock Tesla limits,
and drive the game car: the steering angle through the Model 3's 12:1 ratio and the ETK 800's 38 degree road-wheel lock,
the acceleration through a throttle and brake controller on the car's measured acceleration.

The bridge never closes the game. On exit it removes its sensors, brakes the car and disconnects, and the scenario stays
loaded so the next start takes seconds instead of a minute.

## Run it

Use the repo venv, from the repo root. `OPENPILOT_PREFIX` keeps msgq, params and logs apart from other runs on the Mac.

1. Start openpilot as a comma four in a Model 3:

   ```sh
   OPENPILOT_PREFIX=tpsim openpilot/tools/sim/bridge/beamng/launch_openpilot.sh
   ```

   It is upstream's sim launch with `FINGERPRINT=TESLA_MODEL_3`, alpha longitudinal on, and loggerd kept, so every run
   leaves an rlog in `~/.commatpsim/media/0/realdata`. Add more processes to block with `BLOCK=...`.

2. In a second terminal, start the bridge:

   ```sh
   OPENPILOT_PREFIX=tpsim python -m openpilot.tools.sim.bridge.beamng.run_bridge --engage --rolling-start 20 --set-speed 30
   ```

   `--engage` pulls the cruise stalk as soon as openpilot is engageable. With `--rolling-start`, the bridge first plays
   a driver who presses the gas up to that speed (km/h): the driving model holds a car that stands on an empty road,
   because its standstill acceleration stays under the 0.1 m/s^2 of `drive_helpers.should_stop`.

Other options: `--spawn x,y,z --rot qx,qy,qz,qw` (BeamNG world metres; the default is 140 m before the junction
`group_80_81_82_83`, eastbound in the right-hand lane), `--reload`, `--signals` (leave the traffic lights running; by
default every light is forced green and held), `--capture chase.mp4`, `--duration`, and `--camera-update` and
`--camera-poll` for the camera timing.

Without a TTY, the bridge reads one command per line on stdin: `cruise_up` (engage, or raise the set speed by 5 km/h),
`cruise_down`, `cruise_cancel`, `blinker_left`, `blinker_right`, `throttle_0.5`, `brake_1`, `steer_-0.2`, `reset` and
`quit`. With a TTY, the keys are upstream's (`tools/sim/README.md`).

## Gate 1

`gate1.py` checks whether one desire pulse makes the driving model take a 90-degree junction. It spawns the car on the
approach to a junction from the road-graph export, engages, and publishes `turnpilotPlan` at 20 Hz with the desire set
for `--pulse-frames` messages once the car is `--leads` metres from the junction. It logs the heading change, the path
and whether the car left on the exit road, to `--out`. Block turnpilotd, the other `turnpilotPlan` publisher:

```sh
BLOCK=turnpilotd OPENPILOT_PREFIX=tpsim openpilot/tools/sim/bridge/beamng/launch_openpilot.sh
OPENPILOT_PREFIX=tpsim python -m openpilot.tools.sim.bridge.beamng.gate1 --junction group_80_81_82_83 --turn left --leads 10,20,30,40
```

`--list` prints the signalised junctions with legal 90-degree turns. The road graph is read from
`~/pc_relay/incoming/osm_export/west_coast_usa_graph.json`.

## Scenario hooks

`scenario.py` works with any BeamNGpy client: in the bridge process use `world.bng` under `world.bng_lock`, from another
script open a second client. It can teleport the ego, read its pose, spawn BeamNG traffic, spawn and move the Italy
mannequin (the only human-shaped stock asset) as a pedestrian prop, and force or release traffic signals.
`BeamNGBridge(scenario=...)` runs a callable in the bridge loop at 100 Hz; gate1 is the example.

## Check it without openpilot

```sh
OPENPILOT_PREFIX=tpcheck python -m openpilot.tools.sim.bridge.beamng.check_bridge
```

It starts the bridge and plays the parts of openpilot the bridge talks to: a VisionIPC client, a camera-turn test that
times a frame from the game to VisionIPC, opendbc's Tesla CarState on `can`, the GPS against the game's own position,
and openpilot's Tesla commands on `sendcan`. `check_vipc.py` is the VisionIPC client on its own. Unit tests are in
`tests/`.

## Cameras

Both cameras sit at one mount near the top of the windshield, 1.34 m above the road, level and pointing straight
ahead: the road camera at a 36.82 degree vertical field of view and the wide one at 83.57 degrees, a pinhole stand-in
for the fisheye with the focal length modeld's warp assumes.

The game renders a camera on its frame once the camera's update time has passed. With a 0.02 s update time both cameras
render on every frame, and the game runs at about 24 fps on the PC's RTX 3050. A poll returns the newest render and the
game answers it on its main thread, so polling costs frame time: polling each camera every 10 ms on its own connection
dropped the game to 9 fps. The bridge polls both cameras in one pipelined burst on their own connection every 40 ms,
which puts both answers on the same game frame. A 20 Hz publisher drops a pair unless both frames are new by hash,
republishes the last pair after 150 ms without one (so modeld doesn't starve at a standstill), converts RGB to NV12 with
upstream's integer BT.601 math in uint16 numpy, and stamps both frames with one time.

Measured on 2026-10-27, game on the PC, bridge on the Mac over the USB4 link:

| What | Value |
| --- | --- |
| Fresh pairs published | 19.8 to 20.0 a second (game at 23 to 24 fps) |
| Road and wide from the same game frame | 5 of 5 camera-turn trials |
| Frame age, game frame to VisionIPC | 108 to 152 ms (median 128 ms), from the camera-turn test |
| Frame age, first sight on the Mac to VisionIPC | median 70 ms, p90 88 ms |
| RGB to NV12 | 3.4 ms a frame (upstream's int32 version: 6.6 ms), bit-identical |
| Ad-hoc polling, for comparison | 8 to 10.5 pairs a second, 100 to 570 ms old |
| With `--capture` | about 12 pairs a second |

## Known limits

- The game's own render pipeline adds about 130 ms between a game frame and its VisionIPC publish; the comma four's
  cameras add far less.
- The ETK 800's steering lock (38 degrees) comes from yaw rate at low speed on one street; paramsd learns the rest.
- The game's IMU has no noise or bias.
- `--capture` renders a third camera and costs about 8 camera pairs a second.
- modeld doesn't reconnect to a new VisionIPC server. If you run modeld by hand rather than through the manager, restart
  it whenever you restart the bridge.
- A teleport resets the car to P; the bridge shifts it back to D, but openpilot raises `wrongGear` in between and
  disengages, as a real Model 3 would.
