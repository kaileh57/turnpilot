# BeamNG scenarios

Scored runs of TurnPilot on the BeamNG bridge, and side-by-side demo videos of them.

## What a run does

Each run follows the same steps:

1. Teleport the car to the scenario's start.
2. Place the scenario's mannequin or AI cars.
3. Send the destination to turnpilot_web's `/api/destination` with the `TurnPilotWebToken` bearer, as the phone does.
4. Engage through the bridge with a rolling start.
5. Watch `turnpilotPlan`, `selfdriveState`, `carState`, `navInstruction`, `navRoute` and `modelV2` until the car arrives, disengages or times out. `ped_in_path` ends instead when the car has stood still for 4 s.

While it runs, the runner plays the driver: it holds the stalk whenever `turnpilotPlan.signalPrompt` asks for a signal, which is the PLAN's answer for cars where openpilot can't signal.

Everything a run spawns is removed after it. The parent process sweeps the game again by name (`tp_sc_*`) before the bridge starts and after it exits, so a killed bridge leaves nothing behind.

## Run it

Start openpilot first. The runner sets `NavigationEnabled` and `TurnPilotEnabled`, which navd, turnpilotd and turnpilot_web run on.

```sh
OPENPILOT_PREFIX=tpsim openpilot/tools/sim/bridge/beamng/launch_openpilot.sh
OPENPILOT_PREFIX=tpsim python -m openpilot.tools.sim.bridge.beamng.scenarios.run ped_crosswalk_right --runs 3 --record ~/turnpilot-artifacts/demo
```

Bridge options (`--host`, `--signals`, `--reload` and so on) pass through. Every run of an invocation shares one bridge session, because modeld doesn't reconnect to a new VisionIPC server.

- Without `--record`, results go to `~/turnpilot-artifacts/scenarios/<scenario>-<time>/result.json`.
- With `--record DIR`, they go to `DIR/<scenario>-<time>/`, along with `chase.mp4` (the bridge's `--capture`) and `ui.mp4` (the upstream UI recorded by `openpilot/tools/turnpilot/record_ui.py`). Each video gets a `<video>.start.json` sidecar holding the wall-clock time of its first frame.
- `--cleanup` only sweeps leftover scenario actors out of the game.

## Scenarios

The turn scenarios use junctions from piston's list of signalized junctions with crosswalks. Of the six he listed, (-445.3, 147.0) and (-32.5, 403.8) have no 90-degree turn in the road graph.

| Scenario | Setup | Pass |
| --- | --- | --- |
| `ped_crosswalk_right` | Right turn at (-768.6, 555.2). The mannequin leaves the inside corner, crosses the road being turned into, then walks 10 m away along the far sidewalk. It sets off when the car is 4 s from the hold line (plus its 1.1 s to the road edge). | Holds to a stop for "pedestrian in crosswalk", moves on, takes the turn, arrives, no disengagement, never within 2.5 m |
| `unprotected_left` | Left turn at (-650.1, 481.7). Two AI cars stand 60 and 85 m out on the arm straight across. They go together on an AI script, timed so the first reaches the junction 1.5 s after the car reaches its hold line. | Holds for "waiting for oncoming", goes after, takes the turn, arrives, no disengagement |
| `ped_in_path` | The mannequin stands in the lane 40 m before the junction at (-768.9, 661.4), on gate1's eastbound straight. | Stops with the camera 3 to 8 m from it (scene_policy stops 5 m short, camera to pedestrian), no disengagement |
| `occupied_lane` | A data entry (below). When the car engages, an AI car is placed in the target lane just ahead of it and paces it at the set speed until 6 s into the lane change window, then drops back to 40%. | The planner reports "car in target lane", and the lane change starts with the pacer more than 15 m away |
| `lane_change` | A data entry: a multi-lane approach to a turn that needs a lane change. | Changes lanes, takes the turn, arrives, no disengagement |
| `route_demo` | From 20 m past the bridge's default spawn to road-graph node DR249_3, about 850 m away. | Arrives, takes every maneuver, no disengagement |

Each scenario is an `Entry` in `catalog.py`. The builders there turn an entry into world positions from the road-graph export, without touching the game.

### Junction data

`lane_change` and `occupied_lane` read a JSON list, by default `lane_change_junctions.json` next to this file. Pick another file with `--data` and an entry with `--entry NAME`:

```json
{"name": "wpTown_5_left", "junction": [-600.7, 220.9], "turn": "left",
 "spawn": [-357.39, 472.27, 89.98], "heading_deg": -134.3, "dest": [-558.34, 178.34, 104.36], "speed_kph": 45}
```

Positions are BeamNG world meters and `heading_deg` is counterclockwise from east. `junction` and `turn` are needed only for the pacer. The two entries shipped here come from the road graph, and both spawn 350 m out in the right lane of a two-lanes-each-way approach to a left turn. The set speed is 45 km/h because DesireHelper changes lanes only above 20 mph.

## Results

`result.json` holds every run. Each run records:

- the outcome: `arrived`, `disengaged`, `stopped`, `timeout`, `no destination`, `no route`, `no engagement` or `error`
- maneuvers taken out of attempted. A maneuver is attempted once the car is within 30 m of it while engaged, and taken if navd then moves on to the next step without a reroute.
- disengagements, each with the driver override or the last alert as its cause
- holds, each with its reason, its duration, and whether the car stopped and then moved on
- the closest the camera mount came to the mannequin and to the other cars (`min_gap_ped_m`, `min_gap_car_m`)
- lane changes, and the pacer's offset when each started
- the rlog segments written during the run
- wall-clock start, engage and end times
- pass or fail with reasons
- the 20 Hz rows it was scored from

Rescore saved results, or score a recorded drive's rlogs (a real car drive works too):

```sh
python -m openpilot.tools.sim.bridge.beamng.scenarios.score result.json
python -m openpilot.tools.sim.bridge.beamng.scenarios.score --scenario route_demo ~/.commatpsim/media/0/realdata/<route>--*/rlog.zst
```

## Demo videos

```sh
python -m openpilot.tools.turnpilot.demo_video --chase DIR/chase.mp4 --ui DIR/ui.mp4 --result DIR/result.json --run 1 -o crosswalk.mp4
```

The tool puts the chase view and the UI side by side, each 540 px high, over a caption strip giving the scenario, pass or fail, maneuvers and disengagements. It cuts to the run's span, plus 2 s at the end.

- The two recordings line up on their sidecars.
- The UI recorder writes frames at a nominal 60 fps, but the Mac draws about 57, so the tool retimes the UI by the frame count and real span in its sidecar.
- `--chase-start` and `--ui-start` override the sidecars, for recordings that have none.

To test it without the bridge, play any video into the UI with `openpilot/tools/turnpilot/fake_ui_feed.py --video X --start-out X.start.json`, and record that with `record_ui.py`.

## Not yet run against the game

Everything here was built and tested offline, with unit tests and a dry run of the runner against a fake bridge, director, SubMaster and clock (`tests/`). These parts touch the live game and are unverified:

- The AI scripts (`ai.set_script`) for the oncoming cars and the pacer: whether the AI holds the timing, and whether a spawned car released from `brake=1` drives off.
- Where the crosswalk is. It's placed from road widths (2.5 m past the edge of the road the car comes from), not from the level's crossing decals, and `result.json` notes this.
- The mission-group sweep for mannequins (`getCount`/`getObject` on `scenetree.MissionGroup`) and the vehicle sweep by name.
- Whether turnpilot_web and navd are up, with a route source for the sim map, in the sim launch.
