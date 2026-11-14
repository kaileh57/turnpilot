# Confident test build: what was loosened

This branch is a supervised test build. It drives on its best judgement where a public build would wait, slow or hand
back. This file lists every rule loosened for it, with the earlier value, so a public build can restore them.

The hard rails are unchanged: a read red or a stoppable yellow stops the car, a person in the path holds it, an oncoming or
crossing vehicle the cameras see holds a left or a stop-sign go, a left from the wrong lane is skipped and rerouted, real
faults alert, and driver monitoring is stock.

## Unprotected lefts (`unprotected_left.py`, `turnpilotd.py`)

| Rule | Public value | This build |
| --- | --- | --- |
| `WINDOW_FROM_APPROACH` | the clear window restarts at the hold line | counts on the approach too, so a left goes as it comes to its stand, or without one (`ROLLING_GO`) |
| `ROLLING_GO` | every left stops at its line first | a left whose oncoming band has been clear the whole window on the approach goes without stopping. Asked from `ROLL_LEAD_S` (1.0 s) before the stop for the line would take the car under its corner speed, never later than where a comfortable stop (`ROLL_STOP_LAG_S` 0.5 s, 1 m to spare) still ends short of the oncoming lanes. Needs everything the go from a stand needs, a turn signal lit, the turn lane reached, and at a signal (mapped or boxed by the camera) its light read a steady green (`CAMERA_GREEN_READS` road runs over `CAMERA_GREEN_STEADY_S`); never on an unread light. Given only where a pass that runs covers the range the roll's whole exposure needs seen (0.5 s, down to the corner speed, through the oncoming lanes, plus `PET_S`, at 1.3 times the opposing limit): the full pass to its own range, past it only a far slot whose runs cover the band now, and no further than the opposing road is known straight. Otherwise the car brakes to its line and is judged from the stand as before (`roll_why` 'range'). On the car's 98 m full pass alone that is no roll against an opposing limit of 20 mph or more; with a far slot running at 210 m it rolls to 35 mph and not at 45. Checked on every frame after it; a car, a person or a red seen while the car can still stop withdraws it, and so does a road view gone stale (no road run within the freshness bound) or a turn that has not pulsed in `ROLL_PULSE_S` (0.5 s). "Can still stop" budgets the same 0.5 s of actuator lag as the go's own last point: past where that stop ends `HELD_MIN_M` (4.5 m) short of the junction the left carries through, never braked to a stand in the oncoming lanes. A red read now is stopped for further in, for as long as the red's own stop (`RED_LAG_S`, then `RED_DECEL`) still ends there with 1 m to spare |
| Box motion with the car moving (`scene/tracker.py`) | above 3 m/s (`MOTION_EGO_MAX_MPS`) a box is closing or unknown, never cleared | the car's own travel and turn are taken out of every sighting; a box reads standing only with `MOTION_EGO_MARGIN` (0.2) of the closing speed taken out to spare, receding only with `MOTION_EGO_MARGIN_AWAY` (0.35) to spare, both only where the ground range and the box width agree. `MOTION_AWAY_SIGMA` (3) is stricter than before: receding needs a fit three standard errors clear of zero, with the car at rest too |
| `LEFT_SIGNAL_WAIT_S` | 10 s, and never a go with no turn signal lit | 3 s, then the left goes on its gap unsignalled |
| `LEFT_LAUNCH_STAND_S` | 2.0 s (`NO_MOVE_S`) before the no-move fault | 4.0 s |
| Oncoming band boxes | every vehicle box in the band blocks | a box going away up the road on our own side of the oncoming lanes, or standing wholly outside the opposing lanes, does not block. Standing beyond the lanes' far edge (the map's count) it is exempt only where the driving model's own far road edge, read on the approach, shows a strip there and no room for a lane the map does not count (`FAR_EDGE_SHORT_M` 1 m, `FAR_LANE_MIN_M` 2.5 m, two in three of the last `FAR_EDGE_READS` 40 trusted reads, 10 at least); with no trusted edge it blocks |
| Road view freshness (`ROAD_FRESH_S`) | a road run more than 0.5 s after the last starts the clear window over | 0.5 s, or 1.25 times the road pass's own cadence where it runs slower (the median gap of its last 5 runs), to `ROAD_FRESH_MAX_S` (0.83 s, a pass down to 1.5 Hz); what that takes past 0.5 s is added to T_req, so R_req reaches further |
| A committed left and a red | a red read refuses the go | a left that can no longer stop short of the oncoming lanes finishes: on the comfortable curve for a yellow or a red no longer read, on the red's own stop (0.5 s, then 3.0 m/s², 1 m to spare) for a red read now |

| A go from a stand withdrawn (`go_can_stop`, `go_committed`) | withdrawn wherever the comfortable curve (v²/2·1.5) ends short of the oncoming lanes; committed past the hold line once it no longer does | the rolling go's rule: withdrawn only where that stop begun `ROLL_STOP_LAG_S` (0.5 s) on, 1 m to spare, still stands the car `HELD_MIN_M` (4.5 m) short of the junction. Past it the left carries through at its post-go profile (a slow one carried by the lift), the scene's junction holds yield to it past the hold line, and it commits where not even a read red's firm stop fits (never while another stop brakes it; stood by one where its own hold stands it, it is judged afresh). From the line the last withdrawal comes at about 2.0 m/s, 7.9 m out (it was about 2.8 m/s, 6.6 m); from a full creep (5 m) none after the first motion, from a signal's mouth (6 m) to about 0.7 m/s. L3's slow-launch check moved with it: every launch from the line is judged where its withdrawal still fits |
| A left's 'stale' handback (`STALE_UNDECIDED_S`) | latched for the rest of the left | the car still standing, the first fresh road run judges it again on a fresh window; never raised on the frame the left comes back on after turnpilotd's own detector handback |
| A gap in the route at a left's line | the first invalid `navInstruction` drops the maneuver: the line's latch hands back and the left comes back on a new id at its line ('Junction position unsure') | a left standing on its line's latch, not gone, rides invalid messages as a gap (`route_follower` `line_hold`) up to `NAV_STALE_S` (3 s) from the first of them, nothing said, and never goes while it lasts (its clear window starts again once the route is back). Longer, or navd back on another route (a new routeCount: the drop the gap stood in for), ends it with the same handback. A left on its own hold after a withdrawn go is held as a committed turn, as before |
| "Turning" chime | every `leftGo` onset chimes | a go of the same left within `PROMPT_REPEAT_S` (8 s) of a withdrawn go that was shown keeps its banner and is silent (`unprotectedLeft.goRepeat`); the first go of a left, a go after that window and a quiet retry's first audible go still chime |

## Lights (`map_controls.py`, `detector/light_state.py`)

| Rule | Public value | This build |
| --- | --- | --- |
| `RED_DECEL` | 2.0 m/s² | 3.0 m/s² for a read red |
| `RED_LAG_S`, `RED_PAST_M` | none; an overrun 2 m past the line drops the stop | 0.5 s and 4 m: a red is stopped for if the stop ends within 4 m past its line, otherwise the car goes through and says so |
| `RED_GONE_S` | a camera red holds until a steady green | released after 2.5 s unbroken of a head that read the red in view and reading neither red nor yellow, in at least `RED_GONE_READS` (6) road runs. The head is known again by the reader's id for it, or by its place (`HEAD_PLACE_M` 5 m, `HEAD_PLACE_DEG` 1 degree); another head of the signal says nothing. Nor does one whose read says nothing of its lamp (every lamp dark, not read as a vertical head, two lamps lit), nor any head while a low-confidence red is read for the signal |
| Camera green | every head of the signal agrees | the signal's state is its movement's (`movement_state`), for its red and its green alike. Only heads that read a colour speak, and of them the nearest cluster (`HEAD_CLUSTER_M`, 40 m of the nearest): the next junction's heads have no say while this line's own read a colour. Going straight or turning right: the state most of them read; on a tie the head nearest the car's heading (within `HEAD_PLACE_DEG`), else red or yellow. A route left across the oncoming lanes: the left-most head (its red holds the left beside green balls; its green beside red balls is the left's green, the oncoming lanes judged as before); where bearings do not single one out, red or yellow. One head in view governs. A red or yellow read under `LIGHT_READ_CONF` raises no stop, and is a veto of a green or a red gone only where, were it a head, it would change that state. The same head's green counts wherever its range reads once it has read the signal's red |
| Camera green, steady | `CAMERA_GREEN_STEADY_S` (1.0 s) of green on every frame, unbroken | `CAMERA_GREEN_READS` (3) road runs reading green, the first and the last 1.0 s apart or more and the last no older than `CAMERA_STALE_S` (1.0 s), no red or yellow read between; a run that reads nothing does not start it over (`CAMERA_GREEN_GAP_S` 1.5 s ends it) |
| `VOTE_N` | 4 of 5 reads | 2 of the last 5 sightings within `VOTE_S` (1.5 s), a tie is unknown; a head is kept `HEAD_DROP_S` 1.5 s unseen (was 0.6 s) |
| `LIGHT_GREEN_CONF` | the reader takes traffic-light boxes down to `LIGHT_READ_CONF` (0.2) | down to 0.1 a box within the governing bearing is read too, and is a head only on a run it reads green (at most `GREEN_LOW_MAX` 2 a frame). A red or yellow read there is no head and raises no stop; it vetoes a green and a red gone for its signal. On a run with a low green the low boxes past `GREEN_LOW_MAX` are read for one too |
| `ROUND_MAX` | a glow filling its box is never a lamp | a round red or green glow filling its box is read (1.6 times as tall as wide at most). Not measured by day |
| Lamp position rule | red counted in the upper half, green in the lower | hue only; position places a yellow only |
| Right at a signal | goes with the model when no colour is read | flows only on a read green, at a mapped signal and wherever the camera boxes a governing head at its junction in `TURN_HEAD_RUNS` (2) road runs; otherwise stops at the line, stands `RIGHT_RED_DWELL_S` (1.5 s), looks for `RIGHT_RED_CLEAR_S` (1.0 s) and goes. Never while a red has been read in the last `RIGHT_RED_UNREAD_S` (2.5 s) and `RIGHT_RED_UNREAD_READS` (6) road runs, at any run rate (the seconds alone only once the reader has given no run for `RIGHT_RED_QUIET_S`, 10 s). A red or yellow read again after its go, while a stop still ends within `RED_PAST_M` of the line, stops it again. A stand for an unread light ends `TURN_DROPPED_S` (1.0 s) after the turn is dropped |
| Right on red and an uncontrolled junction: side views | the look and the cross-traffic hold judge the forward passes' tracks only (the side views never ran there) | at or under 2 m/s (`SWEEP_V_MAX`) turnpilotd asks for `sideLook` while a curb-side turn's signal stops or holds the car for its right-on-red look, and while a turn at a junction with no mapped control is where its cross-traffic hold could stand the car; never over the left's view or a stop sweep. The side views' tracks hold both as crossing traffic. A car only a side view sees stays a new track for about 11 s (`VEL_TRUST_HITS` 25 at about 2.3 a second): standing where the right-on-red look's 'new' rule reaches it, it holds the right that long; closing inside `CROSS_Y_NEAR_M` (13 m) it no longer holds an uncontrolled turn (Sim: the car stood short of the through road as it crossed). The wide pass runs about 2.3 a second there (was 5 or more): a walker only it sees must be in view about 0.7 s before the right-on-red go to hold it (0.2 s without the look); seen later, the go is let go and the scene's crosswalk hold stands the car again at under 0.5 m/s, short of them |
| A red's match to its signal | a head within 10 m before to 40 m past the line, by its range as read | the same for every head; a red or yellow no line takes so is the first line's ahead that the span from its nearer range to its read range reaches. The nearer range is the reader's second estimate, from the box's height against a 1.07 m head (`HEAD_M`), never nearer than a head 3.0 m up would be (`LOW_HEAD_M`), and at most the read range over `CAMERA_RANGE_LONG` (1.4). A green and an unread head keep the window as read |
| A signal on its approach's own node | its line `CONTROL_LINE_M` (10 m) before the node | `SIGNAL_NODE_LINE_M` (4 m) before a signal node no other road meets (`routing/controls.py`): route 29's three such signals had the driver standing 0.3-4.8 m short of the node |
| A line the driver crossed | its take-control and reason up until 30 m past it, engaged or not, and `stopOverrun` (owner decision (b)) | the driver's own crossing (his gas, or openpilot disengaged) is logged only: no prompt, no chime. A crossing with openpilot engaged and no gas is said as before |
| Camera stop with no route | dropped with the route's maneuver unless already braking | a stop a camera red or yellow raised stays through a route gap and is the same junction's under the route's next maneuver id; only its light lets it go |
| Off-route speed cap | 11.2 m/s with an alert when a destination is set and no route is valid | no cap, one silent status line (with `TurnPilotAlwaysDrive` on) |
| A stop driven past under the gas | dropped when the gas lifts | dropped 2 m past the line above 2 m/s |

## Stop signs (`stop_go.py`, `stop_rules.py`)

| Rule | Public value | This build |
| --- | --- | --- |
| `VISION_CLEAR_S` | 2.5 s | 1.5 s |
| `VISION_LONG_CLEAR_S` | 4.0 s | 2.5 s |
| `VISION_ANNOUNCE_S` | 1.0 s | 0.5 s |
| `VISION_CREEP_AFTER_S` | 2.0 s | 1.0 s |
| Creep for a view | on every wait for reach, in 0.5 m steps | only for an occluder, or where the furthest reachable stand can show enough |
| Vision gate handbacks | skewed junction, unknown leg, no sidecar, side views off, divided road, roundabout, stale map | none of these; only a rail crossing, a missing visibility file or a lens mismatch hand back |
| Washed or frozen side view, yaw over 8° | holds with no bound | goes on what is seen after the long look |
| A junction the map has nothing for | handback | judged as a plain cross road (`ASSUMED_LINE_M` 10 m) |
| Position fault (H2): navd's fixes of the stop spread over 4 m in its last 20 m | handback after 10 s at the line | not for a step (`H2_STEP_M`, 2 m between two fixes in a row: a GPS jump that stays or comes back): the stop is where the fresh reading puts it and the car stops, looks and goes. Fixes that keep disagreeing, or drift apart with no step, still hand back |

## Lane changes and prompts (`lane_planner.py`, `hooks.py`)

| Rule | Public value | This build |
| --- | --- | --- |
| `LC_END_S` / `LC_END_MIN_M` / `LC_END_MAX_M` | 6 s / 60 m / 120 m | 4.5 s / 35 m / 100 m |
| Change into a turn lane the map's lane list names: confirm frames | `LC_CONFIRM_FRAMES` (10) | `LC_BAY_CONFIRM_FRAMES` (5, two of them frames whose paint was read) once the full count and the offer no longer fit before the window's end; 10 with room to spare. Every other change: 10 |
| Change into a turn lane the map's lane list names: window end | the general end | `LC_CHANGE_S` (5 s) of travel at the car's speed, never under `LC_BAY_END_MIN_M` (15 m), never before the general end. A lane not entered by then is still skipped out loud |
| `LC_ABORT_COOLDOWN_S` | 6 s | 3 s |
| `PAINT_CARRY_S` | none: a frame with no paint read is unread | 0.25 s: judged on the last read |
| Solid white line | never crossed | crossed into a turn lane the map's lane list names, when the model sees the lane and it is free. Never yellow, never a line with no colour read |
| Signal prompt chime | loops for the whole prompt | once, then every `PROMPT_REPEAT_S` (8 s); the lane change's ask to signal too |
| Lane change ask | after a 3 s countdown | with the offer |

## Routing and handbacks (`navd.py`, `routing/avoid.py`, `handback.py`)

| Rule | Public value | This build |
| --- | --- | --- |
| Stop signs under vision stop-go | routed around when the line would hand back | never routed around |
| Detector handback | latched to a full disengage | clears if the detector returns within 3 s |
| Detector re-measure | waits 16 frames (0.8 s) after any pass over budget | skips one run (`FORCE_SKIP_FRAMES` 4) when only slightly over |
| Side views at a stand (`detector/modeld_detector.py` stand schedule) | the held cycle: full wide and road passes 5 a second each, a side view only where it leaves `DEBT_CAP_S` (8 ms) of debt, which on this car is never past the first | once the side views have had no frame for `STAND_STARVED_S` (0.3 s) at or under 2 m/s in a sweep or a `sideLook`: the road pass every `STAND_ROAD_DUE_S` (0.2 s), the wide pass and each side view every `STAND_VIEW_DUE_S` (0.4 s), a side run allowed `SIDE_STAND_CAP_S` (16.7 ms, a third of a frame) of lateness, and counted an overrun only past half a frame (`SIDE_STAND_OVERRUN_S`). On the car's timings: road 4.4-4.7 a second (was 5), wide 2.2-2.3 (was 5), each side view 2.1-2.3 (was 0); a sweep's tracks coast `SWEEP_COAST_S` 0.9 s (was 0.65) |

## Not proven on the car

Everything above is tested in unit simulations and open-loop replays of logged drives
(`openpilot/tools/turnpilot/drive_scorecard.py`). None of it has been driven, and no left's verdict changed on a replayed
drive: the replay is open loop and reuses the logged build's detections. The right-on-red look, an uncontrolled junction's
cross-traffic hold and the stop-sign go now ask for the side views at walking pace, but no drive log has side-view runs at
them yet: on the logs they judged cross traffic on the forward cameras only. No logged left would have taken the rolling go.

Reds at night: on the last logged night drive the car read no red at four of five red stands. That build made a head only
from a box scoring 0.35 or more, and most heads there score 0.20 to 0.35. The reader now reads down to 0.2 (0.1 for a
green), and on that drive's video at reduced resolution it votes red at every stand. That is a proxy, not the car's own
pass: `lightRejects` in the next drive's log will show what the car's detector had. Until then treat a red at night as
possibly unread: a left then rolls to the junction mouth and a straight goes through.

Still seen in replay: the car brakes for several seconds at a light that has just turned green, where the green's heads
range further than the reader's match to the mapped signal allows. Greens are read at a lower confidence than reds, and a
read green can release a camera red's stop, let a right flow and let a left roll. The reader's and tracker's added cost was
timed on a desktop only.

Open after the 2026-11-14 round, each reproduced and left unchanged because no honest fix exists off the car:

- A stand at a green whose only read heads hang on the far side. Those heads range past the 40 m match to the line, so a
  stop raised by a red read on the approach is not released by them: in replay the car brakes 5.2 s into one such green and
  would stand at its line. A release by longer range was built and parked: the map tiles do not mark signalled mid-block
  crossings, so nothing tells this junction's far heads from the next junction's. The driver's gas releases the stand.
- A next junction's green 50-60 m past the line that reads a quarter short falls inside the same match and can release a
  camera red whose own heads read nothing.
- A red first read inside 22 m at 25 mph, 33 m at 30, 45 m at 35 or 58 m at 40 is driven through, since a stop no longer
  fits at 3.0 m/s². The hardest braking ever commanded in the logs is 2.2 m/s²: whether the car delivers 3.0 needs a brake
  test.
- A navd reading that steps several metres nearer while a left stands at its line hands the left back for good.
- The stand schedule's side-view runs were timed on a desktop scaled to the car (8.8 times, give or take a third). A side
  run may leave a third of a frame of lateness; overruns past half a frame turn the side views off for the drive.
- With the side look on, the wide camera's full pass runs about 2.3 times a second at a stand, so a walker only it sees is
  tracked about 0.4 s after first boxed (0.2 s before). A car first boxed by a side view inside the last side period
  (under 0.5 s) before a right's go does not hold it.
