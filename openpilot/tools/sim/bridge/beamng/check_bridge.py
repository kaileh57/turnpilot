#!/usr/bin/env python3
"""Live check of the BeamNG bridge without openpilot: starts the bridge, then plays the parts of openpilot it talks to.

1. cameras: a VisionIPC client reads both streams (rate, duplicates, road/wide timestamps, age)
2. frame age end to end: turns the road camera to the sky in BeamNG and times the first VisionIPC frame that shows it
3. CAN: fingerprints the car from `can` and decodes it with opendbc's Tesla CarState (dashcamOnly, vEgo, angle, cruise)
4. GPS: gpsLocationExternal against frame.to_latlon of BeamNG's own vehicle position
5. control: engages cruise, then sends openpilot's own Tesla DAS_control and DAS_steeringControl on sendcan (accelerate,
   steer left, steer right, brake) and checks that the BeamNG car follows
Needs BeamNG.tech with nobody else driving it, and no openpilot running (this script publishes sendcan).
"""
import argparse
import json
import math
import time
from multiprocessing import Queue

import numpy as np
from beamngpy import BeamNGpy

import openpilot.cereal.messaging as messaging
from msgq.visionipc import VisionIpcClient
from openpilot.cereal.visionipc import VisionStreamType
from opendbc.can.packer import CANPacker
from opendbc.car import gen_empty_fingerprint
from opendbc.car.can_definitions import CanData
from opendbc.car.car_helpers import interfaces
from opendbc.car.tesla.teslacan import TeslaCAN
from opendbc.car.tesla.values import CAR
from openpilot.selfdrive.pandad.pandad_api_impl import can_capnp_to_list, can_list_to_can_capnp
from openpilot.tools.sim.bridge.beamng import check_vipc, frame, scenario
from openpilot.tools.sim.bridge.beamng.bridge import BeamNGBridge
from openpilot.tools.sim.bridge.beamng.cameras import bng_recv, bng_send
from openpilot.tools.sim.bridge.beamng.run_bridge import config_from_args, parse_args
from openpilot.tools.sim.bridge.common import control_cmd_gen

CANDIDATE = CAR.TESLA_MODEL_3


def wait_started(bridge, p, timeout=240):
  t_end = time.monotonic() + timeout
  while not bridge.started.value:
    assert p.is_alive(), "bridge exited"
    assert time.monotonic() < t_end, "bridge didn't start"
    time.sleep(0.2)


def flip_latency(host, port, n=6):
  """Turns both cameras to the sky (and back) in one burst, then finds the first VisionIPC frame of each that shows it.
  Returns [(ack s, ack -> road frame published s, road frame id - wide frame id)]: the age of a frame from the game frame
  that rendered it to VisionIPC, and whether both cameras of a pair come from the same game frame (0)."""
  bng = BeamNGpy(host, port).open(launch=False)
  clients = [VisionIpcClient("camerad", s, False) for s in (VisionStreamType.VISION_STREAM_NARROW_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD)]
  while not all(c.connect(False) for c in clients):
    time.sleep(0.1)

  def mean(c, buf):
    return float(np.frombuffer(buf.data, dtype=np.uint8)[:c.stride * c.height:97].mean())

  def turn(up: bool):
    ids = [bng_send(bng.connection, type="SetCameraSensorDirection", name=name, dirX=0.0, dirY=-0.15 if up else -1.0, dirZ=1.0 if up else 0.0)
           for name in ("tp_road", "tp_wide")]
    for i in ids:
      bng_recv(bng.connection, i)

  out = []
  try:
    for k in range(n):
      base = []
      for c in clients:  # drain, then take the current brightness
        buf = c.recv(200)
        while (b2 := c.recv(0)) is not None:
          buf = b2
        base.append(mean(c, buf))
      t_send = time.monotonic()
      turn(k % 2 == 0)
      t_ack = time.monotonic()
      seen = [None, None]
      while time.monotonic() - t_send < 3.0 and None in seen:
        for j, c in enumerate(clients):
          buf = c.recv(10)
          if buf is not None and seen[j] is None and abs(mean(c, buf) - base[j]) > 12:
            seen[j] = (time.monotonic(), c.frame_id)
      if None not in seen:
        out.append((t_ack - t_send, seen[0][0] - t_ack, seen[0][1] - seen[1][1]))
      time.sleep(0.5)
  finally:
    turn(False)
    bng.disconnect()
  return out


class FakeCard:
  """openpilot's Tesla CarInterface on the bridge's CAN, plus its sendcan."""

  def __init__(self):
    self.can = messaging.sub_sock('can', conflate=False, timeout=100)
    self.sendcan = messaging.pub_sock('sendcan')
    finger = gen_empty_fingerprint()
    n = 0
    while n < 300:
      for msg in messaging.drain_sock_raw(self.can, wait_for_one=True):
        for _nanos, frames in can_capnp_to_list([msg]):
          for addr, dat, src in frames:
            finger[src][addr] = len(dat)
        n += 1
    self.finger = finger
    self.CP = interfaces[CANDIDATE].get_params(CANDIDATE, finger, [], alpha_long=True, is_release=False, docs=False)
    self.CI = interfaces[CANDIDATE](self.CP)
    self.tesla = TeslaCAN(self.CP, CANPacker("tesla_model3_party"))
    self.CS = None
    self.frame = 0

  def update(self):
    msgs = messaging.drain_sock_raw(self.can)
    for nanos, frames in can_capnp_to_list(msgs):
      self.CS = self.CI.update([(nanos, [CanData(*f) for f in frames])])
    return self.CS

  def send(self, angle, accel, acc_state=4):
    msgs = [self.tesla.create_steering_control(angle, True)]
    if self.frame % 2 == 0:
      msgs.append(self.tesla.create_longitudinal_command(acc_state, accel, (self.frame // 2) % 8, self.CS.vEgo if self.CS else 0.0, True))
    self.sendcan.send(can_list_to_can_capnp(msgs, msgtype='sendcan'))
    self.frame += 1


def drive(card, bng, gps_sm, seconds, angle, accel, log):
  """Send commands at 50 Hz for `seconds`; returns [(t, vEgo, heading from the game, heading from gps, steering angle, aEgo)]."""
  rows = []
  t0 = time.monotonic()
  while time.monotonic() - t0 < seconds:
    cs = card.update()
    card.send(angle, accel)
    if card.frame % 10 == 0:
      _, hdg = scenario.ego_pose(bng)
      gps_sm.update(0)
      gps_hdg = 90.0 - gps_sm['gpsLocationExternal'].bearingDeg  # compass to counterclockwise from east
      rows.append((time.monotonic() - t0, cs.vEgo, math.degrees(hdg), gps_hdg, cs.steeringAngleDeg, cs.aEgo))
    time.sleep(0.02)
  d = [(rows[-1][i] - rows[0][i] + 180) % 360 - 180 for i in (2, 3)]
  log(f"  angle {angle:+.0f} accel {accel:+.1f}: speed {rows[0][1]:.1f} -> {rows[-1][1]:.1f} m/s (aEgo mean {np.mean([r[5] for r in rows]):+.2f}), " +
      f"heading change {d[0]:+.1f} deg game / {d[1]:+.1f} deg gps (left positive), steering angle {rows[-1][4]:+.1f}")
  return rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--vipc-seconds', type=float, default=20.0)
  ap.add_argument('--out', default='')
  args, rest = ap.parse_known_args()
  bargs = parse_args(rest)
  cfg = config_from_args(bargs)
  results = {}
  log = print

  q: Queue = Queue()
  bridge = BeamNGBridge(cfg, set_speed_kph=25.0, test_run=False)
  p = bridge.run(q)
  try:
    wait_started(bridge, p)
    log("bridge started; cameras:")
    results['vipc_stationary'] = check_vipc.measure(args.vipc_seconds)
    log(json.dumps(results['vipc_stationary'], indent=1))

    log("frame age by camera flip:")
    flips = flip_latency(cfg.host, cfg.port)
    results['flip'] = [{"ack_ms": round(a * 1e3, 1), "age_ms": round(b * 1e3, 1), "road_minus_wide_frames": d} for a, b, d in flips]
    log(f"  {results['flip']}")

    log("CAN:")
    card = FakeCard()
    for _ in range(50):
      cs = card.update()
      time.sleep(0.01)
    results['can'] = {"dashcamOnly": card.CP.dashcamOnly, "has_0x054_bus0": 0x054 in card.finger[0], "canValid": cs.canValid,
                          "vEgo": cs.vEgo, "steeringAngleDeg": cs.steeringAngleDeg, "cruise_available": cs.cruiseState.available,
                          "gear": str(cs.gearShifter), "alpha_long": card.CP.openpilotLongitudinalControl}
    log(f"  {results['can']}")

    log("GPS:")
    bng = BeamNGpy(cfg.host, cfg.port).open(launch=False)
    sm = messaging.SubMaster(['gpsLocationExternal'])
    errs = []
    for _ in range(10):
      sm.update(200)
      gps = sm['gpsLocationExternal']
      pos, _ = scenario.ego_pose(bng)
      lat, lon, _ = frame.to_latlon(*pos)
      x, y, _ = frame.from_latlon(gps.latitude, gps.longitude, gps.altitude)
      errs.append(math.hypot(x - pos[0], y - pos[1]))
    results['gps'] = {"hasFix": gps.hasFix, "lat": gps.latitude, "lon": gps.longitude, "beamng_lat": lat, "beamng_lon": lon,
                          "err_m_median": float(np.median(errs)), "bearing": gps.bearingDeg}
    log(f"  {results['gps']}  (the GPS sits at the camera mount, about 0.6 m ahead of BeamNG's vehicle origin)")

    log("control (engaged, openpilot's Tesla commands on sendcan):")
    q.put(control_cmd_gen("cruise_up"))
    time.sleep(0.5)
    rows = {}
    rows['accel'] = drive(card, bng, sm, 5.0, 0.0, 1.5, log)
    rows['hold'] = drive(card, bng, sm, 2.0, 0.0, 0.0, log)
    rows['left'] = drive(card, bng, sm, 3.0, 90.0, 0.0, log)
    rows['right'] = drive(card, bng, sm, 3.0, -90.0, 0.0, log)
    rows['brake'] = drive(card, bng, sm, 5.0, 0.0, -2.5, log)
    results['control'] = {k: {"v0": r[0][1], "v1": r[-1][1], "a_mean": float(np.mean([x[5] for x in r])),
                                  "dyaw_game": (r[-1][2] - r[0][2] + 180) % 360 - 180, "dyaw_gps": (r[-1][3] - r[0][3] + 180) % 360 - 180,
                                  "angle_end": r[-1][4]} for k, r in rows.items()}
    q.put(control_cmd_gen("cruise_cancel"))
    bng.disconnect()
  finally:
    q.put(control_cmd_gen("quit"))
    p.join(timeout=30)
  log(json.dumps(results, indent=1, default=str))
  if args.out:
    with open(args.out, 'w') as f:
      json.dump(results, f, indent=1, default=str)


if __name__ == "__main__":
  main()
