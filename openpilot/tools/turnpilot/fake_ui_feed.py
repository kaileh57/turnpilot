#!/usr/bin/env python3
"""Feeds the openpilot UI on this Mac without the driving stack: onroad and engaged, a route banner, a TurnPilot plan,
and road camera frames from a video (BeamNG's chase capture will do), to test UI recordings and demo_video.

  OPENPILOT_PREFIX=tpdemo python -m openpilot.tools.turnpilot.fake_ui_feed --video chase.mp4 --start-out feed.mp4.start.json &
  OPENPILOT_PREFIX=tpdemo python -m openpilot.tools.turnpilot.record_ui ui.mp4
"""
import argparse
import json
import os
import shutil
import subprocess
import threading
import time

import numpy as np

from openpilot.cereal import custom, log, messaging
from openpilot.common.realtime import Ratekeeper
from openpilot.selfdrive.test.helpers import set_params_enabled
from openpilot.tools.sim.lib.camerad import rgb_to_nv12
from openpilot.tools.turnpilot.fake_cameras import H, W, FakeCameras, gray_nv12

SERVICES = ['deviceState', 'pandaStates', 'selfdriveState', 'carState', 'navInstruction', 'turnpilotPlan']


def video_frames(path: str):
  """NV12 frames of a video scaled to the comma four's 1344x760, in a loop."""
  while True:
    proc = subprocess.Popen([shutil.which("ffmpeg") or "ffmpeg", "-loglevel", "error", "-i", path, "-vf", f"scale={W}:{H}",
                             "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    while (buf := proc.stdout.read(W * H * 3)) and len(buf) == W * H * 3:
      yield rgb_to_nv12(np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3))
    proc.wait()


def send_frames(video: str, video_fps: float, stop: threading.Event, start_out: str = ''):
  """The road and wide cameras at 20 Hz, playing the video in real time. start_out: a sidecar for the video with the
  wall-clock time its first frame went out, so demo_video can line the video up against a UI recording of this feed."""
  cams = FakeCameras()
  frames = video_frames(video) if video else None
  frame, shown, t_first = gray_nv12(), 0, None
  rk = Ratekeeper(20, print_delay_threshold=None)
  while not stop.is_set():
    if frames is not None and t_first is not None:
      want = int((time.monotonic() - t_first) * video_fps)  # by the clock, so a slow decode never slows the video
      while shown < want:
        frame, shown = next(frames), shown + 1
    elif frames is not None:
      frame = next(frames)
    cams.send(frame, frame)
    if frames is not None and t_first is None:
      t_first = time.monotonic()
      if start_out:
        with open(start_out, 'w') as f:
          json.dump({'start': time.time(), 'fps': video_fps, 'source': f'fake_ui_feed playing {video}'}, f)  # noqa: TID251
    rk.keep_time()


def messages(t: float) -> dict:
  """Everything but the camera, t s into the drive: rolling up to a right turn 120 m out, holding for a pedestrian at
  the crosswalk from 8 to 12 s, then turning."""
  v = min(2.0 * t, 8.0) if t < 6 else (max(8.0 - 3.0 * (t - 6), 0.0) if t < 12 else min(2.0 * (t - 12), 6.0))
  dist = max(120.0 - 7.0 * t, -10.0)
  holding = 8.0 <= t < 12.0
  out = {s: messaging.new_message(s, 1, valid=True) if s == 'pandaStates' else messaging.new_message(s, valid=True) for s in SERVICES}
  ds = out['deviceState'].deviceState
  ds.started, ds.deviceType = True, log.InitData.DeviceType.pc
  ps = out['pandaStates'].pandaStates[0]
  ps.ignitionLine, ps.pandaType = True, log.PandaState.PandaType.redPanda
  ss = out['selfdriveState'].selfdriveState
  ss.enabled, ss.active, ss.state = True, True, log.SelfdriveState.OpenpilotState.enabled
  cs = out['carState'].carState
  cs.vEgo, cs.vEgoCluster, cs.cruiseState.enabled, cs.cruiseState.speed = v, v, True, 30 / 3.6
  nav = out['navInstruction'].navInstruction
  nav.maneuverPrimaryText, nav.maneuverType, nav.maneuverModifier = "Main Street", "turn", "right"
  nav.maneuverDistance, nav.distanceRemaining, nav.timeRemaining = dist, dist + 60.0, (dist + 60.0) / 8.0
  tp = out['turnpilotPlan'].turnpilotPlan
  State = custom.TurnpilotPlan.State
  tp.state = State.holding if holding else (State.turning if dist < 25 else State.approaching)
  tp.reason = "pedestrian in crosswalk" if holding else ""
  tp.maneuverDistance, tp.maneuverType, tp.maneuverModifier, tp.routeValid = dist, "turn", "right", True
  tp.accelCap, tp.speedCap, tp.shouldStop = 10.0, 100.0, holding
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--video', default='', help='video for the road camera (default: gray frames)')
  ap.add_argument('--video-fps', type=float, default=10.0, help="the video's frame rate (BeamNG chase: 10)")
  ap.add_argument('--seconds', type=float, default=20.0)
  ap.add_argument('--start-out', default='', help="write the video's first frame time here (a demo_video sidecar)")
  args = ap.parse_args()
  if os.environ.get('OPENPILOT_PREFIX'):
    os.makedirs(f"/tmp/msgq_{os.environ['OPENPILOT_PREFIX']}", exist_ok=True)  # macOS msgq lives there
  set_params_enabled()
  pm = messaging.PubMaster(SERVICES)
  stop = threading.Event()
  cam = threading.Thread(target=send_frames, args=(args.video, args.video_fps, stop, args.start_out), daemon=True)
  cam.start()
  rk = Ratekeeper(20, print_delay_threshold=None)
  t0 = time.monotonic()
  try:
    while (t := time.monotonic() - t0) < args.seconds:
      for s, m in messages(t).items():
        pm.send(s, m)
      rk.keep_time()
  finally:
    stop.set()
    cam.join(timeout=2.0)


if __name__ == "__main__":
  main()
