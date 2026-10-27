#!/usr/bin/env python3
"""The upstream openpilot UI, recorded with its own RECORD=1 to RECORD_OUTPUT, plus `<output>.start.json`: the wall-clock
time of the first recorded frame, and how many frames went out over how long. The recorder writes frames at the UI's
target rate whatever the real frame rate, so demo_video aligns on the start and retimes by the real rate.

  OPENPILOT_PREFIX=tpsim python -m openpilot.tools.turnpilot.record_ui ui.mp4

Stop it with Ctrl-C, SIGINT or SIGTERM. SIGKILL leaves an mp4 without its index, which nothing plays.
"""
import atexit
import json
import os
import signal
import sys
import time
from pathlib import Path


def wall() -> float:
  return time.time()  # noqa: TID251  recordings are aligned on wall-clock time


class Stamp:
  def __init__(self, video: str):
    self.path = f"{video}.start.json"
    self.start = self.end = None
    self.frames, self.fps, self.saved = 0, 0, 0.

  def frame(self):
    t = wall()
    self.start = t if self.start is None else self.start
    self.end, self.frames = t, self.frames + 1
    if self.frames == 1 or t - self.saved > 2.:
      self.save()

  def save(self):
    if self.start is None:
      return
    self.saved = wall()
    with open(self.path, 'w') as f:
      json.dump({'start': self.start, 'end': self.end, 'frames': self.frames, 'fps': self.fps, 'source': 'openpilot ui'}, f)


def main():
  out = str(Path(sys.argv[1] if len(sys.argv) > 1 else 'ui.mp4').with_suffix('.mp4'))  # the recorder forces .mp4 too
  os.environ['RECORD'] = '1'
  os.environ['RECORD_OUTPUT'] = out
  from openpilot.system.ui.lib import application  # reads RECORD and RECORD_OUTPUT on import
  stamp = Stamp(out)
  init_window = application.GuiApplication.init_window

  def stamped_init_window(app, *args, **kwargs):
    init_window(app, *args, **kwargs)
    stamp.fps = app.target_fps
    q = app._ffmpeg_queue
    put = q.put

    def put_frame(item, *a, **kw):
      if item is not None:  # None is the recorder's stop sentinel
        stamp.frame()
      put(item, *a, **kw)
    q.put = put_frame

  application.GuiApplication.init_window = stamped_init_window
  atexit.register(stamp.save)
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # exit through atexit, which closes the recorder
  from openpilot.selfdrive.ui import ui
  ui.main()


if __name__ == "__main__":
  main()
