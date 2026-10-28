import json
import os
import shutil
import subprocess

import pytest

from openpilot.tools.turnpilot import demo_video
from openpilot.tools.turnpilot.demo_video import Recording, cut, retime_factor


def rec(start, duration, factor=1.0, w=1280, h=720):
  return Recording('x.mp4', start, duration, w, h, factor)


class TestCut:
  def test_overlap(self):
    # chase from 100 to 160; the UI from 110, 30 s of video that took 31.5 s to draw
    begin, length, chase_off, ui_off = cut(rec(100.0, 60.0), rec(110.0, 30.0, 1.05))
    assert (begin, chase_off, ui_off) == (110.0, 10.0, 0.0) and length == pytest.approx(31.5)

  def test_run_window(self):
    begin, length, chase_off, ui_off = cut(rec(100.0, 60.0), rec(110.0, 30.0, 1.05), (120.0, 130.0))
    assert (begin, length, chase_off, ui_off) == (120.0, 10.0, 20.0, 10.0)

  def test_no_overlap(self):
    with pytest.raises(ValueError):
      cut(rec(100.0, 10.0), rec(200.0, 10.0))

  def test_route_stats_run(self):
    stats = [{'run': 1, 'route': 'town15b2', 'end': 'arrived', 'wall_start': 100.0, 'duration_s': 50.0, 'maneuvers_taken': 7,
              'maneuvers_attempted': 7, 'disengagement_count': 0}]
    assert demo_video.run_window(stats, 1) == (100.0, 150.0 + demo_video.RUN_TAIL_S)
    assert demo_video.caption_for(stats, 1) == "route town15b2   run 1 arrived   maneuvers 7/7   disengagements 0"

  def test_retime(self):
    # 826 frames at a nominal 60 fps drawn over 14.33 s: each video second is 1.041 real seconds
    meta = {'start': 0.0, 'end': 14.33, 'frames': 826, 'fps': 60}
    assert retime_factor(meta) == pytest.approx((14.33 + 1 / 60) / (826 / 60))
    assert retime_factor({'start': 5.0, 'fps': 10}) == 1.0  # the chase capture plays in real time already


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_side_by_side(tmp_path):
  chase, ui = str(tmp_path / "chase.mp4"), str(tmp_path / "ui.mp4")
  for path, size, rate, secs in ((chase, "320x180", 10, 6), (ui, "268x120", 60, 4)):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}:duration={secs}",
                    "-pix_fmt", "yuv420p", path], check=True)
  with open(chase + ".start.json", "w") as f:
    json.dump({'start': 1000.0, 'fps': 10}, f)
  with open(ui + ".start.json", "w") as f:  # 240 frames drawn over 4.4 s: retimed x1.1
    json.dump({'start': 1001.0, 'end': 1001.0 + 4.4 - 1 / 60, 'frames': 240, 'fps': 60}, f)
  out = str(tmp_path / "demo.mp4")
  demo_video.main(["--chase", chase, "--ui", ui, "--height", "180", "--strip", "30", "-o", out])
  info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height:format=duration", "-of", "json", out],
                                   check=True, capture_output=True, text=True).stdout)
  s = info['streams'][0]
  assert (s['width'], s['height']) == (320 + 402, 180 + 30)
  assert float(info['format']['duration']) == pytest.approx(4.4, abs=0.1)  # the UI's real span, inside the chase's 6 s
  assert os.path.getsize(out) > 0
