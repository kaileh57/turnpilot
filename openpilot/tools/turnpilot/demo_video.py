#!/usr/bin/env python3
"""Demo video: the BeamNG chase view and the openpilot UI side by side, cut to the time both recordings cover (or to one
scenario or route_drive run), over a caption strip with the scenario, the maneuvers taken and the disengagements.

The recordings line up by the wall-clock time of their first frames, from `<video>.start.json` next to each:
chase_render.py writes it for the chase view it renders after a drive, and record_ui.py for the UI (upstream RECORD=1). The
UI recorder writes frames at its target rate however fast the UI really drew them, so the UI's sidecar also holds its
frame count and real span, and the UI is retimed to real time. Without a sidecar, pass --chase-start/--ui-start (epoch
seconds); the file's creation time (macOS st_birthtime) is the last resort.

  python -m openpilot.tools.turnpilot.demo_video --chase run/chase.mp4 --ui run/ui.mp4 --result run/result.json --run 1 \\
    -o ~/turnpilot-artifacts/demo/crosswalk.mp4
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = Path(__file__).resolve().parents[2] / "selfdrive/assets/fonts/Inter-Medium.ttf"
FPS = 20
RUN_TAIL_S = 2.  # keep this much after a run ends: the car standing at the destination


@dataclass
class Recording:
  path: str
  start: float      # wall-clock s of the first frame
  duration: float   # s of video
  width: int
  height: int
  factor: float = 1.0  # real seconds per video second

  @property
  def end(self) -> float:
    return self.start + self.duration * self.factor


def probe(path: str) -> tuple[int, int, float]:
  out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration",
                        "-of", "json", path], check=True, capture_output=True, text=True).stdout
  info = json.loads(out)
  s = info["streams"][0]
  return int(s["width"]), int(s["height"]), float(info["format"]["duration"])


def retime_factor(meta: dict) -> float:
  """Real seconds per video second from a sidecar with frames, fps and the first and last frames' wall times."""
  if meta.get('frames', 0) > 1 and meta.get('fps') and meta.get('end') is not None:
    return (meta['end'] - meta['start'] + 1.0 / meta['fps']) / (meta['frames'] / meta['fps'])
  return 1.0


def load(path: str, start: float | None = None) -> Recording:
  w, h, duration = probe(path)
  meta: dict = {}
  side = f"{path}.start.json"
  if os.path.exists(side):
    with open(side) as f:
      meta = json.load(f)
  if start is None:
    start = meta.get('start')
  if start is None:
    start = os.stat(path).st_birthtime
    print(f"warning: no {side}, so {path} is aligned on its file creation time")
  return Recording(path, float(start), duration, w, h, retime_factor(meta))


def cut(chase: Recording, ui: Recording, window: tuple[float, float] | None = None) -> tuple[float, float, float, float]:
  """(begin wall time, length, chase offset, ui offset) in real seconds: the span both recordings cover, inside window."""
  begin, end = max(chase.start, ui.start), min(chase.end, ui.end)
  if window is not None:
    begin, end = max(begin, window[0]), min(end, window[1])
  if end - begin < 1.0:
    spans = f"chase {chase.start:.1f}-{chase.end:.1f}, ui {ui.start:.1f}-{ui.end:.1f}, window {window}"
    raise ValueError(f"the recordings overlap for {end - begin:.1f} s only ({spans})")
  return begin, end - begin, begin - chase.start, begin - ui.start


def even(x: float) -> int:
  return max(2, int(round(x / 2.0)) * 2)


def run_window(result: dict | list, run: int) -> tuple[float, float]:
  """Wall-clock span of a run, plus RUN_TAIL_S: from the scenario runner's result.json or route_stats.py's stats JSON."""
  if isinstance(result, list):
    r = result[run - 1]
    return r['wall_start'], r['wall_start'] + r['duration_s'] + RUN_TAIL_S
  r = result['runs'][run - 1]
  return r['t_start_wall'], r['t_end_wall'] + RUN_TAIL_S


def caption_for(result: dict | list | None, run: int) -> str:
  if result is None:
    return "TurnPilot"
  if isinstance(result, list):  # route_stats.py: one entry a run of route_drive
    r = result[run - 1]
    return "   ".join([f"route {r['route']}", f"run {r['run']} {r['end']}", f"maneuvers {r['maneuvers_taken']}/{r['maneuvers_attempted']}",
                        f"disengagements {r['disengagement_count']}"])
  r = result['runs'][run - 1]
  return "   ".join([result['scenario'], f"run {r['run']} {'pass' if r['passed'] else 'fail'}",
                      f"maneuvers {r['maneuvers_taken']}/{r['maneuvers_attempted']}", f"disengagements {len(r['disengagements'])}"])


def caption_png(text: str, width: int, height: int, path: str):
  """White on black in the system's plain style: one line, left aligned."""
  img = Image.new("RGB", (width, height), (0, 0, 0))
  draw = ImageDraw.Draw(img)
  font = ImageFont.truetype(str(FONT), int(height * 0.5)) if FONT.exists() else ImageFont.load_default()
  draw.text((int(height * 0.4), height // 2), text, fill=(255, 255, 255), font=font, anchor="lm")
  img.save(path)


def widths(chase: Recording, ui: Recording, height: int) -> tuple[int, int]:
  return even(chase.width * height / chase.height), even(ui.width * height / ui.height)


def ffmpeg_cmd(chase: Recording, ui: Recording, offsets, caption_path: str, out: str, height: int) -> list[str]:
  _, length, chase_off, ui_off = offsets
  wa, wb = widths(chase, ui, height)
  # trim on real time: the UI's timestamps are stretched to real seconds first
  graph = ";".join([
    f"[0:v]setpts=PTS-STARTPTS,trim=start={chase_off:.3f}:duration={length:.3f},setpts=PTS-STARTPTS,scale={wa}:{height},fps={FPS}[a]",
    f"[1:v]setpts=(PTS-STARTPTS)*{ui.factor:.6f},trim=start={ui_off:.3f}:duration={length:.3f},setpts=PTS-STARTPTS,"
    + f"scale={wb}:{height},fps={FPS}[b]",
    "[a][b]hstack=inputs=2[top]",
    "[top][2:v]vstack=inputs=2,format=yuv420p[v]",
  ])
  return [shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-i", chase.path, "-i", ui.path,
          "-loop", "1", "-framerate", str(FPS), "-t", f"{length:.3f}", "-i", caption_path,
          "-filter_complex", graph, "-map", "[v]", "-t", f"{length:.3f}", "-r", str(FPS),
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", out]


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--chase', required=True, help='BeamNG chase mp4 (beamng/chase_render.py)')
  ap.add_argument('--ui', required=True, help='openpilot UI mp4 (record_ui.py)')
  ap.add_argument('--chase-start', type=float, default=None, help="wall-clock time of the chase video's first frame")
  ap.add_argument('--ui-start', type=float, default=None, help="wall-clock time of the UI video's first frame")
  ap.add_argument('--result', default='',
                  help="the scenario runner's result.json or route_stats.py's <run>_stats.json, for the caption and the run's span")
  ap.add_argument('--run', type=int, default=1, help='run in --result')
  ap.add_argument('--caption', default='', help='caption text (default: from --result)')
  ap.add_argument('--height', type=int, default=540, help='height of each view')
  ap.add_argument('--strip', type=int, default=44, help='caption strip height')
  ap.add_argument('-o', '--out', default='demo.mp4')
  args = ap.parse_args(argv)

  chase, ui = load(args.chase, args.chase_start), load(args.ui, args.ui_start)
  result: dict | list | None = None
  window = None
  if args.result:
    with open(args.result) as f:
      result = json.load(f)
    window = run_window(result, args.run)
  offsets = cut(chase, ui, window)
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  with tempfile.TemporaryDirectory() as tmp:
    caption = os.path.join(tmp, "caption.png")
    caption_png(args.caption or caption_for(result, args.run), sum(widths(chase, ui, args.height)), args.strip, caption)
    subprocess.run(ffmpeg_cmd(chase, ui, offsets, caption, args.out, args.height), check=True)
  begin, length, chase_off, ui_off = offsets
  print(f"{args.out}: {length:.1f} s from wall time {begin:.2f}, chase +{chase_off:.1f} s, ui +{ui_off:.1f} s retimed x{ui.factor:.3f}")


if __name__ == "__main__":
  main()
