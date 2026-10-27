"""Chase-view recording for demo videos: a third BeamNG camera behind and above the car, 1280x720 at 10 Hz (or fps), piped to
ffmpeg. Every chase render costs the driving cameras game frame time: at 10 Hz the pair rate fell to about 12 a second."""
import shutil
import subprocess
import threading

from beamngpy.sensors import Camera

from openpilot.common.realtime import Ratekeeper
from openpilot.tools.sim.bridge.beamng.cameras import CameraStream

CW, CH, FPS = 1280, 720, 10
CHASE_POS = (0.0, 6.5, 2.6)  # vehicle frame: 6.5 m behind, 2.6 m up
CHASE_DIR = (0.0, -1.0, -0.18)


class ChaseCapture:
  def __init__(self, bng, bng_lock: threading.Lock, vehicle, host: str, port: int, path: str, log=print, fps: float = FPS):
    self.path, self.log, self.bng_lock, self.fps = path, log, bng_lock, fps
    with bng_lock:
      bng.connection.send({"type": "CloseCamera", "name": "tp_chase"}).recv()
      self.cam = Camera("tp_chase", bng, vehicle, requested_update_time=1.0 / fps, pos=CHASE_POS, dir=CHASE_DIR, up=(0, 0, 1),
                        resolution=(CW, CH), field_of_view_y=55, near_far_planes=(0.1, 1000.0), is_using_shared_memory=False,
                        is_render_annotations=False, is_render_depth=False, is_render_instance=False)
    self.stream = CameraStream(host, port, ["tp_chase"], 1.0 / fps, size=(CW, CH))  # every poll costs game frame time
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
      raise RuntimeError("capture needs ffmpeg on PATH")
    self.proc = subprocess.Popen([ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{CW}x{CH}",
                                  "-r", str(fps), "-i", "pipe:0", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", path],
                                 stdin=subprocess.PIPE)
    self.frames = 0
    self.threads: list[threading.Thread] = []

  def start(self, exit_event: threading.Event):
    self.threads = [threading.Thread(target=self.stream.run, args=(exit_event,), daemon=True),
                    threading.Thread(target=self.write_loop, args=(exit_event,), daemon=True)]
    for t in self.threads:
      t.start()

  def write_loop(self, exit_event: threading.Event):
    # one frame per tick, repeating the last when the game is slow, so the video plays in real time
    rk = Ratekeeper(self.fps, print_delay_threshold=None)
    while not exit_event.is_set():
      f = self.stream.latest[0]
      if f is not None:
        self.proc.stdin.write(f.rgb[:, :, :3].tobytes())
        self.frames += 1
      rk.keep_time()

  def close(self):
    for t in self.threads:
      t.join(timeout=2.0)
    self.stream.close()
    self.proc.stdin.close()
    self.proc.wait(timeout=30)
    with self.bng_lock:
      try:
        self.cam.remove()
      except Exception as e:
        self.log(f"chase camera remove failed: {e!r}")
    self.log(f"capture: {self.frames} frames ({self.frames / self.fps:.0f} s) written to {self.path}")
