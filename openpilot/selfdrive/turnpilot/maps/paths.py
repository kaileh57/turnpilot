"""Where the map server's files live on the comma. TURNPILOT_MAPS_DIR overrides it, for tests and PCs."""
import os
import time
from pathlib import Path

DEVICE_MAPS_DIR = '/data/turnpilot/maps'  # beside /data/turnpilot/pyvalhalla, which survives updates


def maps_dir() -> Path:
  return Path(os.environ.get('TURNPILOT_MAPS_DIR', DEVICE_MAPS_DIR))


def wall_time() -> float:
  """Epoch seconds, for timestamps people read ("fetched 2 h ago"); intervals use time.monotonic."""
  return time.time()  # noqa: TID251
