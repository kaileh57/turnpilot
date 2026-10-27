"""The detector comes back: after slow runs, after failures, after TurnPilot is turned on mid-drive, after a model fallback."""
from collections import deque

from openpilot.cereal import log
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.turnpilot.detector.modeld_detector import (FORCE_RUN_FRAMES, FORCE_RUN_MAX_FRAMES, RETRY_MAX_S, RETRY_S,
                                                                     RUN_EVERY, STEP_BUDGET_S, DetectorHost)
from openpilot.selfdrive.turnpilot.detector.tests.test_modeld_detector import PM, Meta, make_detector

METAS = [Meta(1), Meta(2)]


def run_gaps(det, pm, frames: int, model_time: float) -> list[int]:
  """Frames between the detector's runs over `frames` model frames."""
  gaps, last = [], 0
  for k in range(1, frames + 1):
    n = len(pm.sent)
    det.step(pm, det.model_state, model_time, 0.0, log.Desire.none, METAS)
    if len(pm.sent) > n:
      gaps.append(k - last)
      last = k
  return gaps


class FakeParams:
  def __init__(self, enabled=False):
    self.enabled = enabled

  def get_bool(self, key):
    assert key == "TurnPilotEnabled"
    return self.enabled


class FakeLoad:
  """Stands in for modeld_detector.load: a fake detector for each model in `have`, else None (no pkl for that device)."""
  def __init__(self, have):
    self.have, self.calls = have, []

  def __call__(self, model_state, *args):
    self.calls.append(model_state)
    if model_state not in self.have:
      return None
    det = make_detector()
    det.model_state = model_state
    return det


class TestDetectorRecovery(OpenpilotTestCase):
  def test_slow_burst_is_measured_again(self):
    # three slow runs while the Mac was loaded left a 60 ms median; the load has gone, so the next run is fast
    det, pm = make_detector(), PM()
    det.times = deque([0.06, 0.07, 0.09, 0.005, 0.005], maxlen=5)
    gaps = run_gaps(det, pm, 40, 0.020)
    assert gaps[0] == FORCE_RUN_FRAMES - RUN_EVERY  # forced, the estimate starting over from that run...
    assert set(gaps[1:]) == {RUN_EVERY}             # ...which fits, so back on the N-frame grid
    assert det.force_after == FORCE_RUN_FRAMES

  def test_no_room_backs_off(self):
    # the driving model alone overruns the budget: a forced run now and then, each further apart, to the cap
    det, pm = make_detector(), PM()
    gaps = run_gaps(det, pm, 1200, STEP_BUDGET_S + 0.005)
    assert gaps[:5] == [FORCE_RUN_FRAMES - RUN_EVERY, 2 * FORCE_RUN_FRAMES, 4 * FORCE_RUN_FRAMES, FORCE_RUN_MAX_FRAMES,
                        FORCE_RUN_MAX_FRAMES]
    assert det.enabled

  def test_failure_retries_with_backoff(self):
    det, pm = make_detector(), PM()
    det.run.fail = True
    delays = []
    for _ in range(8):
      det.retry_at = 0.0  # the wait is over
      det.step(pm, det.model_state, 0.010, 0.0, log.Desire.none, METAS)
      delays.append(det.retry_at)
      for _ in range(3 * RUN_EVERY):  # nothing while it waits
        det.step(pm, det.model_state, 0.010, 0.0, log.Desire.none, METAS)
      assert pm.sent == [] and det.enabled
    waits = [b - a for a, b in zip(delays, delays[1:], strict=False)]
    assert [round(w) for w in waits[:5]] == [RETRY_S, 2 * RETRY_S, 4 * RETRY_S, 8 * RETRY_S, 16 * RETRY_S]
    assert waits[-1] <= RETRY_MAX_S + 1 and det.failures == 8
    det.run.fail, det.retry_at = False, 0.0
    det.step(pm, det.model_state, 0.010, 0.0, log.Desire.none, METAS)
    assert len(pm.sent) == 1 and det.failures == 0

  def test_host_follows_the_model(self):
    big, small = object(), object()
    loader, pm = FakeLoad({big, small}), PM()
    host = DetectorHost(FakeParams(True), 1344, 760, False, True, [big, small, small], load_fn=loader)
    host.load()
    assert loader.calls == [big, small]
    host.step(pm, big, 0.010, 0.0, log.Desire.none, METAS, True)
    host.step(pm, small, 0.010, 0.0, log.Desire.none, METAS, True)  # modeld fell back to the small model
    assert len(pm.sent) == 2
    big_det, small_det = (d for _, d in host.detectors)
    assert len(big_det.run.calls) == 1 and len(small_det.run.calls) == 1

  def test_host_without_a_detector_for_the_fallback(self):
    big, small = object(), object()
    pm = PM()
    host = DetectorHost(FakeParams(True), 1344, 760, False, True, [big, small], load_fn=FakeLoad({big}))
    host.load()
    for _ in range(4):
      host.step(pm, small, 0.010, 0.0, log.Desire.none, METAS, True)
    assert pm.sent == [] and host.missing is small

  def test_host_turned_on_mid_drive(self):
    model, params, pm = object(), FakeParams(False), PM()
    loader = FakeLoad({model})
    host = DetectorHost(params, 1344, 760, False, True, [model], load_fn=loader)
    assert not host.check_enabled(0.0)  # modeld's startup check: off, nothing loaded
    host.step(pm, model, 0.010, 0.0, log.Desire.none, METAS, False)
    assert loader.calls == [] and pm.sent == []
    params.enabled, host.next_check = True, 0.0  # turned on; read on the next check
    host.step(pm, model, 0.010, 0.0, log.Desire.none, METAS, True)
    assert loader.calls == []  # engaged: loading would stall the model, so wait
    host.step(pm, model, 0.010, 0.0, log.Desire.none, METAS, False)
    assert loader.calls == [model] and len(pm.sent) == 1
    params.enabled, host.next_check = False, 0.0  # turned off: loaded, but not run
    for _ in range(2 * RUN_EVERY):
      host.step(pm, model, 0.010, 0.0, log.Desire.none, METAS, False)
    assert len(pm.sent) == 1

  def test_host_never_raises_into_modeld(self):
    model, pm = object(), PM()
    host = DetectorHost(FakeParams(True), 1344, 760, False, True, [model], load_fn=FakeLoad({model}))
    host.load()
    det = host.detectors[0][1]
    det.room = None  # a bug outside the detector's own try
    host.step(pm, model, 0.010, 0.0, log.Desire.none, METAS, False)
    assert not host.enabled and host.next_check > 0.
