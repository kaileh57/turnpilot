"""turnpilotd's autonomous left standing at its line when its road view goes: the detector stopped for a while (no runs, the
scene unavailable STALE_S after the last), a stream in the band all along. A stop shorter than the daemon's own bound
(DETECTOR_DEADLINE_S) is waited through, nothing said; a longer one is the daemon's real loss and hands back
(detectorStopped), and once the camera is back the left judges again from where the car stands and goes on its gap, never
handed back for good. The Sim, route and road runs of test_turnpilotd_left."""
import pytest

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.turnpilot.hooks import TAKE_CONTROL_PROMPTS
from openpilot.selfdrive.turnpilot.scene.tracker import RawFrame
from openpilot.selfdrive.turnpilot.tests.test_turnpilotd_left import JUNCTION, car_frame, drive, left_sim, raw_box
from openpilot.selfdrive.turnpilot.turnpilotd import DETECTOR_DEADLINE_S
from openpilot.selfdrive.turnpilot.unprotected_left import CLEAR_WINDOW_S

Desire = log.Desire
STREAM_S = 6.  # s after the detector stops that the oncoming stream ends


def stopped_at_the_line(stop_s):
  """LEFT_25 standing at its line, a car always in the band; 2 s on the detector stops for stop_s (no road run, the scene
  unavailable from STALE_S after the last run); the stream ends STREAM_S after the stop began. Returns the plans from the
  stop on."""
  st = {'t0': None, 'n': 0}

  def raw(s):
    st['n'] += 1
    if st['n'] % 2 or (st['t0'] is not None and st['t0'] <= s.t < st['t0'] + stop_s):
      return None
    xs = [JUNCTION + 40. - (s.t * 12.) % 45.] if st['t0'] is None or s.t < st['t0'] + STREAM_S else []
    return RawFrame([raw_box(*car_frame(s, x - s.s, 3.6)) for x in xs if -5. < x - s.s < 400.], 'road', s.t)
  sim = left_sim()
  sim.raw = raw
  while not (sim.trace and str(sim.trace[-1][2].unprotectedLeft.state) == 'waitGap' and sim.v < 0.05):
    drive(sim, DT_MDL)
    assert sim.t < 90.
  drive(sim, 2.)
  st['t0'] = sim.t
  plans = []
  for _ in range(int(40. / DT_MDL)):
    gap = st['t0'] + 0.5 <= sim.t < st['t0'] + stop_s  # tracker.STALE_S after the last run, until the next
    turning = sim.tp.desires.started and abs(JUNCTION - sim.s) < 12. and sim.v > 0.3
    plans.append(sim.step(engaged=True, available=not gap, yaw_rate=sim.v / 8. if turning else 0.))
  assert sim.s > JUNCTION + 10.
  return plans


@pytest.mark.parametrize('stop_s', [1.0, 1.4, 2.0, 5.0])
def test_a_detector_stop_at_the_lefts_line_is_waited_through_or_handed_back_and_the_left_goes_again_once_it_is_back(stop_s):
  """Short of the daemon's bound (DETECTOR_DEADLINE_S from the last run) nothing is said. Past it: the daemon's
  detectorStopped handback, said while the detector is gone, never the left's own 'stale' latched after it (until
  2026-11-13 the left came back on with no road run since its reset and handed back 'Camera not seeing oncoming traffic'
  on that first frame, for good, the detector already back: the matrix's 2 s stop). Either way the left goes one window
  after the stream ends, by itself."""
  plans = stopped_at_the_line(stop_s)
  said = [(k * DT_MDL, str(p.controlPrompt), p.reason) for k, p in enumerate(plans) if str(p.controlPrompt) in TAKE_CONTROL_PROMPTS]
  if stop_s < DETECTOR_DEADLINE_S:
    assert not said, said[:2]
  else:
    assert said and {(c, r) for _, c, r in said} == {('unprotectedTurnTakeControl', 'detector stopped: take control')}, said[:2]
    assert max(t for t, _, _ in said) < stop_s + 0.2  # said only while the detector is gone
  assert not any(str(p.unprotectedLeft.state) == 'undecided' for p in plans[int((stop_s + 0.2) / DT_MDL):])
  go = next(k for k, p in enumerate(plans) if p.desire == Desire.turnLeft)
  assert go * DT_MDL < max(STREAM_S, stop_s) + CLEAR_WINDOW_S + 0.5, go * DT_MDL
