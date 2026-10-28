"""The tiles restart_stack.sh starts navd on: the signals tiles by default, so stop control is there for every drive. Without
them built it refuses before it touches the running stack, rather than routing on tiles with no stops; tiles asked for
by name run, and when they carry no controls it says so (navd then reports them unsupported, which TurnPilot shows)."""
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SIM_TILES = HERE / 'sim_tiles.sh'
RESTART = HERE / 'restart_stack.sh'


def sim_tiles(home, tiles=None):
  env = {k: v for k, v in os.environ.items() if k != 'TURNPILOT_TILES'} | {'HOME': str(home)}
  if tiles is not None:
    env['TURNPILOT_TILES'] = str(tiles)
  return subprocess.run(['bash', str(SIM_TILES)], env=env, capture_output=True, text=True, timeout=10)


def test_default_is_the_signals_tiles(tmp_path):
  signals = tmp_path / 'turnpilot_data/valhalla/west_coast_usa_signals'
  signals.mkdir(parents=True)
  (signals / 'controls_built').touch()
  r = sim_tiles(tmp_path)
  assert r.returncode == 0 and r.stdout.strip() == str(signals) and r.stderr == ''


def test_signals_tiles_not_built_is_refused(tmp_path):
  (tmp_path / 'turnpilot_data/valhalla/west_coast_usa').mkdir(parents=True)  # only the plain tiles
  r = sim_tiles(tmp_path)
  assert r.returncode != 0 and r.stdout == '' and 'build_sim_tiles.sh' in r.stderr
  signals = tmp_path / 'turnpilot_data/valhalla/west_coast_usa_signals'
  signals.mkdir(parents=True)  # built, but not finished: no marker
  r = sim_tiles(tmp_path)
  assert r.returncode != 0 and r.stdout == ''


def test_tiles_asked_for_run_and_say_when_they_have_no_controls(tmp_path):
  plain = tmp_path / 'turnpilot_data/valhalla/west_coast_usa'
  plain.mkdir(parents=True)
  r = sim_tiles(tmp_path, plain)
  assert r.returncode == 0 and r.stdout.strip() == str(plain) and 'no controls' in r.stderr
  (plain / 'controls_built').touch()
  r = sim_tiles(tmp_path, plain)
  assert r.returncode == 0 and r.stdout.strip() == str(plain) and r.stderr == ''


def test_restart_stack_picks_the_tiles_before_it_stops_anything():
  text = RESTART.read_text()
  assert 'sim_tiles.sh' in text and text.index('sim_tiles.sh') < text.index('pkill')
  assert 'valhalla/west_coast_usa}' not in text  # no plain-tiles default of its own


def test_recorders_stop_when_the_stack_restart_fails():
  for name in ('record_scenario.sh', 'record_route.sh'):
    text = (HERE / name).read_text()
    assert 'restart_stack.sh failed' in text and '> /dev/null 2>&1\n' not in text.split('restart_stack.sh"')[1].split('\n')[0], name
