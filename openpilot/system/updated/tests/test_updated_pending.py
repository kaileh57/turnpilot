"""A fetched, finalized update stays pending through a drive and installs itself while the car is parked (updated.py:
keep_finalized, auto_install_ok, maybe_auto_install), at most once a commit, never onroad or with the ignition on.

updated runs only offroad (manager's process list), so a drive stops it and it starts again when the car is switched off;
the launch script installs at boot only with finalized/.overlay_consistent present."""
from pathlib import Path
from types import SimpleNamespace

import pytest

import openpilot.system.updated.updated as up
from openpilot.system.manager.process_config import managed_processes, only_offroad

NEW, OLD = "b" * 40, "a" * 40
PARKED = up.AUTO_INSTALL_PARKED_S


class Stop(BaseException):
  pass


class FakeParams:
  def __init__(self, **bools):
    self.bools, self.puts = bools, {}

  def get_bool(self, key, block=False):
    return self.bools.get(key, False)

  def get(self, key, block=False, return_default=False):
    return "set"

  def put(self, key, val, block=False):
    self.puts[key] = val

  def put_bool(self, key, val, block=False):
    self.puts[key] = val


class FakeUpdater:
  def __init__(self, ready=True, finalized=NEW, current=OLD):
    self.update_ready, self.commits = ready, {up.FINALIZED: finalized, up.BASEDIR: current}

  def get_commit_hash(self, path):
    return self.commits[path]


class FakeWatch(up.ParkedWatch):
  """The car as the test says: state = (offroad, ignition), each None for a service not alive."""
  def __init__(self, state=(True, False)):
    super().__init__(sm=object())
    self.state = state

  def read(self):
    return self.state


@pytest.fixture
def staging(tmp_path, monkeypatch):
  finalized = tmp_path / 'safe_staging' / 'finalized'
  finalized.mkdir(parents=True)
  monkeypatch.setattr(up, 'STAGING_ROOT', str(tmp_path / 'safe_staging'))
  monkeypatch.setattr(up, 'FINALIZED', str(finalized))
  monkeypatch.setattr(up, 'LOCK_FILE', str(tmp_path / 'lock'))
  monkeypatch.setattr(up, 'AUTO_INSTALL_FILE', str(tmp_path / 'auto_install_commit'))
  monkeypatch.setattr(up, 'overlay_valid', lambda: True)  # the overlay this boot's updater made, still mounted
  up.set_consistent_flag(True)
  return finalized


def start_updated(monkeypatch, overlay_ok=True, finalized=NEW, current=OLD):
  """updated.main() up to its loop, as manager starts it when the car is switched off after a drive."""
  def first_loop(*a, **k):
    raise Stop  # main() reached its loop: everything before it has run
  commits = {up.FINALIZED: finalized, up.BASEDIR: current}

  def overlay_valid():
    if overlay_ok is None:
      raise RuntimeError("find failed")
    return overlay_ok
  monkeypatch.setattr(up, 'Params', FakeParams)
  monkeypatch.setattr(up, 'Updater', lambda: object())
  monkeypatch.setattr(up, 'ParkedWatch', lambda: None)
  monkeypatch.setattr(up, 'overlay_valid', overlay_valid)
  monkeypatch.setattr(up, 'run', lambda cmd, cwd=None: commits[cwd] + "\n")
  monkeypatch.setattr(up, 'init_overlay', first_loop)
  with pytest.raises(Stop):
    up.main()


def test_updated_is_stopped_by_a_drive_and_started_again_after_it():
  assert managed_processes['updated'].should_run is only_offroad


def test_a_downloaded_update_is_still_installable_after_a_drive(staging, monkeypatch):
  # the last parked session fetched and finalized an update (the fixture's flag): Settings > Software shows "install now"
  start_updated(monkeypatch)  # the car was driven and switched off: manager starts updated again, its overlay still mounted
  # what launch_chffrplus.sh needs at the next boot to install it
  assert (staging / '.overlay_consistent').is_file(), "the pending update was invalidated when updated started again"


@pytest.mark.parametrize("why, kw", [("the overlay is gone or BASEDIR changed", {'overlay_ok': False}),
                                     ("the overlay can't be checked", {'overlay_ok': None}),
                                     ("the finalized tree is the running commit", {'finalized': OLD})])
def test_a_finalized_update_that_cant_be_vouched_for_is_invalidated_at_start_as_stock_does(staging, monkeypatch, why, kw):
  start_updated(monkeypatch, **kw)
  assert not (staging / '.overlay_consistent').is_file(), why


def test_the_launch_script_installs_only_a_consistent_finalized_update():
  script = (Path(up.BASEDIR) / 'launch_chffrplus.sh').read_text()
  assert 'finalized/.overlay_consistent' in script


def test_the_install_decision():
  ok = up.auto_install_ok
  assert ok(True, NEW, None, True, False, PARKED)                # ready, parked long enough, not tried: install
  assert ok(True, NEW, OLD, True, False, PARKED + 600., current=OLD)  # another commit was tried before: this one is new
  assert not ok(False, NEW, None, True, False, PARKED)           # nothing ready
  assert not ok(True, NEW, None, False, False, PARKED)           # onroad: never
  assert not ok(True, NEW, None, True, True, PARKED)             # the ignition on: a drive could be starting
  assert not ok(True, NEW, None, None, False, PARKED)            # not known offroad
  assert not ok(True, NEW, None, True, None, PARKED)             # the ignition not known
  assert not ok(True, NEW, None, True, False, PARKED - 1.)       # not parked long enough yet
  assert not ok(True, NEW, NEW, True, False, PARKED)             # already tried for this commit: never again
  assert not ok(True, NEW, None, True, False, PARKED, current=NEW)  # the running tree already is it
  assert not ok(True, None, None, True, False, PARKED) and not ok(True, "", None, True, False, PARKED)
  assert not ok(True, NEW, None, True, False, PARKED, lockout=True)  # a restart would clear the driver monitoring lockout


def test_a_parked_car_installs_once_and_a_failed_install_is_never_tried_again(staging):
  params, updater, watch = FakeParams(), FakeUpdater(), FakeWatch()
  assert not up.maybe_auto_install(params, updater, watch, now=0.)
  assert not up.maybe_auto_install(params, updater, watch, now=PARKED - 10.) and not params.puts
  assert up.maybe_auto_install(params, updater, watch, now=PARKED)
  # exactly what the Software screen's install button puts, with its line on the screen
  assert params.puts == {"UpdaterState": up.AUTO_INSTALL_STATE, "DoReboot": True}
  assert up.read_attempted() == NEW
  # the comma came back without the update installed (the launch script refused it) and fetched the same commit again
  params, watch = FakeParams(), FakeWatch()
  for now in (0., PARKED, 10. * PARKED):
    assert not up.maybe_auto_install(params, updater, watch, now=now)
  assert not params.puts
  # the next push is a new commit: tried once more
  later, watch = FakeUpdater(finalized="c" * 40), FakeWatch()
  assert not up.maybe_auto_install(params, later, watch, now=0.)
  assert up.maybe_auto_install(params, later, watch, now=PARKED)
  assert params.puts.get("DoReboot") is True and up.read_attempted() == "c" * 40


def test_a_drive_in_between_keeps_it_pending_and_it_installs_once_parked_again(staging):
  params, updater, watch = FakeParams(), FakeUpdater(), FakeWatch()
  assert not up.maybe_auto_install(params, updater, watch, now=0.)
  watch.state = (True, True)  # the driver gets in before the wait is over: ignition
  assert not up.maybe_auto_install(params, updater, watch, now=PARKED)
  watch.state = (False, True)  # driving (updated is stopped then; were it asked, the answer is no)
  assert not up.maybe_auto_install(params, updater, watch, now=PARKED + 600.)
  assert not params.puts and up.read_attempted() is None  # nothing tried, nothing spent: still pending
  # parked again: updated starts anew (a fresh watch), and the wait runs from the car being parked, not from before the drive
  watch = FakeWatch()
  assert not up.maybe_auto_install(params, updater, watch, now=2000.)
  assert not up.maybe_auto_install(params, updater, watch, now=2000. + PARKED - 10.)
  assert up.maybe_auto_install(params, updater, watch, now=2000. + PARKED)
  assert params.puts.get("DoReboot") is True and up.read_attempted() == NEW


def test_nothing_is_installed_on_an_unknown_car_a_lockout_or_a_failing_check(staging):
  for state in ((None, False), (True, None), (None, None)):
    params, watch = FakeParams(), FakeWatch(state)
    assert not up.maybe_auto_install(params, FakeUpdater(), watch, now=0.)
    assert not up.maybe_auto_install(params, FakeUpdater(), watch, now=10. * PARKED) and not params.puts
  params, watch = FakeParams(DriverTooDistracted=True), FakeWatch()
  up.maybe_auto_install(params, FakeUpdater(), watch, now=0.)
  assert not up.maybe_auto_install(params, FakeUpdater(), watch, now=PARKED) and not params.puts
  # the ignition comes on between the decision and the restart: the last look refuses it
  class Arriving(FakeWatch):
    looks = 0

    def read(self):
      self.looks += 1
      return (True, False) if self.looks <= 2 else (True, True)
  params, watch = FakeParams(), Arriving()
  up.maybe_auto_install(params, FakeUpdater(), watch, now=0.)
  assert not up.maybe_auto_install(params, FakeUpdater(), watch, now=PARKED)
  assert not params.puts and up.read_attempted() is None
  # an updater that can't answer starts nothing
  broken = SimpleNamespace(update_ready=True, get_commit_hash=lambda path: 1 / 0)
  watch = FakeWatch()
  up.maybe_auto_install(params, broken, watch, now=0.)
  assert not up.maybe_auto_install(params, broken, watch, now=PARKED) and not params.puts


def test_parked_watch_reads_offroad_and_ignition_and_unknown_while_a_service_is_not_alive():
  class SM(dict):
    alive = {'deviceState': True, 'pandaStates': True}

    def update(self, timeout):
      pass
  panda = SimpleNamespace(ignitionLine=False, ignitionCan=False)
  sm = SM(deviceState=SimpleNamespace(started=False), pandaStates=[panda])
  watch = up.ParkedWatch(sm)
  assert watch.read() == (True, False)
  panda.ignitionCan = True
  assert watch.read() == (True, True)
  sm['deviceState'].started = True
  assert watch.read() == (False, True)
  sm.alive = {'deviceState': False, 'pandaStates': False}
  assert watch.read() == (None, None)
  # the parked time runs only over looks that all read parked
  assert watch.parked_s(True, False, 100.) == 0. and watch.parked_s(True, False, 160.) == 60.
  assert watch.parked_s(True, None, 170.) == 0. and watch.parked_s(True, False, 180.) == 0.


def test_a_finalized_update_that_is_the_branchs_tip_is_not_fetched_again(staging, monkeypatch):
  monkeypatch.setattr(up, 'Params', FakeParams)
  u = up.Updater()
  heads = {up.BASEDIR: OLD, up.FINALIZED: NEW}
  monkeypatch.setattr(up.Updater, 'target_branch', property(lambda self: 'turnpilot'))
  monkeypatch.setattr(up.Updater, 'get_branch', lambda self, path: 'turnpilot')
  monkeypatch.setattr(up.Updater, 'get_commit_hash', lambda self, path=None: heads[path])
  u.branches['turnpilot'] = NEW
  up.set_consistent_flag(False)
  assert not u.finalized_current  # nothing finalized yet
  up.set_consistent_flag(True)
  assert u.update_ready and u.finalized_current
  u.branches['turnpilot'] = "c" * 40  # a newer push: fetched
  assert not u.finalized_current
  u.branches.clear()  # the remote not read: no claim
  assert not u.finalized_current


def test_idle_looks_at_the_car_while_it_waits_and_a_request_ends_it(monkeypatch):
  import threading
  wait = SimpleNamespace(ready_event=threading.Event())
  wait.sleep = lambda t: wait.ready_event.wait(timeout=0.)
  clock = iter(range(0, 10000, 10))
  monkeypatch.setattr(up.time, 'monotonic', lambda: float(next(clock)))
  looks = []
  up.idle(wait, 60., lambda: looks.append(1))
  assert 4 <= len(looks) <= 6  # every AUTO_INSTALL_POLL_S of the minute
  wait.ready_event.set()
  looks.clear()
  up.idle(wait, 60., lambda: looks.append(1))
  assert not looks


def test_no_install_is_started_or_spent_on_a_boot_that_would_not_install(staging, monkeypatch):
  """A drive that ends away from Wi-Fi: the update is ready and the car parked, but the launch script would refuse it at the
  boot (its init file gone, BASEDIR changed, the finalized tree not consistent, or an earlier swap's backup in the way). No
  restart then, and the commit is not recorded as tried: it installs once the conditions hold."""
  def parked_long(**kw):
    params, watch = FakeParams(), FakeWatch()
    up.maybe_auto_install(params, FakeUpdater(), watch, now=0.)
    return up.maybe_auto_install(params, FakeUpdater(), watch, now=PARKED), params
  monkeypatch.setattr(up, 'overlay_valid', lambda: False)
  started, params = parked_long()
  assert not started and not params.puts and up.read_attempted() is None
  monkeypatch.setattr(up, 'overlay_valid', lambda: True)
  (staging.parent / 'old_openpilot').mkdir()
  started, params = parked_long()
  assert not started and not params.puts and up.read_attempted() is None
  (staging.parent / 'old_openpilot').rmdir()
  up.set_consistent_flag(False)
  assert not up.launch_would_install()
  up.set_consistent_flag(True)
  started, params = parked_long()
  assert started and params.puts.get("DoReboot") is True and up.read_attempted() == NEW


def test_a_check_that_fails_keeps_the_overlay_of_a_pending_update_and_a_failed_fetch_does_not(staging, monkeypatch):
  """updated's loop unlinks BASEDIR's init file after any failure, so the next loop rebuilds the overlay and wipes the staging
  area, and until then the launch script installs nothing. Parked away from Wi-Fi the check fails every time: the finalized
  update the drive kept was thrown away five minutes after arriving. A failure before the fetch began leaves a pending
  update's overlay alone; one in the fetch, or with nothing pending, is stock's."""
  commits = {up.FINALIZED: NEW, up.BASEDIR: OLD}
  monkeypatch.setattr(up, 'run', lambda cmd, cwd=None: commits[cwd] + "\n")
  assert up.keep_overlay_after_failure(fetching=False)
  assert not up.keep_overlay_after_failure(fetching=True)       # the fetch may have left the overlay half done
  up.set_consistent_flag(False)
  assert not up.keep_overlay_after_failure(fetching=False)      # nothing pending
  up.set_consistent_flag(True)
  monkeypatch.setattr(up, 'overlay_valid', lambda: False)
  assert not up.keep_overlay_after_failure(fetching=False)      # the overlay itself is what failed


def test_the_loop_keeps_a_pending_update_through_an_offline_check(staging, monkeypatch, tmp_path):
  import subprocess
  init = tmp_path / 'overlay_init'
  init.touch()
  commits = {up.FINALIZED: NEW, up.BASEDIR: OLD}
  loops = []

  class Offline:
    update_ready = True

    def set_params(self, *a):
      pass

    def check_for_update(self):
      raise subprocess.CalledProcessError(128, ['git', 'ls-remote', '--heads'], output="could not resolve host")

    def get_commit_hash(self, path):
      return commits[path]

  def idle(wait_helper, t, look):
    loops.append(t)
    if len(loops) >= 3:
      raise Stop
  monkeypatch.setattr(up, 'OVERLAY_INIT', init)
  monkeypatch.setattr(up, 'Params', FakeParams)
  monkeypatch.setattr(up, 'Updater', Offline)
  monkeypatch.setattr(up, 'ParkedWatch', lambda: None)
  monkeypatch.setattr(up, 'WaitTimeHelper', lambda: SimpleNamespace(ready_event=SimpleNamespace(clear=lambda: None), user_request=0))
  monkeypatch.setattr(up, 'run', lambda cmd, cwd=None: commits[cwd] + "\n")
  monkeypatch.setattr(up, 'init_overlay', lambda: None)
  monkeypatch.setattr(up, 'system_time_valid', lambda: True)
  monkeypatch.setattr(up, 'idle', idle)
  with pytest.raises(Stop):
    up.main()
  assert loops == [60, 5 * 60, 5 * 60]  # the first minute, then two failed checks
  assert init.is_file() and (staging / '.overlay_consistent').is_file()  # still what the launch script installs at a boot
  # nothing pending: a failed check recreates the overlay, as stock
  loops.clear()
  commits[up.FINALIZED] = OLD
  with pytest.raises(Stop):
    up.main()
  assert not init.is_file()
