#!/usr/bin/env python3
import os
import re
import datetime
import subprocess
import shutil
import signal
import fcntl
import time
import threading
from collections import defaultdict
from pathlib import Path

from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.time_helpers import system_time_valid
from openpilot.common.markdown import parse_markdown
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from openpilot.common.hardware import AGNOS, HARDWARE
from openpilot.common.version import get_build_metadata

LOCK_FILE = os.getenv("UPDATER_LOCK_FILE", "/tmp/safe_staging_overlay.lock")
STAGING_ROOT = os.getenv("UPDATER_STAGING_ROOT", "/data/safe_staging")

OVERLAY_UPPER = os.path.join(STAGING_ROOT, "upper")
OVERLAY_METADATA = os.path.join(STAGING_ROOT, "metadata")
OVERLAY_MERGED = os.path.join(STAGING_ROOT, "merged")
FINALIZED = os.path.join(STAGING_ROOT, "finalized")

OVERLAY_INIT = Path(os.path.join(BASEDIR, ".overlay_init"))

# TurnPilot: a finalized update installs itself while the car is parked (auto_install_ok). The commit an install was last
# tried for is kept here, outside the staging area (init_overlay wipes that after every boot), so a failed install is tried
# once and then left to the Software screen's button
AUTO_INSTALL_FILE = os.getenv("UPDATER_AUTO_INSTALL_FILE", "/data/updater_auto_install_commit")
AUTO_INSTALL_PARKED_S = 3 * 60.  # s offroad with the ignition off, seen on every poll, before it installs
AUTO_INSTALL_POLL_S = 10.        # s between looks at the car while idle
AUTO_INSTALL_RESTART_S = 120.    # s the updater stands by for manager's restart after starting an install
AUTO_INSTALL_STATE = "installing update..."  # UpdaterState, the Software screen's line, while the comma restarts into it

# do not allow to engage after this many hours onroad and this many routes
HOURS_NO_CONNECTIVITY_MAX = 27
ROUTES_NO_CONNECTIVITY_MAX = 84
# send an offroad prompt after this many hours onroad and this many routes
HOURS_NO_CONNECTIVITY_PROMPT = 23
ROUTES_NO_CONNECTIVITY_PROMPT = 80


class UserRequest:
  NONE = 0
  CHECK = 1
  FETCH = 2

class WaitTimeHelper:
  def __init__(self):
    self.ready_event = threading.Event()
    self.user_request = UserRequest.NONE
    signal.signal(signal.SIGHUP, self.update_now)
    signal.signal(signal.SIGUSR1, self.check_now)

  def update_now(self, signum: int, frame) -> None:
    cloudlog.info("caught SIGHUP, attempting to downloading update")
    self.user_request = UserRequest.FETCH
    self.ready_event.set()

  def check_now(self, signum: int, frame) -> None:
    cloudlog.info("caught SIGUSR1, checking for updates")
    self.user_request = UserRequest.CHECK
    self.ready_event.set()

  def sleep(self, t: float) -> None:
    self.ready_event.wait(timeout=t)

def write_time_to_param(params, param) -> None:
  t = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
  params.put(param, t, block=True)

def run(cmd: list[str], cwd: str | None = None) -> str:
  return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, encoding='utf8')


def set_consistent_flag(consistent: bool) -> None:
  os.sync()
  consistent_file = Path(os.path.join(FINALIZED, ".overlay_consistent"))
  if consistent:
    consistent_file.touch()
  elif not consistent:
    consistent_file.unlink(missing_ok=True)
  os.sync()

def parse_release_notes(basedir: str) -> bytes:
  try:
    with open(os.path.join(basedir, "RELEASES.md"), "rb") as f:
      r = f.read().split(b'\n\n', 1)[0]  # Slice latest release notes
    try:
      return bytes(parse_markdown(r.decode("utf-8")), encoding="utf-8")
    except Exception:
      return r + b"\n"
  except FileNotFoundError:
    pass
  except Exception:
    cloudlog.exception("failed to parse release notes")
  return b""

def setup_git_options(cwd: str) -> None:
  # We sync FS object atimes (which NEOS doesn't use) and mtimes, but ctimes
  # are outside user control. Make sure Git is set up to ignore system ctimes,
  # because they change when we make hard links during finalize. Otherwise,
  # there is a lot of unnecessary churn. This appears to be a common need on
  # OSX as well: https://www.git-tower.com/blog/make-git-rebase-safe-on-osx/

  # We are using copytree to copy the directory, which also changes
  # inode numbers. Ignore those changes too.

  # Set protocol to the new version (default after git 2.26) to reduce data
  # usage on git fetch --dry-run from about 400KB to 18KB.
  git_cfg = [
    ("core.trustctime", "false"),
    ("core.checkStat", "minimal"),
    ("protocol.version", "2"),
    ("gc.auto", "0"),
    ("gc.autoDetach", "false"),
  ]
  for option, value in git_cfg:
    run(["git", "config", option, value], cwd)


def dismount_overlay() -> None:
  if os.path.ismount(OVERLAY_MERGED):
    cloudlog.info("unmounting existing overlay")
    run(["sudo", "umount", "-l", OVERLAY_MERGED])


def overlay_valid() -> bool:
  """The overlay this updater made is still mounted and BASEDIR/.git has not changed since: what was fetched and finalized on
  it still applies to the tree it would replace."""
  if not (OVERLAY_INIT.is_file() and os.path.ismount(OVERLAY_MERGED)):
    return False
  git_dir_path = os.path.join(BASEDIR, ".git")
  new_files = run(["find", git_dir_path, "-newer", str(OVERLAY_INIT)])
  return not len(new_files.splitlines())


def keep_finalized() -> bool:
  """Whether a finalized update found at start stays installable: only one made on the overlay that is still valid (updated
  was stopped by a drive and started again, no boot in between). Anything else, or a check that fails, invalidates it as
  stock does: after a boot init_overlay rebuilds the staging area anyway."""
  try:
    if not Path(os.path.join(FINALIZED, ".overlay_consistent")).is_file() or not overlay_valid():
      return False
    # the same commit as the running tree is no update (a fetch finalizes whatever the branch's tip is)
    return run(["git", "rev-parse", "HEAD"], FINALIZED).rstrip() != run(["git", "rev-parse", "HEAD"], BASEDIR).rstrip()
  except Exception:
    cloudlog.exception("updated: finalized update check failed, invalidating it")
    return False


def launch_would_install() -> bool:
  """The launch script's own conditions for swapping the finalized update in at the boot hold now: BASEDIR's init file there
  with nothing in BASEDIR/.git newer (overlay_valid, which asks the mount too), the finalized tree consistent, and no backup
  of an earlier swap in the way. A restart without them installs nothing."""
  return overlay_valid() and Path(os.path.join(FINALIZED, ".overlay_consistent")).is_file() and \
    not os.path.isdir(os.path.join(STAGING_ROOT, "old_openpilot"))


def keep_overlay_after_failure(fetching: bool) -> bool:
  """A loop that failed leaves the overlay as it is (stock recreates it, which wipes the staging area) only when nothing was
  fetched into it this loop and a finalized update is pending on it: a check that fails away from Wi-Fi is no reason to
  throw a finished download away. A failure once the fetch began, or with nothing pending, is stock's."""
  return not fetching and keep_finalized()


def auto_install_ok(update_ready: bool, commit: str | None, attempted: str | None, offroad: bool | None, ignition: bool | None,
                    parked_s: float, lockout: bool = False, current: str | None = None) -> bool:
  """Install the finalized update now, as the Software screen's install button would (DoReboot: the launch script swaps it
  in at the boot). Only with an update ready (Updater.update_ready, stock's own test), its commit known, not the running
  tree's (current) and not the one an install was already tried for, the device known offroad and the ignition known off for
  AUTO_INSTALL_PARKED_S, and no driver monitoring lockout a restart would clear. Anything unknown is a no."""
  if not update_ready or not commit or commit in (attempted, current) or lockout:
    return False
  return offroad is True and ignition is False and parked_s >= AUTO_INSTALL_PARKED_S


def read_attempted(path: str | None = None) -> str | None:
  try:
    return Path(path or AUTO_INSTALL_FILE).read_text().strip() or None
  except FileNotFoundError:
    return None


def write_attempted(commit: str, path: str | None = None) -> None:
  """commit recorded as tried, on disk before anything restarts."""
  path = path or AUTO_INSTALL_FILE
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    f.write(commit)
    f.flush()
    os.fsync(f.fileno())
  os.replace(tmp, path)
  os.sync()


class ParkedWatch:
  """Whether the car is parked, from deviceState and pandaStates: (offroad, ignition), each None while its service isn't
  alive, and for how long both have read parked on every look."""
  def __init__(self, sm=None):
    if sm is None:
      from openpilot.cereal import messaging
      sm = messaging.SubMaster(['deviceState', 'pandaStates'])
    self.sm = sm
    self.since: float | None = None

  def read(self) -> tuple[bool | None, bool | None]:
    self.sm.update(0)
    offroad = (not self.sm['deviceState'].started) if self.sm.alive['deviceState'] else None
    ignition = any(ps.ignitionLine or ps.ignitionCan for ps in self.sm['pandaStates']) if self.sm.alive['pandaStates'] else None
    return offroad, ignition

  def parked_s(self, offroad: bool | None, ignition: bool | None, now: float) -> float:
    if offroad is True and ignition is False:
      self.since = now if self.since is None else self.since
      return now - self.since
    self.since = None
    return 0.


def maybe_auto_install(params, updater, watch: ParkedWatch, now: float | None = None) -> bool:
  """One look at the car and the finalized update; True once the install was started (the commit recorded as tried, the
  Software screen's line set, DoReboot put: manager restarts the comma and the launch script installs). A failure anywhere
  starts nothing."""
  try:
    offroad, ignition = watch.read()
    parked_s = watch.parked_s(offroad, ignition, time.monotonic() if now is None else now)
    if parked_s < AUTO_INSTALL_PARKED_S or not updater.update_ready:
      return False
    commit = updater.get_commit_hash(FINALIZED)
    if not auto_install_ok(True, commit, read_attempted(), offroad, ignition, parked_s, params.get_bool("DriverTooDistracted"),
                           updater.get_commit_hash(BASEDIR)):
      return False
    offroad, ignition = watch.read()  # once more, right before it restarts
    if offroad is not True or ignition is not False or not launch_would_install():
      return False
    write_attempted(commit)
    cloudlog.event("updated.auto_install", commit=commit, parked_s=round(parked_s))
    params.put("UpdaterState", AUTO_INSTALL_STATE, block=True)
    params.put_bool("DoReboot", True, block=True)
    return True
  except Exception:
    cloudlog.exception("updated: auto install check failed")
    return False


def idle(wait_helper, t: float, look) -> None:
  """wait_helper.sleep(t), with a look at the car (look()) every AUTO_INSTALL_POLL_S; a request from the Software screen
  ends it at once, as it ends the sleep."""
  end = time.monotonic() + t
  while not wait_helper.ready_event.is_set():
    left = end - time.monotonic()
    if left <= 0.:
      return
    look()
    wait_helper.sleep(min(left, AUTO_INSTALL_POLL_S))


def init_overlay() -> None:

  # Re-create the overlay if BASEDIR/.git has changed since we created the overlay
  if OVERLAY_INIT.is_file() and os.path.ismount(OVERLAY_MERGED):
    if overlay_valid():
      # A valid overlay already exists
      return
    else:
      cloudlog.info(".git directory changed, recreating overlay")

  cloudlog.info("preparing new safe staging area")

  params = Params()
  params.put_bool("UpdateAvailable", False, block=True)
  set_consistent_flag(False)
  dismount_overlay()
  run(["sudo", "rm", "-rf", STAGING_ROOT])
  if os.path.isdir(STAGING_ROOT):
    shutil.rmtree(STAGING_ROOT)

  for dirname in [STAGING_ROOT, OVERLAY_UPPER, OVERLAY_METADATA, OVERLAY_MERGED]:
    os.mkdir(dirname, 0o755)

  if os.lstat(BASEDIR).st_dev != os.lstat(OVERLAY_MERGED).st_dev:
    raise RuntimeError("base and overlay merge directories are on different filesystems; not valid for overlay FS!")

  # Leave a timestamped canary in BASEDIR to check at startup. The device clock
  # should be correct by the time we get here. If the init file disappears, or
  # critical mtimes in BASEDIR are newer than .overlay_init, continue.sh can
  # assume that BASEDIR has used for local development or otherwise modified,
  # and skips the update activation attempt.
  consistent_file = Path(os.path.join(BASEDIR, ".overlay_consistent"))
  if consistent_file.is_file():
    consistent_file.unlink()
  OVERLAY_INIT.touch()

  os.sync()
  overlay_opts = f"lowerdir={BASEDIR},upperdir={OVERLAY_UPPER},workdir={OVERLAY_METADATA}"

  mount_cmd = ["mount", "-t", "overlay", "-o", overlay_opts, "none", OVERLAY_MERGED]
  run(["sudo"] + mount_cmd)
  run(["sudo", "chmod", "755", os.path.join(OVERLAY_METADATA, "work")])

  git_diff = run(["git", "diff", "--submodule=diff"], OVERLAY_MERGED)
  params.put("GitDiff", git_diff, block=True)
  cloudlog.info(f"git diff output:\n{git_diff}")


def finalize_update() -> None:
  """Take the current OverlayFS merged view and finalize a copy outside of
  OverlayFS, ready to be swapped-in at BASEDIR. Copy using shutil.copytree"""

  # Remove the update ready flag and any old updates
  cloudlog.info("creating finalized version of the overlay")
  set_consistent_flag(False)

  # Copy the merged overlay view and set the update ready flag
  if os.path.exists(FINALIZED):
    shutil.rmtree(FINALIZED)
  shutil.copytree(OVERLAY_MERGED, FINALIZED, symlinks=True)

  run(["git", "reset", "--hard"], FINALIZED)
  run(["git", "submodule", "foreach", "--recursive", "git", "reset", "--hard"], FINALIZED)

  cloudlog.info("Starting git cleanup in finalized update")
  t = time.monotonic()
  try:
    run(["git", "gc"], FINALIZED)
    run(["git", "lfs", "prune"], FINALIZED)
    cloudlog.event("Done git cleanup", duration=time.monotonic() - t)
  except subprocess.CalledProcessError:
    cloudlog.exception(f"Failed git cleanup, took {time.monotonic() - t:.3f} s")

  set_consistent_flag(True)
  cloudlog.info("done finalizing overlay")


def handle_agnos_update() -> None:
  from openpilot.common.hardware.comma.agnos import flash_agnos_update, get_target_slot_number

  cur_version = HARDWARE.get_os_version()
  updated_version = run(["bash", "-c", r"unset AGNOS_VERSION && source launch_env.sh && \
                          echo -n $AGNOS_VERSION"], OVERLAY_MERGED).strip()

  cloudlog.info(f"AGNOS version check: {cur_version} vs {updated_version}")
  if cur_version == updated_version:
    return

  # prevent an openpilot getting swapped in with a mismatched or partially downloaded agnos
  set_consistent_flag(False)

  cloudlog.info(f"Beginning background installation for AGNOS {updated_version}")

  manifest_path = os.path.join(OVERLAY_MERGED, "openpilot/system/hardware/comma/agnos.json")
  if not os.path.isfile(manifest_path):  # releases before the openpilot/ move
    manifest_path = os.path.join(OVERLAY_MERGED, "system/hardware/tici/agnos.json")
  target_slot_number = get_target_slot_number()
  flash_agnos_update(manifest_path, target_slot_number, cloudlog)


class Updater:
  def __init__(self):
    self.params = Params()
    self.branches = defaultdict(str)
    self._has_internet: bool = False

  @property
  def has_internet(self) -> bool:
    return self._has_internet

  @property
  def target_branch(self) -> str:
    b: str | None = self.params.get("UpdaterTargetBranch")
    if b is None:
      b = self.get_branch(BASEDIR)
    b = {
      ("tizi", "release3"): "release-tizi",
      ("tizi", "release3-staging"): "release-tizi-staging",
      ("mici", "release3"): "release-mici",
      ("mici", "release3-staging"): "release-mici-staging",
    }.get((HARDWARE.get_device_type(), b), b)
    return b

  @property
  def update_ready(self) -> bool:
    consistent_file = Path(os.path.join(FINALIZED, ".overlay_consistent"))
    if consistent_file.is_file():
      hash_mismatch = self.get_commit_hash(BASEDIR) != self.branches[self.target_branch]
      branch_mismatch = self.get_branch(BASEDIR) != self.target_branch
      on_target_branch = self.get_branch(FINALIZED) == self.target_branch
      return ((hash_mismatch or branch_mismatch) and on_target_branch)
    return False

  @property
  def finalized_current(self) -> bool:
    """The finalized update is ready and is the target branch's tip as just checked: fetching again would only invalidate
    it for the minutes the copy takes, and a drive starting then would lose it."""
    tip = self.branches[self.target_branch] if len(self.branches) else ""
    return bool(tip) and self.update_ready and self.get_commit_hash(FINALIZED) == tip

  @property
  def update_available(self) -> bool:
    if os.path.isdir(OVERLAY_MERGED) and len(self.branches) > 0:
      hash_mismatch = self.get_commit_hash(OVERLAY_MERGED) != self.branches[self.target_branch]
      branch_mismatch = self.get_branch(OVERLAY_MERGED) != self.target_branch
      return hash_mismatch or branch_mismatch
    return False

  def get_branch(self, path: str) -> str:
    return run(["git", "rev-parse", "--abbrev-ref", "HEAD"], path).rstrip()

  def get_commit_hash(self, path: str = OVERLAY_MERGED) -> str:
    return run(["git", "rev-parse", "HEAD"], path).rstrip()

  def set_params(self, update_success: bool, failed_count: int, exception: str | None) -> None:
    self.params.put("UpdateFailedCount", failed_count, block=True)
    self.params.put("UpdaterTargetBranch", self.target_branch, block=True)

    self.params.put_bool("UpdaterFetchAvailable", self.update_available, block=True)
    if len(self.branches):
      self.params.put("UpdaterAvailableBranches", ','.join(self.branches.keys()), block=True)

    last_uptime_onroad = self.params.get("UptimeOnroad", return_default=True)
    last_route_count = self.params.get("RouteCount", return_default=True)
    if update_success:
      self.params.put("LastUpdateTime", datetime.datetime.now(datetime.UTC).replace(tzinfo=None), block=True)
      self.params.put("LastUpdateUptimeOnroad", last_uptime_onroad, block=True)
      self.params.put("LastUpdateRouteCount", last_route_count, block=True)
    else:
      last_uptime_onroad = self.params.get("LastUpdateUptimeOnroad", return_default=True)
      last_route_count = self.params.get("LastUpdateRouteCount", return_default=True)

    if exception is None:
      self.params.remove("LastUpdateException")
    else:
      self.params.put("LastUpdateException", exception, block=True)

    # Write out current and new version info
    def get_description(basedir: str) -> str:
      if not os.path.exists(basedir):
        return ""

      version = ""
      branch = ""
      commit = ""
      commit_date = ""
      try:
        branch = self.get_branch(basedir)
        commit = self.get_commit_hash(basedir)[:7]
        with open(os.path.join(basedir, "openpilot", "common", "version.h")) as f:
          version = f.read().split('"')[1]

        commit_unix_ts = run(["git", "show", "-s", "--format=%ct", "HEAD"], basedir).rstrip()
        dt = datetime.datetime.fromtimestamp(int(commit_unix_ts))
        commit_date = dt.strftime("%b %d")
      except Exception:
        cloudlog.exception("updater.get_description")
      return f"{version} / {branch} / {commit} / {commit_date}"
    self.params.put("UpdaterCurrentDescription", get_description(BASEDIR), block=True)
    self.params.put("UpdaterCurrentReleaseNotes", parse_release_notes(BASEDIR), block=True)
    self.params.put("UpdaterNewDescription", get_description(FINALIZED), block=True)
    self.params.put("UpdaterNewReleaseNotes", parse_release_notes(FINALIZED), block=True)
    self.params.put_bool("UpdateAvailable", self.update_ready, block=True)

    # Handle user prompt
    for alert in ("Offroad_UpdateFailed", "Offroad_ConnectivityNeeded", "Offroad_ConnectivityNeededPrompt"):
      set_offroad_alert(alert, False)

    dt_uptime_onroad = (self.params.get("UptimeOnroad", return_default=True) - last_uptime_onroad) / (60*60)
    dt_route_count = self.params.get("RouteCount", return_default=True) - last_route_count
    build_metadata = get_build_metadata()
    if failed_count > 15 and exception is not None and self.has_internet:
      if build_metadata.tested_channel:
        extra_text = "Ensure the software is correctly installed. Uninstall and re-install if this error persists."
      else:
        extra_text = exception
      set_offroad_alert("Offroad_UpdateFailed", True, extra_text=extra_text)
    elif failed_count > 0:
      if dt_uptime_onroad > HOURS_NO_CONNECTIVITY_MAX and dt_route_count > ROUTES_NO_CONNECTIVITY_MAX:
        set_offroad_alert("Offroad_ConnectivityNeeded", True)
      elif dt_uptime_onroad > HOURS_NO_CONNECTIVITY_PROMPT and dt_route_count > ROUTES_NO_CONNECTIVITY_PROMPT:
        remaining = max(HOURS_NO_CONNECTIVITY_MAX - dt_uptime_onroad, 1)
        set_offroad_alert("Offroad_ConnectivityNeededPrompt", True, extra_text=f"{remaining} hour{'' if remaining == 1 else 's'}.")

  def check_for_update(self) -> None:
    cloudlog.info("checking for updates")

    excluded_branches = ('release2', 'release2-staging')

    try:
      run(["git", "ls-remote", "origin", "HEAD"], OVERLAY_MERGED)
      self._has_internet = True
    except subprocess.CalledProcessError:
      self._has_internet = False

    setup_git_options(OVERLAY_MERGED)
    output = run(["git", "ls-remote", "--heads"], OVERLAY_MERGED)

    self.branches.clear()
    for line in output.split('\n'):
      ls_remotes_re = r'(?P<commit_sha>\b[0-9a-f]{5,40}\b)(\s+)(refs\/heads\/)(?P<branch_name>.*$)'
      x = re.fullmatch(ls_remotes_re, line.strip())
      if x is not None and x.group('branch_name') not in excluded_branches:
        self.branches[x.group('branch_name')] = x.group('commit_sha')

    cur_branch = self.get_branch(OVERLAY_MERGED)
    cur_commit = self.get_commit_hash(OVERLAY_MERGED)
    new_branch = self.target_branch
    new_commit = self.branches[new_branch]
    if (cur_branch, cur_commit) != (new_branch, new_commit):
      cloudlog.info(f"update available, {cur_branch} ({str(cur_commit)[:7]}) -> {new_branch} ({str(new_commit)[:7]})")
    else:
      cloudlog.info(f"up to date on {cur_branch} ({str(cur_commit)[:7]})")

  def fetch_update(self) -> None:
    cloudlog.info("attempting git fetch inside staging overlay")

    self.params.put("UpdaterState", "downloading...", block=True)

    # TODO: cleanly interrupt this and invalidate old update
    set_consistent_flag(False)
    self.params.put_bool("UpdateAvailable", False, block=True)

    setup_git_options(OVERLAY_MERGED)

    run(["git", "config", "--replace-all", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*"], OVERLAY_MERGED)

    branch = self.target_branch
    git_fetch_output = run(["git", "fetch", "origin", branch], OVERLAY_MERGED)
    cloudlog.info("git fetch success: %s", git_fetch_output)

    cloudlog.info("git reset in progress")
    cmds = [
      ["git", "checkout", "--force", "--no-recurse-submodules", "-B", branch, "FETCH_HEAD"],
      ["git", "branch", "--set-upstream-to", f"origin/{branch}"],
      ["git", "reset", "--hard"],
      ["git", "clean", "-xdff"],
      ["git", "submodule", "sync"],
      ["git", "submodule", "update", "--init", "--recursive"],
      ["git", "submodule", "foreach", "--recursive", "git", "reset", "--hard"],
    ]
    r = [run(cmd, OVERLAY_MERGED) for cmd in cmds]
    cloudlog.info("git reset success: %s", '\n'.join(r))

    # TODO: show agnos download progress
    if AGNOS:
      handle_agnos_update()

    # Create the finalized, ready-to-swap update
    self.params.put("UpdaterState", "finalizing update...", block=True)
    finalize_update()
    cloudlog.info("finalize success!")


def main() -> None:
  params = Params()

  if params.get_bool("DisableUpdates"):
    cloudlog.warning("updates are disabled by the DisableUpdates param")
    exit(0)

  with open(LOCK_FILE, 'w') as ov_lock_fd:
    try:
      fcntl.flock(ov_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
      raise RuntimeError("couldn't get overlay lock; is another instance running?") from e

    # Check if we just performed an update
    if Path(os.path.join(STAGING_ROOT, "old_openpilot")).is_dir():
      cloudlog.event("update installed")

    if not params.get("InstallDate"):
      t = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
      params.put("InstallDate", t, block=True)

    updater = Updater()
    update_failed_count = 0 # TODO: Load from param?
    wait_helper = WaitTimeHelper()

    # invalidate old finalized update, but for one this boot's updater finalized before a drive stopped it: that one is
    # still what the Software screen offered, and stays pending
    if not keep_finalized():
      set_consistent_flag(False)
    watch = ParkedWatch()

    def look() -> bool:
      if maybe_auto_install(params, updater, watch):
        # manager restarts the comma on DoReboot: nothing is fetched or invalidated meanwhile
        time.sleep(AUTO_INSTALL_RESTART_S)
        cloudlog.error("updated: the comma did not restart for the install")
      return False

    # set initial state
    params.put("UpdaterState", "idle", block=True)

    # Run the update loop
    first_run = True
    while True:
      wait_helper.ready_event.clear()

      # Attempt an update
      exception = None
      fetching = False
      try:
        # TODO: reuse overlay from previous updated instance if it looks clean
        init_overlay()

        # ensure we have some params written soon after startup
        updater.set_params(False, update_failed_count, exception)

        if not system_time_valid() or first_run:
          first_run = False
          idle(wait_helper, 60, look)
          continue

        update_failed_count += 1

        # check for update
        params.put("UpdaterState", "checking...", block=True)
        updater.check_for_update()

        # download update
        last_fetch = params.get("UpdaterLastFetchTime")
        timed_out = last_fetch is None or (datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - last_fetch > datetime.timedelta(days=3))
        user_requested_fetch = wait_helper.user_request == UserRequest.FETCH
        if params.get_bool("NetworkMetered") and not timed_out and not user_requested_fetch:
          cloudlog.info("skipping fetch, connection metered")
        elif wait_helper.user_request == UserRequest.CHECK:
          cloudlog.info("skipping fetch, only checking")
        elif not user_requested_fetch and updater.finalized_current:
          cloudlog.info("skipping fetch, the finalized update is the branch's tip")
        else:
          fetching = True
          updater.fetch_update()
          write_time_to_param(params, "UpdaterLastFetchTime")
        update_failed_count = 0
      except subprocess.CalledProcessError as e:
        cloudlog.event(
          "update process failed",
          cmd=e.cmd,
          output=e.output,
          returncode=e.returncode
        )
        exception = f"command failed: {e.cmd}\n{e.output}"
        if not keep_overlay_after_failure(fetching):
          OVERLAY_INIT.unlink(missing_ok=True)
      except Exception as e:
        cloudlog.exception("uncaught updated exception, shouldn't happen")
        exception = str(e)
        if not keep_overlay_after_failure(fetching):
          OVERLAY_INIT.unlink(missing_ok=True)

      try:
        params.put("UpdaterState", "idle", block=True)
        update_successful = (update_failed_count == 0)
        updater.set_params(update_successful, update_failed_count, exception)
      except Exception:
        cloudlog.exception("uncaught updated exception while setting params, shouldn't happen")

      # infrequent attempts if we successfully updated recently
      wait_helper.user_request = UserRequest.NONE
      idle(wait_helper, 5*60 if update_failed_count > 0 else 1.5*60*60, look)


if __name__ == "__main__":
  main()
