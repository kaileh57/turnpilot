"""A corridor pack from /v1/corridor: verified against its manifest, unpacked to its own directory, old ones pruned.

The pack is a tar of manifest.json and a Valhalla tile hierarchy under tiles/. Every member is checked before anything
is written: regular files and directories only, relative paths with no '..', nothing outside manifest.json and tiles/,
and exactly the files the manifest lists, each with its SHA-256. Any problem rejects the whole pack.
"""
import hashlib
import json
import shutil
import tarfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from openpilot.selfdrive.turnpilot.maps.client import MapError

MANIFEST = 'manifest.json'
TILES = 'tiles'
KEEP_PACKS = 3
STALE_DOWNLOAD_S = 3600.  # a download this old was cut off by a crash or a restart
MAX_MANIFEST = 16 * 1024 * 1024
CHUNK = 1024 * 1024


@dataclass(frozen=True)
class Pack:
  path: Path
  manifest: dict

  @property
  def tiles(self) -> Path:
    return self.path / TILES


def bad_pack(detail: str) -> MapError:
  return MapError('bad_pack', f"The downloaded map failed its check ({detail}), so it was thrown away.", 'Downloaded map failed its check')


def member_path(name: str) -> str:
  """The member's path as the manifest names it, or a MapError for anything that could land outside the pack."""
  if not name or name.startswith('/') or '\\' in name or '\x00' in name:
    raise bad_pack(f"unsafe path {name!r}")
  parts = PurePosixPath(name).parts
  if parts and parts[0] == '.':
    parts = parts[1:]
  if not parts or any(p in ('..', '') for p in parts) or ':' in parts[0]:
    raise bad_pack(f"unsafe path {name!r}")
  path = '/'.join(parts)
  if path != MANIFEST and parts[0] != TILES:
    raise bad_pack(f"unexpected file {path!r}")
  return path


def read_manifest(tar: tarfile.TarFile, member: tarfile.TarInfo) -> dict:
  if member.size > MAX_MANIFEST:
    raise bad_pack("manifest too large")
  f = tar.extractfile(member)
  try:
    manifest = json.loads(f.read()) if f is not None else None
  except ValueError:
    manifest = None
  if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), dict) or not manifest['files']:
    raise bad_pack("manifest is not valid")
  if not isinstance(manifest.get('region'), str) or not isinstance(manifest.get('version'), str):
    raise bad_pack("manifest has no region or version")
  for name, digest in manifest['files'].items():
    if member_path(name) != name or name == MANIFEST or not name.startswith(TILES + '/'):
      raise bad_pack(f"manifest lists {name!r}")
    if not (isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest.lower())):
      raise bad_pack(f"manifest checksum for {name!r} is not SHA-256")
  return manifest


def unpack(tar_path: Path, packs_dir: Path) -> Pack:
  """Verify the tar at tar_path and unpack it to a new directory under packs_dir."""
  packs_dir.mkdir(parents=True, exist_ok=True)
  try:
    tar = tarfile.open(tar_path, 'r:*')
  except (tarfile.TarError, OSError) as e:
    raise bad_pack(f"not a tar file: {e}") from None
  staging = packs_dir / f".staging-{uuid.uuid4().hex[:8]}"
  try:
    with tar:
      members, files, manifest_member = {}, {}, None
      for member in tar.getmembers():
        path = member_path(member.name)
        if path in members:
          raise bad_pack(f"{path!r} appears twice")
        members[path] = member
        if member.isdir():
          if path == MANIFEST:
            raise bad_pack("manifest is a directory")
          continue
        if not member.isreg():
          raise bad_pack(f"{path!r} is not a regular file")
        if path == MANIFEST:
          manifest_member = member
        else:
          files[path] = member
      if manifest_member is None:
        raise bad_pack("no manifest")
      manifest = read_manifest(tar, manifest_member)
      listed = set(manifest['files'])
      if missing := sorted(listed - set(files)):
        raise bad_pack(f"missing {missing[0]}" + (f" and {len(missing) - 1} more" if len(missing) > 1 else ''))
      if extra := sorted(set(files) - listed):
        raise bad_pack(f"unlisted file {extra[0]}" + (f" and {len(extra) - 1} more" if len(extra) > 1 else ''))

      staging.mkdir()
      for path, member in files.items():
        target = staging / path
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        src = tar.extractfile(member)
        assert src is not None
        with open(target, 'xb') as out:
          while chunk := src.read(CHUNK):
            digest.update(chunk)
            out.write(chunk)
        if digest.hexdigest() != manifest['files'][path].lower():
          raise bad_pack(f"checksum mismatch for {path}")
      (staging / MANIFEST).write_text(json.dumps(manifest))
    final = packs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    staging.rename(final)
    return Pack(final, manifest)
  except BaseException:
    shutil.rmtree(staging, ignore_errors=True)
    raise


def packs(packs_dir: Path) -> list[Path]:
  """Unpacked packs, oldest first: by when the manifest was written, the last step of an unpack."""
  if not packs_dir.is_dir():
    return []
  found = [p for p in packs_dir.iterdir() if p.is_dir() and not p.name.startswith('.') and (p / MANIFEST).is_file()]
  return sorted(found, key=lambda p: ((p / MANIFEST).stat().st_mtime_ns, p.name))


def prune(packs_dir: Path, keep: int = KEEP_PACKS, in_use: Path | None = None):
  """Delete all but the newest keep packs, never the one in use, and any leftover staging directories or downloads."""
  for stale in packs_dir.glob('.staging-*') if packs_dir.is_dir() else []:
    shutil.rmtree(stale, ignore_errors=True)
  for stale in packs_dir.glob('.download-*.tar') if packs_dir.is_dir() else []:
    if stale.stat().st_mtime < time.time() - STALE_DOWNLOAD_S:  # noqa: TID251  # file times are wall clock
      stale.unlink(missing_ok=True)
  for old in packs(packs_dir)[:-keep] if keep > 0 else packs(packs_dir):
    if in_use is None or old.resolve() != in_use.resolve():
      shutil.rmtree(old, ignore_errors=True)
