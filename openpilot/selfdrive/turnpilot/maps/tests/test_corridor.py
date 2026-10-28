import hashlib
import os
import io
import tarfile
import time

import pytest

from openpilot.selfdrive.turnpilot.maps import corridor
from openpilot.selfdrive.turnpilot.maps.client import MapError
from openpilot.selfdrive.turnpilot.maps.tests.fakes import make_pack

FILES = {'tiles/2/000/123/456.gph': b'level two tile', 'tiles/0/000/003.gph': b'level zero', 'tiles/1/000/045.gph': b'one'}


def unpack(tmp_path, data):
  tar = tmp_path / 'pack.tar'
  tar.write_bytes(data)
  return corridor.unpack(tar, tmp_path / 'packs')


def rejected(tmp_path, data, words):
  with pytest.raises(MapError) as e:
    unpack(tmp_path, data)
  assert e.value.code == 'bad_pack' and words in e.value.message, e.value.message
  packs = tmp_path / 'packs'
  assert not packs.exists() or not any(packs.iterdir()), "a rejected pack leaves nothing behind"
  return e.value


def member(name, data=b'x', kind=tarfile.REGTYPE, link=''):
  def build():
    info = tarfile.TarInfo(name)
    info.type, info.linkname = kind, link
    info.size = len(data) if kind == tarfile.REGTYPE else 0
    return info, data if kind == tarfile.REGTYPE else None
  return build


def test_good_pack(tmp_path):
  pack = unpack(tmp_path, make_pack(FILES))
  assert pack.manifest['region'] == 'indianapolis'
  for name, data in FILES.items():
    assert (pack.path / name).read_bytes() == data
  assert (pack.tiles / '2/000/123/456.gph').is_file()
  assert corridor.packs(tmp_path / 'packs') == [pack.path]


def test_directory_members_are_fine(tmp_path):
  pack = unpack(tmp_path, make_pack(FILES, extra=[member('tiles/', kind=tarfile.DIRTYPE), member('tiles/2', kind=tarfile.DIRTYPE)]))
  assert pack.path.is_dir()


def test_bad_checksum(tmp_path):
  listed = {n: hashlib.sha256(d).hexdigest() for n, d in FILES.items()}
  listed['tiles/1/000/045.gph'] = hashlib.sha256(b'something else').hexdigest()
  rejected(tmp_path, make_pack(FILES, manifest_files=listed), 'checksum mismatch for tiles/1/000/045.gph')


def test_extra_file(tmp_path):
  listed = {n: hashlib.sha256(d).hexdigest() for n, d in FILES.items() if n != 'tiles/0/000/003.gph'}
  rejected(tmp_path, make_pack(FILES, manifest_files=listed), 'unlisted file tiles/0/000/003.gph')
  rejected(tmp_path, make_pack(FILES, extra=[member('readme.txt')]), "unexpected file 'readme.txt'")


def test_missing_file(tmp_path):
  listed = {**{n: hashlib.sha256(d).hexdigest() for n, d in FILES.items()}, 'tiles/2/000/999.gph': '0' * 64}
  rejected(tmp_path, make_pack(FILES, manifest_files=listed), 'missing tiles/2/000/999.gph')


@pytest.mark.parametrize('name', ['../evil.gph', 'tiles/../../evil.gph', 'tiles/2/../../../evil', '/etc/passwd', '/tiles/2/0.gph',
                                  'tiles\\..\\evil', 'C:/tiles/x'])
def test_path_traversal_in_the_tar(tmp_path, name):
  rejected(tmp_path, make_pack(FILES, extra=[member(name)]), 'unsafe path')
  assert not (tmp_path / 'evil.gph').exists() and not (tmp_path.parent / 'evil.gph').exists()


def test_path_traversal_in_the_manifest(tmp_path):
  listed = {**{n: hashlib.sha256(d).hexdigest() for n, d in FILES.items()}, '../evil.gph': '0' * 64}
  rejected(tmp_path, make_pack(FILES, manifest_files=listed), 'unsafe path')


@pytest.mark.parametrize('kind,link', [(tarfile.SYMTYPE, '/etc/passwd'), (tarfile.LNKTYPE, 'tiles/0/000/003.gph'),
                                       (tarfile.FIFOTYPE, ''), (tarfile.CHRTYPE, '')])
def test_links_and_devices(tmp_path, kind, link):
  rejected(tmp_path, make_pack(FILES, extra=[member('tiles/2/link.gph', kind=kind, link=link)]), 'not a regular file')


def test_duplicate_member(tmp_path):
  rejected(tmp_path, make_pack(FILES, extra=[member('tiles/1/000/045.gph', b'one')]), 'appears twice')


def test_no_or_bad_manifest(tmp_path):
  rejected(tmp_path, make_pack(FILES, manifest=False), 'no manifest')
  buf = io.BytesIO()
  with tarfile.open(fileobj=buf, mode='w') as tar:
    info = tarfile.TarInfo('manifest.json')
    info.size = 5
    tar.addfile(info, io.BytesIO(b'nope!'))
  rejected(tmp_path, buf.getvalue(), 'manifest is not valid')
  rejected(tmp_path, b'this is not a tar file at all' * 30, 'not a tar file')
  listed = dict.fromkeys(FILES, 'xyz')
  rejected(tmp_path, make_pack(FILES, manifest_files=listed), 'not SHA-256')


def test_prune_keeps_the_newest_and_the_one_in_use(tmp_path):
  made = []
  for i in range(5):
    made.append(unpack(tmp_path, make_pack({'tiles/2/0.gph': bytes([i])})).path)
    time.sleep(0.01)
  packs_dir = tmp_path / 'packs'
  assert corridor.packs(packs_dir) == made  # oldest first, whatever their names
  (packs_dir / '.staging-dead').mkdir()  # left by a crash mid-unpack
  (packs_dir / '.download-dead.tar').write_bytes(b'cut off')
  os.utime(packs_dir / '.download-dead.tar', (1, 1))
  (packs_dir / '.download-live.tar').write_bytes(b'still coming')  # another download in progress
  corridor.prune(packs_dir, keep=3, in_use=made[0])
  assert corridor.packs(packs_dir) == [made[0]] + made[2:]
  assert not (packs_dir / '.staging-dead').exists() and not (packs_dir / '.download-dead.tar').exists()
  assert (packs_dir / '.download-live.tar').exists()
  corridor.prune(packs_dir, keep=2)
  assert corridor.packs(packs_dir) == made[3:]
