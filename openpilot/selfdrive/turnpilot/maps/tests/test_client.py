import gzip
import json
import socket
from unittest.mock import patch

import pytest
import requests

from openpilot.selfdrive.turnpilot.maps import client as map_client
from openpilot.selfdrive.turnpilot.maps.client import MapClient, MapError, cert_fingerprint, normalize_host
from openpilot.selfdrive.turnpilot.maps.coverage import STATUS_PARAM, CoverageCache, refresh
from openpilot.selfdrive.turnpilot.maps.tests.fakes import INDY, INDY_EAST, KEY, FakeParams, MapServer, coverage_doc, make_pack, self_signed


@pytest.fixture
def server():
  s = MapServer()
  yield s
  s.close()


def client_for(server, tmp_path, **overrides):
  return MapClient.from_params(server.params(**overrides), tmp_path)


def test_health_and_coverage_over_pinned_tls(server, tmp_path):
  c = client_for(server, tmp_path)
  assert c.health() == {'ok': True, 'regions': ['indianapolis', 'seattle']}
  assert c.coverage()['version'] == coverage_doc()['version']
  assert (tmp_path / 'server-cert.pem').read_text().strip() == server.cert.strip()
  health, cov = server.requests
  assert 'Authorization' not in health[2]  # health needs no key
  assert cov[2]['Authorization'] == f'Bearer {KEY}'


def test_requests_always_verify_against_the_saved_certificate(server, tmp_path):
  c = client_for(server, tmp_path)
  real = requests.request
  seen = []

  def spy(*args, **kwargs):
    seen.append(kwargs['verify'])
    return real(*args, **kwargs)

  with patch.object(map_client.requests, 'request', side_effect=spy):
    c.health()
    c.coverage()
  assert seen == [str(tmp_path / 'server-cert.pem')] * 2  # a path, never False


def test_bad_key(server, tmp_path):
  with pytest.raises(MapError) as e:
    client_for(server, tmp_path, TurnPilotMapKey='wrong').coverage()
  assert e.value.code == 'unauthorized' and 'API key' in e.value.message


def test_wrong_certificate_is_a_tls_error(server, tmp_path):
  other, _ = self_signed()
  with pytest.raises(MapError) as e:
    client_for(server, tmp_path, TurnPilotMapCert=other).health()
  assert e.value.code == 'tls'
  assert server.requests == []  # the handshake failed before any request


def test_certificate_for_another_address_is_a_tls_error(server, tmp_path):
  with pytest.raises(MapError) as e:
    client_for(server, tmp_path, TurnPilotMapHost=server.host.replace('127.0.0.1', 'localhost')).health()
  assert e.value.code == 'tls'  # the SAN is the IP, so the name doesn't match


def test_unreachable(tmp_path):
  cert, _ = self_signed()
  with socket.socket() as s:
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]  # closed again before the request
  c = MapClient(f'https://127.0.0.1:{port}', KEY, cert, tmp_path)
  with pytest.raises(MapError) as e:
    c.coverage()
  assert e.value.code == 'unreachable' and e.value.retry


def test_settings_are_checked(tmp_path):
  with pytest.raises(MapError) as e:
    MapClient.from_params(FakeParams({'TurnPilotMapHost': 'https://1.2.3.4'}), tmp_path)
  assert e.value.code == 'not_configured' and 'API key' in e.value.message and 'certificate' in e.value.message
  assert normalize_host('136.113.85.41') == 'https://136.113.85.41'
  assert normalize_host('https://136.113.85.41:8443/') == 'https://136.113.85.41:8443'
  with pytest.raises(MapError):
    normalize_host('http://136.113.85.41')  # the key would go in the clear
  with pytest.raises(MapError):
    MapClient('https://1.2.3.4', KEY, 'not a certificate', tmp_path)
  cert, _ = self_signed()
  assert len(cert_fingerprint(cert).split(':')) == 32


@pytest.mark.parametrize('compressed', [True, False])
def test_corridor_download(server, tmp_path, compressed):
  pack = make_pack({f'tiles/2/{i}.gph': b'tile ' * 2000 for i in range(20)})
  server.corridor, server.gzip = (200, pack, 'application/x-tar'), compressed
  seen = []
  size = client_for(server, tmp_path).corridor(INDY, INDY_EAST, tmp_path / 'out.tar', lambda done, total: seen.append((done, total)))
  assert size == len(pack) and (tmp_path / 'out.tar').read_bytes() == pack  # the tar itself, decoded
  wire = len(gzip.compress(pack)) if compressed else len(pack)
  assert seen[-1] == (wire, wire)  # progress counts bytes off the wire against Content-Length
  assert (wire < len(pack)) == compressed
  assert server.corridor_requests() == [{'start': list(INDY), 'destination': list(INDY_EAST)}]


@pytest.mark.parametrize('body,code,words', [
  ({'error': 'outside_coverage', 'which': 'start'}, 'outside_coverage', 'The car is outside'),
  ({'error': 'outside_coverage', 'which': 'destination'}, 'outside_coverage', 'The destination is outside'),
  ({'error': 'outside_coverage', 'which': 'both'}, 'outside_coverage', 'The car and the destination are outside'),
  ({'error': 'different_regions'}, 'different_regions', 'different mapped areas'),
  ({'error': 'no_route'}, 'no_route', 'no route'),
  ({'error': 'something new'}, 'bad_response', 'refused the route'),
])
def test_corridor_422(server, tmp_path, body, code, words):
  server.corridor = (422, json.dumps(body).encode(), 'application/json')
  with pytest.raises(MapError) as e:
    client_for(server, tmp_path).corridor(INDY, INDY_EAST, tmp_path / 'out.tar', coverage_names='Indianapolis and Seattle')
  assert e.value.code == code and words in e.value.message and not e.value.retry
  if code == 'outside_coverage':
    assert 'Indianapolis and Seattle' in e.value.message


def test_refresh_caches_and_records_status(server, tmp_path):
  params = server.params()
  refresh(params, tmp_path)
  assert CoverageCache(tmp_path).get().names() == 'Indianapolis and Seattle'
  assert params.values[STATUS_PARAM]['ok'] and params.values[STATUS_PARAM]['version'] == '2026-10-27'
  params.values['TurnPilotMapKey'] = 'wrong'
  with pytest.raises(MapError):
    refresh(params, tmp_path)
  assert params.values[STATUS_PARAM]['ok'] is False and params.values[STATUS_PARAM]['code'] == 'unauthorized'
  assert CoverageCache(tmp_path).get() is not None  # a failed refresh keeps the last good cache
  server.coverage = {'type': 'FeatureCollection', 'version': 'x', 'features': []}
  params.values['TurnPilotMapKey'] = KEY
  with pytest.raises(MapError) as e:
    refresh(params, tmp_path)
  assert e.value.code == 'bad_response'
  assert CoverageCache(tmp_path).get().version == '2026-10-27'
