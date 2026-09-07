"""Reference acquisition never persists credentials and supports pinned replay."""

import hashlib
from urllib.error import HTTPError

import pytest

from microcosm.build.us_runtime import cd_reference_sources as sources


def test_capture_fetches_once_and_verifies_offline_content(tmp_path):
    calls = []

    def fetch(url, key):
        calls.append((url, key))
        return b'{"fixture": true}'

    request = sources.census_requests()[0]
    first = sources.capture_request(
        tmp_path, request, api_key="test-secret", fetch=fetch
    )
    second = sources.capture_request(tmp_path, request, offline=True, fetch=fetch)
    assert first == second
    assert len(calls) == 1
    assert first["sha256"] == hashlib.sha256(b'{"fixture": true}').hexdigest()
    assert b"test-secret" not in b"".join(
        path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    )
    (tmp_path / first["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        sources.capture_request(tmp_path, request, offline=True, fetch=fetch)


def test_missing_offline_source_refuses_network_and_key_reflection(tmp_path):
    request = sources.census_requests()[0]
    with pytest.raises(ValueError, match="offline"):
        sources.capture_request(tmp_path, request, offline=True)
    with pytest.raises(ValueError, match="credential"):
        sources.capture_request(
            tmp_path,
            request,
            api_key="reflected-secret",
            fetch=lambda *_: b'{"bad":"reflected-secret"}',
        )
    assert not list(tmp_path.rglob("*.json"))


def test_request_error_does_not_include_key_or_keyed_url(monkeypatch):
    class FailingOpener:
        def open(self, request, **kwargs):
            raise HTTPError(request.full_url, 403, "secret in remote reason", {}, None)

    monkeypatch.setattr(sources, "build_opener", lambda *_: FailingOpener())
    with pytest.raises(ValueError) as error:
        sources.fetch_census(sources.census_requests()[0]["url"], "test-secret")
    assert "test-secret" not in str(error.value)
    assert "key=" not in str(error.value)
    assert "403" in str(error.value)


def test_capture_refuses_undeclared_url_and_request_metadata(tmp_path):
    request = {**sources.census_requests()[0], "url": "https://example.org/"}
    with pytest.raises(ValueError, match="declared"):
        sources.capture_request(tmp_path, request, fetch=lambda *_: b"{}")
