"""A UK spine's engine provenance tells a URL install from the release (microcosm#1411).

A git install of an engine pull request carries the release's version string,
so the version alone gave both one engine identity. PEP 610's
``direct_url.json`` names the commit; an install from an index writes none, so
a release's provenance, and the identity hashed from it, are unchanged.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from microcosm.build.logbook import canonical_json_bytes
from microcosm.build.uk_runtime import spine_build

VERSION = "2.162.12"
COMMIT = "b71ce7934" + "0" * 31


class _Distribution:
    def __init__(self, direct_url: dict | None) -> None:
        self._text = None if direct_url is None else json.dumps(direct_url)

    def read_text(self, filename: str) -> str | None:
        return self._text if filename == "direct_url.json" else None


def _install(monkeypatch: pytest.MonkeyPatch, direct_url: dict | None) -> None:
    monkeypatch.setattr(spine_build.metadata, "version", lambda package: VERSION)
    monkeypatch.setattr(
        spine_build.metadata, "distribution", lambda package: _Distribution(direct_url)
    )


def _identity(provenance: dict) -> str:
    return hashlib.sha256(canonical_json_bytes(provenance)).hexdigest()


def test_an_index_release_keeps_its_provenance_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, None)
    provenance = spine_build._rules_engine_provenance()
    assert provenance == {"package": "policyengine-uk", "version": VERSION}
    assert _identity(provenance) == _identity(
        {"package": "policyengine-uk", "version": VERSION}
    )


def test_a_git_install_records_its_commit_and_takes_its_own_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        {
            "url": "https://github.com/PolicyEngine/policyengine-uk",
            "vcs_info": {
                "vcs": "git",
                "commit_id": COMMIT,
                "requested_revision": "b71ce793",
            },
        },
    )
    provenance = spine_build._rules_engine_provenance()
    assert provenance["direct_install"] == {
        "kind": "vcs",
        "vcs": "git",
        "commit_id": COMMIT,
    }
    assert _identity(provenance) != _identity(
        {"package": "policyengine-uk", "version": VERSION}
    )


@pytest.mark.parametrize(
    ("direct_url", "expected"),
    [
        (
            {
                "url": "file:///home/someone/wheels/x.whl",
                "archive_info": {"hashes": {"sha256": "ab" * 32}},
            },
            {"kind": "archive", "sha256": "ab" * 32},
        ),
        (
            {
                "url": "file:///home/someone/wheels/x.whl",
                "archive_info": {"hash": "sha256=" + "cd" * 32},
            },
            {"kind": "archive", "sha256": "cd" * 32},
        ),
        (
            {"url": "file:///home/someone/policyengine-uk", "dir_info": {}},
            {"kind": "dir", "editable": False},
        ),
    ],
)
def test_other_url_installs_record_their_kind_and_never_a_url(
    monkeypatch: pytest.MonkeyPatch, direct_url: dict, expected: dict
) -> None:
    _install(monkeypatch, direct_url)
    provenance = spine_build._rules_engine_provenance()
    assert provenance["direct_install"] == expected
    assert "someone" not in json.dumps(provenance)


def test_a_missing_engine_still_records_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(package: str) -> str:
        raise spine_build.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(spine_build.metadata, "version", missing)
    assert spine_build._rules_engine_provenance() == {
        "package": "policyengine-uk",
        "version": "unavailable",
    }
