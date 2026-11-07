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
            {"kind": "archive", "hashes": {"sha256": "ab" * 32}},
        ),
        (
            {
                "url": "file:///home/someone/wheels/x.whl",
                "archive_info": {"hash": "sha256=" + "cd" * 32},
            },
            {"kind": "archive", "hashes": {"sha256": "cd" * 32}},
        ),
        (
            {
                "url": "file:///home/someone/wheels/x.whl",
                "archive_info": {"hashes": {"sha512": "ef" * 64}},
            },
            {"kind": "archive", "hashes": {"sha512": "ef" * 64}},
        ),
        (
            {"url": "file:///home/someone/policyengine-uk", "dir_info": {}},
            {"kind": "dir", "editable": False},
        ),
        (
            {
                "url": "file:///home/someone/policyengine-uk",
                "dir_info": {"editable": True},
            },
            {"kind": "dir", "editable": True},
        ),
        (
            {"url": "file:///home/someone/policyengine-uk"},
            {"kind": "unrecognised"},
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


def test_archives_with_other_hashes_take_their_own_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = set()
    for digest in ("ef" * 64, "12" * 64):
        _install(
            monkeypatch,
            {"url": "file:///x.whl", "archive_info": {"hashes": {"sha512": digest}}},
        )
        identities.add(_identity(spine_build._rules_engine_provenance()))
    assert len(identities) == 2


def test_an_archive_without_a_hash_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"url": "file:///x.whl", "archive_info": {}})
    with pytest.raises(ValueError, match="records no hash"):
        spine_build._rules_engine_provenance()


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


class _BatchingEngine:
    def __init__(self, record: dict | None) -> None:
        self.record = record

    def batch_record(self) -> dict | None:
        return self.record


_BATCHES = {
    "by": "household_clone_index",
    "batched_calls": 3,
    "batches": [15],
    "pass": "spine_batched",
    "declaration_sha256": "ab" * 32,
}


def _sidecar(engine: object | None) -> dict:
    import pandas as pd

    from microcosm.build.uk_runtime.national_frame import uk_national_frame
    from microcosm.frame import WeightKind

    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_benunit_id": [1, 2],
                "person_household_id": [1, 2],
                "age": [30.0, 40.0],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2]}),
        household=pd.DataFrame(
            {"household_id": [1, 2], "household_weight": [1.0, 2.0]}
        ),
        time_period=2023,
        weight_kind=WeightKind.DESIGN,
    )
    return spine_build._build_sidecar(
        frame=frame,
        stages=[],
        records=[],
        artifact_pins={},
        resource_pins={},
        input_artifact_pins={},
        hmrc_replay={},
        stochastic_contract_sha256="0" * 64,
        frs_vintage="synthetic",
        sampling=None,
        engine=engine,
    )


def test_the_sidecar_records_batches_only_once_a_call_batched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # National never batches, and today no spine frame carries the clone
    # index, so the sidecar's rules engine stays the provenance; a spine whose
    # calls ran clone by clone adds how (microcosm#931).
    _install(monkeypatch, None)
    provenance = spine_build._rules_engine_provenance()
    for engine in (None, object(), _BatchingEngine(None)):
        assert spine_build._rules_engine_record(engine) == provenance
        assert _sidecar(engine)["rules_engine"] == provenance
    batched = _BatchingEngine(dict(_BATCHES))
    assert spine_build._rules_engine_record(batched) == {
        **provenance,
        "engine_batches": _BATCHES,
    }
    assert _sidecar(batched)["rules_engine"]["engine_batches"] == _BATCHES
    # The engine identity is hashed from the provenance alone.
    assert _identity(spine_build._rules_engine_provenance()) == _identity(
        {"package": "policyengine-uk", "version": VERSION}
    )
