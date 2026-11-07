"""A UK spine's engine provenance tells a URL install from the release (microcosm#1411).

A git install of an engine pull request carries the release's version string,
so the version alone gave both one engine identity. PEP 610's
``direct_url.json`` names the commit; an install from an index writes none, so
a release's provenance, and the identity hashed from it, are unchanged.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

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
        self.frames: list = []

    def batch_record(self, frame) -> dict | None:
        self.frames.append(frame)
        return self.record


_BATCHES = {
    "by": "household_clone_index",
    "batches": 15,
    "pass": "spine_batched",
    "declaration_sha256": "ab" * 32,
}


def _final_frame(clones: tuple[int, ...] | None = None):
    """A made-up final spine frame: one household per clone, or two
    households without the clone index."""

    import pandas as pd

    from microcosm.build.uk_runtime.national_frame import uk_national_frame
    from microcosm.frame import WeightKind

    households = 2 if clones is None else len(clones)
    ids = list(range(1, households + 1))
    household = pd.DataFrame(
        {"household_id": ids, "household_weight": [1.0 + i for i in ids]}
    )
    if clones is not None:
        household["household_clone_index"] = list(clones)
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
                "age": [30.0 + i for i in ids],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=household,
        time_period=2023,
        weight_kind=WeightKind.DESIGN,
    )


def _sidecar(engine: object | None, frame=None) -> dict:
    return spine_build._build_sidecar(
        frame=_final_frame() if frame is None else frame,
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


def test_the_sidecar_records_batches_only_for_a_final_frame_that_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # National never batches, and today no spine frame carries the clone
    # index, so the sidecar's rules engine stays the provenance; a spine
    # whose final frame carries more than one clone adds how the adapter runs
    # a call over it (microcosm#931).
    _install(monkeypatch, None)
    provenance = spine_build._rules_engine_provenance()
    frame = _final_frame()
    for engine in (None, object(), _BatchingEngine(None)):
        assert spine_build._rules_engine_record(engine, frame) == provenance
        assert _sidecar(engine, frame)["rules_engine"] == provenance
    batched = _BatchingEngine(dict(_BATCHES))
    assert spine_build._rules_engine_record(batched, frame) == {
        **provenance,
        "engine_batches": _BATCHES,
    }
    assert _sidecar(batched, frame)["rules_engine"]["engine_batches"] == _BATCHES
    # The record is asked of the final frame the sidecar describes.
    assert batched.frames == [frame, frame]
    # The engine identity is hashed from the provenance alone.
    assert _identity(spine_build._rules_engine_provenance()) == _identity(
        {"package": "policyengine-uk", "version": VERSION}
    )


def test_a_resumed_spine_records_the_batches_a_fresh_run_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A resumed spine's engine stages that hit their checkpoints run no call,
    # so the record is read from the final frame, never from the calls this
    # process ran: an adapter that ran a batched call and one that ran none
    # (every engine stage a cache hit) write the same sidecar (microcosm#931).
    import numpy as np

    from microcosm.frame.adapters import policyengine_uk as adapter_module
    from microcosm.frame.adapters.engine_population import SPINE_BATCHED_PASS
    from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine
    from microcosm.frame.adapters.policyengine_uk_population import (
        UK_ENGINE_POPULATION,
        UK_SPINE_ENGINE_BATCH_COLUMN,
    )

    _install(monkeypatch, None)
    frame = _final_frame((0, 1, 2))
    monkeypatch.setattr(
        adapter_module,
        "materialize_in_batches",
        lambda bundle, variables, period, **kwargs: {
            name: np.zeros(bundle.n("person")) for name in variables
        },
    )
    fresh, ran = PolicyEngineUKEngine(), PolicyEngineUKEngine()
    for engine in (fresh, ran):
        monkeypatch.setattr(
            engine,
            "_import_policyengine_uk",
            lambda: SimpleNamespace(Microsimulation=None),
        )
    ran.materialize(frame, ["age"], 2023)
    expected = {
        **spine_build._rules_engine_provenance(),
        "engine_batches": {
            "by": UK_SPINE_ENGINE_BATCH_COLUMN,
            "batches": 3,
            "pass": SPINE_BATCHED_PASS.name,
            "declaration_sha256": UK_ENGINE_POPULATION.sha256,
        },
    }
    assert _sidecar(fresh, frame)["rules_engine"] == expected
    assert _sidecar(ran, frame)["rules_engine"] == expected
    # One clone, or no clone index (a national spine), records no batches.
    for final in (_final_frame((0,)), _final_frame()):
        assert "engine_batches" not in _sidecar(fresh, final)["rules_engine"]
