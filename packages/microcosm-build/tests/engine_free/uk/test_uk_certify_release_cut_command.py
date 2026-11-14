"""The release-cut certifier command's Logbook pins (microcosm#823 E1).

The certifier had never run end to end on a real candidate: its Logbook pin
roles carried a digest without the byte size the pin digest requires, so
every certification refused up front. The command is exercised here with
its battery stubbed, on the pins alone, and on the inputs it binds before
any gate reads them: the spine the calibration recorded as the candidate's
parent, that spine's build sidecar, the one the build record's digest names,
and the sidecar's geography binding (microcosm#931); and on the candidate it
loads, which must carry no geographic clone index.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from microcosm.build.logbook_adoption import role_pins_digest
from microcosm.build.uk_runtime.geographic_clones import ladder_clone_index_column
from test_support.microcosm_build.uk_full_build_cli import (
    patch_support_register,
    synthetic_spine_geography,
)
from tools import certify_uk_release_cut as certify


def test_certifier_pins_carry_the_byte_sizes_the_logbook_digest_requires(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    candidate = tmp_path / "microcosm_uk_2024_25.h5"
    candidate.write_bytes(b"candidate bytes")
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "consumer_facts.jsonl").write_text('{"fact": 1}\n' * 3)
    candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    # The parser refuses aliased paths, so every evidence file is its own.
    placeholders = {}
    for name in (
        "spine.h5",
        "calibration_diagnostics.json",
        "build_record.json",
        "terminal_gates.json",
        "input_mass_reference.json",
        "score_vs_incumbent.json",
    ):
        placeholders[name] = tmp_path / name
        placeholders[name].write_text("{}")
    facts_sha = "1" * 64
    captured: dict[str, object] = {}

    def fake_run(args, state):
        captured["state"] = state
        return {"shippable": True}

    monkeypatch.setattr(certify, "_run", fake_run)
    monkeypatch.setattr(certify, "resolve_predecessor", lambda digest: None)
    monkeypatch.setattr(certify, "record_terminal_attempt", lambda **kwargs: None)

    assert (
        certify.main(
            [
                "--candidate-h5",
                str(candidate),
                "--candidate-sha256",
                candidate_sha,
                "--candidate-name",
                "microcosm_uk_2024_25",
                "--spine-h5",
                str(placeholders["spine.h5"]),
                "--spine-sha256",
                hashlib.sha256(placeholders["spine.h5"].read_bytes()).hexdigest(),
                "--diagnostics-json",
                str(placeholders["calibration_diagnostics.json"]),
                "--build-record-json",
                str(placeholders["build_record.json"]),
                "--seam-gate-report",
                str(placeholders["terminal_gates.json"]),
                "--ledger-facts",
                str(ledger),
                "--ledger-facts-sha256",
                facts_sha,
                "--ledger-manifest-sha256",
                "2" * 64,
                "--input-mass-reference",
                str(placeholders["input_mass_reference.json"]),
                "--score-receipt",
                str(placeholders["score_vs_incumbent.json"]),
                "--release-id",
                "microcosm-uk-2024-25-national",
                "--release-cut-gate-json",
                str(tmp_path / "release_cut_gates.json"),
                "--certification-json",
                str(tmp_path / "release_certification.json"),
            ]
        )
        == 0
    )

    state = captured["state"]
    assert state.input_pins_digest == role_pins_digest(
        {
            "candidate_h5": {
                "sha256": candidate_sha,
                "size_bytes": len(b"candidate bytes"),
            },
            "ledger_facts": {
                "sha256": facts_sha,
                "size_bytes": (ledger / "consumer_facts.jsonl").stat().st_size,
            },
            "spine_h5": {
                "sha256": hashlib.sha256(
                    placeholders["spine.h5"].read_bytes()
                ).hexdigest(),
                "size_bytes": placeholders["spine.h5"].stat().st_size,
            },
        }
    )
    assert state.build_id.startswith("uk-frs-release-certification-attempt-")
    assert '"shippable": true' in capsys.readouterr().out


def test_certifier_refuses_a_spine_that_is_not_the_candidates_recorded_parent(
    monkeypatch, tmp_path: Path
) -> None:
    """Review C1: the supplied spine pins fine on its own hash but is not the
    parent the build record and the signed diagnostics name; the certifier
    refuses before any gate reads its receipts."""
    candidate = tmp_path / "microcosm_uk_2024_25.h5"
    candidate.write_bytes(b"candidate bytes")
    spine = tmp_path / "spine.h5"
    spine.write_bytes(b"spine B bytes")
    (tmp_path / "spine.build.json").write_text('{"stages": ["frs_spine"]}')
    parent_a = "3" * 64
    diagnostics = tmp_path / "calibration_diagnostics.json"
    diagnostics.write_text(
        json.dumps({"build": {"input_posture": {"sha256": parent_a}}, "targets": []})
    )
    build_record = tmp_path / "build_record.json"
    build_record.write_text(
        json.dumps(
            {
                "artifacts": {
                    "diagnostics_json": {
                        "sha256": hashlib.sha256(diagnostics.read_bytes()).hexdigest()
                    }
                },
                "input_posture": {"sha256": parent_a},
                "source_pins": {"input_h5": {"sha256": parent_a}},
            }
        )
    )
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "consumer_facts.jsonl").write_text('{"fact": 1}\n')
    for name in ("terminal_gates.json", "input_mass_reference.json", "score.json"):
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(certify, "resolve_predecessor", lambda digest: None)
    monkeypatch.setattr(certify, "record_terminal_attempt", lambda **kwargs: None)

    with pytest.raises(SystemExit, match="not the candidate's recorded parent"):
        certify.main(
            [
                "--candidate-h5",
                str(candidate),
                "--candidate-sha256",
                hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "--candidate-name",
                "microcosm_uk_2024_25",
                "--spine-h5",
                str(spine),
                "--spine-sha256",
                hashlib.sha256(spine.read_bytes()).hexdigest(),
                "--diagnostics-json",
                str(diagnostics),
                "--build-record-json",
                str(build_record),
                "--seam-gate-report",
                str(tmp_path / "terminal_gates.json"),
                "--ledger-facts",
                str(ledger),
                "--ledger-facts-sha256",
                "1" * 64,
                "--ledger-manifest-sha256",
                "2" * 64,
                "--input-mass-reference",
                str(tmp_path / "input_mass_reference.json"),
                "--score-receipt",
                str(tmp_path / "score.json"),
                "--release-id",
                "microcosm-uk-2024-25-national",
                "--release-cut-gate-json",
                str(tmp_path / "release_cut_gates.json"),
                "--certification-json",
                str(tmp_path / "release_certification.json"),
            ]
        )


class _PastTheBoundInputsError(Exception):
    """Raised by the first step after the certifier binds its inputs."""


def _certify_bound_inputs(
    tmp_path: Path, monkeypatch, *, recorded_sidecar, sidecar_fields=None
) -> None:
    """Run the certifier on stand-in inputs whose spine is the candidate's
    recorded parent, its build record recording ``recorded_sidecar`` as the
    spine sidecar's digest (``None`` records none); a run past the inputs is
    stopped at the Ledger read by :class:`_PastTheBoundInputsError`.

    The sidecar records one clone and the stand-in spine's geography binding,
    whose toy support pins stand as sources.yaml's conditioned supports
    (``patch_support_register``), with ``sidecar_fields`` replacing any of its
    fields (microcosm#931).
    """
    candidate = tmp_path / "microcosm_uk_2024_25.h5"
    candidate.write_bytes(b"candidate bytes")
    spine = tmp_path / "spine.h5"
    spine.write_bytes(b"spine bytes")
    sidecar = tmp_path / "spine.build.json"
    patch_support_register(monkeypatch)
    sidecar.write_text(
        json.dumps(
            {
                "stages": ["frs_spine"],
                "geographic_clone_count": 1,
                "geography": synthetic_spine_geography(),
                **(sidecar_fields or {}),
            }
        )
    )
    parent = hashlib.sha256(spine.read_bytes()).hexdigest()
    diagnostics = tmp_path / "calibration_diagnostics.json"
    diagnostics.write_text(
        json.dumps({"build": {"input_posture": {"sha256": parent}}, "targets": []})
    )
    record = {
        "artifacts": {
            "diagnostics_json": {
                "sha256": hashlib.sha256(diagnostics.read_bytes()).hexdigest()
            }
        },
        "input_posture": {"sha256": parent},
        "source_pins": {"input_h5": {"sha256": parent}},
    }
    if recorded_sidecar is not None:
        record["spine_provenance"] = {
            "sidecar": {"sha256": recorded_sidecar(sidecar.read_bytes())}
        }
    build_record = tmp_path / "build_record.json"
    build_record.write_text(json.dumps(record))
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "consumer_facts.jsonl").write_text('{"fact": 1}\n')
    for name in ("terminal_gates.json", "input_mass_reference.json", "score.json"):
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(certify, "resolve_predecessor", lambda digest: None)
    monkeypatch.setattr(certify, "record_terminal_attempt", lambda **kwargs: None)

    def past_the_inputs(*args, **kwargs):
        raise _PastTheBoundInputsError

    monkeypatch.setattr(certify, "load_ledger_consumer_artifact", past_the_inputs)
    certify.main(
        [
            "--candidate-h5",
            str(candidate),
            "--candidate-sha256",
            hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "--candidate-name",
            "microcosm_uk_2024_25",
            "--spine-h5",
            str(spine),
            "--spine-sha256",
            parent,
            "--diagnostics-json",
            str(diagnostics),
            "--build-record-json",
            str(build_record),
            "--seam-gate-report",
            str(tmp_path / "terminal_gates.json"),
            "--ledger-facts",
            str(ledger),
            "--ledger-facts-sha256",
            "1" * 64,
            "--ledger-manifest-sha256",
            "2" * 64,
            "--input-mass-reference",
            str(tmp_path / "input_mass_reference.json"),
            "--score-receipt",
            str(tmp_path / "score.json"),
            "--release-id",
            "microcosm-uk-2024-25-national",
            "--release-cut-gate-json",
            str(tmp_path / "release_cut_gates.json"),
            "--certification-json",
            str(tmp_path / "release_certification.json"),
        ]
    )


@pytest.mark.parametrize(
    "recorded_sidecar",
    [
        pytest.param(
            lambda written: hashlib.sha256(written + b"\n").hexdigest(),
            id="rewritten-since",
        ),
        pytest.param(None, id="unrecorded"),
    ],
)
def test_certifier_refuses_a_spine_sidecar_the_calibration_did_not_record(
    monkeypatch, tmp_path: Path, recorded_sidecar
) -> None:
    """The spine's build sidecar is evidence: its stage names and fit-weight
    records feed the release-cut battery, and the certification holds its
    geography binding to a release's (microcosm#931). The certifier refuses a
    sidecar whose bytes are not the ones the build record's
    spine_provenance.sidecar.sha256 names, or one the record names none for,
    before any gate reads it, naming the field and no digest."""
    with pytest.raises(SystemExit) as refused:
        _certify_bound_inputs(tmp_path, monkeypatch, recorded_sidecar=recorded_sidecar)
    message = str(refused.value)
    assert message == (
        "error: the spine build sidecar beside --spine-h5 is not the one the "
        "calibration recorded (build_record.spine_provenance.sidecar.sha256); "
        "its stage names, fit-weight records and geography binding must be the "
        "ones this candidate was calibrated on."
    )


def test_certifier_binds_the_spine_sidecar_the_calibration_recorded(
    monkeypatch, tmp_path: Path
) -> None:
    """A sidecar whose digest the build record carries passes the check, and
    the certifier goes on to read the Ledger (microcosm#931)."""
    with pytest.raises(_PastTheBoundInputsError):
        _certify_bound_inputs(
            tmp_path,
            monkeypatch,
            recorded_sidecar=lambda written: hashlib.sha256(written).hexdigest(),
        )


_GEOGRAPHY_LEAD = (
    "error: a national release certifies only a spine of one geographic clone "
    "whose geography binding is a release's (microcosm#931); "
)


@pytest.mark.parametrize(
    ("sidecar_fields", "refusal"),
    [
        pytest.param(
            {"geographic_clone_count": 3},
            "the spine sidecar's binding fails: geographic_clone_count is not K = 1.",
            id="three-clones",
        ),
        pytest.param(
            {"geography": None},
            "the spine sidecar records no geography binding (a spine that runs "
            "no area draw writes none).",
            id="no-binding",
        ),
    ],
)
def test_certifier_refuses_a_spine_binding_before_any_gate(
    monkeypatch, tmp_path: Path, sidecar_fields, refusal
) -> None:
    """The certifier holds the recorded sidecar's geography binding to a
    release's before any gate runs (microcosm#931): a spine of three clones,
    or one that records no binding, is refused before the Ledger read, so no
    release-cut battery runs and no signed report is written for it; the
    refusal is the composer's, naming fields and K only."""
    with pytest.raises(SystemExit) as refused:
        _certify_bound_inputs(
            tmp_path,
            monkeypatch,
            recorded_sidecar=lambda written: hashlib.sha256(written).hexdigest(),
            sidecar_fields=sidecar_fields,
        )
    assert str(refused.value) == _GEOGRAPHY_LEAD + refusal
    assert not (tmp_path / "release_cut_gates.json").exists()


def _candidate_with(*indexed: str) -> SimpleNamespace:
    """A stand-in candidate frame whose ``indexed`` entities carry their
    geographic clone index, at zero."""
    tables = {}
    for entity in ("person", "benunit", "household"):
        table = pd.DataFrame({f"{entity}_id": [1, 2]})
        if entity in indexed:
            table[ladder_clone_index_column(entity)] = 0
        tables[entity] = table
    return SimpleNamespace(table=tables.__getitem__)


@pytest.mark.parametrize("entity", ["person", "benunit", "household"])
def test_certifier_refuses_a_candidate_that_carries_a_clone_index(entity) -> None:
    """The national release boundary drops each entity's geographic clone
    index, constant at one clone, and the release cut's degenerate gate skips
    what the national line drops, so the certifier refuses a candidate that
    kept one itself, naming the column and no value (microcosm#931)."""
    with pytest.raises(SystemExit) as refused:
        certify._refuse_clone_indexed_candidate(_candidate_with(entity))
    assert str(refused.value) == (
        "error: a national release carries no geographic clone index "
        f"(microcosm#931); the candidate carries {ladder_clone_index_column(entity)}."
    )


def test_certifier_takes_a_candidate_without_a_clone_index() -> None:
    """A national candidate as the national line writes it carries no clone
    index, and passes (microcosm#931)."""
    assert certify._refuse_clone_indexed_candidate(_candidate_with()) is None
