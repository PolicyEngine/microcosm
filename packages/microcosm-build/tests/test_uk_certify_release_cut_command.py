"""The release-cut certifier command's Logbook pins (microcosm#823 E1).

The certifier had never run end to end on a real candidate: its Logbook pin
roles carried a digest without the byte size the pin digest requires, so
every certification refused up front. The command is exercised here with
its battery stubbed, on the pins alone.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from microcosm.build.logbook_adoption import role_pins_digest
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
