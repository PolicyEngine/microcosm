"""The release-cut certification producer and its composer refusals."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from datetime import date
from pathlib import Path

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.gate_battery import gate_signing_key_env
from microcosm.build.uk_runtime import release_certification
from microcosm.build.uk_runtime.calibration_run import (
    UK_CALIBRATION_GATE_SCOPE,
    UK_NATIONAL_GATE_SCOPE,
    UK_SHARED_GATE_IDS,
    UK_SPINE_GATE_SCOPE,
)
from microcosm.build.uk_runtime.release_certification import (
    UKReleaseCertificationError,
    compose_uk_release_certification,
    rehydrate_uk_fit_weight_records,
    run_uk_release_cut_battery,
    uk_release_cut_scope_exclusions,
)


def _load_fixture_module():
    path = Path(__file__).with_name("uk_certification_fixtures.py")
    spec = importlib.util.spec_from_file_location("uk_certification_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_FIXTURES = _load_fixture_module()
_TEST_KEY = _FIXTURES.TEST_KEY
_green_certification_inputs_fixture = _FIXTURES.green_certification_inputs
_signing_key_fixture = _FIXTURES.signing_key
_sha = _FIXTURES.sha256
_stub_registry = _FIXTURES.stub_registry

_SHARED_FIXTURES = (
    _green_certification_inputs_fixture,
    _signing_key_fixture,
)


def test_scope_exclusions_cover_every_non_national_gate():
    exclusions = uk_release_cut_scope_exclusions()
    declared = {entry.id for entry in load_country_spec("uk").gates.gates}
    assert set(exclusions) | set(UK_NATIONAL_GATE_SCOPE) == declared
    assert not set(exclusions) & set(UK_NATIONAL_GATE_SCOPE)
    assert all(exclusions.values())


def test_rehydrate_fit_weight_records():
    assert rehydrate_uk_fit_weight_records({}) is None
    records = rehydrate_uk_fit_weight_records(
        {
            "fit_weight_records": {
                "was_wealth": [
                    {
                        "fit_name": "uk_was_2018_20_wealth:savings",
                        "weight_kind": "design",
                    }
                ],
            }
        }
    )
    assert [(r.fit_name, r.weight_kind) for r in records] == [
        ("uk_was_2018_20_wealth:savings", "design")
    ]
    # A fitting stage that recorded nothing coerces the whole artifact to
    # (), which the weights-audit binding fails — never a vacuous pass.
    assert (
        rehydrate_uk_fit_weight_records({"fit_weight_records": {"was_wealth": []}})
        == ()
    )
    # A malformed block is corruption, not an empty audit: it raises rather
    # than degrading to (), keeping "no weights" and "weights we could not
    # read" distinguishable.
    with pytest.raises(UKReleaseCertificationError, match="unreadable block"):
        rehydrate_uk_fit_weight_records(
            {"fit_weight_records": {"was_wealth": "not-a-list"}}
        )
    with pytest.raises(UKReleaseCertificationError, match="malformed record"):
        rehydrate_uk_fit_weight_records(
            {"fit_weight_records": {"was_wealth": [{"fit_name": "x"}]}}
        )
    with pytest.raises(UKReleaseCertificationError, match="malformed record"):
        rehydrate_uk_fit_weight_records(
            {"fit_weight_records": {"was_wealth": ["not-a-mapping"]}}
        )


def test_parity_evidence_refuses_bare_name_grain():
    # Both parity sides share the name@period grain; a diagnostics row
    # missing the period label refuses loudly instead of silently falling
    # out of the comparison.
    from microcosm.build.uk_runtime.release_certification import (
        uk_release_parity_evidence,
    )

    class _Frame:
        entities = ()

        def table(self, entity):  # pragma: no cover - never reached
            raise AssertionError

    class _Registry:
        specs = ()

    class _Reference:
        input_entities = {}

    with pytest.raises(UKReleaseCertificationError, match="name@period"):
        uk_release_parity_evidence(
            _Frame(),
            diagnostics_targets=[{"name": "dwp.uc.households", "relative_error": 0.1}],
            reference_registry=_Registry(),
            parity_reference=_Reference(),
        )
    evidence = uk_release_parity_evidence(
        _Frame(),
        diagnostics_targets=[{"name": "dwp.uc.households@2025", "relative_error": 0.1}],
        reference_registry=_Registry(),
        parity_reference=_Reference(),
    )
    assert evidence.candidate_targets == {"dwp.uc.households@2025"}


def test_compose_green_certification(green_certification_inputs):
    certification = compose_uk_release_certification(**green_certification_inputs)
    assert certification["shippable"] is True
    assert certification["kind"] == "uk_release_certification"
    assert set(certification["parts"]) == {"spine", "calibration_seam", "release_cut"}
    declared = {entry.id for entry in load_country_spec("uk").gates.gates}
    union = set()
    for part in certification["parts"].values():
        union.update(part["entry_ids"])
    assert union == declared
    assert certification["spec"]["shared_gate_ids"] == sorted(UK_SHARED_GATE_IDS)
    assert certification["doctrine"]["overrides"] == {
        "epochs": {"default": 256, "effective": 1500}
    }
    written = json.loads(
        green_certification_inputs["certification_path"].read_text(encoding="utf-8")
    )
    assert written["attestation"]["signature"]
    # The certification's own signature verifies under the release key.
    import hmac as hmac_module

    from microcosm.build.logbook import canonical_json_bytes

    unsigned = json.loads(json.dumps(written))
    unsigned["attestation"]["signature"] = None
    recomputed = hmac_module.new(
        base64.b64decode(_TEST_KEY), canonical_json_bytes(unsigned), hashlib.sha256
    ).hexdigest()
    assert recomputed == written["attestation"]["signature"]


def test_compose_refuses_tampered_part_signature(green_certification_inputs):
    seam_path = green_certification_inputs["seam_report_path"]
    payload = json.loads(seam_path.read_text(encoding="utf-8"))
    payload["release_id"] = "dev-seam-tampered"
    seam_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Keep the build record's byte pin in step so the signature check is
    # the refusal that fires, not the identity join.
    green_certification_inputs["build_record"]["artifacts"]["terminal_gate_json"][
        "sha256"
    ] = _sha(seam_path)
    with pytest.raises(UKReleaseCertificationError, match="signature"):
        compose_uk_release_certification(**green_certification_inputs)


def test_compose_refuses_entry_id_gap(green_certification_inputs):
    cut_path = green_certification_inputs["release_cut_report_path"]
    payload = json.loads(cut_path.read_text(encoding="utf-8"))
    payload["gates"].pop("uk_qrf_tail_concentration")
    cut_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(UKReleaseCertificationError, match="entry ids"):
        compose_uk_release_certification(**green_certification_inputs)


def test_compose_refuses_blocked_part(green_certification_inputs):
    spine_path = green_certification_inputs["spine_report_path"]
    payload = json.loads(spine_path.read_text(encoding="utf-8"))
    payload["blocked_at_phase"] = "transferred"
    spine_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    green_certification_inputs["spine_sidecar"]["spine_gate_report"]["sha256"] = _sha(
        spine_path
    )
    green_certification_inputs["build_record"]["spine_provenance"]["spine_gate_report"][
        "sha256"
    ] = _sha(spine_path)
    with pytest.raises(UKReleaseCertificationError, match="blocked"):
        compose_uk_release_certification(**green_certification_inputs)


def test_compose_refuses_failing_release_blocking_entry(green_certification_inputs):
    cut_path = green_certification_inputs["release_cut_report_path"]
    payload = json.loads(cut_path.read_text(encoding="utf-8"))
    payload["gates"]["uk_support"]["status"] = "failed"
    payload["gates"]["uk_support"]["failures"] = ["synthetic failure"]
    cut_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(UKReleaseCertificationError, match="release-blocking"):
        compose_uk_release_certification(**green_certification_inputs)


def test_compose_refuses_sidecar_report_mismatch(green_certification_inputs):
    green_certification_inputs["spine_sidecar"]["spine_gate_report"]["sha256"] = (
        "0" * 64
    )
    with pytest.raises(UKReleaseCertificationError, match="sidecar"):
        compose_uk_release_certification(**green_certification_inputs)


def test_compose_refuses_candidate_mismatch(green_certification_inputs):
    green_certification_inputs["build_record"]["artifacts"]["staging_h5"]["sha256"] = (
        "0" * 64
    )
    with pytest.raises(UKReleaseCertificationError, match="staged a different"):
        compose_uk_release_certification(**green_certification_inputs)


def test_compose_refuses_diagnostics_divergence(green_certification_inputs):
    green_certification_inputs["build_record"]["artifacts"]["diagnostics_json"][
        "sha256"
    ] = "0" * 64
    with pytest.raises(UKReleaseCertificationError, match="diagnostics"):
        compose_uk_release_certification(**green_certification_inputs)


def test_compose_refuses_foreign_release_id(green_certification_inputs):
    green_certification_inputs["release_id"] = "uk-some-other-cut"
    with pytest.raises(UKReleaseCertificationError, match="release id"):
        compose_uk_release_certification(**green_certification_inputs)


def test_compose_refuses_unpinned_score_receipt(green_certification_inputs):
    green_certification_inputs["score_receipt_path"].write_text(
        '{"verdict": "scored"}', encoding="utf-8"
    )
    with pytest.raises(UKReleaseCertificationError, match="score receipt"):
        compose_uk_release_certification(**green_certification_inputs)


def test_compose_refuses_score_receipt_scored_on_another_artifact(
    green_certification_inputs,
):
    # The candidate digest appearing elsewhere in the document (an inputs
    # list, a provenance pin) must not satisfy the cross-pin: only
    # artifacts.candidate.sha256 names what was scored.
    candidate_sha = green_certification_inputs["candidate_sha256"]
    green_certification_inputs["score_receipt_path"].write_text(
        json.dumps(
            {
                "artifacts": {
                    "candidate": {"sha256": "8" * 64},
                    "incumbent": {"sha256": candidate_sha},
                },
                "provenance": {"inputs": [candidate_sha]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(UKReleaseCertificationError, match="artifacts.candidate.sha256"):
        compose_uk_release_certification(**green_certification_inputs)


def test_compose_refuses_absent_signing_key(green_certification_inputs, monkeypatch):
    monkeypatch.delenv(gate_signing_key_env("uk"))
    with pytest.raises(UKReleaseCertificationError, match="must be set"):
        compose_uk_release_certification(**green_certification_inputs)


def test_release_cut_battery_runs_and_signs(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        release_certification,
        "uk_aggregate_admin_totals",
        lambda frame, manifest: ({}, {"stub": True}),
    )
    report_path = tmp_path / "release_cut_gates.json"
    payload = run_uk_release_cut_battery(
        object(),
        report_path=report_path,
        release_id="uk-757-first-certified-cut",
        diagnostics_sha256="a" * 64,
        coverage_engine=object(),
        build_stage_names=("frs_spine",),
        ledger_registries={2023: object(), 2025: object()},
        parity_evidence=object(),
        fit_weight_records=None,
        input_mass_reference={},
        exclusions_evaluated_on=date(2026, 8, 27),
        gate_registry=_stub_registry(),
    )
    assert payload["posture"] == "release_cut"
    assert payload["release_candidate"] is True
    assert payload["shippable"] is True
    assert set(payload["gates"]) == set(UK_NATIONAL_GATE_SCOPE)
    assert payload["blocked_at_phase"] is None
    assert set(payload["scope_exclusions"]) == (
        set(UK_SPINE_GATE_SCOPE) | set(UK_CALIBRATION_GATE_SCOPE)
    ) - set(UK_NATIONAL_GATE_SCOPE)
    on_disk = json.loads(report_path.read_text(encoding="utf-8"))
    assert on_disk["attestation"]["signature"] == payload["attestation"]["signature"]
