"""One full graph certifies its own scope and bytes without signed lane joins."""

import hashlib
import json
import sys

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.gate_battery import (
    GateOutcome,
    GatePhaseReport,
    GateStatus,
    gate_phase_report_payload,
)
from microcosm.build.gates import GateResult
from microcosm.build.uk_runtime import full_certification as runtime
from microcosm.build.uk_runtime.full_gates import (
    classify_full_gate_outcomes,
    uk_full_gate_manifest,
    uk_full_gate_scope_receipt,
)
from microcosm.build.uk_runtime.graph_evidence import uk_spine_gate_manifest
from microcosm.graph.canonical import canonical_json


@pytest.fixture(scope="module", autouse=True)
def _one_validated_country_spec():
    """These artifact tests share immutable declarations; source loading is separate."""
    from microcosm.build.uk_runtime import full_gates

    spec = load_country_spec("uk")
    with pytest.MonkeyPatch.context() as patch:
        for module in (runtime, full_gates, sys.modules[__name__]):
            patch.setattr(module, "load_country_spec", lambda country: spec)
        yield


def _passed(gates, phase):
    return GatePhaseReport(
        phase,
        tuple(
            GateOutcome(entry, GateStatus.PASSED, GateResult(entry.gate, True, (), {}))
            for entry in gates.gates
            if entry.phase == phase
        ),
    )


def _fixture(*, country=False, k=None):
    selection = {
        "schema": "microcosm.calibrate.target-selection.v1",
        "selector": {"geography_levels": ["country"] if country else None},
        "included": [
            {"name": "national", "period": 2025, "geography_level": "country"}
        ],
        "excluded": [],
    }
    if not country:
        selection["included"].append(
            {"name": "local", "period": 2025, "geography_level": "constituency"}
        )
    names = set(runtime._REQUIRED) | {"spine_assembled", "spine_transferred"}
    keys = {name: hashlib.sha256(name.encode()).hexdigest() for name in names}
    gate_manifest = uk_full_gate_manifest(selection)
    documents = {
        "selection": {"receipt": selection},
        "diagnostics": {"targets": []},
        "holdout": {
            "skipped": False,
            "method": "rotated_folds",
            "n_folds": 5,
            "folds": [{"holdout_loss": 0.01}] * 5,
            "mean_holdout_loss": 0.01,
            "worst_holdout_loss": 0.01,
        },
    }
    for role, phase in (("preflight", "preflight"), ("full_gates", "terminal")):
        report = _passed(gate_manifest, phase)
        documents[role] = {
            "schema_version": 1,
            "kind": "uk_full_gate_report",
            "selection_receipt": selection,
            "sample_fraction": 1.0,
            "release_candidate": False,
            "scope": uk_full_gate_scope_receipt(selection),
            "report": gate_phase_report_payload(report, gates=gate_manifest),
            "enforcement": classify_full_gate_outcomes(
                report, sample_fraction=1.0, release_candidate=False
            ),
            "artifacts": {
                name: keys[name]
                for name in ("surface", "selection", "preflight", "holdout")
            },
        }
    spine = uk_spine_gate_manifest(load_country_spec("uk"))
    for phase in ("assembled", "transferred"):
        documents[f"spine_{phase}"] = gate_phase_report_payload(
            _passed(spine, phase), gates=spine
        )
    bindings = {"configuration": {"calibration": {"dataset_households": k}}}
    dataset = {"filename": "candidate.h5", "sha256": "d" * 64, "size_bytes": 123}
    documents["export_descriptor"] = {
        "kind": "uk_full_build_export",
        "schema_version": 1,
        "content_sha256": "a" * 64,
        "bindings": bindings,
        "tables": {"household": {"rows": k or 12}},
    }
    documents["export_readback"] = {
        "kind": "uk_full_build_export_readback",
        "passed": True,
        "dataset": dataset,
        "content_sha256": "a" * 64,
        "bindings": bindings,
    }
    documents["package"] = {
        "kind": "uk_full_build_package",
        "readback_passed": True,
        "release_authorized": False,
        "schema_version": 1,
        "dataset": dataset,
        "content_sha256": "a" * 64,
        "build_bindings": bindings,
        "artifacts": {
            name: keys[name]
            for name in ("full_gates", "diagnostics", "holdout", "export_readback")
        },
    }
    documents["surface"] = {
        "source_validation": {
            "ledger_provenance": {"facts_sha256": "b" * 64, "manifest_sha256": "c" * 64}
        }
    }
    return {
        name: (keys[name], canonical_json(document))
        for name, document in documents.items()
    }


def _rewrite(artifacts, name, mutate):
    key, payload = artifacts[name]
    document = json.loads(payload)
    mutate(document)
    artifacts[name] = (key, canonical_json(document))


def _scores(artifacts, monkeypatch, *, k=None):
    def validate(payload, failures, *, expected_identity):
        if payload.get("identity") != expected_identity:
            failures.append("scorecard candidate identity mismatch")

    monkeypatch.setattr(
        "microcosm.data.contract._check_uk_incumbent_surface_evaluation", validate
    )
    identity = {
        "candidate_dataset_sha256": "d" * 64,
        "candidate_manifest_sha256": hashlib.sha256(
            artifacts["package"][1]
        ).hexdigest(),
        "candidate_diagnostics_sha256": hashlib.sha256(
            artifacts["diagnostics"][1]
        ).hexdigest(),
        "ledger_facts_sha256": "b" * 64,
        "ledger_manifest_sha256": "c" * 64,
    }
    sources = {}
    for role in (
        ("native_scorecard", "matched_size_scorecard") if k else ("native_scorecard",)
    ):
        document = {"identity": identity}
        if role == "matched_size_scorecard":
            document["comparison"] = {
                "kind": "matched_size",
                "candidate_households": k,
                "incumbent_households": k,
            }
        payload = canonical_json(document)
        sources[role] = (
            {
                "filename": role + ".json",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            },
            payload,
        )
    return sources


def test_complete_graph_retains_explicit_missing_native_evidence():
    report = runtime.compose_uk_full_certification_readiness(_fixture())
    assert report["comparisons"]["native_scorecard"]["status"] == "evidence_absent"
    assert report["comparisons"]["matched_size_scorecard"]["status"] == "not_required"
    assert not report["ready_for_external_review"]
    assert not report["release_authorized"]
    assert set(report["gate_coverage"]["declared"]) == {
        entry.id for entry in load_country_spec("uk").gates.gates
    }


@pytest.mark.parametrize(
    "country,k", [(False, None), (True, None), (False, 5), (True, 5)]
)
def test_bound_comparisons_can_complete_readiness_without_publication(
    monkeypatch, country, k
):
    artifacts = _fixture(country=country, k=k)
    report = runtime.compose_uk_full_certification_readiness(
        artifacts, comparison_sources=_scores(artifacts, monkeypatch, k=k)
    )
    assert report["ready_for_external_review"]
    assert not report["release_authorized"]
    assert not report["subnational_fit_certified"]
    assert report["target_scope"]["local_fit_claim"] is (not country)


@pytest.mark.parametrize(
    "name,field",
    [
        ("package", "content_sha256"),
        ("export_readback", "content_sha256"),
        ("full_gates", "selection_receipt"),
    ],
)
def test_certification_rejects_foreign_graph_identity(name, field):
    artifacts = _fixture()
    _rewrite(artifacts, name, lambda document: document.__setitem__(field, "foreign"))
    with pytest.raises((ValueError, AttributeError)):
        runtime.compose_uk_full_certification_readiness(artifacts)


def test_actual_surface_validator_refuses_old_score_summary():
    artifacts = _fixture()
    payload = canonical_json({"candidate_train_loss": 0.01})
    sources = {
        "native_scorecard": (
            {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)},
            payload,
        )
    }
    report = runtime.compose_uk_full_certification_readiness(
        artifacts, comparison_sources=sources
    )
    assert report["comparisons"]["native_scorecard"]["status"] == "failed"
    assert any(
        "schema 2" in failure
        for failure in report["comparisons"]["native_scorecard"]["failures"]
    )


def test_exact_size_requires_its_own_bound_comparison(monkeypatch):
    artifacts = _fixture(k=5)
    sources = _scores(artifacts, monkeypatch, k=5)
    sources.pop("matched_size_scorecard")
    report = runtime.compose_uk_full_certification_readiness(
        artifacts, comparison_sources=sources
    )
    assert (
        report["comparisons"]["matched_size_scorecard"]["status"] == "evidence_absent"
    )
    assert not report["ready_for_external_review"]


def test_bound_spine_provenance_replaces_raw_spine_reports():
    artifacts = _fixture()
    gates = uk_spine_gate_manifest(load_country_spec("uk"))
    report = {
        "blocked_at_phase": None,
        "gates": {
            outcome.entry.id: outcome.to_payload()
            for phase in gates.phases
            for outcome in _passed(gates, phase).outcomes
        },
    }
    provenance = {
        "uk_frame_content_identity": "e" * 64,
        "spine_gate_report": {"sha256": "f" * 64, "payload": report},
    }
    artifacts["spine_provenance"] = ("1" * 64, canonical_json(provenance))
    for phase in ("assembled", "transferred"):
        artifacts.pop(f"spine_{phase}")
    _rewrite(
        artifacts,
        "full_gates",
        lambda doc: doc["artifacts"].__setitem__("spine_provenance", "1" * 64),
    )
    result = runtime.compose_uk_full_certification_readiness(artifacts)
    assert result["artifacts"]["spine_provenance"]["graph_artifact_key"] == "1" * 64
