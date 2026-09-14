from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.chronicle_epoch import (
    CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION,
    LEDGER_CONSUMER_FACT_SCHEMA_VERSION,
    PUBLISHED_CONSUMER_ARTIFACT_SCHEMA_VERSION,
)
from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.uk_runtime import calibration_run
from microcosm.build.uk_runtime.calibration_run import (
    UK_CALIBRATION_GATE_SCOPE,
    UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS,
)
from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.etb_services import (
    UK_NHS_SPENDING_COMPONENT_COLUMNS,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
)
from microcosm.frame import WeightKind

SIGNING_KEY = "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY="


@pytest.fixture(autouse=True)
def _signing_key(monkeypatch):
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", SIGNING_KEY)


def _frame():
    ids = np.arange(4, dtype="int64")
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
                "nhs_spending": [50.0, 50.0, 50.0, 50.0],
            }
        ),
        benunit=pd.DataFrame(
            {"benunit_id": ids, "universal_credit": [1.0, 1.0, 0.0, 0.0]}
        ),
        household=pd.DataFrame(
            {
                "household_id": ids,
                "region": "LONDON",
                "household_weight": [10.0, 10.0, 10.0, 10.0],
                "household_is_spi_synthetic": [False, False, False, False],
                "household_is_capital_gains_clone": [False, False, False, False],
                "electricity_consumption": [1.0, 1.0, 1.0, 1.0],
                "gas_consumption": [1.0, 1.0, 1.0, 1.0],
            }
        ),
        time_period="2023",
        weight_kind=WeightKind.DESIGN,
    )


def _bound_checkpoint(tmp_path, frame):
    report_path = tmp_path / "spine.spine_gates.json"
    report = {
        **calibration_run.uk_spine_checkpoint_gate_digests(),
        "blocked_at_phase": None,
        "gates": {
            entry.id: {"status": "passed", "criticality": entry.criticality}
            for entry in load_country_spec("uk").gates.gates
            if entry.id in calibration_run.UK_SPINE_GATE_SCOPE
        },
    }
    report_path.write_text(json.dumps(report))
    sidecar = {
        "entity_row_counts": {
            entity: len(frame.table(entity)) for entity in frame.entities
        },
        "household_weight_kind": frame.weights_for("household").kind.value,
        "household_weight_total": float(frame.weights_for("household").values.sum()),
        "uk_frame_content_identity": uk_frame_content_identity(frame),
        "spine_gate_report": {
            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()
        },
        "fit_weight_records": {"model": {"fit_weights_used": True}},
    }
    sidecar_path = tmp_path / "spine.build.json"
    sidecar_path.write_text(json.dumps(sidecar))
    return sidecar_path, report_path, sidecar


def test_strict_checkpoint_binds_contents_and_retains_gate_payload(tmp_path):
    frame = _frame()
    path, gate_path, _ = _bound_checkpoint(tmp_path, frame)
    sidecar = calibration_run.load_bound_spine_checkpoint(path, frame)
    provenance = calibration_run.strict_spine_provenance_from_sidecar(path, sidecar)
    assert provenance["fit_weight_records"] == sidecar["fit_weight_records"]
    assert provenance["spine_gate_report"]["payload"] == json.loads(
        gate_path.read_bytes()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_identity",
        "wrong_identity",
        "bypass",
        "gate_bytes",
        "missing_gate_binding",
        "gate_roster",
        "gate_policy",
    ],
)
def test_strict_checkpoint_rejects_unbound_or_changed_evidence(tmp_path, mutation):
    frame = _frame()
    path, gate_path, sidecar = _bound_checkpoint(tmp_path, frame)
    if mutation == "missing_identity":
        sidecar.pop("uk_frame_content_identity")
    elif mutation == "wrong_identity":
        sidecar["uk_frame_content_identity"] = "f" * 64
    elif mutation == "bypass":
        sidecar["spine_gate_bypass"] = {"reviewed": True, "reason": "historical"}
    elif mutation == "gate_bytes":
        gate_path.write_text(gate_path.read_text() + "\n")
    elif mutation == "missing_gate_binding":
        sidecar.pop("spine_gate_report")
    elif mutation == "gate_policy":
        report = json.loads(gate_path.read_bytes())
        report["policy_sha256"] = "f" * 64
        gate_path.write_text(json.dumps(report))
        sidecar["spine_gate_report"]["sha256"] = hashlib.sha256(
            gate_path.read_bytes()
        ).hexdigest()
    else:
        report = json.loads(gate_path.read_bytes())
        report["gates"].pop(next(iter(report["gates"])))
        gate_path.write_text(json.dumps(report))
        sidecar["spine_gate_report"]["sha256"] = hashlib.sha256(
            gate_path.read_bytes()
        ).hexdigest()
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError):
        calibration_run.load_bound_spine_checkpoint(path, frame)


def test_strict_checkpoint_accepts_explicit_declared_gate_path(tmp_path):
    frame = _frame()
    path, gate_path, _ = _bound_checkpoint(tmp_path, frame)
    moved = gate_path.rename(tmp_path / "declared-gates.json")
    sidecar = calibration_run.load_bound_spine_checkpoint(
        path, frame, gate_report_path=moved
    )
    provenance = calibration_run.strict_spine_provenance_from_sidecar(
        path, sidecar, gate_report_path=moved
    )
    assert provenance["spine_gate_report"]["path"] == str(moved)


def test_gate_scope_classifies_every_uk_gate():
    all_ids = {entry.id for entry in load_country_spec("uk").gates.gates}
    assert (
        set(UK_CALIBRATION_GATE_SCOPE) | set(UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS)
        == all_ids
    )
    assert set(UK_CALIBRATION_GATE_SCOPE).isdisjoint(
        UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS
    )
    assert all(UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS.values())


def test_import_hygiene_does_not_load_national_build_in_fresh_subprocess():
    source = Path(calibration_run.__file__).read_text(encoding="utf-8")
    legacy_module = ".".join(("microcosm", "build", "uk_runtime", "national_build"))
    assert legacy_module not in source
    assert " ".join(("from", legacy_module, "import")) not in source


def test_aggregate_admin_measurement_convention_and_refusals():
    frame = _frame()
    manifest = calibration_run._calibration_gate_manifest()

    totals, receipt = calibration_run.uk_aggregate_admin_totals(frame, manifest)

    # Small anchors (NEED means) measure as the weighted mean over carriers;
    # the NHS total measures as the person total under mapped household
    # weights: 4 persons x 50.0 x weight 10.0.
    assert totals["need_electricity_mean_spending"] == pytest.approx(1.0)
    assert totals["need_gas_mean_spending"] == pytest.approx(1.0)
    assert totals["nhs_spending_total"] == pytest.approx(2000.0)
    by_anchor = {row["anchor"]: row for row in receipt}
    assert by_anchor["nhs_spending_total"]["entity"] == "person"
    assert (
        by_anchor["need_electricity_mean_spending"]["statistic_convention"]
        == "assessed_by_anchor_magnitude"
    )

    stripped = _frame()
    stripped.table("household").drop(columns=["electricity_consumption"], inplace=True)
    with pytest.raises(ValueError, match="household.electricity_consumption"):
        calibration_run.uk_aggregate_admin_totals(stripped, manifest)


def test_nhs_anchor_composes_from_the_columns_the_spine_actually_carries():
    """The anchor is published as one total; the spine carries it in three parts.

    Composing is the translation from the published concept to ours, and the
    receipt has to say so — the anchor measured a silent zero for as long as it
    named a column no stage produces.
    """

    frame = _frame()
    person = frame.table("person")
    person.drop(columns=["nhs_spending"], inplace=True)
    person["nhs_a_and_e_spending"] = [20.0, 20.0, 20.0, 20.0]
    person["nhs_admitted_patient_spending"] = [25.0, 25.0, 25.0, 25.0]
    person["nhs_outpatient_spending"] = [5.0, 5.0, 5.0, 5.0]
    manifest = calibration_run._calibration_gate_manifest()

    totals, receipt = calibration_run.uk_aggregate_admin_totals(frame, manifest)

    # Same 4 persons x 50.0 x weight 10.0 as the single-column fixture.
    assert totals["nhs_spending_total"] == pytest.approx(2000.0)
    by_anchor = {row["anchor"]: row for row in receipt}
    assert by_anchor["nhs_spending_total"]["composed_from"] == list(
        UK_NHS_SPENDING_COMPONENT_COLUMNS
    )
    assert by_anchor["need_gas_mean_spending"]["composed_from"] == []


def test_partly_carried_derived_anchor_refuses_and_names_the_missing_part():
    frame = _frame()
    person = frame.table("person")
    person.drop(columns=["nhs_spending"], inplace=True)
    person["nhs_a_and_e_spending"] = [20.0, 20.0, 20.0, 20.0]
    manifest = calibration_run._calibration_gate_manifest()

    with pytest.raises(ValueError, match="nhs_admitted_patient_spending"):
        calibration_run.uk_aggregate_admin_totals(frame, manifest)


def _consumer_fact_row(
    *,
    aggregate_fact_key: str,
    semantic_fact_key: str,
    schema_version: str,
    source_release_key: str | None = None,
) -> dict:
    row = {
        "aggregate_fact_key": aggregate_fact_key,
        "semantic_fact_key": semantic_fact_key,
        "schema_version": schema_version,
        "value": 1.0,
        "period": {"type": "tax_year", "value": 2025},
        "geography": {"level": "country", "id": "K02000001"},
        "entity": {"name": "household"},
        "aggregation": {"method": "sum"},
        "observed_measure": {"source_name": "ons", "unit": "gbp"},
        "source": {"source_name": "ons"},
        "lineage": {"source_record_id": "ons.2025.total"},
    }
    if source_release_key is not None:
        row["source_release_key"] = source_release_key
    return row


def _mixed_epoch_artifact_dir(tmp_path: Path) -> Path:
    """A cutover-window feed: ledger-era history, chronicle-era rows, and one
    Chronicle-namespace key spelling nothing declares.

    The manifest declares what Chronicle's ``main`` stamps today
    (``policyengine_ledger.consumer_artifact.v2``), so this is the shape a UK
    run is handed now, not a hypothetical one.
    """
    rows = [
        _consumer_fact_row(
            aggregate_fact_key="ledger.aggregate_fact.v2:aaa",
            semantic_fact_key="ledger.semantic_fact.v2:aaa",
            schema_version=LEDGER_CONSUMER_FACT_SCHEMA_VERSION,
        ),
        _consumer_fact_row(
            aggregate_fact_key="chronicle.aggregate_fact.v3:bbb",
            semantic_fact_key="chronicle.semantic_fact.v3:bbb",
            schema_version=CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION,
        ),
        _consumer_fact_row(
            aggregate_fact_key="ledger.aggregate_fact.v2:ccc",
            semantic_fact_key="ledger.semantic_fact.v2:ccc",
            schema_version=LEDGER_CONSUMER_FACT_SCHEMA_VERSION,
            source_release_key="chronicle.source_release.v9:undeclared",
        ),
    ]
    artifact_dir = tmp_path / "chronicle-artifact"
    artifact_dir.mkdir()
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    (artifact_dir / "consumer_facts.jsonl").write_text(payload)
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": PUBLISHED_CONSUMER_ARTIFACT_SCHEMA_VERSION,
                "artifact_id": "chronicle-uk-artifact-mixed",
                "profile": "uk-national",
                "fact_row_count": len(rows),
                "facts_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return artifact_dir


def test_the_uk_block_delegates_rather_than_reassembling_the_shared_one(tmp_path):
    """Every field of the shared provenance block reaches the UK block.

    Pinned as a property, not as a list: the failure being prevented is a
    field the loader learns and this seam silently drops, and a test that
    enumerated today's fields would not catch tomorrow's.
    """
    artifact = load_ledger_consumer_artifact(_mixed_epoch_artifact_dir(tmp_path))
    shared = artifact.provenance()

    block = calibration_run._ledger_provenance(artifact)

    for field, value in shared.items():
        if field in {"path_name", "schema_version", "profiles"}:
            # Not identity of the feed: the directory name is local, and the
            # manifest id is carried in the narrower manifest sub-block.
            continue
        assert block[field] == value, field
    assert block["manifest"]["schema_version"] == shared["schema_version"]
