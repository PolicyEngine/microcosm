# ruff: noqa: F401
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.chronicle_epoch import (
    CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION,
    LEDGER_CONSUMER_FACT_SCHEMA_VERSION,
    PUBLISHED_CONSUMER_ARTIFACT_SCHEMA_VERSION,
)
from microcosm.build.country_spec import load_country_spec
from microcosm.build.gate_battery import (
    _canonical_json_bytes as canonical_json_bytes,
)
from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.staging_v2 import StagingReadBackError
from microcosm.build.uk_runtime import calibration_run
from microcosm.build.uk_runtime.calibration_run import (
    UK_CALIBRATION_GATE_SCOPE,
    UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS,
    UK_SPINE_GATE_SCOPE,
    UKCalibrationRunPaths,
    run_uk_calibration,
)
from microcosm.build.uk_runtime.etb_services import (
    UK_NHS_SPENDING_COMPONENT_COLUMNS,
)
from microcosm.build.uk_runtime.national_doctrine import UKNationalSolveDoctrine
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    uk_household_weight_kind,
    uk_national_frame,
    write_uk_national_frame,
)
from microcosm.calibrate import (
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyGeography,
    HierarchyNode,
    TargetRegistry,
    TargetSpec,
)
from microcosm.frame import WeightKind
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

SIGNING_KEY = "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY="


@pytest.fixture(autouse=True)
def _signing_key(monkeypatch):
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", SIGNING_KEY)


def _fake_cgt_projection(base_year: int = 2023, horizon_year: int = 2030):
    """A flat 3 percent growth path against a frozen 3,000 exempt amount."""

    from microcosm.build.uk_runtime.cgt_projection import (
        UK_CGT_EXEMPT_AMOUNT_PARAMETER,
        UK_CGT_GAINS_GROWTH_PARAMETER,
        UKCGTProjection,
    )

    # The manifest pins the engine's growth path; the fake must sit on it or
    # the binding's drift check fails the validation check.
    parameters = next(
        entry
        for entry in load_country_spec("uk").gates.gates
        if entry.id == "uk_cgt_projection_entrants"
    ).parameters
    pinned = parameters["expected_yoy_growth_by_year"]
    pinned_exempt = parameters["expected_exempt_amount_by_year"]
    growth: dict[str, float] = {}
    cumulative: dict[str, float] = {}
    factor = 1.0
    for year in range(base_year + 1, horizon_year + 1):
        rate = float(pinned[str(year)])
        factor *= 1.0 + rate
        growth[str(year)] = rate
        cumulative[str(year)] = factor
    return UKCGTProjection(
        base_year=base_year,
        horizon_year=horizon_year,
        growth_parameter=UK_CGT_GAINS_GROWTH_PARAMETER,
        exempt_amount_parameter=UK_CGT_EXEMPT_AMOUNT_PARAMETER,
        yoy_growth_by_year=growth,
        cumulative_gains_factor_by_year=cumulative,
        exempt_amount_by_year={
            str(year): float(pinned_exempt[str(year)])
            for year in range(base_year, horizon_year + 1)
        },
        engine="test",
    )


@pytest.fixture(autouse=True)
def _cgt_projection(monkeypatch):
    """The seam reads the projection from the engine; tests supply a flat one."""

    monkeypatch.setattr(
        calibration_run,
        "uk_cgt_projection_artifact",
        lambda frame, manifest: _fake_cgt_projection(),
    )


def _frame():
    ids = np.arange(4, dtype="int64")
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
                "nhs_spending": [50.0, 50.0, 50.0, 50.0],
                "capital_gains": [0.0, 0.0, 0.0, 0.0],
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


def _uc_hierarchy() -> CalibrationHierarchy:
    return CalibrationHierarchy(
        provider=HierarchyNode(
            id="dwp",
            label="Department for Work and Pensions",
        ),
        category=HierarchyCategory(
            id="dwp.universal_credit",
            label="Universal Credit",
            provider_id="dwp",
        ),
        geography=HierarchyGeography(
            id="K03000001",
            label="Great Britain",
            level="country",
        ),
        dimensions=(),
        target=HierarchyNode(
            id="dwp.uc.households",
            label="Universal Credit households",
        ),
    )


def _registry():
    return TargetRegistry(
        [
            TargetSpec(
                name="dwp.uc.households",
                entity="benunit",
                measure="dwp/uc/households",
                value=20.0,
                source="test",
                family="dwp_universal_credit",
                metadata={"contract_target_id": "dwp.uc.households"},
                hierarchy=_uc_hierarchy(),
            )
        ],
        country="uk",
    )


def _paths(tmp_path: Path) -> UKCalibrationRunPaths:
    return UKCalibrationRunPaths(
        input_h5=tmp_path / "input.h5",
        staging_h5=tmp_path / "staged.h5",
        diagnostics_json=tmp_path / "diagnostics.json",
        build_record_json=tmp_path / "build_record.json",
        terminal_gate_json=tmp_path / "terminal_gates.json",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_spine_sidecar(
    input_h5: Path,
    frame=None,
    **overrides,
) -> dict[str, object]:
    frame = _frame() if frame is None else frame
    sidecar = {
        "schema_version": 2,
        "pipeline": "uk-frs-spine",
        "stages": ["frs_spine", "was_wealth"],
        "stage_records": [
            {
                "stage": "was_wealth",
                "produced": ["property_wealth"],
                "nonzero_share": {"property_wealth": 1.0},
                "seconds": 0.1,
            }
        ],
        "stage_evidence": {
            "was_wealth": {
                "stage": "was_wealth",
                "support_clip": {"columns": {}},
            }
        },
        "artifact_pins": {"person": "a" * 64},
        "input_artifact_pins": {"was_qrf_donor": {"sha256": "b" * 64}},
        "resource_pins": {"wealth.json": "c" * 64},
        "stage_artifact_pins": {"was_wealth": {"was_qrf_donor": "d" * 64}},
        "declared_seeds": {"was_wealth": {"was_wealth": 0}},
        "rules_engine": {"package": "policyengine-uk", "version": "unavailable"},
        "source_vintages": {"frs": "2024_25"},
        "stochastic_contract_sha256": "e" * 64,
        "entity_row_counts": {
            entity: int(len(frame.table(entity))) for entity in frame.entities
        },
        "household_weight_kind": uk_household_weight_kind(frame).value,
        "household_weight_total": float(frame.weights_for("household").values.sum()),
    }
    sidecar.update(overrides)
    input_h5.with_suffix(".build.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    input_h5.with_suffix(".spine_gates.json").write_text(
        json.dumps(
            {
                "blocked_at_phase": None,
                "gates": {
                    gate_id: {
                        "criticality": "release_blocking",
                        "status": "passed",
                    }
                    for gate_id in UK_SPINE_GATE_SCOPE
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return sidecar


def _admin_anchor_values():
    values = {}
    for entry in load_country_spec("uk").gates.gates:
        if entry.id == "uk_aggregate_admin":
            for anchor in entry.parameters["anchors"]:
                values[str(anchor["name"])] = float(anchor["value"])
    return values


def _manifest_with_small_anchor():
    """The committed manifest plus one small (mean-convention) anchor.

    The NEED mean-spend anchors left uk_aggregate_admin under microcosm#890
    (they are checked at stage time now), so the carrier-mean convention is
    exercised with a synthetic small anchor on a frame column.
    """

    committed = calibration_run._calibration_gate_manifest()
    gates = []
    for gate in committed.gates:
        if gate.id == "uk_aggregate_admin":
            anchors = [
                *gate.parameters["anchors"],
                {
                    "name": "electricity_mean_spending",
                    "entity": "household",
                    "measure": "electricity_consumption",
                    "value": 1.0,
                    "period": "2024",
                    "source": "test",
                    "family": "test",
                },
            ]
            gates.append(
                SimpleNamespace(
                    id=gate.id, parameters={**gate.parameters, "anchors": anchors}
                )
            )
        else:
            gates.append(gate)
    return SimpleNamespace(gates=gates)


def _load_logbook_tool():
    import importlib.util

    path = _TEST_PATHS.repository / "tools" / "logbook.py"
    spec = importlib.util.spec_from_file_location("_logbook_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


__all__ = [name for name in globals() if not name.startswith("__")]
