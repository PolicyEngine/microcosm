"""US release input-column coverage + reform-coverage smoke, isolated from PE-US.

microcosm #368. The five acceptance cases the brief pins:

1. a full required set present with signal passes;
2. a missing required column fails, named;
3. a required column present but degenerate (every value the engine default)
   fails without a reviewed exclusion;
4. a reviewed-exclusion column that has caught up (present with signal) is stale
   and fails (#286 cannot-rot);
5. a bound reform scoring ~$0 fails the reform-coverage smoke.

The frame is a real :class:`~microcosm.frame.Frame`; most tests use an engine stub
exposing only ``default_values`` (the surface the gate uses), and the simulation is
injected (with ``_build_reform`` monkeypatched). One optional regression reproduces
the nullable-boolean failure against the real PolicyEngine-US defaults and full US
entity schema. Separate tests assert the shipped manifest keeps the #368
red-by-design guarantee: the SSI countable-resource assets stay hard requirements
with no exclusion, and demoting one is rejected.
"""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.reform_coverage_smoke as smoke_module
from microcosm.build.us_runtime import (
    CPS_CARRIED_PERSON_INPUTS,
    CPS_CARRIED_SPM_UNIT_INPUTS,
    POST_REFERENCE_ECPS_REQUIRED_INPUTS,
    SSI_COUNTABLE_RESOURCE_ASSETS,
    US_ASEC_REPORTED_RECEIPT_REQUIRED_INPUTS,
    US_CGD_ROUTE_REQUIRED_INPUTS,
    US_QBI_OUTPUT_COLUMNS,
    US_RELEASE_INPUT_COVERAGE_RESOURCE,
    ReformCoverageProbe,
    ReleaseInputColumn,
    ReleaseInputCoverageManifest,
    assert_release_input_coverage_manifest_current,
    load_release_input_coverage_manifest,
    us_reform_coverage_smoke_gate,
    us_release_input_coverage_gate,
    us_release_reform_coverage_probes,
)
from microcosm.build.us_runtime.parity_reference import load_ecps_parity_known_gaps
from microcosm.build.us_runtime.release_input_coverage import (
    REFERENCE_ECPS_LAYER_RENAMES,
    RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS,
    project_ecps_parity_known_gap_names,
)
from microcosm.frame import US_SCHEMA, EntitySchema, Frame, WeightKind, Weights
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

_REPO_ROOT = _TEST_PATHS.repository
_MANIFEST_GENERATOR = (
    _REPO_ROOT / "tools" / "build_us_release_input_coverage_manifest.py"
)


def _person_frame(columns: dict[str, np.ndarray | pd.Series]) -> Frame:
    """A real single-household Frame carrying ``columns`` on the person table."""
    n = len(next(iter(columns.values())))
    person = pd.DataFrame(
        {
            "person_id": np.arange(n, dtype="int64"),
            "person_household_id": np.ones(n, dtype="int64"),
            **columns,
        }
    )
    household = pd.DataFrame({"household_id": np.asarray([1], dtype="int64")})
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=np.asarray([1000.0]), kind=WeightKind.DESIGN)},
    )


def _us_person_frame(columns: dict[str, np.ndarray | pd.Series]) -> Frame:
    """A full six-entity US Frame preserving pandas extension dtypes."""
    n = len(next(iter(columns.values())))
    person_columns: dict[str, object] = {
        US_SCHEMA.person_id_column: np.arange(n, dtype="int64"),
        **{
            US_SCHEMA.membership_column(entity): np.ones(n, dtype="int64")
            for entity in US_SCHEMA.group_entities
        },
        **columns,
    }
    tables = {
        entity: pd.DataFrame(
            {US_SCHEMA.id_column(entity): np.asarray([1], dtype="int64")}
        )
        for entity in US_SCHEMA.group_entities
    }
    tables["person"] = pd.DataFrame(person_columns)
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.asarray([1000.0]),
                kind=WeightKind.DESIGN,
            )
        },
    )


def _household_weight_frame(
    typed_values: np.ndarray,
    *,
    stored_values: np.ndarray | None = None,
) -> Frame:
    """A Frame whose authoritative household weights may shadow a stale column."""

    typed_values = np.asarray(typed_values, dtype=np.float64)
    n = len(typed_values)
    person = pd.DataFrame(
        {
            "person_id": np.arange(n, dtype="int64"),
            "person_household_id": np.arange(1, n + 1, dtype="int64"),
        }
    )
    household = pd.DataFrame({"household_id": np.arange(1, n + 1, dtype="int64")})
    if stored_values is not None:
        household["household_weight"] = np.asarray(stored_values, dtype=np.float64)
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {
            "household": Weights(
                values=typed_values,
                kind=WeightKind.CALIBRATED,
            )
        },
    )


class _StubEngine:
    """Only ``default_values(names)`` — the single engine surface the gate uses."""

    def __init__(self, defaults: dict[str, object]) -> None:
        self._defaults = dict(defaults)

    def default_values(self, names) -> dict[str, object]:
        return {name: self._defaults[name] for name in names if name in self._defaults}


def _manifest(
    columns: tuple[ReleaseInputColumn, ...],
    probes: tuple[ReformCoverageProbe, ...] = (),
) -> ReleaseInputCoverageManifest:
    return ReleaseInputCoverageManifest(
        reference={"source": "test"}, columns=columns, probes=probes
    )


# A two-required-plus-one-excluded contract, reused across the gate cases.
_CONTRACT = _manifest(
    (
        ReleaseInputColumn("employment_income", "required"),
        ReleaseInputColumn("stock_assets", "required"),
        ReleaseInputColumn(
            "alimony_income",
            "reviewed_exclusion",
            reason="Residual income-source layer not yet sourced; tracked.",
            issue="PolicyEngine/microcosm#38",
        ),
    )
)

# Every declared column defaults to 0.0 in the stub engine, so an all-zero
# required column reads as degenerate (present but indistinguishable from absent).
_DEFAULTS = {"employment_income": 0.0, "stock_assets": 0.0, "alimony_income": 0.0}


class _Series:
    def __init__(self, total: float) -> None:
        self._total = total

    def sum(self) -> float:
        return self._total


class _Sim:
    """A simulation whose weighted total for the measure is a fixed number."""

    def __init__(self, total: float) -> None:
        self._total = total

    def calculate(self, measure: str, period):  # noqa: ARG002 - stub
        return _Series(self._total)


def _probe(min_abs_effect: float = 1_000_000_000.0) -> ReformCoverageProbe:
    return ReformCoverageProbe(
        id="ssi_probe",
        name="SSI asset limits raised to $10k / $20k",
        parameter_changes={
            "gov.ssa.ssi.eligibility.resources.limit.individual": {
                "2024-01-01.2100-12-31": 10000
            }
        },
        budget_measure="ssi",
        binding_inputs=("bank_account_assets", "stock_assets", "bond_assets"),
        min_abs_effect=min_abs_effect,
        reason="Assets absent → countable resources 0 → the relaxation scores $0.",
        issue="PolicyEngine/microcosm#356",
    )


def _tips_probe() -> ReformCoverageProbe:
    return ReformCoverageProbe(
        id="tips_probe",
        name="OBBBA no-tax-on-tips deduction",
        parameter_changes={
            "gov.irs.deductions.tip_income.cap": {"2026-01-01.2026-12-31": 0}
        },
        budget_measure="income_tax",
        binding_inputs=("tip_income", "treasury_tipped_occupation_code"),
        min_abs_effect=100_000_000.0,
        reason="The cap repeal must bind through qualified tip income.",
        issue="PolicyEngine/microcosm#38",
        effect_direction="baseline_minus_reform",
        period=2026,
        expected_sign="negative",
    )


def _overtime_probe() -> ReformCoverageProbe:
    return ReformCoverageProbe(
        id="obbba_no_tax_on_overtime",
        name="OBBBA no-tax-on-overtime deduction",
        parameter_changes={
            "gov.irs.deductions.overtime_income.cap.SINGLE": {
                "2026-01-01.2026-12-31": 0
            }
        },
        budget_measure="income_tax",
        binding_inputs=("fsla_overtime_premium",),
        min_abs_effect=100_000_000.0,
        reason="The cap repeal must bind through the FLSA overtime premium.",
        issue="PolicyEngine/microcosm#242",
        effect_direction="baseline_minus_reform",
        period=2026,
        expected_sign="negative",
    )


def _auto_loan_probe() -> ReformCoverageProbe:
    return ReformCoverageProbe(
        id="obbba_auto_loan_interest",
        name="OBBBA no-tax-on-auto-loan-interest deduction",
        parameter_changes={
            "gov.irs.deductions.auto_loan_interest.cap": {"2026-01-01.2026-12-31": 0}
        },
        budget_measure="income_tax",
        binding_inputs=("qualified_passenger_vehicle_loan_interest",),
        min_abs_effect=100_000_000.0,
        reason="The repeal must bind through qualifying vehicle-loan interest.",
        issue="PolicyEngine/microcosm#252",
        effect_direction="baseline_minus_reform",
        period=2026,
        expected_sign="negative",
    )


def _load_manifest_generator():
    spec = importlib.util.spec_from_file_location(
        "build_us_release_input_coverage_manifest", _MANIFEST_GENERATOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _us_entity_frame(
    n: int,
    person_columns: dict[str, np.ndarray | pd.Series],
    group_columns: dict[str, dict[str, np.ndarray | pd.Series]] | None = None,
) -> Frame:
    """A full six-entity US Frame with one group unit per person.

    ``group_columns`` maps a group entity (``"spm_unit"``) to the columns stored
    on that entity's own table, one row per unit, so an spm_unit-entity input
    lives where the engine and the export writer expect it.
    """
    person = pd.DataFrame(
        {
            US_SCHEMA.person_id_column: np.arange(n, dtype="int64"),
            **{
                US_SCHEMA.membership_column(entity): np.arange(1, n + 1, dtype="int64")
                for entity in US_SCHEMA.group_entities
            },
            **person_columns,
        }
    )
    tables = {
        entity: pd.DataFrame(
            {
                US_SCHEMA.id_column(entity): np.arange(1, n + 1, dtype="int64"),
                **(group_columns or {}).get(entity, {}),
            }
        )
        for entity in US_SCHEMA.group_entities
    }
    tables["person"] = person
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.full(n, 1000.0),
                kind=WeightKind.DESIGN,
            )
        },
    )


_RECEIPT_DEFAULTS = {name: False for name in US_ASEC_REPORTED_RECEIPT_REQUIRED_INPUTS}
_RECEIPT_CONTRACT = _manifest(
    tuple(
        ReleaseInputColumn(name, "required")
        for name in US_ASEC_REPORTED_RECEIPT_REQUIRED_INPUTS
    )
)


def _receipt_candidate(*, omit: str | None = None, all_default: bool = False) -> Frame:
    """A candidate export carrying the receipt inputs on their engine entities.

    ``receives_wic`` sits on person; ``receives_snap`` and ``receives_tanf`` on
    spm_unit. ``omit`` drops that one column from the export entirely;
    ``all_default`` keeps every column but at the engine default (no signal).
    """
    values = (
        np.zeros(3, dtype=bool) if all_default else np.asarray([False, True, False])
    )
    person = {"receives_wic": values.copy()}
    spm_unit = {"receives_snap": values.copy(), "receives_tanf": values.copy()}
    person.pop(omit, None)
    spm_unit.pop(omit, None)
    return _us_entity_frame(3, person, {"spm_unit": spm_unit})


__all__ = [name for name in globals() if not name.startswith("__")]
