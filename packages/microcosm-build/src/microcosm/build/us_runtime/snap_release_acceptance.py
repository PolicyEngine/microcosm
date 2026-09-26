"""Scoped SNAP checks recomputed from an explicitly identified final export.

This is not a release certificate or a substitute for graph/source custody.
Calibration diagnostics describe the solve; only a simulation reopened from the
export supplies the final SNAP estimates. No public precomputed-estimate seam.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.gates import (
    TargetFitRequirement,
    _matches_target_fit_requirement,
    target_fit_gate,
)
from microcosm.build.us_runtime.engine_lifecycle import release_engine_simulation
from microcosm.build.us_runtime.h5_io import (
    assert_h5_unchanged,
    refuse_denied_pool_h5,
)
from microcosm.build.us_runtime.reform_validation import default_simulate_factory
from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_POSTAL
from microcosm.calibrate.registry import TargetRegistry, TargetSpec
from microcosm.calibrate.solve import TargetDiagnostic
from microcosm.frame import read_frame_table
from microcosm.frame.rules import ExportContract
from microcosm.frame.units import US_SCHEMA

SNAP_STATE_FIPS = frozenset(US_STATE_FIPS_TO_POSTAL)
_ROLES = {"snap_households": "indicator_sum", "snap_total": "sum"}
_CONSUMER = "final-export SNAP acceptance"


def _check(status: str, **details: object) -> dict[str, object]:
    return {"status": status, **details}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _code(value: object, width: int) -> str:
    """Accept integer codes, never truncate decimals or coerce arbitrary IDs."""
    if isinstance(value, bool) or pd.isna(value):
        raise ValueError("Missing or boolean geography code.")
    raw = str(value)
    if isinstance(value, float | np.floating) and math.isfinite(value):
        if value.is_integer():
            raw = str(int(value))
    if not raw.isascii() or not raw.isdigit() or len(raw) > width:
        raise ValueError(f"Invalid geography code {value!r}.")
    return raw.zfill(width)


def _read_export(
    path: Path, contract: ExportContract, period: int
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    from tables.exceptions import HDF5ExtError

    if not contract.required:
        raise ValueError("An explicit nonempty export contract is required.")
    tables = {}
    try:
        with pd.HDFStore(path, mode="r") as store:
            for entity in US_SCHEMA.entities:
                tables[entity] = read_frame_table(store, entity)
            stored_period = store["_time_period"]
    except HDF5ExtError as exc:
        raise ValueError("Export is not a readable HDF5 container.") from exc
    if len(stored_period) != 1 or stored_period.iloc[0] != period:
        raise ValueError("Export period does not match the requested model period.")
    columns = [column for table in tables.values() for column in table.columns]
    if len(columns) != len(set(columns)):
        raise ValueError("Export repeats a column across entity tables.")
    columns = set(columns)
    structural = {US_SCHEMA.person_id_column} | {
        column
        for entity in US_SCHEMA.group_entities
        for column in (US_SCHEMA.id_column(entity), US_SCHEMA.membership_column(entity))
    }
    missing = set(contract.required) - columns
    forbidden = (
        set(contract.forbidden) | set(contract.formula_owned_excluded)
    ) & columns
    # SNAP must be recomputed even if a caller supplies a weaker contract.
    forbidden |= {"snap"} & columns
    extras = (
        columns
        - (
            set(contract.required)
            | set(contract.optional)
            | structural
            | {"household_weight"}
        )
        if contract.closed
        else set()
    )
    if missing or forbidden or extras:
        raise ValueError(
            f"Export contract: missing={sorted(missing)}, forbidden={sorted(forbidden)}, "
            f"unexpected={sorted(extras)}."
        )
    person = tables["person"]
    for entity, table in tables.items():
        ids = table[
            US_SCHEMA.person_id_column
            if entity == "person"
            else US_SCHEMA.id_column(entity)
        ]
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError(f"Invalid or duplicate {entity} IDs.")
        if entity != "person":
            members = person[US_SCHEMA.membership_column(entity)]
            if members.isna().any() or not members.isin(ids).all():
                raise ValueError(f"Invalid {entity} membership.")
            if not ids.isin(members).all():
                raise ValueError(f"Empty {entity} groups in export.")
    household = tables["household"]
    weights = household["household_weight"].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any():
        raise ValueError("Invalid exported household weights.")
    states = household["state_fips"].map(lambda value: _code(value, 2))
    counties = household["county_fips"].map(lambda value: _code(value, 5))
    if set(states) != SNAP_STATE_FIPS:
        raise ValueError("Export must contain exactly the 50 states plus DC.")
    if set(states[weights > 0]) != SNAP_STATE_FIPS:
        raise ValueError("Each of the 51 areas must have positive-weight support.")
    if any(
        county[:2] != state or county[2:] == "000"
        for state, county in zip(states, counties, strict=True)
    ):
        raise ValueError("County/state prefix mismatch or placeholder county 000.")
    membership = (
        person[["person_spm_unit_id", "person_household_id"]]
        .drop_duplicates()
        .rename(
            columns={
                "person_spm_unit_id": "spm_unit_id",
                "person_household_id": "household_id",
            }
        )
    )
    if membership["spm_unit_id"].duplicated().any():
        raise ValueError("An SPM unit crosses household boundaries.")
    state_by_household = pd.Series(states.to_numpy(), index=household["household_id"])
    spm_ids = tables["spm_unit"]["spm_unit_id"].to_numpy()
    household_by_spm = membership.set_index("spm_unit_id")["household_id"]
    spm_states = household_by_spm.reindex(spm_ids).map(state_by_household).to_numpy()
    return (
        spm_ids,
        spm_states,
        _check(
            "passed",
            state_count=len(set(states)),
            county_count=len(set(counties)),
            county_check="Nonmissing numeric FIPS, nonzero county, state prefix and 51-area support; not a county-vintage authority check.",
        ),
    )


def _selected_specs(registry: TargetRegistry, period: int) -> tuple[TargetSpec, ...]:
    if registry.country != "us":
        raise ValueError("SNAP checks require a US TargetRegistry.")
    selected = tuple(
        spec
        for spec in registry.specs
        if spec.metadata.get("target_role") in _ROLES
        and "state_fips" in spec.metadata
        and spec.period == period
    )
    for role, mode in _ROLES.items():
        rows = [spec for spec in selected if spec.metadata["target_role"] == role]
        states = [spec.metadata["state_fips"] for spec in rows]
        if len(states) != 51 or set(states) != SNAP_STATE_FIPS:
            raise ValueError(
                f"{role}: require exactly one target for each of 51 areas."
            )
        for spec in rows:
            metadata = spec.metadata
            if (
                spec.family != "usda_snap"
                or spec.entity != "household"
                or spec.filter is not None
                or spec.value <= 0
                or metadata.get("materializer") != "policyengine_variable"
                or metadata.get("base_variable") != "snap"
                or metadata.get("measure_mode") != mode
                or metadata.get("target_period") != str(period)
                or not metadata.get("source_period")
                or metadata.get("source_measure_id")
                != (
                    "average_monthly_households"
                    if role == "snap_households"
                    else "total_benefits"
                )
                or (
                    role == "snap_households"
                    and metadata.get("fact_aggregation") != "time_mean"
                )
                or any(
                    key.startswith(("indicator_", "age_", "ledger_filter"))
                    or key in {"base_variables", "congressional_district_geoid"}
                    for key in metadata
                )
            ):
                raise ValueError(f"Unsupported SNAP target materializer: {spec.name}.")
    return selected


def _validate_diagnostics(
    specs: tuple[TargetSpec, ...], diagnostics: tuple[TargetDiagnostic, ...]
) -> None:
    if any(not isinstance(row, TargetDiagnostic) for row in diagnostics):
        raise TypeError(
            "Use typed TargetDiagnostic rows, not caller-written summaries."
        )
    by_name = {row.name: row for row in diagnostics}
    if len(by_name) != len(diagnostics):
        raise ValueError("Duplicate calibration diagnostic names.")
    for spec in specs:
        name = f"{spec.name}@{spec.period}"
        row = by_name.get(name)
        if row is None or row.target != spec.value:
            raise ValueError(f"Missing/stale calibration diagnostic: {name}.")
        if not all(
            math.isfinite(value)
            for value in (
                row.target,
                row.initial_estimate,
                row.final_estimate,
                row.relative_error,
            )
        ):
            raise ValueError(f"Nonfinite calibration diagnostic: {name}.")
        error = (row.final_estimate - row.target) / row.target
        within = (
            None
            if spec.tolerance is None
            else abs(row.final_estimate - row.target) <= spec.tolerance
        )
        if (
            not math.isclose(error, row.relative_error, rel_tol=1e-9, abs_tol=1e-9)
            or row.within_tolerance != within
        ):
            raise ValueError(f"Inconsistent calibration diagnostic: {name}.")


def _calculate_estimates(simulation, spm_ids, spm_states, period):
    """Preserve MicroSeries weights; compare at SPM grain before summing."""
    from microdf import MicroSeries

    ids = np.asarray(
        simulation.calculate("spm_unit_id", period=period, map_to="spm_unit")
    )
    if not np.array_equal(ids, spm_ids):
        raise ValueError("Simulation SPM IDs/order differ from the reopened export.")
    snap = simulation.calculate("snap", period=period, map_to="spm_unit")
    if not isinstance(snap, MicroSeries):
        raise TypeError("SNAP calculation must retain MicroSeries weights.")
    values = np.asarray(snap)
    if (
        values.ndim != 1
        or len(values) != len(spm_ids)
        or not np.isfinite(values).all()
        or (values < 0).any()
    ):
        raise ValueError("Invalid exported-artifact SNAP calculation.")
    positive = snap > 0
    return {
        state: {
            "snap_households": float(positive[spm_states == state].sum()),
            "snap_total": float(snap[spm_states == state].sum()),
        }
        for state in sorted(SNAP_STATE_FIPS)
    }


def _fit_rows(specs, estimates, requirements):
    result = []
    for spec in specs:
        name = f"{spec.name}@{spec.period}"
        estimate = estimates[spec.metadata["state_fips"]][spec.metadata["target_role"]]
        if not math.isfinite(estimate) or estimate < 0:
            raise ValueError(f"Nonfinite/negative final estimate: {name}.")
        error = (estimate - spec.value) / spec.value
        matching = tuple(
            requirement
            for requirement in requirements
            if _matches_target_fit_requirement(name, requirement)
        )
        thresholds = [spec.tolerance] if spec.tolerance is not None else []
        thresholds += [
            requirement.max_abs_relative_error * abs(spec.value)
            for requirement in matching
        ]
        status = (
            "pending"
            if not thresholds
            else (
                "passed" if abs(estimate - spec.value) <= min(thresholds) else "failed"
            )
        )
        result.append(
            {
                "name": name,
                "state_fips": spec.metadata["state_fips"],
                "role": spec.metadata["target_role"],
                "source": spec.source,
                "source_period": spec.metadata["source_period"],
                "model_period": spec.period,
                "target": spec.value,
                "final_estimate": estimate,
                "relative_error": error,
                "absolute_tolerance": spec.tolerance,
                "fit_requirement_ids": [
                    requirement.requirement_id for requirement in matching
                ],
                "status": status,
            }
        )
    return result


def snap_export_acceptance(
    dataset_path: str | Path,
    *,
    expected_sha256: str,
    contract: ExportContract,
    period: int,
    registry: TargetRegistry,
    diagnostics: Iterable[TargetDiagnostic],
    fit_requirements: Iterable[TargetFitRequirement] = (),
) -> dict[str, object]:
    """Inspect one final export, without issuing overall release approval.

    Missing engine/export evidence or an undeclared tolerance stays pending.
    Failures remain visible even when another check is pending. TargetSpec's
    absolute tolerances and existing TargetFitRequirement decisions are reused;
    no numerical tolerance, saturation waiver, or old release pin is supplied.
    The requested hash identifies bytes, not their source/calibration authority.
    """
    if not isinstance(contract, ExportContract) or not isinstance(
        registry, TargetRegistry
    ):
        raise TypeError("Use ExportContract and TargetRegistry instances.")
    if isinstance(period, bool) or not isinstance(period, int):
        raise TypeError("period must be an explicit integer model year.")
    if len(expected_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in expected_sha256
    ):
        raise ValueError("expected_sha256 must be an explicit lowercase SHA-256.")
    diagnostics, requirements = tuple(diagnostics), tuple(fit_requirements)
    if any(not isinstance(item, TargetFitRequirement) for item in requirements):
        raise TypeError("fit_requirements must contain TargetFitRequirement decisions.")
    checks = {}
    report = {
        "schema_version": 1,
        "scope": "SNAP final-export checks; not overall release certification",
        "artifact_sha256": expected_sha256,
        "model_period": period,
        "registry_version": registry.version,
        "export_contract_sha256": _digest(asdict(contract)),
        "checks": checks,
        "caseload_semantics": {
            "source": "FNS fiscal-year average monthly participating households",
            "model_proxy": "Weighted SPM/budget units with positive annual SNAP; boolean before household aggregation",
            "not_measured": "Physical households with any recipient, annual-ever recipients, or eligible-person participation",
        },
        "stage_take_up": "Input-stage design-weight fit/saturation cannot waive final-export misses.",
        "fy2022_eligible_person_comparison": _check(
            "pending",
            blocking=False,
            source_period="FY2022",
            source="https://www.fns.usda.gov/research/snap/state-participation-rates/2022",
            reason="Advisory only: no verified FY2022 comparison table or aligned SNAP assistance-unit person universe supplied. Caseload is not eligible-person participation.",
        ),
    }
    try:
        specs = _selected_specs(registry, period)
        _validate_diagnostics(specs, diagnostics)
        checks["registry_and_diagnostics"] = _check("passed", target_rows=len(specs))
    except (ValueError, TypeError) as exc:
        checks["registry_and_diagnostics"] = _check("failed", reason=str(exc))
        return report
    checks["final_export_fit"] = _check(
        "pending", reason="Final exported-artifact simulation not yet available."
    )
    path = Path(dataset_path)
    if not path.is_file():
        checks["artifact_identity"] = _check(
            "pending", reason="Final exported artifact is unavailable."
        )
        return report
    simulation = None
    try:
        actual = refuse_denied_pool_h5(path, consumer=_CONSUMER)
        if actual != expected_sha256:
            raise ValueError("Export SHA-256 does not match expected_sha256.")
        spm_ids, spm_states, geography = _read_export(path, contract, period)
        checks["export_contract_and_geography"] = geography
        simulation = default_simulate_factory(path)(None)
        estimates = _calculate_estimates(simulation, spm_ids, spm_states, period)
        rows = _fit_rows(specs, estimates, requirements)
        gate = target_fit_gate(rows, requirements)
        status = (
            "failed"
            if not gate.passed or any(row["status"] == "failed" for row in rows)
            else (
                "pending"
                if any(row["status"] == "pending" for row in rows)
                else "passed"
            )
        )
        checks["final_export_fit"] = _check(
            status,
            rows=rows,
            requirement_gate=asdict(gate),
            reason="Rows without a declared tolerance remain pending.",
        )
    except (ImportError, ModuleNotFoundError) as exc:
        checks["engine"] = _check(
            "pending", reason=f"Engine dependency unavailable: {exc}"
        )
    except (ValueError, TypeError, KeyError, OSError) as exc:
        checks["artifact_evaluation"] = _check("failed", reason=str(exc))
    finally:
        try:
            release_engine_simulation(simulation)
        except Exception as exc:
            checks["engine_cleanup"] = _check("failed", reason=str(exc))
            checks["final_export_fit"] = _check(
                "failed", reason="Engine cleanup failed; evaluation is incomplete."
            )
        finally:
            # Also executes if engine teardown raises: this is the last relevant I/O.
            try:
                assert_h5_unchanged(path, expected_sha256, consumer=_CONSUMER)
                checks["artifact_identity"] = _check("passed", final_io_verified=True)
            except (ValueError, OSError) as exc:
                checks["artifact_identity"] = _check("failed", reason=str(exc))
                checks["final_export_fit"] = _check(
                    "failed",
                    reason="Artifact changed or mismatched; any computed estimates are invalid.",
                )
    return report
