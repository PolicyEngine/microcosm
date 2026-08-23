"""Calibrate a WS-E UK spine at national level, without the certified-input replay.

The certified-H5 driver (``build_uk_national_dataset.py``) encodes the June
convention: its input arrives with a zero-weight SPI channel, and the driver
re-derives the SPI and CGT surfaces in-run before calibrating. A WS-E spine
already carries those derivations — E7/E8 ported the same instruments (pinned
sources, QRF stages, reviewed fences, conservation receipts) into declarative
source stages, and the spine ships post-allocation. Re-running the replay here
would discard the spine's certified derivation, so this runner deliberately
does not.

The 388-reference register also binds simulated tax-benefit outputs
(income tax, NICs, Universal Credit, benefit caseloads, payment-band
crosstabs). Those are model outputs, not stored columns, and the calibration
stage's frame adapter reads stored columns only — the June release never hit
this because its 149-target surface was demographics-only. This runner adds
the missing materialization: a live policyengine-uk simulation over the same
records computes each needed variable at the declared calibration year and
attaches it to the frame, iterating on the materializer's own skip reports
until the register binds. The same treatment is applied to the incumbent
before scoring, so the #578 rule-1 comparison uses one yardstick. References
that need machinery this posture does not have (the salary-sacrifice
counterfactual deltas) are excluded with a receipt, never silently.

Assessment posture: the artifact is structurally non-releasable (non-certified
input), the declared battery is not enforced, and the build record says both
things. Numeric fences that are honestly measurable here are evaluated against
the thresholds declared in ``uk/gates.json``; every non-evaluated entry is
listed with its reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.logbook import canonical_json_bytes
from microcosm.build.logbook_adoption import (
    AttemptState,
    append_phase,
    git_code_pin,
    local_artifact_reference,
    record_terminal_attempt,
    resolve_predecessor,
    role_pins_digest,
)
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.ledger_targets import (
    UKFrameTargetAdapter,
    compile_uk_target_registry,
    materialize_uk_ledger_targets,
)
from microcosm.build.uk_runtime.national_build import (
    load_uk_national_frame,
    write_uk_national_frame,
)
from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.national_calibration import (
    UKNationalCalibrationStage,
    _CalibrationFrameAdapter,
    _doctrine_bounds,
    _post_solve_calibration_record,
    national_calibration_mass_reason,
)
from microcosm.build.uk_runtime.diagnostics import write_uk_calibration_diagnostics
from microcosm.build.uk_runtime.national_doctrine import UK_NATIONAL_SOLVE_DOCTRINE
from microcosm.calibrate import (
    TargetRegistry,
    calibrate,
    effective_sample_size,
    score_targets,
)

_REPOSITORY = Path(__file__).resolve().parent.parent
_PIPELINE = "uk-frs-calibration"
UK_SCORE_LOSS_CAP = 10.0

# Structural NaN repair allowlist. These twelve person columns are
# SPI-channel concepts the spine leaves undefined (NaN) on the base-FRS
# channel; zero is their semantic value there — base-channel incomes are
# carried by the standard FRS components. Any NaN outside this list aborts.
_STRUCTURAL_NAN_COLUMNS = (
    "other_investment_income",
    "hmrc_spi_employment_benefits",
    "hmrc_spi_employment_expenses",
    "hmrc_spi_other_social_security_income",
    "hmrc_spi_taxable_termination_pay",
    "hmrc_spi_miscellaneous_employment_income",
    "hmrc_spi_other_income",
    "hmrc_spi_state_pension_income",
    "hmrc_spi_employed_income",
    "hmrc_spi_total_earned_income",
    "hmrc_spi_total_investment_income",
    "hmrc_spi_assessable_income",
)

_ENTITY_LINK = {"benunit": "person_benunit_id", "household": "person_household_id"}
_ENTITY_ID = {"person": "person_id", "benunit": "benunit_id", "household": "household_id"}

_MISSING_COLUMN = re.compile(r"^'(?:([a-z_0-9]+)\.)?([A-Za-z_0-9]+)'$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_gate_parameters() -> dict[str, dict[str, Any]]:
    from importlib import resources

    payload = json.loads(
        resources.files("microcosm.build.uk")
        .joinpath("gates.json")
        .read_text(encoding="utf-8")
    )
    return {
        entry["id"]: dict(entry.get("parameters") or {})
        for entry in payload["gates"]
    }


def _repair_structural_nans(frame: Any) -> dict[str, Any]:
    """Zero the allowlisted NaN columns in place; abort on any other NaN.

    ``Frame.table`` documents its return as read-only, but it hands back the
    live DataFrame (the calibration adapter takes defensive copies for
    exactly that reason). This repair leans on that liveness and then
    verifies the mutation stuck.
    """

    receipt: dict[str, Any] = {"repaired": {}, "policy": "structural_zero_fill"}
    for entity in ("person", "benunit", "household"):
        table = frame.table(entity)
        for column in table.columns:
            series = table[column]
            if series.dtype.kind != "f":
                continue
            nan_count = int(series.isna().sum())
            if nan_count == 0:
                continue
            if entity != "person" or column not in _STRUCTURAL_NAN_COLUMNS:
                raise ValueError(
                    f"unexpected NaN outside the structural allowlist: "
                    f"{entity}.{column} has {nan_count} NaN rows."
                )
            table[column] = series.fillna(0.0)
            receipt["repaired"][column] = nan_count
    for column in receipt["repaired"]:
        if int(frame.table("person")[column].isna().sum()) != 0:
            raise RuntimeError(
                f"structural NaN repair did not persist for person.{column}."
            )
    return receipt


def _compute_measure_input(
    frame: Any, simulation: Any, entity: str, variable: str, year: int
) -> tuple[np.ndarray, str]:
    """Compute one simulated variable at ``entity`` grain.

    Returns the values and a short description of the path taken. Values are
    never written to the frame — they are injected at adapter level, where
    column names are table-scoped, so the Frame's global-uniqueness rule
    (which forbids ``region`` living on two entities) is never violated.
    """

    definition = simulation.tax_benefit_system.variables.get(variable)
    if definition is None:
        raise KeyError(f"policyengine-uk has no variable {variable!r}")
    native = definition.entity.key
    table = frame.table(entity)

    if native == entity:
        raw = simulation.calculate(variable, year)
        values = np.asarray(raw.values if hasattr(raw, "values") else raw)
        route = "native"
    else:
        raw = simulation.calculate(variable, year)
        native_values = np.asarray(raw.values if hasattr(raw, "values") else raw)
        if native_values.dtype.kind in {"O", "U", "S", "b"}:
            if entity == "person" and native in ("benunit", "household"):
                # Categorical broadcast: each member inherits its group value.
                native_table = frame.table(native)
                lookup = pd.Series(
                    native_values, index=native_table[_ENTITY_ID[native]].to_numpy()
                )
                keys = frame.table("person")[_ENTITY_LINK[native]].to_numpy()
                values = lookup.loc[keys].to_numpy()
                route = f"categorical_broadcast_{native}_to_person"
            elif native == "person" and native_values.dtype.kind == "b":
                # Boolean any-collapse up to the group.
                person = frame.table("person")
                collapsed = (
                    pd.Series(native_values.astype(bool))
                    .groupby(person[_ENTITY_LINK[entity]].to_numpy())
                    .max()
                )
                keys = table[_ENTITY_ID[entity]].to_numpy()
                values = collapsed.loc[keys].to_numpy().astype(float)
                route = f"bool_any_collapse_person_to_{entity}"
            else:
                raise KeyError(
                    f"no categorical mapping from {native} to {entity} "
                    f"for {variable!r}"
                )
        else:
            raw = simulation.calculate(variable, year, map_to=entity)
            values = np.asarray(raw.values if hasattr(raw, "values") else raw)
            route = f"map_to_{entity}"

    if len(values) != len(table):
        raise ValueError(
            f"{variable!r} produced {len(values)} values for {len(table)} "
            f"{entity} rows."
        )
    if values.dtype.kind not in {"O", "U", "S"}:
        values = values.astype(float)
    return values, route


def _inject_measure_inputs(adapter: Any, measure_inputs: dict) -> None:
    for (entity, variable), values in measure_inputs.items():
        adapter.tables[entity][variable] = values


def _resolve_simulated_measures(
    frame: Any,
    simulation: Any,
    registry: TargetRegistry,
    year: int,
    *,
    side: str,
    max_rounds: int = 8,
) -> tuple[TargetRegistry, dict[tuple[str, str], np.ndarray], dict[str, Any]]:
    """Compute simulated measure inputs until the register binds.

    Drives the materializer's own skip reports: each round computes the
    missing (entity, variable) pairs from a live simulation, injects them at
    adapter level, and retries. A reference is only excluded when its need
    is structurally unmeetable here (counterfactual deltas, unknown
    variables, or a key that was already provided in a previous round and
    still does not satisfy it).
    """

    receipt: dict[str, Any] = {"side": side, "attached": {}, "excluded": {}}
    measure_inputs: dict[tuple[str, str], np.ndarray] = {}
    active = list(registry.specs)
    for round_index in range(max_rounds):
        probe = UKFrameTargetAdapter(frame)
        _inject_measure_inputs(probe, measure_inputs)
        result = materialize_uk_ledger_targets(
            probe, TargetRegistry(active, country="uk"), period=year
        )
        skipped = list(result.skipped)
        if not skipped:
            break
        provided_before = set(measure_inputs)
        provided_this_round: set[tuple[str, str]] = set()
        progressed = False
        for skip in skipped:
            info = skip.__dict__
            name, reason = str(info["name"]), str(info["reason"])
            if "counterfactual delta" in reason:
                receipt["excluded"][name] = reason
                active = [spec for spec in active if spec.name != name]
                progressed = True
                continue
            match = _MISSING_COLUMN.match(reason)
            if match is None:
                receipt["excluded"][name] = f"unrecognized skip reason: {reason}"
                active = [spec for spec in active if spec.name != name]
                progressed = True
                continue
            entity, variable = match.group(1), match.group(2)
            if entity is None:
                definition = simulation.tax_benefit_system.variables.get(variable)
                if definition is None:
                    receipt["excluded"][name] = (
                        f"policyengine-uk has no variable {variable!r}"
                    )
                    active = [spec for spec in active if spec.name != name]
                    progressed = True
                    continue
                entity = definition.entity.key
            key = (entity, variable)
            if key in provided_this_round:
                # Someone else already computed it this round; this reference
                # resolves on the next probe.
                continue
            if key in provided_before:
                receipt["excluded"][name] = (
                    f"still unmaterializable with {entity}.{variable} "
                    f"provided: {reason}"
                )
                active = [spec for spec in active if spec.name != name]
                progressed = True
                continue
            try:
                values, route = _compute_measure_input(
                    frame, simulation, entity, variable, year
                )
            except (KeyError, ValueError) as error:
                receipt["excluded"][name] = str(error)
                active = [spec for spec in active if spec.name != name]
            else:
                measure_inputs[key] = values
                provided_this_round.add(key)
                receipt["attached"][f"{entity}.{variable}"] = route
            progressed = True
        if not progressed:
            raise RuntimeError(
                f"measure materialization made no progress on round "
                f"{round_index}: {[s.__dict__ for s in skipped][:5]}"
            )
    else:
        raise RuntimeError("measure materialization did not converge.")
    return TargetRegistry(active, country="uk"), measure_inputs, receipt


class _SpineCalibrationStage(UKNationalCalibrationStage):
    """The national calibration stage, with adapter-level measure inputs.

    Mirrors the parent's ``__call__`` with two insertions: simulated measure
    inputs are injected into the adapter's table copies before
    materialization, and removed again before the prepared frame is built —
    the solver consumes the materialized measure columns, and dropping the
    injected inputs keeps the prepared frame inside the Frame's
    global-uniqueness rule. ``restore`` then rebuilds the pristine tables
    around the calibrated weights, so the staged artifact carries none of
    the injected columns.
    """

    def __init__(self, registry, *, period, measure_inputs):
        super().__init__(registry, period=period)
        self._measure_inputs = dict(measure_inputs)

    def __call__(self, frame: Any) -> Any:
        declared = len(self.registry.specs) + len(self.compilation.unsupported)
        if self.compilation.unsupported:
            raise RuntimeError(
                f"unsupported references: {self.compilation.unsupported!r}"
            )
        adapter = _CalibrationFrameAdapter(frame)
        original_columns = {
            entity: set(table.columns) for entity, table in adapter.tables.items()
        }
        _inject_measure_inputs(adapter, self._measure_inputs)
        materialized = materialize_uk_ledger_targets(
            adapter, self.registry, period=self.period
        )
        if materialized.skipped:
            skipped = [skip.__dict__ for skip in materialized.skipped]
            raise RuntimeError(
                f"calibration register did not bind: skipped={skipped[:5]}"
            )
        for entity, variable in self._measure_inputs:
            if variable not in original_columns[entity]:
                adapter.tables[entity].drop(columns=[variable], inplace=True)
        prepared = adapter.prepared_frame()
        mass_reason = national_calibration_mass_reason(
            spec.family for spec in self.registry.specs
        )
        mass_log_records_before_calibration = len(frame.mass_log)
        result = calibrate(
            prepared,
            self.registry.to_target_set(),
            weight_entity="household",
            epochs=self.doctrine.epochs,
            learning_rate=self.doctrine.learning_rate,
            mass=self.doctrine.mass_rule,
            mass_reason=mass_reason,
            max_weight_ratio=self.doctrine.max_weight_ratio,
            seed=self.doctrine.seed,
            l0_lambda=self.doctrine.l0_lambda,
            target_loss_cap=self.doctrine.target_loss_cap,
        )
        if result.skipped or len(result.problem.names) != declared:
            skipped_names = [item.name for item in result.skipped]
            raise RuntimeError(
                f"calibration matrix incomplete: declared={declared}, "
                f"rows={len(result.problem.names)}, skipped={skipped_names[:10]}"
            )
        clean_frame = adapter.restore(result.frame)
        self.solve_result = result
        calibration_record = _post_solve_calibration_record(
            frame,
            clean_frame,
            before_count=mass_log_records_before_calibration,
        )
        self.diagnostics = tuple(
            {
                "name": row.name,
                "estimate": row.final_estimate,
                "target": row.target,
                "relative_error": row.relative_error,
            }
            for row in result.diagnostics
        )
        ratios = result.weights / result.initial_weights
        before_kind = frame.weights_for("household").kind
        after_kind = clean_frame.weights_for("household").kind
        self.manifest = {
            "activated_reference_count": declared,
            "resolved_reference_count": len(self.registry.specs),
            "matrix_target_count": len(result.problem.names),
            "loss": result.final_loss,
            "effective_sample_size": effective_sample_size(result.weights),
            "max_weight_ratio": float(ratios.max()),
            "max_weight_ratio_bound": self.doctrine.max_weight_ratio,
            "target_materialization": materialized.report(),
            "weights": {
                "household_weight_kind": after_kind.value,
                "household_weight_kind_chain": [
                    {"stage": "staging", "kind": before_kind.value},
                    {"stage": "national_calibration", "kind": after_kind.value},
                ],
                "mass_log_records": len(clean_frame.mass_log),
                "calibration_mass_change": {
                    "entity": str(calibration_record.entity),
                    "old_total": float(calibration_record.old_total),
                    "new_total": float(calibration_record.new_total),
                    "relative_shift": (
                        float(calibration_record.new_total)
                        - float(calibration_record.old_total)
                    )
                    / float(calibration_record.old_total),
                    "reason": str(calibration_record.reason),
                },
            },
            "solve": {
                "n_targets": len(result.problem.names),
                "n_households": len(clean_frame.table("household")),
                "initial_loss": float(result.initial_loss),
                "final_loss": float(result.final_loss),
                "n_nonzero": int(np.count_nonzero(result.weights)),
            },
            "parameters": {"doctrine": _doctrine_bounds(self.doctrine)},
        }
        self.output_content_identity = uk_frame_content_identity(clean_frame)
        return clean_frame


def _uk_target_geography_levels(registry: Any) -> dict[str, str]:
    """Row-name -> geography level, from the value-free contract."""

    from importlib import resources

    payload = json.loads(
        resources.files("microcosm.build.uk")
        .joinpath("uk_national_targets.json")
        .read_text(encoding="utf-8")
    )
    targets = {row["target_id"]: row for row in payload["targets"]}
    levels: dict[str, str] = {}
    for spec in registry.specs:
        target_id = spec.metadata.get("contract_target_id")
        target = targets.get(str(target_id))
        if target is None:
            raise ValueError(
                f"UK calibration target {spec.name!r} references unknown "
                f"contract target {target_id!r}."
            )
        geography_levels = tuple(target.get("geography_levels") or ())
        if not geography_levels:
            raise ValueError(
                f"UK calibration target {spec.name!r} has no geography level."
            )
        levels[spec.to_target().row_name] = str(geography_levels[0])
    return levels


def _score_frame(
    frame: Any, registry: TargetRegistry, year: int, measure_inputs: dict
) -> Any:
    """Materialize measures and score one frame on the shared register."""

    adapter = _CalibrationFrameAdapter(frame)
    original_columns = {
        entity: set(table.columns) for entity, table in adapter.tables.items()
    }
    _inject_measure_inputs(adapter, measure_inputs)
    result = materialize_uk_ledger_targets(adapter, registry, period=year)
    if result.skipped:
        raise RuntimeError(
            f"scoring register did not bind: {[s.__dict__ for s in result.skipped][:5]}"
        )
    for entity, variable in measure_inputs:
        if variable not in original_columns[entity]:
            adapter.tables[entity].drop(columns=[variable], inplace=True)
    return score_targets(
        adapter.prepared_frame(),
        registry.to_target_set(),
        target_loss_cap=UK_SCORE_LOSS_CAP,
    )


def _score_block(
    candidate_frame: Any,
    incumbent_frame: Any,
    registry: TargetRegistry,
    year: int,
    scorer: Any,
    candidate_inputs: dict,
    incumbent_inputs: dict,
) -> dict[str, Any]:
    """The #578 rule-1 block, on frames instead of H5 paths.

    Reuses the scorer module's own helpers so the payload shape stays
    identical to ``score_uk_national_candidate.py``.
    """

    candidate = _score_frame(candidate_frame, registry, year, candidate_inputs)
    incumbent = _score_frame(incumbent_frame, registry, year, incumbent_inputs)
    candidate_errors = scorer._relative_errors(candidate)
    incumbent_errors = scorer._relative_errors(incumbent)
    missing = sorted(set(candidate_errors) ^ set(incumbent_errors))
    if missing:
        raise RuntimeError(
            f"candidate and incumbent scores produced different rows: {missing[:10]}"
        )
    wins = scorer._target_wins(candidate_errors, incumbent_errors)
    return {
        "candidate_train_loss": float(candidate.final_loss),
        "candidate_holdout_loss": float(candidate.final_loss),
        "candidate_full_loss": float(candidate.final_loss),
        "incumbent_train_loss": float(incumbent.final_loss),
        "incumbent_holdout_loss": float(incumbent.final_loss),
        "incumbent_full_loss": float(incumbent.final_loss),
        "candidate_target_wins": wins["candidate"],
        "incumbent_target_wins": wins["incumbent"],
        "holdout_basis": "none_declared",
        "loss": {
            "objective": "relative_error_loss",
            "target_loss_cap": UK_SCORE_LOSS_CAP,
            "train_equals_full": True,
        },
        "register": {
            "country": registry.country,
            "version": registry.version,
            "n_specs": len(registry.specs),
        },
        "artifacts": {
            "candidate": "populace_uk_2023",
            "incumbent": "enhanced_frs_2024_25",
        },
        "signed_asymmetries": [
            {
                "id": "incumbent_own_registry",
                "note": (
                    "the incumbent was calibrated to its own registry; both "
                    "sides are rescored on the shared frozen register here"
                ),
            },
            {
                "id": "simulated_measures_shared_yardstick",
                "note": (
                    "policyengine-bound measures are computed for both sides "
                    "by the same policyengine-uk version over each dataset's "
                    "own records"
                ),
            },
        ],
        "target_wins_by_family": scorer._target_wins_by_family(
            registry, candidate_errors, incumbent_errors
        ),
    }


def _assessment_gate_receipt(
    stage: UKNationalCalibrationStage,
    frame: Any,
    gate_parameters: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate the honestly-measurable numeric fences from uk/gates.json."""

    weights = np.asarray(frame.weights_for("household").values, dtype=float)
    household = frame.table("household")
    evaluated: dict[str, Any] = {}

    fit_params = gate_parameters["uk_target_fit"]
    bound = float(fit_params["max_abs_relative_error"])
    errors = {
        str(row["name"]): float(row["relative_error"]) for row in stage.diagnostics
    }
    failing = {
        name: error for name, error in errors.items() if abs(error) > bound
    }
    evaluated["uk_target_fit"] = {
        "passed": not failing,
        "threshold": bound,
        "targets_checked": len(errors),
        "failing_count": len(failing),
        "worst": dict(
            sorted(failing.items(), key=lambda kv: -abs(kv[1]))[:20]
        ),
    }

    ratio_params = gate_parameters["uk_weight_ratio"]
    positive = weights[weights > 0]
    max_to_median = (
        float(positive.max() / np.median(positive)) if positive.size else 0.0
    )
    ratio_bound = float(ratio_params["maximum_max_to_median_ratio"])
    evaluated["uk_weight_ratio"] = {
        "passed": max_to_median <= ratio_bound,
        "threshold": ratio_bound,
        "measured_max_to_median": max_to_median,
    }

    ess_params = gate_parameters["uk_weight_ess"]
    ess = float(effective_sample_size(weights))
    ess_fraction = ess / len(weights) if len(weights) else 0.0
    ess_bound = float(ess_params["minimum_ess_fraction"])
    evaluated["uk_weight_ess"] = {
        "passed": ess_fraction >= ess_bound,
        "threshold": ess_bound,
        "measured_ess_fraction": ess_fraction,
        "effective_sample_size": ess,
    }

    strata_params = gate_parameters["uk_zero_weight_strata"]
    zero_mask = weights <= 0.0
    strata_results = []
    matched = np.zeros(len(household), dtype=bool)
    for declaration in strata_params.get("declarations", []):
        selector = declaration.get("selector", {})
        mask = np.ones(len(household), dtype=bool)
        for column, value in selector.items():
            mask &= household[column].to_numpy() == value
        matched |= mask
        zero_rows = int((mask & zero_mask).sum())
        strata_results.append(
            {
                "name": declaration.get("name"),
                "zero_weight_rows": zero_rows,
                "maximum": declaration.get("maximum_zero_weight_rows"),
                "passed": zero_rows <= int(declaration["maximum_zero_weight_rows"]),
            }
        )
    unmatched_zero = int((zero_mask & ~matched).sum())
    evaluated["uk_zero_weight_strata"] = {
        "passed": all(row["passed"] for row in strata_results)
        and unmatched_zero == 0,
        "strata": strata_results,
        "zero_weight_rows_outside_declared_strata": unmatched_zero,
    }

    admin_params = gate_parameters["uk_aggregate_admin"]
    default_rtol = float(admin_params.get("default_rtol", 0.5))
    admin_results = []
    for anchor in admin_params.get("anchors", []):
        measure = str(anchor.get("measure") or anchor.get("name"))
        value = float(anchor["value"])
        if measure not in household.columns:
            admin_results.append(
                {"anchor": anchor.get("name"), "status": "column_absent"}
            )
            continue
        column_values = household[measure].to_numpy(dtype=float)
        total = float(np.dot(column_values, weights))
        carriers = column_values != 0
        carrier_weight = float(weights[carriers].sum())
        mean_carriers = (
            float(np.dot(column_values[carriers], weights[carriers]) / carrier_weight)
            if carrier_weight
            else float("nan")
        )
        # The anchor value's magnitude tells its statistic: NEED per-household
        # means are hundreds of pounds, the NHS anchor is a national total.
        measured = mean_carriers if abs(value) < 1e6 else total
        scale = max(abs(value), 1.0)
        tolerance = anchor.get("tolerance")
        rtol = float(tolerance) / scale if tolerance is not None else default_rtol
        miss = abs(measured - value) / scale
        admin_results.append(
            {
                "anchor": anchor.get("name"),
                "measure": measure,
                "declared_value": value,
                "measured": measured,
                "weighted_total": total,
                "weighted_mean_carriers": mean_carriers,
                "relative_miss": miss,
                "rtol": rtol,
                "passed": miss <= rtol,
                "statistic_convention": "assessed_by_anchor_magnitude",
            }
        )
    evaluated["uk_aggregate_admin"] = {
        "passed": all(row.get("passed", False) for row in admin_results),
        "anchors": admin_results,
    }

    not_evaluated = {
        "uk_release_family_build_stages": (
            "counts in-run stage names; superseded for spine inputs — the "
            "spine's build record carries the source-stage census"
        ),
        "uk_weights_audit": (
            "consumes in-run FitWeightRecords from the replay stage; the "
            "spine's qrf_evidence sidecar holds the equivalent record"
        ),
        "uk_release_input_coverage": "manifest describes the certified-input pipeline",
        "uk_release_input_coverage_manifest_current": "same",
        "uk_ledger_compile_parity_production_2023": (
            "register-coverage receipt verified the 2025 compile pre-run"
        ),
        "uk_ledger_compile_parity_incumbent_2025": "same",
        "uk_export_surface": "deferred to T0 column-inventory evaluation",
        "uk_target_surface": (
            "register identity is enforced by the shared-register scorer"
        ),
        "uk_support": "support-bounds resources bind to the replay stages",
        "uk_take_up_signal": "take-up draws are spine stages; deferred to T-evals",
        "uk_qrf_tail_concentration": "recorded at spine build (L3 baselines)",
        "uk_degenerate_release_surface": "deferred to T0/T3 evaluations",
        "uk_nonnegative_columns": "spine stage-local gates already enforce",
        "uk_brma_enum_domain": "deferred to T0 load check in the model venv",
        "uk_student_loan_plan_enum_domain": "same",
        "uk_input_mass_parity": "input-mass reference binds to the replay stages",
        "uk_calibration_reference_coverage": (
            "register-coverage receipt carries the census"
        ),
    }
    return {
        "posture": "spine_assessment",
        "enforcement": "not_applied",
        "evaluated": evaluated,
        "not_evaluated": not_evaluated,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", required=True, type=Path)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--ledger-facts", required=True, type=Path)
    parser.add_argument("--ledger-facts-sha256", required=True)
    parser.add_argument("--ledger-manifest-sha256", required=True)
    parser.add_argument("--staging-h5", required=True, type=Path)
    parser.add_argument("--register-json", required=True, type=Path,
                        help="Frozen scoring register emitted by the coverage check;"
                        " re-derived here and byte-compared.")
    parser.add_argument("--incumbent-h5", required=True, type=Path)
    parser.add_argument("--diagnostics-json", required=True, type=Path)
    parser.add_argument("--build-record-json", required=True, type=Path)
    parser.add_argument("--logbook-prev-row-digest", default=None)
    parser.add_argument(
        "--assessment-fences",
        type=Path,
        help="JSON of {fences: [{prefix, reason}]} removing references from "
        "the calibration and scoring registers, each with a receipted "
        "reason. Measure-level acts on defective or unsupported measures — "
        "never threshold edits. Recorded in the build record.",
    )
    args = parser.parse_args(argv)

    started_at = time.perf_counter()
    started_ts = datetime.now(timezone.utc)
    code_pin = git_code_pin(_REPOSITORY)

    # --- 1. verify inputs -------------------------------------------------
    input_sha = _sha256_file(args.input_h5)
    if input_sha != args.input_sha256:
        raise SystemExit(
            f"input H5 sha mismatch: measured {input_sha}, pinned {args.input_sha256}"
        )
    artifact = load_ledger_consumer_artifact(args.ledger_facts)
    if artifact.facts_sha256 != args.ledger_facts_sha256:
        raise SystemExit("ledger facts sha mismatch")
    if artifact.manifest_sha256 != args.ledger_manifest_sha256:
        raise SystemExit("ledger manifest sha mismatch")

    source_pins = {
        "input_h5": {
            "sha256": input_sha,
            "size_bytes": args.input_h5.stat().st_size,
        },
        "ledger_facts": {
            "sha256": artifact.facts_sha256,
            "size_bytes": (args.ledger_facts / "consumer_facts.jsonl").stat().st_size
            if args.ledger_facts.is_dir()
            else args.ledger_facts.stat().st_size,
        },
    }

    # --- 2. compile the register at the declared calibration year ---------
    calibration_year = load_uk_frs_release().calibration_year
    compilation = compile_uk_target_registry(
        artifact.facts, target_period=calibration_year
    )
    if compilation.unsupported:
        raise SystemExit(
            f"{len(compilation.unsupported)} target references failed to compile"
        )
    register_payload = {
        "country": compilation.registry.country,
        "version": compilation.registry.version,
        "specs": [
            {
                field: value
                for field, value in spec.__dict__.items()
                if value is not None
            }
            for spec in compilation.registry.specs
        ],
    }
    register_bytes = json.dumps(
        register_payload, indent=2, sort_keys=True
    ).encode("utf-8")
    register_sha = hashlib.sha256(register_bytes).hexdigest()
    frozen_sha = hashlib.sha256(args.register_json.read_bytes()).hexdigest()
    if register_sha != frozen_sha:
        raise SystemExit(
            "re-derived register differs from the frozen scoring register: "
            f"{register_sha} vs {frozen_sha}"
        )

    run_config = {
        "pipeline": _PIPELINE,
        "source_pins": source_pins,
        "register_sha256": register_sha,
        "calibration_year": calibration_year,
        "doctrine": {
            "epochs": UK_NATIONAL_SOLVE_DOCTRINE.epochs,
            "learning_rate": UK_NATIONAL_SOLVE_DOCTRINE.learning_rate,
            "max_weight_ratio": UK_NATIONAL_SOLVE_DOCTRINE.max_weight_ratio,
            "seed": UK_NATIONAL_SOLVE_DOCTRINE.seed,
            "target_loss_cap": UK_NATIONAL_SOLVE_DOCTRINE.target_loss_cap,
            "l0_lambda": UK_NATIONAL_SOLVE_DOCTRINE.l0_lambda,
            "mass_rule": UK_NATIONAL_SOLVE_DOCTRINE.mass_rule,
        },
        "structural_nan_allowlist": list(_STRUCTURAL_NAN_COLUMNS),
    }
    state = AttemptState(
        build_id=f"uk-frs-calibration-{started_ts.strftime('%Y%m%dT%H%M%SZ')}",
        identity_digest=hashlib.sha256(canonical_json_bytes(run_config)).hexdigest(),
        input_pins_digest=role_pins_digest(source_pins),
        phases_reached=["attempt_started"],
        gate_verdicts={},
    )

    # --- 3. load + repair + prepared input for the simulation --------------
    frame, _provenance = load_uk_national_frame(args.input_h5)
    append_phase(state, "input_loaded")
    nan_receipt = _repair_structural_nans(frame)
    prepared_input = args.staging_h5.with_name("input-prepared.h5")
    write_uk_national_frame(frame, prepared_input)
    prepared_sha = _sha256_file(prepared_input)
    append_phase(state, "structural_nans_repaired")

    # --- 4. simulated-measure materialization ------------------------------
    from policyengine_uk import Microsimulation

    fenced: dict[str, str] = {}
    fence_registry = compilation.registry
    if args.assessment_fences:
        fence_config = json.loads(
            args.assessment_fences.read_text(encoding="utf-8")
        )
        rules = [(f["prefix"], f["reason"]) for f in fence_config["fences"]]
        kept = []
        for spec in compilation.registry.specs:
            reason = next(
                (why for prefix, why in rules if spec.name.startswith(prefix)),
                None,
            )
            if reason is None:
                kept.append(spec)
            else:
                fenced[spec.name] = reason
        fence_registry = TargetRegistry(kept, country="uk")
        print(
            f"assessment fences removed {len(fenced)} references "
            f"({len(kept)} remain)"
        )

    print("materializing simulated measures on the candidate spine...")
    candidate_sim = Microsimulation(dataset=str(prepared_input))
    effective_registry, candidate_inputs, candidate_measures = (
        _resolve_simulated_measures(
            frame, candidate_sim, fence_registry, calibration_year,
            side="candidate",
        )
    )
    del candidate_sim
    append_phase(state, "simulated_measures_materialized")
    print(
        f"  bindable references: {len(effective_registry.specs)}/"
        f"{len(compilation.registry.specs)}; "
        f"computed {len(candidate_inputs)} measure inputs; "
        f"excluded {len(candidate_measures['excluded'])}"
    )

    # --- 5. calibrate -------------------------------------------------------
    stage = _SpineCalibrationStage(
        effective_registry,
        period=calibration_year,
        measure_inputs=candidate_inputs,
    )
    calibrated = stage(frame)
    append_phase(state, "national_calibration_solved")

    # --- 6. write the H5 ----------------------------------------------------
    args.staging_h5.parent.mkdir(parents=True, exist_ok=True)
    write_uk_national_frame(calibrated, args.staging_h5)
    staged_sha = _sha256_file(args.staging_h5)
    append_phase(state, "staging_h5_written")

    # --- 7. score vs the incumbent on the shared bindable register ---------
    import importlib.util

    scorer_spec = importlib.util.spec_from_file_location(
        "score_uk_national_candidate",
        Path(__file__).with_name("score_uk_national_candidate.py"),
    )
    scorer = importlib.util.module_from_spec(scorer_spec)
    scorer_spec.loader.exec_module(scorer)

    print("materializing simulated measures on the incumbent...")
    incumbent_frame, _ = load_uk_national_frame(args.incumbent_h5)
    incumbent_sim = Microsimulation(dataset=str(args.incumbent_h5))
    score_registry, incumbent_inputs, incumbent_measures = (
        _resolve_simulated_measures(
            incumbent_frame, incumbent_sim, effective_registry, calibration_year,
            side="incumbent",
        )
    )
    del incumbent_sim
    score = _score_block(
        calibrated,
        incumbent_frame,
        score_registry,
        calibration_year,
        scorer,
        candidate_inputs,
        incumbent_inputs,
    )
    append_phase(state, "scored_vs_incumbent")

    # --- 8. diagnostics with the score merged, single pass ------------------
    build_block = {
        "build_id": state.build_id,
        "ledger_facts": artifact.provenance(),
        "code_pin": code_pin,
        "source_pins": source_pins,
        "input_posture": {
            "posture": "staging_candidate_spine",
            "filename": args.input_h5.name,
            "tier": "staging_candidate",
            "revision": "non_certified_staging_candidate",
            "sha256": input_sha,
            "size_bytes": args.input_h5.stat().st_size,
            "prepared_input_sha256": prepared_sha,
        },
        "structural_nan_repair": nan_receipt,
        "simulated_measures": {
            "candidate": candidate_measures,
            "incumbent": incumbent_measures,
        },
        "score_vs_enhanced_frs": score,
    }
    write_uk_calibration_diagnostics(
        stage.solve_result,
        args.diagnostics_json,
        calibrated,
        target_geography_levels=_uk_target_geography_levels(stage.registry),
        target_registry=stage.registry,
        build=build_block,
    )
    diagnostics_sha = _sha256_file(args.diagnostics_json)
    append_phase(state, "diagnostics_written")

    # --- 9. assessment gate fences ------------------------------------------
    gate_receipt = _assessment_gate_receipt(
        stage, calibrated, _load_gate_parameters()
    )
    for gate_id, payload in gate_receipt["evaluated"].items():
        verdict = payload.get("passed")
        state.gate_verdicts[gate_id] = {
            "verdict": "passed"
            if verdict
            else ("failed" if verdict is not None else "measured"),
            "receipt": f"local://{args.build_record_json.name}#/assessment_gates/"
            f"evaluated/{gate_id}",
        }
    append_phase(state, "assessment_gates_measured")

    # --- 10. build record + logbook ------------------------------------------
    record = {
        "schema_version": 1,
        "build_kind": "uk_spine_calibration_assessment",
        "status": "completed",
        "shippable": False,
        "shippable_reason": (
            "non-certified spine input; declared battery not enforced in "
            "spine-assessment posture"
        ),
        "build_id": state.build_id,
        "code_pin": code_pin,
        "pipeline": _PIPELINE,
        "calibration_year": calibration_year,
        "run_config": run_config,
        "input_posture": build_block["input_posture"],
        "structural_nan_repair": nan_receipt,
        "simulated_measures": build_block["simulated_measures"],
        "register": {
            "compiled_sha256": register_sha,
            "compiled_n_specs": len(compilation.registry.specs),
            "fenced_n_specs": len(fenced),
            "calibrated_n_specs": len(effective_registry.specs),
            "scored_n_specs": len(score_registry.specs),
        },
        "assessment_fences": fenced,
        "calibration": stage.manifest,
        "assessment_gates": gate_receipt,
        "score_vs_enhanced_frs": score,
        "artifacts": {
            "staging_h5": {
                "path": str(args.staging_h5),
                "sha256": staged_sha,
                "size_bytes": args.staging_h5.stat().st_size,
                "retention": "local_untracked",
            },
            "prepared_input_h5": {
                "path": str(prepared_input),
                "sha256": prepared_sha,
            },
            "calibration_diagnostics": {
                "path": str(args.diagnostics_json),
                "sha256": diagnostics_sha,
            },
            "incumbent_h5": {
                "path": str(args.incumbent_h5),
                "sha256": _sha256_file(args.incumbent_h5),
            },
        },
    }
    args.build_record_json.parent.mkdir(parents=True, exist_ok=True)
    args.build_record_json.write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    state.artifact_location = local_artifact_reference(
        args.staging_h5, repository_hint=_REPOSITORY
    )
    append_phase(state, "build_record_written")
    spool = record_terminal_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        pipeline=_PIPELINE,
        rung="f100",
        seed=UK_NATIONAL_SOLVE_DOCTRINE.seed,
        code_pin=code_pin,
        disposition="iterating",
        predecessor=resolve_predecessor(args.logbook_prev_row_digest),
        spool_dir=args.staging_h5.parent / "logbook-spool",
    )

    summary = {
        "build_id": state.build_id,
        "code_pin": code_pin,
        "final_loss": stage.manifest["loss"],
        "targets_calibrated": stage.manifest["matrix_target_count"],
        "targets_scored": len(score_registry.specs),
        "excluded_candidate": list(candidate_measures["excluded"]),
        "excluded_incumbent": list(incumbent_measures["excluded"]),
        "score_rule_1": {
            "candidate_full_loss": score["candidate_full_loss"],
            "incumbent_full_loss": score["incumbent_full_loss"],
            "candidate_target_wins": score["candidate_target_wins"],
            "incumbent_target_wins": score["incumbent_target_wins"],
        },
        "uk_target_fit": gate_receipt["evaluated"]["uk_target_fit"],
        "staging_h5_sha256": staged_sha,
        "diagnostics_sha256": diagnostics_sha,
        "logbook_spool": str(spool),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
