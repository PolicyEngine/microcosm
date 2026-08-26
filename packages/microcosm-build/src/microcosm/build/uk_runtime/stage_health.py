"""Stage-time health gates for the UK FRS spine build."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.resources import files

import numpy as np

from microcosm.build.gates import GateResult

_UK_PACKAGE = "microcosm.build.uk"


def uk_stage_health_gate(
    *,
    evidence: Mapping[str, object],
    stage: str,
    check: str,
    parameters: Mapping[str, object],
) -> GateResult:
    """Evaluate one spine-stage receipt against spec-declared thresholds."""

    if evidence.get("stage") not in {stage, _legacy_receipt_stage(stage)}:
        return GateResult(
            name="stage_health",
            passed=False,
            failures=(
                f"{stage}: receipt stage {evidence.get('stage')!r} does not match.",
            ),
            details={"stage": stage, "check": check},
        )
    if check == "support_clip":
        return _support_clip_gate(stage, evidence, parameters)
    if check == "realization_target":
        return _realization_target_gate(stage, evidence, parameters)
    if check == "student_loan_plans":
        return _student_loan_plans_gate(stage, evidence, parameters)
    if check == "cgt_incidence_mass":
        return _cgt_incidence_mass_gate(stage, evidence, parameters)
    if check == "spi_support_channel":
        return _spi_support_channel_gate(stage, evidence, parameters)
    if check == "spi_income_spine":
        return _spi_income_spine_gate(stage, evidence, parameters)
    if check == "source_signal":
        return _source_signal_gate(stage, evidence, parameters)
    if check == "age_tail_targets":
        return _age_tail_targets_gate(stage, evidence, parameters)
    if check == "cgt_band_donor_support":
        return _cgt_band_donor_support_gate(stage, evidence, parameters)
    if check == "cgt_imputation_summary":
        return _cgt_imputation_summary_gate(stage, evidence, parameters)
    return GateResult(
        name="stage_health",
        passed=False,
        failures=(f"{stage}: unknown stage-health check {check!r}.",),
        details={"stage": stage, "check": check},
    )


def _legacy_receipt_stage(stage: str) -> str:
    if stage == "age_tail":
        return "uk_age_tail_disaggregation"
    return stage


def _pass(stage: str, check: str, details: Mapping[str, object]) -> GateResult:
    return GateResult(
        name="stage_health",
        passed=True,
        details={"stage": stage, "check": check, **dict(details)},
    )


def _fail(
    stage: str,
    check: str,
    failures: list[str],
    details: Mapping[str, object],
) -> GateResult:
    return GateResult(
        name="stage_health",
        passed=False,
        failures=tuple(failures),
        details={"stage": stage, "check": check, **dict(details)},
    )


def _finite_number(value: object, *, label: str) -> float:
    if not isinstance(value, int | float) or not np.isfinite(float(value)):
        raise ValueError(f"{label} must be finite, got {value!r}.")
    return float(value)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    return value


def _support_clip_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    check = "support_clip"
    clip = _mapping(evidence.get("support_clip"), label=f"{stage}.support_clip")
    columns = _mapping(clip.get("columns"), label=f"{stage}.support_clip.columns")
    expected_columns = tuple(str(column) for column in parameters["columns"])
    exempt_columns = {str(column) for column in parameters.get("exempt_columns", ())}
    max_low = _mapping(
        parameters.get("max_clipped_low_rows_by_column", {}),
        label=f"{stage}.max_clipped_low_rows_by_column",
    )
    max_high = _mapping(
        parameters.get("max_clipped_high_rows_by_column", {}),
        label=f"{stage}.max_clipped_high_rows_by_column",
    )
    failures: list[str] = []
    for column in expected_columns:
        receipt = columns.get(column)
        if column in exempt_columns:
            if isinstance(receipt, Mapping) and receipt.get("exempt") is True:
                continue
            failures.append(f"{stage}: exempt column {column!r} is not marked exempt.")
            continue
        if not isinstance(receipt, Mapping):
            failures.append(f"{stage}: missing support-clip receipt for {column!r}.")
            continue
        for key in ("donor_min", "donor_max", "rows_considered"):
            if key not in receipt:
                failures.append(f"{stage}: {column!r} receipt is missing {key}.")
        low_rows = int(receipt.get("clipped_low_rows", -1))
        high_rows = int(receipt.get("clipped_high_rows", -1))
        allowed_low = max_low.get(column)
        allowed_high = max_high.get(column)
        # A missing allowance is not permission: without it the gate asserts
        # receipt shape only, and a stage clipping every row would pass a
        # release-blocking check — a green reflecting the absence of a check.
        if allowed_low is None:
            failures.append(
                f"{stage}: {column!r} declares no clipped_low_rows allowance; "
                "pin one at the receipted baseline or exempt the column."
            )
        elif low_rows > int(allowed_low):
            failures.append(
                f"{stage}: {column!r} clipped_low_rows {low_rows} exceeds {allowed_low}."
            )
        if allowed_high is None:
            failures.append(
                f"{stage}: {column!r} declares no clipped_high_rows allowance; "
                "pin one at the receipted baseline or exempt the column."
            )
        elif high_rows > int(allowed_high):
            failures.append(
                f"{stage}: {column!r} clipped_high_rows {high_rows} exceeds {allowed_high}."
            )
        if "donor_min" in receipt and "donor_max" in receipt:
            lower = _finite_number(receipt["donor_min"], label=f"{column}.donor_min")
            upper = _finite_number(receipt["donor_max"], label=f"{column}.donor_max")
            if lower > upper:
                failures.append(f"{stage}: {column!r} donor_min exceeds donor_max.")
    details = {
        "columns_checked": len(expected_columns) - len(exempt_columns),
        "exempt_columns": sorted(exempt_columns),
    }
    return _fail(stage, check, failures, details) if failures else _pass(stage, check, details)


def _realization_target_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    check = "realization_target"
    receipt = _mapping(
        evidence.get("headcount_receipt"), label=f"{stage}.headcount_receipt"
    )
    max_deviation = _finite_number(
        parameters["maximum_abs_realization_deviation"],
        label=f"{stage}.maximum_abs_realization_deviation",
    )
    target = _finite_number(parameters["target"], label=f"{stage}.target")
    failures: list[str] = []
    observed_target = _finite_number(receipt.get("target"), label=f"{stage}.target")
    if observed_target != target:
        failures.append(f"{stage}: target {observed_target} != declared {target}.")
    deviation = abs(
        _finite_number(
            receipt.get("realization_deviation"),
            label=f"{stage}.realization_deviation",
        )
    )
    if deviation > max_deviation:
        failures.append(
            f"{stage}: realization_deviation {deviation} exceeds {max_deviation}."
        )
    if bool(receipt.get("cap_bound")) and not bool(parameters.get("allow_cap_bound")):
        failures.append(f"{stage}: cap_bound is true but not allowed.")
    details = {"target": target, "abs_realization_deviation": deviation}
    return _fail(stage, check, failures, details) if failures else _pass(stage, check, details)


def _student_loan_plans_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    check = "student_loan_plans"
    plans = _mapping(evidence.get("plans"), label=f"{stage}.plans")
    declared_stocks = _mapping(parameters["stocks"], label=f"{stage}.stocks")
    max_deviation = _finite_number(
        parameters["maximum_abs_realization_deviation"],
        label=f"{stage}.maximum_abs_realization_deviation",
    )
    failures: list[str] = []
    worst = 0.0
    for plan, declared_stock in declared_stocks.items():
        receipt = plans.get(str(plan))
        if not isinstance(receipt, Mapping):
            failures.append(f"{stage}: missing receipt for {plan}.")
            continue
        stock = _finite_number(receipt.get("stock"), label=f"{stage}.{plan}.stock")
        expected = _finite_number(declared_stock, label=f"{stage}.{plan}.declared_stock")
        if stock != expected:
            failures.append(f"{stage}: {plan} stock {stock} != declared {expected}.")
        final = _finite_number(
            receipt.get("final_england_count"),
            label=f"{stage}.{plan}.final_england_count",
        )
        deviation = abs(
            _finite_number(
                receipt.get("realization_deviation"),
                label=f"{stage}.{plan}.realization_deviation",
            )
        )
        worst = max(worst, deviation)
        if final < 0.0:
            failures.append(f"{stage}: {plan} final_england_count is negative.")
        if deviation > max_deviation:
            failures.append(
                f"{stage}: {plan} realization_deviation {deviation} exceeds {max_deviation}."
            )
    details = {"plans_checked": len(declared_stocks), "worst_abs_deviation": worst}
    return _fail(stage, check, failures, details) if failures else _pass(stage, check, details)


def _cgt_incidence_mass_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    check = "cgt_incidence_mass"
    mass = _mapping(evidence.get("mass_by_clone_flag"), label=f"{stage}.mass_by_clone_flag")
    original = _finite_number(mass.get("false"), label=f"{stage}.mass.false")
    clone = _finite_number(mass.get("true"), label=f"{stage}.mass.true")
    tolerance = _finite_number(
        parameters["maximum_relative_mass_imbalance"],
        label=f"{stage}.maximum_relative_mass_imbalance",
    )
    denominator = max(abs(original), 1.0)
    imbalance = abs(clone - original) / denominator
    failures = []
    if original <= 0.0 or clone <= 0.0:
        failures.append(f"{stage}: clone and original mass must both be positive.")
    if imbalance > tolerance:
        failures.append(f"{stage}: clone/original mass imbalance {imbalance} exceeds {tolerance}.")
    details = {"original_mass": original, "clone_mass": clone, "relative_imbalance": imbalance}
    return _fail(stage, check, failures, details) if failures else _pass(stage, check, details)


def _spi_support_channel_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    check = "spi_support_channel"
    expected_share = _finite_number(
        parameters["spi_prior_mass_share"], label=f"{stage}.spi_prior_mass_share"
    )
    share = _finite_number(
        evidence.get("spi_prior_mass_share"), label=f"{stage}.spi_prior_mass_share"
    )
    failures = []
    if abs(share - expected_share) > _finite_number(
        parameters.get("absolute_tolerance", 0.0), label=f"{stage}.absolute_tolerance"
    ):
        failures.append(f"{stage}: spi_prior_mass_share {share} != declared {expected_share}.")
    if evidence.get("household_weight_kind") != parameters.get("household_weight_kind"):
        failures.append(f"{stage}: household_weight_kind drifted.")
    if int(evidence.get("spi_households", 0)) < int(parameters["minimum_spi_households"]):
        failures.append(f"{stage}: spi_households below declared minimum.")
    details = {"spi_prior_mass_share": share, "spi_households": evidence.get("spi_households")}
    return _fail(stage, check, failures, details) if failures else _pass(stage, check, details)


def _spi_income_spine_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    check = "spi_income_spine"
    identity = _mapping(
        evidence.get("post_draw_identity"), label=f"{stage}.post_draw_identity"
    )
    prior = _mapping(evidence.get("spi_prior"), label=f"{stage}.spi_prior")
    targets = _mapping(evidence.get("targets"), label=f"{stage}.targets")
    failures: list[str] = []
    if identity.get("exact") is not True:
        failures.append(f"{stage}: post_draw_identity.exact is not true.")
    if int(identity.get("rows_checked", 0)) < int(parameters["minimum_identity_rows"]):
        failures.append(f"{stage}: post_draw_identity checked too few rows.")
    expected_share = _finite_number(
        parameters["spi_prior_mass_share"], label=f"{stage}.spi_prior_mass_share"
    )
    share = _finite_number(prior.get("mass_share"), label=f"{stage}.spi_prior.mass_share")
    if abs(share - expected_share) > _finite_number(
        parameters.get("absolute_tolerance", 0.0), label=f"{stage}.absolute_tolerance"
    ):
        failures.append(f"{stage}: spi prior mass share {share} != declared {expected_share}.")
    if int(targets.get("count", 0)) < int(parameters["minimum_target_count"]):
        failures.append(f"{stage}: target count below declared minimum.")
    details = {"identity_rows": identity.get("rows_checked"), "target_count": targets.get("count")}
    return _fail(stage, check, failures, details) if failures else _pass(stage, check, details)


def _source_signal_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    check = "source_signal"
    rows = _mapping(evidence.get("source_signal_rows"), label=f"{stage}.source_signal_rows")
    allowed_zero = {str(column) for column in parameters.get("structural_zero_columns", ())}
    reported_zero = {str(column) for column in evidence.get("structural_zero_columns", ())}
    minimum = int(parameters["minimum_signal_rows"])
    failures: list[str] = []
    if reported_zero - allowed_zero:
        failures.append(f"{stage}: unreviewed structural zero columns {sorted(reported_zero - allowed_zero)}.")
    for column, value in rows.items():
        if str(column) in allowed_zero:
            continue
        if int(value) < minimum:
            failures.append(f"{stage}: {column} has {value} source-signal row(s), below {minimum}.")
    details = {"columns_checked": len(rows), "structural_zero_columns": sorted(reported_zero)}
    return _fail(stage, check, failures, details) if failures else _pass(stage, check, details)


def _age_tail_targets_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    check = "age_tail_targets"
    achieved = _mapping(evidence.get("achieved_weighted"), label=f"{stage}.achieved_weighted")
    targets = _mapping(evidence.get("band_populations"), label=f"{stage}.band_populations")
    max_relative = _finite_number(
        parameters["maximum_relative_deviation"],
        label=f"{stage}.maximum_relative_deviation",
    )
    failures: list[str] = []
    worst = 0.0
    for key, target_value in targets.items():
        if ":" not in str(key):
            continue
        gender, band = str(key).split(":", 1)
        gender_rows = achieved.get(gender)
        if not isinstance(gender_rows, Mapping) or band not in gender_rows:
            failures.append(f"{stage}: missing achieved band {key}.")
            continue
        target = _finite_number(target_value, label=f"{stage}.{key}.target")
        value = _finite_number(gender_rows[band], label=f"{stage}.{key}.achieved")
        relative = abs(value - target) / max(abs(target), 1.0)
        worst = max(worst, relative)
        if relative > max_relative:
            failures.append(f"{stage}: {key} relative deviation {relative} exceeds {max_relative}.")
    details = {"bands_checked": len(targets), "worst_relative_deviation": worst}
    return _fail(stage, check, failures, details) if failures else _pass(stage, check, details)


def _cgt_band_donor_support_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    check = "cgt_band_donor_support"
    resource_name = str(parameters["support_bounds_resource"])
    resource = json.loads(files(_UK_PACKAGE).joinpath(resource_name).read_text())
    bounds = _mapping(resource.get("bounds"), label=f"{resource_name}.bounds")
    lower, upper = bounds["capital_gains"]
    global_lower = _finite_number(lower, label="capital_gains.lower")
    bands = evidence.get("bands")
    if not isinstance(bands, list | tuple):
        raise ValueError(f"{stage}.bands must be a list.")
    failures: list[str] = []
    for row in bands:
        if not isinstance(row, Mapping):
            failures.append(f"{stage}: band row is not an object.")
            continue
        realized_min = _finite_number(row.get("realized_min_gain"), label=f"{stage}.realized_min_gain")
        realized_max = _finite_number(row.get("realized_max_gain"), label=f"{stage}.realized_max_gain")
        lower_limit = _finite_number(row.get("lower_limit"), label=f"{stage}.lower_limit")
        band_floor = max(global_lower, lower_limit)
        if realized_min < band_floor:
            failures.append(f"{stage}: realized gain {realized_min} falls below {band_floor}.")
        if upper is not None and realized_max >= _finite_number(upper, label="capital_gains.upper"):
            failures.append(f"{stage}: realized gain {realized_max} exceeds open upper bound.")
    details = {"bands_checked": len(bands), "minimum_lower_limit": global_lower}
    return _fail(stage, check, failures, details) if failures else _pass(stage, check, details)


def _cgt_imputation_summary_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    check = "cgt_imputation_summary"
    rows = evidence.get("rows")
    if not isinstance(rows, list | tuple):
        raise ValueError(f"{stage}.rows must be a list.")
    failures: list[str] = []
    min_rows = int(parameters["minimum_band_rows"])
    if len(rows) < min_rows:
        failures.append(f"{stage}: summary row count {len(rows)} below {min_rows}.")
    for key in ("taxpayer_mass", "published_taxpayer_mass", "remainder_mass"):
        value = _finite_number(evidence.get(key), label=f"{stage}.{key}")
        if value < 0.0:
            failures.append(f"{stage}: {key} is negative.")
    details = {"band_rows": len(rows), "taxpayer_mass": evidence.get("taxpayer_mass")}
    return _fail(stage, check, failures, details) if failures else _pass(stage, check, details)
