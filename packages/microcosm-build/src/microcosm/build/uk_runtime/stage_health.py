"""Stage-time health gates for the UK FRS spine build."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from importlib.resources import files

import numpy as np

from microcosm.build.gates import GateResult

_UK_PACKAGE = "microcosm.build.uk"
_FLOAT_RELATIVE_TOLERANCE = 8.0 * np.finfo(np.float64).eps


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
    if check == "spi_income_band_donor_support":
        return _spi_income_band_donor_support_gate(stage, evidence, parameters)
    if check == "cgt_imputation_summary":
        return _cgt_imputation_summary_gate(stage, evidence, parameters)
    if check == "cgt_asset_type_summary":
        return _cgt_asset_type_summary_gate(stage, evidence, parameters)
    if check == "cgt_incidence_anchor":
        return _cgt_incidence_anchor_gate(stage, evidence, parameters)
    if check == "latent_attribute_realization":
        return _latent_attribute_realization_gate(stage, evidence)
    if check == "household_composition":
        return _household_composition_gate(stage, evidence, parameters)
    if check == "energy_rake":
        return _energy_rake_gate(stage, evidence, parameters)
    if check == "bus_travel_facts":
        return _bus_travel_facts_gate(stage, evidence, parameters)
    if check == "bus_pricing":
        return _bus_pricing_gate(stage, evidence, parameters)
    if check == "bus_support_pricing":
        return _bus_support_pricing_gate(stage, evidence, parameters)
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
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _energy_rake_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The energy kWh rake fits the NEED shape at the DESNZ level, at design weights.

    Fact checks on the lcfs ``energy_rake`` receipt, every published value
    recomputed here from the vendored rows the stage declares (never taken
    from the receipt): each cell's ``shape_target`` must be the vendored NEED
    mean of the declared consumption year; each fuel's level block must be
    the vendored DESNZ Energy Trends fiscal-year total, its ``factor`` that
    total over the frame's pre-level total, and the levelled frame total the
    published one; each cell's ``target`` must be shape times factor; and,
    where the stage declares the published gas-connected share, every region
    with a published share must sit within ``maximum_connected_share_deviation``
    of it after the imposition (a region the draw could not fill records a
    shortfall instead). The residual check then holds the maximum absolute
    relative deviation of any cell mean from its levelled target, per margin
    and fuel, to ``maximum_relative_deviation``, one fixed tolerance on the
    IPF's cross-margin residual. The rake must have run in kWh with gas over
    gas-connected rows and no zero-current cell; a missing margin, block or
    tolerance fails closed.

    This is where NEED, DESNZ and the connection share are checked; the
    calibrated frame is held to the bound ONS 04.5.1 and 04.5.2 spend rows
    instead (María's ruling, 2026-09-15), and no gate re-checks these facts
    after calibration.
    """

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.energy_pricing import (
        ELECTRICITY_KWH,
        GAS_CONNECTED_PUBLISHED_METER_SHARE,
        GAS_KWH,
        PRICE_DOMESTIC_ENERGY_KIND,
        need_margins_from_facts,
        pricing_operation,
        published_energy_level,
        published_gas_connected_shares,
    )

    check = "energy_rake"
    receipt = _mapping(evidence.get("energy_rake"), label=f"{stage}.energy_rake")
    tolerance = _finite_number(
        parameters.get("maximum_relative_deviation"),
        label=f"{stage}.maximum_relative_deviation",
    )
    share_tolerance = _finite_number(
        parameters.get("maximum_connected_share_deviation"),
        label=f"{stage}.maximum_connected_share_deviation",
    )
    expected_margins = [str(m) for m in parameters.get("margins", ())]
    if not expected_margins:
        raise ValueError(f"{stage}: energy_rake declares no margins.")
    margins_period = parameters.get("margins_period_value")
    if not isinstance(margins_period, int) or isinstance(margins_period, bool):
        raise ValueError(f"{stage}: energy_rake declares no margins_period_value.")
    declared = pricing_operation(load_country_spec("uk").sources.stage_map()[stage])
    if declared is None:
        raise ValueError(
            f"{stage}: declares no {PRICE_DOMESTIC_ENERGY_KIND} operation."
        )
    if int(declared.get("margins_period_value", -1)) != margins_period:
        raise ValueError(
            f"{stage}: the gate's margins_period_value {margins_period} differs from "
            f"the stage's declared {declared.get('margins_period_value')!r}."
        )
    failures: list[str] = []
    if receipt.get("unit") != "kwh":
        failures.append(
            f"{stage}: energy rake unit is {receipt.get('unit')!r}, not kwh."
        )
    if receipt.get("gas_rake_population") != "gas_connected_rows":
        failures.append(
            f"{stage}: gas rake population is "
            f"{receipt.get('gas_rake_population')!r}, not gas_connected_rows."
        )
    if receipt.get("margins_period_value") != margins_period:
        failures.append(
            f"{stage}: receipt raked NEED {receipt.get('margins_period_value')!r}, "
            f"not the declared consumption year {margins_period}."
        )
    fit = receipt.get("fit")
    if not isinstance(fit, Mapping):
        failures.append(f"{stage}: energy_rake receipt carries no fit block.")
        fit = {}
    declared_margins = [str(m) for m in receipt.get("margins", ())]
    undeclared = sorted(set(declared_margins) - set(expected_margins))
    if undeclared:
        failures.append(f"{stage}: receipt rakes undeclared margins {undeclared}.")
    published = need_margins_from_facts(period_value=margins_period).targets
    level, _ = published_energy_level(declared)
    factors_block = receipt.get("level_factor")
    level_block = receipt.get("level")
    factors: dict[str, float] = {}
    details: dict[str, object] = {
        "margins": expected_margins,
        "margins_period_value": margins_period,
        "maximum_relative_deviation": tolerance,
        "maximum_connected_share_deviation": share_tolerance,
        "level_factor": {},
        "worst": {},
        "cells_fact_checked": 0,
        "connected_share": {},
    }
    if not isinstance(factors_block, Mapping) or not isinstance(level_block, Mapping):
        failures.append(f"{stage}: energy_rake receipt carries no level block.")
        factors_block, level_block = {}, {}

    def _close(observed: object, expected: float, rtol: float) -> bool:
        return isinstance(observed, int | float) and abs(
            float(observed) - expected
        ) <= rtol * max(1.0, abs(expected))

    for fuel in (ELECTRICITY_KWH, GAS_KWH):
        block = level_block.get(fuel)
        factor = factors_block.get(fuel)
        if not isinstance(block, Mapping) or not isinstance(factor, int | float):
            failures.append(f"{stage}: level block lacks {fuel}.")
            continue
        published_kwh = float(level[fuel])
        before = block.get("frame_kwh_before")
        if not _close(block.get("published_kwh"), published_kwh, 1e-9):
            failures.append(
                f"{stage}: {fuel} was levelled to {block.get('published_kwh')!r}, not "
                f"the vendored DESNZ total {published_kwh}."
            )
        if not isinstance(before, int | float) or float(before) <= 0:
            failures.append(f"{stage}: {fuel} level block has no positive frame total.")
        elif not _close(factor, published_kwh / float(before), 1e-9) or not _close(
            block.get("factor"), float(factor), 1e-12
        ):
            failures.append(
                f"{stage}: {fuel} level factor {factor!r} is not the published total "
                f"over the frame total {published_kwh / float(before)}."
            )
        if not _close(block.get("frame_kwh_after"), published_kwh, 1e-6):
            failures.append(
                f"{stage}: {fuel} frame total after levelling is "
                f"{block.get('frame_kwh_after')!r}, not the published {published_kwh}."
            )
        factors[fuel] = float(factor)
        details["level_factor"][fuel] = float(factor)
    for margin in expected_margins:
        if margin not in declared_margins:
            failures.append(f"{stage}: margin {margin!r} was not raked.")
            continue
        block = fit.get(margin)
        if not isinstance(block, Mapping) or not isinstance(
            block.get("max_abs_relative_deviation"), Mapping
        ):
            failures.append(f"{stage}: fit carries no block for margin {margin!r}.")
            continue
        for key, cell in _mapping(
            block.get("cells"), label=f"{stage}.{margin}.cells"
        ).items():
            geography, _, category = str(key).partition(":")
            fact = published.get(margin, {}).get((geography, category))
            if fact is None:
                failures.append(
                    f"{stage}: {margin} cell {key!r} has no vendored NEED row."
                )
                continue
            for fuel in (ELECTRICITY_KWH, GAS_KWH):
                entry = _mapping(
                    _mapping(cell, label=f"{stage}.{key}").get(fuel),
                    label=f"{stage}.{key}.{fuel}",
                )
                if not _close(entry.get("shape_target"), float(fact[fuel]), 1e-6):
                    failures.append(
                        f"{stage}: {margin} cell {key!r} {fuel} was raked to "
                        f"{entry.get('shape_target')!r}, not the vendored NEED mean "
                        f"{fact[fuel]}."
                    )
                if fuel in factors and not _close(
                    entry.get("target"), float(fact[fuel]) * factors[fuel], 1e-9
                ):
                    failures.append(
                        f"{stage}: {margin} cell {key!r} {fuel} target "
                        f"{entry.get('target')!r} is not the NEED mean times the level "
                        f"factor {factors[fuel]}."
                    )
            details["cells_fact_checked"] = int(details["cells_fact_checked"]) + 1
        worst = block["max_abs_relative_deviation"]
        for fuel in (ELECTRICITY_KWH, GAS_KWH):
            value = _finite_number(worst.get(fuel), label=f"{stage}.{margin}.{fuel}")
            details["worst"][f"{margin}:{fuel}"] = value
            if value > tolerance:
                failures.append(
                    f"{stage}: {margin} {fuel} cell mean deviates {value:.4f} "
                    f"from its levelled NEED target, above the residual tolerance "
                    f"{tolerance}."
                )
    zero_cells = receipt.get("zero_current_cells")
    if zero_cells:
        failures.append(
            f"{stage}: {len(zero_cells)} NEED cell(s) had a zero current mean and "
            "could not be raked."
        )
    if declared.get("gas_connected") == GAS_CONNECTED_PUBLISHED_METER_SHARE:
        connection = receipt.get("gas_connection")
        if (
            not isinstance(connection, Mapping)
            or connection.get("rule") != GAS_CONNECTED_PUBLISHED_METER_SHARE
        ):
            failures.append(
                f"{stage}: gas connection was not imposed at the published meter share."
            )
        else:
            shares, _ = published_gas_connected_shares(declared)
            by_region = connection.get("by_region")
            by_region = by_region if isinstance(by_region, Mapping) else {}
            if not by_region:
                failures.append(f"{stage}: gas-connection receipt names no region.")
            unknown = sorted(set(by_region) - set(shares))
            if unknown:
                failures.append(
                    f"{stage}: gas-connection receipt names regions outside the "
                    f"crosswalk {unknown}."
                )
            # Every region the frame carried is in the receipt (the imposition
            # walks the frame's regions); each with a published share is checked.
            for region, entry in sorted(by_region.items()):
                target = shares.get(region)
                if target is None or not isinstance(entry, Mapping):
                    continue
                after = entry.get("share_after")
                shortfall = entry.get("shortfall") or 0.0
                if not isinstance(after, int | float):
                    failures.append(f"{stage}: {region} has no connected share.")
                    continue
                details["connected_share"][region] = {
                    "published": target,
                    "achieved": float(after),
                    "shortfall": float(shortfall),
                }
                if (
                    float(shortfall) <= 0
                    and abs(float(after) - target) > share_tolerance
                ):
                    failures.append(
                        f"{stage}: {region} gas-connected share {float(after):.4f} is "
                        f"not the published {target:.4f} (tolerance {share_tolerance})."
                    )
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


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
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


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
        expected = _finite_number(
            declared_stock, label=f"{stage}.{plan}.declared_stock"
        )
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
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _cgt_incidence_mass_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    check = "cgt_incidence_mass"
    mass = _mapping(
        evidence.get("mass_by_clone_flag"), label=f"{stage}.mass_by_clone_flag"
    )
    original = _finite_number(mass.get("false"), label=f"{stage}.mass.false")
    clone = _finite_number(mass.get("true"), label=f"{stage}.mass.true")
    tolerance = _finite_number(
        parameters["maximum_relative_mass_imbalance"],
        label=f"{stage}.maximum_relative_mass_imbalance",
    )
    denominator = max(abs(original), 1.0)
    imbalance = abs(clone - original) / denominator
    effective_tolerance = max(tolerance, _FLOAT_RELATIVE_TOLERANCE)
    failures = []
    if original <= 0.0 or clone <= 0.0:
        failures.append(f"{stage}: clone and original mass must both be positive.")
    if imbalance > effective_tolerance:
        failures.append(
            f"{stage}: clone/original mass imbalance {imbalance} exceeds "
            f"effective tolerance {effective_tolerance} (the greater of "
            f"configured tolerance {tolerance} and floating-point comparison "
            f"tolerance {_FLOAT_RELATIVE_TOLERANCE})."
        )
    details = {
        "original_mass": original,
        "clone_mass": clone,
        "relative_imbalance": imbalance,
        "configured_relative_tolerance": tolerance,
        "floating_point_relative_tolerance": _FLOAT_RELATIVE_TOLERANCE,
        "effective_relative_tolerance": effective_tolerance,
    }
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


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
        failures.append(
            f"{stage}: spi_prior_mass_share {share} != declared {expected_share}."
        )
    if evidence.get("household_weight_kind") != parameters.get("household_weight_kind"):
        failures.append(f"{stage}: household_weight_kind drifted.")
    if int(evidence.get("spi_households", 0)) < int(
        parameters["minimum_spi_households"]
    ):
        failures.append(f"{stage}: spi_households below declared minimum.")
    details = {
        "spi_prior_mass_share": share,
        "spi_households": evidence.get("spi_households"),
    }
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


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
    share = _finite_number(
        prior.get("mass_share"), label=f"{stage}.spi_prior.mass_share"
    )
    if abs(share - expected_share) > _finite_number(
        parameters.get("absolute_tolerance", 0.0), label=f"{stage}.absolute_tolerance"
    ):
        failures.append(
            f"{stage}: spi prior mass share {share} != declared {expected_share}."
        )
    if int(targets.get("count", 0)) < int(parameters["minimum_target_count"]):
        failures.append(f"{stage}: target count below declared minimum.")
    details = {
        "identity_rows": identity.get("rows_checked"),
        "target_count": targets.get("count"),
    }
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _source_signal_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    check = "source_signal"
    rows = _mapping(
        evidence.get("source_signal_rows"), label=f"{stage}.source_signal_rows"
    )
    allowed_zero = {
        str(column) for column in parameters.get("structural_zero_columns", ())
    }
    reported_zero = {
        str(column) for column in evidence.get("structural_zero_columns", ())
    }
    minimum = int(parameters["minimum_signal_rows"])
    failures: list[str] = []
    if reported_zero - allowed_zero:
        failures.append(
            f"{stage}: unreviewed structural zero columns {sorted(reported_zero - allowed_zero)}."
        )
    for column, value in rows.items():
        if str(column) in allowed_zero:
            continue
        if int(value) < minimum:
            failures.append(
                f"{stage}: {column} has {value} source-signal row(s), below {minimum}."
            )
    details = {
        "columns_checked": len(rows),
        "structural_zero_columns": sorted(reported_zero),
    }
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _age_tail_targets_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    check = "age_tail_targets"
    achieved = _mapping(
        evidence.get("achieved_weighted"), label=f"{stage}.achieved_weighted"
    )
    targets = _mapping(
        evidence.get("band_populations"), label=f"{stage}.band_populations"
    )
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
            failures.append(
                f"{stage}: {key} relative deviation {relative} exceeds {max_relative}."
            )
    details = {"bands_checked": len(targets), "worst_relative_deviation": worst}
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


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
        realized_min = _finite_number(
            row.get("realized_min_gain"), label=f"{stage}.realized_min_gain"
        )
        realized_max = _finite_number(
            row.get("realized_max_gain"), label=f"{stage}.realized_max_gain"
        )
        lower_limit = _finite_number(
            row.get("lower_limit"), label=f"{stage}.lower_limit"
        )
        band_floor = max(global_lower, lower_limit)
        if realized_min < band_floor:
            failures.append(
                f"{stage}: realized gain {realized_min} falls below {band_floor}."
            )
        if upper is not None and realized_max >= _finite_number(
            upper, label="capital_gains.upper"
        ):
            failures.append(
                f"{stage}: realized gain {realized_max} exceeds open upper bound."
            )
    details = {"bands_checked": len(bands), "minimum_lower_limit": global_lower}
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _spi_income_band_donor_support_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """Every reserved band carries its donors at positive band-exact weight."""

    check = "spi_income_band_donor_support"
    donors_per_band = int(parameters["donors_per_band"])
    expected_bands = [int(value) for value in parameters["band_lower_bounds"]]
    bands = evidence.get("bands")
    if not isinstance(bands, list | tuple):
        raise ValueError(f"{stage}.bands must be a list.")
    failures: list[str] = []
    seen: list[int] = []
    for row in bands:
        if not isinstance(row, Mapping):
            failures.append(f"{stage}: band row is not an object.")
            continue
        lower = int(
            _finite_number(row.get("lower_bound"), label=f"{stage}.lower_bound")
        )
        seen.append(lower)
        donors = int(
            _finite_number(
                row.get("donor_households"), label=f"{stage}.donor_households"
            )
        )
        carriers = int(_finite_number(row.get("carriers"), label=f"{stage}.carriers"))
        weight = _finite_number(row.get("donor_weight"), label=f"{stage}.donor_weight")
        if donors != carriers:
            failures.append(
                f"{stage}: band from {lower} has {donors} donors but {carriers} carriers."
            )
        if weight <= 0.0:
            failures.append(
                f"{stage}: band from {lower} donor weight {weight} is not positive."
            )
        published = _finite_number(
            row.get("published_taxpayers"), label=f"{stage}.published_taxpayers"
        )
        if abs(weight * donors - published) > 0.5 and donors > 0:
            # A scaled rung stacks fewer donors than the declared count; its
            # weighted taxpayers then fall short of the published band mass by
            # construction, which the evidence still has to show honestly.
            if donors == donors_per_band:
                failures.append(
                    f"{stage}: band from {lower} weighted taxpayers "
                    f"{weight * donors} differ from the published {published}."
                )
    if sorted(seen) != sorted(expected_bands):
        failures.append(
            f"{stage}: bands {sorted(seen)} differ from the declared {sorted(expected_bands)}."
        )
    details = {
        "bands_checked": len(bands),
        "donors_per_band": donors_per_band,
        "evidence_donors_per_band": evidence.get("donors_per_band"),
    }
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


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
    # The conditioned redraw (microcosm#725) reports its rake and fallback;
    # a receipt that carries them must carry them finite and non-negative.
    # No threshold is held yet: the first measured builds set it.
    allocation = evidence.get("allocation")
    if allocation is not None:
        if not isinstance(allocation, Mapping):
            raise ValueError(f"{stage}.allocation must be a mapping.")
        rake = allocation.get("rake")
        if not isinstance(rake, Mapping):
            raise ValueError(f"{stage}.allocation.rake must be a mapping.")
        for key in (
            "ipf_max_abs_margin_error",
            "gains_margin_max_abs_error",
            "ipf_zero_seed_cells",
        ):
            value = _finite_number(
                rake.get(key), label=f"{stage}.allocation.rake.{key}"
            )
            if value < 0.0:
                failures.append(f"{stage}: allocation.rake.{key} is negative.")
        released = _finite_number(
            allocation.get("fallback_released_mass"),
            label=f"{stage}.allocation.fallback_released_mass",
        )
        if released < 0.0:
            failures.append(f"{stage}: allocation.fallback_released_mass is negative.")
        details["ipf_max_abs_margin_error"] = rake.get("ipf_max_abs_margin_error")
        details["gains_margin_max_abs_error"] = rake.get("gains_margin_max_abs_error")
        details["fallback_released_mass"] = released
        # The sub-AEA remainder receipt (microcosm#970): every remainder
        # amount must sit strictly above zero and at or below the exempt
        # amount, or the projection fence downstream measures the wrong
        # population. Optional so receipts predating the mapping still read.
        remainder = allocation.get("remainder")
        if remainder is not None:
            if not isinstance(remainder, Mapping):
                raise ValueError(f"{stage}.allocation.remainder must be a mapping.")
            persons = _finite_number(
                remainder.get("persons"), label=f"{stage}.allocation.remainder.persons"
            )
            remainder_mass = _finite_number(
                remainder.get("mass"), label=f"{stage}.allocation.remainder.mass"
            )
            if persons < 0.0 or remainder_mass < 0.0:
                failures.append(
                    f"{stage}: allocation.remainder count or mass is negative."
                )
            if persons > 0.0:
                exempt = _finite_number(
                    remainder.get("annual_exempt_amount"),
                    label=f"{stage}.allocation.remainder.annual_exempt_amount",
                )
                low = _finite_number(
                    remainder.get("min_amount"),
                    label=f"{stage}.allocation.remainder.min_amount",
                )
                high = _finite_number(
                    remainder.get("max_amount"),
                    label=f"{stage}.allocation.remainder.max_amount",
                )
                if not low > 0.0:
                    failures.append(
                        f"{stage}: allocation.remainder.min_amount is not positive."
                    )
                if high > exempt:
                    failures.append(
                        f"{stage}: allocation.remainder.max_amount exceeds the "
                        "annual exempt amount."
                    )
            details["remainder_mass"] = remainder_mass
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _cgt_asset_type_summary_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The residential flag realised the Table 8a totals it was solved to.

    The stage solves the logistic exactly in expectation and realises it by
    systematic sampling. This gate holds the solve to its targets, the
    realised weighted count to within one carrier row's weight of the
    expectation (``max_liable_weight``, the weighted systematic walk's
    deterministic bound; about 3,300 people at full scale), and the realised
    gains to the wider of the reviewed relative band and a multiple of the
    Bernoulli sigma the stage reports, which overstates a systematic draw's
    noise and so is a conservative envelope, so a tiny frame is judged by its
    noise floor and a production frame by the band. Every liable gainer must
    carry an asset type and the composition receipt must be finite
    (microcosm#725).
    """

    check = "cgt_asset_type_summary"
    failures: list[str] = []
    residential = _mapping(evidence.get("residential"), label=f"{stage}.residential")
    max_relative = _finite_number(
        parameters["maximum_relative_deviation"],
        label=f"{stage}.maximum_relative_deviation",
    )
    max_sigma = _finite_number(
        parameters["maximum_gains_sigma"], label=f"{stage}.maximum_gains_sigma"
    )
    max_solve_error = _finite_number(
        parameters["maximum_solve_relative_error"],
        label=f"{stage}.maximum_solve_relative_error",
    )
    details: dict[str, object] = {}

    def number(key: str) -> float:
        return _finite_number(residential.get(key), label=f"{stage}.residential.{key}")

    for measure in ("count", "gains"):
        target = number(f"{measure}_target_individuals_basis")
        expected = number(f"expected_{measure}")
        if target <= 0.0:
            failures.append(f"{stage}: residential {measure} target is not positive.")
            continue
        solve_error = abs(expected - target) / target
        details[f"residential_{measure}_solve_relative_error"] = solve_error
        if solve_error > max_solve_error:
            failures.append(
                f"{stage}: residential {measure} solve error {solve_error} "
                f"exceeds {max_solve_error}."
            )
    count_gap = abs(number("achieved_count") - number("expected_count"))
    count_bound = number("max_liable_weight") * (1.0 + 1e-9)
    details["residential_count_gap"] = count_gap
    if count_gap > count_bound:
        failures.append(
            f"{stage}: residential count gap {count_gap} exceeds one person "
            f"({count_bound})."
        )
    gains_target = number("gains_target_individuals_basis")
    gains_gap = abs(number("achieved_gains") - number("expected_gains"))
    gains_bound = max(
        max_relative * gains_target, max_sigma * number("gains_bernoulli_sigma")
    )
    details["residential_gains_gap"] = gains_gap
    details["residential_gains_bound"] = gains_bound
    if gains_target > 0.0 and gains_gap > gains_bound:
        failures.append(
            f"{stage}: residential gains gap {gains_gap} exceeds {gains_bound} "
            f"(the wider of {max_relative} relative and {max_sigma} sigma)."
        )
    counts = _mapping(evidence.get("value_counts"), label=f"{stage}.value_counts")
    for value, rows in counts.items():
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
            failures.append(f"{stage}: value_counts[{value!r}] is not a row count.")
    asset_type = _mapping(evidence.get("asset_type"), label=f"{stage}.asset_type")
    shares = _mapping(
        asset_type.get("achieved_gains_share"),
        label=f"{stage}.asset_type.achieved_gains_share",
    )
    for name, share in shares.items():
        value = _finite_number(share, label=f"{stage}.asset_type.{name}")
        if value < 0.0 or value > 1.0:
            failures.append(f"{stage}: {name} gains share {value} is not a share.")
    details["residential_rows"] = residential.get("achieved_rows")
    details["classified_values"] = sorted(counts)
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _cgt_incidence_anchor_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The anchor realised its reporter composition without touching anyone else.

    The stage derives a target for the sub-exempt and loss-making clone mass
    from the redrawn liable mass and the Advani-Summers composition, then
    moves the excess to the paired originals. This gate holds each group to
    its target when it was trimmed (never overshooting, never gaining mass,
    untouched when it was already at or below target), the liable clone mass
    to exactly its pre-anchor value, every pair's mass to rounding, and the
    clone side to no more than the original side; the pairing must cover at
    least ``minimum_pair_count`` households (microcosm#970).
    """

    check = "cgt_incidence_anchor"
    failures: list[str] = []
    max_relative = _finite_number(
        parameters["maximum_relative_composition_error"],
        label=f"{stage}.maximum_relative_composition_error",
    )
    max_pair_error = max(
        _finite_number(
            parameters["maximum_pair_relative_error"],
            label=f"{stage}.maximum_pair_relative_error",
        ),
        _FLOAT_RELATIVE_TOLERANCE,
    )
    minimum_pairs = parameters["minimum_pair_count"]
    if not isinstance(minimum_pairs, int) or isinstance(minimum_pairs, bool):
        raise ValueError(f"{stage}.minimum_pair_count must be an integer.")
    liable_mass = _finite_number(
        evidence.get("liable_mass"), label=f"{stage}.liable_mass"
    )
    transferred = _finite_number(
        evidence.get("transferred_mass"), label=f"{stage}.transferred_mass"
    )
    pair_error = _finite_number(
        evidence.get("max_pair_relative_error"),
        label=f"{stage}.max_pair_relative_error",
    )
    pair_count = evidence.get("pair_count")
    if not isinstance(pair_count, int) or isinstance(pair_count, bool):
        raise ValueError(f"{stage}.pair_count must be an integer.")
    targets = _mapping(evidence.get("targets"), label=f"{stage}.targets")
    before = _mapping(evidence.get("before"), label=f"{stage}.before")
    after = _mapping(evidence.get("after"), label=f"{stage}.after")
    mass = _mapping(
        evidence.get("mass_by_clone_flag"), label=f"{stage}.mass_by_clone_flag"
    )
    details: dict[str, object] = {
        "liable_mass": liable_mass,
        "transferred_mass": transferred,
        "pair_count": pair_count,
        "max_pair_relative_error": pair_error,
        "effective_pair_relative_tolerance": max_pair_error,
    }
    if liable_mass <= 0.0:
        failures.append(f"{stage}: liable mass must be positive.")
    if transferred < 0.0:
        failures.append(f"{stage}: transferred mass must be non-negative.")
    if pair_count < minimum_pairs:
        failures.append(
            f"{stage}: {pair_count} clone/original pairs is below the required "
            f"{minimum_pairs}."
        )
    if pair_error > max_pair_error:
        failures.append(
            f"{stage}: pair mass error {pair_error} exceeds {max_pair_error}."
        )
    removed = 0.0
    for group in ("sub_exempt", "loss"):
        target = _finite_number(targets.get(group), label=f"{stage}.targets.{group}")
        was = _finite_number(before.get(group), label=f"{stage}.before.{group}")
        now = _finite_number(after.get(group), label=f"{stage}.after.{group}")
        removed += was - now
        details[f"{group}_target"] = target
        details[f"{group}_before"] = was
        details[f"{group}_after"] = now
        if target <= 0.0:
            failures.append(f"{stage}: {group} target must be positive.")
            continue
        if now > was * (1.0 + _FLOAT_RELATIVE_TOLERANCE):
            failures.append(f"{stage}: {group} clone mass rose from {was} to {now}.")
        if was > target:
            error = abs(now - target) / target
            details[f"{group}_relative_error"] = error
            if error > max(max_relative, _FLOAT_RELATIVE_TOLERANCE):
                failures.append(
                    f"{stage}: {group} clone mass {now} misses its target {target} "
                    f"by {error} (allowed {max_relative})."
                )
        elif not np.isclose(now, was, rtol=_FLOAT_RELATIVE_TOLERANCE, atol=0.0):
            failures.append(
                f"{stage}: {group} clone mass {was} was already at or below its "
                f"target {target} but moved to {now}."
            )
    liable_before = _finite_number(before.get("liable"), label=f"{stage}.before.liable")
    liable_after = _finite_number(after.get("liable"), label=f"{stage}.after.liable")
    details["liable_clone_mass"] = liable_after
    if liable_after != liable_before:
        failures.append(
            f"{stage}: liable clone mass moved from {liable_before} to {liable_after}."
        )
    if not np.isclose(removed, transferred, rtol=1e-9, atol=0.0):
        failures.append(
            f"{stage}: removed group mass {removed} disagrees with the transferred "
            f"mass {transferred}."
        )
    original = _finite_number(mass.get("false"), label=f"{stage}.mass.false")
    clone = _finite_number(mass.get("true"), label=f"{stage}.mass.true")
    details["original_mass"] = original
    details["clone_mass"] = clone
    if clone > original * (1.0 + _FLOAT_RELATIVE_TOLERANCE):
        failures.append(
            f"{stage}: clone mass {clone} exceeds original mass {original}."
        )
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _latent_attribute_realization_gate(
    stage: str,
    evidence: Mapping[str, object],
) -> GateResult:
    """Check a latent-attribute stage's realization receipt.

    The receipt carries, per cell, the declared ``target`` share, the
    **unweighted** ``realized`` share over the ``rows`` behind it, and the
    producer's ``tolerance``. Identity-keyed draws give every unit the same
    probability regardless of its weight, so the unweighted share is the
    statistic that tests the mechanism; the weighted share (``realized_weighted``,
    informational) carries the frame's weight variance on top and is what the
    engine round-trip compares with the publisher. The gate owns the pass rule:
    it recomputes the binomial band from ``target`` and ``rows`` at
    :data:`LATENT_ATTRIBUTE_SIGMA` (floored at one row's worth, capped at one)
    and holds the cell to the tighter of the producer's figure and its own, so
    a widened producer tolerance cannot pass silently.
    """

    check = "latent_attribute_realization"
    failures: list[str] = []
    raw_coherence = evidence.get("coherence_violation_count")
    if not isinstance(raw_coherence, int) or isinstance(raw_coherence, bool):
        failures.append(f"{stage}: coherence_violation_count is missing.")
        coherence = None
    else:
        coherence = int(raw_coherence)
        if coherence != 0:
            failures.append(
                f"{stage}: coherence_violation_count is {coherence}, expected 0."
            )
    cells_checked = 0
    worst_ratio = 0.0
    for block_name in (
        "incidence_by_region",
        "latent_rate_bands",
        "combination_shares",
    ):
        block = _mapping(evidence.get(block_name), label=f"{stage}.{block_name}")
        if not block:
            failures.append(f"{stage}: {block_name} is empty.")
        for name, raw_row in block.items():
            if not isinstance(raw_row, Mapping):
                failures.append(f"{stage}: {block_name}.{name} is not an object.")
                continue
            label = f"{stage}.{block_name}.{name}"
            target = _finite_number(raw_row.get("target"), label=f"{label}.target")
            realized = _finite_number(
                raw_row.get("realized"), label=f"{label}.realized"
            )
            declared_tolerance = _finite_number(
                raw_row.get("tolerance"), label=f"{label}.tolerance"
            )
            rows = int(raw_row.get("rows", 0))
            cells_checked += 1
            if not 0.0 <= target <= 1.0 or not 0.0 <= realized <= 1.0:
                failures.append(f"{label}: target and realized must be shares.")
            if rows <= 0 or not 0.0 <= declared_tolerance <= 1.0:
                failures.append(f"{label}: has invalid rows/tolerance.")
                continue
            tolerance = min(
                declared_tolerance, _binomial_tolerance(target=target, rows=rows)
            )
            deviation = abs(realized - target)
            ratio = deviation / tolerance if tolerance > 0.0 else float("inf")
            worst_ratio = max(worst_ratio, ratio)
            if deviation > tolerance:
                failures.append(
                    f"{label}: deviation {deviation} exceeds {tolerance} "
                    f"(declared {declared_tolerance})."
                )
    details = {
        "cells_checked": cells_checked,
        "coherence_violation_count": coherence,
        "worst_tolerance_ratio": worst_ratio,
    }
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


#: Sigma multiple for the per-cell binomial band. A latent-attribute receipt
#: holds ~30 cells (regions, bands, combinations) per build; three sigma would
#: false-alarm on roughly one build in twelve, four sigma on one in five hundred.
LATENT_ATTRIBUTE_SIGMA = 4.0


def latent_attribute_tolerance(*, target: float, rows: int) -> float:
    """Binomial band for a share realized over ``rows`` identity-keyed draws."""

    if rows <= 0:
        return 1.0
    return min(
        1.0,
        max(
            1.0 / rows,
            LATENT_ATTRIBUTE_SIGMA * math.sqrt(target * (1.0 - target) / rows),
        ),
    )


def _binomial_tolerance(*, target: float, rows: int) -> float:
    return latent_attribute_tolerance(target=target, rows=rows)


def _household_composition_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The #791 relationship derivation left no tape defect unrefused.

    The stage refuses head, index, partner and domain defects itself; this
    gate re-reads the receipt so the battery, not only the transform, holds
    the invariants, and it pins the reviewed reciprocity tolerance and the
    partition closure (every household typed exactly once).
    """

    check = "household_composition"
    failures: list[str] = []
    zero_keys = (
        "head_invariant_violations",
        "hrpnum_mismatches",
        "person_index_gaps",
        "multi_partner_persons",
        "family_index_violations",
        "domain_violations",
    )
    details: dict[str, object] = {}
    for key in zero_keys:
        value = evidence.get(key)
        if not isinstance(value, int | float):
            failures.append(f"{stage}: receipt is missing {key}.")
            continue
        details[key] = int(value)
        if int(value) != 0:
            failures.append(f"{stage}: {key} is {int(value)}, expected 0.")
    tolerance = int(
        _finite_number(
            parameters["max_grid_reciprocity_mismatches"],
            label=f"{stage}.max_grid_reciprocity_mismatches",
        )
    )
    mismatches = evidence.get("grid_reciprocity_mismatches")
    if not isinstance(mismatches, int | float):
        failures.append(f"{stage}: receipt is missing grid_reciprocity_mismatches.")
    else:
        details["grid_reciprocity_mismatches"] = int(mismatches)
        details["max_grid_reciprocity_mismatches"] = tolerance
        if int(mismatches) > tolerance:
            failures.append(
                f"{stage}: grid_reciprocity_mismatches {int(mismatches)} exceeds "
                f"the reviewed tolerance {tolerance}."
            )
    unmapped = evidence.get("relhrp_unmapped_codes")
    if not isinstance(unmapped, Mapping):
        failures.append(f"{stage}: receipt is missing relhrp_unmapped_codes.")
    elif unmapped:
        failures.append(f"{stage}: unmapped relhrp codes {dict(unmapped)!r}.")
    details["relhrp_unmapped_codes"] = (
        dict(unmapped) if isinstance(unmapped, Mapping) else None
    )
    if bool(parameters.get("require_partition_closure", True)):
        counts = evidence.get("household_type_counts")
        households = evidence.get("households")
        if not isinstance(counts, Mapping) or not isinstance(households, int | float):
            failures.append(
                f"{stage}: receipt is missing household_type_counts or households."
            )
        else:
            total = int(sum(int(value) for value in counts.values()))
            details["household_type_counts"] = {
                str(key): int(value) for key, value in counts.items()
            }
            details["households"] = int(households)
            if total != int(households) or evidence.get("partition_closes") is not True:
                failures.append(
                    f"{stage}: household-type partition covers {total} of "
                    f"{int(households)} households."
                )
    weighted = evidence.get("household_type_weighted")
    if isinstance(weighted, Mapping):
        details["household_type_weighted"] = {
            str(key): float(value) for key, value in weighted.items()
        }
    if failures:
        return _fail(stage, check, failures, details)
    return _pass(stage, check, details)


def _bus_travel_facts_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The NTS bus-travel stage reproduces the published incidence and trip rates.

    Fact checks on the ``nts_bus_travel`` receipts, every published value
    recomputed here from the vendored rows the stage declares (never taken
    from the receipt): the frame's design-weighted share of persons who use
    a local bus at least yearly must sit within ``maximum_user_share_deviation``
    (points) of the vendored NTS0313 all-ages share, and the frame's mean
    local-bus trips per person per series within ``maximum_trip_rate_deviation``
    (relative) of the vendored NTS0303 rate of the declared year. The receipt
    must name its frequency source; a missing block fails closed.
    """

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.nts_bus_travel import (
        BAND_IDS,
        NON_USER_BAND,
        SERIES,
        published_band_shares,
        published_trip_rates,
    )

    check = "bus_travel_facts"
    incidence = _mapping(evidence.get("incidence"), label=f"{stage}.incidence")
    trip_rates = _mapping(evidence.get("trip_rates"), label=f"{stage}.trip_rates")
    share_tolerance = _finite_number(
        parameters.get("maximum_user_share_deviation"),
        label=f"{stage}.maximum_user_share_deviation",
    )
    rate_tolerance = _finite_number(
        parameters.get("maximum_trip_rate_deviation"),
        label=f"{stage}.maximum_trip_rate_deviation",
    )
    period_value = parameters.get("trip_rates_period_value")
    if not isinstance(period_value, int) or isinstance(period_value, bool):
        raise ValueError(
            f"{stage}: bus_travel_facts declares no trip_rates_period_value."
        )
    spec_stage = load_country_spec("uk").sources.stage_map()[stage]
    declared_band = next(
        (
            dict(op.parameters)
            for op in spec_stage.operations
            if op.kind == "impute_bus_use_band"
        ),
        None,
    )
    if declared_band is None:
        raise ValueError(f"{stage}: declares no impute_bus_use_band operation.")
    if int(declared_band.get("trip_rates_period_value", -1)) != period_value:
        raise ValueError(
            f"{stage}: the gate's trip_rates_period_value {period_value} differs from "
            f"the stage's declared {declared_band.get('trip_rates_period_value')!r}."
        )
    failures: list[str] = []
    details: dict[str, object] = {
        "trip_rates_period_value": period_value,
        "maximum_user_share_deviation": share_tolerance,
        "maximum_trip_rate_deviation": rate_tolerance,
        "frequency_source": incidence.get("frequency_source"),
    }
    if incidence.get("frequency_source") not in ("interview_band", "published_shares"):
        failures.append(f"{stage}: the incidence receipt names no frequency source.")
    all_ages, _older, _receipt = published_band_shares(spec_stage)
    published_user = 1.0 - float(all_ages[BAND_IDS[NON_USER_BAND]])
    achieved_user = incidence.get("person_user_share")
    if not isinstance(achieved_user, int | float):
        failures.append(f"{stage}: the incidence receipt carries no person_user_share.")
    else:
        details["user_share"] = {
            "published": published_user,
            "achieved": float(achieved_user),
        }
        if abs(float(achieved_user) - published_user) > share_tolerance:
            failures.append(
                f"{stage}: local-bus user share {float(achieved_user):.4f} is not the "
                f"published {published_user:.4f} (tolerance {share_tolerance} points)."
            )
    published_rates = published_trip_rates(declared_band)
    frame_rates = _mapping(
        trip_rates.get("frame_trips_per_person"),
        label=f"{stage}.frame_trips_per_person",
    )
    recorded = _mapping(
        trip_rates.get("published_trips_per_person"),
        label=f"{stage}.published_trips_per_person",
    )
    details["trip_rates"] = {}
    for series in SERIES:
        published = float(published_rates[series])
        if not isinstance(recorded.get(series), int | float) or abs(
            float(recorded[series]) - published
        ) > 1e-9 * max(1.0, published):
            failures.append(
                f"{stage}: the receipt's published {series} rate "
                f"{recorded.get(series)!r} is not the vendored {published}."
            )
        achieved = frame_rates.get(series)
        if not isinstance(achieved, int | float):
            failures.append(f"{stage}: the receipt carries no frame {series} rate.")
            continue
        deviation = float(achieved) / published - 1.0 if published > 0 else None
        details["trip_rates"][series] = {
            "published": published,
            "achieved": float(achieved),
            "relative_deviation": deviation,
        }
        if deviation is None or abs(deviation) > rate_tolerance:
            failures.append(
                f"{stage}: frame {series} trips per person {float(achieved):.3f} "
                f"deviate {deviation!r} from the published {published:.3f}, above "
                f"the tolerance {rate_tolerance}."
            )
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _bus_pricing_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The lcfs bus-fare pricing used the published yields and translations.

    Fact checks on the lcfs ``bus_pricing`` receipt, every published value
    recomputed here from the vendored rows the stage declares (never taken
    from the receipt): for every declared area the receipts, boardings,
    concessionary journeys, trips per person and population, and the
    boardings-per-trip and yield factors derived from them, must equal the
    receipt's to ``maximum_relative_deviation``; the unpriced regions must be
    the declared ones; and the receipt must state that the chain conditioned
    on the raw draw. The frame-implied boardings against the published ones
    are reported, not fenced: that ratio is the survey-versus-admin reading
    the calibration targets then act on (microcosm#930).
    """

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.bus_fare_pricing import bus_fare_prices
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_VENDORED_RESOURCES,
        bus_pricing_operation,
    )

    check = "bus_pricing"
    receipt = _mapping(evidence.get("bus_pricing"), label=f"{stage}.bus_pricing")
    tolerance = _finite_number(
        parameters.get("maximum_relative_deviation"),
        label=f"{stage}.maximum_relative_deviation",
    )
    declared = bus_pricing_operation(load_country_spec("uk").sources.stage_map()[stage])
    if declared is None:
        raise ValueError(f"{stage}: declares no price_bus_journeys operation.")
    prices = bus_fare_prices(declared, allowed_resources=UK_LCFS_VENDORED_RESOURCES)
    failures: list[str] = []
    details: dict[str, object] = {
        "maximum_relative_deviation": tolerance,
        "areas_fact_checked": 0,
        "frame_implied_over_published_boardings": {},
    }
    if receipt.get("chain_conditioned_on") != "raw_draw":
        failures.append(f"{stage}: the chain did not condition on the raw draw.")
    if sorted(str(r) for r in receipt.get("unpriced_regions", ())) != sorted(
        prices.unpriced_regions
    ):
        failures.append(
            f"{stage}: unpriced regions {receipt.get('unpriced_regions')!r} are not "
            f"the declared {sorted(prices.unpriced_regions)}."
        )
    recorded = _mapping(receipt.get("prices"), label=f"{stage}.bus_pricing.prices")
    expected = {
        "london_series": prices.london,
        **{a.label: a for a in prices.other_by_region.values()},
    }

    def _close(observed: object, value: float) -> bool:
        return isinstance(observed, int | float) and abs(float(observed) - value) <= (
            tolerance * max(1.0, abs(value))
        )

    for label, area in expected.items():
        block = recorded.get(label)
        if not isinstance(block, Mapping):
            failures.append(f"{stage}: the receipt prices no area {label!r}.")
            continue
        for key, value in (
            ("boardings_per_trip", area.boardings_per_trip),
            ("yield_per_fare_paying_boarding", area.yield_per_fare_paying_boarding),
            ("fare_per_trip", area.fare_per_trip),
            ("concessionary_boarding_share", area.concessionary_boarding_share),
        ):
            if not _close(block.get(key), value):
                failures.append(
                    f"{stage}: {label} {key} {block.get(key)!r} is not the vendored "
                    f"{value}."
                )
        for key, value in (
            ("receipts", area.receipts),
            ("boardings", area.boardings),
            ("concessionary", area.concessionary_boardings),
            ("trips_per_person", area.trips_per_person),
        ):
            inner = block.get(key)
            observed = inner.get("value") if isinstance(inner, Mapping) else None
            if not _close(observed, value):
                failures.append(
                    f"{stage}: {label} {key} {observed!r} is not the vendored {value}."
                )
        population = block.get("population")
        observed = population.get("value") if isinstance(population, Mapping) else None
        if not _close(observed, area.population):
            failures.append(
                f"{stage}: {label} population {observed!r} is not the vendored "
                f"{area.population}."
            )
        details["areas_fact_checked"] = int(details["areas_fact_checked"]) + 1
    by_area = receipt.get("by_area")
    if isinstance(by_area, Mapping):
        for label, entry in by_area.items():
            if isinstance(entry, Mapping):
                details["frame_implied_over_published_boardings"][str(label)] = (
                    entry.get("frame_implied_over_published_boardings")
                )
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _bus_support_pricing_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The ETB bus-support pricing used the published components per boarding.

    Fact checks on the ``bus_support_pricing`` receipt (microcosm#930): every
    declared support area's reimbursement, net support, boardings and
    concessionary boardings, and the per-boarding rates derived from them,
    are recomputed here from the vendored rows through the stage's own
    declaration (never taken from the receipt) and must equal the receipt's
    to ``maximum_relative_deviation``; the raw-draw regions must be the
    declared ones and the receipt must state that the pricing was applied on
    the chain's raw draw. The priced support against the published net
    support is reported, not fenced: the calibration targets act on it.
    """

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.bus_support_pricing import (
        bus_support_factors,
        bus_support_pricing_operation,
    )
    from microcosm.build.uk_runtime.etb_services import (
        UK_ETB_SERVICES_VENDORED_RESOURCES,
    )

    check = "bus_support_pricing"
    receipt = _mapping(
        evidence.get("bus_support_pricing"), label=f"{stage}.bus_support_pricing"
    )
    tolerance = _finite_number(
        parameters.get("maximum_relative_deviation"),
        label=f"{stage}.maximum_relative_deviation",
    )
    declared = bus_support_pricing_operation(
        load_country_spec("uk").sources.stage_map()[stage]
    )
    if declared is None:
        raise ValueError(f"{stage}: declares no price_bus_support operation.")
    factors = bus_support_factors(
        declared, allowed_resources=UK_ETB_SERVICES_VENDORED_RESOURCES
    )
    failures: list[str] = []
    details: dict[str, object] = {
        "maximum_relative_deviation": tolerance,
        "areas_fact_checked": 0,
        "priced_over_published": {},
    }
    if receipt.get("applied") is not True or receipt.get("chain_conditioned_on") != (
        "raw_draw"
    ):
        failures.append(
            f"{stage}: the receipt does not state an applied pricing on the raw draw."
        )
    declared_raw = sorted(str(r) for r in declared.get("raw_draw_regions", ()))
    if sorted(str(r) for r in receipt.get("raw_draw_regions", ())) != declared_raw:
        failures.append(
            f"{stage}: raw-draw regions {receipt.get('raw_draw_regions')!r} are not "
            f"the declared {declared_raw}."
        )
    recorded = _mapping(
        receipt.get("by_area"), label=f"{stage}.bus_support_pricing.by_area"
    )

    def _close(observed: object, value: float) -> bool:
        return isinstance(observed, int | float) and abs(float(observed) - value) <= (
            tolerance * max(1.0, abs(value))
        )

    for label, block in factors.items():
        entry = recorded.get(label)
        if not isinstance(entry, Mapping):
            failures.append(f"{stage}: the receipt prices no support area {label!r}.")
            continue
        for key in (
            "reimbursement_per_concessionary_boarding",
            "other_support_per_boarding",
            "published_concessionary_boarding_share",
        ):
            if not _close(entry.get(key), float(block[key])):
                failures.append(
                    f"{stage}: {label} {key} {entry.get(key)!r} is not the vendored "
                    f"{block[key]}."
                )
        for key in ("reimbursement", "net_support", "boardings", "concessionary"):
            inner = entry.get(key)
            observed = inner.get("value") if isinstance(inner, Mapping) else None
            if not _close(observed, float(block[key]["value"])):
                failures.append(
                    f"{stage}: {label} {key} {observed!r} is not the vendored "
                    f"{block[key]['value']}."
                )
        ratio = entry.get("priced_over_published")
        details["priced_over_published"][label] = (
            float(ratio) if isinstance(ratio, int | float) else None
        )
        details["areas_fact_checked"] = int(details["areas_fact_checked"]) + 1
    missing = sorted(set(recorded) - set(factors))
    if missing:
        failures.append(f"{stage}: the receipt prices undeclared areas {missing}.")
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )
