"""Stage-time health gates for the UK FRS spine build."""

from __future__ import annotations

import math
from collections.abc import Mapping

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
    if check == "pension_credit_take_up":
        return _pension_credit_take_up_gate(stage, evidence, parameters)
    if check == "child_benefit_take_up":
        return _child_benefit_take_up_gate(stage, evidence, parameters)
    if check == "source_signal":
        return _source_signal_gate(stage, evidence, parameters)
    if check == "age_tail_targets":
        return _age_tail_targets_gate(stage, evidence, parameters)
    if check == "cgt_residential_split":
        return _cgt_residential_split_gate(stage, evidence, parameters)
    if check == "cgt_support_split":
        return _cgt_support_split_gate(stage, evidence, parameters)
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
    if check == "wealth_coherence":
        return _wealth_coherence_gate(stage, evidence, parameters)
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
    # The donor floor (microcosm#1063 c9): negative diary consumption is raised
    # to the declared floor before the clip ranges are read, so no clip range
    # may start below it and no negative donor row may remain.
    declared_floor = parameters.get("donor_floor")
    floor = evidence.get("donor_floor")
    if declared_floor is None:
        pass
    elif floor is None:
        failures.append(f"{stage}: missing the donor_floor receipt.")
    else:
        floor = _mapping(floor, label=f"{stage}.donor_floor")
        level = _finite_number(floor.get("floor"), label=f"{stage}.donor_floor.floor")
        if level != float(declared_floor):
            failures.append(
                f"{stage}: the donor floor {level} is not the declared {declared_floor}."
            )
        remaining = int(floor.get("remaining_negative_rows", -1))
        if remaining != 0:
            failures.append(
                f"{stage}: the donor floor left {remaining} negative consumption row(s)."
            )
        for column, receipt in columns.items():
            if isinstance(receipt, Mapping) and "donor_min" in receipt:
                lower = _finite_number(
                    receipt["donor_min"], label=f"{column}.donor_min"
                )
                if column in _mapping(floor.get("columns", {}), label="floor") and (
                    lower < level
                ):
                    failures.append(
                        f"{stage}: {column!r} clip range starts at {lower}, below "
                        f"the donor floor {level}."
                    )
        details["donor_floor_rows_raised"] = int(floor.get("rows_raised", 0))
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
    """The energy kWh rake fits the NEED shape at the DESNZ level, at prior weights.

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
    IPF's cross-margin residual. That residual must be converged, not
    truncated: across each fuel's last ``convergence_window_sweeps`` sweeps,
    its ``sweep_residuals`` may range (maximum minus minimum) by at most
    ``maximum_residual_change_over_window``, so a rake stopped while its
    residual was still falling fails even when the truncated value sits inside
    the tolerance, and so does one oscillating inside the window. The window
    presupposes the rake runs more sweeps than the window is long; a shorter
    series fails closed. The rake must have run in kWh with gas over gas-connected rows
    and no zero-current cell; a missing margin, block, tolerance or sweep
    series fails closed.

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
    window = parameters.get("convergence_window_sweeps")
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError(
            f"{stage}: energy_rake declares no positive convergence_window_sweeps."
        )
    flatness = _finite_number(
        parameters.get("maximum_residual_change_over_window"),
        label=f"{stage}.maximum_residual_change_over_window",
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
        "convergence_window_sweeps": window,
        "maximum_residual_change_over_window": flatness,
        "residual_change_over_window": {},
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
    sweeps = receipt.get("sweep_residuals")
    if not isinstance(sweeps, Mapping):
        failures.append(f"{stage}: energy_rake receipt carries no sweep_residuals.")
        sweeps = {}
    for fuel in (ELECTRICITY_KWH, GAS_KWH):
        series = sweeps.get(fuel)
        if (
            not isinstance(series, list | tuple)
            or len(series) <= window
            or not all(
                isinstance(v, int | float) and math.isfinite(float(v)) for v in series
            )
        ):
            failures.append(
                f"{stage}: {fuel} sweep_residuals do not cover the "
                f"{window}-sweep convergence window."
            )
            continue
        # The window's range, not its endpoints: a rake oscillating with a
        # period that divides the window would score zero on its endpoints.
        tail = [float(v) for v in series[-1 - window :]]
        change = max(tail) - min(tail)
        details["residual_change_over_window"][fuel] = change
        if change > flatness:
            failures.append(
                f"{stage}: {fuel} residual ranged over {change:.4f} across the last "
                f"{window} of {len(series)} sweeps, above {flatness}: the rake was "
                "truncated, not converged."
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


def _nonnegative_count(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}.")
    return value


def _student_loan_plans_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """Each plan's top-up walked to its SLC stock, or the reason it could not.

    The stage realises a plan's shortfall by a greedy identity-keyed walk
    (microcosm#1049), so the check is on what the walk controls: the realised
    top-up sits within the lightest skipped person's weight of the shortfall
    and the final England count within ``maximum_stock_relative_deviation``
    of the stock. A pool lighter than the shortfall must have been taken
    whole; its attainment is recorded, not refused, because the gap between
    the SLC liable stock and what the FRS carries is a data question the
    stage cannot close. A plan already at or above its stock must have been
    left as reported. Every receipt field is required and the receipt must be
    self-consistent, so a tampered or partial receipt fails closed.
    """

    check = "student_loan_plans"
    plans = _mapping(evidence.get("plans"), label=f"{stage}.plans")
    declared_stocks = _mapping(parameters["stocks"], label=f"{stage}.stocks")
    tolerance = _finite_number(
        parameters["maximum_stock_relative_deviation"],
        label=f"{stage}.maximum_stock_relative_deviation",
    )
    if tolerance < 0.0:
        raise ValueError(f"{stage}: maximum_stock_relative_deviation must be >= 0.")
    failures: list[str] = []
    by_plan: dict[str, dict[str, object]] = {}
    worst_gap = 0.0
    worst_deviation = 0.0
    for plan, declared_stock in declared_stocks.items():
        receipt = plans.get(str(plan))
        if not isinstance(receipt, Mapping):
            failures.append(f"{stage}: missing receipt for {plan}.")
            continue
        label = f"{stage}.{plan}"
        stock = _finite_number(receipt.get("stock"), label=f"{label}.stock")
        expected = _finite_number(declared_stock, label=f"{label}.declared_stock")
        if stock != expected:
            failures.append(f"{stage}: {plan} stock {stock} != declared {expected}.")
        shortfall = _finite_number(receipt.get("shortfall"), label=f"{label}.shortfall")
        eligible_mass = _finite_number(
            receipt.get("eligible_mass"), label=f"{label}.eligible_mass"
        )
        topped_up_mass = _finite_number(
            receipt.get("topped_up_mass"), label=f"{label}.topped_up_mass"
        )
        gap = _finite_number(
            receipt.get("realization_gap"), label=f"{label}.realization_gap"
        )
        reported = _finite_number(
            receipt.get("reported_england_count"),
            label=f"{label}.reported_england_count",
        )
        final = _finite_number(
            receipt.get("final_england_count"), label=f"{label}.final_england_count"
        )
        eligible_rows = _nonnegative_count(
            receipt.get("eligible_rows"), label=f"{label}.eligible_rows"
        )
        topped_up_rows = _nonnegative_count(
            receipt.get("topped_up_rows"), label=f"{label}.topped_up_rows"
        )
        skipped_rows = _nonnegative_count(
            receipt.get("rows_skipped_for_weight"),
            label=f"{label}.rows_skipped_for_weight",
        )
        exhausted = receipt.get("pool_exhausted")
        if not isinstance(exhausted, bool):
            raise ValueError(
                f"{label}.pool_exhausted must be a bool, got {exhausted!r}."
            )
        lightest = receipt.get("lightest_skipped_weight")
        if lightest is not None:
            lightest = _finite_number(
                lightest, label=f"{label}.lightest_skipped_weight"
            )
        if not math.isclose(
            gap, topped_up_mass - shortfall, rel_tol=1e-9, abs_tol=1e-6
        ):
            failures.append(
                f"{stage}: {plan} realization_gap {gap} is not topped_up_mass minus "
                f"shortfall ({topped_up_mass - shortfall})."
            )
        if not math.isclose(
            final, reported + topped_up_mass, rel_tol=1e-9, abs_tol=1e-6
        ):
            failures.append(
                f"{stage}: {plan} final_england_count {final} is not the reported "
                f"count plus the top-up ({reported + topped_up_mass})."
            )
        if final < 0.0:
            failures.append(f"{stage}: {plan} final_england_count is negative.")
        deviation = abs(final - stock) / stock if stock > 0.0 else abs(final - stock)
        if shortfall <= 0.0:
            regime = "reported_at_or_above_stock"
            if topped_up_rows != 0 or topped_up_mass != 0.0:
                failures.append(
                    f"{stage}: {plan} was topped up ({topped_up_rows} rows) with no "
                    "shortfall."
                )
        elif exhausted:
            regime = "pool_exhausted"
            if (
                skipped_rows != 0
                or topped_up_rows != eligible_rows
                or not math.isclose(
                    topped_up_mass, eligible_mass, rel_tol=1e-9, abs_tol=1e-6
                )
            ):
                failures.append(
                    f"{stage}: {plan} pool is receipted as exhausted but was not taken "
                    f"whole ({topped_up_rows} of {eligible_rows} rows, mass "
                    f"{topped_up_mass} of {eligible_mass})."
                )
        else:
            regime = "walked_to_stock"
            # With nobody skipped the walk can only have met the shortfall
            # exactly; otherwise the gap is bounded by the lightest skip.
            if skipped_rows == 0 or lightest is None:
                if skipped_rows != 0 or lightest is not None or gap != 0.0:
                    failures.append(
                        f"{stage}: {plan} walk reports a fit to the shortfall "
                        "without a skipped person to bound it."
                    )
            elif abs(gap) >= lightest:
                failures.append(
                    f"{stage}: {plan} realization_gap {gap} is not within the lightest "
                    f"skipped weight {lightest} of the shortfall."
                )
            if deviation > tolerance:
                failures.append(
                    f"{stage}: {plan} final England count {final} deviates "
                    f"{deviation:.4f} from the stock {stock}, above "
                    f"{tolerance}."
                )
        worst_gap = max(worst_gap, abs(gap))
        if regime == "walked_to_stock":
            worst_deviation = max(worst_deviation, deviation)
        by_plan[str(plan)] = {
            "regime": regime,
            "stock_attainment": final / stock if stock > 0.0 else None,
            "realization_gap": gap,
            "rows_skipped_for_weight": skipped_rows,
            "lightest_skipped_weight": lightest,
        }
    details = {
        "plans_checked": len(declared_stocks),
        "worst_abs_realization_gap": worst_gap,
        "worst_walked_stock_deviation": worst_deviation,
        "plans": by_plan,
    }
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
    tolerance = _finite_number(
        parameters.get("absolute_tolerance", 0.0), label=f"{stage}.absolute_tolerance"
    )
    if abs(share - expected_share) > tolerance:
        failures.append(
            f"{stage}: spi_prior_mass_share {share} != declared {expected_share}."
        )
    details = {
        "spi_prior_mass_share": share,
        "spi_households": evidence.get("spi_households"),
    }
    if "pension_age_spi_prior_mass_share" in parameters:
        # microcosm#1069 c6: the channel takes its own share of the strata of
        # households with a member at or over State Pension age.
        expected_pension_share = _finite_number(
            parameters["pension_age_spi_prior_mass_share"],
            label=f"{stage}.pension_age_spi_prior_mass_share",
        )
        pension_share = evidence.get("pension_age_spi_prior_mass_share")
        details["pension_age_spi_prior_mass_share"] = pension_share
        if pension_share is None:
            failures.append(
                f"{stage}: the receipt carries no pension_age_spi_prior_mass_share."
            )
        elif (
            abs(
                _finite_number(
                    pension_share, label=f"{stage}.pension_age_spi_prior_mass_share"
                )
                - expected_pension_share
            )
            > tolerance
        ):
            failures.append(
                f"{stage}: pension_age_spi_prior_mass_share {pension_share} != "
                f"declared {expected_pension_share}."
            )
    if evidence.get("household_weight_kind") != parameters.get("household_weight_kind"):
        failures.append(f"{stage}: household_weight_kind drifted.")
    if int(evidence.get("spi_households", 0)) < int(
        parameters["minimum_spi_households"]
    ):
        failures.append(f"{stage}: spi_households below declared minimum.")
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _child_benefit_take_up_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """Claims meet HMRC's rates by child age and opt-outs its family share.

    The claimed share of eligible children is held overall against the
    published rates at the frame's own age mix, and at each single year of age
    with enough child rows to measure it; an age whose solved rate was clipped
    (reporters alone exceed the published rate, or every family claiming
    falls short of it) is reported, not held. The opted-out share of claiming
    families is held to the published share, or may fall short of it only
    where the receipt shows the charged families exhausted (microcosm#1063).
    """

    check = "child_benefit_take_up"
    overall_tolerance = _finite_number(
        parameters["maximum_claim_rate_deviation"],
        label=f"{stage}.maximum_claim_rate_deviation",
    )
    age_tolerance = _finite_number(
        parameters["maximum_age_claim_rate_deviation"],
        label=f"{stage}.maximum_age_claim_rate_deviation",
    )
    minimum_age_rows = int(parameters["minimum_age_child_rows"])
    opt_out_tolerance = _finite_number(
        parameters["maximum_opt_out_share_deviation"],
        label=f"{stage}.maximum_opt_out_share_deviation",
    )
    minimum_families = int(parameters["minimum_eligible_family_units"])
    failures: list[str] = []
    claims = _mapping(evidence.get("claims"), label=f"{stage}.claims")
    opt_outs = _mapping(evidence.get("opt_outs"), label=f"{stage}.opt_outs")
    families = int(claims.get("eligible_family_units", 0))
    details: dict[str, object] = {"eligible_family_units": families}
    target = claims.get("target_rate")
    realized = claims.get("realized_rate")
    if families < minimum_families or target is None or realized is None:
        failures.append(f"{stage}: the receipt has {families} eligible families.")
    else:
        target = _finite_number(target, label=f"{stage}.claims.target_rate")
        realized = _finite_number(realized, label=f"{stage}.claims.realized_rate")
        details["claim_rate"] = {"target": target, "realized": realized}
        if abs(realized - target) > overall_tolerance:
            failures.append(
                f"{stage}: {realized:.4f} of eligible children are claimed for "
                f"against {target:.4f} at the published rates by age (tolerance "
                f"{overall_tolerance})."
            )
    ages = claims.get("ages")
    if not isinstance(ages, list) or not ages:
        failures.append(f"{stage}: the receipt carries no claim rates by age.")
        ages = []
    worst: tuple[float, int] | None = None
    clipped: list[int] = []
    measured = 0
    for row in ages:
        row = _mapping(row, label=f"{stage}.claims.ages[]")
        age = int(row.get("age", -1))
        if row.get("clipped") is True:
            clipped.append(age)
            continue
        if int(row.get("eligible_child_rows", 0)) < minimum_age_rows:
            continue
        published = _finite_number(
            row.get("published_rate"), label=f"{stage}.age {age}.published_rate"
        )
        achieved = _finite_number(
            row.get("realized_rate"), label=f"{stage}.age {age}.realized_rate"
        )
        measured += 1
        deviation = abs(achieved - published)
        if worst is None or deviation > worst[0]:
            worst = (deviation, age)
        if deviation > age_tolerance:
            failures.append(
                f"{stage}: at age {age} {achieved:.4f} of eligible children are "
                f"claimed for against the published {published:.4f} (tolerance "
                f"{age_tolerance})."
            )
    details["ages_measured"] = measured
    details["clipped_ages"] = clipped
    if worst is not None:
        details["largest_age_deviation"] = {"age": worst[1], "deviation": worst[0]}

    share = opt_outs.get("realized_share")
    target_share = _finite_number(
        opt_outs.get("target_share"), label=f"{stage}.opt_outs.target_share"
    )
    exhausted = opt_outs.get("pool_exhausted") is True
    if share is None:
        failures.append(f"{stage}: the receipt carries no opted-out share.")
    else:
        share = _finite_number(share, label=f"{stage}.opt_outs.realized_share")
        details["opt_out_share"] = {
            "target": target_share,
            "realized": share,
            "pool_exhausted": exhausted,
        }
        if share > target_share + opt_out_tolerance or (
            not exhausted and share < target_share - opt_out_tolerance
        ):
            failures.append(
                f"{stage}: {share:.4f} of claiming families opted out against "
                f"the published {target_share:.4f} (tolerance {opt_out_tolerance}, "
                f"charged families exhausted: {exhausted})."
            )
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _pension_credit_take_up_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """Each component band realizes its DWP take-up rate (microcosm#1069 R9).

    A band whose reporters alone exceed the rate realizes their share instead;
    the receipt flags it and the gate requires the realized share to be at
    least the rate there.
    """

    check = "pension_credit_take_up"
    tolerance = _finite_number(
        parameters["maximum_take_up_deviation"],
        label=f"{stage}.maximum_take_up_deviation",
    )
    minimum_units = int(parameters["minimum_entitled_units"])
    bands = evidence.get("bands")
    failures: list[str] = []
    details: dict[str, object] = {}
    if not isinstance(bands, list) or not bands:
        failures.append(f"{stage}: the receipt carries no take-up bands.")
        bands = []
    for band in bands:
        band = _mapping(band, label=f"{stage}.bands[]")
        name = str(band.get("band"))
        rate = _finite_number(band.get("rate"), label=f"{stage}.{name}.rate")
        units = int(band.get("entitled_units", 0))
        realized = band.get("realized_take_up")
        details[name] = {"rate": rate, "realized_take_up": realized, "units": units}
        if units < minimum_units or realized is None:
            failures.append(f"{stage}: band {name} has {units} entitled units.")
            continue
        realized = _finite_number(realized, label=f"{stage}.{name}.realized_take_up")
        if band.get("reporters_exceed_rate") is True:
            if realized + 1e-12 < rate:
                failures.append(
                    f"{stage}: band {name} reporters exceed the rate but realize "
                    f"{realized:.4f} < {rate:.4f}."
                )
        elif abs(realized - rate) > tolerance:
            failures.append(
                f"{stage}: band {name} realizes take-up {realized:.4f} against the "
                f"rate {rate:.4f} (tolerance {tolerance})."
            )
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


def _cgt_support_split_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The support split conserved mass, capped every copy and met its rule.

    Exhaustion is a recorded regime, not a failure (microcosm#1045): a column
    whose pool is lighter than its support mass selects the whole pool and
    says so. The split assigns no value, so nothing distributional is held.
    """

    check = "cgt_support_split"
    clone_split_factor = _finite_number(
        parameters["clone_split_factor"], label=f"{stage}.clone_split_factor"
    )
    headroom = _finite_number(parameters["headroom"], label=f"{stage}.headroom")
    maximum_copy_weight = _finite_number(
        parameters["maximum_copy_weight"], label=f"{stage}.maximum_copy_weight"
    )
    tolerance = _finite_number(
        parameters["maximum_relative_mass_deviation"],
        label=f"{stage}.maximum_relative_mass_deviation",
    )
    effective_tolerance = max(tolerance, _FLOAT_RELATIVE_TOLERANCE)
    failures: list[str] = []

    declared = _mapping(evidence.get("parameters"), label=f"{stage}.parameters")
    for key, expected in (
        ("clone_split_factor", clone_split_factor),
        ("headroom", headroom),
        ("maximum_copy_weight", maximum_copy_weight),
    ):
        value = _finite_number(declared.get(key), label=f"{stage}.parameters.{key}")
        if value != expected:
            failures.append(
                f"{stage}: receipt {key} {value} differs from the gate's {expected}."
            )

    mass = _mapping(evidence.get("mass"), label=f"{stage}.mass")
    old_total = _finite_number(mass.get("old_total"), label=f"{stage}.mass.old_total")
    new_total = _finite_number(mass.get("new_total"), label=f"{stage}.mass.new_total")
    if old_total <= 0.0 or new_total <= 0.0:
        failures.append(f"{stage}: household mass must be positive before and after.")
    deviation = abs(new_total - old_total) / max(abs(old_total), 1.0)
    if deviation > effective_tolerance:
        failures.append(
            f"{stage}: household mass deviation {deviation} exceeds the effective "
            f"tolerance {effective_tolerance}."
        )

    rows = evidence.get("bands")
    if not isinstance(rows, list | tuple):
        raise ValueError(f"{stage}.bands must be a list.")
    exhausted: list[float] = []
    sums = {
        "support_mass": 0.0,
        "selected_mass": 0.0,
        "households_selected": 0,
        "copies_created": 0,
    }
    heaviest_copy = 0.0
    for row in rows:
        if not isinstance(row, Mapping):
            failures.append(f"{stage}: band row is not an object.")
            continue
        lower = _finite_number(
            row.get("income_lower_bound"), label=f"{stage}.income_lower_bound"
        )
        published = _finite_number(
            row.get("published_top_band_taxpayers"),
            label=f"{stage}.published_top_band_taxpayers",
        )
        support = _finite_number(row.get("support_mass"), label=f"{stage}.support_mass")
        selected = _finite_number(
            row.get("selected_mass"), label=f"{stage}.selected_mass"
        )
        pool_mass = _finite_number(row.get("pool_mass"), label=f"{stage}.pool_mass")
        counts = {}
        for key in ("pool_households", "households_selected", "copies_created"):
            value = row.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{stage}.{key} must be a non-negative integer.")
            counts[key] = value
        heaviest_selected = _finite_number(
            row.get("heaviest_selected_weight"),
            label=f"{stage}.heaviest_selected_weight",
        )
        row_heaviest_copy = _finite_number(
            row.get("heaviest_copy_weight"), label=f"{stage}.heaviest_copy_weight"
        )
        pool_exhausted = row.get("pool_exhausted")
        if not isinstance(pool_exhausted, bool):
            raise ValueError(f"{stage}.pool_exhausted must be a boolean.")
        expected_support = clone_split_factor * headroom * published
        if not np.isclose(support, expected_support, rtol=1e-12, atol=1e-9):
            failures.append(
                f"{stage}: column from {lower} support mass {support} differs from "
                f"{clone_split_factor} x {headroom} x {published}."
            )
        if row_heaviest_copy > maximum_copy_weight * (1.0 + 1e-12):
            failures.append(
                f"{stage}: column from {lower} copy weight {row_heaviest_copy} "
                f"exceeds the maximum {maximum_copy_weight}."
            )
        if pool_exhausted:
            exhausted.append(lower)
            if counts["households_selected"] != counts["pool_households"] or not (
                np.isclose(selected, pool_mass, rtol=1e-12, atol=1e-9)
            ):
                failures.append(
                    f"{stage}: column from {lower} is recorded as exhausted but did "
                    "not select its whole pool."
                )
        elif selected + 1e-9 < support:
            failures.append(
                f"{stage}: column from {lower} selected mass {selected} falls short "
                f"of its support mass {support} without recording exhaustion."
            )
        if heaviest_selected <= maximum_copy_weight and counts["copies_created"]:
            failures.append(
                f"{stage}: column from {lower} created {counts['copies_created']} "
                "copies although no selected household exceeds the maximum weight."
            )
        sums["support_mass"] += support
        sums["selected_mass"] += selected
        sums["households_selected"] += counts["households_selected"]
        sums["copies_created"] += counts["copies_created"]
        heaviest_copy = max(heaviest_copy, row_heaviest_copy)

    totals = _mapping(evidence.get("totals"), label=f"{stage}.totals")
    for key in ("households_selected", "copies_created"):
        value = totals.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value != sums[key]:
            failures.append(
                f"{stage}: totals.{key} {value!r} differs from the column sum "
                f"{sums[key]}."
            )
    for key in ("support_mass", "selected_mass"):
        value = _finite_number(totals.get(key), label=f"{stage}.totals.{key}")
        if not np.isclose(value, sums[key], rtol=1e-12, atol=1e-6):
            failures.append(
                f"{stage}: totals.{key} {value} differs from the column sum "
                f"{sums[key]}."
            )

    details = {
        "columns_checked": len(rows),
        "exhausted_columns": exhausted,
        "households_selected": sums["households_selected"],
        "copies_created": sums["copies_created"],
        "heaviest_copy_weight": heaviest_copy,
        "relative_mass_deviation": deviation,
        "effective_relative_tolerance": effective_tolerance,
    }
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
    """Every reserved band carries its donors, and their mass was reallocated.

    The donors are a mass-conserving support channel (microcosm#1063): total
    household mass is unchanged, no incumbent stratum gives up more than the
    declared share, and no donor starts above the declared weight. On a full
    frame (both scales one) each band seats its planned count, never fewer
    than the minimum, and carries its published taxpayers; a build declaring
    the full survey sample must be at full scale.
    """

    check = "spi_income_band_donor_support"
    minimum_donors = int(parameters["minimum_donors_per_band"])
    maximum_donor_weight = _finite_number(
        parameters["maximum_donor_weight"], label=f"{stage}.maximum_donor_weight"
    )
    minimum_funding_factor = _finite_number(
        parameters["minimum_funding_factor"], label=f"{stage}.minimum_funding_factor"
    )
    taxpayer_tolerance = _finite_number(
        parameters["maximum_band_taxpayer_deviation"],
        label=f"{stage}.maximum_band_taxpayer_deviation",
    )
    tolerance = _finite_number(
        parameters["maximum_relative_mass_deviation"],
        label=f"{stage}.maximum_relative_mass_deviation",
    )
    expected_bands = [int(value) for value in parameters["band_lower_bounds"]]
    failures: list[str] = []

    for key, expected in (
        ("minimum_donors_per_band", float(minimum_donors)),
        ("maximum_donor_weight", maximum_donor_weight),
    ):
        value = _finite_number(evidence.get(key), label=f"{stage}.{key}")
        if value != expected:
            failures.append(
                f"{stage}: receipt {key} {value} differs from the gate's {expected}."
            )
    scales: dict[str, float] = {}
    for key in ("seating_scale", "mass_scale"):
        scales[key] = _finite_number(evidence.get(key), label=f"{stage}.{key}")
        if not 0.0 < scales[key] <= 1.0:
            failures.append(f"{stage}: {key} {scales[key]} is outside (0, 1].")
    full_seating = scales["seating_scale"] == 1.0
    full_scale = full_seating and scales["mass_scale"] == 1.0
    sample_fraction = _finite_number(
        evidence.get("sample_fraction"), label=f"{stage}.sample_fraction"
    )
    if sample_fraction == 1.0 and not full_scale:
        failures.append(
            f"{stage}: a full-sample build must seat and fund the donors at full "
            f"scale, got seating_scale {scales['seating_scale']} and mass_scale "
            f"{scales['mass_scale']}."
        )

    mass = _mapping(evidence.get("mass"), label=f"{stage}.mass")
    old_total = _finite_number(mass.get("old_total"), label=f"{stage}.mass.old_total")
    new_total = _finite_number(mass.get("new_total"), label=f"{stage}.mass.new_total")
    if old_total <= 0.0 or new_total <= 0.0:
        failures.append(f"{stage}: household mass must be positive before and after.")
    deviation = abs(new_total - old_total) / max(abs(old_total), 1.0)
    effective_tolerance = max(tolerance, _FLOAT_RELATIVE_TOLERANCE)
    if deviation > effective_tolerance:
        failures.append(
            f"{stage}: household mass deviation {deviation} exceeds the effective "
            f"tolerance {effective_tolerance}; the donors must be funded from the "
            "incumbent households, not added."
        )

    bands = evidence.get("bands")
    if not isinstance(bands, list | tuple):
        raise ValueError(f"{stage}.bands must be a list.")
    seen: list[int] = []
    donor_total = 0
    donor_mass = 0.0
    heaviest_donor = 0.0
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
        expected_donors = int(
            _finite_number(
                row.get("expected_donor_households"),
                label=f"{stage}.expected_donor_households",
            )
        )
        carriers = int(_finite_number(row.get("carriers"), label=f"{stage}.carriers"))
        weight = _finite_number(row.get("donor_weight"), label=f"{stage}.donor_weight")
        published = _finite_number(
            row.get("published_taxpayers"), label=f"{stage}.published_taxpayers"
        )
        donor_total += donors
        donor_mass += weight * donors
        heaviest_donor = max(heaviest_donor, weight)
        if donors <= 0:
            failures.append(f"{stage}: band from {lower} seats no donor.")
        if donors != carriers:
            failures.append(
                f"{stage}: band from {lower} has {donors} donors but {carriers} carriers."
            )
        if weight <= 0.0:
            failures.append(
                f"{stage}: band from {lower} donor weight {weight} is not positive."
            )
        if weight > maximum_donor_weight * (1.0 + _FLOAT_RELATIVE_TOLERANCE):
            failures.append(
                f"{stage}: band from {lower} donor weight {weight} exceeds the "
                f"maximum {maximum_donor_weight}."
            )
        if full_seating and (donors != expected_donors or donors < minimum_donors):
            failures.append(
                f"{stage}: band from {lower} seats {donors} donors at full scale; "
                f"the plan seats {expected_donors} and the minimum is "
                f"{minimum_donors}."
            )
        # A scaled frame seats fewer donors or lighter ones, so its bands fall
        # short of the published mass by construction; the receipt shows it.
        if full_scale and abs(weight * donors - published) > taxpayer_tolerance:
            failures.append(
                f"{stage}: band from {lower} weighted taxpayers "
                f"{weight * donors} differ from the published {published}."
            )
    if sorted(seen) != sorted(expected_bands):
        failures.append(
            f"{stage}: bands {sorted(seen)} differ from the declared {sorted(expected_bands)}."
        )
    donor_count = int(
        _finite_number(evidence.get("donor_count"), label=f"{stage}.donor_count")
    )
    if donor_count != donor_total:
        failures.append(
            f"{stage}: receipt donor_count {donor_count} differs from the bands' "
            f"{donor_total}."
        )

    funding = evidence.get("funding")
    if not isinstance(funding, list | tuple) or not funding:
        raise ValueError(f"{stage}.funding must be a non-empty list.")
    funded_mass = 0.0
    smallest_factor = 1.0
    for row in funding:
        if not isinstance(row, Mapping):
            failures.append(f"{stage}: funding row is not an object.")
            continue
        factor = _finite_number(row.get("factor"), label=f"{stage}.funding.factor")
        funded_mass += _finite_number(
            row.get("donor_mass"), label=f"{stage}.funding.donor_mass"
        )
        smallest_factor = min(smallest_factor, factor)
        if not (
            minimum_funding_factor * (1.0 - _FLOAT_RELATIVE_TOLERANCE) <= factor <= 1.0
        ):
            failures.append(
                f"{stage}: funding stratum {row.get('stratum')!r} factor {factor} "
                f"is outside [{minimum_funding_factor}, 1]."
            )
    reallocated = _finite_number(
        evidence.get("reallocated_mass"), label=f"{stage}.reallocated_mass"
    )
    for label, value in (("bands", donor_mass), ("funding strata", funded_mass)):
        if abs(value - reallocated) > effective_tolerance * max(abs(reallocated), 1.0):
            failures.append(
                f"{stage}: the {label} carry donor mass {value}, the receipt "
                f"reallocated {reallocated}."
            )
    details = {
        "bands_checked": len(bands),
        "donor_count": donor_count,
        "seating_scale": scales["seating_scale"],
        "mass_scale": scales["mass_scale"],
        "reallocated_mass": reallocated,
        "relative_mass_deviation": deviation,
        "heaviest_donor_weight": heaviest_donor,
        "smallest_funding_factor": smallest_factor,
        "funding_strata": len(funding),
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


def _cgt_residential_split_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The residential split conserved mass and left Table 8a as identities.

    The split carries the solved residential probability as weight
    (microcosm#1063): every household with liable gainers becomes arms whose
    weights are products of ``p`` and ``1 - p``, so the residential count and
    gains at design weights are the solved expectations in every gain band.
    The gate holds the total household mass to the declared deviation, the
    solve to its targets, the realised masses to the expectations (an
    arithmetic identity), the arm weights to their products, and the number
    of liable gainers per household to the declared ceiling.
    """

    check = "cgt_residential_split"
    tolerance = _finite_number(
        parameters["maximum_relative_mass_deviation"],
        label=f"{stage}.maximum_relative_mass_deviation",
    )
    max_solve_error = _finite_number(
        parameters["maximum_solve_relative_error"],
        label=f"{stage}.maximum_solve_relative_error",
    )
    max_identity_error = _finite_number(
        parameters["maximum_identity_relative_error"],
        label=f"{stage}.maximum_identity_relative_error",
    )
    maximum_gainers = int(parameters["maximum_liable_gainers_per_household"])
    effective_tolerance = max(tolerance, _FLOAT_RELATIVE_TOLERANCE)
    failures: list[str] = []
    details: dict[str, object] = {}

    mass = _mapping(evidence.get("mass"), label=f"{stage}.mass")
    old_total = _finite_number(mass.get("old_total"), label=f"{stage}.mass.old_total")
    new_total = _finite_number(mass.get("new_total"), label=f"{stage}.mass.new_total")
    if old_total <= 0.0 or new_total <= 0.0:
        failures.append(f"{stage}: household mass must be positive before and after.")
    deviation = abs(new_total - old_total) / max(abs(old_total), 1.0)
    details["relative_mass_deviation"] = deviation
    if deviation > effective_tolerance:
        failures.append(
            f"{stage}: household mass deviation {deviation} exceeds the effective "
            f"tolerance {effective_tolerance}."
        )
    if evidence.get("arm_weights_exact") is not True:
        failures.append(f"{stage}: the arm weights are not the declared products.")
    by_k = _mapping(
        evidence.get("households_by_liable_gainers"),
        label=f"{stage}.households_by_liable_gainers",
    )
    arms_expected = 0
    for key, count in by_k.items():
        k = int(key)
        households = int(count)
        if k < 1 or households < 0:
            failures.append(
                f"{stage}: households_by_liable_gainers[{key!r}] is invalid."
            )
            continue
        if k > maximum_gainers:
            failures.append(
                f"{stage}: {households} household(s) carry {k} liable gainers, "
                f"above the declared maximum {maximum_gainers}."
            )
        arms_expected += households * (2**k - 1)
    arms_created = int(evidence.get("arms_created", -1))
    details["arms_created"] = arms_created
    if arms_created != arms_expected:
        failures.append(
            f"{stage}: {arms_created} arms created, the households by liable "
            f"gainers imply {arms_expected}."
        )
    identities = _mapping(evidence.get("identities"), label=f"{stage}.identities")

    def number(key: str) -> float:
        return _finite_number(identities.get(key), label=f"{stage}.identities.{key}")

    for measure in ("count", "gains"):
        target = number(f"{measure}_target_individuals_basis")
        if target <= 0.0:
            failures.append(f"{stage}: residential {measure} target is not positive.")
            continue
        solve_error = number(f"{measure}_solve_relative_error")
        identity_error = number(f"{measure}_identity_relative_error")
        details[f"residential_{measure}_solve_relative_error"] = solve_error
        details[f"residential_{measure}_identity_relative_error"] = identity_error
        if solve_error > max_solve_error:
            failures.append(
                f"{stage}: residential {measure} solve error {solve_error} "
                f"exceeds {max_solve_error}."
            )
        if identity_error > max_identity_error:
            failures.append(
                f"{stage}: residential {measure} at design weights departs from "
                f"its expectation by {identity_error}, above {max_identity_error}."
            )
    bands = evidence.get("bands")
    if not isinstance(bands, list) or not bands:
        failures.append(f"{stage}: the receipt carries no gain bands.")
        bands = []
    for position, band in enumerate(bands):
        label = f"{stage}.bands[{position}]"
        row = _mapping(band, label=label)
        for measure in ("count", "gains"):
            expected = _finite_number(
                row.get(f"expected_{measure}"), label=f"{label}.expected_{measure}"
            )
            achieved = _finite_number(
                row.get(f"achieved_{measure}"), label=f"{label}.achieved_{measure}"
            )
            if abs(achieved - expected) > max_identity_error * max(abs(expected), 1.0):
                failures.append(
                    f"{stage}: residential {measure} in the gain band from "
                    f"{row.get('gain_lower_bound')} is {achieved} against the "
                    f"expectation {expected}."
                )
    details["gain_bands"] = len(bands)
    arm_weights = _mapping(evidence.get("arm_weights"), label=f"{stage}.arm_weights")
    for key in ("residential_arms", "below_one_household", "minimum"):
        details[f"arm_{key}"] = arm_weights.get(key)
    concentration = _mapping(
        evidence.get("concentration"), label=f"{stage}.concentration"
    )
    for key in (
        "top_arms_share_of_achieved_gains",
        "largest_liable_stake_share_of_gains_target",
    ):
        details[key] = concentration.get(key)
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
    """The residential arms carry Table 8a and the BADR claims their bands.

    The residential flag is no longer drawn here: ``cgt_residential_split``
    carries it as weight (microcosm#1063), so this gate holds the residential
    count and gains the arms carry at design weights to the Table 8a targets
    on the individuals basis within the solve tolerance, and requires every
    gain band row of the receipt to be finite. Every liable gainer must carry
    an asset type and the composition receipt must be finite (microcosm#725).

    Each Table 4.1 BADR band is held to deterministic bounds only
    (microcosm#1014): the solve to the band's targets; the realised count to
    within the band pool's largest weight, the walk's own bound; below the
    lifetime limit the realised qualifying gains to that weight times
    (2 max gain - min gain) of the band pool, which bounds the gains the walk
    can drift along its ascending-gain order; in the open top band the gains
    to exactly the limit times the realised count. Every declared invariant
    must be zero and the restricted type fit must have converged.
    """

    check = "cgt_asset_type_summary"
    failures: list[str] = []
    residential = _mapping(evidence.get("residential"), label=f"{stage}.residential")
    max_solve_error = _finite_number(
        parameters["maximum_solve_relative_error"],
        label=f"{stage}.maximum_solve_relative_error",
    )
    details: dict[str, object] = {}

    def number(key: str) -> float:
        return _finite_number(residential.get(key), label=f"{stage}.residential.{key}")

    if residential.get("source_stage") != "cgt_residential_split":
        failures.append(
            f"{stage}: the residential receipt does not name cgt_residential_split "
            "as its source."
        )
    for measure in ("count", "gains"):
        target = number(f"{measure}_target_individuals_basis")
        achieved = number(f"achieved_{measure}")
        if target <= 0.0:
            failures.append(f"{stage}: residential {measure} target is not positive.")
            continue
        error = abs(achieved - target) / target
        details[f"residential_{measure}_relative_error"] = error
        if error > max_solve_error:
            failures.append(
                f"{stage}: residential {measure} on the arms is {achieved} against "
                f"the target {target} (relative error {error}, tolerance "
                f"{max_solve_error})."
            )
    bands = residential.get("bands")
    if not isinstance(bands, list) or not bands:
        failures.append(f"{stage}: residential receipt carries no gain bands.")
        bands = []
    band_totals = {"achieved_count": 0.0, "achieved_gains": 0.0}
    for position, band in enumerate(bands):
        label = f"{stage}.residential.bands[{position}]"
        row = _mapping(band, label=label)
        for measure in band_totals:
            band_totals[measure] += _finite_number(
                row.get(measure), label=f"{label}.{measure}"
            )
    for measure, total in band_totals.items():
        expected = number(measure)
        if bands and abs(total - expected) > max_solve_error * max(abs(expected), 1.0):
            failures.append(
                f"{stage}: the gain bands' {measure} sums to {total}, not the "
                f"residential receipt's {expected}."
            )
    details["residential_gain_bands"] = len(bands)
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
    if asset_type.get("share_fit_converged") is not True:
        failures.append(f"{stage}: the main asset-type fit did not converge.")
    failures.extend(_cgt_badr_failures(stage, evidence, max_solve_error, details))
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _cgt_badr_failures(
    stage: str,
    evidence: Mapping[str, object],
    max_solve_error: float,
    details: dict[str, object],
) -> list[str]:
    """The BADR half of the asset-type gate (microcosm#1014)."""

    failures: list[str] = []
    badr = _mapping(evidence.get("badr"), label=f"{stage}.badr")
    invariants = _mapping(badr.get("invariants"), label=f"{stage}.badr.invariants")
    for name, rows in invariants.items():
        if rows != 0:
            failures.append(f"{stage}: BADR invariant {name} broken on {rows} rows.")
    limit = _finite_number(badr.get("lifetime_limit"), label=f"{stage}.badr.limit")
    bands = badr.get("bands")
    if not isinstance(bands, list) or not bands:
        return [*failures, f"{stage}: BADR receipt carries no bands."]
    checked = 0
    for index, raw in enumerate(bands):
        row = _mapping(raw, label=f"{stage}.badr.bands[{index}]")
        if row.get("skipped") is True:
            continue
        checked += 1
        name = f"BADR band from {row.get('lower_bound')}"

        def number(key: str, *, _row=row, _index=index) -> float:
            return _finite_number(
                _row.get(key), label=f"{stage}.badr.bands[{_index}].{key}"
            )

        for measure in ("count", "gains"):
            target = number(f"{measure}_target")
            if target <= 0.0:
                failures.append(f"{stage}: {name} {measure} target is not positive.")
                continue
            solve_error = abs(number(f"expected_{measure}") - target) / target
            if solve_error > max_solve_error:
                failures.append(
                    f"{stage}: {name} {measure} solve error {solve_error} exceeds "
                    f"{max_solve_error}."
                )
        max_weight = number("max_pool_weight")
        count_gap = abs(number("achieved_count") - number("expected_count"))
        if count_gap > max_weight * (1.0 + 1e-9):
            failures.append(
                f"{stage}: {name} count gap {count_gap} exceeds the band pool's "
                f"largest weight {max_weight}."
            )
        achieved_gains = number("achieved_gains")
        if row.get("qualifying_amount") == "lifetime_limit":
            exact = limit * number("achieved_count")
            if abs(achieved_gains - exact) > 1e-9 * max(abs(exact), 1.0):
                failures.append(
                    f"{stage}: {name} gains {achieved_gains} are not the lifetime "
                    f"limit times the realised count ({exact})."
                )
        else:
            gains_gap = abs(achieved_gains - number("expected_gains"))
            gains_bound = max_weight * (
                2.0 * number("pool_max_gain") - number("pool_min_gain")
            )
            if gains_gap > gains_bound * (1.0 + 1e-9):
                failures.append(
                    f"{stage}: {name} gains gap {gains_gap} exceeds the walk's "
                    f"bound {gains_bound}."
                )
    totals = _mapping(badr.get("totals"), label=f"{stage}.badr.totals")
    details["badr_bands_checked"] = checked
    details["badr_achieved_count"] = totals.get("achieved_count")
    details["badr_achieved_gains"] = totals.get("achieved_gains")
    details["badr_relief_rate_tax"] = totals.get("relief_rate_tax")
    return failures


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
    # Both masses are sums of the same weights; a build whose removed group is
    # empty can still carry float-epsilon residue, so the comparison allows
    # the absolute rounding tolerance the other mass identities use.
    if not np.isclose(removed, transferred, rtol=1e-9, atol=1e-6):
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


#: Receipt counts the WAS wealth stage must report as zero (microcosm#1063):
#: the two tenure rules, and one count per accounting identity.
_WEALTH_TENURE_ZERO_KEYS = (
    "mortgage_debt_off_mortgaged_tenure_rows",
    "main_residence_value_off_owner_tenure_rows",
)
_WEALTH_IDENTITY_ZERO_KEYS = (
    "property_wealth_violation_rows",
    "corporate_wealth_violation_rows",
    "gross_financial_wealth_violation_rows",
    "net_financial_wealth_violation_rows",
)


def _wealth_coherence_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The WAS wealth columns are coherent with tenure and with each other.

    The stage draws the main residence and the mortgage debt inside their
    tenure stratum and derives every total from drawn components; this gate
    re-reads the receipt so the battery, not only the transform, holds that
    no household off a mortgaged tenure carries mortgage debt, none off an
    owner tenure carries a main-residence value, every total equals its
    components, and owners without a main-residence value are no more common
    than on the donor beyond the reviewed excess.
    """

    check = "wealth_coherence"
    failures: list[str] = []
    details: dict[str, object] = {}
    blocks: dict[str, Mapping[str, object]] = {}
    for block in ("tenure_coherence", "identities"):
        value = evidence.get(block)
        if not isinstance(value, Mapping):
            failures.append(f"{stage}: receipt is missing {block}.")
            continue
        blocks[block] = value
    for block, keys in (
        ("tenure_coherence", _WEALTH_TENURE_ZERO_KEYS),
        ("identities", _WEALTH_IDENTITY_ZERO_KEYS),
    ):
        receipt = blocks.get(block)
        if receipt is None:
            continue
        for key in keys:
            value = receipt.get(key)
            if isinstance(value, bool) or not isinstance(value, int | float):
                failures.append(f"{stage}: {block} receipt is missing {key}.")
                continue
            details[key] = int(value)
            if int(value) != 0:
                failures.append(f"{stage}: {key} is {int(value)}, expected 0.")
    tenure = blocks.get("tenure_coherence")
    if tenure is not None:
        maximum_excess = _finite_number(
            parameters["maximum_owner_share_without_main_residence_excess"],
            label=f"{stage}.maximum_owner_share_without_main_residence_excess",
        )
        try:
            share = _finite_number(
                tenure.get("owner_share_without_main_residence_value"),
                label=f"{stage}.owner_share_without_main_residence_value",
            )
            donor_share = _finite_number(
                tenure.get("donor_owner_share_without_main_residence_value"),
                label=f"{stage}.donor_owner_share_without_main_residence_value",
            )
        except ValueError as error:
            failures.append(str(error))
        else:
            details["owner_share_without_main_residence_value"] = share
            details["donor_owner_share_without_main_residence_value"] = donor_share
            details["maximum_owner_share_without_main_residence_excess"] = (
                maximum_excess
            )
            if share - donor_share > maximum_excess:
                failures.append(
                    f"{stage}: {share:.6g} of owner households carry no "
                    f"main-residence value against {donor_share:.6g} on the "
                    f"donor, above the reviewed excess {maximum_excess:.6g}."
                )
    identities = blocks.get("identities")
    if identities is not None:
        details["capped_rows"] = {
            str(key): int(value)
            for key, value in identities.items()
            if str(key).endswith("_rows")
            and "capped" in str(key)
            and isinstance(value, int | float)
        }
    if failures:
        return _fail(stage, check, failures, details)
    return _pass(stage, check, details)


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
    from the receipt): the frame's prior-weighted share of persons who use
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
