"""Stage-time health gates for the UK FRS spine build."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from microcosm.build.gates import GateResult
from microcosm.build.uk_runtime.receipt_counts import (
    UK_BELOW_TEN_BAND,
    UK_DISCLOSURE_FLOOR,
    UK_WITHHELD,
)

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
    if check == "frs_annual_income_spells":
        return _frs_annual_income_spells_gate(stage, evidence, parameters)
    if check == "pension_credit_take_up":
        return _pension_credit_take_up_gate(stage, evidence, parameters)
    if check == "uc_take_up":
        return _uc_take_up_gate(stage, evidence, parameters)
    if check == "child_benefit_take_up":
        return _child_benefit_take_up_gate(stage, evidence, parameters)
    if check == "take_up_signal":
        return _take_up_signal_gate(stage, evidence, parameters)
    if check == "spi_benefit_coherence":
        return _spi_benefit_coherence_gate(stage, evidence, parameters)
    if check == "source_signal":
        return _source_signal_gate(stage, evidence, parameters)
    if check == "age_tail_targets":
        return _age_tail_targets_gate(stage, evidence, parameters)
    if check == "cgt_residential_split":
        return _cgt_residential_split_gate(stage, evidence, parameters)
    if check == "cgt_support_split":
        return _cgt_support_split_gate(stage, evidence, parameters)
    if check == "high_value_home_support":
        return _high_value_home_support_gate(stage, evidence, parameters)
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
    if check == "consumption_basis":
        return _consumption_basis_gate(stage, evidence, parameters)
    if check == "recomposed_totals":
        return _recomposed_totals_gate(stage, evidence, parameters)
    if check == "road_fuel_incidence":
        return _road_fuel_incidence_gate(stage, evidence, parameters)
    if check == "road_fuel_level":
        return _road_fuel_level_gate(stage, evidence, parameters)
    if check == "excise_quantities":
        return _excise_quantities_gate(stage, evidence, parameters)
    if check == "gambling_participation":
        return _gambling_participation_gate(stage, evidence, parameters)
    if check == "bus_support_pricing":
        return _bus_support_pricing_gate(stage, evidence, parameters)
    if check == "property_components":
        return _property_components_gate(stage, evidence, parameters)
    if check == "property_purchases":
        return _property_purchases_gate(stage, evidence, parameters)
    if check == "regional_home_values":
        return _regional_home_values_gate(stage, evidence, parameters)
    if check == "legacy_closure_take_up":
        return _legacy_closure_take_up_gate(stage, evidence, parameters)
    if check == "apprentice_draw":
        return _apprentice_draw_gate(stage, evidence, parameters)
    if check == "isa_subscriptions":
        return _isa_subscriptions_gate(stage, evidence, parameters)
    if check == "isa_subscriptions_invariants":
        return _isa_subscriptions_invariants_gate(stage, evidence, parameters)
    if check == "was_lisa_anchor":
        return _was_lisa_anchor_gate(stage, evidence, parameters)
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
        # Banded in the receipt (receipt_counts.extreme_count): passed through.
        details["donor_floor_rows_raised"] = floor.get("rows_raised", 0)
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
    """The energy kWh rake fits the NEED shape at the published level, at prior weights.

    Fact checks on the lcfs ``energy_rake`` receipt, every published value
    recomputed here from the vendored rows the stage declares (never taken
    from the receipt): each cell's ``shape_target`` must be the vendored NEED
    mean of the declared consumption year; each fuel's level block must be
    the stage's declared published level over its level year (calendar 2024;
    microcosm#1113): the vendored DESNZ Energy Trends total, or for a fuel in
    ``spend_level_fuels`` the kWh the vendored ONS household spending buys at
    QEP's United Kingdom average prices (microcosm#1418); its ``factor`` that
    total over the frame's pre-level total, and the levelled frame total the
    published one; each cell's ``target`` must be shape times factor; and,
    where the stage declares the published gas-connected share, every region
    with a published share must sit within ``maximum_connected_share_deviation``
    of it after the imposition (a region the draw could not fill records a
    shortfall instead; with a declared ``connect_rule`` the walk connects
    households there, the receipt must name the declared rule and salt, and a
    shortfall stands only where the region ran out of households to connect).
    The residual check then holds the maximum absolute
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
    series fails closed. A cell the rake started at its NEED mean because
    every household in it drew no kWh must be one the stage declares in
    ``zero_kwh_cells_at_need_mean`` (microcosm#1240).

    The recomputation shares the stage's inputs, so it cannot see a wrong
    input row (Anthony's review of microcosm#1420). For a fuel levelled to
    ONS spending (electricity; gas keeps Energy Trends while ONS 04.5.2 counts
    bottled gas), the level over Energy Trends' volume must therefore sit
    inside ``ons_level_to_energy_trends_ratio``: 0.926 on the September 2026
    vintage, so a wrong ONS class or vintage, or a unit error, that moves it
    by more than about 8% fails. ONS's own revisions can move it as far: the
    June 2026 vintage gave 0.840 (Vahid's review of microcosm#1420), so a
    re-pin that crosses the bounds asks for the level to be re-adjudicated
    and is not by itself a data error. Every QEP payment method's level falls
    inside the bounds (0.882 to 0.970), so the stage's declared
    ``payment_method`` must equal the gate's ``qep_payment_method`` instead.

    This is where NEED, DESNZ and the connection share are checked; the
    calibrated frame is held to the bound ONS 04.5.1 electricity spend row
    instead (María's ruling, 2026-09-15; the 04.5.2 gas row is measured but
    held out while it counts bottled gas, microcosm#1095), and no gate
    re-checks these facts after calibration.
    """

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.energy_pricing import (
        ELECTRICITY_KWH,
        GAS_CONNECTED_PUBLISHED_METER_SHARE,
        GAS_KWH,
        LEVEL_BASIS_ONS_HOUSEHOLD_SPEND,
        PRICE_DOMESTIC_ENERGY_KIND,
        declared_household_kwh_ceilings,
        declared_zero_kwh_cells_at_need_mean,
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
    payment_method = parameters.get("qep_payment_method")
    if not isinstance(payment_method, str) or not payment_method:
        raise ValueError(f"{stage}: energy_rake declares no qep_payment_method.")
    if str(declared.get("payment_method")) != payment_method:
        raise ValueError(
            f"{stage}: the gate's qep_payment_method {payment_method!r} differs from "
            f"the stage's declared {declared.get('payment_method')!r}."
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
    level, level_receipt = published_energy_level(declared)
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
        "ons_level_to_energy_trends_ratio": {},
    }
    ratio_bounds = parameters.get("ons_level_to_energy_trends_ratio")
    for fuel, fuel_level in sorted(level_receipt["by_fuel"].items()):
        if fuel_level.get("basis") != LEVEL_BASIS_ONS_HOUSEHOLD_SPEND:
            continue
        if not isinstance(ratio_bounds, Mapping) or set(ratio_bounds) != {
            "minimum",
            "maximum",
        }:
            raise ValueError(
                f"{stage}: energy_rake declares no ons_level_to_energy_trends_ratio "
                "minimum and maximum for its ONS-spend levels."
            )
        lower = _finite_number(
            ratio_bounds["minimum"],
            label=f"{stage}.ons_level_to_energy_trends_ratio.minimum",
        )
        upper = _finite_number(
            ratio_bounds["maximum"],
            label=f"{stage}.ons_level_to_energy_trends_ratio.maximum",
        )
        ratio = float(fuel_level["total_kwh"]) / float(fuel_level["energy_trends_kwh"])
        details["ons_level_to_energy_trends_ratio"][fuel] = ratio
        if not lower <= ratio <= upper:
            failures.append(
                f"{stage}: {fuel}'s ONS-spend level is {ratio:.4f} of Energy Trends', "
                f"outside [{ratio_bounds['minimum']}, {ratio_bounds['maximum']}]: check "
                "the ONS class and vintage and the units."
            )
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
                f"the declared published level {published_kwh}."
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
    ceilings = declared_household_kwh_ceilings(declared)
    if ceilings is not None:
        # No household above its fuel's domestic ceiling, and the cap kept the
        # levelled total (microcosm#1164).
        ceiling_block = receipt.get("ceiling")
        if not isinstance(ceiling_block, Mapping):
            failures.append(
                f"{stage}: the stage declares household kWh ceilings but the "
                "energy_rake receipt carries no ceiling block."
            )
            ceiling_block = {}
        details["ceiling"] = {}
        for fuel, ceiling in ceilings.items():
            entry = ceiling_block.get(fuel)
            if not isinstance(entry, Mapping):
                if ceiling_block:
                    failures.append(f"{stage}: ceiling block lacks {fuel}.")
                continue
            if not _close(entry.get("ceiling_kwh"), ceiling, 1e-12):
                failures.append(
                    f"{stage}: {fuel} was capped at {entry.get('ceiling_kwh')!r}, "
                    f"not the declared {ceiling}."
                )
            if entry.get("feasible") is False:
                failures.append(
                    f"{stage}: no household was left below the {fuel} ceiling to "
                    "take what the capped households gave up."
                )
            if entry.get("households_above_after") != 0:
                failures.append(
                    f"{stage}: {fuel} leaves households above the declared "
                    f"ceiling of {ceiling} kWh."
                )
            if not _close(entry.get("frame_kwh_after"), float(level[fuel]), 1e-6):
                failures.append(
                    f"{stage}: {fuel} total after the cap is "
                    f"{entry.get('frame_kwh_after')!r}, not the published "
                    f"{float(level[fuel])}."
                )
            details["ceiling"][fuel] = {
                "ceiling_kwh": ceiling,
                "households_capped": entry.get("households_capped"),
                "capped_share_of_total": entry.get("capped_share_of_total"),
            }
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
    # A cell started at its NEED mean because every household in it drew no
    # kWh stands only where the stage declares it (microcosm#1240).
    held_rule = declared_zero_kwh_cells_at_need_mean(declared)
    held = receipt.get("cells_held_at_need_mean")
    if held is None and held_rule is not None:
        failures.append(
            f"{stage}: the stage declares zero_kwh_cells_at_need_mean but the "
            "energy_rake receipt carries no cells_held_at_need_mean."
        )
    held_entries = list(held or ())
    for entry in held_entries:
        entry = _mapping(entry, label=f"{stage}.cells_held_at_need_mean")
        margin = str(entry.get("margin"))
        cell = str(entry.get("cell"))
        if cell.partition(":")[2] not in (held_rule or {}).get(margin, ()):
            failures.append(
                f"{stage}: {margin} cell {cell!r} was started at its NEED mean, "
                "which the stage does not declare."
            )
    # The count only: a held cell can be a small one, so the details (which
    # receipts may quote) never say which cell drew no kWh.
    details["cells_held_at_need_mean"] = len(held_entries)
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
                # With a declared connect_rule the walk fills a region below
                # its share (microcosm#1164); a shortfall is waived only where
                # the region ran out of households to connect.
                if (
                    declared.get("connect_rule") is not None
                    and float(shortfall) > 0
                    and entry.get("candidates_exhausted") is not True
                ):
                    failures.append(
                        f"{stage}: {region} keeps a gas-connection shortfall of "
                        f"{float(shortfall):.4f} with households left to connect."
                    )
            if declared.get("connect_rule") is not None and (
                connection.get("connect_rule") != declared.get("connect_rule")
                or connection.get("connect_salt") != declared.get("connect_salt")
            ):
                failures.append(
                    f"{stage}: gas connection ran connect rule "
                    f"{connection.get('connect_rule')!r} with salt "
                    f"{connection.get('connect_salt')!r}, not the declared "
                    f"{declared.get('connect_rule')!r} and "
                    f"{declared.get('connect_salt')!r}."
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


def _take_up_signal_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The stage's drawn take-up flags near their contract targets.

    María's ruling of 2026-10-29 (microcosm#1095): the flags the terminal
    ``uk_take_up_signal`` gate measures at the calibrated weights are checked
    where they are drawn too, at the stage's weights before calibration, so a
    draw defect fails at its stage apart from calibration's drift. The receipt
    must carry exactly the declared ``outputs``; each must vary, sit within the
    declared share deviation of its target (the under-2 usage row within its
    own bound) and, for the nested universal hours, have no benefit unit that
    claims the extended hours without them. A share the receipt withholds
    under the disclosure floor cannot be measured, so it fails.
    """

    check = "take_up_signal"
    bound = _finite_number(
        parameters["maximum_share_deviation"],
        label=f"{stage}.maximum_share_deviation",
    )
    under_two = parameters.get("under_two_usage_maximum_share_deviation")
    under_two_bound = (
        bound
        if under_two is None
        else _finite_number(
            under_two, label=f"{stage}.under_two_usage_maximum_share_deviation"
        )
    )
    declared = parameters.get("outputs")
    if not isinstance(declared, list | tuple) or not declared:
        raise ValueError(f"{stage}.outputs must list the stage's take-up flags.")
    expected = [str(output) for output in declared]
    failures: list[str] = []
    details: dict[str, object] = {}
    receipt = _mapping(evidence.get("take_up_signal"), label=f"{stage}.take_up_signal")
    rows = receipt.get("flags")
    if not isinstance(rows, list):
        failures.append(f"{stage}: the receipt carries no take-up flags.")
        rows = []
    by_output: dict[str, Mapping[str, object]] = {}
    for row in rows:
        row = _mapping(row, label=f"{stage}.take_up_signal.flags[]")
        by_output[str(row.get("output"))] = row
    missing = sorted(set(expected) - set(by_output))
    undeclared = sorted(set(by_output) - set(expected))
    if missing:
        failures.append(f"{stage}: the receipt omits the declared flags {missing}.")
    if undeclared:
        failures.append(f"{stage}: the receipt carries undeclared flags {undeclared}.")
    for output in expected:
        row = by_output.get(output)
        if row is None:
            continue
        if "unmeasured" in row:
            failures.append(f"{stage}: {output}: {row['unmeasured']}")
            continue
        share = row.get("weighted_share")
        target = row.get("target")
        if not isinstance(share, int | float) or not isinstance(target, int | float):
            failures.append(
                f"{stage}: {output}'s share or target is withheld ({share!r}, "
                f"{target!r}): its population is too thin to measure at the stage."
            )
            continue
        share = _finite_number(share, label=f"{stage}.{output}.weighted_share")
        target = _finite_number(target, label=f"{stage}.{output}.target")
        limit = under_two_bound if row.get("under_two_usage") is True else bound
        deviation = abs(share - target)
        details[output] = {
            "weighted_share": share,
            "target": target,
            "absolute_deviation": deviation,
            "maximum_share_deviation": limit,
        }
        if int(row.get("distinct_values", 0)) < 2:
            failures.append(
                f"{stage}: {output} is constant; a stochastic take-up draw must "
                "not collapse to a universal default."
            )
        nested = row.get("extended_without_universal_units")
        if nested is not None and nested != 0:
            failures.append(
                f"{stage}: {output}: {nested} benefit unit(s) claim the extended "
                "hours without the universal hours."
            )
        if deviation > limit:
            failures.append(
                f"{stage}: {output}'s weighted share {share:.4f} differs from its "
                f"contract target {target:.4f} by more than {limit}."
            )
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


#: The structural counts the annual income receipt records, banded by
#: ``receipt_counts.extreme_count`` with zero exact; each must be zero.
_ANNUAL_INCOME_STRUCTURAL_ZEROS = (
    "full_year_rows_not_whole",
    "rows_outside_share_bounds",
    "rows_negative_other_spells",
)


def _frs_annual_income_spells_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The annual income descriptors keep their invariants and plausible sizes.

    Every share lies in [0, 1], no other-spell amount is negative and every
    full-year earner keeps the whole year, exactly. The usual-to-DV pay ratio
    the stage measures lies in its bounds, and so does the ratio of annual to
    current employee earners: the survey's spells can add part-year earners
    but cannot remove a current one. Where the receipt measures it, at least
    the declared share of the pay outside the current job is observed rather
    than drawn from a donor. A ratio or share carried by fewer than three
    households, which the receipt prints "<3", is flagged unmeasured, never
    failed: a small-cell rule never stops a build (María's ruling of
    2026-10-26).
    """

    check = "frs_annual_income_spells"
    failures: list[str] = []
    for key in _ANNUAL_INCOME_STRUCTURAL_ZEROS:
        value = evidence.get(key)
        if value != 0:
            failures.append(f"{stage}: {key} is {value!r}, not 0.")
    details: dict[str, object] = {}
    for key, bounds_key in (
        ("pay_concept_ratio", "pay_concept_ratio_bounds"),
        ("annual_over_current_earners", "annual_over_current_earners_bounds"),
    ):
        bounds = parameters[bounds_key]
        if not isinstance(bounds, list | tuple) or len(bounds) != 2:
            raise ValueError(f"{stage}.{bounds_key} must be a [lower, upper] pair.")
        lower = _finite_number(bounds[0], label=f"{stage}.{bounds_key}[0]")
        upper = _finite_number(bounds[1], label=f"{stage}.{bounds_key}[1]")
        value = evidence.get(key)
        details[key] = value
        if value == UK_WITHHELD:
            details[f"{key}_measured"] = False
        elif isinstance(value, bool) or not isinstance(value, int | float):
            failures.append(f"{stage}: {key} is {value!r}, not a measured ratio.")
        elif not lower <= float(value) <= upper:
            failures.append(
                f"{stage}: {key} {float(value):.4f} lies outside [{lower}, {upper}]."
            )
    minimum = _finite_number(
        parameters["minimum_observed_pay_share"],
        label=f"{stage}.minimum_observed_pay_share",
    )
    observed = evidence.get("other_spells_pay_observed_share")
    details["other_spells_pay_observed_share"] = observed
    if observed == UK_WITHHELD:
        details["observed_pay_share_measured"] = False
    elif isinstance(observed, bool) or not isinstance(observed, int | float):
        failures.append(f"{stage}: other_spells_pay_observed_share is {observed!r}.")
    elif float(observed) < minimum:
        failures.append(
            f"{stage}: only {float(observed):.4f} of the pay outside the current "
            f"job is observed, below {minimum}."
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
    least the rate there. DWP's rates cover Great Britain, so each band is
    measured there, and a receipt measured over any other scope fails
    (uk-data#510).
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
        scope = band.get("scope")
        details[name] = {
            "rate": rate,
            "realized_take_up": realized,
            "units": units,
            "scope": scope,
        }
        if scope != "great_britain":
            failures.append(
                f"{stage}: band {name} is measured over {scope!r}, not Great Britain."
            )
            continue
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


def _spi_benefit_coherence_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """SPI-channel benefit reports and take-up are coherent (microcosm#1095).

    Structural zeros only: no SPI row keeps a zeroed report, every restored
    report equals the FRS twin's, no person's ``receives_benefits_in_own_right``
    disagrees with their own reports, base rows are unchanged, every SPI unit
    with a Universal Credit reporter claims, and no SPI unit outside the
    Universal Credit age population claims without reporting.
    """

    check = "spi_benefit_coherence"
    failures: list[str] = []
    zeroed = _mapping(evidence.get("zeroed"), label=f"{stage}.zeroed")
    restored = _mapping(evidence.get("restored"), label=f"{stage}.restored")
    own_right = _mapping(
        evidence.get("benefits_in_own_right"), label=f"{stage}.benefits_in_own_right"
    )
    take_up = _mapping(
        evidence.get("universal_credit_take_up"),
        label=f"{stage}.universal_credit_take_up",
    )
    for column in parameters["zeroed_columns"]:
        receipt = zeroed.get(column)
        if not isinstance(receipt, Mapping):
            failures.append(f"{stage}: the receipt does not record zeroing {column}.")
        elif int(receipt.get("rows_reporting_after", -1)) != 0:
            failures.append(f"{stage}: SPI rows still report {column}.")
    for column in parameters["restored_columns"]:
        receipt = restored.get(column)
        if not isinstance(receipt, Mapping):
            failures.append(f"{stage}: the receipt does not record restoring {column}.")
        elif int(receipt.get("rows_differing_from_twin_after", -1)) != 0:
            failures.append(f"{stage}: SPI {column} differs from the FRS twin.")
    if int(own_right.get("mismatches_after", -1)) != 0:
        failures.append(
            f"{stage}: receives_benefits_in_own_right disagrees with own reports."
        )
    if evidence.get("base_rows_unchanged") is not True:
        failures.append(f"{stage}: base rows are not recorded as unchanged.")
    for key, label in (
        ("reporters_not_claiming", "SPI Universal Credit reporters not claiming"),
        (
            "outside_population_non_reporters_claiming",
            "SPI units outside the Universal Credit population claiming",
        ),
    ):
        if int(take_up.get(key, -1)) != 0:
            failures.append(f"{stage}: {label}: {take_up.get(key)}.")
    details = {
        "zeroed": {
            column: _mapping(zeroed.get(column, {}), label=f"{stage}.zeroed").get(
                "rows_reporting_before"
            )
            for column in parameters["zeroed_columns"]
        },
        "spi_claiming_before": take_up.get("spi_claiming_before"),
        "spi_claiming_after": take_up.get("spi_claiming_after"),
        "own_right_changed": own_right.get("spi_rows_changed"),
    }
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _uc_take_up_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """Each family type's reachable pools realize DWP's caseload (microcosm#1334).

    Every reporter claims and no unit outside the draw changes. Over a family
    type's pools whose target the draw can reach, the realized Great Britain
    caseload is within ``maximum_caseload_deviation`` of DWP's count, and the
    family type's whole caseload is no further from DWP's than before the
    redraw, give or take that share of DWP's count. A family type whose
    caseload rests on fewer than three sources, which the receipt prints "<3",
    is flagged unchecked, never failed: a small-cell rule never stops a build
    (María's ruling of 2026-10-26).
    """

    check = "uc_take_up"
    tolerance = _finite_number(
        parameters["maximum_caseload_deviation"],
        label=f"{stage}.maximum_caseload_deviation",
    )
    failures: list[str] = []
    details: dict[str, object] = {}
    for key in ("reporters_not_claiming", "changed_outside_draw"):
        value = evidence.get(key)
        details[key] = value
        if value != 0:
            failures.append(f"{stage}: {key} is {value!r}, not 0.")
    families = evidence.get("families")
    if not isinstance(families, Mapping) or not families:
        failures.append(f"{stage}: the receipt carries no family types.")
        families = {}
    unchecked: list[str] = []
    for name, family in families.items():
        family = _mapping(family, label=f"{stage}.families.{name}")
        values = {
            key: family.get(key)
            for key in (
                "dwp_caseload",
                "caseload_before",
                "caseload_after",
                "reachable_pools_dwp_caseload",
                "reachable_pools_caseload_after",
            )
        }
        details[str(name)] = values
        if any(value == UK_WITHHELD for value in values.values()):
            unchecked.append(str(name))
            continue
        if any(not isinstance(value, int | float) for value in values.values()):
            failures.append(f"{stage}: family type {name} carries no caseload.")
            continue
        dwp = _finite_number(values["dwp_caseload"], label=f"{stage}.{name}.dwp")
        before = _finite_number(values["caseload_before"], label=f"{stage}.{name}")
        after = _finite_number(values["caseload_after"], label=f"{stage}.{name}")
        target = _finite_number(
            values["reachable_pools_dwp_caseload"], label=f"{stage}.{name}"
        )
        realized = _finite_number(
            values["reachable_pools_caseload_after"], label=f"{stage}.{name}"
        )
        if target > 0.0 and abs(realized - target) > tolerance * target:
            failures.append(
                f"{stage}: family type {name} realizes {realized:,.0f} against "
                f"DWP's {target:,.0f} over its reachable pools (tolerance "
                f"{tolerance})."
            )
        if abs(after - dwp) > abs(before - dwp) + tolerance * dwp:
            failures.append(
                f"{stage}: family type {name} moves from {before:,.0f} to "
                f"{after:,.0f}, further from DWP's {dwp:,.0f}."
            )
    details["family_types_unchecked_small_cells"] = unchecked
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
    "main_residence_mortgage_off_mortgaged_tenure_rows",
    "main_residence_value_off_owner_tenure_rows",
)
_WEALTH_IDENTITY_ZERO_KEYS = (
    "property_wealth_violation_rows",
    "corporate_wealth_violation_rows",
    "gross_financial_wealth_violation_rows",
    "net_financial_wealth_violation_rows",
    "mortgage_debt_violation_rows",
    # The share split and the secured debts (microcosm#1095).
    "share_split_violation_rows",
    "secured_debt_violation_rows",
)


def _wealth_coherence_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The WAS wealth columns are coherent with tenure and with each other.

    The stage draws the main residence and the main-residence mortgage
    inside their tenure stratum and derives every total from drawn
    components (mortgage debt included: the main-residence mortgage plus the
    mortgages on other property, which any tenure may hold); this gate
    re-reads the receipt so the battery, not only the transform, holds that
    no household off a mortgaged tenure carries a main-residence mortgage,
    none off an owner tenure carries a main-residence value, every total
    equals its components, the share split and the ISA sum to corporate
    wealth, every secured debt sits on a held asset within the mortgages on
    other property, and owners without a main-residence value are no more
    common than on the donor beyond the reviewed excess.
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
    receipt's to ``maximum_relative_deviation``, as must the fares-index factor
    that re-prices each fiscal-year yield to the declared calendar year
    (microcosm#1113); the unpriced regions must be the declared ones; and the
    receipt must state that the chain conditioned on the raw draw. The frame-implied boardings against the published ones
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
            ("calendar_price_factor", area.calendar_price_factor),
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


def _consumption_basis_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """Every LCFS spend column is on the declared base year's prices.

    Checks on the receipts of the steps that set the spend levels
    (microcosm#1113). The donor uprating must end at the declared year and
    carry every diary spend column from the April-to-March diary year, placed
    on the calendar-year index at three quarters and one quarter, not by a
    full year's step, which would land a quarter past the calendar year. The
    energy prices' and the energy level's periods, the bus prices' re-pricing
    year and the road-fuel level's period must all be the declared calendar
    year. policyengine-uk can then project every column from the H5's year
    with no per-column bases. A step whose receipt is absent fails the check.
    """

    from microcosm.build.uk_runtime.donor_uprating import (
        FROM_WINDOW_FISCAL_YEAR,
        UK_FISCAL_YEAR_CALENDAR_WEIGHTS,
    )
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_DIARY_PRICE_COLUMNS,
    )

    check = "consumption_basis"
    period_type = str(parameters.get("period_type") or "")
    if period_type != "calendar_year":
        raise ValueError(f"{stage}: the spend basis must be a calendar year.")
    period_value = parameters.get("period_value")
    if isinstance(period_value, bool) or not isinstance(period_value, int):
        raise ValueError(f"{stage}: period_value must be an integer year.")
    expected = (period_type, period_value)
    failures: list[str] = []
    found: dict[str, object] = {}

    def receipt(key: str) -> Mapping[str, object] | None:
        value = evidence.get(key)
        if not isinstance(value, Mapping):
            failures.append(f"{stage}: the stage recorded no {key} receipt.")
            return None
        return value

    def period_of(block: Mapping[str, object]) -> tuple[object, object]:
        return (block.get("period_type"), block.get("period_value"))

    uprating = receipt("donor_uprating")
    if uprating is not None:
        if uprating.get("to_period") != period_value:
            failures.append(
                f"{stage}: the donor uprating moves the diary to "
                f"{uprating.get('to_period')!r}, not {period_value}."
            )
        bridge = uprating.get("fiscal_year_bridge")
        bridge = bridge if isinstance(bridge, Mapping) else {}
        weights = (bridge.get("start_year_weight"), bridge.get("next_year_weight"))
        columns = uprating.get("columns")
        columns = columns if isinstance(columns, Mapping) else {}
        full_step = [
            column
            for column in UK_LCFS_DIARY_PRICE_COLUMNS
            if not isinstance(columns.get(column), Mapping)
            or columns[column].get("from_window") != FROM_WINDOW_FISCAL_YEAR
        ]
        bridged = not full_step and weights == UK_FISCAL_YEAR_CALENDAR_WEIGHTS
        found["donor_uprating"] = (
            "calendar_year" if bridged else "fiscal_year",
            uprating.get("to_period"),
        )
        # The recorded factor must be the bridge's: after / base, where base is
        # the declared mix of the starting and following years (the #1167
        # re-review's nit).
        off_bridge = []
        for column in UK_LCFS_DIARY_PRICE_COLUMNS:
            entry = columns.get(column)
            if not isinstance(entry, Mapping):
                continue
            after, base, factor = (
                entry.get("after"),
                entry.get("base"),
                entry.get("factor"),
            )
            if not all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in (after, base, factor)
            ):
                continue
            if base <= 0 or abs(
                float(factor) - float(after) / float(base)
            ) > 1e-9 * max(1.0, abs(float(factor))):
                off_bridge.append(column)
        if off_bridge:
            failures.append(
                f"{stage}: the donor uprating's recorded factor is not after / base "
                f"for {off_bridge}."
            )
        if full_step:
            failures.append(
                f"{stage}: the donor uprating does not carry {full_step} from "
                "the April-to-March diary year, so a full year's step leaves "
                f"them a quarter past calendar {period_value}."
            )
        elif weights != UK_FISCAL_YEAR_CALENDAR_WEIGHTS:
            failures.append(
                f"{stage}: the donor uprating places the diary year on the "
                f"calendar-year index at {weights}, not "
                f"{UK_FISCAL_YEAR_CALENDAR_WEIGHTS}."
            )
    energy = receipt("energy_pricing")
    if energy is not None:
        level = energy.get("level")
        level = level if isinstance(level, Mapping) else {}
        found["energy_prices"] = period_of(energy)
        found["energy_level"] = period_of(level)
        for label, observed in (
            ("energy prices", found["energy_prices"]),
            ("energy level", found["energy_level"]),
        ):
            if observed != expected:
                failures.append(
                    f"{stage}: the {label} are for {observed}, not {expected}."
                )
    bus = receipt("bus_pricing")
    if bus is not None:
        found["bus_prices"] = bus.get("price_year")
        if bus.get("price_year") != period_value:
            failures.append(
                f"{stage}: bus fares are priced for {bus.get('price_year')!r}, not "
                f"{period_value}."
            )
    fuel = receipt("road_fuel_level")
    if fuel is not None:
        found["road_fuel_level"] = period_of(fuel)
        if period_of(fuel) != expected:
            failures.append(
                f"{stage}: the road-fuel level is for {period_of(fuel)}, not {expected}."
            )
    details = {"period": list(expected), "steps": found}
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _division_levelling_factors(
    stage: str,
    levelling: object,
    declared: Mapping[str, object],
    failures: list[str],
) -> dict[str, float]:
    """Check the receipt's division levelling against the declaration.

    Returns each division's factor (none where the stage declares no
    levelling), appending a failure for a missing or undeclared levelling, a
    rule or division list other than the declared one, a factor outside
    [0, 1], or a division that moves though it is not above the shared ratio.
    """

    expected = declared.get("level_division_excess")
    if expected is None:
        if levelling is not None:
            failures.append(
                f"{stage}: records a division levelling the stage does not declare."
            )
        return {}
    if not isinstance(levelling, Mapping):
        failures.append(
            f"{stage}: records no division levelling; the stage declares "
            f"{expected['rule']!r}."
        )
        return {}
    if (levelling.get("rule"), levelling.get("basis")) != (
        expected["rule"],
        expected["basis"],
    ):
        failures.append(
            f"{stage}: levels the divisions by {levelling.get('rule')!r} on "
            f"{levelling.get('basis')!r}, not the declared levelling."
        )
    shared = _finite_number(
        levelling.get("shared_ratio"), label=f"{stage}.division_levelling.shared_ratio"
    )
    if not shared > 0:
        failures.append(
            f"{stage}: the divisions' shared ratio {shared} is not positive."
        )
    divisions = _mapping(
        levelling.get("divisions"), label=f"{stage}.division_levelling.divisions"
    )
    if sorted(divisions) != sorted(expected["divisions"]):
        failures.append(
            f"{stage}: levels {sorted(divisions)}, not the declared "
            f"{sorted(expected['divisions'])}."
        )
    factors: dict[str, float] = {}
    for division, entry in divisions.items():
        entry = _mapping(entry, label=f"{stage}.division_levelling.{division}")
        factor = _finite_number(
            entry.get("factor"), label=f"{stage}.division_levelling.{division}.factor"
        )
        ratio = entry.get("ratio")
        if not 0.0 <= factor <= 1.0:
            failures.append(
                f"{stage}: {division} levelling factor {factor} is not in [0, 1]."
            )
        if factor != 1.0 and (
            ratio is None
            or _finite_number(
                ratio, label=f"{stage}.division_levelling.{division}.ratio"
            )
            <= shared
        ):
            failures.append(
                f"{stage}: {division} moves by {factor} though its ratio {ratio!r} "
                f"is not above the shared {shared}."
            )
        factors[str(division)] = factor
    return factors


def _recomposed_totals_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """Every recomposed COICOP total contains its levelled components.

    Checks on the ``recomposed_totals`` receipt (microcosm#1113): each total
    the stage declares keeps its own draw less the declared drawn parts and
    adds the declared levelled parts back, held to the declared
    ``maximum_total`` (microcosm#1164); no remainder is negative and no
    household's total is below its components; to
    ``maximum_relative_deviation``, the weighted remainder equals the drawn
    total less the drawn parts plus the floored mass, and the weighted total
    the remainder plus the components less the mass held at the maximum; and
    the published ONS heating-fuel spend is recomputed from the vendored rows
    through the stage's own declaration. Where the stage declares
    ``level_division_excess`` (microcosm#1278), the receipt carries the
    declared rule over the declared divisions, every factor is in [0, 1], no
    division at or below the shared ratio moves, and each recomposed total's
    remainder moves by its division's factor.
    """

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.lcfs_consumption import (
        published_fuel_spend,
        recompose_from_remainder_operation,
    )

    check = "recomposed_totals"
    receipt = _mapping(
        evidence.get("recomposed_totals"), label=f"{stage}.recomposed_totals"
    )
    tolerance = _finite_number(
        parameters.get("maximum_relative_deviation"),
        label=f"{stage}.maximum_relative_deviation",
    )
    declared = recompose_from_remainder_operation(
        load_country_spec("uk").sources.stage_map()[stage]
    )
    if declared is None:
        raise ValueError(f"{stage}: declares no recompose_from_remainder operation.")
    parents = _mapping(
        receipt.get("parents"), label=f"{stage}.recomposed_totals.parents"
    )
    declared_parents = declared["parents"]
    failures: list[str] = []
    division_factors = _division_levelling_factors(
        stage, receipt.get("division_levelling"), declared, failures
    )
    if set(parents) != set(declared_parents):
        failures.append(
            f"{stage}: recomposed {sorted(parents)}, not the declared "
            f"{sorted(declared_parents)}."
        )
    for parent in sorted(set(parents) & set(declared_parents)):
        entry = _mapping(parents[parent], label=f"{stage}.recomposed_totals.{parent}")
        spec = declared_parents[parent]
        for key in ("drawn_subtracts", "components"):
            observed = entry.get(key)
            expected = spec[key]
            if (list(observed) if isinstance(observed, list | tuple) else observed) != (
                list(expected) if isinstance(expected, list | tuple) else expected
            ):
                failures.append(
                    f"{stage}: {parent} {key} {observed!r} is not the declared "
                    f"{expected!r}."
                )
        minimum = _finite_number(
            entry.get("minimum_remainder"), label=f"{stage}.{parent}.minimum_remainder"
        )
        if minimum < 0:
            failures.append(f"{stage}: {parent} has a negative remainder ({minimum}).")
        below = entry.get("rows_below_components")
        if isinstance(below, bool) or not isinstance(below, int) or below != 0:
            failures.append(
                f"{stage}: {below!r} households have {parent} below its components."
            )
        if entry.get("maximum_total") != declared.get("maximum_total"):
            failures.append(
                f"{stage}: {parent} is held to {entry.get('maximum_total')!r}, not "
                f"the declared maximum_total {declared.get('maximum_total')!r}."
            )
        held = entry.get("rows_held_at_maximum")
        if isinstance(held, bool) or not isinstance(held, int) or held < 0:
            failures.append(
                f"{stage}: {parent} records {held!r} rows held at the maximum."
            )

        def number(key: str, parent: str = parent, entry=entry) -> float:
            return _finite_number(entry.get(key), label=f"{stage}.{parent}.{key}")

        # The declared levelling of the divisions' excess (microcosm#1278)
        # scales a recomposed total's remainder by a factor in [0, 1].
        factor = (
            1.0 if entry.get("levelling_factor") is None else number("levelling_factor")
        )
        if not 0.0 <= factor <= 1.0:
            failures.append(
                f"{stage}: {parent} levelling factor {factor} is not in [0, 1]."
            )
        if factor != division_factors.get(parent, 1.0):
            failures.append(
                f"{stage}: {parent} levelling factor {factor} is not its "
                f"division's {division_factors.get(parent, 1.0)}."
            )
        remainder = number("weighted_remainder")
        split = factor * (
            number("weighted_drawn_total")
            - number("weighted_drawn_subtracts")
            + number("weighted_floored_mass")
        )
        if abs(remainder - split) > tolerance * max(1.0, abs(remainder)):
            failures.append(
                f"{stage}: {parent} remainder {remainder} is not its draw less the "
                f"drawn parts plus the floored mass, at its levelling factor ({split})."
            )
        total = number("weighted_total")
        held_mass = number("weighted_held_mass")
        if held_mass < 0:
            failures.append(
                f"{stage}: {parent} records a negative held mass ({held_mass})."
            )
        parts = remainder + number("weighted_components") - held_mass
        if abs(total - parts) > tolerance * max(1.0, abs(total)):
            failures.append(
                f"{stage}: {parent} weighted total {total} is not its remainder plus "
                f"components less the mass held at the maximum ({parts})."
            )
    published = published_fuel_spend(declared)
    observed = _mapping(
        receipt.get("published_fuels"), label=f"{stage}.published_fuels"
    )
    observed_classes = observed.get("classes")
    published_keys = ("coicop", "concept", "column", "source_record_id", "value")
    if (
        not isinstance(observed_classes, list)
        or [
            {key: entry.get(key) for key in published_keys}
            for entry in observed_classes
            if isinstance(entry, Mapping)
        ]
        != published["classes"]
        or observed.get("total") != published["total"]
    ):
        failures.append(
            f"{stage}: the published fuel spend {observed.get('total')!r} is not the "
            f"vendored {published['total']}."
        )
    details = {
        "maximum_relative_deviation": tolerance,
        "parents": sorted(parents),
        "published_fuels_total": published["total"],
    }
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _road_fuel_incidence_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """Every household the fuel flag marks carries road fuel after the redraw.

    Checks on the ``road_fuel_incidence`` receipt (microcosm#1113): the redraw
    ran with the stage's declared rule, seed, salt and trees; no flagged
    household is left at zero and no unflagged household carries fuel; the
    total model fitted the positive-only regime; and the chain's zero share
    among flagged households, the diary artefact the step corrects, is at most
    ``maximum_flagged_zero_share`` (a larger share would mean the chain, not
    the two-week diary, is at fault).
    """

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_ROAD_FUEL_TOTAL,
        road_fuel_incidence_operation,
    )

    check = "road_fuel_incidence"
    receipt = _mapping(
        evidence.get("road_fuel_incidence"), label=f"{stage}.road_fuel_incidence"
    )
    maximum_zero_share = _finite_number(
        parameters.get("maximum_flagged_zero_share"),
        label=f"{stage}.maximum_flagged_zero_share",
    )
    declared = road_fuel_incidence_operation(
        load_country_spec("uk").sources.stage_map()[stage]
    )
    if declared is None:
        raise ValueError(f"{stage}: declares no redraw_zero_road_fuel operation.")
    failures: list[str] = []
    for key in ("rule", "seed", "salt", "n_estimators"):
        if receipt.get(key) != declared.get(key):
            failures.append(
                f"{stage}: the redraw ran with {key} {receipt.get(key)!r}, not the "
                f"declared {declared.get(key)!r}."
            )
    for key in ("flagged_zero_after", "unflagged_with_fuel"):
        value = receipt.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            failures.append(f"{stage}: {key} is {value!r}, not 0.")
    zero_before = receipt.get("flagged_zero_before")
    if isinstance(zero_before, bool) or not isinstance(zero_before, int):
        failures.append(f"{stage}: flagged_zero_before {zero_before!r} is not a count.")
    elif zero_before > 0:
        regimes = receipt.get("regimes")
        regime = (
            regimes.get(UK_LCFS_ROAD_FUEL_TOTAL)
            if isinstance(regimes, Mapping)
            else None
        )
        if regime != "positive_only":
            failures.append(
                f"{stage}: the road-fuel total fitted regime {regime!r}, not "
                "'positive_only'."
            )
    share = _finite_number(
        receipt.get("flagged_zero_share_before"),
        label=f"{stage}.flagged_zero_share_before",
    )
    if not 0.0 <= share <= maximum_zero_share:
        failures.append(
            f"{stage}: the chain left {share:.4f} of flagged households at zero road "
            f"fuel, outside [0, {maximum_zero_share}]."
        )
    details = {
        "maximum_flagged_zero_share": maximum_zero_share,
        "flagged_households": receipt.get("flagged_households"),
        "flagged_zero_before": zero_before,
        "flagged_zero_share_before": share,
        "mean_positive_before": receipt.get("mean_positive_before"),
        "mean_redrawn": receipt.get("mean_redrawn"),
    }
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _road_fuel_level_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The road-fuel step levelled petrol plus diesel to the published ONS spend.

    Fact checks on the ``road_fuel_level`` receipt (microcosm#1113): the ONS
    COICOP 07.2.2 level is recomputed here from the vendored row through the
    stage's own declaration (never taken from the receipt) and must equal the
    receipt's; the levelled prior-weighted total must equal the published
    spend less the donor's other-fuels share, and the factor must reproduce it
    from the drawn total, to ``maximum_relative_deviation``. The other-fuels
    share comes from the licensed LCFS donor, so it is bounded, not recomputed:
    it must lie in [0, ``maximum_other_fuels_share``].
    """

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.lcfs_consumption import (
        road_fuel_level,
        road_fuel_level_operation,
    )

    check = "road_fuel_level"
    receipt = _mapping(
        evidence.get("road_fuel_level"), label=f"{stage}.road_fuel_level"
    )
    tolerance = _finite_number(
        parameters.get("maximum_relative_deviation"),
        label=f"{stage}.maximum_relative_deviation",
    )
    maximum_other = _finite_number(
        parameters.get("maximum_other_fuels_share"),
        label=f"{stage}.maximum_other_fuels_share",
    )
    declared = road_fuel_level_operation(
        load_country_spec("uk").sources.stage_map()[stage]
    )
    if declared is None:
        raise ValueError(f"{stage}: declares no level_road_fuel operation.")
    share = _finite_number(
        receipt.get("other_fuels_share"), label=f"{stage}.other_fuels_share"
    )
    failures: list[str] = []
    if not 0.0 <= share <= maximum_other:
        failures.append(
            f"{stage}: other-fuels share {share} is outside [0, {maximum_other}]."
        )
        share = min(max(share, 0.0), maximum_other)
    level = road_fuel_level(declared, other_fuels_share=share)

    def _close(observed: object, value: float) -> bool:
        return isinstance(observed, int | float) and abs(float(observed) - value) <= (
            tolerance * max(1.0, abs(value))
        )

    for key, value in (("published", level.published), ("level", level.level)):
        if not _close(receipt.get(key), value):
            failures.append(
                f"{stage}: road-fuel {key} {receipt.get(key)!r} is not the vendored "
                f"{value}."
            )
    if receipt.get("source_record_id") != level.receipt["source_record_id"]:
        failures.append(
            f"{stage}: the receipt names {receipt.get('source_record_id')!r}, not the "
            f"declared row {level.receipt['source_record_id']!r}."
        )
    if not _close(receipt.get("frame_after"), level.level):
        failures.append(
            f"{stage}: the levelled road-fuel total {receipt.get('frame_after')!r} is "
            f"not the level {level.level}."
        )
    before = receipt.get("frame_before")
    factor = receipt.get("factor")
    if not (
        isinstance(before, int | float)
        and isinstance(factor, int | float)
        and _close(float(before) * float(factor), level.level)
    ):
        failures.append(
            f"{stage}: factor {factor!r} times the drawn total {before!r} does not "
            f"reproduce the level {level.level}."
        )
    details = {
        "maximum_relative_deviation": tolerance,
        "published": level.published,
        "other_fuels_share": share,
        "level": level.level,
        "factor": factor,
        "petrol_share_of_level": receipt.get("petrol_share_of_level"),
    }
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _excise_quantities_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The excise step gave the frame the published quantities (microcosm#1359).

    Fact checks on the ``excise_quantities`` and ``excise_shares`` receipts:
    each engine input's published quantity (alcohol and tobacco clearances,
    gambling bases) is recomputed here from the vendored rows through the
    stage's own declaration (never taken from the receipt) and must equal the
    receipt's, its frame total at prior weights must equal it and the factor
    must reproduce it from the household spending, to
    ``maximum_relative_deviation``; an input is positive exactly where its
    spending is; each draught share written is the published ratio; the basis
    is the declared fiscal year over the engine's 52 weeks. The shares the
    spending comes from must have stayed in [0, 1] on the frame, each must
    rest on ``required_support_band`` donors, and the donor share whose
    published total sits beside its diary lines must be at most
    ``maximum_parent_residual_share``. A share the receipt withholds for
    resting on under ten donors is flagged unchecked, never failed (María's
    ruling of 2026-10-26).
    """

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_EXCISE_SHARES,
        UK_PE_UK_WEEKS_IN_YEAR,
        excise_levels,
        excise_quantities_operation,
    )

    check = "excise_quantities"
    receipt = _mapping(
        evidence.get("excise_quantities"), label=f"{stage}.excise_quantities"
    )
    shares = _mapping(evidence.get("excise_shares"), label=f"{stage}.excise_shares")
    tolerance = _finite_number(
        parameters.get("maximum_relative_deviation"),
        label=f"{stage}.maximum_relative_deviation",
    )
    maximum_residual = _finite_number(
        parameters.get("maximum_parent_residual_share"),
        label=f"{stage}.maximum_parent_residual_share",
    )
    required_support = parameters.get("required_support_band")
    declared = excise_quantities_operation(
        load_country_spec("uk").sources.stage_map()[stage]
    )
    if declared is None:
        raise ValueError(f"{stage}: declares no level_excise_quantities operation.")
    resolved = excise_levels(declared)
    levels, volume = resolved.quantities, resolved.volume

    def _close(observed: object, value: float) -> bool:
        return isinstance(observed, int | float) and abs(float(observed) - value) <= (
            tolerance * max(1.0, abs(value))
        )

    failures: list[str] = []
    for key, value in (
        ("period_type", "fiscal_year"),
        ("period_value", int(declared["period_value"])),
        ("weeks_per_year", UK_PE_UK_WEEKS_IN_YEAR),
    ):
        if receipt.get(key) != value:
            failures.append(
                f"{stage}: the excise receipt's {key} {receipt.get(key)!r} is not "
                f"{value!r}."
            )
    if receipt.get("alcohol_volume_weights") != dict(volume.receipt):
        failures.append(
            f"{stage}: the alcohol volume weights {receipt.get('alcohol_volume_weights')!r} "
            f"are not the declared {dict(volume.receipt)!r}."
        )
    outputs = _mapping(
        receipt.get("outputs"), label=f"{stage}.excise_quantities.outputs"
    )
    if set(outputs) != set(levels):
        failures.append(
            f"{stage}: the excise receipt levels {sorted(outputs)}, not the declared "
            f"{sorted(levels)}."
        )
    keys = _mapping(receipt.get("keys"), label=f"{stage}.excise_quantities.keys")
    for column, level in levels.items():
        entry = outputs.get(column)
        if not isinstance(entry, Mapping):
            continue
        for key, value in (
            ("published", float(level.receipt["published"])),
            ("annual_level", level.annual),
        ):
            if not _close(entry.get(key), value):
                failures.append(
                    f"{stage}: {column} {key} {entry.get(key)!r} is not the vendored "
                    f"{value}."
                )
        if entry.get("source_record_id") != level.receipt["source_record_id"]:
            failures.append(
                f"{stage}: {column} names {entry.get('source_record_id')!r}, not the "
                f"declared row {level.receipt['source_record_id']!r}."
            )
        if not _close(entry.get("frame_annual_total"), level.annual):
            failures.append(
                f"{stage}: the frame's {column} {entry.get('frame_annual_total')!r} is "
                f"not the level {level.annual}."
            )
        before, factor = entry.get("key_total"), entry.get("factor")
        if not (
            isinstance(before, int | float)
            and isinstance(factor, int | float)
            and _close(float(before) * float(factor), level.annual)
        ):
            failures.append(
                f"{stage}: {column}'s factor {factor!r} times its spending {before!r} "
                f"does not reproduce the level {level.annual}."
            )
        spenders = keys.get(str(entry.get("key")))
        if not (
            isinstance(spenders, Mapping)
            and entry.get("positive_share") == spenders.get("positive_share")
        ):
            failures.append(
                f"{stage}: {column} is positive on a share {entry.get('positive_share')!r} "
                "of households, not exactly where its spending is."
            )
    draught = receipt.get("draught_shares")
    if not isinstance(draught, Mapping) or set(draught) != set(resolved.draught):
        failures.append(
            f"{stage}: the excise receipt writes draught shares "
            f"{sorted(draught) if isinstance(draught, Mapping) else draught!r}, not "
            f"the declared {sorted(resolved.draught)}."
        )
    else:
        for column, published in resolved.draught.items():
            written = (
                draught[column].get("share")
                if isinstance(draught[column], Mapping)
                else None
            )
            if not _close(written, float(published["share"])):
                failures.append(
                    f"{stage}: {column} {written!r} is not the vendored "
                    f"{published['share']}."
                )
    if set(shares) != set(UK_LCFS_EXCISE_SHARES):
        failures.append(
            f"{stage}: the excise_shares receipt names {sorted(shares)}, not "
            f"{sorted(UK_LCFS_EXCISE_SHARES)}."
        )
    unchecked_residuals: list[str] = []
    for name in UK_LCFS_EXCISE_SHARES:
        entry = shares.get(name)
        if not isinstance(entry, Mapping):
            continue
        frame = entry.get("frame") or {}
        donor = entry.get("donor") or {}
        if frame.get("draws_outside_unit_interval") != 0:
            failures.append(f"{stage}: some {name} draws left [0, 1].")
        if donor.get("positive_support") != required_support:
            failures.append(
                f"{stage}: {name} rests on {donor.get('positive_support')!r} donors, "
                f"not {required_support!r}."
            )
        residual = donor.get("total_beside_lines")
        if residual == UK_WITHHELD:
            unchecked_residuals.append(name)
        elif isinstance(residual, int | float) and residual > maximum_residual:
            failures.append(
                f"{stage}: a weighted {residual} of donors carry a published total "
                f"beside {name}'s lines, above {maximum_residual}."
            )
    details = {
        "maximum_relative_deviation": tolerance,
        "levels": {column: level.annual for column, level in levels.items()},
        "alcohol_volume_weights": dict(volume.receipt),
        "residuals_unchecked_small_cells": unchecked_residuals,
    }
    return (
        _fail(stage, check, failures, details)
        if failures
        else _pass(stage, check, details)
    )


def _gambling_participation_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The gambling keys reached published participation (microcosm#1359).

    Fact checks on the ``gambling_participation`` receipt: each key's adult
    rates are recomputed here from the vendored rows through the stage's own
    declaration (never taken from the receipt) and must equal the receipt's
    to ``maximum_relative_deviation``, and the receipt's settings (theta, the
    occasional stake share, the strata, the propensity, the seed and the
    salt) must be the declared ones. Per key, each stratum the receipt gives
    must sit within ``maximum_stratum_shortfall`` of its four-week target, and
    from there of its twelve-month target, or add nothing where the diary
    already passed a target; the national shares must reach the targets less
    that shortfall; no added household may lack a stake; the three classes'
    stake shares must sum to one; the scale that brings the added regular
    players' mean stake to the seen households' must lie inside
    ``amount_scale_bounds``, so a broken stake draw that the anchoring would
    otherwise hide fails; the propensity must rest on
    ``required_support_band`` donors; and the excise levelling must have
    shared the bases by the topped-up keys, so each key is positive on the
    twelve-month share. The receipt withholds a stratum with a cell under ten
    source households, which the gate counts, and a key with such a cell
    nationally, which the gate names as unchecked: a small-cell rule flags and
    never stops a build (María's ruling of 2026-10-26).
    """

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.gambling_participation import (
        UK_GAMBLING_PARTICIPATION_KEYS,
        gambling_participation_operation,
        participation_rates,
        participation_settings,
    )

    check = "gambling_participation"
    receipt = _mapping(
        evidence.get("gambling_participation"),
        label=f"{stage}.gambling_participation",
    )
    excise = _mapping(
        evidence.get("excise_quantities"), label=f"{stage}.excise_quantities"
    )
    tolerance = _finite_number(
        parameters.get("maximum_relative_deviation"),
        label=f"{stage}.maximum_relative_deviation",
    )
    shortfall = _finite_number(
        parameters.get("maximum_stratum_shortfall"),
        label=f"{stage}.maximum_stratum_shortfall",
    )
    required_support = parameters.get("required_support_band")
    scale_bounds = parameters.get("amount_scale_bounds")
    if not (
        isinstance(scale_bounds, list | tuple)
        and len(scale_bounds) == 2
        and all(
            isinstance(bound, int | float) and not isinstance(bound, bool)
            for bound in scale_bounds
        )
        and 0.0 < float(scale_bounds[0]) <= float(scale_bounds[1])
    ):
        raise ValueError(
            f"{stage}.amount_scale_bounds must be [low, high] with 0 < low <= high, "
            f"got {scale_bounds!r}."
        )
    low_scale, high_scale = (float(bound) for bound in scale_bounds)
    declared = gambling_participation_operation(
        load_country_spec("uk").sources.stage_map()[stage]
    )
    if declared is None:
        raise ValueError(
            f"{stage}: declares no top_up_gambling_participation operation."
        )
    settings = participation_settings(declared)
    facts = participation_rates(declared).receipt

    def _number(value: object) -> bool:
        return isinstance(value, int | float) and not isinstance(value, bool)

    def _close(observed: object, value: float) -> bool:
        return _number(observed) and abs(float(observed) - value) <= (
            tolerance * max(1.0, abs(value))
        )

    failures: list[str] = []
    if receipt.get("settings") != settings:
        failures.append(
            f"{stage}: the participation receipt's settings "
            f"{receipt.get('settings')!r} are not the declared {settings!r}."
        )
    recorded = _mapping(
        receipt.get("facts"), label=f"{stage}.gambling_participation.facts"
    )
    for name in (
        "resource",
        "concept",
        "period_type",
        "period_value",
        "geography_id",
        "rows_read",
    ):
        if recorded.get(name) != facts[name]:
            failures.append(
                f"{stage}: the participation facts' {name} {recorded.get(name)!r} "
                f"is not {facts[name]!r}."
            )
    recorded_rates = recorded.get("rates")
    for key, by_window in facts["rates"].items():
        for window, cells in by_window.items():
            for cell, value in cells.items():
                observed = (
                    ((recorded_rates or {}).get(key) or {}).get(window) or {}
                ).get(cell)
                if not _close(observed, value):
                    failures.append(
                        f"{stage}: {key}'s {window} rate for {cell} {observed!r} is "
                        f"not the vendored {value}."
                    )

    def _stratum(key: str, label: str, entry: Mapping[str, object]) -> list[str]:
        names = (
            "before",
            "regular_target",
            "after_regular",
            "annual_target",
            "after_annual",
        )
        if not all(_number(entry.get(name)) for name in names):
            return [f"{stage}: {key} stratum {label} gives no shares to check."]
        values = {name: float(entry[name]) for name in names}
        problems: list[str] = []
        for start, target, after in (
            ("before", "regular_target", "after_regular"),
            ("after_regular", "annual_target", "after_annual"),
        ):
            if values[start] >= values[target]:
                if not _close(values[after], values[start]):
                    problems.append(
                        f"{stage}: {key} stratum {label} was at {values[start]} past "
                        f"its {target} {values[target]} but moved to {values[after]}."
                    )
            elif abs(values[after] - values[target]) > shortfall:
                problems.append(
                    f"{stage}: {key} stratum {label}'s {after} {values[after]} is "
                    f"more than {shortfall} from its {target} {values[target]}."
                )
        return problems

    keys = _mapping(receipt.get("keys"), label=f"{stage}.gambling_participation.keys")
    levelled = _mapping(excise.get("keys"), label=f"{stage}.excise_quantities.keys")
    withheld_strata = 0
    withheld_keys: list[str] = []
    national: dict[str, object] = {}
    for key in UK_GAMBLING_PARTICIPATION_KEYS:
        entry = keys.get(key)
        if not isinstance(entry, Mapping):
            failures.append(f"{stage}: the participation receipt tops up no {key}.")
            continue
        if entry.get("added_without_stake") != 0:
            failures.append(
                f"{stage}: {entry.get('added_without_stake')!r} households added to "
                f"{key} carry no stake."
            )
        propensity = entry.get("propensity")
        support = (
            propensity.get("positive_support")
            if isinstance(propensity, Mapping)
            else None
        )
        if support != required_support:
            failures.append(
                f"{stage}: {key}'s fits rest on {support!r} donors, not "
                f"{required_support!r}."
            )
        strata = entry.get("strata")
        for label, stratum in strata.items() if isinstance(strata, Mapping) else ():
            if isinstance(stratum, Mapping) and stratum.get("withheld") is True:
                withheld_strata += 1
            elif isinstance(stratum, Mapping):
                failures.extend(_stratum(key, str(label), stratum))
            else:
                failures.append(f"{stage}: {key} stratum {label} is not an object.")
        if entry.get("withheld") is True:
            withheld_keys.append(key)
            continue
        for target, after in (
            ("regular_target", "after_regular"),
            ("annual_target", "after_annual"),
        ):
            if not (
                _number(entry.get(target))
                and _number(entry.get(after))
                and float(entry[after]) >= float(entry[target]) - shortfall
            ):
                failures.append(
                    f"{stage}: {key}'s {after} {entry.get(after)!r} falls short of its "
                    f"{target} {entry.get(target)!r}."
                )
        shares = entry.get("stake_shares")
        if not (
            isinstance(shares, Mapping)
            and set(shares) == {"seen", "added_regular", "added_occasional"}
            and all(_number(value) for value in shares.values())
            and _close(sum(float(value) for value in shares.values()), 1.0)
        ):
            failures.append(
                f"{stage}: {key}'s stake shares {shares!r} do not split its stakes."
            )
        scale = entry.get("amount_scale")
        if not (_number(scale) and low_scale <= float(scale) <= high_scale):
            failures.append(
                f"{stage}: {key}'s amount scale {scale!r} is outside "
                f"[{low_scale}, {high_scale}]: the donor's stake draws missed the "
                "seen households' mean by more than the bounds allow."
            )
        excise_key = levelled.get(key)
        positive = (
            excise_key.get("positive_share")
            if isinstance(excise_key, Mapping)
            else None
        )
        if not (
            _number(entry.get("after_annual"))
            and _close(positive, float(entry["after_annual"]))
        ):
            failures.append(
                f"{stage}: the excise levelling's {key} key is positive on {positive!r} "
                f"of households, not the topped-up {entry.get('after_annual')!r}."
            )
        national[key] = {
            name: entry.get(name)
            for name in (
                "regular_target",
                "after_regular",
                "annual_target",
                "after_annual",
            )
        }
    details = {
        "maximum_relative_deviation": tolerance,
        "maximum_stratum_shortfall": shortfall,
        "strata_withheld": withheld_strata,
        "keys_unchecked_small_cells": withheld_keys,
        "national": national,
    }
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


def _property_components_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """Landlords' receipts keep their invariants (microcosm#1106).

    No receipts below profit (the engine would read them as unknown), and no
    landlord placed in a band below their own profit. The walk fills every
    Table 13 band above the lowest to its share of the spine's landlords up to
    whole-person rounding: a band's mass sits within the declared multiple of
    its recorded rounding allowance, the weight of the person straddling each
    of its two boundaries (one straddler for an empty band), capped at twice
    the largest landlord weight. The open top band's profit multiplier stays below
    its declared cap; a multiplier at its floor of 1 is reported, not failed.
    """

    check = "property_components"
    receipts = _mapping(evidence.get("receipts"), label=f"{stage}.receipts")
    failures: list[str] = []
    maximum_below = int(parameters["maximum_rows_receipts_below_profit"])
    below = int(
        _finite_number(
            receipts.get("rows_receipts_below_profit"),
            label=f"{stage}.receipts.rows_receipts_below_profit",
        )
    )
    if below > maximum_below:
        failures.append(
            f"{stage}: {below} rows carry receipts below profit "
            f"(allowed {maximum_below})."
        )
    maximum_above = int(parameters["maximum_rows_with_profit_above_band"])
    above = int(
        _finite_number(
            receipts.get("rows_with_profit_above_their_band"),
            label=f"{stage}.receipts.rows_with_profit_above_their_band",
        )
    )
    if above > maximum_above:
        failures.append(
            f"{stage}: {above} landlords sit in a receipts band below their "
            f"profit (allowed {maximum_above})."
        )
    factor = _finite_number(
        parameters["maximum_band_error_in_boundary_weights"],
        label=f"{stage}.maximum_band_error_in_boundary_weights",
    )
    largest = _finite_number(
        receipts.get("largest_landlord_weight"),
        label=f"{stage}.receipts.largest_landlord_weight",
    )
    bands = receipts.get("bands_top_down")
    if not isinstance(bands, list) or not bands:
        raise ValueError(f"{stage}.receipts.bands_top_down must be a non-empty list.")
    errors = []
    for band in bands[:-1]:
        band = _mapping(band, label=f"{stage}.receipts.bands_top_down[]")
        target = _finite_number(
            band.get("target_landlords"), label=f"{stage} band target"
        )
        walked = _finite_number(
            band.get("walk_landlords_weight"), label=f"{stage} band mass"
        )
        # The stage's allowance is two straddlers' weights at most; a larger
        # reported allowance is capped rather than trusted.
        allowance = min(
            _finite_number(
                band.get("boundary_allowance_weight"),
                label=f"{stage} band rounding allowance",
            ),
            2.0 * largest,
        )
        error = abs(walked - target)
        errors.append(
            {
                "lower_bound": band.get("lower_bound"),
                "error_weight": error,
                "allowance_weight": allowance,
            }
        )
        if error > factor * allowance:
            failures.append(
                f"{stage}: the band from {band.get('lower_bound')} holds "
                f"{walked:.1f} landlords against {target:.1f}, beyond "
                f"{factor} x its rounding allowance {allowance:.1f}."
            )
    top = _mapping(bands[0], label=f"{stage}.receipts.bands_top_down[0]")
    multiplier = _finite_number(
        top.get("profit_multiplier"), label=f"{stage} top-band multiplier"
    )
    bounds = receipts.get("top_band_multiplier_bounds")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError(f"{stage}.receipts.top_band_multiplier_bounds must be a pair.")
    cap = _finite_number(bounds[1], label=f"{stage} top-band multiplier cap")
    if multiplier >= cap:
        failures.append(
            f"{stage}: the top band's profit multiplier {multiplier} reached its "
            f"cap {cap}; the receipts total is out of reach."
        )
    details = {
        "rows_receipts_below_profit": below,
        "rows_with_profit_above_their_band": above,
        "largest_landlord_weight": largest,
        "band_errors": errors,
        "top_band_multiplier": multiplier,
        "top_band_multiplier_cap": cap,
        "top_band_multiplier_at_floor": bool(top.get("multiplier_at_floor", False)),
    }
    if failures:
        return _fail(stage, check, failures, details)
    return _pass(stage, check, details)


def _high_value_home_support_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The high-value home support split conserved mass and met its rule.

    microcosm#1240 fix 1: every region's owner households at or above the
    declared rank are copied the declared number of times at equal weight;
    the total household mass is conserved to the declared deviation and no
    value moves, so nothing distributional is held.
    """

    check = "high_value_home_support"
    stratum_lower = _finite_number(
        parameters["stratum_lower"], label=f"{stage}.stratum_lower"
    )
    copies = _finite_number(parameters["copies"], label=f"{stage}.copies")
    tolerance = _finite_number(
        parameters["maximum_relative_mass_deviation"],
        label=f"{stage}.maximum_relative_mass_deviation",
    )
    effective_tolerance = max(tolerance, _FLOAT_RELATIVE_TOLERANCE)
    failures: list[str] = []
    declared = _mapping(evidence.get("parameters"), label=f"{stage}.parameters")
    for key, expected in (("stratum_lower", stratum_lower), ("copies", copies)):
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
    totals = _mapping(evidence.get("totals"), label=f"{stage}.totals")
    selected = totals.get("households_selected")
    created = totals.get("copies_created")
    for key, value in (("households_selected", selected), ("copies_created", created)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{stage}.totals.{key} must be a non-negative integer.")
    if created != selected * (int(copies) - 1):
        failures.append(
            f"{stage}: {created} copies created for {selected} selected households "
            f"at {int(copies)} copies each."
        )
    rows = evidence.get("regions")
    if not isinstance(rows, list | tuple):
        raise ValueError(f"{stage}.regions must be a list.")
    regions_with_owners = 0
    for row in rows:
        if not isinstance(row, Mapping):
            failures.append(f"{stage}: region row is not an object.")
            continue
        owners = row.get("owner_households")
        chosen = row.get("households_selected")
        if not isinstance(owners, int) or not isinstance(chosen, int):
            raise ValueError(f"{stage}: region counts must be integers.")
        if owners > 0:
            regions_with_owners += 1
            if chosen < 1:
                failures.append(
                    f"{stage}: region {row.get('region')!r} has owner households but "
                    "selected none into the stratum."
                )
    if regions_with_owners == 0:
        failures.append(f"{stage}: no region carries owner households.")
    # The shipped details round the survey counts to the nearest ten (the
    # receipts rule of microcosm#1095); the exact counts stay in the stage
    # receipt the checkpoint holds.
    details = {
        "households_selected_nearest_10": int(round(selected / 10.0)) * 10,
        "copies_created_nearest_10": int(round(created / 10.0)) * 10,
        "mass_deviation": deviation,
        "regions_with_owners": regions_with_owners,
    }
    if failures:
        return _fail(stage, check, failures, details)
    return _pass(stage, check, details)


def _property_purchases_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """Each purchase kind meets HMRC's Stamp Tax totals as the stage contracts (#1308).

    Every declared kind is drawn once. Its transactions meet HMRC's count
    within the declared deviation (the events are drawn, so the count is near,
    not exact); its value is HMRC's less the recorded shortfall, which the
    stage refuses past one fifth, so the gate refuses past the declared share;
    and the individuals' shares that carry the all-buyers totals onto
    households are proper shares. A kind whose buyers rest on fewer than three
    sources, which the receipt prints "<3", is flagged unchecked, never
    failed (María's ruling of 2026-10-26).
    """

    check = "property_purchases"
    expected = sorted(str(kind) for kind in parameters["kinds"])
    deviation = _finite_number(
        parameters["maximum_transactions_relative_deviation"],
        label=f"{stage}.maximum_transactions_relative_deviation",
    )
    maximum_shortfall = _finite_number(
        parameters["maximum_value_shortfall_share"],
        label=f"{stage}.maximum_value_shortfall_share",
    )
    failures: list[str] = []
    details: dict[str, object] = {}
    kinds = evidence.get("kinds")
    if not isinstance(kinds, list):
        kinds = []
    drawn = sorted(
        str(_mapping(kind, label=f"{stage}.kinds[]").get("property_type"))
        for kind in kinds
    )
    if drawn != expected:
        failures.append(f"{stage}: the receipt draws kinds {drawn}, not {expected}.")
    unchecked: list[str] = []
    for kind in kinds:
        name = str(kind.get("property_type"))
        if any(
            kind.get(field) == UK_WITHHELD
            for field in (
                "fitted_transactions",
                "value_shortfall_share",
                "fitted_value",
            )
        ):
            unchecked.append(name)
            continue
        target = _finite_number(
            kind.get("target_transactions"), label=f"{stage}.{name}.target_transactions"
        )
        fitted = _finite_number(
            kind.get("fitted_transactions"), label=f"{stage}.{name}.fitted_transactions"
        )
        shortfall = _finite_number(
            kind.get("value_shortfall_share"),
            label=f"{stage}.{name}.value_shortfall_share",
        )
        target_value = _finite_number(
            kind.get("target_value"), label=f"{stage}.{name}.target_value"
        )
        fitted_value = _finite_number(
            kind.get("fitted_value"), label=f"{stage}.{name}.fitted_value"
        )
        relative = fitted / target - 1.0 if target > 0 else math.inf
        details[name] = {
            "transactions_relative_deviation": relative,
            "value_shortfall_share": shortfall,
        }
        if abs(relative) > deviation:
            failures.append(
                f"{stage}: {name} draws {relative:+.4f} of HMRC's transactions "
                f"(limit {deviation})."
            )
        if not 0.0 <= shortfall <= maximum_shortfall:
            failures.append(
                f"{stage}: {name} falls {shortfall:.4f} short of HMRC's value "
                f"(limit {maximum_shortfall})."
            )
        expected_value = target_value * (1.0 - shortfall)
        if not math.isclose(fitted_value, expected_value, rel_tol=1e-9, abs_tol=1e-6):
            failures.append(
                f"{stage}: {name} fits value {fitted_value:.6g}, not HMRC's less "
                f"the recorded shortfall ({expected_value:.6g})."
            )
    derivation = evidence.get("derivation")
    shares = {}
    if isinstance(derivation, Mapping):
        shares = {
            key: value for key, value in derivation.items() if key.endswith("_share")
        }
    if not shares:
        failures.append(f"{stage}: the receipt carries no individuals' shares.")
    for key, value in shares.items():
        share = _finite_number(value, label=f"{stage}.derivation.{key}")
        if not 0.0 < share <= 1.0:
            failures.append(f"{stage}: derivation {key} is {share}, not a share.")
    details["kinds_unchecked_small_cells"] = unchecked
    if failures:
        return _fail(stage, check, failures, details)
    return _pass(stage, check, details)


def _regional_home_values_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """Home values are re-ranked in every region on the declared knots (#1308).

    Every declared region has rewritten rows and no more than the stage's
    rows; no source sits at the value cap beyond the declared count; the knots
    are the declared basis year and the ranking the declared basis; and the
    weighted value stays positive on both sides of the stage. A count the
    receipt bands for resting on fewer than three records is flagged rather
    than failed where the band cannot tell (María's ruling of 2026-10-26).
    """

    check = "regional_home_values"
    regions = sorted(str(region) for region in parameters["regions"])
    failures: list[str] = []
    rewritten = evidence.get("rows_rewritten_by_region")
    if not isinstance(rewritten, Mapping):
        rewritten = {}
    if sorted(rewritten) != regions:
        failures.append(
            f"{stage}: the receipt rewrites regions {sorted(rewritten)}, not {regions}."
        )
    # A region the receipt prints "<3" rewrote one or two rows: it is not
    # empty, and it is flagged (a small-cell rule never fails a build, María's
    # ruling of 2026-10-26). The sum check takes it at its lower bound.
    banded = sorted(
        str(region) for region, count in rewritten.items() if count == UK_WITHHELD
    )
    counts = {
        str(region): 1 if count == UK_WITHHELD else int(count)
        for region, count in rewritten.items()
    }
    empty = sorted(region for region, count in counts.items() if count <= 0)
    if empty:
        failures.append(f"{stage}: no row is rewritten in {empty}.")
    rows = evidence.get("rows", 0)
    if rows != UK_WITHHELD and sum(counts.values()) > int(rows):
        failures.append(f"{stage}: more rows are rewritten than the stage carries.")
    at_cap = evidence.get("sources_at_cap", -1)
    maximum_at_cap = int(parameters["maximum_sources_at_cap"])
    cap_unchecked = False
    if at_cap in _SUPPORT_BAND_BOUNDS:
        # The receipt bands the count: it fails only when the whole band is
        # past the limit, and a band straddling the limit is flagged.
        lowest, highest = _SUPPORT_BAND_BOUNDS[at_cap]
        if lowest > maximum_at_cap:
            failures.append(
                f"{stage}: {at_cap} sources sit at the value cap "
                f"(limit {maximum_at_cap})."
            )
        elif highest > maximum_at_cap:
            cap_unchecked = True
    elif not 0 <= int(at_cap) <= maximum_at_cap:
        failures.append(
            f"{stage}: {at_cap} sources sit at the value cap (limit {maximum_at_cap})."
        )
    if evidence.get("knots_basis_year") != parameters["knots_basis_year"]:
        failures.append(
            f"{stage}: the knots' basis year is {evidence.get('knots_basis_year')!r}, "
            f"not {parameters['knots_basis_year']!r}."
        )
    ranking = evidence.get("ranking")
    basis = ranking.get("rank_basis") if isinstance(ranking, Mapping) else None
    if basis != parameters["rank_basis"]:
        failures.append(
            f"{stage}: the ranking basis is {basis!r}, not {parameters['rank_basis']!r}."
        )
    for side in ("weighted_value_before", "weighted_value_after"):
        value = _finite_number(evidence.get(side), label=f"{stage}.{side}")
        if value <= 0.0:
            failures.append(f"{stage}: {side} is not positive.")
    details = {
        "regions": regions,
        "sources_at_cap": at_cap,
        "regions_rewritten_under_floor": banded,
        "sources_at_cap_unchecked": cap_unchecked,
    }
    if failures:
        return _fail(stage, check, failures, details)
    return _pass(stage, check, details)


#: The support bands the take-up receipts print (receipt_counts).
_SUPPORT_BANDS = frozenset({UK_WITHHELD, UK_BELOW_TEN_BAND, "10-29", "30+"})
#: The counts each band stands for, lowest and highest.
_SUPPORT_BAND_BOUNDS = {
    UK_WITHHELD: (1, UK_DISCLOSURE_FLOOR - 1),
    UK_BELOW_TEN_BAND: (UK_DISCLOSURE_FLOOR, 9),
    "10-29": (10, 29),
    "30+": (30, math.inf),
}


def _legacy_closure_take_up_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The Move to UC draw records a rate and its basis for every combination (#1308).

    The receipt withholds what a thin combination realizes, so the gate holds
    the contract it can read: the share is measured on the declared basis,
    every combination's rate is a probability from a declared source, its
    support is a printed band, and its unreachable flag is a boolean, a
    withheld "<3" or absent; the share claiming at closure is a share.
    """

    check = "legacy_closure_take_up"
    sources = {str(source) for source in parameters["rate_sources"]}
    failures: list[str] = []
    if evidence.get("p_basis") != parameters["p_basis"]:
        failures.append(
            f"{stage}: the share is measured on {evidence.get('p_basis')!r}, not "
            f"{parameters['p_basis']!r}."
        )
    combinations = evidence.get("combinations")
    if not isinstance(combinations, Mapping) or not combinations:
        failures.append(f"{stage}: the receipt carries no combinations.")
        combinations = {}
    withheld_flags = 0
    for name, combination in combinations.items():
        combination = _mapping(combination, label=f"{stage}.combinations.{name}")
        rate = _finite_number(combination.get("rate"), label=f"{stage}.{name}.rate")
        if not 0.0 <= rate <= 1.0:
            failures.append(f"{stage}: {name} draws at rate {rate}.")
        if combination.get("rate_source") not in sources:
            failures.append(
                f"{stage}: {name} takes its rate from "
                f"{combination.get('rate_source')!r}."
            )
        if combination.get("support") not in _SUPPORT_BANDS:
            failures.append(f"{stage}: {name} prints no support band.")
        flag = combination.get("rate_unreachable")
        if flag == UK_WITHHELD:
            withheld_flags += 1
        elif not (flag is None or isinstance(flag, bool)):
            failures.append(f"{stage}: {name} flags {flag!r} as unreachable.")
    # A drawn cohort on fewer than three sources prints the share withheld.
    printed = evidence.get("great_britain_weighted_share_claiming_at_closure")
    if printed != UK_WITHHELD:
        share = _finite_number(
            printed, label=f"{stage}.great_britain_weighted_share_claiming_at_closure"
        )
        if not 0.0 <= share <= 1.0:
            failures.append(f"{stage}: the share claiming at closure is {share}.")
    details = {
        "combinations": len(combinations),
        "unreachable_flags_withheld": withheld_flags,
        "share_claiming_at_closure_withheld": printed == UK_WITHHELD,
    }
    if failures:
        return _fail(stage, check, failures, details)
    return _pass(stage, check, details)


def _apprentice_draw_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """``is_apprentice`` realizes the contract's per-head rate (microcosm#1373).

    The receipt must carry the contract's rate for its build year under the
    declared key, a target equal to that rate times the weighted population
    aged 21 to 24, and its support as printed bands. Unless the eligible
    employees weigh no more than the target (cap-bound: every one of them is
    flagged), the flagged weight lies within
    ``maximum_relative_realization_deviation`` of the target. The relief the
    flags imply at the engine's own Class 1 schedule must stay at or below
    ``maximum_implied_relief``, HMRC's apprentice relief for every apprentice
    under 25 (Vahid's review of microcosm#1373, S1). A figure resting on fewer
    than ten source households, which the receipt prints "<3", is named
    unchecked, never failed: a small-cell rule never stops a build (María's
    ruling of 2026-10-26).
    """

    from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract

    check = "apprentice_draw"
    receipt = _mapping(
        evidence.get("apprentice_receipt"), label=f"{stage}.apprentice_receipt"
    )
    tolerance = _finite_number(
        parameters["maximum_relative_realization_deviation"],
        label=f"{stage}.maximum_relative_realization_deviation",
    )
    if tolerance < 0.0:
        raise ValueError(
            f"{stage}: maximum_relative_realization_deviation must be >= 0."
        )
    ceiling = _finite_number(
        parameters["maximum_implied_relief"], label=f"{stage}.maximum_implied_relief"
    )
    rate_key = str(parameters["rate_key"])
    failures: list[str] = []
    if receipt.get("rate_key") != rate_key:
        failures.append(
            f"{stage}: the draw reads {receipt.get('rate_key')!r}, not {rate_key!r}."
        )
    build_year = receipt.get("build_year")
    if isinstance(build_year, bool) or not isinstance(build_year, int):
        raise ValueError(f"{stage}.build_year must be an integer, got {build_year!r}.")
    declared = load_uk_take_up_contract().rate(rate_key, build_year)
    rate = _finite_number(receipt.get("rate"), label=f"{stage}.rate")
    if rate != declared:
        failures.append(
            f"{stage}: the draw's rate {rate} is not the contract's {declared} "
            f"for {build_year}."
        )
    unchecked = [
        key
        for key in (
            "population_mass",
            "eligible_mass",
            "target",
            "realized_mass",
            "realization_deviation",
            "implied_relief",
        )
        if receipt.get(key) == UK_WITHHELD
    ]

    def _figure(key: str) -> float | None:
        if key in unchecked:
            return None
        return _finite_number(receipt.get(key), label=f"{stage}.{key}")

    population = _figure("population_mass")
    eligible = _figure("eligible_mass")
    target = _figure("target")
    realized = _figure("realized_mass")
    printed = _figure("realization_deviation")
    if (
        target is not None
        and population is not None
        and not math.isclose(
            target, rate * population, rel_tol=_FLOAT_RELATIVE_TOLERANCE
        )
    ):
        failures.append(
            f"{stage}: target {target} is not the rate times the population aged "
            f"21 to 24 ({rate * population})."
        )
    for key in ("population_support", "eligible_support", "flagged_support"):
        band = receipt.get(key)
        if band != 0 and band not in _SUPPORT_BANDS:
            failures.append(f"{stage}: {key} {band!r} is not a printed band.")
    cap_bound = receipt.get("cap_bound")
    if not isinstance(cap_bound, bool):
        raise ValueError(f"{stage}.cap_bound must be a boolean, got {cap_bound!r}.")
    if target is not None and eligible is not None:
        shortfall = target > 0.0 and eligible <= target
        if cap_bound != shortfall:
            failures.append(
                f"{stage}: cap_bound is {cap_bound} with eligible weight {eligible} "
                f"against target {target}."
            )
    relative: float | None = None
    if realized is not None and target is not None:
        if printed is not None and not math.isclose(
            printed, realized - target, rel_tol=1e-9, abs_tol=1e-6 * max(target, 1.0)
        ):
            failures.append(
                f"{stage}: realization_deviation {printed} is not realized minus "
                f"target ({realized - target})."
            )
        if cap_bound:
            if eligible is not None and not math.isclose(
                realized, eligible, rel_tol=1e-9
            ):
                failures.append(
                    f"{stage}: a cap-bound draw flags every eligible employee, but "
                    f"{realized} of {eligible} is flagged."
                )
        else:
            relative = (
                abs(realized - target) / target if target > 0.0 else float(realized > 0)
            )
            if relative > tolerance:
                failures.append(
                    f"{stage}: the flagged weight is {relative:.4f} of the target "
                    f"away from it, above {tolerance}."
                )
    relief = receipt.get("implied_relief")
    implied: float | None = None
    if relief is None:
        failures.append(
            f"{stage}: the receipt records no implied relief, so the draw cannot be "
            "held below HMRC's apprentice relief."
        )
    elif relief != UK_WITHHELD:
        implied = _finite_number(relief, label=f"{stage}.implied_relief")
        if implied > ceiling:
            failures.append(
                f"{stage}: the flags imply {implied:,.0f} of relief, above the "
                f"{ceiling:,.0f} ceiling."
            )
    details = {
        "rate_key": rate_key,
        "rate": rate,
        "target": target,
        "cap_bound": cap_bound,
        "relative_realization_deviation": relative,
        "implied_relief": implied,
        "maximum_implied_relief": ceiling,
        "figures_unchecked_small_cells": unchecked,
    }
    if failures:
        return _fail(stage, check, failures, details)
    return _pass(stage, check, details)


def _isa_subscriptions_invariants_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """ISA subscriptions keep the rules any population must (microcosm#1415).

    Fails on any record the receipt counts against them (non-finite values, the
    overall and Lifetime ISA limits, ages, holding what is paid into, both parts of
    a both-type subscription, negative amounts) and on person balances that do not
    sum to the household draws within ``maximum_household_sum_error_gbp``. None of
    these depends on the population, so the entry is no population fact check and
    the synthetic smoke build blocks on it (review of microcosm#1421).
    """

    check = "isa_subscriptions_invariants"
    sum_tolerance = _finite_number(
        parameters["maximum_household_sum_error_gbp"],
        label=f"{stage}.maximum_household_sum_error_gbp",
    )
    failures: list[str] = []
    checks = _mapping(evidence.get("checks"), label=f"{stage}.checks")
    violations = _mapping(checks.get("violations"), label=f"{stage}.violations")
    for name, count in violations.items():
        if count != 0:
            failures.append(f"{stage}: {count} record(s) break {name}.")
    split = _mapping(evidence.get("balance_split"), label=f"{stage}.balance_split")
    error = _finite_number(
        split.get("max_abs_household_sum_error_gbp"),
        label=f"{stage}.max_abs_household_sum_error_gbp",
    )
    if error > sum_tolerance:
        failures.append(
            f"{stage}: person balances miss the household draws by up to {error} GBP."
        )
    return GateResult(
        name="stage_health",
        passed=not failures,
        failures=tuple(failures),
        details={"stage": stage, "check": check, "violations": dict(violations)},
    )


def _isa_subscriptions_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """ISA subscriptions meet HMRC's Table 9.4 totals (microcosm#1415).

    Fails on the 2024-25 cash and stocks-and-shares totals off Table 9.4's 2024-25
    amounts, the Lifetime ISA total off Table 9.4's Lifetime ISA amount or the
    both-type split's stocks-and-shares total off its residual by more than
    ``maximum_relative_total_deviation`` (a reported saturation is named, not
    failed). Cells short of eligible adults are named, never failed: the stage
    invents no holders, and how far the result sits from HMRC's tables after
    calibration is a measurement, not a gate (María's ruling of 28 October 2026).
    These are population facts, so a synthetic smoke build records them without
    blocking; the rules any population must keep are
    ``_isa_subscriptions_invariants_gate``.
    """

    check = "isa_subscriptions"
    tolerance = _finite_number(
        parameters["maximum_relative_total_deviation"],
        label=f"{stage}.maximum_relative_total_deviation",
    )
    failures: list[str] = []
    growth = _mapping(evidence.get("growth"), label=f"{stage}.growth")
    named: list[str] = []
    for kind in ("cash", "stocks_and_shares"):
        target = _finite_number(
            growth.get(f"{kind}_target_gbp"), label=f"{stage}.{kind}_target_gbp"
        )
        achieved = _finite_number(
            growth.get(f"{kind}_achieved_gbp"), label=f"{stage}.{kind}_achieved_gbp"
        )
        if target > 0 and abs(achieved - target) / target > tolerance:
            failures.append(
                f"{stage}: 2024-25 {kind} subscriptions miss Table 9.4's 2024-25 "
                "amount."
            )
    both = _mapping(evidence.get("both_type_split"), label=f"{stage}.both_type_split")
    if both.get("saturated"):
        named.append(f"both_type_split:{both['saturated']}")
    else:
        target = _finite_number(
            both.get("stocks_and_shares_target_gbp"), label=f"{stage}.both_target"
        )
        achieved = _finite_number(
            both.get("stocks_and_shares_achieved_gbp"), label=f"{stage}.both_achieved"
        )
        if target > 0 and abs(achieved - target) / target > tolerance:
            failures.append(f"{stage}: the both-type split misses its residual.")
    lifetime = _mapping(evidence.get("lifetime_isa"), label=f"{stage}.lifetime_isa")
    if lifetime.get("saturated"):
        named.append(f"lifetime_isa:{lifetime['saturated']}")
    else:
        target = _finite_number(
            lifetime.get("target_gbp"), label=f"{stage}.lisa_target"
        )
        achieved = _finite_number(
            lifetime.get("achieved_gbp"), label=f"{stage}.lisa_achieved"
        )
        if target > 0 and abs(achieved - target) / target > tolerance:
            failures.append(f"{stage}: Lifetime ISA payments miss Table 9.4's amount.")
    selection = _mapping(evidence.get("selection"), label=f"{stage}.selection")
    # Subscribers the eligible holders could not supply: the stage invents none.
    for isa_type, steps in selection.items():
        for step, receipt in _mapping(steps, label=f"{stage}.{isa_type}").items():
            if not isinstance(receipt, Mapping):
                continue
            unmet = float(receipt.get("shortfall_weight_unmet") or 0.0)
            if unmet > 0:
                named.append(f"selection:{isa_type}:{step}:shortfall_weight_unmet")
    return GateResult(
        name="stage_health",
        passed=not failures,
        failures=tuple(failures),
        details={
            "stage": stage,
            "check": check,
            "saturations_named": named,
            "selection": dict(selection),
            # People who pay in (a positive payment), not every eligible holder:
            # was_lisa already anchors the holders to HMRC's accounts.
            "lifetime_isa_paying_weight": lifetime.get("paying_weight"),
            "lifetime_isa_hmrc_accounts_subscribed": lifetime.get(
                "hmrc_accounts_subscribed"
            ),
        },
    )


def _was_lisa_anchor_gate(
    stage: str,
    evidence: Mapping[str, object],
    parameters: Mapping[str, object],
) -> GateResult:
    """The anchored Lifetime ISA holders reach HMRC's payers (microcosm#1415).

    After the financial-wealth cap the weighted holders at the counted ages lie
    within ``maximum_relative_deviation`` of the anchor's target, or at or above it
    when the drawn holders already were (no shift), and a positive shift removes no
    holder (the same keyed uniforms decide ownership). A target above what the
    largest shift reaches is named, not failed, and so are unshifted holders above
    the target: the anchor only adds holders, so it cannot correct the base
    model's overshoot.
    """

    check = "was_lisa_anchor"
    tolerance = _finite_number(
        parameters["maximum_relative_deviation"],
        label=f"{stage}.maximum_relative_deviation",
    )
    model = _mapping(evidence.get("ownership_model"), label=f"{stage}.ownership_model")
    anchor = _mapping(model.get("anchor"), label=f"{stage}.anchor")
    target = _finite_number(
        anchor.get("target_holders"), label=f"{stage}.target_holders"
    )
    achieved = _finite_number(
        evidence.get("anchored_holders_after_cap"),
        label=f"{stage}.anchored_holders_after_cap",
    )
    shift = _finite_number(anchor.get("logit_shift"), label=f"{stage}.logit_shift")
    saturated = anchor.get("saturated")
    failures: list[str] = []
    named: list[str] = [] if saturated is None else [f"anchor:{saturated}"]
    if target <= 0:
        failures.append(f"{stage}: the anchor's target must be positive.")
    elif shift < 0:
        failures.append(f"{stage}: the anchor's shift must not be negative.")
    elif saturated is not None:
        pass
    elif shift == 0:
        if achieved < target * (1.0 - tolerance):
            failures.append(
                f"{stage}: unshifted holders after the cap fall below the target."
            )
        elif achieved > target * (1.0 + tolerance):
            # The anchor only adds holders, so it cannot bring an overshoot of the
            # base model down: named for the build report, not failed.
            named.append("anchor:unshifted_holders_above_target")
    elif abs(achieved - target) / target > tolerance:
        failures.append(
            f"{stage}: anchored holders after the cap miss the target by more than "
            f"{tolerance:.2%}."
        )
    if anchor.get("removed_holders") != 0:
        failures.append(f"{stage}: the anchor removed holders.")
    return GateResult(
        name="stage_health",
        passed=not failures,
        failures=tuple(failures),
        details={
            "stage": stage,
            "check": check,
            "anchor": dict(anchor),
            "saturations_named": named,
        },
    )
