"""Verify E4 stochastic stage identity stability on an existing UK frame.

Recomputes every E4 column twice from the pure derivations — once in the
frame's row order, once on row-permuted tables — un-permutes by entity id,
and also compares the original-order recomputation against the columns
stored in the artifact. Exit status is nonzero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.frs_brma import (
    UK_BRMA_DECLARED_SEEDS,
    _benunit_regions,
    _enum_name,
    assign_brma_by_cell,
    collapse_benunit_brma_to_household,
    load_brma_count_resource,
)
from microcosm.build.uk_runtime.frs_household_draws import derive_frs_household_draws
from microcosm.build.uk_runtime.frs_person_draws import derive_frs_person_draws
from microcosm.build.uk_runtime.frs_take_up import (
    aggregate_person_reported_to_benunit,
    derive_frs_take_up,
    uc_age_eligible_benunits,
    uk_take_up_population_policy,
)
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    uk_time_period,
)
from microcosm.build.uk_runtime.regional_uprating import (
    load_regional_land_values_resource,
    uprate_household_property_by_region,
)
from microcosm.build.uk_runtime.spi_band_donors import (
    HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR,
    SPI_INCOME_BAND_DONOR_CLONE_INDEX,
)
from microcosm.build.uk_runtime.was_wealth import (
    allocate_student_loan_balance_to_people,
)


def identity_stability_receipt(
    frame,
    *,
    transform: Callable[[object], object],
    columns_by_entity: dict[str, Sequence[str]],
) -> dict[str, object]:
    """Run a transform on original and permuted rows, then compare by id."""

    original = transform(frame)
    permuted_input = _reverse_rows(frame)
    permuted = transform(permuted_input)
    mismatches: dict[str, list[str]] = {}
    for entity, columns in columns_by_entity.items():
        id_column = f"{entity}_id"
        left = original.table(entity).set_index(id_column)
        right = permuted.table(entity).set_index(id_column).reindex(left.index)
        bad = [
            column
            for column in columns
            if not left[column]
            .reset_index(drop=True)
            .equals(right[column].reset_index(drop=True))
        ]
        if bad:
            mismatches[entity] = bad
    return {
        "check": "uk_e4_identity_stability",
        "identical": not mismatches,
        "mismatches": mismatches,
        "columns_by_entity": {
            key: list(value) for key, value in columns_by_entity.items()
        },
    }


def e4_identity_receipt(
    frame,
    *,
    contract,
    count_resource: Mapping[str, object],
    lha_category: Sequence[object],
    permutation_seed: int,
    population_policy=None,
) -> dict[str, object]:
    """Recompute every E4 column in original and permuted row order.

    ``population_policy`` is the engine's working-age bounds the take-up draw
    used (``uk_take_up_population_policy``); it is read from the engine when
    not supplied, so a hermetic caller passes one.

    Two claims are receipted: a row permutation of the input tables changes
    no assignment per entity id, and the original-order recomputation equals
    the columns stored in the artifact (re-derivation identity).
    """

    person = frame.table("person")
    benunit = frame.table("benunit").copy()
    household = frame.table("household")
    if population_policy is None:
        population_policy = uk_take_up_population_policy(uk_time_period(frame))
    if len(lha_category) != len(benunit):
        raise ValueError("LHA_category materialization must align to benunit rows.")
    benunit["LHA_category"] = [_enum_name(value) for value in lha_category]
    benunit["region"] = _benunit_regions(person, household, benunit)

    def recompute(person_t, benunit_t, household_t) -> dict[str, pd.DataFrame]:
        anchors = aggregate_person_reported_to_benunit(person_t, benunit_t)
        take_up = derive_frs_take_up(
            benunit_t,
            anchors=anchors,
            contract=contract,
            uc_age_eligible=uc_age_eligible_benunits(
                person_t, benunit_t, population_policy
            ),
        )
        take_up.index = benunit_t["benunit_id"].to_numpy()
        person_draws = derive_frs_person_draws(person_t, contract=contract)
        person_draws.index = person_t["person_id"].to_numpy()
        household_draws = derive_frs_household_draws(household_t, contract=contract)
        household_draws.index = household_t["household_id"].to_numpy()
        seed = UK_BRMA_DECLARED_SEEDS["brma"]
        benunit_brma = pd.DataFrame(
            {
                "benunit_id": benunit_t["benunit_id"].to_numpy(),
                "brma": assign_brma_by_cell(
                    benunit_t, count_resource=count_resource, seed=seed
                ),
            }
        )
        household_draws["brma"] = collapse_benunit_brma_to_household(
            person_t, benunit_brma, household_t, seed=seed
        )
        return {
            "benunit": take_up,
            "person": person_draws,
            "household": household_draws,
        }

    original = recompute(person, benunit, household)
    rng = np.random.default_rng(permutation_seed)
    permuted = recompute(
        person.iloc[rng.permutation(len(person))].reset_index(drop=True),
        benunit.iloc[rng.permutation(len(benunit))].reset_index(drop=True),
        household.iloc[rng.permutation(len(household))].reset_index(drop=True),
    )

    stored = {
        "person": frame.table("person").set_index("person_id"),
        "benunit": frame.table("benunit").set_index("benunit_id"),
        "household": frame.table("household").set_index("household_id"),
    }
    permutation_mismatches: dict[str, list[str]] = {}
    stored_mismatches: dict[str, list[str]] = {}
    stored_columns_missing: dict[str, list[str]] = {}
    for entity, values in original.items():
        for column in values.columns:
            left = values[column]
            right = permuted[entity][column].reindex(left.index)
            if not np.array_equal(left.to_numpy(), right.to_numpy()):
                permutation_mismatches.setdefault(entity, []).append(column)
            if column not in stored[entity].columns:
                stored_columns_missing.setdefault(entity, []).append(column)
                continue
            kept = stored[entity][column].reindex(left.index)
            if not np.array_equal(
                left.to_numpy(), kept.to_numpy().astype(left.to_numpy().dtype)
            ):
                stored_mismatches.setdefault(entity, []).append(column)
    return {
        "check": "uk_e4_identity_stability",
        "permutation_seed": permutation_seed,
        "identical_under_permutation": not permutation_mismatches,
        "matches_stored_columns": not stored_mismatches and not stored_columns_missing,
        "permutation_mismatches": permutation_mismatches,
        "stored_mismatches": stored_mismatches,
        "stored_columns_missing": stored_columns_missing,
        "columns_by_entity": {
            entity: list(values.columns) for entity, values in original.items()
        },
        "entity_row_counts": {
            entity: int(len(frame.table(entity))) for entity in frame.entities
        },
    }


def e5_identity_receipt(
    frame,
    *,
    regional_resource: Mapping[str, object] | None = None,
    permutation_seed: int,
) -> dict[str, object]:
    """Receipt E5 deterministic layers under row permutation by entity id."""

    resource = regional_resource or load_regional_land_values_resource()

    def recompute(person_t, benunit_t, household_t) -> dict[str, pd.DataFrame]:
        del benunit_t
        household = household_t.copy()
        person = pd.DataFrame(index=person_t["person_id"].to_numpy())
        household_out = pd.DataFrame(index=household_t["household_id"].to_numpy())
        if {"corporate_wealth_excl_isa", "stocks_and_shares_isa"} <= set(
            household.columns
        ):
            household_out["corporate_wealth"] = household[
                "corporate_wealth_excl_isa"
            ].to_numpy(dtype=float) + household["stocks_and_shares_isa"].to_numpy(
                dtype=float
            )
            household["corporate_wealth"] = household_out["corporate_wealth"].to_numpy()
        if {"region", "main_residence_value", "property_wealth"} <= set(
            household.columns
        ):
            uprated = uprate_household_property_by_region(household, resource)
            household_out["main_residence_value"] = uprated[
                "main_residence_value"
            ].to_numpy()
            household_out["property_wealth"] = uprated["property_wealth"].to_numpy()
        if "student_loan_balance" in household.columns:
            person["student_loan_balance"] = allocate_student_loan_balance_to_people(
                household_balances=household["student_loan_balance"],
                household_ids=household["household_id"],
                person=person_t,
            )
        elif {"student_loan_balance", "person_household_id"} <= set(person_t.columns):
            # The stage consumed the household-level balance; the allocation
            # conserves mass, so the household balance is reconstructible as
            # the per-household sum of the person column, and the waterfall
            # can be re-run from it. Sums and proportional splits are float
            # arithmetic, so this layer is compared at fp tolerance rather
            # than bitwise (the math, not the summation order, is the
            # contract).
            canonical = person_t.sort_values("person_id")
            reconstructed = (
                pd.Series(
                    canonical["student_loan_balance"].to_numpy(dtype=float),
                    index=canonical["person_household_id"].to_numpy(),
                )
                .groupby(level=0)
                .sum()
            )
            balances = (
                reconstructed.reindex(household_t["household_id"].to_numpy())
                .fillna(0.0)
                .to_numpy()
            )
            person["student_loan_balance"] = allocate_student_loan_balance_to_people(
                household_balances=pd.Series(balances),
                household_ids=household_t["household_id"],
                person=person_t,
            )
        return {"household": household_out, "person": person}

    # The caller scopes the frame to the population the wealth and uprating
    # stages saw (``_frame_as_stage_saw``): the SPI support rows run before
    # them and are imputed there, while the CGT layers are stacked later.
    person = frame.table("person")
    benunit = frame.table("benunit")
    household = frame.table("household")
    original = recompute(person, benunit, household)
    rng = np.random.default_rng(permutation_seed)
    permuted = recompute(
        person.iloc[rng.permutation(len(person))].reset_index(drop=True),
        benunit.iloc[rng.permutation(len(benunit))].reset_index(drop=True),
        household.iloc[rng.permutation(len(household))].reset_index(drop=True),
    )
    # Permutation comparison is bitwise for the uprating layer (its factor
    # is computed order-independently). The stored-column cross-check re-
    # applies uprating to already-uprated values: the fixed point holds
    # exactly only in exact arithmetic, so one rounding generation of
    # tolerance applies there, as it does to the waterfall re-allocation
    # (sums and proportional splits are float arithmetic).
    tolerances = {("person", "student_loan_balance"): (1e-12, 1e-6)}
    stored_tolerances = {
        ("household", "main_residence_value"): (1e-12, 1e-6),
        ("household", "property_wealth"): (1e-12, 1e-6),
        ("person", "student_loan_balance"): (1e-12, 1e-6),
    }
    mismatches: dict[str, list[str]] = {}
    stored_mismatches: dict[str, list[str]] = {}
    stored_tables = {"household": household, "person": person}
    for entity, values in original.items():
        for column in values.columns:
            rtol, atol = tolerances.get((entity, column), (0.0, 0.0))
            left = values[column]
            right = permuted[entity][column].reindex(left.index)
            if not np.allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=rtol,
                atol=atol,
            ):
                mismatches.setdefault(entity, []).append(column)
            stored_table = stored_tables[entity]
            if column in stored_table.columns:
                stored_rtol, stored_atol = stored_tolerances.get(
                    (entity, column), (rtol, atol)
                )
                if not np.allclose(
                    left.to_numpy(dtype=float),
                    stored_table[column].to_numpy(dtype=float),
                    rtol=stored_rtol,
                    atol=stored_atol,
                ):
                    stored_mismatches.setdefault(entity, []).append(column)
    return {
        "check": "uk_e5_identity_stability",
        "permutation_seed": permutation_seed,
        "identical_under_permutation": not mismatches,
        "permutation_mismatches": mismatches,
        "matches_stored_columns": not stored_mismatches,
        "stored_column_mismatches": stored_mismatches,
        "tolerance_policy": (
            "permutation: bitwise for the regional-uprating rewrite "
            "(order-independent factor), rtol 1e-12 / atol 1e-6 GBP for the "
            "waterfall re-allocation; stored-column cross-check: rtol 1e-12 "
            "/ atol 1e-6 GBP for both layers (re-applying uprating to the "
            "uprated fixed point and re-summing allocations each cost one "
            "float rounding generation); corporate_wealth fold not "
            "reconstructible from the artifact (components consumed) - "
            "unit-tested only"
        ),
        "columns_by_entity": {
            entity: list(values.columns) for entity, values in original.items()
        },
        "qrf_draw_columns_scope": (
            "excluded: seeded-stream QRF draws are covered by twin-build "
            "determinism rather than identity-keyed row permutation"
        ),
    }


def e6_identity_receipt(
    frame,
    *,
    permutation_seed: int,
) -> dict[str, object]:
    """Receipt E6 deterministic layers under row permutation by entity id.

    Covered: the domestic-energy fold (elec + gas), the rail_usage ratio,
    petrol/diesel zeroing idempotence for non-fuel households, and the NHS
    age-gender person allocation recomputed from the committed resource.
    The production contract is stage-time-age-derived: ``age_tail`` now runs
    immediately after ``frs_spine``, so ``etb_services`` allocates NHS use on
    the disaggregated age surface. Stored NHS columns are therefore checked
    against final age without reconstructing the former top code.
    The QRF chain draws and the NEED raking outcome are covered by
    twin-build determinism and the aggregate_admin NEED-margin receipt
    respectively (raking inputs are consumed by the stage and are not
    reconstructible from the artifact).
    """

    from microcosm.build.uk_runtime.etb_services import (
        allocate_nhs_by_age_gender,
        load_etb_services_anchors,
        rail_fare_index_denominator_key,
    )

    rail_fare_index = float(
        load_etb_services_anchors()[rail_fare_index_denominator_key()]["value"]
    )

    def recompute(person_t, benunit_t, household_t) -> dict[str, pd.DataFrame]:
        del benunit_t
        household = household_t.copy()
        household_out = pd.DataFrame(index=household_t["household_id"].to_numpy())
        person_out = pd.DataFrame(index=person_t["person_id"].to_numpy())
        if {"electricity_consumption", "gas_consumption"} <= set(household.columns):
            household_out["domestic_energy_consumption"] = household[
                "electricity_consumption"
            ].to_numpy(dtype=float) + household["gas_consumption"].to_numpy(dtype=float)
        if "rail_subsidy_spending" in household.columns:
            household_out["rail_usage"] = (
                household["rail_subsidy_spending"].to_numpy(dtype=float)
                / rail_fare_index
            )
        if {"has_fuel_consumption", "petrol_spending", "diesel_spending"} <= set(
            household.columns
        ):
            no_fuel = household["has_fuel_consumption"].to_numpy(dtype=float) == 0.0
            for column in ("petrol_spending", "diesel_spending"):
                household_out[column] = np.where(
                    no_fuel, 0.0, household[column].to_numpy(dtype=float)
                )
        if {"age", "gender"} <= set(person_t.columns):
            nhs_person = person_t.copy()
            nhs_person["age"] = (
                pd.to_numeric(nhs_person["age"], errors="coerce")
                .fillna(0)
                .to_numpy(dtype=float)
            )
            nhs = allocate_nhs_by_age_gender(
                nhs_person,
                household_weights=household["household_weight"].to_numpy(dtype=float),
                household=household,
                nhs_table=None,
            )
            for column in nhs.columns:
                person_out[column] = nhs[column].to_numpy(dtype=float)
        return {"household": household_out, "person": person_out}

    person = frame.table("person")
    benunit = frame.table("benunit")
    household = frame.table("household").copy()
    household["household_weight"] = frame.weights_for("household").values
    # Scope to the rows the E6 stages actually saw, and restore the grossing
    # scale they saw them at. Every stacking stage that runs after E6 copies
    # its source row's consumption/services values onto the new rows, so those
    # rows are the later stage's receipt surface, not E6's; and every later
    # stage that redistributes mass leaves these rows carrying a fraction of
    # the weight E6 normalized against. The SPI support rows (and their band
    # donors) are stacked before E6 and imputed there, so they stay in scope at
    # their allocated mass. The CGT support split and the clone are stacked
    # after: their created rows (copies and clones) are dropped and their mass
    # is folded back - the clone's by pair (equal halves before the #970
    # anchor, pair-conserving afterwards) and the split's by family onto the
    # root - so the surviving rows carry the pre-split weights E6 saw.
    later_flags = _flags_stacked_after(_E6_FIRST_STAGE, household)
    if later_flags:
        household["household_weight"] = _pre_split_household_weights(frame)
        spine_mask = ~household[later_flags].astype(bool).any(axis=1)
        household = household.loc[spine_mask].reset_index(drop=True)
        spine_household_ids = set(household["household_id"].tolist())
        person = person.loc[
            person["person_household_id"].isin(spine_household_ids)
        ].reset_index(drop=True)
        applied = tuple(
            _MASS_STAGE_BY_FLAG[flag]
            for flag in later_flags
            if flag in _MASS_STAGE_BY_FLAG
        )
        household["household_weight"] = household["household_weight"].to_numpy(
            dtype=float
        ) / _stage_time_weight_divisor(after_stages=applied)
    original = recompute(person, benunit, household)
    rng = np.random.default_rng(permutation_seed)
    permuted = recompute(
        person.iloc[rng.permutation(len(person))].reset_index(drop=True),
        benunit.iloc[rng.permutation(len(benunit))].reset_index(drop=True),
        household.iloc[rng.permutation(len(household))].reset_index(drop=True),
    )
    # The fold, ratio, and zeroing layers are order-independent elementwise
    # arithmetic on stored columns: bitwise under permutation and against
    # the store. The NHS layer's cell normalization sums weights per
    # (age-band, gender) cell, so permutation changes float summation
    # order: one rounding generation of tolerance applies, and the same
    # tolerance covers the stored cross-check.
    nhs_columns = tuple(original["person"].columns)
    tolerances = {("person", column): (1e-12, 1e-9) for column in nhs_columns}
    stored_tolerances = dict(tolerances)
    mismatches: dict[str, list[str]] = {}
    stored_mismatches: dict[str, list[str]] = {}
    stored_tables = {"household": household, "person": person}
    for entity, values in original.items():
        for column in values.columns:
            rtol, atol = tolerances.get((entity, column), (0.0, 0.0))
            left = values[column]
            right = permuted[entity][column].reindex(left.index)
            if not np.allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=rtol,
                atol=atol,
            ):
                mismatches.setdefault(entity, []).append(column)
            stored_table = stored_tables[entity]
            if column in stored_table.columns:
                stored_rtol, stored_atol = stored_tolerances.get(
                    (entity, column), (rtol, atol)
                )
                if not np.allclose(
                    left.to_numpy(dtype=float),
                    stored_table[column].to_numpy(dtype=float),
                    rtol=stored_rtol,
                    atol=stored_atol,
                ):
                    stored_mismatches.setdefault(entity, []).append(column)
    return {
        "check": "uk_e6_identity_stability",
        "permutation_seed": permutation_seed,
        "nhs_age_basis": "stage_time_disaggregated",
        "identical_under_permutation": not mismatches,
        "permutation_mismatches": mismatches,
        "matches_stored_columns": not stored_mismatches,
        "stored_column_mismatches": stored_mismatches,
        "tolerance_policy": (
            "permutation and stored-column: bitwise for the domestic-energy "
            "fold, the rail_usage ratio, and petrol/diesel zeroing "
            "(order-independent elementwise arithmetic); rtol 1e-12 / "
            "atol 1e-9 for the NHS allocation (cell weight sums cost one "
            "float rounding generation under reordering)"
        ),
        "columns_by_entity": {
            entity: list(values.columns) for entity, values in original.items()
        },
        "qrf_draw_columns_scope": (
            "excluded: seeded-stream QRF draws are covered by twin-build "
            "determinism; the NEED raking outcome is covered by the "
            "aggregate_admin NEED-margin receipt (raking inputs are "
            "consumed by the stage and not reconstructible from the "
            "artifact)"
        ),
    }


def e7_identity_receipt(
    frame,
    *,
    permutation_seed: int,
) -> dict[str, object]:
    """Receipt E7 deterministic layers under row permutation by entity id.

    Covered: the support-channel labelling the SPI stack introduces — each
    entity's channel and clone index, the composite source key, and the
    propagation of a household's channel down to its persons and benefit
    units. These are pure functions of the synthetic flag and the entity ids,
    so they are bitwise both under permutation and against the store, and they
    are the layer E7 actually contributes to the artifact.

    Not covered here, and deliberately so:

    * the stage-1/stage-2 QRF fits, which twin-build
      determinism covers — the e6 and e8 precedent for QRF surfaces;
    * the ``employer_pension_contributions = 3 * employee_pension_contributions``
      derive, which is a genuine E7 deterministic layer but which E8's
      salary_sacrifice rewrites in place afterwards. Measured on the E8
      roster the relation survives on only 95.9% of survey-channel persons,
      so the stage-time relation is not reconstructible from the final
      artifact. That is the #721 rewrites-provenance class, not a defect, and
      asserting it here would fail for the wrong reason.
    """

    def recompute(person_t, benunit_t, household_t) -> dict[str, pd.DataFrame]:
        household_out = pd.DataFrame(index=household_t["household_id"].to_numpy())
        person_out = pd.DataFrame(index=person_t["person_id"].to_numpy())
        benunit_out = pd.DataFrame(index=benunit_t["benunit_id"].to_numpy())

        # An artifact without the synthetic flag carries no E7 layer, so
        # there is nothing this receipt could certify about it. Returning an
        # empty receipt here would report a vacuous pass — the mismatch loops
        # never run over an empty recomputation — which is the one outcome a
        # receipt must never produce. Refuse instead.
        if "household_is_spi_synthetic" not in household_t.columns:
            raise ValueError(
                "e7 identity receipt: the artifact carries no "
                "household_is_spi_synthetic column, so the E7 support-channel "
                "layer is absent and there is nothing to receipt. Run the "
                "check against an artifact built with the SPI channel, or "
                "drop --check e7 for this artifact."
            )
        synthetic = household_t["household_is_spi_synthetic"].astype(bool).to_numpy()
        channel = np.where(synthetic, "spi", "frs")
        household_out["household_support_channel"] = channel
        # The SPI support copy sits at clone index 1 and the income band donors,
        # which carry the same synthetic flag, at their own index; a frame built
        # before the donor stage carries no donor flag and no donors.
        donor = (
            household_t[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR].astype(bool).to_numpy()
            if HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR in household_t.columns
            else np.zeros(len(household_t), dtype=bool)
        )
        household_out["household_support_clone_index"] = np.where(
            synthetic, np.where(donor, SPI_INCOME_BAND_DONOR_CLONE_INDEX, 1), 0
        )

        missing_keys = {"source_year", "source_household_id"} - set(household_t.columns)
        if missing_keys:
            raise ValueError(
                "e7 identity receipt: the artifact carries the synthetic flag "
                f"but not {sorted(missing_keys)}; the source key cannot be "
                "recomputed, and skipping it would silently shrink the "
                "receipt's coverage."
            )
        household_out["source_household_key"] = [
            f"{int(year)}:{int(source)}"
            for year, source in zip(
                household_t["source_year"].to_numpy(),
                household_t["source_household_id"].to_numpy(),
                strict=True,
            )
        ]

        # The channel is a household property; persons and benefit units
        # inherit it through membership, never redraw it.
        by_household = pd.Series(channel, index=household_t["household_id"].to_numpy())
        person_channel = (
            person_t["person_household_id"].map(by_household).to_numpy(dtype=object)
        )
        person_out["person_support_channel"] = person_channel
        by_benunit = pd.Series(
            person_channel, index=person_t["person_benunit_id"].to_numpy()
        )
        by_benunit = by_benunit[~by_benunit.index.duplicated(keep="first")]
        benunit_out["benunit_support_channel"] = (
            benunit_t["benunit_id"].map(by_benunit).to_numpy(dtype=object)
        )
        return {
            "household": household_out,
            "person": person_out,
            "benunit": benunit_out,
        }

    person = frame.table("person")
    benunit = frame.table("benunit")
    household = frame.table("household")

    original = recompute(person, benunit, household)
    rng = np.random.default_rng(permutation_seed)
    permuted = recompute(
        person.iloc[rng.permutation(len(person))].reset_index(drop=True),
        benunit.iloc[rng.permutation(len(benunit))].reset_index(drop=True),
        household.iloc[rng.permutation(len(household))].reset_index(drop=True),
    )

    stored_tables = {
        "person": person.set_index("person_id"),
        "benunit": benunit.set_index("benunit_id"),
        "household": household.set_index("household_id"),
    }
    mismatches: dict[str, list[str]] = {}
    stored_mismatches: dict[str, list[str]] = {}
    # Labels and integer indices: bitwise on both surfaces, no tolerance.
    for entity, values in original.items():
        for column in values.columns:
            left = values[column]
            right = permuted[entity][column].reindex(left.index)
            if not np.array_equal(
                left.to_numpy().astype(str), right.to_numpy().astype(str)
            ):
                mismatches.setdefault(entity, []).append(column)
            stored_table = stored_tables[entity]
            if column not in stored_table.columns:
                # The store not carrying a column this receipt certifies is a
                # failed comparison, not a narrower one.
                stored_mismatches.setdefault(entity, []).append(column)
            else:
                kept = stored_table[column].reindex(left.index)
                if not np.array_equal(
                    left.to_numpy().astype(str), kept.to_numpy().astype(str)
                ):
                    stored_mismatches.setdefault(entity, []).append(column)
    return {
        "check": "uk_e7_identity_stability",
        "permutation_seed": permutation_seed,
        "columns_compared": {
            entity: sorted(values.columns) for entity, values in original.items()
        },
        "identical_under_permutation": not mismatches,
        "permutation_mismatches": mismatches,
        "matches_stored_columns": not stored_mismatches,
        "stored_column_mismatches": stored_mismatches,
        "tolerance_policy": (
            "bitwise on both surfaces: the support channel, clone index and "
            "source key are labels and integer indices, so no float "
            "tolerance applies"
        ),
        "columns_by_entity": {
            entity: list(values.columns) for entity, values in original.items()
        },
        "qrf_draw_columns_scope": (
            "excluded: the stage-1/stage-2 QRF fits are covered by twin-build "
            "determinism (the e6 and e8 precedent)"
        ),
        "rewritten_layer_scope": (
            "excluded: employer_pension_contributions = 3 x "
            "employee_pension_contributions is an E7 derive, but E8 "
            "salary_sacrifice rewrites the multiplicand in place afterwards, "
            "so the stage-time relation is not reconstructible from the "
            "final artifact (#721 rewrites-provenance class)"
        ),
    }


def _e8_clone_pairs(
    frame, problems: dict[str, object], *, parameters=None
) -> dict[str, object]:
    """Check the clone/original pairs, before or after the #970 anchor.

    Every pre-clone household is paired, the support split's copies included
    (they are pre-clone rows with clones of their own). Before the anchor
    both halves carry the same weight. After it (the frame's mass log
    carries the anchor's record) each pair still sums to its pre-clone
    weight, the clone side never exceeds the original, and the anchor is
    recomputed from the reconstructed pre-anchor state (both halves at half
    the pair sum, the clone stage's split up to rounding) in original and
    reversed person order and compared with the stored weights.
    ``parameters`` are the policy parameters the anchor's proxy reads; they
    are read from the engine when not supplied.
    """

    from microcosm.build.uk_runtime.cgt_imputation import uk_cgt_policy_parameters
    from microcosm.build.uk_runtime.cgt_structure import (
        CGT_ANCHOR_MASS_CHANGE_REASON,
        anchor_cgt_incidence,
        load_advani_summers_distribution,
        pair_clone_households,
    )
    from microcosm.build.uk_runtime.national_frame import (
        uk_household_weight_kind,
        uk_national_frame,
    )

    person = frame.table("person")
    benunit = frame.table("benunit")
    household = frame.table("household")
    weights = np.asarray(frame.weights_for("household").values, dtype=float)
    anchor_records = [
        index
        for index, record in enumerate(frame.mass_log)
        if record.reason == CGT_ANCHOR_MASS_CHANGE_REASON
    ]
    receipt: dict[str, object] = {"anchored": bool(anchor_records)}
    try:
        clone_positions, original_positions, _ = pair_clone_households(
            person, benunit, household
        )
    except ValueError as error:
        problems["clone_pairing"] = str(error)
        return receipt
    left = weights[original_positions]
    right = weights[clone_positions]
    receipt["pairs"] = int(len(clone_positions))
    if not anchor_records:
        if not np.allclose(left, right, rtol=1e-12, atol=1e-6):
            problems["clone_pair_weights"] = int(
                (~np.isclose(left, right, rtol=1e-12, atol=1e-6)).sum()
            )
        if not np.isclose(left.sum(), right.sum(), rtol=1e-12, atol=1e-6):
            problems["clone_half_masses"] = [float(left.sum()), float(right.sum())]
        return receipt
    exceeds = right > left * (1.0 + 1e-12) + 1e-6
    if exceeds.any():
        problems["clone_exceeds_original"] = int(exceeds.sum())
    pre = weights.copy()
    halves = 0.5 * (left + right)
    pre[original_positions] = halves
    pre[clone_positions] = halves
    pre_frame = uk_national_frame(
        person=person.copy(),
        benunit=benunit.copy(),
        household=household.copy(),
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=pre,
        mass_log=tuple(frame.mass_log[: anchor_records[0]]),
    )
    distribution = load_advani_summers_distribution()
    if parameters is None:
        parameters = uk_cgt_policy_parameters(uk_time_period(frame))
    recomputed = anchor_cgt_incidence(
        pre_frame, distribution=distribution, parameters=parameters
    )
    stored = pd.Series(weights, index=household["household_id"].to_numpy())

    def by_household_id(result) -> np.ndarray:
        return (
            pd.Series(
                result.frame.weights_for("household").values,
                index=result.frame.table("household")["household_id"].to_numpy(),
            )
            .reindex(stored.index)
            .to_numpy(dtype=float)
        )

    again = by_household_id(recomputed)
    close = np.isclose(again, stored.to_numpy(), rtol=1e-9, atol=1e-6)
    receipt["anchor_max_abs_weight_diff"] = float(np.abs(again - stored).max())
    receipt["anchor_targets"] = dict(recomputed.targets)
    receipt["anchor_after"] = dict(recomputed.after)
    if not close.all():
        problems["anchor_recompute"] = int((~close).sum())
    permuted = anchor_cgt_incidence(
        _reverse_rows(pre_frame), distribution=distribution, parameters=parameters
    )
    if not np.array_equal(by_household_id(permuted), again):
        problems["anchor_permutation"] = True
    return receipt


def _e8_support_split(
    frame,
    problems: dict[str, object],
    *,
    distribution,
    parameters,
    permutation_seed: int,
) -> dict[str, object]:
    """Recompute the CGT support split from the folded pre-split frame.

    The split (microcosm#1045) is deterministic - no draw, no seed - so it is
    reconstructible from the artifact alone: every clone folded onto its
    original and every support copy onto its root gives the pre-split
    household table at the weights the stage saw, and the stage's own helpers
    rerun its rule on that table from the vendored Table 3 joint
    (``distribution``), the stored income components under the policy
    ``parameters``, and the stored wealth columns. The stored layer must
    agree: the households the rule divides are exactly the roots of the
    flagged copies (the pre-clone rows carrying ``cgt_support_copies`` above
    one), each family's count is ``ceil(w / maximum_copy_weight)``, every
    member of a family carries ``w / n`` at stage time (its clone pair sum,
    since the clone runs afterwards), the copies of each root are indexed
    ``1..n-1`` without gaps and carry the root's count, no count is below
    one, and the stage's mass record conserves the total. The recompute is
    repeated on permuted person and household tables and must reproduce the
    divided set and the counts exactly. A selected household light enough
    for a single copy leaves no trace in the artifact, so only families are
    compared. Problems land under ``support_split_*`` keys.
    """

    from microcosm.build.uk_runtime.cgt_support import (
        CGT_SUPPORT_CLONE_SPLIT_FACTOR,
        CGT_SUPPORT_COPIES_COLUMN,
        CGT_SUPPORT_COPY_INDEX_COLUMN,
        CGT_SUPPORT_HEADROOM,
        CGT_SUPPORT_MASS_CHANGE_REASON,
        CGT_SUPPORT_MAXIMUM_COPY_WEIGHT,
        CGT_SUPPORT_MINIMUM_GAIN_BAND_LOWER,
        HOUSEHOLD_IS_CGT_SUPPORT_COPY,
        cgt_support_copy_counts,
        cgt_support_household_wealth,
        cgt_support_income_band,
        cgt_support_mass_by_income_band,
        select_cgt_support_households,
    )

    person = frame.table("person")
    benunit = frame.table("benunit")
    household = frame.table("household")
    missing = [
        column
        for column in (HOUSEHOLD_IS_CGT_SUPPORT_COPY, CGT_SUPPORT_COPIES_COLUMN)
        if column not in household.columns
    ]
    if missing:
        # An artifact without the split layer cannot be receipted for it, and
        # skipping the block would report a pass over an unchecked layer (the
        # E7 rule). Refuse instead.
        raise ValueError(
            "e8 identity receipt: the artifact carries no "
            f"{missing} column(s), so the CGT support-split layer is absent "
            "and there is nothing to receipt. Receipt the artifact with the "
            "tool at the commit that built it."
        )
    receipt: dict[str, object] = {}
    mass_records = [
        record
        for record in frame.mass_log
        if record.reason == CGT_SUPPORT_MASS_CHANGE_REASON
    ]
    if not mass_records:
        problems["support_split_mass_record"] = "missing"
    else:
        record = mass_records[-1]
        receipt["mass"] = {
            "old_total": float(record.old_total),
            "new_total": float(record.new_total),
        }
        if not np.isclose(record.new_total, record.old_total, rtol=1e-9, atol=0.0):
            problems["support_split_mass_record"] = [
                float(record.old_total),
                float(record.new_total),
            ]
    try:
        lineage = _support_copy_lineage(person, benunit, household)
    except ValueError as error:
        problems["support_split_lineage"] = str(error)
        return receipt
    household_ids = lineage.household_ids
    pre_split = lineage.pre_split
    stored_copies = (
        pd.to_numeric(household[CGT_SUPPORT_COPIES_COLUMN], errors="raise")
        .astype("int64")
        .to_numpy()
    )
    receipt["id_multiplier"] = int(lineage.multiplier)
    receipt["pre_split_households"] = int(pre_split.sum())
    receipt["copies"] = int(len(lineage.copy_positions))
    # The explicit copy index (microcosm#1045 review): present on artifacts
    # built since it was added, checked against the id scheme by the lineage
    # above, and recorded so a receipt says which lineage it read.
    receipt["copy_index_column_stored"] = bool(
        CGT_SUPPORT_COPY_INDEX_COLUMN in household.columns
    )

    # Flag and count consistency of the stored layer: every copy carries its
    # root's count and the root's count exceeds one; each family's copies are
    # indexed 1..n-1 exactly; no count is below one.
    root_positions = lineage.root_positions
    root_copies = stored_copies[root_positions]
    copy_copies = stored_copies[lineage.copy_positions]
    flag_problems: dict[str, int] = {}
    disagreeing = int(((copy_copies != root_copies) | (root_copies < 2)).sum())
    if disagreeing:
        flag_problems["copies_disagreeing_with_root"] = disagreeing
    copy_index_by_root: dict[int, list[int]] = {}
    for root, index in zip(
        root_positions.tolist(), lineage.copy_index.tolist(), strict=True
    ):
        copy_index_by_root.setdefault(root, []).append(index)
    family_positions = np.flatnonzero(pre_split & (stored_copies > 1))
    gapped = 0
    for position in family_positions.tolist():
        expected_indices = list(range(1, int(stored_copies[position])))
        if sorted(copy_index_by_root.get(position, [])) != expected_indices:
            gapped += 1
    if gapped:
        flag_problems["families_with_missing_or_extra_copies"] = gapped
    below_one = int((stored_copies < 1).sum())
    if below_one:
        flag_problems["counts_below_one"] = below_one
    if flag_problems:
        problems["support_split_flags"] = flag_problems
    receipt["families"] = int(len(family_positions))

    # Fold: clones onto originals (pair sums), then copies onto roots.
    pre_clone = _pre_clone_household_weights(frame)
    folded = pre_clone.copy()
    np.add.at(folded, root_positions, pre_clone[lineage.copy_positions])

    # Each family member's stage-time weight (its clone pair sum) is w / n.
    expected = np.full(len(household), np.nan)
    expected[family_positions] = folded[family_positions] / np.maximum(
        stored_copies[family_positions], 1
    )
    expected[lineage.copy_positions] = folded[root_positions] / np.maximum(
        stored_copies[root_positions], 1
    )
    members = np.concatenate([family_positions, lineage.copy_positions])
    if members.size:
        difference = np.abs(pre_clone[members] - expected[members])
        receipt["max_abs_family_weight_diff"] = float(difference.max())
        off = ~np.isclose(pre_clone[members], expected[members], rtol=1e-9, atol=1e-6)
        if off.any():
            problems["support_split_family_weights"] = int(off.sum())

    # The rule rerun on the folded pre-split table with the stage's helpers.
    mass_rows = cgt_support_mass_by_income_band(
        distribution,
        clone_split_factor=CGT_SUPPORT_CLONE_SPLIT_FACTOR,
        headroom=CGT_SUPPORT_HEADROOM,
        minimum_gain_band_lower=CGT_SUPPORT_MINIMUM_GAIN_BAND_LOWER,
    )
    support_by_band = {
        int(row["income_lower_bound"]): float(row["support_mass"]) for row in mass_rows
    }
    pre_household = household.loc[pre_split].reset_index(drop=True)
    pre_ids = household_ids[pre_split]
    pre_weights = folded[pre_split]
    pre_person = person.loc[
        person["person_household_id"].isin(set(pre_ids.tolist()))
    ].reset_index(drop=True)

    def recompute(person_t, household_t, weights_t):
        ids_t = (
            pd.to_numeric(household_t["household_id"], errors="raise")
            .astype("int64")
            .to_numpy()
        )
        wealth = cgt_support_household_wealth(household_t)
        band = cgt_support_income_band(
            person_t, household_ids=ids_t, parameters=parameters
        )
        selected, band_rows = select_cgt_support_households(
            ids_t, weights_t, band, wealth, support_by_band
        )
        copies = np.ones(len(ids_t), dtype="int64")
        copies[selected] = cgt_support_copy_counts(
            weights_t[selected], CGT_SUPPORT_MAXIMUM_COPY_WEIGHT
        )
        return (
            pd.DataFrame({"selected": selected, "copies": copies}, index=ids_t),
            band_rows,
        )

    original, band_rows = recompute(pre_person, pre_household, pre_weights)
    receipt["bands"] = [dict(row) for row in band_rows]
    receipt["support_mass"] = float(sum(support_by_band.values()))
    receipt["households_selected"] = int(original["selected"].sum())
    receipt["copies_recomputed"] = int((original["copies"] - 1).sum())

    recomputed_family = original["selected"] & (original["copies"] > 1)
    stored_family = pd.Series(stored_copies[pre_split] > 1, index=pre_ids)
    stored_count = pd.Series(stored_copies[pre_split], index=pre_ids)
    missing_families = int((stored_family & ~recomputed_family).sum())
    extra_families = int((recomputed_family & ~stored_family).sum())
    if missing_families or extra_families:
        problems["support_split_selection_stored"] = {
            "missing": missing_families,
            "extra": extra_families,
        }
    agreed = (recomputed_family & stored_family).to_numpy()
    count_off = int(
        (original["copies"].to_numpy()[agreed] != stored_count.to_numpy()[agreed]).sum()
    )
    if count_off:
        problems["support_split_copies_stored"] = count_off

    rng = np.random.default_rng(permutation_seed)
    household_order = rng.permutation(len(pre_household))
    permuted, _ = recompute(
        pre_person.iloc[rng.permutation(len(pre_person))].reset_index(drop=True),
        pre_household.iloc[household_order].reset_index(drop=True),
        pre_weights[household_order],
    )
    permuted = permuted.reindex(original.index)
    if not (
        np.array_equal(original["selected"].to_numpy(), permuted["selected"].to_numpy())
        and np.array_equal(original["copies"].to_numpy(), permuted["copies"].to_numpy())
    ):
        problems["support_split_selection_permutation"] = True
    return receipt


def _e8_residential_split(
    frame,
    problems: dict[str, object],
    *,
    parameters,
    facts,
    permutation_seed: int,
) -> dict[str, object]:
    """Recompute the residential split from the folded pre-split frame.

    The split (microcosm#1063) is deterministic - no draw, no seed - so it is
    reconstructible from the artifact alone: every arm folded onto the arm
    that kept its household's ids gives the household table at the weights
    the stage saw, and the stage's own function rerun on that table from the
    vendored Table 8 rows and the policy parameters must reproduce the stored
    arms (ids, indices and weights), the stored probabilities and the stored
    residential gains, in original and permuted person order. The stored
    layer must also agree with itself (every arm has a source, flag and index
    agree, the mass record conserves the total at declared factor one).
    Problems land under ``residential_split_*`` keys.
    """

    from microcosm.build.uk_runtime.cgt_asset_type import (
        CGT_RESIDENTIAL_GAINS_COLUMN,
    )
    from microcosm.build.uk_runtime.cgt_residential_split import (
        CGT_RESIDENTIAL_CLONE_INDEX_COLUMN,
        CGT_RESIDENTIAL_PROBABILITY_COLUMN,
        CGT_RESIDENTIAL_SPLIT_MASS_CHANGE_REASON,
        HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE,
        split_cgt_residential_households,
    )
    from microcosm.build.uk_runtime.national_frame import (
        uk_household_weight_kind,
        uk_national_frame,
    )

    household = frame.table("household")
    person = frame.table("person")
    if HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE not in household.columns:
        raise ValueError(
            "e8 identity receipt: the artifact carries no "
            f"{HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE!r} column, so the CGT "
            "residential split layer is absent and there is nothing to receipt. "
            "Receipt the artifact with the tool at the commit that built it."
        )
    receipt: dict[str, object] = {}
    mass_records = [
        record
        for record in frame.mass_log
        if record.reason == CGT_RESIDENTIAL_SPLIT_MASS_CHANGE_REASON
    ]
    if not mass_records:
        problems["residential_split_mass_record"] = "missing"
    else:
        record = mass_records[-1]
        receipt["mass"] = {
            "old_total": float(record.old_total),
            "new_total": float(record.new_total),
            "declared_factor": record.declared_factor,
        }
        if record.declared_factor != 1.0 or not np.isclose(
            record.new_total, record.old_total, rtol=1e-9, atol=0.0
        ):
            problems["residential_split_mass_record"] = [
                float(record.old_total),
                float(record.new_total),
                record.declared_factor,
            ]
    try:
        lineage = _residential_arm_lineage(person, frame.table("benunit"), household)
    except ValueError as error:
        problems["residential_split_lineage"] = str(error)
        return receipt
    receipt["id_multiplier"] = int(lineage.multiplier)
    receipt["arms"] = int(len(lineage.arm_positions))
    receipt["households_split"] = int(np.unique(lineage.source_positions).size)
    receipt["largest_arm_index"] = (
        int(lineage.arm_index.max()) if lineage.arm_index.size else 0
    )

    # The frame the split saw: arms folded onto their sources and dropped,
    # the stage's own columns removed.
    pre = _drop_stacked_layers(frame, [HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE])
    pre_person = pre.table("person").drop(
        columns=[CGT_RESIDENTIAL_PROBABILITY_COLUMN, CGT_RESIDENTIAL_GAINS_COLUMN]
    )
    pre_household = pre.table("household").drop(
        columns=[HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE, CGT_RESIDENTIAL_CLONE_INDEX_COLUMN]
    )

    def pre_frame(person_table: pd.DataFrame):
        return uk_national_frame(
            person=person_table,
            benunit=pre.table("benunit"),
            household=pre_household,
            time_period=uk_time_period(frame),
            weight_kind=uk_household_weight_kind(frame),
            household_weights=pre.weights_for("household").values,
            mass_log=pre.mass_log,
        )

    def recompute(person_table: pd.DataFrame):
        result = split_cgt_residential_households(
            pre_frame(person_table), facts=facts, parameters=parameters
        )
        table = result.frame.table("household")
        weights = pd.Series(
            np.asarray(result.frame.weights_for("household").values, dtype=float),
            index=pd.to_numeric(table["household_id"]).astype("int64").to_numpy(),
        )
        index = pd.Series(
            table[CGT_RESIDENTIAL_CLONE_INDEX_COLUMN].to_numpy(dtype="int64"),
            index=weights.index,
        )
        people = result.frame.table("person")
        ids = pd.to_numeric(people["person_id"]).astype("int64").to_numpy()
        probability = pd.Series(
            people[CGT_RESIDENTIAL_PROBABILITY_COLUMN].to_numpy(dtype=float), index=ids
        )
        gains = pd.Series(
            people[CGT_RESIDENTIAL_GAINS_COLUMN].to_numpy(dtype=float), index=ids
        )
        return weights, index, probability, gains, result.evidence()

    weights, index, probability, gains, evidence = recompute(pre_person)
    receipt["identities"] = dict(evidence["identities"])
    receipt["arm_weights"] = dict(evidence["arm_weights"])
    receipt["concentration"] = dict(evidence["concentration"])
    stored_weights = pd.Series(
        np.asarray(frame.weights_for("household").values, dtype=float),
        index=lineage.household_ids,
    )
    stored_index = pd.Series(
        household[CGT_RESIDENTIAL_CLONE_INDEX_COLUMN].to_numpy(dtype="int64"),
        index=lineage.household_ids,
    )
    missing = int((~stored_weights.index.isin(weights.index)).sum())
    extra = int((~weights.index.isin(stored_weights.index)).sum())
    if missing or extra:
        problems["residential_split_arms_stored"] = {
            "missing": missing,
            "extra": extra,
        }
    else:
        aligned = weights.reindex(stored_weights.index)
        receipt["max_abs_weight_diff"] = float(
            np.abs(aligned.to_numpy() - stored_weights.to_numpy()).max()
        )
        off = ~np.isclose(
            aligned.to_numpy(), stored_weights.to_numpy(), rtol=1e-9, atol=1e-6
        )
        if off.any():
            problems["residential_split_weights_stored"] = int(off.sum())
        if not index.reindex(stored_index.index).equals(stored_index):
            problems["residential_split_index_stored"] = True
    person_ids = pd.to_numeric(person["person_id"]).astype("int64").to_numpy()
    stored_probability = pd.Series(
        person[CGT_RESIDENTIAL_PROBABILITY_COLUMN].to_numpy(dtype=float),
        index=person_ids,
    )
    stored_gains = pd.Series(
        person[CGT_RESIDENTIAL_GAINS_COLUMN].to_numpy(dtype=float), index=person_ids
    )
    if not stored_probability.index.isin(probability.index).all():
        problems["residential_split_people_stored"] = True
    else:
        if not np.allclose(
            probability.reindex(stored_probability.index).to_numpy(),
            stored_probability.to_numpy(),
            rtol=1e-9,
            atol=0.0,
        ):
            problems["residential_split_probability_stored"] = True
        if not np.array_equal(
            gains.reindex(stored_gains.index).to_numpy(), stored_gains.to_numpy()
        ):
            problems["residential_split_gains_stored"] = True

    rng = np.random.default_rng(permutation_seed)
    permuted_weights, permuted_index, permuted_probability, permuted_gains, _ = (
        recompute(
            pre_person.iloc[rng.permutation(len(pre_person))].reset_index(drop=True)
        )
    )
    if not (
        permuted_weights.sort_index().equals(weights.sort_index())
        and permuted_index.sort_index().equals(index.sort_index())
        and permuted_probability.sort_index().equals(probability.sort_index())
        and permuted_gains.sort_index().equals(gains.sort_index())
    ):
        problems["residential_split_permutation"] = True
    return receipt


def _e8_band_donors(
    frame,
    problems: dict[str, object],
    *,
    propensity: pd.DataFrame | None,
    band_taxpayers: Mapping[int, float] | None,
    permutation_seed: int,
) -> dict[str, object]:
    """Reconstruct the SPI income band donor seating from ids (microcosm#1063).

    The donors are a funded support channel: each band's donors carry equal
    weights and their mass left the incumbent households of each donor's
    region in proportion. Folding the CGT layers stacked afterwards gives the
    frame the stage wrote; scaling each region's incumbents back to the
    region's whole mass gives the weights it read. The stored layer must
    agree with itself (one carrier per donor, each source household copied
    once, equal weights within a band, the mass record conserving the total)
    and, when the tape's band propensities are supplied (they need the
    licensed SPI tape), with the stage's own function rerun on the
    reconstructed pre-donor frame: the same source households in the same
    bands, the same carriers and the same band weights, in original and
    permuted row order. The seating is keyed on person ids, so the rerun
    needs no stage-time row order. Without the propensities the receipt says
    the seating was not recomputed. Problems land under ``band_donor_*`` keys.
    """

    from microcosm.build.uk_runtime.cgt_structure import HOUSEHOLD_IS_CGT_CLONE
    from microcosm.build.uk_runtime.cgt_support import HOUSEHOLD_IS_CGT_SUPPORT_COPY
    from microcosm.build.uk_runtime.national_frame import (
        uk_household_weight_kind,
        uk_national_frame,
    )
    from microcosm.build.uk_runtime.spi_band_donors import (
        HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR,
        PERSON_IS_SPI_INCOME_BAND_CARRIER,
        SPI_INCOME_BAND_DONOR_DRAW_SALT,
        SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN,
        SPI_INCOME_BAND_DONOR_LOWER_BOUNDS,
        SPI_INCOME_BAND_DONOR_MASS_CHANGE_REASON,
        SPI_INCOME_BAND_DONOR_SEED,
        _funding_strata,
        stack_spi_income_band_donors,
    )

    if HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR not in frame.table("household").columns:
        raise ValueError(
            "e8 identity receipt: the artifact carries no "
            f"{HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR!r} column, so the SPI income "
            "band donor layer is absent and there is nothing to receipt. "
            "Receipt the artifact with the tool at the commit that built it."
        )
    receipt: dict[str, object] = {
        "seed": SPI_INCOME_BAND_DONOR_SEED,
        "salt": SPI_INCOME_BAND_DONOR_DRAW_SALT,
    }
    mass_records = [
        record
        for record in frame.mass_log
        if record.reason == SPI_INCOME_BAND_DONOR_MASS_CHANGE_REASON
    ]
    if not mass_records:
        problems["band_donor_mass_record"] = "missing"
    else:
        record = mass_records[-1]
        receipt["mass"] = {
            "old_total": float(record.old_total),
            "new_total": float(record.new_total),
            "declared_factor": record.declared_factor,
        }
        if record.declared_factor != 1.0 or not np.isclose(
            record.new_total, record.old_total, rtol=1e-9, atol=0.0
        ):
            problems["band_donor_mass_record"] = [
                float(record.old_total),
                float(record.new_total),
                record.declared_factor,
            ]

    # The frame the donor stage wrote: the CGT layers stacked after it folded
    # back, so every remaining household carries the weight the stage left.
    staged = _drop_stacked_layers(
        frame,
        [
            flag
            for flag in (HOUSEHOLD_IS_CGT_CLONE, HOUSEHOLD_IS_CGT_SUPPORT_COPY)
            if flag in frame.table("household").columns
        ],
    )
    household = staged.table("household")
    person = staged.table("person")
    benunit = staged.table("benunit")
    weights = np.asarray(staged.weights_for("household").values, dtype=float)
    is_donor = household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR].astype(bool).to_numpy()
    donor_band = household[SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN].to_numpy(float)
    donor_source = (
        pd.to_numeric(household.loc[is_donor, "household_source_id"], errors="raise")
        .astype("int64")
        .to_numpy()
    )
    receipt["donor_households"] = int(is_donor.sum())
    stored_seats = pd.Series(donor_band[is_donor], index=donor_source).sort_index()

    # Stored-layer consistency.
    layer: dict[str, int] = {}
    if not stored_seats.index.is_unique:
        layer["source_households_copied_twice"] = int(
            stored_seats.index.duplicated().sum()
        )
    undeclared = ~np.isin(donor_band[is_donor], SPI_INCOME_BAND_DONOR_LOWER_BOUNDS)
    if undeclared.any():
        layer["donors_in_undeclared_bands"] = int(undeclared.sum())
    if (donor_band[~is_donor] != 0.0).any():
        layer["incumbents_carrying_a_band"] = int((donor_band[~is_donor] != 0.0).sum())
    carriers = person[PERSON_IS_SPI_INCOME_BAND_CARRIER].astype(bool)
    carriers_per_household = (
        person.loc[carriers, "person_household_id"].value_counts().to_dict()
    )
    donor_ids = household.loc[is_donor, "household_id"].tolist()
    without_one = sum(
        1 for value in donor_ids if carriers_per_household.get(value, 0) != 1
    )
    if without_one or len(carriers_per_household) != len(donor_ids):
        layer["donors_without_exactly_one_carrier"] = int(without_one)
    band_rows: list[dict[str, object]] = []
    unequal = 0
    for band in SPI_INCOME_BAND_DONOR_LOWER_BOUNDS:
        band_weights = weights[is_donor & (donor_band == band)]
        if band_weights.size and not np.allclose(
            band_weights, band_weights[0], rtol=1e-9, atol=1e-6
        ):
            unequal += 1
        band_rows.append(
            {
                "lower_bound": int(band),
                "donor_households": int(band_weights.size),
                "donor_weight": (float(band_weights[0]) if band_weights.size else None),
                "weighted_taxpayers": float(band_weights.sum()),
            }
        )
    if unequal:
        layer["bands_with_unequal_donor_weights"] = unequal
    if layer:
        problems["band_donor_layer"] = layer
    receipt["bands"] = band_rows

    # The weights the stage read: every region's incumbents gave up the
    # region's donor mass in proportion, so scaling them back to the region's
    # whole mass restores them.
    strata = _funding_strata(household).to_numpy()
    region_mass = pd.Series(weights).groupby(strata).sum()
    incumbent_mass = pd.Series(weights[~is_donor]).groupby(strata[~is_donor]).sum()
    if (incumbent_mass.reindex(region_mass.index).fillna(0.0) <= 0.0).any():
        problems["band_donor_funding"] = "a donor stratum has no incumbent mass"
        return receipt
    factor = incumbent_mass / region_mass.reindex(incumbent_mass.index)
    receipt["funding"] = {
        str(stratum): float(value) for stratum, value in factor.sort_index().items()
    }
    pre_weights = weights[~is_donor] / pd.Series(strata[~is_donor]).map(
        factor
    ).to_numpy(dtype=float)
    pre_household = (
        household.loc[~is_donor]
        .drop(
            columns=[
                HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR,
                SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN,
            ]
        )
        .reset_index(drop=True)
    )
    donor_id_set = set(donor_ids)
    pre_person_rows = ~person["person_household_id"].isin(donor_id_set)
    pre_person = (
        person.loc[pre_person_rows]
        .drop(columns=[PERSON_IS_SPI_INCOME_BAND_CARRIER])
        .reset_index(drop=True)
    )
    pre_benunit = benunit.loc[
        benunit["benunit_id"].isin(set(pre_person["person_benunit_id"]))
    ].reset_index(drop=True)
    stored_carriers = set(
        pd.to_numeric(person.loc[carriers, "person_source_id"], errors="raise")
        .astype("int64")
        .tolist()
    )

    if propensity is None or band_taxpayers is None:
        receipt["seating_recomputed"] = False
        receipt["seating_scope"] = (
            "not recomputed: the propensities come from the licensed SPI tape; "
            "pass --spi-tab to rerun the seating from ids"
        )
        return receipt
    period = int(uk_time_period(frame))

    def recompute(person_t, household_t, weights_t):
        result = stack_spi_income_band_donors(
            uk_national_frame(
                person=person_t,
                benunit=pre_benunit,
                household=household_t,
                time_period=uk_time_period(frame),
                weight_kind=uk_household_weight_kind(frame),
                household_weights=weights_t,
            ),
            propensity=propensity,
            band_taxpayers=band_taxpayers,
            seed=SPI_INCOME_BAND_DONOR_SEED,
            taxpayer_period=period,
        )
        table = result.frame.table("household")
        seated = table[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR].astype(bool)
        seats = pd.Series(
            table.loc[seated, SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN].to_numpy(float),
            index=pd.to_numeric(table.loc[seated, "household_source_id"])
            .astype("int64")
            .to_numpy(),
        ).sort_index()
        people = result.frame.table("person")
        carrier_ids = set(
            pd.to_numeric(
                people.loc[
                    people[PERSON_IS_SPI_INCOME_BAND_CARRIER].astype(bool),
                    "person_source_id",
                ]
            )
            .astype("int64")
            .tolist()
        )
        return seats, carrier_ids, result.evidence()

    seats, carrier_ids, evidence = recompute(pre_person, pre_household, pre_weights)
    receipt["seating_recomputed"] = True
    receipt["seating_scale"] = evidence["seating_scale"]
    receipt["mass_scale"] = evidence["mass_scale"]
    missing = int((~stored_seats.index.isin(seats.index)).sum())
    extra = int((~seats.index.isin(stored_seats.index)).sum())
    if missing or extra:
        problems["band_donor_seats_stored"] = {"missing": missing, "extra": extra}
    else:
        moved = int((seats.to_numpy() != stored_seats.to_numpy()).sum())
        if moved:
            problems["band_donor_bands_stored"] = moved
    if carrier_ids != stored_carriers:
        problems["band_donor_carriers_stored"] = len(carrier_ids ^ stored_carriers)
    recomputed_weight = {
        int(row["lower_bound"]): float(row["donor_weight"]) for row in evidence["bands"]
    }
    weight_off = [
        int(row["lower_bound"])
        for row in band_rows
        if row["donor_weight"] is None
        or not np.isclose(
            row["donor_weight"],
            recomputed_weight[int(row["lower_bound"])],
            rtol=1e-9,
            atol=1e-6,
        )
    ]
    if weight_off:
        problems["band_donor_weights_stored"] = weight_off

    # A frame keeps its group tables sorted by id, so the permutation is over
    # the person rows, the table the candidates are read from.
    rng = np.random.default_rng(permutation_seed)
    permuted_seats, permuted_carriers, _ = recompute(
        pre_person.iloc[rng.permutation(len(pre_person))].reset_index(drop=True),
        pre_household,
        pre_weights,
    )
    if not (permuted_seats.equals(seats) and permuted_carriers == carrier_ids):
        problems["band_donor_seats_permutation"] = True
    return receipt


def _band_donor_seating_inputs(
    frame, spi_tab: Path | None
) -> tuple[pd.DataFrame | None, dict[int, float] | None]:
    """The tape's band propensities and the build year's Table 2.5 taxpayers.

    Built exactly as the stage builds them, from the verified licensed tape;
    ``(None, None)`` without one.
    """

    from microcosm.build.uk_runtime.spi_band_donors import (
        load_hmrc_itl_band_taxpayers,
        spi_income_band_donor_propensity,
    )
    from microcosm.build.uk_runtime.spi_income import (
        load_spi_donor_age_model,
        verify_spi_donor_identity,
    )

    if spi_tab is None:
        return None, None
    identity = verify_spi_donor_identity(spi_tab)
    period = int(uk_time_period(frame))
    propensity = spi_income_band_donor_propensity(
        pd.read_csv(identity.path, delimiter="\t"),
        age_model=load_spi_donor_age_model(period),
    )
    return propensity, load_hmrc_itl_band_taxpayers(period)


def e8_identity_receipt(
    frame,
    *,
    permutation_seed: int,
    spi_tab: Path | None = None,
) -> dict[str, object]:
    """Receipt E8 deterministic layers under row permutation by entity id.

    Covered: (1) the clone-pair structure - every pre-clone household, the
    support split's copies included, has exactly one clone paired by the
    clone stage's id offset; before the #970 anchor their paired weights
    agree to the exact-total correction tolerance and their half-masses
    match, after it the clone side never exceeds the original and the anchor
    recomputed from the reconstructed pre-anchor halves reproduces the
    stored weights in original and reversed person order; (2) the CGT
    support split (microcosm#1045) recomputed from the folded pre-split
    frame with the stage's own helpers - the wealthiest households of each
    Table 3 income band (bands from the stored income components under the
    policy parameters, wealth from the stored columns, support masses from
    the vendored joint) divided into ``ceil(w / 60)`` copies at ``w / n`` -
    in original and permuted row order, against the stored copies, counts,
    family weights, flags and mass record; (3) the student-loan plan column
    recomputed in full (identity-keyed top-ups at the release calibration
    year) in original and permuted row order against the stored column;
    (4) the SPI income band donors (microcosm#1063) - the stored layer's
    structure, equal band weights and conserving mass record, and, with the
    licensed SPI tape (``spi_tab``), the identity-keyed seating rerun on the
    reconstructed pre-donor frame in original and permuted row order;
    (5) the CGT residential split (microcosm#1063) recomputed from the folded
    pre-split frame with the stage's own function - the solved probability
    carried as arm weights - in original and permuted row order against the
    stored arms, indices, weights, probabilities and residential gains. The
    clone-pair, support-split and band-donor checks run on the frame with the
    arms folded onto their sources, the state those layers were written in.
    Every CGT layer ran before ``salary_sacrifice``, whose conversion
    rewrites a converted record's pay in place, so the recomputes first
    reverse it from the stage's pre-conversion pay carrier
    (``_pre_salary_sacrifice_frame``).
    The A&S prior amounts (overwritten by the Table 3 redraw and its
    sub-AEA remainder mapping), the redraw's seeded within-band draws
    (covered by the merged #560 embedded published-surface tests), and the
    salary-sacrifice QRF and conversion (the pre-conversion state is
    consumed by the stage) are covered by twin-build determinism.
    """

    from microcosm.build.uk_runtime.cgt_asset_type import (
        load_hmrc_cgt_asset_type_facts,
    )
    from microcosm.build.uk_runtime.cgt_imputation import uk_cgt_policy_parameters
    from microcosm.build.uk_runtime.cgt_residential_split import (
        HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE,
    )
    from microcosm.build.uk_runtime.cgt_structure import HOUSEHOLD_IS_CGT_CLONE
    from microcosm.build.uk_runtime.cgt_support import (
        CGT_SUPPORT_COPIES_COLUMN,
        HOUSEHOLD_IS_CGT_SUPPORT_COPY,
    )
    from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
    from microcosm.build.uk_runtime.hmrc_capital_gains import (
        load_hmrc_cgt_joint_distribution,
    )
    from microcosm.build.uk_runtime.student_loans import (
        assign_student_loan_plans,
        load_slc_liable_stocks,
    )

    problems: dict[str, object] = {}
    person = frame.table("person")

    # The split's income-band proxy, the anchor and the residential split
    # read the same policy parameters (the tapered Personal Allowance and
    # the annual exempt amount) at the frame's build period.
    parameters = uk_cgt_policy_parameters(uk_time_period(frame))

    # Every CGT layer ran before salary_sacrifice, which rewrites a converted
    # record's pay and employee contribution in place; the recomputes band
    # on the income those stages saw, restored from the stage's carrier.
    pre_sacrifice = _pre_salary_sacrifice_frame(frame)

    # (5) Residential split recomputed from the folded pre-split frame; the
    # CGT layers below it are then checked with the arms folded away.
    residential_split = _e8_residential_split(
        pre_sacrifice,
        problems,
        parameters=parameters,
        facts=load_hmrc_cgt_asset_type_facts(),
        permutation_seed=permutation_seed,
    )
    cgt_frame = _drop_stacked_layers(
        pre_sacrifice, [HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE]
    )
    household = cgt_frame.table("household")

    # (1) Clone-pair structure over every household: the support split's
    # copies are pre-clone rows with clones of their own.
    is_clone = household[HOUSEHOLD_IS_CGT_CLONE].astype(bool).to_numpy()
    if int((~is_clone).sum()) != int(is_clone.sum()):
        problems["clone_half_counts"] = [int((~is_clone).sum()), int(is_clone.sum())]
    clone_pairs = _e8_clone_pairs(cgt_frame, problems, parameters=parameters)

    # (2) Support split recomputed from the folded pre-split frame. Same
    # contract as the E6 NHS check: age_tail runs immediately after
    # frs_spine, so the split classified each household by its oldest-adult
    # carrier on the disaggregated age surface stored in the artifact.
    support_split = _e8_support_split(
        cgt_frame,
        problems,
        distribution=load_hmrc_cgt_joint_distribution(),
        parameters=parameters,
        permutation_seed=permutation_seed,
    )

    # (4) SPI income band donors: the seating reconstructed from ids, on the
    # arm-free frame (the donor stage ran before the residential split).
    propensity, band_taxpayers = _band_donor_seating_inputs(frame, spi_tab)
    band_donors = _e8_band_donors(
        cgt_frame,
        problems,
        propensity=propensity,
        band_taxpayers=band_taxpayers,
        permutation_seed=permutation_seed,
    )

    # (3) Student-loan plan recomputed in full.
    stocks = load_slc_liable_stocks()
    year = load_uk_frs_release().calibration_year
    recomputed = assign_student_loan_plans(frame, stocks=stocks, year=year)
    stored_plan = person.set_index("person_id")["student_loan_plan"]
    recomputed_plan = (
        recomputed.frame.table("person")
        .set_index("person_id")["student_loan_plan"]
        .reindex(stored_plan.index)
    )

    # Compared by value: the stored column comes back from HDF5 under one
    # string dtype and the recomputed one under another, and ``Series.equals``
    # calls two equal columns of different string dtypes unequal.
    def plans_equal(left: pd.Series, right: pd.Series) -> bool:
        return bool(
            np.array_equal(
                left.astype(object).to_numpy(), right.astype(object).to_numpy()
            )
        )

    plan_matches_store = plans_equal(stored_plan, recomputed_plan)
    permuted_result = assign_student_loan_plans(
        _reverse_rows(frame), stocks=stocks, year=year
    )
    permuted_plan = (
        permuted_result.frame.table("person")
        .set_index("person_id")["student_loan_plan"]
        .reindex(stored_plan.index)
    )
    plan_permutation_stable = plans_equal(recomputed_plan, permuted_plan)
    if not plan_matches_store:
        problems["student_loan_plan_stored"] = True
    if not plan_permutation_stable:
        problems["student_loan_plan_permutation"] = True

    return {
        "check": "uk_e8_identity_stability",
        "carrier_age_basis": "stage_time_disaggregated",
        "permutation_seed": permutation_seed,
        "identical_under_permutation": not any(
            key.endswith("_permutation") for key in problems
        ),
        "clone_pairs": clone_pairs,
        "pre_salary_sacrifice_restored": pre_sacrifice is not frame,
        "support_split": support_split,
        "band_donors": band_donors,
        "residential_split": residential_split,
        "permutation_mismatches": {
            key: value
            for key, value in problems.items()
            if key.endswith("_permutation")
        },
        "matches_stored_columns": not any(
            not key.endswith("_permutation") for key in problems
        ),
        "stored_column_mismatches": {
            key: value
            for key, value in problems.items()
            if not key.endswith("_permutation")
        },
        "tolerance_policy": (
            "anchored clone pairs: recomputed #970 anchor weights rtol 1e-9 / "
            "atol 1e-6, reversed-order recompute exact; unanchored clone-pair "
            "weights and half-masses: rtol 1e-12 / atol 1e-6 "
            "(the exact-total correction may move single weights by bit "
            "corrections); support-split family weights: each member's clone "
            "pair sum against w / n at rtol 1e-9 / atol 1e-6 (the fold and "
            "the anchor each cost one float rounding generation); "
            "support-split mass record: new_total against old_total at rtol "
            "1e-9; support-split families, copy counts, flags and "
            "student_loan_plan: exact equality; band-donor seats, bands and "
            "carriers: exact equality; band-donor weights within a band and "
            "against the rerun: rtol 1e-9 / atol 1e-6; band-donor mass "
            "record: new_total against old_total at rtol 1e-9 with declared "
            "factor one; residential-split arms, indices and residential "
            "gains: exact equality; residential-split weights against the "
            "rerun: rtol 1e-9 / atol 1e-6; probabilities: rtol 1e-9; mass "
            "record: new_total against old_total at rtol 1e-9 with declared "
            "factor one"
        ),
        "columns_by_entity": {
            "household": [
                HOUSEHOLD_IS_CGT_CLONE,
                HOUSEHOLD_IS_CGT_SUPPORT_COPY,
                CGT_SUPPORT_COPIES_COLUMN,
                HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE,
                "cgt_residential_clone_index",
                "household_weight",
            ],
            "person": [
                "student_loan_plan",
                "cgt_residential_probability",
                "capital_gains_residential_property",
            ],
        },
        "qrf_draw_columns_scope": (
            "excluded: the A&S prior amounts (overwritten by the Table 3 "
            "redraw and its sub-AEA remainder mapping), the redraw's seeded "
            "within-band draws (the merged #560 embedded published-surface "
            "tests cover the amounts logic), and the salary-sacrifice QRF "
            "and conversion (the pre-conversion column state is consumed "
            "by the stage) are covered by twin-build determinism"
        ),
    }


def e9_identity_receipt(
    frame,
    *,
    permutation_seed: int,
) -> dict[str, object]:
    """Recompute the E9 UC deduction attributes in original and permuted order.

    The four benunit columns are pure functions of ``benunit_id`` and the
    household region under the committed resource, so they are re-derived
    here in full and compared with the stored artifact. The stage runs
    before the CGT support split and the clone, which copy the completed
    columns onto re-keyed rows (support copies and clones); those rows are
    excluded and their count is reported, since their ids are not the ids
    the stage drew on. A split root keeps its ids and stays in scope.
    """

    from microcosm.build.uk_runtime.cgt_structure import HOUSEHOLD_IS_CGT_CLONE
    from microcosm.build.uk_runtime.cgt_support import HOUSEHOLD_IS_CGT_SUPPORT_COPY
    from microcosm.build.uk_runtime.uc_deduction_attributes import (
        UC_DEDUCTION_OUTPUT_COLUMNS,
        UK_UC_DEDUCTION_ATTRIBUTES_DECLARED_SEEDS,
        _banded_rate_mapping,
        _identity_float32_uniforms,
        load_uc_deduction_distributions,
        map_uniform_to_categorical,
        validate_uc_deduction_resource,
    )

    resource = load_uc_deduction_distributions()
    validate_uc_deduction_resource(resource)
    person = frame.table("person")
    benunit = frame.table("benunit")
    household = frame.table("household")
    copied_flags = [
        column
        for column in (HOUSEHOLD_IS_CGT_CLONE, HOUSEHOLD_IS_CGT_SUPPORT_COPY)
        if column in household.columns
    ]
    copied_households = (
        household[copied_flags].astype(bool).any(axis=1)
        if copied_flags
        else pd.Series(False, index=household.index)
    )
    copied_ids = set(household.loc[copied_households, "household_id"])
    benunit_household = person.drop_duplicates("person_benunit_id").set_index(
        "person_benunit_id"
    )["person_household_id"]
    benunit_scope = ~benunit["benunit_id"].map(benunit_household).isin(copied_ids)
    scoped = benunit.loc[benunit_scope.to_numpy()].reset_index(drop=True)
    region_by_household = household.set_index("household_id")["region"]

    def recompute(benunit_t: pd.DataFrame) -> pd.DataFrame:
        ids = benunit_t["benunit_id"].to_numpy()
        regions = (
            benunit_t["benunit_id"]
            .map(benunit_household)
            .map(region_by_household)
            .map(_enum_name)
            .to_numpy(dtype=object)
        )
        u = _identity_float32_uniforms(
            ids,
            seed=UK_UC_DEDUCTION_ATTRIBUTES_DECLARED_SEEDS["uc_deduction_random_draw"],
            salt="uc_deduction_random_draw",
        )
        v = _identity_float32_uniforms(
            ids,
            seed=UK_UC_DEDUCTION_ATTRIBUTES_DECLARED_SEEDS[
                "uc_deduction_type_random_draw"
            ],
            salt="uc_deduction_type_random_draw",
        )
        mapping = _banded_rate_mapping(u, regions, resource)
        combination = map_uniform_to_categorical(
            v, gate=mapping.rates > 0.0, resource=resource
        )
        return pd.DataFrame(
            {
                "uc_deduction_random_draw": u,
                "uc_deduction_type_random_draw": v,
                "uc_latent_deduction_rate": mapping.rates,
                "uc_deduction_combination": combination.astype(str),
            },
            index=ids,
        )

    original = recompute(scoped)
    rng = np.random.default_rng(permutation_seed)
    permuted = recompute(
        scoped.iloc[rng.permutation(len(scoped))].reset_index(drop=True)
    )
    stored = benunit.set_index("benunit_id")
    permutation_mismatches: list[str] = []
    stored_mismatches: list[str] = []
    stored_columns_missing: list[str] = []
    for column in UC_DEDUCTION_OUTPUT_COLUMNS:
        left = original[column]
        right = permuted[column].reindex(left.index)
        if not np.array_equal(left.to_numpy(), right.to_numpy()):
            permutation_mismatches.append(column)
        if column not in stored.columns:
            stored_columns_missing.append(column)
            continue
        kept = stored[column].reindex(left.index)
        if column == "uc_deduction_combination":
            equal = np.array_equal(
                left.to_numpy().astype(str), kept.map(_enum_name).to_numpy().astype(str)
            )
        else:
            equal = np.array_equal(left.to_numpy(), kept.to_numpy(dtype=float))
        if not equal:
            stored_mismatches.append(column)
    return {
        "check": "uk_e9_identity_stability",
        "permutation_seed": permutation_seed,
        "identical_under_permutation": not permutation_mismatches,
        "matches_stored_columns": not stored_mismatches and not stored_columns_missing,
        "permutation_mismatches": {"benunit": permutation_mismatches}
        if permutation_mismatches
        else {},
        "stored_mismatches": {"benunit": stored_mismatches}
        if stored_mismatches
        else {},
        "stored_columns_missing": {"benunit": stored_columns_missing}
        if stored_columns_missing
        else {},
        "columns_by_entity": {"benunit": list(UC_DEDUCTION_OUTPUT_COLUMNS)},
        "benunits_recomputed": int(len(scoped)),
        "benunits_excluded_as_copies": int(len(benunit) - len(scoped)),
        "entity_row_counts": {
            entity: int(len(frame.table(entity))) for entity in frame.entities
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check", choices=("e4", "e5", "e6", "e7", "e8", "e9"), default="e4"
    )
    parser.add_argument("--permutation-seed", type=int, default=123)
    parser.add_argument(
        "--spi-tab",
        type=Path,
        default=None,
        help=(
            "Licensed SPI public use tape (put2223uk.tab). With it the e8 "
            "receipt reruns the income band donor seating from ids; without "
            "it that layer is checked for structure and mass only."
        ),
    )
    args = parser.parse_args()

    frame, _provenance = load_uk_national_frame(args.input_h5)
    if args.check == "e4":
        from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract
        from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

        # E4 draws ran on the pre-stack FRS spine and its take-up targets are
        # unweighted int(rate * n_units): recompute on the survey channel
        # only, or every target moves with the synthetic rows (the post-#717
        # scoping rule the E5 receipt already applies).
        frame = _frs_only_frame(frame)
        engine = PolicyEngineUKEngine()
        lha_category = engine.materialize(
            _engine_safe_frame(frame), ("LHA_category",), uk_time_period(frame)
        )["LHA_category"]
        receipt = e4_identity_receipt(
            frame,
            contract=load_uk_take_up_contract(),
            count_resource=load_brma_count_resource(),
            lha_category=lha_category,
            permutation_seed=args.permutation_seed,
            population_policy=uk_take_up_population_policy(uk_time_period(frame)),
        )
        ok = bool(
            receipt["identical_under_permutation"] and receipt["matches_stored_columns"]
        )
    elif args.check == "e5":
        # E5's wealth stages run after the SPI support rows are stacked and
        # before the CGT layers. The regional property uprating's mean is over
        # FRS-base owners, so a frame still carrying the later CGT copies
        # shifts that denominator and the recomputation stops matching what
        # the stage stored. Scope to the population the stage saw.
        receipt = e5_identity_receipt(
            _frame_as_stage_saw(frame, _E5_FIRST_STAGE),
            permutation_seed=args.permutation_seed,
        )
        ok = bool(
            receipt["identical_under_permutation"] and receipt["matches_stored_columns"]
        )
    elif args.check == "e7":
        # E7's own layer is the support-channel stack, which is defined over
        # the whole frame including the rows it stacks — so unlike e4/e5 this
        # receipt deliberately does NOT scope to the unstacked rows.
        receipt = e7_identity_receipt(
            frame,
            permutation_seed=args.permutation_seed,
        )
        ok = bool(
            receipt["identical_under_permutation"] and receipt["matches_stored_columns"]
        )
    elif args.check == "e6":
        receipt = e6_identity_receipt(
            frame,
            permutation_seed=args.permutation_seed,
        )
        ok = bool(
            receipt["identical_under_permutation"] and receipt["matches_stored_columns"]
        )
    elif args.check == "e9":
        # E9 drew on every benunit present after the SPI stack and before the
        # CGT support split and clone; the block excludes the support copies
        # and the clones itself.
        receipt = e9_identity_receipt(
            frame,
            permutation_seed=args.permutation_seed,
        )
        ok = bool(
            receipt["identical_under_permutation"] and receipt["matches_stored_columns"]
        )
    else:
        receipt = e8_identity_receipt(
            frame,
            permutation_seed=args.permutation_seed,
            spi_tab=args.spi_tab,
        )
        ok = bool(
            receipt["identical_under_permutation"] and receipt["matches_stored_columns"]
        )
    receipt["input_h5"] = str(args.input_h5)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        "identity stability:",
        "PASS" if ok else f"FAIL ({args.output})",
    )
    return 0 if ok else 1


#: Every flag that marks a row as stacked onto the survey channel rather than
#: drawn from the raw FRS. A stage that stacks rows copies its source row's
#: already-computed columns onto the new rows, so an identity-keyed
#: recomputation for a stacked row's *own* id legitimately disagrees with what
#: is stored there — that value was drawn for the source id, not this one.
#: A new stacking stage MUST add its flag here, or these receipts silently
#: start comparing inherited values against fresh draws and report a spine
#: defect that is really an instrument defect.
_STACKING_STAGE_BY_FLAG = {
    # #717 SPI support channel; the #1006 income band donors carry the same flag.
    "household_is_spi_synthetic": "spi_support_channel",
    "household_is_capital_gains_clone": "cgt_incidence_clone",  # E8 clone
    "household_is_cgt_support_copy": "cgt_support_split",  # E8 support copies
    # microcosm#1063: the residential split's arms carry their source's clone
    # and copy flags, so dropping the clone layer drops them too; dropping
    # this layer alone folds each arm onto its source.
    "household_is_cgt_residential_clone": "cgt_residential_split",
}
_STACKED_ROW_FLAGS = tuple(_STACKING_STAGE_BY_FLAG)

#: The first stage of each receipted block. A layer stacked before it was part
#: of the population the block imputed onto; one stacked after it was not.
_E5_FIRST_STAGE = "was_wealth"
_E6_FIRST_STAGE = "nts_bus_travel"

#: The stage that stacks each flag, for artifacts that carry it, when that
#: stage's mass move is a uniform divisor (``_stage_time_weight_divisor``).
#: The weight restoration is driven by what the *artifact* actually contains
#: rather than by the committed roster: a spine built before a stacking stage
#: existed never had that stage's mass factor applied, and dividing it out
#: anyway skews the comparison in the opposite direction. The CGT clone and
#: the CGT support split are deliberately absent: neither is a uniform
#: factor, so their mass is restored by fold instead
#: (``_pre_split_household_weights``).
_MASS_STAGE_BY_FLAG = {
    "household_is_spi_synthetic": "spi_support_channel",
}


def _pre_clone_household_weights(frame) -> np.ndarray:
    """Household weights with each clone's mass folded back onto its original.

    Before the #970 anchor both halves carry half the source weight; after it
    the split varies per pair while the pair sum is still the pre-clone
    weight (the anchor conserves every pair to rounding). Folding the clone
    onto its original is therefore exact either way, where a uniform
    ``mass_split`` divisor would be wrong after the anchor. Every pre-clone
    household is paired, the support split's copies included. An artifact
    without the clone layer is returned unchanged.
    """

    from microcosm.build.uk_runtime.cgt_residential_split import (
        HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE,
    )
    from microcosm.build.uk_runtime.cgt_structure import (
        HOUSEHOLD_IS_CGT_CLONE,
        pair_clone_households,
    )

    household = frame.table("household")
    person = frame.table("person")
    benunit = frame.table("benunit")
    weights = _pre_residential_household_weights(frame)
    if HOUSEHOLD_IS_CGT_CLONE not in household.columns:
        return weights
    # The residential arms (microcosm#1063) are stacked after the clone, so
    # the clone pairing runs on the frame without them, their mass already
    # folded onto their sources.
    keep = np.ones(len(household), dtype=bool)
    if HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE in household.columns:
        keep &= ~household[HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE].to_numpy(dtype=bool)
    if keep.all():
        clone_positions, original_positions, _ = pair_clone_households(
            person, benunit, household
        )
    else:
        kept_household = household.loc[keep].reset_index(drop=True)
        kept_ids = set(kept_household["household_id"].tolist())
        kept_person = person.loc[
            person["person_household_id"].isin(kept_ids)
        ].reset_index(drop=True)
        kept_benunit = benunit.loc[
            benunit["benunit_id"].isin(set(kept_person["person_benunit_id"]))
        ].reset_index(drop=True)
        clone_kept, original_kept, _ = pair_clone_households(
            kept_person, kept_benunit, kept_household
        )
        kept_positions = np.flatnonzero(keep)
        clone_positions = kept_positions[clone_kept]
        original_positions = kept_positions[original_kept]
    weights[original_positions] += weights[clone_positions]
    return weights


def _pre_residential_household_weights(frame) -> np.ndarray:
    """Household weights with every residential arm folded onto its source.

    The arms of ``cgt_residential_split`` (microcosm#1063) divide a
    household's mass by the solved residential probabilities, so summing
    them back onto the arm that kept the household's ids is exact. An
    artifact without the arm layer is returned unchanged. The arms keep their
    own weights in the returned vector; the scoping that calls this drops
    those rows.
    """

    from microcosm.build.uk_runtime.cgt_residential_split import (
        HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE,
    )

    household = frame.table("household")
    weights = np.asarray(frame.weights_for("household").values, dtype=float).copy()
    if HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE not in household.columns:
        return weights
    lineage = _residential_arm_lineage(
        frame.table("person"), frame.table("benunit"), household
    )
    np.add.at(weights, lineage.source_positions, weights[lineage.arm_positions])
    return weights


class _ResidentialArmLineage(NamedTuple):
    """Where every residential arm sits and which source it came from."""

    household_ids: np.ndarray
    pre_split: np.ndarray
    arm_positions: np.ndarray
    source_positions: np.ndarray
    arm_index: np.ndarray
    multiplier: int


def _residential_arm_lineage(
    person: pd.DataFrame, benunit: pd.DataFrame, household: pd.DataFrame
) -> _ResidentialArmLineage:
    """Every residential arm's position, its source's position and its index.

    The split offsets ids by ``id_multiplier_for_values`` over the rows it
    saw, every household of the frame before it ran: the households whose
    arm flag is false. Arm ``j`` of source ``r`` carries ``r + j * M``, so
    ``j = id // M`` and ``r = id - j * M``. An arm without a source, or with
    index zero, fails closed, and the stored ``cgt_residential_clone_index``
    must agree with the id scheme on every row.
    """

    from microcosm.build.uk_runtime.cgt_residential_split import (
        CGT_RESIDENTIAL_CLONE_INDEX_COLUMN,
        HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE,
    )
    from microcosm.build.uk_runtime.rowwise_geography import id_multiplier_for_values

    household_ids = (
        pd.to_numeric(household["household_id"], errors="raise")
        .astype("int64")
        .to_numpy()
    )
    if len(np.unique(household_ids)) != len(household_ids):
        raise ValueError("CGT residential-arm lineage requires unique household ids.")
    is_arm = household[HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE].to_numpy(dtype=bool)
    if CGT_RESIDENTIAL_CLONE_INDEX_COLUMN not in household.columns:
        raise ValueError(
            f"CGT residential-arm lineage requires {CGT_RESIDENTIAL_CLONE_INDEX_COLUMN!r}."
        )
    stored_index = (
        pd.to_numeric(household[CGT_RESIDENTIAL_CLONE_INDEX_COLUMN], errors="raise")
        .astype("int64")
        .to_numpy()
    )
    flag_disagreements = int(((stored_index != 0) != is_arm).sum())
    if flag_disagreements:
        raise ValueError(
            "CGT residential-arm flag and stored arm index disagree on "
            f"{flag_disagreements} household(s)."
        )
    pre_split = ~is_arm
    pre_split_ids = set(household_ids[pre_split].tolist())
    person_pre_split = person["person_household_id"].isin(pre_split_ids).to_numpy()
    benunit_pre_split = (
        benunit["benunit_id"]
        .isin(set(person.loc[person_pre_split, "person_benunit_id"].tolist()))
        .to_numpy()
    )
    multiplier = id_multiplier_for_values(
        person.loc[person_pre_split, "person_id"],
        person.loc[person_pre_split, "person_household_id"],
        person.loc[person_pre_split, "person_benunit_id"],
        benunit.loc[benunit_pre_split, "benunit_id"],
        household_ids[pre_split],
    )
    arm_positions = np.flatnonzero(is_arm)
    arm_ids = household_ids[arm_positions]
    arm_index = arm_ids // multiplier
    source_ids = arm_ids - arm_index * multiplier
    position_by_id = {int(value): index for index, value in enumerate(household_ids)}
    source_positions = np.empty(len(arm_positions), dtype=np.int64)
    for slot, (arm_id, index, source_id) in enumerate(
        zip(arm_ids.tolist(), arm_index.tolist(), source_ids.tolist(), strict=True)
    ):
        source = position_by_id.get(int(source_id))
        if index < 1 or source is None or not pre_split[source]:
            raise ValueError(
                "CGT residential arm without a source household: household_id "
                f"{arm_id} (id multiplier {multiplier})."
            )
        source_positions[slot] = source
    scheme_disagreements = int((stored_index[arm_positions] != arm_index).sum())
    if scheme_disagreements:
        raise ValueError(
            f"CGT residential arm index stored on {scheme_disagreements} arm(s) "
            f"disagrees with the id scheme (id // {multiplier})."
        )
    return _ResidentialArmLineage(
        household_ids=household_ids,
        pre_split=pre_split,
        arm_positions=arm_positions,
        source_positions=source_positions,
        arm_index=arm_index,
        multiplier=multiplier,
    )


class _SupportCopyLineage(NamedTuple):
    """Where every CGT support copy sits and which root it came from."""

    household_ids: np.ndarray
    pre_split: np.ndarray
    copy_positions: np.ndarray
    root_positions: np.ndarray
    copy_index: np.ndarray
    multiplier: int


def _support_copy_lineage(
    person: pd.DataFrame, benunit: pd.DataFrame, household: pd.DataFrame
) -> _SupportCopyLineage:
    """Every support copy's position, its root's position and its index ``k``.

    The split offsets ids by ``id_multiplier_for_values`` over the rows it
    saw, and those rows are recoverable from the artifact: the households
    whose copy flag is false, the clone stage's rows excluded (the split runs
    before the clone, so a clone of a copy carries the copy flag but is not a
    pre-clone row). Copy ``k`` of root ``r`` carries ``r + k * M1``, so ``k =
    id // M1`` and ``r = id - k * M1``. A copy without a root, or with index
    zero, fails closed. An artifact that carries the split's explicit
    ``cgt_support_copy_index`` (0 on roots and unsplit households, ``k`` on
    copy ``k``; the lineage the geography identity keys on) must agree with
    the id scheme on every row, or the lineage fails closed too.
    """

    from microcosm.build.uk_runtime.cgt_structure import HOUSEHOLD_IS_CGT_CLONE
    from microcosm.build.uk_runtime.cgt_support import (
        CGT_SUPPORT_COPY_INDEX_COLUMN,
        HOUSEHOLD_IS_CGT_SUPPORT_COPY,
    )
    from microcosm.build.uk_runtime.rowwise_geography import id_multiplier_for_values

    household_ids = (
        pd.to_numeric(household["household_id"], errors="raise")
        .astype("int64")
        .to_numpy()
    )
    if len(np.unique(household_ids)) != len(household_ids):
        raise ValueError("CGT support-copy lineage requires unique household ids.")
    is_copy = household[HOUSEHOLD_IS_CGT_SUPPORT_COPY].to_numpy(dtype=bool)
    stored_index = None
    if CGT_SUPPORT_COPY_INDEX_COLUMN in household.columns:
        # The explicit lineage is authoritative: the flag and the stored index
        # must agree before any id arithmetic is read.
        stored_index = (
            pd.to_numeric(household[CGT_SUPPORT_COPY_INDEX_COLUMN], errors="raise")
            .astype("int64")
            .to_numpy()
        )
        flag_disagreements = int(((stored_index != 0) != is_copy).sum())
        if flag_disagreements:
            raise ValueError(
                "CGT support-copy flag and stored copy index disagree on "
                f"{flag_disagreements} household(s)."
            )
    pre_clone = (
        ~household[HOUSEHOLD_IS_CGT_CLONE].to_numpy(dtype=bool)
        if HOUSEHOLD_IS_CGT_CLONE in household.columns
        else np.ones(len(household), dtype=bool)
    )
    pre_split = pre_clone & ~is_copy
    pre_split_ids = set(household_ids[pre_split].tolist())
    person_pre_split = person["person_household_id"].isin(pre_split_ids).to_numpy()
    benunit_pre_split = (
        benunit["benunit_id"]
        .isin(set(person.loc[person_pre_split, "person_benunit_id"].tolist()))
        .to_numpy()
    )
    multiplier = id_multiplier_for_values(
        person.loc[person_pre_split, "person_id"],
        person.loc[person_pre_split, "person_household_id"],
        person.loc[person_pre_split, "person_benunit_id"],
        benunit.loc[benunit_pre_split, "benunit_id"],
        household_ids[pre_split],
    )
    copy_positions = np.flatnonzero(pre_clone & is_copy)
    copy_ids = household_ids[copy_positions]
    copy_index = copy_ids // multiplier
    root_ids = copy_ids - copy_index * multiplier
    position_by_id = {int(value): index for index, value in enumerate(household_ids)}
    root_positions = np.empty(len(copy_positions), dtype=np.int64)
    for slot, (copy_id, index, root_id) in enumerate(
        zip(copy_ids.tolist(), copy_index.tolist(), root_ids.tolist(), strict=True)
    ):
        root = position_by_id.get(int(root_id))
        if index < 1 or root is None or not pre_split[root]:
            raise ValueError(
                "CGT support copy without a root household: household_id "
                f"{copy_id} (id multiplier {multiplier})."
            )
        root_positions[slot] = root
    if stored_index is not None:
        scheme_disagreements = int((stored_index[copy_positions] != copy_index).sum())
        if scheme_disagreements:
            raise ValueError(
                f"CGT support copy index stored on {scheme_disagreements} "
                f"copy(ies) disagrees with the id scheme (id // {multiplier})."
            )
    return _SupportCopyLineage(
        household_ids=household_ids,
        pre_split=pre_split,
        copy_positions=copy_positions,
        root_positions=root_positions,
        copy_index=copy_index,
        multiplier=multiplier,
    )


def _pre_split_household_weights(frame) -> np.ndarray:
    """Household weights with every created CGT row folded back onto its source.

    Clones fold onto their originals first (``_pre_clone_household_weights``:
    pair sums, exact before and after the #970 anchor), then every support
    copy folds onto its root, so a household the split divided into ``n``
    rows at ``w / n`` is back at ``w``: the pre-split table's weights, which
    are what every stage before ``cgt_support_split`` saw. The fold is exact
    by construction where no uniform divisor could be - copy counts vary per
    household (microcosm#1045). An artifact without the copy layer returns
    the pre-clone weights; one without the clone layer folds the copies
    only. Copies and clones keep their own weights in the returned vector;
    the scoping that calls this drops those rows.
    """

    from microcosm.build.uk_runtime.cgt_support import HOUSEHOLD_IS_CGT_SUPPORT_COPY

    weights = _pre_clone_household_weights(frame)
    household = frame.table("household")
    if HOUSEHOLD_IS_CGT_SUPPORT_COPY not in household.columns:
        return weights
    lineage = _support_copy_lineage(
        frame.table("person"), frame.table("benunit"), household
    )
    np.add.at(weights, lineage.root_positions, weights[lineage.copy_positions])
    return weights


def _roster_positions() -> dict[str, int]:
    """Stage positions in the committed source-stage roster."""

    from importlib.resources import files as _files

    spec = json.loads(
        _files("microcosm.build.uk")
        .joinpath("source_stages.json")
        .read_text(encoding="utf-8")
    )
    return {stage["stage"]: index for index, stage in enumerate(spec["stages"])}


def _flags_stacked_after(stage: str, household: pd.DataFrame) -> list[str]:
    """Stacking flags present in the artifact whose layer runs after ``stage``.

    Presence is read from the artifact, order from the committed roster, so
    receipt an artifact with the tool at the commit that built it.
    """

    positions = _roster_positions()
    return [
        flag
        for flag, stacker in _STACKING_STAGE_BY_FLAG.items()
        if flag in household.columns and positions[stacker] > positions[stage]
    ]


def _stage_time_weight_divisor(*, after_stages: Sequence[str]) -> float:
    """Product of the declared mass factors applied after a receipted stage.

    Scoping to the unstacked rows restores the stage's *population* but not
    its *weights*: every later stage that redistributes household mass leaves
    the surviving survey rows carrying a fraction of what the receipted stage
    saw. A receipt whose recomputation normalizes against weights — the NHS
    allocation's budget normalization is absolute, not relative — must divide
    that back out or it compares against a different grossing scale.

    On the E8 roster one stage does this uniformly: `spi_support_channel`
    reserves ``share`` of prior mass for the synthetic channel. The factor
    is read from the declared operation rather than hardcoded, so a change
    is picked up automatically; a *new* uniform mass-redistributing op kind
    still has to be added here. The capital-gains clone is not uniform once
    the #970 anchor has run, so its mass is restored by pair sum instead,
    and the capital-gains support split (microcosm#1045) is not a divisor at
    all - it divides each selected household's weight among ``ceil(w / 60)``
    copies at conserved mass - so its mass is restored by folding each copy
    onto its root after the clone fold (``_pre_split_household_weights``).

    ``after_stages`` names only the stages that actually ran in the artifact
    under receipt — see ``_MASS_STAGE_BY_FLAG``. Deriving it from the
    committed roster instead would divide out a factor that a spine built
    before that stage never had applied, which is a regression the pre-E8
    artifact catches immediately.
    """

    from importlib.resources import files as _files

    spec = json.loads(
        _files("microcosm.build.uk")
        .joinpath("source_stages.json")
        .read_text(encoding="utf-8")
    )
    wanted = set(after_stages)
    divisor = 1.0
    for stage in spec["stages"]:
        if stage.get("stage") not in wanted:
            continue
        for operation in stage.get("operations", ()):
            kind = operation.get("kind")
            if kind == "allocate_zero_weight_prior_mass":
                divisor *= 1.0 - float(operation["share"])
    if divisor <= 0.0:
        raise ValueError(
            "stage-time weight divisor collapsed to zero; the declared mass "
            "factors are not usable for a grossing-scale restoration."
        )
    return divisor


def _frs_only_frame(frame):
    """Scope the artifact to the unstacked survey rows (raw FRS only).

    On the E8 roster this leaves the 16,288 raw FRS households, which is the
    population the pre-stacking stages actually drew for. Their weights fold
    each clone back onto its original (exact before and after the #970
    anchor) and each CGT support copy back onto its root, so every surviving
    row carries its pre-split weight. Every pre-stacking stage precedes
    salary_sacrifice, so the conversion's pay rewrite is reversed too.
    """

    frame = _pre_salary_sacrifice_frame(frame)
    household = frame.table("household")
    return _drop_stacked_layers(
        frame, [flag for flag in _STACKED_ROW_FLAGS if flag in household.columns]
    )


def _frame_as_stage_saw(frame, stage: str):
    """Scope the artifact to the rows present when ``stage`` ran.

    A stage before ``salary_sacrifice`` also saw every converted record's
    pay and employee contribution before the conversion rewrote them.
    """

    positions = _roster_positions()
    if positions[stage] < positions[_SALARY_SACRIFICE_STAGE]:
        frame = _pre_salary_sacrifice_frame(frame)
    return _drop_stacked_layers(
        frame, _flags_stacked_after(stage, frame.table("household"))
    )


#: The stage whose conversion rewrites pay in place (microcosm#1069 c9).
_SALARY_SACRIFICE_STAGE = "salary_sacrifice"


def _pre_salary_sacrifice_frame(frame):
    """The artifact with the salary-sacrifice conversion reversed.

    ``salary_sacrifice`` moves a converted record's employee pension
    contribution whole into ``pension_contributions_via_salary_sacrifice``,
    zeroes the source and lowers ``employment_income`` by the amount moved,
    after the capital-gains stages have banded every carrier on that pay
    (microcosm#1063: on the 2 October spine the E8 recomputes banded 2,797
    clone carriers on post-sacrifice pay and derived different anchor targets
    and a different support family). The stage keeps a converted record's
    pre-conversion pay on ``SALSAC_PRE_CONVERSION_PAY_COLUMN`` (zero for
    everyone else), so the rewrite reverses exactly: pay back to the carrier
    and the employee contribution back from the salary-sacrifice column,
    which the conversion set to zero on the record it converted. The stage's
    QRF predictions for the records it did not convert stay as stored; no
    receipted layer before the stage reads them. An artifact without the
    carrier is returned as it is.
    """

    from microcosm.build.uk_runtime.national_frame import (
        uk_household_weight_kind,
        uk_national_frame,
    )
    from microcosm.build.uk_runtime.salary_sacrifice import (
        SALSAC_OUTPUT,
        SALSAC_PRE_CONVERSION_PAY_COLUMN,
    )

    person = frame.table("person")
    if SALSAC_PRE_CONVERSION_PAY_COLUMN not in person.columns:
        return frame
    carrier = pd.to_numeric(
        person[SALSAC_PRE_CONVERSION_PAY_COLUMN], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(carrier).all() or (carrier < 0.0).any():
        raise ValueError(
            f"{SALSAC_PRE_CONVERSION_PAY_COLUMN} must be finite and non-negative."
        )
    converted = carrier > 0.0
    if not converted.any():
        return frame
    person = person.copy()
    pay = pd.to_numeric(person["employment_income"], errors="raise").to_numpy(
        dtype=float, copy=True
    )
    employee = pd.to_numeric(
        person["employee_pension_contributions"], errors="raise"
    ).to_numpy(dtype=float, copy=True)
    sacrifice = pd.to_numeric(person[SALSAC_OUTPUT], errors="raise").to_numpy(
        dtype=float, copy=True
    )
    if (employee[converted] != 0.0).any() or (sacrifice[converted] <= 0.0).any():
        raise ValueError(
            "A converted salary-sacrifice record must carry a zero employee "
            "contribution and a positive salary sacrifice; the artifact's "
            f"{SALSAC_PRE_CONVERSION_PAY_COLUMN} disagrees with its columns."
        )
    pay[converted] = carrier[converted]
    employee[converted] = sacrifice[converted]
    sacrifice[converted] = 0.0
    person["employment_income"] = pay
    person["employee_pension_contributions"] = employee
    person[SALSAC_OUTPUT] = sacrifice
    return uk_national_frame(
        person=person,
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )


def _drop_stacked_layers(frame, flags: Sequence[str]):
    from microcosm.build.uk_runtime.cgt_residential_split import (
        HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE,
    )
    from microcosm.build.uk_runtime.cgt_structure import HOUSEHOLD_IS_CGT_CLONE
    from microcosm.build.uk_runtime.cgt_support import HOUSEHOLD_IS_CGT_SUPPORT_COPY
    from microcosm.build.uk_runtime.national_frame import (
        uk_household_weight_kind,
        uk_national_frame,
    )

    household = frame.table("household")
    flags = list(flags)
    if not flags:
        return frame
    stacked = household[flags].astype(bool).any(axis=1)
    keep = ~stacked
    # Fold created CGT rows' mass back only when their layer is the one being
    # dropped: the clone by pair, the support copy by family onto its root.
    # The split runs before the clone, so dropping the split layer while the
    # artifact's clone layer stays in scope would fold clones that remain.
    if HOUSEHOLD_IS_CGT_SUPPORT_COPY in flags:
        if HOUSEHOLD_IS_CGT_CLONE in household.columns and (
            HOUSEHOLD_IS_CGT_CLONE not in flags
        ):
            raise ValueError(
                "Dropping the CGT support-split layer requires dropping the "
                "clone layer stacked after it; the artifact carries "
                f"{HOUSEHOLD_IS_CGT_CLONE!r} but {flags} does not name it."
            )
        all_weights = _pre_split_household_weights(frame)
    elif HOUSEHOLD_IS_CGT_CLONE in flags:
        all_weights = _pre_clone_household_weights(frame)
    elif HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE in flags:
        all_weights = _pre_residential_household_weights(frame)
    else:
        all_weights = np.asarray(frame.weights_for("household").values, dtype=float)
    weights = all_weights[keep.to_numpy()]
    household = household.loc[keep].reset_index(drop=True)
    ids = set(household["household_id"].tolist())
    person = (
        frame.table("person")
        .loc[lambda t: t["person_household_id"].isin(ids)]
        .reset_index(drop=True)
    )
    benunit_ids = set(person["person_benunit_id"].tolist())
    benunit = (
        frame.table("benunit")
        .loc[lambda t: t["benunit_id"].isin(benunit_ids)]
        .reset_index(drop=True)
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=weights,
        mass_log=frame.mass_log,
    )


def _engine_safe_frame(frame):
    """Fill by-design NaN on channel-only auxiliary columns for engine reads.

    The #717 SPI channel leaves hmrc_spi_* auxiliaries (e.g.
    other_investment_income) NaN on FRS rows by design; instruments fill 0,
    matching stage-time semantics, because the engine adapter rejects NaN.
    LHA_category derivation does not read these columns.
    """

    from microcosm.build.uk_runtime.national_frame import (
        uk_household_weight_kind,
        uk_national_frame,
    )

    tables = {}
    for entity in ("person", "benunit", "household"):
        table = frame.table(entity).copy()
        for column in table.columns:
            if table[column].dtype.kind == "f" and table[column].isna().any():
                table[column] = table[column].fillna(0.0)
        tables[entity] = table
    return uk_national_frame(
        person=tables["person"],
        benunit=tables["benunit"],
        household=tables["household"],
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )


def _reverse_rows(frame):
    from microcosm.build.uk_runtime.national_frame import (
        uk_household_weight_kind,
        uk_national_frame,
    )

    return uk_national_frame(
        person=frame.table("person").iloc[::-1].reset_index(drop=True),
        benunit=frame.table("benunit").copy(),
        household=frame.table("household").copy(),
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )


if __name__ == "__main__":
    sys.exit(main())
