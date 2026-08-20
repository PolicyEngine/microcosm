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
)
from microcosm.build.uk_runtime.national_build import load_uk_national_frame
from microcosm.build.uk_runtime.national_frame import uk_time_period
from microcosm.build.uk_runtime.regional_uprating import (
    load_regional_land_values_resource,
    uprate_household_property_by_region,
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
) -> dict[str, object]:
    """Recompute every E4 column in original and permuted row order.

    Two claims are receipted: a row permutation of the input tables changes
    no assignment per entity id, and the original-order recomputation equals
    the columns stored in the artifact (re-derivation identity).
    """

    person = frame.table("person")
    benunit = frame.table("benunit").copy()
    household = frame.table("household")
    if len(lha_category) != len(benunit):
        raise ValueError("LHA_category materialization must align to benunit rows.")
    benunit["LHA_category"] = [_enum_name(value) for value in lha_category]
    benunit["region"] = _benunit_regions(person, household, benunit)

    def recompute(person_t, benunit_t, household_t) -> dict[str, pd.DataFrame]:
        anchors = aggregate_person_reported_to_benunit(person_t, benunit_t)
        take_up = derive_frs_take_up(benunit_t, anchors=anchors, contract=contract)
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

    person = frame.table("person")
    benunit = frame.table("benunit")
    household = frame.table("household")
    # Scope to the FRS spine rows: the SPI channel stages stack synthetic
    # households AFTER the E5 stages ran (with their own channel-imputed
    # property values), so the deterministic-layer identity claims apply
    # only to the population the wealth and uprating stages actually saw.
    # The stacked rows are E7's receipt surface, not E5's.
    if "household_is_spi_synthetic" in household.columns:
        spine_mask = ~household["household_is_spi_synthetic"].astype(bool)
        household = household.loc[spine_mask].reset_index(drop=True)
        spine_household_ids = set(household["household_id"].tolist())
        person = person.loc[
            person["person_household_id"].isin(spine_household_ids)
        ].reset_index(drop=True)
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
    The QRF chain draws and the NEED raking outcome are covered by
    twin-build determinism and the aggregate_admin NEED-margin receipt
    respectively (raking inputs are consumed by the stage and are not
    reconstructible from the artifact).
    """

    from microcosm.build.uk_runtime.etb_services import (
        allocate_nhs_by_age_gender,
        load_etb_services_anchors,
    )

    rail_fare_index = float(
        load_etb_services_anchors()["rail_fare_index_2023"]["value"]
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
            nhs = allocate_nhs_by_age_gender(
                person_t,
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
    # Scope to the FRS spine rows: the SPI channel stages stack synthetic
    # households AFTER the E6 stages ran (clones inherit their donors'
    # consumption/services values), so the deterministic-layer identity
    # claims apply to the population the consumption stages actually saw.
    # The stacked rows are E7's receipt surface, not E6's. The
    # spi_support_channel stage also scales the survey channel's weights by
    # (1 - share) after the E6 stages ran; the NHS allocation's budget
    # normalization is absolute, so restore the stage-time grossing scale
    # from the declared share before recomputing.
    if "household_is_spi_synthetic" in household.columns:
        spine_mask = ~household["household_is_spi_synthetic"].astype(bool)
        household = household.loc[spine_mask].reset_index(drop=True)
        spine_household_ids = set(household["household_id"].tolist())
        person = person.loc[
            person["person_household_id"].isin(spine_household_ids)
        ].reset_index(drop=True)
        from importlib.resources import files as _files

        spec = json.loads(
            _files("microcosm.build.uk")
            .joinpath("source_stages.json")
            .read_text(encoding="utf-8")
        )
        channel = next(
            (s for s in spec["stages"] if s["stage"] == "spi_support_channel"),
            None,
        )
        if channel is not None:
            share = next(
                op["share"]
                for op in channel["operations"]
                if op["kind"] == "allocate_zero_weight_prior_mass"
            )
            household["household_weight"] = household["household_weight"].to_numpy(
                dtype=float
            ) / (1.0 - float(share))
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", choices=("e4", "e5", "e6"), default="e4")
    parser.add_argument("--permutation-seed", type=int, default=123)
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
        )
        ok = bool(
            receipt["identical_under_permutation"] and receipt["matches_stored_columns"]
        )
    elif args.check == "e5":
        receipt = e5_identity_receipt(
            frame,
            permutation_seed=args.permutation_seed,
        )
        ok = bool(
            receipt["identical_under_permutation"] and receipt["matches_stored_columns"]
        )
    else:
        receipt = e6_identity_receipt(
            frame,
            permutation_seed=args.permutation_seed,
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


def _frs_only_frame(frame):
    """Scope a post-#717 artifact to the survey channel (FRS rows only)."""

    from microcosm.build.uk_runtime.national_frame import (
        uk_household_weight_kind,
        uk_national_frame,
    )

    household = frame.table("household")
    if "household_is_spi_synthetic" not in household.columns:
        return frame
    keep = ~household["household_is_spi_synthetic"].astype(bool)
    weights = frame.weights_for("household").values[keep.to_numpy()]
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
