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
        return {"household": household_out, "person": person}

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
    mismatches: dict[str, list[str]] = {}
    for entity, values in original.items():
        for column in values.columns:
            left = values[column]
            right = permuted[entity][column].reindex(left.index)
            if not np.allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
            ):
                mismatches.setdefault(entity, []).append(column)
    return {
        "check": "uk_e5_identity_stability",
        "permutation_seed": permutation_seed,
        "identical_under_permutation": not mismatches,
        "permutation_mismatches": mismatches,
        "columns_by_entity": {
            entity: list(values.columns) for entity, values in original.items()
        },
        "qrf_draw_columns_scope": (
            "excluded: seeded-stream QRF draws are covered by twin-build "
            "determinism rather than identity-keyed row permutation"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", choices=("e4", "e5"), default="e4")
    parser.add_argument("--permutation-seed", type=int, default=123)
    args = parser.parse_args()

    frame, _provenance = load_uk_national_frame(args.input_h5)
    if args.check == "e4":
        from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract
        from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

        engine = PolicyEngineUKEngine()
        lha_category = engine.materialize(
            frame, ("LHA_category",), uk_time_period(frame)
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
    else:
        receipt = e5_identity_receipt(
            frame,
            permutation_seed=args.permutation_seed,
        )
        ok = bool(receipt["identical_under_permutation"])
    receipt["input_h5"] = str(args.input_h5)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        "identity stability:",
        "PASS" if ok else f"FAIL ({args.output})",
    )
    return 0 if ok else 1


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
