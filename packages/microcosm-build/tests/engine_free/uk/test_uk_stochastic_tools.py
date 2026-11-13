from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.frs_take_up import UKTakeUpPopulationPolicy
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

ROOT = _TEST_PATHS.repository


def _load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_childcare_draws_are_identity_keyed_and_hold_hours_fixed(monkeypatch) -> None:
    tool = _load_tool("fit_uk_childcare_takeup")

    def forbidden_seed(*args, **kwargs):
        raise AssertionError("np.random.seed must not be used")

    monkeypatch.setattr(tool.np.random, "seed", forbidden_seed)
    contract = load_uk_take_up_contract()
    ids = np.array([5, 4, 3, 2, 1])
    first = tool.draw_childcare_inputs(
        ids, tool.INITIAL_PARAMS, seed=42, contract=contract
    )
    second = tool.draw_childcare_inputs(
        ids, tool.INITIAL_PARAMS, seed=42, contract=contract
    )

    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    # The universal flag is nested under the extended one, as frs_take_up
    # draws it.
    assert not (
        first["would_claim_extended_childcare"]
        & ~first["would_claim_universal_childcare"]
    ).any()
    nested = tool.draw_childcare_inputs(
        ids, np.array([0.5, 1.0, 0.5, 0.0, 0.5, 0.5]), seed=42, contract=contract
    )
    assert nested["would_claim_universal_childcare"].all()


def test_childcare_fit_receipt_is_deterministic_with_seeded_runner(
    tmp_path: Path,
) -> None:
    tool = _load_tool("fit_uk_childcare_takeup")
    input_h5 = tmp_path / "fixture.h5"
    input_h5.write_bytes(b"synthetic")

    targets = {name: float(index + 1) for index, name in enumerate(tool.TARGET_IDS)}

    def runner(path: Path, params: np.ndarray, seed: int, period: int, contract):
        scale = float(params.sum()) + seed * 0.0 + len(path.read_bytes()) * 0.0
        assert period == tool.DEFAULT_TARGET_PERIOD == 2025
        assert contract.resource_sha256
        return {name: value + scale * 0.0 for name, value in targets.items()}

    first = tool.fit_childcare_takeup(
        input_h5,
        seed=42,
        maxiter=1,
        runner=runner,
        generated_at="2026-08-17",
        targets=targets,
    )
    second = tool.fit_childcare_takeup(
        input_h5,
        seed=42,
        maxiter=1,
        runner=runner,
        generated_at="2026-08-17",
        targets=targets,
    )

    assert first == second
    assert first["input_sha256"]
    assert first["seed"] == 42
    assert set(first["params"]) == set(tool.RATE_KEYS)
    assert len(first["params"]) == 6
    assert set(first["achieved"]) == set(tool.TARGET_IDS)
    assert first["tool_version"] == 7
    assert first["expectation_basis"] == "dfe_spring_term"
    assert set(first["estimands"]) == set(tool.RATE_KEYS)
    # Vahid's round-2 finding C: the receipt says the Tax-Free Childcare rate
    # is a fit to HMRC's rows, not a take-up estimate for newly eligible
    # families.
    assert "not a take-up estimate" in first["estimands"]["tax_free_childcare"]
    assert first["params_at_bound"] == sorted(
        key
        for key, value in first["params"].items()
        if value <= tool.AT_BOUND_TOLERANCE or value >= 1 - tool.AT_BOUND_TOLERANCE
    )


def test_childcare_expected_counts_are_bilinear_in_the_extended_rate() -> None:
    tool = _load_tool("fit_uk_childcare_takeup")
    # Two rows. Since pe-uk#2177 a working-parent family keeps its targeted
    # hours beside the extended ones, and without its own targeted claim it
    # still receives them, allocated from the working-parent hours (DfE A1.11).
    # The second family is not extended-eligible. A child who receives
    # extended hours is never universal-only.
    with_ext = {
        "hmrc.tfc.government_top_up": np.array([100.0, 50.0]),
        "hmrc.tfc.children_with_used_accounts": np.array([1.0, 1.0]),
        "dfe.funded_childcare.working_parent_children_2_to_4": np.array([1.0, 0.0]),
        "dfe.funded_childcare.early_learning_2_year_olds": np.array([1.0, 1.0]),
        "dfe.funded_childcare.universal_only_children": np.array([0.0, 1.0]),
    }
    without_ext = {
        **with_ext,
        "dfe.funded_childcare.universal_only_children": np.array([1.0, 1.0]),
    }
    allocated = {
        "dfe.funded_childcare.early_learning_2_year_olds": np.array([1.0, 0.0]),
        "dfe.funded_childcare.universal_only_children": np.array([0.0, 0.0]),
    }
    basis = tool.ChildcareExpectationBasis(
        with_extended=with_ext,
        without_extended=without_ext,
        allocated_from_extended=allocated,
        rows=2,
    )
    # RATE_KEYS order: tfc, extended, targeted, universal
    expected = tool.expected_childcare_counts(basis, np.array([0.5, 0.25, 0.6, 0.4]))

    assert expected["hmrc.tfc.government_top_up"] == pytest.approx(75.0)
    assert expected["hmrc.tfc.children_with_used_accounts"] == pytest.approx(1.0)
    assert expected["dfe.funded_childcare.working_parent_children_2_to_4"] == (
        pytest.approx(0.25)
    )
    # targeted: claimed on their own at 0.6 by both families, and for the
    # working-parent family also allocated when it does not claim them but
    # takes up the extended hours (0.4 * 0.25).
    assert expected["dfe.funded_childcare.early_learning_2_year_olds"] == (
        pytest.approx(0.6 * 2.0 + 0.4 * 0.25 * 1.0)
    )
    # universal-only, nested under the extended flag: a family that takes up
    # the extended hours claims the universal ones (the with-extended side,
    # where only the second family's child is universal-only), any other at
    # the conditional universal rate (the without-extended side). No
    # allocated term (tool version 5).
    assert expected["dfe.funded_childcare.universal_only_children"] == (
        pytest.approx(0.25 * 1.0 + 0.75 * 0.4 * 2.0)
    )
    ceiling = tool.expected_childcare_counts(basis, np.ones(4))
    assert ceiling["dfe.funded_childcare.early_learning_2_year_olds"] == 2.0
    # Without its own claim the targeted row rests on the allocation alone.
    allocated_only = tool.expected_childcare_counts(
        basis, np.array([1.0, 1.0, 0.0, 1.0])
    )
    assert allocated_only["dfe.funded_childcare.early_learning_2_year_olds"] == 1.0


def test_childcare_expected_counts_scale_under_two_children_by_their_usage() -> None:
    """Tool version 6: a child under 2 counts only when it uses its hours, so
    each side's under-2 part is scaled by the usage share."""

    tool = _load_tool("fit_uk_childcare_takeup")
    nine_to_eleven = "dfe.funded_childcare.working_parent_children_9_to_11_months"
    two_to_four = "dfe.funded_childcare.working_parent_children_2_to_4"
    # Two working-parent children: one under 2 (whole-year age 1, but 2 on
    # 31 December, so in the 2 to 4 row) and one aged 3; and a baby.
    with_ext = {
        "hmrc.tfc.government_top_up": np.array([0.0, 0.0, 0.0]),
        two_to_four: np.array([1.0, 1.0, 0.0]),
        nine_to_eleven: np.array([0.0, 0.0, 1.0]),
    }
    # The first child is aged 1, the baby aged 0: each age has its share.
    under_two = {
        "with_extended": {
            two_to_four: {0: np.zeros(3), 1: np.array([1.0, 0.0, 0.0])},
            nine_to_eleven: {0: np.array([0.0, 0.0, 1.0]), 1: np.zeros(3)},
        }
    }
    basis = tool.ChildcareExpectationBasis(
        with_extended=with_ext,
        without_extended=with_ext,
        allocated_from_extended={},
        rows=3,
        under_two=under_two,
    )
    expected = tool.expected_childcare_counts(
        basis, np.array([1.0, 0.8, 1.0, 1.0, 0.4, 0.6])
    )
    assert expected[two_to_four] == pytest.approx(0.8 * (1.0 + 0.6 * 1.0))
    assert expected[nine_to_eleven] == pytest.approx(0.8 * 0.4)
    assert basis.total("with_extended", two_to_four, {0: 0.0, 1: 0.0}) == 1.0


def test_the_fit_draws_extended_hours_at_the_contract_anchor() -> None:
    """The spine predates the anchor, so the fit sets the hours the take-up
    stage draws under the current contract (microcosm#1095, #1126)."""

    tool = _load_tool("fit_uk_childcare_takeup")
    from microcosm.build.stochastic_assignment import (
        clipped_normal_from_uniforms,
        stable_identity_uniforms,
    )
    from microcosm.build.uk_runtime.take_up_contract import (
        load_uk_take_up_contract,
    )

    contract = load_uk_take_up_contract()
    ids = np.arange(1, 2001, dtype=np.int64)
    hours = tool._anchor_hours(ids, contract)
    entry = contract.continuous_entry(tool.HOURS_OUTPUT)
    expected = clipped_normal_from_uniforms(
        stable_identity_uniforms(ids, seed=0, salt=tool.HOURS_OUTPUT),
        mean=float(entry["mean"]),
        sd=float(entry["sd"]),
        lower=float(entry["lower"]),
        upper=float(entry["upper"]),
    )
    np.testing.assert_array_equal(hours, expected)
    assert float(entry["mean"]) == 31.141
    assert hours.min() >= 0.0 and hours.max() <= 30.0


def test_childcare_fitter_family_allow_list_excludes_entitlement_spending() -> None:
    tool = _load_tool("fit_uk_childcare_takeup")

    assert tool.TARGET_IDS == (
        "hmrc.tfc.government_top_up",
        "hmrc.tfc.children_with_used_accounts",
        "dfe.funded_childcare.working_parent_children_2_to_4",
        "dfe.funded_childcare.early_learning_2_year_olds",
        "dfe.funded_childcare.universal_only_children",
        "dfe.funded_childcare.working_parent_children_9_to_11_months",
        "dfe.funded_childcare.working_parent_children_age_1",
    )
    assert tool.DIAGNOSTIC_IDS == (
        "dfe.funded_childcare.working_parent_children_age_2",
        "dfe.funded_childcare.working_parent_children_age_3",
        "dfe.funded_childcare.universal_nursery_children_age_3",
    )
    assert not any("spending" in target_id for target_id in tool.TARGET_IDS[1:])


def test_identity_stability_receipt_compares_by_entity_id() -> None:
    tool = _load_tool("verify_uk_identity_stability")
    frame = _frame()

    def transform(candidate):
        household = candidate.table("household").copy()
        household["property_purchased"] = household["household_id"] % 2 == 0
        return uk_national_frame(
            person=candidate.table("person").copy(),
            benunit=candidate.table("benunit").copy(),
            household=household,
            time_period="2023",
            household_weights=candidate.weights_for("household").values,
        )

    receipt = tool.identity_stability_receipt(
        frame,
        transform=transform,
        columns_by_entity={"household": ("property_purchased",)},
    )

    assert receipt["identical"] is True
    assert receipt["mismatches"] == {}


def test_e4_identity_receipt_survives_permutation_on_synthetic_frame() -> None:
    tool = _load_tool("verify_uk_identity_stability")

    class Contract:
        build_year = 2024

        def rate(self, key: str, build_year: int | None = None) -> float:
            if key == "tax_free_childcare_spend_routed_share":
                return 0.593
            return 0.5 if not key.startswith("scp") else 0.9

        def entry(self, key: str):
            assert key == "tax_free_childcare_spend_routed_share"
            return SimpleNamespace(raw={"entity": "person"})

        def continuous_entry(self, key: str):
            return {"mean": 15.019, "sd": 4.972, "lower": 0, "upper": 30}

    person = pd.DataFrame(
        {
            "person_id": [101, 102, 201, 301],
            "person_benunit_id": [10, 10, 20, 30],
            "person_household_id": [1, 1, 1, 2],
            "age": [5, 6, 40, 70],
            "employment_status": ["CHILD", "CHILD", "FT_EMPLOYED", "RETIRED"],
            "months_since_last_birthday": [0.0] * 4,
            "is_uc_claimant": [False, False, True, True],
            "child_benefit_reported": [0.0, 10.0, 0.0, 0.0],
            "pension_credit_reported": [0.0, 0.0, 0.0, 5.0],
            "universal_credit_reported": [0.0, 0.0, 20.0, 0.0],
        }
    )
    # The BRMA is area_brma's (microcosm#931): the London household's is
    # read off its stored drawn area (a toy support's LSOA, on the committed
    # BRMA layer in London), and the Northern Ireland household's drawn from
    # its count-table cell.
    frame = uk_national_frame(
        person=person,
        benunit=pd.DataFrame(
            {"benunit_id": [10, 20, 30], "is_married": [False, True, False]}
        ),
        household=pd.DataFrame(
            {
                "household_id": [1, 2],
                "region": ["LONDON", "NORTHERN_IRELAND"],
                "household_weight": [2.0, 3.0],
                "atomic_area_system": [
                    "uk_ew_output_area_2021",
                    "uk_ni_data_zone_2021",
                ],
                "oa_code": ["E00000001", "N20000001"],
                "lsoa_code": ["E01000001", "N20000001"],
            }
        ),
        time_period="2023",
    )
    count_resource = {
        "cells": {
            "NORTHERN_IRELAND": {"A": {"BELFAST": 2, "SOUTH_EAST_NI": 1}},
        }
    }

    receipt = tool.e4_identity_receipt(
        frame,
        contract=Contract(),
        count_resource=count_resource,
        lha_category=["A", "A", "A"],
        permutation_seed=7,
        population_policy=UKTakeUpPopulationPolicy(
            adult_age=18, state_pension_age=66, instant="2023-01-01", source="test"
        ),
    )

    assert receipt["identical_under_permutation"] is True
    assert receipt["permutation_mismatches"] == {}
    # The synthetic frame carries no stored E4 columns; the receipt says so
    # explicitly instead of silently passing the stored comparison.
    assert receipt["matches_stored_columns"] is False
    assert set(receipt["stored_columns_missing"]) == {
        "person",
        "benunit",
        "household",
    }


#: An invented count table for Northern Ireland, the one region whose BRMAs
#: area_brma draws from it (microcosm#931).
_NI_COUNTS = {
    "cells": {
        "NORTHERN_IRELAND": {
            "A": {"BELFAST": 1, "NORTH": 1},
            "B": {"BELFAST": 2, "NORTH": 1, "WEST": 1},
        }
    }
}


def _ni_units(cells: dict[str, list[tuple[str, int]]]) -> pd.DataFrame:
    """Invented Northern Ireland benefit units: per LHA category, each unit's
    BRMA and true source household."""

    rows = [
        {
            "region": "NORTHERN_IRELAND",
            "LHA_category": category,
            "brma": brma,
            "source_household": source,
        }
        for category, units in cells.items()
        for brma, source in units
    ]
    frame = pd.DataFrame(rows)
    frame.insert(0, "benunit_id", range(1, len(frame) + 1))
    return frame


def test_brma_cell_distribution_counts_rounded_sources_and_reports_z() -> None:
    tool = _load_tool("emit_uk_brma_distribution")
    # Cell A: twelve units of twelve sources, eight in BELFAST and four in
    # NORTH; a household's second unit (source 1 twice) counts once.
    benunit = _ni_units(
        {
            "A": [("BELFAST", source) for source in range(1, 9)]
            + [("NORTH", source) for source in range(9, 13)]
            + [("BELFAST", 1)],
        }
    )

    payload = tool.brma_cell_distribution(benunit, count_resource=_NI_COUNTS)

    rows = {row["brma"]: row for row in payload["rows"]}
    assert payload["private"] is True
    assert payload["minimum_count"] == 3
    # Twelve sources round to ten; eight and four sources are banded.
    assert rows["BELFAST"]["cell_n"] == rows["NORTH"]["cell_n"] == 10
    assert rows["BELFAST"]["built_count"] == "3-9"
    assert rows["NORTH"]["built_count"] == "3-9"
    # The shares read benefit units, the sampler's own unit, floored to the
    # hundredth (9/13 and 4/13), and z is rounded to one decimal, so neither
    # gives the cell's unit counts back.
    assert rows["BELFAST"]["built_share"] == 0.69
    assert rows["NORTH"]["built_share"] == 0.30
    for brma in ("BELFAST", "NORTH"):
        z = rows[brma]["z"]
        assert z is not None and z == round(z, 1)
    assert payload["cells_withheld"] == 0
    assert payload["max_abs_z"] == max(abs(row["z"]) for row in rows.values()) > 0


def test_brma_cell_distribution_withholds_a_cell_with_any_withheld_brma() -> None:
    """A BRMA one or two source households carry withholds every figure of
    its cell, the cell's count, every BRMA's count, share and z, which would
    give it back by difference; another cell keeps its own."""

    tool = _load_tool("emit_uk_brma_distribution")
    benunit = _ni_units(
        {
            "A": [("BELFAST", source) for source in range(1, 9)]
            + [("NORTH", source) for source in range(9, 13)],
            # WEST on two units of one source: withheld, with its cell.
            "B": [("BELFAST", source) for source in range(20, 25)]
            + [("NORTH", source) for source in range(25, 29)]
            + [("WEST", 30), ("WEST", 30)],
        }
    )

    payload = tool.brma_cell_distribution(benunit, count_resource=_NI_COUNTS)

    rows = {(row["lha_category"], row["brma"]): row for row in payload["rows"]}
    for brma in ("BELFAST", "NORTH", "WEST"):
        withheld = rows[("B", brma)]
        assert withheld["cell_n"] == withheld["built_count"] == "<3"
        assert withheld["built_share"] is None
        assert withheld["z"] is None
        assert set(withheld) == {
            "region",
            "lha_category",
            "brma",
            "cell_n",
            "expected_share",
            "built_count",
            "built_share",
            "z",
        }
    assert rows[("A", "BELFAST")]["built_share"] is not None
    assert rows[("A", "BELFAST")]["cell_n"] == 10
    assert payload["cells_withheld"] == 1
    # Only the cells nothing in is withheld enter the largest z.
    assert payload["max_abs_z"] == max(
        abs(row["z"]) for row in payload["rows"] if row["lha_category"] == "A"
    )


def test_brma_cell_distribution_prints_no_share_in_a_cell_of_under_ten_sources() -> (
    None
):
    """A cell resting on fewer than ten source households, none of its BRMAs
    withheld, prints its rounded counts but no share or z, which as
    benefit-unit ratios would narrow its unit counts."""

    tool = _load_tool("emit_uk_brma_distribution")
    benunit = _ni_units(
        {
            "A": [("BELFAST", source) for source in (1, 2, 3)]
            + [("NORTH", source) for source in (4, 5, 6)]
        }
    )

    payload = tool.brma_cell_distribution(benunit, count_resource=_NI_COUNTS)

    for row in payload["rows"]:
        assert row["cell_n"] == "3-9"
        assert row["built_share"] is None and row["z"] is None
    assert {row["brma"]: row["built_count"] for row in payload["rows"]} == {
        "BELFAST": "3-9",
        "NORTH": "3-9",
    }
    assert payload["cells_withheld"] == 0
    assert payload["max_abs_z"] == 0.0


def test_brma_cell_distribution_refuses_great_britain_rows() -> None:
    """Great Britain's BRMAs are read off their drawn areas since area_brma
    (microcosm#931), so they hold no count-table draw to compare."""

    tool = _load_tool("emit_uk_brma_distribution")
    benunit = _ni_units({"A": [("BELFAST", 1)]})
    benunit.loc[0, "region"] = "LONDON"
    with pytest.raises(ValueError, match="Northern Ireland's benefit units alone"):
        tool.brma_cell_distribution(benunit, count_resource=_NI_COUNTS)


def test_brma_cell_distribution_refuses_a_floor_below_the_uk_floor() -> None:
    tool = _load_tool("emit_uk_brma_distribution")
    with pytest.raises(ValueError, match="UK disclosure floor"):
        tool.brma_cell_distribution(
            _ni_units({"A": [("BELFAST", 1)]}),
            count_resource=_NI_COUNTS,
            minimum_count=2,
        )


def test_brma_cell_distribution_fails_closed_on_missing_cell() -> None:
    tool = _load_tool("emit_uk_brma_distribution")
    benunit = _ni_units({"C": [("BELFAST", 1)]})

    with pytest.raises(KeyError, match="LHA_category"):
        tool.brma_cell_distribution(benunit, count_resource=_NI_COUNTS)


def test_brma_distribution_output_stays_outside_every_work_tree(
    monkeypatch, tmp_path
) -> None:
    """The diagnostics are private: an --output git could commit is refused
    before the frame is read."""

    import sys

    tool = _load_tool("emit_uk_brma_distribution")
    loaded: list[object] = []
    monkeypatch.setattr(
        tool, "load_uk_national_frame", lambda path: loaded.append(path)
    )
    inside = paths_for("microcosm-build").repository / "brma-distribution.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "emit_uk_brma_distribution.py",
            "--input-h5",
            str(tmp_path / "spine.h5"),
            "--output",
            str(inside),
        ],
    )
    with pytest.raises(SystemExit, match="outside every checkout"):
        tool.main()
    assert loaded == [] and not inside.exists()


def _frame():
    person = pd.DataFrame(
        {
            "person_id": [101, 102, 103],
            "person_benunit_id": [201, 202, 203],
            "person_household_id": [1, 2, 3],
        }
    )
    benunit = pd.DataFrame(
        {"benunit_id": [201, 202, 203], "is_married": [False, True, False]}
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "household_weight": [1.0, 1.0, 1.0],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
    )
