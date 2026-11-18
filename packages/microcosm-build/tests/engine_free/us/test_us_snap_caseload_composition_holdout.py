"""Held-out SNAP caseload composition without a country engine (microcosm#647).

Invariants:

* The holdout never changes weights or any other input: it reads read-only
  arrays and returns a new payload.
* Weighted units, persons and benefits are linear in the weights, invariant to
  unit order, and additive over any split of the units (household batches).
* The three composition groups partition the receiving units, and receiving
  units never outweigh eligible units.
* A benefit paid to a unit not eligible in December reaches no share.
* Units reporting receipt fall in exactly one of three groups: receiving,
  eligible with no benefit, or not eligible.
* A state code that belongs to no state is refused, never filed under one.
* A share with no weight behind it is null, never zero or a failed gate.
* The USDA shares are validation only: no ledger mapping binds a federal test,
  the pinned Chronicle feed carries no caseload-composition record set, and
  only the release tool imports the holdout.
"""

import json
import math
import re
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime import fiscal_targets
from microcosm.build.us_runtime.snap_caseload_composition_holdout import (
    SNAP_CASELOAD_ASSET_THRESHOLD,
    SNAP_CASELOAD_COMPOSITION_HOLDOUT_VARIABLES,
    SNAP_SATURATION_THRESHOLD,
    USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE_2026,
    USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE_URL,
    snap_caseload_composition_holdout_from_sim,
    snap_caseload_composition_holdout_payload,
    snap_caseload_composition_holdout_unavailable,
    write_snap_caseload_composition_holdout,
)
from test_support.paths import paths_for

_COLUMNS = (
    "unit",
    "weight",
    "snap",
    "is_snap_eligible",
    "meets_snap_gross_income_test",
    "meets_snap_net_income_test",
    "meets_snap_asset_test",
    "snap_assets",
    "takes_up_snap_if_eligible",
    "receives_snap",
    "snap_unit_size",
    "spm_unit_state_fips",
)
# Annual benefits. F is eligible without the take-up flag; I carries the flag
# and a zero benefit; H has a benefit but is not eligible in December; G's
# assets sit exactly at the reporting threshold; D fails an income test and the
# resource test. A, D, F, G and H report receipt: F and H without receiving.
_ROWS = (
    ("A", 10.0, 1_200.0, True, True, True, True, 0.0, True, True, 1, 1),
    ("B", 30.0, 2_400.0, True, True, True, True, 500.0, True, False, 3, 1),
    ("C", 20.0, 600.0, True, False, True, True, 4_000.0, True, False, 2, 1),
    ("D", 10.0, 240.0, True, True, False, False, 5_000.0, True, True, 1, 6),
    ("E", 20.0, 4_800.0, True, True, True, False, 9_000.0, True, False, 4, 6),
    ("F", 40.0, 0.0, True, True, True, True, 0.0, False, True, 2, 6),
    ("G", 10.0, 3_600.0, True, True, True, True, 3_000.0, True, True, 1, 6),
    ("H", 5.0, 900.0, False, True, True, True, 0.0, True, True, 2, 6),
    ("I", 25.0, 0.0, True, True, True, True, 0.0, True, False, 2, 1),
)


_REPORTED_GROUPS = ("modeled_recipient", "eligible_non_recipient", "not_eligible")


def _units(rows=_ROWS) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=_COLUMNS).drop(columns="unit")


def _payload(units=None, *, period=2024, **kwargs) -> dict:
    return snap_caseload_composition_holdout_payload(
        _units() if units is None else units, period=period, **kwargs
    )


def test_usda_shares_match_the_proposed_rule_preamble():
    """91 FR 72457 (shares by limit, note 25) and 72459 (combined shares)."""
    rows = USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE_2026
    income, assets, outside = (
        rows["income_failing"],
        rows["asset_only"],
        rows["outside_federal_limits"],
    )
    assert (income.household_share, income.printed_page) == (0.054, 72457)
    assert (income.households, income.individuals) == (1_200_000, 2_000_000)
    assert (assets.household_share, assets.printed_page) == (0.115, 72457)
    assert (assets.households, assets.individuals) == (2_500_000, 5_400_000)
    assert income.fiscal_year == assets.fiscal_year == 2024
    assert (outside.household_share, outside.participant_share) == (0.169, 0.176)
    assert outside.printed_page == 72459
    assert outside.household_share == pytest.approx(
        income.household_share + assets.household_share
    )
    assert list(rows) == ["income_failing", "asset_only", "outside_federal_limits"]
    with pytest.raises(TypeError):
        rows["within_federal_limits"] = income


def test_holdout_variables_are_the_engine_measures_in_request_order():
    assert SNAP_CASELOAD_COMPOSITION_HOLDOUT_VARIABLES == (
        "snap",
        "is_snap_eligible",
        "meets_snap_gross_income_test",
        "meets_snap_net_income_test",
        "meets_snap_asset_test",
        "snap_assets",
        "takes_up_snap_if_eligible",
        "receives_snap",
        "snap_unit_size",
        "spm_unit_state_fips",
    )


def test_take_up_counts_eligible_flagged_and_receiving_units_and_persons():
    take_up = _payload()["national"]["take_up"]

    # Eligible: every unit but H. 10 + 30 + 20 + 10 + 20 + 40 + 10 + 25 = 165.
    assert take_up["eligible_records"] == 8
    assert take_up["eligible_weighted_units"] == 165
    # 10*1 + 30*3 + 20*2 + 10*1 + 20*4 + 40*2 + 10*1 + 25*2 = 370.
    assert take_up["eligible_weighted_persons"] == 370
    # F is eligible without the flag: 165 - 40.
    assert take_up["flagged_weighted_units"] == 125
    assert take_up["flagged_share_of_eligible_units"] == pytest.approx(125 / 165)
    # Receiving: A, B, C, D, E and G. F and I are eligible with no benefit.
    assert take_up["receiving_weighted_units"] == 100
    assert take_up["receiving_weighted_persons"] == 240
    assert take_up["receiving_share_of_eligible_units"] == pytest.approx(100 / 165)
    assert take_up["receiving_share_of_eligible_persons"] == pytest.approx(240 / 370)


def test_composition_groups_partition_the_receiving_units():
    national = _payload()["national"]
    groups = national["composition"]

    assert national["receiving"]["records"] == 6
    assert national["receiving"]["weighted_units"] == 100
    # C fails the gross test and D the net test: 20 + 10.
    assert groups["income_failing"]["records"] == 2
    assert groups["income_failing"]["weighted_units"] == 30
    assert groups["income_failing"]["share_of_receiving_units"] == pytest.approx(0.30)
    # E passes both income tests and fails the resource test.
    assert groups["asset_only"]["records"] == 1
    assert groups["asset_only"]["share_of_receiving_units"] == pytest.approx(0.20)
    # A, B and G: 10 + 30 + 10.
    assert groups["within_federal_limits"]["records"] == 3
    assert groups["within_federal_limits"]["share_of_receiving_units"] == (
        pytest.approx(0.50)
    )
    assert sum(row["weighted_units"] for row in groups.values()) == 100


def test_a_unit_failing_income_and_resources_counts_as_income_failing():
    groups = _payload()["national"]["composition"]

    # D fails the net test and the resource test; only E is asset-only.
    assert groups["income_failing"]["weighted_units"] == 30
    assert groups["asset_only"]["weighted_units"] == 20


def test_person_shares_weight_each_unit_by_its_size():
    groups = _payload()["national"]["composition"]

    # Receiving persons: 10 + 90 + 40 + 10 + 80 + 10 = 240.
    assert groups["income_failing"]["weighted_persons"] == 50
    assert groups["income_failing"]["share_of_receiving_persons"] == pytest.approx(
        50 / 240
    )
    assert groups["asset_only"]["share_of_receiving_persons"] == pytest.approx(80 / 240)
    assert groups["within_federal_limits"]["share_of_receiving_persons"] == (
        pytest.approx(110 / 240)
    )


def test_benefits_are_annual_totals_and_monthly_averages_divide_by_twelve():
    national = _payload()["national"]
    groups = national["composition"]

    # 12,000 + 72,000 + 12,000 + 2,400 + 96,000 + 36,000.
    assert national["receiving"]["annual_benefits"] == 230_400
    # (20*600 + 10*240) / 12 / 30.
    assert groups["income_failing"]["average_monthly_benefit"] == pytest.approx(40.0)
    assert groups["asset_only"]["average_monthly_benefit"] == pytest.approx(400.0)
    # (10*1,200 + 30*2,400 + 10*3,600) / 12 / 50.
    assert groups["within_federal_limits"]["average_monthly_benefit"] == (
        pytest.approx(200.0)
    )


def test_receiving_unit_shape():
    receiving = _payload()["national"]["receiving"]

    assert receiving["persons_per_unit"] == pytest.approx(2.4)
    # A, D and G are one-person units: 30 of 100.
    assert receiving["single_person_unit_share"] == pytest.approx(0.30)
    # C, D and E hold more than $3,000; G holds exactly $3,000.
    assert SNAP_CASELOAD_ASSET_THRESHOLD == 3_000
    assert receiving["share_with_snap_assets_over_threshold"] == pytest.approx(0.50)


def test_kish_effective_sample_sizes():
    national = _payload()["national"]

    # 100^2 / (100 + 900 + 400 + 100 + 400 + 100).
    assert national["receiving"]["kish_effective_sample_size"] == pytest.approx(5.0)
    # 30^2 / (400 + 100).
    assert national["composition"]["income_failing"][
        "kish_effective_sample_size"
    ] == pytest.approx(1.8)
    # 50^2 / (100 + 900 + 100).
    assert national["composition"]["within_federal_limits"][
        "kish_effective_sample_size"
    ] == pytest.approx(2_500 / 1_100)


def test_reported_receipt_and_the_take_up_fill_split_the_caseload():
    payload = _payload()
    source = payload["national"]["caseload_source"]

    assert payload["reported_receipt_present"] is True

    # A, D and G report receipt.
    assert source["reported_receipt"]["records"] == 3
    assert source["reported_receipt"]["share_of_receiving_units"] == pytest.approx(0.30)
    # B, C and E come from the fill: 70^2 / (900 + 400 + 400).
    assert source["take_up_fill"]["share_of_receiving_units"] == pytest.approx(0.70)
    assert source["take_up_fill"]["kish_effective_sample_size"] == pytest.approx(
        4_900 / 1_700
    )


def test_a_file_without_reported_receipt_says_so():
    """A release that stores no receives_snap reads as all take-up fill."""
    units = _units()
    units["receives_snap"] = False
    payload = _payload(units)
    source = payload["national"]["caseload_source"]

    assert payload["reported_receipt_present"] is False
    assert source["reported_receipt"]["weighted_units"] == 0
    assert source["take_up_fill"]["share_of_receiving_units"] == 1
    assert "stores no receives_snap" in payload["definitions"]["reported_receipt"]


def test_reported_recipients_split_by_what_the_engine_makes_of_them():
    reported = _payload()["national"]["reported_recipients"]

    # A, D, F, G and H report receipt: 10 + 10 + 40 + 10 + 5.
    assert reported["records"] == 5
    assert reported["weighted_units"] == 75
    # A, D and G receive.
    assert reported["modeled_recipient"] == {
        "records": 3,
        "weighted_units": 30,
        "share_of_reported_units": pytest.approx(30 / 75),
    }
    # F is eligible with no benefit.
    assert reported["eligible_non_recipient"] == {
        "records": 1,
        "weighted_units": 40,
        "share_of_reported_units": pytest.approx(40 / 75),
    }
    # H is not eligible in December, whatever its benefit.
    assert reported["not_eligible"] == {
        "records": 1,
        "weighted_units": 5,
        "share_of_reported_units": pytest.approx(5 / 75),
    }


def test_reported_recipients_by_state_and_with_no_reporter():
    payload = _payload()
    states = {row["state_fips"]: row for row in payload["states"]}

    # Alabama's only reporter, A, receives.
    alabama = states["01"]["reported_recipients"]
    assert alabama["weighted_units"] == 10
    assert alabama["modeled_recipient"]["share_of_reported_units"] == 1
    assert alabama["eligible_non_recipient"]["share_of_reported_units"] == 0
    # California: D and G receive (20), F does not (40), H is not eligible (5).
    california = states["06"]["reported_recipients"]
    assert california["weighted_units"] == 65
    assert california["modeled_recipient"]["weighted_units"] == 20
    assert california["eligible_non_recipient"]["weighted_units"] == 40
    assert california["not_eligible"]["weighted_units"] == 5

    units = _units()
    units["receives_snap"] = False
    silent = _payload(units)["national"]["reported_recipients"]
    assert silent["records"] == 0
    assert silent["weighted_units"] == 0
    for group in _REPORTED_GROUPS:
        assert silent[group]["share_of_reported_units"] is None


def test_a_household_code_shared_out_among_its_units_is_refused():
    """A unit's state is its own, never its household's code split by head count.

    A household value mapped to its units by head count gives each unit the
    household's value times the unit's share of the household's members. For a
    California household (06) of four holding units of one and three people
    that is 1.5 and 4.5; for two units of two it is 3 and 3. None is a state.
    """
    units = _units()
    for shared_out in ((1.5, 4.5), (3.0, 3.0)):
        units["spm_unit_state_fips"] = [*shared_out, *([6.0] * (len(units) - 2))]
        with pytest.raises(ValueError, match="state FIPS code"):
            _payload(units)
    units["spm_unit_state_fips"] = 6.0
    assert [row["state_fips"] for row in _payload(units)["states"]] == ["06"]


def test_a_benefit_outside_december_eligibility_reaches_no_share():
    national = _payload()["national"]
    without = _payload(_units([row for row in _ROWS if row[0] != "H"]))["national"]

    assert national["benefit_outside_december_eligibility"] == {
        "records": 1,
        "weighted_units": 5,
        "annual_benefits": 4_500,
    }
    for block in ("take_up", "receiving", "composition", "caseload_source"):
        assert national[block] == without[block]


def test_each_state_uses_its_own_units():
    payload = _payload()
    states = {row["state_fips"]: row for row in payload["states"]}

    assert list(states) == ["01", "06"]
    alabama, california = states["01"], states["06"]
    # A, B, C and I are eligible (85) and flagged; A, B and C receive (60).
    assert alabama["take_up"]["eligible_weighted_units"] == 85
    assert alabama["take_up"]["flagged_share_of_eligible_units"] == 1
    assert alabama["take_up"]["receiving_share_of_eligible_units"] == pytest.approx(
        60 / 85
    )
    assert alabama["composition"]["income_failing"]["share_of_receiving_units"] == (
        pytest.approx(20 / 60)
    )
    assert alabama["composition"]["asset_only"]["share_of_receiving_units"] == 0
    assert alabama["composition"]["asset_only"]["average_monthly_benefit"] is None
    # D, E, F and G are eligible (80); D, E and G receive (40), one per group.
    assert california["take_up"]["receiving_share_of_eligible_units"] == 0.5
    assert california["composition"]["income_failing"]["share_of_receiving_units"] == (
        0.25
    )
    assert california["composition"]["asset_only"]["share_of_receiving_units"] == 0.5
    assert california["benefit_outside_december_eligibility"]["weighted_units"] == 5
    assert payload["states_at_or_above_saturation_threshold"] == {
        "flagged_share_of_eligible_units": ["01"],
        "receiving_share_of_eligible_units": [],
        "receiving_share_of_eligible_persons": [],
    }


def test_states_are_listed_once_a_take_up_share_reaches_the_threshold():
    def unit(weight, benefit, state, *, size=1):
        return (
            "x",
            weight,
            benefit,
            True,
            True,
            True,
            True,
            0.0,
            True,
            False,
            size,
            state,
        )

    # State 2: every eligible unit receives. State 4: 999 of 1,000 eligible
    # units receive, exactly the threshold. State 5: 998 of 1,000. State 6:
    # 999 of 1,000 units, but the unit without a benefit holds four people.
    rows = (
        unit(7.0, 100.0, 2),
        unit(999.0, 100.0, 4),
        unit(1.0, 0.0, 4),
        unit(998.0, 100.0, 5),
        unit(2.0, 0.0, 5),
        unit(999.0, 100.0, 6),
        unit(1.0, 0.0, 6, size=4),
    )
    payload = _payload(_units(rows))

    assert SNAP_SATURATION_THRESHOLD == 0.999
    assert payload["saturation_threshold"] == 0.999
    listed = payload["states_at_or_above_saturation_threshold"]
    assert listed["flagged_share_of_eligible_units"] == ["02", "04", "05", "06"]
    assert listed["receiving_share_of_eligible_units"] == ["02", "04", "06"]
    # State 6: 999 of 1,003 persons.
    assert listed["receiving_share_of_eligible_persons"] == ["02", "04"]


def test_usda_shares_are_labelled_held_out_comparisons():
    payload = _payload(release_id="fixture")
    income, assets, outside = payload["comparisons"]

    assert income["group"] == "income_failing"
    assert income["benchmark_household_share"] == 0.054
    assert income["benchmark_printed_page"] == 72457
    assert income["benchmark_fiscal_year"] == 2024
    assert income["difference_percentage_points"] == pytest.approx(24.6)
    assert income["person_difference_percentage_points"] is None
    assert assets["group"] == "asset_only"
    assert assets["benchmark_household_share"] == 0.115
    assert assets["difference_percentage_points"] == pytest.approx(8.5)
    assert outside["group"] == "outside_federal_limits"
    assert outside["share_of_receiving_units"] == pytest.approx(0.50)
    assert outside["difference_percentage_points"] == pytest.approx(33.1)
    # (50 + 80) / 240 persons against 17.6% of participants.
    assert outside["share_of_receiving_persons"] == pytest.approx(130 / 240)
    assert outside["person_difference_percentage_points"] == pytest.approx(
        100 * (130 / 240 - 0.176)
    )
    assert outside["benchmark_printed_page"] == 72459
    assert payload["release_id"] == "fixture"
    assert payload["status"] == "available"
    assert payload["target_role"] == "validation"
    assert payload["enforced"] is False
    assert payload["used_for_calibration"] is False
    assert "gate" not in payload
    assert payload["benchmark"]["citation"] == "91 FR 72454"
    assert payload["benchmark"]["source_url"] == (
        USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE_URL
    )
    assert "note 25" in payload["benchmark"]["double_counting"]
    assert payload["monthly_values_read_in"] == "December"
    assert "December" in payload["comparability"]
    assert "calibration target" in payload["comparability"]


def test_numeric_flags_are_accepted_and_month_counts_refused():
    booleans = _payload()
    flags = _units()
    for column in (
        "is_snap_eligible",
        "meets_snap_gross_income_test",
        "meets_snap_net_income_test",
        "meets_snap_asset_test",
        "takes_up_snap_if_eligible",
        "receives_snap",
    ):
        flags[column] = flags[column].astype(np.float64)
    assert _payload(flags) == booleans

    # An engine that summed twelve monthly results would report counts.
    counts = _units()
    counts["is_snap_eligible"] = np.where(counts["is_snap_eligible"], 12, 0)
    with pytest.raises(ValueError, match="is_snap_eligible must be boolean"):
        _payload(counts)


def test_no_receiving_unit_reports_null_shares(tmp_path):
    units = _units()
    units["snap"] = 0.0
    payload = _payload(units)
    national = payload["national"]

    assert national["take_up"]["eligible_weighted_units"] == 165
    assert national["take_up"]["receiving_share_of_eligible_units"] == 0
    assert national["receiving"]["weighted_units"] == 0
    assert national["receiving"]["persons_per_unit"] is None
    assert national["receiving"]["kish_effective_sample_size"] is None
    for row in national["composition"].values():
        assert row["weighted_units"] == 0
        assert row["share_of_receiving_units"] is None
        assert row["average_monthly_benefit"] is None
    for row in payload["comparisons"]:
        assert row["share_of_receiving_units"] is None
        assert row["difference_percentage_points"] is None
    path = write_snap_caseload_composition_holdout(payload, tmp_path / "holdout.json")
    assert json.loads(path.read_text()) == payload


def test_no_unit_at_all_is_reported_not_refused():
    """The scoring plan's dry run hands the consumer empty results."""
    payload = _payload(_units(()))

    assert payload["status"] == "available"
    assert payload["states"] == []
    assert payload["national"]["take_up"]["eligible_weighted_units"] == 0
    assert payload["national"]["take_up"]["flagged_share_of_eligible_units"] is None
    json.dumps(payload, allow_nan=False)


def test_records_count_units_with_a_positive_weight():
    rows = list(_ROWS)
    rows.append(("Z", 0.0, 1_200.0, True, True, True, True, 0.0, True, False, 1, 1))
    national = _payload(_units(rows))["national"]

    assert national["receiving"]["records"] == 6
    assert national["take_up"]["eligible_records"] == 8
    assert national["composition"]["within_federal_limits"]["records"] == 3


def test_unavailable_payload_is_observational():
    payload = snap_caseload_composition_holdout_unavailable(
        KeyError("snap_assets"), period=2024, release_id="fixture"
    )

    assert payload["status"] == "unavailable"
    assert payload["enforced"] is False
    assert payload["used_for_calibration"] is False
    assert payload["target_role"] == "validation"
    assert payload["error"] == "KeyError: 'snap_assets'"
    assert payload["release_id"] == "fixture"
    assert payload["source_url"] == USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE_URL


def _with(column, value):
    units = _units()
    units[column] = units[column].astype(object)
    units.loc[0, column] = value
    return units


@pytest.mark.parametrize(
    "units, message",
    [
        (_units().drop(columns="snap_assets"), "snap_assets"),
        (_with("weight", math.nan), "weight must be finite"),
        (_with("weight", -1.0), "weight must be nonnegative"),
        (_with("snap", -1.0), "snap must be nonnegative"),
        (_with("snap", math.inf), "snap must be finite"),
        (_with("snap_assets", math.nan), "snap_assets must be finite"),
        (_with("snap_unit_size", -1), "snap_unit_size must be nonnegative"),
        (_with("meets_snap_asset_test", None), "meets_snap_asset_test must be boolean"),
        (_with("receives_snap", 0.5), "receives_snap must be boolean"),
        (_with("spm_unit_state_fips", 6.5), "state FIPS code"),
        (_with("spm_unit_state_fips", 3), "state FIPS code"),
        (_with("spm_unit_state_fips", 0), "state FIPS code"),
        (_with("spm_unit_state_fips", 100), "state FIPS code"),
    ],
)
def test_malformed_inputs_are_refused_not_silently_reported(units, message):
    with pytest.raises(ValueError, match=message):
        _payload(units)


class _Result:
    def __init__(self, values, weights) -> None:
        self._values = np.asarray(values)
        self.weights = np.asarray(weights, dtype=np.float64)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._values, dtype=dtype)


class _Simulation:
    """Serves the fixture columns and records what it was asked for."""

    def __init__(self, units: pd.DataFrame) -> None:
        self._units = units
        self.requests: list[tuple] = []

    def calculate(self, variable, period=None, map_to=None):
        self.requests.append((variable, period, map_to))
        return _Result(self._units[variable], self._units["weight"])


def test_from_sim_asks_the_scorer_for_each_variable_once_at_the_period():
    simulation = _Simulation(_units())
    payload = snap_caseload_composition_holdout_from_sim(
        simulation, period=2024, release_id="fixture"
    )

    assert simulation.requests == [
        (variable, 2024, None)
        for variable in SNAP_CASELOAD_COMPOSITION_HOLDOUT_VARIABLES
    ]
    assert payload == _payload(release_id="fixture")


# ---------------------------------------------------------------------------
# Property-based invariants
# ---------------------------------------------------------------------------


@st.composite
def _unit_rows(draw, min_size=0, max_size=30):
    n = draw(st.integers(min_value=min_size, max_value=max_size))

    def booleans():
        return draw(st.lists(st.booleans(), min_size=n, max_size=n))

    weights = draw(
        st.lists(
            st.floats(0, 1e4, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    benefits = draw(
        st.lists(st.sampled_from((0.0, 276.0, 3_504.0)), min_size=n, max_size=n)
    )
    return pd.DataFrame(
        {
            "weight": np.asarray(weights, dtype=np.float64),
            "snap": np.asarray(benefits, dtype=np.float64),
            "is_snap_eligible": np.asarray(booleans(), dtype=bool),
            "meets_snap_gross_income_test": np.asarray(booleans(), dtype=bool),
            "meets_snap_net_income_test": np.asarray(booleans(), dtype=bool),
            "meets_snap_asset_test": np.asarray(booleans(), dtype=bool),
            "snap_assets": np.asarray(
                draw(
                    st.lists(
                        st.sampled_from((0.0, 3_000.0, 8_000.0)), min_size=n, max_size=n
                    )
                ),
                dtype=np.float64,
            ),
            "takes_up_snap_if_eligible": np.asarray(booleans(), dtype=bool),
            "receives_snap": np.asarray(booleans(), dtype=bool),
            "snap_unit_size": np.asarray(
                draw(st.lists(st.integers(1, 6), min_size=n, max_size=n)),
                dtype=np.int64,
            ),
            "spm_unit_state_fips": np.asarray(
                draw(st.lists(st.sampled_from((1, 6, 48)), min_size=n, max_size=n)),
                dtype=np.int64,
            ),
        }
    )


def _totals(payload) -> list[float]:
    national = payload["national"]
    totals = [
        national["take_up"]["eligible_weighted_units"],
        national["take_up"]["eligible_weighted_persons"],
        national["take_up"]["flagged_weighted_units"],
        national["receiving"]["weighted_units"],
        national["receiving"]["weighted_persons"],
        national["receiving"]["annual_benefits"],
        national["benefit_outside_december_eligibility"]["weighted_units"],
    ]
    for block in ("composition", "caseload_source"):
        for row in national[block].values():
            totals.extend((row["weighted_units"], row["weighted_persons"]))
    reported = national["reported_recipients"]
    totals.append(reported["weighted_units"])
    totals.extend(reported[group]["weighted_units"] for group in _REPORTED_GROUPS)
    return totals


@settings(max_examples=200, deadline=None)
@given(_unit_rows())
def test_holdout_never_writes_its_inputs(units):
    before = units.copy(deep=True)
    payload = _payload(units)
    pd.testing.assert_frame_equal(units, before)
    json.dumps(payload, allow_nan=False)


@settings(max_examples=200, deadline=None)
@given(_unit_rows(), st.floats(0.5, 4.0, allow_nan=False))
def test_totals_scale_with_the_weights(units, scale):
    reference = _totals(_payload(units))
    scaled_units = units.assign(weight=units["weight"] * scale)
    for total, scaled in zip(reference, _totals(_payload(scaled_units)), strict=True):
        assert scaled == pytest.approx(total * scale, rel=1e-9, abs=1e-3)


@settings(max_examples=200, deadline=None)
@given(_unit_rows(min_size=1), st.randoms(use_true_random=False), st.data())
def test_totals_are_order_invariant_and_additive_over_batches(units, rng, data):
    n = len(units)
    reference = _totals(_payload(units))
    order = list(range(n))
    rng.shuffle(order)
    shuffled = _totals(_payload(units.iloc[order].reset_index(drop=True)))
    split = data.draw(st.integers(0, n))
    first = _totals(_payload(units.iloc[:split].reset_index(drop=True)))
    second = _totals(_payload(units.iloc[split:].reset_index(drop=True)))
    for whole, permuted, left, right in zip(
        reference, shuffled, first, second, strict=True
    ):
        assert permuted == pytest.approx(whole, rel=1e-9, abs=1e-3)
        assert left + right == pytest.approx(whole, rel=1e-9, abs=1e-3)


@settings(max_examples=200, deadline=None)
@given(_unit_rows())
def test_groups_partition_receiving_units_within_the_eligible(units):
    payload = _payload(units)
    for scope in (payload["national"], *payload["states"]):
        take_up, receiving = scope["take_up"], scope["receiving"]
        assert receiving["weighted_units"] <= take_up["eligible_weighted_units"] + 1e-6
        for block in ("composition", "caseload_source"):
            assert sum(
                row["weighted_units"] for row in scope[block].values()
            ) == pytest.approx(receiving["weighted_units"], rel=1e-9, abs=1e-6)
        reported = scope["reported_recipients"]
        assert sum(
            reported[group]["weighted_units"] for group in _REPORTED_GROUPS
        ) == pytest.approx(reported["weighted_units"], rel=1e-9, abs=1e-6)
        assert (
            sum(reported[group]["records"] for group in _REPORTED_GROUPS)
            == (reported["records"])
        )
        share = take_up["receiving_share_of_eligible_units"]
        assert share is None or 0 <= share <= 1 + 1e-9
        if receiving["weighted_units"] > 0:
            assert sum(
                row["share_of_receiving_units"] for row in scope["composition"].values()
            ) == pytest.approx(1.0)
    assert sum(
        row["receiving"]["weighted_units"] for row in payload["states"]
    ) == pytest.approx(payload["national"]["receiving"]["weighted_units"], abs=1e-6)


@settings(max_examples=200, deadline=None)
@given(_unit_rows(min_size=1), st.data())
def test_units_not_eligible_in_december_never_reach_a_share(units, data):
    """Whatever benefit, flag or test result an ineligible unit carries leaves
    every take-up, composition and source figure unchanged."""
    eligible = units["is_snap_eligible"].to_numpy()
    reference = _payload(units)

    def noise(values):
        return np.asarray(
            data.draw(
                st.lists(
                    st.sampled_from(values), min_size=len(units), max_size=len(units)
                )
            )
        )

    perturbed_units = units.copy()
    perturbed_units["snap"] = np.where(eligible, units["snap"], noise((0.0, 99.0)))
    for column in (
        "meets_snap_gross_income_test",
        "meets_snap_net_income_test",
        "meets_snap_asset_test",
        "takes_up_snap_if_eligible",
        "receives_snap",
    ):
        perturbed_units[column] = np.where(
            eligible, units[column], noise((False, True))
        )
    perturbed = _payload(perturbed_units)
    for block in ("take_up", "receiving", "composition", "caseload_source"):
        assert perturbed["national"][block] == reference["national"][block]
    assert perturbed["comparisons"] == reference["comparisons"]


# ---------------------------------------------------------------------------
# Validation only: the USDA shares cannot reach calibration or take-up
# ---------------------------------------------------------------------------


def test_no_ledger_mapping_binds_a_federal_test_measure():
    """Calibration fits SNAP households and benefits, never who passes a test."""
    held_out = set(SNAP_CASELOAD_COMPOSITION_HOLDOUT_VARIABLES) - {"snap"}
    mapped = []
    for mapping in (
        fiscal_targets.DIRECT_LEDGER_TARGETS,
        fiscal_targets.INDICATOR_LEDGER_TARGETS,
    ):
        for key, (base_variable, _, metadata) in mapping.items():
            variables = (
                (base_variable,) if isinstance(base_variable, str) else base_variable
            )
            mapped.append((key, set(variables), dict(metadata)))
    assert any("snap" in variables for _, variables, _ in mapped)
    for key, variables, metadata in mapped:
        assert not variables & held_out, key
        text = json.dumps([key, metadata]).lower()
        assert "categorical" not in text, key
        assert "federal_limit" not in text and "resource_limit" not in text, key


def test_pinned_chronicle_feed_carries_no_caseload_composition_rows():
    """Only a feed re-pin could bring a USDA composition share to calibration."""
    scope = json.loads(
        files("microcosm.build.us").joinpath("chronicle_feed_scope.json").read_text()
    )
    assert scope["pairs"]
    assert any("usda_snap" in json.dumps(pair) for pair in scope["pairs"])
    for pair in scope["pairs"]:
        text = json.dumps(pair).lower()
        assert "categorical" not in text and "quality_control" not in text, pair
        assert "federal_limit" not in text and "resource" not in text, pair


def test_only_the_release_tool_imports_the_holdout():
    """No calibration, take-up, registry or gate module can read the shares."""
    root = paths_for("microcosm-build").repository
    pattern = re.compile(
        r"snap_caseload_composition_holdout"
        r"|USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE"
    )
    module = "snap_caseload_composition_holdout.py"
    readers = sorted(
        path.relative_to(root).as_posix()
        for directory in ("packages", "tools")
        for path in (root / directory).rglob("*.py")
        if "tests" not in path.relative_to(root).parts
        and path.name != module
        and pattern.search(path.read_text(encoding="utf-8"))
    )
    assert readers == ["tools/build_us_fiscal_refresh_release.py"]
