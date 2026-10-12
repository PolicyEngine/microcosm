"""Held-out Medicare IRMAA reporting without a country engine (microcosm#1206).

Invariants:

* The holdout never changes weights or any other input: it reads read-only
  arrays and returns a new payload.
* Weighted payers and premiums are linear in the weights, invariant to person
  order, and additive over any split of the persons (household batches).
* ``0 <= payers <= weighted enrollees`` for each Part; premiums are
  nonnegative, and zero when no weight pays.
* Non-enrollees never reach either Part's measure, whatever premium or
  surcharge the engine assigns them; a Part D surcharge outside enrollment
  is reported, not counted.
* With no weighted enrollee or no weighted payer, the Trustees comparisons
  are withheld (never reported as -100%) and a reason is given.
* The Trustees rows are validation only: no ledger mapping binds an IRMAA
  measure, the pinned Chronicle feed carries no income-related premium record
  set, and only the release tool imports the holdout.
"""

import json
import math
import re
from importlib.resources import files

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime import fiscal_targets
from microcosm.build.us_runtime.medicare_irmaa_holdout import (
    MEDICARE_IRMAA_HOLDOUT_VARIABLES,
    MEDICARE_IRMAA_TRUSTEES_2026,
    TRUSTEES_REPORT_2026_URL,
    medicare_irmaa_holdout_payload,
    medicare_irmaa_holdout_unavailable,
    write_medicare_irmaa_holdout,
)
from test_support.paths import paths_for

# 2024 standard Part B premium and the tier 1 and tier 5 IRMAA amounts from
# Trustees Report Table V.E3 (monthly), and the Part D tier 1 amount (V.E4).
_BASE = 12 * 174.70
_PART_B_TIER_1 = 12 * 69.90
_PART_B_TIER_5 = 12 * 419.30
_PART_D_TIER_1 = 12 * 12.90


def _payload(enrolled, gross, base, part_d, weights, *, period=2024, **kwargs):
    return medicare_irmaa_holdout_payload(
        np.asarray(enrolled),
        np.asarray(gross, dtype=np.float64),
        np.asarray(base, dtype=np.float64),
        np.asarray(part_d, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        period=period,
        **kwargs,
    )


def test_trustees_2026_rows_match_tables_v_e3_and_v_e4():
    """CY2024 historical rows: V.E3 p. 209 (PDF p. 215), V.E4 p. 212 (PDF p. 218)."""
    rows = MEDICARE_IRMAA_TRUSTEES_2026[2024]
    part_b, part_d = rows["part_b"], rows["part_d"]
    assert (part_b.table, part_b.printed_page, part_b.pdf_page) == ("V.E3", 209, 215)
    assert (part_d.table, part_d.printed_page, part_d.pdf_page) == ("V.E4", 212, 218)
    assert part_b.beneficiaries_affected == 4_800_000
    assert part_b.aggregate_premiums == 12_800_000_000
    assert part_d.beneficiaries_affected == 4_200_000
    assert part_d.aggregate_premiums == 2_200_000_000
    assert part_b.average_monthly_premium_per_payer == pytest.approx(222.2222222)
    assert part_d.average_monthly_premium_per_payer == pytest.approx(43.6507937)
    assert "Part B standard premium" in part_b.aggregate_premiums_definition
    assert "Part D plan premium" in part_d.aggregate_premiums_definition
    assert sorted(MEDICARE_IRMAA_TRUSTEES_2026) == [2024]
    with pytest.raises(TypeError):
        MEDICARE_IRMAA_TRUSTEES_2026[2025] = rows
    with pytest.raises(TypeError):
        rows["part_c"] = part_b


def test_holdout_variables_are_the_engine_measures_in_request_order():
    assert MEDICARE_IRMAA_HOLDOUT_VARIABLES == (
        "medicare_enrolled",
        "gross_medicare_part_b_premium_if_enrolled",
        "base_part_b_premium",
        "income_adjusted_part_d_premium_surcharge",
    )


def test_payers_and_premiums_count_only_enrollees_for_part_b():
    # Enrollee in tier 5; enrollee paying no IRMAA; eligible non-enrollee whose
    # eligibility-gated base premium must not count; someone not eligible.
    payload = _payload(
        [True, True, False, False],
        [_BASE + _PART_B_TIER_5, _BASE, 0.0, 0.0],
        [_BASE, _BASE, _BASE, 0.0],
        [_PART_D_TIER_1, 0.0, 0.0, 0.0],
        [2.0, 3.0, 5.0, 7.0],
        release_id="fixture",
    )
    part_b, part_d = payload["parts"]
    assert part_b["part"] == "part_b"
    assert part_b["weighted_payers"] == 2
    assert part_b["weighted_aggregate_premiums"] == pytest.approx(2 * _PART_B_TIER_5)
    assert part_b["average_monthly_premium_per_payer"] == pytest.approx(419.30)
    assert part_b["benchmark_payers"] == 4_800_000
    assert part_b["payers_relative_difference"] == pytest.approx(2 / 4_800_000 - 1)
    assert part_b["aggregate_premiums_relative_difference"] == pytest.approx(
        2 * _PART_B_TIER_5 / 12_800_000_000 - 1
    )
    assert part_b["benchmark_table"] == "V.E3"
    assert part_d["part"] == "part_d"
    assert part_d["weighted_payers"] == 2
    assert part_d["weighted_aggregate_premiums"] == pytest.approx(2 * _PART_D_TIER_1)
    assert part_d["benchmark_table"] == "V.E4"
    assert part_d["outside_enrollment_weighted_amount"] == 0
    assert payload["weighted_medicare_enrollees"] == 5
    assert "reason" not in payload
    assert payload["release_id"] == "fixture"
    assert payload["status"] == "available"
    assert payload["target_role"] == "validation"
    assert payload["enforced"] is False
    assert payload["used_for_calibration"] is False
    assert "gate" not in payload
    assert payload["benchmark"]["source_url"] == TRUSTEES_REPORT_2026_URL
    assert payload["benchmark"]["year"] == 2024
    assert payload["benchmark"]["tables"] == ["V.E3", "V.E4"]
    assert "ever-in-year" in payload["comparability"]
    assert "#10115" in payload["comparability"]


def test_a_part_d_surcharge_outside_enrollment_is_reported_not_counted():
    payload = _payload(
        [True, False],
        [_BASE + _PART_B_TIER_1, 0.0],
        [_BASE, _BASE],
        [_PART_D_TIER_1, _PART_D_TIER_1],
        [2.0, 3.0],
    )
    part_d = payload["parts"][1]
    assert part_d["weighted_payers"] == 2
    assert part_d["weighted_aggregate_premiums"] == pytest.approx(2 * _PART_D_TIER_1)
    assert part_d["outside_enrollment_weighted_amount"] == pytest.approx(
        3 * _PART_D_TIER_1
    )


def test_no_weighted_payer_withholds_the_trustees_comparisons():
    """Enrollees but no income-related premium (pe-us 2.2.1 on a 2024 H5)."""
    payload = _payload(
        [True, True, False], [_BASE, _BASE, 0.0], [_BASE] * 3, [0.0] * 3, [4, 5, 6]
    )
    assert payload["status"] == "no_payers"
    assert "two years" in payload["reason"]
    assert "#10106" in payload["reason"]
    assert payload["weighted_medicare_enrollees"] == 9
    for row in payload["parts"]:
        assert row["weighted_payers"] == 0
        assert row["weighted_aggregate_premiums"] == 0
        assert row["benchmark_payers"] is not None
        assert row["payers_relative_difference"] is None
        assert row["aggregate_premiums_relative_difference"] is None
    assert payload["enforced"] is False
    assert payload["used_for_calibration"] is False


def test_no_weighted_enrollee_withholds_the_trustees_comparisons():
    for enrolled, weights in (([False], [1.0]), ([True], [0.0]), ([], [])):
        n = len(enrolled)
        payload = _payload(
            enrolled, [_BASE + _PART_B_TIER_1] * n, [_BASE] * n, [0.0] * n, weights
        )
        assert payload["status"] == "no_medicare_enrollees"
        assert payload["weighted_medicare_enrollees"] == 0
        for row in payload["parts"]:
            assert row["payers_relative_difference"] is None
            assert row["aggregate_premiums_relative_difference"] is None


def test_numeric_enrollment_flags_are_accepted_and_others_refused():
    flags = _payload(
        [1, 0], [_BASE + _PART_B_TIER_1, 0.0], [_BASE, _BASE], [0, 0], [1, 1]
    )
    booleans = _payload(
        [True, False], [_BASE + _PART_B_TIER_1, 0.0], [_BASE, _BASE], [0, 0], [1, 1]
    )
    assert flags == booleans
    with pytest.raises(ValueError, match="boolean"):
        _payload([2, 0], [0, 0], [0, 0], [0, 0], [1, 1])


def test_a_period_without_a_trustees_row_reports_null_comparisons(tmp_path):
    payload = _payload(
        [True], [_BASE + _PART_B_TIER_1], [_BASE], [0.0], [4.0], period=2025
    )
    assert payload["benchmark"]["year"] is None
    assert payload["benchmark"]["available_years"] == [2024]
    for row in payload["parts"]:
        assert row["benchmark_payers"] is None
        assert row["benchmark_aggregate_premiums"] is None
        assert row["payers_relative_difference"] is None
        assert row["aggregate_premiums_relative_difference"] is None
    part_b, part_d = payload["parts"]
    assert part_b["weighted_payers"] == 4
    assert part_d["weighted_payers"] == 0
    assert part_d["average_monthly_premium_per_payer"] is None
    path = write_medicare_irmaa_holdout(payload, tmp_path / "diagnostic.json")
    assert json.loads(path.read_text()) == payload


def test_unavailable_payload_is_observational():
    payload = medicare_irmaa_holdout_unavailable(
        KeyError("base_part_b_premium"), period=2024, release_id="fixture"
    )
    assert payload["status"] == "unavailable"
    assert payload["enforced"] is False
    assert payload["used_for_calibration"] is False
    assert payload["target_role"] == "validation"
    assert payload["error"] == "KeyError: 'base_part_b_premium'"
    assert payload["release_id"] == "fixture"


@pytest.mark.parametrize(
    "arrays, message",
    [
        (([True, False], [0.0], [0.0, 0.0], [0.0, 0.0], [1, 1]), "aligned"),
        (([True], [math.nan], [0.0], [0.0], [1]), "finite"),
        (([True], [0.0], [0.0], [0.0], [-1]), "nonnegative weights"),
        (([True], [_BASE], [_BASE], [-1.0], [1]), "nonnegative"),
        (([True], [_BASE - 1], [_BASE], [0.0], [1]), "below"),
    ],
)
def test_malformed_inputs_are_refused_not_silently_reported(arrays, message):
    with pytest.raises(ValueError, match=message):
        _payload(*arrays)


# ---------------------------------------------------------------------------
# Property-based invariants
# ---------------------------------------------------------------------------

_TIERS = (0.0, 69.90, 174.70, 279.50, 384.30, 419.30)
_PART_D_TIERS = (0.0, 12.90, 33.30, 53.80, 74.20, 81.00)


@st.composite
def _people(draw, min_size=0, max_size=40):
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    eligible = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    takes_up = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    tiers = draw(st.lists(st.integers(0, 5), min_size=n, max_size=n))
    weights = draw(
        st.lists(
            st.floats(0, 1e4, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    enrolled = np.asarray(eligible, dtype=bool) & np.asarray(takes_up, dtype=bool)
    tier = np.asarray(tiers, dtype=np.int64)
    base = np.where(np.asarray(eligible, dtype=bool), _BASE, 0.0)
    irmaa = 12 * np.asarray(_TIERS)[tier]
    gross = np.where(enrolled, base + irmaa, 0.0)
    part_d = np.where(enrolled, 12 * np.asarray(_PART_D_TIERS)[tier], 0.0)
    return enrolled, gross, base, part_d, np.asarray(weights, dtype=np.float64)


def _totals(payload):
    return [
        (row["weighted_payers"], row["weighted_aggregate_premiums"])
        for row in payload["parts"]
    ]


@settings(max_examples=200, deadline=None)
@given(_people())
def test_holdout_never_writes_its_inputs(people):
    arrays = [np.array(values, copy=True) for values in people]
    for values in arrays:
        values.flags.writeable = False
    payload = _payload(*arrays)
    for before, after in zip(people, arrays, strict=True):
        np.testing.assert_array_equal(before, after)
    json.dumps(payload, allow_nan=False)


@settings(max_examples=200, deadline=None)
@given(_people(), st.floats(0.5, 4.0, allow_nan=False))
def test_totals_scale_with_the_weights(people, scale):
    enrolled, gross, base, part_d, weights = people
    reference = _totals(_payload(enrolled, gross, base, part_d, weights))
    scaled = _totals(_payload(enrolled, gross, base, part_d, weights * scale))
    for (payers, premiums), (scaled_payers, scaled_premiums) in zip(
        reference, scaled, strict=True
    ):
        assert scaled_payers == pytest.approx(payers * scale, rel=1e-9, abs=1e-6)
        assert scaled_premiums == pytest.approx(premiums * scale, rel=1e-9, abs=1e-3)


@settings(max_examples=200, deadline=None)
@given(_people(min_size=1), st.randoms(use_true_random=False), st.data())
def test_totals_are_order_invariant_and_additive_over_batches(people, rng, data):
    enrolled, gross, base, part_d, weights = people
    n = len(weights)
    reference = _totals(_payload(*people))
    order = list(range(n))
    rng.shuffle(order)
    shuffled = _totals(_payload(*(values[order] for values in people)))
    split = data.draw(st.integers(0, n))
    first = _totals(_payload(*(values[:split] for values in people)))
    second = _totals(_payload(*(values[split:] for values in people)))
    for whole, permuted, left, right in zip(
        reference, shuffled, first, second, strict=True
    ):
        for index in range(2):
            assert permuted[index] == pytest.approx(whole[index], rel=1e-9, abs=1e-3)
            assert left[index] + right[index] == pytest.approx(
                whole[index], rel=1e-9, abs=1e-3
            )


@settings(max_examples=200, deadline=None)
@given(_people())
def test_payers_and_premiums_are_bounded_by_enrollment(people):
    enrolled, gross, base, part_d, weights = people
    payload = _payload(*people)
    weighted_enrollees = float(weights[enrolled].sum())
    for row in payload["parts"]:
        assert 0 <= row["weighted_payers"] <= weighted_enrollees + 1e-6
        assert row["weighted_aggregate_premiums"] >= 0
        if row["weighted_payers"] == 0:
            assert row["weighted_aggregate_premiums"] == 0
            assert row["average_monthly_premium_per_payer"] is None


@settings(max_examples=200, deadline=None)
@given(_people(min_size=1), st.data())
def test_non_enrollees_never_reach_either_measure(people, data):
    """Whatever premiums a non-enrollee carries leave both Parts' totals,
    comparisons and status unchanged; their Part D surcharge is reported."""
    enrolled, gross, base, part_d, weights = people
    reference = _payload(*people)

    def noise():
        return np.asarray(
            data.draw(
                st.lists(
                    st.floats(0, 1e5, allow_nan=False),
                    min_size=len(base),
                    max_size=len(base),
                )
            )
        )

    surcharge = noise()
    perturbed = _payload(
        enrolled,
        np.where(enrolled, gross, noise()),
        np.where(enrolled, base, noise()),
        np.where(enrolled, part_d, surcharge),
        weights,
    )
    assert perturbed["status"] == reference["status"]
    assert perturbed["parts"][0] == reference["parts"][0]
    outside = perturbed["parts"][1].pop("outside_enrollment_weighted_amount")
    reference["parts"][1].pop("outside_enrollment_weighted_amount")
    assert perturbed["parts"][1] == reference["parts"][1]
    assert outside == pytest.approx(
        float((weights * np.where(enrolled, 0.0, surcharge)).sum()),
        rel=1e-9,
        abs=1e-6,
    )


@settings(max_examples=200, deadline=None)
@given(_people())
def test_comparisons_are_reported_exactly_when_some_enrollee_pays(people):
    enrolled, gross, base, part_d, weights = people
    payload = _payload(*people)
    payers = (gross - base > 0) | (part_d > 0)
    pays = bool(weights[enrolled & payers].sum() > 0)
    assert (payload["status"] == "available") == pays
    for row in payload["parts"]:
        assert (row["payers_relative_difference"] is not None) == pays


# ---------------------------------------------------------------------------
# Validation only: the Trustees IRMAA rows cannot reach calibration
# ---------------------------------------------------------------------------


def test_no_ledger_mapping_binds_an_irmaa_measure():
    irmaa_only = {"base_part_b_premium", "income_adjusted_part_d_premium_surcharge"}
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
    for key, variables, metadata in mapped:
        assert not variables & irmaa_only, key
        assert "irmaa" not in json.dumps([key, metadata]).lower(), key
        assert "income_related" not in json.dumps([key, metadata]).lower(), key


def test_pinned_chronicle_feed_carries_no_income_related_premium_rows():
    """Only a feed re-pin could bring a Trustees IRMAA row to calibration."""
    scope = json.loads(
        files("microcosm.build.us").joinpath("chronicle_feed_scope.json").read_text()
    )
    assert scope["pairs"]
    for pair in scope["pairs"]:
        text = json.dumps(pair).lower()
        assert "irmaa" not in text and "income_related" not in text, pair


def test_only_the_release_tool_imports_the_holdout():
    """No calibration, registry or gate module can read the Trustees rows."""
    root = paths_for("microcosm-build").repository
    pattern = re.compile(r"medicare_irmaa_holdout|MEDICARE_IRMAA_TRUSTEES_2026")
    module = "medicare_irmaa_holdout.py"
    readers = sorted(
        str(path.relative_to(root))
        for directory in ("packages", "tools")
        for path in (root / directory).rglob("*.py")
        if "/tests/" not in str(path)
        and path.name != module
        and pattern.search(path.read_text(encoding="utf-8"))
    )
    assert readers == ["tools/build_us_fiscal_refresh_release.py"]
