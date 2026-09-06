"""Point, categorical and precision checks on invented, hand-calculable values.

Every expected number here is arithmetic anyone can redo on paper: 102 against
100 is exactly two percent, 5 unclassified out of 1000 is exactly the 0.005
maximum, and the subset-share approximation on (400000, 8000, 1000000, 12000)
is exactly 0.64 points.
"""

from __future__ import annotations

import math
from dataclasses import replace
from decimal import Decimal, localcontext

import pytest

from microcosm.build.cd_benchmark.metrics import (
    ARITHMETIC,
    MetricError,
    categorical_check,
    measure,
    point_check,
)
from microcosm.build.cd_benchmark.protocol import ProtocolError, canonical_protocol
from microcosm.build.cd_benchmark.reasons import PrecisionStatus, Reason
from microcosm.build.cd_benchmark.reference import (
    INVALID_CLASSES,
    ReferenceError,
    ReferenceValidity,
    discrepancy_over_moe,
    published_percentage_precision,
    reference_cell,
    reference_validity_reasons,
    relative_moe_precision,
    subset_share_precision,
)


@pytest.fixture(scope="module")
def protocol():
    return canonical_protocol()


def cell(estimate, moe, *, ea=None, ma=None, variable="S0101_C01_001", table="S0101"):
    return reference_cell(
        {
            "E": estimate,
            "M": moe,
            "EA": ea,
            "MA": ma,
            "published_GEO_ID": "5001900US0601",
            "canonical_geography_id": "5001900US0601",
            "dataset": "acs/acs1/subject",
            "table": table,
            "variable": variable,
            "year": 2024,
            "congress": 119,
            "units": "persons",
            "universe": "population",
        }
    )


def tenure(protocol, *, owner, renter, unclassified, denominator, reference):
    return categorical_check(
        protocol.block("tenure_distribution"),
        partition=protocol.partition,
        candidate_bin_masses={"owner": owner, "renter": renter},
        unclassified_mass=unclassified,
        candidate_denominator=denominator,
        reference_counts=reference,
        reference_total=sum(reference.values()),
        geography_id="5001900US0601",
    )


@pytest.mark.parametrize(
    ("candidate", "expected", "within"),
    [
        (100, "0", True),
        (102, "0.02", True),
        (98, "0.02", True),
        (102.5, "0.025", False),
        (0, "1", False),
    ],
)
def test_point_relative_error_uses_an_inclusive_bound(
    protocol, candidate, expected, within
):
    result = point_check(
        protocol.block("population_total"), candidate=candidate, reference=100
    )
    assert result.relative_discrepancy == Decimal(expected)
    assert result.within_bound is within
    assert result.bound == Decimal("0.02")
    if within:
        assert result.reasons == ()
    else:
        assert result.reasons == (Reason.PUBLISHED_BENCHMARK_NONCONFORMITY,)


def test_the_household_bound_is_three_percent_not_two(protocol):
    block = protocol.block("occupied_household_total")
    assert point_check(block, candidate=103, reference=100).within_bound is True
    assert point_check(block, candidate=103.5, reference=100).within_bound is False


def test_a_zero_reference_needs_a_zero_candidate_and_has_no_ratio(protocol):
    block = protocol.block("population_total")
    matched = point_check(block, candidate=0, reference=0)
    assert matched.within_bound is True
    assert matched.relative_discrepancy is None
    assert matched.evidence["zero_reference"] is True
    mismatched = point_check(block, candidate=1, reference=0)
    assert mismatched.within_bound is False
    assert mismatched.relative_discrepancy is None
    assert mismatched.reasons == (Reason.PUBLISHED_BENCHMARK_NONCONFORMITY,)


@pytest.mark.parametrize(
    "value", [-1, True, False, float("nan"), float("inf"), "100", None]
)
def test_invalid_masses_are_refused_not_repaired(protocol, value):
    result = point_check(
        protocol.block("population_total"), candidate=value, reference=100
    )
    assert result.within_bound is None
    assert result.reasons == (Reason.INVALID_REDUCER_INPUT,)


def test_measure_converts_floats_exactly_and_refuses_booleans():
    assert measure(0.5, "x") == Decimal("0.5")
    assert measure(3, "x") == Decimal(3)
    assert measure(0.1, "x") == Decimal(0.1) != Decimal("0.1")
    for bad in (True, False, math.nan, math.inf, "1"):
        with pytest.raises(MetricError):
            measure(bad, "x")


def test_a_complete_partition_at_the_residual_maximum_conforms(protocol):
    result = tenure(
        protocol,
        owner=600,
        renter=395,
        unclassified=5,
        denominator=1000,
        reference={"owner": 600, "renter": 400},
    )
    assert result.unclassified_fraction == Decimal("0.005")
    assert result.total_variation == Decimal("0.005")
    assert result.max_bin_discrepancy_pp == Decimal("0.5")
    assert result.within_bound is True
    assert result.reasons == ()
    assert [item.bin_id for item in result.bins] == ["owner", "renter", "unclassified"]
    residual = result.bins[-1]
    assert residual.reference_share == Decimal("0")
    assert residual.candidate_share == Decimal("0.005")
    assert residual.reference_count is None


def test_the_residual_maximum_is_inclusive_and_the_next_step_is_not(protocol):
    just_over = tenure(
        protocol,
        owner=600,
        renter=394,
        unclassified=6,
        denominator=1000,
        reference={"owner": 600, "renter": 400},
    )
    assert just_over.unclassified_fraction == Decimal("0.006")
    assert Reason.UNCLASSIFIED_FRACTION_EXCEEDED in just_over.reasons
    # The measured discrepancies survive the scope reason rather than vanishing.
    assert just_over.max_bin_discrepancy_pp == Decimal("0.6")
    assert just_over.within_bound is True


def test_residual_mass_is_never_renormalized_away(protocol):
    result = tenure(
        protocol,
        owner=600,
        renter=395,
        unclassified=5,
        denominator=1000,
        reference={"owner": 600, "renter": 400},
    )
    assert result.evidence["renormalize_known_bins"] is False
    # Renormalizing the two known bins to 1000 would make owner 0.6030..., not 0.6.
    assert result.bins[0].candidate_share == Decimal("0.6")
    assert sum(item.candidate_share for item in result.bins) == Decimal("1")


def test_a_per_bin_discrepancy_exactly_at_the_bound_conforms(protocol):
    at_bound = tenure(
        protocol,
        owner=630,
        renter=370,
        unclassified=0,
        denominator=1000,
        reference={"owner": 600, "renter": 400},
    )
    assert at_bound.max_bin_discrepancy_pp == Decimal("3")
    assert at_bound.total_variation == Decimal("0.03")
    assert at_bound.within_bound is True
    over = tenure(
        protocol,
        owner=631,
        renter=369,
        unclassified=0,
        denominator=1000,
        reference={"owner": 600, "renter": 400},
    )
    assert over.max_bin_discrepancy_pp == Decimal("3.1")
    assert over.within_bound is False
    assert over.reasons == (Reason.PUBLISHED_BENCHMARK_NONCONFORMITY,)


def test_owner_and_renter_are_not_reported_as_independent_votes(protocol):
    result = tenure(
        protocol,
        owner=640,
        renter=360,
        unclassified=0,
        denominator=1000,
        reference={"owner": 600, "renter": 400},
    )
    assert result.independent_votes is False
    assert result.reasons.count(Reason.PUBLISHED_BENCHMARK_NONCONFORMITY) == 1
    assert [item.within_bound for item in result.bins] == [False, False, True]


def test_a_partition_that_does_not_close_is_refused(protocol):
    result = tenure(
        protocol,
        owner=600,
        renter=395,
        unclassified=4,
        denominator=1000,
        reference={"owner": 600, "renter": 400},
    )
    assert result.reasons == (Reason.INVALID_REDUCER_INPUT,)
    assert result.evidence["closure_residual"] == "-1"
    assert result.bins == ()
    assert result.total_variation is None


def test_zero_denominators_produce_their_own_reasons_and_no_shares(protocol):
    zero_candidate = tenure(
        protocol,
        owner=0,
        renter=0,
        unclassified=0,
        denominator=0,
        reference={"owner": 600, "renter": 400},
    )
    assert zero_candidate.reasons == (Reason.ZERO_CANDIDATE_DENOMINATOR,)
    assert zero_candidate.evidence["totals_remain_separately_evaluable"] is True
    assert zero_candidate.bins == ()
    zero_reference = tenure(
        protocol,
        owner=600,
        renter=400,
        unclassified=0,
        denominator=1000,
        reference={"owner": 0, "renter": 0},
    )
    assert zero_reference.reasons == (Reason.ZERO_REFERENCE_DENOMINATOR,)
    assert zero_reference.bins == ()


def test_missing_candidate_and_reference_bins_report_different_reasons(protocol):
    block = protocol.block("tenure_distribution")
    missing_candidate = categorical_check(
        block,
        partition=protocol.partition,
        candidate_bin_masses={"owner": 600},
        unclassified_mass=0,
        candidate_denominator=1000,
        reference_counts={"owner": 600, "renter": 400},
        reference_total=1000,
    )
    assert missing_candidate.reasons == (Reason.INVALID_REDUCER_INPUT,)
    assert missing_candidate.evidence["candidate_bins_missing"] == ("renter",)
    missing_reference = categorical_check(
        block,
        partition=protocol.partition,
        candidate_bin_masses={"owner": 600, "renter": 400},
        unclassified_mass=0,
        candidate_denominator=1000,
        reference_counts={"owner": 600},
        reference_total=1000,
    )
    assert Reason.REQUIRED_CELL_OR_DISTRICT_MISSING in missing_reference.reasons
    assert missing_reference.evidence["reference_cells_missing"] == ("renter",)


def test_an_undeclared_reference_cell_is_a_roster_failure(protocol):
    result = categorical_check(
        protocol.block("tenure_distribution"),
        partition=protocol.partition,
        candidate_bin_masses={"owner": 600, "renter": 400},
        unclassified_mass=0,
        candidate_denominator=1000,
        reference_counts={"owner": 600, "renter": 400, "B25003_004": 1},
        reference_total=1001,
    )
    assert result.reasons == (Reason.INVALID_REDUCER_INPUT,)
    assert result.evidence["reference_cells_unknown"] == ("B25003_004",)
    assert result.evidence["reference_cells_missing"] == ()


def test_an_unknown_bin_id_is_not_quietly_accepted(protocol):
    result = categorical_check(
        protocol.block("tenure_distribution"),
        partition=protocol.partition,
        candidate_bin_masses={"owner": 600, "renter": 395, "squatter": 5},
        unclassified_mass=0,
        candidate_denominator=1000,
        reference_counts={"owner": 600, "renter": 400},
        reference_total=1000,
    )
    assert Reason.INVALID_REDUCER_INPUT in result.reasons
    assert result.evidence["candidate_bins_unknown"] == ("squatter",)


def test_the_income_partition_uses_its_own_two_point_bound(protocol):
    block = protocol.block("income_distribution")
    ids = [item.id for item in block.known_bins]
    reference = dict.fromkeys(ids, 100)
    masses = dict.fromkeys(ids, 100)
    masses[ids[0]] = 132
    masses[ids[1]] = 68
    result = categorical_check(
        block,
        partition=protocol.partition,
        candidate_bin_masses=masses,
        unclassified_mass=0,
        candidate_denominator=1600,
        reference_counts=reference,
        reference_total=1600,
    )
    assert result.max_bin_discrepancy_pp == Decimal("2")
    assert result.within_bound is True
    masses[ids[0]] = 133
    masses[ids[1]] = 67
    over = categorical_check(
        block,
        partition=protocol.partition,
        candidate_bin_masses=masses,
        unclassified_mass=0,
        candidate_denominator=1600,
        reference_counts=reference,
        reference_total=1600,
    )
    assert over.max_bin_discrepancy_pp == Decimal("2.0625")
    assert over.within_bound is False


def test_total_variation_can_fail_while_every_bin_passes(protocol):
    """Many small gaps: each within 1 point, together beyond 0.03 variation."""
    block = protocol.block("age_distribution")
    ids = [item.id for item in block.known_bins]
    reference = dict.fromkeys(ids, 1000)
    masses = dict.fromkeys(ids, 1000)
    for index in range(9):
        masses[ids[index]] = 1160
        masses[ids[index + 9]] = 840
    result = categorical_check(
        block,
        partition=protocol.partition,
        candidate_bin_masses=masses,
        unclassified_mass=0,
        candidate_denominator=18000,
        reference_counts=reference,
        reference_total=18000,
    )
    assert all(item.within_bound for item in result.bins)
    assert result.max_bin_discrepancy_pp < Decimal("1")
    assert result.total_variation > Decimal("0.03")
    assert result.within_bound is False
    assert result.reasons == (Reason.PUBLISHED_BENCHMARK_NONCONFORMITY,)


def test_the_partition_refuses_a_block_it_does_not_apply_to(protocol):
    assert protocol.partition.applies_to == {
        "age_distribution",
        "income_distribution",
        "tenure_distribution",
    }
    outside = replace(protocol.block("tenure_distribution"), id="rent_burden")
    with pytest.raises(ProtocolError):
        categorical_check(
            outside,
            partition=protocol.partition,
            candidate_bin_masses={"owner": 1, "renter": 1},
            unclassified_mass=0,
            candidate_denominator=2,
            reference_counts={"owner": 1, "renter": 1},
            reference_total=2,
        )


def test_reference_classification_comes_from_the_shipped_adapter():
    assert cell("100000", "5000").estimate.validity is ReferenceValidity.NUMERIC
    assert cell("100000", "-555555555", ma="*****").moe.validity is (
        ReferenceValidity.CONTROLLED_ESTIMATE
    )
    assert cell("-666666666", "5", ea="-").estimate.validity is (
        ReferenceValidity.INSUFFICIENT_ESTIMATE
    )
    assert cell("5", "-333333333", ma="***").moe.adapter_class == "moe_open_interval"
    assert cell("5", "-333333333", ma="***").moe.validity is (
        ReferenceValidity.OPEN_INTERVAL
    )
    assert cell("5", "1", ea="250,000+").estimate.validity is (
        ReferenceValidity.OPEN_INTERVAL
    )
    assert cell("5", "1", ea="?!").estimate.validity is (
        ReferenceValidity.UNKNOWN_ANNOTATION
    )
    conflict = cell("-666666666", "1", ea="N")
    assert conflict.estimate.validity is ReferenceValidity.ANNOTATION_CONFLICT
    assert conflict.estimate.annotation_conflict is True


def test_invalid_estimate_classes_make_the_reference_invalid():
    assert reference_validity_reasons(cell("100", "5")) == ()
    for estimate, annotation in (("-666666666", "-"), ("-999999999", "N"), ("", None)):
        invalid = cell(estimate, "5", ea=annotation)
        assert invalid.estimate.validity in INVALID_CLASSES
        assert reference_validity_reasons(invalid) == (Reason.REFERENCE_INVALID,)


def test_a_reference_cell_needs_every_required_raw_field():
    with pytest.raises(ReferenceError):
        reference_cell({"E": "1", "M": "0"})


def test_a_controlled_margin_is_its_own_class_not_an_adequacy_claim():
    result = relative_moe_precision(
        cell("700000", "-555555555", ma="*****"), ceiling=Decimal("0.05")
    )
    assert result.status is PrecisionStatus.CONTROLLED_UNKNOWN
    assert result.status is not PrecisionStatus.ADEQUATE
    assert result.measured is None
    assert result.reasons == ()
    assert result.evidence["numeric_moe"] is None
    assert result.evidence["raw_moe"] == "-555555555"


def test_relative_precision_is_inclusive_at_the_ceiling():
    assert (
        relative_moe_precision(cell("100000", "5000"), ceiling=Decimal("0.05")).status
        is PrecisionStatus.ADEQUATE
    )
    over = relative_moe_precision(cell("100000", "5001"), ceiling=Decimal("0.05"))
    assert over.status is PrecisionStatus.INADEQUATE
    assert over.reasons == (Reason.REFERENCE_PRECISION_INADEQUATE,)


def test_a_zero_estimate_has_no_relative_precision_rather_than_a_perfect_one():
    result = relative_moe_precision(cell("0", "0"), ceiling=Decimal("0.05"))
    assert result.status is PrecisionStatus.UNAVAILABLE
    assert result.evidence["zero_reference_estimate"] is True


def test_precision_never_widens_a_point_bound(protocol):
    """The same masses give the same verdict whatever the published margin is."""
    block = protocol.block("population_total")
    verdict = point_check(block, candidate=102.5, reference=100)
    assert verdict.within_bound is False
    for moe in ("0", "1", "50000", "-555555555"):
        annotation = "*****" if moe == "-555555555" else None
        published = cell("100", moe, ma=annotation)
        precision = relative_moe_precision(published, ceiling=Decimal("0.05"))
        assert precision.status is not PrecisionStatus.ADEQUATE or moe != "50000"
        assert point_check(block, candidate=102.5, reference=100) == verdict


def test_published_percentage_precision_reads_the_printed_points(protocol):
    age = protocol.block("age_distribution")
    at_bound = cell("6.4", "1.0", variable="S0101_C02_002")
    assert (
        published_percentage_precision(
            at_bound, ceiling=age.reference_moe_ceiling_pp
        ).status
        is PrecisionStatus.ADEQUATE
    )
    over = cell("6.4", "1.1", variable="S0101_C02_002")
    result = published_percentage_precision(over, ceiling=age.reference_moe_ceiling_pp)
    assert result.status is PrecisionStatus.INADEQUATE
    assert result.evidence["published_percentage"] == "6.4"


def test_the_subset_share_approximation_is_the_documented_formula():
    numerator = cell("400000", "8000", variable="B25003_003", table="B25003")
    denominator = cell("1000000", "12000", variable="B25003_001", table="B25003")
    result = subset_share_precision(numerator, denominator, ceiling=Decimal("3"))
    # 100 * sqrt(8000^2 - 0.4^2 * 12000^2) / 1000000 = 100 * 6400 / 1000000.
    assert result.measured == Decimal("0.64")
    assert result.status is PrecisionStatus.ADEQUATE
    assert result.evidence["negative_radicand_plus_fallback"] is False


def test_a_negative_radicand_uses_the_documented_plus_fallback():
    numerator = cell("400000", "100", variable="B25003_003", table="B25003")
    denominator = cell("1000000", "12000", variable="B25003_001", table="B25003")
    result = subset_share_precision(numerator, denominator, ceiling=Decimal("3"))
    assert result.evidence["negative_radicand_plus_fallback"] is True
    # 100 * sqrt(100^2 + 0.4^2 * 12000^2) / 1000000 = 100 * sqrt(23050000) / 1e6.
    with localcontext(ARITHMETIC):
        expected = Decimal(100) * Decimal(23050000).sqrt() / Decimal(1000000)
    assert result.measured == expected


def test_a_controlled_or_missing_input_fabricates_no_approximation():
    numerator = cell("400000", "8000", variable="B25003_003", table="B25003")
    controlled = cell(
        "1000000", "-555555555", ma="*****", variable="B25003_001", table="B25003"
    )
    result = subset_share_precision(numerator, controlled, ceiling=Decimal("3"))
    assert result.status is PrecisionStatus.UNAVAILABLE
    assert result.measured is None
    assert result.evidence["no_fabricated_approximation"] is True
    assert result.reasons == (Reason.REFERENCE_PRECISION_INADEQUATE,)


def test_discrepancy_over_margin_is_diagnostic_and_often_undefined():
    assert discrepancy_over_moe(Decimal(1000), cell("100000", "5000")) == Decimal("0.2")
    assert discrepancy_over_moe(Decimal(1000), cell("100000", "0")) is None
    assert (
        discrepancy_over_moe(Decimal(1000), cell("100000", "-555555555", ma="*****"))
        is None
    )
