"""Origin support on invented contributions, with the arithmetic done by hand.

The concentration fixtures are built so the statistic lands exactly on a bound:
900 origins of 1 plus 100 origins of 6 give a sum of 1500 and a sum of squares
of 4500, so Kish's effective sample size is 1500^2 / 4500 = 500 exactly, and one
origin of 6 in a total of 1200 is a share of exactly 0.005.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from microcosm.build.cd_benchmark.origin import (
    AcsHouseholdOrigin,
    AsecHouseholdOrigin,
    OriginKeyError,
    SourceMember,
    origin_key,
    origin_payload_bytes,
)
from microcosm.build.cd_benchmark.protocol import canonical_protocol
from microcosm.build.cd_benchmark.reasons import Reason, SupportVerdict
from microcosm.build.cd_benchmark.support import (
    SourceRecord,
    SupportMeasure,
    aggregate_origins,
    bin_support,
    denominator_support,
)

HOUSEHOLDS = SupportMeasure.OCCUPIED_HOUSING_UNITS
PERSONS = SupportMeasure.PERSONS


@pytest.fixture(scope="module")
def bounds():
    return canonical_protocol().support


def flat(count, *, contribution=1, weight=1, prefix="origin"):
    return [
        SourceRecord(
            origin_key=f"{prefix}-{index}", weight=weight, contribution=contribution
        )
        for index in range(count)
    ]


def denominator(bounds, records, **kwargs):
    return denominator_support(
        bounds,
        records,
        block_id="occupied_household_total",
        geography_id="5001900US0601",
        measure_kind=kwargs.pop("measure_kind", HOUSEHOLDS),
    )


def test_the_origin_carrier_floor_is_inclusive(bounds):
    at_floor = denominator(bounds, flat(1000))
    assert at_floor.counts.positive_contribution_origins == 1000
    assert at_floor.verdict is SupportVerdict.WITHIN_BOUNDS
    assert at_floor.reasons == ()
    below = denominator(bounds, flat(999))
    assert below.verdict is SupportVerdict.BOUND_EXCEEDED
    assert below.reasons == (Reason.SUPPORT_BOUND_EXCEEDED,)
    assert below.evidence["breached_bounds"] == ("positive_contribution_origins",)


def test_clones_of_one_source_household_count_once(bounds):
    clones = [
        SourceRecord(origin_key=f"origin-{index % 1000}", weight=1, contribution=0.5)
        for index in range(2000)
    ]
    result = denominator(bounds, clones)
    assert result.counts.actual_records == 2000
    assert result.counts.authenticated_original_origins == 1000
    assert result.counts.positive_weight_records == 2000
    assert result.effective_sample_size == Decimal(1000)
    assert result.verdict is SupportVerdict.WITHIN_BOUNDS


def test_clone_row_identity_never_splits_one_origin():
    member = SourceMember("acs2024/psam_husa.csv#H0001", "a" * 64)
    first = AcsHouseholdOrigin(
        member, survey_vintage=2024, raw_serialno="2024GQ0000123"
    )
    second = AcsHouseholdOrigin(
        member, survey_vintage=2024, raw_serialno="2024GQ0000123"
    )
    assert origin_key(first) == origin_key(second)
    records = [
        SourceRecord(origin_key=origin_key(first), weight=1, contribution=2),
        SourceRecord(origin_key=origin_key(second), weight=1, contribution=3),
    ]
    totals = aggregate_origins(records)
    assert totals.counts.actual_records == 2
    assert totals.counts.authenticated_original_origins == 1
    assert dict(totals.totals) == {origin_key(first): Decimal(5)}


def test_a_different_original_member_is_a_different_origin():
    first = AcsHouseholdOrigin(
        SourceMember("acs2024/psam_husa.csv#H0001", "a" * 64),
        survey_vintage=2024,
        raw_serialno="2024GQ0000123",
    )
    recompressed = AcsHouseholdOrigin(
        SourceMember("acs2024/psam_husa.csv#H0001", "b" * 64),
        survey_vintage=2024,
        raw_serialno="2024GQ0000123",
    )
    assert origin_key(first) != origin_key(recompressed)


def test_the_effective_sample_size_floor_is_inclusive(bounds):
    at_floor = denominator(
        bounds,
        [*flat(900, contribution=1), *flat(100, contribution=6, prefix="heavy")],
    )
    assert at_floor.effective_sample_size == Decimal(500)
    assert at_floor.largest_origin_share == Decimal(6) / Decimal(1500)
    assert at_floor.verdict is SupportVerdict.WITHIN_BOUNDS
    below = denominator(
        bounds,
        [*flat(900, contribution=1), *flat(100, contribution=7, prefix="heavy")],
    )
    assert below.effective_sample_size < Decimal(500)
    assert below.evidence["breached_bounds"] == ("effective_sample_size",)
    assert below.verdict is SupportVerdict.BOUND_EXCEEDED


def test_the_denominator_concentration_ceiling_is_inclusive(bounds):
    at_ceiling = denominator(
        bounds,
        [*flat(1194, contribution=1), SourceRecord("dominant", 1, 6)],
    )
    assert at_ceiling.contribution_total == Decimal(1200)
    assert at_ceiling.largest_origin_share == Decimal("0.005")
    assert at_ceiling.verdict is SupportVerdict.WITHIN_BOUNDS
    over = denominator(
        bounds,
        [*flat(1194, contribution=1), SourceRecord("dominant", 1, 7)],
    )
    assert over.largest_origin_share > Decimal("0.005")
    assert over.evidence["breached_bounds"] == ("largest_origin_share",)


def test_unauthenticated_lineage_leaves_support_undetermined(bounds):
    result = denominator(
        bounds, [*flat(1000), SourceRecord(origin_key=None, weight=0, contribution=0)]
    )
    assert result.reasons == (Reason.UNRESOLVED_SOURCE_LINEAGE,)
    assert result.verdict is SupportVerdict.UNDETERMINED
    assert result.counts.actual_records == 1001


def test_a_zero_weight_record_still_needs_an_authenticated_origin(bounds):
    result = denominator(
        bounds, [*flat(1000), SourceRecord(origin_key="", weight=0, contribution=0)]
    )
    assert Reason.UNRESOLVED_SOURCE_LINEAGE in result.reasons


@pytest.mark.parametrize(
    "record",
    [
        SourceRecord("origin-0", -1, 1),
        SourceRecord("origin-0", 1, -1),
        SourceRecord("origin-0", True, 1),
        SourceRecord("origin-0", 1, float("nan")),
        SourceRecord("origin-0", 1, float("inf")),
    ],
)
def test_invalid_weights_and_masses_are_refused_not_clipped(bounds, record):
    result = denominator(bounds, [*flat(1000), record])
    assert Reason.INVALID_REDUCER_INPUT in result.reasons
    assert result.verdict is SupportVerdict.UNDETERMINED


def test_bin_support_uses_the_per_bin_floor_and_ceiling(bounds):
    at_floor = bin_support(
        bounds,
        flat(50),
        block_id="income_distribution",
        geography_id="5001900US0601",
        bin_id="B19001_002",
        measure_kind=HOUSEHOLDS,
        reference_share_positive=True,
    )
    assert at_floor.verdict is SupportVerdict.WITHIN_BOUNDS
    assert at_floor.largest_origin_share == Decimal("0.02")
    below = bin_support(
        bounds,
        flat(49),
        block_id="income_distribution",
        geography_id="5001900US0601",
        bin_id="B19001_002",
        measure_kind=HOUSEHOLDS,
        reference_share_positive=True,
    )
    assert below.evidence["breached_bounds"] == ("positive_contribution_origins",)
    concentrated = bin_support(
        bounds,
        [*flat(95, contribution=1), SourceRecord("dominant", 1, 6)],
        block_id="income_distribution",
        geography_id="5001900US0601",
        bin_id="B19001_002",
        measure_kind=HOUSEHOLDS,
        reference_share_positive=True,
    )
    assert concentrated.largest_origin_share > Decimal("0.05")
    assert concentrated.evidence["breached_bounds"] == ("largest_origin_share",)


def test_a_zero_mass_bin_fails_the_carrier_floor_without_dividing_by_zero(bounds):
    result = bin_support(
        bounds,
        [SourceRecord("origin-0", 1, 0)],
        block_id="income_distribution",
        geography_id="5001900US0601",
        bin_id="B19001_002",
        measure_kind=HOUSEHOLDS,
        reference_share_positive=True,
    )
    assert result.contribution_total == Decimal(0)
    assert result.effective_sample_size is None
    assert result.largest_origin_share is None
    assert result.verdict is SupportVerdict.BOUND_EXCEEDED
    assert result.reasons == (Reason.SUPPORT_BOUND_EXCEEDED,)
    assert result.evidence["zero_candidate_mass"] is True


def test_the_residual_bin_has_no_carrier_floor(bounds):
    result = bin_support(
        bounds,
        [SourceRecord("origin-0", 1, 0)],
        block_id="income_distribution",
        geography_id="5001900US0601",
        bin_id="unclassified",
        measure_kind=HOUSEHOLDS,
        reference_share_positive=False,
    )
    assert result.verdict is SupportVerdict.NOT_APPLICABLE
    assert result.reasons == ()
    assert result.evidence["carrier_floor"] == "not_applicable_zero_reference_share_bin"


def test_the_effective_sample_size_floor_is_not_applied_per_bin(bounds):
    """Bins carry the carrier count and the share ceiling; the ESS floor is not."""
    concentrated = bin_support(
        bounds,
        [*flat(60, contribution=1), *flat(10, contribution=3, prefix="heavy")],
        block_id="income_distribution",
        geography_id="5001900US0601",
        bin_id="B19001_002",
        measure_kind=HOUSEHOLDS,
        reference_share_positive=True,
    )
    assert concentrated.effective_sample_size < Decimal(500)
    assert concentrated.evidence["ess_floor_applied"] is False
    assert concentrated.verdict is SupportVerdict.WITHIN_BOUNDS


def test_support_never_claims_independence_or_bias(bounds):
    result = denominator(bounds, flat(1000))
    assert result.evidence["independence_or_bias_guarantee"] is False
    assert result.measure is HOUSEHOLDS
    person = denominator(bounds, flat(1000), measure_kind=PERSONS)
    assert person.measure is PERSONS


def test_the_origin_payload_is_four_ordered_typed_pairs():
    acs = AcsHouseholdOrigin(
        SourceMember("acs2024/psam_husa.csv#H0001", "a" * 64),
        survey_vintage=2024,
        raw_serialno=" 2024GQ0000123",
    )
    assert origin_payload_bytes(acs) == (
        b'[["string","ACS_PUMS"],["member",{"canonical_member_id":'
        b'"acs2024/psam_husa.csv#H0001","member_sha256":"'
        + b"a" * 64
        + b'"}],["integer",2024],["string"," 2024GQ0000123"]]'
    )
    asec = AsecHouseholdOrigin(
        SourceMember("asec2025/hhpub25#H000042", "b" * 64),
        income_cohort_year=2024,
        native_h_seq=42,
    )
    assert origin_payload_bytes(asec).endswith(b'["integer",2024],["integer",42]]')


def test_the_origin_key_is_domain_nul_payload():
    import hashlib

    from microcosm.build.cd_benchmark.origin import ORIGIN_DOMAIN

    acs = AcsHouseholdOrigin(
        SourceMember("acs2024/psam_husa.csv#H0001", "a" * 64),
        survey_vintage=2024,
        raw_serialno="2024GQ0000123",
    )
    expected = hashlib.sha256(
        ORIGIN_DOMAIN.encode("utf-8") + b"\x00" + origin_payload_bytes(acs)
    ).hexdigest()
    assert origin_key(acs) == expected
    assert ORIGIN_DOMAIN == "microcosm.statistical_source_household.v1"


def test_the_two_arms_disambiguate_vintage_from_cohort():
    member = SourceMember("member", "c" * 64)
    acs = AcsHouseholdOrigin(member, survey_vintage=2024, raw_serialno="42")
    asec = AsecHouseholdOrigin(member, income_cohort_year=2024, native_h_seq=42)
    # Same year, same "42": the raw SERIALNO stays a string, H_SEQ an integer.
    assert origin_key(acs) != origin_key(asec)


@pytest.mark.parametrize(
    "build",
    [
        lambda m: AcsHouseholdOrigin(m, survey_vintage=True, raw_serialno="1"),
        lambda m: AcsHouseholdOrigin(m, survey_vintage=2024, raw_serialno=42),
        lambda m: AcsHouseholdOrigin(m, survey_vintage=2024, raw_serialno=""),
        lambda m: AsecHouseholdOrigin(m, income_cohort_year=2024, native_h_seq=0),
        lambda m: AsecHouseholdOrigin(m, income_cohort_year=2024, native_h_seq=True),
        lambda m: AsecHouseholdOrigin(m, income_cohort_year=2024, native_h_seq="42"),
    ],
)
def test_typed_origin_components_refuse_coercion(build):
    with pytest.raises(OriginKeyError):
        build(SourceMember("member", "c" * 64))


def test_an_unregistered_arm_has_no_default_encoding():
    with pytest.raises(OriginKeyError):
        origin_key(object())
