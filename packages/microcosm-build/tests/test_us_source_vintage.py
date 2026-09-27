"""Restamped Chronicle facts are read at their data year (chronicle#117).

The compile-level cases (a restamped W-2 tips amount ageing from TY2020, the
refusals of an unreviewed or shadowing restamp) and the pinned-feed register
check live in ``test_us_fiscal_targets.py`` beside the other W-2 aging tests.
"""

from __future__ import annotations

import json

import pytest

from microcosm.build.us_runtime.source_vintage import (
    US_LATER_PERIOD_OBSERVATION_EXEMPTIONS,
    US_RESTAMPED_SOURCE_PACKAGES,
    LaterPeriodObservationExemption,
    RestampedAgingIndexError,
    RestampedSourcePackage,
    SourceVintageCorrection,
    UnreviewedRestampError,
    apply_source_vintage_corrections,
    check_restamps_stay_out_of_aging_indexes,
    detect_restamped_facts,
    fact_artifact_year,
    source_vintage_corrections,
)
from microcosm.calibrate import TargetRegistry, TargetSpec

_W2 = "soi-w2-statistics-2020"
_W2_SHA = US_RESTAMPED_SOURCE_PACKAGES[_W2].source_sha256
_OTHER_SHA = "0" * 64
_TIPS_2023 = (
    "irs_soi.ty2023.form_w2_social_security_tips.box_7_social_security_tips.amount"
)


def _raw_key(package_id: str, artifact_year: int | str, file: str = "f.xlsx") -> str:
    return f"raw/irs_soi/{package_id}/{artifact_year}/{_OTHER_SHA}/{file}"


def _fact(
    source_record_id: str,
    *,
    period: int | str,
    raw_r2_key: str | None,
    assertion: str | None = "observation",
    sha: str | None = None,
) -> dict[str, object]:
    source: dict[str, object] = {}
    if raw_r2_key is not None:
        source["raw_r2_key"] = raw_r2_key
    if sha is not None:
        source["source_sha256"] = sha
    fact: dict[str, object] = {
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{source_record_id}@{period}",
        "lineage": {"source_record_id": source_record_id},
        "period": {"type": "tax_year", "value": period},
        "source": source,
    }
    if assertion is not None:
        fact["assertion"] = assertion
    return fact


def _spec(name: str, value: float = 1.0, **metadata: str) -> TargetSpec:
    return TargetSpec(
        name=name,
        entity="tax_unit",
        value=value,
        measure="tip_income",
        period=2024,
        source="test",
        family="irs_soi",
        metadata=metadata,
    )


def _entry(package_id: str, data_year: int, *stamps: int, sha: str = "s"):
    return RestampedSourcePackage(
        package_id, data_year, frozenset(stamps), "f.xlsx", sha, "reason"
    )


def test_fact_artifact_year_reads_the_raw_key_package_and_year() -> None:
    assert fact_artifact_year(
        _fact("x", period=2023, raw_r2_key=_raw_key(_W2, 2020))
    ) == (_W2, 2020)
    for key in (
        None,
        "",
        "raw/irs_soi",
        f"raw/irs_soi/{_W2}/source_capture/{_OTHER_SHA}/f.xlsx",
        f"raw/irs_soi/{_W2}/20/{_OTHER_SHA}/f.xlsx",
        f"staged/irs_soi/{_W2}/2020/{_OTHER_SHA}/f.xlsx",
        f"raw/irs_soi//2020/{_OTHER_SHA}/f.xlsx",
    ):
        assert fact_artifact_year(_fact("x", period=2023, raw_r2_key=key)) is None


def test_detect_restamped_facts_flags_only_observations_stamped_after_their_data() -> (
    None
):
    facts = [
        _fact("restamp", period=2023, raw_r2_key=_raw_key(_W2, 2020)),
        _fact("legacy", period=2023, raw_r2_key=_raw_key(_W2, 2020), assertion=None),
        _fact("same_year", period=2020, raw_r2_key=_raw_key(_W2, 2020)),
        # A 2024 release carrying cy2023 rows is the ordinary direction.
        _fact("later_release", period=2023, raw_r2_key=_raw_key("bea-nipa", 2024)),
        # A publisher's projection of a later year is not a restamp.
        _fact(
            "projection",
            period=2027,
            raw_r2_key=_raw_key("jct-obbba", 2025),
            assertion="source_projection",
        ),
        _fact("no_key", period=2023, raw_r2_key=None),
        _fact("month_label", period="2024-12", raw_r2_key=_raw_key("cms", 2026)),
    ]
    detected = detect_restamped_facts(facts)
    assert [row[0]["lineage"]["source_record_id"] for row in detected] == [
        "restamp",
        "legacy",
    ]
    assert {row[1:] for row in detected} == {(_W2, 2020, 2023)}


def test_the_content_rule_detects_a_registered_file_under_any_key_shape() -> None:
    # A Chronicle key-layout change (for example a country segment) must not
    # hide a registered restamp: the pinned file's digest still names it.
    moved_key = f"raw/us/irs_soi/{_W2}/2020/{_W2_SHA}/20in04w2all.xlsx"
    detected = detect_restamped_facts(
        [
            _fact(_TIPS_2023, period=2023, raw_r2_key=moved_key, sha=_W2_SHA),
            _fact("no_key", period=2023, raw_r2_key=None, sha=_W2_SHA),
            # At its data year the registered file is stamped truthfully.
            _fact("honest", period=2020, raw_r2_key=None, sha=_W2_SHA),
        ]
    )
    assert [(row[0]["lineage"]["source_record_id"], *row[1:]) for row in detected] == [
        (_TIPS_2023, _W2, 2020, 2023),
        ("no_key", _W2, 2020, 2023),
    ]
    assert len(source_vintage_corrections([row[0] for row in detected])) == 2


def test_a_reviewed_exemption_admits_a_truthful_later_observation() -> None:
    # BEA's 2024-keyed SAINC.zip carries a 2025 column; once Chronicle builds
    # it truthfully, the structural rule alone would flag it.
    fact = _fact(
        "bea_regional.cy2025.x", period=2025, raw_r2_key=_raw_key("bea-sainc", 2024)
    )
    assert detect_restamped_facts([fact])
    exemptions = {
        "bea-sainc": LaterPeriodObservationExemption(
            "bea-sainc", frozenset({2025}), "multi-year release"
        )
    }
    assert not detect_restamped_facts([fact], exemptions=exemptions)
    assert not source_vintage_corrections([fact], exemptions=exemptions)
    # The exemption covers its reviewed periods only.
    later = _fact(
        "bea_regional.cy2026.x", period=2026, raw_r2_key=_raw_key("bea-sainc", 2024)
    )
    assert detect_restamped_facts([later], exemptions=exemptions)
    assert US_LATER_PERIOD_OBSERVATION_EXEMPTIONS == {}


def test_register_entries_describe_one_earlier_year_each() -> None:
    assert set(US_RESTAMPED_SOURCE_PACKAGES) == {
        "soi-w2-statistics-2020",
        "soi-congressional-district-2022",
        "soi-state-2022",
        "soi-ira-roth-contributions-2022",
        "soi-ira-traditional-contributions-2022",
    }
    digests = [entry.source_sha256 for entry in US_RESTAMPED_SOURCE_PACKAGES.values()]
    assert len(set(digests)) == len(digests)
    for package_id, entry in US_RESTAMPED_SOURCE_PACKAGES.items():
        assert entry.package_id == package_id
        assert entry.stamped_periods
        assert all(stamp > entry.data_year for stamp in entry.stamped_periods)
        # IRS SOI file names lead with the two-digit tax year (20in04w2all).
        assert int(entry.source_file[:2]) == entry.data_year % 100
        assert package_id.endswith(str(entry.data_year))
        assert len(entry.source_sha256) == 64
        assert entry.reason


def test_corrections_map_each_registered_restamp_to_its_data_year() -> None:
    corrections = source_vintage_corrections(
        [
            _fact(_TIPS_2023, period=2023, raw_r2_key=_raw_key(_W2, 2020)),
            _fact(
                "irs_soi.ty2020.form_w2_social_security_tips.x.amount",
                period=2020,
                raw_r2_key=_raw_key(_W2, 2020),
            ),
        ]
    )
    assert dict(corrections) == {
        _TIPS_2023: SourceVintageCorrection(
            source_record_id=_TIPS_2023,
            package_id=_W2,
            stamped_year=2023,
            data_year=2020,
            fact_key=f"ledger.aggregate_fact.v2:{_TIPS_2023}@2023",
        )
    }


@pytest.mark.parametrize(
    ("package_id", "artifact_year", "period", "reason"),
    [
        ("soi-unreviewed-2021", 2021, 2023, "no register entry"),
        (_W2, 2020, 2024, r"register reviews \[2023\]"),
        (_W2, 2019, 2023, "register data year 2020"),
    ],
)
def test_an_unreviewed_restamp_is_refused(
    package_id: str, artifact_year: int, period: int, reason: str
) -> None:
    with pytest.raises(UnreviewedRestampError, match=reason) as raised:
        source_vintage_corrections(
            [_fact("x", period=period, raw_r2_key=_raw_key(package_id, artifact_year))]
        )
    assert len(raised.value.problems) == 1


def test_a_registered_file_under_a_key_naming_another_year_is_refused() -> None:
    fact = _fact(_TIPS_2023, period=2023, raw_r2_key=_raw_key(_W2, 2021), sha=_W2_SHA)
    with pytest.raises(UnreviewedRestampError, match="register data year 2020"):
        source_vintage_corrections([fact])


def test_one_record_with_two_different_corrections_is_refused() -> None:
    register = {"a": _entry("a", 2020, 2023, 2024)}
    with pytest.raises(ValueError, match="two corrections"):
        source_vintage_corrections(
            [
                _fact("x", period=2023, raw_r2_key=_raw_key("a", 2020)),
                _fact("x", period=2024, raw_r2_key=_raw_key("a", 2020)),
            ],
            register=register,
        )


def _chain_fact(year: int) -> dict[str, object]:
    return {
        "lineage": {
            "source_record_id": f"irs_soi.ty{year}.table_1_4.all.wages_salaries_amount"
        },
        "period": {"type": "tax_year", "value": year},
        "value": 1e13,
        "geography": {"level": "country"},
        "layout": {
            "record_set_id": f"irs_soi.ty{year}.table_1_4",
            "groupby_value_id": "all",
        },
        "observed_measure": {
            "source_name": "irs_soi",
            "source_measure_id": "wages_salaries_amount",
        },
    }


def test_a_restamped_fact_in_an_aging_chain_index_is_refused() -> None:
    # A pinned TY2022 Table 1.4 stamped 2023 would stand in for the TY2023
    # wages level in every chained factor that pivots on 2023.
    chain = [_chain_fact(2020), _chain_fact(2023)]
    restamped_id = "irs_soi.ty2023.table_1_4.all.wages_salaries_amount"
    correction = SourceVintageCorrection(restamped_id, "soi-t14-2022", 2023, 2022)
    with pytest.raises(RestampedAgingIndexError, match=restamped_id):
        check_restamps_stay_out_of_aging_indexes(chain, {restamped_id: correction})
    check_restamps_stay_out_of_aging_indexes(chain, {_TIPS_2023: _CORRECTION})
    check_restamps_stay_out_of_aging_indexes(chain, {})


def test_a_restamped_fact_in_the_cbo_projection_index_is_refused() -> None:
    projection_id = (
        "cbo.revenue_projection.ty2023.income_by_source.wages_and_salaries."
        "projected_amount"
    )
    projection = {
        "assertion": "source_projection",
        "lineage": {"source_record_id": projection_id},
        "period": {"type": "tax_year", "value": 2023},
        "value": 1e13,
        "layout": {
            "record_set_id": "cbo.revenue_projection.ty2023.income_by_source",
            "groupby_dimension": "cbo.income_source",
            "groupby_value_id": "wages_and_salaries",
        },
        "observed_measure": {
            "source_name": "cbo",
            "source_measure_id": "projected_amount",
        },
    }
    correction = SourceVintageCorrection(projection_id, "cbo-2026-02", 2023, 2022)
    with pytest.raises(RestampedAgingIndexError, match="cbo.revenue_projection"):
        check_restamps_stay_out_of_aging_indexes(
            [projection], {projection_id: correction}
        )


_CORRECTION = SourceVintageCorrection(
    _TIPS_2023, _W2, 2023, 2020, fact_key="ledger.aggregate_fact.v2:tips"
)


def test_apply_moves_only_the_backing_facts_source_period() -> None:
    registry = TargetRegistry(
        [
            _spec(
                "restamped",
                28.0,
                ledger_source_record_id=_TIPS_2023,
                ledger_fact_period="2023",
                source_period="2023",
            ),
            _spec(
                "other",
                5.0,
                ledger_source_record_id="irs_soi.ty2023.table_1_4.all.x",
                ledger_fact_period="2023",
                source_period="2023",
            ),
        ],
        country="us",
    )
    corrected = {
        spec.name: spec
        for spec in apply_source_vintage_corrections(
            registry, {_TIPS_2023: _CORRECTION}
        ).specs
    }
    restamped = corrected["restamped"]
    assert restamped.value == 28.0
    assert restamped.metadata["source_period"] == "2020"
    assert restamped.metadata["ledger_fact_period"] == "2023"
    assert restamped.metadata["source_vintage_stamped_period"] == "2023"
    assert restamped.metadata["source_vintage_correction"] == (
        "soi-w2-statistics-2020: stamped 2023, data year 2020"
    )
    assert corrected["other"] == registry.specs[1]


def test_apply_refuses_a_backed_spec_compared_at_another_period() -> None:
    # A fiscal-year fact is compared at its coverage start year, which can
    # differ from the stamp the correction was made for; fail closed.
    spec = _spec(
        "fiscal",
        ledger_source_record_id=_TIPS_2023,
        ledger_fact_period="2022",
        source_period="2023",
    )
    with pytest.raises(ValueError, match="not its stamp"):
        apply_source_vintage_corrections(
            TargetRegistry([spec], country="us"), {_TIPS_2023: _CORRECTION}
        )


def test_apply_moves_a_rebased_specs_control_period_to_the_controls_data_year() -> None:
    control_id = (
        "irs_soi.ty2023.congressional_district_2022.all_returns.us."
        "net_capital_gains_returns"
    )
    control = SourceVintageCorrection(
        control_id, "soi-congressional-district-2022", 2023, 2022
    )
    spec = _spec(
        "irs_soi.ty2022.historic_table_2.state_broad.ca.all.net_capital_gains_returns",
        3_774_052.0,
        ledger_source_record_id=(
            "irs_soi.ty2022.historic_table_2.state_broad.ca.all."
            "net_capital_gains_returns"
        ),
        ledger_fact_period="2022",
        source_period="2022",
        uprating_factor="0.97964474977721",
        uprating_from_period="2022",
        uprating_to_period="2023",
        uprating_index_source_period="2023",
        uprating_index_source_record_id=control_id,
    )
    (corrected,) = apply_source_vintage_corrections(
        TargetRegistry([spec], country="us"), {control_id: control}
    ).specs
    assert corrected.value == spec.value
    assert corrected.metadata["uprating_factor"] == "0.97964474977721"
    assert corrected.metadata["uprating_to_period"] == "2022"
    assert corrected.metadata["uprating_index_source_period"] == "2022"
    assert corrected.metadata["uprating_index_source_vintage_stamped_period"] == "2023"
    assert corrected.metadata["source_period"] == "2022"


def test_apply_refuses_a_restamped_fact_that_was_itself_rebased() -> None:
    spec = _spec(
        "restamped",
        ledger_source_record_id=_TIPS_2023,
        ledger_fact_period="2023",
        source_period="2023",
        uprating_factor="1.1",
        uprating_to_period="2023",
    )
    with pytest.raises(ValueError, match="was rebased"):
        apply_source_vintage_corrections(
            TargetRegistry([spec], country="us"), {_TIPS_2023: _CORRECTION}
        )


def test_apply_refuses_a_pooled_uprating_index_with_a_restamped_member() -> None:
    spec = _spec(
        "pooled",
        ledger_source_record_id="irs_soi.ty2022.table_2_5.x",
        ledger_fact_period="2022",
        uprating_index_source_record_ids=f"irs_soi.ty2024.a,{_TIPS_2023}",
    )
    with pytest.raises(ValueError, match="multi-source index"):
        apply_source_vintage_corrections(
            TargetRegistry([spec], country="us"), {_TIPS_2023: _CORRECTION}
        )


def test_apply_refuses_a_multi_fact_spec_with_a_restamped_member() -> None:
    # The spec's source period is its representative's; a restamped member
    # that is not the representative would go uncorrected.
    spec = _spec(
        "summed",
        ledger_source_record_id="irs_soi.ty2023.representative",
        ledger_fact_period="2023",
        ledger_member_fact_keys=json.dumps(
            ["ledger.aggregate_fact.v2:representative", _CORRECTION.fact_key]
        ),
    )
    with pytest.raises(ValueError, match="aggregates several facts"):
        apply_source_vintage_corrections(
            TargetRegistry([spec], country="us"), {_TIPS_2023: _CORRECTION}
        )


def test_correction_invariants_hold_for_generated_registries() -> None:
    """For any registry and any corrections: values, names, periods and order
    never move; the correction is idempotent; a spec changes if and only if
    it is backed by, or rebased onto, a corrected fact at its stamped period;
    and every moved period lands on that fact's data year. Hypothesis is a
    workspace dependency; the wheels job installs no test extras, so it
    skips there."""
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    record_ids = [f"irs_soi.ty2023.r{i}" for i in range(5)]
    years = st.integers(2015, 2026)

    @st.composite
    def cases(draw):
        corrections = {}
        for record_id in draw(st.sets(st.sampled_from(record_ids))):
            stamped = draw(years)
            data = draw(st.integers(2010, stamped - 1))
            corrections[record_id] = SourceVintageCorrection(
                record_id, "pkg", stamped, data
            )
        specs = []
        for index in range(draw(st.integers(0, 8))):
            record_id = draw(st.sampled_from(record_ids))
            # A backed spec is always compared at its fact's stamp; the
            # mismatch case is refused (tested above).
            fact_period = (
                corrections[record_id].stamped_year
                if record_id in corrections
                else draw(years)
            )
            metadata = {
                "ledger_source_record_id": record_id,
                "ledger_fact_period": str(fact_period),
                "source_period": str(draw(years)),
            }
            if draw(st.booleans()):
                metadata["uprating_index_source_record_id"] = draw(
                    st.sampled_from(record_ids)
                )
                metadata["uprating_index_source_period"] = str(draw(years))
                metadata["uprating_to_period"] = metadata[
                    "uprating_index_source_period"
                ]
            specs.append(
                _spec(
                    f"spec{index}",
                    draw(st.floats(0, 1e12, allow_nan=False)),
                    **metadata,
                )
            )
        return specs, corrections

    def backed(spec, corrections):
        return spec.metadata["ledger_source_record_id"] in corrections

    def rebased(spec, corrections):
        correction = corrections.get(
            spec.metadata.get("uprating_index_source_record_id", "")
        )
        return correction is not None and str(correction.stamped_year) == (
            spec.metadata.get("uprating_index_source_period")
        )

    @settings(max_examples=300, deadline=None)
    @given(cases())
    def check(case):
        specs, corrections = case
        registry = TargetRegistry(specs, country="us")
        once = apply_source_vintage_corrections(registry, corrections)
        twice = apply_source_vintage_corrections(once, corrections)
        assert once.specs == twice.specs
        assert [spec.name for spec in once.specs] == [spec.name for spec in specs]
        for before, after in zip(specs, once.specs, strict=True):
            assert after.value == before.value
            assert after.period == before.period
            assert (
                after.metadata["ledger_fact_period"]
                == (before.metadata["ledger_fact_period"])
            )
            is_backed = backed(before, corrections)
            is_rebased = rebased(before, corrections)
            assert (after != before) == (is_backed or is_rebased)
            if is_backed:
                correction = corrections[before.metadata["ledger_source_record_id"]]
                assert after.metadata["source_period"] == str(correction.data_year)
            else:
                assert (
                    after.metadata["source_period"]
                    == (before.metadata["source_period"])
                )
            if is_rebased:
                correction = corrections[
                    before.metadata["uprating_index_source_record_id"]
                ]
                assert after.metadata["uprating_to_period"] == str(correction.data_year)

    check()


def test_detection_matches_its_two_rules_for_generated_facts() -> None:
    """A fact is detected exactly when it is an observation (or carries no
    assertion) and either its digest is a registered file stamped after that
    file's data year, or its raw key parses to a year before its period."""
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @settings(max_examples=400, deadline=None)
    @given(
        artifact_year=st.integers(2000, 2030),
        period=st.one_of(
            st.integers(2000, 2030),
            st.integers(2000, 2030).map(lambda year: f"{year}-12"),
            st.integers(2000, 2030).map(lambda year: f"ty{year}"),
        ),
        assertion=st.sampled_from([None, "observation", "source_projection"]),
        keyed=st.booleans(),
        registered_file=st.booleans(),
    )
    def check(artifact_year, period, assertion, keyed, registered_file):
        fact = _fact(
            "x",
            period=period,
            raw_r2_key=_raw_key("some-package", artifact_year) if keyed else None,
            assertion=assertion,
            sha=_W2_SHA if registered_file else None,
        )
        period_year = int(str(period).removeprefix("ty")[:4])
        observation = assertion != "source_projection"
        by_content = registered_file and period_year > 2020
        by_key = keyed and artifact_year < period_year
        assert bool(detect_restamped_facts([fact])) == (
            observation and (by_content or by_key)
        )

    check()
