"""Restamped Chronicle facts are read at their data year (chronicle#117).

The compile-level cases (a restamped W-2 tips amount ageing from TY2020, the
refusal of an unreviewed restamp) and the pinned-feed register check live in
``test_us_fiscal_targets.py`` beside the other W-2 aging tests.
"""

from __future__ import annotations

import pytest

from microcosm.build.us_runtime.source_vintage import (
    US_RESTAMPED_SOURCE_PACKAGES,
    RestampedSourcePackage,
    SourceVintageCorrection,
    UnreviewedRestampError,
    apply_source_vintage_corrections,
    detect_restamped_facts,
    fact_artifact_year,
    source_vintage_corrections,
)
from microcosm.calibrate import TargetRegistry, TargetSpec

_SHA = "1178d77618cc1d2f873506909eeec660f36e3599854f31337f9dcaec6cfc442f"


def _raw_key(package_id: str, artifact_year: int | str, file: str = "f.xlsx") -> str:
    return f"raw/irs_soi/{package_id}/{artifact_year}/{_SHA}/{file}"


def _fact(
    source_record_id: str,
    *,
    period: int | str,
    raw_r2_key: str | None,
    assertion: str | None = "observation",
) -> dict[str, object]:
    fact: dict[str, object] = {
        "lineage": {"source_record_id": source_record_id},
        "period": {"type": "tax_year", "value": period},
        "source": {} if raw_r2_key is None else {"raw_r2_key": raw_r2_key},
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


_W2 = "soi-w2-statistics-2020"
_TIPS_2023 = (
    "irs_soi.ty2023.form_w2_social_security_tips.box_7_social_security_tips.amount"
)


def test_fact_artifact_year_reads_the_raw_key_package_and_year() -> None:
    assert fact_artifact_year(
        _fact("x", period=2023, raw_r2_key=_raw_key(_W2, 2020))
    ) == (_W2, 2020)
    for key in (
        None,
        "",
        "raw/irs_soi",
        f"raw/irs_soi/{_W2}/source_capture/{_SHA}/f.xlsx",
        f"raw/irs_soi/{_W2}/20/{_SHA}/f.xlsx",
        f"staged/irs_soi/{_W2}/2020/{_SHA}/f.xlsx",
        f"raw/irs_soi//2020/{_SHA}/f.xlsx",
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


def test_register_entries_describe_one_earlier_year_each() -> None:
    assert set(US_RESTAMPED_SOURCE_PACKAGES) == {
        "soi-w2-statistics-2020",
        "soi-congressional-district-2022",
        "soi-state-2022",
        "soi-ira-roth-contributions-2022",
        "soi-ira-traditional-contributions-2022",
    }
    for package_id, entry in US_RESTAMPED_SOURCE_PACKAGES.items():
        assert entry.package_id == package_id
        assert entry.stamped_periods
        assert all(stamp > entry.data_year for stamp in entry.stamped_periods)
        # IRS SOI file names lead with the two-digit tax year (20in04w2all).
        assert int(entry.source_file[:2]) == entry.data_year % 100
        assert package_id.endswith(str(entry.data_year))
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


def test_one_record_with_two_different_corrections_is_refused() -> None:
    register = {
        "a": RestampedSourcePackage("a", 2020, frozenset({2023, 2024}), "20a", "r"),
    }
    with pytest.raises(ValueError, match="two corrections"):
        source_vintage_corrections(
            [
                _fact("x", period=2023, raw_r2_key=_raw_key("a", 2020)),
                _fact("x", period=2024, raw_r2_key=_raw_key("a", 2020)),
            ],
            register=register,
        )


_CORRECTION = SourceVintageCorrection(_TIPS_2023, _W2, 2023, 2020)


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
            # Same id at another stamp is not the corrected fact.
            _spec(
                "other_stamp",
                7.0,
                ledger_source_record_id=_TIPS_2023,
                ledger_fact_period="2024",
                source_period="2024",
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
    for name in ("other", "other_stamp"):
        assert corrected[name] == {spec.name: spec for spec in registry.specs}[name]


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
            metadata = {
                "ledger_source_record_id": draw(st.sampled_from(record_ids)),
                "ledger_fact_period": str(draw(years)),
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
        correction = corrections.get(spec.metadata["ledger_source_record_id"])
        return (
            correction is not None
            and str(correction.stamped_year) == (spec.metadata["ledger_fact_period"])
        )

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
        # A backed spec that is also rebased is refused by design; the
        # generator does not set uprating_factor, so none is refused here.
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


def test_detection_matches_the_artifact_year_rule_for_generated_facts() -> None:
    """A fact is detected exactly when it is an observation (or carries no
    assertion), its raw key parses, and its artifact year precedes its
    period year."""
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @settings(max_examples=300, deadline=None)
    @given(
        artifact_year=st.integers(2000, 2030),
        period=st.one_of(
            st.integers(2000, 2030),
            st.integers(2000, 2030).map(lambda year: f"{year}-12"),
            st.integers(2000, 2030).map(lambda year: f"ty{year}"),
        ),
        assertion=st.sampled_from([None, "observation", "source_projection"]),
        keyed=st.booleans(),
    )
    def check(artifact_year, period, assertion, keyed):
        fact = _fact(
            "x",
            period=period,
            raw_r2_key=_raw_key(_W2, artifact_year) if keyed else None,
            assertion=assertion,
        )
        period_year = int(str(period).removeprefix("ty")[:4])
        expected = (
            keyed and assertion != "source_projection" and artifact_year < period_year
        )
        assert bool(detect_restamped_facts([fact])) == expected

    check()
