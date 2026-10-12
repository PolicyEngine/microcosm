"""Restamped Chronicle facts are read at their data year (chronicle#117).

The compile-level cases (a restamped W-2 tips amount ageing from TY2020, the
refusals of an unreviewed or shadowing restamp) and the pinned-feed register
check live in ``test_us_fiscal_targets.py`` beside the other W-2 aging tests.
"""

from __future__ import annotations

import dataclasses
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
    # The pinned feed's case while the capital-gains control is chosen by
    # period: Historic Table 2 counts rebased onto the congressional-district
    # US row. microcosm#1036 chooses the control by concept, which never picks
    # that row; the branch still covers any restamped control.
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


_CD = "soi-congressional-district-2022"
_CD_RECORD_SET = "irs_soi.ty2023.congressional_district_2022.all_returns"
_CD_NJ_AMOUNT = f"{_CD_RECORD_SET}.nj_total.net_capital_gains_amount"
_CD_US_AMOUNT = f"{_CD_RECORD_SET}.us.net_capital_gains_amount"
_T14_AMOUNT = "irs_soi.ty2023.table_1_4.all.net_capital_gains_amount"
_CD_CORRECTIONS = {
    record_id: SourceVintageCorrection(record_id, _CD, 2023, 2022)
    for record_id in (_CD_NJ_AMOUNT, _CD_US_AMOUNT)
}


def _cd_share_spec(**overrides: str) -> TargetSpec:
    """A congressional-district row rebased as a share of the same package's
    US row times the Table 1.4 control, as microcosm#1040 compiles it. The
    rebase runs before the correction, so it reads the stamp, 2023."""
    metadata = {
        "ledger_source_record_id": _CD_NJ_AMOUNT,
        "ledger_fact_period": "2023",
        "source_period": "2023",
        "uprating_factor": "0.834893818418",
        "uprating_from_period": "2023",
        "uprating_to_period": "2023",
        "uprating_index_source_period": "2023",
        "uprating_index_source_record_id": _T14_AMOUNT,
        "soi_share_total_source_record_id": _CD_US_AMOUNT,
    }
    metadata.update(overrides)
    return _spec(_CD_NJ_AMOUNT, 38_463_637_000.0, **metadata)


def test_apply_admits_a_share_of_a_restamped_total_of_the_same_package() -> None:
    """Row and total come from one restamped file, so the stamp cancels in the
    share; the level is the truthful Table 1.4 control's. Only the share's own
    period moves to the data year. ``uprating_to_period``, where aging starts,
    stays at the control's 2023: moved to 2022 it would age the control's
    TY2022-to-TY2023 change a second time (x0.761 for net capital gains)."""
    spec = _cd_share_spec()
    (corrected,) = apply_source_vintage_corrections(
        TargetRegistry([spec], country="us"), _CD_CORRECTIONS
    ).specs
    assert corrected.value == spec.value
    assert corrected.metadata["uprating_factor"] == "0.834893818418"
    assert corrected.metadata["source_period"] == "2022"
    assert corrected.metadata["source_vintage_stamped_period"] == "2023"
    assert corrected.metadata["uprating_from_period"] == "2022"
    assert corrected.metadata["uprating_from_period_vintage_stamped_period"] == "2023"
    assert corrected.metadata["uprating_to_period"] == "2023"
    assert corrected.metadata["uprating_index_source_period"] == "2023"
    assert "uprating_index_source_vintage_correction" not in corrected.metadata
    assert corrected.metadata["ledger_fact_period"] == "2023"
    again = apply_source_vintage_corrections(
        TargetRegistry([corrected], country="us"), _CD_CORRECTIONS
    )
    assert again.specs == (corrected,)


def test_an_admitted_share_ages_exactly_as_its_truthfully_stamped_twin() -> None:
    """Differential: one congressional-district share compiled from the
    package stamped ty2023 (then corrected) and from the package stamped
    truthfully at 2022 (nothing to correct) ages to the same value, from the
    control's 2023 period on the direct CBO 2024/2023 ratio, and passes the
    period contract. Starting it at the data year instead chains the Table 1.4
    TY2022-to-TY2023 change onto a level that already carries it."""
    from microcosm.build.us_runtime.target_aging import (
        age_us_dollar_targets,
        find_period_contract_violations,
    )
    from test_support.microcosm_build.us_fiscal_targets import (
        _cbo_income_source_projection_fact,
        _dynamic_ledger_fact,
    )

    aging_facts = (
        _cbo_income_source_projection_fact(2023, "net_capital_gain", value=981.4e9),
        _cbo_income_source_projection_fact(2024, "net_capital_gain", value=1290.9e9),
        *(
            _dynamic_ledger_fact(
                source_record_id=f"irs_soi.ty{year}.table_1_4.all.net_capital_gains_amount",
                source_name="irs_soi",
                measure_id="net_capital_gains_amount",
                value=value,
                period_value=year,
                layout_record_set_id=f"irs_soi.ty{year}.table_1_4",
            )
            for year, value in ((2022, 1_400.0e9), (2023, 1_065.2e9))
        ),
    )
    dollars = {"measure_mode": "sum", "source_measure_id": "net_capital_gains_amount"}
    stamped = _cd_share_spec(**dollars)
    truthful_set = "irs_soi.ty2022.congressional_district_2022.all_returns"
    truthful = _cd_share_spec(
        **dollars,
        ledger_source_record_id=f"{truthful_set}.nj_total.net_capital_gains_amount",
        ledger_fact_period="2022",
        source_period="2022",
        uprating_from_period="2022",
        soi_share_total_source_record_id=f"{truthful_set}.us.net_capital_gains_amount",
    )

    def aged(spec, corrections):
        registry = apply_source_vintage_corrections(
            TargetRegistry([spec], country="us"), corrections
        )
        registry = age_us_dollar_targets(registry, aging_facts, target_period=2024)
        assert not find_period_contract_violations(registry, target_period=2024)
        (result,) = registry.specs
        return result

    from_stamp = aged(stamped, _CD_CORRECTIONS)
    from_truth = aged(truthful, {})
    ratio = 1290.9e9 / 981.4e9
    for spec in (from_stamp, from_truth):
        assert spec.metadata["source_period"] == "2023"
        assert spec.metadata["aging_factor_source"] == (
            "cbo.revenue_projection.ty2024.income_by_source.net_capital_gain."
            "projected_amount"
        )
        assert float(spec.metadata["aging_factor"]) == pytest.approx(ratio, rel=1e-12)
    assert (
        from_stamp.value
        == from_truth.value
        == pytest.approx(38_463_637_000.0 * ratio, rel=1e-12)
    )

    wrong = aged(
        dataclasses.replace(
            truthful,
            metadata={**truthful.metadata, "uprating_to_period": "2022"},
        ),
        {},
    )
    assert wrong.value / from_truth.value == pytest.approx(1_065.2 / 1_400.0, rel=1e-12)


@pytest.mark.parametrize(
    ("overrides", "corrections"),
    [
        pytest.param(
            {"soi_share_total_source_record_id": None},
            _CD_CORRECTIONS,
            id="no-share-total",
        ),
        pytest.param(
            {
                "soi_share_total_source_record_id": (
                    "irs_soi.ty2022.historic_table_2.us.all.net_capital_gains_amount"
                )
            },
            _CD_CORRECTIONS,
            id="truthful-share-total",
        ),
        pytest.param(
            {},
            {
                **_CD_CORRECTIONS,
                _CD_US_AMOUNT: SourceVintageCorrection(
                    _CD_US_AMOUNT, "soi-state-2022", 2023, 2022
                ),
            },
            id="share-total-from-another-package",
        ),
        pytest.param(
            {},
            {
                **_CD_CORRECTIONS,
                _CD_US_AMOUNT: SourceVintageCorrection(_CD_US_AMOUNT, _CD, 2024, 2022),
            },
            id="share-total-at-another-stamp",
        ),
        pytest.param(
            {},
            {
                **_CD_CORRECTIONS,
                _CD_US_AMOUNT: SourceVintageCorrection(_CD_US_AMOUNT, _CD, 2023, 2021),
            },
            id="share-total-of-another-data-year",
        ),
        pytest.param(
            {"uprating_from_period": "2021"},
            _CD_CORRECTIONS,
            id="share-read-at-another-period",
        ),
        pytest.param(
            {"uprating_from_period": None},
            _CD_CORRECTIONS,
            id="no-uprating-from-period",
        ),
    ],
)
def test_apply_refuses_a_rebased_restamp_that_is_not_a_same_package_share(
    overrides: dict[str, str | None],
    corrections: dict[str, SourceVintageCorrection],
) -> None:
    """A rebased restamp is admitted only as a share of a restamped total of
    its own package, stamp and data year, read at the stamp or the data year.
    Any other rebase (a missing or truthful denominator, a mixed-vintage share,
    another reading period) mixes vintages the correction cannot vouch for."""
    spec = _cd_share_spec()
    metadata = dict(spec.metadata)
    for key, value in overrides.items():
        if value is None:
            metadata.pop(key)
        else:
            metadata[key] = value
    spec = dataclasses.replace(spec, metadata=metadata)
    with pytest.raises(ValueError, match="was rebased, but not as a share"):
        apply_source_vintage_corrections(
            TargetRegistry([spec], country="us"), corrections
        )


def test_apply_refuses_a_restamped_fact_that_was_rebased_without_a_share() -> None:
    spec = _spec(
        "restamped",
        ledger_source_record_id=_TIPS_2023,
        ledger_fact_period="2023",
        source_period="2023",
        uprating_factor="1.1",
        uprating_from_period="2023",
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


def _vintage(correction: SourceVintageCorrection) -> tuple[str, int, int]:
    return correction.package_id, correction.stamped_year, correction.data_year


def test_correction_invariants_hold_for_generated_registries() -> None:
    """For any registry and any corrections, the correction refuses exactly
    when a restamped spec was rebased other than as a share of a restamped
    total of its own package, stamp and data year, read at the stamp or the
    data year. Otherwise: values, names, periods and order never move; the
    correction is idempotent; a spec changes if and only if it is backed by,
    or rebased onto, a corrected fact at its stamped period; every moved
    period lands on that fact's data year; ``uprating_from_period`` moves only
    for an admitted share and ``uprating_to_period`` only for a restamped
    control. Hypothesis is a workspace dependency; the wheels job installs no
    test extras, so it skips there."""
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    record_ids = [f"irs_soi.ty2023.r{i}" for i in range(5)]
    years = st.integers(2015, 2026)
    # A few fixed vintages make same-package shares and near misses common;
    # the free draw keeps any package, stamp and data year in play.
    vintages = st.one_of(
        # The last three each differ from the first in exactly one of
        # package, stamp and data year.
        st.sampled_from(
            [
                ("pkg_a", 2023, 2022),
                ("pkg_a", 2024, 2022),
                ("pkg_b", 2023, 2022),
                ("pkg_a", 2023, 2020),
            ]
        ),
        st.tuples(st.sampled_from(["pkg_a", "pkg_b"]), years).flatmap(
            lambda pair: st.tuples(
                st.just(pair[0]), st.just(pair[1]), st.integers(2010, pair[1] - 1)
            )
        ),
    )

    @st.composite
    def cases(draw, mode):
        # mode "free" draws shares at random, so most calls with a rebased
        # restamp refuse. "well_formed" gives each rebased restamp a total of
        # its own vintage, so the call is admitted and the admitted-share
        # invariants get exercised. "near_miss" gives it exactly one defect, so
        # each comparison the admission makes is tested alone.
        corrections = {}
        # A small pool per case makes the free mode's totals often match the
        # row's vintage or miss it narrowly, and keeps mixed packages, stamps
        # and data years among the corrections.
        pool = draw(st.lists(vintages, min_size=1, max_size=3))
        for record_id in draw(st.sets(st.sampled_from(record_ids))):
            package_id, stamped, data = draw(st.sampled_from(pool))
            corrections[record_id] = SourceVintageCorrection(
                record_id, package_id, stamped, data
            )

        def period_near(record_id):
            # Bias toward the correction's own stamp and data year, which the
            # branches under test compare against.
            correction = corrections.get(record_id)
            if correction is None:
                return draw(years)
            return draw(
                st.one_of(
                    years,
                    st.sampled_from([correction.stamped_year, correction.data_year]),
                )
            )

        def share(record_id):
            # The share keys of a rebase.
            correction = corrections.get(record_id)
            if mode == "free" or correction is None:
                keys = {}
                if draw(st.integers(0, 4)):
                    keys["uprating_from_period"] = str(period_near(record_id))
                if draw(st.integers(0, 4)):
                    keys["soi_share_total_source_record_id"] = draw(
                        st.sampled_from(
                            sorted({*record_ids, *corrections})
                            + ["irs_soi.ty2023.truthful"]
                        )
                    )
                return keys
            # A total of its own, restamped at the row's vintage, read at the
            # stamp or the data year (well-formed). A near miss changes exactly
            # one of these: the total's package, stamp or data year, or the
            # reading period (another year, or none).
            package_id, stamped, data = _vintage(correction)
            reading = draw(
                st.sampled_from([correction.stamped_year, correction.data_year])
            )
            if mode == "near_miss":
                component = draw(
                    st.sampled_from(["package", "stamp", "data", "reading"])
                )
                if component == "package":
                    package_id = "pkg_b" if package_id == "pkg_a" else "pkg_a"
                elif component == "stamp":
                    stamped += 1
                elif component == "data":
                    data -= 1
                else:
                    reading = draw(
                        st.sampled_from(
                            [
                                None,
                                correction.data_year - 1,
                                correction.stamped_year + 1,
                            ]
                        )
                    )
            total_id = f"irs_soi.ty2023.total{len(corrections)}"
            corrections[total_id] = SourceVintageCorrection(
                total_id, package_id, stamped, data
            )
            keys = {"soi_share_total_source_record_id": total_id}
            if reading is not None:
                keys["uprating_from_period"] = str(reading)
            return keys

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
                control_id = draw(st.sampled_from(record_ids))
                metadata["uprating_index_source_record_id"] = control_id
                metadata["uprating_index_source_period"] = str(period_near(control_id))
                metadata["uprating_to_period"] = metadata[
                    "uprating_index_source_period"
                ]
                if draw(st.booleans()):
                    metadata["uprating_factor"] = "0.5"
                    metadata.update(share(record_id))
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

    def admitted_share(spec, corrections):
        backing = corrections[spec.metadata["ledger_source_record_id"]]
        total = corrections.get(
            spec.metadata.get("soi_share_total_source_record_id", "")
        )
        return (
            total is not None
            and _vintage(total) == _vintage(backing)
            and spec.metadata.get("uprating_from_period")
            in (str(backing.stamped_year), str(backing.data_year))
        )

    def refused(spec, corrections):
        return (
            backed(spec, corrections)
            and "uprating_factor" in spec.metadata
            and not admitted_share(spec, corrections)
        )

    @settings(max_examples=500, deadline=None)
    @given(st.sampled_from(["free", "well_formed", "near_miss"]).flatmap(cases))
    def check(case):
        specs, corrections = case
        registry = TargetRegistry(specs, country="us")
        if any(refused(spec, corrections) for spec in specs):
            with pytest.raises(ValueError, match="was rebased, but not as a share"):
                apply_source_vintage_corrections(registry, corrections)
            return
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
            if is_backed and "uprating_factor" in before.metadata:
                assert after.metadata["uprating_from_period"] == str(
                    correction.data_year
                )
            else:
                assert after.metadata.get("uprating_from_period") == (
                    before.metadata.get("uprating_from_period")
                )
            if is_rebased:
                correction = corrections[
                    before.metadata["uprating_index_source_record_id"]
                ]
                assert after.metadata["uprating_to_period"] == str(correction.data_year)
            else:
                assert after.metadata.get("uprating_to_period") == (
                    before.metadata.get("uprating_to_period")
                )

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
