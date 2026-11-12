"""ACS household types and SOI filing statuses bind on the release surface."""

from copy import deepcopy

import pytest

from microcosm.build.us_runtime.fiscal_targets import (
    compile_us_fiscal_target_registry,
    household_type_fact_exclusion_reason,
    soi_filing_status_fact_exclusion_reason,
)
from test_support.microcosm_build.us_batched_target_materialization import (
    _load_builder_module,
)
from test_support.microcosm_build.us_fiscal_targets import (
    _dynamic_ledger_fact,
    packaged_reference_facts,
)


def household_fact(
    measure: str,
    household_type: str,
    *,
    own_children: str = "all",
    statistic: str = "estimate",
    state: str | None = None,
    record_set: str | None = None,
) -> dict[str, object]:
    table = measure.split("_")[0].lower()
    geography = "state" if state else "national"
    record_set = record_set or f"census_acs.acs1_2024.{table}.{geography}"
    fact = _dynamic_ledger_fact(
        source_record_id=f"{record_set}.{state or 'us'}.{measure}",
        source_name="census_acs",
        measure_id=measure,
        value=100,
        layout_record_set_id=record_set,
        geography_level="state" if state else "country",
        geography_id=f"0400000US{state}" if state else "0100000US",
        groupby_dimension="census_acs.state" if state else "census_acs.nation",
        groupby_value_id=state or "us",
        dimensions={
            "statistic": statistic,
            "household_type": household_type,
            "own_children": own_children,
        },
    )
    fact["entity"] = {"name": "household"}
    fact["observed_measure"]["unit"] = "count"
    fact["universe_constraints"] = {
        "constraints": [
            {"variable": key, "operator": "==", "value": value}
            for key, value in fact["dimensions"].items()
            if value != "all"
        ]
    }
    return fact


_BOUND_ACS = (
    ("B11001_001E", "all", "all"),
    ("B11001_003E", "married_couple_family", "all"),
    ("B11001_005E", "male_householder_no_spouse", "all"),
    ("B11001_006E", "female_householder_no_spouse", "all"),
    ("B11001_008E", "householder_living_alone", "all"),
    ("B11003_003E", "married_couple_family", "with_own_children_under_18"),
    ("B11003_010E", "male_householder_no_spouse", "with_own_children_under_18"),
    ("B11003_016E", "female_householder_no_spouse", "with_own_children_under_18"),
)


def test_acs_eight_national_cells_bind_by_default() -> None:
    facts = [
        household_fact(measure, kind, own_children=children)
        for measure, kind, children in _BOUND_ACS
    ]
    registry = compile_us_fiscal_target_registry(
        [*packaged_reference_facts(), *facts], allow_unaged_dollar_targets=True
    )
    specs = [spec for spec in registry.specs if spec.family == "census_households"]

    assert len(specs) == 8
    _load_builder_module()._assert_supported_ledger_filter_metadata(specs)
    by_measure = {spec.metadata["source_measure_id"]: spec for spec in specs}
    for fact, (measure, kind, children) in zip(facts, _BOUND_ACS, strict=True):
        assert household_type_fact_exclusion_reason(fact) is None
        spec = by_measure[measure]
        assert spec.entity == "household"
        assert spec.value == 100
        assert spec.metadata["materializer"] == "household_type"
        assert spec.metadata["measure_mode"] == "indicator_sum"
        assert spec.metadata["target_role"] == "household_type"
        assert spec.metadata["household_type"] == kind
        assert spec.metadata["own_children"] == children
        assert spec.metadata["source_period"] == "2024"
        assert spec.metadata["target_period"] == "2024"
        assert "state_fips" not in spec.metadata


@pytest.mark.parametrize(
    ("fact", "reason"),
    [
        (
            household_fact(
                "B11001_003M", "married_couple_family", statistic="margin_of_error"
            ),
            "uncertainty",
        ),
        (
            household_fact("B11001_003E", "married_couple_family", state="06"),
            "State household-type",
        ),
        (
            household_fact("B11001_003E", "married_couple_family", state="72"),
            "Puerto Rico",
        ),
        (household_fact("B11001_002E", "family"), "implied"),
        (household_fact("B11001_004E", "other_family"), "implied"),
        (household_fact("B11001_007E", "nonfamily"), "implied"),
        (household_fact("B11001_009E", "householder_not_living_alone"), "implied"),
        (household_fact("B11003_002E", "married_couple_family"), "implied"),
        (
            household_fact(
                "B11003_004E",
                "married_couple_family",
                own_children="own_children_under_6_only",
            ),
            "age splits",
        ),
        (
            household_fact(
                "B11003_007E",
                "married_couple_family",
                own_children="no_own_children_under_18",
            ),
            "implied",
        ),
    ],
)
def test_acs_reviewed_exclusions(fact: dict[str, object], reason: str) -> None:
    assert reason in household_type_fact_exclusion_reason(fact)
    registry = compile_us_fiscal_target_registry(
        [*packaged_reference_facts(), fact], allow_unaged_dollar_targets=True
    )
    assert not any(spec.family == "census_households" for spec in registry.specs)


def test_household_dispatch_does_not_admit_other_acs_tables_or_periods() -> None:
    wrong_table = household_fact(
        "B11001_003E",
        "married_couple_family",
        record_set="census_acs.acs1_2024.b99999.national",
    )
    wrong_period = household_fact("B11001_003E", "married_couple_family")
    wrong_period["period"]["value"] = 2023
    wrong_period_type = deepcopy(wrong_period)
    wrong_period_type["period"] = {"type": "tax_year", "value": 2024}
    registry = compile_us_fiscal_target_registry(
        [*packaged_reference_facts(), wrong_table, wrong_period, wrong_period_type],
        allow_unaged_dollar_targets=True,
    )
    assert not any(spec.family == "census_households" for spec in registry.specs)
    assert "2024 calendar year" in household_type_fact_exclusion_reason(wrong_period)
    assert "2024 calendar year" in household_type_fact_exclusion_reason(
        wrong_period_type
    )


def filing_status_fact(status: str, measure: str) -> dict[str, object]:
    group = {
        "return_count": "returns",
        "adjusted_gross_income": "agi",
        "ctc_claims": "ctc_returns",
        "ctc_amount": "ctc_amount",
    }[measure]
    record_set = f"irs_soi.ty2023.table_1_3.filing_status_{group}"
    fact = _dynamic_ledger_fact(
        source_record_id=f"{record_set}.{status}.{measure}",
        source_name="irs_soi",
        measure_id=measure,
        value=100,
        period_value=2023,
        layout_record_set_id=record_set,
        groupby_dimension="filing_status",
        groupby_value_id=status,
        dimensions={"filing_status": status, "income_range": "all"},
        universe_constraints=(
            []
            if status == "all"
            else [{"variable": "filing_status", "operator": "==", "value": status}]
        ),
    )
    fact["period"]["type"] = "tax_year"
    fact["entity"] = {"name": "tax_unit", "role": "filing_unit"}
    fact["observed_measure"]["unit"] = (
        "count" if measure in {"return_count", "ctc_claims"} else "usd"
    )
    return fact


def test_soi_sixteen_status_cells_bind_and_all_four_totals_are_excluded() -> None:
    statuses = {
        "married_filing_jointly_surviving_spouse": "Married Filing Jointly/Surviving Spouse",
        "married_filing_separately": "Married Filing Separately",
        "head_of_household": "Head of Household",
        "single": "Single",
    }
    measures = ("return_count", "adjusted_gross_income", "ctc_claims", "ctc_amount")
    facts = [
        filing_status_fact(status, measure)
        for status in (*statuses, "all")
        for measure in measures
    ]
    registry = compile_us_fiscal_target_registry(
        [*packaged_reference_facts(), *facts], allow_unaged_dollar_targets=True
    )
    specs = [spec for spec in registry.specs if ".table_1_3." in spec.name]

    assert len(specs) == 16
    for spec in specs:
        assert spec.metadata["materializer"] == "irs_soi_slice"
        assert spec.metadata["filing_status"] in statuses.values()
        assert spec.metadata["soi_return_universe"] == "all_returns"
        assert spec.metadata["source_period"] == "2023"
        assert spec.metadata["target_period"] == "2024"
        assert spec.metadata["agi_lower_bound"] == "-inf"
        assert spec.metadata["agi_upper_bound"] == "inf"
        assert "filer_only" not in spec.metadata
        assert spec.metadata["measure_mode"] == (
            "indicator_sum"
            if spec.metadata["source_measure_id"] in {"return_count", "ctc_claims"}
            else "sum"
        )
    for fact in facts:
        reason = soi_filing_status_fact_exclusion_reason(fact)
        if fact["dimensions"]["filing_status"] == "all":
            assert (
                "Table 1.2"
                if fact["layout"]["measure_id"]
                in {"return_count", "adjusted_gross_income"}
                else "implied"
            ) in reason
        else:
            assert reason is None
