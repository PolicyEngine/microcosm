"""State x AGI-band SOI targets from Historic Table 2 (microcosm#940).

A state's Historic Table 2 AGI band binds, from the $100k floor up, as a share
of the same state's published partition, rebased onto the state all-returns
total that binds (``_rebase_soi_state_agi_bands``). The synthetic tests below
pin the rule and its invariants; the feed-gated tests check it on the pinned
Chronicle feed.
"""

from __future__ import annotations

import importlib.util
import math
import random
from hashlib import sha256
from pathlib import Path

import pytest

from microcosm.build import target_profile_coverage_gate
from microcosm.build.us_runtime import (
    US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    US_FISCAL_TARGET_REFERENCES,
    compile_us_fiscal_target_registry,
)
from microcosm.build.us_runtime.fiscal_targets import (
    US_SOI_STATE_AGI_BAND_MINIMUM_LOWER_BOUND,
)
from microcosm.build.us_runtime.target_aging import age_us_dollar_targets
from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_POSTAL

AGI = "us:statutes/26/62#adjusted_gross_income"
POSTAL_TO_FIPS = {postal: fips for fips, postal in US_STATE_FIPS_TO_POSTAL.items()}
FLAG = "requires_state_agi_band_rebase"
MEASURES = ("return_count", "adjusted_gross_income")

#: The ten Historic Table 2 AGI stubs, as Chronicle labels them.
BANDS: tuple[tuple[str, float | None, float | None], ...] = (
    ("under_1", None, 1),
    ("1_to_10k", 1, 10_000),
    ("10k_to_25k", 10_000, 25_000),
    ("25k_to_50k", 25_000, 50_000),
    ("50k_to_75k", 50_000, 75_000),
    ("75k_to_100k", 75_000, 100_000),
    ("100k_to_200k", 100_000, 200_000),
    ("200k_to_500k", 200_000, 500_000),
    ("500k_to_1m", 500_000, 1_000_000),
    ("1m_plus", 1_000_000, None),
)
#: The TY2022 Chronicle package publishes stubs 9 and 10 summed.
COLLAPSED_BANDS = (*BANDS[:8], ("500k_plus", 500_000, None))
BINDING_BANDS = tuple(
    band
    for band, lower, _ in BANDS
    if lower is not None and lower >= US_SOI_STATE_AGI_BAND_MINIMUM_LOWER_BOUND
)


# --------------------------------------------------------------------------
# Synthetic Chronicle facts


def _fact_id(name: str, period: object) -> str:
    digest = sha256(f"{name}@{period}".encode()).hexdigest()[:12]
    return f"{name.replace('.', '_')[:72]}_{digest}_{period}"


def _fact(
    *,
    source_record_id: str,
    record_set_id: str,
    measure_id: str,
    value: float,
    period: int,
    geography_level: str,
    geography_id: str,
    dimensions: dict[str, str],
    groupby_dimension: str,
    groupby_value_id: str,
    constraints: list[dict[str, object]] | None = None,
    source_name: str = "irs_soi",
) -> dict[str, object]:
    fact_id = _fact_id(source_record_id, period)
    labels = {**dimensions, groupby_dimension: groupby_value_id}
    return {
        "label": f"Test label for {source_record_id}",
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{fact_id}",
        "semantic_fact_key": f"ledger.semantic_fact.v2:{fact_id}",
        "legacy_fact_key": f"ledger.fact.v1:{fact_id}",
        "lineage": {"source_record_id": source_record_id},
        "value": value,
        "period": {"type": "tax_year", "value": period},
        "entity": {"name": "tax_unit"},
        "aggregation": {"method": "sum"},
        "geography": {
            "level": geography_level,
            "id": geography_id,
            "name": f"Test geography {geography_id}",
        },
        "dimensions": dict(dimensions),
        "dimension_labels": {key: f"Test dimension {key}" for key in labels},
        "dimension_value_labels": {
            key: {str(value): f"Test value {key}={value}"}
            for key, value in labels.items()
        },
        "universe_constraints": {"constraints": list(constraints or [])},
        "layout": {
            "record_set_id": record_set_id,
            "groupby_dimension": groupby_dimension,
            "groupby_dimension_label": f"Test dimension {groupby_dimension}",
            "groupby_value_id": groupby_value_id,
            "groupby_value_label": (
                f"Test value {groupby_dimension}={groupby_value_id}"
            ),
            "measure_id": measure_id,
        },
        "observed_measure": {
            "source_name": source_name,
            "source_table": f"{source_name} table",
            "source_measure_id": measure_id,
            "source_concept": measure_id,
            "unit": "count" if measure_id == "return_count" else "usd",
        },
        "source": {
            "source_name": source_name,
            "source_table": f"{source_name} table",
            "vintage": f"tax_year_{period}",
            "url": f"https://example.org/{fact_id}",
        },
    }


def _band_fact(
    postal: str,
    period: int,
    band: str,
    lower: float | None,
    upper: float | None,
    measure_id: str,
    value: float,
    *,
    filing_status: str = "all",
    record_set_kind: str = "state_agi",
) -> dict[str, object]:
    record_set_id = (
        f"irs_soi.ty{period}.historic_table_2.{record_set_kind}.{postal.lower()}"
    )
    constraints = []
    if upper is not None:
        constraints.append({"variable": AGI, "operator": "<", "value": upper})
    if lower is not None:
        constraints.append({"variable": AGI, "operator": ">=", "value": lower})
    geography_level, geography_id = (
        ("country", "0100000US")
        if postal == "US"
        else ("state", f"0400000US{POSTAL_TO_FIPS[postal]}")
    )
    value_id = band if filing_status == "all" else f"{filing_status}_{band}"
    return _fact(
        source_record_id=f"{record_set_id}.{value_id}.{measure_id}",
        record_set_id=record_set_id,
        measure_id=measure_id,
        value=value,
        period=period,
        geography_level=geography_level,
        geography_id=geography_id,
        dimensions={"filing_status": filing_status, "income_range": band},
        groupby_dimension=AGI,
        groupby_value_id=band,
        constraints=constraints,
    )


def _state_total_fact(
    postal: str,
    period: int,
    measure_id: str,
    value: float,
    *,
    record_set_id: str | None = None,
) -> dict[str, object]:
    record_set_id = record_set_id or (
        f"irs_soi.ty{period}.historic_table_2.state_broad.{postal.lower()}"
    )
    return _fact(
        source_record_id=f"{record_set_id}.all.{measure_id}",
        record_set_id=record_set_id,
        measure_id=measure_id,
        value=value,
        period=period,
        geography_level="state",
        geography_id=f"0400000US{POSTAL_TO_FIPS[postal]}",
        dimensions={"filing_status": "all", "income_range": "all"},
        groupby_dimension="state",
        groupby_value_id="all",
    )


def _partition(
    postal: str,
    period: int,
    values: dict[str, tuple[float, float]],
    *,
    bands=BANDS,
) -> list[dict[str, object]]:
    """One state's band facts: ``values[band] = (returns, agi)``."""
    facts = []
    for band, lower, upper in bands:
        returns, agi = values[band]
        facts.append(
            _band_fact(postal, period, band, lower, upper, "return_count", returns)
        )
        facts.append(
            _band_fact(postal, period, band, lower, upper, "adjusted_gross_income", agi)
        )
    return facts


def _totals(postal: str, period: int, returns: float, agi: float) -> list[dict]:
    return [
        _state_total_fact(postal, period, "return_count", returns),
        _state_total_fact(postal, period, "adjusted_gross_income", agi),
    ]


#: Colorado's published TY2023 cells (23in55cmcsv.csv; AGI in dollars).
CO_2023 = {
    "under_1": (47_150, -2_559_604_000),
    "1_to_10k": (291_060, 1_416_216_000),
    "10k_to_25k": (389_460, 6_728_128_000),
    "25k_to_50k": (632_110, 23_615_781_000),
    "50k_to_75k": (477_700, 29_421_989_000),
    "75k_to_100k": (320_010, 27_725_880_000),
    "100k_to_200k": (581_560, 81_122_417_000),
    "200k_to_500k": (257_510, 74_198_170_000),
    "500k_to_1m": (38_030, 25_370_899_000),
    "1m_plus": (15_610, 47_266_883_000),
}
#: Colorado's TY2022 all-returns totals (22in55cmcsv.csv, AGI_STUB 0).
CO_2022_TOTAL_RETURNS = 2_972_380
CO_2022_TOTAL_AGI = 297_676_269_000


def _reference_fact(reference, value: float) -> dict[str, object]:
    selector = dict(reference.ledger_selector)
    source_name = str(selector.get("source_name") or reference.family)
    measure_id = str(
        selector.get("source_measure_id")
        or selector.get("layout_measure_id")
        or reference.measure
        or reference.name
    )
    period = selector.get("period_value") or reference.period
    fact_id = _fact_id(reference.name, period)
    return {
        "label": f"Test label for {reference.name}",
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{fact_id}",
        "semantic_fact_key": f"ledger.semantic_fact.v2:{fact_id}",
        "legacy_fact_key": f"ledger.fact.v1:{fact_id}",
        "lineage": {
            "source_record_id": reference.ledger_source_record_id
            or f"source.record:{fact_id}"
        },
        "value": value,
        "period": {
            "type": str(selector.get("period_type") or "tax_year"),
            "value": period,
        },
        "entity": {"name": str(selector.get("entity_name") or reference.entity)},
        "aggregation": {"method": "sum"},
        "geography": {
            "level": str(selector.get("geography_level") or "country"),
            "id": str(selector.get("geography_id") or "0100000US"),
            "name": "Test geography",
        },
        "dimensions": dict(selector.get("dimensions") or {}),
        "layout": {
            "record_set_id": str(
                selector.get("layout_record_set_id") or f"{source_name}.record_set"
            ),
            "groupby_dimension": str(selector.get("layout_groupby_dimension") or ""),
            "groupby_value_id": str(selector.get("layout_groupby_value_id") or "all"),
            "measure_id": measure_id,
        },
        "observed_measure": {
            "source_name": source_name,
            "source_table": str(selector.get("source_table") or f"{source_name} t"),
            "source_measure_id": measure_id,
            "source_concept": str(selector.get("source_concept") or measure_id),
            "unit": "usd",
        },
        "source": {
            "source_name": source_name,
            "source_table": str(selector.get("source_table") or f"{source_name} t"),
            "vintage": str(period),
            "url": f"https://example.org/{fact_id}",
        },
    }


def _packaged_reference_facts() -> list[dict[str, object]]:
    return [
        _reference_fact(reference, value=index + 1)
        for index, reference in enumerate(US_FISCAL_TARGET_REFERENCES)
    ]


def _compile(facts, *, target_period: int = 2024):
    return compile_us_fiscal_target_registry(
        [*_packaged_reference_facts(), *facts],
        target_period=target_period,
        allow_unaged_dollar_targets=True,
    )


def _band_specs(registry) -> dict[str, object]:
    return {
        spec.name: spec for spec in registry.specs if spec.metadata.get(FLAG) == "true"
    }


def _cbo_agi_fact(period: int, value: float) -> dict[str, object]:
    record_set_id = (
        f"cbo.revenue_projection.ty{period}.income_by_source.adjusted_gross_income"
    )
    fact = _fact(
        source_record_id=f"{record_set_id}.projected_amount",
        record_set_id=record_set_id,
        measure_id="projected_amount",
        value=value,
        period=period,
        geography_level="country",
        geography_id="0100000US",
        dimensions={},
        groupby_dimension="cbo.income_source",
        groupby_value_id="adjusted_gross_income",
        source_name="cbo",
    )
    fact["assertion"] = "source_projection"
    fact["observed_measure"]["source_concept"] = "cbo.adjusted_gross_income"
    fact["observed_measure"]["unit"] = "usd"
    return fact


def _co_name(band: str, measure_id: str, period: int = 2023) -> str:
    return f"irs_soi.ty{period}.historic_table_2.state_agi.co.{band}.{measure_id}"


# --------------------------------------------------------------------------
# The rule


def test_state_bands_bind_as_shares_of_the_bound_state_total() -> None:
    """TY2023 Colorado bands against the TY2022 state total that binds: each
    binding band is control x band / partition, landed at the control's
    period, with the share and the control named in its metadata."""
    registry = _compile(
        [
            *_partition("CO", 2023, CO_2023),
            *_totals("CO", 2022, CO_2022_TOTAL_RETURNS, CO_2022_TOTAL_AGI),
        ]
    )
    bands = _band_specs(registry)

    assert set(bands) == {
        _co_name(band, measure) for band in BINDING_BANDS for measure in MEASURES
    }
    for measure, control, column in (
        ("return_count", CO_2022_TOTAL_RETURNS, 0),
        ("adjusted_gross_income", CO_2022_TOTAL_AGI, 1),
    ):
        partition = sum(values[column] for values in CO_2023.values())
        for band in BINDING_BANDS:
            spec = bands[_co_name(band, measure)]
            share = CO_2023[band][column] / partition
            assert spec.value == pytest.approx(control * share, rel=1e-12)
            assert spec.metadata["state_fips"] == "08"
            assert spec.metadata["uprating_from_period"] == "2023"
            assert spec.metadata["uprating_to_period"] == "2022"
            assert spec.metadata["uprating_index"] == f"state_total_{measure}"
            assert spec.metadata["uprating_index_source_record_id"] == (
                f"irs_soi.ty2022.historic_table_2.state_broad.co.all.{measure}"
            )
            assert float(spec.metadata["state_agi_band_share"]) == pytest.approx(
                share, rel=1e-12
            )
            assert float(spec.metadata["uprating_factor"]) == pytest.approx(
                control / partition, rel=1e-12
            )
    top = bands[_co_name("1m_plus", "adjusted_gross_income")]
    assert top.metadata["agi_lower_bound"] == "1000000.0"
    assert top.metadata["agi_upper_bound"] == "inf"
    assert top.metadata["measure_mode"] == "sum"
    assert bands[_co_name("1m_plus", "return_count")].metadata["measure_mode"] == (
        "indicator_sum"
    )


def test_bands_below_the_floor_and_non_state_rows_never_bind() -> None:
    """The $100k floor is inclusive; national HT2 bands (Table 1.1 owns the
    national shape), filing-status slices and other HT2 state tables stay
    refused across periods."""
    us_bands = [
        _band_fact("US", 2023, band, lower, upper, measure, 1_000.0)
        for band, lower, upper in BANDS
        for measure in MEASURES
    ]
    single_filers = [
        _band_fact(
            "CO", 2023, band, lower, upper, measure, 1_000.0, filing_status="single"
        )
        for band, lower, upper in BANDS
        for measure in MEASURES
    ]
    other_table = [
        _band_fact(
            "CO", 2023, band, lower, upper, measure, 1_000.0, record_set_kind="state_x"
        )
        for band, lower, upper in BANDS
        for measure in MEASURES
    ]
    registry = _compile(
        [
            *_partition("CO", 2023, CO_2023),
            *_totals("CO", 2022, CO_2022_TOTAL_RETURNS, CO_2022_TOTAL_AGI),
            *us_bands,
            *single_filers,
            *other_table,
        ]
    )
    names = {spec.name for spec in registry.specs}

    assert {name for name in names if ".state_agi.co." in name} == {
        _co_name(band, measure) for band in BINDING_BANDS for measure in MEASURES
    }
    assert _co_name("100k_to_200k", "return_count") in names
    assert _co_name("75k_to_100k", "return_count") not in names
    assert not {name for name in names if ".historic_table_2.us." in name}
    assert not {name for name in names if ".state_x." in name}


def test_an_older_vintage_never_binds_beside_a_newer_one() -> None:
    """A collapsed TY2022 ``500k_plus`` row has no TY2023 twin, so latest-period
    selection keeps it; the pass drops it because the state's newest bands
    are TY2023. Only TY2023 bands bind, and they tile the line."""
    ty2022_values = {band: (100.0, 1_000_000.0) for band, _, _ in COLLAPSED_BANDS}
    registry = _compile(
        [
            *_partition("CO", 2022, ty2022_values, bands=COLLAPSED_BANDS),
            *_partition("CO", 2023, CO_2023),
            *_totals("CO", 2022, CO_2022_TOTAL_RETURNS, CO_2022_TOTAL_AGI),
        ]
    )
    bands = _band_specs(registry)

    assert {spec.metadata["uprating_from_period"] for spec in bands.values()} == {
        "2023"
    }
    assert not [name for name in bands if ".500k_plus." in name]
    assert len(bands) == len(BINDING_BANDS) * len(MEASURES)


def test_the_latest_vintage_binds_even_when_it_is_the_collapsed_one() -> None:
    """With only the TY2022 package the collapsed top band binds (as a share),
    and nothing below the floor does."""
    ty2022_values = {band: (100.0, 1_000_000.0) for band, _, _ in COLLAPSED_BANDS}
    registry = _compile(
        [
            *_partition("CO", 2022, ty2022_values, bands=COLLAPSED_BANDS),
            *_totals("CO", 2022, 900.0, 9_000_000.0),
        ]
    )
    bands = _band_specs(registry)

    assert {name.split(".")[5] for name in bands} == {
        "100k_to_200k",
        "200k_to_500k",
        "500k_plus",
    }
    for spec in bands.values():
        assert spec.value == pytest.approx(100.0 if "count" in spec.name else 1e6)


def test_bands_without_a_complete_partition_or_a_control_are_dropped() -> None:
    """A gap in the published partition, or no state total to rebase onto,
    drops the state's bands: never an unanchored nominal level."""
    gap = dict(CO_2023)
    gapped = [
        fact
        for fact in _partition("CO", 2023, gap)
        if ".75k_to_100k." not in str(fact["lineage"]["source_record_id"])
    ]
    registry = _compile(
        [
            *gapped,
            *_totals("CO", 2022, CO_2022_TOTAL_RETURNS, CO_2022_TOTAL_AGI),
            *_partition("WY", 2023, CO_2023),
        ]
    )

    assert _band_specs(registry) == {}


def test_congressional_district_state_totals_never_anchor_a_state() -> None:
    """The CD file's ``<st>_total`` rows are a processing-window subset with a
    wrong vintage stamp; with only those, the bands are dropped."""
    cd_record_set = "irs_soi.ty2023.congressional_district_2022.all_returns"
    registry = _compile(
        [
            *_partition("CO", 2023, CO_2023),
            _state_total_fact(
                "CO", 2023, "return_count", 1.0, record_set_id=cd_record_set
            ),
            _state_total_fact(
                "CO", 2023, "adjusted_gross_income", 1.0, record_set_id=cd_record_set
            ),
        ]
    )

    assert _band_specs(registry) == {}


def test_the_control_is_the_latest_state_total_not_after_the_target() -> None:
    registry = _compile(
        [
            *_partition("CO", 2023, CO_2023),
            *_totals("CO", 2021, 1.0, 1.0),
            *_totals("CO", 2022, CO_2022_TOTAL_RETURNS, CO_2022_TOTAL_AGI),
            *_totals("CO", 2025, 5.0, 5.0),
        ]
    )
    controls = {
        spec.metadata["uprating_index_source_record_id"]
        for spec in _band_specs(registry).values()
    }

    assert controls == {
        f"irs_soi.ty2022.historic_table_2.state_broad.co.all.{measure}"
        for measure in MEASURES
    }


def test_overlapping_bands_of_one_vintage_are_a_packaging_error() -> None:
    both_tops = [
        *_partition("CO", 2023, CO_2023),
        _band_fact("CO", 2023, "500k_plus", 500_000, None, "return_count", 53_640.0),
        *_totals("CO", 2022, CO_2022_TOTAL_RETURNS, CO_2022_TOTAL_AGI),
    ]

    with pytest.raises(ValueError, match="Overlapping state AGI bands"):
        _compile(both_tops)


def test_rebased_bands_age_with_the_state_total() -> None:
    """End to end: the AGI bands and the state AGI total take the same
    2022->2024 link, so each band stays its exact share of the aged total;
    counts never age."""
    registry = _compile(
        [
            *_partition("CO", 2023, CO_2023),
            *_totals("CO", 2022, CO_2022_TOTAL_RETURNS, CO_2022_TOTAL_AGI),
        ]
    )
    cbo = (
        _cbo_agi_fact(2022, 14_000_000_000_000),
        _cbo_agi_fact(2023, 15_000_000_000_000),
        _cbo_agi_fact(2024, 16_000_000_000_000),
    )
    aged = {
        spec.name: spec
        for spec in age_us_dollar_targets(registry, cbo, target_period=2024).specs
    }
    link = 16 / 14
    total_agi = aged[
        "irs_soi.ty2022.historic_table_2.state_broad.co.all.adjusted_gross_income"
    ]
    total_returns = aged[
        "irs_soi.ty2022.historic_table_2.state_broad.co.all.return_count"
    ]

    assert total_agi.value == pytest.approx(CO_2022_TOTAL_AGI * link, rel=1e-12)
    assert total_returns.value == CO_2022_TOTAL_RETURNS
    for band in BINDING_BANDS:
        agi = aged[_co_name(band, "adjusted_gross_income")]
        returns = aged[_co_name(band, "return_count")]
        assert float(agi.metadata["aging_factor"]) == pytest.approx(link, rel=1e-12)
        assert agi.metadata["source_period"] == "2022"
        assert agi.value / total_agi.value == pytest.approx(
            float(agi.metadata["state_agi_band_share"]), rel=1e-12
        )
        assert returns.metadata["aging_factor"] == "1"
        assert returns.value / total_returns.value == pytest.approx(
            float(returns.metadata["state_agi_band_share"]), rel=1e-12
        )


def test_release_coverage_requires_a_top_tail_row_in_every_state() -> None:
    requirement = next(
        requirement
        for requirement in US_FISCAL_TARGET_COVERAGE_REQUIREMENTS
        if requirement.requirement_id == "irs_state_agi_top_tail"
    )
    facts = []
    for postal in sorted(POSTAL_TO_FIPS):
        facts += _partition(postal, 2023, CO_2023)
        facts += _totals(postal, 2022, CO_2022_TOTAL_RETURNS, CO_2022_TOTAL_AGI)
    registry = _compile(facts)
    everything = target_profile_coverage_gate(registry.specs, [requirement])
    one_state_short = target_profile_coverage_gate(
        [
            spec
            for spec in registry.specs
            if spec.name != _co_name("1m_plus", "adjusted_gross_income")
        ],
        [requirement],
    )

    assert requirement.min_matches == 51
    assert everything.passed
    assert not one_state_short.passed


# --------------------------------------------------------------------------
# Invariants over generated states (Hypothesis is a workspace dependency; the
# wheels job installs no test extras, so these skip there).


def _generated_state_facts(draw_values, postal: str, period: int):
    values = {}
    for band, lower, _ in BANDS:
        returns = draw_values(0, 10_000_000)
        mean = (lower or 0) + draw_values(0, 5_000_000)
        agi = -draw_values(0, 10**10) if band == "under_1" else returns * mean
        values[band] = (float(returns), float(agi))
    return values


def test_rebase_invariants_hold_for_generated_partitions() -> None:
    """For generated partitions and controls: every band equals control x
    band / partition; the binding bands of a state sum to control x their
    partition share; scaling a control scales exactly that state's bands;
    and the output never depends on feed order."""
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    states = sorted(POSTAL_TO_FIPS)

    @settings(max_examples=40, deadline=None)
    @given(
        chosen=st.lists(st.sampled_from(states), min_size=1, max_size=3, unique=True),
        seed=st.integers(0, 2**32 - 1),
        scale=st.floats(0.25, 4.0, allow_nan=False, allow_infinity=False),
    )
    def check(chosen, seed, scale):
        rng = random.Random(seed)

        def draw(low, high):
            return rng.randint(low, high)

        facts = []
        expected = {}
        controls = {}
        for postal in chosen:
            values = _generated_state_facts(draw, postal, 2023)
            facts += _partition(postal, 2023, values)
            returns_total = float(draw(1, 10_000_000))
            agi_total = float(draw(1, 10**12))
            facts += _totals(postal, 2022, returns_total, agi_total)
            controls[postal] = (returns_total, agi_total)
            for column, measure in enumerate(MEASURES):
                partition = sum(value[column] for value in values.values())
                for band in BINDING_BANDS:
                    if partition == 0:
                        continue
                    expected[
                        f"irs_soi.ty2023.historic_table_2.state_agi."
                        f"{postal.lower()}.{band}.{measure}"
                    ] = controls[postal][column] * values[band][column] / partition

        bands = {
            name: spec.value for name, spec in _band_specs(_compile(facts)).items()
        }
        assert set(bands) == set(expected)
        for name, value in expected.items():
            assert bands[name] == pytest.approx(value, rel=1e-9, abs=1e-6)

        shuffled = list(facts)
        rng.shuffle(shuffled)
        reordered = {
            name: spec.value for name, spec in _band_specs(_compile(shuffled)).items()
        }
        assert reordered == pytest.approx(bands, rel=1e-12)

        first = chosen[0]
        scaled_facts = [
            {**fact, "value": fact["value"] * scale}
            if ".state_broad." in str(fact["lineage"]["source_record_id"])
            and f".{first.lower()}.all." in str(fact["lineage"]["source_record_id"])
            else fact
            for fact in facts
        ]
        scaled = {
            name: spec.value
            for name, spec in _band_specs(_compile(scaled_facts)).items()
        }
        for name, value in bands.items():
            factor = scale if f".state_agi.{first.lower()}." in name else 1.0
            assert scaled[name] == pytest.approx(value * factor, rel=1e-9, abs=1e-6)

    check()


# --------------------------------------------------------------------------
# The pinned Chronicle feed (skips where the feed is not on disk, as in CI)


def _load_repo_tool(name: str):
    path = Path(__file__).resolve().parents[3] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_tool_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pinned_feed_registry():
    feed_path = _load_repo_tool("build_us_target_parity_manifest").DEFAULT_FEED_PATH
    if not feed_path.exists():
        pytest.skip(f"pinned feed not present at {feed_path}")
    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
    from microcosm.build.us_runtime import (
        default_congressional_district_vintage_crosswalk_path,
        load_congressional_district_vintage_crosswalk,
    )
    from microcosm.build.us_runtime.chronicle_feed import load_us_chronicle_feed

    facts = load_ledger_consumer_artifact(
        feed_path,
        expected_facts_sha256=load_us_chronicle_feed().facts_sha256,
        expected_manifest_sha256=None,
    ).facts
    crosswalk = load_congressional_district_vintage_crosswalk(
        default_congressional_district_vintage_crosswalk_path()
    )
    registry = compile_us_fiscal_target_registry(
        facts,
        target_period=2024,
        congressional_district_vintage_crosswalk=crosswalk,
        age_targets=True,
    )
    return facts, registry


def test_pinned_feed_binds_every_state_band_from_ty2023(pinned_feed_registry) -> None:
    """51 states x 4 bands x 2 measures, all TY2023, each an exact share of
    its bound state total after aging; Colorado's $1M+ rows pinned."""
    _, registry = pinned_feed_registry
    bands = _band_specs(registry)
    totals = {
        (spec.metadata["state_fips"], spec.metadata["source_measure_id"]): spec
        for spec in registry.specs
        if spec.name.startswith("irs_soi.ty")
        and ".historic_table_2.state_broad." in spec.name
        and ".all." in spec.name
        and spec.metadata.get("source_measure_id") in MEASURES
    }

    assert len(bands) == 51 * len(BINDING_BANDS) * len(MEASURES)
    assert {spec.metadata["uprating_from_period"] for spec in bands.values()} == {
        "2023"
    }
    assert {name.split(".")[5] for name in bands} == set(BINDING_BANDS)
    for spec in bands.values():
        key = (spec.metadata["state_fips"], spec.metadata["source_measure_id"])
        assert (
            spec.metadata["uprating_index_source_record_id"]
            == (totals[key].metadata["ledger_source_record_id"])
        )
        assert spec.value / totals[key].value == pytest.approx(
            float(spec.metadata["state_agi_band_share"]), rel=1e-12
        )

    top = bands[_co_name("1m_plus", "adjusted_gross_income")]
    share = 47_266_883_000 / sum(agi for _, agi in CO_2023.values())
    assert float(top.metadata["state_agi_band_share"]) == pytest.approx(
        share, rel=1e-12
    )
    assert top.value == pytest.approx(
        CO_2022_TOTAL_AGI * share * float(top.metadata["aging_factor"]), rel=1e-12
    )
    top_returns = bands[_co_name("1m_plus", "return_count")]
    assert top_returns.value == pytest.approx(
        CO_2022_TOTAL_RETURNS
        * 15_610
        / sum(returns for returns, _ in CO_2023.values()),
        rel=1e-12,
    )


def test_pinned_feed_state_bands_have_means_inside_their_edges(
    pinned_feed_registry,
) -> None:
    """Every state band's aged AGI per return lies inside the band."""
    _, registry = pinned_feed_registry
    bands = _band_specs(registry)
    for name, spec in bands.items():
        if not name.endswith(".adjusted_gross_income"):
            continue
        returns = bands[name.removesuffix("adjusted_gross_income") + "return_count"]
        mean = spec.value / returns.value
        lower = float(spec.metadata["agi_lower_bound"])
        upper = float(spec.metadata["agi_upper_bound"])
        assert lower <= mean < upper, (name, mean)


def test_pinned_feed_state_top_tail_agrees_with_table_1_1(pinned_feed_registry) -> None:
    """Differential against the national owner of the AGI shape: summed over
    the states, the aged $500k-$1M and $1M+ bands come to 97%-100% of SOI
    Table 1.1's TY2023 classes aged to 2024 the same way. Table 1.1 also
    counts returns filed from other areas and Puerto Rico, which no state
    holds (1.2%-1.7% of these classes in the raw HT2 files), and the states'
    TY2022 totals grew like the nation's to TY2023 (AGI x1.0305 in HT2 and in
    Table 1.1; returns +0.19%), so the expected ratio is 0.98-0.99."""
    facts, registry = pinned_feed_registry
    bands = _band_specs(registry)
    table_1_1: dict[tuple[str, float, float], float] = {}
    for fact in facts:
        record_id = str(fact["lineage"]["source_record_id"])
        if not record_id.startswith("irs_soi.ty2023.table_1_1."):
            continue
        measure = record_id.rsplit(".", 1)[1]
        if measure not in MEASURES:
            continue
        if (fact.get("dimensions") or {}).get("filing_status", "all") != "all":
            continue
        lower, upper = -math.inf, math.inf
        for constraint in (fact.get("universe_constraints") or {}).get(
            "constraints"
        ) or []:
            if constraint["operator"] == ">=":
                lower = float(constraint["value"])
            elif constraint["operator"] == "<":
                upper = float(constraint["value"])
        if (lower, upper) != (-math.inf, math.inf):
            table_1_1[(measure, lower, upper)] = float(fact["value"])
    link = float(
        next(
            spec
            for spec in registry.specs
            if spec.name == "irs_soi.ty2023.table_1_1.all.adjusted_gross_income"
        ).metadata["aging_factor"]
    )

    for band, lower, upper in (
        ("500k_to_1m", 500_000.0, 1_000_000.0),
        ("1m_plus", 1_000_000.0, math.inf),
    ):
        for measure in MEASURES:
            national = sum(
                value
                for (class_measure, class_lower, class_upper), value in (
                    table_1_1.items()
                )
                if class_measure == measure
                and class_lower >= lower
                and class_upper <= upper
            )
            if measure == "adjusted_gross_income":
                national *= link
            states = sum(
                spec.value
                for name, spec in bands.items()
                if name.endswith(f".{band}.{measure}")
            )
            assert 0.97 <= states / national <= 1.0, (band, measure, states / national)
