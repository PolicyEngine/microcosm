"""Historic Table 2 rows rebased onto their national controls (microcosm#1243).

Fourteen Historic Table 2 (HT2) columns count the same Form 1040 line as a Pub
1304 Table 1.x column, so an HT2 state row is a share of the nation and the
newer Table 1.x row is the national level. ``_rebase_stale_soi_ht2_distributions``
rebases the state rows onto that control and retires the HT2-lineage national
rows of each rebased quantity. The synthetic tests below pin the rule and its
invariants; the pinned-feed checks sit beside the release fixture in
``test_us_fiscal_targets.py``.
"""

from __future__ import annotations

import math

import pytest

from microcosm.build.us_runtime import compile_us_fiscal_target_registry, fiscal_targets
from test_support.microcosm_build.us_fiscal_targets import (
    _dynamic_ledger_fact,
    packaged_reference_facts,
)

REGISTER = fiscal_targets._SOI_HT2_NATIONAL_CONTROLS
STATES = (("ca", "06"), ("ny", "36"), ("tx", "48"))


def _soi_fact(
    period: object,
    *,
    record_set_id: str,
    measure_id: str,
    value: float,
    geography_level: str = "country",
    geography_id: str = "0100000US",
    record_id: str | None = None,
) -> dict[str, object]:
    return _dynamic_ledger_fact(
        source_record_id=record_id or f"{record_set_id}.all.{measure_id}",
        source_name="irs_soi",
        measure_id=measure_id,
        value=value,
        period_value=period,
        geography_level=geography_level,
        geography_id=geography_id,
        dimensions={"income_range": "all", "filing_status": "all"},
        layout_record_set_id=record_set_id,
        groupby_dimension="us:statutes/26/62#adjusted_gross_income",
        groupby_value_id="all",
    )


def _ht2_us_id(measure_id: str, period: int = 2022) -> str:
    return f"irs_soi.ty{period}.historic_table_2.us.all.{measure_id}"


def _ht2_state_id(measure_id: str, postal: str, period: int = 2022) -> str:
    return f"irs_soi.ty{period}.historic_table_2.state_broad.{postal}.all.{measure_id}"


def _ht2_facts(
    measure_id: str,
    *,
    period: int = 2022,
    us: float = 1_000.0,
    states: tuple[float, ...] = (250.0, 150.0),
) -> list[dict[str, object]]:
    """An HT2 US row and up to three state rows of one measure."""
    return [
        _soi_fact(
            period,
            record_set_id=f"irs_soi.ty{period}.historic_table_2.us",
            measure_id=measure_id,
            value=us,
        ),
        *(
            _soi_fact(
                period,
                record_set_id=(
                    f"irs_soi.ty{period}.historic_table_2.state_broad.{postal}"
                ),
                measure_id=measure_id,
                value=value,
                geography_level="state",
                geography_id=f"0400000US{fips}",
            )
            for (postal, fips), value in zip(STATES, states, strict=False)
        ),
    ]


def _control_record_set(family: str, period: object) -> str:
    # Table 1.2's all-returns section is its own record set on the feed.
    suffix = ".all_returns" if family == "table_1_2" else ""
    return f"irs_soi.ty{period}.{family}{suffix}"


def _control_fact(
    measure_id: str,
    *,
    period: object = 2023,
    value: float = 800.0,
    family: str | None = None,
    record_id: str | None = None,
) -> dict[str, object]:
    rule = REGISTER[measure_id]
    return _soi_fact(
        period,
        record_set_id=_control_record_set(family or rule.control_family, period),
        measure_id=rule.control_measure_id,
        value=value,
        record_id=record_id,
    )


def _state_2022_us_fact(
    measure_id: str, value: float, *, data_year: int = 2022
) -> dict[str, object]:
    """HT2's US workbook (22in54us.xlsx) as the feed packages it: data for
    ``data_year`` stamped the following year under ``state_<data_year>``
    (PolicyEngine/chronicle#292)."""
    stamp = data_year + 1
    record_set_id = f"irs_soi.ty{stamp}.state_{data_year}.us.{measure_id}"
    return _soi_fact(
        stamp,
        record_set_id=record_set_id,
        measure_id=measure_id,
        value=value,
        record_id=f"{record_set_id}.all_returns.{measure_id}",
    )


def _compile(facts, *, target_period: int = 2024):
    return compile_us_fiscal_target_registry(
        [*packaged_reference_facts(), *facts],
        target_period=target_period,
        allow_unaged_dollar_targets=True,
    )


def _by_record(registry) -> dict[str, object]:
    return {spec.metadata["ledger_source_record_id"]: spec for spec in registry.specs}


def _controls(facts, *, target_period: int = 2024):
    return fiscal_targets._soi_ht2_national_controls(
        tuple(facts), target_period=target_period
    )


# --------------------------------------------------------------------------
# The register


def test_each_registered_column_and_its_control_bind_the_same_model_quantity() -> None:
    """A control can stand in for an HT2 column only if the compile maps both
    to the same model variable and measure mode; the register is one-to-one
    and leaves the interest and capital-gains families to their own passes."""

    def binding(measure_id: str) -> tuple[str, str] | None:
        amount = fiscal_targets.SOI_AMOUNT_MEASURE_VARIABLES.get(measure_id)
        if amount is not None:
            return ("sum", amount)
        count = fiscal_targets.SOI_RETURN_MEASURE_VARIABLES.get(measure_id)
        return None if count is None else ("indicator_sum", count)

    assert len(REGISTER) == 14
    for measure_id, rule in REGISTER.items():
        assert binding(measure_id) is not None, measure_id
        assert binding(measure_id) == binding(rule.control_measure_id), measure_id
        assert rule.concept
        assert rule.control_family in {"table_1_1", "table_1_2", "table_1_4"}
    assert len(fiscal_targets._SOI_HT2_MEASURE_BY_CONTROL) == len(REGISTER)
    assert not set(REGISTER) & {
        *fiscal_targets._SOI_TOTAL_UPRATED_AMOUNT_MEASURES,
        *fiscal_targets._SOI_TOTAL_UPRATED_RETURN_MEASURES,
        "net_capital_gains_amount",
        "net_capital_gains_returns",
    }


@pytest.mark.parametrize(
    ("record_set_id", "lineage"),
    [
        ("irs_soi.ty2022.historic_table_2.us", True),
        ("irs_soi.ty2022.historic_table_2.state_broad.ca", True),
        ("irs_soi.ty2023.state_2022.us.return_count", True),
        ("irs_soi.ty2024.state_2023.us", True),
        ("irs_soi.ty2023.state_agi", False),
        ("irs_soi.ty2023.table_1_1", False),
        ("irs_soi.ty2023.congressional_district_2022.all_returns", False),
        ("irs_soi.ty2023.state_data", False),
        ("", False),
    ],
)
def test_ht2_lineage_record_sets(record_set_id: str, lineage: bool) -> None:
    assert fiscal_targets._is_soi_ht2_lineage_record_set(record_set_id) is lineage


# --------------------------------------------------------------------------
# The rebase


@pytest.mark.parametrize("measure_id", sorted(REGISTER))
def test_ht2_state_rows_become_shares_of_their_national_control(
    measure_id: str,
) -> None:
    """CA 250 and NY 150 of an HT2 US 1,000 become 200 and 120 of an 800
    control; the HT2 US row retires and the control's own row is the only
    national level of the quantity."""
    rule = REGISTER[measure_id]
    control_id = f"{_control_record_set(rule.control_family, 2023)}.all.{
        rule.control_measure_id
    }"
    registry = _compile([*_ht2_facts(measure_id), _control_fact(measure_id)])
    specs = _by_record(registry)

    assert _ht2_us_id(measure_id) not in specs
    assert specs[control_id].value == 800.0
    ca = specs[_ht2_state_id(measure_id, "ca")]
    ny = specs[_ht2_state_id(measure_id, "ny")]
    assert (ca.value, ny.value) == (200.0, 120.0)
    assert ca.metadata["uprating_factor"] == "0.8"
    assert ca.metadata["uprating_index"] == f"total_{measure_id}"
    assert ca.metadata["uprating_from_period"] == "2022"
    # Lands at the control's period; aging owns the rest (#488).
    assert ca.metadata["uprating_to_period"] == "2023"
    assert ca.metadata["uprating_index_source_record_id"] == control_id
    assert ca.metadata["stale_distribution_rebased_to_active_total"] == "true"
    assert ca.metadata["soi_source_concept"] == rule.concept
    assert ca.metadata["soi_control_concept"] == rule.concept
    national = [
        spec
        for spec in registry.specs
        if spec.family == "irs_soi"
        and not spec.metadata.get("state_fips")
        and fiscal_targets._soi_national_quantity(spec)
        == fiscal_targets._soi_national_quantity(ca)
    ]
    assert [spec.metadata["ledger_source_record_id"] for spec in national] == [
        control_id
    ]


def test_hts_us_workbook_row_retires_with_its_quantity_only() -> None:
    """HT2's US workbook packaged as ``state_2022`` is the HT2 US total again.
    It retires when its quantity's states are rebased, and stays when only
    another quantity is (here AGI is rebased and the return count is not)."""
    workbook = _state_2022_us_fact("return_count", 1_000.0)
    workbook_id = workbook["lineage"]["source_record_id"]
    rebased_count = _compile(
        [
            *_ht2_facts("return_count"),
            workbook,
            _control_fact("return_count", value=1_010.0),
        ]
    )
    assert workbook_id not in _by_record(rebased_count)

    agi_only = _compile(
        [
            *_ht2_facts("return_count"),
            *_ht2_facts("adjusted_gross_income"),
            workbook,
            _control_fact("adjusted_gross_income", value=1_100.0),
        ]
    )
    specs = _by_record(agi_only)
    assert specs[workbook_id].value == 1_000.0
    assert specs[_ht2_us_id("return_count")].value == 1_000.0
    assert _ht2_us_id("adjusted_gross_income") not in specs


@pytest.mark.parametrize(
    "control",
    [
        None,
        # Older than the HT2 row: the rebase would run backwards.
        {"period": 2021},
        # After the build period: not yet a level for a 2024 build.
        {"period": 2025},
    ],
    ids=["no-control", "older-control", "future-control"],
)
def test_ht2_rows_keep_their_level_without_an_eligible_control(control) -> None:
    measure_id = "taxable_social_security_amount"
    facts = [*_ht2_facts(measure_id)]
    if control is not None:
        facts.append(_control_fact(measure_id, **control))
    specs = _by_record(_compile(facts))

    assert specs[_ht2_us_id(measure_id)].value == 1_000.0
    ca = specs[_ht2_state_id(measure_id, "ca")]
    assert ca.value == 250.0
    assert "uprating_factor" not in ca.metadata


def test_the_ht2_us_row_retires_even_without_state_rows() -> None:
    """The control owns the national quantity whether or not any HT2 state
    row of it compiles."""
    measure_id = "unemployment_compensation_amount"
    specs = _by_record(
        _compile([*_ht2_facts(measure_id, states=()), _control_fact(measure_id)])
    )
    assert _ht2_us_id(measure_id) not in specs


def test_ht2_rows_keep_their_level_without_an_ht2_us_total() -> None:
    measure_id = "taxable_pension_income_returns"
    facts = [*_ht2_facts(measure_id)[1:], _control_fact(measure_id)]
    specs = _by_record(_compile(facts))
    assert specs[_ht2_state_id(measure_id, "ca")].value == 250.0


def test_a_control_at_the_hts_own_period_still_rebases() -> None:
    """When HT2 and its control share a tax year, the factor is the
    same-year HT2 / Pub 1304 difference alone."""
    measure_id = "return_count"
    specs = _by_record(
        _compile(
            [
                *_ht2_facts(measure_id, period=2023),
                _control_fact(measure_id, period=2023, value=1_004.0),
            ]
        )
    )
    ca = specs[_ht2_state_id(measure_id, "ca", period=2023)]
    assert ca.value == pytest.approx(251.0, rel=1e-12)
    assert ca.metadata["uprating_to_period"] == "2023"
    assert _ht2_us_id(measure_id, period=2023) not in specs


def test_a_newer_hts_national_row_survives_an_older_control() -> None:
    """Mixed vintages (microcosm#1243 review): CA has TY2023 HT2 data, NY only
    TY2022, and the control is TY2022. NY becomes a share of the control; CA
    and the TY2023 HT2 US row keep their levels, because the control does
    not cover their year."""
    measure_id = "taxable_social_security_amount"
    facts = [
        *_ht2_facts(measure_id, period=2023, us=1_100.0, states=(500.0,)),
        _soi_fact(
            2022,
            record_set_id="irs_soi.ty2022.historic_table_2.us",
            measure_id=measure_id,
            value=1_000.0,
        ),
        _soi_fact(
            2022,
            record_set_id="irs_soi.ty2022.historic_table_2.state_broad.ny",
            measure_id=measure_id,
            value=400.0,
            geography_level="state",
            geography_id="0400000US36",
        ),
        _control_fact(measure_id, period=2022, value=800.0),
    ]
    specs = _by_record(_compile(facts))
    assert specs[_ht2_state_id(measure_id, "ny")].value == 320.0
    assert specs[_ht2_state_id(measure_id, "ca", period=2023)].value == 500.0
    assert specs[_ht2_us_id(measure_id, period=2023)].value == 1_100.0


@pytest.mark.parametrize(
    ("data_year", "retired"), [(2022, True), (2023, False)], ids=["ty2022", "ty2023"]
)
def test_the_us_workbook_retires_by_its_data_year(data_year, retired) -> None:
    """The workbook's period is stamped a year late, so its data year comes
    from ``state_<year>``: a TY2022 control retires the TY2022 workbook
    (stamped ty2023) but not a TY2023 one."""
    measure_id = "return_count"
    workbook = _state_2022_us_fact(measure_id, 1_000.0, data_year=data_year)
    specs = _by_record(
        _compile(
            [
                *_ht2_facts(measure_id),
                workbook,
                _control_fact(measure_id, period=2022, value=1_010.0),
            ]
        )
    )
    assert (workbook["lineage"]["source_record_id"] not in specs) is retired


def test_conflicting_hts_us_totals_refuse_the_compile() -> None:
    """Two HT2 US records for one key would let feed order pick the rebase
    denominator (microcosm#1243 review); one record seen twice is fine."""
    measure_id = "taxable_ira_distributions_amount"
    us, *states = _ht2_facts(measure_id, states=(600.0, 400.0))
    rival = _soi_fact(
        2022,
        record_set_id="irs_soi.ty2022.historic_table_2.us",
        measure_id=measure_id,
        value=500.0,
        record_id=f"irs_soi.ty2022.historic_table_2.us.v2.{measure_id}",
    )
    for ordered in ((us, rival), (rival, us)):
        with pytest.raises(
            fiscal_targets.AmbiguousSoiNationalControlError,
            match="Historic Table 2 US total",
        ):
            fiscal_targets._soi_ht2_us_totals(ordered)
    assert fiscal_targets._soi_ht2_us_totals((us, us)) == {
        (measure_id, "All", "all_returns", "2022"): 1_000.0
    }
    with pytest.raises(fiscal_targets.AmbiguousSoiNationalControlError):
        _compile([us, rival, *states, _control_fact(measure_id)])


def test_retirement_follows_each_rows_own_year() -> None:
    """Property over compiles: with HT2 states at one vintage, the newest
    HT2 US row at that vintage or later, an optional US workbook and any
    control vintage, (a) the states rebase iff the control is eligible for
    their year, (b) the newest US row retires iff the control is eligible
    for its year, and (c) the workbook retires iff the states' quantity was
    rebased and the control is at or after the workbook's data year."""
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @settings(max_examples=40, deadline=None)
    @given(
        st.sampled_from(sorted(REGISTER)),
        st.integers(min_value=2021, max_value=2023),
        st.integers(min_value=0, max_value=1),
        st.one_of(st.none(), st.integers(min_value=2020, max_value=2025)),
        st.one_of(st.none(), st.integers(min_value=2021, max_value=2023)),
    )
    def check(measure_id, state_year, us_lead, control_year, workbook_year) -> None:
        us_year = state_year + us_lead
        facts = _ht2_facts(measure_id, period=state_year, states=(250.0,))
        if us_lead:
            facts += _ht2_facts(measure_id, period=us_year, us=1_050.0, states=())
        if control_year is not None:
            facts.append(_control_fact(measure_id, period=control_year))
        workbook = None
        if workbook_year is not None:
            workbook = _state_2022_us_fact(measure_id, 1_000.0, data_year=workbook_year)
            facts.append(workbook)
        specs = _by_record(_compile(facts))

        def eligible(year: int) -> bool:
            return control_year is not None and year <= control_year <= 2024

        state = specs[_ht2_state_id(measure_id, "ca", period=state_year)]
        assert ("uprating_factor" in state.metadata) is eligible(state_year)
        assert (_ht2_us_id(measure_id, period=us_year) not in specs) is eligible(
            us_year
        )
        if workbook is not None:
            assert (workbook["lineage"]["source_record_id"] not in specs) is (
                eligible(state_year) and control_year >= workbook_year
            )

    check()


# --------------------------------------------------------------------------
# The control chooser


def test_no_other_family_is_ever_a_control() -> None:
    """HT2, HT2's US workbook, the congressional-district file and the
    excluding-dependents and itemized tables all carry these measures, some
    stamped later than Table 1.4; none is registered, so none anchors."""
    measure_id = "taxable_ira_distributions_amount"
    decoys = [
        _soi_fact(
            2024,
            record_set_id="irs_soi.ty2024.historic_table_2.us",
            measure_id=measure_id,
            value=1.0,
        ),
        _soi_fact(
            2023,
            record_set_id="irs_soi.ty2023.congressional_district_2022.all_returns",
            measure_id=measure_id,
            value=2.0,
            record_id=(
                "irs_soi.ty2023.congressional_district_2022.all_returns.us."
                f"{measure_id}"
            ),
        ),
        _state_2022_us_fact(measure_id, 3.0),
        _soi_fact(
            2023,
            record_set_id="irs_soi.ty2023.table_4_3.all_returns_excluding_dependents",
            measure_id=measure_id,
            value=4.0,
        ),
        _soi_fact(
            2023,
            record_set_id="irs_soi.ty2023.table_2_1.itemized_all_returns",
            measure_id=measure_id,
            value=5.0,
        ),
        # Right table, wrong column: Table 1.4's IRA count is another key.
        _control_fact("taxable_ira_distributions_returns", period=2024, value=6.0),
    ]
    assert measure_id not in {key[0] for key in _controls(decoys)}
    control = _control_fact(measure_id, period=2022, value=7.0)
    (chosen,) = (
        value
        for key, value in _controls([*decoys, control]).items()
        if key[0] == measure_id
    )
    assert (chosen.source_period, chosen.value) == ("2022", 7.0)


def test_the_control_refuses_a_tie_and_accepts_one_record_twice() -> None:
    measure_id = "return_count"
    control = _control_fact(measure_id, value=10.0)
    rival = _control_fact(
        measure_id,
        value=10.0,
        record_id="irs_soi.ty2023.table_1_1.v2.return_count",
    )
    for ordered in ((control, rival), (rival, control)):
        with pytest.raises(
            fiscal_targets.AmbiguousSoiNationalControlError,
            match="table_1_1.v2.return_count",
        ):
            _controls(ordered)
    assert len(_controls((control, control))) == 1
    # A later control supersedes an older tie.
    older = (
        _control_fact(measure_id, period=2022, value=1.0),
        _control_fact(
            measure_id,
            period=2022,
            value=2.0,
            record_id="irs_soi.ty2022.table_1_1.b.return_count",
        ),
    )
    (chosen,) = _controls((*older, control)).values()
    assert chosen.value == 10.0


def test_the_control_is_order_free_latest_and_registered() -> None:
    """Property: for any national facts across the registered and decoy
    families and any build period, the control per HT2 measure is the latest
    fact of its registered table and column not after the build period,
    whatever the feed order; with none, there is no control."""
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    families = (
        "table_1_1",
        "table_1_2",
        "table_1_4",
        "historic_table_2",
        "congressional_district_2022",
        "state_2022",
        "table_4_3",
    )
    columns = sorted(
        {*REGISTER, *(rule.control_measure_id for rule in REGISTER.values())}
    )
    candidate = st.tuples(
        st.sampled_from(families),
        st.integers(min_value=2018, max_value=2027),
        st.sampled_from(columns),
        st.floats(min_value=1.0, max_value=1e13, allow_nan=False),
    )

    def record_set(family: str, period: int) -> str:
        if family == "table_1_2":
            return f"irs_soi.ty{period}.table_1_2.all_returns"
        if family == "historic_table_2":
            return f"irs_soi.ty{period}.historic_table_2.us"
        if family == "congressional_district_2022":
            return f"irs_soi.ty{period}.congressional_district_2022.all_returns"
        if family == "state_2022":
            return f"irs_soi.ty{period}.state_2022.us"
        return f"irs_soi.ty{period}.{family}"

    @settings(max_examples=300, deadline=None)
    @given(
        st.lists(candidate, max_size=20, unique_by=lambda item: item[:3]),
        st.integers(min_value=2019, max_value=2026),
        st.randoms(use_true_random=False),
    )
    def check(candidates, target_period, rng) -> None:
        facts = [
            _soi_fact(
                period,
                record_set_id=record_set(family, period),
                measure_id=column,
                value=value,
            )
            for family, period, column, value in candidates
        ]
        shuffled = list(facts)
        rng.shuffle(shuffled)
        controls = _controls(facts, target_period=target_period)
        assert controls == _controls(shuffled, target_period=target_period)
        chosen = {key[0]: control for key, control in controls.items()}
        for measure_id, rule in REGISTER.items():
            eligible = [
                (period, value)
                for family, period, column, value in candidates
                if family == rule.control_family
                and column == rule.control_measure_id
                and period <= target_period
            ]
            if not eligible:
                assert measure_id not in chosen
                continue
            period, value = max(eligible)
            control = chosen[measure_id]
            assert (control.source_period, control.value) == (str(period), value)
            assert control.source_record_id == (
                f"{record_set(rule.control_family, period)}.all."
                f"{rule.control_measure_id}"
            )
        assert set(chosen) <= set(REGISTER)

    check()


# --------------------------------------------------------------------------
# Invariants over compiles


def test_rebased_states_conserve_shares_and_one_national_level_remains() -> None:
    """Property over compiles, for every registered column, any HT2 vintage,
    any control vintage (or none) and any feed order:

    * **Shares conserved.** With an eligible control (not after the build
      period, not before the HT2 row), each state row is its HT2 value times
      control / HT2 US, so the states keep their HT2 shares and sum to the
      control times the states' share of the HT2 nation.
    * **States never exceed the nation.** When the HT2 states sum to at most
      the HT2 US total, the rebased states sum to at most the control.
    * **One national level.** The control's row is the only national
      all-AGI row of the quantity: the HT2 US row and HT2's US workbook row
      retire.
    * **No eligible control, no change.** Otherwise every HT2 row keeps its
      level and the HT2 US row stays.
    * **Order-free.** Shuffling the facts compiles the same registry.
    """
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @settings(max_examples=40, deadline=None)
    @given(
        st.sampled_from(sorted(REGISTER)),
        st.integers(min_value=2021, max_value=2024),
        st.one_of(st.none(), st.integers(min_value=2020, max_value=2026)),
        st.lists(
            st.integers(min_value=1, max_value=10_000_000), min_size=1, max_size=3
        ),
        st.integers(min_value=0, max_value=5_000_000),
        st.integers(min_value=1, max_value=50_000_000),
        st.booleans(),
        st.randoms(use_true_random=False),
    )
    def check(
        measure_id,
        ht2_period,
        control_period,
        state_values,
        other_areas,
        control_value,
        with_workbook,
        rng,
    ) -> None:
        national = float(sum(state_values) + other_areas)
        facts = _ht2_facts(
            measure_id,
            period=ht2_period,
            us=national,
            states=tuple(float(value) for value in state_values),
        )
        if with_workbook:
            # HT2's US workbook for the same data year, stamped a year late.
            facts.append(
                _state_2022_us_fact(measure_id, national, data_year=ht2_period)
            )
        if control_period is not None:
            facts.append(
                _control_fact(
                    measure_id, period=control_period, value=float(control_value)
                )
            )
        shuffled = list(facts)
        rng.shuffle(shuffled)
        registry = _compile(facts)
        assert sorted(
            (spec.name, spec.value, tuple(sorted(spec.metadata.items())))
            for spec in registry.specs
        ) == sorted(
            (spec.name, spec.value, tuple(sorted(spec.metadata.items())))
            for spec in _compile(shuffled).specs
        )

        specs = _by_record(registry)
        states = [
            specs[_ht2_state_id(measure_id, postal, period=ht2_period)]
            for (postal, _), _ in zip(STATES, state_values, strict=False)
        ]
        quantity = fiscal_targets._soi_national_quantity(states[0])
        national_rows = [
            spec
            for spec in registry.specs
            if spec.family == "irs_soi"
            and not spec.metadata.get("state_fips")
            and fiscal_targets._soi_national_quantity(spec) == quantity
        ]
        eligible = control_period is not None and ht2_period <= control_period <= 2024
        if eligible:
            factor = control_value / national
            for spec, value in zip(states, state_values, strict=False):
                assert math.isclose(spec.value, value * factor, rel_tol=1e-12)
            total = sum(spec.value for spec in states)
            assert math.isclose(
                total,
                control_value * sum(state_values) / national,
                rel_tol=1e-12,
            )
            assert total <= control_value * (1 + 1e-12)
            assert [spec.value for spec in national_rows] == [float(control_value)]
        else:
            assert [spec.value for spec in states] == [
                float(value) for value in state_values
            ]
            assert specs[_ht2_us_id(measure_id, period=ht2_period)].value == national

    check()
