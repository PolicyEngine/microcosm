"""Congressional-district rows rebased onto the national controls (#1250).

The congressional-district file (``22incd.csv``) publishes the columns of
``_SOI_HT2_NATIONAL_CONTROLS`` at the same Form 1040 lines, so its state and
district rows are shares of the nation and its US row is a second, narrower
national level beside the Table 1.x control.
``_rebase_stale_soi_district_distributions`` takes the rows as shares of the
file's own US row times that control and retires the US row. The synthetic
tests below pin the rule and its invariants; the pinned-feed checks sit
beside the release fixture in ``test_us_fiscal_targets.py``.
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
#: The measures the TY2022 district package carries: the register less wages
#: and taxable income, which it does not publish.
CD_MEASURES = sorted(
    set(REGISTER)
    - {"wages_salaries_amount", "wages_salaries_returns", "taxable_income_amount"}
)
#: States with two districts in the 117th and 119th plans, so a state row
#: and both its districts make a complete hierarchy block.
STATES = (
    ("hi", "15", ("1501", "1502")),
    ("nh", "33", ("3301", "3302")),
    ("me", "23", ("2301", "2302")),
)
_REVIEWED = object()


def _record_set(stamp: int) -> str:
    return f"irs_soi.ty{stamp}.congressional_district_2022.all_returns"


def _us_id(measure_id: str, stamp: int = 2023, row: str = "us") -> str:
    return f"{_record_set(stamp)}.{row}.{measure_id}"


def _cd_fact(
    measure_id: str,
    value: float,
    *,
    row: str,
    level: str,
    geography_id: str,
    stamp: int = 2023,
    column: object = _REVIEWED,
) -> dict[str, object]:
    fact = _dynamic_ledger_fact(
        source_record_id=f"{_record_set(stamp)}.{row}.{measure_id}",
        source_name="irs_soi",
        measure_id=measure_id,
        value=value,
        period_value=stamp,
        geography_level=level,
        geography_id=geography_id,
        dimensions={"income_range": "all", "filing_status": "all"},
        layout_record_set_id=_record_set(stamp),
        groupby_dimension="irs_soi.congressional_district",
        groupby_value_id=row,
    )
    if column is _REVIEWED:
        column = REGISTER[measure_id].source_column
    if column is not None:
        fact["layout"]["source_column_id"] = column
    return fact


def _cd_facts(
    measure_id: str,
    *,
    stamp: int = 2023,
    us: float | None = 1_000.0,
    districts: tuple[tuple[float, float], ...] = ((300.0, 200.0), (150.0, 100.0)),
    column: object = _REVIEWED,
) -> list[dict[str, object]]:
    """The district file's US row and, per state, its total and two districts."""
    facts = []
    if us is not None:
        facts.append(
            _cd_fact(
                measure_id,
                us,
                row="us",
                level="country",
                geography_id="0100000US",
                stamp=stamp,
                column=column,
            )
        )
    for (postal, fips, geoids), values in zip(STATES, districts, strict=False):
        facts.append(
            _cd_fact(
                measure_id,
                math.fsum(values),
                row=f"{postal}_total",
                level="state",
                geography_id=f"0400000US{fips}",
                stamp=stamp,
                column=column,
            )
        )
        for geoid, value in zip(geoids, values, strict=True):
            facts.append(
                _cd_fact(
                    measure_id,
                    value,
                    row=f"{postal}_{geoid[2:]}",
                    level="congressional_district",
                    geography_id=f"5001700US{geoid}",
                    stamp=stamp,
                    column=column,
                )
            )
    return facts


def _control_fact(
    measure_id: str, *, period: int = 2023, value: float = 1_100.0
) -> dict[str, object]:
    rule = REGISTER[measure_id]
    suffix = ".all_returns" if rule.control_family == "table_1_2" else ""
    record_set_id = f"irs_soi.ty{period}.{rule.control_family}{suffix}"
    return _dynamic_ledger_fact(
        source_record_id=f"{record_set_id}.all.{rule.control_measure_id}",
        source_name="irs_soi",
        measure_id=rule.control_measure_id,
        value=value,
        period_value=period,
        dimensions={"income_range": "all", "filing_status": "all"},
        layout_record_set_id=record_set_id,
        groupby_dimension="us:statutes/26/62#adjusted_gross_income",
        groupby_value_id="all",
    )


def _control_id(measure_id: str, period: int = 2023) -> str:
    return _control_fact(measure_id, period=period)["lineage"]["source_record_id"]


def _ht2_facts(
    measure_id: str, *, us: float, states: tuple[float, ...], period: int = 2022
) -> list[dict[str, object]]:
    """An HT2 US row and state rows for the first ``len(states)`` STATES."""
    facts = [
        _dynamic_ledger_fact(
            source_record_id=f"irs_soi.ty{period}.historic_table_2.us.all.{measure_id}",
            source_name="irs_soi",
            measure_id=measure_id,
            value=us,
            period_value=period,
            dimensions={"income_range": "all", "filing_status": "all"},
            layout_record_set_id=f"irs_soi.ty{period}.historic_table_2.us",
            groupby_dimension="us:statutes/26/62#adjusted_gross_income",
            groupby_value_id="all",
        )
    ]
    for (postal, fips, _), value in zip(STATES, states, strict=False):
        record_set_id = f"irs_soi.ty{period}.historic_table_2.state_broad.{postal}"
        facts.append(
            _dynamic_ledger_fact(
                source_record_id=f"{record_set_id}.all.{measure_id}",
                source_name="irs_soi",
                measure_id=measure_id,
                value=value,
                period_value=period,
                geography_level="state",
                geography_id=f"0400000US{fips}",
                dimensions={"income_range": "all", "filing_status": "all"},
                layout_record_set_id=record_set_id,
                groupby_dimension="us:statutes/26/62#adjusted_gross_income",
                groupby_value_id="all",
            )
        )
    return facts


def _ht2_state_id(measure_id: str, postal: str, period: int = 2022) -> str:
    return f"irs_soi.ty{period}.historic_table_2.state_broad.{postal}.all.{measure_id}"


def _compile(facts, *, crosswalk=None):
    return compile_us_fiscal_target_registry(
        [*packaged_reference_facts(), *facts],
        target_period=2024,
        congressional_district_vintage_crosswalk=crosswalk,
        allow_unaged_dollar_targets=True,
    )


def _by_record(registry) -> dict[str, object]:
    return {spec.metadata["ledger_source_record_id"]: spec for spec in registry.specs}


def _cd_rows(registry, measure_id: str) -> dict[str, object]:
    """The district-file rows of one measure, keyed by their row id."""
    return {
        spec.metadata["ledger_source_record_id"].split(".")[-2]: spec
        for spec in registry.specs
        if spec.metadata.get("source_measure_id") == measure_id
        and fiscal_targets._soi_record_set_family(
            spec.metadata.get("ledger_layout_record_set_id", "")
        )
        == "congressional_district"
    }


def _national_rows(registry, quantity) -> list[object]:
    return [
        spec
        for spec in registry.specs
        if spec.family == "irs_soi"
        and not spec.metadata.get("state_fips")
        and fiscal_targets._soi_national_quantity(spec) == quantity
    ]


# --------------------------------------------------------------------------
# The register


#: Each registered measure's IRS column in ``22incddocguide.docx`` (TY2022),
#: with the Form 1040 line it gives; the HT2 guides give the same column at
#: the same line (``docs/us-soi-national-controls.md``).
_DISTRICT_GUIDE_COLUMNS = {
    "return_count": "N1",  # number of returns
    "adjusted_gross_income": "A00100",  # 1040:11
    "wages_salaries_amount": "A00200",  # 1040:1z
    "wages_salaries_returns": "N00200",  # 1040:1z
    "taxable_ira_distributions_amount": "A01400",  # 1040:4b
    "taxable_ira_distributions_returns": "N01400",  # 1040:4b
    "taxable_pension_income_amount": "A01700",  # 1040:5b
    "taxable_pension_income_returns": "N01700",  # 1040:5b
    "unemployment_compensation_amount": "A02300",  # Schedule 1:7
    "unemployment_compensation_returns": "N02300",  # Schedule 1:7
    "taxable_social_security_amount": "A02500",  # 1040:6b
    "taxable_social_security_returns": "N02500",  # 1040:6b
    "itemized_deductions_amount": "A04470",  # 1040:12
    "taxable_income_amount": "A04800",  # 1040:15
}


def test_the_register_names_each_columns_reviewed_irs_id() -> None:
    assert {
        measure_id: rule.source_column for measure_id, rule in REGISTER.items()
    } == _DISTRICT_GUIDE_COLUMNS
    assert len(CD_MEASURES) == 11


# --------------------------------------------------------------------------
# The rebase


@pytest.mark.parametrize("measure_id", CD_MEASURES)
def test_district_rows_become_shares_of_their_national_control(
    measure_id: str,
) -> None:
    """A US row of 1,000 with HI 300 + 200 and NH 150 + 100 under an 1,100
    control: every row gains x1.1, the US row retires, and the control's row
    is the quantity's only national level."""
    registry = _compile([*_cd_facts(measure_id), _control_fact(measure_id)])
    rows = _cd_rows(registry, measure_id)

    assert "us" not in rows
    expected = {
        "hi_total": 550.0,
        "hi_01": 330.0,
        "hi_02": 220.0,
        "nh_total": 275.0,
        "nh_01": 165.0,
        "nh_02": 110.0,
    }
    assert {row: spec.value for row, spec in rows.items()} == pytest.approx(
        expected, rel=1e-12
    )
    concept = REGISTER[measure_id].concept
    for row, spec in rows.items():
        metadata = spec.metadata
        assert metadata["uprating_factor"] == "1.1"
        assert metadata["uprating_index"] == f"total_{measure_id}"
        assert metadata["uprating_from_period"] == "2023"
        assert metadata["uprating_to_period"] == "2023"
        assert metadata["uprating_index_source_record_id"] == _control_id(measure_id)
        assert metadata["soi_share_total_source_record_id"] == _us_id(measure_id)
        assert metadata["soi_source_concept"] == concept
        assert metadata["soi_control_concept"] == concept
        assert metadata["stale_distribution_rebased_to_active_total"] == "true"
        if row.endswith(("_01", "_02")):
            # One factor scales the states and their districts alike.
            assert metadata["hierarchy_reconciliation_factor"] == "1"
    assert [
        spec.metadata["ledger_source_record_id"]
        for spec in _national_rows(
            registry, fiscal_targets._soi_national_quantity(rows["hi_total"])
        )
    ] == [_control_id(measure_id)]


def test_each_family_divides_by_its_own_us_row() -> None:
    """HT2 rows divide by the HT2 US row and district rows by the district
    US row, never by the other's; both US rows retire."""
    measure_id = "taxable_social_security_amount"
    registry = _compile(
        [
            *_cd_facts(measure_id),
            *_ht2_facts(measure_id, us=900.0, states=(400.0, 200.0)),
            _control_fact(measure_id),
        ]
    )
    specs = _by_record(registry)
    rows = _cd_rows(registry, measure_id)

    assert specs[_ht2_state_id(measure_id, "hi")].value == pytest.approx(
        400.0 * 1_100.0 / 900.0, rel=1e-12
    )
    assert rows["hi_total"].value == pytest.approx(550.0, rel=1e-12)
    assert rows["hi_total"].metadata["soi_share_total_source_record_id"] == _us_id(
        measure_id
    )
    assert [
        spec.metadata["ledger_source_record_id"]
        for spec in _national_rows(
            registry, fiscal_targets._soi_national_quantity(rows["hi_total"])
        )
    ] == [_control_id(measure_id)]


@pytest.mark.parametrize(
    "control",
    [
        None,
        # Older than the row's (stamped) period: the share would run backwards.
        {"period": 2022},
        # After the build period: not yet a level for a 2024 build.
        {"period": 2025},
    ],
    ids=["no-control", "older-control", "future-control"],
)
def test_district_rows_keep_their_level_without_an_eligible_control(control) -> None:
    measure_id = "adjusted_gross_income"
    facts = _cd_facts(measure_id)
    if control is not None:
        facts.append(_control_fact(measure_id, **control))
    rows = _cd_rows(_compile(facts), measure_id)

    assert rows["us"].value == 1_000.0
    assert rows["hi_total"].value == 500.0
    assert rows["hi_01"].value == 300.0
    assert "uprating_factor" not in rows["hi_total"].metadata


@pytest.mark.parametrize("us", [None, 0.0], ids=["no-us-row", "zero-us-row"])
def test_district_rows_without_a_denominator_are_dropped_beside_a_control(us) -> None:
    """A row with an eligible control but no nonzero US row of its own file
    cannot be a share, and as a level it would be the stale second value the
    pass removes, so it is dropped (as #1229 and #1040 do)."""
    measure_id = "return_count"
    registry = _compile([*_cd_facts(measure_id, us=us), _control_fact(measure_id)])
    assert _cd_rows(registry, measure_id) == {}
    assert _control_id(measure_id) in _by_record(registry)


def test_the_share_ignores_the_package_stamp() -> None:
    """The feed stamps 22incd.csv ty2023; stamped truthfully at 2022 the
    shares and their landing period are the same, because the stamp cancels
    in row / US row."""
    measure_id = "unemployment_compensation_amount"
    stamped, truthful = (
        _cd_rows(
            _compile([*_cd_facts(measure_id, stamp=stamp), _control_fact(measure_id)]),
            measure_id,
        )
        for stamp in (2023, 2022)
    )
    assert set(stamped) == set(truthful)
    for row, spec in stamped.items():
        assert spec.value == truthful[row].value
        assert spec.metadata["uprating_to_period"] == "2023"
        assert truthful[row].metadata["uprating_to_period"] == "2023"
        assert truthful[row].metadata["uprating_from_period"] == "2022"
        assert truthful[row].metadata["soi_share_total_source_record_id"] == _us_id(
            measure_id, stamp=2022
        )


# --------------------------------------------------------------------------
# The column guard


@pytest.mark.parametrize("column", ["N2", "A00101", None], ids=str)
def test_a_district_fact_under_an_unreviewed_column_refuses(column) -> None:
    """Rebasing a measure id that reads another IRS column (or none) would
    bind the wrong concept to the control, so the compile refuses."""
    measure_id = "return_count"
    facts = _cd_facts(measure_id)
    facts[1]["layout"].pop("source_column_id")
    if column is not None:
        facts[1]["layout"]["source_column_id"] = column
    with pytest.raises(
        fiscal_targets.UnreviewedSoiNationalControlColumnError,
        match=facts[1]["lineage"]["source_record_id"],
    ):
        _compile([*facts, _control_fact(measure_id)])
    # Without a control nothing is rebased, so nothing new is claimed.
    rows = _cd_rows(_compile(facts), measure_id)
    assert rows["hi_total"].value == 500.0


@pytest.mark.parametrize("control_period", [2022, 2025], ids=["older", "future"])
def test_the_column_guard_ignores_rows_without_an_eligible_control(
    control_period: int,
) -> None:
    """The guard checks exactly the rows the pass would rebase: beside an
    older or a future control the rows keep their levels, so an unreviewed
    column is not refused."""
    measure_id = "taxable_social_security_returns"
    facts = _cd_facts(measure_id, column="N02510")
    rows = _cd_rows(
        _compile([*facts, _control_fact(measure_id, period=control_period)]),
        measure_id,
    )
    assert rows["us"].value == 1_000.0
    assert rows["hi_total"].value == 500.0


def test_the_column_guard_sees_each_source_district_before_the_crosswalk() -> None:
    """The crosswalk merges source districts into one fact that keeps the
    first contributor's layout, so a mislabeled second contributor would hide
    behind a reviewed column; the guard runs on the facts as fed."""
    measure_id = "taxable_pension_income_amount"
    facts = _cd_facts(measure_id, districts=((300.0, 200.0),))
    hi_02 = next(
        fact
        for fact in facts
        if fact["lineage"]["source_record_id"].endswith(f"hi_02.{measure_id}")
    )
    hi_02["layout"]["source_column_id"] = "A01750"
    crosswalk = [
        {
            "source_geography_id": "5001700US1501",
            "target_geography_id": "5001900US1501",
            "weight": 1.0,
        },
        {
            "source_geography_id": "5001700US1502",
            "target_geography_id": "5001900US1501",
            "weight": 1.0,
        },
    ]
    with pytest.raises(fiscal_targets.UnreviewedSoiNationalControlColumnError):
        _compile([*facts, _control_fact(measure_id)], crosswalk=crosswalk)


def _cd_agi_slice_fact(
    measure_id: str,
    value: float,
    *,
    stamp: int,
    column: object = _REVIEWED,
    national: bool = False,
) -> dict[str, object]:
    """The district file's $50,000-$75,000 AGI slice (agi_stub 5), for Hawaii
    or for the US row."""
    row = "us_50k_to_75k" if national else "hi_total_50k_to_75k"
    fact = _dynamic_ledger_fact(
        source_record_id=f"{_record_set(stamp)}.{row}.{measure_id}",
        source_name="irs_soi",
        measure_id=measure_id,
        value=value,
        period_value=stamp,
        geography_level="country" if national else "state",
        geography_id="0100000US" if national else "0400000US15",
        dimensions={"income_range": "50k_to_75k", "filing_status": "all"},
        universe_constraints=[
            {"variable": "adjusted_gross_income", "operator": ">=", "value": 50_000},
            {"variable": "adjusted_gross_income", "operator": "<", "value": 75_000},
        ],
        layout_record_set_id=_record_set(stamp),
        groupby_dimension="irs_soi.congressional_district",
        groupby_value_id=row,
    )
    fact["layout"]["source_column_id"] = (
        REGISTER[measure_id].source_column if column is _REVIEWED else column
    )
    return fact


def test_an_agi_slice_of_the_file_takes_the_same_factor() -> None:
    """A district-file AGI slice is a share of the same all-AGI US row, so it
    takes the measure's one factor (as #1229 and #1040 rebase district
    slices) and the column guard checks it. The pinned package has no
    slices, and a slice compiles only at its own period (here 2024)."""
    measure_id = "adjusted_gross_income"

    def facts(column: object = _REVIEWED) -> list[dict[str, object]]:
        return [
            *_cd_facts(measure_id, stamp=2024),
            _cd_agi_slice_fact(measure_id, 120.0, stamp=2024, column=column),
            _cd_agi_slice_fact(measure_id, 240.0, stamp=2024, national=True),
            _control_fact(measure_id, period=2024),
        ]

    specs = _by_record(_compile(facts()))
    state_slice = specs[f"{_record_set(2024)}.hi_total_50k_to_75k.{measure_id}"]
    us_slice = specs[f"{_record_set(2024)}.us_50k_to_75k.{measure_id}"]
    assert state_slice.metadata["agi_lower_bound"] == "50000.0"
    # The US slice is a share too, never a second denominator.
    assert (state_slice.value, us_slice.value) == pytest.approx(
        (132.0, 264.0), rel=1e-12
    )
    for spec in (state_slice, us_slice):
        assert spec.metadata["soi_share_total_source_record_id"] == _us_id(
            measure_id, stamp=2024
        )
    assert _us_id(measure_id, stamp=2024) not in specs
    with pytest.raises(fiscal_targets.UnreviewedSoiNationalControlColumnError):
        _compile(facts(column="A00101"))


def test_the_column_guard_skips_facts_that_compile_to_no_target() -> None:
    """The guard checks the rows the pass rebases (#1250 review). A stale AGI
    slice compiles to no target at another period, and a superseded vintage
    loses latest-vintage selection, so neither is checked whatever its
    column."""
    measure_id = "adjusted_gross_income"
    stale_slice = _cd_agi_slice_fact(measure_id, 120.0, stamp=2023, column="A00101")
    specs = _by_record(
        _compile([*_cd_facts(measure_id), stale_slice, _control_fact(measure_id)])
    )
    assert stale_slice["lineage"]["source_record_id"] not in specs
    assert specs[f"{_record_set(2023)}.hi_total.{measure_id}"].value == 550.0

    measure_id = "return_count"
    superseded = _cd_facts(measure_id, stamp=2022, column="N2")
    current = _cd_facts(measure_id, stamp=2024)
    rows = _cd_rows(
        _compile([*superseded, *current, _control_fact(measure_id, period=2024)]),
        measure_id,
    )
    assert rows["hi_total"].metadata["ledger_source_record_id"] == (
        f"{_record_set(2024)}.hi_total.{measure_id}"
    )
    assert rows["hi_total"].value == 550.0


# --------------------------------------------------------------------------
# The denominator


def test_conflicting_district_us_totals_refuse_the_compile() -> None:
    """Two US rows of one key with different values would let feed order pick
    the denominator."""
    measure_id = "taxable_ira_distributions_amount"
    us, *rest = _cd_facts(measure_id)
    rival = _cd_fact(
        measure_id, 900.0, row="us_v2", level="country", geography_id="0100000US"
    )
    for ordered in ((us, rival), (rival, us)):
        with pytest.raises(
            fiscal_targets.AmbiguousSoiNationalControlError,
            match="congressional-district US total",
        ):
            fiscal_targets._soi_district_us_totals(ordered)
    with pytest.raises(fiscal_targets.AmbiguousSoiNationalControlError):
        _compile([rival, us, *rest, _control_fact(measure_id)])


def test_rows_never_borrow_another_district_files_us_row() -> None:
    """Each package divides by its own US row (#1250 review). A 22incd
    package without one keeps no rows beside the control, even when another
    district file at the same stamp carries a US row of the measure."""
    measure_id = "return_count"
    other = _cd_fact(
        measure_id, 2_000.0, row="us", level="country", geography_id="0100000US"
    )
    other_set = "irs_soi.ty2023.congressional_district_2023.all_returns"
    other["layout"]["record_set_id"] = other_set
    other["lineage"]["source_record_id"] = f"{other_set}.us.{measure_id}"
    registry = _compile(
        [*_cd_facts(measure_id, us=None), other, _control_fact(measure_id)]
    )
    assert _cd_rows(registry, measure_id) == {}


def test_equal_district_us_totals_name_the_smaller_record_id() -> None:
    """Equal twins are one total; the named denominator is the smaller record
    id in either feed order, and both twins retire."""
    measure_id = "taxable_ira_distributions_returns"
    us, *rest = _cd_facts(measure_id)
    twin = _cd_fact(
        measure_id, 1_000.0, row="us_2", level="country", geography_id="0100000US"
    )
    named = set()
    for ordered in ((us, twin), (twin, us)):
        registry = _compile([*ordered, *rest, _control_fact(measure_id)])
        rows = _cd_rows(registry, measure_id)
        assert not {"us", "us_2"} & set(rows)
        named |= {
            spec.metadata["soi_share_total_source_record_id"] for spec in rows.values()
        }
    assert named == {min(_us_id(measure_id), _us_id(measure_id, row="us_2"))}


# --------------------------------------------------------------------------
# Invariants over compiles


def test_district_shares_conserve_the_control_and_leave_one_national_level() -> None:
    """Property over compiles, for every carried column, either package
    stamp, any control vintage (or none), any states, other-area residual,
    HT2 rows alongside or not, an equal twin US row or not, and any feed
    order:

    * **Shares conserved.** With an eligible control (not after the build
      period, not before the row's period), each district-file row is its
      value times control / the file's US row, and names the (smallest) US
      row id as its denominator. The states sum to the control times their
      share of the US row, so never past the control, and each state's
      districts sum to the state with reconciliation factor 1.
    * **Own denominator.** HT2 rows divide by the HT2 US row.
    * **One national level.** The control's row is the only national
      all-AGI row of the quantity: every district US row and the HT2 US row
      retire.
    * **No eligible control, no change.** The rows and the US row keep
      their levels.
    * **Order-free.** Shuffling the facts compiles the same registry.
    """
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    pairs = st.tuples(
        st.integers(min_value=1, max_value=5_000_000),
        st.integers(min_value=1, max_value=5_000_000),
    )

    @settings(max_examples=40, deadline=None)
    @given(
        st.sampled_from(CD_MEASURES),
        st.sampled_from((2022, 2023)),
        st.one_of(st.none(), st.integers(min_value=2020, max_value=2026)),
        st.integers(min_value=1, max_value=50_000_000),
        st.lists(pairs, min_size=1, max_size=3),
        st.integers(min_value=0, max_value=5_000_000),
        st.booleans(),
        st.sampled_from((None, "aa_us", "zz_us")),
        st.randoms(use_true_random=False),
    )
    def check(
        measure_id,
        stamp,
        control_period,
        control_value,
        districts,
        other_areas,
        with_ht2,
        twin_row,
        rng,
    ) -> None:
        state_totals = [math.fsum(pair) for pair in districts]
        us_value = float(sum(state_totals) + other_areas)
        facts = _cd_facts(
            measure_id,
            stamp=stamp,
            us=us_value,
            districts=tuple(tuple(map(float, pair)) for pair in districts),
        )
        if twin_row is not None:
            facts.append(
                _cd_fact(
                    measure_id,
                    us_value,
                    row=twin_row,
                    level="country",
                    geography_id="0100000US",
                    stamp=stamp,
                )
            )
        ht2_states = tuple(2.0 * total + 1.0 for total in state_totals)
        ht2_us = math.fsum(ht2_states) + 7.0
        if with_ht2:
            facts += _ht2_facts(measure_id, us=ht2_us, states=ht2_states)
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

        rows = _cd_rows(registry, measure_id)
        raw = {
            fact["layout"]["groupby_value_id"]: float(fact["value"])
            for fact in facts
            if fact["layout"]["record_set_id"] == _record_set(stamp)
        }
        eligible = control_period is not None and stamp <= control_period <= 2024
        us_rows = {"us", *([twin_row] if twin_row else [])}
        if not eligible:
            assert {row: spec.value for row, spec in rows.items()} == raw
            assert not any("uprating_factor" in s.metadata for s in rows.values())
            return

        factor = control_value / us_value
        assert not us_rows & set(rows)
        assert set(rows) == set(raw) - us_rows
        denominator = min(_us_id(measure_id, stamp, row) for row in us_rows)
        for row, spec in rows.items():
            assert math.isclose(spec.value, raw[row] * factor, rel_tol=1e-12)
            assert spec.metadata["soi_share_total_source_record_id"] == denominator
            assert spec.metadata["uprating_to_period"] == str(control_period)
        state_sum = math.fsum(
            spec.value for row, spec in rows.items() if row.endswith("_total")
        )
        assert math.isclose(
            state_sum,
            control_value * sum(state_totals) / us_value,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        assert state_sum <= control_value * (1 + 1e-12)
        for postal, _, _ in STATES[: len(districts)]:
            children = [rows[f"{postal}_01"], rows[f"{postal}_02"]]
            assert math.isclose(
                math.fsum(spec.value for spec in children),
                rows[f"{postal}_total"].value,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            for spec in children:
                assert spec.metadata.get("hierarchy_reconciliation_factor") in (
                    None,
                    "1",
                )
        if with_ht2:
            specs = _by_record(registry)
            for (postal, _, _), value in zip(STATES, ht2_states, strict=False):
                assert math.isclose(
                    specs[_ht2_state_id(measure_id, postal)].value,
                    value * control_value / ht2_us,
                    rel_tol=1e-12,
                )
        quantity = fiscal_targets._soi_national_quantity(rows[f"{STATES[0][0]}_total"])
        assert [
            spec.metadata["ledger_source_record_id"]
            for spec in _national_rows(registry, quantity)
        ] == [_control_id(measure_id, control_period)]

    check()


def test_the_district_and_hts_passes_agree_on_the_same_shares() -> None:
    """Differential: the HT2 pass (#1247) and the district pass implement one
    share rule. Fed the same state values and US total under both families,
    at the same period, they rebase each state to the same value with the
    same factor."""
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @settings(max_examples=40, deadline=None)
    @given(
        st.sampled_from(CD_MEASURES),
        st.sampled_from((2022, 2023)),
        st.lists(
            st.integers(min_value=1, max_value=10_000_000), min_size=1, max_size=3
        ),
        st.integers(min_value=0, max_value=5_000_000),
        st.integers(min_value=1, max_value=50_000_000),
    )
    def check(measure_id, period, state_values, other_areas, control_value) -> None:
        us = float(2 * sum(state_values) + other_areas)
        # Each district-file state is two equal districts, so its total is
        # the HT2 state value.
        facts = [
            *_cd_facts(
                measure_id,
                stamp=period,
                us=us,
                districts=tuple((float(value), float(value)) for value in state_values),
            ),
            *_ht2_facts(
                measure_id,
                us=us,
                states=tuple(2.0 * value for value in state_values),
                period=period,
            ),
            _control_fact(measure_id, period=2023, value=float(control_value)),
        ]
        registry = _compile(facts)
        specs = _by_record(registry)
        rows = _cd_rows(registry, measure_id)
        for postal, _, _ in STATES[: len(state_values)]:
            ht2 = specs[_ht2_state_id(measure_id, postal, period)]
            cd = rows[f"{postal}_total"]
            assert cd.value == ht2.value
            assert cd.metadata["uprating_factor"] == ht2.metadata["uprating_factor"]
            assert (
                cd.metadata["uprating_to_period"] == ht2.metadata["uprating_to_period"]
            )

    check()
