"""Congressional-district interest paid binds as a share of a national control.

The district file publishes Schedule A line 8a (N19300/A19300), not the line
10 total its interest_paid_deduction_* measure ids name. Its rows bind
deductible_mortgage_interest as shares of the file's US row, scaled to Table
2.1's home mortgage interest control (microcosm#1038,
docs/us-soi-interest-paid-concepts.md).

Invariants, for every fact set the compile accepts:

- No district-file row binds interest_deduction.
- One factor (control / the file's US row) scales every row of one record set,
  column and filing status, so district rows keep their within-state shares
  and the states keep their shares of the US row.
- Where the file's states sum to its US row, its states sum to the control,
  and so do its districts.
- The file's US row never binds; Table 2.1's rows own the national concepts.
- A row with no control at or after its period, or no nonzero US row, never
  binds (no component level ships).
- The output does not depend on feed order or on a common scale of the
  district file's values (the package's period stamp cancels likewise).
"""

# ruff: noqa: F403, F405
import dataclasses
import random

from test_support.microcosm_build.us_fiscal_targets import *

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from microcosm.build.us_runtime.fiscal_targets import (  # noqa: E402
    AmbiguousSoiDistrictComponentControlError,
    AmbiguousSoiDistrictComponentShareTotalError,
    UnreviewedSoiDistrictColumnError,
)

_CD_RECORD_SET = "irs_soi.ty2023.congressional_district_2022.all_returns"
_T21_RECORD_SET = "irs_soi.ty2023.table_2_1.itemized_all_returns"
#: States with two districts on both plans, so the hierarchy rule's
#: completeness check passes without a crosswalk.
_TWO_DISTRICT_STATES = {
    "15": "hi",
    "16": "id",
    "23": "me",
    "33": "nh",
    "44": "ri",
    "54": "wv",
}
_COLUMNS = {"amount": "A19300", "returns": "N19300"}
_CONTROLS = {
    "amount": "home_mortgage_interest_amount",
    "returns": "home_mortgage_interest_returns",
}


def _cd_fact(
    measure_id: str,
    column: str | None,
    value: float,
    *,
    row: str = "us",
    level: str = "country",
    geography_id: str = "0100000US",
    record_set: str = _CD_RECORD_SET,
    period: int = 2023,
) -> dict[str, object]:
    fact = _dynamic_ledger_fact(
        source_record_id=f"{record_set}.{row}.{measure_id}",
        source_name="irs_soi",
        measure_id=measure_id,
        value=value,
        period_value=period,
        geography_level=level,
        geography_id=geography_id,
        dimensions={"income_range": "all", "filing_status": "all"},
        layout_record_set_id=record_set,
        groupby_dimension="irs_soi.congressional_district",
        groupby_value_id=row,
    )
    if column is not None:
        fact["layout"]["source_column_id"] = column
    return fact


def _table_2_1_fact(
    measure_id: str,
    value: float,
    *,
    period: int = 2023,
    record_set: str = _T21_RECORD_SET,
) -> dict[str, object]:
    return _dynamic_ledger_fact(
        source_record_id=f"{record_set}.all.{measure_id}",
        source_name="irs_soi",
        measure_id=measure_id,
        value=value,
        period_value=period,
        dimensions={"income_range": "all", "filing_status": "all"},
        layout_record_set_id=record_set,
    )


def _district_file(
    districts: dict[str, tuple[float, float]],
    *,
    kind: str = "amount",
    measure_prefix: str = "interest_paid_deduction",
    us_total: float | None = None,
    scale: float = 1.0,
    record_set: str = _CD_RECORD_SET,
    period: int = 2023,
) -> list[dict[str, object]]:
    """The file's US, state and district rows for ``{state_fips: (d1, d2)}``."""

    measure_id = f"{measure_prefix}_{kind}"
    column = _COLUMNS[kind]
    facts = []
    states_total = 0.0
    for fips, (first, second) in sorted(districts.items()):
        postal = _TWO_DISTRICT_STATES[fips]
        states_total += first + second
        facts.append(
            _cd_fact(
                measure_id,
                column,
                (first + second) * scale,
                row=f"{postal}_total",
                level="state",
                geography_id=f"0400000US{fips}",
                record_set=record_set,
                period=period,
            )
        )
        for index, value in enumerate((first, second), start=1):
            facts.append(
                _cd_fact(
                    measure_id,
                    column,
                    value * scale,
                    row=f"{postal}_{index:02d}",
                    level="congressional_district",
                    geography_id=f"5001700US{fips}{index:02d}",
                    record_set=record_set,
                    period=period,
                )
            )
    facts.append(
        _cd_fact(
            measure_id,
            column,
            (states_total if us_total is None else us_total) * scale,
            record_set=record_set,
            period=period,
        )
    )
    return facts


def _national_rows(kind: str = "amount", *, control: float = 171e9) -> list:
    """Table 2.1's line 10 total and its home mortgage interest control."""

    line_10 = 208e9 if kind == "amount" else 11.9e6
    return [
        _table_2_1_fact(f"interest_paid_deduction_{kind}", line_10),
        _table_2_1_fact(_CONTROLS[kind], control),
    ]


def _compile(facts) -> TargetRegistry:
    return compile_us_fiscal_target_registry(
        [*packaged_reference_facts(), *facts], allow_unaged_dollar_targets=True
    )


def _district_file_specs(registry) -> list:
    return [
        spec
        for spec in registry.specs
        if ".congressional_district_"
        in spec.metadata.get("ledger_layout_record_set_id", "")
    ]


_WORLD = {"15": (30e9, 50e9), "23": (8e9, 12e9)}


@pytest.mark.parametrize("kind", ["amount", "returns"])
def test_congressional_district_mortgage_interest_binds_as_a_national_share(
    kind: str,
) -> None:
    control = 171e9 if kind == "amount" else 11.6e6
    us_total = 100e9  # the two states' rows sum to it
    registry = _compile(
        [*_district_file(_WORLD, kind=kind), *_national_rows(kind, control=control)]
    )
    rows = _district_file_specs(registry)
    factor = control / us_total
    # Two states and four districts bind; the file's US row retires.
    assert len(rows) == 6
    assert not [spec for spec in rows if "state_fips" not in spec.metadata]
    for spec in rows:
        metadata = spec.metadata
        assert metadata["variable"] == "deductible_mortgage_interest"
        assert metadata["source_variable"] == "deductible_mortgage_interest"
        assert metadata["base_variable"] == "deductible_mortgage_interest_tax_unit"
        assert metadata["measure_mode"] == (
            "indicator_sum" if kind == "returns" else "sum"
        )
        assert metadata["itemized_only"] == "true"
        assert metadata["target_role"] == "soi_fiscal_distribution"
        assert metadata["soi_source_column_id"] == _COLUMNS[kind]
        assert metadata["soi_source_concept"] == (
            "schedule_a_line_8a_home_mortgage_interest_paid_to_financial_institutions"
        )
        assert metadata["soi_control_concept"] == (
            "schedule_a_lines_8a_8b_total_home_mortgage_interest"
        )
        assert metadata["soi_share_total_source_record_id"] == (
            f"{_CD_RECORD_SET}.us.interest_paid_deduction_{kind}"
        )
        assert metadata["uprating_index_source_record_id"] == (
            f"{_T21_RECORD_SET}.all.{_CONTROLS[kind]}"
        )
        assert metadata["uprating_to_period"] == "2023"
        assert metadata["soi_return_universe"] == "all_returns"
        assert metadata["soi_return_universe_bridge"] == "itemized_returns"
        assert float(metadata["uprating_factor"]) == pytest.approx(factor, rel=1e-15)
        assert "requires_soi_district_component_share" not in metadata
    by_row = {spec.metadata["ledger_layout_groupby_value_id"]: spec for spec in rows}
    assert by_row["hi_01"].value == pytest.approx(30e9 * factor, rel=1e-12)
    assert by_row["me_total"].value == pytest.approx(20e9 * factor, rel=1e-12)
    states = [
        spec for spec in rows if "congressional_district_geoid" not in spec.metadata
    ]
    assert math.fsum(spec.value for spec in states) == pytest.approx(control, rel=1e-12)
    districts = [
        spec for spec in rows if "congressional_district_geoid" in spec.metadata
    ]
    assert math.fsum(spec.value for spec in districts) == pytest.approx(
        control, rel=1e-12
    )
    # Table 2.1's own rows are untouched: line 10 stays on interest_deduction
    # as its national total, and the control binds the same concept the
    # district shares do.
    national = {
        spec.metadata["source_measure_id"]: spec
        for spec in registry.specs
        if spec.metadata.get("ledger_layout_record_set_id") == _T21_RECORD_SET
    }
    line_10 = national[f"interest_paid_deduction_{kind}"]
    assert line_10.metadata["variable"] == "interest_deduction"
    assert line_10.value == (208e9 if kind == "amount" else 11.9e6)
    assert national[_CONTROLS[kind]].metadata["variable"] == (
        "deductible_mortgage_interest"
    )
    assert national[_CONTROLS[kind]].value == control


def test_a_relabeled_district_measure_binds_the_same_shares() -> None:
    """The IRS column is the key, not the measure id a package uses for it.

    A Chronicle package that relabels the line 8a column to Table 2.1's own
    mortgage_interest_paid_* ids compiles to the same targets.
    """

    def values(prefix: str) -> dict[str, tuple[str, float]]:
        registry = _compile(
            [*_district_file(_WORLD, measure_prefix=prefix), *_national_rows()]
        )
        return {
            spec.metadata["ledger_layout_groupby_value_id"]: (
                spec.metadata["variable"],
                spec.value,
            )
            for spec in _district_file_specs(registry)
        }

    assert values("mortgage_interest_paid") == values("interest_paid_deduction")


@pytest.mark.parametrize("column", [None, "", "A19570", "A19500", "A04470"])
def test_district_interest_paid_without_a_reviewed_column_is_refused(
    column: str | None,
) -> None:
    """The file has no line 10 column, so only line 8a may compile, as shares."""

    facts = [
        *_district_file(_WORLD),
        _cd_fact(
            "interest_paid_deduction_amount",
            column,
            1e9,
            row="hi_extra",
            level="state",
            geography_id="0400000US15",
            record_set="irs_soi.ty2023.congressional_district_2024.all_returns",
        ),
        *_national_rows(),
    ]
    with pytest.raises(UnreviewedSoiDistrictColumnError, match="no column for"):
        _compile(facts)


def test_table_2_1_interest_paid_needs_no_column() -> None:
    """The guard is the district file's: Table 2.1 publishes line 10 itself."""

    registry = _compile(_national_rows())
    (line_10,) = [
        spec
        for spec in registry.specs
        if spec.metadata.get("source_measure_id") == "interest_paid_deduction_amount"
    ]
    assert line_10.metadata["target_role"] == "interest_deduction_total"
    assert line_10.metadata["variable"] == "interest_deduction"


def test_district_mortgage_interest_without_an_anchor_never_binds() -> None:
    """No control, a control older than the rows, or no US row: dropped."""

    no_control = _compile(_district_file(_WORLD))
    assert not _district_file_specs(no_control)

    old_control = _compile(
        [
            *_district_file(_WORLD),
            _table_2_1_fact(
                "home_mortgage_interest_amount",
                160e9,
                period=2022,
                record_set="irs_soi.ty2022.table_2_1.itemized_all_returns",
            ),
        ]
    )
    assert not _district_file_specs(old_control)

    no_us_row = [
        fact
        for fact in _district_file(_WORLD)
        if fact["geography"]["level"] != "country"
    ]
    assert not _district_file_specs(_compile([*no_us_row, *_national_rows()]))

    zero_us_row = _district_file(_WORLD, us_total=0.0)
    assert not _district_file_specs(_compile([*zero_us_row, *_national_rows()]))


def test_district_mortgage_interest_control_must_be_unambiguous() -> None:
    """Feed order never picks a control or a denominator (microcosm#1035)."""

    rival = _table_2_1_fact(
        "home_mortgage_interest_amount",
        170e9,
        record_set="irs_soi.ty2023.table_2_1_revised.itemized_all_returns",
    )
    with pytest.raises(AmbiguousSoiDistrictComponentControlError):
        _compile([*_district_file(_WORLD), *_national_rows(), rival])

    duplicate_us_row = _cd_fact(
        "interest_paid_deduction_amount", "A19300", 99e9, row="us_duplicate"
    )
    with pytest.raises(AmbiguousSoiDistrictComponentShareTotalError):
        _compile([*_district_file(_WORLD), duplicate_us_row, *_national_rows()])

    # The newest control wins outright; an older one is no rival.
    older = _table_2_1_fact(
        "home_mortgage_interest_amount",
        150e9,
        period=2022,
        record_set="irs_soi.ty2022.table_2_1.itemized_all_returns",
    )
    registry = _compile([*_district_file(_WORLD), *_national_rows(), older])
    assert {
        spec.metadata["uprating_index_source_record_id"]
        for spec in _district_file_specs(registry)
    } == {f"{_T21_RECORD_SET}.all.home_mortgage_interest_amount"}


def test_district_mortgage_interest_ignores_the_package_stamp() -> None:
    """The file is TY2022 data stamped 2023 (microcosm#1030); either stamp
    gives the same targets, landed at the control's period."""

    def values(period: int) -> dict[str, tuple[float, str]]:
        record_set = f"irs_soi.ty{period}.congressional_district_2022.all_returns"
        registry = _compile(
            [
                *_district_file(_WORLD, record_set=record_set, period=period),
                *_national_rows(),
            ]
        )
        return {
            spec.metadata["ledger_layout_groupby_value_id"]: (
                spec.value,
                spec.metadata["uprating_to_period"],
            )
            for spec in _district_file_specs(registry)
        }

    stamped, truthful = values(2023), values(2022)
    assert stamped.keys() == truthful.keys()
    for row, (value, landed) in stamped.items():
        assert truthful[row][0] == pytest.approx(value, rel=1e-15)
        assert landed == truthful[row][1] == "2023"


def test_district_mortgage_interest_ages_with_its_control() -> None:
    """Target aging gives the shares the control's own remaining link, so
    the states still sum to the aged national row."""

    from microcosm.build.us_runtime.target_aging import age_us_dollar_targets

    registry = _compile([*_district_file(_WORLD), *_national_rows()])
    projections = (
        _cbo_income_source_projection_fact(
            2023, "adjusted_gross_income", value=15_000_000_000_000
        ),
        _cbo_income_source_projection_fact(
            2024, "adjusted_gross_income", value=16_500_000_000_000
        ),
    )
    aged = age_us_dollar_targets(registry, projections, target_period=2024)
    link = 16_500_000_000_000 / 15_000_000_000_000
    (control,) = [
        spec
        for spec in aged.specs
        if spec.metadata.get("source_measure_id") == "home_mortgage_interest_amount"
    ]
    assert control.value == pytest.approx(171e9 * link, rel=1e-12)
    states = [
        spec
        for spec in _district_file_specs(aged)
        if "congressional_district_geoid" not in spec.metadata
    ]
    assert len(states) == 2
    assert {spec.metadata["aging_factor"] for spec in states} == {
        control.metadata["aging_factor"]
    }
    assert math.fsum(spec.value for spec in states) == pytest.approx(
        control.value, rel=1e-12
    )


@st.composite
def _district_worlds(draw):
    fips = draw(
        st.lists(
            st.sampled_from(sorted(_TWO_DISTRICT_STATES)),
            min_size=1,
            max_size=len(_TWO_DISTRICT_STATES),
            unique=True,
        )
    )
    positive = st.floats(1.0, 1e11, allow_nan=False, allow_infinity=False)
    world = {state: (draw(positive), draw(positive)) for state in fips}
    return {
        "world": world,
        "kind": draw(st.sampled_from(["amount", "returns"])),
        "prefix": draw(
            st.sampled_from(["interest_paid_deduction", "mortgage_interest_paid"])
        ),
        # The US row may exceed the bound states (states missing from the feed).
        "us_excess": draw(st.sampled_from([1.0, 1.0, draw(st.floats(1.0, 3.0))])),
        "control": draw(positive),
        "scale": draw(st.floats(1e-3, 1e3)),
        "seed": draw(st.integers(0, 2**31 - 1)),
    }


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_district_worlds())
def test_district_mortgage_interest_share_invariants(case) -> None:
    world, kind = case["world"], case["kind"]
    states_total = math.fsum(first + second for first, second in world.values())
    us_total = states_total * case["us_excess"]
    control = case["control"]

    def compiled(scale: float, shuffle_seed: int | None) -> dict[str, TargetSpec]:
        facts = [
            *_district_file(
                world,
                kind=kind,
                measure_prefix=case["prefix"],
                us_total=us_total,
                scale=scale,
            ),
            *_national_rows(kind, control=control),
        ]
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(facts)
        registry = _compile(facts)
        return {spec.name: spec for spec in _district_file_specs(registry)}

    base = compiled(1.0, None)
    factor = control / us_total
    raw = {}
    for fips, (first, second) in world.items():
        postal = _TWO_DISTRICT_STATES[fips]
        raw[f"{postal}_total"] = first + second
        raw[f"{postal}_01"], raw[f"{postal}_02"] = first, second
    rows = {
        spec.metadata["ledger_layout_groupby_value_id"]: spec for spec in base.values()
    }
    # The US row retires; every state and district binds.
    assert set(rows) == set(raw)
    for row, spec in rows.items():
        # One factor; never negative; never interest_deduction; never flagged.
        assert spec.value == pytest.approx(raw[row] * factor, rel=1e-12)
        assert spec.value >= 0.0
        assert spec.metadata["variable"] == "deductible_mortgage_interest"
        assert "requires_soi_district_component_share" not in spec.metadata
    # Districts sum to their state; states sum to the control's share of
    # the US row that the bound states hold.
    for fips in world:
        postal = _TWO_DISTRICT_STATES[fips]
        assert rows[f"{postal}_01"].value + rows[f"{postal}_02"].value == (
            pytest.approx(rows[f"{postal}_total"].value, rel=1e-12)
        )
    state_sum = math.fsum(rows[f"{_TWO_DISTRICT_STATES[f]}_total"].value for f in world)
    assert state_sum == pytest.approx(control * states_total / us_total, rel=1e-12)
    if case["us_excess"] == 1.0:
        assert state_sum == pytest.approx(control, rel=1e-12)
    # A common scale of the file (its stamp, its units) changes no target
    # value, only the recorded factor; feed order changes nothing at all.
    scaled = compiled(case["scale"], None)
    assert scaled.keys() == base.keys()
    for name, spec in base.items():
        assert scaled[name].value == pytest.approx(spec.value, rel=1e-12)
        assert scaled[name].metadata["variable"] == spec.metadata["variable"]
    shuffled = compiled(1.0, case["seed"])
    assert shuffled == base
