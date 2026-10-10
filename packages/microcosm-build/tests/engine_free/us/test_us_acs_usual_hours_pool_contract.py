"""Native ACS usual hours at the pool's operator boundary.

``map_acs_native_inputs`` carries ``WKHP`` into
``weekly_hours_worked_before_lsr``, a column the ASEC hours operator also
writes. The boundary lets it through as a native input only when it is the
mapping of the raw columns beside it. Invariants, each for every input
Hypothesis draws:

1. Round trip. The boundary accepts the frame and receipt the mapper returns.
2. Exact acceptance. After any change to one mapped cell, one raw source cell
   or one receipt field, the boundary accepts if and only if the column and
   receipt still equal a per-person reference mapping of the raw columns.
3. Row accounting. ``observed_rows + missing_rows`` is the person count,
   ``source_value_rows + structural_zero_rows == observed_rows``, the
   allocation counts never exceed ``source_value_rows``, and
   ``source_universe_unavailable_rows <= missing_rows``.
4. Domain. A mapped cell is blank, 0 or a whole number in [1, 99]. It equals
   ``WKHP`` where the source observes it and is 0 only where ``WKL`` confirms
   no work in the past 12 months.
5. Differential. The vectorised mapper agrees with the per-person reference.
6. Determinism. Mapping twice gives the same cells and receipt, and never
   changes a raw column.
7. Native cells survive the pool. The ASEC hours operator writes ASEC rows
   only, and the ASEC-to-ACS gap fill writes blank cells only: every native
   ACS cell, zero included, is the same before and after, no blank remains,
   and the receipt counts exactly the blank cells as authorized and imputed.
   A transfer that rewrites an observed recipient cell is refused.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import multispine_pool as multispine_pool_module
from microcosm.build.us_runtime import stacked_spine as stacked_spine_module
from microcosm.build.us_runtime.acs_inputs import (
    _ACS_USUAL_HOURS_EVIDENCE_KEYS,
    _acs_usual_hours_native_mapping,
    map_acs_native_inputs,
)
from microcosm.build.us_runtime.hours_worked import (
    US_HOURS_WORKED_POOL_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.multispine_pool import (
    _POOL_NATIVE_COMPLETE_OUTPUTS,
    POOL_NATIVE_PARTIAL_TRANSFER_TARGETS,
    pool_transfer_target_families,
)
from microcosm.build.us_runtime.operator_boundary import (
    _ACS_NATIVE_INPUT_CONTRACTS,
    _ACS_RECOMPUTED_NATIVE_INPUTS,
    _NATIVE_INPUT_RECEIPT_KEYS,
    PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
    assert_operator_free_source_frame,
)
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.build.us_runtime.stacked_spine import (
    GapFillDirection,
    assemble_stacked_spine,
)
from microcosm.build.us_runtime.support_provenance import support_channel_column
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_multispine_pool import (
    _real_pre_clone_source_frame,
)
from test_support.microcosm_build.us_multispine_pool import (
    _source_frame as _pool_source_frame,
)
from test_support.microcosm_build.us_stacked_spine import (
    _acs_gap_source,
    _asec_gap_source,
    _gap_fill_with_test_authority,
)

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

_HOURS = "weekly_hours_worked_before_lsr"
_RAW_COLUMNS = ("WKHP", "AGEP", "WKL", "FWKHP")
_LABEL = "ACS native-mapped stacked input"
_SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _acs_person_frame(columns: Mapping[str, list[object]]) -> Frame:
    """One-person households carrying only the raw columns under test."""

    count = len(next(iter(columns.values())))
    ids = np.arange(1, count + 1, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            **{f"person_{entity}_id": ids for entity in US_SCHEMA.group_entities},
            **{name: list(values) for name, values in columns.items()},
        }
    )
    tables = {
        "person": person,
        **{
            entity: pd.DataFrame({f"{entity}_id": ids})
            for entity in US_SCHEMA.group_entities
        },
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.full(count, 10.0), WeightKind.DESIGN)},
        pd.Series(["acs_2024_1yr"] * count, name="stratum"),
    )


def _with_person(frame: Frame, person: pd.DataFrame) -> Frame:
    return Frame(
        {
            entity: person if entity == "person" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
    )


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _reference(person: pd.DataFrame) -> tuple[list[float], dict[str, object]] | None:
    """Usual hours and their receipt, one person at a time.

    Written from the 2024 ACS PUMS data dictionary rules rather than from the
    mapper: ``WKHP`` is 1-99 or blank; a blank is zero hours only where ``WKL``
    is 2 or 3 (last worked over a year ago, or never); under 16 is outside the
    survey universe. Returns ``None`` where the mapper must refuse the table.
    """

    if "WKHP" not in person:
        return None
    values: list[float] = []
    counts = dict.fromkeys(
        (
            "source_value_rows",
            "structural_zero_rows",
            "source_universe_unavailable_rows",
            "allocated_value_rows",
            "allocation_unknown_value_rows",
        ),
        0,
    )
    for row in person.to_dict("records"):
        hours = row["WKHP"]
        age = row.get("AGEP", math.nan)
        last_worked = row.get("WKL", math.nan)
        allocation = row.get("FWKHP", math.nan)
        if not _blank(hours) and not (float(hours).is_integer() and 1 <= hours <= 99):
            return None
        if not _blank(last_worked) and last_worked not in (1, 2, 3):
            return None
        if not _blank(allocation) and allocation not in (0, 1):
            return None
        under_sixteen = not _blank(age) and 0 <= age < 16
        no_recent_work = not _blank(last_worked) and last_worked in (2, 3)
        if not _blank(hours) and (under_sixteen or no_recent_work):
            return None
        if under_sixteen and not _blank(last_worked):
            return None
        counts["source_universe_unavailable_rows"] += under_sixteen
        if not _blank(hours):
            values.append(float(hours))
            counts["source_value_rows"] += 1
            counts["allocated_value_rows"] += allocation == 1
            counts["allocation_unknown_value_rows"] += _blank(allocation)
        elif no_recent_work:
            values.append(0.0)
            counts["structural_zero_rows"] += 1
        else:
            values.append(math.nan)
    missing = sum(math.isnan(value) for value in values)
    receipt: dict[str, object] = {
        "entity": "person",
        "source_columns": [column for column in _RAW_COLUMNS if column in person],
        "transformation": "WKHP; zero only for source-confirmed past-year nonwork",
        "provenance": "acs_2024_1yr_native",
        "observed_rows": len(values) - missing,
        "missing_rows": missing,
        **{name: int(count) for name, count in counts.items()},
        "reference": (
            "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/"
            "data_dict/PUMS_Data_Dictionary_2024.pdf#page=45"
        ),
        "allocation_reference_page": 130,
    }
    return values, receipt


def _same_cells(left: list[float], right: list[float]) -> bool:
    return len(left) == len(right) and all(
        (math.isnan(a) and math.isnan(b))
        or (a == b and math.copysign(1.0, a) == math.copysign(1.0, b))
        for a, b in zip(left, right, strict=True)
    )


def _accepted(frame: Frame, receipt: Mapping[str, Mapping[str, object]]) -> bool:
    try:
        assert_operator_free_source_frame(frame, label=_LABEL, native_inputs=receipt)
    except ValueError:
        return False
    return True


_ALLOCATION = st.sampled_from([0, 1, math.nan])


@st.composite
def _valid_people(draw) -> dict[str, list[object]]:
    """Raw ACS work columns in a valid past-year universe state."""

    size = draw(st.integers(min_value=1, max_value=12))
    people: list[tuple[object, object, object, object]] = []
    for _ in range(size):
        state = draw(
            st.sampled_from(
                ["under_16", "worked", "worked_blank", "no_recent_work", "unknown"]
            )
        )
        allocation = draw(_ALLOCATION)
        if state == "under_16":
            people.append((math.nan, draw(st.integers(0, 15)), math.nan, allocation))
            continue
        age: object = draw(st.one_of(st.integers(16, 95), st.just(math.nan)))
        if state == "worked":
            last_worked = draw(st.sampled_from([1, math.nan]))
            people.append((draw(st.integers(1, 99)), age, last_worked, allocation))
        elif state == "worked_blank":
            people.append((math.nan, age, 1, allocation))
        elif state == "no_recent_work":
            people.append((math.nan, age, draw(st.sampled_from([2, 3])), allocation))
        else:
            people.append((math.nan, age, math.nan, allocation))
    present = {
        "WKHP": True,
        "AGEP": draw(st.booleans()),
        "WKL": draw(st.booleans()),
        "FWKHP": draw(st.booleans()),
    }
    return {
        column: [person[position] for person in people]
        for position, column in enumerate(_RAW_COLUMNS)
        if present[column]
    }


_CELL = st.one_of(
    st.just(math.nan),
    st.just(0.0),
    st.just(-0.0),
    st.integers(1, 99).map(float),
    st.sampled_from([0.5, 100.0, -1.0, 40.25]),
)
_RAW_CELL = {
    "WKHP": st.one_of(st.just(math.nan), st.integers(1, 99)),
    "AGEP": st.one_of(st.just(math.nan), st.integers(0, 95)),
    "WKL": st.sampled_from([1, 2, 3, math.nan]),
    "FWKHP": _ALLOCATION,
}


@_SETTINGS
@given(columns=_valid_people())
def test_boundary_accepts_what_the_mapper_returns(columns) -> None:
    raw = _acs_person_frame(columns)
    before = raw.table("person").copy(deep=True)
    mapped = map_acs_native_inputs(raw)
    again = map_acs_native_inputs(raw)

    assert_operator_free_source_frame(
        mapped.frame, label=_LABEL, native_inputs=mapped.native_inputs
    )

    person = mapped.frame.table("person")
    receipt = mapped.native_inputs[_HOURS]
    reference = _reference(before)
    assert reference is not None
    expected_values, expected_receipt = reference
    assert _same_cells(person[_HOURS].tolist(), expected_values)
    assert dict(receipt) == expected_receipt
    assert frozenset(receipt) == (
        _NATIVE_INPUT_RECEIPT_KEYS | _ACS_USUAL_HOURS_EVIDENCE_KEYS
    )

    assert receipt["observed_rows"] + receipt["missing_rows"] == len(person)
    assert (
        receipt["source_value_rows"] + receipt["structural_zero_rows"]
        == receipt["observed_rows"]
    )
    assert (
        receipt["allocated_value_rows"] + receipt["allocation_unknown_value_rows"]
        <= receipt["source_value_rows"]
    )
    assert receipt["source_universe_unavailable_rows"] <= receipt["missing_rows"]

    observed = person[_HOURS].dropna()
    assert ((observed == 0) | ((observed >= 1) & (observed <= 99))).all()
    assert (observed == np.floor(observed)).all()
    source_hours = pd.to_numeric(before["WKHP"], errors="coerce")
    assert (person.loc[source_hours.notna(), _HOURS] == source_hours.dropna()).all()

    assert _same_cells(
        person[_HOURS].tolist(), again.frame.table("person")[_HOURS].tolist()
    )
    assert dict(again.native_inputs[_HOURS]) == dict(receipt)
    pd.testing.assert_frame_equal(person[list(before.columns)], before)
    pd.testing.assert_frame_equal(raw.table("person"), before)


@_SETTINGS
@given(columns=_valid_people(), data=st.data())
def test_boundary_refuses_any_changed_mapped_cell(columns, data) -> None:
    mapped = map_acs_native_inputs(_acs_person_frame(columns))
    person = mapped.frame.table("person").copy()
    row = data.draw(st.integers(0, len(person) - 1))
    replacement = data.draw(_CELL)
    original = float(person[_HOURS].iloc[row])
    person[_HOURS] = person[_HOURS].to_numpy(copy=True)
    person.loc[person.index[row], _HOURS] = replacement

    unchanged = _same_cells([original], [replacement])
    assert _accepted(_with_person(mapped.frame, person), mapped.native_inputs) == (
        unchanged
    )


@_SETTINGS
@given(columns=_valid_people(), data=st.data())
def test_boundary_tracks_any_changed_raw_cell(columns, data) -> None:
    mapped = map_acs_native_inputs(_acs_person_frame(columns))
    person = mapped.frame.table("person").copy()
    column = data.draw(
        st.sampled_from([name for name in _RAW_COLUMNS if name in person])
    )
    row = data.draw(st.integers(0, len(person) - 1))
    values = person[column].astype(object).tolist()
    values[row] = data.draw(_RAW_CELL[column])
    person[column] = values

    reference = _reference(person[[name for name in _RAW_COLUMNS if name in person]])
    still_native = (
        reference is not None
        and _same_cells(person[_HOURS].tolist(), reference[0])
        and dict(mapped.native_inputs[_HOURS]) == reference[1]
    )
    assert _accepted(_with_person(mapped.frame, person), mapped.native_inputs) == (
        still_native
    )


@_SETTINGS
@given(columns=_valid_people(), data=st.data())
def test_boundary_refuses_any_changed_receipt_field(columns, data) -> None:
    mapped = map_acs_native_inputs(_acs_person_frame(columns))
    receipt = dict(mapped.native_inputs[_HOURS])
    field = data.draw(st.sampled_from(sorted(receipt)))
    action = data.draw(st.sampled_from(["change", "drop", "add"]))
    if action == "drop":
        del receipt[field]
    elif action == "add":
        receipt["unreviewed_rows"] = 0
    elif isinstance(receipt[field], int):
        receipt[field] = receipt[field] + data.draw(st.sampled_from([-1, 1]))
    elif isinstance(receipt[field], list):
        receipt[field] = [*receipt[field], "ESR"]
    else:
        receipt[field] = f"{receipt[field]} (edited)"

    forged = {**mapped.native_inputs, _HOURS: receipt}
    assert not _accepted(mapped.frame, forged)


@_SETTINGS
@given(columns=_valid_people())
def test_recomputation_never_reads_the_mapped_column(columns) -> None:
    mapped = map_acs_native_inputs(_acs_person_frame(columns))
    person = mapped.frame.table("person")
    recomputed = _acs_usual_hours_native_mapping(person)
    poisoned = person.copy()
    poisoned[_HOURS] = 40.0
    from_poisoned = _acs_usual_hours_native_mapping(poisoned)

    assert recomputed is not None and from_poisoned is not None
    assert _same_cells(recomputed[0].tolist(), person[_HOURS].tolist())
    assert _same_cells(recomputed[0].tolist(), from_poisoned[0].tolist())
    assert dict(recomputed[1]) == dict(from_poisoned[1])
    assert dict(recomputed[1]) == dict(mapped.native_inputs[_HOURS])


def test_boundary_refuses_operator_hours_under_a_native_receipt() -> None:
    """ASEC-derived or donor-filled hours cannot be relabelled as ACS native."""

    mapped = map_acs_native_inputs(
        _acs_person_frame(
            {
                "WKHP": [40, math.nan, math.nan],
                "AGEP": [40, 12, 50],
                "WKL": [1, math.nan, 3],
                "FWKHP": [0, 0, 0],
            }
        )
    )
    person = mapped.frame.table("person").copy()
    assert _same_cells(person[_HOURS].tolist(), [40.0, math.nan, 0.0])

    filled = person.copy()
    filled[_HOURS] = [40.0, 0.0, 0.0]
    filled_receipt = {
        **mapped.native_inputs,
        _HOURS: {**mapped.native_inputs[_HOURS], "observed_rows": 3, "missing_rows": 0},
    }
    with pytest.raises(ValueError, match="recomputed from the raw ACS columns"):
        assert_operator_free_source_frame(
            _with_person(mapped.frame, filled),
            label=_LABEL,
            native_inputs=filled_receipt,
        )

    swapped = person.copy()
    swapped[_HOURS] = [35.0, math.nan, 0.0]
    with pytest.raises(ValueError, match="not the native mapping of the raw ACS"):
        assert_operator_free_source_frame(
            _with_person(mapped.frame, swapped),
            label=_LABEL,
            native_inputs=mapped.native_inputs,
        )

    without_raw = person.drop(columns=["WKHP"])
    with pytest.raises(ValueError, match="source_columns"):
        assert_operator_free_source_frame(
            _with_person(mapped.frame, without_raw),
            label=_LABEL,
            native_inputs=mapped.native_inputs,
        )

    with pytest.raises(ValueError, match="hours_worked:person"):
        assert_operator_free_source_frame(mapped.frame, label=_LABEL)


def test_boundary_refuses_an_integer_typed_hours_column() -> None:
    mapped = map_acs_native_inputs(
        _acs_person_frame({"WKHP": [40, 20], "AGEP": [40, 30], "WKL": [1, 1]})
    )
    person = mapped.frame.table("person").copy()
    person[_HOURS] = person[_HOURS].astype(np.int64)

    with pytest.raises(ValueError, match="not the native mapping of the raw ACS"):
        assert_operator_free_source_frame(
            _with_person(mapped.frame, person),
            label=_LABEL,
            native_inputs=mapped.native_inputs,
        )


def test_boundary_names_a_raw_contradiction_instead_of_passing_it() -> None:
    mapped = map_acs_native_inputs(
        _acs_person_frame({"WKHP": [40, math.nan], "AGEP": [40, 30], "WKL": [1, 3]})
    )
    person = mapped.frame.table("person").copy()
    person["WKL"] = [3, 3]

    with pytest.raises(ValueError, match="contradict their age or past-year universe"):
        assert_operator_free_source_frame(
            _with_person(mapped.frame, person),
            label=_LABEL,
            native_inputs=mapped.native_inputs,
        )


def test_whitespace_blanks_map_and_pass_like_nulls() -> None:
    mapped = map_acs_native_inputs(
        _acs_person_frame(
            {"WKHP": [" ", 40, ""], "AGEP": [50, 40, 9], "WKL": [2, 1, " "]}
        )
    )

    assert _same_cells(
        mapped.frame.table("person")[_HOURS].tolist(), [0.0, 40.0, math.nan]
    )
    assert_operator_free_source_frame(
        mapped.frame, label=_LABEL, native_inputs=mapped.native_inputs
    )


def test_usual_hours_is_the_only_partly_native_pool_transfer_target() -> None:
    """Every native mapping of an operator output has exactly one pool role.

    A column ACS maps completely is never transferred; a column it maps only
    inside its survey universe stays a transfer target and is recomputed at
    the boundary. Nothing else may be both native and operator-owned.
    """

    operator_outputs = {
        (entity, column)
        for by_entity in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES.values()
        for entity, columns in by_entity.items()
        for column in columns
    }
    plan = pool_transfer_target_families()
    planned = {
        (entity, column)
        for entity, families in plan.items()
        for columns in families.values()
        for column in columns
    }
    complete = {
        (entity, column)
        for entity, columns in _POOL_NATIVE_COMPLETE_OUTPUTS.items()
        for column in columns
    }
    partial = {
        (entity, column)
        for entity, columns in POOL_NATIVE_PARTIAL_TRANSFER_TARGETS.items()
        for column in columns
    }
    native_operator_outputs = {
        (entity, output)
        for output, (entity, _sources, _transformation) in (
            _ACS_NATIVE_INPUT_CONTRACTS.items()
        )
        if (entity, output) in operator_outputs
    }

    assert partial == {("person", _HOURS)}
    assert _HOURS == US_HOURS_WORKED_POOL_OUTPUT_COLUMNS[0]
    assert not partial & complete
    assert partial <= planned
    assert not complete & planned
    assert native_operator_outputs <= complete | partial
    assert native_operator_outputs & planned == partial
    assert {column for _entity, column in partial} == set(_ACS_RECOMPUTED_NATIVE_INPUTS)
    assert plan["person"]["source_operator_hours_worked"] == (_HOURS,)


_NATIVE_CELL = st.one_of(
    st.just(math.nan),
    st.just(0.0),
    st.integers(1, 99).map(float),
)
_ASEC_GAP_HOURS = [40.0, 0.0, 20.0, 45.0, 0.0, 35.0]
_HOURS_GAP_FILL_PLAN = (
    GapFillDirection(
        name="asec_survey_to_acs",
        recipient_channel="acs",
        donor_channel="asec",
        target_families={"person": {"source_operator_hours_worked": (_HOURS,)}},
    ),
)
_HOURS_GAP_FILL_TARGET = f"person/source_operator_hours_worked/{_HOURS}"


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(native=st.lists(_NATIVE_CELL, min_size=2, max_size=2))
def test_asec_hours_operator_writes_asec_rows_only(native) -> None:
    acs = _pool_source_frame(offset=100.0)
    acs.table("person")[_HOURS] = np.asarray(native, dtype=np.float64)
    assembled = assemble_spines(
        {"asec": _real_pre_clone_source_frame(), "acs": acs},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    before = assembled.table("person").copy(deep=True)

    prepared = multispine_pool_module._run_source_operator_chain(
        assembled,
        phase="pre_clone",
        operator_names=("with_us_hours_worked_inputs",),
        operators={
            "with_us_hours_worked_inputs": (
                multispine_pool_module._with_gated_us_hours_worked_inputs
            )
        },
    )

    person = prepared.frame.table("person")
    assert person["person_id"].tolist() == before["person_id"].tolist()
    asec_rows = person["PERIDNUM"].notna().to_numpy()
    assert _same_cells(before.loc[~asec_rows, _HOURS].tolist(), native)
    assert before.loc[asec_rows, _HOURS].isna().all()
    assert person.loc[asec_rows, _HOURS].tolist() == [40.0, 0.0, 35.0, 0.0]
    assert _same_cells(person.loc[~asec_rows, _HOURS].tolist(), native)
    assert person.loc[~asec_rows, "hours_worked_last_week"].isna().all()
    assert prepared.receipt["suboperators"][0]["merged_rows"] == {"person": 4}


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(native=st.lists(_NATIVE_CELL, min_size=11, max_size=11))
def test_gap_fill_fills_blank_usual_hours_and_keeps_native_cells(native) -> None:
    asec = _asec_gap_source()
    asec.table("person")[_HOURS] = np.asarray(_ASEC_GAP_HOURS, dtype=np.float64)
    acs = _acs_gap_source()
    acs.table("person")[_HOURS] = np.asarray(native, dtype=np.float64)
    stacked = assemble_stacked_spine(
        asec,
        acs,
        acs_sample_fraction=1.0,
        acs_sample_seed=578,
    ).frame
    before = stacked.table("person").copy(deep=True)
    blank = before[_HOURS].isna().to_numpy()
    acs_rows = before[support_channel_column("person")].eq("acs").to_numpy()
    assert int(acs_rows.sum()) == len(native)
    assert not (blank & ~acs_rows).any()
    assert int(blank.sum()) == sum(math.isnan(cell) for cell in native)

    result = _gap_fill_with_test_authority(
        stacked,
        plan=_HOURS_GAP_FILL_PLAN,
        seed=578,
        n_estimators=10,
    )

    person = result.frame.table("person")
    assert person["person_id"].tolist() == before["person_id"].tolist()
    assert _same_cells(
        person.loc[~blank, _HOURS].tolist(), before.loc[~blank, _HOURS].tolist()
    )
    filled = person.loc[blank, _HOURS]
    assert filled.notna().all()
    assert filled.between(min(_ASEC_GAP_HOURS), max(_ASEC_GAP_HOURS)).all()
    direction = result.receipt["directions"]["asec_survey_to_acs"]
    assert direction["targets"][_HOURS_GAP_FILL_TARGET] == {
        "authorized_null_rows": int(blank.sum()),
        "imputed_rows": int(blank.sum()),
        "unmodeled_rows": 0,
        "residual_null_rows": 0,
    }


def _stacked_hours_fixture(native: list[float]) -> Frame:
    asec = _asec_gap_source()
    asec.table("person")[_HOURS] = np.asarray(_ASEC_GAP_HOURS, dtype=np.float64)
    acs = _acs_gap_source()
    acs.table("person")[_HOURS] = np.asarray(native, dtype=np.float64)
    return assemble_stacked_spine(
        asec,
        acs,
        acs_sample_fraction=1.0,
        acs_sample_seed=578,
    ).frame


@pytest.mark.parametrize("rewrite", [0.0, 41.0, math.nan])
def test_gap_fill_refuses_a_transfer_that_rewrites_an_observed_cell(
    monkeypatch: pytest.MonkeyPatch,
    rewrite: float,
) -> None:
    native = [
        40.0,
        math.nan,
        0.0,
        35.0,
        math.nan,
        20.0,
        0.0,
        50.0,
        math.nan,
        45.0,
        30.0,
    ]
    stacked = _stacked_hours_fixture(native)
    person = stacked.table("person")
    observed_row = person.index[
        person[support_channel_column("person")].eq("acs") & person[_HOURS].eq(40.0)
    ][0]
    transfer = stacked_spine_module.transfer_acs_inputs

    def rewrite_observed_cell(*args: object, **kwargs: object) -> object:
        result = transfer(*args, **kwargs)
        rewritten = result.frame.table("person").copy()
        rewritten.loc[observed_row, _HOURS] = rewrite
        tables = {
            entity: rewritten if entity == "person" else result.frame.table(entity)
            for entity in result.frame.entities
        }
        frame = Frame(
            tables,
            result.frame.schema,
            {
                entity: result.frame.weights_for(entity)
                for entity in result.frame.weighted_entities
            },
            result.frame.strata,
            mass_log=result.frame.mass_log,
            metadata=result.frame.metadata,
        )
        return replace(result, frame=frame)

    monkeypatch.setattr(
        stacked_spine_module, "transfer_acs_inputs", rewrite_observed_cell
    )
    with pytest.raises(ValueError, match="observed recipient byte identity failed"):
        _gap_fill_with_test_authority(
            stacked,
            plan=_HOURS_GAP_FILL_PLAN,
            seed=578,
            n_estimators=10,
        )
