"""ESI premiums in a stacked pool (microcosm #454).

A stacked pool gives each household source a share of one population's
household mass. Only its CPS-source rows carry the raw ASEC fields the stage
reads; a cross-source transfer fills the employer premium on every other row.

Invariants of :func:`with_us_esi_premium_pool_anchor`, for every valid pool:

* **Conservation**: the pool-wide weighted employer premium equals the
  source-derived rows' total divided by their share of household mass, which
  is what the stage assigns the whole population.
* **Equal intensity**: source-derived and transferred rows carry the same
  weighted employer premium per unit of household mass.
* **Source rows are immutable**: no source-derived cell changes by a byte.
* **Structural zeros**: a transferred person whose source record reports no
  wages carries no premium, on every support clone.
* **Proportionality**: every other transferred employer premium is one common
  multiple of the value the transfer drew.
* **Idempotence**: a pool already on the stage's scale is returned unchanged.
* **Weight homogeneity**: rescaling every household weight leaves each value
  where it was.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime import esi_premiums as esi
from microcosm.build.us_runtime.esi_premiums import (
    US_ESI_EMPLOYER_PREMIUM_COLUMN,
    US_ESI_PREMIUMS_OUTPUT_COLUMNS,
    US_ESI_PREMIUMS_WAGE_COLUMN,
    us_esi_premiums_household_mass_share,
    with_us_esi_premium_pool_anchor,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

EMPLOYER = US_ESI_EMPLOYER_PREMIUM_COLUMN
WAGES = US_ESI_PREMIUMS_WAGE_COLUMN
RAW = esi._RAW_EVIDENCE_COLUMNS
_GROUPS = ("tax_unit", "spm_unit", "family", "marital_unit")


def _pool(
    households: list[dict],
    *,
    clone_of: dict[int, int] | None = None,
) -> Frame:
    """A pool frame from household records.

    Each record has ``source`` (whether its people carry raw ASEC fields),
    ``weight`` and ``people``: a list of ``(employer premium, wages)``. Every
    other stage output takes half the employer premium. ``clone_of`` maps a
    household position to the position of the household it is a support clone
    of; clones then share ``person_source_id`` with their source person.
    """

    clone_of = clone_of or {}
    records = []
    for position, household in enumerate(households):
        origin = clone_of.get(position, position)
        for member, (premium, wages) in enumerate(household["people"]):
            record = {
                "person_household_id": position + 1,
                "person_source_id": (origin + 1) * 100 + member,
                "person_support_clone_index": int(position in clone_of),
                WAGES: float(wages),
            }
            record |= {
                column: float(premium) * (1.0 if column == EMPLOYER else 0.5)
                for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS
            }
            record |= {
                column: 1.0 if household["source"] else np.nan for column in RAW
            }
            records.append(record)
    person = pd.DataFrame(records)
    person.insert(0, "person_id", np.arange(1, len(person) + 1, dtype="int64"))
    for offset, group in enumerate(_GROUPS, start=1):
        person[f"person_{group}_id"] = person["person_id"] + offset * 1_000_000
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {"household_id": np.arange(1, len(households) + 1, dtype="int64")}
        ),
    }
    for group in _GROUPS:
        tables[group] = pd.DataFrame(
            {f"{group}_id": person[f"person_{group}_id"].to_numpy()}
        )
    weights = np.asarray([household["weight"] for household in households], float)
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(values=weights, kind=WeightKind.IMPORTANCE)},
    )


def _column(frame: Frame, column: str) -> np.ndarray:
    return frame.table("person")[column].to_numpy(dtype=float)


def _person_weights(frame: Frame) -> np.ndarray:
    return np.asarray(frame.resolve_weights("person").values, dtype=float)


def _source_rows(frame: Frame) -> np.ndarray:
    return frame.table("person")[RAW[0]].notna().to_numpy()


def _total(frame: Frame, rows: np.ndarray | None = None) -> float:
    weights, values = _person_weights(frame), _column(frame, EMPLOYER)
    rows = np.ones(len(values), dtype=bool) if rows is None else rows
    return float(weights[rows] @ values[rows])


_amount = st.one_of(
    st.just(0.0),
    st.floats(min_value=1.0, max_value=40_000.0, allow_nan=False),
)
_person = st.tuples(_amount, _amount)
_weight = st.floats(min_value=0.5, max_value=5_000.0, allow_nan=False)


def _households(source: bool) -> st.SearchStrategy[list[dict]]:
    return st.lists(
        st.fixed_dictionaries(
            {
                "source": st.just(source),
                "weight": _weight,
                "people": st.lists(_person, min_size=1, max_size=3),
            }
        ),
        min_size=1,
        max_size=8,
    )


@st.composite
def _pools(draw) -> Frame:
    households = draw(_households(True)) + draw(_households(False))
    frame = _pool(households)
    source = _source_rows(frame)
    wages = _column(frame, WAGES)
    employer = _column(frame, EMPLOYER)
    assume(_total(frame, source) > 0)
    assume(((~source) & (wages > 0) & (employer > 0)).any())
    return frame


_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


@given(_pools())
@_SETTINGS
def test_pool_total_is_the_source_total_over_its_mass_share(frame: Frame) -> None:
    source = _source_rows(frame)
    share = us_esi_premiums_household_mass_share(frame, source)

    anchored, receipt = with_us_esi_premium_pool_anchor(frame)

    expected = _total(frame, source) / share
    assert _total(anchored) == pytest.approx(expected, rel=1e-9)
    assert receipt["employer_premium_total"] == pytest.approx(expected, rel=1e-9)
    assert receipt["pool_employer_premium_target"] == pytest.approx(expected, rel=1e-12)
    assert receipt["source_household_mass_share"] == share


@given(_pools())
@_SETTINGS
def test_both_sides_carry_the_same_premium_per_unit_of_household_mass(
    frame: Frame,
) -> None:
    source = _source_rows(frame)
    share = us_esi_premiums_household_mass_share(frame, source)

    anchored, _receipt = with_us_esi_premium_pool_anchor(frame)

    assert _total(anchored, ~source) / (1.0 - share) == pytest.approx(
        _total(anchored, source) / share, rel=1e-9
    )


@given(_pools())
@_SETTINGS
def test_source_rows_and_every_other_column_are_untouched(frame: Frame) -> None:
    source = _source_rows(frame)

    anchored, _receipt = with_us_esi_premium_pool_anchor(frame)

    before, after = frame.table("person"), anchored.table("person")
    assert list(after.columns) == list(before.columns)
    pd.testing.assert_frame_equal(after.loc[source], before.loc[source])
    untouched = [
        column
        for column in before.columns
        if column not in US_ESI_PREMIUMS_OUTPUT_COLUMNS
    ]
    pd.testing.assert_frame_equal(after[untouched], before[untouched])
    for entity in frame.entities:
        if entity != "person":
            pd.testing.assert_frame_equal(anchored.table(entity), frame.table(entity))
    np.testing.assert_array_equal(_person_weights(anchored), _person_weights(frame))


@given(_pools())
@_SETTINGS
def test_transferred_rows_without_wages_carry_nothing_and_the_rest_one_multiple(
    frame: Frame,
) -> None:
    source = _source_rows(frame)
    workers = _column(frame, WAGES) > 0

    anchored, receipt = with_us_esi_premium_pool_anchor(frame)

    cleared = ~source & ~workers
    for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS:
        assert not _column(anchored, column)[cleared].any()
        assert receipt["cleared_rows"][column] == int(
            (cleared & (_column(frame, column) > 0)).sum()
        )
        if column != EMPLOYER:
            kept = ~source & workers
            np.testing.assert_array_equal(
                _column(anchored, column)[kept], _column(frame, column)[kept]
            )
    scaled = ~source & workers
    np.testing.assert_allclose(
        _column(anchored, EMPLOYER)[scaled],
        _column(frame, EMPLOYER)[scaled] * receipt["scale_factor"],
        rtol=1e-12,
    )
    assert np.isfinite(_column(anchored, EMPLOYER)).all()
    assert (_column(anchored, EMPLOYER) >= 0).all()


@given(_pools())
@_SETTINGS
def test_a_pool_on_the_stage_scale_is_returned_unchanged(frame: Frame) -> None:
    anchored, _receipt = with_us_esi_premium_pool_anchor(frame)

    again, receipt = with_us_esi_premium_pool_anchor(anchored)

    assert again is anchored
    assert receipt["status"] == "already_on_anchor"
    assert receipt["scale_factor"] == pytest.approx(1.0, rel=1e-12)


@given(_pools(), st.floats(min_value=1e-3, max_value=1e3, allow_nan=False))
@_SETTINGS
def test_rescaling_every_household_weight_leaves_each_value_where_it_was(
    frame: Frame, factor: float
) -> None:
    rescaled = Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {
            "household": Weights(
                values=np.asarray(frame.weights_for("household").values) * factor,
                kind=WeightKind.IMPORTANCE,
            )
        },
    )

    anchored, _receipt = with_us_esi_premium_pool_anchor(frame)
    anchored_rescaled, _receipt = with_us_esi_premium_pool_anchor(rescaled)

    np.testing.assert_allclose(
        _column(anchored_rescaled, EMPLOYER), _column(anchored, EMPLOYER), rtol=1e-9
    )


def test_the_worked_two_source_pool() -> None:
    # Source side: half the household mass, $600 of weighted premium.
    # Transferred side: $300 drawn on a worker, $500 on a non-worker.
    frame = _pool(
        [
            {"source": True, "weight": 2.0, "people": [(300.0, 50_000.0)]},
            {"source": False, "weight": 1.0, "people": [(300.0, 40_000.0)]},
            {"source": False, "weight": 1.0, "people": [(500.0, 0.0)]},
        ]
    )

    anchored, receipt = with_us_esi_premium_pool_anchor(frame)

    assert receipt["source_household_mass_share"] == 0.5
    assert receipt["transferred_employer_premium_target"] == 600.0
    assert receipt["scale_factor"] == 2.0
    assert receipt["cleared_rows"][EMPLOYER] == 1
    assert receipt["cleared_weighted_total"][EMPLOYER] == 500.0
    np.testing.assert_array_equal(_column(anchored, EMPLOYER), [300.0, 600.0, 0.0])
    assert _total(anchored) == 1_200.0
    assert receipt["status"] == "scaled"


def test_support_clones_follow_their_source_record_wages() -> None:
    # Households 3 and 4 are the support clones of 1 and 2. A clone's own
    # wages are another tax-detail vector; the survey-side wages decide.
    frame = _pool(
        [
            {"source": True, "weight": 1.0, "people": [(400.0, 50_000.0)]},
            {"source": True, "weight": 1.0, "people": [(400.0, 0.0)]},
            {"source": False, "weight": 1.0, "people": [(200.0, 30_000.0)]},
            {"source": False, "weight": 1.0, "people": [(200.0, 0.0)]},
            {"source": False, "weight": 1.0, "people": [(200.0, 0.0)]},
            {"source": False, "weight": 1.0, "people": [(200.0, 9_000.0)]},
        ],
        clone_of={4: 2, 5: 3},
    )

    anchored, receipt = with_us_esi_premium_pool_anchor(frame)

    employer = _column(anchored, EMPLOYER)
    # The worker's clone keeps the premium though its own wages are zero; the
    # non-worker's clone loses it though its own wages are positive.
    assert employer[2] == employer[4] > 0
    assert employer[3] == employer[5] == 0
    assert receipt["transferred_rows_without_source_wages"] == 2
    np.testing.assert_array_equal(employer[:2], [400.0, 400.0])


def test_a_frame_with_no_transferred_row_is_returned_unchanged() -> None:
    frame = _pool([{"source": True, "weight": 3.0, "people": [(100.0, 1.0)]}])

    anchored, receipt = with_us_esi_premium_pool_anchor(frame)

    assert anchored is frame
    assert receipt["status"] == "no_transferred_rows"
    assert receipt["source_household_mass_share"] == 1.0
    assert receipt["employer_premium_total"] == 300.0


def _two_sided(**changes) -> list[dict]:
    households = [
        {"source": True, "weight": 1.0, "people": [(100.0, 1.0)]},
        {"source": False, "weight": 1.0, "people": [(100.0, 1.0)]},
    ]
    for position, change in changes.items():
        households[int(position.removeprefix("h"))] |= change
    return households


def _with_person_cell(frame: Frame, row: int, column: str, value: object) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"].loc[row, column] = value
    return Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
    )


def test_a_row_with_only_some_raw_fields_is_refused() -> None:
    frame = _with_person_cell(_pool(_two_sided()), 1, RAW[0], 1.0)

    with pytest.raises(SourceRuntimeError, match="only some of the raw ASEC fields"):
        with_us_esi_premium_pool_anchor(frame)


def test_a_household_mixing_both_kinds_of_row_is_refused() -> None:
    frame = _pool(
        [{"source": True, "weight": 1.0, "people": [(100.0, 1.0), (50.0, 1.0)]}]
    )
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"].loc[1, list(RAW)] = np.nan
    mixed = Frame(tables, frame.schema, {"household": frame.weights_for("household")})

    with pytest.raises(SourceRuntimeError, match="mix source-derived and transferred"):
        with_us_esi_premium_pool_anchor(mixed)


@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0])
def test_an_unfilled_or_negative_premium_is_refused(value: float) -> None:
    frame = _with_person_cell(_pool(_two_sided()), 1, EMPLOYER, value)

    with pytest.raises(SourceRuntimeError, match="finite and nonnegative"):
        with_us_esi_premium_pool_anchor(frame)


def test_transferred_rows_with_no_premium_on_workers_are_refused() -> None:
    frame = _pool(_two_sided(h1={"people": [(100.0, 0.0)]}))

    with pytest.raises(SourceRuntimeError, match="nothing to scale"):
        with_us_esi_premium_pool_anchor(frame)


def test_source_rows_with_no_premium_mass_are_refused() -> None:
    frame = _pool(_two_sided(h0={"people": [(0.0, 1.0)]}))

    with pytest.raises(SourceRuntimeError, match="source-derived rows hold no"):
        with_us_esi_premium_pool_anchor(frame)


def test_missing_wages_or_outputs_are_refused() -> None:
    frame = _pool(_two_sided())
    for column in (WAGES, EMPLOYER, RAW[0]):
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        tables["person"] = tables["person"].drop(columns=[column])
        stripped = Frame(
            tables, frame.schema, {"household": frame.weights_for("household")}
        )
        with pytest.raises(SourceRuntimeError, match=column):
            with_us_esi_premium_pool_anchor(stripped)


def test_a_support_clone_without_its_source_record_is_refused() -> None:
    frame = _pool(_two_sided(), clone_of={1: 0})
    orphaned = _with_person_cell(frame, 1, "person_source_id", 999_999)

    with pytest.raises(SourceRuntimeError, match="no clone-0 source record"):
        with_us_esi_premium_pool_anchor(orphaned)


def test_household_mass_share_is_weight_not_headcount() -> None:
    frame = _pool(
        [
            {"source": True, "weight": 3.0, "people": [(1.0, 1.0)]},
            {"source": False, "weight": 1.0, "people": [(1.0, 1.0)] * 3},
        ]
    )

    assert us_esi_premiums_household_mass_share(frame, _source_rows(frame)) == 0.75
    assert us_esi_premiums_household_mass_share(frame, ~_source_rows(frame)) == 0.25
    with pytest.raises(SourceRuntimeError, match="one flag per person"):
        us_esi_premiums_household_mass_share(frame, np.array([True]))
