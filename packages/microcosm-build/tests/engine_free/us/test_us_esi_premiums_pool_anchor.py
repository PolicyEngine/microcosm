"""The ESI pool anchor and the row rule for mixed frames (microcosm #454).

A mixed frame holds CPS rows, which carry the raw ASEC fields the stage
reads, beside rows a cross-source transfer filled, which carry none of them
(the ACS local lane's pooled frame is one). Taken from microcosm#1199
(e730a951a, ``test_us_esi_premiums_pool.py``) without its stacked-pool
wiring, for d2283 PR 0.

Invariants of :func:`with_us_esi_premium_pool_anchor`, for every valid frame:

* **Conservation**: the frame-wide weighted employer premium equals the
  source-derived rows' total divided by their share of household mass.
* **Equal intensity**: source-derived and transferred rows carry the same
  weighted employer premium per unit of household mass.
* **Source rows are immutable**: no source-derived cell changes by a byte.
* **One premium per person**: every support clone of a transferred person
  carries its clone-0 source record's draw, whether the transfer ran before
  or after the clone.
* **Structural zeros**: a transferred person whose source record reports zero
  wages carries no premium, on every support clone. A missing wage is
  refused, never read as zero.
* **Proportionality**: every other transferred employer premium is one common
  multiple of the value the transfer drew for the source record.
* **Idempotence**: a frame already on the stage's scale is returned unchanged.
* **Weight homogeneity**: rescaling every household weight leaves each value
  where it was.

The gates grade such a frame only when the caller passes
``allow_transferred_rows=True``; the default keeps the single-source rule
every Route A caller uses.
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
    us_esi_premiums_anchor_gate,
    us_esi_premiums_signal_gate,
    us_esi_premiums_summary,
    with_us_esi_premium_pool_anchor,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_esi_premiums_frames import (
    ANCHOR,
    YEAR,
    asec_arm,
    cps_rows,
    households_of,
    mixed_frame,
    reweighted,
    staged,
    with_person_cells,
    with_scaled_premiums,
)

EMPLOYER = US_ESI_EMPLOYER_PREMIUM_COLUMN
WAGES = US_ESI_PREMIUMS_WAGE_COLUMN
RAW = esi._RAW_EVIDENCE_COLUMNS
CPS_ID = esi.US_ESI_PREMIUMS_CPS_RECORD_COLUMN
RAW_WAGE = esi._RAW_ACS_WAGE_COLUMN

_GROUPS = ("tax_unit", "spm_unit", "family", "marital_unit")


def _pool(
    households: list[dict],
    *,
    clone_of: dict[int, int] | None = None,
    clone_index: dict[int, int] | None = None,
) -> Frame:
    """A pool frame from household records.

    Each record has ``source`` (whether its people carry raw ASEC fields),
    ``weight`` and ``people``: a list of ``(employer premium, wages)``. Every
    other stage output takes half the employer premium. ``clone_of`` maps a
    household position to the position of the household it is a support clone
    of; clones then share ``person_source_id`` with their source person.
    ``clone_index`` gives a clone's support index (1 by default; 2 is the
    capital-gains tail descendant).
    """

    clone_of = clone_of or {}
    clone_index = clone_index or {}
    records = []
    for position, household in enumerate(households):
        origin = clone_of.get(position, position)
        for member, (premium, wages) in enumerate(household["people"]):
            record = {
                "person_household_id": position + 1,
                "person_source_id": (origin + 1) * 100 + member,
                "person_support_clone_index": (
                    clone_index.get(position, 1) if position in clone_of else 0
                ),
                WAGES: float(wages),
            }
            record |= {
                column: float(premium) * (1.0 if column == EMPLOYER else 0.5)
                for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS
            }
            record |= {column: 1.0 if household["source"] else np.nan for column in RAW}
            # The CPS record id corroborates the row kind the raw fields imply.
            record[CPS_ID] = (
                f"cps-{origin + 1:04d}-{member}" if household["source"] else None
            )
            # The ACS earnings universe the anchor resolves wage evidence in:
            # adults, with the raw wage the mapped wage came from.
            record["person_support_channel"] = "asec" if household["source"] else "acs"
            record["age"] = 40.0
            record[RAW_WAGE] = np.nan if household["source"] else float(wages)
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
_weight = st.one_of(
    st.just(0.0),
    st.floats(min_value=0.5, max_value=5_000.0, allow_nan=False),
    # Masses many orders apart, where a share's complement loses precision.
    st.floats(min_value=1e-9, max_value=1e-3, allow_nan=False),
    st.floats(min_value=1e6, max_value=1e10, allow_nan=False),
)


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
        max_size=6,
    )


def _record(frame: Frame, column: str) -> np.ndarray:
    """Each row's value of ``column`` on its clone-0 source record."""

    person = frame.table("person")
    native = person["person_support_clone_index"].eq(0)
    by_source = person.loc[native].set_index("person_source_id")[column]
    return person["person_source_id"].map(by_source).to_numpy(dtype=float)


def _masses(frame: Frame) -> tuple[float, float]:
    """Household mass of the source-derived and the transferred households."""

    person = frame.table("person")
    source_households = set(person.loc[_source_rows(frame), "person_household_id"])
    household = frame.table("household")
    weights = np.asarray(frame.weights_for("household").values, dtype=float)
    on_source = household["household_id"].isin(source_households).to_numpy()
    return float(weights[on_source].sum()), float(weights[~on_source].sum())


def _transferred_mass(frame: Frame) -> float:
    return _masses(frame)[1]


@st.composite
def _pools(draw) -> Frame:
    """Valid pools: either household kind may hold zero-weight households and
    support clones (index 1 and the tail's index 2).

    A source household's clone copies its people, as the pool's clone does. A
    transferred household's clone gets its own draws and wages, as a transfer
    that ran after the clone would leave it.
    """

    households = draw(_households(True)) + draw(_households(False))
    clone_of: dict[int, int] = {}
    clone_index: dict[int, int] = {}
    for position in range(len(households)):
        origin = households[position]
        for index in sorted(draw(st.sets(st.sampled_from([1, 2])))):
            people = (
                origin["people"]
                if origin["source"]
                else draw(
                    st.lists(
                        _person,
                        min_size=len(origin["people"]),
                        max_size=len(origin["people"]),
                    )
                )
            )
            clone_of[len(households)] = position
            clone_index[len(households)] = index
            households.append(
                {"source": origin["source"], "weight": draw(_weight), "people": people}
            )
    # A frame needs weight somewhere, and the anchor needs it on a source row.
    assume(any(household["weight"] > 0 for household in households))
    frame = _pool(households, clone_of=clone_of, clone_index=clone_index)
    source = _source_rows(frame)
    assume(_total(frame, source) > 0)
    weights = _person_weights(frame)
    workers = ~source & (_record(frame, WAGES) > 0)
    drawn_on_workers = float(weights[workers] @ _record(frame, EMPLOYER)[workers])
    # Transferred mass needs premium on workers to scale; none needs nothing.
    assume(drawn_on_workers > 0 or _transferred_mass(frame) == 0)
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
    source_mass, transferred_mass = _masses(frame)
    selected, unselected = esi._household_masses(frame, source)
    share = selected / (selected + unselected)

    anchored, receipt = with_us_esi_premium_pool_anchor(frame)

    expected = _total(frame, source) * (source_mass + transferred_mass) / source_mass
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
    source_mass, transferred_mass = _masses(frame)
    assume(transferred_mass > 0)

    anchored, _receipt = with_us_esi_premium_pool_anchor(frame)

    assert _total(anchored, ~source) / transferred_mass == pytest.approx(
        _total(anchored, source) / source_mass, rel=1e-9
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
    workers = _record(frame, WAGES) > 0

    anchored, receipt = with_us_esi_premium_pool_anchor(frame)

    cleared = ~source & ~workers
    kept = ~source & workers
    for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS:
        drawn = _record(frame, column)
        assert not _column(anchored, column)[cleared].any()
        assert receipt["cleared_rows"][column] == int((cleared & (drawn > 0)).sum())
        assert receipt["transferred_rows_reset_to_source_record"][column] == int(
            (~source & (drawn != _column(frame, column))).sum()
        )
        if column != EMPLOYER:
            np.testing.assert_array_equal(_column(anchored, column)[kept], drawn[kept])
    np.testing.assert_allclose(
        _column(anchored, EMPLOYER)[kept],
        _record(frame, EMPLOYER)[kept] * receipt["scale_factor"],
        rtol=1e-12,
    )
    assert np.isfinite(_column(anchored, EMPLOYER)).all()
    assert (_column(anchored, EMPLOYER) >= 0).all()


@given(_pools())
@_SETTINGS
def test_every_support_clone_of_a_transferred_person_carries_one_premium(
    frame: Frame,
) -> None:
    anchored, _receipt = with_us_esi_premium_pool_anchor(frame)

    person = anchored.table("person")
    transferred = ~_source_rows(anchored)
    per_person = person.loc[transferred].groupby("person_source_id")[
        list(US_ESI_PREMIUMS_OUTPUT_COLUMNS)
    ]
    assert (per_person.nunique() == 1).all().all()
    np.testing.assert_array_equal(
        _column(anchored, EMPLOYER)[transferred],
        _record(anchored, EMPLOYER)[transferred],
    )


@given(_pools())
@_SETTINGS
def test_a_pool_on_the_stage_scale_is_returned_unchanged(frame: Frame) -> None:
    anchored, _receipt = with_us_esi_premium_pool_anchor(frame)

    again, receipt = with_us_esi_premium_pool_anchor(anchored)

    assert again is anchored
    assert receipt["status"] == (
        "already_on_anchor" if _transferred_mass(frame) > 0 else "no_transferred_mass"
    )
    assert receipt["scale_factor"] == pytest.approx(1.0, rel=1e-12)


# Up to 1e150 either way: multiplying the source total by the transferred
# mass before dividing by the source mass overflowed well inside this range.
@given(_pools(), st.floats(min_value=1e-150, max_value=1e150, allow_nan=False))
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


def test_a_transfer_that_ran_after_the_clone_is_reset_to_one_premium_per_person() -> (
    None
):
    # The legacy two-spine order clones first, so its transfer draws every
    # support clone separately: $100 on the source record, $700 and nothing
    # on its clones, whose own wages are another tax-detail vector.
    frame = _pool(
        [
            {"source": True, "weight": 3.0, "people": [(300.0, 50_000.0)]},
            {"source": False, "weight": 1.0, "people": [(100.0, 40_000.0)]},
            {"source": False, "weight": 1.0, "people": [(700.0, 0.0)]},
            {"source": False, "weight": 1.0, "people": [(0.0, 9.0)]},
        ],
        clone_of={2: 1, 3: 1},
        clone_index={3: 2},
    )

    anchored, receipt = with_us_esi_premium_pool_anchor(frame)

    # Every clone takes the source record's $100, then the one factor.
    np.testing.assert_array_equal(
        _column(anchored, EMPLOYER), [300.0, 300.0, 300.0, 300.0]
    )
    assert receipt["transferred_rows_reset_to_source_record"][EMPLOYER] == 2
    assert receipt["cleared_rows"][EMPLOYER] == 0
    assert receipt["scale_factor"] == 3.0
    assert _total(anchored) == 1_800.0


def test_transferred_rows_without_household_mass_are_not_scaled() -> None:
    # Nothing to hold a zero-mass side to: structural zeros still apply, the
    # factor stays at one and the pool total is the source total.
    frame = _pool(
        [
            {"source": True, "weight": 2.0, "people": [(300.0, 50_000.0)]},
            {"source": False, "weight": 0.0, "people": [(250.0, 40_000.0)]},
            {"source": False, "weight": 0.0, "people": [(500.0, 0.0)]},
        ]
    )

    anchored, receipt = with_us_esi_premium_pool_anchor(frame)

    assert receipt["source_household_mass_share"] == 1.0
    assert receipt["transferred_household_mass_share"] == 0.0
    assert receipt["transferred_employer_premium_target"] == 0.0
    assert receipt["scale_factor"] == 1.0
    np.testing.assert_array_equal(_column(anchored, EMPLOYER), [300.0, 250.0, 0.0])
    assert _total(anchored) == 600.0 == receipt["pool_employer_premium_target"]

    again, receipt = with_us_esi_premium_pool_anchor(anchored)
    assert again is anchored
    assert receipt["status"] == "no_transferred_mass"


def test_zero_mass_transferred_rows_need_no_premium_on_workers() -> None:
    frame = _pool(_two_sided(h1={"weight": 0.0, "people": [(100.0, 0.0)]}))

    anchored, receipt = with_us_esi_premium_pool_anchor(frame)

    np.testing.assert_array_equal(_column(anchored, EMPLOYER), [100.0, 0.0])
    assert receipt["scale_factor"] == 1.0


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


def _with_person_cell(
    frame: Frame, row: int, column: str | list[str], value: object
) -> Frame:
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
    mixed = _with_person_cell(frame, 1, [*RAW, CPS_ID], None)

    with pytest.raises(SourceRuntimeError, match="mix source-derived and transferred"):
        with_us_esi_premium_pool_anchor(mixed)


def _two_cps_one_transferred() -> Frame:
    return _pool(
        [
            {"source": True, "weight": 1.0, "people": [(100.0, 1.0)]},
            {"source": True, "weight": 1.0, "people": [(100.0, 1.0)]},
            {"source": False, "weight": 2.0, "people": [(100.0, 1.0)]},
        ]
    )


def test_a_cps_record_that_lost_its_raw_fields_is_not_a_transferred_row() -> None:
    # Nulls in every coverage field look like a transferred row. The CPS
    # record id says otherwise, and the premium is not taken on trust.
    frame = _with_person_cell(_two_cps_one_transferred(), 1, list(RAW), np.nan)

    with pytest.raises(
        SourceRuntimeError, match=r"1 CPS record\(s\).*lack the raw ASEC fields"
    ):
        with_us_esi_premium_pool_anchor(frame)


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_raw_fields_without_a_cps_record_id_are_refused(blank: object) -> None:
    frame = _with_person_cell(_two_cps_one_transferred(), 1, CPS_ID, blank)

    with pytest.raises(SourceRuntimeError, match="without a CPS record id"):
        with_us_esi_premium_pool_anchor(frame)


def test_a_frame_without_the_cps_record_id_cannot_hold_transferred_rows() -> None:
    frame = _two_cps_one_transferred()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = tables["person"].drop(columns=[CPS_ID])
    unmarked = Frame(
        tables, frame.schema, {"household": frame.weights_for("household")}
    )

    with pytest.raises(SourceRuntimeError, match=f"no {CPS_ID!r} column"):
        with_us_esi_premium_pool_anchor(unmarked)


@pytest.mark.parametrize("wage", [np.nan, None])
def test_a_transferred_person_with_a_missing_wage_is_refused(wage: object) -> None:
    # Missing evidence is not a zero: the premium is neither cleared nor kept.
    frame = _with_person_cell(_pool(_two_sided()), 1, WAGES, wage)

    with pytest.raises(SourceRuntimeError, match="A missing wage is not a zero wage"):
        with_us_esi_premium_pool_anchor(frame)


def test_a_missing_wage_on_a_source_record_blocks_its_clones_too() -> None:
    # The clone's own mapped wage is a number; its source record's is not.
    frame = _pool(_two_sided() + [_two_sided()[1]], clone_of={2: 1})
    frame = _with_person_cell(frame, 1, WAGES, np.nan)

    with pytest.raises(SourceRuntimeError, match="A missing wage is not a zero wage"):
        with_us_esi_premium_pool_anchor(frame)


def test_a_zero_filled_wage_without_a_raw_wage_is_refused() -> None:
    # The legacy PUF pass fills a missing mapped wage with zero. Census asks
    # every person 15 or older for WAGP, so a blank raw field there is missing
    # evidence, not a zero, however the mapped column was filled.
    frame = _pool(_two_sided(h1={"people": [(100.0, 0.0)]}))
    frame = _with_person_cell(frame, 1, RAW_WAGE, np.nan)

    with pytest.raises(SourceRuntimeError, match=r"raw_in_universe_null_rows=1"):
        with_us_esi_premium_pool_anchor(frame)


def test_a_child_without_a_raw_wage_carries_the_named_universe_zero() -> None:
    # Below 15 Census leaves WAGP blank; the mapped zero is the universe rule's,
    # so the drawn premium is a structural zero and is cleared.
    frame = _pool(
        [
            {"source": True, "weight": 1.0, "people": [(100.0, 50_000.0)]},
            {"source": False, "weight": 1.0, "people": [(80.0, 40_000.0)]},
            {"source": False, "weight": 1.0, "people": [(50.0, 0.0)]},
        ]
    )
    frame = _with_person_cell(frame, 2, ["age", RAW_WAGE], [9.0, np.nan])

    anchored, receipt = with_us_esi_premium_pool_anchor(frame)

    np.testing.assert_array_equal(_column(anchored, EMPLOYER), [100.0, 200.0, 0.0])
    evidence = receipt["transferred_wage_evidence"]
    assert evidence["rule_id"] == "acs_2024_pums_wagp_age_15_plus"
    assert evidence["structurally_absent_person_rows"] == 1
    assert evidence["mapped_universe_zero_rows"] == 1
    assert evidence["eligible_acs_person_rows"] == 1


def test_a_child_with_a_nonzero_mapped_wage_is_refused() -> None:
    frame = _pool(_two_sided())
    frame = _with_person_cell(frame, 1, ["age", RAW_WAGE], [9.0, np.nan])

    with pytest.raises(
        SourceRuntimeError, match="out_of_universe_mapped_nonzero_rows=1"
    ):
        with_us_esi_premium_pool_anchor(frame)


@pytest.mark.parametrize("multiplier", [1.0, 0.1, 1e-3, 1e3])
def test_a_tiny_transferred_side_keeps_one_factor_at_any_weight_scale(
    multiplier: float,
) -> None:
    # Household masses 1e8 and 1e-8 with equal premium per unit of mass. Taking
    # the transferred share as one minus the source share loses it to rounding
    # and gave factors of 1.11 and 2.22 for the same pool at two weight scales.
    frame = _pool(
        [
            {"source": True, "weight": 1e8 * multiplier, "people": [(100.0, 1.0)]},
            {"source": False, "weight": 1e-8 * multiplier, "people": [(100.0, 1.0)]},
        ]
    )

    anchored, receipt = with_us_esi_premium_pool_anchor(frame)

    assert receipt["scale_factor"] == pytest.approx(1.0, rel=1e-12)
    assert receipt["transferred_household_mass"] == pytest.approx(
        1e-8 * multiplier, rel=1e-12
    )
    np.testing.assert_allclose(_column(anchored, EMPLOYER), [100.0, 100.0], rtol=1e-12)


def test_a_transferred_side_far_below_rounding_is_still_scaled() -> None:
    # Source mass 1e16, transferred mass 1: the share rounds to exactly one,
    # yet the transferred side holds mass and is held to the source intensity.
    frame = _pool(
        [
            {"source": True, "weight": 1e16, "people": [(100.0, 1.0)]},
            {"source": False, "weight": 1.0, "people": [(40.0, 1.0)]},
        ]
    )

    anchored, receipt = with_us_esi_premium_pool_anchor(frame)

    assert receipt["source_household_mass_share"] == 1.0
    assert receipt["transferred_household_mass"] == 1.0
    assert receipt["scale_factor"] == pytest.approx(2.5, rel=1e-12)
    np.testing.assert_allclose(_column(anchored, EMPLOYER), [100.0, 100.0], rtol=1e-12)


@pytest.mark.parametrize("mass", [1.0, 1e-300, 1e-200, 1e152, 1e153, 1e300])
def test_the_factor_and_targets_stay_finite_wherever_the_correct_ones_are(
    mass: float,
) -> None:
    # Equal household mass on each side, $10,000 and $5,000 of employer premium
    # per unit of it: the factor is 2 at every scale. Multiplying the source
    # total by the transferred mass before dividing by the source mass gave an
    # infinite target at 1e152, an infinite factor at 1e153 and a zero one at
    # 1e-200.
    frame = _pool(
        [
            {"source": True, "weight": mass, "people": [(10_000.0, 1.0)]},
            {"source": False, "weight": mass, "people": [(5_000.0, 1.0)]},
        ]
    )

    anchored, receipt = with_us_esi_premium_pool_anchor(frame)

    assert receipt["status"] == "scaled"
    assert receipt["scale_factor"] == pytest.approx(2.0, rel=1e-12)
    # ``abs=0``: approx's default absolute tolerance would pass anything at 1e-300.
    assert receipt["transferred_employer_premium_target"] == pytest.approx(
        1e4 * mass, rel=1e-12, abs=0.0
    )
    assert receipt["pool_employer_premium_target"] == pytest.approx(
        2e4 * mass, rel=1e-12, abs=0.0
    )
    assert receipt["employer_premium_total"] == pytest.approx(
        2e4 * mass, rel=1e-12, abs=0.0
    )
    np.testing.assert_allclose(
        _column(anchored, EMPLOYER), [10_000.0, 10_000.0], rtol=1e-12
    )


@pytest.mark.parametrize(
    "mass",
    [
        # The source total itself is past the largest float.
        1e306,
        # Each side's total is finite; the pool's is not.
        1.5e304,
    ],
)
def test_a_pool_whose_correct_total_is_not_finite_is_refused(mass: float) -> None:
    frame = _pool(
        [
            {"source": True, "weight": mass, "people": [(10_000.0, 1.0)]},
            {"source": False, "weight": mass, "people": [(10_000.0, 1.0)]},
        ]
    )

    with pytest.raises(SourceRuntimeError, match="target is not finite"):
        with_us_esi_premium_pool_anchor(frame)


def test_a_missing_wage_on_a_source_derived_row_is_not_read() -> None:
    frame = _with_person_cell(_pool(_two_sided()), 0, WAGES, np.nan)

    anchored, receipt = with_us_esi_premium_pool_anchor(frame)

    assert anchored is frame
    assert receipt["status"] == "already_on_anchor"


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


def test_household_masses_are_weight_not_headcount() -> None:
    frame = _pool(
        [
            {"source": True, "weight": 3.0, "people": [(1.0, 1.0)]},
            {"source": False, "weight": 1.0, "people": [(1.0, 1.0)] * 3},
        ]
    )

    assert esi._household_masses(frame, _source_rows(frame)) == (3.0, 1.0)
    assert esi._household_masses(frame, ~_source_rows(frame)) == (1.0, 3.0)
    with pytest.raises(SourceRuntimeError, match="one flag per person"):
        esi._household_masses(frame, np.array([True]))


# --- The default keeps the single-source rule ---------------------------------


@pytest.fixture(scope="module")
def alone() -> Frame:
    """A single-source CPS frame, staged as the base builder stages it."""

    return staged(asec_arm())


@pytest.fixture(scope="module")
def mixed() -> Frame:
    """320 CPS rows beside 280 transferred rows, on the stage's scale."""

    return mixed_frame()


def test_the_default_refuses_a_transferred_row(mixed, alone) -> None:
    with pytest.raises(
        SourceRuntimeError, match=r"280 row\(s\) lack some of the raw ASEC fields"
    ):
        us_esi_premiums_summary(mixed)
    for gate in (
        us_esi_premiums_signal_gate(mixed),
        us_esi_premiums_anchor_gate(mixed, time_period=YEAR),
    ):
        assert not gate.passed
        assert "allow_transferred_rows=True" in gate.failures[0]
    # One row with neither the raw fields nor a CPS record id is enough.
    one = with_person_cells(alone, [3], [*RAW, CPS_ID], None)
    with pytest.raises(SourceRuntimeError, match=r"1 row\(s\) lack some"):
        us_esi_premiums_summary(one)
    assert not us_esi_premiums_signal_gate(one).passed
    assert (
        us_esi_premiums_summary(one, allow_transferred_rows=True)["transferred_rows"]
        == 1
    )


def test_the_default_refuses_a_partial_row(mixed, alone) -> None:
    transferred = np.flatnonzero(~cps_rows(mixed))
    for frame in (
        with_person_cells(alone, [5], RAW[0], np.nan),
        with_person_cells(mixed, transferred[:1], RAW[0], 2),
    ):
        with pytest.raises(SourceRuntimeError, match="lack some of the raw ASEC"):
            us_esi_premiums_summary(frame)
        with pytest.raises(SourceRuntimeError, match="only some of the raw ASEC"):
            us_esi_premiums_summary(frame, allow_transferred_rows=True)


def test_the_default_never_reads_the_cps_record_id(alone) -> None:
    frame = with_person_cells(alone, [5], CPS_ID, None)

    assert us_esi_premiums_signal_gate(frame).passed
    assert us_esi_premiums_anchor_gate(frame, time_period=YEAR).passed
    lane = us_esi_premiums_signal_gate(frame, allow_transferred_rows=True)
    assert not lane.passed
    assert "without a CPS record id" in lane.failures[0]


# --- Grading a mixed frame (allow_transferred_rows=True) -----------------------


def _gates(frame: Frame) -> tuple:
    return (
        us_esi_premiums_signal_gate(frame, allow_transferred_rows=True),
        us_esi_premiums_anchor_gate(
            frame, time_period=YEAR, allow_transferred_rows=True
        ),
    )


def test_both_gates_pass_on_the_mixed_frame(mixed, alone) -> None:
    signal, anchor = _gates(mixed)

    assert signal.passed, signal.failures
    assert anchor.passed, anchor.failures
    # The cell proofs ran on the CPS rows, and found them one common multiple
    # of their MEPS-IC shares: the single-source factor.
    reference = us_esi_premiums_summary(alone)
    assert signal.details["source_rows"] == 320
    assert signal.details["transferred_rows"] == 280
    assert signal.details["scale_factor"] == pytest.approx(
        reference["scale_factor"], rel=1e-9
    )
    assert signal.details["employer_premium_outside_universe_rows"] == 0
    assert signal.details["clone_disagreement_source_persons"] == 0
    assert signal.details["transferred_source_records_with_premium_and_no_wages"] == 0
    assert anchor.details["anchor_universe_employer_total"] == pytest.approx(
        ANCHOR, rel=1e-9
    )
    assert anchor.details["relative_error"] == pytest.approx(0.0, abs=1e-9)
    assert anchor.details["source_household_mass_share"] == pytest.approx(
        0.5, rel=1e-12
    )
    assert anchor.details[
        "transferred_to_source_per_household_mass_ratio"
    ] == pytest.approx(1.0, rel=1e-9)
    assert anchor.details["employed_share_of_anchor_universe"] == pytest.approx(
        reference["employed_share_of_anchor_universe"], rel=1e-12
    )
    assert "employed share" in anchor.details["anchor_universe_assumption"]
    # Scaled to the population, the private-sector cross-check matches the
    # single-source one.
    assert anchor.details["private_active_cross_check"][
        "employer_premium_private_sector"
    ] == pytest.approx(reference["employer_premium_by_sector"]["private"], rel=1e-9)


def test_transferred_rows_without_the_premium_fail_the_signal_gate(mixed) -> None:
    flattened = with_scaled_premiums(mixed, 0.0, ~cps_rows(mixed))

    signal, _anchor = _gates(flattened)

    assert not signal.passed
    assert any(
        "positive share on the rows without raw ASEC columns" in failure
        for failure in signal.failures
    )


@pytest.mark.parametrize("factor", [0.7, 1.4])
def test_transferred_rows_off_the_cps_rows_per_mass_total_fail_the_anchor_gate(
    mixed, factor: float
) -> None:
    skewed = with_scaled_premiums(mixed, factor, ~cps_rows(mixed))

    _signal, anchor = _gates(skewed)

    assert not anchor.passed
    assert anchor.details[
        "transferred_to_source_per_household_mass_ratio"
    ] == pytest.approx(factor, rel=1e-9)
    assert any("per unit of household mass" in failure for failure in anchor.failures)
    # Half the mass at ``factor`` moves the anchor-universe total by half of it.
    assert anchor.details["relative_error"] == pytest.approx(
        (factor - 1.0) / 2.0, rel=1e-6
    )


def test_reweighting_one_side_moves_the_total_but_not_the_per_mass_ratio(
    mixed,
) -> None:
    # What calibration does across sources: more weight on one side.
    transferred = households_of(mixed, ~cps_rows(mixed))
    tilted = reweighted(mixed, np.where(transferred, 1.06, 1.0))

    _signal, anchor = _gates(tilted)

    assert anchor.passed, anchor.failures
    assert anchor.details[
        "transferred_to_source_per_household_mass_ratio"
    ] == pytest.approx(1.0, rel=1e-9)
    assert anchor.details["relative_error"] == pytest.approx(0.03, rel=1e-6)
    assert anchor.details["source_household_mass_share"] == pytest.approx(
        1.0 / 2.06, rel=1e-9
    )


def test_a_frame_whose_transferred_rows_hold_no_mass_passes_both_gates(
    mixed,
) -> None:
    # Selection can leave one source without weight. The other source then
    # carries the whole population, and there is no second half to compare.
    transferred = households_of(mixed, ~cps_rows(mixed))
    source_only = reweighted(mixed, np.where(transferred, 0.0, 2.0))

    signal, anchor = _gates(source_only)

    assert signal.passed, signal.failures
    assert anchor.passed, anchor.failures
    assert anchor.details["transferred_household_mass_share"] == 0.0
    assert anchor.details["anchor_universe_employer_total"] == pytest.approx(
        ANCHOR, rel=1e-9
    )
    again, receipt = with_us_esi_premium_pool_anchor(source_only)
    assert again is source_only
    assert receipt["status"] == "no_transferred_mass"


@pytest.mark.parametrize("lane", [True, False])
def test_cps_records_that_lost_their_raw_fields_fail_both_gates(
    mixed, alone, lane: bool
) -> None:
    for frame in (
        # A mixed frame, and a single-source frame: neither may read lost
        # coverage codes as a transfer.
        with_person_cells(
            mixed, np.flatnonzero(cps_rows(mixed))[:4], list(RAW), np.nan
        ),
        with_person_cells(alone, np.arange(4), list(RAW), np.nan),
    ):
        for gate in (
            us_esi_premiums_signal_gate(frame, allow_transferred_rows=lane),
            us_esi_premiums_anchor_gate(
                frame, time_period=YEAR, allow_transferred_rows=lane
            ),
        ):
            assert not gate.passed
            if lane:
                assert "4 CPS record(s)" in gate.failures[0]
                assert "lack the raw ASEC fields" in gate.failures[0]
            else:
                assert "lack some of the raw ASEC fields" in gate.failures[0]


def test_transferred_support_clones_that_disagree_fail_the_signal_gate() -> None:
    cloned = mixed_frame(acs_clones=True)
    person = cloned.table("person")
    clone = np.flatnonzero(
        ~cps_rows(cloned)
        & person["person_support_clone_index"].eq(1).to_numpy()
        & (person[EMPLOYER] > 0).to_numpy()
    )[:1]
    split = with_person_cells(cloned, clone, EMPLOYER, 1.0)

    signal, _anchor = _gates(split)

    assert _gates(cloned)[0].passed
    assert signal.details["transferred_clone_disagreement_source_records"] == 1
    assert any(
        "support-clone disagreement(s) on the rows without raw ASEC columns" in failure
        for failure in signal.failures
    )
    # The anchor restores one premium per person.
    restored, receipt = with_us_esi_premium_pool_anchor(split)
    assert receipt["transferred_rows_reset_to_source_record"][EMPLOYER] == 1
    assert _gates(restored)[0].passed


@pytest.mark.parametrize("factor", [1e-12, 1e12])
def test_the_gates_grade_a_tiny_transferred_side_against_its_own_mass(
    mixed, factor: float
) -> None:
    # The per-mass ratio divides each side's total by that side's own mass, so
    # a side many orders smaller than the other still compares at 1.
    transferred = households_of(mixed, ~cps_rows(mixed))
    tilted = reweighted(mixed, np.where(transferred, factor, 1.0))

    summary = us_esi_premiums_summary(tilted, allow_transferred_rows=True)

    assert summary["transferred_household_mass_share"] > 0
    assert summary["transferred_to_source_per_household_mass_ratio"] == pytest.approx(
        1.0, rel=1e-9
    )


def test_a_transferred_side_whose_mass_share_rounds_to_zero_is_still_gated(
    mixed,
) -> None:
    # Household masses about 1e340 apart: the transferred side's share of the
    # frame underflows to zero though that side holds mass, so both gates read
    # whether it is weighed from its absolute mass. A tenth of the premium on
    # it must still fail them.
    rows = ~cps_rows(mixed)
    tilted = reweighted(
        with_scaled_premiums(mixed, 0.1, rows),
        np.where(households_of(mixed, rows), 1e-170, 1e170),
    )

    summary = us_esi_premiums_summary(tilted, allow_transferred_rows=True)
    signal, anchor = _gates(tilted)

    assert summary["transferred_household_mass_share"] == 0.0
    assert summary["transferred_household_mass"] > 0
    assert summary["transferred_to_source_per_household_mass_ratio"] == pytest.approx(
        0.1, rel=1e-9
    )
    assert any(
        "mean per positive person on the rows without raw ASEC columns" in failure
        for failure in signal.failures
    )
    assert any("per unit of household mass" in failure for failure in anchor.failures)


def _cps_policyholder_without_a_job(frame: Frame) -> np.ndarray:
    """One CPS policyholder without a job whose employer pays all: a raw
    MEPS-IC share outside the employed universe."""

    person = frame.table("person")
    candidates = np.flatnonzero(
        cps_rows(frame)
        & person["NOW_OWNGRP"].eq(1).to_numpy()
        & person["NOW_HIPAID"].eq(1).to_numpy()
        & ~person["PEMLR"].isin([1, 2]).to_numpy()
    )
    assert candidates.size
    rows = np.zeros(len(person), dtype=bool)
    rows[candidates[0]] = True
    return rows


def test_cps_rows_whose_employed_policyholders_hold_no_weight_fail_both_gates(
    mixed,
) -> None:
    # Weight stays on one CPS policyholder without a job and on the
    # transferred rows; every employed CPS policyholder holds none. The CPS
    # rows' raw anchor-universe total is then positive and their employed
    # share of it zero, which the transferred rows' total would be divided by.
    cps = cps_rows(mixed)
    kept = _cps_policyholder_without_a_job(mixed)
    starved = reweighted(mixed, np.where(households_of(mixed, cps & ~kept), 0.0, 1.0))

    for gate in _gates(starved):
        assert not gate.passed
        assert gate.failures == (gate.details["structural_error"],)
        assert (
            "give their employed policyholders no weighted MEPS-IC employer share"
            in gate.failures[0]
        )
        assert "hold no household mass" not in gate.failures[0]


def test_cps_rows_without_weight_fail_both_gates(mixed) -> None:
    # Selection can leave the CPS side without weight. The transferred rows
    # then have nothing to be priced against, and the CPS rows' raw mean has
    # no weight to divide by.
    cps = households_of(mixed, cps_rows(mixed))
    transferred_only = reweighted(mixed, np.where(cps, 0.0, 2.0))

    for gate in _gates(transferred_only):
        assert not gate.passed
        assert gate.failures == (gate.details["structural_error"],)
        assert "hold no household mass (0.0)" in gate.failures[0]
        assert "employed policyholders no weighted" in gate.failures[0]


def test_a_cps_side_whose_mass_share_rounds_to_zero_still_gets_a_verdict(
    mixed,
) -> None:
    # Household masses about 1e340 apart, the CPS side the small one: its
    # share of the frame underflows to zero though it holds mass. The anchor
    # gate still grades the frame, and its private-sector cross-check, which
    # scales the CPS rows' sector total by that share, is recorded without a
    # value instead of dividing by zero.
    cps = cps_rows(mixed)
    tilted = reweighted(mixed, np.where(households_of(mixed, cps), 1e-170, 1e170))

    summary = us_esi_premiums_summary(tilted, allow_transferred_rows=True)
    _signal, anchor = _gates(tilted)

    assert summary["source_household_mass_share"] == 0.0
    assert summary["source_household_mass"] > 0
    assert anchor.details[
        "transferred_to_source_per_household_mass_ratio"
    ] == pytest.approx(1.0, rel=1e-9)
    cross_check = anchor.details["private_active_cross_check"]
    assert np.isnan(cross_check["employer_premium_private_sector"])
    assert np.isnan(cross_check["ratio"])
    assert "share of household mass (0.0)" in cross_check["unavailable"]
    # Nearly all the mass sits on the transferred rows at 1e170 times their
    # weight, so the total is far off the anchor: a red verdict, not a crash.
    assert not anchor.passed
    assert any("from BEA NIPA 7.8" in failure for failure in anchor.failures)


@pytest.mark.parametrize(
    ("private", "share", "named", "not_named"),
    [
        # A share of household mass that scales normally: the sector total is
        # what has no value, and the share is not blamed.
        (
            float("nan"),
            None,
            "private-sector employer total (nan) is not finite",
            "share of household mass",
        ),
        (
            float("inf"),
            None,
            "private-sector employer total (inf) is not finite",
            "share of household mass",
        ),
        # Both finite, but the quotient overflows.
        (
            1e10,
            5e-324,
            "total (10000000000.0) over their share of household mass (5e-324) "
            "is not finite",
            "is not a positive number",
        ),
    ],
)
def test_the_cross_check_names_the_term_without_a_finite_value(
    mixed, monkeypatch, private, share, named, not_named
) -> None:
    # The private-sector cross-check is recorded, not gated. When it has no
    # finite value its reason names the term at fault, which is not always
    # the CPS rows' share of household mass.
    real = esi.us_esi_premiums_summary

    def summary(frame, **kwargs):
        result = dict(real(frame, **kwargs))
        result["employer_premium_by_sector"] = dict(
            result["employer_premium_by_sector"], private=private
        )
        if share is not None:
            result["source_household_mass_share"] = share
        return result

    monkeypatch.setattr(esi, "us_esi_premiums_summary", summary)
    anchor = us_esi_premiums_anchor_gate(
        mixed, time_period=YEAR, allow_transferred_rows=True
    )

    cross_check = anchor.details["private_active_cross_check"]
    assert np.isnan(cross_check["employer_premium_private_sector"])
    assert np.isnan(cross_check["ratio"])
    assert named in cross_check["unavailable"]
    assert not_named not in cross_check["unavailable"]


def _cps_households_tilted(frame: Frame, small: np.ndarray, large: np.ndarray) -> Frame:
    """CPS households of ``small`` rows at 1e-200 times their weight and of
    ``large`` rows at 1e200 times it; transferred households unchanged."""

    return reweighted(
        frame,
        np.where(
            households_of(frame, small),
            1e-200,
            np.where(households_of(frame, large), 1e200, 1.0),
        ),
    )


def test_a_cps_side_whose_premium_per_mass_rounds_to_zero_still_gets_a_verdict(
    mixed,
) -> None:
    # CPS policyholders at 1e-200 times their weight, the rest of the CPS
    # side at 1e200: the CPS rows' premium per unit of household mass rounds
    # to zero though both its terms are positive. Their own evidence is
    # intact (the policyholders compare among themselves), so the frame is
    # graded, and the per-mass ratio, which divides by that rate, has no value
    # and fails the anchor gate instead of dividing by zero.
    cps = cps_rows(mixed)
    holder = mixed.table("person")["NOW_OWNGRP"].eq(1).to_numpy()
    tilted = _cps_households_tilted(mixed, cps & holder, cps & ~holder)

    summary = us_esi_premiums_summary(tilted, allow_transferred_rows=True)
    signal, anchor = _gates(tilted)

    assert summary["source_employer_premium_total"] > 0
    assert summary["source_household_mass"] > 0
    assert np.isnan(summary["transferred_to_source_per_household_mass_ratio"])
    for gate in (signal, anchor):
        assert not gate.passed
        assert "structural_error" not in gate.details
    assert any("per unit of household mass" in failure for failure in anchor.failures)


def test_cps_rows_whose_employed_share_rounds_to_zero_fail_both_gates(
    mixed,
) -> None:
    # CPS premium holders at 1e-200 times their weight, every other CPS
    # person, policyholders without a job among them, at 1e200: the employed
    # policyholders' part of the CPS rows' weighted MEPS-IC share is positive
    # but rounds to zero, and the transferred rows' total would be divided by
    # it. The frame is refused, not crashed.
    cps = cps_rows(mixed)
    held = cps & (mixed.table("person")[EMPLOYER].to_numpy(dtype=float) > 0)
    tilted = _cps_households_tilted(mixed, held, cps & ~held)

    for gate in _gates(tilted):
        assert not gate.passed
        assert gate.failures == (gate.details["structural_error"],)
        assert "that rounds to zero" in gate.failures[0]
        assert "no weighted" not in gate.failures[0]
        assert "no household mass" not in gate.failures[0]


def test_a_row_with_only_some_raw_columns_fails_both_gates(mixed) -> None:
    transferred = np.flatnonzero(~cps_rows(mixed))
    broken = with_person_cells(mixed, transferred[:1], RAW[0], 2)

    for gate in _gates(broken):
        assert not gate.passed
        assert "only some of the raw ASEC fields" in gate.failures[0]


@pytest.mark.parametrize("lane", [False, True])
def test_a_single_source_frame_gets_no_mixed_frame_keys(alone, lane: bool) -> None:
    summary = us_esi_premiums_summary(alone, allow_transferred_rows=lane)
    anchor = us_esi_premiums_anchor_gate(
        alone, time_period=YEAR, allow_transferred_rows=lane
    )

    assert "transferred_rows" not in summary
    assert "source_household_mass_share" not in anchor.details
    assert anchor.passed, anchor.failures
