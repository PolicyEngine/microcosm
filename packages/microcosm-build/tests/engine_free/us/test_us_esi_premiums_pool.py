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
* **One premium per person**: every support clone of a transferred person
  carries its clone-0 source record's draw, whether the pool transferred
  before or after it cloned.
* **Structural zeros**: a transferred person whose source record reports zero
  wages carries no premium, on every support clone. A missing wage is
  refused, never read as zero.
* **Proportionality**: every other transferred employer premium is one common
  multiple of the value the transfer drew for the source record.
* **Idempotence**: a pool already on the stage's scale is returned unchanged.
* **Weight homogeneity**: rescaling every household weight leaves each value
  where it was.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime import esi_premiums as esi
from microcosm.build.us_runtime import multispine_pool as pool_module
from microcosm.build.us_runtime.acs_transfer import transfer_acs_inputs
from microcosm.build.us_runtime.esi_premiums import (
    US_ESI_EMPLOYER_PREMIUM_COLUMN,
    US_ESI_PREMIUMS_OUTPUT_COLUMNS,
    US_ESI_PREMIUMS_WAGE_COLUMN,
    us_esi_premiums_anchor_gate,
    us_esi_premiums_household_mass_share,
    us_esi_premiums_signal_gate,
    us_esi_premiums_summary,
    with_us_esi_premium_inputs,
    with_us_esi_premium_pool_anchor,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.build.us_runtime.stacked_spine import (
    GapFillDirection,
    assemble_stacked_spine,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_stacked_spine import (
    _gap_fill_with_test_authority,
)
from test_support.microcosm_build.us_stacked_spine import (
    _source_frame as _stacked_source_frame,
)
from test_support.paths import paths_for

EMPLOYER = US_ESI_EMPLOYER_PREMIUM_COLUMN
#: CMS NHE Table 24 employer contribution, CY2024: the stage's anchor.
ANCHOR = float(esi.EMPLOYER_PREMIUM_ANCHOR["values"]["2024"])
WAGES = US_ESI_PREMIUMS_WAGE_COLUMN
RAW = esi._RAW_EVIDENCE_COLUMNS
CPS_ID = esi._CPS_EVIDENCE_COLUMN
_GROUPS = ("tax_unit", "spm_unit", "family", "marital_unit")
_REPOSITORY_ROOT = paths_for("microcosm-build").repository


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


def _transferred_mass(frame: Frame) -> float:
    return 1.0 - us_esi_premiums_household_mass_share(frame, _source_rows(frame))


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
    assume(share < 1.0)

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
    frame = _pool(_two_sided() + [_two_sided()[1]], clone_of={2: 1})
    frame = _with_person_cell(frame, 1, WAGES, np.nan)

    with pytest.raises(SourceRuntimeError, match="2 transferred row"):
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


# --- The stage, the cross-source fill and the anchor on one stacked pool ------

_STATES = (6, 36, 48, 12, 17, 39, 53, 13)
_ESI_FAMILY = "source_operator_esi_premiums"


def _replace_tables(frame: Frame, **tables: pd.DataFrame) -> Frame:
    return Frame(
        {entity: tables.get(entity, frame.table(entity)) for entity in frame.entities},
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )


def _asec_arm(households: int = 320) -> Frame:
    """One-person ASEC households: a mix inside the signal-gate bands.

    Of every 20 people, 7 are employed policyholders, 2 are policyholders
    without a job (retirees), 5 are employed without a policy and 6 are
    neither.
    """

    frame = _stacked_source_frame(
        household_ids=list(range(1, households + 1)),
        weights=[900.0 + 37.0 * (index % 11) for index in range(households)],
        stratum="asec_2024",
    )
    index = np.arange(households)
    kind = index % 20
    holder = kind < 9
    employed = (kind < 7) | ((kind >= 9) & (kind < 14))
    tier = np.where(holder, np.asarray([1, 2, 3])[index % 3], 0)
    wages = np.where(employed, 28_000.0 + 850.0 * (index % 97), 0.0)
    person = frame.table("person").copy()
    person["PERIDNUM"] = [f"asec-{position:05d}" for position in index]
    person["source_year"] = 2024
    person["source_household_id"] = person["person_household_id"].to_numpy()
    person["source_person_id"] = 1
    person["age"] = np.where(employed, 24.0 + index % 40, 66.0 + index % 20)
    person["is_female"] = index % 2 == 0
    person[WAGES] = wages
    person["NOW_OWNGRP"] = np.where(holder, 1, 2)
    person["NOW_HIPAID"] = np.where(holder, np.asarray([1, 2, 2, 3])[index % 4], 0)
    person["NOW_GRPFTYP2"] = tier
    person["NOW_GRPFTYP"] = np.asarray([0, 1, 1, 2])[tier]
    person["PEMLR"] = np.where(employed, 1, 5)
    person["NOEMP"] = np.where(employed, np.asarray([1, 3, 6, 0])[index % 4], 0)
    person["PEIO1COW"] = np.where(
        employed, np.asarray([4, 4, 5, 2, 3, 1])[index % 6], 0
    )
    household = frame.table("household").copy()
    household["state_fips"] = np.asarray(_STATES, dtype=np.int64)[index % len(_STATES)]
    return _replace_tables(frame, person=person, household=household)


def _acs_arm(households: int = 280) -> Frame:
    """One-person ACS households: native age, sex, State and wages only."""

    frame = _stacked_source_frame(
        household_ids=list(range(10_001, 10_001 + households)),
        weights=[40.0 + 3.0 * (index % 13) for index in range(households)],
        extra_household_columns={"TYPEHUGQ": 1},
        stratum="acs_2024_1yr",
    )
    index = np.arange(households)
    employed = index % 5 < 3
    person = frame.table("person").copy()
    person["age"] = np.where(employed, 23.0 + index % 42, 67.0 + index % 18)
    person["is_female"] = index % 2 == 1
    person[WAGES] = np.where(employed, 26_000.0 + 1_100.0 * (index % 83), 0.0)
    household = frame.table("household").copy()
    household["state_fips"] = np.asarray(_STATES, dtype=np.int64)[
        (index + 3) % len(_STATES)
    ]
    return _replace_tables(frame, person=person, household=household)


def _stage_on_cps_rows(pool: Frame) -> tuple[Frame, dict]:
    """Run the pool's ESI operator alone through the guarded pre-clone runner."""

    staged = pool_module._run_source_operator_chain(
        pool,
        phase="pre_clone",
        operator_names=("with_us_esi_premium_inputs",),
        operators={
            "with_us_esi_premium_inputs": lambda available: (
                pool_module._with_pool_us_esi_premium_inputs(available, pool=pool)
            )
        },
    )
    return staged.frame, staged.receipt["suboperators"][0]["kernel_receipt"]


def _esi_gap_fill_plan() -> tuple[GapFillDirection, ...]:
    return (
        GapFillDirection(
            name="asec_survey_to_acs",
            recipient_channel="acs",
            donor_channel="asec",
            target_families={
                "person": {_ESI_FAMILY: tuple(sorted(US_ESI_PREMIUMS_OUTPUT_COLUMNS))}
            },
        ),
    )


@pytest.fixture(scope="module")
def stacked_pool() -> dict[str, object]:
    asec, acs = _asec_arm(), _acs_arm()
    assembled = assemble_stacked_spine(
        asec, acs, acs_sample_fraction=1.0, acs_sample_seed=578
    ).frame
    staged, stage_receipt = _stage_on_cps_rows(assembled)
    filled = _gap_fill_with_test_authority(
        staged, plan=_esi_gap_fill_plan(), seed=0, n_estimators=20
    ).frame
    anchored, receipt = with_us_esi_premium_pool_anchor(filled)
    return {
        "asec": asec,
        "assembled": assembled,
        "staged": staged,
        "stage_receipt": stage_receipt,
        "filled": filled,
        "anchored": anchored,
        "receipt": receipt,
    }


def test_the_pool_runs_the_stage_on_cps_rows_at_their_household_mass_share(
    stacked_pool,
) -> None:
    staged = stacked_pool["staged"].table("person")
    cps = staged["PERIDNUM"].notna().to_numpy()

    assert stacked_pool["stage_receipt"] == {"anchor_share": 0.5}
    for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS:
        assert staged.loc[cps, column].notna().all()
        assert staged.loc[~cps, column].isna().all()
        assert staged.loc[cps, column].nunique() > 1


def test_pool_cps_rows_carry_the_dollars_of_a_single_source_build(stacked_pool) -> None:
    # Differential: the same stage on the ASEC arm alone, at its own weights.
    alone = with_us_esi_premium_inputs(
        stacked_pool["asec"], seed=pool_module.POOL_RANDOM_SEED, time_period=2024
    ).table("person")
    pooled = stacked_pool["anchored"].table("person")
    pooled = pooled.loc[pooled["PERIDNUM"].notna()].set_index("PERIDNUM")

    for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS:
        np.testing.assert_allclose(
            pooled.loc[alone["PERIDNUM"], column].to_numpy(),
            alone[column].to_numpy(),
            rtol=1e-12,
        )


def test_pool_wide_employer_total_is_the_single_source_total(stacked_pool) -> None:
    alone = with_us_esi_premium_inputs(
        stacked_pool["asec"], seed=pool_module.POOL_RANDOM_SEED, time_period=2024
    )
    anchored = stacked_pool["anchored"]
    cps = _source_rows(anchored)

    assert _total(anchored) == pytest.approx(_total(alone), rel=1e-9)
    assert _total(anchored, cps) == pytest.approx(0.5 * _total(alone), rel=1e-9)
    assert _total(anchored, ~cps) == pytest.approx(0.5 * _total(alone), rel=1e-9)
    assert stacked_pool["receipt"]["status"] == "scaled"
    assert stacked_pool["receipt"]["source_household_mass_share"] == 0.5


def test_acs_rows_carry_signal_and_no_premium_without_wages(stacked_pool) -> None:
    person = stacked_pool["anchored"].table("person")
    acs = person["PERIDNUM"].isna().to_numpy()
    workers = (person[WAGES] > 0).to_numpy()

    assert acs.sum() == 280
    for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS:
        values = person[column].to_numpy(dtype=float)
        assert np.isfinite(values).all() and (values >= 0).all()
        assert (values[acs & workers] > 0).any()
        assert np.unique(values[acs]).size > 2
        assert not values[acs & ~workers].any()
    # The fill conditions on measured wages, so few draws needed clearing and
    # the transferred rows sit near the source rows before the anchor scale.
    receipt = stacked_pool["receipt"]
    assert receipt["cleared_rows"][EMPLOYER] <= 0.1 * acs.sum()
    assert 0.5 <= receipt["scale_factor"] <= 2.0


def test_the_pool_keeps_its_total_and_clone_agreement_through_the_puf_clone(
    stacked_pool,
) -> None:
    # The pipeline clones after the fill and anchors after the clone. Clone a
    # seeded half of the households so both clone pairs and single lineages
    # are present.
    cloned = clone_us_frame_for_puf_support(
        stacked_pool["filled"],
        clone_attachment_fraction=0.5,
        clone_attachment_seed=7,
    )
    anchored, receipt = with_us_esi_premium_pool_anchor(cloned)
    person = anchored.table("person")

    assert set(person["person_support_clone_index"]) == {0, 1}
    assert _total(anchored) == pytest.approx(_total(stacked_pool["anchored"]), rel=1e-9)
    assert receipt["source_household_mass_share"] == pytest.approx(0.5, rel=1e-12)
    assert receipt["scale_factor"] == pytest.approx(
        stacked_pool["receipt"]["scale_factor"], rel=1e-9
    )
    per_source = person.groupby("person_source_id")[
        list(US_ESI_PREMIUMS_OUTPUT_COLUMNS)
    ].nunique()
    assert (per_source == 1).all().all()


@given(
    share=st.floats(min_value=0.1, max_value=0.9),
    asec_scale=st.floats(min_value=0.01, max_value=100.0),
    acs_scale=st.floats(min_value=0.01, max_value=100.0),
)
@settings(max_examples=12, deadline=None)
def test_any_mass_split_keeps_the_pool_on_the_single_source_total(
    share: float, asec_scale: float, acs_scale: float
) -> None:
    def scaled(frame: Frame, factor: float) -> Frame:
        return Frame(
            {entity: frame.table(entity) for entity in frame.entities},
            frame.schema,
            {
                "household": Weights(
                    values=np.asarray(frame.weights_for("household").values) * factor,
                    kind=WeightKind.DESIGN,
                )
            },
            frame.strata,
        )

    asec = scaled(_asec_arm(120), asec_scale)
    pool = assemble_spines(
        {"asec": asec, "acs": scaled(_acs_arm(90), acs_scale)},
        household_mass_shares={"asec": share, "acs": 1.0 - share},
    )
    staged, stage_receipt = _stage_on_cps_rows(pool)
    cps = staged.table("person")["PERIDNUM"].notna().to_numpy()
    filled = transfer_acs_inputs(
        staged,
        staged.select(cps),
        target_families=_esi_gap_fill_plan()[0].target_families,
        donor_channel=None,
        seed=0,
        n_estimators=5,
    ).frame

    anchored, receipt = with_us_esi_premium_pool_anchor(filled)

    alone = with_us_esi_premium_inputs(asec, seed=0, time_period=2024)
    assert stage_receipt["anchor_share"] == pytest.approx(share, rel=1e-9)
    assert _total(anchored) == pytest.approx(_total(alone), rel=1e-9)
    assert _total(anchored, cps) == pytest.approx(share * _total(alone), rel=1e-9)
    assert receipt["source_household_mass_share"] == pytest.approx(share, rel=1e-9)
    # The anchor counts every policyholder. Held to the pool, that is the
    # column over the employed share of the anchor universe.
    summary = us_esi_premiums_summary(anchored)
    assert summary["anchor_universe_employer_total"] == pytest.approx(ANCHOR, rel=1e-9)
    assert summary["transferred_to_source_per_household_mass_ratio"] == pytest.approx(
        1.0, rel=1e-9
    )


# --- The stage at a share of the anchor ---------------------------------------


def _reweighted(frame: Frame, factors: np.ndarray | float) -> Frame:
    return Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {
            "household": Weights(
                values=np.asarray(frame.weights_for("household").values) * factors,
                kind=frame.weights_for("household").kind,
            )
        },
        frame.strata,
        metadata=frame.metadata,
    )


@given(share=st.floats(min_value=0.02, max_value=1.0))
@settings(max_examples=25, deadline=None)
def test_a_share_of_the_weights_at_that_share_of_the_anchor_moves_no_value(
    share: float,
) -> None:
    # Weight homogeneity across the pool boundary: rows at ``share`` of their
    # weight, held to ``share`` of the anchor, carry single-source dollars.
    asec = _asec_arm(80)
    alone = with_us_esi_premium_inputs(asec, seed=0, time_period=2024)

    shared = with_us_esi_premium_inputs(
        _reweighted(asec, share), seed=0, time_period=2024, anchor_share=share
    )

    np.testing.assert_allclose(
        _column(shared, EMPLOYER), _column(alone, EMPLOYER), rtol=1e-12
    )
    summary = us_esi_premiums_summary(shared)
    assert summary["anchor_universe_employer_total"] == pytest.approx(
        share * ANCHOR, rel=1e-9
    )


def test_the_anchor_share_defaults_to_the_whole_anchor() -> None:
    asec = _asec_arm(80)

    default = with_us_esi_premium_inputs(asec, seed=0, time_period=2024)
    explicit = with_us_esi_premium_inputs(
        asec, seed=0, time_period=2024, anchor_share=1.0
    )

    np.testing.assert_array_equal(
        _column(default, EMPLOYER), _column(explicit, EMPLOYER)
    )
    assert us_esi_premiums_summary(default)[
        "anchor_universe_employer_total"
    ] == pytest.approx(ANCHOR, rel=1e-12)


@pytest.mark.parametrize("share", [0.0, -0.5, 1.0000001, np.nan, np.inf, True, "0.5"])
def test_an_anchor_share_outside_zero_to_one_is_refused(share: object) -> None:
    with pytest.raises(SourceRuntimeError, match="anchor share must be"):
        with_us_esi_premium_inputs(
            _asec_arm(40), seed=0, time_period=2024, anchor_share=share
        )


# --- The gates on a pooled frame ----------------------------------------------


def test_both_gates_pass_on_the_stacked_pool(stacked_pool) -> None:
    anchored = stacked_pool["anchored"]

    signal = us_esi_premiums_signal_gate(anchored)
    anchor = us_esi_premiums_anchor_gate(anchored, time_period=2024)

    assert signal.passed, signal.failures
    assert anchor.passed, anchor.failures
    # The cell proofs ran on the CPS-source rows, and found them one common
    # multiple of their MEPS-IC shares: the single-source factor.
    alone = us_esi_premiums_summary(
        with_us_esi_premium_inputs(stacked_pool["asec"], seed=0, time_period=2024)
    )
    assert signal.details["source_rows"] == 320
    assert signal.details["transferred_rows"] == 280
    assert signal.details["scale_factor"] == pytest.approx(
        alone["scale_factor"], rel=1e-9
    )
    assert signal.details["employer_premium_outside_universe_rows"] == 0
    assert signal.details["clone_disagreement_source_persons"] == 0
    assert signal.details["transferred_source_records_with_premium_and_no_wages"] == 0
    assert anchor.details["anchor_universe_employer_total"] == pytest.approx(
        ANCHOR, rel=1e-9
    )
    assert anchor.details["relative_error"] == pytest.approx(0.0, abs=1e-9)
    assert anchor.details["source_household_mass_share"] == 0.5
    assert anchor.details[
        "transferred_to_source_per_household_mass_ratio"
    ] == pytest.approx(1.0, rel=1e-9)
    assert anchor.details["employed_share_of_anchor_universe"] == pytest.approx(
        alone["employed_share_of_anchor_universe"], rel=1e-12
    )
    assert "employed share" in anchor.details["anchor_universe_assumption"]
    # Scaled to the population, the private-sector cross-check matches the
    # single-source one.
    assert anchor.details["private_active_cross_check"][
        "employer_premium_private_sector"
    ] == pytest.approx(alone["employer_premium_by_sector"]["private"], rel=1e-9)


def test_both_gates_pass_on_the_pool_after_the_puf_clone(stacked_pool) -> None:
    cloned = clone_us_frame_for_puf_support(stacked_pool["filled"])
    anchored, _receipt = with_us_esi_premium_pool_anchor(cloned)

    signal = us_esi_premiums_signal_gate(anchored)
    anchor = us_esi_premiums_anchor_gate(anchored, time_period=2024)

    assert signal.passed, signal.failures
    assert anchor.passed, anchor.failures
    assert signal.details["clone_disagreement_source_persons"] == 0
    assert anchor.details["anchor_universe_employer_total"] == pytest.approx(
        ANCHOR, rel=1e-9
    )


def _with_transferred_scaled(frame: Frame, factor: float) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    transferred = ~_source_rows(frame)
    tables["person"].loc[transferred, EMPLOYER] *= factor
    return Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
        metadata=frame.metadata,
    )


def test_transferred_rows_without_the_premium_fail_the_signal_gate(
    stacked_pool,
) -> None:
    flattened = _with_transferred_scaled(stacked_pool["anchored"], 0.0)

    signal = us_esi_premiums_signal_gate(flattened)

    assert not signal.passed
    assert any(
        "positive share on the rows without raw ASEC columns" in failure
        for failure in signal.failures
    )


@pytest.mark.parametrize("factor", [0.7, 1.4])
def test_transferred_rows_off_the_source_rows_per_mass_total_fail_the_anchor_gate(
    stacked_pool, factor: float
) -> None:
    skewed = _with_transferred_scaled(stacked_pool["anchored"], factor)

    anchor = us_esi_premiums_anchor_gate(skewed, time_period=2024)

    assert not anchor.passed
    assert anchor.details[
        "transferred_to_source_per_household_mass_ratio"
    ] == pytest.approx(factor, rel=1e-9)
    assert any("per unit of household mass" in failure for failure in anchor.failures)
    # Half the mass at ``factor`` moves the anchor-universe total by half of it.
    assert anchor.details["relative_error"] == pytest.approx(
        (factor - 1.0) / 2.0, rel=1e-6
    )


def test_reweighting_one_side_moves_the_pool_total_but_not_the_per_mass_ratio(
    stacked_pool,
) -> None:
    # What calibration does across sources: more weight on one side.
    anchored = stacked_pool["anchored"]
    household = anchored.table("household")
    transferred_households = household["household_id"].isin(
        anchored.table("person").loc[~_source_rows(anchored), "person_household_id"]
    )
    reweighted = _reweighted(anchored, np.where(transferred_households, 1.06, 1.0))

    anchor = us_esi_premiums_anchor_gate(reweighted, time_period=2024)

    assert anchor.passed, anchor.failures
    assert anchor.details[
        "transferred_to_source_per_household_mass_ratio"
    ] == pytest.approx(1.0, rel=1e-9)
    assert anchor.details["relative_error"] == pytest.approx(0.03, rel=1e-6)
    assert anchor.details["source_household_mass_share"] == pytest.approx(
        1.0 / 2.06, rel=1e-9
    )


def _with_person_rows(frame: Frame, rows: np.ndarray, columns, value) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    person = tables["person"]
    person.loc[person.index[rows], columns] = value
    return Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
        metadata=frame.metadata,
    )


def test_a_pool_whose_transferred_rows_hold_no_mass_passes_both_gates(
    stacked_pool,
) -> None:
    # Selection can leave one source without weight. The other source then
    # carries the whole population, and there is no second half to compare.
    anchored = stacked_pool["anchored"]
    household = anchored.table("household")
    transferred_households = household["household_id"].isin(
        anchored.table("person").loc[~_source_rows(anchored), "person_household_id"]
    )
    source_only = _reweighted(anchored, np.where(transferred_households, 0.0, 2.0))

    signal = us_esi_premiums_signal_gate(source_only)
    anchor = us_esi_premiums_anchor_gate(source_only, time_period=2024)

    assert signal.passed, signal.failures
    assert anchor.passed, anchor.failures
    assert anchor.details["transferred_household_mass_share"] == 0.0
    assert anchor.details["anchor_universe_employer_total"] == pytest.approx(
        ANCHOR, rel=1e-9
    )
    again, receipt = with_us_esi_premium_pool_anchor(source_only)
    assert again is source_only
    assert receipt["status"] == "no_transferred_mass"


def test_cps_records_that_lost_their_raw_fields_fail_both_gates(stacked_pool) -> None:
    anchored = stacked_pool["anchored"]
    lost = np.flatnonzero(_source_rows(anchored))[:4]

    for frame in (
        # A pool, and a single-source frame: neither may read lost coverage
        # codes as a transfer.
        _with_person_rows(anchored, lost, list(RAW), np.nan),
        _with_person_rows(
            with_us_esi_premium_inputs(stacked_pool["asec"], seed=0, time_period=2024),
            np.arange(4),
            list(RAW),
            np.nan,
        ),
    ):
        for gate in (
            us_esi_premiums_signal_gate(frame),
            us_esi_premiums_anchor_gate(frame, time_period=2024),
        ):
            assert not gate.passed
            assert "4 CPS record(s)" in gate.failures[0]
            assert "lack the raw ASEC fields" in gate.failures[0]


def test_transferred_support_clones_that_disagree_fail_the_signal_gate(
    stacked_pool,
) -> None:
    cloned = clone_us_frame_for_puf_support(stacked_pool["filled"])
    anchored, _receipt = with_us_esi_premium_pool_anchor(cloned)
    person = anchored.table("person")
    clone = np.flatnonzero(
        ~_source_rows(anchored)
        & person["person_support_clone_index"].eq(1).to_numpy()
        & (person[EMPLOYER] > 0).to_numpy()
    )[:1]
    split = _with_person_rows(anchored, clone, EMPLOYER, 1.0)

    signal = us_esi_premiums_signal_gate(split)

    assert us_esi_premiums_signal_gate(anchored).passed
    assert signal.details["transferred_clone_disagreement_source_records"] == 1
    assert any(
        "support-clone disagreement(s) on the rows without raw ASEC columns" in failure
        for failure in signal.failures
    )
    # The anchor restores one premium per person.
    restored, receipt = with_us_esi_premium_pool_anchor(split)
    assert receipt["transferred_rows_reset_to_source_record"][EMPLOYER] == 1
    assert us_esi_premiums_signal_gate(restored).passed


def test_a_row_with_only_some_raw_columns_fails_both_gates(stacked_pool) -> None:
    anchored = stacked_pool["anchored"]
    tables = {entity: anchored.table(entity).copy() for entity in anchored.entities}
    transferred = np.flatnonzero(~_source_rows(anchored))
    tables["person"].loc[tables["person"].index[transferred[0]], RAW[0]] = 2
    broken = Frame(
        tables,
        anchored.schema,
        {"household": anchored.weights_for("household")},
        anchored.strata,
        metadata=anchored.metadata,
    )

    for gate in (
        us_esi_premiums_signal_gate(broken),
        us_esi_premiums_anchor_gate(broken, time_period=2024),
    ):
        assert not gate.passed
        assert "only some of the raw ASEC fields" in gate.failures[0]


def test_a_single_source_frame_gets_no_pool_keys(stacked_pool) -> None:
    alone = with_us_esi_premium_inputs(stacked_pool["asec"], seed=0, time_period=2024)

    summary = us_esi_premiums_summary(alone)
    anchor = us_esi_premiums_anchor_gate(alone, time_period=2024)

    assert "transferred_rows" not in summary
    assert "source_household_mass_share" not in anchor.details
    assert anchor.passed, anchor.failures


# --- the real stacked fill, measured on the pinned inputs -----------------------------


@pytest.mark.parametrize(
    (
        "fraction",
        "rows",
        "incidence",
        "quantile_distance",
        "cleared",
        "factor",
        "means",
        "total",
    ),
    [
        # Figures quoted in docs/us-esi-employer-premiums.md: person rows
        # (ASEC, ACS), ACS-over-ASEC incidence and quantile distance (after the
        # fill, after the anchor), rows cleared, scale factor, mean per holder
        # after the anchor (ASEC, ACS) and the pool column total.
        (
            "f001",
            (4_177, 34_293),
            (0.933, 0.930),
            (0.072, 0.169),
            34,
            1.102,
            (11_741, 12_935),
            947.3e9,
        ),
        (
            "f005",
            (21_133, 171_381),
            (0.975, 0.969),
            (0.047, 0.077),
            255,
            1.047,
            (12_608, 13_208),
            924.7e9,
        ),
    ],
)
def test_the_pool_fill_receipts_reproduce_the_documented_measurements(
    fraction, rows, incidence, quantile_distance, cleared, factor, means, total
) -> None:
    receipt = json.loads(
        (
            _REPOSITORY_ROOT
            / f"experiments/us-esi-454/receipts/pool_fill_2023_2025_{fraction}.json"
        ).read_text()
    )
    target = f"person/source_operator_esi_premiums/{EMPLOYER}"
    anchor = receipt["pool_anchor"]

    assert receipt["target_year"] == 2024
    assert receipt["anchor"] == ANCHOR
    assert receipt["signal_gate"] == {"passed": True, "failures": []}
    assert receipt["anchor_gate"] == {"passed": True, "failures": []}
    assert receipt["gap_fill_targets"][target] == {
        "authorized_null_rows": rows[1],
        "imputed_rows": rows[1],
        "residual_null_rows": 0,
        "unmodeled_rows": 0,
    }
    assert (anchor["source_rows"], anchor["transferred_rows"]) == rows
    assert anchor["source_household_mass_share"] == pytest.approx(0.5)
    # Every ACS wage is a number by the anchor: the universe producer wrote
    # the zeros below age 15 and no eligible record was missing one.
    universe = receipt["acs_wage_universe"]
    assert universe["rule_id"] == "acs_2024_pums_wagp_age_15_plus"
    assert universe["in_universe_null_rows"] == 0
    assert (
        universe["structurally_absent_person_rows"]
        + universe["eligible_acs_person_rows"]
        == rows[1]
    )
    # The stacked order fills before it clones, so no clone needed resetting.
    assert anchor["transferred_rows_reset_to_source_record"] == {EMPLOYER: 0}
    assert anchor["cleared_rows"] == {EMPLOYER: cleared}
    assert anchor["scale_factor"] == pytest.approx(factor, abs=5e-4)
    # Conservation and equal intensity, on the real fill.
    assert anchor["employer_premium_total"] == pytest.approx(
        anchor["source_employer_premium_total"] / anchor["source_household_mass_share"],
        rel=1e-9,
    )
    assert receipt["pool_employer_premium_total"] == pytest.approx(total, rel=1e-4)
    for stage, at in (("before_anchor", 0), ("after_anchor", 1)):
        battery = receipt[stage]["by_origin_battery"]
        assert battery["passed"] and battery["failures"] == []
        leg = battery["comparisons"][f"{target}[clone_0]"]["legs"]["positive"]
        assert leg["incidence_ratio_acs_over_asec"] == pytest.approx(
            incidence[at], abs=5e-4
        )
        assert leg["quantile_envelope_distance"] == pytest.approx(
            quantile_distance[at], abs=5e-4
        )
    sides = receipt["after_anchor"]["sides"][EMPLOYER]
    assert sides["acs"]["positive_rows_without_wages"] == 0
    assert sides["acs"]["weighted_total"] == pytest.approx(
        sides["cps"]["weighted_total"], rel=1e-9
    )
    assert (
        round(sides["cps"]["mean_per_positive_person"]),
        round(sides["acs"]["mean_per_positive_person"]),
    ) == means
