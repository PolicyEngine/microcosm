"""The shared clone expansion core under the UK's declarations (microcosm#931).

With the copy-id table's ``geographic_support`` field, the ladder's
clone-index columns and household design weights, the core must give what the
clone-first stand-in gives (``clone_first_frame`` in
test_support/microcosm_build/uk_clone_first_frames.py, written without the
core): bit for bit on the five id columns, the household clone index, the
source household id and the weights, at K = 1, 2, 3, 7 and 15. Invented
frames only; the refusals print counts and declared constants, never an id.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.clone_expansion import (
    CloneDeclaration,
    CloneIdRule,
    expand_clones,
)
from microcosm.build.uk_runtime import copy_id_table
from microcosm.build.uk_runtime.national_frame import (
    UK_NATIONAL_SCHEMA,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.rowwise_dataset import ladder_clone_index_column
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.concepts import SOURCE_HOUSEHOLD_ID_COLUMN
from test_support.microcosm_build.uk_clone_first_frames import (
    clone_first_frame,
    with_last_household_moved,
)

_FIELD = "geographic_support"
_REASON = "Geographic clones split each household's weight equally; mass is conserved."


def _mint_geographic_clone(ids: np.ndarray, k: int) -> np.ndarray:
    return copy_id_table.mint(ids, _FIELD, k)


def _require_room_for_geographic_clones(ids: np.ndarray, what: str) -> None:
    copy_id_table.require_below(ids, _FIELD, what=what)


UK_GEOGRAPHIC_CLONES = CloneDeclaration(
    ids=CloneIdRule(
        bound=copy_id_table.ordinal_bound(_FIELD),
        mint=_mint_geographic_clone,
        require_below=_require_room_for_geographic_clones,
    ),
    clone_index_columns={
        entity: ladder_clone_index_column(entity)
        for entity in UK_NATIONAL_SCHEMA.entities
    },
    weight_kinds=frozenset({WeightKind.DESIGN}),
    reason=_REASON,
    source_id_columns={"household": SOURCE_HOUSEHOLD_ID_COLUMN},
)
_ID_COLUMNS = {
    "person": ("person_id", "person_household_id", "person_benunit_id"),
    "benunit": ("benunit_id",),
    "household": ("household_id",),
}
#: The stand-in's columns the core must reproduce bit for bit.
_ORACLE_COLUMNS = {
    **_ID_COLUMNS,
    "household": (
        "household_id",
        ladder_clone_index_column("household"),
        SOURCE_HOUSEHOLD_ID_COLUMN,
    ),
}
_CLONE_COUNTS = (1, 2, 3, 7, 15)


def _uk_frame(households: int = 12) -> Frame:
    """Invented FRS-shaped ids: household sernums, benefit unit sernum x 100 +
    b, person sernum x 1000 + p; one to four people in one or two units."""

    people: list[tuple[int, int, int, float]] = []
    units: list[int] = []
    for sernum in range(1, households + 1):
        size = 1 + sernum % 4
        unit_count = min(size, 1 + sernum % 2)
        units.extend(sernum * 100 + b for b in range(1, unit_count + 1))
        people.extend(
            (
                sernum * 1000 + p,
                sernum,
                sernum * 100 + min(p, unit_count),
                875.5 * p + sernum,
            )
            for p in range(1, size + 1)
        )
    person = pd.DataFrame(
        people,
        columns=[
            "person_id",
            "person_household_id",
            "person_benunit_id",
            "employment_income",
        ],
    ).astype(
        {
            "person_id": "int64",
            "person_household_id": "int64",
            "person_benunit_id": "int64",
        }
    )
    sernums = range(1, households + 1)
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, households + 1, dtype=np.int64),
            "region": ["WALES" if s % 3 == 0 else "NORTH_EAST" for s in sernums],
            "household_label": pd.Series([f"2024:{s}" for s in sernums], dtype=object),
        }
    )
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": np.asarray(units, dtype=np.int64)}),
        household=household,
        household_weights=np.random.default_rng(931).uniform(400.0, 3000.0, households),
        time_period="2024",
    )


@pytest.mark.parametrize("clones", _CLONE_COUNTS)
def test_the_core_under_uk_declarations_is_the_clone_first_stand_in(
    clones: int,
) -> None:
    frame = _uk_frame()
    expanded = expand_clones(frame, clones=clones, declaration=UK_GEOGRAPHIC_CLONES)
    stand_in = clone_first_frame(frame, clones)
    for entity, columns in _ORACLE_COLUMNS.items():
        for column in columns:
            got = expanded.table(entity)[column].to_numpy()
            want = stand_in.table(entity)[column].to_numpy()
            assert got.dtype == want.dtype, (entity, column)
            assert got.tobytes() == want.tobytes(), (entity, column)
    got_weights = expanded.weights_for("household")
    want_weights = stand_in.weights_for("household")
    assert got_weights.kind is want_weights.kind is WeightKind.DESIGN
    assert got_weights.values.tobytes() == want_weights.values.tobytes()
    # Every column the stand-in carries, in its order and dtypes.
    for entity in stand_in.entities:
        table = stand_in.table(entity)
        pd.testing.assert_frame_equal(
            expanded.table(entity)[list(table.columns)], table, check_exact=True
        )
    assert uk_time_period(expanded) == uk_time_period(stand_in)


@pytest.mark.parametrize("clones", _CLONE_COUNTS)
def test_every_id_decodes_to_its_clone_and_its_clone_0_row(clones: int) -> None:
    frame = _uk_frame()
    expanded = expand_clones(frame, clones=clones, declaration=UK_GEOGRAPHIC_CLONES)
    validate_uk_national_frame(expanded)
    for entity, columns in _ID_COLUMNS.items():
        table = expanded.table(entity)
        index = table[ladder_clone_index_column(entity)]
        assert index.dtype == np.int64
        assert sorted(set(index.tolist())) == list(range(clones))
        for column in columns:
            ids = table[column].to_numpy()
            # The field's digits are the row's clone, and clearing them gives
            # the clone-0 row's id: the lineage the expand kernel recovers.
            assert np.array_equal(copy_id_table.ordinal(ids, _FIELD), index.to_numpy())
            assert np.array_equal(
                copy_id_table.parent(ids, _FIELD),
                np.tile(frame.table(entity)[column].to_numpy(), clones),
            )
    # Every member carries its household's and its benefit unit's clone.
    person = expanded.table("person")
    for group in UK_NATIONAL_SCHEMA.group_entities:
        units = expanded.table(group).set_index(UK_NATIONAL_SCHEMA.id_column(group))
        assert np.array_equal(
            units.loc[
                person[UK_NATIONAL_SCHEMA.membership_column(group)],
                ladder_clone_index_column(group),
            ],
            person[ladder_clone_index_column("person")],
        )


def test_the_mass_record_is_the_same_at_every_clone_count() -> None:
    frame = _uk_frame()
    records = [
        expand_clones(frame, clones=clones, declaration=UK_GEOGRAPHIC_CLONES)
        for clones in _CLONE_COUNTS
    ]
    assert {
        (expanded.mass_log[-1].entity, expanded.mass_log[-1].declared_factor)
        for expanded in records
    } == {("household", 1.0)}
    assert {expanded.mass_log[-1].reason for expanded in records} == {_REASON}


def test_a_python_string_column_stays_python_strings_in_every_clone() -> None:
    frame = _uk_frame()
    expanded = expand_clones(frame, clones=3, declaration=UK_GEOGRAPHIC_CLONES)
    labels = expanded.table("household")["household_label"]
    assert labels.dtype == object
    assert {type(value) for value in labels.tolist()} == {str}
    assert labels.tolist() == frame.table("household")["household_label"].tolist() * 3


@pytest.mark.parametrize("clones", [0, copy_id_table.ordinal_bound(_FIELD) + 1])
def test_a_clone_count_outside_the_field_s_bound_is_refused(clones: int) -> None:
    with pytest.raises(
        ValueError, match=r"an integer in \[1, 100\], the declared bound"
    ):
        expand_clones(_uk_frame(), clones=clones, declaration=UK_GEOGRAPHIC_CLONES)


def test_an_id_in_the_field_is_refused_by_count() -> None:
    radix = copy_id_table.radix(_FIELD)
    moved = with_last_household_moved(_uk_frame(), radix)
    with pytest.raises(
        ValueError,
        match="1 of the household_id values reach the 'geographic_support' copy-id",
    ) as error:
        expand_clones(moved, clones=2, declaration=UK_GEOGRAPHIC_CLONES)
    moved_id = int(moved.table("household")["household_id"].iloc[-1])
    assert str(moved_id) not in str(error.value)


def test_importance_weights_and_a_second_expansion_are_refused() -> None:
    frame = _uk_frame()
    importance = Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {
            "household": Weights(
                frame.weights_for("household").values, WeightKind.IMPORTANCE
            )
        },
        frame.strata,
        metadata=frame.metadata,
    )
    with pytest.raises(ValueError, match=r"accepts \['design'\] weights"):
        expand_clones(importance, clones=2, declaration=UK_GEOGRAPHIC_CLONES)
    once = expand_clones(frame, clones=2, declaration=UK_GEOGRAPHIC_CLONES)
    with pytest.raises(ValueError, match="already carries the declared column"):
        expand_clones(once, clones=2, declaration=UK_GEOGRAPHIC_CLONES)
