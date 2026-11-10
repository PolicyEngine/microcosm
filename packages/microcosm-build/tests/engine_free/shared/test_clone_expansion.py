"""The shared clone expansion core, on invented frames (microcosm#931).

Neutral and US-shaped declarations only; no country module is imported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build import clone_expansion
from microcosm.build.clone_expansion import (
    CLONE_MASS_RTOL,
    CloneDeclaration,
    CloneIdRule,
    expand_clones,
)
from microcosm.build.outer_stage_runtime import assert_clone_expansion
from microcosm.frame import (
    EntitySchema,
    Frame,
    LinkSpec,
    MassChangeRecord,
    WeightKind,
    Weights,
)

SCHEMA = EntitySchema(group_entities=("unit", "household"))
#: An invented id field: clone ``k``'s ids are the originals' plus ``k`` times it.
RADIX = 100_000
BOUND = 5
REASON = "Clones split each household's weight equally; the mass is conserved."
INDEX_COLUMNS = {
    "person": "person_clone",
    "unit": "unit_clone",
    "household": "household_clone",
}
SOURCE_COLUMNS = {"household": "household_origin"}
DECLARED_COLUMNS = (*INDEX_COLUMNS.values(), *SOURCE_COLUMNS.values())
HOUSEHOLD_WEIGHTS = np.array([312.5, 47.25, 1009.75])
PRIOR_RECORD = MassChangeRecord(
    entity="household",
    old_total=1000.0,
    new_total=1369.5,
    declared_factor=None,
    reason="an earlier stage's declared change",
)


def _mint(ids: np.ndarray, k: int) -> np.ndarray:
    return ids + k * RADIX


def _require_below(ids: np.ndarray, what: str) -> None:
    reached = int((np.asarray(ids) >= RADIX).sum())
    if reached:
        raise ValueError(f"{reached} of the {what} reach the clone field.")


RULE = CloneIdRule(bound=BOUND, mint=_mint, require_below=_require_below)


def _declaration(**changes: object) -> CloneDeclaration:
    fields: dict[str, object] = {
        "ids": RULE,
        "clone_index_columns": INDEX_COLUMNS,
        "weight_kinds": frozenset({WeightKind.DESIGN}),
        "reason": REASON,
        "source_id_columns": SOURCE_COLUMNS,
    }
    fields.update(changes)
    return CloneDeclaration(**fields)  # type: ignore[arg-type]


DECLARATION = _declaration()


def _tables(households: tuple[int, int, int] = (411, 522, 633)) -> dict:
    """Three invented households: the first holds two units and three people.

    The person rows are out of id order, so a mint must keep the order it is
    handed rather than an ascending one.
    """

    first, second, third = households
    person = pd.DataFrame(
        {
            "person_id": np.array([52201, 41101, 63302, 41102, 63301, 41103]),
            "person_unit_id": np.array([5221, 4111, 6331, 4111, 6331, 4112]),
            "person_household_id": np.array(
                [second, first, third, first, third, first]
            ),
            "pay": [1250.5, 0.0, 87.25, 3020.0, 415.75, 12.5],
            "age": np.array([44, 71, 9, 38, 36, 15], dtype=np.int16),
            "role": pd.Series(["a", "b", "c", "b", "a", "c"], dtype=object),
            "works": [True, False, False, True, True, False],
        }
    )
    unit = pd.DataFrame(
        {
            "unit_id": np.array([4111, 4112, 5221, 6331]),
            "unit_kind": pd.Categorical(["couple", "single", "single", "couple"]),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.array([first, second, third]),
            "area": ["north", "south", "north"],
        }
    )
    return {"person": person, "unit": unit, "household": household}


def _frame(
    tables: dict | None = None,
    *,
    kind: WeightKind = WeightKind.DESIGN,
    schema: EntitySchema = SCHEMA,
    extra_weights: dict | None = None,
) -> Frame:
    return Frame(
        _tables() if tables is None else tables,
        schema,
        {"household": Weights(HOUSEHOLD_WEIGHTS, kind), **(extra_weights or {})},
        pd.Series(["s1", "s2", "s1", "s2", "s1", "s2"], dtype=object),
        mass_log=(PRIOR_RECORD,),
        metadata={"period": "2030"},
    )


def _ids_and_weights(frame: Frame) -> set[str]:
    """Every id, membership and weight of ``frame``, as text."""

    texts = {
        repr(float(value))
        for entity in frame.weighted_entities
        for value in frame.weights_for(entity).values
    }
    for entity in frame.entities:
        table = frame.table(entity)
        for column in table.columns:
            if column.endswith("_id"):
                texts.update(str(value) for value in table[column].tolist())
    return texts


def _assert_names_no_value(error: pytest.ExceptionInfo, frame: Frame) -> None:
    message = str(error.value)
    assert not {text for text in _ids_and_weights(frame) if text in message}, message


@pytest.mark.parametrize("clones", [1, 2, 3, BOUND])
def test_each_clone_repeats_the_frame_with_its_ids_minted(clones: int) -> None:
    frame = _frame()
    expanded = expand_clones(frame, clones=clones, declaration=DECLARATION)
    for entity in frame.entities:
        before = frame.table(entity).reset_index(drop=True)
        after = expanded.table(entity)
        rows = len(before)
        assert len(after) == clones * rows
        id_column = SCHEMA.entity_id_column(entity)
        ids = before[id_column].to_numpy()
        assert after[id_column].tolist() == [
            value + k * RADIX for k in range(clones) for value in ids.tolist()
        ]
        index = after[INDEX_COLUMNS[entity]]
        assert index.dtype == np.int64
        assert index.tolist() == [k for k in range(clones) for _ in range(rows)]
        assert sorted(set(index.tolist())) == list(range(clones))
        # Every other column is the frame's, repeated once per clone.
        copied = [c for c in before.columns if c != id_column and "_id" not in c]
        pd.testing.assert_frame_equal(
            after[copied],
            pd.concat([before[copied]] * clones, ignore_index=True),
            check_exact=True,
        )
    person = expanded.table("person")
    original = frame.table("person")
    for group in SCHEMA.group_entities:
        column = SCHEMA.membership_column(group)
        assert person[column].tolist() == [
            value + k * RADIX
            for k in range(clones)
            for value in original[column].tolist()
        ]
        # A member of clone k belongs to its unit's clone k.
        units = expanded.table(group).set_index(SCHEMA.id_column(group))
        assert np.array_equal(
            units.loc[person[column], INDEX_COLUMNS[group]].to_numpy(),
            person[INDEX_COLUMNS["person"]].to_numpy(),
        )
    household = expanded.table("household")
    origin = household[SOURCE_COLUMNS["household"]]
    assert origin.dtype == np.int64
    assert origin.tolist() == [411, 522, 633] * clones
    assert SOURCE_COLUMNS["household"] not in person.columns


def test_a_numpy_integer_clone_count_is_accepted() -> None:
    expanded = expand_clones(_frame(), clones=np.int64(2), declaration=DECLARATION)
    assert expanded.n("household") == 6


@pytest.mark.parametrize("clones", [1, 2, 3, BOUND])
def test_each_row_s_clones_carry_one_equal_share_of_its_weight(clones: int) -> None:
    frame = _frame()
    weights = expand_clones(frame, clones=clones, declaration=DECLARATION).weights_for(
        "household"
    )
    assert weights.kind is WeightKind.DESIGN
    assert (
        weights.values.tobytes()
        == np.tile(HOUSEHOLD_WEIGHTS / clones, clones).tobytes()
    )
    by_clone = weights.values.reshape(clones, len(HOUSEHOLD_WEIGHTS))
    assert all(np.array_equal(row, by_clone[0]) for row in by_clone)
    assert np.isclose(
        weights.total, HOUSEHOLD_WEIGHTS.sum(), rtol=CLONE_MASS_RTOL, atol=0.0
    )


def test_the_mass_record_states_the_declared_reason_at_every_clone_count() -> None:
    frame = _frame()
    before = frame.weights_for("household").total
    records = []
    for clones in range(1, BOUND + 1):
        expanded = expand_clones(frame, clones=clones, declaration=DECLARATION)
        assert expanded.mass_log[:-1] == (PRIOR_RECORD,)
        records.append(expanded.mass_log[-1])
        assert records[-1].new_total == expanded.weights_for("household").total
    assert {record.reason for record in records} == {REASON}
    assert {record.entity for record in records} == {"household"}
    assert {record.declared_factor for record in records} == {1.0}
    assert {record.old_total for record in records} == {before}


def test_one_clone_keeps_the_frame_s_rows_ids_and_weights() -> None:
    frame = _frame()
    expanded = expand_clones(frame, clones=1, declaration=DECLARATION)
    for entity in frame.entities:
        added = [c for c in expanded.table(entity).columns if c in DECLARED_COLUMNS]
        # Appended after the frame's own columns: the source id, then the index.
        assert added == [
            c
            for c in (SOURCE_COLUMNS.get(entity), INDEX_COLUMNS[entity])
            if c is not None
        ]
        assert list(expanded.table(entity).columns[-len(added) :]) == added
        pd.testing.assert_frame_equal(
            expanded.table(entity).drop(columns=added),
            frame.table(entity).reset_index(drop=True),
            check_exact=True,
        )
    assert (
        expanded.weights_for("household").values.tobytes()
        == HOUSEHOLD_WEIGHTS.tobytes()
    )


def test_every_weighted_entity_is_split_with_a_record_of_its_own() -> None:
    person_weights = np.array([2.5, 0.75, 4.0, 1.25, 3.5, 0.5])
    frame = _frame(extra_weights={"person": Weights(person_weights, WeightKind.DESIGN)})
    expanded = expand_clones(frame, clones=3, declaration=DECLARATION)
    split = {"person": person_weights, "household": HOUSEHOLD_WEIGHTS}
    assert expanded.weighted_entities == tuple(split)
    for entity, values in split.items():
        weights = expanded.weights_for(entity)
        assert weights.kind is WeightKind.DESIGN
        assert weights.values.tobytes() == np.tile(values / 3, 3).tobytes()
    # One record per weighted entity, in the frame's entity order.
    assert expanded.mass_log[:-2] == (PRIOR_RECORD,)
    records = expanded.mass_log[-2:]
    assert [record.entity for record in records] == list(split)
    for record in records:
        before = Weights(split[record.entity], WeightKind.DESIGN)
        assert record.old_total == before.total
        assert record.new_total == expanded.weights_for(record.entity).total
        assert (record.declared_factor, record.reason) == (1.0, REASON)


def test_the_strata_repeat_and_the_metadata_is_kept() -> None:
    frame = _frame()
    expanded = expand_clones(frame, clones=3, declaration=DECLARATION)
    assert expanded.strata.tolist() == frame.strata.tolist() * 3
    assert dict(expanded.metadata) == dict(frame.metadata)


def test_the_frame_handed_in_is_left_as_it_was() -> None:
    frame = _frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    expand_clones(frame, clones=BOUND, declaration=DECLARATION)
    for entity, table in tables.items():
        pd.testing.assert_frame_equal(frame.table(entity), table, check_exact=True)
    assert (
        frame.weights_for("household").values.tobytes() == HOUSEHOLD_WEIGHTS.tobytes()
    )
    assert frame.mass_log == (PRIOR_RECORD,)


# ---------------------------------------------------------------------------
# US-shaped declarations: the shared two-channel invariant accepts the core's
# clones once the channel labels are written from the clone index.
# ---------------------------------------------------------------------------

US_SHAPED_SCHEMA = EntitySchema(
    group_entities=("household", "tax_unit", "spm_unit", "family", "marital_unit")
)
CHANNELS = ("asec", "puf")
US_SHAPED_MULTIPLIER = 1_000_000
US_SHAPED_HOUSEHOLD_WEIGHTS = np.array([4.0, 6.5, 2.25])
US_SHAPED_PERSON_WEIGHTS = np.array([3.5, 4.5, 6.25, 2.0, 2.5])


def _us_shaped_frame() -> Frame:
    groups = US_SHAPED_SCHEMA.group_entities
    person = pd.DataFrame(
        {
            "person_id": np.array([101, 102, 201, 301, 302]),
            **{
                US_SHAPED_SCHEMA.membership_column(group): np.array(
                    [1, 1, 2, 3, 3] if group == "household" else [11, 11, 21, 31, 32]
                )
                for group in groups
            },
            "source_year": np.array([2023, 2023, 2024, 2024, 2024], dtype=np.int16),
            "source_household_id": np.array([7, 7, 9, 12, 12]),
            "source_person_id": pd.Series(["a", "b", "c", "d", "e"], dtype=object),
            "source_row_id": np.array([0, 1, 0, 0, 1], dtype=np.int32),
            "wages": [52000.0, 0.0, 18750.5, 9100.0, 0.0],
        }
    )
    tables = {"person": person}
    for group in groups:
        ids = [1, 2, 3] if group == "household" else [11, 21, 31, 32]
        tables[group] = pd.DataFrame({US_SHAPED_SCHEMA.id_column(group): ids})
    return Frame(
        tables,
        US_SHAPED_SCHEMA,
        {
            "household": Weights(US_SHAPED_HOUSEHOLD_WEIGHTS, WeightKind.IMPORTANCE),
            "person": Weights(US_SHAPED_PERSON_WEIGHTS, WeightKind.DESIGN),
        },
        pd.Series(["asec_2023", "asec_2023", "asec_2024", "asec_2024", "asec_2024"]),
    )


def _us_shaped_mint(ids: np.ndarray, k: int) -> np.ndarray:
    return ids + k * US_SHAPED_MULTIPLIER


def _us_shaped_require_below(ids: np.ndarray, what: str) -> None:
    if (np.asarray(ids) >= US_SHAPED_MULTIPLIER).any():
        raise ValueError(f"Some {what} reach the multiplier.")


US_SHAPED_DECLARATION = CloneDeclaration(
    ids=CloneIdRule(
        bound=len(CHANNELS),
        mint=_us_shaped_mint,
        require_below=_us_shaped_require_below,
    ),
    clone_index_columns={
        entity: f"{entity}_support_clone_index" for entity in US_SHAPED_SCHEMA.entities
    },
    weight_kinds=frozenset({WeightKind.DESIGN, WeightKind.IMPORTANCE}),
    reason="Support clones split each household's weight equally.",
    source_id_columns={
        entity: f"{entity}_source_id" for entity in US_SHAPED_SCHEMA.entities
    },
)


def test_us_shaped_clones_pass_the_shared_clone_invariant() -> None:
    frame = _us_shaped_frame()
    expanded = expand_clones(
        frame, clones=len(CHANNELS), declaration=US_SHAPED_DECLARATION
    )
    labels = np.asarray(CHANNELS, dtype=object)
    tables = {}
    for entity in expanded.entities:
        table = expanded.table(entity).copy()
        index = table[f"{entity}_support_clone_index"].to_numpy()
        table[f"{entity}_support_channel"] = labels[index]
        tables[entity] = table
    labelled = Frame(
        tables,
        expanded.schema,
        {entity: expanded.weights_for(entity) for entity in expanded.weighted_entities},
        expanded.strata,
        mass_log=expanded.mass_log,
    )
    assert_clone_expansion(frame, labelled, channels=CHANNELS)
    # Both weighted entities split, each keeping its kind, with a record each.
    for entity, values, kind in (
        ("person", US_SHAPED_PERSON_WEIGHTS, WeightKind.DESIGN),
        ("household", US_SHAPED_HOUSEHOLD_WEIGHTS, WeightKind.IMPORTANCE),
    ):
        weights = labelled.weights_for(entity)
        assert weights.kind is kind
        assert weights.values.tobytes() == np.tile(values / 2, 2).tobytes()
    assert [record.entity for record in labelled.mass_log] == ["person", "household"]


# ---------------------------------------------------------------------------
# Refusals: each names entities, columns, kinds and declared constants only.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("clones", [0, -1, BOUND + 1, True, 2.0, "2", np.int64(0)])
def test_a_clone_count_outside_the_declared_bound_is_refused(clones: object) -> None:
    frame = _frame()
    with pytest.raises(
        ValueError, match=r"an integer in \[1, 5\], the declared bound"
    ) as error:
        expand_clones(frame, clones=clones, declaration=DECLARATION)  # type: ignore[arg-type]
    _assert_names_no_value(error, frame)


def test_a_frame_that_already_carries_a_declared_column_is_refused() -> None:
    once = expand_clones(_frame(), clones=2, declaration=DECLARATION)
    with pytest.raises(ValueError, match="already carries the declared") as error:
        expand_clones(once, clones=2, declaration=DECLARATION)
    assert str(error.value).endswith(f"{sorted(DECLARED_COLUMNS)}.")
    _assert_names_no_value(error, once)
    tables = _tables()
    tables["household"]["household_origin"] = tables["household"]["household_id"]
    frame = _frame(tables)
    with pytest.raises(
        ValueError, match=r"column\(s\) \['household_origin'\]"
    ) as error:
        expand_clones(frame, clones=2, declaration=DECLARATION)
    _assert_names_no_value(error, frame)


def test_ids_the_mint_has_no_room_for_are_refused_by_the_declared_rule() -> None:
    frame = _frame(_tables(households=(411, 522, RADIX + 633)))
    with pytest.raises(ValueError, match="1 of the household_id values reach") as error:
        expand_clones(frame, clones=2, declaration=DECLARATION)
    _assert_names_no_value(error, frame)
    handed: list[tuple[str, int]] = []

    def record(ids: np.ndarray, what: str) -> None:
        handed.append((what, len(ids)))

    rule = CloneIdRule(bound=BOUND, mint=_mint, require_below=record)
    expand_clones(_frame(), clones=2, declaration=_declaration(ids=rule))
    assert handed == [
        ("person_id values", 6),
        ("unit_id values", 4),
        ("household_id values", 3),
    ]


def test_a_weight_kind_the_declaration_does_not_accept_is_refused() -> None:
    frame = _frame(kind=WeightKind.IMPORTANCE)
    with pytest.raises(
        ValueError,
        match=r"accepts \['design'\] weights, not 'importance' weights on 'household'",
    ) as error:
        expand_clones(frame, clones=2, declaration=DECLARATION)
    _assert_names_no_value(error, frame)
    # Every weighted entity's kind is checked, not the household's alone.
    frame = _frame(
        extra_weights={"person": Weights(np.full(6, 2.5), WeightKind.IMPORTANCE)}
    )
    with pytest.raises(
        ValueError, match="not 'importance' weights on 'person'"
    ) as error:
        expand_clones(frame, clones=2, declaration=DECLARATION)
    _assert_names_no_value(error, frame)


def _offset_mint(ids: np.ndarray, k: int) -> np.ndarray:
    return ids + k * 10


def _reversing_mint(ids: np.ndarray, k: int) -> np.ndarray:
    return k * RADIX + (RADIX - 1 - ids)


def _float_mint(ids: np.ndarray, k: int) -> np.ndarray:
    return (ids + k * RADIX).astype(np.float64)


def _short_mint(ids: np.ndarray, k: int) -> np.ndarray:
    return (ids + k * RADIX)[:-1]


def _uint64_mint(ids: np.ndarray, k: int) -> np.ndarray:
    return (ids + k * RADIX).astype(np.uint64)


def _int64_mint(ids: np.ndarray, k: int) -> np.ndarray:
    return ids.astype(np.int64) + k * RADIX


_OWN_DTYPE = "one integer for each person id, in the ids' own dtype"


@pytest.mark.parametrize(
    ("mint", "match"),
    [
        (_offset_mint, "person ids above the clone before's"),
        (_reversing_mint, "keep the order of the person ids"),
        (_float_mint, _OWN_DTYPE),
        (_short_mint, _OWN_DTYPE),
        (_uint64_mint, _OWN_DTYPE),
    ],
)
def test_a_mint_that_breaks_the_clone_id_rule_is_refused(mint, match: str) -> None:
    frame = _frame()
    rule = CloneIdRule(bound=BOUND, mint=mint, require_below=_require_below)
    with pytest.raises(ValueError, match=match) as error:
        expand_clones(frame, clones=3, declaration=_declaration(ids=rule))
    _assert_names_no_value(error, frame)


def _with_id_dtype(dtype: str, *, memberships: str | None = None) -> dict:
    """The invented tables with every id in ``dtype``, and every membership
    in ``memberships`` (``dtype`` by default)."""

    tables = _tables()
    for entity, table in tables.items():
        column = SCHEMA.entity_id_column(entity)
        table[column] = table[column].astype(dtype)
    for group in SCHEMA.group_entities:
        column = SCHEMA.membership_column(group)
        tables["person"][column] = tables["person"][column].astype(memberships or dtype)
    return tables


@pytest.mark.parametrize("clones", [1, 3])
def test_a_mint_in_the_ids_dtype_keeps_it_at_every_clone_count(clones: int) -> None:
    frame = _frame(_with_id_dtype("int32"))
    expanded = expand_clones(frame, clones=clones, declaration=DECLARATION)
    for entity in frame.entities:
        ids = expanded.table(entity)[SCHEMA.entity_id_column(entity)]
        assert ids.dtype == np.int32, entity
        assert ids.tolist() == [
            value + k * RADIX
            for k in range(clones)
            for value in frame.table(entity)[ids.name].tolist()
        ]
    person = expanded.table("person")
    for group in SCHEMA.group_entities:
        assert person[SCHEMA.membership_column(group)].dtype == np.int32, group
    assert expanded.table("household")[SOURCE_COLUMNS["household"]].dtype == np.int32


def test_a_mint_out_of_the_ids_dtype_is_refused_before_it_floats_them() -> None:
    # Unsigned ids with a mint that returns int64, as a mint that casts to
    # int64 does: concatenating the two would give float64 ids at K = 2.
    frame = _frame(_with_id_dtype("uint64"))
    rule = CloneIdRule(bound=BOUND, mint=_int64_mint, require_below=_require_below)
    once = expand_clones(frame, clones=1, declaration=_declaration(ids=rule))
    assert once.table("household")["household_id"].dtype == np.uint64
    with pytest.raises(ValueError, match=_OWN_DTYPE) as error:
        expand_clones(frame, clones=2, declaration=_declaration(ids=rule))
    _assert_names_no_value(error, frame)


@pytest.mark.parametrize(
    ("ids", "memberships"), [("int64", "int32"), ("uint64", "int64")]
)
def test_memberships_in_another_dtype_than_their_units_ids_are_refused(
    ids: str, memberships: str
) -> None:
    # Concatenating the members with their units' minted ids would change
    # the memberships' dtype at K = 2, or float them.
    frame = _frame(_with_id_dtype(ids, memberships=memberships))
    with pytest.raises(
        TypeError,
        match=f"'person_unit_id' in the dtype of 'unit_id'; it holds {memberships} "
        f"against {ids}",
    ) as error:
        expand_clones(frame, clones=2, declaration=DECLARATION)
    _assert_names_no_value(error, frame)


def test_ids_that_are_not_integers_are_refused() -> None:
    tables = _tables()
    tables["household"]["household_id"] = tables["household"]["household_id"].astype(
        np.float64
    )
    tables["person"]["person_household_id"] = tables["person"][
        "person_household_id"
    ].astype(np.float64)
    frame = _frame(tables)
    with pytest.raises(TypeError, match="'household_id' holds float64") as error:
        expand_clones(frame, clones=2, declaration=DECLARATION)
    _assert_names_no_value(error, frame)


def test_a_frame_with_a_link_table_is_refused() -> None:
    schema = EntitySchema(
        group_entities=SCHEMA.group_entities,
        links=(LinkSpec("unit_household", "unit", "household"),),
    )
    tables = _tables()
    tables["unit_household"] = pd.DataFrame(
        {"unit_id": [4111, 4112, 5221, 6331], "household_id": [411, 411, 522, 633]}
    )
    frame = _frame(tables, schema=schema)
    with pytest.raises(ValueError, match="copies no link table") as error:
        expand_clones(frame, clones=2, declaration=DECLARATION)
    _assert_names_no_value(error, frame)


def test_a_declaration_that_does_not_fit_the_frame_s_entities_is_refused() -> None:
    frame = _frame()
    lacking = _declaration(
        clone_index_columns={**INDEX_COLUMNS, "family": "family_clone"}
    )
    with pytest.raises(ValueError, match=r"entities the frame lacks: \['family'\]"):
        expand_clones(frame, clones=2, declaration=lacking)
    partial = _declaration(
        clone_index_columns={
            entity: column
            for entity, column in INDEX_COLUMNS.items()
            if entity != "unit"
        }
    )
    with pytest.raises(ValueError, match=r"gives \['unit'\] no clone-index column"):
        expand_clones(frame, clones=2, declaration=partial)


def test_clones_whose_total_weight_strays_are_refused(monkeypatch) -> None:
    def biased(values: np.ndarray, count: int) -> np.ndarray:
        return np.tile(values / count, count) * (1.0 + 1e-6)

    monkeypatch.setattr(clone_expansion, "_equal_shares", biased)
    frame = _frame()
    with pytest.raises(
        ValueError, match="beyond the relative tolerance 1e-09"
    ) as error:
        expand_clones(frame, clones=3, declaration=DECLARATION)
    _assert_names_no_value(error, frame)


@pytest.mark.parametrize(
    ("make", "error"),
    [
        (
            lambda: CloneIdRule(bound=0, mint=_mint, require_below=_require_below),
            ValueError,
        ),
        (
            lambda: CloneIdRule(bound=True, mint=_mint, require_below=_require_below),
            ValueError,
        ),
        (
            lambda: CloneIdRule(bound=2.0, mint=_mint, require_below=_require_below),
            ValueError,
        ),
        (
            lambda: CloneIdRule(bound=2, mint=None, require_below=_require_below),
            TypeError,
        ),
        (lambda: _declaration(ids=_mint), TypeError),
        (lambda: _declaration(clone_index_columns={}), ValueError),
        (lambda: _declaration(clone_index_columns=["person_clone"]), TypeError),
        (lambda: _declaration(clone_index_columns={"person": ""}), ValueError),
        (
            lambda: _declaration(source_id_columns={"household": "household_clone"}),
            ValueError,
        ),
        (lambda: _declaration(weight_kinds=frozenset()), ValueError),
        (lambda: _declaration(weight_kinds=frozenset({"design"})), ValueError),
        (lambda: _declaration(reason="  "), ValueError),
    ],
)
def test_a_malformed_declaration_is_refused(make, error: type[Exception]) -> None:
    with pytest.raises(error):
        make()


def test_a_declaration_s_columns_cannot_be_changed_after_it_is_made() -> None:
    columns = dict(INDEX_COLUMNS)
    declaration = _declaration(clone_index_columns=columns)
    columns["person"] = "another"
    assert declaration.clone_index_columns["person"] == "person_clone"
    with pytest.raises(TypeError):
        declaration.clone_index_columns["person"] = "another"  # type: ignore[index]


def test_a_declaration_hashes_by_identity() -> None:
    # Its mappings are frozen views, which do not hash, so a declaration
    # compares and hashes as itself and can key a cache.
    assert {DECLARATION: "cached"}[DECLARATION] == "cached"
    assert _declaration() != DECLARATION
