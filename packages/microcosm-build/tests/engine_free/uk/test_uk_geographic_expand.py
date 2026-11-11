"""The geographic expansion, G1 of the clone-first spine (microcosm#931).

G1 clones every FRS household K times through the shared clone expansion core
under the UK's declaration, writing each household's source columns first.
These tests hold it to the clone-first stand-in (``clone_first_frame`` in
test_support/microcosm_build/uk_clone_first_frames.py, written without the
core) at K = 1, 2, 3, 7 and 15, and check the one-clone frame, the clone-major
ids, indices, sources and weights at K = 3, the dtypes, the mass record, the
evidence and every refusal. Invented frames only; every refusal prints counts,
names and declared constants, never an id or a weight.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from microcosm.build.clone_expansion import CloneDeclaration, CloneIdRule
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime import copy_id_table, geographic_expand
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.geographic_clones import (
    geographic_clone_count,
    ladder_clone_index_column,
)
from microcosm.build.uk_runtime.geographic_expand import (
    GEOGRAPHIC_CLONE_LIMIT,
    GEOGRAPHIC_COPY_ID_FIELD,
    GEOGRAPHIC_EXPAND_MASS_CHANGE_REASON,
    GEOGRAPHIC_EXPAND_STAGE_NAME,
    SOURCE_HOUSEHOLD_KEY_COLUMN,
    SOURCE_YEAR_COLUMN,
    UK_GEOGRAPHIC_CLONES,
    UK_GEOGRAPHIC_EXPAND_OUTPUT_COLUMNS,
    UKGeographicExpandStageTransform,
    expand_uk_geographic_clones,
    mint_geographic_clone_ids,
    require_room_for_geographic_clones,
)
from microcosm.build.uk_runtime.household_lineage import (
    GEOGRAPHIC_SUPPORT_BRANCH,
    UK_COPY_LINEAGE,
)
from microcosm.build.uk_runtime.national_frame import (
    UK_NATIONAL_SCHEMA,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import rounded_count
from microcosm.frame import Frame, MassChangeRecord, WeightKind, Weights
from microcosm.frame.concepts import SOURCE_HOUSEHOLD_ID_COLUMN
from test_support.microcosm_build.uk_clone_first_frames import (
    base_id_tables,
    clone_first_frame,
    with_last_household_moved,
)

_CLONE_COUNTS = (1, 2, 3, 7, 15)
#: The five id columns, by table.
_ID_COLUMNS = {
    "person": ("person_id", "person_household_id", "person_benunit_id"),
    "benunit": ("benunit_id",),
    "household": ("household_id",),
}
#: The pinned FRS survey year, so the frames' time period is the one the
#: geography identity maps to a vintage.
_PERIOD = str(load_uk_frs_release().survey_year)


def _base_frame(households: int = 12, *, time_period: str = _PERIOD) -> Frame:
    """Invented FRS-shaped ids (test_support's ``base_id_tables``) without any
    lineage, a text and a float column, and invented design weights."""

    tables = base_id_tables(households)
    household = tables.household[["household_id"]].copy()
    household["region"] = [
        "WALES" if sernum % 3 == 0 else "NORTH_EAST"
        for sernum in household["household_id"].tolist()
    ]
    person = tables.person.copy()
    person["employment_income"] = 875.5 * (person["person_id"] % 1000) + person[
        "person_household_id"
    ].astype(float)
    return uk_national_frame(
        person=person,
        benunit=tables.benunit,
        household=household,
        household_weights=np.random.default_rng(931).uniform(400.0, 3000.0, households),
        time_period=time_period,
    )


def _stage(**overrides: object) -> SourceStageSpec:
    """A stand-in stage for G1: no stage declares it until the spine wires it."""

    fields: dict[str, object] = {
        "stage": GEOGRAPHIC_EXPAND_STAGE_NAME,
        "survey": "Invented test stage",
        "source": "Invented test stage",
        "grain": "household",
        "artifacts": (),
        "operations": (),
        "outputs": UK_GEOGRAPHIC_EXPAND_OUTPUT_COLUMNS,
    }
    fields.update(overrides)
    return SourceStageSpec(**fields)


def _with_column(frame: Frame, entity: str, column: str) -> Frame:
    tables = {name: frame.table(name).copy() for name in frame.entities}
    tables[entity][column] = np.zeros(len(tables[entity]), dtype=np.int64)
    return Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _with_earlier_mass_record(frame: Frame) -> Frame:
    """``frame`` with one invented household mass record, as an earlier step
    would leave, so a test sees what G1 does to the log it is handed."""

    total = frame.weights_for("household").total
    record = MassChangeRecord(
        entity="household",
        old_total=2.0 * total,
        new_total=total,
        declared_factor=0.5,
        reason="An invented earlier step halved household mass.",
    )
    return Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
        mass_log=(*frame.mass_log, record),
        metadata=frame.metadata,
    )


def _weight_texts(frame: Frame) -> list[str]:
    return [repr(float(w)) for w in frame.weights_for("household").values]


# --- what G1 writes --------------------------------------------------------------


def test_one_clone_adds_no_row_and_changes_no_column() -> None:
    frame = _with_earlier_mass_record(_base_frame())
    expanded = expand_uk_geographic_clones(frame, clones=1).frame
    for entity in frame.entities:
        before, after = frame.table(entity), expanded.table(entity)
        assert len(after) == len(before)
        pd.testing.assert_frame_equal(
            after[list(before.columns)], before, check_exact=True
        )
        index = after[ladder_clone_index_column(entity)]
        assert index.dtype == np.int64 and not index.any()
    household = expanded.table("household")
    ids = household["household_id"]
    assert household[SOURCE_HOUSEHOLD_ID_COLUMN].to_numpy().tobytes() == (
        ids.to_numpy().tobytes()
    )
    assert household[SOURCE_YEAR_COLUMN].dtype == np.int64
    assert set(household[SOURCE_YEAR_COLUMN].tolist()) == {int(_PERIOD)}
    assert household[SOURCE_HOUSEHOLD_KEY_COLUMN].tolist() == [
        f"{_PERIOD}:{household_id}" for household_id in ids.tolist()
    ]
    weights = expanded.weights_for("household")
    assert weights.kind is WeightKind.DESIGN
    assert weights.values.tobytes() == frame.weights_for("household").values.tobytes()
    # The log it is handed, and one record more.
    assert len(frame.mass_log) == 1
    assert len(expanded.mass_log) == len(frame.mass_log) + 1
    assert expanded.mass_log[:-1] == frame.mass_log
    assert uk_time_period(expanded) == uk_time_period(frame)
    assert dict(expanded.metadata) == dict(frame.metadata)
    # Exactly the declared columns are added.
    added = {
        column
        for entity in frame.entities
        for column in expanded.table(entity).columns
        if column not in frame.table(entity).columns
    }
    assert added == set(UK_GEOGRAPHIC_EXPAND_OUTPUT_COLUMNS)


def test_three_clones_are_clone_major_with_exactly_three_indices() -> None:
    frame = _base_frame()
    clones = 3
    expanded = expand_uk_geographic_clones(frame, clones=clones).frame
    validate_uk_national_frame(expanded)
    radix = copy_id_table.radix(GEOGRAPHIC_COPY_ID_FIELD)
    for entity, columns in _ID_COLUMNS.items():
        table = expanded.table(entity)
        rows = len(frame.table(entity))
        index = table[ladder_clone_index_column(entity)].to_numpy()
        # Clone-major: clone k's rows are the frame's rows, in their order.
        assert index.tobytes() == np.repeat(np.arange(clones), rows).tobytes()
        assert sorted(set(index.tolist())) == list(range(clones))
        for column in columns:
            original = frame.table(entity)[column].to_numpy()
            assert table[column].dtype == np.int64
            assert table[column].tolist() == [
                int(value) + k * radix for k in range(clones) for value in original
            ]
    # Every member carries its household's and its benefit unit's clone.
    person = expanded.table("person")
    for group in UK_NATIONAL_SCHEMA.group_entities:
        units = expanded.table(group).set_index(UK_NATIONAL_SCHEMA.id_column(group))
        assert np.array_equal(
            units.loc[
                person[UK_NATIONAL_SCHEMA.membership_column(group)],
                ladder_clone_index_column(group),
            ].to_numpy(),
            person[ladder_clone_index_column("person")].to_numpy(),
        )
    household = expanded.table("household")
    assert geographic_clone_count(household) == clones
    # Each clone keeps its household's source columns.
    base_ids = frame.table("household")["household_id"].tolist()
    assert household[SOURCE_HOUSEHOLD_ID_COLUMN].tolist() == base_ids * clones
    assert (
        household[SOURCE_HOUSEHOLD_KEY_COLUMN].tolist()
        == [f"{_PERIOD}:{household_id}" for household_id in base_ids] * clones
    )
    # One float per household, the same in each clone, summing to the frame's.
    original = frame.weights_for("household").values
    weights = expanded.weights_for("household").values.reshape(clones, -1)
    for clone in weights:
        assert clone.tobytes() == (original / clones).tobytes()
    assert np.isclose(weights.sum(), original.sum(), rtol=1e-9, atol=0.0)
    record = expanded.mass_log[-1]
    assert (record.entity, record.declared_factor) == ("household", 1.0)
    assert record.reason == GEOGRAPHIC_EXPAND_MASS_CHANGE_REASON


@pytest.mark.parametrize("clones", _CLONE_COUNTS)
def test_g1_is_the_clone_first_stand_in(clones: int) -> None:
    """Bit for bit on every column the stand-in carries (the five id columns,
    the household clone index and source id, and the copied columns), in the
    stand-in's dtypes, and on the weights."""

    frame = _base_frame()
    expanded = expand_uk_geographic_clones(frame, clones=clones).frame
    stand_in = clone_first_frame(frame, clones)
    for entity in stand_in.entities:
        table = stand_in.table(entity)
        pd.testing.assert_frame_equal(
            expanded.table(entity)[list(table.columns)], table, check_exact=True
        )
        for column in _ID_COLUMNS[entity]:
            assert (
                expanded.table(entity)[column].to_numpy().tobytes()
                == table[column].to_numpy().tobytes()
            )
    got, want = expanded.weights_for("household"), stand_in.weights_for("household")
    assert got.kind is want.kind is WeightKind.DESIGN
    assert got.values.tobytes() == want.values.tobytes()


def test_the_source_key_is_python_strings_in_an_object_column() -> None:
    """Whatever pandas infers for text: the dtype the SPI support channel's
    own write gives with string inference off, as in the legacy plan, so
    that plan's object-string audit keeps the column."""

    for infer in (True, False):
        with pd.option_context("future.infer_string", infer):
            expanded = expand_uk_geographic_clones(_base_frame(), clones=3).frame
        key = expanded.table("household")[SOURCE_HOUSEHOLD_KEY_COLUMN]
        assert key.dtype == object
        assert {type(value) for value in key.tolist()} == {str}


def test_the_mass_record_names_no_clone_count() -> None:
    frame = _base_frame()
    records = [
        expand_uk_geographic_clones(frame, clones=clones).frame.mass_log[-1]
        for clones in _CLONE_COUNTS
    ]
    assert {record.reason for record in records} == {
        GEOGRAPHIC_EXPAND_MASS_CHANGE_REASON
    }
    assert not re.search(r"\d", GEOGRAPHIC_EXPAND_MASS_CHANGE_REASON)
    total = frame.weights_for("household").total
    for record in records:
        assert (record.entity, record.declared_factor) == ("household", 1.0)
        assert np.isclose(record.old_total, total, rtol=1e-9, atol=0.0)
        assert np.isclose(record.new_total, total, rtol=1e-9, atol=0.0)


def test_the_evidence_gives_k_the_field_rounded_sources_and_the_split() -> None:
    frame = _base_frame(households=26)
    evidence = {
        clones: expand_uk_geographic_clones(frame, clones=clones).evidence()
        for clones in (1, 3)
    }
    assert evidence[3] == {
        "stage": GEOGRAPHIC_EXPAND_STAGE_NAME,
        "clone_count": 3,
        "copy_id_field": GEOGRAPHIC_COPY_ID_FIELD,
        "copy_id_radix": copy_id_table.radix(GEOGRAPHIC_COPY_ID_FIELD),
        # Distinct source households, rounded at the receipts' floor: the
        # same at every K.
        "source_households": rounded_count(26),
        "weight_split": "equal",
        "weight_kind": "design",
    }
    assert {**evidence[1], "clone_count": 3} == evidence[3]
    small = expand_uk_geographic_clones(_base_frame(households=2), clones=3)
    assert small.evidence()["source_households"] == "<3"
    # The result's repr prints no count.
    assert repr(small) == "UKGeographicExpandResult(clone_count=3)"


def test_the_transform_runs_the_step_and_records_its_evidence() -> None:
    frame = _base_frame()
    transform = UKGeographicExpandStageTransform(_stage(), clones=3)
    with pytest.raises(RuntimeError, match="completed stage run"):
        transform.checkpoint_metadata()
    expanded = transform(frame)
    direct = expand_uk_geographic_clones(frame, clones=3)
    for entity in frame.entities:
        pd.testing.assert_frame_equal(
            expanded.table(entity), direct.frame.table(entity), check_exact=True
        )
    assert transform.checkpoint_metadata() == {"evidence": direct.evidence()}
    assert transform.output_columns() == UK_GEOGRAPHIC_EXPAND_OUTPUT_COLUMNS


def test_the_declaration_is_the_copy_id_tables_field_through_named_functions() -> None:
    """The mint and the bound check are this module's named functions, so the
    derived binding rule follows them into the copy-id table; the field is
    the copy lineage's geographic branch."""

    rule = UK_GEOGRAPHIC_CLONES.ids
    assert rule.mint is mint_geographic_clone_ids
    assert rule.require_below is require_room_for_geographic_clones
    assert rule.bound == GEOGRAPHIC_CLONE_LIMIT
    assert GEOGRAPHIC_CLONE_LIMIT == copy_id_table.ordinal_bound(
        GEOGRAPHIC_COPY_ID_FIELD
    )
    assert GEOGRAPHIC_COPY_ID_FIELD == GEOGRAPHIC_SUPPORT_BRANCH
    assert dict(UK_GEOGRAPHIC_CLONES.clone_index_columns) == {
        entity: ladder_clone_index_column(entity)
        for entity in UK_NATIONAL_SCHEMA.entities
    }
    assert dict(UK_GEOGRAPHIC_CLONES.source_id_columns) == {
        "household": SOURCE_HOUSEHOLD_ID_COLUMN
    }
    assert UK_GEOGRAPHIC_CLONES.weight_kinds == frozenset({WeightKind.DESIGN})
    assert UK_GEOGRAPHIC_CLONES.reason == GEOGRAPHIC_EXPAND_MASS_CHANGE_REASON
    ids = np.arange(1, 6, dtype=np.int64)
    assert (
        mint_geographic_clone_ids(ids, 2).tolist()
        == (ids + 2 * copy_id_table.radix(GEOGRAPHIC_COPY_ID_FIELD)).tolist()
    )


# --- refusals ---------------------------------------------------------------------


@pytest.mark.parametrize("clones", [0, GEOGRAPHIC_CLONE_LIMIT + 1, True, 2.0, "3"])
def test_a_clone_count_outside_the_field_s_bound_is_refused(clones: object) -> None:
    message = r"clone count from 1 to 100, the 'geographic_support' copy-id field"
    with pytest.raises(ValueError, match=message):
        UKGeographicExpandStageTransform(_stage(), clones=clones)
    with pytest.raises(ValueError, match=message):
        expand_uk_geographic_clones(_base_frame(), clones=clones)


def test_the_largest_clone_count_expands() -> None:
    expanded = expand_uk_geographic_clones(
        _base_frame(households=3), clones=GEOGRAPHIC_CLONE_LIMIT
    ).frame
    assert geographic_clone_count(expanded.table("household")) == 100


@pytest.mark.parametrize(
    ("entity", "column"),
    [
        *(
            (entity, ladder_clone_index_column(entity))
            for entity in UK_NATIONAL_SCHEMA.entities
        ),
        ("household", SOURCE_HOUSEHOLD_ID_COLUMN),
        ("household", SOURCE_YEAR_COLUMN),
        ("household", SOURCE_HOUSEHOLD_KEY_COLUMN),
        *(
            ("household", column)
            for column in UK_COPY_LINEAGE.copy_columns
            if column != ladder_clone_index_column("household")
        ),
    ],
)
def test_a_frame_with_a_clone_index_source_or_copy_column_is_refused(
    entity: str, column: str
) -> None:
    frame = _with_column(_base_frame(), entity, column)
    with pytest.raises(ValueError, match="before any copy step") as error:
        expand_uk_geographic_clones(frame, clones=2)
    assert repr(column) in str(error.value)


def test_a_second_expansion_is_refused() -> None:
    once = expand_uk_geographic_clones(_base_frame(), clones=2).frame
    with pytest.raises(ValueError, match="already carries") as error:
        expand_uk_geographic_clones(once, clones=2)
    for column in UK_GEOGRAPHIC_EXPAND_OUTPUT_COLUMNS:
        assert repr(column) in str(error.value)


def test_an_id_in_the_field_is_refused_by_count() -> None:
    radix = copy_id_table.radix(GEOGRAPHIC_COPY_ID_FIELD)
    moved = with_last_household_moved(_base_frame(), radix)
    with pytest.raises(
        ValueError,
        match="1 of the household_id values reach the 'geographic_support' copy-id",
    ) as error:
        expand_uk_geographic_clones(moved, clones=2)
    moved_id = int(moved.table("household")["household_id"].iloc[-1])
    assert str(moved_id) not in str(error.value)


def test_importance_weights_are_refused() -> None:
    frame = _base_frame()
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
    with pytest.raises(ValueError, match=r"accepts \['design'\] weights") as error:
        expand_uk_geographic_clones(importance, clones=2)
    for text in _weight_texts(frame):
        assert text not in str(error.value)


def test_weights_on_another_entity_are_refused() -> None:
    frame = _base_frame()
    people = len(frame.table("person"))
    weighted = Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {
            "person": Weights(np.full(people, 2.5), WeightKind.DESIGN),
            "household": frame.weights_for("household"),
        },
        frame.strata,
        metadata=frame.metadata,
    )
    with pytest.raises(
        ValueError, match="splits household weights alone, as the UK weights"
    ) as error:
        expand_uk_geographic_clones(weighted, clones=2)
    assert "'person'" in str(error.value)
    for text in _weight_texts(frame):
        assert text not in str(error.value)


def _mint_every_clone_at_ordinal_one(ids: np.ndarray, k: int) -> np.ndarray:
    return copy_id_table.mint(ids, GEOGRAPHIC_COPY_ID_FIELD, 1)


def test_a_mint_that_does_not_put_each_clone_above_the_last_is_refused(
    monkeypatch,
) -> None:
    patched = CloneDeclaration(
        ids=CloneIdRule(
            bound=GEOGRAPHIC_CLONE_LIMIT,
            mint=_mint_every_clone_at_ordinal_one,
            require_below=require_room_for_geographic_clones,
        ),
        clone_index_columns=dict(UK_GEOGRAPHIC_CLONES.clone_index_columns),
        weight_kinds=UK_GEOGRAPHIC_CLONES.weight_kinds,
        reason=UK_GEOGRAPHIC_CLONES.reason,
        source_id_columns=dict(UK_GEOGRAPHIC_CLONES.source_id_columns),
    )
    monkeypatch.setattr(geographic_expand, "UK_GEOGRAPHIC_CLONES", patched)
    # Two clones mint one clone above the frame, which this mint does.
    expand_uk_geographic_clones(_base_frame(), clones=2)
    with pytest.raises(ValueError, match="above the clone before's"):
        expand_uk_geographic_clones(_base_frame(), clones=3)


@pytest.mark.parametrize("clones", [1, 2])
def test_ids_that_are_not_int64_are_refused_at_every_clone_count(clones: int) -> None:
    """The copy-id table mints int64 ids, so the core refuses narrower ids
    from two clones on; G1 refuses them at one clone too, so every clone
    count takes the same frames."""

    frame = _base_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for entity, columns in _ID_COLUMNS.items():
        for column in columns:
            tables[entity][column] = tables[entity][column].astype(np.int32)
    narrow = Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    with pytest.raises(
        ValueError, match="takes int64 id and membership columns, the dtype"
    ) as error:
        expand_uk_geographic_clones(narrow, clones=clones)
    message = str(error.value)
    for columns in _ID_COLUMNS.values():
        for column in columns:
            assert f"{column!r} holds int32" in message
    for person_id in frame.table("person")["person_id"].tolist():
        assert str(person_id) not in message


def test_a_time_period_that_is_not_a_year_is_refused() -> None:
    with pytest.raises(ValueError, match="must name a year"):
        expand_uk_geographic_clones(_base_frame(time_period="FY24"), clones=2)


def test_a_non_frame_is_refused() -> None:
    with pytest.raises(TypeError, match="clones a Frame"):
        expand_uk_geographic_clones(_base_frame().table("household"), clones=2)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"stage": "spi_support_channel"}, "Expected stage 'geographic_expand'"),
        ({"grain": "person"}, "grain must be 'household'"),
        ({"outputs": UK_GEOGRAPHIC_EXPAND_OUTPUT_COLUMNS[:-1]}, "must declare outputs"),
        ({"rewrites": ("region",)}, "and no rewrites"),
    ],
)
def test_a_stage_that_is_not_g1_s_is_refused(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        UKGeographicExpandStageTransform(_stage(**overrides), clones=2)
