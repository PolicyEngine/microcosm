"""The UK copy-id table: fixed id fields for every copy layer (microcosm#931).

Every id here is synthetic. The table declares each copy layer's decimal field
of the id; these tests hold its fields, its helpers and the code that mints
and decodes through it: the order of ids at K=1 (as the retired rule ordered
them), nesting across K on clone-first frames, the executor's parent recovery,
the static guard against the retired rule and the dense line's refusal.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build as build
from microcosm.build.uk_runtime import (
    cgt_residential_split,
    cgt_structure,
    cgt_support,
    copy_id_table,
    graph_kernels,
    high_value_home_support,
)
from microcosm.build.uk_runtime.cgt_residential_split import (
    CGT_RESIDENTIAL_MAXIMUM_LIABLE_GAINERS,
)
from microcosm.build.uk_runtime.copy_id_table import (
    UK_COPY_ID_BASE_RADIX,
    UK_COPY_ID_FIELDS,
    UK_COPY_ID_LARGEST_ID,
    CopyIdField,
    mint,
    ordinal,
    ordinal_bound,
    parent,
    radix,
    require_below,
)
from microcosm.build.uk_runtime.high_value_home_support import (
    HIGH_VALUE_SUPPORT_COPIES,
)
from microcosm.build.uk_runtime.household_lineage import UK_COPY_LINEAGE
from microcosm.build.uk_runtime.rowwise_dataset import (
    clone_uk_dataset_tables_with_rowwise_geography,
    expand_uk_geographic_pool,
    refuse_dense_clone_id_overflow,
)
from microcosm.build.uk_runtime.spi_band_donors import (
    SPI_INCOME_BAND_DONOR_CLONE_INDEX,
    SPI_INCOME_BAND_DONOR_ID_REMAPPING,
)
from microcosm.build.uk_runtime.spi_support import (
    SPI_SUPPORT_ID_REMAPPING,
    create_uk_spi_support_tables,
    support_channel_column,
)
from test_support.microcosm_build.uk_clone_first_frames import (
    COPY_PATH_COLUMNS,
    GEOGRAPHIC_RADIX,
    IdTables,
    base_id_tables,
    clone_first_tables,
    copy_chain,
    copy_households,
    id_frame,
)
from test_support.paths import paths_for

_FIELD_NAMES = (
    "vintage",
    "geographic_support",
    "spi_support_channel",
    "high_value_support",
    "cgt_support_split",
    "cgt_incidence_clone",
    "cgt_residential_split",
)
#: The spine stage that mints each copy field (the band donors share the SPI
#: copies' field at ordinal 2).
_MINTING_STAGES = {
    "spi_support_channel": "spi_support_channel",
    "high_value_support": "high_value_home_support",
    "cgt_support_split": "cgt_support_split",
    "cgt_incidence_clone": "cgt_incidence_clone",
    "cgt_residential_split": "cgt_residential_split",
}
_INT64_MAX = 2**63 - 1


# --- the table -----------------------------------------------------------------


def test_the_fields_bounds_and_radices_are_the_declared_ones() -> None:
    assert tuple(field.name for field in UK_COPY_ID_FIELDS) == _FIELD_NAMES
    assert {field.name: (field.bound, field.radix) for field in UK_COPY_ID_FIELDS} == {
        "vintage": (10, 10**8),
        "geographic_support": (100, 10**9),
        "spi_support_channel": (3, 10**11),
        "high_value_support": (6, 10**12),
        "cgt_support_split": (10_000, 10**13),
        "cgt_incidence_clone": (2, 10**17),
        "cgt_residential_split": (8, 10**18),
    }
    assert UK_COPY_ID_BASE_RADIX == 10**8
    # The CGT field holds four decimal digits.
    assert copy_id_table._field("cgt_support_split").width == 10**4


def test_the_largest_mintable_id_fits_int64() -> None:
    expected = (
        sum((field.bound - 1) * field.radix for field in UK_COPY_ID_FIELDS) + 10**8 - 1
    )
    assert UK_COPY_ID_LARGEST_ID == expected
    assert 7.1e18 < UK_COPY_ID_LARGEST_ID < 7.2e18 < _INT64_MAX
    # Every field's largest ordinal stays inside it: the next field starts
    # where this one's width ends.
    for lower, upper in zip(UK_COPY_ID_FIELDS, UK_COPY_ID_FIELDS[1:], strict=False):
        assert lower.radix * lower.width == upper.radix


def test_the_copy_fields_are_the_lineage_branches_in_stage_order() -> None:
    lineage = {branch.name for branch in UK_COPY_LINEAGE.branches}
    assert set(_FIELD_NAMES) - {"vintage"} == lineage
    # Application order, not the lineage's branch order: each field's stage
    # runs after every lower field's, so a copy's parent sits below its radix.
    payload = json.loads(
        (Path(build.__file__).parent / "uk" / "source_stages.json").read_text()
    )
    position = {stage["stage"]: i for i, stage in enumerate(payload["stages"])}
    order = [
        position[_MINTING_STAGES[name]]
        for name in _FIELD_NAMES
        if name in _MINTING_STAGES
    ]
    assert order == sorted(order)
    assert position["spi_income_band_donors"] < position["high_value_home_support"]


def test_every_bound_covers_its_stage_constant() -> None:
    assert HIGH_VALUE_SUPPORT_COPIES <= ordinal_bound("high_value_support")
    assert 2**CGT_RESIDENTIAL_MAXIMUM_LIABLE_GAINERS <= ordinal_bound(
        "cgt_residential_split"
    )
    assert SPI_INCOME_BAND_DONOR_CLONE_INDEX < ordinal_bound("spi_support_channel")
    assert ordinal_bound("cgt_support_split") == 10_000
    assert ordinal_bound("geographic_support") == 100


def test_a_malformed_table_is_refused() -> None:
    fields = list(UK_COPY_ID_FIELDS)
    # Ten arms over ten incidence ordinals would mint above int64.
    wide = fields[:5] + [
        CopyIdField("cgt_incidence_clone", 10, 10**17),
        CopyIdField("cgt_residential_split", 10, 10**18),
    ]
    with pytest.raises(ValueError, match="int64 cannot hold"):
        copy_id_table._validate_fields(wide, 10**8)
    gap = [*fields[:2], CopyIdField("spi_support_channel", 3, 10**12), *fields[3:]]
    with pytest.raises(ValueError, match="not contiguous"):
        copy_id_table._validate_fields(gap, 10**8)
    with pytest.raises(ValueError, match="each once"):
        copy_id_table._validate_fields([fields[0], fields[0]], 10**8)
    with pytest.raises(ValueError, match="at least 2"):
        copy_id_table._validate_fields([CopyIdField("vintage", 1, 10**8)], 10**8)
    with pytest.raises(ValueError, match="No UK copy-id field"):
        radix("clone_index")


def _declared_id_remapping(operations, kind: str) -> str:
    return next(
        parameters["id_remapping"]
        for operation_kind, parameters in operations
        if operation_kind == kind
    )


@pytest.mark.parametrize(
    ("field", "remapping", "ordinal_text"),
    [
        ("spi_support_channel", SPI_SUPPORT_ID_REMAPPING, "1 x the field's radix"),
        (
            "spi_support_channel",
            SPI_INCOME_BAND_DONOR_ID_REMAPPING,
            f"{SPI_INCOME_BAND_DONOR_CLONE_INDEX} x the field's radix",
        ),
        (
            high_value_home_support.HIGH_VALUE_SUPPORT_COPY_ID_FIELD,
            _declared_id_remapping(
                high_value_home_support.high_value_support_operation_parameters(),
                high_value_home_support.HIGH_VALUE_SUPPORT_OPERATION_KIND,
            ),
            "k x the field's radix",
        ),
        (
            cgt_support.CGT_SUPPORT_COPY_ID_FIELD,
            _declared_id_remapping(
                cgt_support.cgt_support_split_operation_parameters(),
                "split_top_wealth_support_households",
            ),
            f"k up to {ordinal_bound('cgt_support_split') - 1:,}",
        ),
        (
            cgt_structure.CGT_CLONE_COPY_ID_FIELD,
            cgt_structure.CGT_CLONE_ID_REMAPPING,
            "the field's radix",
        ),
        (
            cgt_residential_split.CGT_RESIDENTIAL_COPY_ID_FIELD,
            _declared_id_remapping(
                cgt_residential_split.cgt_residential_split_operation_parameters(),
                "split_liable_gainers_by_residential_probability",
            ),
            "j x the field's radix",
        ),
    ],
)
def test_each_declared_id_remapping_restates_its_fields_radix(
    field, remapping, ordinal_text
) -> None:
    # The manifests restate these strings verbatim (closed-world checks); the
    # numbers they quote must be the table's, so neither goes stale alone.
    assert remapping.startswith(f"copy-id table field {field}")
    powers = [int(power) for power in re.findall(r"\(10\*\*(\d+)\)", remapping)]
    assert [10**power for power in powers] == [radix(field)]
    assert ordinal_text in remapping


# --- the helpers ----------------------------------------------------------------


def test_mint_ordinal_and_parent_round_trip_in_every_field() -> None:
    rng = np.random.default_rng(931)
    base = rng.integers(1, 10**8, size=200, dtype=np.int64)
    ids = base.copy()
    ordinals = {}
    for field in UK_COPY_ID_FIELDS:
        drawn = rng.integers(1, field.bound, size=len(ids), dtype=np.int64)
        ids = mint(ids, field.name, drawn)
        ordinals[field.name] = drawn
    for field in reversed(UK_COPY_ID_FIELDS):
        np.testing.assert_array_equal(ordinal(ids, field.name), ordinals[field.name])
        ids = parent(ids, field.name)
        np.testing.assert_array_equal(ordinal(ids, field.name), 0)
    np.testing.assert_array_equal(ids, base)
    assert mint(np.zeros(0, dtype=np.int64), "cgt_support_split", 1).size == 0


def test_mint_refuses_every_malformed_copy_without_naming_an_id() -> None:
    ids = np.asarray([5, 7_000_000_042], dtype=np.int64)
    # An ordinal outside the field is refused in one fixed sentence.
    outside = (
        "A 'high_value_support' copy ordinal falls outside [1, 6), the field's "
        "declared bound (microcosm#931)."
    )
    for ordinals in (0, [1, 6], [6, 7]):
        with pytest.raises(ValueError) as caught:
            mint(ids, "high_value_support", ordinals)
        assert str(caught.value) == outside
    with pytest.raises(ValueError, match=r"^1 id\(s\) already carry"):
        mint(ids + np.asarray([0, 10**12]), "high_value_support", 1)
    # Empty CGT digits under higher ones near the top of int64.
    near_top = np.asarray([9_200_000_000_000_000_000], dtype=np.int64)
    with pytest.raises(ValueError, match="overflow int64") as caught:
        mint(near_top, "cgt_support_split", 2_400)
    assert str(int(near_top[0])) not in str(caught.value)
    with pytest.raises(ValueError, match="one integer or one per id"):
        mint(ids, "high_value_support", [1, 2, 3])
    with pytest.raises(TypeError, match="must be integers"):
        mint(ids.astype(float), "high_value_support", 1)
    with pytest.raises(TypeError, match="must be integers"):
        mint(ids, "high_value_support", True)
    with pytest.raises(ValueError, match="negative"):
        mint(np.asarray([-3]), "high_value_support", 1)


def test_require_below_refuses_ids_at_the_radix_printing_only_counts() -> None:
    require_below(np.asarray([1, 10**12 - 1]), "high_value_support")
    secret = 10**12 + 4_321
    with pytest.raises(ValueError, match=r"^2 of the incoming ids reach") as caught:
        require_below(
            pd.Series([3, 10**12, secret]), "high_value_support", what="incoming ids"
        )
    assert str(secret) not in str(caught.value)
    # Object columns of Python ints are read exactly, above 2**53 too.
    require_below(pd.Series([2**53 + 1], dtype=object), "cgt_residential_split")


def test_the_frs_spine_refuses_raw_ids_that_reach_the_first_copy_field() -> None:
    from microcosm.build.uk_runtime import frs_spine

    tables = base_id_tables(3)
    frs_spine._require_raw_ids_below_copy_fields(id_frame(tables))
    # A sernum of 100,000 gives person ids from 10**8, the vintage field.
    person = tables.person.copy()
    person.loc[person.index[-1], "person_id"] = 10**8 + 1
    with pytest.raises(ValueError, match=r"^1 of the raw FRS person_id values reach"):
        frs_spine._require_raw_ids_below_copy_fields(
            id_frame(type(tables)(person, tables.benunit, tables.household))
        )


# --- order and nesting on synthetic chains ---------------------------------------


def test_k1_ids_sort_as_the_retired_rule_sorted_them_and_base_ids_keep() -> None:
    """The table orders ids at K=1 as the retired rule did.

    This catches a table whose fields run out of stage order below the twin,
    but not every misordering: swapping the twin and arm fields leaves every
    K=1 sort unchanged, because arms are copies of twins. The exact pin of
    the fields (``test_the_fields_bounds_and_radices_are_the_declared_ones``)
    and each stage's ``require_below`` carry that order.
    """
    base = base_id_tables()
    table = copy_chain(base, rule="table")
    retired = copy_chain(base, rule="retired")
    for entity in ("household", "benunit", "person"):
        np.testing.assert_array_equal(
            np.argsort(table.ids(entity), kind="stable"),
            np.argsort(retired.ids(entity), kind="stable"),
        )
    path = table.household[list(COPY_PATH_COLUMNS)].to_numpy()
    originals = (path == 0).all(axis=1)
    np.testing.assert_array_equal(
        table.household["household_id"].to_numpy()[originals],
        base.household["household_id"].to_numpy(),
    )
    np.testing.assert_array_equal(
        table.household["household_id"].to_numpy()[originals],
        retired.household["household_id"].to_numpy()[originals],
    )
    # Every layer copied something, and the CGT ordinals reach two digits.
    assert table.household[list(COPY_PATH_COLUMNS)].max().to_dict() == {
        "spi": 2,
        "hv": 5,
        "cgt": 11,
        "twin": 1,
        "arm": 7,
    }
    assert int(table.household["household_id"].max()) > 2**53


def _clone_ids(tables, entity: str, clone: int) -> np.ndarray:
    household_clone = tables.household.set_index("household_id")[
        "household_clone_index"
    ]
    if entity == "household":
        mask = tables.household["household_clone_index"].to_numpy() == clone
        return np.sort(tables.ids("household")[mask])
    person_clone = tables.person["person_household_id"].map(household_clone)
    if entity == "person":
        return np.sort(tables.ids("person")[person_clone.to_numpy() == clone])
    units = set(
        tables.person.loc[person_clone.to_numpy() == clone, "person_benunit_id"]
    )
    return np.sort(
        tables.ids("benunit")[tables.benunit["benunit_id"].isin(units).to_numpy()]
    )


def test_clone_zero_and_its_copies_keep_the_k1_ids_on_clone_first_frames() -> None:
    base = base_id_tables()
    k1 = copy_chain(clone_first_tables(base, 1), rule="table")
    k3 = copy_chain(clone_first_tables(base, 3), rule="table")
    for entity in ("household", "benunit", "person"):
        k1_ids = _clone_ids(k1, entity, 0)
        assert len(k1_ids) == len(k1.ids(entity))
        np.testing.assert_array_equal(_clone_ids(k3, entity, 0), k1_ids)
        for clone in (1, 2):
            ids = _clone_ids(k3, entity, clone)
            np.testing.assert_array_equal(
                ids, np.sort(k1_ids + clone * GEOGRAPHIC_RADIX)
            )
            assert not set(ids.tolist()) & set(k1_ids.tolist())
    # Ten to the frame's digit count would give clone 0 other ids at K=3.
    r1 = copy_chain(clone_first_tables(base, 1), rule="retired")
    r3 = copy_chain(clone_first_tables(base, 3), rule="retired")
    assert not np.array_equal(
        _clone_ids(r3, "household", 0), _clone_ids(r1, "household", 0)
    )


def _spi_support_on_clone_first_tables(base: IdTables, n_clones: int):
    """The real SPI support stage on ``n_clones`` geographic clones of ``base``.

    Every clone of the same base households is selected, so only the stage's
    id minting can tell K=1 from K=3 apart.
    """

    tables = clone_first_tables(base, n_clones)
    sources = [
        int(source)
        for source in base.household["household_id"].tolist()
        if source % 3 != 0
    ]
    return create_uk_spi_support_tables(
        person=tables.person,
        benunit=tables.benunit,
        household=tables.household[
            ["household_id", "source_household_id", "household_clone_index"]
        ].assign(household_weight=1.0),
        expected_spi_households=None,
        selected_household_ids=tuple(
            source + clone * GEOGRAPHIC_RADIX
            for clone in range(n_clones)
            for source in sources
        ),
        source_year=2024,
    )


def test_the_spi_stage_mints_clone_zero_copies_at_their_k1_ids() -> None:
    """Stage 16, where the retired rule grew with K, run on clone-first frames.

    Clone 0's rows and SPI copies keep the K=1 ids exactly, and clone ``k``'s
    are K=1's plus ``k x 10**9``. An offset sized from the ids the stage sees
    would put K=3's copies a digit higher. The later copy stages' selections,
    copy counts and draws move with K until group (a)'s later commits and (b)
    make them per clone, so (b) owns their stage-level nesting tests.
    """

    base = base_id_tables(12)
    k1 = _spi_support_on_clone_first_tables(base, 1)
    k3 = _spi_support_on_clone_first_tables(base, 3)
    channel = support_channel_column("household")
    assert int((k1.household[channel] == "spi").sum()) > 0
    for entity, column in (
        ("household", "household_id"),
        ("benunit", "benunit_id"),
        ("person", "person_id"),
        ("person", "person_household_id"),
        ("person", "person_benunit_id"),
    ):
        k1_ids = getattr(k1, entity)[column].to_numpy(dtype=np.int64)
        k3_ids = getattr(k3, entity)[column].to_numpy(dtype=np.int64)
        np.testing.assert_array_equal(
            np.sort(k3_ids),
            np.sort(
                np.concatenate(
                    [k1_ids + clone * GEOGRAPHIC_RADIX for clone in range(3)]
                )
            ),
        )


# --- the executor ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("stage", "field", "layers"),
    [
        ("high_value_home_support", "high_value_support", 2),
        ("cgt_support_split", "cgt_support_split", 3),
        ("cgt_incidence_clone", "cgt_incidence_clone", 4),
        ("cgt_residential_split", "cgt_residential_split", 5),
    ],
)
def test_the_executor_recovers_each_copy_parent_at_the_declared_radix(
    stage, field, layers
) -> None:
    base = base_id_tables()
    before = id_frame(copy_chain(base, layers=layers))
    after = id_frame(copy_chain(base, layers=layers + 1))
    offset = graph_kernels._declared_copy_radix(stage, before)
    assert offset == radix(field)
    for entity in before.entities:
        lineage = graph_kernels._source_lineage(before, after, entity, id_offset=offset)
        targets = lineage.index.to_numpy(dtype=np.int64)
        assert len(targets) > 0
        assert lineage.dtype == np.dtype("int64")
        np.testing.assert_array_equal(lineage.to_numpy(), parent(targets, field))
        assert set(lineage.tolist()) <= set(
            before.table(entity)[before.schema.entity_id_column(entity)].tolist()
        )


@pytest.mark.parametrize(
    ("stage", "field", "layers", "path_column"),
    [
        # No SPI layer below: the frame's largest id has five digits.
        ("high_value_home_support", "high_value_support", 0, "hv"),
        # SPI copies but no high-value copies: twelve digits, not thirteen.
        ("cgt_support_split", "cgt_support_split", 2, "cgt"),
    ],
)
def test_the_executor_takes_the_declared_radix_where_the_frame_maximum_differs(
    stage, field, layers, path_column
) -> None:
    base = base_id_tables()
    before_tables = copy_chain(base, layers=layers)
    after_tables = copy_households(
        before_tables,
        field=field,
        path_column=path_column,
        ordinals={0: (1, 2), 7: (1,), 11: (3,)},
        rule="table",
    )
    before = id_frame(before_tables)
    largest = max(int(before_tables.ids(entity).max()) for entity in before.entities)
    assert 10 ** len(str(largest)) != radix(field)
    offset = graph_kernels._declared_copy_radix(stage, before)
    assert offset == radix(field)
    for entity in before.entities:
        lineage = graph_kernels._source_lineage(
            before, id_frame(after_tables), entity, id_offset=offset
        )
        targets = lineage.index.to_numpy(dtype=np.int64)
        assert len(targets) > 0
        np.testing.assert_array_equal(lineage.to_numpy(), parent(targets, field))


def test_the_executor_names_each_stages_own_copy_id_field() -> None:
    assert dict(graph_kernels._COPY_ID_FIELDS) == {
        "high_value_home_support": high_value_home_support.HIGH_VALUE_SUPPORT_COPY_ID_FIELD,
        "cgt_support_split": cgt_support.CGT_SUPPORT_COPY_ID_FIELD,
        "cgt_incidence_clone": cgt_structure.CGT_CLONE_COPY_ID_FIELD,
        "cgt_residential_split": cgt_residential_split.CGT_RESIDENTIAL_COPY_ID_FIELD,
    }


def test_the_executor_stays_exact_above_2_53_on_integer_and_float_tables() -> None:
    # The person table holds integer ids and one float column and nothing
    # else, so a row-by-row walk (iterrows) reads each row as float64, inexact
    # above 2**53: the retired walk named parents that are not incumbents here.
    # The vectorised read is exact.
    base = base_id_tables()

    def with_income(tables: IdTables) -> IdTables:
        person = tables.person.assign(employment_income=1_500.5)
        return IdTables(
            person=person, benunit=tables.benunit, household=tables.household
        )

    before = id_frame(with_income(copy_chain(base, layers=5)))
    after = id_frame(with_income(copy_chain(base)))
    person = after.table("person")
    assert set(person.dtypes.map(lambda dtype: dtype.kind)) == {"i", "f"}
    lineage = graph_kernels._source_lineage(
        before, after, "person", id_offset=radix("cgt_residential_split")
    )
    targets = lineage.index.to_numpy(dtype=np.int64)
    assert int(targets.min()) > 2**53
    assert lineage.dtype == np.dtype("int64")
    np.testing.assert_array_equal(
        lineage.to_numpy(), targets % radix("cgt_residential_split")
    )


def test_the_executor_refuses_new_rows_without_an_incumbent_parent_by_count() -> None:
    base = base_id_tables()
    before_tables = copy_chain(base, layers=2)
    stray = 10**12 + 99_999
    after_tables = copy_households(
        before_tables,
        field="high_value_support",
        path_column="hv",
        ordinals={0: (1,)},
        rule="table",
    )
    after_tables = IdTables(
        person=pd.concat(
            [
                after_tables.person,
                pd.DataFrame(
                    {
                        "person_id": [10**12 + 99_999_001],
                        "person_household_id": [stray],
                        "person_benunit_id": [10**12 + 9_999_901],
                    }
                ),
            ],
            ignore_index=True,
        ),
        benunit=pd.concat(
            [after_tables.benunit, pd.DataFrame({"benunit_id": [10**12 + 9_999_901]})],
            ignore_index=True,
        ),
        household=pd.concat(
            [
                after_tables.household,
                after_tables.household.iloc[[0]].assign(household_id=stray),
            ],
            ignore_index=True,
        ),
    )
    before, after = id_frame(before_tables), id_frame(after_tables)
    with pytest.raises(
        ValueError, match=r"^UK stage produced 1 new 'household' row\(s\) whose id"
    ) as caught:
        graph_kernels._source_lineage(
            before, after, "household", id_offset=radix("high_value_support")
        )
    assert str(stray) not in str(caught.value)
    # Without a declared radix or a source-id column, new rows are refused by
    # count too.
    with pytest.raises(
        ValueError, match=r"^UK stage produced 2 new 'household' row\(s\) without"
    ) as caught:
        graph_kernels._source_lineage(before, after, "household", id_offset=None)
    assert str(stray) not in str(caught.value)
    # The SPI support stages name each new row's parent in household_source_id:
    # the copy names an incumbent, the stray a household the frame never held.
    orphan_source = 10**11 + 77_777
    household = after_tables.household.copy()
    incumbent = household["household_id"].isin(before_tables.household["household_id"])
    sources = household["household_id"].where(
        incumbent, int(before_tables.household["household_id"].iloc[0])
    )
    sources[household["household_id"] == stray] = orphan_source
    household["household_source_id"] = sources.astype("int64")
    sourced = id_frame(
        IdTables(
            person=after_tables.person,
            benunit=after_tables.benunit,
            household=household,
        )
    )
    with pytest.raises(
        ValueError,
        match=(
            r"^UK stage produced 1 new 'household' row\(s\) whose "
            r"'household_source_id' is not an incumbent id\.$"
        ),
    ) as caught:
        graph_kernels._source_lineage(before, sourced, "household", id_offset=None)
    assert str(stray) not in str(caught.value)
    assert str(orphan_source) not in str(caught.value)


def test_the_executor_refuses_an_incumbent_id_at_its_radix_by_count() -> None:
    base = base_id_tables()
    before = id_frame(copy_chain(base, layers=4))
    assert graph_kernels._declared_copy_radix("frs_council_tax", before) is None
    # A frame that already carries twins reaches the twin's radix.
    twinned = copy_chain(base, layers=5)
    with pytest.raises(
        ValueError, match=r"^\d+ of the UK stage 'cgt_incidence_clone' incumbent"
    ) as caught:
        graph_kernels._declared_copy_radix("cgt_incidence_clone", id_frame(twinned))
    largest = int(twinned.household["household_id"].max())
    assert str(largest) not in str(caught.value)


# --- no UK spine module keeps the retired rule ------------------------------------

#: The rule the table replaced: ten to the digit count of the largest id seen.
_RETIRED_RULE = "id_multiplier_for_values"
#: The modules that keep the dense line's own clone multiplier, the retired
#: rule over the spine's ids, until #931 (c) re-bases the dense line onto the
#: clone-first spine, each with what it uses it for. Nothing else may name it.
_DENSE_RULE_MODULES = {
    "packages/microcosm-build/src/microcosm/build/uk_runtime/rowwise_geography.py": (
        "defines the dense clone multiplier and clones linked frames at it"
    ),
    "packages/microcosm-build/src/microcosm/build/uk_runtime/rowwise_dataset.py": (
        "the dense expand and the crosswalk clone route offset dense clones by it"
    ),
    "packages/microcosm-build/src/microcosm/build/uk_runtime/__init__.py": (
        "re-exports the dense helper"
    ),
    "packages/microcosm-build/src/microcosm/build/uk_runtime/size_evaluation.py": (
        "decodes dense clone ids back to spine ids"
    ),
    "tools/build_uk_rowwise_dataset.py": (
        "the row-wise tool's dry-run plans size the dense clone's ids by it"
    ),
}
#: Every module that mints, bounds or decodes spine copy ids; none may also
#: name the retired rule. Each reads the copy-id table, save the band
#: donors, which mint through the support channel's clone helper.
_COPY_ID_MODULES = frozenset(
    {
        *(
            f"packages/microcosm-build/src/microcosm/build/uk_runtime/{module}.py"
            for module in (
                "cgt_residential_split",
                "cgt_structure",
                "cgt_support",
                "frs_spine",
                "graph_kernels",
                "high_value_home_support",
                "spi_band_donors",
                "spi_support",
            )
        ),
        "tools/verify_uk_identity_stability.py",
    }
)
_MINTS_THROUGH_SPI_SUPPORT = frozenset(
    {"packages/microcosm-build/src/microcosm/build/uk_runtime/spi_band_donors.py"}
)


def _uk_python_sources() -> dict[str, str]:
    """The UK runtime's and the UK tools' sources by repository path."""

    paths = paths_for("microcosm-build")
    root = paths.repository
    runtime = paths.package / "src" / "microcosm" / "build" / "uk_runtime"
    uk_tool = re.compile(r"(?:^|_)uk(?:_|$)")
    sources = (
        *runtime.rglob("*.py"),
        *(path for path in (root / "tools").glob("*.py") if uk_tool.search(path.stem)),
    )
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in sources
    }


def _source_names(source: str) -> set[str]:
    """Every name a source imports, reads or reaches by attribute."""

    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
            if node.module:
                names.add(node.module.rsplit(".", 1)[-1])
        elif isinstance(node, ast.Import):
            names.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return names


def test_only_the_dense_line_names_the_retired_id_rule() -> None:
    """The retired rule survives only where the dense line keeps its own
    multiplier until #931 (c); a stale entry fails as well as a new user."""

    names = {
        path: _source_names(source) for path, source in _uk_python_sources().items()
    }
    assert {path for path, found in names.items() if _RETIRED_RULE in found} == set(
        _DENSE_RULE_MODULES
    )


#: The modules that compute ten to an id's digit count inline, each with what
#: it computes. A module reimplementing the retired rule without naming it
#: (``10 ** len(str(max(ids)))``) fails the guard below.
_INLINE_DIGIT_COUNT_MODULES = {
    "packages/microcosm-build/src/microcosm/build/uk_runtime/rowwise_geography.py": (
        "defines the dense clone multiplier, id_multiplier_for_values"
    ),
    "packages/microcosm-build/src/microcosm/build/uk_runtime/copy_id_table.py": (
        "sizes each declared field's span from its ordinal bound"
    ),
}


def _computes_ten_to_a_digit_count(source: str) -> bool:
    """Whether ``source`` raises ten to a power built from ``len(str(...))``."""

    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Pow)
            and isinstance(node.left, ast.Constant)
            and node.left.value == 10
        ):
            continue
        for call in ast.walk(node.right):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "len"
                and call.args
                and isinstance(call.args[0], ast.Call)
                and isinstance(call.args[0].func, ast.Name)
                and call.args[0].func.id == "str"
            ):
                return True
    return False


def test_only_the_declared_modules_compute_ten_to_an_ids_digit_count() -> None:
    """The name guards cannot see the retired rule written out inline, as size
    evaluation's clone-0 path was before #931; this one can."""

    found = {
        path
        for path, source in _uk_python_sources().items()
        if _computes_ten_to_a_digit_count(source)
    }
    assert found == set(_INLINE_DIGIT_COUNT_MODULES)
    assert not set(_INLINE_DIGIT_COUNT_MODULES) & _COPY_ID_MODULES


def test_no_module_that_mints_or_decodes_copy_ids_names_the_retired_rule() -> None:
    """Each module that mints, bounds or decodes spine copy ids reads the
    copy-id table, the identity tool among them, and none names the rule the
    table replaced; the readers of the table are exactly these modules."""

    names = {
        path: _source_names(source) for path, source in _uk_python_sources().items()
    }
    readers = {path for path, found in names.items() if "copy_id_table" in found}
    readers.discard(
        "packages/microcosm-build/src/microcosm/build/uk_runtime/copy_id_table.py"
    )
    assert readers == _COPY_ID_MODULES - _MINTS_THROUGH_SPI_SUPPORT
    for path in sorted(_COPY_ID_MODULES):
        assert _RETIRED_RULE not in names[path], path
    assert not _COPY_ID_MODULES & set(_DENSE_RULE_MODULES)


# --- the dense line ---------------------------------------------------------------


def _pool_inputs(tables):
    # A spine carries no clone index and no test path columns.
    household = tables.household.drop(
        columns=["household_clone_index", *COPY_PATH_COLUMNS]
    )
    household["household_weight"] = 1.0
    household["region"] = "LONDON"
    return {
        "person": tables.person,
        "benunit": tables.benunit,
        "household": household,
        "source_year": 2024,
        "time_period": "2024",
    }


def _crosswalk_clone(*, n_clones, person, benunit, household, source_year, **_):
    """The crosswalk clone route, on a one-area England crosswalk."""

    crosswalk = pd.DataFrame(
        [
            {
                "oa_code": "E0001",
                "lsoa_code": "E0101",
                "msoa_code": "E0201",
                "la_code": "E06000063",
                "constituency_code": "E14000001",
                "region_code": "E12000007",
                "country": "England",
                "population": 75,
            }
        ]
    )
    result = clone_uk_dataset_tables_with_rowwise_geography(
        person=person,
        benunit=benunit,
        household=household,
        crosswalk=crosswalk,
        n_clones=n_clones,
        source_year=source_year,
        require_all_countries=False,
    )
    return result.household, result.person


def _pool_expand(*, n_clones, **inputs):
    """The dense expand both the ladder clone route and uk.full.expand run."""

    pool = expand_uk_geographic_pool(n_clones=n_clones, **inputs)
    return pool.frame.table("household"), pool.frame.table("person")


@pytest.mark.parametrize("route", [_pool_expand, _crosswalk_clone])
def test_dense_clones_of_a_table_minted_spine_are_refused_in_one_message(
    route,
) -> None:
    # The row-wise tool's two dry-run plans are refused the same way
    # (test_uk_rowwise_dry_run.py and test_uk_ladder_rowwise_clone.py), and
    # the graph's uk.full.expand in test_uk_full_population_graph.py.
    spine = copy_chain(base_id_tables())
    with pytest.raises(ValueError) as caught:
        route(n_clones=2, **_pool_inputs(spine))
    message = str(caught.value)
    assert "microcosm#931" in message
    assert "clone-first spine" in message
    assert "n_clones=2" in message
    assert str(int(spine.household["household_id"].max())) not in message
    with pytest.raises(ValueError, match="clone-first spine"):
        refuse_dense_clone_id_overflow(
            n_clones=15, id_multiplier=10**18, ids=(spine.ids("household"),)
        )


@pytest.mark.parametrize("route", [_pool_expand, _crosswalk_clone])
def test_one_dense_clone_of_a_table_minted_spine_still_expands(route) -> None:
    spine = copy_chain(base_id_tables())
    household, person = route(n_clones=1, **_pool_inputs(spine))
    assert household["household_id"].dtype == np.dtype("int64")
    np.testing.assert_array_equal(
        household["household_id"].to_numpy(), spine.ids("household")
    )
    np.testing.assert_array_equal(person["person_id"].to_numpy(), spine.ids("person"))


def test_a_spine_without_copy_rows_still_expands_into_dense_clones() -> None:
    base = base_id_tables()
    pool = expand_uk_geographic_pool(n_clones=3, **_pool_inputs(base))
    # Today's dense rule: clones above ten to the spine's digit count.
    multiplier = 10 ** len(str(int(base.person["person_id"].max())))
    assert pool.id_multiplier == multiplier
    np.testing.assert_array_equal(
        pool.frame.table("household")["household_id"].to_numpy(),
        np.concatenate([base.ids("household") + k * multiplier for k in range(3)]),
    )
