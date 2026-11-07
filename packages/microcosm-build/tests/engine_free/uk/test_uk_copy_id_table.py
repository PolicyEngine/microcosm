"""The UK copy-id table: fixed id fields for every copy layer (microcosm#931).

Every id here is synthetic. The table declares each copy layer's decimal field
of the id; these tests hold its fields, its helpers and the code that mints
and decodes through it: the order of ids at K=1 (as the retired rule ordered
them), nesting across K on clone-first frames, the executor's parent recovery,
the static guard against the retired rule and the dense line's refusal.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build as build
from microcosm.build.uk_runtime import copy_id_table, graph_kernels
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
    expand_uk_geographic_pool,
    refuse_dense_clone_id_overflow,
)
from microcosm.build.uk_runtime.spi_band_donors import (
    SPI_INCOME_BAND_DONOR_CLONE_INDEX,
)
from test_support.microcosm_build.uk_clone_first_frames import (
    COPY_PATH_COLUMNS,
    GEOGRAPHIC_RADIX,
    base_id_tables,
    clone_first_tables,
    copy_chain,
    id_frame,
)

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


def test_the_executor_stays_exact_above_2_53_on_all_integer_tables() -> None:
    # The person table carries integer columns only, which a row-by-row walk
    # would have read back as float64.
    base = base_id_tables()
    before = id_frame(copy_chain(base, layers=5))
    after = id_frame(copy_chain(base))
    lineage = graph_kernels._source_lineage(
        before, after, "person", id_offset=radix("cgt_residential_split")
    )
    targets = lineage.index.to_numpy(dtype=np.int64)
    assert int(targets.min()) > 2**53
    np.testing.assert_array_equal(
        lineage.to_numpy(), targets % radix("cgt_residential_split")
    )


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


@pytest.mark.parametrize(
    "module",
    [
        "spi_support",
        "spi_band_donors",
        "high_value_home_support",
        "cgt_support",
        "cgt_structure",
        "cgt_residential_split",
        "graph_kernels",
    ],
)
def test_no_uk_spine_stage_module_imports_the_retired_id_rule(module) -> None:
    import importlib

    source = Path(
        inspect.getfile(importlib.import_module(f"microcosm.build.uk_runtime.{module}"))
    ).read_text(encoding="utf-8")
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    assert "id_multiplier_for_values" not in names


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


def test_dense_clones_of_a_table_minted_spine_are_refused_in_one_message() -> None:
    spine = copy_chain(base_id_tables())
    with pytest.raises(ValueError) as caught:
        expand_uk_geographic_pool(n_clones=2, **_pool_inputs(spine))
    message = str(caught.value)
    assert "microcosm#931" in message
    assert "clone-first spine" in message
    assert "n_clones=2" in message
    assert str(int(spine.household["household_id"].max())) not in message
    with pytest.raises(ValueError, match="clone-first spine"):
        refuse_dense_clone_id_overflow(
            n_clones=15, id_multiplier=10**18, ids=(spine.ids("household"),)
        )


def test_one_dense_clone_of_a_table_minted_spine_still_expands() -> None:
    spine = copy_chain(base_id_tables())
    pool = expand_uk_geographic_pool(n_clones=1, **_pool_inputs(spine))
    np.testing.assert_array_equal(
        pool.frame.table("household")["household_id"].to_numpy(),
        spine.ids("household"),
    )
    np.testing.assert_array_equal(
        pool.frame.table("person")["person_id"].to_numpy(), spine.ids("person")
    )


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
