"""Explicit source lineage survives clone transport and benefit-unit creation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.frame.concepts import (
    SOURCE_HOUSEHOLD_ID_COLUMN,
    TransportRule,
    Unit,
    concept,
    split_for_transport,
)
from microcosm.frame.transport import read_populace_us_donor
from microcosm.frame.unit_construction import build_benefit_units
from test_support.microcosm_build.transport_population import (
    UNIT_RULE,
    donor_pin,
    run_population,
    write_population_sources,
)

PROPERTY = settings(
    max_examples=12,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _cloned_tables(tables):
    """One original household has support copies in two different channels."""
    person = tables["person"]
    clone = person.loc[person["person_household_id"] == 101].copy()
    clone["person_id"] = clone["person_id"] + 100
    clone["person_household_id"] = 303
    clone["person_support_channel"] = "puf"
    tables["person"] = pd.concat([person, clone], ignore_index=True)
    household = tables["household"]
    clone_household = household.loc[household["household_id"] == 101].copy()
    clone_household["household_id"] = 303
    clone_household["household_support_channel"] = "puf"
    tables["household"] = pd.concat([household, clone_household], ignore_index=True)
    # A pool-scoped PH_SEQ is deliberately different from both raw lineage
    # and support-row ids. It is not the source key the reader must use.
    tables["person"]["PH_SEQ"] = tables["person"]["person_household_id"].map(
        {101: 7001, 202: 7002, 303: 7001}
    )


def _read(sources):
    return read_populace_us_donor(sources.donor, **donor_pin(sources.donor))


def test_source_household_identity_is_a_declared_carry_concept():
    item = concept(f"fact:household.{SOURCE_HOUSEHOLD_ID_COLUMN}")
    assert item.dtype == "str"
    assert item.unit is Unit.SOURCE_ID
    assert item.transport is TransportRule.CARRY
    assert not item.nullable


@PROPERTY
@given(order=st.permutations(tuple(range(9))), reverse_households=st.booleans())
def test_source_household_key_merges_cross_channel_clones_without_row_order(
    tmp_path, order, reverse_households
):
    def edit(tables):
        _cloned_tables(tables)
        if reverse_households:
            tables["household"] = tables["household"].iloc[::-1].reset_index(drop=True)

    sources = write_population_sources(
        tmp_path / "sources", edit_tables=edit, permutation=order
    )
    donor = _read(sources)
    households = donor.tables["household"].set_index("household_id")
    keys = households[SOURCE_HOUSEHOLD_ID_COLUMN]
    assert keys.loc[101] == keys.loc[303]
    assert keys.loc[101] != keys.loc[202]
    assert keys.nunique() == 2
    assert json.loads(keys.loc[101]) == [["int", "2024"], ["int", "101"]]
    assert json.loads(keys.loc[202]) == [["int", "2024"], ["int", "202"]]
    tables, dropped = split_for_transport(donor.tables)
    assert SOURCE_HOUSEHOLD_ID_COLUMN not in dropped["household"]
    pd.testing.assert_series_equal(
        tables["household"][SOURCE_HOUSEHOLD_ID_COLUMN],
        donor.tables["household"][SOURCE_HOUSEHOLD_ID_COLUMN],
    )
    units, _ = build_benefit_units(tables["person"], tables["household"], UNIT_RULE)
    expected = units[UNIT_RULE.household_column].map(keys)
    np.testing.assert_array_equal(units[UNIT_RULE.source_household_column], expected)


def test_source_household_key_distinguishes_reused_raw_ids_in_another_vintage(tmp_path):
    def edit(tables):
        _cloned_tables(tables)
        other_year = tables["person"]["person_household_id"] == 303
        tables["person"].loc[other_year, "source_year"] = 2023

    sources = write_population_sources(tmp_path / "sources", edit_tables=edit)
    keys = (
        _read(sources)
        .tables["household"]
        .set_index("household_id")[SOURCE_HOUSEHOLD_ID_COLUMN]
    )
    assert keys.loc[101] != keys.loc[303]
    assert keys.nunique() == 3


@pytest.mark.parametrize("missing", ["source_year", "source_household_id"])
def test_partial_source_household_lineage_is_refused(tmp_path, missing):
    def edit(tables):
        tables["person"] = tables["person"].drop(columns=missing)

    sources = write_population_sources(tmp_path / "sources", edit_tables=edit)
    with pytest.raises(ValueError, match="source lineage requires year and household"):
        _read(sources)


def test_inconsistent_source_household_lineage_is_refused(tmp_path):
    def edit(tables):
        tables["person"].loc[1, "source_household_id"] = 102

    sources = write_population_sources(tmp_path / "sources", edit_tables=edit)
    with pytest.raises(
        ValueError, match="source household lineage disagrees within household"
    ):
        _read(sources)


def test_absent_source_household_lineage_never_falls_back_to_support_row_ids(tmp_path):
    def edit(tables):
        tables["person"] = tables["person"].drop(
            columns=["source_year", "source_household_id"]
        )

    sources = write_population_sources(tmp_path / "sources", edit_tables=edit)
    tables = _read(sources).tables
    assert SOURCE_HOUSEHOLD_ID_COLUMN not in tables["household"]
    units, _ = build_benefit_units(tables["person"], tables["household"], UNIT_RULE)
    assert UNIT_RULE.source_household_column not in units


def test_create_graph_carries_source_household_keys_to_benefit_units(tmp_path):
    sources = write_population_sources(tmp_path / "sources", edit_tables=_cloned_tables)
    result = run_population(tmp_path / "graph", sources)
    # Every transported unit inherits the household key by membership,
    # including the distinct support rows belonging to the same original.
    frame = result.store.load_frame(result.manifest.nodes["nz.open"].frame_key)
    households = frame.table("household").set_index("household_id")
    units = frame.table("family")
    keys = households[SOURCE_HOUSEHOLD_ID_COLUMN]
    assert keys.loc[101] == keys.loc[303]
    np.testing.assert_array_equal(
        units[UNIT_RULE.source_household_column],
        units[UNIT_RULE.household_column].map(keys),
    )
