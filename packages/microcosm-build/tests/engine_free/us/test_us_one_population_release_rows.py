"""Whether a frame declares ACS rows in the one-population form (d2283 PR 3).

``one_population_declares_acs_rows`` reads a frame's spine assembly record
and its person support channels, and returns whether the frame declares and
holds ACS rows. These tests run it on the frames a release can see:

* a Route A frame, X5's assembled frame with no ACS row, and a stacked pool
  whose ACS rows carry the channel ``acs``, each False;
* the generated assembled frame with ACS rows, True, and still True after a
  selection drops some of its ACS households;
* a frame without support columns, False, with no error;
* the assembled frame without its person clone index and source id, still
  True, since the predicate reads only the channel and the record.

It refuses live ACS rows that the frame's record does not declare (no
record, a record without the channel, or a person count of 0), a record
that declares ACS rows the frame lacks, a record without version 1, a
channel list and native row counts, an ACS person count that is not a
nonnegative integer, and an argument that is not a frame.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.one_population_contract import (
    ACS_SUPPORT_CHANNEL,
    OnePopulationContractError,
    one_population_declares_acs_rows,
)
from microcosm.build.us_runtime.support_provenance import (
    SPINE_ASSEMBLY_MANIFEST_KEY,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.frame import Frame
from test_support.microcosm_build.us_one_population_contract import (
    assembled_frame,
    generate_population,
    reader_population,
)
from test_support.microcosm_build.us_stacked_spine import _cloned_stacked_fixture

_PERSON_CHANNEL = support_channel_column("person")
_MALFORMED_COUNT = (
    "Malformed spine assembly record: its person count for "
    f"{ACS_SUPPORT_CHANNEL!r} must be a nonnegative integer"
)
_MALFORMED_RECORD = (
    f"Malformed spine assembly record: {SPINE_ASSEMBLY_MANIFEST_KEY!r} "
    "must hold version 1, a channel list and native row counts."
)


@pytest.fixture(scope="module")
def assembled() -> Frame:
    return assembled_frame(generate_population(0))


@pytest.fixture(scope="module")
def readers():
    return reader_population()


def _rebuild(
    frame: Frame,
    *,
    metadata,
    tables: dict[str, pd.DataFrame] | None = None,
) -> Frame:
    if tables is None:
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=metadata,
    )


def _acs_persons(frame: Frame) -> np.ndarray:
    return frame.table("person")[_PERSON_CHANNEL].eq(ACS_SUPPORT_CHANNEL).to_numpy()


def _record_acs_persons(frame: Frame) -> int:
    record = frame.metadata[SPINE_ASSEMBLY_MANIFEST_KEY]
    return record["native_row_counts"]["person"][ACS_SUPPORT_CHANNEL]


def _with_acs_person_count(frame: Frame, value_of) -> Frame:
    """``frame`` whose record's ACS person count is ``value_of(count)``."""

    record = frame.metadata[SPINE_ASSEMBLY_MANIFEST_KEY]
    counts = {
        entity: dict(by_channel)
        for entity, by_channel in record["native_row_counts"].items()
    }
    counts["person"][ACS_SUPPORT_CHANNEL] = value_of(
        counts["person"][ACS_SUPPORT_CHANNEL]
    )
    return _rebuild(
        frame,
        metadata={SPINE_ASSEMBLY_MANIFEST_KEY: {**record, "native_row_counts": counts}},
    )


def test_a_route_a_frame_declares_no_acs_rows(readers) -> None:
    route_a = readers.route_a
    assert SPINE_ASSEMBLY_MANIFEST_KEY not in route_a.metadata
    assert set(route_a.table("person")[_PERSON_CHANNEL]) == {"asec", "puf_tax_detail"}

    assert one_population_declares_acs_rows(route_a) is False


def test_an_assembled_frame_without_acs_rows_declares_none(readers) -> None:
    # X5's ASEC-only assembled frame lists the ACS channel in its record,
    # with a count of 0.
    assembled = readers.assembled
    record = assembled.metadata[SPINE_ASSEMBLY_MANIFEST_KEY]
    assert ACS_SUPPORT_CHANNEL in record["channels"]
    assert _record_acs_persons(assembled) == 0
    assert not _acs_persons(assembled).any()

    assert one_population_declares_acs_rows(assembled) is False


def test_a_stacked_pool_declares_no_acs_rows() -> None:
    # The stacked pool's ACS rows carry the channel "acs", and its record
    # declares them under that name.
    stacked = _cloned_stacked_fixture()
    record = stacked.metadata[SPINE_ASSEMBLY_MANIFEST_KEY]
    assert ACS_SUPPORT_CHANNEL not in record["channels"]
    assert record["native_row_counts"]["person"]["acs"] > 0
    assert stacked.table("person")[_PERSON_CHANNEL].eq("acs").any()

    assert one_population_declares_acs_rows(stacked) is False


def test_an_assembled_frame_with_acs_rows_declares_them(assembled) -> None:
    live = int(_acs_persons(assembled).sum())
    assert live > 0
    assert _record_acs_persons(assembled) == live

    assert one_population_declares_acs_rows(assembled) is True


def test_a_frame_without_support_columns_declares_none(assembled) -> None:
    # A pre-clone frame carries no support column. The predicate reads the
    # person channel only when the person table has it, and runs no
    # provenance check.
    tables = {}
    for entity in assembled.entities:
        support = {
            support_channel_column(entity),
            support_clone_index_column(entity),
            support_source_id_column(entity),
            spine_source_id_column(entity),
            f"{entity}_spine",
        }
        table = assembled.table(entity)
        assert support <= set(table.columns), entity
        tables[entity] = table.drop(columns=sorted(support))
    bare = _rebuild(assembled, metadata={}, tables=tables)

    assert one_population_declares_acs_rows(bare) is False


def test_the_answer_reads_no_support_column_but_the_channel(assembled) -> None:
    # require_assembled_support_provenance, support_role_series and
    # validate_assembly_provenance each refuse a person table that keeps its
    # spine source id and loses its clone index. The predicate runs none of
    # them, so it still answers from the channel and the record.
    tables = {entity: assembled.table(entity).copy() for entity in assembled.entities}
    tables["person"] = tables["person"].drop(
        columns=[
            support_clone_index_column("person"),
            support_source_id_column("person"),
        ]
    )
    assert spine_source_id_column("person") in tables["person"]
    partial = _rebuild(assembled, metadata=assembled.metadata, tables=tables)

    assert one_population_declares_acs_rows(partial) is True


def test_live_acs_rows_without_a_record_refuse(assembled) -> None:
    stripped = _rebuild(assembled, metadata={})
    live = int(_acs_persons(stripped).sum())
    assert live > 0

    with pytest.raises(
        OnePopulationContractError,
        match=re.escape(
            f"Undeclared ACS rows: {live} person row(s) carry the support channel "
            f"{ACS_SUPPORT_CHANNEL!r}, and the frame has no spine assembly record."
        ),
    ):
        one_population_declares_acs_rows(stripped)


def test_live_acs_rows_under_a_zero_count_refuse(assembled) -> None:
    zero = _with_acs_person_count(assembled, lambda count: 0)
    record = zero.metadata[SPINE_ASSEMBLY_MANIFEST_KEY]
    assert ACS_SUPPORT_CHANNEL in record["channels"]
    assert _record_acs_persons(zero) == 0
    live = int(_acs_persons(zero).sum())
    assert live > 0

    with pytest.raises(
        OnePopulationContractError,
        match=re.escape(
            f"Undeclared ACS rows: {live} person row(s) carry the support channel "
            f"{ACS_SUPPORT_CHANNEL!r}, and the frame's spine assembly record "
            "counts no ACS person."
        ),
    ):
        one_population_declares_acs_rows(zero)


def test_live_acs_rows_under_a_record_without_the_channel_refuse(
    assembled,
) -> None:
    # The record keeps the writer's shape: the channel leaves the channel
    # list and every entity's counts.
    record = assembled.metadata[SPINE_ASSEMBLY_MANIFEST_KEY]
    assert ACS_SUPPORT_CHANNEL in record["channels"]
    unlisted = _rebuild(
        assembled,
        metadata={
            SPINE_ASSEMBLY_MANIFEST_KEY: {
                **record,
                "channels": tuple(
                    channel
                    for channel in record["channels"]
                    if channel != ACS_SUPPORT_CHANNEL
                ),
                "native_row_counts": {
                    entity: {
                        channel: count
                        for channel, count in by_channel.items()
                        if channel != ACS_SUPPORT_CHANNEL
                    }
                    for entity, by_channel in record["native_row_counts"].items()
                },
            }
        },
    )
    live = int(_acs_persons(unlisted).sum())
    assert live > 0

    with pytest.raises(
        OnePopulationContractError,
        match=re.escape(
            f"Undeclared ACS rows: {live} person row(s) carry the support channel "
            f"{ACS_SUPPORT_CHANNEL!r}, and the frame's spine assembly record "
            "does not list that channel."
        ),
    ):
        one_population_declares_acs_rows(unlisted)


def test_a_record_declaring_removed_acs_rows_refuses(assembled) -> None:
    cps_only = assembled.select(~_acs_persons(assembled))
    assert not _acs_persons(cps_only).any()
    declared = _record_acs_persons(cps_only)
    assert declared > 0

    with pytest.raises(
        OnePopulationContractError,
        match=re.escape(
            f"Missing ACS rows: the frame's spine assembly record declares {declared} "
            "ACS person(s), and no person row carries the support channel "
            f"{ACS_SUPPORT_CHANNEL!r}."
        ),
    ):
        one_population_declares_acs_rows(cps_only)


def test_selecting_away_some_acs_households_keeps_the_declaration(assembled) -> None:
    # A stand-in for the release's frozen-support selection: the record keeps
    # its clone-0 counts, and the frame holds fewer ACS persons than it says.
    person = assembled.table("person")
    acs = _acs_persons(assembled)
    households = np.unique(person.loc[acs, "person_household_id"].to_numpy())
    dropped = person["person_household_id"].isin(households[::2]).to_numpy()
    selected = assembled.select(~dropped)
    live = int(_acs_persons(selected).sum())
    assert 0 < live < _record_acs_persons(selected)

    assert one_population_declares_acs_rows(selected) is True


@pytest.mark.parametrize(
    "value_of",
    [float, str, lambda count: True, lambda count: -1],
    ids=["integral float", "digit string", "bool", "negative"],
)
def test_a_malformed_acs_person_count_refuses(assembled, value_of) -> None:
    malformed = _with_acs_person_count(assembled, value_of)

    with pytest.raises(OnePopulationContractError, match=re.escape(_MALFORMED_COUNT)):
        one_population_declares_acs_rows(malformed)


@pytest.mark.parametrize(
    "edit",
    [
        lambda record: {**record, "version": 2},
        lambda record: {**record, "channels": ACS_SUPPORT_CHANNEL},
        lambda record: {
            key: value for key, value in record.items() if key != "native_row_counts"
        },
        lambda record: tuple(record),
    ],
    ids=["version 2", "channel string", "no counts", "not a mapping"],
)
def test_a_malformed_record_refuses(assembled, edit) -> None:
    # The frame keeps its live ACS rows, so a predicate that read a malformed
    # record as no record would raise the undeclared message instead.
    record = assembled.metadata[SPINE_ASSEMBLY_MANIFEST_KEY]
    malformed = _rebuild(
        assembled, metadata={SPINE_ASSEMBLY_MANIFEST_KEY: edit(record)}
    )
    assert _acs_persons(malformed).any()

    with pytest.raises(OnePopulationContractError, match=re.escape(_MALFORMED_RECORD)):
        one_population_declares_acs_rows(malformed)


def test_a_non_frame_refuses(assembled) -> None:
    tables = {entity: assembled.table(entity) for entity in assembled.entities}

    with pytest.raises(TypeError, match=re.escape("frame must be a Frame, got dict.")):
        one_population_declares_acs_rows(tables)
