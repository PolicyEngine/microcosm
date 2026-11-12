"""The UK lineage reader: every household's true source and clone path.

``household_lineage.uk_household_clone_paths`` reads each post-clone
household's clone path from its lineage columns; the copy take-up stages and
property localisation call it, and ``household_draw_key`` over a household's
source and path is the key the spine's ``geographic_identity`` stage writes.
These tests run the reader itself on invented lineage, as the retired dense
identity node's tests ran it through that node (microcosm#931): the branch
path each lineage combination encodes at one geographic clone and at two, the
residential arms' branch, the spine without a clone index, and every refusal.
No FRS, no publisher file, no country engine.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.geographic_clones import ladder_clone_index_column
from microcosm.build.uk_runtime.household_lineage import (
    UK_HOUSEHOLD_LINEAGE_COLUMNS,
    uk_household_clone_paths,
)
from test_support.microcosm_build.uk_full_population_graph import SOURCE_VINTAGE

CLONE = ladder_clone_index_column("household")

# household_id, source_household_id, channel, support clone index, cgt,
# support copy index (0 on roots), region, and an optional residential arm
# index (0 where absent)
LINEAGE = [
    (1, 1, "frs", 0, False, 0, "LONDON"),
    (2, 2, "frs", 0, False, 0, "SCOTLAND"),
    (11, 1, "spi", 1, False, 0, "LONDON"),
    (21, 2, "spi", 1, False, 0, "SCOTLAND"),
    (101, 1, "frs", 0, True, 0, "LONDON"),
    (111, 1, "spi", 1, True, 0, "LONDON"),
    (1001, 2, "frs", 0, False, 1, "SCOTLAND"),
]
# microcosm#1063: the residential split clones a gaining household into
# residential arms; the branch sits between the incidence clone and the
# geographic clones.
RESIDENTIAL_ROWS = sorted(
    [
        *LINEAGE,
        (1101, 2, "frs", 0, True, 1, "SCOTLAND"),  # gaining clone of copy 1001
        (201, 1, "frs", 0, True, 0, "LONDON", 1),  # arm 1 of gaining clone 101
        (211, 1, "spi", 1, True, 0, "LONDON", 1),  # arm 1 of gaining clone 111
        (1201, 2, "frs", 0, True, 1, "SCOTLAND", 1),  # arm 1 of gaining clone 1101
    ],
    key=lambda row: row[0],
)


def _residential_index(row) -> int:
    return int(row[7]) if len(row) > 7 else 0


def lineage_table(rows=LINEAGE, *, clones: int | None = 1) -> pd.DataFrame:
    """The invented lineage's households at ``clones`` geographic clones.

    Clone ``k`` of every row carries index ``k`` and the row's lineage, as the
    spine's ``geographic_expand`` writes them; ``clones=None`` leaves the
    index out, as on a spine built before that stage.
    """

    base = pd.DataFrame(
        {
            "household_id": np.asarray([r[0] for r in rows], dtype=np.int64),
            "region": pd.array([r[6] for r in rows], dtype="string"),
            "source_household_id": np.asarray([r[1] for r in rows], dtype=np.int64),
            "household_support_channel": pd.array([r[2] for r in rows], dtype="string"),
            "household_support_clone_index": np.asarray(
                [r[3] for r in rows], dtype=np.int64
            ),
            "household_is_capital_gains_clone": np.asarray([r[4] for r in rows]),
            "household_is_cgt_support_copy": np.asarray([r[5] != 0 for r in rows]),
            "cgt_support_copy_index": np.asarray([r[5] for r in rows], dtype=np.int64),
            "household_is_cgt_residential_clone": np.asarray(
                [_residential_index(r) != 0 for r in rows]
            ),
            "cgt_residential_clone_index": np.asarray(
                [_residential_index(r) for r in rows], dtype=np.int64
            ),
        }
    )
    if clones is None:
        return base
    copies = []
    for clone in range(clones):
        copy = base.copy()
        copy[CLONE] = np.full(len(copy), clone, dtype=np.int64)
        copies.append(copy)
    return pd.concat(copies, ignore_index=True)


def expected_path(
    channel, support_index, cgt, copy_index, clone_index, residential_index=0
) -> tuple[tuple[str, int], ...]:
    """The declared branch path of one household's lineage, in branch order."""

    path = []
    if channel == "spi":
        path.append(("spi_support_channel", int(support_index)))
    if copy_index:
        path.append(("cgt_support_split", int(copy_index)))
    if cgt:
        path.append(("cgt_incidence_clone", 1))
    if residential_index:
        path.append(("cgt_residential_split", int(residential_index)))
    if clone_index:
        path.append(("geographic_support", int(clone_index)))
    return tuple(path)


def _keys(sources, paths) -> list[str]:
    return [
        household_draw_key(
            source="frs",
            source_vintage=SOURCE_VINTAGE,
            source_household_id=int(source),
            clone_path=path,
        )
        for source, path in zip(sources, paths, strict=True)
    ]


#: The lineage fields :func:`expected_path` reads, in its argument order.
_PATH_FIELDS = (
    "household_support_channel",
    "household_support_clone_index",
    "household_is_capital_gains_clone",
    "cgt_support_copy_index",
    CLONE,
    "cgt_residential_clone_index",
)


def _expected_paths(household: pd.DataFrame) -> list[tuple[tuple[str, int], ...]]:
    columns = [household[field].tolist() for field in _PATH_FIELDS]
    return [expected_path(*fields) for fields in zip(*columns, strict=True)]


@pytest.mark.parametrize("k", [1, 2])
def test_every_lineage_combination_encodes_its_declared_branch_path(k):
    rows = [
        (100 + index, 1, channel, int(channel == "spi"), cgt, copy_index, "LONDON")
        for index, (channel, cgt, copy_index) in enumerate(
            itertools.product(("frs", "spi"), (False, True), (0, 2))
        )
    ]
    household = lineage_table(rows, clones=k)
    sources, paths = uk_household_clone_paths(household)
    # The true source is the lineage's, never a clone's own id.
    assert sources == household["source_household_id"].tolist()
    assert paths == _expected_paths(household)
    # Every combination at every clone keys apart.
    assert len(set(_keys(sources, paths))) == 8 * k


@pytest.mark.parametrize("k", [1, 2])
def test_residential_arm_households_read_their_declared_branch(k):
    household = lineage_table(RESIDENTIAL_ROWS, clones=k)
    sources, paths = uk_household_clone_paths(household)
    assert paths == _expected_paths(household)
    keys = _keys(sources, paths)
    assert len(set(keys)) == len(keys) == k * len(RESIDENTIAL_ROWS)
    arms = [
        path
        for path in paths
        if any(step[0] == "cgt_residential_split" for step in path)
    ]
    assert len(arms) == 3 * k


def test_a_spine_without_the_clone_index_reads_no_geographic_branch():
    """A spine built before ``geographic_expand`` carries no clone index, and
    every path is read without the geographic branch, as clone 0's is."""

    sources, paths = uk_household_clone_paths(lineage_table(clones=None))
    assert (sources, paths) == uk_household_clone_paths(lineage_table(clones=1))
    assert not any(step[0] == "geographic_support" for path in paths for step in path)


def test_two_rows_of_one_lineage_read_one_path():
    """The reader never breaks a tie from ids: two households of one lineage
    read one source and path, so the key on it repeats, which the spine's
    ``geographic_identity`` stage refuses by count."""

    rows = list(LINEAGE)
    rows[2] = (11, 1, "frs", 0, False, 0, "LONDON")
    sources, paths = uk_household_clone_paths(lineage_table(rows))
    assert (sources[0], paths[0]) == (sources[2], paths[2]) == (1, ())


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("channel_index_disagree", "'spi_support_channel' marker and ordinal disagree"),
        ("negative_index", "'spi_support_channel' step refuses negative ordinals"),
        ("unknown_channel", "'spi_support_channel' marker holds a value other than"),
        ("null_lineage", "Household clone paths refuse null lineage values."),
        ("copy_flag_index_disagree", "'cgt_support_split' marker and ordinal disagree"),
        ("negative_copy_index", "'cgt_support_split' step refuses negative ordinals"),
    ],
)
def test_malformed_lineage_is_refused_by_name(defect, message):
    rows = list(LINEAGE)
    if defect == "channel_index_disagree":
        rows[2] = (11, 1, "spi", 0, False, 0, "LONDON")
    elif defect == "negative_index":
        rows[2] = (11, 1, "spi", -1, False, 0, "LONDON")
    elif defect == "unknown_channel":
        rows[2] = (11, 1, "puf", 1, False, 0, "LONDON")
    elif defect == "negative_copy_index":
        rows[6] = (1001, 2, "frs", 0, False, -1, "SCOTLAND")
    household = lineage_table(rows)
    if defect == "null_lineage":
        household["household_support_channel"] = pd.array(
            [pd.NA, *household["household_support_channel"][1:]], dtype="string"
        )
    if defect == "copy_flag_index_disagree":
        household["household_is_cgt_support_copy"] = np.asarray(
            [True, *household["household_is_cgt_support_copy"][1:]]
        )
    with pytest.raises(ValueError, match=message) as refused:
        uk_household_clone_paths(household)
    # A refusal names columns and steps, never an id or an ordinal of a row.
    assert not any(character.isdigit() for character in str(refused.value))


def test_residential_arm_flag_and_index_must_agree():
    household = lineage_table(RESIDENTIAL_ROWS)
    household.loc[
        household["household_id"] == 201, "household_is_cgt_residential_clone"
    ] = False
    with pytest.raises(
        ValueError, match="'cgt_residential_split' marker and ordinal disagree"
    ):
        uk_household_clone_paths(household)


def test_a_missing_lineage_column_and_a_non_int64_index_are_refused_by_name():
    clone_free = [column for column in UK_HOUSEHOLD_LINEAGE_COLUMNS if column != CLONE]
    for column in clone_free:
        with pytest.raises(ValueError) as refused:
            uk_household_clone_paths(lineage_table().drop(columns=[column]))
        assert str(refused.value) == (
            f"The household table lacks lineage column(s) {[column]} that its "
            "clone paths are read from."
        )
    narrowed = lineage_table()
    narrowed["cgt_support_copy_index"] = narrowed["cgt_support_copy_index"].astype(
        "int32"
    )
    with pytest.raises(ValueError) as refused:
        uk_household_clone_paths(narrowed)
    assert str(refused.value) == (
        "Household clone paths require an int64 cgt_support_copy_index."
    )
