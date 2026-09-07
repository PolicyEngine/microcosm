"""Invented ACS loader/relationship contracts, without constructing tax units."""

from __future__ import annotations

import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def acs():
    # The bounded developer runner supplies exact AST-loaded source bodies to
    # avoid the country package initializer. Normal CI imports the real module.
    if "_acs_performance_scope" in sys.modules:
        return sys.modules["_acs_performance_scope"]
    from microcosm.build.us_runtime import acs_pums

    return acs_pums


def archive(path, serials):
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as z:
        z.writestr(
            "psam_pusa.csv",
            pd.DataFrame({"SERIALNO": serials, "SPORDER": range(len(serials))}).to_csv(
                index=False
            ),
        )
    return path


def read(acs, path, valid, retained, *, chunksize=2):
    return acs._read_archive(
        path,
        member_prefix="psam_pus",
        required=("SERIALNO", "SPORDER"),
        optional=(),
        chunksize=chunksize,
        valid_serials=valid,
        retained_serials=retained,
    )


class CountedSet(frozenset):
    def __iter__(self):
        self.iterations = getattr(self, "iterations", 0) + 1
        return super().__iter__()


def test_household_inventory_built_once_per_archive(acs, tmp_path):
    values = ["0001", "1", "HU01", "HU1", "0001", "1", "HU01", "HU1"]
    valid, retained = CountedSet(values), CountedSet({"0001", "HU01"})
    result, _ = read(acs, archive(tmp_path / "p.zip", values), valid, retained)
    assert result["SERIALNO"].tolist() == ["0001", "HU01", "0001", "HU01"]
    assert valid.iterations == retained.iterations == 1


@pytest.mark.parametrize("chunksize", [1, 2, 5, 100])
def test_lexical_keys_selection_and_source_order(acs, tmp_path, chunksize):
    values = ["01", "1", "HU001", "HU1", " A", "A", "01", "HU001"]
    result, members = read(
        acs,
        archive(tmp_path / "p.zip", values),
        frozenset(values),
        frozenset({"01", "HU001", " A"}),
        chunksize=chunksize,
    )
    assert members == ["psam_pusa.csv"]
    assert result["SERIALNO"].tolist() == ["01", "HU001", " A", "01", "HU001"]
    assert result["SPORDER"].tolist() == [0, 2, 4, 6, 7]
    assert isinstance(result["SERIALNO"].dtype, pd.StringDtype)
    assert result.index.equals(pd.RangeIndex(5))


@pytest.mark.parametrize("orphan", ["1", "HU1", "", "missing"])
def test_unretained_orphan_still_refuses(acs, tmp_path, orphan):
    with pytest.raises(ValueError, match="missing from the household archive"):
        read(
            acs,
            archive(tmp_path / "p.zip", ["01", "HU001", orphan]),
            frozenset({"01", "HU001"}),
            frozenset({"01"}),
        )


def test_empty_and_absent_selection(acs, tmp_path):
    path = archive(tmp_path / "p.zip", ["01", "1"])
    empty, _ = read(acs, path, frozenset({"01", "1"}), frozenset())
    assert empty.empty and empty.columns.tolist() == ["SERIALNO", "SPORDER"]
    full, _ = read(acs, path, None, None)
    assert full["SERIALNO"].tolist() == ["01", "1"]


def people(relationships, *, households=None, sexes=None, marital=None):
    n = len(relationships)
    return pd.DataFrame(
        {
            "household_id": np.array(households or [1] * n, dtype=np.int64),
            "SPORDER": np.arange(1, n + 1, dtype=np.int64),
            "AGEP": np.full(n, 40, dtype=np.int64),
            "MAR": np.array(marital or [1] * n, dtype=np.int64),
            "RELSHIPP": np.array(relationships, dtype=np.int64),
            "SEX": np.array(sexes or [1] * n, dtype=np.int64),
            "native_missing": pd.array([None] * n, dtype="Int64"),
        }
    )


def test_relationship_outputs_use_no_pandas_loc_writes(acs, monkeypatch):
    from pandas.core.indexing import _LocIndexer

    writes = []
    original = _LocIndexer.__setitem__

    def record(self, key, value):
        writes.append(key)
        return original(self, key, value)

    monkeypatch.setattr(_LocIndexer, "__setitem__", record)
    result = acs._with_structural_columns(people([20, 21, 25], sexes=[1, 2, 1]))
    assert result["A_SPOUSE"].tolist() == [2, 1, 0]
    assert result["PEPAR1"].tolist() == [0, 0, 1]
    assert result["PEPAR2"].tolist() == [0, 0, 2]
    assert result["A_EXPRRP"].tolist() == [1, 4, 5]
    assert writes == []


@pytest.mark.parametrize("role", list(range(21, 37)))
def test_every_housing_relationship_preserves_source_and_expected_recode(acs, role):
    source = people([20, role], sexes=[1, 2])
    before = source.copy(deep=True)
    result = acs._with_structural_columns(source)
    pd.testing.assert_frame_equal(source, before, check_exact=True)
    pd.testing.assert_frame_equal(result[source.columns], source, check_exact=True)
    expected = 4 if role in (21, 23) else acs._ACS_TO_CPS_RELATIONSHIP[role]
    assert result["A_EXPRRP"].tolist() == [
        1 if role in acs._ACS_RELATED_CODES else 2,
        expected,
    ]
    assert result["PEPAR1"].tolist() == [0, 1 if role in (25, 26, 27) else 0]
    assert result["PEPAR2"].tolist() == [0, 0]
    assert all(
        result[name].dtype == np.dtype("int64")
        for name in (
            "PH_SEQ",
            "A_LINENO",
            "A_AGE",
            "A_MARITL",
            "A_SPOUSE",
            "PEPAR1",
            "PEPAR2",
            "A_EXPRRP",
        )
    )
    assert result.columns.tolist() == source.columns.tolist() + [
        "PH_SEQ",
        "A_LINENO",
        "A_AGE",
        "A_MARITL",
        "A_SPOUSE",
        "PEPAR1",
        "PEPAR2",
        "A_EXPRRP",
    ]


@pytest.mark.parametrize("roles", [[37], [38], [37, 38]])
def test_gq_keeps_existing_no_reference_policy(acs, roles):
    result = acs._with_structural_columns(people(roles))
    assert result["A_EXPRRP"].tolist() == [14] * len(roles)
    assert not result[["A_SPOUSE", "PEPAR1", "PEPAR2"]].to_numpy().any()


def test_unsorted_households_and_preexisting_output_order(acs):
    source = people([20, 20, 21, 25], households=[9, 3, 9, 3])
    source.insert(1, "PEPAR1", [-1] * 4)
    result = acs._with_structural_columns(source)
    assert result["A_SPOUSE"].tolist() == [3, 0, 1, 0]
    assert result["PEPAR1"].tolist() == [0, 0, 0, 2]
    assert result.columns[1] == "PEPAR1"
    assert result.index.equals(source.index)


@pytest.mark.parametrize(
    ("roles", "marital", "error"),
    [
        ([25], None, "exactly one"),
        ([20, 20], None, "exactly one"),
        ([20, 21, 23], None, "multiple spouse"),
        ([20, 21], [1, 5], "MAR=1"),
        ([20, 99], None, "RELSHIPP contains unsupported"),
        ([20, 99], [6, 1], "MAR contains unsupported"),
    ],
)
def test_relationship_refusal_order(acs, roles, marital, error):
    with pytest.raises(ValueError, match=error):
        acs._with_structural_columns(people(roles, marital=marital))


@pytest.mark.parametrize("field", ["SPORDER", "AGEP", "MAR", "RELSHIPP", "SEX"])
def test_missing_structural_value_refuses(acs, field):
    source = people([20])
    source[field] = pd.array([None], dtype="Int64")
    with pytest.raises(ValueError, match=f"{field!r} contains blanks"):
        acs._with_structural_columns(source)
