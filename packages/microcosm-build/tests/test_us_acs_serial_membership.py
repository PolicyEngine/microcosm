"""Prepared serial lookup agrees with pandas; every archive row still validates."""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import acs_pums


@pytest.mark.parametrize("storage", ["python", "pyarrow"])
@pytest.mark.parametrize("na_value", [pd.NA, np.nan])
@pytest.mark.parametrize(
    "keys",
    [
        frozenset(),
        frozenset({"001", "nan", "é", "", "東京"}),
        frozenset({"001", None, pd.NA, float("nan")}),
        frozenset({"001", 1, b"nan"}),
        frozenset({"001", True}),
        frozenset({np.str_("001")}),
    ],
)
def test_lookup_matches_pandas_for_string_storage_and_missing_policies(
    storage, na_value, keys
):
    values = pd.Series(
        ["001", "1", "nan", None, "", "é", "東京", "missing", "001"],
        dtype=pd.StringDtype(storage=storage, na_value=na_value),
        index=pd.Index([9, 7, 7, 2, 1, 3, 4, 6, 0], name="source_row"),
        name="SERIALNO",
    )
    lookup = acs_pums._serial_lookup(keys)
    pd.testing.assert_series_equal(
        acs_pums._serial_isin(values, keys, lookup), values.isin(keys)
    )


@pytest.mark.parametrize("dtype", [object, "category"])
def test_unreviewed_values_keep_original_pandas_membership(dtype):
    values = pd.Series(["001", None, 1, True, "1"], dtype=dtype, name="SERIALNO")
    keys = frozenset({"001", "1"})
    pd.testing.assert_series_equal(
        acs_pums._serial_isin(values, keys, acs_pums._serial_lookup(keys)),
        values.isin(keys),
    )


def test_prepared_lookup_is_reused_without_caching_chunk_verdicts(monkeypatch):
    keys = frozenset({"0001", "0002"})
    lookup = acs_pums._serial_lookup(keys)
    values = pd.Series(["0001", "absent", None], dtype="string")
    expected = values.isin(keys)

    def forbid_original(*args, **kwargs):
        pytest.fail("Rebuilding the all-string membership set for each chunk")

    monkeypatch.setattr(pd.Series, "isin", forbid_original)
    pd.testing.assert_series_equal(
        acs_pums._serial_isin(values, keys, lookup), expected
    )
    values.iloc[0] = "changed"
    assert acs_pums._serial_isin(values, keys, lookup).tolist() == [False] * 3
    later = pd.Series(["0002", "0001"], dtype="string")
    assert acs_pums._serial_isin(later, keys, lookup).tolist() == [True, True]


def _archive(tmp_path, payload):
    path = tmp_path / "invented.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("psam_pusa.csv", payload)
    return path


def _read(path, *, chunksize=2):
    return acs_pums._read_archive(
        path,
        member_prefix="psam_pus",
        required=("SERIALNO", "AGEP"),
        optional=(),
        chunksize=chunksize,
        valid_serials=frozenset({"001", "002", "003"}),
        retained_serials=frozenset({"001", "003"}),
    )


@pytest.mark.parametrize("chunksize", [1, 2, 20])
def test_archive_result_and_order_match_original_isin(tmp_path, monkeypatch, chunksize):
    path = _archive(tmp_path, "SERIALNO,AGEP\n003,09\n002,42\n001,18\n003,10\n")
    actual, members = _read(path, chunksize=chunksize)
    monkeypatch.setattr(
        acs_pums, "_serial_isin", lambda values, keys, lookup: values.isin(keys)
    )
    expected, original_members = _read(path, chunksize=chunksize)
    pd.testing.assert_frame_equal(actual, expected)
    assert members == original_members
    assert actual.SERIALNO.tolist() == ["003", "001", "003"]
    assert actual.AGEP.tolist() == [9, 18, 10]


@pytest.mark.parametrize(
    "payload",
    [
        "SERIALNO,AGEP\n001,18\nBAD2,20\nBAD1,30\n",
        "SERIALNO,AGEP\n001,18\n,20\n",
        # Even an unselected household's age must be checked before selection.
        "SERIALNO,AGEP\n001,18\n002,unknown\n",
    ],
)
@pytest.mark.parametrize("chunksize", [1, 2, 20])
def test_bad_archive_refusal_and_examples_match_original(
    tmp_path, monkeypatch, payload, chunksize
):
    path = _archive(tmp_path, payload)
    with pytest.raises(ValueError) as actual:
        _read(path, chunksize=chunksize)
    monkeypatch.setattr(
        acs_pums, "_serial_isin", lambda values, keys, lookup: values.isin(keys)
    )
    with pytest.raises(ValueError) as original:
        _read(path, chunksize=chunksize)
    assert type(actual.value) is type(original.value)
    assert str(actual.value) == str(original.value)


def test_native_owner_accepts_the_reviewed_lookup_implementation():
    from microcosm.build.us_runtime import acs_native_coverage_binding as native

    producer = native._producer()
    assert (
        producer["preparation"]["modules"]["microcosm.build/us_runtime/acs_pums.py"]
        == native._ACCEPTED["acs_pums.py"]
    )


def test_native_owner_still_refuses_changed_parser_bytes(monkeypatch):
    from microcosm.build.us_runtime import acs_native_coverage_binding as native

    original = Path.read_bytes
    parser_path = Path(acs_pums.__file__).resolve()

    def changed(path):
        payload = original(path)
        return (
            payload + b"\n# changed source\n"
            if path.resolve() == parser_path
            else payload
        )

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(
        native.ACSNativeCoverageBindingError, match="UNREVIEWED_PREPARATION"
    ):
        native._producer()
