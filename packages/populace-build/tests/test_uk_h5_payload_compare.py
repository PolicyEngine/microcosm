"""Disclosure control and classification in the #612 payload comparator.

The comparator's output is written to be posted on the tracking issue, so it
must never surface unit-record values: differences are column names, dtype
names, booleans, and threshold-guarded row counts. These tests plant sentinel
values in a differing pair and assert they appear nowhere in any output.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]

SENTINEL_VALUE = 987_654_321.0


def _comparator_module():
    path = ROOT / "tools" / "compare_uk_h5_payload.py"
    spec = importlib.util.spec_from_file_location("compare_uk_h5_payload", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMPARATOR = _comparator_module()


def _tables(*, weight_two: float = 20.0) -> dict[str, pd.DataFrame]:
    return {
        "person": pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "person_benunit_id": [10, 10, 20],
                "person_household_id": [100, 100, 200],
                "employment_income": [1000.0, 2000.0, 3000.0],
            }
        ),
        "benunit": pd.DataFrame({"benunit_id": [10, 20]}),
        "household": pd.DataFrame(
            {
                "household_id": [100, 200],
                "household_weight": [10.0, weight_two],
            }
        ),
    }


def _write(path: Path, tables: dict[str, pd.DataFrame], *, attr: str = "design"):
    h5py = pytest.importorskip("h5py")
    with pd.HDFStore(path, mode="w") as store:
        for name, table in tables.items():
            store.put(name, table, format="table", data_columns=True)
        store.put("time_period", pd.Series(["2023"]), format="table")
    with h5py.File(path, mode="r+") as file:
        file.attrs["populace_household_weight_kind"] = attr
    return path


def test_identical_payloads_compare_clean(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    left = _write(tmp_path / "left.h5", _tables())
    right = _write(tmp_path / "right.h5", _tables())

    report = COMPARATOR.compare_uk_h5_payload(left, right, minimum=10)

    assert report["payload_identical"] is True
    assert report["keys_equal"] is True
    assert all(table["payload_equal"] for table in report["tables"].values())
    assert report["root_attrs"]["attrs_with_differing_values"] == []


def test_differences_are_classified_masked_and_never_leak_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("tables")
    left = _write(tmp_path / "left.h5", _tables())
    right = _write(
        tmp_path / "right.h5", _tables(weight_two=SENTINEL_VALUE), attr="importance"
    )

    exit_code = COMPARATOR.main(
        [str(left), str(right), "--json-out", str(tmp_path / "diff.json")]
    )
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert exit_code == 1
    assert report["payload_identical"] is False
    household = report["tables"]["household"]
    assert household["value_mismatch_rows_by_column"] == {"household_weight": "< 10"}
    assert report["root_attrs"]["attrs_with_differing_values"] == [
        "populace_household_weight_kind"
    ]
    # The canary: the differing unit-record value appears in no output channel.
    for channel in (
        captured.out,
        captured.err,
        (tmp_path / "diff.json").read_text(),
    ):
        assert str(int(SENTINEL_VALUE)) not in channel
        assert "987654321" not in channel


def test_unsafe_configurations_exit_two(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    left = _write(tmp_path / "left.h5", _tables())

    assert COMPARATOR.main([str(left), str(left)]) == 2
    right = _write(tmp_path / "right.h5", _tables())
    assert COMPARATOR.main([str(left), str(right), "--sdc-minimum-count", "1"]) == 2
    assert (
        COMPARATOR.main([str(left), str(right), "--json-out", str(right)]) == 2
    )


def test_structural_differences_reported_by_name_only(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    left = _write(tmp_path / "left.h5", _tables())
    reordered = _tables()
    reordered["household"] = reordered["household"][
        ["household_weight", "household_id"]
    ]
    right = _write(tmp_path / "right.h5", reordered)

    report = COMPARATOR.compare_uk_h5_payload(left, right, minimum=10)

    assert report["payload_identical"] is False
    assert report["tables"]["household"]["column_order_equal"] is False
    assert report["tables"]["person"]["payload_equal"] is True
