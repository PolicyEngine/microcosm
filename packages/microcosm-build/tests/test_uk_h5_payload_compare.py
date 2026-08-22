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
    h5py = pytest.importorskip("h5py")
    left = _write(tmp_path / "left.h5", _tables())
    right = _write(
        tmp_path / "right.h5", _tables(weight_two=SENTINEL_VALUE), attr="importance"
    )
    # The mass-log attr embeds weighted totals as JSON — the disclosure-
    # relevant attr shape. A differing value must surface by NAME only.
    with h5py.File(left, mode="r+") as file:
        file.attrs["populace_mass_log_json"] = '[{"new_total": 30.0}]'
    with h5py.File(right, mode="r+") as file:
        file.attrs["populace_mass_log_json"] = f'[{{"new_total": {SENTINEL_VALUE}}}]'

    exit_code = COMPARATOR.main(
        [str(left), str(right), "--json-out", str(tmp_path / "diff.json")]
    )
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert exit_code == 1
    assert report["payload_identical"] is False
    household = report["tables"]["household"]
    assert household["value_mismatch_rows_by_column"] == {"household_weight": "< 10"}
    assert sorted(report["root_attrs"]["attrs_with_differing_values"]) == [
        "populace_household_weight_kind",
        "populace_mass_log_json",
    ]
    # The canary: the differing unit-record value — in a column AND inside
    # the mass-log attr JSON — appears in no output channel.
    for channel in (
        captured.out,
        captured.err,
        (tmp_path / "diff.json").read_text(),
    ):
        assert "987654321" not in channel


def test_unsafe_configurations_exit_two(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    left = _write(tmp_path / "left.h5", _tables())

    assert COMPARATOR.main([str(left), str(left)]) == 2
    right = _write(tmp_path / "right.h5", _tables())
    assert COMPARATOR.main([str(left), str(right), "--sdc-minimum-count", "1"]) == 2
    assert COMPARATOR.main([str(left), str(right), "--json-out", str(right)]) == 2


def test_unreadable_artifact_exits_two_with_suppressed_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An I/O failure is not a 'payloads differ' verdict, and no exception
    text (which can embed licensed-data fragments) reaches any channel."""

    pytest.importorskip("tables")
    left = _write(tmp_path / "left.h5", _tables())
    missing = tmp_path / "missing.h5"

    exit_code = COMPARATOR.main([str(left), str(missing)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "exception text was suppressed" in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_sdc_floor_is_unlowerable_at_every_entry_point() -> None:
    """CD171 §5.2.1: the floor holds even for direct helper importers."""

    with pytest.raises(ValueError, match="at least 3"):
        COMPARATOR.sdc_count(1, minimum=1)
    with pytest.raises(ValueError, match="at least 3"):
        COMPARATOR.compare_tables(
            pd.DataFrame({"x": [1.0]}), pd.DataFrame({"x": [2.0]}), minimum=1
        )
    with pytest.raises(ValueError, match="at least 3"):
        COMPARATOR.compare_uk_h5_payload(Path("a.h5"), Path("b.h5"), minimum=0)


def test_categorical_dtypes_compare_beyond_their_string_names() -> None:
    """Two categorical dtypes both stringify to 'category'; a differing
    ordered flag (or category set) must still fail the dtype check."""

    left = pd.DataFrame(
        {"region": pd.Categorical(["a", "b"], categories=["a", "b"], ordered=False)}
    )
    right = pd.DataFrame(
        {"region": pd.Categorical(["a", "b"], categories=["a", "b"], ordered=True)}
    )

    report = COMPARATOR.compare_tables(left, right, minimum=10)

    assert report["payload_equal"] is False
    assert "region" in report["dtype_mismatches"]
    assert "beyond their string names" in report["dtype_mismatches"]["region"]["note"]


def test_report_binds_input_digests(tmp_path: Path) -> None:
    """The verdict is digest-bound so a committed receipt verifies offline."""

    pytest.importorskip("tables")
    import hashlib

    left = _write(tmp_path / "left.h5", _tables())
    right = _write(tmp_path / "right.h5", _tables())

    report = COMPARATOR.compare_uk_h5_payload(left, right, minimum=10)

    assert report["left_sha256"] == hashlib.sha256(left.read_bytes()).hexdigest()
    assert report["right_sha256"] == hashlib.sha256(right.read_bytes()).hexdigest()


def test_int64_values_above_float_precision_are_not_equated() -> None:
    """A float64 cast equates int64 values past 2**53 — the magnitude regime
    rowwise id multiplication grows toward. Raw integer comparison must not."""

    base = 2**53
    left = pd.DataFrame({"person_id": pd.array([base], dtype="int64")})
    right = pd.DataFrame({"person_id": pd.array([base + 1], dtype="int64")})

    report = COMPARATOR.compare_tables(left, right, minimum=10)

    assert report["payload_equal"] is False
    assert report["value_mismatch_rows_by_column"] == {"person_id": "< 10"}


def test_series_vs_dataframe_stored_kind_is_payload(tmp_path: Path) -> None:
    """A DataFrame stored where readers expect a Series corrupts what the
    loader parses; normalizing for comparison must not hide the difference."""

    pytest.importorskip("tables")
    left = _write(tmp_path / "left.h5", _tables())
    right = _write(tmp_path / "right.h5", _tables())
    with pd.HDFStore(right, mode="r+") as store:
        period_frame = pd.DataFrame({"time_period": ["2023"]})
        store.remove("time_period")
        store.put("time_period", period_frame, format="table")

    report = COMPARATOR.compare_uk_h5_payload(left, right, minimum=10)

    assert report["tables"]["time_period"]["stored_kind_equal"] is False
    assert report["payload_identical"] is False


def test_array_attrs_compare_on_raw_values_not_str(tmp_path: Path) -> None:
    """numpy truncates long arrays in str(); a mid-array difference must
    still be detected, reported by attribute name only."""

    pytest.importorskip("tables")
    h5py = pytest.importorskip("h5py")
    import numpy as np

    left = _write(tmp_path / "left.h5", _tables())
    right = _write(tmp_path / "right.h5", _tables())
    left_array = np.arange(2000)
    right_array = np.arange(2000)
    right_array[1000] = -1
    with h5py.File(left, mode="r+") as file:
        file.attrs["probe_array"] = left_array
    with h5py.File(right, mode="r+") as file:
        file.attrs["probe_array"] = right_array

    report = COMPARATOR.compare_uk_h5_payload(left, right, minimum=10)

    assert "probe_array" in report["root_attrs"]["attrs_with_differing_values"]
    assert report["payload_identical"] is False


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


def _register(tmp_path: Path, *entries: dict) -> Path:
    path = tmp_path / "register.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope_note": "test register",
                "differences": list(entries),
            }
        ),
        encoding="utf-8",
    )
    return path


def _entry(identifier: str, *, surface: str, columns: list[str]) -> dict:
    return {
        "id": identifier,
        "class": "mechanism_change",
        "scope": {"surface": surface, "columns": columns, "entities": ["household"]},
        "expectation": "column_differs",
        "magnitude_evidence": "disclosure-safe magnitude statement",
        "evidence": "experiments/686-uk-spine-swap-receipts.md#r0",
        "adjudicator": "juaristi22",
        "adjudicated_on": "2026-08-22",
    }


class TestStructureOnlyVerdict:
    """The #686 swap posture: same surface, differences only where signed."""

    def test_signed_value_difference_passes(self, tmp_path: Path) -> None:
        pytest.importorskip("tables")
        left = _write(tmp_path / "left.h5", _tables())
        right = _write(tmp_path / "right.h5", _tables(weight_two=SENTINEL_VALUE))
        register = _register(
            tmp_path,
            _entry(
                "weights-differ",
                surface="payload_column",
                columns=["household_weight"],
            ),
        )
        out = tmp_path / "report.json"

        code = COMPARATOR.main(
            [
                str(left),
                str(right),
                "--structure-only",
                "--signed-differences",
                str(register),
                "--json-out",
                str(out),
            ]
        )

        report = json.loads(out.read_text(encoding="utf-8"))
        assert code == 0
        assert report["verdict_mode"] == "structure_only"
        assert report["structure_equal"] is True
        assert report["structure_only_ok"] is True
        assert report["signed"]["matched_ids"] == ["weights-differ"]
        # The full-payload verdict is still reported, unchanged.
        assert report["payload_identical"] is False
        assert str(SENTINEL_VALUE) not in json.dumps(report)

    def test_unsigned_value_difference_fails(self, tmp_path: Path) -> None:
        pytest.importorskip("tables")
        left = _write(tmp_path / "left.h5", _tables())
        right = _write(tmp_path / "right.h5", _tables(weight_two=SENTINEL_VALUE))
        out = tmp_path / "report.json"

        code = COMPARATOR.main(
            [
                str(left),
                str(right),
                "--structure-only",
                "--signed-differences",
                str(_register(tmp_path)),
                "--json-out",
                str(out),
            ]
        )

        report = json.loads(out.read_text(encoding="utf-8"))
        assert code == 1
        assert report["structure_equal"] is True
        assert report["structure_only_ok"] is False
        assert report["signed"]["unsigned_columns"] == ["household.household_weight"]

    def test_structural_difference_fails_even_when_signed(self, tmp_path: Path) -> None:
        # A signature excuses differing values, never a differing surface.
        pytest.importorskip("tables")
        left = _write(tmp_path / "left.h5", _tables())
        extra = _tables()
        extra["household"] = extra["household"].assign(surprise_column=[1.0, 2.0])
        right = _write(tmp_path / "right.h5", extra)
        register = _register(
            tmp_path,
            _entry("surprise", surface="payload_column", columns=["surprise_column"]),
        )
        out = tmp_path / "report.json"

        code = COMPARATOR.main(
            [
                str(left),
                str(right),
                "--structure-only",
                "--signed-differences",
                str(register),
                "--json-out",
                str(out),
            ]
        )

        report = json.loads(out.read_text(encoding="utf-8"))
        assert code == 1
        assert report["structure_equal"] is False

    def test_identical_artifacts_pass_with_no_signatures_used(
        self, tmp_path: Path
    ) -> None:
        pytest.importorskip("tables")
        left = _write(tmp_path / "left.h5", _tables())
        right = _write(tmp_path / "right.h5", _tables())
        out = tmp_path / "report.json"

        code = COMPARATOR.main(
            [
                str(left),
                str(right),
                "--structure-only",
                "--signed-differences",
                str(_register(tmp_path)),
                "--json-out",
                str(out),
            ]
        )

        report = json.loads(out.read_text(encoding="utf-8"))
        assert code == 0
        assert report["structure_only_ok"] is True
        assert report["signed"]["matched_ids"] == []

    def test_signed_differences_without_structure_only_is_refused(
        self, tmp_path: Path
    ) -> None:
        pytest.importorskip("tables")
        left = _write(tmp_path / "left.h5", _tables())
        right = _write(tmp_path / "right.h5", _tables())

        assert (
            COMPARATOR.main(
                [
                    str(left),
                    str(right),
                    "--signed-differences",
                    str(_register(tmp_path)),
                ]
            )
            == 2
        )

    def test_root_attr_difference_must_be_signed(self, tmp_path: Path) -> None:
        pytest.importorskip("tables")
        left = _write(tmp_path / "left.h5", _tables())
        right = _write(tmp_path / "right.h5", _tables(), attr="importance")
        out = tmp_path / "report.json"

        code = COMPARATOR.main(
            [
                str(left),
                str(right),
                "--structure-only",
                "--signed-differences",
                str(_register(tmp_path)),
                "--json-out",
                str(out),
            ]
        )

        report = json.loads(out.read_text(encoding="utf-8"))
        assert code == 1
        assert report["signed"]["unsigned_root_attrs"] == [
            "populace_household_weight_kind"
        ]
