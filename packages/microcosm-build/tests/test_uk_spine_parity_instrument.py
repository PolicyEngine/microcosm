"""The whole-spine parity instrument (#686).

The swap decision rests on this tool, so the tests pin the two properties that
make its verdict worth anything: an unsigned difference must fail, and the
verdict must not be manufacturable — not by aliasing the candidate onto the
reference, and not by letting the register drift into a blanket amnesty.
"""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from importlib.resources import files
from pathlib import Path

import pytest

_TOOL_PATH = Path(__file__).resolve().parents[3] / "tools" / "verify_uk_spine_parity.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("verify_uk_spine_parity", _TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_payload() -> dict:
    return json.loads(
        files("microcosm.build.uk")
        .joinpath("efrs_parity_reference.json")
        .read_text(encoding="utf-8")
    )


def _candidate_from_reference(**mutations) -> dict:
    """A candidate extraction that matches the committed reference exactly.

    The candidate's own source identity is deliberately different — the tool
    refuses a candidate that claims the incumbent's bytes.
    """

    payload = deepcopy(_reference_payload())
    payload["source"] = {
        "filename": "microcosm_uk_2024.h5",
        "sha256": "a" * 64,
        "size_bytes": 123456,
        "vintage": "2024_25",
        "period": "2024",
    }
    for key, value in mutations.items():
        payload[key] = value
    return payload


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _register(tmp_path: Path, *entries: dict) -> Path:
    return _write(
        tmp_path / "register.json",
        {
            "schema_version": 1,
            "scope_note": "test register",
            "differences": list(entries),
        },
    )


def _entry(
    identifier: str,
    *,
    surface: str,
    columns: list[str],
    expectation: str = "column_differs",
) -> dict:
    return {
        "id": identifier,
        "class": "mechanism_change",
        "scope": {"surface": surface, "columns": columns, "entities": ["household"]},
        "expectation": expectation,
        "magnitude_evidence": "disclosure-safe magnitude statement",
        "evidence": "experiments/686-uk-spine-swap-receipts.md#r0",
        "adjudicator": "juaristi22",
        "adjudicated_on": "2026-08-22",
    }


def _first_household_column() -> str:
    payload = _reference_payload()
    for column, entity in payload["input_entities"].items():
        if entity == "household":
            return column
    raise AssertionError("reference carries no household column")


class TestVerdicts:
    def test_matching_candidate_is_parity(self, tmp_path: Path) -> None:
        tool = _load_tool()
        candidate = _write(tmp_path / "c.json", _candidate_from_reference())
        code = tool.main(
            ["--candidate-json", str(candidate), "--register", str(_register(tmp_path))]
        )
        assert code == 0

    def test_unsigned_share_difference_is_a_defect(self, tmp_path: Path) -> None:
        tool = _load_tool()
        column = _first_household_column()
        payload = _candidate_from_reference()
        payload["nonzero_shares"][column] = payload["nonzero_shares"][column] + 0.25
        candidate = _write(tmp_path / "c.json", payload)
        receipt = tmp_path / "receipt.json"

        code = tool.main(
            [
                "--candidate-json",
                str(candidate),
                "--register",
                str(_register(tmp_path)),
                "--receipt-json",
                str(receipt),
            ]
        )

        assert code == 1
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["verdict"] == "defect"
        assert column in report["unsigned_differences"]
        assert report["nonzero_shares"]["differing"][column]["signed_id"] is None

    def test_signed_share_difference_is_signed_parity(self, tmp_path: Path) -> None:
        tool = _load_tool()
        column = _first_household_column()
        payload = _candidate_from_reference()
        payload["nonzero_shares"][column] = payload["nonzero_shares"][column] + 0.25
        candidate = _write(tmp_path / "c.json", payload)
        register = _register(
            tmp_path,
            _entry("signed-column", surface="nonzero_shares", columns=[column]),
        )
        receipt = tmp_path / "receipt.json"

        code = tool.main(
            [
                "--candidate-json",
                str(candidate),
                "--register",
                str(register),
                "--receipt-json",
                str(receipt),
            ]
        )

        assert code == 0
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["verdict"] == "signed_parity"
        assert report["register"]["matched_ids"] == ["signed-column"]
        assert report["unsigned_differences"] == []

    def test_a_signature_on_another_surface_does_not_cover_the_share(
        self, tmp_path: Path
    ) -> None:
        # This is the property that keeps the water level entry from silently
        # covering a share regression on the same column.
        tool = _load_tool()
        column = _first_household_column()
        payload = _candidate_from_reference()
        payload["nonzero_shares"][column] = payload["nonzero_shares"][column] + 0.25
        candidate = _write(tmp_path / "c.json", payload)
        register = _register(
            tmp_path,
            _entry("weighted-only", surface="weighted_totals", columns=[column]),
        )

        assert (
            tool.main(
                [
                    "--candidate-json",
                    str(candidate),
                    "--register",
                    str(register),
                ]
            )
            == 1
        )

    def test_entity_count_mismatch_is_a_defect(self, tmp_path: Path) -> None:
        tool = _load_tool()
        payload = _candidate_from_reference()
        payload["entity_stats"]["household"]["records"] += 1
        candidate = _write(tmp_path / "c.json", payload)
        receipt = tmp_path / "receipt.json"

        code = tool.main(
            [
                "--candidate-json",
                str(candidate),
                "--register",
                str(_register(tmp_path)),
                "--receipt-json",
                str(receipt),
            ]
        )

        assert code == 1
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["entity_counts"]["household"]["equal"] is False
        assert "household" in report["unsigned_differences"]

    def test_missing_and_extra_columns_are_defects(self, tmp_path: Path) -> None:
        tool = _load_tool()
        column = _first_household_column()
        payload = _candidate_from_reference()
        del payload["nonzero_shares"][column]
        payload["nonzero_shares"]["a_brand_new_column"] = 0.5
        candidate = _write(tmp_path / "c.json", payload)
        receipt = tmp_path / "receipt.json"

        assert (
            tool.main(
                [
                    "--candidate-json",
                    str(candidate),
                    "--register",
                    str(_register(tmp_path)),
                    "--receipt-json",
                    str(receipt),
                ]
            )
            == 1
        )
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert column in report["nonzero_shares"]["missing_in_candidate"]
        assert "a_brand_new_column" in report["nonzero_shares"]["extra_in_candidate"]


class TestFences:
    def test_candidate_claiming_the_incumbent_bytes_is_refused(
        self, tmp_path: Path
    ) -> None:
        # A copied reference would make the comparison pass by construction.
        tool = _load_tool()
        payload = _candidate_from_reference()
        payload["source"]["sha256"] = _reference_payload()["source"]["sha256"]
        candidate = _write(tmp_path / "c.json", payload)

        assert (
            tool.main(
                [
                    "--candidate-json",
                    str(candidate),
                    "--register",
                    str(_register(tmp_path)),
                ]
            )
            == 2
        )

    def test_strict_fails_an_unused_register_entry(self, tmp_path: Path) -> None:
        tool = _load_tool()
        candidate = _write(tmp_path / "c.json", _candidate_from_reference())
        register = _register(
            tmp_path,
            _entry("never-matches", surface="nonzero_shares", columns=["nothing_here"]),
        )
        receipt = tmp_path / "receipt.json"

        assert (
            tool.main(
                [
                    "--candidate-json",
                    str(candidate),
                    "--register",
                    str(register),
                    "--strict",
                    "--receipt-json",
                    str(receipt),
                ]
            )
            == 1
        )
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["strict_failure"] is True
        assert report["register"]["unused_ids"] == ["never-matches"]
        # Without --strict the same run is a clean parity verdict.
        assert (
            tool.main(
                [
                    "--candidate-json",
                    str(candidate),
                    "--register",
                    str(register),
                ]
            )
            == 0
        )

    def test_one_sided_weighted_totals_is_refused(self, tmp_path: Path) -> None:
        tool = _load_tool()
        candidate = _write(tmp_path / "c.json", _candidate_from_reference())
        totals = _write(tmp_path / "t.json", {"identity": {}, "totals": {"a": 1.0}})

        assert (
            tool.main(
                [
                    "--candidate-json",
                    str(candidate),
                    "--register",
                    str(_register(tmp_path)),
                    "--reference-weighted-totals",
                    str(totals),
                ]
            )
            == 2
        )

    def test_aliased_weighted_totals_are_refused(self, tmp_path: Path) -> None:
        tool = _load_tool()
        candidate = _write(tmp_path / "c.json", _candidate_from_reference())
        totals = _write(tmp_path / "t.json", {"identity": {}, "totals": {"a": 1.0}})

        assert (
            tool.main(
                [
                    "--candidate-json",
                    str(candidate),
                    "--register",
                    str(_register(tmp_path)),
                    "--reference-weighted-totals",
                    str(totals),
                    "--candidate-weighted-totals",
                    str(totals),
                ]
            )
            == 2
        )

    def test_unreadable_candidate_yields_no_verdict(self, tmp_path: Path) -> None:
        tool = _load_tool()
        assert (
            tool.main(
                [
                    "--candidate-json",
                    str(tmp_path / "absent.json"),
                    "--register",
                    str(_register(tmp_path)),
                ]
            )
            == 2
        )


class TestWeightedTotals:
    def test_relative_deltas_are_reported_without_absolute_values(
        self, tmp_path: Path
    ) -> None:
        tool = _load_tool()
        candidate = _write(tmp_path / "c.json", _candidate_from_reference())
        left = _write(
            tmp_path / "ref.json",
            {"identity": {"filename": "incumbent"}, "totals": {"col": 100.0}},
        )
        right = _write(
            tmp_path / "cand.json",
            {"identity": {"filename": "candidate"}, "totals": {"col": 125.0}},
        )
        register = _register(
            tmp_path,
            _entry("totals-signed", surface="weighted_totals", columns=["col"]),
        )
        receipt = tmp_path / "receipt.json"

        code = tool.main(
            [
                "--candidate-json",
                str(candidate),
                "--register",
                str(register),
                "--reference-weighted-totals",
                str(left),
                "--candidate-weighted-totals",
                str(right),
                "--receipt-json",
                str(receipt),
            ]
        )

        assert code == 0
        report = json.loads(receipt.read_text(encoding="utf-8"))
        entry = report["weighted_totals"]["differing"]["col"]
        assert entry["relative_delta"] == pytest.approx(0.25)
        assert entry["signed_id"] == "totals-signed"
        # Absolute licensed totals must never reach the receipt.
        assert "100.0" not in json.dumps(report)
        assert "125.0" not in json.dumps(report)

    def test_unsigned_weighted_divergence_is_a_defect(self, tmp_path: Path) -> None:
        tool = _load_tool()
        candidate = _write(tmp_path / "c.json", _candidate_from_reference())
        left = _write(tmp_path / "ref.json", {"identity": {}, "totals": {"col": 100.0}})
        right = _write(
            tmp_path / "cand.json", {"identity": {}, "totals": {"col": 125.0}}
        )

        assert (
            tool.main(
                [
                    "--candidate-json",
                    str(candidate),
                    "--register",
                    str(_register(tmp_path)),
                    "--reference-weighted-totals",
                    str(left),
                    "--candidate-weighted-totals",
                    str(right),
                ]
            )
            == 1
        )
