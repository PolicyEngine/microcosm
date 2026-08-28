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
    direction: str = "candidate_above",
    max_abs_delta: float = 1.0,
) -> dict:
    entry = {
        "id": identifier,
        "class": "mechanism_change",
        "scope": {"surface": surface, "columns": columns, "entities": ["household"]},
        "expectation": expectation,
        "magnitude_evidence": "disclosure-safe magnitude statement",
        "evidence": "experiments/686-uk-spine-swap-receipts.md#r0",
        "adjudicator": "juaristi22",
        "adjudicated_on": "2026-08-22",
    }
    if surface == "nonzero_shares" and expectation == "column_differs":
        reference = _reference_payload()["nonzero_shares"]
        entry["quantitative"] = {
            "shares": {
                column: {
                    "incumbent_share": reference.get(column, 0.0),
                    "direction": direction,
                    "max_abs_delta": max_abs_delta,
                }
                for column in columns
            }
        }
    elif surface == "entity_counts" and expectation == "count_differs":
        entry["quantitative"] = {"expected_deltas": {column: 1 for column in columns}}
    elif surface == "weighted_totals":
        entry["quantitative"] = {"weighted_totals": {"expected_columns": columns}}
    else:
        entry["quantitative"] = {"structural": {"expected_columns": columns}}
    return entry


def _first_household_column() -> str:
    payload = _reference_payload()
    for column, entity in payload["input_entities"].items():
        if entity == "household":
            return column
    raise AssertionError("reference carries no household column")


def _weighted_identity(sha256: str) -> dict[str, str]:
    return {"filename": "artifact.h5", "sha256": sha256}


def _reference_weighted_identity() -> dict[str, str]:
    return _weighted_identity(_reference_payload()["source"]["sha256"])


def _candidate_weighted_identity() -> dict[str, str]:
    return _weighted_identity("a" * 64)


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

    def test_strict_treats_a_dormant_surface_entry_as_dormant_not_unused(
        self, tmp_path: Path
    ) -> None:
        # The weighted-totals surface stays unexamined until there is a
        # calibrated candidate, so an entry scoped to it has had no chance to
        # match. Failing --strict for that would make the swap-acceptance
        # posture impossible to satisfy before calibration.
        tool = _load_tool()
        candidate = _write(tmp_path / "c.json", _candidate_from_reference())
        register = _register(
            tmp_path,
            _entry("totals-only", surface="weighted_totals", columns=["col"]),
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
            == 0
        )
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["strict_failure"] is False
        assert report["register"]["unused_ids"] == []
        assert report["register"]["dormant_ids"] == ["totals-only"]
        assert "weighted_totals" not in report["register"]["compared_surfaces"]

    def test_dormancy_is_not_a_loophole_once_the_surface_is_compared(
        self, tmp_path: Path
    ) -> None:
        # Same entry, same register — but this run supplies the sidecars, so
        # the surface is examined and a signature that matches nothing on it is
        # register rot again.
        tool = _load_tool()
        candidate = _write(tmp_path / "c.json", _candidate_from_reference())
        left = _write(
            tmp_path / "ref.json",
            {"identity": _reference_weighted_identity(), "totals": {"col": 100.0}},
        )
        right = _write(
            tmp_path / "cand.json",
            {"identity": _candidate_weighted_identity(), "totals": {"col": 100.0}},
        )
        register = _register(
            tmp_path,
            _entry("totals-only", surface="weighted_totals", columns=["col"]),
        )
        receipt = tmp_path / "receipt.json"

        assert (
            tool.main(
                [
                    "--candidate-json",
                    str(candidate),
                    "--register",
                    str(register),
                    "--reference-weighted-totals",
                    str(left),
                    "--candidate-weighted-totals",
                    str(right),
                    "--strict",
                    "--receipt-json",
                    str(receipt),
                ]
            )
            == 1
        )
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["register"]["unused_ids"] == ["totals-only"]
        assert report["register"]["dormant_ids"] == []

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
        left = _write(
            tmp_path / "ref.json",
            {"identity": _reference_weighted_identity(), "totals": {"col": 100.0}},
        )
        right = _write(
            tmp_path / "cand.json",
            {"identity": _candidate_weighted_identity(), "totals": {"col": 125.0}},
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

    def test_strict_weighted_totals_reference_only_key_is_a_defect(
        self, tmp_path: Path
    ) -> None:
        tool = _load_tool()
        candidate = _write(tmp_path / "c.json", _candidate_from_reference())
        left = _write(
            tmp_path / "ref.json",
            {
                "identity": _reference_weighted_identity(),
                "totals": {"kept": 100.0, "omitted": 1.0},
            },
        )
        right = _write(
            tmp_path / "cand.json",
            {"identity": _candidate_weighted_identity(), "totals": {"kept": 100.0}},
        )
        receipt = tmp_path / "receipt.json"

        code = tool.main(
            [
                "--candidate-json",
                str(candidate),
                "--register",
                str(_register(tmp_path)),
                "--reference-weighted-totals",
                str(left),
                "--candidate-weighted-totals",
                str(right),
                "--strict",
                "--receipt-json",
                str(receipt),
            ]
        )

        assert code == 1
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["verdict"] == "defect"
        assert report["weighted_totals"]["only_in_reference"] == ["omitted"]
        assert "omitted" in report["unsigned_differences"]

    def test_strict_weighted_totals_candidate_only_key_is_a_defect(
        self, tmp_path: Path
    ) -> None:
        tool = _load_tool()
        candidate = _write(tmp_path / "c.json", _candidate_from_reference())
        left = _write(
            tmp_path / "ref.json",
            {"identity": _reference_weighted_identity(), "totals": {"kept": 100.0}},
        )
        right = _write(
            tmp_path / "cand.json",
            {
                "identity": _candidate_weighted_identity(),
                "totals": {"extra": 1.0, "kept": 100.0},
            },
        )
        receipt = tmp_path / "receipt.json"

        code = tool.main(
            [
                "--candidate-json",
                str(candidate),
                "--register",
                str(_register(tmp_path)),
                "--reference-weighted-totals",
                str(left),
                "--candidate-weighted-totals",
                str(right),
                "--strict",
                "--receipt-json",
                str(receipt),
            ]
        )

        assert code == 1
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["verdict"] == "defect"
        assert report["weighted_totals"]["only_in_candidate"] == ["extra"]
        assert "extra" in report["unsigned_differences"]

    def test_strict_weighted_totals_missing_candidate_identity_is_refused(
        self, tmp_path: Path
    ) -> None:
        tool = _load_tool()
        candidate = _write(tmp_path / "c.json", _candidate_from_reference())
        left = _write(
            tmp_path / "ref.json",
            {"identity": _reference_weighted_identity(), "totals": {"kept": 100.0}},
        )
        right = _write(tmp_path / "cand.json", {"totals": {"kept": 100.0}})

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
                    "--strict",
                ]
            )
            == 2
        )

    def test_strict_weighted_totals_cross_artifact_identity_is_refused(
        self, tmp_path: Path
    ) -> None:
        tool = _load_tool()
        candidate = _write(tmp_path / "c.json", _candidate_from_reference())
        left = _write(
            tmp_path / "ref.json",
            {"identity": _reference_weighted_identity(), "totals": {"kept": 100.0}},
        )
        right = _write(
            tmp_path / "cand.json",
            {"identity": _weighted_identity("b" * 64), "totals": {"kept": 100.0}},
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
                    "--strict",
                ]
            )
            == 2
        )


class TestAcceptanceBand:
    """The band decides what must be adjudicated, never what is reported."""

    def test_an_in_band_difference_needs_no_signature(self, tmp_path: Path) -> None:
        tool = _load_tool()
        column = _first_household_column()
        payload = _candidate_from_reference()
        payload["nonzero_shares"][column] += 0.01
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

        assert code == 0
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["verdict"] == "parity"
        assert report["unsigned_differences"] == []
        # Reported, not dropped: the reader still sees the movement.
        assert column in report["nonzero_shares"]["within_band"]
        assert column not in report["nonzero_shares"]["differing"]
        assert report["nonzero_shares"]["within_band"][column][
            "delta"
        ] == pytest.approx(0.01)
        assert report["nonzero_shares"]["within_band_max_abs_delta"] == pytest.approx(
            0.01
        )

    def test_a_difference_just_beyond_the_band_still_signs(
        self, tmp_path: Path
    ) -> None:
        tool = _load_tool()
        column = _first_household_column()
        payload = _candidate_from_reference()
        payload["nonzero_shares"][column] += 0.021
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
            == 1
        )

    def test_a_zero_band_restores_the_exact_grain_check(self, tmp_path: Path) -> None:
        tool = _load_tool()
        column = _first_household_column()
        payload = _candidate_from_reference()
        payload["nonzero_shares"][column] += 0.01
        candidate = _write(tmp_path / "c.json", payload)

        assert (
            tool.main(
                [
                    "--candidate-json",
                    str(candidate),
                    "--register",
                    str(_register(tmp_path)),
                    "--share-band",
                    "0",
                ]
            )
            == 1
        )

    def test_the_band_never_covers_a_structural_difference(
        self, tmp_path: Path
    ) -> None:
        # A column that appears or vanishes is not a magnitude, so no band can
        # absorb it; nor can one absorb an entity-count difference.
        tool = _load_tool()
        payload = _candidate_from_reference()
        payload["nonzero_shares"]["a_column_the_incumbent_never_had"] = 0.000001
        payload["entity_stats"]["household"]["records"] += 1
        candidate = _write(tmp_path / "c.json", payload)
        receipt = tmp_path / "receipt.json"

        assert (
            tool.main(
                [
                    "--candidate-json",
                    str(candidate),
                    "--register",
                    str(_register(tmp_path)),
                    "--share-band",
                    "0.9",
                    "--receipt-json",
                    str(receipt),
                ]
            )
            == 1
        )
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert "a_column_the_incumbent_never_had" in report["unsigned_differences"]
        assert "household" in report["unsigned_differences"]

    def test_an_out_of_range_band_yields_no_verdict(self, tmp_path: Path) -> None:
        tool = _load_tool()
        candidate = _write(tmp_path / "c.json", _candidate_from_reference())
        for bad in ("-0.01", "1.0", "5"):
            assert (
                tool.main(
                    [
                        "--candidate-json",
                        str(candidate),
                        "--register",
                        str(_register(tmp_path)),
                        "--share-band",
                        bad,
                    ]
                )
                == 2
            )

    def test_strict_refuses_a_non_contract_band(self, tmp_path: Path) -> None:
        tool = _load_tool()
        candidate = _write(tmp_path / "c.json", _candidate_from_reference())

        assert (
            tool.main(
                [
                    "--candidate-json",
                    str(candidate),
                    "--register",
                    str(_register(tmp_path)),
                    "--strict",
                    "--share-band",
                    "0.9",
                ]
            )
            == 2
        )

    def test_diagnostic_non_contract_band_is_not_parity(
        self, tmp_path: Path
    ) -> None:
        tool = _load_tool()
        candidate = _write(tmp_path / "c.json", _candidate_from_reference())
        receipt = tmp_path / "receipt.json"

        code = tool.main(
            [
                "--candidate-json",
                str(candidate),
                "--register",
                str(_register(tmp_path)),
                "--share-band",
                "0.9",
                "--receipt-json",
                str(receipt),
            ]
        )

        assert code == 0
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["verdict"] == "diagnostic"
        assert report["share_band"] == {"contract": 0.02, "effective": 0.9}

    def test_strict_contract_band_behaviour_is_unchanged(
        self, tmp_path: Path
    ) -> None:
        tool = _load_tool()
        candidate = _write(tmp_path / "c.json", _candidate_from_reference())
        receipt = tmp_path / "receipt.json"

        code = tool.main(
            [
                "--candidate-json",
                str(candidate),
                "--register",
                str(_register(tmp_path)),
                "--strict",
                "--share-band",
                "0.02",
                "--receipt-json",
                str(receipt),
            ]
        )

        assert code == 0
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["verdict"] == "parity"
        assert report["strict_failure"] is False


class TestReviewFindings:
    """Regressions for the review findings on the proof machinery (#747).

    Each of these let the instrument reach a wrong verdict — or no verdict —
    for reasons unrelated to whether the spine's data is right.
    """

    def test_a_candidate_without_a_source_identity_is_refused(
        self, tmp_path: Path
    ) -> None:
        # The anti-self-comparison fence compares identities, so an anonymous
        # candidate would pass it vacuously.
        payload = _candidate_from_reference()
        del payload["source"]
        candidate = _write(tmp_path / "c.json", payload)
        assert (
            tool_main := _load_tool().main(
                [
                    "--candidate-json",
                    str(candidate),
                    "--register",
                    str(_register(tmp_path)),
                ]
            )
        ) == 2, tool_main

    def test_a_candidate_without_a_sha256_is_refused(self, tmp_path: Path) -> None:
        payload = _candidate_from_reference()
        payload["source"] = {"filename": "microcosm_uk_2024.h5"}
        candidate = _write(tmp_path / "c.json", payload)
        assert (
            _load_tool().main(
                [
                    "--candidate-json",
                    str(candidate),
                    "--register",
                    str(_register(tmp_path)),
                ]
            )
            == 2
        )

    def test_a_zero_reference_total_still_reaches_a_verdict(
        self, tmp_path: Path
    ) -> None:
        # A new-in-candidate column with a nonzero weighted total has no
        # finite relative delta; reporting float("inf") made json.dumps
        # refuse the receipt and turned a real divergence into "no verdict".
        tool = _load_tool()
        candidate = _write(tmp_path / "c.json", _candidate_from_reference())
        left = _write(tmp_path / "ref.json", {"identity": {}, "totals": {"col": 0.0}})
        right = _write(
            tmp_path / "cand.json", {"identity": {}, "totals": {"col": 125.0}}
        )
        receipt = tmp_path / "receipt.json"

        code = tool.main(
            [
                "--candidate-json",
                str(candidate),
                "--register",
                str(_register(tmp_path)),
                "--reference-weighted-totals",
                str(left),
                "--candidate-weighted-totals",
                str(right),
                "--receipt-json",
                str(receipt),
            ]
        )

        assert code == 1
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["verdict"] == "defect"
        entry = report["weighted_totals"]["differing"]["col"]
        assert entry["reference_total_zero"] is True
        assert entry["relative_delta"] is None

    def test_strict_accepts_an_entry_matching_a_within_band_column(
        self, tmp_path: Path
    ) -> None:
        # A signed column whose divergence has since shrunk under the band is
        # still a matched entry: the difference it adjudicates is real and
        # reported. --strict exists to catch entries matching nothing at all.
        tool = _load_tool()
        column = _first_household_column()
        payload = _candidate_from_reference()
        payload["nonzero_shares"][column] += 0.01
        candidate = _write(tmp_path / "c.json", payload)
        register = _register(
            tmp_path,
            _entry("shrunk-since-signing", surface="nonzero_shares", columns=[column]),
        )
        receipt = tmp_path / "receipt.json"

        code = tool.main(
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

        assert code == 0
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["strict_failure"] is False
        assert report["register"]["unused_ids"] == []
        assert report["register"]["matched_ids"] == ["shrunk-since-signing"]
        assert column in report["nonzero_shares"]["within_band"]

    def test_a_structural_expectation_does_not_sign_a_value_divergence(
        self, tmp_path: Path
    ) -> None:
        tool = _load_tool()
        column = _first_household_column()
        payload = _candidate_from_reference()
        payload["nonzero_shares"][column] += 0.25
        candidate = _write(tmp_path / "c.json", payload)
        register = _register(
            tmp_path,
            _entry(
                "net-new-only",
                surface="nonzero_shares",
                columns=[column],
                expectation="column_missing_in_reference",
            ),
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

    def test_signed_count_omitted_from_candidate_is_a_defect(
        self, tmp_path: Path
    ) -> None:
        tool = _load_tool()
        payload = _candidate_from_reference()
        del payload["entity_stats"]["person"]
        candidate = _write(tmp_path / "c.json", payload)
        register = _register(
            tmp_path,
            _entry(
                "counts-signed",
                surface="entity_counts",
                columns=["person"],
                expectation="count_differs",
            ),
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

        assert code == 1
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["verdict"] == "defect"
        assert report["entity_counts"]["person"]["signed_id"] is None
        assert "person" in report["unsigned_differences"]

    def test_signed_count_with_wrong_delta_is_a_defect(self, tmp_path: Path) -> None:
        tool = _load_tool()
        payload = _candidate_from_reference()
        payload["entity_stats"]["person"]["records"] = 1
        candidate = _write(tmp_path / "c.json", payload)
        register = _register(
            tmp_path,
            _entry(
                "counts-signed",
                surface="entity_counts",
                columns=["person"],
                expectation="count_differs",
            ),
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

        assert code == 1
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["verdict"] == "defect"
        assert report["entity_counts"]["person"]["signed_id"] is None
        assert "person" in report["unsigned_differences"]

    def test_signed_share_with_reversed_direction_is_a_defect(
        self, tmp_path: Path
    ) -> None:
        tool = _load_tool()
        column = "water_and_sewerage_charges"
        payload = _candidate_from_reference()
        payload["nonzero_shares"][column] = (
            _reference_payload()["nonzero_shares"][column] - 0.1
        )
        candidate = _write(tmp_path / "c.json", payload)
        register = _register(
            tmp_path,
            _entry(
                "water-signed",
                surface="nonzero_shares",
                columns=[column],
                direction="candidate_above",
                max_abs_delta=0.100897,
            ),
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

        assert code == 1
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["verdict"] == "defect"
        assert report["nonzero_shares"]["differing"][column]["signed_id"] is None
        assert column in report["unsigned_differences"]

    def test_signed_share_beyond_magnitude_is_a_defect(
        self, tmp_path: Path
    ) -> None:
        tool = _load_tool()
        column = "water_and_sewerage_charges"
        payload = _candidate_from_reference()
        payload["nonzero_shares"][column] = (
            _reference_payload()["nonzero_shares"][column] + 0.2
        )
        candidate = _write(tmp_path / "c.json", payload)
        register = _register(
            tmp_path,
            _entry(
                "water-signed",
                surface="nonzero_shares",
                columns=[column],
                direction="candidate_above",
                max_abs_delta=0.100897,
            ),
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

        assert code == 1
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["verdict"] == "defect"
        assert report["nonzero_shares"]["differing"][column]["signed_id"] is None
        assert column in report["unsigned_differences"]


class TestWeightedTotalsRegisterAccounting:
    """A totals-scoped entry counts as matched, not as register rot.

    Latent while the surface is dormant, and it bites exactly when the
    surface is un-dormanted for a calibrated candidate — which is the moment
    the accounting starts mattering.
    """

    def test_a_totals_entry_that_matched_is_not_reported_unused(
        self, tmp_path: Path
    ) -> None:
        tool = _load_tool()
        candidate = _write(tmp_path / "c.json", _candidate_from_reference())
        left = _write(
            tmp_path / "ref.json",
            {"identity": _reference_weighted_identity(), "totals": {"col": 100.0}},
        )
        right = _write(
            tmp_path / "cand.json",
            {"identity": _candidate_weighted_identity(), "totals": {"col": 125.0}},
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
                "--strict",
                "--receipt-json",
                str(receipt),
            ]
        )

        assert code == 0
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["strict_failure"] is False
        assert report["register"]["matched_ids"] == ["totals-signed"]
        assert report["register"]["unused_ids"] == []
        assert report["register"]["dormant_ids"] == []
