"""The local compile-parity receipt's signed texts (microcosm#1300)."""

from __future__ import annotations

from tools import build_uk_ledger_compile_parity_signed_differences as parity_tool

_UC_TOTALS = "dwp.uc.households_by_area"
_UC_CHILD_BANDS = tuple(
    f"dwp.uc.households_by_area_children_{band}" for band in ("0", "1", "2", "3plus")
)


def _row(name: str, kind: str, fixture: float | None, ours: float | None) -> dict:
    row = {"kind": kind, "name": name, "period": 2025, "reason": "generic"}
    if fixture is not None:
        row["fixture_value"] = fixture
    if ours is not None:
        row["ledger_value"] = ours
    return row


def _report() -> dict:
    # Synthetic values. Twenty-one area totals whose fixture runs from 0.900
    # to 1.000 of ours in steps of 0.005, so the 5th and 95th percentiles are
    # 0.905 and 0.995; twenty-one child-band cells, spread over the four
    # bands, from 0.80 to 1.20 in steps of 0.02 (0.820 and 1.180).
    totals = [
        _row(f"{_UC_TOTALS}@E1400{1000 + i}", "calibration_drift", 1000.0 - 5 * i, 1e3)
        for i in range(21)
    ]
    children = [
        _row(
            f"{_UC_CHILD_BANDS[i % 4]}@E1400{1000 + i}",
            "calibration_drift",
            200.0 * (0.80 + 0.02 * i),
            200.0,
        )
        for i in range(21)
    ]
    return {
        "differences": [
            *totals,
            *children,
            _row(f"{_UC_TOTALS}@N05000001", "fixture_only", 10.0, None),
            _row("ons.tenure.private_rent@E14001000", "calibration_drift", 9.0, 10.0),
        ]
    }


def test_the_uc_drift_rationales_quote_the_spread_of_the_rows_they_cover():
    report = _report()
    parity_tool._add_local_signed_rationale_notes(report)
    reasons = {row["name"]: row["reason"] for row in report["differences"]}
    totals = {
        reason for name, reason in reasons.items() if name.startswith(f"{_UC_TOTALS}@E")
    }
    children = {
        reason for name, reason in reasons.items() if name.startswith(_UC_CHILD_BANDS)
    }
    # One text per rationale, each quoting its own rows.
    (totals_reason,) = totals
    (children_reason,) = children
    assert "(P5-P95 0.905-0.995 across the 21 rows)" in totals_reason
    assert "(P5-P95 fixture/ours 0.820-1.180 across the 21 rows)" in children_reason
    # Both describe the paid-claims series the area cells bind, not the
    # uniform rescale of all claims they replaced (microcosm#1302 re-review).
    for reason in (totals_reason, children_reason):
        assert "Payment Indicator 'Yes'" in reason
        assert "0.8925" not in reason and "982" not in reason
    assert "each area's own paid claims" in totals_reason
    # The other rows keep their own rationales.
    assert (
        reasons[f"{_UC_TOTALS}@N05000001"] == parity_tool._UC_NI_FIXTURE_ONLY_RATIONALE
    )
    assert reasons["ons.tenure.private_rent@E14001000"] == (
        parity_tool._TENURE_DRIFT_RATIONALE
    )


def test_a_receipt_without_uc_drift_rows_formats_no_uc_rationale():
    report = {
        "differences": [_row(f"{_UC_TOTALS}@N05000001", "fixture_only", 10.0, None)]
    }
    metric_by_target_id = parity_tool._local_metric_by_target_id()
    assert parity_tool._uc_drift_rationales(report, metric_by_target_id) == {}
    parity_tool._add_local_signed_rationale_notes(report)
    (row,) = report["differences"]
    assert row["reason"] == parity_tool._UC_NI_FIXTURE_ONLY_RATIONALE
