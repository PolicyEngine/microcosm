"""UK paybill anchors (ONS D.11, PAYE RTI): report-only ratios."""

from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.uk_runtime.firm_paybill_anchor import (
    read_uk_firm_paybill_anchors,
    uk_firm_paybill_anchor_ratios,
)
from test_support.paths import paths_for

FIXTURE = (
    paths_for("microcosm-build").tests / "fixtures" / "uk" / "firm_paybill_anchor_2024"
)


def test_anchors_are_the_published_values():
    anchors = read_uk_firm_paybill_anchors(FIXTURE)
    assert anchors.values["ons_d11_wages_and_salaries"] == 1_162_752e6
    assert anchors.values["paye_rti_aggregate_pay"] == pytest.approx(1_189_583_389_662)
    assert anchors.values["paye_rti_payrolled_employees"] == pytest.approx(
        30_430_260.17
    )


def test_reader_refuses_an_edited_file(tmp_path):
    for name in ("paybill_anchors.csv", "provenance.json"):
        (tmp_path / name).write_bytes((FIXTURE / name).read_bytes())
    path = tmp_path / "paybill_anchors.csv"
    path.write_text(path.read_text().replace("1162752", "1162753"))
    with pytest.raises(ValueError, match="sha256"):
        read_uk_firm_paybill_anchors(tmp_path)


def test_ratios_are_weighted_and_report_only():
    anchors = read_uk_firm_paybill_anchors(FIXTURE)
    got = uk_firm_paybill_anchor_ratios(
        np.array([30_000.0, 50_000.0]), np.array([1, 2]), np.array([2.0, 1.0]), anchors
    )
    assert got["weighted_paybill_gbp"] == 110_000.0
    assert got["weighted_employees"] == 4.0
    assert got["paybill_over_ons_d11"] == pytest.approx(110_000 / 1_162_752e6)
    assert got["status"] == "report_only"
