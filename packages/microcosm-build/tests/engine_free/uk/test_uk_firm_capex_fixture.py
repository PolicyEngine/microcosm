"""UK business capex fixture (fixture only; no build reads it)."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from test_support.paths import paths_for

FIXTURE = paths_for("microcosm-build").tests / "fixtures" / "uk" / "firm_capex_2024"


def test_capex_fixture_is_pinned_and_adds_up():
    provenance = json.loads((FIXTURE / "provenance.json").read_text())
    for name, meta in provenance["files"].items():
        assert (
            hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == meta["sha256"]
        )
    bi = pd.read_csv(FIXTURE / "business_investment_by_asset.csv").set_index(
        "series_id"
    )
    assert bi.loc["NPEK", "value_gbp_million"] == 309068
    assert bi.drop(index="NPEK")["value_gbp_million"].sum() == 309068
    abs_capex = pd.read_csv(FIXTURE / "abs_capex_by_section.csv").set_index(
        "sic_section"
    )
    assert abs_capex.loc["A-S (part)", "acquisitions_gbp_million"] == 243900
