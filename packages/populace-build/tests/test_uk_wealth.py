"""Contract tests for the UK household-wealth source manifest.

The UK build package is spec-only: the wealth holdings are declared in
``uk/wealth_source_stages.json`` and loaded by the shared source-manifest
runtime. These tests assert the manifest is valid and, in particular, that it
surfaces the cash / stocks-and-shares ISA split (with the investment-ISA
component folded into corporate_wealth for back-compatibility).
"""

from __future__ import annotations

import json
from pathlib import Path

from populace.build.source_manifest import load_source_manifest

UK_PACKAGE = Path(__file__).resolve().parents[1] / "src/populace/build/uk"
WEALTH_MANIFEST_PATH = UK_PACKAGE / "wealth_source_stages.json"
ISA_OUTPUTS = {"cash_isa", "stocks_and_shares_isa"}


def _manifest():
    return load_source_manifest(WEALTH_MANIFEST_PATH)


class TestUkWealthManifest:
    def test_manifest_is_uk_household_wealth(self) -> None:
        manifest = _manifest()
        assert manifest.country == "uk"
        assert manifest.version >= 1
        assert {stage.stage for stage in manifest.stages} == {"household_wealth"}

    def test_isa_outputs_present_and_nonnegative(self) -> None:
        stage = _manifest().stages[0]
        assert ISA_OUTPUTS <= set(stage.outputs)
        assert ISA_OUTPUTS <= set(stage.nonnegative_outputs)

    def test_stage_imputes_then_clips(self) -> None:
        kinds = [op.kind for op in _manifest().stages[0].operations]
        assert "fit_weighted_qrf" in kinds
        assert "support_clip" in kinds

    def test_investment_isa_folded_into_corporate_wealth(self) -> None:
        # Back-compat: investment ISAs remain part of corporate_wealth.
        folds = [
            op
            for op in _manifest().stages[0].operations
            if op.kind == "fold_into"
        ]
        assert any(
            op.parameters.get("output") == "corporate_wealth"
            and op.parameters.get("component") == "stocks_and_shares_isa"
            for op in folds
        )

    def test_donor_source_is_cited(self) -> None:
        assert _manifest().stages[0].source.startswith("https://")


class TestUkCountryPackage:
    def test_manifest_is_registered_as_a_resource(self) -> None:
        country_package = json.loads(
            (UK_PACKAGE / "country_package.json").read_text(encoding="utf-8")
        )
        assert "wealth_source_stages.json" in country_package["resources"]
