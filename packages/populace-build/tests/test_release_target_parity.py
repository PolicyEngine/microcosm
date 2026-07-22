"""US release target-parity contract: gate + anti-rot, isolated from the feed.

The target-side analog of ``test_release_input_coverage``. The acceptance cases:

1. every ``compiled`` family present in the registry passes;
2. a ``compiled`` family missing from the registry fails, named;
3. a ``reviewed_exclusion`` family the registry now compiles is stale and fails
   (#286/#337 cannot-rot);
4. the anti-rot check rejects an undeclared feed family, a feed-surface family
   the feed no longer carries, a feed-sha mismatch, and — the red line — any
   attempt to downgrade the core SSA SSI recipient family off ``compiled``;
5. the shipped manifest is self-consistent with the checked-in feed inventory,
   and (when the pinned feed is present) reproduces exactly from it.

The registry is a lightweight stub exposing only ``.specs[].name`` (the single
surface the gate reads), so nothing here needs the 131 MB feed or
policyengine-us. Feed-dependent reproduction is guarded.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from populace.build.us_runtime.release_target_parity import (
    COMPILED_STATUS,
    RED_LINE_COMPILED_FAMILIES,
    REVIEWED_EXCLUSION_STATUS,
    TargetFamily,
    TargetFence,
    TargetParityManifest,
    assert_target_parity_manifest_current,
    load_target_parity_feed_families,
    load_target_parity_manifest,
    registry_target_family_ids,
    us_release_target_parity_gate,
    us_target_family_id,
)

_FENCE = TargetFence(
    origin="not a us-data calibration target (test)",
    purpose="n/a — test fixture",
    verdict_basis="no fence to rebuild (test)",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_GENERATOR = _REPO_ROOT / "tools" / "build_us_target_parity_manifest.py"
_US_PACKAGE_DIR = (
    _REPO_ROOT / "packages" / "populace-build" / "src" / "populace" / "build" / "us"
)


def _registry(family_names) -> SimpleNamespace:
    """A stub registry whose specs' names resolve to ``family_names``.

    Appends a period tag + leaf so ``us_target_family_id`` maps each name back to
    its family id (the concept token is the first non-period token).
    """
    return SimpleNamespace(
        specs=[
            SimpleNamespace(name=f"{family}.cy2024.leaf{index}")
            for index, family in enumerate(family_names)
        ]
    )


def _manifest(families: tuple[TargetFamily, ...]) -> TargetParityManifest:
    return TargetParityManifest(
        reference={"feed": "test", "feed_sha256": "abc"}, families=families
    )


_CONTRACT = _manifest(
    (
        TargetFamily("ssa_supplement.ssi_recipients", COMPILED_STATUS),
        TargetFamily("bea_nipa.total_wages_salaries", COMPILED_STATUS),
        TargetFamily("usda_snap.state_benefits", COMPILED_STATUS),
        TargetFamily(
            "bea_nipa.personal_income",
            REVIEWED_EXCLUSION_STATUS,
            classification="macro_control_total",
            reason="NIPA macro aggregate; income fit from SOI + CBO.",
            evidence="compiled sibling irs_soi.historic_table_2",
            fence=_FENCE,
        ),
    )
)

#: The feed inventory consistent with ``_CONTRACT`` (every compiled + reviewed
#: family present, both red-line families declared).
_CONTRACT_FEED = {
    "feed_sha256": "abc",
    "families": {
        "ssa_supplement.ssi_recipients": 52,
        "bea_nipa.total_wages_salaries": 1,
        "usda_snap.state_benefits": 51,
        "bea_nipa.personal_income": 1,
    },
}

_CONTRACT_COMPILED = (
    "ssa_supplement.ssi_recipients",
    "bea_nipa.total_wages_salaries",
    "usda_snap.state_benefits",
)


class TestFamilyId:
    def test_ssa_recipients(self) -> None:
        assert (
            us_target_family_id(
                "ssa_supplement.cy2024.ssi_recipients.by_area_category"
                ".all_areas_total.recipient_count"
            )
            == "ssa_supplement.ssi_recipients"
        )

    def test_soi_historic_table(self) -> None:
        assert (
            us_target_family_id("irs_soi.ty2022.historic_table_2.us.all.ctc_amount")
            == "irs_soi.historic_table_2"
        )

    def test_cbo_revenues(self) -> None:
        assert (
            us_target_family_id(
                "cbo.fy2023.revenues.individual_income_taxes.actual_amount"
            )
            == "cbo.revenues"
        )

    def test_namespace_only_when_no_concept(self) -> None:
        assert us_target_family_id("cbo.fy2023") == "cbo"

    def test_empty(self) -> None:
        assert us_target_family_id("") == ""


class TestGate:
    def test_every_compiled_family_present_passes(self) -> None:
        registry = _registry(_CONTRACT_COMPILED)
        result = us_release_target_parity_gate(registry, manifest=_CONTRACT)
        assert result.passed
        assert result.name == "us_release_target_parity"
        assert result.details["compiled_families"] == 3

    def test_missing_compiled_family_fails_named(self) -> None:
        registry = _registry(["ssa_supplement.ssi_recipients"])
        result = us_release_target_parity_gate(registry, manifest=_CONTRACT)
        assert not result.passed
        assert any("usda_snap.state_benefits" in failure for failure in result.failures)

    def test_stale_reviewed_exclusion_fails(self) -> None:
        # A reviewed-exclusion family the registry now compiles must be promoted.
        registry = _registry([*_CONTRACT_COMPILED, "bea_nipa.personal_income"])
        result = us_release_target_parity_gate(registry, manifest=_CONTRACT)
        assert not result.passed
        assert any("bea_nipa.personal_income" in failure for failure in result.failures)
        assert any("cannot rot" in failure for failure in result.failures)


class TestTargetFamilyValidation:
    def test_reviewed_exclusion_requires_reason(self) -> None:
        with pytest.raises(ValueError, match="needs a reason"):
            TargetFamily(
                "x.y", REVIEWED_EXCLUSION_STATUS, classification="c", evidence="e"
            )

    def test_reviewed_exclusion_requires_classification(self) -> None:
        with pytest.raises(ValueError, match="needs a classification"):
            TargetFamily("x.y", REVIEWED_EXCLUSION_STATUS, reason="r", evidence="e")

    def test_reviewed_exclusion_requires_evidence(self) -> None:
        with pytest.raises(ValueError, match="needs evidence"):
            TargetFamily(
                "x.y", REVIEWED_EXCLUSION_STATUS, classification="c", reason="r"
            )

    def test_reviewed_exclusion_requires_fence(self) -> None:
        with pytest.raises(ValueError, match="needs a fence"):
            TargetFamily(
                "x.y",
                REVIEWED_EXCLUSION_STATUS,
                classification="c",
                reason="r",
                evidence="e",
            )

    def test_fence_requires_all_three_fields(self) -> None:
        with pytest.raises(ValueError, match="TargetFence.verdict_basis is required"):
            TargetFence(origin="o", purpose="p", verdict_basis="")

    def test_unknown_status_rejected(self) -> None:
        with pytest.raises(ValueError, match="status must be one of"):
            TargetFamily("x.y", "maybe")


class TestAntiRot:
    def test_undeclared_feed_family_fails(self) -> None:
        feed = {
            "feed_sha256": "abc",
            "families": {**_CONTRACT_FEED["families"], "new_source.new_concept": 3},
        }
        with pytest.raises(ValueError, match="not declared in the manifest"):
            assert_target_parity_manifest_current(
                manifest=_CONTRACT, feed_families=feed
            )

    def test_feed_surface_family_no_longer_in_feed_fails(self) -> None:
        # usda_snap.state_benefits is a declared compiled (feed-surface) family;
        # a feed inventory omitting it is drift.
        families = dict(_CONTRACT_FEED["families"])
        del families["usda_snap.state_benefits"]
        feed = {"feed_sha256": "abc", "families": families}
        with pytest.raises(ValueError, match="no longer carries"):
            assert_target_parity_manifest_current(
                manifest=_CONTRACT, feed_families=feed
            )

    def test_feed_sha_mismatch_fails(self) -> None:
        feed = {**_CONTRACT_FEED, "feed_sha256": "different"}
        with pytest.raises(ValueError, match="does not match the feed-family"):
            assert_target_parity_manifest_current(
                manifest=_CONTRACT, feed_families=feed
            )

    def test_red_line_downgrade_is_rejected(self) -> None:
        downgraded_families = tuple(
            replace(
                family,
                status=REVIEWED_EXCLUSION_STATUS,
                classification="deferred",
                reason="pretend we dropped it",
                evidence="none",
                fence=_FENCE,
            )
            if family.name in RED_LINE_COMPILED_FAMILIES
            else family
            for family in _CONTRACT.families
        )
        downgraded = _manifest(downgraded_families)
        with pytest.raises(ValueError, match="must stay status='compiled'"):
            assert_target_parity_manifest_current(
                manifest=downgraded, feed_families=_CONTRACT_FEED
            )

    def test_registry_half_flags_undeclared_registry_family(self) -> None:
        registry = _registry([*_CONTRACT_COMPILED, "surprise.family"])
        with pytest.raises(ValueError, match="does not declare"):
            assert_target_parity_manifest_current(
                manifest=_CONTRACT, feed_families=_CONTRACT_FEED, registry=registry
            )


class TestShippedManifest:
    def test_manifest_loads_nonempty_with_compiled_and_reviewed(self) -> None:
        manifest = load_target_parity_manifest()
        assert manifest.compiled_families
        assert manifest.reviewed_exclusions
        assert manifest.schema_version == 1

    def test_every_reviewed_exclusion_carries_reason_evidence_and_fence(self) -> None:
        manifest = load_target_parity_manifest()
        for family in manifest.families:
            if family.status == REVIEWED_EXCLUSION_STATUS:
                assert family.reason, family.name
                assert family.classification, family.name
                assert family.evidence, family.name
                assert family.fence is not None, family.name
                assert family.fence.origin, family.name
                assert family.fence.purpose, family.name
                assert family.fence.verdict_basis, family.name

    def test_red_line_families_are_compiled(self) -> None:
        manifest = load_target_parity_manifest()
        for family in RED_LINE_COMPILED_FAMILIES:
            assert manifest.by_name[family].status == COMPILED_STATUS

    def test_wired_nipa_and_liheap_families_are_compiled(self) -> None:
        manifest = load_target_parity_manifest()
        for family in (
            "bea_nipa.total_wages_salaries",
            "bea_nipa.proprietors_income",
            "federal_reserve_z1.households_nonprofits_balance_sheet",
            "hhs_acf_liheap.national_profile",
        ):
            assert manifest.by_name[family].status == COMPILED_STATUS

    def test_deferred_state_wages_carry_a_fence(self) -> None:
        manifest = load_target_parity_manifest()
        state_wages = manifest.by_name["bea_regional.state_wages_salaries"]
        assert state_wages.status == REVIEWED_EXCLUSION_STATUS
        assert "PR #1034" in state_wages.fence.origin

    def test_ssi_state_payments_family_is_compiled(self) -> None:
        manifest = load_target_parity_manifest()
        assert manifest.by_name["ssa_supplement.ssi_payments"].status == COMPILED_STATUS

    def test_oasdi_family_stays_compiled(self) -> None:
        manifest = load_target_parity_manifest()
        assert (
            manifest.by_name["ssa_supplement.oasdi_ssi_payments"].status
            == COMPILED_STATUS
        )

    def test_manifest_feed_surface_matches_checked_in_inventory(self) -> None:
        manifest = load_target_parity_manifest()
        feed = load_target_parity_feed_families()
        assert manifest.feed_surface_families == frozenset(feed["families"])

    def test_manifest_sha_matches_feed_inventory_sha(self) -> None:
        manifest = load_target_parity_manifest()
        feed = load_target_parity_feed_families()
        assert manifest.reference["feed_sha256"] == feed["feed_sha256"]

    def test_checked_in_anti_rot_passes(self) -> None:
        # The checked-in half (manifest vs committed feed inventory + red line)
        # runs without the feed and must pass on the shipped artifacts.
        assert_target_parity_manifest_current()

    def test_source_absent_families_are_not_in_feed_inventory(self) -> None:
        manifest = load_target_parity_manifest()
        feed = load_target_parity_feed_families()
        for family in manifest.source_absent_families:
            assert family not in feed["families"]


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "build_us_target_parity_manifest", _MANIFEST_GENERATOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestJctObbbaAdjudication:
    """populace#451 items 3-4: the JCX-35-25 no-tax anchors' parity disposition.

    The jct.obbba_title_vii facts (ledger jct-obbba-revenue-estimates-2025)
    ride a future feed cut; the generator must already carry their fenced
    reviewed exclusion or regeneration hard-fails on the new family.
    """

    def test_obbba_fact_ids_familize_to_the_adjudicated_family(self) -> None:
        from populace.build.us_runtime.release_target_parity import (
            us_target_family_id,
        )

        assert (
            us_target_family_id(
                "jct.obbba_title_vii.fy2026.no_tax_on_overtime.revenue_effect"
            )
            == "jct.obbba_title_vii"
        )
        assert (
            us_target_family_id(
                "jct.obbba_title_vii.fy2029.no_tax_on_tips.revenue_effect"
            )
            == "jct.obbba_title_vii"
        )

    def test_jct_obbba_family_has_fenced_deferred_exclusion(self) -> None:
        generator = _load_generator()
        classification, reason, evidence, fence = generator._exclusion_for(
            "jct.obbba_title_vii"
        )
        assert classification == "deferred"
        assert "JCX-35-25" in reason
        assert "structurally zero at 2024 law" in reason
        assert "fsla_overtime_premium_neutralization" in reason
        assert "-$32.806B" in evidence
        for key in ("origin", "purpose", "verdict_basis"):
            assert fence[key]
        assert "TY2025-TY2028" in fence["purpose"]


class TestRegeneration:
    """Feed-dependent reproduction — guarded on the pinned feed being present."""

    def _feed_or_skip(self, generator):
        feed_path = generator.DEFAULT_FEED_PATH
        if not feed_path.exists():
            pytest.skip(f"pinned feed not present at {feed_path}")
        return feed_path

    def test_committed_artifacts_match_regeneration(self) -> None:
        generator = _load_generator()
        feed_path = self._feed_or_skip(generator)
        facts, feed_sha256 = generator._load_feed(feed_path)
        manifest, feed_families = generator.build_manifest(
            facts, feed_sha256, generator.DEFAULT_FEED_NAME
        )
        committed_manifest = json.loads(
            (_US_PACKAGE_DIR / "target_parity_manifest.json").read_text()
        )
        committed_feed = json.loads(
            (_US_PACKAGE_DIR / "target_parity_feed_families.json").read_text()
        )
        assert manifest == committed_manifest
        assert feed_families == committed_feed

    def test_gate_passes_on_real_compiled_registry(self) -> None:
        generator = _load_generator()
        feed_path = self._feed_or_skip(generator)
        from populace.build.us_runtime.fiscal_targets import (
            compile_us_fiscal_target_registry,
        )
        from populace.build.us_runtime.medicaid_take_up import (
            apply_us_medicaid_enrollment_substitutions,
        )

        facts, _ = generator._load_feed(feed_path)
        # Mirror the generator's declared regime (CD_SURFACE_REGIME): parity
        # is declared and checked against the registry compiled the same way,
        # so flipping the regime constant updates both sides together.
        from populace.build.us_runtime import (
            default_congressional_district_vintage_crosswalk_path,
            load_congressional_district_vintage_crosswalk,
        )

        cd_on = generator.CD_SURFACE_REGIME == "on"
        registry = compile_us_fiscal_target_registry(
            facts,
            target_period=2024,
            age_targets=True,
            include_congressional_district_targets=cd_on,
            congressional_district_vintage_crosswalk=(
                load_congressional_district_vintage_crosswalk(
                    default_congressional_district_vintage_crosswalk_path()
                )
                if cd_on
                else None
            ),
        )
        registry, _ = apply_us_medicaid_enrollment_substitutions(registry)
        result = us_release_target_parity_gate(registry)
        assert result.passed, result.failures
        assert_target_parity_manifest_current(registry=registry)
        assert "ssa_supplement.ssi_recipients" in registry_target_family_ids(registry)
        assert "ssa_ssi_monthly.ssi_federal_payment_recipients" in (
            registry_target_family_ids(registry)
        )
