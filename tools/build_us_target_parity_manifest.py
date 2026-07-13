"""Regenerate the US release target-parity manifest and feed-family inventory.

The target-side analog of ``tools/build_us_release_input_coverage_manifest.py``.
Where that tool declares which reference-eCPS input columns a release must
persist, this one declares which administrative target *families* the compiled
populace registry must carry — the families the retired us-data/eCPS pipeline
calibrated to.

Derivation (fully from the sha-pinned consumer feed + the deterministic registry
compile):

- **Feed-family inventory** = every ``namespace.concept`` family id the pinned
  ``consumer_facts_*.jsonl`` carries, with its fact count. Written to
  ``target_parity_feed_families.json`` so the manifest consistency check runs in
  CI without the 131 MB feed.
- **Compiled families** = the families the registry compiles today
  (``compile_us_fiscal_target_registry(age_targets=True)`` + the reviewed CMS
  Medicaid enrollment substitution, exactly as the release builder compiles it).
  Every compiled family gets ``status: compiled``.
- **Reviewed exclusions** = every feed family that does NOT compile, each
  classified with a reason naming its evidence (a sample ``source_record_id``, a
  code constant, or the compiled sibling family that supersedes it). A feed
  family with neither a compile nor a declared exclusion HALTS generation — the
  anti-rot guarantee that a new administrative family cannot be silently ignored.
- **Source-absent us-data families** = administrative targets the retired
  us-data pipeline calibrated to for which the pinned feed carries no ledger
  fact (BLS Consumer Expenditure, WIC, HUD housing assistance). Declared as
  ``source_absent`` reviewed exclusions so the diff records them.

Run:  uv run python tools/build_us_target_parity_manifest.py
It rewrites packages/populace-build/src/populace/build/us/target_parity_manifest.json
and target_parity_feed_families.json. A test asserts the committed files match
this regeneration, so the manifest cannot silently drift from the pinned feed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from populace.build.us_runtime.fiscal_targets import compile_us_fiscal_target_registry
from populace.build.us_runtime.medicaid_take_up import (
    apply_us_medicaid_enrollment_substitutions,
)
from populace.build.us_runtime.release_target_parity import (
    COMPILED_STATUS,
    REVIEWED_EXCLUSION_STATUS,
    SOURCE_ABSENT_CLASSIFICATION,
    us_target_family_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
US_PACKAGE_DIR = (
    REPO_ROOT / "packages" / "populace-build" / "src" / "populace" / "build" / "us"
)
MANIFEST_PATH = US_PACKAGE_DIR / "target_parity_manifest.json"
FEED_FAMILIES_PATH = US_PACKAGE_DIR / "target_parity_feed_families.json"

#: The pinned consumer feed the certified registry compiles from.
DEFAULT_FEED_PATH = (
    Path.home()
    / "PolicyEngine"
    / "_buildh-runtime"
    / "inputs"
    / "consumer_facts_buildh_v8.jsonl"
)
DEFAULT_FEED_NAME = "consumer_facts_buildh_v8.jsonl"
EXPECTED_FEED_SHA256_PREFIX = "94b7155f"
TARGET_PERIOD = 2024

# ---------------------------------------------------------------------------
# Reviewed exclusions — every feed family that does not compile, classified with
# an evidence-naming reason. Namespace rules cover the many one-per-series BEA
# aggregates; exact rules cover the rest. Reasons cite a compiled sibling
# family, a code constant, or the mechanism that makes the family non-linear /
# survey-derived / off-by-default.
# ---------------------------------------------------------------------------

# (classification, reason, evidence) keyed by source_record_id namespace.
_NAMESPACE_EXCLUSIONS: dict[str, tuple[str, str, str]] = {
    "bea_nipa": (
        "macro_control_total",
        "BEA NIPA national-accounts aggregate (calendar-year macro control "
        "total). Not a household-linear administrative level PolicyEngine-US "
        "calibrates against: household income components are fit from the IRS "
        "SOI micro tables (irs_soi.historic_table_2, compiled) and CBO "
        "revenue-by-source projections (cbo.revenue_projection, compiled). The "
        "NIPA aggregate is retained in the ledger as a macro cross-check, not a "
        "calibration target.",
        "compiled siblings irs_soi.historic_table_2 and cbo.revenue_projection "
        "supply the household-linear income surface; sample fact "
        "bea_nipa.cy2023.personal_income.a065rc.amount",
    ),
    "bea_regional": (
        "macro_control_total",
        "BEA regional (state) personal-income account aggregate. State income "
        "is calibrated from the IRS SOI state tables (irs_soi.state_2022 and "
        "the historic_table_2 state rows, compiled) and Census STC "
        "(census_stc.individual_income_tax_collections, compiled); the BEA "
        "regional total is a macro state control without a per-record model "
        "counterpart.",
        "compiled siblings irs_soi.state_2022 and "
        "census_stc.individual_income_tax_collections; sample fact "
        "bea_regional.cy2023.state_personal_income.us.amount",
    ),
}

_RETIREMENT_CONTRIBUTION_EXCLUSION = (
    "input_side",
    "IRS SOI / W-2 retirement-contribution aggregate. Retirement contributions "
    "are modeled as imputed INPUT columns (traditional/roth 401(k) and IRA "
    "contributions), governed by the input-coverage contract, not as reweighting "
    "targets. Calibrating them as fiscal targets would double-govern a quantity "
    "the input-coverage gate already owns.",
    "RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS declares the "
    "*_contributions_desired input columns in release_input_coverage.py",
)

_SNAP_PERSONS_EXCLUSION = (
    "not_modeled",
    "USDA FNS SNAP average-monthly PERSONS. The SNAP assistance unit is often a "
    "subset of the SPM unit (FY2024 FNS persons-per-household 1.88 vs 2.82 "
    "simulated members per taker unit), so a person indicator overcounts FNS "
    "participants by ~50%. The household caseload "
    "(usda_snap.national/state_average_monthly_households, compiled) is the "
    "calibrated count; persons stay unmapped until sub-unit participation is "
    "modeled.",
    "INDICATOR_LEDGER_TARGETS average_monthly_households comment + "
    "test_snap_person_caseload_fact_is_not_compiled",
)

# (classification, reason, evidence) keyed by exact family id.
_FAMILY_EXCLUSIONS: dict[str, tuple[str, str, str]] = {
    "cbo.revenues": (
        "macro_control_total",
        "CBO total federal revenue line (fiscal-year budget aggregate). The "
        "household-linear CBO surface populace calibrates is the income-by-"
        "source projection (cbo.revenue_projection, compiled as cbo:5); a total "
        "revenue outturn is an aggregate, not a per-record target.",
        "compiled sibling cbo.revenue_projection; sample fact "
        "cbo.fy2023.revenues.individual_income_taxes.actual_amount",
    ),
    "census.popproj2023": (
        "superseded",
        "Census Bureau population PROJECTIONS (forward vintage popproj2023). "
        "Population is calibrated to the OBSERVED Census resident-population "
        "estimates (census_pep.v2024 and national_resident_population_age, "
        "compiled as census_pep:936); the projection series is a forward-looking "
        "alternative, not the observed administrative count.",
        "compiled sibling census_pep.v2024; sample fact "
        "census.popproj2023.cy2023.national_population.age_0.population",
    ),
    "census_acs.acs1_2023": (
        "survey_derived",
        "American Community Survey 1-year estimates. ACS is a household SAMPLE "
        "SURVEY, not an administrative universe, so its aggregates are excluded "
        "by principle. ACS age detail is used only at congressional-district "
        "grain when include_congressional_district_targets is enabled (off for "
        "the national release).",
        "census_acs facts return None unless include_congressional_district_"
        "targets in fiscal_targets._reference_from_ledger_fact; sample fact "
        "census_acs.acs1_2023.b01001.female_age.01.age_15_to_17.female_population",
    ),
    "cms_nhe.medicaid_title_xix_expenditures": (
        "non_linear",
        "CMS National Health Expenditure Medicaid Title XIX spending. Already "
        "declared calibration_role=validation_only in DIRECT_LEDGER_TARGETS: "
        "PolicyEngine-US allocates Medicaid spending from state totals through "
        "person_weight-dependent denominators, so reweighting recomputes "
        "per-person costs and this is not a linear calibration row.",
        "DIRECT_LEDGER_TARGETS ('cms_nhe','expenditure_amount','medicaid_title_"
        "xix') metadata calibration_role=validation_only",
    ),
    "federal_reserve_z1.households_nonprofits_balance_sheet": (
        "macro_control_total",
        "Federal Reserve Financial Accounts (Z.1) households & nonprofits "
        "net-worth aggregate. A macro balance-sheet total; household net worth "
        "is imputed and calibrated from SCF micro on the input side "
        "(scf_wealth source stage), not from this national aggregate.",
        "scf_wealth.py source stage supplies the net-worth micro; sample fact "
        "federal_reserve_z1.cy2023.households_nonprofits_balance_sheet.net_worth"
        ".fl152090005.amount_outstanding",
    ),
    "hhs_acf_liheap.national_profile": (
        "deferred",
        "HHS LIHEAP national profile (households served / funds). PolicyEngine-US "
        "does not yet expose a LIHEAP receipt outcome to fit this count to a "
        "per-household model counterpart, so it is deferred rather than dropped; "
        "retained as a ledger reference fact.",
        "no LIHEAP target_role in INDICATOR_LEDGER_TARGETS/DIRECT_LEDGER_TARGETS; "
        "sample fact hhs_acf_liheap.fy2023.national_profile.state_programs."
        "households_served",
    ),
    "hhs_acf_tanf.average_monthly_families": (
        "deferred",
        "HHS ACF TANF average-monthly family caseload. TANF is calibrated on "
        "BENEFIT DOLLARS (hhs_acf_tanf.cash_assistance, compiled as "
        "hhs_acf_tanf:30); the caseload COUNT is not yet wired to a TANF-receipt "
        "indicator (the SNAP-household caseload analog for TANF), so it is "
        "deferred.",
        "compiled sibling hhs_acf_tanf.cash_assistance carries the dollar "
        "targets; sample fact "
        "hhs_acf_tanf.fy2024.average_monthly_families.us.us_total.total_families",
    ),
    "hhs_acf_tanf.average_monthly_recipients": (
        "deferred",
        "HHS ACF TANF average-monthly recipient caseload. As with the family "
        "caseload, TANF is calibrated on benefit dollars "
        "(hhs_acf_tanf.cash_assistance, compiled); the recipient COUNT has no "
        "wired TANF-receipt indicator yet and is deferred.",
        "compiled sibling hhs_acf_tanf.cash_assistance; sample fact "
        "hhs_acf_tanf.fy2024.average_monthly_recipients.us.us_total."
        "total_recipients",
    ),
    "irs_soi.congressional_district_2022": (
        "off_by_default",
        "IRS SOI congressional-district table. CD-level targets are opt-in "
        "(include_congressional_district_targets=False for the national "
        "release); the national and state SOI surfaces are compiled instead. "
        "Enabling CD targets compiles these — they are excluded for the national "
        "build by design, not dropped.",
        "test_soi_congressional_district_targets_are_opt_in + the "
        "include_congressional_district_targets gate in "
        "fiscal_targets._soi_reference_from_fact",
    ),
    "irs_soi.form_w2_401k_elective_deferrals": _RETIREMENT_CONTRIBUTION_EXCLUSION,
    "irs_soi.form_w2_designated_roth_401k_contributions": (
        _RETIREMENT_CONTRIBUTION_EXCLUSION
    ),
    "irs_soi.roth_ira_contributions": _RETIREMENT_CONTRIBUTION_EXCLUSION,
    "irs_soi.traditional_ira_contributions": _RETIREMENT_CONTRIBUTION_EXCLUSION,
    "irs_soi.form_w2_social_security_tips": (
        "not_modeled",
        "IRS W-2 Social Security tip aggregates. Already declared in "
        "US_FISCAL_TARGET_SUPPORT_EXCLUSIONS: current US support does not "
        "materialize a positive tip_income source column, so the W-2 tip return "
        "counts need the SIPP/ORG tip source stage wired before calibration.",
        "US_FISCAL_TARGET_SUPPORT_EXCLUSIONS entry "
        "irs_soi.ty2023.form_w2_social_security_tips.box_7_social_security_tips."
        "return_count",
    ),
    "kff.marketplace_effectuated_enrollment": (
        "superseded",
        "Kaiser Family Foundation state marketplace effectuated-enrollment "
        "compilation. ACA marketplace enrollment is calibrated from the primary "
        "CMS administrative source (cms_aca.oep2024, compiled as cms_aca:102); "
        "KFF is a secondary aggregator of the same underlying CMS data, excluded "
        "to avoid a duplicate target.",
        "compiled sibling cms_aca.oep2024; sample fact "
        "kff.marketplace_effectuated_enrollment.2024.state.us."
        "total_effectuated_marketplace_enrollment",
    ),
    "usda_snap.national_average_monthly_persons": _SNAP_PERSONS_EXCLUSION,
    "usda_snap.state_average_monthly_persons": _SNAP_PERSONS_EXCLUSION,
}

# Administrative target families the retired us-data pipeline calibrated to for
# which the pinned feed carries NO ledger fact — source-absent by the feed.
# (family id, reason, evidence) with classification source_absent.
_SOURCE_ABSENT_US_DATA_FAMILIES: dict[str, tuple[str, str]] = {
    "bls.consumer_expenditure": (
        "BLS Consumer Expenditure Survey aggregates targeted by the retired "
        "us-data pipeline (nation/bls/ce). The pinned consumer feed carries no "
        "BLS source fact, so there is no ledger-shaped fact to compile; "
        "source-absent pending a Ledger BLS CE ingest.",
        "retired us-data pipeline (archived) references nation/bls/ce; feed "
        "source families carry no 'bls' source",
    ),
    "wic.national_summary": (
        "USDA WIC national annual summary targeted by the retired us-data "
        "pipeline (WIC_NATIONAL_ANNUAL_SUMMARY_SOURCE in etl_national_targets). "
        "The pinned feed carries no WIC source fact; source-absent. WIC receipt "
        "is modeled via the would_claim_wic take-up input, not a target.",
        "retired us-data pipeline (archived) db/etl_national_targets.py "
        "WIC_NATIONAL_ANNUAL_SUMMARY_SOURCE; feed carries no 'wic' source",
    ),
    "hud.housing_assistance": (
        "HUD housing-assistance aggregates targeted by the retired us-data "
        "pipeline (db/etl_housing_assistance.py). The pinned feed carries no HUD "
        "source fact; source-absent pending a Ledger HUD ingest.",
        "retired us-data pipeline (archived) db/etl_housing_assistance.py; "
        "feed carries no 'hud' source",
    ),
}


def _load_feed(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    facts = [
        json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()
    ]
    return facts, sha256


def _feed_family_counts(facts: list[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for fact in facts:
        source_record_id = (fact.get("lineage") or {}).get("source_record_id", "")
        family = us_target_family_id(str(source_record_id))
        if family:
            counts[family] += 1
    return counts


def _compiled_families(facts: list[dict]) -> set[str]:
    registry = compile_us_fiscal_target_registry(
        facts, target_period=TARGET_PERIOD, age_targets=True
    )
    registry, _ = apply_us_medicaid_enrollment_substitutions(registry)
    return {
        us_target_family_id(spec.name)
        for spec in registry.specs
        if us_target_family_id(spec.name)
    }


def _exclusion_for(family: str) -> tuple[str, str, str]:
    if family in _FAMILY_EXCLUSIONS:
        return _FAMILY_EXCLUSIONS[family]
    namespace = family.split(".", 1)[0]
    if namespace in _NAMESPACE_EXCLUSIONS:
        return _NAMESPACE_EXCLUSIONS[namespace]
    raise SystemExit(
        f"Feed family {family!r} neither compiles nor has a declared reviewed "
        "exclusion. Add an entry to _FAMILY_EXCLUSIONS or _NAMESPACE_EXCLUSIONS "
        "in tools/build_us_target_parity_manifest.py naming its evidence — a new "
        "administrative family must never be silently ignored."
    )


def build_manifest(
    facts: list[dict], feed_sha256: str, feed_name: str
) -> tuple[dict, dict]:
    feed_counts = _feed_family_counts(facts)
    compiled = _compiled_families(facts)

    families: dict[str, dict] = {}
    for family in sorted(feed_counts):
        if family in compiled:
            families[family] = {"status": COMPILED_STATUS}
        else:
            classification, reason, evidence = _exclusion_for(family)
            families[family] = {
                "status": REVIEWED_EXCLUSION_STATUS,
                "classification": classification,
                "reason": reason,
                "evidence": evidence,
            }

    # Compiled families with no feed fact should not exist (every compiled spec
    # traces to a feed fact); guard against a mapping that invents a family.
    invented = sorted(compiled - set(feed_counts))
    if invented:
        raise SystemExit(
            f"Registry compiles family(ies) with no feed fact: {invented}. "
            "The family-id derivation or the wiring is inconsistent."
        )

    for family, (reason, evidence) in sorted(_SOURCE_ABSENT_US_DATA_FAMILIES.items()):
        families[family] = {
            "status": REVIEWED_EXCLUSION_STATUS,
            "classification": SOURCE_ABSENT_CLASSIFICATION,
            "reason": reason,
            "evidence": evidence,
        }

    n_compiled = sum(1 for e in families.values() if e["status"] == COMPILED_STATUS)
    n_reviewed = len(families) - n_compiled
    manifest = {
        "schema_version": 1,
        "reference": {
            "feed": feed_name,
            "feed_sha256": feed_sha256,
            "target_period": str(TARGET_PERIOD),
            "registry_compile": (
                "compile_us_fiscal_target_registry(age_targets=True) + "
                "apply_us_medicaid_enrollment_substitutions"
            ),
            "us_data_source": (
                "retired us-data pipeline (archived): db/etl_national_targets.py, "
                "db/etl_*.py, utils/national_target_parity.py"
            ),
            "family_granularity": (
                "namespace.concept of the ledger source_record_id (us_target_family_id)"
            ),
            "compiled_families": str(n_compiled),
            "reviewed_exclusions": str(n_reviewed),
        },
        "families": dict(sorted(families.items())),
    }

    feed_families = {
        "feed": feed_name,
        "feed_sha256": feed_sha256,
        "families": dict(sorted(feed_counts.items())),
    }
    return manifest, feed_families


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger-facts",
        type=Path,
        default=DEFAULT_FEED_PATH,
        help="Path to the pinned consumer feed JSONL.",
    )
    parser.add_argument(
        "--feed-name",
        default=DEFAULT_FEED_NAME,
        help="Feed filename recorded in the manifest/inventory.",
    )
    args = parser.parse_args()

    facts, feed_sha256 = _load_feed(args.ledger_facts)
    if not feed_sha256.startswith(EXPECTED_FEED_SHA256_PREFIX):
        raise SystemExit(
            f"Feed sha256 {feed_sha256[:8]} does not match the pinned prefix "
            f"{EXPECTED_FEED_SHA256_PREFIX}; refusing to regenerate against an "
            "unexpected feed."
        )

    manifest, feed_families = build_manifest(facts, feed_sha256, args.feed_name)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    FEED_FAMILIES_PATH.write_text(
        json.dumps(feed_families, indent=1) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {MANIFEST_PATH.name}: "
        f"{manifest['reference']['compiled_families']} compiled, "
        f"{manifest['reference']['reviewed_exclusions']} reviewed exclusions "
        f"({len(feed_families['families'])} feed families)."
    )


if __name__ == "__main__":
    main()
