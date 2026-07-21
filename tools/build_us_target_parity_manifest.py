"""Regenerate the US release target-parity manifest and feed-family inventory.

The target-side analog of ``tools/build_us_release_input_coverage_manifest.py``.
It declares which administrative target *families* the compiled populace registry
must carry — the families the retired us-data/eCPS pipeline calibrated to.

Chesterton's fence is the governing rule. A category label (``macro_control_total``,
``not_modeled``, ``superseded``) is not a sufficient reason to drop a target the
retired pipeline calibrated to. Every reviewed exclusion must carry a ``fence``
recovering *why the fence was built*: the target's origin (the introducing
us-data PR/commit, recovered by archaeology), the failure mode it
guarded (the PR/issue rationale, quoted where one exists), and the
purpose-informed verdict basis for not rebuilding it here. The absolute rule is
**us-data-targeted ⇒ compiled, unless source-absent**; ``macro_control_total`` is
only valid for families us-data did NOT target. Where archaeology is
inconclusive (no discoverable rationale), the fence is rebuilt — the target is
wired, not excluded.

Derivation (from the sha-pinned feed + the deterministic registry compile):

- **Feed-family inventory** = every ``namespace.concept`` family id the feed
  carries, with fact counts (``target_parity_feed_families.json``).
- **Compiled families** = the families the registry compiles today
  (``compile_us_fiscal_target_registry(age_targets=True)`` + the reviewed CMS
  Medicaid enrollment substitution). Status ``compiled``.
- **Reviewed exclusions** = every non-compiling feed family, each with a fence.
  A feed family with neither a compile nor a fenced exclusion HALTS generation.
- **Source-absent us-data families** = us-data targets with no feed fact.

Run:  uv run python tools/build_us_target_parity_manifest.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from populace.build.us_runtime import (
    default_congressional_district_vintage_crosswalk_path,
    load_congressional_district_vintage_crosswalk,
)
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

DEFAULT_FEED_PATH = (
    Path.home()
    / "PolicyEngine"
    / "_buildh-runtime"
    / "inputs"
    / "consumer_facts_buildn_v9_2.jsonl"
)
DEFAULT_FEED_NAME = "consumer_facts_buildn_v9_2.jsonl"
EXPECTED_FEED_SHA256_PREFIX = "61b115c0"
TARGET_PERIOD = 2024


def _fence(origin: str, purpose: str, verdict_basis: str) -> dict[str, str]:
    return {"origin": origin, "purpose": purpose, "verdict_basis": verdict_basis}


# The fence for every BEA series the archaeology confirmed us-data never
# calibrated to (only nipa_wages_and_salaries + proprietors_income were
# direct-sum targets, PR #994; both are now compiled). The programs some NIPA
# series measure ARE calibrated by us-data, but from SSA/CMS/CBO administrative
# sources (compiled), never the BEA NIPA aggregate.
_BEA_NOT_A_TARGET_FENCE = _fence(
    origin=(
        "not a us-data calibration target: absent from us-data "
        "utils/loss.py and db/etl_national_targets.py (archaeology). The only BEA "
        "direct-sum targets were nipa_wages_and_salaries and proprietors_income "
        "(PR #994), both now compiled."
    ),
    purpose="n/a — a ledger reference fact carried by the feed, never a target.",
    verdict_basis=(
        "no fence to rebuild. Programs some of these NIPA aggregates measure "
        "(Social Security, Medicare, Medicaid, unemployment) are calibrated from "
        "SSA/CMS/CBO admin sources (compiled), not the BEA NIPA series."
    ),
)

# (classification, reason, evidence, fence) keyed by source_record_id namespace.
_NAMESPACE_EXCLUSIONS: dict[str, tuple[str, str, str, dict[str, str]]] = {
    "bea_nipa": (
        "macro_control_total",
        "BEA NIPA national-accounts aggregate the retired pipeline did not "
        "calibrate to (only wages and proprietors' income were NIPA direct-sum "
        "targets, both compiled). A macro control total retained as a ledger "
        "cross-check.",
        "compiled BEA targets bea_nipa.total_wages_salaries and "
        "bea_nipa.proprietors_income carry the household-linear NIPA income; "
        "sample fact bea_nipa.cy2023.personal_income.a065rc.amount",
        _BEA_NOT_A_TARGET_FENCE,
    ),
    "bea_regional": (
        "macro_control_total",
        "BEA regional (state) account aggregate us-data did not calibrate to "
        "(only state wages were a regional target, and it is deferred). State "
        "income is fit from IRS SOI state tables and Census STC, compiled.",
        "compiled siblings irs_soi.state_2022 and "
        "census_stc.individual_income_tax_collections; sample fact "
        "bea_regional.cy2023.state_personal_income.us.amount",
        _fence(
            origin=(
                "not a us-data calibration target: absent from loss.py and "
                "etl_national_targets.py (archaeology). The only targeted "
                "bea_regional series is state_wages_salaries (PR #1034), a "
                "separate deferred exclusion."
            ),
            purpose="n/a — a ledger reference fact, never a target.",
            verdict_basis=(
                "no fence to rebuild; state income is calibrated from IRS SOI "
                "state tables + Census STC (compiled)."
            ),
        ),
    ),
}

_RETIREMENT_CONTRIBUTION_EXCLUSION = (
    "input_side",
    "IRS SOI / W-2 retirement-contribution aggregate the retired pipeline "
    "calibrated (PR #496/#554) but which it and populace both moved to the input "
    "side: modeled as imputed pre-limit contribution INPUT columns, not "
    "reweighting targets.",
    "POST_REFERENCE_ECPS_REQUIRED_INPUTS declares the *_contributions_desired "
    "input columns in release_input_coverage.py",
    _fence(
        origin=(
            "us-data PR #496 (IRA targets) and PR #554 (401k/SE "
            "pension targets), converged at PR #1125"
        ),
        purpose=(
            'ensure income-contribution consistency ("$0-wage PUF clones get $0 '
            'in 401(k) contributions", PR #554); target the ALD deduction not '
            'raw contributions because "the variable flows directly into the ALD '
            "with no deductibility logic in policyengine-us, so the target must "
            'match the deduction, not total contributions" (loss.py)'
        ),
        verdict_basis=(
            "architecture retires it: us-data PR #1125 and populace both govern "
            "retirement contributions as imputed pre-limit INPUT columns "
            "(*_contributions_desired, POST_REFERENCE_ECPS_REQUIRED_INPUTS in the "
            "input-coverage contract) with PolicyEngine-US applying statutory "
            "limits — calibrating them here would double-govern a quantity the "
            "input-coverage gate owns. The feed carries the amounts, so this is "
            "not a source gap."
        ),
    ),
)

_SNAP_PERSONS_EXCLUSION = (
    "not_modeled",
    "USDA FNS SNAP average-monthly PERSONS. A person indicator overcounts FNS "
    "participants ~50% (FY2024 persons-per-household 1.88 vs 2.82 simulated "
    "members); the household caseload is the calibrated count.",
    "INDICATOR_LEDGER_TARGETS average_monthly_households comment + "
    "test_snap_person_caseload_fact_is_not_compiled",
    _fence(
        origin=(
            "not a us-data calibration target as a persons row: us-data "
            "calibrates SNAP dollars (CBO_PROGRAMS 'snap'), and the household "
            "caseload (usda_snap.*_average_monthly_households) is compiled."
        ),
        purpose="n/a — the household caseload is the model counterpart, not persons.",
        verdict_basis=(
            "not_modeled: PolicyEngine-US does not model sub-unit SNAP "
            "participation, so a person indicator overcounts ~50% vs the compiled "
            "household caseload."
        ),
    ),
)

# (classification, reason, evidence, fence) keyed by exact family id.
_ESI_ANCHOR_EXCLUSION = (
    "not_modeled",
    "NHE employer-contribution ESI premium aggregate, banked in the v9.2 "
    "feed (2026-07-21) as the external anchor for populace#454's future "
    "ESI input gate. The pe-us input it would anchor "
    "(employer_sponsored_insurance_premiums, the employer-paid share per "
    "the variable's own documentation) has no live producer in the "
    "hermetic lineage, so a target would bind against a structural zero. "
    "The all-employer series is the future anchor (pe-us covers government "
    "and private employees alike); the private-employer subset rides as its "
    "cross-check.",
    "ecps_parity_known_gaps.json entry employer_sponsored_insurance_"
    "premiums (NOW_* source-unavailability evidence); populace#454 "
    "descope comment 2026-07-21",
    _fence(
        origin=(
            "not a us-data calibration target: the retired eCPS IMPUTED "
            "employer premiums as an input (MEPS-IC priors over ASEC "
            "NOW_* policyholder fields, archived 42ed5d45 cps.py "
            "L197-271/L1575-1581); no loss.py or etl_national_targets.py "
            "target existed. The feed facts are the banked anchor for a "
            "future gate, entered deliberately ahead of the producer."
        ),
        purpose=(
            "anchor a validation gate for the CBO market-income ESI "
            "component once the column has a live producer: NIPA 7.8 "
            "employer contributions for group health insurance and the "
            "NHEA sponsor-of-funds employer share are same-concept "
            "estimates (~4 percent apart, CY2024)."
        ),
        verdict_basis=(
            "source-absent for the PRODUCER, not the anchor: the declared "
            "meps_esi_premiums stage cannot execute because all three "
            "required NOW_* fields are missing from the sha-pinned "
            "2022-2024 ASEC h5 inputs (recorded parity known-gap). "
            "Compiles when populace#454 restores the source fields or "
            "lands a validated PHIP_VAL+MEPS re-derivation."
        ),
    ),
)

_FAMILY_EXCLUSIONS: dict[str, tuple[str, str, str, dict[str, str]]] = {
    "cms_nhe.esi_employer_contribution_premiums": _ESI_ANCHOR_EXCLUSION,
    "cms_nhe.esi_private_employer_contribution_premiums": _ESI_ANCHOR_EXCLUSION,
    "bea_nipa.personal_interest_income": (
        "macro_control_total",
        "BEA NIPA personal interest income. Briefly a direct target (PR #994), "
        "then explicitly declined (PR #1059): the NIPA total includes imputed "
        "interest and trust flows that are not a close microdata concept.",
        "irs_soi.historic_table_2 (compiled) carries the tax-return interest; "
        "sample fact bea_nipa.cy2023.personal_interest_income.a064rc.amount",
        _fence(
            origin="us-data PR #994 (added) -> PR #1059 (removed)",
            purpose=(
                '"BEA personal interest/dividends include imputed interest, '
                "pension-plan dividends, and trust flows, so those macro totals "
                "should not directly calibrate tax/CPS interest and dividend "
                'variables" (loss.py:67-70, added by PR #1059)'
            ),
            verdict_basis=(
                "us-data itself declined it as a direct target (PR #1059): a "
                "macro benchmark, not a close microdata concept. Tax-return "
                "interest is calibrated from IRS SOI + CBO (compiled)."
            ),
        ),
    ),
    "bea_nipa.personal_dividend_income": (
        "macro_control_total",
        "BEA NIPA personal dividend income. Briefly a direct target (PR #994), "
        "then explicitly declined (PR #1059): the NIPA total includes dividends "
        "received through pension funds and private trusts.",
        "cbo.revenue_projection (qualified dividends) + irs_soi (compiled); "
        "sample fact bea_nipa.cy2023.personal_dividend_income.b703rc.amount",
        _fence(
            origin="us-data PR #994 (added) -> PR #1059 (removed)",
            purpose=(
                '"NIPA includes dividends received through pension funds and '
                "private trusts, so this is a macro benchmark rather than a pure "
                'tax concept" (deleted etl_national_targets.py note, PR #1059; '
                "FRED B703RC1A027NBEA)"
            ),
            verdict_basis=(
                "us-data itself declined it (PR #1059) as non-comparable; "
                "dividends are calibrated from the CBO qualified-dividend series "
                "(cbo.revenue_projection) and SOI (compiled)."
            ),
        ),
    ),
    "bea_regional.state_wages_salaries": (
        "deferred",
        "BEA regional state wages — a real us-data target (PR #1034), but us-data "
        "residence-adjusts SAINC4 line-50 place-of-work wages and scales them to "
        "the national total, an adjustment the feed's raw place-of-work facts "
        "cannot reproduce. The national wage aggregate is compiled.",
        "compiled national aggregate bea_nipa.total_wages_salaries; feed fact "
        "bea_regional.cy2023.state_wages_salaries.dc.amount = $96.8B is "
        "place-of-work, far above DC residence wages",
        _fence(
            origin="us-data PR #1034 (Fixes #1033)",
            purpose=(
                '"residence-adjust state wages using SAINC4 line 42 before '
                "scaling to the national NIPA wages target ... so the national "
                'and state wage controls bind" (PR #1034); the state distribution '
                "of the nonfiler-inclusive wage universe"
            ),
            verdict_basis=(
                "deferred: the feed carries raw place-of-work SAINC4 line-50 state "
                "wages, but us-data applies a place-of-work -> residence "
                "adjustment (line 42 apportioned by wage share) plus national "
                "scaling in bea_regional.py that is not reproducible from the "
                "feed. The national wage aggregate (bea_nipa.total_wages_salaries) "
                "is compiled; the state distribution is deferred pending "
                "residence-adjusted state wages in the ledger feed."
            ),
        ),
    ),
    "cbo.revenues": (
        "macro_control_total",
        "CBO total federal revenue line (fiscal-year budget aggregate) us-data "
        "did not calibrate to. The household-linear CBO surface it used is the "
        "income-by-source projection (cbo.revenue_projection, compiled).",
        "compiled sibling cbo.revenue_projection; sample fact "
        "cbo.fy2023.revenues.individual_income_taxes.actual_amount",
        _fence(
            origin=(
                "not a us-data calibration target: us-data calibrates the CBO "
                "income-by-source projections (CBO_INCOME_BY_SOURCE_TARGETS), not "
                "the total-revenue outturn."
            ),
            purpose="n/a — an aggregate outturn, never a per-record target.",
            verdict_basis=(
                "no fence to rebuild; component income is compiled via "
                "cbo.revenue_projection."
            ),
        ),
    ),
    "census.popproj2023": (
        "superseded",
        "Census Bureau population PROJECTIONS (forward vintage). us-data "
        "calibrated the OBSERVED resident population, compiled via census_pep.",
        "compiled sibling census_pep.v2024; sample fact "
        "census.popproj2023.cy2023.national_population.age_0.population",
        _fence(
            origin=(
                "not a us-data calibration target: us-data calibrates observed "
                "Census population (census_pep), not the projection series."
            ),
            purpose="n/a — a forward projection, never the observed control.",
            verdict_basis=(
                "superseded by compiled census_pep.v2024 / "
                "national_resident_population_age (observed resident population)."
            ),
        ),
    ),
    "census_acs.acs1_2023": (
        "survey_derived",
        "American Community Survey 1-year estimates — a household SAMPLE SURVEY, "
        "not an administrative universe. Used at congressional-district grain "
        "only, off for the national release.",
        "census_acs facts return None unless include_congressional_district_"
        "targets in fiscal_targets._reference_from_ledger_fact; sample fact "
        "census_acs.acs1_2023.b01001.female_age.01.age_15_to_17.female_population",
        _fence(
            origin=(
                "not a national us-data calibration target: ACS feeds regional/CD "
                "H5 targets in us-data, off for the national build."
            ),
            purpose="n/a nationally — survey-derived CD detail, not a national fence.",
            verdict_basis=(
                "off_by_default + survey_derived: excluded by principle (ACS is a "
                "sample survey) and only relevant at CD grain "
                "(include_congressional_district_targets, off here)."
            ),
        ),
    ),
    "cms_nhe.medicaid_title_xix_expenditures": (
        "non_linear",
        "CMS NHE Medicaid Title XIX spending — a real us-data target (PR #292) "
        "whose linear-target assumption PolicyEngine-US retired: medicaid cost is "
        "now a person_weight-dependent allocation, so this is not a linear "
        "calibration row. Kept as calibration_role=validation_only.",
        "DIRECT_LEDGER_TARGETS ('cms_nhe','expenditure_amount','medicaid_title_"
        "xix') metadata calibration_role=validation_only",
        _fence(
            origin="us-data PR #292 (medicaid spending target)",
            purpose=(
                "keep weighted Medicaid outlays aligned to the CMS national total "
                "(nation/hhs/medicaid_spending). No discoverable prose rationale "
                "in PR #292 (only a value/source note); the mechanical purpose is "
                "the aggregate-spending control."
            ),
            verdict_basis=(
                "architecture retires it: PolicyEngine-US PR #1138 made "
                "medicaid_cost_if_enrolled an SLCSP allocation from state totals "
                "normalized against enrollees — person_weight-dependent, so "
                "reweighting recomputes per-person costs and it is not a linear "
                "calibration row. populace keeps it calibration_role="
                "validation_only in DIRECT_LEDGER_TARGETS."
            ),
        ),
    ),
    "hhs_acf_tanf.average_monthly_families": (
        "deferred",
        "HHS ACF TANF average-monthly family caseload. TANF is calibrated on "
        "BENEFIT DOLLARS (compiled); the caseload COUNT has no wired TANF-receipt "
        "indicator yet.",
        "compiled sibling hhs_acf_tanf.cash_assistance; sample fact "
        "hhs_acf_tanf.fy2024.average_monthly_families.us.us_total.total_families",
        _fence(
            origin=(
                "not a us-data calibration target: us-data calibrates TANF dollars "
                '(HARD_CODED_TOTALS "tanf"), compiled via cash_assistance; the '
                "caseload count is not a us-data target."
            ),
            purpose="n/a — TANF dollars are the target, not the caseload count.",
            verdict_basis=(
                "deferred: no TANF-receipt indicator is wired (unlike the SNAP / "
                "LIHEAP household caseloads); the dollar target is compiled."
            ),
        ),
    ),
    "hhs_acf_tanf.average_monthly_recipients": (
        "deferred",
        "HHS ACF TANF average-monthly recipient caseload. As with families, TANF "
        "is calibrated on benefit dollars (compiled); the recipient COUNT has no "
        "wired TANF-receipt indicator.",
        "compiled sibling hhs_acf_tanf.cash_assistance; sample fact "
        "hhs_acf_tanf.fy2024.average_monthly_recipients.us.us_total."
        "total_recipients",
        _fence(
            origin=(
                "not a us-data calibration target: us-data calibrates TANF "
                "dollars (compiled); the recipient count is not a us-data target."
            ),
            purpose="n/a — TANF dollars are the target, not the recipient count.",
            verdict_basis=(
                "deferred: no TANF-receipt indicator wired; the dollar target is "
                "compiled via cash_assistance."
            ),
        ),
    ),
    "irs_soi.congressional_district_2022": (
        "off_by_default",
        "IRS SOI congressional-district table. CD-level targets are opt-in "
        "(include_congressional_district_targets=False for the national release); "
        "the national and state SOI surfaces are compiled instead.",
        "test_soi_congressional_district_targets_are_opt_in + the "
        "include_congressional_district_targets gate in "
        "fiscal_targets._soi_reference_from_fact",
        _fence(
            origin=(
                "us-data CD targets serve the regional/local H5 outputs "
                "(build_outputs/target_universe.py), not the national build."
            ),
            purpose=(
                "distributional shape within a congressional district for local "
                "H5 outputs."
            ),
            verdict_basis=(
                "off_by_default: CD targets are opt-in for the national release; "
                "enabling include_congressional_district_targets compiles them. "
                "The national and state SOI surfaces are compiled."
            ),
        ),
    ),
    "irs_soi.form_w2_401k_elective_deferrals": _RETIREMENT_CONTRIBUTION_EXCLUSION,
    "irs_soi.form_w2_designated_roth_401k_contributions": (
        _RETIREMENT_CONTRIBUTION_EXCLUSION
    ),
    "irs_soi.roth_ira_contributions": _RETIREMENT_CONTRIBUTION_EXCLUSION,
    "irs_soi.traditional_ira_contributions": _RETIREMENT_CONTRIBUTION_EXCLUSION,
    # irs_soi.form_w2_social_security_tips was a reviewed exclusion while
    # tip_income was a structural zero. The SIPP tips source stage now
    # populates it and #465/#474 wired the amount target (named role,
    # wages-series aging), so the family compiles and the gate promotes it.
    # The return_count sub-row alone remains a support exclusion
    # (US_FISCAL_TARGET_SUPPORT_EXCLUSIONS: tip support is under 1% of the
    # 6.04M-return Box 7 class; the count target waits for support widening,
    # populace#451 item 3).
    "kff.marketplace_effectuated_enrollment": (
        "superseded",
        "Kaiser Family Foundation state marketplace enrollment — a secondary "
        "aggregator of the CMS data us-data calibrated. ACA enrollment is fit "
        "from the primary CMS source, compiled.",
        "compiled sibling cms_aca.oep2024; sample fact "
        "kff.marketplace_effectuated_enrollment.2024.state.us."
        "total_effectuated_marketplace_enrollment",
        _fence(
            origin=(
                "not a us-data calibration target: us-data calibrates ACA "
                "enrollment from CMS (nation/gov/aca_enrollment), not KFF."
            ),
            purpose="n/a — a secondary compilation of the same CMS data.",
            verdict_basis=(
                "superseded by compiled cms_aca.oep2024 (the primary CMS "
                "administrative source); avoids a duplicate target."
            ),
        ),
    ),
    "usda_snap.national_average_monthly_persons": _SNAP_PERSONS_EXCLUSION,
    "usda_snap.state_average_monthly_persons": _SNAP_PERSONS_EXCLUSION,
}

# us-data targets the pinned feed carries NO ledger fact for. (reason, evidence,
# fence) with classification source_absent.
_SOURCE_ABSENT_US_DATA_FAMILIES: dict[str, tuple[str, str, dict[str, str]]] = {
    "bls.consumer_expenditure": (
        "BLS Consumer Expenditure childcare-expense target the retired pipeline "
        "calibrated (loss.py BLS_CE_TOTALS childcare_expenses = $63.09B). The "
        "pinned feed carries no BLS source fact.",
        "retired us-data pipeline (archived) loss.py BLS_CE_TOTALS + "
        "nation/bls/ce; feed source families carry no 'bls' source",
        _fence(
            origin="us-data loss.py BLS_CE_TOTALS (BLS CE LABSTAT)",
            purpose=(
                "anchor modeled childcare expenses to the BLS Consumer "
                "Expenditure aggregate ($63.09B, 2024)."
            ),
            verdict_basis=(
                "source-absent: the pinned feed carries no BLS CE ledger fact, so "
                "there is nothing to compile; deferred pending a Ledger BLS CE "
                "ingest."
            ),
        ),
    ),
    "wic.national_summary": (
        "USDA WIC national annual summary the retired pipeline calibrated "
        "(WIC_NATIONAL_ANNUAL_SUMMARY_SOURCE). The pinned feed carries no WIC "
        "fact; WIC receipt is modeled via the would_claim_wic input.",
        "retired us-data pipeline (archived) db/etl_national_targets.py "
        "WIC_NATIONAL_ANNUAL_SUMMARY_SOURCE; feed carries no 'wic' source",
        _fence(
            origin=(
                "us-data db/etl_national_targets.py WIC_NATIONAL_ANNUAL_SUMMARY_SOURCE"
            ),
            purpose="anchor WIC participation to the USDA national summary.",
            verdict_basis=(
                "source-absent: no WIC ledger fact in the feed. WIC receipt is "
                "modeled via the would_claim_wic take-up input, not a target here."
            ),
        ),
    ),
    "hud.housing_assistance": (
        "HUD housing-assistance aggregates the retired pipeline calibrated "
        "(db/etl_housing_assistance.py). The pinned feed carries no HUD fact.",
        "retired us-data pipeline (archived) db/etl_housing_assistance.py; feed "
        "carries no 'hud' source",
        _fence(
            origin="us-data db/etl_housing_assistance.py",
            purpose="anchor modeled housing assistance to HUD admin totals.",
            verdict_basis=(
                "source-absent: no HUD ledger fact in the feed; deferred pending a "
                "Ledger HUD ingest."
            ),
        ),
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
    # The N regime compiles the congressional-district surface (populace#449:
    # CD targets ON for Build N's timed A/B), so parity is declared against
    # the CD-on registry with the packaged vintage crosswalk.
    registry = compile_us_fiscal_target_registry(
        facts,
        target_period=TARGET_PERIOD,
        age_targets=True,
        include_congressional_district_targets=True,
        congressional_district_vintage_crosswalk=(
            load_congressional_district_vintage_crosswalk(
                default_congressional_district_vintage_crosswalk_path()
            )
        ),
    )
    registry, _ = apply_us_medicaid_enrollment_substitutions(registry)
    return {
        us_target_family_id(spec.name)
        for spec in registry.specs
        if us_target_family_id(spec.name)
    }


def _exclusion_for(family: str) -> tuple[str, str, str, dict[str, str]]:
    if family in _FAMILY_EXCLUSIONS:
        return _FAMILY_EXCLUSIONS[family]
    namespace = family.split(".", 1)[0]
    if namespace in _NAMESPACE_EXCLUSIONS:
        return _NAMESPACE_EXCLUSIONS[namespace]
    raise SystemExit(
        f"Feed family {family!r} neither compiles nor has a declared, fenced "
        "reviewed exclusion. Add an entry to _FAMILY_EXCLUSIONS or "
        "_NAMESPACE_EXCLUSIONS in tools/build_us_target_parity_manifest.py with a "
        "fence {origin, purpose, verdict_basis} — a category label alone is not a "
        "sufficient reason to drop a us-data-era calibration target."
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
            classification, reason, evidence, fence = _exclusion_for(family)
            families[family] = {
                "status": REVIEWED_EXCLUSION_STATUS,
                "classification": classification,
                "reason": reason,
                "evidence": evidence,
                "fence": fence,
            }

    invented = sorted(compiled - set(feed_counts))
    if invented:
        raise SystemExit(
            f"Registry compiles family(ies) with no feed fact: {invented}. "
            "The family-id derivation or the wiring is inconsistent."
        )

    for family, (reason, evidence, fence) in sorted(
        _SOURCE_ABSENT_US_DATA_FAMILIES.items()
    ):
        families[family] = {
            "status": REVIEWED_EXCLUSION_STATUS,
            "classification": SOURCE_ABSENT_CLASSIFICATION,
            "reason": reason,
            "evidence": evidence,
            "fence": fence,
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
                "retired us-data pipeline (archived): utils/loss.py (eCPS loss "
                "matrix), db/etl_national_targets.py, db/etl_*.py, "
                "utils/national_target_parity.py"
            ),
            "family_granularity": (
                "namespace.concept of the ledger source_record_id (us_target_family_id)"
            ),
            "governing_rule": (
                "us-data-targeted => compiled unless source-absent; every "
                "reviewed exclusion carries a Chesterton's-fence "
                "{origin, purpose, verdict_basis}"
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
