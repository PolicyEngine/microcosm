"""Regenerate the US release target-parity manifest and feed-family inventory.

The target-side analog of ``tools/build_us_release_input_coverage_manifest.py``.
It declares which administrative target *families* the compiled microcosm registry
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
  carries, with fact counts (``target_parity_feed_families.json``), and every
  dimension value the feed labels more than one way, with its labels
  (``dimension_value_label_variants``; microcosm#1343). CI reads the
  committed listing, so a re-pin that brings back a second label fails there
  without the feed.
- **Compiled families** = the families the one-surface registry compiles today
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

from microcosm.build.us_runtime import (
    default_congressional_district_vintage_crosswalk_path,
    load_congressional_district_vintage_crosswalk,
)
from microcosm.build.us_runtime.chronicle_feed import load_us_chronicle_feed
from microcosm.build.us_runtime.fiscal_targets import compile_us_fiscal_target_registry
from microcosm.build.us_runtime.medicaid_take_up import (
    apply_us_medicaid_enrollment_substitutions,
)
from microcosm.build.us_runtime.release_target_parity import (
    COMPILED_STATUS,
    REVIEWED_EXCLUSION_STATUS,
    SOURCE_ABSENT_CLASSIFICATION,
    us_target_family_id,
    us_target_family_vintage_free_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
US_PACKAGE_DIR = (
    REPO_ROOT / "packages" / "microcosm-build" / "src" / "microcosm" / "build" / "us"
)
MANIFEST_PATH = US_PACKAGE_DIR / "target_parity_manifest.json"
FEED_FAMILIES_PATH = US_PACKAGE_DIR / "target_parity_feed_families.json"

DEFAULT_FEED_NAME = "consumer_facts_us_a9026cc.jsonl"
DEFAULT_FEED_PATH = (
    Path.home() / "PolicyEngine" / "_buildh-runtime" / "inputs" / DEFAULT_FEED_NAME
)
#: The pinned feed digest (us/chronicle_feed.json); any other feed is refused.
EXPECTED_FEED_SHA256 = load_us_chronicle_feed().facts_sha256
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
    "calibrated (PR #496/#554) but which it and microcosm both moved to the input "
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
            "architecture retires it: us-data PR #1125 and microcosm both govern "
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
_JCT_OBBBA_NO_TAX_ANCHOR_EXCLUSION = (
    "deferred",
    "JCT OBBBA Title VII no-tax-on-tips / no-tax-on-overtime revenue-effect "
    "projections (JCX-35-25, FY2026-FY2029), banked as the exact-valued "
    "official anchors for the OBBBA deduction channels (microcosm#451 items "
    "3-4; the only such anchors — no BLS or Census FLSA-premium aggregate "
    "exists, and Treasury filing-season releases publish claim floors only). "
    "Not compilable at the 2024 target period: both provisions are effective "
    "tyba 12/31/24, so a neutralize-variable reform_minus_baseline_income_tax "
    "matrix row is structurally zero at 2024 law. The 2026-law "
    "reform-coverage probes carry the anchors today "
    "(fsla_overtime_premium_neutralization, obbba_no_tax_on_overtime, "
    "tip_income_neutralization, obbba_no_tax_on_tips).",
    "reform_coverage_smoke, certified N (c3e378a-20260722T010408Z): overtime "
    "probes -$16.86B at 2026 law vs JCX-35-25 FY2026 -$32.806B; tips probes "
    "-$1.63B vs -$10.121B; support oracle 8,748 carriers / $114.79B "
    "(fsla_overtime_premium), 549 / $34.28B (tip_income)",
    _fence(
        origin=(
            "not a us-data calibration target: OBBBA (P.L. 119-21, July 2025) "
            "postdates the retired us-data pipeline. The anchors enter with "
            "the microcosm#451 quartet sourcing (ledger "
            "jct-obbba-revenue-estimates-2025 package, JCX-35-25 pinned)."
        ),
        purpose=(
            "exact-valued validation anchors for the OBBBA qualified-tips and "
            "qualified-overtime deduction channels, and the declared-value "
            "source for future reform-encoded jct targets (the "
            "fiscal_target_references.json SimpleTaxExpenditureReform "
            "pattern) once a build's target period sits inside TY2025-TY2028 "
            "law. Two conversion requirements are part of this fence so it "
            "cannot be followed naively: (a) sign — these facts are signed "
            "budget effects (negative = revenue loss) while a "
            "neutralize_variable reform_minus_baseline_income_tax row is "
            "positive when a deduction is removed, and the declared-reference "
            "contract currently allows only the identity value operation, so "
            "activation needs an explicit sign-normalizing step (extend "
            "allowed_value_operations or mint magnitude-normalized companion "
            "facts); (b) scope — the tips line bundles the section 45B "
            "employer-credit expansion (its FY2030-2034 remnant), which a "
            "tax-unit income-tax delta does not model, so tips activation "
            "must subtract or separately fence that component."
        ),
        verdict_basis=(
            "deferred, not droppable: the channel is modeled "
            "(overtime_income_deduction, tip_income_deduction) but both "
            "provisions score exactly $0 at TARGET_PERIOD 2024 law, so a "
            "compiled 2024-period row cannot bind; the 2026-law probes are "
            "the live gate surface until the target period advances."
        ),
    ),
)

_ESI_ORIGIN = (
    "a REAL us-data calibration target: utils/loss.py pinned "
    "employer_sponsored_insurance_premiums at 1_002.9e9 (this series, the NIPA "
    "7.8 employer-contribution aggregate, before BEA's 2026-09-30 revision; "
    "verified at archived commit 2e044571, loss.py line 33, with "
    "etl_national_targets carrying the row), removed at 9f320d2c ('Remove "
    "survey-based national calibration targets', PR #831). The input itself "
    "was imputed by the retired eCPS (MEPS-IC priors over ASEC NOW_* "
    "policyholder fields, archived commit 42ed5d45 datasets/cps/cps.py lines "
    "197-271 and 1575-1581)."
)

_ESI_NHE_ORIGIN = (
    "not itself a us-data calibration target: the retired pipeline calibrated "
    "employer_sponsored_insurance_premiums to the BEA NIPA 7.8 series (family "
    "bea_nipa.private_group_health_insurance; utils/loss.py 1_002.9e9 at "
    "archived commit 2e044571, removed at 9f320d2c, PR #831). The v9.2 feed "
    "banked the NHE Table 24 series for the microcosm#454 gate before "
    "decision d1317 moved the stage's anchor to BEA."
)

_ESI_BEA_ANCHOR_EXCLUSION = (
    "deferred",
    "BEA NIPA Table 7.8 line 17 (B4923C), employer contributions for private "
    "group health insurance: the scale anchor of the meps_esi_premiums base "
    "stage (microcosm#454, decision d1317). The stage prices every current ESI "
    "policyholder from MEPS-IC, scales that universe to this fact (BEA State "
    "Personal Income methodology, December 2025, paragraphs 3.18-3.19: "
    "contributions for active and retired employees) and writes the pe-us "
    "input (employer_sponsored_insurance_premiums) for employed policyholders "
    "only. So the fact is not a sum target on the column: the column carries "
    "about 88% of it and the rest is priced, not exported. Validation-gated "
    "instead: us_esi_premiums_anchor_gate recomputes the anchor-universe total "
    "at release weights and holds it within 5% of this fact, and the release "
    "tool refuses a feed whose values differ from the stage's pinned anchor "
    "(us_esi_premiums_anchor_feed_mismatches).",
    "experiments/us-esi-454/receipts/stage_on_pool_2023_2025.json: 92.4M "
    "weighted current policyholders; anchor-universe total $977.0B at scale "
    "factor 0.971; employed column $863.9B (88.4%); other policyholders "
    "$113.2B; private-sector part of the column $665.6B against MEPS-IC's own "
    "private active-employee total of $668.6B (2024)",
    _fence(
        origin=_ESI_ORIGIN,
        purpose=(
            "anchor the CBO market-income ESI component once the column has a "
            "live producer. The BEA annual update of 2026-09-30 revised CY2024 "
            "from the $1,002.9B the retired target used to $977.034B."
        ),
        verdict_basis=(
            "deferred, not dropped: the producer is live and scaled to this "
            "fact, but the column is the employed policyholders' part of it, "
            "so the fact cannot be compiled as a sum target on the column. "
            "Release validation (us_esi_premiums_anchor_gate, both dense and "
            "sparse exports) holds the anchor-universe total to it. Whether "
            "and how to calibrate the column is decision d1264."
        ),
    ),
)

_ESI_NHE_EXCLUSION = (
    "deferred",
    "NHE Table 24 employer contribution to ESI premiums, a second estimate of "
    "the concept the meps_esi_premiums stage scales to (microcosm#454): "
    "employer contributions for active employees, COBRA enrollees and "
    "retirees (NHEA methodology paper 2024, p. 29). Decision d1317 chose BEA "
    "NIPA 7.8 line 17 as the stage's anchor instead; this fact is 7.2% above "
    "it for CY2024. It is neither compiled nor gated: "
    "us_esi_premiums_anchor_gate records the anchor-universe total against it "
    "beside the verdict (cross_check_ratio).",
    "experiments/us-esi-454/receipts/stage_on_pool_2023_2025.json: "
    "anchor-universe total $977.0B, 0.933 of NHE's $1,047.0B; scaled to NHE "
    "instead, the employed column would be $925.7B",
    _fence(
        origin=_ESI_NHE_ORIGIN,
        purpose=(
            "a cross-check on the ESI anchor's level: an independent estimate "
            "of the same active-plus-retired concept from the national health "
            "accounts."
        ),
        verdict_basis=(
            "deferred, not dropped: the stage's anchor is BEA (d1317), and "
            "this fact, like it, includes retirees, so it is not a sum target "
            "on the employed column. It is recorded beside the anchor "
            "verdict. Whether and how to calibrate the column is decision "
            "d1264."
        ),
    ),
)

_ESI_PRIVATE_EMPLOYER_EXCLUSION = (
    "deferred",
    "NHE Table 24 private-employer contribution to ESI premiums, the private "
    "part of the NHE all-employer fact (microcosm#454). It is neither "
    "compiled nor gated: the stage scales one factor to BEA NIPA 7.8 line 17 "
    "over every current ESI policyholder, and nothing holds the private-sector "
    "part of the column to this fact, which also counts private retirees and "
    "COBRA enrollees. The private part is instead recorded beside the anchor "
    "verdict against MEPS-IC's own active-employee total "
    "(us_esi_premiums_anchor_gate, private_active_cross_check).",
    "experiments/us-esi-454/receipts/stage_on_pool_2023_2025.json: "
    "private-sector part of the column $665.6B; MEPS-IC private "
    "active-employee total $668.6B (2024)",
    _fence(
        origin=_ESI_NHE_ORIGIN,
        purpose=(
            "a sector cross-check that private employers' share of the ESI "
            "total is in range."
        ),
        verdict_basis=(
            "deferred, not dropped: no gate reads this fact. The private part "
            "of the column is compared with MEPS-IC's active-employee total "
            "instead, because this fact includes private retirees and COBRA "
            "enrollees. Whether and how to calibrate the column is decision "
            "d1264."
        ),
    ),
)

_FAMILY_EXCLUSIONS: dict[str, tuple[str, str, str, dict[str, str]]] = {
    "bea_nipa.private_group_health_insurance": _ESI_BEA_ANCHOR_EXCLUSION,
    "cms_nhe.esi_employer_contribution_premiums": _ESI_NHE_EXCLUSION,
    "cms_nhe.esi_private_employer_contribution_premiums": (
        _ESI_PRIVATE_EMPLOYER_EXCLUSION
    ),
    "jct.obbba_title_vii": _JCT_OBBBA_NO_TAX_ANCHOR_EXCLUSION,
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
                'variables" (utils/loss.py:67-70 at '
                "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe, added by PR #1059)"
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
        "not an administrative universe. The compiler admits ACS population-age "
        "facts only at congressional-district geography, while every acs1_2023 "
        "fact in this pinned feed is state-grain.",
        "fiscal_targets._reference_from_ledger_fact accepts census_acs only when "
        "geography_level == congressional_district; sample fact "
        "census_acs.acs1_2023.b01001.female_age.01.age_15_to_17.female_population",
        _fence(
            origin=(
                "ACS feeds regional/CD detail in us-data, but the pinned family "
                "contains state-grain sample-survey facts rather than the "
                "compiler's CD population-age facts."
            ),
            purpose="survey-derived geographic detail, not an administrative control.",
            verdict_basis=(
                "survey_derived and mechanically inert: every acs1_2023 fact in "
                "the feed is state-grain (468/468 verified), so the CD-only ACS "
                "source gate compiles none of them."
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
                "calibration row. microcosm keeps it calibration_role="
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
    # microcosm#451 item 3), at every vintage since decision d179
    # (US_FISCAL_TARGET_ALL_VINTAGE_SUPPORT_EXCLUSIONS, microcosm#956).
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

#: Keyed by vintage-free family id, so a repin to the next IRS
#: congressional-district release keeps its note.
_COMPILED_FAMILY_NOTES = {
    "irs_soi.congressional_district": (
        "One US target surface: national, state, and congressional-district "
        "facts compile for every calibrated artifact. Geography is a constraint; "
        "artifact record count changes only L0, never target membership. "
        "There is no local-versus-national surface. "
        "Congressional-district rows inform shape while national and state "
        "actuals retain control ownership."
    ),
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
        "fact; WIC receipt is modeled via the takes_up_wic_if_eligible input.",
        "retired us-data pipeline (archived) db/etl_national_targets.py "
        "WIC_NATIONAL_ANNUAL_SUMMARY_SOURCE; feed carries no 'wic' source",
        _fence(
            origin=(
                "us-data db/etl_national_targets.py WIC_NATIONAL_ANNUAL_SUMMARY_SOURCE"
            ),
            purpose="anchor WIC participation to the USDA national summary.",
            verdict_basis=(
                "source-absent: no WIC ledger fact in the feed. WIC receipt is "
                "modeled via the takes_up_wic_if_eligible take-up input, not a target here."
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


def _dimension_value_label_variants(
    facts: list[dict],
) -> dict[str, dict[str, list[str]]]:
    """Every (dimension, value id) the feed labels more than one way.

    Labels come from each row's ``dimension_value_labels`` and, for the row's
    group-by value, ``layout.groupby_value_label``, stripped as the ledger
    compiler strips them. Returned as {dimension: {value id: sorted labels}}.
    """
    labels_by_value: dict[tuple[str, str], set[str]] = {}

    def record(dimension_id: object, value_id: object, label: object) -> None:
        text = str(label).strip() if label is not None else ""
        if dimension_id and value_id is not None and text:
            key = (str(dimension_id), str(value_id))
            labels_by_value.setdefault(key, set()).add(text)

    for fact in facts:
        for dimension_id, value_labels in (
            fact.get("dimension_value_labels") or {}
        ).items():
            for value_id, label in (value_labels or {}).items():
                record(dimension_id, value_id, label)
        layout = fact.get("layout") or {}
        record(
            layout.get("groupby_dimension"),
            layout.get("groupby_value_id"),
            layout.get("groupby_value_label"),
        )

    variants: dict[str, dict[str, list[str]]] = {}
    for (dimension_id, value_id), labels in sorted(labels_by_value.items()):
        if len(labels) > 1:
            variants.setdefault(dimension_id, {})[value_id] = sorted(labels)
    return variants


def _compiled_families(facts: list[dict]) -> set[str]:
    # Parity is declared against the one registry every calibrated artifact
    # compiles, using the canonical source-to-current CD vintage translation.
    registry = compile_us_fiscal_target_registry(
        facts,
        target_period=TARGET_PERIOD,
        age_targets=True,
        congressional_district_vintage_crosswalk=load_congressional_district_vintage_crosswalk(
            default_congressional_district_vintage_crosswalk_path()
        ),
    )
    registry, _ = apply_us_medicaid_enrollment_substitutions(registry)
    families = {
        us_target_family_id(spec.name)
        for spec in registry.specs
        if us_target_family_id(spec.name)
    }
    return families


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
            note = _COMPILED_FAMILY_NOTES.get(us_target_family_vintage_free_id(family))
            if note is not None:
                families[family]["note"] = note
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
                "compile_us_fiscal_target_registry(age_targets=True, canonical "
                "CD vintage crosswalk) + apply_us_medicaid_enrollment_substitutions; "
                "one national + state + congressional-district surface for every "
                "artifact (microcosm#449)"
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
        "dimension_value_label_variants": _dimension_value_label_variants(facts),
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
    if feed_sha256 != EXPECTED_FEED_SHA256:
        raise SystemExit(
            f"Feed sha256 {feed_sha256} does not match the pinned feed "
            f"{EXPECTED_FEED_SHA256}; refusing to regenerate against an "
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
