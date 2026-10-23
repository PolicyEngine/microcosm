"""Federal student loan balances from the 2025 Survey of Consumer Finances.

microcosm#1073 carries SCF education debt into the US build for the student debt
pilot (PolicyEngine/us-student-debt#1). It writes four person inputs of
PolicyEngine-US: ``federal_student_loan_balance`` and
``federal_parent_plus_loan_balance`` (policyengine-us#9721), and
``is_income_based_repayment_new_borrower`` and
``federal_student_loan_graduate_share`` (policyengine-us#9724).

**Vintage.** Only education debt reads the 2025 SCF. Net worth, the liquid-asset
leaves and auto loans stay on the 2022 SCF in :mod:`scf_wealth` and
:mod:`scf_auto_loans` until the whole ``scf_wealth`` stage moves. The 2025 wave
is three years closer to the build, follows the end of the payment pause, and
records each loan's year borrowed and program type, which 2022 does not. Every
2025-specific coordinate lives in :data:`SCF_2025_SUMMARY_EXTRACT` and
:data:`SCF_2025_FULL_EXTRACT`, and the predictor construction is the shared
summary-extract one from :mod:`scf_wealth`. The 2025 summary extract carries
every column that construction reads, so moving the stage is a change of pinned
coordinates, not of code.

**Donor targets, per SCF family and implicate.** For each of education loans
1-6 the full public file records the amount still owed (X7824, X7847, X7870,
X7924, X7947, X7970), whether it is a federal loan (X7879, X7884, X7889, X7894,
X7899, X7994; 1 = yes), whose education it paid for (X7978, X7883, X7888,
X7893, X7898, X7993), the year borrowed (X9332-X9337), the program it financed
(X9338-X9343) and the amount borrowed (X7805, X7828, X7851, X7905, X7928,
X7951). Loans beyond the sixth are one total, X7179, with none of these items.

- Only loans flagged federal (code 1) count. A loan whose federal status is
  not recorded (code 0; the 2025 public file leaves some "don't know" answers
  blank) counts as non-federal.
- **X7179 rule.** The remaining-loans total is allocated in proportion to the
  family's balances on loans 1-6: every loan 1-6 is scaled by
  ``1 + X7179 / (sum of balances on loans 1-6)``. This applies the family's
  federal share of loans 1-6 to X7179, and splits the federal part among people
  in the same proportions as loans 1-6. A family with X7179 but no balance on
  loans 1-6 contributes nothing (none in the public files read).
- **Whose-education rule** (:data:`SCF_WHOSE_EDUCATION_HOLDER`). Code 1
  (respondent) is the head's own loan and code 2 (spouse or partner) the
  spouse's; both go to ``federal_student_loan_balance``. Where X8000 = 1 the
  public file reverses every respondent and spouse/partner variable so that
  code 1 is the SCF reference person, but it left the codes of loans 4 and 6
  (X7893, X7993) unreversed. The test: a completed loan for a degree should
  be consistent with the coded person's highest degree (X5931 for the
  reference person, X6111 for the spouse or partner). Of the loans consistent
  with exactly one partner's degree, loans 1, 2, 3 and 5 match the coded person
  in reversed and other families alike, and loans 4 and 6 match in all 25 such
  loans (five implicates) of families that were not reversed, but in none of
  the 54 of reversed families (10 in implicate 1). So in a reversed family
  codes 1 and 2 of loans 4 and 6 are swapped back
  (:data:`SCF_WHOSE_EDUCATION_UNREVERSED_LOANS`); the receipt reruns the test.
  The codebook names no such exception, so the correction is tied to the
  pinned member SHA-256. Code 6 (multiple people), code 7 (unrelated person)
  and code -7 (other) go to the head's own balance: none can be a Parent PLUS
  loan for one dependent student, and the SCF reference person is the
  household's default borrower. Any other code (the public file folds 4 and 5
  into 3, so none is expected) also goes to the head.
- **Loans for a child's education** (code 3, child or stepchild; the public
  file folds grandchild and other relative into it). The survey asks about
  the loans of the family living in the household, and the screener (X7801)
  leaves out loans of people who live elsewhere, so a code-3 loan is a
  co-resident child's own loan or the parent's Parent PLUS loan. It is the child's own
  (``child_federal_student_loan_balance``) when the household listing
  (persons 3-12, X108-X226) has a child (code 4) aged 18 or over who could
  have been in school when it was borrowed (16 or older in that year, at the
  oldest age the child's public-file age band allows: the file bands listed
  ages, so 18-27 all read 25), unless only a parent could have borrowed it:
  an undergraduate or certificate loan whose amount borrowed exceeds the
  $31,000 aggregate limit on a dependent undergraduate's own Direct Loans
  (34 CFR 685.203(e)(1)), borrowed while the child could still have been
  under 24, and so dependent (20 U.S.C. 1087vv(d)(1)(A)). Parent PLUS
  finances only a dependent undergraduate (34 CFR 685.200(c)(2)(i)), so a
  graduate loan, or one borrowed after every fitting child turned 24, is
  never Parent PLUS. Every other code-3 loan is the head's
  ``federal_parent_plus_loan_balance``: no listed adult child could have
  borrowed it, so the student lives elsewhere and only the parent's loan is
  reported, or it exceeds the dependent limit. An undergraduate loan within
  the limit for a child of dependent age could be either, and is read as the
  child's own, the household member the listing names; the receipt reports
  the Parent PLUS share under the other reading too.
- **IBR new-borrower status** is true when the person holds a federal loan and
  every federal loan the person holds (own, and for the head Parent PLUS too,
  since 34 CFR 685.209(b)(13)(ii) looks at any outstanding Direct or FFEL
  balance) was borrowed in 2014 or later. The item records a year only, so
  2014 counts as after July 1, 2014. The share is an upper bound: the SCF sees
  only loans still owed, so a borrower whose pre-2014 loans have been paid
  off reads as new, and a consolidated loan is dated by its most recent
  underlying loan, so a consolidation of pre-2014 loans can read as new too.
- **Graduate share** is the share of the original amount of the person's own
  federal loans (Parent PLUS excluded) that financed a master's, professional
  or doctoral program (codes 13-15), weighted by the amount borrowed (falling
  back to the balance when the amount is missing), as 34 CFR 685.209(f)(1)
  weights the SAVE rates.

**Imputation.** Two regime-gated weighted QRFs (:class:`microcosm.fit.QRF`)
condition on the same predictors (:data:`SCF_EDUCATION_DEBT_PREDICTORS`): the
reference person's age, sex and race, whether a spouse or partner is present,
the number of children, wages, and income excluding capital gains. The first
draws each recipient household's federal education debt from every SCF family.
The second, fit on borrower families only, conditions also on that total and
on the number of adult children (children of the reference person or partner
aged 18 or over), and draws the chained composition: the Parent PLUS, adult
child's, head's and spouse's shares of the total, in that order, then the
three graduate shares and the three new-borrower flags. The four shares are
clipped to [0, 1] and normalized to sum to one, and a borrower drawn with no
share holds its own loan. Parent PLUS is therefore drawn for itself, first: a
family that resembles donors without Parent PLUS draws none, instead of
receiving whatever the own shares leave.

Two of the shared SCF predictors are left out
(:data:`SCF_EDUCATION_DEBT_CANDIDATE_PREDICTORS` keeps the full list for the
diagnostics). Interest and dividends, and Social Security and pensions, are
measured differently on the two sides: the CPS host carries tax-record
imputations, the SCF survey reports. At reference ages 25-34, 42% of host
households have interest or dividends against 16% of SCF families, and 18%
have Social Security or pension income against 6%. A QRF reads equal values
as equal positions, so the host's older-looking households drew less debt:
mapping these two predictors to their SCF distribution at equal rank moves the
host's borrower share most of any predictor
(``experiments/receipts/1073-predictor-shift-diagnosis.json``). Wages and
total income stay in, because the issue asks for income and they carry the
income gradient.

**Recipients** (:data:`SCF_EDUCATION_DEBT_RECIPIENT_DEFINITION`). The CPS side
builds each predictor the way the SCF defines it. The reference person is the
CPS householder or, when a spouse or cohabiting partner is present, the man of
a mixed-sex couple or else the older partner: the SCF's ordering of its
economically dominant couple. ``is_married`` counts a cohabiting partner, as
SCF ``MARRIED`` does, and children count those of the reference person or
partner, as SCF ``KIDS`` does; adult children are the ones aged 18 or over,
as listing code 4 aged 18 or over counts them. Incomes sum over the reference
person, spouse or partner and their children: the SCF primary economic unit,
which leaves out a financially independent co-resident relative or roommate.
The spouse is the member whose line number is the reference person's
``A_SPOUSE``, or else its ``PECOHAB`` partner when that column is present. A
household drawn with a spouse's balance but no spouse or partner present
gives that balance to the reference person, whose graduate share becomes the
balance-weighted mean of the two and whose new-borrower flag requires both.
The child's draw goes to the youngest child of the reference person or
partner aged 18 or over (``PEPAR1``/``PEPAR2``). A household without one
cannot use it, and the draw makes it the head's Parent PLUS loan, the SCF
side's rule for a loan whose student lives elsewhere. Everyone else gets zero.
:data:`SCF_EDUCATION_DEBT_FIRST_RECEIPT_DEFINITION` keeps the first receipt's
construction (oldest adult, ``A_MARITL`` marriage, own children, whole
household) for its before/after table.

**Known limits.** The SCF asks about the loans of the family living in the
household (its primary economic unit), so a financially independent adult in
someone else's household is missed unless the family reports the loan as one
for a child's education. No education-loan item names the listed person a
loan financed, and the public file bands listed ages, so the child rule works
from age bands, and with two or more adult children the youngest holds the
loan. The SCF federal flag includes FFEL and Perkins loans that
``federal_student_loan_balance`` excludes until consolidated. Balances are
2025 survey-date dollars. The FSA calibration (microcosm#1074) pins totals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_manifest import SourceStageSpec, load_source_manifest
from microcosm.build.us_runtime.eligibility_inputs import _own_children_in_household
from microcosm.build.us_runtime.scf_auto_loans import _scf_reference_person_mask
from microcosm.build.us_runtime.scf_wealth import (
    SCF_WEALTH_PREDICTORS,
    _recipient_cps_race,
    _replace_sentinels,
    _scf_summary_predictor_table,
    _sum_present,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "SCF_2025_FULL_EXTRACT",
    "SCF_2025_SUMMARY_EXTRACT",
    "SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR",
    "SCF_EDUCATION_DEBT_BALANCE_BANDS",
    "SCF_EDUCATION_DEBT_BORROWER_AGE_BANDS",
    "SCF_EDUCATION_DEBT_CANDIDATE_PREDICTORS",
    "SCF_EDUCATION_DEBT_COMPOSITION_TARGETS",
    "SCF_EDUCATION_DEBT_EXCLUDED_PREDICTORS",
    "SCF_EDUCATION_DEBT_FIRST_RECEIPT_DEFINITION",
    "SCF_EDUCATION_DEBT_HOLDER_SHARE_TARGETS",
    "SCF_EDUCATION_DEBT_PREDICTORS",
    "SCF_EDUCATION_DEBT_RECIPIENT_DEFINITION",
    "SCF_EDUCATION_DEBT_TOTAL_TARGET",
    "SCF_EDUCATION_DEBT_VINTAGE",
    "SCF_EDUCATION_LOAN_AMOUNT_BORROWED_COLUMNS",
    "SCF_EDUCATION_LOAN_BALANCE_COLUMNS",
    "SCF_EDUCATION_LOAN_FEDERAL_COLUMNS",
    "SCF_EDUCATION_LOAN_PROGRAM_COLUMNS",
    "SCF_EDUCATION_LOAN_REMAINING_BALANCE_COLUMN",
    "SCF_EDUCATION_LOAN_WHOSE_COLUMNS",
    "SCF_EDUCATION_LOAN_YEAR_COLUMNS",
    "SCF_HOUSEHOLD_LISTING_AGE_COLUMNS",
    "SCF_HOUSEHOLD_LISTING_RELATIONSHIP_COLUMNS",
    "SCF_REVERSED_COLUMN",
    "SCF_WHOSE_EDUCATION_HOLDER",
    "SCF_WHOSE_EDUCATION_UNREVERSED_LOANS",
    "US_SCF_EDUCATION_DEBT_BALANCE_COLUMNS",
    "US_SCF_EDUCATION_DEBT_OUTPUT_COLUMNS",
    "RecipientDefinition",
    "ScfPublicFile",
    "derive_scf_education_debt_targets",
    "draw_scf_education_debt_households",
    "education_debt_fit_table",
    "fetch_scf_public_file",
    "impute_us_scf_education_debt",
    "load_scf_2025_education_debt_donor",
    "scf_education_debt_composition_predictors",
    "scf_education_debt_donor_fit_table",
    "us_scf_education_debt_fit_table",
    "us_scf_education_debt_signal_gate",
    "us_scf_education_debt_stage_spec",
    "us_scf_education_debt_summary",
    "with_us_scf_education_debt_inputs",
]

_SCF_STAGE_NAME = "scf_wealth"
_DONOR_WEIGHT_COLUMN = "scf_weight"
_HOUSEHOLD_ID_COLUMN = "person_household_id"
_DEFAULT_N_ESTIMATORS = 100
_SCF_HTTP_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

#: The SCF wave education debt reads (``scf_wealth`` itself stays on 2022).
SCF_EDUCATION_DEBT_VINTAGE = "2025"


@dataclass(frozen=True)
class ScfPublicFile:
    """One pinned Federal Reserve SCF public file: a Stata member in a zip."""

    url: str
    member: str
    zip_sha256: str
    member_sha256: str
    rows: int

    @property
    def zip_name(self) -> str:
        return self.url.rsplit("/", maxsplit=1)[-1]


#: The 2025 public summary extract. Both 2025 files were released 2026-10-09;
#: these digests match a fresh Federal Reserve download of 2026-10-12. Each
#: file has one row per family and implicate (4,360 families x 5).
SCF_2025_SUMMARY_EXTRACT = ScfPublicFile(
    url="https://www.federalreserve.gov/econres/files/scfp2025s.zip",
    member="rscfp2025.dta",
    zip_sha256="32dcf86be48b20fb5fd9e659f79fa22a4d9cccf456e44b6ca9b1329e9869bbeb",
    member_sha256="454a287f188bf13f94efc09820fa878482d897db1d2a16ac63bec14eb580d2dd",
    rows=21_800,
)
#: The 2025 full public file.
SCF_2025_FULL_EXTRACT = ScfPublicFile(
    url="https://www.federalreserve.gov/econres/files/scf2025s.zip",
    member="p25i6.dta",
    zip_sha256="c4010bd5decaa13d5d027862ed4b6c41b4106925089290f879749f8f5a467741",
    member_sha256="38926275dfb8f00ea843b80ced7a9412ad4653d39c7fab8fe24c3bdf912902ad",
    rows=21_800,
)

#: Education loans 1-6 (codebook "EDUCATION LOANS"): amount still owed.
SCF_EDUCATION_LOAN_BALANCE_COLUMNS: tuple[str, ...] = (
    "x7824",
    "x7847",
    "x7870",
    "x7924",
    "x7947",
    "x7970",
)
#: Total owed on all remaining loans beyond the sixth.
SCF_EDUCATION_LOAN_REMAINING_BALANCE_COLUMN = "x7179"
#: "Is this loan a federal student loan?" 1 = yes, 5 = no.
SCF_EDUCATION_LOAN_FEDERAL_COLUMNS: tuple[str, ...] = (
    "x7879",
    "x7884",
    "x7889",
    "x7894",
    "x7899",
    "x7994",
)
#: "For whose education was this loan taken out?"
SCF_EDUCATION_LOAN_WHOSE_COLUMNS: tuple[str, ...] = (
    "x7978",
    "x7883",
    "x7888",
    "x7893",
    "x7898",
    "x7993",
)
#: "In what year was this loan (or the most recent underlying loan) borrowed?"
SCF_EDUCATION_LOAN_YEAR_COLUMNS: tuple[str, ...] = (
    "x9332",
    "x9333",
    "x9334",
    "x9335",
    "x9336",
    "x9337",
)
#: "What type of educational program did this loan help finance?"
SCF_EDUCATION_LOAN_PROGRAM_COLUMNS: tuple[str, ...] = (
    "x9338",
    "x9339",
    "x9340",
    "x9341",
    "x9342",
    "x9343",
)
#: "How much was borrowed for this loan?" (or the balance after the most
#: recent consolidation or refinancing).
SCF_EDUCATION_LOAN_AMOUNT_BORROWED_COLUMNS: tuple[str, ...] = (
    "x7805",
    "x7828",
    "x7851",
    "x7905",
    "x7928",
    "x7951",
)
_SCF_FULL_EXTRACT_SPOUSE_AGE_COLUMN = "x19"
#: "Original respondent and spouse/partner variables reversed": 1 = reversed.
SCF_REVERSED_COLUMN = "x8000"
_SCF_REVERSED = 1
#: Loans (1-6) whose whose-education code the 2025 public file did not reverse
#: with the rest of a reversed record: loans 4 and 6 (X7893, X7993). See the
#: module docstring for the evidence.
SCF_WHOSE_EDUCATION_UNREVERSED_LOANS: tuple[int, ...] = (4, 6)
#: Household listing, persons 3-12: relationship to the respondent and age.
SCF_HOUSEHOLD_LISTING_RELATIONSHIP_COLUMNS: tuple[str, ...] = (
    "x108",
    "x114",
    "x120",
    "x126",
    "x132",
    "x202",
    "x208",
    "x214",
    "x220",
    "x226",
)
SCF_HOUSEHOLD_LISTING_AGE_COLUMNS: tuple[str, ...] = (
    "x110",
    "x116",
    "x122",
    "x128",
    "x134",
    "x204",
    "x210",
    "x216",
    "x222",
    "x228",
)
#: Listing relationship code 4: child (or child-in-law) of R or spouse/partner.
_SCF_LISTING_CHILD = 4
#: The public file codes listed members' ages in bands (codebook X110): 1-3 as
#: 3, 4-6 as 6, 7-12 as 12, 13-17 as 17, 18-25 as 25, and older ages rounded
#: to the nearest 5 (so 26-27 also read 25), top-coded at 95. Code -> the
#: youngest and oldest age it can stand for.
_SCF_LISTING_AGE_BANDS: Mapping[int, tuple[float, float]] = {
    3: (0.0, 3.0),
    6: (4.0, 6.0),
    12: (7.0, 12.0),
    17: (13.0, 17.0),
    25: (18.0, 27.0),
    95: (93.0, 120.0),
}
#: The first listing age code that is certainly an adult (18-27).
_SCF_LISTING_FIRST_ADULT_CODE = 25
#: The survey year that ages and years borrowed are counted from.
_SCF_SURVEY_YEAR = 2025
#: Youngest age at which a loan can have financed a listed child's schooling.
_EARLIEST_STUDENT_AGE = 16.0
#: A student 24 or older by December 31 of the award year is independent
#: (20 U.S.C. 1087vv(d)(1)(A)), so a parent cannot borrow Parent PLUS for them.
_INDEPENDENT_STUDENT_AGE = 24.0
#: Aggregate limit on a dependent undergraduate's own Direct Subsidized and
#: Unsubsidized Loans (34 CFR 685.203(e)(1)).
_DEPENDENT_UNDERGRADUATE_AGGREGATE_LIMIT = 31_000.0

_FEDERAL_LOAN_YES = 1
#: Programs counted as graduate or professional: master's (13), professional
#: school (14) and doctorate (15).
_GRADUATE_PROGRAM_CODES = frozenset({13, 14, 15})
#: First year counted as a new IBR borrower (20 U.S.C. 1098e(e): no
#: outstanding balance on July 1, 2014). The SCF records only the year.
_IBR_NEW_BORROWER_FIRST_YEAR = 2014

_HEAD = "head"
_SPOUSE = "spouse"
_CHILD = "child"
_PARENT_PLUS = "parent_plus"
_CHILD_EDUCATION = "child_education"
#: Whose-education code -> who holds the loan in the build. A loan for a
#: child's education is a co-resident child's own loan or the head's Parent
#: PLUS loan, by the rule in the module docstring.
SCF_WHOSE_EDUCATION_HOLDER: Mapping[int, str] = {
    1: _HEAD,
    2: _SPOUSE,
    3: _CHILD_EDUCATION,
    6: _HEAD,
    7: _HEAD,
    -7: _HEAD,
}

#: Household-total target of the first QRF.
SCF_EDUCATION_DEBT_TOTAL_TARGET = "federal_education_debt"
#: The second QRF's extra predictor: children of the reference person or
#: partner aged 18 or over in the household (SCF listing code 4).
SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR = "adult_children_in_household"
#: Each holder's share of the household total, drawn first in this order and
#: then normalized to sum to one.
SCF_EDUCATION_DEBT_HOLDER_SHARE_TARGETS: tuple[str, ...] = (
    "parent_plus_share",
    "child_own_share",
    "head_own_share",
    "spouse_own_share",
)
#: Chained targets of the second (borrower-family) QRF.
SCF_EDUCATION_DEBT_COMPOSITION_TARGETS: tuple[str, ...] = (
    *SCF_EDUCATION_DEBT_HOLDER_SHARE_TARGETS,
    "head_graduate_share",
    "spouse_graduate_share",
    "child_graduate_share",
    "head_is_ibr_new_borrower",
    "spouse_is_ibr_new_borrower",
    "child_is_ibr_new_borrower",
)
#: Every predictor both sides build: the shared SCF predictors plus income
#: excluding capital gains. The diagnostics and ablations compare all of them.
#: Educational attainment joins once microcosm#297 carries it on the CPS side.
SCF_EDUCATION_DEBT_CANDIDATE_PREDICTORS: tuple[str, ...] = (
    *SCF_WEALTH_PREDICTORS,
    "household_income",
)
#: Candidate predictors the QRFs leave out, each with its reason. Mapping each
#: to its SCF distribution at equal rank within five-year reference-person age
#: bins moves the host's borrower share from 14.2% to 15.8% (interest and
#: dividends) and 14.7% (Social Security and pensions); all candidates together
#: reach 17.3%, the SCF hold-out's level
#: (experiments/receipts/1073-predictor-shift-diagnosis.json).
SCF_EDUCATION_DEBT_EXCLUDED_PREDICTORS: Mapping[str, str] = {
    "interest_dividend_income": (
        "CPS interest and dividends are tax-record imputations: 42% of host "
        "households at reference ages 25-34 have some, against 16% of SCF "
        "families."
    ),
    "social_security_pension_income": (
        "CPS Social Security and pension income is likewise imputed: 18% of "
        "host households at reference ages 25-34 have some, against 6% of SCF "
        "families."
    ),
}
#: The predictors both QRFs condition on.
SCF_EDUCATION_DEBT_PREDICTORS: tuple[str, ...] = tuple(
    predictor
    for predictor in SCF_EDUCATION_DEBT_CANDIDATE_PREDICTORS
    if predictor not in SCF_EDUCATION_DEBT_EXCLUDED_PREDICTORS
)


def scf_education_debt_composition_predictors(
    predictors: Sequence[str] = SCF_EDUCATION_DEBT_PREDICTORS,
) -> tuple[str, ...]:
    """The second QRF's predictors: the first's, adult children and the total."""

    return (
        *predictors,
        SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR,
        SCF_EDUCATION_DEBT_TOTAL_TARGET,
    )


US_SCF_EDUCATION_DEBT_BALANCE_COLUMNS: tuple[str, ...] = (
    "federal_student_loan_balance",
    "federal_parent_plus_loan_balance",
)
US_SCF_EDUCATION_DEBT_OUTPUT_COLUMNS: tuple[str, ...] = (
    *US_SCF_EDUCATION_DEBT_BALANCE_COLUMNS,
    "is_income_based_repayment_new_borrower",
    "federal_student_loan_graduate_share",
)

#: Federal Student Aid portfolio bands (dollars, lower bound inclusive).
SCF_EDUCATION_DEBT_BALANCE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("under_5k", 0.0, 5_000.0),
    ("5k_to_10k", 5_000.0, 10_000.0),
    ("10k_to_20k", 10_000.0, 20_000.0),
    ("20k_to_40k", 20_000.0, 40_000.0),
    ("40k_to_60k", 40_000.0, 60_000.0),
    ("60k_to_80k", 60_000.0, 80_000.0),
    ("80k_to_100k", 80_000.0, 100_000.0),
    ("100k_to_200k", 100_000.0, 200_000.0),
    ("200k_plus", 200_000.0, np.inf),
)
#: Federal Student Aid borrower age bands (years, inclusive bounds).
SCF_EDUCATION_DEBT_BORROWER_AGE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("24_and_younger", -np.inf, 24.0),
    ("25_to_34", 25.0, 34.0),
    ("35_to_49", 35.0, 49.0),
    ("50_to_61", 50.0, 61.0),
    ("62_and_older", 62.0, np.inf),
)
#: SCF ``racecl4`` labels (2025 folds Middle Eastern or North African into 1).
_RACECL4_LABELS: Mapping[int, str] = {
    1: "white_non_hispanic",
    2: "black_non_hispanic",
    3: "hispanic",
    4: "other_or_multiple",
}

#: Summary-extract columns read beyond the shared predictor surface.
_SUMMARY_EXTRA_COLUMNS: tuple[str, ...] = (
    "y1",
    "yy1",
    "income",
    "kginc",
    "racecl4",
    "edn_inst",
    "educ",
)
#: Person income inputs summed into the recipient household's income, the CPS
#: counterpart of SCF ``income`` less ``kginc``. Absent columns count as zero.
_RECIPIENT_HOUSEHOLD_INCOME_COLUMNS: tuple[str, ...] = (
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "farm_operations_income",
    "farm_rent_income",
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "rental_income",
    "estate_income",
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
    "taxable_private_pension_income",
    "tax_exempt_private_pension_income",
    "taxable_ira_distributions",
    "taxable_401k_distributions",
    "taxable_403b_distributions",
    "taxable_sep_distributions",
    "unemployment_compensation",
    "alimony_income",
    "child_support_received",
    "veterans_benefits",
    "workers_compensation",
    "disability_benefits",
)
_RECIPIENT_INTEREST_DIVIDEND_COLUMNS = (
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
)
_RECIPIENT_SS_PENSION_COLUMNS = (
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
    "taxable_private_pension_income",
    "tax_exempt_private_pension_income",
)
_REQUIRED_RECIPIENT_PERSON_COLUMNS: tuple[str, ...] = (
    _HOUSEHOLD_ID_COLUMN,
    "age",
    "is_female",
    "PRDTRACE",
    "PRDTHSP",
    "A_MARITL",
    "A_LINENO",
    "A_SPOUSE",
    "PH_SEQ",
    "PEPAR1",
    "PEPAR2",
    "employment_income_before_lsr",
)
_OPTIONAL_PARTNER_COLUMN = "PECOHAB"

#: Age from which a CPS child of the reference person or partner is an adult.
_ADULT_AGE = 18.0

#: Weighted share of households holding federal education debt allowed by the
#: signal gate. The SCF 2025 share holding any education debt is 19.9%.
_BORROWER_HOUSEHOLD_SHARE_BAND = (0.08, 0.35)


# --------------------------------------------------------------------------- #
# Source files                                                                  #
# --------------------------------------------------------------------------- #
def _sha256_hexdigest(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def fetch_scf_public_file(
    public_file: ScfPublicFile,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download, verify and cache one pinned SCF public Stata member.

    A cached member whose SHA-256 matches the pin is reused without a network
    call; anything else is fetched from the Federal Reserve, and both the zip
    and the member must match their pins before the member is cached.
    """

    import io
    import urllib.request
    import zipfile

    root = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path.home() / ".cache" / "microcosm" / "scf"
    )
    root.mkdir(parents=True, exist_ok=True)
    target = root / public_file.member
    if target.exists() and target.stat().st_size > 0:
        if _sha256_hexdigest(target.read_bytes()) == public_file.member_sha256:
            return target

    request = urllib.request.Request(
        public_file.url,
        headers={"User-Agent": _SCF_HTTP_USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
        payload = response.read()
    digest = _sha256_hexdigest(payload)
    if digest != public_file.zip_sha256:
        raise ValueError(
            f"SCF {public_file.zip_name} failed sha-256 verification: "
            f"expected {public_file.zip_sha256}, got {digest}."
        )
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if public_file.member not in archive.namelist():
            raise ValueError(
                f"SCF {public_file.zip_name} is missing member {public_file.member!r}."
            )
        member_bytes = archive.read(public_file.member)
    digest = _sha256_hexdigest(member_bytes)
    if digest != public_file.member_sha256:
        raise ValueError(
            f"SCF {public_file.member} failed sha-256 verification: "
            f"expected {public_file.member_sha256}, got {digest}."
        )
    target.write_bytes(member_bytes)
    return target


def _read_stata_columns(
    path: str | Path,
    columns: Sequence[str],
    *,
    label: str,
    chunksize: int = 4_000,
) -> pd.DataFrame:
    """Read named columns of a Stata file in row chunks.

    The full public file is about 220 MB with over 5,000 variables; reading it
    whole would hold every variable in memory. Chunks keep only the requested
    columns resident.
    """

    wanted = list(dict.fromkeys(columns))
    try:
        with pd.read_stata(
            path,
            convert_categoricals=False,
            columns=wanted,
            chunksize=chunksize,
        ) as reader:
            parts = [chunk for chunk in reader]
    except ValueError as error:
        if "not found in the Stata data set" not in str(error):
            raise
        raise ValueError(f"SCF {label} missing required column(s): {error}.") from error
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    missing = sorted(set(wanted) - set(table.columns))
    if missing:
        raise ValueError(f"SCF {label} missing required column(s): {missing}.")
    return table.loc[:, wanted]


# --------------------------------------------------------------------------- #
# Donor targets                                                                 #
# --------------------------------------------------------------------------- #
def _numeric(table: pd.DataFrame, column: str) -> np.ndarray:
    return (
        pd.to_numeric(table[column], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def _listing_age_range(code: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The youngest and oldest age each public-file listing age code stands for.

    Codes 30-90 are ages rounded to the nearest 5, so ``code - 2`` to
    ``code + 2``; zero or negative (no such person) gives NaN.
    """

    code = np.asarray(code, dtype=np.float64)
    youngest = code - 2.0
    oldest = code + 2.0
    for band_code, (low, high) in _SCF_LISTING_AGE_BANDS.items():
        youngest = np.where(code == band_code, low, youngest)
        oldest = np.where(code == band_code, high, oldest)
    missing = code <= 0
    return np.where(missing, np.nan, youngest), np.where(missing, np.nan, oldest)


def _scf_whose_education_codes(full: pd.DataFrame) -> np.ndarray:
    """Whose-education codes of loans 1-6, with the X8000 correction applied.

    In a reversed family (X8000 = 1) the public file swaps every respondent and
    spouse/partner code so that code 1 is the reference person, except on
    loans 4 and 6, whose codes still name the original respondent: those two
    codes are swapped here.
    """

    whose = np.column_stack(
        [_numeric(full, c) for c in SCF_EDUCATION_LOAN_WHOSE_COLUMNS]
    ).reshape(len(full), len(SCF_EDUCATION_LOAN_WHOSE_COLUMNS))
    reversed_family = _numeric(full, SCF_REVERSED_COLUMN) == _SCF_REVERSED
    for loan in SCF_WHOSE_EDUCATION_UNREVERSED_LOANS:
        column = whose[:, loan - 1]
        whose[:, loan - 1] = np.where(
            reversed_family & (column == 1),
            2.0,
            np.where(reversed_family & (column == 2), 1.0, column),
        )
    return whose


def _scf_adult_children(full: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Listed children aged 18 or over: a (rows x 10) mask and the age codes."""

    shape = (len(full), len(SCF_HOUSEHOLD_LISTING_RELATIONSHIP_COLUMNS))
    relationship = np.column_stack(
        [_numeric(full, c) for c in SCF_HOUSEHOLD_LISTING_RELATIONSHIP_COLUMNS]
    ).reshape(shape)
    age_code = np.column_stack(
        [_numeric(full, c) for c in SCF_HOUSEHOLD_LISTING_AGE_COLUMNS]
    ).reshape(shape)
    adult_child = (relationship == _SCF_LISTING_CHILD) & (
        age_code >= _SCF_LISTING_FIRST_ADULT_CODE
    )
    return adult_child, age_code


def derive_scf_education_debt_targets(full: pd.DataFrame) -> pd.DataFrame:
    """Derive household federal education-debt targets from per-loan items.

    Pure and row-preserving: one output row per input (family, implicate) row.
    See the module docstring for the federal, X7179, X8000, whose-education,
    co-resident child, IBR and graduate-share rules this applies.

    Returns columns:
        ``federal_education_debt``: federal balance after the X7179 rule.
        ``head_federal_student_loan_balance``,
        ``spouse_federal_student_loan_balance``,
        ``child_federal_student_loan_balance``,
        ``federal_parent_plus_loan_balance``: its split by holder.
        ``head_own_share``, ``spouse_own_share``, ``child_own_share``,
        ``parent_plus_share``: each part over the total (zero for a family
        without federal education debt).
        ``head_is_ibr_new_borrower``, ``spouse_is_ibr_new_borrower``,
        ``child_is_ibr_new_borrower``: 0/1.
        ``head_graduate_share``, ``spouse_graduate_share``,
        ``child_graduate_share``: in [0, 1].
        ``adult_children_in_household``: listed children aged 18 or over.
        ``youngest_adult_child_age_code``: the youngest one's listing age code
        (0 when there is none).
        ``child_borrower_age_code``: the listing age code of the youngest
        child a child-own loan fits (0 when there is none).
        ``education_debt``: every education loan, federal or not (EDN_INST
        without the education-coded installment loans).
        ``federal_status_unrecorded_balance``: loans 1-6 balances whose federal
        flag is blank (counted as non-federal).
        ``remaining_loans_without_detail``: X7179 in a family with no balance
        on loans 1-6 (dropped).
    """

    required = (
        *SCF_EDUCATION_LOAN_BALANCE_COLUMNS,
        SCF_EDUCATION_LOAN_REMAINING_BALANCE_COLUMN,
        *SCF_EDUCATION_LOAN_FEDERAL_COLUMNS,
        *SCF_EDUCATION_LOAN_WHOSE_COLUMNS,
        *SCF_EDUCATION_LOAN_YEAR_COLUMNS,
        *SCF_EDUCATION_LOAN_PROGRAM_COLUMNS,
        *SCF_EDUCATION_LOAN_AMOUNT_BORROWED_COLUMNS,
        SCF_REVERSED_COLUMN,
        *SCF_HOUSEHOLD_LISTING_RELATIONSHIP_COLUMNS,
        *SCF_HOUSEHOLD_LISTING_AGE_COLUMNS,
    )
    missing = [column for column in required if column not in full.columns]
    if missing:
        raise ValueError(f"SCF education-debt derivation missing column(s): {missing}.")

    n = len(full)
    balances = np.column_stack(
        [np.maximum(_numeric(full, c), 0.0) for c in SCF_EDUCATION_LOAN_BALANCE_COLUMNS]
    ).reshape(n, len(SCF_EDUCATION_LOAN_BALANCE_COLUMNS))
    remaining = np.maximum(
        _numeric(full, SCF_EDUCATION_LOAN_REMAINING_BALANCE_COLUMN), 0.0
    )
    federal_code = np.column_stack(
        [_numeric(full, c) for c in SCF_EDUCATION_LOAN_FEDERAL_COLUMNS]
    ).reshape(balances.shape)
    whose = _scf_whose_education_codes(full)
    year = np.column_stack(
        [_numeric(full, c) for c in SCF_EDUCATION_LOAN_YEAR_COLUMNS]
    ).reshape(balances.shape)
    program = np.column_stack(
        [_numeric(full, c) for c in SCF_EDUCATION_LOAN_PROGRAM_COLUMNS]
    ).reshape(balances.shape)
    borrowed = np.column_stack(
        [
            np.maximum(_numeric(full, c), 0.0)
            for c in SCF_EDUCATION_LOAN_AMOUNT_BORROWED_COLUMNS
        ]
    ).reshape(balances.shape)

    has_balance = balances > 0
    federal = has_balance & (federal_code == _FEDERAL_LOAN_YES)
    loans_1_6 = balances.sum(axis=1)
    gross_up = np.ones(n, dtype=np.float64)
    positive_detail = loans_1_6 > 0
    gross_up[positive_detail] = (
        1.0 + remaining[positive_detail] / loans_1_6[positive_detail]
    )
    federal_balance = np.where(federal, balances, 0.0) * gross_up[:, None]

    holder = np.full(balances.shape, _HEAD, dtype=object)
    for code, role in SCF_WHOSE_EDUCATION_HOLDER.items():
        holder[whose == code] = role
    graduate = np.isin(program, list(_GRADUATE_PROGRAM_CODES))

    # A loan for a child's education is a co-resident child's own loan when a
    # listed child aged 18 or over could have been in school when it was
    # borrowed, unless only a parent could have borrowed it: an undergraduate
    # amount above a dependent student's own aggregate limit, borrowed while
    # the child could still have been dependent. Ages are the public file's
    # bands; the arrays are (families x 10 listed persons x 6 loans).
    adult_child, age_code = _scf_adult_children(full)
    youngest_age, oldest_age = _listing_age_range(age_code)
    years_ago = (_SCF_SURVEY_YEAR - year)[:, None, :]
    # NaN ages (no such person) compare false, and adult_child excludes them.
    fits = adult_child[:, :, None] & (
        oldest_age[:, :, None] - years_ago >= _EARLIEST_STUDENT_AGE
    )
    could_be_dependent = (
        fits & (youngest_age[:, :, None] - years_ago < _INDEPENDENT_STUDENT_AGE)
    ).any(axis=1)
    parents_only = (
        could_be_dependent
        & ~graduate
        & (borrowed > _DEPENDENT_UNDERGRADUATE_AGGREGATE_LIMIT)
    )
    child_education = holder == _CHILD_EDUCATION
    childs_own = child_education & fits.any(axis=1) & ~parents_only
    holder[child_education] = _PARENT_PLUS
    holder[childs_own] = _CHILD

    held = {
        role: federal & (holder == role)
        for role in (_HEAD, _SPOUSE, _CHILD, _PARENT_PLUS)
    }
    amount = {
        role: np.where(mask, federal_balance, 0.0).sum(axis=1)
        for role, mask in held.items()
    }
    total = amount[_HEAD] + amount[_SPOUSE] + amount[_CHILD] + amount[_PARENT_PLUS]

    def _share(numerator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            total,
            out=np.zeros(n, dtype=np.float64),
            where=total > 0,
        )

    new_loan = year >= _IBR_NEW_BORROWER_FIRST_YEAR

    def _new_borrower(holds: np.ndarray) -> np.ndarray:
        return (holds.any(axis=1) & np.all(new_loan | ~holds, axis=1)).astype(
            np.float64
        )

    weight = np.where(borrowed > 0, borrowed, balances)

    def _graduate_share(holds: np.ndarray) -> np.ndarray:
        denominator = np.where(holds, weight, 0.0).sum(axis=1)
        numerator = np.where(holds & graduate, weight, 0.0).sum(axis=1)
        return np.divide(
            numerator,
            denominator,
            out=np.zeros(n, dtype=np.float64),
            where=denominator > 0,
        )

    youngest_adult = np.where(adult_child, age_code, np.inf).min(axis=1)
    fitted_codes = np.where(fits, age_code[:, :, None], np.inf).min(axis=1)
    child_borrower = np.where(held[_CHILD], fitted_codes, np.inf).min(axis=1)

    result = pd.DataFrame(index=full.index)
    result[SCF_EDUCATION_DEBT_TOTAL_TARGET] = total
    result["head_federal_student_loan_balance"] = amount[_HEAD]
    result["spouse_federal_student_loan_balance"] = amount[_SPOUSE]
    result["child_federal_student_loan_balance"] = amount[_CHILD]
    result["federal_parent_plus_loan_balance"] = amount[_PARENT_PLUS]
    result["parent_plus_share"] = _share(amount[_PARENT_PLUS])
    result["child_own_share"] = _share(amount[_CHILD])
    result["head_own_share"] = _share(amount[_HEAD])
    result["spouse_own_share"] = _share(amount[_SPOUSE])
    result["head_graduate_share"] = _graduate_share(held[_HEAD])
    result["spouse_graduate_share"] = _graduate_share(held[_SPOUSE])
    result["child_graduate_share"] = _graduate_share(held[_CHILD])
    # The head's flag covers Parent PLUS loans too: 34 CFR 685.209(b)(13)(ii)
    # looks at any outstanding Direct or FFEL balance.
    result["head_is_ibr_new_borrower"] = _new_borrower(held[_HEAD] | held[_PARENT_PLUS])
    result["spouse_is_ibr_new_borrower"] = _new_borrower(held[_SPOUSE])
    result["child_is_ibr_new_borrower"] = _new_borrower(held[_CHILD])
    result[SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR] = adult_child.sum(
        axis=1
    ).astype(np.float64)
    result["youngest_adult_child_age_code"] = np.where(
        np.isfinite(youngest_adult), youngest_adult, 0.0
    )
    result["child_borrower_age_code"] = np.where(
        np.isfinite(child_borrower), child_borrower, 0.0
    )
    result["education_debt"] = loans_1_6 + remaining
    result["federal_status_unrecorded_balance"] = np.where(
        has_balance & (federal_code == 0), balances, 0.0
    ).sum(axis=1)
    result["remaining_loans_without_detail"] = np.where(positive_detail, 0.0, remaining)
    return result


def load_scf_2025_education_debt_donor(
    summary_extract_path: str | Path,
    full_extract_path: str | Path,
    *,
    expected_rows: int | None = SCF_2025_FULL_EXTRACT.rows,
) -> pd.DataFrame:
    """Join the 2025 summary and full files and derive the donor table.

    The two files must join one-to-one on ``(y1, yy1)`` across all five
    implicates, each must have ``expected_rows`` rows (``None`` skips that
    check, for test fixtures), and the per-loan balances must reproduce the
    summary extract's ``edn_inst``. Returns one row per (family, implicate)
    with a positive weight: the predictors, the targets of
    :func:`derive_scf_education_debt_targets` (which include the second QRF's
    ``adult_children_in_household``), the ``scf_weight`` donor weight and the
    fit-table covariates (``racecl4``, ``income``, ``educ``, ``spouse_age``,
    ``yy1``, ``implicate``).
    """

    summary_columns = (
        *dict.fromkeys(
            (
                *_SUMMARY_EXTRA_COLUMNS,
                "wgt",
                "age",
                "hhsex",
                "racecl5",
                "married",
                "kids",
                "wageinc",
                "intdivinc",
                "ssretinc",
            )
        ),
    )
    full_columns = (
        "y1",
        "yy1",
        _SCF_FULL_EXTRACT_SPOUSE_AGE_COLUMN,
        *SCF_EDUCATION_LOAN_BALANCE_COLUMNS,
        SCF_EDUCATION_LOAN_REMAINING_BALANCE_COLUMN,
        *SCF_EDUCATION_LOAN_FEDERAL_COLUMNS,
        *SCF_EDUCATION_LOAN_WHOSE_COLUMNS,
        *SCF_EDUCATION_LOAN_YEAR_COLUMNS,
        *SCF_EDUCATION_LOAN_PROGRAM_COLUMNS,
        *SCF_EDUCATION_LOAN_AMOUNT_BORROWED_COLUMNS,
        SCF_REVERSED_COLUMN,
        *SCF_HOUSEHOLD_LISTING_RELATIONSHIP_COLUMNS,
        *SCF_HOUSEHOLD_LISTING_AGE_COLUMNS,
    )
    summary = _read_stata_columns(
        summary_extract_path, summary_columns, label="2025 summary extract"
    )
    full = _read_stata_columns(full_extract_path, full_columns, label="2025 full file")
    keys = ["y1", "yy1"]
    for label, table in (("summary", summary), ("full", full)):
        if expected_rows is not None and len(table) != int(expected_rows):
            raise ValueError(
                f"SCF 2025 {label} file has {len(table)} rows; expected "
                f"{int(expected_rows)}."
            )
        if table.duplicated(keys).any():
            raise ValueError(
                f"SCF 2025 {label} file does not have one-to-one (y1, yy1) keys."
            )
    keyed_summary = summary.reset_index(drop=True)
    keyed_summary["_summary_order"] = np.arange(len(keyed_summary))
    joined = keyed_summary.merge(
        full,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    unmatched = int((joined["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(
            "SCF 2025 summary/full files have unmatched (y1, yy1) implicates; "
            f"a one-to-one join is required ({unmatched} unmatched row(s))."
        )
    joined = joined.sort_values("_summary_order").reset_index(drop=True)
    implicate = pd.to_numeric(joined["y1"], errors="coerce").astype("int64") % 10
    if sorted(implicate.unique().tolist()) != [1, 2, 3, 4, 5]:
        raise ValueError("SCF 2025 rows must carry implicates 1-5 (y1 mod 10).")

    targets = derive_scf_education_debt_targets(joined)
    edn_inst = _numeric(joined, "edn_inst")
    mismatch = np.abs(edn_inst - targets["education_debt"].to_numpy()) > 1.0
    if mismatch.any():
        raise ValueError(
            "SCF 2025 education-loan balances do not reproduce edn_inst on "
            f"{int(mismatch.sum())} row(s); the per-loan definition has drifted."
        )

    predictors = _scf_summary_predictor_table(joined)
    donor = predictors.reset_index(drop=True)
    donor["household_income"] = _numeric(joined, "income") - _numeric(joined, "kginc")
    for column in targets.columns:
        donor[column] = targets[column].to_numpy(dtype=np.float64)
    donor["racecl4"] = _numeric(joined, "racecl4")
    donor["income"] = _numeric(joined, "income")
    donor["educ"] = _numeric(joined, "educ")
    donor["spouse_age"] = _replace_sentinels(
        joined[_SCF_FULL_EXTRACT_SPOUSE_AGE_COLUMN]
    ).to_numpy(dtype=np.float64)
    donor["yy1"] = pd.to_numeric(joined["yy1"], errors="coerce").to_numpy()
    donor["implicate"] = implicate.to_numpy()
    return donor.loc[donor[_DONOR_WEIGHT_COLUMN] > 0].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Recipients                                                                    #
# --------------------------------------------------------------------------- #
_REFERENCE_PERSON_RULES = frozenset({"householder_couple", "oldest_adult"})
_PARTNERED_RULES = frozenset({"spouse_or_partner", "married_spouse"})
_CHILDREN_RULES = frozenset({"of_reference_or_partner", "of_reference"})
_INCOME_UNITS = frozenset({"household", "spm_unit", "reference_couple_and_children"})


@dataclass(frozen=True)
class RecipientDefinition:
    """How the CPS side builds the SCF predictors for one household.

    Each field names the CPS construction of one SCF concept:

    - ``reference_person``: ``householder_couple`` takes the CPS householder
      (lowest line number) and, when a spouse or cohabiting partner is present,
      the man of a mixed-sex couple or else the older partner, which is how
      the SCF orders its economically dominant couple. ``oldest_adult`` is the
      auto-loan rule (the only adult, the man of a two-adult married couple,
      else the oldest adult).
    - ``partnered``: ``spouse_or_partner`` is SCF ``married`` (married or living
      with a partner, X8023 in 1-2): a spouse (``A_SPOUSE``) or cohabiting
      partner (``PECOHAB``) is present. ``married_spouse`` is ``A_MARITL`` 1-2.
    - ``children``: ``of_reference_or_partner`` is SCF ``kids`` (children,
      step-children and children-in-law of the reference person or partner,
      any age, anywhere in the household); ``of_reference`` counts only the
      reference person's own children through ``PEPAR1``/``PEPAR2``.
    - ``income_unit``: the people whose incomes are summed. ``household`` sums
      every member; ``spm_unit`` the reference person's SPM unit;
      ``reference_couple_and_children`` the reference person, spouse or
      partner and their children.
    """

    reference_person: str = "householder_couple"
    partnered: str = "spouse_or_partner"
    children: str = "of_reference_or_partner"
    income_unit: str = "reference_couple_and_children"

    def __post_init__(self) -> None:
        for value, allowed, name in (
            (self.reference_person, _REFERENCE_PERSON_RULES, "reference_person"),
            (self.partnered, _PARTNERED_RULES, "partnered"),
            (self.children, _CHILDREN_RULES, "children"),
            (self.income_unit, _INCOME_UNITS, "income_unit"),
        ):
            if value not in allowed:
                raise ValueError(
                    f"RecipientDefinition.{name} must be one of {sorted(allowed)}."
                )


#: The definition the stage uses (see the module docstring for the evidence).
SCF_EDUCATION_DEBT_RECIPIENT_DEFINITION = RecipientDefinition()
#: The definition of the first #1073 receipt, kept for its before/after table.
SCF_EDUCATION_DEBT_FIRST_RECEIPT_DEFINITION = RecipientDefinition(
    reference_person="oldest_adult",
    partnered="married_spouse",
    children="of_reference",
    income_unit="household",
)


@dataclass(frozen=True)
class _Recipients:
    """Household-ordered predictors and each household's recipient rows."""

    features: pd.DataFrame
    head_position: np.ndarray
    spouse_position: np.ndarray  # -1 where no spouse or partner is present
    child_position: np.ndarray  # the youngest adult child; -1 where none


def _line_lookup(person: pd.DataFrame) -> dict[tuple[object, int], int]:
    household = person[_HOUSEHOLD_ID_COLUMN].to_numpy()
    line = pd.to_numeric(person["A_LINENO"], errors="coerce").fillna(-1).to_numpy()
    return {
        (household_id, int(line_number)): position
        for position, (household_id, line_number) in enumerate(
            zip(household, line, strict=True)
        )
    }


def _resolve_partner_positions(
    person: pd.DataFrame, head_positions: np.ndarray
) -> np.ndarray:
    """Positional row of each head's spouse (A_SPOUSE) or partner (PECOHAB)."""

    household = person[_HOUSEHOLD_ID_COLUMN].to_numpy()
    lookup = _line_lookup(person)
    pointers = [pd.to_numeric(person["A_SPOUSE"], errors="coerce").fillna(0).to_numpy()]
    if _OPTIONAL_PARTNER_COLUMN in person.columns:
        pointers.append(
            pd.to_numeric(person[_OPTIONAL_PARTNER_COLUMN], errors="coerce")
            .fillna(0)
            .to_numpy()
        )
    result = np.full(len(head_positions), -1, dtype=np.int64)
    for index, head in enumerate(head_positions):
        for pointer in pointers:
            target = int(pointer[head])
            if target <= 0:
                continue
            partner = lookup.get((household[head], target))
            if partner is not None and partner != head:
                result[index] = partner
                break
    return result


def _householder_couple_positions(
    person: pd.DataFrame, order: np.ndarray
) -> np.ndarray:
    """The SCF-ordered reference person of each household, aligned to ``order``."""

    household = person[_HOUSEHOLD_ID_COLUMN].to_numpy()
    line = pd.to_numeric(person["A_LINENO"], errors="coerce").fillna(9_999).to_numpy()
    position = np.arange(len(person))
    ranked = np.lexsort((position, line, household))
    first = np.ones(len(person), dtype=bool)
    first[1:] = household[ranked][1:] != household[ranked][:-1]
    householders = ranked[first]
    by_household = pd.Series(householders, index=household[householders]).reindex(order)
    if by_household.isna().any():
        missing_ids = order[by_household.isna().to_numpy()][:5].tolist()
        raise ValueError(
            "SCF education-debt receiver household alignment failed: "
            f"no householder for household id(s) {missing_ids}."
        )
    householder = by_household.to_numpy(dtype=np.int64)
    partner = _resolve_partner_positions(person, householder)
    female = person["is_female"].astype(bool).to_numpy()
    age = pd.to_numeric(person["age"], errors="coerce").fillna(0.0).to_numpy()
    has_partner = partner >= 0
    safe_partner = np.where(has_partner, partner, householder)
    mixed = female[householder] != female[safe_partner]
    partner_first = np.where(
        mixed,
        female[householder],  # the man of a mixed-sex couple
        age[safe_partner] > age[householder],  # else the older partner
    )
    return np.where(has_partner & partner_first, safe_partner, householder)


def _children_counts(
    person: pd.DataFrame,
    head_position: np.ndarray,
    spouse_position: np.ndarray,
    *,
    include_partner: bool,
) -> np.ndarray:
    """Household members whose CPS parent pointer names the head (or partner)."""

    household = person[_HOUSEHOLD_ID_COLUMN].to_numpy()
    line = pd.to_numeric(person["A_LINENO"], errors="coerce").fillna(-1).to_numpy()
    parent_of: dict[tuple[object, int], int] = {
        (household[head], int(line[head])): index
        for index, head in enumerate(head_position)
    }
    if include_partner:
        for index, spouse in enumerate(spouse_position):
            if spouse >= 0:
                parent_of.setdefault((household[spouse], int(line[spouse])), index)
    counts = np.zeros(len(head_position), dtype=np.float64)
    pointers = [
        pd.to_numeric(person[column], errors="coerce").fillna(-1).to_numpy()
        for column in ("PEPAR1", "PEPAR2")
    ]
    for row in range(len(person)):
        owners = {
            parent_of.get((household[row], int(pointer[row])))
            for pointer in pointers
            if pointer[row] > 0
        }
        owners.discard(None)
        # A child of both partners counts once.
        if owners:
            counts[next(iter(owners))] += 1.0
    return counts


def _adult_children(
    person: pd.DataFrame,
    head_position: np.ndarray,
    spouse_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Children of the head or partner aged 18 or over, and the youngest one.

    A child is a member whose CPS parent pointer (``PEPAR1``/``PEPAR2``) names
    the head or the spouse or partner, as SCF listing code 4 counts children of
    the respondent or spouse/partner. Returns the count per household and the
    positional row of the youngest (lowest line number on a tie), or -1.
    """

    household = person[_HOUSEHOLD_ID_COLUMN].to_numpy()
    line = pd.to_numeric(person["A_LINENO"], errors="coerce").fillna(-1).to_numpy()
    age = pd.to_numeric(person["age"], errors="coerce").fillna(0.0).to_numpy()
    parent_of: dict[tuple[object, int], int] = {
        (household[head], int(line[head])): index
        for index, head in enumerate(head_position)
    }
    for index, spouse in enumerate(spouse_position):
        if spouse >= 0:
            parent_of.setdefault((household[spouse], int(line[spouse])), index)
    pointers = [
        pd.to_numeric(person[column], errors="coerce").fillna(-1).to_numpy()
        for column in ("PEPAR1", "PEPAR2")
    ]
    counts = np.zeros(len(head_position), dtype=np.float64)
    youngest = np.full(len(head_position), -1, dtype=np.int64)
    for row in np.flatnonzero(age >= _ADULT_AGE):
        owners = {
            parent_of.get((household[row], int(pointer[row])))
            for pointer in pointers
            if pointer[row] > 0
        }
        owners.discard(None)
        if not owners:
            continue
        owner = next(iter(owners))
        counts[owner] += 1.0
        current = youngest[owner]
        if current < 0 or (age[row], line[row]) < (age[current], line[current]):
            youngest[owner] = row
    return counts, youngest


def _income_unit_owner(
    person: pd.DataFrame,
    head_position: np.ndarray,
    spouse_position: np.ndarray,
    unit: str,
) -> np.ndarray:
    """For each person, the household row whose income unit includes them, or -1."""

    household = person[_HOUSEHOLD_ID_COLUMN].to_numpy()
    if unit == "household":
        index = pd.Series(np.arange(len(head_position)), index=household[head_position])
        return index.reindex(household).to_numpy(dtype=np.int64)
    owner = np.full(len(person), -1, dtype=np.int64)
    if unit == "spm_unit":
        if "person_spm_unit_id" not in person.columns:
            raise ValueError("The spm_unit income unit requires person_spm_unit_id.")
        spm = person["person_spm_unit_id"].to_numpy()
        index = pd.Series(np.arange(len(head_position)), index=spm[head_position])
        index = index[~index.index.duplicated()]
        mapped = index.reindex(spm).to_numpy(dtype=np.float64)
        owner = np.where(np.isnan(mapped), -1, mapped).astype(np.int64)
    owner[head_position] = np.arange(len(head_position))
    has_spouse = spouse_position >= 0
    owner[spouse_position[has_spouse]] = np.flatnonzero(has_spouse)
    if unit == "reference_couple_and_children":
        line = pd.to_numeric(person["A_LINENO"], errors="coerce").fillna(-1).to_numpy()
        parent_of = {
            (household[row], int(line[row])): owner[row]
            for row in np.flatnonzero(owner >= 0)
        }
        for column in ("PEPAR1", "PEPAR2"):
            pointer = (
                pd.to_numeric(person[column], errors="coerce").fillna(-1).to_numpy()
            )
            for row in np.flatnonzero((owner < 0) & (pointer > 0)):
                found = parent_of.get((household[row], int(pointer[row])))
                if found is not None:
                    owner[row] = found
    return owner


def _recipients(
    frame: Frame,
    definition: RecipientDefinition = SCF_EDUCATION_DEBT_RECIPIENT_DEFINITION,
) -> _Recipients:
    person = frame.table("person")
    household = frame.table("household")
    missing = sorted(set(_REQUIRED_RECIPIENT_PERSON_COLUMNS) - set(person.columns))
    if missing:
        raise ValueError(
            "US SCF education-debt imputation requires recipient person "
            f"column(s): {missing}."
        )
    if "household_id" not in household.columns:
        raise ValueError("US SCF education-debt imputation requires household_id.")

    order = household["household_id"].to_numpy()
    household_ids = person[_HOUSEHOLD_ID_COLUMN].to_numpy()
    if definition.reference_person == "householder_couple":
        head_position = _householder_couple_positions(person, order)
    else:
        mask = _scf_reference_person_mask(person)
        head_rows = np.flatnonzero(mask)
        head_by_household = pd.Series(head_rows, index=household_ids[head_rows])
        if head_by_household.index.duplicated().any():
            raise ValueError(
                "SCF education-debt receiver selected two reference persons."
            )
        aligned = head_by_household.reindex(order)
        if aligned.isna().any():
            missing_ids = order[aligned.isna().to_numpy()][:5].tolist()
            raise ValueError(
                "SCF education-debt receiver household alignment failed: "
                f"no reference person for household id(s) {missing_ids}."
            )
        head_position = aligned.to_numpy(dtype=np.int64)
    spouse_position = _resolve_partner_positions(person, head_position)

    features = pd.DataFrame(index=np.arange(len(head_position)))
    features["age"] = (
        pd.to_numeric(person["age"], errors="coerce").fillna(0.0).to_numpy()
    )[head_position]
    features["is_female"] = (
        person["is_female"].astype(bool).astype(float).to_numpy()[head_position]
    )
    features["cps_race"] = _recipient_cps_race(person)[head_position]
    if definition.partnered == "spouse_or_partner":
        features["is_married"] = (spouse_position >= 0).astype(np.float64)
    else:
        features["is_married"] = (
            pd.to_numeric(person["A_MARITL"], errors="coerce")
            .isin([1, 2])
            .astype(float)
            .to_numpy()[head_position]
        )
    if definition.children == "of_reference_or_partner":
        features["own_children_in_household"] = _children_counts(
            person, head_position, spouse_position, include_partner=True
        )
    else:
        features["own_children_in_household"] = _own_children_in_household(person)[
            head_position
        ]

    owner = _income_unit_owner(
        person, head_position, spouse_position, definition.income_unit
    )
    member = owner >= 0
    for column, values in (
        (
            "employment_income",
            np.maximum(
                pd.to_numeric(person["employment_income_before_lsr"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float64),
                0.0,
            ),
        ),
        (
            "interest_dividend_income",
            _sum_present(person, _RECIPIENT_INTEREST_DIVIDEND_COLUMNS),
        ),
        (
            "social_security_pension_income",
            _sum_present(person, _RECIPIENT_SS_PENSION_COLUMNS),
        ),
        (
            "household_income",
            _sum_present(person, _RECIPIENT_HOUSEHOLD_INCOME_COLUMNS),
        ),
    ):
        sums = np.zeros(len(head_position), dtype=np.float64)
        np.add.at(sums, owner[member], values[member])
        features[column] = sums
    adult_children, child_position = _adult_children(
        person, head_position, spouse_position
    )
    features[SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR] = adult_children
    features.index = household.index
    return _Recipients(
        features=features.loc[
            :,
            [
                *SCF_EDUCATION_DEBT_CANDIDATE_PREDICTORS,
                SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR,
            ],
        ],
        head_position=head_position,
        spouse_position=spouse_position,
        child_position=child_position,
    )


# --------------------------------------------------------------------------- #
# Imputation                                                                    #
# --------------------------------------------------------------------------- #
def _stage_seed(seed: int, step: int) -> int:
    """A model seed for one of the stage's two QRFs, independent of other stages."""

    sequence = np.random.SeedSequence([int(seed), 1073, int(step)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _fit_frame(donor: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    missing = [column for column in columns if column not in donor.columns]
    if missing:
        raise ValueError(f"SCF education-debt donor missing column(s): {missing}.")
    fit = donor.loc[:, list(columns)].copy()
    for column in columns:
        fit[column] = pd.to_numeric(fit[column], errors="coerce")
        if not np.isfinite(fit[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(
                f"SCF education-debt donor column {column!r} contains nonfinite values."
            )
    return fit


def _split_whole_dollars(total: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """Split whole-dollar totals by row shares; a zero share stays zero.

    Each part is the floor of its share of the total, and the few dollars of
    remainder go to the row's largest share, so the parts sum to the total.
    """

    parts = np.floor(total[:, None] * shares)
    remainder = total - parts.sum(axis=1)
    largest = np.argmax(shares, axis=1)
    parts[np.arange(len(total)), largest] += remainder
    return parts


def draw_scf_education_debt_households(
    features: pd.DataFrame,
    donor: pd.DataFrame,
    *,
    seed: int,
    predictors: Sequence[str] = SCF_EDUCATION_DEBT_PREDICTORS,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> pd.DataFrame:
    """Draw household totals and their composition for recipient rows.

    ``features`` carries the predictor columns and
    ``adult_children_in_household``, one row per recipient household (or
    held-out SCF family). Returns, on the same index, the drawn total and the
    normalized composition: the head's, spouse's and adult child's own
    balances, the head's Parent PLUS balance, the three graduate shares and the
    three new-borrower flags. Used by the build and by the SCF hold-out fit
    check.

    The four holder shares are drawn in turn (Parent PLUS first, so a family
    unlike any Parent PLUS donor draws none) and normalized to sum to one; a
    borrower whose every share is drawn zero holds its own loan. A child's
    share drawn for a household without an adult child is the head's Parent
    PLUS loan: as on the SCF side, a loan for a child who is not in the
    household is the parent's.
    """

    from microcosm.fit import QRF

    predictors = list(predictors)
    total_fit = _fit_frame(
        donor, (*predictors, SCF_EDUCATION_DEBT_TOTAL_TARGET, _DONOR_WEIGHT_COLUMN)
    )
    total_model = QRF(n_estimators=int(n_estimators), seed=_stage_seed(seed, 1)).fit(
        total_fit,
        predictors=predictors,
        targets=[SCF_EDUCATION_DEBT_TOTAL_TARGET],
        weights=_DONOR_WEIGHT_COLUMN,
    )
    drawn_total = np.maximum(
        np.asarray(
            total_model.predict(features.loc[:, predictors], release_models=True)[
                SCF_EDUCATION_DEBT_TOTAL_TARGET
            ],
            dtype=np.float64,
        ),
        0.0,
    )
    drawn_total = np.round(drawn_total)

    result = pd.DataFrame(index=features.index)
    result[SCF_EDUCATION_DEBT_TOTAL_TARGET] = drawn_total
    for column in SCF_EDUCATION_DEBT_COMPOSITION_TARGETS:
        result[column] = 0.0
    borrowers = drawn_total > 0
    if borrowers.any():
        composition_predictors = list(
            scf_education_debt_composition_predictors(predictors)
        )
        borrower_donor = donor.loc[
            pd.to_numeric(donor[SCF_EDUCATION_DEBT_TOTAL_TARGET], errors="coerce") > 0
        ]
        composition_fit = _fit_frame(
            borrower_donor,
            (
                *composition_predictors,
                *SCF_EDUCATION_DEBT_COMPOSITION_TARGETS,
                _DONOR_WEIGHT_COLUMN,
            ),
        )
        composition_model = QRF(
            n_estimators=int(n_estimators), seed=_stage_seed(seed, 2)
        ).fit(
            composition_fit,
            predictors=composition_predictors,
            targets=list(SCF_EDUCATION_DEBT_COMPOSITION_TARGETS),
            weights=_DONOR_WEIGHT_COLUMN,
        )
        recipients = features.loc[borrowers, composition_predictors[:-1]].copy()
        recipients[SCF_EDUCATION_DEBT_TOTAL_TARGET] = drawn_total[borrowers]
        drawn = composition_model.predict(recipients, release_models=True)
        for column in SCF_EDUCATION_DEBT_COMPOSITION_TARGETS:
            result.loc[borrowers, column] = np.asarray(drawn[column], dtype=np.float64)

    shares = np.column_stack(
        [
            result[column].clip(0.0, 1.0).to_numpy(dtype=np.float64)
            for column in SCF_EDUCATION_DEBT_HOLDER_SHARE_TARGETS
        ]
    )
    plus, child, head, spouse = range(len(SCF_EDUCATION_DEBT_HOLDER_SHARE_TARGETS))
    shares[~borrowers] = 0.0
    shares[borrowers & (shares.sum(axis=1) <= 0), head] = 1.0
    no_adult_child = (
        pd.to_numeric(
            features[SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR], errors="coerce"
        )
        .fillna(0.0)
        .to_numpy()
        <= 0
    )
    child_new = result["child_is_ibr_new_borrower"].to_numpy() >= 0.5
    head_new = result["head_is_ibr_new_borrower"].to_numpy() >= 0.5
    moved = no_adult_child & (shares[:, child] > 0)
    # The head then holds Parent PLUS for that child too: new only when every
    # loan the head holds is.
    head_held = (shares[:, head] + shares[:, plus]) > 0
    head_new = np.where(
        moved, np.where(head_held, head_new, True) & child_new, head_new
    )
    shares[moved, plus] += shares[moved, child]
    shares[moved, child] = 0.0
    sums = shares.sum(axis=1)
    shares = np.divide(
        shares, sums[:, None], out=np.zeros_like(shares), where=sums[:, None] > 0
    )
    parts = _split_whole_dollars(drawn_total, shares)
    result["head_federal_student_loan_balance"] = parts[:, head]
    result["spouse_federal_student_loan_balance"] = parts[:, spouse]
    result["child_federal_student_loan_balance"] = parts[:, child]
    result["federal_parent_plus_loan_balance"] = parts[:, plus]
    for prefix, column in (("head", head), ("spouse", spouse), ("child", child)):
        graduate = result[f"{prefix}_graduate_share"].clip(0.0, 1.0).to_numpy()
        result[f"{prefix}_graduate_share"] = np.where(
            parts[:, column] > 0, graduate, 0.0
        )
    result["head_is_ibr_new_borrower"] = (
        ((parts[:, head] + parts[:, plus]) > 0) & head_new
    ).astype(np.float64)
    result["spouse_is_ibr_new_borrower"] = (
        (parts[:, spouse] > 0)
        & (result["spouse_is_ibr_new_borrower"].to_numpy() >= 0.5)
    ).astype(np.float64)
    result["child_is_ibr_new_borrower"] = ((parts[:, child] > 0) & child_new).astype(
        np.float64
    )
    for index, column in enumerate(SCF_EDUCATION_DEBT_HOLDER_SHARE_TARGETS):
        result[column] = shares[:, index]
    return result


def impute_us_scf_education_debt(
    frame: Frame,
    donor: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
    definition: RecipientDefinition = SCF_EDUCATION_DEBT_RECIPIENT_DEFINITION,
    predictors: Sequence[str] = SCF_EDUCATION_DEBT_PREDICTORS,
) -> pd.DataFrame:
    """Impute the four person columns onto a US frame.

    Returns a DataFrame indexed like the person table with
    :data:`US_SCF_EDUCATION_DEBT_OUTPUT_COLUMNS`; only reference persons,
    their spouses or partners, and the youngest child of either aged 18 or
    over can be nonzero. ``definition`` selects the CPS construction of the
    SCF predictors and ``predictors`` the subset the QRFs condition on; the
    receipts pass the first receipt's choices for their before/after table.
    """

    recipients = _recipients(frame, definition)
    drawn = draw_scf_education_debt_households(
        recipients.features,
        donor,
        seed=seed,
        predictors=predictors,
        n_estimators=n_estimators,
    )
    person = frame.table("person")
    n_person = len(person)
    own = np.zeros(n_person, dtype=np.float64)
    parent_plus = np.zeros(n_person, dtype=np.float64)
    graduate = np.zeros(n_person, dtype=np.float64)
    new_borrower = np.zeros(n_person, dtype=bool)

    head = recipients.head_position
    spouse = recipients.spouse_position
    child = recipients.child_position
    has_spouse = spouse >= 0
    has_child = child >= 0
    head_own = drawn["head_federal_student_loan_balance"].to_numpy()
    spouse_own = drawn["spouse_federal_student_loan_balance"].to_numpy()
    child_own = drawn["child_federal_student_loan_balance"].to_numpy()
    head_plus = drawn["federal_parent_plus_loan_balance"].to_numpy()
    head_graduate = drawn["head_graduate_share"].to_numpy()
    spouse_graduate = drawn["spouse_graduate_share"].to_numpy()
    child_graduate = drawn["child_graduate_share"].to_numpy()
    head_new = drawn["head_is_ibr_new_borrower"].to_numpy() >= 0.5
    spouse_new = drawn["spouse_is_ibr_new_borrower"].to_numpy() >= 0.5
    child_new = drawn["child_is_ibr_new_borrower"].to_numpy() >= 0.5
    # The draw gives a child's balance only to households with an adult child,
    # which is exactly where the youngest one was found.
    if ((child_own > 0) & ~has_child).any():
        raise ValueError(
            "SCF education-debt draw gave a child's balance to a household "
            "without an adult child."
        )

    # Without a spouse or partner present, the reference person holds the
    # spouse's draw too: a balance-weighted graduate share, and new-borrower
    # status only when both holders' loans are new.
    moved = ~has_spouse & (spouse_own > 0)
    combined_own = np.where(moved, head_own + spouse_own, head_own)
    combined_graduate = np.where(
        moved,
        np.divide(
            head_own * head_graduate + spouse_own * spouse_graduate,
            combined_own,
            out=np.zeros_like(combined_own),
            where=combined_own > 0,
        ),
        head_graduate,
    )
    head_holds_any = (head_own + head_plus) > 0
    combined_new = np.where(
        moved,
        np.where(head_holds_any, head_new, True) & spouse_new,
        head_new,
    )
    own[head] = combined_own
    parent_plus[head] = head_plus
    graduate[head] = combined_graduate
    new_borrower[head] = combined_new
    spouse_rows = spouse[has_spouse]
    own[spouse_rows] = spouse_own[has_spouse]
    graduate[spouse_rows] = spouse_graduate[has_spouse]
    new_borrower[spouse_rows] = spouse_new[has_spouse]
    child_rows = child[has_child]
    own[child_rows] = child_own[has_child]
    graduate[child_rows] = child_graduate[has_child]
    new_borrower[child_rows] = child_new[has_child]

    holds_any = (own + parent_plus) > 0
    result = pd.DataFrame(index=person.index)
    result["federal_student_loan_balance"] = own
    result["federal_parent_plus_loan_balance"] = parent_plus
    result["is_income_based_repayment_new_borrower"] = new_borrower & holds_any
    result["federal_student_loan_graduate_share"] = np.where(own > 0, graduate, 0.0)
    return result


def _surface_has_signal(person: pd.DataFrame) -> bool:
    if not all(column in person for column in US_SCF_EDUCATION_DEBT_OUTPUT_COLUMNS):
        return False
    for column in US_SCF_EDUCATION_DEBT_BALANCE_COLUMNS:
        values = pd.to_numeric(person[column], errors="coerce")
        if values.isna().any() or values.nunique() < 2 or not (values > 0).any():
            return False
    return True


def with_us_scf_education_debt_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    scf_education_debt_donor: pd.DataFrame,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
    definition: RecipientDefinition = SCF_EDUCATION_DEBT_RECIPIENT_DEFINITION,
) -> Frame:
    """Write the four education-debt person inputs; idempotent on a signal."""

    del time_period  # The draw depends on the seed only; kept for the stage API.
    if frame.schema != US_SCHEMA:
        raise ValueError("US SCF education-debt inputs require the US schema.")
    us_scf_education_debt_stage_spec()
    person = frame.table("person")
    if _surface_has_signal(person):
        return frame
    imputed = impute_us_scf_education_debt(
        frame,
        scf_education_debt_donor,
        seed=int(seed),
        n_estimators=int(n_estimators),
        definition=definition,
    )
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column in US_SCF_EDUCATION_DEBT_OUTPUT_COLUMNS:
        tables["person"][column] = imputed[column].to_numpy()
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


# --------------------------------------------------------------------------- #
# Stage declaration, diagnostics and gate                                       #
# --------------------------------------------------------------------------- #
def us_scf_education_debt_stage_spec() -> SourceStageSpec:
    """Load the ``scf_wealth`` declaration and check its 2025 education-debt part."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    spec = manifest.stage_map()[_SCF_STAGE_NAME]
    missing = sorted(set(US_SCF_EDUCATION_DEBT_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{_SCF_STAGE_NAME!r} manifest stage does not declare education-debt "
            f"output(s) {missing}."
        )
    for public_file in (SCF_2025_SUMMARY_EXTRACT, SCF_2025_FULL_EXTRACT):
        declared = [
            artifact
            for artifact in spec.artifacts
            if artifact.get("locator") == public_file.url
        ]
        if len(declared) != 1:
            raise ValueError(
                f"{_SCF_STAGE_NAME!r} manifest must declare {public_file.url} once."
            )
        artifact = declared[0]
        expected = {
            "vintage": SCF_EDUCATION_DEBT_VINTAGE,
            "member": public_file.member,
            "sha256": public_file.zip_sha256,
            "member_sha256": public_file.member_sha256,
        }
        drift = {
            key: (artifact.get(key), value)
            for key, value in expected.items()
            if artifact.get(key) != value
        }
        if drift:
            raise ValueError(
                f"{_SCF_STAGE_NAME!r} SCF {SCF_EDUCATION_DEBT_VINTAGE} artifact pin "
                f"drift for {public_file.member}: {drift}."
            )
    expected_fits = {
        "scf_2025_education_debt": (
            SCF_EDUCATION_DEBT_PREDICTORS,
            (SCF_EDUCATION_DEBT_TOTAL_TARGET,),
        ),
        "scf_2025_education_debt_borrowers": (
            scf_education_debt_composition_predictors(),
            SCF_EDUCATION_DEBT_COMPOSITION_TARGETS,
        ),
    }
    for donor, (predictors, targets) in expected_fits.items():
        fits = [
            operation
            for operation in spec.operations
            if operation.kind == "fit_weighted_qrf"
            and operation.parameters.get("donor") == donor
        ]
        if (
            len(fits) != 1
            or tuple(fits[0].parameters.get("predictors", ())) != predictors
            or tuple(fits[0].parameters.get("targets", ())) != targets
        ):
            raise ValueError(
                f"{_SCF_STAGE_NAME!r} education-debt QRF {donor!r} predictors or "
                "targets have drifted."
            )
    return spec


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """First value whose cumulative weight reaches ``q`` of the total."""

    if len(values) == 0 or float(weights.sum()) <= 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, q * cumulative[-1], side="left"))
    return float(values[order][min(index, len(values) - 1)])


def _band_shares(
    values: np.ndarray,
    weights: np.ndarray,
    bands: Sequence[tuple[str, float, float]],
    *,
    inclusive_upper: bool,
) -> dict[str, float]:
    total = float(weights.sum())
    shares: dict[str, float] = {}
    for label, low, high in bands:
        if inclusive_upper:
            inside = (values >= low) & (values <= high)
        else:
            inside = (values >= low) & (values < high)
        shares[label] = float(weights[inside].sum()) / total if total > 0 else 0.0
    return shares


def _income_quintiles(income: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted income quintile (1-5) of each household."""

    order = np.argsort(income, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    total = cumulative[-1] if len(cumulative) else 0.0
    quintile = np.empty(len(income), dtype=np.int64)
    if total <= 0:
        quintile[:] = 1
        return quintile
    # Midpoint rule: a household's rank is the cumulative weight at its centre.
    midpoint = (cumulative - weights[order] / 2.0) / total
    quintile[order] = np.minimum((midpoint * 5).astype(np.int64) + 1, 5)
    return quintile


def education_debt_fit_table(
    *,
    household_debt: np.ndarray,
    household_weight: np.ndarray,
    head_age: np.ndarray,
    race_group: np.ndarray,
    income_quintile: np.ndarray,
    borrower_balance: np.ndarray,
    borrower_age: np.ndarray,
    borrower_weight: np.ndarray,
    parent_plus_total: float,
    graduate_numerator: float,
    graduate_denominator: float,
    new_borrower_weight: float,
) -> dict[str, object]:
    """The microcosm#1073 fit table from household and borrower arrays.

    Household statistics follow the Fed's SCF tables (share of households with
    federal education debt, mean and median among them, by the age of the
    reference person, race and income quintile). Borrower statistics follow
    Federal Student Aid's portfolio tables: people holding a federal balance
    (own plus Parent PLUS), by balance band and borrower age band.
    """

    household_debt = np.asarray(household_debt, dtype=np.float64)
    household_weight = np.asarray(household_weight, dtype=np.float64)
    borrower = household_debt > 0
    total_weight = float(household_weight.sum())

    def _group(mask: np.ndarray) -> dict[str, float]:
        weight = household_weight[mask]
        debt = household_debt[mask]
        holders = debt > 0
        group_weight = float(weight.sum())
        holder_weight = float(weight[holders].sum())
        return {
            "household_weight": group_weight,
            "share_with_federal_education_debt": (
                holder_weight / group_weight if group_weight > 0 else float("nan")
            ),
            "mean_among_borrowers": (
                float(np.average(debt[holders], weights=weight[holders]))
                if holder_weight > 0
                else float("nan")
            ),
            "median_among_borrowers": _weighted_quantile(
                debt[holders], weight[holders], 0.5
            ),
        }

    head_age = np.asarray(head_age, dtype=np.float64)
    race_group = np.asarray(race_group)
    income_quintile = np.asarray(income_quintile)
    borrower_balance = np.asarray(borrower_balance, dtype=np.float64)
    borrower_age = np.asarray(borrower_age, dtype=np.float64)
    borrower_weight = np.asarray(borrower_weight, dtype=np.float64)
    federal_total = float(np.dot(household_debt, household_weight))
    return {
        "households": {
            **_group(np.ones(len(household_debt), dtype=bool)),
            "total_household_weight": total_weight,
            "federal_education_debt_total": federal_total,
            "parent_plus_share_of_total": (
                parent_plus_total / federal_total if federal_total > 0 else float("nan")
            ),
        },
        "household_balance_bands": _band_shares(
            household_debt[borrower],
            household_weight[borrower],
            SCF_EDUCATION_DEBT_BALANCE_BANDS,
            inclusive_upper=False,
        ),
        "by_head_age_band": {
            label: _group((head_age >= low) & (head_age <= high))
            for label, low, high in SCF_EDUCATION_DEBT_BORROWER_AGE_BANDS
        },
        "by_race": {
            label: _group(race_group == code) for code, label in _RACECL4_LABELS.items()
        },
        "by_income_quintile": {
            str(quintile): _group(income_quintile == quintile)
            for quintile in range(1, 6)
        },
        "borrowers": {
            "borrower_weight": float(borrower_weight.sum()),
            "mean_balance": (
                float(np.average(borrower_balance, weights=borrower_weight))
                if float(borrower_weight.sum()) > 0
                else float("nan")
            ),
            "median_balance": _weighted_quantile(
                borrower_balance, borrower_weight, 0.5
            ),
            "balance_bands": _band_shares(
                borrower_balance,
                borrower_weight,
                SCF_EDUCATION_DEBT_BALANCE_BANDS,
                inclusive_upper=False,
            ),
            "age_bands": _band_shares(
                np.floor(borrower_age),
                borrower_weight,
                SCF_EDUCATION_DEBT_BORROWER_AGE_BANDS,
                inclusive_upper=True,
            ),
            "graduate_share_of_own_original_principal": (
                graduate_numerator / graduate_denominator
                if graduate_denominator > 0
                else float("nan")
            ),
            "ibr_new_borrower_share": (
                new_borrower_weight / float(borrower_weight.sum())
                if float(borrower_weight.sum()) > 0
                else float("nan")
            ),
        },
    }


def _spread_listing_ages(
    codes: np.ndarray, weights: np.ndarray, balances: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spread each listing age code's weight evenly over the ages it stands for.

    The public file bands listed members' ages (18-27 read 25), so a child
    borrower's weight is split evenly across the whole years of the band for
    the borrower age table. Returns ages, weights and balances, one row per
    (borrower, year of age).
    """

    youngest, oldest = _listing_age_range(codes)
    ages: list[np.ndarray] = []
    spread: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for low, high, weight, balance in zip(
        youngest, oldest, weights, balances, strict=True
    ):
        years = np.arange(low, min(high, 99.0) + 1.0)
        ages.append(years)
        spread.append(np.full(len(years), weight / len(years)))
        values.append(np.full(len(years), balance))
    if not ages:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return np.concatenate(ages), np.concatenate(spread), np.concatenate(values)


def scf_education_debt_donor_fit_table(
    donor: pd.DataFrame,
    *,
    imputed: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Fit table of the SCF donor itself, or of draws made for its rows.

    ``imputed`` (from :func:`draw_scf_education_debt_households`, on the
    donor's index) replaces the observed targets, so a hold-out draw and the
    observed SCF are tabulated identically. Implicates are pooled with
    ``wgt`` (each implicate carries one fifth of the population), as the Fed
    pools them for medians. Borrowers are the reference person, the spouse or
    partner and the adult child holding a child-own balance, whose age is the
    child's listing age band (the child the loan fits when observed, the
    youngest adult child when drawn), spread evenly over the band.
    """

    source = donor if imputed is None else imputed
    weight = donor[_DONOR_WEIGHT_COLUMN].to_numpy(dtype=np.float64)

    def _column(name: str) -> np.ndarray:
        return source[name].to_numpy(dtype=np.float64)

    head_own = _column("head_federal_student_loan_balance")
    spouse_own = _column("spouse_federal_student_loan_balance")
    child_own = _column("child_federal_student_loan_balance")
    parent_plus = _column("federal_parent_plus_loan_balance")
    head_balance = head_own + parent_plus
    head_holds = head_balance > 0
    spouse_holds = spouse_own > 0
    child_holds = child_own > 0
    head_age = donor["age"].to_numpy(dtype=np.float64)
    spouse_age = donor["spouse_age"].to_numpy(dtype=np.float64)
    child_age_code = donor[
        "child_borrower_age_code"
        if imputed is None
        else "youngest_adult_child_age_code"
    ].to_numpy(dtype=np.float64)
    child_ages, child_weights, child_balances = _spread_listing_ages(
        child_age_code[child_holds], weight[child_holds], child_own[child_holds]
    )
    head_graduate = _column("head_graduate_share")
    spouse_graduate = _column("spouse_graduate_share")
    child_graduate = _column("child_graduate_share")
    head_new = _column("head_is_ibr_new_borrower") >= 0.5
    spouse_new = _column("spouse_is_ibr_new_borrower") >= 0.5
    child_new = _column("child_is_ibr_new_borrower") >= 0.5
    income = donor["household_income"].to_numpy(dtype=np.float64)
    return education_debt_fit_table(
        household_debt=head_own + spouse_own + child_own + parent_plus,
        household_weight=weight,
        head_age=head_age,
        race_group=donor["racecl4"].to_numpy(),
        income_quintile=_income_quintiles(income, weight),
        borrower_balance=np.concatenate(
            [head_balance[head_holds], spouse_own[spouse_holds], child_balances]
        ),
        borrower_age=np.concatenate(
            [head_age[head_holds], spouse_age[spouse_holds], child_ages]
        ),
        borrower_weight=np.concatenate(
            [weight[head_holds], weight[spouse_holds], child_weights]
        ),
        parent_plus_total=float(np.dot(parent_plus, weight)),
        graduate_numerator=float(
            np.dot(
                head_own * head_graduate
                + spouse_own * spouse_graduate
                + child_own * child_graduate,
                weight,
            )
        ),
        graduate_denominator=float(np.dot(head_own + spouse_own + child_own, weight)),
        new_borrower_weight=float(
            weight[head_holds & head_new].sum()
            + weight[spouse_holds & spouse_new].sum()
            + weight[child_holds & child_new].sum()
        ),
    )


def _racecl4_from_cps(person: pd.DataFrame) -> np.ndarray:
    """SCF ``racecl4`` grouping of CPS race and Hispanic origin."""

    race = pd.to_numeric(person["PRDTRACE"], errors="coerce").fillna(0).to_numpy()
    hispanic = pd.to_numeric(person["PRDTHSP"], errors="coerce").fillna(0).to_numpy()
    group = np.full(len(person), 4, dtype=np.int64)
    group[race == 1] = 1
    group[race == 2] = 2
    group[hispanic > 0] = 3
    return group


def us_scf_education_debt_fit_table(
    frame: Frame,
    definition: RecipientDefinition = SCF_EDUCATION_DEBT_RECIPIENT_DEFINITION,
) -> dict[str, object]:
    """The microcosm#1073 fit table of an imputed US frame, at its weights.

    Households are grouped by the reference person's age and
    ``racecl4``-equivalent race, and by weighted quintiles of the household
    income the imputation conditions on (income quintiles of the frame's own
    households, like the SCF's), all under ``definition``.
    """

    person = frame.table("person")
    recipients = _recipients(frame, definition)
    household_weight = np.asarray(
        frame.weights_for("household").values, dtype=np.float64
    )
    own = (
        pd.to_numeric(person["federal_student_loan_balance"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    parent_plus = (
        pd.to_numeric(person["federal_parent_plus_loan_balance"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    graduate = (
        pd.to_numeric(person["federal_student_loan_graduate_share"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    new_borrower = (
        person["is_income_based_repayment_new_borrower"].astype(bool).to_numpy()
    )
    household_ids = person[_HOUSEHOLD_ID_COLUMN].to_numpy()
    order = frame.table("household")["household_id"].to_numpy()
    weight_by_household = pd.Series(household_weight, index=order)
    person_weight = weight_by_household.reindex(household_ids).to_numpy(
        dtype=np.float64
    )
    balance = own + parent_plus
    household_debt = (
        pd.Series(balance, index=household_ids)
        .groupby(level=0, sort=False)
        .sum()
        .reindex(order)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    holds = balance > 0
    age = pd.to_numeric(person["age"], errors="coerce").fillna(0.0).to_numpy()
    head = recipients.head_position
    return education_debt_fit_table(
        household_debt=household_debt,
        household_weight=household_weight,
        head_age=age[head],
        race_group=_racecl4_from_cps(person)[head],
        income_quintile=_income_quintiles(
            recipients.features["household_income"].to_numpy(dtype=np.float64),
            household_weight,
        ),
        borrower_balance=balance[holds],
        borrower_age=age[holds],
        borrower_weight=person_weight[holds],
        parent_plus_total=float(np.dot(parent_plus, person_weight)),
        graduate_numerator=float(np.dot(own * graduate, person_weight)),
        graduate_denominator=float(np.dot(own, person_weight)),
        new_borrower_weight=float(person_weight[holds & new_borrower].sum()),
    )


def us_scf_education_debt_summary(frame: Frame) -> dict[str, object]:
    """Weighted signal and consistency diagnostics for the four columns."""

    person = frame.table("person")
    present = [c for c in US_SCF_EDUCATION_DEBT_OUTPUT_COLUMNS if c in person.columns]
    summary: dict[str, object] = {
        "present_columns": present,
        "household_income_columns_present": [
            c for c in _RECIPIENT_HOUSEHOLD_INCOME_COLUMNS if c in person.columns
        ],
        "borrower_household_share_band": list(_BORROWER_HOUSEHOLD_SHARE_BAND),
    }
    if len(present) != len(US_SCF_EDUCATION_DEBT_OUTPUT_COLUMNS):
        return summary
    balances = {
        column: pd.to_numeric(person[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        for column in US_SCF_EDUCATION_DEBT_BALANCE_COLUMNS
    }
    graduate = pd.to_numeric(
        person["federal_student_loan_graduate_share"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    new_borrower = person["is_income_based_repayment_new_borrower"]
    holds = (
        balances["federal_student_loan_balance"]
        + balances["federal_parent_plus_loan_balance"]
    ) > 0
    household_ids = person[_HOUSEHOLD_ID_COLUMN].to_numpy()
    order = frame.table("household")["household_id"].to_numpy()
    household_weight = np.asarray(
        frame.weights_for("household").values, dtype=np.float64
    )
    household_holds = (
        pd.Series(holds, index=household_ids)
        .groupby(level=0, sort=False)
        .any()
        .reindex(order)
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    total_weight = float(household_weight.sum())
    summary.update(
        {
            "nonfinite_counts": {
                column: int((~np.isfinite(values)).sum())
                for column, values in balances.items()
            },
            "negative_counts": {
                column: int((values < 0).sum()) for column, values in balances.items()
            },
            "unique_counts": {
                column: int(pd.Series(values).nunique())
                for column, values in balances.items()
            },
            "graduate_share_out_of_range": int(
                ((graduate < 0) | (graduate > 1) | ~np.isfinite(graduate)).sum()
            ),
            "graduate_share_without_own_balance": int(
                ((graduate > 0) & (balances["federal_student_loan_balance"] <= 0)).sum()
            ),
            "new_borrower_dtype_is_bool": bool(
                pd.api.types.is_bool_dtype(new_borrower)
            ),
            "new_borrower_without_balance": int(
                (new_borrower.astype(bool).to_numpy() & ~holds).sum()
            ),
            "borrower_household_share": (
                float(household_weight[household_holds].sum()) / total_weight
                if total_weight > 0
                else 0.0
            ),
        }
    )
    return summary


def us_scf_education_debt_signal_gate(frame: Frame) -> GateResult:
    """Require finite, nonnegative, nonconstant and coherent education debt."""

    summary = us_scf_education_debt_summary(frame)
    missing = sorted(
        set(US_SCF_EDUCATION_DEBT_OUTPUT_COLUMNS) - set(summary["present_columns"])
    )
    if missing:
        return GateResult(
            name="scf_education_debt_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details=summary,
        )
    failures: list[str] = []
    for column, count in summary["nonfinite_counts"].items():
        if count:
            failures.append(f"{column}: {count} nonfinite value(s).")
    for column, count in summary["negative_counts"].items():
        if count:
            failures.append(f"{column}: {count} negative value(s).")
    for column, count in summary["unique_counts"].items():
        if count < 2:
            failures.append(f"{column}: constant column carries no signal.")
    if summary["graduate_share_out_of_range"]:
        failures.append(
            "federal_student_loan_graduate_share: "
            f"{summary['graduate_share_out_of_range']} value(s) outside [0, 1]."
        )
    if summary["graduate_share_without_own_balance"]:
        failures.append(
            "federal_student_loan_graduate_share is positive for "
            f"{summary['graduate_share_without_own_balance']} person(s) without "
            "a federal_student_loan_balance."
        )
    if not summary["new_borrower_dtype_is_bool"]:
        failures.append("is_income_based_repayment_new_borrower is not boolean.")
    if summary["new_borrower_without_balance"]:
        failures.append(
            "is_income_based_repayment_new_borrower is true for "
            f"{summary['new_borrower_without_balance']} person(s) without a "
            "federal balance."
        )
    share = float(summary["borrower_household_share"])
    low, high = _BORROWER_HOUSEHOLD_SHARE_BAND
    if not low <= share <= high:
        failures.append(
            f"federal education-debt household share {share:.3f} outside "
            f"plausibility band [{low}, {high}]."
        )
    return GateResult(
        name="scf_education_debt_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
