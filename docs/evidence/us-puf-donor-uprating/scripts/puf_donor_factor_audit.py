"""Classify every PUF donor column by the uprating factor it carries.

Usage:
    python -I puf_donor_factor_audit.py \
        --donor <puf_2024.h5 read by the base build> \
        --source-csv <puf_2015.csv read by the base build> \
        --release-2021 <irs-soi-puf 1.8.0 puf_2021.h5> \
        --release-2024 <irs-soi-puf 1.8.0 puf_2024.h5> \
        --archived-dir <this folder's archived/> \
        --microcosm-root <microcosm checkout at the build commit> \
        --out <receipt.json>

Every output is an aggregate: counts, shares, factors and weighted totals.
No record value leaves the inputs.

Method. The archived generator (policyengine-us-data 42ed5d45,
``datasets/puf/uprate_puf.py``, copied unchanged into ``archived/``) uprates
the raw TY2015 PUF field by field. This script runs that ``uprate_puf``
unchanged on the source CSV to each year the archived SOI target table
holds (2021, 2022, 2023), keeps the raw file as the 2015 hypothesis, and
evaluates each donor column's archived source expression under every
hypothesis. A record matches a hypothesis when the donor's tax-unit total
equals the hypothesis value to a relative tolerance (1e-9 for float64
arrays, 1e-6 for float32 arrays). For single-field columns it also reports
the modal per-record ratio of donor to raw value, which reads the factor off
the data without assuming any hypothesis.

Columns Microcosm consumes are those of the donor table its own loader
builds (``tools/build_us_puf_support_base.py::_puf_tax_unit_donor_from_h5``
at the build commit), which is called here on the same files.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

AGGREGATE_RECIDS = (999_996, 999_997, 999_998, 999_999)
SYNTHETIC_START = 1_000_000
HYPOTHESIS_YEARS = (2021, 2022, 2023)
OWN_SERIES_ONLY = ("estate_income", "rent_and_royalty_net_income")

# Archived generator source expressions (42ed5d45 datasets/puf/puf.py
# ``preprocess_puf`` lines 636-746), keyed by the processed array name. Each
# value is (expression over the PUF frame, fields it reads).
ARRAY_EXPRESSIONS: dict[str, tuple] = {
    "alimony_expense": (lambda p: p.E03500, ("E03500",)),
    "alimony_income": (lambda p: p.E00800, ("E00800",)),
    "casualty_loss": (lambda p: p.E20500, ("E20500",)),
    "cdcc_relevant_expenses": (lambda p: p.E32800, ("E32800",)),
    "charitable_cash_donations": (lambda p: p.E19800, ("E19800",)),
    "charitable_non_cash_donations": (lambda p: p.E20100, ("E20100",)),
    "domestic_production_ald": (lambda p: p.E03240, ("E03240",)),
    # preprocess_puf assigns E03400 then overwrites with E09900.
    "early_withdrawal_penalty": (lambda p: p.E09900, ("E09900",)),
    "educator_expense": (lambda p: p.E03220, ("E03220",)),
    "employment_income": (lambda p: p.E00200, ("E00200",)),
    "estate_income": (lambda p: p.E26390 - p.E26400, ("E26390", "E26400")),
    "farm_income": (lambda p: p.T27800, ("T27800",)),
    "health_savings_account_ald": (lambda p: p.E03290, ("E03290",)),
    "long_term_capital_gains": (lambda p: p.P23250, ("P23250",)),
    "long_term_capital_gains_on_collectibles": (lambda p: p.E24518, ("E24518",)),
    "unreimbursed_business_employee_expenses": (lambda p: p.E20400, ("E20400",)),
    "non_qualified_dividend_income": (
        lambda p: p.E00600 - p.E00650,
        ("E00600", "E00650"),
    ),
    "qualified_dividend_income": (lambda p: p.E00650, ("E00650",)),
    # Two candidate definitions; the processed file decides.
    "qualified_tuition_expenses[max(E03230,E87530)]": (
        lambda p: np.maximum(p.E03230, p.E87530),
        ("E03230", "E87530"),
    ),
    "qualified_tuition_expenses[E03230]": (lambda p: p.E03230, ("E03230",)),
    "real_estate_taxes": (lambda p: p.E18500, ("E18500",)),
    "rental_income": (lambda p: p.E25850 - p.E25860, ("E25850", "E25860")),
    "farm_operations_income": (lambda p: p.E02100, ("E02100",)),
    "farm_rent_income": (lambda p: p.E27200, ("E27200",)),
    # Before the SSTB split; tested on the sum of the two split arrays.
    "self_employment_income+sstb_self_employment_income": (
        lambda p: p.E00900,
        ("E00900",),
    ),
    "self_employed_health_insurance_ald": (lambda p: p.E03270, ("E03270",)),
    "self_employed_pension_contribution_ald": (lambda p: p.E03300, ("E03300",)),
    "short_term_capital_gains": (lambda p: p.P22250, ("P22250",)),
    "social_security": (lambda p: p.E02400, ("E02400",)),
    "state_and_local_sales_or_income_tax": (lambda p: p.E18400, ("E18400",)),
    "student_loan_interest": (lambda p: p.E03210, ("E03210",)),
    "taxable_interest_income": (lambda p: p.E00300, ("E00300",)),
    "taxable_pension_income": (lambda p: p.E01700, ("E01700",)),
    "taxable_unemployment_compensation": (lambda p: p.E02300, ("E02300",)),
    "taxable_ira_distributions": (lambda p: p.E01400, ("E01400",)),
    "tax_exempt_interest_income": (lambda p: p.E00400, ("E00400",)),
    "tax_exempt_pension_income": (
        lambda p: p.E01500 - p.E01700,
        ("E01500", "E01700"),
    ),
    "traditional_ira_contributions": (lambda p: p.E03150, ("E03150",)),
    "unrecaptured_section_1250_gain": (lambda p: p.E24515, ("E24515",)),
    "foreign_tax_credit": (lambda p: p.E07300, ("E07300",)),
    "amt_foreign_tax_credit": (lambda p: p.E62900, ("E62900",)),
    "miscellaneous_income": (lambda p: p.E01200, ("E01200",)),
    "salt_refund_income": (lambda p: p.E00700, ("E00700",)),
    "investment_income_elected_form_4952": (lambda p: p.E58990, ("E58990",)),
    "general_business_credit": (lambda p: p.E07400, ("E07400",)),
    "prior_year_minimum_tax_credit": (lambda p: p.E07600, ("E07600",)),
    "excess_withheld_payroll_tax": (lambda p: p.E11200, ("E11200",)),
    "non_sch_d_capital_gains": (lambda p: p.E01100, ("E01100",)),
    "american_opportunity_credit": (lambda p: p.E87521, ("E87521",)),
    "energy_efficient_home_improvement_credit": (lambda p: p.E07260, ("E07260",)),
    "other_credits": (lambda p: p.P08000, ("P08000",)),
    "savers_credit": (lambda p: p.E07240, ("E07240",)),
    "recapture_of_investment_credit": (lambda p: p.E09700, ("E09700",)),
    "unreported_payroll_tax": (lambda p: p.E09800, ("E09800",)),
    # Two candidate definitions; the processed file decides. E26270 is the
    # generator's definition before 2025-05-22 (c33e3bff), the one the
    # published 1.8.0 file carries; the split components are the definition
    # at 42ed5d45.
    "partnership_s_corp_income[E26270]": (lambda p: p.E26270, ("E26270",)),
    "partnership_s_corp_income[components]": (
        lambda p: (p.E26190 - p.E26180) + (p.E25980 - p.E25960),
        ("E26190", "E26180", "E25980", "E25960"),
    ),
    # E19200 lands in the structural mortgage arrays plus the investment
    # interest residual; tested on their sum.
    "home_mortgage_interest+investment_interest_expense": (
        lambda p: p.E19200,
        ("E19200",),
    ),
    "interest_deduction": (lambda p: p.E19200, ("E19200",)),
}

# How each array named in ARRAY_EXPRESSIONS is read from a processed file.
ARRAY_SOURCES = {
    "self_employment_income+sstb_self_employment_income": (
        "self_employment_income",
        "sstb_self_employment_income",
    ),
    "qualified_tuition_expenses[max(E03230,E87530)]": ("qualified_tuition_expenses",),
    "qualified_tuition_expenses[E03230]": ("qualified_tuition_expenses",),
    "partnership_s_corp_income[E26270]": ("partnership_s_corp_income",),
    "partnership_s_corp_income[components]": ("partnership_s_corp_income",),
    "home_mortgage_interest+investment_interest_expense": (
        "home_mortgage_interest",
        "investment_interest_expense",
    ),
}

# Microcosm donor column -> processed array key in ARRAY_EXPRESSIONS
# (puf_support.py ``_person_source_values`` and ``_tax_unit_source_values``).
MICROCOSM_COLUMN_SOURCES = {
    "alimony_expense": "alimony_expense",
    "alimony_income": "alimony_income",
    "casualty_loss": "casualty_loss",
    "charitable_cash_donations": "charitable_cash_donations",
    "charitable_non_cash_donations": "charitable_non_cash_donations",
    "domestic_production_ald": "domestic_production_ald",
    "educator_expense": "educator_expense",
    "employment_income_before_lsr": "employment_income",
    "estate_income": "estate_income",
    "farm_income": "farm_income",
    "farm_operations_income": "farm_operations_income",
    "farm_rent_income": "farm_rent_income",
    "health_savings_account_ald": "health_savings_account_ald",
    "investment_income_elected_form_4952": "investment_income_elected_form_4952",
    "long_term_capital_gains_before_response": "long_term_capital_gains",
    "long_term_capital_gains_on_collectibles": (
        "long_term_capital_gains_on_collectibles"
    ),
    "miscellaneous_income": "miscellaneous_income",
    "non_qualified_dividend_income": "non_qualified_dividend_income",
    "non_sch_d_capital_gains": "non_sch_d_capital_gains",
    "qualified_dividend_income": "qualified_dividend_income",
    "real_estate_taxes": "real_estate_taxes",
    "rental_income": "rental_income",
    "salt_refund_income": "salt_refund_income",
    "self_employed_pension_contributions_desired": (
        "self_employed_pension_contribution_ald"
    ),
    "short_term_capital_gains": "short_term_capital_gains",
    "social_security_retirement": "social_security",
    "student_loan_interest": "student_loan_interest",
    "tax_exempt_interest_income": "tax_exempt_interest_income",
    "taxable_interest_income": "taxable_interest_income",
    "taxable_ira_distributions": "taxable_ira_distributions",
    "taxable_private_pension_income": "taxable_pension_income",
    "traditional_ira_contributions_desired": "traditional_ira_contributions",
    "unrecaptured_section_1250_gain": "unrecaptured_section_1250_gain",
    "unreimbursed_business_employee_expenses": (
        "unreimbursed_business_employee_expenses"
    ),
}
# Donor columns tested as sums of two Microcosm columns.
MICROCOSM_SUM_SOURCES = {
    "self_employment_income_before_lsr+sstb_self_employment_income_before_lsr": (
        "self_employment_income+sstb_self_employment_income"
    ),
    "home_mortgage_interest+investment_interest_expense": (
        "home_mortgage_interest+investment_interest_expense"
    ),
}
# Microcosm columns read from an array with two candidate definitions; the
# definition the processed file matches is used. partnership_income reads
# partnership_s_corp_income when the file has no split leaves
# (puf_support.py ``_person_source_values``).
MICROCOSM_CANDIDATE_SOURCES = {
    "partnership_income": (
        "partnership_s_corp_income[E26270]",
        "partnership_s_corp_income[components]",
    ),
    "qualified_tuition_expenses": (
        "qualified_tuition_expenses[max(E03230,E87530)]",
        "qualified_tuition_expenses[E03230]",
    ),
}

# (label, Microcosm donor column, archived SOI variable) for lines whose donor
# column is one PUF field aged by that field's own SOI line.
SOI_TOTAL_LINES = (
    ("Wages", "employment_income_before_lsr", "employment_income"),
    ("Taxable interest", "taxable_interest_income", "taxable_interest_income"),
    ("Tax-exempt interest", "tax_exempt_interest_income", "exempt_interest"),
    ("Qualified dividends", "qualified_dividend_income", "qualified_dividends"),
    (
        "Capital gain distributions",
        "non_sch_d_capital_gains",
        "capital_gains_distributions",
    ),
    ("Taxable IRA distributions", "taxable_ira_distributions", "ira_distributions"),
    ("Taxable pensions", "taxable_private_pension_income", "taxable_pension_income"),
    ("Social Security benefits", "social_security_retirement", "total_social_security"),
)

NON_MONETARY = {
    "tax_unit_id": "identifier",
    "filing_status_code": "categorical",
    "head_age": "demographic",
    "spouse_age": "demographic",
    "head_is_female": "demographic",
    "dependent_count": "demographic",
    "tax_unit_person_count": "demographic",
    "first_home_mortgage_origination_year": "year",
    "second_home_mortgage_origination_year": "year",
    "puf_predictor_filing_status_code": "predictor alias",
    "puf_predictor_tax_unit_person_count": "predictor alias",
    "puf_predictor_head_age": "predictor alias",
    "puf_predictor_spouse_age": "predictor alias",
    "puf_predictor_head_is_female": "predictor alias",
    "puf_predictor_dependent_count": "predictor alias",
    "puf_predictor_has_earnings": "predictor (0/1)",
    "puf_predictor_income_rank_share": "predictor (weighted rank, scale-free)",
    "business_is_sstb": "flag (simulated)",
    "estate_income_would_be_qualified": "flag (simulated)",
    "farm_operations_income_would_be_qualified": "flag (simulated)",
    "farm_rent_income_would_be_qualified": "flag (simulated)",
    "partnership_s_corp_income_would_be_qualified": "flag (simulated)",
    "rental_income_would_be_qualified": "flag (simulated)",
    "self_employment_income_would_be_qualified": "flag (simulated)",
    "sstb_self_employment_income_would_be_qualified": "flag (simulated)",
}
SIMULATED_FROM_UPRATED = {
    "w2_wages_from_qualified_business": (
        "simulate_w2_and_ubia_from_puf: draws scaled by uprated business income"
    ),
    "unadjusted_basis_qualified_property": (
        "simulate_w2_and_ubia_from_puf: draws scaled by uprated business income"
    ),
    "sstb_w2_wages_from_qualified_business": "SSTB share of the W-2 wage draw",
    "sstb_unadjusted_basis_qualified_property": "SSTB share of the UBIA draw",
    "qualified_reit_and_ptp_income": (
        "simulate_investment_qbi_income_from_puf: beta draws scaled by uprated "
        "non-qualified dividends and partnership/S-corp income"
    ),
    "qualified_bdc_income": (
        "simulate_investment_qbi_income_from_puf: beta draws scaled by uprated "
        "non-qualified dividends and partnership/S-corp income"
    ),
    "partnership_self_employment_net_earnings": (
        "gross SE (E30400+E30500)/0.9235 less E00900 and E02100 where the "
        "record has partnership activity; mixes unuprated and uprated fields"
    ),
    "first_home_mortgage_balance": "structural mortgage input derived from E19200",
    "second_home_mortgage_balance": "structural mortgage input derived from E19200",
    "first_home_mortgage_interest": "structural mortgage split of E19200",
    "second_home_mortgage_interest": "structural mortgage split of E19200",
}
ZERO_BY_CONSTRUCTION = {
    "social_security_disability": "puf_support.py sets zeros (PUF has no split)",
    "social_security_survivors": "puf_support.py sets zeros (PUF has no split)",
    "social_security_dependents": "puf_support.py sets zeros (PUF has no split)",
    "s_corp_income": (
        "puf_support.py sets zeros when the file has no split leaves; "
        "partnership_income carries the combined amount"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_archived_uprate(archived_dir: Path):
    """Exec archived uprate_puf.py unchanged with its storage import stubbed."""

    calibration = Path(tempfile.mkdtemp(prefix="soi_"))
    (calibration / "soi_targets.csv").write_bytes(
        gzip.decompress((archived_dir / "soi_targets_42ed5d45.csv.gz").read_bytes())
    )
    storage = types.ModuleType("policyengine_us_data.storage")
    storage.CALIBRATION_FOLDER = calibration
    sys.modules.setdefault("policyengine_us_data", types.ModuleType("pud"))
    sys.modules["policyengine_us_data.storage"] = storage
    source = archived_dir / "uprate_puf_42ed5d45.py.txt"
    namespace: dict = {}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    return namespace


def read_arrays(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {key: handle[key][()] for key in handle.keys()}


def tax_unit_totals(arrays: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict]:
    """Sum person arrays to tax units; carry tax-unit arrays as stored."""

    tax_unit_id = arrays["tax_unit_id"].astype(np.int64)
    person_tax_unit = arrays["person_tax_unit_id"].astype(np.int64)
    n_units, n_persons = len(tax_unit_id), len(person_tax_unit)
    position = pd.Index(tax_unit_id).get_indexer(person_tax_unit)
    if (position < 0).any():
        raise ValueError("person_tax_unit_id not found among tax_unit_id")
    columns, dtypes = {}, {}
    for key, values in arrays.items():
        if values.dtype.kind not in "fiub" or key.endswith("_id"):
            continue
        values = values.astype(np.float64)
        if len(values) == n_persons and n_persons != n_units:
            columns[key] = np.bincount(position, weights=values, minlength=n_units)
        elif len(values) == n_units:
            columns[key] = values
        else:
            continue
        dtypes[key] = str(arrays[key].dtype)
    frame = pd.DataFrame(columns)
    frame.insert(0, "tax_unit_id", tax_unit_id)
    return frame, dtypes


def source_values(frame: pd.DataFrame, key: str) -> np.ndarray | None:
    names = ARRAY_SOURCES.get(key, (key,))
    if not all(name in frame for name in names):
        return None
    return np.sum([frame[name].to_numpy(dtype=np.float64) for name in names], axis=0)


def match(observed: np.ndarray, expected: np.ndarray, rtol: float) -> np.ndarray:
    return np.abs(observed - expected) <= rtol * np.maximum(1.0, np.abs(expected))


def classify(
    observed: np.ndarray,
    hypotheses: dict[str, pd.DataFrame],
    expression,
    fields: tuple[str, ...],
    rtol: float,
    raw: pd.DataFrame,
) -> dict:
    raw_value = np.asarray(expression(raw), dtype=np.float64)
    nonzero = raw_value != 0
    result = {
        "fields": list(fields),
        "records": int(len(observed)),
        "raw_nonzero_records": int(nonzero.sum()),
        "processed_nonzero_records": int((observed != 0).sum()),
        "processed_nonzero_where_raw_zero": int(((observed != 0) & ~nonzero).sum()),
        "match_share_of_raw_nonzero": {},
        "match_share_of_all_records": {},
    }
    for label, frame in hypotheses.items():
        expected = np.asarray(expression(frame), dtype=np.float64)
        ok = match(observed, expected, rtol)
        if label == "2015 to 2021" and nonzero.any():
            result["unmatched_raw_nonzero_above_2021_value"] = int(
                (~ok & nonzero & (observed > expected)).sum()
            )
            result["unmatched_raw_nonzero_below_2021_value"] = int(
                (~ok & nonzero & (observed < expected)).sum()
            )
            result["unmatched_raw_nonzero_below_and_exactly_zero"] = int(
                (~ok & nonzero & (observed < expected) & (observed == 0)).sum()
            )
        result["match_share_of_all_records"][label] = round(float(ok.mean()), 6)
        result["match_share_of_raw_nonzero"][label] = (
            round(float(ok[nonzero].mean()), 6) if nonzero.any() else None
        )
    shares = {
        k: v for k, v in result["match_share_of_raw_nonzero"].items() if v is not None
    }
    result["best_hypothesis"] = max(shares, key=shares.get) if shares else None
    if len(fields) == 1 and nonzero.any():
        ratio = observed[nonzero] / raw_value[nonzero]
        rounded = np.round(ratio, 9)
        counts = Counter(rounded.tolist())
        mode, count = counts.most_common(1)[0]
        result["modal_ratio_processed_to_raw"] = mode
        result["modal_ratio_share"] = round(count / int(nonzero.sum()), 6)
    return result


def classify_processed(
    units: pd.DataFrame,
    dtypes: dict,
    raw: pd.DataFrame,
    hypotheses: dict[str, pd.DataFrame],
    every_field: set,
) -> dict:
    out = {}
    for key, (expression, fields) in ARRAY_EXPRESSIONS.items():
        observed = source_values(units, key)
        if observed is None:
            continue
        names = ARRAY_SOURCES.get(key, (key,))
        rtol = 1e-6 if any(dtypes.get(n) == "float32" for n in names) else 1e-9
        result = classify(observed, hypotheses, expression, fields, rtol, raw)
        result["fields_outside_every_uprate_list"] = sorted(set(fields) - every_field)
        result["dtype"] = sorted({dtypes.get(n, "?") for n in names})
        out[key] = result
    return out


def factor_table(uprate_ns) -> dict:
    """The archived per-field factors, 2015 to Y and 2021 to Y.

    The 2015-to-Y rows read the ratios. The 2021-to-2022 and 2021-to-2023
    rows size the leg the 2024 branch never applied, by the archived method
    (SOI aggregate growth over return-count growth); the table stops at 2023.
    """

    get_growth = uprate_ns["get_growth"]
    table: dict[str, dict] = {}
    spans = [(2015, year) for year in HYPOTHESIS_YEARS] + [(2021, 2022), (2021, 2023)]
    for start, year in spans:
        row = {}
        for variable, field in uprate_ns["SOI_TO_PUF_STRAIGHT_RENAMES"].items():
            row[f"{field} ({variable})"] = get_growth(variable, start, year)
        row["itemized deductions (1.02 per year)"] = (
            1 + uprate_ns["ITMDED_GROW_RATE"]
        ) ** (year - start)
        for name in ("POS_ONLY", "NEG_ONLY"):
            for variable, field in uprate_ns[f"SOI_TO_PUF_{name}_RENAMES"].items():
                row[f"{field} {name.lower()} ({variable})"] = get_growth(
                    variable, start, year
                )
        row["REMAINING_VARIABLES (adjusted_gross_income)"] = get_growth(
            "adjusted_gross_income", start, year
        )
        get_agg = uprate_ns["get_soi_aggregate"]
        row["S006 weights (return count)"] = get_agg("count", year, True) / get_agg(
            "count", start, True
        )
        # Own SOI lines the archived generator does not use for these fields
        # (it ages E26390/E26400 and E25850/E25860 by AGI); read for the
        # own-series comparison only.
        for variable in OWN_SERIES_ONLY:
            row[f"own series ({variable})"] = get_growth(variable, start, year)
        table[str(year) if start == 2015 else f"{start} to {year}"] = row
    return table


def field_lists(uprate_ns) -> dict:
    return {
        "straight": sorted(uprate_ns["SOI_TO_PUF_STRAIGHT_RENAMES"].values()),
        "positive_only": sorted(uprate_ns["SOI_TO_PUF_POS_ONLY_RENAMES"].values()),
        "negative_only": sorted(uprate_ns["SOI_TO_PUF_NEG_ONLY_RENAMES"].values()),
        "remaining_agi_growth": sorted(uprate_ns["REMAINING_VARIABLES"]),
    }


def release_identity(path_2021: Path, path_2024: Path) -> dict:
    a, b = read_arrays(path_2021), read_arrays(path_2024)
    identical, differing = [], []
    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        same = (
            np.array_equal(x, y, equal_nan=True)
            if x.dtype.kind == "f" and y.dtype.kind == "f"
            else np.array_equal(x, y)
        )
        (identical if same else differing).append(key)
    return {
        "arrays_in_2021_only": sorted(set(a) - set(b)),
        "arrays_in_2024_only": sorted(set(b) - set(a)),
        "arrays_identical": len(identical),
        "arrays_differing": differing,
        "rows": {k: int(len(v)) for k, v in a.items() if k in ("tax_unit_id",)},
    }


def donor_vs_release(donor: pd.DataFrame, release: pd.DataFrame) -> dict:
    """Compare common tax-unit totals of the build donor and the 1.8.0 asset."""

    joined = donor.merge(
        release, on="tax_unit_id", how="inner", suffixes=("_donor", "_release")
    )
    regular = joined["tax_unit_id"] < SYNTHETIC_START
    out = {
        "common_regular_tax_units": int(regular.sum()),
        "donor_tax_units": int(len(donor)),
        "release_tax_units": int(len(release)),
        "columns": {},
    }
    for column in sorted(set(donor.columns) & set(release.columns) - {"tax_unit_id"}):
        x = joined.loc[regular, f"{column}_donor"].to_numpy(dtype=np.float64)
        y = joined.loc[regular, f"{column}_release"].to_numpy(dtype=np.float64)
        nonzero = (x != 0) | (y != 0)
        out["columns"][column] = {
            "nonzero_records": int(nonzero.sum()),
            "equal_share_of_nonzero": (
                round(float(match(x, y, 1e-9)[nonzero].mean()), 6)
                if nonzero.any()
                else None
            ),
        }
    return out


def microcosm_donor(microcosm_root: Path, donor_path: Path, csv_path: Path):
    tools = microcosm_root / "tools" / "build_us_puf_support_base.py"
    spec = importlib.util.spec_from_file_location("base_tool", tools)
    module = importlib.util.module_from_spec(spec)
    sys.modules["base_tool"] = module
    spec.loader.exec_module(module)
    summary: dict = {}
    donor = module._puf_tax_unit_donor_from_h5(
        donor_path, source_puf_csv=csv_path, donor_build_summary=summary
    )
    return donor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--release-2021", type=Path, required=True)
    parser.add_argument("--release-2024", type=Path, required=True)
    parser.add_argument("--archived-dir", type=Path, required=True)
    parser.add_argument("--microcosm-root", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    receipt: dict = {
        "inputs": {
            name: {
                "file": getattr(args, name).name,
                "sha256": sha256(getattr(args, name)),
            }
            for name in (
                "donor",
                "source_csv",
                "release_2021",
                "release_2024",
                "summary_json",
            )
        },
        "archived": {
            path.name: sha256(path) for path in sorted(args.archived_dir.iterdir())
        },
        "archived_soi_targets_uncompressed_sha256": hashlib.sha256(
            gzip.decompress(
                (args.archived_dir / "soi_targets_42ed5d45.csv.gz").read_bytes()
            )
        ).hexdigest(),
        "tolerance": {"float64": 1e-9, "float32": 1e-6},
    }

    # 1. Published release: is puf_2024 the same as puf_2021?
    receipt["release_1_8_0_2021_vs_2024"] = release_identity(
        args.release_2021, args.release_2024
    )

    # 2. Raw TY2015 records aligned to the donor's regular tax units.
    raw = pd.read_csv(args.source_csv)
    raw = raw.loc[~raw["RECID"].isin(AGGREGATE_RECIDS)].reset_index(drop=True)
    for column in raw.columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(0.0)
    uprate_ns = load_archived_uprate(args.archived_dir)
    hypotheses = {"2015 (raw, no uprating)": raw}
    for year in HYPOTHESIS_YEARS:
        hypotheses[f"2015 to {year}"] = uprate_ns["uprate_puf"](raw, 2015, year)
    receipt["archived_factor_table"] = factor_table(uprate_ns)
    receipt["archived_field_lists"] = field_lists(uprate_ns)
    every_field = set().union(*field_lists(uprate_ns).values())

    donor_arrays = read_arrays(args.donor)
    receipt["donor_attrs"] = {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in h5py.File(args.donor, "r").attrs.items()
    }
    donor_units, dtypes = tax_unit_totals(donor_arrays)
    regular = donor_units["tax_unit_id"] < SYNTHETIC_START
    regular_units = donor_units.loc[regular].reset_index(drop=True)
    if not np.array_equal(
        regular_units["tax_unit_id"].to_numpy(), raw["RECID"].to_numpy(np.int64)
    ):
        raise ValueError("donor regular tax units are not in source RECID order")
    receipt["donor_shape"] = {
        "persons": int(len(donor_arrays["person_tax_unit_id"])),
        "tax_units": int(len(donor_units)),
        "regular_tax_units": int(regular.sum()),
        "synthetic_tax_units": int((~regular).sum()),
        "arrays": len(donor_arrays),
    }

    # 3. Weights.
    weight = regular_units["household_weight"].to_numpy()
    raw_weight = raw["S006"].to_numpy() / 100
    weight_ratio = np.round(weight / raw_weight, 9)
    mode, count = Counter(weight_ratio.tolist()).most_common(1)[0]
    receipt["donor_weight"] = {
        "modal_ratio_to_S006_over_100": mode,
        "modal_ratio_share": round(count / len(weight), 6),
    }

    # 4. Every processed array with an archived source expression.
    arrays_out = classify_processed(regular_units, dtypes, raw, hypotheses, every_field)
    receipt["processed_arrays"] = arrays_out
    covered = set()
    for key in arrays_out:
        covered.update(ARRAY_SOURCES.get(key, (key,)))
    receipt["processed_arrays_without_source_expression"] = sorted(
        k
        for k in donor_arrays
        if k not in covered and donor_arrays[k].dtype.kind in "fiub"
    )

    # 5. GitHub 1.8.0 puf_2024 against the build donor.
    release_units, release_dtypes = tax_unit_totals(read_arrays(args.release_2024))
    receipt["donor_vs_release_1_8_0_puf_2024"] = donor_vs_release(
        donor_units, release_units
    )
    release_regular = release_units.loc[
        release_units["tax_unit_id"] < SYNTHETIC_START
    ].reset_index(drop=True)
    if np.array_equal(
        release_regular["tax_unit_id"].to_numpy(), raw["RECID"].to_numpy(np.int64)
    ):
        receipt["release_1_8_0_puf_2024_processed_arrays"] = classify_processed(
            release_regular, release_dtypes, raw, hypotheses, every_field
        )
    else:
        receipt["release_1_8_0_puf_2024_processed_arrays"] = (
            "not classified: tax units are not in source RECID order"
        )

    # 6. Microcosm's donor table: classify all of its columns.
    table = microcosm_donor(args.microcosm_root, args.donor, args.source_csv)
    summary = json.loads(args.summary_json.read_text())
    expected_columns = summary["puf_donor_columns"]
    receipt["microcosm_donor"] = {
        "columns": len(table.columns),
        "columns_match_base_summary": sorted(table.columns) == sorted(expected_columns),
        "rows": int(len(table)),
        "rows_match_base_summary": int(len(table)) == int(summary["puf_donor_rows"]),
    }
    table_regular = table.loc[table["tax_unit_id"] < SYNTHETIC_START].reset_index(
        drop=True
    )
    if not np.array_equal(
        table_regular["tax_unit_id"].to_numpy(np.int64),
        raw["RECID"].to_numpy(np.int64),
    ):
        raise ValueError("Microcosm donor regular rows are not in RECID order")
    classes = {}
    for column in expected_columns:
        entry: dict = {}
        if column in NON_MONETARY:
            entry["class"] = "not a dollar amount"
            entry["kind"] = NON_MONETARY[column]
        elif column == "weight":
            ratio = np.round(
                table_regular["weight"].to_numpy() / (raw["S006"].to_numpy() / 100), 9
            )
            mode, count = Counter(ratio.tolist()).most_common(1)[0]
            entry["class"] = "design weight"
            entry["modal_ratio_to_S006_over_100"] = mode
            entry["modal_ratio_share"] = round(count / len(ratio), 6)
        elif column == "puf_source_adjusted_gross_income":
            values = table_regular[column].to_numpy(dtype=np.float64)
            entry["class"] = "TY2015 source value by construction"
            entry["equals_raw_E00100_share"] = round(
                float(match(values, raw["E00100"].to_numpy(np.float64), 1e-12).mean()),
                6,
            )
        elif column in ZERO_BY_CONSTRUCTION:
            values = table[column].to_numpy(dtype=np.float64)
            entry["class"] = "zero by construction"
            entry["note"] = ZERO_BY_CONSTRUCTION[column]
            entry["all_zero"] = bool((values == 0).all())
        elif (
            column in MICROCOSM_COLUMN_SOURCES or column in MICROCOSM_CANDIDATE_SOURCES
        ):
            if column in MICROCOSM_CANDIDATE_SOURCES:
                key = max(
                    MICROCOSM_CANDIDATE_SOURCES[column],
                    key=lambda k: max(
                        v
                        for v in arrays_out.get(k, {})
                        .get("match_share_of_raw_nonzero", {})
                        .values()
                        if v is not None
                    ),
                )
            else:
                key = MICROCOSM_COLUMN_SOURCES[column]
            expression, fields = ARRAY_EXPRESSIONS[key]
            observed = table_regular[column].to_numpy(dtype=np.float64)
            entry = classify(observed, hypotheses, expression, fields, 1e-9, raw)
            entry["class"] = "monetary, single source expression"
            entry["processed_array"] = key
        elif column in SIMULATED_FROM_UPRATED:
            entry["class"] = "derived or simulated from processed amounts"
            entry["note"] = SIMULATED_FROM_UPRATED[column]
        else:
            entry["class"] = "tested as part of a sum"
        classes[column] = entry
    for label, key in MICROCOSM_SUM_SOURCES.items():
        parts = label.split("+")
        if not all(part in table_regular for part in parts):
            continue
        observed = np.sum(
            [table_regular[part].to_numpy(dtype=np.float64) for part in parts], axis=0
        )
        expression, fields = ARRAY_EXPRESSIONS[key]
        rtol = 1e-6 if "mortgage" in label else 1e-9
        entry = classify(observed, hypotheses, expression, fields, rtol, raw)
        entry["class"] = "monetary, tested as a sum"
        entry["note"] = (
            "E19200 split by Microcosm into mortgage and investment interest by "
            "TY2015 SOI AGI band; field-local quarantine zeroes raw totals of "
            "$10M or more"
            "; the archived structural conversion (utils/mortgage_interest.py "
            "lines 274-283 at 42ed5d45) divides deductible interest by the "
            "deductible share where balances exceed the cap, so unmatched "
            "records sit above the uprated E19200"
            if "mortgage" in label
            else "SSTB split of Schedule C income"
        )
        classes[label] = entry
    receipt["microcosm_donor_columns"] = classes

    # 6b. Dividends: the donor stores qualified = Q x a and non-qualified =
    # O x b - Q x a. Recover TY2015 O and Q from the two stored columns and
    # compare with the source fields.
    factors_2021 = receipt["archived_factor_table"]["2021"]
    carried_ordinary = factors_2021["E00600 (ordinary_dividends)"]
    carried_qualified = factors_2021["E00650 (qualified_dividends)"]
    stored_qualified = table_regular["qualified_dividend_income"].to_numpy(np.float64)
    stored_non_qualified = table_regular["non_qualified_dividend_income"].to_numpy(
        np.float64
    )
    recovered_ordinary = np.rint(
        (stored_non_qualified + stored_qualified) / carried_ordinary
    )
    recovered_qualified = np.rint(stored_qualified / carried_qualified)
    source_ordinary = raw["E00600"].to_numpy(np.float64)
    source_qualified = raw["E00650"].to_numpy(np.float64)
    receipt["dividends"] = {
        "records": int(len(raw)),
        "recovered_ordinary_equals_E00600_share": float(
            (recovered_ordinary == source_ordinary).mean()
        ),
        "recovered_qualified_equals_E00650_share": float(
            (recovered_qualified == source_qualified).mean()
        ),
        "source_qualified_exceeds_ordinary_records": int(
            (source_qualified > source_ordinary).sum()
        ),
        "source_ordinary_equals_qualified_and_positive_records": int(
            ((source_ordinary == source_qualified) & (source_ordinary > 0)).sum()
        ),
        "stored_non_qualified_nonzero_where_source_difference_zero": int(
            ((stored_non_qualified != 0) & (source_ordinary == source_qualified)).sum()
        ),
    }

    # The synthetic records (aggregate-record replacements) have no source
    # record. Apply the same recovery to them and count where it would fail.
    table_synthetic = table.loc[table["tax_unit_id"] >= SYNTHETIC_START]
    syn_qualified = table_synthetic["qualified_dividend_income"].to_numpy(np.float64)
    syn_non_qualified = table_synthetic["non_qualified_dividend_income"].to_numpy(
        np.float64
    )
    syn_ordinary_raw = (syn_non_qualified + syn_qualified) / carried_ordinary
    syn_qualified_raw = syn_qualified / carried_qualified

    def off_grid(values: np.ndarray) -> np.ndarray:
        return np.abs(values - np.rint(values)) > 1e-6 + 1e-13 * np.abs(values)

    receipt["dividends"]["synthetic"] = {
        "records": int(len(table_synthetic)),
        "with_any_dividend": int(
            ((syn_qualified != 0) | (syn_non_qualified != 0)).sum()
        ),
        "stored_non_qualified_negative": int((syn_non_qualified < 0).sum()),
        "stored_qualified_negative": int((syn_qualified < 0).sum()),
        "recovered_ordinary_off_whole_dollars": int(off_grid(syn_ordinary_raw).sum()),
        "recovered_qualified_off_whole_dollars": int(off_grid(syn_qualified_raw).sum()),
        "recovered_ordinary_below_recovered_qualified": int(
            (np.rint(syn_ordinary_raw) < np.rint(syn_qualified_raw)).sum()
        ),
        "weight_total": float(table_synthetic["weight"].sum()),
        "weighted_negative_non_qualified": float(
            table_synthetic["weight"].to_numpy(np.float64)
            @ np.clip(syn_non_qualified, None, 0)
        ),
        "weighted_positive_non_qualified": float(
            table_synthetic["weight"].to_numpy(np.float64)
            @ np.clip(syn_non_qualified, 0, None)
        ),
        "weighted_qualified": float(
            table_synthetic["weight"].to_numpy(np.float64) @ syn_qualified
        ),
    }

    # 7. Donor weighted totals beside the SOI actuals of the same line. The
    # donor weights are S006/100 times return-count growth to 2021, so a file
    # at TY2021 levels should total near the TY2021 actuals.
    get_agg = uprate_ns["get_soi_aggregate"]
    weights = table["weight"].to_numpy(dtype=np.float64)
    totals = {}
    for label, column, variable in SOI_TOTAL_LINES:
        values = table[column].to_numpy(dtype=np.float64)
        totals[label] = {
            "donor_column": column,
            "soi_variable": variable,
            "donor_weighted_total": float(weights @ values),
            "soi_actual": {
                str(year): float(get_agg(variable, year, False))
                for year in (2015, 2021, 2022, 2023)
            },
        }
    # 8. Weighted positive and negative mass of every dollar column of the
    # donor table, all rows, for comparison with the frame's PUF arm.
    receipt["donor_weighted_mass"] = {
        "tax_units_weighted": float(weights.sum()),
        "columns": {
            column: {
                "positive": float(
                    weights @ np.clip(table[column].to_numpy(dtype=np.float64), 0, None)
                ),
                "negative": float(
                    weights @ np.clip(table[column].to_numpy(dtype=np.float64), None, 0)
                ),
            }
            for column in expected_columns
            if classes.get(column, {}).get("class")
            not in ("not a dollar amount", "design weight")
        },
    }
    receipt["donor_weighted_totals_vs_soi"] = {
        "donor_weight_total": float(weights.sum()),
        "soi_returns": {
            str(year): float(get_agg("count", year, True))
            for year in (2015, 2021, 2022, 2023)
        },
        "lines": totals,
    }

    args.out.write_text(json.dumps(receipt, indent=1, sort_keys=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
