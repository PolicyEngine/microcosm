"""List counties whose county SOI AGI row R2 would build empty (pre-check).

The ACS local release (R2, ``tools/build_us_acs_local_release.py``) builds a
county's IRS SOI rows in ``attach_surface`` from the stored entries of the
state SOI rows for the county's households (``split_state_rows``). The state
AGI row stores no zero, so when every household of a county has an AGI of
exactly 0 the county's AGI row is empty. ``attach_surface`` then refuses the
run, after the materialize stage's engine pass (microcosm#1630 review,
finding 1). Minimum county support fills only counties with no household, so
it does not help here.

This check runs on the staging H5 (R1's output) before R2, with no engine
pass and in minutes:

- It reads the household table, and only the AGI income columns of the
  person and tax-unit tables.
- The AGI income columns are the money inputs behind policyengine-us's
  ``irs_gross_income`` (its gross income sources), read from the installed
  engine's source without importing it, less the leaves that cannot make
  gross income nonzero on their own: Social Security (only its taxable share
  counts, which is 0 without other income) and deductions, expenses and
  contributions (``NOT_INCOME_LEAVES``).
- A household can carry a nonzero AGI when any of its persons or tax units
  has a nonzero, non-null value in one of those columns. R2 fills a null
  income cell with the engine default, 0.
- With ``--puma-ladder`` it applies R2's own minimum-support moves
  (``plan_minimum_support`` with ``--county-min-support``, default R2's 3)
  and takes the county list from the ladder, as R2 does.
- A county is **at risk** when it has households but none can carry a
  nonzero AGI. The check is conservative: a household whose only income is
  Social Security counts as zero, so a listed county may still pass.
  Return-count rows count every tax unit, so they are never empty.

Exit status 0: no county at risk. Exit status 2: at least one county at
risk; the JSON report (``--out``) lists them. Each needs a ruling before R2
(drop the county's AGI row with a receipt, or give it a declared SOI cell).

Usage::

    python tools/check_us_acs_county_agi_support.py \\
        --h5 <run>/acs_multispine_staging.h5 \\
        --county-column county_fips_2023 \\
        --puma-ladder build/us/us_puma_ladder_2020.npz \\
        --out <run>/county_agi_support.json
"""

# No ``from __future__ import annotations``: test loaders run this file
# without registering it in ``sys.modules``.
import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import us_acs_local_county_surface as county_surface  # noqa: E402

from microcosm.frame import read_frame_table  # noqa: E402

TOOL = "tools/check_us_acs_county_agi_support.py"
ISSUE = "microcosm#922"
#: The engine variable whose input leaves are the AGI income columns.
GROSS_INCOME_ROOT = "irs_gross_income"
#: Social Security enters gross income only through its taxable share, which
#: is 0 for a household without enough other income.
SOCIAL_SECURITY_LEAVES = frozenset(
    {
        "social_security_dependents",
        "social_security_disability",
        "social_security_retirement",
        "social_security_survivors",
    }
)
#: Money leaves of the ``irs_gross_income`` closure that are deductions,
#: expenses, contributions or tax-exempt amounts: none makes gross income
#: nonzero on its own.
NOT_INCOME_LEAVES = frozenset(
    {
        "alimony_expense",
        "domestic_production_ald",
        "early_withdrawal_penalty",
        "educator_expense",
        "health_insurance_premiums",
        "health_savings_account_ald",
        "health_savings_account_payroll_contributions",
        "pre_tax_health_insurance_premiums",
        "qualified_adoption_assistance_expense",
        "qualified_tuition_expenses",
        "roth_401k_contributions_desired",
        "roth_403b_contributions_desired",
        "roth_ira_contributions_desired",
        "self_employed_pension_contributions_desired",
        "student_loan_interest",
        "tax_exempt_interest_income",
        "traditional_401k_contributions_desired",
        "traditional_403b_contributions_desired",
        "traditional_ira_contributions_desired",
        "us_bonds_for_higher_ed",
    }
)
EXIT_AT_RISK = 2
_ROW_CHUNK = 250_000


def agi_income_columns(index: Any) -> dict[str, Any]:
    """The AGI income columns by entity, from the engine's static index.

    ``index`` is a ``PolicyEngineUSVariableMetadataIndex`` (or a stand-in
    with ``variable_dependency_closure``, ``variable_metadata`` and
    ``variable_unit``). Returns the person and tax-unit columns, the leaves
    left out and why, and the closure's engine version and digest.
    """

    closure = index.variable_dependency_closure(GROSS_INCOME_ROOT)
    columns: dict[str, list[str]] = defaultdict(list)
    excluded: dict[str, str] = {}
    for leaf in closure.input_leaves:
        metadata = index.variable_metadata(leaf)
        if metadata.dtype != "float" or index.variable_unit(leaf) != "currency-USD":
            continue
        if leaf in SOCIAL_SECURITY_LEAVES:
            excluded[leaf] = "social_security_taxable_share_only"
        elif leaf in NOT_INCOME_LEAVES:
            excluded[leaf] = "deduction_expense_or_contribution"
        else:
            columns[metadata.entity].append(leaf)
    unexpected = sorted(set(columns) - {"person", "tax_unit"})
    if unexpected:
        raise ValueError(
            f"{GROSS_INCOME_ROOT} money leaves on entities {unexpected}; the "
            "check reads person and tax-unit columns only."
        )
    return {
        "root": GROSS_INCOME_ROOT,
        "engine_version": closure.engine_version,
        "closure_sha256": closure.sha256,
        "person": sorted(columns.get("person", [])),
        "tax_unit": sorted(columns.get("tax_unit", [])),
        "excluded": dict(sorted(excluded.items())),
    }


def read_columns(path: Path, key: str, columns: Sequence[str]) -> pd.DataFrame:
    """Read only ``columns`` (those present) of one pandas H5 table.

    A table-format table is read with ``select``. A fixed-format table's
    numeric blocks are streamed in row chunks, so memory holds only the
    requested columns; a requested column in an object block is read through
    the shared frame-table reader instead.
    """

    with pd.HDFStore(path, mode="r") as store:
        if store.get_storer(key).is_table:
            names = set(store.select(key, start=0, stop=0).columns)
            present = list(dict.fromkeys(c for c in columns if c in names))
            return store.select(key, columns=present)
    wanted = list(dict.fromkeys(columns))
    result: dict[str, np.ndarray] = {}
    slow: list[str] = []
    with h5py.File(path, "r") as handle:
        group = handle[key]
        names = [_text(name) for name in group["axis0"][:]]
        rows = int(group["axis1"].shape[0])
        for block in range(int(group.attrs["nblocks"])):
            items = [_text(name) for name in group[f"block{block}_items"][:]]
            values = group[f"block{block}_values"]
            hits = [
                (position, item)
                for position, item in enumerate(items)
                if item in wanted
            ]
            if not hits:
                continue
            if values.dtype == object or values.ndim != 2:
                slow.extend(item for _position, item in hits)
                continue
            parts: dict[str, list[np.ndarray]] = {item: [] for _p, item in hits}
            for start in range(0, rows, _ROW_CHUNK):
                chunk = values[start : start + _ROW_CHUNK]
                for position, item in hits:
                    parts[item].append(np.asarray(chunk[:, position]))
            for item, chunks in parts.items():
                result[item] = np.concatenate(chunks) if chunks else np.empty(0)
        del names
    if slow:
        with pd.HDFStore(path, mode="r") as store:
            table = read_frame_table(store, key)
        for column in slow:
            result[column] = table[column].to_numpy()
        del table
    return pd.DataFrame({c: result[c] for c in wanted if c in result})


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def household_income_flags(
    households: pd.DataFrame,
    person: pd.DataFrame,
    tax_unit: pd.DataFrame | None,
    *,
    person_columns: Sequence[str],
    tax_unit_columns: Sequence[str] = (),
) -> np.ndarray:
    """Per household (in ``households`` order): can it carry a nonzero AGI?

    True when any of its persons, or any tax unit one of its persons belongs
    to, has a nonzero, non-null value in an AGI income column. A null cell
    counts as 0, the engine default R2 fills.
    """

    household_ids = households["household_id"].to_numpy()
    position = pd.Index(household_ids)
    if not position.is_unique:
        raise ValueError("Household ids must be unique.")
    person_household = person["person_household_id"].to_numpy()
    person_rows = position.get_indexer(person_household)
    if (person_rows < 0).any():
        raise ValueError("Persons name households the household table lacks.")
    person_income = np.zeros(len(person), dtype=bool)
    for column in person_columns:
        if column in person.columns:
            values = pd.to_numeric(person[column], errors="coerce").to_numpy(float)
            person_income |= np.nan_to_num(values, nan=0.0) != 0.0
    flags = np.zeros(len(households), dtype=bool)
    np.logical_or.at(flags, person_rows, person_income)
    present = [
        column
        for column in tax_unit_columns
        if tax_unit is not None and column in tax_unit.columns
    ]
    if present:
        unit_income = np.zeros(len(tax_unit), dtype=bool)
        for column in present:
            values = pd.to_numeric(tax_unit[column], errors="coerce").to_numpy(float)
            unit_income |= np.nan_to_num(values, nan=0.0) != 0.0
        units = pd.Index(tax_unit["tax_unit_id"].to_numpy())
        person_units = units.get_indexer(person["person_tax_unit_id"].to_numpy())
        if (person_units < 0).any():
            raise ValueError("Persons name tax units the tax-unit table lacks.")
        np.logical_or.at(flags, person_rows, unit_income[person_units])
    return flags


def county_agi_report(
    households: pd.DataFrame,
    income: np.ndarray,
    *,
    county_column: str,
    county_populations: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    """Per-county household counts and the counties at risk.

    ``county_populations`` (the ladder's, as R2 uses) sets the county list;
    without it the list is the counties the households name.
    """

    codes = county_surface.county_codes(households, county_column=county_column)
    support = county_surface.support_universe(households)
    totals, supporting, with_income = (defaultdict(int) for _ in range(3))
    for code, supports, earns in zip(
        codes.tolist(), support.tolist(), income.tolist(), strict=True
    ):
        totals[code] += 1
        supporting[code] += int(supports)
        with_income[code] += int(earns)
    if county_populations is None:
        counties = sorted(totals)
    else:
        counties = sorted(
            int(code) for code, value in county_populations.items() if float(value) > 0
        )
    at_risk, no_household, thin = [], [], []
    rows = {}
    for code in counties:
        record = {
            "households": totals.get(code, 0),
            "supporting_households": supporting.get(code, 0),
            "households_with_agi_income": with_income.get(code, 0),
        }
        rows[f"{code:05d}"] = record
        if record["households"] == 0:
            no_household.append(f"{code:05d}")
        elif record["households_with_agi_income"] == 0:
            at_risk.append(f"{code:05d}")
        elif (
            record["households_with_agi_income"]
            < county_surface.DEFAULT_COUNTY_MIN_SUPPORT
        ):
            thin.append(f"{code:05d}")
    outside = (
        sorted(set(totals) - set(counties)) if county_populations is not None else []
    )
    return {
        "county_column": county_column,
        "counties": len(counties),
        "households": int(len(households)),
        "households_with_agi_income": int(income.sum()),
        "at_risk": at_risk,
        "at_risk_detail": {code: rows[code] for code in at_risk},
        "no_household": no_household,
        "fewer_than_min_support_with_agi_income": thin,
        "counties_outside_ladder": [f"{code:05d}" for code in outside],
        "by_county": rows,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List counties whose county SOI AGI row R2 would build empty: "
            "counties with households but none that can carry a nonzero AGI."
        )
    )
    parser.add_argument(
        "--h5", required=True, type=Path, help="Staging H5 (R1 output)."
    )
    parser.add_argument(
        "--county-column",
        default=county_surface.COUNTY_COLUMN_2023,
        choices=county_surface.COUNTY_FIPS_COLUMNS,
        help="R2's --county-column (default county_fips_2023).",
    )
    parser.add_argument(
        "--puma-ladder",
        type=Path,
        help=(
            "R2's PUMA ladder. With it, the check takes the county list from "
            "the ladder and applies R2's minimum-support moves first."
        ),
    )
    parser.add_argument(
        "--county-min-support",
        type=int,
        default=county_surface.DEFAULT_COUNTY_MIN_SUPPORT,
        help=(
            "R2's --county-min-support (default "
            f"{county_surface.DEFAULT_COUNTY_MIN_SUPPORT}); used only with --puma-ladder."
        ),
    )
    parser.add_argument("--out", type=Path, help="JSON report path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.time()
    from microcosm.frame.adapters.policyengine_us import (
        PolicyEngineUSVariableMetadataIndex,
    )

    sources = agi_income_columns(PolicyEngineUSVariableMetadataIndex())
    with pd.HDFStore(args.h5, mode="r") as store:
        households = read_frame_table(store, "household")
    person = read_columns(
        args.h5,
        "person",
        ["person_household_id", "person_tax_unit_id", *sources["person"]],
    )
    tax_unit = (
        read_columns(args.h5, "tax_unit", ["tax_unit_id", *sources["tax_unit"]])
        if sources["tax_unit"]
        else None
    )
    present_person = [
        column for column in sources["person"] if column in person.columns
    ]
    present_unit = [
        column
        for column in sources["tax_unit"]
        if tax_unit is not None and column in tax_unit.columns
    ]
    income = household_income_flags(
        households,
        person,
        tax_unit,
        person_columns=present_person,
        tax_unit_columns=present_unit,
    )
    populations = None
    support = None
    if args.puma_ladder is not None:
        from microcosm.build.us_runtime.puma_ladder import load_us_puma_ladder

        ladder = load_us_puma_ladder(args.puma_ladder)
        populations = county_surface.with_county_populations(
            ladder, {}, {"county": True}, county_column=args.county_column
        )["county"]
        needed = county_surface.unsupported_counties(
            households, populations, county_column=args.county_column
        )
        if args.county_min_support > 0 and needed:
            overlaps = county_surface.county_puma_overlaps(
                args.puma_ladder, args.county_column
            )
            moves, support = county_surface.plan_minimum_support(
                households,
                populations,
                overlaps,
                county_column=args.county_column,
                per_county=args.county_min_support,
            )
            households = county_surface.apply_support_moves(
                households,
                moves,
                county_column=args.county_column,
                columns=support["rewritten_columns"],
            )
    report = county_agi_report(
        households,
        income,
        county_column=args.county_column,
        county_populations=populations,
    )
    report.update(
        {
            "tool": TOOL,
            "issue": ISSUE,
            "h5": str(args.h5),
            "agi_income_columns": {
                **sources,
                "person_present": present_person,
                "tax_unit_present": present_unit,
            },
            "minimum_support": (
                None
                if support is None
                else {
                    "households_per_county": args.county_min_support,
                    "moves": len(support.get("moves", [])),
                    "moves_sha256": support.get("moves_sha256"),
                    "moved_counties": support.get("moved_counties", []),
                }
            ),
            "seconds": round(time.time() - started, 1),
        }
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    summary = {
        key: report[key]
        for key in (
            "county_column",
            "counties",
            "households",
            "at_risk",
            "at_risk_detail",
            "no_household",
            "fewer_than_min_support_with_agi_income",
            "seconds",
        )
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return EXIT_AT_RISK if report["at_risk"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
