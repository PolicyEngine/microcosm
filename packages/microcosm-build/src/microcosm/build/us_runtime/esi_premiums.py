"""MEPS-IC employer premiums for ESI policyholders.

Microcosm #454. PolicyEngine-US 2.2.1 reads a person input that no Microcosm
stage produced: ``employer_sponsored_insurance_premiums``, "annual
employer-paid health insurance premiums", one of
``gov.household.cbo_market_income_additions`` (CBO household market income).

The retired eCPS derivation (archived 42ed5d45 ``datasets/cps/cps.py``
L197-271) read the CPS ASEC current employment-based coverage fields, which the
pinned H5 inputs omit; :mod:`.asec_census_person_columns` now restores them,
with employment status and employer size, from the pinned Census person files.

Employer premium
----------------
Both national accounts that publish an employer-contribution total (CMS NHE
Table 24 and BEA NIPA Table 7.8 line 17) count contributions for active
**and** retired employees, and both build the non-federal part from MEPS-IC.
So the stage prices every **current ESI policyholder** (``NOW_OWNGRP`` 1,
employed or not: the accounts' universe), scales that universe to the account
total, and writes the result only for the **employed** policyholders
(``PEMLR`` 1-2) whose current job has an employer (``PEIO1COW`` 1-6). The
column is the compensation of a current job: dependents carry nothing (the
policyholder carries the premium), and retired, unemployed and other
non-employed policyholders, self-employed-unincorporated workers and workers
without pay carry nothing either. Their scaled share stays out of the column
rather than being loaded onto workers.

Each policyholder's raw employer share comes from the MEPS-IC cell of their
coverage tier (``NOW_GRPFTYP2``: family, self plus one, self-only), employer
sector (``PEIO1COW``) and, for private employers, State and firm size
(``NOEMP`` under 50 / 50 or more):

* employer paid all of the premium (``NOW_HIPAID`` 1): the cell's average
  total premium;
* employer paid some (2): the average total premium minus the contribution of
  an employee who pays one. MEPS-IC averages the employee contribution over
  every enrollee, including those who pay nothing, so the stage divides it by
  one minus the published share of enrollees whose coverage required no
  contribution (Tables II.C/D/E.4.a, national by firm size; most State cells
  of those tables are flagged unreliable). A cohort with MEPS-IC's own
  all/some mix then reproduces the cell's published employer mean;
* employer paid none (3): zero. MEPS-IC publishes no share of enrollees who
  pay the whole premium, so they are treated as outside its averages.

Private-sector cells are MEPS-IC 2025 Series II (State by firm size).
Government cells are MEPS-IC 2024 Series III by census division (the latest
year published), aged to 2025 by the private-sector national 2025/2024 ratio of
the same tier and measure: State government employees take the
State-government column, local and federal employees the all-governments
column (MEPS-IC does not survey the federal government, so that is a
stand-in). Series III publishes no no-contribution share, so government cells
take the private 50-or-more share (a stand-in). A policyholder with no
employer class (not employed and never asked, self-employed unincorporated or
without pay) takes the State's all-sizes private cell. No MEPS-IC table prices
retiree coverage, so non-employed policyholders are priced as active
employees; if retiree plans cost less, the employed column is understated.
AHRQ suppresses nine small-firm employee-contribution cells; each takes the
national small-firm cell times the State's total-to-national ratio.

One factor scales every policyholder's raw share so the weighted total equals
CMS NHE Table 24's employer contribution to private health insurance premiums
for the build year, the anchor microcosm#454 chose. Its methodology paper says
the estimate covers "active employees, continuation of health coverage
(COBRA), and retirees" of private and State and local sponsors (from MEPS-IC)
and federal "employers, employees and retirees" (from OPM). BEA NIPA Table 7.8
line 17 (series B4923C) is a second estimate of the same concept, 6.7% lower
for 2024: BEA's State Personal Income methodology (December 2025, paragraphs
3.18-3.19) says its MEPS source "covers both health insurance purchased by
employers for their active and retired employees" and its federal estimate
covers "active and retired federal civilian employees". It is recorded as a
cross-check, not gated.

Not produced here
-----------------
``pre_tax_health_insurance_premiums``, the employee share paid by pre-tax
payroll deduction, has no producer yet. PolicyEngine-US treats it as disjoint
from the other premium inputs (PolicyEngine/policyengine-us#10046), so
producing it means moving those dollars out of the reported premium the base
carries, on an engine that counts the pre-tax input where the reported premium
was counted. That is a separate change.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_manifest import (
    SourceOperationSpec,
    SourceStageSpec,
    load_source_manifest,
)
from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
    run_source_stage,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "US_ESI_EMPLOYER_PREMIUM_COLUMN",
    "US_ESI_PREMIUMS_NONCONSTANT_PERSON_COLUMNS",
    "US_ESI_PREMIUMS_OUTPUT_COLUMNS",
    "US_ESI_PREMIUMS_REQUIRED_SOURCE_COLUMNS",
    "US_ESI_PREMIUMS_STAGE_NAME",
    "derive_us_employer_esi_premiums_from_manifest",
    "load_meps_ic_esi_premium_cells",
    "meps_ic_private_active_employer_totals",
    "refuse_unassigned_us_esi_premiums",
    "us_esi_premiums_anchor_gate",
    "us_esi_premiums_signal_gate",
    "us_esi_premiums_stage_spec",
    "us_esi_premiums_summary",
    "with_us_esi_premium_inputs",
]

US_ESI_PREMIUMS_STAGE_NAME = "meps_esi_premiums"
US_ESI_EMPLOYER_PREMIUM_COLUMN = "employer_sponsored_insurance_premiums"
US_ESI_PREMIUMS_OUTPUT_COLUMNS: tuple[str, ...] = (US_ESI_EMPLOYER_PREMIUM_COLUMN,)
US_ESI_PREMIUMS_NONCONSTANT_PERSON_COLUMNS = US_ESI_PREMIUMS_OUTPUT_COLUMNS

#: Raw CPS ASEC person columns the stage reads (the six NOW_*/PEMLR/NOEMP
#: columns through :mod:`.asec_census_person_columns`), plus the household
#: State (``state_fips``, from GESTFIPS), joined onto the person table by
#: ``person_household_id`` when it lives on the household table.
US_ESI_PREMIUMS_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "NOW_OWNGRP",
    "NOW_HIPAID",
    "NOW_GRPFTYP",
    "NOW_GRPFTYP2",
    "PEMLR",
    "NOEMP",
    "PEIO1COW",
    "state_fips",
)

#: Census CPS ASEC person codes (survey years 2023-2026 codebooks, matching the
#: pinned person members on 2026-10-10).
_CODE_DOMAINS: Mapping[str, frozenset[int]] = MappingProxyType(
    {
        "NOW_OWNGRP": frozenset({0, 1, 2}),
        "NOW_HIPAID": frozenset({0, 1, 2, 3}),
        "NOW_GRPFTYP": frozenset({0, 1, 2}),
        "NOW_GRPFTYP2": frozenset({0, 1, 2, 3}),
        "PEMLR": frozenset(range(8)),
        "NOEMP": frozenset(range(7)),
        "PEIO1COW": frozenset(range(9)),
    }
)
_POLICYHOLDER = 1
_EMPLOYED = (1, 2)
_PAYS_ALL, _PAYS_SOME, _PAYS_NONE = 1, 2, 3
#: NOW_GRPFTYP2 -> MEPS-IC tier; NOW_GRPFTYP must agree (1 family, 2 self-only).
_TIER_BY_CODE: Mapping[int, str] = MappingProxyType(
    {1: "family", 2: "employee_plus_one", 3: "single"}
)
_GRPFTYP_FOR_GRPFTYP2: Mapping[int, int] = MappingProxyType({1: 1, 2: 1, 3: 2})
#: NOEMP ("Work experience, persons who work for employer"): 1 under 10 and
#: 2 10-49 -> under 50; 3-6 -> 50 or more; 0 (not in universe) -> the State's
#: all-sizes cell.
_SIZE_BY_NOEMP: Mapping[int, str] = MappingProxyType(
    {
        0: "total",
        1: "lt50",
        2: "lt50",
        3: "50plus",
        4: "50plus",
        5: "50plus",
        6: "50plus",
    }
)
#: PEIO1COW: 1 federal, 2 state, 3 local, 4 private for profit, 5 private
#: nonprofit, 6 self-employed incorporated, 7 self-employed unincorporated,
#: 8 without pay.
_FEDERAL, _STATE_GOVERNMENT, _LOCAL_GOVERNMENT = 1, 2, 3
_GOVERNMENT_EMPLOYER_CODES = (_FEDERAL, _STATE_GOVERNMENT, _LOCAL_GOVERNMENT)
_PRIVATE_EMPLOYER_CODES = (4, 5, 6)
_EMPLOYER_SECTOR_CODES = (_FEDERAL, _STATE_GOVERNMENT, _LOCAL_GOVERNMENT, 4, 5, 6)

_CELLS_RESOURCE = "meps_ic_esi_premium_cells.json"
_CELLS_SHA256 = "29a5502abbd921e70007d6bcec8ed1ae065b309cacfa80f49f9bc4dd9a28ec93"
_PERSON_WEIGHT_COLUMN = "person_weight"

_BEA_SECTION7_URL = (
    "https://apps.bea.gov/national/Release/XLS/Survey/Section7All_xls.xlsx"
)
_NHE_TABLES_URL = "https://www.cms.gov/files/zip/nhe-tables.zip"

#: CMS NHE Table 24 (2024 National Health Expenditure Accounts, data through
#: CY2024), row "Employer Contribution to Private Health Insurance Premiums":
#: private, federal and State and local employers, for active employees, COBRA
#: enrollees and retirees. The anchor microcosm#454 chose; banked in the
#: pinned feed as cms_nhe.cy{2023,2024}.esi_employer_contribution_premiums.
#: ``sha256`` is the digest of the zip archive at ``source`` (520,391 bytes),
#: not of the Table 24 workbook inside it.
EMPLOYER_PREMIUM_ANCHOR: Mapping[str, Any] = MappingProxyType(
    {
        "source": _NHE_TABLES_URL,
        "sha256": "a09ef6d3e84e25d745047a47b6b08a0d96b303085b4c725b67ce67a0eb0c4420",
        "table": "NHE Table 24 Employer-Sponsored Private Health Insurance",
        "row": "Employer Contribution to Private Health Insurance Premiums",
        "coverage": (
            "active employees, COBRA enrollees and retirees of private, "
            "federal and State and local employers"
        ),
        # Integer dollars: the manifest round-trips through YAML.
        "values": {"2023": 975_700_000_000, "2024": 1_047_000_000_000},
    }
)
#: BEA NIPA Table 7.8 line 17 (B4923C), millions of dollars converted to
#: dollars, from the annual update published 2026-09-30 (file created
#: 2026-09-28). That update revised 2024 from the $1,002.9B the retired
#: us-data loss matrix pinned. A second estimate of the same concept; recorded,
#: not gated.
EMPLOYER_PREMIUM_CROSS_CHECK: Mapping[str, Any] = MappingProxyType(
    {
        "source": _BEA_SECTION7_URL,
        "sha256": "de1c34e37da9b8b8d765efc0611cc653102373fe522fe9218c7689067b29b7fd",
        "table": "NIPA Table 7.8 Supplements to Wages and Salaries by Type",
        "line": 17,
        "series": "B4923C",
        "label": "Private group health insurance",
        "published": "2026-09-30",
        "values": {
            "2023": 923_195_000_000,
            "2024": 977_034_000_000,
            "2025": 1_027_929_000_000,
        },
        "relation": (
            "second estimate of the same concept: employer contributions for "
            "active and retired employees (BEA State Personal Income "
            "methodology, December 2025, paragraphs 3.18-3.19)"
        ),
    }
)
#: Tolerance of the calibrated release aggregate around the anchor, over the
#: anchor's universe (every policyholder). The stage pins the pre-calibration
#: aggregate exactly, so the band is the room a release gives calibration and
#: sparse selection to move it. It is a chosen release criterion, not a
#: measured error: no release has calibrated this column yet.
ANCHOR_RELATIVE_TOLERANCE = 0.05
#: Weighted-raw employer share per positive employed holder, before scaling.
#: MEPS-IC 2025 employer shares run $7.2k (single) to $19.0k (family) and the
#: pinned pools measure $12.0-12.1k. The band catches a gross unit error in the
#: cell table (monthly for annual, cents for dollars), which the scale factor
#: would otherwise hide; it does not catch a level error inside the band.
_RAW_MEAN_BAND = (6_000.0, 20_000.0)
#: Weighted share of all persons with a positive employer premium. Measured
#: on both pinned pools: 22.7%.
_POSITIVE_SHARE_BAND = (0.15, 0.32)
#: Employed policyholders' share of the anchor-universe employer total.
#: Measured on the pinned pools: 88.4% and 88.2%. A release whose calibration
#: moves the split far from that has reweighted workers against retirees.
_EMPLOYED_SHARE_BAND = (0.80, 0.95)

_EMPLOYER_PREMIUM_PARAMETERS: Mapping[str, Any] = MappingProxyType(
    {
        "cells_resource": f"microcosm.build.us_runtime.data/{_CELLS_RESOURCE}",
        "cells_sha256": _CELLS_SHA256,
        "anchor_universe": "NOW_OWNGRP == 1",
        "universe": (
            "NOW_OWNGRP == 1 and PEMLR in (1, 2) and PEIO1COW in (1, 2, 3, 4, 5, 6)"
        ),
        "tier_column": "NOW_GRPFTYP2",
        "size_column": "NOEMP",
        "sector_column": "PEIO1COW",
        "payment_column": "NOW_HIPAID",
        "payment_rule": (
            "1 (employer paid all): average total premium; 2 (some): average "
            "total premium minus average employee contribution / (1 - "
            "no-contribution share); 3 (none): 0"
        ),
        "no_contribution_share": (
            "MEPS-IC 2025 Tables II.C/D/E.4.a, United States row by firm size; "
            "government cells take the 50-or-more share"
        ),
        "other_policyholder_cell": (
            "PEIO1COW outside 1-6: the State's all-sizes private cell"
        ),
        "suppressed_cell_fallback": (
            "national firm-size cell times the State total-to-national ratio"
        ),
        "government_cell_aging": (
            "MEPS-IC 2024 Series III x private-sector national 2025/2024 "
            "ratio of the same tier and measure"
        ),
        "scaling": (
            "one factor, anchor / weighted raw share of the anchor universe; "
            "written to the universe only"
        ),
        "anchor": {
            "source": EMPLOYER_PREMIUM_ANCHOR["source"],
            "sha256": EMPLOYER_PREMIUM_ANCHOR["sha256"],
            "table": EMPLOYER_PREMIUM_ANCHOR["table"],
            "row": EMPLOYER_PREMIUM_ANCHOR["row"],
            "values": dict(EMPLOYER_PREMIUM_ANCHOR["values"]),
        },
    }
)


def us_esi_premiums_stage_spec() -> SourceStageSpec:
    """Load the ``meps_esi_premiums`` stage and pin it to this contract."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    spec = manifest.stage_map().get(US_ESI_PREMIUMS_STAGE_NAME)
    if spec is None:
        raise ValueError(
            f"US source manifest declares no {US_ESI_PREMIUMS_STAGE_NAME!r} stage."
        )
    if spec.grain != "person":
        raise ValueError("US ESI premium stage must have person grain.")
    if tuple(spec.outputs) != US_ESI_PREMIUMS_OUTPUT_COLUMNS:
        raise ValueError("US ESI premium manifest outputs drifted from the runtime.")
    expected = (
        ("read_table", {"table": "person", "weight": _PERSON_WEIGHT_COLUMN}),
        (
            "derive_employer_sponsored_insurance_premiums",
            _plain(_EMPLOYER_PREMIUM_PARAMETERS),
        ),
    )
    actual = tuple(
        (operation.kind, dict(operation.parameters)) for operation in spec.operations
    )
    if actual != expected:
        raise ValueError(
            "US ESI premium operations drifted from the reviewed MEPS-IC cell "
            "assignment contract."
        )
    return spec


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def load_meps_ic_esi_premium_cells() -> Mapping[str, Any]:
    """The packaged MEPS-IC cell table, verified against its pinned digest."""

    raw = (
        files("microcosm.build.us_runtime.data").joinpath(_CELLS_RESOURCE).read_bytes()
    )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _CELLS_SHA256:
        raise SourceRuntimeError(
            f"{_CELLS_RESOURCE} digest {digest} is not the reviewed {_CELLS_SHA256}; "
            "regenerate it with tools/build_us_meps_ic_esi_cells.py and re-pin."
        )
    return json.loads(raw)


@dataclass(frozen=True)
class _EsiCodes:
    owner: np.ndarray
    hipaid: np.ndarray
    tier: np.ndarray
    employed: np.ndarray
    noemp: np.ndarray
    sector: np.ndarray
    state_fips: np.ndarray

    @property
    def policyholder(self) -> np.ndarray:
        """Every current ESI policyholder: the universe the anchor measures."""

        return self.owner == _POLICYHOLDER

    @property
    def universe(self) -> np.ndarray:
        """Employed policyholders with an employer: the column's universe."""

        return (
            self.policyholder
            & self.employed
            & np.isin(self.sector, _EMPLOYER_SECTOR_CODES)
        )


def _integer_codes(person: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(person[column], errors="coerce")
    numeric = values.to_numpy(dtype=np.float64, na_value=np.nan)
    bad = ~np.isfinite(numeric) | (numeric != np.round(numeric))
    if bad.any():
        raise SourceRuntimeError(
            f"US ESI premium stage: {column} has {int(bad.sum())} missing or "
            "non-integer row(s); the Census column was not restored for every "
            "vintage (asec_census_person_columns) and is never defaulted."
        )
    codes = numeric.astype(np.int64)
    domain = _CODE_DOMAINS.get(column)
    if domain is not None:
        outside = ~np.isin(codes, sorted(domain))
        if outside.any():
            raise SourceRuntimeError(
                f"US ESI premium stage: {column} has {int(outside.sum())} code(s) "
                f"outside the Census codebook {sorted(domain)}, e.g. "
                f"{sorted(set(codes[outside].tolist()))[:5]}."
            )
    return codes


def _esi_person_codes(person: pd.DataFrame) -> _EsiCodes:
    """Validate the raw ASEC coverage, employment and employer fields.

    Reads NOW_OWNGRP, NOW_HIPAID, NOW_GRPFTYP, NOW_GRPFTYP2, PEMLR, NOEMP and
    PEIO1COW, and refuses the Census universe relations the pinned person
    members satisfy exactly: a policyholder reports a payment status and a
    tier, everyone else reports neither, and NOW_GRPFTYP (family/self-only)
    agrees with NOW_GRPFTYP2.
    """

    missing = [
        column
        for column in US_ESI_PREMIUMS_REQUIRED_SOURCE_COLUMNS
        if column not in person.columns
    ]
    if missing:
        raise SourceRuntimeError(
            "US ESI premium stage is source-unavailable: the person table lacks "
            f"{missing}. Restore them from the pinned Census person files "
            "(asec_census_person_columns); they are never proxied or defaulted."
        )
    owner = _integer_codes(person, "NOW_OWNGRP")
    hipaid = _integer_codes(person, "NOW_HIPAID")
    grpftyp = _integer_codes(person, "NOW_GRPFTYP")
    tier = _integer_codes(person, "NOW_GRPFTYP2")
    pemlr = _integer_codes(person, "PEMLR")
    noemp = _integer_codes(person, "NOEMP")
    sector = _integer_codes(person, "PEIO1COW")
    state_fips = _integer_codes(person, "state_fips")
    holder = owner == _POLICYHOLDER
    violations = {
        "policyholder_without_payment_status": int(
            (holder & ~np.isin(hipaid, [_PAYS_ALL, _PAYS_SOME, _PAYS_NONE])).sum()
        ),
        "policyholder_without_tier": int((holder & ~np.isin(tier, [1, 2, 3])).sum()),
        "non_policyholder_with_payment_status": int((~holder & (hipaid != 0)).sum()),
        "non_policyholder_with_tier": int((~holder & (tier != 0)).sum()),
        "grpftyp_disagrees_with_grpftyp2": int(
            (
                holder
                & (
                    grpftyp
                    != pd.Series(tier)
                    .map(dict(_GRPFTYP_FOR_GRPFTYP2))
                    .fillna(-1)
                    .to_numpy()
                )
            ).sum()
        ),
    }
    if any(violations.values()):
        rendered = ", ".join(
            f"{key}={value}" for key, value in violations.items() if value
        )
        raise SourceRuntimeError(
            f"US ESI premium stage refused the ASEC coverage fields: {rendered}."
        )
    return _EsiCodes(
        owner=owner,
        hipaid=hipaid,
        tier=tier,
        employed=np.isin(pemlr, _EMPLOYED),
        noemp=noemp,
        sector=sector,
        state_fips=state_fips,
    )


def _private_cell(
    cells: Mapping[str, Any], tier: str, measure: str, state: str, size: str
) -> float:
    rows = cells["private_state_2025"][tier][measure]["rows"]
    if state not in rows:
        raise SourceRuntimeError(
            f"US ESI premium stage: no MEPS-IC row for State {state}."
        )
    value = rows[state][size]
    if value is None:
        # AHRQ suppressed the cell: national firm-size cell scaled by the
        # State's all-sizes level relative to the nation.
        national = rows["US"]
        value = national[size] * rows[state]["total"] / national["total"]
    return float(value)


def _government_cell(
    cells: Mapping[str, Any], tier: str, measure: str, division: str, column: str
) -> float:
    value = cells["public_division_2024"][tier][measure]["rows"][division][column]
    current = cells["private_state_2025"][tier][measure]["rows"]["US"]["total"]
    prior = cells["private_national_2024"][tier][measure]["total"]
    return float(value) * float(current) / float(prior)


def _no_contribution_share(cells: Mapping[str, Any], tier: str, size: str) -> float:
    return float(cells["no_contribution_share_2025"][tier][size]) / 100.0


def _cell_values(
    codes: _EsiCodes, cells: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each policyholder's cell: premium, employee contribution, no-contribution share.

    NaN for everyone who is not a policyholder.
    """

    premium = np.full(len(codes.owner), np.nan)
    contribution = np.full(len(codes.owner), np.nan)
    no_contribution = np.full(len(codes.owner), np.nan)
    divisions = cells["state_census_division"]
    keys = pd.DataFrame(
        {
            "tier": codes.tier,
            "noemp": codes.noemp,
            "sector": codes.sector,
            "state": codes.state_fips,
        }
    )
    for (tier_code, noemp, sector, state), index in (
        keys[codes.policyholder]
        .groupby(["tier", "noemp", "sector", "state"])
        .groups.items()
    ):
        tier = _TIER_BY_CODE[int(tier_code)]
        state_key = f"{int(state):02d}"
        if state_key not in divisions:
            raise SourceRuntimeError(
                f"US ESI premium stage: State FIPS {int(state)} has no MEPS-IC cell."
            )
        if int(sector) in _GOVERNMENT_EMPLOYER_CODES:
            column = (
                "state" if int(sector) == _STATE_GOVERNMENT else "all_state_and_local"
            )
            values = [
                _government_cell(cells, tier, measure, divisions[state_key], column)
                for measure in ("premium", "employee_contribution")
            ]
            # Series III publishes no no-contribution share.
            size = "50plus"
        else:
            # Private employers by firm size; a policyholder with no employer
            # class takes the State's all-sizes private cell.
            size = (
                _SIZE_BY_NOEMP[int(noemp)]
                if int(sector) in _PRIVATE_EMPLOYER_CODES
                else "total"
            )
            values = [
                _private_cell(cells, tier, measure, state_key, size)
                for measure in ("premium", "employee_contribution")
            ]
        rows = np.asarray(index, dtype=np.int64)
        premium[rows], contribution[rows] = values
        no_contribution[rows] = _no_contribution_share(cells, tier, size)
    return premium, contribution, no_contribution


def _raw_employer_share(
    codes: _EsiCodes,
    premium: np.ndarray,
    contribution: np.ndarray,
    no_contribution: np.ndarray,
) -> np.ndarray:
    """Unscaled employer share of every policyholder (zero for everyone else)."""

    raw = np.zeros(len(codes.owner), dtype=np.float64)
    holder = codes.policyholder
    pays_all = holder & (codes.hipaid == _PAYS_ALL)
    pays_some = holder & (codes.hipaid == _PAYS_SOME)
    raw[pays_all] = premium[pays_all]
    # The published contribution averages over enrollees who pay nothing;
    # dividing by the share who pay gives the contribution of those who do.
    raw[pays_some] = np.clip(
        premium[pays_some]
        - contribution[pays_some] / (1.0 - no_contribution[pays_some]),
        0.0,
        None,
    )
    return raw


def _person_raw_shares(
    person: pd.DataFrame,
) -> tuple[_EsiCodes, np.ndarray, np.ndarray, np.ndarray]:
    """Codes, raw employer share, cell premium and cell employee contribution."""

    codes = _esi_person_codes(person)
    premium, contribution, no_contribution = _cell_values(
        codes, load_meps_ic_esi_premium_cells()
    )
    raw = _raw_employer_share(codes, premium, contribution, no_contribution)
    return codes, raw, premium, contribution


def meps_ic_private_active_employer_totals(
    cells: Mapping[str, Any],
) -> dict[str, float]:
    """MEPS-IC's own private-sector employer total for enrolled employees.

    Enrolled employees (employees x the share in establishments that offer
    health insurance x the share enrolled there) times each tier's share of
    enrollees and its average premium less average employee contribution,
    from the United States rows of the pinned Series II tables. It covers
    active private-sector employees only, so it checks the private part of
    the column against the survey the cells come from.
    """

    national = {
        "2024": lambda tier, measure: cells["private_national_2024"][tier][measure][
            "total"
        ],
        "2025": lambda tier, measure: cells["private_state_2025"][tier][measure][
            "rows"
        ]["US"]["total"],
    }
    totals: dict[str, float] = {}
    for year, rows in cells["private_enrollment_national"].items():
        enrolled = (
            float(rows["employees"]["total"])
            * float(rows["offer_percent"]["total"])
            / 100.0
            * float(rows["enrolled_percent"]["total"])
            / 100.0
        )
        totals[year] = sum(
            enrolled
            * float(rows["tier_share_percent"][tier]["total"])
            / 100.0
            * (
                float(national[year](tier, "premium"))
                - float(national[year](tier, "employee_contribution"))
            )
            for tier in _TIER_BY_CODE.values()
        )
    return totals


def _anchor(year: int) -> float:
    values = EMPLOYER_PREMIUM_ANCHOR["values"]
    if str(year) not in values:
        raise SourceRuntimeError(
            f"US ESI premium stage has no NHE Table 24 anchor for {year}; pinned "
            f"years: {sorted(values)}."
        )
    return float(values[str(year)])


def _check_parameters(
    operation: SourceOperationSpec, expected: Mapping[str, Any]
) -> None:
    if dict(operation.parameters) != _plain(expected):
        raise SourceRuntimeError(
            f"US ESI premium operation {operation.kind!r} parameters drifted from "
            "the reviewed contract."
        )


def _person_weights(frame: pd.DataFrame) -> np.ndarray:
    if _PERSON_WEIGHT_COLUMN not in frame.columns:
        raise SourceRuntimeError("US ESI premium stage requires person weights.")
    weights = frame[_PERSON_WEIGHT_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise SourceRuntimeError(
            "US ESI premium stage requires finite nonnegative weights."
        )
    return weights


def derive_us_employer_esi_premiums_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Assign MEPS-IC employer shares and scale them to the NHE anchor.

    The factor is set over every policyholder, the anchor's universe; the
    column carries the employed policyholders' part.
    """

    if operation.kind != "derive_employer_sponsored_insurance_premiums":
        raise SourceRuntimeError(
            f"Unexpected ESI premium operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError("ESI premium assignment requires the person table.")
    _check_parameters(operation, _EMPLOYER_PREMIUM_PARAMETERS)
    codes, raw, _premium, _contribution = _person_raw_shares(frame)
    weights = _person_weights(frame)
    universe = codes.universe
    if not float(weights[universe] @ raw[universe]) > 0:
        raise SourceRuntimeError(
            "US ESI premium stage found no weighted employer-paid premium mass to "
            "scale: no employed policyholder with an employer share."
        )
    scale = _anchor(context.config.target_year) / float(weights @ raw)
    result = frame.copy(deep=True)
    result[US_ESI_EMPLOYER_PREMIUM_COLUMN] = np.where(universe, raw * scale, 0.0)
    return result


def _stable_person_keys(frame: pd.DataFrame) -> pd.Series:
    if {"source_year", "source_household_id", "source_person_id"} <= set(frame.columns):
        return (
            frame["source_year"].astype(str)
            + ":"
            + frame["source_household_id"].astype(str)
            + ":"
            + frame["source_person_id"].astype(str)
        )
    return frame["person_id"].astype(str)


def _person_with_state(frame: Frame) -> pd.DataFrame:
    """The person table with the household ``state_fips`` joined on."""

    person = frame.table("person").copy(deep=True)
    if "state_fips" in person.columns:
        return person
    household = frame.table("household")
    if "state_fips" not in household.columns:
        return person
    state = household.set_index("household_id")["state_fips"]
    if not state.index.is_unique:
        raise SourceRuntimeError("US ESI premium stage: household ids repeat.")
    joined = person["person_household_id"].map(state)
    if joined.isna().any():
        raise SourceRuntimeError(
            f"US ESI premium stage: {int(joined.isna().sum())} person(s) have no "
            "household state_fips."
        )
    person["state_fips"] = joined.to_numpy()
    return person


def _outputs_carry_signal(person: pd.DataFrame) -> bool:
    return all(
        column in person.columns and person[column].dropna().nunique() > 1
        for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS
    )


def with_us_esi_premium_inputs(frame: Frame, *, seed: int, time_period: int) -> Frame:
    """Run the ``meps_esi_premiums`` stage over a US frame.

    A frame that already carries the output with signal and passes the signal
    gate is returned unchanged (idempotent). Otherwise the person table must
    still carry the raw ASEC columns; the stage never defaults them. The
    assignment is deterministic: ``seed`` only fills the runtime config that
    every source stage takes.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("US ESI premium inputs require the US schema.")
    person = frame.table("person")
    if _outputs_carry_signal(person):
        gate = us_esi_premiums_signal_gate(frame)
        if not gate.passed:
            raise SourceRuntimeError(
                "US ESI premium preexisting inputs fail the signal gate: "
                + "; ".join(gate.failures)
            )
        return frame
    stage_person = _person_with_state(frame)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    output = run_source_stage(
        us_esi_premiums_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_employer_sponsored_insurance_premiums": (
                derive_us_employer_esi_premiums_from_manifest
            ),
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS:
        values = aligned[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"US ESI premium stage output does not cover {column!r}.")
        tables["person"][column] = values
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def refuse_unassigned_us_esi_premiums(frame: Frame, *, consumer: str) -> None:
    """Refuse a frame whose ESI employer premium input is null on some rows.

    A lineage that pools rows the stage never ran on (ACS-spine rows beside an
    ASEC donor that carries the column) leaves it null there. Filling those
    with the engine default would ship a column that is zero on part of the
    population yet looks populated, and no anchor gate runs on that lineage, so
    the consumer refuses instead.
    """

    person = frame.table("person")
    gaps = {
        column: int(person[column].isna().sum())
        for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS
        if column in person.columns and person[column].isna().any()
    }
    if gaps:
        raise SourceRuntimeError(
            f"{consumer}: ESI premium input(s) are null on some rows {gaps}. The "
            "meps_esi_premiums stage (microcosm#454) ran on the ASEC rows only; "
            "its output is never filled with an engine default. Assign or "
            "transfer it for every spine before building this lineage."
        )


def _weighted(
    weights: np.ndarray, values: np.ndarray, mask: np.ndarray | None = None
) -> float:
    if mask is None:
        return float(weights @ values)
    return float(weights[mask] @ values[mask])


def _clone_disagreements(person: pd.DataFrame) -> int:
    if not {"source_year", "source_household_id", "source_person_id"} <= set(
        person.columns
    ):
        return 0
    keys = _stable_person_keys(person)
    work = person[list(US_ESI_PREMIUMS_OUTPUT_COLUMNS)].copy()
    work["_key"] = keys.to_numpy()
    return int(
        work.groupby("_key", sort=False)[list(US_ESI_PREMIUMS_OUTPUT_COLUMNS)]
        .nunique()
        .gt(1)
        .any(axis=1)
        .sum()
    )


def us_esi_premiums_summary(frame: Frame) -> dict[str, object]:
    """Weighted ESI premium lineage and structure, for gates and manifests.

    It recomputes the raw MEPS-IC shares from the raw ASEC columns, so the
    scale factor, the anchor-universe total and every structural zero are
    checked against the cells rather than trusted. A frame that lacks those
    columns gets the column totals only, with ``source_columns_missing``
    naming what is absent; both gates fail on it.
    """

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    employer = person[US_ESI_EMPLOYER_PREMIUM_COLUMN].to_numpy(dtype=np.float64)
    positive = employer > 0
    total_weight = float(weights.sum())
    holders = float(weights[positive].sum())
    person = _person_with_state(frame)
    summary: dict[str, object] = {
        "employer_premium_total": _weighted(weights, employer),
        "employer_premium_positive_persons": holders,
        "employer_premium_mean_per_positive_person": (
            _weighted(weights, employer, positive) / holders if holders else 0.0
        ),
        "employer_premium_positive_share": holders / total_weight
        if total_weight
        else 0.0,
        "nonfinite_rows": int((~np.isfinite(employer)).sum()),
        "negative_rows": int((employer < 0).sum()),
        "clone_disagreement_source_persons": _clone_disagreements(person),
        "anchor": dict(EMPLOYER_PREMIUM_ANCHOR),
        "cross_check": dict(EMPLOYER_PREMIUM_CROSS_CHECK),
        "cells_sha256": _CELLS_SHA256,
        "source_columns_missing": [
            column
            for column in US_ESI_PREMIUMS_REQUIRED_SOURCE_COLUMNS
            if column not in person.columns
        ],
    }
    if summary["source_columns_missing"]:
        return summary
    codes, raw, premium, contribution = _person_raw_shares(person)
    universe = codes.universe
    holder = codes.policyholder
    other = holder & ~universe
    raw_positive = universe & (raw > 0)
    raw_universe_total = _weighted(weights, raw, universe)
    raw_anchor_total = _weighted(weights, raw)
    scale = (
        summary["employer_premium_total"] / raw_universe_total
        if raw_universe_total
        else float("nan")
    )
    by_tier = {
        tier: _weighted(weights, employer, universe & (codes.tier == code))
        for code, tier in _TIER_BY_CODE.items()
    }
    by_sector = {
        "private": _weighted(
            weights, employer, np.isin(codes.sector, _PRIVATE_EMPLOYER_CODES)
        ),
        "federal": _weighted(weights, employer, codes.sector == _FEDERAL),
        "state_and_local": _weighted(
            weights,
            employer,
            np.isin(codes.sector, [_STATE_GOVERNMENT, _LOCAL_GOVERNMENT]),
        ),
    }
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(
            raw_positive, employer / np.where(raw_positive, raw, 1.0), np.nan
        )
    # MEPS-IC's published employer mean of a cell is premium less average
    # contribution over every enrollee the employer pays for.
    employer_pays = universe & np.isin(codes.hipaid, [_PAYS_ALL, _PAYS_SOME])
    published = _weighted(weights, premium - contribution, employer_pays)
    summary |= {
        "weighted_policyholders": float(weights[holder].sum()),
        "weighted_employed_policyholders": float(weights[universe].sum()),
        "weighted_employed_policyholders_by_payment": {
            name: float(weights[universe & (codes.hipaid == code)].sum())
            for name, code in (
                ("all", _PAYS_ALL),
                ("some", _PAYS_SOME),
                ("none", _PAYS_NONE),
            )
        },
        "weighted_other_policyholders": float(weights[other].sum()),
        "weighted_other_policyholders_with_employer_share": float(
            weights[other & (raw > 0)].sum()
        ),
        "raw_employer_share_total": raw_universe_total,
        "raw_anchor_universe_total": raw_anchor_total,
        "raw_employer_share_mean_per_positive_person": (
            _weighted(weights, raw, raw_positive) / float(weights[raw_positive].sum())
            if raw_positive.any()
            else 0.0
        ),
        "raw_employer_share_band": list(_RAW_MEAN_BAND),
        "raw_over_published_employer_mean": (
            _weighted(weights, raw, employer_pays) / published
            if published
            else float("nan")
        ),
        "scale_factor": scale,
        "scale_factor_spread": (
            float(np.nanmax(ratio) - np.nanmin(ratio)) if raw_positive.any() else 0.0
        ),
        "anchor_universe_employer_total": scale * raw_anchor_total,
        "other_policyholder_employer_total": scale
        * (raw_anchor_total - raw_universe_total),
        "employed_share_of_anchor_universe": (
            raw_universe_total / raw_anchor_total if raw_anchor_total else float("nan")
        ),
        "employer_premium_by_tier": by_tier,
        "employer_premium_by_sector": by_sector,
        "employer_premium_outside_universe_rows": int((positive & ~universe).sum()),
        "employer_premium_where_employer_pays_none_rows": int(
            (positive & (codes.hipaid == _PAYS_NONE)).sum()
        ),
        "employer_premium_without_raw_share_rows": int(
            (positive & ~raw_positive).sum()
        ),
        "meps_ic_private_active_employer_total": (
            meps_ic_private_active_employer_totals(load_meps_ic_esi_premium_cells())
        ),
    }
    return summary


def _missing_source_failure(missing: list[str], consequence: str) -> str:
    return (
        f"assignment provenance incomplete: the person table lacks {missing}, so "
        f"{consequence}. Rebuild the base with the meps_esi_premiums stage; a "
        "release must carry the raw ASEC coverage columns the assignment read."
    )


def us_esi_premiums_signal_gate(frame: Frame) -> GateResult:
    """Structure and scale-invariant signal of the ESI employer premium input.

    Holds on any frame size (smoke builds included). It proves every positive
    value sits in the employed-policyholder universe and equals one common
    multiple of its MEPS-IC cell. Those proofs read the raw ASEC columns, so a
    frame without them fails: it cannot be certified.
    """

    person = frame.table("person")
    missing = [
        column
        for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS
        if column not in person.columns
    ]
    if missing:
        return GateResult(
            name="esi_premiums_signal",
            passed=False,
            failures=tuple(f"person column missing: {column}." for column in missing),
            details={"missing": missing},
        )
    try:
        summary = us_esi_premiums_summary(frame)
    except SourceRuntimeError as exc:
        return GateResult(
            name="esi_premiums_signal",
            passed=False,
            failures=(str(exc),),
            details={"structural_error": str(exc)},
        )
    failures: list[str] = []
    if summary["source_columns_missing"]:
        failures.append(
            _missing_source_failure(
                list(summary["source_columns_missing"]),
                "the universe, payment-status and MEPS-IC cell checks did not run",
            )
        )
    for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS:
        if person[column].dropna().nunique() < 2:
            failures.append(f"{column}: constant column — zero mass or no signal.")
    for key, label in (
        ("nonfinite_rows", "non-finite value(s)"),
        ("negative_rows", "negative value(s)"),
        ("clone_disagreement_source_persons", "support-clone disagreement(s)"),
        (
            "employer_premium_outside_universe_rows",
            "employer premium(s) outside employed policyholders",
        ),
        (
            "employer_premium_where_employer_pays_none_rows",
            "employer premium(s) where the employer pays none",
        ),
        (
            "employer_premium_without_raw_share_rows",
            "employer premium(s) with no MEPS-IC share",
        ),
    ):
        count = int(summary.get(key, 0))
        if count:
            failures.append(f"ESI premiums: {count} {label}.")
    share = float(summary["employer_premium_positive_share"])
    low, high = _POSITIVE_SHARE_BAND
    if not low <= share <= high:
        failures.append(
            f"employer premium positive share {share:.4f} outside [{low}, {high}]."
        )
    if "raw_employer_share_mean_per_positive_person" in summary:
        mean = float(summary["raw_employer_share_mean_per_positive_person"])
        low, high = _RAW_MEAN_BAND
        if not low <= mean <= high:
            failures.append(
                f"raw MEPS-IC employer share per holder ${mean:,.0f} outside "
                f"[${low:,.0f}, ${high:,.0f}]: the cell table's units are broken."
            )
        # Relative to the factor: a float32 round trip leaves ~1e-7.
        spread = float(summary["scale_factor_spread"])
        if spread > 1e-6 * float(summary["scale_factor"]):
            failures.append(
                f"employer premiums are not one common multiple of their MEPS-IC "
                f"cells (scale-factor spread {spread:.3g})."
            )
    return GateResult(
        name="esi_premiums_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )


def us_esi_premiums_anchor_gate(frame: Frame, *, time_period: int) -> GateResult:
    """Hold a release's weighted ESI employer premium total to its anchor.

    Runs on the calibrated export (dense and sparse). The anchor, CMS NHE
    Table 24, counts employer contributions for every policyholder, so the
    gate compares the anchor-universe total: the column (employed
    policyholders) plus the same scale factor applied to the other
    policyholders' MEPS-IC shares, recomputed from the raw ASEC columns at the
    frame's weights. It fails when the column is absent or zero-mass, when
    the raw columns are absent, when that total leaves
    ±``ANCHOR_RELATIVE_TOLERANCE`` of the anchor for ``time_period``, or when
    the employed share of it leaves its band. BEA NIPA 7.8 line 17 and
    MEPS-IC's own private-sector enrollment total are recorded beside the
    verdict, not gated.
    """

    person = frame.table("person")
    missing = [
        column
        for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS
        if column not in person.columns
    ]
    if missing:
        return GateResult(
            name="esi_premiums_anchor",
            passed=False,
            failures=tuple(f"person column missing: {column}." for column in missing),
            details={"missing": missing},
        )
    try:
        summary = us_esi_premiums_summary(frame)
    except SourceRuntimeError as exc:
        return GateResult(
            name="esi_premiums_anchor",
            passed=False,
            failures=(str(exc),),
            details={"structural_error": str(exc)},
        )
    employer_total = float(summary["employer_premium_total"])
    year = str(int(time_period))
    failures: list[str] = []
    details: dict[str, object] = {
        "time_period": int(time_period),
        "employer_premium_total": employer_total,
        "employer_premium_positive_persons": summary[
            "employer_premium_positive_persons"
        ],
        "employer_premium_mean_per_positive_person": summary[
            "employer_premium_mean_per_positive_person"
        ],
        "relative_tolerance": ANCHOR_RELATIVE_TOLERANCE,
        "anchor": dict(EMPLOYER_PREMIUM_ANCHOR),
        "cross_check": dict(EMPLOYER_PREMIUM_CROSS_CHECK),
    }
    if not employer_total > 0:
        failures.append(f"{US_ESI_EMPLOYER_PREMIUM_COLUMN}: zero weighted mass.")
    anchor = EMPLOYER_PREMIUM_ANCHOR["values"].get(year)
    if anchor is None:
        failures.append(f"no NHE Table 24 employer-contribution anchor for {year}.")
    if summary["source_columns_missing"]:
        failures.append(
            _missing_source_failure(
                list(summary["source_columns_missing"]),
                "the anchor-universe total (every policyholder) cannot be computed",
            )
        )
    else:
        universe_total = float(summary["anchor_universe_employer_total"])
        employed_share = float(summary["employed_share_of_anchor_universe"])
        details |= {
            key: summary[key]
            for key in (
                "anchor_universe_employer_total",
                "other_policyholder_employer_total",
                "employed_share_of_anchor_universe",
                "scale_factor",
                "weighted_policyholders",
                "weighted_employed_policyholders",
                "weighted_other_policyholders",
                "employer_premium_by_sector",
            )
        }
        if not np.isfinite(universe_total):
            failures.append(
                "the anchor-universe employer total is not finite: no employed "
                "policyholder carries a MEPS-IC share to recover the scale from."
            )
        elif anchor is not None:
            relative = universe_total / float(anchor) - 1.0
            details |= {"anchor_value": float(anchor), "relative_error": relative}
            if abs(relative) > ANCHOR_RELATIVE_TOLERANCE:
                failures.append(
                    "employer premiums over every policyholder total "
                    f"${universe_total / 1e9:,.1f}B, {relative:+.1%} from NHE "
                    f"Table 24 ${float(anchor) / 1e9:,.1f}B ({year}); tolerance "
                    f"±{ANCHOR_RELATIVE_TOLERANCE:.0%}."
                )
        low, high = _EMPLOYED_SHARE_BAND
        details["employed_share_band"] = [low, high]
        if not low <= employed_share <= high:
            failures.append(
                f"employed policyholders carry {employed_share:.4f} of the "
                f"anchor-universe employer total, outside [{low}, {high}]."
            )
        bea = EMPLOYER_PREMIUM_CROSS_CHECK["values"].get(year)
        if bea is not None and np.isfinite(universe_total):
            details["cross_check_ratio"] = universe_total / float(bea)
        active = dict(summary["meps_ic_private_active_employer_total"])
        if year in active:
            private = float(dict(summary["employer_premium_by_sector"])["private"])
            details["private_active_cross_check"] = {
                "meps_ic_private_active_employer_total": active[year],
                "employer_premium_private_sector": private,
                "ratio": private / active[year],
            }
    return GateResult(
        name="esi_premiums_anchor",
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )
