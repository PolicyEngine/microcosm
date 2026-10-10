"""MEPS-IC employer premiums and pre-tax employee premiums for ESI policyholders.

Microcosm #454. PolicyEngine-US 2.2.1 reads two person inputs that no
Microcosm stage produced:

* ``employer_sponsored_insurance_premiums`` — "annual employer-paid health
  insurance premiums", one of ``gov.household.cbo_market_income_additions``
  (CBO household market income); and
* ``pre_tax_health_insurance_premiums`` — premiums paid through pre-tax
  payroll deductions, listed in ``gov.irs.gross_income.pre_tax_contributions``
  (subtracted from ``employment_income`` in ``irs_employment_income``) and
  ``gov.irs.gross_income.fica_pre_tax_contributions`` (subtracted in
  ``payroll_tax_gross_wages``).

The retired eCPS derivation (archived 42ed5d45 ``datasets/cps/cps.py``
L197-271) read the CPS ASEC current employment-based coverage fields, which the
pinned H5 inputs omit; :mod:`.asec_census_person_columns` now restores them,
with employment status and employer size, from the pinned Census person files.

Employer premium
----------------
The universe is the **employed** (``PEMLR`` 1-2) **current ESI policyholder**
(``NOW_OWNGRP`` 1). Dependents carry nothing (the policyholder carries the
premium) and non-employed policyholders carry nothing (retiree and COBRA
coverage is not compensation of a current job). Self-employed-unincorporated
workers and workers without pay (``PEIO1COW`` 7-8) carry nothing either: their
current job has no employer to pay a share.

Each policyholder's raw employer share comes from the MEPS-IC cell of their
coverage tier (``NOW_GRPFTYP2``: family, self plus one, self-only), employer
sector (``PEIO1COW``) and, for private employers, State and firm size
(``NOEMP`` under 50 / 50 or more):

* employer paid all of the premium (``NOW_HIPAID`` 1): the cell's average
  total premium;
* employer paid some (2): the average total premium minus the average employee
  contribution;
* employer paid none (3): zero.

Private-sector cells are MEPS-IC 2025 Series II (State by firm size).
Government cells are MEPS-IC 2024 Series III by census division (the latest
year published), aged to 2025 by the private-sector national 2025/2024 ratio of
the same tier and measure: State government employees take the
State-government column, local and federal employees the all-governments
column (MEPS-IC does not survey the federal government, so that is a
stand-in). AHRQ suppresses nine small-firm employee-contribution cells; each
takes the national small-firm cell times the State's total-to-national ratio.

A single factor then scales every raw share so the weighted total equals BEA
NIPA Table 7.8 line 17 (series B4923C, "Private group health insurance",
employer contributions as a supplement to wages and salaries; government
employers' contributions to privately administered plans included) for the
build year. BEA's NIPA handbook (chapter 10, table 10.B) says the private and
State and local parts of that series come from MEPS data "on insurance
purchased by employers for employees", the active-employee concept this column
carries. CMS NHE Table 24's employer contribution is the cross-check, and an
upper bound: the NHEA methodology counts premiums for active employees, COBRA
enrollees and retirees.

Pre-tax employee premium
------------------------
Eligible: an employed policyholder with wages last year (``WSAL_VAL`` > 0), a
positive reported premium (``PHIP_VAL``) and an employee share to pay
(``NOW_HIPAID`` 2 or 3). An eligible person pays the reported premium pre-tax
with the firm-size probability that their employer offers pretax employee
contributions, among employers that offer health insurance: MEPS-IC 2025 Table
I.A.2.j over Table I.A.2. The rates are establishment-weighted; the published
bands rise with firm size (8.9% under 10 employees, 92.6% at 1,000 or more),
so they understate the enrollee-weighted rate and the column is a lower
estimate.
The draw is a seeded blake2b uniform keyed by stable source identity, so
support clones agree. Others carry zero.
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
    "US_ESI_PRE_TAX_PREMIUM_COLUMN",
    "derive_us_employer_esi_premiums_from_manifest",
    "derive_us_pre_tax_health_insurance_premiums_from_manifest",
    "load_meps_ic_esi_premium_cells",
    "us_esi_premiums_anchor_gate",
    "us_esi_premiums_signal_gate",
    "us_esi_premiums_stage_spec",
    "us_esi_premiums_summary",
    "with_us_esi_premium_inputs",
]

US_ESI_PREMIUMS_STAGE_NAME = "meps_esi_premiums"
US_ESI_EMPLOYER_PREMIUM_COLUMN = "employer_sponsored_insurance_premiums"
US_ESI_PRE_TAX_PREMIUM_COLUMN = "pre_tax_health_insurance_premiums"
US_ESI_PREMIUMS_OUTPUT_COLUMNS: tuple[str, ...] = (
    US_ESI_EMPLOYER_PREMIUM_COLUMN,
    US_ESI_PRE_TAX_PREMIUM_COLUMN,
)
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
    "PHIP_VAL",
    "WSAL_VAL",
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
_PRIVATE_EMPLOYER_CODES = (4, 5, 6)
_EMPLOYER_SECTOR_CODES = (_FEDERAL, _STATE_GOVERNMENT, _LOCAL_GOVERNMENT, 4, 5, 6)

_CELLS_RESOURCE = "meps_ic_esi_premium_cells.json"
_CELLS_SHA256 = "fb07b8c27ff94fd1423623f7385868fb26e0fce7ad99f80c7cf0d205364e77f8"
_PERSON_WEIGHT_COLUMN = "person_weight"
_DRAW_SALT = US_ESI_PRE_TAX_PREMIUM_COLUMN

_BEA_SECTION7_URL = (
    "https://apps.bea.gov/national/Release/XLS/Survey/Section7All_xls.xlsx"
)
_NHE_TABLES_URL = "https://www.cms.gov/files/zip/nhe-tables.zip"

#: BEA NIPA Table 7.8 line 17 (B4923C), millions of dollars converted to
#: dollars, from the annual update published 2026-09-30 (file created
#: 2026-09-28). That update revised 2024 from the $1,002.9B the retired
#: us-data loss matrix pinned.
EMPLOYER_PREMIUM_ANCHOR: Mapping[str, Any] = MappingProxyType(
    {
        "source": _BEA_SECTION7_URL,
        "sha256": "de1c34e37da9b8b8d765efc0611cc653102373fe522fe9218c7689067b29b7fd",
        "table": "NIPA Table 7.8 Supplements to Wages and Salaries by Type",
        "line": 17,
        "series": "B4923C",
        "label": "Private group health insurance",
        "published": "2026-09-30",
        # Integer dollars: the manifest round-trips through YAML.
        "values": {
            "2023": 923_195_000_000,
            "2024": 977_034_000_000,
            "2025": 1_027_929_000_000,
        },
    }
)
#: CMS NHE Table 24 (2024 release, data through CY2024), the cross-check and
#: upper bound; banked in the pinned feed as
#: cms_nhe.cy{2023,2024}.esi_employer_contribution_premiums and
#: ..._private_employer_contribution_premiums.
EMPLOYER_PREMIUM_CROSS_CHECK: Mapping[str, Any] = MappingProxyType(
    {
        "source": _NHE_TABLES_URL,
        "sha256": "a09ef6d3e84e25d745047a47b6b08a0d96b303085b4c725b67ce67a0eb0c4420",
        "table": "Table 24 Employer-Sponsored Private Health Insurance",
        "employer_contribution": {"2023": 975.7e9, "2024": 1_047.0e9},
        "employee_contribution": {"2023": 357.8e9, "2024": 382.1e9},
        "relation": (
            "upper bound: NHEA counts employer contributions for active "
            "employees, COBRA enrollees and retirees"
        ),
    }
)
#: Tolerance of the calibrated release aggregate around the BEA anchor. The
#: stage pins the pre-calibration aggregate exactly; the band absorbs
#: calibration and sparse-selection drift. It is wider than BEA's own 2024
#: revision (2.6%) and narrower than the NHE-BEA gap (7.2%), so a release that
#: drifts toward the retiree-inclusive NHE concept fails.
ANCHOR_RELATIVE_TOLERANCE = 0.05
#: Weighted-raw employer share per positive holder, before scaling. MEPS-IC
#: 2025 employer shares run $7.2k (single) to $19.0k (family) and the pinned
#: pools measure $12.3-12.4k; this band catches a broken cell table, which the
#: scale factor would otherwise hide.
_RAW_MEAN_BAND = (6_000.0, 20_000.0)
#: Weighted share of all persons with a positive employer premium. Measured
#: on both pinned pools: 22.7%.
_POSITIVE_SHARE_BAND = (0.15, 0.32)
#: Weighted share of all persons with a positive pre-tax premium. Measured on
#: both pinned pools: 16.1%.
_PRE_TAX_POSITIVE_SHARE_BAND = (0.05, 0.25)

_EMPLOYER_PREMIUM_PARAMETERS: Mapping[str, Any] = MappingProxyType(
    {
        "cells_resource": f"microcosm.build.us_runtime.data/{_CELLS_RESOURCE}",
        "cells_sha256": _CELLS_SHA256,
        "universe": (
            "NOW_OWNGRP == 1 and PEMLR in (1, 2) and PEIO1COW in (1, 2, 3, 4, 5, 6)"
        ),
        "tier_column": "NOW_GRPFTYP2",
        "size_column": "NOEMP",
        "sector_column": "PEIO1COW",
        "payment_column": "NOW_HIPAID",
        "payment_rule": (
            "1 (employer paid all): average total premium; 2 (some): average "
            "total premium minus average employee contribution; 3 (none): 0"
        ),
        "suppressed_cell_fallback": (
            "national firm-size cell times the State total-to-national ratio"
        ),
        "government_cell_aging": (
            "MEPS-IC 2024 Series III x private-sector national 2025/2024 "
            "ratio of the same tier and measure"
        ),
        "anchor": {
            "source": EMPLOYER_PREMIUM_ANCHOR["source"],
            "sha256": EMPLOYER_PREMIUM_ANCHOR["sha256"],
            "series": EMPLOYER_PREMIUM_ANCHOR["series"],
            "values": dict(EMPLOYER_PREMIUM_ANCHOR["values"]),
        },
    }
)
_PRE_TAX_PARAMETERS: Mapping[str, Any] = MappingProxyType(
    {
        "seed_from_build_config": True,
        "premium_column": "PHIP_VAL",
        "wage_column": "WSAL_VAL",
        "employer_payment_codes": [_PAYS_SOME, _PAYS_NONE],
        "pre_tax_share": (
            "MEPS-IC 2025 Table I.A.2.j pretax-contribution offer percent / "
            "Table I.A.2 health-insurance offer percent, by firm size"
        ),
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
        ("derive_pre_tax_health_insurance_premiums", _plain(_PRE_TAX_PARAMETERS)),
    )
    actual = tuple(
        (operation.kind, dict(operation.parameters)) for operation in spec.operations
    )
    if actual != expected:
        raise ValueError(
            "US ESI premium operations drifted from the reviewed MEPS-IC cell "
            "assignment and pre-tax share contract."
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
    def universe(self) -> np.ndarray:
        return (
            (self.owner == _POLICYHOLDER)
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


def _cell_values(
    codes: _EsiCodes, cells: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Average total premium and employee contribution of each person's cell.

    NaN outside the employed-policyholder universe.
    """

    premium = np.full(len(codes.owner), np.nan)
    contribution = np.full(len(codes.owner), np.nan)
    divisions = cells["state_census_division"]
    keys = pd.DataFrame(
        {
            "tier": codes.tier,
            "noemp": codes.noemp,
            "sector": codes.sector,
            "state": codes.state_fips,
        }
    )
    universe = codes.universe
    for (tier_code, noemp, sector, state), index in (
        keys[universe].groupby(["tier", "noemp", "sector", "state"]).groups.items()
    ):
        tier = _TIER_BY_CODE[int(tier_code)]
        state_key = f"{int(state):02d}"
        if state_key not in divisions:
            raise SourceRuntimeError(
                f"US ESI premium stage: State FIPS {int(state)} has no MEPS-IC cell."
            )
        if int(sector) in _PRIVATE_EMPLOYER_CODES:
            size = _SIZE_BY_NOEMP[int(noemp)]
            values = [
                _private_cell(cells, tier, measure, state_key, size)
                for measure in ("premium", "employee_contribution")
            ]
        else:
            column = (
                "state" if int(sector) == _STATE_GOVERNMENT else "all_state_and_local"
            )
            values = [
                _government_cell(cells, tier, measure, divisions[state_key], column)
                for measure in ("premium", "employee_contribution")
            ]
        rows = np.asarray(index, dtype=np.int64)
        premium[rows], contribution[rows] = values
    return premium, contribution


def _raw_employer_share(
    codes: _EsiCodes, premium: np.ndarray, contribution: np.ndarray
) -> np.ndarray:
    raw = np.zeros(len(codes.owner), dtype=np.float64)
    universe = codes.universe
    pays_all = universe & (codes.hipaid == _PAYS_ALL)
    pays_some = universe & (codes.hipaid == _PAYS_SOME)
    raw[pays_all] = premium[pays_all]
    raw[pays_some] = np.clip(premium[pays_some] - contribution[pays_some], 0.0, None)
    return raw


def _anchor(year: int) -> float:
    values = EMPLOYER_PREMIUM_ANCHOR["values"]
    if str(year) not in values:
        raise SourceRuntimeError(
            f"US ESI premium stage has no BEA anchor for {year}; pinned years: "
            f"{sorted(values)}."
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
    """Assign MEPS-IC employer shares and scale them to the BEA anchor."""

    if operation.kind != "derive_employer_sponsored_insurance_premiums":
        raise SourceRuntimeError(
            f"Unexpected ESI premium operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError("ESI premium assignment requires the person table.")
    _check_parameters(operation, _EMPLOYER_PREMIUM_PARAMETERS)
    codes = _esi_person_codes(frame)
    premium, contribution = _cell_values(codes, load_meps_ic_esi_premium_cells())
    raw = _raw_employer_share(codes, premium, contribution)
    weights = _person_weights(frame)
    raw_total = float(weights @ raw)
    if not raw_total > 0:
        raise SourceRuntimeError(
            "US ESI premium stage found no weighted employer-paid premium mass to "
            "scale: no employed policyholder with an employer share."
        )
    scale = _anchor(context.config.target_year) / raw_total
    result = frame.copy(deep=True)
    result[US_ESI_EMPLOYER_PREMIUM_COLUMN] = raw * scale
    return result


def _pre_tax_shares(cells: Mapping[str, Any]) -> dict[str, float]:
    rows = cells["pretax_contribution_2025"]["rows"]
    return {
        size: rows[size]["pretax_contribution_offer_percent"]
        / rows[size]["health_insurance_offer_percent"]
        for size in ("total", "lt50", "50plus")
    }


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


def _stable_person_draws(frame: pd.DataFrame, *, seed: int) -> np.ndarray:
    """Seeded uniform draws keyed by stable source identity per person.

    Support-channel clones share their source identity, so they always
    receive the same draw; frames without source columns key on the person
    id itself.
    """

    denominator = float(2**64)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:{_DRAW_SALT}:{key}".encode(),
                    digest_size=8,
                ).digest(),
                byteorder="big",
                signed=False,
            )
            / denominator
            for key in _stable_person_keys(frame)
        ],
        dtype=np.float64,
    )


def _pre_tax_eligible(frame: pd.DataFrame, codes: _EsiCodes) -> np.ndarray:
    premium = pd.to_numeric(frame["PHIP_VAL"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    wages = pd.to_numeric(frame["WSAL_VAL"], errors="coerce").to_numpy(dtype=np.float64)
    if not (np.isfinite(premium).all() and np.isfinite(wages).all()):
        raise SourceRuntimeError(
            "US ESI premium stage requires finite PHIP_VAL and WSAL_VAL."
        )
    return (
        codes.universe
        & np.isin(codes.hipaid, [_PAYS_SOME, _PAYS_NONE])
        & (wages > 0)
        & (premium > 0)
    )


def derive_us_pre_tax_health_insurance_premiums_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Route an eligible policyholder's reported premium through payroll."""

    if operation.kind != "derive_pre_tax_health_insurance_premiums":
        raise SourceRuntimeError(
            f"Unexpected ESI premium operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "Pre-tax premium assignment requires the person table."
        )
    _check_parameters(operation, _PRE_TAX_PARAMETERS)
    codes = _esi_person_codes(frame)
    eligible = _pre_tax_eligible(frame, codes)
    shares = _pre_tax_shares(load_meps_ic_esi_premium_cells())
    probability = (
        pd.Series(codes.noemp).map(dict(_SIZE_BY_NOEMP)).map(shares).to_numpy()
    )
    draws = _stable_person_draws(frame, seed=context.config.seed)
    premium = pd.to_numeric(frame["PHIP_VAL"]).to_numpy(dtype=np.float64)
    result = frame.copy(deep=True)
    result[US_ESI_PRE_TAX_PREMIUM_COLUMN] = np.where(
        eligible & (draws < probability), premium, 0.0
    )
    return result


def _has_raw_columns(person: pd.DataFrame) -> bool:
    return set(US_ESI_PREMIUMS_REQUIRED_SOURCE_COLUMNS) <= set(person.columns)


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

    A frame that already carries both outputs with signal and passes the
    signal gate is returned unchanged (idempotent). Otherwise the person table
    must still carry the raw ASEC columns; the stage never defaults them.
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
            "derive_pre_tax_health_insurance_premiums": (
                derive_us_pre_tax_health_insurance_premiums_from_manifest
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

    With the raw ASEC columns present it also recomputes the raw MEPS-IC
    shares, so the scale factor and every structural zero are checked against
    the cells rather than trusted.
    """

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    employer = person[US_ESI_EMPLOYER_PREMIUM_COLUMN].to_numpy(dtype=np.float64)
    pre_tax = person[US_ESI_PRE_TAX_PREMIUM_COLUMN].to_numpy(dtype=np.float64)
    positive = employer > 0
    total_weight = float(weights.sum())
    holders = float(weights[positive].sum())
    summary: dict[str, object] = {
        "employer_premium_total": _weighted(weights, employer),
        "employer_premium_positive_persons": holders,
        "employer_premium_mean_per_positive_person": (
            _weighted(weights, employer, positive) / holders if holders else 0.0
        ),
        "employer_premium_positive_share": holders / total_weight
        if total_weight
        else 0.0,
        "pre_tax_premium_total": _weighted(weights, pre_tax),
        "pre_tax_premium_positive_persons": float(weights[pre_tax > 0].sum()),
        "pre_tax_premium_positive_share": (
            float(weights[pre_tax > 0].sum()) / total_weight if total_weight else 0.0
        ),
        "nonfinite_rows": int(
            (~np.isfinite(employer)).sum() + (~np.isfinite(pre_tax)).sum()
        ),
        "negative_rows": int((employer < 0).sum() + (pre_tax < 0).sum()),
        "clone_disagreement_source_persons": _clone_disagreements(person),
        "anchor": dict(EMPLOYER_PREMIUM_ANCHOR),
        "cross_check": dict(EMPLOYER_PREMIUM_CROSS_CHECK),
        "cells_sha256": _CELLS_SHA256,
    }
    person = _person_with_state(frame)
    if not _has_raw_columns(person):
        return summary
    codes = _esi_person_codes(person)
    cells = load_meps_ic_esi_premium_cells()
    premium, contribution = _cell_values(codes, cells)
    raw = _raw_employer_share(codes, premium, contribution)
    raw_positive = raw > 0
    raw_total = _weighted(weights, raw)
    scale = summary["employer_premium_total"] / raw_total if raw_total else float("nan")
    eligible = _pre_tax_eligible(person, codes)
    phip = pd.to_numeric(person["PHIP_VAL"]).to_numpy(dtype=np.float64)
    holder = codes.owner == _POLICYHOLDER
    by_tier = {
        tier: _weighted(weights, employer, codes.universe & (codes.tier == code))
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
    summary |= {
        "weighted_policyholders": float(weights[holder].sum()),
        "weighted_employed_policyholders": float(weights[codes.universe].sum()),
        "weighted_employed_policyholders_by_payment": {
            name: float(weights[codes.universe & (codes.hipaid == code)].sum())
            for name, code in (
                ("all", _PAYS_ALL),
                ("some", _PAYS_SOME),
                ("none", _PAYS_NONE),
            )
        },
        "raw_employer_share_total": raw_total,
        "raw_employer_share_mean_per_positive_person": (
            _weighted(weights, raw, raw_positive) / float(weights[raw_positive].sum())
            if raw_positive.any()
            else 0.0
        ),
        "raw_employer_share_band": list(_RAW_MEAN_BAND),
        "scale_factor": scale,
        "scale_factor_spread": (
            float(np.nanmax(ratio) - np.nanmin(ratio)) if raw_positive.any() else 0.0
        ),
        "employer_premium_by_tier": by_tier,
        "employer_premium_by_sector": by_sector,
        "employer_premium_outside_universe_rows": int(
            (positive & ~codes.universe).sum()
        ),
        "employer_premium_where_employer_pays_none_rows": int(
            (positive & (codes.hipaid == _PAYS_NONE)).sum()
        ),
        "employer_premium_without_raw_share_rows": int(
            (positive & ~raw_positive).sum()
        ),
        "pre_tax_ineligible_rows": int(((pre_tax > 0) & ~eligible).sum()),
        "pre_tax_not_reported_premium_rows": int(
            ((pre_tax > 0) & ~np.isclose(pre_tax, phip)).sum()
        ),
        "pre_tax_eligible_weight": float(weights[eligible].sum()),
        "pre_tax_selected_share_of_eligible": (
            float(weights[eligible & (pre_tax > 0)].sum())
            / float(weights[eligible].sum())
            if eligible.any()
            else 0.0
        ),
        "pre_tax_shares": _pre_tax_shares(cells),
    }
    return summary


def us_esi_premiums_signal_gate(frame: Frame) -> GateResult:
    """Structure and scale-invariant signal of the two ESI premium inputs.

    Holds on any frame size (smoke builds included). On frames that still
    carry the raw ASEC columns it also proves every positive value sits in the
    employed-policyholder universe, equals one common multiple of its MEPS-IC
    cell, and that pre-tax premiums are reported premiums of eligible people.
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
        ("pre_tax_ineligible_rows", "pre-tax premium(s) for ineligible people"),
        (
            "pre_tax_not_reported_premium_rows",
            "pre-tax premium(s) not equal to PHIP_VAL",
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
    share = float(summary["pre_tax_premium_positive_share"])
    low, high = _PRE_TAX_POSITIVE_SHARE_BAND
    if not low <= share <= high:
        failures.append(
            f"pre-tax premium positive share {share:.4f} outside [{low}, {high}]."
        )
    if "raw_employer_share_mean_per_positive_person" in summary:
        mean = float(summary["raw_employer_share_mean_per_positive_person"])
        low, high = _RAW_MEAN_BAND
        if not low <= mean <= high:
            failures.append(
                f"raw MEPS-IC employer share per holder ${mean:,.0f} outside "
                f"[${low:,.0f}, ${high:,.0f}]: the cell table is broken."
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
    """Hold a release's weighted ESI premium totals to their external anchors.

    Runs on the calibrated export (dense and sparse). Fails when either column
    is absent or zero-mass, when the employer premium total leaves
    ±``ANCHOR_RELATIVE_TOLERANCE`` of BEA NIPA 7.8 line 17 for ``time_period``
    or exceeds the retiree-inclusive NHE Table 24 employer contribution, or
    when pre-tax premiums exceed NHE's employee contribution (which also
    counts retiree- and COBRA-paid premiums).
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
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    employer = person[US_ESI_EMPLOYER_PREMIUM_COLUMN].to_numpy(dtype=np.float64)
    pre_tax = person[US_ESI_PRE_TAX_PREMIUM_COLUMN].to_numpy(dtype=np.float64)
    employer_total = _weighted(weights, employer)
    pre_tax_total = _weighted(weights, pre_tax)
    year = str(int(time_period))
    failures: list[str] = []
    details: dict[str, object] = {
        "time_period": int(time_period),
        "employer_premium_total": employer_total,
        "pre_tax_premium_total": pre_tax_total,
        "employer_premium_positive_persons": float(weights[employer > 0].sum()),
        "relative_tolerance": ANCHOR_RELATIVE_TOLERANCE,
        "anchor": dict(EMPLOYER_PREMIUM_ANCHOR),
        "cross_check": dict(EMPLOYER_PREMIUM_CROSS_CHECK),
    }
    for column, total in (
        (US_ESI_EMPLOYER_PREMIUM_COLUMN, employer_total),
        (US_ESI_PRE_TAX_PREMIUM_COLUMN, pre_tax_total),
    ):
        if not total > 0:
            failures.append(f"{column}: zero weighted mass.")
    anchor = EMPLOYER_PREMIUM_ANCHOR["values"].get(year)
    if anchor is None:
        failures.append(f"no BEA NIPA 7.8 line 17 anchor for {year}.")
    else:
        relative = employer_total / float(anchor) - 1.0
        details |= {"anchor_value": float(anchor), "relative_error": relative}
        if abs(relative) > ANCHOR_RELATIVE_TOLERANCE:
            failures.append(
                f"{US_ESI_EMPLOYER_PREMIUM_COLUMN}: weighted total "
                f"${employer_total / 1e9:,.1f}B is {relative:+.1%} from BEA NIPA "
                f"7.8 line 17 ${float(anchor) / 1e9:,.1f}B ({year}); tolerance "
                f"±{ANCHOR_RELATIVE_TOLERANCE:.0%}."
            )
    nhe_employer = EMPLOYER_PREMIUM_CROSS_CHECK["employer_contribution"].get(year)
    nhe_employee = EMPLOYER_PREMIUM_CROSS_CHECK["employee_contribution"].get(year)
    if nhe_employer is not None:
        details["cross_check_employer_ratio"] = employer_total / float(nhe_employer)
        if employer_total > float(nhe_employer):
            failures.append(
                f"{US_ESI_EMPLOYER_PREMIUM_COLUMN}: weighted total "
                f"${employer_total / 1e9:,.1f}B exceeds the retiree-inclusive NHE "
                f"Table 24 employer contribution ${float(nhe_employer) / 1e9:,.1f}B."
            )
    if nhe_employee is not None:
        details["cross_check_pre_tax_ratio"] = pre_tax_total / float(nhe_employee)
        if pre_tax_total > float(nhe_employee):
            failures.append(
                f"{US_ESI_PRE_TAX_PREMIUM_COLUMN}: weighted total "
                f"${pre_tax_total / 1e9:,.1f}B exceeds NHE Table 24's employee "
                f"contribution ${float(nhe_employee) / 1e9:,.1f}B."
            )
    return GateResult(
        name="esi_premiums_anchor",
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )
