"""Held-out SNAP caseload composition and take-up, never a calibration target.

The US release reports this diagnostic beside its other held-out reports.
Calibration fits state SNAP household counts and benefit dollars, which say
nothing about who the modeled recipients are: how many sit outside the federal
income and resource limits, and whether every eligible unit ends up receiving.
The shares of participating households USDA estimates above those limits, in
its 2026 proposed rule on categorical eligibility, are comparators here, not
fitted rows or a release gate (microcosm#647).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.calibrate.geography_constants import US_STATE_NUMERIC_FIPS_TO_POSTAL

__all__ = [
    "SNAP_CASELOAD_ASSET_THRESHOLD",
    "SNAP_CASELOAD_COMPOSITION_HOLDOUT_VARIABLES",
    "SNAP_SATURATION_THRESHOLD",
    "USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE_2026",
    "USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE_CITATION",
    "USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE_TITLE",
    "USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE_URL",
    "SnapFederalLimitBenchmark",
    "snap_caseload_composition_holdout_from_sim",
    "snap_caseload_composition_holdout_payload",
    "snap_caseload_composition_holdout_unavailable",
    "write_snap_caseload_composition_holdout",
]

_BENEFIT = "snap"
_ELIGIBLE = "is_snap_eligible"
_GROSS_TEST = "meets_snap_gross_income_test"
_NET_TEST = "meets_snap_net_income_test"
_ASSET_TEST = "meets_snap_asset_test"
_ASSETS = "snap_assets"
_TAKE_UP_FLAG = "takes_up_snap_if_eligible"
_REPORTED_RECEIPT = "receives_snap"
_SIZE = "snap_unit_size"
_STATE = "spm_unit_state_fips"
_WEIGHT = "weight"
#: The SPM-unit variables the release scorer computes, in request order.
SNAP_CASELOAD_COMPOSITION_HOLDOUT_VARIABLES: tuple[str, ...] = (
    _BENEFIT,
    _ELIGIBLE,
    _GROSS_TEST,
    _NET_TEST,
    _ASSET_TEST,
    _ASSETS,
    _TAKE_UP_FLAG,
    _REPORTED_RECEIPT,
    _SIZE,
    _STATE,
)
_BOOLEAN_VARIABLES = (
    _ELIGIBLE,
    _GROSS_TEST,
    _NET_TEST,
    _ASSET_TEST,
    _TAKE_UP_FLAG,
    _REPORTED_RECEIPT,
)

#: Reporting threshold for liquid assets among receiving units, in dollars.
SNAP_CASELOAD_ASSET_THRESHOLD = 3_000.0
#: A state is listed once one of its take-up shares reaches this value.
SNAP_SATURATION_THRESHOLD = 0.999

_INCOME_FAILING = "income_failing"
_ASSET_ONLY = "asset_only"
_WITHIN_LIMITS = "within_federal_limits"
_OUTSIDE_LIMITS = "outside_federal_limits"
_MODELED_RECIPIENT = "modeled_recipient"
_ELIGIBLE_NON_RECIPIENT = "eligible_non_recipient"
_NOT_ELIGIBLE = "not_eligible"
_TAKE_UP_SHARES = (
    "flagged_share_of_eligible_units",
    "receiving_share_of_eligible_units",
    "receiving_share_of_eligible_persons",
)
_DEFINITIONS = MappingProxyType(
    {
        "eligible": f"SPM units with {_ELIGIBLE} in December",
        "flagged": f"eligible units with {_TAKE_UP_FLAG}",
        "receiving": f"eligible units with a positive annual {_BENEFIT}",
        "persons": (
            f"units weighted by {_SIZE} in December: the unit's members SNAP "
            "does not exclude"
        ),
        "records": "units with a positive weight",
        "average_monthly_benefit": f"annual {_BENEFIT} divided by twelve",
        _INCOME_FAILING: (
            "receiving units that fail the federal gross or net income test, "
            "whatever their resource test result"
        ),
        _ASSET_ONLY: (
            "receiving units that pass both federal income tests and fail the "
            "federal resource test"
        ),
        _WITHIN_LIMITS: "receiving units that pass all three federal tests",
        "reported_receipt": (
            f"receiving units with {_REPORTED_RECEIPT} in December; a file that "
            f"stores no {_REPORTED_RECEIPT} reads every receiving unit as "
            "take_up_fill"
        ),
        "take_up_fill": "every other receiving unit",
        "reported_recipients": (
            f"units with {_REPORTED_RECEIPT} in December, whatever the engine "
            "makes of them"
        ),
        _MODELED_RECIPIENT: "reported units that are receiving units",
        _ELIGIBLE_NON_RECIPIENT: (
            "reported units eligible in December with no annual benefit"
        ),
        _NOT_ELIGIBLE: (
            "reported units not eligible in December, whatever their annual benefit"
        ),
        "benefit_outside_december_eligibility": (
            f"units with a positive annual {_BENEFIT} that are not eligible in "
            "December; counted in no share"
        ),
    }
)


@dataclass(frozen=True)
class SnapFederalLimitBenchmark:
    """One USDA estimate of participating households above a federal limit."""

    group: str
    label: str
    printed_page: int
    household_share: float
    participant_share: float | None = None
    fiscal_year: int | None = None
    households: int | None = None
    individuals: int | None = None


USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE_TITLE = (
    "Supplemental Nutrition Assistance Program: Reforming Categorical Eligibility"
)
USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE_CITATION = "91 FR 72454"
USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE_URL = (
    "https://www.federalregister.gov/documents/2026/11/10/2026-22936/"
    "supplemental-nutrition-assistance-program-reforming-categorical-eligibility"
)

#: USDA's estimates in the proposed rule's preamble (November 10, 2026). Fixed
#: held-out observations, never passed to the calibration target registry or to
#: a take-up stage. Page 72457 gives the fiscal year 2024 households with
#: incomes above the federal gross or net limits and with resources above the
#: federal limit; its note 25 counts a household failing both under income.
#: Page 72459 gives the combined shares of households and participants above
#: the limits, without a year.
USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE_2026 = MappingProxyType(
    {
        _INCOME_FAILING: SnapFederalLimitBenchmark(
            group=_INCOME_FAILING,
            label="Incomes above the federal gross or net income limits",
            printed_page=72457,
            household_share=0.054,
            fiscal_year=2024,
            households=1_200_000,
            individuals=2_000_000,
        ),
        _ASSET_ONLY: SnapFederalLimitBenchmark(
            group=_ASSET_ONLY,
            label="Resources above the federal limit, incomes within the limits",
            printed_page=72457,
            household_share=0.115,
            fiscal_year=2024,
            households=2_500_000,
            individuals=5_400_000,
        ),
        _OUTSIDE_LIMITS: SnapFederalLimitBenchmark(
            group=_OUTSIDE_LIMITS,
            label="Income or resources above the federal limits",
            printed_page=72459,
            household_share=0.169,
            participant_share=0.176,
        ),
    }
)


def snap_caseload_composition_holdout_unavailable(
    error: Exception, *, period: int, release_id: str | None = None
) -> dict[str, Any]:
    """An observational receipt when plan discovery or scoring is unavailable."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "unavailable",
        "period": int(period),
        "target_role": "validation",
        "enforced": False,
        "used_for_calibration": False,
        "source_url": USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE_URL,
        "error": f"{type(error).__name__}: {error}",
    }
    if release_id is not None:
        payload["release_id"] = release_id
    return payload


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _kish_effective_sample_size(weights: np.ndarray) -> float | None:
    """Kish effective sample size, ``(sum w)^2 / sum w^2``."""
    squared = float(np.square(weights).sum())
    return float(weights.sum()) ** 2 / squared if squared > 0 else None


def _boolean(values: np.ndarray, name: str) -> np.ndarray:
    if values.dtype == np.bool_:
        return values
    try:
        numeric = values.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be boolean.") from error
    if not np.isin(numeric, (0.0, 1.0)).all():
        # A count of months is not a December value: refuse it instead of
        # reading every positive count as true.
        raise ValueError(f"{name} must be boolean.")
    return numeric == 1.0


def _numeric(values: np.ndarray, name: str, *, nonnegative: bool = True) -> np.ndarray:
    try:
        numeric = values.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric.") from error
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} must be finite.")
    if nonnegative and np.any(numeric < 0):
        raise ValueError(f"{name} must be nonnegative.")
    return numeric


_STATE_FIPS_CODES = np.fromiter(US_STATE_NUMERIC_FIPS_TO_POSTAL, dtype=np.float64)


def _state_codes(values: np.ndarray) -> np.ndarray:
    numeric = _numeric(values, _STATE)
    # A household's code shared out among its units by head count is a
    # fraction of the state's code. A code that belongs to no state is refused
    # here. One that lands on another state's code cannot be told apart, so
    # the scorer must be asked for the unit's own state, as from_sim does.
    if not np.isin(numeric, _STATE_FIPS_CODES).all():
        raise ValueError(f"{_STATE} must be a state FIPS code.")
    return np.array([f"{int(code):02d}" for code in numeric], dtype="U2")


@dataclass(frozen=True)
class _Units:
    """Validated SPM-unit columns; a selection copies and writes no input."""

    weight: np.ndarray
    benefit: np.ndarray
    eligible: np.ndarray
    passes_income: np.ndarray
    passes_assets: np.ndarray
    assets: np.ndarray
    flagged: np.ndarray
    reported: np.ndarray
    size: np.ndarray
    state: np.ndarray

    def select(self, mask: np.ndarray) -> _Units:
        return _Units(
            **{name: getattr(self, name)[mask] for name in self.__dataclass_fields__}
        )


def _validated_units(units: pd.DataFrame) -> _Units:
    required = (_WEIGHT, *SNAP_CASELOAD_COMPOSITION_HOLDOUT_VARIABLES)
    missing = [column for column in required if column not in units.columns]
    if missing:
        raise ValueError(f"SNAP caseload composition needs column(s): {missing}.")
    columns = {name: units[name].to_numpy() for name in required}
    booleans = {name: _boolean(columns[name], name) for name in _BOOLEAN_VARIABLES}
    return _Units(
        weight=_numeric(columns[_WEIGHT], _WEIGHT),
        benefit=_numeric(columns[_BENEFIT], _BENEFIT),
        eligible=booleans[_ELIGIBLE],
        passes_income=booleans[_GROSS_TEST] & booleans[_NET_TEST],
        passes_assets=booleans[_ASSET_TEST],
        assets=_numeric(columns[_ASSETS], _ASSETS, nonnegative=False),
        flagged=booleans[_TAKE_UP_FLAG],
        reported=booleans[_REPORTED_RECEIPT],
        size=_numeric(columns[_SIZE], _SIZE),
        state=_state_codes(columns[_STATE]),
    )


def _records(weights: np.ndarray) -> int:
    return int(np.count_nonzero(weights > 0))


def _group_row(
    units: _Units,
    mask: np.ndarray,
    *,
    receiving_units: float,
    receiving_persons: float,
) -> dict[str, Any]:
    weights = units.weight[mask]
    weighted_units = float(weights.sum())
    weighted_persons = float((weights * units.size[mask]).sum())
    annual_benefits = float((weights * units.benefit[mask]).sum())
    return {
        "records": _records(weights),
        "weighted_units": weighted_units,
        "weighted_persons": weighted_persons,
        "share_of_receiving_units": _ratio(weighted_units, receiving_units),
        "share_of_receiving_persons": _ratio(weighted_persons, receiving_persons),
        "average_monthly_benefit": _ratio(annual_benefits / 12, weighted_units),
        "kish_effective_sample_size": _kish_effective_sample_size(weights),
    }


def _scope_row(units: _Units) -> dict[str, Any]:
    weights = units.weight
    persons = weights * units.size
    positive = units.benefit > 0
    receiving = units.eligible & positive
    outside = ~units.eligible & positive
    eligible_units = float(weights[units.eligible].sum())
    eligible_persons = float(persons[units.eligible].sum())
    flagged_units = float(weights[units.eligible & units.flagged].sum())
    receiving_weights = weights[receiving]
    receiving_units = float(receiving_weights.sum())
    receiving_persons = float(persons[receiving].sum())

    def group(mask: np.ndarray) -> dict[str, Any]:
        return _group_row(
            units,
            receiving & mask,
            receiving_units=receiving_units,
            receiving_persons=receiving_persons,
        )

    reported_units = float(weights[units.reported].sum())

    def reported(mask: np.ndarray) -> dict[str, Any]:
        selected = weights[units.reported & mask]
        weighted_units = float(selected.sum())
        return {
            "records": _records(selected),
            "weighted_units": weighted_units,
            "share_of_reported_units": _ratio(weighted_units, reported_units),
        }

    over_threshold = receiving & (units.assets > SNAP_CASELOAD_ASSET_THRESHOLD)
    return {
        "take_up": {
            "eligible_records": _records(weights[units.eligible]),
            "eligible_weighted_units": eligible_units,
            "eligible_weighted_persons": eligible_persons,
            "flagged_weighted_units": flagged_units,
            "receiving_weighted_units": receiving_units,
            "receiving_weighted_persons": receiving_persons,
            "flagged_share_of_eligible_units": _ratio(flagged_units, eligible_units),
            "receiving_share_of_eligible_units": _ratio(
                receiving_units, eligible_units
            ),
            "receiving_share_of_eligible_persons": _ratio(
                receiving_persons, eligible_persons
            ),
        },
        "receiving": {
            "records": _records(receiving_weights),
            "weighted_units": receiving_units,
            "weighted_persons": receiving_persons,
            "annual_benefits": float((weights * units.benefit)[receiving].sum()),
            "kish_effective_sample_size": _kish_effective_sample_size(
                receiving_weights
            ),
            "persons_per_unit": _ratio(receiving_persons, receiving_units),
            "single_person_unit_share": _ratio(
                float(weights[receiving & (units.size == 1)].sum()), receiving_units
            ),
            "share_with_snap_assets_over_threshold": _ratio(
                float(weights[over_threshold].sum()), receiving_units
            ),
        },
        "composition": {
            _INCOME_FAILING: group(~units.passes_income),
            _ASSET_ONLY: group(units.passes_income & ~units.passes_assets),
            _WITHIN_LIMITS: group(units.passes_income & units.passes_assets),
        },
        "caseload_source": {
            "reported_receipt": group(units.reported),
            "take_up_fill": group(~units.reported),
        },
        "reported_recipients": {
            "records": _records(weights[units.reported]),
            "weighted_units": reported_units,
            _MODELED_RECIPIENT: reported(receiving),
            _ELIGIBLE_NON_RECIPIENT: reported(units.eligible & ~positive),
            _NOT_ELIGIBLE: reported(~units.eligible),
        },
        "benefit_outside_december_eligibility": {
            "records": _records(weights[outside]),
            "weighted_units": float(weights[outside].sum()),
            "annual_benefits": float((weights * units.benefit)[outside].sum()),
        },
    }


def _percentage_points(share: float | None, benchmark: float | None) -> float | None:
    if share is None or benchmark is None:
        return None
    return 100 * (share - benchmark)


def _comparisons(national: dict[str, Any]) -> list[dict[str, Any]]:
    """The national composition beside USDA's shares; nothing is compared by state."""
    composition = national["composition"]
    receiving = national["receiving"]
    outside = (_INCOME_FAILING, _ASSET_ONLY)
    shares = {
        group: (
            composition[group]["share_of_receiving_units"],
            composition[group]["share_of_receiving_persons"],
        )
        for group in outside
    }
    shares[_OUTSIDE_LIMITS] = (
        _ratio(
            sum(composition[group]["weighted_units"] for group in outside),
            receiving["weighted_units"],
        ),
        _ratio(
            sum(composition[group]["weighted_persons"] for group in outside),
            receiving["weighted_persons"],
        ),
    )
    rows = []
    for group, benchmark in USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE_2026.items():
        unit_share, person_share = shares[group]
        rows.append(
            {
                "group": group,
                "label": benchmark.label,
                "share_of_receiving_units": unit_share,
                "share_of_receiving_persons": person_share,
                "benchmark_household_share": benchmark.household_share,
                "benchmark_participant_share": benchmark.participant_share,
                "benchmark_households": benchmark.households,
                "benchmark_individuals": benchmark.individuals,
                "benchmark_fiscal_year": benchmark.fiscal_year,
                "benchmark_printed_page": benchmark.printed_page,
                "difference_percentage_points": _percentage_points(
                    unit_share, benchmark.household_share
                ),
                "person_difference_percentage_points": _percentage_points(
                    person_share, benchmark.participant_share
                ),
            }
        )
    return rows


def _states_at_or_above_threshold(states: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        share: [
            row["state_fips"]
            for row in states
            if row["take_up"][share] is not None
            and row["take_up"][share] >= SNAP_SATURATION_THRESHOLD
        ]
        for share in _TAKE_UP_SHARES
    }


def snap_caseload_composition_holdout_payload(
    units: pd.DataFrame, *, period: int, release_id: str | None = None
) -> dict[str, Any]:
    """SNAP recipients by federal test result, and take-up among the eligible.

    ``units`` holds one row per SPM unit: its calibrated ``weight`` and the
    ``SNAP_CASELOAD_COMPOSITION_HOLDOUT_VARIABLES`` as the written-H5 scorer
    returns them for the period. The engine reports a monthly true-or-false
    variable for a year as its December value, so eligibility, the income
    tests and reported receipt are December's; the benefit is the year's
    total. A unit receives when it is eligible in December and its annual
    benefit is positive. A benefit paid to a unit not eligible in December is
    reported apart and counted in no share.

    Persons are each unit's SNAP unit size, which leaves out the members SNAP
    excludes. A unit's state and weight are the engine's own values for the
    unit, never a household value shared out among its units by head count; a
    code that belongs to no state is refused. Units that report receipt are
    also split by what the engine makes of them: receiving, eligible with no
    benefit, or not eligible.

    Each receiving unit falls in exactly one composition group. A unit failing
    both an income test and the resource test counts as income-failing, the
    convention of the USDA shares. A share or average with no weight behind it
    is null, not zero. The function reads its inputs and never writes them.
    """
    validated = _validated_units(units)
    national = _scope_row(validated)
    states = [
        {
            "state_fips": str(state),
            **_scope_row(validated.select(validated.state == state)),
        }
        for state in np.unique(validated.state)
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "available",
        "period": int(period),
        "target_role": "validation",
        "enforced": False,
        "used_for_calibration": False,
        "measure": (
            "SNAP recipients by federal income and resource test result, and "
            "take-up among eligible units"
        ),
        "weight_basis": "SPM unit's calibrated household_weight",
        "monthly_values_read_in": "December",
        "definitions": dict(_DEFINITIONS),
        "asset_threshold": SNAP_CASELOAD_ASSET_THRESHOLD,
        "saturation_threshold": SNAP_SATURATION_THRESHOLD,
        "reported_receipt_present": bool(validated.reported.any()),
        "benchmark": {
            "source": USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE_TITLE,
            "citation": USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE_CITATION,
            "source_url": USDA_CATEGORICAL_ELIGIBILITY_PROPOSED_RULE_URL,
            "published": "2026-11-10",
            "population": "participating SNAP households",
            "double_counting": (
                "A household failing both the income and resource tests is "
                "counted under income (note 25)."
            ),
            "resource_data": (
                "Survey of Income and Program Participation, December 2019 (note 14)"
            ),
        },
        "comparability": (
            "USDA estimates shares of participating SNAP households, by limit "
            "for fiscal year 2024, with resources estimated from the December "
            "2019 Survey of Income and Program Participation. The model counts "
            f"SPM units in {int(period)} that are eligible in December and "
            "carry a positive annual benefit, and reads each monthly test in "
            "December. The model's SNAP household is the SPM unit less the "
            "members SNAP excludes, which need not be the household a State "
            "agency forms, and the model spreads annual income evenly over the "
            "months. This is a held-out diagnostic, not an exact population or "
            "period match, a calibration target, or a release gate."
        ),
        "national": national,
        "states": states,
        "states_at_or_above_saturation_threshold": _states_at_or_above_threshold(
            states
        ),
        "comparisons": _comparisons(national),
    }
    if release_id is not None:
        payload["release_id"] = release_id
    return payload


def snap_caseload_composition_holdout_from_sim(
    simulation: Any, *, period: int, release_id: str | None = None
) -> dict[str, Any]:
    """Measure aligned SPM-unit results from the household-batched H5 scorer."""
    results = {
        variable: simulation.calculate(variable, period)
        for variable in SNAP_CASELOAD_COMPOSITION_HOLDOUT_VARIABLES
    }
    units = pd.DataFrame(
        {variable: np.asarray(result) for variable, result in results.items()}
    )
    units[_WEIGHT] = np.asarray(results[_BENEFIT].weights)
    return snap_caseload_composition_holdout_payload(
        units, period=period, release_id=release_id
    )


def write_snap_caseload_composition_holdout(
    payload: dict[str, Any], path: Path | str
) -> Path:
    path = Path(path)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return path
