"""Archived PUF Section 199A input family for the US build.

The retired eCPS PUF pipeline used a pinned, seeded QBI simulation to create
source-level qualification flags, SSTB classification and self-employment
splits, allocable W-2 wages / UBIA, and qualified REIT/PTP and BDC income.
Release 1.8.0 did not physically carry that surface: a later retired loader
generated it while mutating the downloaded file. Populace now runs the
versioned source simulation in :mod:`qbi_simulation` before the shared weighted
PUF QRF places the leaves on the PUF support channel.

This module restores the cross-column identities after imputation. In
particular, the archived PUF model is all-or-nothing at tax-record grain: the
SSTB flag routes the entire predicted Schedule C amount to the SSTB leaf and
duplicates the total W-2 wage and UBIA pools into their SSTB-allocable leaves.
The base W-2/UBIA leaves remain total pools, not non-SSTB complements.
PolicyEngine-US owns the QBI deduction formulas and statutory limits.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import files
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.frame import US_SCHEMA, Frame

if TYPE_CHECKING:
    from populace.build.us_runtime.qbi_simulation import (
        QbiSimulationAssumptionsV2,
        SstbCrosswalk,
    )

__all__ = [
    "QBI_ARCHIVED_ASSUMPTIONS_URL",
    "QBI_ARCHIVED_CLONE_URL",
    "QBI_ARCHIVED_DERIVATION_URL",
    "QBI_ARCHIVED_EXPORT_URL",
    "QBI_ARCHIVED_IMPUTATION_URL",
    "QBI_ARCHIVED_PUF_ARTIFACT_URL",
    "QBI_ARCHIVED_SIMULATION_URL",
    "US_QBI_BOOLEAN_OUTPUT_COLUMNS",
    "US_QBI_NONCONSTANT_PERSON_COLUMNS",
    "US_QBI_NONNEGATIVE_OUTPUT_COLUMNS",
    "US_QBI_OUTPUT_COLUMNS",
    "US_QBI_STAGE_NAME",
    "US_QBI_V2_HOST_REQUIRED_SOURCE_COLUMNS",
    "us_qbi_inputs_signal_gate",
    "us_qbi_inputs_stage_spec",
    "us_qbi_inputs_summary",
    "with_host_sstb_classification",
    "with_us_qbi_input_reconciliation",
]

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_COMMIT = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/{_ARCHIVED_COMMIT}/"
    "policyengine_" + "us_data/"
)
QBI_ARCHIVED_DERIVATION_URL = _ARCHIVED_ROOT + "datasets/puf/puf.py#L748-L787"
QBI_ARCHIVED_SIMULATION_URL = _ARCHIVED_ROOT + "datasets/puf/puf.py#L105-L405"
QBI_ARCHIVED_EXPORT_URL = _ARCHIVED_ROOT + "datasets/puf/puf.py#L860-L879"
QBI_ARCHIVED_ASSUMPTIONS_URL = (
    _ARCHIVED_ROOT + "datasets/puf/qbi_assumptions.yaml#L1-L118"
)
QBI_ARCHIVED_IMPUTATION_URL = _ARCHIVED_ROOT + "calibration/puf_impute.py#L99-L198"
QBI_ARCHIVED_CLONE_URL = _ARCHIVED_ROOT + "calibration/puf_impute.py#L513-L685"
QBI_ARCHIVED_PUF_ARTIFACT_URL = _ARCHIVED_ROOT + "datasets/puf/puf.py#L1655-L1660"

US_QBI_STAGE_NAME = "puf_tax_detail"

_GENERAL_QUALIFICATION_FLAGS: tuple[str, ...] = (
    "estate_income_would_be_qualified",
    "farm_operations_income_would_be_qualified",
    "farm_rent_income_would_be_qualified",
    "partnership_s_corp_income_would_be_qualified",
    "rental_income_would_be_qualified",
    "self_employment_income_would_be_qualified",
)
_SSTB_QUALIFICATION_FLAG = "sstb_self_employment_income_would_be_qualified"
US_QBI_BOOLEAN_OUTPUT_COLUMNS: tuple[str, ...] = (
    *_GENERAL_QUALIFICATION_FLAGS,
    _SSTB_QUALIFICATION_FLAG,
    # Keep the classifier last so the chained QRF can condition the SSTB draw
    # on the qualification flags it must agree with.
    "business_is_sstb",
)
US_QBI_NONNEGATIVE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "qualified_bdc_income",
    "qualified_reit_and_ptp_income",
    "sstb_unadjusted_basis_qualified_property",
    "sstb_w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "w2_wages_from_qualified_business",
)
US_QBI_OUTPUT_COLUMNS: tuple[str, ...] = (
    *US_QBI_BOOLEAN_OUTPUT_COLUMNS,
    "qualified_bdc_income",
    "qualified_reit_and_ptp_income",
    "sstb_self_employment_income_before_lsr",
    "sstb_unadjusted_basis_qualified_property",
    "sstb_w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "w2_wages_from_qualified_business",
)
US_QBI_NONCONSTANT_PERSON_COLUMNS = US_QBI_OUTPUT_COLUMNS
US_QBI_V2_HOST_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "AGI",
    "PEIOOCC",
)

_PERSON_SUPPORT_CHANNEL_COLUMN = "person_support_channel"
_BASE_ASEC_SUPPORT_CHANNEL = "asec"
_SELF_EMPLOYMENT_COLUMN = "self_employment_income_before_lsr"
_SSTB_SELF_EMPLOYMENT_COLUMN = "sstb_self_employment_income_before_lsr"
_BOOLEAN_SHARE_BANDS: dict[str, tuple[float, float]] = {
    "business_is_sstb": (0.001, 0.25),
    **{column: (0.01, 0.9999) for column in _GENERAL_QUALIFICATION_FLAGS},
    _SSTB_QUALIFICATION_FLAG: (0.0001, 0.25),
}
_NUMERIC_NONZERO_SHARE_BANDS: dict[str, tuple[float, float]] = {
    "qualified_bdc_income": (0.0001, 0.15),
    "qualified_reit_and_ptp_income": (0.001, 0.35),
    _SSTB_SELF_EMPLOYMENT_COLUMN: (0.0001, 0.25),
    "sstb_unadjusted_basis_qualified_property": (0.0001, 0.25),
    "sstb_w2_wages_from_qualified_business": (0.0001, 0.20),
    "unadjusted_basis_qualified_property": (0.001, 0.45),
    "w2_wages_from_qualified_business": (0.001, 0.35),
}
_INVARIANT_ATOL = 1e-8


def us_qbi_inputs_stage_spec() -> SourceStageSpec:
    """Load and validate the shared PUF stage's QBI output contract."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    spec = manifest.stage_map()[US_QBI_STAGE_NAME]
    missing = sorted(set(US_QBI_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_QBI_STAGE_NAME!r} manifest stage does not declare QBI "
            f"output(s) {missing}."
        )
    return spec


def _numeric(person: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(person[column], errors="coerce").to_numpy(dtype=np.float64)
    nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    if nonfinite:
        raise ValueError(
            f"US QBI input {column!r} contains {nonfinite} nonfinite value(s)."
        )
    return values


def _optional_numeric(person: pd.DataFrame, column: str) -> np.ndarray:
    if column not in person:
        return np.zeros(len(person), dtype=np.float64)
    return _numeric(person, column)


def with_us_qbi_input_reconciliation(frame: Frame) -> Frame:
    """Restore archived all-or-nothing SSTB split identities after PUF QRF."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US QBI input reconciliation requires the US schema.")
    person = frame.table("person")
    required = {*US_QBI_OUTPUT_COLUMNS, _SELF_EMPLOYMENT_COLUMN}
    missing = sorted(required - set(person.columns))
    if missing:
        raise ValueError(
            "US QBI input reconciliation requires PUF-imputed source column(s): "
            f"{missing}."
        )

    result = person.copy(deep=True)
    asec_mask = np.zeros(len(result), dtype=bool)
    if _PERSON_SUPPORT_CHANNEL_COLUMN in result:
        asec_mask = (
            result[_PERSON_SUPPORT_CHANNEL_COLUMN].to_numpy(dtype=object)
            == _BASE_ASEC_SUPPORT_CHANNEL
        )

    flags: dict[str, np.ndarray] = {}
    for column in US_QBI_BOOLEAN_OUTPUT_COLUMNS:
        values = _numeric(result, column)
        flags[column] = values > 0.0
    # Populace's base ASEC support has no observations from the PUF QBI source
    # simulation. Deliberately preserve PolicyEngine's ordinary qualification
    # defaults on that channel and do not invent an SSTB classification there.
    # This is a hermetic two-channel choice, not a claim about the retired
    # clone implementation (which imputed absent PUF variables onto both halves).
    for column in _GENERAL_QUALIFICATION_FLAGS:
        flags[column][asec_mask] = True
    flags["business_is_sstb"][asec_mask] = False
    flags[_SSTB_QUALIFICATION_FLAG][asec_mask] = False

    non_sstb_self_employment = _numeric(result, _SELF_EMPLOYMENT_COLUMN)
    sstb_self_employment = _numeric(result, _SSTB_SELF_EMPLOYMENT_COLUMN)
    total_self_employment = non_sstb_self_employment + sstb_self_employment

    base_self_employment_qualified = (
        flags["self_employment_income_would_be_qualified"]
        | flags[_SSTB_QUALIFICATION_FLAG]
    )
    partnership_s_corp_income = _optional_numeric(
        result, "partnership_income"
    ) + _optional_numeric(result, "s_corp_income")
    estate_income = _optional_numeric(result, "estate_income")
    has_positive_qualified_mapped_source = (
        ((total_self_employment > 0.0) & base_self_employment_qualified)
        | (
            (partnership_s_corp_income > 0.0)
            & flags["partnership_s_corp_income_would_be_qualified"]
        )
        | ((estate_income > 0.0) & flags["estate_income_would_be_qualified"])
    )
    business_is_sstb = flags["business_is_sstb"] & has_positive_qualified_mapped_source
    return _with_qbi_routing(
        frame,
        result=result,
        flags=flags,
        business_is_sstb=business_is_sstb,
        base_self_employment_qualified=base_self_employment_qualified,
        total_self_employment=total_self_employment,
        partnership_s_corp_income=partnership_s_corp_income,
    )


def with_host_sstb_classification(
    frame: Frame,
    *,
    qbi_simulation_version: int,
    assumptions: QbiSimulationAssumptionsV2 | None = None,
    sstb_crosswalk: SstbCrosswalk | Mapping[str, Any] | None = None,
) -> Frame:
    """Apply v2 qualification derivations and host-conditioned SSTB routing.

    This pure post-QRF transform classifies positive qualified Schedule C
    income from host industry when configured, falling back to detailed
    occupation. The live crosswalk supplies every mapped probability; codes
    absent from the relevant map are non-SSTB with probability zero.
    Records with no positive qualified Schedule C income but positive qualified
    partnership/S-corporation or estate income use the versioned AGI-band
    passive prior.
    """

    from populace.build.us_runtime.qbi_simulation import (
        QBI_SIMULATION_V2,
        QbiSimulationAssumptionsV2,
        load_qbi_simulation_assumptions,
        resolve_sstb_crosswalk,
    )

    if qbi_simulation_version != QBI_SIMULATION_V2:
        raise ValueError("Host SSTB classification requires qbi_simulation_version=2.")
    if frame.schema != US_SCHEMA:
        raise ValueError("Host SSTB classification requires the US schema.")
    resolved_assumptions = assumptions or load_qbi_simulation_assumptions(
        qbi_simulation_version
    )
    if not isinstance(resolved_assumptions, QbiSimulationAssumptionsV2):
        raise TypeError("Host SSTB classification requires v2 assumptions.")
    resolved_assumptions.validate()
    crosswalk = resolve_sstb_crosswalk(
        resolved_assumptions,
        sstb_crosswalk,
    )
    classification = resolved_assumptions.sstb_classification

    person = frame.table("person")
    prior_flag_columns = {
        f"{derivation.source}_would_be_qualified"
        for derivation in resolved_assumptions.qualification_derivations
        if derivation.mode == "prior"
    }
    required = {
        _SELF_EMPLOYMENT_COLUMN,
        _SSTB_SELF_EMPLOYMENT_COLUMN,
        "qualified_bdc_income",
        "qualified_reit_and_ptp_income",
        "unadjusted_basis_qualified_property",
        "w2_wages_from_qualified_business",
        classification.occupation_column,
        classification.agi_column,
        *prior_flag_columns,
    }
    if classification.industry_column is not None:
        required.add(classification.industry_column)
    missing = sorted(required - set(person.columns))
    if missing:
        raise ValueError(
            f"Host SSTB classification requires post-QRF source column(s): {missing}."
        )

    result = person.copy(deep=True)
    total_self_employment = _numeric(
        result,
        _SELF_EMPLOYMENT_COLUMN,
    ) + _numeric(result, _SSTB_SELF_EMPLOYMENT_COLUMN)
    partnership_s_corp_income = _optional_numeric(
        result,
        "partnership_income",
    ) + _optional_numeric(result, "s_corp_income")
    source_values = {
        "self_employment_income": total_self_employment,
        "farm_operations_income": _optional_numeric(
            result,
            "farm_operations_income",
        ),
        "farm_rent_income": _optional_numeric(result, "farm_rent_income"),
        "rental_income": _optional_numeric(result, "rental_income"),
        "estate_income": _optional_numeric(result, "estate_income"),
        "partnership_s_corp_income": partnership_s_corp_income,
    }

    flags: dict[str, np.ndarray] = {}
    for derivation in resolved_assumptions.qualification_derivations:
        column = f"{derivation.source}_would_be_qualified"
        if derivation.mode == "derived":
            flags[column] = source_values[derivation.source] != 0.0
        else:
            flags[column] = _numeric(result, column) > 0.0
    base_self_employment_qualified = flags["self_employment_income_would_be_qualified"]

    host_probabilities = _host_sstb_probabilities(
        result,
        occupation_column=classification.occupation_column,
        occupation_mapping=crosswalk.mapping_for("occupation"),
        industry_column=classification.industry_column,
        industry_mapping=crosswalk.mapping_for("industry"),
    )
    rng = np.random.default_rng(resolved_assumptions.sstb_seed)
    schedule_c_qualified = (
        total_self_employment > 0.0
    ) & base_self_employment_qualified
    schedule_c_sstb = rng.random(len(result)) < host_probabilities

    estate_income = source_values["estate_income"]
    has_positive_qualified_passive_income = (
        (partnership_s_corp_income > 0.0)
        & flags["partnership_s_corp_income_would_be_qualified"]
    ) | ((estate_income > 0.0) & flags["estate_income_would_be_qualified"])
    passive_only = ~schedule_c_qualified & has_positive_qualified_passive_income
    passive_probabilities = _passive_sstb_probabilities(
        _numeric(result, classification.agi_column),
        resolved_assumptions,
    )
    passive_draw = rng.random(len(result)) < passive_probabilities
    business_is_sstb = (schedule_c_qualified & schedule_c_sstb) | (
        passive_only & passive_draw
    )
    flags["business_is_sstb"] = business_is_sstb
    flags[_SSTB_QUALIFICATION_FLAG] = np.zeros(len(result), dtype=bool)

    routed = _with_qbi_routing(
        frame,
        result=result,
        flags=flags,
        business_is_sstb=business_is_sstb,
        base_self_employment_qualified=base_self_employment_qualified,
        total_self_employment=total_self_employment,
        partnership_s_corp_income=partnership_s_corp_income,
    )
    invariant_failures = {
        name: count
        for name, count in us_qbi_inputs_summary(routed)["invariants"].items()
        if count
    }
    if invariant_failures:
        raise AssertionError(
            "Host SSTB classification violated QBI reconciliation invariants: "
            f"{invariant_failures}."
        )
    return routed


def _host_sstb_probabilities(
    person: pd.DataFrame,
    *,
    occupation_column: str,
    occupation_mapping: Mapping[int, float],
    industry_column: str | None,
    industry_mapping: Mapping[int, float],
) -> np.ndarray:
    resolved = np.full(len(person), None, dtype=object)
    unresolved = np.ones(len(person), dtype=bool)
    if industry_column is not None:
        industry = _mapped_sstb_values(
            person[industry_column],
            industry_mapping,
        )
        observed = pd.notna(industry)
        resolved[observed] = industry[observed]
        unresolved &= ~observed
    occupation = _mapped_sstb_values(
        person[occupation_column],
        occupation_mapping,
    )
    observed = unresolved & pd.notna(occupation)
    resolved[observed] = occupation[observed]
    resolved[pd.isna(resolved)] = 0.0
    return resolved.astype(np.float64)


def _mapped_sstb_values(
    values: pd.Series,
    mapping: Mapping[int, float],
) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.map(mapping).to_numpy(dtype=object)


def _passive_sstb_probabilities(
    agi: np.ndarray,
    assumptions: QbiSimulationAssumptionsV2,
) -> np.ndarray:
    result = np.zeros(len(agi), dtype=np.float64)
    assigned = np.zeros(len(agi), dtype=bool)
    for band in assumptions.sstb_classification.passive_passthrough_sstb_prior_by_agi:
        in_band = (agi >= band.lower) & (agi < band.upper)
        result[in_band] = band.probability
        assigned |= in_band
    if not np.all(assigned):
        raise ValueError("QBI v2 passive SSTB AGI bands did not cover every record.")
    return result


def _with_qbi_routing(
    frame: Frame,
    *,
    result: pd.DataFrame,
    flags: dict[str, np.ndarray],
    business_is_sstb: np.ndarray,
    base_self_employment_qualified: np.ndarray,
    total_self_employment: np.ndarray,
    partnership_s_corp_income: np.ndarray,
) -> Frame:
    flags["business_is_sstb"] = business_is_sstb
    flags["self_employment_income_would_be_qualified"] = (
        ~business_is_sstb & base_self_employment_qualified
    )
    flags[_SSTB_QUALIFICATION_FLAG] = business_is_sstb & base_self_employment_qualified

    result[_SELF_EMPLOYMENT_COLUMN] = np.where(
        business_is_sstb,
        0.0,
        total_self_employment,
    )
    result[_SSTB_SELF_EMPLOYMENT_COLUMN] = np.where(
        business_is_sstb,
        total_self_employment,
        0.0,
    )

    w2_wages = _numeric(result, "w2_wages_from_qualified_business")
    ubia = _numeric(result, "unadjusted_basis_qualified_property")
    result["sstb_w2_wages_from_qualified_business"] = np.where(
        business_is_sstb,
        w2_wages,
        0.0,
    )
    result["sstb_unadjusted_basis_qualified_property"] = np.where(
        business_is_sstb,
        ubia,
        0.0,
    )
    non_qualified_dividends = np.maximum(
        _optional_numeric(result, "non_qualified_dividend_income"),
        0.0,
    )
    result["qualified_bdc_income"] = np.minimum(
        _numeric(result, "qualified_bdc_income"),
        non_qualified_dividends,
    )
    result["qualified_reit_and_ptp_income"] = np.minimum(
        _numeric(result, "qualified_reit_and_ptp_income"),
        non_qualified_dividends + np.maximum(partnership_s_corp_income, 0.0),
    )
    for column, values in flags.items():
        result[column] = values.astype(bool)

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = result
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def us_qbi_inputs_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal, validity, and split-identity diagnostics."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    columns: dict[str, dict[str, object]] = {}
    for column in US_QBI_OUTPUT_COLUMNS:
        values = pd.to_numeric(person[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite = np.isfinite(values)
        nonzero = finite & (values != 0.0)
        nonzero_share = (
            float(weights[nonzero].sum()) / total_weight if total_weight > 0.0 else 0.0
        )
        band = (
            _BOOLEAN_SHARE_BANDS[column]
            if column in _BOOLEAN_SHARE_BANDS
            else _NUMERIC_NONZERO_SHARE_BANDS[column]
        )
        columns[column] = {
            "nonzero_share": nonzero_share,
            "nonzero_share_band": list(band),
            "nonfinite": int(np.count_nonzero(~finite)),
            "negative": int(np.count_nonzero(finite & (values < 0.0))),
        }

    business = person["business_is_sstb"].astype(bool).to_numpy()
    self_employment = pd.to_numeric(
        person[_SELF_EMPLOYMENT_COLUMN], errors="coerce"
    ).to_numpy(dtype=np.float64)
    sstb_self_employment = pd.to_numeric(
        person[_SSTB_SELF_EMPLOYMENT_COLUMN], errors="coerce"
    ).to_numpy(dtype=np.float64)
    w2 = pd.to_numeric(
        person["w2_wages_from_qualified_business"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    sstb_w2 = pd.to_numeric(
        person["sstb_w2_wages_from_qualified_business"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    ubia = pd.to_numeric(
        person["unadjusted_basis_qualified_property"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    sstb_ubia = pd.to_numeric(
        person["sstb_unadjusted_basis_qualified_property"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    self_qualified = (
        person["self_employment_income_would_be_qualified"].astype(bool).to_numpy()
    )
    sstb_qualified = person[_SSTB_QUALIFICATION_FLAG].astype(bool).to_numpy()
    non_qualified_dividends = np.maximum(
        _optional_numeric(person, "non_qualified_dividend_income"),
        0.0,
    )
    partnership_s_corp_income = _optional_numeric(
        person, "partnership_income"
    ) + _optional_numeric(person, "s_corp_income")
    qualified_bdc_income = _numeric(person, "qualified_bdc_income")
    qualified_reit_and_ptp_income = _numeric(person, "qualified_reit_and_ptp_income")
    invariants = {
        "sstb_rows_with_non_sstb_income": int(
            np.count_nonzero(business & ~np.isclose(self_employment, 0.0))
        ),
        "non_sstb_rows_with_sstb_income": int(
            np.count_nonzero(~business & ~np.isclose(sstb_self_employment, 0.0))
        ),
        "sstb_w2_split_mismatches": int(
            np.count_nonzero(
                ~np.isclose(sstb_w2, np.where(business, w2, 0.0), atol=_INVARIANT_ATOL)
            )
        ),
        "sstb_ubia_split_mismatches": int(
            np.count_nonzero(
                ~np.isclose(
                    sstb_ubia,
                    np.where(business, ubia, 0.0),
                    atol=_INVARIANT_ATOL,
                )
            )
        ),
        "self_employment_qualification_overlap": int(
            np.count_nonzero(self_qualified & sstb_qualified)
        ),
        "sstb_qualification_route_mismatches": int(
            np.count_nonzero(sstb_qualified & ~business)
        ),
        "non_sstb_qualification_route_mismatches": int(
            np.count_nonzero(self_qualified & business)
        ),
        "qualified_bdc_exposure_mismatches": int(
            np.count_nonzero(
                qualified_bdc_income > non_qualified_dividends + _INVARIANT_ATOL
            )
        ),
        "qualified_reit_ptp_exposure_mismatches": int(
            np.count_nonzero(
                qualified_reit_and_ptp_income
                > non_qualified_dividends
                + np.maximum(partnership_s_corp_income, 0.0)
                + _INVARIANT_ATOL
            )
        ),
    }
    return {"columns": columns, "invariants": invariants}


def us_qbi_inputs_signal_gate(frame: Frame) -> GateResult:
    """Require nondefault QBI signal and archived SSTB split identities."""

    person = frame.table("person")
    missing = sorted(
        {*US_QBI_OUTPUT_COLUMNS, _SELF_EMPLOYMENT_COLUMN} - set(person.columns)
    )
    if missing:
        return GateResult(
            name="qbi_inputs_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_qbi_inputs_summary(frame)
    failures: list[str] = []
    for column, details in summary["columns"].items():
        if details["nonfinite"]:
            failures.append(
                f"{column}: {int(details['nonfinite'])} nonfinite value(s)."
            )
        if column in US_QBI_NONNEGATIVE_OUTPUT_COLUMNS and details["negative"]:
            failures.append(f"{column}: {int(details['negative'])} negative value(s).")
        share = float(details["nonzero_share"])
        low, high = details["nonzero_share_band"]
        if not (low <= share <= high):
            failures.append(
                f"{column}: nonzero share {share:.6f} outside plausibility "
                f"band [{low}, {high}]."
            )
    for name, count in summary["invariants"].items():
        if count:
            failures.append(f"{name}: {int(count)} violating row(s).")
    return GateResult(
        name="qbi_inputs_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
