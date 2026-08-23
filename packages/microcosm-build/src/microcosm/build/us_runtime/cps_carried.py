"""CPS ASEC variables carried into PolicyEngine leaf inputs.

This stage translates raw CPS ASEC value columns already present on the
source spine into PolicyEngine input leaves needed by downstream donor stages.
It intentionally does not emit PolicyEngine formula-owned totals such as
``dividend_income`` or ``social_security``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.alimony import (
    US_ASEC_OTHER_INCOME_OUTPUT_COLUMNS,
    derive_us_alimony_from_asec,
)
from microcosm.build.us_runtime.public_assistance_type_source import (
    PAW_TYPE_TANF_CODES,
    fill_asec_public_assistance_type_source,
)
from microcosm.frame import US_SCHEMA, Frame

__all__ = [
    "CPS_CARRIED_FORMULA_OWNED_COLUMNS",
    "CPS_CARRIED_PERSON_INPUTS",
    "CPS_CARRIED_SPM_UNIT_INPUTS",
    "CPS_REPORTED_SNAP_RAW_COLUMN",
    "CPS_REPORTED_TANF_AMOUNT_RAW_COLUMN",
    "CPS_REPORTED_TANF_TYPE_RAW_COLUMN",
    "CPS_REPORTED_WIC_RAW_COLUMN",
    "US_REPORTED_COVERAGE_PERSON_INPUTS",
    "US_REPORTED_COVERAGE_VINTAGE_GATE_MIN_ROWS",
    "WIC_CARRIER_ADJUDICATION_URL",
    "derive_us_cps_carried_inputs",
    "reported_snap_receipt_by_spm_unit",
    "reported_tanf_enrollment_by_spm_unit",
    "reported_wic_receipt_carrier",
    "us_reported_coverage_vintage_signal_gate",
]

TAXABLE_INTEREST_FRACTION = 0.680
QUALIFIED_DIVIDEND_FRACTION = 0.448
TAXABLE_PENSION_FRACTION = 0.590
LONG_TERM_CAPITAL_GAIN_FRACTION = 0.880
CPS_REPORTED_SNAP_RAW_COLUMN = "SPM_SNAPSUB"
CPS_REPORTED_TANF_AMOUNT_RAW_COLUMN = "PAW_VAL"
CPS_REPORTED_TANF_TYPE_RAW_COLUMN = "PAW_TYP"
CPS_REPORTED_WIC_RAW_COLUMN = "WICYN"
WIC_CARRIER_ADJUDICATION_URL = (
    "https://github.com/PolicyEngine/microcosm/issues/591#issuecomment-5160668979"
)

# The nine reported-coverage person inputs _fill_health_coverage_inputs derives
# from the ASEC NOW_* at-interview recodes (microcosm #720).
US_REPORTED_COVERAGE_PERSON_INPUTS: tuple[str, ...] = (
    "has_champva_health_coverage_at_interview",
    "has_esi",
    "has_indian_health_service_coverage_at_interview",
    "has_marketplace_health_coverage_at_interview",
    "has_medicaid_health_coverage_at_interview",
    "has_non_marketplace_direct_purchase_health_coverage_at_interview",
    "has_other_means_tested_health_coverage_at_interview",
    "has_tricare_health_coverage_at_interview",
    "has_va_health_coverage_at_interview",
)

# Below this row count a vintage is a smoke pool and zero reporters for a rare
# flag is sampling noise; at or above it, a zero is a missing-source symptom
# (the rarest flag, CHAMPVA at ~0.3% of persons, is expected ~15 times in
# 5,000 rows).
US_REPORTED_COVERAGE_VINTAGE_GATE_MIN_ROWS = 5_000

CPS_CARRIED_FORMULA_OWNED_COLUMNS = frozenset(
    {
        "capital_gains",
        "dividend_income",
        "employment_income",
        "interest_income",
        "long_term_capital_gains",
        "ordinary_dividend_income",
        "self_employment_income",
        "social_security",
        "taxable_pension_income",
    }
)

CPS_CARRIED_PERSON_INPUTS = frozenset(
    {
        "age",
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
        "taxable_interest_income",
        "qualified_dividend_income",
        "non_qualified_dividend_income",
        "short_term_capital_gains",
        "long_term_capital_gains_before_response",
        "social_security_retirement",
        "social_security_disability",
        "social_security_survivors",
        "social_security_dependents",
        "taxable_private_pension_income",
        "tax_exempt_private_pension_income",
        "taxable_ira_distributions",
        "health_insurance_premiums_without_medicare_part_b",
        "other_medical_expenses",
        "over_the_counter_health_expenses",
        "receives_wic",
        "rental_income",
        "farm_operations_income",
        "has_champva_health_coverage_at_interview",
        "has_esi",
        "has_indian_health_service_coverage_at_interview",
        "has_marketplace_health_coverage_at_interview",
        "has_medicaid_health_coverage_at_interview",
        "has_non_marketplace_direct_purchase_health_coverage_at_interview",
        "has_other_means_tested_health_coverage_at_interview",
        "has_tricare_health_coverage_at_interview",
        "has_va_health_coverage_at_interview",
        "is_female",
        "unemployment_compensation",
        *US_ASEC_OTHER_INCOME_OUTPUT_COLUMNS,
    }
)

CPS_CARRIED_SPM_UNIT_INPUTS = frozenset(
    {
        "is_tanf_enrolled",
        "receives_snap",
        "spm_unit_pre_subsidy_childcare_expenses",
    }
)


def derive_us_cps_carried_inputs(
    frame: Frame,
    *,
    public_assistance_type_source: pd.DataFrame | None = None,
) -> Frame:
    """Carry raw CPS ASEC values onto PE input leaves.

    Existing leaf input columns are preserved, making the transform idempotent.
    The transform refuses to run on a non-US frame and never creates
    formula-owned aggregate variables.

    ``PAW_VAL``, ``SPM_SNAPSUB``, and ``WICYN`` are annual reported facts,
    while the engine's ``is_tanf_enrolled``, ``receives_snap``, and
    ``receives_wic`` leaves are monthly booleans. An annual receipt report is
    therefore broadcast to all 12 modeled months. The annual source cannot
    reveal entry or exit timing, and ``PAW_VAL > 0`` misses enrolled TANF
    units receiving zero dollars, including sanctioned cases.

    ``is_tanf_enrolled`` is additionally gated on ``PAW_TYP`` because
    ``PAW_VAL`` alone conflates TANF with other cash welfare; see
    :func:`reported_tanf_enrollment_by_spm_unit`. When the frame does not
    already carry ``PAW_TYP`` (the frozen census_cps inputs never did),
    callers with positive ``PAW_VAL`` rows must pass
    ``public_assistance_type_source`` — the pinned sidecar loaded by
    :func:`~microcosm.build.us_runtime.public_assistance_type_source.load_asec_public_assistance_type_sources`
    — which is joined transiently and never lands on the output frame.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("CPS-carried derivations require the US schema.")
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    person = tables["person"]

    _fill_missing(person, "age", _source(person, "A_AGE"))
    _fill_bool_missing(person, "is_female", _integer_source(person, "A_SEX") == 2)
    _fill_bool_missing(person, "receives_wic", reported_wic_receipt_carrier(person))
    _fill_missing(person, "employment_income_before_lsr", _source(person, "WSAL_VAL"))
    _fill_missing(
        person,
        "self_employment_income_before_lsr",
        _source(person, "SEMP_VAL"),
    )

    interest = _source(person, "INT_VAL")
    _fill_missing(
        person,
        "taxable_interest_income",
        interest * TAXABLE_INTEREST_FRACTION,
    )

    dividends = _source(person, "DIV_VAL")
    _fill_missing(
        person,
        "qualified_dividend_income",
        dividends * QUALIFIED_DIVIDEND_FRACTION,
    )
    _fill_missing(
        person,
        "non_qualified_dividend_income",
        dividends * (1 - QUALIFIED_DIVIDEND_FRACTION),
    )

    capital_gains = _source(person, "CAP_VAL")
    _fill_missing(
        person,
        "long_term_capital_gains_before_response",
        capital_gains * LONG_TERM_CAPITAL_GAIN_FRACTION,
    )
    _fill_missing(
        person,
        "short_term_capital_gains",
        capital_gains * (1 - LONG_TERM_CAPITAL_GAIN_FRACTION),
    )

    _fill_social_security_leaves(person)
    pensions = _source(person, "PNSN_VAL") + _source(person, "ANN_VAL")
    _fill_missing(
        person,
        "taxable_private_pension_income",
        pensions * TAXABLE_PENSION_FRACTION,
    )
    _fill_missing(
        person,
        "tax_exempt_private_pension_income",
        pensions * (1 - TAXABLE_PENSION_FRACTION),
    )
    _fill_missing(person, "taxable_ira_distributions", _ira_distributions(person))

    person = derive_us_alimony_from_asec(person)
    tables["person"] = person

    direct_sources: Mapping[str, str] = {
        "rental_income": "RNT_VAL",
        "farm_operations_income": "FRSE_VAL",
        "unemployment_compensation": "UC_VAL",
        "health_insurance_premiums_without_medicare_part_b": "PHIP_VAL",
        "other_medical_expenses": "PMED_VAL",
        "over_the_counter_health_expenses": "POTC_VAL",
    }
    # PEMCPREM remains on the source frame as evidence only. The corresponding
    # reported Part B leaf has no engine-formula consumers, so the pool must not
    # promote or transfer it onto ACS rows.
    for output, source in direct_sources.items():
        _fill_missing(person, output, _source(person, source))

    _fill_health_coverage_inputs(person)
    _fill_spm_unit_reported_enrollment_inputs(
        person,
        tables["spm_unit"],
        public_assistance_type_source=public_assistance_type_source,
    )
    _fill_spm_unit_childcare_inputs(person, tables["spm_unit"])

    formula_owned = sorted(CPS_CARRIED_FORMULA_OWNED_COLUMNS.intersection(person))
    if formula_owned:
        raise ValueError(
            "CPS-carried derivations must not carry formula-owned aggregate "
            f"columns: {formula_owned}."
        )

    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def reported_tanf_enrollment_by_spm_unit(
    person: pd.DataFrame,
    public_assistance_type_source: pd.DataFrame | None = None,
) -> pd.Series:
    """Return reported TANF enrollment per SPM unit, gated on ``PAW_TYP``.

    A unit is TANF-enrolled when any member reports a positive annual
    public-assistance amount (``PAW_VAL > 0``) whose reported type is
    TANF/AFDC (``PAW_TYP`` 1) or both TANF/AFDC and other assistance
    (``PAW_TYP`` 3). ``PAW_TYP`` 2 (other cash welfare, e.g. general
    assistance) and unreported-type PAW recipients are NOT marked
    TANF-enrolled: the engine's ``is_tanf_enrolled`` variable is
    TANF-specific, and conflating other cash welfare with TANF would poison
    its 13 cycle-safe TANF helper consumers (microcosm#591). The conflation
    is material — in the 2023 ASEC (income year 2022), 271 of 682
    PAW-positive SPM units (39.7% unweighted, 42.6% SPM-weighted) have no
    member reporting a TANF type.

    ``PAW_TYP`` is read from ``person`` when present (the pool's raw-stage
    checkpoint carries it). Otherwise, when any ``PAW_VAL`` is positive, the
    pinned public-assistance-type sidecar is required and joined transiently
    without mutating ``person``; refusing to fall back to ``PAW_VAL > 0``
    keeps the gate impossible to silently drop.
    """

    if "person_spm_unit_id" not in person.columns:
        raise ValueError(
            "Reported TANF enrollment requires person column(s): "
            "['person_spm_unit_id']."
        )
    amounts = _source(person, CPS_REPORTED_TANF_AMOUNT_RAW_COLUMN)
    if CPS_REPORTED_TANF_TYPE_RAW_COLUMN in person.columns:
        types = _integer_source(person, CPS_REPORTED_TANF_TYPE_RAW_COLUMN)
    elif not (amounts > 0.0).any():
        types = np.zeros(len(person), dtype=np.int64)
    elif public_assistance_type_source is not None:
        joined = fill_asec_public_assistance_type_source(
            person,
            public_assistance_type_source,
        )
        types = _integer_source(joined, CPS_REPORTED_TANF_TYPE_RAW_COLUMN)
    else:
        raise ValueError(
            "Reported TANF enrollment requires PAW_TYP for the positive "
            f"{CPS_REPORTED_TANF_AMOUNT_RAW_COLUMN} rows: carry the column on "
            "the frame or pass public_assistance_type_source; PAW_VAL alone "
            "conflates TANF with other cash welfare (microcosm#591)."
        )
    member_enrolled = (amounts > 0.0) & np.isin(types, sorted(PAW_TYPE_TANF_CODES))
    return (
        pd.DataFrame(
            {
                "person_spm_unit_id": person["person_spm_unit_id"],
                "_reported_tanf_enrollment": member_enrolled,
            }
        )
        .groupby("person_spm_unit_id", sort=True)["_reported_tanf_enrollment"]
        .any()
        .rename("reported_tanf_enrollment")
    )


def reported_snap_receipt_by_spm_unit(person: pd.DataFrame) -> pd.Series:
    """Return the shared max-positive annual SNAP receipt interpretation.

    ``SPM_SNAPSUB`` is an annual SPM-unit amount replicated on person rows.
    A unit reports SNAP receipt when the maximum member value is positive;
    non-numeric and missing values are treated as zero.
    """

    required = [CPS_REPORTED_SNAP_RAW_COLUMN, "person_spm_unit_id"]
    missing = [column for column in required if column not in person.columns]
    if missing:
        raise ValueError(f"Reported SNAP receipt requires person column(s): {missing}.")
    subsidy = pd.to_numeric(
        person[CPS_REPORTED_SNAP_RAW_COLUMN], errors="coerce"
    ).fillna(0.0)
    return (
        person.assign(_reported_snap_subsidy=subsidy)
        .groupby("person_spm_unit_id", sort=True)["_reported_snap_subsidy"]
        .max()
        .gt(0.0)
        .rename("reported_snap_receipt")
    )


def reported_wic_receipt_carrier(person: pd.DataFrame) -> np.ndarray:
    """Carry the SPM unit's reported WIC receipt fact on its reporting adult.

    Census asks ``WICYN`` only of adult women, and a woman may report receipt
    for herself or for a child beneficiary. ``receives_wic`` therefore remains
    stored on that reporting adult solely as the carrier for her SPM unit's
    receipt fact; it does not identify the person who received WIC. Unit-level
    aggregation is the ONLY supported consumption grain. Any person-grain use
    requires re-adjudication under microcosm#591:
    https://github.com/PolicyEngine/microcosm/issues/591#issuecomment-5160668979
    """

    return _integer_source(person, CPS_REPORTED_WIC_RAW_COLUMN) == 1


def _fill_missing(table: pd.DataFrame, column: str, values: np.ndarray) -> None:
    if column not in table.columns:
        table[column] = values.astype("float64")


def _fill_bool_missing(table: pd.DataFrame, column: str, values: np.ndarray) -> None:
    if column not in table.columns:
        table[column] = values.astype(bool)


def _source(person: pd.DataFrame, column: str) -> np.ndarray:
    if column not in person.columns:
        return np.zeros(len(person), dtype=np.float64)
    return (
        pd.to_numeric(person[column], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def _fill_social_security_leaves(person: pd.DataFrame) -> None:
    social_security = _source(person, "SS_VAL")
    reason_1 = _integer_source(person, "RESNSS1")
    reason_2 = _integer_source(person, "RESNSS2")
    age = _integer_source(person, "A_AGE")

    is_retirement = (reason_1 == 1) | (reason_2 == 1)
    is_disability = (reason_1 == 2) | (reason_2 == 2)
    is_survivor = np.isin(reason_1, [3, 5]) | np.isin(reason_2, [3, 5])
    is_dependent = np.isin(reason_1, [4, 6, 7]) | np.isin(reason_2, [4, 6, 7])
    unclassified = (
        (social_security > 0)
        & ~is_retirement
        & ~is_disability
        & ~is_survivor
        & ~is_dependent
    )

    _fill_missing(
        person,
        "social_security_retirement",
        np.where(is_retirement | (unclassified & (age >= 62)), social_security, 0.0),
    )
    _fill_missing(
        person,
        "social_security_disability",
        np.where(
            (is_disability & ~is_retirement) | (unclassified & (age < 62)),
            social_security,
            0.0,
        ),
    )
    _fill_missing(
        person,
        "social_security_survivors",
        np.where(is_survivor & ~is_retirement & ~is_disability, social_security, 0.0),
    )
    _fill_missing(
        person,
        "social_security_dependents",
        np.where(
            is_dependent & ~is_retirement & ~is_disability & ~is_survivor,
            social_security,
            0.0,
        ),
    )


def _fill_health_coverage_inputs(person: pd.DataFrame) -> None:
    marketplace = _yes_code(person, "NOW_MRK")
    _fill_bool_missing(
        person,
        "has_marketplace_health_coverage_at_interview",
        marketplace,
    )
    source_columns: Mapping[str, str] = {
        "has_non_marketplace_direct_purchase_health_coverage_at_interview": "NOW_NONM",
        "has_medicaid_health_coverage_at_interview": "NOW_MCAID",
        "has_esi": "NOW_GRP",
        "has_champva_health_coverage_at_interview": "NOW_CHAMPVA",
        "has_tricare_health_coverage_at_interview": "NOW_MIL",
        "has_va_health_coverage_at_interview": "NOW_VACARE",
        "has_other_means_tested_health_coverage_at_interview": "NOW_OTHMT",
        "has_indian_health_service_coverage_at_interview": "NOW_IHSFLG",
    }
    for output, source in source_columns.items():
        _fill_bool_missing(person, output, _yes_code(person, source))


def _fill_spm_unit_childcare_inputs(
    person: pd.DataFrame,
    spm_unit: pd.DataFrame,
) -> None:
    """Carry ASEC SPM childcare expenses onto the SPM-unit input leaf.

    ``SPM_CHILDCAREXPNS`` is an SPM-unit value replicated on every member's
    person record, so the unit's value is its members' maximum. The engine
    derives person, tax-unit, and CDCC childcare expenses from this leaf;
    a base without it zeroes every CDCC baseline (microcosm issue #278).
    """

    column = "spm_unit_pre_subsidy_childcare_expenses"
    if column in spm_unit.columns:
        return
    member_values = pd.DataFrame(
        {
            "person_spm_unit_id": person["person_spm_unit_id"],
            column: _source(person, "SPM_CHILDCAREXPNS"),
        }
    )
    unit_values = member_values.groupby("person_spm_unit_id", sort=False)[column].max()
    spm_unit[column] = (
        unit_values.reindex(spm_unit["spm_unit_id"])
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def _fill_spm_unit_reported_enrollment_inputs(
    person: pd.DataFrame,
    spm_unit: pd.DataFrame,
    *,
    public_assistance_type_source: pd.DataFrame | None = None,
) -> None:
    """Carry annual reported TANF and SNAP receipt onto monthly leaves."""

    if "is_tanf_enrolled" not in spm_unit.columns:
        tanf = reported_tanf_enrollment_by_spm_unit(
            person,
            public_assistance_type_source,
        )
        spm_unit["is_tanf_enrolled"] = (
            tanf.reindex(spm_unit["spm_unit_id"]).fillna(False).to_numpy(dtype=bool)
        )

    if "receives_snap" not in spm_unit.columns:
        snap_person = person
        if CPS_REPORTED_SNAP_RAW_COLUMN not in snap_person.columns:
            snap_person = person.assign(**{CPS_REPORTED_SNAP_RAW_COLUMN: 0.0})
        reported = reported_snap_receipt_by_spm_unit(snap_person)
        spm_unit["receives_snap"] = (
            reported.reindex(spm_unit["spm_unit_id"]).fillna(False).to_numpy(dtype=bool)
        )


def _ira_distributions(person: pd.DataFrame) -> np.ndarray:
    values = np.zeros(len(person), dtype=np.float64)
    for suffix in ("1", "2", "1_YNG", "2_YNG"):
        code = _integer_source(person, f"DST_SC{suffix}")
        amount = _source(person, f"DST_VAL{suffix}")
        values += np.where(code == 4, amount, 0.0)
    return values


def _integer_source(person: pd.DataFrame, column: str) -> np.ndarray:
    return _source(person, column).astype("int64")


def _yes_code(person: pd.DataFrame, column: str) -> np.ndarray:
    return _integer_source(person, column) == 1


def us_reported_coverage_vintage_signal_gate(
    frame: Frame,
    *,
    min_vintage_rows: int = US_REPORTED_COVERAGE_VINTAGE_GATE_MIN_ROWS,
) -> GateResult:
    """Require every pooled source vintage to carry reported-coverage signal.

    Microcosm #720: the pooled income-year 2022/2023 ASEC inputs carried only
    ``NOW_GRP``/``NOW_MRK`` of the 18 ``NOW_*`` at-interview recodes, so
    :func:`derive_us_cps_carried_inputs` silently mapped every 2022/2023
    -vintage person to ``False`` for seven of the nine reported-coverage
    flags and the certified Build P artifact reported 24.6M under-65
    Medicaid at interview against ~58M survey. A flag populated for one
    vintage passes the presence-style checks (``release_input_coverage``,
    ``degenerate_input_signal``); this gate enforces the per-vintage
    invariant those checks cannot see.

    Groups are ``person_support_channel`` x ``source_year`` when the channel
    column is present (ACS-spine rows also carry ``source_year``, so a
    year-only key would let ACS signal mask a missing ASEC recode), else
    ``source_year`` alone. Every group with at least ``min_vintage_rows``
    person rows must have, for every reported-coverage input, a boolean-like
    column with no nulls and at least one reporter. Provenance must be
    present: a missing ``source_year`` column, null source years, or an
    empty person table fail the gate rather than collapsing into one group.

    Groups below ``min_vintage_rows`` (smoke pools) are recorded in the
    details but not enforced. This is a zero-signal sentinel, not a survey
    mass check: it observes that a vintage has no reporters; the #720
    cause (a source input lacking the recode) is the documented reading.
    """

    person = frame.table("person")
    missing = [
        column
        for column in US_REPORTED_COVERAGE_PERSON_INPUTS
        if column not in person.columns
    ]
    if "source_year" not in person.columns:
        missing.append("source_year")
    if missing:
        return GateResult(
            name="reported_coverage_vintage_signal",
            passed=False,
            failures=tuple(f"person column missing: {column}." for column in missing),
            details={"missing": missing},
        )
    if len(person) == 0:
        # Unreachable through a valid Frame (weights cannot be empty); kept
        # so a direct caller cannot pass an empty table as "no failures".
        return GateResult(
            name="reported_coverage_vintage_signal",
            passed=False,
            failures=("person table is empty: no vintage carries any signal.",),
            details={"rows": 0},
        )
    failures: list[str] = []
    null_years = int(person["source_year"].isna().sum())
    if null_years:
        failures.append(
            f"source_year: {null_years} person rows have no source year; the "
            "per-vintage invariant cannot be proven for unprovenanced rows."
        )
    keys: list[pd.Series] = []
    if "person_support_channel" in person.columns:
        keys.append(person["person_support_channel"].astype(str))
    keys.append(person["source_year"])
    vintages: dict[str, dict[str, object]] = {}
    for key, group in person.groupby(keys, sort=True, dropna=True):
        parts = key if isinstance(key, tuple) else (key,)
        label = "/".join(str(part) for part in parts)
        rows = int(len(group))
        reporter_counts: dict[str, int] = {}
        null_counts: dict[str, int] = {}
        dtype_failures: list[str] = []
        for column in US_REPORTED_COVERAGE_PERSON_INPUTS:
            values = group[column]
            if not (
                pd.api.types.is_bool_dtype(values)
                or pd.api.types.is_numeric_dtype(values)
            ):
                dtype_failures.append(column)
                reporter_counts[column] = 0
                null_counts[column] = int(values.isna().sum())
                continue
            null_counts[column] = int(values.isna().sum())
            reporter_counts[column] = int(values.fillna(False).astype(bool).sum())
        enforced = rows >= min_vintage_rows
        vintages[label] = {
            "rows": rows,
            "enforced": enforced,
            "reporter_counts": reporter_counts,
            "null_counts": null_counts,
        }
        if not enforced:
            continue
        for column in dtype_failures:
            failures.append(
                f"{column}: vintage {label} stores a non-boolean dtype "
                f"({group[column].dtype}); reported-coverage inputs must be "
                "boolean."
            )
        for column, count in null_counts.items():
            if count and column not in dtype_failures:
                failures.append(
                    f"{column}: vintage {label} has {count} null values over "
                    f"{rows} person rows; the flag must be fully populated."
                )
        for column, count in reporter_counts.items():
            if count == 0 and column not in dtype_failures:
                failures.append(
                    f"{column}: vintage {label} has 0 reporters over {rows} "
                    "person rows (consistent with a source input lacking the "
                    "at-interview recode, microcosm #720)."
                )
    return GateResult(
        name="reported_coverage_vintage_signal",
        passed=not failures,
        failures=tuple(failures),
        details={
            "min_vintage_rows": int(min_vintage_rows),
            "grouping": (
                ["person_support_channel", "source_year"]
                if "person_support_channel" in person.columns
                else ["source_year"]
            ),
            "vintages": vintages,
        },
    )
