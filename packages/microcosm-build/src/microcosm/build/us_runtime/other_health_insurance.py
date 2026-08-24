"""CPS ASEC other-health-insurance premium inputs.

The retired eCPS pipeline kept the CPS-reported non-Medicare premium and
derived ``other_health_insurance_premiums`` as its nonnegative residual after
subtracting baseline CHIP, Marketplace, and Medicaid premiums.  Those three
modeled premiums live on tax units and were assigned wholly to the first
person in each tax unit before subtraction.  The retired extended-CPS stage
then jointly QRF-imputed the reported and residual leaves onto its PUF clone
half using eight documented predictors and at most 5,000 ASEC people.

This port preserves measured ASEC values exactly and replaces only the PUF
support channel.  It pins 100 trees and supplies typed person weights as
Microcosm reproducibility hardening; the archive left the tree count implicit
and did not weight the fit.  The separate employer-premium target is not
proxied here; ``has_esi`` remains only the archived predictor shared by the
restored targets.
"""

from __future__ import annotations

import gc
from importlib.resources import files
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
from microcosm.build.us_runtime.support_provenance import (
    has_support_role_metadata,
    support_role_series,
)
from microcosm.frame import Frame
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "OTHER_HEALTH_INSURANCE_ARCHIVED_DERIVATION_URL",
    "OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_IMPUTATION_URL",
    "OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_OUTPUTS_URL",
    "OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_PREDICTORS_URL",
    "OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_SPLICE_URL",
    "US_OTHER_HEALTH_INSURANCE_MODELED_PREMIUM_VARIABLES",
    "US_OTHER_HEALTH_INSURANCE_NONCONSTANT_PERSON_COLUMNS",
    "US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS",
    "US_OTHER_HEALTH_INSURANCE_REQUIRED_SOURCE_COLUMNS",
    "US_OTHER_HEALTH_INSURANCE_STAGE_NAME",
    "US_OTHER_HEALTH_INSURANCE_STAGE_OUTPUT_COLUMNS",
    "US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS",
    "US_SE_HEALTH_MEDICARE_AGE_THRESHOLD",
    "US_SE_HEALTH_SELF_EMPLOYMENT_INCOME_SOURCES",
    "attribute_us_se_health_premiums",
    "attribute_us_se_health_premiums_from_manifest",
    "derive_us_other_health_insurance_from_asec",
    "derive_us_other_health_insurance_from_manifest",
    "impute_us_other_health_insurance_to_puf_support_from_manifest",
    "us_other_health_insurance_signal_gate",
    "us_other_health_insurance_stage_spec",
    "us_other_health_insurance_summary",
    "with_us_other_health_insurance_inputs",
]

QRF: Any | None = None

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/"
    "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
    "policyengine_" + "us_data/"
)
OTHER_HEALTH_INSURANCE_ARCHIVED_DERIVATION_URL = (
    _ARCHIVED_ROOT + "datasets/cps/cps.py#L828-L944"
)
OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_OUTPUTS_URL = (
    _ARCHIVED_ROOT + "datasets/cps/extended_cps.py#L135-L194"
)
OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_PREDICTORS_URL = (
    _ARCHIVED_ROOT + "datasets/cps/extended_cps.py#L234-L248"
)
OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_IMPUTATION_URL = (
    _ARCHIVED_ROOT + "datasets/cps/extended_cps.py#L639-L745"
)
OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_SPLICE_URL = (
    _ARCHIVED_ROOT + "datasets/cps/extended_cps.py#L1014-L1076"
)

US_OTHER_HEALTH_INSURANCE_STAGE_NAME = "other_health_insurance_premiums"
US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "health_insurance_premiums_without_medicare_part_b",
    "other_health_insurance_premiums",
)
US_OTHER_HEALTH_INSURANCE_NONCONSTANT_PERSON_COLUMNS: tuple[str, ...] = (
    US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS[1],
)
US_OTHER_HEALTH_INSURANCE_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS[0],
)
US_OTHER_HEALTH_INSURANCE_MODELED_PREMIUM_VARIABLES: tuple[str, ...] = (
    "chip_premium",
    "marketplace_net_premium",
    "medicaid_premium",
)
# Self-employed premium attribution (PolicyEngine/microcosm#451 item 2): the
# section 162(l) ALD chain in PolicyEngine-US 1.819.0 reads person inputs
# health_insurance_premiums (via the self_employed_health_insurance_premiums
# adds-aggregation, gated by is_self_employed) and caps the deduction at
# total_self_employment_income = self_employment_income +
# sstb_self_employment_income. self_employed_health_insurance_premiums itself
# is formula-owned (adds) and cannot ship as a column, so the stage ships the
# two pure inputs the engine intends.
US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS: tuple[str, ...] = (
    "health_insurance_premiums",
    "is_self_employed",
)
US_SE_HEALTH_SELF_EMPLOYMENT_INCOME_SOURCES: tuple[str, ...] = (
    "self_employment_income_before_lsr",
    "sstb_self_employment_income_before_lsr",
)
# Attribution is withheld from people on the Medicare proxy (age 65+ or any
# Social Security disability income) so the statutory medical-expense premium
# concept stays numerically invariant: for everyone attributed, the direct
# premium input equals the decomposed reported premium because the modeled
# Part B add-on is zero for non-enrollees.
US_SE_HEALTH_MEDICARE_AGE_THRESHOLD = 65
US_OTHER_HEALTH_INSURANCE_STAGE_OUTPUT_COLUMNS: tuple[str, ...] = (
    *US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS,
    *US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS,
)

_REPORTED_OUTPUT, _OTHER_OUTPUT = US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS
_SE_PREMIUMS_OUTPUT, _SE_FLAG_OUTPUT = US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS
_SE_MEDICARE_AGE_SOURCE = "age"
_SE_MEDICARE_SSDI_SOURCE = "social_security_disability"
# 26 USC 162(l)(2)(B): no deduction for months the taxpayer is eligible to
# participate in a subsidized employer plan. Measured employer-sponsored
# coverage is the conservative, incomplete proxy for that eligibility.
_SE_EMPLOYER_COVERAGE_SOURCE = "has_esi"
_PERSON_WEIGHT_COLUMN = "person_weight"
_BASE_ASEC_SUPPORT_CHANNEL = "asec"
_PUF_TAX_DETAIL_SUPPORT_CHANNEL = "puf_tax_detail"
_PREDICTORS: tuple[str, ...] = (
    "age",
    "is_male",
    "has_esi",
    "tax_unit_is_joint",
    "tax_unit_count_dependents",
    "employment_income",
    "self_employment_income",
    "social_security",
)
_PREDICTOR_PREFIX = "other_health_insurance_predictor_"
_EXPECTED_DIRECT_PARAMETERS = {
    "reported_source": _REPORTED_OUTPUT,
    "chip_premium_source": "chip_premium",
    "marketplace_net_premium_source": "marketplace_net_premium",
    "medicaid_premium_source": "medicaid_premium",
    "output": _OTHER_OUTPUT,
}
_PUF_IMPUTATION_PARAMETER_KEYS = frozenset(
    {
        "predictors",
        "max_train_samples",
        "n_estimators",
        "seed_from_build_config",
        "weight",
    }
)
_MAX_TRAIN_SAMPLES = 5_000
_N_ESTIMATORS = 100
_POSITIVE_SHARE_BAND = (0.15, 0.65)


def us_other_health_insurance_stage_spec() -> SourceStageSpec:
    """Load and validate the packaged other-premium stage declaration."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_OTHER_HEALTH_INSURANCE_STAGE_NAME not in stage_map:
        raise ValueError(
            "US source manifest declares no "
            f"{US_OTHER_HEALTH_INSURANCE_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_OTHER_HEALTH_INSURANCE_STAGE_NAME]
    if tuple(spec.outputs) != US_OTHER_HEALTH_INSURANCE_STAGE_OUTPUT_COLUMNS:
        raise ValueError(
            f"{US_OTHER_HEALTH_INSURANCE_STAGE_NAME!r} outputs must preserve "
            "the archived target order followed by the attribution outputs "
            f"{list(US_OTHER_HEALTH_INSURANCE_STAGE_OUTPUT_COLUMNS)}; got "
            f"{list(spec.outputs)}."
        )
    return spec


def _strict_nonnegative_values(person: pd.DataFrame, column: str) -> np.ndarray:
    if column not in person.columns:
        raise SourceRuntimeError(
            f"US other-health-insurance derivation requires source column {column!r}."
        )
    values = pd.to_numeric(person[column], errors="coerce").to_numpy(dtype=np.float64)
    nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    if nonfinite:
        raise SourceRuntimeError(
            f"US other-health-insurance source {column!r} contains {nonfinite} "
            "nonnumeric or nonfinite value(s)."
        )
    negative = int(np.count_nonzero(values < 0.0))
    if negative:
        raise SourceRuntimeError(
            f"US other-health-insurance source {column!r} contains {negative} "
            "negative value(s)."
        )
    return values


def derive_us_other_health_insurance_from_asec(
    person: pd.DataFrame,
    *,
    reported_source_column: str = _REPORTED_OUTPUT,
    chip_premium_source_column: str = "chip_premium",
    marketplace_net_premium_source_column: str = "marketplace_net_premium",
    medicaid_premium_source_column: str = "medicaid_premium",
    output_column: str = _OTHER_OUTPUT,
) -> pd.DataFrame:
    """Derive the nonnegative residual after baseline modeled premiums."""

    reported = _strict_nonnegative_values(person, reported_source_column)
    modeled = np.zeros(len(person), dtype=np.float64)
    for column in (
        chip_premium_source_column,
        marketplace_net_premium_source_column,
        medicaid_premium_source_column,
    ):
        modeled += _strict_nonnegative_values(person, column)

    result = person.copy(deep=True)
    result[output_column] = np.clip(reported - modeled, 0.0, None)
    return result


def derive_us_other_health_insurance_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    _context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Interpret the manifest's archived premium-residual operation."""

    if operation.kind != "derive_other_health_insurance_premiums":
        raise SourceRuntimeError(
            "US other-health-insurance derivation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US other-health-insurance derivation requires the person table first."
        )
    parameters = dict(operation.parameters)
    if parameters != _EXPECTED_DIRECT_PARAMETERS:
        raise SourceRuntimeError(
            "US other-health-insurance direct mapping drifted from the archived "
            f"method: expected {_EXPECTED_DIRECT_PARAMETERS}, got {parameters}."
        )
    return derive_us_other_health_insurance_from_asec(
        frame,
        reported_source_column=parameters["reported_source"],
        chip_premium_source_column=parameters["chip_premium_source"],
        marketplace_net_premium_source_column=parameters[
            "marketplace_net_premium_source"
        ],
        medicaid_premium_source_column=parameters["medicaid_premium_source"],
        output_column=parameters["output"],
    )


def impute_us_other_health_insurance_to_puf_support_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Jointly QRF-impute both premium leaves onto PUF-support people."""

    if operation.kind != "impute_other_health_insurance_premiums_to_puf_support":
        raise SourceRuntimeError(
            "US other-health-insurance PUF imputation received unexpected "
            f"operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US other-health-insurance PUF imputation requires the person table first."
        )
    unexpected = sorted(set(operation.parameters) - _PUF_IMPUTATION_PARAMETER_KEYS)
    missing_parameters = sorted(
        _PUF_IMPUTATION_PARAMETER_KEYS - set(operation.parameters)
    )
    if unexpected or missing_parameters:
        raise SourceRuntimeError(
            "US other-health-insurance PUF imputation parameters must match the "
            f"archived method; missing={missing_parameters}, "
            f"unexpected={unexpected}."
        )
    if not has_support_role_metadata(frame, entity="person"):
        return frame.copy(deep=True)

    predictors = tuple(str(value) for value in operation.parameters["predictors"])
    if predictors != _PREDICTORS:
        raise SourceRuntimeError(
            "US other-health-insurance PUF predictors drifted from the archived "
            f"method: expected {list(_PREDICTORS)}, got {list(predictors)}."
        )
    if operation.parameters["weight"] != _PERSON_WEIGHT_COLUMN:
        raise SourceRuntimeError(
            "US other-health-insurance PUF imputation must use typed person weights."
        )
    if operation.parameters["seed_from_build_config"] is not True:
        raise SourceRuntimeError(
            "US other-health-insurance PUF imputation seed must come from the "
            "build config."
        )
    max_train_samples = int(operation.parameters["max_train_samples"])
    n_estimators = int(operation.parameters["n_estimators"])
    if not 0 < max_train_samples <= _MAX_TRAIN_SAMPLES:
        raise SourceRuntimeError(
            "US other-health-insurance PUF max_train_samples must be between "
            f"1 and {_MAX_TRAIN_SAMPLES}."
        )
    if n_estimators != _N_ESTIMATORS:
        raise SourceRuntimeError(
            "US other-health-insurance PUF imputation must pin exactly "
            f"{_N_ESTIMATORS} trees; got {n_estimators}."
        )

    predictor_columns = [_PREDICTOR_PREFIX + name for name in predictors]
    required = [
        _PERSON_WEIGHT_COLUMN,
        *predictor_columns,
        *US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SourceRuntimeError(
            f"US other-health-insurance PUF imputation is missing column(s): {missing}."
        )

    role = support_role_series(frame, entity="person")
    asec_mask = role == _BASE_ASEC_SUPPORT_CHANNEL
    puf_mask = role == _PUF_TAX_DETAIL_SUPPORT_CHANNEL
    if not asec_mask.any() or not puf_mask.any():
        raise SourceRuntimeError(
            "US other-health-insurance PUF imputation requires nonempty ASEC "
            "and PUF-tax-detail support channels."
        )

    training = frame.loc[
        asec_mask,
        [*predictor_columns, *US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS],
    ].copy()
    training.columns = [*predictors, *US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS]
    test = frame.loc[puf_mask, predictor_columns].copy()
    test.columns = list(predictors)
    weights = pd.to_numeric(
        frame.loc[asec_mask, _PERSON_WEIGHT_COLUMN], errors="coerce"
    )
    numeric_weights = weights.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_weights).all() or bool((numeric_weights < 0.0).any()):
        raise SourceRuntimeError(
            "US other-health-insurance QRF person weights must be finite and "
            "nonnegative."
        )
    if float(numeric_weights.sum()) <= 0.0:
        raise SourceRuntimeError(
            "US other-health-insurance QRF person weights sum to zero."
        )

    if len(training) > max_train_samples:
        sample = training.sample(
            n=max_train_samples,
            random_state=(context.config.seed if context is not None else 0),
        ).index
        training = training.loc[sample]
        weights = weights.loc[sample]

    for column in (*predictors, *US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS):
        training[column] = pd.to_numeric(training[column], errors="coerce")
        if not np.isfinite(training[column].to_numpy(dtype=np.float64)).all():
            raise SourceRuntimeError(
                "US other-health-insurance QRF training column "
                f"{column!r} contains nonfinite values."
            )
    for column in predictors:
        test[column] = pd.to_numeric(test[column], errors="coerce")
        if not np.isfinite(test[column].to_numpy(dtype=np.float64)).all():
            raise SourceRuntimeError(
                "US other-health-insurance QRF prediction column "
                f"{column!r} contains nonfinite values."
            )

    global QRF
    if QRF is None:
        from importlib import import_module

        QRF = import_module("microcosm.fit").QRF
    seed = context.config.seed if context is not None else 0
    fitted = QRF(n_estimators=n_estimators, seed=seed).fit(
        training,
        list(predictors),
        list(US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS),
        weights=weights.to_numpy(dtype=np.float64),
    )
    predictions = fitted.predict(test)
    missing_outputs = [
        output
        for output in US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS
        if output not in predictions
    ]
    if missing_outputs:
        raise SourceRuntimeError(
            "US other-health-insurance QRF prediction is missing output(s): "
            f"{missing_outputs}."
        )

    result = frame.copy(deep=True)
    for output in US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS:
        predicted = pd.to_numeric(predictions[output], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(predicted).all():
            raise SourceRuntimeError(
                f"US other-health-insurance QRF produced nonfinite {output} values."
            )
        if bool((predicted < 0.0).any()):
            raise SourceRuntimeError(
                f"US other-health-insurance QRF produced negative {output} values."
            )
        result.loc[puf_mask, output] = predicted
    return result


def _finite_values(person: pd.DataFrame, column: str) -> np.ndarray:
    if column not in person.columns:
        raise SourceRuntimeError(
            f"US self-employed premium attribution requires source column {column!r}."
        )
    values = pd.to_numeric(person[column], errors="coerce").to_numpy(dtype=np.float64)
    nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    if nonfinite:
        raise SourceRuntimeError(
            f"US self-employed premium attribution source {column!r} contains "
            f"{nonfinite} nonnumeric or nonfinite value(s)."
        )
    return values


def attribute_us_se_health_premiums(
    person: pd.DataFrame,
    *,
    reported_source_column: str = _REPORTED_OUTPUT,
    self_employment_income_source_columns: tuple[str, ...] = (
        US_SE_HEALTH_SELF_EMPLOYMENT_INCOME_SOURCES
    ),
    medicare_age_source_column: str = _SE_MEDICARE_AGE_SOURCE,
    medicare_age_threshold: int = US_SE_HEALTH_MEDICARE_AGE_THRESHOLD,
    medicare_ssdi_source_column: str = _SE_MEDICARE_SSDI_SOURCE,
    employer_coverage_source_column: str = _SE_EMPLOYER_COVERAGE_SOURCE,
    output_premiums_column: str = _SE_PREMIUMS_OUTPUT,
    output_flag_column: str = _SE_FLAG_OUTPUT,
) -> pd.DataFrame:
    """Attribute reported premiums to self-employed people deterministically.

    The premium output copies the reported non-Part-B premium exactly for
    people with strictly positive combined Schedule C income who are outside
    the Medicare proxy AND outside measured employer-sponsored coverage, and
    is zero elsewhere: attribution never invents premium mass. Section
    162(l)(2)(B) disallows the deduction for months the taxpayer is eligible
    to participate in a subsidized employer plan; the measured ``has_esi``
    indicator (actual employer-sponsored coverage) is the conservative,
    incomplete proxy for that eligibility — eligibility through a spouse's or
    dependent's employer is not measured and remains documented residual
    overbreadth. The flag marks every strictly-positive Schedule C person so
    the engine's ``defined_for`` gate opens exactly where the section 162(l)
    earned-income cap can bind. Schedule C losses are legitimate measured
    signal and simply leave the flag off.
    """

    reported = _strict_nonnegative_values(person, reported_source_column)
    self_employment = np.zeros(len(person), dtype=np.float64)
    for column in self_employment_income_source_columns:
        self_employment += _finite_values(person, column)
    age = _finite_values(person, medicare_age_source_column)
    ssdi = _strict_nonnegative_values(person, medicare_ssdi_source_column)
    employer_covered = _boolean_values(person, employer_coverage_source_column)

    self_employed = self_employment > 0.0
    medicare_proxy = (age >= float(medicare_age_threshold)) | (ssdi > 0.0)

    result = person.copy(deep=True)
    result[output_premiums_column] = np.where(
        self_employed & ~medicare_proxy & ~employer_covered, reported, 0.0
    )
    result[output_flag_column] = self_employed
    return result


def _is_boolean_like_dtype(series: pd.Series) -> bool:
    """Whether a column can carry a trustworthy boolean indicator.

    Only bool and plain numeric dtypes qualify: object/string dtypes are
    rejected outright because the engine parses any nonempty string —
    including "0" — as True.
    """

    kind = getattr(series.dtype, "kind", "")
    return kind in "biuf"


def _boolean_values(person: pd.DataFrame, column: str) -> np.ndarray:
    if column not in person.columns:
        raise SourceRuntimeError(
            f"US self-employed premium attribution requires source column {column!r}."
        )
    if not _is_boolean_like_dtype(person[column]):
        raise SourceRuntimeError(
            f"US self-employed premium attribution source {column!r} carries "
            f"an object dtype ({person[column].dtype}); a boolean indicator "
            "is required."
        )
    values = pd.to_numeric(person[column], errors="coerce").to_numpy(dtype=np.float64)
    invalid = int(np.count_nonzero(~np.isfinite(values) | ~np.isin(values, (0.0, 1.0))))
    if invalid:
        raise SourceRuntimeError(
            f"US self-employed premium attribution source {column!r} contains "
            f"{invalid} null or non-boolean value(s)."
        )
    return values == 1.0


def attribute_us_se_health_premiums_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    _context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Interpret the manifest's self-employed premium attribution operation."""

    if operation.kind != "attribute_self_employed_health_premiums":
        raise SourceRuntimeError(
            "US self-employed premium attribution received unexpected "
            f"operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US self-employed premium attribution requires the person table first."
        )
    expected = {
        "reported_source": _REPORTED_OUTPUT,
        "self_employment_income_sources": list(
            US_SE_HEALTH_SELF_EMPLOYMENT_INCOME_SOURCES
        ),
        "medicare_age_source": _SE_MEDICARE_AGE_SOURCE,
        "medicare_age_threshold": US_SE_HEALTH_MEDICARE_AGE_THRESHOLD,
        "medicare_ssdi_source": _SE_MEDICARE_SSDI_SOURCE,
        "employer_coverage_source": _SE_EMPLOYER_COVERAGE_SOURCE,
        "output_premiums": _SE_PREMIUMS_OUTPUT,
        "output_self_employed_flag": _SE_FLAG_OUTPUT,
    }
    parameters = dict(operation.parameters)
    if parameters != expected:
        raise SourceRuntimeError(
            "US self-employed premium attribution drifted from the pinned "
            f"method: expected {expected}, got {parameters}."
        )
    return attribute_us_se_health_premiums(
        frame,
        reported_source_column=parameters["reported_source"],
        self_employment_income_source_columns=tuple(
            parameters["self_employment_income_sources"]
        ),
        medicare_age_source_column=parameters["medicare_age_source"],
        medicare_age_threshold=int(parameters["medicare_age_threshold"]),
        medicare_ssdi_source_column=parameters["medicare_ssdi_source"],
        employer_coverage_source_column=parameters["employer_coverage_source"],
        output_premiums_column=parameters["output_premiums"],
        output_flag_column=parameters["output_self_employed_flag"],
    )


def _person_other_health_insurance_predictors(frame: Frame) -> pd.DataFrame:
    """Build the retired eight second-stage QRF predictors on people."""

    person = frame.table("person")
    tax_unit = frame.table("tax_unit")
    predictors = pd.DataFrame(index=person.index)

    def _numeric(*columns: str) -> np.ndarray:
        for column in columns:
            if column in person:
                values = pd.to_numeric(person[column], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                if not np.isfinite(values).all():
                    raise SourceRuntimeError(
                        "US other-health-insurance QRF predictor source "
                        f"{column!r} contains nonfinite values."
                    )
                return values
        raise SourceRuntimeError(
            "US other-health-insurance PUF imputation cannot construct a "
            f"predictor from any of {list(columns)}."
        )

    predictors["age"] = _numeric("age", "A_AGE")
    if "is_male" in person:
        predictors["is_male"] = person["is_male"].fillna(False).astype(bool)
    elif "is_female" in person:
        predictors["is_male"] = ~person["is_female"].fillna(False).astype(bool)
    elif "A_SEX" in person:
        predictors["is_male"] = _numeric("A_SEX") == 1
    else:
        raise SourceRuntimeError(
            "US other-health-insurance PUF imputation requires is_male, "
            "is_female, or measured A_SEX."
        )
    if "has_esi" not in person:
        raise SourceRuntimeError(
            "US other-health-insurance PUF imputation requires has_esi."
        )
    predictors["has_esi"] = person["has_esi"].fillna(False).astype(bool)

    filing_status_column = next(
        (
            column
            for column in ("filing_status_input", "filing_status")
            if column in tax_unit
        ),
        None,
    )
    if filing_status_column is None:
        raise SourceRuntimeError(
            "US other-health-insurance PUF imputation requires tax-unit filing status."
        )
    filing_status = tax_unit[filing_status_column].map(
        lambda value: (
            value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        )
    )
    joint_by_id = pd.Series(
        filing_status.eq("JOINT").to_numpy(),
        index=tax_unit["tax_unit_id"].to_numpy(),
    )
    predictors["tax_unit_is_joint"] = (
        person["person_tax_unit_id"].map(joint_by_id).fillna(False).astype(bool)
    )

    if "tax_unit_role_input" not in person:
        raise SourceRuntimeError(
            "US other-health-insurance PUF imputation requires "
            "tax_unit_role_input to count dependents."
        )
    dependent = (
        person["tax_unit_role_input"]
        .map(
            lambda value: (
                value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
            )
        )
        .eq("DEPENDENT")
    )
    predictors["tax_unit_count_dependents"] = (
        dependent.groupby(person["person_tax_unit_id"])
        .transform("sum")
        .to_numpy(dtype=np.float64)
    )
    predictors["employment_income"] = _numeric(
        "employment_income_before_lsr", "WSAL_VAL"
    )
    predictors["self_employment_income"] = _numeric(
        "self_employment_income_before_lsr", "SEMP_VAL"
    )
    social_security_columns = (
        "social_security_retirement",
        "social_security_disability",
        "social_security_dependents",
        "social_security_survivors",
    )
    if all(column in person for column in social_security_columns):
        predictors["social_security"] = np.sum(
            np.column_stack([_numeric(column) for column in social_security_columns]),
            axis=1,
        )
    else:
        predictors["social_security"] = _numeric("SS_VAL")
    return predictors


def _tax_unit_values_on_first_person(
    frame: Frame,
    values: np.ndarray,
    *,
    variable: str,
) -> np.ndarray:
    """Allocate each tax-unit formula amount only to its first person."""

    tax_unit = frame.table("tax_unit")
    person = frame.table("person")
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.shape != (len(tax_unit),):
        raise ValueError(
            f"Materialized {variable!r} has shape {numeric.shape}; expected "
            f"{(len(tax_unit),)} for tax units."
        )
    if not np.isfinite(numeric).all() or bool((numeric < 0.0).any()):
        raise ValueError(f"Materialized {variable!r} must be finite and nonnegative.")

    by_id = pd.Series(numeric, index=tax_unit["tax_unit_id"].to_numpy())
    memberships = person["person_tax_unit_id"]
    mapped = memberships.map(by_id)
    if mapped.isna().any():
        missing = sorted(set(memberships[mapped.isna()].tolist()))
        raise ValueError(
            f"Cannot allocate {variable!r}; unknown person tax-unit IDs {missing}."
        )
    allocated = np.zeros(len(person), dtype=np.float64)
    first_person = ~memberships.duplicated(keep="first")
    allocated[first_person.to_numpy()] = mapped[first_person].to_numpy(dtype=np.float64)
    represented = set(memberships.tolist())
    missing_tax_units = sorted(set(tax_unit["tax_unit_id"].tolist()) - represented)
    if missing_tax_units:
        raise ValueError(
            f"Cannot allocate {variable!r}; tax units without people "
            f"{missing_tax_units}."
        )
    return allocated


def _materialize_modeled_premiums(
    frame: Frame,
    *,
    time_period: int,
    maximum_microsim_batch_size: int | None,
) -> dict[str, np.ndarray]:
    """Materialize tax-unit premiums in household batches and realign by ID."""

    household = frame.table("household")
    tax_unit = frame.table("tax_unit")
    person = frame.table("person")
    household_ids = household["household_id"].to_numpy()
    tax_unit_ids = tax_unit["tax_unit_id"].to_numpy()
    tax_unit_positions = pd.Series(
        np.arange(len(tax_unit_ids), dtype=np.int64),
        index=tax_unit_ids,
    )
    modeled = {
        variable: np.zeros(len(tax_unit_ids), dtype=np.float64)
        for variable in US_OTHER_HEALTH_INSURANCE_MODELED_PREMIUM_VARIABLES
    }
    assigned = np.zeros(len(tax_unit_ids), dtype=bool)
    batch_size = (
        len(household_ids)
        if maximum_microsim_batch_size is None or maximum_microsim_batch_size <= 0
        else min(int(maximum_microsim_batch_size), len(household_ids))
    )
    if batch_size == 0:
        raise ValueError(
            "US other-health-insurance premium materialization requires at "
            "least one household."
        )

    engine = PolicyEngineUSEngine()
    for start in range(0, len(household_ids), batch_size):
        positions = np.arange(
            start,
            min(start + batch_size, len(household_ids)),
            dtype=np.int64,
        )
        full_batch = len(positions) == len(household_ids)
        batch_frame = (
            frame
            if full_batch
            else frame.select(
                person["person_household_id"].isin(household_ids[positions])
            )
        )
        batch_tax_unit_ids = batch_frame.table("tax_unit")["tax_unit_id"].to_numpy()
        full_positions = tax_unit_positions.reindex(batch_tax_unit_ids).to_numpy()
        if np.isnan(full_positions).any():
            raise ValueError(
                "US other-health-insurance premium batch produced tax-unit IDs "
                "outside the full frame."
            )
        full_positions = full_positions.astype(np.int64)
        if assigned[full_positions].any():
            raise ValueError(
                "US other-health-insurance premium batches overlap tax units; "
                "tax units must not span household batches."
            )
        materialized = engine.materialize(
            batch_frame,
            list(US_OTHER_HEALTH_INSURANCE_MODELED_PREMIUM_VARIABLES),
            period=int(time_period),
        )
        for variable in US_OTHER_HEALTH_INSURANCE_MODELED_PREMIUM_VARIABLES:
            if variable not in materialized:
                raise ValueError(
                    "PolicyEngine-US did not materialize required formula "
                    f"{variable!r}."
                )
            values = np.asarray(materialized[variable], dtype=np.float64)
            if values.shape != (len(batch_tax_unit_ids),):
                raise ValueError(
                    f"Materialized {variable!r} has shape {values.shape}; expected "
                    f"{(len(batch_tax_unit_ids),)} for the tax-unit batch."
                )
            modeled[variable][full_positions] = values
        assigned[full_positions] = True
        del materialized, batch_frame
        gc.collect()

    if not assigned.all():
        missing_ids = tax_unit_ids[~assigned].tolist()
        raise ValueError(
            "US other-health-insurance premium batches did not cover tax-unit "
            f"IDs {missing_ids}."
        )
    return modeled


def with_us_other_health_insurance_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    maximum_microsim_batch_size: int | None = None,
    allow_existing_without_source: bool = False,
) -> Frame:
    """Materialize modeled premiums, derive ASEC residuals, and impute PUF."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US other-health-insurance inputs require the US schema.")
    person = frame.table("person")
    source_available = all(
        column in person for column in US_OTHER_HEALTH_INSURANCE_REQUIRED_SOURCE_COLUMNS
    )
    if not source_available:
        if (
            allow_existing_without_source
            and _other_health_insurance_surface_carries_signal(frame)
        ):
            return frame
        missing = [
            column
            for column in US_OTHER_HEALTH_INSURANCE_REQUIRED_SOURCE_COLUMNS
            if column not in person
        ]
        raise ValueError(
            "US other-health-insurance stage cannot heal a default surface "
            f"without measured ASEC source column(s): {missing}."
        )

    modeled = _materialize_modeled_premiums(
        frame,
        time_period=int(time_period),
        maximum_microsim_batch_size=maximum_microsim_batch_size,
    )
    stage_person = person.copy(deep=True)
    for variable in US_OTHER_HEALTH_INSURANCE_MODELED_PREMIUM_VARIABLES:
        stage_person[variable] = _tax_unit_values_on_first_person(
            frame,
            np.asarray(modeled[variable]),
            variable=variable,
        )
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    if has_support_role_metadata(person, entity="person"):
        predictors = _person_other_health_insurance_predictors(frame)
        for column in _PREDICTORS:
            stage_person[_PREDICTOR_PREFIX + column] = predictors[column].to_numpy()

    output = run_source_stage(
        us_other_health_insurance_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_other_health_insurance_premiums": (
                derive_us_other_health_insurance_from_manifest
            ),
            "impute_other_health_insurance_premiums_to_puf_support": (
                impute_us_other_health_insurance_to_puf_support_from_manifest
            ),
            "attribute_self_employed_health_premiums": (
                attribute_us_se_health_premiums_from_manifest
            ),
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    for column in US_OTHER_HEALTH_INSURANCE_STAGE_OUTPUT_COLUMNS:
        if aligned[column].isna().any():
            raise ValueError(
                "US other-health-insurance stage output "
                f"{column!r} does not cover every person."
            )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column in (*US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS, _SE_PREMIUMS_OUTPUT):
        tables["person"][column] = aligned[column].to_numpy(dtype=np.float64)
    tables["person"][_SE_FLAG_OUTPUT] = aligned[_SE_FLAG_OUTPUT].astype(bool).to_numpy()
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_other_health_insurance_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal, validity, channel, and residual diagnostics."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    channel = (
        support_role_series(person, entity="person").to_numpy()
        if has_support_role_metadata(person, entity="person")
        else None
    )
    columns: dict[str, dict[str, object]] = {}
    numeric_outputs: dict[str, np.ndarray] = {}
    for output in US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS:
        values = pd.to_numeric(person[output], errors="coerce").to_numpy(
            dtype=np.float64
        )
        numeric_outputs[output] = values
        finite = np.isfinite(values)
        positive = finite & (values > 0.0)
        detail: dict[str, object] = {
            "positive_share": (
                float(weights[positive].sum()) / total_weight
                if total_weight > 0.0
                else 0.0
            ),
            "positive_share_band": list(_POSITIVE_SHARE_BAND),
            "weighted_total": float((np.nan_to_num(values) * weights).sum()),
            "nonfinite": int(np.count_nonzero(~finite)),
            "negative": int(np.count_nonzero(finite & (values < 0.0))),
        }
        if channel is not None:
            channels: dict[str, dict[str, float | int | list[float]]] = {}
            for name in (_BASE_ASEC_SUPPORT_CHANNEL, _PUF_TAX_DETAIL_SUPPORT_CHANNEL):
                mask = channel == name
                channel_weight = float(weights[mask].sum())
                channels[name] = {
                    "positive_rows": int(np.count_nonzero(mask & positive)),
                    "positive_share": (
                        float(weights[mask & positive].sum()) / channel_weight
                        if channel_weight > 0.0
                        else 0.0
                    ),
                    "positive_share_band": list(_POSITIVE_SHARE_BAND),
                    "weighted_total": float(
                        (np.nan_to_num(values[mask]) * weights[mask]).sum()
                    ),
                }
            detail["channels"] = channels
        columns[output] = detail

    reported = numeric_outputs[_REPORTED_OUTPUT]
    other = numeric_outputs[_OTHER_OUTPUT]
    comparable = np.isfinite(reported) & np.isfinite(other)
    residual_violation = comparable & (other > reported)
    result: dict[str, object] = {
        "columns": columns,
        "other_exceeds_reported": int(np.count_nonzero(residual_violation)),
        "weighted_other_excess": float(
            (np.maximum(other - reported, 0.0)[comparable] * weights[comparable]).sum()
        ),
    }
    if channel is not None:
        result["other_exceeds_reported_by_channel"] = {
            name: int(np.count_nonzero(residual_violation & (channel == name)))
            for name in (_BASE_ASEC_SUPPORT_CHANNEL, _PUF_TAX_DETAIL_SUPPORT_CHANNEL)
        }
    result["se_attribution"] = _se_attribution_summary(person, weights, reported)
    return result


def _se_attribution_summary(
    person: pd.DataFrame,
    weights: np.ndarray,
    reported: np.ndarray,
) -> dict[str, object]:
    """Recompute the deterministic attribution identity from shipped columns."""

    missing = [
        column
        for column in (
            *US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS,
            *US_SE_HEALTH_SELF_EMPLOYMENT_INCOME_SOURCES,
            _SE_MEDICARE_AGE_SOURCE,
            _SE_MEDICARE_SSDI_SOURCE,
            _SE_EMPLOYER_COVERAGE_SOURCE,
        )
        if column not in person.columns
    ]
    if missing:
        return {"missing": missing}

    premiums = pd.to_numeric(person[_SE_PREMIUMS_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    # A malformed flag column (nulls, values outside {0, 1}, or a string or
    # object dtype — the engine parses the nonempty string "0" as True) must
    # fail the gate rather than be silently coerced.
    flag_numeric = pd.to_numeric(person[_SE_FLAG_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    invalid_flag_values = int(
        np.count_nonzero(
            ~np.isfinite(flag_numeric) | ~np.isin(flag_numeric, (0.0, 1.0))
        )
    )
    if not _is_boolean_like_dtype(person[_SE_FLAG_OUTPUT]):
        invalid_flag_values = max(invalid_flag_values, len(person))
    flag = flag_numeric == 1.0

    # The identity sources are release-required inputs; a nonfinite value in
    # any of them makes the recomputed identity meaningless, so it is a
    # failure in its own right, not a silently not-self-employed person.
    nonfinite_sources: dict[str, int] = {}
    self_employment = np.zeros(len(person), dtype=np.float64)
    for column in US_SE_HEALTH_SELF_EMPLOYMENT_INCOME_SOURCES:
        values = pd.to_numeric(person[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        bad = int(np.count_nonzero(~np.isfinite(values)))
        if bad:
            nonfinite_sources[column] = bad
        self_employment += values
    age = pd.to_numeric(person[_SE_MEDICARE_AGE_SOURCE], errors="coerce").to_numpy(
        dtype=np.float64
    )
    ssdi = pd.to_numeric(person[_SE_MEDICARE_SSDI_SOURCE], errors="coerce").to_numpy(
        dtype=np.float64
    )
    employer_numeric = pd.to_numeric(
        person[_SE_EMPLOYER_COVERAGE_SOURCE], errors="coerce"
    ).to_numpy(dtype=np.float64)
    for column, values in (
        (_SE_MEDICARE_AGE_SOURCE, age),
        (_SE_MEDICARE_SSDI_SOURCE, ssdi),
        (_SE_EMPLOYER_COVERAGE_SOURCE, employer_numeric),
    ):
        bad = int(np.count_nonzero(~np.isfinite(values)))
        if bad:
            nonfinite_sources[column] = bad
    self_employed = self_employment > 0.0
    medicare_proxy = (age >= float(US_SE_HEALTH_MEDICARE_AGE_THRESHOLD)) | (ssdi > 0.0)
    employer_covered = employer_numeric == 1.0
    expected = np.where(
        self_employed & ~medicare_proxy & ~employer_covered & np.isfinite(reported),
        reported,
        0.0,
    )

    finite = np.isfinite(premiums)
    positive = finite & (premiums > 0.0)
    total_weight = float(weights.sum())
    identity_violations = int(
        np.count_nonzero(~np.isclose(premiums, expected, rtol=0.0, atol=1e-9))
    )
    flag_violations = int(np.count_nonzero(flag != self_employed))
    return {
        "positive_rows": int(np.count_nonzero(positive)),
        "positive_share": (
            float(weights[positive].sum()) / total_weight if total_weight > 0.0 else 0.0
        ),
        "weighted_total": float((np.nan_to_num(premiums) * weights).sum()),
        "nonfinite": int(np.count_nonzero(~finite)),
        "negative": int(np.count_nonzero(finite & (premiums < 0.0))),
        "flag_rows": int(np.count_nonzero(flag)),
        "flag_share": (
            float(weights[flag].sum()) / total_weight if total_weight > 0.0 else 0.0
        ),
        "invalid_flag_values": invalid_flag_values,
        "nonfinite_sources": nonfinite_sources,
        "identity_violations": identity_violations,
        "flag_violations": flag_violations,
    }


def us_other_health_insurance_signal_gate(frame: Frame) -> GateResult:
    """Require plausible finite premiums and the residual ordering identity."""

    person = frame.table("person")
    missing = [
        output
        for output in US_OTHER_HEALTH_INSURANCE_STAGE_OUTPUT_COLUMNS
        if output not in person
    ]
    if missing:
        return GateResult(
            name="other_health_insurance_premiums_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_other_health_insurance_summary(frame)
    failures: list[str] = []
    columns = summary["columns"]
    for output in US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS:
        detail = columns[output]
        if detail["nonfinite"]:
            failures.append(f"{output}: {int(detail['nonfinite'])} nonfinite values.")
        if detail["negative"]:
            failures.append(f"{output}: {int(detail['negative'])} negative values.")
        share = float(detail["positive_share"])
        low, high = detail["positive_share_band"]
        if not (low <= share <= high):
            failures.append(
                f"{output}: positive share {share:.4f} outside plausibility "
                f"band [{low}, {high}]."
            )
        channels = detail.get("channels")
        if isinstance(channels, dict):
            for name in (_BASE_ASEC_SUPPORT_CHANNEL, _PUF_TAX_DETAIL_SUPPORT_CHANNEL):
                channel_detail = channels.get(name)
                if not isinstance(channel_detail, dict):
                    failures.append(f"{output}: missing {name} channel diagnostics.")
                    continue
                channel_share = float(channel_detail["positive_share"])
                channel_low, channel_high = channel_detail["positive_share_band"]
                if not (channel_low <= channel_share <= channel_high):
                    failures.append(
                        f"{output}: {name} positive share {channel_share:.4f} "
                        "outside plausibility band "
                        f"[{channel_low}, {channel_high}]."
                    )
    channel_violations = summary.get("other_exceeds_reported_by_channel")
    # The measured ASEC residual must preserve its exact source identity.  The
    # archived joint QRF did not reconcile its two independently predicted
    # premium leaves, so PUF-only exceedances remain diagnostics rather than a
    # post-hoc clipping rule the source never applied.
    identity_violations = (
        int(channel_violations.get(_BASE_ASEC_SUPPORT_CHANNEL, 0))
        if isinstance(channel_violations, dict)
        else int(summary["other_exceeds_reported"])
    )
    if identity_violations:
        failures.append(
            f"{_OTHER_OUTPUT}: {identity_violations} measured ASEC value(s) "
            f"exceed {_REPORTED_OUTPUT}."
        )
    attribution = summary.get("se_attribution")
    if not isinstance(attribution, dict):
        failures.append("se_attribution diagnostics missing from summary.")
    elif "missing" in attribution:
        failures.append(
            f"se_attribution identity sources missing: {attribution['missing']}."
        )
    else:
        if attribution["invalid_flag_values"]:
            failures.append(
                f"{_SE_FLAG_OUTPUT}: {int(attribution['invalid_flag_values'])} "
                "null or non-boolean value(s)."
            )
        if attribution["nonfinite_sources"]:
            failures.append(
                "se_attribution identity sources carry nonfinite values: "
                f"{dict(attribution['nonfinite_sources'])}."
            )
        if attribution["nonfinite"]:
            failures.append(
                f"{_SE_PREMIUMS_OUTPUT}: {int(attribution['nonfinite'])} "
                "nonfinite values."
            )
        if attribution["negative"]:
            failures.append(
                f"{_SE_PREMIUMS_OUTPUT}: {int(attribution['negative'])} "
                "negative values."
            )
        if attribution["identity_violations"] or attribution["flag_violations"]:
            failures.append(
                "se_attribution identity broken: "
                f"{int(attribution['identity_violations'])} premium and "
                f"{int(attribution['flag_violations'])} flag value(s) diverge "
                "from the deterministic reported-premium attribution."
            )
        if not attribution["positive_rows"]:
            failures.append(
                f"{_SE_PREMIUMS_OUTPUT}: no positive attributed premiums; the "
                "self-employed health ALD surface would be a structural zero."
            )
    return GateResult(
        name="other_health_insurance_premiums_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )


def _other_health_insurance_surface_carries_signal(frame: Frame) -> bool:
    if any(
        output not in frame.table("person")
        for output in US_OTHER_HEALTH_INSURANCE_STAGE_OUTPUT_COLUMNS
    ):
        return False
    return us_other_health_insurance_signal_gate(frame).passed
