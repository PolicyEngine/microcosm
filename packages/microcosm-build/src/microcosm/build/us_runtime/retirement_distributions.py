"""Measured ASEC retirement-account distributions by account type.

The archived eCPS construction at commit
``42ed5d45c56df80d754fbe24cce21cfeb8d05cbe`` reads four paired ASEC
distribution slots in ``datasets/cps/cps.py`` lines 1448-1481.  Each slot has
an account code (``DST_SC1``, ``DST_SC2``, ``DST_SC1_YNG``, or
``DST_SC2_YNG``) and its measured annual amount.  Codes 1 through 6 identify
401(k), 403(b), Roth IRA, regular IRA, Keogh, and SEP accounts respectively.

The same archived code and
``datasets/cps/imputation_parameters.yaml`` lines 10-15 treat 401(k), 403(b),
regular-IRA, SEP, and Keogh distributions as fully taxable, while Roth-IRA
distributions are tax exempt.  After support cloning, archived
``datasets/cps/extended_cps.py`` lines 140-148, 639-745, and 1014-1073 replace,
among the populated leaves restored here, the 401(k), 403(b), Keogh, and SEP
values on the PUF half with CPS-trained QRF predictions. The archive also names
three tax-exempt employer-account companions; their taxable fractions are 1.0,
so they remain zero and are outside the populated coverage surface. The
measured ASEC half remains exact, the PUF taxable-IRA source remains untouched,
and the copied Roth-IRA value remains a direct ASEC mapping. The stage does not
allocate a total across account types or manufacture values for the ASEC code-7
residual category.
"""

from __future__ import annotations

from importlib import import_module
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
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "RETIREMENT_DISTRIBUTIONS_ARCHIVED_DERIVATION_URL",
    "RETIREMENT_DISTRIBUTIONS_ARCHIVED_PARAMETERS_URL",
    "US_RETIREMENT_DISTRIBUTION_NONCONSTANT_PERSON_COLUMNS",
    "US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS",
    "US_RETIREMENT_DISTRIBUTION_REQUIRED_SOURCE_COLUMNS",
    "US_RETIREMENT_DISTRIBUTION_STAGE_NAME",
    "derive_us_retirement_distributions_from_manifest",
    "impute_us_retirement_distributions_to_puf_support_from_manifest",
    "us_retirement_distributions_signal_gate",
    "us_retirement_distributions_stage_spec",
    "us_retirement_distributions_summary",
    "with_us_retirement_distribution_inputs",
]

QRF: Any | None = None

_ARCHIVED_COMMIT = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_PACKAGE_PATH = "policyengine_" + "us_data"
RETIREMENT_DISTRIBUTIONS_ARCHIVED_DERIVATION_URL = (
    f"https://github.com/PolicyEngine/{_ARCHIVED_DATA_REPOSITORY}/blob/"
    f"{_ARCHIVED_COMMIT}/{_ARCHIVED_PACKAGE_PATH}/datasets/cps/"
    "cps.py#L1448-L1481"
)
RETIREMENT_DISTRIBUTIONS_ARCHIVED_PARAMETERS_URL = (
    f"https://github.com/PolicyEngine/{_ARCHIVED_DATA_REPOSITORY}/blob/"
    f"{_ARCHIVED_COMMIT}/{_ARCHIVED_PACKAGE_PATH}/datasets/cps/"
    "imputation_parameters.yaml#L10-L15"
)

US_RETIREMENT_DISTRIBUTION_STAGE_NAME = "retirement_distributions"

US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS: tuple[str, ...] = (
    "taxable_401k_distributions",
    "taxable_403b_distributions",
    "tax_exempt_ira_distributions",
    "taxable_ira_distributions",
    "keogh_distributions",
    "taxable_sep_distributions",
)
US_RETIREMENT_DISTRIBUTION_NONCONSTANT_PERSON_COLUMNS = (
    US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS
)

_DISTRIBUTION_SLOT_SUFFIXES = ("1", "2", "1_YNG", "2_YNG")
US_RETIREMENT_DISTRIBUTION_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = tuple(
    column
    for suffix in _DISTRIBUTION_SLOT_SUFFIXES
    for column in (f"DST_SC{suffix}", f"DST_VAL{suffix}")
)

_EXPECTED_OUTPUT_BY_ACCOUNT_CODE: dict[int, str] = {
    1: "taxable_401k_distributions",
    2: "taxable_403b_distributions",
    3: "tax_exempt_ira_distributions",
    4: "taxable_ira_distributions",
    5: "keogh_distributions",
    6: "taxable_sep_distributions",
}
_VALID_ACCOUNT_CODES = frozenset(range(8))
_DERIVE_PARAMETER_KEYS = frozenset({"output_by_account_code"})
_PERSON_WEIGHT_COLUMN = "person_weight"
_BASE_ASEC_SUPPORT_CHANNEL = "asec"
_PUF_TAX_DETAIL_SUPPORT_CHANNEL = "puf_tax_detail"
_PUF_PREDICTORS: tuple[str, ...] = (
    "age",
    "is_male",
    "has_esi",
    "tax_unit_is_joint",
    "tax_unit_count_dependents",
    "employment_income",
    "self_employment_income",
    "social_security",
)
# Populated/restored subset of the archived stage-2 CPS-only retirement outputs.
# IRA distributions were not stage-2 QRF targets; the three archived tax-exempt
# employer-account companions remain default zero under the 1.0 taxable shares.
_PUF_QRF_OUTPUT_COLUMNS: tuple[str, ...] = (
    "taxable_401k_distributions",
    "taxable_403b_distributions",
    "keogh_distributions",
    "taxable_sep_distributions",
)
_PUF_PREDICTOR_PREFIX = "retirement_distribution_predictor_"
_PUF_IMPUTATION_PARAMETER_KEYS = frozenset(
    {
        "predictors",
        "max_train_samples",
        "n_estimators",
        "seed_from_build_config",
        "weight",
    }
)

# Weighted shares are intentionally broad but family-specific.  Their purpose
# is to reject a missing/default surface while accommodating source-year and
# L0-selection variation.  The pinned eCPS shares range from 0.00002 (Keogh) to
# 0.0658 (regular IRA).
_NONZERO_SHARE_BANDS: dict[str, tuple[float, float]] = {
    "taxable_401k_distributions": (0.001, 0.20),
    "taxable_403b_distributions": (0.00005, 0.05),
    "tax_exempt_ira_distributions": (0.0001, 0.05),
    "taxable_ira_distributions": (0.001, 0.25),
    "keogh_distributions": (0.0000001, 0.005),
    "taxable_sep_distributions": (0.00001, 0.05),
}


def us_retirement_distributions_stage_spec() -> SourceStageSpec:
    """Load the packaged retirement-distribution stage declaration."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_RETIREMENT_DISTRIBUTION_STAGE_NAME not in stage_map:
        raise ValueError(
            "US source manifest declares no "
            f"{US_RETIREMENT_DISTRIBUTION_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_RETIREMENT_DISTRIBUTION_STAGE_NAME]
    if tuple(spec.outputs) != US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS:
        raise ValueError(
            f"{US_RETIREMENT_DISTRIBUTION_STAGE_NAME!r} manifest outputs do "
            "not match the runtime-owned retirement-distribution family."
        )
    return spec


def _manifest_output_by_code(operation: SourceOperationSpec) -> dict[int, str]:
    unexpected = sorted(set(operation.parameters) - _DERIVE_PARAMETER_KEYS)
    if unexpected:
        raise SourceRuntimeError(
            "US retirement-distribution derivation received unsupported "
            f"parameter(s): {unexpected}."
        )
    raw = operation.parameters.get("output_by_account_code")
    if not isinstance(raw, dict):
        raise SourceRuntimeError(
            "US retirement-distribution derivation requires an "
            "output_by_account_code mapping."
        )
    try:
        parsed = {int(code): str(output) for code, output in raw.items()}
    except (TypeError, ValueError) as exc:
        raise SourceRuntimeError(
            "US retirement-distribution account codes must be integers."
        ) from exc
    if parsed != _EXPECTED_OUTPUT_BY_ACCOUNT_CODE:
        raise SourceRuntimeError(
            "US retirement-distribution account-code mapping drifted from "
            f"the archived source: expected {_EXPECTED_OUTPUT_BY_ACCOUNT_CODE}, "
            f"got {parsed}."
        )
    return parsed


def _strict_source_arrays(
    frame: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    codes: dict[str, np.ndarray] = {}
    amounts: dict[str, np.ndarray] = {}
    for suffix in _DISTRIBUTION_SLOT_SUFFIXES:
        code_column = f"DST_SC{suffix}"
        amount_column = f"DST_VAL{suffix}"
        numeric_code = pd.to_numeric(frame[code_column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        valid_code = np.isfinite(numeric_code) & (
            numeric_code == np.floor(numeric_code)
        )
        valid_code &= np.isin(
            numeric_code,
            np.fromiter(_VALID_ACCOUNT_CODES, dtype=np.int64),
        )
        if not valid_code.all():
            rows = np.flatnonzero(~valid_code)[:5].tolist()
            raise SourceRuntimeError(
                "US retirement-distribution source requires account codes in "
                f"0..7; {code_column} has invalid row(s) {rows}."
            )
        numeric_amount = pd.to_numeric(frame[amount_column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        valid_amount = np.isfinite(numeric_amount) & (numeric_amount >= 0)
        if not valid_amount.all():
            rows = np.flatnonzero(~valid_amount)[:5].tolist()
            raise SourceRuntimeError(
                "US retirement-distribution source requires finite, "
                f"nonnegative amounts; {amount_column} has invalid row(s) {rows}."
            )
        code = numeric_code.astype(np.int64)
        orphan_amount = (code == 0) & (numeric_amount != 0)
        if orphan_amount.any():
            rows = np.flatnonzero(orphan_amount)[:5].tolist()
            raise SourceRuntimeError(
                f"US retirement-distribution source {amount_column} has a "
                f"positive amount with NIU code 0 at row(s) {rows}."
            )
        codes[suffix] = code
        amounts[suffix] = numeric_amount
    return codes, amounts


def _derived_outputs(
    frame: pd.DataFrame,
    output_by_code: dict[int, str],
) -> dict[str, np.ndarray]:
    codes, amounts = _strict_source_arrays(frame)
    outputs = {
        output: np.zeros(len(frame), dtype=np.float64)
        for output in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS
    }
    for suffix in _DISTRIBUTION_SLOT_SUFFIXES:
        for code, output in output_by_code.items():
            outputs[output] += np.where(codes[suffix] == code, amounts[suffix], 0.0)
    return outputs


def derive_us_retirement_distributions_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    _context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Map the four measured ASEC account/amount pairs to six input leaves."""

    if operation.kind != "derive_retirement_distributions":
        raise SourceRuntimeError(
            "US retirement-distribution derivation received unexpected "
            f"operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US retirement-distribution derivation requires the person table "
            "to be read first."
        )
    missing = [
        column
        for column in US_RETIREMENT_DISTRIBUTION_REQUIRED_SOURCE_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise SourceRuntimeError(
            "US retirement-distribution derivation requires measured ASEC "
            f"source column(s): {missing}."
        )
    output_by_code = _manifest_output_by_code(operation)
    result = frame.copy(deep=True)
    derived = _derived_outputs(result, output_by_code)
    preserved_puf_taxable_ira: np.ndarray | None = None
    puf_mask: np.ndarray | None = None
    if has_support_role_metadata(result, entity="person"):
        if "taxable_ira_distributions" not in result:
            raise SourceRuntimeError(
                "US retirement-distribution support derivation requires the "
                "PUF-sourced taxable_ira_distributions column."
            )
        puf_mask = (
            support_role_series(result, entity="person").to_numpy()
            == _PUF_TAX_DETAIL_SUPPORT_CHANNEL
        )
        preserved_puf_taxable_ira = pd.to_numeric(
            result["taxable_ira_distributions"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        valid = np.isfinite(preserved_puf_taxable_ira[puf_mask]) & (
            preserved_puf_taxable_ira[puf_mask] >= 0
        )
        if not valid.all():
            rows = np.flatnonzero(puf_mask)[np.flatnonzero(~valid)[:5]].tolist()
            raise SourceRuntimeError(
                "US retirement-distribution support derivation requires "
                "finite, nonnegative PUF taxable IRA values; invalid row(s) "
                f"{rows}."
            )

    for output, values in derived.items():
        result[output] = values
        if output == "taxable_ira_distributions" and puf_mask is not None:
            assert preserved_puf_taxable_ira is not None
            result.loc[puf_mask, output] = preserved_puf_taxable_ira[puf_mask]
    return result


def impute_us_retirement_distributions_to_puf_support_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Replace PUF-support copies with the retired CPS-trained QRF outputs.

    The first (unexpanded ASEC) pass is a no-op.  Once the support spine has an
    ASEC half and a PUF-tax-detail half, the common archived eight-predictor QRF
    trains on measured ASEC distributions and replaces the four populated
    non-IRA outputs restored here on the PUF half. PUF taxable IRA remains
    sourced from the tax-detail stage, and Roth IRA remains the direct copied
    ASEC mapping. The fixed training cap retains the union of nonzero donors
    across all four QRF targets, then samples only the common all-zero stratum
    and calibrates its weights back to the full all-zero mass. This preserves
    every realized sign regime and conditional nonzero-magnitude support while
    the measured ASEC half remains authoritative and is never overwritten.
    """

    if operation.kind != "impute_retirement_distributions_to_puf_support":
        raise SourceRuntimeError(
            "US retirement-distribution PUF imputation received unexpected "
            f"operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US retirement-distribution PUF imputation requires the person "
            "table to be read first."
        )
    unexpected = sorted(set(operation.parameters) - _PUF_IMPUTATION_PARAMETER_KEYS)
    missing_parameters = sorted(
        _PUF_IMPUTATION_PARAMETER_KEYS - set(operation.parameters)
    )
    if unexpected or missing_parameters:
        raise SourceRuntimeError(
            "US retirement-distribution PUF imputation parameters must match "
            f"the archived method; missing={missing_parameters}, "
            f"unexpected={unexpected}."
        )
    if not has_support_role_metadata(frame, entity="person"):
        return frame.copy(deep=True)

    predictors = tuple(str(value) for value in operation.parameters["predictors"])
    if predictors != _PUF_PREDICTORS:
        raise SourceRuntimeError(
            "US retirement-distribution PUF predictors drifted from the "
            f"archived method: expected {list(_PUF_PREDICTORS)}, got "
            f"{list(predictors)}."
        )
    if operation.parameters["weight"] != _PERSON_WEIGHT_COLUMN:
        raise SourceRuntimeError(
            "US retirement-distribution PUF imputation must use the typed "
            f"person-weight column {_PERSON_WEIGHT_COLUMN!r}."
        )
    if operation.parameters["seed_from_build_config"] is not True:
        raise SourceRuntimeError(
            "US retirement-distribution PUF imputation seed must come from "
            "the build config."
        )
    max_train_samples = int(operation.parameters["max_train_samples"])
    n_estimators = int(operation.parameters["n_estimators"])
    if max_train_samples <= 0 or n_estimators <= 0:
        raise SourceRuntimeError(
            "US retirement-distribution PUF max_train_samples and n_estimators "
            "must be positive."
        )

    required = [
        _PERSON_WEIGHT_COLUMN,
        *(_PUF_PREDICTOR_PREFIX + predictor for predictor in predictors),
        *_PUF_QRF_OUTPUT_COLUMNS,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SourceRuntimeError(
            "US retirement-distribution PUF imputation is missing source "
            f"column(s): {missing}."
        )

    roles = support_role_series(frame, entity="person")
    asec_mask = roles == _BASE_ASEC_SUPPORT_CHANNEL
    puf_mask = roles == _PUF_TAX_DETAIL_SUPPORT_CHANNEL
    if not asec_mask.any() or not puf_mask.any():
        raise SourceRuntimeError(
            "US retirement-distribution PUF imputation requires nonempty ASEC "
            "and PUF-tax-detail support channels."
        )

    predictor_columns = [_PUF_PREDICTOR_PREFIX + value for value in predictors]
    training = frame.loc[
        asec_mask,
        [*predictor_columns, *_PUF_QRF_OUTPUT_COLUMNS],
    ].copy()
    training.columns = [*predictors, *_PUF_QRF_OUTPUT_COLUMNS]
    weights = pd.to_numeric(
        frame.loc[asec_mask, _PERSON_WEIGHT_COLUMN], errors="coerce"
    ).fillna(0.0)
    for column in _PUF_QRF_OUTPUT_COLUMNS:
        training[column] = pd.to_numeric(training[column], errors="coerce")
        if not np.isfinite(training[column].to_numpy(dtype=np.float64)).all():
            raise SourceRuntimeError(
                "US retirement-distribution QRF training column "
                f"{column!r} contains nonfinite values."
            )
    fit_module = import_module("microcosm.fit")
    seed = context.config.seed if context is not None else 0
    training, capped_weights = _cap_retirement_distribution_training(
        training,
        weights,
        targets=_PUF_QRF_OUTPUT_COLUMNS,
        max_train_samples=max_train_samples,
        seed=seed,
        zero_atol=float(fit_module.DEFAULT_ZERO_ATOL),
    )

    test = frame.loc[puf_mask, predictor_columns].copy()
    test.columns = list(predictors)
    for column in predictors:
        training[column] = pd.to_numeric(training[column], errors="coerce")
        if not np.isfinite(training[column].to_numpy(dtype=np.float64)).all():
            raise SourceRuntimeError(
                "US retirement-distribution QRF training column "
                f"{column!r} contains nonfinite values."
            )
    for column in predictors:
        test[column] = pd.to_numeric(test[column], errors="coerce")
        if not np.isfinite(test[column].to_numpy(dtype=np.float64)).all():
            raise SourceRuntimeError(
                "US retirement-distribution QRF prediction column "
                f"{column!r} contains nonfinite values."
            )

    global QRF
    if QRF is None:
        QRF = fit_module.QRF
    fitted = QRF(n_estimators=n_estimators, seed=seed).fit(
        training,
        list(predictors),
        list(_PUF_QRF_OUTPUT_COLUMNS),
        weights=capped_weights,
    )
    predictions = fitted.predict(test).clip(lower=0.0)

    result = frame.copy(deep=True)
    for column in _PUF_QRF_OUTPUT_COLUMNS:
        result.loc[puf_mask, column] = predictions[column].to_numpy(dtype=np.float64)
    return result


def _cap_retirement_distribution_training(
    training: pd.DataFrame,
    weights: pd.Series | np.ndarray,
    *,
    targets: tuple[str, ...],
    max_train_samples: int,
    seed: int,
    zero_atol: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Cap a joint retirement fit without deleting any target's sign support.

    Every row that is nonzero for at least one target remains a census row.
    The fixed-cap remainder is a seeded sample of rows that are QRF-zero for
    every target. Scaling only those sampled rows back to their stratum's full
    weight preserves total donor mass and every target's weighted sign masses,
    while carrier weights and conditional magnitudes remain exact.
    """

    if max_train_samples <= 0:
        raise SourceRuntimeError(
            "US retirement-distribution training cap must be positive."
        )
    if zero_atol < 0.0 or not np.isfinite(zero_atol):
        raise SourceRuntimeError(
            "US retirement-distribution QRF zero tolerance must be finite and "
            "nonnegative."
        )
    weight_values = np.asarray(weights, dtype=np.float64)
    if weight_values.ndim != 1 or len(weight_values) != len(training):
        raise SourceRuntimeError(
            "US retirement-distribution weights must be one-dimensional and "
            "positionally aligned with the training donor."
        )
    if not np.isfinite(weight_values).all() or (weight_values < 0.0).any():
        raise SourceRuntimeError(
            "US retirement-distribution weights must be finite and nonnegative."
        )
    if float(weight_values.sum()) <= 0.0:
        raise SourceRuntimeError(
            "US retirement-distribution training donor has zero total weight."
        )

    target_matrix = training.loc[:, list(targets)].to_numpy(dtype=np.float64)
    if not np.isfinite(target_matrix).all():
        raise SourceRuntimeError(
            "US retirement-distribution training targets must be finite before "
            "the support-preserving cap."
        )
    if len(training) <= max_train_samples:
        return training.copy(), weight_values.copy()

    union_nonzero = np.any(np.abs(target_matrix) > zero_atol, axis=1)
    carrier_positions = np.flatnonzero(union_nonzero)
    zero_positions = np.flatnonzero(~union_nonzero)
    if len(carrier_positions) >= max_train_samples:
        raise SourceRuntimeError(
            "US retirement-distribution nonzero donor union leaves no room for "
            "an all-target-zero row under the fixed training cap: "
            f"union={len(carrier_positions)}, cap={max_train_samples}."
        )

    zero_slots = max_train_samples - len(carrier_positions)
    positive_weight_zero_positions = zero_positions[weight_values[zero_positions] > 0.0]
    zero_weight_zero_positions = zero_positions[weight_values[zero_positions] == 0.0]
    if len(positive_weight_zero_positions) >= zero_slots:
        sampled_zero_positions = (
            pd.Series(positive_weight_zero_positions, dtype=np.int64)
            .sample(n=zero_slots, random_state=seed)
            .to_numpy(dtype=np.int64)
        )
    else:
        zero_weight_slots = zero_slots - len(positive_weight_zero_positions)
        sampled_zero_positions = np.concatenate(
            (
                positive_weight_zero_positions,
                pd.Series(zero_weight_zero_positions, dtype=np.int64)
                .sample(n=zero_weight_slots, random_state=seed)
                .to_numpy(dtype=np.int64),
            )
        )
    selected_positions = np.sort(
        np.concatenate((carrier_positions, sampled_zero_positions))
    )
    capped_weights = weight_values[selected_positions].copy()
    selected_zero = ~union_nonzero[selected_positions]
    full_zero_mass = float(weight_values[zero_positions].sum())
    sampled_zero_mass = float(capped_weights[selected_zero].sum())
    if sampled_zero_mass <= 0.0 < full_zero_mass:
        raise SourceRuntimeError(
            "US retirement-distribution sampled all-target-zero donor has zero "
            "weight but must represent positive full-stratum mass."
        )
    if full_zero_mass > 0.0:
        capped_weights[selected_zero] *= full_zero_mass / sampled_zero_mass
    if not np.isfinite(capped_weights).all():
        raise SourceRuntimeError(
            "US retirement-distribution calibrated training weights are nonfinite."
        )
    return training.iloc[selected_positions].copy(), capped_weights


def _person_retirement_distribution_predictors(frame: Frame) -> pd.DataFrame:
    """Build the archived eight QRF predictors on person rows."""

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
                        "US retirement-distribution QRF predictor source "
                        f"{column!r} contains nonfinite values."
                    )
                return values
        raise SourceRuntimeError(
            "US retirement-distribution PUF imputation cannot construct "
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
            "US retirement-distribution PUF imputation requires is_male, "
            "is_female, or measured A_SEX."
        )
    if "has_esi" not in person:
        raise SourceRuntimeError(
            "US retirement-distribution PUF imputation requires has_esi."
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
            "US retirement-distribution PUF imputation requires tax-unit filing status."
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
            "US retirement-distribution PUF imputation requires "
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
    predictors["tax_unit_count_dependents"] = dependent.groupby(
        person["person_tax_unit_id"]
    ).transform("sum")
    predictors["employment_income"] = _numeric(
        "employment_income_before_lsr",
        "WSAL_VAL",
    )
    predictors["self_employment_income"] = _numeric(
        "self_employment_income_before_lsr",
        "SEMP_VAL",
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


def _retirement_distribution_surface_carries_signal(frame: Frame) -> bool:
    person = frame.table("person")
    if any(
        column not in person for column in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS
    ):
        return False
    return all(
        person[column].dropna().nunique() > 1
        for column in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS
    )


def with_us_retirement_distribution_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    force_puf_imputation: bool = False,
) -> Frame:
    """Materialize measured retirement-distribution leaves on a US frame.

    ``force_puf_imputation`` belongs only at the base builder's post-clone
    boundary.  Every later support-frame call is consume-only, including when
    a frozen selection is missing or has flattened a rare leaf.  Refitting
    there would make support selection redefine the donor universe and can
    broadcast a rare leaf such as ``keogh_distributions`` across the retained
    PUF rows.  The downstream signal gate, rather than a refit, owns support
    surface completeness and signal.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("US retirement distributions require the US schema.")
    person = frame.table("person")
    has_support_roles = has_support_role_metadata(person, entity="person")
    if has_support_roles and not force_puf_imputation:
        return frame
    if not has_support_roles and _retirement_distribution_surface_carries_signal(frame):
        return frame

    stage_person = person.copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    if has_support_roles:
        predictors = _person_retirement_distribution_predictors(frame)
        for column in _PUF_PREDICTORS:
            stage_person[_PUF_PREDICTOR_PREFIX + column] = predictors[column].to_numpy()
    output = run_source_stage(
        us_retirement_distributions_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_retirement_distributions": (
                derive_us_retirement_distributions_from_manifest
            ),
            "impute_retirement_distributions_to_puf_support": (
                impute_us_retirement_distributions_to_puf_support_from_manifest
            ),
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    for column in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS:
        if aligned[column].isna().any():
            raise ValueError(
                "US retirement-distribution stage output does not cover every "
                f"person for {column!r}."
            )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS:
        tables["person"][column] = aligned[column].to_numpy(dtype=np.float64)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_retirement_distributions_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal and exact source-reconciliation diagnostics."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    values = {
        column: pd.to_numeric(person[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        for column in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS
    }

    source_mismatches: dict[str, int] = {}
    if all(
        column in person
        for column in US_RETIREMENT_DISTRIBUTION_REQUIRED_SOURCE_COLUMNS
    ):
        expected = _derived_outputs(person, _EXPECTED_OUTPUT_BY_ACCOUNT_CODE)
        compare = np.ones(len(person), dtype=bool)
        if has_support_role_metadata(person, entity="person"):
            compare = (
                support_role_series(person, entity="person").to_numpy()
                == _BASE_ASEC_SUPPORT_CHANNEL
            )
        source_mismatches = {
            column: int(
                np.count_nonzero(
                    compare
                    & ~np.isclose(values[column], expected[column], rtol=0, atol=0)
                )
            )
            for column in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS
        }

    def _share(array: np.ndarray) -> float:
        positive = np.isfinite(array) & (array > 0)
        return (
            float(weights[positive].sum()) / total_weight if total_weight > 0 else 0.0
        )

    return {
        "nonzero_shares": {column: _share(array) for column, array in values.items()},
        "weighted_totals": {
            column: float(np.sum(np.nan_to_num(array) * weights))
            for column, array in values.items()
        },
        "nonzero_share_bands": {
            column: list(_NONZERO_SHARE_BANDS[column])
            for column in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS
        },
        "unique_counts": {
            column: int(person[column].dropna().nunique())
            for column in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS
        },
        "nonfinite": {
            column: int(np.count_nonzero(~np.isfinite(array)))
            for column, array in values.items()
        },
        "negative": {
            column: int(np.count_nonzero(array < 0)) for column, array in values.items()
        },
        "source_mismatches": source_mismatches,
    }


def us_retirement_distributions_signal_gate(frame: Frame) -> GateResult:
    """Require every measured account-type leaf to carry valid source signal."""

    person = frame.table("person")
    missing = [
        column
        for column in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS
        if column not in person
    ]
    if missing:
        return GateResult(
            name="retirement_distributions_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_retirement_distributions_summary(frame)
    failures: list[str] = []
    for column in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS:
        nonfinite = int(summary["nonfinite"][column])
        negative = int(summary["negative"][column])
        unique = int(summary["unique_counts"][column])
        share = float(summary["nonzero_shares"][column])
        low, high = summary["nonzero_share_bands"][column]
        mismatch = int(summary["source_mismatches"].get(column, 0))
        if nonfinite:
            failures.append(f"{column}: {nonfinite} nonfinite values.")
        if negative:
            failures.append(f"{column}: {negative} negative values.")
        if unique < 2:
            failures.append(f"{column}: degenerate with {unique} distinct value(s).")
        if not low <= share <= high:
            failures.append(
                f"{column}: weighted nonzero share {share:.8f} outside "
                f"[{low:.8f}, {high:.8f}]."
            )
        if mismatch:
            failures.append(
                f"{column}: {mismatch} row(s) differ from measured ASEC slots."
            )
    return GateResult(
        name="retirement_distributions_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
