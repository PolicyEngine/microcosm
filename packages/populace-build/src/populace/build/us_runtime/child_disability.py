"""SIPP-informed child disability input for ages 0--14.

CPS ASEC's six-question disability battery starts at age 15.  The certified
Populace base therefore carries ``is_disabled = False`` for every younger
child, making child SSI structurally absent (populace #453).  This stage
owns the generic under-15 ``is_disabled`` repair without replacing the adult
ASEC surface:

* Ages 5--14 receive seeded Bernoulli draws from a survey-weighted logistic
  classifier fit to the December 2023 SIPP.  Its outcome is any affirmative
  answer among ``ECOGNIT``, ``EHEARING``, ``ESEEING``, ``EAMBULAT``, and
  ``ESELFCARE``.  The deliberately small shared predictor set is age, sex, and
  a household-income proxy.  SIPP's personal monthly ``TPTOTINC`` is summed
  over December residence members aged 15+ and annualized; the recipient
  analogue is household-summed ``employment_income_before_lsr`` for members
  aged 15+.  Excluding younger members avoids using a child's own mostly-blank
  ``TPTOTINC`` as a near-direct SSI-receipt indicator.
* Ages 0--4 receive seeded Bernoulli draws at an explicit 8.991746% calibration
  target.  SIPP does not ask the full functional battery in that age band, so
  the target carries SIPP's observed 5--14 ratio of any-item disability to SSI
  receipt into the issue-specified 240,000 youngest-child caseload allocation.
  Receipt is measured on the matching monthly concept: December rows with
  ``RSSI_MNYN == 1``, weighted by ``WPFINWGT``.  The 240,000 value is a model
  target, not an observed SSA age-0--4 count.

Draws are keyed by stable person source identity, so support clones and row
reordering do not change assignment.  Existing child ``True`` values are never
cleared, and writes are restricted to under-15 rows so adult stored values are
untouched even when a source uses noncanonical boolean bytes.

The later ``ssi_disability_criteria`` stage remains the sole owner of
``meets_ssi_disability_criteria``.  It consumes this exact under-15 assignment
after its adult SIPP/QRF pass, so SSI sees the same seeded draw without a
second random decision.  The generic signal also intentionally reaches the
other PolicyEngine-US program families for which child disability is a real
input: HUD housing, SNAP, Medicaid, TANF, child-care assistance, and tax
provisions.  The stage gate reports weighted before/after shares so that
deliberate blast radius is measured rather than silent.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.build.us_runtime.voluntary_filing import (
    SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION,
    SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256,
    SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES,
    SIPP_2023_VOLUNTARY_FILING_DONOR_URL,
    fetch_sipp_2023_voluntary_filing_donor,
)
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

__all__ = [
    "CHILD_DISABILITY_SIPP_DICTIONARY_URL",
    "SIPP_2023_CHILD_DISABILITY_DONOR_REVISION",
    "SIPP_2023_CHILD_DISABILITY_DONOR_SHA256",
    "SIPP_2023_CHILD_DISABILITY_DONOR_SIZE_BYTES",
    "SIPP_2023_CHILD_DISABILITY_DONOR_URL",
    "SIPP_2023_CHILD_DISABILITY_LOCAL_PATH",
    "SIPP_CHILD_DISABILITY_AGE_0_4_PARAMETERS",
    "SIPP_CHILD_DISABILITY_FIT_PARAMETERS",
    "SIPP_CHILD_DISABILITY_MODEL_PREDICTORS",
    "SIPP_CHILD_DISABILITY_READ_PARAMETERS",
    "SIPP_CHILD_DISABILITY_SOURCE_COLUMNS",
    "SSA_SSI_AGE_0_4_CASELOAD_TARGET",
    "SSA_SSI_MONTHLY_MAY_2024_URL",
    "SSA_SSI_UNDER_18_RECIPIENT_TARGET",
    "US_CHILD_DISABILITY_AGE_0_4_SHARE_BAND",
    "US_CHILD_DISABILITY_AGE_0_4_TARGET_RATE",
    "US_CHILD_DISABILITY_AGE_5_14_SHARE_BAND",
    "US_CHILD_DISABILITY_OUTPUT_COLUMNS",
    "US_CHILD_DISABILITY_STAGE_NAME",
    "load_sipp_2023_child_disability_donor",
    "resolve_sipp_2023_child_disability_donor",
    "us_child_disability_signal_gate",
    "us_child_disability_stage_spec",
    "us_child_disability_summary",
    "with_us_child_disability_inputs",
]

CHILD_DISABILITY_SIPP_DICTIONARY_URL = (
    "https://www2.census.gov/programs-surveys/sipp/tech-documentation/"
    "data-dictionaries/2023/2023_SIPP_Data_Dictionary.pdf"
)

# Reuse the exact full-file coordinate already pinned by the SIPP vehicle,
# voluntary-filing, Head Start, and SSI-disability-criteria stages.
SIPP_2023_CHILD_DISABILITY_DONOR_REVISION = SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION
SIPP_2023_CHILD_DISABILITY_DONOR_SHA256 = SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256
SIPP_2023_CHILD_DISABILITY_DONOR_SIZE_BYTES = (
    SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES
)
SIPP_2023_CHILD_DISABILITY_DONOR_URL = SIPP_2023_VOLUNTARY_FILING_DONOR_URL

# Developer-machine fast path requested by populace #453.  Release builds use
# it only when no explicit --sipp-vehicle-donor path was supplied; the loader
# still verifies the immutable artifact's byte length and SHA-256.
SIPP_2023_CHILD_DISABILITY_LOCAL_PATH = (
    Path("/Users/maxghenis/PolicyEngine")
    / ("policyengine-" + "us-data")
    / ("policyengine_" + "us_data")
    / "storage"
    / "pu2023.csv"
)

# SSA SSI Monthly Statistics, May 2024, Table 1.  These constants are
# calibration targets, not SIPP survey observations.  The monthly table gives
# the under-18 total but no age-0--4 subtotal; 240,000 is the explicit age-curve
# allocation requested by #453 and must not be described as an SSA observation.
SSA_SSI_MONTHLY_MAY_2024_URL = (
    "https://www.ssa.gov/policy/docs/statcomps/ssi_monthly/2024-05/table01.html"
)
SSA_SSI_UNDER_18_RECIPIENT_TARGET = 983_176
SSA_SSI_AGE_0_4_CASELOAD_TARGET = 240_000

US_CHILD_DISABILITY_STAGE_NAME = "child_disability"
US_CHILD_DISABILITY_OUTPUT_COLUMNS: tuple[str, ...] = ("is_disabled",)

_OUTCOME_SOURCE_COLUMNS: tuple[str, ...] = (
    "ECOGNIT",
    "EHEARING",
    "ESEEING",
    "EAMBULAT",
    "ESELFCARE",
)
SIPP_CHILD_DISABILITY_SOURCE_COLUMNS: tuple[str, ...] = (
    "SSUID",
    "ERESIDENCEID",
    "PNUM",
    "MONTHCODE",
    "WPFINWGT",
    "TAGE",
    "ESEX",
    "TPTOTINC",
    "RSSI_MNYN",
    *_OUTCOME_SOURCE_COLUMNS,
)
SIPP_CHILD_DISABILITY_MODEL_PREDICTORS: tuple[str, ...] = (
    "age",
    "is_female",
    "household_income_proxy",
)

_OUTPUT = US_CHILD_DISABILITY_OUTPUT_COLUMNS[0]
_DONOR_WEIGHT_COLUMN = "sipp_weight"
_HOUSEHOLD_INCOME_COLUMN = "household_income_proxy"
_MIN_MODEL_AGE = 5
_MAX_MODEL_AGE = 14
_MAX_YOUNGEST_AGE = 4
_PERSON_SOURCE_ID_COLUMN = "person_source_id"
_PERSON_SUPPORT_CHANNEL_COLUMN = "person_support_channel"
_ASEC_SUPPORT_CHANNEL = "asec"
_PUF_SUPPORT_CHANNEL = "puf_tax_detail"

# Immutable-source audit, calculated over all 5--14 December rows.  A missing
# difficulty response is not affirmative under the requested ``any == 1``
# outcome; only three of 4,242 rows have any such missing item.
_PINNED_RAW_ROWS = 476_744
_PINNED_DECEMBER_ROWS = 39_513
_PINNED_AGE_5_14_ROWS = 4_242
_PINNED_AGE_5_14_WEIGHT = 40_772_887.197600
_PINNED_ANY_ITEM_ROWS = 495
_PINNED_ANY_ITEM_WEIGHT = 4_495_348.941131
_PINNED_ANY_ITEM_RATE = 0.110253387732
_PINNED_AGE_5_14_MONTHLY_SSI_ROWS = 70
_PINNED_AGE_5_14_MONTHLY_SSI_WEIGHT = 698_957.560971
_PINNED_AGE_5_14_MONTHLY_SSI_RATE = 0.017142704601
_PINNED_AGE_0_4_ROWS = 1_654
_PINNED_AGE_0_4_WEIGHT = 17_166_422.199210
_PINNED_FLOAT_RTOL = 1e-10

# Carry SIPP's observed functional-disability / SSI-receipt relationship into
# the youngest band after replacing the SIPP receipt prevalence with #453's
# explicit SSA caseload allocation:
#   0.110253387732 * [(240,000 / 17,166,422.199210) / 0.017142704601]
# The receipt denominator is the December monthly concept matching SSA's
# monthly table:
#   MONTHCODE == 12 and 5 <= TAGE <= 14 and RSSI_MNYN == 1,
#   weighted by WPFINWGT.
# The pinned pu2023.csv query yields 70 recipient rows with weighted mass
# 698,957.560971 over 40,772,887.197600 age-5--14 person-weight.
# This is a calibration choice, not a survey estimate or statutory threshold.
US_CHILD_DISABILITY_AGE_0_4_TARGET_RATE = round(
    _PINNED_ANY_ITEM_RATE
    * (
        (SSA_SSI_AGE_0_4_CASELOAD_TARGET / _PINNED_AGE_0_4_WEIGHT)
        / _PINNED_AGE_5_14_MONTHLY_SSI_RATE
    ),
    12,
)

# The 5--14 gate directly brackets the SIPP any-item observation.  The 0--4
# band preserves the reviewed +/-1.5 percentage-point finite-draw tolerance
# around the corrected monthly-concept target.
US_CHILD_DISABILITY_AGE_5_14_SHARE_BAND = (0.10, 0.13)
US_CHILD_DISABILITY_AGE_0_4_SHARE_BAND = (0.075, 0.105)

SIPP_CHILD_DISABILITY_READ_PARAMETERS: dict[str, object] = {
    "table": "sipp_person",
    "delimiter": "|",
    "month_column": "MONTHCODE",
    "month": 12,
    "source_columns": list(SIPP_CHILD_DISABILITY_SOURCE_COLUMNS),
}
SIPP_CHILD_DISABILITY_FIT_PARAMETERS: dict[str, object] = {
    "predictors": list(SIPP_CHILD_DISABILITY_MODEL_PREDICTORS),
    "target": _OUTPUT,
    "target_rule": f"any({list(_OUTCOME_SOURCE_COLUMNS)}) == 1",
    "weight": _DONOR_WEIGHT_COLUMN,
    "age_domain": [_MIN_MODEL_AGE, _MAX_MODEL_AGE],
    "classifier": "weighted_logistic_irls",
    "income_proxy": (
        "sum December TPTOTINC for age-15+ members within SSUID + "
        "ERESIDENCEID, multiply by 12; recipient analogue is age-15+ "
        "household-summed employment_income_before_lsr"
    ),
    "income_transform": "log1p_nonnegative_then_standardize",
    "calibrate_receiver_intercept_to_sipp_weighted_rate": True,
    "assignment": "seeded_bernoulli_by_person_source_id",
    "seed_from_build_config": True,
}
SIPP_CHILD_DISABILITY_AGE_0_4_PARAMETERS: dict[str, object] = {
    "age_domain": [0, _MAX_YOUNGEST_AGE],
    "rate": US_CHILD_DISABILITY_AGE_0_4_TARGET_RATE,
    "rate_role": "calibration_target_not_survey_observation",
    "ssa_under_18_recipient_target": SSA_SSI_UNDER_18_RECIPIENT_TARGET,
    "ssa_age_0_4_caseload_target": SSA_SSI_AGE_0_4_CASELOAD_TARGET,
    "ssa_source": SSA_SSI_MONTHLY_MAY_2024_URL,
    "sipp_any_item_rate_age_5_14": _PINNED_ANY_ITEM_RATE,
    "sipp_monthly_ssi_receipt_rate_age_5_14": (_PINNED_AGE_5_14_MONTHLY_SSI_RATE),
    "sipp_monthly_ssi_receipt_query": (
        "MONTHCODE == 12 and 5 <= TAGE <= 14 and RSSI_MNYN == 1, weighted by WPFINWGT"
    ),
    "sipp_weighted_population_age_0_4": _PINNED_AGE_0_4_WEIGHT,
    "assignment": "seeded_bernoulli_by_person_source_id",
    "seed_from_build_config": True,
}


def us_child_disability_stage_spec() -> SourceStageSpec:
    """Load and strictly validate the packaged child-disability stage."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_CHILD_DISABILITY_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_CHILD_DISABILITY_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_CHILD_DISABILITY_STAGE_NAME]
    if spec.grain != "person":
        raise ValueError("US child-disability stage must have person grain.")
    if tuple(spec.outputs) != US_CHILD_DISABILITY_OUTPUT_COLUMNS:
        raise ValueError(
            "US child-disability manifest outputs drifted from the runtime-owned "
            "family."
        )
    expected_operations = [
        ("read_table", SIPP_CHILD_DISABILITY_READ_PARAMETERS),
        ("fit_weighted_logistic", SIPP_CHILD_DISABILITY_FIT_PARAMETERS),
        ("assign_binary_from_rate", SIPP_CHILD_DISABILITY_AGE_0_4_PARAMETERS),
    ]
    if [operation.kind for operation in spec.operations] != [
        kind for kind, _ in expected_operations
    ]:
        raise ValueError(
            "US child-disability stage must contain read_table, "
            "fit_weighted_logistic, then assign_binary_from_rate."
        )
    for operation, (kind, parameters) in zip(
        spec.operations, expected_operations, strict=True
    ):
        if dict(operation.parameters) != parameters:
            raise ValueError(
                f"US child-disability {kind} contract drifted from the reviewed "
                "source transform."
            )
    pinned = [
        artifact
        for artifact in spec.artifacts
        if artifact.get("revision") == SIPP_2023_CHILD_DISABILITY_DONOR_REVISION
        and artifact.get("sha256") == SIPP_2023_CHILD_DISABILITY_DONOR_SHA256
        and artifact.get("size_bytes") == SIPP_2023_CHILD_DISABILITY_DONOR_SIZE_BYTES
    ]
    if not pinned:
        raise ValueError(
            "US child-disability stage does not pin the full SIPP revision, "
            "SHA-256, and byte length."
        )
    return spec


def resolve_sipp_2023_child_disability_donor(
    path: str | Path | None = None,
) -> Path:
    """Resolve an explicit path, a verified local file, then the pinned fetch.

    Explicit paths fail closed in the loader so a caller typo is never hidden.
    The hard-coded developer fast path is only preferred when both its byte
    length and SHA-256 match the immutable artifact.  A missing, stale, or
    unreadable local file falls through to the shared pinned donor chain.
    """

    if path is not None:
        return Path(path).expanduser()
    local_path = SIPP_2023_CHILD_DISABILITY_LOCAL_PATH
    try:
        local_is_pinned = (
            local_path.is_file()
            and local_path.stat().st_size == SIPP_2023_CHILD_DISABILITY_DONOR_SIZE_BYTES
            and _sha256_file(local_path) == SIPP_2023_CHILD_DISABILITY_DONOR_SHA256
        )
    except OSError:
        local_is_pinned = False
    if local_is_pinned:
        return local_path
    return fetch_sipp_2023_voluntary_filing_donor()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype(np.float64)


def _assert_pinned_audit(audit: dict[str, int | float]) -> None:
    expected_counts = {
        "raw_rows": _PINNED_RAW_ROWS,
        "december_rows": _PINNED_DECEMBER_ROWS,
        "age_5_14_rows": _PINNED_AGE_5_14_ROWS,
        "any_item_rows": _PINNED_ANY_ITEM_ROWS,
        "age_5_14_monthly_ssi_rows": _PINNED_AGE_5_14_MONTHLY_SSI_ROWS,
        "age_0_4_rows": _PINNED_AGE_0_4_ROWS,
    }
    mismatches: dict[str, tuple[int | float, int | float]] = {
        key: (expected, audit[key])
        for key, expected in expected_counts.items()
        if audit[key] != expected
    }
    expected_floats = {
        "age_5_14_weight": _PINNED_AGE_5_14_WEIGHT,
        "any_item_weight": _PINNED_ANY_ITEM_WEIGHT,
        "any_item_rate": _PINNED_ANY_ITEM_RATE,
        "age_5_14_monthly_ssi_weight": _PINNED_AGE_5_14_MONTHLY_SSI_WEIGHT,
        "age_5_14_monthly_ssi_rate": _PINNED_AGE_5_14_MONTHLY_SSI_RATE,
        "age_0_4_weight": _PINNED_AGE_0_4_WEIGHT,
    }
    for key, expected in expected_floats.items():
        actual = float(audit[key])
        if not np.isclose(actual, expected, rtol=_PINNED_FLOAT_RTOL, atol=1e-9):
            mismatches[key] = (expected, actual)
    if mismatches:
        raise ValueError(
            "Pinned SIPP child-disability source audit drifted from the "
            f"reviewed 2023 transform: {mismatches}."
        )


def load_sipp_2023_child_disability_donor(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = SIPP_2023_CHILD_DISABILITY_DONOR_SIZE_BYTES,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Load December SIPP children and construct the three-feature donor.

    All December persons are retained until residence income is aggregated;
    only then is the age-5--14 training universe selected.  Tests pass a tiny
    pipe-delimited fixture and disable artifact pins.
    """

    source_path = Path(path)
    if chunksize < 1:
        raise ValueError("chunksize must be a positive integer")
    actual_size = source_path.stat().st_size
    if expected_size_bytes is not None and actual_size != expected_size_bytes:
        raise ValueError(
            "SIPP 2023 child-disability donor failed byte-length verification: "
            f"expected {expected_size_bytes}, got {actual_size}."
        )
    if expected_sha256 is not None:
        actual_sha256 = _sha256_file(source_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "SIPP 2023 child-disability donor failed SHA-256 verification: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )

    header = pd.read_csv(source_path, sep="|", nrows=0)
    missing = sorted(set(SIPP_CHILD_DISABILITY_SOURCE_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"SIPP child-disability donor missing column(s): {missing}.")

    raw_rows = 0
    december_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        source_path,
        sep="|",
        usecols=list(SIPP_CHILD_DISABILITY_SOURCE_COLUMNS),
        chunksize=int(chunksize),
        low_memory=False,
    ):
        raw_rows += len(chunk)
        month = _numeric(chunk, "MONTHCODE")
        december = chunk.loc[month.eq(12.0)].copy()
        if not december.empty:
            december_parts.append(december)
    if not december_parts:
        raise ValueError("SIPP child-disability donor contains no December rows.")
    december = pd.concat(december_parts, ignore_index=True)

    identity_columns = ["SSUID", "ERESIDENCEID", "PNUM"]
    if december.loc[:, identity_columns].isna().any(axis=None):
        raise ValueError("SIPP child-disability December identity is incomplete.")
    if december.duplicated(identity_columns).any():
        raise ValueError("SIPP child-disability December identity is not unique.")

    age = _numeric(december, "TAGE")
    sex = _numeric(december, "ESEX")
    weight = _numeric(december, "WPFINWGT")
    valid_base = (
        age.between(0.0, 120.0, inclusive="both")
        & sex.isin([1.0, 2.0])
        & np.isfinite(weight)
        & weight.ge(0.0)
    )
    if not valid_base.all():
        rows = np.flatnonzero(~valid_base.to_numpy())[:5].tolist()
        raise ValueError(
            "SIPP child-disability age, sex, and weights must be valid and "
            "weights nonnegative; "
            f"invalid December row(s): {rows}."
        )

    residence_key = pd.MultiIndex.from_frame(december.loc[:, ["SSUID", "ERESIDENCEID"]])
    monthly_income = (
        _numeric(december, "TPTOTINC")
        .fillna(0.0)
        .clip(lower=0.0)
        .where(age.ge(15.0), 0.0)
    )
    annual_household_income = (
        pd.Series(monthly_income.to_numpy() * 12.0, index=december.index)
        .groupby(residence_key, sort=False)
        .transform("sum")
    )

    outcome_matrix = np.column_stack(
        [_numeric(december, column).eq(1.0) for column in _OUTCOME_SOURCE_COLUMNS]
    )
    any_item = outcome_matrix.any(axis=1)
    received_monthly_ssi = _numeric(december, "RSSI_MNYN").eq(1.0).to_numpy()
    age_5_14 = age.between(_MIN_MODEL_AGE, _MAX_MODEL_AGE, inclusive="both")
    age_0_4 = age.between(0, _MAX_YOUNGEST_AGE, inclusive="both")
    model_eligible = age_5_14 & weight.gt(0.0)

    donor = pd.DataFrame(
        {
            "age": age.loc[model_eligible].to_numpy(dtype=np.float64),
            "is_female": sex.loc[model_eligible].eq(2.0).to_numpy(),
            _HOUSEHOLD_INCOME_COLUMN: annual_household_income.loc[
                model_eligible
            ].to_numpy(dtype=np.float64),
            _OUTPUT: any_item[model_eligible.to_numpy()],
            _DONOR_WEIGHT_COLUMN: weight.loc[model_eligible].to_numpy(dtype=np.float64),
        }
    ).reset_index(drop=True)
    if donor.empty or donor[_OUTPUT].nunique() != 2:
        raise ValueError(
            "SIPP child-disability donor must contain both outcome classes."
        )

    band_weight = float(weight.loc[age_5_14].sum())
    positive_weight = float(weight.loc[age_5_14 & any_item].sum())
    monthly_ssi_weight = float(weight.loc[age_5_14 & received_monthly_ssi].sum())
    audit: dict[str, int | float] = {
        "raw_rows": raw_rows,
        "december_rows": len(december),
        "age_5_14_rows": int(age_5_14.sum()),
        "age_5_14_weight": band_weight,
        "any_item_rows": int(np.count_nonzero(age_5_14.to_numpy() & any_item)),
        "any_item_weight": positive_weight,
        "any_item_rate": positive_weight / band_weight,
        "age_5_14_monthly_ssi_rows": int(
            np.count_nonzero(age_5_14.to_numpy() & received_monthly_ssi)
        ),
        "age_5_14_monthly_ssi_weight": monthly_ssi_weight,
        "age_5_14_monthly_ssi_rate": monthly_ssi_weight / band_weight,
        "age_0_4_rows": int(age_0_4.sum()),
        "age_0_4_weight": float(weight.loc[age_0_4].sum()),
    }
    pinned_transform = bool(
        actual_size == SIPP_2023_CHILD_DISABILITY_DONOR_SIZE_BYTES
        and expected_sha256 == SIPP_2023_CHILD_DISABILITY_DONOR_SHA256
    )
    if pinned_transform:
        _assert_pinned_audit(audit)
    audit["pinned_transform"] = int(pinned_transform)
    donor.attrs["source_audit"] = audit
    return donor


@dataclass(frozen=True)
class _WeightedLogisticModel:
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray

    def predict_probability(self, features: pd.DataFrame) -> np.ndarray:
        raw = _logistic_feature_array(features)
        standardized = (raw - self.feature_mean) / self.feature_scale
        design = np.column_stack([np.ones(len(raw)), standardized])
        return _sigmoid(design @ self.coefficients)


def _logistic_feature_array(features: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(SIPP_CHILD_DISABILITY_MODEL_PREDICTORS) - set(features))
    if missing:
        raise ValueError(
            f"Child-disability classifier missing predictor column(s): {missing}."
        )
    age = pd.to_numeric(features["age"], errors="coerce").to_numpy(np.float64)
    female = pd.to_numeric(features["is_female"], errors="coerce").to_numpy(np.float64)
    income = pd.to_numeric(
        features[_HOUSEHOLD_INCOME_COLUMN], errors="coerce"
    ).to_numpy(np.float64)
    valid = (
        np.isfinite(age)
        & np.isfinite(female)
        & np.isin(female, [0.0, 1.0])
        & np.isfinite(income)
    )
    if not valid.all():
        rows = np.flatnonzero(~valid)[:5].tolist()
        raise ValueError(
            "Child-disability classifier predictors must be finite and sex "
            f"must be boolean; invalid row(s): {rows}."
        )
    return np.column_stack([age, female, np.log1p(np.maximum(income, 0.0))])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_weighted_logistic(donor: pd.DataFrame) -> _WeightedLogisticModel:
    required = {
        *SIPP_CHILD_DISABILITY_MODEL_PREDICTORS,
        _OUTPUT,
        _DONOR_WEIGHT_COLUMN,
    }
    missing = sorted(required - set(donor))
    if missing:
        raise ValueError(f"SIPP child-disability donor missing column(s): {missing}.")

    raw = _logistic_feature_array(donor)
    outcome = pd.to_numeric(donor[_OUTPUT], errors="coerce").to_numpy(np.float64)
    weights = pd.to_numeric(donor[_DONOR_WEIGHT_COLUMN], errors="coerce").to_numpy(
        np.float64
    )
    if not (np.isfinite(outcome) & np.isin(outcome, [0.0, 1.0])).all():
        raise ValueError("SIPP child-disability outcome must be boolean.")
    if np.unique(outcome).size != 2:
        raise ValueError("SIPP child-disability outcome must contain both classes.")
    if not (np.isfinite(weights) & (weights > 0.0)).all():
        raise ValueError(
            "SIPP child-disability model weights must be finite and positive."
        )

    normalized_weights = weights * (len(weights) / weights.sum())
    feature_mean = np.average(raw, axis=0, weights=normalized_weights)
    centered = raw - feature_mean
    variance = np.average(centered**2, axis=0, weights=normalized_weights)
    feature_scale = np.sqrt(np.maximum(variance, 1e-12))
    design = np.column_stack([np.ones(len(raw)), centered / feature_scale])

    weighted_rate = float(np.average(outcome, weights=normalized_weights))
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    coefficients[0] = np.log(weighted_rate / (1.0 - weighted_rate))
    ridge = np.diag([0.0, 1e-6, 1e-6, 1e-6])
    for _ in range(100):
        probability = _sigmoid(design @ coefficients)
        bernoulli_variance = np.maximum(probability * (1.0 - probability), 1e-8)
        working_weight = normalized_weights * bernoulli_variance
        working_response = (
            design @ coefficients + (outcome - probability) / bernoulli_variance
        )
        lhs = design.T @ (working_weight[:, None] * design) + ridge
        rhs = design.T @ (working_weight * working_response)
        try:
            updated = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            updated = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        if np.max(np.abs(updated - coefficients)) < 1e-10:
            coefficients = updated
            break
        coefficients = updated
    return _WeightedLogisticModel(coefficients, feature_mean, feature_scale)


def _decoded_strings(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: (
            value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        )
    )


def _stable_person_keys(person: pd.DataFrame) -> pd.Series:
    if _PERSON_SOURCE_ID_COLUMN in person:
        if person[_PERSON_SOURCE_ID_COLUMN].isna().any():
            raise ValueError("Child-disability person_source_id must be complete.")
        return _decoded_strings(person[_PERSON_SOURCE_ID_COLUMN])
    source_columns = ("source_year", "source_household_id", "source_person_id")
    if set(source_columns) <= set(person):
        if person.loc[:, source_columns].isna().any(axis=None):
            raise ValueError("Child-disability source identity must be complete.")
        return (
            person["source_year"].astype(str)
            + ":"
            + person["source_household_id"].astype(str)
            + ":"
            + person["source_person_id"].astype(str)
        )
    if "person_id" not in person or person["person_id"].isna().any():
        raise ValueError("Child-disability assignment requires stable person ids.")
    return person["person_id"].astype(str)


def _stable_uniform_draws(keys: pd.Series, *, seed: int, stream: str) -> np.ndarray:
    denominator = float(2**64)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"{int(seed)}:{US_CHILD_DISABILITY_STAGE_NAME}:{stream}:{key}".encode(),
                    digest_size=8,
                ).digest(),
                byteorder="big",
                signed=False,
            )
            / denominator
            for key in keys
        ],
        dtype=np.float64,
    )


def _recipient_features(
    frame: Frame,
) -> tuple[pd.DataFrame, pd.Series, pd.Index]:
    person = frame.table("person")
    required = {
        "person_id",
        "person_household_id",
        "age",
        "is_female",
        "employment_income_before_lsr",
        _OUTPUT,
    }
    missing = sorted(required - set(person))
    if missing:
        raise ValueError(
            f"US child-disability receiver missing person column(s): {missing}."
        )
    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(np.float64)
    female = pd.to_numeric(person["is_female"], errors="coerce").to_numpy(np.float64)
    earnings = pd.to_numeric(
        person["employment_income_before_lsr"], errors="coerce"
    ).to_numpy(np.float64)
    valid = (
        np.isfinite(age)
        & (age >= 0.0)
        & (age <= 120.0)
        & np.isfinite(female)
        & np.isin(female, [0.0, 1.0])
        & np.isfinite(earnings)
        & (earnings >= 0.0)
    )
    if not valid.all():
        rows = np.flatnonzero(~valid)[:5].tolist()
        raise ValueError(
            "US child-disability receiver age, sex, and employment income "
            f"must be valid; invalid row(s): {rows}."
        )
    if person["person_household_id"].isna().any():
        raise ValueError("US child-disability receiver household links are missing.")

    household = person["person_household_id"]
    features = pd.DataFrame(
        {
            "age": age,
            "is_female": female,
            _HOUSEHOLD_INCOME_COLUMN: (
                pd.Series(np.where(age >= 15.0, earnings, 0.0), index=person.index)
                .groupby(household, sort=False)
                .transform("sum")
            ),
        },
        index=person.index,
    )
    keys = _stable_person_keys(person)
    consistency = (
        pd.DataFrame({"key": keys, "age": age, "is_female": female}, index=person.index)
        .groupby("key", sort=False)[["age", "is_female"]]
        .nunique()
    )
    inconsistent = consistency.index[(consistency > 1).any(axis=1)].tolist()
    if inconsistent:
        raise ValueError(
            "US child-disability support clones disagree on age or sex for "
            f"source id(s): {inconsistent[:5]}."
        )

    order = pd.DataFrame({"key": keys}, index=person.index)
    if _PERSON_SUPPORT_CHANNEL_COLUMN in person:
        if person[_PERSON_SUPPORT_CHANNEL_COLUMN].isna().any():
            raise ValueError("Child-disability support channel must be complete.")
        channel = _decoded_strings(person[_PERSON_SUPPORT_CHANNEL_COLUMN])
        unexpected = sorted(
            set(channel.unique()) - {_ASEC_SUPPORT_CHANNEL, _PUF_SUPPORT_CHANNEL}
        )
        if unexpected:
            raise ValueError(
                "Child-disability receiver has unsupported support channel(s): "
                f"{unexpected}."
            )
        order["channel_priority"] = channel.map(
            {_ASEC_SUPPORT_CHANNEL: 0, _PUF_SUPPORT_CHANNEL: 1}
        )
    else:
        order["channel_priority"] = 0
    order["person_key"] = person["person_id"].astype(str)
    canonical_index = (
        order.sort_values(["key", "channel_priority", "person_key"], kind="mergesort")
        .drop_duplicates("key", keep="first")
        .index
    )
    return features, keys, canonical_index


def _calibrate_probabilities(
    base_probability: np.ndarray,
    *,
    current_true: np.ndarray,
    weights: np.ndarray,
    target_share: float,
) -> np.ndarray:
    """Shift probability logits so expected final OR-share hits a target."""

    probability = np.clip(
        np.asarray(base_probability, dtype=np.float64), 1e-9, 1 - 1e-9
    )
    current_true = np.asarray(current_true, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Child-disability calibration requires positive weight.")
    current_share = float(weights[current_true].sum()) / total
    if current_share >= target_share:
        return np.where(current_true, 1.0, 0.0)

    logit = np.log(probability / (1.0 - probability))

    def expected(shift: float) -> float:
        added = _sigmoid(logit + shift)
        final_probability = np.where(current_true, 1.0, added)
        return float(np.dot(weights, final_probability) / total)

    low, high = -30.0, 30.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if expected(midpoint) < target_share:
            low = midpoint
        else:
            high = midpoint
    shifted = _sigmoid(logit + (low + high) / 2.0)
    return np.where(current_true, 1.0, shifted)


def with_us_child_disability_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    sipp_donor: pd.DataFrame,
) -> Frame:
    """Augment under-15 ``is_disabled`` without writing the adult slice."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US child-disability inputs require the US schema.")
    del time_period  # The source and target vintages are fixed in the manifest.
    us_child_disability_stage_spec()
    person = frame.table("person")
    features, keys, canonical_index = _recipient_features(frame)
    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(np.float64)
    current_numeric = pd.to_numeric(person[_OUTPUT], errors="coerce").to_numpy(
        np.float64
    )
    under_15 = (age >= 0.0) & (age <= _MAX_MODEL_AGE)
    valid_child = np.isfinite(current_numeric) & np.isin(current_numeric, [0.0, 1.0])
    if not valid_child[under_15].all():
        raise ValueError(
            "US child-disability input is_disabled must be boolean for under-15 rows."
        )
    # Adult storage is outside this stage's ownership.  Keep an internal
    # child-only boolean vector and never normalize or assign the adult slice.
    current = np.zeros(len(person), dtype=bool)
    current[under_15] = current_numeric[under_15].astype(bool)
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    if not (np.isfinite(weights) & (weights >= 0.0)).all():
        raise ValueError("US child-disability person weights must be nonnegative.")

    model = _fit_weighted_logistic(sipp_donor)
    canonical = features.loc[canonical_index]
    canonical_probability = model.predict_probability(canonical)
    probability_by_key = pd.Series(
        canonical_probability,
        index=keys.loc[canonical_index].to_numpy(),
    )
    model_probability = keys.map(probability_by_key).to_numpy(np.float64)
    if not np.isfinite(model_probability).all():
        raise ValueError("Child-disability model did not cover every source person.")

    donor_outcome = pd.to_numeric(sipp_donor[_OUTPUT], errors="coerce").to_numpy(
        np.float64
    )
    donor_weight = pd.to_numeric(
        sipp_donor[_DONOR_WEIGHT_COLUMN], errors="coerce"
    ).to_numpy(np.float64)
    sipp_target_rate = float(np.average(donor_outcome, weights=donor_weight))

    result = current.copy()
    age_5_14 = (age >= _MIN_MODEL_AGE) & (age <= _MAX_MODEL_AGE)
    probability_5_14 = _calibrate_probabilities(
        model_probability[age_5_14],
        current_true=current[age_5_14],
        weights=weights[age_5_14],
        target_share=sipp_target_rate,
    )
    draws_5_14 = _stable_uniform_draws(keys.loc[age_5_14], seed=seed, stream="age_5_14")
    result[age_5_14] |= draws_5_14 < probability_5_14

    age_0_4 = (age >= 0.0) & (age <= _MAX_YOUNGEST_AGE)
    probability_0_4 = _calibrate_probabilities(
        np.full(int(age_0_4.sum()), US_CHILD_DISABILITY_AGE_0_4_TARGET_RATE),
        current_true=current[age_0_4],
        weights=weights[age_0_4],
        target_share=US_CHILD_DISABILITY_AGE_0_4_TARGET_RATE,
    )
    draws_0_4 = _stable_uniform_draws(keys.loc[age_0_4], seed=seed, stream="age_0_4")
    result[age_0_4] |= draws_0_4 < probability_0_4

    # The operation is additive: no pre-existing True is ever cleared.
    if np.any(current[under_15] & ~result[under_15]):
        raise AssertionError("Child-disability stage cleared an existing True value.")
    if np.array_equal(current[under_15], result[under_15]):
        return frame

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"].loc[under_15, _OUTPUT] = result[under_15]
    adult = age >= 15.0
    if not tables["person"].loc[adult].equals(person.loc[adult]):
        raise AssertionError("Child-disability stage changed a row aged 15 or over.")
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def us_child_disability_summary(frame: Frame) -> dict[str, object]:
    """Return weighted child-disability shares for the two owned age bands."""

    person = frame.table("person")
    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(np.float64)
    values = pd.to_numeric(person[_OUTPUT], errors="coerce").to_numpy(np.float64)
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    if not (np.isfinite(weights) & (weights >= 0.0)).all():
        raise ValueError("Child-disability gate weights must be nonnegative.")
    owned = (age >= 0.0) & (age <= _MAX_MODEL_AGE)
    finite_boolean = np.isfinite(values) & np.isin(values, [0.0, 1.0])
    disabled = finite_boolean & (values == 1.0)

    def band_summary(mask: np.ndarray) -> tuple[float, float, int]:
        band_weight = float(weights[mask].sum())
        disabled_weight = float(weights[mask & disabled].sum())
        share = disabled_weight / band_weight if band_weight > 0.0 else 0.0
        unique_count = int(pd.Series(values[mask & np.isfinite(values)]).nunique())
        return share, band_weight, unique_count

    age_0_4 = (age >= 0.0) & (age <= _MAX_YOUNGEST_AGE)
    age_5_14 = (age >= _MIN_MODEL_AGE) & (age <= _MAX_MODEL_AGE)
    share_0_4, weight_0_4, unique_0_4 = band_summary(age_0_4)
    share_5_14, weight_5_14, unique_5_14 = band_summary(age_5_14)
    return {
        "age_0_4_disabled_share": share_0_4,
        "age_0_4_weight": weight_0_4,
        "age_0_4_unique_count": unique_0_4,
        "age_0_4_target_rate": US_CHILD_DISABILITY_AGE_0_4_TARGET_RATE,
        "age_0_4_share_band": list(US_CHILD_DISABILITY_AGE_0_4_SHARE_BAND),
        "age_5_14_disabled_share": share_5_14,
        "age_5_14_weight": weight_5_14,
        "age_5_14_unique_count": unique_5_14,
        "age_5_14_sipp_observed_rate": _PINNED_ANY_ITEM_RATE,
        "age_5_14_share_band": list(US_CHILD_DISABILITY_AGE_5_14_SHARE_BAND),
        "missing_or_nonfinite_count": int(
            np.count_nonzero(owned & ~np.isfinite(values))
        ),
        "invalid_boolean_count": int(
            np.count_nonzero(owned & np.isfinite(values) & ~finite_boolean)
        ),
    }


def us_child_disability_signal_gate(
    frame: Frame,
    *,
    input_frame: Frame | None = None,
) -> GateResult:
    """Require plausible child shares and, when supplied, exact adult parity."""

    person = frame.table("person")
    if _OUTPUT not in person or "age" not in person:
        missing = [column for column in ("age", _OUTPUT) if column not in person]
        return GateResult(
            name="child_disability_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )
    summary = us_child_disability_summary(frame)
    failures: list[str] = []
    if summary["missing_or_nonfinite_count"]:
        failures.append(
            f"{_OUTPUT}: {summary['missing_or_nonfinite_count']} missing or "
            "nonfinite value(s)."
        )
    if summary["invalid_boolean_count"]:
        failures.append(
            f"{_OUTPUT}: {summary['invalid_boolean_count']} non-boolean value(s)."
        )
    for label, share_key, weight_key, unique_key, band in (
        (
            "0-4",
            "age_0_4_disabled_share",
            "age_0_4_weight",
            "age_0_4_unique_count",
            US_CHILD_DISABILITY_AGE_0_4_SHARE_BAND,
        ),
        (
            "5-14",
            "age_5_14_disabled_share",
            "age_5_14_weight",
            "age_5_14_unique_count",
            US_CHILD_DISABILITY_AGE_5_14_SHARE_BAND,
        ),
    ):
        if float(summary[weight_key]) <= 0.0:
            failures.append(f"age {label}: weighted domain is empty.")
            continue
        share = float(summary[share_key])
        low, high = band
        if not low <= share <= high:
            failures.append(
                f"age {label}: disabled share {share:.4f} outside plausibility "
                f"band [{low}, {high}]."
            )
        if int(summary[unique_key]) < 2:
            failures.append(f"age {label}: disability surface is constant.")

    adult_unchanged: bool | None = None
    if input_frame is not None:
        input_person = input_frame.table("person")
        same_identity = (
            "person_id" in input_person
            and "person_id" in person
            and person["person_id"].equals(input_person["person_id"])
        )
        if same_identity and "age" in input_person:
            input_age = pd.to_numeric(input_person["age"], errors="coerce")
            adult = input_age >= 15
            adult_unchanged = person.loc[adult].equals(input_person.loc[adult])
        else:
            adult_unchanged = False
        if not adult_unchanged:
            failures.append("age 15+: output rows differ from the stage input.")
    details = dict(summary)
    if input_frame is not None:
        input_summary = us_child_disability_summary(input_frame)
        details["weighted_child_is_disabled_share_change"] = {
            "age_0_4": {
                "before": input_summary["age_0_4_disabled_share"],
                "after": summary["age_0_4_disabled_share"],
                "absolute_change": (
                    float(summary["age_0_4_disabled_share"])
                    - float(input_summary["age_0_4_disabled_share"])
                ),
            },
            "age_5_14": {
                "before": input_summary["age_5_14_disabled_share"],
                "after": summary["age_5_14_disabled_share"],
                "absolute_change": (
                    float(summary["age_5_14_disabled_share"])
                    - float(input_summary["age_5_14_disabled_share"])
                ),
            },
        }
    details["age_15_plus_unchanged"] = adult_unchanged
    return GateResult(
        name="child_disability_signal",
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )
