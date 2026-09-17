"""SIPP-imputed latent SSI disability-criteria input.

The retired enhanced-CPS pipeline trained a person-grain boolean QRF on the
December SIPP.  Its label was deliberately narrower than a general disability
flag: an under-65 SIPP SSI recipient whose *reported* benefit reason was
disabled/blind was positive.  Observed nonrecipients were usable negatives only
when they passed a source-time approximation of the SSI resource, countable-
income, and substantial-gainful-activity screens.  The exact nineteen archived
predictors are retained below.

The source transform also retains two distinctions that are easy to lose:

* the receipt and benefit-reason allocation flags decide whether a label is
  observed; and
* the final QRF result is accepted only for a person carrying at least one of
  the six measured disability difficulties, Social Security disability, or
  other disability income.

The extended-CPS pipeline predicted its ASEC and PUF-support people separately,
because the latter carried separately imputed income and asset predictors.  We
do the same.  Direct under-65 ASEC ``SSI_VAL`` reporters are then preserved as
positive anchors; that anchor is not copied onto the PUF channel.  A stacked
ACS row instead contributes its harmonized native ``ssi_reported`` value to the
same ``> 0`` predicate.  The row-wise coalesce is source-blind and preserves
the real below-age-15 ACS amount-universe blank.  An arbitrary pre-existing
criterion column is never trusted: every run recomputes the full source-backed
surface and uses an equality check only for idempotent return.

The full 2023 SIPP public-use file is the same immutable 3.73 GB artifact
already pinned by the vehicle and voluntary-filing stages.  It contains 39,513
December person rows.  Under the archived 2024 financial screen it yields
9,346 training candidates (577 positive and 8,769 negative).
"""

from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_manifest import SourceStageSpec, load_source_manifest
from microcosm.build.us_runtime.support_provenance import (
    has_assembled_support_metadata,
    has_support_role_metadata,
    support_role_series,
)
from microcosm.build.us_runtime.voluntary_filing import (
    SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION,
    SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256,
    SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES,
    SIPP_2023_VOLUNTARY_FILING_DONOR_URL,
    fetch_sipp_2023_voluntary_filing_donor,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "SIPP_2023_SSI_DISABILITY_DONOR_REVISION",
    "SIPP_2023_SSI_DISABILITY_DONOR_SHA256",
    "SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES",
    "SIPP_2023_SSI_DISABILITY_DONOR_URL",
    "SIPP_SSI_DISABILITY_DIFFICULTY_PREDICTORS",
    "SIPP_SSI_DISABILITY_FIT_PARAMETERS",
    "SIPP_SSI_DISABILITY_MODEL_PREDICTORS",
    "SIPP_SSI_DISABILITY_READ_PARAMETERS",
    "SIPP_SSI_DISABILITY_SOURCE_COLUMNS",
    "SSI_DISABILITY_ARCHIVED_CPS_URL",
    "SSI_DISABILITY_ARCHIVED_EXTENDED_CPS_URL",
    "SSI_DISABILITY_ARCHIVED_SIPP_URL",
    "SSI_DISABILITY_ARCHIVED_SOURCE_IMPUTE_URL",
    "SSI_DISABILITY_SIPP_DICTIONARY_URL",
    "US_SSI_DISABILITY_CRITERIA_NONCONSTANT_PERSON_COLUMNS",
    "US_SSI_DISABILITY_CRITERIA_OUTPUT_COLUMNS",
    "US_SSI_DISABILITY_CRITERIA_STAGE_NAME",
    "fetch_sipp_2023_ssi_disability_donor",
    "impute_us_ssi_disability_criteria",
    "load_sipp_2023_ssi_disability_donor",
    "us_ssi_disability_criteria_signal_gate",
    "us_ssi_disability_criteria_stage_spec",
    "us_ssi_disability_criteria_summary",
    "with_us_ssi_disability_criteria",
]

QRF: Any | None = None

_ARCHIVED_COMMIT = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
_RETIRED_REPOSITORY = "policyengine-" + "us-data"
_RETIRED_PACKAGE = "policyengine_" + "us_data"
_ARCHIVED_ROOT = (
    f"https://github.com/PolicyEngine/{_RETIRED_REPOSITORY}/blob/"
    f"{_ARCHIVED_COMMIT}/{_RETIRED_PACKAGE}/"
)
SSI_DISABILITY_ARCHIVED_SIPP_URL = _ARCHIVED_ROOT + "datasets/sipp/sipp.py#L63-L105"
SSI_DISABILITY_ARCHIVED_CPS_URL = _ARCHIVED_ROOT + "datasets/cps/cps.py#L2853-L2886"
SSI_DISABILITY_ARCHIVED_SOURCE_IMPUTE_URL = (
    _ARCHIVED_ROOT + "calibration/source_impute.py#L869-L990"
)
SSI_DISABILITY_ARCHIVED_EXTENDED_CPS_URL = (
    _ARCHIVED_ROOT + "datasets/cps/extended_cps.py#L392-L424"
)
SSI_DISABILITY_SIPP_DICTIONARY_URL = (
    "https://www2.census.gov/programs-surveys/sipp/tech-documentation/"
    "data-dictionaries/2023/2023_SIPP_Data_Dictionary.pdf"
)

# Reuse the exact full-file coordinate and fetch/cache implementation already
# owned by the voluntary-filing stage.  Aliases keep this family's provenance
# legible to manifests and release telemetry.
SIPP_2023_SSI_DISABILITY_DONOR_REVISION = SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION
SIPP_2023_SSI_DISABILITY_DONOR_SHA256 = SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256
SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES = SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES
SIPP_2023_SSI_DISABILITY_DONOR_URL = SIPP_2023_VOLUNTARY_FILING_DONOR_URL

US_SSI_DISABILITY_CRITERIA_STAGE_NAME = "ssi_disability_criteria"
US_SSI_DISABILITY_CRITERIA_OUTPUT_COLUMNS: tuple[str, ...] = (
    "meets_ssi_disability_criteria",
)
US_SSI_DISABILITY_CRITERIA_NONCONSTANT_PERSON_COLUMNS = (
    US_SSI_DISABILITY_CRITERIA_OUTPUT_COLUMNS
)

SIPP_SSI_DISABILITY_DIFFICULTY_PREDICTORS: tuple[str, ...] = (
    "difficulty_dressing_or_bathing",
    "difficulty_hearing",
    "difficulty_seeing",
    "difficulty_doing_errands",
    "difficulty_walking_or_climbing_stairs",
    "difficulty_remembering_or_making_decisions",
)
_SIPP_DIFFICULTY_SOURCE_COLUMNS: dict[str, str] = {
    "difficulty_dressing_or_bathing": "ESELFCARE",
    "difficulty_hearing": "EHEARING",
    "difficulty_seeing": "ESEEING",
    "difficulty_doing_errands": "EERRANDS",
    "difficulty_walking_or_climbing_stairs": "EAMBULAT",
    "difficulty_remembering_or_making_decisions": "ECOGNIT",
}
_ASEC_DIFFICULTY_SOURCE_COLUMNS: dict[str, str] = {
    "difficulty_dressing_or_bathing": "PEDISDRS",
    "difficulty_hearing": "PEDISEAR",
    "difficulty_seeing": "PEDISEYE",
    "difficulty_doing_errands": "PEDISOUT",
    "difficulty_walking_or_climbing_stairs": "PEDISPHY",
    "difficulty_remembering_or_making_decisions": "PEDISREM",
}
SIPP_SSI_DISABILITY_MODEL_PREDICTORS: tuple[str, ...] = (
    "age",
    "is_female",
    "is_married",
    "employment_income",
    "interest_income",
    "dividend_income",
    "rental_income",
    "bank_account_assets",
    "stock_assets",
    "bond_assets",
    "count_under_18",
    *SIPP_SSI_DISABILITY_DIFFICULTY_PREDICTORS,
    "social_security_disability",
    "has_disability_income",
)

_SIPP_JOB_EARNINGS_COLUMNS = tuple(f"TJB{i}_MSUM" for i in range(1, 8))
_SIPP_DISABILITY_INCOME_AMOUNT_COLUMNS = tuple(f"TDIS{i}AMT" for i in range(1, 11))
_SIPP_ASSET_ALLOCATION_COLUMNS: tuple[str, ...] = (
    "AJSSAVVAL",
    "AJOSAVVAL",
    "AOSAVVAL",
    "AJSMMVAL",
    "AJOMMVAL",
    "AOMMVAL",
    "AJSCDVAL",
    "AJOCDVAL",
    "AOCDVAL",
    "AJSCHKVAL",
    "AJOCHKVAL",
    "AOCHKVAL",
    "AJSSTVAL",
    "AJOSTVAL",
    "AOSTVAL",
    "AJSMFVAL",
    "AJOMFVAL",
    "AOMFVAL",
    "AJSGOVSVAL",
    "AJOGOVSVAL",
    "AOGOVSVAL",
    "AJSMCBDVAL",
    "AJOMCBDVAL",
    "AOMCBDVAL",
)
_SIPP_BASE_ASSET_COLUMNS: tuple[str, ...] = (
    "SSUID",
    "PNUM",
    "MONTHCODE",
    "SPANEL",
    "SWAVE",
    "WPFINWGT",
    "TAGE",
    "ESEX",
    "EMS",
    "TSSSAMT",
    "TRETINCAMT",
    "TVAL_BANK",
    "TVAL_STMF",
    "TVAL_BOND",
    "TINC_BANK",
    "TINC_STMF",
    "TINC_BOND",
    "TINC_RENT",
    *_SIPP_JOB_EARNINGS_COLUMNS,
    *_SIPP_ASSET_ALLOCATION_COLUMNS,
)
SIPP_SSI_DISABILITY_SOURCE_COLUMNS: tuple[str, ...] = tuple(
    sorted(
        {
            *_SIPP_BASE_ASSET_COLUMNS,
            "TPTOTINC",
            "RSSI_YRYN",
            "EDISABL",
            "EHLTHCOND",
            "RDIS",
            "RDIS_ALT",
            "EDISANY",
            "ENJ_NOWRK3",
            "ESSRSN2YN",
            "ESSI_BRSN",
            *_SIPP_DIFFICULTY_SOURCE_COLUMNS.values(),
            *_SIPP_DISABILITY_INCOME_AMOUNT_COLUMNS,
            "ASSI_YRYN",
            "ASSI_BRSN",
        }
    )
)

_OUTPUT = US_SSI_DISABILITY_CRITERIA_OUTPUT_COLUMNS[0]
_DONOR_WEIGHT_COLUMN = "household_weight"
_TRAINING_CANDIDATE_COLUMN = "ssi_disability_training_candidate"
_DEFAULT_N_ESTIMATORS = 100
_MAX_TRAIN_SAMPLES = 20_000
_TRAINING_SAMPLE_SEED_NAME = "sipp_ssi_disability_model_training_sample"
_TRAINING_SAMPLE_SEED = 8_386_123_572_872_638_692
_ARCHIVED_MODEL_SEED = 42
_PERSON_SOURCE_ID_COLUMN = "person_source_id"
_BASE_ASEC_SUPPORT_CHANNEL = "asec"
_PUF_TAX_DETAIL_SUPPORT_CHANNEL = "puf_tax_detail"
_RECIPIENT_INTEREST_COMPONENT_COLUMNS = (
    "taxable_interest_income",
    "tax_exempt_interest_income",
)
_RECIPIENT_DIVIDEND_COMPONENT_COLUMNS = (
    "qualified_dividend_income",
    "non_qualified_dividend_income",
)
_TRUE_SHARE_BAND = (0.001, 0.25)
_OBSERVED_ALLOCATION_VALUES = frozenset((0, 1, 9))

_PINNED_DECEMBER_ROWS = 39_513
_PINNED_TRAINING_ROWS = 9_346
_PINNED_POSITIVE_ROWS = 577
_PINNED_NEGATIVE_ROWS = 8_769
_PINNED_WEIGHT_SUM = 88_690_359.47893329
_PINNED_POSITIVE_WEIGHT_SUM = 4_937_167.914119501
_PINNED_WEIGHTED_TRUE_SHARE = 0.05566746987074994
_PINNED_RESAMPLE_ROWS = 9_346
_PINNED_RESAMPLE_UNIQUE_SOURCE_ROWS = 5_314
_PINNED_RESAMPLE_POSITIVE_ROWS = 524
_PINNED_RESAMPLE_TRUE_SHARE = 0.05606676653113631
_PINNED_FLOAT_ATOL = 1e-9

SIPP_SSI_DISABILITY_READ_PARAMETERS: dict[str, object] = {
    "table": "sipp_person",
    "delimiter": "|",
    "month_column": "MONTHCODE",
    "month": 12,
    "source_columns": list(SIPP_SSI_DISABILITY_SOURCE_COLUMNS),
}
SIPP_SSI_DISABILITY_FIT_PARAMETERS: dict[str, object] = {
    "predictors": list(SIPP_SSI_DISABILITY_MODEL_PREDICTORS),
    "target": _OUTPUT,
    "weight": _DONOR_WEIGHT_COLUMN,
    "training_candidate": _TRAINING_CANDIDATE_COLUMN,
    "label_source_columns": ["RSSI_YRYN", "ESSI_BRSN"],
    "label_allocation_columns": ["ASSI_YRYN", "ASSI_BRSN"],
    "max_train_samples": _MAX_TRAIN_SAMPLES,
    "sample_with_replacement": True,
    "training_sample_seed_name": _TRAINING_SAMPLE_SEED_NAME,
    "training_sample_seed": _TRAINING_SAMPLE_SEED,
    "n_estimators": _DEFAULT_N_ESTIMATORS,
    "model_seed": _ARCHIVED_MODEL_SEED,
    "seed_from_build_config": False,
    "postprediction_signal_predictors": [
        *SIPP_SSI_DISABILITY_DIFFICULTY_PREDICTORS,
        "social_security_disability",
        "has_disability_income",
    ],
    "preserve_under_65_asec_ssi_reporters": True,
}


def us_ssi_disability_criteria_stage_spec() -> SourceStageSpec:
    """Load and strictly validate the packaged source-stage declaration."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_SSI_DISABILITY_CRITERIA_STAGE_NAME not in stage_map:
        raise ValueError(
            "US source manifest declares no "
            f"{US_SSI_DISABILITY_CRITERIA_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_SSI_DISABILITY_CRITERIA_STAGE_NAME]
    if spec.grain != "person":
        raise ValueError("US SSI disability-criteria stage must have person grain.")
    if tuple(spec.outputs) != US_SSI_DISABILITY_CRITERIA_OUTPUT_COLUMNS:
        raise ValueError(
            "US SSI disability-criteria manifest outputs drifted from the "
            "runtime-owned family."
        )
    if [operation.kind for operation in spec.operations] != [
        "read_table",
        "fit_weighted_qrf",
    ]:
        raise ValueError(
            "US SSI disability-criteria stage must contain read_table then "
            "fit_weighted_qrf."
        )
    if dict(spec.operations[0].parameters) != SIPP_SSI_DISABILITY_READ_PARAMETERS:
        raise ValueError(
            "US SSI disability-criteria read_table contract drifted from the "
            "pinned SIPP transform."
        )
    if dict(spec.operations[1].parameters) != SIPP_SSI_DISABILITY_FIT_PARAMETERS:
        raise ValueError(
            "US SSI disability-criteria QRF contract drifted from the archived method."
        )
    pinned = [
        artifact
        for artifact in spec.artifacts
        if artifact.get("sha256") == SIPP_2023_SSI_DISABILITY_DONOR_SHA256
        and artifact.get("size_bytes") == SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES
    ]
    if not pinned:
        raise ValueError(
            "US SSI disability-criteria stage does not pin the full SIPP "
            "SHA-256 and byte length."
        )
    return spec


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_sipp_2023_ssi_disability_donor(
    cache_dir: str | Path | None = None,
    *,
    expected_sha256: str | None = SIPP_2023_SSI_DISABILITY_DONOR_SHA256,
    expected_size_bytes: int | None = SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Fetch the shared pinned full SIPP artifact through its canonical cache."""

    return fetch_sipp_2023_voluntary_filing_donor(
        cache_dir,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        chunk_size=chunk_size,
    )


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _yes(frame: pd.DataFrame, column: str) -> pd.Series:
    return _numeric(frame[column]).fillna(0.0).eq(1.0)


def _monthly_earned_income(frame: pd.DataFrame) -> pd.Series:
    return frame.loc[:, list(_SIPP_JOB_EARNINGS_COLUMNS)].fillna(0.0).sum(axis=1)


def _ssi_policy_screen_values(time_period: int) -> dict[str, float]:
    """Return the exact 2024-style values read by the archived source screen."""

    try:
        from policyengine_us import CountryTaxBenefitSystem

        parameters = CountryTaxBenefitSystem().parameters(f"{time_period}-01-01")
        ssi = parameters.gov.ssa.ssi
        exclusions = ssi.income.exclusions
        return {
            "individual_resource_limit": float(
                ssi.eligibility.resources.limit.individual
            ),
            "couple_resource_limit": float(ssi.eligibility.resources.limit.couple),
            "individual_fbr": float(ssi.amount.individual),
            "couple_fbr": float(ssi.amount.couple),
            "general_exclusion": float(exclusions.general),
            "earned_exclusion": float(exclusions.earned),
            "earned_share_excluded": float(exclusions.earned_share),
            "non_blind_sga": float(parameters.gov.ssa.sga.non_blind),
        }
    except Exception:
        # These are the archived function's explicit 2024 fallback constants.
        return {
            "individual_resource_limit": 2_000.0,
            "couple_resource_limit": 3_000.0,
            "individual_fbr": 943.0,
            "couple_fbr": 1_415.0,
            "general_exclusion": 20.0,
            "earned_exclusion": 65.0,
            "earned_share_excluded": 0.5,
            "non_blind_sga": 1_550.0,
        }


def _financial_candidate_mask(
    frame: pd.DataFrame,
    *,
    time_period: int,
) -> pd.Series:
    values = _ssi_policy_screen_values(time_period)
    married = frame["is_married"].astype(bool)
    resource_limit = np.where(
        married,
        values["couple_resource_limit"],
        values["individual_resource_limit"],
    )
    income_limit = np.where(
        married,
        values["couple_fbr"],
        values["individual_fbr"],
    )
    liquid_resources = (
        frame["bank_account_assets"].fillna(0.0)
        + frame["stock_assets"].fillna(0.0)
        + frame["bond_assets"].fillna(0.0)
    )
    earned = _monthly_earned_income(frame)
    unearned = (frame["TPTOTINC"].fillna(0.0) - earned).clip(lower=0.0)
    applied_general = np.minimum(values["general_exclusion"], unearned)
    countable_unearned = unearned - applied_general
    leftover_general = values["general_exclusion"] - applied_general
    earned_after_flat = (earned - values["earned_exclusion"] - leftover_general).clip(
        lower=0.0
    )
    countable_earned = earned_after_flat * (1.0 - values["earned_share_excluded"])
    countable_income = countable_unearned + countable_earned
    is_blind = frame["difficulty_seeing"].fillna(False).astype(bool)
    passes_sga = is_blind | earned.le(values["non_blind_sga"])
    return (
        liquid_resources.le(resource_limit)
        & countable_income.le(income_limit)
        & passes_sga
    )


def _observed_label_mask(
    frame: pd.DataFrame,
    received_ssi: pd.Series,
) -> pd.Series:
    receipt_flag = _numeric(frame["ASSI_YRYN"]).fillna(0.0)
    reason_flag = _numeric(frame["ASSI_BRSN"]).fillna(0.0)
    receipt_observed = (
        frame["RSSI_YRYN"].notna()
        & receipt_flag.isin(_OBSERVED_ALLOCATION_VALUES)
        & _numeric(frame["RSSI_YRYN"]).isin([1.0, 2.0])
    )
    reason_observed = (
        frame["ESSI_BRSN"].notna()
        & reason_flag.isin(_OBSERVED_ALLOCATION_VALUES)
        & _numeric(frame["ESSI_BRSN"]).isin([1.0, 2.0])
    )
    return receipt_observed & (~received_ssi | reason_observed)


def _assert_pinned_source_audit(audit: dict[str, object]) -> None:
    expected_counts = {
        "december_rows": _PINNED_DECEMBER_ROWS,
        "training_rows": _PINNED_TRAINING_ROWS,
        "positive_rows": _PINNED_POSITIVE_ROWS,
        "negative_rows": _PINNED_NEGATIVE_ROWS,
        "resample_rows": _PINNED_RESAMPLE_ROWS,
        "resample_unique_source_rows": _PINNED_RESAMPLE_UNIQUE_SOURCE_ROWS,
        "resample_positive_rows": _PINNED_RESAMPLE_POSITIVE_ROWS,
    }
    mismatches = {
        key: (expected, audit.get(key))
        for key, expected in expected_counts.items()
        if audit.get(key) != expected
    }
    expected_floats = {
        "weight_sum": _PINNED_WEIGHT_SUM,
        "positive_weight_sum": _PINNED_POSITIVE_WEIGHT_SUM,
        "weighted_true_share": _PINNED_WEIGHTED_TRUE_SHARE,
        "resample_true_share": _PINNED_RESAMPLE_TRUE_SHARE,
    }
    for key, expected in expected_floats.items():
        actual = float(audit[key])
        if not np.isclose(actual, expected, rtol=0.0, atol=_PINNED_FLOAT_ATOL):
            mismatches[key] = (expected, actual)
    if mismatches:
        raise ValueError(
            "Pinned SIPP SSI disability source audit drifted from the reviewed "
            f"2024 transform: {mismatches}."
        )


def load_sipp_2023_ssi_disability_donor(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES,
    chunksize: int = 100_000,
    time_period: int = 2024,
) -> pd.DataFrame:
    """Build the exact observed/candidate SIPP SSI disability training frame."""

    source_path = Path(path)
    actual_size = source_path.stat().st_size
    if expected_size_bytes is not None and actual_size != expected_size_bytes:
        raise ValueError(
            "SIPP 2023 SSI disability donor failed byte-length verification: "
            f"expected {expected_size_bytes}, got {actual_size}."
        )
    if expected_sha256 is not None:
        actual_sha256 = _sha256_file(source_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "SIPP 2023 SSI disability donor failed sha-256 verification: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )
    if chunksize < 1:
        raise ValueError("chunksize must be a positive integer")

    header = pd.read_csv(source_path, delimiter="|", nrows=0)
    missing = sorted(set(SIPP_SSI_DISABILITY_SOURCE_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"SIPP SSI disability donor missing column(s): {missing}.")

    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        source_path,
        delimiter="|",
        usecols=list(SIPP_SSI_DISABILITY_SOURCE_COLUMNS),
        chunksize=int(chunksize),
        low_memory=False,
    ):
        month = _numeric(chunk["MONTHCODE"])
        december = chunk.loc[month.eq(12.0)].copy()
        if not december.empty:
            parts.append(december)
    if not parts:
        raise ValueError("SIPP SSI disability donor has no December rows.")
    frame = pd.concat(parts, ignore_index=True)

    for column in SIPP_SSI_DISABILITY_SOURCE_COLUMNS:
        if column == "SSUID":
            continue
        original = frame[column]
        converted = _numeric(original)
        invalid = original.notna() & converted.isna()
        if invalid.any():
            rows = np.flatnonzero(invalid.to_numpy())[:5].tolist()
            raise ValueError(
                f"SIPP SSI disability source {column!r} contains nonnumeric "
                f"value(s) at row(s) {rows}."
            )
        frame[column] = converted
    if frame[["SSUID", "PNUM"]].isna().any(axis=None):
        raise ValueError("SIPP SSI disability December source has missing IDs.")

    frame["bank_account_assets"] = frame["TVAL_BANK"].fillna(0.0)
    frame["stock_assets"] = frame["TVAL_STMF"].fillna(0.0)
    frame["bond_assets"] = frame["TVAL_BOND"].fillna(0.0)
    frame["age"] = frame["TAGE"]
    frame["is_female"] = frame["ESEX"].eq(2.0)
    frame["is_married"] = frame["EMS"].eq(1.0)
    earned = _monthly_earned_income(frame)
    frame["employment_income"] = earned * 12.0
    frame["interest_income"] = (
        frame["TINC_BANK"].fillna(0.0) + frame["TINC_BOND"].fillna(0.0)
    ) * 12.0
    frame["dividend_income"] = frame["TINC_STMF"].fillna(0.0) * 12.0
    frame["rental_income"] = frame["TINC_RENT"].fillna(0.0) * 12.0
    frame[_DONOR_WEIGHT_COLUMN] = frame["WPFINWGT"].fillna(0.0)
    is_under_18 = frame["TAGE"].lt(18.0)
    frame["count_under_18"] = is_under_18.groupby(frame["SSUID"]).transform("sum")
    for predictor, source_column in _SIPP_DIFFICULTY_SOURCE_COLUMNS.items():
        frame[predictor] = _yes(frame, source_column)

    disability_income_amount = pd.Series(0.0, index=frame.index)
    for column in _SIPP_DISABILITY_INCOME_AMOUNT_COLUMNS:
        disability_income_amount += frame[column].fillna(0.0)
    frame["social_security_disability"] = np.where(
        _yes(frame, "ESSRSN2YN"),
        frame["TSSSAMT"].fillna(0.0) * 12.0,
        0.0,
    )
    frame["has_disability_income"] = _yes(
        frame, "EDISANY"
    ) | disability_income_amount.gt(0.0)

    received_ssi = _yes(frame, "RSSI_YRYN")
    under_65 = frame["age"].lt(65.0)
    reason = frame["ESSI_BRSN"].fillna(-9.0).astype(float)
    disabled_or_blind_reason = reason.eq(1.0)
    aged_reason = reason.eq(2.0)
    frame[_OUTPUT] = received_ssi & under_65 & (disabled_or_blind_reason | ~aged_reason)
    financial_candidate = _financial_candidate_mask(
        frame,
        time_period=int(time_period),
    )
    frame[_TRAINING_CANDIDATE_COLUMN] = (financial_candidate & under_65) | frame[
        _OUTPUT
    ]
    observed = _observed_label_mask(frame, received_ssi)

    columns = [
        *SIPP_SSI_DISABILITY_MODEL_PREDICTORS,
        _OUTPUT,
        _TRAINING_CANDIDATE_COLUMN,
        _DONOR_WEIGHT_COLUMN,
    ]
    observed_frame = frame.loc[observed, columns].dropna().copy()
    donor = observed_frame.loc[observed_frame[_TRAINING_CANDIDATE_COLUMN]].copy()
    donor = donor.drop(columns=[_TRAINING_CANDIDATE_COLUMN]).reset_index(drop=True)

    numeric_columns = [*SIPP_SSI_DISABILITY_MODEL_PREDICTORS, _DONOR_WEIGHT_COLUMN]
    finite = np.isfinite(donor.loc[:, numeric_columns].to_numpy(dtype=np.float64)).all(
        axis=1
    )
    if not finite.all():
        rows = np.flatnonzero(~finite)[:5].tolist()
        raise ValueError(
            "SIPP SSI disability donor contains nonfinite training values at "
            f"row(s) {rows}."
        )
    weights = donor[_DONOR_WEIGHT_COLUMN].to_numpy(dtype=np.float64)
    if not (weights > 0.0).all():
        rows = np.flatnonzero(weights <= 0.0)[:5].tolist()
        raise ValueError(
            "SIPP SSI disability donor weights must be positive; invalid "
            f"row(s) {rows}."
        )
    target = donor[_OUTPUT].astype(bool).to_numpy()
    if np.unique(target).size != 2:
        raise ValueError("SIPP SSI disability donor target must contain both classes.")

    probability = weights / weights.sum()
    selected = np.random.default_rng(_TRAINING_SAMPLE_SEED).choice(
        len(donor),
        size=min(_MAX_TRAIN_SAMPLES, len(donor)),
        replace=True,
        p=probability,
    )
    resample_target = target[selected]
    audit: dict[str, object] = {
        "december_rows": int(len(frame)),
        "observed_label_rows_before_predictor_dropna": int(observed.sum()),
        "observed_complete_rows": int(len(observed_frame)),
        "training_rows": int(len(donor)),
        "positive_rows": int(target.sum()),
        "negative_rows": int((~target).sum()),
        "weight_sum": float(weights.sum()),
        "positive_weight_sum": float(weights[target].sum()),
        "weighted_true_share": float(weights[target].sum() / weights.sum()),
        "resample_rows": int(len(selected)),
        "resample_unique_source_rows": int(np.unique(selected).size),
        "resample_positive_rows": int(resample_target.sum()),
        "resample_true_share": float(resample_target.mean()),
        "time_period": int(time_period),
    }
    pinned_transform = (
        actual_size == SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES
        and int(time_period) == 2024
    )
    audit["pinned_transform"] = pinned_transform
    if pinned_transform:
        _assert_pinned_source_audit(audit)
    donor.attrs["source_audit"] = audit
    return donor


def _decoded_strings(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: (
            value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        )
    )


def _strict_person_numeric(
    person: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    label: str,
) -> np.ndarray:
    for column in columns:
        if column in person:
            values = pd.to_numeric(person[column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"US SSI disability receiver {label} source {column!r} "
                    "contains nonfinite values."
                )
            return values
    raise ValueError(
        f"US SSI disability receiver requires one of {list(columns)} for {label}."
    )


def _strict_person_aggregate(
    person: pd.DataFrame,
    *,
    aggregate_column: str,
    component_columns: tuple[str, ...],
) -> np.ndarray:
    if aggregate_column in person:
        return _strict_person_numeric(
            person,
            (aggregate_column,),
            label=aggregate_column,
        )
    missing = [column for column in component_columns if column not in person]
    if missing:
        raise ValueError(
            f"US SSI disability receiver requires person.{aggregate_column} or "
            f"all measured component columns {list(component_columns)}; missing "
            f"{missing}."
        )
    return np.sum(
        np.column_stack(
            [
                _strict_person_numeric(person, (column,), label=column)
                for column in component_columns
            ]
        ),
        axis=1,
    )


def _reported_ssi_anchor(person: pd.DataFrame, *, age: np.ndarray) -> np.ndarray:
    """Coalesce CPS and harmonized ACS reporter amounts for the ``> 0`` test.

    ``SSI_VAL`` is the measured CPS ASEC amount. ``ssi_reported`` is the
    adjusted native ACS SSIP amount produced by ``map_acs_native_inputs``.
    Stacked rows carry exactly one of the two. ACS SSIP is out of universe
    below age 15, so that genuine blank is interpreted only at predicate time
    and is never rewritten into a fake measured zero.
    """

    available = [column for column in ("SSI_VAL", "ssi_reported") if column in person]
    if not available:
        raise ValueError(
            "US SSI disability receiver requires measured SSI_VAL or harmonized "
            "ssi_reported for the under-65 reporter anchor."
        )

    numeric: dict[str, pd.Series] = {}
    for column in available:
        raw = person[column]
        values = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & values.isna()
        finite = np.isfinite(values.fillna(0.0).to_numpy(dtype=np.float64))
        if invalid.any() or not finite.all():
            raise ValueError(
                "US SSI disability receiver reported SSI source "
                f"{column!r} contains nonnumeric or nonfinite values."
            )
        numeric[column] = values

    combined = pd.Series(np.nan, index=person.index, dtype=np.float64)
    if "SSI_VAL" in numeric:
        combined = numeric["SSI_VAL"].copy()
    if "ssi_reported" in numeric:
        if "SSI_VAL" in numeric:
            both = numeric["SSI_VAL"].notna() & numeric["ssi_reported"].notna()
            positivity_conflict = both & numeric["SSI_VAL"].gt(0).ne(
                numeric["ssi_reported"].gt(0)
            )
            if positivity_conflict.any():
                raise ValueError(
                    "US SSI disability receiver SSI_VAL and ssi_reported "
                    "disagree on reporter status."
                )
        combined = combined.combine_first(numeric["ssi_reported"])

    age_values = np.asarray(age, dtype=np.float64)
    if len(age_values) != len(combined) or not np.isfinite(age_values).all():
        raise ValueError("US SSI disability receiver age must be finite.")
    invalid_blank = combined.isna().to_numpy() & (age_values >= 15.0)
    if invalid_blank.any():
        raise ValueError(
            "US SSI disability receiver reported SSI amount may be blank only "
            "below the ACS age-15 universe."
        )
    return combined.fillna(0.0).to_numpy(dtype=np.float64)


def _person_ssi_disability_predictors(frame: Frame) -> pd.DataFrame:
    """Build the exact nineteen predictors on every recipient support row."""

    person = frame.table("person")
    required = {
        "person_household_id",
        "bank_account_assets",
        "stock_assets",
        "bond_assets",
        "rental_income",
        "social_security_disability",
        "disability_benefits",
        *_ASEC_DIFFICULTY_SOURCE_COLUMNS.values(),
    }
    missing = sorted(required - set(person.columns))
    if missing:
        raise ValueError(
            f"US SSI disability receiver missing measured person column(s): {missing}."
        )

    receiver = pd.DataFrame(index=person.index)
    receiver["age"] = _strict_person_numeric(person, ("age", "A_AGE"), label="age")
    if "is_female" in person:
        female = _strict_person_numeric(person, ("is_female",), label="sex")
        if not np.isin(female, [0.0, 1.0]).all():
            raise ValueError("US SSI disability receiver is_female must be boolean.")
        receiver["is_female"] = female
    elif "is_male" in person:
        male = _strict_person_numeric(person, ("is_male",), label="sex")
        if not np.isin(male, [0.0, 1.0]).all():
            raise ValueError("US SSI disability receiver is_male must be boolean.")
        receiver["is_female"] = 1.0 - male
    elif "A_SEX" in person:
        sex = _strict_person_numeric(person, ("A_SEX",), label="sex")
        if not np.isin(sex, [1.0, 2.0]).all():
            raise ValueError("US SSI disability receiver A_SEX must be coded 1 or 2.")
        receiver["is_female"] = (sex == 2.0).astype(np.float64)
    else:
        raise ValueError(
            "US SSI disability receiver requires is_female, is_male, or A_SEX."
        )

    if "is_married" in person:
        married = _strict_person_numeric(person, ("is_married",), label="marriage")
        if not np.isin(married, [0.0, 1.0]).all():
            raise ValueError("US SSI disability receiver is_married must be boolean.")
        receiver["is_married"] = married
    elif "A_MARITL" in person:
        marital = _strict_person_numeric(person, ("A_MARITL",), label="marriage")
        receiver["is_married"] = np.isin(marital, [1.0, 2.0]).astype(np.float64)
    else:
        raise ValueError(
            "US SSI disability receiver requires measured is_married or A_MARITL."
        )

    receiver["employment_income"] = _strict_person_numeric(
        person,
        ("employment_income_before_lsr", "employment_income", "WSAL_VAL"),
        label="employment income",
    )
    receiver["interest_income"] = _strict_person_aggregate(
        person,
        aggregate_column="interest_income",
        component_columns=_RECIPIENT_INTEREST_COMPONENT_COLUMNS,
    )
    receiver["dividend_income"] = _strict_person_aggregate(
        person,
        aggregate_column="dividend_income",
        component_columns=_RECIPIENT_DIVIDEND_COMPONENT_COLUMNS,
    )
    for column in (
        "rental_income",
        "bank_account_assets",
        "stock_assets",
        "bond_assets",
        "social_security_disability",
    ):
        receiver[column] = _strict_person_numeric(person, (column,), label=column)

    age = receiver["age"]
    receiver["count_under_18"] = (
        age.lt(18.0)
        .groupby(person["person_household_id"], sort=False)
        .transform("sum")
        .to_numpy(dtype=np.float64)
    )
    for predictor, source_column in _ASEC_DIFFICULTY_SOURCE_COLUMNS.items():
        source = _strict_person_numeric(
            person,
            (source_column,),
            label=predictor,
        )
        receiver[predictor] = (source == 1.0).astype(np.float64)
    disability_income = _strict_person_numeric(
        person,
        ("disability_benefits",),
        label="disability income",
    )
    receiver["has_disability_income"] = (disability_income > 0.0).astype(np.float64)

    values = receiver.loc[:, list(SIPP_SSI_DISABILITY_MODEL_PREDICTORS)].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("US SSI disability receiver predictors must be finite.")
    return receiver.loc[:, list(SIPP_SSI_DISABILITY_MODEL_PREDICTORS)]


def _validate_support_provenance(person: pd.DataFrame) -> None:
    if not has_support_role_metadata(person, entity="person"):
        return
    channels = support_role_series(person, entity="person")
    known = {_BASE_ASEC_SUPPORT_CHANNEL, _PUF_TAX_DETAIL_SUPPORT_CHANNEL}
    observed = set(channels.unique())
    unexpected = sorted(observed - known)
    missing = sorted(known - observed)
    if unexpected or missing:
        raise ValueError(
            "US SSI disability receiver requires exact native/PUF support "
            f"roles; missing {missing}, unsupported support role(s) "
            f"{unexpected}."
        )
    if (
        _PERSON_SOURCE_ID_COLUMN not in person
        or person[_PERSON_SOURCE_ID_COLUMN].isna().any()
    ):
        raise ValueError(
            "US SSI disability support rows require complete person_source_id "
            "provenance; PUF-only IDs are allowed but missing IDs are not."
        )


def _coerce_boolean_predictions(values: pd.Series | np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    if np.issubdtype(series.dtype, np.number):
        numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("SIPP SSI disability QRF produced nonfinite values.")
        if ((numeric < 0.0) | (numeric > 1.0)).any():
            raise ValueError("SIPP SSI disability QRF produced values outside [0, 1].")
        return numeric >= 0.5
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    valid = normalized.isin(["true", "false", "1", "0", "yes", "no"])
    if not valid.all():
        raise ValueError("SIPP SSI disability QRF produced invalid class labels.")
    return normalized.isin(["true", "1", "yes"]).to_numpy(dtype=bool)


def _weighted_replacement_sample(donor: pd.DataFrame) -> pd.DataFrame:
    weights = pd.to_numeric(donor[_DONOR_WEIGHT_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not (np.isfinite(weights) & (weights > 0.0)).all():
        raise ValueError(
            "SIPP SSI disability donor weights must be finite and positive."
        )
    probability = weights / weights.sum()
    rng = np.random.default_rng(_TRAINING_SAMPLE_SEED)
    selected = rng.choice(
        len(donor),
        size=min(_MAX_TRAIN_SAMPLES, len(donor)),
        replace=True,
        p=probability,
    )
    return donor.iloc[selected].reset_index(drop=True)


def impute_us_ssi_disability_criteria(
    frame: Frame,
    donor: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> pd.Series:
    """Fit the archived SIPP model and predict every ASEC/PUF support row."""

    required = {
        *SIPP_SSI_DISABILITY_MODEL_PREDICTORS,
        _OUTPUT,
        _DONOR_WEIGHT_COLUMN,
    }
    missing = sorted(required - set(donor.columns))
    if missing:
        raise ValueError(f"SIPP SSI disability donor missing column(s): {missing}.")
    del seed  # The archived model fixed both its source draw and forest seed.
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")

    training = donor.loc[
        :, [*SIPP_SSI_DISABILITY_MODEL_PREDICTORS, _OUTPUT, _DONOR_WEIGHT_COLUMN]
    ].copy()
    for column in [*SIPP_SSI_DISABILITY_MODEL_PREDICTORS, _DONOR_WEIGHT_COLUMN]:
        training[column] = pd.to_numeric(training[column], errors="coerce")
    values = training.loc[
        :, [*SIPP_SSI_DISABILITY_MODEL_PREDICTORS, _DONOR_WEIGHT_COLUMN]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("SIPP SSI disability donor predictors/weights must be finite.")
    target = pd.to_numeric(training[_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not (np.isfinite(target) & np.isin(target, [0.0, 1.0])).all():
        raise ValueError("SIPP SSI disability donor target must be boolean.")
    if np.unique(target).size != 2:
        raise ValueError("SIPP SSI disability donor target must contain both classes.")
    training[_OUTPUT] = target
    training = _weighted_replacement_sample(training)

    person = frame.table("person")
    _validate_support_provenance(person)
    receiver = _person_ssi_disability_predictors(frame)
    global QRF
    if QRF is None:
        from importlib import import_module

        QRF = import_module("microcosm.fit").QRF
    fitted = QRF(n_estimators=int(n_estimators), seed=_ARCHIVED_MODEL_SEED).fit(
        training,
        predictors=list(SIPP_SSI_DISABILITY_MODEL_PREDICTORS),
        targets=[_OUTPUT],
        weights="none",
    )
    if not has_support_role_metadata(person, entity="person"):
        prediction = fitted.predict(receiver)
        if _OUTPUT not in prediction:
            raise ValueError(f"SIPP SSI disability QRF prediction missing {_OUTPUT!r}.")
        predicted = _coerce_boolean_predictions(prediction[_OUTPUT])
    else:
        # The retired direct-CPS and extended-CPS paths each loaded a fresh
        # copy of the same fitted model before drawing their channel. Microcosm
        # QRF predictions advance a seeded draw RNG, so one combined call would
        # make PUF outcomes depend on the number/order of preceding ASEC rows.
        # Deep-copying the pristine fit per channel reproduces the two fresh
        # prediction streams without refitting the reviewed forest.
        channels = support_role_series(person, entity="person")
        predicted = np.zeros(len(person), dtype=bool)
        for channel in (
            _BASE_ASEC_SUPPORT_CHANNEL,
            _PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        ):
            mask = channels.eq(channel).to_numpy()
            channel_model = deepcopy(fitted)
            prediction = channel_model.predict(receiver.loc[mask])
            if _OUTPUT not in prediction:
                raise ValueError(
                    f"SIPP SSI disability QRF prediction missing {_OUTPUT!r} "
                    f"for support role {channel!r}."
                )
            predicted[mask] = _coerce_boolean_predictions(prediction[_OUTPUT])
            del channel_model

    difficulty_signal = (
        receiver.loc[:, list(SIPP_SSI_DISABILITY_DIFFICULTY_PREDICTORS)]
        .astype(bool)
        .any(axis=1)
        .to_numpy()
    )
    disability_signal = (
        difficulty_signal
        | (receiver["social_security_disability"].to_numpy(dtype=np.float64) > 0.0)
        | receiver["has_disability_income"].to_numpy(dtype=np.float64).astype(bool)
    )
    result = predicted & disability_signal

    # The archived direct-CPS pass preserves measured SSI reporters.  Its PUF
    # clone override does not, even though raw ASEC columns were duplicated.
    receiver_age = receiver["age"].to_numpy(dtype=np.float64)
    reported_ssi = _reported_ssi_anchor(person, age=receiver_age) > 0.0
    under_65 = receiver_age < 65.0
    if has_support_role_metadata(person, entity="person"):
        channels = support_role_series(person, entity="person")
        asec = channels.eq(_BASE_ASEC_SUPPORT_CHANNEL).to_numpy()
    else:
        asec = np.ones(len(person), dtype=bool)
    result |= asec & under_65 & reported_ssi
    return pd.Series(result, index=person.index, name=_OUTPUT, dtype=bool)


def with_us_ssi_disability_criteria(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    sipp_donor: pd.DataFrame,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> Frame:
    """Recompute and materialize the source-backed person input."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US SSI disability criteria require the US schema.")
    del time_period  # Donor construction owns the reviewed parameter vintage.
    us_ssi_disability_criteria_stage_spec()
    predicted = impute_us_ssi_disability_criteria(
        frame,
        sipp_donor,
        seed=int(seed),
        n_estimators=int(n_estimators),
    )
    person = frame.table("person")
    if _OUTPUT in person:
        current = pd.to_numeric(person[_OUTPUT], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if (
            np.isfinite(current).all()
            and np.isin(current, [0.0, 1.0]).all()
            and np.array_equal(current.astype(bool), predicted.to_numpy())
        ):
            return frame

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"][_OUTPUT] = predicted.to_numpy(dtype=bool)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_ssi_disability_criteria_summary(frame: Frame) -> dict[str, object]:
    """Return weighted incidence, channel signal, and source-anchor diagnostics."""

    person = frame.table("person")
    values = pd.to_numeric(person[_OUTPUT], errors="coerce").to_numpy(dtype=np.float64)
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    if not (np.isfinite(weights) & (weights >= 0.0)).all() or weights.sum() <= 0.0:
        raise ValueError(
            "US SSI disability gate requires finite nonnegative person weights "
            "with positive total."
        )
    finite = np.isfinite(values)
    boolean = finite & np.isin(values, [0.0, 1.0])
    positive = boolean & (values == 1.0)
    total_weight = float(weights.sum())

    channels: dict[str, dict[str, float | int]] = {}
    has_support_roles = has_support_role_metadata(person, entity="person")
    channel_values: pd.Series | None = None
    if has_support_roles:
        try:
            channel_values = support_role_series(person, entity="person")
        except ValueError:
            has_support_roles = False
    provenance_missing = bool(
        not has_support_roles
        or _PERSON_SOURCE_ID_COLUMN not in person
        or person[_PERSON_SOURCE_ID_COLUMN].isna().any()
    )
    clone_divergence_source_people = 0
    if channel_values is not None:
        for channel in sorted(channel_values.unique()):
            mask = channel_values.eq(channel).to_numpy()
            channel_weight = float(weights[mask].sum())
            channels[channel] = {
                "rows": int(mask.sum()),
                "unique_count": int(pd.Series(values[mask & finite]).nunique()),
                "weighted_true_share": (
                    float(weights[mask & positive].sum()) / channel_weight
                    if channel_weight > 0.0
                    else 0.0
                ),
            }
        if not provenance_missing:
            clone_table = pd.DataFrame(
                {
                    "source_id": _decoded_strings(person[_PERSON_SOURCE_ID_COLUMN]),
                    "role": channel_values,
                    "value": values,
                }
            )
            if has_assembled_support_metadata(person, entity="person"):
                clone_groups = ["source_id"]
            else:
                clone_table["source_occurrence"] = clone_table.groupby(
                    ["source_id", "role"], sort=False
                ).cumcount()
                clone_groups = ["source_id", "source_occurrence"]
            unique = clone_table.groupby(clone_groups, sort=False)["value"].nunique(
                dropna=False
            )
            clone_divergence_source_people = int((unique > 1).sum())

    age_column = "age" if "age" in person else "A_AGE"
    age = pd.to_numeric(person[age_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    reported = _reported_ssi_anchor(person, age=age) > 0.0
    native_role = np.ones(len(person), dtype=bool)
    if channel_values is not None:
        native_role = channel_values.eq(_BASE_ASEC_SUPPORT_CHANNEL).to_numpy()
    anchor = reported & (age < 65.0) & native_role
    reporter_mismatches = int(np.count_nonzero(anchor & ~positive))

    return {
        "weighted_true_share": float(weights[positive].sum()) / total_weight,
        "weighted_true_total": float(weights[positive].sum()),
        "weighted_false_total": float(weights[boolean & ~positive].sum()),
        "true_share_band": list(_TRUE_SHARE_BAND),
        "unique_count": int(pd.Series(values[finite]).nunique()),
        "missing_or_nonfinite_count": int((~finite).sum()),
        "invalid_boolean_count": int((finite & ~boolean).sum()),
        "reporter_anchor_mismatches": reporter_mismatches,
        "support_provenance_missing": provenance_missing,
        "clone_divergence_source_people": clone_divergence_source_people,
        "channels": channels,
    }


def us_ssi_disability_criteria_signal_gate(frame: Frame) -> GateResult:
    """Require a valid, nonconstant signal in ASEC and PUF support channels."""

    person = frame.table("person")
    if _OUTPUT not in person:
        return GateResult(
            name="ssi_disability_criteria_signal",
            passed=False,
            failures=(f"person column missing: {_OUTPUT!r}.",),
            details={"missing": [_OUTPUT]},
        )
    summary = us_ssi_disability_criteria_summary(frame)
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
    if summary["unique_count"] < 2:
        failures.append(f"{_OUTPUT}: constant column carries no signal.")
    low, high = summary["true_share_band"]
    share = float(summary["weighted_true_share"])
    if not (low <= share <= high):
        failures.append(
            f"{_OUTPUT}: weighted true share {share:.4f} outside plausibility "
            f"band [{low}, {high}]."
        )
    if float(summary["weighted_true_total"]) <= 0.0:
        failures.append(f"{_OUTPUT}: weighted true total is not positive.")
    if float(summary["weighted_false_total"]) <= 0.0:
        failures.append(f"{_OUTPUT}: weighted false total is not positive.")
    if summary["reporter_anchor_mismatches"]:
        failures.append(
            f"{_OUTPUT}: {summary['reporter_anchor_mismatches']} under-65 "
            "native-role SSI reporter anchor(s) were lost."
        )
    if summary["support_provenance_missing"]:
        failures.append(
            "SSI disability support rows lack complete clone-role or "
            "person_source_id provenance."
        )
    channels = summary["channels"]
    required_channels = {
        _BASE_ASEC_SUPPORT_CHANNEL,
        _PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    }
    unexpected_channels = sorted(set(channels) - required_channels)
    if unexpected_channels:
        failures.append(
            f"{_OUTPUT}: unsupported support channel(s) {unexpected_channels}."
        )
    for required_channel in sorted(required_channels):
        diagnostics = channels.get(required_channel)
        if diagnostics is None:
            failures.append(
                f"{_OUTPUT}: support channel {required_channel!r} is missing."
            )
        elif diagnostics["unique_count"] < 2:
            failures.append(
                f"{_OUTPUT}: support channel {required_channel!r} is constant."
            )
        elif not (low <= float(diagnostics["weighted_true_share"]) <= high):
            failures.append(
                f"{_OUTPUT}: support channel {required_channel!r} weighted "
                "true share "
                f"{float(diagnostics['weighted_true_share']):.4f} outside "
                f"plausibility band [{low}, {high}]."
            )
    return GateResult(
        name="ssi_disability_criteria_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
