"""Measured SIPP Head Start proxy for eligible preschool-age children.

The 2023 SIPP asks whether a nursery-school or preschool enrollee attends a
federally sponsored program, naming Head Start, Even Start, and Fair Start as
examples.  This stage uses that measured December response as the available
Head Start proxy to restore ``takes_up_head_start_if_eligible``; it does not
misdescribe the instrument as a program-only identifier.  It also does not use
the retired NIEER scalar or stand in for Early Head Start, whose infant/toddler
population is outside the source question and PolicyEngine Head Start domain.

Labels fail closed.  Direct ``EEDHEADST`` answers are retained only when
``AEDHEADST == 1`` (as reported).  A not-in-universe row is a negative only
when an upstream reported screen proves either no school enrollment or a
reported grade other than nursery/preschool.  Hot-decked Head Start answers,
imputed upstream screens, and unknown screen states never enter training.
That yields 45 positive and 740 strict negative labels; the latter deliberately
differs from a looser status-only 743-negative mask because three rows lack a
reported upstream screen and therefore cannot support a measured negative.

A weighted QRF is fit to the immutable, SHA-pinned full 2023 SIPP donor.  It
predicts once for each stable ``person_source_id`` in the PolicyEngine age
3--5 domain and fans that decision to all support clones.  Existing output is
used only for an exact idempotence check; a stale or default surface is healed.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.build.us_runtime.full_sipp_donor import full_sipp_sha256
from populace.build.us_runtime.support_provenance import (
    has_support_role_metadata,
    support_role_series,
)
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
    "HEAD_START_SIPP_DICTIONARY_URL",
    "SIPP_2023_HEAD_START_DONOR_REVISION",
    "SIPP_2023_HEAD_START_DONOR_SHA256",
    "SIPP_2023_HEAD_START_DONOR_SIZE_BYTES",
    "SIPP_2023_HEAD_START_DONOR_URL",
    "SIPP_HEAD_START_FIT_PARAMETERS",
    "SIPP_HEAD_START_MODEL_PREDICTORS",
    "SIPP_HEAD_START_READ_PARAMETERS",
    "SIPP_HEAD_START_SOURCE_COLUMNS",
    "US_SIPP_HEAD_START_NONCONSTANT_PERSON_COLUMNS",
    "US_SIPP_HEAD_START_OUTPUT_COLUMNS",
    "US_SIPP_HEAD_START_REQUIRED_SOURCE_COLUMNS",
    "US_SIPP_HEAD_START_STAGE_NAME",
    "fetch_sipp_2023_head_start_donor",
    "impute_us_sipp_head_start",
    "load_sipp_2023_head_start_donor",
    "us_sipp_head_start_signal_gate",
    "us_sipp_head_start_stage_spec",
    "us_sipp_head_start_summary",
    "with_us_sipp_head_start_input",
]

QRF: Any | None = None

HEAD_START_SIPP_DICTIONARY_URL = (
    "https://www2.census.gov/programs-surveys/sipp/tech-documentation/"
    "data-dictionaries/2023/2023_SIPP_Data_Dictionary.pdf"
)

# This family deliberately aliases the already reviewed full-file coordinate
# instead of introducing another mutable donor locator.
SIPP_2023_HEAD_START_DONOR_REVISION = SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION
SIPP_2023_HEAD_START_DONOR_SHA256 = SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256
SIPP_2023_HEAD_START_DONOR_SIZE_BYTES = SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES
SIPP_2023_HEAD_START_DONOR_URL = SIPP_2023_VOLUNTARY_FILING_DONOR_URL

US_SIPP_HEAD_START_STAGE_NAME = "sipp_head_start"
US_SIPP_HEAD_START_OUTPUT_COLUMNS: tuple[str, ...] = (
    "takes_up_head_start_if_eligible",
)
US_SIPP_HEAD_START_NONCONSTANT_PERSON_COLUMNS = US_SIPP_HEAD_START_OUTPUT_COLUMNS
US_SIPP_HEAD_START_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "person_source_id",
    "person_household_id",
    "age",
    "is_female",
    "employment_income_before_lsr",
)

_OUTPUT = US_SIPP_HEAD_START_OUTPUT_COLUMNS[0]
_DONOR_WEIGHT_COLUMN = "sipp_weight"
_PERSON_SOURCE_ID_COLUMN = "person_source_id"
_ASEC_CHANNEL = "asec"
_PUF_CHANNEL = "puf_tax_detail"
_DEFAULT_N_ESTIMATORS = 100
_ELIGIBLE_MIN_AGE = 3
_ELIGIBLE_MAX_AGE = 5
_ELIGIBLE_TAKE_UP_SHARE_BAND = (0.005, 0.25)

_SIPP_JOB_EARNINGS_COLUMNS = tuple(f"TJB{i}_MSUM" for i in range(1, 8))
SIPP_HEAD_START_SOURCE_COLUMNS: tuple[str, ...] = (
    "SSUID",
    "PNUM",
    "MONTHCODE",
    "WPFINWGT",
    "TAGE",
    "ESEX",
    "EED_SCRNR",
    "AED_SCRNR",
    "EEDGRADE",
    "AEDGRADE",
    "EEDEMONTH",
    "AEDMONTH",
    "EEDHEADST",
    "AEDHEADST",
    *_SIPP_JOB_EARNINGS_COLUMNS,
)
SIPP_HEAD_START_MODEL_PREDICTORS: tuple[str, ...] = (
    "age",
    "is_female",
    "household_size",
    "count_under_18",
    "count_under_6",
    "household_employment_income",
)

SIPP_HEAD_START_READ_PARAMETERS: dict[str, object] = {
    "table": "sipp_person",
    "delimiter": "|",
    "month_column": "MONTHCODE",
    "month": 12,
    "source_columns": list(SIPP_HEAD_START_SOURCE_COLUMNS),
}
SIPP_HEAD_START_FIT_PARAMETERS: dict[str, object] = {
    "predictors": list(SIPP_HEAD_START_MODEL_PREDICTORS),
    "target": _OUTPUT,
    "weight": _DONOR_WEIGHT_COLUMN,
    "age_domain": [_ELIGIBLE_MIN_AGE, _ELIGIBLE_MAX_AGE],
    "direct_response_filter": "AEDHEADST == 1 and EEDHEADST in [1, 2]",
    "structural_no_filters": [
        "AEDHEADST == 0 and AED_SCRNR == 1 and EED_SCRNR == 2",
        (
            "AEDHEADST == 0 and AED_SCRNR == 1 and EED_SCRNR == 1 "
            "and AEDGRADE == 1 and EEDGRADE != 21"
        ),
    ],
    "excluded_statuses": "AEDHEADST == 2 or AED_SCRNR == 4",
    "n_estimators": _DEFAULT_N_ESTIMATORS,
    "seed_from_build_config": True,
    "assignment_unit": _PERSON_SOURCE_ID_COLUMN,
    "fan_to_support_clones": True,
}

# Exact audit of the immutable full 2023 artifact under the transform above.
_PINNED_RAW_ROWS = 476_744
_PINNED_DECEMBER_ROWS = 39_513
_PINNED_AGE_DOMAIN_ROWS = 1_177
_PINNED_TRAINING_ROWS = 785
_PINNED_POSITIVE_ROWS = 45
_PINNED_NEGATIVE_ROWS = 740
_PINNED_DIRECT_RESPONSE_ROWS = 215
_PINNED_REPORTED_NO_ENROLLMENT_ROWS = 440
_PINNED_REPORTED_OTHER_GRADE_ROWS = 130
_PINNED_WEIGHT_SUM = 7_978_494.5412483
_PINNED_POSITIVE_WEIGHT_SUM = 491_970.1041311
_PINNED_WEIGHTED_TRUE_SHARE = 0.06166202177461505


def us_sipp_head_start_stage_spec() -> SourceStageSpec:
    """Load and validate the packaged Head Start stage declaration."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_SIPP_HEAD_START_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_SIPP_HEAD_START_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_SIPP_HEAD_START_STAGE_NAME]
    if spec.grain != "person":
        raise ValueError("US SIPP Head Start stage must have person grain.")
    if tuple(spec.outputs) != US_SIPP_HEAD_START_OUTPUT_COLUMNS:
        raise ValueError(
            "US SIPP Head Start manifest outputs do not match the runtime-owned family."
        )
    if [operation.kind for operation in spec.operations] != [
        "read_table",
        "fit_weighted_qrf",
    ]:
        raise ValueError(
            "US SIPP Head Start stage must contain read_table then fit_weighted_qrf."
        )
    if dict(spec.operations[0].parameters) != SIPP_HEAD_START_READ_PARAMETERS:
        raise ValueError(
            "US SIPP Head Start read contract drifted from the pinned transform."
        )
    if dict(spec.operations[1].parameters) != SIPP_HEAD_START_FIT_PARAMETERS:
        raise ValueError(
            "US SIPP Head Start fit contract drifted from the reviewed labels."
        )
    pinned = [
        artifact
        for artifact in spec.artifacts
        if artifact.get("sha256") == SIPP_2023_HEAD_START_DONOR_SHA256
        and artifact.get("size_bytes") == SIPP_2023_HEAD_START_DONOR_SIZE_BYTES
    ]
    if not pinned:
        raise ValueError(
            "US SIPP Head Start stage does not pin the full donor SHA-256 and "
            "byte length."
        )
    return spec


def _sha256_stream(stream: BinaryIO, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(chunk_size), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return full_sipp_sha256(path)


def fetch_sipp_2023_head_start_donor(
    cache_dir: str | Path | None = None,
    *,
    expected_sha256: str | None = SIPP_2023_HEAD_START_DONOR_SHA256,
    expected_size_bytes: int | None = SIPP_2023_HEAD_START_DONOR_SIZE_BYTES,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Fetch the already-pinned full SIPP donor through its shared cache."""

    return fetch_sipp_2023_voluntary_filing_donor(
        cache_dir,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        chunk_size=chunk_size,
    )


def _numeric(source: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(source[column], errors="coerce").to_numpy(dtype=np.float64)


def _require_finite(
    values: np.ndarray,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> np.ndarray:
    valid = np.isfinite(values)
    if minimum is not None:
        valid &= values >= minimum
    if maximum is not None:
        valid &= values <= maximum
    if not valid.all():
        rows = np.flatnonzero(~valid)[:5].tolist()
        raise ValueError(f"SIPP Head Start {label} is invalid at row(s): {rows}.")
    return values


def _assert_pinned_audit(audit: dict[str, int | float]) -> None:
    expected_counts = {
        "raw_rows": _PINNED_RAW_ROWS,
        "december_rows": _PINNED_DECEMBER_ROWS,
        "age_domain_rows": _PINNED_AGE_DOMAIN_ROWS,
        "training_rows": _PINNED_TRAINING_ROWS,
        "positive_rows": _PINNED_POSITIVE_ROWS,
        "negative_rows": _PINNED_NEGATIVE_ROWS,
        "direct_response_rows": _PINNED_DIRECT_RESPONSE_ROWS,
        "reported_no_enrollment_rows": _PINNED_REPORTED_NO_ENROLLMENT_ROWS,
        "reported_other_grade_rows": _PINNED_REPORTED_OTHER_GRADE_ROWS,
    }
    for key, expected in expected_counts.items():
        if int(audit[key]) != expected:
            raise ValueError(
                f"Pinned SIPP Head Start audit drifted for {key}: "
                f"expected {expected}, got {audit[key]}."
            )
    expected_floats = {
        "weight_sum": _PINNED_WEIGHT_SUM,
        "positive_weight_sum": _PINNED_POSITIVE_WEIGHT_SUM,
        "weighted_true_share": _PINNED_WEIGHTED_TRUE_SHARE,
    }
    for key, expected in expected_floats.items():
        if not np.isclose(float(audit[key]), expected, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"Pinned SIPP Head Start audit drifted for {key}: "
                f"expected {expected}, got {audit[key]}."
            )


def load_sipp_2023_head_start_donor(
    path: str | Path,
    *,
    expected_sha256: str | None = SIPP_2023_HEAD_START_DONOR_SHA256,
    expected_size_bytes: int | None = SIPP_2023_HEAD_START_DONOR_SIZE_BYTES,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Load strict December age-3--5 labels from the pinned full SIPP file."""

    source_path = Path(path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    if (
        expected_size_bytes is not None
        and source_path.stat().st_size != expected_size_bytes
    ):
        raise ValueError(
            "SIPP Head Start donor byte length does not match the pinned artifact."
        )
    if expected_sha256 is not None:
        digest = _sha256_file(source_path)
        if digest != expected_sha256:
            raise ValueError(
                "SIPP Head Start donor SHA-256 does not match the pinned artifact."
            )

    header = pd.read_csv(source_path, sep="|", nrows=0)
    missing = sorted(set(SIPP_HEAD_START_SOURCE_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"SIPP Head Start donor missing source column(s): {missing}.")

    december_parts: list[pd.DataFrame] = []
    raw_rows = 0
    for chunk in pd.read_csv(
        source_path,
        sep="|",
        usecols=list(SIPP_HEAD_START_SOURCE_COLUMNS),
        chunksize=chunksize,
        low_memory=False,
    ):
        raw_rows += len(chunk)
        month = pd.to_numeric(chunk["MONTHCODE"], errors="coerce")
        december_parts.append(chunk.loc[month.eq(12)].copy())
    december = pd.concat(december_parts, ignore_index=True)
    if december.empty:
        raise ValueError("SIPP Head Start donor contains no December rows.")
    if december[["SSUID", "PNUM"]].isna().any(axis=None):
        raise ValueError("SIPP Head Start December identity contains missing values.")
    if december.duplicated(["SSUID", "PNUM"]).any():
        raise ValueError("SIPP Head Start December identity is not person-unique.")

    age = _require_finite(
        _numeric(december, "TAGE"), label="age", minimum=0.0, maximum=120.0
    )
    sex = _require_finite(
        _numeric(december, "ESEX"), label="sex code", minimum=1.0, maximum=2.0
    )
    if not np.isin(sex, [1.0, 2.0]).all():
        raise ValueError("SIPP Head Start ESEX must contain only 1 or 2.")
    weight = _require_finite(
        _numeric(december, "WPFINWGT"), label="weight", minimum=0.0
    )

    jobs = december.loc[:, list(_SIPP_JOB_EARNINGS_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    # Blank job slots are structural zeroes. Negative earnings are not a
    # valid value for the SIPP monthly job-earnings recodes.
    jobs = jobs.fillna(0.0)
    if (jobs.to_numpy(dtype=np.float64) < 0.0).any():
        raise ValueError("SIPP Head Start job earnings must be nonnegative.")
    monthly_person_earnings = jobs.sum(axis=1).to_numpy(dtype=np.float64)

    features = pd.DataFrame(index=december.index)
    features["age"] = age
    features["is_female"] = sex == 2.0
    household = december["SSUID"]
    features["household_size"] = household.groupby(household, sort=False).transform(
        "size"
    )
    features["count_under_18"] = (
        pd.Series(age < 18.0, index=december.index)
        .groupby(household, sort=False)
        .transform("sum")
    )
    features["count_under_6"] = (
        pd.Series(age < 6.0, index=december.index)
        .groupby(household, sort=False)
        .transform("sum")
    )
    features["household_employment_income"] = (
        pd.Series(monthly_person_earnings * 12.0, index=december.index)
        .groupby(household, sort=False)
        .transform("sum")
    )

    in_age_domain = (age >= _ELIGIBLE_MIN_AGE) & (age <= _ELIGIBLE_MAX_AGE)
    direct_status = _numeric(december, "AEDHEADST")
    direct_answer = _numeric(december, "EEDHEADST")
    screen_status = _numeric(december, "AED_SCRNR")
    screen = _numeric(december, "EED_SCRNR")
    grade_status = _numeric(december, "AEDGRADE")
    grade = _numeric(december, "EEDGRADE")

    reported_direct = (
        in_age_domain & (direct_status == 1.0) & np.isin(direct_answer, [1.0, 2.0])
    )
    reported_no_enrollment = (
        in_age_domain
        & (direct_status == 0.0)
        & (screen_status == 1.0)
        & (screen == 2.0)
    )
    reported_other_grade = (
        in_age_domain
        & (direct_status == 0.0)
        & (screen_status == 1.0)
        & (screen == 1.0)
        & (grade_status == 1.0)
        & np.isfinite(grade)
        & (grade != 21.0)
    )
    training_mask = reported_direct | reported_no_enrollment | reported_other_grade
    positive = reported_direct & (direct_answer == 1.0)

    donor = features.loc[training_mask, list(SIPP_HEAD_START_MODEL_PREDICTORS)].copy()
    donor[_OUTPUT] = positive[training_mask]
    donor[_DONOR_WEIGHT_COLUMN] = weight[training_mask]
    donor = donor.reset_index(drop=True)
    if (donor[_DONOR_WEIGHT_COLUMN] <= 0.0).any():
        raise ValueError("SIPP Head Start training weights must be positive.")
    if donor.empty or donor[_OUTPUT].nunique() != 2:
        raise ValueError("SIPP Head Start donor must contain both observed classes.")

    positive_weight = float(donor.loc[donor[_OUTPUT], _DONOR_WEIGHT_COLUMN].sum())
    weight_sum = float(donor[_DONOR_WEIGHT_COLUMN].sum())
    audit: dict[str, int | float] = {
        "raw_rows": raw_rows,
        "december_rows": len(december),
        "age_domain_rows": int(np.count_nonzero(in_age_domain)),
        "training_rows": len(donor),
        "positive_rows": int(donor[_OUTPUT].sum()),
        "negative_rows": int((~donor[_OUTPUT]).sum()),
        "direct_response_rows": int(np.count_nonzero(reported_direct)),
        "reported_no_enrollment_rows": int(np.count_nonzero(reported_no_enrollment)),
        "reported_other_grade_rows": int(np.count_nonzero(reported_other_grade)),
        "weight_sum": weight_sum,
        "positive_weight_sum": positive_weight,
        "weighted_true_share": positive_weight / weight_sum,
    }
    pinned_transform = bool(
        expected_sha256 == SIPP_2023_HEAD_START_DONOR_SHA256
        and expected_size_bytes == SIPP_2023_HEAD_START_DONOR_SIZE_BYTES
    )
    if pinned_transform:
        _assert_pinned_audit(audit)
    audit["pinned_transform"] = pinned_transform
    donor.attrs["source_audit"] = audit
    return donor


def _decoded_strings(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: value.decode() if isinstance(value, (bytes, bytearray)) else value
    ).astype(str)


def _support_group_keys(
    person: pd.DataFrame,
    source_id: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Return clone-pair keys without consulting source-spine channels."""

    if not has_support_role_metadata(person, entity="person"):
        return (
            source_id.astype(object),
            pd.Series(_ASEC_CHANNEL, index=person.index),
        )
    try:
        roles = support_role_series(person, entity="person")
    except ValueError as exc:
        raise ValueError(
            "US SIPP Head Start found unsupported support channel or "
            f"clone-role metadata: {exc}"
        ) from exc
    return source_id.astype(object), roles


def _recipient_predictors(frame: Frame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    person = frame.table("person")
    missing = [
        column
        for column in US_SIPP_HEAD_START_REQUIRED_SOURCE_COLUMNS
        if column not in person
    ]
    if missing:
        raise ValueError(
            f"US SIPP Head Start receiver missing person column(s): {missing}."
        )
    if person[_PERSON_SOURCE_ID_COLUMN].isna().any():
        raise ValueError("US SIPP Head Start requires complete person_source_id.")

    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(dtype=np.float64)
    female_raw = pd.to_numeric(person["is_female"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    earnings = pd.to_numeric(
        person["employment_income_before_lsr"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    valid = (
        np.isfinite(age)
        & (age >= 0.0)
        & (age <= 120.0)
        & np.isfinite(female_raw)
        & np.isin(female_raw, [0.0, 1.0])
        & np.isfinite(earnings)
        & (earnings >= 0.0)
    )
    if not valid.all():
        rows = np.flatnonzero(~valid)[:5].tolist()
        raise ValueError(
            "US SIPP Head Start receiver age, sex, and earnings must be valid; "
            f"invalid row(s): {rows}."
        )
    if person["person_household_id"].isna().any():
        raise ValueError("US SIPP Head Start requires complete household linkage.")

    household = person["person_household_id"]
    features = pd.DataFrame(index=person.index)
    features["age"] = age
    features["is_female"] = female_raw
    features["household_size"] = household.groupby(household, sort=False).transform(
        "size"
    )
    features["count_under_18"] = (
        pd.Series(age < 18.0, index=person.index)
        .groupby(household, sort=False)
        .transform("sum")
    )
    features["count_under_6"] = (
        pd.Series(age < 6.0, index=person.index)
        .groupby(household, sort=False)
        .transform("sum")
    )
    features["household_employment_income"] = (
        pd.Series(earnings, index=person.index)
        .groupby(household, sort=False)
        .transform("sum")
    )

    source_id = _decoded_strings(person[_PERSON_SOURCE_ID_COLUMN])
    source_key, roles = _support_group_keys(person, source_id)
    age_unique = (
        pd.Series(age, index=person.index).groupby(source_key, sort=False).nunique()
    )
    inconsistent = age_unique.index[age_unique > 1].tolist()
    if inconsistent:
        raise ValueError(
            "US SIPP Head Start source clones disagree on age for "
            f"person_source_id(s): {inconsistent[:5]}."
        )
    role_rows = pd.DataFrame({"source_id": source_id, "role": roles})
    duplicate_roles = role_rows.duplicated(
        ["source_id", "role"],
        keep=False,
    )
    if duplicate_roles.any():
        bad = (
            role_rows.loc[duplicate_roles, ["source_id", "role"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        raise ValueError(
            "US SIPP Head Start source units carry duplicated same-role rows; "
            f"invalid source role(s): {list(bad)[:5]}."
        )

    order = pd.DataFrame(index=person.index)
    order["source_id"] = source_id
    order["source_key"] = source_key
    order["role_priority"] = roles.map({_ASEC_CHANNEL: 0, _PUF_CHANNEL: 1})
    order["person_key"] = person["person_id"].astype(str)
    canonical_index = (
        order.sort_values(
            ["source_id", "role_priority", "person_key"],
            kind="mergesort",
        )
        .drop_duplicates("source_key", keep="first")
        .index
    )
    canonical = features.loc[canonical_index].copy()
    canonical.insert(0, "source_key", source_key.loc[canonical_index].to_numpy())
    canonical.insert(1, "source_id", source_id.loc[canonical_index].to_numpy())
    canonical = canonical.sort_values("source_id", kind="mergesort").reset_index(
        drop=True
    )
    values = canonical.loc[:, list(SIPP_HEAD_START_MODEL_PREDICTORS)].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("US SIPP Head Start receiver predictors must be finite.")
    eligible = (
        canonical["age"]
        .between(_ELIGIBLE_MIN_AGE, _ELIGIBLE_MAX_AGE, inclusive="both")
        .to_numpy()
    )
    return canonical, source_key, eligible


def _coerce_boolean_prediction(values: pd.Series | np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if not (np.isfinite(numeric) & (numeric >= 0.0) & (numeric <= 1.0)).all():
        raise ValueError("SIPP Head Start QRF produced invalid boolean predictions.")
    return numeric >= 0.5


def impute_us_sipp_head_start(
    frame: Frame,
    donor: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> pd.Series:
    """Fit the measured SIPP model and predict once per stable source person."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US SIPP Head Start imputation requires the US schema.")
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    required = [
        *SIPP_HEAD_START_MODEL_PREDICTORS,
        _OUTPUT,
        _DONOR_WEIGHT_COLUMN,
    ]
    missing = [column for column in required if column not in donor]
    if missing:
        raise ValueError(f"SIPP Head Start donor missing column(s): {missing}.")
    training = donor.loc[:, required].copy()
    for column in [*SIPP_HEAD_START_MODEL_PREDICTORS, _DONOR_WEIGHT_COLUMN]:
        training[column] = pd.to_numeric(training[column], errors="coerce")
    numeric = training.loc[
        :, [*SIPP_HEAD_START_MODEL_PREDICTORS, _DONOR_WEIGHT_COLUMN]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or np.any(
        training[_DONOR_WEIGHT_COLUMN].to_numpy(dtype=np.float64) <= 0.0
    ):
        raise ValueError(
            "SIPP Head Start donor predictors and weights must be finite, with "
            "positive weights."
        )
    target = pd.to_numeric(training[_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not (np.isfinite(target) & np.isin(target, [0.0, 1.0])).all():
        raise ValueError("SIPP Head Start donor target must be boolean.")
    if np.unique(target).size != 2:
        raise ValueError("SIPP Head Start donor target must contain both classes.")
    training[_OUTPUT] = target.astype(bool)

    canonical, source_key, eligible = _recipient_predictors(frame)
    global QRF
    if QRF is None:
        from importlib import import_module

        QRF = import_module("populace.fit").QRF
    fitted = QRF(n_estimators=int(n_estimators), seed=int(seed)).fit(
        training,
        predictors=list(SIPP_HEAD_START_MODEL_PREDICTORS),
        targets=[_OUTPUT],
        weights=_DONOR_WEIGHT_COLUMN,
    )
    decisions = np.zeros(len(canonical), dtype=bool)
    if eligible.any():
        predicted = fitted.predict(
            canonical.loc[eligible, list(SIPP_HEAD_START_MODEL_PREDICTORS)]
        )
        if _OUTPUT not in predicted:
            raise ValueError(f"SIPP Head Start QRF prediction missing {_OUTPUT!r}.")
        decisions[eligible] = _coerce_boolean_prediction(predicted[_OUTPUT])
    decision_by_source = pd.Series(decisions, index=canonical["source_key"])
    result = source_key.map(decision_by_source)
    if result.isna().any():
        raise ValueError(
            "SIPP Head Start prediction did not cover every source person."
        )
    return pd.Series(
        result.to_numpy(dtype=bool), index=frame.table("person").index, name=_OUTPUT
    )


def with_us_sipp_head_start_input(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    sipp_donor: pd.DataFrame,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> Frame:
    """Recompute and materialize measured Head Start take-up on a US frame."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US SIPP Head Start input requires the US schema.")
    del time_period  # The immutable donor fixes the observation vintage.
    us_sipp_head_start_stage_spec()
    predicted = impute_us_sipp_head_start(
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


def us_sipp_head_start_summary(frame: Frame) -> dict[str, object]:
    """Return age-domain incidence, provenance, and clone diagnostics."""

    person = frame.table("person")
    if _OUTPUT not in person:
        raise ValueError(f"US SIPP Head Start summary requires {_OUTPUT!r}.")
    raw = pd.to_numeric(person[_OUTPUT], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(raw)
    boolean = finite & np.isin(raw, [0.0, 1.0])
    values = boolean & (raw == 1.0)
    age = pd.to_numeric(person.get("age"), errors="coerce").to_numpy(dtype=np.float64)
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    if not (
        np.isfinite(age).all()
        and np.isfinite(weights).all()
        and (weights >= 0.0).all()
        and weights.sum() > 0.0
    ):
        raise ValueError("US SIPP Head Start summary requires valid age and weights.")
    eligible = (age >= _ELIGIBLE_MIN_AGE) & (age <= _ELIGIBLE_MAX_AGE)
    eligible_weight = float(weights[eligible].sum())

    provenance_missing = bool(
        _PERSON_SOURCE_ID_COLUMN not in person
        or person.get(_PERSON_SOURCE_ID_COLUMN, pd.Series(dtype=object)).isna().any()
    )
    role_invalid = False
    roles: pd.Series | None = None
    if has_support_role_metadata(person, entity="person"):
        try:
            roles = support_role_series(person, entity="person")
        except ValueError:
            role_invalid = True
    clone_groups = clone_mismatches = 0
    if not provenance_missing:
        source_ids = _decoded_strings(person[_PERSON_SOURCE_ID_COLUMN])
        if role_invalid:
            keys = source_ids.astype(object)
        else:
            keys, _ = _support_group_keys(person, source_ids)
        work = pd.DataFrame({"key": keys, "value": values})
        sizes = work.groupby("key", sort=False).size()
        clone_groups = int((sizes > 1).sum())
        clone_mismatches = int(
            (work.groupby("key", sort=False)["value"].nunique() > 1).sum()
        )

    summary: dict[str, object] = {
        "missing_count": int(np.count_nonzero(~finite)),
        "invalid_count": int(np.count_nonzero(finite & ~boolean)),
        "unique_count": int(np.unique(raw[boolean]).size),
        "positive_count": int(np.count_nonzero(values)),
        "eligible_count": int(np.count_nonzero(eligible)),
        "eligible_weight": eligible_weight,
        "eligible_weighted_take_up_share": (
            float(weights[eligible & values].sum()) / eligible_weight
            if eligible_weight > 0.0
            else 0.0
        ),
        "eligible_weighted_take_up_share_band": list(_ELIGIBLE_TAKE_UP_SHARE_BAND),
        "out_of_domain_positive_count": int(np.count_nonzero(~eligible & values)),
        "support_provenance_missing": provenance_missing,
        "support_channel_invalid": role_invalid,
        "clone_group_count": clone_groups,
        "clone_mismatch_count": clone_mismatches,
    }
    if roles is not None:
        channel_shares: dict[str, float] = {}
        for name in sorted(roles.unique()):
            mask = roles.eq(name).to_numpy() & eligible
            denominator = float(weights[mask].sum())
            channel_shares[name] = (
                float(weights[mask & values].sum()) / denominator
                if denominator > 0.0
                else 0.0
            )
        summary["channel_eligible_weighted_take_up_shares"] = channel_shares
    return summary


def us_sipp_head_start_signal_gate(frame: Frame) -> GateResult:
    """Require a nondefault, in-domain, source-clone-consistent take-up flag."""

    person = frame.table("person")
    if _OUTPUT not in person:
        return GateResult(
            name="sipp_head_start_signal",
            passed=False,
            failures=(f"person.{_OUTPUT}: missing",),
            details={"missing": [_OUTPUT]},
        )
    try:
        summary = us_sipp_head_start_summary(frame)
    except (TypeError, ValueError) as exc:
        return GateResult(
            name="sipp_head_start_signal",
            passed=False,
            failures=(str(exc),),
            details={},
        )
    failures: list[str] = []
    if int(summary["missing_count"]):
        failures.append(f"{_OUTPUT}: missing values")
    if int(summary["invalid_count"]):
        failures.append(f"{_OUTPUT}: non-boolean values")
    if int(summary["unique_count"]) < 2:
        failures.append(f"{_OUTPUT}: constant column")
    if int(summary["eligible_count"]) == 0 or float(summary["eligible_weight"]) <= 0:
        failures.append(f"{_OUTPUT}: no weighted age-3--5 domain")
    share = float(summary["eligible_weighted_take_up_share"])
    low, high = _ELIGIBLE_TAKE_UP_SHARE_BAND
    if not low <= share <= high:
        failures.append(
            f"{_OUTPUT}: eligible weighted take-up share {share:.6f} outside "
            f"[{low:.3f}, {high:.3f}]"
        )
    if int(summary["out_of_domain_positive_count"]):
        failures.append(f"{_OUTPUT}: positive values outside the age-3--5 domain")
    if bool(summary["support_provenance_missing"]):
        failures.append(f"{_OUTPUT}: person_source_id provenance is missing")
    if bool(summary["support_channel_invalid"]):
        failures.append(f"{_OUTPUT}: support-channel provenance is invalid")
    if int(summary["clone_mismatch_count"]):
        failures.append(
            f"{_OUTPUT}: {summary['clone_mismatch_count']} support-clone mismatch(es)"
        )
    return GateResult(
        name="sipp_head_start_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
