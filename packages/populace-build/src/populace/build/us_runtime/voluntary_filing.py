"""Measured SIPP filing propensity for the voluntary-filer input.

The retired enhanced-CPS pipeline seeded
``would_file_taxes_voluntarily`` from a synthetic demographic rate table.  Its
three dimensions were tax-unit wages, children, and head age.  The full 2023
SIPP donor already pinned for the vehicle stage contains directly reported
federal filing and expected-filing answers, so this stage replaces the
synthetic rate draw with measured survey signal while preserving those three
dimensions.

The source transform is deliberately strict.  It keeps December responses
only when ``AFILING == 1`` (the filing answer is reported) and, for people who
have not filed, ``AWILLFILE == 1`` (the expected-filing answer is reported).
The target is ``EFILING == 1 or EWILLFILE == 1``.  Respondents reported as
claimed dependents (``EDEPCLM == 1``) are not standalone tax units.  Reciprocal
``EPNSPOUSE`` pairs are collapsed once, and disagreeing spouse targets fail
closed.  Unit earnings are the annualized sum of the seven SIPP monthly job-
earnings recodes; age and sex come from the minimum-PNUM reference member;
marriage is the reciprocal-pair fact; and under-18 count is household context
computed before response filtering.  No parent pointer is used to invent a
federal dependent attachment.

A weighted QRF is fit at source-tax-unit grain.  On an expanded support frame,
the ASEC member of each ``tax_unit_source_id`` supplies predictors, the model
draws once, and the result fans out to every clone.  Re-running with the same
donor and build seed recomputes the same surface rather than trusting an
arbitrary pre-existing nonconstant column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.build.us_runtime.support_provenance import (
    has_support_role_metadata,
    support_role_series,
)
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

__all__ = [
    "SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION",
    "SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256",
    "SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES",
    "SIPP_2023_VOLUNTARY_FILING_DONOR_URL",
    "SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS",
    "SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS",
    "US_VOLUNTARY_FILING_NONCONSTANT_TAX_UNIT_COLUMNS",
    "US_VOLUNTARY_FILING_OUTPUT_COLUMNS",
    "US_VOLUNTARY_FILING_STAGE_NAME",
    "VOLUNTARY_FILING_ARCHIVED_DERIVATION_URL",
    "VOLUNTARY_FILING_ARCHIVED_PARAMETERS_URL",
    "VOLUNTARY_FILING_SIPP_DICTIONARY_URL",
    "fetch_sipp_2023_voluntary_filing_donor",
    "impute_us_voluntary_filing",
    "load_sipp_2023_voluntary_filing_donor",
    "us_voluntary_filing_signal_gate",
    "us_voluntary_filing_stage_spec",
    "us_voluntary_filing_summary",
    "with_us_voluntary_filing_input",
]

QRF: Any | None = None

_ARCHIVED_COMMIT = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
_RETIRED_REPOSITORY = "policyengine-" + "us-data"
_RETIRED_PACKAGE = "policyengine_" + "us_data"
_ARCHIVED_ROOT = (
    f"https://github.com/PolicyEngine/{_RETIRED_REPOSITORY}/blob/"
    f"{_ARCHIVED_COMMIT}/{_RETIRED_PACKAGE}/"
)
VOLUNTARY_FILING_ARCHIVED_DERIVATION_URL = (
    _ARCHIVED_ROOT + "datasets/cps/cps.py#L726-L747"
)
VOLUNTARY_FILING_ARCHIVED_PARAMETERS_URL = (
    _ARCHIVED_ROOT + "parameters/take_up/voluntary_filing.yaml#L1-L43"
)
VOLUNTARY_FILING_SIPP_DICTIONARY_URL = (
    "https://www2.census.gov/programs-surveys/sipp/tech-documentation/"
    "data-dictionaries/2023/2023_SIPP_Data_Dictionary.pdf"
)

SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION = "21280dca5995e978d706740a8a4b9b7860cfd7b6"
SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256 = (
    "5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2"
)
SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES = 3_726_010_471
SIPP_2023_VOLUNTARY_FILING_DONOR_URL = (
    f"https://huggingface.co/policyengine/{_RETIRED_REPOSITORY}/resolve/"
    f"{SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION}/pu2023.csv"
)

US_VOLUNTARY_FILING_STAGE_NAME = "voluntary_filing_input"
US_VOLUNTARY_FILING_OUTPUT_COLUMNS: tuple[str, ...] = ("would_file_taxes_voluntarily",)
US_VOLUNTARY_FILING_NONCONSTANT_TAX_UNIT_COLUMNS = US_VOLUNTARY_FILING_OUTPUT_COLUMNS

_JOB_MONTHLY_EARNINGS_COLUMNS = tuple(f"TJB{i}_MSUM" for i in range(1, 8))
SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS: tuple[str, ...] = (
    "SSUID",
    "PNUM",
    "MONTHCODE",
    "WPFINWGT",
    "TAGE",
    "ESEX",
    "EPNSPOUSE",
    "AFILING",
    "EFILING",
    "AWILLFILE",
    "EWILLFILE",
    "EDEPCLM",
    *_JOB_MONTHLY_EARNINGS_COLUMNS,
)
SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS: tuple[str, ...] = (
    "employment_income",
    "reference_age",
    "reference_is_female",
    "reference_is_married",
    "count_under_18",
)

_OUTPUT = US_VOLUNTARY_FILING_OUTPUT_COLUMNS[0]
_DONOR_FILENAME = "pu2023.csv"
_DONOR_WEIGHT_COLUMN = "tax_unit_weight"
_DONOR_SOURCE_KEY_COLUMN = "source_tax_unit_key"
_DEFAULT_N_ESTIMATORS = 100
_BASE_ASEC_SUPPORT_CHANNEL = "asec"
_TAX_UNIT_SOURCE_ID_COLUMN = "tax_unit_source_id"
_TRUE_SHARE_BAND = (0.45, 0.95)
_EXPECTED_READ_PARAMETERS: dict[str, object] = {
    "table": "sipp_person",
    "delimiter": "|",
    "month_column": "MONTHCODE",
    "month": 12,
    "source_columns": list(SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS),
}
_EXPECTED_FIT_PARAMETERS: dict[str, object] = {
    "predictors": list(SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS),
    "target": _OUTPUT,
    "weight": _DONOR_WEIGHT_COLUMN,
    "response_filter": (
        "AFILING == 1 and (EFILING == 1 or (EFILING == 2 and AWILLFILE == 1))"
    ),
    "dependent_exclusion": "EDEPCLM == 1",
    "canonical_unit": (
        "SSUID plus the sorted PNUM/EPNSPOUSE pair for reciprocal spouses; "
        "otherwise PNUM"
    ),
    "n_estimators": _DEFAULT_N_ESTIMATORS,
    "seed_from_build_config": True,
}


def us_voluntary_filing_stage_spec() -> SourceStageSpec:
    """Load and validate the packaged voluntary-filing stage declaration."""

    from importlib.resources import files

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_VOLUNTARY_FILING_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_VOLUNTARY_FILING_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_VOLUNTARY_FILING_STAGE_NAME]
    if spec.grain != "tax_unit":
        raise ValueError("US voluntary-filing stage must have tax_unit grain.")
    if tuple(spec.outputs) != US_VOLUNTARY_FILING_OUTPUT_COLUMNS:
        raise ValueError(
            "US voluntary-filing manifest outputs do not match the runtime-owned "
            "family."
        )
    if [operation.kind for operation in spec.operations] != [
        "read_table",
        "fit_weighted_qrf",
    ]:
        raise ValueError(
            "US voluntary-filing stage must contain read_table then fit_weighted_qrf."
        )
    if dict(spec.operations[0].parameters) != _EXPECTED_READ_PARAMETERS:
        raise ValueError(
            "US voluntary-filing read_table contract drifted from the pinned "
            "SIPP transform."
        )
    if dict(spec.operations[1].parameters) != _EXPECTED_FIT_PARAMETERS:
        raise ValueError(
            "US voluntary-filing QRF contract drifted from the reviewed source "
            "semantics."
        )
    matching_artifacts = [
        artifact
        for artifact in spec.artifacts
        if artifact.get("sha256") == SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256
        and artifact.get("size_bytes") == SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES
    ]
    if not matching_artifacts:
        raise ValueError(
            "US voluntary-filing stage does not pin the full 2023 SIPP donor "
            "SHA-256 and byte length."
        )
    return spec


def _sha256_stream(stream: BinaryIO, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(chunk_size), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _file_matches(
    path: Path,
    *,
    expected_sha256: str | None,
    expected_size_bytes: int | None,
) -> bool:
    if not path.is_file():
        return False
    if expected_size_bytes is not None and path.stat().st_size != expected_size_bytes:
        return False
    return expected_sha256 is None or _sha256_file(path) == expected_sha256


def fetch_sipp_2023_voluntary_filing_donor(
    cache_dir: str | Path | None = None,
    *,
    expected_sha256: str | None = SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256,
    expected_size_bytes: int | None = SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Stream, verify, and atomically cache the pinned 3.73 GB SIPP file."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")

    import hashlib
    import urllib.request

    root = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path.home() / ".cache" / "populace" / "sipp"
    )
    if cache_dir is None:
        snapshot = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / f"models--policyengine--{_RETIRED_REPOSITORY}"
            / "snapshots"
            / SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION
            / _DONOR_FILENAME
        )
        if _file_matches(
            snapshot,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        ):
            return snapshot

    root.mkdir(parents=True, exist_ok=True)
    target = root / _DONOR_FILENAME
    if _file_matches(
        target,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    ):
        return target

    partial = target.with_name(f"{target.name}.part")
    digest = hashlib.sha256()
    written = 0
    try:
        with (
            urllib.request.urlopen(SIPP_2023_VOLUNTARY_FILING_DONOR_URL) as response,  # noqa: S310
            partial.open("wb") as output,
        ):
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                written += len(chunk)

        if expected_size_bytes is not None and written != expected_size_bytes:
            raise ValueError(
                "SIPP 2023 voluntary-filing donor failed byte-length "
                f"verification: expected {expected_size_bytes}, got {written}."
            )
        actual_sha256 = digest.hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(
                "SIPP 2023 voluntary-filing donor failed sha-256 verification: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return target


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _source_unit_keys(december: pd.DataFrame) -> pd.Series:
    """Return reciprocal-spouse pair keys, leaving every other person singleton."""

    if december.duplicated(["SSUID", "PNUM"]).any():
        raise ValueError("SIPP December donor contains duplicate SSUID/PNUM rows.")
    spouse_by_person = pd.Series(
        december["EPNSPOUSE"].to_numpy(dtype=np.float64),
        index=pd.MultiIndex.from_frame(december[["SSUID", "PNUM"]]),
    )
    keys: list[str] = []
    for ssuid, person_number, spouse_number in december[
        ["SSUID", "PNUM", "EPNSPOUSE"]
    ].itertuples(index=False, name=None):
        reciprocal = False
        if np.isfinite(spouse_number):
            partner_key = (ssuid, spouse_number)
            if partner_key in spouse_by_person.index:
                reciprocal = bool(spouse_by_person.loc[partner_key] == person_number)
        if reciprocal:
            low = min(float(person_number), float(spouse_number))
            high = max(float(person_number), float(spouse_number))
        else:
            low = high = float(person_number)
        keys.append(f"{ssuid}:{low:g}:{high:g}")
    return pd.Series(keys, index=december.index, dtype="string")


def load_sipp_2023_voluntary_filing_donor(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Transform the pinned SIPP person file to measured filing tax units."""

    path = Path(path)
    if expected_size_bytes is not None and path.stat().st_size != expected_size_bytes:
        raise ValueError(
            "SIPP 2023 voluntary-filing donor failed byte-length verification: "
            f"expected {expected_size_bytes}, got {path.stat().st_size}."
        )
    if expected_sha256 is not None:
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "SIPP 2023 voluntary-filing donor failed sha-256 verification: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )
    if chunksize < 1:
        raise ValueError("chunksize must be a positive integer")

    header = pd.read_csv(path, delimiter="|", nrows=0)
    missing = sorted(set(SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(
            f"SIPP 2023 voluntary-filing donor missing column(s): {missing}."
        )

    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        delimiter="|",
        usecols=list(SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS),
        chunksize=int(chunksize),
        low_memory=False,
    )
    for chunk in reader:
        month = _numeric(chunk["MONTHCODE"])
        december = chunk.loc[month.eq(12)].copy()
        if not december.empty:
            parts.append(december)
    if not parts:
        raise ValueError("SIPP 2023 voluntary-filing donor has no December rows.")
    december = pd.concat(parts, ignore_index=True)

    numeric_columns = [
        column for column in SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS if column != "SSUID"
    ]
    for column in numeric_columns:
        original = december[column]
        converted = _numeric(original)
        invalid = original.notna() & converted.isna()
        if invalid.any():
            rows = np.flatnonzero(invalid.to_numpy())[:5].tolist()
            raise ValueError(
                f"SIPP voluntary-filing source {column!r} contains nonnumeric "
                f"value(s) at row(s) {rows}."
            )
        december[column] = converted

    if december[["SSUID", "PNUM"]].isna().any(axis=None):
        raise ValueError("SIPP voluntary-filing December source has missing IDs.")
    december[_DONOR_SOURCE_KEY_COLUMN] = _source_unit_keys(december)
    december["_is_under_18"] = december["TAGE"] < 18
    under_18_by_household = december.groupby("SSUID", sort=True)["_is_under_18"].sum()
    monthly_earnings = (
        december.loc[:, list(_JOB_MONTHLY_EARNINGS_COLUMNS)].fillna(0.0).sum(axis=1)
    )
    invalid_earnings = ~np.isfinite(monthly_earnings) | monthly_earnings.lt(0.0)
    if invalid_earnings.any():
        rows = np.flatnonzero(invalid_earnings.to_numpy())[:5].tolist()
        raise ValueError(
            "SIPP voluntary-filing monthly job earnings must be finite and "
            f"nonnegative; invalid row(s) {rows}."
        )
    december["_annual_employment_income"] = monthly_earnings * 12.0

    filing = december["EFILING"]
    reported = december["AFILING"].eq(1) & (
        filing.eq(1) | (filing.eq(2) & december["AWILLFILE"].eq(1))
    )
    reported_target = december["EFILING"].eq(1) | december["EWILLFILE"].eq(1)
    dependent_reported = reported & december["EDEPCLM"].eq(1)
    nondependent = ~december["EDEPCLM"].eq(1)
    response = december.loc[reported & nondependent].copy()
    if response.empty:
        raise ValueError(
            "SIPP voluntary-filing donor has no directly reported, "
            "nondependent filing responses."
        )
    response[_OUTPUT] = reported_target.loc[response.index]

    disagreement = response.groupby(_DONOR_SOURCE_KEY_COLUMN, sort=True)[
        _OUTPUT
    ].nunique()
    disagreeing_keys = disagreement.index[disagreement > 1].tolist()
    if disagreeing_keys:
        raise ValueError(
            "SIPP voluntary-filing reciprocal spouses disagree on the filing "
            f"target for source unit(s) {disagreeing_keys[:5]}."
        )

    ordered = december.sort_values([_DONOR_SOURCE_KEY_COLUMN, "PNUM"], kind="stable")
    source_units = ordered.groupby(_DONOR_SOURCE_KEY_COLUMN, sort=True)
    # Reference attributes and weight come from the minimum-PNUM member of the
    # complete canonical unit, before response filtering. The full reciprocal
    # pair likewise supplies unit earnings; only the filing target is limited
    # to directly reported, nondependent responses.
    reference = ordered.drop_duplicates(_DONOR_SOURCE_KEY_COLUMN).set_index(
        _DONOR_SOURCE_KEY_COLUMN
    )
    employment = source_units["_annual_employment_income"].sum()
    target = response.groupby(_DONOR_SOURCE_KEY_COLUMN, sort=True)[_OUTPUT].first()

    donor = pd.DataFrame(index=target.index)
    donor[_DONOR_SOURCE_KEY_COLUMN] = donor.index.astype(str)
    donor["employment_income"] = employment.reindex(donor.index).to_numpy(
        dtype=np.float64
    )
    donor["reference_age"] = reference.reindex(donor.index)["TAGE"].to_numpy(
        dtype=np.float64
    )
    donor["reference_is_female"] = (
        reference.reindex(donor.index)["ESEX"].eq(2).to_numpy(dtype=np.float64)
    )
    donor["reference_is_married"] = (
        source_units.size().reindex(donor.index).gt(1).to_numpy(dtype=np.float64)
    )
    donor["count_under_18"] = (
        reference.reindex(donor.index)["SSUID"]
        .map(under_18_by_household)
        .to_numpy(dtype=np.float64)
    )
    donor[_OUTPUT] = target.to_numpy(dtype=bool)
    donor[_DONOR_WEIGHT_COLUMN] = reference.reindex(donor.index)["WPFINWGT"].to_numpy(
        dtype=np.float64
    )
    donor = donor.reset_index(drop=True)

    age = donor["reference_age"].to_numpy(dtype=np.float64)
    if not (np.isfinite(age) & (age >= 15.0) & (age <= 120.0)).all():
        raise ValueError(
            "SIPP voluntary-filing reference ages must be finite and in [15, 120]."
        )
    reference_sex = reference.reindex(target.index)["ESEX"].to_numpy(dtype=np.float64)
    if not np.isin(reference_sex, [1.0, 2.0]).all():
        raise ValueError("SIPP voluntary-filing reference sex must be coded 1 or 2.")
    for column in SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS:
        values = donor[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(
                f"SIPP voluntary-filing predictor {column!r} must be finite "
                "and nonnegative."
            )

    preweight_units = len(donor)
    preweight_true = int(donor[_OUTPUT].sum())
    weights = donor[_DONOR_WEIGHT_COLUMN].to_numpy(dtype=np.float64)
    valid_weight = np.isfinite(weights) & (weights > 0.0)
    donor = donor.loc[valid_weight].copy().reset_index(drop=True)
    if donor.empty:
        raise ValueError(
            "SIPP voluntary-filing donor has no positive finite tax-unit weights."
        )
    if donor[_OUTPUT].nunique() < 2:
        raise ValueError(
            "SIPP voluntary-filing donor target is constant after source filtering."
        )
    final_weights = donor[_DONOR_WEIGHT_COLUMN].to_numpy(dtype=np.float64)
    final_target = donor[_OUTPUT].to_numpy(dtype=bool)
    donor.attrs["source_audit"] = {
        "december_rows": int(len(december)),
        "observed_response_rows": int(reported.sum()),
        "observed_response_true_rows": int((reported & reported_target).sum()),
        "claimed_dependent_observed_rows": int(dependent_reported.sum()),
        "claimed_dependent_observed_true_rows": int(
            (dependent_reported & reported_target).sum()
        ),
        "spouse_target_disagreement_units": int(len(disagreeing_keys)),
        "canonical_preweight_units": int(preweight_units),
        "canonical_preweight_true_units": int(preweight_true),
        "positive_finite_weight_units": int(len(donor)),
        "positive_finite_weight_true_units": int(final_target.sum()),
        "positive_finite_weight_sum": float(final_weights.sum()),
        "weighted_true_share": float(
            final_weights[final_target].sum() / final_weights.sum()
        ),
    }
    return donor


def _decoded_strings(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: (
            value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        )
    )


def _strict_recipient_numeric(
    person: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    label: str,
) -> np.ndarray:
    for column in columns:
        if column in person.columns:
            values = pd.to_numeric(person[column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"US voluntary-filing receiver {label} source {column!r} "
                    "contains nonfinite values."
                )
            return values
    raise ValueError(
        f"US voluntary-filing receiver requires one of {list(columns)} for {label}."
    )


def _recipient_tax_unit_predictor_table(frame: Frame) -> pd.DataFrame:
    """Build the approved five predictors in receiver tax-unit order."""

    person = frame.table("person")
    tax_unit = frame.table("tax_unit")
    required = {
        "person_id",
        "person_tax_unit_id",
        "person_household_id",
        "age",
        "is_female",
    }
    missing = sorted(required - set(person.columns))
    if missing:
        raise ValueError(
            f"US voluntary-filing receiver missing person column(s): {missing}."
        )
    if "tax_unit_id" not in tax_unit:
        raise ValueError("US voluntary-filing receiver requires tax_unit_id.")

    age = _strict_recipient_numeric(person, ("age", "A_AGE"), label="age")
    if not ((age >= 0.0) & (age <= 120.0)).all():
        raise ValueError("US voluntary-filing receiver age must be in [0, 120].")
    income = _strict_recipient_numeric(
        person,
        ("employment_income_before_lsr", "employment_income", "WSAL_VAL"),
        label="employment income",
    )
    if (income < 0.0).any():
        raise ValueError("US voluntary-filing receiver wages must be nonnegative.")
    female_numeric = pd.to_numeric(person["is_female"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not (np.isfinite(female_numeric) & np.isin(female_numeric, [0.0, 1.0])).all():
        raise ValueError("US voluntary-filing receiver is_female must be boolean.")

    work = pd.DataFrame(
        {
            "person_id": person["person_id"].to_numpy(),
            "tax_unit_id": person["person_tax_unit_id"].to_numpy(),
            "household_id": person["person_household_id"].to_numpy(),
            "age": age,
            "is_female": female_numeric,
            "employment_income": income,
        },
        index=person.index,
    )
    if "tax_unit_role_input" not in person:
        raise ValueError(
            "US voluntary-filing receiver requires tax_unit_role_input to "
            "identify exactly one tax-unit head."
        )
    roles = _decoded_strings(person["tax_unit_role_input"])
    head = roles.eq("HEAD")
    head_counts = head.groupby(person["person_tax_unit_id"], sort=False).sum()
    if not head_counts.eq(1).all():
        bad = head_counts.index[~head_counts.eq(1)].tolist()
        raise ValueError(
            "US voluntary-filing receiver requires exactly one HEAD per tax "
            f"unit; invalid unit(s) {bad[:5]}."
        )
    work["_head_rank"] = (~head).astype(np.int8)
    if "A_LINENO" in person:
        work["_line"] = pd.to_numeric(person["A_LINENO"], errors="coerce").fillna(
            np.inf
        )
    else:
        work["_line"] = work["person_id"]

    unit_household_counts = work.groupby("tax_unit_id", sort=False)[
        "household_id"
    ].nunique()
    if (unit_household_counts != 1).any():
        bad = unit_household_counts.index[unit_household_counts != 1].tolist()
        raise ValueError(
            f"US voluntary-filing receiver tax units span households: {bad[:5]}."
        )
    household_under_18 = (
        pd.DataFrame(
            {
                "household_id": work["household_id"],
                "under_18": work["age"].lt(18),
            }
        )
        .groupby("household_id", sort=False)["under_18"]
        .sum()
    )
    grouped = work.groupby("tax_unit_id", sort=False)
    unit_income = grouped["employment_income"].sum()
    unit_household = grouped["household_id"].first()
    reference = (
        work.sort_values(
            ["tax_unit_id", "_head_rank", "_line", "person_id"], kind="stable"
        )
        .drop_duplicates("tax_unit_id")
        .set_index("tax_unit_id")
    )

    filing_status_column = next(
        (
            column
            for column in ("filing_status_input", "filing_status")
            if column in tax_unit
        ),
        None,
    )
    if filing_status_column is None:
        raise ValueError(
            "US voluntary-filing receiver requires tax-unit filing_status_input."
        )
    filing_status = _decoded_strings(tax_unit[filing_status_column])
    married = pd.Series(
        filing_status.eq("JOINT").to_numpy(),
        index=tax_unit["tax_unit_id"].to_numpy(),
    )

    unit_ids = tax_unit["tax_unit_id"].to_numpy()
    receiver = pd.DataFrame(index=unit_ids)
    receiver["employment_income"] = unit_income.reindex(unit_ids).to_numpy(
        dtype=np.float64
    )
    receiver["reference_age"] = reference.reindex(unit_ids)["age"].to_numpy(
        dtype=np.float64
    )
    receiver["reference_is_female"] = reference.reindex(unit_ids)["is_female"].to_numpy(
        dtype=np.float64
    )
    receiver["reference_is_married"] = (
        married.reindex(unit_ids).fillna(False).to_numpy(dtype=np.float64)
    )
    receiver["count_under_18"] = (
        unit_household.reindex(unit_ids)
        .map(household_under_18)
        .to_numpy(dtype=np.float64)
    )
    if receiver.isna().any(axis=None):
        bad = receiver.index[receiver.isna().any(axis=1)].tolist()
        raise ValueError(
            f"US voluntary-filing receiver does not cover tax unit(s) {bad[:5]}."
        )
    return receiver.loc[:, list(SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS)]


def _source_receiver_rows(
    frame: Frame,
    receiver: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select one ASEC predictor row per source unit and return fan-out keys."""

    tax_unit = frame.table("tax_unit")
    if _TAX_UNIT_SOURCE_ID_COLUMN in tax_unit:
        source_ids = tax_unit[_TAX_UNIT_SOURCE_ID_COLUMN]
        if source_ids.isna().any():
            raise ValueError(
                "US voluntary-filing receiver has missing tax_unit_source_id."
            )
    else:
        if has_support_role_metadata(tax_unit, entity="tax_unit"):
            raise ValueError(
                "US voluntary-filing support clones require tax_unit_source_id."
            )
        source_ids = tax_unit["tax_unit_id"]

    rows = receiver.copy()
    rows["_source_id"] = _decoded_strings(source_ids).to_numpy()
    rows["_tax_unit_id"] = tax_unit["tax_unit_id"].to_numpy()
    if has_support_role_metadata(tax_unit, entity="tax_unit"):
        rows["_support_role"] = support_role_series(
            tax_unit, entity="tax_unit"
        ).to_numpy()
        role_counts = rows.groupby(
            ["_source_id", "_support_role"],
            sort=False,
        ).size()
        duplicated_roles = role_counts[role_counts > 1]
        if not duplicated_roles.empty:
            bad = duplicated_roles.index.tolist()
            raise ValueError(
                "US voluntary-filing support source units carry duplicated "
                f"same-role rows; invalid source role(s) {bad[:5]}."
            )
        rows["_source_key"] = rows["_source_id"]
        asec_counts = (
            rows["_support_role"]
            .eq(_BASE_ASEC_SUPPORT_CHANNEL)
            .groupby(rows["_source_key"])
            .sum()
        )
        if asec_counts.gt(1).any():
            bad = asec_counts.index[asec_counts.gt(1)].tolist()
            raise ValueError(
                "US voluntary-filing support source units carry duplicated "
                f"ASEC rows; invalid source unit(s) {bad[:5]}."
            )
        # Prefer each unit's ASEC row, but a frozen-support selection may
        # legitimately keep only a unit's PUF clone (the L0-survivor case the
        # SSI reporter lineage also handles — Build M's certified 57,240
        # selection does exactly this). Clones carry the unit's source
        # predictors, so the surviving row predicts identically; pick it
        # deterministically by channel then tax-unit id.
        ordered_rows = rows.copy()
        ordered_rows["_asec_rank"] = (
            ~ordered_rows["_support_role"].eq(_BASE_ASEC_SUPPORT_CHANNEL)
        ).astype(int)
        source_rows = (
            ordered_rows.sort_values(
                [
                    "_source_id",
                    "_asec_rank",
                    "_support_role",
                    "_tax_unit_id",
                ],
                kind="stable",
            )
            .drop_duplicates("_source_key", keep="first")
            .drop(columns="_asec_rank")
        )
    else:
        if rows["_source_id"].duplicated().any():
            duplicates = rows.loc[
                rows["_source_id"].duplicated(), "_source_id"
            ].tolist()
            raise ValueError(
                "US voluntary-filing unexpanded receiver has duplicate source "
                f"unit(s) {duplicates[:5]}."
            )
        source_rows = rows
        rows["_source_key"] = rows["_source_id"]
        source_rows["_source_key"] = source_rows["_source_id"]

    source_rows = source_rows.sort_values(["_source_id", "_source_key"], kind="stable")
    prediction_rows = source_rows.loc[:, list(SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS)]
    prediction_rows.index = pd.Index(
        source_rows["_source_key"].tolist(), tupleize_cols=False
    )
    return prediction_rows, rows["_source_key"]


def impute_us_voluntary_filing(
    frame: Frame,
    donor: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> pd.Series:
    """Fit the weighted SIPP QRF and draw once per source tax unit."""

    required = {
        *SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS,
        _OUTPUT,
        _DONOR_WEIGHT_COLUMN,
    }
    missing = sorted(required - set(donor.columns))
    if missing:
        raise ValueError(
            f"SIPP voluntary-filing donor table missing column(s): {missing}."
        )
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")

    training = (
        donor.loc[:, [*SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS, _OUTPUT]]
        .copy()
        .reset_index(drop=True)
    )
    for column in SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS:
        training[column] = pd.to_numeric(training[column], errors="coerce")
        values = training[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(
                f"SIPP voluntary-filing donor predictor {column!r} must be "
                "finite and nonnegative."
            )
    target = pd.to_numeric(training[_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not (np.isfinite(target) & np.isin(target, [0.0, 1.0])).all():
        raise ValueError("SIPP voluntary-filing donor target must be boolean.")
    if np.unique(target).size < 2:
        raise ValueError("SIPP voluntary-filing donor target must be nonconstant.")
    training[_OUTPUT] = target
    weights = pd.to_numeric(donor[_DONOR_WEIGHT_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not (np.isfinite(weights) & (weights > 0.0)).all():
        raise ValueError(
            "SIPP voluntary-filing donor weights must be finite and positive."
        )

    # Sort on source identity when available, otherwise on the complete fit
    # row, so reruns do not depend on incidental input ordering.
    if _DONOR_SOURCE_KEY_COLUMN in donor:
        order = np.argsort(
            donor[_DONOR_SOURCE_KEY_COLUMN].astype(str).to_numpy(), kind="stable"
        )
    else:
        order = (
            training.assign(_weight=weights)
            .sort_values(
                [
                    *SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS,
                    _OUTPUT,
                    "_weight",
                ],
                kind="stable",
            )
            .index.to_numpy(dtype=np.int64)
        )
    training = training.iloc[order].reset_index(drop=True)
    weights = weights[order]

    receiver = _recipient_tax_unit_predictor_table(frame)
    source_receiver, fanout_keys = _source_receiver_rows(frame, receiver)
    global QRF
    if QRF is None:
        from importlib import import_module

        QRF = import_module("populace.fit").QRF
    fitted = QRF(n_estimators=int(n_estimators), seed=int(seed)).fit(
        training,
        predictors=list(SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS),
        targets=[_OUTPUT],
        weights=weights,
    )
    prediction = fitted.predict(source_receiver)
    if _OUTPUT not in prediction:
        raise ValueError(
            f"SIPP voluntary-filing QRF prediction is missing {_OUTPUT!r}."
        )
    values = pd.to_numeric(prediction[_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("SIPP voluntary-filing QRF produced nonfinite values.")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("SIPP voluntary-filing QRF produced values outside [0, 1].")
    by_source = pd.Series(values >= 0.5, index=source_receiver.index)
    fanout = fanout_keys.map(by_source)
    if fanout.isna().any():
        raise ValueError("SIPP voluntary-filing source-unit fan-out is incomplete.")
    return pd.Series(
        fanout.to_numpy(dtype=bool),
        index=frame.table("tax_unit").index,
        name=_OUTPUT,
    )


def with_us_voluntary_filing_input(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    sipp_donor: pd.DataFrame,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> Frame:
    """Materialize the deterministic, source-backed tax-unit input."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US voluntary-filing input requires the US schema.")
    del time_period  # The measured donor vintage is pinned independently.
    us_voluntary_filing_stage_spec()
    predicted = impute_us_voluntary_filing(
        frame,
        sipp_donor,
        seed=int(seed),
        n_estimators=int(n_estimators),
    )
    tax_unit = frame.table("tax_unit")
    if _OUTPUT in tax_unit:
        current = pd.to_numeric(tax_unit[_OUTPUT], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if (
            np.isfinite(current).all()
            and np.isin(current, [0.0, 1.0]).all()
            and np.array_equal(current.astype(bool), predicted.to_numpy())
        ):
            return frame

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["tax_unit"][_OUTPUT] = predicted.to_numpy(dtype=bool)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def us_voluntary_filing_summary(frame: Frame) -> dict[str, object]:
    """Return weighted incidence, boolean validity, and clone diagnostics."""

    tax_unit = frame.table("tax_unit")
    values = pd.to_numeric(tax_unit[_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    weights = np.asarray(frame.resolve_weights("tax_unit").values, dtype=np.float64)
    valid_weights = np.isfinite(weights) & (weights >= 0.0)
    if not valid_weights.all() or float(weights.sum()) <= 0.0:
        rows = np.flatnonzero(~valid_weights)[:5].tolist()
        raise ValueError(
            "US voluntary-filing gate requires finite nonnegative tax-unit "
            f"weights with positive total; invalid row(s) {rows}."
        )
    finite = np.isfinite(values)
    boolean = finite & np.isin(values, [0.0, 1.0])
    true = boolean & (values == 1.0)
    total_weight = float(weights.sum())

    clone_mismatch_source_units = 0
    clone_source_units = 0
    clone_metadata_missing = False
    channel_diagnostics: dict[str, dict[str, float | int]] = {}
    if has_support_role_metadata(tax_unit, entity="tax_unit"):
        channels = support_role_series(tax_unit, entity="tax_unit")
        for channel in sorted(channels.unique()):
            mask = channels.eq(channel).to_numpy()
            channel_weight = float(weights[mask].sum())
            channel_true = true & mask
            channel_diagnostics[channel] = {
                "unique_count": int(pd.Series(values[mask & finite]).nunique()),
                "weighted_true_share": (
                    float(weights[channel_true].sum()) / channel_weight
                    if channel_weight > 0.0
                    else 0.0
                ),
            }
        if _TAX_UNIT_SOURCE_ID_COLUMN not in tax_unit:
            clone_metadata_missing = True
        else:
            source_ids = tax_unit[_TAX_UNIT_SOURCE_ID_COLUMN]
            if source_ids.isna().any():
                clone_metadata_missing = True
            else:
                clone_table = pd.DataFrame(
                    {
                        "source_id": source_ids.astype(str),
                        "role": channels.to_numpy(),
                        "value": values,
                    }
                )
                clone_table["source_occurrence"] = clone_table.groupby(
                    ["source_id", "role"], sort=False
                ).cumcount()
                clone_groups = ["source_id", "source_occurrence"]
                sizes = clone_table.groupby(clone_groups, sort=False).size()
                clone_source_units = int((sizes > 1).sum())
                unique = clone_table.groupby(clone_groups, sort=False)["value"].nunique(
                    dropna=False
                )
                clone_mismatch_source_units = int((unique > 1).sum())

    return {
        "weighted_true_share": float(weights[true].sum()) / total_weight,
        "weighted_true_total": float(weights[true].sum()),
        "weighted_false_total": float(weights[boolean & ~true].sum()),
        "true_share_band": list(_TRUE_SHARE_BAND),
        "unique_count": int(pd.Series(values[finite]).nunique()),
        "missing_or_nonfinite_count": int((~finite).sum()),
        "invalid_boolean_count": int((finite & ~boolean).sum()),
        "clone_source_units": clone_source_units,
        "clone_mismatch_source_units": clone_mismatch_source_units,
        "clone_metadata_missing": clone_metadata_missing,
        "channel_diagnostics": channel_diagnostics,
    }


def us_voluntary_filing_signal_gate(frame: Frame) -> GateResult:
    """Require nonconstant boolean signal and identical support clones."""

    tax_unit = frame.table("tax_unit")
    if _OUTPUT not in tax_unit:
        return GateResult(
            name="voluntary_filing_signal",
            passed=False,
            failures=(f"tax_unit column missing: {_OUTPUT!r}.",),
            details={"missing": [_OUTPUT]},
        )
    summary = us_voluntary_filing_summary(frame)
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
    share = float(summary["weighted_true_share"])
    low, high = summary["true_share_band"]
    if not (low <= share <= high):
        failures.append(
            f"{_OUTPUT}: weighted true share {share:.4f} outside plausibility "
            f"band [{low}, {high}]."
        )
    if float(summary["weighted_true_total"]) <= 0.0:
        failures.append(f"{_OUTPUT}: weighted true total is not positive.")
    if float(summary["weighted_false_total"]) <= 0.0:
        failures.append(f"{_OUTPUT}: weighted false total is not positive.")
    if summary["clone_metadata_missing"]:
        failures.append(
            "Voluntary-filing support clones lack complete tax_unit_source_id "
            "provenance."
        )
    if summary["clone_mismatch_source_units"]:
        failures.append(
            "Voluntary-filing support clones disagree for "
            f"{summary['clone_mismatch_source_units']} source unit(s)."
        )
    for channel, diagnostics in summary["channel_diagnostics"].items():
        if diagnostics["unique_count"] < 2:
            failures.append(f"{_OUTPUT}: support channel {channel!r} is constant.")
        channel_share = float(diagnostics["weighted_true_share"])
        if not (low <= channel_share <= high):
            failures.append(
                f"{_OUTPUT}: support channel {channel!r} weighted true share "
                f"{channel_share:.4f} outside plausibility band [{low}, {high}]."
            )
    return GateResult(
        name="voluntary_filing_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
