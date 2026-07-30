"""Household vehicle count and value restored from the full 2023 SIPP.

The retired enhanced-CPS pipeline trained a household-grain, weighted model
from December SIPP records.  Vehicle count was the household maximum of
``TVEH_NUM`` and vehicle value was the first household ``THVAL_VEH``.  The
model conditioned on household-summed income, the oldest SIPP member's
demographics, household composition, and homeownership.  On the CPS receiver,
the same income concepts were summed to households while demographics came
from the CPS household head.

This module ports that source layer without folding vehicle value into
``net_worth``.  The retired pipeline did that only later, after a stable
SIPP/SCF source draw and a complete SCF balance-sheet reconciliation.  Adding
vehicle value to Populace's currently excluded/default net-worth surface would
therefore create a false partial measure.

The final archived implementation moved to a mutable Census SIPP 2024 URL.
For a hermetic build, this port uses the archived pipeline's full SIPP 2023
artifact at an immutable Hugging Face revision and verifies both byte length
and SHA-256.  The 65.6 MB ``pu2023_slim.csv`` used by the tip stage cannot be
reused: it omits the vehicle targets, allocation flags, asset-income fields,
and home value required here.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from populace.build.gates import GateResult
from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

__all__ = [
    "ARCHIVED_SIPP_VEHICLE_IMPUTE_URL",
    "ARCHIVED_SIPP_VEHICLE_RECEIVER_URL",
    "ARCHIVED_SIPP_VEHICLE_SOURCE_URL",
    "ARCHIVED_SIPP_VEHICLE_TRANSFORM_URL",
    "SIPP_2023_VEHICLE_DONOR_REVISION",
    "SIPP_2023_VEHICLE_DONOR_SHA256",
    "SIPP_2023_VEHICLE_DONOR_SIZE_BYTES",
    "SIPP_2023_VEHICLE_DONOR_URL",
    "SIPP_VEHICLE_MODEL_PREDICTORS",
    "SIPP_VEHICLE_SOURCE_COLUMNS",
    "US_SIPP_VEHICLE_NONCONSTANT_HOUSEHOLD_COLUMNS",
    "US_SIPP_VEHICLE_OUTPUT_COLUMNS",
    "fetch_sipp_2023_vehicle_donor",
    "impute_us_sipp_vehicles",
    "load_sipp_2023_vehicle_donor",
    "us_sipp_vehicles_signal_gate",
    "us_sipp_vehicles_stage_spec",
    "us_sipp_vehicles_summary",
    "with_us_sipp_vehicle_inputs",
]

_RETIRED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_REPOSITORY = (
    "https://github.com/PolicyEngine/" + _RETIRED_DATA_REPOSITORY
)
_ARCHIVED_COMMIT = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
_RETIRED_PACKAGE_PATH = "policyengine_" + "us_data"
ARCHIVED_SIPP_VEHICLE_SOURCE_URL = (
    f"{_ARCHIVED_REPOSITORY}/blob/{_ARCHIVED_COMMIT}/"
    f"{_RETIRED_PACKAGE_PATH}/datasets/sipp/sipp.py#L430-L456"
)
ARCHIVED_SIPP_VEHICLE_TRANSFORM_URL = (
    f"{_ARCHIVED_REPOSITORY}/blob/{_ARCHIVED_COMMIT}/"
    f"{_RETIRED_PACKAGE_PATH}/datasets/sipp/sipp.py#L953-L1059"
)
ARCHIVED_SIPP_VEHICLE_IMPUTE_URL = (
    f"{_ARCHIVED_REPOSITORY}/blob/{_ARCHIVED_COMMIT}/"
    f"{_RETIRED_PACKAGE_PATH}/calibration/source_impute.py#L992-L1084"
)
ARCHIVED_SIPP_VEHICLE_RECEIVER_URL = (
    f"{_ARCHIVED_REPOSITORY}/blob/{_ARCHIVED_COMMIT}/"
    f"{_RETIRED_PACKAGE_PATH}/utils/asset_imputation.py#L503-L603"
)

SIPP_2023_VEHICLE_DONOR_REVISION = "21280dca5995e978d706740a8a4b9b7860cfd7b6"
SIPP_2023_VEHICLE_DONOR_SHA256 = (
    "5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2"
)
SIPP_2023_VEHICLE_DONOR_SIZE_BYTES = 3_726_010_471
SIPP_2023_VEHICLE_DONOR_URL = (
    "https://huggingface.co/policyengine/"
    f"{_RETIRED_DATA_REPOSITORY}/resolve/"
    f"{SIPP_2023_VEHICLE_DONOR_REVISION}/pu2023.csv"
)

SIPP_VEHICLE_SOURCE_COLUMNS: tuple[str, ...] = (
    "SSUID",
    "PNUM",
    "MONTHCODE",
    "WPFINWGT",
    "TAGE",
    "ESEX",
    "EMS",
    "TPTOTINC",
    "TINC_BANK",
    "TINC_STMF",
    "TINC_BOND",
    "TINC_RENT",
    "TVEH_NUM",
    "THVAL_VEH",
    "THVAL_HOME",
    "AVEH_NUM",
    "AHVAL_VEH",
    "AVEH1VAL",
    "AVEH2VAL",
    "AVEH3VAL",
)

SIPP_VEHICLE_MODEL_PREDICTORS: tuple[str, ...] = (
    "household_employment_income",
    "household_interest_income",
    "household_dividend_income",
    "household_rental_income",
    "reference_age",
    "reference_is_female",
    "reference_is_married",
    "count_under_18",
    "household_size",
    "is_homeowner",
)

US_SIPP_VEHICLE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "household_vehicles_owned",
    "household_vehicles_value",
)
US_SIPP_VEHICLE_NONCONSTANT_HOUSEHOLD_COLUMNS = US_SIPP_VEHICLE_OUTPUT_COLUMNS

_STAGE_NAME = "vehicle_assets"
_DONOR_FILENAME = "pu2023.csv"
_DONOR_WEIGHT_COLUMN = "household_weight"
_OWNED_OBSERVED_COLUMN = "household_vehicles_owned_is_observed"
_VALUE_OBSERVED_COLUMN = "household_vehicles_value_is_observed"
_DEFAULT_N_ESTIMATORS = 100
_ARCHIVED_MODEL_SEED = 42
_MAX_TRAIN_SAMPLES = 20_000
_TRAINING_SAMPLE_SEED_NAME = "calibration_sipp_vehicle_training_sample"
_OBSERVED_SIPP_STATUSES = frozenset((0, 1, 9))
_OWNED_NONZERO_SHARE_BAND = (0.40, 0.99)
_VALUE_NONZERO_SHARE_BAND = (0.25, 0.99)

_REQUIRED_RECIPIENT_PERSON_COLUMNS: tuple[str, ...] = (
    "person_household_id",
    "age",
    "is_female",
    "A_LINENO",
    "employment_income_before_lsr",
    "rental_income",
)

_RECIPIENT_INTEREST_COLUMNS = (
    "taxable_interest_income",
    "tax_exempt_interest_income",
)
_RECIPIENT_DIVIDEND_COLUMNS = (
    "qualified_dividend_income",
    "non_qualified_dividend_income",
)


def us_sipp_vehicles_stage_spec() -> SourceStageSpec:
    """Load and validate the packaged ``vehicle_assets`` stage declaration."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if _STAGE_NAME not in stage_map:
        raise ValueError(f"US source manifest declares no {_STAGE_NAME!r} stage.")
    spec = stage_map[_STAGE_NAME]
    missing = sorted(set(US_SIPP_VEHICLE_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{_STAGE_NAME!r} manifest stage does not declare output(s) "
            f"{missing}; the runtime and manifest have drifted."
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
    if not path.exists() or not path.is_file():
        return False
    if expected_size_bytes is not None and path.stat().st_size != expected_size_bytes:
        return False
    return expected_sha256 is None or _sha256_file(path) == expected_sha256


def fetch_sipp_2023_vehicle_donor(
    cache_dir: str | Path | None = None,
    *,
    expected_sha256: str | None = SIPP_2023_VEHICLE_DONOR_SHA256,
    expected_size_bytes: int | None = SIPP_2023_VEHICLE_DONOR_SIZE_BYTES,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Stream, verify, and atomically cache the pinned full SIPP donor.

    Streaming is mandatory for this 3.73 GB artifact: reading the response or
    file into one bytes object would needlessly double peak memory.
    """

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
            / f"models--policyengine--{_RETIRED_DATA_REPOSITORY}"
            / "snapshots"
            / SIPP_2023_VEHICLE_DONOR_REVISION
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
            urllib.request.urlopen(SIPP_2023_VEHICLE_DONOR_URL) as response,  # noqa: S310
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
                "SIPP 2023 vehicle donor failed byte-length verification: "
                f"expected {expected_size_bytes}, got {written}."
            )
        actual_sha256 = digest.hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(
                "SIPP 2023 vehicle donor failed sha-256 verification: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return target


def _stable_string_seed(value: str) -> int:
    """Match the archived pipeline's uint64 stable-string seed."""

    mask = 2**64 - 1
    hashed = 0
    for byte in value.encode("utf-8"):
        hashed = (hashed * 31 + byte) & mask
    hashed ^= hashed >> 33
    hashed = (hashed * 0xFF51AFD7ED558CCD) & mask
    hashed ^= hashed >> 33
    return hashed % (2**63)


def _sample_rng(seed_name: str, *, salt: str | None = None) -> np.random.Generator:
    key = seed_name if salt is None else f"{seed_name}:{salt}"
    return np.random.default_rng(_stable_string_seed(key))


def _cap_vehicle_training_sample(donor: pd.DataFrame) -> pd.DataFrame:
    """Apply the archived target-balanced 20,000-row positional cap."""

    filters = {
        US_SIPP_VEHICLE_OUTPUT_COLUMNS[0]: donor[_OWNED_OBSERVED_COLUMN].astype(bool),
        US_SIPP_VEHICLE_OUTPUT_COLUMNS[1]: donor[_VALUE_OBSERVED_COLUMN].astype(bool),
    }
    union = filters[US_SIPP_VEHICLE_OUTPUT_COLUMNS[0]] | filters[
        US_SIPP_VEHICLE_OUTPUT_COLUMNS[1]
    ]
    union_positions = np.flatnonzero(union.to_numpy())
    if len(union_positions) <= _MAX_TRAIN_SAMPLES:
        positions = union_positions
    else:
        selected: list[int] = []
        selected_set: set[int] = set()
        per_target_cap = _MAX_TRAIN_SAMPLES // len(filters)
        for target, target_filter in filters.items():
            target_positions = np.flatnonzero(target_filter.to_numpy())
            sampled = _sample_rng(
                _TRAINING_SAMPLE_SEED_NAME, salt=target
            ).choice(
                target_positions,
                size=min(per_target_cap, len(target_positions)),
                replace=False,
            )
            for position in sampled:
                if int(position) not in selected_set:
                    selected.append(int(position))
                    selected_set.add(int(position))

        remaining_n = _MAX_TRAIN_SAMPLES - len(selected)
        remaining = np.asarray(
            [
                position
                for position in union_positions
                if int(position) not in selected_set
            ],
            dtype=int,
        )
        if remaining_n > 0 and len(remaining):
            fill = _sample_rng(_TRAINING_SAMPLE_SEED_NAME, salt="fill").choice(
                remaining,
                size=min(remaining_n, len(remaining)),
                replace=False,
            )
            selected.extend(int(position) for position in fill)
        positions = np.asarray(selected, dtype=int)

    return donor.iloc[positions].copy().reset_index(drop=True)


def load_sipp_2023_vehicle_donor(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = SIPP_2023_VEHICLE_DONOR_SIZE_BYTES,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Load and transform the pinned person-month file to household donors."""

    path = Path(path)
    if expected_size_bytes is not None and path.stat().st_size != expected_size_bytes:
        raise ValueError(
            "SIPP 2023 vehicle donor failed byte-length verification: "
            f"expected {expected_size_bytes}, got {path.stat().st_size}."
        )
    if expected_sha256 is not None:
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "SIPP 2023 vehicle donor failed sha-256 verification: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )
    if chunksize < 1:
        raise ValueError("chunksize must be a positive integer")

    header = pd.read_csv(path, delimiter="|", nrows=0)
    missing = sorted(set(SIPP_VEHICLE_SOURCE_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"SIPP 2023 vehicle donor missing column(s): {missing}.")

    december_parts: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        delimiter="|",
        usecols=list(SIPP_VEHICLE_SOURCE_COLUMNS),
        chunksize=int(chunksize),
        low_memory=False,
    )
    for chunk in reader:
        month = pd.to_numeric(chunk["MONTHCODE"], errors="coerce")
        december = chunk.loc[month.eq(12)].copy()
        if not december.empty:
            december_parts.append(december)
    if not december_parts:
        raise ValueError("SIPP 2023 vehicle donor has no December person records.")
    person = pd.concat(december_parts, ignore_index=True)

    for column in SIPP_VEHICLE_SOURCE_COLUMNS:
        if column != "SSUID":
            person[column] = pd.to_numeric(person[column], errors="coerce")

    person["employment_income"] = person["TPTOTINC"].fillna(0.0) * 12.0
    person["interest_income"] = (
        person["TINC_BANK"].fillna(0.0) + person["TINC_BOND"].fillna(0.0)
    ) * 12.0
    person["dividend_income"] = person["TINC_STMF"].fillna(0.0) * 12.0
    person["rental_income"] = person["TINC_RENT"].fillna(0.0) * 12.0
    person["is_under_18"] = person["TAGE"].fillna(0.0) < 18

    grouped = person.groupby("SSUID", sort=True)
    reference_index = grouped["TAGE"].idxmax()
    if reference_index.isna().any():
        raise ValueError("SIPP vehicle donor has a household with no finite age.")
    reference = (
        person.loc[reference_index.astype(int), ["SSUID", "TAGE", "ESEX", "EMS"]]
        .rename(
            columns={
                "TAGE": "reference_age",
                "ESEX": "reference_sex",
                "EMS": "reference_marital_status",
            }
        )
        .set_index("SSUID")
    )

    donor = pd.DataFrame(
        {
            "household_id": grouped["SSUID"].first(),
            _DONOR_WEIGHT_COLUMN: grouped["WPFINWGT"].first().fillna(0.0),
            "household_employment_income": grouped["employment_income"].sum(),
            "household_interest_income": grouped["interest_income"].sum(),
            "household_dividend_income": grouped["dividend_income"].sum(),
            "household_rental_income": grouped["rental_income"].sum(),
            "count_under_18": grouped["is_under_18"].sum(),
            "household_size": grouped.size(),
            "household_vehicles_owned": grouped["TVEH_NUM"].max().fillna(0.0),
            "household_vehicles_value": grouped["THVAL_VEH"].first().fillna(0.0),
            "AVEH_NUM": grouped["AVEH_NUM"].max().fillna(0.0),
            "AHVAL_VEH": grouped["AHVAL_VEH"].first().fillna(0.0),
            "AVEH1VAL": grouped["AVEH1VAL"].max().fillna(0.0),
            "AVEH2VAL": grouped["AVEH2VAL"].max().fillna(0.0),
            "AVEH3VAL": grouped["AVEH3VAL"].max().fillna(0.0),
            "is_homeowner": (
                grouped["THVAL_HOME"].first().fillna(0.0) > 0
            ).astype(np.float32),
        }
    ).reset_index(drop=True)
    donor = donor.merge(
        reference,
        left_on="household_id",
        right_index=True,
        how="left",
    )
    donor["reference_is_female"] = (
        donor["reference_sex"].fillna(1.0).eq(2)
    ).astype(np.float32)
    donor["reference_is_married"] = (
        donor["reference_marital_status"].fillna(0.0).eq(1)
    ).astype(np.float32)
    donor = donor.drop(
        columns=["reference_sex", "reference_marital_status"],
        errors="ignore",
    ).fillna(0.0)

    donor[_OWNED_OBSERVED_COLUMN] = pd.to_numeric(
        donor["AVEH_NUM"], errors="coerce"
    ).fillna(0).isin(_OBSERVED_SIPP_STATUSES)
    value_observed = pd.Series(True, index=donor.index)
    for column in ("AVEH1VAL", "AVEH2VAL", "AVEH3VAL"):
        value_observed &= pd.to_numeric(donor[column], errors="coerce").fillna(
            0
        ).isin(_OBSERVED_SIPP_STATUSES)
    donor[_VALUE_OBSERVED_COLUMN] = value_observed

    weights = pd.to_numeric(donor[_DONOR_WEIGHT_COLUMN], errors="coerce")
    valid_weight = np.isfinite(weights) & weights.gt(0)
    donor = donor.loc[valid_weight].copy().reset_index(drop=True)
    if donor.empty:
        raise ValueError("SIPP vehicle donor has no positive finite household weights.")
    if not donor[_OWNED_OBSERVED_COLUMN].any():
        raise ValueError("SIPP vehicle donor has no observed vehicle-count rows.")
    if not donor[_VALUE_OBSERVED_COLUMN].any():
        raise ValueError("SIPP vehicle donor has no observed vehicle-value rows.")
    return _cap_vehicle_training_sample(donor)


def _is_numeric_categorical(values: pd.Series) -> bool:
    """Replicate MicroImpute 2.1's low-cardinality numeric test."""

    if not pd.api.types.is_numeric_dtype(values) or values.nunique() >= 10:
        return False
    unique = np.sort(pd.to_numeric(values, errors="coerce").dropna().unique())
    if len(unique) < 2:
        return True
    differences = np.diff(unique)
    return bool(np.allclose(differences, differences[0], rtol=1e-9))


@dataclass(frozen=True)
class _PredictorEncoding:
    numeric_columns: tuple[str, ...]
    categorical_levels: dict[str, tuple[float, ...]]

    @property
    def columns(self) -> tuple[str, ...]:
        encoded: list[str] = list(self.numeric_columns)
        for column, levels in self.categorical_levels.items():
            encoded.extend(_dummy_name(column, level) for level in levels[1:])
        return tuple(encoded)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        for column in self.numeric_columns:
            result[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        for column, levels in self.categorical_levels.items():
            values = pd.to_numeric(frame[column], errors="coerce").fillna(levels[0])
            for level in levels[1:]:
                result[_dummy_name(column, level)] = values.eq(level).astype(np.float64)
        return result.loc[:, list(self.columns)]


def _dummy_name(column: str, level: float) -> str:
    return f"{column}__{float(level):g}"


def _predictor_encoding(donor: pd.DataFrame) -> _PredictorEncoding:
    numeric: list[str] = []
    categorical: dict[str, tuple[float, ...]] = {}
    for column in SIPP_VEHICLE_MODEL_PREDICTORS:
        values = pd.to_numeric(donor[column], errors="coerce").fillna(0.0)
        if _is_numeric_categorical(values):
            categorical[column] = tuple(float(value) for value in np.sort(values.unique()))
        else:
            numeric.append(column)
    return _PredictorEncoding(tuple(numeric), categorical)


def _append_owned_dummies(
    frame: pd.DataFrame,
    owned: pd.Series | np.ndarray,
    *,
    levels: tuple[float, ...],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    result = frame.copy()
    values = np.asarray(owned, dtype=np.float64)
    columns: list[str] = []
    for level in levels[1:]:
        name = _dummy_name("household_vehicles_owned", level)
        result[name] = (values == level).astype(np.float64)
        columns.append(name)
    return result, tuple(columns)


def _household_tenure_status(
    frame: Frame,
    *,
    reference_people: pd.DataFrame,
    household_ids: np.ndarray,
) -> np.ndarray:
    """Align raw CPS ``SPM_TENMORTSTATUS`` to household order."""

    household = frame.table("household")
    person = frame.table("person")
    if "SPM_TENMORTSTATUS" in household.columns:
        return pd.to_numeric(
            household["SPM_TENMORTSTATUS"], errors="coerce"
        ).fillna(3).to_numpy()
    if "SPM_TENMORTSTATUS" in person.columns:
        values = pd.Series(
            pd.to_numeric(
                reference_people["SPM_TENMORTSTATUS"], errors="coerce"
            ).fillna(3).to_numpy(),
            index=reference_people["household_id"].to_numpy(),
        )
        return values.reindex(household_ids).fillna(3).to_numpy()
    if "spm_unit" in frame.entities:
        spm_unit = frame.table("spm_unit")
        if (
            "SPM_TENMORTSTATUS" in spm_unit.columns
            and "spm_unit_id" in spm_unit.columns
            and "person_spm_unit_id" in reference_people.columns
        ):
            by_spm_unit = pd.Series(
                pd.to_numeric(
                    spm_unit["SPM_TENMORTSTATUS"], errors="coerce"
                ).fillna(3).to_numpy(),
                index=spm_unit["spm_unit_id"].to_numpy(),
            )
            by_household = pd.Series(
                reference_people["person_spm_unit_id"].map(by_spm_unit).to_numpy(),
                index=reference_people["household_id"].to_numpy(),
            )
            return by_household.reindex(household_ids).fillna(3).to_numpy()
    raise ValueError(
        "US SIPP vehicle imputation requires raw CPS SPM_TENMORTSTATUS on "
        "household, person, or spm_unit."
    )


def _recipient_income(
    person: pd.DataFrame,
    *,
    aggregate_column: str,
    component_columns: tuple[str, ...],
) -> pd.Series:
    """Use the archived aggregate concept or its current-frame components."""

    if aggregate_column in person.columns:
        return pd.to_numeric(person[aggregate_column], errors="coerce").fillna(0.0)
    present = [column for column in component_columns if column in person.columns]
    if not present:
        raise ValueError(
            "US SIPP vehicle imputation requires either person."
            f"{aggregate_column} or component column(s) {list(component_columns)}."
        )
    return (
        person.loc[:, present]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .sum(axis=1)
    )


def _recipient_is_married(person: pd.DataFrame) -> np.ndarray:
    """Match the archive: existing flag, marital-unit pairing, then CPS code."""

    if "is_married" in person.columns:
        return person["is_married"].astype(bool).to_numpy(dtype=np.float64)
    if "person_marital_unit_id" in person.columns:
        marital_unit_id = person["person_marital_unit_id"]
        counts = marital_unit_id.map(marital_unit_id.value_counts())
        return counts.gt(1).to_numpy(dtype=np.float64)
    if "A_MARITL" in person.columns:
        return (
            pd.to_numeric(person["A_MARITL"], errors="coerce")
            .isin([1, 2])
            .to_numpy(dtype=np.float64)
        )
    return np.zeros(len(person), dtype=np.float64)


def _recipient_household_predictor_table(frame: Frame) -> pd.DataFrame:
    person = frame.table("person")
    household = frame.table("household")
    missing = sorted(set(_REQUIRED_RECIPIENT_PERSON_COLUMNS) - set(person.columns))
    if missing:
        raise ValueError(
            f"US SIPP vehicle imputation requires recipient person column(s): {missing}."
        )
    if "household_id" not in household.columns:
        raise ValueError("US SIPP vehicle imputation requires household.household_id.")

    work = pd.DataFrame(index=person.index)
    work["household_id"] = person["person_household_id"].to_numpy()
    work["age"] = pd.to_numeric(person["age"], errors="coerce").fillna(0.0)
    work["is_female"] = person["is_female"].astype(bool).astype(np.float64)
    work["is_married"] = _recipient_is_married(person)
    work["A_LINENO"] = pd.to_numeric(
        person["A_LINENO"], errors="coerce"
    ).fillna(np.inf)
    if "person_spm_unit_id" in person.columns:
        work["person_spm_unit_id"] = person["person_spm_unit_id"].to_numpy()
    if "SPM_TENMORTSTATUS" in person.columns:
        work["SPM_TENMORTSTATUS"] = person["SPM_TENMORTSTATUS"].to_numpy()

    income = pd.DataFrame(
        {
            "household_id": work["household_id"],
            "household_employment_income": pd.to_numeric(
                person["employment_income_before_lsr"], errors="coerce"
            ).fillna(0.0),
            "household_interest_income": _recipient_income(
                person,
                aggregate_column="interest_income",
                component_columns=_RECIPIENT_INTEREST_COLUMNS,
            ),
            "household_dividend_income": _recipient_income(
                person,
                aggregate_column="dividend_income",
                component_columns=_RECIPIENT_DIVIDEND_COLUMNS,
            ),
            "household_rental_income": pd.to_numeric(
                person["rental_income"], errors="coerce"
            ).fillna(0.0),
            "is_under_18": work["age"].lt(18),
        },
        index=person.index,
    )
    aggregated = (
        income.groupby("household_id", sort=False)
        .agg(
            household_employment_income=("household_employment_income", "sum"),
            household_interest_income=("household_interest_income", "sum"),
            household_dividend_income=("household_dividend_income", "sum"),
            household_rental_income=("household_rental_income", "sum"),
            count_under_18=("is_under_18", "sum"),
            household_size=("household_id", "size"),
        )
        .reset_index()
    )

    # CPS ASEC line 1 is the household reference person. Selecting the minimum
    # line number retains that rule while remaining robust to a missing line 1.
    reference = (
        work.sort_values(["household_id", "A_LINENO"], kind="stable")
        .drop_duplicates("household_id")
        .copy()
    )
    reference = reference.rename(
        columns={
            "age": "reference_age",
            "is_female": "reference_is_female",
            "is_married": "reference_is_married",
        }
    )
    household_ids = household["household_id"].to_numpy()
    tenure_status = _household_tenure_status(
        frame,
        reference_people=reference,
        household_ids=household_ids,
    )
    reference = reference.loc[
        :,
        [
            "household_id",
            "reference_age",
            "reference_is_female",
            "reference_is_married",
        ],
    ]
    receiver = aggregated.merge(reference, on="household_id", how="left")
    receiver = receiver.set_index("household_id").reindex(household_ids)
    if receiver.isna().any(axis=None):
        missing_ids = receiver.index[receiver.isna().any(axis=1)].tolist()
        raise ValueError(
            "SIPP vehicle receiver household alignment failed for household(s) "
            f"{missing_ids[:5]}."
        )
    receiver["is_homeowner"] = np.isin(tenure_status, [1, 2]).astype(np.float64)
    return receiver.loc[:, list(SIPP_VEHICLE_MODEL_PREDICTORS)].astype(np.float64)


def impute_us_sipp_vehicles(
    frame: Frame,
    donor: pd.DataFrame,
    *,
    seed: int = _ARCHIVED_MODEL_SEED,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> pd.DataFrame:
    """Fit count first, then draw value conditional on predicted count."""

    from populace.fit import QRF

    required = {
        *SIPP_VEHICLE_MODEL_PREDICTORS,
        *US_SIPP_VEHICLE_OUTPUT_COLUMNS,
        _DONOR_WEIGHT_COLUMN,
        _OWNED_OBSERVED_COLUMN,
        _VALUE_OBSERVED_COLUMN,
    }
    missing = sorted(required - set(donor.columns))
    if missing:
        raise ValueError(f"SIPP vehicle donor table missing column(s): {missing}.")

    donor = donor.copy().reset_index(drop=True)
    for column in (
        *SIPP_VEHICLE_MODEL_PREDICTORS,
        *US_SIPP_VEHICLE_OUTPUT_COLUMNS,
        _DONOR_WEIGHT_COLUMN,
    ):
        donor[column] = pd.to_numeric(donor[column], errors="coerce").fillna(0.0)
    encoding = _predictor_encoding(donor)
    donor_features = encoding.transform(donor)
    receiver = _recipient_household_predictor_table(frame)
    receiver_features = encoding.transform(receiver)

    owned_mask = donor[_OWNED_OBSERVED_COLUMN].astype(bool).to_numpy()
    value_mask = donor[_VALUE_OBSERVED_COLUMN].astype(bool).to_numpy()
    if not owned_mask.any() or not value_mask.any():
        raise ValueError("SIPP vehicle donor lacks observed rows for one or both targets.")

    owned_target = donor.loc[owned_mask, "household_vehicles_owned"]
    owned_levels = tuple(float(value) for value in np.sort(owned_target.unique()))
    count_model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=int(seed),
    )
    count_model.fit(
        donor_features.loc[owned_mask],
        owned_target,
        sample_weight=donor.loc[owned_mask, _DONOR_WEIGHT_COLUMN].to_numpy(
            dtype=np.float64
        ),
    )
    predicted_owned = np.asarray(
        count_model.predict(receiver_features), dtype=np.float64
    )

    donor_value_features, owned_dummy_columns = _append_owned_dummies(
        donor_features,
        donor["household_vehicles_owned"],
        levels=owned_levels,
    )
    receiver_value_features, _ = _append_owned_dummies(
        receiver_features,
        predicted_owned,
        levels=owned_levels,
    )
    value_predictors = [*encoding.columns, *owned_dummy_columns]
    value_fit = donor_value_features.loc[value_mask, value_predictors].copy()
    value_fit["household_vehicles_value"] = donor.loc[
        value_mask, "household_vehicles_value"
    ].to_numpy(dtype=np.float64)
    value_fit[_DONOR_WEIGHT_COLUMN] = donor.loc[
        value_mask, _DONOR_WEIGHT_COLUMN
    ].to_numpy(dtype=np.float64)
    fitted_value = QRF(n_estimators=int(n_estimators), seed=int(seed)).fit(
        value_fit,
        predictors=value_predictors,
        targets=["household_vehicles_value"],
        weights=_DONOR_WEIGHT_COLUMN,
    )
    predicted_value = np.asarray(
        fitted_value.predict(receiver_value_features.loc[:, value_predictors])[
            "household_vehicles_value"
        ],
        dtype=np.float64,
    )

    return pd.DataFrame(
        {
            "household_vehicles_owned": np.clip(
                np.rint(predicted_owned), 0, None
            ).astype(np.int32),
            "household_vehicles_value": np.clip(
                predicted_value, 0, None
            ).astype(np.float32),
        },
        index=frame.table("household").index,
    )


def _surface_has_signal(household: pd.DataFrame) -> bool:
    if not all(column in household.columns for column in US_SIPP_VEHICLE_OUTPUT_COLUMNS):
        return False
    owned = pd.to_numeric(
        household["household_vehicles_owned"], errors="coerce"
    ).dropna()
    value = pd.to_numeric(
        household["household_vehicles_value"], errors="coerce"
    ).dropna()
    return bool(
        owned.nunique() > 1
        and value.nunique() > 1
        and (owned > 0).any()
        and (value > 0).any()
        and (owned >= 0).all()
        and (value >= 0).all()
        and np.allclose(owned, np.rint(owned))
    )


def with_us_sipp_vehicle_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    sipp_donor: pd.DataFrame,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> Frame:
    """Restore both vehicle inputs on the household table."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US SIPP vehicle inputs require the US schema.")
    del time_period  # Source transformation is fixed by the pinned donor vintage.
    us_sipp_vehicles_stage_spec()
    household = frame.table("household")
    if _surface_has_signal(household):
        return frame

    imputed = impute_us_sipp_vehicles(
        frame,
        sipp_donor,
        seed=int(seed),
        n_estimators=int(n_estimators),
    )
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column in US_SIPP_VEHICLE_OUTPUT_COLUMNS:
        tables["household"][column] = imputed[column].to_numpy()
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_sipp_vehicles_summary(frame: Frame) -> dict[str, object]:
    """Return weighted incidence, amount, type, and relation diagnostics."""

    household = frame.table("household")
    weights = np.asarray(frame.weights_for("household").values, dtype=np.float64)
    total_weight = float(weights.sum())
    values: dict[str, np.ndarray] = {}
    for column in US_SIPP_VEHICLE_OUTPUT_COLUMNS:
        if column in household.columns:
            values[column] = pd.to_numeric(
                household[column], errors="coerce"
            ).to_numpy(dtype=np.float64)
    owned = values.get("household_vehicles_owned", np.zeros(len(household)))
    value = values.get("household_vehicles_value", np.zeros(len(household)))
    finite_owned = np.nan_to_num(owned, nan=0.0, posinf=0.0, neginf=0.0)
    finite_value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    nonzero_shares = {
        "household_vehicles_owned": (
            float(weights[finite_owned > 0].sum()) / total_weight
            if total_weight > 0
            else 0.0
        ),
        "household_vehicles_value": (
            float(weights[finite_value > 0].sum()) / total_weight
            if total_weight > 0
            else 0.0
        ),
    }
    return {
        "nonzero_shares": nonzero_shares,
        "weighted_totals": {
            "household_vehicles_owned": float(np.dot(finite_owned, weights)),
            "household_vehicles_value": float(np.dot(finite_value, weights)),
        },
        "unique_counts": {
            column: int(pd.Series(column_values).dropna().nunique())
            for column, column_values in values.items()
        },
        "negative_counts": {
            column: int((column_values < 0).sum())
            for column, column_values in values.items()
        },
        "nonfinite_counts": {
            column: int((~np.isfinite(column_values)).sum())
            for column, column_values in values.items()
        },
        "owned_noninteger_count": int(
            (~np.isclose(finite_owned, np.rint(finite_owned))).sum()
        ),
        "positive_value_with_positive_owned_weighted_share": (
            float(weights[(finite_owned > 0) & (finite_value > 0)].sum()) / total_weight
            if total_weight > 0
            else 0.0
        ),
        "nonzero_share_bands": {
            "household_vehicles_owned": list(_OWNED_NONZERO_SHARE_BAND),
            "household_vehicles_value": list(_VALUE_NONZERO_SHARE_BAND),
        },
    }


def us_sipp_vehicles_signal_gate(frame: Frame) -> GateResult:
    """Require finite, nonnegative, nonconstant household vehicle signal."""

    household = frame.table("household")
    missing = [
        column
        for column in US_SIPP_VEHICLE_OUTPUT_COLUMNS
        if column not in household.columns
    ]
    if missing:
        return GateResult(
            name="sipp_vehicles_signal",
            passed=False,
            failures=(f"household columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_sipp_vehicles_summary(frame)
    failures: list[str] = []
    for column in US_SIPP_VEHICLE_OUTPUT_COLUMNS:
        if summary["unique_counts"][column] < 2:
            failures.append(f"{column}: constant column carries no signal.")
        if summary["negative_counts"][column]:
            failures.append(
                f"{column}: {summary['negative_counts'][column]} negative value(s)."
            )
        if summary["nonfinite_counts"][column]:
            failures.append(
                f"{column}: {summary['nonfinite_counts'][column]} non-finite value(s)."
            )
        share = float(summary["nonzero_shares"][column])
        low, high = summary["nonzero_share_bands"][column]
        if not (low <= share <= high):
            failures.append(
                f"{column}: nonzero share {share:.3f} outside plausibility band "
                f"[{low}, {high}]."
            )
        if float(summary["weighted_totals"][column]) <= 0:
            failures.append(f"{column}: weighted total is not positive.")
    if summary["owned_noninteger_count"]:
        failures.append(
            "household_vehicles_owned: "
            f"{summary['owned_noninteger_count']} non-integer value(s)."
        )
    if float(summary["positive_value_with_positive_owned_weighted_share"]) <= 0:
        failures.append("No weighted households carry both positive count and value.")
    return GateResult(
        name="sipp_vehicles_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
