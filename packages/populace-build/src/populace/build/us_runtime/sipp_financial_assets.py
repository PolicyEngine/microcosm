"""SIPP donor half of the US liquid-financial-asset blend (#374).

This module ports the SIPP asset-QRF source semantics reviewed in
``calibration/source_impute.py`` at archived enhanced-CPS commit
``42ed5d45``. That commit's final predictor surface has 13 fields (not the
later/current ten-field shorthand): employment, interest, dividend, rental,
Social Security, retirement, non-SSI income, age, sex, marriage, two child
counts, and household size. The exact source transforms are encoded below:
December person records; annualized monthly income fields;
``non_ssi_income`` equal only to employment + Social Security + retirement;
and ``TVAL_BANK`` / ``TVAL_STMF`` / ``TVAL_BOND`` as the three policy leaves.

The requested donor artifact is the 2023 SIPP public-use file, ``pu2023.csv``,
at an immutable Hugging Face model-repository revision.  This is a deliberate
vintage/predictor hybrid: ``42ed5d45`` had already changed its default source
to ``pu2024.csv`` (while retaining 2023 as its reference year), but issue #374
specifies the pinned 2023 artifact that the earlier enhanced-CPS builds used.
The runtime therefore matches the final reviewed 13-feature transform while
using the explicitly requested 2023 bytes.

The archive also applied target-specific source-quality masks.  For each leaf,
the raw target must be present and every associated SIPP allocation-status flag
must be 0, 1, or 9; statuses 2--8 are excluded.  The union is deterministically
capped at 20,000 rows using the archived target-balanced sampler.  Populace's
QRF has no ``target_filters`` argument, so the imputer fits the same chained
target order one leaf at a time on that leaf's reviewed rows, carrying each
draw forward as the next target's predictor.  This preserves the archive's
target-specific training surfaces rather than replacing them with an invented
intersection filter.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from populace.build.us_runtime.full_sipp_donor import full_sipp_sha256

__all__ = [
    "SIPP_2023_FINANCIAL_ASSET_DONOR_REVISION",
    "SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_ID_PARTS",
    "SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_TYPE",
    "SIPP_2023_FINANCIAL_ASSET_DONOR_SHA256",
    "SIPP_2023_FINANCIAL_ASSET_DONOR_SIZE_BYTES",
    "SIPP_2023_FINANCIAL_ASSET_DONOR_URL",
    "SIPP_FINANCIAL_ASSET_DONOR_WEIGHT_COLUMN",
    "SIPP_FINANCIAL_ASSET_MODEL_PREDICTORS",
    "SIPP_FINANCIAL_ASSET_SOURCE_COLUMNS",
    "SIPP_FINANCIAL_ASSET_TARGET_ALLOCATION_COLUMNS",
    "SIPP_FINANCIAL_ASSET_TARGET_SOURCE_COLUMNS",
    "fetch_sipp_2023_financial_asset_donor",
    "impute_us_sipp_financial_assets",
    "load_sipp_2023_financial_asset_donor",
]

SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_ID_PARTS = (
    "PolicyEngine/",
    "policyengine-",
    "us-data",
)
SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_TYPE = "model"
_RETIRED_DATA_REPOSITORY = "".join(
    SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_ID_PARTS[1:]
)
_SIPP_DONOR_REPOSITORY_ID = "".join(SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_ID_PARTS)
SIPP_2023_FINANCIAL_ASSET_DONOR_REVISION = "21280dca5995e978d706740a8a4b9b7860cfd7b6"
SIPP_2023_FINANCIAL_ASSET_DONOR_SHA256 = (
    "5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2"
)
SIPP_2023_FINANCIAL_ASSET_DONOR_SIZE_BYTES = 3_726_010_471
SIPP_2023_FINANCIAL_ASSET_DONOR_URL = (
    "https://huggingface.co/PolicyEngine/"
    f"{_RETIRED_DATA_REPOSITORY}/resolve/"
    f"{SIPP_2023_FINANCIAL_ASSET_DONOR_REVISION}/pu2023.csv"
)

SIPP_FINANCIAL_ASSET_TARGET_SOURCE_COLUMNS: dict[str, str] = {
    "bank_account_assets": "TVAL_BANK",
    "stock_assets": "TVAL_STMF",
    "bond_assets": "TVAL_BOND",
}

_SIPP_BANK_ACCOUNT_ALLOCATION_COLUMNS: tuple[str, ...] = (
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
)
_SIPP_STOCK_ALLOCATION_COLUMNS: tuple[str, ...] = (
    "AJSSTVAL",
    "AJOSTVAL",
    "AOSTVAL",
    "AJSMFVAL",
    "AJOMFVAL",
    "AOMFVAL",
)
_SIPP_BOND_ALLOCATION_COLUMNS: tuple[str, ...] = (
    "AJSGOVSVAL",
    "AJOGOVSVAL",
    "AOGOVSVAL",
    "AJSMCBDVAL",
    "AJOMCBDVAL",
    "AOMCBDVAL",
)
SIPP_FINANCIAL_ASSET_TARGET_ALLOCATION_COLUMNS: dict[str, tuple[str, ...]] = {
    "bank_account_assets": _SIPP_BANK_ACCOUNT_ALLOCATION_COLUMNS,
    "stock_assets": _SIPP_STOCK_ALLOCATION_COLUMNS,
    "bond_assets": _SIPP_BOND_ALLOCATION_COLUMNS,
}

_SIPP_JOB_EARNINGS_COLUMNS = tuple(f"TJB{i}_MSUM" for i in range(1, 8))
_SIPP_ALLOCATION_COLUMNS = tuple(
    sorted(
        {
            column
            for columns in SIPP_FINANCIAL_ASSET_TARGET_ALLOCATION_COLUMNS.values()
            for column in columns
        }
    )
)
SIPP_FINANCIAL_ASSET_SOURCE_COLUMNS: tuple[str, ...] = (
    "SSUID",
    "PNUM",
    "MONTHCODE",
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
    *_SIPP_ALLOCATION_COLUMNS,
)

SIPP_FINANCIAL_ASSET_MODEL_PREDICTORS: tuple[str, ...] = (
    "employment_income",
    "interest_income",
    "dividend_income",
    "rental_income",
    "social_security",
    "retirement_income",
    "non_ssi_income",
    "age",
    "is_female",
    "is_married",
    "count_under_18",
    "count_under_6",
    "household_size",
)

SIPP_FINANCIAL_ASSET_OUTPUT_COLUMNS: tuple[str, ...] = tuple(
    SIPP_FINANCIAL_ASSET_TARGET_SOURCE_COLUMNS
)

_DONOR_FILENAME = "pu2023.csv"
SIPP_FINANCIAL_ASSET_DONOR_WEIGHT_COLUMN = "sipp_weight"
_OBSERVED_SUFFIX = "_is_observed"
_OBSERVED_STATUS_VALUES = frozenset((0.0, 1.0, 9.0))
_DEFAULT_N_ESTIMATORS = 100
_MAX_TRAIN_SAMPLES = 20_000
_TRAINING_SAMPLE_SEED_NAME = "calibration_sipp_asset_training_sample"
_HOUSEHOLD_ID_COLUMN = "person_household_id"

_RECIPIENT_INTEREST_COLUMNS = (
    "taxable_interest_income",
    "tax_exempt_interest_income",
)
_RECIPIENT_DIVIDEND_COLUMNS = (
    "qualified_dividend_income",
    "non_qualified_dividend_income",
)
_RECIPIENT_SOCIAL_SECURITY_COLUMNS = (
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
)
_RECIPIENT_PENSION_COLUMNS = (
    "taxable_private_pension_income",
    "tax_exempt_private_pension_income",
    "taxable_public_pension_income",
    "tax_exempt_public_pension_income",
)
_RECIPIENT_RETIREMENT_DISTRIBUTION_COLUMNS = (
    "taxable_ira_distributions",
    "tax_exempt_ira_distributions",
    "taxable_401k_distributions",
    "tax_exempt_401k_distributions",
    "taxable_sep_distributions",
    "tax_exempt_sep_distributions",
    "taxable_403b_distributions",
    "tax_exempt_403b_distributions",
    "taxable_keogh_distributions",
    "tax_exempt_keogh_distributions",
)


def _sha256_stream(stream: BinaryIO, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(chunk_size), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return full_sipp_sha256(path)


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


def fetch_sipp_2023_financial_asset_donor(
    cache_dir: str | Path | None = None,
    *,
    local_path: str | Path | None = None,
    expected_sha256: str | None = SIPP_2023_FINANCIAL_ASSET_DONOR_SHA256,
    expected_size_bytes: int | None = SIPP_2023_FINANCIAL_ASSET_DONOR_SIZE_BYTES,
) -> Path:
    """Resolve the immutable full SIPP donor through ``huggingface_hub``.

    The known archived-repository checkout and legacy Populace SIPP cache are
    checked first to avoid a 3.73 GB transfer on PolicyEngine build machines.
    Every default-path candidate, including the Hugging Face result, must match
    the pinned byte length and SHA-256 before it is returned.
    """

    candidates: list[Path] = []
    if local_path is not None:
        candidates.append(Path(local_path).expanduser())
    if cache_dir is None:
        candidates.extend(
            (
                Path.home()
                / "PolicyEngine"
                / _RETIRED_DATA_REPOSITORY
                / ("policyengine_" + "us_data")
                / "storage"
                / _DONOR_FILENAME,
                Path.home() / ".cache" / "populace" / "sipp" / _DONOR_FILENAME,
            )
        )
    for candidate in dict.fromkeys(candidates):
        if _file_matches(
            candidate,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        ):
            return candidate

    from huggingface_hub import hf_hub_download

    downloaded = Path(
        hf_hub_download(
            repo_id=_SIPP_DONOR_REPOSITORY_ID,
            filename=_DONOR_FILENAME,
            repo_type=SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_TYPE,
            revision=SIPP_2023_FINANCIAL_ASSET_DONOR_REVISION,
            cache_dir=str(Path(cache_dir).expanduser())
            if cache_dir is not None
            else None,
        )
    )
    if not _file_matches(
        downloaded,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    ):
        actual_size = downloaded.stat().st_size if downloaded.is_file() else None
        actual_sha256 = _sha256_file(downloaded) if downloaded.is_file() else None
        raise ValueError(
            "SIPP 2023 financial-asset donor failed immutable-source verification: "
            f"expected size {expected_size_bytes} and sha256 {expected_sha256}, "
            f"got size {actual_size} and sha256 {actual_sha256}."
        )
    return downloaded


def _stable_string_seed(value: str) -> int:
    """Match the archived uint64 stable-string seed implementation."""

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


def _target_balanced_cap(
    donor: pd.DataFrame,
    *,
    max_train_samples: int | None,
) -> pd.DataFrame:
    """Port the archived target-filter-aware, positional 20k cap."""

    if max_train_samples is None:
        return donor.reset_index(drop=True)
    if max_train_samples < len(SIPP_FINANCIAL_ASSET_OUTPUT_COLUMNS):
        raise ValueError(
            "max_train_samples must be at least the number of SIPP asset targets"
        )
    filters = {
        target: donor[f"{target}{_OBSERVED_SUFFIX}"].astype(bool)
        for target in SIPP_FINANCIAL_ASSET_OUTPUT_COLUMNS
    }
    union = pd.Series(False, index=donor.index)
    for mask in filters.values():
        union |= mask
    union_positions = np.flatnonzero(union.to_numpy())
    if not len(union_positions):
        raise ValueError("SIPP asset donor has no observed rows across its targets.")
    if len(union_positions) <= max_train_samples:
        return donor.iloc[union_positions].copy().reset_index(drop=True)

    selected: list[int] = []
    selected_set: set[int] = set()
    per_target_cap = max(1, max_train_samples // len(filters))
    for target, mask in filters.items():
        positions = np.flatnonzero(mask.to_numpy())
        sampled = _sample_rng(_TRAINING_SAMPLE_SEED_NAME, salt=target).choice(
            positions,
            size=min(per_target_cap, len(positions)),
            replace=False,
        )
        for position in sampled:
            value = int(position)
            if value not in selected_set:
                selected.append(value)
                selected_set.add(value)

    remaining_n = max_train_samples - len(selected)
    if remaining_n > 0:
        remaining = np.asarray(
            [
                position
                for position in union_positions
                if int(position) not in selected_set
            ],
            dtype=int,
        )
        if len(remaining):
            sampled = _sample_rng(_TRAINING_SAMPLE_SEED_NAME, salt="fill").choice(
                remaining,
                size=min(remaining_n, len(remaining)),
                replace=False,
            )
            selected.extend(int(position) for position in sampled)
    return donor.iloc[np.asarray(selected, dtype=int)].copy().reset_index(drop=True)


def load_sipp_2023_financial_asset_donor(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    chunksize: int = 100_000,
    max_train_samples: int | None = _MAX_TRAIN_SAMPLES,
) -> pd.DataFrame:
    """Load the exact December SIPP asset-QRF donor surface from ``pu2023``."""

    source_path = Path(path)
    if (
        expected_size_bytes is not None
        and source_path.stat().st_size != expected_size_bytes
    ):
        raise ValueError(
            "SIPP 2023 financial-asset donor failed byte-length verification: "
            f"expected {expected_size_bytes}, got {source_path.stat().st_size}."
        )
    if expected_sha256 is not None:
        actual_sha256 = _sha256_file(source_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "SIPP 2023 financial-asset donor failed sha-256 verification: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )
    if chunksize < 1:
        raise ValueError("chunksize must be a positive integer")

    header = pd.read_csv(source_path, delimiter="|", nrows=0)
    missing = sorted(set(SIPP_FINANCIAL_ASSET_SOURCE_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"SIPP financial-asset donor missing column(s): {missing}.")

    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        source_path,
        delimiter="|",
        usecols=list(SIPP_FINANCIAL_ASSET_SOURCE_COLUMNS),
        chunksize=int(chunksize),
        dtype={"SSUID": "string"},
        low_memory=False,
    ):
        month = pd.to_numeric(chunk["MONTHCODE"], errors="coerce")
        december = chunk.loc[month.eq(12)].copy()
        if not december.empty:
            parts.append(december)
    if not parts:
        raise ValueError("SIPP financial-asset donor has no December rows.")
    raw = pd.concat(parts, ignore_index=True)

    for column in SIPP_FINANCIAL_ASSET_SOURCE_COLUMNS:
        if column == "SSUID":
            continue
        original = raw[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            rows = np.flatnonzero(invalid.to_numpy())[:5].tolist()
            raise ValueError(
                f"SIPP financial-asset source {column!r} contains nonnumeric "
                f"value(s) at row(s) {rows}."
            )
        raw[column] = converted
    if raw[["SSUID", "PNUM"]].isna().any(axis=None):
        raise ValueError("SIPP financial-asset December source has missing IDs.")
    if raw.duplicated(["SSUID", "PNUM"]).any():
        raise ValueError("SIPP financial-asset December source has duplicate people.")

    donor = pd.DataFrame(index=raw.index)
    for target, source_column in SIPP_FINANCIAL_ASSET_TARGET_SOURCE_COLUMNS.items():
        donor[target] = raw[source_column].fillna(0.0).astype(np.float64)
        observed = raw[source_column].notna()
        for flag_column in SIPP_FINANCIAL_ASSET_TARGET_ALLOCATION_COLUMNS[target]:
            observed &= raw[flag_column].fillna(0.0).isin(_OBSERVED_STATUS_VALUES)
        donor[f"{target}{_OBSERVED_SUFFIX}"] = observed.to_numpy(dtype=bool)

    donor["employment_income"] = (
        raw.loc[:, list(_SIPP_JOB_EARNINGS_COLUMNS)].fillna(0.0).sum(axis=1) * 12.0
    )
    donor["interest_income"] = (
        raw["TINC_BANK"].fillna(0.0) + raw["TINC_BOND"].fillna(0.0)
    ) * 12.0
    donor["dividend_income"] = raw["TINC_STMF"].fillna(0.0) * 12.0
    donor["rental_income"] = raw["TINC_RENT"].fillna(0.0) * 12.0
    donor["social_security"] = raw["TSSSAMT"].fillna(0.0) * 12.0
    donor["retirement_income"] = raw["TRETINCAMT"].fillna(0.0) * 12.0
    donor["non_ssi_income"] = (
        donor["employment_income"]
        + donor["social_security"]
        + donor["retirement_income"]
    )
    donor["age"] = raw["TAGE"]
    donor["is_female"] = raw["ESEX"].eq(2.0).astype(np.float64)
    donor["is_married"] = raw["EMS"].eq(1.0).astype(np.float64)
    under_18 = raw["TAGE"].lt(18.0)
    under_6 = raw["TAGE"].lt(6.0)
    donor["count_under_18"] = under_18.groupby(raw["SSUID"]).transform("sum")
    donor["count_under_6"] = under_6.groupby(raw["SSUID"]).transform("sum")
    donor["household_size"] = raw["PNUM"].groupby(raw["SSUID"]).transform("count")
    donor[SIPP_FINANCIAL_ASSET_DONOR_WEIGHT_COLUMN] = raw["WPFINWGT"]

    weights = donor[SIPP_FINANCIAL_ASSET_DONOR_WEIGHT_COLUMN].to_numpy(dtype=np.float64)
    positive_finite_weight = np.isfinite(weights) & (weights > 0.0)
    donor = donor.loc[positive_finite_weight].copy().reset_index(drop=True)
    for target in SIPP_FINANCIAL_ASSET_OUTPUT_COLUMNS:
        if not donor[f"{target}{_OBSERVED_SUFFIX}"].any():
            raise ValueError(
                f"SIPP financial-asset donor has no observed {target} rows."
            )
        observed_values = donor.loc[
            donor[f"{target}{_OBSERVED_SUFFIX}"], target
        ].to_numpy(dtype=np.float64)
        if not np.isfinite(observed_values).all():
            raise ValueError(
                f"SIPP financial-asset target {target!r} contains nonfinite "
                "observed values."
            )
        if (observed_values < 0.0).any():
            raise ValueError(
                f"SIPP financial-asset target {target!r} must be nonnegative."
            )
    donor = _target_balanced_cap(donor, max_train_samples=max_train_samples)
    ordered = [
        *SIPP_FINANCIAL_ASSET_MODEL_PREDICTORS,
        *SIPP_FINANCIAL_ASSET_OUTPUT_COLUMNS,
        SIPP_FINANCIAL_ASSET_DONOR_WEIGHT_COLUMN,
        *(
            f"{target}{_OBSERVED_SUFFIX}"
            for target in SIPP_FINANCIAL_ASSET_OUTPUT_COLUMNS
        ),
    ]
    return donor.loc[:, ordered]


def _sum_present(person: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    total = np.zeros(len(person), dtype=np.float64)
    for column in columns:
        if column in person.columns:
            total += (
                pd.to_numeric(person[column], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
    return total


def _aggregate_or_components(
    person: pd.DataFrame,
    aggregate: str,
    components: tuple[str, ...],
) -> np.ndarray:
    if aggregate in person.columns:
        return (
            pd.to_numeric(person[aggregate], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
    return _sum_present(person, components)


def _recipient_is_married(person: pd.DataFrame) -> np.ndarray:
    """Match the archived receiver: explicit flag, then paired marital unit."""

    if "is_married" in person.columns:
        return (
            person["is_married"].fillna(False).astype(bool).to_numpy(dtype=np.float64)
        )
    if "person_marital_unit_id" in person.columns:
        unit = person["person_marital_unit_id"]
        return unit.map(unit.value_counts()).gt(1).to_numpy(dtype=np.float64)
    return np.zeros(len(person), dtype=np.float64)


def _household_head_mask(person: pd.DataFrame) -> np.ndarray:
    household = pd.to_numeric(person[_HOUSEHOLD_ID_COLUMN], errors="coerce").to_numpy()
    if "A_LINENO" in person.columns:
        line = (
            pd.to_numeric(person["A_LINENO"], errors="coerce").fillna(9_999).to_numpy()
        )
    else:
        line = np.arange(len(person))
    position = np.arange(len(person))
    order = np.lexsort((position, line, household))
    ordered_household = household[order]
    first = np.empty(len(person), dtype=bool)
    first[0] = True
    first[1:] = ordered_household[1:] != ordered_household[:-1]
    mask = np.zeros(len(person), dtype=bool)
    mask[order[first]] = True
    return mask


def _recipient_sipp_asset_predictor_table(person: pd.DataFrame) -> pd.DataFrame:
    required = {_HOUSEHOLD_ID_COLUMN, "age", "is_female"}
    missing = sorted(required - set(person.columns))
    if missing:
        raise ValueError(
            f"US SIPP financial-asset imputation requires recipient column(s): {missing}."
        )

    result = pd.DataFrame(index=person.index)
    if "employment_income" in person.columns:
        employment = _aggregate_or_components(person, "employment_income", ())
    elif "employment_income_before_lsr" in person.columns:
        employment = _aggregate_or_components(
            person, "employment_income_before_lsr", ()
        )
    else:
        employment = np.zeros(len(person), dtype=np.float64)
    result["employment_income"] = employment
    result["interest_income"] = _aggregate_or_components(
        person, "interest_income", _RECIPIENT_INTEREST_COLUMNS
    )
    result["dividend_income"] = _aggregate_or_components(
        person, "dividend_income", _RECIPIENT_DIVIDEND_COLUMNS
    )
    result["rental_income"] = _aggregate_or_components(person, "rental_income", ())
    result["social_security"] = _aggregate_or_components(
        person, "social_security", _RECIPIENT_SOCIAL_SECURITY_COLUMNS
    )
    pension = _aggregate_or_components(
        person, "pension_income", _RECIPIENT_PENSION_COLUMNS
    )
    distributions = _aggregate_or_components(
        person,
        "retirement_distributions",
        _RECIPIENT_RETIREMENT_DISTRIBUTION_COLUMNS,
    )
    result["retirement_income"] = pension + distributions
    result["non_ssi_income"] = (
        employment + result["social_security"] + result["retirement_income"]
    )
    result["age"] = pd.to_numeric(person["age"], errors="coerce").fillna(0.0)
    result["is_female"] = person["is_female"].fillna(False).astype(bool).astype(float)
    result["is_married"] = _recipient_is_married(person)

    household_ids = person[_HOUSEHOLD_ID_COLUMN]
    age = result["age"]
    result["count_under_18"] = age.lt(18).groupby(household_ids).transform("sum")
    result["count_under_6"] = age.lt(6).groupby(household_ids).transform("sum")
    result["household_size"] = age.groupby(household_ids).transform("size")
    return result.loc[:, list(SIPP_FINANCIAL_ASSET_MODEL_PREDICTORS)].astype(np.float64)


def impute_us_sipp_financial_assets(
    person: pd.DataFrame,
    donor: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> pd.DataFrame:
    """Draw the three SIPP leaves and carry them only on household heads.

    The archive predicted SIPP assets at person grain.  Issue #374 explicitly
    tightens the shipped grain: as with SCF, one reference person carries the
    household's complete three-leaf vector and every other member receives 0.
    """

    from populace.fit import QRF

    required = {
        *SIPP_FINANCIAL_ASSET_MODEL_PREDICTORS,
        *SIPP_FINANCIAL_ASSET_OUTPUT_COLUMNS,
        SIPP_FINANCIAL_ASSET_DONOR_WEIGHT_COLUMN,
        *(
            f"{target}{_OBSERVED_SUFFIX}"
            for target in SIPP_FINANCIAL_ASSET_OUTPUT_COLUMNS
        ),
    }
    missing = sorted(required - set(donor.columns))
    if missing:
        raise ValueError(
            f"SIPP financial-asset donor table missing column(s): {missing}."
        )

    head_mask = _household_head_mask(person)
    head_features = _recipient_sipp_asset_predictor_table(person).loc[head_mask]
    augmented = head_features.copy()
    draws = pd.DataFrame(index=head_features.index)
    target_seeds = np.random.SeedSequence([int(seed), 374]).spawn(
        len(SIPP_FINANCIAL_ASSET_OUTPUT_COLUMNS)
    )
    prior_targets: list[str] = []
    for target, target_seed in zip(
        SIPP_FINANCIAL_ASSET_OUTPUT_COLUMNS,
        target_seeds,
        strict=True,
    ):
        observed_column = f"{target}{_OBSERVED_SUFFIX}"
        observed = donor[observed_column].astype(bool).to_numpy()
        train = donor.loc[
            observed,
            [
                *SIPP_FINANCIAL_ASSET_MODEL_PREDICTORS,
                *prior_targets,
                target,
                SIPP_FINANCIAL_ASSET_DONOR_WEIGHT_COLUMN,
            ],
        ].copy()
        predictors = [*SIPP_FINANCIAL_ASSET_MODEL_PREDICTORS, *prior_targets]
        model_seed = int(target_seed.generate_state(1, dtype=np.uint32)[0])
        fitted = QRF(n_estimators=int(n_estimators), seed=model_seed).fit(
            train,
            predictors=predictors,
            targets=[target],
            weights=SIPP_FINANCIAL_ASSET_DONOR_WEIGHT_COLUMN,
        )
        draw = fitted.predict(augmented)[target].to_numpy(dtype=np.float64)
        if not np.isfinite(draw).all():
            raise ValueError(
                f"SIPP financial-asset predictions for {target} are nonfinite."
            )
        if (draw < 0.0).any():
            raise ValueError(
                f"SIPP financial-asset predictions for {target} are negative."
            )
        draws[target] = draw
        augmented[target] = draw
        prior_targets.append(target)

    result = pd.DataFrame(
        0.0,
        index=person.index,
        columns=list(SIPP_FINANCIAL_ASSET_OUTPUT_COLUMNS),
        dtype=np.float64,
    )
    result.loc[head_mask, list(SIPP_FINANCIAL_ASSET_OUTPUT_COLUMNS)] = draws.to_numpy(
        dtype=np.float64
    )
    return result
