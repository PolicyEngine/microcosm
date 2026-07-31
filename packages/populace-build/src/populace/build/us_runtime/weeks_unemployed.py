"""Measured CPS ASEC weeks looking for work and the PUF-half QRF.

The retired eCPS build carried person-level ``LKWEEKS`` directly, replacing
only its historical ``-1`` not-in-universe sentinel with zero.  The locked
2022-income-year ASEC HDF5 omitted that raw column, so this stage repairs only
that cohort from the official, immutable 2023 ASEC public-use archive.  The
repair is an exact ``PERIDNUM`` join with redundant Census identity checks; it
never predicts an ASEC source value.

After support cloning, the retired PUF-specific routine replaces only the PUF
half with a QRF using age, sex, joint filing, the three tax-unit roles, and
unemployment compensation when available.  Populace adds typed person weights
and a build seed.  QRF interpolation can be fractional, so predictions are
clipped and rounded to the integer-valued 0--52 input domain before the
archived unemployment-compensation zero rule is applied.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from importlib.resources import files
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import (
    SourceOperationSpec,
    SourceStageSpec,
    load_source_manifest,
)
from populace.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
    run_source_stage,
)
from populace.build.us_runtime.support_provenance import (
    has_support_role_metadata,
    support_role_series,
)
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

__all__ = [
    "ASEC_2023_WEEKS_UNEMPLOYED_MEMBER",
    "ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_CRC32",
    "ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SHA256",
    "ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SIZE_BYTES",
    "ASEC_2023_WEEKS_UNEMPLOYED_POSITIVE_ROWS",
    "ASEC_2023_WEEKS_UNEMPLOYED_RAW_ROWS",
    "ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_COLUMNS",
    "ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_SHA256",
    "ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_YEAR",
    "ASEC_2023_WEEKS_UNEMPLOYED_UNIQUE_KEYS",
    "ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_SOURCE_SHARE",
    "ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_WEEKS",
    "ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SHA256",
    "ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SIZE_BYTES",
    "ASEC_2023_WEEKS_UNEMPLOYED_ZIP_URL",
    "WEEKS_UNEMPLOYED_ARCHIVED_DERIVATION_URL",
    "WEEKS_UNEMPLOYED_ARCHIVED_PUF_IMPUTATION_URL",
    "WEEKS_UNEMPLOYED_ARCHIVED_SOURCE_URL",
    "WEEKS_UNEMPLOYED_DERIVE_PARAMETERS",
    "WEEKS_UNEMPLOYED_PUF_IMPUTATION_PARAMETERS",
    "WEEKS_UNEMPLOYED_PUF_PREDICTORS",
    "WEEKS_UNEMPLOYED_READ_PARAMETERS",
    "US_WEEKS_UNEMPLOYED_NONCONSTANT_PERSON_COLUMNS",
    "US_WEEKS_UNEMPLOYED_OUTPUT_COLUMNS",
    "US_WEEKS_UNEMPLOYED_REQUIRED_SOURCE_COLUMNS",
    "US_WEEKS_UNEMPLOYED_STAGE_NAME",
    "derive_us_weeks_unemployed_from_manifest",
    "fetch_asec_2023_weeks_unemployed_source",
    "fill_asec_2022_weeks_unemployed_source",
    "impute_us_weeks_unemployed_to_puf_support_from_manifest",
    "load_asec_2023_weeks_unemployed_source",
    "us_weeks_unemployed_signal_gate",
    "us_weeks_unemployed_stage_spec",
    "us_weeks_unemployed_summary",
    "with_us_weeks_unemployed",
]

QRF: Any | None = None

_ARCHIVED_COMMIT = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
_ARCHIVED_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_PACKAGE = "policyengine_" + "us_data"
_ARCHIVED_ROOT = (
    f"https://github.com/PolicyEngine/{_ARCHIVED_REPOSITORY}/blob/"
    f"{_ARCHIVED_COMMIT}/{_ARCHIVED_PACKAGE}"
)
WEEKS_UNEMPLOYED_ARCHIVED_DERIVATION_URL = (
    _ARCHIVED_ROOT + "/datasets/cps/cps.py#L1438-L1442"
)
WEEKS_UNEMPLOYED_ARCHIVED_SOURCE_URL = (
    _ARCHIVED_ROOT + "/datasets/cps/census_cps.py#L39-L58"
)
WEEKS_UNEMPLOYED_ARCHIVED_PUF_IMPUTATION_URL = (
    _ARCHIVED_ROOT + "/calibration/puf_impute.py#L703-L785"
)

ASEC_2023_WEEKS_UNEMPLOYED_ZIP_URL = (
    "https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asecpub23csv.zip"
)
ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SIZE_BYTES = 150_165_063
ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SHA256 = (
    "d2e000250782adfbdd7f29c82b66d866591a30f0d330496698ec19f9c784ce11"
)
# Stable upstream identity used by release checkpoints and reform-vector
# caches.  Fetching may return the extracted member, but both representations
# are accepted only after verification against this archive and its member.
ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_SHA256 = ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SHA256
ASEC_2023_WEEKS_UNEMPLOYED_MEMBER = "pppub23.csv"
ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SIZE_BYTES = 281_065_733
ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_CRC32 = "49c09e5f"
ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SHA256 = (
    "19b56537e50e7663f954361ef2bb5ce9cef8d9d45f156fe1a69a99b654198ffe"
)
ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_YEAR = 2022
ASEC_2023_WEEKS_UNEMPLOYED_RAW_ROWS = 146_133
ASEC_2023_WEEKS_UNEMPLOYED_UNIQUE_KEYS = 146_133
ASEC_2023_WEEKS_UNEMPLOYED_POSITIVE_ROWS = 4_200
ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_SOURCE_SHARE = 0.03145609427589182
ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_WEEKS = 172_563_622.04

ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_COLUMNS: tuple[str, ...] = (
    "PH_SEQ",
    "P_SEQ",
    "A_LINENO",
    "PERIDNUM",
    "LKWEEKS",
)
_AUDIT_WEIGHT_COLUMN = "A_FNLWGT"

US_WEEKS_UNEMPLOYED_STAGE_NAME = "weeks_unemployed_input"
US_WEEKS_UNEMPLOYED_OUTPUT_COLUMNS: tuple[str, ...] = ("weeks_unemployed",)
US_WEEKS_UNEMPLOYED_NONCONSTANT_PERSON_COLUMNS = US_WEEKS_UNEMPLOYED_OUTPUT_COLUMNS
US_WEEKS_UNEMPLOYED_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "source_year",
    "PERIDNUM",
    "LKWEEKS",
)

_OUTPUT = US_WEEKS_UNEMPLOYED_OUTPUT_COLUMNS[0]
_SOURCE = "LKWEEKS"
_PERSON_WEIGHT_COLUMN = "person_weight"
_ASEC_CHANNEL = "asec"
_PUF_CHANNEL = "puf_tax_detail"
_PREDICTOR_PREFIX = "weeks_unemployed_predictor_"
_REQUIRED_PUF_PREDICTORS: tuple[str, ...] = (
    "age",
    "is_male",
    "tax_unit_is_joint",
    "is_tax_unit_head",
    "is_tax_unit_spouse",
    "is_tax_unit_dependent",
)
_OPTIONAL_UC_PREDICTOR = "unemployment_compensation"
WEEKS_UNEMPLOYED_PUF_PREDICTORS: tuple[str, ...] = (
    *_REQUIRED_PUF_PREDICTORS,
    _OPTIONAL_UC_PREDICTOR,
)

WEEKS_UNEMPLOYED_READ_PARAMETERS: dict[str, object] = {
    "table": "person",
    "weight": _PERSON_WEIGHT_COLUMN,
}
WEEKS_UNEMPLOYED_DERIVE_PARAMETERS: dict[str, object] = {
    "source": _SOURCE,
    "niu_value": -1,
    "niu_replacement": 0,
    "output": _OUTPUT,
}
WEEKS_UNEMPLOYED_PUF_IMPUTATION_PARAMETERS: dict[str, object] = {
    "predictors": list(WEEKS_UNEMPLOYED_PUF_PREDICTORS),
    "max_train_samples": 5_000,
    "n_estimators": 100,
    "seed_from_build_config": True,
    "weight": _PERSON_WEIGHT_COLUMN,
    "clip": [0, 52],
    "round": "nearest_integer",
    "zero_unless": "unemployment_compensation > 0",
}

# Deliberately broad source-plausibility bands. The full official 2023 ASEC
# source has a 3.1456% weighted positive share and 0.5234 weighted weeks per
# person; the persisted sparse production smoke has 5.2182% / 0.9036 on ASEC
# and 0.5594% / 0.0913 on PUF support. These floors reject a one-row or
# effectively collapsed QRF surface while allowing support selection and
# adjacent ASEC vintages ample room to move.
_CHANNEL_POSITIVE_SHARE_BANDS: dict[str, tuple[float, float]] = {
    _ASEC_CHANNEL: (0.015, 0.10),
    _PUF_CHANNEL: (0.001, 0.03),
}
_CHANNEL_WEIGHTED_MEAN_WEEKS_BANDS: dict[str, tuple[float, float]] = {
    _ASEC_CHANNEL: (0.20, 2.0),
    _PUF_CHANNEL: (0.01, 0.50),
}


def us_weeks_unemployed_stage_spec() -> SourceStageSpec:
    """Load and strictly validate the packaged weeks-unemployed stage."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    spec = manifest.stage_map().get(US_WEEKS_UNEMPLOYED_STAGE_NAME)
    if spec is None:
        raise ValueError(
            f"US source manifest declares no {US_WEEKS_UNEMPLOYED_STAGE_NAME!r} stage."
        )
    if spec.grain != "person":
        raise ValueError("US weeks-unemployed stage must have person grain.")
    if tuple(spec.outputs) != US_WEEKS_UNEMPLOYED_OUTPUT_COLUMNS:
        raise ValueError(
            "US weeks-unemployed manifest outputs do not match the runtime family."
        )
    expected_operations = (
        ("read_table", WEEKS_UNEMPLOYED_READ_PARAMETERS),
        ("derive_weeks_unemployed", WEEKS_UNEMPLOYED_DERIVE_PARAMETERS),
        (
            "impute_weeks_unemployed_to_puf_support",
            WEEKS_UNEMPLOYED_PUF_IMPUTATION_PARAMETERS,
        ),
    )
    actual_operations = tuple(
        (operation.kind, dict(operation.parameters)) for operation in spec.operations
    )
    if actual_operations != expected_operations:
        raise ValueError(
            "US weeks-unemployed operations drifted from the reviewed direct "
            "carry and PUF-imputation contract."
        )
    pinned = [
        artifact
        for artifact in spec.artifacts
        if artifact.get("locator") == ASEC_2023_WEEKS_UNEMPLOYED_ZIP_URL
        and artifact.get("size_bytes") == ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SIZE_BYTES
        and artifact.get("sha256") == ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SHA256
        and artifact.get("member") == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER
        and artifact.get("member_size_bytes")
        == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SIZE_BYTES
        and str(artifact.get("member_crc32", "")).lower()
        == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_CRC32
        and artifact.get("member_sha256") == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SHA256
    ]
    if not pinned:
        raise ValueError(
            "US weeks-unemployed stage does not pin the official archive and "
            "person member identities."
        )
    return spec


def _sha256_stream(stream: BinaryIO, *, chunk_size: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(chunk_size), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _verify_file(
    path: Path,
    *,
    label: str,
    expected_size_bytes: int | None,
    expected_sha256: str | None,
    chunk_size: int,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if expected_size_bytes is not None and size != expected_size_bytes:
        raise ValueError(
            f"{label} byte length mismatch: expected {expected_size_bytes}, got {size}."
        )
    if expected_sha256 is not None:
        with path.open("rb") as stream:
            digest, _ = _sha256_stream(stream, chunk_size=chunk_size)
        if digest != expected_sha256:
            raise ValueError(
                f"{label} SHA-256 mismatch: expected {expected_sha256}, got {digest}."
            )


def _verified_member_info(
    archive: zipfile.ZipFile,
    *,
    expected_size_bytes: int | None,
    expected_crc32: str | None,
) -> zipfile.ZipInfo:
    members = [
        info
        for info in archive.infolist()
        if info.filename == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER
    ]
    if len(members) != 1:
        raise ValueError(
            "ASEC 2023 archive must contain exactly one "
            f"{ASEC_2023_WEEKS_UNEMPLOYED_MEMBER!r} member; found {len(members)}."
        )
    info = members[0]
    if expected_size_bytes is not None and info.file_size != expected_size_bytes:
        raise ValueError(
            "ASEC 2023 person member byte length mismatch: expected "
            f"{expected_size_bytes}, got {info.file_size}."
        )
    actual_crc = f"{info.CRC:08x}"
    if expected_crc32 is not None and actual_crc != expected_crc32.lower():
        raise ValueError(
            "ASEC 2023 person member CRC32 mismatch: expected "
            f"{expected_crc32.lower()}, got {actual_crc}."
        )
    return info


def fetch_asec_2023_weeks_unemployed_source(
    cache_dir: str | Path | None = None,
    *,
    expected_zip_size_bytes: int | None = (ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SIZE_BYTES),
    expected_zip_sha256: str | None = ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SHA256,
    expected_member_size_bytes: int | None = (
        ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SIZE_BYTES
    ),
    expected_member_crc32: str | None = (ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_CRC32),
    expected_member_sha256: str | None = (ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SHA256),
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Download, verify, extract, and cache the official ASEC person member."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    root = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path.home() / ".cache" / "populace" / "cps" / "asec_2023"
    )
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "asecpub23csv.zip"
    member_path = root / ASEC_2023_WEEKS_UNEMPLOYED_MEMBER

    if not archive_path.exists():
        temporary = archive_path.with_suffix(".zip.part")
        request = urllib.request.Request(
            ASEC_2023_WEEKS_UNEMPLOYED_ZIP_URL,
            headers={"User-Agent": "populace-build/asec-weeks-unemployed"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
                with temporary.open("wb") as destination:
                    digest = hashlib.sha256()
                    size = 0
                    for chunk in iter(lambda: response.read(chunk_size), b""):
                        destination.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            if expected_zip_size_bytes is not None and size != expected_zip_size_bytes:
                raise ValueError(
                    "ASEC 2023 archive download byte length mismatch: expected "
                    f"{expected_zip_size_bytes}, got {size}."
                )
            actual_digest = digest.hexdigest()
            if expected_zip_sha256 is not None and actual_digest != expected_zip_sha256:
                raise ValueError(
                    "ASEC 2023 archive download SHA-256 mismatch: expected "
                    f"{expected_zip_sha256}, got {actual_digest}."
                )
            os.replace(temporary, archive_path)
        finally:
            temporary.unlink(missing_ok=True)

    _verify_file(
        archive_path,
        label="ASEC 2023 archive",
        expected_size_bytes=expected_zip_size_bytes,
        expected_sha256=expected_zip_sha256,
        chunk_size=chunk_size,
    )
    with zipfile.ZipFile(archive_path) as archive:
        info = _verified_member_info(
            archive,
            expected_size_bytes=expected_member_size_bytes,
            expected_crc32=expected_member_crc32,
        )
        if member_path.exists():
            _verify_file(
                member_path,
                label="ASEC 2023 person member",
                expected_size_bytes=expected_member_size_bytes,
                expected_sha256=expected_member_sha256,
                chunk_size=chunk_size,
            )
            return member_path

        temporary = member_path.with_suffix(".csv.part")
        try:
            with archive.open(info) as source, temporary.open("wb") as destination:
                digest = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: source.read(chunk_size), b""):
                    destination.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if (
                expected_member_size_bytes is not None
                and size != expected_member_size_bytes
            ):
                raise ValueError(
                    "ASEC 2023 extracted person member byte length mismatch: "
                    f"expected {expected_member_size_bytes}, got {size}."
                )
            actual_digest = digest.hexdigest()
            if (
                expected_member_sha256 is not None
                and actual_digest != expected_member_sha256
            ):
                raise ValueError(
                    "ASEC 2023 extracted person member SHA-256 mismatch: "
                    f"expected {expected_member_sha256}, got {actual_digest}."
                )
            os.replace(temporary, member_path)
        finally:
            temporary.unlink(missing_ok=True)
    return member_path


def _fixed_width_peridnum(values: pd.Series, *, label: str) -> pd.Series:
    if values.isna().any():
        rows = values.index[values.isna()].tolist()[:5]
        raise ValueError(f"{label} PERIDNUM is missing at row(s): {rows}.")
    decoded = values.map(
        lambda value: (
            value.decode()
            if isinstance(value, (bytes, bytearray, np.bytes_))
            else value
        )
    ).astype(str)
    valid = decoded.str.fullmatch(r"[0-9]{22}", na=False)
    if not valid.all():
        rows = decoded.index[~valid].tolist()[:5]
        raise ValueError(
            f"{label} PERIDNUM must be an exact 22-digit string at row(s): {rows}."
        )
    return decoded


def _strict_integer(
    values: pd.Series,
    *,
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(numeric) & (numeric == np.floor(numeric))
    if minimum is not None:
        valid &= numeric >= minimum
    if maximum is not None:
        valid &= numeric <= maximum
    if not valid.all():
        rows = np.flatnonzero(~valid)[:5].tolist()
        raise ValueError(f"{label} must be finite integers at row(s): {rows}.")
    return numeric.astype(np.int64)


def _load_csv_source(
    source: Path,
    *,
    expected_zip_size_bytes: int | None,
    expected_zip_sha256: str | None,
    expected_member_size_bytes: int | None,
    expected_member_crc32: str | None,
    expected_member_sha256: str | None,
    chunk_size: int,
) -> pd.DataFrame:
    usecols = [*ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_COLUMNS, _AUDIT_WEIGHT_COLUMN]
    if zipfile.is_zipfile(source):
        _verify_file(
            source,
            label="ASEC 2023 archive",
            expected_size_bytes=expected_zip_size_bytes,
            expected_sha256=expected_zip_sha256,
            chunk_size=chunk_size,
        )
        with zipfile.ZipFile(source) as archive:
            info = _verified_member_info(
                archive,
                expected_size_bytes=expected_member_size_bytes,
                expected_crc32=expected_member_crc32,
            )
            with archive.open(info) as member:
                digest, size = _sha256_stream(member, chunk_size=chunk_size)
            if (
                expected_member_size_bytes is not None
                and size != expected_member_size_bytes
            ):
                raise ValueError("ASEC 2023 person member byte length mismatch.")
            if expected_member_sha256 is not None and digest != expected_member_sha256:
                raise ValueError("ASEC 2023 person member SHA-256 mismatch.")
            with archive.open(info) as member:
                return pd.read_csv(
                    member,
                    usecols=usecols,
                    dtype={"PERIDNUM": "string"},
                    low_memory=False,
                )

    _verify_file(
        source,
        label="ASEC 2023 person member",
        expected_size_bytes=expected_member_size_bytes,
        expected_sha256=expected_member_sha256,
        chunk_size=chunk_size,
    )
    return pd.read_csv(
        source,
        usecols=usecols,
        dtype={"PERIDNUM": "string"},
        low_memory=False,
    )


def _assert_pinned_source_audit(audit: dict[str, int | float]) -> None:
    expected_counts = {
        "raw_rows": ASEC_2023_WEEKS_UNEMPLOYED_RAW_ROWS,
        "unique_keys": ASEC_2023_WEEKS_UNEMPLOYED_UNIQUE_KEYS,
        "positive_rows": ASEC_2023_WEEKS_UNEMPLOYED_POSITIVE_ROWS,
    }
    for key, expected in expected_counts.items():
        if int(audit[key]) != expected:
            raise ValueError(
                f"Pinned ASEC weeks-unemployed audit drifted for {key}: "
                f"expected {expected}, got {audit[key]}."
            )
    expected_floats = {
        "weighted_source_share": (
            ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_SOURCE_SHARE,
            1e-12,
        ),
        # A 172.6m person-week dot product accumulates a harmless ~2e-8
        # binary-float residue on the pinned file.  One millionth of a
        # person-week remains far tighter than the source's 0.01 precision.
        "weighted_weeks": (ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_WEEKS, 1e-6),
    }
    for key, (expected, tolerance) in expected_floats.items():
        if not np.isclose(float(audit[key]), expected, rtol=0.0, atol=tolerance):
            raise ValueError(
                f"Pinned ASEC weeks-unemployed audit drifted for {key}: "
                f"expected {expected}, got {audit[key]}."
            )


def load_asec_2023_weeks_unemployed_source(
    path: str | Path,
    *,
    expected_zip_size_bytes: int | None = (ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SIZE_BYTES),
    expected_zip_sha256: str | None = ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SHA256,
    expected_member_size_bytes: int | None = (
        ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SIZE_BYTES
    ),
    expected_member_crc32: str | None = (ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_CRC32),
    expected_member_sha256: str | None = (ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SHA256),
    expected_rows: int | None = ASEC_2023_WEEKS_UNEMPLOYED_RAW_ROWS,
    expected_unique_keys: int | None = ASEC_2023_WEEKS_UNEMPLOYED_UNIQUE_KEYS,
    chunk_size: int = 8 * 1024 * 1024,
) -> pd.DataFrame:
    """Load the exact identity/LKWEEKS sidecar and attach its source audit."""

    source = Path(path).expanduser()
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    raw = _load_csv_source(
        source,
        expected_zip_size_bytes=expected_zip_size_bytes,
        expected_zip_sha256=expected_zip_sha256,
        expected_member_size_bytes=expected_member_size_bytes,
        expected_member_crc32=expected_member_crc32,
        expected_member_sha256=expected_member_sha256,
        chunk_size=chunk_size,
    )
    missing = sorted(
        {
            *ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_COLUMNS,
            _AUDIT_WEIGHT_COLUMN,
        }
        - set(raw.columns)
    )
    if missing:
        raise ValueError(f"ASEC weeks-unemployed source missing column(s): {missing}.")
    if expected_rows is not None and len(raw) != expected_rows:
        raise ValueError(
            "ASEC weeks-unemployed source row count mismatch: expected "
            f"{expected_rows}, got {len(raw)}."
        )

    raw["PERIDNUM"] = _fixed_width_peridnum(raw["PERIDNUM"], label="ASEC source")
    unique_keys = int(raw["PERIDNUM"].nunique(dropna=False))
    if raw["PERIDNUM"].duplicated(keep=False).any():
        examples = (
            raw.loc[raw["PERIDNUM"].duplicated(keep=False), "PERIDNUM"]
            .drop_duplicates()
            .tolist()[:5]
        )
        raise ValueError(
            "ASEC weeks-unemployed PERIDNUM must be unique; duplicate key(s): "
            f"{examples}."
        )
    if expected_unique_keys is not None and unique_keys != expected_unique_keys:
        raise ValueError(
            "ASEC weeks-unemployed unique-key count mismatch: expected "
            f"{expected_unique_keys}, got {unique_keys}."
        )

    raw["PH_SEQ"] = _strict_integer(
        raw["PH_SEQ"], label="ASEC PH_SEQ", minimum=1, maximum=99_999
    )
    raw["P_SEQ"] = _strict_integer(
        raw["P_SEQ"], label="ASEC P_SEQ", minimum=0, maximum=16
    )
    raw["A_LINENO"] = _strict_integer(
        raw["A_LINENO"], label="ASEC A_LINENO", minimum=1, maximum=16
    )
    weeks = _strict_integer(raw[_SOURCE], label="ASEC LKWEEKS")
    valid_weeks = (weeks == -1) | ((weeks >= 0) & (weeks <= 52))
    if not valid_weeks.all():
        rows = np.flatnonzero(~valid_weeks)[:5].tolist()
        raise ValueError(f"ASEC LKWEEKS is outside [-1, 52] at row(s): {rows}.")
    raw[_SOURCE] = weeks.astype(np.int16)
    weights = pd.to_numeric(raw[_AUDIT_WEIGHT_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(weights).all()
        or (weights < 0.0).any()
        or float(weights.sum()) <= 0.0
    ):
        raise ValueError(
            "ASEC weeks-unemployed A_FNLWGT must be finite and nonnegative "
            "with positive total mass."
        )
    normalized = np.where(weeks == -1, 0, weeks).astype(np.int16)
    positive = normalized > 0
    scaled_weights = weights / 100.0
    audit: dict[str, int | float] = {
        "raw_rows": int(len(raw)),
        "unique_keys": unique_keys,
        "positive_rows": int(np.count_nonzero(positive)),
        "niu_rows": int(np.count_nonzero(weeks == -1)),
        "minimum": int(weeks.min()),
        "maximum": int(weeks.max()),
        "weighted_source_share": float(
            scaled_weights[positive].sum() / scaled_weights.sum()
        ),
        "weighted_weeks": float(np.dot(scaled_weights, normalized)),
    }
    pinned_transform = bool(
        expected_member_size_bytes == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SIZE_BYTES
        and expected_member_sha256 == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SHA256
        and expected_rows == ASEC_2023_WEEKS_UNEMPLOYED_RAW_ROWS
        and expected_unique_keys == ASEC_2023_WEEKS_UNEMPLOYED_UNIQUE_KEYS
    )
    if pinned_transform:
        _assert_pinned_source_audit(audit)
    audit["pinned_transform"] = int(pinned_transform)
    result = raw.loc[:, list(ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_COLUMNS)].copy()
    result.attrs["source_audit"] = audit
    return result


def fill_asec_2022_weeks_unemployed_source(
    person: pd.DataFrame,
    source: pd.DataFrame,
) -> pd.DataFrame:
    """Fill only missing 2022 ``LKWEEKS`` via exact Census person identity."""

    required_person = ("source_year", "PERIDNUM")
    missing_person = [column for column in required_person if column not in person]
    if missing_person:
        raise ValueError(
            f"ASEC weeks-unemployed repair requires person column(s): {missing_person}."
        )
    missing_source = [
        column
        for column in ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_COLUMNS
        if column not in source
    ]
    if missing_source:
        raise ValueError(
            f"ASEC weeks-unemployed sidecar missing column(s): {missing_source}."
        )
    result = person.copy(deep=True)
    year = pd.to_numeric(result["source_year"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    valid_year = np.isfinite(year) & (year == np.floor(year))
    if not valid_year.all():
        rows = np.flatnonzero(~valid_year)[:5].tolist()
        raise ValueError(
            f"ASEC weeks-unemployed source_year is invalid at row(s): {rows}."
        )
    repair_mask = year.astype(np.int64) == ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_YEAR
    if not repair_mask.any():
        return result

    donor = source.copy(deep=True)
    donor["PERIDNUM"] = _fixed_width_peridnum(donor["PERIDNUM"], label="ASEC sidecar")
    if donor["PERIDNUM"].duplicated(keep=False).any():
        raise ValueError("ASEC weeks-unemployed sidecar PERIDNUM is not unique.")
    target_keys = _fixed_width_peridnum(
        result.loc[repair_mask, "PERIDNUM"], label="ASEC frame"
    )
    donor = donor.set_index("PERIDNUM")
    missing_keys = target_keys[~target_keys.isin(donor.index)].drop_duplicates()
    if not missing_keys.empty:
        raise ValueError(
            "ASEC weeks-unemployed sidecar does not cover frame PERIDNUM key(s): "
            f"{missing_keys.tolist()[:5]}."
        )
    aligned = donor.reindex(target_keys.to_numpy())
    aligned.index = result.index[repair_mask]

    identity_pairs = [
        (
            "PH_SEQ",
            "source_household_id" if "source_household_id" in result else "PH_SEQ",
        ),
        ("P_SEQ", "P_SEQ"),
        ("A_LINENO", "A_LINENO"),
    ]
    for donor_column, frame_column in identity_pairs:
        if frame_column not in result:
            continue
        observed = pd.to_numeric(
            result.loc[repair_mask, frame_column], errors="coerce"
        ).to_numpy(dtype=np.float64)
        expected = pd.to_numeric(aligned[donor_column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        mismatch = (
            ~np.isfinite(observed) | ~np.isfinite(expected) | (observed != expected)
        )
        if mismatch.any():
            rows = result.index[repair_mask].to_numpy()[mismatch][:5].tolist()
            raise ValueError(
                "ASEC weeks-unemployed redundant identity mismatch for "
                f"{frame_column} against sidecar {donor_column} at row(s): {rows}."
            )

    donor_weeks = _strict_integer(aligned[_SOURCE], label="ASEC sidecar LKWEEKS")
    valid_donor_weeks = (donor_weeks == -1) | ((donor_weeks >= 0) & (donor_weeks <= 52))
    if not valid_donor_weeks.all():
        rows = result.index[repair_mask].to_numpy()[~valid_donor_weeks][:5].tolist()
        raise ValueError(
            f"ASEC sidecar LKWEEKS is outside [-1, 52] at frame row(s): {rows}."
        )
    if _SOURCE not in result:
        result[_SOURCE] = np.nan
    existing = pd.to_numeric(
        result.loc[repair_mask, _SOURCE], errors="coerce"
    ).to_numpy(dtype=np.float64)
    raw_existing = result.loc[repair_mask, _SOURCE]
    present = raw_existing.notna().to_numpy()
    invalid_existing = present & (
        ~np.isfinite(existing) | (existing != np.floor(existing))
    )
    if invalid_existing.any():
        rows = result.index[repair_mask].to_numpy()[invalid_existing][:5].tolist()
        raise ValueError(f"Existing ASEC LKWEEKS is invalid at frame row(s): {rows}.")
    mismatch = present & (existing != donor_weeks)
    if mismatch.any():
        rows = result.index[repair_mask].to_numpy()[mismatch][:5].tolist()
        raise ValueError(
            "Existing ASEC LKWEEKS disagrees with the pinned sidecar at frame "
            f"row(s): {rows}."
        )
    missing = ~present
    repair_indices = result.index[repair_mask].to_numpy()
    if missing.any():
        result.loc[repair_indices[missing], _SOURCE] = donor_weeks[missing]
    result.attrs["weeks_unemployed_source_audit"] = source.attrs.get("source_audit", {})
    return result


def _direct_weeks(values: pd.Series, *, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(numeric) & (numeric == np.floor(numeric))
    valid &= (numeric == -1.0) | ((numeric >= 0.0) & (numeric <= 52.0))
    if not valid.all():
        rows = np.flatnonzero(~valid)[:5].tolist()
        raise SourceRuntimeError(
            f"{label} must contain only finite integer -1 or 0--52 at row(s) {rows}."
        )
    return np.where(numeric == -1.0, 0.0, numeric).astype(np.int16)


def derive_us_weeks_unemployed_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    _context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Apply the retired direct ``LKWEEKS`` carry."""

    if operation.kind != "derive_weeks_unemployed":
        raise SourceRuntimeError(
            "US weeks-unemployed derivation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US weeks-unemployed derivation requires the person table."
        )
    if dict(operation.parameters) != WEEKS_UNEMPLOYED_DERIVE_PARAMETERS:
        raise SourceRuntimeError(
            "US weeks-unemployed derivation parameters drifted from the "
            "archived LKWEEKS carry."
        )
    if _SOURCE not in frame:
        raise SourceRuntimeError(
            f"US weeks-unemployed derivation requires source column {_SOURCE!r}."
        )
    result = frame.copy(deep=True)
    result[_OUTPUT] = _direct_weeks(frame[_SOURCE], label="ASEC LKWEEKS")
    return result


def impute_us_weeks_unemployed_to_puf_support_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """QRF-replace weeks on the PUF support half only."""

    if operation.kind != "impute_weeks_unemployed_to_puf_support":
        raise SourceRuntimeError(
            "US weeks-unemployed PUF imputation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US weeks-unemployed PUF imputation requires the person table."
        )
    if dict(operation.parameters) != WEEKS_UNEMPLOYED_PUF_IMPUTATION_PARAMETERS:
        raise SourceRuntimeError(
            "US weeks-unemployed PUF imputation parameters drifted from the "
            "reviewed QRF contract."
        )
    if not has_support_role_metadata(frame, entity="person"):
        return frame.copy(deep=True)

    roles = support_role_series(frame, entity="person")
    asec_mask = roles.eq(_ASEC_CHANNEL)
    puf_mask = roles.eq(_PUF_CHANNEL)
    if not asec_mask.any() or not puf_mask.any():
        raise SourceRuntimeError(
            "US weeks-unemployed PUF imputation requires nonempty ASEC and "
            "PUF-tax-detail channels."
        )

    required_columns = [
        _PERSON_WEIGHT_COLUMN,
        _OUTPUT,
        *(_PREDICTOR_PREFIX + value for value in _REQUIRED_PUF_PREDICTORS),
    ]
    missing = [column for column in required_columns if column not in frame]
    if missing:
        raise SourceRuntimeError(
            f"US weeks-unemployed PUF imputation is missing column(s): {missing}."
        )
    optional_uc_column = _PREDICTOR_PREFIX + _OPTIONAL_UC_PREDICTOR
    predictors = list(_REQUIRED_PUF_PREDICTORS)
    if optional_uc_column in frame:
        predictors.append(_OPTIONAL_UC_PREDICTOR)
    predictor_columns = [_PREDICTOR_PREFIX + value for value in predictors]

    training = frame.loc[asec_mask, [*predictor_columns, _OUTPUT]].copy()
    training.columns = [*predictors, _OUTPUT]
    test = frame.loc[puf_mask, predictor_columns].copy()
    test.columns = predictors
    weights = pd.to_numeric(
        frame.loc[asec_mask, _PERSON_WEIGHT_COLUMN], errors="coerce"
    )
    weight_values = weights.to_numpy(dtype=np.float64)
    if (
        not np.isfinite(weight_values).all()
        or (weight_values < 0.0).any()
        or float(weight_values.sum()) <= 0.0
    ):
        raise SourceRuntimeError(
            "US weeks-unemployed QRF requires finite, nonnegative typed person "
            "weights with positive total mass."
        )

    max_train_samples = int(operation.parameters["max_train_samples"])
    n_estimators = int(operation.parameters["n_estimators"])
    seed = context.config.seed if context is not None else 0
    if len(training) > max_train_samples:
        sampled_index = training.sample(
            n=max_train_samples,
            random_state=seed,
        ).index
        training = training.loc[sampled_index]
        weights = weights.loc[sampled_index]

    for column in (*predictors, _OUTPUT):
        training[column] = pd.to_numeric(training[column], errors="coerce")
        values = training[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise SourceRuntimeError(
                f"US weeks-unemployed QRF training column {column!r} contains "
                "nonfinite values."
            )
    target = training[_OUTPUT].to_numpy(dtype=np.float64)
    if not ((target == np.floor(target)) & (target >= 0.0) & (target <= 52.0)).all():
        raise SourceRuntimeError(
            "US weeks-unemployed QRF target must be integer-valued in 0--52."
        )
    for column in predictors:
        test[column] = pd.to_numeric(test[column], errors="coerce")
        values = test[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise SourceRuntimeError(
                f"US weeks-unemployed QRF prediction column {column!r} contains "
                "nonfinite values."
            )
        if column == _OPTIONAL_UC_PREDICTOR and (values < 0.0).any():
            raise SourceRuntimeError(
                "US weeks-unemployed unemployment-compensation predictor must "
                "be nonnegative."
            )

    global QRF
    if QRF is None:
        from importlib import import_module

        QRF = import_module("populace.fit").QRF
    fitted = QRF(n_estimators=n_estimators, seed=seed).fit(
        training,
        predictors,
        [_OUTPUT],
        weights=weights.to_numpy(dtype=np.float64),
    )
    predictions = fitted.predict(test)
    if not isinstance(predictions, pd.DataFrame) or _OUTPUT not in predictions:
        raise SourceRuntimeError(
            f"US weeks-unemployed QRF prediction is missing {_OUTPUT!r}."
        )
    if len(predictions) != int(puf_mask.sum()):
        raise SourceRuntimeError(
            "US weeks-unemployed QRF returned "
            f"{len(predictions)} rows; expected {int(puf_mask.sum())}."
        )
    predicted = pd.to_numeric(predictions[_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(predicted).all():
        raise SourceRuntimeError(
            "US weeks-unemployed QRF produced nonfinite predictions."
        )
    lower, upper = operation.parameters["clip"]
    predicted = np.rint(np.clip(predicted, float(lower), float(upper)))
    if _OPTIONAL_UC_PREDICTOR in predictors:
        uc = test[_OPTIONAL_UC_PREDICTOR].to_numpy(dtype=np.float64)
        predicted = np.where(uc > 0.0, predicted, 0.0)
    if not (
        np.isfinite(predicted).all()
        and (predicted == np.floor(predicted)).all()
        and (predicted >= 0.0).all()
        and (predicted <= 52.0).all()
    ):
        raise SourceRuntimeError(
            "US weeks-unemployed postprocessed predictions left the integer "
            "0--52 domain."
        )
    result = frame.copy(deep=True)
    result[_OUTPUT] = pd.to_numeric(result[_OUTPUT], errors="coerce").astype("float64")
    result.loc[puf_mask, _OUTPUT] = predicted
    return result


def _decoded_strings(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: (
            value.decode()
            if isinstance(value, (bytes, bytearray, np.bytes_))
            else value
        )
    ).astype(str)


def _boolean_predictor(values: pd.Series, *, label: str) -> np.ndarray:
    if values.isna().any():
        rows = values.index[values.isna()].tolist()[:5]
        raise SourceRuntimeError(f"{label} is missing at row(s): {rows}.")
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.to_numpy(dtype=bool)
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(numeric) & np.isin(numeric, [0.0, 1.0])
    if not valid.all():
        rows = np.flatnonzero(~valid)[:5].tolist()
        raise SourceRuntimeError(
            f"{label} must be complete boolean values at row(s): {rows}."
        )
    return numeric.astype(bool)


def _person_weeks_unemployed_predictors(frame: Frame) -> pd.DataFrame:
    person = frame.table("person")
    tax_unit = frame.table("tax_unit")
    predictors = pd.DataFrame(index=person.index)

    age_column = "age" if "age" in person else "A_AGE"
    if age_column not in person:
        raise SourceRuntimeError("US weeks-unemployed QRF requires person age.")
    age = pd.to_numeric(person[age_column], errors="coerce").to_numpy(dtype=np.float64)
    if not (np.isfinite(age) & (age >= 0.0) & (age <= 120.0)).all():
        raise SourceRuntimeError("US weeks-unemployed QRF age must be in 0--120.")
    predictors["age"] = age

    if "is_male" in person:
        predictors["is_male"] = _boolean_predictor(person["is_male"], label="is_male")
    elif "is_female" in person:
        predictors["is_male"] = ~_boolean_predictor(
            person["is_female"], label="is_female"
        )
    elif "A_SEX" in person:
        sex = pd.to_numeric(person["A_SEX"], errors="coerce").to_numpy(dtype=np.float64)
        if not (np.isfinite(sex) & np.isin(sex, [1.0, 2.0])).all():
            raise SourceRuntimeError("US weeks-unemployed A_SEX must be 1 or 2.")
        predictors["is_male"] = sex == 1.0
    else:
        raise SourceRuntimeError(
            "US weeks-unemployed QRF requires is_male, is_female, or A_SEX."
        )

    if "tax_unit_is_joint" in person:
        predictors["tax_unit_is_joint"] = _boolean_predictor(
            person["tax_unit_is_joint"], label="tax_unit_is_joint"
        )
    else:
        if "person_tax_unit_id" not in person:
            raise SourceRuntimeError(
                "US weeks-unemployed QRF requires person tax-unit linkage."
            )
        filing_column = next(
            (
                column
                for column in ("filing_status_input", "filing_status")
                if column in tax_unit
            ),
            None,
        )
        if filing_column is None:
            raise SourceRuntimeError(
                "US weeks-unemployed QRF requires tax-unit filing status."
            )
        filing = _decoded_strings(tax_unit[filing_column])
        joint_by_id = pd.Series(
            filing.eq("JOINT").to_numpy(),
            index=tax_unit["tax_unit_id"].to_numpy(),
        )
        joint = person["person_tax_unit_id"].map(joint_by_id)
        if joint.isna().any():
            raise SourceRuntimeError(
                "US weeks-unemployed QRF found missing tax-unit filing linkage."
            )
        predictors["tax_unit_is_joint"] = joint.to_numpy(dtype=bool)

    role_outputs = {
        "is_tax_unit_head": "HEAD",
        "is_tax_unit_spouse": "SPOUSE",
        "is_tax_unit_dependent": "DEPENDENT",
    }
    if all(column in person for column in role_outputs):
        for column in role_outputs:
            predictors[column] = _boolean_predictor(person[column], label=column)
    else:
        if "tax_unit_role_input" not in person:
            raise SourceRuntimeError(
                "US weeks-unemployed QRF requires explicit tax-unit roles or "
                "tax_unit_role_input."
            )
        if person["tax_unit_role_input"].isna().any():
            raise SourceRuntimeError(
                "US weeks-unemployed tax_unit_role_input must be complete."
            )
        role = _decoded_strings(person["tax_unit_role_input"])
        unexpected = sorted(set(role) - set(role_outputs.values()))
        if unexpected:
            raise SourceRuntimeError(
                f"US weeks-unemployed found unsupported tax-unit role(s): {unexpected}."
            )
        for column, value in role_outputs.items():
            predictors[column] = role.eq(value).to_numpy(dtype=bool)

    uc_column: str | None = None
    if _OPTIONAL_UC_PREDICTOR in person:
        uc_column = _OPTIONAL_UC_PREDICTOR
    elif not has_support_role_metadata(person, entity="person") and "UC_VAL" in person:
        uc_column = "UC_VAL"
    if uc_column is not None:
        uc = pd.to_numeric(person[uc_column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not (np.isfinite(uc) & (uc >= 0.0)).all():
            raise SourceRuntimeError(
                "US weeks-unemployed unemployment compensation must be finite "
                "and nonnegative."
            )
        predictors[_OPTIONAL_UC_PREDICTOR] = uc
    return predictors


def _replace_person_table(frame: Frame, person: pd.DataFrame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def with_us_weeks_unemployed(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    asec_2023_source: pd.DataFrame | None = None,
) -> Frame:
    """Repair the 2022 source, carry ASEC weeks, and replace the PUF half."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US weeks unemployed requires the US schema.")
    person = frame.table("person")
    stage_person = person.copy(deep=True)
    has_2022 = False
    if "source_year" in stage_person:
        source_year = pd.to_numeric(stage_person["source_year"], errors="coerce")
        has_2022 = bool(source_year.eq(ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_YEAR).any())
    needs_2022_source = bool(
        has_2022
        and (
            _SOURCE not in stage_person
            or stage_person.loc[
                pd.to_numeric(stage_person["source_year"], errors="coerce").eq(
                    ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_YEAR
                ),
                _SOURCE,
            ]
            .isna()
            .any()
        )
    )
    if asec_2023_source is not None and has_2022:
        stage_person = fill_asec_2022_weeks_unemployed_source(
            stage_person, asec_2023_source
        )
    elif needs_2022_source:
        raise ValueError(
            "US weeks-unemployed stage requires the pinned ASEC 2023 sidecar "
            "to repair missing source-year 2022 LKWEEKS."
        )

    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    if has_support_role_metadata(stage_person, entity="person"):
        # Predictor construction only needs modeled person/tax-unit columns;
        # keep the temporary reserved person_weight column outside Frame.
        predictors = _person_weeks_unemployed_predictors(frame)
        for column in predictors:
            stage_person[_PREDICTOR_PREFIX + column] = predictors[column].to_numpy()
    output = run_source_stage(
        us_weeks_unemployed_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_weeks_unemployed": derive_us_weeks_unemployed_from_manifest,
            "impute_weeks_unemployed_to_puf_support": (
                impute_us_weeks_unemployed_to_puf_support_from_manifest
            ),
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    ).drop(
        columns=[
            _PERSON_WEIGHT_COLUMN,
            *(_PREDICTOR_PREFIX + value for value in WEEKS_UNEMPLOYED_PUF_PREDICTORS),
        ],
        errors="ignore",
    )
    return _replace_person_table(frame, output)


def us_weeks_unemployed_summary(frame: Frame) -> dict[str, object]:
    """Return integer-domain, source-reconciliation, and channel diagnostics."""

    person = frame.table("person")
    if _OUTPUT not in person:
        raise ValueError(f"US weeks-unemployed summary requires {_OUTPUT!r}.")
    values = pd.to_numeric(person[_OUTPUT], errors="coerce").to_numpy(dtype=np.float64)
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    if (
        not np.isfinite(weights).all()
        or (weights < 0.0).any()
        or float(weights.sum()) <= 0.0
    ):
        raise ValueError(
            "US weeks-unemployed summary requires finite nonnegative person "
            "weights with positive total mass."
        )
    finite = np.isfinite(values)
    integer = finite & (values == np.rint(values))
    in_range = integer & (values >= 0.0) & (values <= 52.0)
    positive = in_range & (values > 0.0)

    if has_support_role_metadata(person, entity="person"):
        channel = support_role_series(person, entity="person").to_numpy()
    else:
        channel = np.full(len(person), _ASEC_CHANNEL, dtype=object)
    channels: dict[str, dict[str, float | int]] = {}
    for name in (_ASEC_CHANNEL, _PUF_CHANNEL):
        mask = channel == name
        channel_weight = float(weights[mask].sum())
        channels[name] = {
            "rows": int(np.count_nonzero(mask)),
            "positive_rows": int(np.count_nonzero(mask & positive)),
            "positive_share": (
                float(weights[mask & positive].sum()) / channel_weight
                if channel_weight > 0.0
                else 0.0
            ),
            "weighted_weeks": float(np.dot(weights[mask], np.nan_to_num(values[mask]))),
            "weighted_mean_weeks": (
                float(np.dot(weights[mask], np.nan_to_num(values[mask])))
                / channel_weight
                if channel_weight > 0.0
                else 0.0
            ),
            "positive_share_band": list(_CHANNEL_POSITIVE_SHARE_BANDS[name]),
            "weighted_mean_weeks_band": list(_CHANNEL_WEIGHTED_MEAN_WEEKS_BANDS[name]),
        }

    source_missing = _SOURCE not in person
    source_invalid = source_mismatch = 0
    if not source_missing:
        source_raw = pd.to_numeric(person[_SOURCE], errors="coerce").to_numpy(
            dtype=np.float64
        )
        asec_mask = channel == _ASEC_CHANNEL
        source_valid = np.isfinite(source_raw) & (source_raw == np.floor(source_raw))
        source_valid &= (source_raw == -1.0) | (
            (source_raw >= 0.0) & (source_raw <= 52.0)
        )
        source_invalid = int(np.count_nonzero(asec_mask & ~source_valid))
        expected = np.where(source_raw == -1.0, 0.0, source_raw)
        source_mismatch = int(
            np.count_nonzero(asec_mask & source_valid & finite & (values != expected))
        )
    puf_uc_zero_mismatch = 0
    if _OPTIONAL_UC_PREDICTOR in person:
        uc = pd.to_numeric(person[_OPTIONAL_UC_PREDICTOR], errors="coerce").to_numpy(
            dtype=np.float64
        )
        puf_mask = channel == _PUF_CHANNEL
        puf_uc_zero_mismatch = int(
            np.count_nonzero(
                puf_mask & np.isfinite(uc) & (uc <= 0.0) & finite & (values != 0.0)
            )
        )

    return {
        "rows": int(len(person)),
        "missing_or_nonfinite": int(np.count_nonzero(~finite)),
        "noninteger": int(np.count_nonzero(finite & ~integer)),
        "out_of_range": int(np.count_nonzero(integer & ~in_range)),
        "positive_rows": int(np.count_nonzero(positive)),
        "positive_share": float(weights[positive].sum() / weights.sum()),
        "weighted_weeks": float(np.dot(weights, np.nan_to_num(values))),
        "source_missing": source_missing,
        "source_invalid": source_invalid,
        "source_mismatch_count": source_mismatch,
        "puf_uc_zero_mismatch_count": puf_uc_zero_mismatch,
        "channels": channels,
    }


def us_weeks_unemployed_signal_gate(frame: Frame) -> GateResult:
    """Require exact ASEC carry and integer, nondefault signal on both halves."""

    person = frame.table("person")
    if _OUTPUT not in person:
        return GateResult(
            name="weeks_unemployed_signal",
            passed=False,
            failures=(f"person column missing: {_OUTPUT}.",),
            details={"missing": [_OUTPUT]},
        )
    try:
        summary = us_weeks_unemployed_summary(frame)
    except (TypeError, ValueError) as error:
        return GateResult(
            name="weeks_unemployed_signal",
            passed=False,
            failures=(str(error),),
            details={},
        )
    failures: list[str] = []
    for key, label in (
        ("missing_or_nonfinite", "missing or nonfinite value(s)"),
        ("noninteger", "noninteger value(s)"),
        ("out_of_range", "value(s) outside 0--52"),
    ):
        if int(summary[key]):
            failures.append(f"{_OUTPUT}: {summary[key]} {label}.")
    if bool(summary["source_missing"]):
        failures.append("ASEC LKWEEKS source column is missing.")
    if int(summary["source_invalid"]):
        failures.append(
            f"ASEC LKWEEKS has {summary['source_invalid']} invalid source value(s)."
        )
    if int(summary["source_mismatch_count"]):
        failures.append(
            f"{_OUTPUT} has {summary['source_mismatch_count']} ASEC source "
            "reconciliation mismatch(es)."
        )
    if int(summary["puf_uc_zero_mismatch_count"]):
        failures.append(
            f"{_OUTPUT} has {summary['puf_uc_zero_mismatch_count']} PUF row(s) "
            "positive without unemployment compensation."
        )
    has_support_roles = has_support_role_metadata(person, entity="person")
    required_channels = (
        (_ASEC_CHANNEL, _PUF_CHANNEL) if has_support_roles else (_ASEC_CHANNEL,)
    )
    channels = summary["channels"]
    for name in required_channels:
        channel = channels[name]
        if int(channel["rows"]) == 0:
            failures.append(f"{name} weeks-unemployed support is empty.")
        elif (
            int(channel["positive_rows"]) == 0 or float(channel["weighted_weeks"]) <= 0
        ):
            failures.append(f"{name} weeks_unemployed is default-only.")
        else:
            positive_share = float(channel["positive_share"])
            positive_low, positive_high = channel["positive_share_band"]
            if not (positive_low <= positive_share <= positive_high):
                failures.append(
                    f"{name} weeks_unemployed positive share "
                    f"{positive_share:.6f} outside plausibility band "
                    f"[{positive_low}, {positive_high}]."
                )
            weighted_mean = float(channel["weighted_mean_weeks"])
            mean_low, mean_high = channel["weighted_mean_weeks_band"]
            if not (mean_low <= weighted_mean <= mean_high):
                failures.append(
                    f"{name} weeks_unemployed weighted mean weeks "
                    f"{weighted_mean:.6f} outside plausibility band "
                    f"[{mean_low}, {mean_high}]."
                )
    return GateResult(
        name="weeks_unemployed_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
