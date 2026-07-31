"""PolicyEngine-compatible US H5 I/O for nullable build artifacts.

The writer stores entity tables, household weights, the period, and a small
artifact metadata record.  Fixed-format pandas tables preserve nullable
object columns without filling or coercing measured values.  The companion
build manifest, rather than this consumer H5, owns stage receipts such as
``Frame.metadata`` and ``Frame.mass_log``.
"""

from __future__ import annotations

import json
import os
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from populace.frame import Frame, WeightKind, Weights
from populace.frame.units import US_SCHEMA

__all__ = [
    "LEGACY_NULLABLE_STAGING_ARTIFACT_KIND",
    "load_legacy_calibrated_us_h5",
    "write_nullable_us_h5",
]

LEGACY_NULLABLE_STAGING_ARTIFACT_KIND = "nullable_precalibration_staging_h5"
_METADATA_KEY = "_populace_staging_metadata"
_TIME_PERIOD_KEY = "_time_period"


def load_legacy_calibrated_us_h5(path: str | Path) -> Frame:
    """Load a legacy US single-year H5 as a calibrated-weight frame.

    Legacy PolicyEngine US artifacts do not expose typed weight provenance
    through ``USSingleYearDataset``.  This loader therefore preserves the
    historical builder contract and labels their household weights
    ``CALIBRATED``.  It is not the loader for the new pre-calibration
    multispine pool, whose importance-weight receipt lives in its manifest.
    """

    from policyengine_us.data import USSingleYearDataset

    dataset = USSingleYearDataset(file_path=str(Path(path)))
    tables = {
        "person": dataset.person,
        "household": dataset.household.copy(),
        "tax_unit": dataset.tax_unit,
        "spm_unit": dataset.spm_unit,
        "family": dataset.family,
        "marital_unit": dataset.marital_unit,
    }
    household_weights = (
        tables["household"].pop("household_weight").to_numpy(dtype=np.float64)
    )
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                household_weights,
                WeightKind.CALIBRATED,
            )
        },
    )


def write_nullable_us_h5(
    frame: Frame,
    path: str | Path,
    *,
    period: int,
    artifact_kind: str,
    publication_run_id: str | None = None,
) -> None:
    """Atomically write and verify a nullable US single-year H5.

    The destination is replaced only after a temporary sibling has round-trip
    verified every nonempty entity table, household weights, period metadata,
    fixed-format storage, and the caller-declared ``artifact_kind``.  A failed
    write or verification leaves any existing destination bytes untouched.
    """

    if not isinstance(frame, Frame):
        raise TypeError(f"frame must be a Frame, got {type(frame).__name__}.")
    if not isinstance(artifact_kind, str) or not artifact_kind.strip():
        raise ValueError("artifact_kind must be a non-empty string.")
    if publication_run_id is not None and (
        not isinstance(publication_run_id, str) or not publication_run_id.strip()
    ):
        raise ValueError("publication_run_id must be a non-empty string when set.")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_nullable_us_h5_file(
            frame,
            temporary,
            period=int(period),
            artifact_kind=artifact_kind,
            publication_run_id=publication_run_id,
        )
        _verify_nullable_us_h5(
            frame,
            temporary,
            period=int(period),
            artifact_kind=artifact_kind,
            publication_run_id=publication_run_id,
        )
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_nullable_us_h5_file(
    frame: Frame,
    path: Path,
    *,
    period: int,
    artifact_kind: str,
    publication_run_id: str | None,
) -> None:
    with pd.HDFStore(path, mode="w") as store:
        for entity in frame.entities:
            table = _export_table(frame, entity)
            if not len(table):
                continue
            # Fixed format preserves mixed bool/null object columns
            # losslessly. Table format rejects them, which would force a
            # fill or type rewrite.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
                store.put(entity, table, format="fixed")
        store.put(
            _TIME_PERIOD_KEY,
            pd.Series([period]),
            format="table",
        )
        store.put(
            _METADATA_KEY,
            pd.Series(
                [
                    json.dumps(
                        _artifact_metadata(
                            frame,
                            artifact_kind=artifact_kind,
                            publication_run_id=publication_run_id,
                        ),
                        sort_keys=True,
                    )
                ]
            ),
            format="table",
        )


def _verify_nullable_us_h5(
    frame: Frame,
    path: Path,
    *,
    period: int,
    artifact_kind: str,
    publication_run_id: str | None,
) -> None:
    with pd.HDFStore(path, mode="r") as store:
        for entity in frame.entities:
            expected = _export_table(frame, entity)
            if not len(expected):
                continue
            try:
                stored = store[entity]
            except KeyError as exc:
                raise RuntimeError(
                    f"Nullable US H5 round trip omitted entity {entity!r}."
                ) from exc
            try:
                pd.testing.assert_frame_equal(
                    stored,
                    expected,
                    check_exact=True,
                )
            except AssertionError as exc:
                raise RuntimeError(
                    f"Nullable US H5 round trip changed entity {entity!r}: {exc}"
                ) from exc
            if store.get_storer(entity).is_table:
                raise RuntimeError(
                    f"Nullable US H5 stored entity {entity!r} in table format."
                )

        stored_period = store[_TIME_PERIOD_KEY]
        if stored_period.tolist() != [period]:
            raise RuntimeError(
                "Nullable US H5 round trip changed the time period: "
                f"expected {period}, got {stored_period.tolist()}."
            )
        raw_metadata = store[_METADATA_KEY]
        if len(raw_metadata) != 1:
            raise RuntimeError(
                "Nullable US H5 must carry exactly one artifact metadata row."
            )
        try:
            stored_metadata = json.loads(str(raw_metadata.iloc[0]))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Nullable US H5 artifact metadata is not valid JSON."
            ) from exc
        expected_metadata = _artifact_metadata(
            frame,
            artifact_kind=artifact_kind,
            publication_run_id=publication_run_id,
        )
        if stored_metadata != expected_metadata:
            raise RuntimeError(
                "Nullable US H5 round trip changed artifact metadata: "
                f"expected {expected_metadata}, got {stored_metadata}."
            )


def _export_table(frame: Frame, entity: str) -> pd.DataFrame:
    table = frame.table(entity)
    if entity != "household":
        return table
    household = table.copy()
    household["household_weight"] = frame.weights_for("household").values
    return household


def _artifact_metadata(
    frame: Frame,
    *,
    artifact_kind: str,
    publication_run_id: str | None,
) -> dict[str, str]:
    metadata = {
        "artifact_kind": artifact_kind,
        "entity_hdf_format": "fixed_nullable",
        "household_weight_kind": frame.weights_for("household").kind.value,
    }
    if publication_run_id is not None:
        metadata["publication_run_id"] = publication_run_id
    return metadata
