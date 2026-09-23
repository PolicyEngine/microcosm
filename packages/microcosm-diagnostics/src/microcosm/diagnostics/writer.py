"""The only filesystem writer for ``calibration_diagnostics.json``."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from microcosm.diagnostics.schema import (
    CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION,
    CalibrationDiagnosticsV8,
)

_LOGGER = logging.getLogger(__name__)

DiagnosticsFailureCode = Literal[
    "construction_error",
    "validation_error",
    "serialization_error",
    "io_error",
    "unexpected_error",
]


@dataclass(frozen=True)
class DiagnosticsWriteSuccess:
    status: Literal["available"]
    path: Path
    schema_version: Literal[8]
    sha256: str


@dataclass(frozen=True)
class DiagnosticsWriteFailure:
    status: Literal["failed"]
    expected_schema_version: Literal[8]
    error_code: DiagnosticsFailureCode
    message: str


DiagnosticsWriteOutcome = DiagnosticsWriteSuccess | DiagnosticsWriteFailure


def _failure_message(error: Exception) -> str:
    message = " ".join(str(error).split())
    rendered = f"{type(error).__name__}: {message}" if message else type(error).__name__
    return rendered[:500]


def _remove_without_raising(path: Path) -> None:
    """Best-effort cleanup that cannot turn diagnostics failure into build failure."""

    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        _LOGGER.warning(
            "Could not remove incomplete calibration diagnostics at %s: %s",
            path,
            _failure_message(error),
        )


def failed_diagnostics_outcome(
    error: Exception,
    *,
    error_code: DiagnosticsFailureCode = "construction_error",
    path: Path | str | None = None,
) -> DiagnosticsWriteFailure:
    """Create and log a sanitized failure result, removing stale output."""

    if path is not None:
        _remove_without_raising(Path(path))
    outcome = DiagnosticsWriteFailure(
        status="failed",
        expected_schema_version=CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION,
        error_code=error_code,
        message=_failure_message(error),
    )
    _LOGGER.warning(
        "Calibration diagnostics were not produced [%s]: %s",
        outcome.error_code,
        outcome.message,
    )
    return outcome


def write_calibration_diagnostics(
    diagnostics: CalibrationDiagnosticsV8,
    path: Path | str,
) -> DiagnosticsWriteOutcome:
    """Validate and atomically write diagnostics without failing its dataset build."""

    output = Path(path)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        validated = CalibrationDiagnosticsV8.model_validate(diagnostics)
        payload = validated.model_dump(mode="python", exclude_defaults=True)
        encoded = json.dumps(
            payload,
            indent=1,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        temporary.write_bytes(encoded)
        os.replace(temporary, output)
    except ValidationError as error:
        return failed_diagnostics_outcome(
            error,
            error_code="validation_error",
            path=output,
        )
    except (TypeError, ValueError) as error:
        return failed_diagnostics_outcome(
            error,
            error_code="serialization_error",
            path=output,
        )
    except OSError as error:
        return failed_diagnostics_outcome(error, error_code="io_error", path=output)
    except Exception as error:  # diagnostics must never abort the dataset build
        return failed_diagnostics_outcome(
            error,
            error_code="unexpected_error",
            path=output,
        )
    finally:
        _remove_without_raising(temporary)
    return DiagnosticsWriteSuccess(
        status="available",
        path=output,
        schema_version=CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION,
        sha256=hashlib.sha256(encoded).hexdigest(),
    )
