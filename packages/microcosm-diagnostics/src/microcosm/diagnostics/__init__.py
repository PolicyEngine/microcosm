"""Typed calibration diagnostics shared across the Microcosm stack."""

from microcosm.diagnostics.schema import (
    CALIBRATION_DIAGNOSTICS_ADAPTER,
    CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION,
    SUPPORTED_CALIBRATION_DIAGNOSTICS_SCHEMA_VERSIONS,
    CalibrationDiagnostics,
    CalibrationDiagnosticsV8,
    calibration_diagnostics_json_schema,
    parse_calibration_diagnostics,
)
from microcosm.diagnostics.writer import (
    DiagnosticsFailureCode,
    DiagnosticsWriteFailure,
    DiagnosticsWriteOutcome,
    DiagnosticsWriteSuccess,
    failed_diagnostics_outcome,
    write_calibration_diagnostics,
)

__all__ = [
    "CALIBRATION_DIAGNOSTICS_ADAPTER",
    "CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION",
    "SUPPORTED_CALIBRATION_DIAGNOSTICS_SCHEMA_VERSIONS",
    "CalibrationDiagnostics",
    "CalibrationDiagnosticsV8",
    "DiagnosticsFailureCode",
    "DiagnosticsWriteFailure",
    "DiagnosticsWriteOutcome",
    "DiagnosticsWriteSuccess",
    "calibration_diagnostics_json_schema",
    "failed_diagnostics_outcome",
    "parse_calibration_diagnostics",
    "write_calibration_diagnostics",
]

__version__ = "0.1.0"
