"""Opt-in logical verification of a complete-SPM-input PolicyEngine-US H5 export.

This compares a file with supplied Frames and calibration arguments. It does
not authenticate their source, run a calibration, or qualify a release. The
caller still owns graph ancestry, target and scope interpretation, SPM settings,
and the lifetime of those inputs. Existing adapter defaults remain unchanged.

The admitted codec change is complete pandas BooleanDtype to NumPy bool. Numeric
dtypes and complete Python StringDtype must otherwise match exactly. Missing
roles, nullable integers, object columns and missing strings are unsupported.
Masked NumPy float NaN payloads may change, but masks and known bytes may not.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.frame.adapters.policyengine_us import _HOUSEHOLD_WEIGHT_COLUMN
from microcosm.frame.materialize import _WEIGHT_COLUMN_SUFFIX, read_frame_table

from ..spm_input_contract import ROLE_INPUT, UNIVERSE_INPUT, UNIVERSE_STATUSES
from .common_frame_export_contract import _same_series, verify_retained_frame_export
from .population_input_coverage import _frame_stamp

if TYPE_CHECKING:
    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine


class PolicyEngineH5ReadbackError(ValueError):
    """The supplied export or its logical H5 readback is not exact."""


@dataclass(frozen=True)
class PolicyEngineH5ReadbackReceipt:
    """A file comparison receipt, never source or release authority.

    Only tables, materialized household weights and period came from H5.
    The other Frame context is supplied and sealed across this operation.
    """

    binding: bytes
    sha256: str
    period: int
    normalizations: tuple[str, ...]
    nonserialized_context: tuple[str, ...] = (
        "strata",
        "schema",
        "frame_metadata",
        "typed_weight_kind",
        "mass_log",
        "SPM_settings",
    )
    source_ancestry_verified: bool = False
    release_eligible: bool = False


@dataclass(frozen=True)
class _FileSeal:
    device: int
    inode: int
    size: int
    sha256: str


def _require(condition, reason):
    if not condition:
        raise PolicyEngineH5ReadbackError("H5_" + reason)


def _dtype_profile(series):
    dtype = series.dtype
    if isinstance(dtype, np.dtype):
        _require(dtype.kind in "biuf", "UNSUPPORTED_DTYPE")
        return ("numpy", dtype.str)
    if isinstance(dtype, pd.BooleanDtype):
        _require(not series.isna().any(), "MISSING_BOOLEAN")
        return ("complete_boolean",)
    _require(
        isinstance(dtype, pd.StringDtype)
        and dtype.storage == "python"
        and dtype.na_value is pd.NA,
        "UNSUPPORTED_DTYPE",
    )
    _require(
        not series.isna().any() and all(type(value) is str for value in series),
        "MISSING_OR_NONSTRING_VALUE",
    )
    return ("complete_python_string", "pd.NA")


def _string_policy(dtype):
    # The shared content stamp records string values, not their storage backend.
    if isinstance(dtype, pd.StringDtype):
        return (dtype.storage, dtype.na_value is pd.NA)
    return None


def _frame_profile(frame, *, calibrated):
    _require(isinstance(frame, Frame) and frame.schema == US_SCHEMA, "US_SCHEMA")
    _require(frame.weighted_entities == ("household",), "WEIGHT_TOPOLOGY")
    _require(
        not calibrated or frame.weights_for("household").kind is WeightKind.CALIBRATED,
        "CALIBRATED_WEIGHT_KIND",
    )
    _require(ROLE_INPUT in frame.person, "SPM_ROLE_REQUIRED")
    role = frame.person[ROLE_INPUT]
    _require(
        (role.dtype == np.dtype("bool") or isinstance(role.dtype, pd.BooleanDtype))
        and not role.isna().any(),
        "COMPLETE_BOOLEAN_SPM_ROLE",
    )
    spm = frame.table("spm_unit")
    _require(UNIVERSE_INPUT in spm, "SPM_SCOPE_REQUIRED")
    _require(
        isinstance(spm[UNIVERSE_INPUT].dtype, pd.StringDtype)
        and spm[UNIVERSE_INPUT].isin(UNIVERSE_STATUSES).all(),
        "CANONICAL_SPM_SCOPE",
    )
    reserved = {entity + _WEIGHT_COLUMN_SUFFIX for entity in US_SCHEMA.entities}
    result = []
    for entity in US_SCHEMA.entities:
        table = frame.table(entity)
        _require(len(table) > 0 and not table.columns.has_duplicates, "TABLE_SHAPE")
        _require(all(type(name) is str for name in table), "COLUMN_NAMES")
        _require(not reserved.intersection(table.columns), "SOURCE_WEIGHT_COLUMN")
        structural = [US_SCHEMA.entity_id_column(entity)]
        if entity == US_SCHEMA.person_entity:
            structural += [
                US_SCHEMA.membership_column(group) for group in US_SCHEMA.group_entities
            ]
        for column in structural:
            _require(column in table, "STRUCTURAL_COLUMN")
            series = table[column]
            _require(
                isinstance(series.dtype, np.dtype)
                and series.dtype.kind in "iu"
                and not series.isna().any(),
                "INTEGER_IDS_AND_MEMBERSHIPS",
            )
        result.append(
            (
                entity,
                tuple((name, _dtype_profile(table[name])) for name in table),
                _string_policy(table.index.dtype),
                _string_policy(table.columns.dtype),
            )
        )
    # Validate mutable Frames without normalizing the caller's strata in place.
    Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
        metadata=frame.metadata,
        mass_log=frame.mass_log,
    )
    return (
        tuple(result),
        _string_policy(frame.strata.dtype),
        _string_policy(frame.strata.index.dtype),
    )


def _verify_period(stored_period, period):
    _require(type(period) is int and period > 0, "PERIOD_ARGUMENT")
    _require(
        isinstance(stored_period, pd.Series)
        and len(stored_period) == 1
        and isinstance(stored_period.dtype, np.dtype)
        and stored_period.dtype.kind in "iu"
        and int(stored_period.iloc[0]) == period,
        "STORED_PERIOD",
    )


def verify_policyengine_h5_readback(
    candidate: Frame,
    tables: Mapping[str, pd.DataFrame],
    stored_period: pd.Series,
    *,
    period: int,
) -> tuple[Frame, tuple[str, ...]]:
    """Compare detached logical tables; this helper does no I/O.

    Requires complete boolean SPM roles and explicit canonical scope strings;
    UNRESOLVED is preserved, not upgraded to measurement readiness. The returned
    Frame carries actual read-back values and household weights.
    Its strata, schema, metadata, weight kind and mass log are supplied context.
    Incidental pandas row indices may reset; entity-ID order must not change.
    This pure helper alone does not attest that a file was written or read.
    """
    _frame_profile(candidate, calibrated=True)
    _verify_period(stored_period, period)
    _require(isinstance(tables, Mapping), "TABLE_MAPPING")
    _require(set(tables) == set(US_SCHEMA.entities), "TABLE_ROSTER")
    restored, normalizations = {}, []
    for entity in US_SCHEMA.entities:
        expected, actual = candidate.table(entity), tables[entity]
        columns = tuple(expected.columns)
        if entity == "household":
            columns += (_HOUSEHOLD_WEIGHT_COLUMN,)
        _require(
            isinstance(actual, pd.DataFrame)
            and not actual.columns.has_duplicates
            and tuple(actual.columns) == columns
            and len(actual) == len(expected),
            "COLUMN_OR_ROW_ROSTER:" + entity,
        )
        result = actual.copy(deep=True)
        for column in expected:
            left, right = expected[column], actual[column]
            label = entity + "." + column
            if isinstance(left.dtype, pd.BooleanDtype):
                _require(
                    right.dtype == np.dtype("bool")
                    and np.array_equal(left.to_numpy(dtype=np.bool_), right.to_numpy()),
                    "BOOLEAN_CODEC:" + label,
                )
                result[column] = right.astype(left.dtype)
                normalizations.append(label + ":bool->boolean")
            else:
                _require(
                    _dtype_profile(left) == _dtype_profile(right), "DTYPE:" + label
                )
            _require(
                _same_series(left, result[column], readback=True), "VALUE:" + label
            )
        restored[entity] = result
    actual_weights = restored["household"].pop(_HOUSEHOLD_WEIGHT_COLUMN).to_numpy()
    expected_weights = candidate.weights_for("household").values
    _require(
        actual_weights.dtype == np.dtype("float64")
        and actual_weights.tobytes() == expected_weights.tobytes(),
        "HOUSEHOLD_WEIGHTS",
    )
    # H5 can reset indices, so supplied strata follow the verified person order.
    strata = candidate.strata.copy(deep=True)
    strata.index = restored[US_SCHEMA.person_entity].index.copy()
    frame = Frame(
        restored,
        candidate.schema,
        {"household": Weights(actual_weights, WeightKind.CALIBRATED)},
        strata,
        metadata=candidate.metadata,
        mass_log=candidate.mass_log,
    )
    return frame, tuple(normalizations)


def _read_h5(path):
    with pd.HDFStore(str(path), mode="r") as store:
        _require(
            set(store.keys())
            == {"/" + entity for entity in US_SCHEMA.entities} | {"/_time_period"},
            "FILE_TABLE_ROSTER",
        )
        return (
            {entity: read_frame_table(store, entity) for entity in US_SCHEMA.entities},
            store["_time_period"],
        )


def _file_seal(path):
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "REGULAR_FILE")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        opened = os.fstat(source.fileno())
        _require(
            (before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino),
            "FILE_REPLACED",
        )
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        after_read = os.fstat(source.fileno())
    after_path = path.lstat()
    _require(
        all(
            stat.S_ISREG(item.st_mode)
            and item.st_nlink == 1
            and (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
            == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            for item in (opened, after_read, after_path)
        ),
        "FILE_CHANGED",
    )
    return _FileSeal(before.st_dev, before.st_ino, before.st_size, digest.hexdigest())


def write_verified_policyengine_h5_export(
    parent: Frame,
    candidate: Frame,
    engine: PolicyEngineUSEngine,
    path: str | Path,
    *,
    period: int,
    parent_reference: str,
    ordered_household_ids: np.ndarray,
    calibrated_weights: np.ndarray,
    calibration_specification: bytes,
    scope_household_ids: np.ndarray | None = None,
    prune_zero_weight: bool = True,
) -> PolicyEngineH5ReadbackReceipt:
    """Write once with the maintained adapter, then compare exact logical data.

    Requires complete SPM roles and scope, and a fresh .h5 destination in an
    existing caller-owned directory.
    On failure no success receipt is returned; a partially written file is
    retained for diagnosis and must not be consumed. No unrelated path is
    removed. File/Frame seals are cooperative mutation checks, not a sandbox.
    Runtime and population-scale feasibility are the caller's responsibility.
    """
    _require(type(period) is int and period > 0, "PERIOD_ARGUMENT")
    output = Path(path).absolute()
    _require(output.suffix == ".h5", "FILE_SUFFIX")
    _require(output.parent.is_dir(), "OUTPUT_DIRECTORY")
    _require(not output.exists() and not output.is_symlink(), "FRESH_DESTINATION")
    profiles = (
        _frame_profile(parent, calibrated=False),
        _frame_profile(candidate, calibrated=True),
    )
    args = {
        "parent_reference": parent_reference,
        "ordered_household_ids": ordered_household_ids,
        "calibrated_weights": calibrated_weights,
        "calibration_specification": calibration_specification,
        "scope_household_ids": scope_household_ids,
        "prune_zero_weight": prune_zero_weight,
    }
    binding = verify_retained_frame_export(parent, candidate, **args)
    stamps = (_frame_stamp(parent), _frame_stamp(candidate))
    engine.write_dataset(candidate, output, period=period)
    before = _file_seal(output)
    tables, saved_period = _read_h5(output)
    after = _file_seal(output)
    # No foreign I/O follows this point. Check inputs and detached readback
    # after the final file read, including excluded-row arguments in binding.
    _require(before == after, "FILE_CHANGED_DURING_READBACK")
    _require(
        profiles
        == (
            _frame_profile(parent, calibrated=False),
            _frame_profile(candidate, calibrated=True),
        )
        and stamps == (_frame_stamp(parent), _frame_stamp(candidate)),
        "INPUT_CHANGED_DURING_IO",
    )
    verify_retained_frame_export(parent, candidate, expected_binding=binding, **args)
    readback, normalizations = verify_policyengine_h5_readback(
        candidate, tables, saved_period, period=period
    )
    verify_retained_frame_export(
        parent,
        readback,
        comparison="frame-checkpoint-readback",
        expected_binding=binding,
        **args,
    )
    return PolicyEngineH5ReadbackReceipt(binding, after.sha256, period, normalizations)
