"""National UK build orchestration with a hard final input-coverage gate.

This module is deliberately table-oriented. UK source stages operate on the
same person, benunit, and household tables persisted by a PolicyEngine-UK
single-year H5, including ``household_weight`` as a real export column. The
local-geography clone remains a separate downstream build product.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.uk_runtime.release_input_coverage import (
    PolicyEngineUKCoverageEngine,
    assert_uk_release_input_coverage_build_stages,
    assert_uk_release_input_coverage_manifest_current,
    uk_release_input_coverage_gate,
)
from populace.frame import MassChangeRecord, WeightKind

__all__ = [
    "UKNationalBuildResult",
    "UKNationalDataset",
    "UKNationalStage",
    "build_uk_national_dataset",
    "load_uk_national_dataset",
    "validate_uk_national_dataset",
    "write_uk_national_dataset",
]

UK_NATIONAL_H5_TABLES = ("person", "benunit", "household", "time_period")
UK_HOUSEHOLD_WEIGHT_KIND_ATTR = "populace_household_weight_kind"
UK_MASS_LOG_ATTR = "populace_mass_log_json"


@dataclass(frozen=True)
class _UKSourceFileFingerprint:
    """Cheap stable-file identity used to bind a prior hash to an H5 load."""

    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


def _uk_source_file_fingerprint(path: Path) -> _UKSourceFileFingerprint:
    stat = path.stat()
    return _UKSourceFileFingerprint(
        device=stat.st_dev,
        inode=stat.st_ino,
        size_bytes=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        changed_ns=stat.st_ctime_ns,
    )


@dataclass(frozen=True)
class UKNationalDataset:
    """Explicit entity tables at one point in the national build pipeline."""

    person: pd.DataFrame
    benunit: pd.DataFrame
    household: pd.DataFrame
    time_period: str
    household_weight_kind: WeightKind = WeightKind.DESIGN
    mass_log: tuple[MassChangeRecord, ...] = ()
    _source_h5: Path | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _source_file_fingerprint: _UKSourceFileFingerprint | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def source_h5(self) -> Path | None:
        """Resolved source path set by the H5 loader, never by table callers."""

        return self._source_h5

    @property
    def source_file_fingerprint(self) -> _UKSourceFileFingerprint | None:
        """Stable identity of the file bytes opened by the H5 loader."""

        return self._source_file_fingerprint

    def with_tables(
        self,
        *,
        person: pd.DataFrame | None = None,
        benunit: pd.DataFrame | None = None,
        household: pd.DataFrame | None = None,
        time_period: int | str | None = None,
        household_weight_kind: WeightKind | None = None,
        mass_log: tuple[MassChangeRecord, ...] | None = None,
    ) -> UKNationalDataset:
        """Return a dataset with selected tables replaced."""

        result = UKNationalDataset(
            person=self.person if person is None else person,
            benunit=self.benunit if benunit is None else benunit,
            household=self.household if household is None else household,
            time_period=(self.time_period if time_period is None else str(time_period)),
            household_weight_kind=(
                self.household_weight_kind
                if household_weight_kind is None
                else household_weight_kind
            ),
            mass_log=self.mass_log if mass_log is None else tuple(mass_log),
        )
        object.__setattr__(result, "_source_h5", self._source_h5)
        object.__setattr__(
            result,
            "_source_file_fingerprint",
            self._source_file_fingerprint,
        )
        return result


@dataclass(frozen=True)
class UKNationalStage:
    """One named, deterministic national table transform."""

    name: str
    transform: Callable[[UKNationalDataset], UKNationalDataset]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("UKNationalStage.name must be non-empty.")
        if not callable(self.transform):
            raise TypeError("UKNationalStage.transform must be callable.")

    def run(self, dataset: UKNationalDataset) -> UKNationalDataset:
        """Apply this stage and require an explicit UK dataset result."""

        result = self.transform(dataset)
        if not isinstance(result, UKNationalDataset):
            raise TypeError(
                f"UK national stage {self.name!r} must return UKNationalDataset, "
                f"got {type(result).__name__}."
            )
        return result


@dataclass(frozen=True)
class UKNationalBuildResult:
    """A gated national staging artifact and its execution evidence."""

    dataset: UKNationalDataset
    input_h5: Path
    staging_h5: Path
    stage_names: tuple[str, ...]
    input_coverage: GateResult
    input_coverage_path: Path | None = None


def load_uk_national_dataset(path: str | Path) -> UKNationalDataset:
    """Load and validate a compact UK single-year H5."""

    input_path = Path(path).expanduser().resolve()
    if input_path.suffix != ".h5":
        raise ValueError("UK national dataset path must end with '.h5'.")
    if not input_path.is_file():
        raise FileNotFoundError(f"UK national dataset not found: {input_path}.")

    fingerprint_before = _uk_source_file_fingerprint(input_path)
    stored_kind, stored_mass_log = _read_weight_metadata(input_path)
    with pd.HDFStore(input_path, mode="r") as store:
        keys = {key.lstrip("/") for key in store.keys()}
        missing = sorted(set(UK_NATIONAL_H5_TABLES) - keys)
        if missing:
            raise ValueError(f"UK national dataset is missing table(s): {missing}.")
        raw_period = store["time_period"]
        if len(raw_period) != 1:
            raise ValueError(
                "UK national dataset time_period must contain exactly one value."
            )
        dataset = UKNationalDataset(
            person=store["person"],
            benunit=store["benunit"],
            household=store["household"],
            time_period=str(raw_period.iloc[0]),
            household_weight_kind=_weight_kind_from_stored(stored_kind),
            mass_log=_mass_log_from_stored(stored_mass_log),
        )
    fingerprint_after = _uk_source_file_fingerprint(input_path)
    if fingerprint_after != fingerprint_before:
        raise RuntimeError(
            "UK national source H5 changed while it was being loaded; refusing "
            "to bind mixed or stale bytes to build stages."
        )
    object.__setattr__(dataset, "_source_h5", input_path)
    object.__setattr__(dataset, "_source_file_fingerprint", fingerprint_after)
    validate_uk_national_dataset(dataset)
    return dataset


def validate_uk_national_dataset(dataset: UKNationalDataset) -> None:
    """Validate UK entity IDs, memberships, weights, and period metadata."""

    if not isinstance(dataset, UKNationalDataset):
        raise TypeError(
            "UK national stages must operate on UKNationalDataset instances."
        )
    for name in ("person", "benunit", "household"):
        if not isinstance(getattr(dataset, name), pd.DataFrame):
            raise TypeError(f"UK national {name} table must be a pandas DataFrame.")
    if not isinstance(dataset.time_period, str) or not dataset.time_period.strip():
        raise ValueError("UK national dataset time_period must be a non-empty string.")
    if not isinstance(dataset.household_weight_kind, WeightKind):
        raise TypeError(
            "UK national dataset household_weight_kind must be a WeightKind."
        )
    if not isinstance(dataset.mass_log, tuple) or any(
        not isinstance(record, MassChangeRecord) for record in dataset.mass_log
    ):
        raise TypeError(
            "UK national dataset mass_log must be a tuple of MassChangeRecord."
        )

    _require_columns(
        dataset.person,
        ("person_id", "person_household_id", "person_benunit_id"),
        label="person",
    )
    _require_columns(dataset.benunit, ("benunit_id",), label="benunit")
    _require_columns(
        dataset.household,
        ("household_id", "household_weight"),
        label="household",
    )
    _require_unique(dataset.person, "person_id", label="person")
    _require_unique(dataset.benunit, "benunit_id", label="benunit")
    _require_unique(dataset.household, "household_id", label="household")

    missing_households = sorted(
        set(dataset.person["person_household_id"])
        - set(dataset.household["household_id"])
    )
    if missing_households:
        raise ValueError(
            "person.person_household_id contains value(s) absent from household: "
            f"{missing_households[:5]}."
        )
    missing_benunits = sorted(
        set(dataset.person["person_benunit_id"]) - set(dataset.benunit["benunit_id"])
    )
    if missing_benunits:
        raise ValueError(
            "person.person_benunit_id contains value(s) absent from benunit: "
            f"{missing_benunits[:5]}."
        )

    weights = pd.to_numeric(dataset.household["household_weight"], errors="coerce")
    if weights.isna().any() or not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("household.household_weight must contain finite numbers.")
    if (weights < 0).any():
        raise ValueError("household.household_weight must be non-negative.")
    if not (weights > 0).any():
        raise ValueError(
            "household.household_weight must retain at least one positive value."
        )
    household_records = [
        record for record in dataset.mass_log if record.entity == "household"
    ]
    if household_records and not np.isclose(
        household_records[-1].new_total,
        float(weights.sum()),
        rtol=1e-9,
        atol=0.0,
    ):
        raise ValueError(
            "UK national dataset household_weight total disagrees with the "
            "latest household MassChangeRecord."
        )


def write_uk_national_dataset(
    dataset: UKNationalDataset,
    path: str | Path,
) -> Path:
    """Atomically write a validated UK national single-year staging H5."""

    validate_uk_national_dataset(dataset)
    output_path = Path(path)
    if output_path.suffix != ".h5":
        raise ValueError("UK national staging path must end with '.h5'.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp.h5"
    )
    try:
        with pd.HDFStore(temporary_path) as store:
            store.put("person", dataset.person, format="table", data_columns=True)
            store.put("benunit", dataset.benunit, format="table", data_columns=True)
            store.put("household", dataset.household, format="table", data_columns=True)
            store.put(
                "time_period",
                pd.Series([dataset.time_period]),
                format="table",
                data_columns=True,
            )
        _write_weight_metadata(temporary_path, dataset)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def build_uk_national_dataset(
    *,
    input_h5: str | Path,
    staging_h5: str | Path,
    stages: Sequence[UKNationalStage] = (),
    coverage_engine: Any | None = None,
    input_coverage_path: str | Path | None = None,
) -> UKNationalBuildResult:
    """Run ordered national stages, hard-gate the result, and stage an H5."""

    input_path = Path(input_h5).resolve()
    staging_path = Path(staging_h5).resolve()
    if input_path == staging_path:
        raise ValueError("input_h5 and staging_h5 must differ.")
    if staging_path.suffix != ".h5":
        raise ValueError("UK national staging path must end with '.h5'.")

    diagnostic_path = (
        Path(input_coverage_path).resolve() if input_coverage_path is not None else None
    )
    if diagnostic_path in {input_path, staging_path}:
        raise ValueError(
            "input_coverage_path must differ from the input and staging H5 paths."
        )

    materialized_stages = tuple(stages)
    _validate_stages(materialized_stages)
    staging_path.unlink(missing_ok=True)
    if diagnostic_path is not None:
        diagnostic_path.unlink(missing_ok=True)

    engine = (
        coverage_engine
        if coverage_engine is not None
        else PolicyEngineUKCoverageEngine()
    )
    # Mirrors the US cheap preflight: graph or reference drift aborts before
    # source stages and, once added, before national target-registry compilation.
    assert_uk_release_input_coverage_manifest_current(engine=engine)
    assert_uk_release_input_coverage_build_stages(
        tuple(stage.name for stage in materialized_stages)
    )
    dataset = load_uk_national_dataset(input_path)
    for stage in materialized_stages:
        dataset = stage.run(dataset)
        validate_uk_national_dataset(dataset)

    # Mirrors the US final-export placement: enforce the complete reference
    # surface after all stages and immediately before the staging writer.
    input_coverage = uk_release_input_coverage_gate(dataset, engine)
    if diagnostic_path is not None:
        _write_input_coverage_diagnostic(diagnostic_path, input_coverage)
    if not input_coverage.passed:
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Input coverage failed: {failure}"
                for failure in input_coverage.failures
            )
        )

    write_uk_national_dataset(dataset, staging_path)
    return UKNationalBuildResult(
        dataset=dataset,
        input_h5=input_path,
        staging_h5=staging_path,
        stage_names=tuple(stage.name for stage in materialized_stages),
        input_coverage=input_coverage,
        input_coverage_path=diagnostic_path,
    )


def _validate_stages(stages: tuple[UKNationalStage, ...]) -> None:
    names: set[str] = set()
    for stage in stages:
        if not isinstance(stage, UKNationalStage):
            raise TypeError(
                "UK national stages must be UKNationalStage instances, "
                f"got {type(stage).__name__}."
            )
        if stage.name in names:
            raise ValueError(f"Duplicate UK national stage {stage.name!r}.")
        names.add(stage.name)


def _write_input_coverage_diagnostic(path: Path, gate: GateResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "enforced": True,
        "input_coverage": {
            "passed": gate.passed,
            "failures": list(gate.failures),
            "details": dict(gate.details),
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} table is missing column(s): {missing}.")


def _require_unique(frame: pd.DataFrame, column: str, *, label: str) -> None:
    if frame[column].isna().any():
        raise ValueError(f"{label}.{column} contains missing values.")
    if frame[column].duplicated().any():
        duplicates = frame.loc[frame[column].duplicated(), column].unique()
        raise ValueError(
            f"{label}.{column} must be unique; duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
        )


def _weight_kind_from_stored(value: object) -> WeightKind:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return WeightKind(str(value))
    except ValueError as exc:
        raise ValueError(f"Unknown stored UK household weight kind {value!r}.") from exc


def _read_weight_metadata(path: Path) -> tuple[object, object]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - UK H5 runtime dependency
        raise RuntimeError("h5py is required to read UK national metadata.") from exc
    with h5py.File(path, mode="r") as file:
        return (
            file.attrs.get(
                UK_HOUSEHOLD_WEIGHT_KIND_ATTR,
                WeightKind.DESIGN.value,
            ),
            file.attrs.get(UK_MASS_LOG_ATTR, "[]"),
        )


def _write_weight_metadata(path: Path, dataset: UKNationalDataset) -> None:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - UK H5 runtime dependency
        raise RuntimeError("h5py is required to write UK national metadata.") from exc
    with h5py.File(path, mode="r+") as file:
        file.attrs[UK_HOUSEHOLD_WEIGHT_KIND_ATTR] = dataset.household_weight_kind.value
        file.attrs[UK_MASS_LOG_ATTR] = json.dumps(
            [_mass_change_record_payload(record) for record in dataset.mass_log],
            sort_keys=True,
        )


def _mass_log_from_stored(value: object) -> tuple[MassChangeRecord, ...]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError("Stored UK populace mass log is not valid JSON.") from exc
    if not isinstance(payload, list):
        raise ValueError("Stored UK populace mass log must be a JSON list.")
    records: list[MassChangeRecord] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("Stored UK populace mass-log entries must be objects.")
        try:
            records.append(
                MassChangeRecord(
                    entity=str(entry["entity"]),
                    old_total=float(entry["old_total"]),
                    new_total=float(entry["new_total"]),
                    declared_factor=(
                        None
                        if entry.get("declared_factor") is None
                        else float(entry["declared_factor"])
                    ),
                    reason=str(entry["reason"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Stored UK populace mass-log entry is malformed.") from exc
    return tuple(records)


def _mass_change_record_payload(record: MassChangeRecord) -> dict[str, object]:
    return {
        "entity": record.entity,
        "old_total": record.old_total,
        "new_total": record.new_total,
        "declared_factor": record.declared_factor,
        "reason": record.reason,
    }
