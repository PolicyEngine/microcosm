"""Attach authenticated raw ASEC household observations by exact original keys.

The caller authenticates the v4 checkpoint bytes and loads its actual Frame and
metadata with ``load_asec_raw_stage_checkpoint_v4`` before calling this helper.
Structural checks here do not authenticate arbitrary caller-supplied Frame
values. The augmented source has a new attachment receipt; it is not the input
checkpoint's v4 content identity. No graph registration or price conversion runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import h5py
import numpy as np
import pandas as pd

from microcosm.build.outer_stage_runtime import FrameIdentity, frame_identity
from microcosm.build.us_runtime import asec_checkpoint
from microcosm.build.us_runtime.operator_boundary import (
    assert_operator_free_source_frame,
)
from microcosm.frame import Frame

ASEC_HOUSEHOLD_OBSERVATION_COLUMNS = (
    "HTOTVAL",
    "H_LIVQRT",
    "HRHTYPE",
    "H_HHTYPE",
    "H_SEQ",
    "H_TENURE",
    "GESTFIPS",
)
_OUTPUT_COLUMNS = tuple(
    f"asec_{column}" for column in ASEC_HOUSEHOLD_OBSERVATION_COLUMNS
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CONTRACT_LABEL = Path("authenticated-v4-frame")


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


@dataclass(frozen=True)
class AsecHouseholdObservationSource:
    """Explicit raw household file binding to the checkpoint's income year."""

    year: int
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if type(self.year) is not int:
            raise ValueError("ASEC household source income year must be an integer.")
        _require_sha(self.sha256, "ASEC household source pin")
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True)
class AsecHouseholdObservationResult:
    """The augmented full source Frame and a reproducible, value-free receipt."""

    frame: Frame
    _receipt_json: str

    @property
    def receipt(self) -> dict[str, object]:
        """Return an independent JSON-safe copy of the immutable receipt."""
        return json.loads(self._receipt_json)


def _stat_identity(stat: os.stat_result) -> tuple[int, ...]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _hash_file(handle: BinaryIO) -> str:
    handle.seek(0)
    return hashlib.file_digest(handle, "sha256").hexdigest()


def _hard_child(group: h5py.Group, name: str) -> h5py.Group | h5py.Dataset:
    if not isinstance(group.get(name, getlink=True), h5py.HardLink):
        raise ValueError(f"ASEC household HDF requires local hard-linked {name!r}.")
    child = group[name]
    if isinstance(child, h5py.Dataset) and (child.is_virtual or child.external):
        raise ValueError("ASEC household HDF cannot read virtual or external storage.")
    return child


def _dataset(group: h5py.Group, name: str) -> h5py.Dataset:
    child = _hard_child(group, name)
    if not isinstance(child, h5py.Dataset):
        raise ValueError(f"ASEC household HDF {name!r} must be a dataset.")
    return child


def _column_names(group: h5py.Group, name: str) -> tuple[str, ...]:
    dataset = _dataset(group, name)
    if dataset.ndim != 1 or dataset.dtype.kind != "S":
        raise ValueError("ASEC household column metadata must be fixed-width strings.")
    try:
        values = tuple(value.decode("utf-8") for value in dataset[()])
    except UnicodeDecodeError as error:
        raise ValueError("ASEC household column metadata must be UTF-8.") from error
    if len(values) != len(set(values)) or any(not name for name in values):
        raise ValueError("ASEC household column metadata is empty or duplicated.")
    return values


def _read_numeric_household(
    handle: BinaryIO,
    *,
    columns: Sequence[str] = ASEC_HOUSEHOLD_OBSERVATION_COLUMNS,
    require_same_block: bool = False,
) -> pd.DataFrame:
    """Read selected primitive observations without decoding any object block."""
    if (
        isinstance(columns, str)
        or not columns
        or any(not isinstance(name, str) or not name for name in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("ASEC household numeric projection requires unique names.")
    requested = tuple(columns)
    with h5py.File(handle, "r") as h5:
        group = _hard_child(h5, "household")
        if not isinstance(group, h5py.Group):
            raise ValueError("ASEC household HDF lacks its household group.")
        if group.attrs.get("pandas_type") != b"frame" or group.attrs.get("ndim") != 2:
            raise ValueError(
                "ASEC household HDF requires the pandas fixed frame format."
            )
        nblocks = group.attrs.get("nblocks")
        if (
            isinstance(nblocks, (bool, np.bool_))
            or not isinstance(nblocks, (int, np.integer))
            or nblocks < 1
        ):
            raise ValueError("ASEC household HDF has an invalid block count.")
        columns = _column_names(group, "axis0")
        if set(requested) - set(columns):
            raise ValueError("ASEC household HDF lacks required observed columns.")
        axis1 = _dataset(group, "axis1")
        if axis1.ndim != 1:
            raise ValueError("ASEC household HDF row axis must be one-dimensional.")
        rows = axis1.shape[0]
        if not rows:
            raise ValueError("ASEC household HDF has no household rows.")
        all_items = []
        output = {}
        for index in range(int(nblocks)):
            names = _column_names(group, f"block{index}_items")
            all_items.extend(names)
            wanted = [i for i, name in enumerate(names) if name in requested]
            if not wanted:
                # Do not even open an unrelated object/VLARRAY value dataset.
                continue
            if require_same_block and output:
                raise ValueError(
                    "ASEC household selected observations require one shared block."
                )
            values = _dataset(group, f"block{index}_values")
            if values.dtype != np.dtype("<i8") or values.shape != (rows, len(names)):
                raise ValueError(
                    "ASEC household observations require exact int64 block shape."
                )
            selected = values[:, wanted]
            output.update(
                {
                    names[position]: selected[:, offset]
                    for offset, position in enumerate(wanted)
                }
            )
        if len(all_items) != len(set(all_items)) or set(all_items) != set(columns):
            raise ValueError(
                "ASEC household block columns disagree with the frame inventory."
            )
    return pd.DataFrame({name: output[name] for name in requested})


def _load_source(source: AsecHouseholdObservationSource) -> pd.DataFrame:
    with source.path.open("rb", buffering=0) as handle:
        before = _stat_identity(os.fstat(handle.fileno()))
        if (
            _stat_identity(source.path.stat()) != before
            or _hash_file(handle) != source.sha256
        ):
            raise ValueError(f"ASEC household source {source.year} SHA-256 mismatch.")
        handle.seek(0)
        table = _read_numeric_household(handle)
        if (
            _hash_file(handle) != source.sha256
            or _stat_identity(os.fstat(handle.fileno())) != before
            or _stat_identity(source.path.stat()) != before
        ):
            raise ValueError(
                f"ASEC household source {source.year} changed during reading."
            )
    if table.H_SEQ.duplicated().any():
        raise ValueError(f"ASEC household source {source.year} repeats H_SEQ.")
    return table


def _int64_column(table: pd.DataFrame, name: str) -> np.ndarray:
    if name not in table or table[name].dtype != np.dtype("int64"):
        raise ValueError(
            f"ASEC household join requires complete ordinary int64 {name}."
        )
    return table[name].to_numpy(copy=False)


def _validate_input(frame: Frame, metadata: Mapping[str, object]) -> str:
    if not isinstance(frame, Frame) or not isinstance(metadata, Mapping):
        raise TypeError(
            "ASEC household attachment requires an actual v4 Frame and metadata."
        )
    asec_checkpoint._validate_raw_stage_binding(
        metadata,
        path=_CONTRACT_LABEL,
        policy=asec_checkpoint._RAW_STAGE_V4,
    )
    asec_checkpoint._validate_asec_frame(
        frame, path=_CONTRACT_LABEL, artifact_label="ASEC v4 source"
    )
    assert_operator_free_source_frame(frame, label="ASEC household observation input")
    asec_checkpoint._validate_raw_stage_source_columns(
        frame,
        path=_CONTRACT_LABEL,
        policy=asec_checkpoint._RAW_STAGE_V4,
    )
    asec_checkpoint._validate_coverage_v4_binding(frame, metadata, path=_CONTRACT_LABEL)
    actual = frame_identity(frame)
    for key in ("identity", "source_construction_identity"):
        if FrameIdentity.from_payload(metadata[key], label=key) != actual:
            raise ValueError(
                "ASEC household input structural identity differs from v4 metadata."
            )
    if set(_OUTPUT_COLUMNS) & set(frame.table("household").columns):
        raise ValueError(
            "ASEC household observation attachment refuses existing output columns."
        )
    return actual.sha256


def _household_keys(frame: Frame) -> pd.DataFrame:
    person, household = frame.person, frame.table("household")
    person_keys = pd.DataFrame(
        {
            "household_id": _int64_column(person, "person_household_id"),
            "source_year": _int64_column(person, "source_year"),
            "H_SEQ": _int64_column(person, "source_household_id"),
        }
    )
    keys = person_keys.drop_duplicates()
    if keys.household_id.duplicated().any():
        raise ValueError("ASEC persons disagree on the source key within a household.")
    if keys.duplicated(["source_year", "H_SEQ"]).any():
        raise ValueError(
            "ASEC raw source key is claimed by multiple normalized households."
        )
    ids = _int64_column(household, "household_id")
    if set(keys.household_id) != set(ids):
        raise ValueError(
            "ASEC household source keys do not cover actual household IDs."
        )
    return keys.set_index("household_id").loc[ids].reset_index()


def with_asec_household_observations(
    frame: Frame,
    *,
    checkpoint_metadata: Mapping[str, object],
    checkpoint_sha256: str,
    sources: Sequence[AsecHouseholdObservationSource],
) -> AsecHouseholdObservationResult:
    """Augment the actual authenticated v4 source without changing its input.

    ``checkpoint_sha256`` records the caller's authenticated input-byte pin; this
    in-memory helper cannot verify that file. Explicit household sources must
    match all per-year pins in the loaded checkpoint receipt. Every raw source
    is authenticated locally before and after numeric-only reading. Source codes
    remain unclassified; nominal signed amounts remain exact int64 observations.
    """
    checkpoint_sha256 = _require_sha(checkpoint_sha256, "Input checkpoint pin")
    input_identity = _validate_input(frame, checkpoint_metadata)
    declared = {
        item["year"]: item["sha256"]
        for item in checkpoint_metadata["source_receipt"]["sources"]
    }
    if any(
        not isinstance(source, AsecHouseholdObservationSource) for source in sources
    ):
        raise TypeError("ASEC household sources require explicit typed file bindings.")
    actual = {source.year: source.sha256 for source in sources}
    if len(actual) != len(sources) or actual != declared:
        raise ValueError("ASEC household source years/pins differ from the v4 receipt.")
    keys = _household_keys(frame)
    household = frame.table("household").copy()
    _int64_column(household, "state_fips")
    _int64_column(household, "H_TENURE")
    outputs = {
        name: np.empty(len(household), dtype=np.int64) for name in _OUTPUT_COLUMNS
    }
    source_receipts = []
    for source in sorted(sources, key=lambda value: value.year):
        raw = _load_source(source)
        positions = np.flatnonzero(keys.source_year.to_numpy() == source.year)
        source_positions = pd.Index(raw.H_SEQ).get_indexer(keys.H_SEQ.iloc[positions])
        if (source_positions < 0).any():
            raise ValueError(
                f"ASEC household source {source.year} lacks a referenced H_SEQ."
            )
        joined = raw.iloc[source_positions]
        for raw_name, existing_name in (
            ("H_TENURE", "H_TENURE"),
            ("GESTFIPS", "state_fips"),
        ):
            if not np.array_equal(
                joined[raw_name].to_numpy(),
                household[existing_name].iloc[positions].to_numpy(),
            ):
                raise ValueError(
                    f"ASEC household joined {raw_name} disagrees with carried {existing_name}."
                )
        for column in ASEC_HOUSEHOLD_OBSERVATION_COLUMNS:
            outputs[f"asec_{column}"][positions] = joined[column].to_numpy(copy=False)
        source_receipts.append(
            {
                "income_year": source.year,
                "sha256": source.sha256,
                "source_rows": len(raw),
                "joined_rows": len(positions),
                "unreferenced_source_rows": len(raw) - len(positions),
            }
        )
    for name, values in outputs.items():
        household[name] = values
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["household"] = household
    augmented = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    assert_operator_free_source_frame(
        augmented, label="ASEC household observation output"
    )
    if frame_identity(augmented).sha256 != input_identity:
        raise AssertionError("ASEC household attachment changed structural identity.")
    receipt = {
        "schema_version": 1,
        "artifact_kind": "microcosm.asec_household_observation_attachment",
        "input_checkpoint_sha256": checkpoint_sha256,
        "input_structural_identity_sha256": input_identity,
        "reader": "pandas_fixed_hdf_numeric_int64_v1",
        "join_keys": ["income_year", "H_SEQ"],
        "sources": source_receipts,
        "household_identity": {
            "rows": len(household),
            "id_dtype": "int64",
            "ordered_ids_sha256": hashlib.sha256(
                household.household_id.to_numpy(dtype="<i8", copy=False).tobytes()
            ).hexdigest(),
        },
        "outputs": {
            name: {
                "source_column": name.removeprefix("asec_"),
                "dtype": "int64",
                "sha256": hashlib.sha256(
                    values.astype("<i8", copy=False).tobytes()
                ).hexdigest(),
            }
            for name, values in outputs.items()
        },
        "semantics": "Observed nominal signed integers and raw codes; no price adjustment or universe classification.",
    }
    return AsecHouseholdObservationResult(
        augmented, json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    )
