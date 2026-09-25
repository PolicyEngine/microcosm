#!/usr/bin/env python3
"""Recover raw ACS hours through reviewed numeric source linkage, locally only.

No pandas, PyTables, pickle, country model, or object HDF block is used. The
original staging revision is unknown: this implements the reviewed 592ae5d6
sort/rank reconstruction and verifies its accessible numeric anchors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sqlite3
import tempfile
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import h5py
import numpy as np

PERSON_ANCHORS = (
    "SPORDER",
    "ADJINC",
    "PWGTP",
    "AGEP",
    "INTP",
    "MAR",
    "RELSHIPP",
    "RETP",
    "SEMP",
    "SEX",
    "SSIP",
    "SSP",
    "WAGP",
)
HOUSEHOLD_ANCHORS = ("NP", "ADJHSG", "TEN", "RNTP", "GRNTP", "TAXAMT", "TYPEHUGQ")
HOURS = ("WKHP", "WKL", "FWKHP")
PERSON_KEYS = (
    "PH_SEQ",
    "A_LINENO",
    "A_AGE",
    "source_year",
    "source_household_id",
    "source_row_id",
    "person_household_id",
    "person_id",
    "person_source_id",
    "person_support_clone_index",
)
HOUSEHOLD_KEYS = (
    "household_id",
    "state_fips",
    "household_source_id",
    "household_support_clone_index",
)
SIDECAR_DTYPE = np.dtype(
    [
        (name, "<i8")
        for name in (
            "parent_person_row",
            "person_id",
            "person_household_id",
            "source_year",
            "source_household_id",
            "source_row_id",
            "SPORDER",
        )
    ]
    + [(name, "<f8") for name in HOURS]
)


def _require(condition: Any, message: str) -> None:
    okay = condition if isinstance(condition, (bool, np.bool_)) else np.all(condition)
    if not okay:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    """Stream a file digest without interpreting its contents."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def _authenticated(path: Path, digest: str, size: int | None = None):
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        _require(size is None or before.st_size == size, "Input size mismatch")
        _require(
            hashlib.file_digest(stream, "sha256").hexdigest() == digest,
            "Input SHA256 mismatch",
        )
        stream.seek(0)
        yield stream
        after = os.fstat(stream.fileno())
        _require(
            (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "Input changed during recovery",
        )


def _csv_rows(stream: BinaryIO, role: str) -> Iterator[dict[str, str]]:
    prefix = "psam_hus" if role == "household" else "psam_pus"
    required = {"SERIALNO"} | set(
        (*HOUSEHOLD_ANCHORS, "WGTP", "PUMA")
        if role == "household"
        else (*PERSON_ANCHORS, *HOURS)
    )
    with zipfile.ZipFile(stream) as archive:
        names = archive.namelist()
        _require(len(names) == len(set(names)), "Duplicate ZIP members")
        members = sorted(
            name
            for name in names
            if Path(name).name.lower().startswith(prefix)
            and name.lower().endswith(".csv")
        )
        _require(
            members == [prefix + "a.csv", prefix + "b.csv"],
            "Unexpected ACS CSV member roster",
        )
        for member in members:
            with (
                archive.open(member) as raw,
                io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text,
            ):
                reader = csv.reader(text)
                header = next(reader)
                _require(len(header) == len(set(header)), "Duplicate CSV columns")
                _require(required <= set(header), "Missing ACS source columns")
                if role == "household":
                    _require(
                        "ST" in header or "STATE" in header,
                        "Missing household state column",
                    )
                # Parse only retained columns, while streaming the wide CSV records.
                wanted = required | ({"ST", "STATE"} if role == "household" else set())
                indices = [(i, name) for i, name in enumerate(header) if name in wanted]
                for row in reader:
                    _require(len(row) == len(header), "Malformed ACS CSV row")
                    yield {name: row[i] for i, name in indices}


def _number(token: str) -> float:
    if not token.strip():
        return float("nan")
    # The retained ACS fields are integer-coded; avoid lossy float parsing.
    value = int(token)
    if abs(value) > 2**53:
        raise ValueError("Source number cannot be represented exactly")
    return float(value)


def _vector(row: Mapping[str, str], fields: Sequence[str]) -> np.ndarray:
    return np.asarray([_number(row[name]) for name in fields], dtype="<f8")


def _array(work: Path, name: str, shape: tuple[int, ...], dtype="<f8"):
    return np.lib.format.open_memmap(
        work / (name + ".npy"), mode="w+", dtype=dtype, shape=shape
    )


@dataclass
class Reconstructed:
    ranks: np.ndarray
    household: np.ndarray
    state: np.ndarray
    person: np.ndarray
    hours: np.ndarray
    person_household: np.ndarray
    source_rows: np.ndarray
    source_counts: dict[str, int]


def _select_ranks(
    count: int, kinds: np.ndarray, pilot: int | None, explicit: Sequence[int] | None
) -> np.ndarray:
    if explicit is not None:
        ranks = np.asarray(explicit, dtype=np.int64)
        _require(
            len(ranks) > 0 and len(set(ranks)) == len(ranks),
            "Pilot ranks must be nonempty and unique",
        )
        _require((ranks >= 1) & (ranks <= count), "Pilot rank out of range")
        return np.sort(ranks)
    if pilot is None:
        return np.arange(1, count + 1, dtype=np.int64)
    _require(2 <= pilot <= count, "Spread pilot requires 2..H households")
    ranks = np.linspace(1, count, pilot, dtype=np.int64)
    # SERIALNO order can cluster GQ records; retain both kinds in the pilot.
    for mask in (kinds == 1, kinds > 1):
        if np.any(mask[1:]) and not np.any(mask[ranks]):
            ranks[-1] = np.flatnonzero(mask)[0]
    return np.sort(ranks)


def reconstruct_sources(
    household_zip: Path,
    person_zip: Path,
    work: Path,
    contract: Mapping[str, Any],
    sources: Mapping[str, Any],
    *,
    pilot_households: int | None = None,
    household_ranks: Sequence[int] | None = None,
    chunk_rows: int = 8192,
) -> Reconstructed:
    """Stream authenticated CSVs into numeric arrays in global source order.

    The pilot retains global ranks; it does not re-rank a subset or claim to
    validate person multiplicities in unselected households.
    """
    specs = {item["role"]: item for item in sources["artifacts"]}
    db = sqlite3.connect(work / "households.sqlite")
    try:
        db.execute("PRAGMA cache_size=-32768")
        db.execute(
            "CREATE TABLE household (serial TEXT PRIMARY KEY, n INTEGER, kind INTEGER, state INTEGER, anchors BLOB)"
        )
        spec = specs["household"]
        with _authenticated(household_zip, spec["sha256"], spec["size_bytes"]) as raw:
            batch = []
            for row in _csv_rows(raw, "household"):
                _require(bool(row["SERIALNO"]), "Missing SERIALNO")
                anchors = _vector(row, HOUSEHOLD_ANCHORS)
                n, kind, weight = anchors[0], anchors[-1], _number(row["WGTP"])
                _require(
                    np.isfinite(n) and n >= 0 and kind in (1, 2, 3),
                    "Invalid NP or TYPEHUGQ",
                )
                _require(
                    np.isfinite(weight)
                    and (
                        (kind == 1 and weight > 0)
                        or (kind in (2, 3) and n == 1 and weight == 0)
                    ),
                    "Invalid housing-unit/GQ NP or weight",
                )
                state = _number(row["ST"] if "ST" in row else row["STATE"])
                _require(
                    np.isfinite(state) and 1 <= state <= 99, "Invalid source state"
                )
                batch.append(
                    (row["SERIALNO"], int(n), int(kind), int(state), anchors.tobytes())
                )
                if len(batch) >= chunk_rows:
                    db.executemany("INSERT INTO household VALUES (?,?,?,?,?)", batch)
                    batch.clear()
            db.executemany("INSERT INTO household VALUES (?,?,?,?,?)", batch)
        db.commit()
        counts = {
            "households": db.execute(
                "SELECT count(*) FROM household WHERE n>0"
            ).fetchone()[0],
            "people": db.execute("SELECT coalesce(sum(n),0) FROM household").fetchone()[
                0
            ],
            "housing_units": db.execute(
                "SELECT count(*) FROM household WHERE n>0 AND kind=1"
            ).fetchone()[0],
            "group_quarters": db.execute(
                "SELECT count(*) FROM household WHERE kind>1"
            ).fetchone()[0],
        }
        for key, value in counts.items():
            _require(
                value == contract["acs_" + key],
                "Source household totals disagree with contract",
            )
        h_count = counts["households"]
        serials = {}
        starts = np.zeros(h_count + 1, dtype=np.int64)
        sizes = np.zeros(h_count + 1, dtype=np.int64)
        kinds = np.zeros(h_count + 1, dtype=np.int8)
        rank = row_start = 0
        for serial, n, kind in db.execute(
            "SELECT serial,n,kind FROM household ORDER BY serial COLLATE BINARY"
        ):
            if n:
                rank += 1
                starts[rank], sizes[rank], kinds[rank] = row_start, n, kind
                row_start += n
            serials[serial] = rank if n else 0
        ranks = _select_ranks(h_count, kinds, pilot_households, household_ranks)
        local_h = np.full(h_count + 1, -1, dtype=np.int64)
        local_h[ranks] = np.arange(len(ranks))
        household = _array(work, "household", (len(ranks), len(HOUSEHOLD_ANCHORS)))
        state = np.empty(len(ranks), dtype=np.int64)
        for serial, source_state, blob in db.execute(
            "SELECT serial,state,anchors FROM household WHERE n>0 ORDER BY serial COLLATE BINARY"
        ):
            index = local_h[serials[serial]]
            if index >= 0:
                household[index] = np.frombuffer(blob, dtype="<f8")
                state[index] = source_state
    finally:
        db.close()
    selected_sizes = sizes[ranks]
    local_starts = np.r_[0, np.cumsum(selected_sizes)[:-1]]
    p_count = int(selected_sizes.sum())
    person = _array(work, "person_unsorted", (p_count, len(PERSON_ANCHORS)))
    hours = _array(work, "hours_unsorted", (p_count, len(HOURS)))
    seen = np.zeros(len(ranks), dtype=np.int64)
    spec = specs["person"]
    with _authenticated(person_zip, spec["sha256"], spec["size_bytes"]) as raw:
        for row in _csv_rows(raw, "person"):
            _require(row["SERIALNO"] in serials, "Orphan ACS person")
            source_h = serials[row["SERIALNO"]]
            if source_h == 0:
                raise ValueError("Person attached to vacant household")
            index = local_h[source_h]
            if index < 0:
                continue
            _require(
                seen[index] < selected_sizes[index], "Person count exceeds household NP"
            )
            target = local_starts[index] + seen[index]
            person[target] = _vector(row, PERSON_ANCHORS)
            _require(
                np.isfinite(person[target, 0]) and person[target, 0] >= 1,
                "SPORDER must be a positive integer",
            )
            hours[target] = _vector(row, HOURS)
            seen[index] += 1
    _require(seen == selected_sizes, "Person counts do not equal household NP")
    del serials
    person_h = np.repeat(ranks, selected_sizes)
    order = np.lexsort((person[:, 0], person_h))
    sorted_person = _array(work, "person", person.shape)
    sorted_hours = _array(work, "hours", hours.shape)
    for start in range(0, p_count, chunk_rows):
        stop = start + chunk_rows
        sorted_person[start:stop] = person[order[start:stop]]
        sorted_hours[start:stop] = hours[order[start:stop]]
    _require(
        ~(
            (person_h[1:] == person_h[:-1])
            & (sorted_person[1:, 0] == sorted_person[:-1, 0])
        ),
        "Duplicate ACS composite source key",
    )
    source_rows = np.arange(p_count, dtype=np.int64) + np.repeat(
        starts[ranks] - local_starts, selected_sizes
    )
    return Reconstructed(
        ranks,
        household,
        state,
        sorted_person,
        sorted_hours,
        person_h,
        source_rows,
        counts,
    )


class NumericBlocks:
    """Read only the four declared plain numeric blocks and byte-string labels."""

    def __init__(self, handle: h5py.File, contract: Mapping[str, Any]):
        self.blocks = {}
        for path, spec in contract["blocks"].items():
            group = handle
            for part in path.strip("/").split("/")[:-1]:
                _require(
                    isinstance(group.get(part, getlink=True), h5py.HardLink),
                    "H5 group link is not local",
                )
                group = group[part]
            for suffix in ("_items", "_values"):
                name = path.rsplit("/", 1)[1] + suffix
                _require(
                    isinstance(group.get(name, getlink=True), h5py.HardLink),
                    "H5 dataset link is not local",
                )
            labels = handle[path + "_items"]
            values = handle[path + "_values"]
            _require(
                isinstance(labels, h5py.Dataset)
                and labels.dtype.kind == "S"
                and labels.shape == (len(spec["columns"]),)
                and not labels.is_virtual
                and not labels.external,
                "Unsafe H5 labels",
            )
            _require(
                isinstance(values, h5py.Dataset)
                and values.dtype.kind in "if"
                and str(values.dtype) == spec["dtype"]
                and list(values.shape) == spec["shape"]
                and not values.is_virtual
                and not values.external,
                "Unsafe or mismatched H5 numeric block",
            )
            _require(
                [item.decode("ascii") for item in labels[:]] == spec["columns"],
                "H5 block column labels differ from contract",
            )
            self.blocks[path] = (values, spec["columns"])

    def read(
        self, entity: str, fields: Sequence[str], start: int, stop: int
    ) -> dict[str, np.ndarray]:
        result = {}
        for path, (dataset, columns) in self.blocks.items():
            if not path.startswith("/" + entity + "/"):
                continue
            selected = [(i, name) for i, name in enumerate(columns) if name in fields]
            if selected:
                values = dataset[start:stop, [i for i, _ in selected]]
                result.update(
                    {name: values[:, j] for j, (_, name) in enumerate(selected)}
                )
        _require(set(result) == set(fields), "Requested numeric column absent")
        return result


def _equal(left: np.ndarray, right: np.ndarray, name: str) -> None:
    _require(
        (left == right) | (np.isnan(left) & np.isnan(right)),
        "Raw anchor mismatch: " + name,
    )


def validate_parent(
    blocks: NumericBlocks,
    source: Reconstructed,
    sidecar: np.ndarray,
    contract: Mapping[str, Any],
    *,
    chunk_rows: int = 8192,
    full: bool,
) -> dict[str, Any]:
    """Prove the selected join and every numeric anchor before committing output."""
    p_count, h_count = len(source.person), len(source.ranks)
    seen_p, seen_h = np.zeros(p_count, dtype=bool), np.zeros(h_count, dtype=bool)
    current_h = np.empty(h_count, dtype=np.int64)
    membership = np.zeros(h_count, dtype=np.int64)
    offsets: dict[str, int] = {}
    all_ids = {}
    candidate_counts = {"person": 0, "household": 0}
    output_count = 0
    for entity, marker, fields, total in (
        (
            "household",
            "NP",
            (*HOUSEHOLD_KEYS, *HOUSEHOLD_ANCHORS),
            contract["total_households"],
        ),
        (
            "person",
            "SPORDER",
            (*PERSON_KEYS, *PERSON_ANCHORS),
            contract["total_people"],
        ),
    ):
        ids = np.empty(total, dtype=np.int64)
        for start in range(0, total, chunk_rows):
            stop = min(start + chunk_rows, total)
            data = blocks.read(entity, fields, start, stop)
            ids[start:stop] = data[entity + "_id"]
            mask = np.isfinite(data[marker])
            _require(~np.isinf(data[marker]), "Infinite ACS membership marker")
            candidate_counts[entity] += int(mask.sum())
            h = data[
                "household_source_id"
                if entity == "household"
                else "source_household_id"
            ]
            positions = np.searchsorted(source.ranks, h)
            selected = (positions < h_count) & (
                source.ranks[np.minimum(positions, h_count - 1)] == h
            )
            if full:
                _require(~mask | selected, "Unexpected ACS household key")
            mask &= selected
            if not np.any(mask):
                continue
            parent_rows = np.arange(start, stop, dtype=np.int64)[mask]
            data = {name: values[mask] for name, values in data.items()}
            h_index, h = positions[mask], h[mask]
            if entity == "household":
                _require(
                    len(np.unique(h_index)) == len(h_index)
                    and not np.any(seen_h[h_index]),
                    "Duplicate target household source key",
                )
                seen_h[h_index] = True
                current_h[h_index] = data["household_id"]
                _require(
                    data["household_support_clone_index"] == 0,
                    "ACS household clone present",
                )
                _equal(data["state_fips"], source.state[h_index], "state_fips")
                for i, name in enumerate(HOUSEHOLD_ANCHORS):
                    _equal(data[name], source.household[h_index, i], name)
                difference = data["household_id"] - h
            else:
                r = data["source_row_id"]
                index = np.searchsorted(source.source_rows, r)
                _require(index < p_count, "Unexpected source row key")
                _require(source.source_rows[index] == r, "Unexpected source row key")
                _require(
                    len(np.unique(index)) == len(index) and not np.any(seen_p[index]),
                    "Duplicate target source row key",
                )
                seen_p[index] = True
                # Unique raw (h, SPORDER), source-row coverage and these equalities
                # are an independent composite join to the same raw record.
                _require(
                    source.person_household[index] == h,
                    "Composite/source-row joins disagree",
                )
                _require(data["source_year"] == 2024, "ACS source year mismatch")
                _require(data["person_source_id"] == r, "Person source ID mismatch")
                _require(
                    data["person_support_clone_index"] == 0, "ACS person clone present"
                )
                _require(seen_h[h_index], "ACS household absent from parent")
                _require(
                    data["person_household_id"] == current_h[h_index],
                    "Person household membership mismatch",
                )
                np.add.at(membership, h_index, 1)
                for i, name in enumerate(PERSON_ANCHORS):
                    _equal(data[name], source.person[index, i], name)
                for name, expected in (
                    ("PH_SEQ", h),
                    ("A_LINENO", data["SPORDER"]),
                    ("A_AGE", data["AGEP"]),
                ):
                    _equal(data[name], expected, name)
                difference = data["person_id"] - r
                out = sidecar[output_count : output_count + len(index)]
                out["parent_person_row"] = parent_rows
                for name in SIDECAR_DTYPE.names:
                    if name in data:
                        out[name] = data[name]
                for i, name in enumerate(HOURS):
                    out[name] = source.hours[index, i]
                output_count += len(index)
            offset = int(difference[0])
            _require(offset >= 0, "Negative current/source ID offset")
            _require(difference == offset, "Nonconstant current/source ID offset")
            _require(
                offsets.setdefault(entity, offset) == offset,
                "Current/source ID offset changed across chunks",
            )
        _require(len(np.unique(ids)) == total, "Duplicate current " + entity + " ID")
        all_ids[entity] = total
    _require(seen_p.all() and seen_h.all(), "Missing target source keys")
    _require(
        membership == source.household[:, 0], "Target household NP/membership mismatch"
    )
    if full:
        _require(
            candidate_counts == {"person": p_count, "household": h_count},
            "Incomplete ACS candidate set",
        )
    return {
        "selected_people": p_count,
        "selected_households": h_count,
        "parent_rows_checked_for_unique_current_ids": all_ids,
        "candidate_counts": candidate_counts,
        "current_id_offsets": offsets,
        "full_source_bijection": full,
        "selected_source_bijection": True,
        "person_anchors_checked": list(PERSON_ANCHORS),
        "household_anchors_checked": [*HOUSEHOLD_ANCHORS, "state_fips"],
        "puma_compared": False,
        "object_blocks_read": False,
    }


def recover(
    household_zip: Path,
    person_zip: Path,
    parent_h5: Path,
    output_dir: Path,
    *,
    contract: Mapping[str, Any] | None = None,
    sources: Mapping[str, Any] | None = None,
    pilot_households: int | None = None,
    household_ranks: Sequence[int] | None = None,
    chunk_rows: int = 8192,
) -> dict[str, Any]:
    """Write a numeric .npy sidecar and receipt only after successful validation.

    Contract/source overrides support invented fixtures. The CLI always uses
    the committed parent contract and existing public ACS source manifest.
    """
    _require(chunk_rows > 0, "chunk_rows must be positive")
    _require(
        pilot_households is None or household_ranks is None, "Choose one pilot selector"
    )
    _require(not output_dir.exists(), "Output directory already exists")
    root = Path(__file__).resolve().parents[1]
    contract_file_sha256 = source_manifest_file_sha256 = None
    if contract is None:
        contract_path = Path(__file__).with_name(
            "acs_numeric_source_recovery_contract.json"
        )
        contract_bytes = contract_path.read_bytes()
        contract_file_sha256 = hashlib.sha256(contract_bytes).hexdigest()
        contract = json.loads(contract_bytes)
    if sources is None:
        manifest = (
            root
            / "packages/microcosm-build/src/microcosm/build/us_runtime/acs_2024_1yr_sources.json"
        )
        source_bytes = manifest.read_bytes()
        source_manifest_file_sha256 = hashlib.sha256(source_bytes).hexdigest()
        _require(
            source_manifest_file_sha256 == contract["source_manifest_sha256"],
            "Source manifest differs from recovery contract",
        )
        sources = json.loads(source_bytes)
    full = pilot_households is None and household_ranks is None
    with tempfile.TemporaryDirectory(
        prefix=".acs-recovery-", dir=output_dir.parent
    ) as temp:
        temp = Path(temp)
        work, result = temp / "work", temp / "result"
        work.mkdir()
        result.mkdir()
        source = reconstruct_sources(
            household_zip,
            person_zip,
            work,
            contract,
            sources,
            pilot_households=pilot_households,
            household_ranks=household_ranks,
            chunk_rows=chunk_rows,
        )
        sidecar = np.lib.format.open_memmap(
            result / "acs-source-hours.npy",
            mode="w+",
            dtype=SIDECAR_DTYPE,
            shape=(len(source.person),),
        )
        with (
            _authenticated(parent_h5, contract["parent_sha256"]) as raw,
            h5py.File(raw, "r") as handle,
        ):
            checked = validate_parent(
                NumericBlocks(handle, contract),
                source,
                sidecar,
                contract,
                chunk_rows=chunk_rows,
                full=full,
            )
        sidecar.flush()
        del sidecar
        receipt = {
            "protocol": "acs-numeric-source-recovery-v1",
            "mode": "full" if full else "household_pilot",
            "linkage_basis": "reviewed_sort_rank_reconstruction",
            "original_staging_revision": None,
            "reviewed_packaging_revision": contract["reviewed_packaging_revision"],
            "parent_build_id": contract["parent_build_id"],
            "parent_sha256": contract["parent_sha256"],
            "sources": sources,
            "source_counts": source.source_counts,
            "selected_household_ranks": None if full else source.ranks.tolist(),
            "contract_file_sha256": contract_file_sha256,
            "source_manifest_file_sha256": source_manifest_file_sha256,
            "contract_canonical_json_sha256": hashlib.sha256(
                json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "tool_sha256": sha256_file(Path(__file__)),
            "sidecar_sha256": sha256_file(result / "acs-source-hours.npy"),
            "sidecar_bytes": (result / "acs-source-hours.npy").stat().st_size,
            "sidecar_order": "parent_person_row ascending",
            "missing_hours": "NaN preserves blank source tokens; no default repair applied",
            **checked,
        }
        (result / "RECOVERY.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        # An existing output is never replaced. Only this owned temporary tree
        # is cleaned on failure; the parent and source archives stay read-only.
        output_dir.mkdir()
        try:
            for name in ("acs-source-hours.npy", "RECOVERY.json"):
                os.rename(result / name, output_dir / name)
        except BaseException:
            shutil.rmtree(output_dir)
            raise
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--household-zip", type=Path, required=True)
    parser.add_argument("--person-zip", type=Path, required=True)
    parser.add_argument("--parent-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--pilot-households", type=int)
    selector.add_argument(
        "--household-ranks", help="Comma-separated global occupied household ranks"
    )
    parser.add_argument("--chunk-rows", type=int, default=8192)
    args = parser.parse_args(argv)
    ranks = (
        None
        if args.household_ranks is None
        else tuple(int(item) for item in args.household_ranks.split(","))
    )
    receipt = recover(
        args.household_zip,
        args.person_zip,
        args.parent_h5,
        args.output_dir,
        pilot_households=args.pilot_households,
        household_ranks=ranks,
        chunk_rows=args.chunk_rows,
    )
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "mode",
                    "selected_people",
                    "selected_households",
                    "full_source_bijection",
                    "sidecar_sha256",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
