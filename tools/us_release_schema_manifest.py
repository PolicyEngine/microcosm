#!/usr/bin/env python3
"""Write the schema manifest of a US release H5 and its sidecars, and diff it.

For each file (the release H5 and any sidecar, such as the household
measures ``tools/us_household_measures.py`` writes) the manifest lists every
table, and for every column its dtype, row count and null count, as pandas
reads them. It then diffs the columns against a prior manifest: additions,
removals, renames where declared, and dtype changes. The prior may be an
earlier manifest JSON or a ``table,column,dtype`` column list, the form of the
sample column list sent with the County Files agreement (microcosm#922).

Outputs: the manifest JSON, and a Markdown rendering of it for publication
beside the files.

What is read and how:

- **H5.** Every key of the HDF5 store. A pandas frame (``frame`` or
  ``frame_table``) whose key does not start with ``_`` is a table; anything
  else (``_time_period``, ``_populace_staging_metadata``) is listed under
  ``other_keys`` with its value when it is one short scalar, and the root
  node's user attributes under ``root_attributes``. Table-format tables are
  read in row chunks with ``HDFStore.select(start=, stop=)``. A dtype is what
  ``pd.read_hdf`` gives for the whole table. Table-format
  columns whose chunks disagree are read whole to resolve their dtype.
  Fixed-format tables are read whole, since pandas otherwise unpickles each
  object block again for every chunk. A column the Microcosm frame codec stores as object-backed
  nullable booleans is also given ``logical_dtype: "boolean"``.
- **Sidecar CSV.** Read in row chunks with ``pd.read_csv``. A household
  measures sidecar is read with its declared dtypes; any other CSV with
  pandas' own inference.

Usage::

    uv run python tools/us_release_schema_manifest.py \\
        --h5 releases/<release>/<build>/<file>.h5 \\
        --sidecar releases/<release>/<build>/<file>.household_measures.csv.gz \\
        --prior ~/Downloads/lwi-agreement-v3/microcosm_us_2024_sample_columns.csv \\
        --output-json <dir>/schema_manifest.json \\
        --output-md <dir>/schema_manifest.md

``--renames FILE`` adds declared renames to :data:`DECLARED_RENAMES` (a JSON
list of ``{"table", "from", "to", "reason"}`` objects).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import secrets
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import us_household_measures as measures  # noqa: E402 - needs tools/ on sys.path

__all__ = [
    "DECLARED_RENAMES",
    "MANIFEST_KIND",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestError",
    "build_manifest",
    "column_index",
    "describe_csv",
    "describe_h5",
    "diff_manifests",
    "load_column_list",
    "load_prior",
    "main",
    "manifest_column_list",
    "render_markdown",
]

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_KIND = "microcosm_us_release_schema_manifest"
COLUMN_LIST_FORMAT = "column_list_csv"

#: Columns renamed between delivered schemas. A diff reports such a pair as a
#: rename only when the prior has ``from`` and not ``to`` and the current has
#: ``to`` and not ``from``; otherwise it says why the declaration did not apply.
DECLARED_RENAMES: tuple[Mapping[str, str], ...] = (
    {
        "table": "person",
        "from": "would_claim_wic",
        "to": "takes_up_wic_if_eligible",
        "reason": (
            "policyengine-us names the WIC take-up input "
            "takes_up_wic_if_eligible; it defines no would_claim_wic."
        ),
    },
)

#: Microcosm's frame codec attribute (``microcosm.frame.materialize``; a test
#: holds them equal): object-backed nullable boolean columns of one table.
FRAME_TABLE_CODEC_ATTR = "_microcosm_frame_table_codec"
TABLE_PANDAS_TYPES = ("frame", "frame_table")
ENTITY_ORDER = (
    "person",
    "household",
    "tax_unit",
    "spm_unit",
    "family",
    "marital_unit",
)
DEFAULT_CHUNK_BYTES = 256 * 2**20
MINIMUM_CHUNK_ROWS = 1_000
DEFAULT_CSV_CHUNK_ROWS = 500_000
#: Longer metadata strings are noted by type, not copied into the manifest.
MAX_METADATA_TEXT = 4_000


class ManifestError(ValueError):
    """A file, prior or rename declaration this tool cannot describe or diff."""


# ---------------------------------------------------------------------------
# Describing files
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


class _ColumnTally:
    """Accumulates one table's per-column dtypes and null counts over chunks.

    pandas may infer different dtypes for different chunks. HDF readers
    resolve such columns with a whole-column read before emitting the entry.
    Other readers refuse a mix of dtypes. An empty table keeps the dtypes of
    its empty read.
    """

    def __init__(self, table: str) -> None:
        self.table = table
        self.columns: list[str] | None = None
        self.empty_dtypes: dict[str, str] = {}
        self.seen: dict[str, set[str]] = {}
        self.nulls: dict[str, int] = {}
        self.rows = 0
        self.resolved_dtypes: dict[str, str] = {}

    @property
    def dtypes(self) -> dict[str, str]:
        resolved = {}
        for name in self.columns or []:
            if name in self.resolved_dtypes:
                resolved[name] = self.resolved_dtypes[name]
                continue
            seen = self.seen.get(name) or {self.empty_dtypes[name]}
            if len(seen) == 1:
                resolved[name] = next(iter(seen))
            else:
                raise ManifestError(
                    f"{self.table}.{name} reads as {sorted(seen)} in different chunks."
                )
        return resolved

    def add(self, chunk: pd.DataFrame) -> None:
        if isinstance(chunk.columns, pd.MultiIndex) or any(
            not isinstance(name, str) for name in chunk.columns
        ):
            raise ManifestError(
                f"Table {self.table!r} needs single-level string column names."
            )
        names = list(chunk.columns)
        if len(set(names)) != len(names):
            raise ManifestError(f"Table {self.table!r} repeats a column name.")
        if self.columns is None:
            self.columns = names
            self.empty_dtypes = {name: str(chunk[name].dtype) for name in names}
            self.seen = {name: set() for name in names}
            self.nulls = dict.fromkeys(names, 0)
        elif names != self.columns:
            raise ManifestError(f"Table {self.table!r} changed columns between chunks.")
        if len(chunk) == 0:
            return
        counts = chunk.isna().sum()
        for name in names:
            self.nulls[name] += int(counts[name])
            self.seen[name].add(str(chunk[name].dtype))
        self.rows += len(chunk)

    def entry(
        self, *, storage: str, logical: Mapping[str, str] | None = None
    ) -> dict[str, Any]:
        logical = logical or {}
        dtypes = self.dtypes
        columns = []
        for name in self.columns or []:
            column: dict[str, Any] = {
                "name": name,
                "dtype": dtypes[name],
                "null_count": self.nulls[name],
            }
            if name in logical:
                column["logical_dtype"] = logical[name]
            columns.append(column)
        return {
            "name": self.table,
            "storage": storage,
            "rows": self.rows,
            "columns": columns,
        }


def _storer_rows(storer: Any) -> int:
    """Rows of one stored object.

    A ``table`` storer counts its rows. A ``fixed`` frame reports its first
    block's physical shape; an object block can store one pickled array
    instead of one item per row. Use the row index (``axis1``). Empty fixed
    indexes have a one-element placeholder node;
    their ``shape`` attribute records the actual zero length.
    """

    nrows = getattr(storer, "nrows", None)
    if nrows is not None:
        return int(nrows)
    group = getattr(storer, "group", None)
    if getattr(storer, "pandas_type", None) == "frame" and group is not None:
        return int(getattr(group.axis1._v_attrs, "shape", group.axis1.shape)[0])
    if group is not None and "index" in group._v_children:
        return int(getattr(group.index._v_attrs, "shape", group.index.shape)[0])
    raise ManifestError(f"Cannot count the rows of {storer.pathname!r}.")


def _codec_columns(storer: Any, key: str) -> dict[str, str]:
    raw = getattr(storer.attrs, FRAME_TABLE_CODEC_ATTR, None)
    if raw is None:
        return {}
    try:
        codec = json.loads(raw)
        if (
            not isinstance(codec, dict)
            or set(codec) != {"codec_version", "nullable_boolean_columns"}
            or type(codec["codec_version"]) is not int
            or codec["codec_version"] != 1
        ):
            raise ValueError("unsupported frame codec")
        columns = codec["nullable_boolean_columns"]
        if (
            not isinstance(columns, list)
            or not columns
            or any(not isinstance(column, str) or not column for column in columns)
            or len(columns) != len(set(columns))
        ):
            raise ValueError("nullable_boolean_columns must be a list of strings")
    except (TypeError, ValueError, KeyError) as exc:
        raise ManifestError(
            f"Table {key!r} has malformed frame codec metadata."
        ) from exc
    return {str(column): "boolean" for column in columns}


def _chunk_rows(store: pd.HDFStore, key: str, storer: Any, chunk_bytes: int) -> int:
    rowsize = getattr(getattr(storer, "table", None), "rowsize", None)
    if not rowsize:
        sample = store.select(key, start=0, stop=MINIMUM_CHUNK_ROWS)
        rowsize = (
            int(sample.memory_usage(deep=True, index=False).sum()) // len(sample)
            if len(sample)
            else 1024
        )
    return max(MINIMUM_CHUNK_ROWS, int(chunk_bytes) // max(1, int(rowsize)))


def _key_order(keys: Iterable[str]) -> list[str]:
    names = [key.strip("/") for key in keys]
    entities = [name for name in ENTITY_ORDER if name in names]
    return entities + sorted(name for name in names if name not in ENTITY_ORDER)


def describe_h5(
    path: Path,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    chunk_rows: int | None = None,
) -> dict[str, Any]:
    """The manifest entry of one HDF5 file (tables first, entities in order)."""

    path = Path(path)
    sha256, size_bytes = sha256_file(path), path.stat().st_size
    tables: list[dict[str, Any]] = []
    other_keys: list[dict[str, Any]] = []
    with pd.HDFStore(str(path), mode="r") as store:
        root_attributes = _root_attributes(store)
        for key in _key_order(store.keys()):
            storer = store.get_storer(key)
            raw_type = getattr(storer, "pandas_type", None)
            pandas_type = None if raw_type is None else str(raw_type)
            nrows = _storer_rows(storer)
            if pandas_type not in TABLE_PANDAS_TYPES or key.startswith("_"):
                other_keys.append(_other_key(store, path, key, pandas_type, nrows))
                continue
            step = chunk_rows or _chunk_rows(store, key, storer, chunk_bytes)
            tally = _ColumnTally(key)
            if pandas_type == "frame":
                tally.add(store.select(key))
            else:
                tally.add(store.select(key, start=0, stop=0))
                for start in range(0, nrows, step):
                    tally.add(
                        store.select(key, start=start, stop=min(nrows, start + step))
                    )
                for name, seen in tally.seen.items():
                    if len(seen) > 1:
                        whole_column = store.select(key, columns=[name])[name]
                        tally.resolved_dtypes[name] = str(whole_column.dtype)
            if tally.rows != nrows:
                raise ManifestError(
                    f"Table {key!r} read {tally.rows} rows; its storer says {nrows}."
                )
            tables.append(
                tally.entry(
                    storage="table" if pandas_type == "frame_table" else "fixed",
                    logical=_codec_columns(storer, key),
                )
            )
    return {
        "name": path.name,
        "format": "hdf5",
        "sha256": sha256,
        "size_bytes": size_bytes,
        "tables": tables,
        "other_keys": other_keys,
        "root_attributes": root_attributes,
    }


def _short(value: Any) -> Any:
    """A JSON-safe scalar, or a note of its type when it is long or not scalar."""

    if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, bool | int) or (
        isinstance(value, float) and value == value and abs(value) != float("inf")
    ):
        return value
    if isinstance(value, str) and len(value) <= MAX_METADATA_TEXT:
        return value
    return f"<{type(value).__name__}, not shown>"


def _root_attributes(store: pd.HDFStore) -> dict[str, Any]:
    attributes = store.get_node("/")._v_attrs
    return {
        name: _short(attributes[name]) for name in sorted(attributes._v_attrnamesuser)
    }


def _other_key(
    store: pd.HDFStore, path: Path, key: str, pandas_type: Any, nrows: int
) -> dict[str, Any]:
    """A metadata key: ``_time_period``, a Microcosm metadata row, or a non-frame."""

    entry: dict[str, Any] = {"key": key, "pandas_type": pandas_type, "rows": nrows}
    if key == measures.TIME_PERIOD_KEY:
        entry["value"] = measures.read_time_period(store, path)
    elif pandas_type in TABLE_PANDAS_TYPES:
        entry["columns"] = [str(name) for name in store.select(key, start=0, stop=0)]
    elif nrows == 1 and pandas_type in ("series", "series_table"):
        entry["value"] = _short(store.select(key).iloc[0])
    return entry


def _csv_table_name(path: Path) -> str:
    name = path.name
    if name.endswith(measures.SIDECAR_SUFFIX):
        return "household_measures"
    for suffix in (".csv.gz", ".csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def describe_csv(
    path: Path,
    *,
    table: str | None = None,
    chunk_rows: int = DEFAULT_CSV_CHUNK_ROWS,
) -> dict[str, Any]:
    """The manifest entry of one CSV (optionally gzip) file, one table."""

    path = Path(path)
    sha256, size_bytes = sha256_file(path), path.stat().st_size
    name = table or _csv_table_name(path)
    options: dict[str, Any] = {"chunksize": chunk_rows}
    if path.name.endswith(measures.SIDECAR_SUFFIX):
        options["dtype"] = dict(measures.SIDECAR_COLUMNS)
        options["float_precision"] = "round_trip"
    tally = _ColumnTally(name)
    with pd.read_csv(path, **options) as reader:
        for chunk in reader:
            tally.add(chunk)
    if tally.columns is None:
        raise ManifestError(f"{path} has no header row.")
    return {
        "name": path.name,
        "format": "csv",
        "sha256": sha256,
        "size_bytes": size_bytes,
        "tables": [tally.entry(storage="csv")],
        "other_keys": [],
    }


def describe_file(path: Path, **options: Any) -> dict[str, Any]:
    name = Path(path).name
    if name.endswith((".h5", ".hdf5", ".hdf")):
        return describe_h5(path, **options)
    if name.endswith((".csv", ".csv.gz")):
        return describe_csv(path)
    raise ManifestError(f"No reader for {name}; expected .h5 or .csv(.gz).")


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


def build_manifest(
    files: Sequence[Mapping[str, Any]],
    *,
    diff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "files": [dict(entry) for entry in files],
        "diff": dict(diff) if diff is not None else None,
    }
    column_index(manifest)  # refuses a table or column named twice
    return manifest


def column_index(manifest: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """``(table, column) -> column entry`` over every file of ``manifest``."""

    index: dict[tuple[str, str], dict[str, Any]] = {}
    seen_tables: dict[str, str] = {}
    for entry in manifest.get("files", []):
        for table in entry.get("tables", []):
            table_name = table["name"]
            if table_name in seen_tables:
                raise ManifestError(
                    f"Table {table_name!r} appears in both {seen_tables[table_name]} "
                    f"and {entry['name']}."
                )
            seen_tables[table_name] = entry["name"]
            for column in table.get("columns", []):
                key = (table_name, column["name"])
                if key in index:
                    raise ManifestError(f"Column {key[0]}.{key[1]} is listed twice.")
                index[key] = dict(column)
    return index


def _tables(manifest: Mapping[str, Any]) -> list[str]:
    return [
        table["name"]
        for entry in manifest.get("files", [])
        for table in entry.get("tables", [])
    ]


def load_column_list(path: Path) -> dict[str, Any]:
    """A ``table,column,dtype`` CSV as a manifest without row or null counts."""

    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or [
            name.strip() for name in reader.fieldnames
        ] != ["table", "column", "dtype"]:
            raise ManifestError(
                f"{path} must have the header table,column,dtype; got "
                f"{reader.fieldnames}."
            )
        tables: dict[str, list[dict[str, Any]]] = {}
        for line, row in enumerate(reader, start=2):
            if None in row or None in row.values():
                raise ManifestError(f"{path}:{line} does not have exactly 3 fields.")
            values = {key.strip(): (value or "").strip() for key, value in row.items()}
            if not all(values.get(key) for key in ("table", "column", "dtype")):
                raise ManifestError(f"{path}:{line} has an empty field.")
            tables.setdefault(values["table"], []).append(
                {"name": values["column"], "dtype": values["dtype"], "null_count": None}
            )
    entry = {
        "name": path.name,
        "format": COLUMN_LIST_FORMAT,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "tables": [
            {"name": name, "storage": None, "rows": None, "columns": columns}
            for name, columns in tables.items()
        ],
        "other_keys": [],
    }
    return build_manifest([entry])


def load_manifest(path: Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    if document.get("kind") != MANIFEST_KIND:
        raise ManifestError(
            f"{path} is not a {MANIFEST_KIND} (kind {document.get('kind')!r})."
        )
    if document.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"{path} has schema_version {document.get('schema_version')!r}; this "
            f"tool reads {MANIFEST_SCHEMA_VERSION}."
        )
    column_index(document)
    return document


def load_prior(path: Path) -> dict[str, Any]:
    if Path(path).suffix.lower() == ".csv":
        return load_column_list(path)
    return load_manifest(path)


def manifest_column_list(manifest: Mapping[str, Any]) -> str:
    """The manifest's columns as ``table,column,dtype`` CSV text."""

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("table", "column", "dtype"))
    for (table, column), entry in column_index(manifest).items():
        writer.writerow((table, column, entry["dtype"]))
    return output.getvalue()


def load_renames(path: Path | None) -> tuple[dict[str, str], ...]:
    declared = [dict(item) for item in DECLARED_RENAMES]
    if path is not None:
        extra = json.loads(Path(path).read_text())
        if not isinstance(extra, list):
            raise ManifestError(f"{path} must hold a JSON list of renames.")
        declared.extend(dict(item) for item in extra)
    return validate_renames(declared)


def validate_renames(
    renames: Iterable[Mapping[str, str]],
) -> tuple[dict[str, str], ...]:
    checked: list[dict[str, str]] = []
    sources: set[tuple[str, str]] = set()
    targets: set[tuple[str, str]] = set()
    for item in renames:
        if set(item) - {"table", "from", "to", "reason"} or not all(
            isinstance(item.get(key), str) and item.get(key)
            for key in ("table", "from", "to")
        ):
            raise ManifestError(
                f"A declared rename needs non-empty table, from and to (and an "
                f"optional reason); got {dict(item)!r}."
            )
        source = (item["table"], item["from"])
        target = (item["table"], item["to"])
        if item["from"] == item["to"]:
            raise ManifestError(f"Rename {source} renames a column to itself.")
        if source in sources or target in targets:
            raise ManifestError(f"Rename {source} -> {target} is declared twice.")
        sources.add(source)
        targets.add(target)
        checked.append(dict(item))
    if sources & targets:
        raise ManifestError(f"Renames chain through {sorted(sources & targets)}.")
    return tuple(checked)


def diff_manifests(
    prior: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    renames: Iterable[Mapping[str, str]] = DECLARED_RENAMES,
    prior_label: str | None = None,
) -> dict[str, Any]:
    """Columns added, removed, renamed (declared only) and retyped.

    Keyed by ``(table, column)``. A declared rename applies when the prior
    has ``from`` but not ``to`` and the current has ``to`` but not ``from``;
    it then appears under ``renamed`` instead of under ``removed`` and
    ``added``, with both dtypes. A declaration that does not apply is listed,
    with the reason, under ``declared_renames_not_applied``. A column kept
    under its name whose dtype changed appears under ``dtype_changed``, with
    its current ``logical_dtype`` when it has one.
    """

    before = column_index(prior)
    after = column_index(current)
    removed = set(before) - set(after)
    added = set(after) - set(before)
    renamed: list[dict[str, Any]] = []
    not_applied: list[dict[str, Any]] = []
    for item in validate_renames(renames):
        source = (item["table"], item["from"])
        target = (item["table"], item["to"])
        reasons = []
        if source not in before:
            reasons.append("the prior has no column 'from'")
        if target in before:
            reasons.append("the prior already has 'to'")
        if target not in after:
            reasons.append("the current has no column 'to'")
        if source in after:
            reasons.append("the current still has 'from'")
        if reasons:
            not_applied.append({**item, "why": "; ".join(reasons)})
            continue
        removed.discard(source)
        added.discard(target)
        renamed.append(
            {
                "table": item["table"],
                "from": item["from"],
                "to": item["to"],
                "dtype_before": before[source]["dtype"],
                "dtype_after": after[target]["dtype"],
                **({"reason": item["reason"]} if item.get("reason") else {}),
            }
        )
    dtype_changed = [
        {
            "table": table,
            "column": column,
            "dtype_before": before[(table, column)]["dtype"],
            "dtype_after": after[(table, column)]["dtype"],
            **(
                {"logical_dtype_after": after[(table, column)]["logical_dtype"]}
                if after[(table, column)].get("logical_dtype")
                else {}
            ),
        }
        for (table, column) in sorted(set(before) & set(after))
        if before[(table, column)]["dtype"] != after[(table, column)]["dtype"]
    ]
    prior_tables, current_tables = _tables(prior), _tables(current)

    def listed(keys: set[tuple[str, str]], index: Mapping) -> list[dict[str, Any]]:
        return [
            {"table": table, "column": column, "dtype": index[(table, column)]["dtype"]}
            for table, column in sorted(keys)
        ]

    return {
        "prior": {
            "label": prior_label,
            "files": [
                {"name": entry["name"], "sha256": entry.get("sha256")}
                for entry in prior.get("files", [])
            ],
        },
        "tables_added": sorted(set(current_tables) - set(prior_tables)),
        "tables_removed": sorted(set(prior_tables) - set(current_tables)),
        "added": listed(added, after),
        "removed": listed(removed, before),
        "renamed": sorted(renamed, key=lambda item: (item["table"], item["from"])),
        "dtype_changed": dtype_changed,
        "declared_renames_not_applied": sorted(
            not_applied, key=lambda item: (item["table"], item["from"])
        ),
    }


def diff_is_empty(diff: Mapping[str, Any]) -> bool:
    return not any(
        diff[key]
        for key in (
            "tables_added",
            "tables_removed",
            "added",
            "removed",
            "renamed",
            "dtype_changed",
        )
    )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value).replace("|", "\\|")


def _row(*cells: Any) -> str:
    return "| " + " | ".join(_cell(cell) for cell in cells) + " |"


def render_markdown(manifest: Mapping[str, Any], *, title: str | None = None) -> str:
    """A publication rendering: files, tables, every column, then the diff."""

    lines = [f"# {title or 'Schema manifest'}", ""]
    lines += [
        f"Manifest kind `{manifest['kind']}`, schema version "
        f"{manifest['schema_version']}. Dtypes are as pandas reads each file; "
        "null counts count missing values in every row.",
        "",
        "## Files",
        "",
        _row("File", "Format", "Size (bytes)", "SHA-256"),
        _row("---", "---", "---:", "---"),
    ]
    for entry in manifest["files"]:
        lines.append(
            _row(
                entry["name"],
                entry["format"],
                entry.get("size_bytes"),
                entry.get("sha256"),
            )
        )
    metadata = [
        (entry["name"], f"key `{key['key']}`", key.get("value", key.get("columns")))
        for entry in manifest["files"]
        for key in entry.get("other_keys", [])
    ] + [
        (entry["name"], f"root attribute `{name}`", value)
        for entry in manifest["files"]
        for name, value in (entry.get("root_attributes") or {}).items()
    ]
    if metadata:
        lines += ["", "## Metadata", "", _row("File", "Item", "Value")]
        lines.append(_row("---", "---", "---"))
        lines += [
            # str() so a year is not printed with a thousands separator.
            _row(name, item, str(value))
            for name, item, value in metadata
        ]
    lines += ["", "## Tables", "", _row("Table", "File", "Storage", "Rows", "Columns")]
    lines.append(_row("---", "---", "---", "---:", "---:"))
    for entry in manifest["files"]:
        for table in entry["tables"]:
            lines.append(
                _row(
                    table["name"],
                    entry["name"],
                    table.get("storage"),
                    table.get("rows"),
                    len(table["columns"]),
                )
            )
    for entry in manifest["files"]:
        for table in entry["tables"]:
            lines += [
                "",
                f"### `{table['name']}`",
                "",
                _row("Column", "Dtype", "Logical dtype", "Nulls"),
                _row("---", "---", "---", "---:"),
            ]
            for column in table["columns"]:
                lines.append(
                    _row(
                        f"`{column['name']}`",
                        column["dtype"],
                        column.get("logical_dtype", ""),
                        column.get("null_count"),
                    )
                )
    diff = manifest.get("diff")
    lines += ["", "## Changes from the prior schema", ""]
    if not diff:
        lines.append("No prior schema was given.")
        return "\n".join(lines) + "\n"
    prior_names = ", ".join(f"`{item['name']}`" for item in diff["prior"]["files"])
    label = diff["prior"].get("label")
    lines.append(f"Compared with {label + ': ' if label else ''}{prior_names}.")
    if diff_is_empty(diff):
        lines += ["", "No table or column changed."]
    for heading, key in (
        ("Tables added", "tables_added"),
        ("Tables removed", "tables_removed"),
    ):
        if diff[key]:
            lines += ["", f"### {heading}", ""]
            lines += [f"- `{name}`" for name in diff[key]]
    for heading, key in (("Columns added", "added"), ("Columns removed", "removed")):
        if diff[key]:
            lines += ["", f"### {heading}", "", _row("Table", "Column", "Dtype")]
            lines.append(_row("---", "---", "---"))
            lines += [
                _row(item["table"], f"`{item['column']}`", item["dtype"])
                for item in diff[key]
            ]
    if diff["renamed"]:
        lines += [
            "",
            "### Columns renamed",
            "",
            _row("Table", "From", "To", "Dtype", "Why"),
        ]
        lines.append(_row("---", "---", "---", "---", "---"))
        for item in diff["renamed"]:
            dtype = (
                item["dtype_after"]
                if item["dtype_before"] == item["dtype_after"]
                else f"{item['dtype_before']} -> {item['dtype_after']}"
            )
            lines.append(
                _row(
                    item["table"],
                    f"`{item['from']}`",
                    f"`{item['to']}`",
                    dtype,
                    item.get("reason", ""),
                )
            )
    if diff["dtype_changed"]:
        lines += [
            "",
            "### Dtypes changed",
            "",
            _row("Table", "Column", "Before", "After"),
        ]
        lines.append(_row("---", "---", "---", "---"))
        lines += [
            _row(
                item["table"],
                f"`{item['column']}`",
                item["dtype_before"],
                item["dtype_after"],
            )
            for item in diff["dtype_changed"]
        ]
    if diff["declared_renames_not_applied"]:
        lines += ["", "### Declared renames that did not apply", ""]
        lines += [
            f"- `{item['table']}.{item['from']}` -> `{item['to']}`: {item['why']}."
            for item in diff["declared_renames_not_applied"]
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _write_exclusive(path: Path, text: str) -> None:
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(f"{path} exists; refusing to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.{secrets.token_hex(16)}.partial"
    handle = os.open(temporary_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(text)
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def manifest_json(manifest: Mapping[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"


def emit(
    *,
    h5: Path,
    sidecars: Sequence[Path] = (),
    prior: Path | None = None,
    renames: Path | None = None,
    prior_label: str | None = None,
    describe: Callable[[Path], dict[str, Any]] = describe_file,
) -> dict[str, Any]:
    files = [describe(Path(h5)), *(describe(Path(path)) for path in sidecars)]
    manifest = build_manifest(files)
    if prior is not None:
        manifest["diff"] = diff_manifests(
            load_prior(Path(prior)),
            manifest,
            renames=load_renames(renames),
            prior_label=prior_label or Path(prior).name,
        )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, action="append", default=[])
    parser.add_argument("--prior", type=Path, default=None)
    parser.add_argument("--prior-label", default=None)
    parser.add_argument("--renames", type=Path, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_json.resolve() == args.output_md.resolve():
        raise ManifestError("--output-json and --output-md must be different files.")
    for path in (args.output_json, args.output_md):
        if os.path.lexists(path):
            raise FileExistsError(f"{path} exists; refusing to overwrite it.")
    manifest = emit(
        h5=args.h5,
        sidecars=args.sidecar,
        prior=args.prior,
        renames=args.renames,
        prior_label=args.prior_label,
    )
    _write_exclusive(args.output_json, manifest_json(manifest))
    try:
        _write_exclusive(args.output_md, render_markdown(manifest, title=args.title))
    except BaseException:
        args.output_json.unlink()
        raise
    print(f"wrote {args.output_json} and {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
