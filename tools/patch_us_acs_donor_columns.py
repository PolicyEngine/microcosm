"""Write a patched copy of a US national donor H5 for the ACS staging (R1).

The County Files proof of concept (microcosm#922, plan microcosm#1647) stages
its ACS rows from the national donor of build 766b7c7
(``populace-us-2024-spm-receipts-20260923``). The ACS staging tool
(``tools/_legacy/build_us_acs_multispine_base.py``, R1) refuses that donor:
it predates three later donor contracts. This tool writes a patched copy of
the donor for the proof of concept only. It reads the published file and
never changes it. The real County Files donor will be a new national build
from ``main``.

The copy keeps every table, column, dtype and HDF5 root attribute of the
input. Each patch step only adds named columns to the entity tables:

- ``reviewed_zero_columns`` adds reviewed all-zero columns. It looks up each
  column's entity, dtype and default in the US engine and refuses a column
  whose engine default is not 0. ``REVIEWED_ZERO_COLUMNS`` holds the reason
  for each column.
- The ASEC work-column restore step (:func:`asec_work_column_step`) is
  reserved for a later commit. It restores ``detailed_industry_recode``,
  ``major_industry_recode`` and ``worked_last_year`` from the pinned ASEC
  person files.

A step never overwrites data: when a column is already present, the step
must give exactly the stored values, or the tool refuses the input. The tool
writes the copy atomically, verifies it against the input, and writes a JSON
receipt with both SHA-256 digests, the row counts per entity, each added
column with its entity and reason, and the tool's commit.

Two options record lineage (both optional):

- ``--input-release-manifest`` is the published donor's
  ``release_manifest.json``. The input's SHA-256 must match its one microdata
  artifact, and the receipt records the donor build it names.
- ``--release-manifest`` with ``--manifest-release`` and ``--manifest-build``
  writes a release manifest in the build-folder layout that names the copy
  and its receipt. Upload the copy, the receipt and this file (as
  ``release_manifest.json``) to ``releases/<release>/<build>/`` in
  ``--manifest-repo-id``, where ``referenced_build`` reads a build. R1's
  ``--donor-release-manifest`` then verifies the copy against it and records
  the copy's build as ``donor_release``, which the local-area release
  contract requires before publication.

Usage::

    python tools/patch_us_acs_donor_columns.py \\
        --input-h5 populace_us_2024_receipt_qualified.h5 \\
        --input-release-manifest published_release_manifest.json \\
        --output-h5 run/donor_patched_poc.h5 \\
        --release-manifest run/release_manifest.json \\
        --manifest-release RELEASE --manifest-build BUILD

R1 then reads the copy with ``--base-h5 run/donor_patched_poc.h5`` and
``--donor-release-manifest run/release_manifest.json``, or without a manifest
(``donor_release`` is then null, and the publisher refuses the release).
"""

# No ``from __future__ import annotations``: test loaders run this file
# without registering it in ``sys.modules``, where string annotations break
# the dataclasses below.
import argparse
import hashlib
import json
import numbers
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.h5_io import (
    assert_h5_unchanged,
    refuse_denied_pool_h5,
)
from microcosm.data.build_folder_manifests import referenced_build
from microcosm.frame import put_frame_table, read_frame_table
from microcosm.frame.materialize import _PYTABLES_FRAME_TABLE_CODEC_ATTR
from microcosm.frame.units import US_SCHEMA

TOOL = "tools/patch_us_acs_donor_columns.py"
RECEIPT_VERSION = 1
ISSUE = "microcosm#922"
PLAN = "microcosm#1647"
PURPOSE = (
    "County Files proof of concept only: a patched copy of the published "
    "national donor that the ACS staging (R1) accepts. The real County Files "
    "donor is a new national build from main."
)
_CONSUMER = f"US ACS donor column patch ({TOOL})"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

#: The repository the amended ruling names for the copy (microcosm#1647).
DEFAULT_MANIFEST_REPO_ID = "policyengine/populace-us-staging"
#: The copy's dataset role: not a release role, so the release contract
#: refuses the copy's folder as a release of its own.
COPY_DATASET_ROLE = "proof_of_concept_donor_copy"
#: A release or build name is one folder name (``releases/<release>/<build>``).
_FOLDER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")

#: Each reviewed all-zero column and the reason its true values are unknown
#: or zero on the donor. The fill must equal the engine default (0).
REVIEWED_ZERO_COLUMNS: Mapping[str, str] = MappingProxyType(
    {
        "strike_benefits": "no donor or ASEC pppub23–26 row has OI_OFF == 12",
        "passive_partnership_s_corp_income": (
            "the donor holds partnership and S-corp income as one combined "
            "amount in `partnership_income`; the split is unknown"
        ),
    }
)
ZERO_STEP = "reviewed_zero_columns"

#: The ASEC work columns that R1's ``_require_donor_work_experience`` needs
#: and that the reserved restore step adds (see :func:`asec_work_column_step`).
ASEC_WORK_COLUMNS: tuple[str, ...] = (
    "detailed_industry_recode",
    "major_industry_recode",
    "worked_last_year",
)
ASEC_WORK_STEP = "asec_work_columns"

_NUMERIC_DTYPES: Mapping[str, type[np.generic]] = MappingProxyType(
    {"float": np.float64, "int": np.int64}
)


@dataclass(frozen=True)
class ColumnPatch:
    """One column that a step adds to one entity table.

    ``values`` must have the entity table's index. ``record`` goes into the
    receipt with the column, entity, dtype, step and action.
    """

    entity: str
    column: str
    values: pd.Series
    record: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PatchStepResult:
    """A step's column patches and its step-level receipt (sources, counts)."""

    patches: tuple[ColumnPatch, ...]
    receipt: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PatchStep:
    """A named step. ``run`` gets the entity tables after earlier steps.

    ``run`` must not change the tables it gets; it returns column patches.
    """

    name: str
    run: Callable[[Mapping[str, pd.DataFrame]], PatchStepResult]


def reviewed_zero_column_step(columns: Sequence[str], *, engine: Any) -> PatchStep:
    """Add the named reviewed columns as zeros on their engine entity.

    ``engine`` is the US engine adapter (or a stand-in with the same
    ``variable_metadata`` and ``default_values``). Each column must be in
    ``REVIEWED_ZERO_COLUMNS``, be a numeric engine input with a default, and
    have the engine default 0, so the fill is the value the engine would use.
    """

    unknown = sorted(set(columns) - set(REVIEWED_ZERO_COLUMNS))
    if unknown:
        raise ValueError(
            f"No reviewed reason for zero column(s) {unknown}; reviewed columns "
            f"are {sorted(REVIEWED_ZERO_COLUMNS)}."
        )
    if len(set(columns)) != len(columns) or not columns:
        raise ValueError(
            f"Zero columns must be distinct and non-empty; got {list(columns)}."
        )
    specs: list[tuple[str, str, str, object]] = []
    for column in columns:
        metadata = engine.variable_metadata(column)
        if metadata.dtype not in _NUMERIC_DTYPES:
            raise ValueError(
                f"{column}: the engine dtype is {metadata.dtype!r}; a zero fill "
                "needs a numeric input."
            )
        defaults = engine.default_values([column])
        if column not in defaults:
            raise ValueError(
                f"{column}: the engine reports no default for it (an unknown, "
                "formula-owned or declared source input); a zero fill must "
                "equal the engine default."
            )
        default = defaults[column]
        if isinstance(default, np.generic):
            default = default.item()
        if isinstance(default, bool) or not isinstance(default, numbers.Real):
            raise ValueError(f"{column}: engine default {default!r} is not numeric.")
        if default != 0:
            raise ValueError(
                f"{column}: the engine default is {default!r}, not 0; a zero "
                "fill would differ from the engine default."
            )
        specs.append((column, metadata.entity, metadata.dtype, default))

    def run(tables: Mapping[str, pd.DataFrame]) -> PatchStepResult:
        patches = []
        for column, entity, dtype, default in specs:
            if entity not in tables:
                raise ValueError(
                    f"{column}: the engine entity {entity!r} has no table in "
                    "the donor H5."
                )
            index = tables[entity].index
            patches.append(
                ColumnPatch(
                    entity=entity,
                    column=column,
                    values=pd.Series(
                        np.zeros(len(index), dtype=_NUMERIC_DTYPES[dtype]),
                        index=index,
                        name=column,
                    ),
                    record={
                        "engine_dtype": dtype,
                        "engine_default": default,
                        "value": 0,
                        "reason": REVIEWED_ZERO_COLUMNS[column],
                    },
                )
            )
        return PatchStepResult(patches=tuple(patches))

    return PatchStep(name=ZERO_STEP, run=run)


def asec_work_column_step(*args: object, **kwargs: object) -> PatchStep:
    """Restore ``ASEC_WORK_COLUMNS`` from the pinned ASEC person files.

    Reserved for the next commit on this tool, which Max's agents add from
    their verified local step. Its contract:

    - Read the pinned ASEC person files (``pppub23``-``pppub26``) and refuse a
      file whose SHA-256 is not the pinned one.
    - Match each donor person on its source year and ``PERIDNUM``. Refuse a
      donor person with no match or with more than one match.
    - Return a :class:`PatchStep` named ``ASEC_WORK_STEP``. Its ``run``
      returns a :class:`PatchStepResult` with one :class:`ColumnPatch` per
      column on ``person`` (the source item in each ``record``), and the
      files, their digests and the match counts in the step ``receipt``.
    - Register it in :func:`build_patch_steps` before the zero step, with its
      command-line options in :func:`_parse_args`.

    R1's ``_require_donor_work_experience`` must then pass on the copy.
    """

    raise NotImplementedError(
        "The ASEC work-column restore step is not in this commit; the next "
        "commit on this tool adds it (microcosm#1647)."
    )


@dataclass(frozen=True)
class CopyReleaseManifest:
    """Where to write the copy's release manifest, and the build it names."""

    path: Path
    release: str
    build: str
    repo_id: str = DEFAULT_MANIFEST_REPO_ID

    def __post_init__(self) -> None:
        for name in ("release", "build"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _FOLDER_NAME.fullmatch(value):
                raise ValueError(
                    f"The copy's manifest {name} must be one folder name "
                    f"(letters, digits, '.', '_', '-'); got {value!r}."
                )
        if not isinstance(self.repo_id, str) or not _REPO_ID.fullmatch(self.repo_id):
            raise ValueError(
                f"The copy's manifest repo_id must be OWNER/NAME; got {self.repo_id!r}."
            )

    def identity(self) -> dict[str, str]:
        return {
            "path": str(Path(self.path).resolve()),
            "repo_id": self.repo_id,
            "release": self.release,
            "build": self.build,
            "folder": referenced_build(
                {"repo_id": self.repo_id, "release": self.release, "build": self.build}
            ).folder,
        }


def input_release_identity(manifest_path: Path, input_sha256: str) -> dict[str, object]:
    """Verify the input against its published release manifest; name its build.

    The manifest must carry exactly one microdata artifact, whose SHA-256 is
    the input's, and name a build that ``referenced_build`` accepts: a build
    folder (``release`` and ``build``) or a tag (``build.build_id`` and the
    artifact's ``revision``).
    """

    manifest_path = Path(manifest_path)
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Cannot read --input-release-manifest {manifest_path}: {error}."
        ) from error
    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
    if not isinstance(artifacts, dict):
        raise ValueError(f"{manifest_path} carries no artifacts map.")
    microdata = [
        (name, entry)
        for name, entry in artifacts.items()
        if isinstance(entry, dict) and entry.get("kind") == "microdata"
    ]
    if len(microdata) != 1:
        raise ValueError(
            f"{manifest_path} must carry exactly one microdata artifact; found "
            f"{sorted(name for name, _ in microdata)}."
        )
    name, entry = microdata[0]
    if entry.get("sha256") != input_sha256:
        raise ValueError(
            f"--input-h5 has SHA-256 {input_sha256}, but {manifest_path} pins "
            f"{entry.get('sha256')!r} for {name!r}."
        )
    build = manifest.get("build")
    if isinstance(build, dict):
        reference = {
            "repo_id": entry.get("repo_id"),
            "release_id": build.get("build_id"),
            "revision": entry.get("revision"),
        }
    else:
        reference = {
            "repo_id": entry.get("repo_id"),
            "release": manifest.get("release"),
            "build": build,
        }
    try:
        location = referenced_build(reference)
    except ValueError as error:
        raise ValueError(f"{manifest_path} names no readable build: {error}") from error
    return {
        "manifest": manifest_path.name,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact": name,
        **{key: value for key, value in reference.items() if value is not None},
        "folder": location.folder,
        "sha256": input_sha256,
    }


def copy_release_manifest(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
    receipt_sha256: str,
    target: CopyReleaseManifest,
    built_at: str,
) -> dict[str, object]:
    """The copy's release manifest in the build-folder layout.

    Its one microdata artifact is the copy, so R1's
    ``--donor-release-manifest`` check (the base H5's SHA-256 against it)
    passes on the copy only. ``derived_from`` names the input; ``patch``
    names the tool, its commit, the receipt and the added columns.
    """

    output, source = receipt["output"], receipt["input"]
    assert isinstance(output, Mapping) and isinstance(source, Mapping)
    output_path = Path(str(output["path"]))
    added = receipt["added_columns"]
    assert isinstance(added, list)
    return {
        "schema_version": 1,
        "release": target.release,
        "build": target.build,
        "built_at": built_at,
        "dataset_role": COPY_DATASET_ROLE,
        "is_default": False,
        "issue": ISSUE,
        "plan": PLAN,
        "purpose": PURPOSE,
        "derived_from": {
            "file": Path(str(source["path"])).name,
            "sha256": source["sha256"],
            "rows": source["rows"],
            "release": source.get("release"),
        },
        "patch": {
            "tool": TOOL,
            "tool_commit": receipt.get("tool_commit"),
            "engine": receipt.get("engine"),
            "receipt": Path(receipt_path).name,
            "added_columns": [f"{r['entity']}.{r['column']}" for r in added],
        },
        "artifacts": {
            output_path.stem: {
                "kind": "microdata",
                "path": output_path.name,
                "repo_id": target.repo_id,
                "sha256": output["sha256"],
                "role": "dataset",
            },
            "patch_receipt": {
                "kind": "receipt",
                "path": Path(receipt_path).name,
                "repo_id": target.repo_id,
                "sha256": receipt_sha256,
                "role": "patch_receipt",
            },
        },
    }


def build_patch_steps(
    *,
    zero_columns: Sequence[str],
    engine: Any,
) -> list[PatchStep]:
    """The steps the command runs, in order.

    The ASEC work-column restore step (:func:`asec_work_column_step`) goes
    first when it lands, then the reviewed zero columns.
    """

    return [reviewed_zero_column_step(zero_columns, engine=engine)]


def patch_donor_h5(
    input_h5: Path,
    output_h5: Path,
    *,
    steps: Sequence[PatchStep],
    receipt_path: Path,
    tool_identity: Mapping[str, object] | None = None,
    engine_identity: Mapping[str, object] | None = None,
    input_release_manifest: Path | None = None,
    release_manifest: CopyReleaseManifest | None = None,
) -> dict[str, object]:
    """Write the patched copy and its receipt; return the receipt.

    The input must have every US entity table that R1's ``_load_base_frame``
    reads. The copy starts as a byte copy of the input; only the entity
    tables a step changes are rewritten, through the shared frame-table
    codec, in their stored HDF format. The copy is verified before it
    replaces ``output_h5``. ``input_release_manifest`` is checked before any
    work; ``release_manifest`` is written last, after the receipt.
    """

    input_h5, output_h5 = Path(input_h5), Path(output_h5)
    receipt_path = Path(receipt_path)
    _validate_paths(
        input_h5,
        output_h5,
        receipt_path,
        *(() if input_release_manifest is None else (Path(input_release_manifest),)),
        *(() if release_manifest is None else (Path(release_manifest.path),)),
    )
    input_sha256 = refuse_denied_pool_h5(input_h5, consumer=_CONSUMER)
    input_release = (
        None
        if input_release_manifest is None
        else input_release_identity(Path(input_release_manifest), input_sha256)
    )
    with pd.HDFStore(input_h5, mode="r") as store:
        keys = sorted(store.keys())
        absent = [entity for entity in US_SCHEMA.entities if f"/{entity}" not in keys]
        if absent:
            raise ValueError(
                f"{input_h5} lacks entity table(s) {absent}; R1 reads every "
                f"US entity table ({list(US_SCHEMA.entities)})."
            )
        tables = {
            entity: read_frame_table(store, entity) for entity in US_SCHEMA.entities
        }
        layouts = {
            entity: _stored_layout(store, entity, tables[entity])
            for entity in US_SCHEMA.entities
        }
        other_keys = [key for key in keys if key.lstrip("/") not in tables]
        others = {key: store[key] for key in other_keys}
        root_attributes = _root_attributes(store)
    assert_h5_unchanged(input_h5, input_sha256, consumer=_CONSUMER)

    patched, columns, step_receipts = _apply_steps(tables, steps)
    changed = sorted(
        entity for entity in patched if patched[entity] is not tables[entity]
    )

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_h5.with_name(f".{output_h5.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(input_h5, temporary)
        _rewrite_entity_tables(
            temporary, {entity: patched[entity] for entity in changed}, layouts
        )
        _verify_copy(
            temporary,
            original=tables,
            patched=patched,
            formats={entity: layout.format for entity, layout in layouts.items()},
            keys=keys,
            others=others,
            root_attributes=root_attributes,
        )
        os.replace(temporary, output_h5)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    assert_h5_unchanged(input_h5, input_sha256, consumer=_CONSUMER)

    receipt: dict[str, object] = {
        "version": RECEIPT_VERSION,
        "tool": TOOL,
        "tool_commit": dict(tool_identity) if tool_identity is not None else None,
        "issue": ISSUE,
        "plan": PLAN,
        "purpose": PURPOSE,
        "engine": dict(engine_identity) if engine_identity is not None else None,
        "input": {
            "path": str(input_h5.resolve()),
            "sha256": input_sha256,
            "rows": {entity: len(tables[entity]) for entity in US_SCHEMA.entities},
            "release": input_release,
        },
        "output": {
            "path": str(output_h5.resolve()),
            "sha256": _sha256(output_h5),
            "rows": {entity: len(patched[entity]) for entity in US_SCHEMA.entities},
        },
        "rewritten_tables": changed,
        "preserved": {
            "tables": keys,
            "root_attributes": sorted(root_attributes),
        },
        "added_columns": [r for r in columns if r["action"] == "added"],
        "kept_columns": [r for r in columns if r["action"] == "kept_existing"],
        "steps": step_receipts,
        "release_manifest": (
            None if release_manifest is None else release_manifest.identity()
        ),
    }
    receipt_sha256 = _write_json(receipt_path, receipt)
    if release_manifest is not None:
        _write_json(
            Path(release_manifest.path),
            copy_release_manifest(
                receipt,
                receipt_path=receipt_path,
                receipt_sha256=receipt_sha256,
                target=release_manifest,
                built_at=datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
    return receipt


def _write_json(path: Path, document: Mapping[str, object]) -> str:
    """Write ``document`` as sorted, indented JSON; return the bytes' SHA-256."""

    data = (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class _StoredLayout:
    """How the input stored one entity table, so the rewrite can keep it."""

    format: str
    data_columns: bool | tuple[str, ...] | None
    nullable_boolean_columns: tuple[str, ...]


def _stored_layout(
    store: pd.HDFStore, entity: str, table: pd.DataFrame
) -> _StoredLayout:
    storer = store.get_storer(entity)
    raw_codec = getattr(storer.attrs, _PYTABLES_FRAME_TABLE_CODEC_ATTR, None)
    nullable = (
        tuple(json.loads(raw_codec)["nullable_boolean_columns"])
        if raw_codec is not None
        else ()
    )
    if not storer.is_table:
        return _StoredLayout("fixed", None, nullable)
    data_columns = tuple(getattr(storer, "data_columns", None) or ())
    if data_columns and set(data_columns) == set(table.columns):
        return _StoredLayout("table", True, nullable)
    return _StoredLayout("table", data_columns or None, nullable)


def _rewrite_entity_tables(
    path: Path,
    tables: Mapping[str, pd.DataFrame],
    layouts: Mapping[str, _StoredLayout],
) -> None:
    """Replace the changed entity tables in the copy, in their stored layout.

    Columns the input's frame-table codec declared nullable Boolean go back
    to ``boolean`` dtype first, so the shared codec writes them, and its
    table attribute, as the input had them.
    """

    with pd.HDFStore(path, mode="a") as store:
        for entity, table in tables.items():
            layout = layouts[entity]
            if layout.nullable_boolean_columns:
                table = table.copy(deep=False)
                for column in layout.nullable_boolean_columns:
                    table[column] = table[column].astype("boolean")
            put_frame_table(
                store,
                entity,
                table,
                preferred_format=layout.format,
                data_columns=(
                    list(layout.data_columns)
                    if isinstance(layout.data_columns, tuple)
                    else layout.data_columns
                ),
            )


def _apply_steps(
    tables: Mapping[str, pd.DataFrame],
    steps: Sequence[PatchStep],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]], list[dict[str, object]]]:
    patched = dict(tables)
    added: list[dict[str, object]] = []
    step_receipts: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    names = [step.name for step in steps]
    if len(set(names)) != len(names):
        raise ValueError(f"Patch step names must be distinct; got {names}.")
    for step in steps:
        result = step.run(MappingProxyType(dict(patched)))
        records = []
        for patch in result.patches:
            key = (patch.entity, patch.column)
            if key in seen:
                raise ValueError(
                    f"{patch.entity}.{patch.column}: two patches name the column."
                )
            seen.add(key)
            if patch.entity not in patched:
                raise ValueError(
                    f"{step.name}: no {patch.entity!r} table for {patch.column}."
                )
            table = patched[patch.entity]
            values = patch.values
            if not isinstance(values, pd.Series) or not values.index.equals(
                table.index
            ):
                raise ValueError(
                    f"{step.name}: {patch.entity}.{patch.column} values must be "
                    "a Series on the table's index."
                )
            action = "added"
            if patch.column in table.columns:
                stored = table[patch.column]
                if not _same_values(stored, values):
                    raise ValueError(
                        f"{step.name}: {patch.entity}.{patch.column} is already "
                        "in the donor with other values (a nonzero, null or "
                        "different value); refusing to overwrite it."
                    )
                action = "kept_existing"
            else:
                table = table.copy(deep=False)
                table[patch.column] = values.rename(patch.column)
                patched[patch.entity] = table
            record = {
                **dict(patch.record),
                "column": patch.column,
                "entity": patch.entity,
                "dtype": str(patched[patch.entity][patch.column].dtype),
                "step": step.name,
                "action": action,
            }
            records.append(record)
            added.append(record)
        step_receipts.append(
            {
                **dict(result.receipt),
                "name": step.name,
                "columns": [f"{r['entity']}.{r['column']}" for r in records],
            }
        )
    return patched, added, step_receipts


def _same_values(stored: pd.Series, values: pd.Series) -> bool:
    """Whether a stored column already holds exactly the patch values."""

    if stored.isna().any() or values.isna().any():
        return False
    try:
        left = pd.to_numeric(stored, errors="raise").to_numpy(dtype=np.float64)
        right = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError):
        return bool(stored.astype(object).equals(values.astype(object)))
    return bool(np.array_equal(left, right))


def _verify_copy(
    path: Path,
    *,
    original: Mapping[str, pd.DataFrame],
    patched: Mapping[str, pd.DataFrame],
    formats: Mapping[str, str],
    keys: Sequence[str],
    others: Mapping[str, Any],
    root_attributes: Mapping[str, object],
) -> None:
    """Refuse a copy that lost or changed anything but the added columns."""

    with pd.HDFStore(path, mode="r") as store:
        if sorted(store.keys()) != list(keys):
            raise RuntimeError(
                f"Patched copy changed the table list: {sorted(store.keys())} "
                f"against {list(keys)}."
            )
        if _root_attributes(store) != dict(root_attributes):
            raise RuntimeError("Patched copy changed the HDF5 root attributes.")
        for key, expected in others.items():
            stored = store[key]
            if type(stored) is not type(expected) or not stored.equals(expected):
                raise RuntimeError(f"Patched copy changed {key!r}.")
        for entity, before in original.items():
            stored = read_frame_table(store, entity)
            after = patched[entity]
            if list(stored.columns) != list(after.columns):
                raise RuntimeError(
                    f"Patched copy has {entity!r} columns {list(stored.columns)}; "
                    f"expected {list(after.columns)}."
                )
            try:
                pd.testing.assert_frame_equal(
                    stored[list(before.columns)], before, check_exact=True
                )
                pd.testing.assert_frame_equal(stored, after, check_exact=True)
            except AssertionError as error:
                raise RuntimeError(
                    f"Patched copy changed {entity!r}: {error}"
                ) from error
            stored_format = "table" if store.get_storer(entity).is_table else "fixed"
            if stored_format != formats[entity]:
                raise RuntimeError(
                    f"Patched copy stored {entity!r} in {stored_format} format; "
                    f"the input used {formats[entity]}."
                )


def _root_attributes(store: pd.HDFStore) -> dict[str, object]:
    attributes = store.get_node("/")._v_attrs
    return {
        name: _attribute_value(attributes[name])
        for name in sorted(attributes._v_attrnames)
    }


def _attribute_value(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_paths(
    input_h5: Path, output_h5: Path, receipt_path: Path, *manifests: Path
) -> None:
    if not input_h5.is_file():
        raise ValueError(f"--input-h5 is not a file: {input_h5}.")
    if output_h5.suffix != ".h5":
        raise ValueError(f"--output-h5 must end with .h5, got {output_h5.name!r}.")
    resolved = {input_h5.resolve(), output_h5.resolve()}
    if len(resolved) != 2:
        raise ValueError("--output-h5 must differ from --input-h5.")
    if receipt_path.resolve() in resolved:
        raise ValueError("--receipt must differ from --input-h5 and --output-h5.")
    paths = [*resolved, receipt_path.resolve(), *(path.resolve() for path in manifests)]
    if len(set(paths)) != len(paths):
        raise ValueError(
            "--input-h5, --output-h5, --receipt, --input-release-manifest and "
            "--release-manifest must be different files."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def current_tool_identity(repository: Path = _REPOSITORY_ROOT) -> dict[str, object]:
    """The commit this tool runs from, and whether its file is edited."""

    def git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repository), *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--", TOOL)
    return {
        "commit": commit,
        "tool_dirty": None if status is None else bool(status),
    }


def _engine_identity() -> dict[str, object]:
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("policyengine-us")
    except PackageNotFoundError:
        installed = None
    return {"package": "policyengine-us", "version": installed}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a patched copy of a US national donor H5 for the ACS "
            "staging, for the County Files proof of concept only."
        )
    )
    parser.add_argument("--input-h5", required=True, type=Path)
    parser.add_argument("--output-h5", required=True, type=Path)
    parser.add_argument(
        "--receipt",
        type=Path,
        help="JSON receipt path (default: OUTPUT-H5 with a .receipt.json suffix).",
    )
    parser.add_argument(
        "--zero-column",
        action="append",
        choices=sorted(REVIEWED_ZERO_COLUMNS),
        help=(
            "Reviewed column to add as zeros on its engine entity; repeat for "
            "more than one (default: every reviewed column)."
        ),
    )
    parser.add_argument(
        "--input-release-manifest",
        type=Path,
        help=(
            "The published donor's release_manifest.json. The input's SHA-256 "
            "must match its one microdata artifact; the receipt records the "
            "donor build it names."
        ),
    )
    parser.add_argument(
        "--release-manifest",
        type=Path,
        help=(
            "Write a build-folder release manifest for the copy here, for R1's "
            "--donor-release-manifest. Needs --manifest-release and "
            "--manifest-build. Upload the copy, the receipt and this file (as "
            "release_manifest.json) to releases/<release>/<build>/ in "
            "--manifest-repo-id."
        ),
    )
    parser.add_argument(
        "--manifest-repo-id",
        default=DEFAULT_MANIFEST_REPO_ID,
        help=f"Repository the copy is uploaded to (default: {DEFAULT_MANIFEST_REPO_ID}).",
    )
    parser.add_argument("--manifest-release", help="Release folder name of the copy.")
    parser.add_argument("--manifest-build", help="Build folder name of the copy.")
    args = parser.parse_args(argv)
    names = (args.manifest_release, args.manifest_build)
    if args.release_manifest is None and any(name is not None for name in names):
        parser.error("--manifest-release and --manifest-build need --release-manifest.")
    if args.release_manifest is not None and any(name is None for name in names):
        parser.error(
            "--release-manifest needs --manifest-release and --manifest-build."
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    receipt_path = args.receipt or args.output_h5.with_suffix(".receipt.json")
    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    try:
        steps = build_patch_steps(
            zero_columns=args.zero_column or list(REVIEWED_ZERO_COLUMNS),
            engine=PolicyEngineUSEngine(),
        )
        release_manifest = (
            None
            if args.release_manifest is None
            else CopyReleaseManifest(
                path=args.release_manifest,
                release=args.manifest_release,
                build=args.manifest_build,
                repo_id=args.manifest_repo_id,
            )
        )
        receipt = patch_donor_h5(
            args.input_h5,
            args.output_h5,
            steps=steps,
            receipt_path=receipt_path,
            tool_identity=current_tool_identity(),
            engine_identity=_engine_identity(),
            input_release_manifest=args.input_release_manifest,
            release_manifest=release_manifest,
        )
    except ValueError as error:
        raise SystemExit(f"Donor patch refused: {error}") from error
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
