"""Build local, pinned annual US H5 candidates without publishing a release.

Each projection starts from the same base frame. Only one projected year is
held in memory at a time. A candidate projects either a national build
(``microcosm-us-2024-national``) or a local-area build
(``microcosm-us-2024-local``, the County Files base); the base build's release
picks the line. Run ``python -m microcosm.build.us_annual_static_aging --help``.

**National line.** The original H5 is copied byte for byte, and all later
files retain its entity/column/row layout in table format with an explicit
``_time_period``. Each projected year's H5 is ``microcosm_us_<year>`` (key and
file name); the base year, and the family, take the key of the base build's
dataset (``base_dataset``, ``microcosm_us_<base year>`` unless given), so the
candidate's names match the build it projects (:func:`annual_dataset_names`).

**Local-area line.** The base is a build folder of ``microcosm-us-2024-local``
(``base_build_dir``; its H5 may sit in ``base_artifact_root``), whose release
manifest pins the H5's SHA-256. Each projected year's H5 is
``microcosm_us_<year>_acs_local``, written in the base's own layout: the
fixed-format nullable tables of
:func:`microcosm.build.us_runtime.h5_io.write_nullable_us_h5`, the same
artifact-metadata row and root attributes, and ``_time_period`` set to the
year. The base H5 is never copied into the candidate; the candidate manifest
records it as the base build's artifact (``source: base_build``) and the
assembler carries it from the base build. Every column the year's uprating did
not factor (ids, memberships, every geography column such as ``state_fips``,
``county_fips``, ``county_fips_2023``, ``congressional_district_geoid``,
``puma`` and ``block_geoid``, and the person attributes: age, relationships,
work indicators, industry and occupation) is checked equal, values and dtype,
to the base before the year is written; a factor on a geography or protected
attribute column is refused.

**What the projection changes.** Static aging reweights households so that
each SSA age-by-sex cell grows from the base year by SSA's national projected
growth (``anchor="frame"``: the base-year calibration is kept and only growth
is applied), and multiplies the monetary columns that follow a national series
by that series' growth (:mod:`microcosm.calibrate.static_aging`). It imposes no
state or county target. State and county totals and shares therefore move only
through each household's age-sex composition, bounded by ``max_weight_ratio``
(no household's weight moves above 5 times its base weight); they are not
re-targeted, and county rows still sum to their state by construction because
each household keeps its one county and state. The local receipt for each
year records the drift (:func:`geography_share_drift`). Every projected year,
2026 and 2027 included, is a forecast from the 2024 base. This module adds no
county-level re-targeting.

A local frame mixes single-year ACS ages with CPS-coded donor ages (80 for
80-84, 85 for 85 and over), so aging bands every age the CPS way (80-84 to 80,
85 and over to 85) in a projection-only column of a slim aging frame; the
stored ``age`` column is unchanged.

**Memory.** For a base frame of S bytes in memory the national line holds at
most five full copies of the entity tables at once, inside
:func:`microcosm.frame.adapters.policyengine_us.multi_year_dataset` (the frame,
its export copy, the base-year dataset, the year's tables and the year's
dataset); the calibration result's frame copy is released before that call.
The local line holds the same five plus the slim aging frame and its
calibration copy (only ids, memberships, the age-sex cells and the factored
columns), then the frame, the year's frame and one entity table during the
write, and the frame plus the native reload during the loader check.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.calibrate import (
    SeriesProjection,
    ssa_population_projection,
    static_aging,
)
from microcosm.data.contract import US_LOCAL_RELEASE_ID, US_NATIONAL_RELEASE_ID
from microcosm.frame import US_SCHEMA, Frame, SignedScale, WeightKind, Weights
from microcosm.frame.adapters.policyengine_us import multi_year_dataset, uprating_series
from microcosm.frame.materialize import (
    engine_tables,
    materialize_nullable_booleans_for_pytables,
    put_frame_table,
    read_frame_table,
)

#: A dataset key a release manifest can name an artifact by.
_DATASET_KEY = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
#: One line of a checksum ledger, as ``sha256sum`` writes it.
_LEDGER_LINE = re.compile(r"(?P<digest>[0-9a-f]{64}) [ *](?P<name>.+)")

#: The two lines a candidate can project, named by their base release.
NATIONAL_LINE = "national"
LOCAL_LINE = "local_area"
BASE_RELEASE_LINES: Mapping[str, str] = MappingProxyType(
    {US_NATIONAL_RELEASE_ID: NATIONAL_LINE, US_LOCAL_RELEASE_ID: LOCAL_LINE}
)
#: The local line's H5 layout, recorded as the candidate manifest's ``layout``.
LOCAL_LAYOUT = "fixed_nullable"
#: The artifact-metadata row a local base and each projected local year carry.
LOCAL_ARTIFACT_KIND = "calibrated_local_area_artifact"
LOCAL_BASE_METADATA: Mapping[str, str] = MappingProxyType(
    {
        "artifact_kind": LOCAL_ARTIFACT_KIND,
        "entity_hdf_format": LOCAL_LAYOUT,
        "household_weight_kind": WeightKind.CALIBRATED.value,
    }
)
#: Household geography columns a projection must carry unchanged.
GEOGRAPHY_COLUMNS = frozenset(
    {
        "state_fips",
        "state_code",
        "county_fips",
        "county_fips_2023",
        "congressional_district_geoid",
        "puma",
        "puma_geoid",
        "block_geoid",
        "tract_geoid",
        "place_fips",
        "sldu",
        "sldl",
        "cbsa_code",
        "zip_code",
        "ST",
        "PUMA",
    }
)
_GEOGRAPHY_PREFIXES = ("congressional_district_geoid__",)
#: Person attributes no factor may touch, besides every ``is_*`` flag and every
#: column whose name holds ``relationship``, ``occupation`` or ``industry``.
_PROTECTED_ATTRIBUTES = frozenset({"age", "is_female", "worked_last_year"})
#: The columns whose weighted household shares a local receipt reports, the
#: county column being the first of the two the household table has.
_SHARE_STATE_COLUMN = "state_fips"
_SHARE_COUNTY_COLUMNS = ("county_fips_2023", "county_fips")
#: The SSA cells (CPS ASEC age coding) and the projection-only column that
#: holds each person's banded age on the local line.
_SSA_AGE_TOP = 85
_SSA_AGE_BANDS = MappingProxyType({80: 84})
_AGE_CELL_COLUMN = "_ssa_age_cell"
#: PyTables' own root attributes, which are not the artifact's.
_PYTABLES_ROOT_ATTRIBUTES = frozenset(
    {"CLASS", "PYTABLES_FORMAT_VERSION", "TITLE", "VERSION"}
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def projection_years(
    base_year: int, end_year: int, years: Iterable[int] | None = None
) -> tuple[int, ...]:
    """The projected years a candidate writes, in ascending order.

    Without ``years``, every year after ``base_year`` through ``end_year``.
    With them, exactly those: a non-empty set of distinct integer years, each
    after ``base_year`` and no later than ``end_year``.
    """
    if years is None:
        return tuple(range(base_year + 1, end_year + 1))
    chosen = list(years)
    if (
        not chosen
        or any(not isinstance(y, Integral) or isinstance(y, bool) for y in chosen)
        or len(set(chosen)) != len(chosen)
        or any(not base_year < y <= end_year for y in chosen)
    ):
        raise ValueError(
            "years must be distinct integer years after base_year "
            f"{base_year} and no later than end_year {end_year}; got {chosen!r}"
        )
    return tuple(sorted(int(y) for y in chosen))


def _projected_name(year: int, line: str) -> str:
    if line == NATIONAL_LINE:
        return f"microcosm_us_{year}"
    if line == LOCAL_LINE:
        return f"microcosm_us_{year}_acs_local"
    raise ValueError(f"line must be {NATIONAL_LINE!r} or {LOCAL_LINE!r}; got {line!r}")


def annual_dataset_names(
    base_year: int,
    end_year: int,
    *,
    base_dataset: str | None = None,
    line: str = NATIONAL_LINE,
    years: Iterable[int] | None = None,
) -> dict[int, str]:
    """The dataset key of the base year and each projected year's H5.

    Each H5 is written as ``<key>.h5``. On the national line a projected year
    is ``microcosm_us_<year>``; on the local-area line it is
    ``microcosm_us_<year>_acs_local``. The base year is ``base_dataset``, the
    key of the base build's dataset: by default ``microcosm_us_<base year>``
    (national) or ``microcosm_us_<base year>_acs_local`` (local), or
    ``populace_us_<base year>`` for a national build the switch moved. It must
    be a lowercase dataset key that no projected year has. The projected years
    are :func:`projection_years` of ``years``.
    """
    projected = projection_years(base_year, end_year, years)
    names = {base_year: _projected_name(base_year, line)}
    names.update({year: _projected_name(year, line) for year in projected})
    base_key = names[base_year] if base_dataset is None else base_dataset
    taken = {
        _projected_name(year, line)
        for year in (*range(base_year + 1, end_year + 1), *projected)
    }
    if (
        not isinstance(base_key, str)
        or not _DATASET_KEY.fullmatch(base_key)
        or base_key in taken
    ):
        raise ValueError(
            "base_dataset must be the base build's dataset key (lowercase "
            f"letters, digits and underscores), not a projected year's; got "
            f"{base_dataset!r}"
        )
    names[base_year] = base_key
    return names


def _json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _check_pin(path: Path, expected: str) -> None:
    if len(expected) != 64 or _sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch for {path}")


@dataclass(frozen=True)
class BaseBuild:
    """A base build read from its folder, its H5 checked against its manifest.

    ``build`` is the base build's ID (the candidate's ``parent_release``),
    ``dataset`` the H5's artifact key and ``file_name`` its bare file name.
    """

    release: str
    build: str
    line: str
    dataset: str
    file_name: str
    path: Path
    sha256: str


def _read_object(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{path} is not a JSON object")
    return document


def read_base_build(
    build_dir: str | Path, artifact_root: str | Path | None = None
) -> BaseBuild:
    """Read a base build folder and authenticate its H5 against its manifest.

    Both manifests must name the same release and build, the release being
    the national or the local-area release. The dataset is the national
    build's ``default_datasets.national`` artifact, or the local build's one
    microdata artifact (a local build claims no default). Its path is a bare
    file name, found in the build folder or else in ``artifact_root`` (where
    the local release tool's package stage writes it). Its bytes must have
    the SHA-256 the release manifest records and, when the folder has a
    ``sha256sums.txt`` ledger, the digest the ledger lists for it. Nothing is
    written.
    """
    folder = Path(build_dir).expanduser().absolute()
    release_manifest = _read_object(folder / "release_manifest.json")
    build_manifest = _read_object(folder / "build_manifest.json")
    release = release_manifest.get("release")
    build = release_manifest.get("build")
    if release not in BASE_RELEASE_LINES or not (isinstance(build, str) and build):
        raise ValueError(
            f"{folder} is not a build of {US_NATIONAL_RELEASE_ID} or "
            f"{US_LOCAL_RELEASE_ID}: its release manifest names release "
            f"{release!r} and build {build!r}"
        )
    if (build_manifest.get("release"), build_manifest.get("build")) != (
        release,
        build,
    ):
        raise ValueError(
            f"{folder}/build_manifest.json does not name release {release} and "
            f"build {build}"
        )
    line = BASE_RELEASE_LINES[release]
    artifacts = release_manifest.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    defaults = release_manifest.get("default_datasets")
    if line == NATIONAL_LINE:
        key = defaults.get("national") if isinstance(defaults, Mapping) else None
        keys = [key] if isinstance(key, str) else []
    else:
        if defaults != {}:
            raise ValueError(
                f"local build {build} must claim no default dataset "
                f"(default_datasets {{}}); it has {defaults!r}"
            )
        keys = [
            name
            for name, entry in artifacts.items()
            if isinstance(entry, Mapping) and entry.get("kind") == "microdata"
        ]
    entry = artifacts.get(keys[0]) if len(keys) == 1 else None
    if not isinstance(entry, Mapping) or entry.get("kind") != "microdata":
        raise ValueError(
            f"base build {build} must declare one microdata dataset; found {keys!r}"
        )
    dataset = keys[0]
    file_name = entry.get("path")
    sha256 = entry.get("sha256")
    if (
        not isinstance(file_name, str)
        or not file_name
        or file_name in {".", ".."}
        or set(file_name) & {"/", "\\"}
    ):
        raise ValueError(
            f"base build {build}'s H5 path must be a bare file name, not {file_name!r}"
        )
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        raise ValueError(f"base build {build}'s H5 has no sha256 in its manifest")
    path = folder / file_name
    if not path.is_file() and artifact_root is not None:
        path = Path(artifact_root).expanduser().absolute() / file_name
    if not path.is_file():
        raise ValueError(
            f"{file_name} is not in {folder}"
            + (
                f" or in {Path(artifact_root).expanduser().absolute()}"
                if artifact_root is not None
                else ""
            )
            + "; pass --base-artifact-root, the directory that holds the base "
            "build's H5"
        )
    actual = _sha256(path)
    if actual != sha256:
        raise ValueError(
            f"{path} has sha256 {actual}, not the {sha256} base build {build}'s "
            "release manifest records; the base build's manifest and H5 do not "
            "match"
        )
    ledger = folder / "sha256sums.txt"
    if ledger.is_file():
        listed = []
        for line_text in ledger.read_text().splitlines():
            match = _LEDGER_LINE.fullmatch(line_text)
            if match is not None and match["name"] == file_name:
                listed.append(match["digest"])
        if listed != [sha256]:
            raise ValueError(
                f"{ledger} must list {file_name} once with sha256 {sha256}; "
                f"it lists {listed}"
            )
    return BaseBuild(
        release=release,
        build=build,
        line=line,
        dataset=dataset,
        file_name=file_name,
        path=path,
        sha256=sha256,
    )


def _model_provenance(*, source: Path, commit: str, version: str) -> dict:
    import policyengine_us

    source = source.resolve()
    if (
        not Path(policyengine_us.__file__)
        .resolve()
        .is_relative_to(source / "policyengine_us")
    ):
        raise ValueError("Imported policyengine_us does not come from model_source")
    actual_version = importlib.metadata.version("policyengine-us")
    if actual_version != version:
        raise ValueError(
            f"Model version mismatch: expected {version}, got {actual_version}"
        )

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True
        ).strip()

    head = git("rev-parse", "HEAD")
    if len(commit) != 40 or head != commit:
        raise ValueError(f"Model commit mismatch: expected {commit}, got {head}")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Model source must be a clean, committed checkout")
    files = {
        name: _sha256(source / name)
        for name in git(
            "ls-files", "--", "policyengine_us", "pyproject.toml", "uv.lock"
        ).splitlines()
        if (source / name).is_file()
    }
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "version": actual_version,
        "commit": head,
        "source": str(source),
        "source_tree_sha256": digest,
        "source_files": files,
    }


def _runtime_provenance() -> dict:
    # Hash the installed implementation, including solver/Frame helpers, rather
    # than inferring a source identity from an editable package version.
    import microcosm.calibrate
    import microcosm.frame

    sources = {}
    for package in (microcosm.calibrate, microcosm.frame):
        root = Path(package.__file__).resolve().parent
        sources.update(
            {
                f"{package.__name__}/{p.relative_to(root)}": _sha256(p)
                for p in sorted(root.rglob("*.py"))
            }
        )
    sources["annual_static_aging.py"] = _sha256(Path(__file__))
    return {
        "source_sha256": sources,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "policyengine-us",
                "policyengine-core",
                "spm-calculator",
                "microdf-python",
                "numpy",
                "pandas",
                "scipy",
                "torch",
                "tables",
                "microcosm-frame",
                "microcosm-calibrate",
                "microcosm-build",
            )
        },
    }


def _local_writer_sha256() -> str:
    """The local line's writer module, which the runtime record also pins."""
    from microcosm.build.us_runtime import h5_io

    return _sha256(Path(h5_io.__file__))


def _period(store: pd.HDFStore, year: int) -> None:
    if "/_time_period" not in store.keys():
        raise ValueError("Annual H5 must have explicit _time_period metadata")
    if not store.get_storer("_time_period").is_table:
        raise ValueError("Annual H5 requires table-format _time_period metadata")
    values = store["_time_period"]
    if (
        len(values) != 1
        or not isinstance(values.iloc[0], Integral)
        or values.iloc[0] != year
    ):
        raise ValueError(f"Annual H5 _time_period must be exactly {year}")


def _verify(path: Path, tables: Mapping[str, pd.DataFrame], year: int) -> dict:
    from policyengine_us.data import USSingleYearDataset

    native = USSingleYearDataset(file_path=str(path))
    if str(native.time_period) != str(year):
        raise ValueError(f"Native loader did not retain year {year}")
    with pd.HDFStore(path, "r") as store:
        _period(store, year)
        for entity, expected in tables.items():
            if not store.get_storer(entity).is_table:
                raise ValueError(
                    f"Annual native layout requires table-format storage for {entity}; "
                    "fixed-format nullable inputs need a separately certified contract"
                )
            if set(expected.columns) - set(store.get_storer(entity).table.colnames):
                raise ValueError(
                    f"Annual native layout requires direct HDF fields for {entity}; "
                    "packed value blocks need a separately certified contract"
                )
            actual = read_frame_table(store, entity)
            pd.testing.assert_frame_equal(
                actual, expected, check_dtype=False, check_exact=True
            )
            external = materialize_nullable_booleans_for_pytables(expected).table
            pd.testing.assert_frame_equal(
                getattr(native, entity), external, check_dtype=False, check_exact=True
            )
    return {"logical_tables": True, "native_loader": True, "time_period": year}


def _write_year(path: Path, tables: Mapping[str, pd.DataFrame], year: int) -> dict:
    temporary = path.with_suffix(".tmp.h5")
    with pd.HDFStore(temporary, "w") as store:
        for entity, table in tables.items():
            put_frame_table(
                store, entity, table, preferred_format="table", data_columns=True
            )
        store.put("_time_period", pd.Series([year]), format="table")
    receipt = _verify(temporary, tables, year)
    temporary.replace(path)
    return receipt


# --- The local-area line ----------------------------------------------------


def is_geography_column(column: str) -> bool:
    """Whether ``column`` is a geography column a projection carries unchanged."""
    return column in GEOGRAPHY_COLUMNS or column.startswith(_GEOGRAPHY_PREFIXES)


def _is_protected(column: str) -> bool:
    return (
        is_geography_column(column)
        or column in _PROTECTED_ATTRIBUTES
        or column.startswith("is_")
        or "relationship" in column
        or "occupation" in column
        or "industry" in column
    )


def refuse_protected_factors(columns: Iterable[str], *, what: str) -> None:
    """Refuse a factor on a geography column or a protected person attribute.

    Static aging changes weights and monetary amounts only; a factor on a
    geography code, age, relationship, work indicator, an ``is_*`` flag, or an
    industry or occupation column would change a carried attribute.
    """
    protected = sorted(column for column in columns if _is_protected(column))
    if protected:
        raise ValueError(
            f"{what} would scale {protected}, which the local projection must "
            "carry unchanged (geography and person attributes)"
        )


@dataclass(frozen=True)
class LocalBase:
    """A local base H5: its frame, its stored column order, its root attributes."""

    frame: Frame
    columns: dict[str, list[str]]
    root_attributes: dict[str, str]


def _attribute_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.bytes_):
        return bytes(value).decode()
    if isinstance(value, str):
        return str(value)
    raise ValueError(f"root attribute value {value!r} is not text")


def local_root_attributes(store: pd.HDFStore) -> dict[str, str]:
    """The artifact's own root attributes (the block_v1 location attributes).

    PyTables' own attributes are left out; every other attribute must be text,
    as :func:`~microcosm.build.us_runtime.h5_io.write_nullable_us_h5` writes it.
    """
    attributes = store.get_node("/")._v_attrs
    return {
        name: _attribute_text(attributes[name])
        for name in sorted(attributes._v_attrnamesuser)
        if name not in _PYTABLES_ROOT_ATTRIBUTES
    }


def _stored_columns(store: pd.HDFStore, entity: str) -> list[str]:
    """A fixed-format table's column order, read without loading the table."""
    storer = store.get_storer(entity)
    if storer is None or storer.is_table:
        raise ValueError(
            f"a local base stores {entity} in the fixed nullable layout; this "
            "file does not"
        )
    return [_attribute_text(name) for name in store.get_node(f"/{entity}/axis0").read()]


def read_local_base(path: str | Path, base_year: int) -> LocalBase:
    """Read a local-area base H5 and check it is the local release's layout.

    The file must hold the six entity tables in fixed format, a table-format
    ``_time_period`` equal to ``base_year``, exactly the artifact-metadata row
    :data:`LOCAL_BASE_METADATA`, and no stored weight but ``household_weight``.
    """
    from microcosm.build.us_runtime.h5_io import (
        load_legacy_calibrated_us_h5,
        read_nullable_us_h5_metadata,
    )

    path = Path(path)
    metadata = read_nullable_us_h5_metadata(path)
    if metadata != dict(LOCAL_BASE_METADATA):
        raise ValueError(
            f"{path} is not a calibrated local-area artifact: its metadata row is "
            f"{metadata!r}, not {dict(LOCAL_BASE_METADATA)!r}"
        )
    with pd.HDFStore(path, "r") as store:
        _period(store, base_year)
        columns = {
            entity: _stored_columns(store, entity) for entity in US_SCHEMA.entities
        }
        root_attributes = local_root_attributes(store)
    stray = sorted(
        f"{entity}.{column}"
        for entity, names in columns.items()
        for column in names
        if column.endswith("_weight")
        and column.removesuffix("_weight") in US_SCHEMA.entities
        and (entity, column) != ("household", "household_weight")
    )
    if stray or "household_weight" not in columns["household"]:
        raise ValueError(
            f"{path} must store household_weight as its only weight; it stores "
            f"{stray or 'no household_weight'}"
        )
    frame = load_legacy_calibrated_us_h5(path)
    _require_positive_local_weights(frame.weights_for("household").values)
    return LocalBase(frame=frame, columns=columns, root_attributes=root_attributes)


def age_cells(age: pd.Series) -> np.ndarray:
    """Each person's SSA age cell: 80-84 pooled at 80, 85 and over at 85.

    Single-year ages and CPS-coded ages (80 meaning 80-84, 85 meaning 85 and
    over) land in the same cell. Ages must be whole, nonnegative numbers.
    """
    values = pd.to_numeric(age, errors="coerce").to_numpy(dtype=np.float64)
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.array_equal(values, np.floor(values))
    ):
        raise ValueError("age must be a whole nonnegative number for every person")
    cells = np.minimum(values.astype(np.int64), _SSA_AGE_TOP)
    for lower, upper in _SSA_AGE_BANDS.items():
        cells[(cells >= lower) & (cells <= upper)] = lower
    return cells


def aging_frame(frame: Frame, factored: Iterable[str]) -> Frame:
    """The slim frame static aging calibrates on the local line.

    It keeps each entity's id, the person memberships, ``is_female``, the
    banded age cell (:func:`age_cells`) and the columns that follow a series,
    with the base household weights. Static aging reads nothing else, so its
    weights and factors equal those of the full frame with banded ages.
    """
    schema = frame.schema
    wanted = set(factored)
    tables = {}
    for entity in frame.entities:
        table = frame.table(entity)
        keep = [schema.entity_id_column(entity)]
        if entity == schema.person_entity:
            keep += [schema.membership_column(group) for group in schema.group_entities]
            keep.append("is_female")
        keep += [c for c in table.columns if c in wanted and c not in keep]
        tables[entity] = table.loc[:, keep]
    tables[schema.person_entity][_AGE_CELL_COLUMN] = age_cells(frame.person["age"])
    return Frame(
        tables,
        schema,
        {
            "household": Weights(
                frame.weights_for("household").values, WeightKind.CALIBRATED
            )
        },
    )


def check_carried_columns(
    base: Frame,
    projected: Mapping[str, pd.DataFrame],
    *,
    factored: Iterable[str],
) -> None:
    """Refuse a projected year whose carried columns differ from the base's.

    Every entity keeps its rows and its columns in order (the household table
    adds ``household_weight``), and every column the year did not factor
    (ids, memberships, geography, person attributes) equals the base column
    exactly, values and dtype.
    """
    factored = set(factored)
    for entity in base.entities:
        expected = base.table(entity)
        actual = projected[entity]
        names = [c for c in actual.columns if c != f"{entity}_weight"]
        if names != list(expected.columns):
            raise ValueError(
                f"the projected {entity} table's columns differ from the base"
            )
        if len(actual) != len(expected):
            raise ValueError(f"the projected {entity} table has another row count")
        for column in expected.columns:
            if column in factored:
                continue
            try:
                pd.testing.assert_series_equal(
                    actual[column].reset_index(drop=True),
                    expected[column].reset_index(drop=True),
                    check_exact=True,
                )
            except AssertionError as error:
                raise ValueError(
                    f"the projected {entity}.{column} differs from the base: {error}"
                ) from error


def geography_share_drift(
    households: pd.DataFrame,
    base_weights: np.ndarray,
    projected_weights: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Each state's and county's weighted household share, base and projected.

    The county column is ``county_fips_2023`` when the household table has
    it, else ``county_fips``. Geography is carried unchanged, so only the
    weights differ. ``max_abs_share_change`` is the largest absolute change
    of one value's share.
    """
    base = np.asarray(base_weights, dtype=np.float64)
    year = np.asarray(projected_weights, dtype=np.float64)
    if (
        base.sum() <= 0
        or year.sum() <= 0
        or not len(base) == len(year) == len(households)
    ):
        raise ValueError("share drift needs one positive-mass weight per household")
    columns = [_SHARE_STATE_COLUMN] if _SHARE_STATE_COLUMN in households else []
    columns += [c for c in _SHARE_COUNTY_COLUMNS if c in households][:1]
    drift = {}
    for column in columns:
        codes = households[column].astype(str).to_numpy()
        shares = pd.DataFrame({"code": codes, "base": base, "projected": year})
        totals = shares.groupby("code", sort=True)[["base", "projected"]].sum()
        totals["base"] /= base.sum()
        totals["projected"] /= year.sum()
        change = (totals["projected"] - totals["base"]).abs()
        drift[column] = {
            "values": {
                str(code): {
                    "base_share": float(row.base),
                    "projected_share": float(row.projected),
                }
                for code, row in totals.iterrows()
            },
            "value_count": int(len(totals)),
            "max_abs_share_change": float(change.max()) if len(change) else 0.0,
        }
    return drift


def _require_positive_local_weights(weights: np.ndarray) -> None:
    if not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("local household weights must be finite and strictly positive")


def projected_local_frame(tables: Mapping[str, pd.DataFrame], base: Frame) -> Frame:
    """A projected year's tables as a frame with calibrated household weights.

    Refuses weights that are not finite and strictly positive on every row.
    """
    tables = {entity: tables[entity] for entity in base.entities}
    household = tables["household"]
    weights = household["household_weight"].to_numpy(dtype=np.float64)
    _require_positive_local_weights(weights)
    tables["household"] = household.drop(columns="household_weight")
    return Frame(
        tables, base.schema, {"household": Weights(weights, WeightKind.CALIBRATED)}
    )


def write_local_year(
    frame: Frame,
    path: Path,
    year: int,
    *,
    root_attributes: Mapping[str, str],
    columns: Mapping[str, list[str]],
) -> dict:
    """Write one projected local year in the base's layout and check it.

    :func:`~microcosm.build.us_runtime.h5_io.write_nullable_us_h5` round-trips
    every table, the period, the metadata row and the root attributes before
    the file is moved into place; this then checks the stored column order
    against the base's and the stored root attributes against the base's.
    """
    from microcosm.build.us_runtime.h5_io import (
        read_nullable_us_h5_metadata,
        write_nullable_us_h5,
    )

    _require_positive_local_weights(frame.weights_for("household").values)
    write_nullable_us_h5(
        frame,
        path,
        period=year,
        artifact_kind=LOCAL_ARTIFACT_KIND,
        root_attributes=dict(root_attributes),
    )
    if read_nullable_us_h5_metadata(path) != dict(LOCAL_BASE_METADATA):
        raise ValueError(f"{path} does not carry the local metadata row")
    with pd.HDFStore(path, "r") as store:
        _period(store, year)
        stored = {entity: _stored_columns(store, entity) for entity in columns}
        stored_attributes = local_root_attributes(store)
    if stored != {entity: list(names) for entity, names in columns.items()}:
        raise ValueError(f"{path} does not keep the base's columns in order")
    if stored_attributes != dict(root_attributes):
        raise ValueError(f"{path} does not keep the base's root attributes")
    return {
        "logical_tables": True,
        "time_period": year,
        "layout": LOCAL_LAYOUT,
        "metadata_row": True,
        "root_attributes": True,
        "carried_columns": True,
    }


def _verify_native_local(path: Path, year: int, rows: Mapping[str, int]) -> bool:
    """The engine's own loader opens the projected local year at its year."""
    from policyengine_us.data import USSingleYearDataset

    native = USSingleYearDataset(file_path=str(path))
    if str(native.time_period) != str(year):
        raise ValueError(f"Native loader did not retain year {year}")
    for entity, count in rows.items():
        if len(getattr(native, entity)) != count:
            raise ValueError(f"Native loader read another {entity} row count")
    return True


# --- The build ----------------------------------------------------------------


def _factor_record(factors: Mapping[str, Any]) -> dict:
    return {
        column: {"positive": factor.positive, "negative": factor.negative}
        if isinstance(factor, SignedScale)
        else float(factor)
        for column, factor in factors.items()
    }


def build_annual_static_aging(
    *,
    ssa_csv: str | Path,
    ssa_sha256: str,
    model_source: str | Path,
    model_commit: str,
    model_version: str,
    output_dir: str | Path,
    base_h5: str | Path | None = None,
    base_sha256: str | None = None,
    parent_release: str | None = None,
    base_year: int = 2024,
    end_year: int = 2035,
    epochs: int = 300,
    seed: int = 0,
    max_weight_ratio: float = 5.0,
    base_dataset: str | None = None,
    base_build_dir: str | Path | None = None,
    base_artifact_root: str | Path | None = None,
    years: Iterable[int] | None = None,
) -> dict:
    """Build a fresh local candidate, requiring explicit source and input pins.

    The base is either a build folder (``base_build_dir``, read by
    :func:`read_base_build`, which derives the base H5, its SHA-256, the base
    build's ID and its dataset key, and picks the line from the base build's
    release), or, for a national build only, the explicit ``base_h5``,
    ``base_sha256`` and ``parent_release``, which the caller attests;
    certification must independently verify that build's base SHA.
    ``base_dataset`` is the base build's dataset key, which the base year and
    the family keep (:func:`annual_dataset_names`). ``years`` picks the
    projected years (:func:`projection_years`). A complete candidate is not a
    certified release. Existing output directories are never reused,
    including failed attempts. A failure preserves its partial artifacts and
    status receipt.
    """
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output}")
    if (
        any(
            not isinstance(y, Integral) or isinstance(y, bool)
            for y in (base_year, end_year)
        )
        or end_year <= base_year
    ):
        raise ValueError("end_year must be an integer after integer base_year")
    explicit = (base_h5, base_sha256, parent_release, base_dataset)
    base_build: BaseBuild | None = None
    if base_build_dir is not None:
        if any(value is not None for value in explicit):
            raise ValueError(
                "a base build folder names its own H5, SHA-256, build and dataset; "
                "do not also pass base_h5, base_sha256, parent_release or "
                "base_dataset"
            )
        base_build = read_base_build(base_build_dir, base_artifact_root)
        base_h5, base_sha256 = base_build.path, base_build.sha256
        parent_release, base_dataset = base_build.build, base_build.dataset
        line = base_build.line
    else:
        if base_artifact_root is not None:
            raise ValueError("base_artifact_root needs base_build_dir")
        if base_h5 is None or base_sha256 is None or parent_release is None:
            raise ValueError(
                "name the base with base_build_dir, or (a national build only) "
                "with base_h5, base_sha256 and parent_release"
            )
        if isinstance(base_dataset, str) and base_dataset.endswith("_acs_local"):
            raise ValueError(
                "a local-area base needs its build folder (base_build_dir), so "
                "its H5 is checked against the build's manifest"
            )
        line = NATIONAL_LINE
    if not parent_release.strip():
        raise ValueError("parent_release must identify the pinned base release")
    projected = projection_years(base_year, end_year, years)
    names = annual_dataset_names(
        base_year, end_year, base_dataset=base_dataset, line=line, years=projected
    )
    local = line == LOCAL_LINE
    # Hugging Face snapshot paths are .h5 symlinks to extensionless blobs.
    # Keep the logical suffix required by the native loader; hashing and reads
    # still follow the link to authenticate the actual bytes.
    base = Path(base_h5).expanduser().absolute()
    ssa = Path(ssa_csv).expanduser().absolute()
    _check_pin(base, base_sha256)
    _check_pin(ssa, ssa_sha256)
    model_kwargs = {
        "source": Path(model_source),
        "commit": model_commit,
        "version": model_version,
    }
    model = _model_provenance(**model_kwargs)
    base_runtime = _runtime_provenance()
    runtime = base_runtime
    if local:
        writer_sha256 = _local_writer_sha256()
        runtime = {
            **base_runtime,
            "source_sha256": {
                **base_runtime["source_sha256"],
                "us_runtime/h5_io.py": writer_sha256,
            },
        }
        local_base = read_local_base(base, base_year)
        frame = local_base.frame
        columns = local_base.columns
    else:
        local_base = None
        with pd.HDFStore(base, "r") as store:
            _period(store, base_year)
            tables = {
                entity: read_frame_table(store, entity) for entity in US_SCHEMA.entities
            }
        columns = {entity: list(table.columns) for entity, table in tables.items()}
        weights = tables["household"].pop("household_weight").to_numpy(dtype=float)
        if any(
            f"{entity}_weight" in table
            for entity in US_SCHEMA.entities
            for table in tables.values()
        ):
            raise ValueError("Base must store only household weights")
        frame = Frame(
            tables, US_SCHEMA, {"household": Weights(weights, WeightKind.CALIBRATED)}
        )
        base_tables = {
            entity: table.loc[:, columns[entity]]
            for entity, table in engine_tables(
                frame, weighted_entities=("household",)
            ).items()
        }
        _verify(base, base_tables, base_year)
        del tables, base_tables
    age_column = _AGE_CELL_COLUMN if local else "age"
    demographics = ssa_population_projection(
        ssa,
        age_top=_SSA_AGE_TOP,
        age_bands=dict(_SSA_AGE_BANDS),
        age_column=age_column,
    )
    for year in range(base_year, end_year + 1):
        demographics.for_year(year)
    cells = list(demographics.cells)
    known_cells = pd.MultiIndex.from_frame(demographics.for_year(base_year)[cells])
    if local:
        person_cells = pd.DataFrame(
            {
                _AGE_CELL_COLUMN: age_cells(frame.person["age"]),
                "is_female": frame.person["is_female"].to_numpy(),
            }
        )
    else:
        person_cells = frame.person[cells]
    observed_cells = pd.MultiIndex.from_frame(person_cells.drop_duplicates())
    if not observed_cells.isin(known_cells).all():
        raise ValueError("Base demographic cells do not match SSA age/sex coding")
    del person_cells

    from policyengine_us import CountryTaxBenefitSystem

    system = CountryTaxBenefitSystem()
    totals, indices, mapping = uprating_series(
        [column for entity in frame.entities for column in frame.table(entity)],
        tuple(range(base_year, end_year + 1)),
        system=system,
    )
    if local:
        refuse_protected_factors(mapping, what="the uprating map")
        aged_frame = aging_frame(frame, mapping)
    else:
        aged_frame = frame
    family = names[base_year]
    settings = {
        "base_year": base_year,
        "end_year": end_year,
        "age_top": _SSA_AGE_TOP,
        "age_bands": {str(lower): upper for lower, upper in _SSA_AGE_BANDS.items()},
        "epochs": epochs,
        "seed": seed,
        "max_weight_ratio": max_weight_ratio,
        "anchor": "frame",
        "learning_rate": 0.02,
        "l2_lambda": 0.0,
    }
    if years is not None or local:
        settings["years"] = list(projected)
    if local:
        settings["age_cells"] = (
            "age banded at aging time, 80-84 to 80 and 85 and over to 85; the "
            "stored age column is unchanged"
        )
    base_record = {
        "dataset": family,
        "year": base_year,
        "parent_release": parent_release,
        "path": str(base),
        "sha256": base_sha256,
    }
    if local:
        base_record["release"] = base_build.release
    rows = {entity: frame.n(entity) for entity in frame.entities}
    manifest = {
        "schema_version": 1,
        "kind": "us_annual_static_aging_candidate",
        "status": "complete",
        "base": base_record,
        "inputs": {"ssa": {"path": str(ssa), "sha256": ssa_sha256}},
        "model": model,
        "runtime": runtime,
        "settings": settings,
        "metadata": {
            "dataset_years": {
                family: {str(year): names[year] for year in sorted(names)}
            }
        },
        "artifacts": {},
    }
    if local:
        manifest["layout"] = LOCAL_LAYOUT
        manifest["artifacts"][family] = {
            "path": base_build.file_name,
            "sha256": base_sha256,
            "year": base_year,
            "rows": rows,
            "columns": columns,
            "source": "base_build",
        }
    written = list(projected) if local else [base_year, *projected]
    completed: list[int] = []
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "build_status.json", {"status": "running", "completed_years": []})
    try:
        for year in written:
            name = names[year]
            path = output / f"{name}.h5"
            if year == base_year:
                shutil.copyfile(base, path)
                _check_pin(path, base_sha256)
                round_trip = {
                    "logical_tables": True,
                    "native_loader": True,
                    "time_period": year,
                }
                projection_receipt = None
            else:
                result = static_aging(
                    aged_frame,
                    base_year=base_year,
                    years=(year,),
                    demographics=demographics,
                    series=SeriesProjection(totals=totals, indices=indices),
                    column_series=mapping,
                    epochs=epochs,
                    seed=seed,
                    max_weight_ratio=max_weight_ratio,
                    anchor="frame",
                    learning_rate=0.02,
                    l2_lambda=0.0,
                )
                projection = result.year(year)
                year_weights = projection.weights.values
                factors = dict(projection.factors)
                receipt = {
                    "year": year,
                    "base_year": base_year,
                    "column_series": mapping,
                    "totals": totals,
                    "indices": indices,
                    "factors": _factor_record(factors),
                    "demographic_fit": projection.demographic_fit.to_dict(
                        orient="records"
                    ),
                    "fraction_within_10pct": float(projection.fraction_within_10pct),
                }
                # The calibration result holds a full copy of the frame it
                # calibrated; release it before the export copies are made.
                del projection, result
                if local:
                    refuse_protected_factors(factors, what=f"the {year} factors")
                dataset = multi_year_dataset(
                    frame, base_year, {year: (year_weights, factors)}
                ).datasets[year]
                if local:
                    projected_tables = {
                        entity: getattr(dataset, entity) for entity in frame.entities
                    }
                    del dataset
                    check_carried_columns(frame, projected_tables, factored=factors)
                    year_frame = projected_local_frame(projected_tables, frame)
                    del projected_tables
                    receipt["geography_shares"] = geography_share_drift(
                        frame.table("household"),
                        frame.weights_for("household").values,
                        year_frame.weights_for("household").values,
                    )
                    round_trip = write_local_year(
                        year_frame,
                        path,
                        year,
                        root_attributes=local_base.root_attributes,
                        columns=columns,
                    )
                    del year_frame
                    round_trip["native_loader"] = _verify_native_local(path, year, rows)
                else:
                    projected_tables = {
                        entity: getattr(dataset, entity).loc[:, columns[entity]]
                        for entity in frame.entities
                    }
                    del dataset
                    round_trip = _write_year(path, projected_tables, year)
                    del projected_tables
                receipt_path = output / f"projection_{year}.json"
                _json(receipt_path, receipt)
                projection_receipt = {
                    "path": receipt_path.name,
                    "sha256": _sha256(receipt_path),
                }
                del receipt
            artifact = {
                "path": path.name,
                "sha256": _sha256(path),
                "year": year,
                "rows": rows,
                "columns": columns,
                "round_trip": round_trip,
            }
            if projection_receipt:
                artifact["projection_receipt"] = projection_receipt
            manifest["artifacts"][name] = artifact
            completed.append(year)
            _json(
                output / "build_status.json",
                {"status": "running", "completed_years": list(completed)},
            )
        _check_pin(base, base_sha256)
        _check_pin(ssa, ssa_sha256)
        if (
            _model_provenance(**model_kwargs) != model
            or _runtime_provenance() != base_runtime
            or (local and _local_writer_sha256() != writer_sha256)
        ):
            raise ValueError("Implementation or model source changed during the build")
        _json(
            output / "build_status.json",
            {"status": "complete", "completed_years": list(completed)},
        )
        # The manifest is the completion marker and is written last.
        _json(output / "annual_manifest.json", manifest)
    except BaseException as exc:
        _json(
            output / "build_status.json",
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "completed_years": list(completed),
            },
        )
        raise
    return manifest


def command_parser() -> argparse.ArgumentParser:
    """The command's arguments, which name :func:`build_annual_static_aging`'s."""
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in (
        "ssa-csv",
        "ssa-sha256",
        "model-source",
        "model-commit",
        "model-version",
        "output-dir",
    ):
        parser.add_argument(f"--{argument}", required=True)
    parser.add_argument(
        "--base-build-dir",
        help=(
            "The base build's folder (release_manifest.json, build_manifest.json "
            "and, if present, sha256sums.txt), from which the base H5, its "
            "SHA-256, the build and the dataset key are read; required for a "
            "build of microcosm-us-2024-local."
        ),
    )
    parser.add_argument(
        "--base-artifact-root",
        help=(
            "The directory holding the base build's H5 when it is not in "
            "--base-build-dir (the local release tool's package stage writes "
            "it beside releases/)."
        ),
    )
    parser.add_argument("--base-h5", help="National only, without --base-build-dir.")
    parser.add_argument(
        "--base-sha256", help="National only, without --base-build-dir."
    )
    parser.add_argument(
        "--parent-release", help="National only, without --base-build-dir."
    )
    parser.add_argument("--base-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2035)
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        help=(
            "Write exactly these projected years (after --base-year, no later "
            "than --end-year); default every year after --base-year through "
            "--end-year."
        ),
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-weight-ratio", type=float, default=5.0)
    parser.add_argument(
        "--base-dataset",
        help=(
            "The base build's dataset key, which the base year and the family "
            "keep (default microcosm_us_<base year>; populace_us_2024 for a "
            "build the switch moved). Without --base-build-dir only."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    build_annual_static_aging(**vars(command_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
