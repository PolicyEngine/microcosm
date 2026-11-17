"""Local-area annual builds (microcosm#922): fixtures with real fixed-format H5s.

A build of ``microcosm-us-2024-local`` is written as the ACS release tool's
``package`` stage writes it: its release directory under
``<root>/releases/microcosm-us-2024-local/`` and its dataset,
``microcosm_us_2024_acs_local.h5``, one level above in ``<root>``, the
artifact root. Its H5 and every projected year's H5 are pandas fixed-format
files with the six US entity tables, an explicit ``_time_period`` and the
staging metadata row, as ``write_nullable_us_h5`` writes them, so the
contract's h5py reader meets the real layout.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.data.build_folder_manifests import rewrite_release_dir
from microcosm.data.contract import (
    NON_DEFAULT_LOCAL_AREA_DATASET_ROLE,
    US_LOCAL_ANNUAL_RELEASE_ID,
    US_LOCAL_AREA_SPM_MEASUREMENT_STAGE,
    US_LOCAL_AREA_STORED_INPUTS_STAGE,
    US_LOCAL_RELEASE_ID,
    US_SPM_MEASUREMENT_EVIDENCE_KEY,
    US_STORED_INPUTS_EVIDENCE_KEY,
)
from microcosm.data.stored_inputs import register_sha256

#: The local-area base build every fixture projects.
LOCAL_BUILD = "populace-us-2024-buildo-acs-local-abc1234-20260723T000000Z"
#: The dataset a new local-area build declares, by its key.
LOCAL_DATASET_KEY = "microcosm_us_2024_acs_local"
LOCAL_REPO = "policyengine/populace-us"
BUILT_WITH_MODEL = {"name": "policyengine-us", "version": "2.2.1"}
BUILT_WITH_CORE = {"name": "policyengine-core", "version": "3.33.3"}
#: A local-annual build of LOCAL_BUILD.
LOCAL_ANNUAL_BUILD = f"{LOCAL_BUILD}-annual-20261114T000000Z-0123abcd"
ENTITIES = ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")
ACCEPTANCE_CHECKS = (
    "schema",
    "source_identity",
    "year",
    "demographics",
    "input_aggregates",
    "runtime",
)
_METADATA_KEY = "_populace_staging_metadata"
_METADATA = {
    "artifact_kind": "calibrated_local_area_artifact",
    "entity_hdf_format": "fixed_nullable",
    "household_weight_kind": "calibrated",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: Path, document: object) -> None:
    Path(path).write_text(json.dumps(document, indent=2) + "\n")


def projected_key(year: int) -> str:
    """A projected year's artifact key; its file is ``<key>.h5``."""
    return f"microcosm_us_{year}_acs_local"


def local_releases_record(base_build: str | None = "listed") -> dict | None:
    """The repository's ``releases.json``, recording LOCAL_BUILD as
    ``base_build`` says (``"listed"``, ``"withdrawn"`` or ``"unlisted"``)
    beside the local release's current build; ``None`` for no record."""
    if base_build is None:
        return None
    builds = {
        "an-earlier-local-build": {
            "published_at": "2026-11-01T00:00:00+00:00",
            "withdrawn": False,
        }
    }
    if base_build != "unlisted":
        builds[LOCAL_BUILD] = {
            "published_at": "2026-11-02T00:00:00+00:00",
            "withdrawn": base_build == "withdrawn",
        }
    return {
        "schema_version": 1,
        "default": None,
        "releases": {
            US_LOCAL_RELEASE_ID: {
                "role": "local_area",
                "current": "an-earlier-local-build",
                "builds": builds,
            }
        },
    }


def local_tables(
    households: int = 3,
    *,
    weight_scale: float = 1.0,
    weights: Iterable[float] | None = None,
) -> dict[str, pd.DataFrame]:
    """The six entity tables of a small local-area population.

    Every household has two people, who share its tax unit, SPM unit, family
    and marital unit; IDs are not row positions, so a reordering is visible.
    The household table carries county geography as strings, so the file has
    an object block that h5py cannot read beside its numeric blocks.
    """
    household_ids = np.arange(households, dtype=np.int64) * 10 + 101
    person_ids = np.arange(2 * households, dtype=np.int64) * 7 + 1001
    members = np.repeat(household_ids, 2)
    weight_values = (
        np.asarray(list(weights), dtype=np.float64)
        if weights is not None
        else (np.arange(households, dtype=np.float64) + 1.0) * 10.0 * weight_scale
    )
    county = np.array([f"06{index:03d}" for index in range(households)], dtype=object)
    person = pd.DataFrame(
        {
            "person_id": person_ids,
            "person_household_id": members,
            "person_tax_unit_id": members,
            "person_spm_unit_id": members,
            "person_family_id": members,
            "person_marital_unit_id": members,
            "age": np.tile(np.array([40, 12], dtype=np.int64), households),
            "employment_income": np.tile(
                np.array([50_000.0, 0.0]) * weight_scale, households
            ),
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": household_ids,
                "household_weight": weight_values,
                "state_fips": np.full(households, 6, dtype=np.int64),
                "county_fips": county,
                "county_fips_2023": county.copy(),
            }
        ),
    }
    for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
        tables[entity] = pd.DataFrame({f"{entity}_id": household_ids})
    return {entity: tables[entity] for entity in ENTITIES}


def write_local_h5(
    path: Path,
    year: int,
    tables: dict[str, pd.DataFrame] | None = None,
    **table_options,
) -> Path:
    """A pandas fixed-format H5 in the local-area layout for ``year``."""
    tables = local_tables(**table_options) if tables is None else tables
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        with pd.HDFStore(path, mode="w") as store:
            for entity in ENTITIES:
                store.put(entity, tables[entity], format="fixed")
            store.put("_time_period", pd.Series([year]), format="table")
            store.put(
                _METADATA_KEY,
                pd.Series([json.dumps(_METADATA, sort_keys=True)]),
                format="table",
            )
    return Path(path)


def _stored_inputs_gate(artifact_sha256: str) -> dict:
    return {
        "passed": True,
        "failures": [],
        "engine": f"policyengine-us {BUILT_WITH_MODEL['version']}",
        "register_sha256": register_sha256(),
        "registered_non_variables": ["household_spine", "person_spine"],
        "artifact_sha256": artifact_sha256,
        "checked_at_stage": US_LOCAL_AREA_STORED_INPUTS_STAGE,
    }


def _spm_measurement_gate(artifact_sha256: str) -> dict:
    return {
        "passed": True,
        "failures": [],
        "engine": f"policyengine-us {BUILT_WITH_MODEL['version']}",
        "details": {
            "role_column_present": True,
            "role_null_values": 0,
            "role_non_boolean_values": 0,
            "engine_scopes_spm_universe": False,
            "universe_declared_in_release": False,
            "measured_scope": "all_units",
            "n_units": 3,
            "n_units_by_universe": {"INCLUDED": 3, "OUTSIDE": 0},
            "n_measured_units": 3,
            "n_measured_units_without_classified_adult": 0,
            "n_units_without_classified_adult": 0,
            "n_units_without_classified_adult_by_universe": {
                "INCLUDED": 0,
                "OUTSIDE": 0,
            },
        },
        "artifact_sha256": artifact_sha256,
        "checked_at_stage": US_LOCAL_AREA_SPM_MEASUREMENT_STAGE,
    }


def local_build_folder(root: Path, *, households: int = 3) -> Path:
    """A build of ``microcosm-us-2024-local`` as the package stage writes it.

    Returns ``<root>/releases/microcosm-us-2024-local/``; the dataset is
    ``<root>/microcosm_us_2024_acs_local.h5``. Its stored-input and SPM gates
    certify that H5, and ``sha256sums.txt`` covers every file, the H5 by name.
    """
    release_dir = root / "releases" / US_LOCAL_RELEASE_ID
    release_dir.mkdir(parents=True)
    dataset = write_local_h5(
        root / f"{LOCAL_DATASET_KEY}.h5", 2024, households=households
    )
    dataset_sha = sha256(dataset)
    gates = {
        "us_puma_ladder_gate": {"passed": True, "failures": []},
        "calibration": {"passed": True, "final_loss": 0.05},
        US_STORED_INPUTS_EVIDENCE_KEY: _stored_inputs_gate(dataset_sha),
        US_SPM_MEASUREMENT_EVIDENCE_KEY: _spm_measurement_gate(dataset_sha),
    }
    files = {
        "calibration_diagnostics.json": {
            "households": households,
            "n_targets": 1,
            "final_loss": 0.05,
            "fraction_within_10pct": 1.0,
            "targets": [
                {
                    "name": "pop_state_06",
                    "target": 100.0,
                    "compiled_target": 100.0,
                    "initial_estimate": 90.0,
                    "final_estimate": 99.0,
                }
            ],
        },
        "gate_summary.json": {"gates": gates, "reviewed_limitations": []},
        "us_source_coverage.json": {
            "schema_version": 1,
            "acs_sources": {"manifest": "pinned"},
            "geography_ladder": {"sha256": "f" * 64},
            "transfer_coverage": {"targets": 91},
            "donor_release": {
                "repo_id": LOCAL_REPO,
                "release_id": "populace-us-2024-buildo-sparse-x",
                "revision": "populace-us-2024-buildo-sparse-x",
            },
        },
        "county_reliability.json": {"counties": {"06001": {"grade": "A"}}},
        "build_manifest.json": {
            "build_id": LOCAL_BUILD,
            "build_sha": "abc1234",
            "gates": {
                key: copy.deepcopy(gates[key])
                for key in (
                    US_STORED_INPUTS_EVIDENCE_KEY,
                    US_SPM_MEASUREMENT_EVIDENCE_KEY,
                )
            },
        },
    }
    for name, document in files.items():
        write_json(release_dir / name, document)

    def declared(kind: str, path: Path) -> dict:
        return {
            "kind": kind,
            "path": path.name,
            "repo_id": LOCAL_REPO,
            "revision": LOCAL_BUILD,
            "sha256": sha256(path),
        }

    manifest = {
        "schema_version": 1,
        "data_package": {"name": "microcosm-data", "version": "0.1.0"},
        "default_datasets": {},
        "dataset_role": NON_DEFAULT_LOCAL_AREA_DATASET_ROLE,
        "is_default": False,
        "namespace": "buildo_acs_local",
        "build": {
            "build_id": LOCAL_BUILD,
            "built_at": "2026-07-23T00:00:00+00:00",
            "built_with_core_package": dict(BUILT_WITH_CORE),
            "built_with_model_package": dict(BUILT_WITH_MODEL),
        },
        "artifacts": {
            LOCAL_DATASET_KEY: declared("microdata", dataset),
            **{
                Path(name).stem: declared("diagnostics", release_dir / name)
                for name in (
                    "gate_summary.json",
                    "us_source_coverage.json",
                    "calibration_diagnostics.json",
                    "county_reliability.json",
                )
            },
        },
        "reviewed_limitations": [{"id": "cd_population_marginal_vintage_2020"}],
    }
    write_json(release_dir / "release_manifest.json", manifest)
    ledger = {path.name: sha256(path) for path in sorted(release_dir.iterdir())} | {
        dataset.name: dataset_sha
    }
    (release_dir / "sha256sums.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(ledger.items()))
    )
    return rewrite_release_dir(release_dir, build=LOCAL_BUILD)


def reseal(release_dir: Path, root: Path) -> None:
    """Give every release file's manifest entry and ledger line its digest.

    After a test edits a file in a local-area folder, the manifest entry that
    pins it and the ledger line that lists it no longer match; this rewrites
    both. The release manifest is not an artifact, so its ledger line is
    written last.
    """
    manifest_path = release_dir / "release_manifest.json"
    manifest = read_json(manifest_path)
    for entry in manifest["artifacts"].values():
        local = release_dir / entry["path"]
        if local.is_file():
            entry["sha256"] = sha256(local)
    write_json(manifest_path, manifest)
    names = {
        path.name for path in release_dir.iterdir() if path.name != "sha256sums.txt"
    } | {entry["path"] for entry in manifest["artifacts"].values()}
    lines = []
    for name in sorted(names):
        local = release_dir / name if (release_dir / name).is_file() else root / name
        lines.append(f"{sha256(local)}  {name}")
    (release_dir / "sha256sums.txt").write_text("\n".join(lines) + "\n")


def rewrite_json(
    release_dir: Path, root: Path, name: str, mutate: Callable[[dict], None]
) -> None:
    """Edit one JSON file of a local-area folder, then :func:`reseal` it."""
    document = read_json(release_dir / name)
    mutate(document)
    write_json(release_dir / name, document)
    reseal(release_dir, root)


def candidate_manifest(
    base_manifest: dict,
    base_h5: Path,
    projected: dict[int, tuple[Path, Path]],
) -> dict:
    """The candidate manifest the builder writes for a local base (the spec in
    docs/us-local-annual-projection.md): the base-year H5 recorded, not copied,
    and one artifact per projected year with its receipt."""
    base_key = LOCAL_DATASET_KEY
    base = base_manifest["artifacts"][base_key]
    with pd.HDFStore(base_h5, mode="r") as store:
        columns = {
            entity: [str(c) for c in store[entity].columns] for entity in ENTITIES
        }
        rows = {entity: len(store[entity]) for entity in ENTITIES}
    runtime = {
        "policyengine-us": base_manifest["built_with_model_package"]["version"],
        "policyengine-core": base_manifest["built_with_core_package"]["version"],
    }
    years = sorted(projected)
    return {
        "schema_version": 1,
        "kind": "us_annual_static_aging_candidate",
        "status": "complete",
        "layout": "fixed_nullable",
        "base": {
            "dataset": base_key,
            "year": 2024,
            "parent_release": base_manifest["build"],
            "release": US_LOCAL_RELEASE_ID,
            "path": str(base_h5),
            "sha256": base["sha256"],
        },
        "model": {
            "version": runtime["policyengine-us"],
            "commit": "a" * 40,
            "source_tree_sha256": "b" * 64,
        },
        "runtime": {"versions": runtime},
        "settings": {"base_year": 2024, "years": years},
        "metadata": {
            "dataset_years": {
                base_key: {
                    "2024": base_key,
                    **{str(year): projected_key(year) for year in years},
                }
            }
        },
        "artifacts": {
            base_key: {
                "path": base["path"],
                "sha256": base["sha256"],
                "year": 2024,
                "rows": rows,
                "columns": columns,
                "source": "base_build",
            },
            **{
                projected_key(year): {
                    "path": h5.name,
                    "sha256": sha256(h5),
                    "year": year,
                    "rows": rows,
                    "columns": columns,
                    "round_trip": {"logical_tables": True, "time_period": year},
                    "projection_receipt": {
                        "path": receipt.name,
                        "sha256": sha256(receipt),
                    },
                }
                for year, (h5, receipt) in sorted(projected.items())
            },
        },
    }


def acceptance_report(evidence_path: Path) -> dict:
    """A passing acceptance report bound to the candidate manifest's bytes,
    covering every year of its family, the base year included."""
    evidence = read_json(evidence_path)
    (family,) = evidence["metadata"]["dataset_years"].values()
    return {
        "schema_version": 1,
        "kind": "us_annual_projection_acceptance",
        "status": "passed",
        "projection_manifest_sha256": sha256(evidence_path),
        "model": evidence["model"],
        "runtime": evidence["runtime"]["versions"],
        "years": {
            year: {
                "dataset": key,
                "sha256": evidence["artifacts"][key]["sha256"],
                "checks": dict.fromkeys(ACCEPTANCE_CHECKS, "passed"),
            }
            for year, key in family.items()
        },
    }


def write_local_candidate(
    local: Path,
    root: Path,
    directory: Path,
    *,
    years: Iterable[int] = (2026, 2027),
    projected_tables: Callable[[int], dict[str, pd.DataFrame]] | None = None,
) -> Path:
    """The directory the builder writes for ``local``: one fixed-format H5
    and one receipt per projected year, the candidate manifest and the
    build status, but no copy of the base H5."""
    directory.mkdir(parents=True)
    manifest = read_json(local / "release_manifest.json")
    base_h5 = root / manifest["artifacts"][LOCAL_DATASET_KEY]["path"]
    with pd.HDFStore(base_h5, mode="r") as store:
        households = len(store["household"])
    projected = {}
    for year in years:
        tables = (
            projected_tables(year)
            if projected_tables is not None
            else local_tables(households, weight_scale=1.0 + (year - 2024) / 100)
        )
        h5 = write_local_h5(directory / f"{projected_key(year)}.h5", year, tables)
        receipt = directory / f"projection_{year}.json"
        write_json(receipt, {"base_year": 2024, "year": year, "factors": {}})
        projected[year] = (h5, receipt)
    write_json(
        directory / "annual_manifest.json",
        candidate_manifest(manifest, base_h5, projected),
    )
    write_json(
        directory / "build_status.json",
        {"status": "complete", "completed_years": sorted(projected)},
    )
    return directory


def local_annual_build_folder_of(
    local: Path,
    root: Path,
    *,
    years: Iterable[int] = (2026, 2027),
    build: str = LOCAL_ANNUAL_BUILD,
    projected_tables: Callable[[int], dict[str, pd.DataFrame]] | None = None,
) -> Path:
    """A build of ``microcosm-us-2024-local-annual`` projecting ``local``.

    It carries every file of the local build but its two manifests, with the
    base H5 and one H5 per projected year in ``root``, the candidate
    manifest, its passing acceptance report and each year's receipt; both
    manifests name the local-annual release and ``build``, the release
    manifest names the local build in ``metadata.annual_projection_base``,
    and the ledger covers every file.
    """
    annual = local.with_name(US_LOCAL_ANNUAL_RELEASE_ID)
    annual.mkdir()
    for path in local.iterdir():
        if path.name not in {"release_manifest.json", "build_manifest.json"}:
            shutil.copyfile(path, annual / path.name)
    candidate = write_local_candidate(
        local,
        root,
        root / "candidate",
        years=years,
        projected_tables=projected_tables,
    )
    evidence_path = annual / "annual_manifest.json"
    shutil.copyfile(candidate / "annual_manifest.json", evidence_path)
    evidence = read_json(evidence_path)
    acceptance_path = annual / "annual_acceptance.json"
    write_json(acceptance_path, acceptance_report(evidence_path))
    manifest = read_json(local / "release_manifest.json")
    base = manifest["artifacts"][LOCAL_DATASET_KEY]

    def declared(key: str, kind: str, path: Path) -> dict:
        return {
            "kind": kind,
            "path": path.name,
            "repo_id": LOCAL_REPO,
            "sha256": sha256(path),
            "role": "dataset" if kind == "microdata" else key,
        }

    added = {
        "annual_manifest": declared("annual_manifest", "diagnostics", evidence_path),
        "annual_acceptance": declared(
            "annual_acceptance", "diagnostics", acceptance_path
        ),
    }
    for key, record in evidence["artifacts"].items():
        if key == LOCAL_DATASET_KEY:
            continue
        h5 = root / record["path"]
        shutil.move(candidate / record["path"], h5)
        receipt = annual / record["projection_receipt"]["path"]
        shutil.copyfile(candidate / receipt.name, receipt)
        added[key] = declared(key, "microdata", h5)
        added[receipt.stem] = declared(receipt.stem, "diagnostics", receipt)
    shutil.rmtree(candidate)
    base_build = manifest["build"]
    manifest["release"] = US_LOCAL_ANNUAL_RELEASE_ID
    manifest["build"] = build
    manifest["metadata"] = {
        **manifest.get("metadata", {}),
        "dataset_years": evidence["metadata"]["dataset_years"],
        "annual_projection_manifest": "annual_manifest",
        "annual_projection_acceptance": "annual_acceptance",
        "annual_projection_base": {
            "repo_id": LOCAL_REPO,
            "release": US_LOCAL_RELEASE_ID,
            "build": base_build,
            "path": base["path"],
            "sha256": base["sha256"],
        },
    }
    manifest["artifacts"].update(added)
    write_json(annual / "release_manifest.json", manifest)
    build_manifest = read_json(local / "build_manifest.json")
    build_manifest.update(release=US_LOCAL_ANNUAL_RELEASE_ID, build=build)
    write_json(annual / "build_manifest.json", build_manifest)
    reseal(annual, root)
    return annual


def rewrite_local_annual_evidence(
    annual: Path, root: Path, mutate: Callable[[dict], None]
) -> None:
    """Change a local-annual folder's candidate manifest and reseal what binds
    it: the acceptance report's ``projection_manifest_sha256``, both files'
    manifest digests and the ledger."""
    evidence_path = annual / "annual_manifest.json"
    evidence = read_json(evidence_path)
    mutate(evidence)
    write_json(evidence_path, evidence)
    acceptance = read_json(annual / "annual_acceptance.json")
    acceptance["projection_manifest_sha256"] = sha256(evidence_path)
    write_json(annual / "annual_acceptance.json", acceptance)
    reseal(annual, root)


def replace_projected_h5(
    annual: Path,
    root: Path,
    year: int,
    tables: dict[str, pd.DataFrame],
    *,
    stored_year: int | None = None,
) -> None:
    """Rewrite one projected year's H5 and reseal every digest that pins it:
    the candidate record, the acceptance year, the release manifest entry
    and the ledger line."""
    key = projected_key(year)
    h5 = write_local_h5(
        root / f"{key}.h5", year if stored_year is None else stored_year, tables
    )
    digest = sha256(h5)

    def repin(evidence: dict) -> None:
        evidence["artifacts"][key]["sha256"] = digest

    rewrite_local_annual_evidence(annual, root, repin)
    acceptance = read_json(annual / "annual_acceptance.json")
    acceptance["years"][str(year)]["sha256"] = digest
    write_json(annual / "annual_acceptance.json", acceptance)
    manifest = read_json(annual / "release_manifest.json")
    manifest["artifacts"][key]["sha256"] = digest
    write_json(annual / "release_manifest.json", manifest)
    reseal(annual, root)


__all__ = [name for name in globals() if not name.startswith("__")]
