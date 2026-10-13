"""Annual publication binds accepted years to exact native H5 artifacts."""

from __future__ import annotations

# ruff: noqa: F401
import hashlib
import json
import shutil

import h5py
import numpy as np
import pytest

from microcosm.data.annual_projections import validate_annual_projection_extension
from microcosm.data.contract import ReleaseContractError, validate_release_dir


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _h5(path, year, *, person_id=1, weight=10.0):
    with h5py.File(path, "w") as store:
        store.create_dataset(
            "_time_period/table",
            data=np.array([(0, year)], dtype=[("index", "i8"), ("values", "i8")]),
        )
        for entity in (
            "person",
            "household",
            "tax_unit",
            "spm_unit",
            "family",
            "marital_unit",
        ):
            fields = [("index", "i8"), (f"{entity}_id", "i8")]
            row = [0, person_id if entity == "person" else 1]
            if entity == "person":
                fields.extend(
                    [("person_household_id", "i8"), ("employment_income", "f8")]
                )
                row.extend([1, 100.0 * (year - 2023)])
            if entity == "household":
                fields.append(("household_weight", "f8"))
                row.append(weight)
            store.create_dataset(
                f"{entity}/table", data=np.array([tuple(row)], dtype=fields)
            )


@pytest.fixture
def candidate(tmp_path):
    release = tmp_path / "release-id"
    release.mkdir()
    root = tmp_path / "artifacts"
    root.mkdir()
    mapping = {
        "populace_us_2024": {"2024": "populace_us_2024", "2025": "populace_us_2025"}
    }
    manifest = {
        "metadata": {"dataset_years": mapping},
        "artifacts": {},
        "default_datasets": {"national": "populace_us_2024"},
        "build": {
            "build_id": release.name,
            "built_with_model_package": {
                "name": "policyengine-us",
                "version": "2.6.15",
            },
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": "3.32.5",
            },
        },
    }
    evidence = {
        "schema_version": 1,
        "kind": "us_annual_static_aging_candidate",
        "status": "complete",
        "base": {
            "dataset": "populace_us_2024",
            "year": 2024,
            "parent_release": release.name,
        },
        "metadata": {"dataset_years": mapping},
        "artifacts": {},
        "model": {
            "version": "2.6.15",
            "commit": "a" * 40,
            "source_tree_sha256": "b" * 64,
        },
        "runtime": {
            "versions": {"policyengine-us": "2.6.15", "policyengine-core": "3.32.5"}
        },
    }
    acceptance = {
        "schema_version": 1,
        "kind": "us_annual_projection_acceptance",
        "status": "passed",
        "years": {},
        "model": dict(evidence["model"]),
        "runtime": dict(evidence["runtime"]["versions"]),
    }
    for year in (2024, 2025):
        key = f"populace_us_{year}"
        path = root / f"{key}.h5"
        _h5(path, year)
        sha = _sha(path)
        manifest["artifacts"][key] = {
            "path": path.name,
            "sha256": sha,
            "revision": f"{release.name}-annual-20260919T220000Z-a1b2c3d4",
            "kind": "microdata",
        }
        evidence["artifacts"][key] = {
            "year": year,
            "sha256": sha,
            "rows": {
                name: 1
                for name in (
                    "person",
                    "household",
                    "tax_unit",
                    "spm_unit",
                    "family",
                    "marital_unit",
                )
            },
        }
        acceptance["years"][str(year)] = {
            "dataset": key,
            "sha256": sha,
            "checks": {
                name: "passed"
                for name in (
                    "schema",
                    "source_identity",
                    "year",
                    "demographics",
                    "input_aggregates",
                    "runtime",
                )
            },
        }
    evidence["base"]["sha256"] = manifest["artifacts"]["populace_us_2024"]["sha256"]
    projection_path = release / "projection_2025.json"
    projection_path.write_text(
        json.dumps({"base_year": 2024, "year": 2025, "factors": {}})
    )
    evidence["artifacts"]["populace_us_2025"]["projection_receipt"] = {
        "path": projection_path.name,
        "sha256": _sha(projection_path),
    }
    manifest["artifacts"]["projection_2025"] = {
        "path": f"releases/{release.name}/{projection_path.name}",
        "sha256": _sha(projection_path),
        "revision": f"{release.name}-annual-20260919T220000Z-a1b2c3d4",
        "kind": "diagnostics",
    }

    def save():
        evidence_path = release / "annual_manifest.json"
        evidence_path.write_text(json.dumps(evidence))
        acceptance["projection_manifest_sha256"] = _sha(evidence_path)
        acceptance_path = release / "annual_acceptance.json"
        acceptance_path.write_text(json.dumps(acceptance))
        for key, field, path in (
            ("annual_manifest", "annual_projection_manifest", evidence_path),
            ("annual_acceptance", "annual_projection_acceptance", acceptance_path),
        ):
            manifest["metadata"][field] = key
            manifest["artifacts"][key] = {
                "path": f"releases/{release.name}/{path.name}",
                "sha256": _sha(path),
                "revision": f"{release.name}-annual-20260919T220000Z-a1b2c3d4",
            }
        (release / "release_manifest.json").write_text(json.dumps(manifest))

    save()
    return release, root, manifest, evidence, acceptance, save


def add_annual_extension(release, root):
    """Augment an otherwise valid release without replacing its base evidence."""
    manifest = json.loads((release / "release_manifest.json").read_text())
    base_key = manifest["default_datasets"]["national"]
    base = manifest["artifacts"][base_key]
    projected = root / "annual_2025.h5"
    shutil.copyfile(root / base["path"], projected)
    with h5py.File(projected, "r+") as store:
        row = store["_time_period/table"][:]
        row["values"] = 2025
        store["_time_period/table"][:] = row
        rows = {
            entity: len(store[f"{entity}/table"])
            for entity in (
                "person",
                "household",
                "tax_unit",
                "spm_unit",
                "family",
                "marital_unit",
            )
        }
    runtime = {
        "policyengine-us": manifest["build"]["built_with_model_package"]["version"],
        "policyengine-core": manifest["build"]["built_with_core_package"]["version"],
    }
    model = {
        "version": runtime["policyengine-us"],
        "commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
    }
    mapping = {base_key: {"2024": base_key, "2025": "annual_2025"}}
    evidence = {
        "schema_version": 1,
        "kind": "us_annual_static_aging_candidate",
        "status": "complete",
        "base": {
            "dataset": base_key,
            "year": 2024,
            "sha256": base["sha256"],
            "parent_release": release.name,
        },
        "metadata": {"dataset_years": mapping},
        "model": model,
        "runtime": {"versions": runtime},
        "artifacts": {},
    }
    acceptance = {
        "schema_version": 1,
        "kind": "us_annual_projection_acceptance",
        "status": "passed",
        "model": model,
        "runtime": runtime,
        "years": {},
    }
    manifest["artifacts"]["annual_2025"] = {
        "kind": "microdata",
        "path": projected.name,
        "sha256": _sha(projected),
        "repo_id": base["repo_id"],
    }
    for year, key in mapping[base_key].items():
        sha = manifest["artifacts"][key]["sha256"]
        evidence["artifacts"][key] = {"year": int(year), "sha256": sha, "rows": rows}
        acceptance["years"][year] = {
            "dataset": key,
            "sha256": sha,
            "checks": {
                name: "passed"
                for name in (
                    "schema",
                    "source_identity",
                    "year",
                    "demographics",
                    "input_aggregates",
                    "runtime",
                )
            },
        }
    projection_path = release / "projection_2025.json"
    projection_path.write_text(
        json.dumps({"base_year": 2024, "year": 2025, "factors": {}})
    )
    evidence["artifacts"]["annual_2025"]["projection_receipt"] = {
        "path": projection_path.name,
        "sha256": _sha(projection_path),
    }
    manifest["artifacts"]["projection_2025"] = {
        "kind": "diagnostics",
        "path": f"releases/{release.name}/{projection_path.name}",
        "sha256": _sha(projection_path),
        "repo_id": base["repo_id"],
    }
    evidence_path = release / "annual_manifest.json"
    evidence_path.write_text(json.dumps(evidence))
    acceptance["projection_manifest_sha256"] = _sha(evidence_path)
    acceptance_path = release / "annual_acceptance.json"
    acceptance_path.write_text(json.dumps(acceptance))
    manifest["metadata"] = {
        **manifest.get("metadata", {}),
        "dataset_years": mapping,
        "annual_projection_manifest": "annual_manifest",
        "annual_projection_acceptance": "annual_acceptance",
    }
    for key, path in (
        ("annual_manifest", evidence_path),
        ("annual_acceptance", acceptance_path),
    ):
        manifest["artifacts"][key] = {
            "kind": "diagnostics",
            "path": f"releases/{release.name}/{path.name}",
            "sha256": _sha(path),
            "repo_id": base["repo_id"],
        }
    tag = f"{release.name}-annual-20260919T220000Z-a1b2c3d4"
    for artifact in manifest["artifacts"].values():
        artifact["revision"] = tag
    (release / "release_manifest.json").write_text(json.dumps(manifest))
    return tag


__all__ = [name for name in globals() if not name.startswith("__")]
