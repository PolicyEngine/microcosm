"""Assemble a spec country's transport build into a release build.

The release step for a country whose builds run through the spec-driven
transport command (:mod:`microcosm.build.transport.cli`), as
``tools/assemble_uk_release_dir.py`` is for the UK national release and
``tools/certify_us_national_build.py`` for the US national one (tracker
5b · 13). Like them it reads the build's own outputs, here what the transport
command wrote under ``--out``:

- ``build-status.json``: the outputs the build still owes (it refuses unless
  there are none), the Microcosm commit, the Python and package versions and
  the digest of each JSON spec resource the build ran on (they must equal
  those of the spec this step loads), and the digests of the H5, its
  descriptor and the diagnostics it wrote;
- ``skeleton.h5`` and ``export-descriptor.json``: the population, read back
  against its descriptor, and its period and weight entity;
- ``calibration-diagnostics.json``: the calibration's diagnostics;
- ``manifest.json``: the run manifest, whose gate nodes must all have passed.

From the country's spec it reads the release contract (the dataset file name,
the release repository and the required files), the target and holdout
references and the Chronicle pin. It then writes the release directory
``<out-dir>/<release>/``, holding each file the contract requires
(:mod:`microcosm.build.spec_release_files` writes the source coverage, the
demographics and the reform validation), puts the H5 and a weights file
beside the build under the names the manifests give them, validates the
directory with the release contract and prints the
``microcosm-publish-release-build`` commands that publish the build and then
make it current. Nothing is uploaded.

The release is ``microcosm-<country>-<year>-national`` for the export's
period, refused unless :func:`microcosm.data.contract.publishes_to_build_folder`
admits it. The build is ``microcosm-<country>-<year>-chronicle-<pin>-<time>``:
the first seven characters of the spec's Chronicle pin and the UTC time the
run finished.

Run it as ``python -m microcosm.build.spec_release_assembly --build-dir
<transport --out> --country be``, or through ``tools/assemble_spec_release_dir.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from microcosm.build.spec_release_files import (
    SPEC_RELEASE_FILES,
    demographics_payload,
    reform_validation_payload,
    source_coverage_payload,
)
from microcosm.build.transport.cli import EXPORT_FILENAME
from microcosm.build.transport.compose import load_transport_spec
from microcosm.build.transport.terminal_kernels import validate_export
from microcosm.data.contract import (
    REQUIRED_RELEASE_FILES,
    ReleaseContractError,
    publishes_to_build_folder,
    release_artifact_role,
    validate_release_dir,
)
from microcosm.diagnostics import SUPPORTED_CALIBRATION_DIAGNOSTICS_SCHEMA_VERSIONS
from microcosm.frame.adapters.axiom import AxiomEntityTableDataset
from microcosm.graph import Capabilities, KernelRole, NodeReceipt, RunManifest

__all__ = [
    "BUILD_STATUS_FILE",
    "AssembledRelease",
    "SpecReleaseAssemblyError",
    "assemble_spec_release",
    "main",
    "spec_release_identity",
    "spec_release_manifests",
]

#: The transport build's files this step reads, under its ``--out``.
BUILD_STATUS_FILE = "build-status.json"
RUN_MANIFEST_FILE = "manifest.json"
EXPORT_DESCRIPTOR_FILE = "export-descriptor.json"
BUILD_DIAGNOSTICS_FILE = "calibration-diagnostics.json"
#: The release directory's name for the calibration diagnostics.
RELEASE_DIAGNOSTICS_FILE = "calibration_diagnostics.json"
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
#: Versions a runtime record may carry that name no installed version.
_UNRESOLVED_VERSIONS = frozenset({"", "unavailable", "not-installed", "unknown"})
#: The engine a transport build's populations run through.
_TRANSPORT_ENGINE = "axiom_rules_engine"


class SpecReleaseAssemblyError(ValueError):
    """The build cannot be assembled into a publishable release build."""


@dataclass(frozen=True)
class AssembledRelease:
    """An assembled release build and the commands that publish it."""

    release: str
    build: str
    release_dir: Path
    artifact_root: Path
    dataset: Path
    calibration: Path
    publish_command: str
    promote_command: str

    def summary(self) -> dict[str, object]:
        """What the command prints, as JSON."""
        return {
            "release": self.release,
            "build": self.build,
            "release_dir": str(self.release_dir),
            "artifact_root": str(self.artifact_root),
            "dataset": {"path": str(self.dataset), "sha256": _sha256(self.dataset)},
            "calibration": {
                "path": str(self.calibration),
                "sha256": _sha256(self.calibration),
            },
            "publish_command": self.publish_command,
            "promote_command": self.promote_command,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping:
    try:
        payload = json.loads(path.read_bytes())
    except FileNotFoundError:
        raise SpecReleaseAssemblyError(
            f"The build folder has no {path.name}; pass the transport build's "
            "--out directory."
        ) from None
    except (OSError, ValueError) as error:
        raise SpecReleaseAssemblyError(
            f"{path.name} is not readable JSON: {error}."
        ) from error
    if not isinstance(payload, Mapping):
        raise SpecReleaseAssemblyError(f"{path.name} must hold a JSON object.")
    return payload


def _mapping(value: object, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise SpecReleaseAssemblyError(f"{label} must be an object.")
    return value


def _code_record(status: Mapping) -> dict[str, object]:
    """The clean Microcosm commit the build recorded that it ran at."""
    code = status.get("code")
    if not isinstance(code, Mapping):
        raise SpecReleaseAssemblyError(
            f"{BUILD_STATUS_FILE} records no Microcosm commit ('code'); a "
            "transport build made before it recorded one cannot be assembled."
        )
    repository = code.get("repository")
    commit = code.get("git_commit")
    if not isinstance(repository, str) or not repository:
        raise SpecReleaseAssemblyError(
            f"{BUILD_STATUS_FILE} 'code.repository' must name the repository "
            "the commit belongs to."
        )
    if not isinstance(commit, str) or _GIT_COMMIT.fullmatch(commit) is None:
        raise SpecReleaseAssemblyError(
            f"{BUILD_STATUS_FILE} 'code.git_commit' must be a full 40-character "
            f"lowercase git commit, got {commit!r}."
        )
    dirty = code.get("git_dirty")
    if dirty is not False:
        raise SpecReleaseAssemblyError(
            f"{BUILD_STATUS_FILE} records 'code.git_dirty' as {dirty!r}: only a "
            "build from a checkout measured as clean is publishable."
        )
    return {"repository": repository, "git_commit": commit, "git_dirty": False}


def _runtime_record(status: Mapping) -> dict[str, str]:
    """The Python and package versions the build recorded that it ran on."""
    runtime = status.get("runtime")
    if not isinstance(runtime, Mapping) or not runtime:
        raise SpecReleaseAssemblyError(
            f"{BUILD_STATUS_FILE} records no 'runtime' (the Python and package "
            "versions the build ran on)."
        )
    record: dict[str, str] = {}
    for name, version in runtime.items():
        if not isinstance(name, str) or not isinstance(version, str) or not version:
            raise SpecReleaseAssemblyError(
                f"{BUILD_STATUS_FILE} 'runtime' must map each name to a version "
                f"string, got {name!r}: {version!r}."
            )
        record[name] = version
    for required in ("python", "microcosm-data"):
        if record.get(required, "") in _UNRESOLVED_VERSIONS:
            raise SpecReleaseAssemblyError(
                f"{BUILD_STATUS_FILE} 'runtime.{required}' must be a resolved "
                f"version, got {record.get(required)!r}; the release manifest "
                "records it."
            )
    return record


def _require_build_spec(
    status: Mapping, *, country: str, resource_hashes: Mapping[str, str], commit: str
) -> None:
    """The spec this step loaded is the one the build ran on.

    The release files, the build's name and the source coverage check read
    the spec the assembler loads, so a checkout that moved since the build (a
    Chronicle re-pin, say) would pair the build with references it did not
    compile. Every resource's digest must equal the one the build recorded.
    """
    spec = status.get("spec")
    recorded = spec.get("resource_hashes") if isinstance(spec, Mapping) else None
    if not isinstance(recorded, Mapping) or spec.get("country") != country:
        raise SpecReleaseAssemblyError(
            f"{BUILD_STATUS_FILE} records no digests of the {country!r} spec "
            "the build ran on ('spec'); a transport build made before it "
            "recorded them cannot be assembled."
        )
    differing = sorted(
        name
        for name in {*recorded, *resource_hashes}
        if recorded.get(name) != resource_hashes.get(name)
    )
    if differing:
        raise SpecReleaseAssemblyError(
            f"The build ran on another {country!r} spec: {differing} differ from "
            "the spec this step loaded. Assemble with the spec the build ran "
            f"on, from a checkout at the build's commit {commit}."
        )


def _require_produced(produced: Mapping, name: str, path: Path) -> None:
    """The file is the one the build recorded that it wrote."""
    entry = produced.get(name)
    expected = entry.get("sha256") if isinstance(entry, Mapping) else None
    if not isinstance(expected, str):
        raise SpecReleaseAssemblyError(
            f"{BUILD_STATUS_FILE} records no SHA-256 for {path.name} "
            f"('produced.{name}')."
        )
    if not path.is_file():
        raise SpecReleaseAssemblyError(f"The build folder has no {path.name}.")
    observed = _sha256(path)
    if observed != expected:
        raise SpecReleaseAssemblyError(
            f"{path.name} has SHA-256 {observed}, but {BUILD_STATUS_FILE} "
            f"records {expected}: the file changed after the build wrote it."
        )


def _role(node: NodeReceipt) -> KernelRole:
    capabilities = node.capabilities
    if isinstance(capabilities, Capabilities):
        return capabilities.role
    return KernelRole(str(capabilities.get("role", KernelRole.COMPUTE.value)))


def _run_record(path: Path, country: str) -> tuple[RunManifest, dict[str, str]]:
    """The run manifest and its gate outcomes; every gate must have passed.

    ``known_failures`` lists each gate node whose outcome is neither ``pass``
    nor ``not_applicable`` and each rejected node, so refusing on it refuses
    every gate that did not pass.
    """
    try:
        run = RunManifest.from_json(path.read_bytes())
    except FileNotFoundError:
        raise SpecReleaseAssemblyError(
            f"The build folder has no {path.name}."
        ) from None
    except (TypeError, ValueError, KeyError) as error:
        raise SpecReleaseAssemblyError(
            f"{path.name} is not a valid run manifest: {error}."
        ) from error
    if run.country != country:
        raise SpecReleaseAssemblyError(
            f"{path.name} is a run of {run.country!r}, not {country!r}."
        )
    try:
        failures = run.known_failures
        tier = run.tier
    except ValueError as error:
        raise SpecReleaseAssemblyError(f"{path.name}: {error}.") from error
    if failures:
        raise SpecReleaseAssemblyError(
            f"The run's gate or rejected nodes {list(failures)} did not pass; a "
            "build whose gates failed is not publishable."
        )
    if tier not in (None, "certified"):
        raise SpecReleaseAssemblyError(
            f"{path.name} records the release tier {tier!r}, not 'certified'."
        )
    outcomes = {
        node_id: str(node.receipt.get("outcome"))
        for node_id, node in sorted(run.nodes.items())
        if _role(node) is KernelRole.GATE
    }
    return run, outcomes


def _utc_stamp(value: str, label: str) -> str:
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise SpecReleaseAssemblyError(
            f"{label} must be an ISO 8601 time, got {value!r}."
        ) from None
    if moment.tzinfo is None:
        raise SpecReleaseAssemblyError(f"{label} must name its time zone: {value!r}.")
    return moment.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def spec_release_identity(
    *, country: str, period: int, chronicle_commit: str, finished_at: str
) -> tuple[str, str]:
    """The release and the build a spec country's build publishes as.

    The release is the country's national release for ``period``; the build
    names the period, the first seven characters of the Chronicle pin and
    the UTC time the run finished, as BE's builder has named its builds.
    """
    if _GIT_COMMIT.fullmatch(chronicle_commit) is None:
        raise SpecReleaseAssemblyError(
            "The spec's Chronicle pin must be a full 40-character commit, got "
            f"{chronicle_commit!r}."
        )
    stamp = _utc_stamp(finished_at, "The run manifest's 'finished_at'")
    release = f"microcosm-{country}-{period}-national"
    build = f"microcosm-{country}-{period}-chronicle-{chronicle_commit[:7]}-{stamp}"
    return release, build


def spec_release_manifests(
    *,
    country: str,
    release: str,
    build: str,
    repository: str,
    code: Mapping[str, object],
    runtime: Mapping[str, str],
    period: int,
    dataset_filename: str,
    dataset_sha256: str,
    calibration_filename: str,
    calibration_sha256: str,
    weight_entity: str,
    diagnostics_sha256: str,
    target_surface: Mapping[str, object],
    target_registry: Mapping[str, object],
    added_files: Mapping[str, tuple[str, str]],
    gate_outcomes: Mapping[str, str],
    tier: str | None,
    chronicle: Mapping[str, object],
    spec_resource_hashes: Mapping[str, str],
    run: Mapping[str, object],
    created_at: str,
    release_manifest_country: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """The build's build manifest and release manifest: a pure function.

    ``added_files`` maps each added file's artifact key to its file name and
    SHA-256. A transport build runs its population through the Axiom rules
    engine, so the build manifest records that engine with the packages the
    build recorded (``engines``), as a BE build's manifest must; it records
    no PolicyEngine package.
    """
    dataset_key = dataset_filename.removesuffix(".h5")
    packages = {name: version for name, version in runtime.items() if name != "python"}
    build_manifest: dict[str, object] = {
        "release": release,
        "build": build,
        "country": country,
        "code": dict(code),
        "build_sha": str(code["git_commit"])[:7],
        "runtime": dict(runtime),
        "engines": {
            _TRANSPORT_ENGINE: {"python": runtime["python"], "packages": packages}
        },
        "population_year": period,
        "dataset": {"filename": dataset_filename, "sha256": dataset_sha256},
        "calibration": {
            "filename": calibration_filename,
            "sha256": calibration_sha256,
            "weight_entity": weight_entity,
            "target_surface": {
                "sha256": target_surface.get("sha256"),
                "n_targets": target_surface.get("n_targets"),
            },
            "target_registry": {
                "version": target_registry.get("version"),
                "n_specs": target_registry.get("n_specs"),
            },
        },
        "gates": {
            "tier": tier,
            "known_failures": [],
            "outcomes": dict(gate_outcomes),
        },
        "chronicle": dict(chronicle),
        "spec": {"country": country, "resource_hashes": dict(spec_resource_hashes)},
        "run": dict(run),
        "created_at": created_at,
    }

    def artifact(key: str, kind: str, path: str, sha256: str) -> dict[str, str]:
        return {
            "kind": kind,
            "path": path,
            "repo_id": repository,
            "sha256": sha256,
            "role": release_artifact_role(key, kind),
        }

    artifacts = {
        dataset_key: artifact(
            dataset_key, "microdata", dataset_filename, dataset_sha256
        ),
        f"{dataset_key}_calibration": artifact(
            f"{dataset_key}_calibration",
            "calibration",
            calibration_filename,
            calibration_sha256,
        ),
        "calibration_diagnostics": artifact(
            "calibration_diagnostics",
            "diagnostics",
            RELEASE_DIAGNOSTICS_FILE,
            diagnostics_sha256,
        ),
        **{
            key: artifact(key, "diagnostics", filename, sha256)
            for key, (filename, sha256) in added_files.items()
        },
    }
    release_manifest: dict[str, object] = {
        "schema_version": 1,
        "release": release,
        "build": build,
        "built_at": run.get("finished_at"),
        "data_package": {
            "name": "microcosm-data",
            "version": runtime["microcosm-data"],
        },
        "default_datasets": {"national": dataset_key},
        "artifacts": artifacts,
        "description": (
            f"{country.upper()} transport build {build} assembled into release "
            f"{release}."
        ),
    }
    if release_manifest_country is not None:
        release_manifest["country"] = release_manifest_country
    return build_manifest, release_manifest


def _write_json(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    return _sha256(path)


def _spec_documents(country: str) -> tuple[Mapping, Mapping[str, str]]:
    try:
        spec = load_transport_spec(country)
    except Exception as error:  # noqa: BLE001 - reported as a refusal
        raise SpecReleaseAssemblyError(
            f"The {country!r} country spec does not load: {error}"
        ) from error
    return spec["resources"], spec["resource_hashes"]


def _contract_files(
    contract: Mapping, *, period: int
) -> tuple[str, str, str, tuple[str, ...]]:
    """The dataset file, its repository, its key and the required files."""
    template = contract.get("dataset_filename_template")
    if not isinstance(template, str) or not template.endswith(".h5"):
        raise SpecReleaseAssemblyError(
            "The release contract's 'dataset_filename_template' must name an "
            f".h5 file, got {template!r}."
        )
    dataset_filename = template.format(year=period)
    hf = contract.get("hf")
    repository = hf.get("artifact_repo") if isinstance(hf, Mapping) else None
    if not isinstance(repository, str) or not repository:
        raise SpecReleaseAssemblyError(
            "The release contract's 'hf.artifact_repo' must name the release "
            "repository."
        )
    required = contract.get("required_release_files", ())
    if not isinstance(required, list | tuple) or not all(
        isinstance(name, str) for name in required
    ):
        raise SpecReleaseAssemblyError(
            "The release contract's 'required_release_files' must list file names."
        )
    return (
        dataset_filename,
        repository,
        dataset_filename.removesuffix(".h5"),
        tuple(name.format(year=period) for name in required),
    )


def _weights(dataset: Path, weight_entity: str) -> dict[str, np.ndarray]:
    """The weight entity's IDs and weights, from the H5's own table."""
    table = AxiomEntityTableDataset(file_path=dataset).tables.get(weight_entity)
    id_column, weight_column = f"{weight_entity}_id", f"{weight_entity}_weight"
    if table is None or {id_column, weight_column} - set(table.columns):
        raise SpecReleaseAssemblyError(
            f"{dataset.name} has no {weight_entity!r} table with {id_column!r} and "
            f"{weight_column!r} columns, so its weights cannot be written."
        )
    return {
        id_column: table[id_column].to_numpy(),
        weight_column: table[weight_column].to_numpy(dtype=np.float64),
    }


def assemble_spec_release(
    build_dir: Path | str,
    *,
    country: str,
    out_dir: Path | str,
    created_at: str | None = None,
) -> AssembledRelease:
    """Assemble the transport build in ``build_dir`` into ``out_dir/<release>``.

    Raises:
        SpecReleaseAssemblyError: If the build cannot be published: it owes
            outputs; records no clean commit, no runtime or no spec digests;
            ran on another spec; a file changed after the build wrote it; a
            gate failed; its year has no release; its source coverage fails;
            an earlier assembly is in the way; or the release contract
            refuses the assembled directory. Nothing is left in the build
            folder or the release directory then; ``out_dir`` itself may
            have been created.
    """
    build_dir = Path(build_dir).absolute()
    out_dir = Path(out_dir).absolute()
    resources, resource_hashes = _spec_documents(country)
    contract = _mapping(resources.get("release_contract"), "The release contract")

    status = _load_json(build_dir / BUILD_STATUS_FILE)
    if status.get("country") != country:
        raise SpecReleaseAssemblyError(
            f"{BUILD_STATUS_FILE} is a build of {status.get('country')!r}, not "
            f"{country!r}."
        )
    owed = status.get("not_yet_produced")
    if not isinstance(owed, list):
        raise SpecReleaseAssemblyError(
            f"{BUILD_STATUS_FILE} 'not_yet_produced' must list what the build owes."
        )
    if owed:
        raise SpecReleaseAssemblyError(
            f"The build still owes outputs its spec declares: {owed}. Only a "
            "build that owes nothing is publishable."
        )
    code = _code_record(status)
    runtime = _runtime_record(status)
    _require_build_spec(
        status,
        country=country,
        resource_hashes=resource_hashes,
        commit=str(code["git_commit"]),
    )
    produced = _mapping(status.get("produced"), f"{BUILD_STATUS_FILE} 'produced'")
    dataset_source = build_dir / EXPORT_FILENAME
    diagnostics_source = build_dir / BUILD_DIAGNOSTICS_FILE
    _require_produced(produced, "skeleton_dataset", dataset_source)
    _require_produced(produced, "calibration_diagnostics", diagnostics_source)
    _require_produced(produced, "export_descriptor", build_dir / EXPORT_DESCRIPTOR_FILE)

    descriptor = _load_json(build_dir / EXPORT_DESCRIPTOR_FILE)
    try:
        readback = validate_export(dataset_source, descriptor)
    except (KeyError, TypeError, ValueError) as error:
        raise SpecReleaseAssemblyError(
            f"{EXPORT_FILENAME} does not read back against "
            f"{EXPORT_DESCRIPTOR_FILE}: {error}"
        ) from error
    if readback.get("passed") is not True:
        raise SpecReleaseAssemblyError(
            f"{EXPORT_FILENAME} differs from {EXPORT_DESCRIPTOR_FILE}: "
            f"{readback.get('failures')}."
        )
    period = int(descriptor["time_period"])
    weight_entity = descriptor.get("weight_entity")
    if not isinstance(weight_entity, str) or not weight_entity:
        raise SpecReleaseAssemblyError(
            f"{EXPORT_DESCRIPTOR_FILE} names no 'weight_entity'."
        )

    run, gate_outcomes = _run_record(build_dir / RUN_MANIFEST_FILE, country)
    feed = _mapping(resources.get("chronicle_feed"), "The spec's Chronicle feed")
    chronicle_commit = str(feed.get("source_commit"))
    release, build = spec_release_identity(
        country=country,
        period=period,
        chronicle_commit=chronicle_commit,
        finished_at=run.finished_at,
    )
    if not publishes_to_build_folder(release):
        raise SpecReleaseAssemblyError(
            f"A {period} build would publish into {release}, which is not a "
            "release that publishes into build folders "
            "(microcosm.data.contract.publishes_to_build_folder)."
        )

    diagnostics_bytes = diagnostics_source.read_bytes()
    try:
        decoded = json.loads(diagnostics_bytes)
    except ValueError as error:
        raise SpecReleaseAssemblyError(
            f"{BUILD_DIAGNOSTICS_FILE} is not readable JSON: {error}."
        ) from error
    diagnostics = _mapping(decoded, BUILD_DIAGNOSTICS_FILE)
    if diagnostics.get("schema_version") not in (
        SUPPORTED_CALIBRATION_DIAGNOSTICS_SCHEMA_VERSIONS
    ):
        raise SpecReleaseAssemblyError(
            f"{BUILD_DIAGNOSTICS_FILE} declares schema_version "
            f"{diagnostics.get('schema_version')!r}, which the release contract "
            "does not take."
        )
    target_surface = _mapping(
        diagnostics.get("target_surface"), f"{BUILD_DIAGNOSTICS_FILE} 'target_surface'"
    )
    target_registry = _mapping(
        diagnostics.get("target_registry"),
        f"{BUILD_DIAGNOSTICS_FILE} 'target_registry'",
    )

    dataset_filename, repository, dataset_key, required = _contract_files(
        contract, period=period
    )
    calibration_filename = f"{dataset_key}_calibration.npz"
    written = {
        *REQUIRED_RELEASE_FILES,
        *SPEC_RELEASE_FILES.values(),
        dataset_filename,
        calibration_filename,
    }
    unwritable = sorted(set(required) - written)
    if unwritable:
        raise SpecReleaseAssemblyError(
            f"The release contract requires {unwritable}, which this step does "
            "not write."
        )
    target_references = _mapping(
        resources.get("target_references"), "The spec's target references"
    )
    holdout_references = resources.get("holdout_references")
    payloads: dict[str, object] = {}
    if SPEC_RELEASE_FILES["source_coverage"] in required:
        coverage = source_coverage_payload(
            country=country,
            target_references=target_references,
            holdout_references=(
                holdout_references if isinstance(holdout_references, Mapping) else None
            ),
            diagnostics=diagnostics,
            source_contract={
                "chronicle_repository": feed.get("source_repo"),
                "chronicle_commit": chronicle_commit,
                "resources": {
                    name: resource_hashes[name]
                    for name in ("target_references", "holdout_references")
                    if name in resource_hashes
                },
            },
        )
        gate = _mapping(coverage["gate"], "The source coverage gate")
        if gate.get("passed") is not True:
            raise SpecReleaseAssemblyError(
                "The build's source coverage fails, and it is a release gate: "
                + " ".join(str(failure) for failure in gate.get("failures", ()))
            )
        payloads["source_coverage"] = coverage
    if SPEC_RELEASE_FILES["demographics"] in required:
        payloads["demographics"] = demographics_payload(
            country=country,
            target_references=target_references,
            diagnostics=diagnostics,
            period=period,
        )
    if SPEC_RELEASE_FILES["reform_validation"] in required:
        payloads["reform_validation"] = reform_validation_payload(
            country=country, period=period
        )

    destination = out_dir / release
    if destination.exists() or destination.is_symlink():
        raise SpecReleaseAssemblyError(
            f"{destination} already exists; remove the earlier assembly before "
            "assembling again."
        )
    dataset_destination = build_dir / dataset_filename
    calibration_destination = build_dir / calibration_filename
    for path in (dataset_destination, calibration_destination):
        if path.exists() or path.is_symlink():
            raise SpecReleaseAssemblyError(
                f"{path} already exists; remove it before assembling again."
            )
    weights = _weights(dataset_source, weight_entity)
    created_at = created_at or datetime.now(UTC).isoformat()

    # Nothing is written in place, as in the UK assemblers: the release
    # directory is staged and validated, then renamed into a destination
    # proven empty. The H5 copy and the weights stage beside their own
    # destinations, so no rename crosses a file system boundary. The temporary
    # weights name keeps the .npz suffix, which np.savez would otherwise add.
    token = uuid.uuid4().hex
    staging_parent = out_dir / f".assemble-{token}"
    staged = staging_parent / release
    staged_dataset = build_dir / f".{dataset_filename}.{token}.tmp.h5"
    staged_calibration = build_dir / f".{calibration_filename}.{token}.tmp.npz"
    try:
        staged.mkdir(parents=True)
        shutil.copyfile(dataset_source, staged_dataset)
        np.savez(staged_calibration, **weights)
        dataset_sha256 = _sha256(staged_dataset)
        calibration_sha256 = _sha256(staged_calibration)
        (staged / RELEASE_DIAGNOSTICS_FILE).write_bytes(diagnostics_bytes)
        added_files = {
            key: (
                SPEC_RELEASE_FILES[key],
                _write_json(staged / SPEC_RELEASE_FILES[key], payload),
            )
            for key, payload in payloads.items()
        }
        country_label = contract.get("release_manifest_country")
        build_manifest, release_manifest = spec_release_manifests(
            country=country,
            release=release,
            build=build,
            repository=repository,
            code=code,
            runtime=runtime,
            period=period,
            dataset_filename=dataset_filename,
            dataset_sha256=dataset_sha256,
            calibration_filename=calibration_filename,
            calibration_sha256=calibration_sha256,
            weight_entity=weight_entity,
            diagnostics_sha256=_sha256(staged / RELEASE_DIAGNOSTICS_FILE),
            target_surface=target_surface,
            target_registry=target_registry,
            added_files=added_files,
            gate_outcomes=gate_outcomes,
            tier=run.tier,
            chronicle={
                "repository": feed.get("source_repo"),
                "commit": chronicle_commit,
            },
            spec_resource_hashes=resource_hashes,
            # The manifest's start is that of the command's last pass over
            # the stored graph, not of the build, so it is not recorded.
            run={"key": run.key, "finished_at": run.finished_at},
            created_at=created_at,
            release_manifest_country=(
                country_label if isinstance(country_label, str) else None
            ),
        )
        _write_json(staged / "build_manifest.json", build_manifest)
        _write_json(staged / "release_manifest.json", release_manifest)
        try:
            validate_release_dir(staged)
        except ReleaseContractError as error:
            raise SpecReleaseAssemblyError(
                "The release contract refuses the assembled directory: "
                + " ".join(error.failures)
            ) from error
        staged_calibration.rename(calibration_destination)
        try:
            staged_dataset.rename(dataset_destination)
            try:
                staged.rename(destination)
            except BaseException:
                dataset_destination.unlink(missing_ok=True)
                raise
        except BaseException:
            calibration_destination.unlink(missing_ok=True)
            raise
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)
        staged_dataset.unlink(missing_ok=True)
        staged_calibration.unlink(missing_ok=True)

    publish_command = shlex.join(
        [
            "uv",
            "run",
            "microcosm-publish-release-build",
            str(destination),
            "--repo-id",
            repository,
            "--artifact-root",
            str(build_dir),
        ]
    )
    promote_command = shlex.join(
        [
            "uv",
            "run",
            "microcosm-publish-release-build",
            str(destination),
            "--repo-id",
            repository,
            "--promote",
        ]
    )
    return AssembledRelease(
        release=release,
        build=build,
        release_dir=destination,
        artifact_root=build_dir,
        dataset=dataset_destination,
        calibration=calibration_destination,
        publish_command=publish_command,
        promote_command=promote_command,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command; return 0 on an assembled build and 1 on a refusal."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--build-dir",
        required=True,
        type=Path,
        help="The transport build's --out directory.",
    )
    parser.add_argument(
        "--country", required=True, help="The packaged country spec, such as be."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("releases"),
        help="Where the release directory is written (default: releases).",
    )
    args = parser.parse_args(argv)
    try:
        assembled = assemble_spec_release(
            args.build_dir, country=args.country, out_dir=args.out_dir
        )
    except SpecReleaseAssemblyError as error:
        print(f"release assembly refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps(assembled.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
