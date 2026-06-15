"""The release artifact contract: what a published release MUST contain.

The releases already on the Hub disagree with each other — one carries no
``build_manifest.json`` at all, and two different ``release_manifest.json``
schemas coexist (an unversioned early shape next to ``schema_version: 1``).
A consumer iterating ``releases/`` therefore cannot trust the listing, and
every consumer ends up re-implementing its own defensive filter. The charter
makes "stage manifests are load-bearing" a binding process rule; the release
directory is the most public manifest of all, so its contract lives here,
with the producer — not in every consumer.

:func:`validate_release_dir` is the single gate: it checks a local release
directory against the contract and raises :class:`ReleaseContractError`
naming **every** failure at once (a publisher should see the full repair
list, not play whack-a-mole one failure per run). Publishing code calls it
before any byte reaches the Hub.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "RELEASE_MANIFEST_SCHEMA_VERSION",
    "REQUIRED_RELEASE_FILES",
    "US_SOURCE_COVERAGE_DIAGNOSTICS_FILE",
    "ReleaseContractError",
    "required_release_files",
    "validate_release_dir",
]

#: The release-manifest schema this library reads and writes. Bump it with the
#: schema, and keep :func:`validate_release_dir` rejecting drift loudly — the
#: unversioned 1abddeb-era manifest is exactly the silence this guards against.
RELEASE_MANIFEST_SCHEMA_VERSION = 1

#: Files a release directory must contain to count as published. A release
#: missing any of these is invisible to :func:`validate_release_dir`-respecting
#: publishers, by design.
REQUIRED_RELEASE_FILES = (
    "build_manifest.json",
    "release_manifest.json",
    "calibration_diagnostics.json",
)

CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION = 2
US_SOURCE_COVERAGE_DIAGNOSTICS_FILE = "us_source_coverage.json"
SOURCE_COVERAGE_DIAGNOSTICS_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def required_release_files(release_id: str) -> tuple[str, ...]:
    """Files required for a release id's country-specific contract."""
    if release_id.startswith("populace-us-"):
        return (*REQUIRED_RELEASE_FILES, US_SOURCE_COVERAGE_DIAGNOSTICS_FILE)
    return REQUIRED_RELEASE_FILES


class ReleaseContractError(ValueError):
    """A release directory violates the release contract.

    Attributes:
        failures: Every contract violation found, each a self-contained
            human-readable sentence naming the file and field at fault.
    """

    def __init__(self, release_dir: Path, failures: list[str]) -> None:
        self.failures = list(failures)
        bullet_list = "\n".join(f"  - {failure}" for failure in self.failures)
        super().__init__(
            f"Release directory {release_dir} violates the release contract "
            f"({len(self.failures)} failure(s)):\n{bullet_list}"
        )


def _load_json(path: Path, failures: list[str]) -> Mapping | None:
    try:
        loaded = json.loads(
            path.read_text(),
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        failures.append(f"{path.name} is not valid JSON: {exc}.")
        return None
    if not isinstance(loaded, Mapping):
        failures.append(
            f"{path.name} must be a JSON object, got {type(loaded).__name__}."
        )
        return None
    return loaded


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON constant {token}")


def _check_sha256_field(
    *,
    filename: str,
    owner: str,
    value: object,
    failures: list[str],
) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        failures.append(f"{filename} {owner} must be a 64-character lowercase sha256.")


def _check_target_surface_ref(
    surface: object,
    *,
    filename: str,
    owner: str,
    failures: list[str],
) -> None:
    if not isinstance(surface, Mapping):
        failures.append(f"{filename} is missing {owner} target_surface object.")
        return
    _check_sha256_field(
        filename=filename,
        owner=f"{owner}.target_surface.sha256",
        value=surface.get("sha256"),
        failures=failures,
    )
    n_targets = surface.get("n_targets")
    if not isinstance(n_targets, int) or n_targets <= 0:
        failures.append(f"{filename} {owner}.target_surface.n_targets must be > 0.")


def _check_target_registry_ref(
    registry: object,
    *,
    filename: str,
    owner: str,
    failures: list[str],
) -> None:
    if not isinstance(registry, Mapping):
        failures.append(f"{filename} is missing {owner} target_registry object.")
        return
    if not registry.get("version"):
        failures.append(f"{filename} {owner}.target_registry.version is required.")
    n_specs = registry.get("n_specs")
    if not isinstance(n_specs, int) or n_specs <= 0:
        failures.append(f"{filename} {owner}.target_registry.n_specs must be > 0.")


def _check_build_manifest(
    manifest: Mapping, release_id: str, failures: list[str]
) -> None:
    build_id = manifest.get("build_id")
    if not build_id:
        failures.append("build_manifest.json is missing 'build_id'.")
    elif build_id != release_id:
        failures.append(
            f"build_manifest.json 'build_id' is {build_id!r} but the release "
            f"directory is named {release_id!r}; the directory name IS the "
            f"build id."
        )
    code = manifest.get("code")
    if not isinstance(code, Mapping):
        failures.append(
            "build_manifest.json is missing the 'code' object (repository, "
            "git_commit, git_dirty)."
        )
    else:
        if not code.get("repository"):
            failures.append("build_manifest.json 'code.repository' is required.")
        git_commit = code.get("git_commit")
        if not isinstance(git_commit, str) or not _GIT_COMMIT_RE.fullmatch(git_commit):
            failures.append(
                "build_manifest.json 'code.git_commit' must be a full "
                "40-character lowercase git commit."
            )
        if code.get("git_dirty") is not False:
            failures.append(
                "build_manifest.json 'code.git_dirty' must be false for a "
                "publishable release."
            )
        build_sha = manifest.get("build_sha")
        if isinstance(build_sha, str) and isinstance(git_commit, str):
            if not git_commit.startswith(build_sha):
                failures.append(
                    "build_manifest.json 'build_sha' must be a prefix of "
                    "'code.git_commit'."
                )
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        failures.append("build_manifest.json is missing the 'dataset' object.")
    else:
        if not dataset.get("filename"):
            failures.append("build_manifest.json 'dataset' is missing 'filename'.")
        _check_sha256_field(
            filename="build_manifest.json",
            owner="'dataset.sha256'",
            value=dataset.get("sha256"),
            failures=failures,
        )
    calibration = manifest.get("calibration")
    if not isinstance(calibration, Mapping):
        failures.append("build_manifest.json is missing the 'calibration' object.")
    else:
        if not calibration.get("filename"):
            failures.append("build_manifest.json 'calibration.filename' is required.")
        _check_sha256_field(
            filename="build_manifest.json",
            owner="'calibration.sha256'",
            value=calibration.get("sha256"),
            failures=failures,
        )
        _check_target_surface_ref(
            calibration.get("target_surface"),
            filename="build_manifest.json",
            owner="'calibration'",
            failures=failures,
        )
        _check_target_registry_ref(
            calibration.get("target_registry"),
            filename="build_manifest.json",
            owner="'calibration'",
            failures=failures,
        )
    if not isinstance(manifest.get("gates"), Mapping):
        failures.append(
            "build_manifest.json is missing the 'gates' object (the "
            "acceptance-gate verdicts are the point of the manifest)."
        )


def _check_release_manifest(
    manifest: Mapping, release_id: str, failures: list[str]
) -> None:
    schema_version = manifest.get("schema_version")
    if schema_version is None:
        failures.append(
            "release_manifest.json has no 'schema_version'; unversioned "
            "manifests (the 1abddeb-era shape) are not publishable."
        )
    elif schema_version != RELEASE_MANIFEST_SCHEMA_VERSION:
        failures.append(
            f"release_manifest.json 'schema_version' is {schema_version!r}; "
            f"this library publishes version "
            f"{RELEASE_MANIFEST_SCHEMA_VERSION}."
        )
    build = manifest.get("build")
    if not isinstance(build, Mapping) or not build.get("build_id"):
        failures.append("release_manifest.json is missing 'build.build_id'.")
    elif build["build_id"] != release_id:
        failures.append(
            f"release_manifest.json 'build.build_id' is "
            f"{build['build_id']!r} but the release directory is named "
            f"{release_id!r}."
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        failures.append(
            "release_manifest.json must declare a non-empty 'artifacts' mapping."
        )
    else:
        diagnostics_artifact = artifacts.get("calibration_diagnostics")
        if not isinstance(diagnostics_artifact, Mapping):
            failures.append(
                "release_manifest.json artifacts must include "
                "'calibration_diagnostics'."
            )
        elif diagnostics_artifact.get("path") != "calibration_diagnostics.json":
            failures.append(
                "release_manifest.json artifact 'calibration_diagnostics' "
                "must point to calibration_diagnostics.json."
            )
        for key, entry in artifacts.items():
            if not isinstance(entry, Mapping):
                failures.append(
                    f"release_manifest.json artifact {key!r} must be an object."
                )
                continue
            for field in ("path", "repo_id", "sha256"):
                if not entry.get(field):
                    failures.append(
                        f"release_manifest.json artifact {key!r} is missing {field!r}."
                    )
            if isinstance(entry, Mapping):
                _check_sha256_field(
                    filename="release_manifest.json",
                    owner=f"artifact {key!r}.sha256",
                    value=entry.get("sha256"),
                    failures=failures,
                )


def _check_calibration_diagnostics(diagnostics: Mapping, failures: list[str]) -> None:
    schema_version = diagnostics.get("schema_version")
    if schema_version is None:
        failures.append("calibration_diagnostics.json is missing 'schema_version'.")
    elif schema_version != CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION:
        failures.append(
            f"calibration_diagnostics.json 'schema_version' is {schema_version!r}; "
            f"this library publishes version "
            f"{CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION}."
        )

    expected_sections = {
        "target_surface": Mapping,
        "target_registry": Mapping,
        "targets": list,
        "loss_trajectory": list,
        "skipped": list,
        "options": Mapping,
    }
    for section, expected_type in expected_sections.items():
        value = diagnostics.get(section)
        if not isinstance(value, expected_type):
            failures.append(
                f"calibration_diagnostics.json is missing a {section!r} "
                f"{expected_type.__name__}."
            )

    _check_target_surface_ref(
        diagnostics.get("target_surface"),
        filename="calibration_diagnostics.json",
        owner="top-level",
        failures=failures,
    )
    _check_target_registry_ref(
        diagnostics.get("target_registry"),
        filename="calibration_diagnostics.json",
        owner="top-level",
        failures=failures,
    )

    targets = diagnostics.get("targets")
    if isinstance(targets, list):
        surface = diagnostics.get("target_surface")
        if isinstance(surface, Mapping) and surface.get("n_targets") != len(targets):
            failures.append(
                "calibration_diagnostics.json target_surface.n_targets must "
                "equal len(targets)."
            )
        for index, target in enumerate(targets):
            if not isinstance(target, Mapping):
                failures.append(
                    f"calibration_diagnostics.json target row {index} must be an object."
                )
                continue
            for field in (
                "name",
                "target_name",
                "period",
                "entity",
                "aggregation",
                "target",
                "compiled_target",
                "initial_estimate",
                "final_estimate",
                "relative_error",
            ):
                if field not in target:
                    failures.append(
                        "calibration_diagnostics.json target row "
                        f"{index} is missing {field!r}."
                    )
            if not target.get("source"):
                failures.append(
                    "calibration_diagnostics.json target row "
                    f"{index} is missing non-empty 'source'."
                )
            if not isinstance(target.get("metadata"), Mapping):
                failures.append(
                    "calibration_diagnostics.json target row "
                    f"{index} is missing 'metadata' object."
                )


def _check_source_coverage_diagnostics(
    diagnostics: Mapping, failures: list[str]
) -> None:
    schema_version = diagnostics.get("schema_version")
    if schema_version is None:
        failures.append(
            f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} is missing 'schema_version'."
        )
    elif schema_version != SOURCE_COVERAGE_DIAGNOSTICS_SCHEMA_VERSION:
        failures.append(
            f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} 'schema_version' is "
            f"{schema_version!r}; this library publishes version "
            f"{SOURCE_COVERAGE_DIAGNOSTICS_SCHEMA_VERSION}."
        )
    if diagnostics.get("classification") != "release_gate":
        failures.append(
            f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} must declare "
            "'classification'='release_gate'."
        )

    source_contract = diagnostics.get("source_contract")
    if not isinstance(source_contract, Mapping):
        failures.append(
            f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} is missing the "
            "'source_contract' object."
        )
    else:
        if source_contract.get("name") != "us_source_coverage":
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} source_contract.name must "
                "be 'us_source_coverage'."
            )
        arch_commit = source_contract.get("arch_commit")
        if not isinstance(arch_commit, str) or len(arch_commit) != 40:
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} source_contract.arch_commit "
                "must be a 40-character commit hash."
            )

    gate = diagnostics.get("gate")
    if not isinstance(gate, Mapping):
        failures.append(
            f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} is missing the 'gate' object."
        )
    else:
        if gate.get("name") != "us_source_coverage":
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} gate.name must be "
                "'us_source_coverage'."
            )
        if gate.get("passed") is not True:
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} gate.passed must be true."
            )
        gate_failures = gate.get("failures")
        if not isinstance(gate_failures, list):
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} gate.failures must be a list."
            )
        elif gate_failures:
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} gate.failures must be empty."
            )

    expected_sections = {
        "coverage_summary": Mapping,
        "hard_target_families": Mapping,
        "validation_only_families": Mapping,
        "source_gap_families": Mapping,
        "missing_hard_targets": list,
        "reviewed_exclusions": Mapping,
        "validation_only_activated": list,
    }
    for section, expected_type in expected_sections.items():
        value = diagnostics.get(section)
        if not isinstance(value, expected_type):
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} is missing a "
                f"{section!r} {expected_type.__name__}."
            )

    reviewed = diagnostics.get("reviewed_exclusions")
    if isinstance(reviewed, Mapping):
        bad_reviewed = sorted(
            str(alias)
            for alias, reason in reviewed.items()
            if not isinstance(reason, str) or not reason.strip()
        )
        if bad_reviewed:
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} reviewed_exclusions "
                f"need non-empty string reasons for {bad_reviewed}."
            )


def validate_release_dir(release_dir: Path | str) -> None:
    """Check a local release directory against the release contract.

    The directory name is the build id (``populace-us-2024-<sha>-<date>``);
    its files are what :data:`REQUIRED_RELEASE_FILES` names; and both
    manifests must agree with the directory about which build this is.

    Args:
        release_dir: The local ``releases/<build_id>`` directory about to be
            published.

    Raises:
        ReleaseContractError: Naming every violation found — missing files,
            unparseable or unversioned manifests, schema drift, and build-id
            mismatches between the manifests and the directory name.
    """
    release_dir = Path(release_dir)
    release_id = release_dir.name
    failures: list[str] = []

    if not release_dir.is_dir():
        raise ReleaseContractError(release_dir, [f"{release_dir} is not a directory."])

    build_manifest: Mapping | None = None
    release_manifest: Mapping | None = None
    calibration_diagnostics: Mapping | None = None

    for filename in required_release_files(release_id):
        if not (release_dir / filename).is_file():
            failures.append(f"required file {filename!r} is missing.")

    build_manifest_path = release_dir / "build_manifest.json"
    if build_manifest_path.is_file():
        manifest = _load_json(build_manifest_path, failures)
        if manifest is not None:
            build_manifest = manifest
            _check_build_manifest(manifest, release_id, failures)

    release_manifest_path = release_dir / "release_manifest.json"
    if release_manifest_path.is_file():
        manifest = _load_json(release_manifest_path, failures)
        if manifest is not None:
            release_manifest = manifest
            _check_release_manifest(manifest, release_id, failures)

    calibration_diagnostics_path = release_dir / "calibration_diagnostics.json"
    if calibration_diagnostics_path.is_file():
        diagnostics = _load_json(calibration_diagnostics_path, failures)
        if diagnostics is not None:
            calibration_diagnostics = diagnostics
            _check_calibration_diagnostics(diagnostics, failures)

    _check_cross_manifest_consistency(
        build_manifest,
        release_manifest,
        calibration_diagnostics,
        failures,
    )

    source_coverage_path = release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE
    if release_id.startswith("populace-us-") and source_coverage_path.is_file():
        diagnostics = _load_json(source_coverage_path, failures)
        if diagnostics is not None:
            _check_source_coverage_diagnostics(diagnostics, failures)

    if failures:
        raise ReleaseContractError(release_dir, failures)


def _check_cross_manifest_consistency(
    build_manifest: Mapping | None,
    release_manifest: Mapping | None,
    calibration_diagnostics: Mapping | None,
    failures: list[str],
) -> None:
    """Fields duplicated across files must agree exactly."""
    if build_manifest is not None and calibration_diagnostics is not None:
        calibration = build_manifest.get("calibration")
        build_surface = (
            calibration.get("target_surface")
            if isinstance(calibration, Mapping)
            else None
        )
        diagnostics_surface = calibration_diagnostics.get("target_surface")
        if isinstance(build_surface, Mapping) and isinstance(
            diagnostics_surface, Mapping
        ):
            for field in ("sha256", "n_targets"):
                if build_surface.get(field) != diagnostics_surface.get(field):
                    failures.append(
                        "build_manifest.json calibration.target_surface."
                        f"{field} must match calibration_diagnostics.json "
                        f"target_surface.{field}."
                    )
        build_registry = (
            calibration.get("target_registry")
            if isinstance(calibration, Mapping)
            else None
        )
        diagnostics_registry = calibration_diagnostics.get("target_registry")
        if isinstance(build_registry, Mapping) and isinstance(
            diagnostics_registry, Mapping
        ):
            for field in ("version", "n_specs"):
                if build_registry.get(field) != diagnostics_registry.get(field):
                    failures.append(
                        "build_manifest.json calibration.target_registry."
                        f"{field} must match calibration_diagnostics.json "
                        f"target_registry.{field}."
                    )

    if release_manifest is not None and calibration_diagnostics is not None:
        artifacts = release_manifest.get("artifacts")
        if isinstance(artifacts, Mapping):
            diagnostics_artifact = artifacts.get("calibration_diagnostics")
            if isinstance(
                diagnostics_artifact, Mapping
            ) and not diagnostics_artifact.get("sha256"):
                failures.append(
                    "release_manifest.json artifact 'calibration_diagnostics' "
                    "must record the diagnostics sha256."
                )
