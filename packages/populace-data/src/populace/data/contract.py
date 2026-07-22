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

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from populace.data.us_critical_targets import (
    US_CRITICAL_TARGET_FIT_REQUIREMENTS as _US_CRITICAL_TARGET_FIT_REQUIREMENTS,
)
from populace.data.us_critical_targets import (
    US_CRITICAL_TARGET_IMPROVEMENT_MAX_ABS_RELATIVE_ERROR as _US_CRITICAL_TARGET_IMPROVEMENT_MAX_ABS_RELATIVE_ERROR,
)

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

CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION = 4
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        failures.append(
            "build_manifest.json is missing the 'runtime' object "
            "(Python and package versions used for target materialization)."
        )
    else:
        required_packages = ["python", "policyengine-core"]
        model_package = _expected_model_package(release_id)
        if model_package is not None:
            required_packages.append(model_package)
        for package in required_packages:
            value = runtime.get(package)
            if not value or value in {"not-installed", "unknown"}:
                failures.append(
                    f"build_manifest.json 'runtime.{package}' must be a resolved "
                    "version, not missing or unknown."
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
    if isinstance(build, Mapping):
        built_with_core_package = build.get("built_with_core_package")
        built_with_model_package = build.get("built_with_model_package")
        _check_release_manifest_package(
            built_with_core_package,
            field="build.built_with_core_package",
            expected_name="policyengine-core",
            failures=failures,
        )
        _check_release_manifest_package(
            built_with_model_package,
            field="build.built_with_model_package",
            expected_name=_expected_model_package(release_id),
            failures=failures,
        )
        _check_compatible_package_entries(
            manifest.get("compatible_core_packages"),
            field="compatible_core_packages",
            expected_name="policyengine-core",
            built_with_package=built_with_core_package,
            failures=failures,
        )
        _check_compatible_package_entries(
            manifest.get("compatible_model_packages"),
            field="compatible_model_packages",
            expected_name=_expected_model_package(release_id),
            built_with_package=built_with_model_package,
            failures=failures,
        )
    data_package = manifest.get("data_package")
    _check_release_manifest_package(
        data_package,
        field="data_package",
        expected_name="populace-data",
        failures=failures,
    )
    default_datasets = manifest.get("default_datasets")
    if not isinstance(default_datasets, Mapping):
        failures.append(
            "release_manifest.json is missing the 'default_datasets' object."
        )
    elif not default_datasets.get("national"):
        failures.append(
            "release_manifest.json 'default_datasets.national' is required."
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
            for field in ("kind", "path", "repo_id", "revision", "sha256"):
                if not entry.get(field):
                    failures.append(
                        f"release_manifest.json artifact {key!r} is missing {field!r}."
                    )
            revision = entry.get("revision")
            if isinstance(revision, str) and revision != release_id:
                failures.append(
                    f"release_manifest.json artifact {key!r} revision is "
                    f"{revision!r}, expected the release id {release_id!r}."
                )
            if isinstance(entry, Mapping):
                _check_sha256_field(
                    filename="release_manifest.json",
                    owner=f"artifact {key!r}.sha256",
                    value=entry.get("sha256"),
                    failures=failures,
                )
        if release_id.startswith("populace-us-"):
            _check_us_release_has_no_split_microdata_artifacts(
                artifacts,
                failures=failures,
            )
        if (
            release_id.startswith("populace-us-")
            and _artifact_by_path(manifest, US_SOURCE_COVERAGE_DIAGNOSTICS_FILE) is None
        ):
            failures.append(
                "release_manifest.json artifacts must include "
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE!r} for US releases."
            )
        if isinstance(default_datasets, Mapping):
            national = default_datasets.get("national")
            if isinstance(national, str) and national not in artifacts:
                failures.append(
                    "release_manifest.json 'default_datasets.national' "
                    f"points to {national!r}, which is not an artifact key."
                )
            elif isinstance(national, str):
                default_artifact = artifacts.get(national)
                if (
                    isinstance(default_artifact, Mapping)
                    and default_artifact.get("kind") != "microdata"
                ):
                    failures.append(
                        "release_manifest.json 'default_datasets.national' "
                        f"points to artifact {national!r}, whose kind is "
                        f"{default_artifact.get('kind')!r}, not 'microdata'."
                    )


def _check_us_release_has_no_split_microdata_artifacts(
    artifacts: Mapping, *, failures: list[str]
) -> None:
    split_keys = {
        key
        for key in artifacts
        if isinstance(key, str)
        and (key.startswith("states/") or key.startswith("districts/"))
    }
    if not split_keys:
        return
    failures.append(
        "release_manifest.json US releases must publish a single national "
        "microdata artifact; split state/district microdata artifacts are no "
        f"longer part of the release contract: {_sample_values(sorted(split_keys))}"
    )


def _sample_values(values: list[object], limit: int = 10) -> list[object]:
    sample = list(values[:limit])
    if len(values) > limit:
        sample.append(f"... +{len(values) - limit} more")
    return sample


def _check_release_manifest_package(
    package: object,
    *,
    field: str,
    expected_name: str | None,
    failures: list[str],
) -> None:
    if not isinstance(package, Mapping):
        failures.append(f"release_manifest.json is missing the '{field}' object.")
        return
    name = package.get("name")
    if expected_name is not None and name != expected_name:
        failures.append(
            f"release_manifest.json '{field}.name' must be {expected_name!r}."
        )
    elif not name:
        failures.append(f"release_manifest.json '{field}.name' is required.")
    version = package.get("version")
    if not version:
        failures.append(f"release_manifest.json '{field}.version' is required.")
    elif version in {"not-installed", "unknown"}:
        failures.append(
            f"release_manifest.json '{field}.version' must be resolved, "
            f"not {version!r}."
        )


def _check_compatible_package_entries(
    entries: object,
    *,
    field: str,
    expected_name: str | None,
    built_with_package: object,
    failures: list[str],
) -> None:
    if expected_name is None:
        return
    if not isinstance(entries, list) or not entries:
        failures.append(
            f"release_manifest.json must declare a non-empty '{field}' list."
        )
        return

    matching_specifiers: list[str] = []
    for index, entry in enumerate(entries):
        owner = f"release_manifest.json {field}[{index}]"
        if not isinstance(entry, Mapping):
            failures.append(f"{owner} must be an object.")
            continue
        name = entry.get("name")
        specifier = entry.get("specifier")
        if not isinstance(name, str) or not name:
            failures.append(f"{owner}.name is required.")
        if not isinstance(specifier, str) or not specifier.strip():
            failures.append(f"{owner}.specifier is required.")
            continue
        try:
            SpecifierSet(specifier)
        except InvalidSpecifier:
            failures.append(
                f"{owner}.specifier {specifier!r} is not a valid PEP 440 specifier."
            )
            continue
        if name == expected_name:
            matching_specifiers.append(specifier)

    if not matching_specifiers:
        failures.append(
            f"release_manifest.json '{field}' must include {expected_name!r}."
        )
        return

    built_version = (
        built_with_package.get("version")
        if isinstance(built_with_package, Mapping)
        and built_with_package.get("name") == expected_name
        else None
    )
    if not isinstance(built_version, str) or not built_version:
        return
    try:
        version = Version(built_version)
    except InvalidVersion:
        built_field = (
            "build.built_with_core_package"
            if field == "compatible_core_packages"
            else "build.built_with_model_package"
        )
        failures.append(
            "release_manifest.json "
            f"{built_field}.version {built_version!r} is not a valid PEP 440 "
            "version."
        )
        return
    if not any(version in SpecifierSet(specifier) for specifier in matching_specifiers):
        failures.append(
            f"release_manifest.json '{field}' must include the built "
            f"{expected_name} version {built_version!r}."
        )


def _expected_model_package(release_id: str) -> str | None:
    if release_id.startswith("populace-us-"):
        return "policyengine-us"
    if release_id.startswith("populace-uk-"):
        return "policyengine-uk"
    return None


def _check_local_artifact_hashes(
    release_dir: Path,
    release_manifest: Mapping | None,
    failures: list[str],
) -> None:
    if release_manifest is None:
        return
    artifacts = release_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return
    for key, entry in artifacts.items():
        if not isinstance(entry, Mapping):
            continue
        path = entry.get("path")
        expected_sha = entry.get("sha256")
        if not isinstance(path, str) or not isinstance(expected_sha, str):
            continue
        local = release_dir / path
        if not local.is_file():
            continue
        observed_sha = _sha256(local)
        if observed_sha != expected_sha:
            failures.append(
                f"release_manifest.json artifact {key!r} declares sha256 "
                f"{expected_sha} for local file {path!r}, but observed "
                f"{observed_sha}."
            )


def _artifact_by_path(release_manifest: Mapping, path: str) -> Mapping | None:
    artifacts = release_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    for artifact in artifacts.values():
        if isinstance(artifact, Mapping) and artifact.get("path") == path:
            return artifact
    return None


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
                "measure",
                "filter",
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
            if not isinstance(target.get("measure"), Mapping):
                failures.append(
                    "calibration_diagnostics.json target row "
                    f"{index} is missing 'measure' selector object."
                )
            target_filter = target.get("filter")
            if target_filter is not None and not isinstance(target_filter, Mapping):
                failures.append(
                    "calibration_diagnostics.json target row "
                    f"{index} has non-null 'filter' that is not a selector object."
                )
            if not isinstance(target.get("metadata"), Mapping):
                failures.append(
                    "calibration_diagnostics.json target row "
                    f"{index} is missing 'metadata' object."
                )


def _check_us_critical_target_fit(diagnostics: Mapping, failures: list[str]) -> None:
    targets = diagnostics.get("targets")
    if not isinstance(targets, list):
        return
    incumbent_targets = _incumbent_critical_targets(diagnostics)
    for requirement in _US_CRITICAL_TARGET_FIT_REQUIREMENTS:
        matches = [
            target
            for target in targets
            if isinstance(target, Mapping)
            and not _is_congressional_district_layout_target(target)
            and requirement.matches(
                name=str(target.get("name") or ""),
                family=_target_registry_family(target),
                target_role=(
                    str(target["metadata"].get("target_role") or "")
                    if isinstance(target.get("metadata"), Mapping)
                    else ""
                ),
            )
        ]
        if not matches:
            failures.append(
                "calibration_diagnostics.json is missing required US critical "
                f"target {requirement.requirement_id!r} "
                f"({requirement.label})."
            )
            continue
        for target in matches:
            relative_error = target.get("relative_error")
            computed_relative_error = _target_relative_error(target, failures)
            if computed_relative_error is None:
                continue
            if not isinstance(relative_error, int | float):
                failures.append(
                    "calibration_diagnostics.json critical target "
                    f"{target.get('name')!r} has non-numeric relative_error "
                    f"{relative_error!r}."
                )
            elif not math.isclose(
                float(relative_error),
                computed_relative_error,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                failures.append(
                    "calibration_diagnostics.json critical target "
                    f"{target.get('name')!r} has stale relative_error "
                    f"{relative_error!r}; computed "
                    f"{computed_relative_error:.6g} from target and "
                    "final_estimate."
                )
            max_abs = float(requirement.max_abs_relative_error)
            if abs(computed_relative_error) > max_abs:
                incumbent_relative_error = _incumbent_relative_error_for_target(
                    target,
                    incumbent_targets.get(str(target.get("name"))),
                    failures,
                )
                improved_over_incumbent = incumbent_relative_error is not None and abs(
                    computed_relative_error
                ) < abs(incumbent_relative_error)
                if (
                    requirement.allow_incumbent_improvement
                    and improved_over_incumbent
                    and abs(computed_relative_error)
                    <= _US_CRITICAL_TARGET_IMPROVEMENT_MAX_ABS_RELATIVE_ERROR
                ):
                    continue
                failures.append(
                    "calibration_diagnostics.json critical target "
                    f"{target.get('name')!r} ({requirement.label}) has "
                    f"relative_error={computed_relative_error:.6g}, exceeding "
                    f"{max_abs:.6g}; target={target.get('target')!r}, "
                    f"final_estimate={target.get('final_estimate')!r}"
                    + (
                        "."
                        if incumbent_relative_error is None
                        else (
                            "; incumbent_relative_error="
                            f"{incumbent_relative_error:.6g}; "
                            "improvement_hard_stop="
                            f"{_US_CRITICAL_TARGET_IMPROVEMENT_MAX_ABS_RELATIVE_ERROR:.6g}."
                        )
                    )
                )


def _incumbent_critical_targets(diagnostics: Mapping) -> Mapping:
    build = diagnostics.get("build")
    if not isinstance(build, Mapping):
        return {}
    incumbent = build.get("incumbent_diagnostics")
    if not isinstance(incumbent, Mapping):
        return {}
    targets = incumbent.get("critical_targets")
    if not isinstance(targets, Mapping):
        return {}
    return targets


def _incumbent_relative_error_for_target(
    target: Mapping,
    incumbent: object,
    failures: list[str],
) -> float | None:
    if not isinstance(incumbent, Mapping):
        return None
    current_target = target.get("target")
    incumbent_target = incumbent.get("target")
    incumbent_final = incumbent.get("final_estimate")
    if not isinstance(current_target, int | float) or not isinstance(
        incumbent_target, int | float
    ):
        return None
    if not isinstance(incumbent_final, int | float):
        return None
    current_target = float(current_target)
    incumbent_target = float(incumbent_target)
    incumbent_final = float(incumbent_final)
    if not (
        math.isfinite(current_target)
        and math.isfinite(incumbent_target)
        and math.isfinite(incumbent_final)
    ):
        return None
    if not math.isclose(
        incumbent_target,
        current_target,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        failures.append(
            "calibration_diagnostics.json critical target "
            f"{target.get('name')!r} has incumbent target "
            f"{incumbent_target!r}, which does not match current target "
            f"{current_target!r}."
        )
        return None
    if current_target == 0.0:
        return incumbent_final - current_target
    return (incumbent_final - current_target) / current_target


def _target_relative_error(target: Mapping, failures: list[str]) -> float | None:
    target_value = target.get("target")
    final_estimate = target.get("final_estimate")
    if not isinstance(target_value, int | float) or not isinstance(
        final_estimate, int | float
    ):
        failures.append(
            "calibration_diagnostics.json critical target "
            f"{target.get('name')!r} has non-numeric target/final_estimate: "
            f"target={target_value!r}, final_estimate={final_estimate!r}."
        )
        return None
    target_value = float(target_value)
    final_estimate = float(final_estimate)
    if not math.isfinite(target_value) or not math.isfinite(final_estimate):
        failures.append(
            "calibration_diagnostics.json critical target "
            f"{target.get('name')!r} has non-finite target/final_estimate: "
            f"target={target_value!r}, final_estimate={final_estimate!r}."
        )
        return None
    if target_value == 0.0:
        return final_estimate - target_value
    return (final_estimate - target_value) / target_value


def _target_registry_family(target: Mapping) -> str:
    registry = target.get("registry")
    if isinstance(registry, Mapping):
        family = registry.get("family")
        if family is not None:
            return str(family)
    return ""


def _is_congressional_district_layout_target(target: Mapping) -> bool:
    metadata = target.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    groupby_dimension = metadata.get("ledger_layout_groupby_dimension")
    if str(groupby_dimension) == "irs_soi.congressional_district":
        return True
    source_record_id = metadata.get("ledger_source_record_id")
    if (
        isinstance(source_record_id, str)
        and ".congressional_district_" in source_record_id
    ):
        return True
    name = target.get("name")
    return isinstance(name, str) and ".congressional_district_" in name


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
        ledger_commit = source_contract.get("ledger_commit")
        if not isinstance(ledger_commit, str) or len(ledger_commit) != 40:
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} source_contract.ledger_commit "
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
        "fiscal_target_sources": Mapping,
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

    fiscal_sources = diagnostics.get("fiscal_target_sources")
    if isinstance(fiscal_sources, Mapping):
        for family, source in fiscal_sources.items():
            if not isinstance(source, Mapping):
                failures.append(
                    f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} "
                    f"fiscal_target_sources[{family!r}] must be an object."
                )
                continue
            target_count = source.get("target_count")
            if not isinstance(target_count, int) or target_count <= 0:
                failures.append(
                    f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} "
                    f"fiscal_target_sources[{family!r}].target_count must be > 0."
                )
            sources = source.get("sources")
            if (
                not isinstance(sources, list)
                or not sources
                or any(not isinstance(item, str) or not item for item in sources)
            ):
                failures.append(
                    f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} "
                    f"fiscal_target_sources[{family!r}].sources must be a "
                    "non-empty list of strings."
                )
            reference_urls = source.get("reference_urls")
            if not isinstance(reference_urls, list) or any(
                not isinstance(item, str) or not item for item in reference_urls
            ):
                failures.append(
                    f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} "
                    f"fiscal_target_sources[{family!r}].reference_urls must "
                    "be a list of strings."
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
    source_coverage_diagnostics: Mapping | None = None

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
            if release_id.startswith("populace-us-"):
                _check_us_critical_target_fit(diagnostics, failures)

    _check_cross_manifest_consistency(
        build_manifest,
        release_manifest,
        calibration_diagnostics,
        failures,
    )
    _check_local_artifact_hashes(release_dir, release_manifest, failures)

    source_coverage_path = release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE
    if release_id.startswith("populace-us-") and source_coverage_path.is_file():
        diagnostics = _load_json(source_coverage_path, failures)
        if diagnostics is not None:
            source_coverage_diagnostics = diagnostics
            _check_source_coverage_diagnostics(diagnostics, failures)

    _check_us_fiscal_source_consistency(
        calibration_diagnostics, source_coverage_diagnostics, failures
    )

    if failures:
        raise ReleaseContractError(release_dir, failures)


def _check_us_fiscal_source_consistency(
    calibration_diagnostics: Mapping | None,
    source_coverage_diagnostics: Mapping | None,
    failures: list[str],
) -> None:
    if calibration_diagnostics is None or source_coverage_diagnostics is None:
        return
    targets = calibration_diagnostics.get("targets")
    fiscal_sources = source_coverage_diagnostics.get("fiscal_target_sources")
    if not isinstance(targets, list) or not isinstance(fiscal_sources, Mapping):
        return
    calibrated_family_counts: dict[str, int] = {}
    for family in (
        registry.get("family")
        for target in targets
        if isinstance(target, Mapping)
        for registry in (target.get("registry"),)
        if isinstance(registry, Mapping) and registry.get("family")
    ):
        calibrated_family_counts[str(family)] = (
            calibrated_family_counts.get(str(family), 0) + 1
        )
    calibrated_families = set(calibrated_family_counts)
    missing = sorted(
        str(family) for family in calibrated_families - fiscal_sources.keys()
    )
    if missing:
        failures.append(
            f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} fiscal_target_sources must "
            f"cover every calibrated target family; missing {missing}."
        )
    unexpected = sorted(
        str(family) for family in fiscal_sources.keys() - calibrated_families
    )
    if unexpected:
        failures.append(
            f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} fiscal_target_sources must "
            f"only describe calibrated target families; unexpected {unexpected}."
        )
    for family, expected_count in sorted(calibrated_family_counts.items()):
        source = fiscal_sources.get(family)
        if not isinstance(source, Mapping):
            continue
        target_count = source.get("target_count")
        if target_count != expected_count:
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} "
                f"fiscal_target_sources[{family!r}].target_count is "
                f"{target_count!r} but calibration_diagnostics.json has "
                f"{expected_count} calibrated target(s) for that family."
            )


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

    if build_manifest is not None and release_manifest is not None:
        dataset = build_manifest.get("dataset")
        if isinstance(dataset, Mapping):
            _check_root_artifact_matches_build_manifest(
                release_manifest,
                path=dataset.get("filename"),
                sha256=dataset.get("sha256"),
                description="dataset",
                failures=failures,
            )
            _check_default_dataset_matches_build_manifest(
                release_manifest,
                path=dataset.get("filename"),
                sha256=dataset.get("sha256"),
                failures=failures,
            )
        calibration = build_manifest.get("calibration")
        if isinstance(calibration, Mapping):
            _check_root_artifact_matches_build_manifest(
                release_manifest,
                path=calibration.get("filename"),
                sha256=calibration.get("sha256"),
                description="calibration",
                failures=failures,
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


def _check_default_dataset_matches_build_manifest(
    release_manifest: Mapping,
    *,
    path: object,
    sha256: object,
    failures: list[str],
) -> None:
    default_datasets = release_manifest.get("default_datasets")
    artifacts = release_manifest.get("artifacts")
    if not isinstance(default_datasets, Mapping) or not isinstance(artifacts, Mapping):
        return
    default_key = default_datasets.get("national")
    if not isinstance(default_key, str):
        return
    default_artifact = artifacts.get(default_key)
    if not isinstance(default_artifact, Mapping):
        return
    if default_artifact.get("path") != path:
        failures.append(
            "release_manifest.json 'default_datasets.national' must point to "
            "the dataset root artifact declared by build_manifest.json."
        )
    if default_artifact.get("sha256") != sha256:
        failures.append(
            "release_manifest.json default dataset artifact must have sha256 "
            "matching build_manifest.json."
        )


def _check_root_artifact_matches_build_manifest(
    release_manifest: Mapping,
    *,
    path: object,
    sha256: object,
    description: str,
    failures: list[str],
) -> None:
    if not isinstance(path, str) or not path:
        return
    artifact = _artifact_by_path(release_manifest, path)
    if artifact is None:
        failures.append(
            f"release_manifest.json artifacts must include the {description} "
            f"root artifact {path!r} declared by build_manifest.json."
        )
        return
    if artifact.get("sha256") != sha256:
        failures.append(
            f"release_manifest.json artifact for {description} root artifact "
            f"{path!r} must have sha256 matching build_manifest.json."
        )
