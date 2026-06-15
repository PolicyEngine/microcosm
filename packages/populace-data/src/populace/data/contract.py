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

CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION = 1
US_SOURCE_COVERAGE_DIAGNOSTICS_FILE = "us_source_coverage.json"
SOURCE_COVERAGE_DIAGNOSTICS_SCHEMA_VERSION = 1


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
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        failures.append("build_manifest.json is missing the 'dataset' object.")
    else:
        for key in ("filename", "sha256"):
            if not dataset.get(key):
                failures.append(f"build_manifest.json 'dataset' is missing {key!r}.")
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

    targets = diagnostics.get("targets")
    if isinstance(targets, list):
        for index, target in enumerate(targets):
            if not isinstance(target, Mapping):
                failures.append(
                    f"calibration_diagnostics.json target row {index} must be an object."
                )
                continue
            for field in ("name", "target", "initial_estimate", "final_estimate"):
                if field not in target:
                    failures.append(
                        "calibration_diagnostics.json target row "
                        f"{index} is missing {field!r}."
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

    for filename in required_release_files(release_id):
        if not (release_dir / filename).is_file():
            failures.append(f"required file {filename!r} is missing.")

    build_manifest_path = release_dir / "build_manifest.json"
    if build_manifest_path.is_file():
        manifest = _load_json(build_manifest_path, failures)
        if manifest is not None:
            _check_build_manifest(manifest, release_id, failures)

    release_manifest_path = release_dir / "release_manifest.json"
    if release_manifest_path.is_file():
        manifest = _load_json(release_manifest_path, failures)
        if manifest is not None:
            _check_release_manifest(manifest, release_id, failures)

    calibration_diagnostics_path = release_dir / "calibration_diagnostics.json"
    if calibration_diagnostics_path.is_file():
        diagnostics = _load_json(calibration_diagnostics_path, failures)
        if diagnostics is not None:
            _check_calibration_diagnostics(diagnostics, failures)

    source_coverage_path = release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE
    if release_id.startswith("populace-us-") and source_coverage_path.is_file():
        diagnostics = _load_json(source_coverage_path, failures)
        if diagnostics is not None:
            _check_source_coverage_diagnostics(diagnostics, failures)

    if failures:
        raise ReleaseContractError(release_dir, failures)
