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
from populace.data.us_critical_targets import (
    is_congressional_district_target,
)

__all__ = [
    "LOCAL_AREA_REQUIRED_RELEASE_FILES",
    "NATIONAL_DEFAULT_DATASET_ROLE",
    "NON_DEFAULT_LOCAL_AREA_DATASET_ROLE",
    "RELEASE_MANIFEST_SCHEMA_VERSION",
    "REQUIRED_RELEASE_FILES",
    "US_SOURCE_COVERAGE_DIAGNOSTICS_FILE",
    "ReleaseContractError",
    "release_dataset_role",
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

# Dataset-role classes (populace#398). The national default keeps the full
# historical contract; a non-default local-area artifact gets its own
# contract and can never move the latest.json pointer.
NATIONAL_DEFAULT_DATASET_ROLE = "national_default"
NON_DEFAULT_LOCAL_AREA_DATASET_ROLE = "non_default_local_area"

#: Files a non-default local-area release directory must carry. The source
#: coverage entry matches :data:`US_SOURCE_COVERAGE_DIAGNOSTICS_FILE`.
LOCAL_AREA_REQUIRED_RELEASE_FILES = (
    "build_manifest.json",
    "release_manifest.json",
    "calibration_diagnostics.json",
    "gate_summary.json",
    "us_source_coverage.json",
    "sha256sums.txt",
)

#: Provenance-chain keys the local-area source coverage must carry (the
#: local product's analog of the national fiscal_target_sources map).
LOCAL_AREA_SOURCE_COVERAGE_KEYS = (
    "acs_sources",
    "geography_ladder",
    "transfer_coverage",
    "donor_release",
)

# Lockstep with populace.calibrate.diagnostics.CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION
# (schema 5 = the #492 past_cap_census block). populace-data cannot import
# populace-calibrate (dependency direction), so the builder test suite pins the
# two constants equal — see test_calibration_diagnostics_schema_lockstep.
CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION = 5
US_SOURCE_COVERAGE_DIAGNOSTICS_FILE = "us_source_coverage.json"
SOURCE_COVERAGE_DIAGNOSTICS_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_UK_EXACT_K_RELEASE_ID_RE = re.compile(
    r"^populace-uk-(?P<year>[1-9][0-9]*)-(?P<tier>.+)-k(?P<record_count>[1-9][0-9]*)$"
)
_UK_JUNE_RELEASE_ID = "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z"
_UK_LEGACY_RELEASE_IDS = frozenset({_UK_JUNE_RELEASE_ID})
_UK_RELEASE_TIERS = frozenset({"frs", "cps-transfer"})
_UK_DIAGNOSTICS_SCHEMA_VERSION = 1
_UK_TERMINAL_GATE_REPORT_FILE = "terminal_gates.json"
_UK_TERMINAL_GATE_SCHEMA_VERSION = 2
_UK_TERMINAL_GATE_ATTESTATION_SCHEMA_VERSION = 2
_UK_TERMINAL_GATE_PRODUCER = (
    "populace.build.uk_runtime.terminal_gates.uk_terminal_gate_report"
)
# Lockstep with
# populace.build.uk_runtime.terminal_gates.UK_TERMINAL_GATE_POLICY_SHA256.  The
# data shard deliberately does not depend on the build shard: publication must
# independently pin the reviewed gate policy rather than trust a producer's
# self-description.
_UK_TERMINAL_GATE_POLICY_SHA256 = (
    "7404db805d5fdb8ff389e87a6dcca0378a88636ba62bdb1fb81ba963d2d78cd8"
)
_UK_ALWAYS_APPLICABLE_GATE_NAMES = (
    "uk_release_input_coverage",
    "degenerate_release_surface",
    "zero_weight_strata",
    "weight_ess",
    "weight_ratio",
)
_UK_TERMINAL_EVIDENCE_GATE_NAMES = {
    "hmrc_spi_income": ("weights_audit",),
    "release_parity": ("export_surface", "target_surface", "target_fit"),
}
_UK_TERMINAL_EVIDENCE_STAGES = frozenset(
    {"release_dataset", *_UK_TERMINAL_EVIDENCE_GATE_NAMES}
)
_UK_WEIGHT_SUMMARY_FIELDS = (
    "n_records",
    "positive_weight_records",
    "zero_weight_records",
    "total_weight",
    "effective_sample_size",
    "ess_fraction",
    "median_positive_weight",
    "max_weight",
    "max_to_median_positive_weight",
    "top_1pct_weight_share",
)
_UK_MIN_ESS_FRACTION = 0.01
_UK_MAX_TO_MEDIAN_WEIGHT_RATIO = 1_151.2542195939373
_UK_MAX_TARGET_ABS_RELATIVE_ERROR = 0.25
_UK_TERMINAL_GATE_DETAIL_FIELDS = {
    "uk_release_input_coverage": frozenset(
        {
            "required_columns",
            "present_columns",
            "missing",
            "degenerate_required",
            "reviewed_exclusions",
            "stale_exclusions",
            "dormant_exclusions",
        }
    ),
    "degenerate_release_surface": frozenset(
        {
            "columns_checked",
            "findings",
            "all_null_columns",
            "all_zero_columns",
            "constant_columns",
            "reviewed_exclusions",
            "stale_exclusions",
            "dormant_exclusions",
        }
    ),
    "zero_weight_strata": frozenset(
        {
            "household_rows",
            "zero_weight_rows",
            "declared_strata",
            "unmatched_zero_weight_rows",
            "unmatched_household_examples",
            "ambiguous_zero_weight_rows",
            "ambiguous_household_examples",
        }
    ),
    "weight_ess": frozenset({*_UK_WEIGHT_SUMMARY_FIELDS, "minimum_ess_fraction"}),
    "weight_ratio": frozenset(
        {*_UK_WEIGHT_SUMMARY_FIELDS, "maximum_max_to_median_ratio"}
    ),
    "weights_audit": frozenset(
        {
            "fits_checked",
            "resolved_weight_kinds",
            "unweighted_fits",
            "allowed_unweighted",
            "unused_allowed_unweighted",
        }
    ),
    "export_surface": frozenset(
        {
            "candidate_columns",
            "reference_columns",
            "missing_reference_columns",
            "unexpected_candidate_columns",
            "forbidden_candidate_columns",
        }
    ),
    "target_surface": frozenset(
        {
            "candidate_targets",
            "reference_targets",
            "extra_candidate_targets",
            "missing_reference_targets",
        }
    ),
    "target_fit": frozenset(
        {
            "targets_checked",
            "max_abs_relative_error",
            "failing_targets",
        }
    ),
}
_UK_TARGET_GEOGRAPHY_LEVELS = frozenset(
    {"national", "region", "country", "local_authority", "constituency"}
)


def required_release_files(release_id: str) -> tuple[str, ...]:
    """Files required for a release id's country-specific contract."""
    if release_id.startswith("populace-us-"):
        return (*REQUIRED_RELEASE_FILES, US_SOURCE_COVERAGE_DIAGNOSTICS_FILE)
    if _is_uk_exact_k_release_id(release_id):
        return (*REQUIRED_RELEASE_FILES, _UK_TERMINAL_GATE_REPORT_FILE)
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


def _canonical_sha256(value: object) -> str:
    """Hash JSON exactly as the UK gate aggregator's attestation does."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    gates = manifest.get("gates")
    if not isinstance(gates, Mapping):
        failures.append(
            "build_manifest.json is missing the 'gates' object (the "
            "acceptance-gate verdicts are the point of the manifest)."
        )
    if _is_uk_exact_k_release_id(release_id):
        _check_uk_terminal_build_manifest(manifest, failures)
    _check_uk_exact_k_manifest_fields(
        manifest,
        release_id,
        filename="build_manifest.json",
        count_fields=("n_records",),
        failures=failures,
    )


def _check_uk_terminal_build_manifest(
    manifest: Mapping,
    failures: list[str],
) -> None:
    """Require exact-k UK builds to bind gate evidence and the report bytes."""

    evidence = manifest.get("terminal_gate_evidence")
    if not isinstance(evidence, Mapping):
        failures.append(
            "build_manifest.json canonical UK releases require a "
            "'terminal_gate_evidence' object."
        )
    else:
        stages = set(evidence)
        missing = sorted({"release_dataset"} - stages)
        unexpected = sorted(
            str(stage) for stage in stages - _UK_TERMINAL_EVIDENCE_STAGES
        )
        if missing:
            failures.append(
                "build_manifest.json terminal_gate_evidence is missing "
                f"always-applicable stage(s): {missing}."
            )
        if unexpected:
            failures.append(
                "build_manifest.json terminal_gate_evidence has unknown "
                f"stage(s): {unexpected}."
            )
        for stage, digest in evidence.items():
            _check_sha256_field(
                filename="build_manifest.json",
                owner=f"terminal_gate_evidence[{stage!r}]",
                value=digest,
                failures=failures,
            )

    gates = manifest.get("gates")
    terminal = gates.get("uk_terminal") if isinstance(gates, Mapping) else None
    if not isinstance(terminal, Mapping):
        failures.append(
            "build_manifest.json canonical UK gates must include an "
            "'uk_terminal' report pointer."
        )
        return
    if terminal.get("passed") is not True:
        failures.append("build_manifest.json gates.uk_terminal.passed must be true.")
    if terminal.get("path") != _UK_TERMINAL_GATE_REPORT_FILE:
        failures.append(
            "build_manifest.json gates.uk_terminal.path must be "
            f"{_UK_TERMINAL_GATE_REPORT_FILE!r}."
        )
    _check_sha256_field(
        filename="build_manifest.json",
        owner="gates.uk_terminal.sha256",
        value=terminal.get("sha256"),
        failures=failures,
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
    _check_uk_release_identity(manifest, release_id, failures)
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
        if _is_uk_exact_k_release_id(release_id):
            terminal_artifact = _artifact_by_path(
                manifest, _UK_TERMINAL_GATE_REPORT_FILE
            )
            if terminal_artifact is None:
                failures.append(
                    "release_manifest.json artifacts must include "
                    f"{_UK_TERMINAL_GATE_REPORT_FILE!r} for canonical UK "
                    "releases."
                )
            elif terminal_artifact.get("kind") != "diagnostics":
                failures.append(
                    "release_manifest.json terminal gate report artifact "
                    "must have kind 'diagnostics'."
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


def _check_uk_release_identity(
    manifest: Mapping,
    release_id: str,
    failures: list[str],
) -> None:
    """Enforce canonical UK identity while narrowly grandfathering old ids."""

    if not release_id.startswith("populace-uk-"):
        return

    match = _UK_EXACT_K_RELEASE_ID_RE.fullmatch(release_id)
    if match is None:
        if release_id not in _UK_LEGACY_RELEASE_IDS:
            failures.append(
                "release_manifest.json UK release id is neither canonical "
                "'populace-uk-<year>-<tier>-k<N>' nor the grandfathered June "
                f"hash/timestamp id: {release_id!r}."
            )
            return
        # This one known artifact predates the tier contract and does not need
        # a re-cut. Its recorded construction is FRS-derived, so an added tier
        # may only make that lineage explicit; it cannot relabel the artifact.
        legacy_tier = manifest.get("tier")
        if legacy_tier not in (None, "frs"):
            failures.append(
                "release_manifest.json grandfathered UK 'tier', when present, "
                f"must be 'frs' for the known FRS lineage, got {legacy_tier!r}."
            )
        return

    release_id_tier = match.group("tier")
    if release_id_tier not in _UK_RELEASE_TIERS:
        failures.append(
            "release_manifest.json exact-k UK release id has unratified tier "
            f"{release_id_tier!r}; expected one of {sorted(_UK_RELEASE_TIERS)}."
        )

    manifest_tier = manifest.get("tier")
    if manifest_tier is None:
        failures.append(
            "release_manifest.json exact-k UK releases require top-level 'tier'."
        )
    elif not isinstance(manifest_tier, str) or manifest_tier not in _UK_RELEASE_TIERS:
        failures.append(
            "release_manifest.json exact-k UK 'tier' must be one of "
            f"{sorted(_UK_RELEASE_TIERS)}, got {manifest_tier!r}."
        )
    if isinstance(manifest_tier, str) and manifest_tier != release_id_tier:
        failures.append(
            f"release_manifest.json top-level 'tier' is {manifest_tier!r} but "
            f"the release id names tier {release_id_tier!r}."
        )
    _check_uk_exact_k_manifest_fields(
        manifest,
        release_id,
        filename="release_manifest.json",
        count_fields=("record_count", "n_records"),
        failures=failures,
    )


def _is_uk_exact_k_release_id(release_id: str) -> bool:
    return _UK_EXACT_K_RELEASE_ID_RE.fullmatch(release_id) is not None


def _check_uk_exact_k_manifest_fields(
    manifest: Mapping,
    release_id: str,
    *,
    filename: str,
    count_fields: tuple[str, ...],
    failures: list[str],
) -> None:
    """Bind canonical UK manifest identity fields to the exact-k id."""

    match = _UK_EXACT_K_RELEASE_ID_RE.fullmatch(release_id)
    if match is None:
        return
    expected_year = int(match.group("year"))
    expected_records = int(match.group("record_count"))
    if manifest.get("country") != "uk":
        failures.append(
            f"{filename} canonical UK 'country' must be 'uk', got "
            f"{manifest.get('country')!r}."
        )
    year = manifest.get("year")
    if isinstance(year, bool) or not isinstance(year, int) or year != expected_year:
        failures.append(
            f"{filename} canonical UK 'year' must equal release-id year "
            f"{expected_year}, got {year!r}."
        )
    for field in count_fields:
        value = manifest.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value != expected_records
        ):
            failures.append(
                f"{filename} canonical UK {field!r} must equal release-id "
                f"record count {expected_records}, got {value!r}."
            )


def _check_uk_exact_k_diagnostics_identity(
    diagnostics: Mapping,
    release_id: str,
    failures: list[str],
) -> None:
    """Reconcile the exact-k id with every shipped-record diagnostic."""

    match = _UK_EXACT_K_RELEASE_ID_RE.fullmatch(release_id)
    if match is None:
        return
    expected = int(match.group("record_count"))
    observations: list[tuple[str, object]] = [
        ("top-level n_records", diagnostics.get("n_records")),
    ]
    uk = diagnostics.get("uk_diagnostics")
    if isinstance(uk, Mapping):
        weights = uk.get("weights")
        if isinstance(weights, Mapping):
            observations.append(
                ("uk_diagnostics.weights.n_records", weights.get("n_records"))
            )
    target_surface = diagnostics.get("target_surface")
    if isinstance(target_surface, Mapping) and "n_records" in target_surface:
        observations.append(("target_surface.n_records", target_surface["n_records"]))
    for field, value in observations:
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            failures.append(
                "calibration_diagnostics.json canonical UK "
                f"{field} must equal release-id record count {expected}, got "
                f"{value!r}."
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


def _check_uk_terminal_gate_links(
    *,
    build_manifest: Mapping | None,
    release_manifest: Mapping | None,
    report_sha256: str,
    failures: list[str],
) -> None:
    """Cross-link both manifests to the exact terminal-report bytes."""

    build_terminal: object = None
    if build_manifest is not None:
        gates = build_manifest.get("gates")
        if isinstance(gates, Mapping):
            build_terminal = gates.get("uk_terminal")
    build_sha: object = None
    if isinstance(build_terminal, Mapping):
        build_sha = build_terminal.get("sha256")
        if build_sha != report_sha256:
            failures.append(
                "build_manifest.json gates.uk_terminal.sha256 must match the "
                f"local {_UK_TERMINAL_GATE_REPORT_FILE} bytes."
            )

    release_artifact: Mapping | None = None
    if release_manifest is not None:
        release_artifact = _artifact_by_path(
            release_manifest, _UK_TERMINAL_GATE_REPORT_FILE
        )
    if release_artifact is not None:
        release_sha = release_artifact.get("sha256")
        if release_sha != report_sha256:
            failures.append(
                "release_manifest.json terminal gate report artifact sha256 "
                f"must match the local {_UK_TERMINAL_GATE_REPORT_FILE} bytes."
            )
        if (
            isinstance(build_sha, str)
            and isinstance(release_sha, str)
            and (build_sha != release_sha)
        ):
            failures.append(
                "build_manifest.json and release_manifest.json terminal gate "
                "report sha256 values must match."
            )


def _uk_terminal_observable_matches(observed: object, expected: object) -> bool:
    """Compare JSON observables without letting booleans masquerade as counts."""

    if isinstance(observed, bool) or isinstance(expected, bool):
        return type(observed) is type(expected) and observed == expected
    if isinstance(observed, int | float) and isinstance(expected, int | float):
        return (
            math.isfinite(float(observed))
            and math.isfinite(float(expected))
            and (
                math.isclose(
                    float(observed),
                    float(expected),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        )
    return observed == expected


def _uk_terminal_gate_details(
    gates: Mapping,
    name: str,
) -> Mapping | None:
    gate = gates.get(name)
    if not isinstance(gate, Mapping):
        return None
    details = gate.get("details")
    return details if isinstance(details, Mapping) else None


def _check_uk_terminal_gate_observables(
    gates: Mapping,
    *,
    calibration_diagnostics: Mapping | None,
    failures: list[str],
) -> None:
    """Bind passing gate detail schemas to the accepted release diagnostics."""

    for name, required_fields in _UK_TERMINAL_GATE_DETAIL_FIELDS.items():
        if name not in gates:
            continue
        details = _uk_terminal_gate_details(gates, name)
        if details is None:
            continue
        missing = sorted(required_fields - set(details))
        if missing:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} gate {name!r}.details is not "
                f"the honest aggregator detail schema; missing {missing}."
            )

    coverage = _uk_terminal_gate_details(gates, "uk_release_input_coverage")
    if coverage is not None:
        required = coverage.get("required_columns")
        present = coverage.get("present_columns")
        if (
            isinstance(required, bool)
            or not isinstance(required, int)
            or required <= 0
            or isinstance(present, bool)
            or not isinstance(present, int)
            or present < required
        ):
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} input-coverage details must "
                "record positive required_columns covered by present_columns."
            )
        for field in ("missing", "degenerate_required", "stale_exclusions"):
            if coverage.get(field) != []:
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} passing input coverage "
                    f"requires details.{field} to be an empty list."
                )

    degenerate = _uk_terminal_gate_details(gates, "degenerate_release_surface")
    if degenerate is not None:
        checked = degenerate.get("columns_checked")
        if isinstance(checked, bool) or not isinstance(checked, int) or checked <= 0:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} degenerate-surface details "
                "must record a positive columns_checked count."
            )
        for field, empty in (
            ("findings", {}),
            ("all_null_columns", []),
            ("all_zero_columns", []),
            ("constant_columns", []),
            ("stale_exclusions", []),
        ):
            if degenerate.get(field) != empty:
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} passing degenerate-surface "
                    f"gate requires details.{field} to be {empty!r}."
                )

    diagnostic_weights: Mapping | None = None
    if calibration_diagnostics is not None:
        uk = calibration_diagnostics.get("uk_diagnostics")
        if isinstance(uk, Mapping) and isinstance(uk.get("weights"), Mapping):
            diagnostic_weights = uk["weights"]

    for name in ("weight_ess", "weight_ratio"):
        details = _uk_terminal_gate_details(gates, name)
        if details is None or diagnostic_weights is None:
            continue
        for field in _UK_WEIGHT_SUMMARY_FIELDS:
            if field not in details or field not in diagnostic_weights:
                continue
            if not _uk_terminal_observable_matches(
                details[field], diagnostic_weights[field]
            ):
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} gate {name!r}.details.{field} "
                    "must match calibration_diagnostics.json "
                    f"uk_diagnostics.weights.{field}."
                )

    ess = _uk_terminal_gate_details(gates, "weight_ess")
    if ess is not None and not _uk_terminal_observable_matches(
        ess.get("minimum_ess_fraction"), _UK_MIN_ESS_FRACTION
    ):
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} weight_ess.details."
            f"minimum_ess_fraction must equal {_UK_MIN_ESS_FRACTION}."
        )
    ratio = _uk_terminal_gate_details(gates, "weight_ratio")
    if ratio is not None and not _uk_terminal_observable_matches(
        ratio.get("maximum_max_to_median_ratio"),
        _UK_MAX_TO_MEDIAN_WEIGHT_RATIO,
    ):
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} weight_ratio.details."
            "maximum_max_to_median_ratio must equal the certified June bound "
            f"{_UK_MAX_TO_MEDIAN_WEIGHT_RATIO}."
        )

    zero = _uk_terminal_gate_details(gates, "zero_weight_strata")
    if zero is not None and diagnostic_weights is not None:
        for terminal_field, diagnostic_field in (
            ("household_rows", "n_records"),
            ("zero_weight_rows", "zero_weight_records"),
        ):
            if not _uk_terminal_observable_matches(
                zero.get(terminal_field), diagnostic_weights.get(diagnostic_field)
            ):
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} zero_weight_strata.details."
                    f"{terminal_field} must match calibration_diagnostics.json "
                    f"uk_diagnostics.weights.{diagnostic_field}."
                )
        for field in ("unmatched_zero_weight_rows", "ambiguous_zero_weight_rows"):
            if zero.get(field) != 0:
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} passing zero-weight gate "
                    f"requires details.{field} to be zero."
                )
        declared = zero.get("declared_strata")
        if not isinstance(declared, list) or not declared:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} zero_weight_strata.details."
                "declared_strata must be a non-empty list."
            )
        elif all(isinstance(row, Mapping) for row in declared):
            declared_zero = sum(
                row.get("zero_weight_rows", 0)
                for row in declared
                if isinstance(row.get("zero_weight_rows"), int)
                and not isinstance(row.get("zero_weight_rows"), bool)
            )
            if declared_zero != zero.get("zero_weight_rows"):
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} declared zero-weight "
                    "strata must reconcile to details.zero_weight_rows."
                )

    weights_audit = _uk_terminal_gate_details(gates, "weights_audit")
    if weights_audit is not None:
        fits = weights_audit.get("fits_checked")
        kinds = weights_audit.get("resolved_weight_kinds")
        if (
            isinstance(fits, bool)
            or not isinstance(fits, int)
            or fits <= 0
            or not isinstance(kinds, Mapping)
            or len(kinds) != fits
        ):
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing weights_audit details "
                "must enumerate at least one resolved production fit."
            )
        if weights_audit.get("unweighted_fits") != []:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing weights_audit details "
                "must not contain unweighted fits."
            )

    export = _uk_terminal_gate_details(gates, "export_surface")
    if export is not None:
        for field in ("candidate_columns", "reference_columns"):
            value = export.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} export_surface.details."
                    f"{field} must be a positive integer."
                )
        for field in (
            "missing_reference_columns",
            "unexpected_candidate_columns",
            "forbidden_candidate_columns",
        ):
            if export.get(field) != []:
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} passing export surface "
                    f"requires details.{field} to be an empty list."
                )

    target_count = None
    if calibration_diagnostics is not None and isinstance(
        calibration_diagnostics.get("targets"), list
    ):
        target_count = len(calibration_diagnostics["targets"])
    target_surface = _uk_terminal_gate_details(gates, "target_surface")
    if target_surface is not None:
        if target_surface.get("candidate_targets") != target_count:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} target_surface.details."
                "candidate_targets must match len(calibration diagnostics targets)."
            )
        reference_targets = target_surface.get("reference_targets")
        if (
            isinstance(reference_targets, bool)
            or not isinstance(reference_targets, int)
            or reference_targets <= 0
        ):
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} target_surface.details."
                "reference_targets must be a positive integer."
            )
        if target_surface.get("missing_reference_targets") != []:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing target surface requires "
                "details.missing_reference_targets to be an empty list."
            )

    target_fit = _uk_terminal_gate_details(gates, "target_fit")
    if target_fit is not None:
        if target_fit.get("targets_checked") != target_count:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} target_fit.details."
                "targets_checked must match len(calibration diagnostics targets)."
            )
        if not _uk_terminal_observable_matches(
            target_fit.get("max_abs_relative_error"),
            _UK_MAX_TARGET_ABS_RELATIVE_ERROR,
        ):
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} target_fit.details."
                "max_abs_relative_error must equal "
                f"{_UK_MAX_TARGET_ABS_RELATIVE_ERROR}."
            )
        if target_fit.get("failing_targets") != {}:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing target fit requires "
                "details.failing_targets to be an empty object."
            )


def _check_uk_terminal_gate_report(
    report: Mapping,
    *,
    build_manifest: Mapping | None,
    calibration_diagnostics: Mapping | None,
    failures: list[str],
) -> None:
    """Independently verify the exact-k UK terminal-gate attestation.

    The gate producer is not imported into populace-data.  This verifier pins
    its producer and policy identities, derives the only permissible gate set
    from build-manifest evidence stages, and recomputes the two attestation
    digests.  A caller therefore cannot promote a hand-composed collection of
    passing ``GateResult`` objects as the canonical terminal verdict.
    """

    required_report_fields = {
        "schema_version",
        "enforced",
        "passed",
        "gates",
        "attestation",
    }
    if set(report) != required_report_fields:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} must contain exactly "
            f"{sorted(required_report_fields)}, got {sorted(map(str, report))}."
        )
    if report.get("schema_version") != _UK_TERMINAL_GATE_SCHEMA_VERSION:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} schema_version must be "
            f"{_UK_TERMINAL_GATE_SCHEMA_VERSION}."
        )
    if report.get("enforced") is not True:
        failures.append(f"{_UK_TERMINAL_GATE_REPORT_FILE} enforced must be true.")
    if report.get("passed") is not True:
        failures.append(f"{_UK_TERMINAL_GATE_REPORT_FILE} passed must be true.")

    build_evidence: Mapping = {}
    if build_manifest is not None and isinstance(
        build_manifest.get("terminal_gate_evidence"), Mapping
    ):
        build_evidence = build_manifest["terminal_gate_evidence"]
    expected_gate_names = list(_UK_ALWAYS_APPLICABLE_GATE_NAMES)
    for stage, stage_gate_names in _UK_TERMINAL_EVIDENCE_GATE_NAMES.items():
        if stage in build_evidence:
            expected_gate_names.extend(stage_gate_names)
    expected_gate_set = set(expected_gate_names)

    gates = report.get("gates")
    valid_gates: Mapping = {}
    if not isinstance(gates, Mapping):
        failures.append(f"{_UK_TERMINAL_GATE_REPORT_FILE} gates must be an object.")
    else:
        valid_gates = gates
        actual_gate_set = set(gates)
        if actual_gate_set != expected_gate_set:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} evaluated gate membership "
                "must be derived from build_manifest.json evidence stages; "
                f"expected {expected_gate_names}, got {sorted(map(str, gates))}."
            )
        for name, gate in gates.items():
            owner = f"{_UK_TERMINAL_GATE_REPORT_FILE} gate {name!r}"
            if not isinstance(gate, Mapping):
                failures.append(f"{owner} must be an object.")
                continue
            if set(gate) != {"passed", "failures", "details"}:
                failures.append(
                    f"{owner} must contain exactly passed, failures, and details."
                )
            if gate.get("passed") is not True:
                failures.append(f"{owner}.passed must be true.")
            if gate.get("failures") != []:
                failures.append(f"{owner}.failures must be an empty list.")
            if not isinstance(gate.get("details"), Mapping):
                failures.append(f"{owner}.details must be an object.")

    _check_uk_terminal_gate_observables(
        valid_gates,
        calibration_diagnostics=calibration_diagnostics,
        failures=failures,
    )

    attestation = report.get("attestation")
    if not isinstance(attestation, Mapping):
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation must be an object."
        )
        return
    unsigned_fields = {
        "schema_version",
        "producer",
        "policy_sha256",
        "evaluated_gates",
        "evidence_sha256",
        "gate_results_sha256",
    }
    required_attestation_fields = {*unsigned_fields, "sha256"}
    if set(attestation) != required_attestation_fields:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation must contain exactly "
            f"{sorted(required_attestation_fields)}."
        )
    if (
        attestation.get("schema_version")
        != _UK_TERMINAL_GATE_ATTESTATION_SCHEMA_VERSION
    ):
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.schema_version must be "
            f"{_UK_TERMINAL_GATE_ATTESTATION_SCHEMA_VERSION}."
        )
    if attestation.get("producer") != _UK_TERMINAL_GATE_PRODUCER:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.producer must name the "
            "honest UK terminal gate aggregator."
        )
    if attestation.get("policy_sha256") != _UK_TERMINAL_GATE_POLICY_SHA256:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.policy_sha256 does "
            "not match the certified UK gate policy."
        )

    evaluated_gates = attestation.get("evaluated_gates")
    if evaluated_gates != expected_gate_names:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.evaluated_gates must "
            "exactly follow build-manifest evidence membership; "
            f"expected {expected_gate_names}, got {evaluated_gates!r}."
        )

    evidence = attestation.get("evidence_sha256")
    if not isinstance(evidence, Mapping):
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.evidence_sha256 must "
            "be an object."
        )
    else:
        for stage, digest in evidence.items():
            _check_sha256_field(
                filename=_UK_TERMINAL_GATE_REPORT_FILE,
                owner=f"attestation.evidence_sha256[{stage!r}]",
                value=digest,
                failures=failures,
            )
        if dict(evidence) != dict(build_evidence):
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.evidence_sha256 "
                "must exactly match build_manifest.json terminal_gate_evidence."
            )
        diagnostic_weights: Mapping | None = None
        if calibration_diagnostics is not None:
            uk = calibration_diagnostics.get("uk_diagnostics")
            if isinstance(uk, Mapping) and isinstance(uk.get("weights"), Mapping):
                diagnostic_weights = uk["weights"]
        if diagnostic_weights is not None:
            expected_release_dataset_sha = _canonical_sha256(
                {
                    "weights": {
                        field: diagnostic_weights.get(field)
                        for field in _UK_WEIGHT_SUMMARY_FIELDS
                    }
                }
            )
            if evidence.get("release_dataset") != expected_release_dataset_sha:
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} release_dataset evidence "
                    "digest must bind calibration_diagnostics.json shipped-weight "
                    "observables."
                )

    expected_gate_results_sha = _canonical_sha256(valid_gates)
    if attestation.get("gate_results_sha256") != expected_gate_results_sha:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.gate_results_sha256 "
            "does not match gates."
        )

    unsigned_attestation = {
        field: attestation.get(field)
        for field in (
            "schema_version",
            "producer",
            "policy_sha256",
            "evaluated_gates",
            "evidence_sha256",
            "gate_results_sha256",
        )
    }
    expected_attestation_sha = _canonical_sha256(
        {
            "schema_version": report.get("schema_version"),
            "enforced": report.get("enforced"),
            "passed": report.get("passed"),
            "gates": valid_gates,
            "attestation": unsigned_attestation,
        }
    )
    if attestation.get("sha256") != expected_attestation_sha:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.sha256 does not bind "
            "the complete terminal report."
        )


def _check_calibration_diagnostics(
    diagnostics: Mapping,
    failures: list[str],
    *,
    grandfathered_uk_june: bool = False,
) -> None:
    """Validate shared diagnostics, with one byte-lineage-scoped exemption.

    The release ``populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z``
    predates diagnostics schema 5. Its real schema-2 rows use the legacy
    ``aggregation`` selector instead of modern ``measure``/``filter`` selector
    objects. Only that exact release id may use this path; every other release
    remains on the modern contract.
    """

    schema_version = diagnostics.get("schema_version")
    if schema_version is None:
        failures.append("calibration_diagnostics.json is missing 'schema_version'.")
    elif grandfathered_uk_june and schema_version != 2:
        failures.append(
            "calibration_diagnostics.json grandfathered June UK release "
            f"requires legacy schema version 2, got {schema_version!r}."
        )
    elif (
        not grandfathered_uk_june
        and schema_version != CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION
    ):
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
            required_fields = (
                "name",
                "target_name",
                "period",
                "entity",
                "target",
                "compiled_target",
                "initial_estimate",
                "final_estimate",
                "relative_error",
                *(("aggregation",) if grandfathered_uk_june else ("measure", "filter")),
            )
            for field in required_fields:
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
            if not grandfathered_uk_june:
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


def _uk_non_negative_int(
    value: object,
    *,
    field: str,
    failures: list[str],
) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        failures.append(
            f"calibration_diagnostics.json {field} must be a non-negative integer, "
            f"got {value!r}."
        )
        return None
    return value


def _uk_finite_number(
    value: object,
    *,
    field: str,
    failures: list[str],
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        failures.append(
            f"calibration_diagnostics.json {field} must be a finite number, "
            f"got {value!r}."
        )
        return None
    number = float(value)
    if not math.isfinite(number):
        failures.append(
            f"calibration_diagnostics.json {field} must be finite, got {value!r}."
        )
        return None
    if minimum is not None and number < minimum:
        failures.append(
            f"calibration_diagnostics.json {field} must be at least {minimum}, "
            f"got {number}."
        )
    if maximum is not None and number > maximum:
        failures.append(
            f"calibration_diagnostics.json {field} must be at most {maximum}, "
            f"got {number}."
        )
    return number


def _check_uk_calibration_diagnostics(
    diagnostics: Mapping,
    failures: list[str],
) -> None:
    """Require the versioned UK release diagnostics on canonical exact-k ids."""

    uk = diagnostics.get("uk_diagnostics")
    if not isinstance(uk, Mapping):
        failures.append(
            "calibration_diagnostics.json canonical UK releases require a "
            "'uk_diagnostics' object."
        )
        return
    if uk.get("schema_version") != _UK_DIAGNOSTICS_SCHEMA_VERSION:
        failures.append(
            "calibration_diagnostics.json 'uk_diagnostics.schema_version' is "
            f"{uk.get('schema_version')!r}; expected "
            f"{_UK_DIAGNOSTICS_SCHEMA_VERSION}."
        )

    weights = uk.get("weights")
    if not isinstance(weights, Mapping):
        failures.append(
            "calibration_diagnostics.json 'uk_diagnostics.weights' must be an object."
        )
        weights = {}
    n_records = _uk_non_negative_int(
        weights.get("n_records"),
        field="uk_diagnostics.weights.n_records",
        failures=failures,
    )
    if n_records == 0:
        failures.append(
            "calibration_diagnostics.json UK release diagnostics require at "
            "least one weight record."
        )
    positive_records = _uk_non_negative_int(
        weights.get("positive_weight_records"),
        field="uk_diagnostics.weights.positive_weight_records",
        failures=failures,
    )
    zero_records = _uk_non_negative_int(
        weights.get("zero_weight_records"),
        field="uk_diagnostics.weights.zero_weight_records",
        failures=failures,
    )
    _uk_finite_number(
        weights.get("total_weight"),
        field="uk_diagnostics.weights.total_weight",
        failures=failures,
        minimum=0.0,
    )
    max_weight = _uk_finite_number(
        weights.get("max_weight"),
        field="uk_diagnostics.weights.max_weight",
        failures=failures,
        minimum=0.0,
    )
    ess = _uk_finite_number(
        weights.get("effective_sample_size"),
        field="uk_diagnostics.weights.effective_sample_size",
        failures=failures,
        minimum=0.0,
        maximum=float(n_records) if n_records is not None else None,
    )
    ess_fraction = _uk_finite_number(
        weights.get("ess_fraction"),
        field="uk_diagnostics.weights.ess_fraction",
        failures=failures,
        minimum=0.0,
        maximum=1.0,
    )
    top_share = _uk_finite_number(
        weights.get("top_1pct_weight_share"),
        field="uk_diagnostics.weights.top_1pct_weight_share",
        failures=failures,
        minimum=0.0,
        maximum=1.0,
    )
    if (
        n_records is not None
        and n_records > 0
        and ess is not None
        and ess_fraction is not None
        and not math.isclose(
            ess_fraction,
            ess / n_records,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        failures.append(
            "calibration_diagnostics.json UK ess_fraction must equal "
            "effective_sample_size/n_records."
        )
    shared_n_records = diagnostics.get("n_records")
    if not isinstance(shared_n_records, int) or isinstance(shared_n_records, bool):
        failures.append(
            "calibration_diagnostics.json canonical UK releases require numeric "
            "top-level n_records."
        )
    elif n_records is not None and shared_n_records != n_records:
        failures.append(
            "calibration_diagnostics.json UK weights.n_records must match "
            "top-level n_records."
        )
    target_surface = diagnostics.get("target_surface")
    if (
        isinstance(target_surface, Mapping)
        and "n_records" in target_surface
        and n_records is not None
        and target_surface.get("n_records") != n_records
    ):
        failures.append(
            "calibration_diagnostics.json UK weights.n_records must match "
            "target_surface.n_records."
        )
    for field, observed in (
        ("effective_sample_size", ess),
        ("top_1pct_weight_share", top_share),
    ):
        shared = diagnostics.get(field)
        if isinstance(shared, bool) or not isinstance(shared, int | float):
            failures.append(
                "calibration_diagnostics.json canonical UK releases require "
                f"numeric top-level {field}."
            )
        elif observed is not None and not math.isclose(
            float(shared),
            observed,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            failures.append(
                f"calibration_diagnostics.json UK weights.{field} must match "
                f"top-level {field}."
            )
    ratio_fields = ("median_positive_weight", "max_to_median_positive_weight")
    for field in ratio_fields:
        if field not in weights:
            failures.append(
                "calibration_diagnostics.json UK weight diagnostics require "
                f"uk_diagnostics.weights.{field}."
            )
    median = weights.get("median_positive_weight")
    ratio = weights.get("max_to_median_positive_weight")
    if positive_records == 0:
        if median is not None or ratio is not None:
            failures.append(
                "calibration_diagnostics.json UK all-zero weights require null "
                "median_positive_weight and max_to_median_positive_weight."
            )
    elif positive_records is not None:
        valid_median = _uk_finite_number(
            median,
            field="uk_diagnostics.weights.median_positive_weight",
            failures=failures,
            minimum=0.0,
        )
        valid_ratio = _uk_finite_number(
            ratio,
            field="uk_diagnostics.weights.max_to_median_positive_weight",
            failures=failures,
            minimum=1.0,
        )
        if valid_median is not None and valid_median <= 0.0:
            failures.append(
                "calibration_diagnostics.json UK positive weights require a "
                "strictly positive median_positive_weight."
            )
        if (
            valid_median is not None
            and valid_median > 0.0
            and max_weight is not None
            and valid_ratio is not None
            and not math.isclose(
                valid_ratio,
                max_weight / valid_median,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            failures.append(
                "calibration_diagnostics.json UK "
                "max_to_median_positive_weight must equal "
                "max_weight/median_positive_weight."
            )
    if (
        n_records is not None
        and positive_records is not None
        and zero_records is not None
        and positive_records + zero_records != n_records
    ):
        failures.append(
            "calibration_diagnostics.json UK positive- and zero-weight record "
            "counts must sum to weights.n_records."
        )

    strata = uk.get("zero_weight_rows_by_stratum")
    if not isinstance(strata, list) or not strata:
        failures.append(
            "calibration_diagnostics.json "
            "'uk_diagnostics.zero_weight_rows_by_stratum' must be a non-empty list."
        )
        strata = []
    stratum_rows = 0
    stratum_positive = 0
    stratum_zero = 0
    for index, row in enumerate(strata):
        if not isinstance(row, Mapping):
            failures.append(
                "calibration_diagnostics.json UK zero-weight stratum row "
                f"{index} must be an object."
            )
            continue
        if not isinstance(row.get("stratum"), Mapping) or not row["stratum"]:
            failures.append(
                "calibration_diagnostics.json UK zero-weight stratum row "
                f"{index} needs a non-empty 'stratum' object."
            )
        rows = _uk_non_negative_int(
            row.get("rows"),
            field=f"uk_diagnostics.zero_weight_rows_by_stratum[{index}].rows",
            failures=failures,
        )
        positive = _uk_non_negative_int(
            row.get("positive_weight_rows"),
            field=(
                "uk_diagnostics.zero_weight_rows_by_stratum"
                f"[{index}].positive_weight_rows"
            ),
            failures=failures,
        )
        zero = _uk_non_negative_int(
            row.get("zero_weight_rows"),
            field=(
                f"uk_diagnostics.zero_weight_rows_by_stratum[{index}].zero_weight_rows"
            ),
            failures=failures,
        )
        _uk_finite_number(
            row.get("weight_sum"),
            field=(f"uk_diagnostics.zero_weight_rows_by_stratum[{index}].weight_sum"),
            failures=failures,
            minimum=0.0,
        )
        if rows is None or positive is None or zero is None:
            continue
        if positive + zero != rows:
            failures.append(
                "calibration_diagnostics.json UK zero-weight stratum row "
                f"{index} counts do not reconcile."
            )
        stratum_rows += rows
        stratum_positive += positive
        stratum_zero += zero
    if n_records is not None and strata and stratum_rows != n_records:
        failures.append(
            "calibration_diagnostics.json UK stratum rows do not reconcile to "
            "weights.n_records."
        )
    if positive_records is not None and strata and stratum_positive != positive_records:
        failures.append(
            "calibration_diagnostics.json UK positive stratum rows do not "
            "reconcile to weights.positive_weight_records."
        )
    if zero_records is not None and strata and stratum_zero != zero_records:
        failures.append(
            "calibration_diagnostics.json UK zero-weight stratum rows do not "
            "reconcile to weights.zero_weight_records."
        )

    rates = uk.get("target_pass_rates_by_geography_level")
    if not isinstance(rates, list):
        failures.append(
            "calibration_diagnostics.json "
            "'uk_diagnostics.target_pass_rates_by_geography_level' must be a list."
        )
        rates = []
    seen_levels: set[str] = set()
    total_targets = total_scored = total_skipped = 0
    for index, row in enumerate(rates):
        if not isinstance(row, Mapping):
            failures.append(
                "calibration_diagnostics.json UK geography pass-rate row "
                f"{index} must be an object."
            )
            continue
        level = row.get("geography_level")
        if not isinstance(level, str) or level not in _UK_TARGET_GEOGRAPHY_LEVELS:
            failures.append(
                "calibration_diagnostics.json UK geography pass-rate row "
                f"{index} has unknown level {level!r}."
            )
        elif level in seen_levels:
            failures.append(
                "calibration_diagnostics.json UK geography pass-rate level "
                f"{level!r} appears more than once."
            )
        else:
            seen_levels.add(level)
        counts = [
            _uk_non_negative_int(
                row.get(field),
                field=(
                    "uk_diagnostics.target_pass_rates_by_geography_level"
                    f"[{index}].{field}"
                ),
                failures=failures,
            )
            for field in ("n_targets", "n_scored", "n_skipped", "n_within_10pct")
        ]
        if any(value is None for value in counts):
            continue
        n_targets, n_scored, n_skipped, n_within = counts
        assert n_targets is not None
        assert n_scored is not None
        assert n_skipped is not None
        assert n_within is not None
        if n_scored + n_skipped != n_targets or n_within > n_scored:
            failures.append(
                "calibration_diagnostics.json UK geography pass-rate row "
                f"{index} counts do not reconcile."
            )
        pass_rate = row.get("pass_rate")
        if n_targets == 0:
            if pass_rate is not None:
                failures.append(
                    "calibration_diagnostics.json UK empty geography pass-rate "
                    f"row {index} must use null pass_rate."
                )
        else:
            observed_rate = _uk_finite_number(
                pass_rate,
                field=(
                    "uk_diagnostics.target_pass_rates_by_geography_level"
                    f"[{index}].pass_rate"
                ),
                failures=failures,
                minimum=0.0,
                maximum=1.0,
            )
            expected_rate = n_within / n_targets
            if observed_rate is not None and not math.isclose(
                observed_rate,
                expected_rate,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                failures.append(
                    "calibration_diagnostics.json UK geography pass-rate row "
                    f"{index} pass_rate does not match n_within_10pct/n_targets."
                )
        total_targets += n_targets
        total_scored += n_scored
        total_skipped += n_skipped
    missing_levels = sorted(_UK_TARGET_GEOGRAPHY_LEVELS - seen_levels)
    if missing_levels:
        failures.append(
            "calibration_diagnostics.json UK geography pass rates are missing "
            f"level(s): {missing_levels}."
        )
    registry = diagnostics.get("target_registry")
    if isinstance(registry, Mapping):
        if registry.get("country") != "uk":
            failures.append(
                "calibration_diagnostics.json canonical UK releases require "
                "target_registry.country == 'uk'."
            )
        if isinstance(registry.get("n_specs"), int) and (
            total_targets != registry["n_specs"]
        ):
            failures.append(
                "calibration_diagnostics.json UK geography target counts do not "
                "reconcile to target_registry.n_specs."
            )
    targets = diagnostics.get("targets")
    if isinstance(targets, list) and total_scored != len(targets):
        failures.append(
            "calibration_diagnostics.json UK geography scored counts do not "
            "reconcile to len(targets)."
        )
    skipped = diagnostics.get("skipped")
    if isinstance(skipped, list) and total_skipped != len(skipped):
        failures.append(
            "calibration_diagnostics.json UK geography skipped counts do not "
            "reconcile to len(skipped)."
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
    return is_congressional_district_target(
        target.get("name"),
        target.get("metadata"),
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


def _validate_local_area_release_dir(release_dir: Path, release_id: str) -> None:
    """The non-default local-area release contract (populace#398).

    The artifact is calibrated to a local surface (population marginals +
    state administrative families) by design, so the national critical-target
    checks do not apply. What must hold instead: the declared gates all
    passed, the per-target diagnostics are complete and consistent, the
    source coverage carries the local provenance chain, a
    reviewed-limitations register exists, the manifest claims no default
    slot, and every artifact is pinned to the immutable release id.
    """

    failures: list[str] = []
    for filename in LOCAL_AREA_REQUIRED_RELEASE_FILES:
        if not (release_dir / filename).is_file():
            failures.append(f"required file {filename!r} is missing.")

    release_manifest: Mapping | None = None
    manifest_path = release_dir / "release_manifest.json"
    if manifest_path.is_file():
        release_manifest = _load_json(manifest_path, failures)
    if release_manifest is not None:
        _check_local_area_release_manifest(release_manifest, release_id, failures)

    build_manifest_path = release_dir / "build_manifest.json"
    if build_manifest_path.is_file():
        build_manifest = _load_json(build_manifest_path, failures)
        if build_manifest is not None:
            build_id = build_manifest.get("build_id")
            if build_id != release_id:
                failures.append(
                    f"build_manifest.json 'build_id' is {build_id!r} but the "
                    f"release directory is named {release_id!r}."
                )

    gate_path = release_dir / "gate_summary.json"
    if gate_path.is_file():
        gate_summary = _load_json(gate_path, failures)
        if gate_summary is not None:
            _check_local_area_gates(gate_summary, failures)

    diagnostics_path = release_dir / "calibration_diagnostics.json"
    if diagnostics_path.is_file():
        diagnostics = _load_json(diagnostics_path, failures)
        if diagnostics is not None:
            _check_local_area_calibration_diagnostics(diagnostics, failures)
            if _is_uk_exact_k_release_id(release_id):
                _check_uk_calibration_diagnostics(diagnostics, failures)
                _check_uk_exact_k_diagnostics_identity(
                    diagnostics, release_id, failures
                )

    coverage_path = release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE
    if coverage_path.is_file():
        coverage = _load_json(coverage_path, failures)
        if coverage is not None:
            for key in LOCAL_AREA_SOURCE_COVERAGE_KEYS:
                if coverage.get(key) is None:
                    failures.append(
                        f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} is missing "
                        f"the local-area provenance key {key!r} (or it is "
                        "null)."
                    )
            donor = coverage.get("donor_release")
            if isinstance(donor, Mapping) and not donor.get("release_id"):
                failures.append(
                    f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} donor_release "
                    "must pin a non-empty release_id (stage with "
                    "--donor-release-manifest)."
                )

    _check_local_area_checksum_ledger(release_dir, release_manifest, failures)
    _check_local_artifact_hashes(release_dir, release_manifest, failures)

    if failures:
        raise ReleaseContractError(release_dir, failures)


def _check_local_area_checksum_ledger(
    release_dir: Path,
    release_manifest: Mapping | None,
    failures: list[str],
) -> None:
    """Validate sha256sums.txt as a real ledger, not a presence token.

    Every required bundle file (and the manifest-declared artifacts,
    including the root H5 by name) must appear with a valid digest; entries
    naming files present in the directory must hash-match; unsafe or
    duplicate names are rejected.
    """

    ledger_path = release_dir / "sha256sums.txt"
    if not ledger_path.is_file():
        return
    entries: dict[str, str] = {}
    for line_number, line in enumerate(ledger_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not _SHA256_RE.fullmatch(parts[0]):
            failures.append(
                f"sha256sums.txt line {line_number} is not '<sha256>  <filename>'."
            )
            continue
        digest, name = parts[0], parts[1].strip()
        if "/" in name or name.startswith(".."):
            failures.append(
                f"sha256sums.txt line {line_number} names an unsafe path {name!r}."
            )
            continue
        if name in entries:
            failures.append(f"sha256sums.txt lists {name!r} more than once.")
            continue
        entries[name] = digest

    for filename in LOCAL_AREA_REQUIRED_RELEASE_FILES:
        if filename == "sha256sums.txt":
            continue
        if filename not in entries:
            failures.append(
                f"sha256sums.txt does not cover required file {filename!r}."
            )
    if isinstance(release_manifest, Mapping):
        artifacts = release_manifest.get("artifacts")
        if isinstance(artifacts, Mapping):
            for key, entry in artifacts.items():
                if not isinstance(entry, Mapping):
                    continue
                path = entry.get("path")
                declared = entry.get("sha256")
                if not isinstance(path, str) or not isinstance(declared, str):
                    continue
                if path not in entries:
                    failures.append(
                        f"sha256sums.txt does not cover manifest artifact "
                        f"{key!r} ({path!r})."
                    )
                elif entries[path] != declared:
                    failures.append(
                        f"sha256sums.txt digest for {path!r} disagrees with "
                        "release_manifest.json."
                    )
    for name, digest in entries.items():
        local = release_dir / name
        if local.is_file() and _sha256(local) != digest:
            failures.append(
                f"sha256sums.txt digest for {name!r} does not match the file's bytes."
            )


def _check_local_area_release_manifest(
    manifest: Mapping, release_id: str, failures: list[str]
) -> None:
    schema_version = manifest.get("schema_version")
    if schema_version != RELEASE_MANIFEST_SCHEMA_VERSION:
        failures.append(
            f"release_manifest.json 'schema_version' is {schema_version!r}; "
            f"this library publishes version {RELEASE_MANIFEST_SCHEMA_VERSION}."
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
    _check_uk_release_identity(manifest, release_id, failures)
    _check_release_manifest_package(
        manifest.get("data_package"),
        field="data_package",
        expected_name="populace-data",
        failures=failures,
    )
    default_datasets = manifest.get("default_datasets")
    if default_datasets != {}:
        failures.append(
            "release_manifest.json 'default_datasets' must be an empty "
            "object for a non-default local-area release; it claims no "
            f"default slot (got {default_datasets!r})."
        )
    if manifest.get("is_default") is not False:
        failures.append(
            "release_manifest.json 'is_default' must be false for a "
            "non-default local-area release."
        )
    namespace = manifest.get("namespace")
    if not isinstance(namespace, str) or not namespace:
        failures.append(
            "release_manifest.json must declare a non-empty 'namespace' for "
            "a non-default local-area release."
        )
    limitations = manifest.get("reviewed_limitations")
    if not isinstance(limitations, list):
        failures.append(
            "release_manifest.json must carry a 'reviewed_limitations' list "
            "(explicitly empty if none) for a non-default local-area release."
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        failures.append(
            "release_manifest.json must declare a non-empty 'artifacts' map."
        )
        return
    microdata = [
        name
        for name, entry in artifacts.items()
        if isinstance(entry, Mapping) and entry.get("kind") == "microdata"
    ]
    if len(microdata) != 1:
        failures.append(
            "release_manifest.json must declare exactly one microdata "
            f"artifact; found {sorted(microdata)!r}."
        )
    for name, entry in artifacts.items():
        if not isinstance(entry, Mapping):
            failures.append(
                f"release_manifest.json artifact {name!r} must be an object."
            )
            continue
        for field in ("kind", "path", "repo_id", "revision", "sha256"):
            if not entry.get(field):
                failures.append(
                    f"release_manifest.json artifact {name!r} is missing {field!r}."
                )
        revision = entry.get("revision")
        if revision is not None and revision != release_id:
            failures.append(
                f"release_manifest.json artifact {name!r} revision "
                f"{revision!r} is not pinned to the release id "
                f"{release_id!r}; non-default artifacts are discoverable by "
                "their immutable tag only."
            )
        sha = entry.get("sha256")
        if isinstance(sha, str) and not _SHA256_RE.fullmatch(sha):
            failures.append(
                f"release_manifest.json artifact {name!r} sha256 {sha!r} is "
                "not a 64-hex digest."
            )


def _check_local_area_gates(gate_summary: Mapping, failures: list[str]) -> None:
    gates = gate_summary.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        failures.append("gate_summary.json must carry a non-empty 'gates' object.")
        return
    for name, gate in gates.items():
        if not isinstance(gate, Mapping) or "passed" not in gate:
            failures.append(
                f"gate_summary.json gate {name!r} must be an object with a "
                "'passed' flag."
            )
            continue
        if gate["passed"] is not True:
            failures.append(
                f"gate_summary.json gate {name!r} did not pass; a "
                "non-default local-area release still publishes only green "
                "gates."
            )


def _check_local_area_calibration_diagnostics(
    diagnostics: Mapping, failures: list[str]
) -> None:
    n_targets = diagnostics.get("n_targets")
    targets = diagnostics.get("targets")
    if not isinstance(n_targets, int) or n_targets <= 0:
        failures.append(
            "calibration_diagnostics.json must carry a positive integer 'n_targets'."
        )
    if not isinstance(targets, list) or not targets:
        failures.append(
            "calibration_diagnostics.json must carry a non-empty 'targets' list."
        )
        return
    if isinstance(n_targets, int) and len(targets) != n_targets:
        failures.append(
            f"calibration_diagnostics.json 'targets' has {len(targets)} "
            f"rows but 'n_targets' is {n_targets}."
        )
    required_fields = (
        "target",
        "compiled_target",
        "initial_estimate",
        "final_estimate",
    )
    seen_names: set[str] = set()
    for index, row in enumerate(targets):
        if not isinstance(row, Mapping):
            failures.append(
                f"calibration_diagnostics.json target row {index} must be an object."
            )
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name:
            failures.append(
                f"calibration_diagnostics.json target row {index} is missing "
                "a non-empty 'name'."
            )
        elif name in seen_names:
            failures.append(
                f"calibration_diagnostics.json target name {name!r} appears "
                "more than once."
            )
        else:
            seen_names.add(name)
        for field in required_fields:
            value = row.get(field)
            if field not in row:
                failures.append(
                    f"calibration_diagnostics.json target row {index} is "
                    f"missing {field!r}."
                )
            elif not isinstance(value, (int, float)) or not math.isfinite(value):
                failures.append(
                    f"calibration_diagnostics.json target row {index} field "
                    f"{field!r} must be a finite number."
                )
    households = diagnostics.get("households")
    if not isinstance(households, int) or households <= 0:
        failures.append(
            "calibration_diagnostics.json must carry a positive integer 'households'."
        )
    for field in ("final_loss", "fraction_within_10pct"):
        value = diagnostics.get(field)
        if field not in diagnostics:
            failures.append(f"calibration_diagnostics.json is missing {field!r}.")
        elif not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(
                f"calibration_diagnostics.json {field!r} must be a finite number."
            )


def release_dataset_role(release_dir: Path | str) -> str:
    """The release's declared dataset role (default: national default).

    Read from ``release_manifest.json``'s ``dataset_role``. An absent field
    or unreadable manifest is the national default — the pre-role-class
    shape — so every historical release keeps its meaning; the publish path
    separately refuses pointer updates for any non-default role.
    """

    manifest_path = Path(release_dir) / "release_manifest.json"
    if not manifest_path.is_file():
        return NATIONAL_DEFAULT_DATASET_ROLE
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return NATIONAL_DEFAULT_DATASET_ROLE
    if not isinstance(manifest, Mapping):
        return NATIONAL_DEFAULT_DATASET_ROLE
    role = manifest.get("dataset_role", NATIONAL_DEFAULT_DATASET_ROLE)
    return role if isinstance(role, str) and role else NATIONAL_DEFAULT_DATASET_ROLE


def validate_release_dir(release_dir: Path | str) -> None:
    """Check a local release directory against its dataset-role contract.

    The directory name is the build id (``populace-us-2024-<sha>-<date>``)
    and both manifests must agree with the directory about which build this
    is. Which checks apply is keyed off ``release_manifest.json``'s
    ``dataset_role`` (populace#398):

    - ``national_default`` (or absent — every pre-role release): the full
      national contract, unchanged — :data:`REQUIRED_RELEASE_FILES`, the
      US critical-target set, national source-coverage enumeration.
    - ``non_default_local_area``: the local-area contract — gates object
      with every gate passed, per-target diagnostics shape, provenance-chain
      source coverage, a reviewed-limitations register, an empty
      ``default_datasets`` map, and artifacts pinned to the release id. The
      national critical-target set deliberately does not apply: the artifact
      is calibrated to a local surface by design.

    TODO(#578 H5 household-count reconciliation): when the first modern UK
    exact-k release is actually cut, bind these manifest/diagnostic counts to
    the household row count read from its shipped H5. There is deliberately no
    stub H5 check before that release artifact exists.

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

    # Role dispatch is strict about a PRESENT dataset_role: only an absent
    # field gets the legacy national semantics; an explicit null, boolean,
    # number, empty string, or unknown value is a contract error rather
    # than a silent fallback.
    role: str = NATIONAL_DEFAULT_DATASET_ROLE
    manifest_probe_path = release_dir / "release_manifest.json"
    if manifest_probe_path.is_file():
        try:
            manifest_probe = json.loads(manifest_probe_path.read_text())
        except (OSError, ValueError):
            manifest_probe = None
        if isinstance(manifest_probe, Mapping) and "dataset_role" in manifest_probe:
            declared_role = manifest_probe["dataset_role"]
            if declared_role not in (
                NATIONAL_DEFAULT_DATASET_ROLE,
                NON_DEFAULT_LOCAL_AREA_DATASET_ROLE,
            ):
                raise ReleaseContractError(
                    release_dir,
                    [
                        "release_manifest.json declares unknown dataset_role "
                        f"{declared_role!r}; known roles are "
                        f"{NATIONAL_DEFAULT_DATASET_ROLE!r} and "
                        f"{NON_DEFAULT_LOCAL_AREA_DATASET_ROLE!r}."
                    ],
                )
            role = declared_role
    if role == NON_DEFAULT_LOCAL_AREA_DATASET_ROLE:
        _validate_local_area_release_dir(release_dir, release_id)
        return

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
            _check_calibration_diagnostics(
                diagnostics,
                failures,
                grandfathered_uk_june=release_id == _UK_JUNE_RELEASE_ID,
            )
            if _is_uk_exact_k_release_id(release_id):
                _check_uk_calibration_diagnostics(diagnostics, failures)
                _check_uk_exact_k_diagnostics_identity(
                    diagnostics, release_id, failures
                )
            if release_id.startswith("populace-us-"):
                _check_us_critical_target_fit(diagnostics, failures)

    terminal_gate_path = release_dir / _UK_TERMINAL_GATE_REPORT_FILE
    if _is_uk_exact_k_release_id(release_id) and terminal_gate_path.is_file():
        terminal_gate_sha256 = _sha256(terminal_gate_path)
        _check_uk_terminal_gate_links(
            build_manifest=build_manifest,
            release_manifest=release_manifest,
            report_sha256=terminal_gate_sha256,
            failures=failures,
        )
        terminal_gate_report = _load_json(terminal_gate_path, failures)
        if terminal_gate_report is not None:
            _check_uk_terminal_gate_report(
                terminal_gate_report,
                build_manifest=build_manifest,
                calibration_diagnostics=calibration_diagnostics,
                failures=failures,
            )

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
