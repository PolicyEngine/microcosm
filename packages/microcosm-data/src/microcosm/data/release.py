"""Check a release build before it publishes, and read release pointer files.

Every release publishes each build into its own folder,
``releases/<release>/<build>/``: the UK national and dense releases, each UK
local-area release that the dataset registry names, the US releases
(:data:`~microcosm.data.contract.US_RELEASES`) and BE's releases
(:data:`~microcosm.data.contract.BE_RELEASES`), which
:func:`~microcosm.data.contract.publishes_to_build_folder` names.
``microcosm-publish-release-build`` (``microcosm.build.release_publication``,
in ``microcosm-build``) hands each build to the emitter's ``publish_release``
call, and the repository's ``releases.json`` records which build is current.
Before it does, :func:`publication_refusal` refuses non-release smoke output,
a reform validation that skipped its out-of-sample reforms and staging that
never arrived, and :func:`prepare_build_folder_release` checks the build
against its release contract and lists its files with their roles. Nothing
here writes to the Hub. No command publishes a release that
``publishes_to_build_folder`` does not name: the June 2023 UK release and
``populace-uk-<year>-<tier>-k<N>`` releases take no new builds, and a new
country's releases join it in the pull request that authorizes their first
publication.

Before each repository switched to ``releases.json``, its builds were
published under ``releases/<build_id>/`` with a tag each, and a
``latest.json`` pointer file at its root named the current release and the
path of each of its contract files. The tagged publisher that wrote those
pointer files is gone; the readers here still follow them.
:func:`latest_release` reads ``latest.json`` and returns the typed pointer,
and :func:`latest_line_release` reads the schema-1 ``latest-<line>.json``
pointers that made the UK national and local-area releases current. Those
add the fields ``line`` and ``revision``, so each could name an immutable
per-build tag without changing the repository-global ``latest.json``.

The EVIDENCE tier (microcosm#506) moved ``latest-evidence.json``, never
``latest.json``, and :func:`latest_evidence_release` reads it. Since the US
switch, an evidence-tier build is a build of
:data:`~microcosm.data.contract.US_EVIDENCE_RELEASE_ID` instead, whose
current build in ``releases.json`` takes the place of
``latest-evidence.json``. The switch deleted every ``latest*.json`` file, and
:func:`latest_evidence_release` reads that current build instead.

The Hub client is injected (``api=``) everywhere it is used, so the tests
read pointers from a fake Hub, with no network.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from microcosm.data.contract import (
    BE_RELEASES,
    BUILD_MANIFEST_FILE_ROLE,
    CHECKSUMS_FILE,
    CHECKSUMS_FILE_ROLE,
    EVIDENCE_RELEASE_ID_SEGMENT,
    NATIONAL_DEFAULT_DATASET_ROLE,
    RELEASE_MANIFEST_FILE_ROLE,
    RELEASE_ROLES,
    US_EVIDENCE_RELEASE_ID,
    US_RELEASES,
    _uk_line_cut_tag_re,
    line_for_release_id,
    publishes_to_build_folder,
    required_release_files,
    unregistered_uk_local_release,
    validate_evidence_release_dir,
    validate_release_dir,
)
from microcosm.data.releases_record import (
    RELEASES_RECORD_PATH,
    build_folder,
    parse_releases_record,
)

__all__ = [
    "BUILD_FOLDER_PUBLICATION_COMMAND",
    "BuildFolderFile",
    "BuildFolderRelease",
    "LATEST_EVIDENCE_POINTER_PATH",
    "LATEST_POINTER_PATH",
    "LATEST_POINTER_SCHEMA_VERSION",
    "LINE_POINTER_PATH_TEMPLATE",
    "RELEASE_TIER_CERTIFIED",
    "RELEASE_TIER_EVIDENCE",
    "TAGGED_RELEASE_ADVICE",
    "UNREGISTERED_UK_LOCAL_RELEASE_ADVICE",
    "LatestPointer",
    "build_folder_identity",
    "prepare_build_folder_release",
    "latest_evidence_pointer_payload",
    "latest_line_release",
    "latest_pointer_payload",
    "line_pointer_path",
    "missing_on_hub",
    "not_a_build_folder_release_reason",
    "line_pointer_payload",
    "publication_refusal",
    "latest_release",
    "latest_evidence_release",
]

#: Where the pointer lives in the dataset repo. The root, not a release
#: directory: the pointer is repo state, not release state.
LATEST_POINTER_PATH = "latest.json"

#: Per-line pointers coexist with the frozen repository-global pointer.
LINE_POINTER_PATH_TEMPLATE = "latest-{line}.json"

_LINE_RE = re.compile(r"^[a-z]+(-k[1-9][0-9]*)?$")
#: The evidence-tier pointer (microcosm#506): which evidence release is the
#: best *current* one. A separate file at the repo root, so certified
#: consumers reading ``latest.json`` (and the pe.py certification path built
#: on it) can never pick up an evidence artifact by accident.
LATEST_EVIDENCE_POINTER_PATH = "latest-evidence.json"

#: Version of the pointer payload itself, so the pointer can evolve without
#: consumers guessing (the same discipline the release manifest learned).
LATEST_POINTER_SCHEMA_VERSION = 1

#: Publication tiers (microcosm#506). Certified is the default everywhere;
#: the evidence tier is opted into explicitly and carries its tier in the
#: pointer payload, the release id, and the release manifest.
RELEASE_TIER_CERTIFIED = "certified"
RELEASE_TIER_EVIDENCE = "evidence"


@dataclass(frozen=True)
class LatestPointer:
    """A parsed release pointer: which release is current, and where.

    Attributes:
        release_id: The current build id (the ``releases/`` directory name).
        updated_at: ISO-8601 UTC timestamp of when the pointer was written.
        paths: Repo-relative path of each contract file, keyed by its stem
            (``"build_manifest"``, ``"release_manifest"``,
            ``"calibration_diagnostics"``, plus country-specific contract
            files such as US source coverage).
        tier: The publication tier the pointer names —
            :data:`RELEASE_TIER_CERTIFIED` for ``latest.json`` (whose payload
            predates tiers and carries no field), or
            :data:`RELEASE_TIER_EVIDENCE` for ``latest-evidence.json``.
        revision: Revision to fetch. Repository-global and evidence
            pointers default to ``release_id``, an immutable tag; a line
            pointer may name another tag; an evidence build in a build folder
            is read at ``main``, at which its folder never changes.
        line: Publication line for ``latest-<line>.json``; otherwise ``None``.
    """

    release_id: str
    updated_at: str
    paths: dict[str, str]
    tier: str = RELEASE_TIER_CERTIFIED
    revision: str | None = None
    line: str | None = None

    def __post_init__(self) -> None:
        if self.revision is None:
            object.__setattr__(self, "revision", self.release_id)


def latest_pointer_payload(
    release_id: str,
    *,
    updated_at: str | None = None,
    calibration_diagnostics_available: bool = True,
) -> dict:
    """The ``latest.json`` payload for ``release_id``.

    Paths are derived from the release contract — the pointer names exactly
    the files :func:`~microcosm.data.contract.required_release_files`
    guarantees exist for this release id.

    Args:
        release_id: The build id being made current.
        updated_at: ISO-8601 UTC timestamp; defaults to now.
    """
    if updated_at is None:
        updated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    return {
        "schema_version": LATEST_POINTER_SCHEMA_VERSION,
        "release_id": release_id,
        "updated_at": updated_at,
        "paths": {
            filename.removesuffix(".json"): f"releases/{release_id}/{filename}"
            for filename in required_release_files(release_id)
            if calibration_diagnostics_available
            or filename != "calibration_diagnostics.json"
        },
    }


def line_pointer_path(line: str) -> str:
    """Return the reserved root pointer path for a publication line."""
    if _LINE_RE.fullmatch(line) is None:
        raise ValueError(
            f"invalid release line {line!r}; expected ^[a-z]+(-k[1-9][0-9]*)?$"
        )
    if line == "evidence":
        # The grammar would render latest-evidence.json, the evidence-tier
        # pointer: refused at the grammar, not only downstream.
        raise ValueError(
            "release line 'evidence' collides with the evidence-tier pointer "
            f"{LATEST_EVIDENCE_POINTER_PATH}; it is not a publication line."
        )
    return LINE_POINTER_PATH_TEMPLATE.format(line=line)


def line_pointer_payload(
    release_id: str,
    *,
    line: str,
    revision: str,
    updated_at: str | None = None,
    calibration_diagnostics_available: bool = True,
) -> dict:
    """Build a schema-1 per-line pointer naming an immutable revision."""
    line_pointer_path(line)
    expected_line = line_for_release_id(release_id)
    if line != expected_line:
        raise ValueError(
            f"release {release_id!r} belongs to line {expected_line!r}, not {line!r}."
        )
    cut_tag_re = _uk_line_cut_tag_re(release_id)
    if (
        not isinstance(revision, str)
        or not revision
        or not (
            revision == release_id
            or (cut_tag_re is not None and cut_tag_re.fullmatch(revision) is not None)
        )
    ):
        raise ValueError(
            f"revision {revision!r} is not the release id {release_id!r} or "
            "a per-cut tag in that release's line family."
        )
    return {
        **latest_pointer_payload(
            release_id,
            updated_at=updated_at,
            calibration_diagnostics_available=calibration_diagnostics_available,
        ),
        "line": line,
        "revision": revision,
    }


def latest_evidence_pointer_payload(
    release_id: str,
    *,
    updated_at: str | None = None,
    calibration_diagnostics_available: bool = True,
) -> dict:
    """The ``latest-evidence.json`` payload for ``release_id``.

    Mirrors :func:`latest_pointer_payload` exactly, plus a ``tier`` field —
    so evidence consumers reuse certified pointer tooling, while a reader
    that lands on the wrong file sees the tier immediately.
    """
    return {
        **latest_pointer_payload(
            release_id,
            updated_at=updated_at,
            calibration_diagnostics_available=calibration_diagnostics_available,
        ),
        "tier": RELEASE_TIER_EVIDENCE,
    }


def _hf_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ImportError(
            "microcosm-data needs huggingface_hub to read release pointers; "
            "reinstall microcosm-data with its dependencies."
        ) from exc
    return HfApi()


#: Where a build-folder release publishes from: the workspace command that
#: hands each build to the emitter (``microcosm.build.release_publication``).
BUILD_FOLDER_PUBLICATION_COMMAND = "microcosm-publish-release-build"
#: Where a release that does not publish into build folders goes. Once its
#: repository records its builds in ``releases.json``, every US build
#: publishes into one of the US releases (``microcosm.data.contract.
#: US_RELEASES``), every BE build into one of BE's (``microcosm.data.
#: contract.BE_RELEASES``), and the other UK releases take no new builds;
#: the UK switch moved the June 2023 build into a build folder of release
#: ``populace-uk-2023``.
TAGGED_RELEASE_ADVICE = (
    "US builds publish into microcosm-us-2024-national, -local, "
    "-receipt-donor, -evidence, -annual or -local-annual, and BE builds into "
    f"{' or '.join(sorted(BE_RELEASES))}, in a directory named after the "
    "release; "
    "the June 2023 UK release and populace-uk-<year>-<tier>-k<N> releases "
    "take no new builds once the UK repository records its builds in "
    "releases.json."
)


#: Why ``microcosm-publish-release-build`` refuses a UK local-area release
#: that the dataset registry does not name.
UNREGISTERED_UK_LOCAL_RELEASE_ADVICE = (
    "The dataset registry (microcosm.data.registry.REGISTRY) names no UK "
    "local-area release with this k. The pull request that authorizes a k's "
    "first publication adds its registry entry, and its builds then publish "
    f"with `{BUILD_FOLDER_PUBLICATION_COMMAND}`."
)


def not_a_build_folder_release_reason(release_id: str) -> str:
    """Why ``release_id`` cannot publish into a build folder, for a refusal."""
    advice = (
        UNREGISTERED_UK_LOCAL_RELEASE_ADVICE
        if unregistered_uk_local_release(release_id)
        else TAGGED_RELEASE_ADVICE
    )
    return f"release {release_id!r} does not publish into build folders. {advice}"


@dataclass(frozen=True)
class BuildFolderFile:
    """One file of a build folder: its bare name, role, local path and digest."""

    name: str
    role: str
    path: Path
    sha256: str
    bytes: int


@dataclass(frozen=True)
class BuildFolderRelease:
    """A checked build of a build-folder release, ready to hand to the emitter.

    ``release_role`` is the release's role in ``releases.json``
    (:data:`~microcosm.data.contract.RELEASE_ROLES`). ``files`` lists the
    release manifest, the build manifest, every artifact the release manifest
    declares and the checksum ledger when there is one, each once.
    ``contract_warnings`` are the release contract's warnings, which never
    refuse a build.
    """

    release_dir: Path
    repository: str
    release: str
    build: str
    release_role: str
    files: tuple[BuildFolderFile, ...]
    calibration_diagnostics_available: bool
    contract_warnings: tuple[str, ...] = ()


def build_folder_identity(release_dir: Path | str) -> tuple[str, str, str]:
    """``(repository, release, build)`` named by a build folder's release manifest.

    Reads only the manifest, so a published build can be promoted from its
    local directory without checking it again; raises ``ValueError`` if the
    directory is not a build-folder release or its manifest does not name
    the release, the build and one repository.
    """
    release_dir = Path(release_dir)
    if not publishes_to_build_folder(release_dir.name):
        raise ValueError(not_a_build_folder_release_reason(release_dir.name))
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{release_dir}/release_manifest.json is not an object.")
    release, build = manifest.get("release"), manifest.get("build")
    if release != release_dir.name or not isinstance(build, str) or not build:
        raise ValueError(
            f"{release_dir}/release_manifest.json does not name release "
            f"{release_dir.name!r} and a build."
        )
    artifacts = manifest.get("artifacts")
    repositories = sorted(
        {
            str(entry.get("repo_id"))
            for entry in (artifacts.values() if isinstance(artifacts, Mapping) else ())
            if isinstance(entry, Mapping)
        }
    )
    if len(repositories) != 1:
        raise ValueError(
            f"release {release!r} declares artifacts in more than one repository "
            f"({repositories}); a build is published into one repository."
        )
    return repositories[0], release, build


def _refuse_misplaced_parent_inputs(
    release_dir: Path,
    parent_h5: Path | str | None,
    compatibility_wheels: tuple[Path | str, ...],
) -> None:
    """Refuse a parent H5 or tested wheels for a build whose contract takes none.

    A source enrichment's contract replays its parent's H5 and the tested
    wheels; a recertification replays its loader under its tested wheels too,
    but its H5 is its parent's, so it takes no parent H5.
    """
    if parent_h5 is None and not compatibility_wheels:
        return
    from microcosm.data.compatibility_recertification import (
        RECERTIFICATION_RELEASE_TYPE,
    )
    from microcosm.data.source_enrichment import SOURCE_ENRICHMENT_RELEASE_TYPE

    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    release_type = (
        manifest.get("release_type") if isinstance(manifest, Mapping) else None
    )
    if not (
        release_type == SOURCE_ENRICHMENT_RELEASE_TYPE
        or (release_type == RECERTIFICATION_RELEASE_TYPE and parent_h5 is None)
    ):
        raise ValueError(
            "parent_h5 and compatibility_wheels require a source_enrichment "
            "release; compatibility_wheels alone also serve a "
            "compatibility_recertification release."
        )


def prepare_build_folder_release(
    release_dir: Path | str,
    *,
    artifact_root: Path | str | None = None,
    repository: str | None = None,
    parent_h5: Path | str | None = None,
    compatibility_wheels: tuple[Path | str, ...] = (),
) -> BuildFolderRelease:
    """Check a build-folder release's build and list the files it publishes.

    The release contract runs first: the evidence contract for a build of
    the US evidence-tier release, the certified one otherwise. Each artifact is read from the release
    directory, or else from ``artifact_root`` (the H5 and the calibration
    weights live beside the candidate), and must match the SHA-256 its
    manifest records. ``repository``, when given, must be the one the
    artifacts name. A source enrichment's contract also takes its parent's
    H5 (``parent_h5``) and the tested wheels (``compatibility_wheels``), and a
    recertification's the wheels; any other build refuses them. Nothing on
    the Hub is read or written.

    Raises:
        ReleaseContractError: If the build breaks its release contract.
        ValueError: If the directory is not a build-folder release, names more
            than one repository, or a file's bytes differ from the manifest.
        FileNotFoundError: If an artifact is in neither directory.
    """
    release_dir = Path(release_dir)
    named_repository, release, build = build_folder_identity(release_dir)
    _refuse_misplaced_parent_inputs(release_dir, parent_h5, compatibility_wheels)
    contract_warnings: list[str] = []
    if release == US_EVIDENCE_RELEASE_ID:
        # An evidence-tier build records the validation checks it failed, so
        # its own contract runs; the certified one would refuse it.
        validate_evidence_release_dir(release_dir)
    else:
        contract_warnings = validate_release_dir(
            release_dir,
            parent_h5=parent_h5,
            artifact_root=artifact_root,
            compatibility_wheels=tuple(compatibility_wheels),
        )
        for contract_warning in contract_warnings:
            warnings.warn(f"release contract: {contract_warning}", stacklevel=2)
    if repository is not None and repository != named_repository:
        raise ValueError(
            f"release {release!r} names {named_repository} in its artifacts, not "
            f"{repository}."
        )
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    dataset_role = manifest.get("dataset_role", NATIONAL_DEFAULT_DATASET_ROLE)
    release_role = RELEASE_ROLES.get(dataset_role)
    if release_role is None:
        raise ValueError(
            f"release {release!r} declares dataset_role {dataset_role!r}, which "
            "has no role in releases.json."
        )
    expected_role = US_RELEASES.get(release)
    if expected_role is not None and release_role != expected_role:
        raise ValueError(
            f"release {release!r} has role {expected_role!r} in releases.json, but "
            f"this build declares dataset_role {dataset_role!r} (role "
            f"{release_role!r}); publish it into a release of its role."
        )
    root = Path(artifact_root) if artifact_root is not None else None
    sources: list[tuple[str, str, Path, str | None]] = [
        ("release_manifest.json", RELEASE_MANIFEST_FILE_ROLE, release_dir, None),
        ("build_manifest.json", BUILD_MANIFEST_FILE_ROLE, release_dir, None),
    ]
    if (release_dir / CHECKSUMS_FILE).is_file():
        sources.append((CHECKSUMS_FILE, CHECKSUMS_FILE_ROLE, release_dir, None))
    for entry in manifest["artifacts"].values():
        name = entry["path"]
        directory = release_dir
        if not (release_dir / name).is_file() and root is not None:
            directory = root
        sources.append((name, entry["role"], directory, entry["sha256"]))
    files: list[BuildFolderFile] = []
    for name, role, directory, expected_sha256 in sources:
        local = directory / name
        if not local.is_file():
            raise FileNotFoundError(
                f"{name} of release {release!r} is in neither {release_dir} nor "
                f"the artifact root {root}."
            )
        digest = _sha256(local)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(
                f"{name} has sha256 {digest}, but release_manifest.json records "
                f"{expected_sha256}."
            )
        files.append(BuildFolderFile(name, role, local, digest, local.stat().st_size))
    names = [item.name for item in files]
    if len(set(names)) != len(names):
        raise ValueError(f"release {release!r} names a file twice: {sorted(names)}.")
    diagnostics_declaration = manifest.get("calibration_diagnostics")
    diagnostics_available = not (
        isinstance(diagnostics_declaration, Mapping)
        and diagnostics_declaration.get("status") == "failed"
    )
    if not diagnostics_available:
        warnings.warn(
            "Publishing a dataset whose calibration diagnostics generation failed: "
            f"{diagnostics_declaration.get('error_code')}: "
            f"{diagnostics_declaration.get('message')}",
            stacklevel=2,
        )
    return BuildFolderRelease(
        release_dir=release_dir,
        repository=named_repository,
        release=release,
        build=build,
        release_role=release_role,
        files=tuple(files),
        calibration_diagnostics_available=diagnostics_available,
        contract_warnings=tuple(contract_warnings),
    )


def _non_release_artifact(release_dir: Path) -> bool:
    """Return whether the manifest explicitly identifies non-release output."""

    path = release_dir / "build_manifest.json"
    if not path.exists():
        return False
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(manifest, dict) and (
        manifest.get("non_release") is True
        or manifest.get("release_posture") == "non_release_smoke"
    )


def _staging_undelivered(release_dir: Path) -> bool:
    """True if a build that should have staged has nothing to show for it.

    The check is scoped by the presence of the ``staging`` key. Builders with
    no staging path are untouched, while an explicit ``enabled: false`` is a
    legitimate opt-out. A present but empty block, or an enabled run with no
    successful upload, records intended staging that never arrived.

    A US national build is certified as a step of its own (microcosm#1585
    item 11): the ``staging`` block is its certification's, and the build's
    own rides inside it under ``build``. Both must have arrived or been
    declared opted out, so a build whose own uploads never arrived is
    refused as it was before the split, however its certification staged.
    """
    path = release_dir / "build_manifest.json"
    if not path.exists():
        return False
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(manifest, dict) or "staging" not in manifest:
        return False
    staging = manifest["staging"]
    if not isinstance(staging, dict) or "build" not in staging:
        return _staging_block_undelivered(staging)
    certification = {key: value for key, value in staging.items() if key != "build"}
    return _staging_block_undelivered(certification) or _staging_block_undelivered(
        staging["build"]
    )


def _staging_block_undelivered(staging: object) -> bool:
    """True if one staging block records intended staging that never arrived."""

    if not isinstance(staging, dict) or not staging:
        return True
    if "contract_version" in staging:
        return _version_2_staging_undelivered(staging)
    if staging.get("enabled") is False:
        return not bool(staging.get("reason"))
    return not staging.get("uploads_succeeded")


def _version_2_staging_undelivered(staging: dict[str, object]) -> bool:
    """Validate publication-relevant version 2 delivery semantics."""

    required = {
        "contract_version",
        "enabled",
        "mode",
        "run_id",
        "configured_repository",
        "upload_attempts",
        "upload_successes",
        "read_back",
        "last_error_code",
        "opt_out_reason",
    }
    if set(staging) != required or staging.get("contract_version") != 2:
        return True
    enabled = staging.get("enabled")
    mode = staging.get("mode")
    run_id = staging.get("run_id")
    repository = staging.get("configured_repository")
    attempts = staging.get("upload_attempts")
    successes = staging.get("upload_successes")
    read_back = staging.get("read_back")
    error_code = staging.get("last_error_code")
    reason = staging.get("opt_out_reason")
    if (
        not isinstance(enabled, bool)
        or isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or attempts < 0
        or isinstance(successes, bool)
        or not isinstance(successes, int)
        or successes < 0
        or successes > attempts
        or read_back not in {"not_requested", "passed", "failed"}
        or (error_code is not None and not isinstance(error_code, str))
    ):
        return True
    if not enabled:
        return not (
            mode == "disabled"
            and run_id is None
            and repository is None
            and attempts == 0
            and successes == 0
            and read_back == "not_requested"
            and error_code is None
            and isinstance(reason, str)
            and bool(reason.strip())
        )
    if reason is not None or not isinstance(run_id, str) or not run_id:
        return True
    if mode == "local_only":
        return True
    if (
        mode != "local_and_remote"
        or not isinstance(repository, str)
        or not repository.strip()
    ):
        return True
    if read_back == "passed":
        return successes == 0 or error_code is not None
    if read_back == "failed":
        return True
    if error_code not in {None, "UPLOAD_FAILED"}:
        return True
    return successes == 0


def _reform_validation_skipped(release_dir: Path) -> bool:
    """True if the release carries a reform_validation.json that was built with
    out-of-sample reforms skipped (``out_of_sample_simulated`` false).

    Such a release publishes blank out-of-sample (OBBBA) rows on the dashboard,
    indistinguishable from a real result — so publishing one is refused unless
    explicitly allowed. Absent/unreadable file or a true flag → not skipped.
    """
    path = release_dir / "reform_validation.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return payload.get("out_of_sample_simulated") is False


def publication_refusal(
    release_dir: Path,
    *,
    allow_incomplete_reform_validation: bool = False,
    allow_missing_staging: bool = False,
) -> str | None:
    """Why a release must not be published, read from its local files, or None.

    The checks ``microcosm-publish-release-build`` runs before a build's
    release contract: non-release smoke output, a reform validation built
    with out-of-sample reforms skipped, and staging telemetry that never
    arrived. Each override flag lifts one check.
    """
    if _non_release_artifact(release_dir):
        return (
            "refusing to publish: build_manifest.json identifies this as "
            "non-release smoke output."
        )
    if not allow_incomplete_reform_validation and _reform_validation_skipped(
        release_dir
    ):
        return (
            "refusing to publish: reform_validation.json has "
            "out_of_sample_simulated=false (built with "
            "--skip-out-of-sample-reforms), so the dashboard would show blank "
            "out-of-sample reforms. Rebuild without skipping, or pass "
            "--allow-incomplete-reform-validation to publish anyway."
        )
    if not allow_missing_staging and _staging_undelivered(release_dir):
        return (
            "refusing to publish: this release's build_manifest records no "
            "delivered staging telemetry for the build (for a build "
            "certified separately: for its certification, or for the build "
            "itself under staging.build), so the build never appeared on the "
            "staging dashboard and there is no pre-publication review of it. "
            "Either the staging destination was lost mid-build (uploads "
            "pause after repeated failures — check the build machine's "
            "Hugging Face write token), or the manifest predates staging "
            "provenance. Rebuild with staging reaching its repo, or pass "
            "--allow-missing-staging to publish anyway. A build that declared "
            "--no-staging publishes without the flag."
        )
    return None


def missing_on_hub(error: BaseException) -> bool:
    """Whether a download failed because the Hub answered that the file is absent.

    Only that answer counts: ``EntryNotFoundError`` for the Hub's 404
    (``RemoteEntryNotFoundError`` in huggingface_hub 1.x). huggingface_hub
    also raises ``LocalEntryNotFoundError``, a subclass of it and of
    ``FileNotFoundError``, when it could not ask the Hub (a 5xx that outlasted
    its retries, a 403, offline mode) and nothing is cached; that says
    nothing about the file, so it, like every other error, does not count.
    """
    try:
        from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
    except ImportError:  # pragma: no cover - declared dependency
        return False
    return isinstance(error, EntryNotFoundError) and not isinstance(
        error, LocalEntryNotFoundError
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_ABSENT_REVISION_ERRORS = frozenset({"RevisionNotFoundError", "KeyError"})
_ABSENT_ENTRY_ERRORS = frozenset({"EntryNotFoundError", "FileNotFoundError"})


def _is_absent_revision_error(exc: BaseException) -> bool:
    """A Hub (or fake) answer that the revision does not exist, nothing else."""
    return bool({cls.__name__ for cls in type(exc).__mro__} & _ABSENT_REVISION_ERRORS)


def _is_absent_entry_error(exc: BaseException) -> bool:
    """A Hub (or fake) answer that the file does not exist at that revision.

    huggingface_hub's ``LocalEntryNotFoundError`` is also an
    ``EntryNotFoundError`` and a ``FileNotFoundError``, but it is raised when
    the Hub could not be asked (a 403, a 5xx that outlasted the retries, or
    offline with nothing cached), so it is not that answer.
    """
    names = {cls.__name__ for cls in type(exc).__mro__}
    return bool(names & _ABSENT_ENTRY_ERRORS) and "LocalEntryNotFoundError" not in names


def _read_pointer(repo_id: str, api, *, pointer_path: str) -> dict:
    """Download and structurally validate a release pointer file."""
    if api is None:
        api = _hf_api()
    local = api.hf_hub_download(
        repo_id=repo_id, filename=pointer_path, repo_type="dataset"
    )
    payload = json.loads(Path(local).read_text())
    schema_version = payload.get("schema_version")
    if schema_version != LATEST_POINTER_SCHEMA_VERSION:
        raise ValueError(
            f"{pointer_path} in {repo_id} has schema_version "
            f"{schema_version!r}; this microcosm-data reads version "
            f"{LATEST_POINTER_SCHEMA_VERSION}. Upgrade microcosm-data."
        )
    release_id = payload.get("release_id")
    if not release_id:
        raise ValueError(f"{pointer_path} in {repo_id} has no 'release_id'.")
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        raise ValueError(f"{pointer_path} in {repo_id} has no 'paths' object.")
    expected_paths = latest_pointer_payload(str(release_id), updated_at="")["paths"]
    observed_paths = {str(key): value for key, value in paths.items()}
    missing_paths = sorted(set(expected_paths) - set(observed_paths))
    if missing_paths == ["calibration_diagnostics"]:
        missing_paths = []
    unexpected_paths = sorted(set(observed_paths) - set(expected_paths))
    malformed_paths = sorted(
        key
        for key in set(expected_paths) & set(observed_paths)
        if observed_paths[key] != expected_paths[key]
    )
    if missing_paths or unexpected_paths or malformed_paths:
        raise ValueError(
            f"{pointer_path} in {repo_id} has incomplete paths: "
            f"missing={missing_paths}, unexpected={unexpected_paths}, "
            f"malformed={malformed_paths}."
        )
    return payload


def latest_release(repo_id: str, *, api=None) -> LatestPointer:
    """Read ``latest.json`` from a dataset repo: which release is current.

    Args:
        repo_id: Hub dataset repo, e.g. ``"policyengine/populace-us"``.
        api: A ``huggingface_hub.HfApi``-shaped object (anything with
            ``hf_hub_download(repo_id=, filename=, repo_type=)``);
            constructed lazily when omitted.

    Raises:
        ValueError: If the pointer is malformed, its schema version is newer
            than this library understands, or it carries a ``tier``, ``line``,
            or ``revision`` field. Those fields belong to other pointer
            families and must never be consumed as the certified default.
    """
    payload = _read_pointer(repo_id, api, pointer_path=LATEST_POINTER_PATH)
    if "tier" in payload:
        raise ValueError(
            f"{LATEST_POINTER_PATH} in {repo_id} carries a 'tier' field "
            f"({payload.get('tier')!r}); the certified pointer never does — "
            f"evidence releases live at {LATEST_EVIDENCE_POINTER_PATH}."
        )
    for field in ("line", "revision"):
        if field in payload:
            raise ValueError(
                f"{LATEST_POINTER_PATH} in {repo_id} carries a {field!r} field; "
                "the repository-global certified pointer never does."
            )
    return LatestPointer(
        release_id=str(payload["release_id"]),
        updated_at=str(payload.get("updated_at", "")),
        paths={str(k): str(v) for k, v in payload["paths"].items()},
        tier=RELEASE_TIER_CERTIFIED,
        revision=str(payload["release_id"]),
    )


def latest_line_release(repo_id: str, *, line: str, api=None) -> LatestPointer:
    """Read and validate the schema-1 pointer for ``line``."""
    pointer_path = line_pointer_path(line)
    payload = _read_pointer(repo_id, api, pointer_path=pointer_path)
    if "tier" in payload:
        raise ValueError(
            f"{pointer_path} in {repo_id} carries a 'tier' field; line pointers "
            "are certified pointers, not evidence pointers."
        )
    pointer_line = payload.get("line")
    if pointer_line != line:
        raise ValueError(
            f"{pointer_path} in {repo_id} declares line {pointer_line!r}, "
            f"expected {line!r}."
        )
    release_id = str(payload["release_id"])
    revision = payload.get("revision")
    expected_line = line_for_release_id(release_id)
    if line != expected_line:
        raise ValueError(
            f"{pointer_path} in {repo_id} names release {release_id!r}, which "
            f"belongs to line {expected_line!r}, not {line!r}."
        )
    cut_tag_re = _uk_line_cut_tag_re(release_id)
    if (
        not isinstance(revision, str)
        or not revision
        or not (
            revision == release_id
            or (cut_tag_re is not None and cut_tag_re.fullmatch(revision) is not None)
        )
    ):
        raise ValueError(
            f"{pointer_path} in {repo_id} has revision {revision!r}, which is "
            f"outside release {release_id!r}'s line family."
        )
    return LatestPointer(
        release_id=release_id,
        updated_at=str(payload.get("updated_at", "")),
        paths={str(k): str(v) for k, v in payload["paths"].items()},
        tier=RELEASE_TIER_CERTIFIED,
        revision=revision,
        line=line,
    )


def never_cached_offline(error: BaseException) -> bool:
    """Whether huggingface_hub is offline and the file was never cached.

    Offline, the cache is all a reader has, so a ``releases.json`` that was
    never cached reads as absent, as the pointer files then do.
    """
    try:
        from huggingface_hub import constants
        from huggingface_hub.utils import LocalEntryNotFoundError
    except ImportError:  # pragma: no cover - declared dependency
        return False
    return bool(constants.HF_HUB_OFFLINE) and isinstance(error, LocalEntryNotFoundError)


def _releases_record_bytes(api: object, *, repo_id: str) -> bytes | None:
    """The repository's ``releases.json``, or ``None`` when the Hub says it
    has none, or, offline, when it was never cached
    (:func:`never_cached_offline`); any other failure to read it propagates."""
    try:
        local = api.hf_hub_download(
            repo_id=repo_id, filename=RELEASES_RECORD_PATH, repo_type="dataset"
        )
    except Exception as exc:
        if missing_on_hub(exc) or never_cached_offline(exc):
            return None
        raise
    return Path(local).read_bytes()


def _current_evidence_build(repo_id: str, record_bytes: bytes) -> LatestPointer:
    """The evidence release's current build in ``releases.json``, as a pointer."""
    try:
        record = parse_releases_record(record_bytes)
    except ValueError as exc:
        raise ValueError(
            f"{RELEASES_RECORD_PATH} in {repo_id} is not a valid release record: {exc}."
        ) from exc
    release = US_EVIDENCE_RELEASE_ID
    entry = record.releases.get(release)
    if entry is None:
        raise ValueError(
            f"{RELEASES_RECORD_PATH} in {repo_id} does not list the evidence "
            f"release {release!r}."
        )
    build = entry.current
    if build is None:
        raise ValueError(
            f"The evidence release {release!r} in {repo_id} has no current build."
        )
    folder = build_folder(release, build)
    paths = {
        stem: f"{folder}/{posixpath.basename(path)}"
        for stem, path in latest_pointer_payload(build, updated_at="")["paths"].items()
    }
    return LatestPointer(
        release_id=build,
        updated_at=entry.builds[build].published_at,
        paths=paths,
        tier=RELEASE_TIER_EVIDENCE,
        revision="main",
    )


def latest_evidence_release(repo_id: str, *, api=None) -> LatestPointer:
    """Read ``latest-evidence.json``: the best *current* evidence release.

    The evidence-tier sibling of :func:`latest_release` (microcosm#506) — how
    consumers discover the best available artifact when no certified release
    carries it yet. Each evidence publish supersedes the last, so this
    pointer always names the current one.

    A repository that records its builds in ``releases.json`` (the US
    repository since its switch, microcosm#1570) has no
    ``latest-evidence.json``: its evidence tier is
    :data:`~microcosm.data.contract.US_EVIDENCE_RELEASE_ID`, and this returns
    that release's current build. ``release_id`` is then the build,
    ``paths`` are in its folder ``releases/<release>/<build>/``,
    ``revision`` is ``main``, at which a build folder never changes, and
    ``updated_at`` is when the build was published.

    Args:
        repo_id: Hub dataset repo, e.g. ``"policyengine/populace-us"``.
        api: A ``huggingface_hub.HfApi``-shaped object (anything with
            ``hf_hub_download(repo_id=, filename=, repo_type=)``);
            constructed lazily when omitted.

    Raises:
        ValueError: If the pointer is malformed, does not declare the
            evidence tier, or names a release id without the
            ``-evidence-`` segment; or, with ``releases.json``, if the
            evidence release is not listed or has no current build.
    """
    if api is None:
        api = _hf_api()
    record_bytes = _releases_record_bytes(api, repo_id=repo_id)
    if record_bytes is not None:
        return _current_evidence_build(repo_id, record_bytes)
    payload = _read_pointer(repo_id, api, pointer_path=LATEST_EVIDENCE_POINTER_PATH)
    tier = payload.get("tier")
    if tier != RELEASE_TIER_EVIDENCE:
        raise ValueError(
            f"{LATEST_EVIDENCE_POINTER_PATH} in {repo_id} declares tier "
            f"{tier!r}; the evidence pointer must declare "
            f"{RELEASE_TIER_EVIDENCE!r}."
        )
    release_id = str(payload["release_id"])
    if EVIDENCE_RELEASE_ID_SEGMENT not in release_id:
        raise ValueError(
            f"{LATEST_EVIDENCE_POINTER_PATH} in {repo_id} names release "
            f"{release_id!r}, which does not carry the "
            f"{EVIDENCE_RELEASE_ID_SEGMENT!r} segment."
        )
    return LatestPointer(
        release_id=release_id,
        updated_at=str(payload.get("updated_at", "")),
        paths={str(k): str(v) for k, v in payload["paths"].items()},
        tier=RELEASE_TIER_EVIDENCE,
    )
