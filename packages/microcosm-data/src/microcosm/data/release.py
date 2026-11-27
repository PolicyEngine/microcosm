"""Check a release build before it publishes.

A release publishes each build into its own folder,
``releases/<release>/<build>/``, when
:func:`~microcosm.data.contract.publishes_to_build_folder` names it: the UK
national and dense releases, each UK local-area release that the dataset
registry names, the US releases
(:data:`~microcosm.data.contract.US_RELEASES`) and BE's releases
(:data:`~microcosm.data.contract.BE_RELEASES`).
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
path of each of its contract files. Each repository's switch deleted its
``latest.json``, the UK releases' ``latest-<line>.json`` files and the US
repository's ``latest-evidence.json``, and Microcosm neither reads nor
writes pointer files: ``releases.json`` names each release's current build.
Since the US switch, an evidence-tier build (microcosm#506) is a build of
:data:`~microcosm.data.contract.US_EVIDENCE_RELEASE_ID`, whose current
build in ``releases.json`` took the place of ``latest-evidence.json``.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from microcosm.data.contract import (
    BE_RELEASES,
    BUILD_MANIFEST_FILE_ROLE,
    CHECKSUMS_FILE,
    CHECKSUMS_FILE_ROLE,
    NATIONAL_DEFAULT_DATASET_ROLE,
    RELEASE_MANIFEST_FILE_ROLE,
    RELEASE_ROLES,
    US_EVIDENCE_RELEASE_ID,
    US_RELEASES,
    publishes_to_build_folder,
    unregistered_uk_local_release,
    validate_evidence_release_dir,
    validate_release_dir,
)

__all__ = [
    "BUILD_FOLDER_PUBLICATION_COMMAND",
    "BuildFolderFile",
    "BuildFolderRelease",
    "TAGGED_RELEASE_ADVICE",
    "UNREGISTERED_UK_LOCAL_RELEASE_ADVICE",
    "build_folder_identity",
    "prepare_build_folder_release",
    "missing_on_hub",
    "not_a_build_folder_release_reason",
    "publication_refusal",
]

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


def never_cached_offline(error: BaseException) -> bool:
    """Whether huggingface_hub is offline and the file was never cached.

    Offline, the cache is all a reader has, so the loader refuses a
    ``releases.json`` that was never cached, saying so.
    """
    try:
        from huggingface_hub import constants
        from huggingface_hub.utils import LocalEntryNotFoundError
    except ImportError:  # pragma: no cover - declared dependency
        return False
    return bool(constants.HF_HUB_OFFLINE) and isinstance(error, LocalEntryNotFoundError)
