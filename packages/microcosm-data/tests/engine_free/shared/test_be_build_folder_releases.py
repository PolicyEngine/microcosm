"""BE's releases publish into build folders, as the UK's do (tracker 5b · 3).

BE's two releases, one per population year, publish each build into its own
folder, ``releases/<release>/<build>/``, which the repository's
``releases.json`` lists. As for the UK national and dense releases, the
contract names them exactly, keys the files their builds carry on the name,
and ties them to no role in ``releases.json``: the build's ``dataset_role``
gives it. The tagged publisher refuses them by name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from microcosm.data import contract
from microcosm.data import release as release_module
from microcosm.data.contract import (
    REQUIRED_RELEASE_FILES,
    ReleaseContractError,
    publishes_to_build_folder,
    required_release_files,
    validate_release_dir,
)
from test_support.microcosm_data.build_folders import BE_BUILD, BE_RELEASE_IDS
from test_support.microcosm_data.release import (
    BE_CONTRACT_FILES,
    BE_FIXTURE_REPO_ID,
    _as_be_build_folder_release,
    artifact_root,  # noqa: F401  (a fixture)
    release_dir,  # noqa: F401  (a fixture)
)

#: The six files BE's release contract requires of every build.
BE_REQUIRED_FILES = (
    *REQUIRED_RELEASE_FILES,
    "source_coverage.json",
    "demographics.json",
    "reform_validation.json",
)
#: BE names outside the two releases: another population year, another
#: kind of release, and a folder name in the form of BE's builds published
#: before the move, each of which was its own release.
OTHER_BE_IDS = (
    "microcosm-be-2024-national",
    "microcosm-be-2025-local",
    "microcosm-be-2025-chronicle-0000000-20260101T000000Z",
)


def _contract_failures(release_directory: Path) -> list[str]:
    """Every failure the release contract finds, or none."""
    try:
        validate_release_dir(release_directory)
    except ReleaseContractError as refusal:
        return refusal.failures
    return []


def test_the_be_releases_are_a_closed_list_of_two_names() -> None:
    """The fixtures name the same two releases the contract does."""
    assert contract.BE_RELEASES == frozenset(BE_RELEASE_IDS)


@pytest.mark.parametrize("release_id", BE_RELEASE_IDS)
def test_each_be_release_publishes_into_build_folders(release_id) -> None:
    assert publishes_to_build_folder(release_id)


@pytest.mark.parametrize("release_id", OTHER_BE_IDS)
def test_no_other_be_name_publishes_into_build_folders(release_id) -> None:
    assert not publishes_to_build_folder(release_id)


@pytest.mark.parametrize("release_id", BE_RELEASE_IDS)
def test_a_be_release_requires_the_six_files_of_its_contract(release_id) -> None:
    assert required_release_files(release_id) == BE_REQUIRED_FILES


@pytest.mark.parametrize("release_id", BE_RELEASE_IDS)
def test_a_be_build_folder_with_its_six_files_validates(
    release_dir: Path,  # noqa: F811
    release_id: str,
) -> None:
    be_release_dir = _as_be_build_folder_release(release_dir, release=release_id)

    assert _contract_failures(be_release_dir) == []


@pytest.mark.parametrize("missing", sorted(BE_CONTRACT_FILES.values()))
def test_a_be_build_folder_without_a_contract_file_is_refused(
    release_dir: Path,  # noqa: F811
    missing: str,
) -> None:
    be_release_dir = _as_be_build_folder_release(release_dir, drop=(missing,))

    assert f"required file {missing!r} is missing." in _contract_failures(
        be_release_dir
    )


@pytest.mark.parametrize("undeclared", sorted(BE_CONTRACT_FILES.values()))
def test_a_be_build_folder_that_does_not_declare_a_contract_file_is_refused(
    release_dir: Path,  # noqa: F811
    undeclared: str,
) -> None:
    """The publisher hands over only the manifests and the declared
    artifacts, so a contract file that is present but not declared would
    not be published; as a US build must declare its source coverage, a
    BE build must declare each file its release contract adds."""
    be_release_dir = _as_be_build_folder_release(release_dir, undeclared=(undeclared,))

    assert (
        f"release_manifest.json artifacts must include {undeclared!r} for BE releases."
    ) in _contract_failures(be_release_dir)


def test_a_be_build_without_its_engines_record_is_refused(
    release_dir: Path,  # noqa: F811
) -> None:
    """A BE build runs on the Axiom rules engine and EUROMOD, not on
    PolicyEngine packages, so it records the engines it ran on instead of
    the PolicyEngine package versions a UK or US build records."""
    be_release_dir = _as_be_build_folder_release(release_dir, engines=None)

    assert (
        "build_manifest.json 'engines' must be a non-empty object naming each "
        "engine a BE build ran on."
    ) in _contract_failures(be_release_dir)


@pytest.mark.parametrize(
    "engine",
    [
        {"python": "3.14.0"},
        {"packages": {}},
        {"packages": {"numpy": ""}},
        {"packages": {"numpy": 2}},
        {"packages": {"": "2.4.6"}},
        {"model": ""},
        {"model": "EUROMOD J2.0+ BE_2025", "packages": {}},
        {"packages": {"numpy": "2.4.6"}, "model": ""},
        "axiom",
    ],
    ids=[
        "neither",
        "no-packages",
        "unversioned-package",
        "non-string-version",
        "empty-package-name",
        "empty-model",
        "model-with-empty-packages",
        "packages-with-empty-model",
        "not-an-object",
    ],
)
def test_a_be_engine_must_record_its_packages_or_its_model(
    release_dir: Path,  # noqa: F811
    engine: object,
) -> None:
    be_release_dir = _as_be_build_folder_release(
        release_dir, engines={"axiom_rules_engine": engine}
    )

    assert (
        "build_manifest.json 'engines.axiom_rules_engine' must record its "
        "'packages' (each package's name and version), its 'model', or both, "
        "each well formed."
    ) in _contract_failures(be_release_dir)


def test_a_be_build_in_the_local_area_role_records_its_engines(
    release_dir: Path,  # noqa: F811
) -> None:
    """A BE release is tied to no role, so a BE build in the local-area role
    is held to the local-area contract, and records its engines there too."""
    be_release_dir = _as_be_build_folder_release(release_dir, engines=None)
    manifest_path = be_release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_role"] = "non_default_local_area"
    manifest_path.write_text(json.dumps(manifest))

    assert (
        "build_manifest.json 'engines' must be a non-empty object naming each "
        "engine a BE build ran on."
    ) in _contract_failures(be_release_dir)


def test_a_policyengine_core_version_a_be_build_records_is_checked(
    release_dir: Path,  # noqa: F811
) -> None:
    """A BE build need not record policyengine-core in its runtime, but a
    version it does record must be resolved, as for any build."""
    be_release_dir = _as_be_build_folder_release(release_dir)
    build_path = be_release_dir / "build_manifest.json"
    build_manifest = json.loads(build_path.read_text())
    build_manifest["runtime"]["policyengine-core"] = "unknown"
    build_path.write_text(json.dumps(build_manifest))

    assert (
        "build_manifest.json 'runtime.policyengine-core' must be a resolved "
        "version, not missing or unknown."
    ) in _contract_failures(be_release_dir)


def test_a_be_build_still_records_its_python(
    release_dir: Path,  # noqa: F811
) -> None:
    """Only the PolicyEngine package records are BE's to leave out."""
    be_release_dir = _as_be_build_folder_release(release_dir)
    build_path = be_release_dir / "build_manifest.json"
    build_manifest = json.loads(build_path.read_text())
    build_manifest["runtime"] = {}
    build_path.write_text(json.dumps(build_manifest))

    assert (
        "build_manifest.json 'runtime.python' must be a resolved version, not "
        "missing or unknown."
    ) in _contract_failures(be_release_dir)


def test_a_policyengine_package_a_be_build_records_is_checked(
    release_dir: Path,  # noqa: F811
) -> None:
    """A BE build need not record a PolicyEngine package, but one it records
    must still be resolved."""
    be_release_dir = _as_be_build_folder_release(release_dir)
    manifest_path = be_release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["built_with_core_package"] = {
        "name": "policyengine-core",
        "version": "unknown",
    }
    manifest_path.write_text(json.dumps(manifest))

    assert (
        "release_manifest.json 'built_with_core_package.version' must be "
        "resolved, not 'unknown'."
    ) in _contract_failures(be_release_dir)


@pytest.mark.parametrize("key", sorted(BE_CONTRACT_FILES))
def test_a_be_contract_file_declared_under_another_key_is_refused(
    release_dir: Path,  # noqa: F811
    key: str,
) -> None:
    """In a build folder an artifact's role is its key, and the emitter hands
    a build's demographics and reform validation to the collector by role,
    so a BE build declares each added file under the key its contract
    gives it."""
    be_release_dir = _as_be_build_folder_release(release_dir)
    manifest_path = be_release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    other_key = f"be_{key}"
    manifest["artifacts"][other_key] = {
        **manifest["artifacts"].pop(key),
        "role": other_key,
    }
    manifest_path.write_text(json.dumps(manifest))

    assert (
        f"release_manifest.json declares {BE_CONTRACT_FILES[key]!r} under the "
        f"key {other_key!r}; a BE build declares it under {key!r}, the role "
        "its build folder gives it."
    ) in _contract_failures(be_release_dir)


def test_a_be_build_folder_in_the_tagged_layout_is_refused(
    release_dir: Path,  # noqa: F811
) -> None:
    """A BE build names its release and build as two strings in both
    manifests; the tagged layout's ``build_id`` is a retired field there."""
    be_release_dir = _as_be_build_folder_release(release_dir)
    tagged = json.loads((be_release_dir / "build_manifest.json").read_text())
    tagged["build_id"] = BE_BUILD
    (be_release_dir / "build_manifest.json").write_text(json.dumps(tagged))

    assert (
        "build_manifest.json carries 'build_id'; a build-folder manifest names "
        "its build only as 'build', and the build has no tag."
    ) in _contract_failures(be_release_dir)


def test_a_be_build_is_prepared_with_the_role_its_dataset_role_gives(
    release_dir: Path,  # noqa: F811
    artifact_root: Path,  # noqa: F811
) -> None:
    be_release_dir = _as_be_build_folder_release(release_dir)

    prepared = release_module.prepare_build_folder_release(
        be_release_dir, artifact_root=artifact_root
    )

    assert (prepared.repository, prepared.release, prepared.build) == (
        BE_FIXTURE_REPO_ID,
        BE_RELEASE_IDS[0],
        BE_BUILD,
    )
    assert prepared.release_role == "national"
    assert set(BE_REQUIRED_FILES) <= {item.name for item in prepared.files}


def test_a_be_build_named_by_its_old_folder_is_pointed_at_the_be_releases() -> None:
    """The publisher's refusal of a directory that is not a build-folder
    release names the releases BE's builds now publish into."""
    reason = release_module.not_a_build_folder_release_reason(OTHER_BE_IDS[-1])

    assert all(release_id in reason for release_id in BE_RELEASE_IDS)
