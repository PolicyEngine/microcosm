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


@pytest.mark.parametrize("release_id", BE_RELEASE_IDS)
def test_the_tagged_publisher_refuses_a_be_release(tmp_path, release_id) -> None:
    with pytest.raises(ValueError, match="publishes each build into its own folder"):
        release_module.prepare_release(tmp_path / release_id)


def test_a_be_build_named_by_its_old_folder_is_pointed_at_the_be_releases() -> None:
    """The publisher's refusal of a directory that is not a build-folder
    release names the releases BE's builds now publish into."""
    reason = release_module.not_a_build_folder_release_reason(OTHER_BE_IDS[-1])

    assert all(release_id in reason for release_id in BE_RELEASE_IDS)
