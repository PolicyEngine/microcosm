"""Executable statements of how the release contract keys on name strings today.

docs/release-contract.md (C1, "The id-string keying") describes these gaps, and
C1 version 1, planned for milestone M3, replaces them with dispatch on the
manifest's declared country. Each test pins one of today's behaviours so that
the change is deliberate: when a change flips one, update the document and the
test together.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from microcosm.data import contract
from microcosm.data.contract import (
    REQUIRED_RELEASE_FILES,
    ReleaseContractError,
    line_for_release_id,
    publishes_to_build_folder,
    required_release_files,
    validate_release_dir,
)
from microcosm.data.registry import DEFAULT_VARIANT, REGISTRY
from test_support.microcosm_data.release import release_dir  # noqa: F401

#: Names in the release grammar, ``microcosm-<country>-<vintage>-<line>``, that
#: the contract does not list: three countries it does not know, and a US
#: vintage outside ``contract.US_RELEASES``.
UNLISTED_IDS = (
    "microcosm-nz-2025-national",
    "microcosm-be-2025-national",
    "microcosm-am-2025-national",
    "microcosm-us-2025-national",
)
#: A US build's own release in the tagged layout, where the release was the
#: build.
TAGGED_US_ID = "populace-us-2024-0000000-20260101"


@pytest.mark.parametrize("release_id", UNLISTED_IDS)
def test_an_unlisted_id_gets_only_the_base_files_and_no_model_package(release_id):
    assert required_release_files(release_id) == REQUIRED_RELEASE_FILES
    assert contract._expected_model_package(release_id) is None


def test_the_us_releases_are_a_closed_list_of_five_names():
    assert dict(contract.US_RELEASES) == {
        "microcosm-us-2024-national": "national",
        "microcosm-us-2024-local": "local_area",
        "microcosm-us-2024-receipt-donor": "national",
        "microcosm-us-2024-evidence": "national",
        "microcosm-us-2024-annual": "national",
    }


@pytest.mark.parametrize("release_id", (*contract.US_RELEASES, TAGGED_US_ID))
def test_the_us_contract_keys_on_five_exact_names_and_one_prefix(release_id):
    """The US requirements apply to the five names in ``contract.US_RELEASES``,
    matched exactly, and to every name with the ``populace-us-`` prefix. A US
    name outside both, such as ``microcosm-us-2025-national``, is unlisted."""

    assert contract.is_us_release(release_id)
    assert required_release_files(release_id) == (
        *REQUIRED_RELEASE_FILES,
        contract.US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
    )
    assert contract._expected_model_package(release_id) == "policyengine-us"


@pytest.mark.parametrize("release_id", (*UNLISTED_IDS, TAGGED_US_ID))
def test_an_unlisted_id_does_not_publish_into_a_build_folder(release_id):
    """``publishes_to_build_folder`` lists the build-folder releases by name.
    An unlisted name falls outside the list, and so does a US build's own
    release in the tagged layout."""

    assert not publishes_to_build_folder(release_id)


@pytest.mark.parametrize(
    "release_id",
    (
        contract._UK_NATIONAL_RELEASE_ID,
        contract._UK_DENSE_RELEASE_ID,
        "microcosm-uk-2024-25-local-k55000",
        *contract.US_RELEASES,
    ),
)
def test_the_uk_and_us_releases_publish_into_build_folders(release_id):
    assert publishes_to_build_folder(release_id)


@pytest.mark.parametrize(
    "release_id", (*UNLISTED_IDS, TAGGED_US_ID, *contract.US_RELEASES)
)
def test_only_uk_names_have_a_pointer_line(release_id):
    """``line_for_release_id`` gives the ``<line>`` of a ``latest-<line>.json``
    pointer file, which only the pointer reader still needs. No US release and
    no unlisted name has one."""

    assert line_for_release_id(release_id) is None


def test_the_uk_national_and_local_lines_are_the_only_lines():
    assert line_for_release_id(contract._UK_NATIONAL_RELEASE_ID) == "national"
    assert line_for_release_id("microcosm-uk-2024-25-local-k55000") == "local-k55000"
    assert line_for_release_id(contract._UK_DENSE_RELEASE_ID) is None


def test_the_registry_year_is_not_the_uk_national_releases_vintage():
    # The national release is the UK 2025 default load since microcosm#1432
    # (Max's #823 ruling).
    spec = REGISTRY[("uk", 2025, DEFAULT_VARIANT)]
    assert spec.release == contract._UK_NATIONAL_RELEASE_ID
    assert spec.release == "microcosm-uk-2024-25-national"
    assert not spec.release.startswith(f"microcosm-uk-{spec.year}-")


def _retarget(directory: Path, release_id: str, *, drop: tuple[str, ...] = ()):
    """Rename a tagged release to ``release_id``, optionally dropping files."""

    target = directory.with_name(release_id)
    directory.rename(target)
    build = json.loads((target / "build_manifest.json").read_text())
    build["build_id"] = release_id
    (target / "build_manifest.json").write_text(json.dumps(build))
    manifest = json.loads((target / "release_manifest.json").read_text())
    manifest["build"]["build_id"] = release_id
    for key in list(manifest["artifacts"]):
        if manifest["artifacts"][key]["path"] in drop:
            del manifest["artifacts"][key]
        else:
            manifest["artifacts"][key]["revision"] = release_id
    (target / "release_manifest.json").write_text(json.dumps(manifest))
    for name in drop:
        (target / name).unlink()
    return target


def test_the_same_bytes_pass_under_an_unlisted_id_and_fail_under_a_us_id(
    release_dir,  # noqa: F811
):
    """A new country's release fails open. In the tagged layout, which an
    unlisted name takes, renaming a US release to an NZ name lifts the
    source-coverage and stored-input requirements with nothing else changed."""

    drop = ("us_source_coverage.json", "stored_inputs.json")
    unlisted = _retarget(release_dir, "microcosm-nz-2025-national", drop=drop)
    validate_release_dir(unlisted)

    us = _retarget(unlisted, TAGGED_US_ID)
    with pytest.raises(ReleaseContractError) as refusal:
        validate_release_dir(us)
    assert "required file 'us_source_coverage.json' is missing." in (
        refusal.value.failures
    )
