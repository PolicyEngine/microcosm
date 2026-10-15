"""Executable statements of how the release contract keys on id strings today.

docs/release-contract.md (C1, "The id-string keying") describes these gaps, and
milestone M3 (2026-11-06) replaces them with dispatch on the manifest's declared
country. Each test pins one of today's behaviours so that the change is
deliberate: when M3 flips one, update the document and the test together.
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
    required_release_files,
    validate_release_dir,
)
from microcosm.data.registry import REGISTRY
from microcosm.data.release import line_pointer_payload
from test_support.microcosm_data.release import release_dir  # noqa: F401

#: Ids in the C5 grammar for countries and lines the contract does not list.
UNLISTED_IDS = (
    "microcosm-nz-2025-national",
    "microcosm-be-2025-national",
    "microcosm-am-2025-national",
    "microcosm-us-2024-national",
)


@pytest.mark.parametrize("release_id", UNLISTED_IDS)
def test_an_unlisted_id_gets_only_the_base_files_and_no_model_package(release_id):
    assert required_release_files(release_id) == REQUIRED_RELEASE_FILES
    assert contract._expected_model_package(release_id) is None


@pytest.mark.parametrize(
    "release_id", (*UNLISTED_IDS, "populace-us-2024-0000000-20260101")
)
def test_only_uk_ids_belong_to_a_publication_line(release_id):
    assert line_for_release_id(release_id) is None


def test_the_uk_national_and_local_lines_are_the_only_lines():
    assert line_for_release_id(contract._UK_NATIONAL_RELEASE_ID) == "national"
    assert line_for_release_id("microcosm-uk-2024-25-local-k55000") == "local-k55000"
    assert line_for_release_id(contract._UK_DENSE_RELEASE_ID) is None


def test_no_other_country_can_build_a_line_pointer():
    with pytest.raises(ValueError, match="belongs to line None"):
        line_pointer_payload(
            "microcosm-nz-2025-national",
            line="national",
            revision="microcosm-nz-2025-national",
        )


def test_the_registry_year_is_not_the_uk_national_ids_vintage():
    spec = REGISTRY[("uk", 2025, "national")]
    assert contract._UK_NATIONAL_RELEASE_ID == "microcosm-uk-2024-25-national"
    assert not contract._UK_NATIONAL_RELEASE_ID.startswith(f"microcosm-uk-{spec.year}-")


def _retarget(directory: Path, release_id: str, *, drop: tuple[str, ...] = ()):
    """Rename a release to ``release_id``, optionally dropping files."""

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
    """A new country's release fails open: renaming a US release to an NZ id
    lifts the source-coverage and stored-input requirements with nothing else
    changed."""

    drop = ("us_source_coverage.json", "stored_inputs.json")
    unlisted = _retarget(release_dir, "microcosm-nz-2025-national", drop=drop)
    validate_release_dir(unlisted)

    us = _retarget(unlisted, "populace-us-2024-0000000-20260101")
    with pytest.raises(ReleaseContractError) as refusal:
        validate_release_dir(us)
    assert "required file 'us_source_coverage.json' is missing." in (
        refusal.value.failures
    )
