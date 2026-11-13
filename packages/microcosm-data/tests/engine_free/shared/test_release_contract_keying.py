"""Executable statements of how the release contract keys on name strings today.

docs/release-contract.md (C1, "The id-string keying") describes these gaps, and
C1 version 1, planned for milestone M3, replaces them with dispatch on the
manifest's declared country. Each test pins one of today's behaviours so that
the change is deliberate: when a change flips one, update the document and the
test together.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from unittest import mock

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from microcosm.data import contract
from microcosm.data import release as release_module
from microcosm.data.contract import (
    REQUIRED_RELEASE_FILES,
    ReleaseContractError,
    line_for_release_id,
    publishes_to_build_folder,
    required_release_files,
    validate_release_dir,
)
from microcosm.data.registry import DEFAULT_VARIANT, REGISTRY, DatasetSpec
from test_support.microcosm_data.build_folders import uk_local_release_spec
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
        *contract.US_RELEASES,
    ),
)
def test_the_uk_and_us_releases_publish_into_build_folders(release_id):
    assert publishes_to_build_folder(release_id)


def _uk_local_release_id(k: int) -> str:
    return f"microcosm-uk-2024-25-local-k{k}"


@st.composite
def _registered_ks_and_a_k(draw):
    """Some registered ``k`` values and one ``k``, registered about half the time."""

    registered = draw(st.frozensets(st.integers(1, 10**9), max_size=4))
    any_k = st.integers(1, 10**9)
    k = draw(
        st.one_of(st.sampled_from(sorted(registered)), any_k) if registered else any_k
    )
    return registered, k


def test_the_registry_names_no_uk_local_area_release():
    """docs/release-contract.md (C5) says so; a pull request that registers a
    ``k`` updates the document and this test together."""

    assert not [
        spec.release
        for spec in REGISTRY.values()
        if spec.release is not None
        and contract._UK_LOCAL_LINE_RELEASE_ID_RE.fullmatch(spec.release)
    ]
    assert not publishes_to_build_folder("microcosm-uk-2024-25-local-k55000")
    assert contract.unregistered_uk_local_release("microcosm-uk-2024-25-local-k55000")


@settings(max_examples=200, deadline=None, derandomize=True)
@given(_registered_ks_and_a_k())
@example((frozenset(), 1))
@example((frozenset({12}), 12))
# A registered id that contains, or sits inside, the id asked about.
@example((frozenset({12}), 1))
@example((frozenset({12}), 2))
@example((frozenset({12}), 123))
@example((frozenset({1, 123}), 12))
def test_a_uk_local_area_release_publishes_only_once_the_registry_names_it(drawn):
    """For any positive ``k``, ``microcosm-uk-2024-25-local-k<k>`` publishes
    into build folders exactly when a registry entry names it as its release,
    the tagged publisher refuses it either way, and its pointer line is the
    same whatever the registry holds (microcosm#1586)."""

    registered, k = drawn
    release_id = _uk_local_release_id(k)
    entries = {
        spec.key: spec
        for spec in map(
            uk_local_release_spec, map(_uk_local_release_id, sorted(registered))
        )
    }
    with mock.patch.dict(REGISTRY, entries):
        assert publishes_to_build_folder(release_id) is (k in registered)
        assert contract.unregistered_uk_local_release(release_id) is (
            k not in registered
        )
        with pytest.raises(ValueError) as refusal:
            release_module._refuse_build_folder_release(release_id)
        reason = release_module.not_a_build_folder_release_reason(release_id)
        assert line_for_release_id(release_id) == f"local-k{k}"
    assert line_for_release_id(release_id) == f"local-k{k}"
    if k in registered:
        assert "publishes each build into its own folder" in str(refusal.value)
        assert release_module.TAGGED_RELEASE_ADVICE in reason
    else:
        assert release_module.UNREGISTERED_UK_LOCAL_RELEASE_ADVICE in str(refusal.value)
        assert release_module.UNREGISTERED_UK_LOCAL_RELEASE_ADVICE in reason
    # Outside the patch the shipped registry names no local-area release.
    assert not publishes_to_build_folder(release_id)


def test_a_registered_k_admits_its_own_name_and_no_other():
    """An entry admits the release it names, whole: registering ``local-k12``
    admits neither a ``k`` whose id sits inside that name nor one whose id
    contains it."""

    spec = uk_local_release_spec(_uk_local_release_id(12))
    with mock.patch.dict(REGISTRY, {spec.key: spec}):
        assert publishes_to_build_folder(_uk_local_release_id(12))
        assert not contract.unregistered_uk_local_release(_uk_local_release_id(12))
        for other in (1, 2, 120, 123, 112, 1212):
            assert not publishes_to_build_folder(_uk_local_release_id(other))
            assert contract.unregistered_uk_local_release(_uk_local_release_id(other))


@pytest.mark.parametrize(
    "changes",
    (
        {"release": None},
        {"release": contract._UK_DENSE_RELEASE_ID},
        {"release": "microcosm-uk-2024-25-local-k70"},
        {"country": "us"},
    ),
    ids=("no-release", "another-release", "a-longer-k", "a-us-entry"),
)
def test_only_a_uk_entrys_release_admits_a_k(changes):
    """A ``k`` is registered by a UK entry's ``release`` and by nothing else.
    An entry whose variant and file name carry ``local-k7`` admits nothing
    when its release is missing or names another release, and a US entry
    that names the UK release admits nothing either."""

    release_id = _uk_local_release_id(7)
    spec = dataclasses.replace(uk_local_release_spec(release_id), **changes)
    assert spec.variant == "local-k7"
    with mock.patch.dict(REGISTRY, {spec.key: spec}):
        assert not publishes_to_build_folder(release_id)
        assert contract.unregistered_uk_local_release(release_id)


@pytest.mark.parametrize(
    "release_id",
    (
        "microcosm-uk-2024-25-local-k0",
        "microcosm-uk-2024-25-local-k01",
        "microcosm-uk-2024-25-local-k",
        "microcosm-uk-2024-25-local-k7-extra",
        *UNLISTED_IDS,
    ),
)
def test_registering_a_name_outside_the_local_area_grammar_admits_nothing(release_id):
    """The registry admits a UK local-area name only. An entry that names any
    other release leaves ``publishes_to_build_folder`` unchanged, and such a
    name never reads as an unregistered local-area release."""

    spec = DatasetSpec(
        country="uk",
        year=2025,
        variant="outside-the-grammar",
        hf_repo="policyengine/populace-uk-private",
        filename="outside.h5",
        engine_module="policyengine_uk.data",
        engine_class="UKSingleYearDataset",
        engine_package="policyengine-uk",
        release=release_id,
    )
    with mock.patch.dict(REGISTRY, {spec.key: spec}):
        assert not publishes_to_build_folder(release_id)
        assert not contract.unregistered_uk_local_release(release_id)


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
