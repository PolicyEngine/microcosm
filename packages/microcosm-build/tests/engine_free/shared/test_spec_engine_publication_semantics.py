"""Golden gates for publication rung and release-id compilation."""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import UTC, datetime

import pytest

from microcosm.build.spec_engine import ResourceKind, load_bundle
from microcosm.build.spec_engine.errors import SpecValidationError
from microcosm.build.spec_engine.publication_semantics import (
    project_publication_legacy_release,
    project_spine_legacy_sampling,
    publication_rung_rows,
    publication_rung_token,
)
from tools.build_us_multispine_pool import (
    _STACKED_RELEASE_ID_PATTERN,
    _STACKED_SAMPLE_RUNG_TOKENS,
    _new_stacked_release_id,
)
from tools.us_bundle_generation.core import build_publication, build_spine


@pytest.fixture(scope="module")
def publication() -> dict[str, object]:
    value = load_bundle("us").domain(ResourceKind.PUBLICATION).to_wire()
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def spine() -> dict[str, object]:
    value = load_bundle("us").domain(ResourceKind.SPINE).to_wire()
    assert isinstance(value, dict)
    return value


def test_publication_rungs_match_live_constants(
    publication: dict[str, object],
) -> None:
    rows = publication_rung_rows(publication)
    assert [(row["fraction"], row["token"]) for row in rows] == list(
        _STACKED_SAMPLE_RUNG_TOKENS.items()
    )
    assert publication_rung_token(publication, 0.01) == "f001"
    assert publication_rung_token(publication, 0.25) == "f025"
    assert publication_rung_token(publication, 1.0) == "f100"


def test_publication_release_projection_matches_live_reader_grammar(
    publication: dict[str, object],
) -> None:
    release = project_publication_legacy_release(publication)
    assert release["legacy_compiled_regexes"] == [_STACKED_RELEASE_ID_PATTERN.pattern]
    legacy_id = _new_stacked_release_id(
        sample_fraction=0.25,
        sample_seed=578,
        realized_asec_households=11,
        realized_acs_households=13,
        timestamp=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        nonce="deadbeef",
    )
    assert legacy_id == (
        "populace-us-2024-stacked-f025-s578-asec11-acs13-20240102T030405Z-deadbeef"
    )
    assert re.fullmatch(release["legacy_compiled_regexes"][0], legacy_id)
    new_id = legacy_id.replace("populace-us-2024", "microcosm-us-2024")
    assert re.fullmatch(release["compiled_regex"], new_id)
    assert not re.fullmatch(release["compiled_regex"], legacy_id)


def test_publication_projections_survive_resolved_normalization(
    publication: dict[str, object],
    spine: dict[str, object],
) -> None:
    assert project_publication_legacy_release(publication) == (
        project_publication_legacy_release(build_publication())
    )
    assert project_spine_legacy_sampling(
        spine, publication=publication
    ) == project_spine_legacy_sampling(build_spine(), publication=build_publication())
    assert isinstance(
        project_spine_legacy_sampling(spine, publication=publication)["fraction"][
            "default"
        ],
        float,
    )


def test_publication_projection_refuses_grammar_mutations(
    publication: dict[str, object],
) -> None:
    mutated = deepcopy(publication)
    mutated["release"]["rung_fractions"][0]["token"] = "f002"
    with pytest.raises(SpecValidationError, match="unsupported rung"):
        publication_rung_rows(mutated)

    mutated = deepcopy(publication)
    mutated["release"]["pattern"] = "{line}-{rung}-{seed}"
    with pytest.raises(SpecValidationError, match="expected placeholders"):
        project_publication_legacy_release(mutated)
