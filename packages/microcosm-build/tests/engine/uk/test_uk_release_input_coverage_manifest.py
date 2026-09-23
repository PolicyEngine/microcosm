"""Tests split from packages/microcosm-build/tests/test_uk_release_input_coverage_manifest.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_release_input_coverage_manifest import *


def test_cached_candidate_current_reader_preserves_frozen_measurements() -> None:
    generator = _load_generator()
    committed = _resource("efrs_parity_known_gaps.json")["candidate_evidence"]
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        pytest.skip("huggingface_hub is unavailable")
    cached = try_to_load_from_cache(
        repo_id=generator.CANDIDATE_REPO_ID,
        filename=generator.CANDIDATE_FILENAME,
        revision=generator.CANDIDATE_REVISION,
        repo_type=generator.CANDIDATE_REPO_TYPE,
    )
    if not isinstance(cached, str):
        pytest.skip("pinned certified candidate revision is not cached")

    regenerated = generator.build_candidate_evidence(Path(cached))
    # The reader's version is a new observation, not a replacement for the
    # frozen receipt's provenance. Every actual measurement must still match.
    assert regenerated["engine"]["version"] == version("policyengine-uk")
    assert {k: v for k, v in regenerated["engine"].items() if k != "version"} == {
        k: v for k, v in committed["engine"].items() if k != "version"
    }
    assert {k: v for k, v in regenerated.items() if k != "engine"} == {
        k: v for k, v in committed.items() if k != "engine"
    }
