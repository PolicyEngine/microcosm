"""Entry-point refusals with MOCK_METADATA_ONLY source-parent metadata.

Headers below reach the context guard; they are not a certified candidate.
No test invokes an engine, worker subprocess, Hub or publication emitter.
"""

from __future__ import annotations

import copy

import pytest

from microcosm.data import compatibility_recertification as recert
from microcosm.data import release as release_module
from microcosm.data import source_parent_replay as replay
from microcosm.data.contract import validate_release_dir
from test_support.microcosm_data.source_parent_replay import (
    carry_mock_source_parent,
    make_mock_source_parent,
    mock_current_record,
    write_mock_pending_headers,
)


@pytest.fixture
def carried_source(tmp_path, monkeypatch):
    source = make_mock_source_parent(tmp_path)
    monkeypatch.setattr(replay, "_reviewed_profile", lambda: source.profile)
    calls = []

    def transport(context, *, mode, release_dir, artifact_root):
        calls.append(mode)
        assert context is source.context
        if mode == "legacy":
            return {
                "legacy_actual": copy.deepcopy(source.legacy),
                "legacy_runtime": dict(source.runtime),
            }
        assert mode == "current"
        return mock_current_record(source)

    monkeypatch.setattr(replay, "_run_worker", transport)
    evidence = replay.qualify_source_parent(
        source.release_dir, artifact_root=source.artifact_root, context=source.context
    ).to_dict()
    carried = carry_mock_source_parent(source, tmp_path, evidence)
    write_mock_pending_headers(carried)
    calls.clear()
    return source, carried, calls


def test_builder_requires_original_context_before_creating_output(
    carried_source, tmp_path
):
    import tools.build_us_compatibility_recertification as tool

    source, _, calls = carried_source
    output = tmp_path / "MOCK_METADATA_ONLY_uncreated_candidate"
    with pytest.raises(ValueError, match="explicit original replay context"):
        tool.build_candidate(
            parent_release_dir=source.release_dir,
            parent_artifact_root=source.artifact_root,
            output_dir=output,
            build_id="populace-us-2024-MOCK_METADATA_ONLY-new-child",
        )
    assert not output.exists()
    assert calls == []


@pytest.mark.parametrize("entrypoint", ["candidate", "strict", "prepare"])
def test_missing_context_refuses_at_each_local_validation_gate(
    carried_source, entrypoint
):
    _, carried, calls = carried_source
    functions = {
        "candidate": recert.validate_recertification_candidate,
        "strict": validate_release_dir,
        "prepare": release_module.prepare_build_folder_release,
    }
    with pytest.raises(ValueError, match="explicit original replay context"):
        functions[entrypoint](carried.release_dir, artifact_root=carried.artifact_root)
    assert calls == []


def test_certifier_refuses_before_reading_scores_or_creating_output(
    carried_source, tmp_path
):
    _, carried, calls = carried_source
    output = tmp_path / "MOCK_METADATA_ONLY_uncreated" / carried.release_dir.name
    with pytest.raises(ValueError, match="explicit original replay context"):
        recert.certify_recertification(
            carried.release_dir,
            output,
            artifact_root=carried.artifact_root,
            compatibility_wheels=(),
            baseline_scores=tmp_path / "MOCK_UNREAD_baseline.json",
            tested_scores=tmp_path / "MOCK_UNREAD_tested.json",
            counterfactual_scores=None,
            compatible_model_specifier="policyengine-us>=0.1,<0.2",
            compatibility_claim_declared_by="MOCK_METADATA_ONLY fixture declarer",
        )
    assert calls == []
    assert not output.exists()


def test_scorer_refuses_before_runtime_authentication_or_engine_import(
    carried_source, tmp_path
):
    import tools.build_us_compatibility_recertification as tool

    _, carried, calls = carried_source
    output = tmp_path / "MOCK_METADATA_ONLY_uncreated_scores.json"
    with pytest.raises(ValueError, match="explicit original context"):
        tool.score_candidate(
            release_dir=carried.release_dir,
            artifact_root=carried.artifact_root,
            output=output,
            variant=recert.VARIANT_AS_STORED,
            compatibility_wheels=(),
        )
    assert calls == []
    assert not output.exists()


@pytest.mark.parametrize("entrypoint", ["prepare", "publish"])
def test_tagged_publication_never_accepts_source_parent_context(
    carried_source, entrypoint
):
    source, _, calls = carried_source
    kwargs = {"source_parent_replay": source.context}
    if entrypoint == "publish":
        kwargs.update(
            repo_id="MOCK_METADATA_ONLY/not-a-published-dataset", notify=False
        )
    function = (
        release_module.prepare_release
        if entrypoint == "prepare"
        else release_module.publish_release
    )
    with pytest.raises(ValueError, match="build-folder publication lane"):
        function(source.release_dir, **kwargs)
    assert calls == []


def test_build_publisher_refuses_before_emitter_use(carried_source, monkeypatch):
    from microcosm.build import release_publication as publisher

    _, carried, calls = carried_source
    listings = []

    def repository_transport(repository):
        listings.append(repository)
        return []

    class NeverEmitter:
        def __getattr__(self, name):
            pytest.fail(f"invalid mock metadata reached emitter.{name}")

    monkeypatch.setattr(publisher, "_repository_files", repository_transport)
    with pytest.raises(ValueError, match="explicit original replay context"):
        publisher.publish_release_build(
            carried.release_dir,
            artifact_root=carried.artifact_root,
            client=NeverEmitter(),
        )
    assert listings == ["MOCK_METADATA_ONLY/not-a-published-dataset"]
    assert calls == []


def test_recertification_cannot_be_promoted_to_current(carried_source, monkeypatch):
    from microcosm.build import release_publication as publisher

    _, carried, calls = carried_source

    def never_repository_transport(repository):
        pytest.fail("make_current refusal must precede Hub transport")

    monkeypatch.setattr(publisher, "_repository_files", never_repository_transport)
    with pytest.raises(ValueError, match="current"):
        publisher.publish_release_build(
            carried.release_dir,
            artifact_root=carried.artifact_root,
            client=object(),
            make_current=True,
        )
    assert calls == []


def test_parent_publication_requires_replay_before_hub_transport(carried_source):
    _, carried, calls = carried_source

    class NeverHub:
        def __getattr__(self, name):
            pytest.fail(f"missing original context reached Hub.{name}")

    with pytest.raises(ValueError, match="explicit original context"):
        recert.require_parent_publication(
            NeverHub(),
            carried.release_dir,
            repo_id="policyengine/populace-us",
            artifact_root=carried.artifact_root,
        )
    assert calls == []
