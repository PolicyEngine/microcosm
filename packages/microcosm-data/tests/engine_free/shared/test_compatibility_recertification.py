"""A recertification republishes a published parent unchanged and says what moved.

Invariants (each held by tests below):

- Byte preservation: every parent artifact reaches the child with the parent's
  digest; changing one byte of any carried file refuses the child.
- Identity: the report's parent block is a function of the carried parent
  manifest and files alone; editing either refuses the child.
- Additivity of compatibility: the child's compatibility lists are the
  parent's entries followed by the claim, never fewer.
- Drift arithmetic: for every row, ``difference = tested - baseline`` and
  ``newly_read_inputs_effect + other_engine_changes_effect = difference``.
- Classification: ``stored_column_changes`` partitions the model-named
  columns it reports, and its ``ignored`` set is exactly what the stored-input
  rule refuses.
- Publication: a child whose parent the Hub does not serve, byte for byte,
  publishes nothing.
"""

# ruff: noqa: F403, F405

from __future__ import annotations

import json
import os
import shutil

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.data import compatibility_recertification as recert
from microcosm.data import release as release_module
from microcosm.data.contract import ReleaseContractError, validate_release_dir
from microcosm.data.stored_inputs import undefined_stored_inputs
from test_support.microcosm_data.compatibility_recertification import *


def _rewrite_json(path, update):
    os.chmod(path, 0o600)
    payload = json.loads(path.read_text())
    update(payload)
    path.write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# The pending candidate.
# ---------------------------------------------------------------------------


def test_candidate_republishes_every_parent_artifact_byte_for_byte(parent, candidate):
    parent_dir, parent_root = parent
    release_dir, artifact_root = candidate
    report = recert.validate_recertification_candidate(
        release_dir, artifact_root=artifact_root
    )
    assert report["compatibility"] == {"status": "pending"}
    assert report["transformation"] == "none"
    assert report["parent"] == recert.describe_parent(parent_dir, parent_root)
    for name in report["parent"]["release_files"]:
        assert (release_dir / name).read_bytes() == (parent_dir / name).read_bytes()
    for path in report["parent"]["root_artifacts"]:
        assert (artifact_root / path).read_bytes() == (parent_root / path).read_bytes()
    assert (release_dir / recert.PARENT_RELEASE_MANIFEST_FILE).read_bytes() == (
        parent_dir / "release_manifest.json"
    ).read_bytes()
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    parent_manifest = json.loads((parent_dir / "release_manifest.json").read_text())
    assert (
        manifest["compatible_model_packages"]
        == parent_manifest["compatible_model_packages"]
    )
    assert (
        manifest["build"]["built_with_model_package"]
        == parent_manifest["build"]["built_with_model_package"]
    )
    assert manifest["build"]["recertified_from"] == parent_dir.name
    for key, entry in parent_manifest["artifacts"].items():
        assert manifest["artifacts"][key] == {**entry, "revision": CHILD_ID}


def test_pending_candidate_cannot_pass_the_publication_contract(candidate):
    release_dir, artifact_root = candidate
    with pytest.raises(ReleaseContractError, match="pending"):
        validate_release_dir(release_dir, artifact_root=artifact_root)


def test_the_publication_contract_refuses_a_parent_h5(candidate):
    release_dir, artifact_root = candidate
    with pytest.raises(ReleaseContractError, match="takes no parent_h5"):
        validate_release_dir(
            release_dir,
            artifact_root=artifact_root,
            parent_h5=artifact_root / "populace_us_2024.h5",
        )


def _carried_files(candidate):
    release_dir, artifact_root = candidate
    report = json.loads((release_dir / recert.REPORT_FILE).read_text())
    return [
        *(release_dir / name for name in report["parent"]["release_files"]),
        release_dir / recert.PARENT_RELEASE_MANIFEST_FILE,
        release_dir / recert.PARENT_BUILD_MANIFEST_FILE,
        *(artifact_root / path for path in report["parent"]["root_artifacts"]),
    ]


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_changing_any_byte_of_any_carried_file_refuses(
    candidate, tmp_path_factory, data
):
    root = tmp_path_factory.mktemp("byte")
    release_dir = root / "releases" / CHILD_ID
    artifact_root = root / "artifacts"
    shutil.copytree(candidate[0], release_dir)
    shutil.copytree(candidate[1], artifact_root)
    copy = (release_dir, artifact_root)
    target = data.draw(st.sampled_from(_carried_files(copy)))
    os.chmod(target, 0o600)
    payload = bytearray(target.read_bytes())
    index = data.draw(st.integers(0, len(payload) - 1))
    payload[index] ^= data.draw(st.integers(1, 255))
    target.write_bytes(bytes(payload))
    with pytest.raises(ReleaseContractError):
        recert.validate_recertification_candidate(
            release_dir, artifact_root=artifact_root
        )


@pytest.mark.parametrize(
    "tamper, message",
    [
        (
            lambda release_dir: _rewrite_json(
                release_dir / recert.REPORT_FILE,
                lambda report: report["parent"].update(
                    build_id="populace-us-2024-other"
                ),
            ),
            "carried parent does not describe",
        ),
        (
            lambda release_dir: (release_dir / "extra.json").write_text("{}"),
            "unexpected",
        ),
        (
            lambda release_dir: _rewrite_json(
                release_dir / "release_manifest.json",
                lambda manifest: manifest["compatible_model_packages"].append(
                    {"name": "policyengine-us", "specifier": "==2.60.4"}
                ),
            ),
            "must carry the parent's compatible_model_packages",
        ),
        (
            lambda release_dir: _rewrite_json(
                release_dir / "release_manifest.json",
                lambda manifest: manifest["build"].update(
                    built_with_model_package={
                        "name": "policyengine-us",
                        "version": "2.60.4",
                    }
                ),
            ),
            "must stay the parent's",
        ),
        (
            lambda release_dir: _rewrite_json(
                release_dir / recert.REPORT_FILE,
                lambda report: report.update(transformation="reweighted"),
            ),
            "transformation",
        ),
    ],
)
def test_tampered_candidates_are_refused(candidate, tamper, message):
    release_dir, artifact_root = candidate
    tamper(release_dir)
    with pytest.raises(ReleaseContractError, match=message):
        recert.validate_recertification_candidate(
            release_dir, artifact_root=artifact_root
        )


def test_the_h5_may_not_live_in_the_release_directory(candidate):
    release_dir, artifact_root = candidate
    shutil.copyfile(
        artifact_root / "populace_us_2024.h5", release_dir / "populace_us_2024.h5"
    )
    with pytest.raises(ReleaseContractError, match="only in artifact_root"):
        recert.validate_recertification_candidate(
            release_dir, artifact_root=artifact_root
        )


# ---------------------------------------------------------------------------
# Which parents a recertification accepts.
# ---------------------------------------------------------------------------


def test_build_refuses_reusing_the_parent_id(parent, tmp_path, monkeypatch):
    tool = load_tool()
    monkeypatch.setattr(tool, "_producer_identity", lambda: dict(FIXTURE_PRODUCER))
    release_dir, artifact_root = parent
    with pytest.raises(ValueError, match="new bare US release id"):
        tool.build_candidate(
            parent_release_dir=release_dir,
            parent_artifact_root=artifact_root,
            output_dir=tmp_path / "out",
            release_id=release_dir.name,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "update, message",
    [
        (
            lambda m: m.update(release_type="source_enrichment"),
            "not a calibration release",
        ),
        (
            lambda m: m.update(release_type=recert.RECERTIFICATION_RELEASE_TYPE),
            "not a calibration",
        ),
        (
            lambda m: m.update(dataset_role="non_default_local_area"),
            "not supported yet",
        ),
    ],
)
def test_describe_parent_refuses_unsupported_parents(parent, update, message):
    release_dir, artifact_root = parent
    _rewrite_json(release_dir / "release_manifest.json", update)
    with pytest.raises(ValueError, match=message):
        recert.describe_parent(release_dir, artifact_root)


def test_describe_parent_refuses_an_artifact_that_differs_from_its_pin(parent):
    release_dir, artifact_root = parent
    (artifact_root / "populace_us_2024.h5").write_bytes(b"other bytes")
    with pytest.raises(ValueError, match="differs from its pin"):
        recert.describe_parent(release_dir, artifact_root)


def test_parent_failing_its_own_contract_cannot_be_recertified(
    parent, tmp_path, monkeypatch
):
    release_dir, artifact_root = parent
    _rewrite_json(
        release_dir / "release_manifest.json",
        lambda manifest: manifest.update(data_package={"name": "microcosm-data"}),
    )
    tool = load_tool()
    monkeypatch.setattr(tool, "_producer_identity", lambda: dict(FIXTURE_PRODUCER))
    with pytest.raises(ValueError, match="fails the release contract"):
        tool.build_candidate(
            parent_release_dir=release_dir,
            parent_artifact_root=artifact_root,
            output_dir=tmp_path / "out",
            release_id=CHILD_ID,
        )


def _strip_stored_input_evidence(release_dir):
    from microcosm.data.contract import US_STORED_INPUTS_EVIDENCE_KEY

    (release_dir / "stored_inputs.json").unlink()
    for name in ("build_manifest.json",):
        _rewrite_json(
            release_dir / name,
            lambda payload: (
                payload.pop(US_STORED_INPUTS_EVIDENCE_KEY, None),
                payload.get("gate_evidence", {}).pop(
                    US_STORED_INPUTS_EVIDENCE_KEY, None
                ),
            ),
        )

    def strip_release(manifest):
        manifest["build"].pop(US_STORED_INPUTS_EVIDENCE_KEY, None)
        manifest["build"].get("gate_evidence", {}).pop(
            US_STORED_INPUTS_EVIDENCE_KEY, None
        )
        manifest["artifacts"].pop(US_STORED_INPUTS_EVIDENCE_KEY, None)

    _rewrite_json(release_dir / "release_manifest.json", strip_release)


def test_a_parent_without_stored_input_evidence_is_replaced_not_waived(
    parent, tmp_path, monkeypatch
):
    release_dir, artifact_root = parent
    _strip_stored_input_evidence(release_dir)
    with pytest.raises(ReleaseContractError, match="stored_inputs"):
        validate_release_dir(release_dir, artifact_root=artifact_root)
    tool = load_tool()
    monkeypatch.setattr(tool, "_producer_identity", lambda: dict(FIXTURE_PRODUCER))
    tool.build_candidate(
        parent_release_dir=release_dir,
        parent_artifact_root=artifact_root,
        output_dir=tmp_path / "candidate",
        release_id=CHILD_ID,
    )
    child = tmp_path / "candidate" / "releases" / CHILD_ID
    report = json.loads((child / recert.REPORT_FILE).read_text())
    contract = report["parent_contract"]
    assert contract["status"] == recert.PARENT_CONTRACT_STORED_INPUT_EVIDENCE_REPLACED
    assert contract["replaced_failures"] and not contract["other_failures"]

    # The replacement is the baseline engine's own verdict: a failing one refuses.
    candidate = (child, tmp_path / "candidate" / "artifacts")
    monkeypatch.setattr(recert, "run_recertification_loader_replay", fake_replay())
    monkeypatch.setattr(recert, "_check_producer", lambda code, failures, label: None)
    scores = write_scores(tmp_path / "scores", child)
    baseline = json.loads(scores["baseline"].read_text())
    baseline["engine"]["stored_input_verdict"]["passed"] = False
    scores["baseline"].write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match="passing stored-input verdict"):
        recert.certify_recertification(
            candidate[0],
            tmp_path / "certified" / CHILD_ID,
            artifact_root=candidate[1],
            compatibility_wheels=(),
            baseline_scores=scores["baseline"],
            tested_scores=scores["tested"],
            counterfactual_scores=scores["counterfactual"],
            compatible_model_specifier="policyengine-us==2.60.4",
            compatibility_claim_declared_by=DECLARER,
        )


# ---------------------------------------------------------------------------
# The stored-column classification and its reviewed register.
# ---------------------------------------------------------------------------

_COLUMN = st.sampled_from(
    [
        "age",
        "income_tax",
        "first_home_mortgage_interest",
        "tax_unit_role_input",
        "person_source_id",
        "A_AGE",
        "parent_1_id",
        "novel_column",
        "_hidden",
    ]
)


@settings(max_examples=200, deadline=None)
@given(
    tables=st.dictionaries(
        st.sampled_from(["person", "tax_unit"]), st.lists(_COLUMN), max_size=2
    ),
    variables=st.sets(_COLUMN),
    register=st.sets(_COLUMN),
)
def test_stored_column_changes_partition_and_match_the_stored_input_rule(
    tables, variables, register
):
    register = {column: "reason" for column in register}
    changes = recert.stored_column_changes(
        tables, engine_variables=variables, stored_input_register=register
    )
    ignored = {item["column"] for item in changes[recert.CHANGE_IGNORED]}
    read = {item["column"] for item in changes[recert.CHANGE_READ]}
    stored = {column for columns in tables.values() for column in columns}
    assert not ignored & read
    assert ignored == set(
        undefined_stored_inputs(stored, engine_variables=variables, register=register)
    )
    assert read == {
        c
        for c in stored
        if c in variables and c in register and c.islower() and not c.startswith("_")
    }


def test_unreviewed_or_misdirected_changes_are_refused():
    changes = {
        recert.CHANGE_IGNORED: [{"column": "novel_column", "tables": ["person"]}],
        recert.CHANGE_READ: [
            {"column": "first_home_mortgage_interest", "tables": ["tax_unit"]}
        ],
    }
    failures = recert.reviewed_column_change_failures(changes)
    assert len(failures) == 2
    assert "no reviewed entry" in failures[0]
    assert "says the engine change makes it 'ignored'" in failures[1]


def test_every_register_entry_has_a_direction_and_a_reason():
    for column, entry in recert.RECERTIFICATION_COLUMN_CHANGES.items():
        assert entry.change in {recert.CHANGE_IGNORED, recert.CHANGE_READ}, column
        assert entry.reason.strip(), column


# ---------------------------------------------------------------------------
# The claim.
# ---------------------------------------------------------------------------


def _parent_manifest(candidate):
    return json.loads((candidate[0] / recert.PARENT_RELEASE_MANIFEST_FILE).read_text())


def test_claim_keeps_every_parent_entry_and_adds_the_measured_versions(candidate):
    parent_manifest = _parent_manifest(candidate)
    models, cores, claims = recert.claim_entries(
        parent_manifest,
        tested_packages=TESTED_PACKAGES,
        specifier="==2.60.4",
        declared_by=DECLARER,
    )
    assert models[:-1] == parent_manifest["compatible_model_packages"]
    assert (
        models[-1]
        == claims["model"]
        == {
            "name": "policyengine-us",
            "specifier": "==2.60.4",
            "basis": "publisher_claim",
            "declared_by": DECLARER,
        }
    )
    assert cores[:-1] == parent_manifest["compatible_core_packages"]
    assert cores[-1]["specifier"] == "==3.33.3"


@pytest.mark.parametrize(
    "specifier, message",
    [
        (">=1.0,<2.61", "reaches back to the built-with"),
        (">=2.60.4", "reaches 3.0.0"),
        ("==2.60.3", "excludes the tested"),
    ],
)
def test_claims_that_overreach_are_refused(candidate, specifier, message):
    with pytest.raises(ValueError, match=message):
        recert.claim_entries(
            _parent_manifest(candidate),
            tested_packages=TESTED_PACKAGES,
            specifier=specifier,
            declared_by=DECLARER,
        )


def test_a_parent_that_already_covers_the_tested_version_needs_no_child(candidate):
    packages = {**TESTED_PACKAGES, "policyengine-us": {"version": "1.729.0"}}
    with pytest.raises(ValueError, match="nothing to recertify"):
        recert.claim_entries(
            _parent_manifest(candidate),
            tested_packages=packages,
            specifier="==1.729.0",
            declared_by=DECLARER,
        )


# ---------------------------------------------------------------------------
# Certification and drift.
# ---------------------------------------------------------------------------


def test_certified_child_passes_the_publication_contract(
    candidate, tmp_path, monkeypatch
):
    output, artifact_root = certify(candidate, tmp_path, monkeypatch)
    validate_release_dir(output, artifact_root=artifact_root)
    report = json.loads((output / recert.REPORT_FILE).read_text())
    compatibility = report["compatibility"]
    assert compatibility["publisher_claims"]["model"]["declared_by"] == DECLARER
    measurement = compatibility["claim_measurement"]["model"]
    assert measurement["measured_versions"] == ["2.60.4"]
    assert measurement["baseline_version"] == "1.729.0"
    summary = measurement["drift"]["summary"]
    assert summary["exact_critical_rows"] == 18
    assert summary["headline"]["income_tax_net"][
        "relative_difference"
    ] == pytest.approx(0.1)
    read = compatibility["stored_column_changes"][recert.CHANGE_READ]
    assert (
        read[0]["reason"]
        == recert.RECERTIFICATION_COLUMN_CHANGES["tax_unit_role_input"].reason
    )


@pytest.mark.parametrize(
    "path, update",
    [
        (
            recert.DRIFT_FILE,
            lambda drift: drift["rows"][0].update(tested=0.0),
        ),
        (
            recert.REPORT_FILE,
            lambda report: report["compatibility"]["publisher_claims"]["model"].update(
                specifier=">=2.60.4,<2.61"
            ),
        ),
        (
            "release_manifest.json",
            lambda manifest: manifest["compatible_model_packages"][-1].update(
                specifier=">=2.60.4,<2.61"
            ),
        ),
        (
            recert.REPORT_FILE,
            lambda report: report["compatibility"]["claim_measurement"]["model"].update(
                measured_versions=["2.60.4", "2.60.5"]
            ),
        ),
    ],
)
def test_tampered_certified_children_are_refused(
    candidate, tmp_path, monkeypatch, path, update
):
    output, artifact_root = certify(candidate, tmp_path, monkeypatch)
    _rewrite_json(output / path, update)
    with pytest.raises(ReleaseContractError):
        validate_release_dir(output, artifact_root=artifact_root)


def test_drift_refuses_a_baseline_on_another_engine(candidate):
    release_dir, _ = candidate
    diagnostics = json.loads((release_dir / "calibration_diagnostics.json").read_text())
    report = json.loads((release_dir / recert.REPORT_FILE).read_text())
    baseline = score_document(release_dir, packages=TESTED_PACKAGES)
    with pytest.raises(ValueError, match="built-with policyengine-us"):
        recert.combine_drift(
            calibration_diagnostics=diagnostics,
            dataset_sha256=report["dataset"]["sha256"],
            baseline=baseline,
            tested=score_document(release_dir),
            counterfactual=None,
            baseline_engine={
                "model": {"version": "1.729.0"},
                "core": {"version": "3.19.0"},
            },
            tested_packages=TESTED_PACKAGES,
            newly_read_columns=[],
        )


def test_drift_requires_the_counterfactual_when_columns_are_newly_read(candidate):
    release_dir, _ = candidate
    diagnostics = json.loads((release_dir / "calibration_diagnostics.json").read_text())
    report = json.loads((release_dir / recert.REPORT_FILE).read_text())
    with pytest.raises(ValueError, match="counterfactual"):
        recert.combine_drift(
            calibration_diagnostics=diagnostics,
            dataset_sha256=report["dataset"]["sha256"],
            baseline=score_document(release_dir, packages=BASELINE_PACKAGES),
            tested=score_document(release_dir),
            counterfactual=None,
            baseline_engine={
                "model": {"version": "1.729.0"},
                "core": {"version": "3.19.0"},
            },
            tested_packages=TESTED_PACKAGES,
            newly_read_columns=["tax_unit_role_input"],
        )


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    base=st.floats(-1e12, 1e12, allow_nan=False),
    new=st.floats(-1e12, 1e12, allow_nan=False),
    withheld=st.floats(-1e12, 1e12, allow_nan=False),
)
def test_drift_rows_decompose_exactly(parent, base, new, withheld):
    release_dir, _ = parent
    diagnostics = json.loads((release_dir / "calibration_diagnostics.json").read_text())
    definitions = recert.drift_metric_definitions(diagnostics)
    dataset = {"artifact": "x", "filename": "x.h5", "sha256": "c" * 64}

    def document(
        packages, value, variant=recert.VARIANT_AS_STORED, withheld_columns=()
    ):
        return {
            "schema_version": recert.SCORES_SCHEMA_VERSION,
            "kind": recert.SCORES_KIND,
            "dataset": dataset,
            "variant": {"id": variant, "withheld_columns": list(withheld_columns)},
            "metric_definitions_sha256": recert.canonical_sha256(definitions),
            "runtime": {"packages": packages},
            "metrics": {
                m["metric_id"]: {"value": value} for m in definitions["metrics"]
            },
            "code": dict(FIXTURE_PRODUCER),
        }

    drift = recert.combine_drift(
        calibration_diagnostics=diagnostics,
        dataset_sha256=dataset["sha256"],
        baseline=document(BASELINE_PACKAGES, base),
        tested=document(TESTED_PACKAGES, new),
        counterfactual=document(
            TESTED_PACKAGES,
            withheld,
            recert.VARIANT_WITHOUT_NEWLY_READ,
            ["parent_1_id"],
        ),
        baseline_engine={
            "model": {"version": "1.729.0"},
            "core": {"version": "3.19.0"},
        },
        tested_packages=TESTED_PACKAGES,
        newly_read_columns=["parent_1_id"],
    )
    for row in drift["rows"]:
        assert row["difference"] == new - base
        assert row["newly_read_inputs_effect"] + row["other_engine_changes_effect"] == (
            pytest.approx(row["difference"], abs=1e-3)
        )
    assert recert.drift_summary(drift) == recert.drift_summary(
        json.loads(json.dumps(drift))
    )


def test_signed_component_scores_soi_rows_as_the_release_tool_does():
    import numpy as np

    values = np.array([-5.0, 0.0, 7.0])
    assert recert.signed_component(values, "income_tax").tolist() == [0.0, 0.0, 7.0]
    assert recert.signed_component(values, "adjusted_gross_income").tolist() == [
        -5.0,
        0.0,
        7.0,
    ]
    assert recert.signed_component(values, "estate_losses").tolist() == [5.0, 0.0, 0.0]
    assert recert.signed_component(values, "capital_gains_losses").tolist() == [
        0.0,
        0.0,
        7.0,
    ]


# ---------------------------------------------------------------------------
# Publication.
# ---------------------------------------------------------------------------


def _seed_parent(hub, parent_dir):
    tag = parent_dir.name
    hub._commits["commit-parent"] = {
        f"releases/{tag}/release_manifest.json": (
            parent_dir / "release_manifest.json"
        ).read_bytes(),
        f"releases/{tag}/build_manifest.json": (
            parent_dir / "build_manifest.json"
        ).read_bytes(),
    }
    hub._refs[tag] = "commit-parent"


def _publish(hub, output, artifact_root, monkeypatch):
    monkeypatch.setattr(recert, "run_recertification_loader_replay", fake_replay())
    monkeypatch.setattr(recert, "_check_producer", lambda code, failures, label: None)
    return release_module.publish_release(
        output,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        compatibility_wheels=(),
        update_latest=False,
        tag_only=True,
        notify=False,
    )


def test_publication_refuses_a_parent_the_hub_does_not_serve(
    hub, parent, candidate, tmp_path, monkeypatch
):
    output, artifact_root = certify(candidate, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="is not published"):
        _publish(hub, output, artifact_root, monkeypatch)
    assert hub.events == []


def test_publication_refuses_a_parent_whose_published_manifest_differs(
    hub, parent, candidate, tmp_path, monkeypatch
):
    parent_dir, _ = parent
    output, artifact_root = certify(candidate, tmp_path, monkeypatch)
    _seed_parent(hub, parent_dir)
    hub._commits["commit-parent"][
        f"releases/{parent_dir.name}/release_manifest.json"
    ] = b"{}"
    with pytest.raises(ValueError, match="differs from the carried"):
        _publish(hub, output, artifact_root, monkeypatch)
    assert hub.events == []


def test_publication_of_a_published_parents_child_is_tag_only(
    hub, parent, candidate, tmp_path, monkeypatch
):
    parent_dir, _ = parent
    output, artifact_root = certify(candidate, tmp_path, monkeypatch)
    _seed_parent(hub, parent_dir)
    _publish(hub, output, artifact_root, monkeypatch)
    uploads = dict(hub.uploads)
    assert (
        uploads["populace_us_2024.h5"]
        == (artifact_root / "populace_us_2024.h5").read_bytes()
    )
    assert f"releases/{CHILD_ID}/{recert.DRIFT_FILE}" in uploads
    assert "latest.json" not in uploads
    assert [tag["tag"] for tag in hub.tags] == [CHILD_ID]


def test_prepare_release_takes_wheels_but_not_a_parent_h5(
    candidate, tmp_path, monkeypatch
):
    output, artifact_root = certify(candidate, tmp_path, monkeypatch)
    monkeypatch.setattr(recert, "run_recertification_loader_replay", fake_replay())
    monkeypatch.setattr(recert, "_check_producer", lambda code, failures, label: None)
    with pytest.raises(ValueError, match="compatibility_wheels alone"):
        release_module.prepare_release(
            output,
            artifact_root=artifact_root,
            parent_h5=artifact_root / "populace_us_2024.h5",
            update_latest=False,
            tag_only=True,
        )
    wheel = tmp_path / "policyengine_us-2.60.4-py3-none-any.whl"
    wheel.write_bytes(b"")
    prepared = release_module.prepare_release(
        output,
        artifact_root=artifact_root,
        compatibility_wheels=(wheel,),
        update_latest=False,
        tag_only=True,
    )
    assert "populace_us_2024.h5" in prepared.root_artifacts
