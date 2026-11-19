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
- Layout: the child is a build of the national release in the build-folder
  layout, its H5 the parent's bytes under the release's dataset file name,
  whether the parent is in a build folder or keeps only its tag.
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
from microcosm.data.build_folder_manifests import referenced_build
from microcosm.data.contract import (
    US_NATIONAL_RELEASE_ID,
    ReleaseContractError,
    validate_release_dir,
)
from microcosm.data.stored_inputs import undefined_stored_inputs
from test_support.microcosm_data.compatibility_recertification import *


def _rewrite_json(path, update):
    os.chmod(path, 0o600)
    payload = json.loads(path.read_text())
    update(payload)
    path.write_text(json.dumps(payload))


def _child_path(parent_path):
    """Where the child keeps a parent root artifact: the H5 under its new name."""

    return CHILD_DATASET_FILE if parent_path == "populace_us_2024.h5" else parent_path


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
        assert (artifact_root / _child_path(path)).read_bytes() == (
            parent_root / path
        ).read_bytes()
    assert not (artifact_root / "populace_us_2024.h5").exists()
    assert (release_dir / recert.PARENT_RELEASE_MANIFEST_FILE).read_bytes() == (
        parent_dir / "release_manifest.json"
    ).read_bytes()
    parent_dataset = report["parent"]["dataset"]
    assert parent_dataset["filename"] == "populace_us_2024.h5"
    assert report["dataset"] == {
        "artifact": CHILD_DATASET_KEY,
        "filename": CHILD_DATASET_FILE,
        "sha256": parent_dataset["sha256"],
    }
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    build = json.loads((release_dir / "build_manifest.json").read_text())
    parent_manifest = json.loads((parent_dir / "release_manifest.json").read_text())
    assert release_dir.name == US_NATIONAL_RELEASE_ID
    assert (manifest["release"], manifest["build"]) == (
        US_NATIONAL_RELEASE_ID,
        CHILD_ID,
    )
    assert (build["release"], build["build"]) == (US_NATIONAL_RELEASE_ID, CHILD_ID)
    assert "build_id" not in build and build["dataset"] == report["dataset"]
    assert build["recertified_from"] == parent_dir.name
    assert (
        manifest["compatible_model_packages"]
        == parent_manifest["compatible_model_packages"]
    )
    for field in ("built_at", "built_with_model_package", "built_with_core_package"):
        assert manifest[field] == parent_manifest["build"][field]
    assert manifest["recertified_from"] == parent_dir.name
    assert manifest["default_datasets"] == {"national": CHILD_DATASET_KEY}
    for key, entry in parent_manifest["artifacts"].items():
        carried = {name: value for name, value in entry.items() if name != "revision"}
        if key == "populace_us_2024":
            key, carried["path"] = CHILD_DATASET_KEY, CHILD_DATASET_FILE
        assert {
            name: value
            for name, value in manifest["artifacts"][key].items()
            if name != "role"
        } == carried
    assert "populace_us_2024" not in manifest["artifacts"]
    # The build-folder publisher uploads the build manifest itself.
    assert "build_manifest" not in manifest["artifacts"]
    assert all("revision" not in entry for entry in manifest["artifacts"].values())
    assert manifest["artifacts"][CHILD_DATASET_KEY]["role"] == "dataset"


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
            parent_h5=artifact_root / CHILD_DATASET_FILE,
        )


def test_a_candidate_whose_directory_names_its_build_is_refused(candidate, tmp_path):
    """The directory is named after the release; the build is named inside."""

    release_dir, artifact_root = candidate
    renamed = tmp_path / "renamed" / CHILD_ID
    shutil.copytree(release_dir, renamed)
    with pytest.raises(ReleaseContractError, match="build of the national release"):
        recert.validate_recertification_candidate(renamed, artifact_root=artifact_root)


def _carried_files(candidate):
    release_dir, artifact_root = candidate
    report = json.loads((release_dir / recert.REPORT_FILE).read_text())
    return [
        *(release_dir / name for name in report["parent"]["release_files"]),
        release_dir / recert.PARENT_RELEASE_MANIFEST_FILE,
        release_dir / recert.PARENT_BUILD_MANIFEST_FILE,
        *(
            artifact_root / _child_path(path)
            for path in report["parent"]["root_artifacts"]
        ),
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
    release_dir = root / "releases" / US_NATIONAL_RELEASE_ID
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
                lambda manifest: manifest.update(
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
                release_dir / "release_manifest.json",
                lambda manifest: manifest["artifacts"].update(
                    build_manifest={
                        "kind": "evidence",
                        "path": "build_manifest.json",
                        "repo_id": "policyengine/populace-us",
                        "sha256": recert.sha256_file(
                            release_dir / "build_manifest.json"
                        ),
                        "role": "build_manifest",
                    }
                ),
            ),
            "release manifest artifacts must be the parent's",
        ),
        (
            lambda release_dir: _rewrite_json(
                release_dir / recert.REPORT_FILE,
                lambda report: report["dataset"].update(filename="populace_us_2024.h5"),
            ),
            "dataset must be the parent's microdata",
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


@pytest.mark.parametrize(
    "build_id", ["ABSOLUTE", "../outside", "populace-us-x/../../y"]
)
def test_parent_reconstruction_never_writes_outside_staging(
    candidate, tmp_path, build_id
):
    """Sol r4 finding 3: the report's parent id names the rebuild directory."""

    release_dir, artifact_root = candidate
    target = tmp_path / "recert-outside"
    bad = str(target) if build_id == "ABSOLUTE" else build_id
    _rewrite_json(
        release_dir / recert.REPORT_FILE,
        lambda report: report["parent"].update(build_id=bad),
    )
    with pytest.raises(ReleaseContractError):
        recert.validate_recertification_candidate(
            release_dir, artifact_root=artifact_root
        )
    assert not target.exists()
    assert not (release_dir.parent.parent / "outside").exists()


def test_the_h5_may_not_live_in_the_release_directory(candidate):
    release_dir, artifact_root = candidate
    shutil.copyfile(
        artifact_root / CHILD_DATASET_FILE, release_dir / CHILD_DATASET_FILE
    )
    with pytest.raises(ReleaseContractError, match="only in artifact_root"):
        recert.validate_recertification_candidate(
            release_dir, artifact_root=artifact_root
        )


def test_a_recertification_is_a_build_of_the_national_release_only(candidate):
    """A recertification republishes a calibrated national build's data, so a
    directory named after another US release, here the receipt-donor
    release, whose role is also national, is refused (review of #1571)."""
    release_dir, artifact_root = candidate
    moved = release_dir.with_name("microcosm-us-2024-receipt-donor")
    release_dir.rename(moved)

    with pytest.raises(ReleaseContractError, match="build of the national release"):
        recert.validate_recertification_candidate(moved, artifact_root=artifact_root)


# ---------------------------------------------------------------------------
# Which parents a recertification accepts.
# ---------------------------------------------------------------------------


def test_build_refuses_reusing_the_parent_id(parent, tmp_path, monkeypatch):
    tool = load_tool()
    monkeypatch.setattr(tool, "_producer_identity", lambda: dict(FIXTURE_PRODUCER))
    release_dir, artifact_root = parent
    with pytest.raises(ValueError, match="new bare US build ID"):
        tool.build_candidate(
            parent_release_dir=release_dir,
            parent_artifact_root=artifact_root,
            output_dir=tmp_path / "out",
            build_id=release_dir.name,
        )
    assert not (tmp_path / "out").exists()


def test_a_parent_that_keeps_only_its_tag_is_named_by_its_id_and_tag(parent):
    parent_dir, parent_root = parent
    described = recert.describe_parent(parent_dir, parent_root)
    assert (described["build_id"], described["revision"]) == (parent_dir.name,) * 2
    assert "release" not in described and "build" not in described
    assert referenced_build(described).revision == parent_dir.name
    assert referenced_build(described).folder == f"releases/{parent_dir.name}"


def test_a_build_folder_parent_is_named_by_its_release_and_build(folder_parent):
    parent_dir, parent_root = folder_parent
    described = recert.describe_parent(parent_dir, parent_root)
    assert (described["release"], described["build"]) == (
        US_NATIONAL_RELEASE_ID,
        FOLDER_PARENT_BUILD,
    )
    for field in ("revision", "build_id", "release_id"):
        assert field not in described
    location = referenced_build(described)
    assert (location.revision, location.folder) == (
        "main",
        f"releases/{US_NATIONAL_RELEASE_ID}/{FOLDER_PARENT_BUILD}",
    )
    assert described["dataset"]["filename"] == CHILD_DATASET_FILE
    assert set(described["root_artifacts"]) == {
        CHILD_DATASET_FILE,
        f"{CHILD_DATASET_KEY}_calibration.npz",
    }


def test_a_build_folder_parent_whose_artifact_pins_a_revision_is_refused(
    folder_parent,
):
    parent_dir, parent_root = folder_parent
    _rewrite_json(
        parent_dir / "release_manifest.json",
        lambda manifest: manifest["artifacts"]["calibration_diagnostics"].update(
            revision=FOLDER_PARENT_BUILD
        ),
    )
    with pytest.raises(ValueError, match="build folder"):
        recert.describe_parent(parent_dir, parent_root)


def test_a_child_of_a_build_folder_parent_is_a_build_of_the_national_release(
    folder_parent, tmp_path, monkeypatch
):
    parent_dir, parent_root = folder_parent
    release_dir, artifact_root = build_child(
        folder_parent, tmp_path / "folder-candidate", monkeypatch
    )
    report = recert.validate_recertification_candidate(
        release_dir, artifact_root=artifact_root
    )
    assert report["parent"] == recert.describe_parent(parent_dir, parent_root)
    assert report["parent_contract"]["status"] == recert.PARENT_CONTRACT_PASSED
    assert report["dataset"] == {
        "artifact": CHILD_DATASET_KEY,
        "filename": CHILD_DATASET_FILE,
        "sha256": report["parent"]["dataset"]["sha256"],
    }
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    build = json.loads((release_dir / "build_manifest.json").read_text())
    assert manifest["build"] == build["build"] == CHILD_ID
    assert manifest["recertified_from"] == build["recertified_from"]
    assert manifest["recertified_from"] == FOLDER_PARENT_BUILD
    assert build["calibration"]["parent_build_id"] == FOLDER_PARENT_BUILD
    assert (artifact_root / CHILD_DATASET_FILE).read_bytes() == (
        parent_root / CHILD_DATASET_FILE
    ).read_bytes()


def test_describe_parent_accepts_a_parent_that_declares_its_own_build_manifest(
    parent, tmp_path, monkeypatch
):
    """An existing build may list its build manifest; the child does not."""

    parent_dir, parent_root = parent
    _rewrite_json(
        parent_dir / "release_manifest.json",
        lambda manifest: manifest["artifacts"].update(
            build_manifest={
                "kind": "evidence",
                "path": "build_manifest.json",
                "repo_id": "policyengine/populace-us",
                "revision": parent_dir.name,
                "sha256": recert.sha256_file(parent_dir / "build_manifest.json"),
            }
        ),
    )
    validate_release_dir(parent_dir, artifact_root=parent_root)
    described = recert.describe_parent(parent_dir, parent_root)
    assert "build_manifest.json" not in described["release_files"]
    release_dir, artifact_root = build_child(
        parent, tmp_path / "declared-build-manifest", monkeypatch
    )
    recert.validate_recertification_candidate(release_dir, artifact_root=artifact_root)
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    assert "build_manifest" not in manifest["artifacts"]


def test_describe_parent_refuses_a_declared_build_manifest_of_other_bytes(parent):
    parent_dir, parent_root = parent
    _rewrite_json(
        parent_dir / "release_manifest.json",
        lambda manifest: manifest["artifacts"].update(
            build_manifest={
                "kind": "evidence",
                "path": "build_manifest.json",
                "repo_id": "policyengine/populace-us",
                "revision": parent_dir.name,
                "sha256": "0" * 64,
            }
        ),
    )
    with pytest.raises(ValueError, match="differs from its pin"):
        recert.describe_parent(parent_dir, parent_root)


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


@pytest.mark.parametrize("key", ["build_manifest", "compatibility_drift"])
def test_describe_parent_refuses_a_key_the_child_writes_its_evidence_under(parent, key):
    """Sol r1 finding 6: the child's entry would silently replace the artifact."""

    release_dir, artifact_root = parent

    def rename(manifest):
        manifest["artifacts"][key] = manifest["artifacts"].pop(
            "populace_us_2024_calibration"
        )

    _rewrite_json(release_dir / "release_manifest.json", rename)
    with pytest.raises(ValueError, match="child writes its own"):
        recert.describe_parent(release_dir, artifact_root)


@pytest.mark.parametrize(
    "path",
    [
        "releases/legacy/evidence.json",
        "./populace_us_2024.h5",
        "/abs.h5",
        "../../escaped.npz",
    ],
)
def test_describe_parent_refuses_paths_the_publisher_does_not_write(parent, path):
    """Sol r2 finding 4: a releases/ root artifact is never uploaded."""

    release_dir, artifact_root = parent
    _rewrite_json(
        release_dir / "release_manifest.json",
        lambda manifest: manifest["artifacts"]["populace_us_2024_calibration"].update(
            path=path
        ),
    )
    with pytest.raises(ValueError, match="not a release file or root artifact path"):
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
            build_id=CHILD_ID,
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


def test_the_substitution_does_not_excuse_an_unrelated_gate_disagreement(
    parent, tmp_path, monkeypatch
):
    """Sol r1 finding 7: only absent stored-input evidence is replaceable."""

    release_dir, artifact_root = parent
    _strip_stored_input_evidence(release_dir)
    _rewrite_json(
        release_dir / "build_manifest.json",
        lambda build: build.setdefault("gate_evidence", {}).update(
            reform_coverage_smoke="skipped"
        ),
    )
    _rewrite_json(
        release_dir / "release_manifest.json",
        lambda manifest: (
            manifest["build"]
            .setdefault("gate_evidence", {})
            .update(reform_coverage_smoke="bound")
        ),
    )
    tool = load_tool()
    monkeypatch.setattr(tool, "_producer_identity", lambda: dict(FIXTURE_PRODUCER))
    with pytest.raises(ValueError, match="different gate_evidence blocks"):
        tool.build_candidate(
            parent_release_dir=release_dir,
            parent_artifact_root=artifact_root,
            output_dir=tmp_path / "out",
            build_id=CHILD_ID,
        )


def test_the_substitution_compares_gate_blocks_as_written(
    parent, tmp_path, monkeypatch
):
    """Sol r2 finding 5: [] and {} disagree, as the evidence check says."""

    release_dir, artifact_root = parent
    _strip_stored_input_evidence(release_dir)
    _rewrite_json(
        release_dir / "build_manifest.json",
        lambda build: build.update(gate_evidence=[]),
    )
    _rewrite_json(
        release_dir / "release_manifest.json",
        lambda manifest: manifest["build"].update(gate_evidence={}),
    )
    tool = load_tool()
    monkeypatch.setattr(tool, "_producer_identity", lambda: dict(FIXTURE_PRODUCER))
    with pytest.raises(ValueError, match="fails the release contract"):
        tool.build_candidate(
            parent_release_dir=release_dir,
            parent_artifact_root=artifact_root,
            output_dir=tmp_path / "out",
            build_id=CHILD_ID,
        )


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
        build_id=CHILD_ID,
    )
    child = tmp_path / "candidate" / "releases" / US_NATIONAL_RELEASE_ID
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
            tmp_path / "certified" / US_NATIONAL_RELEASE_ID,
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
    "specifier",
    [
        ">=1.0,<2.61",
        ">=2.60.4",
        "==2.60.3",
        ">=2.60.4,!=3.0.0,!=99999.0.0",
        ">=2.60.4,<2.62",
        "~=2.60.4",
        "==2.60.*",
        ">2.60.3,<2.61",
    ],
)
def test_claims_outside_the_bounded_grammar_are_refused(candidate, specifier):
    with pytest.raises(ValueError, match="is not '==2.60.4'"):
        recert.claim_entries(
            _parent_manifest(candidate),
            tested_packages=TESTED_PACKAGES,
            specifier=specifier,
            declared_by=DECLARER,
        )


@pytest.mark.parametrize("specifier", ["==2.60.4", ">=2.60.0,<2.61", ">=2.60.4,<2.61"])
def test_claims_inside_the_bounded_grammar_are_accepted(candidate, specifier):
    models, _, claims = recert.claim_entries(
        _parent_manifest(candidate),
        tested_packages=TESTED_PACKAGES,
        specifier=specifier,
        declared_by=DECLARER,
    )
    assert claims["model"]["specifier"] == specifier == models[-1]["specifier"]


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
    manifest = json.loads((output / "release_manifest.json").read_text())
    assert output.name == US_NATIONAL_RELEASE_ID and manifest["build"] == CHILD_ID
    for key, path in recert.CERTIFIED_ARTIFACT_KEYS.items():
        assert manifest["artifacts"][key] == {
            "kind": "diagnostics",
            "path": path,
            "repo_id": "policyengine/populace-us",
            "sha256": recert.sha256_file(output / path),
            "role": key,
        }
    assert "build_manifest" not in manifest["artifacts"]


def test_a_child_of_a_build_folder_parent_certifies(
    folder_parent, tmp_path, monkeypatch
):
    child = build_child(folder_parent, tmp_path / "folder-candidate", monkeypatch)
    output, artifact_root = certify(child, tmp_path / "folder", monkeypatch)
    validate_release_dir(output, artifact_root=artifact_root)
    report = json.loads((output / recert.REPORT_FILE).read_text())
    assert report["compatibility"]["status"] == "passed"
    assert report["parent"]["build"] == FOLDER_PARENT_BUILD


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


def _combine(release_dir, **overrides):
    diagnostics = json.loads((release_dir / "calibration_diagnostics.json").read_text())
    report = json.loads((release_dir / recert.REPORT_FILE).read_text())
    arguments = {
        "calibration_diagnostics": diagnostics,
        "dataset_sha256": report["dataset"]["sha256"],
        "baseline": score_document(release_dir, packages=BASELINE_PACKAGES),
        "tested": score_document(release_dir),
        "counterfactual": None,
        "baseline_engine": {
            "model": {"version": "1.729.0"},
            "core": {"version": "3.19.0"},
        },
        "tested_packages": TESTED_PACKAGES,
        "newly_read_columns": [],
        "diagnostics_sha256": recert.sha256_file(
            release_dir / "calibration_diagnostics.json"
        ),
        "expected_households": HOUSEHOLDS,
    }
    arguments.update(overrides)
    return recert.combine_drift(**arguments)


def test_drift_refuses_a_baseline_on_another_engine(candidate):
    release_dir, _ = candidate
    with pytest.raises(ValueError, match="built-with policyengine-us"):
        _combine(release_dir, baseline=score_document(release_dir))


def test_drift_requires_the_counterfactual_when_columns_are_newly_read(candidate):
    release_dir, _ = candidate
    with pytest.raises(ValueError, match="counterfactual"):
        _combine(release_dir, newly_read_columns=["tax_unit_role_input"])


@pytest.mark.parametrize(
    "role, document_overrides, message",
    [
        ("tested", {"households": 2}, "cover all 3 households"),
        ("baseline", {"households": 0}, "cover all 3 households"),
        ("tested", {"spm": {"geography_kind": "state"}}, "post-export SPM selection"),
    ],
)
def test_drift_refuses_scores_that_did_not_cover_or_bind_the_parent(
    candidate, role, document_overrides, message
):
    release_dir, _ = candidate
    packages = BASELINE_PACKAGES if role == "baseline" else TESTED_PACKAGES
    document = score_document(release_dir, packages=packages, **document_overrides)
    with pytest.raises(ValueError, match=message):
        _combine(release_dir, **{role: document})


def test_drift_refuses_scores_of_other_diagnostics(candidate):
    release_dir, _ = candidate
    with pytest.raises(ValueError, match="carried parent diagnostics"):
        _combine(release_dir, diagnostics_sha256="0" * 64)


@pytest.mark.parametrize("field", ["content_sha256", "wheel_sha256"])
def test_drift_refuses_a_tested_runtime_the_replay_did_not_certify(candidate, field):
    release_dir, _ = candidate
    packages = {
        **TESTED_PACKAGES,
        "policyengine-us": {**TESTED_PACKAGES["policyengine-us"], field: "f" * 64},
    }
    with pytest.raises(ValueError, match="certified policyengine-us runtime"):
        _combine(release_dir, tested=score_document(release_dir, packages=packages))


def test_drift_refuses_a_runtime_not_proven_against_pypi(candidate):
    release_dir, _ = candidate
    document = score_document(release_dir, packages=BASELINE_PACKAGES)
    document["runtime"]["packages"]["policyengine-us"]["pypi_sha256_verified"] = False
    with pytest.raises(ValueError, match="PyPI publishes"):
        _combine(release_dir, baseline=document)


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
def test_drift_rows_decompose_exactly(candidate, base, new, withheld):
    release_dir, _ = candidate
    drift = _combine(
        release_dir,
        baseline=score_document(
            release_dir, packages=BASELINE_PACKAGES, value=lambda metric_id: base
        ),
        tested=score_document(release_dir, value=lambda metric_id: new),
        counterfactual=score_document(
            release_dir,
            variant=recert.VARIANT_WITHOUT_NEWLY_READ,
            withheld=("parent_1_id",),
            value=lambda metric_id: withheld,
        ),
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


def _publish(hub, output, artifact_root, monkeypatch):
    """Check the child as the build-folder publisher does, then its parent on
    the Hub, which must serve what the child carries before anything is
    handed over."""

    monkeypatch.setattr(recert, "run_recertification_loader_replay", fake_replay())
    monkeypatch.setattr(recert, "_check_producer", lambda code, failures, label: None)
    prepared = release_module.prepare_build_folder_release(
        output, artifact_root=artifact_root, repository="policyengine/populace-us"
    )
    recert.require_parent_publication(
        hub,
        prepared.release_dir,
        repo_id=prepared.repository,
        artifact_root=artifact_root,
    )
    return prepared


def test_publication_refuses_a_parent_the_hub_does_not_serve(
    paths_hub, parent, candidate, tmp_path, monkeypatch
):
    output, artifact_root = certify(candidate, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="is not published"):
        _publish(paths_hub, output, artifact_root, monkeypatch)
    assert paths_hub.events == []


def test_publication_refuses_a_parent_whose_published_manifest_differs(
    paths_hub, parent, candidate, tmp_path, monkeypatch
):
    parent_dir, parent_root = parent
    output, artifact_root = certify(candidate, tmp_path, monkeypatch)
    seed_published_parent(paths_hub, parent_dir, parent_root)
    paths_hub._commits["commit-parent"][
        f"releases/{parent_dir.name}/release_manifest.json"
    ] = b"{}"
    with pytest.raises(ValueError, match="differs from the carried"):
        _publish(paths_hub, output, artifact_root, monkeypatch)
    assert paths_hub.events == []


def test_publication_needs_a_hub_that_can_say_where_artifacts_are(
    hub, parent, candidate, tmp_path, monkeypatch
):
    parent_dir, parent_root = parent
    output, artifact_root = certify(candidate, tmp_path, monkeypatch)
    seed_published_parent(hub, parent_dir, parent_root)
    with pytest.raises(TypeError, match="get_paths_info"):
        _publish(hub, output, artifact_root, monkeypatch)
    assert hub.events == []


def test_publication_refuses_a_root_artifact_moved_into_the_release_directory(
    paths_hub, parent, candidate, tmp_path, monkeypatch
):
    """Sol r1 finding 3: an unchanged root NPZ reclassified as a release file."""

    parent_dir, parent_root = parent
    output, artifact_root = certify(candidate, tmp_path, monkeypatch)
    npz = "populace_us_2024_calibration.npz"
    shutil.copyfile(artifact_root / npz, output / npz)
    os.chmod(output / npz, 0o400)

    def reclassify(report):
        sha = report["parent"]["root_artifacts"].pop(npz)
        report["parent"]["release_files"][npz] = sha

    _rewrite_json(output / recert.REPORT_FILE, reclassify)
    _rewrite_json(
        output / "release_manifest.json",
        lambda manifest: manifest["artifacts"]["compatibility_recertification"].update(
            sha256=recert.sha256_file(output / recert.REPORT_FILE)
        ),
    )
    seed_published_parent(paths_hub, parent_dir, parent_root)
    with pytest.raises(ValueError, match="at both a release-file and a root"):
        _publish(paths_hub, output, artifact_root, monkeypatch)
    assert paths_hub.events == []


def test_publication_refuses_a_parent_that_publishes_an_artifact_twice(
    paths_hub, parent, candidate, tmp_path, monkeypatch
):
    """Sol r2 finding 3: a duplicate release-dir copy of a root artifact."""

    parent_dir, parent_root = parent
    output, artifact_root = certify(candidate, tmp_path, monkeypatch)
    seed_published_parent(paths_hub, parent_dir, parent_root)
    npz = "populace_us_2024_calibration.npz"
    paths_hub._commits["commit-parent"][f"releases/{parent_dir.name}/{npz}"] = (
        artifact_root / npz
    ).read_bytes()
    with pytest.raises(ValueError, match="at both a release-file and a root"):
        _publish(paths_hub, output, artifact_root, monkeypatch)
    assert paths_hub.events == []


def test_publication_refuses_a_root_artifact_the_parent_tag_holds_other_bytes_of(
    paths_hub, parent, candidate, tmp_path, monkeypatch
):
    parent_dir, parent_root = parent
    output, artifact_root = certify(candidate, tmp_path, monkeypatch)
    seed_published_parent(paths_hub, parent_dir, parent_root)
    paths_hub._commits["commit-parent"]["populace_us_2024.h5"] = b"other bytes"
    with pytest.raises(ValueError, match="is not the artifact the child carries"):
        _publish(paths_hub, output, artifact_root, monkeypatch)
    assert paths_hub.events == []


def test_publication_of_a_published_parents_child_publishes_its_build_folder(
    paths_hub, parent, candidate, tmp_path, monkeypatch
):
    parent_dir, parent_root = parent
    output, artifact_root = certify(candidate, tmp_path, monkeypatch)
    seed_published_parent(paths_hub, parent_dir, parent_root)
    prepared = _publish(paths_hub, output, artifact_root, monkeypatch)
    assert (prepared.release, prepared.build) == (US_NATIONAL_RELEASE_ID, CHILD_ID)
    files = {item.name: item for item in prepared.files}
    assert len(files) == len(prepared.files)
    dataset = files[CHILD_DATASET_FILE]
    assert dataset.role == "dataset"
    assert (
        dataset.path.read_bytes() == (parent_root / "populace_us_2024.h5").read_bytes()
    )
    assert "populace_us_2024.h5" not in files
    assert files[recert.DRIFT_FILE].path == output / recert.DRIFT_FILE
    assert files["build_manifest.json"].role == "build_manifest"
    # Checking the parent reads the Hub and writes nothing.
    assert paths_hub.events == [] and paths_hub.tags == []


def test_publication_reads_a_build_folder_parent_at_main(
    paths_hub, folder_parent, tmp_path, monkeypatch
):
    parent_dir, parent_root = folder_parent
    child = build_child(folder_parent, tmp_path / "folder-candidate", monkeypatch)
    output, artifact_root = certify(child, tmp_path / "folder", monkeypatch)
    # A child is never published before its parent.
    with pytest.raises(ValueError, match="is not published"):
        _publish(paths_hub, output, artifact_root, monkeypatch)
    seed_published_parent(paths_hub, parent_dir, parent_root)
    prepared = _publish(paths_hub, output, artifact_root, monkeypatch)
    assert prepared.build == CHILD_ID
    folder = f"releases/{US_NATIONAL_RELEASE_ID}/{FOLDER_PARENT_BUILD}"
    paths_hub.seed_main_file(f"{folder}/{CHILD_DATASET_FILE}", b"other bytes")
    with pytest.raises(ValueError, match="is not the artifact the child carries"):
        _publish(paths_hub, output, artifact_root, monkeypatch)
    paths_hub.seed_main_file(f"{folder}/release_manifest.json", b"{}")
    with pytest.raises(ValueError, match="differs from the carried"):
        _publish(paths_hub, output, artifact_root, monkeypatch)
    assert paths_hub.events == []


@pytest.mark.parametrize(
    ("seeding", "refusal"),
    [
        ({"withdrawn": True}, "has been withdrawn"),
        ({"listed": False}, "does not list"),
    ],
    ids=["withdrawn", "unlisted"],
)
def test_publication_refuses_a_build_folder_parent_releases_json_does_not_offer(
    paths_hub, folder_parent, tmp_path, monkeypatch, seeding, refusal
):
    """A recertification republishes its parent's data under a wider claim,
    so a parent its repository has withdrawn, or no longer lists, is refused
    before anything is handed over."""
    parent_dir, parent_root = folder_parent
    child = build_child(folder_parent, tmp_path / "folder-candidate", monkeypatch)
    output, artifact_root = certify(child, tmp_path / "folder", monkeypatch)
    seed_published_parent(paths_hub, parent_dir, parent_root, **seeding)
    try:
        _publish(paths_hub, output, artifact_root, monkeypatch)
    except ValueError as error:
        outcome = str(error)
    else:
        outcome = "published"
    assert refusal in outcome
    assert paths_hub.events == []


def test_the_publication_contract_takes_wheels_but_not_a_parent_h5(
    candidate, tmp_path, monkeypatch
):
    output, artifact_root = certify(candidate, tmp_path, monkeypatch)
    monkeypatch.setattr(recert, "run_recertification_loader_replay", fake_replay())
    monkeypatch.setattr(recert, "_check_producer", lambda code, failures, label: None)
    wheel = tmp_path / "policyengine_us-2.60.4-py3-none-any.whl"
    wheel.write_bytes(b"")
    with pytest.raises(ReleaseContractError, match="takes no parent_h5"):
        validate_release_dir(
            output,
            artifact_root=artifact_root,
            parent_h5=artifact_root / CHILD_DATASET_FILE,
            compatibility_wheels=(wheel,),
        )
    assert (
        validate_release_dir(
            output, artifact_root=artifact_root, compatibility_wheels=(wheel,)
        )
        == []
    )


def test_a_recertification_never_moves_latest(
    paths_hub, parent, candidate, tmp_path, monkeypatch
):
    """The tagged publisher, which writes latest.json, refuses the child: it
    is a build of a release that publishes into build folders."""

    parent_dir, parent_root = parent
    output, artifact_root = certify(candidate, tmp_path, monkeypatch)
    seed_published_parent(paths_hub, parent_dir, parent_root)
    monkeypatch.setattr(recert, "run_recertification_loader_replay", fake_replay())
    monkeypatch.setattr(recert, "_check_producer", lambda code, failures, label: None)
    wheel = tmp_path / "policyengine_us-2.60.4-py3-none-any.whl"
    wheel.write_bytes(b"")
    with pytest.raises(ValueError, match="publishes each build into its own folder"):
        release_module.prepare_release(
            output, artifact_root=artifact_root, compatibility_wheels=(wheel,)
        )
    events = list(paths_hub.events)
    with pytest.raises(ValueError, match="publishes each build into its own folder"):
        release_module.publish_release(
            output,
            "policyengine/populace-us",
            api=paths_hub,
            artifact_root=artifact_root,
            compatibility_wheels=(wheel,),
            notify=False,
        )
    assert paths_hub.events == events
    assert "latest.json" not in dict(paths_hub.uploads)


def test_validation_prints_the_recertification_measurement(
    candidate, tmp_path, monkeypatch, capsys
):
    output, artifact_root = certify(candidate, tmp_path, monkeypatch)
    monkeypatch.setattr(recert, "run_recertification_loader_replay", fake_replay())
    monkeypatch.setattr(recert, "_check_producer", lambda code, failures, label: None)
    wheel = tmp_path / "policyengine_us-2.60.4-py3-none-any.whl"
    wheel.write_bytes(b"")
    status = recert.main(
        [
            "--release-dir",
            str(output),
            "--artifact-root",
            str(artifact_root),
            "--require-compatibility",
            "--compatibility-wheel",
            str(wheel),
        ]
    )
    captured = capsys.readouterr()
    assert status == 0, captured.err
    verdict = json.loads(captured.out.strip().splitlines()[-1])
    assert verdict["valid"] is True and verdict["compatibility"] == "passed"
    assert verdict["claim_measurement"]["model"]["measured_versions"] == ["2.60.4"]


# ---------------------------------------------------------------------------
# Runtime identity (Sol r1 findings 1 and 8).
# ---------------------------------------------------------------------------


import importlib.util  # noqa: E402
from types import SimpleNamespace  # noqa: E402


def _fake_install(root, name, version):
    """An installed distribution and its wheel, sharing every package byte."""

    from zipfile import ZipFile

    module = name.replace("-", "_")
    dist_info = f"{module}-{version}.dist-info"
    files = {
        f"{module}/__init__.py": b"VALUE = 1\n",
        f"{module}/parameters/rate.yaml": b"values:\n  2024-01-01: 0.1\n",
        f"{dist_info}/METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode()
        ),
    }
    site = root / "site"
    for relative, content in files.items():
        (site / relative).parent.mkdir(parents=True, exist_ok=True)
        (site / relative).write_bytes(content)
    (site / dist_info / "uv_cache.json").write_text("{}")
    record = [*files, f"{dist_info}/uv_cache.json", f"{dist_info}/RECORD"]
    (site / dist_info / "RECORD").write_text("".join(f"{path},,\n" for path in record))
    wheel = root / f"{module}-{version}-py3-none-any.whl"
    with ZipFile(wheel, "w") as archive:
        for relative, content in files.items():
            archive.writestr(relative, content)
        archive.writestr(f"{dist_info}/RECORD", "")
    return site / dist_info, wheel


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    from importlib import metadata

    installs = {
        name: _fake_install(tmp_path, name, version)
        for name, version in (
            ("policyengine-us", "2.60.4"),
            ("policyengine-core", "3.33.3"),
            ("spm-calculator", "1.0.0"),
        )
    }
    monkeypatch.setattr(
        metadata,
        "distribution",
        lambda name: metadata.PathDistribution(installs[name][0]),
    )
    site = tmp_path / "site"
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(
            origin=str(site / name / "__init__.py"),
            submodule_search_locations=[str(site / name)],
        ),
    )
    monkeypatch.setattr(recert, "_VERIFIED_TREES", {})
    _forget_engine_modules(monkeypatch)
    return site, tuple(wheel for _, wheel in installs.values())


def _forget_engine_modules(monkeypatch):
    """Hide any real engine modules this test process imported earlier."""

    import sys

    for name in list(sys.modules):
        if name.split(".", 1)[0] in recert.IMPORT_NAMES.values():
            monkeypatch.delitem(sys.modules, name)


def test_runtime_identity_covers_parameters_and_ignores_installer_files(fake_runtime):
    site, wheels = fake_runtime
    identities = recert.runtime_identities(wheels, require_wheels=True)
    assert identities["policyengine-us"]["files"] == 3
    assert identities["policyengine-us"]["wheel_sha256"] == recert.sha256_file(
        wheels[0]
    )
    assert "direct_url" not in identities["policyengine-us"]
    before = identities["policyengine-us"]["content_sha256"]
    (site / "policyengine_us-2.60.4.dist-info" / "uv_cache.json").write_text('{"x": 1}')
    assert (
        recert.runtime_identities(wheels, require_wheels=True)["policyengine-us"][
            "content_sha256"
        ]
        == before
    )


def test_a_patched_parameter_file_is_not_the_wheels_runtime(fake_runtime):
    site, wheels = fake_runtime
    (site / "policyengine_us" / "parameters" / "rate.yaml").write_bytes(
        b"values:\n  2024-01-01: 0.2\n"
    )
    with pytest.raises(ValueError, match="rate.yaml differs from its wheel"):
        recert.runtime_identities(wheels, require_wheels=True)
    unbound = recert.runtime_identities((), require_wheels=False)
    assert unbound["policyengine-us"]["wheel_sha256"] is None


def test_the_runtime_needs_all_three_wheels(fake_runtime):
    _, wheels = fake_runtime
    with pytest.raises(ValueError, match="all three wheels"):
        recert.runtime_identities(wheels[:2], require_wheels=True)


def test_an_unrecorded_module_in_the_package_tree_is_refused(fake_runtime):
    """Sol r2 finding 1: Core imports every variable module it finds."""

    site, wheels = fake_runtime
    (site / "policyengine_us" / "z_extra.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="never recorded"):
        recert.runtime_identities(wheels, require_wheels=True)


def test_a_symlinked_overlay_package_is_refused(fake_runtime, monkeypatch, tmp_path):
    """Sol r4 finding 1: the overlay's __init__ resolves into the real tree."""

    site, wheels = fake_runtime
    overlay = tmp_path / "overlay" / "policyengine_core"
    overlay.mkdir(parents=True)
    (overlay / "__init__.py").symlink_to(site / "policyengine_core" / "__init__.py")

    def find_spec(name):
        if name == "policyengine_core":
            return SimpleNamespace(
                origin=str(overlay / "__init__.py"),
                submodule_search_locations=[str(overlay)],
            )
        return SimpleNamespace(
            origin=str(site / name / "__init__.py"),
            submodule_search_locations=[str(site / name)],
        )

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    with pytest.raises(ValueError, match="does not resolve to the installed"):
        recert.runtime_identities(wheels, require_wheels=True)


def test_a_loaded_engine_module_outside_its_tree_is_refused_by_name(
    fake_runtime, monkeypatch, tmp_path
):
    import sys
    import types

    site, wheels = fake_runtime
    recert.runtime_identities(wheels, require_wheels=True)
    package = types.ModuleType("policyengine_core")
    package.__file__ = str(site / "policyengine_core" / "__init__.py")
    package.__path__ = [str(tmp_path / "overlay" / "policyengine_core")]
    monkeypatch.setitem(sys.modules, "policyengine_core", package)
    with pytest.raises(RuntimeError, match="not from the installed policyengine-core"):
        recert.verify_loaded_modules()
    package.__path__ = [str(site / "policyengine_core")]
    helper = types.ModuleType("policyengine_core.simulations.microsimulation")
    helper.__file__ = str(tmp_path / "overlay" / "microsimulation.py")
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    with pytest.raises(RuntimeError, match="not from the installed policyengine-core"):
        recert.verify_loaded_modules()


def _namespace_module(name, location):
    """A module as Python imports a directory without ``__init__.py``."""

    from importlib.machinery import ModuleSpec, NamespaceLoader, PathFinder

    loader = NamespaceLoader(name, [str(location)], PathFinder._get_spec)
    spec = ModuleSpec(name, loader, is_package=True)
    spec.submodule_search_locations = [str(location)]
    return importlib.util.module_from_spec(spec)


def test_a_namespace_package_is_checked_by_its_path(
    fake_runtime, monkeypatch, tmp_path
):
    """The real-runtime smoke: policyengine_us.parameters.gov has no file."""

    import sys
    import types

    site, wheels = fake_runtime
    recert.runtime_identities(wheels, require_wheels=True)
    package = types.ModuleType("policyengine_us")
    package.__file__ = str(site / "policyengine_us" / "__init__.py")
    package.__path__ = [str(site / "policyengine_us")]
    monkeypatch.setitem(sys.modules, "policyengine_us", package)
    inside = _namespace_module(
        "policyengine_us.parameters", site / "policyengine_us" / "parameters"
    )
    assert getattr(inside, "__file__", None) is None
    assert recert._is_namespace_package(inside)
    monkeypatch.setitem(sys.modules, inside.__name__, inside)
    assert recert.verify_loaded_modules() >= 1
    outside = _namespace_module(
        "policyengine_us.reforms", tmp_path / "overlay" / "reforms"
    )
    monkeypatch.setitem(sys.modules, outside.__name__, outside)
    with pytest.raises(RuntimeError, match="not from the installed policyengine-us"):
        recert.verify_loaded_modules()
    monkeypatch.delitem(sys.modules, outside.__name__)
    # No file and no namespace loader: code that came from somewhere unchecked.
    in_memory = types.ModuleType("policyengine_us.generated")
    in_memory.__path__ = [str(site / "policyengine_us")]
    monkeypatch.setitem(sys.modules, in_memory.__name__, in_memory)
    with pytest.raises(RuntimeError, match="not from the installed policyengine-us"):
        recert.verify_loaded_modules()


def test_the_engine_must_not_run_before_its_runtime_is_authenticated(
    fake_runtime, monkeypatch
):
    """Sol r4 finding 2: Core runs every variable source on first import."""

    import sys
    import types

    _, wheels = fake_runtime
    monkeypatch.setitem(
        sys.modules, "policyengine_us", types.ModuleType("policyengine_us")
    )
    with pytest.raises(RuntimeError, match="imported before its runtime"):
        recert.authenticate_runtime_before_import(wheels, require_wheels=True)


def test_an_import_that_resolves_elsewhere_is_refused(
    fake_runtime, monkeypatch, tmp_path
):
    _, wheels = fake_runtime
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(
            origin=str(tmp_path / "shadow" / name / "__init__.py"),
            submodule_search_locations=[str(tmp_path / "shadow" / name)],
        ),
    )
    with pytest.raises(ValueError, match="does not resolve to the installed"):
        recert.runtime_identities(wheels, require_wheels=True)


def test_bytecode_isolation_requires_a_prefix_and_modules_compiled_under_it(
    monkeypatch, tmp_path
):
    """Sol r2/r3: a replaced .pyc would run instead of the authenticated source."""

    import sys
    import types

    monkeypatch.setattr(sys, "pycache_prefix", None)
    assert recert.bytecode_isolation_status()["isolated"] is False
    with pytest.raises(RuntimeError, match="PYTHONPYCACHEPREFIX"):
        recert.require_trusted_bytecode()

    prefix = tmp_path / "prefix"
    prefix.mkdir()
    monkeypatch.setattr(sys, "pycache_prefix", str(prefix))
    for name in list(sys.modules):
        if name.split(".", 1)[0] in {
            "microcosm",
            "tools",
            *recert.IMPORT_NAMES.values(),
        }:
            monkeypatch.delitem(sys.modules, name)
    module = types.ModuleType("microcosm.data.fake")
    module.__file__ = str(tmp_path / "fake.py")
    module.__cached__ = str(tmp_path / "__pycache__" / "fake.cpython-313.pyc")
    monkeypatch.setitem(sys.modules, "microcosm.data.fake", module)
    status = recert.bytecode_isolation_status()
    assert status == {"isolated": False, "offending": ["microcosm.data.fake"]}
    module.__cached__ = str(prefix / "fake.cpython-313.pyc")
    assert recert.bytecode_isolation_status()["isolated"] is True
    module.__file__ = str(tmp_path / "fake.pyc")
    assert recert.bytecode_isolation_status()["isolated"] is False


def test_the_bootstrap_keeps_a_startup_prefix_and_otherwise_makes_one(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "pycache_prefix", "/already/set")
    assert recert.bootstrap_bytecode_isolation() == "/already/set"
    monkeypatch.setattr(sys, "pycache_prefix", None)
    prefix = recert.bootstrap_bytecode_isolation()
    assert sys.pycache_prefix == prefix and not os.listdir(prefix)


def test_loaded_engine_modules_are_matched_by_file_not_name(fake_runtime, monkeypatch):
    """Sol r3 finding 1: Core registers variable modules under synthetic names."""

    import sys
    import types

    site, wheels = fake_runtime
    recert.runtime_identities(wheels, require_wheels=True)
    verified = types.ModuleType("us_system.variables.rate")
    verified.__file__ = str(site / "policyengine_us" / "__init__.py")
    monkeypatch.setitem(sys.modules, "us_system.variables.rate", verified)
    assert recert.verify_loaded_modules() >= 1
    stray = types.ModuleType("us_system.variables.extra")
    stray.__file__ = str(site / "policyengine_us" / "extra.py")
    monkeypatch.setitem(sys.modules, "us_system.variables.extra", stray)
    with pytest.raises(RuntimeError, match="not a verified policyengine-us source"):
        recert.verify_loaded_modules()


def test_source_hidden_in_pycache_is_unrecorded(fake_runtime):
    """Sol r3 finding 1: Core's directory glob descends into __pycache__/."""

    site, wheels = fake_runtime
    cache = site / "policyengine_us" / "__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-313.pyc").write_bytes(b"real bytecode is excluded")
    recert.runtime_identities(wheels, require_wheels=True)
    (cache / "extra.py").write_text("VALUE = 3\n")
    with pytest.raises(ValueError, match="never recorded"):
        recert.runtime_identities(wheels, require_wheels=True)


def test_a_pyc_beside_the_sources_is_unrecorded(fake_runtime):
    site, wheels = fake_runtime
    (site / "policyengine_us" / "extra.pyc").write_bytes(b"sourceless module")
    with pytest.raises(ValueError, match="never recorded"):
        recert.runtime_identities(wheels, require_wheels=True)


def test_a_symlink_in_the_package_tree_is_refused(fake_runtime, tmp_path):
    site, wheels = fake_runtime
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "extra.py").write_text("VALUE = 4\n")
    (site / "policyengine_us" / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink in the installed package tree"):
        recert.runtime_identities(wheels, require_wheels=True)


def test_the_scorer_refuses_a_nonpositive_batch_size_before_any_work(
    candidate, tmp_path
):
    tool = load_tool()
    release_dir, artifact_root = candidate
    for size in (0, -1):
        with pytest.raises(ValueError, match="positive integer"):
            tool.score_candidate(
                release_dir=release_dir,
                artifact_root=artifact_root,
                output=tmp_path / f"scores-{size}.json",
                variant=recert.VARIANT_AS_STORED,
                compatibility_wheels=(),
                batch_households=size,
            )


@pytest.mark.parametrize("build_id", ["/tmp/outside", "../outside", "not-us"])
def test_fetch_parent_refuses_an_id_before_creating_anything(tmp_path, build_id):
    tool = load_tool()
    with pytest.raises(ValueError, match="is not a US build ID"):
        tool.fetch_parent(build_id=build_id, output_dir=tmp_path / "parent")
    assert not (tmp_path / "parent").exists()


@pytest.mark.parametrize("release", ["../outside", "populace-us-2024-national"])
def test_fetch_parent_refuses_a_release_before_creating_anything(tmp_path, release):
    tool = load_tool()
    with pytest.raises(ValueError, match="build folders"):
        tool.fetch_parent(
            build_id=FOLDER_PARENT_BUILD,
            release=release,
            output_dir=tmp_path / "parent",
        )
    assert not (tmp_path / "parent").exists()


def test_fetch_parent_reads_a_tagged_parent_at_its_tag(paths_hub, parent, tmp_path):
    parent_dir, parent_root = parent
    seed_published_parent(paths_hub, parent_dir, parent_root)
    tool = load_tool()
    output = tmp_path / "fetched"
    described = tool.fetch_parent(
        build_id=parent_dir.name, output_dir=output, api=paths_hub
    )
    assert described == recert.describe_parent(parent_dir, parent_root)
    fetched = output / "releases" / parent_dir.name
    assert sorted(path.name for path in fetched.iterdir()) == sorted(
        [
            "release_manifest.json",
            "build_manifest.json",
            *described["release_files"],
        ]
    )
    assert sorted(path.name for path in (output / "artifacts").iterdir()) == sorted(
        described["root_artifacts"]
    )


def test_fetch_parent_reads_a_build_folder_parent_at_main(
    paths_hub, folder_parent, tmp_path
):
    parent_dir, parent_root = folder_parent
    seed_published_parent(paths_hub, parent_dir, parent_root)
    tool = load_tool()
    output = tmp_path / "fetched"
    described = tool.fetch_parent(
        build_id=FOLDER_PARENT_BUILD,
        release=US_NATIONAL_RELEASE_ID,
        output_dir=output,
        api=paths_hub,
    )
    assert described == recert.describe_parent(parent_dir, parent_root)
    # The parent's directory is named after its release, as its contract needs.
    fetched = output / "releases" / US_NATIONAL_RELEASE_ID
    assert sorted(path.name for path in fetched.iterdir()) == sorted(
        [
            "release_manifest.json",
            "build_manifest.json",
            *described["release_files"],
        ]
    )
    assert sorted(path.name for path in (output / "artifacts").iterdir()) == sorted(
        described["root_artifacts"]
    )
    validate_release_dir(fetched, artifact_root=output / "artifacts")
