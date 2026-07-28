import builtins
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from populace.calibrate import TargetRegistry, TargetSpec, calibrate
from populace.frame import Frame, WeightKind, Weights


def _load_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_scorer_module():
    root = Path(__file__).resolve().parents[3]
    tools_path = str(root / "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    path = root / "tools" / "score_us_fiscal_targets.py"
    spec = importlib.util.spec_from_file_location("score_us_fiscal_targets", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test__given_matching_warm_start_npz__then_builder_loads_household_weights(
    tmp_path,
) -> None:
    builder = _load_builder_module()
    path = tmp_path / "populace_us_2024_calibration.npz"
    initial = np.asarray([10.0, 20.0, 30.0])
    weights = np.asarray([12.0, 18.0, 35.0])
    np.savez_compressed(
        path,
        household_weight=weights,
        initial_household_weight=initial,
    )

    loaded, payload = builder._load_warm_start_calibration_npz(
        path,
        expected_initial_weights=initial.copy(),
    )

    np.testing.assert_allclose(loaded, weights)
    assert payload["enabled"] is True
    assert payload["n_households"] == 3
    assert payload["sha256"] == builder._sha256(path)


def test_final_household_weight_evidence_round_trips_exact_vector(tmp_path) -> None:
    builder = _load_builder_module()
    import hashlib as _hashlib

    import pandas as _pd

    weights = Weights(
        np.asarray([0.0, 12.0, 35.0]),
        WeightKind.CALIBRATED,
    )
    ids = np.asarray([101, 202, 303], dtype="int64")
    frame = SimpleNamespace(
        n=lambda entity: 3 if entity == "household" else None,
        weights_for=lambda entity: weights if entity == "household" else None,
        table=lambda entity: _pd.DataFrame({"household_id": ids}),
    )
    identity = {"base_dataset_sha256": "probe-base", "seed": 0}

    metadata = builder._write_final_household_weight_evidence(
        tmp_path, frame, identity=identity
    )

    values_path = tmp_path / builder.FINAL_HOUSEHOLD_WEIGHTS_FILENAME
    ids_path = tmp_path / builder.FINAL_HOUSEHOLD_WEIGHT_IDS_FILENAME
    metadata_path = tmp_path / builder.FINAL_HOUSEHOLD_WEIGHTS_METADATA_FILENAME
    np.testing.assert_array_equal(
        np.load(values_path, allow_pickle=False),
        np.asarray([0.0, 12.0, 35.0]),
    )
    np.testing.assert_array_equal(np.load(ids_path, allow_pickle=False), ids)
    assert json.loads(metadata_path.read_text()) == metadata
    assert metadata == {
        "artifact_kind": "populace_final_household_weight_evidence",
        "schema_version": 1,
        "measurement_phase": "release_final",
        "entity": "household",
        "weight_kind": "calibrated",
        "identity": {"base_dataset_sha256": "probe-base", "seed": 0},
        "values": {
            "file": "final_household_weights.npy",
            "dtype": "float64",
            "shape": [3],
            "sha256": builder._sha256(values_path),
        },
        "household_ids": {
            "file": "final_household_weight_ids.npy",
            "dtype": "int64",
            "shape": [3],
            "sha256": builder._sha256(ids_path),
            "ordering_sha256": _hashlib.sha256(ids.tobytes()).hexdigest(),
        },
        "summary": {
            "n_households": 3,
            "household_weight_sum": 47.0,
            "minimum": 0.0,
            "maximum": 35.0,
            "nonzero_count": 2,
            "zero_count": 1,
        },
    }
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_certified_release_dir_refusal_precedes_all_side_effects() -> None:
    """populace#568 round 4: refusal placement is part of the contract —
    one refusal call must precede the base download (the --release-id
    path), and one must precede every output-directory mkdir (the
    auto-generated-id path)."""
    import ast

    builder = _load_builder_module()
    tree = ast.parse(Path(builder.__file__).read_text())
    main_fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    refusals = [
        n
        for n in ast.walk(main_fn)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", "") == "_refuse_certified_release_dir_reuse"
    ]
    downloads = [
        n
        for n in ast.walk(main_fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_download_base_h5"
    ]
    mkdirs = [
        n
        for n in ast.walk(main_fn)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "mkdir"
    ]
    assert len(refusals) == 2
    assert downloads and min(r.lineno for r in refusals) < min(
        d.lineno for d in downloads
    ), "the known-id refusal must precede the base download"
    assert mkdirs and sorted(r.lineno for r in refusals)[1] < min(
        m.lineno for m in mkdirs
    ), "the unconditional refusal must precede every output mkdir"


def test_certified_release_dir_reuse_is_refused(tmp_path) -> None:
    """populace#568 round 3: a run pointed at a directory already carrying
    release_manifest.json must refuse before writing anything — a failed
    retry would otherwise mix its weight evidence with the prior certified
    release."""
    builder = _load_builder_module()
    release_dir = tmp_path / "releases" / "some-id"
    release_dir.mkdir(parents=True)
    builder._refuse_certified_release_dir_reuse(release_dir)  # absent: fine
    (release_dir / "release_manifest.json").write_text("{}")
    with pytest.raises(RuntimeError, match="already carries a certified"):
        builder._refuse_certified_release_dir_reuse(release_dir)


def test_final_household_weight_evidence_writes_only_on_gate_failure_path() -> None:
    """populace#568 review blocker 2: the evidence pair must be written on
    the batched gate-failure path ONLY — green runs carry weights in the
    certified H5. Enforced structurally (the #443 AST-guard pattern): the
    sole main() call site must sit inside the ``if terminal_gate_failures:``
    branch, before its raise."""
    import ast

    builder = _load_builder_module()
    source = Path(builder.__file__).read_text()
    tree = ast.parse(source)
    calls: list[tuple[ast.Call, list[ast.AST]]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[ast.AST] = []

        def generic_visit(self, node):
            self.stack.append(node)
            super().generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node):
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if name == "_write_final_household_weight_evidence":
                calls.append((node, list(self.stack)))
            self.generic_visit(node)

    Visitor().visit(tree)
    main_calls = [
        (node, stack)
        for node, stack in calls
        if any(isinstance(anc, ast.FunctionDef) and anc.name == "main" for anc in stack)
    ]
    assert len(main_calls) == 1
    call_node, stack = main_calls[0]
    guarding_ifs = [
        anc
        for anc in stack
        if isinstance(anc, ast.If)
        and isinstance(anc.test, ast.Name)
        and anc.test.id == "terminal_gate_failures"
    ]
    assert guarding_ifs, (
        "final-household-weight evidence must be written inside the "
        "terminal_gate_failures branch only"
    )
    guard = guarding_ifs[-1]
    # The call must sit in the IF BODY (an else-branch call would run on
    # green runs) and strictly before the branch's raise.
    body_nodes = [n for stmt in guard.body for n in ast.walk(stmt)]
    assert call_node in body_nodes, (
        "evidence call must be in the terminal_gate_failures if-body, "
        "not its else branch"
    )
    raises = [n for n in body_nodes if isinstance(n, ast.Raise)]
    assert raises and call_node.lineno < min(r.lineno for r in raises), (
        "evidence must be persisted before the batched raise"
    )
    # The green continuation must clean up a prior failed attempt's
    # evidence (release-dir reuse, populace#568 round 2) before the
    # certified dataset write.
    main_fn = next(
        anc for anc in stack if isinstance(anc, ast.FunctionDef) and anc.name == "main"
    )

    def _is_bound_cleanup_for(node):
        if not isinstance(node, ast.For) or node.lineno <= guard.end_lineno:
            return False
        # The iterable must BE a tuple (no slicing/subscript tricks) whose
        # elements name all three evidence filename constants.
        if not isinstance(node.iter, ast.Tuple):
            return False
        iter_names = {
            name.id for name in ast.walk(node.iter) if isinstance(name, ast.Name)
        }
        if not iter_names >= {
            "FINAL_HOUSEHOLD_WEIGHTS_FILENAME",
            "FINAL_HOUSEHOLD_WEIGHT_IDS_FILENAME",
            "FINAL_HOUSEHOLD_WEIGHTS_METADATA_FILENAME",
        }:
            return False
        # The unlink must be called ON THE LOOP TARGET, not a decoy.
        target = node.target.id if isinstance(node.target, ast.Name) else None
        return target is not None and any(
            isinstance(c, ast.Call)
            and getattr(c.func, "attr", "") == "unlink"
            and isinstance(getattr(c.func, "value", None), ast.Name)
            and c.func.value.id == target
            for stmt in node.body
            for c in ast.walk(stmt)
        )

    cleanup_unlinks = [n for n in ast.walk(main_fn) if _is_bound_cleanup_for(n)]
    dataset_writes = [
        n
        for n in ast.walk(main_fn)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", "") == "write_dataset"
        and n.lineno > guard.end_lineno
    ]
    assert cleanup_unlinks and dataset_writes, (
        "green path must unlink ALL THREE stale evidence files (bound by "
        "constant name) and write the dataset"
    )
    assert min(n.lineno for n in cleanup_unlinks) < min(
        n.lineno for n in dataset_writes
    ), "stale-evidence cleanup must precede the certified dataset write"


def test__given_mismatched_warm_start_initial_weights__then_builder_rejects_npz(
    tmp_path,
) -> None:
    builder = _load_builder_module()
    path = tmp_path / "populace_us_2024_calibration.npz"
    np.savez_compressed(
        path,
        household_weight=np.asarray([12.0, 18.0, 35.0]),
        initial_household_weight=np.asarray([10.0, 20.0, 30.0]),
    )

    with pytest.raises(ValueError, match="different initial household weights"):
        builder._load_warm_start_calibration_npz(
            path,
            expected_initial_weights=np.asarray([10.0, 20.0, 31.0]),
        )


def test__given_target_frame_checkpoint__then_builder_round_trips_frame(
    monkeypatch,
    tmp_path,
    small_frame,
) -> None:
    builder = _load_builder_module()
    monkeypatch.setattr(builder, "US_SCHEMA", small_frame.schema)
    tables = {
        entity: small_frame.table(entity).copy() for entity in small_frame.entities
    }
    tables["household"]["mock_measure"] = np.asarray([1.5, 2.5])
    tables["household"]["mock_filter"] = np.asarray([1, 0], dtype=np.int64)
    frame = Frame(
        tables,
        small_frame.schema,
        {"household": small_frame.weights_for("household")},
        small_frame.strata,
    )
    target = TargetSpec(
        name="mock.measure",
        entity="household",
        measure="mock_measure",
        filter="mock_filter",
        value=1500.0,
        source="Mock source",
    )
    identity = builder._target_frame_checkpoint_identity(
        base_dataset_sha256="base-sha",
        policyengine_us_version="1.2.3",
        seed=0,
        target_period=builder.PERIOD,
        target_registry_version="registry-sha",
        weeks_unemployed_source_sha256="weeks-source-sha",
        congressional_district_vintage_crosswalk_sha256="crosswalk-sha",
        ssi_take_up_assignment_sha256="ssi-flags-sha",
        selection_identities_sha256=None,
    )
    # 10 = #557 preserves the staged retirement surface through release
    # materialization; pre-#557 QRF-refitted checkpoints (9) must not serve.
    assert identity["materializer_version"] == 10
    # The SSI prior-weight basis is identity-bearing (populace#543 instance
    # 2): unflagged runs carry the key as None.
    assert identity["ssi_take_up_prior_weight_basis_sha256"] is None
    assert identity["weeks_unemployed_source_sha256"] == "weeks-source-sha"
    path = tmp_path / "target_frame_checkpoint.h5"

    payload = builder._write_target_frame_checkpoint(
        path,
        frame=frame,
        identity=identity,
        compilation={"declared_targets": 1},
    )
    loaded = builder._read_target_frame_checkpoint(
        path,
        identity=identity,
        target_specs=(target,),
    )

    assert payload["status"] == "miss_written"
    assert loaded is not None
    loaded_frame, loaded_registry, loaded_compilation = loaded
    assert np.array_equal(
        loaded_frame.table("household")["mock_measure"].to_numpy(),
        np.asarray([1.5, 2.5]),
    )
    assert np.array_equal(
        loaded_frame.table("household")["mock_filter"].to_numpy(),
        np.asarray([1, 0], dtype=np.int64),
    )
    assert np.array_equal(
        loaded_frame.weights_for("household").values,
        small_frame.weights_for("household").values,
    )
    assert loaded_frame.weights_for("household").kind is WeightKind.DESIGN
    pd.testing.assert_series_equal(
        loaded_frame.strata,
        small_frame.strata,
        check_dtype=False,
    )
    assert len(loaded_registry) == 1
    assert loaded_compilation["compiled_candidate_targets"] == 1
    assert loaded_compilation["target_frame_checkpoint"]["status"] == "hit"
    assert (
        loaded_compilation["target_frame_checkpoint"]["stored_compilation"][
            "declared_targets"
        ]
        == 1
    )


def test__given_stale_materializer_version_checkpoint__then_builder_rejects_it(
    monkeypatch,
    tmp_path,
    small_frame,
) -> None:
    """A checkpoint stored under a superseded materializer version must not load.

    #557 changed the staged retirement-surface semantics: version-9
    checkpoints can carry release-refitted leaves instead of the preserved
    support-built surface. The version constant participates in the identity
    comparison; this pins the stored-9 versus current-10 rejection directly.
    """
    builder = _load_builder_module()
    monkeypatch.setattr(builder, "US_SCHEMA", small_frame.schema)
    tables = {
        entity: small_frame.table(entity).copy() for entity in small_frame.entities
    }
    tables["household"]["mock_measure"] = np.asarray([1.5, 2.5])
    tables["household"]["mock_filter"] = np.asarray([1, 0], dtype=np.int64)
    frame = Frame(
        tables,
        small_frame.schema,
        {"household": small_frame.weights_for("household")},
        small_frame.strata,
    )
    target = TargetSpec(
        name="mock.measure",
        entity="household",
        measure="mock_measure",
        filter="mock_filter",
        value=1500.0,
        source="Mock source",
    )
    identity = builder._target_frame_checkpoint_identity(
        base_dataset_sha256="base-sha",
        policyengine_us_version="1.2.3",
        seed=0,
        target_period=builder.PERIOD,
        target_registry_version="registry-sha",
        weeks_unemployed_source_sha256="weeks-source-sha",
        congressional_district_vintage_crosswalk_sha256="crosswalk-sha",
        ssi_take_up_assignment_sha256="ssi-flags-sha",
        selection_identities_sha256=None,
    )
    # 9 = the pre-#557 release-refit world; 8 = the still-older pre-#374 blend
    # world. Both must miss against expected version 10.
    stale_identity = {**dict(identity), "materializer_version": 9}
    older_identity = {**dict(identity), "materializer_version": 8}
    path = tmp_path / "target_frame_checkpoint.h5"
    builder._write_target_frame_checkpoint(
        path,
        frame=frame,
        identity=stale_identity,
        compilation={"declared_targets": 1},
    )

    loaded = builder._read_target_frame_checkpoint(
        path,
        identity=identity,
        target_specs=(target,),
    )

    assert loaded is None

    builder._write_target_frame_checkpoint(
        path,
        frame=frame,
        identity=older_identity,
        compilation={"declared_targets": 1},
    )
    assert (
        builder._read_target_frame_checkpoint(
            path,
            identity=identity,
            target_specs=(target,),
        )
        is None
    )

    # Instance 2 of the same class (populace#543): a checkpoint written by a
    # run without --ssi-take-up-prior-weight-basis must not serve a run that
    # passes it (O attempt 3 warm-hit attempt 2's checkpoint and solved on
    # the other basis's SSI rows).
    basis_identity = builder._target_frame_checkpoint_identity(
        base_dataset_sha256="base-sha",
        policyengine_us_version="1.2.3",
        seed=0,
        target_period=builder.PERIOD,
        target_registry_version="registry-sha",
        weeks_unemployed_source_sha256="weeks-source-sha",
        congressional_district_vintage_crosswalk_sha256="crosswalk-sha",
        # Held equal to the writing run's digest so this case still isolates
        # the basis-artifact key (populace#543 instance 2). The broader
        # frozen-assignment digest is covered separately below.
        ssi_take_up_assignment_sha256="ssi-flags-sha",
        selection_identities_sha256=None,
        ssi_take_up_prior_weight_basis_sha256="basis-artifact-sha",
    )
    basis_path = tmp_path / "target_frame_checkpoint_basis.h5"
    builder._write_target_frame_checkpoint(
        basis_path,
        frame=frame,
        identity=identity,
        compilation={"declared_targets": 1},
    )

    loaded_with_basis = builder._read_target_frame_checkpoint(
        basis_path,
        identity=basis_identity,
        target_specs=(target,),
    )

    assert loaded_with_basis is None


def test_ssi_candidate_amount_uses_december_person_values() -> None:
    builder = _load_builder_module()

    class FakeSimulation:
        def calculate(self, variable, *, period, map_to):
            assert variable == "uncapped_ssi"
            assert period == "2024-12"
            assert map_to == "person"
            return np.asarray([0.0, 125.0, -2.0])

    values = builder._ssi_person_uncapped_amount(
        SimpleNamespace(),
        simulation=FakeSimulation(),
    )

    np.testing.assert_array_equal(values, np.asarray([0.0, 125.0, -2.0]))


def _band_spec(value, lower, upper, name, *, role=None, extra=None):
    """A minimal registry target spec carrying first-class age bounds."""
    metadata = {
        "target_role": role,
        "age_lower_bound": lower,
        "age_upper_bound": upper,
        **(extra or {}),
    }
    return SimpleNamespace(value=value, metadata=metadata, name=name)


def test_ssi_band_targets_from_registry_read_the_ledger_band_specs() -> None:
    builder = _load_builder_module()
    role = builder.SSA_SSI_AGE_BAND_RECIPIENTS_TARGET_ROLE
    # The real feed's under-18 fact carries an explicit "age >= 0"
    # constraint, so its lower bound compiles as "0", not "-inf"; ages are
    # nonnegative, so both mean the same stratum (PR #477 review finding 1).
    specs = (
        _band_spec(1_001_922.0, "0", "18", "under-18", role=role),
        _band_spec(3_905_779.0, "18", "65", "18-64", role=role),
        _band_spec(2_382_142.0, "65", "inf", "65-plus", role=role),
        SimpleNamespace(value=9.0, metadata={"target_role": "other"}, name="unrelated"),
    )
    assert builder._ssi_take_up_band_targets_from_registry(specs) == {
        "under_18": pytest.approx(1_001_922.0),
        "18_64": pytest.approx(3_905_779.0),
        "65_plus": pytest.approx(2_382_142.0),
    }


def test_ssi_band_targets_accept_unbounded_lower_edge_spelling() -> None:
    builder = _load_builder_module()
    role = builder.SSA_SSI_AGE_BAND_RECIPIENTS_TARGET_ROLE
    specs = (
        _band_spec(1_001_922.0, "-inf", "18", "under-18", role=role),
        _band_spec(3_905_779.0, "18", "65", "18-64", role=role),
        _band_spec(2_382_142.0, "65", "inf", "65-plus", role=role),
    )
    assert builder._ssi_take_up_band_targets_from_registry(specs)[
        "under_18"
    ] == pytest.approx(1_001_922.0)


def test_ssi_band_targets_fail_closed_when_a_band_is_missing() -> None:
    builder = _load_builder_module()
    role = builder.SSA_SSI_AGE_BAND_RECIPIENTS_TARGET_ROLE
    specs = (
        _band_spec(1_001_922.0, "-inf", "18", "under-18", role=role),
        _band_spec(2_382_142.0, "65", "inf", "65-plus", role=role),
    )
    with pytest.raises(RuntimeError, match=r"missing band\(s\) \['18_64'\]"):
        builder._ssi_take_up_band_targets_from_registry(specs)


def test_ssi_band_targets_reject_unrecognized_bounds() -> None:
    builder = _load_builder_module()
    role = builder.SSA_SSI_AGE_BAND_RECIPIENTS_TARGET_ROLE
    specs = (_band_spec(1_001_922.0, "-inf", "19", "off-by-one", role=role),)
    with pytest.raises(RuntimeError, match="unrecognized age bounds"):
        builder._ssi_take_up_band_targets_from_registry(specs)


def test_ssi_band_targets_reject_duplicate_bands() -> None:
    builder = _load_builder_module()
    role = builder.SSA_SSI_AGE_BAND_RECIPIENTS_TARGET_ROLE
    specs = (
        _band_spec(1_001_922.0, "-inf", "18", "under-18", role=role),
        _band_spec(999_999.0, "-inf", "18", "under-18-again", role=role),
    )
    with pytest.raises(RuntimeError, match="duplicate registry targets"):
        builder._ssi_take_up_band_targets_from_registry(specs)


def test_ssi_band_targets_reject_nonpositive_values() -> None:
    builder = _load_builder_module()
    role = builder.SSA_SSI_AGE_BAND_RECIPIENTS_TARGET_ROLE
    specs = (_band_spec(0.0, "-inf", "18", "under-18", role=role),)
    with pytest.raises(RuntimeError, match="finite and positive"):
        builder._ssi_take_up_band_targets_from_registry(specs)


def test__given_stale_target_frame_checkpoint__then_builder_ignores_it(
    tmp_path,
    small_frame,
) -> None:
    builder = _load_builder_module()
    fresh_identity = builder._target_frame_checkpoint_identity(
        base_dataset_sha256="base-sha",
        policyengine_us_version="1.2.3",
        seed=0,
        target_period=builder.PERIOD,
        target_registry_version="registry-sha",
        weeks_unemployed_source_sha256="weeks-source-sha",
        congressional_district_vintage_crosswalk_sha256="crosswalk-sha",
        ssi_take_up_assignment_sha256="ssi-flags-sha",
        selection_identities_sha256=None,
    )
    stale_identity = {
        **fresh_identity,
        "weeks_unemployed_source_sha256": "old-weeks-source-sha",
    }
    path = tmp_path / "target_frame_checkpoint.h5"
    builder._write_target_frame_checkpoint(
        path,
        frame=small_frame,
        identity=stale_identity,
        compilation={},
    )

    loaded = builder._read_target_frame_checkpoint(
        path,
        identity=fresh_identity,
        target_specs=(),
    )

    assert loaded is None


def test__given_matching_target_frame_checkpoint__then_builder_skips_materialization(
    monkeypatch,
    tmp_path,
    small_frame,
) -> None:
    builder = _load_builder_module()
    target = TargetSpec(
        name="mock.measure",
        entity="household",
        measure="household_id",
        value=1.0,
        source="Mock source",
    )
    registry = TargetRegistry((target,), country="us")
    identity = builder._target_frame_checkpoint_identity(
        base_dataset_sha256="base-sha",
        policyengine_us_version="1.2.3",
        seed=0,
        target_period=builder.PERIOD,
        target_registry_version=registry.version,
        weeks_unemployed_source_sha256="weeks-source-sha",
        congressional_district_vintage_crosswalk_sha256=None,
        ssi_take_up_assignment_sha256="ssi-flags-sha",
        selection_identities_sha256=None,
    )

    def fail_materialize(*args, **kwargs):
        raise AssertionError("materialization should not run on checkpoint hit")

    monkeypatch.setattr(builder, "_materialize_target_frame", fail_materialize)
    monkeypatch.setattr(
        builder,
        "_read_target_frame_checkpoint",
        lambda path, **kwargs: (
            small_frame,
            registry,
            {"target_frame_checkpoint": {"status": "hit"}},
        ),
    )

    loaded_frame, loaded_registry, compilation = (
        builder._load_or_materialize_target_frame(
            small_frame,
            (target,),
            target_frame_checkpoint_path=tmp_path / "target_frame_checkpoint.h5",
            target_frame_checkpoint_identity=identity,
        )
    )

    assert loaded_frame is small_frame
    assert loaded_registry is registry
    assert compilation["target_frame_checkpoint"]["status"] == "hit"


def test_runtime_versions_use_local_workspace_package_version(
    monkeypatch, tmp_path
) -> None:
    builder = _load_builder_module()
    package = tmp_path / "packages" / "populace-data"
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text(
        '[project]\nname = "populace-data"\nversion = "0.1.0"\n'
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        builder.importlib.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(
            builder.importlib.metadata.PackageNotFoundError(name)
        ),
    )

    versions = builder._runtime_versions()

    assert versions["populace-data"] == "0.1.0"


def test_reviewed_exclusions_do_not_report_opted_in_cd_sources() -> None:
    builder = _load_builder_module()
    acs_cd_alias = "census-acs-s0101-congressional-district-age-2024"
    soi_cd_alias = "soi-congressional-district-2022"

    reviewed = builder._reviewed_exclusions(
        builder.DIRECT_ACTIVE_ALIASES + (acs_cd_alias, soi_cd_alias)
    )

    assert acs_cd_alias not in reviewed
    assert soi_cd_alias not in reviewed
    assert "census-acs-s0101-national-age-2024" in reviewed
    assert "census-acs-s0101-state-age-2024" in reviewed


def test_cd_vintage_support_provenance_requires_matching_h5_attrs(
    monkeypatch, tmp_path
) -> None:
    builder = _load_builder_module()
    h5_path = tmp_path / "support.h5"
    h5_path.write_text("")

    metadata = {"sha256": "crosswalk-sha"}
    monkeypatch.setattr(
        builder,
        "_read_cd_vintage_support_provenance",
        lambda path: {},
    )

    with pytest.raises(ValueError, match="crosswalk provenance mismatch"):
        builder._assert_cd_vintage_support_matches(h5_path, metadata)

    monkeypatch.setattr(
        builder,
        "_read_cd_vintage_support_provenance",
        lambda path: {
            builder.CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR: (
                "crosswalk-sha"
            ),
            builder.CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR: (
                builder.CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
            ),
            "household_congressional_district_geoid": {
                "exists": True,
                "positive_unique_count": 436,
            },
        },
    )

    builder._assert_cd_vintage_support_matches(h5_path, metadata)


def test_cd_vintage_support_provenance_rejects_missing_cd_lookup(
    monkeypatch, tmp_path
) -> None:
    builder = _load_builder_module()
    h5_path = tmp_path / "support.h5"
    h5_path.write_text("")
    metadata = {"sha256": "crosswalk-sha"}
    monkeypatch.setattr(
        builder,
        "_read_cd_vintage_support_provenance",
        lambda path: {
            builder.CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR: (
                "crosswalk-sha"
            ),
            builder.CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR: (
                builder.CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
            ),
            "household_congressional_district_geoid": {"exists": False},
        },
    )

    with pytest.raises(ValueError, match="missing household congressional"):
        builder._assert_cd_vintage_support_matches(h5_path, metadata)


def test_cd_vintage_support_provenance_counts_only_positive_numeric_lookup() -> None:
    builder = _load_builder_module()

    assert (
        builder._positive_numeric_unique_count(
            np.asarray(["", "0", "0000", "not-a-geoid"])
        )
        == 0
    )
    assert (
        builder._positive_numeric_unique_count(np.asarray(["0101", "0101", "0200"]))
        == 2
    )


def test_cd_vintage_support_provenance_names_us_extra_when_h5py_missing(
    monkeypatch, tmp_path
) -> None:
    builder = _load_builder_module()
    h5_path = tmp_path / "support.h5"
    h5_path.write_text("")
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "h5py":
            raise ModuleNotFoundError("No module named 'h5py'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as excinfo:
        builder._read_cd_vintage_support_provenance(h5_path)

    message = str(excinfo.value)
    assert "--extra us" in message
    assert "before calibration or donor imputation" in message


def _passing_critical_diagnostics(builder) -> tuple[SimpleNamespace, ...]:
    def diagnostic(name, target, final_estimate):
        return SimpleNamespace(
            name=f"{name}@{builder.PERIOD}",
            target=target,
            initial_estimate=target,
            final_estimate=final_estimate,
            relative_error=(final_estimate - target) / target,
        )

    return (
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.income_tax_liability_amount",
            2_105_345_646_000.0,
            2_067_762_165_736.424,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.income_tax_liability_returns",
            113_562_590.0,
            105_437_267.69738781,
        ),
        diagnostic(
            "ssa_supplement.cy2024.oasdi_ssi_payments."
            "social_security_benefits.payment_amount",
            1_471_195_000_000.0,
            1_541_540_768_722.367,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.ctc_amount",
            82_863_353_000.0,
            88_000_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.ctc_claims",
            38_068_980.0,
            40_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.actc_amount",
            33_857_960_000.0,
            35_300_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.actc_claims",
            17_691_400.0,
            17_100_000.0,
        ),
        diagnostic(
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_amount",
            69_041_649_000.0,
            70_000_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_returns",
            23_837_149.0,
            23_800_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_amount",
            53_910_190_000.0,
            58_000_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_returns",
            7_841_370.0,
            8_200_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_amount",
            455_904_900_000.0,
            490_000_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_returns",
            24_475_100.0,
            26_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.itemized_deductions_amount",
            1_000_000_000_000.0,
            1_020_000_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
            "total_itemized_deductions_amount",
            1_000_000_000_000.0,
            1_020_000_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.limited_state_local_taxes_amount",
            120_000_000_000.0,
            121_000_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
            "limited_state_local_taxes_amount",
            120_000_000_000.0,
            121_000_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.medical_dental_expense_amount",
            80_000_000_000.0,
            79_000_000_000.0,
        ),
        # populace#511: the Table 2.1 mortgage rows are registered critical
        # (certified O-1 shipped the amount row at +29.5% with no gate).
        diagnostic(
            "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
            "home_mortgage_interest_amount",
            186_310_104_604.0,
            199_110_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
            "home_mortgage_interest_returns",
            11_644_348.0,
            11_929_445.0,
        ),
        # The SOI Table 1.4 national dollar blanket (populace#462) needs at
        # least one Table 1.4 amount row on the surface, within tolerance.
        diagnostic(
            "irs_soi.ty2023.table_1_4.all.wages_salaries_amount",
            10_773_360_188_645.0,
            10_774_383_029_502.0,
        ),
    )


def _critical_surface(builder, *rows) -> tuple[SimpleNamespace, ...]:
    replacement_names = {row.name for row in rows}
    return tuple(
        diagnostic
        for diagnostic in _passing_critical_diagnostics(builder)
        if diagnostic.name not in replacement_names
    ) + tuple(rows)


def _critical_contract_failures(
    builder,
    diagnostics,
    *,
    specs: tuple[TargetSpec, ...] = (),
    incumbent: dict[str, dict[str, float]] | None = None,
) -> tuple[list[str], list[str]]:
    from populace.data.contract import _check_us_critical_target_fit

    incumbent = incumbent or {}
    registry = TargetRegistry(specs, country="us")
    specs_by_name = {builder._target_row_name(spec): spec for spec in registry.specs}
    result = SimpleNamespace(
        skipped=(),
        diagnostics=tuple(diagnostics),
        problem=SimpleNamespace(
            names=tuple(specs_by_name),
            targets=tuple(spec.to_target() for spec in registry.specs),
        ),
        initial_loss=10.0,
        final_loss=5.0,
    )
    builder_failures = builder._release_gate_failures(
        result,
        {"dropped_target_names": []},
        target_registry=registry,
        incumbent_diagnostics=incumbent,
    )
    publisher_rows = []
    for diagnostic in diagnostics:
        spec = specs_by_name.get(diagnostic.name)
        publisher_rows.append(
            {
                "name": diagnostic.name,
                "target": diagnostic.target,
                "final_estimate": diagnostic.final_estimate,
                "relative_error": diagnostic.relative_error,
                "metadata": dict(spec.metadata) if spec is not None else {},
                "registry": {
                    "family": spec.family if spec is not None else "",
                },
            }
        )
    publisher_failures: list[str] = []
    _check_us_critical_target_fit(
        {
            "targets": publisher_rows,
            "build": {
                "incumbent_diagnostics": {
                    "critical_targets": incumbent,
                }
            },
        },
        publisher_failures,
    )
    return builder_failures, publisher_failures


def _assert_table_requirement_matches_shared(builder_requirement, shared) -> None:
    assert builder_requirement.max_abs_relative_error == (shared.max_abs_relative_error)
    assert builder_requirement.accepted_names == shared.names
    assert builder_requirement.accepted_name_prefixes == ()
    assert builder_requirement.accepted_name_substrings == shared.name_substrings
    assert builder_requirement.accepted_name_suffixes == shared.name_suffixes


def test_soi_component_amounts_use_source_specific_signs() -> None:
    builder = _load_builder_module()

    assert np.array_equal(
        builder._signed_component(np.array([-5.0, 0.0, 7.0]), "capital_gains_gross"),
        np.array([0.0, 0.0, 7.0]),
    )
    assert np.array_equal(
        builder._signed_component(np.array([-5.0, 0.0, 7.0]), "capital_gains_losses"),
        np.array([0.0, 0.0, 7.0]),
    )
    assert np.array_equal(
        builder._signed_component(np.array([-5.0, 0.0, 7.0]), "business_net_losses"),
        np.array([5.0, -0.0, -0.0]),
    )
    assert np.array_equal(
        builder._signed_component(
            np.array([-5.0, 0.0, 7.0]), "rent_and_royalty_net_income"
        ),
        np.array([-5.0, 0.0, 7.0]),
    )
    assert np.array_equal(
        builder._signed_component(np.array([-5.0, 7.0]), "adjusted_gross_income"),
        np.array([-5.0, 7.0]),
    )


def test_export_target_audit_is_opt_in(monkeypatch) -> None:
    builder = _load_builder_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
        ],
    )
    args = builder._parse_args()
    assert not args.audit_export_targets

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
            "--audit-export-targets",
        ],
    )
    args = builder._parse_args()
    assert args.audit_export_targets


def test_sipp_tip_donor_override_parses(monkeypatch) -> None:
    builder = _load_builder_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
            "--sipp-tip-donor",
            "pu2023_slim.csv",
        ],
    )

    args = builder._parse_args()

    assert args.sipp_tip_donor == Path("pu2023_slim.csv")


def test_weeks_unemployed_source_override_parses(monkeypatch) -> None:
    builder = _load_builder_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
            "--asec-2023-weeks-unemployed-source",
            "asecpub23csv.zip",
        ],
    )

    args = builder._parse_args()

    assert args.asec_2023_weeks_unemployed_source == Path("asecpub23csv.zip")


def test_frozen_support_selection_is_followed_by_weeks_unemployed_regate() -> None:
    builder = _load_builder_module()
    source = Path(builder.__file__).read_text(encoding="utf-8")

    base_load = source.index("base_frame = _load_frame(base_h5)")
    tail_presence = source.index(
        "capital_gains_tail_presence = assert_puf_capital_gains_tail_survives_selection(",
        base_load,
    )
    selection = source.index("base_frame, selection_report = select_frozen_support(")
    tail_retention = source.index(
        "assert_puf_capital_gains_tail_survives_selection(",
        selection,
    )
    regate = source.index(
        "post_selection_weeks_unemployed_gate = us_weeks_unemployed_signal_gate("
    )
    mass_repair = source.index(
        "base_frame, base_population_repair = _with_base_population_mass_repair("
    )

    assert base_load < tail_presence < selection < tail_retention < regate < mass_repair
    assert "Post-selection weeks-unemployed input signal failed" in source


def test_sipp_vehicle_donor_override_parses(monkeypatch) -> None:
    builder = _load_builder_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
            "--sipp-vehicle-donor",
            "pu2023.csv",
        ],
    )

    args = builder._parse_args()

    assert args.sipp_vehicle_donor == Path("pu2023.csv")


def test_scf_full_extract_override_parses(monkeypatch) -> None:
    builder = _load_builder_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
            "--scf-full-extract",
            "p22i6.dta",
        ],
    )

    args = builder._parse_args()

    assert args.scf_full_extract == Path("p22i6.dta")


def test_org_wages_donor_override_parses(monkeypatch) -> None:
    builder = _load_builder_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
            "--org-wages-donor",
            "census_cps_org_2024_wages.csv.gz",
        ],
    )

    args = builder._parse_args()

    assert args.org_wages_donor == Path("census_cps_org_2024_wages.csv.gz")


def test_cd_targets_default_to_the_packaged_vintage_crosswalk(monkeypatch) -> None:
    builder = _load_builder_module()

    # CD targets with no explicit crosswalk fall back to the packaged
    # Census-built default so the build works out of the box.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
            "--include-congressional-district-targets",
        ],
    )
    args = builder._parse_args()
    default_path = args.congressional_district_vintage_crosswalk
    assert default_path is not None
    assert default_path.name == "congressional_district_vintage_crosswalk.csv"
    assert default_path.exists()

    # An explicit path still overrides the default.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
            "--include-congressional-district-targets",
            "--congressional-district-vintage-crosswalk",
            "crosswalk.csv",
        ],
    )

    args = builder._parse_args()

    assert args.congressional_district_vintage_crosswalk == Path("crosswalk.csv")


def test_maximum_microsim_batch_size_defaults_and_overrides(monkeypatch) -> None:
    builder = _load_builder_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
        ],
    )
    args = builder._parse_args()
    assert (
        args.maximum_microsim_batch_size == builder.DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
            "--maximum-microsim-batch-size",
            "0",
        ],
    )
    args = builder._parse_args()
    assert args.maximum_microsim_batch_size == 0


def test_staging_repo_can_default_from_environment(monkeypatch) -> None:
    builder = _load_builder_module()
    monkeypatch.setenv("POPULACE_STAGING_REPO_ID", "policyengine/populace-us-staging")
    monkeypatch.setenv("POPULACE_STAGING_PREFIX", "candidate-runs")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
        ],
    )

    args = builder._parse_args()

    assert args.staging_repo_id == "policyengine/populace-us-staging"
    assert args.staging_prefix == "candidate-runs"


def test_soi_indicator_rows_flag_positive_component_items() -> None:
    builder = _load_builder_module()

    assert np.array_equal(
        builder._soi_component_row(
            np.array([-5.0, 0.0, 7.0]),
            "capital_gains_gross",
            indicator=True,
        ),
        np.array([0.0, 0.0, 1.0]),
    )
    assert np.array_equal(
        builder._soi_component_row(
            np.array([-5.0, 0.0, 7.0]),
            "capital_gains_losses",
            indicator=True,
        ),
        np.array([0.0, 0.0, 1.0]),
    )
    assert np.array_equal(
        builder._soi_component_row(
            np.array([-5.0, 0.0, 7.0]),
            "business_net_losses",
            indicator=True,
        ),
        np.array([1.0, 0.0, 0.0]),
    )


def test_soi_eitc_child_count_filter_uses_ledger_filter_first() -> None:
    builder = _load_builder_module()

    assert (
        builder._soi_eitc_child_count_filter(
            {
                "ledger_filter_eitc_child_count": "2",
                "source_measure_id": "eitc_no_children_amount",
            }
        )
        == "2"
    )
    assert (
        builder._soi_eitc_child_count_filter(
            {"source_measure_id": "eitc_no_children_amount"}
        )
        == "0"
    )
    assert (
        builder._soi_eitc_child_count_filter(
            {"source_measure_id": "eitc_one_child_claims"}
        )
        == "1"
    )
    assert (
        builder._soi_eitc_child_count_filter(
            {"source_measure_id": "eitc_two_children_amount"}
        )
        == "2"
    )
    assert (
        builder._soi_eitc_child_count_filter(
            {"source_measure_id": "eitc_three_or_more_children_claims"}
        )
        == "3plus"
    )
    assert (
        builder._soi_eitc_child_count_filter(
            {
                "ledger_layout_record_set_id": (
                    "irs_soi.ty2022.table_2_5.eitc_by_agi_children."
                    "no_qualifying_children"
                ),
                "source_measure_id": "eitc_total",
            }
        )
        == "0"
    )
    assert (
        builder._soi_eitc_child_count_filter(
            {
                "ledger_layout_record_set_id": (
                    "irs_soi.ty2022.table_2_5.eitc_by_agi_children."
                    "three_or_more_qualifying_children"
                ),
                "source_measure_id": "eitc_total",
            }
        )
        == "3plus"
    )
    assert (
        builder._soi_eitc_child_count_filter({"source_measure_id": "eitc_total"})
        is None
    )


def test_unsupported_soi_ledger_filters_require_materializer_support() -> None:
    builder = _load_builder_module()

    assert (
        builder._unsupported_soi_ledger_filters(
            {
                "ledger_filter_income_range": "25k_to_30k",
                "ledger_filter_filing_status": "all",
                "ledger_filter_eitc_child_count": "1",
            }
        )
        == ()
    )
    assert (
        builder._unsupported_soi_ledger_filters(
            {
                "ledger_filter_new_dimension": "all",
            }
        )
        == ()
    )
    assert builder._unsupported_soi_ledger_filters(
        {
            "ledger_filter_new_dimension": "specific_slice",
        }
    ) == ("ledger_filter_new_dimension",)


def test_unsupported_ledger_filter_metadata_all_value_is_noop() -> None:
    builder = _load_builder_module()
    specs = (
        SimpleNamespace(
            name="all_child_count",
            metadata={"ledger_filter_qualifying_children": "all"},
        ),
        SimpleNamespace(
            name="specific_child_count",
            metadata={"ledger_filter_qualifying_children": "one"},
        ),
    )

    assert builder._unsupported_ledger_filter_metadata(specs) == {
        "specific_child_count": ("ledger_filter_qualifying_children",)
    }


def test_identity_ledger_filter_qualifiers_are_inert_not_unsupported() -> None:
    """Series-identity qualifiers pass the guard; unknown domain filters fail.

    Build M's sparse run stopped here: the #405 NIPA and LIHEAP targets carry
    fact metadata identifying WHICH published series the registry selected
    (a NIPA table line code, the LIHEAP state-programs count) — applied at
    fact-selection, restricting nothing in the microdata. The guard now
    recognizes the reviewed identity-qualifier class as inert while any
    unknown ledger_filter_* key stays fatal, so a genuine domain filter can
    never be silently ignored.
    """

    builder = _load_builder_module()
    specs = (
        SimpleNamespace(
            name="bea_nipa.cy2024.total_wages_salaries.a034rc.wages_salaries_amount",
            metadata={"ledger_filter_bea_nipa.series_code": "a034rc"},
        ),
        SimpleNamespace(
            name="hhs_acf_liheap.fy2024.national_profile.state_programs.households_served",
            metadata={
                "ledger_filter_administering_entity": "state_programs",
                "ledger_filter_program": "liheap",
            },
        ),
        SimpleNamespace(
            name="unknown_domain_filter",
            metadata={"ledger_filter_novel_dimension": "specific_slice"},
        ),
    )

    assert builder._unsupported_ledger_filter_metadata(specs) == {
        "unknown_domain_filter": ("ledger_filter_novel_dimension",)
    }


def test_eitc_child_count_mask_supports_soi_child_groups() -> None:
    builder = _load_builder_module()
    counts = np.asarray([0, 1, 2, 3, 4], dtype=np.float64)

    assert np.array_equal(
        builder._eitc_child_count_mask(counts, "0"),
        np.asarray([True, False, False, False, False]),
    )
    assert np.array_equal(
        builder._eitc_child_count_mask(counts, "1"),
        np.asarray([False, True, False, False, False]),
    )
    assert np.array_equal(
        builder._eitc_child_count_mask(counts, "2"),
        np.asarray([False, False, True, False, False]),
    )
    assert np.array_equal(
        builder._eitc_child_count_mask(counts, "3plus"),
        np.asarray([False, False, False, True, True]),
    )
    assert np.array_equal(
        builder._eitc_child_count_mask(counts, "3+"),
        np.asarray([False, False, False, True, True]),
    )


def test_combined_household_values_unions_positive_person_support(small_frame) -> None:
    builder = _load_builder_module()

    variable_values = {
        "medicaid_enrolled": np.asarray([1.0, 1.0, 0.0, 0.0]),
        "chip_enrolled": np.asarray([1.0, 0.0, 1.0, 0.0]),
    }

    class FakeSimulation:
        def calculate(self, variable, *, period, map_to=None):
            assert period == builder.PERIOD
            assert map_to is None
            return variable_values[variable]

    person_entity = SimpleNamespace(key="person")
    system = SimpleNamespace(
        variables={
            variable: SimpleNamespace(entity=person_entity)
            for variable in variable_values
        }
    )

    values = builder._combined_household_values(
        frame=small_frame,
        simulation=FakeSimulation(),
        system=system,
        variables=("medicaid_enrolled", "chip_enrolled"),
        tax_unit_positions=np.asarray([], dtype=np.int64),
        positive_indicator=True,
    )
    assert np.array_equal(values, np.asarray([2.0, 1.0]))

    summed_values = builder._combined_household_values(
        frame=small_frame,
        simulation=FakeSimulation(),
        system=system,
        variables=("medicaid_enrolled", "chip_enrolled"),
        tax_unit_positions=np.asarray([], dtype=np.int64),
        positive_indicator=False,
    )
    assert np.array_equal(summed_values, np.asarray([3.0, 1.0]))


def test_combined_household_values_can_count_tax_unit_variable_on_people(
    small_frame,
) -> None:
    builder = _load_builder_module()

    mapped_values = {
        "assigned_aca_ptc": np.asarray([5_000.0, 5_000.0, 3_000.0, 0.0]),
        "is_aca_ptc_eligible": np.asarray([1.0, 0.0, 1.0, 1.0]),
    }

    class FakeSimulation:
        def calculate(self, variable, *, period, map_to=None):
            assert period == builder.PERIOD
            assert map_to == "person"
            return mapped_values[variable]

    system = SimpleNamespace(
        variables={
            "assigned_aca_ptc": SimpleNamespace(entity=SimpleNamespace(key="tax_unit")),
            "is_aca_ptc_eligible": SimpleNamespace(
                entity=SimpleNamespace(key="person")
            ),
        }
    )

    values = builder._combined_household_values(
        frame=small_frame,
        simulation=FakeSimulation(),
        system=system,
        variables=("assigned_aca_ptc",),
        tax_unit_positions=np.asarray([], dtype=np.int64),
        positive_indicator=True,
        map_to="person",
        filter_variable="is_aca_ptc_eligible",
    )

    assert np.array_equal(values, np.asarray([1.0, 1.0]))


def test_combined_household_values_threshold_count_keeps_domain_filter(
    small_frame,
) -> None:
    builder = _load_builder_module()

    mapped_values = {
        "selected_marketplace_plan_benchmark_ratio": np.asarray([0.8, 1.2, 0.7]),
        "assigned_aca_ptc": np.asarray([500.0, 500.0, 0.0]),
    }

    class FakeSimulation:
        def calculate(self, variable, *, period, map_to=None):
            assert period == builder.PERIOD
            assert map_to is None
            return mapped_values[variable]

    system = SimpleNamespace(
        variables={
            "selected_marketplace_plan_benchmark_ratio": SimpleNamespace(
                entity=SimpleNamespace(key="tax_unit")
            ),
            "assigned_aca_ptc": SimpleNamespace(entity=SimpleNamespace(key="tax_unit")),
        }
    )

    values = builder._combined_household_values(
        frame=small_frame,
        simulation=FakeSimulation(),
        system=system,
        variables=("selected_marketplace_plan_benchmark_ratio",),
        tax_unit_positions=np.asarray([0, 0, 1], dtype=np.int64),
        filter_variable="assigned_aca_ptc",
        less_than=1.0,
    )

    assert np.array_equal(values, np.asarray([1.0, 0.0]))


def test_release_gate_failures_are_not_unconditional() -> None:
    builder = _load_builder_module()
    result = SimpleNamespace(
        skipped=(),
        diagnostics=_passing_critical_diagnostics(builder),
        initial_loss=10.0,
        final_loss=5.0,
    )

    assert builder._release_gate_failures(result, {"dropped_target_names": []}) == []

    assert builder._release_gate_failures(
        result,
        {"dropped_target_names": ["missing"]},
    ) == ["1 fiscal targets were not materialized."]

    skipped = SimpleNamespace(target=SimpleNamespace(name="skipped"), reason="bad")
    with_skipped = SimpleNamespace(
        skipped=(skipped,),
        diagnostics=_passing_critical_diagnostics(builder),
        initial_loss=10.0,
        final_loss=5.0,
    )
    assert builder._release_gate_failures(
        with_skipped,
        {"dropped_target_names": []},
    ) == ["1 fiscal targets were skipped by calibration."]

    worse = SimpleNamespace(
        skipped=(),
        diagnostics=_passing_critical_diagnostics(builder),
        initial_loss=5.0,
        final_loss=10.0,
    )
    assert builder._release_gate_failures(worse, {"dropped_target_names": []}) == [
        "Calibration final loss is worse than the initial loss (10.0 > 5.0)."
    ]


def test_release_gate_failures_include_target_profile_coverage() -> None:
    builder = _load_builder_module()
    result = SimpleNamespace(
        skipped=(),
        diagnostics=_passing_critical_diagnostics(builder),
        initial_loss=10.0,
        final_loss=5.0,
    )
    target_profile_gate = builder.GateResult(
        name="target_profile_coverage",
        passed=False,
        failures=("medicaid_chip_enrollment: missing",),
    )

    assert builder._release_gate_failures(
        result,
        {"dropped_target_names": []},
        target_profile_gate,
    ) == [
        "Target profile coverage failed: medicaid_chip_enrollment: missing",
    ]


def test_release_gate_failures_include_health_input_signal() -> None:
    builder = _load_builder_module()
    result = SimpleNamespace(
        skipped=(),
        diagnostics=_passing_critical_diagnostics(builder),
        initial_loss=10.0,
        final_loss=5.0,
    )
    health_input_gate = builder.GateResult(
        name="health_input_signal",
        passed=False,
        failures=("takes_up_aca_if_eligible: constant",),
    )

    assert builder._release_gate_failures(
        result,
        {"dropped_target_names": []},
        health_input_gate=health_input_gate,
    ) == [
        "Health input signal failed: takes_up_aca_if_eligible: constant",
    ]


def test_base_population_scale_gate_rejects_underweighted_base(small_frame) -> None:
    builder = _load_builder_module()

    gate = builder._base_population_scale_gate(small_frame)

    assert not gate.passed
    assert gate.name == "base_population_scale"
    assert gate.details["population"] == 6000.0
    assert "mass='conserve'" in gate.failures[0]


def test_base_population_scale_gate_accepts_national_scale_base(small_frame) -> None:
    builder = _load_builder_module()
    benchmark = builder.US_BASE_PERSON_POPULATION_BENCHMARK
    frame = small_frame.with_weights(
        "household",
        builder.Weights(
            values=np.asarray([benchmark / 4.0, benchmark / 4.0]),
            kind=WeightKind.DESIGN,
        ),
        mass=builder.MassChange(
            factor=benchmark / 6000.0,
            reason="test fixture national-scale base",
        ),
    )

    gate = builder._base_population_scale_gate(frame)

    assert gate.passed
    assert gate.details["population"] == benchmark
    assert gate.details["relative_error"] == 0.0


def test_base_population_mass_repair_rescales_to_census_benchmark(
    small_frame,
) -> None:
    builder = _load_builder_module()
    benchmark = builder.US_BASE_PERSON_POPULATION_BENCHMARK

    repaired, repair = builder._with_base_population_mass_repair(small_frame)

    assert repair["applied"]
    assert repair["method"] == "rescale_household_weights_to_census_person_population"
    assert repair["initial_population"] == 6000.0
    assert np.isclose(repair["factor"], benchmark / 6000.0)
    assert np.isclose(repair["repaired_population"], benchmark)
    assert np.isclose(float(repaired.resolve_weights("person").values.sum()), benchmark)
    assert repaired.mass_log[-1].entity == "household"
    assert (
        repaired.mass_log[-1].reason == builder.US_BASE_PERSON_POPULATION_REPAIR_REASON
    )

    gate = builder._base_population_scale_gate(repaired, mass_repair=repair)
    assert gate.passed
    assert gate.details["mass_repair"]["initial_population"] == 6000.0
    assert np.isclose(gate.details["mass_repair"]["factor"], benchmark / 6000.0)


def test_social_security_component_value_repair_uses_registry_targets(
    small_frame,
) -> None:
    builder = _load_builder_module()
    person = small_frame.table("person").copy()
    person["social_security_retirement"] = [1.0, 0.0, 2.0, 0.0]
    person["social_security_disability"] = [0.0, 3.0, 0.0, 1.0]
    person["social_security_dependents"] = [2.0, 0.0, 0.0, 1.0]
    person["social_security_survivors"] = [0.0, 1.0, 2.0, 0.0]
    frame = Frame(
        {
            "person": person,
            "household": small_frame.table("household").copy(),
        },
        small_frame.schema,
        {"household": small_frame.weights_for("household")},
    )
    targets = {
        "ssa_retirement_total": 10_000.0,
        "ssa_disability_total": 8_000.0,
        "ssa_dependents_total": 6_000.0,
        "ssa_survivors_total": 12_000.0,
    }
    specs = tuple(
        TargetSpec(
            name=f"ssa.{role}",
            entity="household",
            value=value,
            measure="unused",
            period=builder.PERIOD,
            source="SSA",
            metadata={"target_role": role},
        )
        for role, value in targets.items()
    )

    repaired, repair = builder._with_social_security_component_value_repair(
        frame,
        specs,
    )

    assert repair["applied"]
    weights = pd.Series(repaired.resolve_weights("person").values)
    for role, column in builder.US_SOCIAL_SECURITY_COMPONENT_TARGET_ROLES.items():
        total = float((repaired.table("person")[column] * weights).sum())
        assert np.isclose(total, targets[role])
        assert np.isclose(
            repair["components"][column]["repaired_estimate"],
            targets[role],
        )


def test_non_sch_d_cgd_value_repair_pins_the_aged_soi_fact(small_frame) -> None:
    builder = _load_builder_module()
    person = small_frame.table("person").copy()
    person["non_sch_d_capital_gains"] = [100.0, 0.0, 300.0, 0.0]
    frame = Frame(
        {
            "person": person,
            "household": small_frame.table("household").copy(),
        },
        small_frame.schema,
        {"household": small_frame.weights_for("household")},
    )
    # The REAL compiled spec name is the unsuffixed ledger source_record_id
    # (verified against a live v9.2 compile; PR #486 review finding 1).
    spec = TargetSpec(
        name="irs_soi.ty2023.table_1_4.all.capital_gain_distributions_amount",
        entity="household",
        value=500.0,
        measure="unused",
        period=builder.PERIOD,
        source="IRS SOI",
        metadata={"source_measure_id": "payment_amount", "aged_to": "2024"},
    )
    returns_decoy = TargetSpec(
        name="irs_soi.ty2023.table_1_4.all.capital_gain_distributions_returns",
        entity="household",
        value=3_209_131.0,
        measure="unused",
        period=builder.PERIOD,
        source="IRS SOI",
        metadata={"source_measure_id": "return_count"},
    )
    state_decoy = TargetSpec(
        name="irs_soi.ty2023.table_1_4.all.capital_gain_distributions_amount",
        entity="household",
        value=9.0,
        measure="unused",
        period=builder.PERIOD,
        source="IRS SOI",
        metadata={"source_measure_id": "payment_amount", "state_fips": "06"},
    )

    repaired, repair = builder._with_non_sch_d_cgd_value_repair(
        frame, (returns_decoy, spec, state_decoy)
    )

    assert repair["applied"]
    weights = pd.Series(repaired.resolve_weights("person").values)
    total = float((repaired.table("person")["non_sch_d_capital_gains"] * weights).sum())
    assert np.isclose(total, 500.0)
    assert np.isclose(repair["repaired_estimate"], 500.0)
    assert np.isclose(repair["factor"], repair["target"] / repair["initial_estimate"])
    assert repair["target_aged_to"] == "2024"
    assert "mean-reverting" in repair["reason"]

    with pytest.raises(RuntimeError, match="exactly one aged Table 1.4"):
        builder._with_non_sch_d_cgd_value_repair(frame, ())
    person_missing = small_frame.table("person").copy()
    frame_missing = Frame(
        {
            "person": person_missing,
            "household": small_frame.table("household").copy(),
        },
        small_frame.schema,
        {"household": small_frame.weights_for("household")},
    )
    with pytest.raises(RuntimeError, match="requires person column"):
        builder._with_non_sch_d_cgd_value_repair(frame_missing, (spec,))


def test_load_qrf_tail_concentration_exclusions(tmp_path) -> None:
    builder = _load_builder_module()
    assert builder._load_qrf_tail_concentration_exclusions(None) == {}
    good = tmp_path / "tail.json"
    good.write_text('{"estate_income": "tracked defect populace#481"}')
    assert builder._load_qrf_tail_concentration_exclusions(good) == {
        "estate_income": "tracked defect populace#481"
    }
    bad = tmp_path / "bad.json"
    bad.write_text('{"estate_income": "  "}')
    with pytest.raises(ValueError, match="non-empty reason"):
        builder._load_qrf_tail_concentration_exclusions(bad)
    notdict = tmp_path / "list.json"
    notdict.write_text("[1]")
    with pytest.raises(ValueError, match="JSON object"):
        builder._load_qrf_tail_concentration_exclusions(notdict)


def test_release_gate_failures_reject_positive_zero_support_targets() -> None:
    builder = _load_builder_module()
    result = SimpleNamespace(
        skipped=(),
        diagnostics=(
            SimpleNamespace(
                name=f"nation/irs/zero@{builder.PERIOD}",
                target=1_000.0,
                initial_estimate=0.0,
                final_estimate=0.0,
            ),
            SimpleNamespace(
                name=f"nation/irs/nonzero@{builder.PERIOD}",
                target=1_000.0,
                initial_estimate=10.0,
                final_estimate=20.0,
            ),
            *_passing_critical_diagnostics(builder),
        ),
        initial_loss=10.0,
        final_loss=5.0,
    )

    assert builder._release_gate_failures(result, {"dropped_target_names": []}) == [
        "1 positive fiscal targets have zero materialized support "
        f"(examples: nation/irs/zero@{builder.PERIOD})."
    ]


def test_release_gate_failures_reject_certified_o1_mortgage_overshoot() -> None:
    # populace#511 regression: certified O-1 shipped the Table 2.1 mortgage
    # amount row at +29.5% and no gate objected because mortgage was not in
    # the critical register. The exact shipped diagnostics must now fail the
    # release gate, and the expected post-remap fit (+6.9%) must pass.
    builder = _load_builder_module()
    row_name = (
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
        "home_mortgage_interest_amount"
    )
    shipped_o1 = SimpleNamespace(
        name=f"{row_name}@{builder.PERIOD}",
        target=186_310_104_604.0,
        initial_estimate=344_449_138_996.0,
        final_estimate=241_268_995_041.0,
        relative_error=(241_268_995_041.0 - 186_310_104_604.0) / 186_310_104_604.0,
    )
    result = SimpleNamespace(
        skipped=(),
        diagnostics=_critical_surface(builder, shipped_o1),
        initial_loss=10.0,
        final_loss=5.0,
    )

    failures = builder._release_gate_failures(result, {"dropped_target_names": []})

    # Pin the overshoot arithmetic, not just the row identity: the missing-row
    # formatter also names the row and label, so a fixture slip that drops the
    # replacement instead of appending it must not satisfy this assertion.
    assert any(
        row_name in failure
        and "home mortgage interest deduction amount" in failure
        and "relative_error=0.294986" in failure
        and "exceeding 0.2" in failure
        for failure in failures
    ), failures

    passing = SimpleNamespace(
        skipped=(),
        diagnostics=_critical_surface(builder),
        initial_loss=10.0,
        final_loss=5.0,
    )
    assert builder._release_gate_failures(passing, {"dropped_target_names": []}) == []


def test_release_gate_failures_keep_cd_targets_diagnostic_by_default() -> None:
    builder = _load_builder_module()
    cd_spec = TargetSpec(
        name="irs_soi.ty2023.congressional_district_2022.all_returns."
        "ak_00.tax_exempt_interest_amount",
        entity="household",
        measure="tax_exempt_interest",
        value=1_000.0,
        source="fixture",
        family="irs_soi",
        metadata={
            "ledger_geography_level": "congressional_district",
            "congressional_district_geoid": "0200",
        },
    )
    cd_target = cd_spec.to_target()
    cd_row_name = f"{cd_spec.name}@{builder.PERIOD}"
    result = SimpleNamespace(
        skipped=(SimpleNamespace(target=cd_target, reason="missing column"),),
        diagnostics=(
            SimpleNamespace(
                name=cd_row_name,
                target=1_000.0,
                initial_estimate=0.0,
                final_estimate=0.0,
            ),
            *_passing_critical_diagnostics(builder),
        ),
        problem=SimpleNamespace(
            names=(cd_row_name,),
            targets=(cd_target,),
        ),
        initial_loss=10.0,
        final_loss=5.0,
    )
    compilation = {
        "dropped_target_names": [cd_spec.name],
        "gate_congressional_district_targets": False,
        "diagnostic_only_dropped_target_names": [cd_spec.name],
    }

    assert builder._release_gate_failures(result, compilation) == []

    gated_compilation = {
        **compilation,
        "gate_congressional_district_targets": True,
    }

    assert builder._release_gate_failures(result, gated_compilation) == [
        "1 fiscal targets were not materialized.",
        "1 fiscal targets were skipped by calibration.",
        "1 positive fiscal targets have zero materialized support "
        f"(examples: {cd_row_name}).",
    ]


def test_builder_contains_publisher_cd_exclusions_from_real_registry() -> None:
    from collections import UserDict
    from dataclasses import replace

    from populace.build.ledger_targets import (
        LedgerTargetReference,
        compile_ledger_target_references,
    )
    from populace.data.contract import (
        _check_us_critical_target_fit,
        _is_congressional_district_layout_target,
    )
    from populace.data.us_critical_targets import is_congressional_district_target

    builder = _load_builder_module()
    source_record_id = (
        "irs_soi.ty2023.table_1_4.congressional_district_2022.al_01."
        "medical_expense_amount"
    )
    fact = {
        "aggregate_fact_key": "ledger.aggregate_fact.v2:cd-fixture",
        "legacy_fact_key": "ledger.fact.v1:cd-fixture",
        "lineage": {
            "source_record_id": source_record_id,
            "source_cell_keys": ["ledger.source_cell.v1:cd-fixture"],
            "source_row_keys": [],
        },
        "value": 100.0,
        "period": {"type": "tax_year", "value": 2023},
        "geography": {
            "level": "congressional_district",
            "id": "5001700US0101",
            "name": "Alabama District 1",
            "vintage": "2022",
        },
        "entity": {"name": "tax_unit", "role": "filing_unit"},
        "observed_measure": {
            "source_name": "irs_soi",
            "source_table": "Publication 1304 Table 1.4",
            "source_measure_id": "medical_expense_amount",
            "source_concept": "irs_soi.medical_expense_amount",
            "unit": "usd",
        },
        "concept_alignment": {
            "source_concept": "irs_soi.medical_expense_amount",
            "canonical_concept": "irs_soi.medical_expense_amount",
            "relation": "exact",
            "authority": "policyengine-ledger",
            "legal_vintage": "tax_year_2023",
        },
        "aggregation": {"method": "sum"},
        "source": {
            "source_name": "irs_soi",
            "source_table": "Publication 1304 Table 1.4",
            "source_file": "fixture.xlsx",
            "url": "https://www.irs.gov/",
            "vintage": "tax_year_2023",
        },
        "dimensions": {},
        "universe_constraints": {"domain": "all_individual_income_tax_returns"},
        "layout": {
            "record_set_id": "irs_soi.ty2023.table_1_4.cd_fixture",
            "groupby_dimension": "irs_soi.congressional_district",
            "groupby_value_id": "al_01",
            "measure_id": "medical_expense_amount",
        },
    }
    reference = LedgerTargetReference(
        name=(
            "irs_soi.ty2023.table_1_4.congressional_district_2022.al_01."
            "medical_expense_amount"
        ),
        ledger_source_record_id=source_record_id,
        entity="household",
        measure="medical_expense",
        period=builder.PERIOD,
        family="irs_soi",
        metadata={
            "target_role": "medical_expense_deduction_total",
            "geography_scope": "congressional_district",
            "congressional_district_geoid": "0101",
        },
    )
    compiled = compile_ledger_target_references([fact], [reference], country="us")
    (compiled_spec,) = compiled.specs

    evidence = {
        "layout": (
            "ledger_layout_groupby_dimension",
            "irs_soi.congressional_district",
        ),
        "source": (
            "ledger_source_record_id",
            "fixture.congressional_district_01",
        ),
        "level": ("ledger_geography_level", "congressional_district"),
        "scope": ("geography_scope", "congressional_district"),
        "geoid": ("congressional_district_geoid", "0101"),
        "name": (None, None),
    }
    metadata_evidence_keys = {key for key, _ in evidence.values() if key is not None}
    assert {key: compiled_spec.metadata[key] for key in metadata_evidence_keys} == {
        "ledger_layout_groupby_dimension": "irs_soi.congressional_district",
        "ledger_source_record_id": source_record_id,
        "ledger_geography_level": "congressional_district",
        "geography_scope": "congressional_district",
        "congressional_district_geoid": "0101",
    }

    cd_specs = []
    for label, (metadata_key, metadata_value) in evidence.items():
        metadata = {
            key: value
            for key, value in compiled_spec.metadata.items()
            if key not in metadata_evidence_keys
        }
        if metadata_key is not None:
            metadata[metadata_key] = metadata_value
        name = f"other.table_1_4.all.cd_{label}_amount"
        if label == "name":
            name = "other.table_1_4.congressional_district_name.bad_amount"
        cd_specs.append(replace(compiled_spec, name=name, metadata=metadata))

    control = replace(
        compiled_spec,
        name="other.table_1_4.all.non_cd_control_amount",
        metadata={
            key: value
            for key, value in compiled_spec.metadata.items()
            if key not in metadata_evidence_keys
        },
    )
    registry = TargetRegistry((*cd_specs, control), country="us")
    publisher_rows = [
        {"name": builder._target_row_name(spec), "metadata": spec.metadata}
        for spec in registry.specs
    ]
    builder_excluded = {
        builder._target_row_name(spec)
        for spec in registry.specs
        if builder._target_is_congressional_district(spec)
    }
    publisher_excluded = {
        str(row["name"])
        for row in publisher_rows
        if _is_congressional_district_layout_target(row)
    }
    expected_excluded = {builder._target_row_name(spec) for spec in cd_specs}

    assert builder_excluded == publisher_excluded == expected_excluded
    assert len(builder_excluded) == len(publisher_excluded) == 6
    assert not builder._target_is_congressional_district(control)
    assert not _is_congressional_district_layout_target(publisher_rows[-1])
    assert not is_congressional_district_target(123, None)
    assert is_congressional_district_target(
        "ordinary",
        UserDict({"geography_scope": "congressional_district"}),
    )

    diagnostics = tuple(
        SimpleNamespace(
            name=builder._target_row_name(spec),
            target=100.0,
            initial_estimate=100.0,
            final_estimate=100.0 if spec is control else 200.0,
            relative_error=0.0 if spec is control else 1.0,
        )
        for spec in registry.specs
    )
    result = SimpleNamespace(
        skipped=(),
        diagnostics=_passing_critical_diagnostics(builder) + diagnostics,
        problem=SimpleNamespace(
            names=tuple(builder._target_row_name(spec) for spec in registry.specs),
            targets=tuple(spec.to_target() for spec in registry.specs),
        ),
        initial_loss=10.0,
        final_loss=5.0,
    )
    assert (
        builder._release_gate_failures(
            result,
            {"dropped_target_names": []},
            target_registry=registry,
        )
        == []
    )

    diagnostics_by_name = {
        diagnostic.name: diagnostic
        for diagnostic in (*_passing_critical_diagnostics(builder), *diagnostics)
    }
    specs_by_name = {builder._target_row_name(spec): spec for spec in registry.specs}
    publisher_diagnostics = {
        "targets": [
            {
                "name": diagnostic.name,
                "target": diagnostic.target,
                "final_estimate": diagnostic.final_estimate,
                "relative_error": diagnostic.relative_error,
                "metadata": dict(
                    getattr(specs_by_name.get(diagnostic.name), "metadata", {})
                ),
                "registry": {
                    "family": getattr(
                        specs_by_name.get(diagnostic.name),
                        "family",
                        "",
                    )
                },
            }
            for diagnostic in diagnostics_by_name.values()
        ]
    }
    publisher_failures: list[str] = []
    _check_us_critical_target_fit(publisher_diagnostics, publisher_failures)
    assert publisher_failures == []


def test_release_gate_failures_reject_bad_critical_target_fit() -> None:
    builder = _load_builder_module()
    result = SimpleNamespace(
        skipped=(),
        diagnostics=(
            SimpleNamespace(
                name=(
                    "irs_soi.ty2022.historic_table_2.us.all."
                    f"income_tax_liability_amount@{builder.PERIOD}"
                ),
                target=2_105_345_646_000.0,
                initial_estimate=2_000_000_000_000.0,
                final_estimate=735_173_331_468.564,
                relative_error=0.0,
            ),
            *_passing_critical_diagnostics(builder)[1:],
        ),
        initial_loss=10.0,
        final_loss=5.0,
    )

    failures = builder._release_gate_failures(
        result,
        {"dropped_target_names": []},
    )

    assert len(failures) == 2
    assert "stale relative_error" in failures[0]
    assert "federal income tax liability amount" in failures[1]
    assert "relative_error=-0.650806" in failures[1]


def test_release_gate_failures_reject_missing_critical_targets() -> None:
    builder = _load_builder_module()
    result = SimpleNamespace(
        skipped=(),
        diagnostics=_passing_critical_diagnostics(builder)[1:],
        initial_loss=10.0,
        final_loss=5.0,
    )

    failures = builder._release_gate_failures(
        result,
        {"dropped_target_names": []},
    )

    assert failures == [
        "Critical fiscal target "
        "'irs_soi.ty2022.historic_table_2.us.all."
        f"income_tax_liability_amount@{builder.PERIOD}' "
        "(federal income tax liability amount) is missing from calibration "
        "diagnostics."
    ]


def test_builder_critical_register_covers_publish_contract() -> None:
    from populace.data.contract import _US_CRITICAL_TARGET_FIT_REQUIREMENTS

    builder = _load_builder_module()
    publish_by_id = {
        requirement.requirement_id: requirement
        for requirement in _US_CRITICAL_TARGET_FIT_REQUIREMENTS
    }
    builder_by_id = {
        requirement.requirement_id: requirement
        for requirement in builder.US_CRITICAL_TARGET_FIT_REQUIREMENTS
    }
    table_builder = builder.US_SOI_TABLE_1_4_NATIONAL_DOLLAR_FIT_REQUIREMENT

    assert set(builder_by_id) >= set(publish_by_id) - {table_builder.requirement_id}
    for requirement_id, publish in publish_by_id.items():
        if requirement_id == table_builder.requirement_id:
            _assert_table_requirement_matches_shared(table_builder, publish)
            continue
        built = builder_by_id[requirement_id]
        assert built.max_abs_relative_error <= publish.max_abs_relative_error
        assert set(built.names) >= set(publish.names)
        assert set(built.families) >= set(publish.families)
        assert set(built.target_roles) >= set(publish.target_roles)
        assert set(built.name_substrings) >= set(publish.name_substrings)
        assert set(built.name_suffixes) >= set(publish.name_suffixes)
        if not publish.allow_incumbent_improvement:
            assert not built.allow_incumbent_improvement


def test_builder_anti_drift_guard_rejects_any_prefix_narrowing() -> None:
    from dataclasses import replace

    from populace.data.contract import _US_CRITICAL_TARGET_FIT_REQUIREMENTS

    builder = _load_builder_module()
    table_builder = builder.US_SOI_TABLE_1_4_NATIONAL_DOLLAR_FIT_REQUIREMENT
    shared = next(
        requirement
        for requirement in _US_CRITICAL_TARGET_FIT_REQUIREMENTS
        if requirement.requirement_id == table_builder.requirement_id
    )
    narrowed = replace(
        table_builder,
        accepted_name_prefixes=("any-prefix-narrows-a-conjunctive-selector.",),
    )

    with pytest.raises(AssertionError):
        _assert_table_requirement_matches_shared(narrowed, shared)


def test_builder_behaviorally_contains_publisher_critical_rejections() -> None:
    builder = _load_builder_module()

    def row(
        name: str,
        *,
        target: float,
        final_estimate: float,
        relative_error: float | None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            name=name,
            target=target,
            initial_estimate=target,
            final_estimate=final_estimate,
            relative_error=relative_error,
        )

    exact = row(
        "irs_soi.ty2022.historic_table_2.us.all.income_tax_liability_amount@2024",
        target=100.0,
        final_estimate=200.0,
        relative_error=1.0,
    )
    alias_spec = TargetSpec(
        name="adversarial_income_tax_alias",
        entity="household",
        measure="income_tax",
        value=100.0,
        period=builder.PERIOD,
        source="fixture",
        family="irs_soi",
        metadata={"target_role": "federal_income_tax_total"},
    )
    semantic_alias = row(
        builder._target_row_name(alias_spec),
        target=100.0,
        final_estimate=200.0,
        relative_error=1.0,
    )
    table_pattern = row(
        "other.table_1_4.all.bad_amount@2024",
        target=100.0,
        final_estimate=200.0,
        relative_error=1.0,
    )
    missing_error = row(
        "irs_soi.ty2023.table_1_4.all.adversarial_amount@2024",
        target=100.0,
        final_estimate=100.0,
        relative_error=None,
    )
    nonfinite_target = row(
        "other.table_1_4.all.nonfinite_target_amount@2024",
        target=float("nan"),
        final_estimate=100.0,
        relative_error=float("nan"),
    )
    nonfinite_final = row(
        "other.table_1_4.all.nonfinite_final_amount@2024",
        target=100.0,
        final_estimate=float("inf"),
        relative_error=float("inf"),
    )
    nonfinite_recorded = row(
        "other.table_1_4.all.nonfinite_recorded_amount@2024",
        target=100.0,
        final_estimate=100.0,
        relative_error=float("nan"),
    )
    incumbent_escape = row(
        "irs_soi.ty2022.historic_table_2.us.all.itemized_deductions_amount@2024",
        target=100.0,
        final_estimate=125.0,
        relative_error=0.25,
    )
    incumbent = {
        incumbent_escape.name: {
            "target": 100.0,
            "final_estimate": 300.0,
        }
    }
    # Round-2 boundary: np.isclose's additive rtol+atol formula accepts this
    # 1.05e-9 stale delta at |computed|=0.1; math.isclose (the publish
    # contract's predicate) rejects it. Both gates must reject.
    narrowly_stale = row(
        "other.table_1_4.all.round2_stale_amount@2024",
        target=100.0,
        final_estimate=110.0,
        relative_error=0.10000000105000001,
    )
    # allow_incumbent_improvement=True requirement pushed just past the 0.25
    # improvement hard stop: improving on the incumbent must not save it.
    beyond_hard_stop_final = 125.0000001
    beyond_hard_stop = row(
        "irs_soi.ty2022.historic_table_2.us.all.income_tax_liability_amount@2024",
        target=100.0,
        final_estimate=beyond_hard_stop_final,
        relative_error=(beyond_hard_stop_final - 100.0) / 100.0,
    )
    beyond_hard_stop_incumbent = {
        beyond_hard_stop.name: {
            "target": 100.0,
            "final_estimate": 300.0,
        }
    }
    cases = (
        ("exact name", exact, (), None),
        ("family and role alias", semantic_alias, (alias_spec,), None),
        ("Table 1.4 substring and suffix", table_pattern, (), None),
        ("missing recorded relative error", missing_error, (), None),
        ("non-finite target", nonfinite_target, (), None),
        ("non-finite final estimate", nonfinite_final, (), None),
        ("non-finite recorded error", nonfinite_recorded, (), None),
        ("incumbent improvement disallowed by law", incumbent_escape, (), incumbent),
        ("narrowly stale recorded error", narrowly_stale, (), None),
        (
            "improvement past the 0.25 hard stop",
            beyond_hard_stop,
            (),
            beyond_hard_stop_incumbent,
        ),
    )

    baseline_builder, baseline_publisher = _critical_contract_failures(
        builder,
        _passing_critical_diagnostics(builder),
    )
    assert baseline_publisher == baseline_builder == []

    for label, adversarial, specs, incumbent_rows in cases:
        builder_failures, publisher_failures = _critical_contract_failures(
            builder,
            _critical_surface(builder, adversarial),
            specs=specs,
            incumbent=incumbent_rows,
        )
        assert any(adversarial.name in failure for failure in publisher_failures), label
        assert any(adversarial.name in failure for failure in builder_failures), label

    # Pass-side boundaries, asserted on BOTH consumers so drift in either
    # direction trips the battery:
    # a 0.9e-9 stale delta is inside math.isclose tolerance;
    within_tolerance = row(
        "other.table_1_4.all.round2_within_tolerance_amount@2024",
        target=100.0,
        final_estimate=110.0,
        relative_error=0.1000000009,
    )
    # an allow-enabled row exactly AT the 0.25 hard stop, improving on a 2.0
    # incumbent, legitimately passes via incumbent improvement on both sides.
    at_hard_stop = row(
        "irs_soi.ty2022.historic_table_2.us.all.income_tax_liability_amount@2024",
        target=100.0,
        final_estimate=125.0,
        relative_error=0.25,
    )
    at_hard_stop_incumbent = {
        at_hard_stop.name: {"target": 100.0, "final_estimate": 300.0}
    }
    for label, passing, incumbent_rows in (
        ("narrowly stale within shared tolerance", within_tolerance, None),
        (
            "improvement exactly at the 0.25 hard stop",
            at_hard_stop,
            at_hard_stop_incumbent,
        ),
    ):
        builder_failures, publisher_failures = _critical_contract_failures(
            builder,
            _critical_surface(builder, passing),
            incumbent=incumbent_rows,
        )
        assert not any(passing.name in failure for failure in publisher_failures), label
        assert not any(passing.name in failure for failure in builder_failures), label


def test_builder_critical_gate_matches_publish_role_aliases() -> None:
    builder = _load_builder_module()
    alias_spec = TargetSpec(
        name="irs_soi.ty2023.table_1_2.all_returns.all."
        "total_itemized_deductions_amount",
        entity="household",
        measure="itemized_deductions",
        value=100.0,
        period=builder.PERIOD,
        source="fixture",
        family="irs_soi",
        metadata={"target_role": "itemized_deduction_total"},
    )
    alias_diagnostic = SimpleNamespace(
        name=f"{alias_spec.name}@{alias_spec.period}",
        target=100.0,
        initial_estimate=100.0,
        final_estimate=150.0,
        relative_error=0.5,
    )
    result = SimpleNamespace(
        skipped=(),
        diagnostics=_passing_critical_diagnostics(builder) + (alias_diagnostic,),
        problem=SimpleNamespace(
            names=(alias_diagnostic.name,),
            targets=(alias_spec.to_target(),),
        ),
        initial_loss=10.0,
        final_loss=5.0,
    )
    registry = TargetRegistry((alias_spec,), country="us")

    failures = builder._release_gate_failures(
        result,
        {"dropped_target_names": []},
        target_registry=registry,
    )

    assert len(failures) == 1
    assert "total_itemized_deductions_amount@2024" in failures[0]
    assert "relative_error=0.5" in failures[0]
    assert "exceeding 0.15" in failures[0]


def test_builder_critical_gate_rejects_medical_incumbent_escape() -> None:
    builder = _load_builder_module()
    medical_name = (
        "irs_soi.ty2022.historic_table_2.us.all."
        f"medical_dental_expense_amount@{builder.PERIOD}"
    )
    # Past the row's own absolute cap (medical sits at the adjudicated 0.25
    # bound, 2026-07-22): even improving on the incumbent never passes it.
    diagnostics = tuple(
        SimpleNamespace(
            **{
                **vars(diagnostic),
                "final_estimate": 104_000_000_000.0,
                "relative_error": 0.3,
            }
        )
        if diagnostic.name == medical_name
        else diagnostic
        for diagnostic in _passing_critical_diagnostics(builder)
    )
    result = SimpleNamespace(
        skipped=(),
        diagnostics=diagnostics,
        initial_loss=10.0,
        final_loss=5.0,
    )

    failures = builder._release_gate_failures(
        result,
        {"dropped_target_names": []},
        incumbent_diagnostics={
            medical_name: {
                "target": 80_000_000_000.0,
                "final_estimate": 240_000_000_000.0,
            }
        },
    )

    assert len(failures) == 1
    assert "medical_dental_expense_amount@2024" in failures[0]
    assert "relative_error=0.3" in failures[0]


def test_fiscal_target_loss_weights_ignore_roles_and_geography() -> None:
    builder = _load_builder_module()
    registry = TargetRegistry(
        (
            TargetSpec(
                name="national_critical_role",
                entity="household",
                measure="national_critical_role",
                value=100.0,
                source="fixture",
                metadata={"target_role": "federal_income_tax_total"},
            ),
            TargetSpec(
                name="state_role_row",
                entity="household",
                measure="state_role_row",
                value=100.0,
                source="fixture",
                metadata={"state_fips": "06", "target_role": "tanf_total"},
            ),
            TargetSpec(
                name="ordinary_distribution_row",
                entity="household",
                measure="ordinary_distribution_row",
                value=100.0,
                source="fixture",
            ),
        ),
        country="us",
    )

    weights = builder._fiscal_target_loss_weights(registry)

    assert weights.shape == (3,)
    assert weights.mean() == 1.0
    assert np.array_equal(weights, np.ones(3))


def test_fiscal_target_loss_weights_hold_concept_budget_when_geography_expands() -> (
    None
):
    builder = _load_builder_module()

    def spec(name: str, value: float, **metadata: str) -> TargetSpec:
        return TargetSpec(
            name=name,
            entity="household",
            measure=metadata.get("variable", "amount"),
            value=value,
            source="fixture",
            metadata={
                "source_measure_id": "amount",
                "source_period": "2024",
                "target_role": "fixture_distribution",
                "measure_mode": "sum",
                **metadata,
            },
        )

    national_income_tax = spec(
        "income_tax_national",
        100.0,
        variable="income_tax",
        ledger_geography_level="country",
        ledger_geography_id="0100000US",
    )
    ctc_national = spec(
        "ctc_national",
        400.0,
        variable="ctc",
        ledger_geography_level="country",
        ledger_geography_id="0100000US",
    )
    ctc_cd_1 = spec(
        "ctc_cd_1",
        100.0,
        variable="ctc",
        ledger_geography_level="congressional_district",
        ledger_geography_id="5001700US0101",
        ledger_geography_name="Alabama Congressional District 1",
        congressional_district_geoid="0101",
        state_fips="01",
    )
    ctc_cd_2 = spec(
        "ctc_cd_2",
        100.0,
        variable="ctc",
        ledger_geography_level="congressional_district",
        ledger_geography_id="5001700US0102",
        ledger_geography_name="Alabama Congressional District 2",
        congressional_district_geoid="0102",
        state_fips="01",
    )

    base_weights = builder._fiscal_target_loss_weights(
        TargetRegistry((national_income_tax, ctc_national), country="us")
    )
    one_child_weights = builder._fiscal_target_loss_weights(
        TargetRegistry(
            (national_income_tax, ctc_national, ctc_cd_1),
            country="us",
        )
    )
    two_child_weights = builder._fiscal_target_loss_weights(
        TargetRegistry(
            (national_income_tax, ctc_national, ctc_cd_1, ctc_cd_2),
            country="us",
        )
    )

    assert np.isclose(base_weights[1] / base_weights.sum(), 2 / 3)
    assert np.isclose(
        one_child_weights[2:].sum() / one_child_weights.sum(),
        two_child_weights[2:].sum() / two_child_weights.sum(),
    )
    assert two_child_weights[1] > two_child_weights[2:].sum()
    assert two_child_weights[2] == two_child_weights[3]


def test_fiscal_target_loss_weights_budget_unparented_cd_rows_by_concept() -> None:
    builder = _load_builder_module()

    def cd_spec(name: str, geoid: str) -> TargetSpec:
        return TargetSpec(
            name=name,
            entity="household",
            measure="tax_filer_individual_count",
            value=100.0,
            source="fixture",
            metadata={
                "source_measure_id": "tax_filer_individual_count",
                "source_period": "2023",
                "target_role": "soi_fiscal_distribution",
                "variable": "tax_filer_individual_count",
                "source_variable": "tax_filer_individual_count",
                "measure_mode": "sum",
                "ledger_geography_level": "congressional_district",
                "ledger_geography_id": f"5001700US{geoid}",
                "ledger_geography_name": f"Congressional District {geoid}",
                "congressional_district_geoid": geoid,
                "state_fips": geoid[:2],
            },
        )

    comparison = TargetSpec(
        name="comparison_amount",
        entity="household",
        measure="adjusted_gross_income",
        value=100.0,
        source="fixture",
        metadata={
            "source_measure_id": "adjusted_gross_income",
            "source_period": "2023",
            "target_role": "soi_fiscal_distribution",
            "variable": "adjusted_gross_income",
            "source_variable": "adjusted_gross_income",
            "measure_mode": "sum",
        },
    )
    one_cd_registry = TargetRegistry(
        (comparison, cd_spec("cd_1", "0101")), country="us"
    )
    many_cd_registry = TargetRegistry(
        (
            comparison,
            cd_spec("cd_1", "0101"),
            cd_spec("cd_2", "0102"),
            cd_spec("cd_3", "0103"),
            cd_spec("cd_4", "0104"),
        ),
        country="us",
    )

    one_cd_weights = builder._fiscal_target_loss_weights(one_cd_registry)
    many_cd_weights = builder._fiscal_target_loss_weights(many_cd_registry)

    assert np.isclose(one_cd_weights[1:].sum() / one_cd_weights.sum(), 0.5)
    assert np.isclose(many_cd_weights[1:].sum() / many_cd_weights.sum(), 0.5)
    assert np.allclose(many_cd_weights[1:], many_cd_weights[1])


def test_fiscal_target_loss_weights_scale_by_sqrt_value_within_basis() -> None:
    builder = _load_builder_module()
    registry = TargetRegistry(
        (
            TargetSpec(
                name="amount_small",
                entity="household",
                measure="amount_small",
                value=100.0,
                source="fixture",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="amount_large",
                entity="household",
                measure="amount_large",
                value=300.0,
                source="fixture",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="returns_small",
                entity="household",
                measure="returns_small",
                value=10.0,
                source="fixture",
                metadata={
                    "source_measure_id": "income_tax_liability_returns",
                    "measure_mode": "indicator_sum",
                },
            ),
            TargetSpec(
                name="returns_large",
                entity="household",
                measure="returns_large",
                value=30.0,
                source="fixture",
                metadata={
                    "source_measure_id": "ctc_claims",
                    "measure_mode": "indicator_sum",
                },
            ),
        ),
        country="us",
    )

    weights = builder._fiscal_target_loss_weights(registry)

    assert weights.mean() == 1.0
    assert np.isclose(weights[1] / weights[0], np.sqrt(3.0))
    assert np.isclose(weights[3] / weights[2], np.sqrt(3.0))
    assert weights[0] == weights[2]
    assert weights[1] == weights[3]


def test_fiscal_target_loss_weights_apply_family_multipliers() -> None:
    builder = _load_builder_module()
    registry = TargetRegistry(
        (
            TargetSpec(
                name="snap_state_row",
                entity="household",
                measure="snap_state_row",
                value=100.0,
                source="fixture",
                family="usda_snap",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="ordinary_amount_row",
                entity="household",
                measure="ordinary_amount_row",
                value=100.0,
                source="fixture",
                family="other_family",
                metadata={"source_measure_id": "payment_amount"},
            ),
        ),
        country="us",
    )

    base_weights = builder._fiscal_target_loss_weights(registry)
    boosted_weights = builder._fiscal_target_loss_weights(registry, {"usda_snap": 8.0})

    assert np.isclose(boosted_weights.mean(), 1.0)
    assert np.isclose(
        boosted_weights[0] / boosted_weights[1],
        8.0 * base_weights[0] / base_weights[1],
    )

    with pytest.raises(ValueError, match="matches no compiled target"):
        builder._fiscal_target_loss_weights(registry, {"missing_family": 2.0})


def test_fiscal_target_loss_weights_split_evenly_between_amount_and_count() -> None:
    builder = _load_builder_module()
    registry = TargetRegistry(
        (
            TargetSpec(
                name="amount_small",
                entity="household",
                measure="amount_small",
                value=100.0,
                source="fixture",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="amount_large",
                entity="household",
                measure="amount_large",
                value=300.0,
                source="fixture",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="returns",
                entity="household",
                measure="returns",
                value=10.0,
                source="fixture",
                metadata={
                    "source_measure_id": "ctc_claims",
                    "measure_mode": "indicator_sum",
                },
            ),
        ),
        country="us",
    )

    weights = builder._fiscal_target_loss_weights(registry)
    bases = np.asarray(
        [builder._fiscal_target_value_basis(spec) for spec in registry.specs],
        dtype=object,
    )

    assert weights.mean() == 1.0
    assert weights[bases == "amount"].sum() == weights[bases == "count"].sum()
    assert np.isclose(weights[1] / weights[0], np.sqrt(3.0))


def test_fiscal_target_loss_weights_floor_zero_subunit_and_abs_values() -> None:
    builder = _load_builder_module()
    registry = TargetRegistry(
        (
            TargetSpec(
                name="zero",
                entity="household",
                measure="zero",
                value=0.0,
                source="fixture",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="subunit",
                entity="household",
                measure="subunit",
                value=0.25,
                source="fixture",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="negative",
                entity="household",
                measure="negative",
                value=-9.0,
                source="fixture",
                signed=True,
                metadata={"source_measure_id": "payment_amount"},
            ),
        ),
        country="us",
    )

    weights = builder._fiscal_target_loss_weights(registry)

    assert weights.mean() == 1.0
    assert weights[0] == weights[1]
    assert np.isclose(weights[2] / weights[0], 3.0)


def test_fiscal_target_value_basis_uses_only_amount_and_count() -> None:
    builder = _load_builder_module()
    amount = TargetSpec(
        name="amount",
        entity="household",
        measure="amount",
        value=100.0,
        source="fixture",
        metadata={"source_measure_id": "payment_amount"},
    )
    return_count = TargetSpec(
        name="return_count",
        entity="household",
        measure="return_count",
        value=100.0,
        source="fixture",
        metadata={
            "source_measure_id": "ctc_claims",
            "measure_mode": "indicator_sum",
        },
    )
    person_count = TargetSpec(
        name="person_count",
        entity="household",
        measure="person_count",
        value=100.0,
        source="fixture",
        metadata={
            "measure_mode": "indicator_sum",
            "source_measure_id": "aptc_recipients",
            "target_role": "aca_ptc_recipients",
            "indicator_map_to": "person",
        },
    )
    bronze_count = TargetSpec(
        name="bronze_count",
        entity="household",
        measure="bronze_count",
        value=100.0,
        source="fixture",
        metadata={
            "measure_mode": "less_than_indicator_sum",
            "source_measure_id": "bronze_aptc_consumers",
            "target_role": "aca_bronze_aptc_consumers",
        },
    )

    assert builder._fiscal_target_value_basis(amount) == "amount"
    assert builder._fiscal_target_value_basis(return_count) == "count"
    assert builder._fiscal_target_value_basis(person_count) == "count"
    assert builder._fiscal_target_value_basis(bronze_count) == "count"


def test_release_calibration_diagnostics_include_gate_failures(
    monkeypatch, tmp_path
) -> None:
    builder = _load_builder_module()
    captured: dict[str, object] = {}

    def fake_write_calibration_diagnostics(result, path, *, target_registry, build):
        captured["result"] = result
        captured["path"] = path
        captured["target_registry"] = target_registry
        captured["build"] = build
        return path

    monkeypatch.setattr(builder, "_sha256", lambda path: "base-sha")
    monkeypatch.setattr(
        builder, "write_calibration_diagnostics", fake_write_calibration_diagnostics
    )
    result = SimpleNamespace()
    registry = TargetRegistry((), country="us")
    profile_gate = SimpleNamespace(passed=True, failures=(), details={"n": 1})
    health_gate = SimpleNamespace(passed=True, failures=(), details={"n": 2})
    base_population_gate = SimpleNamespace(
        passed=True,
        failures=(),
        details={"population": 334_200_000.0},
    )

    builder._write_release_calibration_diagnostics(
        result=result,
        release_dir=tmp_path,
        registry=registry,
        base_h5=tmp_path / "base.h5",
        compilation={"dropped_target_names": []},
        target_profile_gate=profile_gate,
        health_input_gate=health_gate,
        base_population_gate=base_population_gate,
        support_value_repairs={"social_security_components": {"applied": True}},
        audit_export_targets=False,
        gate_failures=["ctc failed"],
        timing={
            "target_compilation_seconds": 1.25,
            "calibration_seconds": 2.5,
        },
    )

    assert captured["path"] == tmp_path / "calibration_diagnostics.json"
    build = captured["build"]
    assert build["base_dataset_sha256"] == "base-sha"
    assert build["target_loss_weighting"].endswith("_cap_100pct")
    assert build["target_loss_cap"] == 1.0
    assert build["release_gates"] == {
        "passed": False,
        "failures": ["ctc failed"],
    }
    assert build["health_input_signal"] == {
        "passed": True,
        "failures": [],
        "details": {"n": 2},
    }
    assert build["base_population_scale"] == {
        "passed": True,
        "failures": [],
        "details": {"population": 334_200_000.0},
    }
    assert build["support_value_repairs"] == {
        "social_security_components": {"applied": True}
    }
    assert build["timing"] == {
        "target_compilation_seconds": 1.25,
        "calibration_seconds": 2.5,
    }


def test_release_calibration_diagnostics_writes_nan_final_loss_as_null(
    small_frame,
    tmp_path,
) -> None:
    builder = _load_builder_module()
    base_h5 = tmp_path / "base.h5"
    base_h5.write_bytes(b"h5")
    registry = TargetRegistry(
        (
            TargetSpec(
                name="income",
                entity="person",
                measure="income",
                value=500_000.0,
                source="fixture",
            ),
        ),
        country="us",
    )
    result = replace(
        calibrate(
            small_frame,
            registry.to_target_set(),
            epochs=1,
            seed=0,
        ),
        closing_loss=float("nan"),
    )
    passing_gate = builder.GateResult(
        name="passing",
        passed=True,
        details={"checked": True},
    )

    builder._write_release_calibration_diagnostics(
        result=result,
        release_dir=tmp_path,
        registry=registry,
        base_h5=base_h5,
        compilation={"dropped_target_names": []},
        target_profile_gate=passing_gate,
        health_input_gate=passing_gate,
        base_population_gate=passing_gate,
        support_value_repairs={},
        audit_export_targets=False,
        gate_failures=["Calibration final loss is non-finite."],
        # main() applies the committed 012733e scrub before invoking this real
        # writer; keep that call boundary explicit rather than moving the fix.
        default_dataset={"method": "dense_no_l0", "final_loss": None},
    )

    diagnostics = json.loads((tmp_path / "calibration_diagnostics.json").read_text())
    assert diagnostics["final_loss"] is None
    assert diagnostics["build"]["default_dataset"]["final_loss"] is None


@pytest.mark.parametrize(
    "terminal_mode",
    ["merge", "integrity", "retirement", "crash", "telemetry"],
)
def test_main_writes_diagnostics_before_post_calibration_gate_failure(
    monkeypatch, tmp_path, terminal_mode
) -> None:
    """The populace#547 corridor contract, end to end through main().

    ``merge``: SSI delivery + export other-health + the ctc sentinel all
    fail in one run — the batch carries every group in corridor order and
    the diagnostics artifact is written first.
    ``integrity``: the persisted-flag FINAL-INTEGRITY gate reports a
    Bernoulli-law violation; that failure reaches the written diagnostics
    and the single terminal batch while the delivery gate passes and every
    later terminal group still runs.
    ``retirement``: an incomplete consume-only retirement surface reports its
    missing leaves through the same written diagnostics and terminal batch,
    while later terminal groups still run.
    ``crash``: the degraded-mode guards themselves are exercised — the
    health-input evaluation raises, the incumbent path is missing, and
    ``_release_gate_failures`` raises; each records a line instead of
    masking the SSI failure, and the incumbent path is nulled for the
    writer so the caught I/O failure is not replayed at the re-hash.
    ``telemetry``: live telemetry raises while attaching the already-written
    calibration diagnostics; the exception becomes a batch line and every
    later terminal gate group still evaluates before the terminal raise.
    """
    builder = _load_builder_module()
    release_id = "populace-us-2024-gate-failure-test"
    base_h5 = tmp_path / "base.h5"
    weeks_source = tmp_path / "asecpub23csv.zip"
    facts = tmp_path / "facts.jsonl"
    out = tmp_path / "out"
    base_h5.write_bytes(b"h5")
    facts.write_text("{}\n")
    target_spec = TargetSpec(
        name="amount",
        entity="household",
        measure="income",
        value=100.0,
        source="fixture",
        metadata={"source_measure_id": "payment_amount"},
    )
    registry = TargetRegistry((target_spec,), country="us")
    result = SimpleNamespace(
        skipped=(),
        diagnostics=(),
        initial_loss=2.0,
        final_loss=1.0,
        l0_lambda=0.2,
        n_nonzero=2,
        frame=SimpleNamespace(n=lambda entity: 2),
        weight_entity="household",
        selection=SimpleNamespace(n_nonzero=2, final_loss=1.5),
    )
    captured: dict[str, object] = {
        "health_stage_events": [],
        "source_stage_events": [],
        "terminal_gate_events": [],
    }
    retirement_missing_failure = (
        "person columns missing: ['taxable_403b_distributions', 'keogh_distributions']."
    )

    class FakeFrame:
        def n(self, entity):
            assert entity == "household"
            return 4

    class FakeExportFrame:
        def n(self, entity):
            assert entity == "household"
            return 2

        def weights_for(self, entity):
            assert entity == "household"
            return Weights(
                np.asarray([12.0, 35.0]),
                WeightKind.CALIBRATED,
            )

        def table(self, entity):
            assert entity == "household"
            return pd.DataFrame({"household_id": np.asarray([10, 20], dtype="int64")})

    argv = [
        "build_us_fiscal_refresh_release.py",
        "--base-h5",
        str(base_h5),
        "--ledger-facts",
        str(facts),
        "--out",
        str(out),
        "--release-id",
        release_id,
        "--asec-2023-weeks-unemployed-source",
        str(weeks_source),
        "--no-target-frame-checkpoint",
    ]
    if terminal_mode != "telemetry":
        argv.append("--no-staging")
    if terminal_mode == "crash":
        # Nonexistent incumbent: the degraded-mode guard must record the
        # load failure, null the path for the writer (no re-hash replay of
        # the caught I/O error), and still reach the diagnostics artifact.
        argv += [
            "--incumbent-diagnostics",
            str(tmp_path / "missing-incumbent.json"),
        ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(builder, "_git_dirty", lambda: False)
    monkeypatch.setattr(
        builder,
        "_sha256",
        lambda path: "weeks-source-sha" if Path(path) == weeks_source else "base-sha",
    )
    monkeypatch.setattr(builder, "_git_output", lambda *args: "commit")
    if terminal_mode == "telemetry":

        class LiveTelemetry:
            run_id = "live-telemetry-test"

            def stage(self, stage, **details):
                captured.setdefault("telemetry_events", []).append(("stage", stage))

            def attach_artifact(self, name, path, **details):
                captured.setdefault("telemetry_events", []).append(
                    ("attach_artifact", name)
                )
                if name == "calibration_diagnostics" and not captured.get(
                    "telemetry_crashed"
                ):
                    captured["telemetry_crashed"] = True
                    raise RuntimeError(
                        "calibration diagnostics attach exploded "
                        "[telemetry-crash-sentinel]"
                    )

            def calibration_progress(self, event):
                captured.setdefault("telemetry_events", []).append(
                    ("calibration_progress", event.get("kind"))
                )

            def complete(self):
                captured.setdefault("telemetry_events", []).append(
                    ("complete", "complete")
                )

        live_telemetry = LiveTelemetry()
        monkeypatch.setattr(
            builder,
            "_staging_telemetry",
            lambda *args, **kwargs: live_telemetry,
        )
    if terminal_mode in {"integrity", "retirement", "telemetry"}:
        monkeypatch.setattr(
            builder,
            "PolicyEngineUSEngine",
            lambda: SimpleNamespace(),
        )

        def fake_input_coverage_gate(frame, engine):
            captured["terminal_gate_events"].append("input_coverage")
            return builder.GateResult(
                name="input_coverage",
                passed=True,
                details={"checked": True},
            )

        def fake_export_input_mass_gate(export_frame, base_frame, **kwargs):
            captured["terminal_gate_events"].append("input_mass_parity")
            return builder.GateResult(
                name="export_input_mass_parity",
                passed=True,
                details={"checked": True},
            )

        def fake_qrf_tail_concentration_gate(
            export_frame,
            *,
            reviewed_exclusions,
        ):
            captured["terminal_gate_events"].append("qrf_tail_concentration")
            return (
                builder.GateResult(
                    name="qrf_tail_concentration",
                    passed=True,
                    details={"reviewed_exclusions": []},
                ),
                {"checked": True},
            )

        monkeypatch.setattr(
            builder,
            "us_release_input_coverage_gate",
            fake_input_coverage_gate,
        )
        monkeypatch.setattr(
            builder,
            "_export_input_mass_gate",
            fake_export_input_mass_gate,
        )
        monkeypatch.setattr(
            builder,
            "_qrf_tail_concentration_gate",
            fake_qrf_tail_concentration_gate,
        )
    # The consistency/contract preflights hit the installed policyengine-us
    # (absent in CI); this test pins diagnostics ordering, not engine metadata.
    monkeypatch.setattr(
        builder, "assert_validation_leaf_registry_current", lambda: None
    )
    monkeypatch.setattr(
        builder, "assert_release_input_coverage_manifest_current", lambda: None
    )
    monkeypatch.setattr(builder, "assert_take_up_contract_current", lambda: None)
    monkeypatch.setattr(builder, "assert_take_up_treatments_consistent", lambda: None)
    monkeypatch.setattr(
        builder,
        "assert_puf_capital_gains_tail_survives_selection",
        lambda base_frame, selected_frame, *, require_present=False: {
            "passed": True,
            "status": "fixture",
        },
    )
    monkeypatch.setattr(
        builder,
        "load_ledger_consumer_artifact",
        lambda path, **kwargs: SimpleNamespace(
            facts=({"fact": 1},),
            provenance=lambda: {
                "path_name": "facts.jsonl",
                "fact_row_count": 1,
                "facts_sha256": "facts-sha",
                "schema_version": None,
                "manifest_sha256": None,
            },
        ),
    )
    monkeypatch.setattr(
        builder,
        "compile_us_fiscal_target_registry",
        lambda facts, **kwargs: registry,
    )
    monkeypatch.setattr(
        builder,
        "target_profile_coverage_gate",
        lambda specs, requirements: builder.GateResult(
            name="target_profile_coverage",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder, "assert_target_parity_manifest_current", lambda **kwargs: None
    )
    monkeypatch.setattr(
        builder,
        "us_release_target_parity_gate",
        lambda registry, **kwargs: builder.GateResult(
            name="us_release_target_parity",
            passed=True,
            details={"checked": True},
        ),
    )

    def fake_load_frame(path):
        captured["source_stage_events"].append("load_frame")
        return FakeFrame()

    monkeypatch.setattr(builder, "_load_frame", fake_load_frame)

    def fake_load_weeks_unemployed_source(path, **kwargs):
        captured["source_stage_events"].append("load_weeks_source")
        captured["weeks_unemployed_source_path"] = Path(path)
        captured["weeks_unemployed_source_load_kwargs"] = kwargs
        return pd.DataFrame({"LKWEEKS": [0, 12]})

    def fake_with_weeks_unemployed(
        frame,
        *,
        seed,
        time_period,
        asec_2023_source,
    ):
        captured["source_stage_events"].append("weeks_stage")
        captured["weeks_unemployed_stage_seed"] = seed
        captured["weeks_unemployed_stage_period"] = time_period
        captured["weeks_unemployed_stage_source"] = asec_2023_source
        return frame

    def fake_weeks_unemployed_signal_gate(frame):
        captured["source_stage_events"].append("weeks_gate")
        captured["weeks_unemployed_gate_called"] = True
        return builder.GateResult(
            name="weeks_unemployed_signal",
            passed=True,
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "load_asec_2023_weeks_unemployed_source",
        fake_load_weeks_unemployed_source,
    )
    monkeypatch.setattr(
        builder,
        "with_us_weeks_unemployed",
        fake_with_weeks_unemployed,
    )
    monkeypatch.setattr(
        builder,
        "us_weeks_unemployed_signal_gate",
        fake_weeks_unemployed_signal_gate,
    )

    def fake_ssi_reporter_source_ids(frame):
        captured["source_stage_events"].append("ssi_reporters")
        return frozenset({"asec-reporter"})

    monkeypatch.setattr(
        builder,
        "us_ssi_take_up_reporter_source_ids",
        fake_ssi_reporter_source_ids,
    )
    repair_payload = {
        "method": "rescale_household_weights_to_census_person_population",
        "applied": True,
        "factor": 2.0,
    }

    def fake_base_population_mass_repair(frame):
        captured["source_stage_events"].append("population_repair")
        return frame, repair_payload

    monkeypatch.setattr(
        builder,
        "_with_base_population_mass_repair",
        fake_base_population_mass_repair,
    )
    ss_repair_payload = {
        "method": "rescale_social_security_component_leaves_to_ssa_targets",
        "applied": True,
    }
    monkeypatch.setattr(
        builder,
        "_with_social_security_component_value_repair",
        lambda frame, specs: (frame, ss_repair_payload),
    )
    cgd_repair_payload = {
        "method": "rescale_non_sch_d_capital_gains_to_soi_table_1_4_fact",
        "applied": True,
    }
    monkeypatch.setattr(
        builder,
        "_with_non_sch_d_cgd_value_repair",
        lambda frame, specs: (frame, cgd_repair_payload),
    )
    monkeypatch.setattr(
        builder,
        "_base_population_scale_gate",
        lambda frame, *, mass_repair=None: builder.GateResult(
            name="base_population_scale",
            passed=True,
            details={"checked": True, "mass_repair": mass_repair},
        ),
    )
    monkeypatch.setattr(
        builder,
        "with_us_qbi_input_reconciliation",
        lambda frame: frame,
    )
    monkeypatch.setattr(
        builder,
        "us_qbi_inputs_signal_gate",
        lambda frame: builder.GateResult(
            name="qbi_inputs_signal",
            passed=True,
            details={"checked": True},
        ),
    )

    def fake_farm_business_income_signal_gate(frame):
        captured["farm_business_income_gate_called"] = True
        return builder.GateResult(
            name="farm_business_income_signal",
            passed=True,
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "us_farm_business_income_signal_gate",
        fake_farm_business_income_signal_gate,
    )
    monkeypatch.setattr(
        builder,
        "us_domestic_production_ald_signal_gate",
        lambda frame: builder.GateResult(
            name="domestic_production_ald_signal",
            passed=True,
            details={"checked": True},
        ),
    )

    def fake_child_support_signal_gate(frame):
        captured["child_support_gate_called"] = True
        return builder.GateResult(
            name="child_support_inputs_signal",
            passed=True,
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "us_child_support_signal_gate",
        fake_child_support_signal_gate,
    )

    def fake_disability_benefits_signal_gate(frame):
        captured["disability_benefits_gate_called"] = True
        return builder.GateResult(
            name="disability_benefits_signal",
            passed=True,
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "us_disability_benefits_signal_gate",
        fake_disability_benefits_signal_gate,
    )
    monkeypatch.setattr(
        builder,
        "us_workers_compensation_signal_gate",
        lambda frame: builder.GateResult(
            name="workers_compensation_signal",
            passed=True,
            details={"checked": True},
        ),
    )

    def fake_educator_expense_signal_gate(frame):
        captured["educator_expense_gate_called"] = True
        return builder.GateResult(
            name="educator_expense_signal",
            passed=True,
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "us_educator_expense_signal_gate",
        fake_educator_expense_signal_gate,
    )

    def fake_form_4952_election_signal_gate(frame):
        captured["form_4952_election_gate_called"] = True
        return builder.GateResult(
            name="form_4952_election_signal",
            passed=True,
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "us_form_4952_election_signal_gate",
        fake_form_4952_election_signal_gate,
    )

    def fake_salt_refund_income_signal_gate(frame):
        captured["salt_refund_income_gate_called"] = True
        return builder.GateResult(
            name="salt_refund_income_signal",
            passed=True,
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "us_salt_refund_income_signal_gate",
        fake_salt_refund_income_signal_gate,
    )

    def fake_capital_gain_details_signal_gate(frame):
        captured["capital_gain_details_gate_called"] = True
        return builder.GateResult(
            name="capital_gain_details_signal",
            passed=True,
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "us_capital_gain_details_signal_gate",
        fake_capital_gain_details_signal_gate,
    )
    monkeypatch.setattr(
        builder,
        "with_us_childcare_inputs",
        lambda frame, *, seed, time_period, allow_existing_without_source: frame,
    )
    monkeypatch.setattr(
        builder,
        "us_childcare_signal_gate",
        lambda frame: builder.GateResult(
            name="childcare_inputs_signal",
            passed=True,
            details={"checked": True},
        ),
    )

    monkeypatch.setattr(
        builder,
        "with_us_energy_subsidy_input",
        lambda frame, *, seed, time_period, allow_existing_without_source: frame,
    )

    def fake_energy_subsidy_signal_gate(frame):
        captured["energy_subsidy_gate_called"] = True
        return builder.GateResult(
            name="energy_subsidy_signal",
            passed=True,
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "us_energy_subsidy_signal_gate",
        fake_energy_subsidy_signal_gate,
    )
    monkeypatch.setattr(
        builder,
        "us_alimony_signal_gate",
        lambda frame: builder.GateResult(
            name="alimony_inputs_signal",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder,
        "us_casualty_loss_signal_gate",
        lambda frame: builder.GateResult(
            name="casualty_loss_signal",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder,
        "us_misc_itemized_signal_gate",
        lambda frame: builder.GateResult(
            name="misc_itemized_signal",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder,
        "with_us_retirement_contribution_inputs",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "us_retirement_contributions_signal_gate",
        lambda frame: builder.GateResult(
            name="retirement_contributions_signal",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder,
        "with_us_immigration_inputs",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "us_immigration_composition_gate",
        lambda frame: builder.GateResult(
            name="immigration_composition",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder,
        "with_us_take_up_inputs",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "with_us_hours_worked_inputs",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "with_us_snap_take_up_inputs",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "with_us_eligibility_inputs",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "with_us_relationship_inputs",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "with_us_medicare_take_up_input",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "with_us_retirement_distribution_inputs",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "with_us_education_inputs",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "with_us_pregnancy_inputs",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "with_us_wic_claim_input",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "with_us_snap_discretionary_exemption_inputs",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "us_take_up_signal_gate",
        lambda frame: builder.GateResult(
            name="us_take_up_signal",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder,
        "us_take_up_participation_diagnostics",
        lambda frame: {
            "schema_version": 1,
            "classification": "release_diagnostics",
            "programs": [],
            "gate": {"passed": True},
        },
    )
    monkeypatch.setattr(
        builder,
        "us_hours_worked_signal_gate",
        lambda frame: builder.GateResult(
            name="hours_worked_signal",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder,
        "us_snap_take_up_signal_gate",
        lambda frame: builder.GateResult(
            name="snap_take_up_signal",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder,
        "us_eligibility_inputs_signal_gate",
        lambda frame: builder.GateResult(
            name="eligibility_inputs_signal",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder,
        "us_relationship_inputs_signal_gate",
        lambda frame: builder.GateResult(
            name="relationship_inputs_signal",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder,
        "us_medicare_take_up_signal_gate",
        lambda frame: builder.GateResult(
            name="medicare_take_up_input_signal",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder,
        "us_prior_year_income_signal_gate",
        lambda frame: builder.GateResult(
            name="prior_year_income_signal",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder,
        "us_housing_inputs_signal_gate",
        lambda frame: builder.GateResult(
            name="housing_inputs_signal",
            passed=True,
            details={"checked": True},
        ),
    )

    def fake_retirement_distributions_signal_gate(frame):
        missing = terminal_mode == "retirement"
        return builder.GateResult(
            name="retirement_distributions_signal",
            passed=not missing,
            failures=((retirement_missing_failure,) if missing else ()),
            details=(
                {
                    "missing": [
                        "taxable_403b_distributions",
                        "keogh_distributions",
                    ]
                }
                if missing
                else {"checked": True}
            ),
        )

    monkeypatch.setattr(
        builder,
        "us_retirement_distributions_signal_gate",
        fake_retirement_distributions_signal_gate,
    )
    monkeypatch.setattr(
        builder,
        "us_education_inputs_signal_gate",
        lambda frame: builder.GateResult(
            name="education_inputs_signal",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder,
        "us_pregnancy_signal_gate",
        lambda frame: builder.GateResult(
            name="pregnancy_signal",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder,
        "us_wic_claim_signal_gate",
        lambda frame: builder.GateResult(
            name="wic_claim_signal",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder,
        "us_snap_discretionary_exemption_signal_gate",
        lambda frame: builder.GateResult(
            name="snap_discretionary_exemption_signal",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder,
        "fetch_scf_2022_summary_extract",
        lambda *args, **kwargs: Path("rscfp2022.dta"),
    )
    monkeypatch.setattr(
        builder,
        "load_scf_2022_financial_asset_donor",
        lambda path: pd.DataFrame(),
    )

    def fake_load_sipp_financial_asset_donor(
        path,
        *,
        expected_sha256=None,
        expected_size_bytes=None,
    ):
        captured["sipp_financial_asset_donor_path"] = path
        captured["sipp_financial_asset_donor_sha256"] = expected_sha256
        captured["sipp_financial_asset_donor_size_bytes"] = expected_size_bytes
        return pd.DataFrame()

    monkeypatch.setattr(
        builder,
        "fetch_sipp_2023_financial_asset_donor",
        lambda *args, **kwargs: Path("pu2023.csv"),
    )
    monkeypatch.setattr(
        builder,
        "load_sipp_2023_financial_asset_donor",
        fake_load_sipp_financial_asset_donor,
    )

    def fake_with_scf_wealth_inputs(
        frame, *, seed, time_period, scf_donor, sipp_donor=None
    ):
        captured["sipp_scf_wealth_blend_called"] = sipp_donor is not None
        return frame

    monkeypatch.setattr(
        builder,
        "with_us_scf_wealth_inputs",
        fake_with_scf_wealth_inputs,
    )

    def fake_scf_wealth_signal_gate(frame, *, require_sipp_blend=False):
        captured["sipp_scf_wealth_blend_gate_required"] = require_sipp_blend
        return builder.GateResult(
            name="scf_wealth_signal",
            passed=True,
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "us_scf_wealth_signal_gate",
        fake_scf_wealth_signal_gate,
    )
    monkeypatch.setattr(
        builder,
        "fetch_scf_2022_full_extract",
        lambda *args, **kwargs: Path("p22i6.dta"),
    )

    def fake_load_scf_auto_loan_donor(summary_path, full_path):
        captured["scf_auto_summary_path"] = summary_path
        captured["scf_auto_full_path"] = full_path
        return pd.DataFrame()

    def fake_with_scf_auto_loan_inputs(
        frame, *, seed, time_period, scf_auto_loan_donor
    ):
        captured["scf_auto_stage_called"] = True
        return frame

    monkeypatch.setattr(
        builder,
        "load_scf_2022_auto_loan_donor",
        fake_load_scf_auto_loan_donor,
    )
    monkeypatch.setattr(
        builder,
        "with_us_scf_auto_loan_inputs",
        fake_with_scf_auto_loan_inputs,
    )
    monkeypatch.setattr(
        builder,
        "us_scf_auto_loans_signal_gate",
        lambda frame: builder.GateResult(
            name="scf_auto_loans_signal",
            passed=True,
            details={"checked": True},
        ),
    )

    def fake_load_sipp_vehicle_donor(
        path,
        *,
        expected_sha256=None,
        expected_size_bytes=None,
    ):
        captured["sipp_vehicle_donor_path"] = path
        captured["sipp_vehicle_donor_sha256"] = expected_sha256
        captured["sipp_vehicle_donor_size_bytes"] = expected_size_bytes
        return pd.DataFrame()

    def fake_with_sipp_vehicle_inputs(frame, *, seed, time_period, sipp_donor):
        captured["sipp_vehicle_stage_called"] = True
        captured["sipp_vehicle_seed"] = seed
        return frame

    def fake_sipp_vehicles_signal_gate(frame):
        captured["sipp_vehicle_gate_called"] = True
        return builder.GateResult(
            name="sipp_vehicles_signal",
            passed=True,
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "load_sipp_2023_vehicle_donor",
        fake_load_sipp_vehicle_donor,
    )
    monkeypatch.setattr(
        builder,
        "with_us_sipp_vehicle_inputs",
        fake_with_sipp_vehicle_inputs,
    )
    monkeypatch.setattr(
        builder,
        "us_sipp_vehicles_signal_gate",
        fake_sipp_vehicles_signal_gate,
    )

    def fake_load_ssi_disability_donor(
        path,
        *,
        expected_sha256=None,
        expected_size_bytes=None,
        time_period=2024,
    ):
        captured["ssi_disability_donor_path"] = path
        captured["ssi_disability_donor_sha256"] = expected_sha256
        captured["ssi_disability_donor_size_bytes"] = expected_size_bytes
        captured["ssi_disability_donor_period"] = time_period
        return pd.DataFrame()

    def fake_with_ssi_disability_criteria(frame, *, seed, time_period, sipp_donor):
        captured["ssi_disability_stage_called"] = True
        captured["ssi_disability_seed"] = seed
        return frame

    def fake_ssi_disability_signal_gate(frame):
        captured["ssi_disability_gate_called"] = True
        return builder.GateResult(
            name="ssi_disability_criteria_signal",
            passed=True,
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "load_sipp_2023_ssi_disability_donor",
        fake_load_ssi_disability_donor,
    )
    monkeypatch.setattr(
        builder,
        "with_us_ssi_disability_criteria",
        fake_with_ssi_disability_criteria,
    )
    monkeypatch.setattr(
        builder,
        "us_ssi_disability_criteria_signal_gate",
        fake_ssi_disability_signal_gate,
    )

    def fake_load_sipp_head_start_donor(
        path,
        *,
        expected_sha256=None,
        expected_size_bytes=None,
    ):
        captured["sipp_head_start_donor_path"] = path
        captured["sipp_head_start_donor_sha256"] = expected_sha256
        captured["sipp_head_start_donor_size_bytes"] = expected_size_bytes
        return pd.DataFrame()

    def fake_with_sipp_head_start_input(
        frame,
        *,
        seed,
        time_period,
        sipp_donor,
    ):
        captured["sipp_head_start_stage_called"] = True
        captured["sipp_head_start_seed"] = seed
        captured["sipp_head_start_period"] = time_period
        captured["sipp_head_start_donor"] = sipp_donor
        return frame

    def fake_sipp_head_start_signal_gate(frame):
        captured["sipp_head_start_gate_called"] = True
        return builder.GateResult(
            name="sipp_head_start_signal",
            passed=True,
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "load_sipp_2023_head_start_donor",
        fake_load_sipp_head_start_donor,
    )
    monkeypatch.setattr(
        builder,
        "with_us_sipp_head_start_input",
        fake_with_sipp_head_start_input,
    )
    monkeypatch.setattr(
        builder,
        "us_sipp_head_start_signal_gate",
        fake_sipp_head_start_signal_gate,
    )

    def fake_ssi_uncapped_amount(
        frame,
        *,
        simulation=None,
        maximum_microsim_batch_size=None,
    ):
        captured["ssi_uncapped_stage_called"] = True
        captured["ssi_uncapped_batch_size"] = maximum_microsim_batch_size
        return np.zeros(4, dtype=np.float64)

    fake_band_targets = {
        "under_18": 1_001_922.0,
        "18_64": 3_905_779.0,
        "65_plus": 2_382_142.0,
    }

    def fake_band_targets_from_registry(specs):
        captured["ssi_band_targets_specs"] = specs
        return dict(fake_band_targets)

    fake_stage_priors = {"under_18": 0.3, "18_64": 0.4, "65_plus": 0.5}
    # Current schema-4 shape (populace#507/#508): main() reconstructs the prior
    # weight basis from these stage diagnostics with the REAL module helper
    # and threads it into the final release-weight measurement.
    fake_stage_diagnostics = {
        "checked": True,
        "schema_version": 4,
        "measurement_phase": "assignment_stage",
        "prior_weight_basis": {
            "kind": "current_frame",
            "source_sha256": None,
            "source_schema_version": None,
        },
        "age_bands": [
            {
                "age_band": key,
                "assignment_prior": prior,
                "prior_basis_candidate_capacity": 1_000.0,
                "prior_basis_reporter_candidate_floor": 100.0,
            }
            for key, prior in fake_stage_priors.items()
        ],
    }

    def fake_with_ssi_take_up(
        frame,
        *,
        uncapped_ssi,
        seed,
        targets,
        reporter_source_ids,
        prior_basis=None,
    ):
        captured["ssi_take_up_stage_called"] = True
        captured["ssi_take_up_seed"] = seed
        captured["ssi_take_up_uncapped"] = np.asarray(uncapped_ssi)
        captured["ssi_take_up_targets"] = dict(targets)
        captured["ssi_reporter_source_ids"] = reporter_source_ids
        captured["ssi_take_up_prior_basis"] = prior_basis
        return frame, dict(fake_stage_diagnostics)

    def fake_ssi_take_up_gate(diagnostics, *, targets):
        captured["ssi_take_up_gate_called"] = True
        gate_calls = captured.setdefault("ssi_take_up_gate_calls", [])
        gate_calls.append({"diagnostics": diagnostics, "targets": dict(targets)})
        captured.setdefault("ssi_event_order", []).append("integrity_gate")
        final_integrity_failure = terminal_mode == "integrity" and len(gate_calls) == 2
        return builder.GateResult(
            name="ssi_take_up",
            passed=not final_integrity_failure,
            failures=(
                ("Bernoulli-law violation [final-integrity-sentinel]",)
                if final_integrity_failure
                else ()
            ),
            details=diagnostics,
        )

    monkeypatch.setattr(
        builder,
        "_ssi_take_up_band_targets_from_registry",
        fake_band_targets_from_registry,
    )
    monkeypatch.setattr(
        builder,
        "_ssi_person_uncapped_amount",
        fake_ssi_uncapped_amount,
    )
    monkeypatch.setattr(builder, "with_us_ssi_take_up", fake_with_ssi_take_up)
    monkeypatch.setattr(builder, "us_ssi_take_up_gate", fake_ssi_take_up_gate)
    # The digest reads the persisted flag column, which the stub above never
    # writes; the sentinel keeps the checkpoint/cache-identity threading
    # observable without a real assignment (populace#507/#508).
    monkeypatch.setattr(
        builder,
        "_ssi_take_up_assignment_digest",
        lambda frame, *, assignment_priors, prior_basis: "ssi-digest-sentinel",
    )

    def fake_load_voluntary_filing_donor(
        path,
        *,
        expected_sha256=None,
        expected_size_bytes=None,
    ):
        captured["voluntary_filing_donor_path"] = path
        captured["voluntary_filing_donor_sha256"] = expected_sha256
        captured["voluntary_filing_donor_size_bytes"] = expected_size_bytes
        return pd.DataFrame()

    def fake_with_voluntary_filing_input(frame, *, seed, time_period, sipp_donor):
        captured["voluntary_filing_stage_called"] = True
        captured["voluntary_filing_seed"] = seed
        return frame

    def fake_voluntary_filing_signal_gate(frame):
        captured["voluntary_filing_gate_called"] = True
        return builder.GateResult(
            name="voluntary_filing_signal",
            passed=True,
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "load_sipp_2023_voluntary_filing_donor",
        fake_load_voluntary_filing_donor,
    )
    monkeypatch.setattr(
        builder,
        "with_us_voluntary_filing_input",
        fake_with_voluntary_filing_input,
    )
    monkeypatch.setattr(
        builder,
        "us_voluntary_filing_signal_gate",
        fake_voluntary_filing_signal_gate,
    )
    monkeypatch.setattr(
        builder,
        "fetch_sipp_2023_tip_donor",
        lambda *args, **kwargs: Path("pu2023_slim.csv"),
    )

    def fake_load_sipp_tip_donor(path, *, expected_sha256=None):
        captured["sipp_tip_donor_path"] = path
        captured["sipp_tip_donor_sha256"] = expected_sha256
        return pd.DataFrame()

    def fake_with_sipp_tip_inputs(frame, *, seed, time_period, sipp_donor):
        captured["sipp_tip_stage_called"] = True
        return frame

    def fake_sipp_tips_signal_gate(frame):
        captured["sipp_tip_gate_called"] = True
        return builder.GateResult(
            name="sipp_tips_signal",
            passed=True,
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "load_sipp_2023_tip_donor",
        fake_load_sipp_tip_donor,
    )
    monkeypatch.setattr(
        builder,
        "with_us_sipp_tip_inputs",
        fake_with_sipp_tip_inputs,
    )
    monkeypatch.setattr(
        builder,
        "us_sipp_tips_signal_gate",
        fake_sipp_tips_signal_gate,
    )
    monkeypatch.setattr(
        builder,
        "fetch_org_2024_donor",
        lambda *args, **kwargs: Path("census_cps_org_2024_wages.csv.gz"),
    )

    def fake_load_org_donor(path, *, expected_content_sha256=None):
        captured["org_donor_path"] = path
        captured["org_donor_sha256"] = expected_content_sha256
        return pd.DataFrame()

    def fake_with_org_inputs(frame, *, seed, time_period, org_donor):
        captured["org_stage_called"] = True
        return frame

    monkeypatch.setattr(builder, "load_org_2024_donor", fake_load_org_donor)
    monkeypatch.setattr(builder, "with_us_org_wages_inputs", fake_with_org_inputs)
    monkeypatch.setattr(
        builder,
        "us_org_wages_signal_gate",
        lambda frame: builder.GateResult(
            name="org_wages_signal", passed=True, details={"checked": True}
        ),
    )
    monkeypatch.setattr(
        builder,
        "_ecps_parity_gate",
        lambda frame: builder.GateResult(
            name="ecps_parity",
            passed=True,
            details={"checked": True},
        ),
    )

    def fake_with_aca_outputs(
        frame,
        specs,
        *,
        seed,
        maximum_microsim_batch_size=None,
    ):
        captured["health_stage_events"].append("aca")
        return frame

    monkeypatch.setattr(
        builder,
        "_with_aca_marketplace_source_outputs",
        fake_with_aca_outputs,
    )

    def fake_health_input_signal_gate(frame):
        calls = captured.setdefault("health_input_gate_calls", 0) + 1
        captured["health_input_gate_calls"] = calls
        # Like other-health, this gate has a staging callsite (base frame)
        # before the corridor callsite (export frame). The staging call must
        # succeed — a crash there is green-path and rightly raises; only the
        # corridor call exercises the #547 degraded-mode guard.
        if terminal_mode == "crash" and calls > 1:
            raise RuntimeError("health-input exploded [crash-sentinel]")
        return builder.GateResult(
            name="health_input_signal",
            passed=True,
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "_health_input_signal_gate",
        fake_health_input_signal_gate,
    )

    def fake_with_medicaid_outputs(
        frame,
        specs,
        *,
        seed,
        substitutions=(),
        maximum_microsim_batch_size=None,
    ):
        captured["health_stage_events"].append("medicaid")
        return frame, {}

    def fake_medicaid_gate(diagnostics):
        captured["health_stage_events"].append("medicaid_gate")
        return builder.GateResult(
            name="medicaid_take_up",
            passed=True,
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "_with_medicaid_take_up_outputs",
        fake_with_medicaid_outputs,
    )
    monkeypatch.setattr(
        builder,
        "us_medicaid_take_up_gate",
        fake_medicaid_gate,
    )

    def fake_with_other_health_insurance_inputs(
        frame,
        *,
        seed,
        time_period,
        maximum_microsim_batch_size,
    ):
        captured["health_stage_events"].append("other_health")
        captured["other_health_insurance_stage_called"] = True
        captured["other_health_insurance_seed"] = seed
        captured["other_health_insurance_period"] = time_period
        captured["other_health_insurance_batch_size"] = maximum_microsim_batch_size
        return frame

    def fake_other_health_insurance_signal_gate(frame):
        captured["health_stage_events"].append("other_health_gate")
        captured["other_health_insurance_gate_called"] = True
        calls = captured.setdefault("other_health_gate_calls", 0) + 1
        captured["other_health_gate_calls"] = calls
        if calls == 1:
            # Staging call on the base frame passes: the pre-solve gate
            # fails fast by design (nothing to preserve yet).
            return builder.GateResult(
                name="other_health_insurance_premiums_signal",
                passed=True,
                details={"checked": True},
            )
        # Export-frame call fails deliberately: the populace#547 cofailure
        # regression proves a failing post-solve signal gate batches
        # alongside the SSI delivery failure instead of masking it with an
        # in-place raise (the sparse-selection signal-flattening scenario).
        return builder.GateResult(
            name="other_health_insurance_premiums_signal",
            passed=False,
            failures=("premiums signal flattened [cofailure-sentinel]",),
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "with_us_other_health_insurance_inputs",
        fake_with_other_health_insurance_inputs,
    )
    monkeypatch.setattr(
        builder,
        "us_other_health_insurance_signal_gate",
        fake_other_health_insurance_signal_gate,
    )
    monkeypatch.setattr(
        builder,
        "_with_snap_state_take_up_outputs",
        lambda frame, specs, *, seed, maximum_microsim_batch_size=None: (frame, {}),
    )
    monkeypatch.setattr(
        builder,
        "us_snap_state_take_up_gate",
        lambda diagnostics: builder.GateResult(
            name="snap_state_take_up",
            passed=True,
            details={"checked": True},
        ),
    )

    def fake_materialize_target_frame(frame, specs, **kwargs):
        captured["materialize_kwargs"] = kwargs
        return frame, registry, {"dropped_target_names": []}

    def fake_degenerate_input_signal_gate(frame, engine):
        # In retirement mode this gate ALSO fails: the production masking
        # route (PR #557 round 2 finding 1) was the generic degenerate raise
        # superseding the specific missing-leaf diagnosis. The degraded-mode
        # append must carry BOTH lines to the single terminal batch while the
        # run continues through the solve (the #547/#548 evidence contract).
        if terminal_mode == "retirement":
            return builder.GateResult(
                name="degenerate_input_signal",
                passed=False,
                failures=("keogh_distributions flattened [degenerate-sentinel]",),
                details={"checked": True},
            )
        return builder.GateResult(
            name="degenerate_input_signal",
            passed=True,
            details={"checked": True},
        )

    monkeypatch.setattr(
        builder,
        "_degenerate_input_signal_gate",
        fake_degenerate_input_signal_gate,
    )
    monkeypatch.setattr(
        builder,
        "_materialize_target_frame",
        fake_materialize_target_frame,
    )

    def fake_calibrate_l0_refit(*args, **kwargs):
        captured["l0_args"] = args
        captured["l0_kwargs"] = kwargs
        captured["target_loss_weights"] = kwargs["target_loss_weights"]
        captured["target_loss_cap"] = kwargs["target_loss_cap"]
        return result

    real_write_release_diagnostics = builder._write_release_calibration_diagnostics

    def recording_write_release_diagnostics(**kwargs):
        captured["diagnostics"] = kwargs
        return real_write_release_diagnostics(**kwargs)

    def fake_write_calibration_diagnostics(result, path, *, target_registry, build):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "build": {
                        "release_gates": dict(build["release_gates"]),
                    }
                }
            )
        )
        return path

    monkeypatch.setattr(builder, "calibrate_l0_refit", fake_calibrate_l0_refit)

    def fake_l0_refit_weights(frame, refit_result):
        captured["export_frame_from_l0_refit"] = True
        return FakeExportFrame()

    def fake_final_ssi_diagnostics(
        frame,
        *,
        uncapped_ssi,
        seed,
        targets,
        assignment_priors,
        prior_basis,
        reporter_source_ids,
    ):
        captured["final_ssi_diagnostics_called"] = True
        captured["final_ssi_diagnostics_targets"] = dict(targets)
        captured["final_ssi_diagnostics_assignment_priors"] = dict(assignment_priors)
        captured["final_ssi_diagnostics_prior_basis"] = prior_basis
        captured["final_ssi_diagnostics_reporter_source_ids"] = reporter_source_ids
        return {"checked": True}

    def fake_final_medicaid_diagnostics(
        frame,
        specs,
        *,
        seed,
        substitutions,
        maximum_microsim_batch_size=None,
    ):
        captured["final_medicaid_diagnostics_called"] = True
        return {}

    monkeypatch.setattr(builder, "_with_l0_refit_weights", fake_l0_refit_weights)
    monkeypatch.setattr(
        builder,
        "us_ssi_take_up_diagnostics",
        fake_final_ssi_diagnostics,
    )

    def fake_ssi_delivery_gate(diagnostics, *, targets):
        captured["ssi_delivery_gate_called"] = True
        captured["ssi_delivery_gate_targets"] = dict(targets)
        # The integrity and retirement cases pass delivery to isolate their
        # own early failure. Other modes retain the populace#547 delivery
        # cofailure and its written retry basis.
        captured.setdefault("ssi_event_order", []).append("delivery_gate")
        passes = terminal_mode in {"integrity", "retirement"}
        return builder.GateResult(
            name="ssi_take_up_delivery",
            passed=passes,
            failures=(
                ()
                if passes
                else ("18_64 delivered over envelope [cofailure-sentinel]",)
            ),
            details=diagnostics,
        )

    monkeypatch.setattr(
        builder,
        "us_ssi_take_up_delivery_gate",
        fake_ssi_delivery_gate,
    )

    real_ssi_write = builder.write_us_ssi_take_up_diagnostics

    def recording_ssi_write(diagnostics, path):
        captured.setdefault("ssi_event_order", []).append(f"write:{Path(path).name}")
        return real_ssi_write(diagnostics, path)

    monkeypatch.setattr(
        builder,
        "write_us_ssi_take_up_diagnostics",
        recording_ssi_write,
    )
    monkeypatch.setattr(
        builder,
        "_medicaid_diagnostics_for_existing_output",
        fake_final_medicaid_diagnostics,
    )

    def fake_release_gate_failures(*args, **kwargs):
        if terminal_mode == "crash":
            raise RuntimeError("release-gate evaluation exploded [crash-sentinel]")
        if terminal_mode == "retirement":
            # The degraded pre-solve contract (PR #557 round 3): the failing
            # degenerate gate is NOT raised early — the gate object itself
            # must arrive here, failures intact, and ride the single
            # terminal batch. Emitting from the argument (the real
            # function's contract) proves the un-raised object reached us.
            degenerate_gate = args[8]
            assert degenerate_gate is not None
            assert not degenerate_gate.passed
            return [
                *(
                    f"Degenerate input signal failed: {failure}"
                    for failure in degenerate_gate.failures
                ),
                "ctc failed",
            ]
        return ["ctc failed"]

    monkeypatch.setattr(
        builder,
        "_release_gate_failures",
        fake_release_gate_failures,
    )
    monkeypatch.setattr(
        builder,
        "_write_release_calibration_diagnostics",
        recording_write_release_diagnostics,
    )
    monkeypatch.setattr(
        builder,
        "write_calibration_diagnostics",
        fake_write_calibration_diagnostics,
    )

    try:
        builder.main()
    except RuntimeError as exc:
        # populace#547 cofailure contract: the batched report leads with the
        # early terminal failures (SSI delivery + its retry-basis note, then
        # the corridor lines in evaluation order); degraded-mode
        # coverage/parity evaluation errors on the fake frame may append
        # further lines after them.
        message = str(exc)
        if terminal_mode == "retirement":
            assert message.startswith(
                "Release gates failed: Retirement-distribution signal failed: "
                f"{retirement_missing_failure}"
            )
            # The co-failing degenerate gate must batch AFTER the specific
            # retirement diagnosis, never supersede it with an early raise
            # (PR #557 round 2 finding 1).
            assert (
                "Degenerate input signal failed: keogh_distributions "
                "flattened [degenerate-sentinel]" in message
            )
            assert "SSI take-up delivery failed:" not in message
        elif terminal_mode == "integrity":
            assert message.startswith(
                "Release gates failed: SSI take-up final measurement failed: "
                "Bernoulli-law violation [final-integrity-sentinel]"
            )
            assert "SSI take-up delivery failed:" not in message
        else:
            assert message.startswith(
                "Release gates failed: SSI take-up delivery failed: "
                "18_64 delivered over envelope [cofailure-sentinel]"
            )
        assert (
            "Other health insurance signal failed on the export frame: "
            "premiums signal flattened [cofailure-sentinel]" in message
        )
        if terminal_mode != "crash":
            assert "ctc failed" in message
            if terminal_mode == "telemetry":
                assert (
                    "Terminal-batch telemetry "
                    "attach_artifact('calibration_diagnostics') crashed" in message
                )
                assert "telemetry-crash-sentinel" in message
        else:
            assert "health-input exploded [crash-sentinel]" in message
            assert "release-gate evaluation exploded [crash-sentinel]" in message
            assert "ctc failed" not in message
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected post-calibration gate failure.")

    release_dir = out / "releases" / release_id
    written_diagnostics = json.loads(
        (release_dir / "calibration_diagnostics.json").read_text()
    )
    if terminal_mode == "retirement":
        assert (
            "Retirement-distribution signal failed: "
            f"{retirement_missing_failure}"
            in written_diagnostics["build"]["release_gates"]["failures"]
        )
    elif terminal_mode == "integrity":
        assert (
            "SSI take-up final measurement failed: "
            "Bernoulli-law violation [final-integrity-sentinel]"
            in written_diagnostics["build"]["release_gates"]["failures"]
        )
    # The SSI retry-basis artifact is written even though the run fails
    # terminally — it IS the remedy input for the next attempt.
    assert (release_dir / "us_ssi_take_up.json").exists()
    final_weights_path = release_dir / "final_household_weights.npy"
    final_ids_path = release_dir / "final_household_weight_ids.npy"
    final_weights_metadata = json.loads(
        (release_dir / "final_household_weights.json").read_text()
    )
    np.testing.assert_array_equal(
        np.load(final_weights_path, allow_pickle=False),
        np.asarray([12.0, 35.0]),
    )
    np.testing.assert_array_equal(
        np.load(final_ids_path, allow_pickle=False),
        np.asarray([10, 20], dtype="int64"),
    )
    # Identity binds the evidence to this run's target-frame context; the
    # ids block reattaches every weight to its household. Their inner
    # values are run-derived, so assert them structurally and compare the
    # stable remainder exactly.
    evidence_identity = final_weights_metadata.pop("identity")
    cache_context = captured["materialize_kwargs"][
        "target_materialization_cache_context"
    ]
    expected_evidence_identity = builder._target_frame_checkpoint_identity(
        base_dataset_sha256=cache_context["base_dataset_sha256"],
        policyengine_us_version=cache_context["policyengine_us_version"],
        seed=cache_context["seed"],
        target_period=cache_context["target_period"],
        target_registry_version=cache_context["target_registry_version"],
        weeks_unemployed_source_sha256=cache_context["weeks_unemployed_source_sha256"],
        congressional_district_vintage_crosswalk_sha256=cache_context[
            "congressional_district_vintage_crosswalk_sha256"
        ],
        ssi_take_up_assignment_sha256=cache_context["ssi_take_up_assignment_sha256"],
        selection_identities_sha256=cache_context["selection_identities_sha256"],
    )
    assert evidence_identity == dict(expected_evidence_identity)
    ids_block = final_weights_metadata.pop("household_ids")
    assert ids_block["file"] == "final_household_weight_ids.npy"
    assert ids_block["shape"] == [2]
    assert (
        ids_block["ordering_sha256"]
        == __import__("hashlib")
        .sha256(np.asarray([10, 20], dtype="int64").tobytes())
        .hexdigest()
    )
    assert final_weights_metadata == {
        "artifact_kind": "populace_final_household_weight_evidence",
        "schema_version": 1,
        "measurement_phase": "release_final",
        "entity": "household",
        "weight_kind": "calibrated",
        "values": {
            "file": "final_household_weights.npy",
            "dtype": "float64",
            "shape": [2],
            # This end-to-end fixture stubs every non-weeks file hash; the
            # direct helper test above validates the real hash path.
            "sha256": "base-sha",
        },
        "summary": {
            "n_households": 2,
            "household_weight_sum": 47.0,
            "minimum": 12.0,
            "maximum": 35.0,
            "nonzero_count": 2,
            "zero_count": 0,
        },
    }
    # Artifact exclusion: the failed run leaves evidence, never artifacts.
    # H5s land under the out root (not the release dir), so sweep the tree.
    assert not list(out.rglob("*.h5"))
    assert not list(release_dir.glob("*manifest*"))
    if terminal_mode == "telemetry":
        assert captured["telemetry_crashed"] is True
    if terminal_mode in {"integrity", "retirement", "telemetry"}:
        assert captured["terminal_gate_events"] == [
            "input_coverage",
            "input_mass_parity",
            "qrf_tail_concentration",
        ]
    if terminal_mode != "crash":
        if terminal_mode == "retirement":
            expected_gate_failures = [
                f"Retirement-distribution signal failed: {retirement_missing_failure}",
                "Other health insurance signal failed on the export frame: "
                "premiums signal flattened [cofailure-sentinel]",
                # The co-failing pre-solve degenerate gate is never raised
                # early and never duplicated: its line arrives once, through
                # _release_gate_failures (PR #557 round 3).
                "Degenerate input signal failed: keogh_distributions "
                "flattened [degenerate-sentinel]",
                "ctc failed",
            ]
        elif terminal_mode == "integrity":
            expected_gate_failures = [
                "SSI take-up final measurement failed: "
                "Bernoulli-law violation [final-integrity-sentinel]",
                "Medicaid final diagnostics not evaluated: SSI decision "
                "integrity failed upstream (Bernoulli-law violation) and "
                "Medicaid eligibility consumes the frozen SSI decisions; "
                "quarantined instead of mis-measured (populace#547).",
                "Other health insurance signal failed on the export frame: "
                "premiums signal flattened [cofailure-sentinel]",
                "ctc failed",
            ]
        else:
            # The retry line carries the written artifact's sha256 — the
            # required --ssi-take-up-prior-weight-basis-sha256 pin, handed out
            # by the failure itself (sol round 2, new minor).
            import hashlib

            written_sha = hashlib.sha256(
                (release_dir / "us_ssi_take_up.json").read_bytes()
            ).hexdigest()
            expected_gate_failures = [
                "SSI take-up delivery failed: 18_64 delivered over envelope "
                "[cofailure-sentinel]",
                "SSI take-up delivered-weight prior basis written to "
                f"{release_dir / 'us_ssi_take_up.json'} (sha256 {written_sha}) "
                "for the --ssi-take-up-prior-weight-basis retry.",
                "Other health insurance signal failed on the export frame: "
                "premiums signal flattened [cofailure-sentinel]",
                "ctc failed",
            ]
        assert captured["diagnostics"]["gate_failures"] == expected_gate_failures
    else:
        # Corridor order: SSI delivery + basis note, health-input crash
        # guard, other-health gate failure, incumbent guard, release-gate
        # crash guard. Exact error suffixes vary (OS error text), so pin
        # order-exact prefixes.
        expected_prefixes = [
            "SSI take-up delivery failed: 18_64 delivered over envelope",
            "SSI take-up delivered-weight prior basis written to",
            "Health-input signal evaluation crashed in degraded mode",
            "Other health insurance signal failed on the export frame:",
            "Incumbent diagnostics could not be loaded/validated in degraded mode",
            "Release gate evaluation crashed in degraded mode",
        ]
        actual = captured["diagnostics"]["gate_failures"]
        assert len(actual) == len(expected_prefixes), actual
        for line, prefix in zip(actual, expected_prefixes, strict=True):
            assert line.startswith(prefix), (line, prefix)
        # The caught incumbent I/O failure must not be replayed at the
        # writer's re-hash: the path is nulled for the writer.
        assert captured["diagnostics"]["incumbent_diagnostics_path"] is None
    assert (
        captured["diagnostics"]["base_population_gate"].details["mass_repair"]
        == repair_payload
    )
    assert captured["diagnostics"]["support_value_repairs"] == {
        "social_security_components": ss_repair_payload,
        "non_sch_d_capital_gains": cgd_repair_payload,
    }
    assert captured["diagnostics"]["default_dataset"] == {
        "method": "l0_refit",
        "sparse": True,
        "n_candidate_households": 4,
        "n_selected_households": 2,
        "n_exported_households": 2,
        "l0_lambda_share": 0.8,
        "l0_lambda": 0.2,
        "selection_epochs": 1500,
        "refit_epochs": 1500,
        "selection_l2_lambda": 0.0,
        "refit_l2_lambda": 0.0,
        "selection_final_loss": 1.5,
        "refit_initial_loss": 2.0,
        "refit_final_loss": 1.0,
        "final_loss": 1.0,
    }
    assert captured["l0_kwargs"]["l0_lambda"] == 0.2
    assert captured["l0_kwargs"]["l2_lambda"] == 0.0
    assert captured["l0_kwargs"]["refit_l2_lambda"] is None
    assert captured["l0_kwargs"]["epochs"] == 1500
    assert captured["l0_kwargs"]["refit_epochs"] == 1500
    assert captured["l0_kwargs"]["warm_start_weights"] is None
    assert captured["target_loss_cap"] == 1.0
    assert np.array_equal(captured["target_loss_weights"], np.asarray([1.0]))
    assert (
        captured["materialize_kwargs"]["target_materialization_cache_dir"]
        == out / "artifacts" / "target_materialization_cache"
    )
    assert not captured["materialize_kwargs"]["gate_congressional_district_targets"]
    assert captured["sipp_tip_donor_path"] == Path("pu2023_slim.csv")
    assert captured["weeks_unemployed_source_path"] == weeks_source
    assert captured["weeks_unemployed_stage_seed"] == 0
    assert captured["weeks_unemployed_stage_period"] == builder.PERIOD
    assert isinstance(captured["weeks_unemployed_stage_source"], pd.DataFrame)
    assert captured["weeks_unemployed_gate_called"] is True
    assert captured["source_stage_events"].index("weeks_stage") < captured[
        "source_stage_events"
    ].index("ssi_reporters")
    assert captured["source_stage_events"].index("weeks_gate") < captured[
        "source_stage_events"
    ].index("population_repair")
    assert (
        captured["materialize_kwargs"]["target_materialization_cache_context"][
            "weeks_unemployed_source_sha256"
        ]
        == builder.ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_SHA256
    )
    assert captured["sipp_tip_donor_sha256"] == builder.SIPP_2023_TIP_DONOR_SHA256
    assert captured["sipp_tip_stage_called"] is True
    assert captured["sipp_tip_gate_called"] is True
    assert captured["sipp_financial_asset_donor_path"] == Path("pu2023.csv")
    assert (
        captured["sipp_financial_asset_donor_sha256"]
        == builder.SIPP_2023_FINANCIAL_ASSET_DONOR_SHA256
    )
    assert (
        captured["sipp_financial_asset_donor_size_bytes"]
        == builder.SIPP_2023_FINANCIAL_ASSET_DONOR_SIZE_BYTES
    )
    assert captured["sipp_scf_wealth_blend_called"] is True
    assert captured["sipp_scf_wealth_blend_gate_required"] is True
    assert captured["sipp_vehicle_donor_path"] == Path("pu2023.csv")
    assert (
        captured["sipp_vehicle_donor_sha256"] == builder.SIPP_2023_VEHICLE_DONOR_SHA256
    )
    assert (
        captured["sipp_vehicle_donor_size_bytes"]
        == builder.SIPP_2023_VEHICLE_DONOR_SIZE_BYTES
    )
    assert captured["sipp_vehicle_stage_called"] is True
    assert captured["sipp_vehicle_seed"] == 42
    assert captured["sipp_vehicle_gate_called"] is True
    assert captured["ssi_disability_donor_path"] == Path("pu2023.csv")
    assert (
        captured["ssi_disability_donor_sha256"]
        == builder.SIPP_2023_SSI_DISABILITY_DONOR_SHA256
    )
    assert (
        captured["ssi_disability_donor_size_bytes"]
        == builder.SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES
    )
    assert captured["ssi_disability_donor_period"] == builder.PERIOD
    assert captured["ssi_disability_stage_called"] is True
    assert captured["ssi_disability_seed"] == 42
    assert captured["ssi_disability_gate_called"] is True
    assert captured["sipp_head_start_donor_path"] == Path("pu2023.csv")
    assert (
        captured["sipp_head_start_donor_sha256"]
        == builder.SIPP_2023_HEAD_START_DONOR_SHA256
    )
    assert (
        captured["sipp_head_start_donor_size_bytes"]
        == builder.SIPP_2023_HEAD_START_DONOR_SIZE_BYTES
    )
    assert captured["sipp_head_start_stage_called"] is True
    assert captured["sipp_head_start_seed"] == 0
    assert captured["sipp_head_start_period"] == builder.PERIOD
    assert isinstance(captured["sipp_head_start_donor"], pd.DataFrame)
    assert captured["sipp_head_start_gate_called"] is True
    assert captured["ssi_uncapped_stage_called"] is True
    assert (
        captured["ssi_uncapped_batch_size"]
        == builder.DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE
    )
    assert captured["ssi_take_up_stage_called"] is True
    assert captured["ssi_take_up_seed"] == 0
    assert captured["ssi_take_up_uncapped"].shape == (4,)
    assert captured["ssi_take_up_targets"] == fake_band_targets
    assert captured["ssi_reporter_source_ids"] == frozenset({"asec-reporter"})
    assert captured["ssi_take_up_gate_called"] is True
    # The gate binds twice: the fresh stage diagnostics at assignment time,
    # then the persisted-flag measurement on the export frame (PR #477
    # review finding 3) — both against the registry band targets.
    gate_calls = captured["ssi_take_up_gate_calls"]
    assert [call["diagnostics"] for call in gate_calls] == [
        fake_stage_diagnostics,
        {"checked": True},
    ]
    assert all(call["targets"] == fake_band_targets for call in gate_calls)
    # One-shot regime (populace#469): the frozen flags are measured on the
    # release weights, never reassigned or reconciled, and the final
    # measurement republishes the stage's assignment priors verbatim.
    assert captured["export_frame_from_l0_refit"] is True
    assert captured["final_ssi_diagnostics_called"] is True
    assert captured["final_ssi_diagnostics_targets"] == fake_band_targets
    assert captured["final_ssi_diagnostics_assignment_priors"] == fake_stage_priors
    assert captured["final_ssi_diagnostics_reporter_source_ids"] == frozenset(
        {"asec-reporter"}
    )
    # No CLI basis: the stage draws on current-frame capacities, and the final
    # measurement republishes the basis reconstructed from the stage's own
    # diagnostics (populace#507/#508) — then the delivery gate binds it.
    assert captured["ssi_take_up_prior_basis"] is None
    final_basis = captured["final_ssi_diagnostics_prior_basis"]
    assert final_basis.kind == "current_frame"
    assert final_basis.band("65_plus").candidate_capacity == pytest.approx(1_000.0)
    assert captured["ssi_delivery_gate_called"] is True
    assert captured["ssi_delivery_gate_targets"] == fake_band_targets
    # The frozen-assignment digest invalidates the materialization cache on
    # any retry whose flags differ (populace#507/#508 split-brain fix).
    cache_context = captured["materialize_kwargs"][
        "target_materialization_cache_context"
    ]
    assert cache_context["ssi_take_up_assignment_sha256"] == "ssi-digest-sentinel"
    assert cache_context["selection_identities_sha256"] is None
    expected_materializer_identity = builder._target_frame_checkpoint_identity(
        base_dataset_sha256=cache_context["base_dataset_sha256"],
        policyengine_us_version=cache_context["policyengine_us_version"],
        seed=cache_context["seed"],
        target_period=cache_context["target_period"],
        target_registry_version=cache_context["target_registry_version"],
        weeks_unemployed_source_sha256=cache_context["weeks_unemployed_source_sha256"],
        congressional_district_vintage_crosswalk_sha256=cache_context[
            "congressional_district_vintage_crosswalk_sha256"
        ],
        ssi_take_up_assignment_sha256=cache_context["ssi_take_up_assignment_sha256"],
        selection_identities_sha256=cache_context["selection_identities_sha256"],
    )
    assert cache_context[
        "target_frame_materializer_identity_sha256"
    ] == builder._target_frame_checkpoint_digest(expected_materializer_identity)
    # Evidence-first ordering (sol round 2, findings 3/10, reconciled with
    # the #548 batched terminal gates): the final measurement hits disk
    # BEFORE the final integrity gate runs. Delivery still evaluates after
    # an integrity failure; on a delivery miss, the enforce helper rewrites
    # the same artifact and its failure line carries the sha pin.
    expected_ssi_event_order = [
        "integrity_gate",  # stage diagnostics, at assignment time
        "write:us_ssi_take_up.json",  # final measurement, written first
        "integrity_gate",  # persisted-flag recheck on the export frame
        "delivery_gate",  # enforced-band delivery, after the artifact exists
    ]
    if terminal_mode not in {"integrity", "retirement"}:
        # A delivery miss rewrites the final measurement as the retry basis.
        expected_ssi_event_order.append("write:us_ssi_take_up.json")
    assert captured["ssi_event_order"] == expected_ssi_event_order
    if terminal_mode == "integrity":
        assert "final_medicaid_diagnostics_called" not in captured
    else:
        assert captured["final_medicaid_diagnostics_called"] is True
    assert captured["voluntary_filing_donor_path"] == Path("pu2023.csv")
    assert (
        captured["voluntary_filing_donor_sha256"]
        == builder.SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256
    )
    assert (
        captured["voluntary_filing_donor_size_bytes"]
        == builder.SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES
    )
    assert captured["voluntary_filing_stage_called"] is True
    assert captured["voluntary_filing_seed"] == 0
    assert captured["voluntary_filing_gate_called"] is True
    assert captured["org_donor_path"] == Path("census_cps_org_2024_wages.csv.gz")
    assert captured["org_donor_sha256"] == builder.ORG_2024_DONOR_CONTENT_SHA256
    assert captured["org_stage_called"] is True
    assert captured["scf_auto_summary_path"] == Path("rscfp2022.dta")
    assert captured["scf_auto_full_path"] == Path("p22i6.dta")
    assert captured["scf_auto_stage_called"] is True
    assert captured["child_support_gate_called"] is True
    assert captured["disability_benefits_gate_called"] is True
    assert captured["educator_expense_gate_called"] is True
    assert captured["form_4952_election_gate_called"] is True
    assert captured["salt_refund_income_gate_called"] is True
    assert captured["capital_gain_details_gate_called"] is True
    assert captured["energy_subsidy_gate_called"] is True
    assert captured["farm_business_income_gate_called"] is True
    assert captured["other_health_insurance_stage_called"] is True
    assert captured["other_health_insurance_gate_called"] is True
    assert captured["other_health_insurance_seed"] == 0
    assert captured["other_health_insurance_period"] == builder.PERIOD
    assert (
        captured["other_health_insurance_batch_size"]
        == builder.DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE
    )
    assert captured["health_stage_events"] == [
        "aca",
        "medicaid",
        "medicaid_gate",
        "other_health",
        "other_health_gate",
        # The export-frame signal re-check (one-shot regime, populace#469).
        "other_health_gate",
    ]


def test_release_gate_failures_reject_bad_national_credit_and_ss_fits() -> None:
    builder = _load_builder_module()
    cases = (
        (
            "irs_soi.ty2022.historic_table_2.us.all.ctc_amount",
            "Child Tax Credit amount",
            82_863_353_000.0,
            99_282_300_000.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.ctc_claims",
            "Child Tax Credit claims",
            38_068_980.0,
            43_994_700.0,
        ),
        (
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_amount",
            "Earned Income Tax Credit amount",
            69_041_649_000.0,
            83_000_000_000.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_amount",
            "Premium Tax Credit amount",
            53_910_190_000.0,
            84_823_800_000.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_returns",
            "Premium Tax Credit returns",
            7_841_370.0,
            11_637_100.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_amount",
            "taxable Social Security amount",
            455_904_900_000.0,
            540_351_000_000.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_returns",
            "taxable Social Security returns",
            24_475_100.0,
            31_887_700.0,
        ),
    )

    for target_name, label, target, final_estimate in cases:
        diagnostics = list(_passing_critical_diagnostics(builder))
        name = f"{target_name}@{builder.PERIOD}"
        index = next(
            i for i, diagnostic in enumerate(diagnostics) if diagnostic.name == name
        )
        diagnostics[index] = SimpleNamespace(
            name=name,
            target=target,
            initial_estimate=target,
            final_estimate=final_estimate,
            relative_error=(final_estimate - target) / target,
        )
        result = SimpleNamespace(
            skipped=(),
            diagnostics=tuple(diagnostics),
            initial_loss=10.0,
            final_loss=5.0,
        )

        failures = builder._release_gate_failures(
            result,
            {"dropped_target_names": []},
        )

        assert len(failures) == 1
        assert label in failures[0]
        assert "exceeding 0.15" in failures[0]


def test_critical_gate_allows_eitc_amount_within_credit_tolerance() -> None:
    builder = _load_builder_module()
    name = (
        "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
        f"earned_income_credit.total_earned_income_credit_amount@{builder.PERIOD}"
    )
    target = 69_041_649_000.0
    diagnostics = list(_passing_critical_diagnostics(builder))
    index = next(
        i for i, diagnostic in enumerate(diagnostics) if diagnostic.name == name
    )
    diagnostics[index] = SimpleNamespace(
        name=name,
        target=target,
        initial_estimate=target,
        final_estimate=58_954_970_066.74941,
        relative_error=(58_954_970_066.74941 - target) / target,
    )
    result = SimpleNamespace(
        skipped=(),
        diagnostics=tuple(diagnostics),
        initial_loss=10.0,
        final_loss=5.0,
    )

    assert builder._release_gate_failures(result, {"dropped_target_names": []}) == []


def test_critical_gate_allows_bounded_improvement_over_incumbent() -> None:
    builder = _load_builder_module()
    name = f"irs_soi.ty2022.historic_table_2.us.all.ctc_amount@{builder.PERIOD}"
    target = 82_863_353_000.0
    diagnostics = list(_passing_critical_diagnostics(builder))
    index = next(
        i for i, diagnostic in enumerate(diagnostics) if diagnostic.name == name
    )
    diagnostics[index] = SimpleNamespace(
        name=name,
        target=target,
        initial_estimate=99_315_000_000.0,
        final_estimate=99_282_300_000.0,
        relative_error=(99_282_300_000.0 - target) / target,
    )
    result = SimpleNamespace(
        skipped=(),
        diagnostics=tuple(diagnostics),
        initial_loss=10.0,
        final_loss=5.0,
    )
    incumbent = {
        name: {
            "target": target,
            "final_estimate": 134_904_000_000.0,
        }
    }

    assert (
        builder._release_gate_failures(
            result,
            {"dropped_target_names": []},
            incumbent_diagnostics=incumbent,
        )
        == []
    )


def test_incumbent_diagnostics_must_match_current_target_surface(tmp_path) -> None:
    builder = _load_builder_module()
    incumbent_path = tmp_path / "calibration_diagnostics.json"
    current_surface = {"sha256": "a" * 64, "n_targets": 33_127}
    incumbent_payload = {
        "target_surface": {"sha256": "b" * 64, "n_targets": 6_877},
        "targets": [],
    }

    with pytest.raises(
        RuntimeError,
        match="Score the incumbent on the current target surface",
    ):
        builder._assert_incumbent_target_surface_matches(
            current_surface,
            incumbent_payload,
            path=incumbent_path,
        )


def test_legacy_cd_provenance_requires_crosswalk_metadata() -> None:
    scorer = _load_scorer_module()

    with pytest.raises(
        ValueError,
        match="requires --congressional-district-vintage-crosswalk",
    ):
        scorer._assert_legacy_cd_provenance_options(
            allow_legacy_cd_provenance=True,
            congressional_district_vintage_crosswalk_metadata=None,
        )

    scorer._assert_legacy_cd_provenance_options(
        allow_legacy_cd_provenance=True,
        congressional_district_vintage_crosswalk_metadata={"sha256": "x"},
    )


def test_scorer_accepts_legacy_pe_flat_h5_flag(
    monkeypatch,
) -> None:
    scorer = _load_scorer_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_us_fiscal_targets.py",
            "--h5",
            "enhanced_cps_2024.h5",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "score",
            "--legacy-pe-flat-h5",
        ],
    )

    args = scorer._parse_args()

    assert args.legacy_pe_flat_h5


def test_legacy_pe_flat_h5_loads_entity_frame(
    tmp_path,
) -> None:
    scorer = _load_scorer_module()
    h5_path = tmp_path / "legacy_us_data.h5"

    def write_column(name: str, values: object) -> None:
        with h5py.File(h5_path, "a") as h5:
            group = h5.create_group(name)
            group.create_dataset(str(scorer.release.PERIOD), data=np.asarray(values))

    write_column("person_id", [1, 2, 3, 4])
    write_column("person_household_id", [10, 10, 20, 20])
    write_column("person_tax_unit_id", [100, 100, 200, 300])
    write_column("person_spm_unit_id", [1000, 1000, 1000, 1000])
    write_column("person_family_id", [2000, 2000, 2001, 2002])
    write_column("person_marital_unit_id", [3000, 3001, 3002, 3003])
    write_column("household_id", [10, 20])
    write_column("household_weight", [100.0, 200.0])
    write_column("state_fips", [6, 36])
    write_column("tax_unit_id", [100, 200, 300])
    write_column("spm_unit_id", [1000])
    write_column("family_id", [2000, 2001, 2002])
    write_column("marital_unit_id", [3000, 3001, 3002, 3003])
    write_column("age", [40, 38, 10, 7])
    write_column("income_tax", [1_000.0, 2_000.0, 3_000.0])
    write_column("unknown_household_signal", [1, 0])
    write_column("bad_matrix", [[1, 2], [3, 4]])

    frame, metadata = scorer._load_legacy_pe_flat_frame(
        h5_path,
        variable_entity_by_name={
            "age": "person",
            "income_tax": "tax_unit",
            "state_fips": "household",
        },
    )

    assert frame.n("person") == 4
    assert frame.n("household") == 2
    assert frame.n("tax_unit") == 3
    assert frame.table("person")["age"].tolist() == [40, 38, 10, 7]
    assert frame.table("tax_unit")["income_tax"].tolist() == [
        1_000.0,
        2_000.0,
        3_000.0,
    ]
    assert frame.table("household")["unknown_household_signal"].tolist() == [1, 0]
    assert frame.weights_for("household").values.tolist() == [100.0, 200.0]
    assert metadata["layout"] == "legacy_pe_flat_h5"
    assert metadata["inferred_unknown_columns_by_entity"] == {
        "household": ["unknown_household_signal"]
    }
    assert any(
        skipped["column"] == "bad_matrix" and "not one-dimensional" in skipped["reason"]
        for skipped in metadata["skipped_columns"]
    )


def test_legacy_pe_flat_h5_drops_zero_weight_households(
    tmp_path,
) -> None:
    scorer = _load_scorer_module()
    h5_path = tmp_path / "legacy_us_data_zero_weights.h5"

    def write_column(name: str, values: object) -> None:
        with h5py.File(h5_path, "a") as h5:
            group = h5.create_group(name)
            group.create_dataset(str(scorer.release.PERIOD), data=np.asarray(values))

    write_column("person_id", [1, 2, 3, 4])
    write_column("person_household_id", [10, 10, 20, 20])
    write_column("person_tax_unit_id", [100, 100, 200, 200])
    write_column("person_spm_unit_id", [1000, 1000, 2000, 2000])
    write_column("person_family_id", [3000, 3000, 4000, 4000])
    write_column("person_marital_unit_id", [5000, 5001, 6000, 6001])
    write_column("household_id", [10, 20])
    write_column("household_weight", [100.0, 0.0])
    write_column("tax_unit_id", [100, 200])
    write_column("spm_unit_id", [1000, 2000])
    write_column("family_id", [3000, 4000])
    write_column("marital_unit_id", [5000, 5001, 6000, 6001])
    write_column("age", [40, 38, 10, 7])

    frame, metadata = scorer._load_legacy_pe_flat_frame(
        h5_path,
        variable_entity_by_name={"age": "person"},
    )

    assert frame.n("household") == 1
    assert frame.n("person") == 2
    assert frame.table("household")["household_id"].tolist() == [10]
    assert frame.table("person")["person_id"].tolist() == [1, 2]
    assert frame.weights_for("household").values.tolist() == [100.0]
    assert metadata["dropped_zero_weight_households"] == 1
    assert metadata["dropped_zero_weight_persons"] == 2


def test_critical_gate_rejects_improved_miss_past_hard_stop() -> None:
    builder = _load_builder_module()
    name = f"irs_soi.ty2022.historic_table_2.us.all.ctc_amount@{builder.PERIOD}"
    target = 82_863_353_000.0
    diagnostics = list(_passing_critical_diagnostics(builder))
    index = next(
        i for i, diagnostic in enumerate(diagnostics) if diagnostic.name == name
    )
    diagnostics[index] = SimpleNamespace(
        name=name,
        target=target,
        initial_estimate=99_315_000_000.0,
        final_estimate=105_000_000_000.0,
        relative_error=(105_000_000_000.0 - target) / target,
    )
    result = SimpleNamespace(
        skipped=(),
        diagnostics=tuple(diagnostics),
        initial_loss=10.0,
        final_loss=5.0,
    )
    incumbent = {
        name: {
            "target": target,
            "final_estimate": 134_904_000_000.0,
        }
    }

    failures = builder._release_gate_failures(
        result,
        {"dropped_target_names": []},
        incumbent_diagnostics=incumbent,
    )

    assert len(failures) == 1
    assert "Child Tax Credit amount" in failures[0]
    assert "exceeding 0.15" in failures[0]
    assert "incumbent_relative_error=" in failures[0]
    assert "improvement_hard_stop=0.25" in failures[0]


def test_critical_gate_rejects_miss_when_incumbent_is_better() -> None:
    builder = _load_builder_module()
    name = f"irs_soi.ty2022.historic_table_2.us.all.ctc_amount@{builder.PERIOD}"
    target = 82_863_353_000.0
    diagnostics = list(_passing_critical_diagnostics(builder))
    index = next(
        i for i, diagnostic in enumerate(diagnostics) if diagnostic.name == name
    )
    diagnostics[index] = SimpleNamespace(
        name=name,
        target=target,
        initial_estimate=99_315_000_000.0,
        final_estimate=99_282_300_000.0,
        relative_error=(99_282_300_000.0 - target) / target,
    )
    result = SimpleNamespace(
        skipped=(),
        diagnostics=tuple(diagnostics),
        initial_loss=10.0,
        final_loss=5.0,
    )
    incumbent = {
        name: {
            "target": target,
            "final_estimate": 90_000_000_000.0,
        }
    }

    failures = builder._release_gate_failures(
        result,
        {"dropped_target_names": []},
        incumbent_diagnostics=incumbent,
    )

    assert len(failures) == 1
    assert "Child Tax Credit amount" in failures[0]
    assert "incumbent_relative_error=" in failures[0]


def test_health_input_signal_gate_rejects_degenerate_aca_inputs() -> None:
    builder = _load_builder_module()

    class FakeFrame:
        def table(self, name):
            assert name == "tax_unit"
            return pd.DataFrame(
                {
                    "takes_up_aca_if_eligible": [True, True, True],
                    "selected_marketplace_plan_benchmark_ratio": [1.0, 1.0, 1.0],
                }
            )

    gate = builder._health_input_signal_gate(FakeFrame())

    assert not gate.passed
    assert gate.name == "health_input_signal"
    assert len(gate.failures) == 2
    assert any("takes_up_aca_if_eligible" in failure for failure in gate.failures)
    assert any(
        "selected_marketplace_plan_benchmark_ratio" in failure
        for failure in gate.failures
    )


def test_health_input_signal_gate_accepts_varied_aca_inputs() -> None:
    builder = _load_builder_module()

    class FakeFrame:
        def table(self, name):
            assert name == "tax_unit"
            return pd.DataFrame(
                {
                    "takes_up_aca_if_eligible": [True, False, True],
                    "selected_marketplace_plan_benchmark_ratio": [1.0, 0.8, 1.2],
                }
            )

    gate = builder._health_input_signal_gate(FakeFrame())

    assert gate.passed
    assert gate.details["unique_counts"] == {
        "selected_marketplace_plan_benchmark_ratio": 3,
        "takes_up_aca_if_eligible": 2,
    }
    ratio_diagnostics = gate.details["selected_marketplace_plan_benchmark_ratio"]
    assert ratio_diagnostics["support"] == {"lower": 0.5, "upper": 1.5}
    assert ratio_diagnostics["all_tax_units"] == {
        "count": 3,
        "min": 0.8,
        "max": 1.2,
        "mean": 1.0,
        "neutral_count": 1,
        "below_benchmark_count": 1,
        "above_benchmark_count": 1,
        "below_support_count": 0,
        "above_support_count": 0,
    }
    marketplace_takers = ratio_diagnostics["marketplace_takers"]
    assert marketplace_takers["count"] == 2
    assert abs(marketplace_takers["mean"] - 1.1) < 1e-12
    assert marketplace_takers["below_benchmark_count"] == 0
    assert marketplace_takers["above_benchmark_count"] == 1


def test_aca_source_runtime_refreshes_degenerate_release_inputs(monkeypatch) -> None:
    builder = _load_builder_module()
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2], dtype="int64"),
            "person_tax_unit_id": np.asarray([10, 10, 20], dtype="int64"),
            "person_spm_unit_id": np.asarray([100, 100, 200], dtype="int64"),
            "person_family_id": np.asarray([1000, 1000, 2000], dtype="int64"),
            "person_marital_unit_id": np.asarray([10000, 10000, 20000], dtype="int64"),
            "has_marketplace_health_coverage_at_interview": [False, False, True],
        }
    )
    frame = Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {
                    "household_id": np.asarray([1, 2], dtype="int64"),
                    "state_fips": np.asarray([1, 1]),
                }
            ),
            "tax_unit": pd.DataFrame(
                {
                    "tax_unit_id": np.asarray([10, 20], dtype="int64"),
                    "stable_tax_unit_draw": [0.1, 0.2],
                    "takes_up_aca_if_eligible": [False, False],
                    "selected_marketplace_plan_benchmark_ratio": [1.0, 1.0],
                }
            ),
            "spm_unit": pd.DataFrame({"spm_unit_id": [100, 200]}),
            "family": pd.DataFrame({"family_id": [1000, 2000]}),
            "marital_unit": pd.DataFrame({"marital_unit_id": [10000, 20000]}),
        },
        builder.US_SCHEMA,
        {
            "household": builder.Weights(
                values=np.asarray([1.0, 1.0]),
                kind=WeightKind.DESIGN,
            )
        },
    )
    specs = (
        TargetSpec(
            name="cms_aca.oep2024.state_marketplace.al.aptc_recipients",
            entity="household",
            measure="takes_up_aca_if_eligible",
            value=3.0,
            source="CMS Marketplace OEP",
            family="cms_aca",
            metadata={
                "target_role": "aca_ptc_recipients",
                "state_fips": "01",
                "ledger_geography_level": "state",
            },
        ),
    )
    values = {
        "is_aca_ptc_eligible": np.asarray([1.0, 1.0]),
        "is_aca_ptc_eligible:person": np.asarray([1.0, 1.0, 1.0]),
        "health_insurance_premiums_without_medicare_part_b": np.asarray(
            [400.0, 1200.0]
        ),
        "assigned_aca_ptc": np.asarray([0.0, 0.0]),
        "aca_ptc": np.asarray([100.0, 0.0]),
        "slcsp": np.asarray([1000.0, 1000.0]),
    }

    def fake_calculate_array(simulation, variable, *, map_to=None):
        assert simulation is fake_simulation
        assert map_to in {"tax_unit", "person"}
        if map_to == "person":
            return values[f"{variable}:person"]
        return values[variable]

    fake_simulation = object()
    monkeypatch.setattr(builder, "_calculate_array", fake_calculate_array)

    refreshed = builder._with_aca_marketplace_source_outputs(
        frame,
        specs,
        seed=42,
        simulation=fake_simulation,
    )

    tax_unit = refreshed.table("tax_unit")
    assigned = tax_unit.set_index("tax_unit_id")["takes_up_aca_if_eligible"]
    assert bool(assigned.loc[10]) is True
    assert bool(assigned.loc[20]) is False
    assert tax_unit["takes_up_aca_if_eligible"].nunique() == 2
    assert tax_unit["selected_marketplace_plan_benchmark_ratio"].nunique() == 2
    person_counts = person.assign(
        assigned=person["person_tax_unit_id"].map(assigned).fillna(False)
    )
    assert float(person_counts["assigned"].sum()) == 2.0
    assert builder._health_input_signal_gate(refreshed).passed
    assert frame.table("tax_unit")["takes_up_aca_if_eligible"].nunique() == 1
    assert (
        frame.table("tax_unit")["selected_marketplace_plan_benchmark_ratio"].nunique()
        == 1
    )


def test_aca_source_tax_unit_table_batches_policyengine_inputs(monkeypatch) -> None:
    builder = _load_builder_module()
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3, 4], dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2, 3], dtype="int64"),
            "person_tax_unit_id": np.asarray([10, 10, 20, 30], dtype="int64"),
            "person_spm_unit_id": np.asarray([100, 100, 200, 300], dtype="int64"),
            "person_family_id": np.asarray([1000, 1000, 2000, 3000], dtype="int64"),
            "person_marital_unit_id": np.asarray(
                [10000, 10000, 20000, 30000], dtype="int64"
            ),
        }
    )
    frame = Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {
                    "household_id": np.asarray([1, 2, 3], dtype="int64"),
                    "state_fips": np.asarray([1, 1, 2]),
                }
            ),
            "tax_unit": pd.DataFrame(
                {
                    "tax_unit_id": np.asarray([10, 20, 30], dtype="int64"),
                    "stable_tax_unit_draw": [0.1, 0.2, 0.3],
                }
            ),
            "spm_unit": pd.DataFrame({"spm_unit_id": [100, 200, 300]}),
            "family": pd.DataFrame({"family_id": [1000, 2000, 3000]}),
            "marital_unit": pd.DataFrame({"marital_unit_id": [10000, 20000, 30000]}),
        },
        builder.US_SCHEMA,
        {
            "household": builder.Weights(
                values=np.asarray([10.0, 20.0, 30.0]),
                kind=WeightKind.DESIGN,
            )
        },
    )
    target_tables = {
        builder.US_ACA_APTC_TARGET_TABLE: pd.DataFrame(
            {
                "state_fips": ["01"],
                "target": [3.0],
            }
        )
    }
    tax_values = {
        10: {
            "is_aca_ptc_eligible": 1.0,
            "aca_ptc": 100.0,
            "health_insurance_premiums_without_medicare_part_b": 400.0,
            "slcsp": 1000.0,
        },
        20: {
            "is_aca_ptc_eligible": 0.0,
            "aca_ptc": 200.0,
            "health_insurance_premiums_without_medicare_part_b": 500.0,
            "slcsp": 1100.0,
        },
        30: {
            "is_aca_ptc_eligible": 1.0,
            "aca_ptc": 300.0,
            "health_insurance_premiums_without_medicare_part_b": 600.0,
            "slcsp": 1200.0,
        },
    }
    person_eligible = {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0}
    seen_tax_unit_batches: list[tuple[int, ...]] = []
    formula_owned_assertions: list[int] = []
    dataset_assert_flags: list[bool | None] = []

    class FakeMicrosimulation:
        def __init__(self, *, dataset):
            self.dataset = dataset
            seen_tax_unit_batches.append(
                tuple(dataset.table("tax_unit")["tax_unit_id"].astype(int))
            )

        def _invalidate_all_caches(self):
            pass

    def fake_calculate_array(simulation, variable, *, map_to=None):
        if map_to == "person":
            return np.asarray(
                [
                    person_eligible[int(person_id)]
                    for person_id in simulation.dataset.table("person")["person_id"]
                ],
                dtype=np.float64,
            )
        assert map_to == "tax_unit"
        return np.asarray(
            [
                tax_values[int(tax_unit_id)][variable]
                for tax_unit_id in simulation.dataset.table("tax_unit")["tax_unit_id"]
            ],
            dtype=np.float64,
        )

    def fake_assert_no_formula_owned_columns(frame_arg):
        formula_owned_assertions.append(frame_arg.n("household"))

    def fake_dataset_from_frame(frame_arg, **kwargs):
        dataset_assert_flags.append(kwargs.get("assert_no_formula_owned_columns"))
        return frame_arg

    monkeypatch.setattr(
        builder,
        "_assert_no_formula_owned_columns",
        fake_assert_no_formula_owned_columns,
    )
    monkeypatch.setattr(builder, "_dataset_from_frame", fake_dataset_from_frame)
    monkeypatch.setattr(builder, "_calculate_array", fake_calculate_array)

    tax_unit = builder._aca_source_tax_unit_table_batched(
        frame,
        target_tables,
        microsimulation_cls=FakeMicrosimulation,
        maximum_microsim_batch_size=1,
    ).set_index("tax_unit_id")

    assert seen_tax_unit_batches == [(10,), (20,), (30,)]
    assert formula_owned_assertions == [3]
    assert dataset_assert_flags == [False, False, False]
    assert tax_unit.loc[10, "tax_unit_weight"] == 20.0
    assert tax_unit.loc[20, "tax_unit_weight"] == 20.0
    assert tax_unit.loc[30, "tax_unit_weight"] == 0.0
    assert bool(tax_unit.loc[10, "is_aca_ptc_eligible"]) is True
    assert bool(tax_unit.loc[20, "is_aca_ptc_eligible"]) is True
    assert bool(tax_unit.loc[30, "is_aca_ptc_eligible"]) is False
    assert tax_unit.loc[10, "assigned_aca_ptc"] == 100.0
    assert (
        tax_unit.loc[20, "health_insurance_premiums_without_medicare_part_b"] == 500.0
    )
    assert tax_unit.loc[30, "slcsp"] == 1200.0
    assert tax_unit.loc[10, "aca_take_up_rate"] == 0.075
    assert tax_unit.loc[30, "aca_take_up_rate"] == 0.0


def test_aca_source_runtime_rejects_enrollment_only_fallback() -> None:
    builder = _load_builder_module()
    specs = (
        TargetSpec(
            name="cms_aca.oep2024.state_marketplace.al.marketplace_enrollment",
            entity="household",
            measure="has_marketplace_health_coverage_at_interview",
            value=2.0,
            source="CMS Marketplace OEP",
            family="cms_aca",
            metadata={"target_role": "aca_enrollment", "state_fips": "01"},
        ),
    )

    try:
        builder._with_aca_marketplace_source_outputs(
            object(),
            specs,
            seed=42,
            simulation=object(),
        )
    except RuntimeError as exc:
        assert "requires an APTC-recipient target" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected enrollment-only ACA source refresh to fail.")


def test_aca_source_runtime_uses_bronze_targets_when_available(
    monkeypatch,
) -> None:
    builder = _load_builder_module()
    captured: dict[str, object] = {}
    tax_unit = pd.DataFrame(
        {
            "tax_unit_id": [10, 20],
            "takes_up_aca_if_eligible": [False, False],
            "selected_marketplace_plan_benchmark_ratio": [1.0, 1.0],
        }
    )

    class FakeFrame:
        entities = ("tax_unit",)
        schema = object()
        weighted_entities = ()
        strata = None

        def table(self, entity):
            assert entity == "tax_unit"
            return tax_unit

    def fake_run_source_stage(
        stage,
        *,
        tables,
        operation_handlers,
        config,
        stop_after,
    ):
        captured["stage"] = stage.stage
        captured["tables"] = tables
        captured["stop_after"] = stop_after
        return pd.DataFrame(
            {
                "tax_unit_id": [10, 20],
                "takes_up_aca_if_eligible": [True, False],
                "selected_marketplace_plan_benchmark_ratio": [0.8, 1.0],
            }
        )

    monkeypatch.setattr(builder, "_aca_source_person_table", lambda frame: object())
    monkeypatch.setattr(
        builder,
        "_aca_source_tax_unit_table",
        lambda frame, target_tables, *, simulation=None, maximum_microsim_batch_size=None: (
            pd.DataFrame({"tax_unit_id": [10, 20], "state_fips": ["06", "06"]})
        ),
    )
    monkeypatch.setattr(builder, "run_source_stage", fake_run_source_stage)
    monkeypatch.setattr(
        builder,
        "Frame",
        lambda tables, schema, weights, strata: SimpleNamespace(tables=tables),
    )

    specs = (
        TargetSpec(
            name="cms_aca.oep2024.state_marketplace.ca.aptc_recipients",
            entity="household",
            measure="takes_up_aca_if_eligible",
            value=1.0,
            source="CMS Marketplace OEP",
            family="cms_aca",
            metadata={
                "target_role": "aca_ptc_recipients",
                "state_fips": "06",
                "ledger_geography_level": "state",
            },
        ),
        TargetSpec(
            name="cms_aca.oep2024.state_metal.ca.bronze_aptc_consumers",
            entity="household",
            measure="selected_marketplace_plan_benchmark_ratio",
            value=1.0,
            source="CMS Marketplace OEP",
            family="cms_aca",
            metadata={
                "target_role": "aca_bronze_aptc_consumers",
                "state_fips": "06",
                "ledger_geography_level": "state",
            },
        ),
    )

    builder._with_aca_marketplace_source_outputs(
        FakeFrame(),
        specs,
        seed=42,
        simulation=object(),
    )

    assert captured["stage"] == builder.US_ACA_MARKETPLACE_STAGE
    assert captured["stop_after"] is None
    target_tables = captured["tables"]
    assert set(target_tables) >= {
        builder.US_ACA_APTC_TARGET_TABLE,
        "cms_aca_bronze_aptc_consumers_by_state",
    }
    bronze_table = target_tables["cms_aca_bronze_aptc_consumers_by_state"]
    assert bronze_table.to_dict("records") == [
        {
            "state_fips": "06",
            "target": 1.0,
            "source_record_id": (
                "cms_aca.oep2024.state_metal.ca.bronze_aptc_consumers"
            ),
        }
    ]


def test_aca_source_target_tables_ignore_congressional_district_targets() -> None:
    builder = _load_builder_module()

    specs = (
        TargetSpec(
            name="irs_soi.ty2022.historic_table_2.state_broad.ca.all."
            "premium_tax_credit_amount",
            entity="household",
            measure="assigned_aca_ptc",
            value=100.0,
            source="SOI",
            family="irs_soi",
            metadata={
                "target_role": "aca_spending",
                "state_fips": "06",
                "ledger_geography_level": "state",
            },
        ),
        TargetSpec(
            name="irs_soi.ty2023.congressional_district_2022.all_returns."
            "ca_01.premium_tax_credit_amount",
            entity="household",
            measure="assigned_aca_ptc",
            value=75.0,
            source="SOI",
            family="irs_soi",
            metadata={
                "target_role": "aca_spending",
                "state_fips": "06",
                "ledger_geography_level": "congressional_district",
                "congressional_district_geoid": "0601",
            },
        ),
        TargetSpec(
            name="irs_soi.ty2023.congressional_district_2022.all_returns."
            "ca_total.premium_tax_credit_amount",
            entity="household",
            measure="assigned_aca_ptc",
            value=125.0,
            source="SOI",
            family="irs_soi",
            metadata={
                "target_role": "aca_spending",
                "state_fips": "06",
                "ledger_geography_level": "state",
                "ledger_layout_groupby_dimension": ("irs_soi.congressional_district"),
                "ledger_layout_groupby_value_id": "ca_total",
            },
        ),
    )

    tables = builder._aca_source_target_tables(specs)

    amount_table = tables["irs_soi_premium_tax_credit_amount_by_state"]
    assert amount_table.to_dict("records") == [
        {
            "state_fips": "06",
            "target": 100.0,
            "source_record_id": (
                "irs_soi.ty2022.historic_table_2.state_broad.ca.all."
                "premium_tax_credit_amount"
            ),
        }
    ]


def test_jct_materialization_collapses_reform_tax_units_and_clears_caches(
    monkeypatch,
    tmp_path,
) -> None:
    builder = _load_builder_module()
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.asarray([1, 2, 3], dtype="int64"),
                    "person_household_id": np.asarray([1, 1, 2], dtype="int64"),
                    "person_tax_unit_id": np.asarray([10, 20, 30], dtype="int64"),
                    "person_spm_unit_id": np.asarray([100, 100, 200], dtype="int64"),
                    "person_family_id": np.asarray([1000, 1000, 2000], dtype="int64"),
                    "person_marital_unit_id": np.asarray(
                        [10000, 20000, 30000], dtype="int64"
                    ),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.asarray([1, 2], dtype="int64"),
                    "state_fips": np.asarray([6, 36], dtype="int64"),
                }
            ),
            "tax_unit": pd.DataFrame(
                {"tax_unit_id": np.asarray([10, 20, 30], dtype="int64")}
            ),
            "spm_unit": pd.DataFrame(
                {"spm_unit_id": np.asarray([100, 200], dtype="int64")}
            ),
            "family": pd.DataFrame(
                {"family_id": np.asarray([1000, 2000], dtype="int64")}
            ),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": np.asarray([10000, 20000, 30000], dtype="int64")}
            ),
        },
        builder.US_SCHEMA,
        {
            "household": builder.Weights(
                values=np.asarray([1.0, 1.0]), kind=WeightKind.DESIGN
            )
        },
    )
    target = TargetSpec(
        name=f"jct.mock_tax_expenditure@{builder.PERIOD}",
        entity="household",
        measure="jct_mock_tax_expenditure",
        value=-45.0,
        source="Mock JCT",
        family="jct",
        signed=True,
    )
    reform_spec = SimpleNamespace(
        neutralized_variable="mock_credit", measure="jct_mock_tax_expenditure"
    )
    datasets = []
    simulations = []
    reform_systems = []
    formula_owned_assertions: list[int] = []

    class FakeVariable:
        entity = SimpleNamespace(key="tax_unit")

    class FakeSystem:
        variables = {
            "state_income_tax": FakeVariable(),
            "mock_credit": FakeVariable(),
        }

        def __init__(self, reform=None):
            self.reform = reform
            reform_systems.append(self)

    class FakeMicrosimulation:
        default_tax_benefit_system = FakeSystem

        def __init__(self, *, dataset, reform=None, tax_benefit_system=None):
            self.dataset = dataset
            self.reform = reform
            self.tax_benefit_system = tax_benefit_system
            self.cache_invalidations = 0
            simulations.append(self)

        def calculate(self, variable, *, period, **kwargs):
            assert period == builder.PERIOD
            tax_unit_ids = (
                self.dataset["frame"].table("tax_unit")["tax_unit_id"].to_numpy()
            )
            if self.reform is not None:
                assert variable == "income_tax"
                assert kwargs == {}
                reform_income_tax_by_id = {10: 90.0, 20: 25.0, 30: 40.0}
                return np.asarray(
                    [reform_income_tax_by_id[id_] for id_ in tax_unit_ids]
                )
            arrays_by_id = {
                "income_tax": {10: 100.0, 20: 30.0, 30: 70.0},
                "taxable_income": {10: 1000.0, 20: 2000.0, 30: 3000.0},
                "adjusted_gross_income": {10: 1100.0, 20: 2100.0, 30: 3100.0},
                "filing_status": {10: "SINGLE", 20: "SINGLE", 30: "SINGLE"},
                "state_income_tax": {10: 5.0, 20: 6.0, 30: 7.0},
            }
            assert kwargs == {}
            return np.asarray([arrays_by_id[variable][id_] for id_ in tax_unit_ids])

        def _invalidate_all_caches(self):
            self.cache_invalidations += 1

    def fake_dataset_from_frame(
        frame_arg,
        *,
        zero_variables=(),
        system=None,
        assert_no_formula_owned_columns=True,
    ):
        datasets.append(
            (
                frame_arg,
                tuple(zero_variables),
                system,
                assert_no_formula_owned_columns,
            )
        )
        return {"frame": frame_arg, "zero_variables": tuple(zero_variables)}

    def fake_make_zero_variable_reform(system, variable_name):
        assert isinstance(system, FakeSystem)
        assert variable_name == "mock_credit"
        return object()

    def fake_assert_no_formula_owned_columns(frame_arg):
        formula_owned_assertions.append(frame_arg.n("household"))

    monkeypatch.setitem(
        sys.modules,
        "policyengine_us",
        SimpleNamespace(
            CountryTaxBenefitSystem=FakeSystem,
            Microsimulation=FakeMicrosimulation,
        ),
    )
    monkeypatch.setattr(
        builder,
        "_assert_no_formula_owned_columns",
        fake_assert_no_formula_owned_columns,
    )
    monkeypatch.setattr(builder, "_dataset_from_frame", fake_dataset_from_frame)
    monkeypatch.setattr(
        builder, "_make_zero_variable_reform", fake_make_zero_variable_reform
    )
    monkeypatch.setattr(builder, "US_JCT_TAX_EXPENDITURE_REFORMS", (reform_spec,))
    monkeypatch.setattr(builder, "SOI_VARIABLE_MAP", {})

    cache_context = {
        "base_dataset_sha256": "test-base-sha",
        "build_commit": "test-commit",
        "policyengine_us_version": "test-policyengine-us",
        "seed": 0,
        "target_period": builder.PERIOD,
        "target_registry_version": "test-target-registry",
        # Required declaration (PR #557): the reform-vector projection
        # fail-closes without it — see the dedicated rejection test.
        "target_frame_materializer_identity_sha256": "test-materializer-digest",
    }
    with pytest.raises(ValueError, match="target_frame_materializer_identity_sha256"):
        builder._reform_vector_cache_context(
            {
                k: v
                for k, v in cache_context.items()
                if k != "target_frame_materializer_identity_sha256"
            }
        )
    target_frame, registry, dropped = builder._materialize_target_frame(
        frame,
        (target,),
        maximum_microsim_batch_size=1,
        target_materialization_cache_dir=tmp_path,
        target_materialization_cache_context=cache_context,
    )

    household = target_frame.table("household")
    assert np.array_equal(household["income_tax"], np.asarray([130.0, 70.0]))
    assert np.array_equal(
        household["jct_mock_tax_expenditure"], np.asarray([-15.0, -30.0])
    )
    assert len(registry) == 1
    assert dropped["dropped_target_names"] == []
    assert dropped["target_materialization_cache"]["hits"] == 0
    assert dropped["target_materialization_cache"]["misses"] == 1
    assert dropped["target_materialization_cache"]["writes"] == 1
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert len(list(tmp_path.glob("*.npy"))) == 1
    assert [dataset[1] for dataset in datasets] == [
        (),
        ("mock_credit",),
        ("mock_credit",),
    ]
    assert [dataset[0].n("household") for dataset in datasets] == [2, 1, 1]
    assert [dataset[3] for dataset in datasets] == [False, False, False]
    assert formula_owned_assertions == [2, 2]
    assert len(simulations) == 3
    # populace#456: one reform system per target family (the metadata system
    # plus one family system), shared by every batch simulation of the family
    # — not one engine build per batch.
    assert len(reform_systems) == 2
    assert [system.reform is not None for system in reform_systems] == [False, True]
    assert [simulation.tax_benefit_system for simulation in simulations] == [
        None,
        reform_systems[1],
        reform_systems[1],
    ]
    # Each simulation was released (dataset reference severed), not merely
    # cache-invalidated.
    assert [simulation.dataset for simulation in simulations] == [None, None, None]
    assert [simulation.cache_invalidations for simulation in simulations] == [0, 0, 0]

    target_frame_again, registry_again, dropped_again = (
        builder._materialize_target_frame(
            frame,
            (target,),
            maximum_microsim_batch_size=1,
            target_materialization_cache_dir=tmp_path,
            target_materialization_cache_context=cache_context,
        )
    )

    household_again = target_frame_again.table("household")
    assert np.array_equal(
        household_again["jct_mock_tax_expenditure"], np.asarray([-15.0, -30.0])
    )
    assert len(registry_again) == 1
    assert dropped_again["dropped_target_names"] == []
    assert dropped_again["target_materialization_cache"]["hits"] == 1
    assert dropped_again["target_materialization_cache"]["misses"] == 0
    assert dropped_again["target_materialization_cache"]["writes"] == 0
    assert [dataset[1] for dataset in datasets] == [
        (),
        ("mock_credit",),
        ("mock_credit",),
        (),
    ]
    assert [dataset[0].n("household") for dataset in datasets] == [2, 1, 1, 2]
    assert [dataset[3] for dataset in datasets] == [False, False, False, False]
    assert formula_owned_assertions == [2, 2, 2]
    assert len(simulations) == 4
    # The cache hit skips reform materialization entirely, so the second run
    # adds only its metadata system — no new family system is built.
    assert len(reform_systems) == 3
    assert [system.reform is not None for system in reform_systems] == [
        False,
        True,
        False,
    ]
    assert [simulation.dataset for simulation in simulations] == [
        None,
        None,
        None,
        None,
    ]
    assert [simulation.cache_invalidations for simulation in simulations] == [
        0,
        0,
        0,
        0,
    ]


def test_target_materialization_cache_rejects_value_hash_mismatch(tmp_path) -> None:
    builder = _load_builder_module()
    identity = {
        "schema_version": builder.TARGET_MATERIALIZATION_CACHE_SCHEMA_VERSION,
        "kind": "jct_reform_income_tax_by_household",
        "reform_measure": "mock_credit",
    }
    _, values_path = builder._write_reform_income_tax_cache(
        tmp_path,
        identity,
        np.asarray([1.0, 2.0]),
    )
    with values_path.open("wb") as stream:
        np.save(stream, np.asarray([3.0, 4.0]), allow_pickle=False)

    with pytest.raises(RuntimeError, match="values hash mismatch"):
        builder._read_reform_income_tax_cache(
            tmp_path,
            identity,
            n_households=2,
        )


def test_target_materialization_cache_rejects_pre_557_identities(tmp_path) -> None:
    """Schema-2 and pre-preservation materializer vectors cannot serve."""

    builder = _load_builder_module()
    assert builder.TARGET_MATERIALIZATION_CACHE_SCHEMA_VERSION == 3
    reform_spec = SimpleNamespace(
        measure="jct_mock_tax_expenditure",
        neutralized_variable="mock_credit",
    )
    current_context = {
        "target_frame_materializer_identity_sha256": "version-10-preserved-surface",
    }
    current_identity = builder._target_materialization_cache_identity(
        context=current_context,
        reform_spec=reform_spec,
        n_households=2,
    )

    for stale_schema in (2, 1):
        stale_identity = {
            **current_identity,
            "schema_version": stale_schema,
        }
        builder._write_reform_income_tax_cache(
            tmp_path,
            stale_identity,
            np.asarray([1.0, 2.0]),
        )
        assert (
            builder._read_reform_income_tax_cache(
                tmp_path,
                current_identity,
                n_households=2,
            )
            is None
        )

    pre_557_identity = builder._target_materialization_cache_identity(
        context={
            "target_frame_materializer_identity_sha256": (
                "version-9-release-refitted-surface"
            ),
        },
        reform_spec=reform_spec,
        n_households=2,
    )
    builder._write_reform_income_tax_cache(
        tmp_path,
        pre_557_identity,
        np.asarray([3.0, 4.0]),
    )
    assert (
        builder._read_reform_income_tax_cache(
            tmp_path,
            current_identity,
            n_households=2,
        )
        is None
    )


def test_soi_filtered_targets_keep_mortgage_and_broad_interest_distinct(
    monkeypatch,
) -> None:
    from populace.build.us_runtime import split_us_puf_e19200_by_agi_band

    builder = _load_builder_module()
    e19200_total = np.asarray([100.0, 200.0, 300.0, 400.0])
    source_year_agi = np.asarray([-5_000.0, 20_000.0, 100_000.0, 10_000_000.0])
    mortgage_interest, non_mortgage_interest = split_us_puf_e19200_by_agi_band(
        e19200_total, source_year_agi
    )
    assert np.all(non_mortgage_interest > 0)
    broader_interest = mortgage_interest + non_mortgage_interest
    np.testing.assert_array_equal(broader_interest, e19200_total)

    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.asarray([1, 2, 3, 4], dtype="int64"),
                    "person_household_id": np.asarray([1, 1, 2, 2], dtype="int64"),
                    "person_tax_unit_id": np.asarray([10, 20, 30, 40], dtype="int64"),
                    "person_spm_unit_id": np.asarray(
                        [100, 100, 200, 200], dtype="int64"
                    ),
                    "person_family_id": np.asarray(
                        [1000, 1000, 2000, 2000], dtype="int64"
                    ),
                    "person_marital_unit_id": np.asarray(
                        [10000, 20000, 30000, 40000], dtype="int64"
                    ),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.asarray([1, 2], dtype="int64"),
                    "state_fips": np.asarray([6, 6], dtype="int64"),
                    "congressional_district_geoid": np.asarray(
                        [601, 602], dtype="int64"
                    ),
                }
            ),
            "tax_unit": pd.DataFrame(
                {"tax_unit_id": np.asarray([10, 20, 30, 40], dtype="int64")}
            ),
            "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([100, 200])}),
            "family": pd.DataFrame({"family_id": np.asarray([1000, 2000])}),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": np.asarray([10000, 20000, 30000, 40000])}
            ),
        },
        builder.US_SCHEMA,
        {
            "household": builder.Weights(
                values=np.asarray([1.0, 1.0]), kind=WeightKind.DESIGN
            )
        },
    )

    def eitc_spec(name, measure, child_filter, *, count=False, variable="eitc"):
        metadata = {
            "variable": variable,
            "agi_lower_bound": "-inf",
            "agi_upper_bound": "inf",
            "filing_status": "All",
            "source_measure_id": "eitc_returns" if count else "eitc_total",
            "ledger_filter_eitc_child_count": child_filter,
            "measure_mode": "indicator_sum" if count else "sum",
        }
        return TargetSpec(
            name=name,
            entity="household",
            measure=measure,
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata=metadata,
        )

    targets = (
        eitc_spec("no_child_amount", "no_child_amount", "0"),
        eitc_spec("two_child_amount", "two_child_amount", "2"),
        eitc_spec("three_plus_amount", "three_plus_amount", "3plus"),
        eitc_spec("two_child_returns", "two_child_returns", "2", count=True),
        eitc_spec(
            "three_plus_return_count",
            "three_plus_return_count",
            "three_or_more_qualifying_children",
            count=True,
        ),
        TargetSpec(
            name="eitc_return_agi",
            entity="household",
            measure="eitc_return_agi",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "adjusted_gross_income",
                "source_variable": "adjusted_gross_income",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "adjusted_gross_income",
                "ledger_domain": (
                    "individual_income_tax_returns_with_earned_income_credit"
                ),
            },
        ),
        TargetSpec(
            name="eitc_return_count",
            entity="household",
            measure="eitc_return_count",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "count",
                "source_variable": "count",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "return_count",
                "ledger_domain": (
                    "individual_income_tax_returns_with_earned_income_credit"
                ),
                "measure_mode": "indicator_sum",
            },
        ),
        TargetSpec(
            name="form_w2_social_security_tips",
            entity="household",
            measure="form_w2_social_security_tips",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "tip_income",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "return_count",
                "measure_mode": "indicator_sum",
            },
        ),
        TargetSpec(
            name="cd_0601_agi",
            entity="household",
            measure="cd_0601_agi",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "adjusted_gross_income",
                "source_variable": "adjusted_gross_income",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "adjusted_gross_income",
                "congressional_district_geoid": "0601",
            },
        ),
        TargetSpec(
            name="cd_0601_tax_filer_individual_count",
            entity="household",
            measure="cd_0601_tax_filer_individual_count",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "tax_filer_individual_count",
                "source_variable": "tax_filer_individual_count",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "tax_filer_individual_count",
                "congressional_district_geoid": "0601",
            },
        ),
        TargetSpec(
            name="medical_dental_expense_amount",
            entity="household",
            measure="medical_dental_expense_amount",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "medical_expense_deduction",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "medical_dental_expense_amount",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="medical_dental_expense_returns",
            entity="household",
            measure="medical_dental_expense_returns",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "medical_expense_deduction",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "medical_dental_expense_returns",
                "measure_mode": "indicator_sum",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="real_estate_taxes_amount",
            entity="household",
            measure="real_estate_taxes_amount",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "real_estate_taxes",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "real_estate_taxes_amount",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="real_estate_taxes_claims",
            entity="household",
            measure="real_estate_taxes_claims",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "real_estate_taxes",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "real_estate_taxes_claims",
                "measure_mode": "indicator_sum",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="limited_state_local_taxes_amount",
            entity="household",
            measure="limited_state_local_taxes_amount",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "salt_deduction",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "limited_state_local_taxes_amount",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="limited_state_local_taxes_returns",
            entity="household",
            measure="limited_state_local_taxes_returns",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "salt_deduction",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "limited_state_local_taxes_returns",
                "measure_mode": "indicator_sum",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="itemized_deductions_amount",
            entity="household",
            measure="itemized_deductions_amount",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "itemized_taxable_income_deductions",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "itemized_deductions_amount",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="total_itemized_deductions_amount",
            entity="household",
            measure="total_itemized_deductions_amount",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "itemized_taxable_income_deductions",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "total_itemized_deductions_amount",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="charitable_amount",
            entity="household",
            measure="charitable_amount",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "charitable_deduction",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "charitable_amount",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="interest_paid_deduction_amount",
            entity="household",
            measure="interest_paid_deduction_amount",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "interest_deduction",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "interest_paid_deduction_amount",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="home_mortgage_interest_amount",
            entity="household",
            measure="home_mortgage_interest_amount",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "deductible_mortgage_interest",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "home_mortgage_interest_amount",
                "itemized_only": "true",
            },
        ),
    )

    class FakeVariable:
        entity = SimpleNamespace(key="tax_unit")

    class FakeSystem:
        variables = {
            name: FakeVariable()
            for name in (
                "income_tax",
                "taxable_income",
                "adjusted_gross_income",
                "filing_status",
                "state_income_tax",
                "eitc",
                "eitc_child_count",
                "itemized_taxable_income_deductions",
                "charitable_deduction",
                "deductible_mortgage_interest",
                "interest_deduction",
                "medical_expense_deduction",
                "real_estate_taxes",
                "salt_deduction",
                "tip_income",
                "tax_unit_size",
                "tax_unit_itemizes",
            )
        }

        def __init__(self, reform=None):
            self.reform = reform

    class FakeMicrosimulation:
        default_tax_benefit_system = FakeSystem

        def __init__(self, *, dataset, reform=None, tax_benefit_system=None):
            self.dataset = dataset
            self.reform = reform
            self.tax_benefit_system = tax_benefit_system

        def calculate(self, variable, *, period, **kwargs):
            assert period == builder.PERIOD
            assert kwargs == {}
            arrays = {
                "income_tax": np.asarray([0.0, 0.0, 0.0, 0.0]),
                "taxable_income": np.asarray([0.0, 0.0, 0.0, 0.0]),
                "adjusted_gross_income": np.asarray(
                    [10_000.0, 20_000.0, 30_000.0, 40_000.0]
                ),
                "filing_status": np.asarray(["SINGLE", "SINGLE", "SINGLE", "SINGLE"]),
                "state_income_tax": np.asarray([0.0, 0.0, 0.0, 0.0]),
                "eitc": np.asarray([100.0, 200.0, 300.0, 0.0]),
                "eitc_child_count": np.asarray([0.0, 2.0, 3.0, 3.0]),
                "itemized_taxable_income_deductions": np.asarray(
                    [1_000.0, 2_000.0, 3_000.0, 4_000.0]
                ),
                "charitable_deduction": np.asarray([10.0, 20.0, 30.0, 40.0]),
                "deductible_mortgage_interest": mortgage_interest,
                "interest_deduction": broader_interest,
                "medical_expense_deduction": np.asarray([100.0, 200.0, 300.0, 400.0]),
                "real_estate_taxes": np.asarray([5_000.0, 6_000.0, 7_000.0, 8_000.0]),
                "salt_deduction": np.asarray([500.0, 600.0, 700.0, 800.0]),
                "tip_income": np.asarray([0.0, 50.0, 0.0, 0.0]),
                "tax_unit_size": np.asarray([1.0, 2.0, 3.0, 4.0]),
                "tax_unit_itemizes": np.asarray([False, True, False, False]),
            }
            return arrays[variable]

        def _invalidate_all_caches(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "policyengine_us",
        SimpleNamespace(
            CountryTaxBenefitSystem=FakeSystem,
            Microsimulation=FakeMicrosimulation,
        ),
    )
    monkeypatch.setattr(builder, "_dataset_from_frame", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        builder,
        "SOI_VARIABLE_MAP",
        {
            "adjusted_gross_income": "adjusted_gross_income",
            "eitc": "eitc",
            "itemized_taxable_income_deductions": (
                "itemized_taxable_income_deductions"
            ),
            "charitable_deduction": "charitable_deduction",
            "deductible_mortgage_interest": "deductible_mortgage_interest",
            "interest_deduction": "interest_deduction",
            "medical_expense_deduction": "medical_expense_deduction",
            "real_estate_taxes": "real_estate_taxes",
            "salt_deduction": "salt_deduction",
            "tip_income": "tip_income",
            "tax_filer_individual_count": "tax_unit_size",
        },
    )
    monkeypatch.setattr(builder, "US_JCT_TAX_EXPENDITURE_REFORMS", ())

    target_frame, registry, compilation = builder._materialize_target_frame(
        frame, targets
    )

    household = target_frame.table("household")
    assert np.array_equal(household["no_child_amount"], np.asarray([100.0, 0.0]))
    assert np.array_equal(household["two_child_amount"], np.asarray([200.0, 0.0]))
    assert np.array_equal(household["three_plus_amount"], np.asarray([0.0, 300.0]))
    assert np.array_equal(household["two_child_returns"], np.asarray([1.0, 0.0]))
    assert np.array_equal(household["three_plus_return_count"], np.asarray([0.0, 1.0]))
    assert np.array_equal(
        household["eitc_return_agi"], np.asarray([30_000.0, 30_000.0])
    )
    assert np.array_equal(household["eitc_return_count"], np.asarray([2.0, 1.0]))
    assert np.array_equal(
        household["form_w2_social_security_tips"], np.asarray([1.0, 0.0])
    )
    assert np.array_equal(household["cd_0601_agi"], np.asarray([30_000.0, 0.0]))
    assert np.array_equal(
        household["cd_0601_tax_filer_individual_count"], np.asarray([3.0, 0.0])
    )
    assert np.array_equal(
        household["medical_dental_expense_amount"], np.asarray([200.0, 0.0])
    )
    assert np.array_equal(
        household["medical_dental_expense_returns"], np.asarray([1.0, 0.0])
    )
    assert np.array_equal(
        household["real_estate_taxes_amount"], np.asarray([6_000.0, 0.0])
    )
    assert np.array_equal(household["real_estate_taxes_claims"], np.asarray([1.0, 0.0]))
    assert np.array_equal(
        household["limited_state_local_taxes_amount"], np.asarray([600.0, 0.0])
    )
    assert np.array_equal(
        household["limited_state_local_taxes_returns"], np.asarray([1.0, 0.0])
    )
    assert np.array_equal(
        household["itemized_deductions_amount"], np.asarray([2_000.0, 0.0])
    )
    assert np.array_equal(
        household["total_itemized_deductions_amount"], np.asarray([2_000.0, 0.0])
    )
    assert np.array_equal(household["charitable_amount"], np.asarray([20.0, 0.0]))
    assert np.array_equal(
        household["interest_paid_deduction_amount"],
        np.asarray([broader_interest[1], 0.0]),
    )
    assert np.array_equal(
        household["home_mortgage_interest_amount"],
        np.asarray([mortgage_interest[1], 0.0]),
    )
    np.testing.assert_allclose(
        household["interest_paid_deduction_amount"]
        - household["home_mortgage_interest_amount"],
        np.asarray([non_mortgage_interest[1], 0.0]),
    )
    assert not np.array_equal(
        household["home_mortgage_interest_amount"],
        household["interest_paid_deduction_amount"],
    )
    assert len(registry) == 21
    assert compilation["dropped_target_names"] == []


def test_soi_ctc_targets_materialize_nonrefundable_credit(
    monkeypatch,
) -> None:
    builder = _load_builder_module()
    assert builder.SOI_VARIABLE_MAP["ctc"] == "ctc"
    assert builder.SOI_VARIABLE_MAP["refundable_ctc"] == "refundable_ctc"
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.asarray([1, 2, 3], dtype="int64"),
                    "person_household_id": np.asarray([1, 1, 2], dtype="int64"),
                    "person_tax_unit_id": np.asarray([10, 20, 30], dtype="int64"),
                    "person_spm_unit_id": np.asarray([100, 100, 200], dtype="int64"),
                    "person_family_id": np.asarray([1000, 1000, 2000], dtype="int64"),
                    "person_marital_unit_id": np.asarray(
                        [10000, 20000, 30000], dtype="int64"
                    ),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.asarray([1, 2], dtype="int64"),
                    "state_fips": np.asarray([6, 6], dtype="int64"),
                }
            ),
            "tax_unit": pd.DataFrame(
                {"tax_unit_id": np.asarray([10, 20, 30], dtype="int64")}
            ),
            "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([100, 200])}),
            "family": pd.DataFrame({"family_id": np.asarray([1000, 2000])}),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": np.asarray([10000, 20000, 30000])}
            ),
        },
        builder.US_SCHEMA,
        {
            "household": builder.Weights(
                values=np.asarray([1.0, 1.0]), kind=WeightKind.DESIGN
            )
        },
    )

    def soi_spec(name, measure, source_name, source_measure_id, *, count=False):
        metadata = {
            "variable": source_name,
            "agi_lower_bound": "-inf",
            "agi_upper_bound": "inf",
            "filing_status": "All",
            "source_measure_id": source_measure_id,
            "measure_mode": "indicator_sum" if count else "sum",
        }
        return TargetSpec(
            name=name,
            entity="household",
            measure=measure,
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata=metadata,
        )

    targets = (
        soi_spec("ctc_amount", "ctc_amount", "ctc", "ctc_amount"),
        soi_spec("ctc_claims", "ctc_claims", "ctc", "ctc_claims", count=True),
        soi_spec(
            "actc_amount",
            "actc_amount",
            "refundable_ctc",
            "actc_amount",
        ),
        soi_spec(
            "actc_claims",
            "actc_claims",
            "refundable_ctc",
            "actc_claims",
            count=True,
        ),
    )

    class FakeVariable:
        entity = SimpleNamespace(key="tax_unit")

    class FakeSystem:
        variables = {
            name: FakeVariable()
            for name in (
                "income_tax",
                "taxable_income",
                "adjusted_gross_income",
                "filing_status",
                "state_income_tax",
                "ctc",
                "ctc_limiting_tax_liability",
                "refundable_ctc",
            )
        }

        def __init__(self, reform=None):
            self.reform = reform

    class FakeMicrosimulation:
        default_tax_benefit_system = FakeSystem

        def __init__(self, *, dataset, reform=None, tax_benefit_system=None):
            self.dataset = dataset
            self.reform = reform
            self.tax_benefit_system = tax_benefit_system

        def calculate(self, variable, *, period, **kwargs):
            assert period == builder.PERIOD
            assert kwargs == {}
            arrays = {
                "income_tax": np.asarray([0.0, 0.0, 0.0]),
                "taxable_income": np.asarray([0.0, 0.0, 0.0]),
                "adjusted_gross_income": np.asarray([10_000.0, 20_000.0, 30_000.0]),
                "filing_status": np.asarray(["SINGLE", "SINGLE", "SINGLE"]),
                "state_income_tax": np.asarray([0.0, 0.0, 0.0]),
                "ctc": np.asarray([1_000.0, 2_000.0, 3_000.0]),
                "ctc_limiting_tax_liability": np.asarray([80.0, 0.0, 20.0]),
                "refundable_ctc": np.asarray([10.0, 30.0, 0.0]),
            }
            return arrays[variable]

        def _invalidate_all_caches(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "policyengine_us",
        SimpleNamespace(
            CountryTaxBenefitSystem=FakeSystem,
            Microsimulation=FakeMicrosimulation,
        ),
    )
    monkeypatch.setattr(builder, "_dataset_from_frame", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        builder,
        "SOI_VARIABLE_MAP",
        {
            "ctc": "ctc",
            "refundable_ctc": "refundable_ctc",
        },
    )
    monkeypatch.setattr(builder, "US_JCT_TAX_EXPENDITURE_REFORMS", ())

    target_frame, registry, compilation = builder._materialize_target_frame(
        frame, targets
    )

    household = target_frame.table("household")
    assert np.array_equal(household["ctc_amount"], np.asarray([80.0, 20.0]))
    assert np.array_equal(household["ctc_claims"], np.asarray([1.0, 1.0]))
    assert np.array_equal(household["actc_amount"], np.asarray([40.0, 0.0]))
    assert np.array_equal(household["actc_claims"], np.asarray([2.0, 0.0]))
    assert len(registry) == 4
    assert compilation["dropped_target_names"] == []


def test_population_age_targets_materialize_person_age_counts(
    monkeypatch,
) -> None:
    builder = _load_builder_module()
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.asarray([1, 2, 3, 4], dtype="int64"),
                    "person_household_id": np.asarray([1, 1, 2, 2], dtype="int64"),
                    "person_tax_unit_id": np.asarray([10, 20, 30, 30], dtype="int64"),
                    "person_spm_unit_id": np.asarray(
                        [100, 100, 200, 200], dtype="int64"
                    ),
                    "person_family_id": np.asarray(
                        [1000, 1000, 2000, 2000], dtype="int64"
                    ),
                    "person_marital_unit_id": np.asarray(
                        [10000, 20000, 30000, 30000], dtype="int64"
                    ),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.asarray([1, 2], dtype="int64"),
                    "state_fips": np.asarray([6, 12], dtype="int64"),
                    "congressional_district_geoid": np.asarray(
                        ["0601", "1201"], dtype=object
                    ),
                }
            ),
            "tax_unit": pd.DataFrame(
                {"tax_unit_id": np.asarray([10, 20, 30], dtype="int64")}
            ),
            "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([100, 200])}),
            "family": pd.DataFrame({"family_id": np.asarray([1000, 2000])}),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": np.asarray([10000, 20000, 30000])}
            ),
        },
        builder.US_SCHEMA,
        {
            "household": builder.Weights(
                values=np.asarray([1.0, 1.0]), kind=WeightKind.DESIGN
            )
        },
    )

    def population_age_spec(
        name,
        lower,
        upper,
        *,
        state_fips=None,
        congressional_district_geoid=None,
    ):
        metadata = {
            "materializer": "population_age",
            "measure_mode": "indicator_sum",
            "target_role": "population_age",
            "geography_scope": (
                "congressional_district"
                if congressional_district_geoid
                else "state"
                if state_fips
                else "national"
            ),
            "age_lower_bound": str(lower),
            "age_upper_bound": str(upper),
        }
        if state_fips:
            metadata["state_fips"] = state_fips
        if congressional_district_geoid:
            metadata["congressional_district_geoid"] = congressional_district_geoid
        return TargetSpec(
            name=name,
            entity="household",
            measure=name,
            value=1.0,
            source="fixture",
            family="census_population",
            metadata=metadata,
        )

    targets = (
        population_age_spec("national_age_0_to_4", 0, 5),
        population_age_spec("ca_age_0_to_4", 0, 5, state_fips="06"),
        population_age_spec("ca_age_5_to_9", 5, 10, state_fips="06"),
        population_age_spec(
            "ca_01_age_0_to_4",
            0,
            5,
            state_fips="06",
            congressional_district_geoid="0601",
        ),
    )

    class FakeVariable:
        def __init__(self, entity):
            self.entity = SimpleNamespace(key=entity)

    class FakeSystem:
        variables = {
            "income_tax": FakeVariable("tax_unit"),
            "taxable_income": FakeVariable("tax_unit"),
            "adjusted_gross_income": FakeVariable("tax_unit"),
            "filing_status": FakeVariable("tax_unit"),
            "state_income_tax": FakeVariable("tax_unit"),
            "age": FakeVariable("person"),
        }

        def __init__(self, reform=None):
            self.reform = reform

    class FakeMicrosimulation:
        default_tax_benefit_system = FakeSystem

        def __init__(self, *, dataset, reform=None, tax_benefit_system=None):
            self.dataset = dataset
            self.reform = reform
            self.tax_benefit_system = tax_benefit_system

        def calculate(self, variable, *, period, **kwargs):
            assert period == builder.PERIOD
            assert kwargs == {}
            arrays = {
                "income_tax": np.asarray([0.0, 0.0, 0.0]),
                "taxable_income": np.asarray([0.0, 0.0, 0.0]),
                "adjusted_gross_income": np.asarray([0.0, 0.0, 0.0]),
                "filing_status": np.asarray(["SINGLE", "SINGLE", "SINGLE"]),
                "state_income_tax": np.asarray([0.0, 0.0, 0.0]),
                "age": np.asarray([2.0, 7.0, 4.0, 11.0]),
            }
            return arrays[variable]

        def _invalidate_all_caches(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "policyengine_us",
        SimpleNamespace(
            CountryTaxBenefitSystem=FakeSystem,
            Microsimulation=FakeMicrosimulation,
        ),
    )
    monkeypatch.setattr(builder, "_dataset_from_frame", lambda *args, **kwargs: {})
    monkeypatch.setattr(builder, "SOI_VARIABLE_MAP", {})
    monkeypatch.setattr(builder, "US_JCT_TAX_EXPENDITURE_REFORMS", ())

    target_frame, registry, compilation = builder._materialize_target_frame(
        frame, targets
    )

    household = target_frame.table("household")
    assert np.array_equal(household["national_age_0_to_4"], np.asarray([1.0, 1.0]))
    assert np.array_equal(household["ca_age_0_to_4"], np.asarray([1.0, 0.0]))
    assert np.array_equal(household["ca_age_5_to_9"], np.asarray([1.0, 0.0]))
    assert np.array_equal(household["ca_01_age_0_to_4"], np.asarray([1.0, 0.0]))
    assert len(registry) == 4
    assert compilation["dropped_target_names"] == []


def test_unknown_ledger_filter_metadata_fails_closed() -> None:
    builder = _load_builder_module()
    target = TargetSpec(
        name="unknown_filter_target",
        entity="household",
        measure="income_tax",
        value=1.0,
        source="fixture",
        family="irs_soi",
        metadata={"ledger_filter_unmodeled_axis": "example"},
    )

    try:
        builder._assert_supported_ledger_filter_metadata((target,))
    except RuntimeError as exc:
        assert "ledger_filter_unmodeled_axis" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected unknown Ledger filter metadata to fail closed.")


def test_build_manifests_emits_policyengine_certifiable_release_manifest(
    monkeypatch, tmp_path
) -> None:
    builder = _load_builder_module()
    release_id = "populace-us-2024-abcdef1-20260615"
    release_dir = tmp_path / "release" / release_id
    release_dir.mkdir(parents=True)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / builder.DATASET_FILENAME).write_bytes(b"h5")
    (artifact_root / builder.CALIBRATION_FILENAME).write_bytes(b"npz")
    (release_dir / "calibration_diagnostics.json").write_text("{}")
    (release_dir / "us_source_coverage.json").write_text("{}")
    ssi_diagnostics_path = release_dir / "us_ssi_take_up.json"
    ssi_diagnostics_path.write_text('{"variable":"takes_up_ssi_if_eligible"}')

    monkeypatch.setattr(
        builder,
        "_runtime_versions",
        lambda: {
            "python": "3.14.0",
            "populace-data": "0.1.0",
            "policyengine-core": "3.26.11",
            "policyengine-us": "1.729.0",
        },
    )
    monkeypatch.setattr(
        builder,
        "_git_output",
        lambda *args: "a" * 40 if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        builder,
        "diagnostics_payload",
        lambda result, target_registry: {
            "initial_loss": 2.0,
            "final_loss": 1.0,
            "fraction_within_10pct": 1.0,
            "target_surface": {"sha256": "b" * 64, "n_targets": 1},
        },
    )

    result = SimpleNamespace(
        skipped=(),
        diagnostics=(
            SimpleNamespace(
                name=f"nation/cbo/individual_income_tax@{builder.PERIOD}",
                target=1.0,
                initial_estimate=1.0,
                final_estimate=1.0,
            ),
        ),
        initial_loss=2.0,
        final_loss=1.0,
    )

    class FakeRegistry:
        version = "registry-sha"
        specs = ()

        def __len__(self):
            return 1

    registry = FakeRegistry()

    builder._build_manifests(
        release_id=release_id,
        release_dir=release_dir,
        artifact_root=artifact_root,
        result=result,
        registry=registry,
        dropped={"dropped_target_names": []},
        target_profile_gate=builder.GateResult(
            name="target_profile_coverage",
            passed=True,
            details={"requirements_checked": 1},
        ),
        base_population_gate=builder.GateResult(
            name="base_population_scale",
            passed=True,
            details={
                "population": 334_200_000.0,
                "benchmark": 334_200_000.0,
                "relative_error": 0.0,
                "mass_repair": {
                    "method": "rescale_household_weights_to_census_person_population",
                    "applied": True,
                    "factor": 5.87,
                },
            },
        ),
        health_input_gate=builder.GateResult(
            name="health_input_signal",
            passed=True,
            details={
                "unique_counts": {
                    "takes_up_aca_if_eligible": 2,
                    "selected_marketplace_plan_benchmark_ratio": 3,
                }
            },
        ),
        timing={
            "target_compilation_seconds": 3.0,
            "calibration_seconds": 4.0,
            "total_build_seconds": 7.0,
        },
        default_dataset={
            "method": "l0_refit",
            "sparse": True,
            "n_candidate_households": 337_704,
            "n_selected_households": 57_240,
            "n_exported_households": 57_240,
            "l0_lambda_share": 0.8,
        },
    )

    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    build_manifest = json.loads((release_dir / "build_manifest.json").read_text())
    assert build_manifest["gates"]["target_profile_coverage"]["passed"]
    assert (
        build_manifest["gates"]["target_profile_coverage"]["details"][
            "requirements_checked"
        ]
        == 1
    )
    assert build_manifest["gates"]["health_input_signal"]["passed"]
    assert build_manifest["gates"]["health_input_signal"]["details"][
        "unique_counts"
    ] == {
        "takes_up_aca_if_eligible": 2,
        "selected_marketplace_plan_benchmark_ratio": 3,
    }
    assert build_manifest["gates"]["base_population_scale"]["passed"]
    assert (
        build_manifest["gates"]["base_population_scale"]["details"]["relative_error"]
        == 0.0
    )
    assert (
        build_manifest["gates"]["base_population_scale"]["details"]["mass_repair"][
            "method"
        ]
        == "rescale_household_weights_to_census_person_population"
    )
    assert manifest["data_package"] == {"name": "populace-data", "version": "0.1.0"}
    assert manifest["default_datasets"] == {"national": "populace_us_2024"}
    assert manifest["build"]["built_with_model_package"] == {
        "name": "policyengine-us",
        "version": "1.729.0",
    }
    assert build_manifest["timing"] == {
        "target_compilation_seconds": 3.0,
        "calibration_seconds": 4.0,
        "total_build_seconds": 7.0,
    }
    assert build_manifest["dataset"]["default"] == {
        "method": "l0_refit",
        "sparse": True,
        "n_candidate_households": 337_704,
        "n_selected_households": 57_240,
        "n_exported_households": 57_240,
        "l0_lambda_share": 0.8,
    }
    assert "area_artifacts" not in build_manifest
    assert manifest["build"]["timing"] == {
        "target_compilation_seconds": 3.0,
        "calibration_seconds": 4.0,
        "total_build_seconds": 7.0,
    }
    assert manifest["build"]["default_dataset"] == build_manifest["dataset"]["default"]
    assert (
        manifest["build"]["base_population_scale"]["details"]["mass_repair"]["factor"]
        == 5.87
    )
    assert manifest["compatible_core_packages"] == [
        {"name": "policyengine-core", "specifier": "==3.26.11"}
    ]
    assert manifest["compatible_model_packages"] == [
        {"name": "policyengine-us", "specifier": "==1.729.0"}
    ]
    assert not any(
        key.startswith(("states/", "districts/")) for key in manifest["artifacts"]
    )
    assert manifest["artifacts"]["us_ssi_take_up"] == {
        "kind": "diagnostics",
        "path": "us_ssi_take_up.json",
        "repo_id": builder.REPO_ID,
        "revision": release_id,
        "sha256": builder._sha256(ssi_diagnostics_path),
    }
    for artifact in manifest["artifacts"].values():
        assert artifact["repo_id"] == builder.REPO_ID
        assert artifact["revision"] == release_id
        assert artifact["kind"]
        assert artifact["sha256"]


def _minimal_manifest_kwargs(builder, release_id, release_dir, artifact_root):
    result = SimpleNamespace(
        skipped=(),
        diagnostics=(
            SimpleNamespace(
                name=f"nation/cbo/individual_income_tax@{builder.PERIOD}",
                target=1.0,
                initial_estimate=1.0,
                final_estimate=1.0,
            ),
        ),
        initial_loss=2.0,
        final_loss=1.0,
    )

    class FakeRegistry:
        version = "registry-sha"
        specs = ()

        def __len__(self):
            return 1

    return dict(
        release_id=release_id,
        release_dir=release_dir,
        artifact_root=artifact_root,
        result=result,
        registry=FakeRegistry(),
        dropped={"dropped_target_names": []},
        target_profile_gate=builder.GateResult(
            name="target_profile_coverage",
            passed=True,
            details={"requirements_checked": 1},
        ),
        default_dataset={"method": "dense_no_l0", "sparse": False},
    )


def test_build_manifests_records_selection_source_provenance(
    monkeypatch, tmp_path
) -> None:
    # A frozen-support build records its selection provenance in both manifests
    # so the informed-L0 step is reproducible from main (populace#328).
    builder = _load_builder_module()
    release_id = "populace-us-2024-sel-20260706"
    release_dir = tmp_path / "release" / release_id
    release_dir.mkdir(parents=True)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / builder.DATASET_FILENAME).write_bytes(b"h5")
    (artifact_root / builder.CALIBRATION_FILENAME).write_bytes(b"npz")
    (release_dir / "calibration_diagnostics.json").write_text("{}")
    (release_dir / "us_source_coverage.json").write_text("{}")
    (release_dir / "us_ssi_take_up.json").write_text("{}")
    monkeypatch.setattr(
        builder,
        "_runtime_versions",
        lambda: {
            "python": "3.14.0",
            "populace-data": "0.1.0",
            "policyengine-core": "3.26.11",
            "policyengine-us": "1.752.2",
        },
    )
    monkeypatch.setattr(
        builder,
        "_git_output",
        lambda *args: "a" * 40 if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        builder,
        "diagnostics_payload",
        lambda result, target_registry: {
            "initial_loss": 2.0,
            "final_loss": 1.0,
            "fraction_within_10pct": 1.0,
            "target_surface": {"sha256": "b" * 64, "n_targets": 1},
        },
    )

    selection_source = {
        "mode": "frozen_support",
        "join_key": [
            "source_year",
            "source_household_id",
            "household_support_channel",
            "household_support_clone_index",
        ],
        "source": {
            "kind": "h5",
            "path": "certified.h5",
            "sha256": "c" * 64,
        },
        "n_source": 57_240,
        "n_base_candidates": 337_704,
        "n_selected": 57_240,
        "n_unmapped": 0,
        "n_ambiguous": 0,
    }

    builder._build_manifests(
        selection_source=selection_source,
        **_minimal_manifest_kwargs(builder, release_id, release_dir, artifact_root),
    )

    build_manifest = json.loads((release_dir / "build_manifest.json").read_text())
    release_manifest = json.loads((release_dir / "release_manifest.json").read_text())
    assert build_manifest["calibration"]["selection_source"] == selection_source
    assert release_manifest["build"]["selection_source"] == selection_source
    assert build_manifest["calibration"]["selection_source"]["n_selected"] == 57_240
    assert build_manifest["calibration"]["selection_source"]["n_unmapped"] == 0


def test_build_manifests_selection_source_absent_by_default(
    monkeypatch, tmp_path
) -> None:
    # A build with no selection source records the disabled sentinel, not None.
    builder = _load_builder_module()
    release_id = "populace-us-2024-nosel-20260706"
    release_dir = tmp_path / "release" / release_id
    release_dir.mkdir(parents=True)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / builder.DATASET_FILENAME).write_bytes(b"h5")
    (artifact_root / builder.CALIBRATION_FILENAME).write_bytes(b"npz")
    (release_dir / "calibration_diagnostics.json").write_text("{}")
    (release_dir / "us_source_coverage.json").write_text("{}")
    (release_dir / "us_ssi_take_up.json").write_text("{}")
    monkeypatch.setattr(
        builder,
        "_runtime_versions",
        lambda: {
            "python": "3.14.0",
            "populace-data": "0.1.0",
            "policyengine-core": "3.26.11",
            "policyengine-us": "1.752.2",
        },
    )
    monkeypatch.setattr(
        builder,
        "_git_output",
        lambda *args: "a" * 40 if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        builder,
        "diagnostics_payload",
        lambda result, target_registry: {
            "initial_loss": 2.0,
            "final_loss": 1.0,
            "fraction_within_10pct": 1.0,
            "target_surface": {"sha256": "b" * 64, "n_targets": 1},
        },
    )

    builder._build_manifests(
        **_minimal_manifest_kwargs(builder, release_id, release_dir, artifact_root),
    )

    build_manifest = json.loads((release_dir / "build_manifest.json").read_text())
    release_manifest = json.loads((release_dir / "release_manifest.json").read_text())
    assert build_manifest["calibration"]["selection_source"] == {"enabled": False}
    assert release_manifest["build"]["selection_source"] == {"enabled": False}


def test_build_manifests_uses_incumbent_aware_calibration_gate(
    monkeypatch, tmp_path
) -> None:
    builder = _load_builder_module()
    release_id = "populace-us-2024-abcdef1-20260615"
    release_dir = tmp_path / "release" / release_id
    release_dir.mkdir(parents=True)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / builder.DATASET_FILENAME).write_bytes(b"h5")
    (artifact_root / builder.CALIBRATION_FILENAME).write_bytes(b"npz")
    (release_dir / "calibration_diagnostics.json").write_text("{}")
    (release_dir / "us_source_coverage.json").write_text("{}")
    (release_dir / "us_ssi_take_up.json").write_text("{}")

    monkeypatch.setattr(
        builder,
        "_runtime_versions",
        lambda: {
            "python": "3.14.0",
            "populace-data": "0.1.0",
            "policyengine-core": "3.26.11",
            "policyengine-us": "1.729.0",
        },
    )
    monkeypatch.setattr(
        builder,
        "_git_output",
        lambda *args: "a" * 40 if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        builder,
        "diagnostics_payload",
        lambda result, target_registry: {
            "initial_loss": 2.0,
            "final_loss": 1.0,
            "fraction_within_10pct": 1.0,
            "target_surface": {"sha256": "b" * 64, "n_targets": 1},
        },
    )

    name = f"irs_soi.ty2022.historic_table_2.us.all.ctc_amount@{builder.PERIOD}"
    target = 82_863_353_000.0
    diagnostics = list(_passing_critical_diagnostics(builder))
    index = next(
        i for i, diagnostic in enumerate(diagnostics) if diagnostic.name == name
    )
    diagnostics[index] = SimpleNamespace(
        name=name,
        target=target,
        initial_estimate=99_315_000_000.0,
        final_estimate=99_282_300_000.0,
        relative_error=(99_282_300_000.0 - target) / target,
    )
    result = SimpleNamespace(
        skipped=(),
        diagnostics=tuple(diagnostics),
        initial_loss=2.0,
        final_loss=1.0,
    )

    class FakeRegistry:
        version = "registry-sha"
        specs = ()

        def __len__(self):
            return 1

    builder._build_manifests(
        release_id=release_id,
        release_dir=release_dir,
        artifact_root=artifact_root,
        result=result,
        registry=FakeRegistry(),
        dropped={"dropped_target_names": []},
        target_profile_gate=builder.GateResult(
            name="target_profile_coverage",
            passed=True,
            details={"requirements_checked": 1},
        ),
        incumbent_diagnostics={
            name: {
                "target": target,
                "final_estimate": 134_904_000_000.0,
            }
        },
    )

    build_manifest = json.loads((release_dir / "build_manifest.json").read_text())
    assert build_manifest["gates"]["calibration"] == {
        "passed": True,
        "failures": [],
        "initial_loss": 2.0,
        "final_loss": 1.0,
        "fraction_within_10pct": 1.0,
    }


def test_export_frame_rejects_formula_owned_columns(monkeypatch, small_frame) -> None:
    builder = _load_builder_module()

    class FakePolicyEngineUSEngine:
        def _engine_computed_columns(self, tables, *, period):
            assert period == builder.PERIOD
            assert "income" in tables["person"]
            return {"income"}

    monkeypatch.setattr(builder, "PolicyEngineUSEngine", FakePolicyEngineUSEngine)

    with pytest.raises(ValueError, match="Formula-owned.*income"):
        builder._with_calibrated_weights(
            small_frame,
            np.array([1000.0, 2000.0]),
        )


def test_dataset_from_frame_rejects_formula_owned_columns_by_default(
    monkeypatch,
    small_frame,
) -> None:
    builder = _load_builder_module()

    def fake_assert_no_formula_owned_columns(frame):
        assert frame is small_frame
        raise ValueError("formula-owned guard fired")

    monkeypatch.setattr(
        builder,
        "_assert_no_formula_owned_columns",
        fake_assert_no_formula_owned_columns,
    )

    with pytest.raises(ValueError, match="formula-owned guard fired"):
        builder._dataset_from_frame(small_frame)


def test_export_frame_accepts_leaf_only_columns(monkeypatch, small_frame) -> None:
    builder = _load_builder_module()

    class FakePolicyEngineUSEngine:
        def _engine_computed_columns(self, tables, *, period):
            assert period == builder.PERIOD
            assert "income" in tables["person"]
            return set()

    monkeypatch.setattr(builder, "PolicyEngineUSEngine", FakePolicyEngineUSEngine)

    exported = builder._with_calibrated_weights(
        small_frame,
        np.array([1000.0, 2000.0]),
    )

    assert "income" in exported.table("person")
    assert exported.weights_for("household").kind == WeightKind.CALIBRATED


def test_l0_refit_export_subsets_clean_base_frame(monkeypatch, small_frame) -> None:
    builder = _load_builder_module()
    monkeypatch.setattr(builder, "_assert_no_formula_owned_columns", lambda frame: None)
    result = SimpleNamespace(
        selected_entity_ids=np.asarray([2], dtype="int64"),
        weight_entity="household",
        weights=np.asarray([3333.0]),
    )

    exported = builder._with_l0_refit_weights(small_frame, result)

    assert exported.table("household")["household_id"].to_list() == [2]
    assert exported.table("person")["person_id"].to_list() == [2, 3]
    np.testing.assert_allclose(
        exported.weights_for("household").values,
        np.asarray([3333.0]),
    )
    assert exported.weights_for("household").kind == WeightKind.CALIBRATED


def test_post_export_sanity_checks_full_target_surface(monkeypatch, tmp_path) -> None:
    builder = _load_builder_module()

    class FakeWeights:
        values = np.asarray([1.0])

    class FakeFrame:
        def weights_for(self, entity):
            assert entity == "household"
            return FakeWeights()

    class FakeTarget:
        entity = "household"
        row_name = f"nation/cbo/individual_income_tax@{builder.PERIOD}"

        def __init__(self):
            self.observed = 2_000_000_000_000.0

        def achieved_value(self, frame, weights):
            assert isinstance(frame, FakeFrame)
            assert np.array_equal(weights, np.asarray([1.0]))
            return self.observed

    target = FakeTarget()

    class FakeRegistry:
        def to_target_set(self):
            return (target,)

    monkeypatch.setattr(builder, "_load_frame", lambda path: f"loaded:{path}")
    monkeypatch.setattr(
        builder,
        "_materialize_target_frame",
        lambda frame, target_specs, **kwargs: (
            FakeFrame(),
            FakeRegistry(),
            {"dropped_target_names": []},
        ),
    )

    result = SimpleNamespace(
        diagnostics=(
            SimpleNamespace(
                name=f"nation/cbo/individual_income_tax@{builder.PERIOD}",
                final_estimate=2_000_000_000_000.0,
            ),
        )
    )

    builder._assert_export_matches_calibration(tmp_path / "candidate.h5", result, ())

    target.observed = 2_000_900_000_000.0
    builder._assert_export_matches_calibration(tmp_path / "candidate.h5", result, ())

    target.observed = 1_990_000_000_000.0
    try:
        builder._assert_export_matches_calibration(
            tmp_path / "candidate.h5", result, ()
        )
    except RuntimeError as exc:
        assert "Post-export sanity failed" in str(exc)
        assert "nation/cbo/individual_income_tax@2024 exported value" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected post-export sanity failure.")


def test_post_export_sanity_rejects_dropped_export_targets(
    monkeypatch, tmp_path
) -> None:
    builder = _load_builder_module()
    monkeypatch.setattr(builder, "_load_frame", lambda path: object())
    monkeypatch.setattr(
        builder,
        "_materialize_target_frame",
        lambda frame, target_specs, **kwargs: (
            object(),
            object(),
            {"dropped_target_names": ["missing"]},
        ),
    )

    try:
        builder._assert_export_matches_calibration(
            tmp_path / "candidate.h5", SimpleNamespace(diagnostics=()), ()
        )
    except RuntimeError as exc:
        assert "1 fiscal targets were not materialized after export" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected dropped-target post-export sanity failure.")


def test_reviewed_exclusions_are_exact_for_fiscal_refresh() -> None:
    builder = _load_builder_module()

    exclusions = builder._reviewed_exclusions(builder.DIRECT_ACTIVE_ALIASES)

    assert tuple(exclusions) == builder.REVIEWED_EXCLUDED_ALIASES


def test_fiscal_refresh_uses_target_period_medicaid_source() -> None:
    builder = _load_builder_module()

    assert (
        "cms-medicaid-chip-monthly-enrollment-december-2024"
        in builder.DIRECT_ACTIVE_ALIASES
    )
    assert (
        "cms-medicaid-chip-monthly-enrollment-dataset"
        in builder.REVIEWED_EXCLUDED_ALIASES
    )


def test_fiscal_refresh_keeps_unregistered_aca_state_metal_alias_inactive() -> None:
    builder = _load_builder_module()

    assert "cms-aca-oep-state-level" in builder.DIRECT_ACTIVE_ALIASES
    assert "cms-aca-oep-state-metal" not in builder.DIRECT_ACTIVE_ALIASES


def test_reviewed_exclusions_fail_when_hard_target_surface_changes(
    monkeypatch,
) -> None:
    builder = _load_builder_module()
    monkeypatch.setattr(
        builder,
        "hard_target_package_aliases",
        lambda: (*builder.DIRECT_ACTIVE_ALIASES, "new-hard-target"),
    )

    try:
        builder._reviewed_exclusions(builder.DIRECT_ACTIVE_ALIASES)
    except RuntimeError as exc:
        assert "Reviewed hard-target exclusion list is stale" in str(exc)
        assert "new-hard-target" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected stale reviewed-exclusion failure.")


def test_fiscal_target_source_provenance_covers_active_families() -> None:
    builder = _load_builder_module()
    specs = (
        TargetSpec(
            name="income_tax",
            entity="household",
            measure="income_tax",
            value=1,
            source="CBO source",
            family="cbo",
        ),
        TargetSpec(
            name="salt",
            entity="household",
            measure="salt",
            value=1,
            source="JCT source",
            family="jct",
            metadata={"reference_url": "https://example.org/jct"},
        ),
        TargetSpec(
            name="agi",
            entity="household",
            measure="agi",
            value=1,
            source="SOI source",
            family="irs_soi",
        ),
        TargetSpec(
            name="state_income_tax",
            entity="household",
            measure="state_income_tax",
            value=1,
            source="Census source",
            family="state_income_tax",
            metadata={"reference_url": "https://example.org/stc"},
        ),
    )

    provenance = builder._fiscal_target_source_provenance(specs)

    assert set(provenance) == {"cbo", "irs_soi", "jct", "state_income_tax"}
    assert provenance["cbo"]["target_count"] == 1
    assert provenance["jct"]["target_count"] == 1
    assert provenance["irs_soi"]["sources"]
    assert provenance["state_income_tax"]["reference_urls"]


def test_us_release_id_guard() -> None:
    builder = _load_builder_module()

    builder._assert_us_release_id("populace-us-2024-base-commit-20260615T000000Z")

    try:
        builder._assert_us_release_id("populace-uk-2024-base-commit-20260615T000000Z")
    except ValueError as exc:
        assert "must start with 'populace-us-'" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected non-US release id to fail.")


def test_staging_telemetry_defaults_on_and_no_staging_disables(tmp_path, monkeypatch):
    module = _load_builder_module()

    # The parser defaults staging uploads ON (overridable by env).
    monkeypatch.delenv("POPULACE_STAGING_REPO_ID", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
        ],
    )
    args = module._parse_args()
    assert args.staging_repo_id == "policyengine/populace-us-staging"
    assert not args.no_staging

    def namespace(no_staging: bool) -> SimpleNamespace:
        # repo_id None keeps the constructed telemetry offline in tests.
        return SimpleNamespace(
            no_staging=no_staging,
            staging_dir=tmp_path / "stage",
            staging_repo_id=None,
            staging_run_id=None,
            staging_prefix=args.staging_prefix,
            staging_upload_interval_seconds=60.0,
        )

    telemetry = module._staging_telemetry(
        namespace(no_staging=False), release_root=tmp_path, release_id="rel-1"
    )
    assert telemetry is not None
    assert telemetry.run_id == "rel-1"

    # --no-staging wins even when a staging destination is configured.
    assert (
        module._staging_telemetry(
            namespace(no_staging=True), release_root=tmp_path, release_id="rel-1"
        )
        is None
    )


# ---------------------------------------------------------------------------
# #299 / #217: per-reform materialization checkpoint resume + cache-key safety.
#
# These prove that a run killed mid target_compilation resumes from the durable
# per-reform cache (only the un-computed reforms recompute), and that the
# reform-vector cache key invalidates when the reform vector or the frame
# identity changes (so a stale checkpoint can never poison a build), while a
# build-commit-only change reuses the cache (the #217 acceptance criterion).
# ---------------------------------------------------------------------------


class _ReformKillError(RuntimeError):
    """Sentinel raised to simulate a process kill mid target_compilation."""


def _multi_reform_frame(builder):
    """A 2-household frame with 3 tax units, matching the JCT-loop test shape."""
    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.asarray([1, 2, 3], dtype="int64"),
                    "person_household_id": np.asarray([1, 1, 2], dtype="int64"),
                    "person_tax_unit_id": np.asarray([10, 20, 30], dtype="int64"),
                    "person_spm_unit_id": np.asarray([100, 100, 200], dtype="int64"),
                    "person_family_id": np.asarray([1000, 1000, 2000], dtype="int64"),
                    "person_marital_unit_id": np.asarray(
                        [10000, 20000, 30000], dtype="int64"
                    ),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.asarray([1, 2], dtype="int64"),
                    "state_fips": np.asarray([6, 36], dtype="int64"),
                }
            ),
            "tax_unit": pd.DataFrame(
                {"tax_unit_id": np.asarray([10, 20, 30], dtype="int64")}
            ),
            "spm_unit": pd.DataFrame(
                {"spm_unit_id": np.asarray([100, 200], dtype="int64")}
            ),
            "family": pd.DataFrame({"family_id": np.asarray([1000, 2000])}),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": np.asarray([10000, 20000, 30000])}
            ),
        },
        builder.US_SCHEMA,
        {
            "household": builder.Weights(
                values=np.asarray([1.0, 1.0]), kind=WeightKind.DESIGN
            )
        },
    )


def _install_multi_reform_fakes(
    builder,
    monkeypatch,
    *,
    reforms,
    reform_income_tax_by_id,
    reform_sim_calls,
    raise_on_call=None,
):
    """Wire fake PE-US so ``_materialize_target_frame`` runs over the tiny frame.

    ``reforms`` is a tuple of ``(measure, neutralized_variable)`` pairs.
    ``reform_income_tax_by_id`` maps ``neutralized_variable -> {tax_unit_id: tax}``.
    Every real reform simulation appends its ``neutralized_variable`` to
    ``reform_sim_calls``; if the resulting call count equals ``raise_on_call`` the
    fake raises ``_ReformKillError`` before returning (simulating a kill while that
    reform is being materialized, so it is never cached).
    """
    reform_specs = tuple(
        SimpleNamespace(measure=measure, neutralized_variable=variable)
        for measure, variable in reforms
    )
    base_income_tax_by_id = {10: 100.0, 20: 30.0, 30: 70.0}

    class FakeVariable:
        entity = SimpleNamespace(key="tax_unit")

    class FakeSystem:
        variables = {
            "state_income_tax": FakeVariable(),
            **{variable: FakeVariable() for _, variable in reforms},
        }

        def __init__(self, reform=None):
            self.reform = reform

    class FakeMicrosimulation:
        default_tax_benefit_system = FakeSystem

        def __init__(self, *, dataset, reform=None, tax_benefit_system=None):
            self.dataset = dataset
            self.reform = reform
            self.tax_benefit_system = tax_benefit_system
            self.cache_invalidations = 0

        def calculate(self, variable, *, period, **kwargs):
            assert period == builder.PERIOD
            tax_unit_ids = (
                self.dataset["frame"].table("tax_unit")["tax_unit_id"].to_numpy()
            )
            if self.reform is not None:
                assert variable == "income_tax"
                lookup = reform_income_tax_by_id[self.reform]
                return np.asarray([lookup[id_] for id_ in tax_unit_ids])
            arrays_by_id = {
                "income_tax": base_income_tax_by_id,
                "taxable_income": {10: 1000.0, 20: 2000.0, 30: 3000.0},
                "adjusted_gross_income": {10: 1100.0, 20: 2100.0, 30: 3100.0},
                "filing_status": {10: "SINGLE", 20: "SINGLE", 30: "SINGLE"},
                "state_income_tax": {10: 5.0, 20: 6.0, 30: 7.0},
            }
            return np.asarray([arrays_by_id[variable][id_] for id_ in tax_unit_ids])

        def _invalidate_all_caches(self):
            self.cache_invalidations += 1

    def fake_dataset_from_frame(
        frame_arg,
        *,
        zero_variables=(),
        system=None,
        assert_no_formula_owned_columns=True,
    ):
        return {"frame": frame_arg, "zero_variables": tuple(zero_variables)}

    def fake_make_zero_variable_reform(system, variable_name):
        # The loop passes the neutralized variable straight through; the fake sim
        # keys its reform result off this value.
        return variable_name

    monkeypatch.setitem(
        sys.modules,
        "policyengine_us",
        SimpleNamespace(
            CountryTaxBenefitSystem=FakeSystem,
            Microsimulation=FakeMicrosimulation,
        ),
    )
    monkeypatch.setattr(
        builder,
        "_assert_no_formula_owned_columns",
        lambda frame_arg: None,
    )
    monkeypatch.setattr(builder, "_dataset_from_frame", fake_dataset_from_frame)
    monkeypatch.setattr(
        builder, "_make_zero_variable_reform", fake_make_zero_variable_reform
    )
    monkeypatch.setattr(builder, "US_JCT_TAX_EXPENDITURE_REFORMS", reform_specs)
    monkeypatch.setattr(builder, "SOI_VARIABLE_MAP", {})

    real_reform_household_income_tax = builder._reform_household_income_tax

    def counting_reform_household_income_tax(*, reform_spec, **kwargs):
        reform_sim_calls.append(reform_spec.neutralized_variable)
        if raise_on_call is not None and len(reform_sim_calls) == raise_on_call:
            raise _ReformKillError(
                f"killed while materializing {reform_spec.measure!r}"
            )
        return real_reform_household_income_tax(reform_spec=reform_spec, **kwargs)

    monkeypatch.setattr(
        builder,
        "_reform_household_income_tax",
        counting_reform_household_income_tax,
    )

    targets = tuple(
        TargetSpec(
            name=f"jct.{measure}@{builder.PERIOD}",
            entity="household",
            measure=measure,
            value=-45.0,
            source="Mock JCT",
            family="jct",
            signed=True,
        )
        for measure, _ in reforms
    )
    return reform_specs, targets


def _base_cache_context(builder):
    return {
        "base_dataset_sha256": "base-sha-A",
        "weeks_unemployed_source_sha256": "weeks-source-sha-A",
        "build_commit": "commit-A",
        "policyengine_us_version": "pe-us-A",
        "seed": 0,
        "target_period": builder.PERIOD,
        "target_registry_version": "registry-A",
        "congressional_district_vintage_crosswalk_sha256": None,
        "target_frame_materializer_identity_sha256": "materializer-sha-A",
    }


def test__given_kill_after_two_reforms__then_restart_only_recomputes_the_third(
    monkeypatch,
    tmp_path,
) -> None:
    builder = _load_builder_module()
    reforms = (
        ("jct_reform_a", "credit_a"),
        ("jct_reform_b", "credit_b"),
        ("jct_reform_c", "credit_c"),
    )
    reform_income_tax_by_id = {
        "credit_a": {10: 90.0, 20: 25.0, 30: 40.0},
        "credit_b": {10: 80.0, 20: 20.0, 30: 35.0},
        "credit_c": {10: 70.0, 20: 15.0, 30: 30.0},
    }
    context = _base_cache_context(builder)
    frame = _multi_reform_frame(builder)

    # First pass: die while materializing the 3rd reform. Reforms 1 and 2 complete
    # and are written to the durable cache; reform 3 never is.
    first_calls: list[str] = []
    _, targets = _install_multi_reform_fakes(
        builder,
        monkeypatch,
        reforms=reforms,
        reform_income_tax_by_id=reform_income_tax_by_id,
        reform_sim_calls=first_calls,
        raise_on_call=3,
    )
    with pytest.raises(_ReformKillError):
        builder._materialize_target_frame(
            frame,
            targets,
            maximum_microsim_batch_size=1,
            target_materialization_cache_dir=tmp_path,
            target_materialization_cache_context=context,
        )
    # Exactly three reform sims were attempted (1, 2 succeeded; 3 raised).
    assert first_calls == ["credit_a", "credit_b", "credit_c"]
    # Two durable cache entries exist on disk (reforms 1 and 2 only).
    assert len(list(tmp_path.glob("*.npy"))) == 2
    assert len(list(tmp_path.glob("*.json"))) == 2

    # Restart: same inputs, no kill. Reforms 1 and 2 must load from cache; only
    # reform 3 recomputes.
    second_calls: list[str] = []
    _install_multi_reform_fakes(
        builder,
        monkeypatch,
        reforms=reforms,
        reform_income_tax_by_id=reform_income_tax_by_id,
        reform_sim_calls=second_calls,
        raise_on_call=None,
    )
    target_frame, registry, compilation = builder._materialize_target_frame(
        frame,
        targets,
        maximum_microsim_batch_size=1,
        target_materialization_cache_dir=tmp_path,
        target_materialization_cache_context=context,
    )

    # ONLY the third reform recomputed on restart.
    assert second_calls == ["credit_c"]
    stats = compilation["target_materialization_cache"]
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["writes"] == 1

    # Results are correct for all three reforms (reform_income_tax - base).
    household = target_frame.table("household")
    # base income_tax by household: hh1 = 100+30 = 130, hh2 = 70.
    np.testing.assert_allclose(
        household["jct_reform_a"], [90.0 + 25.0 - 130.0, 40.0 - 70.0]
    )
    np.testing.assert_allclose(
        household["jct_reform_b"], [80.0 + 20.0 - 130.0, 35.0 - 70.0]
    )
    np.testing.assert_allclose(
        household["jct_reform_c"], [70.0 + 15.0 - 130.0, 30.0 - 70.0]
    )
    assert len(registry) == 3


def test__given_changed_reform_vector__then_stale_checkpoint_is_not_reused(
    monkeypatch,
    tmp_path,
) -> None:
    builder = _load_builder_module()
    context = _base_cache_context(builder)
    frame = _multi_reform_frame(builder)

    # Materialize reform "credit_a" and cache it.
    calls_one: list[str] = []
    _, targets_one = _install_multi_reform_fakes(
        builder,
        monkeypatch,
        reforms=(("jct_reform_a", "credit_a"),),
        reform_income_tax_by_id={"credit_a": {10: 90.0, 20: 25.0, 30: 40.0}},
        reform_sim_calls=calls_one,
        raise_on_call=None,
    )
    builder._materialize_target_frame(
        frame,
        targets_one,
        maximum_microsim_batch_size=1,
        target_materialization_cache_dir=tmp_path,
        target_materialization_cache_context=context,
    )
    assert calls_one == ["credit_a"]

    # Now the SAME measure name but a DIFFERENT neutralized variable (the reform
    # vector changed). The old entry must NOT be reused: a different vector means a
    # different per-household estimate. If the key ignored the vector this would
    # silently reuse the stale credit_a values and poison the build.
    calls_two: list[str] = []
    _, targets_two = _install_multi_reform_fakes(
        builder,
        monkeypatch,
        reforms=(("jct_reform_a", "credit_a_v2"),),
        reform_income_tax_by_id={"credit_a_v2": {10: 10.0, 20: 5.0, 30: 1.0}},
        reform_sim_calls=calls_two,
        raise_on_call=None,
    )
    target_frame, _, compilation = builder._materialize_target_frame(
        frame,
        targets_two,
        maximum_microsim_batch_size=1,
        target_materialization_cache_dir=tmp_path,
        target_materialization_cache_context=context,
    )
    # It recomputed rather than reusing the stale entry.
    assert calls_two == ["credit_a_v2"]
    assert compilation["target_materialization_cache"]["misses"] == 1
    assert compilation["target_materialization_cache"]["hits"] == 0
    household = target_frame.table("household")
    # Uses the NEW vector: hh1 = 10+5-130 = -115, hh2 = 1-70 = -69.
    np.testing.assert_allclose(household["jct_reform_a"], [-115.0, -69.0])


@pytest.mark.parametrize(
    ("identity_key", "new_value"),
    [
        ("base_dataset_sha256", "base-sha-B"),
        ("weeks_unemployed_source_sha256", "weeks-source-sha-B"),
        ("target_frame_materializer_identity_sha256", "materializer-sha-B"),
    ],
)
def test__given_changed_frame_identity__then_stale_checkpoint_is_not_reused(
    monkeypatch,
    tmp_path,
    identity_key,
    new_value,
) -> None:
    builder = _load_builder_module()
    frame = _multi_reform_frame(builder)
    reforms = (("jct_reform_a", "credit_a"),)
    reform_income_tax_by_id = {"credit_a": {10: 90.0, 20: 25.0, 30: 40.0}}

    # Cache under a base H5 identity "base-sha-A".
    context_a = _base_cache_context(builder)
    calls_a: list[str] = []
    _, targets = _install_multi_reform_fakes(
        builder,
        monkeypatch,
        reforms=reforms,
        reform_income_tax_by_id=reform_income_tax_by_id,
        reform_sim_calls=calls_a,
        raise_on_call=None,
    )
    builder._materialize_target_frame(
        frame,
        targets,
        maximum_microsim_batch_size=1,
        target_materialization_cache_dir=tmp_path,
        target_materialization_cache_context=context_a,
    )
    assert calls_a == ["credit_a"]

    # A different base H5, measured LKWEEKS source, or complete target-frame
    # materializer identity must not share vectors at the same record count.
    context_b = _base_cache_context(builder)
    context_b[identity_key] = new_value
    calls_b: list[str] = []
    _install_multi_reform_fakes(
        builder,
        monkeypatch,
        reforms=reforms,
        reform_income_tax_by_id=reform_income_tax_by_id,
        reform_sim_calls=calls_b,
        raise_on_call=None,
    )
    _, _, compilation = builder._materialize_target_frame(
        frame,
        targets,
        maximum_microsim_batch_size=1,
        target_materialization_cache_dir=tmp_path,
        target_materialization_cache_context=context_b,
    )
    assert calls_b == ["credit_a"]
    assert compilation["target_materialization_cache"]["misses"] == 1
    assert compilation["target_materialization_cache"]["hits"] == 0


def test__given_only_build_commit_changed__then_reform_cache_is_reused(
    monkeypatch,
    tmp_path,
) -> None:
    # #217 acceptance criterion 1: a rerun that changes only the build commit
    # must reuse the cached reform vectors rather than recompute them.
    builder = _load_builder_module()
    frame = _multi_reform_frame(builder)
    reforms = (("jct_reform_a", "credit_a"),)
    reform_income_tax_by_id = {"credit_a": {10: 90.0, 20: 25.0, 30: 40.0}}

    context_a = _base_cache_context(builder)
    calls_a: list[str] = []
    _, targets = _install_multi_reform_fakes(
        builder,
        monkeypatch,
        reforms=reforms,
        reform_income_tax_by_id=reform_income_tax_by_id,
        reform_sim_calls=calls_a,
        raise_on_call=None,
    )
    builder._materialize_target_frame(
        frame,
        targets,
        maximum_microsim_batch_size=1,
        target_materialization_cache_dir=tmp_path,
        target_materialization_cache_context=context_a,
    )
    assert calls_a == ["credit_a"]

    # Only build_commit changes. The full materializer identity remains equal,
    # so the reform must load from cache.
    context_b = _base_cache_context(builder)
    context_b["build_commit"] = "commit-B"
    calls_b: list[str] = []
    _install_multi_reform_fakes(
        builder,
        monkeypatch,
        reforms=reforms,
        reform_income_tax_by_id=reform_income_tax_by_id,
        reform_sim_calls=calls_b,
        raise_on_call=None,
    )
    _, _, compilation = builder._materialize_target_frame(
        frame,
        targets,
        maximum_microsim_batch_size=1,
        target_materialization_cache_dir=tmp_path,
        target_materialization_cache_context=context_b,
    )
    # No reform sim ran on the second pass — it was a pure cache hit.
    assert calls_b == []
    assert compilation["target_materialization_cache"]["hits"] == 1
    assert compilation["target_materialization_cache"]["misses"] == 0
    assert compilation["target_materialization_cache"]["writes"] == 0


def _sentinel_frame(label: str):
    """A stand-in frame whose only role is object identity.

    The export-gate tests patch ``us_input_mass_totals`` to key off ``id(frame)``
    and never touch the schema, so a bare tagged object suffices — building a
    real US-schema Frame (which needs the engine + every entity table) is
    unnecessary to exercise the #327 reference-selection logic.
    """
    return SimpleNamespace(label=label)


def test_export_input_mass_gate_defaults_to_base_reference(monkeypatch) -> None:
    """#327: with no reference_frame, the export gate compares vs the raw base.

    This is the historical behaviour, preserved: a PUF-imputed column that
    calibration scales far above its raw-base mass (capital gains here) fails
    the ±50% band against the raw base.
    """
    builder = _load_builder_module()

    export = _sentinel_frame("export")
    base = _sentinel_frame("base")

    totals = {
        id(export): {"long_term_capital_gains": 7.02e11, "employment_income": 1.1e13},
        id(base): {"long_term_capital_gains": 2.12e11, "employment_income": 1.1e13},
    }
    monkeypatch.setattr(builder, "_engine_input_variables", lambda: ())
    monkeypatch.setattr(
        builder,
        "us_input_mass_totals",
        lambda frame, columns=None: totals[id(frame)],
    )

    gate = builder._export_input_mass_gate(
        export,
        base,
        relative_tolerance=0.5,
        minimum_reference_total=1e9,
    )
    assert not gate.passed
    assert any("long_term_capital_gains" in f for f in gate.failures)
    # employment_income (unchanged) does not fail.
    assert not any("employment_income" in f for f in gate.failures)


def test_export_input_mass_gate_passes_against_certified_reference(monkeypatch) -> None:
    """#327: with the live-default reference, calibration gains are in-band.

    The export mass (capital gains scaled up toward SOI/CBO) is far above the
    raw base but ~equal to the certified live-default reference — so against the
    reference the gate passes, vindicating the 11/14 mis-referenced columns.
    """
    builder = _load_builder_module()

    export = _sentinel_frame("export")
    base = _sentinel_frame("base")
    reference = _sentinel_frame("reference")

    totals = {
        # export >> raw base (the +230% the raw-base gate flagged), but export
        # is within ±50% of the certified reference (per #327: -18.8%).
        id(export): {"long_term_capital_gains": 7.02e11},
        id(base): {"long_term_capital_gains": 2.12e11},
        id(reference): {"long_term_capital_gains": 8.64e11},
    }
    monkeypatch.setattr(builder, "_engine_input_variables", lambda: ())
    monkeypatch.setattr(
        builder,
        "us_input_mass_totals",
        lambda frame, columns=None: totals[id(frame)],
    )

    gate = builder._export_input_mass_gate(
        export,
        base,
        relative_tolerance=0.5,
        minimum_reference_total=1e9,
        reference_frame=reference,
        reference_name="populace_us_2024.h5",
    )
    assert gate.passed, gate.failures


def test_export_input_mass_gate_still_fails_genuine_drift_vs_reference(
    monkeypatch,
) -> None:
    """#327: the loss/zeroing arm stays strict against the reference.

    A genuine #278 zeroing (a sparse selection dropping an untargeted input the
    reference carries) still fails even when the reference — not the raw base —
    is the yardstick.
    """
    builder = _load_builder_module()

    export = _sentinel_frame("export")
    base = _sentinel_frame("base")
    reference = _sentinel_frame("reference")

    totals = {
        # traditional_ira_contributions zeroed in the export (the #278 signature);
        # health_savings_account halved (drift beyond ±50% vs the reference).
        id(export): {
            "traditional_ira_contributions": 0.0,
            "health_savings_account_ald": 5.0e9,
        },
        id(base): {
            "traditional_ira_contributions": 3.0e10,
            "health_savings_account_ald": 1.4e10,
        },
        id(reference): {
            "traditional_ira_contributions": 3.1e10,
            "health_savings_account_ald": 1.38e10,
        },
    }
    monkeypatch.setattr(builder, "_engine_input_variables", lambda: ())
    monkeypatch.setattr(
        builder,
        "us_input_mass_totals",
        lambda frame, columns=None: totals[id(frame)],
    )

    gate = builder._export_input_mass_gate(
        export,
        base,
        relative_tolerance=0.5,
        minimum_reference_total=1e9,
        reference_frame=reference,
        reference_name="populace_us_2024.h5",
    )
    assert not gate.passed
    assert any("traditional_ira_contributions" in f for f in gate.failures)
    assert any("health_savings_account_ald" in f for f in gate.failures)


def test_main_runs_cross_register_and_take_up_contract_preflights() -> None:
    """main() must call the cheap consistency preflights before source stages.

    populace#377 (register consistency) and populace#381 (take-up contract
    engine-drift) both abort a build in seconds when a register is stale. A
    regression that drops the preflight call would only surface after hours of
    source staging, so pin the wiring at the code-object level (these globals
    are looked up by name inside ``main``).
    """
    builder = _load_builder_module()
    called = set(builder.main.__code__.co_names)
    for preflight in (
        "assert_release_input_coverage_manifest_current",
        "us_register_consistency_gate",
        "assert_take_up_contract_current",
        "assert_take_up_treatments_consistent",
    ):
        assert preflight in called, f"main() no longer calls {preflight}"


def _spm_state_frame(states: list[str], *, split_unit: bool = False):
    import numpy as np
    import pandas as pd

    from populace.frame import Frame, WeightKind, Weights
    from populace.frame.units import US_SCHEMA

    person_rows = []
    for index in range(len(states)):
        for member in range(2):
            person_rows.append(
                {
                    "person_id": index * 10 + member,
                    "person_household_id": index,
                    # A split unit wires its second member to another
                    # household (and so another state) to hit the guard.
                    "person_spm_unit_id": (
                        (index + 1) % len(states)
                        if split_unit and index == 0 and member == 1
                        else index
                    ),
                    "person_tax_unit_id": index,
                    "person_family_id": index,
                    "person_marital_unit_id": index * 10 + member,
                    "age": 40,
                }
            )
    person = pd.DataFrame(person_rows)
    ids = np.arange(len(states), dtype="int64")
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": ids, "state_fips": states}),
        "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": person["person_marital_unit_id"].to_numpy()}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.ones(len(states), dtype=np.float64),
                kind=WeightKind.DESIGN,
            )
        },
    )


def test_spm_unit_state_fips_routes_through_persons() -> None:
    """SPM-unit state codes come via the person linkage, not broadcast().

    Build M's sparse run died calling ``frame.broadcast(..., to="spm_unit")``
    — the Frame API only broadcasts to persons, and the SNAP state take-up
    path (a coverage-campaign restoration) had never run at full scale. The
    helper now routes household state through persons and collapses per SPM
    unit, failing closed if a unit ever spans two states.
    """

    builder = _load_builder_module()
    frame = _spm_state_frame(["06", "36"])
    aligned = builder._spm_unit_state_fips(frame)
    assert list(aligned) == ["06", "36"]

    with pytest.raises(ValueError, match="span multiple state"):
        builder._spm_unit_state_fips(_spm_state_frame(["06", "36"], split_unit=True))


def test_release_h5_write_sits_between_batched_raise_and_smoke() -> None:
    """populace#443: #437 dropped release_engine.write_dataset(...) while
    inserting the batched pre-export raise, so the smoke gate scored a stale
    artifact from a prior run (and the manifest would have sha-pinned it).
    Pin main()'s ordering contract at the AST level until the green-path
    main() harness exists: exactly one export H5 write, strictly after the
    single batched pre-export raise and before the reform-coverage smoke
    reads dataset_path."""
    import ast
    import inspect

    builder = _load_builder_module()
    source = inspect.getsource(builder.main)
    tree = ast.parse(source)

    batched_raises: list[int] = []
    writes: list[int] = []
    smokes: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            segment = ast.get_source_segment(source, node) or ""
            if "terminal_gate_failures" in segment:
                batched_raises.append(node.lineno)
        elif isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", None)
            )
            if name == "write_dataset":
                writes.append(node.lineno)
            elif name == "us_reform_coverage_smoke_gate":
                smokes.append(node.lineno)

    assert len(batched_raises) == 1, batched_raises
    assert len(writes) == 1, (
        "main() must write the export H5 exactly once; the smoke gate and "
        f"release manifest read that file (populace#443). Found: {writes}"
    )
    assert len(smokes) == 1, smokes
    assert batched_raises[0] < writes[0] < smokes[0], (
        "Ordering contract violated: batched pre-export raise "
        f"(line {batched_raises[0]}) < H5 write ({writes[0]}) < smoke "
        f"({smokes[0]}) must hold so a gate-failed run never produces the "
        "H5 and the smoke scores the just-written file."
    )


def test_selection_mass_protection_specs_measure_locked_source_mass(
    small_frame,
) -> None:
    """populace#445: the protection target's value is the base pool's own
    locked-source mass at base weights, measured at build time — and it rides
    the standard policyengine_variable materializer contract so both the
    fresh-materialize and checkpoint-reload paths compile it."""
    builder = _load_builder_module()

    (spec,) = builder._selection_mass_protection_specs(small_frame, ("income",))

    assert spec.name == "selection_mass_protection.income"
    assert spec.measure == "selection_mass_protection.income"
    assert spec.entity == "household"
    assert spec.value == 100.0 * 1000 + 250.0 * 2000 + 50.0 * 2000
    assert spec.metadata["materializer"] == "policyengine_variable"
    assert spec.metadata["measure_mode"] == "sum"
    assert spec.metadata["base_variable"] == "income"
    assert spec.metadata["target_role"] == "selection_mass_protection"
    assert spec.metadata["protected_entity"] == "person"
    assert spec.metadata["base_pool_carriers"] == "3"
    assert spec.metadata["issue"] == "PolicyEngine/populace#445"
    assert not spec.signed


def test_selection_mass_protection_specs_fail_closed(small_frame) -> None:
    builder = _load_builder_module()

    with pytest.raises(RuntimeError, match="absent from every entity table"):
        builder._selection_mass_protection_specs(small_frame, ("keogh_missing",))

    zeroed = small_frame.table("person").copy()
    zeroed["income"] = 0.0
    frame = Frame(
        {"person": zeroed, "household": small_frame.table("household").copy()},
        small_frame.schema,
        {"household": small_frame.weights_for("household")},
        small_frame.strata,
    )
    with pytest.raises(RuntimeError, match="no nonzero carriers"):
        builder._selection_mass_protection_specs(frame, ("income",))


def test_checkpoint_identity_protection_key_and_stale_checkpoint_miss(
    monkeypatch, tmp_path, small_frame
) -> None:
    """populace#445: unprotected runs keep their legacy identity (digest
    bit-identical — the dense arm's warm checkpoints must stay valid), and a
    protected run MISSES a column-less legacy checkpoint instead of loading
    it (a load would silently drop the protection spec, because
    _compile_materialized_target_registry keeps only specs whose measures
    exist on the materialized household table)."""
    builder = _load_builder_module()
    monkeypatch.setattr(builder, "US_SCHEMA", small_frame.schema)

    common = dict(
        base_dataset_sha256="base-sha",
        policyengine_us_version="1.2.3",
        seed=0,
        target_period=builder.PERIOD,
        target_registry_version="registry-sha",
        weeks_unemployed_source_sha256="weeks-source-sha",
        congressional_district_vintage_crosswalk_sha256="crosswalk-sha",
        ssi_take_up_assignment_sha256="ssi-flags-sha",
        selection_identities_sha256=None,
    )
    legacy = builder._target_frame_checkpoint_identity(**common)
    default_kwarg = builder._target_frame_checkpoint_identity(
        **common, selection_mass_protections=()
    )
    protected = builder._target_frame_checkpoint_identity(
        **common, selection_mass_protections=("keogh_distributions",)
    )

    assert "selection_mass_protections" not in legacy
    assert builder._target_frame_checkpoint_digest(
        default_kwarg
    ) == builder._target_frame_checkpoint_digest(legacy)
    assert protected["selection_mass_protections"] == ["keogh_distributions"]

    # populace#507/#508: a retry whose frozen SSI assignment differs (the
    # --ssi-take-up-prior-weight-basis path) must MISS the previous
    # attempt's checkpoint — otherwise the solve runs on stale SSI rows
    # while the export ships fresh ones (split-brain certification).
    retried = builder._target_frame_checkpoint_identity(
        **{**common, "ssi_take_up_assignment_sha256": "ssi-flags-sha-retry"}
    )
    assert legacy["ssi_take_up_assignment_sha256"] == "ssi-flags-sha"
    assert builder._target_frame_checkpoint_digest(
        retried
    ) != builder._target_frame_checkpoint_digest(legacy)
    assert builder._target_frame_checkpoint_digest(
        protected
    ) != builder._target_frame_checkpoint_digest(legacy)

    path = tmp_path / "target_frame_checkpoint.h5"
    builder._write_target_frame_checkpoint(
        path,
        frame=small_frame,
        identity=legacy,
        compilation={"declared_targets": 0},
    )
    assert (
        builder._read_target_frame_checkpoint(path, identity=protected, target_specs=())
        is None
    )


def test_checkpoint_identity_tracks_selection_and_rejects_prefix_shape(
    monkeypatch, tmp_path, small_frame
) -> None:
    """A frozen-support change invalidates the checkpoint identity.

    The assignment digest hashes positional flags, priors, and provenance,
    but two same-length supports can share those flag bytes. The selected
    source identities therefore remain an independent checkpoint input, and
    even the full-pool ``None`` value must reject a pre-fix identity that
    omitted the key entirely.
    """

    builder = _load_builder_module()
    monkeypatch.setattr(builder, "US_SCHEMA", small_frame.schema)
    common = {
        "base_dataset_sha256": "base-sha",
        "policyengine_us_version": "1.2.3",
        "seed": 0,
        "target_period": builder.PERIOD,
        "target_registry_version": "registry-sha",
        "weeks_unemployed_source_sha256": "weeks-source-sha",
        "congressional_district_vintage_crosswalk_sha256": None,
        "ssi_take_up_assignment_sha256": "ssi-flags-sha",
    }
    full_pool = builder._target_frame_checkpoint_identity(
        **common, selection_identities_sha256=None
    )
    selected = builder._target_frame_checkpoint_identity(
        **common, selection_identities_sha256="cd" * 32
    )
    selected_other = builder._target_frame_checkpoint_identity(
        **common, selection_identities_sha256="ef" * 32
    )
    digest = builder._target_frame_checkpoint_digest
    assert digest(full_pool) != digest(selected)
    assert digest(selected) != digest(selected_other)

    prefix_shape = {
        key: value
        for key, value in full_pool.items()
        if key != "selection_identities_sha256"
    }
    path = tmp_path / "target_frame_checkpoint.h5"
    builder._write_target_frame_checkpoint(
        path,
        frame=small_frame,
        identity=prefix_shape,
        compilation={},
    )
    assert (
        builder._read_target_frame_checkpoint(
            path,
            identity=full_pool,
            target_specs=(),
        )
        is None
    )


def _table_1_4_diagnostic(builder, name: str, target: float, final: float):
    return SimpleNamespace(
        name=f"{name}@{builder.PERIOD}",
        target=target,
        initial_estimate=target,
        final_estimate=final,
        relative_error=(final - target) / target,
    )


def test_release_gate_failures_block_table_1_4_dollar_breaches() -> None:
    builder = _load_builder_module()
    breached = (
        # The live Build M defect pair (populace#462): +634.8% on the
        # capital-gain-distributions dollar row, -25.6% on net capital gains.
        _table_1_4_diagnostic(
            builder,
            "irs_soi.ty2023.table_1_4.all.capital_gain_distributions_amount",
            10_155_465_319.0,
            74_617_447_202.0,
        ),
        _table_1_4_diagnostic(
            builder,
            "irs_soi.ty2023.table_1_4.all.net_capital_gains_amount",
            1_270_864_366_489.0,
            945_431_772_792.0,
        ),
    )
    result = SimpleNamespace(
        skipped=(),
        diagnostics=_passing_critical_diagnostics(builder) + breached,
        initial_loss=10.0,
        final_loss=5.0,
    )

    failures = builder._release_gate_failures(result, {"dropped_target_names": []})

    assert len(failures) == 2
    assert all(
        failure.startswith("SOI Table 1.4 national dollar fit failed: ")
        for failure in failures
    )
    joined = "\n".join(failures)
    assert "capital_gain_distributions_amount@2024" in joined
    assert "net_capital_gains_amount@2024" in joined
    assert "6.3475" in joined


def test_release_gate_failures_block_table_1_4_row_outside_irs_prefix() -> None:
    builder = _load_builder_module()
    adversarial = _table_1_4_diagnostic(
        builder,
        "other.table_1_4.all.bad_amount",
        100.0,
        200.0,
    )
    result = SimpleNamespace(
        skipped=(),
        diagnostics=_passing_critical_diagnostics(builder) + (adversarial,),
        initial_loss=10.0,
        final_loss=5.0,
    )

    failures = builder._release_gate_failures(result, {"dropped_target_names": []})

    assert failures == [
        "SOI Table 1.4 national dollar fit failed: "
        "other.table_1_4.all.bad_amount@2024: relative_error=1 exceeds 0.25 "
        "for SOI Pub 1304 Table 1.4 national dollar rows "
        "(soi_table_1_4_national_dollar_rows); target=100.0, "
        "final_estimate=200.0."
    ]


def test_release_gate_failures_require_recorded_table_1_4_relative_error() -> None:
    builder = _load_builder_module()
    adversarial = SimpleNamespace(
        name=(f"irs_soi.ty2023.table_1_4.all.adversarial_amount@{builder.PERIOD}"),
        target=100.0,
        initial_estimate=100.0,
        final_estimate=100.0,
        relative_error=None,
    )
    result = SimpleNamespace(
        skipped=(),
        diagnostics=_passing_critical_diagnostics(builder) + (adversarial,),
        initial_loss=10.0,
        final_loss=5.0,
    )

    failures = builder._release_gate_failures(result, {"dropped_target_names": []})

    assert failures == [
        "SOI Table 1.4 national dollar fit failed: "
        "irs_soi.ty2023.table_1_4.all.adversarial_amount@2024: "
        "missing recorded relative_error; the publish contract requires a "
        "numeric value."
    ]


def test_release_gate_failures_ignore_table_1_4_returns_rows() -> None:
    builder = _load_builder_module()
    # A wildly-missed returns (count) row is outside the dollar blanket: the
    # live Build M estate_trust_net_loss_returns row landed at +495.9% and is
    # a distinct defect class, not this gate's scope.
    returns_row = _table_1_4_diagnostic(
        builder,
        "irs_soi.ty2023.table_1_4.all.estate_trust_net_loss_returns",
        36_592.0,
        218_052.0,
    )
    result = SimpleNamespace(
        skipped=(),
        diagnostics=_passing_critical_diagnostics(builder) + (returns_row,),
        initial_loss=10.0,
        final_loss=5.0,
    )

    assert builder._release_gate_failures(result, {"dropped_target_names": []}) == []


def test_release_gate_failures_require_a_table_1_4_dollar_surface() -> None:
    builder = _load_builder_module()
    without_table_1_4 = tuple(
        diagnostic
        for diagnostic in _passing_critical_diagnostics(builder)
        if ".table_1_4." not in diagnostic.name
    )
    result = SimpleNamespace(
        skipped=(),
        diagnostics=without_table_1_4,
        initial_loss=10.0,
        final_loss=5.0,
    )

    failures = builder._release_gate_failures(result, {"dropped_target_names": []})

    assert any("soi_table_1_4_national_dollar_rows" in failure for failure in failures)


def test_qrf_imputed_source_outputs_come_from_the_stage_manifest() -> None:
    builder = _load_builder_module()

    outputs = builder._qrf_imputed_source_outputs()

    assert "non_sch_d_capital_gains" in outputs
    assert "taxable_interest_income" in outputs
    assert len(outputs) >= 60
    # The capital_gain_distributions stage is a share split, not a QRF fit.
    assert "schedule_d_capital_gain_distributions" not in outputs


def _qrf_export_frame(builder, non_sch_d_values: np.ndarray) -> Frame:
    n = int(non_sch_d_values.size)
    ids = np.arange(1, n + 1, dtype="int64")
    taxable_interest = np.zeros(n)
    taxable_interest[: n // 2] = 1_000.0
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids,
            "person_spm_unit_id": ids,
            "person_family_id": ids,
            "person_marital_unit_id": ids,
            "non_sch_d_capital_gains": non_sch_d_values,
            "taxable_interest_income": taxable_interest,
        }
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame({"household_id": ids}),
            "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
            "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
            "family": pd.DataFrame({"family_id": ids}),
            "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
        },
        builder.US_SCHEMA,
        {
            "household": builder.Weights(
                values=np.ones(n),
                kind=WeightKind.DESIGN,
            )
        },
    )


def test_qrf_tail_concentration_gate_flags_the_build_m_point_mass() -> None:
    builder = _load_builder_module()
    # 500 carriers of 12,000 person records (4.2% — sparse); the top 100 carry
    # the repeated $594,484 donor ceiling, ~98% of the weighted mass.
    values = np.zeros(12_000)
    values[:100] = 594_484.0
    values[100:500] = 2_979.0

    gate, surface = builder._qrf_tail_concentration_gate(
        _qrf_export_frame(builder, values)
    )

    assert not gate.passed
    assert any("non_sch_d_capital_gains" in line for line in gate.failures)
    assert surface["checked_sparse_columns"] == ["non_sch_d_capital_gains"]
    assert "taxable_interest_income" in surface["dense_columns"]
    assert "short_term_capital_gains" in surface["absent_columns"]


def test_qrf_tail_concentration_gate_passes_dispersed_sparse_mass() -> None:
    builder = _load_builder_module()
    values = np.zeros(12_000)
    values[:500] = 2_979.0

    gate, _ = builder._qrf_tail_concentration_gate(_qrf_export_frame(builder, values))

    assert gate.passed


def test_allow_qrf_tail_concentration_flag_parses(monkeypatch) -> None:
    builder = _load_builder_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
        ],
    )
    assert not builder._parse_args().allow_qrf_tail_concentration

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
            "--allow-qrf-tail-concentration",
        ],
    )
    assert builder._parse_args().allow_qrf_tail_concentration


# --- SSI take-up delivered-weight prior basis + delivery gate (#507/#508) ---

_SSI_BAND_TARGETS = {
    "under_18": 1_001_922.0,
    "18_64": 3_905_779.0,
    "65_plus": 2_382_142.0,
}


def _ssi_prior_final_artifact_payload() -> dict:
    """A prior attempt's final us_ssi_take_up.json, schema 2 (Build N shape).

    Contract strings are frozen LITERALS on purpose: this fixture documents
    what Build N's certified artifact actually carries, so drift in the
    module constants cannot silently redefine what the loader accepts
    (populace#507 sol review finding 10).
    """

    bands = [
        ("under_18", 1_001_922.0, 177_582.0, 60_000.0),
        ("18_64", 3_905_779.0, 6_000_000.0, 2_500_000.0),
        ("65_plus", 2_382_142.0, 3_995_000.0, 900_000.0),
    ]
    return {
        "schema_version": 2,
        "classification": "release_diagnostics",
        "variable": "takes_up_ssi_if_eligible",
        "candidate_definition": "uncapped_ssi > 0 at 2024-12",
        "target_table": "ssa_ssi_federal_payment_recipients_by_age",
        "target_source": (
            "https://www.ssa.gov/policy/docs/statcomps/ssi_monthly/2024-12/table01.html"
        ),
        "target_period": "2024-12",
        "target_measure": "Total with—Federal payment",
        "age_bands": [
            {
                "age_band": key,
                "target": target,
                "candidate_capacity": capacity,
                "reporter_candidate_floor": floor,
            }
            for key, target, capacity, floor in bands
        ],
    }


def test_ssi_prior_weight_basis_loads_a_prior_final_artifact(tmp_path) -> None:
    import hashlib

    builder = _load_builder_module()
    path = tmp_path / "us_ssi_take_up.json"
    path.write_text(json.dumps(_ssi_prior_final_artifact_payload()))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    basis = builder._load_ssi_take_up_prior_weight_basis(
        path, targets=_SSI_BAND_TARGETS, expected_sha256=sha
    )

    assert basis.kind == "release_artifact"
    assert basis.source_schema_version == 2
    assert basis.source_sha256 == sha
    aged = basis.band("65_plus")
    assert aged.candidate_capacity == pytest.approx(3_995_000.0)
    assert aged.reporter_candidate_floor == pytest.approx(900_000.0)
    assert (
        builder._load_ssi_take_up_prior_weight_basis(
            None, targets=_SSI_BAND_TARGETS, expected_sha256=None
        )
        is None
    )


def test_ssi_prior_weight_basis_fails_fast_on_bad_artifacts(tmp_path) -> None:
    import hashlib

    builder = _load_builder_module()
    valid = tmp_path / "us_ssi_take_up.json"
    valid.write_text(json.dumps(_ssi_prior_final_artifact_payload()))
    valid_sha = hashlib.sha256(valid.read_bytes()).hexdigest()

    # The sha256 pin is the trust receipt (populace#507 sol review
    # finding 1): no pin, wrong pin, and a pin without a path all fail fast.
    with pytest.raises(RuntimeError, match="companion"):
        builder._load_ssi_take_up_prior_weight_basis(
            valid, targets=_SSI_BAND_TARGETS, expected_sha256=None
        )
    with pytest.raises(RuntimeError, match="not the pinned"):
        builder._load_ssi_take_up_prior_weight_basis(
            valid, targets=_SSI_BAND_TARGETS, expected_sha256="ab" * 32
        )
    with pytest.raises(RuntimeError, match="requires"):
        builder._load_ssi_take_up_prior_weight_basis(
            None, targets=_SSI_BAND_TARGETS, expected_sha256=valid_sha
        )

    with pytest.raises(RuntimeError, match="does not exist"):
        builder._load_ssi_take_up_prior_weight_basis(
            tmp_path / "missing.json",
            targets=_SSI_BAND_TARGETS,
            expected_sha256=valid_sha,
        )

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not json")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        builder._load_ssi_take_up_prior_weight_basis(
            invalid,
            targets=_SSI_BAND_TARGETS,
            expected_sha256=hashlib.sha256(invalid.read_bytes()).hexdigest(),
        )

    # A basis measured against a different SSA band target contract must be
    # refused — one coherent target system (populace#508).
    drifted_payload = _ssi_prior_final_artifact_payload()
    drifted_payload["age_bands"][2]["target"] = 2_000_000.0
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(drifted_payload))
    with pytest.raises(RuntimeError, match="target contract"):
        builder._load_ssi_take_up_prior_weight_basis(
            drifted,
            targets=_SSI_BAND_TARGETS,
            expected_sha256=hashlib.sha256(drifted.read_bytes()).hexdigest(),
        )


def test_ssi_prior_weight_basis_flag_defaults_to_none(monkeypatch, tmp_path) -> None:
    builder = _load_builder_module()
    base_argv = [
        "build_us_fiscal_refresh_release.py",
        "--ledger-facts",
        "facts.jsonl",
        "--out",
        "release",
    ]
    monkeypatch.setattr(sys, "argv", base_argv)
    assert builder._parse_args().ssi_take_up_prior_weight_basis is None

    basis_path = tmp_path / "us_ssi_take_up.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *base_argv,
            "--ssi-take-up-prior-weight-basis",
            str(basis_path),
            "--ssi-take-up-prior-weight-basis-sha256",
            "ab" * 32,
        ],
    )
    args = builder._parse_args()
    assert args.ssi_take_up_prior_weight_basis == basis_path
    assert args.ssi_take_up_prior_weight_basis_sha256 == "ab" * 32


def _ssi_delivery_diagnostics(selected: dict[str, float]) -> dict:
    return {
        "schema_version": 4,
        "measurement_phase": "release_final",
        "age_bands": [
            {
                "age_band": key,
                "target": _SSI_BAND_TARGETS[key],
                "selected_recipient_weight": selected[key],
            }
            for key in _SSI_BAND_TARGETS
        ],
    }


def test_enforce_ssi_delivery_returns_batch_failures_and_writes_the_basis(
    tmp_path,
) -> None:
    """A delivery miss returns batchable failures instead of raising.

    populace#547: the old in-place raise destroyed the failed run's
    calibration diagnostics and skipped every other terminal gate group.
    The failures now join the #437 batch; the basis artifact write — the
    retry remedy — is unchanged.
    """
    builder = _load_builder_module()
    # Build N's measured delivery: 65+ landed 0.98M against 2.38M — the
    # populace#507 collapse — while under-18 stays fenced (#453/#509).
    diagnostics = _ssi_delivery_diagnostics(
        {"under_18": 120_000.0, "18_64": 4_100_000.0, "65_plus": 984_000.0}
    )
    release_dir = tmp_path / "release"
    release_dir.mkdir()

    failures = builder._enforce_ssi_take_up_delivery(
        diagnostics,
        targets=_SSI_BAND_TARGETS,
        release_dir=release_dir,
        telemetry=None,
    )

    assert failures
    assert all(failure.startswith("SSI take-up deliver") for failure in failures)
    assert any("--ssi-take-up-prior-weight-basis" in failure for failure in failures)
    written_path = release_dir / "us_ssi_take_up.json"
    written = json.loads(written_path.read_text())
    assert written == diagnostics
    # The failure itself hands the operator BOTH halves of the retry remedy:
    # the artifact path and its sha256 pin (a failed attempt never reaches
    # the release manifest that would otherwise carry the hash).
    import hashlib

    written_sha = hashlib.sha256(written_path.read_bytes()).hexdigest()
    assert any(written_sha in failure for failure in failures)


def test_enforce_ssi_delivery_passes_in_tolerance_and_writes_nothing(
    tmp_path,
) -> None:
    builder = _load_builder_module()
    diagnostics = _ssi_delivery_diagnostics(
        {
            "under_18": 120_000.0,  # fenced: an 88% miss must not fail
            "18_64": 3_900_000.0,
            "65_plus": 2_350_000.0,
        }
    )
    release_dir = tmp_path / "release"
    release_dir.mkdir()

    failures = builder._enforce_ssi_take_up_delivery(
        diagnostics,
        targets=_SSI_BAND_TARGETS,
        release_dir=release_dir,
        telemetry=None,
    )

    assert failures == []
    assert not (release_dir / "us_ssi_take_up.json").exists()


def test_enforce_ssi_delivery_survives_unwritable_retry_artifact(
    tmp_path,
) -> None:
    """A nonfinite delivery fails the gate AND breaks the retry writer.

    The strict-JSON basis writer (allow_nan=False) raises on the very
    diagnostics that fail the gate; the reporting crash must not mask the
    gate failure or destroy the diagnostics artifact downstream
    (populace#547, confirm round 2 finding 1). The failure line tells the
    retry it must recompute delivery itself.
    """
    builder = _load_builder_module()
    diagnostics = _ssi_delivery_diagnostics(
        {"under_18": 120_000.0, "18_64": float("nan"), "65_plus": 984_000.0}
    )
    release_dir = tmp_path / "release"
    release_dir.mkdir()

    failures = builder._enforce_ssi_take_up_delivery(
        diagnostics,
        targets=_SSI_BAND_TARGETS,
        release_dir=release_dir,
        telemetry=None,
    )

    assert failures
    assert failures[0].startswith("SSI take-up delivery failed:")
    assert any("could NOT be written" in failure for failure in failures)
    # json.dumps runs before write_text, so no partial artifact exists.
    assert not (release_dir / "us_ssi_take_up.json").exists()


def test_final_medicaid_quarantines_on_ssi_law_violation() -> None:
    """A Bernoulli-law violation must quarantine, never evaluate (#547).

    pe-us Medicaid eligibility consumes the frozen SSI decisions
    (takes_up_ssi_if_eligible -> ssi -> is_ssi_recipient_for_medicaid ->
    medicaid_category), so evaluating on corrupted decisions would
    mis-measure rather than fail.
    """
    builder = _load_builder_module()

    def must_not_evaluate() -> dict:
        raise AssertionError("Medicaid must not be evaluated under quarantine")

    diagnostics, failures = builder._final_medicaid_diagnostics_or_quarantine(
        ssi_law_degraded=True,
        degraded=True,
        evaluate=must_not_evaluate,
    )

    assert diagnostics == {}
    assert len(failures) == 1
    assert "quarantined" in failures[0]
    assert "Bernoulli-law violation" in failures[0]


def test_final_medicaid_guard_records_crash_only_in_degraded_mode() -> None:
    builder = _load_builder_module()

    def boom() -> dict:
        raise RuntimeError("medicaid recompute exploded")

    diagnostics, failures = builder._final_medicaid_diagnostics_or_quarantine(
        ssi_law_degraded=False,
        degraded=True,
        evaluate=boom,
    )
    assert diagnostics == {}
    assert len(failures) == 1
    assert "medicaid recompute exploded" in failures[0]

    with pytest.raises(RuntimeError, match="medicaid recompute exploded"):
        builder._final_medicaid_diagnostics_or_quarantine(
            ssi_law_degraded=False,
            degraded=False,
            evaluate=boom,
        )


def test_final_medicaid_green_path_evaluates_normally() -> None:
    builder = _load_builder_module()

    diagnostics, failures = builder._final_medicaid_diagnostics_or_quarantine(
        ssi_law_degraded=False,
        degraded=False,
        evaluate=lambda: {"enrolled": 1},
    )

    assert diagnostics == {"enrolled": 1}
    assert failures == []


def test_reform_vector_cache_context_tracks_support_and_materializer() -> None:
    """The reform-vector whitelist carries support and materializer digests.

    Whether a JCT reform income-tax estimate can move with
    takes_up_ssi_if_eligible is an engine-graph question the build must not
    answer by assumption, while two selected supports can share positional
    SSI flag bytes. The assignment, selection, and complete target-frame
    materializer digests therefore invalidate reform vectors independently."""

    builder = _load_builder_module()
    base = {
        "base_dataset_sha256": "b",
        "weeks_unemployed_source_sha256": "w",
        "policyengine_us_version": "1",
        "target_period": 2024,
        "congressional_district_vintage_crosswalk_sha256": None,
        "build_commit": "irrelevant-to-reform-vectors",
        "ssi_take_up_assignment_sha256": "digest-a",
        "selection_identities_sha256": None,
        "target_frame_materializer_identity_sha256": "materializer-a",
    }
    changed_assignment = {**base, "ssi_take_up_assignment_sha256": "digest-b"}
    selected = {**base, "selection_identities_sha256": "cd" * 32}
    selected_other = {**base, "selection_identities_sha256": "ef" * 32}
    changed_materializer = {
        **base,
        "target_frame_materializer_identity_sha256": "materializer-b",
    }
    projected = builder._reform_vector_cache_context(base)
    assert builder._reform_vector_cache_context(
        base
    ) != builder._reform_vector_cache_context(changed_assignment)
    assert projected != builder._reform_vector_cache_context(selected)
    assert builder._reform_vector_cache_context(
        selected
    ) != builder._reform_vector_cache_context(selected_other)
    assert projected != builder._reform_vector_cache_context(changed_materializer)
    assert "selection_identities_sha256" in builder.REFORM_VECTOR_CACHE_CONTEXT_KEYS
    assert projected["selection_identities_sha256"] is None
    assert (
        "target_frame_materializer_identity_sha256"
        in builder.REFORM_VECTOR_CACHE_CONTEXT_KEYS
    )
    assert projected["target_frame_materializer_identity_sha256"] == "materializer-a"
    assert "build_commit" not in projected


def test_ssi_assignment_digest_tracks_flags_priors_and_basis(small_frame) -> None:
    """Any change to the frozen assignment must invalidate checkpoint/cache."""

    from populace.build.us_runtime.ssi_take_up import (
        SSITakeUpBandPriorBasis,
        SSITakeUpPriorBasis,
    )

    builder = _load_builder_module()

    def _frame_with_flags(flags):
        tables = {
            entity: small_frame.table(entity).copy() for entity in small_frame.entities
        }
        tables["person"]["takes_up_ssi_if_eligible"] = np.asarray(flags, dtype=bool)
        return Frame(
            tables,
            small_frame.schema,
            {
                entity: small_frame.weights_for(entity)
                for entity in small_frame.weighted_entities
            },
        )

    priors = {"under_18": 0.1, "18_64": 0.2, "65_plus": 0.3}
    basis = SSITakeUpPriorBasis(
        kind="current_frame",
        bands=tuple(
            SSITakeUpBandPriorBasis(
                key=key, candidate_capacity=100.0, reporter_candidate_floor=10.0
            )
            for key in priors
        ),
    )
    baseline = builder._ssi_take_up_assignment_digest(
        _frame_with_flags([True, False, True, False]),
        assignment_priors=priors,
        prior_basis=basis,
    )
    assert baseline == builder._ssi_take_up_assignment_digest(
        _frame_with_flags([True, False, True, False]),
        assignment_priors=priors,
        prior_basis=basis,
    )
    assert baseline != builder._ssi_take_up_assignment_digest(
        _frame_with_flags([True, True, True, False]),
        assignment_priors=priors,
        prior_basis=basis,
    )
    assert baseline != builder._ssi_take_up_assignment_digest(
        _frame_with_flags([True, False, True, False]),
        assignment_priors={**priors, "65_plus": 0.9},
        prior_basis=basis,
    )
    assert baseline != builder._ssi_take_up_assignment_digest(
        _frame_with_flags([True, False, True, False]),
        assignment_priors=priors,
        prior_basis=SSITakeUpPriorBasis(
            kind="release_artifact",
            bands=basis.bands,
            source_sha256="cd" * 32,
            source_schema_version=2,
        ),
    )


def test_calibration_diagnostics_schema_lockstep() -> None:
    """The writer (populace-calibrate) and the publish contract (populace-data)
    pin the same diagnostics schema version. They cannot share a constant —
    populace-data must not import populace-calibrate — so drift fails here,
    in the one suite that imports both (the #494 cross-package break class:
    calibrate moved to schema 5 while the contract still rejected != 4)."""
    from populace.calibrate.diagnostics import (
        CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION as WRITER_SCHEMA_VERSION,
    )
    from populace.data.contract import (
        CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION as CONTRACT_SCHEMA_VERSION,
    )

    assert WRITER_SCHEMA_VERSION == CONTRACT_SCHEMA_VERSION
