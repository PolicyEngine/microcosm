"""Small-fixture tests for the terminal US multispine pool build tool."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

import populace.build.us_runtime.acs_transfer as acs_transfer_module
from populace.build.gates import GateReport, GateResult
from populace.build.us_runtime.acs_transfer import transfer_acs_inputs
from populace.build.us_runtime.multispine_pool import PoolStageOutput
from populace.build.us_runtime.operator_boundary import (
    PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
)
from populace.build.us_runtime.puf_support import (
    PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID,
)
from populace.build.us_runtime.support_provenance import (
    support_channel_column,
    support_clone_index_column,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


@pytest.fixture(scope="module")
def pool_tool() -> ModuleType:
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_multispine_pool.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_multispine_pool_fixture",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_frame(*, measured_offset: float = 0.0) -> Frame:
    ids = np.asarray([1, 2], dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids,
            "person_spm_unit_id": ids,
            "person_family_id": ids,
            "person_marital_unit_id": ids,
            "A_AGE": np.asarray([30.0, 50.0]),
            "A_SEX": np.asarray([1, 2], dtype=np.int64),
            "PERIDNUM": np.asarray(["1", "2"], dtype=object),
            "source_year": np.asarray([2024, 2024], dtype=np.int64),
            "measured": np.asarray([1.0, 2.0]) + measured_offset,
        }
    )
    tables = {
        "person": person,
        **{
            entity: pd.DataFrame({f"{entity}_id": ids})
            for entity in US_SCHEMA.group_entities
        },
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([2.0, 2.0]),
                WeightKind.DESIGN,
            )
        },
        pd.Series(["fixture", "fixture"], dtype=object),
    )


def _replace_person(
    frame: Frame,
    person: pd.DataFrame,
    *,
    preserve_metadata: bool = True,
) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata if preserve_metadata else None,
    )


class _MeanQRF:
    def __init__(self, *, n_estimators: int, seed: int) -> None:
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(
        self,
        frame: Frame,
        predictors: list[str],
        targets: list[str],
        *,
        weights: str,
    ) -> _MeanFitted:
        assert (
            weights == frame.resolve_weights(frame.column_entity(targets[0])).kind.value
        )
        table = frame.table(frame.column_entity(targets[0]))
        return _MeanFitted(
            {target: float(table[target].mean()) for target in targets},
            weights,
        )


class _MeanFitted:
    def __init__(self, means: dict[str, float], weight_kind: str) -> None:
        self.means = means
        self.weight_kind = weight_kind

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                target: np.full(len(frame), mean, dtype=np.float64)
                for target, mean in self.means.items()
            },
            index=frame.index,
        )


def _transfer_source_frame(targets: list[float]) -> Frame:
    frame = _source_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"]["fixture_transfer"] = np.asarray(targets, dtype=np.float64)
    tables["household"]["state_fips"] = np.asarray([6, 36], dtype=np.int64)
    return Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )


def _red_pool_result(pool_tool: ModuleType, tmp_path: Path):
    order: list[str] = []

    def stage(
        name: str,
        transform: Callable[[pd.DataFrame], None],
    ) -> Callable[[Frame], PoolStageOutput]:
        def apply(frame: Frame) -> PoolStageOutput:
            order.append(name)
            person = frame.table("person").copy()
            assert set(person[support_clone_index_column("person")].astype(int)) == {
                0,
                1,
            }
            assert set(person[support_channel_column("person")]) == {
                "asec",
                "acs",
            }
            transform(person)
            return PoolStageOutput(
                _replace_person(frame, person),
                {"fixture_stage": name},
            )

        return apply

    def transfer(person: pd.DataFrame) -> None:
        person["fixture_transfer"] = person["measured"]

    def derive(person: pd.DataFrame) -> None:
        person["fixture_derived"] = person["fixture_transfer"] + 1.0

    def seed(person: pd.DataFrame) -> None:
        person["fixture_seed"] = person["fixture_derived"] > 0.0

    def simulate(person: pd.DataFrame) -> None:
        channels = person[support_channel_column("person")]
        person["ssi"] = np.where(channels.eq("asec"), 1.0, 100.0)

    result = pool_tool.build_multispine_pool(
        _source_frame(),
        _source_frame(measured_offset=99.0),
        puf_donor=pd.DataFrame(),
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
        impute=stage("impute", transfer),
        derive=stage("derive", derive),
        seed=stage("seed", seed),
        simulate=stage("simulate", simulate),
    )

    assert order == ["impute", "derive", "seed", "simulate"]
    assert not result.agreement_gate.passed
    assert not result.simulation_ready
    assert result.agreement_gate.name == "us_spine_agreement"
    assert result.agreement_gate.details["tolerances"] == {
        "incidence_ratio_bounds": [0.8, 1.25],
        "max_quantile_envelope_distance": 0.25,
        "max_categorical_total_variation_distance": 0.25,
    }
    assert "ssi" not in result.frame.table("person")
    return result


def _output_context(
    pool_tool: ModuleType,
    tmp_path: Path,
):
    result = _red_pool_result(pool_tool, tmp_path)
    outputs = pool_tool._output_paths(tmp_path / "pool.h5")
    source_manifest = pool_tool.load_acs_source_manifest()
    verified_inputs = {}
    for index, role in enumerate(
        (
            "asec_raw_stage",
            "acs_household",
            "acs_person",
            "acs_rent_donor",
            "processed_puf",
            "puf_source_year",
        ),
        start=1,
    ):
        path = tmp_path / f"{role}.fixture"
        path.write_bytes(f"input-{index}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        verified_inputs[role] = pool_tool._VerifiedInput(
            role=role,
            path=path,
            expected_sha256=digest,
            actual_sha256=digest,
            size_bytes=path.stat().st_size,
        )
    loaded = pool_tool._LoadedInputs(
        asec=_source_frame(),
        acs=_source_frame(measured_offset=99.0),
        acs_rent_donor=pd.DataFrame({"fixture": [1]}),
        puf_donor=pd.DataFrame({"RECID": [1]}),
        asec_raw_stage_checkpoint={"artifact": "fixture-raw-stage"},
        acs_build={"artifact": "fixture-unit-frame"},
        acs_native_inputs={"person": {"age": {"source": "fixture"}}},
        puf_donor_build={"artifact": "fixture-donor"},
    )
    return result, outputs, verified_inputs, source_manifest, loaded


def _seed_stale_green_outputs(outputs) -> None:
    outputs.pool_h5.write_bytes(b"stale green h5")
    outputs.agreement_diagnostics.write_text(
        json.dumps(
            {
                "simulation_ready": True,
                "publication_run_id": "stale-run",
            }
        ),
        encoding="utf-8",
    )
    outputs.manifest.write_text(
        json.dumps(
            {
                "status": "simulation_ready",
                "simulation_ready": True,
                "publication_run_id": "stale-run",
            }
        ),
        encoding="utf-8",
    )


def _assert_publication_tombstone(
    pool_tool: ModuleType,
    outputs,
    *,
    publication_run_id: str,
) -> None:
    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert manifest == {
        "agreement_diagnostics": {
            "path": str(outputs.agreement_diagnostics.resolve()),
            "publication_run_id": publication_run_id,
        },
        "artifact_kind": "populace_us_multispine_pool_manifest",
        "message": "publication in progress",
        "pool_h5": {
            "artifact_kind": "populace_us_multispine_input_pool",
            "path": str(outputs.pool_h5.resolve()),
            "publication_run_id": publication_run_id,
        },
        "publication_run_id": publication_run_id,
        "schema_version": pool_tool.POOL_MANIFEST_SCHEMA_VERSION,
        "simulation_ready": False,
        "status": "publication_in_progress",
    }
    with pytest.raises(ValueError, match="not simulation-ready"):
        pool_tool.load_simulation_ready_us_multispine_pool_manifest(
            outputs.manifest
        )


def test_parser_exposes_only_six_pinned_inputs_and_out(
    pool_tool: ModuleType,
) -> None:
    parser = pool_tool._parser()
    actions = {
        action.dest: action for action in parser._actions if action.dest != "help"
    }
    pairs = (
        ("asec_raw_stage_h5", "asec_raw_stage_h5_sha256"),
        ("acs_household_zip", "acs_household_zip_sha256"),
        ("acs_person_zip", "acs_person_zip_sha256"),
        ("acs_rent_h5", "acs_rent_h5_sha256"),
        ("puf_h5", "puf_h5_sha256"),
        ("puf_source_year_csv", "puf_source_year_csv_sha256"),
    )
    expected_destinations = {destination for pair in pairs for destination in pair} | {
        "out"
    }

    assert set(actions) == expected_destinations
    assert all(action.required for action in actions.values())
    assert actions["out"].option_strings == ["--out"]
    for path_destination, sha_destination in pairs:
        assert actions[path_destination].type is Path
        assert actions[sha_destination].type is pool_tool._sha256_argument
        assert len(actions[path_destination].option_strings) == 1
        assert len(actions[sha_destination].option_strings) == 1

    option_names = {
        option for action in actions.values() for option in action.option_strings
    }
    assert not any(
        forbidden in option
        for option in option_names
        for forbidden in ("tolerance", "target", "per-target")
    )


def test_pool_tool_structurally_accepts_only_the_raw_stage_loader(
    pool_tool: ModuleType,
) -> None:
    source = inspect.getsource(pool_tool)
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
    }

    assert "load_asec_raw_stage_checkpoint" in imported
    assert "load_asec_raw_stage_checkpoint" in called
    assert "load_asec_pre_clone_checkpoint" not in imported
    assert "load_asec_pre_clone_checkpoint" not in called
    assert "pre_clone_enrichment" not in source


def test_pool_imputation_wires_full_source_chain_after_primary_and_tail(
    pool_tool: ModuleType,
) -> None:
    source = inspect.getsource(pool_tool._impute_pool)
    tree = ast.parse(source)
    expected = (
        "prepare_multispine_puf_predictors",
        "_initialize_or_resume_primary_qrf",
        "run_primary_puf_qrf_chain",
        "finalize_primary_puf_qrf_chain",
        "transfer_puf_capital_gains_tail",
        "complete_multispine_source_inputs",
        "pool_transfer_target_families",
        "transfer_acs_inputs",
    )
    calls = sorted(
        (
            node.lineno,
            node.func.id,
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected
    )

    assert tuple(name for _line, name in calls) == expected


def test_direct_pool_fixtures_are_operator_free_before_assembly() -> None:
    frame = _source_frame()
    for entities in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES.values():
        for entity, columns in entities.items():
            assert set(frame.table(entity)).isdisjoint(columns)


@pytest.mark.parametrize(
    ("family", "entity", "column"),
    [
        (family, entity, sorted(columns)[0])
        for family, entities in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES.items()
        for entity, columns in entities.items()
    ],
)
def test_each_operator_output_family_is_rejected_before_assembly(
    pool_tool: ModuleType,
    family: str,
    entity: str,
    column: str,
) -> None:
    frame = _source_frame()
    tables = {name: frame.table(name).copy() for name in frame.entities}
    tables[entity][column] = 0.0
    contaminated = Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )

    with pytest.raises(
        ValueError,
        match=rf"fixture {family}.*{column}|fixture.*{column}",
    ):
        pool_tool.assert_operator_free_source_frame(
            contaminated,
            label=f"fixture {family}",
        )


def test_sha_mismatch_refuses_before_loading_or_writing(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_paths = {
        "asec-raw-stage-h5": tmp_path / "asec.h5",
        "acs-household-zip": tmp_path / "household.zip",
        "acs-person-zip": tmp_path / "person.zip",
        "acs-rent-h5": tmp_path / "rent.h5",
        "puf-h5": tmp_path / "puf.h5",
        "puf-source-year-csv": tmp_path / "puf.csv",
    }
    for path in source_paths.values():
        path.write_bytes(b"fixture input")

    called = {"load": False, "write": False}

    def unexpected_load(*_args, **_kwargs):
        called["load"] = True
        raise AssertionError("SHA mismatch must precede source-frame loading.")

    def unexpected_write(*_args, **_kwargs):
        called["write"] = True
        raise AssertionError("SHA mismatch must precede output writing.")

    monkeypatch.setattr(pool_tool, "_load_inputs", unexpected_load)
    monkeypatch.setattr(pool_tool, "_write_outputs", unexpected_write)
    output = tmp_path / "pool.h5"
    argv: list[str] = []
    for option, path in source_paths.items():
        argv.extend([f"--{option}", str(path)])
        digest = (
            pool_tool.ACS_2022_RENT_ARTIFACT_SHA256
            if option == "acs-rent-h5"
            else "0" * 64
        )
        argv.extend([f"--{option}-sha256", digest])
    argv.extend(["--out", str(output)])

    with pytest.raises(ValueError, match="ASEC raw-stage.*SHA-256 mismatch"):
        pool_tool.main(argv)

    assert called == {"load": False, "write": False}
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
    assert not output.with_suffix(".agreement.json").exists()


def test_primary_qrf_resume_refuses_a_changed_input_binding(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "primary-qrf"
    checkpoint_dir.mkdir()
    (checkpoint_dir / pool_tool.PRIMARY_QRF_MANIFEST_FILENAME).write_text(
        "{}\n",
        encoding="utf-8",
    )
    original = {
        "artifact_kind": "fixture_binding",
        "schema_version": 1,
        "inputs": {"processed_puf": {"sha256": "a" * 64}},
    }
    pool_tool._atomic_write_json(
        checkpoint_dir / pool_tool._PRIMARY_QRF_INPUT_BINDING_FILENAME,
        original,
    )
    changed = {
        **original,
        "inputs": {"processed_puf": {"sha256": "b" * 64}},
    }

    with pytest.raises(ValueError, match="refusing to reuse stale predictions"):
        pool_tool._initialize_or_resume_primary_qrf(
            _source_frame(),
            pd.DataFrame(),
            checkpoint_dir,
            input_binding=changed,
        )


def test_synthetic_two_spine_path_reaches_fixed_red_terminal_gate(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    result = _red_pool_result(pool_tool, tmp_path)

    assert result.assembly_receipt["channels"] == ["asec", "acs"]
    assert result.assembly_receipt["native_row_counts"]["person"] == {
        "asec": 2,
        "acs": 2,
    }
    assert result.provenance_counts["person"] == {
        "rows": 8,
        "by_source_channel": {"asec": 4, "acs": 4},
        "by_clone_index": {"0": 4, "1": 4},
        "by_source_channel_and_clone_index": {
            "asec": {"0": 2, "1": 2},
            "acs": {"0": 2, "1": 2},
        },
    }
    assert result.stage_receipts == {
        stage: {"fixture_stage": stage}
        for stage in ("impute", "derive", "seed", "simulate")
    }
    assert any(
        "person/simulated_output/ssi/acs_vs_asec" in failure
        for failure in result.agreement_gate.failures
    )


def test_wired_path_uses_real_raw_preserving_transfer_before_gate(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)
    transfer_receipts = []

    def impute(frame: Frame) -> PoolStageOutput:
        person = frame.table("person").copy()
        person["age"] = pd.to_numeric(person["A_AGE"], errors="raise")
        person["is_female"] = (
            pd.to_numeric(person["A_SEX"], errors="raise") == 2
        )
        prepared = _replace_person(frame, person)
        transferred = transfer_acs_inputs(
            prepared,
            prepared,
            target_families={"person": {"fixture": ("fixture_transfer",)}},
            seed=0,
            n_estimators=3,
        )
        transfer_receipts.extend(transferred.imputed_inputs)
        return PoolStageOutput(
            transferred.frame,
            {"fit_records": list(transferred.fit_records)},
        )

    def no_op(frame: Frame) -> PoolStageOutput:
        return PoolStageOutput(frame)

    def simulate(frame: Frame) -> PoolStageOutput:
        person = frame.table("person").copy()
        person["ssi"] = person["fixture_transfer"]
        return PoolStageOutput(_replace_person(frame, person))

    result = pool_tool.build_multispine_pool(
        _transfer_source_frame([10.0, 20.0]),
        _transfer_source_frame([np.nan, np.nan]),
        puf_donor=pd.DataFrame(),
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
        impute=impute,
        derive=no_op,
        seed=no_op,
        simulate=simulate,
    )

    person = result.frame.table("person")
    channels = person[support_channel_column("person")]
    assert sorted(person.loc[channels.eq("asec"), "fixture_transfer"]) == [
        10.0,
        10.0,
        20.0,
        20.0,
    ]
    assert person.loc[channels.eq("acs"), "fixture_transfer"].tolist() == [
        15.0,
        15.0,
        15.0,
        15.0,
    ]
    assert sum(item.imputed_recipient_rows for item in transfer_receipts) == 4
    assert not result.agreement_gate.passed
    assert result.agreement_gate.name == "us_spine_agreement"
    assert "ssi" not in person


def test_red_outputs_preserve_receipts_and_exclude_simulation_output(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )

    pool_tool._write_outputs(
        result,
        outputs=outputs,
        verified_inputs=verified_inputs,
        acs_source_manifest=source_manifest,
        loaded=loaded,
    )

    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    diagnostics = json.loads(outputs.agreement_diagnostics.read_text(encoding="utf-8"))
    expected_gate = GateReport((result.agreement_gate,)).to_manifest()

    assert manifest["status"] == "agreement_failed"
    assert manifest["simulation_ready"] is False
    assert manifest["calibration_applied"] is False
    assert manifest["calibration"]["applied"] is False
    assert manifest["assembly_receipt"] == result.assembly_receipt
    assert manifest["provenance_counts"] == result.provenance_counts
    assert manifest["stage_receipts"] == result.stage_receipts
    assert manifest["agreement_gate"] == expected_gate
    assert diagnostics["agreement_gate"] == expected_gate
    assert diagnostics["simulation_ready"] is False
    assert diagnostics["publication_run_id"] == manifest["publication_run_id"]
    assert manifest["provenance_pins"] == {
        role: pin.to_manifest() for role, pin in verified_inputs.items()
    }
    assert manifest["pool_h5"]["formula_outputs_persisted"] is False
    assert manifest["pool_h5"]["input_only"] is True
    assert (
        manifest["pool_h5"]["publication_run_id"]
        == manifest["publication_run_id"]
    )
    assert manifest["pool_h5"]["sha256"] == hashlib.sha256(
        outputs.pool_h5.read_bytes()
    ).hexdigest()

    with pd.HDFStore(outputs.pool_h5, mode="r") as store:
        assert "ssi" not in store["person"].columns
        metadata = json.loads(str(store["_populace_staging_metadata"].iloc[0]))
    assert metadata["publication_run_id"] == manifest["publication_run_id"]


def test_ready_reader_binds_manifest_h5_and_diagnostics_to_one_run(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    ready = replace(
        result,
        agreement_gate=GateResult("us_spine_agreement", True),
    )
    pool_tool._write_outputs(
        ready,
        outputs=outputs,
        verified_inputs=verified_inputs,
        acs_source_manifest=source_manifest,
        loaded=loaded,
    )

    manifest = pool_tool.load_simulation_ready_us_multispine_pool_manifest(
        outputs.manifest
    )

    assert manifest["simulation_ready"] is True
    assert (
        manifest["agreement_diagnostics"]["publication_run_id"]
        == manifest["publication_run_id"]
    )


def test_ready_reader_rejects_manifest_h5_run_id_mismatch(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    ready = replace(
        result,
        agreement_gate=GateResult("us_spine_agreement", True),
    )
    pool_tool._write_outputs(
        ready,
        outputs=outputs,
        verified_inputs=verified_inputs,
        acs_source_manifest=source_manifest,
        loaded=loaded,
    )
    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    manifest["publication_run_id"] = "substituted-run"
    manifest["pool_h5"]["publication_run_id"] = "substituted-run"
    outputs.manifest.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="H5.*run ID does not match"):
        pool_tool.load_simulation_ready_us_multispine_pool_manifest(
            outputs.manifest
        )


def test_h5_publication_failure_invalidates_stale_green_manifest(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    _seed_stale_green_outputs(outputs)
    publication_run_id = "h5-failure-run"
    monkeypatch.setattr(
        pool_tool,
        "_new_publication_run_id",
        lambda: publication_run_id,
    )

    def fail_h5(*_args, **_kwargs) -> None:
        _assert_publication_tombstone(
            pool_tool,
            outputs,
            publication_run_id=publication_run_id,
        )
        raise RuntimeError("injected H5 publication failure")

    monkeypatch.setattr(pool_tool, "write_nullable_us_h5", fail_h5)

    with pytest.raises(RuntimeError, match="injected H5 publication failure"):
        pool_tool._write_outputs(
            result,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            loaded=loaded,
        )

    _assert_publication_tombstone(
        pool_tool,
        outputs,
        publication_run_id=publication_run_id,
    )
    assert outputs.pool_h5.read_bytes() == b"stale green h5"


def test_diagnostics_publication_failure_keeps_pool_not_ready(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    _seed_stale_green_outputs(outputs)
    publication_run_id = "diagnostics-failure-run"
    monkeypatch.setattr(
        pool_tool,
        "_new_publication_run_id",
        lambda: publication_run_id,
    )
    temporary_diagnostics = pool_tool._publication_temporary_path(
        outputs.agreement_diagnostics,
        publication_run_id=publication_run_id,
    )
    atomic_write_json = pool_tool._atomic_write_json

    def fail_diagnostics(path, payload) -> None:
        if Path(path) == temporary_diagnostics:
            raise RuntimeError("injected diagnostics publication failure")
        atomic_write_json(path, payload)

    monkeypatch.setattr(pool_tool, "_atomic_write_json", fail_diagnostics)

    with pytest.raises(
        RuntimeError,
        match="injected diagnostics publication failure",
    ):
        pool_tool._write_outputs(
            result,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            loaded=loaded,
        )

    _assert_publication_tombstone(
        pool_tool,
        outputs,
        publication_run_id=publication_run_id,
    )
    assert outputs.pool_h5.read_bytes() == b"stale green h5"
    assert not pool_tool._publication_temporary_path(
        outputs.pool_h5,
        publication_run_id=publication_run_id,
    ).exists()


def test_final_manifest_failure_leaves_tombstone_as_readiness_authority(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    _seed_stale_green_outputs(outputs)
    publication_run_id = "manifest-failure-run"
    monkeypatch.setattr(
        pool_tool,
        "_new_publication_run_id",
        lambda: publication_run_id,
    )
    atomic_write_json = pool_tool._atomic_write_json

    def fail_final_manifest(path, payload) -> None:
        if (
            Path(path) == outputs.manifest
            and payload["status"] != "publication_in_progress"
        ):
            raise RuntimeError("injected final manifest publication failure")
        atomic_write_json(path, payload)

    monkeypatch.setattr(pool_tool, "_atomic_write_json", fail_final_manifest)

    with pytest.raises(
        RuntimeError,
        match="injected final manifest publication failure",
    ):
        pool_tool._write_outputs(
            result,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            loaded=loaded,
        )

    _assert_publication_tombstone(
        pool_tool,
        outputs,
        publication_run_id=publication_run_id,
    )
    diagnostics = json.loads(
        outputs.agreement_diagnostics.read_text(encoding="utf-8")
    )
    assert diagnostics["publication_run_id"] == publication_run_id
    with pd.HDFStore(outputs.pool_h5, mode="r") as store:
        metadata = json.loads(str(store["_populace_staging_metadata"].iloc[0]))
    assert metadata["publication_run_id"] == publication_run_id


def test_clone_safe_id_error_surfaces_unchanged_through_tool(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    asec = _source_frame()
    tables = {entity: asec.table(entity).copy() for entity in asec.entities}
    tables["person"].loc[0, "person_id"] = PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID + 1
    invalid = Frame(
        tables,
        asec.schema,
        {"household": asec.weights_for("household")},
        asec.strata,
    )

    def unreachable(_frame: Frame) -> PoolStageOutput:
        raise AssertionError("Assembly errors must precede every pool stage.")

    with pytest.raises(ValueError, match="Spine 'asec'.*clone-safe bound"):
        pool_tool.build_multispine_pool(
            invalid,
            _source_frame(),
            puf_donor=pd.DataFrame(),
            primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
            impute=unreachable,
            derive=unreachable,
            seed=unreachable,
            simulate=unreachable,
        )


def test_assembly_receipt_loss_surfaces_unchanged_through_tool(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    def no_op(frame: Frame) -> PoolStageOutput:
        return PoolStageOutput(frame)

    def drop_receipt(frame: Frame) -> PoolStageOutput:
        return PoolStageOutput(
            _replace_person(
                frame,
                frame.table("person").copy(),
                preserve_metadata=False,
            )
        )

    with pytest.raises(
        ValueError,
        match="multispine pool derive output:.*no assembly manifest",
    ):
        pool_tool.build_multispine_pool(
            _source_frame(),
            _source_frame(),
            puf_donor=pd.DataFrame(),
            primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
            impute=no_op,
            derive=drop_receipt,
            seed=no_op,
            simulate=no_op,
        )
