"""Small-fixture tests for the terminal US multispine pool build tool."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from populace.build.gates import GateReport
from populace.build.us_runtime.multispine_pool import PoolStageOutput
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
            "age": np.asarray([30.0, 50.0]),
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
    }
    assert "ssi" not in result.frame.table("person")
    return result


def test_parser_exposes_only_five_pinned_inputs_and_out(
    pool_tool: ModuleType,
) -> None:
    parser = pool_tool._parser()
    actions = {
        action.dest: action for action in parser._actions if action.dest != "help"
    }
    pairs = (
        ("asec_pre_clone_h5", "asec_pre_clone_h5_sha256"),
        ("acs_household_zip", "acs_household_zip_sha256"),
        ("acs_person_zip", "acs_person_zip_sha256"),
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


def test_sha_mismatch_refuses_before_loading_or_writing(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_paths = {
        "asec-pre-clone-h5": tmp_path / "asec.h5",
        "acs-household-zip": tmp_path / "household.zip",
        "acs-person-zip": tmp_path / "person.zip",
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
        argv.extend([f"--{option}-sha256", "0" * 64])
    argv.extend(["--out", str(output)])

    with pytest.raises(ValueError, match="ASEC pre-clone.*SHA-256 mismatch"):
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


def test_red_outputs_preserve_receipts_and_exclude_simulation_output(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    result = _red_pool_result(pool_tool, tmp_path)
    outputs = pool_tool._output_paths(tmp_path / "pool.h5")
    source_manifest = pool_tool.load_acs_source_manifest()

    verified_inputs = {}
    for index, role in enumerate(
        (
            "asec_pre_clone",
            "acs_household",
            "acs_person",
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
        puf_donor=pd.DataFrame({"RECID": [1]}),
        asec_checkpoint={"artifact": "fixture-pre-clone"},
        acs_build={"artifact": "fixture-unit-frame"},
        acs_native_inputs={"person": {"age": {"source": "fixture"}}},
        puf_donor_build={"artifact": "fixture-donor"},
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
    assert manifest["provenance_pins"] == {
        role: pin.to_manifest() for role, pin in verified_inputs.items()
    }
    assert manifest["pool_h5"]["formula_outputs_persisted"] is False
    assert manifest["pool_h5"]["input_only"] is True

    with pd.HDFStore(outputs.pool_h5, mode="r") as store:
        assert "ssi" not in store["person"].columns


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
