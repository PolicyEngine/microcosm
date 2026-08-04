"""Small-fixture tests for the terminal US multispine pool build tool."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import stat
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import populace.build.us_runtime.acs_transfer as acs_transfer_module
from populace.build.gates import GateReport, GateResult
from populace.build.us_runtime.acs_transfer import transfer_acs_inputs
from populace.build.us_runtime.acs_transfer_bank import (
    ACS_TRANSFER_TARGET_BANK_MATERIALIZER_VERSION,
)
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
from populace.build.us_runtime.take_up_contract import (
    load_take_up_contract,
    take_up_contract_identity,
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


def _checkpoint_fixture_store(
    pool_tool: ModuleType,
    root: Path,
    *,
    changed_role: str | None = None,
):
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
        path = root.parent / f"{role}.checkpoint-fixture"
        if not path.exists():
            path.write_bytes(f"checkpoint-input-{index}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        actual = "f" * 64 if role == changed_role else digest
        verified_inputs[role] = pool_tool._VerifiedInput(
            role=role,
            path=path,
            expected_sha256=actual,
            actual_sha256=actual,
            size_bytes=path.stat().st_size,
        )
    base_identity = pool_tool._pool_checkpoint_base_identity(
        verified_inputs,
        policyengine_us_version="fixture-engine-1",
    )
    return pool_tool._PoolStageCheckpointStore(
        root,
        base_identity=base_identity,
    )


def _checkpoint_fixture_input_receipts() -> dict[str, object]:
    return {
        "asec_raw_stage_checkpoint": {"artifact": "fixture-raw-stage"},
        "acs_pums_build": {"artifact": "fixture-unit-frame"},
        "acs_native_inputs": {"person": {"age": {"source": "fixture"}}},
        "puf_donor": {
            "rows": 0,
            "columns": [],
            "build_receipt": {"artifact": "fixture-donor"},
        },
    }


def _run_checkpoint_fixture(
    pool_tool: ModuleType,
    tmp_path: Path,
    *,
    store,
    resume=None,
    target_bank_receipt: Mapping[str, object] | None = None,
):
    order: list[str] = []

    def stage(
        name: str,
        transform: Callable[[pd.DataFrame], None],
    ) -> Callable[[Frame], PoolStageOutput]:
        def apply(frame: Frame) -> PoolStageOutput:
            order.append(name)
            person = frame.table("person").copy()
            transform(person)
            receipt: dict[str, object] = {"fixture_stage": name}
            if name == "impute" and target_bank_receipt is not None:
                receipt = {
                    "source_operator_chain": {"post_primary_completion": {}},
                    "primary_puf_qrf": {"fixture_manifest": True},
                    "puf_capital_gains_tail_transfer": {"fixture": True},
                    "acs_qrf_transfer": {
                        "target_families": {"person": {"fixture": ["target"]}},
                        "n_estimators": 100,
                        "max_targets_per_fit": 8,
                        "resolved_donor_channel": "puf_tax_detail",
                        "imputed_inputs": [],
                        "fit_records": [],
                        "deferred_inputs": [],
                        "target_bank": dict(target_bank_receipt),
                    },
                    "weights_audit": {"passed": True},
                }
            return PoolStageOutput(_replace_person(frame, person), receipt)

        return apply

    result = pool_tool.build_multispine_pool(
        _source_frame() if resume is None else None,
        _source_frame(measured_offset=99.0) if resume is None else None,
        puf_donor=pd.DataFrame(),
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
        impute=stage(
            "impute",
            lambda person: person.__setitem__(
                "fixture_transfer",
                person["measured"],
            ),
        ),
        derive=stage(
            "derive",
            lambda person: person.__setitem__(
                "fixture_derived",
                person["fixture_transfer"] + 1.0,
            ),
        ),
        seed=stage(
            "seed",
            lambda person: person.__setitem__(
                "fixture_seed",
                person["fixture_derived"] > 0.0,
            ),
        ),
        simulate=stage(
            "simulate",
            lambda person: person.__setitem__(
                "ssi",
                np.where(
                    person[support_channel_column("person")].eq("asec"),
                    1.0,
                    100.0,
                ),
            ),
        ),
        checkpoint=store.write,
        resume=resume,
    )
    return result, order


def _run_production_impute_checkpoint_fixture(
    pool_tool: ModuleType,
    *,
    store,
    primary_qrf_checkpoint_dir: Path,
    acs_transfer_checkpoint_dir: Path,
    checkpoint_input_binding: Mapping[str, object],
):
    """Run the production impute closure while keeping later fixture stages tiny."""

    def stage(
        transform: Callable[[pd.DataFrame], None],
    ) -> Callable[[Frame], PoolStageOutput]:
        def apply(frame: Frame) -> PoolStageOutput:
            person = frame.table("person").copy()
            transform(person)
            return PoolStageOutput(
                _replace_person(frame, person),
                {"fixture_stage": True},
            )

        return apply

    return pool_tool.build_multispine_pool(
        _source_frame(),
        _source_frame(measured_offset=99.0),
        puf_donor=pd.DataFrame(),
        acs_rent_donor=pd.DataFrame(),
        primary_qrf_checkpoint_dir=primary_qrf_checkpoint_dir,
        acs_transfer_checkpoint_dir=acs_transfer_checkpoint_dir,
        checkpoint_identity=store.base_identity,
        checkpoint_input_binding=checkpoint_input_binding,
        prepare_clone=stage(lambda _person: None),
        derive=stage(
            lambda person: person.__setitem__("fixture_derived", 1.0),
        ),
        seed=stage(
            lambda person: person.__setitem__("fixture_seed", True),
        ),
        simulate=stage(
            lambda person: person.__setitem__(
                "ssi",
                np.where(
                    person[support_channel_column("person")].eq("asec"),
                    1.0,
                    100.0,
                ),
            ),
        ),
        checkpoint=store.write,
    )


def _stub_production_impute_kernels(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    *,
    active_bank_receipt: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Replace expensive kernels while retaining real routing and resume logic."""

    def initialize_primary_fixture(
        _frame: Frame,
        _donor: pd.DataFrame,
        checkpoint_dir: Path,
        **_kwargs,
    ) -> None:
        checkpoint_dir.mkdir(parents=True)
        pool_tool._atomic_write_json(
            pool_tool._primary_qrf_manifest_path(checkpoint_dir),
            {
                "artifact_kind": "fixture_primary_qrf_manifest",
                "target_order": ["fixture_target"],
            },
        )

    tail_receipt = {
        "tail_distribution_receipts": {
            "frame_after_stage": {
                "positive_mass_five_x_target_exceeded": True,
            }
        }
    }

    monkeypatch.setattr(
        pool_tool,
        "initialize_primary_puf_qrf_chain",
        initialize_primary_fixture,
    )
    monkeypatch.setattr(pool_tool, "run_primary_puf_qrf_chain", lambda *_args: None)
    monkeypatch.setattr(
        pool_tool,
        "finalize_primary_puf_qrf_chain",
        lambda frame, *_args, **_kwargs: (frame, "calibrated"),
    )
    monkeypatch.setattr(
        pool_tool,
        "transfer_puf_capital_gains_tail",
        lambda frame, *_args, **_kwargs: (frame, tail_receipt),
    )
    monkeypatch.setattr(
        pool_tool,
        "validate_puf_capital_gains_tail_manifest",
        lambda _receipt: None,
    )
    monkeypatch.setattr(
        pool_tool,
        "complete_multispine_source_inputs",
        lambda frame: SimpleNamespace(frame=frame, receipt={"fixture": True}),
    )
    monkeypatch.setattr(
        pool_tool,
        "pool_transfer_target_families",
        lambda: {"person": {"fixture": ("fixture_target",)}},
    )

    def transfer_fixture(frame: Frame, *_args, target_bank=None, **_kwargs):
        assert isinstance(target_bank, pool_tool.AcsTransferTargetBankStore)
        return SimpleNamespace(
            frame=frame,
            fit_records=(),
            resolved_donor_channel="fixture",
            imputed_inputs=(),
            deferred_inputs=(),
        )

    monkeypatch.setattr(pool_tool, "transfer_acs_inputs", transfer_fixture)
    if active_bank_receipt is not None:
        monkeypatch.setattr(
            pool_tool.AcsTransferTargetBankStore,
            "receipt",
            lambda _self: copy.deepcopy(active_bank_receipt["value"]),
        )
    return {
        "artifact_kind": "fixture_pool_input_binding",
        "schema_version": 1,
    }


def _target_bank_receipt_after_interruption(
    durable_target_index: int | None,
    *,
    total_targets: int = 9,
) -> dict[str, object]:
    targets: dict[str, object] = {}
    for index in range(total_targets):
        resumed = durable_target_index is not None and index <= durable_target_index
        source = "checkpoint" if resumed else "rebuilt"
        record: dict[str, object] = {
            "source": source,
            "descriptor": {
                "target_index": index,
                "total_targets": total_targets,
                "model_target": f"target_{index}",
            },
            "load_status": "resumed" if resumed else "missing",
            "path": f"/fixture/targets/{index:03d}__target_{index}.h5",
            "checkpoint_sha256": f"{index:064x}",
            "size_bytes": 1_000 + index,
        }
        if not resumed:
            record["write_status"] = "rebuilt"
            record["write_seconds"] = index + 0.25
        targets[str(index)] = record
    return {
        "artifact_kind": "populace_us_multispine_acs_transfer_target_bank_provenance",
        "schema_version": 1,
        "materializer_version": ACS_TRANSFER_TARGET_BANK_MATERIALIZER_VERSION,
        "root": "/fixture/acs-transfer",
        "identity": {"fixture": "identity"},
        "identity_sha256": "a" * 64,
        "targets": targets,
    }


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
        pool_tool.load_simulation_ready_us_multispine_pool_manifest(outputs.manifest)


def test_parser_exposes_six_pinned_inputs_out_and_checkpoint_root(
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
        "checkpoint_root",
        "out",
    }

    assert set(actions) == expected_destinations
    assert all(
        actions[destination].required
        for destination in expected_destinations - {"checkpoint_root"}
    )
    assert not actions["checkpoint_root"].required
    assert actions["out"].option_strings == ["--out"]
    assert actions["checkpoint_root"].option_strings == ["--checkpoint-root"]
    assert actions["checkpoint_root"].type is Path
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
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "load_asec_raw_stage_checkpoint" in imported
    assert "load_asec_raw_stage_checkpoint" in called
    assert "load_asec_pre_clone_checkpoint" not in imported
    assert "load_asec_pre_clone_checkpoint" not in called
    assert "pre_clone_enrichment" not in source


def test_checkpoint_root_defaults_alongside_out_and_accepts_override(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    output = tmp_path / "run" / "pool.h5"
    default = pool_tool._output_paths(output)

    assert default.checkpoint_root == output.with_suffix(".checkpoints")
    assert default.primary_qrf_checkpoint_dir == (
        output.with_suffix(".checkpoints") / "primary-qrf"
    )
    assert default.acs_transfer_checkpoint_dir == (
        output.with_suffix(".checkpoints") / "acs-transfer"
    )

    explicit_root = tmp_path / "persistent" / "pool-checkpoints"
    explicit = pool_tool._output_paths(
        output,
        checkpoint_root=explicit_root,
    )

    assert explicit.checkpoint_root == explicit_root
    assert explicit.primary_qrf_checkpoint_dir == explicit_root / "primary-qrf"
    assert explicit.acs_transfer_checkpoint_dir == explicit_root / "acs-transfer"

    identity_sha256 = "a" * 64
    bound = pool_tool._with_checkpoint_identity(
        explicit,
        base_identity_sha256=identity_sha256,
    )
    assert bound.primary_qrf_checkpoint_dir == (
        explicit_root / "primary-qrf" / identity_sha256
    )
    assert bound.acs_transfer_checkpoint_dir == (
        explicit_root / "acs-transfer" / identity_sha256
    )


def test_identity_routed_bank_sibling_scan_is_bounded_and_ignores_junk(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bank_root = tmp_path / "primary-qrf"
    bank_root.mkdir()
    current_digest = "f" * 64
    selected = bank_root / current_digest

    class _Entry:
        def __init__(self, name: str, *, directory: bool) -> None:
            self.name = name
            self._directory = directory

        def is_dir(self, *, follow_symlinks: bool) -> bool:
            assert follow_symlinks is False
            return self._directory

    entries = [
        _Entry(current_digest, directory=True),
        _Entry("not-a-digest", directory=True),
        _Entry("1" * 64, directory=False),
        _Entry("2" * 64, directory=False),
        _Entry("3" * 64, directory=True),
        *[_Entry(f"{index:064x}", directory=True) for index in range(10, 70)],
    ]

    class _Scandir:
        def __enter__(self):
            return iter(entries)

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(pool_tool.os, "scandir", lambda _root: _Scandir())

    receipt = pool_tool._identity_routed_bank_open_receipt(
        selected,
        current_base_identity_sha256=current_digest,
    )

    assert receipt["scan"] == {
        "limit": pool_tool._BANK_IDENTITY_SIBLING_SCAN_LIMIT,
        "entries_examined": pool_tool._BANK_IDENTITY_SIBLING_SCAN_LIMIT,
        "truncated": True,
    }
    stale_digests = {
        record["stale_base_identity_sha256"]
        for record in receipt["identity_mismatches"]
    }
    assert "3" * 64 in stale_digests
    assert current_digest not in stale_digests
    assert "1" * 64 not in stale_digests
    assert "2" * 64 not in stale_digests
    assert all(
        record["load_status"] == "identity_mismatch"
        and record["disposition"] == "bypassed"
        for record in receipt["identity_mismatches"]
    )


def test_checkpoint_root_allows_safe_input_colocation_on_persistent_volume(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "persistent-volume"
    outputs = pool_tool._output_paths(
        tmp_path / "published" / "pool.h5",
        checkpoint_root=checkpoint_root,
    )

    pool_tool._validate_checkpoint_path_layout(
        outputs,
        source_paths={checkpoint_root / "inputs" / "asec.h5"},
    )


@pytest.mark.parametrize(
    "outputs",
    (
        lambda pool_tool, tmp_path: pool_tool._output_paths(
            tmp_path / "pool.h5",
            checkpoint_root=tmp_path / "pool.h5",
        ),
        lambda pool_tool, tmp_path: pool_tool._output_paths(
            tmp_path / "checkpoints" / "assembled.checkpoint.h5",
            checkpoint_root=tmp_path / "checkpoints",
        ),
    ),
)
def test_checkpoint_root_rejects_publication_file_collisions(
    pool_tool: ModuleType,
    tmp_path: Path,
    outputs: Callable[[ModuleType, Path], object],
) -> None:
    with pytest.raises(
        ValueError,
        match="checkpoint paths collide with publication files",
    ):
        pool_tool._validate_checkpoint_path_layout(
            outputs(pool_tool, tmp_path),
            source_paths=set(),
        )


def test_atomic_json_fsyncs_parent_directory_after_rename(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, str]] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracked_fsync(descriptor: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        events.append(("fsync", kind))
        real_fsync(descriptor)

    def tracked_replace(source: Path, destination: Path) -> None:
        events.append(("replace", Path(destination).name))
        real_replace(source, destination)

    monkeypatch.setattr(pool_tool.os, "fsync", tracked_fsync)
    monkeypatch.setattr(pool_tool.os, "replace", tracked_replace)
    output = tmp_path / "receipt.json"

    pool_tool._atomic_write_json(output, {"fixture": True})

    assert events == [
        ("fsync", "file"),
        ("replace", output.name),
        ("fsync", "directory"),
    ]


def test_pool_imputation_wires_post_clone_source_chain_after_primary_and_tail(
    pool_tool: ModuleType,
) -> None:
    source = inspect.getsource(pool_tool._impute_pool)
    tree = ast.parse(source)
    expected = (
        "_initialize_or_resume_primary_qrf",
        "run_primary_puf_qrf_chain",
        "finalize_primary_puf_qrf_chain",
        "transfer_puf_capital_gains_tail",
        "complete_multispine_source_inputs",
        "pool_transfer_target_families",
        "AcsTransferTargetBankStore",
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

    bank_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AcsTransferTargetBankStore"
    )
    identity_value = next(
        keyword.value for keyword in bank_call.keywords if keyword.arg == "identity"
    )
    assert isinstance(identity_value, ast.Call)
    assert isinstance(identity_value.func, ast.Name)
    assert identity_value.func.id == "_pool_checkpoint_stage_identity"
    assert isinstance(identity_value.args[0], ast.Name)
    assert identity_value.args[0].id == "checkpoint_identity"
    assert isinstance(identity_value.args[1], ast.Constant)
    assert identity_value.args[1].value == "transferred"

    transfer_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "transfer_acs_inputs"
    )
    target_bank_value = next(
        keyword.value
        for keyword in transfer_call.keywords
        if keyword.arg == "target_bank"
    )
    assert isinstance(target_bank_value, ast.Name)
    assert target_bank_value.id == "target_bank"


def test_pool_imputation_binds_and_publishes_acs_target_bank(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _source_frame()
    checkpoint_identity = {
        "artifact_kind": "fixture-pool-checkpoint-identity",
        "materializer_version": 7,
        "inputs": {"fixture": {"sha256": "b" * 64}},
    }
    base_identity_sha256 = pool_tool._pool_checkpoint_identity_sha256(
        checkpoint_identity
    )
    primary_qrf_dir = tmp_path / "primary-qrf" / base_identity_sha256
    primary_qrf_dir.mkdir(parents=True)
    pool_tool._atomic_write_json(
        pool_tool._primary_qrf_manifest_path(primary_qrf_dir),
        {"artifact_kind": "fixture-primary-manifest"},
    )
    pool_tool._atomic_write_json(
        pool_tool._primary_qrf_input_binding_path(primary_qrf_dir),
        {"artifact_kind": "fixture-primary-binding"},
    )
    acs_bank_dir = tmp_path / "acs-transfer" / base_identity_sha256
    observed: dict[str, object] = {}
    bank_receipt = {
        "artifact_kind": "fixture-acs-target-bank-receipt",
        "targets": {"0": {"source": "checkpoint"}},
    }

    class _RecordingBank:
        def __init__(self, root: Path, *, identity: object) -> None:
            observed["bank"] = self
            observed["root"] = root
            observed["identity"] = identity

        def receipt(self) -> dict[str, object]:
            return bank_receipt

    def fake_transfer(*args, target_bank=None, **kwargs):
        observed["transfer_target_bank"] = target_bank
        return SimpleNamespace(
            frame=args[0],
            fit_records=(),
            resolved_donor_channel="fixture",
            imputed_inputs=(),
            deferred_inputs=(),
        )

    tail_receipt = {
        "tail_distribution_receipts": {
            "frame_after_stage": {
                "positive_mass_five_x_target_exceeded": True,
            }
        }
    }
    monkeypatch.setattr(
        pool_tool,
        "_initialize_or_resume_primary_qrf",
        lambda *args, **kwargs: "resumed",
    )
    monkeypatch.setattr(pool_tool, "run_primary_puf_qrf_chain", lambda *args: None)
    monkeypatch.setattr(
        pool_tool,
        "finalize_primary_puf_qrf_chain",
        lambda *args, **kwargs: (frame, "calibrated"),
    )
    monkeypatch.setattr(
        pool_tool,
        "transfer_puf_capital_gains_tail",
        lambda *args, **kwargs: (frame, tail_receipt),
    )
    monkeypatch.setattr(
        pool_tool,
        "validate_puf_capital_gains_tail_manifest",
        lambda receipt: None,
    )
    monkeypatch.setattr(
        pool_tool,
        "complete_multispine_source_inputs",
        lambda input_frame: SimpleNamespace(frame=input_frame, receipt={}),
    )
    monkeypatch.setattr(
        pool_tool,
        "pool_transfer_target_families",
        lambda: {"person": {"fixture": ("fixture_target",)}},
    )
    monkeypatch.setattr(pool_tool, "AcsTransferTargetBankStore", _RecordingBank)
    monkeypatch.setattr(pool_tool, "transfer_acs_inputs", fake_transfer)

    result = pool_tool._impute_pool(
        frame,
        puf_donor=pd.DataFrame(),
        primary_qrf_checkpoint_dir=primary_qrf_dir,
        acs_transfer_checkpoint_dir=acs_bank_dir,
        checkpoint_identity=checkpoint_identity,
        checkpoint_input_binding={"artifact_kind": "fixture-input-binding"},
    )

    assert observed["root"] == acs_bank_dir
    assert observed["identity"] == pool_tool._pool_checkpoint_stage_identity(
        checkpoint_identity,
        "transferred",
    )
    assert observed["transfer_target_bank"] is observed["bank"]
    assert result.receipt["acs_qrf_transfer"]["target_bank"] == bank_receipt


def test_production_pool_wires_source_preparation_into_clone_stage(
    pool_tool: ModuleType,
) -> None:
    source = inspect.getsource(pool_tool.build_multispine_pool)

    assert "prepare_multispine_source_inputs_for_clone" in source
    assert "prepare_clone=prepare_clone_operator" in source


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


@pytest.mark.parametrize(
    ("interruption_label", "durable_target_index"),
    (
        ("j0", 0),
        ("j1", 1),
        ("j2", 2),
        ("mid", 4),
        ("last", 8),
    ),
)
def test_production_transferred_checkpoint_bytes_ignore_bank_interruption_provenance(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    interruption_label: str,
    durable_target_index: int,
) -> None:
    active_bank_receipt: dict[str, Mapping[str, object]] = {}
    checkpoint_input_binding = _stub_production_impute_kernels(
        pool_tool,
        monkeypatch,
        active_bank_receipt=active_bank_receipt,
    )

    canonical_acs_receipt_keys = {
        "target_families",
        "n_estimators",
        "max_targets_per_fit",
        "resolved_donor_channel",
        "imputed_inputs",
        "fit_records",
        "deferred_inputs",
    }
    canonical_primary_receipt_keys = {
        "checkpoint_manifest",
        "checkpoint_manifest_sha256",
        "input_binding",
        "input_binding_sha256",
        "n_estimators",
        "tail_bound_diagnostics",
    }
    canonical_impute_receipt_keys = {
        "source_operator_chain",
        "primary_puf_qrf",
        "puf_capital_gains_tail_transfer",
        "acs_qrf_transfer",
        "weights_audit",
    }
    cold_receipt = _target_bank_receipt_after_interruption(None)
    cold_store = _checkpoint_fixture_store(pool_tool, tmp_path / "cold-checkpoints")
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    primary_qrf_dir = tmp_path / "primary-qrf" / cold_store.base_identity_sha256
    acs_transfer_dir = tmp_path / "acs-transfer" / cold_store.base_identity_sha256
    active_bank_receipt["value"] = cold_receipt
    cold_result = _run_production_impute_checkpoint_fixture(
        pool_tool,
        store=cold_store,
        primary_qrf_checkpoint_dir=primary_qrf_dir,
        acs_transfer_checkpoint_dir=acs_transfer_dir,
        checkpoint_input_binding=checkpoint_input_binding,
    )
    cold_bytes = cold_store.checkpoint_path("transferred").read_bytes()
    cold_sidecar = pool_tool._read_json_object(
        cold_store.checkpoint_receipts_path("transferred")
    )["operational_stage_receipts"]
    cold_primary_operational = cold_sidecar["impute"]["primary_puf_qrf"]
    assert cold_primary_operational["resume_status"] == "initialized"
    assert cold_primary_operational["identity_routing"]["identity_mismatches"] == []
    cold_target_bank = copy.deepcopy(
        cold_sidecar["impute"]["acs_qrf_transfer"]["target_bank"]
    )
    assert cold_target_bank.pop("identity_routing")["identity_mismatches"] == []
    assert cold_target_bank == cold_receipt
    assert (
        cold_result.stage_receipts["impute"]["primary_puf_qrf"]["resume_status"]
        == "initialized"
    )

    resumed_receipt = _target_bank_receipt_after_interruption(durable_target_index)
    resumed_store = _checkpoint_fixture_store(
        pool_tool,
        tmp_path / f"{interruption_label}-checkpoints",
    )
    resumed_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    assert resumed_store.base_identity_sha256 == cold_store.base_identity_sha256
    active_bank_receipt["value"] = resumed_receipt
    resumed_result = _run_production_impute_checkpoint_fixture(
        pool_tool,
        store=resumed_store,
        primary_qrf_checkpoint_dir=primary_qrf_dir,
        acs_transfer_checkpoint_dir=acs_transfer_dir,
        checkpoint_input_binding=checkpoint_input_binding,
    )

    transferred_path = resumed_store.checkpoint_path("transferred")
    assert transferred_path.read_bytes() == cold_bytes
    metadata = pool_tool.load_frame_checkpoint(transferred_path).metadata
    canonical_receipts = metadata["stage_receipts"]
    assert set(canonical_receipts) == {"clone", "impute"}
    canonical_impute_receipt = canonical_receipts["impute"]
    assert set(canonical_impute_receipt) == canonical_impute_receipt_keys
    canonical_primary_receipt = canonical_impute_receipt["primary_puf_qrf"]
    assert set(canonical_primary_receipt) == canonical_primary_receipt_keys
    assert "resume_status" not in canonical_primary_receipt
    canonical_acs_receipt = canonical_impute_receipt["acs_qrf_transfer"]
    assert set(canonical_acs_receipt) == canonical_acs_receipt_keys
    assert "target_bank" not in canonical_acs_receipt

    receipts_path = resumed_store.checkpoint_receipts_path("transferred")
    sidecar = pool_tool._read_json_object(receipts_path)
    operational_impute = sidecar["operational_stage_receipts"]["impute"]
    assert operational_impute["primary_puf_qrf"]["resume_status"] == "resumed"
    assert (
        operational_impute["primary_puf_qrf"]["identity_routing"]["identity_mismatches"]
        == []
    )
    resumed_target_bank = copy.deepcopy(
        operational_impute["acs_qrf_transfer"]["target_bank"]
    )
    assert resumed_target_bank.pop("identity_routing")["identity_mismatches"] == []
    assert resumed_target_bank == resumed_receipt

    resumed_store.checkpoint_path("simulated").unlink()
    resumed_store.checkpoint_manifest_path("simulated").unlink()
    warm_store = _checkpoint_fixture_store(
        pool_tool,
        tmp_path / f"{interruption_label}-checkpoints",
    )
    checkpoint = warm_store.load_deepest()
    assert checkpoint is not None
    assert checkpoint.stage == "transferred"
    assert (
        checkpoint.stage_receipts["impute"]["primary_puf_qrf"]["resume_status"]
        == "resumed"
    )
    restored_target_bank = copy.deepcopy(
        checkpoint.stage_receipts["impute"]["acs_qrf_transfer"]["target_bank"]
    )
    restored_target_bank.pop("identity_routing")
    assert restored_target_bank == resumed_receipt
    runtime_target_bank = copy.deepcopy(
        resumed_result.stage_receipts["impute"]["acs_qrf_transfer"]["target_bank"]
    )
    runtime_target_bank.pop("identity_routing")
    assert runtime_target_bank == resumed_receipt
    receipts_provenance = warm_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf"
    )["stages"]["transferred"]["receipts_sidecar"]
    assert receipts_provenance["load_status"] == "loaded"
    assert receipts_provenance["path"] == str(receipts_path.resolve())


@pytest.mark.parametrize("damage", ("missing", "corrupt"))
def test_operational_receipts_sidecar_damage_does_not_invalidate_checkpoint(
    pool_tool: ModuleType,
    tmp_path: Path,
    damage: str,
) -> None:
    root = tmp_path / f"{damage}-checkpoints"
    target_bank_receipt = _target_bank_receipt_after_interruption(2)
    cold_store = _checkpoint_fixture_store(pool_tool, root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=cold_store,
        target_bank_receipt=target_bank_receipt,
    )
    transferred_path = cold_store.checkpoint_path("transferred")
    canonical_bytes = transferred_path.read_bytes()
    expected_identity_sha256 = pool_tool.load_frame_checkpoint(
        transferred_path
    ).metadata["identity_sha256"]
    cold_store.checkpoint_path("simulated").unlink()
    cold_store.checkpoint_manifest_path("simulated").unlink()
    receipts_path = cold_store.checkpoint_receipts_path("transferred")
    if damage == "missing":
        receipts_path.unlink()
        expected_status = "missing"
    else:
        receipts_path.write_text("{not-json", encoding="utf-8")
        expected_status = "invalid_ignored"

    warm_store = _checkpoint_fixture_store(pool_tool, root)
    checkpoint = warm_store.load_deepest()

    assert checkpoint is not None
    assert checkpoint.stage == "transferred"
    assert transferred_path.read_bytes() == canonical_bytes
    assert (
        pool_tool.load_frame_checkpoint(transferred_path).metadata["identity_sha256"]
        == expected_identity_sha256
    )
    assert "target_bank" not in checkpoint.stage_receipts["impute"]["acs_qrf_transfer"]
    receipts_provenance = warm_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf"
    )["stages"]["transferred"]["receipts_sidecar"]
    assert receipts_provenance["load_status"] == expected_status


def test_same_identity_rewrite_cannot_reattach_stale_operational_receipts(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "rewrite-checkpoints"
    receipt_a = _target_bank_receipt_after_interruption(0)
    cold_store = _checkpoint_fixture_store(pool_tool, root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=cold_store,
        target_bank_receipt=receipt_a,
    )
    receipts_path = cold_store.checkpoint_receipts_path("transferred")
    assert receipts_path.is_file()
    cold_store.checkpoint_path("simulated").unlink()
    cold_store.checkpoint_manifest_path("simulated").unlink()
    cold_store.checkpoint_receipts_path("simulated").unlink()

    real_atomic_write_json = pool_tool._atomic_write_json

    def interrupt_before_receipts_install(path: Path, payload: Mapping[str, object]):
        if Path(path) == receipts_path:
            assert not receipts_path.exists()
            raise RuntimeError("fixture crash before fresh receipts install")
        return real_atomic_write_json(path, payload)

    monkeypatch.setattr(
        pool_tool,
        "_atomic_write_json",
        interrupt_before_receipts_install,
    )
    rewrite_store = _checkpoint_fixture_store(pool_tool, root)
    rewrite_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    with pytest.raises(RuntimeError, match="before fresh receipts install"):
        _run_checkpoint_fixture(
            pool_tool,
            tmp_path,
            store=rewrite_store,
            target_bank_receipt=_target_bank_receipt_after_interruption(4),
        )
    assert not receipts_path.exists()

    warm_store = _checkpoint_fixture_store(pool_tool, root)
    checkpoint = warm_store.load_deepest()

    assert checkpoint is not None
    assert checkpoint.stage == "transferred"
    assert "target_bank" not in checkpoint.stage_receipts["impute"]["acs_qrf_transfer"]
    receipts_provenance = warm_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf"
    )["stages"]["transferred"]["receipts_sidecar"]
    assert receipts_provenance["load_status"] == "missing"


@pytest.mark.parametrize(
    ("resume_stage", "expected_order"),
    (
        ("assembled", ["impute", "derive", "seed", "simulate"]),
        ("transferred", ["derive", "seed", "simulate"]),
        ("simulated", []),
    ),
)
def test_pool_checkpoint_round_trip_resumes_each_boundary_byte_identically(
    pool_tool: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    resume_stage: str,
    expected_order: list[str],
) -> None:
    pytest.importorskip("h5py")
    pytest.importorskip("tables")
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    uninterrupted, cold_order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=cold_store,
    )
    assert cold_order == ["impute", "derive", "seed", "simulate"]
    cold_output = capsys.readouterr().out
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        assert cold_output.count(f"Rebuilt pool stage {stage!r}") == 1
    checkpoint_identity_sha256: dict[str, str] = {}
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        stored_metadata = pool_tool.load_frame_checkpoint(
            cold_store.checkpoint_path(stage)
        ).metadata
        checkpoint_identity_sha256[stage] = stored_metadata["identity_sha256"]
        assert "agreement_gate" not in stored_metadata
        assert "simulation_ready" not in stored_metadata
        assert "agreement" not in stored_metadata
        assert "agreement" not in stored_metadata["stage_receipts"]

    resume_index = pool_tool.POOL_CHECKPOINT_STAGE_ORDER.index(resume_stage)
    for deeper_stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER[resume_index + 1 :]:
        cold_store.checkpoint_path(deeper_stage).unlink()
        cold_store.checkpoint_manifest_path(deeper_stage).unlink()

    warm_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    resume = warm_store.load_deepest()
    assert resume is not None
    assert resume.stage == resume_stage
    resumed, warm_order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=warm_store,
        resume=resume,
    )

    assert warm_order == expected_order
    warm_output = capsys.readouterr().out
    cached_stages = pool_tool.POOL_CHECKPOINT_STAGE_ORDER[: resume_index + 1]
    assert (
        f"Resumed pool checkpoint {resume_stage!r} from " in warm_output
        and f"cached stages: {', '.join(cached_stages)}." in warm_output
    )
    for stage_index, stage in enumerate(pool_tool.POOL_CHECKPOINT_STAGE_ORDER):
        assert warm_output.count(f"Rebuilt pool stage {stage!r}") == (
            0 if stage_index <= resume_index else 1
        )
    assert resumed.assembly_receipt == uninterrupted.assembly_receipt
    assert resumed.provenance_counts == uninterrupted.provenance_counts
    assert resumed.stage_receipts == uninterrupted.stage_receipts
    assert pool_tool._json_ready(resumed.frame.metadata) == pool_tool._json_ready(
        uninterrupted.frame.metadata
    )
    for entity in uninterrupted.frame.entities:
        pd.testing.assert_frame_equal(
            resumed.frame.table(entity),
            uninterrupted.frame.table(entity),
            check_dtype=True,
            check_exact=True,
        )

    uninterrupted_pool_path = tmp_path / f"{resume_stage}.uninterrupted.pool.h5"
    resumed_pool_path = tmp_path / f"{resume_stage}.resumed.pool.h5"
    for path, result in (
        (uninterrupted_pool_path, uninterrupted),
        (resumed_pool_path, resumed),
    ):
        pool_tool.write_nullable_us_h5(
            result.frame,
            path,
            period=pool_tool.POOL_TIME_PERIOD,
            artifact_kind=pool_tool.POOL_H5_ARTIFACT_KIND,
            publication_run_id="fixture-publication",
        )
    with (
        pd.HDFStore(uninterrupted_pool_path, mode="r") as uninterrupted_store,
        pd.HDFStore(resumed_pool_path, mode="r") as resumed_store,
    ):
        assert resumed_store.keys() == uninterrupted_store.keys()
        for key in uninterrupted_store.keys():
            expected = uninterrupted_store[key]
            observed = resumed_store[key]
            if isinstance(expected, pd.DataFrame):
                pd.testing.assert_frame_equal(
                    observed,
                    expected,
                    check_dtype=True,
                    check_exact=True,
                )
            else:
                pd.testing.assert_series_equal(
                    observed,
                    expected,
                    check_dtype=True,
                    check_exact=True,
                )

    # PyTables' published container carries nondeterministic HDF metadata. The
    # deterministic Frame serializer gives the literal byte-level assertion
    # over the exact input-pool state after the production writer is checked.
    uninterrupted_canonical = tmp_path / f"{resume_stage}.uninterrupted.canonical.h5"
    resumed_canonical = tmp_path / f"{resume_stage}.resumed.canonical.h5"
    pool_tool.write_frame_checkpoint(uninterrupted_canonical, uninterrupted.frame)
    pool_tool.write_frame_checkpoint(resumed_canonical, resumed.frame)
    assert resumed_canonical.read_bytes() == uninterrupted_canonical.read_bytes()

    provenance = warm_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
    )
    assert provenance["deepest_resumed_stage"] == resume_stage
    assert provenance["base_identity_sha256"] == warm_store.base_identity_sha256
    assert (
        provenance["primary_qrf"]["base_identity_sha256"]
        == warm_store.base_identity_sha256
    )
    assert (
        provenance["acs_transfer"]["base_identity_sha256"]
        == warm_store.base_identity_sha256
    )
    assert provenance["acs_transfer"]["boundary_stage"] == "transferred"
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        assert (
            provenance["stages"][stage]["identity_sha256"]
            == checkpoint_identity_sha256[stage]
        )
    assert provenance["stages"][resume_stage]["source"] == "checkpoint"
    assert provenance["stages"][resume_stage]["resume_kind"] == "direct"
    for covered_stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER[:resume_index]:
        covered = provenance["stages"][covered_stage]
        assert covered["source"] == "checkpoint"
        assert covered["resume_kind"] == "covered_by_deeper_checkpoint"
        assert covered["source_checkpoint_stage"] == resume_stage
        assert covered["path"] == str(
            warm_store.checkpoint_path(resume_stage).resolve()
        )
        assert covered["nominal_stage_path"] == str(
            warm_store.checkpoint_path(covered_stage).resolve()
        )
    assert provenance["agreement"] == {
        "source": "always_fresh",
        "cached": False,
        "terminal_verdict_persisted": False,
    }


def test_resumed_checkpoint_provenance_is_published_in_final_manifest(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    _unused, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    base_identity = pool_tool._pool_checkpoint_base_identity(
        verified_inputs,
        policyengine_us_version="fixture-engine-1",
    )
    cold_store = pool_tool._PoolStageCheckpointStore(
        outputs.checkpoint_root,
        base_identity=base_identity,
    )
    cold_store.bind_input_receipts(pool_tool._loaded_input_receipts(loaded))
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)

    warm_store = pool_tool._PoolStageCheckpointStore(
        outputs.checkpoint_root,
        base_identity=base_identity,
    )
    resume = warm_store.load_deepest()
    assert resume is not None
    assert resume.stage == "simulated"
    resumed, order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=warm_store,
        resume=resume,
    )
    assert order == []
    checkpoint_provenance = warm_store.provenance(
        primary_qrf_checkpoint_dir=outputs.primary_qrf_checkpoint_dir,
    )

    pool_tool._write_outputs(
        resumed,
        outputs=outputs,
        verified_inputs=verified_inputs,
        acs_source_manifest=source_manifest,
        input_receipts=warm_store.input_receipts,
        checkpoint_provenance=checkpoint_provenance,
    )

    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert manifest["stage_checkpoints"] == checkpoint_provenance
    assert (
        manifest["stage_checkpoints"]["base_identity_sha256"]
        == warm_store.base_identity_sha256
    )
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        stage_manifest = pool_tool._read_json_object(
            warm_store.checkpoint_manifest_path(stage)
        )
        assert (
            manifest["stage_checkpoints"]["stages"][stage]["identity_sha256"]
            == stage_manifest["identity_sha256"]
        )


def test_pool_checkpoint_input_sha_mismatch_rebuilds_every_stage(
    pool_tool: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)
    capsys.readouterr()

    changed_store = _checkpoint_fixture_store(
        pool_tool,
        checkpoint_root,
        changed_role="processed_puf",
    )
    assert changed_store.load_deepest() is None
    changed_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    rebuilt, order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=changed_store,
    )

    assert order == ["impute", "derive", "seed", "simulate"]
    assert not rebuilt.simulation_ready
    output = capsys.readouterr().out
    assert output.count("Ignored stale pool checkpoint") == 3
    provenance = changed_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
    )
    assert provenance["deepest_resumed_stage"] is None
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        assert provenance["stages"][stage]["source"] == "rebuilt"
        assert provenance["stages"][stage]["load_status"] == "identity_mismatch"


@pytest.mark.parametrize(
    "contract_field",
    (
        pytest.param("asserted_constraint", id="asserted_constraint_changed"),
        pytest.param(
            "inventory_built_against",
            id="inventory_built_against_changed",
        ),
    ),
)
def test_production_bank_routing_names_stale_contract_identity_siblings(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contract_field: str,
) -> None:
    checkpoint_root = tmp_path / "production-routing-checkpoints"
    base_outputs = pool_tool._output_paths(
        tmp_path / "pool.h5",
        checkpoint_root=checkpoint_root,
    )
    contract = load_take_up_contract()
    current_contract_identity = take_up_contract_identity(contract)
    stale_contract = replace(
        contract,
        **{contract_field: f"{getattr(contract, contract_field)}-stale"},
    )
    stale_contract_identity = take_up_contract_identity(stale_contract)
    monkeypatch.setattr(
        pool_tool,
        "take_up_contract_identity",
        lambda: stale_contract_identity,
    )
    stale_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    stale_outputs = pool_tool._with_checkpoint_identity(
        base_outputs,
        base_identity_sha256=stale_store.base_identity_sha256,
    )
    stale_markers = []
    for bank_dir in (
        stale_outputs.primary_qrf_checkpoint_dir,
        stale_outputs.acs_transfer_checkpoint_dir,
    ):
        bank_dir.mkdir(parents=True)
        marker = bank_dir / "must-not-be-opened"
        marker.write_text("stale bank marker\n", encoding="utf-8")
        stale_markers.append(marker)

    monkeypatch.setattr(
        pool_tool,
        "take_up_contract_identity",
        lambda: current_contract_identity,
    )
    current_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    current_outputs = pool_tool._with_checkpoint_identity(
        base_outputs,
        base_identity_sha256=current_store.base_identity_sha256,
    )
    assert current_store.base_identity_sha256 != stale_store.base_identity_sha256
    assert (
        current_outputs.primary_qrf_checkpoint_dir
        != stale_outputs.primary_qrf_checkpoint_dir
    )
    assert (
        current_outputs.acs_transfer_checkpoint_dir
        != stale_outputs.acs_transfer_checkpoint_dir
    )

    checkpoint_input_binding = _stub_production_impute_kernels(
        pool_tool,
        monkeypatch,
    )
    current_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    result = _run_production_impute_checkpoint_fixture(
        pool_tool,
        store=current_store,
        primary_qrf_checkpoint_dir=current_outputs.primary_qrf_checkpoint_dir,
        acs_transfer_checkpoint_dir=current_outputs.acs_transfer_checkpoint_dir,
        checkpoint_input_binding=checkpoint_input_binding,
    )

    operational_impute = pool_tool._read_json_object(
        current_store.checkpoint_receipts_path("transferred")
    )["operational_stage_receipts"]["impute"]
    assert operational_impute["primary_puf_qrf"]["resume_status"] == "initialized"
    primary_routing = operational_impute["primary_puf_qrf"]["identity_routing"]
    target_bank = operational_impute["acs_qrf_transfer"]["target_bank"]
    acs_routing = target_bank["identity_routing"]
    assert target_bank["root"] == str(
        current_outputs.acs_transfer_checkpoint_dir.resolve()
    )
    assert target_bank["targets"] == {}

    for routing, bank_root, selected_path, stale_path in (
        (
            primary_routing,
            current_outputs.primary_qrf_checkpoint_dir.parent,
            current_outputs.primary_qrf_checkpoint_dir,
            stale_outputs.primary_qrf_checkpoint_dir,
        ),
        (
            acs_routing,
            current_outputs.acs_transfer_checkpoint_dir.parent,
            current_outputs.acs_transfer_checkpoint_dir,
            stale_outputs.acs_transfer_checkpoint_dir,
        ),
    ):
        assert routing["bank_root"] == str(bank_root.resolve())
        assert routing["selected_path"] == str(selected_path.resolve())
        assert (
            routing["current_base_identity_sha256"]
            == current_store.base_identity_sha256
        )
        assert routing["scan"] == {
            "limit": pool_tool._BANK_IDENTITY_SIBLING_SCAN_LIMIT,
            "entries_examined": 1,
            "truncated": False,
        }
        assert routing["identity_mismatches"] == [
            {
                "load_status": "identity_mismatch",
                "stale_base_identity_sha256": stale_store.base_identity_sha256,
                "current_base_identity_sha256": current_store.base_identity_sha256,
                "disposition": "bypassed",
                "path": str(stale_path.resolve()),
            }
        ]
    assert all(
        marker.read_text(encoding="utf-8") == "stale bank marker\n"
        for marker in stale_markers
    )
    assert (
        result.stage_receipts["impute"]["primary_puf_qrf"]["identity_routing"]
        == primary_routing
    )


@pytest.mark.parametrize(
    "contract_field",
    (
        pytest.param("asserted_constraint", id="asserted_constraint_changed"),
        pytest.param(
            "inventory_built_against",
            id="inventory_built_against_changed",
        ),
    ),
)
def test_take_up_contract_identity_mutation_rebuilds_every_pool_boundary(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contract_field: str,
) -> None:
    checkpoint_root = tmp_path / "take-up-contract-checkpoints"
    contract = load_take_up_contract()
    original_contract_identity = take_up_contract_identity(contract)
    monkeypatch.setattr(
        pool_tool,
        "take_up_contract_identity",
        lambda: original_contract_identity,
    )
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    assert (
        cold_store.base_identity["pool_code"]["take_up_contract"]
        == original_contract_identity
    )
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)

    changed_contract = replace(
        contract,
        **{contract_field: f"{getattr(contract, contract_field)}-changed"},
    )
    changed_contract_identity = take_up_contract_identity(changed_contract)
    monkeypatch.setattr(
        pool_tool,
        "take_up_contract_identity",
        lambda: changed_contract_identity,
    )
    changed_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)

    assert (
        changed_store.base_identity["pool_code"]["take_up_contract"]
        == changed_contract_identity
    )
    assert changed_store.base_identity_sha256 != cold_store.base_identity_sha256
    assert changed_store.load_deepest() is None
    changed_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    rebuilt, order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=changed_store,
    )

    assert order == ["impute", "derive", "seed", "simulate"]
    assert not rebuilt.simulation_ready
    provenance = changed_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
    )
    assert provenance["deepest_resumed_stage"] is None
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        assert provenance["stages"][stage]["source"] == "rebuilt"
        assert provenance["stages"][stage]["load_status"] == "identity_mismatch"


def test_pool_materializer_v1_artifacts_fail_closed_with_named_receipts(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_root = tmp_path / "materializer-v1-checkpoints"

    with monkeypatch.context() as legacy:
        legacy.setattr(pool_tool, "POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION", 1)
        legacy_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
        assert legacy_store.base_identity["materializer_version"] == 1
        legacy_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
        _run_checkpoint_fixture(pool_tool, tmp_path, store=legacy_store)
        for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
            metadata = pool_tool.load_frame_checkpoint(
                legacy_store.checkpoint_path(stage)
            ).metadata
            manifest = pool_tool._read_json_object(
                legacy_store.checkpoint_manifest_path(stage)
            )
            assert metadata["materializer_version"] == 1
            assert metadata["identity"]["materializer_version"] == 1
            assert manifest["materializer_version"] == 1
            assert manifest["identity"]["materializer_version"] == 1
    capsys.readouterr()

    assert pool_tool.POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION == 2
    current_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    assert current_store.base_identity["materializer_version"] == 2
    assert current_store.load_deepest() is None

    output = capsys.readouterr().out
    provenance = current_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
    )
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        assert f"Ignored corrupt pool checkpoint '{stage}'" in output
        receipt = provenance["stages"][stage]
        assert receipt["source"] == "rebuilt"
        assert receipt["load_status"] == "invalid_rebuild"
        invalid = receipt["invalid_checkpoint"]
        assert invalid["reason"] == "checkpoint_validation_failed"
        assert invalid["message"] == (
            f"{stage} checkpoint manifest has an unsupported binding"
        )


@pytest.mark.parametrize(
    ("corrupt_stage", "expected_resume", "expected_order"),
    (
        (
            "assembled",
            None,
            ["impute", "derive", "seed", "simulate"],
        ),
        (
            "transferred",
            "assembled",
            ["impute", "derive", "seed", "simulate"],
        ),
        (
            "simulated",
            "transferred",
            ["derive", "seed", "simulate"],
        ),
    ),
)
def test_corrupt_pool_checkpoint_names_boundary_and_rebuilds(
    pool_tool: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    corrupt_stage: str,
    expected_resume: str | None,
    expected_order: list[str],
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    baseline, _order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=cold_store,
    )
    capsys.readouterr()

    corrupt_index = pool_tool.POOL_CHECKPOINT_STAGE_ORDER.index(corrupt_stage)
    for deeper_stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER[corrupt_index + 1 :]:
        cold_store.checkpoint_path(deeper_stage).unlink()
        cold_store.checkpoint_manifest_path(deeper_stage).unlink()
    cold_store.checkpoint_path(corrupt_stage).write_bytes(b"corrupt checkpoint")

    warm_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    resume = warm_store.load_deepest()
    assert (None if resume is None else resume.stage) == expected_resume
    if resume is None:
        warm_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    rebuilt, order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=warm_store,
        resume=resume,
    )

    assert order == expected_order
    output = capsys.readouterr().out
    assert f"Ignored corrupt pool checkpoint '{corrupt_stage}'" in output
    assert "SHA-256 mismatch" in output
    for entity in baseline.frame.entities:
        pd.testing.assert_frame_equal(
            rebuilt.frame.table(entity),
            baseline.frame.table(entity),
            check_dtype=True,
            check_exact=True,
        )
    provenance = warm_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
    )
    corrupt_receipt = provenance["stages"][corrupt_stage]
    assert corrupt_receipt["source"] == "rebuilt"
    assert corrupt_receipt["load_status"] == "invalid_rebuild"
    assert corrupt_receipt["invalid_checkpoint"]["reason"] == (
        "checkpoint_validation_failed"
    )


@pytest.mark.parametrize(
    ("drift_field", "expected_error"),
    (
        ("row_counts", "row_counts differs from its sidecar"),
        ("frame_schema", "frame_schema differs from its sidecar"),
    ),
)
def test_checkpoint_sidecar_shape_drift_names_failure_and_falls_back(
    pool_tool: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    drift_field: str,
    expected_error: str,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)
    capsys.readouterr()

    cold_store.checkpoint_path("simulated").unlink()
    cold_store.checkpoint_manifest_path("simulated").unlink()
    transferred_manifest_path = cold_store.checkpoint_manifest_path("transferred")
    transferred_manifest = pool_tool._read_json_object(transferred_manifest_path)
    if drift_field == "row_counts":
        transferred_manifest["row_counts"]["person"] += 1
    else:
        transferred_manifest["frame_schema"]["entities"]["person"][0]["dtype"] = (
            "corrupt-dtype"
        )
    pool_tool._atomic_write_json(transferred_manifest_path, transferred_manifest)

    warm_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    resume = warm_store.load_deepest()

    assert resume is not None
    assert resume.stage == "assembled"
    output = capsys.readouterr().out
    assert "Ignored corrupt pool checkpoint 'transferred'" in output
    assert expected_error in output
    transferred_receipt = warm_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
    )["stages"]["transferred"]
    assert transferred_receipt["load_status"] == "invalid_rebuild"
    assert transferred_receipt["invalid_checkpoint"]["reason"] == (
        "checkpoint_validation_failed"
    )


def test_invalid_deep_checkpoint_receipts_do_not_poison_valid_fallback(
    pool_tool: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)
    capsys.readouterr()

    simulated_path = cold_store.checkpoint_path("simulated")
    simulated = pool_tool.load_frame_checkpoint(simulated_path)
    poisoned_metadata = dict(simulated.metadata)
    poisoned_metadata["input_receipts"] = {"poisoned": True}
    poisoned_metadata["simulation_output"] = None
    pool_tool.write_frame_checkpoint(
        simulated_path,
        simulated.frame,
        metadata=poisoned_metadata,
    )
    simulated_manifest_path = cold_store.checkpoint_manifest_path("simulated")
    simulated_manifest = pool_tool._read_json_object(simulated_manifest_path)
    simulated_manifest["checkpoint"]["sha256"] = pool_tool._file_sha256(simulated_path)
    simulated_manifest["checkpoint"]["size_bytes"] = simulated_path.stat().st_size
    pool_tool._atomic_write_json(simulated_manifest_path, simulated_manifest)

    warm_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    resume = warm_store.load_deepest()

    assert resume is not None
    assert resume.stage == "transferred"
    assert warm_store.input_receipts == _checkpoint_fixture_input_receipts()
    output = capsys.readouterr().out
    assert "Ignored corrupt pool checkpoint 'simulated'" in output
    assert "invalid SSI output binding" in output


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
        person["is_female"] = pd.to_numeric(person["A_SEX"], errors="raise") == 2
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
    assert manifest["stage_checkpoints"]["enabled"] is False
    assert manifest["stage_checkpoints"]["agreement"] == {
        "source": "always_fresh",
        "cached": False,
        "terminal_verdict_persisted": False,
    }
    assert manifest["agreement_gate"] == expected_gate
    assert diagnostics["agreement_gate"] == expected_gate
    assert diagnostics["simulation_ready"] is False
    assert diagnostics["publication_run_id"] == manifest["publication_run_id"]
    assert manifest["provenance_pins"] == {
        role: pin.to_manifest() for role, pin in verified_inputs.items()
    }
    assert manifest["pool_h5"]["formula_outputs_persisted"] is False
    assert manifest["pool_h5"]["input_only"] is True
    assert manifest["pool_h5"]["publication_run_id"] == manifest["publication_run_id"]
    assert (
        manifest["pool_h5"]["sha256"]
        == hashlib.sha256(outputs.pool_h5.read_bytes()).hexdigest()
    )

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
        pool_tool.load_simulation_ready_us_multispine_pool_manifest(outputs.manifest)


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
    diagnostics = json.loads(outputs.agreement_diagnostics.read_text(encoding="utf-8"))
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
