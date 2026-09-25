"""Small-fixture tests for the terminal US multispine pool build tool."""

# ruff: noqa: F401

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
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.acs_transfer as acs_transfer_module
import microcosm.build.us_runtime.multispine_pool as multispine_pool_module
import microcosm.build.us_runtime.post_transfer_calibration as post_transfer_calibration_runtime
import microcosm.build.us_runtime.stacked_spine as stacked_spine_module
from microcosm.build.gates import GateReport, GateResult
from microcosm.build.logbook import LOGBOOK_ROW_FIELDS, load_logbook_row
from microcosm.build.serialization_dtypes import CANONICAL_STRING_DTYPE
from microcosm.build.spec_engine import LegacyPayloadMismatchError
from microcosm.build.us_runtime.acs_transfer import transfer_acs_inputs
from microcosm.build.us_runtime.acs_transfer_bank import (
    ACS_TRANSFER_TARGET_BANK_MATERIALIZER_VERSION,
)
from microcosm.build.us_runtime.multispine_pool import (
    MultispinePoolCheckpoint,
    PoolStageOutput,
)
from microcosm.build.us_runtime.operator_boundary import (
    PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID,
)
from microcosm.build.us_runtime.qbi_inputs import (
    US_QBI_OUTPUT_COLUMNS,
    bind_us_qbi_reconciliation_transition_authority,
    us_qbi_reconciliation_change_receipt,
    with_us_qbi_input_reconciliation,
)
from microcosm.build.us_runtime.stacked_spine import GapFillDirection
from microcosm.build.us_runtime.support_provenance import (
    support_channel_column,
    support_clone_index_column,
)
from microcosm.build.us_runtime.take_up_contract import (
    load_take_up_contract,
    take_up_contract_identity,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights, read_frame_table
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

_FIXTURE_SEED_PERSON_COLUMN = "takes_up_medicaid_if_eligible"


@pytest.fixture(autouse=True)
def _prime_worker_identity(prime_primary_qrf_worker_identity: None) -> None:
    """Share the real session attestation unless a test opts into live identity."""


@pytest.fixture(scope="module")
def pool_tool() -> ModuleType:
    root = _TEST_PATHS.repository
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


@pytest.fixture(scope="module")
def release_tool() -> ModuleType:
    root = _TEST_PATHS.repository
    path = root / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release_pool_fixture",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_frame(
    *,
    measured_offset: float = 0.0,
    include_peridnum: bool = True,
) -> Frame:
    ids = np.asarray([1, 2], dtype=np.int64)
    person_data = {
        "person_id": ids,
        "person_household_id": ids,
        "person_tax_unit_id": ids,
        "person_spm_unit_id": ids,
        "person_family_id": ids,
        "person_marital_unit_id": ids,
        "A_AGE": np.asarray([30.0, 50.0]),
        "A_SEX": np.asarray([1, 2], dtype=np.int64),
        "source_year": np.asarray([2024, 2024], dtype=np.int64),
        "measured": np.asarray([1.0, 2.0]) + measured_offset,
    }
    if include_peridnum:
        person_data["PERIDNUM"] = np.asarray(["1", "2"], dtype=object)
    person = pd.DataFrame(person_data)
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


def _many_household_source_frame(
    *,
    count: int = 100,
    measured_offset: float = 0.0,
    state_fips: str | None = None,
    puma: str | None = None,
) -> Frame:
    ids = np.arange(1, count + 1, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids,
            "person_spm_unit_id": ids,
            "person_family_id": ids,
            "person_marital_unit_id": ids,
            "A_AGE": np.full(count, 40.0),
            "A_SEX": np.where(ids % 2, 1, 2).astype(np.int64),
            "source_year": np.full(count, 2024, dtype=np.int64),
            "measured": ids.astype(np.float64) + measured_offset,
        }
    )
    tables = {
        "person": person,
        **{
            entity: pd.DataFrame({f"{entity}_id": ids})
            for entity in US_SCHEMA.group_entities
        },
    }
    if measured_offset:
        tables["household"]["TYPEHUGQ"] = 1
    if state_fips is not None:
        tables["household"]["state_fips"] = pd.Series(
            state_fips,
            index=tables["household"].index,
            dtype="string",
        )
    if puma is not None:
        tables["household"]["puma"] = pd.Series(
            puma,
            index=tables["household"].index,
            dtype="string",
        )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.full(count, 2.0), WeightKind.DESIGN)},
        pd.Series(["fixture"] * count, dtype=object),
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


def _fixture_qbi_stage_output(
    frame: Frame,
    receipt: Mapping[str, object],
) -> PoolStageOutput:
    """Attach a real, live-frame-bound QBI receipt to a tiny derive fixture."""

    person = frame.table("person").copy()
    if "age" not in person:
        person["age"] = pd.to_numeric(person["A_AGE"], errors="raise")
    if "SEMP" not in person:
        person["SEMP"] = 0.0
    if "self_employment_income_before_lsr" not in person:
        person["self_employment_income_before_lsr"] = 0.0
    if "non_qualified_dividend_income" not in person:
        person["non_qualified_dividend_income"] = 0.0
    for column in US_QBI_OUTPUT_COLUMNS:
        if column not in person:
            person[column] = 0.0
    before = _replace_person(frame, person)
    after = with_us_qbi_input_reconciliation(before)
    qbi_receipt = us_qbi_reconciliation_change_receipt(before, after)
    after = bind_us_qbi_reconciliation_transition_authority(after, qbi_receipt)
    return PoolStageOutput(
        after,
        {
            **dict(receipt),
            "qbi_input_reconciliation": qbi_receipt,
        },
        qbi_transition_authority_sha256=qbi_receipt["sha256"],
    )


def _with_fixture_pre_clone_strike_benefits(frame: Frame) -> Frame:
    """Fixture producer for one real pre-clone operator-owned target."""

    person = frame.table("person").copy()
    assert "strike_benefits" not in person
    channel = person[support_channel_column("person")].astype(str)
    person["strike_benefits"] = np.nan
    person.loc[channel.eq("asec"), "strike_benefits"] = 125.0
    return _replace_person(frame, person)


def _with_fixture_household_geography(frame: Frame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    household = tables["household"]
    household["puma"] = pd.Series("0600100", index=household.index, dtype="string")
    household["congressional_district_geoid"] = np.full(
        len(household), 601, dtype=np.int64
    )
    household["county_fips"] = pd.Series("06001", index=household.index, dtype="string")
    tables.update({link: frame.link(link) for link in frame.links})
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _fixture_geography_assignment_receipt(
    pool_tool: ModuleType,
    frame: Frame,
    *,
    gate: GateResult | None = None,
) -> dict[str, object]:
    household = frame.table("household")
    fixture_gate = gate or GateResult(
        name="us_puma_ladder",
        passed=True,
        details={"fixture": True},
    )
    return {
        "artifact_kind": "populace_us_stacked_household_geography_assignment",
        "schema_version": 1,
        "contract": pool_tool._stacked_geography_assignment_contract(),
        "pre_assignment_household_order": pool_tool._ordered_household_id_receipt(
            household
        ),
        "assigned_household_geography": (
            pool_tool._ordered_household_geography_receipt(household)
        ),
        "target_universe": (
            pool_tool._target_congressional_district_universe_receipt((601,))
        ),
        "output": {
            "household_rows": len(household),
            "positive_congressional_district_rows": len(household),
            "unique_congressional_district_values": 1,
        },
        "summary": {"applied": True, "household_rows": len(household)},
        "gate": GateReport((fixture_gate,)).to_manifest(),
    }


def _fixture_puma_ladder(pool_tool: ModuleType):
    puma = np.asarray([600_100], dtype=np.int64)
    population = np.asarray([100.0])
    return pool_tool.UsPumaLadder(
        puma=puma,
        puma_population=population,
        cd_overlap_puma=puma.copy(),
        cd_overlap_cd=np.asarray([601], dtype=np.int64),
        cd_overlap_population=population.copy(),
        county_overlap_puma=puma.copy(),
        county_overlap_county=np.asarray([6_001], dtype=np.int32),
        county_overlap_population=population.copy(),
        tract_overlap_puma=puma.copy(),
        tract_overlap_tract=np.asarray([6_001_000_100], dtype=np.int64),
        tract_overlap_population=population.copy(),
        metadata={
            "schema_version": 1,
            "kind": "us_puma_ladder",
            "puma_vintage": "2020_puma",
            "sampling_basis": "population",
            "layers": {
                "congressional_district": {"vintage": "119th_congress"},
                "county": {"vintage": "2020_census"},
                "tract": {"vintage": "2020_census"},
            },
        },
    )


def _semantic_string_columns(table: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in table.columns
        if isinstance(table[column].dtype, pd.StringDtype)
        or (
            pd.api.types.is_object_dtype(table[column].dtype)
            and pd.api.types.infer_dtype(table[column], skipna=True) == "string"
        )
    )


def _with_object_backed_strings(frame: Frame) -> Frame:
    tables = {}
    for entity in frame.entities:
        table = frame.table(entity).copy()
        for column in _semantic_string_columns(table):
            table[column] = table[column].astype(object)
        tables[entity] = table
    tables.update({link: frame.link(link) for link in frame.links})
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
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


def _red_pool_result(
    pool_tool: ModuleType,
    tmp_path: Path,
    *,
    authenticated_qbi: bool = False,
):
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
            if name == "derive" and authenticated_qbi:
                return _fixture_qbi_stage_output(
                    _replace_person(frame, person),
                    {"fixture_stage": name},
                )
            receipt: dict[str, object] = {"fixture_stage": name}
            if name == "seed" and authenticated_qbi:
                receipt["programs"] = {
                    _FIXTURE_SEED_PERSON_COLUMN: {"entity": "person"},
                }
            return PoolStageOutput(
                _replace_person(frame, person),
                receipt,
            )

        return apply

    def transfer(person: pd.DataFrame) -> None:
        person["fixture_transfer"] = person["measured"]

    def derive(person: pd.DataFrame) -> None:
        person["fixture_derived"] = person["fixture_transfer"] + 1.0

    def seed(person: pd.DataFrame) -> None:
        column = _FIXTURE_SEED_PERSON_COLUMN if authenticated_qbi else "fixture_seed"
        person[column] = person["fixture_derived"] > 0.0

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
    *,
    authenticated_qbi: bool = True,
):
    result = _red_pool_result(
        pool_tool,
        tmp_path,
        authenticated_qbi=authenticated_qbi,
    )
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


def _verified_inputs_fixture(pool_tool: ModuleType, root: Path):
    verified = {}
    for index, role in enumerate(
        (
            "asec_raw_stage",
            "acs_household",
            "acs_person",
            "acs_rent_donor",
            "processed_puf",
            "puf_source_year",
            "puma_ladder",
            "congressional_district_vintage_crosswalk",
        ),
        start=1,
    ):
        path = root / f"{role}.stacked-fixture"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"stacked-input-{index}".encode())
        digest = {
            "puma_ladder": pool_tool._STACKED_PUMA_LADDER_SHA256,
            "congressional_district_vintage_crosswalk": (
                pool_tool._STACKED_CD_CROSSWALK_SHA256
            ),
        }.get(role, hashlib.sha256(path.read_bytes()).hexdigest())
        verified[role] = pool_tool._VerifiedInput(
            role=role,
            path=path,
            expected_sha256=digest,
            actual_sha256=digest,
            size_bytes=path.stat().st_size,
        )
    return verified


def _run_checkpoint_fixture(
    pool_tool: ModuleType,
    tmp_path: Path,
    *,
    store,
    resume=None,
    target_bank_receipt: Mapping[str, object] | None = None,
    primary_qrf_manifest_path: Path | None = None,
    authenticated_qbi: bool = True,
    checkpoint_nullable_booleans: bool = False,
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
            if name == "impute" and checkpoint_nullable_booleans:
                complete = np.resize(
                    np.asarray([True, False], dtype=np.bool_),
                    len(person),
                )
                missing = pd.array(complete, dtype="boolean")
                missing[1] = pd.NA
                person["is_female"] = pd.Series(
                    complete,
                    index=person.index,
                    dtype="boolean",
                )
                person["fixture_declared_boolean"] = pd.Series(
                    missing,
                    index=person.index,
                )
            receipt: dict[str, object] = {"fixture_stage": name}
            if name == "impute" and primary_qrf_manifest_path is not None:
                receipt = {
                    "primary_puf_qrf": {
                        "mode": "checkpoint_chain",
                        "checkpoint_manifest_path": str(
                            primary_qrf_manifest_path.resolve()
                        ),
                    }
                }
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
            if name == "derive" and authenticated_qbi:
                return _fixture_qbi_stage_output(
                    _replace_person(frame, person),
                    receipt,
                )
            if name == "seed" and authenticated_qbi:
                receipt["programs"] = {
                    _FIXTURE_SEED_PERSON_COLUMN: {"entity": "person"},
                }
            return PoolStageOutput(_replace_person(frame, person), receipt)

        return apply

    result = pool_tool.build_multispine_pool(
        _source_frame() if resume is None else None,
        (
            _source_frame(measured_offset=99.0, include_peridnum=False)
            if resume is None
            else None
        ),
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
                (_FIXTURE_SEED_PERSON_COLUMN if authenticated_qbi else "fixture_seed"),
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
        *,
        reconcile_qbi: bool = False,
        seed_person_output: str | None = None,
    ) -> Callable[[Frame], PoolStageOutput]:
        def apply(frame: Frame) -> PoolStageOutput:
            person = frame.table("person").copy()
            transform(person)
            if reconcile_qbi:
                return _fixture_qbi_stage_output(
                    _replace_person(frame, person),
                    {"fixture_stage": True},
                )
            receipt: dict[str, object] = {"fixture_stage": True}
            if seed_person_output is not None:
                receipt["programs"] = {
                    seed_person_output: {"entity": "person"},
                }
            return PoolStageOutput(
                _replace_person(frame, person),
                receipt,
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
            reconcile_qbi=True,
        ),
        seed=stage(
            lambda person: person.__setitem__(_FIXTURE_SEED_PERSON_COLUMN, True),
            seed_person_output=_FIXTURE_SEED_PERSON_COLUMN,
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
        "schema_version": pool_tool._LEGACY_POOL_MANIFEST_SCHEMA_VERSION,
        "simulation_ready": False,
        "status": "publication_in_progress",
    }
    with pytest.raises(ValueError, match="not simulation-ready"):
        pool_tool.load_simulation_ready_us_multispine_pool_manifest(outputs.manifest)


def _stacked_main_argv(
    tmp_path: Path,
    *,
    predecessor: str | None = None,
) -> list[str]:
    arguments: list[str] = []
    for option in (
        "asec-raw-stage-h5",
        "acs-household-zip",
        "acs-person-zip",
        "acs-rent-h5",
        "puf-h5",
        "puf-source-year-csv",
    ):
        arguments.extend([f"--{option}", str(tmp_path / option)])
        arguments.extend([f"--{option}-sha256", "1" * 64])
    arguments.extend(
        [
            "--puma-ladder",
            str(tmp_path / "puma-ladder"),
            "--puma-ladder-sha256",
            "39a2ab2abeab07a88362af7ab2940e0e1d50a297c919e4bbc6fb65bab51147d8",
            "--congressional-district-vintage-crosswalk",
            str(tmp_path / "congressional-district-vintage-crosswalk"),
            "--congressional-district-vintage-crosswalk-sha256",
            "c7cb040b1f57ca2ea2adcbfe60cc2b250ca23acbc4b640cd421e766fa54c1aec",
        ]
    )
    arguments.extend(
        [
            "--sample-fraction",
            "0.01",
            "--sample-seed",
            "578",
            "--clone-attachment-fraction",
            "1.0",
            "--clone-attachment-seed",
            "579",
            "--out",
            str(tmp_path / "stacked-pool.h5"),
        ]
    )
    if predecessor is not None:
        arguments.extend(["--logbook-prev-row-digest", predecessor])
    return arguments


def _noncanonical_post_puf_authority_receipt() -> dict[str, object]:
    surface = {"person": {"model_required_boolean": ("is_pregnant",)}}
    test_authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface=surface,
        gap_fill_plan=(),
        post_puf_transfer_surface=surface,
    )
    return stacked_spine_module._authority_receipt(test_authority)


def _canonical_late_calibration_owner_receipt(
    spec: post_transfer_calibration_runtime.PostTransferCalibrationSpec,
    *,
    frame: Frame | None = None,
) -> dict[str, object]:
    if frame is not None:
        table = frame.table(spec.entity)
        channel = table[support_channel_column(spec.entity)].astype(str)
        clone_index = pd.to_numeric(
            table[support_clone_index_column(spec.entity)],
            errors="raise",
        )
        reference = (channel.eq("asec") & clone_index.eq(0)).to_numpy(dtype=bool)
        recipient = (channel.eq("acs") & clone_index.eq(0)).to_numpy(dtype=bool)
        constrained = spec.special_constraint != "none"
        application = post_transfer_calibration_runtime.apply_post_transfer_calibration(
            frame,
            entity=spec.entity,
            family=spec.family,
            target=spec.target,
            reference_rows=reference,
            recipient_rows=recipient,
            mutable_rows=recipient,
            allowed_carrier_rows=recipient if constrained else None,
            addition_candidate_rows=recipient if constrained else None,
        )
        before_values = table[spec.target].to_numpy(copy=False)
        after_values = application.frame.table(spec.entity)[spec.target].to_numpy(
            copy=False
        )
        if not np.array_equal(
            before_values.view(np.uint64),
            after_values.view(np.uint64),
        ):
            raise AssertionError(
                f"Live calibration fixture unexpectedly changed {spec.key}."
            )
        calibration = application.receipt
        scope = calibration["scope"]
        constraint: dict[str, object] = {"constraint": spec.special_constraint}
        if spec.special_constraint == "adult_care_qualifying_one_per_tax_unit":
            constraint.update(
                {
                    "qualifying_mutable_rows": scope["allowed_carrier_rows"],
                    "one_per_empty_tax_unit_addition_candidates": scope[
                        "addition_candidate_rows"
                    ],
                }
            )
        elif (
            spec.special_constraint
            == "weeks_requires_positive_unemployment_compensation"
        ):
            constraint["positive_unemployment_mutable_rows"] = scope[
                "allowed_carrier_rows"
            ]
        owner: dict[str, object] = {
            "stage": "late_transfer",
            "reference_selection": "asec_origin_clone_0",
            "recipient_selection": "acs_origin_clone_0",
            "mutable_selection": "recipient_null_before_nonnull_after",
            "reference_rows": scope["reference_rows"],
            "recipient_rows": scope["recipient_rows"],
            "mutable_rows": scope["mutable_rows"],
            "constraint": constraint,
            "context_binding": (
                stacked_spine_module._post_transfer_calibration_context_binding(
                    frame,
                    application.frame,
                    entity=spec.entity,
                    target=spec.target,
                    reference_rows=reference,
                    recipient_rows=recipient,
                    mutable_rows=recipient,
                    allowed_carrier_rows=recipient,
                    addition_candidate_rows=recipient,
                )
            ),
            "calibration": calibration,
        }
        if spec.special_constraint == "adult_care_qualifying_one_per_tax_unit":
            owner["post_reconciliation"] = {"status": "verified_no_op"}
        return owner

    values = np.asarray(
        [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0]
    )
    weights = np.asarray([2.0, 3.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    reference = np.asarray([True] * 5 + [False] * 5)
    recipient = ~reference
    constrained = spec.special_constraint != "none"
    calibration_result = (
        post_transfer_calibration_runtime.calibrate_post_transfer_values(
            values,
            weights,
            np.arange(1, len(values) + 1),
            spec=spec,
            reference_rows=reference,
            recipient_rows=recipient,
            mutable_rows=recipient,
            allowed_carrier_rows=recipient if constrained else None,
            addition_candidate_rows=recipient if constrained else None,
        )
    )
    calibration = calibration_result.receipt
    scope = calibration["scope"]
    constraint: dict[str, object] = {"constraint": spec.special_constraint}
    if spec.special_constraint == "adult_care_qualifying_one_per_tax_unit":
        constraint.update(
            {
                "qualifying_mutable_rows": scope["allowed_carrier_rows"],
                "one_per_empty_tax_unit_addition_candidates": scope[
                    "addition_candidate_rows"
                ],
            }
        )
    elif spec.special_constraint == "weeks_requires_positive_unemployment_compensation":
        constraint["positive_unemployment_mutable_rows"] = scope["allowed_carrier_rows"]
    owner: dict[str, object] = {
        "stage": "late_transfer",
        "reference_selection": "asec_origin_clone_0",
        "recipient_selection": "acs_origin_clone_0",
        "mutable_selection": "recipient_null_before_nonnull_after",
        "reference_rows": scope["reference_rows"],
        "recipient_rows": scope["recipient_rows"],
        "mutable_rows": scope["mutable_rows"],
        "constraint": constraint,
        "context_binding": {
            "scope": dict(scope),
            "weights_sha256": calibration["weights"]["sha256"],
            "live_output": {
                "reference_rows": int(reference.sum()),
                "recipient_rows": int(recipient.sum()),
                "reference_entity_ids_sha256": (
                    stacked_spine_module._post_transfer_entity_ids_sha256(
                        np.arange(1, len(values) + 1)[reference]
                    )
                ),
                "recipient_entity_ids_sha256": (
                    stacked_spine_module._post_transfer_entity_ids_sha256(
                        np.arange(1, len(values) + 1)[recipient]
                    )
                ),
                "reference_output_values_sha256": (
                    stacked_spine_module._post_transfer_float64_sha256(
                        calibration_result.values[reference],
                        boundary="synthetic reference calibration output",
                    )
                ),
                "recipient_output_values_sha256": (
                    stacked_spine_module._post_transfer_float64_sha256(
                        calibration_result.values[recipient],
                        boundary="synthetic recipient calibration output",
                    )
                ),
                "reference_weights_sha256": (
                    stacked_spine_module._post_transfer_float64_sha256(
                        weights[reference],
                        boundary="synthetic reference calibration weights",
                    )
                ),
                "recipient_weights_sha256": (
                    stacked_spine_module._post_transfer_float64_sha256(
                        weights[recipient],
                        boundary="synthetic recipient calibration weights",
                    )
                ),
            },
        },
        "calibration": calibration,
    }
    if spec.special_constraint == "adult_care_qualifying_one_per_tax_unit":
        owner["post_reconciliation"] = {"status": "verified_no_op"}
    return owner


def _canonical_pregnancy_structural_receipt() -> dict[str, object]:
    policy = acs_transfer_module.acs_transfer_execution_contract_identity(
        targets=("is_pregnant",),
        derive_schedule_d=False,
    )["structural_target_policies"]["is_pregnant"]
    return {
        "policy_sha256": policy["sha256"],
        "source_person_key": "person_source_id",
        "source_persons_checked": 1,
        "physical_rows_checked": 1,
        "clone_rows_checked": 0,
        "donor_rows_checked": 1,
        "qrf_draw_source_persons": 0,
        "qrf_draw_rows": 0,
        "qrf_fanout_rows": 0,
        "preexisting_value_fanout_rows": 0,
        "ineligible_rows_assigned_false": 0,
        "donor_preexisting_domain_violation_rows": 0,
        "recipient_preexisting_domain_violation_rows": 0,
        "preexisting_clone_disagreement_source_persons": 0,
        "inconsistent_eligibility_source_persons": 0,
        "maximum_clones_per_source_person": 1,
        "final_incomplete_rows": 0,
        "final_domain_violation_rows": 0,
        "final_clone_disagreement_source_persons": 0,
        "status": "verified",
    }


def _canonical_late_transfer_receipt(
    pool_tool: ModuleType,
    *,
    authority: Mapping[str, object] | None = None,
    frame: Frame | None = None,
) -> dict[str, object]:
    canonical_family = {
        (entity, target): family
        for entity, families in (
            pool_tool.CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE.items()
        )
        for family, targets in families.items()
        for target in targets
    }
    groups: dict[str, object] = {}
    targets: dict[str, object] = {}
    late_specs = {
        spec.key: spec
        for spec in post_transfer_calibration_runtime.POST_TRANSFER_CALIBRATION_SPECS.values()
        if spec.stage == "late_transfer"
    }
    policy_sha256 = (
        post_transfer_calibration_runtime.post_transfer_calibration_policy_identity()[
            "sha256"
        ]
    )
    for group in pool_tool.CANONICAL_US_LATE_TRANSFER_GROUPS:
        group_targets = {
            f"{group.entity}/{group.family}/{target}": {
                "authorized_null_rows": 0,
                "imputed_rows": 0,
                "unmodeled_rows": 0,
                "residual_null_rows": 0,
            }
            for target in group.targets
        }
        pregnancy_key = f"{group.entity}/{group.family}/is_pregnant"
        if pregnancy_key in group_targets:
            group_targets[pregnancy_key]["structural_policy"] = (
                _canonical_pregnancy_structural_receipt()
            )
        calibrated_keys = sorted(set(group_targets) & set(late_specs))
        for key in calibrated_keys:
            group_targets[key]["post_transfer_calibration"] = (
                _canonical_late_calibration_owner_receipt(
                    late_specs[key],
                    frame=frame,
                )
            )
        groups[group.name] = {
            "producer": group.name,
            "ordered_targets": list(group.targets),
            "targets": group_targets,
            "post_transfer_calibration": {
                "policy_sha256": policy_sha256,
                "target_count": len(calibrated_keys),
                "targets": calibrated_keys,
            },
        }
        for target in group.targets:
            targets[
                f"{group.entity}/{canonical_family[(group.entity, target)]}/{target}"
            ] = group_targets[f"{group.entity}/{group.family}/{target}"]
    return {
        "fixture": "post_puf_transfer",
        "authority": dict(
            pool_tool.stacked_spine_authority_receipt()
            if authority is None
            else authority
        ),
        "producer_schedule": pool_tool._json_ready(
            pool_tool.us_late_producer_schedule_receipt()
        ),
        "producer_execution_order": [
            producer
            for producer in stacked_spine_module.CANONICAL_US_LATE_PRODUCER_SCHEDULE.order
            if producer != stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE
        ],
        "groups": groups,
        "targets": targets,
        "completion": {
            "status": "complete",
            "group_count": 19,
            "target_count": 70,
            "residual_null_rows": 0,
        },
    }


def _canonical_late_dag_receipt(
    pool_tool: ModuleType,
    *,
    authority: Mapping[str, object] | None = None,
    output_frame_sha256: str = "f" * 64,
    frame: Frame | None = None,
) -> dict[str, object]:
    schedule = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_SCHEDULE
    schedule_receipt = pool_tool._json_ready(
        pool_tool.us_late_producer_schedule_receipt()
    )
    source_order = [
        producer.removeprefix("source:")
        for producer in schedule.order
        if producer.startswith("source:")
    ]
    cps_source_evidence = {"fixture": "shared_cps_source_evidence"}
    source_receipts = {
        operator: {
            "phase": "post_clone",
            "operator_order": [operator],
            "suboperators": [{"operator": operator, "order_index": 0}],
            "cps_source_evidence": cps_source_evidence,
        }
        for operator in source_order
    }
    source_completion = {
        "phase": "post_clone",
        "operator_order": source_order,
        "suboperators": [
            {"operator": operator, "order_index": index}
            for index, operator in enumerate(source_order)
        ],
        "cps_source_evidence": cps_source_evidence,
        "deferred_transfer_inputs": {
            "inputs": {
                column: {}
                for column in (
                    "bank_account_assets",
                    "bond_assets",
                    "stock_assets",
                )
            }
        },
    }
    transfer = _canonical_late_transfer_receipt(
        pool_tool,
        authority=authority,
        frame=frame,
    )
    input_frame_sha256 = "e" * 64
    previous_sha256 = stacked_spine_module._late_execution_genesis_sha256(
        producer_schedule_sha256=schedule_receipt["payload_sha256"],
        input_frame_sha256=input_frame_sha256,
    )
    execution = []
    for index, producer_name in enumerate(schedule.order):
        contract = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_REGISTRY[
            producer_name
        ]
        declared_inputs = []
        if contract.kind == "acs_earnings_universe":
            available = (
                stacked_spine_module._late_acs_earnings_universe_resource_receipts()
            )
        elif contract.kind == "primary_puf":
            available: dict[str, object] = (
                stacked_spine_module.stacked_late_primary_resource_receipts(
                    pd.DataFrame({"fixture_donor": [1.0]}),
                    primary_qrf_checkpoint_identity_sha256="c" * 64,
                    clone_attachment_fraction=1.0,
                    clone_attachment_seed=578,
                    seed=0,
                    n_estimators=100,
                    fit_records_enabled=True,
                    tail_bound_diagnostics_enabled=True,
                )
            )
        elif contract.kind == "post_clone_source":
            available = stacked_spine_module._late_source_resource_receipts(
                producer_name=producer_name,
            )
        elif contract.kind == "source_finalizer":
            available = {
                f"person.@source_receipt:{operator}": (
                    stacked_spine_module._late_available_input_receipt(
                        producer=producer_name,
                        entity="person",
                        column=f"@source_receipt:{operator}",
                        rows=1,
                        binding={
                            "resource_kind": "source_operator_receipt",
                            "schema_version": 1,
                            "source_operator": operator,
                            "source_receipt_sha256": (
                                stacked_spine_module._canonical_sha256(
                                    source_receipts[operator]
                                )
                            ),
                        },
                    )
                )
                for operator in source_order
            }
            available.update(
                stacked_spine_module._late_source_finalizer_resource_receipts()
            )
        elif contract.kind == "late_transfer":
            group = next(
                group
                for group in pool_tool.CANONICAL_US_LATE_TRANSFER_GROUPS
                if group.name == producer_name
            )
            available = stacked_spine_module._late_transfer_resource_receipts(
                group_name=group.name,
                entity=group.entity,
                family=group.family,
                targets=group.targets,
                seed=0,
                n_estimators=100,
                max_targets_per_fit=(
                    stacked_spine_module.DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
                ),
                target_bank=None,
            )
        else:
            available = {}
        for item in contract.inputs:
            alternatives = []
            for alternative in item.alternatives:
                physical_evidence = []
                for column in alternative:
                    is_virtual = (
                        column.column.startswith("@")
                        and column.column != "@resolved_weight"
                        and column.entity != "frame"
                    )
                    key = f"{column.entity}.{column.column}"
                    resource_receipt = available.get(key) if is_virtual else None
                    present = not is_virtual or resource_receipt is not None
                    physical_evidence.append(
                        {
                            "entity": column.entity,
                            "column": column.column,
                            "value_kind": column.value_kind,
                            "required_scope": item.required_scope,
                            "scope_rows": 1,
                            "missing_rows": 0 if present else 1,
                            "invalid_rows": 0,
                            "status": "present" if present else "absent",
                            "content_sha256": (
                                stacked_spine_module._canonical_sha256(resource_receipt)
                                if resource_receipt is not None
                                else "a" * 64
                            ),
                            **(
                                {"weight_kind": "household_weight"}
                                if column.column == "@resolved_weight"
                                else {}
                            ),
                        }
                    )
                alternatives.append(physical_evidence)
            evidence = {"alternatives": alternatives}
            evidence["sha256"] = stacked_spine_module._canonical_sha256(evidence)
            declared_inputs.append(
                {
                    "entity": item.entity,
                    "column": item.column,
                    "required_scope": item.required_scope,
                    "producing_stage": item.producing_stage,
                    "unfilled_rows": 0,
                    "invalid_rows": 0,
                    "evidence": evidence,
                }
            )
        if contract.kind == "acs_earnings_universe":
            producer_receipt = {"fixture": "acs_earnings_universe"}
        elif contract.kind == "primary_puf":
            producer_receipt: Mapping[str, object] = {
                "fixture": "primary_puf",
                "primary_resource_receipts_sha256": (
                    stacked_spine_module._canonical_sha256(available)
                ),
            }
        elif contract.kind == "post_clone_source":
            producer_receipt = source_receipts[producer_name.removeprefix("source:")]
        elif contract.kind == "source_finalizer":
            producer_receipt = source_completion
        else:
            producer_receipt = transfer["groups"][producer_name]
        output_surface = [
            {
                "entity": output.entity,
                "column": output.column,
                "coverage_scope": output.coverage_scope,
                "status": "present",
                "content_sha256": (
                    stacked_spine_module._canonical_sha256(producer_receipt)
                    if output.column.startswith("@source_receipt:")
                    else "b" * 64
                ),
                **({} if output.entity == "frame" else {"scope_rows": 1}),
                **(
                    {"weight_kind": "household_weight"}
                    if output.column == "@resolved_weight"
                    else {}
                ),
            }
            for output in contract.outputs
        ]
        row: dict[str, object] = {
            "execution_index": index,
            "producer": producer_name,
            "kind": contract.kind,
            "declared_inputs": declared_inputs,
            "declared_absence_receipts": {},
            "available_input_receipts": available,
            "input_surface_sha256": stacked_spine_module._canonical_sha256(
                declared_inputs
            ),
            "output_surface": output_surface,
            "output_surface_sha256": stacked_spine_module._canonical_sha256(
                output_surface
            ),
            "producer_receipt": producer_receipt,
            "producer_receipt_sha256": stacked_spine_module._canonical_sha256(
                producer_receipt
            ),
            "previous_execution_sha256": previous_sha256,
            "status": "complete",
        }
        row["sha256"] = stacked_spine_module._canonical_sha256(row)
        previous_sha256 = row["sha256"]
        execution.append(row)
    receipt: dict[str, object] = {
        "version": stacked_spine_module.US_LATE_PRODUCER_RECEIPT_SCHEMA_VERSION,
        "producer_schedule": schedule_receipt,
        "input_frame_sha256": input_frame_sha256,
        "output_frame_sha256": output_frame_sha256,
        "execution_chain_sha256": previous_sha256,
        "execution": execution,
        "source_completion": source_completion,
        "post_puf_transfer": transfer,
    }
    receipt["sha256"] = stacked_spine_module._canonical_sha256(receipt)
    return receipt


def _canonical_late_impute_receipts(
    pool_tool: ModuleType,
    *,
    authority: Mapping[str, object] | None = None,
) -> dict[str, object]:
    dag = _canonical_late_dag_receipt(pool_tool, authority=authority)
    return {
        "source_operator_chain": {
            "late_dag_completion": dag["source_completion"],
        },
        "stacked_late_producer_dag": dag,
        "stacked_post_puf_transfer": dag["post_puf_transfer"],
    }


def _authorized_late_impute_fixture(
    pool_tool: ModuleType,
    frame: Frame,
    *,
    authority: Mapping[str, object] | None = None,
) -> tuple[Frame, dict[str, object], str]:
    """Bind one structurally signed synthetic DAG proof to a live fixture frame."""

    tables = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    for entity, table in tables.items():
        if support_channel_column(entity) not in table:
            table[support_channel_column(entity)] = np.resize(
                np.asarray(["asec", "acs"], dtype=object),
                len(table),
            )
        if support_clone_index_column(entity) not in table:
            table[support_clone_index_column(entity)] = np.zeros(
                len(table), dtype=np.int64
            )
    for (
        spec
    ) in post_transfer_calibration_runtime.POST_TRANSFER_CALIBRATION_SPECS.values():
        if spec.stage == "late_transfer" and spec.target not in tables[spec.entity]:
            tables[spec.entity][spec.target] = np.ones(
                len(tables[spec.entity]), dtype=np.float64
            )
    person = tables["person"]
    person["unemployment_compensation"] = np.ones(len(person), dtype=np.float64)
    person["is_incapable_of_self_care"] = pd.Series(
        True,
        index=person.index,
        dtype="boolean",
    )
    person["tax_unit_role_input"] = pd.Series(
        "DEPENDENT",
        index=person.index,
        dtype="string",
    )
    tables.update({name: frame.link(name) for name in frame.links})
    frame = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    dag = _canonical_late_dag_receipt(
        pool_tool,
        authority=authority,
        output_frame_sha256=stacked_spine_module._late_frame_content_sha256(frame),
        frame=frame,
    )
    authorized, transition_authority_sha256 = (
        stacked_spine_module._bind_late_producer_transition_authority(frame, dag)
    )
    return (
        authorized,
        {
            "source_operator_chain": {
                "late_dag_completion": dag["source_completion"],
            },
            "stacked_late_producer_dag": dag,
            "stacked_post_puf_transfer": dag["post_puf_transfer"],
        },
        transition_authority_sha256,
    )


def _install_stacked_entrypoint_stubs(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    terminal: str,
    post_puf_authority: Mapping[str, object] | None = None,
    real_geography_assignment: bool = False,
) -> tuple[list[str], int]:
    order: list[str] = []
    verified = _verified_inputs_fixture(pool_tool, tmp_path / "pins")
    source_manifest = pool_tool.load_acs_source_manifest()
    puf_donor = pd.DataFrame({"fixture": np.arange(7)})
    loaded = pool_tool._LoadedInputs(
        asec=_many_household_source_frame(
            state_fips="06" if real_geography_assignment else None,
        ),
        acs=_many_household_source_frame(
            measured_offset=1_000.0,
            state_fips="06" if real_geography_assignment else None,
            puma="0600100" if real_geography_assignment else None,
        ),
        acs_rent_donor=pd.DataFrame({"fixture": [1.0]}),
        puf_donor=puf_donor,
        asec_raw_stage_checkpoint={"artifact": "fixture-raw-stage"},
        acs_build={"artifact": "fixture-acs-build"},
        acs_native_inputs={},
        puf_donor_build={"artifact": "fixture-puf-build"},
    )
    monkeypatch.setattr(
        pool_tool,
        "_verify_inputs",
        lambda _args, _outputs: (verified, source_manifest),
    )
    monkeypatch.setattr(
        pool_tool,
        "_load_inputs",
        lambda _args, *, acs_source_manifest: loaded,
    )
    monkeypatch.setattr(
        pool_tool,
        "load_acs_2022_rent_donor",
        lambda _path: loaded.acs_rent_donor,
    )
    monkeypatch.setattr(
        pool_tool,
        "_load_puf_donor",
        lambda _args: (loaded.puf_donor, loaded.puf_donor_build),
    )
    monkeypatch.setattr(pool_tool, "_git_code_pin", lambda: "a" * 40)
    crosswalk = pd.DataFrame(
        {
            "source_geography_id": ["5001700US0601"],
            "target_geography_id": ["5001900US0601"],
            "weight": [1.0],
        }
    )
    monkeypatch.setattr(
        pool_tool,
        "load_congressional_district_vintage_crosswalk",
        lambda _path: crosswalk,
    )
    fixture_ladder = (
        _fixture_puma_ladder(pool_tool) if real_geography_assignment else object()
    )
    monkeypatch.setattr(
        pool_tool,
        "load_us_puma_ladder",
        lambda _path: fixture_ladder,
    )
    fixture_puma_gate = GateResult(
        name="us_puma_ladder",
        passed=True,
        details={"fixture": True},
    )
    monkeypatch.setattr(
        pool_tool,
        "us_puma_ladder_gate",
        lambda *_args, **_kwargs: fixture_puma_gate,
    )

    real_stack = pool_tool.assemble_stacked_spine

    def stack(*args, **kwargs):
        order.append("stack")
        return real_stack(*args, **kwargs)

    monkeypatch.setattr(pool_tool, "assemble_stacked_spine", stack)

    real_assign_geography = pool_tool._assign_stacked_household_geography

    def assign_geography(frame: Frame, **kwargs: object):
        order.append("geography")
        if real_geography_assignment:
            return real_assign_geography(frame, **kwargs)
        assigned = _with_fixture_household_geography(frame)
        receipt = _fixture_geography_assignment_receipt(
            pool_tool,
            assigned,
            gate=fixture_puma_gate,
        )
        return assigned, receipt, (601,)

    monkeypatch.setattr(
        pool_tool,
        "_assign_stacked_household_geography",
        assign_geography,
    )

    real_build_stacked_pool = pool_tool.build_stacked_pool

    def build_stacked_pool(*args, **kwargs):
        order.append("build_stacked_pool")
        return real_build_stacked_pool(*args, **kwargs)

    monkeypatch.setattr(pool_tool, "build_stacked_pool", build_stacked_pool)

    def fixture_pre_clone_source_chain(
        frame: Frame,
        *,
        phase: str,
        operator_names: tuple[str, ...],
        operators: Mapping[str, object],
        **_kwargs: object,
    ) -> PoolStageOutput:
        order.append("prepare")
        assert phase == "pre_clone"
        assert operator_names
        assert set(operator_names) == set(operators)
        return PoolStageOutput(
            _with_fixture_pre_clone_strike_benefits(frame),
            {"fixture": "pre_clone_source_chain"},
        )

    monkeypatch.setattr(
        multispine_pool_module,
        "_run_source_operator_chain",
        fixture_pre_clone_source_chain,
    )
    directions = (
        GapFillDirection(
            name="asec_survey_to_acs",
            recipient_channel="acs",
            donor_channel="asec",
            target_families={
                "person": {
                    "source_operator_cps_carried": ("strike_benefits",),
                }
            },
        ),
    )
    monkeypatch.setattr(pool_tool, "stacked_gap_fill_plan", lambda: directions)

    def gap_fill(frame: Frame, **kwargs):
        order.append("gap")
        assert set(kwargs["target_banks"]) == {"asec_survey_to_acs"}
        counts = stacked_spine_module._verify_gap_fill_activation_authority(
            frame,
            direction=directions[0],
        )
        assert counts == {
            ("person", "strike_benefits"): {
                "authorized_null_rows": 1,
                "recipient_rows": 1,
                "donor_rows": 1,
            }
        }
        return SimpleNamespace(
            frame=frame,
            receipt={"fixture": "gap"},
            transfer_results={},
        )

    monkeypatch.setattr(pool_tool, "gap_fill_stacked_spine", gap_fill)
    monkeypatch.setattr(
        pool_tool,
        "weights_audit_gate",
        lambda _records: GateResult(name="fixture_weights", passed=True),
    )
    monkeypatch.setattr(
        pool_tool,
        "validate_puf_capital_gains_tail_manifest",
        lambda _manifest: None,
    )

    observed_primary_qrf_binding: dict[str, object] = {}

    def puf_pass(frame: Frame, donor: pd.DataFrame, **kwargs):
        order.append("puf")
        assert donor is puf_donor
        assert len(donor) == 7
        assert kwargs["clone_attachment_fraction"] == 1.0
        assert kwargs["clone_attachment_seed"] == 579
        primary_binding = kwargs["primary_qrf_input_binding"]
        stacked_spine_module._validate_stacked_late_primary_checkpoint_input_binding(
            primary_binding,
            boundary="tool wiring fixture",
        )
        observed_primary_qrf_binding.update(primary_binding)
        if terminal == "error":
            raise RuntimeError("fixture stacked error")
        checkpoint_dir = Path(kwargs["primary_qrf_checkpoint_dir"])
        pool_tool._atomic_write_json(
            checkpoint_dir / pool_tool.PRIMARY_QRF_MANIFEST_FILENAME,
            {"fixture": "primary-qrf"},
        )
        return SimpleNamespace(
            frame=frame,
            receipt={
                "primary_puf_qrf": {
                    "mode": "checkpoint_chain",
                    "resume_status": "initialized",
                },
                "puf_capital_gains_tail_transfer": {"fixture": "tail"},
                "tail_status": "applied",
                "primary_resource_receipts_sha256": (
                    stacked_spine_module._canonical_sha256(
                        primary_binding["primary_resource_receipts"]
                    )
                ),
            },
        )

    monkeypatch.setattr(pool_tool, "run_stacked_puf_pass", puf_pass)

    def late_producer_dag(frame: Frame, **kwargs: object):
        primary_puf_result = kwargs["primary_puf_producer"](frame)
        order.append("late_producer_dag")
        assert set(kwargs["primary_resource_receipts"]) == {
            "tax_unit.@puf_donor_tax_units",
            "tax_unit.@primary_qrf_checkpoint",
            "tax_unit.@primary_puf_execution_config",
        }
        assert (
            observed_primary_qrf_binding["primary_resource_receipts"]
            == kwargs["primary_resource_receipts"]
        )
        primary_config = kwargs["primary_resource_receipts"][
            "tax_unit.@primary_puf_execution_config"
        ]["binding"]
        assert primary_config["clone_attachment"] == {
            "fraction": 1.0,
            "seed": 579,
            "support_channels": ["asec", "puf_tax_detail"],
            "puf_clone_index": 1,
        }
        assert primary_config["qrf"]["seed"] == pool_tool.POOL_RANDOM_SEED
        assert (
            primary_config["qrf"]["n_estimators"] == pool_tool._PRIMARY_QRF_N_ESTIMATORS
        )
        target_banks = kwargs["target_banks"]
        assert isinstance(target_banks, Mapping)
        assert set(target_banks) == {
            group.name for group in pool_tool.CANONICAL_US_LATE_TRANSFER_GROUPS
        }
        schedule_sha256 = pool_tool.us_late_producer_schedule_receipt()[
            "payload_sha256"
        ]
        dag_sha256 = pool_tool.us_late_producer_schedule_receipt()["schedule_sha256"]
        for group in pool_tool.CANONICAL_US_LATE_TRANSFER_GROUPS:
            bank = target_banks[group.name]
            assert bank.root.parts[-3:] == (
                "late_producer_dag",
                group.entity,
                group.family,
            )
            assert bank._identity["late_producer_dag_sha256"] == dag_sha256
            assert bank._identity["late_producer_schedule_sha256"] == schedule_sha256
            assert bank._identity["late_producer"] == {
                "name": group.name,
                "entity": group.entity,
                "family": group.family,
                "ordered_targets": list(group.targets),
            }
        late_tables = {
            entity: primary_puf_result.frame.table(entity).copy(deep=True)
            for entity in primary_puf_result.frame.entities
        }
        for (
            spec
        ) in post_transfer_calibration_runtime.POST_TRANSFER_CALIBRATION_SPECS.values():
            if (
                spec.stage == "late_transfer"
                and spec.target not in late_tables[spec.entity]
            ):
                late_tables[spec.entity][spec.target] = np.ones(
                    len(late_tables[spec.entity]),
                    dtype=np.float64,
                )
        late_person = late_tables["person"]
        late_person["unemployment_compensation"] = np.ones(
            len(late_person),
            dtype=np.float64,
        )
        late_person["is_incapable_of_self_care"] = pd.Series(
            True,
            index=late_person.index,
            dtype="boolean",
        )
        late_person["tax_unit_role_input"] = pd.Series(
            "DEPENDENT",
            index=late_person.index,
            dtype="string",
        )
        late_tables.update(
            {
                name: primary_puf_result.frame.link(name)
                for name in primary_puf_result.frame.links
            }
        )
        late_frame = Frame(
            late_tables,
            primary_puf_result.frame.schema,
            {
                entity: primary_puf_result.frame.weights_for(entity)
                for entity in primary_puf_result.frame.weighted_entities
            },
            primary_puf_result.frame.strata,
            mass_log=primary_puf_result.frame.mass_log,
            metadata=primary_puf_result.frame.metadata,
        )
        dag_receipt = _canonical_late_dag_receipt(
            pool_tool,
            authority=post_puf_authority,
            output_frame_sha256=stacked_spine_module._late_frame_content_sha256(
                late_frame
            ),
            frame=late_frame,
        )
        authorized_frame, transition_authority_sha256 = (
            stacked_spine_module._bind_late_producer_transition_authority(
                late_frame,
                dag_receipt,
            )
        )
        return SimpleNamespace(
            frame=authorized_frame,
            receipt=dag_receipt,
            primary_puf_result=primary_puf_result,
            source_completion_receipt=dag_receipt["source_completion"],
            transfer_result=SimpleNamespace(fit_records=()),
            transition_authority_sha256=transition_authority_sha256,
        )

    monkeypatch.setattr(
        pool_tool,
        "run_stacked_late_producer_dag",
        late_producer_dag,
    )
    monkeypatch.setattr(
        pool_tool,
        "assert_stacked_tail_cells_preserved",
        lambda _frame, _manifest: {"passed": True},
    )

    def tail_prepare(frame: Frame):
        order.append("tail_prepare")
        return frame, {"fixture": "tail_prepare"}

    monkeypatch.setattr(pool_tool, "prepare_stacked_tail_derivation", tail_prepare)

    def identity_stage(name: str):
        def stage(frame: Frame):
            order.append(name)
            if name == "derive":
                return _fixture_qbi_stage_output(frame, {"fixture": name})
            return PoolStageOutput(frame, {"fixture": name})

        return stage

    monkeypatch.setattr(
        pool_tool,
        "derive_multispine_pool_inputs",
        identity_stage("derive"),
    )
    monkeypatch.setattr(
        pool_tool,
        "seed_multispine_pool_inputs",
        identity_stage("seed"),
    )

    def simulate(frame: Frame):
        order.append("simulate")
        person = frame.table("person").copy()
        person["ssi"] = 0.0
        return PoolStageOutput(_replace_person(frame, person), {"fixture": "simulate"})

    monkeypatch.setattr(
        pool_tool,
        "materialize_multispine_agreement_outputs",
        simulate,
    )

    def completeness(
        _frame: Frame,
        *,
        tail_manifest: Mapping[str, object],
    ) -> GateResult:
        order.append("completeness")
        assert tail_manifest == {"fixture": "tail"}
        return GateResult(name="fixture_completeness", passed=True)

    def battery(
        _frame: Frame,
        *,
        tail_manifest: Mapping[str, object],
    ) -> GateResult:
        order.append("battery")
        assert tail_manifest == {"fixture": "tail"}
        if terminal == "red":
            return GateResult(
                name="fixture_battery",
                passed=False,
                failures=("fixture terminal failure",),
            )
        return GateResult(name="fixture_battery", passed=True)

    monkeypatch.setattr(pool_tool, "stacked_completeness_gate", completeness)
    monkeypatch.setattr(pool_tool, "by_origin_battery", battery)
    real_publish = pool_tool._write_stacked_outputs

    def publish(*args, **kwargs):
        order.append("publish")
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(pool_tool, "_write_stacked_outputs", publish)
    return order, len(puf_donor)


def _receipt_file_from_reference(reference: str) -> Path:
    """Map one exported ``local://`` receipt reference back to a real file.

    Rows never embed host-absolute paths, so tests reconstruct the file
    location from the reference's anchor: ``~/`` means home, and the
    stripped-absolute fallback (the only form pytest tmp paths produce)
    re-roots at ``/``.
    """

    location = reference.split("#", maxsplit=1)[0]
    assert location.startswith("local://")
    tail = location.removeprefix("local://")
    assert not tail.startswith("/")
    if tail.startswith("~/"):
        return Path.home() / tail[2:]
    return Path("/") / tail


__all__ = [name for name in globals() if not name.startswith("__")]
