"""Kernels for the stacked US pool's post-transfer graph."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

import microcosm.build.us_runtime.capital_gain_distributions as distributions_module
import microcosm.build.us_runtime.multispine_pool as multispine_pool_module
import microcosm.build.us_runtime.qbi_inputs as qbi_inputs_module
import microcosm.build.us_runtime.stacked_spine as stacked_spine_module
import microcosm.build.us_runtime.take_up as take_up_module
import microcosm.build.us_runtime.take_up_contract as take_up_contract_module
import microcosm.frame.adapters.policyengine_us as policyengine_us_module
from microcosm.frame import US_SCHEMA, Frame, MassChangeRecord
from microcosm.graph import (
    Capabilities,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Numeric,
    SeedSource,
    StructuralDelta,
    load_source,
    source_hash,
)
from microcosm.graph.population import dtype_for_token

from .capital_gain_distributions import (
    capital_gain_distribution_shares_asset_identity,
)
from .multispine_pool import (
    POOL_ENGINE_INPUT_PROJECTION_CONTRACT,
    POOL_RANDOM_SEED,
    POOL_REMAINING_STAGE_INPUT_MANIFEST_SHA256,
    POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE,
    POOL_SSI_DEPENDENCY_CONTRACT,
    POOL_TIME_PERIOD,
    PoolStageOutput,
    derive_multispine_pool_inputs,
    materialize_multispine_agreement_outputs,
    seed_multispine_pool_inputs,
)
from .qbi_inputs import us_qbi_reconciliation_contract_identity
from .stacked_spine import prepare_stacked_tail_derivation
from .take_up_contract import load_take_up_contract, take_up_contract_identity

__all__ = [
    "US_POST_TRANSFER_CONTEXT_FILENAME",
    "USPostTransferCreateKernel",
    "USPostTransferDeriveKernel",
    "USPostTransferIdentityKernel",
    "USPostTransferMaterializeKernel",
    "USPostTransferPrepareKernel",
    "USPostTransferSeedKernel",
    "build_us_post_transfer_registry",
]


US_POST_TRANSFER_CONTEXT_FILENAME = "frame.json"

_COMPUTE = Capabilities(
    determinism=Determinism.DETERMINISTIC,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.NONE,
)
_DERIVE = Capabilities(
    determinism=Determinism.DETERMINISTIC,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.NONE,
    dependencies=("policyengine-us",),
)
_SEEDED = Capabilities(
    determinism=Determinism.SEEDED,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.PARAM,
    dependencies=("policyengine-us",),
)
_ENGINE_COMPUTE = Capabilities(
    determinism=Determinism.DETERMINISTIC,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.NONE,
    dependencies=("policyengine-us",),
)
_CREATE = Capabilities(
    determinism=Determinism.DETERMINISTIC,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.NONE,
    structural=StructuralDelta.CREATE,
)
_FILTER = Capabilities(
    determinism=Determinism.DETERMINISTIC,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.NONE,
    structural=StructuralDelta.FILTER,
)


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(
        "US post-transfer receipt contains a non-JSON value: "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def _context_document(source: Path) -> Mapping[str, Any]:
    path = source / US_POST_TRANSFER_CONTEXT_FILENAME
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read US post-transfer context at {path}.") from error
    if not isinstance(document, Mapping):
        raise ValueError("US post-transfer frame.json must be an object.")
    expected = {"schema_version", "columns", "metadata", "mass_log"}
    if set(document) != expected or document.get("schema_version") != 1:
        raise ValueError(
            "US post-transfer frame.json has an unsupported context schema."
        )
    if not isinstance(document["columns"], Mapping):
        raise ValueError("US post-transfer frame.json columns must be an object.")
    if not isinstance(document["metadata"], Mapping):
        raise ValueError("US post-transfer frame.json metadata must be an object.")
    if not isinstance(document["mass_log"], list):
        raise ValueError("US post-transfer frame.json mass_log must be an array.")
    return document


def _mass_log(document: Mapping[str, Any]) -> tuple[MassChangeRecord, ...]:
    records: list[MassChangeRecord] = []
    expected = {
        "entity",
        "old_total",
        "new_total",
        "declared_factor",
        "reason",
    }
    for index, raw in enumerate(document["mass_log"]):
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise ValueError(f"US post-transfer mass_log record {index} is malformed.")
        records.append(
            MassChangeRecord(
                entity=str(raw["entity"]),
                old_total=float(raw["old_total"]),
                new_total=float(raw["new_total"]),
                declared_factor=(
                    None
                    if raw["declared_factor"] is None
                    else float(raw["declared_factor"])
                ),
                reason=str(raw["reason"]),
            )
        )
    return tuple(records)


def _ordered_tables(
    tables: Mapping[str, pd.DataFrame],
    document: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    raw_columns = document["columns"]
    result: dict[str, pd.DataFrame] = {}
    for entity in US_SCHEMA.entities:
        if entity not in tables:
            raise ValueError(
                f"US post-transfer context lacks the {entity!r} entity table."
            )
        raw_order = raw_columns.get(entity)
        if not isinstance(raw_order, list) or any(
            not isinstance(column, str) for column in raw_order
        ):
            raise ValueError(
                f"US post-transfer frame.json has no column order for {entity!r}."
            )
        table = tables[entity]
        order = [column for column in raw_order if column in table]
        order.extend(column for column in table.columns if column not in order)
        result[entity] = table.loc[:, order].copy(deep=True)
    return result


def _frame_from_context(context: KernelContext) -> Frame:
    document = _context_document(context.sources["stacked"])
    missing = set(US_SCHEMA.entities) - set(context.tables)
    if missing:
        raise ValueError(
            f"US stage {context.node.id!r} lacks declared tables {sorted(missing)}."
        )
    try:
        household_weights = context.weights["household"]
    except KeyError as error:
        raise ValueError(
            f"US stage {context.node.id!r} lacks resolved household weights."
        ) from error
    return Frame(
        _ordered_tables(context.tables, document),
        US_SCHEMA,
        {"household": household_weights},
        context.strata.copy(deep=True),
        mass_log=_mass_log(document),
        metadata=dict(document["metadata"]),
    )


def _id_index(frame: Frame, entity: str) -> pd.Index:
    id_column = frame.schema.entity_id_column(entity)
    return pd.Index(
        frame.table(entity)[id_column].to_numpy(copy=True),
        name=id_column,
    )


def _owned_columns(frame: Frame, context: KernelContext) -> Mapping:
    columns: dict[tuple[str, str], pd.Series] = {}
    for owned in context.node.outputs:
        values = frame.table(owned.entity)[owned.column].copy(deep=True)
        dtype = dtype_for_token(owned.dtype)
        if values.dtype != dtype:
            values = values.astype(dtype)
        values.index = _id_index(frame, owned.entity)
        values.name = owned.column
        columns[(owned.entity, owned.column)] = values
    return MappingProxyType(columns)


def _require_params(context: KernelContext, expected: Mapping[str, object]) -> None:
    actual = dict(context.params)
    if actual != dict(expected):
        raise ValueError(
            f"US stage {context.node.id!r} parameters differ from its runtime "
            f"contract: expected={dict(expected)!r}, actual={actual!r}."
        )


def _stage_result(
    output: PoolStageOutput,
    context: KernelContext,
) -> KernelResult:
    return KernelResult(
        columns=_owned_columns(output.frame, context),
        receipt=_json_ready(output.receipt),
    )


class USPostTransferCreateKernel(KernelBase):
    """Load the serialized stacked pool and restore validation context."""

    ref = "us.post_transfer.create@1"
    capabilities = _CREATE

    def implementation_hash(self) -> str:
        return source_hash(type(self))

    def run(self, context: KernelContext) -> KernelResult:
        _require_params(context, {"context_schema_version": 1})
        source = context.sources["stacked"]
        loaded = load_source("csv-tables", source)
        document = _context_document(source)
        tables = {
            entity: loaded.table(entity).copy(deep=True) for entity in loaded.entities
        }
        for owned in context.node.outputs:
            table = tables[owned.entity]
            if owned.column not in table:
                raise ValueError(
                    "US post-transfer source is missing declared cell "
                    f"{owned.entity}.{owned.column}."
                )
            table[owned.column] = table[owned.column].astype(
                dtype_for_token(owned.dtype)
            )
        frame = Frame(
            _ordered_tables(tables, document),
            US_SCHEMA,
            {"household": loaded.weights_for("household")},
            loaded.strata.copy(deep=True),
            mass_log=_mass_log(document),
            metadata=dict(document["metadata"]),
        )
        return KernelResult(frame=frame)


class USPostTransferIdentityKernel(KernelBase):
    """Keep every person while opening a rewrite-capable population version."""

    ref = "us.post_transfer.identity@1"
    capabilities = _FILTER

    def run(self, context: KernelContext) -> KernelResult:
        person = context.tables[US_SCHEMA.person_entity]
        id_column = US_SCHEMA.person_id_column
        ids = pd.Index(person[id_column].to_numpy(copy=True), name=id_column)
        return KernelResult(keep=pd.Series(True, index=ids, dtype="bool"))


class USPostTransferPrepareKernel(KernelBase):
    """Wrap `prepare_stacked_tail_derivation` unchanged."""

    ref = "us.post_transfer.prepare@1"
    capabilities = _COMPUTE

    def implementation_hash(self) -> str:
        return source_hash(type(self), stacked_spine_module)

    def run(self, context: KernelContext) -> KernelResult:
        _require_params(context, {"stage": "prepare_stacked_tail_derivation"})
        prepared, receipt = prepare_stacked_tail_derivation(
            _frame_from_context(context)
        )
        return KernelResult(
            columns=_owned_columns(prepared, context),
            receipt=_json_ready(receipt),
        )


class USPostTransferDeriveKernel(KernelBase):
    """Wrap `derive_multispine_pool_inputs` unchanged."""

    ref = "us.post_transfer.derive@1"
    capabilities = _DERIVE

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            multispine_pool_module,
            qbi_inputs_module,
            distributions_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        schedule_identity = capital_gain_distribution_shares_asset_identity()
        qbi_identity = us_qbi_reconciliation_contract_identity()
        _require_params(
            context,
            {
                "stage": "derive_multispine_pool_inputs",
                "remaining_stage_manifest_sha256": (
                    POOL_REMAINING_STAGE_INPUT_MANIFEST_SHA256
                ),
                "schedule_d_asset_sha256": str(schedule_identity["asset_sha256"]),
                "qbi_contract_version": int(qbi_identity["version"]),
            },
        )
        return _stage_result(
            derive_multispine_pool_inputs(_frame_from_context(context)),
            context,
        )


class _EngineKernel(KernelBase):
    def __init__(self, engine_ref: str, engine: object) -> None:
        if not isinstance(engine_ref, str) or not engine_ref:
            raise ValueError("engine_ref must be a non-empty string.")
        required = ("default_values", "materialize", "variable_metadata", "variables")
        missing = [
            name for name in required if not callable(getattr(engine, name, None))
        ]
        if missing:
            raise TypeError(
                f"US post-transfer engine lacks method(s): {sorted(missing)}."
            )
        self.engine_ref = engine_ref
        self._engine = engine

    def _require_engine_ref(self, context: KernelContext) -> None:
        if context.params.get("engine_ref") != self.engine_ref:
            raise ValueError(
                f"US stage {context.node.id!r} engine_ref "
                f"{context.params.get('engine_ref')!r} does not match bound "
                f"engine {self.engine_ref!r}."
            )


class USPostTransferSeedKernel(_EngineKernel):
    """Wrap `seed_multispine_pool_inputs` with a registry-bound engine."""

    ref = "us.post_transfer.seed@1"
    capabilities = _SEEDED

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            type(self._engine),
            multispine_pool_module,
            take_up_module,
            take_up_contract_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        self._require_engine_ref(context)
        identity = take_up_contract_identity(load_take_up_contract())
        _require_params(
            context,
            {
                "stage": "seed_multispine_pool_inputs",
                "engine_ref": self.engine_ref,
                "seed": POOL_RANDOM_SEED,
                "time_period": POOL_TIME_PERIOD,
                "take_up_contract_version": int(identity["version"]),
                "take_up_resource_sha256": str(identity["resource_sha256"]),
            },
        )
        return _stage_result(
            seed_multispine_pool_inputs(
                _frame_from_context(context),
                engine=self._engine,
            ),
            context,
        )


class USPostTransferMaterializeKernel(_EngineKernel):
    """Wrap agreement SSI materialization with a registry-bound engine."""

    ref = "us.post_transfer.materialize@1"
    capabilities = _ENGINE_COMPUTE

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            type(self._engine),
            multispine_pool_module,
            policyengine_us_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        self._require_engine_ref(context)
        _require_params(
            context,
            {
                "stage": "materialize_multispine_agreement_outputs",
                "engine_ref": self.engine_ref,
                "time_period": POOL_TIME_PERIOD,
                "household_batch_size": POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE,
                "ssi_dependency_sha256": POOL_SSI_DEPENDENCY_CONTRACT.sha256,
                "engine_input_projection_sha256": (
                    POOL_ENGINE_INPUT_PROJECTION_CONTRACT.sha256
                ),
                "engine_input_defaults_sha256": (
                    POOL_ENGINE_INPUT_PROJECTION_CONTRACT.defaults_sha256
                ),
            },
        )
        return _stage_result(
            materialize_multispine_agreement_outputs(
                _frame_from_context(context),
                engine=self._engine,
            ),
            context,
        )


def build_us_post_transfer_registry(
    graph: Graph,
    *,
    engine: object | None,
    engine_ref: str,
) -> KernelRegistry:
    """Return exact kernel coverage for `graph`."""

    if engine is None:
        from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

        engine = PolicyEngineUSEngine()
    registry = KernelRegistry()
    registry.register(USPostTransferCreateKernel())
    registry.register(USPostTransferIdentityKernel())
    registry.register(USPostTransferPrepareKernel())
    registry.register(USPostTransferDeriveKernel())
    registry.register(USPostTransferSeedKernel(engine_ref, engine))
    registry.register(USPostTransferMaterializeKernel(engine_ref, engine))
    required = {node.kernel for node in graph.nodes}
    actual = set(registry.refs())
    if actual != required:
        raise ValueError(
            "US post-transfer registry does not exactly cover its graph "
            f"(missing={sorted(required - actual)}, extra={sorted(actual - required)})."
        )
    return registry
