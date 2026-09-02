"""Kernel family for the UK FRS spine graph.

Real-build kernels reconstruct a minimal UK :class:`~microcosm.frame.Frame`
from only their declared ``KernelContext.tables``, invoke the existing stage
transform unchanged, and project the owned cells back out.  Structural stages
return source lineage, memberships, cell overlays, and explicit weights; the
graph executor, not the kernel, materializes the expanded population.

The committed parity fixture uses the same kernel contracts with recorded
direct-StagePlan deltas under its content-bound source bundle.  This keeps the
fixture runnable without licensed donor files while still exercising every
node, ownership edge, rewrite boundary, row expansion, and byte-level Frame
surface.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.frame import Frame, WeightKind, Weights
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

from .national_frame import UK_NATIONAL_SCHEMA
from .rowwise_geography import id_multiplier_for_values

__all__ = [
    "UKClaimKernel",
    "UKCreateKernel",
    "UKExpandStageKernel",
    "UKIdentityKernel",
    "UKStageKernel",
    "build_uk_registry",
]


_STAGE_MODULES = {
    "frs_spine": "frs_spine",
    "frs_employment": "frs_employment",
    "frs_council_tax": "frs_council_tax",
    "frs_disability": "frs_disability",
    "frs_education": "frs_education",
    "frs_legacy_proxies": "frs_legacy_proxies",
    "frs_education_grant_split": "frs_education_grants",
    "frs_take_up": "frs_take_up",
    "frs_person_draws": "frs_person_draws",
    "frs_household_draws": "frs_household_draws",
    "frs_brma": "frs_brma",
    "was_wealth": "was_wealth",
    "regional_property_uprating": "regional_uprating",
    "lcfs_consumption": "lcfs_consumption",
    "etb_vat": "etb_vat",
    "etb_services": "etb_services",
    "frs_hmrc_spine_leaves": "frs_hmrc_leaves",
    "spi_support_channel": "spi_spine",
    "hmrc_spi_income_spine": "spi_spine",
    "uc_capital_coherence": "uc_capital_coherence",
    "cgt_incidence_clone": "cgt_structure",
    "cgt_band_donors": "cgt_structure",
    "hmrc_cgt_gains_spine": "cgt_imputation",
    "salary_sacrifice": "salary_sacrifice",
    "student_loans": "student_loans",
    "age_tail": "age_tail",
}

_COMPUTE = Capabilities(
    determinism=Determinism.DETERMINISTIC,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.NONE,
)
_FILTER = Capabilities(
    determinism=Determinism.DETERMINISTIC,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.NONE,
    structural=StructuralDelta.FILTER,
)
_CREATE = Capabilities(
    determinism=Determinism.DETERMINISTIC,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.NONE,
    structural=StructuralDelta.CREATE,
)
_EXPAND = Capabilities(
    determinism=Determinism.DETERMINISTIC,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.NONE,
    structural=StructuralDelta.EXPAND,
)


def _stage_module(stage: str):
    try:
        leaf = _STAGE_MODULES[stage]
    except KeyError as error:
        raise ValueError(
            f"No implementation module is registered for {stage!r}."
        ) from error
    return importlib.import_module(f"microcosm.build.uk_runtime.{leaf}")


def _implementation_hash(kernel: object, stage: str, transform: object | None) -> str:
    # The stage module is the behavior-bearing source in both real and fixture
    # mode.  Hashing an injected transform's dynamic test wrapper would make
    # hermetic registries unhashable and, more importantly, would fail to bind
    # production edits made elsewhere in that stage's module.
    del transform
    return source_hash(type(kernel), _stage_module(stage))


def _mass_log_payload(frame: Frame) -> list[dict[str, object]]:
    return [
        {
            "entity": record.entity,
            "old_total": record.old_total,
            "new_total": record.new_total,
            "declared_factor": record.declared_factor,
            "reason": record.reason,
        }
        for record in frame.mass_log
    ]


def _invoke_transform(transform: object, frame: Frame, context: KernelContext):
    """Invoke a stage, giving context-bound adapters only declared sources."""

    run_with_sources = getattr(transform, "run_with_sources", None)
    if callable(run_with_sources):
        return run_with_sources(frame, context.sources)
    return transform(frame)


def _minimal_frame(context: KernelContext) -> Frame:
    missing = set(UK_NATIONAL_SCHEMA.entities) - context.tables.keys()
    if missing:
        raise ValueError(
            f"UK stage {context.node.id!r} lacks declared slices for {sorted(missing)}."
        )
    if "household" not in context.weights:
        raise ValueError(f"UK stage {context.node.id!r} lacks household weights.")
    return Frame(
        {
            entity: context.tables[entity].copy(deep=True)
            for entity in UK_NATIONAL_SCHEMA.entities
        },
        UK_NATIONAL_SCHEMA,
        {"household": context.weights["household"]},
        context.strata.copy(deep=True),
        metadata={"time_period": str(context.params.get("time_period", "2024"))},
    )


def _id_index(frame: Frame, entity: str) -> pd.Index:
    id_column = frame.schema.entity_id_column(entity)
    return pd.Index(frame.table(entity)[id_column].to_numpy(copy=True), name=id_column)


def _cast_owned(series: pd.Series, dtype: str) -> pd.Series:
    target = dtype_for_token(dtype)
    if series.dtype != target:
        series = series.astype(target)
    return series


def _owned_series(frame: Frame, entity: str, column: str, dtype: str) -> pd.Series:
    values = _cast_owned(frame.table(entity)[column].copy(deep=True), dtype)
    values.index = _id_index(frame, entity)
    values.name = column
    return values


def _normalize_create_frame(frame: Frame, context: KernelContext) -> Frame:
    tables = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    for owned in context.node.outputs:
        tables[owned.entity][owned.column] = _cast_owned(
            tables[owned.entity][owned.column], owned.dtype
        ).array
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


class UKCreateKernel(KernelBase):
    """Load fixture CSV tables or invoke the bound FRS root transform."""

    ref = "uk.create@1"
    capabilities = _CREATE

    def __init__(self, transform: object | None = None) -> None:
        self.transform = transform

    def implementation_hash(self) -> str:
        return _implementation_hash(self, "frs_spine", self.transform)

    def run(self, context: KernelContext) -> KernelResult:
        source = context.sources["frs"]
        if self.transform is None:
            frame = load_source("csv-tables", source)
        else:
            from .frs_spine import uk_frs_spine_seed_frame

            frame = _invoke_transform(
                self.transform, uk_frs_spine_seed_frame(), context
            )
        if not isinstance(frame, Frame):
            raise TypeError(
                f"The UK root transform returned {type(frame).__name__}, not Frame."
            )
        return KernelResult(frame=_normalize_create_frame(frame, context))


class UKIdentityKernel(KernelBase):
    """Keep every person, creating a new ownership/population boundary."""

    ref = "uk.identity@1"
    capabilities = _FILTER

    def run(self, context: KernelContext) -> KernelResult:
        person = context.tables[UK_NATIONAL_SCHEMA.person_entity]
        id_column = UK_NATIONAL_SCHEMA.person_id_column
        ids = pd.Index(person[id_column].to_numpy(copy=True), name=id_column)
        return KernelResult(keep=pd.Series(True, index=ids, dtype="bool"))


class UKClaimKernel(KernelBase):
    """Claim executor-materialized incumbent cells in a new population."""

    ref = "uk.claim@1"
    capabilities = _COMPUTE

    def run(self, context: KernelContext) -> KernelResult:
        columns = {
            (owned.entity, owned.column): pd.Series(
                _cast_owned(
                    context.tables[owned.entity][owned.column].copy(deep=True),
                    owned.dtype,
                ).array,
                index=pd.Index(
                    context.tables[owned.entity][
                        UK_NATIONAL_SCHEMA.entity_id_column(owned.entity)
                    ].to_numpy(copy=True),
                    name=UK_NATIONAL_SCHEMA.entity_id_column(owned.entity),
                ),
                name=owned.column,
                dtype=dtype_for_token(owned.dtype),
            )
            for owned in context.node.outputs
        }
        return KernelResult(columns=MappingProxyType(columns))


def _read_receipt(root: Path) -> dict[str, object]:
    path = root / "receipt.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Fixture receipt {path} is not a JSON object.")
    return payload


def _fixture_table(root: Path, entity: str) -> pd.DataFrame:
    path = root / f"{entity}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"UK parity delta is missing {path}.")
    return pd.read_csv(path, float_precision="round_trip")


class UKStageKernel(KernelBase):
    """One ordinary UK stage, parameterized by its manifest stage name."""

    capabilities = _COMPUTE

    def __init__(self, stage: str, transform: object | None = None) -> None:
        self.stage = stage
        self.transform = transform
        self.ref = f"uk.stage.{stage}@1"

    def implementation_hash(self) -> str:
        return _implementation_hash(self, self.stage, self.transform)

    def run(self, context: KernelContext) -> KernelResult:
        if self.transform is None:
            return self._run_fixture(context)
        before = _minimal_frame(context)
        after = _invoke_transform(self.transform, before, context)
        if not isinstance(after, Frame):
            raise TypeError(
                f"UK stage {self.stage!r} returned {type(after).__name__}, not Frame."
            )
        columns = {
            (owned.entity, owned.column): _owned_series(
                after, owned.entity, owned.column, owned.dtype
            )
            for owned in context.node.outputs
        }
        return KernelResult(
            columns=MappingProxyType(columns),
            receipt={
                "stage": self.stage,
                "frame_mass_log_append": _mass_log_payload(after),
            },
        )

    def _run_fixture(self, context: KernelContext) -> KernelResult:
        root = context.sources["frs"] / "deltas" / self.stage
        tables: dict[str, pd.DataFrame] = {}
        columns: dict[tuple[str, str], pd.Series] = {}
        for owned in context.node.outputs:
            table = tables.setdefault(owned.entity, _fixture_table(root, owned.entity))
            id_column = UK_NATIONAL_SCHEMA.entity_id_column(owned.entity)
            values = _cast_owned(table[owned.column], owned.dtype)
            columns[(owned.entity, owned.column)] = pd.Series(
                values.array.copy(),
                index=pd.Index(table[id_column].to_numpy(copy=True), name=id_column),
                name=owned.column,
                dtype=dtype_for_token(owned.dtype),
            )
        receipt = {"stage": self.stage, **_read_receipt(root)}
        return KernelResult(
            columns=MappingProxyType(columns),
            receipt=MappingProxyType(receipt),
        )


def _expand_cells(context: KernelContext) -> tuple[tuple[str, str, str], ...]:
    raw = context.params.get("expand_cells")
    if not isinstance(raw, tuple):
        raise ValueError(f"UK EXPAND stage {context.node.id!r} has no cell contract.")
    return tuple(tuple(str(part) for part in item) for item in raw)  # type: ignore[arg-type]


def _source_lineage(
    before: Frame,
    after: Frame,
    entity: str,
    *,
    id_offset: int | None,
) -> pd.Series:
    """Derive immediate target-to-source ids from a real structural result."""

    id_column = before.schema.entity_id_column(entity)
    before_table = before.table(entity)
    after_table = after.table(entity)
    before_ids = pd.Index(before_table[id_column])
    values: list[object] = []
    source_column = f"{entity}_source_id"
    for _, row in after_table.iterrows():
        target = row[id_column]
        if target in before_ids:
            values.append(target)
            continue
        if source_column in after_table and row[source_column] in before_ids:
            values.append(row[source_column])
            continue

        if id_offset is not None:
            candidate = target - id_offset
            if candidate in before_ids:
                values.append(candidate)
                continue
            raise ValueError(
                f"UK stage produced {entity!r} target id {target!r} whose "
                f"offset lineage {candidate!r} is not an incumbent id."
            )

        raise ValueError(
            f"UK stage produced {entity!r} target id {target!r} without an "
            f"explicit {source_column!r} or a declared ID-offset lineage rule."
        )
    return pd.Series(
        values,
        index=pd.Index(after_table[id_column].to_numpy(copy=True), name=id_column),
        dtype=before_table[id_column].dtype,
        name=id_column,
    )


class UKExpandStageKernel(KernelBase):
    """A UK row-expanding stage returning lineage rather than a Frame."""

    capabilities = _EXPAND

    def __init__(self, stage: str, transform: object | None = None) -> None:
        self.stage = stage
        self.transform = transform
        self.ref = f"uk.stage.expand.{stage}@1"

    def implementation_hash(self) -> str:
        return _implementation_hash(self, self.stage, self.transform)

    def run(self, context: KernelContext) -> KernelResult:
        if self.transform is None:
            return self._run_fixture(context)
        before = _minimal_frame(context)
        after = _invoke_transform(self.transform, before, context)
        if not isinstance(after, Frame):
            raise TypeError(
                f"UK stage {self.stage!r} returned {type(after).__name__}, not Frame."
            )
        cells = _expand_cells(context)
        id_offset = None
        if self.stage in {"cgt_incidence_clone", "cgt_band_donors"}:
            id_offset = id_multiplier_for_values(
                *(
                    before.table(entity)[before.schema.entity_id_column(entity)]
                    for entity in before.entities
                ),
                *(
                    before.table(before.schema.person_entity)[
                        before.schema.membership_column(group)
                    ]
                    for group in before.schema.group_entities
                ),
            )
        columns: dict[tuple[str, str], pd.Series] = {}
        for entity in before.entities:
            id_column = before.schema.entity_id_column(entity)
            columns[(entity, id_column)] = _source_lineage(
                before,
                after,
                entity,
                id_offset=id_offset,
            )
        person = before.schema.person_entity
        person_ids = _id_index(after, person)
        for group in before.schema.group_entities:
            membership = before.schema.membership_column(group)
            columns[(person, membership)] = pd.Series(
                after.table(person)[membership].array.copy(),
                index=person_ids,
                name=membership,
                dtype=after.table(person)[membership].dtype,
            )
        for entity, column, dtype in cells:
            columns[(entity, column)] = _owned_series(after, entity, column, dtype)
        weight_entity = str(context.params["expand_weight_entity"])
        after_weights = after.weights_for(weight_entity)
        declared_kind = WeightKind(str(context.params["expand_weight_kind"]))
        if after_weights.kind is not declared_kind:
            raise ValueError(
                f"UK EXPAND stage {self.stage!r} returned weight kind "
                f"{after_weights.kind.value!r}, not declared "
                f"{declared_kind.value!r}."
            )
        return KernelResult(
            columns=MappingProxyType(columns),
            weights=after_weights,
            receipt={
                "stage": self.stage,
                "frame_mass_log_append": _mass_log_payload(after),
            },
        )

    def _run_fixture(self, context: KernelContext) -> KernelResult:
        root = context.sources["frs"] / "deltas" / self.stage
        cells = _expand_cells(context)
        by_entity: dict[str, list[tuple[str, str]]] = {}
        for entity, column, dtype in cells:
            by_entity.setdefault(entity, []).append((column, dtype))

        columns: dict[tuple[str, str], pd.Series] = {}
        for entity in UK_NATIONAL_SCHEMA.entities:
            table = _fixture_table(root, entity)
            id_column = UK_NATIONAL_SCHEMA.entity_id_column(entity)
            target_ids = pd.Index(table[id_column].to_numpy(copy=True), name=id_column)
            source_dtype = context.tables[entity][id_column].dtype
            columns[(entity, id_column)] = pd.Series(
                table["__source_id__"].to_numpy(copy=True),
                index=target_ids,
                dtype=source_dtype,
                name=id_column,
            )
            if entity == UK_NATIONAL_SCHEMA.person_entity:
                for group in UK_NATIONAL_SCHEMA.group_entities:
                    membership = UK_NATIONAL_SCHEMA.membership_column(group)
                    columns[(entity, membership)] = pd.Series(
                        table[membership].to_numpy(copy=True),
                        index=target_ids,
                        dtype=context.tables[entity][membership].dtype,
                        name=membership,
                    )
            for column, dtype in by_entity.get(entity, ()):
                values = _cast_owned(table[column], dtype)
                columns[(entity, column)] = pd.Series(
                    values.array.copy(),
                    index=target_ids,
                    name=column,
                    dtype=dtype_for_token(dtype),
                )

        weight_entity = str(context.params["expand_weight_entity"])
        weight_table = pd.read_csv(root / "weights.csv", float_precision="round_trip")
        expected_ids = columns[
            (weight_entity, UK_NATIONAL_SCHEMA.entity_id_column(weight_entity))
        ].index
        id_column = UK_NATIONAL_SCHEMA.entity_id_column(weight_entity)
        actual_ids = pd.Index(weight_table[id_column], name=id_column)
        if not actual_ids.equals(expected_ids):
            raise ValueError(
                f"UK parity weights for {self.stage!r} do not align to targets."
            )
        weights = Weights(
            weight_table["weight"].to_numpy(dtype=np.float64, copy=True),
            WeightKind(str(context.params["expand_weight_kind"])),
        )
        receipt = {"stage": self.stage, **_read_receipt(root)}
        return KernelResult(
            columns=MappingProxyType(columns),
            weights=weights,
            receipt=MappingProxyType(receipt),
        )


def build_uk_registry(
    graph: Graph,
    implementations: Mapping[str, object],
) -> KernelRegistry:
    """Bind graph refs to fixture readers or the supplied real transforms."""

    stage_names = {
        str(node.params["stage"]) for node in graph.nodes if "stage" in node.params
    }
    unknown = set(implementations) - (stage_names | {"frs_spine"})
    if unknown:
        raise ValueError(f"Unknown UK graph stage implementations: {sorted(unknown)}.")
    if implementations:
        missing = stage_names - implementations.keys()
        if missing:
            raise ValueError(
                f"Missing UK graph stage implementations: {sorted(missing)}."
            )

    registry = KernelRegistry()
    registry.register(UKCreateKernel(implementations.get("frs_spine")))
    registry.register(UKIdentityKernel())
    registry.register(UKClaimKernel())
    for stage in sorted(stage_names):
        transform = implementations.get(stage)
        if stage in {
            "spi_support_channel",
            "cgt_incidence_clone",
            "cgt_band_donors",
        }:
            registry.register(UKExpandStageKernel(stage, transform))
        else:
            registry.register(UKStageKernel(stage, transform))

    required = {node.kernel for node in graph.nodes}
    if set(registry.refs()) != required:
        missing = sorted(required - set(registry.refs()))
        extra = sorted(set(registry.refs()) - required)
        raise ValueError(
            f"UK registry does not exactly cover its graph (missing={missing}, "
            f"extra={extra})."
        )
    return registry
