#!/usr/bin/env python3
"""Regenerate the deterministic charter-H2 UK spine parity fixture.

The source bundle begins with 300 synthetic one-person FRS-shaped households.
A direct :class:`microcosm.build.plan.StagePlan` executes all 26 named stages:
100 SPI support rows are appended, CGT incidence doubles the resulting 400
households, and the CGT-band stage adds 270 donors.  Each transform writes its
owned delta beside the raw root tables.  The graph kernels consume those
content-bound deltas independently, and acceptance compares the executor's
final Frame identity with the direct StagePlan result recorded here.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.plan import Stage, StagePlan
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_SPINE_MASS_CONSERVATION_REASON,
)
from microcosm.build.uk_runtime.cgt_structure import (
    CGT_CLONE_MASS_CHANGE_REASON,
    CGT_DONOR_MASS_CHANGE_REASON,
)
from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.graph import (
    UK_SPINE_EXCLUSIONS,
    UK_SPINE_STRUCTURAL_STAGES,
    uk_spine_graph,
)
from microcosm.build.uk_runtime.salary_sacrifice import SALSAC_MASS_CHANGE_REASON
from microcosm.build.uk_runtime.spi_support import SPI_PRIOR_MASS_CHANGE_REASON
from microcosm.build.uk_runtime.student_loans import (
    STUDENT_LOANS_MASS_CHANGE_REASON,
)
from microcosm.frame import Frame, MassChangeRecord, WeightKind, Weights
from microcosm.graph import graph_to_json, load_source
from microcosm.graph.population import dtype_for_token

_REPOSITORY = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = (
    _REPOSITORY
    / "packages"
    / "microcosm-graph"
    / "tests"
    / "fixtures"
    / "parity"
    / "uk_spine"
)
_BASE_HOUSEHOLDS = 300
_SPI_SUPPORT_HOUSEHOLDS = 100
_CGT_DONOR_HOUSEHOLDS = 270
_SEED = 20260901

_ORDINARY_MASS_REASONS = {
    "hmrc_cgt_gains_spine": UK_CGT_SPINE_MASS_CONSERVATION_REASON,
    "salary_sacrifice": SALSAC_MASS_CHANGE_REASON,
    "student_loans": STUDENT_LOANS_MASS_CHANGE_REASON,
}
_EXPAND_MASS_REASONS = {
    "spi_support_channel": SPI_PRIOR_MASS_CHANGE_REASON,
    "cgt_incidence_clone": CGT_CLONE_MASS_CHANGE_REASON,
    "cgt_band_donors": CGT_DONOR_MASS_CHANGE_REASON,
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )


def _dtype_values(
    dtype: str,
    ids: np.ndarray,
    *,
    ordinal: int,
    column: str,
) -> pd.Series:
    if dtype == "float64":
        values = ordinal * 10.0 + (ids % 997).astype(np.float64) / 100.0
    elif dtype == "int64":
        values = ordinal * 1000 + ids % 997
    elif dtype == "bool":
        values = (ids + ordinal) % 2 == 0
    elif dtype == "string":
        values = [f"{column.upper()}_{int(value % 5)}" for value in ids]
    else:  # all UK fixture declarations use the frozen non-nullable subset
        raise ValueError(f"Unsupported fixture dtype {dtype!r}.")
    return pd.Series(values, dtype=dtype_for_token(dtype), name=column)


def _root_tables(graph) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]]]:
    create = graph.node("create_uk_frs")
    ids = np.arange(1, _BASE_HOUSEHOLDS + 1, dtype=np.int64)
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
            }
        ),
        "benunit": pd.DataFrame({"benunit_id": ids}),
        "household": pd.DataFrame({"household_id": ids}),
    }
    dtypes: dict[str, dict[str, str]] = {
        "person": {
            "person_id": "int64",
            "person_benunit_id": "int64",
            "person_household_id": "int64",
        },
        "benunit": {"benunit_id": "int64"},
        "household": {"household_id": "int64"},
    }
    for ordinal, owned in enumerate(create.outputs, start=1):
        if owned.column == "age":
            values = pd.Series(18 + ids % 63, dtype="int64", name="age")
        elif owned.column == "gender":
            values = pd.Series(
                np.where(ids % 2, "MALE", "FEMALE"), dtype="string", name="gender"
            )
        elif owned.column == "region":
            values = pd.Series(
                np.take(
                    np.array(["LONDON", "WALES", "SCOTLAND"], dtype=object),
                    ids % 3,
                ),
                dtype="string",
                name="region",
            )
        else:
            values = _dtype_values(
                owned.dtype,
                ids,
                ordinal=ordinal,
                column=owned.column,
            )
        tables[owned.entity][owned.column] = values.array
        dtypes[owned.entity][owned.column] = owned.dtype

    tables["person"]["__stratum__"] = pd.Series(["uk"] * len(ids), dtype="string").array
    dtypes["person"]["__stratum__"] = "string"
    return tables, dtypes


def _write_root_source(sources: Path, graph) -> Frame:
    tables, dtypes = _root_tables(graph)
    for entity, table in tables.items():
        _write_csv(sources / f"{entity}.csv", table)
    weights = 1.0 + (np.arange(_BASE_HOUSEHOLDS) % 7) / 10.0
    _write_json(
        sources / "schema.json",
        {
            "schema": {
                "person_entity": "person",
                "group_entities": ["benunit", "household"],
                "links": [],
            },
            "dtypes": dtypes,
            "strata_column": "__stratum__",
            "weights": {"household": {"kind": "design"}},
        },
    )
    _write_json(
        sources / "weights.json",
        {"weights": {"household": {"values": weights.tolist()}}},
    )
    return load_source("csv-tables", sources)


def _frame(
    incumbent: Frame,
    tables: dict[str, pd.DataFrame],
    *,
    weights: Weights | None = None,
    strata: pd.Series | None = None,
    append_mass: MassChangeRecord | None = None,
) -> Frame:
    mass_log = incumbent.mass_log
    if append_mass is not None:
        mass_log = (*mass_log, append_mass)
    return Frame(
        tables,
        incumbent.schema,
        {
            "household": (
                incumbent.weights_for("household") if weights is None else weights
            )
        },
        incumbent.strata if strata is None else strata,
        mass_log=mass_log,
        metadata=incumbent.metadata,
    )


def _rewrite_values(
    incumbent: pd.Series,
    dtype: str,
    *,
    ordinal: int,
    stage: str,
) -> pd.Series:
    if dtype == "float64":
        values = incumbent.astype("float64") + ordinal / 100.0
    elif dtype == "int64":
        values = incumbent.astype("int64") + ordinal
    elif dtype == "bool":
        values = ~incumbent.astype("bool")
    elif dtype == "string":
        values = incumbent.astype("string") + f"|{stage}"
    else:
        raise ValueError(f"Unsupported rewrite dtype {dtype!r}.")
    return pd.Series(values, dtype=dtype_for_token(dtype), name=incumbent.name)


def _stage_outputs(graph, stage: str):
    node = graph.node(
        f"{stage}.owned" if stage in UK_SPINE_STRUCTURAL_STAGES else stage
    )
    return node.outputs


def _write_ordinary_delta(
    root: Path,
    stage: str,
    frame: Frame,
    outputs,
    mass_record: MassChangeRecord | None,
) -> None:
    by_entity: dict[str, list[object]] = {}
    for owned in outputs:
        by_entity.setdefault(owned.entity, []).append(owned)
    for entity, owned_cells in by_entity.items():
        id_column = frame.schema.entity_id_column(entity)
        table = frame.table(entity)
        delta = pd.DataFrame({id_column: table[id_column].copy(deep=True)})
        for owned in owned_cells:
            delta[owned.column] = table[owned.column].array.copy()
        _write_csv(root / f"{entity}.csv", delta)
    payload = (
        []
        if mass_record is None
        else [
            {
                "entity": mass_record.entity,
                "old_total": mass_record.old_total,
                "new_total": mass_record.new_total,
                "declared_factor": mass_record.declared_factor,
                "reason": mass_record.reason,
            }
        ]
    )
    _write_json(root / "receipt.json", {"frame_mass_log_append": payload})


def _ordinary_transform(
    graph,
    stage: str,
    ordinal: int,
    delta_root: Path,
) -> Callable[[Frame], Frame]:
    outputs = _stage_outputs(graph, stage)

    def transform(incumbent: Frame) -> Frame:
        tables = {
            entity: incumbent.table(entity).copy(deep=True)
            for entity in incumbent.entities
        }
        for output_ordinal, owned in enumerate(outputs, start=1):
            table = tables[owned.entity]
            if owned.column in table:
                values = _rewrite_values(
                    table[owned.column],
                    owned.dtype,
                    ordinal=ordinal,
                    stage=stage,
                )
            else:
                id_column = incumbent.schema.entity_id_column(owned.entity)
                values = _dtype_values(
                    owned.dtype,
                    table[id_column].to_numpy(copy=False),
                    ordinal=ordinal * 100 + output_ordinal,
                    column=owned.column,
                )
            table[owned.column] = values.array

        mass_record = None
        reason = _ORDINARY_MASS_REASONS.get(stage)
        if reason is not None:
            total = incumbent.weights_for("household").total
            mass_record = MassChangeRecord(
                entity="household",
                old_total=total,
                new_total=total,
                declared_factor=None,
                reason=reason,
            )
        result = _frame(incumbent, tables, append_mass=mass_record)
        _write_ordinary_delta(delta_root / stage, stage, result, outputs, mass_record)
        return result

    return transform


def _append_rows(
    incumbent: Frame,
    source_positions: np.ndarray,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    tables: dict[str, pd.DataFrame] = {}
    lineage: dict[str, np.ndarray] = {}
    for entity in incumbent.entities:
        table = incumbent.table(entity)
        id_column = incumbent.schema.entity_id_column(entity)
        old_ids = table[id_column].to_numpy(copy=True)
        new_ids = np.arange(
            int(old_ids.max()) + 1,
            int(old_ids.max()) + 1 + len(source_positions),
            dtype=old_ids.dtype,
        )
        appended = table.iloc[source_positions].copy(deep=True).reset_index(drop=True)
        appended[id_column] = new_ids
        if entity == incumbent.schema.person_entity:
            for group in incumbent.schema.group_entities:
                membership = incumbent.schema.membership_column(group)
                appended[membership] = new_ids
        tables[entity] = pd.concat([table, appended], ignore_index=True)
        lineage[entity] = np.concatenate([old_ids, old_ids[source_positions]])
    return tables, lineage


def _mass_record(
    incumbent: Frame,
    weights: Weights,
    reason: str,
) -> MassChangeRecord:
    return MassChangeRecord(
        entity="household",
        old_total=incumbent.weights_for("household").total,
        new_total=weights.total,
        declared_factor=None,
        reason=reason,
    )


def _write_expand_delta(
    root: Path,
    frame: Frame,
    lineage: dict[str, np.ndarray],
    outputs,
    record: MassChangeRecord,
) -> None:
    by_entity: dict[str, list[object]] = {}
    for owned in outputs:
        by_entity.setdefault(owned.entity, []).append(owned)
    for entity in frame.entities:
        table = frame.table(entity)
        id_column = frame.schema.entity_id_column(entity)
        delta = pd.DataFrame(
            {
                id_column: table[id_column].copy(deep=True),
                "__source_id__": lineage[entity],
            }
        )
        if entity == frame.schema.person_entity:
            for group in frame.schema.group_entities:
                membership = frame.schema.membership_column(group)
                delta[membership] = table[membership].array.copy()
        for owned in by_entity.get(entity, ()):
            delta[owned.column] = table[owned.column].array.copy()
        _write_csv(root / f"{entity}.csv", delta)
    household = frame.table("household")
    _write_csv(
        root / "weights.csv",
        pd.DataFrame(
            {
                "household_id": household["household_id"].copy(deep=True),
                "weight": frame.weights_for("household").values,
            }
        ),
    )
    _write_json(
        root / "receipt.json",
        {
            "frame_mass_log_append": [
                {
                    "entity": record.entity,
                    "old_total": record.old_total,
                    "new_total": record.new_total,
                    "declared_factor": record.declared_factor,
                    "reason": record.reason,
                }
            ]
        },
    )


def _spi_transform(graph, delta_root: Path) -> Callable[[Frame], Frame]:
    stage = "spi_support_channel"
    outputs = _stage_outputs(graph, stage)

    def transform(incumbent: Frame) -> Frame:
        sources = np.arange(_SPI_SUPPORT_HOUSEHOLDS, dtype=np.int64)
        tables, lineage = _append_rows(incumbent, sources)
        old_count = incumbent.n("household")
        new_count = len(tables["household"])
        cell_values: dict[tuple[str, str], pd.Series] = {}
        for entity in incumbent.entities:
            table = tables[entity]
            ids = table[incumbent.schema.entity_id_column(entity)].to_numpy(copy=False)
            source_ids = lineage[entity]
            channel = np.where(np.arange(new_count) < old_count, "frs", "spi")
            cell_values[(entity, f"{entity}_support_channel")] = pd.Series(
                channel, dtype="string"
            )
            cell_values[(entity, f"{entity}_support_clone_index")] = pd.Series(
                np.where(np.arange(new_count) < old_count, 0, 1), dtype="int64"
            )
            cell_values[(entity, f"{entity}_source_id")] = pd.Series(
                source_ids, dtype="int64"
            )
            assert len(ids) == new_count
        cell_values[("household", "household_is_spi_synthetic")] = pd.Series(
            np.arange(new_count) >= old_count, dtype="bool"
        )
        cell_values[("household", "source_household_id")] = pd.Series(
            lineage["household"], dtype="int64"
        )
        cell_values[("household", "source_year")] = pd.Series(
            np.full(new_count, 2022), dtype="int64"
        )
        cell_values[("household", "source_household_key")] = pd.Series(
            [f"2022:{value}" for value in lineage["household"]], dtype="string"
        )
        for owned in outputs:
            tables[owned.entity][owned.column] = cell_values[
                (owned.entity, owned.column)
            ].array

        old = incumbent.weights_for("household").values
        base = old * 0.5
        support = old[sources]
        support = support / support.sum() * (old.sum() * 0.5)
        weights = Weights(np.concatenate([base, support]), WeightKind.IMPORTANCE)
        person_sources = lineage["person"]
        before_person_ids = pd.Index(incumbent.table("person")["person_id"])
        positions = before_person_ids.get_indexer(person_sources)
        strata = pd.Series(
            incumbent.strata.iloc[positions].array.copy(),
            index=tables["person"].index,
            name=incumbent.strata.name,
            dtype=incumbent.strata.dtype,
        )
        record = _mass_record(incumbent, weights, _EXPAND_MASS_REASONS[stage])
        result = _frame(
            incumbent,
            tables,
            weights=weights,
            strata=strata,
            append_mass=record,
        )
        _write_expand_delta(delta_root / stage, result, lineage, outputs, record)
        return result

    return transform


def _cgt_clone_transform(graph, delta_root: Path) -> Callable[[Frame], Frame]:
    stage = "cgt_incidence_clone"
    outputs = _stage_outputs(graph, stage)

    def transform(incumbent: Frame) -> Frame:
        old_count = incumbent.n("household")
        sources = np.arange(old_count, dtype=np.int64)
        tables, lineage = _append_rows(incumbent, sources)
        new_count = old_count * 2
        tables["household"]["household_is_capital_gains_clone"] = pd.Series(
            np.arange(new_count) >= old_count, dtype="bool"
        ).array
        capital = np.zeros(new_count, dtype=np.float64)
        capital[old_count:] = (
            lineage["person"][old_count:].astype(np.float64) % 113 + 1
        ) * 100.0
        tables["person"]["capital_gains"] = pd.Series(capital, dtype="float64").array
        old_weights = incumbent.weights_for("household").values
        weights = Weights(
            np.concatenate([old_weights * 0.5, old_weights * 0.5]),
            WeightKind.IMPORTANCE,
        )
        positions = pd.Index(incumbent.table("person")["person_id"]).get_indexer(
            lineage["person"]
        )
        strata = pd.Series(
            incumbent.strata.iloc[positions].array.copy(),
            index=tables["person"].index,
            name=incumbent.strata.name,
            dtype=incumbent.strata.dtype,
        )
        record = _mass_record(incumbent, weights, _EXPAND_MASS_REASONS[stage])
        result = _frame(
            incumbent,
            tables,
            weights=weights,
            strata=strata,
            append_mass=record,
        )
        _write_expand_delta(delta_root / stage, result, lineage, outputs, record)
        return result

    return transform


def _cgt_donor_transform(graph, delta_root: Path) -> Callable[[Frame], Frame]:
    stage = "cgt_band_donors"
    outputs = _stage_outputs(graph, stage)

    def transform(incumbent: Frame) -> Frame:
        sources = np.arange(_CGT_DONOR_HOUSEHOLDS, dtype=np.int64)
        old_count = incumbent.n("household")
        tables, lineage = _append_rows(incumbent, sources)
        new_count = old_count + _CGT_DONOR_HOUSEHOLDS
        tables["household"]["household_is_cgt_band_donor"] = pd.Series(
            np.arange(new_count) >= old_count, dtype="bool"
        ).array
        capital = tables["person"]["capital_gains"].astype("float64").copy()
        capital.iloc[old_count:] = (
            12_300.0 + np.arange(_CGT_DONOR_HOUSEHOLDS, dtype=np.float64) * 137.0
        )
        tables["person"]["capital_gains"] = capital.array
        donor_weights = 0.05 + (np.arange(_CGT_DONOR_HOUSEHOLDS) % 9) / 100.0
        weights = Weights(
            np.concatenate([incumbent.weights_for("household").values, donor_weights]),
            WeightKind.IMPORTANCE,
        )
        positions = pd.Index(incumbent.table("person")["person_id"]).get_indexer(
            lineage["person"]
        )
        strata = pd.Series(
            incumbent.strata.iloc[positions].array.copy(),
            index=tables["person"].index,
            name=incumbent.strata.name,
            dtype=incumbent.strata.dtype,
        )
        record = _mass_record(incumbent, weights, _EXPAND_MASS_REASONS[stage])
        result = _frame(
            incumbent,
            tables,
            weights=weights,
            strata=strata,
            append_mass=record,
        )
        _write_expand_delta(delta_root / stage, result, lineage, outputs, record)
        return result

    return transform


def _stage_names() -> tuple[str, ...]:
    from microcosm.build.country_spec import load_country_spec

    spec = load_country_spec("uk")
    assert spec.sources is not None
    return tuple(
        stage.stage
        for stage in spec.sources.stages
        if stage.stage not in UK_SPINE_EXCLUSIONS
    )


def _direct_plan(graph, root: Frame, sources: Path) -> StagePlan:
    transforms: dict[str, Callable[[Frame], Frame]] = {
        "frs_spine": lambda _seed: root,
        "spi_support_channel": _spi_transform(graph, sources / "deltas"),
        "cgt_incidence_clone": _cgt_clone_transform(graph, sources / "deltas"),
        "cgt_band_donors": _cgt_donor_transform(graph, sources / "deltas"),
    }
    for ordinal, stage in enumerate(_stage_names(), start=1):
        if stage not in transforms:
            transforms[stage] = _ordinary_transform(
                graph, stage, ordinal, sources / "deltas"
            )
    return StagePlan(
        Stage(
            name=stage,
            transform=transforms[stage],
            produces=(
                tuple(owned.column for owned in graph.node("create_uk_frs").outputs)
                if stage == "frs_spine"
                else tuple(owned.column for owned in _stage_outputs(graph, stage))
            ),
            rewrites=(
                ()
                if stage == "frs_spine"
                else tuple(owned.column for owned in _stage_outputs(graph, stage))
            ),
        )
        for stage in _stage_names()
    )


def generate(output: Path) -> None:
    """Write every H2 fixture artifact beneath ``output``."""

    output.mkdir(parents=True, exist_ok=True)
    sources = output / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    graph = uk_spine_graph()
    root = _write_root_source(sources, graph)
    plan = _direct_plan(graph, root, sources)
    final, records = plan.run(root)
    if tuple(record.stage for record in records) != _stage_names():
        raise RuntimeError("The direct fixture StagePlan did not execute all stages.")
    if final.n("household") != 1070:
        raise RuntimeError(
            f"The UK fixture ended with {final.n('household')} households, not 1070."
        )

    (output / "uk_spine.json").write_text(graph_to_json(graph) + "\n", encoding="utf-8")
    (output / "uk_frame_content_identity.txt").write_text(
        uk_frame_content_identity(final) + "\n", encoding="utf-8"
    )
    (output / "PRODUCED_BY.txt").write_text(
        "Lane F / tools/graph_uk_spine_fixture.py; direct 26-stage StagePlan "
        "synthetic parity oracle.\n",
        encoding="utf-8",
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    generate(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
