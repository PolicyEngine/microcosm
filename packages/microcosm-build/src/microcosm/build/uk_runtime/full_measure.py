"""Rules-engine evaluation for the selected full-build target surface.

Temporary inputs and prepared columns remain inside this operation. Graph
consumers receive compiled numerical contributions, never injected engine state.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.target_materialization import resolve_target_measures
from microcosm.build.uk_runtime import (
    CalibrationFrameAdapter,
    UKRowwiseNationalRows,
    compute_household_metrics,
    drop_injected_measure_inputs,
    inject_measure_inputs,
    ladder_clone_index_column,
    materialize_uk_ledger_targets,
)
from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
from microcosm.frame import Frame, MassChangeRecord

UK_BLOCK_SENSITIVE_MEASURE_COLUMNS = (
    "ons/corporate_land_value",
    "ons/land_value",
    "slc/student_loan_repayment/england",
)


def resolve_uk_full_measures(
    frame,
    national_registry,
    *,
    period: int,
    scratch_dir: Path,
    band_edge_registry=None,
    resolver_factory=UKMeasureResolver,
    blocks: int = 1,
    local_grains: tuple[str, ...] = ("constituency", "la"),
) -> tuple[Any, Any, UKRowwiseNationalRows, dict[str, pd.DataFrame], dict[str, Any]]:
    """Resolve national inputs and local metrics on the cloned frame.

    ``blocks=1`` uses one scratch-mode engine for the whole clone.  The
    reviewed escape hatch ``blocks=K`` resolves each clone index separately,
    then rejoins every entity-level prepared column by its stable entity id so
    the full-frame target materialization and single solve retain frame order.
    """

    household = frame.table("household")
    if blocks < 1:
        raise ValueError("engine resolution blocks must be positive.")
    if blocks == 1:
        block_frames = [(None, frame)]
    else:
        clone_column = ladder_clone_index_column("household")
        if clone_column not in household.columns:
            raise ValueError(f"per-clone engine resolution requires {clone_column}.")
        clone_indices = tuple(sorted(household[clone_column].unique().tolist()))
        if len(clone_indices) != blocks:
            raise ValueError(
                "engine resolution blocks must match the realized clone indices: "
                f"requested {blocks}, found {clone_indices}."
            )
        person = frame.table("person")
        block_frames = []
        for clone_index in clone_indices:
            household_ids = set(
                household.loc[
                    household[clone_column] == clone_index,
                    "household_id",
                ].tolist()
            )
            person_mask = person["person_household_id"].isin(household_ids)
            block = frame.select(person_mask)
            # The block carries a K-th of the cloned mass while its log still
            # ends on the full-clone record, and the scratch export validates
            # the chain. Declare the subset explicitly: old = the cloned
            # total, new = the block total, reason naming the block. The block
            # frame is engine scratch and is discarded after resolution.
            block_weights = block.weights_for("household")
            full_total = float(frame.weights_for("household").total)
            block_total = float(block_weights.total)
            subset_record = MassChangeRecord(
                entity="household",
                old_total=full_total,
                new_total=block_total,
                declared_factor=block_total / full_total,
                reason=(
                    f"engine resolution block {clone_index} of {blocks}: "
                    "scratch subset of the cloned frame for measure "
                    "resolution only, discarded after resolution"
                ),
            )
            block = Frame(
                {
                    **{name: block.table(name) for name in block.entities},
                    **{name: block.link(name) for name in block.links},
                },
                block.schema,
                {
                    entity: block.weights_for(entity)
                    for entity in block.weighted_entities
                },
                block.strata,
                mass_log=(*block.mass_log, subset_record),
                metadata=block.metadata,
            )
            block_frames.append((clone_index, block))

    measure_parts: dict[tuple[str, str], list[pd.Series]] = {}
    metric_parts: dict[str, list[pd.DataFrame]] = {grain: [] for grain in local_grains}
    resolver_receipts: list[Mapping[str, Any]] = []
    national_input_keys: set[tuple[str, str]] | None = None
    for clone_index, block_frame in block_frames:
        block_scratch = (
            scratch_dir if clone_index is None else scratch_dir / f"clone-{clone_index}"
        )
        resolver = resolver_factory(
            simulation_source=None,
            scratch_dir=block_scratch,
            year=period,
            frame=block_frame,
        )
        resolution = resolve_target_measures(
            lambda block_frame=block_frame: CalibrationFrameAdapter(block_frame),
            national_registry,
            resolver,
            period=period,
        )
        keys = set(resolution.measure_inputs)
        if national_input_keys is None:
            national_input_keys = keys
        elif keys != national_input_keys:
            raise RuntimeError(
                "per-clone engine resolution returned inconsistent national inputs."
            )
        for (entity, variable), values in resolution.measure_inputs.items():
            entity_table = block_frame.table(entity)
            entity_id = f"{entity}_id"
            measure_parts.setdefault((entity, variable), []).append(
                pd.Series(
                    np.asarray(values),
                    index=entity_table[entity_id].tolist(),
                )
            )
        block_household_ids = block_frame.table("household")["household_id"].tolist()
        for area_type in metric_parts:
            metric_parts[area_type].append(
                compute_household_metrics(
                    resolver.simulation,
                    area_type,
                    period=period,
                    household_ids=block_household_ids,
                )
            )
        resolver_receipts.append(resolver.receipt())
        del resolver
        simulation_input = block_scratch / "simulation-input.h5"
        simulation_input.unlink(missing_ok=True)
        try:
            block_scratch.rmdir()
        except OSError:
            pass

    measure_inputs: dict[tuple[str, str], np.ndarray] = {}
    for (entity, variable), parts in measure_parts.items():
        combined = pd.concat(parts)
        if combined.index.has_duplicates:
            raise RuntimeError(
                f"per-clone engine resolution duplicated {entity} ids for {variable}."
            )
        ordered_ids = frame.table(entity)[f"{entity}_id"]
        ordered = combined.reindex(ordered_ids.tolist())
        if ordered.isna().any():
            raise RuntimeError(
                f"per-clone engine resolution missed {entity} rows for {variable}."
            )
        measure_inputs[(entity, variable)] = ordered.to_numpy()

    full_household_ids = household["household_id"].tolist()
    local_metrics = {}
    for area_type, parts in metric_parts.items():
        combined = pd.concat(parts)
        if combined.index.has_duplicates:
            raise RuntimeError(
                f"per-clone engine resolution duplicated {area_type} household ids."
            )
        ordered = combined.reindex(full_household_ids)
        if ordered.isna().any().any():
            raise RuntimeError(
                f"per-clone engine resolution missed {area_type} household rows."
            )
        local_metrics[area_type] = ordered

    adapter = CalibrationFrameAdapter(frame)
    # Injected engine inputs are scratch state for materialization only:
    # they must be dropped before the prepared frame is assembled, or the
    # flattening rule refuses columns that now exist on two entities
    # (region, esa_* on the live spine). Same lifecycle as the national stage.
    original_columns = {
        entity: set(table.columns) for entity, table in adapter.tables.items()
    }
    inject_measure_inputs(adapter, measure_inputs)
    materialized = materialize_uk_ledger_targets(
        adapter,
        national_registry,
        period=period,
        band_edge_registry=(
            national_registry if band_edge_registry is None else band_edge_registry
        ),
    )
    if materialized.skipped:
        raise RuntimeError(
            "candidate national target materialization skipped row(s): "
            f"{[skip.__dict__ for skip in materialized.skipped]}."
        )
    modes = {receipt.get("mode") for receipt in resolver_receipts}
    versions = {receipt.get("policyengine_uk_version") for receipt in resolver_receipts}
    if len(modes) != 1 or len(versions) != 1:
        raise RuntimeError("per-clone engine resolver provenance is inconsistent.")
    cgt_period_contract = resolver_receipts[0].get("cgt_period_contract")
    if any(
        block_receipt.get("cgt_period_contract") != cgt_period_contract
        for block_receipt in resolver_receipts[1:]
    ):
        raise RuntimeError("per-clone CGT period contract is inconsistent.")
    receipt = {
        "mode": next(iter(modes)),
        "engine_version": next(iter(versions)),
        "households": len(frame.table("household")),
        "persons": len(frame.table("person")),
        "benunits": len(frame.table("benunit")),
        "national_inputs": len(measure_inputs),
        "local_metrics": {
            area_type: len(metrics.columns)
            for area_type, metrics in local_metrics.items()
        },
        "blocks": blocks,
    }
    if cgt_period_contract is not None:
        receipt["cgt_period_contract"] = cgt_period_contract
    if blocks > 1:
        receipt["deviation"] = "per_clone_block_engine_resolution"
        present = sorted(
            column
            for column in UK_BLOCK_SENSITIVE_MEASURE_COLUMNS
            if column in {variable for _, variable in measure_inputs}
        )
        receipt["block_sensitivity"] = {
            "known_population_normalised_measures": list(
                UK_BLOCK_SENSITIVE_MEASURE_COLUMNS
            ),
            "present_in_this_run": present,
            "caveat": (
                "per-block engine resolution mis-measures population-normalised "
                "formulas (each block reproduces a national aggregate); rows "
                "on these measures are not evidence for adjudication from this "
                "run. Resolve in a single block before ruling on them."
            ),
        }
    try:
        scratch_dir.rmdir()
    except OSError:
        pass
    drop_injected_measure_inputs(adapter, measure_inputs, original_columns)
    national_rows = UKRowwiseNationalRows(
        targets=national_registry.to_target_set(),
        registry=national_registry,
        families=tuple(sorted({spec.family for spec in national_registry.specs})),
    )
    return (
        adapter.prepared_frame(),
        adapter.restore,
        national_rows,
        local_metrics,
        receipt,
    )
