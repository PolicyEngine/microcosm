"""Rules-engine evaluation for the selected full-build target surface.

Temporary inputs and prepared columns remain inside this operation. Graph
consumers receive compiled numerical contributions, never injected engine state.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

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
from microcosm.calibrate.matrix import CalibrationProblem, build_constraint_matrix
from microcosm.frame import Frame, MassChangeRecord

UK_BLOCK_SENSITIVE_MEASURE_COLUMNS = (
    "ons/corporate_land_value",
    "ons/land_value",
    "slc/student_loan_repayment/england",
)


def _without_scratch_paths(value, scratch_dir: Path):
    """Record scratch-relative paths in a receipt, never the scratch root.

    A scratch-mode resolver names the file it simulated from (its
    ``source_path``), which lies under the per-run temporary directory the
    measure node creates. The receipt rides into the problem's bindings and
    the stored measure artifact, so an absolute scratch path would change a
    deterministic node's output between identical runs; the path is kept
    relative to the scratch root, which is stable (``simulation-input.h5``,
    ``clone-0/simulation-input.h5``).
    """
    if isinstance(value, Mapping):
        return {
            key: _without_scratch_paths(item, scratch_dir)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_without_scratch_paths(item, scratch_dir) for item in value]
    if isinstance(value, str) and value:
        root = scratch_dir.resolve()
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                return str(candidate.resolve().relative_to(root))
            except ValueError:
                return value
    return value


def _engine_block_frames(frame, blocks: int) -> list[tuple[int | None, Frame]]:
    """The engine blocks: the whole clone, or one scratch frame per clone index."""

    household = frame.table("household")
    if blocks < 1:
        raise ValueError("engine resolution blocks must be positive.")
    if blocks == 1:
        return [(None, frame)]
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
            {entity: block.weights_for(entity) for entity in block.weighted_entities},
            block.strata,
            mass_log=(*block.mass_log, subset_record),
            metadata=block.metadata,
        )
        block_frames.append((clone_index, block))

    return block_frames


def _run_engine_blocks(
    block_frames,
    national_registry,
    *,
    period: int,
    scratch_dir: Path,
    resolver_factory,
    local_grains: tuple[str, ...],
    on_block,
) -> tuple[
    dict[str, list[pd.DataFrame]],
    list[Mapping[str, Any]],
    list[dict[str, Any]],
    set[tuple[str, str]],
]:
    """Resolve every block's measures, hand each block's inputs to ``on_block``
    while its engine is alive, and release the engine before the next loads."""

    metric_parts: dict[str, list[pd.DataFrame]] = {grain: [] for grain in local_grains}
    resolver_receipts: list[Mapping[str, Any]] = []
    resolution_receipts: list[dict[str, Any]] = []
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
        resolution_receipts.append(
            _without_scratch_paths(dict(resolution.receipt), scratch_dir)
        )
        keys = set(resolution.measure_inputs)
        if national_input_keys is None:
            national_input_keys = keys
        elif keys != national_input_keys:
            raise RuntimeError(
                "per-clone engine resolution returned inconsistent national inputs."
            )
        on_block(block_frame, resolution.measure_inputs)
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
        resolver_receipts.append(
            _without_scratch_paths(dict(resolver.receipt()), scratch_dir)
        )
        # A policyengine simulation is a large cyclic object graph; the
        # collector does not reclaim it on `del`. Collect before the next
        # block loads so one engine is alive at a time.
        del resolution, resolver
        gc.collect()
        simulation_input = block_scratch / "simulation-input.h5"
        simulation_input.unlink(missing_ok=True)
        try:
            block_scratch.rmdir()
        except OSError:
            pass
    return (
        metric_parts,
        resolver_receipts,
        resolution_receipts,
        set() if national_input_keys is None else national_input_keys,
    )


def _rejoin_local_metrics(frame, metric_parts) -> dict[str, pd.DataFrame]:
    full_household_ids = frame.table("household")["household_id"].tolist()
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
    return local_metrics


def _block_provenance(resolver_receipts) -> tuple[Any, Any, Any]:
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
    return next(iter(modes)), next(iter(versions)), cgt_period_contract


def _measures_receipt(
    frame,
    *,
    mode,
    engine_version,
    cgt_period_contract,
    national_input_keys: set[tuple[str, str]],
    local_metrics: dict[str, pd.DataFrame],
    blocks: int,
    materialization_report,
    resolution_receipts,
) -> dict[str, Any]:
    receipt = {
        "mode": mode,
        "engine_version": engine_version,
        "households": len(frame.table("household")),
        "persons": len(frame.table("person")),
        "benunits": len(frame.table("benunit")),
        "national_inputs": len(national_input_keys),
        "local_metrics": {
            area_type: len(metrics.columns)
            for area_type, metrics in local_metrics.items()
        },
        "blocks": blocks,
        "target_materialization": materialization_report,
        # The resolution loop's own receipt per block (which measure came from
        # which provider, the rounds, the provider's receipt): the seam
        # manifest's ``measure_resolution`` block, one per engine block.
        "resolution": resolution_receipts,
    }
    if cgt_period_contract is not None:
        receipt["cgt_period_contract"] = cgt_period_contract
    if blocks > 1:
        receipt["deviation"] = "per_clone_block_engine_resolution"
        present = sorted(
            column
            for column in UK_BLOCK_SENSITIVE_MEASURE_COLUMNS
            if column in {variable for _, variable in national_input_keys}
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
    return receipt


def _national_rows(national_registry, targets) -> UKRowwiseNationalRows:
    return UKRowwiseNationalRows(
        targets=targets,
        registry=national_registry,
        families=tuple(sorted({spec.family for spec in national_registry.specs})),
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
    The dense role's measures node uses :func:`resolve_uk_full_national_problem`
    instead, which never materializes the whole pool.
    """

    block_frames = _engine_block_frames(frame, blocks)
    measure_parts: dict[tuple[str, str], list[pd.Series]] = {}

    def collect(block_frame, block_inputs):
        for (entity, variable), values in block_inputs.items():
            entity_table = block_frame.table(entity)
            entity_id = f"{entity}_id"
            measure_parts.setdefault((entity, variable), []).append(
                pd.Series(
                    np.asarray(values),
                    index=entity_table[entity_id].tolist(),
                )
            )

    metric_parts, resolver_receipts, resolution_receipts, _keys = _run_engine_blocks(
        block_frames,
        national_registry,
        period=period,
        scratch_dir=scratch_dir,
        resolver_factory=resolver_factory,
        local_grains=local_grains,
        on_block=collect,
    )

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

    local_metrics = _rejoin_local_metrics(frame, metric_parts)

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
    mode, engine_version, cgt_period_contract = _block_provenance(resolver_receipts)
    receipt = _measures_receipt(
        frame,
        mode=mode,
        engine_version=engine_version,
        cgt_period_contract=cgt_period_contract,
        national_input_keys=set(measure_inputs),
        local_metrics=local_metrics,
        blocks=blocks,
        materialization_report=materialized.report(),
        resolution_receipts=resolution_receipts,
    )
    try:
        scratch_dir.rmdir()
    except OSError:
        pass
    drop_injected_measure_inputs(adapter, measure_inputs, original_columns)
    national_rows = _national_rows(national_registry, national_registry.to_target_set())
    return (
        adapter.prepared_frame(),
        adapter.restore,
        national_rows,
        local_metrics,
        receipt,
    )


def _materialize_national_problem(
    block_frame,
    block_inputs,
    national_registry,
    targets,
    *,
    period: int,
    band_edge_registry,
) -> tuple[CalibrationProblem | None, dict[str, Any]]:
    """Materialize the national targets on one engine block and compile its
    columns of the national matrix, leaving the block untouched."""

    adapter = CalibrationFrameAdapter(block_frame)
    original_columns = {
        entity: set(table.columns) for entity, table in adapter.tables.items()
    }
    inject_measure_inputs(adapter, block_inputs)
    materialized = materialize_uk_ledger_targets(
        adapter,
        national_registry,
        period=period,
        band_edge_registry=band_edge_registry,
    )
    if materialized.skipped:
        raise RuntimeError(
            "candidate national target materialization skipped row(s): "
            f"{[skip.__dict__ for skip in materialized.skipped]}."
        )
    drop_injected_measure_inputs(adapter, block_inputs, original_columns)
    prepared = adapter.prepared_frame()
    problem = None
    if len(targets):
        problem = build_constraint_matrix(prepared, targets, "household")
        if problem.skipped:
            failures = "; ".join(
                f"{item.target.key}: {item.reason}" for item in problem.skipped
            )
            raise ValueError(
                "Selected national constraints failed to compile: " + failures
            )
    # The materialization is scratch state: restoring the prepared frame must
    # give the block's own tables back, column for column.
    clean = adapter.restore(prepared)
    for entity in block_frame.entities:
        if not clean.table(entity).equals(block_frame.table(entity)):
            raise RuntimeError(
                f"national target materialization altered the {entity} table."
            )
    return problem, materialized.report()


def _stitch_national_problems(frame, block_problems) -> CalibrationProblem:
    """One national problem over the pool from the per-block problems, columns
    in the pool's household order and the pool's weights as the start."""

    first, _ = block_problems[0]
    for problem, _ in block_problems[1:]:
        if (
            problem.names != first.names
            or problem.weight_entity != first.weight_entity
            or not np.array_equal(problem.target_vector, first.target_vector)
        ):
            raise RuntimeError(
                "per-clone national problems compiled different target rows."
            )
    household_ids = frame.table("household")["household_id"].to_numpy()
    position = pd.Index(household_ids)
    if not position.is_unique:
        raise RuntimeError("the pool's household ids are not unique.")
    stacked_ids = np.concatenate([ids for _, ids in block_problems])
    columns = position.get_indexer(stacked_ids)
    if (
        len(columns) != len(household_ids)
        or (columns < 0).any()
        or len(np.unique(columns)) != len(columns)
    ):
        raise RuntimeError(
            "per-clone national problems do not cover the pool's households "
            "exactly once."
        )
    stacked = sparse.hstack(
        [problem.matrix for problem, _ in block_problems], format="csc"
    )
    ordered = sparse.csr_array(stacked[:, np.argsort(columns)])
    return CalibrationProblem(
        matrix=ordered,
        target_vector=first.target_vector,
        names=first.names,
        initial_weights=frame.resolve_weights(first.weight_entity),
        weight_entity=first.weight_entity,
        targets=first.targets,
        skipped=first.skipped,
    )


def resolve_uk_full_national_problem(
    frame,
    national_registry,
    *,
    period: int,
    scratch_dir: Path,
    band_edge_registry=None,
    resolver_factory=UKMeasureResolver,
    blocks: int = 1,
    local_grains: tuple[str, ...] = ("constituency", "la"),
) -> tuple[
    CalibrationProblem | None,
    UKRowwiseNationalRows,
    dict[str, pd.DataFrame],
    dict[str, Any],
]:
    """Resolve the national constraint problem block by block on the cloned frame.

    The dense role's measures node calls this instead of
    :func:`resolve_uk_full_measures`: materializing every national target as a
    column on a K-clone pool, then copying the prepared pool to compile it and
    again to check it was left untouched, needs tens of gigabytes on a
    1.6 million-household pool (the first K=25 build, 2026-10-05, reached a
    140 GB footprint and was killed before the solve). Here each engine block
    materializes its own targets while its engine is alive, compiles its own
    columns of the national matrix and is released; the per-block matrices are
    stitched into the pool's household order. Materialization is row-wise
    (band edges come from the compiled register), so the stitched problem is
    the pool's problem. ``None`` when the registry selects no national target.
    """

    block_frames = _engine_block_frames(frame, blocks)
    targets = national_registry.to_target_set()
    band_edges = national_registry if band_edge_registry is None else band_edge_registry
    block_problems: list[tuple[CalibrationProblem, np.ndarray]] = []
    reports: list[dict[str, Any]] = []

    def materialize(block_frame, block_inputs):
        problem, report = _materialize_national_problem(
            block_frame,
            block_inputs,
            national_registry,
            targets,
            period=period,
            band_edge_registry=band_edges,
        )
        reports.append(report)
        if problem is not None:
            block_problems.append(
                (problem, block_frame.table("household")["household_id"].to_numpy())
            )

    metric_parts, resolver_receipts, resolution_receipts, national_input_keys = (
        _run_engine_blocks(
            block_frames,
            national_registry,
            period=period,
            scratch_dir=scratch_dir,
            resolver_factory=resolver_factory,
            local_grains=local_grains,
            on_block=materialize,
        )
    )
    if any(report != reports[0] for report in reports[1:]):
        raise RuntimeError(
            "per-clone national target materialization differs across blocks."
        )
    local_metrics = _rejoin_local_metrics(frame, metric_parts)
    national_problem = (
        _stitch_national_problems(frame, block_problems) if block_problems else None
    )
    mode, engine_version, cgt_period_contract = _block_provenance(resolver_receipts)
    receipt = _measures_receipt(
        frame,
        mode=mode,
        engine_version=engine_version,
        cgt_period_contract=cgt_period_contract,
        national_input_keys=national_input_keys,
        local_metrics=local_metrics,
        blocks=blocks,
        materialization_report=reports[0],
        resolution_receipts=resolution_receipts,
    )
    receipt["national_materialization"] = "per_engine_block"
    try:
        scratch_dir.rmdir()
    except OSError:
        pass
    return (
        national_problem,
        _national_rows(national_registry, targets),
        local_metrics,
        receipt,
    )
