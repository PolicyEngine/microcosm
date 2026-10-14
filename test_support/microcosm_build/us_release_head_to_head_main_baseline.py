"""Frozen original household scoring loop for sequential compatibility tests.

Copied verbatim from tools/score_us_release_head_to_head.py at origin/main
7f235941c. Keep this loop independent of the production slice scorer and
reducer: it is the reference for the refactor's array bits and receipts.
The unchanged surrounding scorer helpers and release seams remain shared.
"""

from __future__ import annotations

import gc
from collections.abc import Sequence

import numpy as np
from score_us_release_head_to_head import (
    ScoredChunk,
    _assert_full_chunk_surface,
    _assert_nothing_dropped,
    _assert_rss_below_limit,
    _canonical_sha256,
    _spec_keys,
    release,
    scored_column_contract,
)

from microcosm.calibrate import score_targets
from microcosm.frame import Frame


def _score_chunk_household_sliced(
    base_frame: Frame,
    chunk_specs: Sequence,
    *,
    chunk_loss_weights: np.ndarray,
    artifact_name: str,
    chunk_label: str,
    maximum_microsim_batch_size: int | None,
) -> ScoredChunk:
    """Materialize, score, and reduce one chunk without a dense full-pool table.

    Each household slice runs the canonical materializer and scorer with the
    population-aggregate guard armed when the pool spans multiple slices.
    Every slice must reproduce the exact target, scale, diagnostic-name, and
    scored-column contracts before its estimates enter the fixed-order sum.
    """

    n_households = base_frame.n("household")
    expected_keys = _spec_keys(chunk_specs)
    slice_batches = tuple(
        release._household_position_batches(
            n_households,
            maximum_microsim_batch_size,
        )
    )
    if not slice_batches:
        raise ValueError(f"{artifact_name} has no households to score.")
    if len(slice_batches) > 1:
        release._assert_group_entities_nest_in_households(base_frame)
        release._assert_medicaid_claiming_tax_units_local(base_frame)
    accumulated_estimates: np.ndarray | None = None
    reference_targets: np.ndarray | None = None
    reference_scales: np.ndarray | None = None
    reference_names: tuple[str, ...] | None = None
    reference_contract: tuple[tuple[str, str, str], ...] | None = None
    first_compilation: dict[str, object] | None = None
    compilation_digests: list[str] = []
    slice_sizes: list[int] = []
    for slice_index, positions in enumerate(slice_batches):
        slice_label = f"{chunk_label} slice {slice_index + 1}/{len(slice_batches)}"
        full_slice = len(positions) == n_households
        slice_frame = (
            base_frame
            if full_slice
            else release._select_households_by_position(base_frame, positions)
        )
        slice_target_frame, slice_registry, slice_compilation = (
            release._materialize_target_frame(
                slice_frame,
                chunk_specs,
                maximum_microsim_batch_size=maximum_microsim_batch_size,
                refuse_population_aggregates=True if len(slice_batches) > 1 else None,
                target_materialization_cache_dir=None,
                target_materialization_cache_context=None,
            )
        )
        _assert_nothing_dropped(
            artifact_name=f"{artifact_name} {slice_label}",
            compilation=slice_compilation,
        )
        # Slice sizes can differ; record them in household_slice_row_counts
        # instead of the shared compilation contract or its digest. The
        # size-independent population-aggregate guard receipt stays in both.
        slice_compilation = dict(slice_compilation)
        slice_compilation.pop("target_materialization_batching", None)
        if first_compilation is None:
            first_compilation = dict(slice_compilation)
        compilation_digests.append(_canonical_sha256(slice_compilation))
        slice_sizes.append(len(positions))
        if _spec_keys(slice_registry.specs) != expected_keys:
            raise ValueError(
                f"{artifact_name} {slice_label} compiled a different target "
                "contract than the chunk."
            )
        if slice_target_frame.n("household") != len(positions):
            raise RuntimeError(
                f"{artifact_name} {slice_label} returned "
                f"{slice_target_frame.n('household')} households for {len(positions)} "
                "positions."
            )
        slice_contract = scored_column_contract(
            slice_target_frame,
            chunk_specs,
            artifact_name=f"{artifact_name} {slice_label}",
        )
        result = score_targets(
            slice_target_frame,
            slice_registry.to_target_set(),
            target_loss_weights=chunk_loss_weights,
            target_loss_cap=release.US_FISCAL_TARGET_LOSS_CAP,
            options={
                "mass": "existing_weights",
                "target_loss_weighting": release.US_FISCAL_TARGET_LOSS_WEIGHTING,
                "maximum_microsim_batch_size": maximum_microsim_batch_size,
            },
        )
        _assert_full_chunk_surface(
            artifact_name=artifact_name,
            chunk_label=slice_label,
            chunk_specs=chunk_specs,
            materialized_registry=slice_registry,
            result=result,
        )
        expected_slice_weights = np.asarray(
            slice_frame.weights_for("household").values,
            dtype=np.float64,
        )
        if not np.array_equal(
            expected_slice_weights,
            np.asarray(result.weights, dtype=np.float64),
        ):
            raise RuntimeError(
                f"{artifact_name} {slice_label} scorer changed the shipped "
                "household weight vector."
            )
        estimates = np.asarray(
            [row.final_estimate for row in result.diagnostics],
            dtype=np.float64,
        )
        targets = np.asarray(
            [row.target for row in result.diagnostics],
            dtype=np.float64,
        )
        scales = np.asarray(result.target_loss_scales, dtype=np.float64)
        names = tuple(row.name for row in result.diagnostics)
        if accumulated_estimates is None:
            accumulated_estimates = estimates.copy()
            reference_targets = targets.copy()
            reference_scales = scales.copy()
            reference_names = names
            reference_contract = slice_contract
        else:
            if not np.array_equal(reference_targets, targets):
                raise RuntimeError(
                    f"{artifact_name} {slice_label} target vector differs from "
                    "the first household slice."
                )
            if not np.array_equal(reference_scales, scales):
                raise RuntimeError(
                    f"{artifact_name} {slice_label} loss-scale vector differs "
                    "from the first household slice."
                )
            if reference_names != names:
                raise RuntimeError(
                    f"{artifact_name} {slice_label} diagnostic names differ "
                    "from the first household slice."
                )
            if reference_contract != slice_contract:
                raise RuntimeError(
                    f"{artifact_name} {slice_label} scored-column contract "
                    "differs from the first household slice."
                )
            np.add(accumulated_estimates, estimates, out=accumulated_estimates)
        del slice_target_frame, slice_registry, result
        if not full_slice:
            del slice_frame
        gc.collect()
        _assert_rss_below_limit(f"after scoring {artifact_name} {slice_label}")
    if any(
        value is None
        for value in (
            accumulated_estimates,
            reference_targets,
            reference_scales,
            reference_names,
            reference_contract,
            first_compilation,
        )
    ):  # pragma: no cover - non-empty batches are enforced above
        raise RuntimeError(f"{artifact_name} {chunk_label} produced no slice result.")
    compilation = {
        **first_compilation,
        "household_slices": len(slice_batches),
        "household_slice_size": maximum_microsim_batch_size,
        "household_slice_row_counts": slice_sizes,
        "slice_compilation_sha256s": compilation_digests,
    }
    return ScoredChunk(
        estimates=accumulated_estimates,
        targets=reference_targets,
        scales=reference_scales,
        diagnostic_names=reference_names,
        scored_contract=reference_contract,
        compilation=compilation,
    )
