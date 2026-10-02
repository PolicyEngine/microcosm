#!/usr/bin/env python3
"""Run the US release's post-export stages on a household subsample of an export.

``tools/build_us_fiscal_refresh_release.py`` runs its post-export stages only
after a full build: Route A's attempt on build 310842b98 ran 49,304 s before
its reform-coverage smoke failed on four stale probe definitions (#1046), and
nothing after the smoke had ever run on a Route A export. This tool runs those
stages on a subsample written by ``tools/sample_us_export_households.py`` (or
on any written US release H5) through the release tool's own functions,
consumers, writers and file names:

1. ``stored_inputs``: the #1031 stored-input gate (``_stored_input_gate_failures``)
   on the loaded frame. The release's written-H5 premise compares the
   pre-write export frame with the written bytes, so it is recorded as not
   reproducible rather than run against itself.
2. ``qrf_tail_concentration``: the terminal QRF tail gate and register
   (``_qrf_tail_concentration_gate``, ``_qrf_tail_gate_lines``), which the
   release runs on the export frame just before the write.
3. ``take_up_participation``: ``us_take_up_participation_diagnostics`` and the
   stale count-calibrated check ``_main`` raises on.
4. ``reform_coverage_smoke``: the release's smoke consumer on the
   household-batched scorer (``_HouseholdBatchedPostExportScorer``), from the
   baseline plan the engine-free dry run records.
5. ``reform_validation``: ``_reform_validation_consumer``. With the build's
   ``calibration_diagnostics.json`` its in-sample JCT rows come from the fit,
   as in the release; without it they are simulated (one extra reform pass
   each, recorded).
6. ``demographics``: the age distribution and the geography coverage.
7. ``source_coverage``: ``us_source_coverage_diagnostics`` for the build's
   target surface (the reference release manifest's, unless
   ``--target-surface`` says otherwise).

The engine stages run in ``_main``'s order (smoke, validation, demographics on
one scorer, then source coverage). The frame-based stages (1-3) run first,
while the loaded frame is held, instead of after the scorer as in ``_main``:
they read only the frame, so the order changes no verdict. Unlike the
release, a stage that raises is recorded with its traceback and the next
stage still runs, so one probe names every failure the build would have
reached one at a time; each verdict's ``release_consequence`` says what the
release would do. The batch size gives at least three batches, so the
multi-batch path (and its population-aggregate refusals) runs. The report's
``not_reproduced`` block lists the release steps that need build state the
export does not carry.

**Which verdicts transfer to full scale.** A subsample is a stratified,
ratio-adjusted Horvitz-Thompson sample (see the sampler's docstring). Every
verdict carries an ``authority``:

- ``authoritative``: the verdict does not depend on the population's size (a
  stage that raised, a stored column name, a baseline plan), it was settled on
  the source export itself, or it is a smoke probe whose effect clears or
  misses its floor by at least ``--se-multiplier`` design-based standard
  errors with at least ``--min-effect-households`` drawn households carrying
  the effect.
- ``informational``: scale-dependent. A smoke probe with too few
  effect-bearing draws or within ``k`` standard errors of its floor, the QRF
  tail on the subsample (its top-k and minimum-carrier rules count records),
  unweighted record counts (demographics' ``n_under_50`` and ``n_under_100``,
  a state with no sampled record), a take-up column constant on the
  subsample, and every weighted estimate (validation budget effects, age
  bands, take-up shares).

The smoke's standard errors come from the gate's own arrays: each probe's
baseline and reform values are captured through the ``simulate`` seam and
summed per household with the engine's own row weights (policyengine-core
stores weights as float32 and uprates them to later periods, so the mapping
of rows to households is checked against one weight factor per key, not
equality). The per-household effects must add up to the gate's effect, or
no standard error is reported. They feed the linearized variance of the
sampler's estimator: zero for a certainty household, and
``N_h^2 (1 - n_h / N_h) / n_h * S^2(w (y - R_h))`` per stratum for the drawn
ones (the ratio estimator's Taylor linearization; exact when weights are equal
within a stratum).

``--reference-release-dir`` (a full-size build's ``releases/<id>/``) adds each
smoke probe's full-scale effect and z-score, a row-by-row validation
comparison, the build's target surface, its QRF register and its in-sample
fit, and two fidelity checks: the probe's baseline plans and reform pass
counts must equal the build manifest's, and the source export's QRF tail
result must equal the release's ``qrf_tail_concentration.json``. With
``--source-export`` (the receipt's source by default) the tool settles the
engine-free scale-dependent verdicts exactly on the source: QRF tail,
take-up diagnostics and geography record counts, read column by column.

Every stage records wall time, CPU time and peak resident memory (sampled
every 0.5 s with psutil, else the process peak); every batch-outer scoring
pass is one line of ``passes.jsonl``.

Usage::

    .venv/bin/python tools/probe_us_post_export.py \\
        --export <subsample dir>/populace_us_2024.h5 --out <dir> \\
        [--sample-receipt <subsample dir>/sample_receipt.json] \\
        [--reference-release-dir <release dir>/releases/<release id>]

writes ``<dir>/probe_report.json``, ``<dir>/passes.jsonl`` and each stage's
artifact under the release tool's file name (``reform_coverage_smoke.json``,
``reform_validation.json``, ``demographics.json``, ``us_source_coverage.json``,
``us_take_up_participation.json``, ``qrf_tail_concentration.json``). None of
them certifies anything.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

#: Bump with any change to the report layout or the authority rules.
PROBE_REPORT_SCHEMA_VERSION = 1
REPORT_FILENAME = "probe_report.json"
PASSES_FILENAME = "passes.jsonl"
#: Route A's ``--maximum-microsim-batch-size``; the per-batch cost then
#: matches the build's.
DEFAULT_BATCH_SIZE = 2_000
MINIMUM_BATCHES = 3
DEFAULT_SE_MULTIPLIER = 3.0
#: Drawn households with a nonzero effect needed for a reliable standard error.
DEFAULT_MIN_EFFECT_HOUSEHOLDS = 5
#: Tolerance of the per-household effect decomposition check, relative to
#: ``sum |w r| + sum |w b|`` (the same arrays summed in another order).
DECOMPOSITION_RTOL = 1e-9
#: Largest relative spread of the engine's row weights around one factor of
#: the household weights: float32 storage and a float32 uprating step each
#: round by at most 2^-24 (6e-8).
ROW_WEIGHT_SCALE_RTOL = 1e-6
STAGES = (
    "stored_inputs",
    "qrf_tail_concentration",
    "take_up_participation",
    "reform_coverage_smoke",
    "reform_validation",
    "demographics",
    "source_coverage",
)
#: Release steps after (or around) the export write the probe cannot run
#: standalone, and why; recorded in every report.
NOT_REPRODUCED = (
    {
        "step": "written-H5 stored-input premise (#1031)",
        "why": "compares the pre-write export frame's modeled stored tables with "
        "the written bytes; the probe has only the written bytes",
    },
    {
        "step": "calibration NPZ (_write_npz)",
        "why": "needs the calibration result and target registry",
    },
    {
        "step": "us_medicaid_take_up.json, us_snap_state_take_up.json, "
        "us_ssi_take_up.json",
        "why": "computed by pre-export stages from target specs, seeds and "
        "priors; the export does not carry their inputs",
    },
    {
        "step": "source coverage fiscal_target_sources, CD vintage crosswalk, "
        "fiscal-target exclusion receipt",
        "why": "need the compiled target specs and build metadata",
    },
    {
        "step": "release and build manifests (_build_manifests), publisher preflight",
        "why": "bind the build's gate evidence, calibration and artifacts",
    },
    {
        "step": "--audit-export-targets",
        "why": "needs the calibration result and target specs (and was off in "
        "the published Route A build)",
    },
)
AUTHORITATIVE = "authoritative"
INFORMATIONAL = "informational"

_TOOLS = Path(__file__).resolve().parent


def _load_tool(module_name: str, filename: str):
    """Import a sibling tool by path (``tools/`` is not a package).

    The module is registered in ``sys.modules`` before it executes, as
    ``dataclasses`` resolves string annotations through it; an already
    imported module of that name is reused.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _TOOLS / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_sampler():
    return _load_tool("sample_us_export_households", "sample_us_export_households.py")


def load_builder():
    return _load_tool(
        "build_us_fiscal_refresh_release", "build_us_fiscal_refresh_release.py"
    )


# ---------------------------------------------------------------------------
# Batch size
# ---------------------------------------------------------------------------


def choose_batch_size(
    n_households: int,
    requested: int | None = None,
    *,
    minimum_batches: int = MINIMUM_BATCHES,
) -> int:
    """The largest batch size up to ``requested`` giving ``minimum_batches``.

    ``_household_position_batches`` cuts ``n`` households into
    ``ceil(n / size)`` batches, which is at least ``m`` exactly when
    ``size < n / (m - 1)``; the largest such size is ``ceil(n / (m - 1)) - 1``
    (``n`` itself when ``m`` is 1). ``ceil(n / m)`` is not enough: four
    households in batches of two make two batches.
    """
    n_households = int(n_households)
    minimum_batches = int(minimum_batches)
    if minimum_batches < 1:
        raise ValueError(f"minimum_batches must be positive; got {minimum_batches}.")
    if n_households < minimum_batches:
        raise ValueError(
            f"{n_households} household(s) cannot fill {minimum_batches} batches; "
            "the probe needs the multi-batch scoring path."
        )
    if requested is not None and int(requested) < 1:
        raise ValueError(f"batch size must be positive; got {requested!r}.")
    if minimum_batches == 1:
        ceiling = n_households
    else:
        ceiling = -(-n_households // (minimum_batches - 1)) - 1
    return ceiling if requested is None else min(int(requested), ceiling)


def batch_count(n_households: int, batch_size: int) -> int:
    return math.ceil(int(n_households) / int(batch_size))


# ---------------------------------------------------------------------------
# The sample design, rebuilt from the subsample and its receipt
# ---------------------------------------------------------------------------


class SampleDesign:
    """What the variance formula needs, one entry per sampled household.

    Rebuilt from the subsample itself (household ids, adjusted weights, the
    stratum columns the receipt used) and the receipt (certainty ids and each
    stratum's eligible and drawn counts and weight factor). ``verify`` checks
    the rebuilt design against the receipt's counts and weight totals.
    """

    def __init__(
        self,
        *,
        household_ids: np.ndarray,
        labels: np.ndarray,
        certainty: np.ndarray,
        adjusted_weights: np.ndarray,
        source_weights: np.ndarray,
        strata: Mapping[str, Mapping[str, object]],
        fraction: float | None,
    ) -> None:
        self.household_ids = np.asarray(household_ids, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=object)
        self.certainty = np.asarray(certainty, dtype=bool)
        self.adjusted_weights = np.asarray(adjusted_weights, dtype=np.float64)
        self.source_weights = np.asarray(source_weights, dtype=np.float64)
        self.strata = {str(key): dict(value) for key, value in strata.items()}
        self.fraction = fraction
        n = len(self.household_ids)
        for name in ("labels", "certainty", "adjusted_weights", "source_weights"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"SampleDesign.{name} is not aligned to its ids.")
        if len(np.unique(self.household_ids)) != n:
            raise ValueError("SampleDesign household ids must be unique.")
        self._position = pd.Series(np.arange(n), index=self.household_ids)

    @property
    def is_census(self) -> bool:
        """True when nothing was sampled (a full export: zero design variance)."""
        return not self.strata or all(
            int(record["drawn_noncertainty_households"])
            == int(record["eligible_noncertainty_households"])
            for record in self.strata.values()
        )

    def aligned(self, values: pd.Series) -> np.ndarray:
        """``values`` (indexed by household id) in design order; absent = 0."""
        series = pd.Series(values, dtype=np.float64)
        unknown = series.index.difference(self._position.index)
        if len(unknown):
            raise ValueError(
                f"{len(unknown)} household id(s) are not in the sample design."
            )
        return series.reindex(self.household_ids).fillna(0.0).to_numpy()

    def per_unit(self, weighted: pd.Series) -> pd.Series:
        """``weighted[h] / W_h`` per sampled household: the ``y_h`` whose
        estimator ``sum_h W_h y_h`` is ``weighted.sum()``.

        A household weighing 0 has ``y = 0``; refused if it carries a nonzero
        weighted value (no weight can produce one).
        """
        aligned = self.aligned(weighted)
        zero = self.adjusted_weights == 0.0
        if np.any(aligned[zero] != 0.0):
            raise ValueError(
                "A zero-weight household carries a nonzero weighted effect."
            )
        out = np.zeros_like(aligned)
        out[~zero] = aligned[~zero] / self.adjusted_weights[~zero]
        return pd.Series(out, index=self.household_ids)

    def total(self, values: pd.Series) -> float:
        """The estimator ``sum_h W_h y_h`` over the sample (adjusted weights)."""
        return float(np.dot(self.adjusted_weights, self.aligned(values)))

    def variance(self, values: pd.Series) -> float:
        """Linearized design variance of :meth:`total` (``nan``: not estimable)."""
        return stratified_ratio_variance(
            self.aligned(values),
            source_weights=self.source_weights,
            labels=self.labels,
            certainty=self.certainty,
            strata=self.strata,
        )

    def verify(self) -> list[str]:
        """Disagreements between the rebuilt design and the receipt."""
        problems: list[str] = []
        present = set(self.labels.tolist())
        unknown = sorted(present - set(self.strata))
        if unknown:
            problems.append(f"sampled households in unreceipted strata {unknown}")
        for label, record in self.strata.items():
            in_stratum = self.labels == label
            drawn = int((in_stratum & ~self.certainty).sum())
            certain = int((in_stratum & self.certainty).sum())
            if drawn != int(record["drawn_noncertainty_households"]):
                problems.append(
                    f"stratum {label!r}: {drawn} drawn households in the "
                    f"subsample, {record['drawn_noncertainty_households']} "
                    "in the receipt"
                )
            if certain != int(record["certainty_households"]):
                problems.append(
                    f"stratum {label!r}: {certain} certainty households in the "
                    f"subsample, {record['certainty_households']} in the receipt"
                )
            realized = float(self.adjusted_weights[in_stratum].sum())
            source = float(record["source_weight_total"])
            if not math.isclose(realized, source, rel_tol=1e-9, abs_tol=0.0):
                problems.append(
                    f"stratum {label!r}: subsample weight total {realized!r} "
                    f"differs from the receipt's source total {source!r}"
                )
        return problems


def stratified_ratio_variance(
    values: np.ndarray,
    *,
    source_weights: np.ndarray,
    labels: np.ndarray,
    certainty: np.ndarray,
    strata: Mapping[str, Mapping[str, object]],
) -> float:
    """Linearized variance of the sampler's estimator of ``sum w_h y_h``.

    The estimator is ``sum_certainty w y + sum_h X_h * R_h`` with ``X_h`` the
    stratum's eligible non-certainty weight mass and ``R_h`` the drawn
    households' weighted mean of ``y`` (the sampler's ratio adjustment). A
    certainty household contributes no variance. Stratum ``h`` (``n`` of ``N``
    eligible households drawn without replacement) contributes
    ``N^2 (1 - n/N) / n * S^2(e)``, ``e = w (y - R_h)`` over its drawn
    households: the ratio estimator's first-order Taylor linearization.
    Returns ``nan`` when a stratum drew one household of several (``S^2`` is
    not estimable).
    """
    values = np.asarray(values, dtype=np.float64)
    source_weights = np.asarray(source_weights, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)
    certainty = np.asarray(certainty, dtype=bool)
    variance = 0.0
    for label, record in strata.items():
        drawn = (labels == label) & ~certainty
        n = int(drawn.sum())
        eligible = int(record["eligible_noncertainty_households"])
        if n == 0 or n >= eligible:
            continue
        if n < 2:
            return math.nan
        weights = source_weights[drawn]
        y = values[drawn]
        mass = float(weights.sum())
        if mass <= 0.0:
            # The sampler draws a weightless stratum only when all of it
            # weighs 0 (it refuses a zero-mass draw from a weighted stratum):
            # the stratum adds exactly 0 to every total.
            continue
        ratio = float(np.dot(weights, y)) / mass
        residual = weights * (y - ratio)
        variance += eligible**2 * (1.0 - n / eligible) / n * float(residual.var(ddof=1))
    return float(variance)


def design_from_sample(
    household: pd.DataFrame,
    person: pd.DataFrame,
    adjusted_weights: np.ndarray,
    receipt: Mapping[str, Any] | None,
    *,
    sampler,
) -> SampleDesign:
    """Rebuild the design of a subsample (or a census, with no receipt)."""
    household_ids = household["household_id"].to_numpy()
    adjusted = np.asarray(adjusted_weights, dtype=np.float64)
    if receipt is None:
        return SampleDesign(
            household_ids=household_ids,
            labels=np.full(len(household_ids), "all", dtype=object),
            certainty=np.ones(len(household_ids), dtype=bool),
            adjusted_weights=adjusted,
            source_weights=adjusted,
            strata={},
            fraction=None,
        )
    design = receipt["design"]
    used = list(design["strata_columns"]["used"])
    labels, _ = sampler.household_stratum_labels(household, person, used)
    certain_ids = np.asarray(receipt["certainty"]["household_ids"], dtype=np.int64)
    certainty = np.isin(household_ids, certain_ids)
    strata = receipt["strata"]
    factors = np.asarray(
        [
            1.0
            if is_certain
            else float(strata[str(label)]["noncertainty_weight_factor"])
            for label, is_certain in zip(labels, certainty, strict=True)
        ],
        dtype=np.float64,
    )
    return SampleDesign(
        household_ids=household_ids,
        labels=labels,
        certainty=certainty,
        adjusted_weights=adjusted,
        source_weights=adjusted / factors,
        strata=strata,
        fraction=float(design["fraction"]),
    )


# ---------------------------------------------------------------------------
# Capturing the smoke's arrays through the simulate seam
# ---------------------------------------------------------------------------


class SimulateRecorder:
    """Wraps a consumer's ``simulate`` seam and keeps every scored result.

    ``baseline`` holds the baseline simulation's results by
    ``(variable, period, map_to)``; ``reforms`` holds one such mapping per
    ``simulate(reform)`` call, in call order. The smoke gate builds one reform
    per probe, in probe order, so ``reforms[i]`` is probe ``i``'s.
    """

    def __init__(self, simulate: Callable[[Any], Any]) -> None:
        self._simulate = simulate
        self.baseline: dict[tuple[str, int, str | None], Any] = {}
        self.reforms: list[dict[tuple[str, int, str | None], Any]] = []

    def __call__(self, reform):
        simulation = self._simulate(reform)
        if reform is None:
            store = self.baseline
        else:
            store = {}
            self.reforms.append(store)
        return _RecordingSimulation(simulation, store)


class _RecordingSimulation:
    def __init__(self, simulation, store) -> None:
        self._simulation = simulation
        self._store = store

    def calculate(self, variable, period=None, map_to=None):
        if map_to is None:
            result = self._simulation.calculate(variable, period)
        else:
            result = self._simulation.calculate(variable, period, map_to=map_to)
        self._store[(str(variable), int(str(period)), map_to)] = result
        return result


def _as_float(values) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype == bool:
        return array.astype(np.float64)
    return np.asarray(pd.to_numeric(pd.Series(array), errors="coerce"), np.float64)


class HouseholdRowMap:
    """Household id of every engine row, per entity, over the scorer's batches.

    The scorer concatenates each batch engine's values in batch order; within
    a batch the engine keeps the dataset's table order, which is the batch
    frame's. ``households(entity)`` is that household id per row, and
    ``weights(entity)`` the batch frame's household weight of each row's
    household (float64, at the dataset's period).
    """

    def __init__(self, batch_frames: Sequence[Any], *, sampler) -> None:
        self._batches = tuple(batch_frames)
        self._sampler = sampler
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _build(self, entity: str) -> tuple[np.ndarray, np.ndarray]:
        if entity not in self._cache:
            households: list[np.ndarray] = []
            weights: list[np.ndarray] = []
            for batch in self._batches:
                person = batch.table("person")
                ids = (
                    None
                    if entity == "person"
                    else batch.table(entity)[f"{entity}_id"].to_numpy()
                )
                rows = np.asarray(
                    self._sampler.household_of_rows(entity, person, ids), np.int64
                )
                household_weight = pd.Series(
                    np.asarray(batch.weights_for("household").values, np.float64),
                    index=batch.table("household")["household_id"].to_numpy(),
                )
                households.append(rows)
                weights.append(household_weight.reindex(rows).to_numpy())
            self._cache[entity] = (np.concatenate(households), np.concatenate(weights))
        return self._cache[entity]

    def households(self, entity: str) -> np.ndarray:
        return self._build(entity)[0]

    def weights(self, entity: str) -> np.ndarray:
        return self._build(entity)[1]

    def n_rows(self, entity: str) -> int:
        return int(sum(batch.n(entity) for batch in self._batches))


def row_weight_scale(engine_weights, household_weights) -> float | None:
    """The one positive factor taking household weights to the engine's row
    weights, or ``None`` if no single factor does.

    policyengine-core stores ``household_weight`` as float32 and projects it to
    every row of every entity (``Microsimulation.get_weights``); at a period
    after the dataset's it uprates the weight by the population ratio
    (``household_weight``'s ``uprating``: 1.0165 from 2024 to 2026). So the
    engine's row weights equal the household weights up to one per-period
    factor and float32 rounding (relative 2^-24 per step). A row whose
    household weighs 0 must weigh 0.
    """
    engine = np.asarray(engine_weights, dtype=np.float64)
    household = np.asarray(household_weights, dtype=np.float64)
    if engine.shape != household.shape or not np.isfinite(engine).all():
        return None
    zero = household == 0.0
    if np.any(engine[zero] != 0.0):
        return None
    if zero.all():
        return 1.0
    ratios = engine[~zero] / household[~zero]
    scale = float(np.median(ratios))
    if not (math.isfinite(scale) and scale > 0.0):
        return None
    if float(np.max(np.abs(ratios / scale - 1.0))) > ROW_WEIGHT_SCALE_RTOL:
        return None
    return scale


class HouseholdEffects:
    """One probe's weighted effect per household and how it was mapped.

    ``weighted[h]`` is ``sum over household h's rows of w_row * (reform -
    baseline)`` with the engine's own row weights, so ``weighted.sum()`` is
    the gate's ``reform_total - baseline_total`` up to summation order. The
    design's estimator of that total is ``sum_h W_h y_h`` with ``W_h`` the
    subsample's household weight and ``y_h = weighted[h] / W_h``.
    """

    def __init__(
        self,
        weighted: pd.Series,
        *,
        entity: str,
        weight_scale: float,
        magnitude: float,
    ) -> None:
        self.weighted = weighted
        self.entity = entity
        self.weight_scale = weight_scale
        #: ``sum |w r| + sum |w b|``: the scale of the summation-order error.
        self.magnitude = magnitude


def household_effects(
    baseline,
    reform,
    *,
    entity: str | None,
    row_map: HouseholdRowMap,
    entities: Iterable[str],
) -> tuple[HouseholdEffects | None, str | None, str]:
    """One probe's per-household weighted effects, or why they are refused.

    Returns ``(effects, entity, reason)``. The measure's rows are mapped to
    households through the scorer's batch frames: the measure's entity (from
    the engine's variable metadata, or else every entity whose row count
    matches) must have one weight factor per row against its household's
    weight (:func:`row_weight_scale`), and an inferred entity must be
    unambiguous. A missing value counts as 0, as ``MicroSeries.sum`` skips it.
    """
    base_weights = np.asarray(getattr(baseline, "weights", None), dtype=np.float64)
    reform_weights = np.asarray(getattr(reform, "weights", None), dtype=np.float64)
    if base_weights.shape != reform_weights.shape or not np.array_equal(
        base_weights, reform_weights
    ):
        return None, entity, "baseline and reform rows carry different weights"
    n = len(base_weights)
    candidates = [entity] if entity is not None else list(entities)
    matches: list[tuple[str, float]] = []
    for candidate in candidates:
        if row_map.n_rows(candidate) != n:
            continue
        scale = row_weight_scale(base_weights, row_map.weights(candidate))
        if scale is not None:
            matches.append((candidate, scale))
    if not matches:
        return (
            None,
            entity,
            "no entity's batch-frame households reproduce the engine's row "
            "weights up to one factor",
        )
    households = row_map.households(matches[0][0])
    if any(
        not np.array_equal(row_map.households(other), households)
        for other, _ in matches[1:]
    ):
        return None, entity, f"row entity is ambiguous among {[m for m, _ in matches]}"
    base = np.nan_to_num(_as_float(baseline), nan=0.0) * base_weights
    reformed = np.nan_to_num(_as_float(reform), nan=0.0) * base_weights
    weighted = pd.Series(reformed - base).groupby(households).sum()
    effects = HouseholdEffects(
        weighted,
        entity=matches[0][0],
        weight_scale=matches[0][1],
        magnitude=float(np.abs(reformed).sum() + np.abs(base).sum()),
    )
    return effects, effects.entity, "decomposed"


# ---------------------------------------------------------------------------
# Authority rules
# ---------------------------------------------------------------------------


def classify_probe(
    *,
    signed_magnitude: float,
    floor: float,
    standard_error: float | None,
    take_all: bool,
    drawn_effect_households: int | None,
    census: bool,
    se_multiplier: float = DEFAULT_SE_MULTIPLIER,
    min_effect_households: int = DEFAULT_MIN_EFFECT_HOUSEHOLDS,
) -> tuple[str, str]:
    """``(authority, reason)`` for one smoke probe's verdict at this sample.

    Rules, in order:

    1. A census (no sample) is authoritative.
    2. No finite standard error is informational.
    3. Fewer than ``min_effect_households`` drawn (non-certainty) households
       with a nonzero effect is informational: ``S^2`` is estimated from the
       drawn households, so a sample in which none or few of them carry the
       effect estimates a variance near 0 whatever the population's is. The
       exception is a take-all probe (every pool carrier kept at its source
       weight) whose drawn households carry no effect at all: its effect is
       the certainty households' exactly, so its variance is 0.
    4. A margin to the floor of at least ``se_multiplier`` standard errors is
       authoritative; anything nearer is informational.
    """
    if census:
        return AUTHORITATIVE, "scored on the full export (no sample)"
    if standard_error is None or not math.isfinite(standard_error):
        return (
            INFORMATIONAL,
            "no design-based standard error for this probe, so a full-scale "
            "verdict cannot be bounded",
        )
    drawn = 0 if drawn_effect_households is None else int(drawn_effect_households)
    exact_take_all = take_all and drawn == 0
    if drawn < min_effect_households and not exact_take_all:
        return (
            INFORMATIONAL,
            f"only {drawn} drawn household(s) carry an effect (fewer than "
            f"{min_effect_households}); the standard error is not reliable",
        )
    margin = float(signed_magnitude) - float(floor)
    if exact_take_all:
        return (
            AUTHORITATIVE,
            "take-all probe: every pool carrier is a certainty household at its "
            "source weight and no drawn household carries an effect, so the "
            "effect has zero design variance",
        )
    if abs(margin) >= se_multiplier * standard_error:
        return (
            AUTHORITATIVE,
            f"the effect is {abs(margin) / standard_error:.1f} standard errors "
            f"{'above' if margin >= 0 else 'below'} the floor "
            f"(threshold {se_multiplier:g})"
            if standard_error > 0
            else "zero estimated design variance",
        )
    return (
        INFORMATIONAL,
        f"the effect is within {se_multiplier:g} standard errors of the floor "
        f"({abs(margin) / standard_error:.2f} SE)"
        if standard_error > 0
        else "the effect equals the floor",
    )


def signed_magnitude(effect: float, expected_sign: str) -> float:
    """The gate's direction-normalized magnitude (``us_reform_coverage_smoke_gate``)."""
    if expected_sign == "either":
        return abs(float(effect))
    return float(effect) if expected_sign == "positive" else -float(effect)


#: Reference-smoke fields that define a probe; a change in any of them since
#: the reference run makes its effect a different quantity.
PROBE_DEFINITION_FIELDS = (
    "budget_measure",
    "binding_inputs",
    "min_abs_effect",
    "expected_sign",
    "period",
)


def reference_comparison(
    row: Mapping[str, Any], reference: Mapping[str, Any], *, probe
) -> dict[str, Any]:
    """One probe's subsample effect against a full-size run's.

    ``same_definition`` compares the fields the reference recorded with the
    probe's current definition; ``z`` is the difference in standard errors
    (``None`` without one).
    """
    current = {
        "budget_measure": str(probe.budget_measure),
        "binding_inputs": [str(leaf) for leaf in probe.binding_inputs],
        "min_abs_effect": float(probe.min_abs_effect),
        "expected_sign": str(probe.expected_sign),
        "period": int(row["period"]),
    }
    recorded = {
        "budget_measure": str(reference.get("budget_measure")),
        "binding_inputs": [str(leaf) for leaf in reference.get("binding_inputs", ())],
        "min_abs_effect": float(reference.get("min_abs_effect", math.nan)),
        "expected_sign": str(reference.get("expected_sign")),
        "period": int(reference.get("period", -1)),
    }
    changed = [
        field for field in PROBE_DEFINITION_FIELDS if current[field] != recorded[field]
    ]
    effect = float(row["effect"])
    reference_effect = float(reference["effect"])
    standard_error = row.get("standard_error")
    return {
        "effect": reference_effect,
        "passed": bool(reference["passed"]),
        "same_definition": not changed,
        "changed_fields": changed,
        "difference": effect - reference_effect,
        "relative_difference": None
        if reference_effect == 0.0
        else (effect - reference_effect) / abs(reference_effect),
        "z": None
        if not standard_error
        else (effect - reference_effect) / float(standard_error),
        "verdict_agrees": bool(row["passed"]) == bool(reference["passed"]),
    }


def calibration_result_from_diagnostics(payload: Mapping[str, Any]) -> SimpleNamespace:
    """The calibration result the release's in-sample rows read, rebuilt from
    a build's ``calibration_diagnostics.json``.

    ``_in_sample_estimates`` and ``_in_sample_targets`` read each target's
    ``name`` and its diagnostic's ``final_estimate`` and ``target``; the
    diagnostics file records them as ``target_name``, ``final_estimate`` and
    ``target`` (``microcosm.calibrate.diagnostics._target_row``).
    """
    rows = list(payload["targets"])
    return SimpleNamespace(
        diagnostics=[
            SimpleNamespace(
                final_estimate=row.get("final_estimate"), target=row.get("target")
            )
            for row in rows
        ],
        problem=SimpleNamespace(
            targets=[SimpleNamespace(name=str(row["target_name"])) for row in rows]
        ),
    )


def _plan_digest(plan: Sequence[Any]) -> str:
    return hashlib.sha256(
        json.dumps([list(map(str, key)) for key in plan], sort_keys=True).encode()
    ).hexdigest()


def compare_reform_validation(
    payload: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    """Subsample budget effects against a full-size run's, row by row."""

    def effects(document) -> dict[str, float | None]:
        return {
            str(row["id"]): (row.get("microcosm") or {}).get("budget_effect")
            for row in document.get("reforms", ())
        }

    mine = effects(payload)
    theirs = effects(reference)
    relative: dict[str, float] = {}
    for reform_id in sorted(set(mine) & set(theirs)):
        left, right = mine[reform_id], theirs[reform_id]
        if left is None or right is None or float(right) == 0.0:
            continue
        relative[reform_id] = (float(left) - float(right)) / abs(float(right))
    magnitudes = sorted(abs(value) for value in relative.values())
    return {
        "rows_compared": len(relative),
        "only_in_probe": sorted(set(mine) - set(theirs)),
        "only_in_reference": sorted(set(theirs) - set(mine)),
        "median_abs_relative_difference": None
        if not magnitudes
        else magnitudes[len(magnitudes) // 2],
        "max_abs_relative_difference": None if not magnitudes else magnitudes[-1],
        "relative_difference": relative,
        "authority": INFORMATIONAL,
    }


def compare_qrf_tail(
    result: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    """The probe's source-export QRF tail result against a release's
    ``qrf_tail_concentration.json`` (same frame columns and weights, so the
    shares must agree to rounding)."""
    differences: list[str] = []
    tail = reference["tail_concentration"]
    if bool(tail["passed"]) != bool(result["passed"]):
        differences.append(
            f"verdict {result['passed']} vs the release's {tail['passed']}"
        )
    mine = result["surface"].get("checked_sparse_columns", [])
    theirs = reference["surface"].get("checked_sparse_columns", [])
    if sorted(mine) != sorted(theirs):
        differences.append(
            f"checked columns differ: only probe {sorted(set(mine) - set(theirs))}, "
            f"only release {sorted(set(theirs) - set(mine))}"
        )
    shares = result["details"].get("top_share", {}) or {}
    reference_shares = tail["details"].get("top_share", {}) or {}
    for column in sorted(set(shares) & set(reference_shares)):
        left, right = shares[column], reference_shares[column]
        if left is None or right is None:
            if left != right:
                differences.append(f"{column}: top share {left} vs {right}")
            continue
        if not math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12):
            differences.append(f"{column}: top share {left!r} vs {right!r}")
    return {"differences": differences, "columns_compared": len(shares)}


def compare_post_export_plans(
    records: Mapping[str, Mapping[str, Any]],
    reference_manifest: Mapping[str, Any],
) -> list[str]:
    """Differences between the probe's scoring records and the plans a
    release's ``build_manifest.json`` recorded (``post_export_scoring``)."""
    reference = (reference_manifest.get("post_export_scoring") or {}).get(
        "consumers", {}
    )
    problems: list[str] = []
    for name, record in records.items():
        recorded = reference.get(name)
        if recorded is None:
            problems.append(f"{name}: the reference release recorded no plan")
            continue
        if record["baseline_plan"] != recorded["baseline_plan"]:
            mine = {
                json.dumps(key, sort_keys=True)
                for key in record["baseline_plan"]["keys"]
            }
            theirs = {
                json.dumps(key, sort_keys=True)
                for key in recorded["baseline_plan"]["keys"]
            }
            problems.append(
                f"{name}: baseline plan differs ({len(mine - theirs)} key(s) only "
                f"in the probe, {len(theirs - mine)} only in the reference)"
            )
        for field in ("reform_passes", "reform_systems"):
            if int(record[field]) != int(recorded[field]):
                problems.append(
                    f"{name}: {field} {record[field]} vs the reference's "
                    f"{recorded[field]}"
                )
    return problems


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class RssSampler:
    """Peak resident memory of this process since the last reset.

    Samples ``psutil``'s current RSS every ``interval`` seconds. Without
    psutil (it comes with policyengine-core, so an engine-free environment
    lacks it) the peak is the process's lifetime ``ru_maxrss``, which cannot
    be reset; ``source`` says which one a report holds.
    """

    def __init__(self, interval: float = 0.5) -> None:
        try:
            import psutil
        except ImportError:
            self._process = None
            self.source = "ru_maxrss (process lifetime peak; psutil absent)"
        else:
            self._process = psutil.Process()
            self.source = "psutil rss sampled every 0.5 s"
        self._interval = interval
        self._peak = 0
        self._lock = threading.Lock()
        if self._process is not None:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            rss = self._process.memory_info().rss
            with self._lock:
                self._peak = max(self._peak, rss)
            time.sleep(self._interval)

    def current(self) -> int:
        if self._process is None:
            return _lifetime_peak_rss()
        return int(self._process.memory_info().rss)

    def reset(self) -> int:
        with self._lock:
            self._peak = self.current()
            return self._peak

    def peak(self) -> int:
        with self._lock:
            return max(self._peak, self.current())


def _lifetime_peak_rss() -> int:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _gib(value: float) -> float:
    return round(float(value) / 2**30, 3)


class StageClock:
    """Runs each stage, recording wall/CPU/peak RSS and any exception."""

    def __init__(self, sampler: RssSampler | None) -> None:
        self.sampler = sampler
        self.current = "setup"

    def run(self, name: str, fn: Callable[[], Mapping[str, Any]]) -> dict[str, Any]:
        self.current = name
        before = self.sampler.reset() if self.sampler else 0
        wall = time.perf_counter()
        cpu = time.process_time()
        record: dict[str, Any]
        try:
            record = {"status": "completed", **dict(fn() or {})}
        except Exception as error:  # every stage failure is a finding
            record = {
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        record["timing"] = {
            "wall_seconds": round(time.perf_counter() - wall, 3),
            "cpu_seconds": round(time.process_time() - cpu, 3),
            "rss_before_gib": _gib(before),
            "peak_rss_gib": _gib(self.sampler.peak()) if self.sampler else None,
        }
        print(
            f"[probe] {name}: {record['status']} in "
            f"{record['timing']['wall_seconds']:.1f} s",
            flush=True,
        )
        return record


def _timed_scorer_class(builder, passes_path: Path, clock: StageClock):
    """The release scorer, with one ``passes.jsonl`` line per scoring pass."""

    base = builder._HouseholdBatchedPostExportScorer

    class TimedScorer(base):
        def _score(self, keys, *, label, reform_system=None):
            sampler = clock.sampler
            before = sampler.reset() if sampler else 0
            wall = time.perf_counter()
            cpu = time.process_time()
            try:
                return super()._score(keys, label=label, reform_system=reform_system)
            finally:
                row = {
                    "stage": clock.current,
                    "label": label,
                    "keys": len(keys),
                    "reform": reform_system is not None,
                    "batches": self.n_batches,
                    "wall_seconds": round(time.perf_counter() - wall, 3),
                    "cpu_seconds": round(time.process_time() - cpu, 3),
                    "rss_before_gib": _gib(before),
                    "peak_rss_gib": _gib(sampler.peak()) if sampler else None,
                }
                with passes_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")

    return TimedScorer


# ---------------------------------------------------------------------------
# Source settlement (engine-free, column by column)
# ---------------------------------------------------------------------------


def source_columns_frame(
    source_path: Path,
    columns: Iterable[str],
    *,
    sampler,
    chunk_bytes: int,
    chunk_rows: int | None = None,
):
    """The source export restricted to ids, memberships, household weights and
    ``columns`` (each read from whichever entity table stores it).

    The take-up diagnostics and the QRF tail gate read only their own columns
    and the household weights (resolved to each entity), so this frame gives
    their exact full-scale results without loading the source's 900k-row
    person table whole. Columns no table stores are simply absent, as they
    are from the source.
    """
    from microcosm.frame import Frame, WeightKind, Weights
    from microcosm.frame.units import US_SCHEMA

    wanted = set(columns)
    tables: dict[str, pd.DataFrame] = {}
    with pd.HDFStore(str(source_path), mode="r") as store:
        for entity in sampler.US_ENTITIES:
            stored = sampler.table_columns(store, entity)
            structural = (
                ["person_id", *sampler.PERSON_MEMBERSHIP_COLUMNS]
                if entity == "person"
                else [f"{entity}_id"]
            )
            if entity == "household":
                structural.append(sampler.HOUSEHOLD_WEIGHT_COLUMN)
            selected = [*structural, *[column for column in stored if column in wanted]]
            tables[entity] = sampler.read_table_rows(
                store,
                entity,
                columns=list(dict.fromkeys(selected)),
                chunk_bytes=chunk_bytes,
                chunk_rows=chunk_rows,
            )
    weights = tables["household"].pop(sampler.HOUSEHOLD_WEIGHT_COLUMN)
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(weights.to_numpy(np.float64), WeightKind.CALIBRATED)},
    )


def split_take_up_gate_failures(
    failures: Iterable[str],
) -> tuple[list[str], list[str], list[str]]:
    """``us_take_up_signal_gate`` failures as (missing, constant, share band)."""
    missing: list[str] = []
    constant: list[str] = []
    band: list[str] = []
    for failure in failures:
        if "missing seeded take-up column" in failure:
            missing.append(failure)
        elif "constant column" in failure:
            constant.append(failure)
        else:
            band.append(failure)
    return missing, constant, band


def stale_count_calibrated(payload: Mapping[str, Any]) -> list[str]:
    """``_main``'s stale count-calibrated check over a take-up payload."""
    return [
        str(row["variable"])
        for row in payload["programs"]
        if row.get("populace_treatment") == "count_calibrated"
        and row.get("ships_at_engine_default")
    ]


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------


def _verdict(
    stage: str,
    check: str,
    verdict: str,
    authority: str,
    reason: str,
    consequence: str,
) -> dict[str, str]:
    return {
        "stage": stage,
        "check": check,
        "verdict": verdict,
        "authority": authority,
        "reason": reason,
        "release_consequence": consequence,
    }


def _tool_source() -> dict[str, object]:
    import subprocess

    root = _TOOLS.parent
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--",
                "tools",
                "packages",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": head, "dirty": bool(dirty)}


def _package_versions() -> dict[str, str | None]:
    import importlib.metadata as metadata

    versions: dict[str, str | None] = {}
    for name in ("policyengine-us", "policyengine-core", "numpy", "pandas"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


class ExportProbe:
    """One probe run: the stages, their records and the verdicts.

    Construct it, then :meth:`run`. Stage methods run in ``_main``'s order;
    the frame-based stages (stored inputs, take-up) run while the loaded frame
    is held and the frame is freed before the scorer loads its own copy.
    """

    def __init__(
        self,
        export_path: Path | str,
        out_dir: Path | str,
        *,
        sample_receipt: Mapping[str, Any] | None = None,
        receipt_path: Path | None = None,
        source_export: Path | None = None,
        source_settlement: bool = True,
        stages: Sequence[str] = STAGES,
        batch_size: int | None = DEFAULT_BATCH_SIZE,
        target_surface: str = "full",
        target_surface_source: str = "default",
        dropped_congressional_district_targets: int | None = None,
        reference_smoke: Mapping[str, Any] | None = None,
        reference_build_manifest: Mapping[str, Any] | None = None,
        reference_validation: Mapping[str, Any] | None = None,
        calibration_diagnostics: Mapping[str, Any] | None = None,
        reference_qrf_tail: Mapping[str, Any] | None = None,
        qrf_tail_register: Mapping[str, str] | None = None,
        reference_paths: Mapping[str, str | None] | None = None,
        se_multiplier: float = DEFAULT_SE_MULTIPLIER,
        min_effect_households: int = DEFAULT_MIN_EFFECT_HOUSEHOLDS,
        release_id: str | None = None,
        probes: Sequence[Any] | None = None,
        builder=None,
        sampler=None,
        scorer_options: Mapping[str, Any] | None = None,
        load_frame: Callable[..., Any] | None = None,
        measure_entity: Callable[[str], str | None] | None = None,
        sample_rss: bool = True,
        chunk_bytes: int | None = None,
        chunk_rows: int | None = None,
    ) -> None:
        self.builder = builder or load_builder()
        self.sampler = sampler or load_sampler()
        unknown = sorted(set(stages) - set(STAGES))
        if unknown:
            raise ValueError(f"unknown stage(s) {unknown}; choose from {list(STAGES)}.")
        if target_surface not in self.builder.TARGET_SURFACE_MODES:
            raise ValueError(
                f"target_surface must be one of {self.builder.TARGET_SURFACE_MODES}."
            )
        if not (math.isfinite(se_multiplier) and se_multiplier > 0):
            raise ValueError("se_multiplier must be a positive finite number.")
        self.export_path = Path(export_path).resolve()
        self.out_dir = Path(out_dir).resolve()
        self.receipt = sample_receipt
        self.census = sample_receipt is None
        if not source_settlement:
            source_export = None
        elif source_export is None and sample_receipt is not None:
            candidate = Path(str(sample_receipt["source"]["path"]))
            source_export = candidate if candidate.exists() else None
        self.source_export = (
            None if source_export is None else Path(source_export).resolve()
        )
        self.stages = tuple(stages)
        self.batch_size_requested = batch_size
        self.target_surface = target_surface
        self.target_surface_source = target_surface_source
        self.dropped_cd_targets = dropped_congressional_district_targets
        self.reference_smoke = reference_smoke
        self.reference_build_manifest = reference_build_manifest
        self.reference_validation = reference_validation
        self.calibration_diagnostics = calibration_diagnostics
        self.reference_qrf_tail = reference_qrf_tail
        if qrf_tail_register is None and reference_qrf_tail is not None:
            qrf_tail_register = reference_qrf_tail["surface"]["reviewed_exclusions"]
        self.qrf_tail_register = (
            None if qrf_tail_register is None else dict(qrf_tail_register)
        )
        self.reference_paths = dict(reference_paths or {})
        self.se_multiplier = float(se_multiplier)
        self.min_effect_households = int(min_effect_households)
        self.release_id = release_id
        self.custom_probes = probes is not None
        if probes is None:
            from microcosm.build.us_runtime.release_input_coverage import (
                us_release_reform_coverage_probes,
            )

            probes = us_release_reform_coverage_probes()
        self.probes = tuple(probes)
        self.scorer_options = dict(scorer_options or {})
        self.load_frame = load_frame
        self.measure_entity = measure_entity
        self.chunk_bytes = int(chunk_bytes or self.sampler.DEFAULT_CHUNK_BYTES)
        self.chunk_rows = chunk_rows
        self.clock = StageClock(RssSampler() if sample_rss else None)
        self.passes_path = self.out_dir / PASSES_FILENAME
        self.frame = None
        self.scorer = None
        self.sha256: str | None = None
        self.design: SampleDesign | None = None
        self.design_problems: list[str] = []
        self.batch_size: int | None = None
        self.carriers_by_probe: dict[str, np.ndarray] = {}
        self.verdicts: list[dict[str, str]] = []
        self.report: dict[str, Any] = self._report_header(receipt_path)

    # ---- plumbing ---------------------------------------------------------

    def _report_header(self, receipt_path: Path | None) -> dict[str, Any]:
        receipt = self.receipt
        return {
            "schema_version": PROBE_REPORT_SCHEMA_VERSION,
            "tool": "tools/probe_us_post_export.py",
            "tool_source": _tool_source(),
            "packages": _package_versions(),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "host": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
            },
            "classification": "probe diagnostics (not release evidence)",
            "export": str(self.export_path),
            "stages_requested": list(self.stages),
            "rules": {
                "se_multiplier": self.se_multiplier,
                "min_effect_households": self.min_effect_households,
                "authoritative": (
                    "does not depend on the population's size, clears or misses "
                    "its floor by at least se_multiplier design-based standard "
                    "errors, or was computed on the source export"
                ),
                "informational": (
                    "scale-dependent at this sample: few sampled carriers, "
                    "within se_multiplier standard errors of the floor, an "
                    "unweighted record count, a column constant on the "
                    "subsample, or a weighted estimate"
                ),
            },
            "sample": None
            if receipt is None
            else {
                "receipt_path": None if receipt_path is None else str(receipt_path),
                "fraction": receipt["design"]["fraction"],
                "seed": receipt["design"]["seed"],
                "certainty_threshold": receipt["design"]["certainty_threshold"],
                "source_sha256": receipt["source"]["sha256"],
                "source_households": receipt["source"]["rows"]["household"],
                "sampled_households": receipt["selection"]["households"],
                "certainty_households": receipt["selection"]["certainty_households"],
                "subsample_sha256": receipt["output"]["sha256"],
            },
            "source_export": None
            if self.source_export is None
            else str(self.source_export),
            "target_surface": {
                "mode": self.target_surface,
                "dropped_congressional_district_targets": self.dropped_cd_targets,
                "source": self.target_surface_source,
            },
            "not_reproduced": [dict(step) for step in NOT_REPRODUCED],
            "references": {
                "paths": self.reference_paths,
                "smoke": self.reference_smoke is not None,
                "validation": self.reference_validation is not None,
                "build_manifest": self.reference_build_manifest is not None,
                "calibration_diagnostics": self.calibration_diagnostics is not None,
                "qrf_tail": self.reference_qrf_tail is not None,
            },
            "stages": {},
            "verdicts": self.verdicts,
        }

    def _write_report(self) -> None:
        _write_json(self.out_dir / REPORT_FILENAME, self.report)

    def _stage(self, name: str, fn: Callable[[], Mapping[str, Any]], consequence):
        record = self.clock.run(name, fn)
        self.report["stages"][name] = record
        if record["status"] == "error":
            self.verdicts.append(
                _verdict(
                    name,
                    "the stage runs",
                    "error",
                    AUTHORITATIVE,
                    "the stage raised (a code or definition failure does not "
                    f"depend on the sample): {record['error']}",
                    consequence,
                )
            )
        self._write_report()
        return record

    def _add(self, *args) -> None:
        self.verdicts.append(_verdict(*args))

    def _release_id(self) -> str:
        return self.release_id or f"probe-{(self.sha256 or 'unknown')[:12]}"

    # ---- run --------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.passes_path.write_text("")
        load = self._stage("load", self._load, "raises")
        if load["status"] == "completed":
            if "stored_inputs" in self.stages:
                self._stage("stored_inputs", self._stored_inputs, "raises")
            if "qrf_tail_concentration" in self.stages:
                self._stage("qrf_tail_concentration", self._qrf_tail, "raises")
            if "take_up_participation" in self.stages:
                self._stage("take_up_participation", self._take_up, "raises")
            # Free the loaded frame before the scorer loads its own copy.
            self.frame = None
            engine = [
                stage
                for stage in (
                    "reform_coverage_smoke",
                    "reform_validation",
                    "demographics",
                )
                if stage in self.stages
            ]
            if engine:
                self._stage("open_scorer", self._open_scorer, "raises")
            if self.scorer is not None:
                try:
                    if "reform_coverage_smoke" in self.stages:
                        self._stage("reform_coverage_smoke", self._smoke, "raises")
                    if "reform_validation" in self.stages:
                        self._stage("reform_validation", self._validation, "raises")
                    if "demographics" in self.stages:
                        self._stage("demographics", self._demographics, "raises")
                finally:
                    builder = self.builder
                    self.report["post_export_scoring"] = (
                        builder._post_export_scoring_manifest_block(self.scorer)
                    )
                    self.report["scored_dataset_sha256"] = (
                        builder._close_post_export_scorer(self.scorer)
                    )
                    self.scorer = None
                self._compare_plans()
            if "source_coverage" in self.stages:
                self._stage("source_coverage", self._source_coverage, "recorded")
        self.report["summary"] = _summary(self.report, self.receipt)
        self.report["process_peak_rss_gib"] = _gib(_lifetime_peak_rss())
        self.report["rss_source"] = (
            None if self.clock.sampler is None else self.clock.sampler.source
        )
        self._write_report()
        return self.report

    def _compare_plans(self) -> None:
        """Did the probe score the plans the reference release recorded?

        A consumer's baseline plan and its reform pass and system counts do
        not depend on the data, so a difference means the probe's code or
        inputs (probes, specs, calibration result) differ from the build's.
        """
        reference = self.reference_build_manifest
        scoring = self.report.get("post_export_scoring")
        if reference is None or scoring is None:
            return
        records = {
            name: record
            for name, record in scoring["consumers"].items()
            if not (
                name == "reform_validation" and self.calibration_diagnostics is None
            )
        }
        problems = compare_post_export_plans(records, reference)
        self.report["reference_plan_comparison"] = {
            "consumers": sorted(records),
            "problems": problems,
        }
        self._add(
            "post_export_scoring",
            "baseline plans and reform passes match the reference release's "
            "build manifest",
            "fail" if problems else "pass",
            AUTHORITATIVE,
            "; ".join(problems)
            or f"identical for {', '.join(sorted(records))} (plans do not "
            "depend on the data)",
            "fidelity of this probe to the release",
        )

    # ---- load ---------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        sampler = self.sampler
        self.sha256 = sampler.sha256_file(self.export_path)
        if self.receipt is not None and self.receipt["output"]["sha256"] != self.sha256:
            raise ValueError(
                f"{self.export_path} (SHA-256 {self.sha256}) is not the subsample "
                f"the receipt describes ({self.receipt['output']['sha256']})."
            )
        load = self.load_frame or self.builder._load_frame
        frame = load(self.export_path, expected_sha256=self.sha256)
        self.frame = frame
        self.design = design_from_sample(
            frame.table("household"),
            frame.table("person"),
            frame.weights_for("household").values,
            self.receipt,
            sampler=sampler,
        )
        self.design_problems = [] if self.census else self.design.verify()
        if not self.census:
            self._add(
                "load",
                "the sample design rebuilt from the subsample verifies against "
                "its receipt",
                "fail" if self.design_problems else "pass",
                AUTHORITATIVE,
                "; ".join(self.design_problems)
                or "strata, certainty households, draws and weight totals agree "
                "(without this no standard error is reported)",
                "probe integrity",
            )
        self._verify_source()
        n = int(frame.n("household"))
        self.batch_size = choose_batch_size(n, self.batch_size_requested)
        for probe in self.probes:
            self.carriers_by_probe[str(probe.id)] = probe_carrier_households(
                frame, probe.binding_inputs, sampler=sampler
            )
        return {
            "sha256": self.sha256,
            "rows": {entity: int(frame.n(entity)) for entity in frame.entities},
            "household_weight_total": float(self.design.adjusted_weights.sum()),
            "batch_size": self.batch_size,
            "n_batches": batch_count(n, self.batch_size),
            "design": {
                "census": self.census,
                "strata": len(self.design.strata),
                "verified_against_receipt": not self.design_problems,
                "problems": self.design_problems,
            },
        }

    def _verify_source(self) -> None:
        """Bind the source export to the receipt's digest before any verdict
        is settled on it; a different file disables source settlement."""
        if self.source_export is None or self.receipt is None:
            return
        expected = str(self.receipt["source"]["sha256"])
        observed = self.sampler.sha256_file(self.source_export)
        self.report["source_export_sha256"] = observed
        if observed == expected:
            return
        self._add(
            "load",
            "the source export is the receipt's source",
            "fail",
            AUTHORITATIVE,
            f"{self.source_export} has SHA-256 {observed}, the receipt's source "
            f"{expected}; no verdict is settled on it",
            "probe integrity",
        )
        self.source_export = None

    # ---- 1. stored inputs (#1031) -------------------------------------------

    def _stored_inputs(self) -> dict[str, Any]:
        builder = self.builder
        failures, details = builder._stored_input_gate_failures(
            self.frame, stage="export frame"
        )
        self._add(
            "stored_inputs",
            "stored-input gate (#1031): stored columns the installed engine "
            "does not define",
            "fail" if failures else "pass",
            AUTHORITATIVE,
            "column names do not depend on the sample, and the sampler verified "
            "that the subsample stores exactly the source's tables and columns: "
            + ("; ".join(failures) or "no refused stored column"),
            "raises",
        )
        # The release's written-H5 premise compares the writer's model of the
        # in-memory export frame with the bytes the writer produced. Here the
        # frame is loaded from the bytes being graded, so that comparison
        # holds by construction and proves nothing; it is recorded, not run.
        self._add(
            "stored_inputs",
            "written-H5 stored-input premise (#1031)",
            "not_run",
            INFORMATIONAL,
            "not reproducible standalone: it compares the pre-write export "
            "frame's modeled stored tables with the written bytes, and the "
            "probe has only the written bytes",
            "raises",
        )
        return {"gate_failures": failures, "gate_details": details}

    # ---- QRF tail concentration (pre-export terminal gate) ---------------------

    def _qrf_tail(self) -> dict[str, Any]:
        builder = self.builder
        register = dict(self.qrf_tail_register or {})
        register_source = (
            "none: every concentrated column fails"
            if self.qrf_tail_register is None
            else f"{len(register)} reviewed exclusion(s)"
        )

        def evaluate(frame) -> dict[str, Any]:
            gate, surface = builder._qrf_tail_concentration_gate(
                frame, reviewed_exclusions=register
            )
            mismatch = builder._qrf_tail_register_mismatch(register, gate)
            # The release's own terminal lines (no --allow-qrf-tail-concentration).
            gate_lines, register_lines = builder._qrf_tail_gate_lines(
                gate, mismatch, allow_concentration=False
            )
            failures = [*gate_lines, *register_lines]
            return {
                "passed": not failures,
                "failures": failures,
                "register_mismatch": mismatch,
                "details": dict(gate.details),
                "surface": surface,
            }

        subsample = evaluate(self.frame)
        record: dict[str, Any] = {"register": register_source, "subsample": subsample}
        self._add(
            "qrf_tail_concentration",
            "QRF tail gate and register (subsample)",
            "pass" if subsample["passed"] else "fail",
            AUTHORITATIVE if self.census else INFORMATIONAL,
            "top_k = 100 and min_nonzero_records = 500 are record counts, so "
            "the subsample checks different columns and tails than the pool: "
            + ("; ".join(subsample["failures"]) or "passed"),
            "raises (pre-export terminal gate)",
        )
        if self.source_export is not None:
            qrf_columns = sorted(builder._qrf_imputed_source_outputs())
            source_frame = source_columns_frame(
                self.source_export,
                qrf_columns,
                sampler=self.sampler,
                chunk_bytes=self.chunk_bytes,
                chunk_rows=self.chunk_rows,
            )
            source = evaluate(source_frame)
            del source_frame
            record["source"] = source
            self._add(
                "qrf_tail_concentration",
                "QRF tail gate and register (source export)",
                "pass" if source["passed"] else "fail",
                AUTHORITATIVE,
                "settled on the source export's QRF columns at its weights: "
                + ("; ".join(source["failures"]) or "passed"),
                "raises (pre-export terminal gate)",
            )
            if self.reference_qrf_tail is not None:
                comparison = compare_qrf_tail(source, self.reference_qrf_tail)
                record["reference"] = comparison
                self._add(
                    "qrf_tail_concentration",
                    "source-export QRF tail result equals the release's "
                    "qrf_tail_concentration.json",
                    "pass" if not comparison["differences"] else "fail",
                    AUTHORITATIVE,
                    "; ".join(comparison["differences"])
                    or "same verdict, columns and top-k shares",
                    "fidelity of this probe to the release",
                )
        _write_json(self.out_dir / "qrf_tail_concentration.json", record)
        return {
            key: value
            for key, value in record.items()
            if key in ("register", "reference")
        } | {
            name: {"passed": result["passed"], "failures": result["failures"]}
            for name, result in record.items()
            if name in ("subsample", "source")
        }

    # ---- take-up participation ------------------------------------------------

    def _take_up(self) -> dict[str, Any]:
        builder = self.builder
        payload = builder.us_take_up_participation_diagnostics(self.frame)
        builder.write_us_take_up_participation_diagnostics(
            payload, self.out_dir / "us_take_up_participation.json"
        )
        stale = stale_count_calibrated(payload)
        record: dict[str, Any] = {
            "gate": payload["gate"],
            "stale_count_calibrated": stale,
            "seeded_program_count": payload["seeded_program_count"],
            "shares": _take_up_shares(payload),
        }
        # A stored column's absence does not depend on the sample. A column
        # that varies on the subsample varies on the pool, but a constant one
        # may not, and a share is a weighted estimate.
        failures = list(payload["gate"]["failures"])
        missing, constant, band = split_take_up_gate_failures(failures)
        from microcosm.build.us_runtime.take_up_contract import (
            load_take_up_contract,
        )

        programs = load_take_up_contract().programs
        entity_of = {program.variable: program.entity for program in programs}
        stale_missing = [
            variable
            for variable in stale
            if variable not in self.frame.table(entity_of[variable]).columns
        ]
        stale_constant = [
            variable for variable in stale if variable not in stale_missing
        ]
        record["stale_count_calibrated_missing"] = stale_missing
        record["stale_count_calibrated_constant"] = stale_constant
        scale_note = "" if self.census else " (weighted estimates at this sample)"
        self._add(
            "take_up_participation",
            "take-up signal gate: missing seeded columns",
            "fail" if missing else "pass",
            AUTHORITATIVE,
            "; ".join(missing) or "every seeded column is stored",
            "diagnostic",
        )
        self._add(
            "take_up_participation",
            "take-up signal gate: constant columns",
            "fail" if constant else "pass",
            INFORMATIONAL if constant and not self.census else AUTHORITATIVE,
            "; ".join(constant)
            or "every stored seeded column varies here, so it varies on the pool",
            "diagnostic",
        )
        self._add(
            "take_up_participation",
            "take-up signal gate: share bands",
            "fail" if band else "pass",
            AUTHORITATIVE if self.census else INFORMATIONAL,
            ("; ".join(band) or "every share inside its band") + scale_note,
            "diagnostic",
        )
        self._add(
            "take_up_participation",
            "stale count-calibrated take-up columns: missing",
            "fail" if stale_missing else "pass",
            AUTHORITATIVE,
            f"not stored, so they ship at the engine default: {stale_missing}"
            if stale_missing
            else "every count-calibrated column is stored",
            "raises",
        )
        self._add(
            "take_up_participation",
            "stale count-calibrated take-up columns: constant",
            "fail" if stale_constant else "pass",
            INFORMATIONAL if stale_constant and not self.census else AUTHORITATIVE,
            f"constant here, so they ship at the engine default: {stale_constant}"
            if stale_constant
            else "every stored count-calibrated column varies here, so it varies "
            "on the pool",
            "raises",
        )
        if self.source_export is not None:
            source_frame = source_columns_frame(
                self.source_export,
                [program.variable for program in programs],
                sampler=self.sampler,
                chunk_bytes=self.chunk_bytes,
                chunk_rows=self.chunk_rows,
            )
            source_payload = builder.us_take_up_participation_diagnostics(source_frame)
            del source_frame
            builder.write_us_take_up_participation_diagnostics(
                source_payload, self.out_dir / "us_take_up_participation.source.json"
            )
            source_stale = stale_count_calibrated(source_payload)
            record["source"] = {
                "gate": source_payload["gate"],
                "stale_count_calibrated": source_stale,
                "shares": _take_up_shares(source_payload),
            }
            self._add(
                "take_up_participation",
                "take-up signal gate (source export)",
                "pass" if source_payload["gate"]["passed"] else "fail",
                AUTHORITATIVE,
                "settled on the source export: "
                + ("; ".join(source_payload["gate"]["failures"]) or "passed"),
                "diagnostic",
            )
            self._add(
                "take_up_participation",
                "stale count-calibrated take-up columns (source export)",
                "fail" if source_stale else "pass",
                AUTHORITATIVE,
                f"settled on the source export: {source_stale or 'none'}",
                "raises",
            )
        return record

    # ---- the household-batched scorer ----------------------------------------

    def _open_scorer(self) -> dict[str, Any]:
        scorer_cls = _timed_scorer_class(self.builder, self.passes_path, self.clock)
        scorer = scorer_cls(
            self.export_path,
            maximum_microsim_batch_size=self.batch_size,
            load_frame=self.load_frame,
            **self.scorer_options,
        )
        if scorer.dataset_sha256 != self.sha256:
            scorer.close()
            raise ValueError("The scorer loaded different bytes than the probe.")
        if scorer.n_batches < MINIMUM_BATCHES:
            scorer.close()
            raise ValueError(
                f"The scorer built {scorer.n_batches} batch(es); the probe needs "
                f"at least {MINIMUM_BATCHES}."
            )
        self.scorer = scorer
        return {
            "n_households": scorer.n_households,
            "n_batches": scorer.n_batches,
            "max_batch_households": scorer.max_batch_households,
            "maximum_batch_size": scorer.maximum_batch_size,
        }

    def _resolve_entity(self, variable: str) -> str | None:
        if self.measure_entity is not None:
            return self.measure_entity(variable)
        system = getattr(
            getattr(self.scorer, "_microsimulation_cls", None),
            "default_tax_benefit_system_instance",
            None,
        )
        variables = getattr(system, "variables", None)
        if variables is None or variable not in variables:
            return None
        return str(variables[variable].entity.key)

    # ---- 2. reform-coverage smoke ----------------------------------------------

    def _smoke(self) -> dict[str, Any]:
        builder = self.builder
        probes = self.probes

        if self.custom_probes:

            def gate_for(simulate):
                return builder.us_reform_coverage_smoke_gate(
                    simulate=simulate, probes=probes, period=builder.PERIOD
                )

        else:
            # The release's own consumer: the shipped probes, in its order.
            gate_for = builder._reform_coverage_smoke_consumer

        plan = builder._record_post_export_baseline_plan(gate_for)
        scoring = self.scorer.open_consumer("reform_coverage_smoke", plan)
        recorder = SimulateRecorder(scoring.simulate)
        gate = gate_for(recorder)
        scoring_record = self.scorer.finish_consumer(scoring)
        _write_json(
            self.out_dir / "reform_coverage_smoke.json",
            {
                "schema_version": 1,
                "enforced": True,
                "reform_coverage_smoke": {
                    "passed": gate.passed,
                    "failures": list(gate.failures),
                    "details": dict(gate.details),
                },
                "post_export_scoring": scoring_record,
            },
        )
        if len(recorder.reforms) != len(probes):
            raise RuntimeError(
                f"The smoke built {len(recorder.reforms)} reform simulations for "
                f"{len(probes)} probes."
            )
        row_map = HouseholdRowMap(self.scorer._batches(), sampler=self.sampler)
        reference_results = (
            {}
            if self.reference_smoke is None
            else dict(
                self.reference_smoke["reform_coverage_smoke"]["details"]["results"]
            )
        )
        rows = []
        for probe, reform_results in zip(probes, recorder.reforms, strict=True):
            row = self._smoke_row(
                probe,
                gate.details["results"][str(probe.id)],
                recorder.baseline,
                reform_results,
                row_map,
            )
            reference = reference_results.get(str(probe.id))
            if reference is not None:
                row["reference"] = reference_comparison(row, reference, probe=probe)
            rows.append(row)
            self._add(
                "reform_coverage_smoke",
                f"probe {probe.id}",
                "pass" if row["passed"] else "fail",
                row["authority"],
                row["authority_reason"],
                "raises",
            )
        return {
            "passed": gate.passed,
            "failures": list(gate.failures),
            "baseline_keys": len(plan),
            "reform_passes": scoring_record["reform_passes"],
            "reform_systems": scoring_record["reform_systems"],
            "probes": rows,
        }

    def _smoke_row(self, probe, result, baseline, reform_results, row_map):
        builder = self.builder
        key = (probe.budget_measure, int(result["period"]), None)
        effects, entity, decomposition = household_effects(
            baseline[key],
            reform_results[key],
            entity=self._resolve_entity(probe.budget_measure),
            row_map=row_map,
            entities=("person", *builder.US_SCHEMA.group_entities),
        )
        direction = 1.0 if probe.effect_direction == "reform_minus_baseline" else -1.0
        carriers = self.carriers_by_probe[str(probe.id)]
        standard_error = None
        noncarrier = None
        drawn_effect = None
        certainty_effect = None
        weight_scale = None
        if effects is not None and self.design_problems:
            decomposition = "the sample design did not verify against its receipt"
        elif effects is not None:
            weight_scale = effects.weight_scale
            reconstructed = direction * float(effects.weighted.sum())
            if not math.isclose(
                reconstructed,
                float(result["effect"]),
                rel_tol=0.0,
                abs_tol=DECOMPOSITION_RTOL * effects.magnitude + 1e-6,
            ):
                decomposition = (
                    f"per-household effects total {reconstructed!r}; the gate "
                    f"scored {result['effect']!r}"
                )
            else:
                per_unit = self.design.per_unit(effects.weighted)
                variance = self.design.variance(per_unit)
                standard_error = (
                    math.sqrt(variance) if math.isfinite(variance) else None
                )
                bearing = self.design.aligned(effects.weighted) != 0.0
                drawn_effect = int((bearing & ~self.design.certainty).sum())
                certainty_effect = int((bearing & self.design.certainty).sum())
                noncarrier = direction * float(
                    effects.weighted[~effects.weighted.index.isin(carriers)].sum()
                )
        pool = (
            None
            if self.receipt is None
            else {str(r["probe"]): r for r in self.receipt["probes"]}.get(str(probe.id))
        )
        take_all = bool(pool and pool["certainty"])
        magnitude = signed_magnitude(result["effect"], probe.expected_sign)
        authority, reason = classify_probe(
            signed_magnitude=magnitude,
            floor=float(probe.min_abs_effect),
            standard_error=standard_error,
            take_all=take_all,
            drawn_effect_households=drawn_effect,
            census=self.census,
            se_multiplier=self.se_multiplier,
            min_effect_households=self.min_effect_households,
        )
        return {
            "probe": str(probe.id),
            "period": int(result["period"]),
            "budget_measure": probe.budget_measure,
            "measure_entity": entity,
            "baseline_total": float(result["baseline_total"]),
            "reform_total": float(result["reform_total"]),
            "effect": float(result["effect"]),
            "min_abs_effect": float(probe.min_abs_effect),
            "expected_sign": probe.expected_sign,
            "signed_magnitude": magnitude,
            "margin_to_floor": magnitude - float(probe.min_abs_effect),
            "passed": bool(result["passed"]),
            "standard_error": standard_error,
            "decomposition": decomposition,
            "engine_weight_scale": weight_scale,
            "drawn_effect_households": drawn_effect,
            "certainty_effect_households": certainty_effect,
            "noncarrier_effect": noncarrier,
            "pool_carrier_households": None
            if pool is None
            else int(pool["carrier_households"]),
            "expected_sampled_carriers": None
            if pool is None
            else float(pool["expected_sampled_carriers"]),
            "take_all": take_all,
            "sampled_carrier_households": int(len(carriers)),
            "authority": authority,
            "authority_reason": reason,
        }

    # ---- 3. reform validation -----------------------------------------------

    def _validation(self) -> dict[str, Any]:
        builder = self.builder
        if self.calibration_diagnostics is None:
            result = SimpleNamespace(
                diagnostics=[], problem=SimpleNamespace(targets=[])
            )
            in_sample = (
                "simulated on the subsample: no --calibration-diagnostics, so "
                "the release's in-sample rows (taken from its calibration fit, "
                "never simulated) cost one extra reform pass each here"
            )
        else:
            result = calibration_result_from_diagnostics(self.calibration_diagnostics)
            in_sample = (
                "taken from the build's calibration fit (--calibration-"
                "diagnostics), as the release does; full-scale values, not "
                "subsample estimates"
            )
        payload_for = builder._reform_validation_consumer(
            result=result, release_id=self._release_id()
        )
        plan = builder._record_post_export_baseline_plan(payload_for)
        scoring = self.scorer.open_consumer("reform_validation", plan)
        payload = payload_for(scoring.simulate)
        scoring_record = self.scorer.finish_consumer(scoring)
        builder.write_reform_validation(
            payload, self.out_dir / "reform_validation.json"
        )
        rows = payload.get("reforms", [])
        scored = [
            row
            for row in rows
            if (row.get("microcosm") or {}).get("budget_effect") is not None
        ]
        self._add(
            "reform_validation",
            "every reform scores through the batched scorer",
            "pass",
            AUTHORITATIVE,
            "the stage ran to completion; whether a reform raises does not "
            "depend on the sample, but its budget effects are weighted "
            "estimates (informational)",
            "raises",
        )
        record: dict[str, Any] = {
            "baseline_keys": len(plan),
            "baseline_plan_sha256": _plan_digest(plan),
            "reform_passes": scoring_record["reform_passes"],
            "reform_systems": scoring_record["reform_systems"],
            "rows": len(rows),
            "rows_with_budget_effect": len(scored),
            "in_sample_rows": in_sample,
        }
        if self.reference_validation is not None:
            record["reference"] = compare_reform_validation(
                payload, self.reference_validation
            )
        return record

    # ---- 4. demographics ------------------------------------------------------

    def _demographics(self) -> dict[str, Any]:
        builder = self.builder
        plan = builder._record_post_export_baseline_plan(builder._demographics_consumer)
        scoring = self.scorer.open_consumer("demographics", plan)
        ages, weights = builder._demographics_consumer(scoring.simulate)
        self.scorer.finish_consumer(scoring)
        payload = builder.demographics_payload(
            ages, weights, period=builder.PERIOD, release_id=self._release_id()
        )
        payload["geography_coverage"] = builder.geography_coverage_payload(
            self.export_path
        )
        builder.write_demographics(payload, self.out_dir / "demographics.json")
        geography = payload["geography_coverage"]
        record: dict[str, Any] = {
            "total_population": payload["total_population"],
            "states": _geography_summary(geography["states"]),
            "congressional_districts": _geography_summary(
                geography["congressional_districts"]
            ),
        }
        self._add(
            "demographics",
            "age distribution and geography coverage",
            "pass",
            AUTHORITATIVE,
            "the stage scored ages through the batched scorer and read the "
            "geography counts; the age bands are weighted estimates",
            "raises",
        )
        scale = "" if self.census else "; unweighted record counts scale with p"
        for level in ("states", "congressional_districts"):
            summary = record[level]
            if summary is not None:
                self._add(
                    "demographics",
                    f"{level}: geographies under 50 / 100 household records",
                    f"{summary['n_under_50']} / {summary['n_under_100']}",
                    AUTHORITATIVE if self.census else INFORMATIONAL,
                    f"{summary['n_geographies']} geographies{scale}",
                    "diagnostic",
                )
        if self.source_export is not None:
            source = builder.geography_coverage_payload(self.source_export)
            record["source"] = {
                level: _geography_summary(source[level])
                for level in ("states", "congressional_districts")
            }
            for level in ("states", "congressional_districts"):
                summary = record["source"][level]
                if summary is not None:
                    self._add(
                        "demographics",
                        f"{level}: geographies under 50 / 100 household records "
                        "(source export)",
                        f"{summary['n_under_50']} / {summary['n_under_100']}",
                        AUTHORITATIVE,
                        f"settled on the source export: {summary['n_geographies']} "
                        "geographies",
                        "diagnostic",
                    )
            missing = sorted(
                set(source["states"]["counts"]) - set(geography["states"]["counts"])
            )
            record["states_without_sampled_records"] = missing
            self._add(
                "demographics",
                "states with source records but none sampled",
                "fail" if missing else "pass",
                INFORMATIONAL,
                f"{missing or 'none'}: such a state has zero support only in the "
                "subsample",
                "diagnostic",
            )
        return record

    # ---- 5. source coverage ---------------------------------------------------

    def _source_coverage(self) -> dict[str, Any]:
        builder = self.builder
        reproduced = "complete"
        if self.target_surface == builder.TARGET_SURFACE_FULL:
            active, surface_exclusions = builder._source_coverage_aliases(None)
        elif self.dropped_cd_targets is not None:
            active, surface_exclusions = builder._source_coverage_aliases(
                {
                    "mode": self.target_surface,
                    "dropped_congressional_district_targets": int(
                        self.dropped_cd_targets
                    ),
                }
            )
        else:
            # The alias sets do not depend on the count; only the reason text
            # names it, and only the build knows it.
            active = builder.DIRECT_ACTIVE_ALIASES
            surface_exclusions = {
                alias: (
                    "Not calibrated by this release: --target-surface "
                    f"{self.target_surface} dropped all congressional-district-"
                    "classified targets (the count is not known to the probe)."
                )
                for alias in builder.CONGRESSIONAL_DISTRICT_SOURCE_ALIASES
            }
            reproduced = "gate reproduced; the dropped-target count is not"
        coverage = builder.us_source_coverage_diagnostics(
            active_target_aliases=active,
            reviewed_exclusions={
                **builder._reviewed_exclusions(active),
                **surface_exclusions,
            },
        )
        coverage["fiscal_target_support_exclusions"] = [
            {"source_record_id": source_record_id, "reason": reason}
            for source_record_id, reason in sorted(
                builder.US_FISCAL_TARGET_SUPPORT_EXCLUSIONS.items()
            )
        ]
        coverage["probe_not_reproduced"] = [
            "fiscal_target_sources (needs the compiled target specs)",
            "congressional_district_vintage_crosswalk (build metadata)",
            f"{builder.US_FISCAL_TARGET_EXCLUSION_RECEIPT_KEY} (build receipt)",
        ]
        builder.write_us_source_coverage_diagnostics(
            coverage, self.out_dir / "us_source_coverage.json"
        )
        gate = coverage["gate"]
        self._add(
            "source_coverage",
            "source coverage gate",
            "pass" if gate["passed"] else "fail",
            AUTHORITATIVE,
            "does not read the dataset: " + ("; ".join(gate["failures"]) or "passed"),
            "recorded (surfaces at publish)",
        )
        return {
            "target_surface": self.target_surface,
            "reproduced": reproduced,
            "gate": gate,
        }


def probe_carrier_households(frame, binding_inputs, *, sampler) -> np.ndarray:
    """Households of ``frame`` with a nonzero value of any binding input.

    The sampler's support semantics (booleans count as 1, missing as 0); a
    leaf is read from the first entity table storing it.
    """
    union: list[np.ndarray] = []
    person = frame.table("person")
    for leaf in binding_inputs:
        for entity in frame.entities:
            table = frame.table(entity)
            if leaf not in table.columns:
                continue
            ids = None if entity == "person" else table[f"{entity}_id"].to_numpy()
            households = np.asarray(sampler.household_of_rows(entity, person, ids))
            nonzero = sampler.numeric_support_values(table[leaf]) != 0.0
            union.append(np.unique(households[nonzero]))
            break
    return np.unique(np.concatenate(union)) if union else np.asarray([], np.int64)


def probe_export(export_path, out_dir, **options) -> dict[str, Any]:
    """Run the probe on ``export_path``; see :class:`ExportProbe`."""
    return ExportProbe(export_path, out_dir, **options).run()


def _take_up_shares(payload: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(row["variable"]): float(row["take_up_share"])
        for row in payload["programs"]
        if row.get("take_up_share") is not None
    }


def _geography_summary(summary: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {key: value for key, value in summary.items() if key != "counts"}


def _summary(report: Mapping[str, Any], receipt) -> dict[str, Any]:
    failing = [row for row in report["verdicts"] if row["verdict"] in ("fail", "error")]
    summary: dict[str, Any] = {
        "authoritative_failures": [
            row for row in failing if row["authority"] == AUTHORITATIVE
        ],
        "informational_failures": [
            row for row in failing if row["authority"] == INFORMATIONAL
        ],
        "stage_status": {
            name: record.get("status") for name, record in report["stages"].items()
        },
    }
    if receipt is not None:
        ratio = receipt["source"]["rows"]["household"] / max(
            1, receipt["selection"]["households"]
        )
        summary["full_scale_extrapolation"] = {
            "method": (
                "stage wall time x source households / sampled households: a "
                "rough linear extrapolation of the engine stages, not a "
                "measurement"
            ),
            "household_ratio": ratio,
            "wall_seconds": {
                name: round(record["timing"]["wall_seconds"] * ratio, 1)
                for name, record in report["stages"].items()
                if name
                in (
                    "open_scorer",
                    "reform_coverage_smoke",
                    "reform_validation",
                    "demographics",
                )
                and "timing" in record
            },
        }
    return summary


#: Files of a release directory (``releases/<release id>/``) the probe reads.
REFERENCE_FILES = {
    "smoke": "reform_coverage_smoke.json",
    "validation": "reform_validation.json",
    "build_manifest": "build_manifest.json",
    "release_manifest": "release_manifest.json",
    "calibration_diagnostics": "calibration_diagnostics.json",
    "qrf_tail": "qrf_tail_concentration.json",
}


def resolve_target_surface(
    flag: str | None,
    dropped: int | None,
    release_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The build's target surface: the flag, else the release manifest's
    ``build.target_surface_selection``, else ``full``."""
    selection = (
        None
        if release_manifest is None
        else (release_manifest.get("build") or {}).get("target_surface_selection")
    )
    if flag is not None:
        return {
            "mode": flag,
            "dropped_congressional_district_targets": dropped,
            "source": "--target-surface",
        }
    if selection:
        return {
            "mode": str(selection["mode"]),
            "dropped_congressional_district_targets": (
                dropped
                if dropped is not None
                else selection.get("dropped_congressional_district_targets")
            ),
            "source": "reference release_manifest.json build.target_surface_selection",
        }
    return {
        "mode": "full",
        "dropped_congressional_district_targets": dropped,
        "source": "default (no flag and no reference release manifest)",
    }


def reference_paths(
    release_dir: Path | None, **explicit: Path | None
) -> dict[str, Path | None]:
    """Each reference file: the explicit path, else the release directory's."""
    paths: dict[str, Path | None] = {}
    for key, filename in REFERENCE_FILES.items():
        path = explicit.get(key)
        if path is None and release_dir is not None:
            candidate = Path(release_dir) / filename
            path = candidate if candidate.exists() else None
        paths[key] = path
    return paths


def _read_json(path: Path | None):
    return None if path is None else json.loads(Path(path).read_text())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--export", type=Path, required=True, help="H5 to score")
    parser.add_argument("--out", type=Path, required=True, help="report directory")
    parser.add_argument(
        "--sample-receipt",
        type=Path,
        default=None,
        help="the sampler's receipt (default: sample_receipt.json beside "
        "--export; without one the export is scored as a full population)",
    )
    parser.add_argument(
        "--source-export",
        type=Path,
        default=None,
        help="the sampled export, for settling verdicts (default: the receipt's)",
    )
    parser.add_argument(
        "--no-source-settlement",
        action="store_true",
        help="never read the source export",
    )
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--target-surface",
        default=None,
        help="the build's --target-surface (default: the reference release "
        "manifest's, else full)",
    )
    parser.add_argument(
        "--dropped-congressional-district-targets", type=int, default=None
    )
    parser.add_argument(
        "--reference-release-dir",
        type=Path,
        default=None,
        help="a full-size build's release directory (releases/<release id>/): "
        "defaults --reference-smoke, --reference-validation, "
        "--reference-build-manifest and --calibration-diagnostics to its files",
    )
    parser.add_argument(
        "--reference-smoke",
        type=Path,
        default=None,
        help="a full-size run's reform_coverage_smoke.json to compare against",
    )
    parser.add_argument(
        "--reference-validation",
        type=Path,
        default=None,
        help="a full-size run's reform_validation.json to compare against",
    )
    parser.add_argument(
        "--reference-build-manifest",
        type=Path,
        default=None,
        help="a full-size run's build_manifest.json, whose post_export_scoring "
        "plans the probe's must match",
    )
    parser.add_argument(
        "--calibration-diagnostics",
        type=Path,
        default=None,
        help="the build's calibration_diagnostics.json: reform validation then "
        "takes its in-sample rows from the fit, as the release does",
    )
    parser.add_argument("--se-multiplier", type=float, default=DEFAULT_SE_MULTIPLIER)
    parser.add_argument(
        "--min-effect-households", type=int, default=DEFAULT_MIN_EFFECT_HOUSEHOLDS
    )
    parser.add_argument(
        "--qrf-tail-register",
        type=Path,
        default=None,
        help="the build's --qrf-tail-concentration-exclusions JSON (default: "
        "the reference qrf_tail_concentration.json's reviewed exclusions)",
    )
    parser.add_argument("--release-id", default=None)
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=None,
        help="rows per chunked source read (default: sized from 128 MiB)",
    )
    args = parser.parse_args(argv)

    receipt_path = args.sample_receipt
    if receipt_path is None:
        candidate = args.export.resolve().parent / "sample_receipt.json"
        receipt_path = candidate if candidate.exists() else None
    receipt = None if receipt_path is None else json.loads(receipt_path.read_text())
    references = reference_paths(
        args.reference_release_dir,
        smoke=args.reference_smoke,
        validation=args.reference_validation,
        build_manifest=args.reference_build_manifest,
        calibration_diagnostics=args.calibration_diagnostics,
    )
    surface = resolve_target_surface(
        args.target_surface,
        args.dropped_congressional_district_targets,
        _read_json(references["release_manifest"]),
    )
    register = None
    if args.qrf_tail_register is not None:
        register = load_builder()._load_qrf_tail_concentration_exclusions(
            args.qrf_tail_register
        )
    report = probe_export(
        args.export,
        args.out,
        sample_receipt=receipt,
        receipt_path=receipt_path,
        source_export=args.source_export,
        source_settlement=not args.no_source_settlement,
        stages=tuple(stage.strip() for stage in args.stages.split(",") if stage),
        batch_size=args.batch_size,
        target_surface=surface["mode"],
        target_surface_source=surface["source"],
        dropped_congressional_district_targets=surface[
            "dropped_congressional_district_targets"
        ],
        reference_smoke=_read_json(references["smoke"]),
        reference_validation=_read_json(references["validation"]),
        reference_build_manifest=_read_json(references["build_manifest"]),
        calibration_diagnostics=_read_json(references["calibration_diagnostics"]),
        reference_qrf_tail=_read_json(references["qrf_tail"]),
        qrf_tail_register=register,
        chunk_rows=args.chunk_rows,
        reference_paths={
            key: None if path is None else str(path) for key, path in references.items()
        },
        se_multiplier=args.se_multiplier,
        min_effect_households=args.min_effect_households,
        release_id=args.release_id,
    )
    summary = report["summary"]
    print(
        f"[probe] {len(summary['authoritative_failures'])} authoritative and "
        f"{len(summary['informational_failures'])} informational failure(s); "
        f"report {Path(args.out) / REPORT_FILENAME}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
