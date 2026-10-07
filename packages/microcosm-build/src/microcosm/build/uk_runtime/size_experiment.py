"""Selection-only UK dataset-size experiments on a finished graph build (#1124).

A ``--dataset-households`` graph build writes its numerical evidence into its
out directory as flat artifacts (the ordered problem, the dense result, the
size search, the draw, the refit; ``evidence-index.json`` records their
digests) and keeps its pool frame in the run's content store. This module
reads them without writing to either:

* :func:`build_uk_size_experiment_cache` loads the pool frame once, writes a
  skeleton of it (household, person and benefit-unit ids, memberships and
  the typed weights; no survey variables) to a harness-owned store, and the
  pool profile the scorecard reads.
* :func:`load_uk_size_experiment_baseline` decodes the problem, the dense
  solve, the search, the draw and the stored refit with the graph's own
  decoders onto the skeleton, and checks that the run's weighting rule,
  recomputed from the problem's rows, reproduces the bound loss weights bit
  for bit.
* :func:`run_uk_size_control` re-runs the stored refit from the stored search
  and draw; it must reproduce the stored weights before any variant is read.
* :func:`run_uk_size_census` is step 0's census of what the stored support
  and its stretch caps allow before any variant runs: mass capacity by nation
  and household type at several refit baseline floors, per-area ceilings
  against the relative-collapse rule, the search's capacity bound k_min, the
  L2 penalty scales that place the λ grid, and each rule's loss shares.
* :func:`run_uk_size_experiment` runs one :class:`UKSizeExperiment`: a refit
  on the stored support under another rule, L2 penalty or baseline floor; a
  new search from the stored dense solve (warm-started) then a draw and a
  refit; or a refit-level rotated holdout on the stored support.

Each experiment returns its receipt and its weights; the tool
``tools/run_uk_size_experiment.py`` writes them outside the repository and the
run directory. No release gate runs here: the scorecard's criteria are
measurements, and a configuration the solver chain refuses is recorded as a
failed result (:func:`uk_size_failed_receipt`), not a stop. The dense
reference is never re-solved: every delta is the selection's.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import resource
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.holdout import rotated_folds
from microcosm.build.uk_runtime.dataset_size import (
    UKDatasetSize,
    UKSizeDraw,
    UKSizeSelection,
    draw_uk_dataset_size,
    refit_uk_dataset_size,
    select_uk_dataset_size,
    uk_size_l2,
    uk_size_l2_from_options,
)
from microcosm.build.uk_runtime.local_doctrine import UK_LOCAL_TARGET_WEIGHT_RULES
from microcosm.build.uk_runtime.local_rowwise import (
    UK_LOCAL_HOLDOUT_FOLDS,
    UK_LOCAL_HOLDOUT_SEED,
)
from microcosm.build.uk_runtime.rowwise_cli import git_commit, git_dirty
from microcosm.build.uk_runtime.size_experiment_scorecard import (
    UK_SIZE_ACCEPTANCE,
    UK_SIZE_CAP_TOLERANCE,
    UKPoolProfile,
    UKSizeWeights,
    build_uk_pool_profile,
    load_uk_pool_profile,
    uk_size_scorecard,
    write_uk_pool_profile,
)
from microcosm.build.uk_runtime.target_weights import (
    UKStageTargetWeighting,
    UKTargetRows,
    uk_rule_loss_weights,
    uk_target_loss_weight_receipt,
    uk_target_rows_from_problem,
)
from microcosm.calibrate import (
    CalibrationResult,
    chi_square_distance,
    default_target_loss_scales,
    relative_error_loss,
)
from microcosm.calibrate.artifacts import (
    OrderedProblem,
    decode_calibration_result,
    decode_problem,
    decode_solution,
)
from microcosm.frame import Frame
from microcosm.graph.canonical import canonical_json
from microcosm.graph.store import ContentStore

__all__ = [
    "UK_SIZE_CENSUS_FLOORS",
    "UK_SIZE_CENSUS_L2_GRID",
    "UK_SIZE_CENSUS_PRESSURE_BAND",
    "UK_SIZE_EXPERIMENT_ARTIFACTS",
    "UK_SIZE_EXPERIMENT_MODES",
    "UK_SIZE_EXPERIMENT_RESERVED_NAMES",
    "UKSizeExperiment",
    "UKSizeExperimentBaseline",
    "build_uk_size_experiment_cache",
    "load_uk_size_experiment_baseline",
    "parse_uk_size_experiments",
    "run_uk_size_census",
    "run_uk_size_control",
    "run_uk_size_experiment",
    "UKSizeScoringInputs",
    "load_uk_size_scoring_inputs",
    "load_uk_size_weights",
    "save_uk_size_weights",
    "score_uk_size_experiments",
    "uk_size_failed_receipt",
    "uk_size_weights_of",
]

#: The flat evidence artifacts a size experiment reads, by evidence-index key.
UK_SIZE_EXPERIMENT_ARTIFACTS = {
    "problem": "uk.full.problem/problem",
    "dense": "uk.full.dense/result",
    "search": "uk.full.size_search/result",
    "selection": "uk.full.size_search/selection",
    "draw": "uk.full.size_draw/draw",
    "refit_solution": "uk.full.size_refit/solution",
    "size": "uk.full.size_refit/size",
}
UK_SIZE_EXPERIMENT_MODES = ("refit", "selection", "refit_holdout")
#: Output directory names the tool itself writes; no experiment may take one.
UK_SIZE_EXPERIMENT_RESERVED_NAMES = frozenset({"control", "census"})
#: The refit baseline floors step 0's census measures (the plan's 0, 0.1, 0.5, 1).
UK_SIZE_CENSUS_FLOORS = (0.0, 0.1, 0.5, 1.0)
#: The pre-registered refit L2 grid whose penalty pressure the census places.
UK_SIZE_CENSUS_L2_GRID = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
#: The band the grid's pressure λ·P/L should span (the plan's grid placement).
UK_SIZE_CENSUS_PRESSURE_BAND = (0.05, 5.0)
_POOL_NODE = "uk.full.pool"
_CACHE_MANIFEST = "size_experiment_cache.json"
_SKELETON_KEY_PREFIX = b"microcosm-uk-size-experiment-skeleton/1\n"
_CONTROL_TOLERANCE = 1e-6


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _evidence(run_dir: Path) -> dict[str, Mapping[str, Any]]:
    path = Path(run_dir) / "evidence-index.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: not a finished graph build.")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_artifact(run_dir: Path, evidence: Mapping[str, Any], name: str) -> bytes:
    key = UK_SIZE_EXPERIMENT_ARTIFACTS[name]
    if key not in evidence:
        raise ValueError(f"the run's evidence index has no {key!r} artifact.")
    entry = evidence[key]
    payload = (Path(run_dir) / str(entry["filename"])).read_bytes()
    if (
        len(payload) != int(entry["size_bytes"])
        or _sha256_bytes(payload) != entry["sha256"]
    ):
        raise ValueError(f"{entry['filename']} differs from its evidence-index digest.")
    return payload


def _pool_frame_key(run_dir: Path) -> str:
    manifest = json.loads((Path(run_dir) / "numerical.graph.json").read_text("utf-8"))
    nodes = manifest.get("content_addressed", manifest).get("nodes", {})
    record = nodes.get(_POOL_NODE)
    if not isinstance(record, Mapping) or not record.get("frame_key"):
        raise ValueError(f"the run's graph manifest records no {_POOL_NODE} frame.")
    return str(record["frame_key"])


def _skeleton(pool: Frame) -> Frame:
    """The pool with only its id and membership columns, weights, strata and log."""

    schema = pool.schema
    if schema.links:
        raise ValueError("size experiments support frames without declared links.")
    tables = {}
    for entity in schema.entities:
        keep = [schema.entity_id_column(entity)]
        if entity == schema.person_entity:
            keep += [schema.membership_column(group) for group in schema.group_entities]
        tables[entity] = pool.table(entity)[keep]
    return Frame(
        tables,
        schema,
        {"household": pool.weights_for("household")},
        pool.strata,
        mass_log=pool.mass_log,
        metadata=pool.metadata,
    )


def _check_pool_axis(frame: Frame, problem: OrderedProblem) -> None:
    ids = tuple(frame.table("household")["household_id"])
    if ids != problem.entity_ids:
        raise ValueError("the pool frame's household axis differs from the problem's.")
    weights = frame.weights_for("household")
    initial = problem.problem.initial_weights
    if weights.kind != initial.kind or not np.array_equal(
        weights.values, initial.values
    ):
        raise ValueError("the pool frame's design weights differ from the problem's.")


def build_uk_size_experiment_cache(run_dir: Path, cache_dir: Path) -> dict[str, Any]:
    """Load a run's pool once; cache its skeleton and profile under ``cache_dir``."""

    run_dir, cache_dir = Path(run_dir).resolve(), Path(cache_dir).resolve()
    if cache_dir.is_relative_to(run_dir):
        raise ValueError("the experiment cache must live outside the run directory.")
    evidence = _evidence(run_dir)
    problem_bytes = _read_artifact(run_dir, evidence, "problem")
    problem = decode_problem(problem_bytes)
    frame_key = _pool_frame_key(run_dir)
    object_path = run_dir / ".graph-store" / "objects" / frame_key[:2] / frame_key
    pool = ContentStore.load_frame_path(object_path)
    _check_pool_axis(pool, problem)
    profile = build_uk_pool_profile(pool, problem_sha256=problem.sha256)
    skeleton = _skeleton(pool)
    del pool
    cache_dir.mkdir(parents=True, exist_ok=True)
    store = ContentStore(cache_dir / "store")
    skeleton_key = hashlib.sha256(_SKELETON_KEY_PREFIX + frame_key.encode()).hexdigest()
    if not store.has(skeleton_key):
        store.put_frame(skeleton_key, skeleton)
    profile_manifest = write_uk_pool_profile(profile, cache_dir / "profile")
    manifest = {
        "run_dir": str(run_dir),
        "problem_sha256": problem.sha256,
        "pool_frame_key": frame_key,
        "skeleton_key": skeleton_key,
        "households": len(problem.entity_ids),
        "profile": profile_manifest,
        "code": {"git_commit": git_commit(), "git_dirty": git_dirty()},
    }
    (cache_dir / _CACHE_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


@dataclass(frozen=True)
class UKSizeExperimentBaseline:
    """A finished size build, decoded onto its cached pool skeleton."""

    run_dir: Path
    problem: OrderedProblem
    rows: UKTargetRows
    run_rule: str
    frame: Frame
    dense: CalibrationResult
    selection: UKSizeSelection
    draw: UKSizeDraw
    stored_refit_weights: np.ndarray
    stored_refit_ids: tuple[int | str, ...]
    size_receipt: Mapping[str, Any]
    settings: Mapping[str, Any]
    profile: UKPoolProfile
    digests: Mapping[str, str]


def _run_rule(rows: UKTargetRows, bound: np.ndarray) -> str:
    for rule in UK_LOCAL_TARGET_WEIGHT_RULES:
        try:
            weights = uk_rule_loss_weights(rows, rule=rule)
        except ValueError:
            continue
        if np.array_equal(weights, bound):
            return rule
    raise ValueError(
        "no rule in the local vocabulary reproduces the run's bound loss weights "
        "from its problem rows: the row carrier disagrees with the build."
    )


def load_uk_size_experiment_baseline(
    run_dir: Path, cache_dir: Path
) -> UKSizeExperimentBaseline:
    """Decode a run's size evidence onto the cached skeleton, with its checks."""

    run_dir, cache_dir = Path(run_dir).resolve(), Path(cache_dir).resolve()
    manifest = json.loads((cache_dir / _CACHE_MANIFEST).read_text(encoding="utf-8"))
    if Path(manifest["run_dir"]) != run_dir:
        raise ValueError(f"the cache at {cache_dir} belongs to {manifest['run_dir']}.")
    evidence = _evidence(run_dir)
    payloads = {
        name: _read_artifact(run_dir, evidence, name)
        for name in UK_SIZE_EXPERIMENT_ARTIFACTS
    }
    problem = decode_problem(payloads["problem"])
    if problem.sha256 != manifest["problem_sha256"]:
        raise ValueError("the cache was built for another problem artifact.")
    frame = ContentStore(cache_dir / "store").load_frame(manifest["skeleton_key"])
    _check_pool_axis(frame, problem)
    rows = uk_target_rows_from_problem(problem)
    bound = np.asarray(problem.bindings["target_loss_weights"], dtype=np.float64)
    run_rule = _run_rule(rows, bound)
    dense = decode_calibration_result(payloads["dense"], frame=frame, problem=problem)
    search = decode_calibration_result(payloads["search"], frame=frame, problem=problem)
    meta = json.loads(payloads["selection"])
    if meta.get("problem_sha256") != problem.sha256 or meta.get("method") != (
        "contribution_informed_l0"
    ):
        raise ValueError(
            "the stored size search is not an informed L0 search of this problem."
        )
    protected = np.asarray(meta["protected"], dtype=bool)
    selection = UKSizeSelection(
        search,
        protected,
        int(meta["households"]),
        int(meta["epochs"]),
        float(meta["learning_rate"]),
        int(meta["seed"]),
        float(meta["pi_hi"]),
    )
    draw = UKSizeDraw.from_payload(payloads["draw"], problem_sha256=problem.sha256)
    if draw is None:
        raise ValueError("the stored run is a full-pool size: no selection to vary.")
    solution = decode_solution(
        payloads["refit_solution"], problem_sha256=problem.sha256
    )
    size_receipt = json.loads(payloads["size"])
    settings = {
        "households": int(meta["households"]),
        "epochs": int(meta["epochs"]),
        "learning_rate": float(meta["learning_rate"]),
        "seed": int(meta["seed"]),
        "pi_hi": float(draw.pi_hi),
        "baseline_pi_floor": float(size_receipt.get("baseline_pi_floor", 0.0)),
        "selected_l0_lambda": float(search.l0_lambda),
        "initial_lambda": meta.get("initial_lambda"),
        "max_weight_ratio": dense.options["max_weight_ratio"],
        "target_loss_cap": float(dense.target_loss_cap),
        # The stored stages' own L2 penalties: a refit on the stored search must
        # name the search's (the reuse check binds it), and the control
        # re-runs the stored refit under the refit's.
        "selection_l2": uk_size_l2_from_options(search.options),
        "refit_l2": _stored_refit_l2(size_receipt),
    }
    profile = load_uk_pool_profile(
        cache_dir / "profile",
        problem_sha256=problem.sha256,
        household_ids=problem.entity_ids,
    )
    digests = {
        UK_SIZE_EXPERIMENT_ARTIFACTS[name]: str(
            evidence[UK_SIZE_EXPERIMENT_ARTIFACTS[name]]["sha256"]
        )
        for name in UK_SIZE_EXPERIMENT_ARTIFACTS
    }
    return UKSizeExperimentBaseline(
        run_dir=run_dir,
        problem=problem,
        rows=rows,
        run_rule=run_rule,
        frame=frame,
        dense=dense,
        selection=selection,
        draw=draw,
        stored_refit_weights=np.asarray(solution.weights, dtype=np.float64),
        stored_refit_ids=tuple(solution.entity_ids),
        size_receipt=size_receipt,
        settings=settings,
        profile=profile,
        digests=digests,
    )


@dataclass(frozen=True)
class UKSizeExperiment:
    """One declared size experiment (read from JSON with strict keys).

    ``refit_rule`` / ``selection_rule`` name a local weighting rule for the
    stage (``None``: the run's own weights). ``refit_l2`` / ``selection_l2``
    are ``{"lambda", "anchor", "basis"}`` mappings (``None``: off).
    ``baseline_pi_floor`` and ``learning_rate`` override the run's (the
    learning rate is a refit-only perturbation, recorded as such).
    ``epochs`` is a refit-only epoch count for smoke runs of the setup (a
    shortened refit is a plumbing check, not a result; no pre-registered
    experiment sets it). ``initial_lambda`` (selection mode) is a warm-start
    penalty, ``"scaled"`` for the stored search's penalty times the rule's
    loss ratio at S0, or ``None`` for a cold search.
    """

    name: str
    mode: str = "refit"
    refit_rule: str | None = None
    selection_rule: str | None = None
    refit_l2: Mapping[str, Any] | None = None
    selection_l2: Mapping[str, Any] | None = None
    baseline_pi_floor: float | None = None
    learning_rate: float | None = None
    epochs: int | None = None
    initial_lambda: float | str | None = "scaled"
    folds: int = UK_LOCAL_HOLDOUT_FOLDS
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or self.name.startswith("."):
            raise ValueError(
                f"experiment name {self.name!r} is not a plain directory name."
            )
        if self.name in UK_SIZE_EXPERIMENT_RESERVED_NAMES:
            raise ValueError(
                f"experiment name {self.name!r} is reserved for the tool's own output."
            )
        if self.mode not in UK_SIZE_EXPERIMENT_MODES:
            raise ValueError(
                f"experiment {self.name}: mode must be one of {UK_SIZE_EXPERIMENT_MODES}."
            )
        for rule in (self.refit_rule, self.selection_rule):
            if rule is not None and rule not in UK_LOCAL_TARGET_WEIGHT_RULES:
                raise ValueError(f"experiment {self.name}: unknown rule {rule!r}.")
        if self.mode != "selection" and (self.selection_rule or self.selection_l2):
            raise ValueError(
                f"experiment {self.name}: selection settings need mode 'selection'."
            )
        if self.epochs is not None and (
            self.mode == "selection"
            or isinstance(self.epochs, bool)
            or not isinstance(self.epochs, int)
            or self.epochs <= 0
        ):
            raise ValueError(
                f"experiment {self.name}: epochs is a positive refit-only override "
                "(modes 'refit' and 'refit_holdout')."
            )
        for stage, block in (
            ("refit", self.refit_l2),
            ("selection", self.selection_l2),
        ):
            if block is not None:
                unknown = set(block) - {"lambda", "anchor", "basis"}
                if unknown:
                    raise ValueError(
                        f"experiment {self.name}: {stage}_l2 keys {sorted(unknown)}."
                    )
                uk_size_l2(
                    stage,
                    l2_lambda=block.get("lambda", 0.0),
                    anchor=block.get("anchor"),
                    basis=block.get("basis"),
                )
        if isinstance(self.initial_lambda, str) and self.initial_lambda != "scaled":
            raise ValueError(
                f"experiment {self.name}: initial_lambda must be a number, 'scaled' or null."
            )


def parse_uk_size_experiments(
    payload: Sequence[Mapping[str, Any]],
) -> tuple[UKSizeExperiment, ...]:
    """Experiments from a JSON list; unknown keys and duplicate names refuse."""

    allowed = {item.name for item in UKSizeExperiment.__dataclass_fields__.values()}
    experiments = []
    for entry in payload:
        unknown = set(entry) - allowed
        if unknown:
            raise ValueError(
                f"experiment {entry.get('name')!r} has unknown keys {sorted(unknown)}."
            )
        experiments.append(UKSizeExperiment(**entry))
    names = [experiment.name for experiment in experiments]
    if len(set(names)) != len(names):
        raise ValueError("experiment names must be unique.")
    return tuple(experiments)


def _l2_kwargs(stage: str, block: Mapping[str, Any] | None) -> dict[str, Any]:
    if block is None:
        return {}
    return {
        f"{stage}_l2_lambda": block.get("lambda", 0.0),
        f"{stage}_l2_anchor": block.get("anchor"),
        f"{stage}_l2_basis": block.get("basis"),
    }


def _weighting(
    baseline: UKSizeExperimentBaseline, rule: str | None, *, held_out=()
) -> UKStageTargetWeighting | None:
    if rule is None and not held_out:
        return None
    return UKStageTargetWeighting(
        baseline.run_rule if rule is None else rule,
        baseline.rows,
        held_out=tuple(held_out),
    )


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _loss(
    baseline: UKSizeExperimentBaseline, estimates: np.ndarray, weights: np.ndarray
) -> float:
    targets = np.asarray(baseline.problem.problem.target_vector, dtype=np.float64)
    return float(
        relative_error_loss(
            estimates,
            targets,
            target_loss_weights=weights,
            target_loss_scales=default_target_loss_scales(targets),
            target_loss_cap=float(baseline.settings["target_loss_cap"]),
        )
    )


def _estimates(
    baseline: UKSizeExperimentBaseline, support: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    return np.asarray(
        baseline.problem.problem.matrix[:, support] @ weights, dtype=np.float64
    )


def _stored_refit_l2(size_receipt: Mapping[str, Any]) -> dict[str, Any] | None:
    block = size_receipt.get("refit_l2")
    if not block:
        return None
    return {key: block[key] for key in ("lambda", "anchor", "basis")}


def _stored_selection_l2_kwargs(baseline: UKSizeExperimentBaseline) -> dict[str, Any]:
    return _l2_kwargs("selection", baseline.settings["selection_l2"])


def _refit_settings(
    baseline: UKSizeExperimentBaseline, experiment: UKSizeExperiment | None
) -> dict[str, Any]:
    settings = baseline.settings
    floor = settings["baseline_pi_floor"]
    if experiment is not None and experiment.baseline_pi_floor is not None:
        floor = float(experiment.baseline_pi_floor)
    return {
        "households": settings["households"],
        "epochs": settings["epochs"],
        "learning_rate": settings["learning_rate"],
        "seed": settings["seed"],
        "pi_hi": settings["pi_hi"],
        "baseline_pi_floor": floor,
    }


def _support_positions(
    baseline: UKSizeExperimentBaseline, sized: UKDatasetSize
) -> np.ndarray:
    return np.asarray(sized.support, dtype=np.int64)


def run_uk_size_control(
    baseline: UKSizeExperimentBaseline, *, accept_drift_reason: str | None = None
) -> dict[str, Any]:
    """Re-run the stored refit; it must reproduce the stored weights.

    Ids must be equal and the weights equal bit for bit; a relative
    difference within 1e-6 passes only with ``accept_drift_reason``
    (recorded). Every key of the stored size receipt must come back equal.
    """

    started = time.perf_counter()
    sized = refit_uk_dataset_size(
        baseline.frame,
        baseline.dense,
        selection=baseline.selection,
        draw=baseline.draw,
        **_refit_settings(baseline, None),
        **_stored_selection_l2_kwargs(baseline),
        **_l2_kwargs("refit", baseline.settings["refit_l2"]),
    )
    ids = tuple(sized.result.frame.table("household")["household_id"])
    weights = np.asarray(sized.result.weights, dtype=np.float64)
    ids_equal = ids == baseline.stored_refit_ids
    exact = ids_equal and np.array_equal(weights, baseline.stored_refit_weights)
    relative = (
        float(
            np.max(
                np.abs(weights - baseline.stored_refit_weights)
                / baseline.stored_refit_weights
            )
        )
        if ids_equal and weights.shape == baseline.stored_refit_weights.shape
        else None
    )
    stored = dict(baseline.size_receipt)
    stored.pop("problem_sha256", None)
    stored.pop("household_ids", None)
    receipt_differences = sorted(
        key
        for key, value in stored.items()
        if json.loads(canonical_json(sized.receipt.get(key))) != value
    )
    drift_ok = (
        not exact
        and relative is not None
        and relative <= _CONTROL_TOLERANCE
        and bool(accept_drift_reason)
    )
    passed = (exact or drift_ok) and not receipt_differences
    return {
        "passed": bool(passed),
        "exact": bool(exact),
        "ids_equal": bool(ids_equal),
        "max_relative_weight_difference": relative,
        "tolerance": _CONTROL_TOLERANCE,
        "accepted_drift_reason": accept_drift_reason if drift_ok else None,
        "receipt_differences": receipt_differences,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
        "sized": sized,
    }


def _kish_by_area(
    area: np.ndarray, positions: np.ndarray, weights: np.ndarray, n_areas: int
) -> np.ndarray:
    sums = np.bincount(area[positions], weights=weights, minlength=n_areas)
    squares = np.bincount(area[positions], weights=weights**2, minlength=n_areas)
    return np.divide(sums**2, squares, out=np.zeros_like(sums), where=squares > 0)


def _refit_start(
    design: np.ndarray, support: np.ndarray, q: np.ndarray, floor: float
) -> np.ndarray:
    """The refit's start at ``floor``: design over max(q, floor), to the pool mass.

    The solver's normalized Horvitz-Thompson baseline
    (``refit_l0_selection`` on an exact-k support), recomputed here.
    """

    floored = q if floor == 0.0 else np.maximum(q, floor)
    expanded = design[support] / floored
    return expanded * (design.sum() / expanded.sum())


def _k_min(design: np.ndarray, mass: float, ratio: float) -> int | None:
    """Fewest rows whose capped design weights (heaviest first) carry ``mass``."""

    if mass <= 0.0:
        return 0
    reach = np.cumsum(ratio * np.sort(design)[::-1])
    index = int(np.searchsorted(reach, mass, side="left"))
    return index + 1 if index < len(reach) else None


def _mass_to(
    labels: np.ndarray, values: np.ndarray, reference: Mapping[str, float]
) -> dict[str, float | None]:
    return {
        label: float(values[labels == label].sum() / mass) if mass > 0 else None
        for label, mass in reference.items()
    }


def run_uk_size_census(
    baseline: UKSizeExperimentBaseline,
    *,
    floors: Sequence[float] = UK_SIZE_CENSUS_FLOORS,
    l2_grid: Sequence[float] = UK_SIZE_CENSUS_L2_GRID,
    pressure_band: tuple[float, float] = UK_SIZE_CENSUS_PRESSURE_BAND,
) -> dict[str, Any]:
    """Step 0's census of the stored support and its caps (no solve).

    For each refit baseline floor in ``floors`` and the run's own: the start's
    and the stretch cap's mass against D's, in total, by nation and by
    household type (a capacity below one means the refit cannot reach D's mass
    there whatever it fits); the areas whose start already falls below the
    relative-collapse rule; and the ``initial`` anchor's chi-square penalty
    scale P, measured at S0. Beside them:

    * per grain, the areas whose kept rows cannot reach the rule even at equal
      weights (the support's ceiling, which is also the ``uniform`` anchor's);
    * the search's share of kept rows on its cap and its capacity bound k_min,
      the fewest pool rows whose capped design weights carry D's mass, in
      total and nation by nation;
    * the penalty pressure λ·P/L of ``l2_grid`` per anchor (L is S0's loss
      under the run's weights) and the half-decade shift that would put the
      grid's low end on ``pressure_band``'s;
    * each rule's loss shares and its losses at D and S0;
    * the early size (F) triggers: k_min above the requested size, or areas
      the support can never lift to the rule.

    Self-checks: S0's households are the draw's support in pool order, S0
    never exceeds its cap at the run's floor, and the recomputed start
    reproduces the stored receipt's certainty mass share.
    """

    started = time.perf_counter()
    problem = baseline.problem
    design = np.asarray(problem.problem.initial_weights.values, dtype=np.float64)
    dense = np.asarray(baseline.dense.weights, dtype=np.float64)
    support = np.asarray(baseline.draw.support, dtype=np.int64)
    q = np.asarray(baseline.draw.inclusion_probabilities, dtype=np.float64)
    s0 = np.asarray(baseline.stored_refit_weights, dtype=np.float64)
    if tuple(problem.entity_ids[index] for index in support) != tuple(
        baseline.stored_refit_ids
    ):
        raise ValueError(
            "the stored refit's households are not the draw's support in pool order."
        )
    ratio = float(baseline.settings["max_weight_ratio"])
    households = int(baseline.settings["households"])
    stored_floor = float(baseline.settings["baseline_pi_floor"])
    profile = baseline.profile
    nations = sorted(set(profile.nation.tolist()))
    types = sorted(set(profile.household_type.tolist()))
    dense_total = float(dense.sum())
    dense_by_nation = {n: float(dense[profile.nation == n].sum()) for n in nations}
    dense_by_type = {t: float(dense[profile.household_type == t].sum()) for t in types}
    kept_nation, kept_type = profile.nation[support], profile.household_type[support]
    certain = q >= 1.0
    share = float(UK_SIZE_ACCEPTANCE["relative_collapse_share"])
    pool_positions = np.arange(len(design), dtype=np.int64)
    grains = {}
    for grain, codes in (
        ("constituency", profile.constituency_code),
        ("la", profile.local_authority_code),
    ):
        area, roster = pd.factorize(pd.Series(codes, dtype=object), sort=True)
        dense_ess = _kish_by_area(area, pool_positions, dense, len(roster))
        grains[grain] = (area, len(roster), share * dense_ess, dense_ess > 0)

    def below_rule(weights: np.ndarray) -> dict[str, int]:
        return {
            grain: int(
                (
                    (_kish_by_area(area, support, weights, n_areas) < threshold)
                    & present
                ).sum()
            )
            for grain, (area, n_areas, threshold, present) in grains.items()
        }

    def masses(values: np.ndarray) -> dict[str, Any]:
        return {
            "total": float(values.sum() / dense_total),
            "by_nation": _mass_to(kept_nation, values, dense_by_nation),
            "by_household_type": _mass_to(kept_type, values, dense_by_type),
        }

    run_weights = np.asarray(problem.bindings["target_loss_weights"], dtype=np.float64)
    estimates_s0 = _estimates(baseline, support, s0)
    loss_s0 = _loss(baseline, estimates_s0, run_weights)
    floor_blocks: dict[str, dict[str, Any]] = {}
    for floor in sorted({*(float(value) for value in floors), stored_floor}):
        start = _refit_start(design, support, q, floor)
        cap = ratio * start
        block: dict[str, Any] = {
            "floored_rows": int(np.count_nonzero(q < floor)),
            "start_mass_share_certainties": float(start[certain].sum() / start.sum()),
            "start_to_dense": masses(start),
            "capacity_to_dense": masses(cap),
            "areas_start_below_rule": below_rule(start),
            "initial_anchor_penalty_at_s0": float(chi_square_distance(s0, start)),
        }
        if floor == stored_floor:
            block["s0_share_at_cap"] = float(
                np.mean(s0 >= (1.0 - UK_SIZE_CAP_TOLERANCE) * cap)
            )
            block["s0_above_cap_rows"] = int(
                np.count_nonzero(s0 > (1.0 + UK_SIZE_CAP_TOLERANCE) * cap)
            )
        floor_blocks[f"{floor:g}"] = block
    area_blocks = {}
    for grain, (area, n_areas, threshold, present) in grains.items():
        kept = np.bincount(area[support], minlength=n_areas).astype(np.float64)
        s0_ess = _kish_by_area(area, support, s0, n_areas)
        area_blocks[grain] = {
            "areas": int(present.sum()),
            "areas_support_ceiling": int(((kept < threshold) & present).sum()),
            "areas_without_support": int(((kept == 0) & present).sum()),
            "areas_s0_below_rule": int(((s0_ess < threshold) & present).sum()),
        }

    def pressure(scale: float) -> dict[str, Any]:
        block: dict[str, Any] = {"scale": scale}
        if loss_s0 > 0.0 and scale > 0.0:
            block["pressure"] = {
                f"{value:g}": float(value * scale / loss_s0) for value in l2_grid
            }
            low = min(l2_grid) * scale / loss_s0
            shift = int(np.round(2.0 * np.log10(pressure_band[0] / low)))
            block["suggested_half_decade_shift"] = shift
            block["shifted_grid"] = [
                float(value * 10 ** (shift / 2)) for value in l2_grid
            ]
        return block

    uniform = np.full(len(s0), design.sum() / len(s0))
    penalty = {
        "loss_s0_run_rule": loss_s0,
        "grid": [float(value) for value in l2_grid],
        "band": [float(value) for value in pressure_band],
        "uniform": pressure(float(chi_square_distance(s0, uniform))),
        "initial": {
            key: pressure(block["initial_anchor_penalty_at_s0"])
            for key, block in floor_blocks.items()
        },
    }
    search = baseline.selection.selection
    search_ratio = float(search.options.get("max_weight_ratio") or ratio)
    kept_weights = np.asarray(search.weights, dtype=np.float64)[support]
    kept_start = np.asarray(search.initial_weights, dtype=np.float64)[support]
    on_cap = kept_weights >= (1.0 - UK_SIZE_CAP_TOLERANCE) * search_ratio * kept_start
    k_by_nation = {
        n: _k_min(design[profile.nation == n], dense_by_nation[n], search_ratio)
        for n in nations
    }
    k_total = _k_min(design, dense_total, search_ratio)
    k_nations = (
        None
        if any(value is None for value in k_by_nation.values())
        else int(sum(k_by_nation.values()))
    )
    kept_design = design[support]
    search_block = {
        "l0_lambda": float(search.l0_lambda),
        "max_weight_ratio": search_ratio,
        "kept_share_at_cap": float(on_cap.mean()),
        "certainties_share_at_cap": (
            float(on_cap[certain].mean()) if certain.any() else None
        ),
        "pool_design_weight_mean": float(design.mean()),
        "pool_design_weight_median": float(np.median(design)),
        "kept_design_weight_median": float(np.median(kept_design)),
        "certainties_design_weight_mean": (
            float(kept_design[certain].mean()) if certain.any() else None
        ),
        "k_min": {
            "total": k_total,
            "by_nation": k_by_nation,
            "sum_of_nations": k_nations,
        },
    }
    estimates_dense = np.asarray(problem.problem.matrix @ dense, dtype=np.float64)
    rules = {}
    for rule in UK_LOCAL_TARGET_WEIGHT_RULES:
        weights = uk_rule_loss_weights(baseline.rows, rule=rule)
        rule_s0 = _loss(baseline, estimates_s0, weights)
        rules[rule] = {
            "loss_shares": uk_target_loss_weight_receipt(
                baseline.rows, weights, rule=rule
            ),
            "loss_dense": _loss(baseline, estimates_dense, weights),
            "loss_s0": rule_s0,
            "loss_s0_to_run_rule": rule_s0 / loss_s0 if loss_s0 > 0.0 else None,
        }
    triggers: dict[str, Any] = {
        "k_min_exceeds_households": k_total is None or k_total > households,
        "nation_k_min_exceeds_households": k_nations is None or k_nations > households,
        "areas_support_ceiling": {
            grain: block["areas_support_ceiling"]
            for grain, block in area_blocks.items()
        },
    }
    triggers["any"] = bool(
        triggers["k_min_exceeds_households"]
        or triggers["nation_k_min_exceeds_households"]
        or any(count > 0 for count in triggers["areas_support_ceiling"].values())
    )
    stored_share = baseline.size_receipt.get("baseline_mass_share_certainties")
    recomputed = floor_blocks[f"{stored_floor:g}"]["start_mass_share_certainties"]
    checks = {
        "s0_above_cap_rows": floor_blocks[f"{stored_floor:g}"]["s0_above_cap_rows"],
        "start_matches_receipt": (
            None
            if stored_share is None
            else bool(abs(recomputed - float(stored_share)) <= 1e-9)
        ),
    }
    checks["passed"] = checks["s0_above_cap_rows"] == 0 and checks[
        "start_matches_receipt"
    ] in (True, None)
    return {
        "households": households,
        "pool_households": int(len(design)),
        "max_weight_ratio": ratio,
        "stored_floor": stored_floor,
        "dense_mass": {
            "total": dense_total,
            "by_nation": dense_by_nation,
            "by_household_type": dense_by_type,
        },
        "search": search_block,
        "floors": floor_blocks,
        "areas": area_blocks,
        "penalty": penalty,
        "rules": rules,
        "early_size_triggers": triggers,
        "checks": checks,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
        "baseline": {
            "run_dir": str(baseline.run_dir),
            "run_rule": baseline.run_rule,
            "digests": dict(baseline.digests),
        },
        "code": {"git_commit": git_commit(), "git_dirty": git_dirty()},
    }


def _scaled_initial_lambda(
    baseline: UKSizeExperimentBaseline, rule: str | None
) -> float:
    stored = float(baseline.settings["selected_l0_lambda"])
    if rule is None or rule == baseline.run_rule:
        return stored
    support = np.asarray(baseline.draw.support, dtype=np.int64)
    estimates = _estimates(baseline, support, baseline.stored_refit_weights)
    run_weights = uk_rule_loss_weights(baseline.rows, rule=baseline.run_rule)
    rule_weights = uk_rule_loss_weights(baseline.rows, rule=rule)
    return (
        stored
        * _loss(baseline, estimates, rule_weights)
        / _loss(baseline, estimates, run_weights)
    )


def run_uk_size_experiment(
    baseline: UKSizeExperimentBaseline, experiment: UKSizeExperiment
) -> dict[str, Any]:
    """Run one experiment; returns its receipt, its sized result(s) and weights."""

    started = time.perf_counter()
    settings = _refit_settings(baseline, experiment)
    selection = baseline.selection
    receipt: dict[str, Any] = {
        "status": "finished",
        "experiment": asdict(experiment),
        "settings": dict(settings),
    }
    if experiment.learning_rate is not None and experiment.mode != "selection":
        # A refit-only learning-rate perturbation: the stored search keeps its
        # own rate; the refit (which reuses the search's rate) gets this one.
        selection = replace(selection, learning_rate=float(experiment.learning_rate))
        settings["learning_rate"] = float(experiment.learning_rate)
        receipt["refit_learning_rate_override"] = float(experiment.learning_rate)
    if experiment.epochs is not None:
        # A smoke run of the setup: the refit (which reuses the search's epoch
        # count) runs this many; the stored search keeps its own.
        selection = replace(selection, epochs=int(experiment.epochs))
        settings["epochs"] = int(experiment.epochs)
        receipt["refit_epochs_override"] = int(experiment.epochs)
    refit_l2 = _l2_kwargs("refit", experiment.refit_l2)
    if experiment.mode == "refit":
        sized = refit_uk_dataset_size(
            baseline.frame,
            baseline.dense,
            selection=selection,
            draw=baseline.draw,
            refit_target_weighting=_weighting(baseline, experiment.refit_rule),
            **_stored_selection_l2_kwargs(baseline),
            **refit_l2,
            **settings,
        )
        results = {"refit": sized}
    elif experiment.mode == "selection":
        weighting = _weighting(baseline, experiment.selection_rule)
        search_kwargs = dict(
            households=settings["households"],
            epochs=settings["epochs"],
            learning_rate=settings["learning_rate"],
            seed=settings["seed"],
            pi_hi=float(baseline.selection.search_pi_hi),
            target_weighting=weighting,
            **_l2_kwargs("selection", experiment.selection_l2),
        )
        warm = experiment.initial_lambda
        if warm == "scaled":
            warm = _scaled_initial_lambda(baseline, experiment.selection_rule)
        if warm is not None:
            if (
                "initial_lambda"
                not in inspect.signature(select_uk_dataset_size).parameters
            ):
                raise RuntimeError(
                    "this tree's select_uk_dataset_size has no initial_lambda warm "
                    "start (microcosm#1115); run selection experiments from a tree "
                    "that has it, or set initial_lambda to null for a cold search."
                )
            search_kwargs["initial_lambda"] = float(warm)
        receipt["initial_lambda"] = warm
        new_selection = select_uk_dataset_size(
            baseline.frame, baseline.dense, **search_kwargs
        )
        new_draw = draw_uk_dataset_size(
            baseline.frame,
            baseline.dense,
            selection=new_selection,
            households=settings["households"],
            seed=settings["seed"],
            pi_hi=settings["pi_hi"],
        )
        sized = refit_uk_dataset_size(
            baseline.frame,
            baseline.dense,
            selection=new_selection,
            draw=new_draw,
            selection_target_weighting=weighting,
            refit_target_weighting=_weighting(baseline, experiment.refit_rule),
            **_l2_kwargs("selection", experiment.selection_l2),
            **refit_l2,
            **settings,
        )
        results = {"refit": sized}
        receipt["search"] = {
            "l0_lambda": float(new_selection.selection.l0_lambda),
            "budget_search": new_selection.selection.options.get("budget_search"),
            "options": dict(new_selection.selection.options),
        }
    else:
        results, receipt["holdout"] = _run_holdout(
            baseline, experiment, selection, settings, refit_l2
        )
        sized = None
    if sized is not None:
        support = _support_positions(baseline, sized)
        estimates = _estimates(baseline, support, sized.result.weights)
        receipt["size_receipt"] = {
            key: value
            for key, value in sized.receipt.items()
            if key not in {"pool_row_indices", "inclusion_probabilities"}
        }
        receipt["refit_options"] = dict(sized.result.options)
        receipt["losses"] = {
            "training": float(sized.result.final_loss),
            "grain_equal_yardstick": _loss(
                baseline,
                estimates,
                uk_rule_loss_weights(baseline.rows, rule="grain_equal"),
            ),
            "family_equal_yardstick": _loss(
                baseline,
                estimates,
                uk_rule_loss_weights(baseline.rows, rule="grain_family_equal"),
            ),
        }
    receipt["wall_seconds"] = time.perf_counter() - started
    receipt["peak_rss_bytes"] = _peak_rss_bytes()
    receipt["baseline"] = {
        "run_dir": str(baseline.run_dir),
        "run_rule": baseline.run_rule,
        "digests": dict(baseline.digests),
    }
    receipt["code"] = {"git_commit": git_commit(), "git_dirty": git_dirty()}
    try:
        import torch

        receipt["torch"] = {
            "version": torch.__version__,
            "threads": torch.get_num_threads(),
        }
    except ImportError:  # pragma: no cover - torch is a calibrate dependency
        receipt["torch"] = None
    return {"receipt": receipt, "results": results}


def uk_size_failed_receipt(
    baseline: UKSizeExperimentBaseline,
    experiment: UKSizeExperiment,
    error: Exception,
    *,
    wall_seconds: float,
) -> dict[str, Any]:
    """The receipt of a configuration the solver chain refused: a result, not a stop.

    The refusal's type and message are the finding (a refit that lost
    positive support, a search with no drawable probe, ...). The traceback is
    kept for the out directory and dropped on publication.
    """

    return {
        "status": "failed",
        "experiment": asdict(experiment),
        "error": {"type": type(error).__name__, "message": str(error)},
        "traceback": traceback.format_exception(error)[-20:],
        "wall_seconds": float(wall_seconds),
        "peak_rss_bytes": _peak_rss_bytes(),
        "baseline": {
            "run_dir": str(baseline.run_dir),
            "run_rule": baseline.run_rule,
            "digests": dict(baseline.digests),
        },
        "code": {"git_commit": git_commit(), "git_dirty": git_dirty()},
    }


def _run_holdout(baseline, experiment, selection, settings, refit_l2):
    local = np.flatnonzero(np.asarray(baseline.rows.local, dtype=bool))
    folds = rotated_folds(
        len(local), n_folds=int(experiment.folds), seed=UK_LOCAL_HOLDOUT_SEED
    )
    rule = baseline.run_rule if experiment.refit_rule is None else experiment.refit_rule
    targets = np.asarray(baseline.problem.problem.target_vector, dtype=np.float64)
    scales = default_target_loss_scales(targets)
    results, fold_rows = {}, []
    for index, fold in enumerate(folds):
        held = local[np.asarray(fold, dtype=np.int64)]
        weighting = UKStageTargetWeighting(
            rule, baseline.rows, held_out=tuple(held.tolist())
        )
        sized = refit_uk_dataset_size(
            baseline.frame,
            baseline.dense,
            selection=selection,
            draw=baseline.draw,
            refit_target_weighting=weighting,
            **_stored_selection_l2_kwargs(baseline),
            **refit_l2,
            **settings,
        )
        support = np.asarray(sized.support, dtype=np.int64)
        held_sorted = np.asarray(weighting.held_out, dtype=np.int64)
        estimates = np.asarray(
            baseline.problem.problem.matrix[held_sorted][:, support]
            @ sized.result.weights,
            dtype=np.float64,
        )
        held_targets = targets[held_sorted]
        error = np.abs(estimates - held_targets) / scales[held_sorted]
        cap = float(baseline.settings["target_loss_cap"])
        fold_rows.append(
            {
                "fold": index,
                "n_held": int(held_sorted.size),
                "held_loss_rule": float(
                    relative_error_loss(
                        estimates,
                        held_targets,
                        target_loss_weights=weighting.held_out_weights(),
                        target_loss_scales=scales[held_sorted],
                        target_loss_cap=cap,
                    )
                ),
                "held_loss_grain_equal": float(
                    relative_error_loss(
                        estimates,
                        held_targets,
                        target_loss_weights=uk_rule_loss_weights(
                            baseline.rows.take(held_sorted), rule="grain_equal"
                        ),
                        target_loss_scales=scales[held_sorted],
                        target_loss_cap=cap,
                    )
                ),
                "held_within_10pct": float((error <= 0.10).mean()),
                "held_within_25pct": float((error <= 0.25).mean()),
                "training_loss": float(sized.result.final_loss),
            }
        )
        results[f"fold_{index}"] = sized
    summary = {
        key: float(np.mean([row[key] for row in fold_rows]))
        for key in (
            "held_loss_rule",
            "held_loss_grain_equal",
            "held_within_10pct",
            "held_within_25pct",
        )
    }
    return results, {
        "method": "refit_level_rotated_local_folds",
        "seed": UK_LOCAL_HOLDOUT_SEED,
        "rule": rule,
        "folds": fold_rows,
        "mean": summary,
        "limitation": "support selected with every target visible; ranks refit variants on one support only",
    }


def uk_size_weights_of(
    label: str, sized: UKDatasetSize, *, max_weight_ratio
) -> UKSizeWeights:
    """A sized result as the scorecard's weight set (support, weights, start)."""

    return UKSizeWeights(
        label,
        np.asarray(sized.support, dtype=np.int64),
        np.asarray(sized.result.weights, dtype=np.float64),
        initial=np.asarray(sized.result.initial_weights, dtype=np.float64),
        max_weight_ratio=max_weight_ratio,
        training_loss_weights=np.asarray(
            sized.result.target_loss_weights, dtype=np.float64
        ),
    )


def save_uk_size_weights(weights: UKSizeWeights, path: Path) -> None:
    """Write a weight set's arrays (unit-level: never into the repository)."""

    arrays = {"support": weights.support, "weights": weights.weights}
    if weights.initial is not None:
        arrays["initial"] = weights.initial
    if weights.training_loss_weights is not None:
        arrays["training_loss_weights"] = weights.training_loss_weights
    np.savez(Path(path), **arrays)


def load_uk_size_weights(label: str, path: Path, *, max_weight_ratio) -> UKSizeWeights:
    with np.load(Path(path), allow_pickle=False) as stored:
        return UKSizeWeights(
            label,
            stored["support"],
            stored["weights"],
            initial=stored["initial"] if "initial" in stored else None,
            max_weight_ratio=max_weight_ratio,
            training_loss_weights=(
                stored["training_loss_weights"]
                if "training_loss_weights" in stored
                else None
            ),
        )


@dataclass(frozen=True)
class UKSizeScoringInputs:
    """What the scorecard reads: the problem, its rows, the profile and D."""

    problem: OrderedProblem
    rows: UKTargetRows
    profile: UKPoolProfile
    dense: UKSizeWeights
    max_weight_ratio: float | None
    target_loss_cap: float


def load_uk_size_scoring_inputs(run_dir: Path, cache_dir: Path) -> UKSizeScoringInputs:
    """The scorecard's inputs without decoding any calibration result."""

    run_dir, cache_dir = Path(run_dir).resolve(), Path(cache_dir).resolve()
    evidence = _evidence(run_dir)
    problem = decode_problem(_read_artifact(run_dir, evidence, "problem"))
    entry = evidence["uk.full.dense/solution"]
    payload = (run_dir / str(entry["filename"])).read_bytes()
    if _sha256_bytes(payload) != entry["sha256"]:
        raise ValueError(f"{entry['filename']} differs from its evidence-index digest.")
    solution = decode_solution(
        payload, problem_sha256=problem.sha256, entity_ids=problem.entity_ids
    )
    bindings = problem.bindings
    max_ratio = bindings.get("max_weight_ratio")
    dense = UKSizeWeights(
        "D",
        np.arange(len(problem.entity_ids), dtype=np.int64),
        np.asarray(solution.weights, dtype=np.float64),
        initial=np.asarray(problem.problem.initial_weights.values, dtype=np.float64),
        max_weight_ratio=max_ratio,
    )
    return UKSizeScoringInputs(
        problem=problem,
        rows=uk_target_rows_from_problem(problem),
        profile=load_uk_pool_profile(
            cache_dir / "profile",
            problem_sha256=problem.sha256,
            household_ids=problem.entity_ids,
        ),
        dense=dense,
        max_weight_ratio=max_ratio,
        target_loss_cap=float(bindings["target_loss_cap"]),
    )


def score_uk_size_experiments(
    inputs: UKSizeScoringInputs,
    control: UKSizeWeights,
    candidates: Sequence[UKSizeWeights] = (),
) -> dict[str, Any]:
    """The scorecard of D, the stored run S0 (as reproduced) and candidates."""

    return uk_size_scorecard(
        profile=inputs.profile,
        rows=inputs.rows,
        matrix=inputs.problem.problem.matrix,
        targets=np.asarray(inputs.problem.problem.target_vector, dtype=np.float64),
        yardstick=uk_rule_loss_weights(inputs.rows, rule="grain_equal"),
        target_loss_cap=inputs.target_loss_cap,
        dense=inputs.dense,
        control=control,
        candidates=candidates,
    )
