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
* :func:`run_uk_size_experiment` runs one :class:`UKSizeExperiment`: a refit
  on the stored support under another rule, L2 penalty or baseline floor; a
  new search from the stored dense solve (warm-started) then a draw and a
  refit; or a refit-level rotated holdout on the stored support.

Each experiment returns its receipt and its weights; the tool
``tools/run_uk_size_experiment.py`` writes them outside the repository and the
run directory. The dense reference is never re-solved: every delta is the
selection's.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from microcosm.build.holdout import rotated_folds
from microcosm.build.uk_runtime.dataset_size import (
    UKDatasetSize,
    UKSizeDraw,
    UKSizeSelection,
    draw_uk_dataset_size,
    refit_uk_dataset_size,
    select_uk_dataset_size,
    uk_size_l2,
)
from microcosm.build.uk_runtime.local_doctrine import UK_LOCAL_TARGET_WEIGHT_RULES
from microcosm.build.uk_runtime.local_rowwise import (
    UK_LOCAL_HOLDOUT_FOLDS,
    UK_LOCAL_HOLDOUT_SEED,
)
from microcosm.build.uk_runtime.rowwise_cli import git_commit, git_dirty
from microcosm.build.uk_runtime.size_experiment_scorecard import (
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
    uk_target_rows_from_problem,
)
from microcosm.calibrate import (
    CalibrationResult,
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
    "UK_SIZE_EXPERIMENT_ARTIFACTS",
    "UK_SIZE_EXPERIMENT_MODES",
    "UKSizeExperiment",
    "UKSizeExperimentBaseline",
    "build_uk_size_experiment_cache",
    "load_uk_size_experiment_baseline",
    "parse_uk_size_experiments",
    "run_uk_size_control",
    "run_uk_size_experiment",
    "UKSizeScoringInputs",
    "load_uk_size_scoring_inputs",
    "load_uk_size_weights",
    "save_uk_size_weights",
    "score_uk_size_experiments",
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
    ``initial_lambda`` (selection mode) is a warm-start penalty, ``"scaled"``
    for the stored search's penalty times the rule's loss ratio at S0, or
    ``None`` for a cold search.
    """

    name: str
    mode: str = "refit"
    refit_rule: str | None = None
    selection_rule: str | None = None
    refit_l2: Mapping[str, Any] | None = None
    selection_l2: Mapping[str, Any] | None = None
    baseline_pi_floor: float | None = None
    learning_rate: float | None = None
    initial_lambda: float | str | None = "scaled"
    folds: int = UK_LOCAL_HOLDOUT_FOLDS
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or self.name.startswith("."):
            raise ValueError(
                f"experiment name {self.name!r} is not a plain directory name."
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
        "experiment": asdict(experiment),
        "settings": dict(settings),
    }
    if experiment.learning_rate is not None and experiment.mode != "selection":
        # A refit-only learning-rate perturbation: the stored search keeps its
        # own rate; the refit (which reuses the search's rate) gets this one.
        selection = replace(selection, learning_rate=float(experiment.learning_rate))
        settings["learning_rate"] = float(experiment.learning_rate)
        receipt["refit_learning_rate_override"] = float(experiment.learning_rate)
    refit_l2 = _l2_kwargs("refit", experiment.refit_l2)
    if experiment.mode == "refit":
        sized = refit_uk_dataset_size(
            baseline.frame,
            baseline.dense,
            selection=selection,
            draw=baseline.draw,
            refit_target_weighting=_weighting(baseline, experiment.refit_rule),
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
