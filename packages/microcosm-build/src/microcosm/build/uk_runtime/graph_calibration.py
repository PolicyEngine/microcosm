"""UK solver bindings over shared, ordered calibration artifacts.

Every solver consumes the original pool. Dense weights are not a selection
prior; filtering and installing the completed solution are separate structural
operations. Completed search and draw artifacts resume without repeating RNG.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from microcosm.calibrate import artifacts as calibration_artifacts
from microcosm.calibrate import calibrate
from microcosm.calibrate.artifacts import (
    PROBLEM_TYPE,
    RESULT_TYPE,
    SOLUTION_TYPE,
    decode_calibration_result,
    decode_problem,
    decode_solution,
    encode_calibration_result,
    encode_problem,
    encode_solution,
)
from microcosm.calibrate.exact_k import select_exact_k
from microcosm.calibrate.gates import HardConcrete
from microcosm.calibrate.initialization import contribution_initialization
from microcosm.calibrate.kernels import CalibrateAdamKernel
from microcosm.frame import Frame, MassChangeRecord, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    SeedSource,
    StructuralDelta,
    WeightTransition,
    source_hash,
)
from microcosm.graph.canonical import canonical_json

from . import dataset_size, size_checkpoint
from .dataset_size import UKSizeDraw, UKSizeSelection
from .graph_population import context_frame, population_slices
from .local_doctrine import UK_LOCAL_SOLVE_EPOCHS

SIZE_SEARCH_TYPE = ArtifactType("microcosm.uk.size-search", 1)
SIZE_DRAW_TYPE = ArtifactType("microcosm.uk.size-draw", 1)
SIZE_RECEIPT_TYPE = ArtifactType("microcosm.uk.size-receipt", 1)
_DEPENDENCIES = ("numpy", "pandas", "scipy", "torch")


@dataclass(frozen=True)
class UKGraphCalibrationConfig:
    epochs: int = UK_LOCAL_SOLVE_EPOCHS
    learning_rate: float = 0.15
    seed: int = 42
    dataset_households: int | None = None
    selection_seed: int | None = None
    selection_pi_hi: float = 1.0
    target_weight_rule: str = "uniform"

    def __post_init__(self):
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("Calibration epochs must be a positive integer.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Calibration learning rate must be positive and finite.")
        for seed in (self.seed, self.selection_seed):
            if seed is not None and (type(seed) is not int or seed < 0):
                raise ValueError("Calibration seeds must be nonnegative integers.")
        if self.dataset_households is not None and (
            type(self.dataset_households) is not int or self.dataset_households < 1
        ):
            raise ValueError("Dataset household count must be a positive integer.")
        dataset_size._check_pi_hi(self.selection_pi_hi)


@dataclass(frozen=True)
class UKCalibrationNodes:
    nodes: tuple[Node, ...]
    population: str
    result_producer: str
    problem_producer: str
    solution_producer: str
    size_producer: str | None
    dense_producer: str


def _axis(frame: Frame) -> list:
    return frame.table("household")["household_id"].tolist()


def _inputs(context: KernelContext):
    frame = context_frame(context)
    problem = decode_problem(context.artifacts["problem"].payload)
    if (
        tuple(_axis(frame)) != problem.entity_ids
        or problem.problem.weight_entity != "household"
    ):
        raise ValueError("UK solve requires its exact original household axis.")
    weights = frame.weights_for("household")
    if weights.kind != problem.problem.initial_weights.kind or not np.array_equal(
        weights.values, problem.problem.initial_weights.values
    ):
        raise ValueError("UK solve requires the original pool weights.")
    if problem.problem.skipped:
        raise ValueError("UK solve cannot silently omit selected targets.")
    return frame, problem


def _dense(context, frame, problem):
    return decode_calibration_result(
        context.artifacts["dense"].payload, frame=frame, problem=problem
    )


def _solution(result, frame, problem):
    if result.frame.mass_log[: len(frame.mass_log)] != frame.mass_log:
        raise ValueError("Calibration replaced the original pool mass ledger.")
    return encode_solution(
        result.weights,
        entity_ids=_axis(result.frame),
        problem_sha256=problem.sha256,
        diagnostics={
            "frame_mass_log_append": [
                asdict(record)
                for record in result.frame.mass_log[len(frame.mass_log) :]
            ]
        },
    )


def restore_uk_graph_result(
    pool: Frame,
    *,
    problem_payload: bytes,
    result_payload: bytes,
    solution_payload: bytes,
    original_problem_payload: bytes | None = None,
):
    """Rebuild completed diagnostics and exact legacy mass evidence, never solve.

    A compact result has its own HT-normalized initial weights and matrix.
    Its original problem is required to authenticate the supplied full pool.
    ``solution_payload`` is the original-bound installation solution.
    """
    problem = decode_problem(problem_payload)
    original = (
        problem
        if original_problem_payload is None
        else decode_problem(original_problem_payload)
    )
    if (
        tuple(_axis(pool)) != original.entity_ids
        or not np.array_equal(
            pool.weights_for("household").values,
            original.problem.initial_weights.values,
        )
        or pool.weights_for("household").kind != original.problem.initial_weights.kind
    ):
        raise ValueError("Completed result requires its authenticated original pool.")
    if (
        problem.sha256 != original.sha256
        and problem.bindings.get("original_problem_sha256") != original.sha256
    ):
        raise ValueError(
            "Compact result is not bound to the supplied original problem."
        )
    solution = decode_solution(
        solution_payload, problem_sha256=original.sha256, entity_ids=problem.entity_ids
    )
    ids = set(problem.entity_ids)
    selected = pool.select(
        pool.table("person")["person_household_id"].isin(ids).to_numpy()
    )
    if tuple(_axis(selected)) != problem.entity_ids:
        raise ValueError("Completed result is not an ordered subset of the pool.")
    initial_frame = Frame(
        {e: selected.table(e) for e in selected.entities},
        selected.schema,
        {"household": problem.problem.initial_weights},
        selected.strata,
        mass_log=pool.mass_log,
        metadata=pool.metadata,
    )
    result = decode_calibration_result(
        result_payload, frame=initial_frame, problem=problem
    )
    if not np.array_equal(result.weights, solution.weights):
        raise ValueError("Completed result disagrees with its installed solution.")
    records = tuple(
        MassChangeRecord(**dict(row))
        for row in solution.diagnostics["frame_mass_log_append"]
    )
    if records and (
        not np.isclose(records[0].old_total, selected.weights_for("household").total)
        or not np.isclose(records[-1].new_total, float(result.weights.sum()))
    ):
        raise ValueError("Completed result mass evidence differs from its boundary.")
    final_frame = Frame(
        {e: selected.table(e) for e in selected.entities},
        selected.schema,
        {"household": result.frame.weights_for("household")},
        selected.strata,
        mass_log=(*pool.mass_log, *records),
        metadata=pool.metadata,
    )
    return replace(result, frame=final_frame)


class _CalibrationKernel(KernelBase):
    def implementation_hash(self):
        return hashlib.sha256(
            canonical_json(
                {
                    "solver": CalibrateAdamKernel().implementation_hash(),
                    "adapter": source_hash(
                        type(self),
                        dataset_size,
                        size_checkpoint,
                        calibration_artifacts,
                        context_frame,
                        select_exact_k,
                        HardConcrete,
                        contribution_initialization,
                        dependencies=self.capabilities.dependencies,
                    ),
                }
            )
        ).hexdigest()


class UKSizeCheckpointImportKernel(_CalibrationKernel):
    """Authenticate existing external checkpoints as explicit graph sources."""

    ref = "uk.full.size_checkpoint_import@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC, dependencies=_DEPENDENCIES)

    def run(self, context):
        frame, problem = _inputs(context)
        manifest_path = context.sources[context.params["manifest_source"]]
        arrays_path = context.sources[context.params["arrays_source"]]
        identity = json.loads(context.params["identity_json"])
        stored = json.loads(manifest_path.read_bytes())
        # The historical loader accepts requested subsets. Graph import must
        # authenticate every stored caller pin rather than leave old source,
        # selector or K settings outside the comparison.
        if set(identity) != set(stored.get("identity", {})):
            raise ValueError(
                "Size checkpoint import requires every original identity field."
            )
        restored = size_checkpoint.load_uk_size_checkpoint_files(
            manifest_path,
            arrays_path,
            frame=frame,
            target_set=problem.to_target_set(),
            identity=identity,
        )
        dense, selection = restored.dense, restored.selection
        for key in ("epochs", "learning_rate"):
            if dense.options[key] != context.params[key]:
                raise ValueError(f"Imported dense solve has different {key}.")
        if dense.options["seed"] != context.params["dense_seed"]:
            raise ValueError("Imported dense solve has a different seed.")
        for key in ("households", "epochs", "learning_rate", "seed"):
            if getattr(selection, key) != context.params[key]:
                raise ValueError(f"Imported size search has different {key}.")
        binding = problem.bindings
        if (
            dense.options["mass"] != "free"
            or dense.options["mass_reason"] != binding["mass_reason"]
            or dense.options["max_weight_ratio"] != binding["max_weight_ratio"]
            or dense.target_loss_cap != binding["target_loss_cap"]
            or not np.array_equal(
                dense.target_loss_weights, binding["target_loss_weights"]
            )
        ):
            raise ValueError("Imported dense solve has a different solve doctrine.")
        metadata = {
            "method": "contribution_informed_l0",
            "problem_sha256": problem.sha256,
            "protected": selection.protected.tolist(),
            "households": selection.households,
            "epochs": selection.epochs,
            "learning_rate": selection.learning_rate,
            "seed": selection.seed,
            "pi_hi": selection.search_pi_hi,
        }
        return KernelResult(
            artifacts={
                "dense": encode_calibration_result(
                    dense, entity_ids=problem.entity_ids, problem_sha256=problem.sha256
                ),
                "search": encode_calibration_result(
                    selection.selection,
                    entity_ids=problem.entity_ids,
                    problem_sha256=problem.sha256,
                ),
                "selection": canonical_json(metadata),
            }
        )


class UKDenseSolveKernel(_CalibrationKernel):
    ref = "uk.full.dense@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        seed_source=SeedSource.PARAM,
        dependencies=_DEPENDENCIES,
    )

    def run(self, context):
        from .graph_terminal import decode_full_gate_report

        if "preflight" not in context.artifacts:
            raise ValueError(
                "Dense calibration requires its source preflight artifact."
            )
        report, classification = decode_full_gate_report(
            context.artifacts["preflight"].payload
        )
        if report.phase != "preflight":
            raise ValueError("Dense calibration requires a preflight phase report.")
        if not classification["artifact_permitted"]:
            raise ValueError("Dense calibration refused by the source preflight.")
        frame, problem = _inputs(context)
        if "imported_dense" in context.artifacts:
            result = decode_calibration_result(
                context.artifacts["imported_dense"].payload,
                frame=frame,
                problem=problem,
            )
            return KernelResult(
                artifacts={
                    "result": context.artifacts["imported_dense"].payload,
                    "solution": _solution(result, frame, problem),
                }
            )
        binding = problem.bindings
        # These are the maintained full-build doctrine values carried by the
        # selected problem, regardless of the geographic scope of its rows.
        required = {
            "mass_reason",
            "max_weight_ratio",
            "target_loss_weights",
            "target_loss_cap",
        }
        if not required <= set(binding):
            raise ValueError("UK ordered problem is missing its solve doctrine.")
        result = calibrate(
            frame,
            problem.to_target_set(),
            weight_entity="household",
            epochs=context.params["epochs"],
            learning_rate=context.params["learning_rate"],
            seed=context.params["seed"],
            mass="free",
            mass_reason=binding["mass_reason"],
            max_weight_ratio=binding["max_weight_ratio"],
            target_loss_weights=np.asarray(
                binding["target_loss_weights"], dtype=np.float64
            ),
            target_loss_cap=binding["target_loss_cap"],
        )
        return KernelResult(
            artifacts={
                "result": encode_calibration_result(
                    result, entity_ids=problem.entity_ids, problem_sha256=problem.sha256
                ),
                "solution": _solution(result, frame, problem),
            }
        )


def _full_pool(frame, context):
    k = context.params["households"]
    if k > frame.n("household"):
        raise ValueError(
            "Requested dataset size exceeds the original pool; never clamped."
        )
    return k == frame.n("household")


class UKSizeSearchKernel(_CalibrationKernel):
    ref = "uk.full.size_search@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        seed_source=SeedSource.PARAM,
        dependencies=_DEPENDENCIES,
    )

    def run(self, context):
        frame, problem = _inputs(context)
        dense = _dense(context, frame, problem)
        if "search" in context.artifacts:
            _selection(context, frame, problem)
            return KernelResult(
                artifacts={
                    "result": context.artifacts["search"].payload,
                    "selection": context.artifacts["selection"].payload,
                }
            )
        if _full_pool(frame, context):
            metadata = {
                "method": "full_pool",
                "problem_sha256": problem.sha256,
                **dict(context.params),
            }
            result_payload = context.artifacts["dense"].payload
        else:
            selection = dataset_size.select_uk_dataset_size(
                frame, dense, **dict(context.params)
            )
            metadata = {
                "method": "contribution_informed_l0",
                "problem_sha256": problem.sha256,
                "protected": selection.protected.tolist(),
                **dict(context.params),
            }
            result_payload = encode_calibration_result(
                selection.selection,
                entity_ids=problem.entity_ids,
                problem_sha256=problem.sha256,
            )
        return KernelResult(
            artifacts={"result": result_payload, "selection": canonical_json(metadata)}
        )


def _selection(context, frame, problem):
    metadata = json.loads(context.artifacts["selection"].payload)
    if metadata["problem_sha256"] != problem.sha256:
        raise ValueError("Size search belongs to a different ordered problem.")
    for key in ("households", "epochs", "learning_rate", "seed"):
        if metadata[key] != context.params[key]:
            raise ValueError(f"Size search has a different {key}.")
    if metadata["method"] == "full_pool":
        if not _full_pool(frame, context):
            raise ValueError("Full-pool receipt used for a compact request.")
        return None
    if metadata["method"] != "contribution_informed_l0":
        raise ValueError("Unknown UK size-search method.")
    raw_protected = np.asarray(metadata["protected"])
    if raw_protected.dtype != np.bool_ or raw_protected.shape != (
        frame.n("household"),
    ):
        raise ValueError("Size search protected mask is not aligned.")
    return UKSizeSelection(
        decode_calibration_result(
            context.artifacts["search"].payload, frame=frame, problem=problem
        ),
        raw_protected,
        metadata["households"],
        metadata["epochs"],
        metadata["learning_rate"],
        metadata["seed"],
        metadata["pi_hi"],
    )


class UKSizeDrawKernel(_CalibrationKernel):
    ref = "uk.full.size_draw@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        seed_source=SeedSource.PARAM,
        dependencies=_DEPENDENCIES,
    )

    def run(self, context):
        frame, problem = _inputs(context)
        dense = _dense(context, frame, problem)
        selection = _selection(context, frame, problem)
        if selection is None:
            draw = {"method": "full_pool", "support": list(range(frame.n("household")))}
        else:
            result = dataset_size.draw_uk_dataset_size(
                frame,
                dense,
                selection=selection,
                households=context.params["households"],
                seed=context.params["seed"],
                pi_hi=context.params["pi_hi"],
            )
            draw = {"method": "exact_count", **asdict(result)}
            for key in ("support", "inclusion_probabilities"):
                draw[key] = draw[key].tolist()
        return KernelResult(
            artifacts={
                "draw": canonical_json({"problem_sha256": problem.sha256, **draw})
            }
        )


class UKSizeRefitKernel(_CalibrationKernel):
    ref = "uk.full.size_refit@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        seed_source=SeedSource.PARAM,
        dependencies=_DEPENDENCIES,
    )

    def run(self, context):
        frame, problem = _inputs(context)
        dense = _dense(context, frame, problem)
        selection = _selection(context, frame, problem)
        draw = json.loads(context.artifacts["draw"].payload)
        if draw.pop("problem_sha256") != problem.sha256:
            raise ValueError("Exact-count draw belongs to another ordered problem.")
        method = draw.pop("method")
        if selection is None:
            if method != "full_pool" or draw["support"] != list(
                range(frame.n("household"))
            ):
                raise ValueError("Full-pool draw has a different support.")
            cached_draw = None
        else:
            if method != "exact_count":
                raise ValueError("Compact refit requires a completed exact-count draw.")
            cached_draw = UKSizeDraw(
                **{
                    **draw,
                    "support": np.asarray(draw["support"]),
                    "inclusion_probabilities": np.asarray(
                        draw["inclusion_probabilities"], dtype=np.float64
                    ),
                }
            )
        compact = dataset_size.refit_uk_dataset_size(
            frame, dense, selection=selection, draw=cached_draw, **dict(context.params)
        )
        result = compact.result
        ids = _axis(result.frame)
        if selection is None:
            compact_payload = context.artifacts["problem"].payload
            result_payload = context.artifacts["dense"].payload
            compact_problem = problem
        else:
            compact_payload = encode_problem(
                result.problem,
                entity_ids=ids,
                target_metadata=problem.target_metadata,
                bindings={
                    **dict(problem.bindings),
                    "original_problem_sha256": problem.sha256,
                },
            )
            compact_problem = decode_problem(compact_payload)
            result_payload = encode_calibration_result(
                result, entity_ids=ids, problem_sha256=compact_problem.sha256
            )
        return KernelResult(
            artifacts={
                "problem": compact_payload,
                "result": result_payload,
                "solution": _solution(result, frame, problem),
                "refit_solution": _solution(result, frame, compact_problem),
                "size": canonical_json(
                    {
                        **compact.receipt,
                        "problem_sha256": problem.sha256,
                        "household_ids": ids,
                    }
                ),
            }
        )


class UKSizeFilterKernel(_CalibrationKernel):
    ref = "uk.full.selected@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.FILTER,
        dependencies=_DEPENDENCIES,
    )

    def run(self, context):
        frame = context_frame(context)
        problem = decode_problem(context.artifacts["problem"].payload)
        solution = decode_solution(
            context.artifacts["solution"].payload, problem_sha256=problem.sha256
        )
        selected = set(solution.entity_ids)
        if [i for i in _axis(frame) if i in selected] != list(solution.entity_ids):
            raise ValueError("Selected household IDs are not an ordered pool subset.")
        person = frame.table("person")
        keep = pd.Series(
            person["person_household_id"].isin(selected).to_numpy(),
            index=pd.Index(person["person_id"], name="person_id"),
            dtype=bool,
        )
        return KernelResult(keep=keep)


class UKInstallCalibrationKernel(_CalibrationKernel):
    ref = "uk.full.calibrated@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.REWEIGHT,
        dependencies=_DEPENDENCIES,
    )

    def run(self, context):
        frame = context_frame(context)
        problem = decode_problem(context.artifacts["problem"].payload)
        solution = decode_solution(
            context.artifacts["solution"].payload,
            problem_sha256=problem.sha256,
            entity_ids=_axis(frame),
        )
        return KernelResult(
            weights=Weights(solution.weights, kind=WeightKind.CALIBRATED),
            receipt={
                "frame_mass_log_append": solution.diagnostics["frame_mass_log_append"]
            },
        )


def uk_calibration_nodes(
    *,
    base: str,
    columns: Mapping[tuple[str, str], str],
    problem_producer: str,
    problem_artifact: str = "problem",
    prefix: str = "uk.full",
    config: UKGraphCalibrationConfig | None = None,
    checkpoint_identity: Mapping | None = None,
    checkpoint_sources: tuple[str, str] = (
        "uk_size_checkpoint_manifest",
        "uk_size_checkpoint_arrays",
    ),
) -> UKCalibrationNodes:
    """Compose the same numerical route for every selected target scope."""
    config = UKGraphCalibrationConfig() if config is None else config
    inputs = population_slices(columns)
    problem_input = ArtifactInput(
        "problem", problem_producer, problem_artifact, PROBLEM_TYPE
    )
    dense_id = f"{prefix}.dense"
    checkpoint_id = f"{prefix}.size_checkpoint_import"
    imported_dense = (
        ()
        if checkpoint_identity is None
        else (ArtifactInput("imported_dense", checkpoint_id, "dense", RESULT_TYPE),)
    )
    nodes = [
        Node(
            id=dense_id,
            kernel=UKDenseSolveKernel.ref,
            population=base,
            inputs=inputs,
            params={
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "seed": config.seed,
            },
            artifact_inputs=(problem_input, *imported_dense),
            artifact_outputs=(
                ArtifactOutput("result", RESULT_TYPE),
                ArtifactOutput("solution", SOLUTION_TYPE),
            ),
        )
    ]
    if checkpoint_identity is not None:
        if config.dataset_households is None:
            raise ValueError(
                "Importing a size checkpoint requires an explicit dataset size."
            )
        nodes.insert(
            0,
            Node(
                id=checkpoint_id,
                kernel=UKSizeCheckpointImportKernel.ref,
                population=base,
                inputs=inputs,
                sources=checkpoint_sources,
                params={
                    "identity_json": canonical_json(checkpoint_identity).decode(),
                    "manifest_source": checkpoint_sources[0],
                    "arrays_source": checkpoint_sources[1],
                    "epochs": config.epochs,
                    "learning_rate": config.learning_rate,
                    "dense_seed": config.seed,
                    "seed": config.seed
                    if config.selection_seed is None
                    else config.selection_seed,
                    "households": config.dataset_households,
                },
                artifact_inputs=(problem_input,),
                artifact_outputs=(
                    ArtifactOutput("dense", RESULT_TYPE),
                    ArtifactOutput("search", RESULT_TYPE),
                    ArtifactOutput("selection", SIZE_SEARCH_TYPE),
                ),
            ),
        )
    solution_producer = result_producer = dense_id
    result_problem_producer = problem_producer
    size_producer = None
    final_base = base
    if config.dataset_households is not None:
        params = {
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "seed": config.seed
            if config.selection_seed is None
            else config.selection_seed,
            "households": config.dataset_households,
            "pi_hi": config.selection_pi_hi,
        }
        dense_input = ArtifactInput("dense", dense_id, "result", RESULT_TYPE)
        search_id, draw_id, refit_id = (
            f"{prefix}.{s}" for s in ("size_search", "size_draw", "size_refit")
        )
        nodes.append(
            Node(
                id=search_id,
                kernel=UKSizeSearchKernel.ref,
                population=base,
                inputs=inputs,
                params=params,
                artifact_inputs=(
                    problem_input,
                    dense_input,
                    *(
                        ()
                        if checkpoint_identity is None
                        else (
                            ArtifactInput(
                                "search", checkpoint_id, "search", RESULT_TYPE
                            ),
                            ArtifactInput(
                                "selection",
                                checkpoint_id,
                                "selection",
                                SIZE_SEARCH_TYPE,
                            ),
                        )
                    ),
                ),
                artifact_outputs=(
                    ArtifactOutput("result", RESULT_TYPE),
                    ArtifactOutput("selection", SIZE_SEARCH_TYPE),
                ),
            )
        )
        search_inputs = (
            problem_input,
            dense_input,
            ArtifactInput("search", search_id, "result", RESULT_TYPE),
            ArtifactInput("selection", search_id, "selection", SIZE_SEARCH_TYPE),
        )
        nodes.append(
            Node(
                id=draw_id,
                kernel=UKSizeDrawKernel.ref,
                population=base,
                inputs=inputs,
                params=params,
                artifact_inputs=search_inputs,
                artifact_outputs=(ArtifactOutput("draw", SIZE_DRAW_TYPE),),
            )
        )
        nodes.append(
            Node(
                id=refit_id,
                kernel=UKSizeRefitKernel.ref,
                population=base,
                inputs=inputs,
                params=params,
                artifact_inputs=(
                    *search_inputs,
                    ArtifactInput("draw", draw_id, "draw", SIZE_DRAW_TYPE),
                ),
                artifact_outputs=(
                    ArtifactOutput("result", RESULT_TYPE),
                    ArtifactOutput("problem", PROBLEM_TYPE),
                    ArtifactOutput("solution", SOLUTION_TYPE),
                    ArtifactOutput("refit_solution", SOLUTION_TYPE),
                    ArtifactOutput("size", SIZE_RECEIPT_TYPE),
                ),
            )
        )
        solution_producer = result_producer = result_problem_producer = (
            size_producer
        ) = refit_id
        final_base = f"{prefix}.selected"
        nodes.append(
            Node(
                id=final_base,
                kernel=UKSizeFilterKernel.ref,
                base=base,
                inputs=inputs,
                structural=StructuralDelta.FILTER,
                mass="free",
                artifact_inputs=(
                    problem_input,
                    ArtifactInput("solution", refit_id, "solution", SOLUTION_TYPE),
                ),
            )
        )
    calibrated = f"{prefix}.calibrated"
    nodes.append(
        Node(
            id=calibrated,
            kernel=UKInstallCalibrationKernel.ref,
            base=final_base,
            inputs=inputs,
            structural=StructuralDelta.REWEIGHT,
            mass="free",
            weights=WeightTransition("household", "calibrated", mass="free"),
            artifact_inputs=(
                problem_input,
                ArtifactInput("solution", solution_producer, "solution", SOLUTION_TYPE),
            ),
        )
    )
    return UKCalibrationNodes(
        tuple(nodes),
        calibrated,
        result_producer,
        result_problem_producer,
        solution_producer,
        size_producer,
        dense_id,
    )


def register_uk_calibration_kernels(registry: KernelRegistry) -> KernelRegistry:
    for kernel in (
        UKSizeCheckpointImportKernel,
        UKDenseSolveKernel,
        UKSizeSearchKernel,
        UKSizeDrawKernel,
        UKSizeRefitKernel,
        UKSizeFilterKernel,
        UKInstallCalibrationKernel,
    ):
        registry.register(kernel())
    return registry
