"""The ordered-problem solve kernel: ``calibrate.ordered_adam@1``.

``calibrate.adam@1`` (:mod:`microcosm.calibrate.kernels`) takes its targets as
node parameters, measured on columns of the one calibrated entity. A target
surface compiled inside the graph cannot reach it that way: its rows are facts
resolved at run time, and a person- or family-level fact reaches the household
weights only after collapsing through the person memberships. This kernel
consumes the shared portable form instead, a ``microcosm.calibrate.ordered-
problem`` artifact (:mod:`microcosm.calibrate.artifacts`), and otherwise runs
the very same solve:

1. It decodes the problem and rebuilds its target set with
   :meth:`~microcosm.calibrate.artifacts.OrderedProblem.to_target_set`, whose
   rows are the exact compiled contributions, bound to the problem's ordered
   entity axis.
2. It builds the same minimal one-entity frame ``calibrate.adam@1`` builds
   (:func:`microcosm.calibrate.kernels._frame_from_context`, imported, not
   copied) and refuses unless that frame's id axis and starting weights are
   the ones the problem was compiled against.
3. It calls :func:`microcosm.calibrate.calibrate` with exactly the keyword
   arguments ``calibrate.adam@1`` passes (``method="adam"``, ``seed=0``,
   the node's ``epochs``, ``learning_rate``, ``mass`` and
   ``max_weight_ratio``).

It returns the calibrated weights, which the executor installs as the
declared ``WeightTransition``, plus two typed artifacts bound to the problem's
SHA-256: an ``ordered-solution`` and a ``calibration-result``. A consumer
(diagnostics, gates) rebuilds the completed result from those bytes without
solving again.

When both kernels compile the same :class:`~microcosm.calibrate.TargetSet`
against the same frame through :mod:`microcosm.calibrate.matrix`, the two
solves see the same constraint matrix and return byte-identical weights;
that is the differential test this module ships with.

The design-weight cap is the executor's, not this kernel's: with
``max_weight_ratio`` set the node must also declare ``weight_anchor="design"``,
and the executor measures the installed weights against the design weights
captured at ``CREATE`` (``microcosm.graph.population``), records
``realized_max_weight_ratio`` in the receipt, and rejects the node on a
violation. The solver's own cap is relative to the weights the problem starts
from; the two agree exactly when those are still the design weights.
"""

from __future__ import annotations

import hashlib
import math
from types import MappingProxyType
from zipfile import BadZipFile

import numpy as np

import microcosm.calibrate.artifacts as artifacts_module
import microcosm.calibrate.kernels as adam_kernels_module
from microcosm.calibrate import calibrate
from microcosm.calibrate.artifacts import (
    PROBLEM_TYPE,
    RESULT_TYPE,
    SOLUTION_TYPE,
    decode_problem,
    encode_calibration_result,
    encode_solution,
)
from microcosm.calibrate.kernels import CalibrateAdamKernel, _frame_from_context
from microcosm.graph import (
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelResult,
    Numeric,
    SeedSource,
    StructuralDelta,
    source_hash,
)
from microcosm.graph.canonical import canonical_json

__all__ = [
    "CALIBRATE_ORDERED_ADAM",
    "ORDERED_PROBLEM_INPUT",
    "OUTPUT_TYPES",
    "RESULT_OUTPUT",
    "SOLUTION_OUTPUT",
    "CalibrateOrderedAdamKernel",
]

#: The artifact alias the kernel reads its ordered problem under.
ORDERED_PROBLEM_INPUT = "problem"
#: The artifact names the kernel returns.
SOLUTION_OUTPUT = "solution"
RESULT_OUTPUT = "result"
#: The typed artifacts a ``calibrate.ordered_adam@1`` node declares, exactly.
OUTPUT_TYPES = MappingProxyType(
    {SOLUTION_OUTPUT: SOLUTION_TYPE, RESULT_OUTPUT: RESULT_TYPE}
)

_REF = "calibrate.ordered_adam@1"

_PARAMS = frozenset(
    {
        "epochs",
        "learning_rate",
        "mass",
        "max_weight_ratio",
        # Which weights the cap is measured against. The executor enforces the
        # anchor (charter D3); the kernel refuses a cap that names none.
        "weight_anchor",
    }
)
_REQUIRED_PARAMS = frozenset({"epochs", "learning_rate", "mass"})


def _integer_param(context: KernelContext, name: str) -> int:
    value = context.params.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{_REF} parameter {name!r} must be an integer.")
    return value


def _numeric_param(context: KernelContext, name: str) -> int | float:
    value = context.params.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{_REF} parameter {name!r} must be numeric.")
    return value


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


class CalibrateOrderedAdamKernel(KernelBase):
    """Solve a decoded ordered problem with the public Adam calibrator."""

    ref = _REF
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        # The node installs calibrated weights: a new population version
        # every later node reads (charter D1; interface amendment 6).
        structural=StructuralDelta.REWEIGHT,
        # The ordered problem carries no standard errors to the solver, and
        # the public loss would not read them if it did.
        consumes_se=False,
        dependencies=("numpy", "pandas", "scipy", "torch"),
    )

    def implementation_hash(self) -> str:
        """Bind this adapter, the artifact codec and everything adam@1 binds.

        ``calibrate.adam@1``'s own hash covers the public solver, its
        diagnostics, the matrix compiler, the target model and the frame
        modules; this kernel runs the same solve, so any edit there re-keys
        both. The adapter half covers this module and the artifact codec it
        decodes and encodes through.
        """

        return hashlib.sha256(
            canonical_json(
                {
                    "solver": CalibrateAdamKernel().implementation_hash(),
                    "adapter": source_hash(
                        type(self),
                        artifacts_module,
                        adam_kernels_module,
                        dependencies=self.capabilities.dependencies,
                    ),
                }
            )
        ).hexdigest()

    def run(self, context: KernelContext) -> KernelResult:
        """Solve the node's ordered problem and return its typed weights."""

        unknown = sorted(set(context.params) - _PARAMS)
        if unknown:
            raise ValueError(f"{_REF} received unknown parameter(s): {unknown}.")
        missing = sorted(_REQUIRED_PARAMS - set(context.params))
        if missing:
            raise ValueError(f"{_REF} is missing required parameter(s): {missing}.")
        declared = {
            output.name: output.type for output in context.node.artifact_outputs
        }
        if declared != dict(OUTPUT_TYPES):
            raise ValueError(
                f"{_REF} nodes declare exactly the typed outputs {dict(OUTPUT_TYPES)!r}; "
                f"this node declares {declared!r}."
            )

        transition = context.node.weights
        if transition is None or transition.to_kind != "calibrated":
            raise ValueError(
                f"{_REF} requires a WeightTransition whose to_kind is 'calibrated'."
            )
        entity = transition.entity

        mass = context.params.get("mass")
        if not isinstance(mass, str):
            raise TypeError(f"{_REF} parameter 'mass' must be a string.")
        if mass != transition.mass:
            raise ValueError(
                f"{_REF} mass parameter must match the declared weight "
                f"transition: parameter {mass!r}, transition {transition.mass!r}."
            )

        max_weight_ratio = context.params.get("max_weight_ratio")
        if max_weight_ratio is not None and (
            isinstance(max_weight_ratio, bool)
            or not isinstance(max_weight_ratio, int | float)
        ):
            raise TypeError(
                f"{_REF} parameter 'max_weight_ratio' must be numeric or None."
            )
        weight_anchor = context.params.get("weight_anchor")
        if max_weight_ratio is not None and weight_anchor != "design":
            # The executor refuses the same node; failing here names the
            # kernel's own contract: a cap is a statement about design weights.
            raise ValueError(
                f"{_REF} max_weight_ratio requires weight_anchor='design', got "
                f"{weight_anchor!r}."
            )
        if weight_anchor is not None and weight_anchor != "design":
            raise ValueError(
                f"{_REF} weight_anchor must be 'design' when declared, got "
                f"{weight_anchor!r}."
            )
        epochs = _integer_param(context, "epochs")
        learning_rate = _numeric_param(context, "learning_rate")

        if ORDERED_PROBLEM_INPUT not in context.artifacts:
            raise ValueError(
                f"{_REF} requires an ordered-problem artifact input named "
                f"{ORDERED_PROBLEM_INPUT!r}."
            )
        value = context.artifacts[ORDERED_PROBLEM_INPUT]
        if value.type != PROBLEM_TYPE:
            raise ValueError(
                f"{_REF} artifact {ORDERED_PROBLEM_INPUT!r} must be "
                f"{PROBLEM_TYPE.name} v{PROBLEM_TYPE.schema_version}."
            )
        try:
            ordered = decode_problem(value.payload)
        except BadZipFile as error:
            raise ValueError(
                f"{_REF} artifact {ORDERED_PROBLEM_INPUT!r} is not an ordered "
                "problem archive."
            ) from error
        problem = ordered.problem
        if problem.weight_entity != entity:
            raise ValueError(
                f"{_REF} problem calibrates {problem.weight_entity!r}, but the "
                f"node's weight transition is on {entity!r}."
            )
        if problem.skipped:
            raise ValueError(
                f"{_REF} refuses a problem with {len(problem.skipped)} skipped "
                "target(s); every declared target must enter the matrix."
            )
        if entity not in context.tables:
            raise ValueError(
                f"{_REF} needs the {entity!r} table: declare one data-column "
                f"slice on {entity!r} so the executor projects its id axis."
            )

        frame = _frame_from_context(context, entity)
        id_column = frame.schema.entity_id_column(entity)
        axis = tuple(
            int(item) if isinstance(item, np.integer) else item
            for item in frame.table(entity)[id_column].tolist()
        )
        if axis != ordered.entity_ids:
            raise ValueError(
                f"{_REF} problem was compiled against a different ordered "
                f"{entity!r} axis than the node's base population."
            )
        weights = frame.weights_for(entity)
        if weights.kind != problem.initial_weights.kind or not np.array_equal(
            weights.values, problem.initial_weights.values
        ):
            raise ValueError(
                f"{_REF} problem was compiled against different {entity!r} "
                "starting weights than the node's base population carries."
            )

        result = calibrate(
            frame,
            ordered.to_target_set(),
            weight_entity=entity,
            method="adam",
            max_weight_ratio=max_weight_ratio,
            epochs=epochs,
            learning_rate=learning_rate,
            mass=mass,
            seed=0,
        )
        calibrated = result.frame.weights_for(entity)
        solution = encode_solution(
            calibrated.values,
            entity_ids=ordered.entity_ids,
            problem_sha256=ordered.sha256,
        )
        result_payload = encode_calibration_result(
            result,
            entity_ids=ordered.entity_ids,
            problem_sha256=ordered.sha256,
        )
        return KernelResult(
            weights=calibrated,
            artifacts={SOLUTION_OUTPUT: solution, RESULT_OUTPUT: result_payload},
            receipt={
                "problem_sha256": ordered.sha256,
                "solution_sha256": hashlib.sha256(solution).hexdigest(),
                "result_sha256": hashlib.sha256(result_payload).hexdigest(),
                "n_targets": int(problem.n_targets),
                "n_records": int(problem.n_weights),
                "initial_loss": _finite_or_none(result.initial_loss),
                "final_loss": _finite_or_none(result.final_loss),
                # The solver's ratio is against the problem's starting weights;
                # the executor adds ``realized_max_weight_ratio`` against the
                # CREATE design anchor, which is the one the cap means.
                "ratio_to_starting_weights_max": _finite_or_none(
                    result.realized_max_weight_ratio
                ),
            },
        )


CALIBRATE_ORDERED_ADAM = CalibrateOrderedAdamKernel()
