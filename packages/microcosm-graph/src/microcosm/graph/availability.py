"""Executor-owned exceptional outcomes, with no fabricated data products."""

from collections.abc import Mapping

from .decl import StructuralDelta
from .kernel import Capabilities, KernelRole

EXECUTION_SCHEMA = "microcosm.graph.execution.v1"


def has_execution(receipt: Mapping[str, object]) -> bool:
    """Leave pre-existing free-form execution diagnostics uninterpreted."""
    execution = receipt.get("execution")
    return (
        isinstance(execution, Mapping) and execution.get("schema") == EXECUTION_SCHEMA
    )


def execution_state(receipt: Mapping[str, object]) -> str | None:
    execution = receipt.get("execution")
    state = execution.get("state") if has_execution(receipt) else None
    return state if isinstance(state, str) else None


def unavailable_artifacts(
    receipt: Mapping[str, object], outputs: Mapping[str, object]
) -> frozenset[str]:
    return (
        frozenset(outputs)
        if execution_state(receipt) in {"gate_exception", "unreached"}
        else frozenset()
    )


def validate_execution(
    receipt: Mapping[str, object],
    capabilities: Capabilities,
    outputs: Mapping[str, object],
    opaque: Mapping[str, object],
    *,
    has_products: bool,
) -> None:
    """Validate exceptional receipt shape at cache and manifest boundaries."""
    if not has_execution(receipt):
        return
    execution = receipt["execution"]
    if not isinstance(execution, Mapping):
        raise ValueError("Executor execution metadata must be a mapping.")
    state = execution.get("state")
    if state == "gate_exception":
        evidence = receipt.get("evidence")
        missing = execution.get("unavailable_artifacts")
        if (
            set(execution) != {"schema", "state", "unavailable_artifacts"}
            or capabilities.role is not KernelRole.GATE
            or capabilities.structural is not StructuralDelta.NONE
            or receipt.get("outcome") != "fail"
            or not outputs
            or not isinstance(missing, list | tuple)
            or tuple(missing) != tuple(sorted(outputs))
            or opaque
            or not isinstance(evidence, Mapping)
            or not isinstance(evidence.get("exception_type"), str)
            or not isinstance(evidence.get("message"), str)
        ):
            raise ValueError("Invalid gate exception or unavailable artifact evidence.")
    elif state == "unreached":
        blockers = execution.get("blocked_by")
        if (
            set(execution) != {"schema", "state", "blocked_by"}
            or receipt.get("outcome") != "unreached"
            or not isinstance(blockers, Mapping)
            or not blockers
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(key, str)
                or len(key) != 64
                or any(char not in "0123456789abcdef" for char in key)
                for name, key in blockers.items()
            )
            or has_products
            or opaque
        ):
            raise ValueError("Invalid unreached blocker evidence or invented products.")
        if (
            capabilities.role is KernelRole.RELEASE
            and receipt.get("tier") != "evidence"
        ):
            raise ValueError("An unreached release must remain evidence-tier.")
    else:
        raise ValueError(f"Unknown executor execution state {state!r}.")
