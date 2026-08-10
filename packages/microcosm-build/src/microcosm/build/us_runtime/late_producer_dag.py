"""Deterministic producer-input DAG primitives for the US late stage.

The graph is deliberately data-only.  Country-specific code declares which
stage produces each scoped input; this module validates those declarations,
derives a stable topological schedule, names cycles, and fences callback
execution when a required input remains unfilled.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

__all__ = [
    "ProducerInputColumn",
    "ProducerContract",
    "ProducerInput",
    "ProducerOutput",
    "ProducerSchedule",
    "derive_producer_schedule",
    "run_producer_when_ready",
]


def _nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value


@dataclass(frozen=True, order=True)
class ProducerInputColumn:
    """One physical column participating in an effective input alternative."""

    entity: str
    column: str
    value_kind: str = "non_null"

    def __post_init__(self) -> None:
        _nonempty(self.entity, label="ProducerInputColumn.entity")
        _nonempty(self.column, label="ProducerInputColumn.column")
        if self.value_kind not in {"non_null", "finite_numeric"}:
            raise ValueError(
                "ProducerInputColumn.value_kind must be 'non_null' or "
                f"'finite_numeric'; got {self.value_kind!r}."
            )


@dataclass(frozen=True, order=True)
class ProducerInput:
    """One scoped input and the stage expected to make it ready."""

    entity: str
    column: str
    required_scope: str
    producing_stage: str
    tolerated_absence_receipts: tuple[str, ...] = ()
    alternatives: tuple[tuple[ProducerInputColumn, ...], ...] = ()

    def __post_init__(self) -> None:
        for label, value in (
            ("ProducerInput.entity", self.entity),
            ("ProducerInput.column", self.column),
            ("ProducerInput.required_scope", self.required_scope),
            ("ProducerInput.producing_stage", self.producing_stage),
        ):
            _nonempty(value, label=label)
        receipts = tuple(self.tolerated_absence_receipts)
        if any(not isinstance(item, str) or not item.strip() for item in receipts):
            raise ValueError(
                "ProducerInput.tolerated_absence_receipts must contain only "
                "non-empty strings."
            )
        if len(set(receipts)) != len(receipts):
            raise ValueError(
                "ProducerInput.tolerated_absence_receipts contains duplicates."
            )
        object.__setattr__(self, "tolerated_absence_receipts", tuple(sorted(receipts)))
        alternatives = tuple(tuple(option) for option in self.alternatives)
        if not alternatives:
            alternatives = ((ProducerInputColumn(self.entity, self.column),),)
        if any(not option for option in alternatives) or any(
            not isinstance(item, ProducerInputColumn)
            for option in alternatives
            for item in option
        ):
            raise TypeError(
                "ProducerInput.alternatives require nonempty tuples of "
                "ProducerInputColumn values."
            )
        canonical_alternatives = tuple(
            sorted(
                {tuple(sorted(set(option))) for option in alternatives},
                key=lambda option: tuple(
                    (item.entity, item.column, item.value_kind) for item in option
                ),
            )
        )
        object.__setattr__(self, "alternatives", canonical_alternatives)


@dataclass(frozen=True, order=True)
class ProducerOutput:
    """One column and the row scope covered by its producing stage."""

    entity: str
    column: str
    coverage_scope: str

    def __post_init__(self) -> None:
        for label, value in (
            ("ProducerOutput.entity", self.entity),
            ("ProducerOutput.column", self.column),
            ("ProducerOutput.coverage_scope", self.coverage_scope),
        ):
            _nonempty(value, label=label)


@dataclass(frozen=True)
class ProducerContract:
    """One executable producer and its complete declared graph surface."""

    name: str
    kind: str
    inputs: tuple[ProducerInput, ...]
    outputs: tuple[ProducerOutput, ...]

    def __post_init__(self) -> None:
        _nonempty(self.name, label="ProducerContract.name")
        _nonempty(self.kind, label="ProducerContract.kind")
        inputs = tuple(self.inputs)
        outputs = tuple(self.outputs)
        if any(not isinstance(item, ProducerInput) for item in inputs):
            raise TypeError("ProducerContract.inputs require ProducerInput values.")
        if any(not isinstance(item, ProducerOutput) for item in outputs):
            raise TypeError("ProducerContract.outputs require ProducerOutput values.")
        if len(set(inputs)) != len(inputs):
            raise ValueError(f"Late producer {self.name!r} repeats an input.")
        if len(set(outputs)) != len(outputs):
            raise ValueError(f"Late producer {self.name!r} repeats an output.")
        object.__setattr__(self, "inputs", tuple(sorted(inputs)))
        object.__setattr__(self, "outputs", tuple(sorted(outputs)))


@dataclass(frozen=True)
class ProducerSchedule:
    """Canonical topological waves and their byte-stable identity."""

    order: tuple[str, ...]
    waves: tuple[tuple[str, ...], ...]
    edges: tuple[tuple[str, str], ...]
    canonical_json: bytes
    sha256: str


def _contract_payload(contract: ProducerContract) -> dict[str, object]:
    return {
        "name": contract.name,
        "kind": contract.kind,
        "inputs": [
            {
                "entity": item.entity,
                "column": item.column,
                "required_scope": item.required_scope,
                "producing_stage": item.producing_stage,
                "tolerated_absence_receipts": list(item.tolerated_absence_receipts),
                "alternatives": [
                    [
                        {
                            "entity": column.entity,
                            "column": column.column,
                            "value_kind": column.value_kind,
                        }
                        for column in alternative
                    ]
                    for alternative in item.alternatives
                ],
            }
            for item in contract.inputs
        ],
        "outputs": [
            {
                "entity": item.entity,
                "column": item.column,
                "coverage_scope": item.coverage_scope,
            }
            for item in contract.outputs
        ],
    }


def _named_cycle(
    adjacency: Mapping[str, set[str]],
    remaining: set[str],
) -> tuple[str, ...]:
    """Return the first deterministic DFS cycle from a Kahn residual."""

    state: dict[str, int] = {}
    stack: list[str] = []
    stack_positions: dict[str, int] = {}

    def visit(node: str) -> tuple[str, ...] | None:
        state[node] = 1
        stack_positions[node] = len(stack)
        stack.append(node)
        for child in sorted(adjacency[node] & remaining):
            if state.get(child, 0) == 0:
                found = visit(child)
                if found is not None:
                    return found
            elif state.get(child) == 1:
                start = stack_positions[child]
                return (*stack[start:], child)
        stack.pop()
        stack_positions.pop(node)
        state[node] = 2
        return None

    for node in sorted(remaining):
        if state.get(node, 0) == 0:
            found = visit(node)
            if found is not None:
                return found
    raise AssertionError("A nonempty Kahn residual did not contain a cycle.")


def derive_producer_schedule(
    registry: Mapping[str, ProducerContract],
    *,
    external_stages: tuple[str, ...] = (),
) -> ProducerSchedule:
    """Validate declarations and derive deterministic topological waves."""

    if not isinstance(registry, Mapping):
        raise TypeError("Late producer registry must be a mapping.")
    external = tuple(external_stages)
    if any(not isinstance(stage, str) or not stage.strip() for stage in external):
        raise ValueError("External producer stages must be non-empty strings.")
    if len(set(external)) != len(external):
        raise ValueError("External producer stages contain duplicates.")
    contracts = dict(registry)
    invalid_values = sorted(
        name
        for name, contract in contracts.items()
        if not isinstance(contract, ProducerContract)
    )
    if invalid_values:
        raise TypeError(
            "Late producer registry values must be ProducerContract instances: "
            f"{invalid_values}."
        )
    mismatched = sorted(
        (name, contract.name)
        for name, contract in contracts.items()
        if name != contract.name
    )
    if mismatched:
        raise ValueError(
            f"Late producer registry keys must equal contract names: {mismatched}."
        )
    overlap = sorted(set(contracts) & set(external))
    if overlap:
        raise ValueError(
            f"Late producer stages cannot also be external stages: {overlap}."
        )

    adjacency = {name: set() for name in contracts}
    indegree = {name: 0 for name in contracts}
    edges: set[tuple[str, str]] = set()
    unknown: list[tuple[str, str]] = []
    missing_outputs: list[tuple[str, str, str]] = []
    for consumer_name in sorted(contracts):
        contract = contracts[consumer_name]
        for item in contract.inputs:
            producer_name = item.producing_stage
            if producer_name in external:
                continue
            producer = contracts.get(producer_name)
            if producer is None:
                unknown.append((consumer_name, producer_name))
                continue
            if not any(
                output.entity == item.entity and output.column == item.column
                for output in producer.outputs
            ):
                missing_outputs.append(
                    (consumer_name, producer_name, f"{item.entity}.{item.column}")
                )
                continue
            edge = (producer_name, consumer_name)
            if edge not in edges:
                edges.add(edge)
                adjacency[producer_name].add(consumer_name)
                indegree[consumer_name] += 1
    if unknown or missing_outputs:
        raise ValueError(
            "Late producer dependency declarations are invalid; "
            f"unknown_stages={unknown}, missing_outputs={missing_outputs}."
        )

    remaining = set(contracts)
    waves: list[tuple[str, ...]] = []
    while remaining:
        ready = tuple(sorted(name for name in remaining if indegree[name] == 0))
        if not ready:
            cycle = _named_cycle(adjacency, remaining)
            raise RuntimeError(
                "Late producer dependency cycle: " + " -> ".join(cycle) + "."
            )
        waves.append(ready)
        for producer_name in ready:
            remaining.remove(producer_name)
        for producer_name in ready:
            for consumer_name in adjacency[producer_name]:
                indegree[consumer_name] -= 1

    order = tuple(name for wave in waves for name in wave)
    sorted_edges = tuple(sorted(edges))
    payload = {
        "schema_version": 1,
        "external_stages": sorted(external),
        "contracts": [_contract_payload(contracts[name]) for name in sorted(contracts)],
        "edges": [list(edge) for edge in sorted_edges],
        "waves": [list(wave) for wave in waves],
        "order": list(order),
    }
    canonical_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return ProducerSchedule(
        order=order,
        waves=tuple(waves),
        edges=sorted_edges,
        canonical_json=canonical_json,
        sha256=hashlib.sha256(canonical_json).hexdigest(),
    )


def _absence_receipt_matches(
    receipt_id: str,
    receipt: object,
    requirement: ProducerInput,
    rows: int,
) -> bool:
    return bool(
        isinstance(receipt, Mapping)
        and receipt.get("receipt_id") == receipt_id
        and receipt.get("status") == "declared_absence"
        and receipt.get("entity") == requirement.entity
        and receipt.get("column") == requirement.column
        and receipt.get("required_scope") == requirement.required_scope
        and receipt.get("rows") == rows
    )


def run_producer_when_ready[ResultT](
    contract: ProducerContract,
    callback: Callable[[], ResultT],
    *,
    unfilled_rows: Mapping[ProducerInput, int],
    invalid_rows: Mapping[ProducerInput, int],
    absence_receipts: Mapping[str, Mapping[str, object]],
) -> ResultT:
    """Fence one producer callback on exact input or absence evidence.

    Missing values and invalid values are separate states.  Only missing
    values can be authorized by a declared-absence receipt; nonnumeric or
    nonfinite values always fail closed.
    """

    if not isinstance(contract, ProducerContract):
        raise TypeError("Producer readiness requires a ProducerContract.")
    if not callable(callback):
        raise TypeError(f"Late producer {contract.name!r} callback is not callable.")
    declared_inputs = set(contract.inputs)

    def sort_key(item: ProducerInput) -> tuple[str, str, str, str]:
        return (
            item.entity,
            item.column,
            item.required_scope,
            item.producing_stage,
        )

    missing_unfilled = sorted(declared_inputs - set(unfilled_rows), key=sort_key)
    unexpected_unfilled = sorted(set(unfilled_rows) - declared_inputs, key=sort_key)
    if missing_unfilled or unexpected_unfilled:
        raise ValueError(
            f"Late producer {contract.name!r} unfilled-row readiness surface "
            f"drifted; missing={missing_unfilled}, extra={unexpected_unfilled}."
        )
    missing_invalid = sorted(declared_inputs - set(invalid_rows), key=sort_key)
    unexpected_invalid = sorted(set(invalid_rows) - declared_inputs, key=sort_key)
    if missing_invalid or unexpected_invalid:
        raise ValueError(
            f"Late producer {contract.name!r} invalid-value readiness surface "
            f"drifted; missing={missing_invalid}, extra={unexpected_invalid}."
        )
    failures: list[str] = []
    for requirement in contract.inputs:
        rows = unfilled_rows[requirement]
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise ValueError(
                f"Late producer {contract.name!r} unfilled count for "
                f"{requirement.entity}.{requirement.column} must be a "
                f"non-negative integer; got {rows!r}."
            )
        invalid = invalid_rows[requirement]
        if isinstance(invalid, bool) or not isinstance(invalid, int) or invalid < 0:
            raise ValueError(
                f"Late producer {contract.name!r} invalid count for "
                f"{requirement.entity}.{requirement.column} must be a "
                f"non-negative integer; got {invalid!r}."
            )
        if invalid:
            failures.append(
                f"{requirement.entity}.{requirement.column}: {invalid} invalid "
                f"value(s) in required scope {requirement.required_scope!r}; "
                f"declared producing stage is {requirement.producing_stage!r}; "
                "declared absence cannot authorize invalid values."
            )
        if rows == 0:
            continue
        tolerated = any(
            _absence_receipt_matches(
                receipt_id,
                absence_receipts.get(receipt_id),
                requirement,
                rows,
            )
            for receipt_id in requirement.tolerated_absence_receipts
        )
        if tolerated:
            continue
        allowed = list(requirement.tolerated_absence_receipts)
        failures.append(
            f"{requirement.entity}.{requirement.column}: {rows} unfilled row(s) "
            f"in required scope {requirement.required_scope!r}; declared "
            f"producing stage is {requirement.producing_stage!r}; tolerated "
            f"absence receipts={allowed}."
        )
    if failures:
        raise ValueError(
            f"Late producer {contract.name!r} refused unfilled input(s):\n  "
            + "\n  ".join(failures)
        )
    return callback()
