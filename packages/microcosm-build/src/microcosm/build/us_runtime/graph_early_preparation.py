"""Measured hours use the registered source boundary and an explicit gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from importlib.resources import files

import pandas as pd

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
    KernelRole,
    Node,
    Numeric,
    Owned,
    Slice,
    StructuralDelta,
)
from microcosm.graph.canonical import canonical_json

from .graph_context import US_FRAME_CONTEXT_TYPE, _decode, _mass_records, _row_identity
from .graph_sources import _source_implementation_hash
from .hours_worked import (
    US_HOURS_WORKED_OUTPUT_COLUMNS,
    US_HOURS_WORKED_POOL_OUTPUT_COLUMNS,
    US_HOURS_WORKED_REQUIRED_SOURCE_COLUMNS,
    us_hours_worked_gate_from_summary,
)
from .multispine_pool import (
    multispine_hours_worked_boundary_columns,
    observe_multispine_hours_person,
    validate_multispine_hours_person_boundary,
)

HOURS_PHASE = "early_preparation.hours"
US_HOURS_EVIDENCE_TYPE = ArtifactType("microcosm.us.measured_hours_evidence", 1)


class _HoursKernel(KernelBase):
    def implementation_hash(self) -> str:
        digest = hashlib.sha256(_source_implementation_hash().encode())
        digest.update(
            files("microcosm.build.us").joinpath("source_stages.json").read_bytes()
        )
        digest.update(files("microcosm.graph").joinpath("canonical.py").read_bytes())
        return digest.hexdigest()


def _params(context: KernelContext) -> None:
    if context.params != {"phase": HOURS_PHASE}:
        raise ValueError("Measured hours requires its registered production phase.")


def _person_context(context: KernelContext) -> tuple[pd.DataFrame, dict[str, object]]:
    artifact = context.artifacts["frame_context"]
    if artifact.type != US_FRAME_CONTEXT_TYPE:
        raise ValueError("Measured hours requires the full typed frame context.")
    document = _decode(artifact.payload)
    _mass_records(document["mass_log"])
    if set(context.tables) != {"person"}:
        raise ValueError("Measured hours requires only its declared person view.")
    person = context.tables["person"]
    declared = document["entities"]["person"]
    if any(
        declared[key] != value for key, value in _row_identity(person, "person").items()
    ):
        raise ValueError("Measured hours person identity does not match its context.")
    if set(person.columns) - set(declared["columns"]):
        raise ValueError("Measured hours person view includes unknown columns.")
    sources = document["weight_sources"]
    if len(sources) != 1:
        raise ValueError("Measured hours graph requires one explicit weight source.")
    if context.weights["person"].kind.value != next(iter(sources.values())):
        raise ValueError("Measured hours canonical weight kind differs from context.")
    ordered = [column for column in declared["columns"] if column in person]
    return person.loc[:, ordered], document


class USHoursBoundaryKernel(_HoursKernel):
    ref = "us.production.hours_boundary@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.FILTER,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def run(self, context: KernelContext) -> KernelResult:
        _params(context)
        person, document = _person_context(context)
        receipt = validate_multispine_hours_person_boundary(
            person, metadata=document["metadata"]
        )
        return KernelResult(
            keep=pd.Series(
                True,
                index=pd.Index(person.person_id, name="person_id"),
                dtype=bool,
            ),
            artifacts={"frame_context": context.artifacts["frame_context"].payload},
            receipt={"source_boundary": receipt},
        )


class USHoursDeriveKernel(_HoursKernel):
    ref = "us.production.measured_hours@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def run(self, context: KernelContext) -> KernelResult:
        _params(context)
        person, document = _person_context(context)
        required = {
            column
            for column in (
                *US_HOURS_WORKED_REQUIRED_SOURCE_COLUMNS,
                *US_HOURS_WORKED_OUTPUT_COLUMNS,
            )
            if column in document["entities"]["person"]["columns"]
        }
        if required - set(person.columns):
            raise ValueError(
                "Measured hours declaration omits available raw or incumbent inputs."
            )
        produced, receipt = observe_multispine_hours_person(
            person,
            weights=context.weights["person"],
            metadata=document["metadata"],
            pool_rows={
                entity: value["rows"] for entity, value in document["entities"].items()
            },
        )
        expected = set(US_HOURS_WORKED_POOL_OUTPUT_COLUMNS)
        if {
            (value.entity, value.column, value.dtype) for value in context.node.outputs
        } != {("person", column, "float64") for column in expected}:
            raise ValueError(
                "Measured hours must own exactly its two float64 input leaves."
            )
        columns = {}
        for owned in context.node.outputs:
            values = produced[owned.column]
            if str(values.dtype) != owned.dtype:
                raise ValueError(
                    "Measured hours output dtype differs from its declaration."
                )
            columns[("person", owned.column)] = pd.Series(
                values.array,
                index=pd.Index(produced.person_id, name="person_id"),
            )
        # Person identity was verified against the full context. The registered
        # source boundary verifies selected person structure and the ID merge.
        # Preserve untouched descriptors; never encode the narrow view as a
        # replacement for the full population's column inventory or history.
        declared = document["entities"]["person"]["columns"]
        for column in produced.columns:
            if column in expected and column not in declared:
                declared.append(column)
        payload = canonical_json(document)
        evidence = canonical_json(
            {
                "schema_version": 1,
                "frame_context_sha256": hashlib.sha256(payload).hexdigest(),
                "source_preparation": receipt,
            }
        )
        return KernelResult(
            columns=columns,
            artifacts={"frame_context": payload, "hours_evidence": evidence},
            receipt={
                "source_preparation": receipt,
                "acceptance": "pending_mandatory_gate",
            },
        )


class USHoursGateKernel(_HoursKernel):
    ref = "us.production.measured_hours_gate@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        role=KernelRole.GATE,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def run(self, context: KernelContext) -> KernelResult:
        _params(context)
        artifact = context.artifacts["hours_evidence"]
        if artifact.type != US_HOURS_EVIDENCE_TYPE:
            raise ValueError("Measured hours gate requires typed producer evidence.")
        document = json.loads(artifact.payload)
        if (
            canonical_json(document) != artifact.payload
            or set(document)
            != {"schema_version", "frame_context_sha256", "source_preparation"}
            or type(document["schema_version"]) is not int
            or document["schema_version"] != 1
        ):
            raise ValueError(
                "Measured hours gate received malformed producer evidence."
            )
        frame_context = context.artifacts["frame_context"]
        if frame_context.type != US_FRAME_CONTEXT_TYPE:
            raise ValueError("Measured hours gate requires the full typed context.")
        if frame_context.producer_key != artifact.producer_key:
            raise ValueError(
                "Measured hours evidence and context require the same producer."
            )
        _decode(frame_context.payload)
        if (
            document["frame_context_sha256"]
            != hashlib.sha256(frame_context.payload).hexdigest()
        ):
            raise ValueError(
                "Measured hours evidence belongs to a different frame context."
            )
        receipt = document["source_preparation"]
        if (
            receipt["operator_order"] != ["with_us_hours_worked_inputs"]
            or receipt["phase"] != "pre_clone"
            or len(receipt["suboperators"]) != 1
            or receipt["suboperators"][0]["operator"] != "with_us_hours_worked_inputs"
            or receipt["suboperators"][0]["family"] != "hours_worked"
        ):
            raise ValueError(
                "Measured hours evidence has a different source operation."
            )
        recorded = receipt["suboperators"][0]["kernel_receipt"][
            "hours_worked_signal_gate"
        ]
        gate = us_hours_worked_gate_from_summary(recorded["details"])
        if recorded != {
            "name": gate.name,
            "passed": gate.passed,
            "failures": list(gate.failures),
            "details": gate.details,
        }:
            raise ValueError(
                "Measured hours evidence disagrees with the original gate checks."
            )
        if not gate.passed:
            # The exact direct-path message remains in exceptional gate evidence.
            # Withhold the typed context so every dependent stays unreached.
            raise ValueError(
                "Pool pre-clone hours-worked signal gate failed:\n  "
                + "\n  ".join(gate.failures)
            )
        return KernelResult(
            artifacts={"frame_context": frame_context.payload},
            receipt={"outcome": "pass", "source_preparation": receipt},
        )


def us_hours_nodes(
    columns: Sequence[Owned], *, base: str, context_producer: str
) -> tuple[Node, ...]:
    """Declare real measured hours; consumers use the gate's context output."""
    inventory = {(item.entity, item.column): item for item in columns}
    if len(inventory) != len(columns):
        raise ValueError("Measured hours input declarations repeat coordinates.")
    boundary_columns = multispine_hours_worked_boundary_columns()
    for column in boundary_columns:
        if ("person", column) not in inventory:
            raise ValueError(f"Measured hours boundary input {column!r} is absent.")
    output_names = tuple(sorted(US_HOURS_WORKED_POOL_OUTPUT_COLUMNS))
    for column in output_names:
        incumbent = inventory.get(("person", column))
        if incumbent is not None and incumbent.dtype != "float64":
            raise ValueError(f"Unsupported incumbent measured hours dtype: {column}.")
    actual = tuple(
        column
        for column in (
            *US_HOURS_WORKED_REQUIRED_SOURCE_COLUMNS,
            *US_HOURS_WORKED_OUTPUT_COLUMNS,
        )
        if ("person", column) in inventory
    )

    def slices(person_columns: Sequence[str]) -> tuple[Slice, ...]:
        # The executor supplies structural person ID/memberships implicitly.
        return (Slice("person", tuple(dict.fromkeys(person_columns))),)

    boundary = f"{HOURS_PHASE}.boundary"
    context_output = (ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),)
    return (
        Node(
            boundary,
            USHoursBoundaryKernel.ref,
            base=base,
            structural=StructuralDelta.FILTER,
            inputs=slices(boundary_columns),
            params={"phase": HOURS_PHASE},
            artifact_inputs=(
                ArtifactInput(
                    "frame_context",
                    context_producer,
                    "frame_context",
                    US_FRAME_CONTEXT_TYPE,
                ),
            ),
            artifact_outputs=context_output,
        ),
        Node(
            HOURS_PHASE,
            USHoursDeriveKernel.ref,
            population=boundary,
            inputs=slices((*boundary_columns, *actual)),
            outputs=tuple(
                Owned("person", name, "float64", rewrite=("person", name) in inventory)
                for name in output_names
            ),
            params={"phase": HOURS_PHASE},
            artifact_inputs=(
                ArtifactInput(
                    "frame_context", boundary, "frame_context", US_FRAME_CONTEXT_TYPE
                ),
            ),
            artifact_outputs=(
                *context_output,
                ArtifactOutput("hours_evidence", US_HOURS_EVIDENCE_TYPE),
            ),
        ),
        Node(
            f"{HOURS_PHASE}.gate",
            USHoursGateKernel.ref,
            population=boundary,
            params={"phase": HOURS_PHASE},
            artifact_inputs=(
                ArtifactInput(
                    "frame_context", HOURS_PHASE, "frame_context", US_FRAME_CONTEXT_TYPE
                ),
                ArtifactInput(
                    "hours_evidence",
                    HOURS_PHASE,
                    "hours_evidence",
                    US_HOURS_EVIDENCE_TYPE,
                ),
            ),
            artifact_outputs=context_output,
        ),
    )


def register_us_hours_kernels(registry: KernelRegistry) -> None:
    for kernel in (USHoursBoundaryKernel(), USHoursDeriveKernel(), USHoursGateKernel()):
        registry.register(kernel)
