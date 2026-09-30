"""Typed graph nodes for the shared atomic-area geography contract.

Countries supply declarations and normalized source bytes. Import, assignment,
derivation and the integrity gate remain independently inspectable operations.
These nodes do not change weights, rows, memberships or observed columns.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pandas as pd

from microcosm.build import atomic_geography as geography
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
    Numeric,
    Owned,
    SeedSource,
    Slice,
    source_hash,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import load_source_bytes
from microcosm.graph.kernel import KernelRole
from microcosm.graph.randomness import keyed_uniform

ATOMIC_SUPPORT_TYPE = ArtifactType("microcosm.geography.atomic_support_npz", 1)
ATOMIC_GEOGRAPHY_VALIDATION_TYPE = ArtifactType(
    "microcosm.geography.validation_result", 1
)
_DEPENDENCIES = ("numpy", "pandas")


class _GeographyKernel(KernelBase):
    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            geography,
            canonical_json,
            keyed_uniform,
            load_source_bytes,
            dependencies=_DEPENDENCIES,
        )


class AtomicSupportImportKernel(_GeographyKernel):
    ref = "geography.support_import@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        if set(context.params) != {"source", "system"} or set(context.sources) != {
            context.params["source"]
        }:
            raise ValueError(
                "Atomic geography import requires its sole declared source."
            )
        payload = load_source_bytes(
            "raw-bytes-v1", context.sources[context.params["source"]]
        )
        support = geography.decode_atomic_support(payload)
        if support.metadata["system"] != context.params["system"]:
            raise ValueError("Atomic geography source belongs to a different system.")
        return KernelResult(
            artifacts={"support": payload},
            receipt={
                "support_sha256": support.sha256,
                "areas": len(support.arrays["area"]),
                "metadata": support.metadata,
            },
        )


def _inputs(context):
    if set(context.params) != {"definition", "stream"}:
        raise ValueError(
            "Atomic geography requires its canonical definition and stream."
        )
    spec = geography.validate_assignment_spec(json.loads(context.params["definition"]))
    if (
        canonical_json(spec).decode() != context.params["definition"]
        or tuple(spec["stream"]) != context.params["stream"]
    ):
        raise ValueError("Atomic geography definition or stream differs.")
    if set(context.artifacts) != {s["id"] for s in spec["systems"]}:
        raise ValueError(
            "Atomic geography requires exactly its declared support artifacts."
        )
    supports = {}
    for name, value in context.artifacts.items():
        if value.type != ATOMIC_SUPPORT_TYPE:
            raise ValueError("Atomic geography artifact type differs.")
        supports[name] = geography.decode_atomic_support(value.payload)
    return context.tables["household"], spec, supports


def _result(context, frame, receipt):
    if {(o.entity, o.column, o.dtype) for o in context.node.outputs} != {
        ("household", c, "string") for c in frame
    }:
        raise ValueError("Atomic geography owned outputs differ from the declaration.")
    index = pd.Index(context.tables["household"]["household_id"], name="household_id")
    return KernelResult(
        columns={
            ("household", c): pd.Series(frame[c].array, index=index) for c in frame
        },
        receipt=receipt,
    )


class AtomicAssignKernel(_GeographyKernel):
    ref = "geography.assign_atomic@1"
    capabilities = Capabilities(
        determinism=Determinism.SEEDED,
        seed_source=SeedSource.KEYED,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        households, spec, supports = _inputs(context)
        output = geography.assign_atomic(households, spec, supports)
        return _result(
            context,
            output,
            {
                "scope": "atomic_area_assignment",
                "households": len(output),
                "support_sha256": {k: v.sha256 for k, v in supports.items()},
            },
        )


class AtomicDeriveKernel(_GeographyKernel):
    ref = "geography.derive@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        households, spec, supports = _inputs(context)
        output = geography.derive_geography(households, spec, supports)
        return _result(
            context,
            output,
            {
                "scope": "atomic_area_functional_lookup",
                "layers": {s["id"]: s["layers"] for s in spec["systems"]},
            },
        )


class AtomicGeographyGateKernel(_GeographyKernel):
    ref = "geography.gate@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        role=KernelRole.GATE,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        households, spec, supports = _inputs(context)
        outputs = context.node.artifact_outputs
        expected = (ArtifactOutput("validation", ATOMIC_GEOGRAPHY_VALIDATION_TYPE),)
        if outputs not in ((), expected):
            raise ValueError(
                "Atomic geography validation artifact declaration differs."
            )
        receipt = geography.validate_geography(households, spec, supports)
        return KernelResult(
            receipt=receipt,
            artifacts={"validation": canonical_json(receipt)} if outputs else {},
        )


def atomic_geography_nodes(
    spec: Mapping,
    columns: Sequence[Owned],
    *,
    base: str,
    prefix: str = "geography",
    emit_validation_artifact: bool = False,
) -> tuple[Node, ...]:
    """Append shared nodes to an existing population, refusing column rewrites.

    SourceRef declarations use each system's ``source`` and ``raw-bytes-v1``.
    ``identity`` must be stable across the country's sampling rungs; this builder
    cannot establish that property merely from a column name.

    The optional typed gate result orders downstream consumers. Its bytes alone
    do not authenticate their receiving population or source ancestry.
    """
    if type(emit_validation_artifact) is not bool:
        raise ValueError("Atomic geography validation artifact flag must be boolean.")
    spec = geography.validate_assignment_spec(spec)
    inventory = {(o.entity, o.column): o for o in columns}
    if len(inventory) != len(columns):
        raise ValueError("Atomic geography input inventory repeats columns.")
    inputs = set(spec["identity"])
    for system in spec["systems"]:
        inputs.update(system["selector"])
        inputs.update(c["input"] for c in system["constraints"])
        if system["observed_area"]:
            inputs.add(system["observed_area"])
    assignment = tuple(spec["outputs"].values())
    layers = tuple(
        sorted({layer["output"] for s in spec["systems"] for layer in s["layers"]})
    )
    if any(("household", c) not in inventory for c in inputs):
        raise ValueError("Atomic geography is missing a declared household input.")
    if any(("household", c) in inventory for c in (*assignment, *layers)):
        raise ValueError("Atomic geography refuses to overwrite existing columns.")
    imports, artifacts = [], []
    for i, system in enumerate(spec["systems"]):
        name = f"{prefix}.support.{i}"
        imports.append(
            Node(
                id=name,
                kernel=AtomicSupportImportKernel.ref,
                population=base,
                sources=(system["source"],),
                params={"source": system["source"], "system": system["id"]},
                artifact_outputs=(ArtifactOutput("support", ATOMIC_SUPPORT_TYPE),),
            )
        )
        artifacts.append(
            ArtifactInput(system["id"], name, "support", ATOMIC_SUPPORT_TYPE)
        )

    def node(suffix, kernel, read, write=()):
        return Node(
            id=f"{prefix}.{suffix}",
            kernel=kernel.ref,
            population=base,
            inputs=(Slice("household", tuple(sorted(read))),),
            outputs=tuple(Owned("household", c, "string") for c in write),
            params={
                "definition": canonical_json(spec).decode(),
                "stream": tuple(spec["stream"]),
            },
            artifact_inputs=tuple(artifacts),
            artifact_outputs=(
                (ArtifactOutput("validation", ATOMIC_GEOGRAPHY_VALIDATION_TYPE),)
                if suffix == "gate" and emit_validation_artifact
                else ()
            ),
        )

    return (
        *imports,
        node("assign", AtomicAssignKernel, inputs, assignment),
        node("derive", AtomicDeriveKernel, inputs | set(assignment), layers),
        node("gate", AtomicGeographyGateKernel, inputs | set(assignment) | set(layers)),
    )


def register_atomic_geography_kernels(registry: KernelRegistry) -> None:
    for kernel in (
        AtomicSupportImportKernel(),
        AtomicAssignKernel(),
        AtomicDeriveKernel(),
        AtomicGeographyGateKernel(),
    ):
        registry.register(kernel)
