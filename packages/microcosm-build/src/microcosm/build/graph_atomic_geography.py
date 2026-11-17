"""Typed graph nodes for the shared atomic-area geography contract.

Countries supply declarations and normalized source bytes. Import, assignment,
derivation and the integrity gate remain independently inspectable operations.
These nodes do not change weights, rows, memberships or observed columns.
"""

from __future__ import annotations

import json

import pandas as pd

from microcosm.build import atomic_geography as geography
from microcosm.graph import (
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Numeric,
    SeedSource,
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
        from microcosm.frame import materialize as frame_materialize
        from microcosm.graph import store as graph_store

        # load_source_bytes takes a content store, whose frame encoding writes
        # through frame_materialize (microcosm#931).
        return source_hash(
            type(self),
            geography,
            canonical_json,
            keyed_uniform,
            load_source_bytes,
            graph_store,
            frame_materialize,
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


def register_atomic_geography_kernels(registry: KernelRegistry) -> None:
    for kernel in (
        AtomicSupportImportKernel(),
        AtomicAssignKernel(),
        AtomicDeriveKernel(),
        AtomicGeographyGateKernel(),
    ):
        registry.register(kernel)
