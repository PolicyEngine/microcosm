"""Kernel-protocol contracts of the frozen interface (amendments 13, 17, 19, 20).

A kernel that claims bounded numeric movement declares the bound; a bitwise
kernel declares none; the context hands readers their inputs' declared
tolerances and numeric scopes; declared typed artifacts reach a consumer as
verified immutable bytes; a ``KEYED`` kernel declares that its draws come from
stable coordinates rather than from a position in the executor's generator;
and the new declaration fields round-trip through JSON.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pandas as pd
import pytest

from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    ArtifactValue,
    Capabilities,
    Determinism,
    Graph,
    GraphError,
    KernelBase,
    KernelContext,
    KernelRegistry,
    Node,
    Numeric,
    NumericScope,
    Owned,
    SeedSource,
    Slice,
    SourceRef,
    StructuralDelta,
    Tolerance,
    graph_from_json,
    graph_to_json,
    keyed_uniform,
)


def test_tolerance_must_allow_some_movement_and_stay_finite() -> None:
    assert Tolerance(rtol=1e-9).ulps == 0
    assert Tolerance(ulps=2) == Tolerance(0.0, 0.0, 2)
    with pytest.raises(ValueError, match="allow some movement"):
        Tolerance()
    with pytest.raises(ValueError, match="non-negative and finite"):
        Tolerance(rtol=-1e-9)
    with pytest.raises(ValueError, match="non-negative and finite"):
        Tolerance(atol=float("inf"))
    with pytest.raises(ValueError, match="non-negative and finite"):
        Tolerance(rtol=float("nan"))
    with pytest.raises(ValueError, match="must be an integer"):
        Tolerance(ulps=1.5)  # type: ignore[arg-type]


def test_tolerance_bound_kernels_declare_a_bound_and_bitwise_kernels_none() -> None:
    bounded = Capabilities(
        determinism=Determinism.SEEDED,
        numeric=Numeric.TOLERANCE_BOUND,
        tolerance=Tolerance(rtol=1e-6),
    )
    assert bounded.tolerance == Tolerance(rtol=1e-6)
    with pytest.raises(ValueError, match="must declare its Tolerance"):
        Capabilities(determinism=Determinism.SEEDED, numeric=Numeric.TOLERANCE_BOUND)
    with pytest.raises(ValueError, match="bitwise kernel declares no Tolerance"):
        Capabilities(determinism=Determinism.DETERMINISTIC, tolerance=Tolerance(ulps=1))
    with pytest.raises(TypeError, match="must be a Tolerance or None"):
        Capabilities(
            determinism=Determinism.DETERMINISTIC,
            numeric=Numeric.TOLERANCE_BOUND,
            tolerance=1e-6,  # type: ignore[arg-type]
        )


def test_context_carries_declared_tolerances_and_defaults_to_none() -> None:
    node = Node("gate", "gate.check@1")
    context = KernelContext(
        node=node,
        tables={},
        weights={},
        strata=pd.Series([], dtype=object, name="stratum"),
        params={},
        rng=np.random.default_rng(0),
    )
    assert dict(context.tolerances) == {}
    carried = KernelContext(
        node=node,
        tables={},
        weights={},
        strata=pd.Series([], dtype=object, name="stratum"),
        params={},
        rng=np.random.default_rng(0),
        tolerances={
            ("person", "income"): Tolerance(rtol=1e-6),
            ("person", "age"): None,
        },
    )
    assert carried.tolerances[("person", "income")] == Tolerance(rtol=1e-6)
    assert carried.tolerances[("person", "age")] is None


def test_entrants_and_mass_partition_round_trip_through_canonical_json() -> None:
    source = SourceRef("survey", "frame-h5")
    create = Node(
        "survey",
        "source.frame@1",
        sources=("survey",),
        structural=StructuralDelta.CREATE,
        outputs=(Owned("person", "age", "int64"), Owned("person", "period", "int64")),
    )
    cohort = Node(
        "cohort",
        "enter.immigrants@1",
        base="survey",
        structural=StructuralDelta.EXPAND,
        mass="declared",
        entrants=True,
    )
    graph = Graph(
        "toy", (source,), (create, cohort), mass_partition=("person", "period")
    )
    text = graph_to_json(graph)
    assert '"entrants":true' in text and '"mass_partition":["person","period"]' in text
    assert graph_from_json(text) == graph
    # A declaration without either field serializes exactly as it did before
    # amendments 11 and 12, so every pinned graph JSON still matches.
    plain = Graph("toy", (source,), (create,))
    plain_text = graph_to_json(plain)
    assert "entrants" not in plain_text and "mass_partition" not in plain_text
    assert graph_from_json(plain_text) == plain


def test_capabilities_reject_look_alike_fields_and_registration_needs_the_real_thing() -> (
    None
):
    """A string spelling an enum member is not the member (review of #851, finding 5)."""
    with pytest.raises(TypeError, match="Capabilities.numeric must be a Numeric"):
        Capabilities(determinism=Determinism.DETERMINISTIC, numeric="tolerance_bound")  # type: ignore[arg-type]
    with pytest.raises(
        TypeError, match="Capabilities.determinism must be a Determinism"
    ):
        Capabilities(determinism="deterministic")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="consumes_se must be a boolean"):
        Capabilities(determinism=Determinism.DETERMINISTIC, consumes_se=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="dependencies must be a tuple"):
        Capabilities(determinism=Determinism.DETERMINISTIC, dependencies=["numpy"])  # type: ignore[arg-type]

    class LookAlike:
        determinism = Determinism.DETERMINISTIC
        numeric = "tolerance_bound"
        seed_source = "none"
        structural = "none"
        role = "compute"
        consumes_se = False
        dependencies = ()
        tolerance = None

    class Impostor(KernelBase):
        ref = "impostor@1"
        capabilities = LookAlike()  # type: ignore[assignment]

        def run(self, context):  # pragma: no cover - never reached
            raise AssertionError

    with pytest.raises(TypeError, match="must carry a Capabilities instance"):
        KernelRegistry().register(Impostor())


def test_platform_bitwise_forbids_a_tolerance_like_bitwise() -> None:
    """Amendment 16: platform-bitwise kernels declare no per-cell tolerance."""
    Capabilities(Determinism.SEEDED, numeric=Numeric.PLATFORM_BITWISE)
    with pytest.raises(ValueError, match="bitwise kernel declares no Tolerance"):
        Capabilities(
            Determinism.SEEDED,
            numeric=Numeric.PLATFORM_BITWISE,
            tolerance=Tolerance(rtol=1e-6),
        )


def test_numeric_scope_validates_class_tolerance_and_platform() -> None:
    """Amendment 17: a scope carries exactly the fields its class permits."""
    assert NumericScope().numeric is Numeric.BITWISE
    bound = Tolerance(rtol=1e-6)
    NumericScope(numeric=Numeric.TOLERANCE_BOUND, tolerance=bound)
    NumericScope(
        numeric=Numeric.TOLERANCE_BOUND, tolerance=bound, platform="arm64/darwin/py3.13"
    )
    NumericScope(numeric=Numeric.PLATFORM_BITWISE, platform="arm64/darwin/py3.13")
    with pytest.raises(ValueError, match="must carry its Tolerance"):
        NumericScope(numeric=Numeric.TOLERANCE_BOUND)
    with pytest.raises(ValueError, match="carries no Tolerance"):
        NumericScope(numeric=Numeric.PLATFORM_BITWISE, platform="x", tolerance=bound)
    with pytest.raises(ValueError, match="must name its platform"):
        NumericScope(numeric=Numeric.PLATFORM_BITWISE)
    with pytest.raises(ValueError, match="every platform"):
        NumericScope(platform="arm64/darwin/py3.13")


def test_context_numerics_default_empty_and_carry_scopes() -> None:
    """Amendment 17: ``numerics`` defaults empty and rides at the end of the context."""
    fields = [f.name for f in dataclasses.fields(KernelContext)]
    assert fields[-2:] == ["tolerances", "numerics"]
    scope = NumericScope(
        numeric=Numeric.PLATFORM_BITWISE, platform="arm64/darwin/py3.13"
    )
    context = KernelContext(
        node=Node("gate", "gate.check@1"),
        tables={},
        weights={},
        strata=pd.Series(dtype="int64"),
        params={},
        rng=np.random.default_rng(0),
        tolerances={("person", "income"): None},
        numerics={("person", "income"): scope},
    )
    assert context.numerics[("person", "income")] is scope
    assert context.tolerances[("person", "income")] is None


def test_artifact_type_names_a_versioned_payload_contract() -> None:
    """Amendment 19: a nominal type is a non-empty name and a positive version."""
    assert ArtifactType("qrf.forest", 1).schema_version == 1
    with pytest.raises(GraphError, match="ArtifactType.name"):
        ArtifactType("", 1)
    with pytest.raises(GraphError, match="positive integer"):
        ArtifactType("qrf.forest", 0)
    with pytest.raises(GraphError, match="positive integer"):
        ArtifactType("qrf.forest", True)  # type: ignore[arg-type]
    with pytest.raises(GraphError, match="positive integer"):
        ArtifactType("qrf.forest", 1.0)  # type: ignore[arg-type]


def test_artifact_declarations_require_a_real_artifact_type() -> None:
    """Amendment 19: a look-alike mapping is not an ArtifactType."""
    type_ = ArtifactType("qrf.forest", 2)
    assert ArtifactOutput("forest", type_).type is type_
    binding = ArtifactInput("donor", "fit", "forest", type_)
    assert (binding.producer, binding.artifact) == ("fit", "forest")
    with pytest.raises(GraphError, match="ArtifactOutput.name"):
        ArtifactOutput("", type_)
    with pytest.raises(GraphError, match="ArtifactOutput.type"):
        ArtifactOutput("forest", {"name": "qrf.forest", "schema_version": 2})  # type: ignore[arg-type]
    for empty in ("name", "producer", "artifact"):
        values = {
            "name": "donor",
            "producer": "fit",
            "artifact": "forest",
            **{empty: ""},
        }
        with pytest.raises(GraphError, match=f"ArtifactInput.{empty}"):
            ArtifactInput(type=type_, **values)
    with pytest.raises(GraphError, match="ArtifactInput.type"):
        ArtifactInput("donor", "fit", "forest", "qrf.forest@2")  # type: ignore[arg-type]


def test_node_artifact_declarations_are_tuples_with_unique_aliases() -> None:
    """Amendment 19: the two new declaration fields validate at construction."""
    type_ = ArtifactType("qrf.forest", 1)
    node = Node(
        "draw",
        "fit.draw@1",
        population="survey",
        artifact_inputs=(ArtifactInput("donor", "fit", "forest", type_),),
        artifact_outputs=(ArtifactOutput("diagnostics", ArtifactType("table", 1)),),
    )
    assert node.artifact_inputs[0].producer == "fit"
    assert node.artifact_outputs[0].name == "diagnostics"
    with pytest.raises(GraphError, match="artifact_inputs must be a tuple"):
        Node(
            "draw", "fit.draw@1", artifact_inputs=[ArtifactInput("d", "f", "o", type_)]
        )  # type: ignore[arg-type]
    with pytest.raises(GraphError, match="artifact_outputs must be a tuple"):
        Node(
            "draw",
            "fit.draw@1",
            artifact_outputs=(ArtifactInput("d", "f", "o", type_),),
        )  # type: ignore[arg-type]
    with pytest.raises(GraphError, match="duplicate names in artifact_inputs"):
        Node(
            "draw",
            "fit.draw@1",
            artifact_inputs=(
                ArtifactInput("donor", "fit", "forest", type_),
                ArtifactInput("donor", "other", "forest", type_),
            ),
        )
    with pytest.raises(GraphError, match="duplicate names in artifact_outputs"):
        Node(
            "fit",
            "fit.train@1",
            artifact_outputs=(
                ArtifactOutput("forest", type_),
                ArtifactOutput("forest", ArtifactType("other", 1)),
            ),
        )


def test_a_node_declaring_no_artifacts_projects_exactly_as_before() -> None:
    """Amendment 19: empty declarations leave the canonical projection alone."""
    plain = Node("draw", "fit.draw@1", population="survey")
    projection = plain.normative()
    assert "artifact_inputs" not in projection
    assert "artifact_outputs" not in projection
    declared = dataclasses.replace(
        plain,
        artifact_outputs=(ArtifactOutput("forest", ArtifactType("qrf.forest", 1)),),
    )
    assert declared.normative()["artifact_outputs"] == declared.artifact_outputs
    assert set(declared.normative()) - set(projection) == {"artifact_outputs"}


def test_artifact_value_validates_its_payload_type_and_identities() -> None:
    """Amendment 19: verified immutable bytes with typed producer provenance."""
    type_ = ArtifactType("qrf.forest", 1)
    scope = NumericScope()
    key = "a" * 64
    value = ArtifactValue(b"payload", type_, key, "b" * 64, scope)
    assert value.payload == b"payload" and value.numerics is scope
    with pytest.raises(TypeError, match="immutable bytes"):
        ArtifactValue(bytearray(b"payload"), type_, key, "b" * 64, scope)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ArtifactType and NumericScope"):
        ArtifactValue(b"payload", type_, key, "b" * 64, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ArtifactType and NumericScope"):
        ArtifactValue(b"payload", "qrf.forest@1", key, "b" * 64, scope)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ArtifactValue.key"):
        ArtifactValue(b"payload", type_, "a" * 63, "b" * 64, scope)
    with pytest.raises(ValueError, match="ArtifactValue.key"):
        ArtifactValue(b"payload", type_, "A" * 64, "b" * 64, scope)
    with pytest.raises(ValueError, match="ArtifactValue.producer_key"):
        ArtifactValue(b"payload", type_, key, "not-a-key", scope)


def test_context_artifacts_default_empty_and_are_immutable() -> None:
    """Amendment 19: ``artifacts`` rides before the amendment-13/17 pair."""
    fields = [f.name for f in dataclasses.fields(KernelContext)]
    assert fields[-2:] == ["tolerances", "numerics"]
    assert fields[fields.index("artifacts") + 1] == "tolerances"
    node = Node("draw", "fit.draw@1")
    bare = KernelContext(
        node=node,
        tables={},
        weights={},
        strata=pd.Series(dtype="int64"),
        params={},
        rng=np.random.default_rng(0),
    )
    assert dict(bare.artifacts) == {}
    value = ArtifactValue(
        b"payload", ArtifactType("qrf.forest", 1), "a" * 64, "b" * 64, NumericScope()
    )
    carried = KernelContext(
        node=node,
        tables={},
        weights={},
        strata=pd.Series(dtype="int64"),
        params={},
        rng=np.random.default_rng(0),
        artifacts={"donor": value},
    )
    assert carried.artifacts["donor"] is value
    with pytest.raises(TypeError):
        carried.artifacts["other"] = value  # type: ignore[index]
    for bad in ({"": value}, {"donor": b"payload"}, {1: value}):
        with pytest.raises(TypeError, match="non-empty aliases to ArtifactValue"):
            KernelContext(
                node=node,
                tables={},
                weights={},
                strata=pd.Series(dtype="int64"),
                params={},
                rng=np.random.default_rng(0),
                artifacts=bad,  # type: ignore[arg-type]
            )


def test_artifact_declarations_round_trip_through_canonical_json() -> None:
    """Amendment 19: declared artifact edges survive serialization exactly."""
    source = SourceRef("survey", "frame-h5")
    create = Node(
        "survey",
        "source.frame@1",
        sources=("survey",),
        structural=StructuralDelta.CREATE,
        outputs=(Owned("person", "age", "int64"),),
    )
    type_ = ArtifactType("qrf.forest", 3)
    fit = Node(
        "fit",
        "fit.train@1",
        population="survey",
        inputs=(Slice("person", ("age",)),),
        artifact_outputs=(ArtifactOutput("forest", type_),),
    )
    draw = Node(
        "draw",
        "fit.draw@1",
        population="survey",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "income", "float64"),),
        artifact_inputs=(ArtifactInput("donor", "fit", "forest", type_),),
    )
    graph = Graph("toy", (source,), (create, fit, draw))
    text = graph_to_json(graph)
    assert '"artifact_outputs":' in text and '"artifact_inputs":' in text
    assert '"schema_version":3' in text
    assert graph_from_json(text) == graph
    # A graph declaring no artifacts serializes exactly as it did before
    # amendment 19, so every pinned graph JSON still matches.
    plain = Graph("toy", (source,), (create,))
    plain_text = graph_to_json(plain)
    assert "artifact_inputs" not in plain_text
    assert "artifact_outputs" not in plain_text
    assert graph_from_json(plain_text) == plain


def test_seed_source_keyed_is_an_additive_declaration() -> None:
    """Amendment 20: a third seed source, and the two others are untouched.

    ``KEYED`` says the kernel's draws come from normative stream parameters and
    stable coordinates rather than from ``KernelContext.rng`` or a literal
    ``seed`` param. The member is additive: every existing kernel keeps the
    value it declares, so no existing node key moves (A5/A7).
    """
    assert SeedSource.KEYED.value == "keyed"
    assert SeedSource.EXECUTOR.value == "executor"
    assert SeedSource.PARAM.value == "param"
    assert SeedSource.NONE.value == "none"
    assert set(SeedSource) == {
        SeedSource.EXECUTOR,
        SeedSource.PARAM,
        SeedSource.KEYED,
        SeedSource.NONE,
    }

    keyed = Capabilities(determinism=Determinism.SEEDED, seed_source=SeedSource.KEYED)
    assert keyed.seed_source is SeedSource.KEYED
    with pytest.raises(
        TypeError, match="Capabilities.seed_source must be a SeedSource"
    ):
        Capabilities(determinism=Determinism.SEEDED, seed_source="keyed")  # type: ignore[arg-type]


def test_keyed_is_part_of_a_node_identity_and_survives_the_manifest() -> None:
    """The declaration is contract, so it keys the node and round-trips.

    A kernel that reads the same inputs through a keyed stream is not the
    kernel that reads them through the executor's generator; the capability
    projection separates them, and a receipt spells the member back.
    """
    from microcosm.graph.keys import _capabilities_projection
    from microcosm.graph.manifest import _capability_contract_fields

    base = Capabilities(determinism=Determinism.SEEDED, seed_source=SeedSource.EXECUTOR)
    keyed = dataclasses.replace(base, seed_source=SeedSource.KEYED)
    projection = _capabilities_projection(keyed)
    assert projection["seed_source"] == "keyed"
    assert projection != _capabilities_projection(base)
    assert _capability_contract_fields(projection)[2] is SeedSource.KEYED


def test_a_keyed_kernel_draws_from_coordinates_not_from_the_context_rng() -> None:
    """The context still offers ``rng``; a keyed kernel simply does not spend it.

    Both halves of that sentence are asserted against a toy kernel body, which
    is as far as this can go: the executor has no ``seed_source`` branch, so
    there is no production path that treats a KEYED node differently and none
    is claimed here. What the toy body pins is the shape a keyed kernel has —
    it leaves the generator where it found it, and two contexts whose
    generators sit 512 variates apart hand the same coordinates the same draws.
    """
    node = Node("impute", "toy.keyed@1", params={"experiment": "amendment-20"})

    def draw(context: KernelContext) -> np.ndarray:
        return keyed_uniform(
            stream=(
                "sha256-u53-v1",
                str(context.params["experiment"]),
                0,
                0,
            ),
            keys=[(person, "wages") for person in (11, 12, 13)],
        )

    def context_at(position: int) -> KernelContext:
        rng = np.random.default_rng(7)
        rng.random(position)
        return KernelContext(
            node=node,
            tables={},
            weights={},
            strata=pd.Series(dtype="int64"),
            params=node.params,
            rng=rng,
        )

    early, late = context_at(0), context_at(512)
    assert early.rng.bit_generator.state != late.rng.bit_generator.state
    # "Does not spend it" is the half a drawing body could violate silently, so
    # assert it rather than leave it to the closure's own restraint.
    untouched = copy.deepcopy(early.rng.bit_generator.state)
    assert draw(early).tobytes() == draw(late).tobytes()
    assert early.rng.bit_generator.state == untouched
