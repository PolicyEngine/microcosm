"""Kernel-protocol contracts of the frozen interface (amendment 13).

A kernel that claims bounded numeric movement declares the bound; a bitwise
kernel declares none; the context hands readers their inputs' declared
tolerances; and the two new declaration fields round-trip through JSON.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.graph import (
    Capabilities,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    Node,
    Numeric,
    Owned,
    SourceRef,
    StructuralDelta,
    Tolerance,
    graph_from_json,
    graph_to_json,
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
