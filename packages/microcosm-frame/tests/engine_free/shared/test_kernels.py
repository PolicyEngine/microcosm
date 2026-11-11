"""Tests split from packages/microcosm-frame/tests/test_kernels.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_frame.kernels import *


def test_stub_rules_kernel_has_byte_parity_shape_ids_and_capabilities() -> None:
    frame = _stub_frame()
    expected_engine = _StubRulesEngine()
    expected = expected_engine.materialize(frame, _STUB_VARIABLES, period=2025)
    engine = _StubRulesEngine()
    kernel = SimulateRulesKernel("stub", engine)
    node = Node(
        id="simulate",
        kernel=kernel.ref,
        inputs=(
            Slice("person", ("earnings",)),
            Slice("household", ("housing_cost",)),
        ),
        outputs=(
            Owned("person", "net_earnings", "float64"),
            Owned("household", "housing_allowance", "float64"),
        ),
        params={
            "engine_ref": "stub",
            "variables": _STUB_VARIABLES,
            "period": 2025,
        },
    )

    context = _context(frame, node, weighted_entities=("person", "household"))
    assert set(context.tables) == {"person", "household"}
    result = kernel.run(context)

    assert engine.materialize_calls == 1
    assert set(result.columns) == {
        ("person", "net_earnings"),
        ("household", "housing_allowance"),
    }
    _assert_byte_parity(frame, engine, expected, result.columns, _STUB_VARIABLES)
    assert isinstance(kernel, Kernel)
    assert RulesKernel is SimulateRulesKernel
    assert kernel.capabilities.determinism is Determinism.DETERMINISTIC
    assert kernel.capabilities.numeric is Numeric.BITWISE
    assert kernel.capabilities.seed_source is SeedSource.NONE
    assert kernel.capabilities.structural is StructuralDelta.NONE
    assert kernel.capabilities.consumes_se is False
    assert kernel.capabilities.dependencies == ()
    assert kernel.implementation_hash() == source_hash(
        SimulateRulesKernel,
        _StubRulesEngine,
        frame_bundle_module,
        frame_rules_module,
        frame_schema_module,
    )
    assert result.receipt == {
        "engine_ref": "stub",
        "period": 2025,
        "variables": _STUB_VARIABLES,
        "output_rows": (
            ("net_earnings", "person", 4),
            ("housing_allowance", "household", 2),
        ),
    }
