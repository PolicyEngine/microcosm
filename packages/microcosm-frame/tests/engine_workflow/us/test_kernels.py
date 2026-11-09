"""Tests split from packages/microcosm-frame/tests/test_kernels.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_frame.kernels import *


def test_policyengine_us_twenty_household_kernel_matches_direct_materialize() -> None:
    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    frame = _twenty_household_us_frame()
    engine = PolicyEngineUSEngine()
    variables = ("employment_income", "household_net_income")
    expected = engine.materialize(frame, variables, period=2024)
    kernel = SimulateRulesKernel(
        "policyengine-us", engine, dependencies=("policyengine-us",)
    )
    node = Node(
        id="simulate-us",
        kernel=kernel.ref,
        inputs=(
            Slice("person", ("age", "employment_income_before_lsr")),
            Slice("household", ("state_fips",)),
        ),
        outputs=(
            Owned("person", "employment_income", "float32"),
            Owned("household", "household_net_income", "float32"),
        ),
        params={
            "engine_ref": "policyengine-us",
            "variables": variables,
            "period": 2024,
        },
    )

    context = _context(frame, node, weighted_entities=("person", "household"))
    assert set(context.tables) == {"person", "household"}
    result = kernel.run(context)

    _assert_byte_parity(frame, engine, expected, result.columns, variables)
    assert kernel.capabilities.dependencies == ("policyengine-us",)
    assert len(kernel.implementation_hash()) == 64
