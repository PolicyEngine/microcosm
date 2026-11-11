"""Engine-free Axiom cases for the shared RulesEngine contract."""

# ruff: noqa: F403, F405
from test_support.microcosm_frame.rules_engine_contract import *


@pytest.fixture(
    params=[
        pytest.param(
            AXIOM_CASE,
            id="axiom",
            marks=pytest.mark.skipif(
                not _AXIOM_DENSE,
                reason="axiom_rules_engine dense extension is not installed",
            ),
        )
    ]
)
def case(request) -> AdapterCase:
    return request.param
