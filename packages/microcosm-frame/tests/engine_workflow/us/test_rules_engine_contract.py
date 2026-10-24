"""PolicyEngine-US cases for the shared RulesEngine contract."""

# ruff: noqa: F403, F405
from test_support.microcosm_frame.rules_engine_contract import *


@pytest.fixture(params=[pytest.param(POLICYENGINE_US_CASE, id="policyengine-us")])
def case(request) -> AdapterCase:
    return request.param
