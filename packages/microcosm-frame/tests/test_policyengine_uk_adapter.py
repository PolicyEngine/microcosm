"""PolicyEngine-UK adapter import/protocol behavior."""

from __future__ import annotations

import pytest

from microcosm.frame import RulesEngine
from microcosm.frame.adapters.policyengine_uk import (
    UK_SCHEMA,
    PolicyEngineUKEngine,
)


def test_policyengine_uk_adapter_satisfies_rules_protocol_without_importing_engine() -> (
    None
):
    adapter = PolicyEngineUKEngine()

    assert isinstance(adapter, RulesEngine)
    assert adapter.country == "uk"
    assert adapter.entity_schema() == UK_SCHEMA


def test_policyengine_uk_adapter_export_side_is_not_implemented() -> None:
    adapter = PolicyEngineUKEngine()

    with pytest.raises(NotImplementedError, match="write_uk_national_frame"):
        adapter.write_dataset(object(), "unused.h5", period=2023)  # type: ignore[arg-type]
