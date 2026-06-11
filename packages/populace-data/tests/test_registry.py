"""The published-dataset registry: keys, specs, and safe registration."""

from __future__ import annotations

import pytest

from populace.data import REGISTRY, DatasetSpec, available, register


def test_us_2024_is_registered_with_a_complete_spec() -> None:
    spec = REGISTRY[("us", 2024)]
    assert spec.country == "us"
    assert spec.year == 2024
    assert spec.hf_repo == "policyengine/populace-us"
    assert spec.filename == "populace_us_2024.h5"
    assert spec.engine_module == "policyengine_us.data"
    assert spec.engine_class == "USSingleYearDataset"
    assert spec.engine_package == "policyengine-us"
    assert spec.key == ("us", 2024)
    assert spec.hf_url == "hf://policyengine/populace-us/populace_us_2024.h5"


def test_available_lists_registered_keys_sorted() -> None:
    keys = available()
    assert ("us", 2024) in keys
    assert keys == sorted(keys)


def test_register_is_idempotent_for_the_same_spec() -> None:
    same = DatasetSpec(
        country="us",
        year=2024,
        hf_repo="policyengine/populace-us",
        filename="populace_us_2024.h5",
        engine_module="policyengine_us.data",
        engine_class="USSingleYearDataset",
        engine_package="policyengine-us",
    )
    # Re-registering the identical spec must not raise (re-import safety).
    assert register(same) is same


def test_register_refuses_to_shadow_a_key_with_a_different_spec() -> None:
    conflicting = DatasetSpec(
        country="us",
        year=2024,
        hf_repo="someone-else/populace-us",
        filename="populace_us_2024.h5",
        engine_module="policyengine_us.data",
        engine_class="USSingleYearDataset",
        engine_package="policyengine-us",
    )
    with pytest.raises(ValueError, match="already registered for"):
        register(conflicting)
