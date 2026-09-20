"""The published-dataset registry: keys, specs, and safe registration."""

from __future__ import annotations

import pytest

import microcosm.data.loader as loader_module
import microcosm.data.registry as registry_module
from microcosm.data import (
    DEFAULT_VARIANT,
    REGISTRY,
    DatasetSpec,
    available,
    available_variants,
    register,
)
from microcosm.data.release import LATEST_POINTER_PATH


def test_us_2024_is_registered_with_a_complete_spec() -> None:
    spec = REGISTRY[("us", 2024, DEFAULT_VARIANT)]
    assert spec.country == "us"
    assert spec.year == 2024
    assert spec.variant == DEFAULT_VARIANT
    assert spec.hf_repo == "policyengine/populace-us"
    assert spec.filename == "populace_us_2024.h5"
    assert spec.engine_module == "policyengine_us.data"
    assert spec.engine_class == "USSingleYearDataset"
    assert spec.engine_package == "policyengine-us"
    assert spec.key == ("us", 2024, DEFAULT_VARIANT)
    assert spec.hf_url == "hf://policyengine/populace-us/populace_us_2024.h5"


def test_uk_2023_compact_is_registered_with_a_complete_spec() -> None:
    spec = REGISTRY[("uk", 2023, DEFAULT_VARIANT)]
    assert spec.country == "uk"
    assert spec.year == 2023
    assert spec.variant == DEFAULT_VARIANT
    assert spec.hf_repo == "policyengine/populace-uk-private"
    assert spec.filename == "populace_uk_2023.h5"
    assert spec.engine_module == "policyengine_uk.data"
    assert spec.engine_class == "UKSingleYearDataset"
    assert spec.engine_package == "policyengine-uk"


def test_dataset_spec_default_pointer_path_matches_release_default() -> None:
    assert (
        DatasetSpec.__dataclass_fields__["pointer_path"].default == LATEST_POINTER_PATH
    )


def test_available_lists_default_variant_keys_only_sorted(monkeypatch) -> None:
    dense_only = DatasetSpec(
        country="zz",
        year=2099,
        variant="dense",
        hf_repo="policyengine/populace-zz",
        filename="populace_zz_2099_dense.h5",
        engine_module="policyengine_zz.data",
        engine_class="ZZSingleYearDataset",
        engine_package="policyengine-zz",
    )
    monkeypatch.setattr(
        loader_module,
        "REGISTRY",
        {**REGISTRY, dense_only.key: dense_only},
    )

    keys = available()
    assert ("us", 2024) in keys
    assert ("uk", 2025) in keys
    assert ("zz", 2099) not in keys
    assert keys == sorted(keys)


def test_available_variants_lists_registered_variant_keys_sorted() -> None:
    keys = available_variants()
    assert ("us", 2024, DEFAULT_VARIANT) in keys
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


def test_register_allows_distinct_variants_for_same_country_year(monkeypatch) -> None:
    monkeypatch.setattr(registry_module, "REGISTRY", dict(registry_module.REGISTRY))
    compact = DatasetSpec(
        country="zz",
        year=2099,
        variant="compact",
        hf_repo="policyengine/populace-zz",
        filename="populace_zz_2099.h5",
        engine_module="policyengine_zz.data",
        engine_class="ZZSingleYearDataset",
        engine_package="policyengine-zz",
    )
    local = DatasetSpec(
        country="zz",
        year=2099,
        variant="local",
        hf_repo="policyengine/populace-zz",
        filename="populace_zz_2099_local.h5",
        engine_module="policyengine_zz.data",
        engine_class="ZZSingleYearDataset",
        engine_package="policyengine-zz",
    )

    assert register(compact) is compact
    assert register(local) is local
    assert registry_module.REGISTRY[("zz", 2099, "compact")] == compact
    assert registry_module.REGISTRY[("zz", 2099, "local")] == local


def test_register_refuses_to_shadow_a_key_with_a_different_spec() -> None:
    conflicting = DatasetSpec(
        country="us",
        year=2024,
        variant=DEFAULT_VARIANT,
        hf_repo="someone-else/populace-us",
        filename="populace_us_2024.h5",
        engine_module="policyengine_us.data",
        engine_class="USSingleYearDataset",
        engine_package="policyengine-us",
    )
    with pytest.raises(ValueError, match="already registered for"):
        register(conflicting)


def test_uk_2025_dense_is_registered_off_the_default_variant() -> None:
    spec = REGISTRY[("uk", 2025, "dense")]
    assert spec.variant == "dense" != DEFAULT_VARIANT
    assert spec.hf_repo == "policyengine/populace-uk-private"
    assert spec.filename == "microcosm_uk_2024_25_dense.h5"
    assert spec.engine_class == "UKSingleYearDataset"
    assert spec.engine_package == "policyengine-uk"
    assert ("uk", 2025, DEFAULT_VARIANT) in REGISTRY


def test_uk_2025_compact_is_registered_as_the_national_line() -> None:
    spec = REGISTRY[("uk", 2025, DEFAULT_VARIANT)]
    assert spec.variant == DEFAULT_VARIANT
    assert spec.hf_repo == "policyengine/populace-uk-private"
    assert spec.filename == "microcosm_uk_2024_25.h5"
    assert spec.engine_class == "UKSingleYearDataset"
    assert spec.engine_package == "policyengine-uk"
    assert spec.pointer_path == "latest-national.json"
