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


def test_us_2024_is_registered_with_a_complete_spec() -> None:
    spec = REGISTRY[("us", 2024, DEFAULT_VARIANT)]
    assert spec.country == "us"
    assert spec.year == 2024
    assert spec.variant == DEFAULT_VARIANT
    assert spec.hf_repo == "policyengine/populace-us"
    # The root file, and the file the builds moved by the switch keep; new
    # builds of the national release declare the new name.
    assert spec.filename == "populace_us_2024.h5"
    assert spec.build_folder_filename == "microcosm_us_2024.h5"
    assert spec.release == "microcosm-us-2024-national"
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
        release="microcosm-zz-2099-dense",
    )
    monkeypatch.setattr(
        loader_module,
        "REGISTRY",
        {**REGISTRY, dense_only.key: dense_only},
    )

    keys = available()
    assert ("us", 2024) in keys
    assert ("uk", 2023) in keys
    # The national line is the default variant since its promotion (#823).
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
        release="microcosm-us-2024-national",
        build_folder_filename="microcosm_us_2024.h5",
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
        release="microcosm-zz-2099-national",
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
        release="microcosm-zz-2099-local",
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
        release="microcosm-us-2024-national",
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
    # Each new dense build becomes current when it is published.
    assert spec.release == "microcosm-uk-2024-25-dense"
    assert ("uk", 2025, DEFAULT_VARIANT) in REGISTRY


def test_uk_2025_national_release_is_the_default_variant():
    """The national release is the default UK 2025 load: its current build in
    releases.json. The June 2023 release stays addressable at its year, as
    release populace-uk-2023, whose one build the UK switch moved."""
    spec = REGISTRY[("uk", 2025, DEFAULT_VARIANT)]
    assert spec.release == "microcosm-uk-2024-25-national"
    assert spec.filename == "microcosm_uk_2024_25.h5"
    assert ("uk", 2025, "national") not in REGISTRY
    june = REGISTRY[("uk", 2023, DEFAULT_VARIANT)]
    assert june.filename == "populace_uk_2023.h5"
    assert june.release == "populace-uk-2023"


def test_each_registry_entry_names_its_release():
    """The loader reads each entry's release through its repository's
    releases.json."""
    assert {spec.key: spec.release for spec in REGISTRY.values()} == {
        ("us", 2024, DEFAULT_VARIANT): "microcosm-us-2024-national",
        ("uk", 2023, DEFAULT_VARIANT): "populace-uk-2023",
        ("uk", 2025, DEFAULT_VARIANT): "microcosm-uk-2024-25-national",
        ("uk", 2025, "dense"): "microcosm-uk-2024-25-dense",
    }
