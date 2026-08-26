from __future__ import annotations

import copy
import hashlib
import json
from importlib import metadata
from importlib.resources import files

import pytest

from microcosm.build.uk_runtime.take_up_contract import (
    load_uk_take_up_contract,
    uk_take_up_contract_identity,
)


def _resource() -> dict:
    return json.loads(
        files("microcosm.build.uk").joinpath("take_up_contract.json").read_text()
    )


def test_uk_contract_loads_and_selects_build_year_rates() -> None:
    contract = load_uk_take_up_contract()

    assert contract.country == "uk"
    assert contract.build_year == 2024
    assert [entry.key for entry in contract.programs] == [
        "child_benefit",
        "child_benefit_opts_out_rate",
        "pension_credit",
        "universal_credit",
        "marriage_allowance",
        "tax_free_childcare",
        "extended_childcare",
        "universal_childcare",
        "targeted_childcare",
        "scp_under_6",
        "scp_6_plus",
    ]
    assert [contract.rate(entry.key) for entry in contract.programs] == [
        0.89,
        0.23,
        0.7,
        0.85,
        0.5,
        0.88,
        0.812,
        0.563,
        0.597,
        0.97,
        0.85,
    ]
    assert [contract.rate(entry.key) for entry in contract.stochastic] == [
        0.95,
        0.1252,
        0.384,
        0.0385,
    ]


def test_year_selection_uses_latest_value_at_or_before_build_year() -> None:
    contract = load_uk_take_up_contract()

    assert contract.rate("child_benefit", 2021) == 0.97
    assert contract.rate("child_benefit", 2023) == 0.89
    assert contract.rate("tax_free_childcare", 2024) == 0.88
    assert contract.rate("tv_licence_evasion_rate", 2024) == 0.1252


def test_rate_without_source_is_refused(monkeypatch) -> None:
    raw = _resource()
    mutated = copy.deepcopy(raw)
    del mutated["programs"][0]["source"]["source"]

    _reload_with(monkeypatch, mutated)

    with pytest.raises(ValueError, match="requires source"):
        load_uk_take_up_contract()


def test_frozen_and_fitted_status_blocks_are_required(monkeypatch) -> None:
    raw = _resource()
    mutated = copy.deepcopy(raw)
    del mutated["programs"][3]["source"]["freeze"]
    _reload_with(monkeypatch, mutated)
    with pytest.raises(ValueError, match="freeze block"):
        load_uk_take_up_contract()

    mutated = copy.deepcopy(raw)
    del mutated["continuous"][0]["fitting_receipt"]
    _reload_with(monkeypatch, mutated)
    with pytest.raises(ValueError, match="fitting_receipt"):
        load_uk_take_up_contract()


def test_resource_digest_binds_complete_json() -> None:
    raw = _resource()
    canonical = json.dumps(
        raw,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    expected = hashlib.sha256(canonical).hexdigest()
    contract = load_uk_take_up_contract()

    assert contract.resource_sha256 == expected
    assert uk_take_up_contract_identity(contract)["resource_sha256"] == expected


@pytest.mark.requires_uk
def test_engine_version_is_read_dynamically_and_tie_break_absent() -> None:
    import policyengine_uk

    package_version = metadata.version("policyengine-uk")
    assert package_version
    assert getattr(policyengine_uk, "__version__", None) is None
    assert (
        "higher_earner_tie_break"
        not in policyengine_uk.CountryTaxBenefitSystem().variables
    )


def test_forbidden_source_dependency_strings_absent_from_uk_resources() -> None:
    for resource in ("take_up_contract.json", "brma_rent_counts.json"):
        text = files("microcosm.build.uk").joinpath(resource).read_text().lower()
        assert "policyengine_" + "uk_data" not in text
        assert "policyengine-" + "uk-data" not in text


def _reload_with(monkeypatch, mutated: dict) -> None:
    load_uk_take_up_contract.cache_clear()
    payload = json.dumps(mutated)

    class _FakePath:
        def read_text(self, *args, **kwargs):
            return payload

    monkeypatch.setattr(
        "microcosm.build.uk_runtime.take_up_contract._contract_path",
        lambda: _FakePath(),
    )
