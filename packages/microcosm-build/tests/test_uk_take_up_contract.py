from __future__ import annotations

import copy
import hashlib
import json
from importlib import metadata
from importlib.resources import files
from pathlib import Path

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
        0.55,
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
        0.593,
    ]


def test_year_selection_uses_latest_value_at_or_before_build_year() -> None:
    contract = load_uk_take_up_contract()

    assert contract.rate("child_benefit", 2021) == 0.97
    assert contract.rate("child_benefit", 2023) == 0.89
    assert contract.rate("tax_free_childcare", 2024) == 0.88
    assert contract.rate("tv_licence_evasion_rate", 2024) == 0.1252
    assert contract.rate("tax_free_childcare_spend_routed_share", 2017) == 0.359
    assert contract.rate("tax_free_childcare_spend_routed_share", 2024) == 0.593
    assert contract.rate("tax_free_childcare_spend_routed_share", 2025) == 0.582


def test_rate_entity_is_closed_world(monkeypatch) -> None:
    raw = _resource()
    mutated = copy.deepcopy(raw)
    mutated["stochastic"][-1]["entity"] = "government"
    _reload_with(monkeypatch, mutated)
    with pytest.raises(ValueError, match="invalid entity"):
        load_uk_take_up_contract()


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
    mutated["continuous"].append(
        {
            "key": "synthetic_fitted_entry",
            "source": {
                "source": "synthetic test source",
                "status": "fitted_offline",
            },
        }
    )
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


def test_tfc_routed_share_series_matches_pinned_chronicle_feed_when_present() -> None:
    feed = Path(__file__).resolve().parents[3] / ".codex-work/consumer_facts_uk.jsonl"
    if not feed.exists():
        pytest.skip("pinned Chronicle UK feed is not present")

    annual: dict[int, float] = {}
    monthly: dict[int, list[float]] = {}
    matched = 0
    with feed.open(encoding="utf-8") as rows:
        for line in rows:
            if "hmrc-tax-free-childcare-march-2026" not in line:
                continue
            row = json.loads(line)
            matched += 1
            period = row["period"]
            measure = row["observed_measure"]["source_measure_id"]
            if measure == "annual_unique_children_with_used_accounts":
                annual[int(period["value"])] = float(row["value"])
            elif measure == "monthly_children_with_used_accounts":
                year, month = map(int, str(period["value"]).split("-"))
                financial_year = year if month >= 4 else year - 1
                monthly.setdefault(financial_year, []).append(float(row["value"]))

    assert matched == 126
    contract = load_uk_take_up_contract()
    entry = contract.entry("tax_free_childcare_spend_routed_share")
    for year in range(2017, 2026):
        assert len(monthly[year]) == 12
        derived = sum(monthly[year]) / annual[year] / 12
        assert round(derived, 3) == entry.values[f"{year}-04-01"]
    assert entry.values["2017-04-01"] == 0.359
    assert entry.values["2024-04-01"] == 0.593


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
