from __future__ import annotations

import json
from importlib import resources as importlib_resources

import pytest

from microcosm.build.uk_runtime.local_validation_levels import (
    load_uk_local_validation_levels,
)


def test_local_validation_levels_are_registered_and_loadable() -> None:
    package = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("country_package.json")
        .read_text()
    )
    assert "local_validation_levels.json" in {
        row["path"] for row in package["resources"]
    }
    assert load_uk_local_validation_levels()["schema_version"] == 1


def test_local_validation_level_membership_and_statuses_are_drift_pinned() -> None:
    rows = {row["id"]: row for row in load_uk_local_validation_levels()["rows"]}
    assert {row_id: row["status"] for row_id, row in rows.items()} == {
        "scotland_private_rent_country_total": "awaiting_facts",
        "wales_private_rent_country_total": "awaiting_facts",
        "wales_constituency_uc_sum": "awaiting_facts",
        "wales_constituency_employment_income_sum": "available",
        "wales_constituency_self_employment_income_sum": "available",
        "england_council_tax_net": "available",
        "wales_council_tax_net": "available",
        "scotland_council_tax_net": "available",
        "regional_public_sector_employment": "awaiting_facts",
    }
    assert rows["wales_constituency_uc_sum"]["in_sample"] is True
    assert rows["regional_public_sector_employment"]["in_sample"] is False
    assert all("benchmark_value" not in row for row in rows.values())


def test_local_validation_loader_rejects_status_drift(monkeypatch) -> None:
    from microcosm.build.uk_runtime import local_validation_levels as module

    real_loads = module.json.loads

    def drifted_loads(text):
        payload = real_loads(text)
        payload["rows"][0]["status"] = "invented"
        return payload

    monkeypatch.setattr(module.json, "loads", drifted_loads)
    with pytest.raises(ValueError, match="invalid status"):
        load_uk_local_validation_levels()
