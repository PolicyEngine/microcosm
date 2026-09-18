from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
UK_PACKAGE = ROOT / "packages/microcosm-build/src/microcosm/build/uk"


def _load(name: str) -> dict:
    return json.loads((UK_PACKAGE / name).read_text(encoding="utf-8"))


def test_ofgem_region_crosswalk_covers_every_frs_region_once() -> None:
    from microcosm.build.uk_runtime.energy_pricing import load_ofgem_region_crosswalk
    from microcosm.build.uk_runtime.lcfs_consumption import LCFS_REGIONS
    from microcosm.build.uk_runtime.ledger_fact_vendoring import load_vendored_resource

    payload = load_ofgem_region_crosswalk()
    assert payload["version"] == 1
    assert payload["country"] == "uk"
    assert not (UK_PACKAGE / "need_energy_targets.json").exists()
    mapping = payload["mapping"]
    assert set(mapping) == set(LCFS_REGIONS.values())
    ofgem_ids = {
        row["geography"]["id"]
        for row in load_vendored_resource("ofgem_price_cap_facts.json")["rows"]
    }
    assert set(mapping.values()) <= ofgem_ids
    assert mapping["NORTHERN_IRELAND"] == payload["gb_geography_id"] == "K03000001"
    assert all(
        value.startswith("ofgem:")
        for region, value in mapping.items()
        if region != "NORTHERN_IRELAND"
    )
    # The QEP price-region and DESNZ subnational-area legs cover the same
    # twelve FRS regions; QEP ids are the vendored groupby ids, areas the
    # vendored geography ids (Northern Ireland has no subnational row).
    qep = payload["qep_price_region_mapping"]
    assert set(qep) == set(LCFS_REGIONS.values())
    qep_ids = {
        row["layout"]["groupby_value_id"]
        for row in load_vendored_resource("qep_energy_prices.json")["rows"]
    }
    assert set(qep.values()) <= qep_ids
    assert payload["qep_uk_average_id"] in qep_ids
    areas = payload["subnational_area_mapping"]
    assert set(areas) == set(LCFS_REGIONS.values())
    area_ids = {
        row["geography"]["id"]
        for row in load_vendored_resource("desnz_domestic_energy_facts.json")["rows"]
    }
    assert {a for a in areas.values() if a is not None} <= area_ids
    assert areas["NORTHERN_IRELAND"] is None


def test_policy_anchor_resources_carry_parameter_paths() -> None:
    vat = _load("etb_policy_anchors.json")
    services = _load("etb_services_anchors.json")

    assert not (UK_PACKAGE / "lcfs_consumption_anchors.json").exists()
    assert vat["source"]["urls"]
    assert services["source"]["urls"]
    assert vat["vat"]["standard_rate"]["parameter_path"] == (
        "gov.hmrc.vat.standard_rate"
    )
    assert vat["vat"]["reduced_rate_share"]["value"] == 0.025
    for key in ("rail_fare_index_2023", "rail_fare_index_2024"):
        assert services[key]["parameter_path"] == "gov.dft.rail.fare_index"
    assert services["rail_fare_index_2024"]["period"] == 2024
    assert services["nhs_budget_2025_26"]["value"] == 202_000_000_000


@pytest.mark.requires_uk
def test_policy_anchor_values_lockstep_with_engine_parameter_tree() -> None:
    """Every anchor value with a parameter_path equals the installed tree's value.

    This is the Option A drift guard adjudicated on microcosm#682: an engine
    bump that moves one of these historical values fails here and forces a
    reviewed resource diff. Skips where policyengine-uk is absent — PR CI's
    hermetic lanes never import the engine.
    """
    system_module = pytest.importorskip("policyengine_uk.system")

    parameters = system_module.system.parameters
    vat = _load("etb_policy_anchors.json")["vat"]
    for name, anchor in vat.items():
        node = parameters
        for part in anchor["parameter_path"].split("."):
            node = getattr(node, part)
        assert float(node(str(anchor["period"]))) == anchor["value"], name

    services = _load("etb_services_anchors.json")
    for key in ("rail_fare_index_2023", "rail_fare_index_2024"):
        anchor = services[key]
        node = parameters
        for part in anchor["parameter_path"].split("."):
            node = getattr(node, part)
        assert float(node(str(anchor["period"]))) == anchor["value"], key


def test_nhs_consumption_resource_ports_public_csv_rows() -> None:
    payload = _load("nhs_consumption_by_age_gender.json")

    assert payload["version"] == 1
    assert payload["source"]["sdc_treatment"].startswith("Public aggregate")
    assert len(payload["rows"]) == 252
    first = payload["rows"][0]
    assert set(first) == {"Service", "Gender", "Age group", "Metric", "Total"}
    assert isinstance(first["Total"], float)


def test_e6_support_bounds_resources_are_sha_bound_and_non_placeholder() -> None:
    lcfs = _load("lcfs_consumption_support_bounds.json")
    vat = _load("etb_vat_support_bounds.json")
    services = _load("etb_services_support_bounds.json")

    assert lcfs["source"]["household_tab_sha256"] == (
        "6e78f0914be38e63853165486d641cbd790753cc471086210c6f672bfa18ca72"
    )
    assert lcfs["source"]["person_tab_sha256"] == (
        "f32d54d83cdecf023f0ac73530be3a99372099b596e0106a56eae42a64929e50"
    )
    # Raked columns (energy and, since microcosm#890, bus fares) carry no bounds.
    assert len(lcfs["bounds"]) == 14
    assert "bus_fare_spending" not in lcfs["bounds"]
    assert vat["source"]["tab_sha256"] == (
        "d0e94ebc92e85ca1b9fb3a7353dcaf41db2c5110c9f07c7793dc8c0b695250d8"
    )
    assert set(vat["bounds"]) == {"full_rate_vat_expenditure_rate"}
    assert set(services["bounds"]) == {
        "dfe_education_spending",
        "rail_subsidy_spending",
    }
    for name in (
        "lcfs_consumption_support_bounds.json",
        "etb_vat_support_bounds.json",
        "etb_services_support_bounds.json",
    ):
        assert (
            "placeholder" not in (UK_PACKAGE / name).read_text(encoding="utf-8").lower()
        )


def test_e6_support_bounds_round_trip_against_licensed_tabs() -> None:
    lcfs_hh = os.environ.get("POPULACE_UK_LCFS_HH_TAB")
    lcfs_person = os.environ.get("POPULACE_UK_LCFS_PERSON_TAB")
    etb = os.environ.get("POPULACE_UK_ETB_TAB")
    if not all(
        value and Path(value).is_file() for value in (lcfs_hh, lcfs_person, etb)
    ):
        pytest.skip(
            "licensed E6 tabs not available "
            "(set POPULACE_UK_LCFS_HH_TAB, POPULACE_UK_LCFS_PERSON_TAB, "
            "POPULACE_UK_ETB_TAB)"
        )
    import importlib.util

    path = ROOT / "tools/build_uk_e6_support_bounds.py"
    spec = importlib.util.spec_from_file_location("build_uk_e6_support_bounds", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    expected = {
        "lcfs_consumption_support_bounds.json": module.build_lcfs_support_bounds(
            Path(lcfs_hh), Path(lcfs_person)
        ),
        "etb_vat_support_bounds.json": module.build_etb_vat_support_bounds(Path(etb)),
        "etb_services_support_bounds.json": module.build_etb_services_support_bounds(
            Path(etb)
        ),
    }
    for name, payload in expected.items():
        assert json.dumps(payload, indent=2, sort_keys=False) + "\n" == (
            UK_PACKAGE / name
        ).read_text(encoding="utf-8")
