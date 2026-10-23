"""Reviewed selector inventory and ERP bridge, including a pinned-feed audit."""

from __future__ import annotations

import copy
import json
import math
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_artifact import (
    ChronicleFeedPinMismatchError,
    load_ledger_consumer_artifact,
)
from microcosm.build.ledger_targets import (
    compile_ledger_target_references,
    fact_matches_selector,
)
from microcosm.build.transport.target_kernels import parse_reference_document
from test_support.paths import paths_for
from tools.generate_nz_target_references import build_reference_documents

_PATHS = paths_for("microcosm-build")
_SPEC = _PATHS.package / "src/microcosm/build/nz"
REGIONS = (
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
)
ERP = [
    ("01", 198500, 201100),
    ("02", 1755200, 1816000),
    ("03", 516800, 532100),
    ("04", 346500, 351500),
    ("05", 52300, 52700),
    ("06", 179000, 179700),
    ("07", 129400, 130300),
    ("08", 257800, 260700),
    ("09", 538000, 543400),
    ("12", 33900, 34700),
    ("13", 676600, 698200),
    ("14", 248500, 253900),
    ("15", 102700, 104800),
    ("16", 59100, 59900),
    ("17", 54300, 54300),
    ("18", 50800, 50800),
    ("NZ", 5200000, 5324700),
]


def _document(name):
    return json.loads((_SPEC / name).read_text())


def _rows(name="target_references.json"):
    return _document(name)["target_references"]


def _parsed(rows):
    return parse_reference_document(
        {"country": "nz", "target_references": rows}, country="nz"
    )


def test_inventory_has_exact_contract_counts_and_named_parking_reasons():
    counts = Counter(row["metadata"]["contract_target_id"] for row in _rows())
    assert counts == {
        "stats_nz_population_by_age_region": 64,
        "stats_nz_census_households_by_size_region": 136,
        "stats_nz_census_families_by_type_region": 51,
        "stats_nz_census_households_by_tenure_region": 34,
        "ird_taxable_income_people_by_band": 237,
        "ird_taxable_income_total_by_band": 237,
        "msd_main_benefit_recipients_by_program_age": 12,
        "msd_nz_super_recipients": 1,
        "msd_veterans_pension_recipients": 1,
    }
    assert len(_rows("precal_references.json")) == 318
    for filename, active in (
        ("target_references.json", 98),
        ("precal_references.json", 67),
    ):
        rows = _rows(filename)
        assert (
            sum(row["metadata"]["activation_status"] == "active" for row in rows)
            == active
        )
        assert len({row["name"] for row in rows}) == len(rows)
        assert all(row.get("tolerance") is None for row in rows)
        for row in rows:
            if row["metadata"]["activation_status"] != "active":
                assert row["metadata"]["parking_reason"].strip(), row["name"]
    exclusions = _document("reference_inventory.json")["exclusions"]
    assert {row["id"] for row in exclusions} >= {
        "other_main_benefits",
        "all_main_benefits",
        "combined_youth_and_young_parent",
        "ta_999_erp",
        "qes_prefu_and_b5_ageing",
        "wff_tripwires",
    }
    assert all(row["reason"].strip() for row in exclusions)


def test_every_parked_row_refuses_direct_compilation():
    for filename in ("target_references.json", "precal_references.json"):
        for reference in _parsed(_rows(filename)):
            if reference.metadata["activation_status"] != "active":
                with pytest.raises(ValueError, match="non-executable placeholder"):
                    compile_ledger_target_references([], [reference], country="nz")


def test_b1_uses_only_64_all_sex_regional_age_leaves():
    rows = [row for row in _rows() if row["family"] == "demography"]
    assert len(rows) == 64
    pairs = set()
    for row in rows:
        selector = row["ledger_selector"]
        region = selector["geography_id"]
        band = selector["dimension_values"]["person.age_band"]
        pairs.add((region, band))
        assert selector["dimensions"] == ["person.age_band"]
        assert selector["period_value"] == 2025
        assert (
            selector["record_set_id"]
            == f"stats_nz.cy2025.subnational_population.age_{band}"
        )
        assert row["filter"] == f"population_region_{region}_age_{band}_indicator"
        assert "sex" not in row["name"] + row["filter"]
    assert pairs == {
        (region, band)
        for region in REGIONS
        for band in ("0_14", "15_39", "40_64", "65_plus")
    }


def test_grouped_tenure_family_multiplicities_and_parked_definitions():
    for row in _rows():
        selector = row["ledger_selector"]
        if row["family"] == "housing_tenure":
            code = selector["dimension_values"][
                "nz_census_2023.household_tenure_source_label"
            ]
            if row["name"].endswith("__owned"):
                assert code == [1, 3] and all(type(member) is int for member in code)
                assert row["value_operands"][0] == {
                    "role": "base",
                    "expected_member_count": 2,
                }
            else:
                assert type(code) is int and code == 2
            assert row["measure"] == "households"
            assert row["hierarchy"]["target_label"]
            assert row["value_operation"] == "rolled_forward_by_ratio"
            assert row["metadata"]["projection_from_june2025_to_model_period"] == "none"
        if "families_by_type" in row["name"]:
            assert row["entity"] == "household"
            assert row["measure"] in {
                "census_families_type_1",
                "census_families_type_2",
                "census_families_type_3",
            }
            assert row["filter"] == f"region_{selector['geography_id']}_indicator"
            assert (
                row["metadata"]["activation_status"]
                == "requires_census_family_nucleus_standard"
            )
        if "households_by_size" in row["name"]:
            assert "61,725" in row["metadata"]["parking_reason"]
            assert "unexplained" in row["metadata"]["parking_reason"]
        if row["family"] == "income":
            assert row["metadata"]["activation_status"].startswith("requires_g8r2")


def test_receipt_snapshots_are_split_and_stratified_inputs_stay_parked():
    source_sets = []
    for filename in ("target_references.json", "precal_references.json"):
        rows = [
            row
            for row in _rows(filename)
            if row["ledger_selector"]["source_name"] == "msd"
        ]
        assert len(rows) == 14
        source_sets.append(
            {json.dumps(row["ledger_selector"], sort_keys=True) for row in rows}
        )
        pensions = [row for row in rows if row["family"] == "superannuation"]
        assert {row["measure"] for row in pensions} == {
            "receives_nz_super",
            "receives_veterans_pension",
        }
        assert {row["ledger_selector"]["source_measure_id"] for row in pensions} == {
            "nzs_recipients",
            "vp_recipients",
        }
        for row in rows:
            assert row["period"] == row["ledger_selector"]["period_value"] == "2026-06"
            assert "snapshot stock" in row["metadata"]["measurement_basis"]
            if row["family"] == "main_benefits":
                assert (
                    row["metadata"]["activation_status"]
                    == "requires_stratified_receipt_assignment"
                )
                assert row["filter"].startswith("age_")
                assert row["ledger_selector"]["dimension_values"][
                    "msd_main_benefit_type"
                ] in {
                    "jobseeker_support",
                    "sole_parent_support",
                    "supported_living_payment",
                }
            if row["measure"] == "receives_veterans_pension":
                assert (
                    row["metadata"]["activation_status"]
                    == "requires_unsupported_veterans_pension"
                )
    assert source_sets[0] == source_sets[1]
    wages = [
        row for row in _rows("precal_references.json") if row["family"] == "earnings"
    ]
    assert len(wages) == 235
    assert all(
        row["measure"] == "people"
        and row["entity"] == "person"
        and row.get("filter") is None
        for row in wages
    )
    assert all(
        row["metadata"]["activation_status"].startswith("requires_g8r2")
        for row in wages
    )


@pytest.fixture(scope="module")
def artifact():
    path = os.environ.get("NZ_CHRONICLE_ARTIFACT")
    if not path:
        pytest.skip("NZ_CHRONICLE_ARTIFACT is unset; public NZ feed audit unavailable")
    return load_ledger_consumer_artifact(
        Path(path), feed_pin=_document("chronicle_feed.json")
    )


@pytest.fixture(scope="module")
def facts_by_key(artifact):
    return {fact["aggregate_fact_key"]: fact for fact in artifact.facts}


def _matched(selector, facts_by_key):
    if "aggregate_fact_key" in selector:
        fact = facts_by_key.get(selector["aggregate_fact_key"])
        return (
            [fact] if fact is not None and fact_matches_selector(fact, selector) else []
        )
    return [
        fact
        for fact in facts_by_key.values()
        if fact.get("layout", {}).get("record_set_id") == selector.get("record_set_id")
        and fact_matches_selector(fact, selector)
    ]


def test_generation_reproduces_committed_resources_with_exact_feed_identity(artifact):
    generated = build_reference_documents(artifact)
    assert _document("reference_inventory.json")["feed_identity"] == _document(
        "chronicle_feed.json"
    )
    for filename, document in generated.items():
        assert document == _document(filename), filename
    assert artifact.fact_row_count == _document("chronicle_feed.json")["fact_row_count"]
    assert artifact.fact_schema_versions == ("chronicle.consumer_fact.v4",)


def test_every_active_scalar_resolves_once_and_grouped_tenure_has_exact_roster(
    artifact, facts_by_key
):
    for filename in ("target_references.json", "precal_references.json"):
        references = _parsed(
            [
                row
                for row in _rows(filename)
                if row["metadata"]["activation_status"] == "active"
            ]
        )
        needed = {}
        for ref in references:
            base = _matched(ref.ledger_selector, facts_by_key)
            expected = 2 if ref.name.endswith("__owned") else 1
            assert len(base) == expected, ref.name
            if expected == 2:
                codes = [
                    fact["dimensions"]["nz_census_2023.household_tenure_source_label"]
                    for fact in base
                ]
                assert sorted(codes) == [1, 3] and all(
                    type(code) is int for code in codes
                )
            members = list(base)
            for operand in ref.value_operands:
                if operand.get("role") == "base":
                    continue
                selector = {
                    key: value for key, value in operand.items() if key != "role"
                }
                resolved = _matched(selector, facts_by_key)
                assert len(resolved) == 1, (ref.name, operand)
                members.extend(resolved)
            needed.update({fact["aggregate_fact_key"]: fact for fact in members})
        registry = compile_ledger_target_references(
            needed.values(), references, country="nz"
        )
        assert len(registry.specs) == len(references)
        for spec in registry.specs:
            assert spec.hierarchy.geography.id
            assert spec.hierarchy.target.label


@pytest.mark.parametrize(("region", "denominator", "numerator"), ERP)
def test_all_17_erp_ratios_match_the_reviewed_source_table(
    artifact, facts_by_key, region, denominator, numerator
):
    receipt = next(
        row
        for row in _document("reference_inventory.json")["erp_bridges"]
        if row["geography_id"] == region
    )
    assert receipt["denominator"] == denominator
    assert receipt["numerator"] == numerator
    assert receipt["ratio"] == numerator / denominator
    for role, year, value in (
        ("numerator", 2025, numerator),
        ("denominator", 2023, denominator),
    ):
        selector = receipt[f"{role}_selector"]
        assert selector["geography_id"] == region
        assert selector["period_value"] == year
        assert (
            selector["record_set_id"]
            == f"stats_nz.cy{year}.subnational_population.all_ages"
        )
        assert selector["dimensions"] == []
        (fact,) = _matched(selector, facts_by_key)
        assert fact["value"] == value


def test_scaled_tenure_values_are_fractional_exact_bridges_and_national_diagnostics_are_separate(
    artifact, facts_by_key
):
    rows = [row for row in _rows() if row["family"] == "housing_tenure"]
    registry = compile_ledger_target_references(
        artifact.facts, _parsed(rows), country="nz"
    )
    expected = []
    for row, spec in zip(rows, registry.specs, strict=True):
        base = math.fsum(
            fact["value"] for fact in _matched(row["ledger_selector"], facts_by_key)
        )
        operands = row["value_operands"]
        num = _matched(
            {k: v for k, v in operands[1].items() if k != "role"}, facts_by_key
        )[0]["value"]
        den = _matched(
            {k: v for k, v in operands[2].items() if k != "role"}, facts_by_key
        )[0]["value"]
        assert spec.value == base * (num / den)
        expected.append(base * (num / den))
    diagnostic = next(
        row
        for row in _document("reference_inventory.json")["national_bridge_diagnostics"]
        if row["table"] == "tenure"
    )
    assert math.fsum(spec.value for spec in registry.specs) == pytest.approx(
        math.fsum(expected), abs=1e-8
    )
    assert diagnostic["regional_leaf_sum_scaled"] == pytest.approx(
        math.fsum(expected), abs=1e-8
    )
    assert diagnostic["independently_scaled_published_national_total"] == 1780527 * (
        5324700 / 5200000
    )
    assert (
        diagnostic["regional_leaf_sum_scaled"]
        != diagnostic["independently_scaled_published_national_total"]
    )
    assert any(value != round(value) for value in expected)


def test_source_discrepancies_remain_unreconciled_and_no_totals_enter_the_fit(
    artifact, facts_by_key
):
    receipt = _document("reference_inventory.json")
    for row in receipt["source_discrepancies"]:
        (total,) = _matched(row["published_national_total_selector"], facts_by_key)
        leaves = [
            _matched(selector, facts_by_key)[0]
            for selector in row["national_leaf_selectors"]
        ]
        assert row["published_national_total"] == total["value"]
        assert row["published_national_leaf_sum"] == math.fsum(
            fact["value"] for fact in leaves
        )
        assert row["deficit"] == total["value"] - row["published_national_leaf_sum"]
    assert {
        row["table"]: row["deficit"] for row in receipt["source_discrepancies"]
    } == {"tenure": 426, "household_size": 61725, "family_type": 0}
    for row in _rows():
        assert "residual" not in row["name"].lower()
        if row["family"] in {"demography", "housing_tenure", "household_structure"}:
            assert row["ledger_selector"]["dimensions"], row["name"]
    assert (
        _document("target_references.json")["target_profile"][
            "hierarchy_reconciliations"
        ]
        == []
    )


def test_published_regional_totals_and_national_totals_remain_independent_diagnostics(
    artifact, facts_by_key
):
    inventory = _document("reference_inventory.json")
    ratios = {row["geography_id"]: row["ratio"] for row in inventory["erp_bridges"]}
    for row in inventory["national_bridge_diagnostics"]:
        totals = [
            _matched(selector, facts_by_key)[0]
            for selector in row["regional_published_total_selectors"]
        ]
        assert len(totals) == 17
        scaled = [
            fact["value"]
            * ratios[
                "NZ" if fact["geography"]["id"] == "99" else fact["geography"]["id"]
            ]
            for fact in totals
        ]
        assert row["regional_published_total_sum_scaled"] == math.fsum(scaled)
        assert (
            row["regional_published_total_sum_scaled"]
            != row["independently_scaled_published_national_total"]
        )
        if row["table"] == "tenure":
            assert math.fsum(scaled) == pytest.approx(1822217.446186, abs=5e-7)
        elif row["table"] == "family_type":
            assert math.fsum(scaled) == pytest.approx(1325417.426784, abs=5e-7)


def test_region_99_is_a_separate_national_growth_proxy_and_ta_999_has_no_erp(artifact):
    rows = [
        row
        for row in _rows()
        if row["family"] == "housing_tenure"
        and row["ledger_selector"]["geography_id"] == "99"
    ]
    assert len(rows) == 2
    registry = compile_ledger_target_references(
        artifact.facts, _parsed(rows), country="nz"
    )
    for row, spec in zip(rows, registry.specs, strict=True):
        assert spec.hierarchy.geography.id == "99"
        assert row["metadata"]["bridge_status"] == "national-growth proxy"
        assert "national-growth proxy" in spec.hierarchy.target.label
        assert all(
            operand["geography_id"] == "NZ" for operand in row["value_operands"][1:]
        )
    erp_facts = [
        fact
        for fact in artifact.facts
        if fact.get("observed_measure", {}).get("source_concept")
        == "stats_nz.estimated_resident_population"
    ]
    assert not any(fact["geography"]["id"] in {"99", "nz-ta-999"} for fact in erp_facts)
    support = [
        row
        for row in _rows("precal_references.json")
        if row["metadata"].get("precal_use") == "atomic_geography_support"
    ]
    assert len(support) == 67
    assert all(
        row["ledger_selector"]["geography_vintage"] == "ta_2025" for row in support
    )


@pytest.mark.parametrize(
    ("field", "altered"),
    [
        ("facts_sha256", "0" * 64),
        ("manifest_sha256", "0" * 64),
        ("fact_row_count", 12334),
        ("consumer_fact_schema_versions", ["wrong"]),
        ("consumer_fact_schema_sha256", "0" * 64),
        ("artifact_schema_version", "wrong"),
    ],
)
def test_changed_feed_identity_refuses_authoring_before_reference_compilation(
    artifact, field, altered, tmp_path
):
    pin = _document("chronicle_feed.json")
    pin[field] = altered
    (tmp_path / "chronicle_feed.json").write_text(json.dumps(pin))
    with pytest.raises(ChronicleFeedPinMismatchError):
        build_reference_documents(artifact, package_dir=tmp_path)


def test_wrong_source_pins_and_wrong_snapshot_refuse_even_with_matching_fact_key(
    artifact,
):
    spec = load_country_spec("nz")
    reference = spec.target_references[0]
    selector = {**reference.ledger_selector, "source_name": "wrong"}
    with pytest.raises(ValueError):
        compile_ledger_target_references(
            artifact.facts, [replace(reference, ledger_selector=selector)], country="nz"
        )
    receipt = next(row for row in _rows() if row["name"] == "msd_nz_super_recipients")
    receipt = copy.deepcopy(receipt)
    receipt["metadata"]["activation_status"] = "active"
    receipt["ledger_selector"]["period_value"] = "2025-06"
    with pytest.raises(ValueError):
        compile_ledger_target_references(
            artifact.facts, _parsed([receipt]), country="nz"
        )
