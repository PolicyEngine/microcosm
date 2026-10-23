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
    reference_fact_selectors,
)
from microcosm.build.reference_profiles import (
    REFERENCE_DOCUMENT_KEYS,
    reference_document_sha256,
    resolve_reference_profiles,
    validate_reference_resolution,
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
RESIDUALS = [
    ("01", 70476, 66261, 4215, 70470, 6),
    ("02", 539490, 521835, 17655, 539358, 132),
    ("03", 177543, 170253, 7290, 177447, 96),
    ("04", 118452, 113598, 4854, 118437, 15),
    ("05", 17124, 16236, 888, 17124, 0),
    ("06", 62964, 59523, 3441, 62910, 54),
    ("07", 48108, 46329, 1779, 48090, 18),
    ("08", 95187, 92073, 3114, 95175, 12),
    ("09", 194406, 188772, 5634, 194370, 36),
    ("12", 14205, 13347, 858, 14199, 6),
    ("13", 248550, 242292, 6258, 248520, 30),
    ("14", 91563, 89067, 2496, 91551, 12),
    ("15", 40185, 38892, 1293, 40182, 3),
    ("16", 21864, 21030, 834, 21864, 0),
    ("17", 20577, 20112, 465, 20571, 6),
    ("18", 19590, 18966, 624, 19587, 3),
    ("99", 240, 225, 15, 240, 0),
]


def _document(name):
    return json.loads((_SPEC / name).read_text())


def _rows(name="target_references.json"):
    return _document(name)["target_references"]


def _parsed(rows):
    return parse_reference_document(
        {"country": "nz", "target_references": rows}, country="nz"
    )


def _effective_rows(profile="mar"):
    document, _, _ = resolve_reference_profiles(
        _document("target_references.json"),
        _document("reference_profiles.json"),
        {"census_residual_allocation": profile, "aging_index": "central"},
    )
    return document["target_references"]


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
        ("target_references.json", 64),
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


def test_residual_profiles_have_exact_recipient_rosters_and_preserve_other_targets():
    raw = _rows()
    assert len([row for row in raw if "households_by_size" in row["name"]]) == 136
    for profile in ("mar", "small_households"):
        effective = _effective_rows(profile)
        assert (
            sum(row["metadata"]["activation_status"] == "active" for row in effective)
            == 234
        )
        assert {
            row["ledger_selector"]["geography_id"]
            for row in effective
            if "households_by_size" in row["name"]
        } == {*REGIONS, "99"}
        for before, after in zip(raw, effective, strict=True):
            if (
                before["metadata"]["activation_status"]
                != "requires_reference_profile_resolution"
            ):
                assert before == after
                continue
            assert before["ledger_selector"] == after["ledger_selector"]
            assert after["metadata"]["activation_status"] == "active"
            assert after["value_operation"] == "allocated_residual_by_ratio"
            operands = {operand["role"]: operand for operand in after["value_operands"]}
            partition = operands["partition"]
            size = "households_by_size" in after["name"]
            expected_codes = list(range(1, 9 if size else 4))
            member_selectors = partition["member_selectors"]
            dimension = (
                "nz_census_2023.household_size_source_label"
                if size
                else "nz_census_2023.household_tenure_source_label"
            )
            assert [
                selector["dimension_values"][dimension] for selector in member_selectors
            ] == expected_codes
            recipients = [
                selector["dimension_values"][dimension]
                for selector in member_selectors
                if selector["aggregate_fact_key"] in partition["allocation_member_keys"]
            ]
            assert recipients == (
                [1, 2] if size and profile == "small_households" else expected_codes
            )
            assert (
                operands["total"]["geography_id"]
                == after["ledger_selector"]["geography_id"]
            )
            assert operands["total"]["dimensions"] == []
            if after["ledger_selector"]["geography_id"] == "99":
                assert (
                    after["metadata"]["growth_geography_proxy"]
                    == "national-growth proxy"
                )
                assert "national-growth proxy" in after["hierarchy"]["target_label"]
            assert after["metadata"]["allocation_ruling"] == "d1886"
            assert after["metadata"]["census_residual_allocation"] == profile
            assert profile in after["hierarchy"]["target_label"]


@pytest.mark.parametrize("profile", ["mar", "small_households"])
def test_full_nz_resource_resolves_directly_to_the_strict_compiler_document(profile):
    raw = _document("target_references.json")
    assert {
        "reference_set",
        "description",
        "allowed_value_operations",
        "target_profile",
    } <= set(raw)
    effective, intervals, receipt = resolve_reference_profiles(
        raw,
        _document("reference_profiles.json"),
        {"census_residual_allocation": profile, "aging_index": "central"},
    )
    assert intervals is None
    assert set(effective) == set(raw) & REFERENCE_DOCUMENT_KEYS
    references = parse_reference_document(effective, country="nz")
    assert len(references) == 773
    active = [
        row for row in references if row.metadata["activation_status"] == "active"
    ]
    parked = [
        row for row in references if row.metadata["activation_status"] != "active"
    ]
    assert len(active) == 234
    assert len(parked) == 539
    assert receipt["base_references_sha256"] == reference_document_sha256(raw)
    assert receipt["effective_references_sha256"] == reference_document_sha256(
        effective
    )
    assert receipt["base_references_sha256"] != receipt["effective_references_sha256"]
    assert (
        validate_reference_resolution(
            effective,
            receipt,
            references_sha256=reference_document_sha256(effective),
            interval_document=intervals,
        )
        == receipt
    )
    for reference in parked:
        with pytest.raises(ValueError, match="non-executable placeholder"):
            compile_ledger_target_references([], [reference], country="nz")


def test_raw_tenure_and_size_cannot_bypass_profile_resolution():
    references = _parsed(
        [
            row
            for row in _rows()
            if "households_by_size" in row["name"] or row["family"] == "housing_tenure"
        ]
    )
    assert len(references) == 170
    for reference in references:
        with pytest.raises(ValueError, match="non-executable placeholder"):
            compile_ledger_target_references([], [reference], country="nz")


def test_raw_tenure_refuses_even_when_all_real_operands_are_present(artifact):
    rows = [row for row in _rows() if row["family"] == "housing_tenure"]
    assert len(rows) == 34
    for reference in _parsed(rows):
        with pytest.raises(ValueError, match="non-executable placeholder"):
            compile_ledger_target_references(artifact.facts, [reference], country="nz")


def test_v4_changes_only_the_residual_knob_and_prior_variants_keep_their_meaning():
    document = _document("scenarios.json")
    assert document["central_knobs"]["census_residual_allocation"] == "mar"
    assert document["knob_domains"]["census_residual_allocation"] == [
        "mar",
        "small_households",
    ]
    rows = {row["id"]: row for row in document["scenarios"]}
    assert rows["V1"]["knobs"] == {"aging_index": "low"}
    assert rows["V2"]["knobs"] == {"aging_index": "high"}
    assert rows["V3"]["knobs"] == {"beneficiary_assignment": "variant"}
    assert rows["V4"]["knobs"] == {"census_residual_allocation": "small_households"}
    assert rows["V4"]["tier"] == "calibration"
    for scenario in rows.values():
        knobs = {**document["central_knobs"], **scenario["knobs"]}
        assert knobs["census_residual_allocation"] == (
            "small_households" if scenario["id"] == "V4" else "mar"
        )


@pytest.mark.parametrize("profile", ["mar", "small_households"])
def test_inventory_asserts_exact_regional_residual_table(profile):
    receipts = [
        row
        for row in _document("reference_inventory.json")["regional_profile_allocations"]
        if row["profile"] == profile
    ]
    actual = {}
    for region in (*REGIONS, "99"):
        size = next(
            row
            for row in receipts
            if row["geography_id"] == region and row["table"] == "household_size"
        )
        tenure = next(
            row
            for row in receipts
            if row["geography_id"] == region and row["table"] == "tenure"
        )
        actual[region] = (
            region,
            size["published_total"],
            size["raw_leaf_sum"],
            size["computed_residual"],
            tenure["raw_leaf_sum"],
            tenure["computed_residual"],
        )
        assert size["published_total"] == tenure["published_total"]
        for receipt in (size, tenure):
            assert math.isclose(
                receipt["allocated_closure"],
                receipt["published_total"],
                rel_tol=1e-12,
                abs_tol=1e-7,
            )
            assert math.isclose(
                receipt["final_closure"],
                receipt["published_total"] * receipt["growth_ratio"],
                rel_tol=1e-12,
                abs_tol=1e-7,
            )
            assert receipt["growth_applications"] == 1
            assert receipt["total_selector"]["geography_id"] == region
    assert list(actual.values()) == RESIDUALS
    assert math.fsum(row[3] for row in actual.values()) == 61713
    assert math.fsum(row[5] for row in actual.values()) == 429


def test_sensitivity_removes_mar_additions_from_sizes_three_to_eight():
    receipts = _document("reference_inventory.json")["regional_profile_allocations"]
    for region in (*REGIONS, "99"):
        size_rows = {
            row["profile"]: row
            for row in receipts
            if row["geography_id"] == region and row["table"] == "household_size"
        }
        mar, small = size_rows["mar"], size_rows["small_households"]
        assert math.fsum(small["allocated_counts"]) == pytest.approx(
            math.fsum(mar["allocated_counts"]), rel=1e-12, abs=1e-7
        )
        for index in range(2, 8):
            stated = small["allocated_counts"][index]
            mar_increment = mar["computed_residual"] * stated / mar["raw_leaf_sum"]
            assert mar["allocated_counts"][index] - small["allocated_counts"][
                index
            ] == pytest.approx(mar_increment, rel=1e-12, abs=1e-7)
        tenure_rows = [
            row
            for row in receipts
            if row["geography_id"] == region and row["table"] == "tenure"
        ]
        assert tenure_rows[0]["allocated_counts"] == tenure_rows[1]["allocated_counts"]


@pytest.mark.parametrize("profile", ["mar", "small_households"])
def test_real_artifact_regional_closure_shares_and_sensitivity(
    artifact, facts_by_key, profile
):
    rows = [
        row
        for row in _effective_rows(profile)
        if "households_by_size" in row["name"] or row["family"] == "housing_tenure"
    ]
    registry = compile_ledger_target_references(
        artifact.facts, _parsed(rows), country="nz"
    )
    actual = {}
    for row, target in zip(rows, registry.specs, strict=True):
        assert target.hierarchy.geography.id == row["ledger_selector"]["geography_id"]
        actual[row["name"]] = target.value
    for (
        region,
        total,
        size_sum,
        size_residual,
        tenure_sum,
        tenure_residual,
    ) in RESIDUALS:
        for table, expected_sum, residual in (
            ("household_size", size_sum, size_residual),
            ("tenure", tenure_sum, tenure_residual),
        ):
            selected = [
                row
                for row in rows
                if row["ledger_selector"]["geography_id"] == region
                and ("households_by_size" in row["name"]) == (table == "household_size")
            ]
            operands = {
                operand["role"]: operand for operand in selected[0]["value_operands"]
            }
            partition = operands["partition"]
            counts = [
                _matched(selector, facts_by_key)[0]["value"]
                for selector in partition["member_selectors"]
            ]
            source_total = _matched(
                {
                    key: value
                    for key, value in operands["total"].items()
                    if key != "role"
                },
                facts_by_key,
            )[0]["value"]
            assert source_total == total
            assert math.fsum(counts) == expected_sum
            assert source_total - math.fsum(counts) == residual
            num = _matched(
                {
                    key: value
                    for key, value in operands["numerator"].items()
                    if key != "role"
                },
                facts_by_key,
            )[0]["value"]
            den = _matched(
                {
                    key: value
                    for key, value in operands["denominator"].items()
                    if key != "role"
                },
                facts_by_key,
            )[0]["value"]
            ratio = num / den
            assert math.isclose(
                math.fsum(actual[row["name"]] for row in selected),
                total * ratio,
                rel_tol=1e-12,
                abs_tol=1e-7,
            )
            if table == "household_size":
                unbridged = [actual[row["name"]] / ratio for row in selected]
                for index, (stated, allocated) in enumerate(
                    zip(counts, unbridged, strict=True)
                ):
                    denominator = (
                        math.fsum(counts[:2])
                        if profile == "small_households"
                        else expected_sum
                    )
                    increment = (
                        residual * stated / denominator
                        if profile == "mar" or index < 2
                        else 0
                    )
                    assert allocated == pytest.approx(
                        stated + increment, rel=1e-12, abs=1e-7
                    )
                    if profile == "small_households" and index >= 2:
                        assert allocated == pytest.approx(stated, rel=1e-12, abs=1e-7)
                if profile == "mar":
                    for first in range(len(counts)):
                        for second in range(len(counts)):
                            assert unbridged[first] * counts[second] == pytest.approx(
                                unbridged[second] * counts[first], rel=1e-12, abs=1e-7
                            )
    assert math.fsum(
        value for name, value in actual.items() if "households_by_size" in name
    ) == pytest.approx(1822217.4461861975, rel=1e-12, abs=1e-7)


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
            assert "d1886" in row["metadata"]["parking_reason"]
        if "households_by_size" in row["name"] or row["family"] == "housing_tenure":
            assert (
                row["metadata"]["activation_status"]
                == "requires_reference_profile_resolution"
            )
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
                for row in (
                    _effective_rows()
                    if filename == "target_references.json"
                    else _rows(filename)
                )
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
            for selector in reference_fact_selectors(ref):
                resolved = _matched(selector, facts_by_key)
                assert resolved, (ref.name, selector)
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
    rows = [row for row in _effective_rows() if row["family"] == "housing_tenure"]
    registry = compile_ledger_target_references(
        artifact.facts, _parsed(rows), country="nz"
    )
    expected = []
    for row, spec in zip(rows, registry.specs, strict=True):
        base = math.fsum(
            fact["value"] for fact in _matched(row["ledger_selector"], facts_by_key)
        )
        operands = {operand["role"]: operand for operand in row["value_operands"]}
        num = _matched(
            {k: v for k, v in operands["numerator"].items() if k != "role"},
            facts_by_key,
        )[0]["value"]
        den = _matched(
            {k: v for k, v in operands["denominator"].items() if k != "role"},
            facts_by_key,
        )[0]["value"]
        total = _matched(
            {k: v for k, v in operands["total"].items() if k != "role"}, facts_by_key
        )[0]["value"]
        leaf_sum = math.fsum(
            _matched(selector, facts_by_key)[0]["value"]
            for selector in operands["partition"]["member_selectors"]
        )
        allocated = base + (total - leaf_sum) * base / leaf_sum
        assert spec.value == pytest.approx(allocated * (num / den), rel=1e-12, abs=1e-7)
        expected.append(allocated * (num / den))
    diagnostic = next(
        row
        for row in _document("reference_inventory.json")["national_bridge_diagnostics"]
        if row["table"] == "tenure"
    )
    assert math.fsum(spec.value for spec in registry.specs) == pytest.approx(
        math.fsum(expected), abs=1e-8
    )
    assert diagnostic["regional_published_total_sum_scaled"] == pytest.approx(
        math.fsum(expected), abs=1e-7
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
        for row in _effective_rows()
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
            operand["geography_id"] == "NZ"
            for operand in row["value_operands"]
            if operand["role"] in {"numerator", "denominator"}
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
