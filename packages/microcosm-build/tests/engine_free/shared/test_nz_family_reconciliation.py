"""Signed Census family reconciliation stays exact, regional and parked."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.calibration_columns import prepare_calibration_columns
from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.ledger_targets import compile_ledger_target_references
from microcosm.build.reference_profiles import resolve_reference_profiles
from microcosm.build.transport.target_kernels import parse_reference_document
from microcosm.calibrate import Target
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.frame.concepts import TENURE_DOMAIN
from microcosm.graph.canonical import canonical_json
from test_support.paths import paths_for

_SPEC = paths_for("microcosm-build").package / "src/microcosm/build/nz"
_CONTRACT = "stats_nz_census_families_by_type_region"
_DIMENSION = "nz_census_2023.family_type_source_label"
_SENSITIVITY = "family_discrepancy_positive_to_type3"
_PROFILES = ("central", _SENSITIVITY)
# Independent transcription of the hub's authenticated-feed arithmetic check.
# Region, published type leaves, published family total, ERP 2023, ERP 2025.
_SOURCE = (
    ("01", (22413, 16704, 9264), 48381, 198500, 201100),
    ("02", (151557, 201435, 73689), 426684, 1755200, 1816000),
    ("03", (54138, 52164, 22146), 128448, 516800, 532100),
    ("04", (37311, 33090, 15723), 86121, 346500, 351500),
    ("05", (4344, 4731, 3180), 12258, 52300, 52700),
    ("06", (18738, 16665, 8070), 43473, 179000, 179700),
    ("07", (14601, 13401, 5541), 33543, 129400, 130300),
    ("08", (28287, 24369, 11841), 64500, 257800, 260700),
    ("09", (57903, 57582, 21024), 136509, 538000, 543400),
    ("12", (4524, 2811, 1185), 8517, 33900, 34700),
    ("13", (78600, 71214, 24552), 174369, 676600, 698200),
    ("14", (30606, 23133, 7497), 61236, 248500, 253900),
    ("15", (12258, 10443, 3966), 26664, 102700, 104800),
    ("16", (8085, 5880, 1926), 15888, 59100, 59900),
    ("17", (6780, 5133, 2184), 14097, 54300, 54300),
    ("18", (7266, 4677, 1725), 13665, 50800, 50800),
    ("99", (78, 45, 24), 150, 5200000, 5324700),
)
_REGIONS = tuple(row[0] for row in _SOURCE)
_POLICY_KEYS = {"kind", "max_abs", "max_relative", "combine", "ruling", "disclosure"}
_DISCLOSURE = (
    "Census family-type counts in each region are scaled to that region's "
    "published family total. The published type counts and family totals are "
    "independently rounded to base 3, so their sums need not agree; the "
    "observed regional gaps are at most 3 families, and the scaling is a "
    "modelling adjustment for that rounding noise, not an observed category "
    "(Max ruling d1932)."
)


def _document(filename):
    return json.loads((_SPEC / filename).read_text())


def _family_rows(document):
    return [
        row
        for row in document["target_references"]
        if row["metadata"]["contract_target_id"] == _CONTRACT
    ]


def _resolve(profile="central", household_profile="mar"):
    return resolve_reference_profiles(
        _document("target_references.json"),
        _document("reference_profiles.json"),
        {
            "family_discrepancy": profile,
            "census_residual_allocation": household_profile,
            "aging_index": "central",
        },
        _document("reference_intervals.json"),
        factor_context=_document("reference_aging.json"),
    )


def _fraction(value):
    assert set(value) == {"numerator", "denominator"}
    assert type(value["numerator"]) is int
    assert type(value["denominator"]) is int and value["denominator"] > 0
    result = Fraction(value["numerator"], value["denominator"])
    assert result.numerator == value["numerator"]
    assert result.denominator == value["denominator"]
    return result


def _oracle(leaves, total, profile):
    discrepancy = total - sum(leaves)
    if profile == _SENSITIVITY and discrepancy > 0:
        return tuple(
            Fraction(value + (discrepancy if i == 2 else 0))
            for i, value in enumerate(leaves)
        )
    return tuple(Fraction(value * total, sum(leaves)) for value in leaves)


@pytest.fixture(scope="module")
def artifact():
    artifact_path = os.environ.get("NZ_CHRONICLE_ARTIFACT")
    if not artifact_path:
        pytest.skip("NZ_CHRONICLE_ARTIFACT is unset; public NZ feed audit unavailable")
    return load_ledger_consumer_artifact(
        Path(artifact_path), feed_pin=_document("chronicle_feed.json")
    )


@pytest.fixture(scope="module")
def compiled(artifact):
    result = {}
    for profile in _PROFILES:
        rows = _family_rows(_resolve(profile)[0])
        # Numerical auditing activates only copies. Packaged and effective rows
        # remain blocked on the G8-r3b household multiplicity classifier.
        executable = copy.deepcopy(rows)
        for row in executable:
            row["metadata"]["activation_status"] = "active"
        references = parse_reference_document(
            {"country": "nz", "target_references": executable}, country="nz"
        )
        registry = compile_ledger_target_references(
            artifact.facts, references, country="nz"
        )
        result[profile] = {
            row["name"]: (
                row,
                target,
                json.loads(target.metadata["ledger_allocation_receipt"]),
            )
            for row, target in zip(rows, registry.specs, strict=True)
        }
    return result


def test_family_policy_declares_either_exact_bound_and_the_full_disclosure():
    dimension = _document("reference_profiles.json")["dimensions"]["family_discrepancy"]
    assert dimension["default"] == "central"
    assert set(dimension["profiles"]) == set(_PROFILES)
    expected_names = {
        row["name"] for row in _family_rows(_document("target_references.json"))
    }
    for profile in _PROFILES:
        transforms = dimension["profiles"][profile]["transformations"]
        assert len(transforms) == 51
        assert {
            name for transform in transforms for name in transform["reference_names"]
        } == expected_names
        for transform in transforms:
            assert transform["operation"] == "allocated_residual_by_ratio"
            partition = transform["value_operands"][2]
            policy = partition["discrepancy_policy"]
            assert set(policy) == _POLICY_KEYS
            assert policy["kind"] == "signed_proportional_reconciliation"
            assert policy["max_abs"] == 3 and type(policy["max_abs"]) is int
            assert policy["max_relative"] == "0.001"
            assert policy["combine"] == "either"
            assert policy["ruling"] == "d1932"
            assert _DISCLOSURE in policy["disclosure"]
            assert "99" in policy["disclosure"] and "+3" in policy["disclosure"]
            assert "2.0%" in policy["disclosure"]


def test_exactly_51_regional_type_leaves_and_no_family_or_national_totals_enter_fit():
    raw = _document("target_references.json")
    family = _family_rows(raw)
    assert len(family) == 51
    assert {
        (
            row["ledger_selector"]["geography_id"],
            row["ledger_selector"]["dimension_values"][_DIMENSION],
        )
        for row in family
    } == {(region, code) for region in _REGIONS for code in (1, 2, 3)}
    for row in family:
        selector = row["ledger_selector"]
        assert selector["geography_level"] == "region"
        assert selector["geography_vintage"] == "unstated"
        assert selector["dimensions"] == [_DIMENSION]
        assert (
            row["measure"]
            == f"census_families_type_{selector['dimension_values'][_DIMENSION]}"
        )
    for filename in ("target_references.json", "precal_references.json"):
        for row in _document(filename)["target_references"]:
            selector = row["ledger_selector"]
            # Some selectors name no record set (precal's MBIE bond rents).
            record_set = selector.get("record_set_id", "")
            if record_set.endswith(".family_type"):
                assert selector["geography_id"] in _REGIONS
                assert selector["dimensions"] == [_DIMENSION]
            assert not record_set.endswith(".families")


@pytest.mark.parametrize("profile", _PROFILES)
def test_family_rows_stay_parked_and_exempt_until_g8r3b(profile):
    raw = _family_rows(_document("target_references.json"))
    effective = _family_rows(_resolve(profile)[0])
    for row in [*raw, *effective]:
        assert (
            row["metadata"]["activation_status"]
            == "requires_census_family_nucleus_standard"
        )
        assert row["metadata"]["exempt"] == "true"
        assert "G8-r3b" in row["metadata"]["parking_reason"]
        (reference,) = parse_reference_document(
            {"country": "nz", "target_references": [row]}, country="nz"
        )
        with pytest.raises(ValueError, match="non-executable placeholder"):
            compile_ledger_target_references([], [reference], country="nz")


def test_family_selection_is_independent_of_household_size_sensitivity():
    effective = {}
    for household_profile in ("mar", "small_households"):
        for profile in _PROFILES:
            document, _, receipt = _resolve(profile, household_profile)
            assert receipt["selection"]["family_discrepancy"] == profile
            rows = document["target_references"]
            effective[household_profile, profile] = rows
    identity_keys = {"reference_selection_sha256", "effective_intervals_sha256"}

    def semantics(rows):
        rows = copy.deepcopy(rows)
        for row in rows:
            for key in identity_keys:
                row["metadata"].pop(key, None)
        return rows

    for profile in _PROFILES:
        assert semantics(
            _family_rows({"target_references": effective["mar", profile]})
        ) == semantics(
            _family_rows({"target_references": effective["small_households", profile]})
        )
    for household_profile in ("mar", "small_households"):
        assert semantics(
            [
                row
                for row in effective[household_profile, "central"]
                if row["metadata"]["contract_target_id"] != _CONTRACT
            ]
        ) == semantics(
            [
                row
                for row in effective[household_profile, _SENSITIVITY]
                if row["metadata"]["contract_target_id"] != _CONTRACT
            ]
        )


@pytest.mark.parametrize("profile", _PROFILES)
def test_authenticated_feed_reconciles_to_family_totals_with_exact_signed_increments(
    compiled, artifact, profile
):
    by_key = {fact["aggregate_fact_key"]: fact for fact in artifact.facts}
    discrepancies = {}
    for region, leaves, total, denominator, numerator in _SOURCE:
        expected = _oracle(leaves, total, profile)
        q = Fraction(numerator, denominator)
        discrepancies[region] = total - sum(leaves)
        for index, published in enumerate(leaves, 1):
            name = f"{_CONTRACT}__r{region}__t{index}"
            row, target, receipt = compiled[profile][name]
            key = row["ledger_selector"]["aggregate_fact_key"]
            assert by_key[key]["value"] == published
            assert receipt["region"] == region
            assert _fraction(receipt["x"][key]) == published
            assert _fraction(receipt["F"]) == total
            assert _fraction(receipt["L"]) == sum(leaves)
            assert _fraction(receipt["R"]) == total - sum(leaves)
            assert (
                _fraction(receipt["increments"][key]) == expected[index - 1] - published
            )
            assert _fraction(receipt["reconciled"][key]) == expected[index - 1]
            assert _fraction(receipt["Q"]) == q
            assert _fraction(receipt["final_targets"][key]) == expected[index - 1] * q
            assert _fraction(receipt["final_target"]) == expected[index - 1] * q
            assert target.value == float(expected[index - 1] * q)
            assert target.hierarchy.geography.id == region
            assert (
                _fraction(receipt["growth_operands"]["numerator"]["value"]) == numerator
            )
            assert (
                _fraction(receipt["growth_operands"]["denominator"]["value"])
                == denominator
            )
            total_fact = by_key[receipt["fact_keys"]["total"]]
            assert total_fact["value"] == total
            assert total_fact["layout"]["record_set_id"].endswith(".families")
            assert total_fact["dimensions"] == {}
            assert receipt["operation"] == "allocated_residual_by_ratio"
            assert receipt["version"] == 2
            assert receipt["ruling"] == "d1932"
            assert receipt["selected_profile"] == profile
            assert receipt["policy"] == row["value_operands"][2]["discrepancy_policy"]
            assert target.metadata["ledger_allocation_receipt"] == json.dumps(
                receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        assert sum(expected) == total
    assert {region for region, value in discrepancies.items() if value == 3} == {
        "02",
        "05",
        "08",
        "13",
        "99",
    }
    assert {region for region, value in discrepancies.items() if value == -3} == {
        "04",
        "12",
        "15",
        "16",
        "18",
    }
    assert sum(value for value in discrepancies.values() if value > 0) == 15
    assert sum(value for value in discrepancies.values() if value < 0) == -15
    assert sum(discrepancies.values()) == 0


@pytest.mark.parametrize(
    ("profile", "printed"),
    [
        ("central", ("549923.187358", "556879.098527", "218615.140900")),
        (_SENSITIVITY, ("549916.657264", "556873.115175", "218627.654345")),
    ],
)
def test_family_type_totals_match_hub_printed_decimals_to_declared_absolute_1e_minus_6(
    compiled, profile, printed
):
    tolerance = Fraction("0.000001")
    totals = [Fraction(0), Fraction(0), Fraction(0)]
    for row, _, receipt in compiled[profile].values():
        index = row["ledger_selector"]["dimension_values"][_DIMENSION] - 1
        totals[index] += _fraction(receipt["final_target"])
    for actual, expected in zip(totals, printed, strict=True):
        assert abs(actual - Fraction(expected)) <= tolerance
    assert abs(sum(totals) - Fraction("1325417.426784")) <= tolerance


def test_sensitivity_allocates_positive_gaps_only_to_type3_and_keeps_negative_central(
    compiled,
):
    for region, leaves, total, _, _ in _SOURCE:
        central = []
        sensitivity = []
        for code in (1, 2, 3):
            name = f"{_CONTRACT}__r{region}__t{code}"
            for profile, values in (("central", central), (_SENSITIVITY, sensitivity)):
                row, _, receipt = compiled[profile][name]
                key = row["ledger_selector"]["aggregate_fact_key"]
                values.append(_fraction(receipt["reconciled"][key]))
        if total > sum(leaves):
            assert sensitivity == [
                Fraction(leaves[0]),
                Fraction(leaves[1]),
                Fraction(leaves[2] + total - sum(leaves)),
            ]
            assert sensitivity != central
        else:
            assert sensitivity == central
        assert sum(central) == sum(sensitivity) == total


@pytest.mark.parametrize("profile", _PROFILES)
def test_region_99_passes_absolute_bound_and_uses_national_erp_growth_once(
    compiled, profile
):
    for code in (1, 2, 3):
        row, _, receipt = compiled[profile][f"{_CONTRACT}__r99__t{code}"]
        assert receipt["bounds_passed"] == ["absolute"]
        assert _fraction(receipt["R"]) == 3
        assert _fraction(receipt["L"]) == 147
        assert _fraction(receipt["Q"]) == Fraction(5324700, 5200000)
        assert receipt["growth_geography_proxy"] == "national-growth proxy"
        assert "national-growth proxy" in row["hierarchy"]["target_label"]
        operands = {operand["role"]: operand for operand in row["value_operands"]}
        assert set(operands) == {
            "base",
            "total",
            "partition",
            "numerator",
            "denominator",
        }
        assert operands["numerator"]["geography_id"] == "NZ"
        assert operands["denominator"]["geography_id"] == "NZ"
        assert operands["numerator"]["period_value"] == 2025
        assert operands["denominator"]["period_value"] == 2023
        assert row["metadata"]["projection_from_june2025_to_model_period"] == "none"


def test_all_51_family_bindings_use_matching_regions_and_unavailable_type_measures():
    bindings = _document("calibration_bindings.json")
    household = pd.DataFrame(
        {
            "household_id": range(17),
            "household_region_code": _REGIONS,
            "tenure": [TENURE_DOMAIN[0]] * 17,
        }
    )
    person = pd.DataFrame(
        {
            "person_id": range(17),
            "person_household_id": range(17),
            "age": [30] * 17,
            "region_code": _REGIONS,
        }
    )
    tables = prepare_calibration_columns(
        {"person": person, "household": household}, bindings
    )
    frame = Frame(
        tables,
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(17), WeightKind.DESIGN)},
    )
    columns = {
        row["name"]: row for row in bindings["columns"] if row["entity"] == "household"
    }
    parked = {
        row["name"]: row
        for row in bindings["parked_columns"]
        if row.get("kind") == "household_multiplicity_measure"
    }
    explicit = {
        binding["reference_name"]: (measure, binding)
        for measure, declaration in parked.items()
        for binding in declaration["reference_bindings"]
    }
    assert len(explicit) == 51
    missing = set()
    for row in _family_rows(_resolve()[0]):
        region = row["ledger_selector"]["geography_id"]
        assert row["filter"] == f"region_{region}_indicator"
        assert columns[row["filter"]]["predicates"] == [
            {"column": "household_region_code", "operator": "eq", "value": region}
        ]
        assert tables["household"][row["filter"]].tolist() == [
            candidate == region for candidate in _REGIONS
        ]
        code = row["ledger_selector"]["dimension_values"][_DIMENSION]
        assert row["measure"] == f"census_families_type_{code}"
        assert parked[row["measure"]]["family_type_code"] == code
        assert "G8-r3b" in parked[row["measure"]]["reason"]
        bound_measure, binding = explicit[row["name"]]
        assert bound_measure == row["measure"]
        assert binding["filter"] == row["filter"]
        assert binding["region_predicate"] == columns[row["filter"]]["predicates"][0]
        assert binding["type_predicate"] == {
            "classification": "family_type",
            "operator": "eq",
            "value": code,
        }
        assert row["measure"] not in tables["household"]
        missing.add(row["measure"])
        target = Target(
            name=row["name"],
            entity="household",
            measure=row["measure"],
            filter=row["filter"],
        )
        with pytest.raises(KeyError, match=row["measure"]):
            target.constraint_row(frame)
    assert missing == {
        "census_families_type_1",
        "census_families_type_2",
        "census_families_type_3",
    }


def test_inventory_receipts_bind_published_facts_and_exact_regional_outputs(
    artifact, compiled
):
    facts = {fact["aggregate_fact_key"]: fact for fact in artifact.facts}
    inventory = _document("reference_inventory.json")
    receipts = {
        (row["geography_id"], row["profile"]): row
        for row in inventory["regional_family_reconciliations"]
    }
    assert set(receipts) == {
        (region, profile) for region in _REGIONS for profile in _PROFILES
    }
    for region, leaves, total, denominator, numerator in _SOURCE:
        for profile in _PROFILES:
            receipt = receipts[region, profile]
            expected = _oracle(leaves, total, profile)
            q = Fraction(numerator, denominator)
            assert receipt["x"] == {
                str(code): value for code, value in enumerate(leaves, 1)
            }
            assert (receipt["F"], receipt["L"], receipt["R"]) == (
                total,
                sum(leaves),
                total - sum(leaves),
            )
            assert receipt["ruling"] == "d1932"
            assert receipt["geography_vintage"] == "unstated"
            assert receipt["bounds_passed"] == (
                ["absolute"] if region == "99" else ["absolute", "relative"]
            )
            assert receipt["recipient_codes"] == (
                [3] if profile == _SENSITIVITY and total > sum(leaves) else [1, 2, 3]
            )
            assert receipt["growth_applications"] == 1
            assert _fraction(receipt["q"]) == q
            assert _fraction(receipt["final_closure"]) == total * q
            assert facts[receipt["raw_fact_keys"]["total"]]["value"] == total
            assert (
                receipt["raw_fact_keys"]["total"]
                == receipt["total_selector"]["aggregate_fact_key"]
            )
            for code, value in enumerate(expected, 1):
                key = receipt["raw_fact_keys"]["leaves"][str(code)]
                assert facts[key]["value"] == leaves[code - 1]
                assert (
                    _fraction(receipt["increments"][str(code)])
                    == value - leaves[code - 1]
                )
                assert _fraction(receipt["a"][str(code)]) == value
                assert _fraction(receipt["T"][str(code)]) == value * q
                _, _, compiler_receipt = compiled[profile][
                    f"{_CONTRACT}__r{region}__t{code}"
                ]
                assert receipt["policy"] == compiler_receipt["policy"]
                assert receipt["T"][str(code)] == compiler_receipt["final_targets"][key]
            assert (
                sum((_fraction(value) for value in receipt["a"].values()), Fraction())
                == total
            )
            assert (
                sum((_fraction(value) for value in receipt["T"].values()), Fraction())
                == total * q
            )
            for role, year, value in (
                ("numerator", 2025, numerator),
                ("denominator", 2023, denominator),
            ):
                operand = receipt["growth_operands"][role]
                assert operand["value"] == value
                assert operand["selector"]["period_value"] == year
                assert operand["selector"]["geography_id"] == (
                    "NZ" if region == "99" else region
                )
                assert (
                    facts[operand["selector"]["aggregate_fact_key"]]["value"] == value
                )


def test_national_type_facts_and_regional_differences_remain_independent_diagnostics(
    artifact,
):
    facts = {fact["aggregate_fact_key"]: fact for fact in artifact.facts}
    inventory = _document("reference_inventory.json")
    diagnostic = next(
        row
        for row in inventory["source_discrepancies"]
        if row["table"] == "family_type"
    )
    assert diagnostic["geography_vintage"] == "unstated"
    assert diagnostic["regional_type_sums"] == {"1": 537489, "2": 543477, "3": 213537}
    assert diagnostic["regional_minus_national_by_type"] == {"1": 3, "2": -6, "3": 3}
    assert diagnostic["published_national_total"] == 1294503
    assert diagnostic["published_national_leaf_sum"] == 1294503
    national = [
        facts[selector["aggregate_fact_key"]]
        for selector in diagnostic["national_leaf_selectors"]
    ]
    assert [fact["value"] for fact in national] == [537486, 543483, 213534]
    assert all(fact["geography"]["id"] == "NZ" for fact in national)
    # Roll-forward selectors (the tenure rows) name no aggregate fact key.
    selected_keys = {
        row["ledger_selector"]["aggregate_fact_key"]
        for row in _document("target_references.json")["target_references"]
        if "aggregate_fact_key" in row["ledger_selector"]
    }
    assert not selected_keys & {fact["aggregate_fact_key"] for fact in national}
    assert (
        diagnostic["published_national_total_selector"]["aggregate_fact_key"]
        not in selected_keys
    )
    check = inventory["family_reconciliation_totals"]["numerical_check"]
    assert check["absolute_tolerance"] == "0.000001"
    assert check["type_order"] == [1, 2, 3]


def test_family_receipts_are_byte_deterministic_and_digest_pinned():
    receipts = _document("reference_inventory.json")["regional_family_reconciliations"]
    assert len(receipts) == 34
    assert (
        hashlib.sha256(canonical_json(receipts)).hexdigest()
        == "8a83c526da697a1e47f6c273579792f3cb03c11678c2cb74158d035f7669d43b"
    )
