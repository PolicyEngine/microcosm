"""Exhaustive invariants for the NZ declaration hand-off to G8-w3b."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from functools import cache
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.calibration_predicates import predicate_mask
from microcosm.build.transport import compose
from microcosm.build.transport.column_kernels import validate_receipt_contract
from microcosm.build.transport.reference_selection import validate_exemptions
from microcosm.calibrate.ordered_kernels import _PARAMS
from test_support.paths import paths_for


@pytest.fixture(autouse=True)
def _prove_mutant_import():
    if prefix := os.environ.get("NZ_MUTANT_PATH"):
        loaded = Path(compose.__file__).resolve()
        assert loaded.is_relative_to(Path(prefix).resolve())
        print(f"Mutant loaded compose.__file__={loaded}")


@cache
def _pristine_spec():
    return compose.load_transport_spec("nz")


def _spec():
    return copy.deepcopy(_pristine_spec())


def _references(spec):
    for resource, document in spec["resources"].items():
        if "target_references" in document:
            yield resource, compose.transport_resource(spec, resource)


def _required(spec):
    result = {"mass_references": set()}
    result["mass_references"].update(
        row["name"] for row in spec["resources"]["mass_references"]["target_references"]
    )
    result["precal_references"] = set()
    for program in spec["resources"]["receipt_contract"]["receipts"]["programs"]:
        if "target" in program:
            result["precal_references"].add(program["target"])
        result["precal_references"].update(
            row["target"] for row in program.get("strata", ())
        )
    # A receipt target appears in both assignment and calibration surfaces.
    result["target_references"] = result["precal_references"]
    for resource, names in compose._reference_selection_closure(spec).items():
        result.setdefault(resource, set()).update(names)
    return result


def test_every_reference_is_active_or_exempt_with_matching_status():
    spec = _spec()
    selections = compose._reference_selections(spec)
    for resource, document in _references(spec):
        selection = selections.get(resource, {"exempt": []})
        exemptions = {
            row["name"]: row
            for row in validate_exemptions(document, selection, resource=resource)
        }
        for row in document["target_references"]:
            status = row["metadata"].get("activation_status", "active")
            # G8-r2c F: the coarse stand-in rows park on their measurement
            # prerequisite, unexempt, until composition proves its producer.
            if row["metadata"].get("required_measurement_binding"):
                # Active once G's producer meets A's proof; never exempt.
                assert status in {"requires_prerequisites", "active"}, row["name"]
                assert row["name"] not in exemptions, row["name"]
                continue
            assert (status == "active") != (row["name"] in exemptions), (
                resource,
                row["name"],
            )
            if row["name"] in exemptions:
                exemption = exemptions[row["name"]]
                assert exemption["activation_status"] == status
                assert exemption["reason"].strip()
                assert "G8-" in exemption["reason"]


def test_no_exemption_covers_a_required_reference():
    spec = _spec()
    required = _required(spec)
    for resource, selection in compose._reference_selections(spec).items():
        assert not required.get(resource, set()).intersection(
            row["name"] for row in selection["exempt"]
        ), resource


def test_receipt_targets_and_filters_resolve_to_active_bound_rows():
    spec = _spec()
    receipts = spec["resources"]["receipt_contract"]["receipts"]
    validate_receipt_contract(receipts)
    rows = {
        row["name"]: row
        for row in spec["resources"]["precal_references"]["target_references"]
    }
    predicates = {
        entry["name"]
        for entry in spec["resources"]["calibration_bindings"]["columns"]
        if "predicates" in entry
    }
    for program in receipts["programs"]:
        targets = program.get("strata", [{"target": program.get("target")}])
        for target in targets:
            row = rows[target["target"]]
            assert row["metadata"]["activation_status"] == "active"
            assert row["measure"] == program["output"]
            if "filter" in target:
                assert target["filter"] in predicates
                assert row["filter"] == target["filter"]
            else:
                assert not row.get("filter")


def test_receipt_strata_are_disjoint_for_every_age():
    spec = _spec()
    bindings = {
        entry["name"]: entry
        for entry in spec["resources"]["calibration_bindings"]["columns"]
        if "predicates" in entry
    }
    table = pd.DataFrame({"age": np.arange(0, 121)})
    for program in spec["resources"]["receipt_contract"]["receipts"]["programs"]:
        strata = program.get("strata", ())
        if not strata:
            continue
        intervals = []
        for row in strata:
            predicates = bindings[row["filter"]]["predicates"]
            assert len(predicates) == 2
            assert all(predicate["column"] == "age" for predicate in predicates)
            bounds = {
                predicate["operator"]: predicate["value"] for predicate in predicates
            }
            assert set(bounds) == {"ge", "lt"}
            intervals.append((bounds["ge"], bounds["lt"]))
        for index, (lower, upper) in enumerate(intervals):
            assert lower < upper
            for other_lower, other_upper in intervals[index + 1 :]:
                assert max(lower, other_lower) >= min(upper, other_upper)
        masks = [
            predicate_mask(
                table,
                bindings[row["filter"]]["predicates"],
                entity="person",
                name=row["filter"],
                validated_inputs={"age"},
            )
            for row in strata
        ]
        hits = np.stack(masks).sum(axis=0)
        assert np.all(hits <= 1), program["program"]
        assert np.array_equal(hits == 1, table.age.between(18, 64))
        assert program["outside_strata"] == "not_assigned"


def test_veterans_pension_stays_unsupported_and_exempt():
    spec = _spec()
    contract = spec["resources"]["receipt_contract"]["receipts"]
    assert "veterans_pension" in contract["unmodelled_programs"]
    assert all(row["program"] != "veterans_pension" for row in contract["programs"])
    selections = compose._reference_selections(spec)
    for resource, document in _references(spec):
        for row in document["target_references"]:
            if "veterans_pension" in row["name"]:
                assert row["metadata"]["activation_status"] != "active"
                assert row["name"] in {
                    entry["name"] for entry in selections[resource]["exempt"]
                }
    classifier = spec["resources"]["reporting_classifiers"]
    assert "receives_veterans_pension" not in json.dumps(classifier["receipt_group"])
    assert all(
        row.get("receipt_column") != "receives_veterans_pension"
        for row in classifier["income_decile"]["components"]
    )


def test_scenario_ids_agree_and_s4_appears_in_no_list():
    resources = _spec()["resources"]
    scenarios = {row["id"] for row in resources["scenarios"]["scenarios"]}
    assert "S4" not in scenarios
    assert (
        set(
            resources["reporting_classifiers"]["baseline_cell_contract"]["scenario_ids"]
        )
        == scenarios
    )
    required = [
        row["parameters"]["required_scenarios"]
        for row in resources["gates"]["gates"]
        if "required_scenarios" in row.get("parameters", {})
    ]
    assert required
    assert all(set(ids) == scenarios for ids in required)
    for name in ("scenarios", "gates", "reporting_classifiers"):
        assert '"S4"' not in json.dumps(resources[name])


def test_h1_counts_positive_rounded_payments_and_h2_annualises_dollars():
    resources = _spec()["resources"]
    rows = {
        row["metadata"]["holdout_use"]: row
        for row in resources["holdout_references"]["target_references"]
        if row["metadata"].get("activation_status") == "active"
    }
    h1 = rows["accommodation_supplement_recipient_comparator"]
    h2 = rows["accommodation_supplement_dollar_comparator"]
    assert h1["ledger_selector"]["geography_level"] == "country"
    assert h1["measure"] != h2["measure"]
    assert h1["measure"] == "accommodation_supplement_positive_payment_indicator"
    assert h1["metadata"]["indicator_input_column"] == (
        "accommodation_supplement_entitled_weekly_payment"
    )
    assert h1["metadata"]["indicator_operator"] == "positive"
    assert h1["metadata"]["indicator_threshold"] == "0"
    assert h2["measure"] == "accommodation_supplement_entitled_weekly_payment"
    assert h2["ledger_selector"]["record_set_id"]
    assert h2["metadata"]["source_period_start"] == "2024-07-01"
    assert str(h1["metadata"]["annualisation_factor"]) == "1"
    assert str(h2["metadata"]["annualisation_factor"]) == "52"
    measure = resources["scenarios"]["gap_conventions"]["reporting_cells"]["measure"]
    assert measure == "accommodation_supplement_entitled_weekly_payment"


def test_non_exempt_inactive_rows_still_name_their_activation_gap():
    spec = _spec()
    spec["resources"].pop("reference_selection")
    with pytest.raises(ValueError) as error:
        compose.validate_transport_activation(spec)
    message = str(error.value)
    for resource in ("target_references", "precal_references", "holdout_references"):
        for row in compose.transport_resource(spec, resource)["target_references"]:
            if row["metadata"].get("activation_status", "active") != "active":
                assert f"placeholder reference {resource}.{row['name']}" in message


def test_settings_export_and_receipt_profiles_declare_the_reviewed_scope():
    resources = _spec()["resources"]
    settings = resources["calibration_settings"]
    assert set(settings["reasons"]) == _PARAMS
    assert {key: settings[key] for key in _PARAMS} == {
        "epochs": 3000,
        "learning_rate": 0.15,
        "mass": "free",
        "max_weight_ratio": 5,
        "weight_anchor": "design",
    }
    assert "time_period" not in settings
    assert "positive median" in settings["cap_disclosure"]
    receipts = resources["receipt_contract"]["receipts"]
    assert [row["program"] for row in receipts["programs"]] == [
        "nz_super",
        "supported_living_payment",
        "sole_parent_support",
        "jobseeker_support",
    ]
    suffixes = [
        row["draw_key_suffix"] for row in receipts["assignment_profiles"].values()
    ]
    assert len(suffixes) == len(set(suffixes)) == 2
    export = resources["export_contract"]
    for program in receipts["programs"]:
        assert program["output"] in export["optional"]
        assert "assigns" in export["_assigned_receipt_inputs"][program["output"]]
    assert export["closed"] is True
    release = {row["gate"]: row for row in resources["gates"]["gates"]}
    for name in ("support", "release_input_coverage"):
        assert release[name]["criticality"] == "release_blocking"


@pytest.mark.skipif(
    not os.environ.get("NZ_CHRONICLE_ARTIFACT"), reason="Pinned feed not staged"
)
@pytest.mark.parametrize(
    "generator",
    [
        "generate_nz_target_references.py",
        "generate_nz_msd_pre_activation_reconciliation.py",
    ],
)
def test_declaration_generators_round_trip_through_check(generator):
    root = paths_for("microcosm-build").repository
    result = subprocess.run(
        [sys.executable, str(root / "tools" / generator), "--check"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "builder",
    [
        "build_nz_benefit_gross_up.py",
        "build_nz_entitlement_graph.py",
        "build_nz_transport_graph.py",
    ],
)
def test_packaged_graph_resources_match_their_builders(builder):
    # The transport builder reads the adapter's nz.json, so this also keeps the
    # transport kernels and the Axiom adapter on one concept mapping.
    root = paths_for("microcosm-build").repository
    result = subprocess.run(
        [sys.executable, str(root / "tools" / builder), "--check"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _instantiated_nz_rows(spec):
    from microcosm.build.transport.entitlement import _rows

    return [
        *spec["resources"]["transport_graph"]["nodes"],
        *[row for row, _ in _rows(spec)[0]],
    ]


def test_nz_every_variant_id_reaches_entitlement_rows_with_explicit_populations():
    from microcosm.build.transport.entitlement import _rows

    spec = _spec()
    rows, gaps = _rows(spec)
    variants = {
        row["id"]
        for row in spec["resources"]["scenarios"]["scenarios"]
        if row["tier"] == "calibration"
    }
    assert variants == {"V1", "V2", "V3", "V4", "VHF", "VSI", "VEX", "VPT"}
    ids = [row["id"] for row, _ in rows]
    assert len(ids) == len(set(ids))
    for variant in variants:
        prefix = f"nz.v.{variant}"
        branch = [
            row for row, scenario in rows if scenario and scenario["id"] == variant
        ]
        assert branch
        start = next(row for row in branch if row["id"] == prefix)
        assert start["structural"] == "filter"
        assert start["base"] == "nz.open"
        assert gaps[variant] == prefix + ".gap"
        for row in branch:
            if row.get("structural", "none") == "none":
                assert row["population"].startswith(prefix), row["id"]
            else:
                assert row.get("base"), row["id"]


def test_nz_s5_ancestry_keeps_as_holds_and_inventory_after_each_calibration():
    spec = _spec()
    compose.validate_transport_calibration_ancestry(spec)
    rows = _instantiated_nz_rows(spec)
    structural = {
        row["id"]: row for row in rows if row.get("structural", "none") != "none"
    }
    by_id = {row["id"]: row for row in rows}

    def lineage(population):
        result = []
        while population:
            result.append(population)
            population = structural[population].get("base")
        return result

    for scenario in spec["resources"]["scenarios"]["scenarios"]:
        variant = scenario["tier"] == "calibration"
        prefix = ("nz.v." if variant else "nz.scn.") + scenario["id"]
        calibration = prefix + ".calibrate" if variant else "nz.calibrate"
        for suffix in (".as", ".positive_payment", ".gap", ".bridge.base_rate"):
            row = by_id[prefix + suffix]
            assert calibration in lineage(row["population"]), row["id"]
        for row in rows:
            if row["kernel"] == "transport.package@1":
                assert "nz.calibrate" in lineage(row["population"])
        if variant:
            target = by_id[prefix + ".targets.compile"]
            assert target["params"]["references"]["resource"] == "target_references"
    listed = {
        stage["rules_binding"]
        for stage in spec["resources"]["precalibration_stages"]["stages"]
    }
    for row in rows:
        if row.get("structural", "none") != "none":
            assert row.get("rules_binding") not in listed
            text = json.dumps(row.get("params", {}))
            assert not any(f'"binding": "{name}"' in text for name in listed)


@pytest.mark.parametrize("encoding", ["json", "sha256"])
def test_nz_variant_holdout_selection_refuses_before_calibration(encoding):
    spec, config = _nz_composition_spec_and_config()
    variant = next(
        row for row in spec["resources"]["scenarios"]["scenarios"] if row["id"] == "V1"
    )
    target = next(
        row for row in variant["nodes"] if row["id"] == "nz.v.V1.targets.compile"
    )
    key = "references" if encoding == "json" else "references_sha256"
    target["params"][key] = {"resource": "holdout_references", "encoding": encoding}
    with pytest.raises(
        ValueError, match="nz.v.V1.targets.compile.*holdout_references selection"
    ):
        compose.compose_transport_graph(spec, config)


def test_nz_money_profile_selection_equals_each_scenarios_own_knobs():
    from microcosm.build.transport.entitlement import _parameter, _rows
    from microcosm.graph import Owned

    spec = _spec()
    config = compose.TransportGraphConfig(
        engine_refs={
            row["id"]: "fixture:" + row["id"]
            for row in spec["resources"]["axiom_rules_bindings"]["bindings"]
        },
        create_outputs=(Owned("person", "age", "int64"),),
    )
    for row, scenario in _rows(spec)[0]:
        if row["kernel"] != "money.normalize@1":
            continue
        parameters = {
            key: _parameter(spec, config, value, scenario)
            for key, value in row["params"].items()
        }
        resolved = compose._node(
            spec,
            row,
            config,
            {},
            resolved_params=parameters,
            profile_selection=scenario["knobs"],
        )
        assert json.loads(resolved.params["profile_selection"]) == scenario["knobs"], (
            row["id"]
        )
    central = next(
        row
        for row in spec["resources"]["transport_graph"]["nodes"]
        if row["kernel"] == "money.normalize@1"
    )
    assert (
        json.loads(central["params"]["profile_selection"])
        == spec["resources"]["scenarios"]["central_knobs"]
    )


def test_nz_hold_fixed_has_identity_families_in_its_own_normalization_scenario():
    spec = _spec()
    scenarios = spec["resources"]["scenarios"]
    row = next(row for row in scenarios["scenarios"] if row["id"] == "VHF")
    assert row["knobs"] == {"donor_money_index": "hold_fixed"}
    assert "hold_fixed" not in scenarios["knob_domains"]["aging_index"]
    normalizers = [
        node for node in row["nodes"] if node["kernel"] == "money.normalize@1"
    ]
    assert len(normalizers) == 1
    declaration = normalizers[0]["params"]["declaration"]["resource"]
    assert declaration == "money_normalization_hold_fixed"
    factors = spec["resources"][declaration]["index_family_factors"]
    assert set(factors) == {
        "mixed_income",
        "capital_income",
        "pensions",
        "owner_housing_costs",
        "consumer_prices",
    }
    assert all(family.get("identity") is True for family in factors.values())


def test_nz_only_s0_computes_deciles_and_every_other_scenario_carries_by_id():
    from microcosm.build.transport.entitlement import _rows

    spec = _spec()
    rows = _rows(spec)[0]
    classifiers = [
        (row, scenario)
        for row, scenario in rows
        if row["kernel"]
        in {"reporting.income_decile@1", "reporting.carry_income_decile@1"}
    ]
    baseline = [
        row
        for row, scenario in classifiers
        if row["kernel"] == "reporting.income_decile@1"
    ]
    assert [row["id"] for row in baseline] == ["nz.scn.S0.income_decile"]
    for row, scenario in classifiers:
        assert row["params"]["scenario"] == scenario["id"]
        if scenario["id"] == "S0":
            continue
        assert row["kernel"] == "reporting.carry_income_decile@1"
        assert row["artifact_inputs"] == [
            {
                "name": "baseline_cells",
                "producer": baseline[0]["id"],
                "artifact": "baseline_cells",
            }
        ]
        assert all(
            component["column"] not in item["columns"]
            for component in spec["resources"]["reporting_classifiers"][
                "income_decile"
            ]["components"]
            for item in row["inputs"]
        )
    assert (
        spec["resources"]["reporting_classifiers"]["income_decile"]["unit_id_column"]
        == "family_id"
    )


def test_nz_h1_has_a_positive_payment_producer_and_native_annualisation_factors():
    from microcosm.build.transport.entitlement import _rows

    spec = _spec()
    factors = spec["resources"]["gap_annualisation"]
    assert factors == {
        "msd_accommodation_supplement_recipients_national": 1,
        "msd_annual_report_accommodation_assistance_expenditure": 52,
    }
    assert all(type(value) is int for value in factors.values())
    rows = _rows(spec)[0]
    bindings = spec["resources"]["holdout_comparator_bindings"]
    # H25 and H26: payment > 0, the judgment holds, and outside S5 the unit is
    # not superannuitant.
    payment = {
        "column": "accommodation_supplement_rounded_weekly_payment",
        "operator": "gt",
        "value": 0,
    }
    holds = {
        "column": "accommodation_supplement_non_rate_entitlement_requirements",
        "operator": "eq",
        "value": 1,
    }
    receipt = {
        "column": "receipt_group",
        "operator": "in",
        "value": ["beneficiary", "other"],
    }
    (predicate,) = bindings["central"]["columns"]
    assert predicate["predicates"] == [payment, holds, receipt]
    (alternative,) = bindings["s5"]["columns"]
    assert alternative["predicates"] == [payment, holds]
    assert alternative["name"] == predicate["name"]
    for row, _ in rows:
        if row["kernel"] == "takeup.gap@1":
            assert (
                row["params"]["annualisation_factors"]["resource"]
                == "gap_annualisation"
            )
            positive = next(
                producer
                for producer, _ in rows
                if producer["id"]
                == row["id"].removesuffix(".gap") + ".positive_payment"
            )
            assert positive["kernel"] == "calibration.prepare@1"
            assert positive["population"] == row["population"]
            branch = "s5" if row["id"] == "nz.scn.S5.gap" else "central"
            assert positive["params"]["bindings"]["path"] == [branch]
            assert positive["outputs"] == [
                {"entity": "family", "column": predicate["name"], "dtype": "bool"}
            ]


def test_nz_packaged_declaration_never_rewrites_a_money_bridge_column_later():
    rows = _instantiated_nz_rows(_spec())
    versions = {
        row["id"]: row for row in rows if row.get("structural", "none") != "none"
    }
    bridge_outputs = []
    for index, row in enumerate(rows):
        if row["kernel"] in {"money.minor_units@1", "money.annual_transfers@1"}:
            bridge_outputs.extend(
                (index, row["population"], output["entity"], output["column"])
                for output in row["outputs"]
            )
            bridge_outputs.extend(
                (index, row["population"], item["entity"], column)
                for item in row.get("inputs", ())
                for column in item["columns"]
            )
    assert bridge_outputs
    for index, row in enumerate(rows):
        population = row.get("population", row["id"])
        lineage = []
        while population:
            lineage.append(population)
            population = versions[population].get("base")
        for output in row.get("outputs", ()):
            for owner_index, owner_version, entity, column in bridge_outputs:
                if (
                    owner_index < index
                    and owner_version in lineage
                    and (entity, column) == (output["entity"], output["column"])
                ):
                    assert not output.get("rewrite"), row["id"]


def test_nz_packaged_declaration_has_no_entrant_expand_that_supplies_bridge_columns():
    rows = _instantiated_nz_rows(_spec())
    bridges = {
        (output["entity"], output["column"])
        for row in rows
        if row["kernel"] in {"money.minor_units@1", "money.annual_transfers@1"}
        for output in row["outputs"]
    }
    bridges.update(
        (item["entity"], column)
        for row in rows
        if row["kernel"] in {"money.minor_units@1", "money.annual_transfers@1"}
        for item in row.get("inputs", ())
        for column in item["columns"]
    )
    assert bridges
    for row in rows:
        if row.get("structural") == "expand" and row.get("entrants") is True:
            assert not bridges.intersection(
                (output["entity"], output["column"])
                for output in row.get("outputs", ())
            ), row["id"]


def test_nz_executable_support_keeps_region99_separate_from_ta999():
    spec = _spec()
    resources = spec["resources"]
    support_node = next(
        row
        for row in resources["transport_graph"]["nodes"]
        if row["kernel"] == "geography.support_from_facts@1"
    )
    resource = support_node["params"]["definition"]["resource"]
    assert resource == "transport_atomic_geography_support"
    rows = resources[resource]["rows"]
    residual = [row for row in rows if row["codes"]["region_code"] == "99"]
    assert residual
    assert {row["codes"]["ta_code"] for row in residual} == {"067"}
    assert all(row["codes"]["ta_code"] != "999" for row in rows)
    assert all(
        row["target"] == "stats_nz_population_by_territorial_authority__ta067"
        for row in residual
    )


_MEASUREMENT = "ird_taxable_income_ty2027_v1"
_BRIDGE_AND_REGION = {
    ("person", "region_code"),
    ("family", "accommodation_supplement_base_rate_weekly_amount"),
    ("family", "accommodation_supplement_non_beneficiary_income_cutout_weekly_amount"),
}


def _nz_composition_spec_and_config():
    """Supply synthetic engine refs and CREATE's inventory.

    The packaged graph declares every money producer, and the packaged
    bindings carry the engine pins, so the compiler checks the complete
    packaged node and population structure with no synthetic node or pin. The
    synthetic engine refs certify no runtime result. CREATE's inventory holds
    canonical concepts and unit outputs only: no bridge or region column
    enters as a root.
    """
    from microcosm.graph import Owned

    spec = _spec()
    resources = spec["resources"]
    compose.validate_transport_activation(spec)

    rows = _instantiated_nz_rows(spec)
    produced = {
        (output["entity"], output["column"])
        for row in rows
        for output in row.get("outputs", ())
        if not output.get("rewrite")
    }
    needed = {
        (item["entity"], column)
        for row in rows
        for item in row.get("inputs", ())
        for column in item["columns"]
    } - produced
    assert needed.isdisjoint(_BRIDGE_AND_REGION), sorted(needed & _BRIDGE_AND_REGION)
    from microcosm.frame import EntitySchema
    from microcosm.frame.concepts import CONCEPTS, TransportRule
    from microcosm.frame.unit_construction import BenefitUnitRule

    kinds = {"int": "int64", "float": "float64", "bool": "bool", "str": "string"}
    inventory = {
        (item.entity, item.name): "Int64" if item.nullable else kinds[item.dtype]
        for item in CONCEPTS
        if item.transport is not TransportRule.DROP
    }
    rule = BenefitUnitRule.from_spec(resources["benefit_unit_rule"])
    schema = EntitySchema(group_entities=("household", rule.entity))
    automatic = {
        (schema.person_entity, schema.person_id_column),
        *(
            (schema.person_entity, schema.membership_column(group))
            for group in schema.group_entities
        ),
        *((group, schema.id_column(group)) for group in schema.group_entities),
    }
    # CLI preparation excludes these columns; frame projection carries them.
    # A node must not name them as data reads that require a column owner.
    needed -= automatic
    inventory.update(
        {
            ("household", "donor_support_stratum"): "string",
            (rule.entity, rule.household_column): "int64",
            (rule.entity, rule.head_column): "int64",
            (rule.entity, rule.source_household_column): "string",
        }
    )
    assert needed <= inventory.keys(), (
        "The declaration reads donor columns outside the canonical concepts and actual CREATE/unit outputs",
        sorted(needed - inventory.keys()),
    )
    config = compose.TransportGraphConfig(
        engine_refs={
            row["id"]: "synthetic:" + row["id"]
            for row in resources["axiom_rules_bindings"]["bindings"]
        },
        create_outputs=tuple(
            Owned(entity, column, inventory[entity, column])
            for entity, column in sorted(needed)
        ),
    )
    return spec, config


def test_nz_terminal_filter_keeps_the_skeletons_predecessors_and_ancestry():
    from microcosm.graph import compile_graph

    spec, config = _nz_composition_spec_and_config()
    extended = compile_graph(compose.compose_transport_graph(spec, config))
    bare = copy.deepcopy(spec)
    del bare["resources"]["entitlement_graph"]
    skeleton = compile_graph(compose.compose_transport_graph(bare, config))
    assert set(extended.predecessors["nz.terminal"]) == set(
        skeleton.predecessors["nz.terminal"]
    )
    # Export and the terminal gates descend from here, so no bridge column can.
    seen, frontier = set(), ["nz.terminal"]
    while frontier:
        fresh = set(extended.predecessors[frontier.pop()]) - seen
        seen |= fresh
        frontier += fresh
    assert seen <= set(skeleton.order)
    assert not {name for name in seen if ".bridge" in name}


def test_nz_packaged_graph_produces_region_and_bridge_columns_from_named_nodes():
    spec = _spec()
    rows = _instantiated_nz_rows(spec)
    producers = {}
    for row in rows:
        for output in row.get("outputs", ()):
            producers.setdefault((output["entity"], output["column"]), set()).add(
                row["kernel"]
            )
    assert producers["person", "region_code"] == {"geography.person_location@1"}
    assert producers["family", "accommodation_supplement_base_rate_weekly_amount"] == {
        "simulate.counterfactual@1"
    }
    assert producers[
        "family",
        "accommodation_supplement_non_beneficiary_income_cutout_weekly_amount",
    ] == {"simulate.solve_zero@1"}
    pending = " ".join(spec["resources"]["transport_graph"]["pending_outputs"])
    assert "region" not in pending and "Reg 17" not in pending


@cache
def _nz_composed():
    """NZ's packaged graph, composed and compiled once per process."""
    from microcosm.graph import compile_graph

    spec, config = _nz_composition_spec_and_config()
    graph = compose.compose_transport_graph(spec, config)
    return spec, config, graph, compile_graph(graph)


def test_nz_packaged_graph_compiles_with_only_pins_and_money_producers():
    _, config, _, compiled = _nz_composed()
    # No bridge or region root stays synthetic: CREATE declares none of them.
    assert _BRIDGE_AND_REGION.isdisjoint(
        (owned.entity, owned.column) for owned in config.create_outputs
    )
    assert "nz.package" in compiled.order
    assert "nz.scn.S0.income_decile" in compiled.order
    for variant in ("V1", "V2", "V3", "V4", "VHF"):
        assert f"nz.v.{variant}.calibrate" in compiled.order
        assert f"nz.v.{variant}.gap" in compiled.order


def test_every_composed_battery_keeps_the_d1982_release_candidate_off():
    """d1982: the batteries an extension branch adds keep the flag off too.

    test_nz_money_normalization.py checks the packaged battery. A battery
    that omits the flag fails here, since only an explicit ``False`` keeps
    it off.
    """
    _, _, graph, _ = _nz_composed()
    composed = {
        node.id: node.params.get("release_candidate")
        for node in graph.nodes
        if node.kernel == "gates.battery@1"
    }
    assert "nz.gates.terminal" in composed
    assert composed == dict.fromkeys(composed, False)


def test_only_the_composed_standins_declare_the_measurement():
    """R2C-1 across every branch: each calibration's one declarer is a stand-in.

    The income evidence readers name the measurement under ``measurement``,
    so the composed graph's ``measurement_id`` declarers are its
    ``measure.combine@1`` stand-ins alone, one per calibration branch.
    """
    _, _, graph, _ = _nz_composed()
    declarers = sorted(
        node.id for node in graph.nodes if "measurement_id" in node.params
    )
    assert declarers == sorted(
        node.id for node in graph.nodes if node.kernel == "measure.combine@1"
    )
    calibrations = [
        node for node in graph.nodes if node.kernel.startswith("calibrate.")
    ]
    assert len(declarers) == len(calibrations)
    assert {
        node.params["measurement_id"] for node in graph.nodes if node.id in declarers
    } == {_MEASUREMENT}


def test_every_calibration_node_has_its_own_standin_as_the_only_writer():
    """R2C-15 on every calibration node, the variant branches' included.

    Composition runs ``_check_measurement_proofs`` on the skeleton and again
    on the extended graph, where the declaration names no producer for
    ``nz.v.<V>.calibrate`` and the compiled graph has to prove one. This test
    checks NZ's graph directly: for each calibration node of the composed
    graph, exactly one compiled ancestor declares the measurement, it is that
    branch's stand-in, and it is the only ancestor whose outputs or EXPAND
    cells hold each proved column.
    """
    spec, _, graph, compiled = _nz_composed()
    proofs, refusals = compose.measurement_proofs(spec)
    assert refusals == {}
    columns = [tuple(name.split(".", 1)) for name in proofs[_MEASUREMENT]["columns"]]
    _, cells = compose._precalibration_expansions(graph)
    by_id = {node.id: node for node in graph.nodes}
    roots = sorted(
        node.id for node in graph.nodes if node.kernel.startswith("calibrate.")
    )
    variants = ("V1", "V2", "V3", "V4", "VEX", "VHF", "VPT", "VSI")
    assert roots == ["nz.calibrate", *(f"nz.v.{name}.calibrate" for name in variants)]
    for root in roots:
        branch = root.removesuffix(".calibrate")
        standin = "nz.income.standin" if branch == "nz" else f"{branch}.income.standin"
        ancestors = compose._ancestor_closure(compiled.predecessors, {root})
        declarers = sorted(
            name
            for name in ancestors
            if (by_id[name].params or {}).get(compose.MEASUREMENT_PARAM) == _MEASUREMENT
        )
        assert declarers == [standin], root
        for entity, column in columns:
            writers = sorted(
                name
                for name in ancestors
                if (entity, column) in cells[name]
                or any(
                    (output.entity, output.column) == (entity, column)
                    for output in by_id[name].outputs
                )
            )
            assert writers == [standin], (root, f"{entity}.{column}")


def test_nz_as_area_state_bindings_compare_the_declared_string_geography_codes():
    resources = _spec()["resources"]
    support = resources["transport_atomic_geography_support"]["rows"]
    codes = {row["codes"]["as_area"] for row in support}
    alternative = {row["codes"]["as_area_alt"] for row in support}
    assert codes == {"1", "2", "3", "4"}
    assert alternative <= codes
    for area in ("1", "2", "3"):
        entry = next(
            row
            for row in resources["axiom_input_closure"]["entries"]
            if row["input"] == f"accommodation_supplement_resides_in_area_{area}"
        )
        assert entry["state_binding"]["test"] == "equals"
        value = entry["state_binding"]["value"]
        assert type(value) is str
        assert value == area
        assert value in codes


_IMMUTABLE_PROFILE_VIEWS = {}


@pytest.fixture(autouse=True)
def _memoize_immutable_reference_profile_views(monkeypatch):
    """Reuse real resolved views by content while isolating each test's copy.

    This file leaves profile resources unchanged within a test. Each new spec
    gets a content digest of every input the resolver reads, so mutations in a
    later test select a different cache entry. Deep copies isolate returned
    views from the file-level immutable cache.
    """
    original = compose.reference_profile_view
    views = {}
    retained_specs = {}
    input_digests = {}

    def cached(spec, selection=None):
        if not compose._has_resource(spec, "reference_profiles"):
            return original(spec, selection)
        declarations = compose._raw_transport_resource(spec, "reference_profiles")
        dimensions = declarations.get("dimensions", {})
        selected = {
            name: value
            for name, value in (selection or {}).items()
            if name in dimensions
        }
        selected_json = json.dumps(selected, sort_keys=True)
        key = (id(spec), selected_json)
        retained_specs[id(spec)] = spec
        if id(spec) not in input_digests:
            names = {
                "reference_profiles",
                declarations.get("reference_resource", "target_references"),
                *(
                    declarations[field]
                    for field in ("interval_resource", "factor_context_resource")
                    if declarations.get(field) is not None
                ),
            }
            if compose._has_resource(spec, compose.CHRONICLE_FEED_RESOURCE):
                names.add(compose.CHRONICLE_FEED_RESOURCE)
            # G8-r2c: A's measurement proof reads the graph, so the resolved
            # view depends on it too.
            if compose._has_resource(spec, compose.GRAPH_RESOURCE):
                names.add(compose.GRAPH_RESOURCE)
            documents = {
                name: compose._raw_transport_resource(spec, name)
                for name in sorted(names)
            }
            input_digests[id(spec)] = sha256(
                json.dumps(documents, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        if key not in views:
            content_key = (input_digests[id(spec)], selected_json)
            if content_key not in _IMMUTABLE_PROFILE_VIEWS:
                _IMMUTABLE_PROFILE_VIEWS[content_key] = original(spec, selection)
            views[key] = copy.deepcopy(_IMMUTABLE_PROFILE_VIEWS[content_key])
        return views[key]

    monkeypatch.setattr(compose, "reference_profile_view", cached)


def test_nz_every_instantiated_encoder_output_declares_an_explicit_graph_dtype():
    from microcosm.graph import DTYPES

    encoders = {
        "concepts.encode@1",
        "concepts.encode_groups@1",
        "transport.scenario_override@1",
    }
    rows = [row for row in _instantiated_nz_rows(_spec()) if row["kernel"] in encoders]
    assert rows
    for row in rows:
        assert row["outputs"], row["id"]
        for output in row["outputs"]:
            assert output.get("dtype") in DTYPES, (row["id"], output)


def test_nz_export_slices_need_no_ids_because_compile_refuses_them():
    """_check_export_contract counts projected ids; no slice may name one.

    The executor projects ids and memberships for every sliced entity, so a
    slice that names one asks for a cell that no node owns.
    """
    from microcosm.frame import EntitySchema
    from microcosm.frame.unit_construction import BenefitUnitRule
    from microcosm.graph import GraphError

    spec, config = _nz_composition_spec_and_config()
    resources = spec["resources"]
    rule = BenefitUnitRule.from_spec(resources["benefit_unit_rule"])
    schema = EntitySchema(group_entities=("household", rule.entity))
    automatic = {
        schema.person_id_column,
        *(schema.membership_column(group) for group in schema.group_entities),
        *(schema.id_column(group) for group in schema.group_entities),
    }
    for roster in resources["export_contract"]["entity_slices"].values():
        assert not automatic & {column for names in roster.values() for column in names}
    for role in ("export", "gates"):
        resources["export_contract"]["entity_slices"][role]["household"].append(
            "household_id"
        )
    for row in resources["transport_graph"]["nodes"]:
        if row["id"] in {"nz.export.prepare", "nz.gates.terminal"}:
            for item in row["inputs"]:
                if item["entity"] == "household":
                    item["columns"].append("household_id")
    with pytest.raises(
        GraphError, match=r"reads household\.household_id, which no node owns"
    ):
        compose.compose_transport_graph(spec, config)


# ---------------------------------------------------------------------------
# G8-r2c E: the P1 wage map on scaled band edges
# ---------------------------------------------------------------------------

_WAGE_TABLE = "ird_wage_salary_national_ty2026"
_PROFILES = ("low", "central", "high")


def _earnings_sources():
    spec = _spec()
    resources = spec["resources"]
    (table,) = [
        row
        for row in resources["reference_intervals"]["tables"]
        if row["id"] == _WAGE_TABLE
    ]
    factors = {
        entry["receipt"]["factor_id"]: entry
        for entry in resources["reference_aging"]["factors"]
    }
    return resources, table, factors


def _branches(spec):
    """Each calibration branch's node list: the skeleton's and every variant's."""
    yield "nz", spec["resources"]["transport_graph"]["nodes"]
    for scenario in spec["resources"]["scenarios"]["scenarios"]:
        if scenario["tier"] == "calibration":
            yield f"nz.v.{scenario['id']}", scenario["nodes"]


def test_every_profiles_edges_equal_cut_times_factor_exactly():
    """Each edge is float(cut_cents × P1 factor / 100), rounded once from Fraction."""
    from fractions import Fraction

    resources, table, factors = _earnings_sources()
    document = resources["earnings_bands"]
    cuts = [table["first_lower_cut_cents"]] + [b["upper_cent"] for b in table["bands"]]
    # The September 2026 QES re-declaration gives the P1 factors as 1.028904,
    # 1.029853 and 1.030799 (design 80 gave 1.026690, 1.028580 and 1.030467).
    assert {
        profile: factors[f"p1_ty2026_to_ty2027_{profile}"]["operational_factor"]
        for profile in _PROFILES
    } == {
        "low": {"numerator": 1028904, "denominator": 1000000},
        "central": {"numerator": 1029853, "denominator": 1000000},
        "high": {"numerator": 1030799, "denominator": 1000000},
    }
    assert set(document["profiles"]) == set(_PROFILES)
    for profile in _PROFILES:
        entry = factors[f"p1_ty2026_to_ty2027_{profile}"]
        factor = Fraction(
            entry["operational_factor"]["numerator"],
            entry["operational_factor"]["denominator"],
        )
        declared = document["profiles"][profile]
        assert declared["factor_receipt_sha256"] == entry["receipt"]["receipt_sha256"]
        bands = declared["map_bands"]["bands"]
        assert [row["reference"] for row in bands] == [
            f"ird_wage_salary_people_by_band__{band['id']}" for band in table["bands"]
        ]
        edges = [bands[0]["lower"]] + [row["upper"] for row in bands]
        assert edges == [
            None if cut is None else float(Fraction(cut) * factor / 100) for cut in cuts
        ]
    # Design 80's worked band "$50,001 - $51,000" at the central factor.
    row_57 = document["profiles"]["central"]["map_bands"]["bands"][50]
    assert (row_57["lower"], row_57["upper"]) == (51492.65, 52522.503)


def test_wage_table_has_235_right_closed_bands_and_no_companions():
    _, table, _ = _earnings_sources()
    assert len(table["bands"]) == 235
    assert table["bands"][0]["id"] == "row_7" and table["bands"][-1]["id"] == "row_241"
    assert table["bands"][-1]["upper_cent"] is None
    assert table["bands"][-2]["upper_cent"] == 100_000_000
    assert (
        table["entity"],
        table["income_column"],
        table["positive_band_closure"],
        table["nil_band"],
        table["negative_income_policy"],
        table["first_lower_cut_cents"],
    ) == ("person", "employment_income_cents", "right", "absent", "refuse", 0)
    assert not any("count_reference" in band for band in table["bands"])


# The rows whose label repeats the previous row's upper endpoint ($20,000,
# $50,000 and $80,000). Right closure gives each shared endpoint to the lower band.
_WAGE_SHARED_ENDPOINT_ROWS = ["row_27", "row_57", "row_87"]


def _label_dollars(text):
    return 1_000_000 if text == "1 million" else int(text.replace(",", ""))


def test_each_wage_cut_follows_from_its_committed_label():
    """Each band's cents follow from its own label, with no feed.

    "$a - $b" is the right-closed band (a - 1, b] in dollars, so the cut below
    it is 100(a - 1) cents and its ``upper_cent`` is 100b. A label whose lower
    endpoint repeats the previous band's upper one starts at that cut instead,
    and only the three pinned rows do. "Over $1 million" is the open tail. The
    edges test above scales these cuts, and the feed audit checks the labels
    against the feed, which CI doesn't load.
    """
    _, table, _ = _earnings_sources()
    bands = table["bands"]
    cuts = [table["first_lower_cut_cents"]] + [band["upper_cent"] for band in bands]
    shared, previous = [], None
    for index, band in enumerate(bands[:-1]):
        label = " ".join(band["source_label"].split())
        match = re.fullmatch(r"\$([\d,]+) - \$([\d,]+|1 million)", label)
        assert match, (band["id"], label)
        lower, upper = (_label_dollars(text) for text in match.groups())
        if lower == previous:
            shared.append(band["id"])
            assert cuts[index] == 100 * lower, band["id"]
        else:
            assert cuts[index] == 100 * (lower - 1), band["id"]
        assert cuts[index + 1] == 100 * upper, band["id"]
        previous = upper
    assert shared == _WAGE_SHARED_ENDPOINT_ROWS
    assert bands[-1]["source_label"] == "Over $1 million"
    assert previous == 1_000_000 and cuts[-2:] == [100 * previous, None]


@pytest.mark.parametrize(
    "mutate",
    ["hand_edited_edge", "stale_factor_receipt", "stale_interval_table", "tail"],
)
def test_the_builder_check_refuses_each_stale_earnings_declaration(mutate):
    from tools import build_nz_transport_graph as builder

    resources, _, _ = _earnings_sources()
    document = copy.deepcopy(resources["earnings_bands"])
    intervals = copy.deepcopy(resources["reference_intervals"])
    aging = copy.deepcopy(resources["reference_aging"])
    builder.validate_earnings_bands(document, intervals, aging)
    if mutate == "hand_edited_edge":
        document["profiles"]["high"]["map_bands"]["bands"][100]["upper"] += 0.01
        match = "edges differ"
    elif mutate == "stale_factor_receipt":
        for entry in aging["factors"]:
            if entry["receipt"]["factor_id"] == "p1_ty2026_to_ty2027_low":
                entry["receipt"]["receipt_sha256"] = "0" * 64
        match = "stale factor_receipt_sha256"
    elif mutate == "stale_interval_table":
        (table,) = [t for t in intervals["tables"] if t["id"] == _WAGE_TABLE]
        table["source_locator"] += " (edited)"
        match = "stale wage interval table"
    else:
        document["tails"]["central"]["estimator"]["anchor_references"].pop(0)
        match = "tails differ"
    with pytest.raises(ValueError, match=match):
        builder.validate_earnings_bands(document, intervals, aging)


def test_the_rent_map_reads_the_region_and_precedes_the_encoder_on_every_branch():
    spec = _spec()
    for prefix, rows in _branches(spec):
        ids = [row["id"] for row in rows]
        by_id = {row["id"]: row for row in rows}
        mapper, money = by_id[f"{prefix}.rent.map"], by_id[f"{prefix}.money.normalize"]
        assert (
            ids.index(money["id"])
            < ids.index(mapper["id"])
            < ids.index(f"{prefix}.encode.person")
        )
        assert mapper["kernel"] == "transport.quantile_map@1"
        assert mapper["population"] == money["population"]
        assert mapper["inputs"] == [
            {"entity": "household", "columns": ["rent", "household_region_code"]}
        ]
        assert mapper["outputs"] == [
            {
                "entity": "household",
                "column": "rent",
                "dtype": "float64",
                "rewrite": True,
            }
        ]
        assert mapper["artifact_inputs"] == [
            {"name": "surface", "producer": "nz.facts.precal", "artifact": "surface"}
        ]
        assert mapper["params"]["component_column"] == "household_region_code"
        assert mapper["params"]["bands"] == {
            "resource": "rent_bands",
            "path": ["map_bands"],
            "encoding": "json",
        }


def test_each_branch_converts_the_liquid_asset_stock_at_one_node():
    """d1982 converts once, so a second writer on any branch is a defect."""
    spec = _spec()
    for prefix, rows in _branches(spec):
        writers = [
            row["id"]
            for row in rows
            for output in row.get("outputs", ())
            if output["column"] == "liquid_financial_assets"
        ]
        assert writers == [f"{prefix}.money.normalize"], prefix


def test_the_wage_map_and_cents_precede_the_person_encoder_on_every_branch():
    spec = _spec()
    selected = {"V1": "low", "V2": "high"}
    for prefix, rows in _branches(spec):
        ids = [row["id"] for row in rows]
        by_id = {row["id"]: row for row in rows}
        mapper, cents = (
            by_id[f"{prefix}.earnings.map"],
            by_id[f"{prefix}.earnings.cents"],
        )
        assert (
            ids.index(mapper["id"])
            < ids.index(cents["id"])
            < ids.index(f"{prefix}.encode.person")
        )
        assert mapper["kernel"] == "transport.quantile_map@2"
        assert mapper["population"] == by_id[f"{prefix}.money.normalize"]["population"]
        assert mapper["outputs"] == [
            {
                "entity": "person",
                "column": "employment_income",
                "dtype": "float64",
                "rewrite": True,
            }
        ]
        assert mapper["artifact_inputs"] == [
            {"name": "surface", "producer": "nz.facts.precal", "artifact": "surface"}
        ]
        variant = prefix.removeprefix("nz.v.") if prefix != "nz" else None
        assert mapper["params"]["bands"]["path"] == [
            "profiles",
            selected.get(variant, "central"),
            "map_bands",
        ]
        assert mapper["params"]["interpolation"]["path"] == [
            "tails",
            "vpt" if variant == "VPT" else "central",
        ]
        assert cents["kernel"] == "money.minor_units@1"
        assert json.loads(cents["params"]["declaration"])["columns"] == [
            {
                "output_column": "employment_income_cents",
                "quantum": "0.01",
                "rounding": "half_even",
                "source_column": "employment_income",
            }
        ]
        # The encoder reads the mapped wages from the same population.
        encode = by_id[f"{prefix}.encode.person"]
        assert any("employment_income" in item["columns"] for item in encode["inputs"])


def test_the_tails_anchor_on_the_r2c9_rows_with_the_declared_bracket():
    resources, table, _ = _earnings_sources()
    labels = {band["id"]: band["source_label"] for band in table["bands"]}
    tails = resources["earnings_bands"]["tails"]
    central = tails["central"]["estimator"]["anchor_references"]
    vpt = tails["vpt"]["estimator"]["anchor_references"]
    assert central == [
        f"ird_wage_salary_people_by_band__row_{row}" for row in range(231, 242)
    ]
    assert vpt == [
        f"ird_wage_salary_people_by_band__row_{row}" for row in range(227, 242)
    ]
    assert " ".join(labels["row_231"].split()) == "$500,001 - $550,000"
    assert " ".join(labels["row_227"].split()) == "$300,001 - $350,000"
    assert labels["row_241"] == "Over $1 million"
    for tail in tails.values():
        assert tail["method"] == "pareto"
        assert tail["estimator"]["id"] == "grouped_pareto_mle@1"
        assert tail["estimator"]["bracket"] == ["1.00000095367431640625", "64"]
        assert tail["estimator"]["iterations"] == 64


def test_r2c6_exports_employment_income_on_the_export_and_both_slices():
    contract = _spec()["resources"]["export_contract"]
    assert "employment_income" in contract["optional"]
    for role in ("export", "gates"):
        assert "employment_income" in contract["entity_slices"][role]["person"]


def test_the_wage_bands_reach_quantile_map_v2_as_a_bare_band_list():
    resources, _, _ = _earnings_sources()
    for profile in resources["earnings_bands"]["profiles"].values():
        assert set(profile["map_bands"]) == {"bands"}


def test_pending_outputs_no_longer_name_p1_earnings():
    pending = _spec()["resources"]["transport_graph"]["pending_outputs"]
    assert "Census rent and liquid-assets quantile producers" not in pending
    assert not any("P1" in entry or "earnings" in entry for entry in pending)


# ---------------------------------------------------------------------------
# G8-r2c G: the stand-in income measure, its variants, S5 and the gates
# ---------------------------------------------------------------------------

_TRANSFERS = (
    "jobseeker_support_annual_gross_transfer_cents",
    "sole_parent_support_annual_gross_transfer_cents",
    "supported_living_payment_annual_gross_transfer_cents",
    "nz_super_annual_gross_transfer_cents",
)
_MARKET = (
    "employment_income_cents",
    "nonfarm_self_employment_income_cents",
    "farm_self_employment_income_cents",
    "interest_income_cents",
    "dividend_income_cents",
    "rental_income_cents",
)
_PROXY = (
    "ird_taxable_income_proxy_cents",
    "ird_taxable_income_proxy_nzd",
    "ird_taxable_income_proxy_universe_status",
)


def test_the_measure_declarations_are_c_shaped_with_r2c10_floor_and_scale():
    from microcosm.build.transport.measure_kernels import validate_measure_declaration

    document = _spec()["resources"]["income_measurements"]
    assert document["measurement_id"] == _MEASUREMENT
    assert set(document["measures"]) == {"central", "vsi"}
    for entry in document["measures"].values():
        declaration = entry["declaration"]
        assert validate_measure_declaration(declaration) == declaration
        assert (
            entry["declaration_sha256"]
            == sha256(
                json.dumps(declaration, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        assert (declaration["entity"], declaration["floor"], declaration["scale"]) == (
            "person",
            0,
            100,
        )
        assert declaration["add_backs"] == ["rental_income_cents"]
        assert declaration["outputs"] == dict(
            zip(("measure", "unit", "universe_status"), _PROXY, strict=True)
        )
        assert all(row["sign"] == 1 for row in declaration["components"])
    central = [
        row["column"]
        for row in document["measures"]["central"]["declaration"]["components"]
    ]
    vsi = [
        row["column"]
        for row in document["measures"]["vsi"]["declaration"]["components"]
    ]
    assert central == [*_MARKET, *_TRANSFERS]
    # VSI omits exactly the four transfer addends.
    assert set(central) - set(vsi) == set(_TRANSFERS)
    assert set(vsi) <= set(central)


def test_the_omission_register_gives_no_unsourced_size():
    register = _spec()["resources"]["income_measurements"]["omission_register"]
    items = {row["item"]: row for row in register["items"]}
    assert {
        "private_pension_income",
        "veterans_pension",
        "acc_weekly_compensation",
    } <= set(items)
    # Design 80's percentages are the proposal's claims, with no repo source.
    assert {row["size"] for row in items.values()} == {"unknown"}


def test_every_calibration_branch_has_one_standin_between_transfers_and_prepare():
    spec = _spec()
    measures = {"VSI": "vsi"}
    for prefix, rows in _branches(spec):
        ids = [row["id"] for row in rows]
        by_id = {row["id"]: row for row in rows}
        standin = by_id[f"{prefix}.income.standin"]
        assert (
            ids.index(f"{prefix}.benefits.annual_transfers")
            < ids.index(standin["id"])
            < ids.index(f"{prefix}.targets.prepare")
        )
        assert standin["kernel"] == "measure.combine@1"
        assert standin["params"]["measurement_id"] == _MEASUREMENT
        variant = prefix.removeprefix("nz.v.") if prefix != "nz" else None
        measure = measures.get(variant, "central")
        assert standin["params"]["measure"]["path"] == [
            "measures",
            measure,
            "declaration",
        ]
        assert {(o["column"], o["dtype"]) for o in standin["outputs"]} == {
            (_PROXY[0], "Int64"),
            (_PROXY[1], "float64"),
            (_PROXY[2], "string"),
        }
        declared = spec["resources"]["income_measurements"]["measures"][measure]
        (inputs,) = standin["inputs"]
        assert inputs["columns"] == [
            row["column"] for row in declared["declaration"]["components"]
        ]
        prepare, problem = (
            by_id[f"{prefix}.targets.prepare"],
            by_id[f"{prefix}.targets.problem"],
        )
        read = {
            (item["entity"], column)
            for item in prepare["inputs"]
            for column in item["columns"]
        }
        assert {("person", _PROXY[0]), ("person", _PROXY[2])} <= read
        read = {
            (item["entity"], column)
            for item in problem["inputs"]
            for column in item["columns"]
        }
        # The problem slices every target's measure: the amount rows' unit
        # column, the band indicators' income and universe columns, and the
        # source households per-band support groups by.
        assert {
            ("person", _PROXY[0]),
            ("person", _PROXY[1]),
            ("person", _PROXY[2]),
            ("household", "source_household_id"),
        } <= read
        targets = spec["resources"]["target_references"]["target_references"]
        assert {
            ("person", row["measure"]) for row in targets if row["family"] == "income"
        } <= read | {("person", "people")}
        annual = json.loads(
            by_id[f"{prefix}.benefits.annual_transfers"]["params"]["declaration"]
        )
        assert annual["exposure"]["central"] == ("0.75" if variant == "VEX" else 1)


def test_vsi_vex_and_vpt_change_only_their_own_knob():
    document = _spec()["resources"]["scenarios"]
    rows = {row["id"]: row for row in document["scenarios"]}
    assert rows["VSI"]["knobs"] == {"income_measure": "vsi"}
    assert rows["VEX"]["knobs"] == {"annual_transfer_exposure": 0.75}
    assert rows["VPT"]["knobs"] == {"earnings_tail_anchor": "vpt"}
    assert {rows[name]["tier"] for name in ("VSI", "VEX", "VPT")} == {"calibration"}
    central = document["central_knobs"]
    assert (
        central["income_measure"],
        central["annual_transfer_exposure"],
        central["earnings_tail_anchor"],
    ) == ("central", 1, "central")
    domains = document["knob_domains"]
    assert domains["income_measure"] == ["central", "vsi"]
    assert domains["earnings_tail_anchor"] == ["central", "vpt"]
    assert "declared judgment" in domains["annual_transfer_exposure"]
    assert "declared judgment" in rows["VEX"]["description"]


def test_a_proves_nz_income_standin_the_measurements_only_producer():
    spec = _spec()
    proofs, refusals = compose.measurement_proofs(spec)
    assert refusals == {}
    assert proofs[_MEASUREMENT]["producers"] == {"nz.calibrate": "nz.income.standin"}
    assert proofs[_MEASUREMENT]["columns"] == [
        "person.ird_taxable_income_proxy_cents",
        "person.ird_taxable_income_proxy_universe_status",
    ]


def test_a_refuses_a_second_declared_producer_upstream_of_calibration():
    spec = _spec()
    rows = spec["resources"]["transport_graph"]["nodes"]
    standin = next(row for row in rows if row["id"] == "nz.income.standin")
    second = copy.deepcopy(standin)
    second["id"] = "nz.income.second"
    rows.insert(rows.index(standin) + 1, second)
    proofs, refusals = compose.measurement_proofs(spec)
    assert _MEASUREMENT not in proofs
    assert "2 nodes declaring measurement_id" in refusals[_MEASUREMENT]


def test_without_the_standin_the_24_rows_refuse_activation_by_name():
    spec = _spec()
    rows = spec["resources"]["transport_graph"]["nodes"]
    rows[:] = [row for row in rows if row["id"] != "nz.income.standin"]
    with pytest.raises(ValueError) as error:
        compose.validate_transport_activation(spec)
    message = str(error.value)
    assert (
        "placeholder reference target_references.ird_taxable_income_standin" in message
    )
    assert f"awaits measurement:{_MEASUREMENT}" in message


def test_the_nz_super_stage_lists_its_rules_outputs_for_r2c2():
    spec = _spec()
    stages = {
        stage["id"]: stage
        for stage in spec["resources"]["precalibration_stages"]["stages"]
    }
    stage = stages["nz.precal.nz_super"]
    assert stage["rules_binding"] == "nz_super"
    assert [(row["column"], row["purpose"]) for row in stage["outputs"]] == [
        ("entitled_to_new_zealand_superannuation", "classification"),
        ("nz_super_weekly_amount_before_tax", "calibration_measure"),
    ]
    assert [row["column"] for row in stage["derived_outputs"]] == [
        "nz_super_weekly_amount_before_tax_cents"
    ]
    # The stand-in itself stays off S5: no stage lists its columns.
    listed = json.dumps(stages)
    assert not any(column in listed for column in _PROXY)


def test_the_s5_guard_refuses_an_undeclared_nz_super_derived_output():
    from microcosm.graph import compile_graph

    spec, config = _nz_composition_spec_and_config()
    compile_graph(compose.compose_transport_graph(spec, config))
    stage = next(
        stage
        for stage in spec["resources"]["precalibration_stages"]["stages"]
        if stage["id"] == "nz.precal.nz_super"
    )
    stage["derived_outputs"] = []
    with pytest.raises(ValueError, match="undeclared pre-calibration stage output"):
        compose.compose_transport_graph(spec, config)


def test_measurement_prefit_carries_r2c3s_frozen_numbers_and_six_aliases():
    from test_support.microcosm_build.reference_intervals import R2C3_PREFIT_THRESHOLDS

    gates = {row["id"]: row for row in _spec()["resources"]["gates"]["gates"]}
    entry = gates["measurement_prefit"]
    assert (entry["gate"], entry["phase"], entry["criticality"]) == (
        "measurement_prefit",
        "terminal",
        "release_blocking",
    )
    parameters = entry["parameters"]
    assert parameters["roster_total_bounds"] == [0.75, 1.25]
    assert parameters["band_bounds"] == [0.5, 2.0]
    assert parameters["maximum_discrepant_bands"] == 2
    assert parameters["minimum_source_households"] == 30
    assert parameters["minimum_kish_ess"] == 30
    assert {
        key: value for key, value in parameters.items() if key in R2C3_PREFIT_THRESHOLDS
    } == R2C3_PREFIT_THRESHOLDS
    assert parameters["tables"] == ["ird_taxable_income_standin_ty2024"]
    assert parameters["required_evidence"] == [
        "native_band_report",
        "omission_register",
        "scenario_vsi",
        "scenario_vex",
        "scenario_vpt",
        "identity_receipts",
    ]


def test_the_prefit_table_has_12_bands_each_with_both_companions():
    (table,) = [
        row
        for row in _spec()["resources"]["reference_intervals"]["tables"]
        if row["id"] == "ird_taxable_income_standin_ty2024"
    ]
    assert len(table["bands"]) == 12
    assert all(
        band["count_reference"] and band["amount_reference"] for band in table["bands"]
    )
    assert table["nil_band"] == "absent"
    assert table["diagnostics"]["support_household_entity"] == "household"


#: Each income alias's producer and population. All three sit at nz.terminal,
#: after the fit, so their params and kernel sources stay out of
#: nz.calibrate's key and none of them can precede a calibration (R2C-1).
_INCOME_EVIDENCE = {
    "identity_receipts": ("nz.evidence.income_identity", "nz.terminal"),
    "omission_register": ("nz.evidence.omission_register", "nz.terminal"),
    "native_band_report": ("nz.evidence.native_bands", "nz.terminal"),
}


def test_the_income_evidence_readers_name_the_measurement_but_dont_declare_it():
    """R2C-1: the stand-ins are the only nodes that declare measurement_id.

    measurement_proofs counts every node upstream of a calibration that
    declares the measurement's id and needs exactly one. The three readers
    name the stand-in's id under ``measurement``, selected from
    income_measurements' own ``measurement_id``. The stand-in keeps the literal
    that R2C-1's proof reads, and the composed readers resolve to it. The
    register reads the problem's household weights and checks the frame's
    against the fit's solution. The readers sit two boundary filters after the
    reweight, and only stand-ins write the three proxy columns, so nz.terminal
    holds what nz.income.standin wrote at nz.central.
    """
    spec = _spec()
    rows = spec["resources"]["transport_graph"]["nodes"]
    declarers = [row["id"] for row in rows if "measurement_id" in row.get("params", {})]
    assert declarers == [
        row["id"] for row in rows if row["kernel"] == "measure.combine@1"
    ]
    assert declarers == ["nz.income.standin"]
    by_id = {row["id"]: row for row in rows}
    assert by_id["nz.income.standin"]["params"]["measurement_id"] == _MEASUREMENT
    selection = {
        "resource": "income_measurements",
        "path": ["measurement_id"],
        "encoding": "value",
    }
    _, _, graph, _ = _nz_composed()
    for producer, _ in _INCOME_EVIDENCE.values():
        assert by_id[producer]["params"]["measurement"] == selection, producer
        assert graph.node(producer).params["measurement"] == _MEASUREMENT, producer
    register = by_id["nz.evidence.omission_register"]
    assert register["params"]["weight_entity"] == "household"
    assert [
        (edge["name"], edge["producer"]) for edge in register["artifact_inputs"]
    ] == [("problem", "nz.targets.problem"), ("solution", "nz.calibrate")]
    for name, base in (("nz.as", "nz.calibrate"), ("nz.terminal", "nz.as")):
        assert (by_id[name]["kernel"], by_id[name]["base"]) == (
            "transport.boundary@1",
            base,
        )
    instantiated = _instantiated_nz_rows(spec)
    measured = {("person", name) for name in _PROXY}
    assert {
        row["id"]
        for row in instantiated
        for output in row.get("outputs", ())
        if (output["entity"], output["column"]) in measured
    } == {row["id"] for row in instantiated if row["kernel"] == "measure.combine@1"}


def test_r2c18_wires_each_income_alias_from_its_real_producer():
    """R2C-29: every measurement_prefit alias reaches the battery from its producer.

    The three income aliases come from the packaged graph's own evidence nodes,
    each under its producer's artifact name. The three variant aliases come
    from the entitlement extension's gap nodes, so only the composed graph
    holds them.
    """
    from microcosm.graph import compile_graph

    rows = {row["id"]: row for row in _spec()["resources"]["transport_graph"]["nodes"]}
    edges = {
        edge["name"]: (edge["producer"], edge["artifact"])
        for edge in rows["nz.gates.terminal"]["artifact_inputs"]
    }
    for alias, (producer, population) in _INCOME_EVIDENCE.items():
        assert edges[alias] == (producer, alias)
        assert rows[producer]["population"] == population
        assert [output["name"] for output in rows[producer]["artifact_outputs"]] == [
            alias
        ]
    assert not edges.keys() & {"scenario_vsi", "scenario_vex", "scenario_vpt"}
    spec, config = _nz_composition_spec_and_config()
    graph = compose.compose_transport_graph(spec, config)
    composed = {
        edge.name: (edge.producer, edge.artifact)
        for edge in graph.node("nz.gates.terminal").artifact_inputs
    }
    for scenario in ("VSI", "VEX", "VPT"):
        assert composed[f"scenario_{scenario.lower()}"] == (
            f"nz.v.{scenario}.gap",
            "gap",
        )
    gates = {row["id"]: row for row in spec["resources"]["gates"]["gates"]}
    required = gates["measurement_prefit"]["parameters"]["required_evidence"]
    assert len(required) == 6
    assert set(required) <= set(composed)
    compile_graph(graph)


def test_the_identity_receipts_name_the_kernels_identities_in_order():
    from microcosm.build.transport.income_evidence_kernels import INCOME_IDENTITIES

    spec = _spec()
    document = spec["resources"]["income_measurements"]["identity_receipts"]
    assert set(document) == {"description", "identities"}
    assert [row["id"] for row in document["identities"]] == list(INCOME_IDENTITIES)
    assert all(row["statement"] for row in document["identities"])
    pending = " ".join(spec["resources"]["transport_graph"]["pending_outputs"])
    assert "R2C-18" not in pending
    assert "measurement_prefit" not in pending


def test_the_native_band_node_reads_both_ird_tables_on_the_standin_columns():
    spec = _spec()
    rows = {row["id"]: row for row in spec["resources"]["transport_graph"]["nodes"]}
    node = rows["nz.evidence.native_bands"]
    tables = {
        row["id"]: row for row in spec["resources"]["reference_intervals"]["tables"]
    }
    native = tables[node["params"]["native_table"]]
    coarse = tables[node["params"]["coarse_table"]]
    assert native["id"] != coarse["id"]
    assert len(native["bands"]) == 237
    assert len(coarse["bands"]) == 12
    for field in ("income_column", "universe_column", "source_period", "currency"):
        assert native[field] == coarse[field]
    sliced = {
        (item["entity"], column)
        for item in node["inputs"]
        for column in item["columns"]
    }
    assert sliced == {
        ("person", native["income_column"]),
        ("person", native["universe_column"]),
        ("household", "source_household_id"),
    }
    assert node["sources"] == ["nz_chronicle_facts"]
    assert [edge["producer"] for edge in node["artifact_inputs"]] == [
        "nz.targets.prepare",
        "nz.targets.problem",
        "nz.calibrate",
    ]


def test_the_composed_preparation_carries_both_ird_tables_for_the_native_report():
    """run() reads both tables from the preparation's effective intervals.

    The composer hands nz.targets.prepare the effective interval document,
    which keeps every declared table and scales the ones a profile names. The
    native kernel ages the national table by the stand-in table's own factor,
    so the national table must arrive unscaled and the stand-in scaled.
    """
    spec, config = _nz_composition_spec_and_config()
    node = _spec()["resources"]["transport_graph"]["nodes"]
    (native_node,) = [row for row in node if row["id"] == "nz.evidence.native_bands"]
    graph = compose.compose_transport_graph(spec, config)
    intervals = json.loads(graph.node("nz.targets.prepare").params["intervals"])
    effective = {table["id"]: table for table in intervals["tables"]}
    native = effective[native_node["params"]["native_table"]]
    coarse = effective[native_node["params"]["coarse_table"]]
    assert "scaling" not in native
    assert "scaling" in coarse
    # A binding column reads the stand-in table's bands, so the preparation
    # embeds the effective intervals the report reads.
    bindings = spec["resources"]["calibration_bindings"]
    assert coarse["id"] in {
        row["interval"]["table"]
        for row in bindings["columns"]
        if isinstance(row.get("interval"), dict)
    }


def test_cell_support_gates_require_the_three_new_variants_and_income_lists_hold():
    gates = {row["id"]: row for row in _spec()["resources"]["gates"]["gates"]}
    for name in ("reported_cell_support", "reported_cell_support_target_a"):
        assert {"VSI", "VEX", "VPT"} <= set(
            gates[name]["parameters"]["required_scenarios"]
        )
    families = [
        "demography",
        "household_structure",
        "housing_tenure",
        "income",
        "main_benefits",
        "superannuation",
    ]
    assert gates["calibration_per_family_fit"]["parameters"]["families"] == families
    assert (
        gates["national_and_region_admin_aggregates"]["parameters"]["families"]
        == families
    )
    assert (
        gates["target_profile_coverage"]["parameters"]["required_families"] == families
    )
    assert gates["calibration_per_family_fit"]["parameters"]["within"] == 0.05


def test_vsi_review_triggers_and_the_wage_earner_count_stay_diagnostic():
    document = _spec()["resources"]["income_measurements"]
    triggers = document["review_triggers"]
    assert triggers["scenario"] == "VSI"
    assert [
        row.get("relative_change_above", row.get("share_above"))
        for row in triggers["triggers"]
    ] == [
        "0.10",
        "0.10",
        "0.05",
    ]
    assert document["diagnostics"]["wage_earner_count"]["ruling"] == "R2C-8"
    gates = json.dumps(_spec()["resources"]["gates"])
    assert "wage_earner" not in gates


@pytest.mark.parametrize(
    "column",
    ["entitled_to_new_zealand_superannuation", "nz_super_weekly_amount_before_tax"],
)
def test_the_s5_guard_refuses_a_reader_of_a_non_public_nz_super_column(column):
    # takeup.assign reads both rules outputs of nz.benefits.super; once the
    # stage stops listing one, that column is not public and its reader refuses.
    spec, config = _nz_composition_spec_and_config()
    stage = next(
        stage
        for stage in spec["resources"]["precalibration_stages"]["stages"]
        if stage["id"] == "nz.precal.nz_super"
    )
    stage["outputs"] = [row for row in stage["outputs"] if row["column"] != column]
    with pytest.raises(ValueError, match="undeclared pre-calibration stage output"):
        compose.compose_transport_graph(spec, config)


def test_every_prepared_table_on_the_standins_columns_has_no_nil_band():
    # C excludes every record at x <= 0 (R2C-14), so a nil band on a table C
    # feeds would never count them. Only declared-and-validated tables, with no
    # companions and no prepared band indicator, may keep IRD's nil band.
    spec = _spec()
    resources = spec["resources"]
    prepared = {
        entry["interval"]["table"]
        for entry in resources["calibration_bindings"]["columns"]
        if "interval" in entry
    }
    fed = [
        table
        for table in resources["reference_intervals"]["tables"]
        if (table["income_column"], table["universe_column"]) == (_PROXY[0], _PROXY[2])
    ]
    assert "ird_taxable_income_standin_ty2024" in prepared
    for table in fed:
        companions = any(
            band.get("count_reference") or band.get("amount_reference")
            for band in table["bands"]
        )
        if table["id"] in prepared or companions:
            assert table["nil_band"] == "absent", table["id"]
        else:
            assert table["nil_band"] == "present", table["id"]


_cents = st.integers(min_value=-(10**11), max_value=10**11)


@settings(max_examples=150, deadline=None)
@given(
    rows=st.lists(
        st.tuples(*([_cents] * len(_MARKET)), *([st.integers(0, 10**10)] * 4)),
        min_size=1,
        max_size=12,
    )
)
def test_property_vsi_is_the_central_measure_less_its_four_transfers(rows):
    from microcosm.build.transport.measure_kernels import combine_measure

    # combine_measure only reads the declarations, so the shared copy serves.
    measures = _pristine_spec()["resources"]["income_measurements"]["measures"]
    columns = {
        name: pd.Series([row[index] for row in rows], dtype="Int64")
        for index, name in enumerate((*_MARKET, *_TRANSFERS))
    }
    weights = np.ones(len(rows))
    central, central_status, _ = combine_measure(
        measures["central"]["declaration"], columns, weights
    )
    vsi, vsi_status, _ = combine_measure(
        measures["vsi"]["declaration"], columns, weights
    )
    for index, row in enumerate(rows):
        market, transfers = row[: len(_MARKET)], row[len(_MARKET) :]
        rental = market[_MARKET.index("rental_income_cents")]
        # The rental add-back counts rental profit as max(0, r) in both.
        assert vsi[index] == sum(market) + max(0, -rental)
        assert central[index] - vsi[index] == sum(transfers)
        for value, status in ((central, central_status), (vsi, vsi_status)):
            assert status[index] == ("included" if value[index] > 0 else "excluded")
