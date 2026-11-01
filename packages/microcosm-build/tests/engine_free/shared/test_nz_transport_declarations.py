"""Exhaustive invariants for the NZ declaration hand-off to G8-w3b."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from functools import cache
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    assert variants == {"V1", "V2", "V3", "V4", "VHF"}
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


_BRIDGE_AND_REGION = {
    ("person", "region_code"),
    ("family", "accommodation_supplement_base_rate_weekly_amount"),
    ("family", "accommodation_supplement_non_beneficiary_income_cutout_weekly_amount"),
}


def _nz_composition_spec_and_config():
    """Supply compiler-only toy pins, the three money producers and CREATE's inventory.

    The synthetic quantile declarations certify no distribution or runtime
    result. They only remove the three deliberate money-producer gaps while
    the compiler checks the complete packaged node and population structure.
    CREATE's inventory holds canonical concepts and unit outputs only: no
    bridge or region column enters as a root.
    """
    from microcosm.graph import Owned

    spec = _spec()
    resources = spec["resources"]
    resources["axiom_rules_bindings"]["engine"].update(
        commit="1" * 40, wheel_sha256="2" * 64
    )

    def add_quantiles(rows):
        money = next(row for row in rows if row["kernel"] == "money.normalize@1")
        at = rows.index(money) + 1
        synthetic = []
        for entity, columns in resources["money_normalization"][
            "quantile_mapped"
        ].items():
            for column in columns:
                synthetic.append(
                    {
                        "id": money["id"] + ".synthetic_quantile." + column,
                        "kernel": "transport.quantile_map@1",
                        "population": money["population"],
                        "inputs": [{"entity": entity, "columns": [column]}],
                        "outputs": [
                            {
                                "entity": entity,
                                "column": column,
                                "dtype": "float64",
                                "rewrite": True,
                            }
                        ],
                        "params": {},
                    }
                )
        rows[at:at] = synthetic

    add_quantiles(resources["transport_graph"]["nodes"])
    for scenario in resources["scenarios"]["scenarios"]:
        if scenario["tier"] == "calibration":
            add_quantiles(scenario["nodes"])
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


def test_nz_packaged_graph_compiles_with_only_pins_and_money_producers():
    from microcosm.graph import compile_graph

    spec, config = _nz_composition_spec_and_config()
    # No bridge or region root stays synthetic: CREATE declares none of them.
    assert _BRIDGE_AND_REGION.isdisjoint(
        (owned.entity, owned.column) for owned in config.create_outputs
    )
    graph = compose.compose_transport_graph(spec, config)
    compiled = compile_graph(graph)
    assert "nz.package" in compiled.order
    assert "nz.scn.S0.income_decile" in compiled.order
    for variant in ("V1", "V2", "V3", "V4", "VHF"):
        assert f"nz.v.{variant}.calibrate" in compiled.order
        assert f"nz.v.{variant}.gap" in compiled.order


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
