"""Exhaustive invariants for the NZ declaration hand-off to G8-w3b."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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


def _spec():
    return compose.load_transport_spec("nz")


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
        "accommodation_supplement_rounded_weekly_payment"
    )
    assert h1["metadata"]["indicator_operator"] == "positive"
    assert h1["metadata"]["indicator_threshold"] == "0"
    assert h2["measure"] == "accommodation_supplement_rounded_weekly_payment"
    assert h2["ledger_selector"]["record_set_id"]
    assert h2["metadata"]["source_period_start"] == "2024-07-01"
    assert str(h1["metadata"]["annualisation_factor"]) == "1"
    assert str(h2["metadata"]["annualisation_factor"]) == "52"
    measure = resources["scenarios"]["gap_conventions"]["reporting_cells"]["measure"]
    assert measure == "accommodation_supplement_rounded_weekly_payment"


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
    assert all(settings[key] is not None for key in _PARAMS)
    assert "time_period" not in settings
    assert settings["max_weight_ratio"] > 0
    assert settings["weight_anchor"] == "design"
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
