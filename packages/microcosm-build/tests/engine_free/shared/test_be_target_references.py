"""Belgian reference documents: separation, ageing pairs, bands and a pinned-feed audit.

The static tests read the committed documents. The feed audit needs the
Chronicle ``be`` consumer artifact, which CI does not hold: set
``BE_CHRONICLE_ARTIFACT`` to its directory to run it, as for NZ.
"""

from __future__ import annotations

import copy
import functools
import json
import math
import os
import re
from fractions import Fraction
from pathlib import Path
from unittest import mock

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from microcosm.build.country_spec import (
    _carried_value_keys,
    _typed_geography_vintage_aliases,
    load_country_spec,
)
from microcosm.build.ledger_artifact import (
    ChronicleFeedPinMismatchError,
    chronicle_feed_identity,
    load_ledger_consumer_artifact,
)
from microcosm.build.ledger_targets import (
    compile_ledger_target_references,
    fact_matches_selector,
    reference_fact_selectors,
)
from microcosm.build.reference_factors import (
    replay_reference_factor,
    validate_factor_context,
)
from microcosm.build.reference_intervals import (
    BELOW_FIRST_CUT,
    classify_records,
    interval_table,
    validate_interval_document,
)
from microcosm.build.reference_profiles import (
    REFERENCE_DOCUMENT_KEYS,
    reference_document_sha256,
    resolve_reference_profiles,
    validate_reference_resolution,
)
from microcosm.build.transport.compose import (
    load_transport_spec,
    validate_transport_activation,
)
from microcosm.build.transport.target_kernels import parse_reference_document
from test_support.paths import paths_for
from tools import generate_be_target_references as generator

_SPEC = paths_for("microcosm-build").package / "src/microcosm/build/be"
SETS = {
    "calibration": "target_references.json",
    "precal": "precal_references.json",
    "holdout": "holdout_references.json",
}
GENERATED = (
    *SETS.values(),
    "reference_aging.json",
    "reference_intervals.json",
    "reference_inventory.json",
    "reference_profiles.json",
)
BELGIAN_LEVELS = {"country": {"BE"}, "nuts1": {"BE1", "BE2", "BE3"}}
# B1's schema-2 profile required these; schema 1 can't, so this package test does.
REQUIRED_FAMILIES = {
    "demography",
    "fiscal_income",
    "income_tax",
    "social_security",
    "caseloads",
}
# Rows whose Eurostat base equals the pair's denominator without being its fact.
BASES_EQUAL_TO_DENOMINATOR = {
    "eurostat_household_current_taxes_paid",
    "eurostat_household_social_benefits_received",
    "eurostat_household_social_contributions_paid",
}
A1_CLASSES = [
    (0, 1000, "class_under_1"),
    *(
        (1000 * low, 1000 * (low + 1), f"class_{low}_{low + 1}")
        for low in range(1, 100)
    ),
    (100_000, None, "class_100_plus"),
]


def _document(name):
    return json.loads((_SPEC / name).read_text())


def _documents():
    return {reference_set: _document(name) for reference_set, name in SETS.items()}


def _rows(reference_set="calibration"):
    return _document(SETS[reference_set])["target_references"]


def _parsed(rows, hierarchy):
    return parse_reference_document(
        {
            "country": "be",
            "schema_version": 2,
            "hierarchy": hierarchy,
            "target_references": rows,
        },
        country="be",
    )


def _selectors(row):
    """The base selector (with any geography overlay), then each operand."""

    base = dict(row["ledger_selector"])
    operands = row.get("value_operands", [])
    for operand in operands:
        if operand.get("role") == "base":
            base.update(
                {
                    key: value
                    for key, value in operand.items()
                    if key in {"geography_level", "geography_id"}
                }
            )
    yield base
    for operand in operands:
        if operand.get("role") != "base":
            yield {key: value for key, value in operand.items() if key != "role"}


def _row_cells(row):
    """One identity per cell a row reads, whether pinned by key or by roster.

    A pinned selector also names its record set, measure, geography and
    dimension values, so pinned and roster cells share one identity and a
    holdout pinning a fact that calibration reads through a roster collides.
    """

    cells = set()
    for selector in _selectors(row):
        pins = dict(selector.get("dimension_values", {}))
        rosters = [key for key, value in pins.items() if isinstance(value, list)]
        members = pins[rosters[0]] if rosters else [None]
        areas = selector["geography_id"]
        for area in areas if isinstance(areas, list) else [areas]:
            for member in members:
                if rosters:
                    pins[rosters[0]] = member
                cells.add(
                    (
                        selector["record_set_id"],
                        selector["source_measure_id"],
                        area,
                        json.dumps(pins, sort_keys=True),
                    )
                )
    return cells


def _separation_violations(documents):
    """Every way a row could sit in two sets or a holdout could feed calibration."""

    violations = []
    names, bases, reads, measures = {}, {}, {}, {}
    for reference_set, document in documents.items():
        for row in document["target_references"]:
            if row["metadata"]["reference_set"] != reference_set:
                violations.append(("misfiled", row["name"]))
            names.setdefault(row["name"], set()).add(reference_set)
            base = {
                cell for cell in _row_cells({"ledger_selector": row["ledger_selector"]})
            }
            for cell in base:
                bases.setdefault(cell, set()).add(reference_set)
            for cell in _row_cells(row):
                reads.setdefault(cell, set()).add(reference_set)
            measures.setdefault((row["entity"], row["measure"]), set()).add(
                reference_set
            )
    violations += [("name", name) for name, sets in names.items() if len(sets) > 1]
    violations += [("base", cell) for cell, sets in bases.items() if len(sets) > 1]
    violations += [
        ("holdout_read", cell)
        for cell, sets in reads.items()
        if "holdout" in sets and len(sets) > 1
    ]
    violations += [
        ("holdout_measure", measure)
        for measure, sets in measures.items()
        if "holdout" in sets and len(sets) > 1
    ]
    return violations


def _reads():
    return _document("reference_inventory.json")["resolved_reads"]


def _read_violations(documents, factors, reads):
    """Disjointness on the listed feed keys, for every selector as committed.

    A digest ties each listed entry to its row's exact selectors, so a selector
    respelled to dodge the cell check fails here, and the feed audit proves the
    listed keys. Partition totals count as read, because a holdout on a band
    set's total shares its information.
    """

    violations = []
    listed = {entry["name"]: entry for entry in reads}
    expected = {
        row["name"]: generator.selector_digest(row)
        for document in documents.values()
        for row in document["target_references"]
    } | {
        f"factor:{entry['declaration']['id']}": generator.factor_digest(
            entry["declaration"]
        )
        for entry in factors["factors"]
    }
    violations += [("unlisted", name) for name in expected.keys() - listed.keys()]
    violations += [("unlisted", name) for name in listed.keys() - expected.keys()]
    violations += [
        ("digest", name)
        for name in expected.keys() & listed.keys()
        if listed[name]["selector_sha256"] != expected[name]
    ]
    facts, semantic, bases = {}, {}, {}
    for entry in reads:
        reference_set = entry["reference_set"]
        for key in entry["base_fact_keys"]:
            bases.setdefault(key, set()).add(reference_set)
            facts.setdefault(key, set()).add(reference_set)
        for key in entry["operand_fact_keys"]:
            facts.setdefault(key, set()).add(reference_set)
        total = entry.get("partition_total_semantic_key")
        for key in [*entry["semantic_fact_keys"], *([total] if total else [])]:
            semantic.setdefault(key, set()).add(reference_set)
    violations += [("shared_base", key) for key, sets in bases.items() if len(sets) > 1]
    violations += [
        (kind, key)
        for kind, index in (("holdout_fact", facts), ("holdout_semantic", semantic))
        for key, sets in index.items()
        if "holdout" in sets and len(sets) > 1
    ]
    return violations


def _vintage_violations(rows, layers):
    """Each subnational row must map its vintage to its layer's typed authority."""

    violations = []
    for row in rows:
        selector, metadata = row["ledger_selector"], row["metadata"]
        level = selector["geography_level"]
        if level == "country":
            continue
        vintage, _ = generator.GEOGRAPHY_VINTAGES.get(level, (None, None))
        if (
            selector["geography_vintage"] != vintage
            or metadata.get("geography_vintage") != vintage
            or metadata.get("geography_vintage_authority") != layers.get(level)
        ):
            violations.append(row["name"])
    return violations


def _money_context_violations(package):
    """One factor context must serve references and money, as #1436 can't check.

    Run entry validates each money node's context on its own, so two contexts
    could carry one factor id under different declarations. A non-diagnostic
    factor may live only in reference_profiles.factor_context_resource, and
    every money.normalize@1 node, in the graph or in entitlement rows, must
    select that resource.
    """

    context = json.loads((package / "reference_profiles.json").read_text())[
        "factor_context_resource"
    ]
    stem = context.removesuffix(".json")

    def nodes(value):
        if isinstance(value, dict):
            if value.get("kernel") == "money.normalize@1":
                yield value
            value = list(value.values())
        if isinstance(value, list):
            for item in value:
                yield from nodes(item)

    violations = []
    for path in sorted(package.glob("*.json")):
        document = json.loads(path.read_text())
        factors = document.get("factors") if isinstance(document, dict) else None
        if path.name != context and isinstance(factors, list):
            violations += [
                ("stray_factor", path.name)
                for entry in factors
                if entry["declaration"]["purpose"] != "diagnostic"
            ]
        for node in nodes(document):
            selection = node.get("params", {}).get("factor_context")
            resource = selection.get("resource") if isinstance(selection, dict) else ""
            if str(resource).removesuffix(".json") != stem:
                violations.append(("money_context", path.name))
    return violations


@functools.cache
def _rolled_rows():
    """Every rolled row across the three sets, as JSON text to copy per example."""

    return tuple(
        json.dumps(row)
        for reference_set in SETS
        for row in _rows(reference_set)
        if row.get("value_operation") == "rolled_forward_by_ratio"
    )


def _year_free(record_set_id):
    return re.sub(r"\.cy\d{4}\.", ".cy*.", record_set_id)


def _pair_violations(row):
    """A rolled row's operands must be one FPB row at the declared periods."""

    if row.get("value_operation") != "rolled_forward_by_ratio":
        return []
    base, numerator, denominator = row["value_operands"]
    problems = []
    if [base["role"], numerator["role"], denominator["role"]] != [
        "base",
        "numerator",
        "denominator",
    ]:
        problems.append("roles")
    for key in (
        "source_name",
        "source_measure_id",
        "source_concept",
        "entity_name",
        "geography_level",
        "geography_id",
        "dimensions",
    ):
        if numerator.get(key) != denominator.get(key):
            problems.append(key)
    if _year_free(numerator["record_set_id"]) != _year_free(
        denominator["record_set_id"]
    ):
        problems.append("row")
    pair = generator.AGEING_PAIRS.get(row["metadata"].get("ageing_pair"))
    if pair is None:
        problems.append("undeclared_pair")
    else:
        expected = f"{generator.FPB}cy*.{pair['row']}"
        if _year_free(numerator["record_set_id"]) != expected:
            problems.append("pair_row")
        if set(pair["operands"]) != {
            numerator["period_value"],
            denominator["period_value"],
        }:
            problems.append("pair_periods")
    if (
        numerator["period_value"] != row["period"]
        or numerator["assertion"] != "source_projection"
    ):
        problems.append("numerator_period")
    if (
        denominator["period_value"] != row["ledger_selector"]["period_value"]
        or denominator["assertion"] != "observation"
        or str(denominator["period_value"]) != row["metadata"]["source_period"]
        or str(row["period"]) != row["metadata"]["target_period"]
    ):
        problems.append("denominator_period")
    return problems


def _belgian_selector(selector):
    level, area = selector["geography_level"], selector["geography_id"]
    if level == "commune":
        areas = area if isinstance(area, list) else [area]
        return all(re.fullmatch(r"[1-9][0-9]{4}", code) for code in areas)
    return area in BELGIAN_LEVELS.get(level, set())


def test_documents_are_value_free_uniquely_named_and_registered():
    package = _document("country_package.json")
    declared = {row["path"] for row in package["resources"]}
    assert {"chronicle_feed.json", *GENERATED} <= declared
    spec = load_country_spec(_SPEC)
    assert {"chronicle_feed.json", *GENERATED} <= set(spec.resources)
    for name in GENERATED:
        payload = _document(name)
        if name not in {"reference_aging.json", "reference_inventory.json"}:
            # Factor receipts and the inventory record evidence; references,
            # intervals and profiles carry no observation at any depth.
            assert not _carried_value_keys(payload), name
        if name in SETS.values():
            rows = payload["target_references"]
            assert len({row["name"] for row in rows}) == len(rows)
            assert payload["schema_version"] == 2
            assert payload["target_profile"] == {
                "schema_version": 1,
                "hierarchy_reconciliations": [],
            }
            for row in rows:
                if row["metadata"]["activation_status"] != "active":
                    assert row["metadata"]["parking_reason"].strip(), row["name"]
                assert row["metadata"]["method_card_status"] == (
                    "proposed_pending_d1902"
                )


def test_counts_by_family_and_reference_set():
    inventory = _document("reference_inventory.json")["reference_counts"]
    assert {name: counts["total"] for name, counts in inventory.items()} == {
        "target_references.json": 60,
        "precal_references.json": 565,
        "holdout_references.json": 7,
    }
    assert inventory["target_references.json"]["by_family"] == {
        "demography": 18,
        "fiscal_income": 18,
        "income_tax": 19,
        "social_security": 1,
        "caseloads": 1,
        "benefit_expenditure": 2,
        "self_employment": 1,
    }
    for reference_set, name in SETS.items():
        rows = _rows(reference_set)
        assert inventory[name]["total"] == len(rows)
    active = {
        reference_set: sum(
            row["metadata"]["activation_status"] == "active"
            for row in _rows(reference_set)
        )
        for reference_set in SETS
    }
    assert active == {"calibration": 18, "precal": 565, "holdout": 0}


def test_reference_sets_are_separated():
    assert _separation_violations(_documents()) == []
    holdout = _rows("holdout")
    assert {row["metadata"]["target_role"] for row in holdout} == {"validation"}
    assert {row["metadata"]["target_role"] for row in _rows("precal")} == {
        "pre_calibration"
    }


@pytest.mark.parametrize(
    ("mutant", "kind"),
    [
        ("calibration_row_in_holdout", "name"),
        ("precal_row_in_calibration", "base"),
        ("holdout_reads_calibration_operand", "holdout_read"),
        ("holdout_measures_calibration_column", "holdout_measure"),
        ("holdout_base_is_calibration_fact", "holdout_read"),
        ("holdout_pins_a_class_calibration_reads_by_roster", "holdout_read"),
    ],
)
def test_separation_mutants_are_caught(mutant, kind):
    documents = copy.deepcopy(_documents())
    calibration = documents["calibration"]["target_references"]
    holdout = documents["holdout"]["target_references"]
    if mutant == "calibration_row_in_holdout":
        row = copy.deepcopy(calibration[0])
        row["metadata"]["reference_set"] = "holdout"
        holdout.append(row)
    elif mutant == "precal_row_in_calibration":
        row = copy.deepcopy(documents["precal"]["target_references"][0])
        row["name"] += "__copy"
        row["metadata"]["reference_set"] = "calibration"
        calibration.append(row)
    elif mutant == "holdout_reads_calibration_operand":
        donor = next(
            row
            for row in calibration
            if row["metadata"].get("ageing_pair") == "fpb_t11_total_resources"
        )
        holdout[0]["value_operands"][1] = copy.deepcopy(donor["value_operands"][1])
    elif mutant == "holdout_measures_calibration_column":
        holdout[0]["entity"], holdout[0]["measure"] = "tax_unit", "tax_units"
    elif mutant == "holdout_pins_a_class_calibration_reads_by_roster":
        band = next(
            row
            for row in calibration
            if row["name"] == "statbel_a1_standin_declarations__10k_20k"
        )
        selector = copy.deepcopy(band["ledger_selector"])
        selector["dimension_values"] = {generator.A1_CLASS: "class_12_13"}
        selector["aggregate_fact_key"] = "ledger.aggregate_fact.v2:pinned-class"
        holdout[0]["ledger_selector"] = selector
    else:
        donor = next(
            row
            for row in calibration
            if row["name"] == "onss_worker_personal_contributions"
        )
        holdout[0]["ledger_selector"] = copy.deepcopy(donor["ledger_selector"])
    assert kind in {violation[0] for violation in _separation_violations(documents)}


def test_resolved_reads_bind_every_selector_and_stay_disjoint():
    assert (
        _read_violations(_documents(), _document("reference_aging.json"), _reads())
        == []
    )


@pytest.mark.parametrize(
    ("mutant", "kind"),
    [
        ("respelled_selector", "digest"),
        ("dropped_entry", "unlisted"),
        ("holdout_reads_a_band_class", "holdout_fact"),
        ("holdout_reads_a_factor_operand", "holdout_fact"),
        ("holdout_on_a_partition_total", "holdout_semantic"),
        ("calibration_base_in_precal", "shared_base"),
    ],
)
def test_read_mutants_are_caught(mutant, kind):
    documents, reads = copy.deepcopy(_documents()), copy.deepcopy(_reads())
    entries = {entry["name"]: entry for entry in reads}
    holdout = next(entry for entry in reads if entry["reference_set"] == "holdout")
    band = entries["statbel_a1_standin_taxable_income__10k_20k"]
    if mutant == "respelled_selector":
        # A band row's selector with the class named only by its aggregate key:
        # the cell check sees a new cell, yet the selector still reads the class.
        donor = next(
            row
            for row in documents["calibration"]["target_references"]
            if row["name"] == band["name"]
        )
        selector = copy.deepcopy(donor["ledger_selector"])
        del selector["dimension_values"]
        selector["aggregate_fact_key"] = band["base_fact_keys"][2]
        documents["holdout"]["target_references"][0]["ledger_selector"] = selector
        assert _separation_violations(documents) == []
    elif mutant == "dropped_entry":
        reads.remove(holdout)
    elif mutant == "holdout_reads_a_band_class":
        holdout["base_fact_keys"].append(band["base_fact_keys"][0])
    elif mutant == "holdout_reads_a_factor_operand":
        factor = next(
            entry for entry in reads if entry["reference_set"] == "factor_context"
        )
        holdout["operand_fact_keys"].append(factor["operand_fact_keys"][0])
    elif mutant == "holdout_on_a_partition_total":
        holdout["semantic_fact_keys"].append(band["partition_total_semantic_key"])
    else:
        precal = next(entry for entry in reads if entry["reference_set"] == "precal")
        precal["base_fact_keys"].append(band["base_fact_keys"][0])
    assert kind in {
        violation[0]
        for violation in _read_violations(
            documents, _document("reference_aging.json"), reads
        )
    }


@functools.cache
def _typed_layers():
    resolved = load_country_spec(_SPEC).resolved_spec
    assignment = resolved.domain("geography").to_wire()["assignment"]
    return json.dumps(assignment["layer_vintages"])


def test_subnational_rows_map_their_vintage_to_the_typed_layer():
    layers = json.loads(_typed_layers())
    for reference_set in SETS:
        assert _vintage_violations(_rows(reference_set), layers) == []
    aliases = _typed_geography_vintage_aliases(
        load_country_spec(_SPEC).resolved_spec, country="be"
    )
    bindings = _document("reference_inventory.json")["geography_vintage_bindings"]
    assert {row["layer"] for row in bindings} == set(generator.GEOGRAPHY_VINTAGES)
    for binding in bindings:
        layer = binding["layer"]
        assert binding["typed_authority"] == layers[layer]
        assert (binding["chronicle_vintage"], binding["typed_authority"]) == (
            generator.GEOGRAPHY_VINTAGES[layer]
        )
        exact = binding["chronicle_vintage"] in aliases[layer]
        assert binding["relation"] == ("exact_alias" if exact else "crosswalk")


@pytest.mark.parametrize(
    ("reference_set", "change"),
    [
        ("calibration", {"selector": "NUTS_2021"}),
        ("calibration", {"metadata": "NUTS_2021"}),
        ("calibration", {"metadata": None}),
        ("calibration", {"authority": "vintage:be_province_2025"}),
        ("calibration", {"authority": None}),
        ("precal", {"selector": "nis_2024"}),
    ],
)
def test_vintage_mutants_refuse(reference_set, change):
    rows = copy.deepcopy(_rows(reference_set)[:1])
    selector, metadata = rows[0]["ledger_selector"], rows[0]["metadata"]
    for field, value in change.items():
        if field == "selector":
            selector["geography_vintage"] = value
        else:
            key = {
                "metadata": "geography_vintage",
                "authority": "geography_vintage_authority",
            }[field]
            if value is None:
                del metadata[key]
            else:
                metadata[key] = value
    assert _vintage_violations(rows, json.loads(_typed_layers())) == [rows[0]["name"]]


def test_an_unreviewed_vintage_label_refuses_generation():
    fact = {
        "aggregate_fact_key": "ledger.aggregate_fact.v2:synthetic",
        "geography": {"level": "nuts1", "id": "BE1", "vintage": "NUTS_2027"},
    }
    with pytest.raises(ValueError, match="reviewed mapping"):
        generator._vintage_authority(fact)


def test_calibration_keeps_every_family_b1_required():
    assert REQUIRED_FAMILIES <= {row["family"] for row in _rows()}


def test_belgian_selectors_never_name_german_or_french_rows():
    for reference_set in SETS:
        for row in _rows(reference_set):
            for selector in _selectors(row):
                assert _belgian_selector(selector), (row["name"], selector)
                assert not str(selector.get("geography_id")).startswith(("DE", "FR"))
    mutant = copy.deepcopy(_rows("holdout")[0]["ledger_selector"])
    mutant["geography_id"] = "DE"
    assert not _belgian_selector(mutant)
    for level, area in (("country", "DE"), ("country", "FR"), ("nuts1", "FR1")):
        fact = {
            "aggregate_fact_key": "ledger.aggregate_fact.v2:synthetic",
            "geography": {"level": level, "id": area},
        }
        with pytest.raises(ValueError, match="Belgian selectors read Belgian facts"):
            generator._belgian(fact)


def test_every_rolled_row_uses_one_declared_fpb_row_at_its_periods():
    rolled = [
        row
        for reference_set in SETS
        for row in _rows(reference_set)
        if row.get("value_operation") == "rolled_forward_by_ratio"
    ]
    assert len(rolled) == 22 + 7
    for row in rolled:
        assert _pair_violations(row) == [], row["name"]
        assert row["period"] == 2026
        assert row["period_match_policy"] == "latest_not_after"
        assert row["assertion_policy"] == "allow_source_projection"
    for row in _rows():
        if row.get("value_operation", "identity") == "identity":
            assert row["period"] == row["ledger_selector"]["period_value"]
            assert row["period_match_policy"] == "exact"
    pairs = {row["metadata"]["ageing_pair"] for row in rolled}
    assert pairs == set(generator.AGEING_PAIRS)


_FPB_ROWS = sorted({pair["row"] for pair in generator.AGEING_PAIRS.values()})


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    index=st.integers(min_value=0, max_value=30),
    other_row=st.sampled_from(
        [*_FPB_ROWS, "labour_market.levels.working_age_population"]
    ),
    other_year=st.sampled_from([2022, 2023, 2024, 2025, 2026, 2027]),
    field=st.sampled_from(["row", "numerator_year", "denominator_year", "concept"]),
)
def test_ratio_pairs_share_one_row_and_their_declared_periods(
    index, other_row, other_year, field
):
    rolled = _rolled_rows()
    row = json.loads(rolled[index % len(rolled)])
    assert _pair_violations(row) == []
    numerator, denominator = row["value_operands"][1:]
    changed = False
    if field == "row":
        current = _year_free(denominator["record_set_id"])
        target = f"{generator.FPB}cy*.{other_row}"
        changed = current != target
        denominator["record_set_id"] = target.replace(
            "cy*", f"cy{denominator['period_value']}"
        )
    elif field == "numerator_year":
        changed = other_year != numerator["period_value"]
        numerator["period_value"] = other_year
        numerator["record_set_id"] = re.sub(
            r"\.cy\d{4}\.", f".cy{other_year}.", numerator["record_set_id"]
        )
    elif field == "denominator_year":
        changed = other_year != denominator["period_value"]
        denominator["period_value"] = other_year
        denominator["record_set_id"] = re.sub(
            r"\.cy\d{4}\.", f".cy{other_year}.", denominator["record_set_id"]
        )
    else:
        changed = True
        denominator["source_concept"] += ".other"
    assert bool(_pair_violations(row)) == changed


@functools.cache
def _hierarchy(reference_set):
    return json.dumps(_document(SETS[reference_set])["hierarchy"])


def _synthetic_fact(selector, value, member=None):
    """A minimal fact matching one selector; a resolver probe, not a Belgian value."""

    dimensions = dict(selector.get("dimension_values", {}))
    key = selector.get("aggregate_fact_key")
    if member is not None:
        dimensions[member[0]] = member[1]
        key = f"synthetic:{selector['record_set_id']}:{member[1]}"
    return {
        "aggregate_fact_key": key,
        "observed_measure": {
            name: selector[name]
            for name in ("source_name", "source_measure_id", "source_concept")
        }
        | {"unit": "count"},
        "entity": {"name": selector["entity_name"]},
        "period": {"type": selector["period_type"], "value": selector["period_value"]},
        "geography": {
            "level": selector["geography_level"],
            "id": selector["geography_id"],
            "vintage": selector["geography_vintage"],
            "name": "Belgium",
        },
        "layout": {"record_set_id": selector["record_set_id"]},
        "dimensions": dimensions,
        "dimension_labels": {name: name for name in dimensions},
        "dimension_value_labels": {
            name: {str(item): str(item)} for name, item in dimensions.items()
        },
        "assertion": selector["assertion"],
        "value": value,
        "aggregation": {"method": "sum"},
        "label": "Synthetic resolver probe",
    }


def _synthetic_roll(row, draw):
    """Facts for one rolled row and the independently computed exact target."""

    selector = row["ledger_selector"]
    numerator_selector, denominator_selector = (
        {key: value for key, value in operand.items() if key != "role"}
        for operand in row["value_operands"][1:]
    )
    amount = st.integers(min_value=1, max_value=10**12)
    denominator = draw(amount)
    # The shared operation refuses ratios outside (0.5, 2]: stay inside.
    numerator = draw(
        st.integers(
            min_value=denominator * 6 // 10 + 1, max_value=denominator * 19 // 10
        )
    )
    rosters = [
        (key, value)
        for key, value in selector.get("dimension_values", {}).items()
        if isinstance(value, list)
    ]
    facts = []
    if rosters:
        dimension, members = rosters[0]
        bases = [draw(amount) for _ in members]
        facts += [
            _synthetic_fact(selector, value, (dimension, member))
            for value, member in zip(bases, members, strict=True)
        ]
    elif selector["aggregate_fact_key"] == denominator_selector["aggregate_fact_key"]:
        bases = [denominator]
    else:
        bases = [draw(amount)]
        facts.append(_synthetic_fact(selector, bases[0]))
    facts.append(_synthetic_fact(numerator_selector, numerator))
    facts.append(_synthetic_fact(denominator_selector, denominator))
    return facts, Fraction(sum(bases)) * Fraction(numerator, denominator)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(index=st.integers(min_value=0, max_value=60), data=st.data())
def test_shared_compiler_rolls_each_committed_row_as_declared(index, data):
    """Differential: the shared resolver against exact base * num / den, in CI."""

    rolled = _rolled_rows()
    row = _executable(json.loads(rolled[index % len(rolled)]))
    facts, expected = _synthetic_roll(row, data.draw)
    (reference,) = _parsed(
        [row], json.loads(_hierarchy(row["metadata"]["reference_set"]))
    )
    (spec,) = compile_ledger_target_references(facts, [reference], country="be").specs
    assert spec.value == pytest.approx(float(expected), rel=1e-12)


@pytest.mark.parametrize(
    "name",
    ["statbel_a1_standin_declarations__50k_75k", "statbel_b4_declarations__joint"],
)
def test_a_missing_roster_member_refuses_compilation(name):
    row = _executable(next(row for row in _rows() if row["name"] == name))
    facts, _ = _synthetic_roll(row, lambda strategy: 1000)
    (reference,) = _parsed([row], _document(SETS["calibration"])["hierarchy"])
    with pytest.raises(ValueError, match="base"):
        compile_ledger_target_references(facts[1:], [reference], country="be")


def test_rows_equal_to_the_fpb_projection_declare_it():
    flagged = set()
    for reference_set in SETS:
        for row in _rows(reference_set):
            if row.get("value_operation") != "rolled_forward_by_ratio":
                continue
            own = row["ledger_selector"].get("aggregate_fact_key") == row[
                "value_operands"
            ][2].get("aggregate_fact_key")
            declared = "ageing_target_equals" in row["metadata"]
            assert declared == (own or row["name"] in BASES_EQUAL_TO_DENOMINATOR)
            flagged |= {row["name"]} if declared else set()
    assert len(flagged) == 7


def test_band_edges_strictly_increase_and_partition_the_positive_range():
    document = _document("reference_intervals.json")
    tables = validate_interval_document(document)
    assert list(tables) == [generator.STANDIN_TABLE]
    table = tables[generator.STANDIN_TABLE]
    assert table.entity == "tax_unit"
    assert table.closure == "left" and table.nil_present
    assert table.cuts[0] == 0
    assert all(a < b for a, b in zip(table.cuts, table.cuts[1:], strict=False))
    bands = [band for band in document["tables"][0]["bands"] if band["kind"] != "nil"]
    assert [band["id"] for band in bands] == [
        band for band, _, _ in generator.STANDIN_BANDS
    ]
    assert [band["upper_cent"] for band in bands] == [
        None if upper is None else upper * 100
        for _, _, upper in generator.STANDIN_BANDS
    ]
    inventory = _document("reference_inventory.json")["standin_bands"]
    members = [code for band in inventory for code in band["a1_classes"]]
    assert members == [code for _, _, code in A1_CLASSES]
    assert len(inventory) == 8
    filters = {
        row["filter"]
        for row in _rows()
        if row["metadata"].get("interval_table_id") == generator.STANDIN_TABLE
    }
    assert filters == {band["indicator"] for band in bands}


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(min_value=-(10**9), max_value=10**9))
@example(0)
@example(1)
@example(999_999)
@example(1_000_000)
@example(9_999_999)
@example(10_000_000)
def test_every_amount_lands_in_exactly_one_band(cents):
    document = _document("reference_intervals.json")
    table = interval_table(document["tables"][0])
    (position,) = classify_records(table, [cents], ["included"])
    floored = max(0, cents)
    if floored == 0:
        assert position == table.nil_index
        return
    assert position != BELOW_FIRST_CUT
    band = document["tables"][0]["bands"][position]
    upper = band["upper_cent"]
    lower = table.cuts[table.positive_indices.index(position)]
    assert lower <= floored and (upper is None or floored < upper)


_EDGES = [lower for lower, _, _ in A1_CLASSES]


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.sets(st.sampled_from(_EDGES[1:]), max_size=20))
def test_any_band_list_on_class_edges_partitions_a1(cuts):
    edges = [0, *sorted(cuts)]
    bands = tuple(
        (f"b{index}", lower, upper)
        for index, (lower, upper) in enumerate(
            zip(edges, [*edges[1:], None], strict=True)
        )
    )
    with mock.patch.object(generator, "STANDIN_BANDS", bands):
        members = generator._band_members(A1_CLASSES)
    assert [code for band in members.values() for code in band] == [
        code for _, _, code in A1_CLASSES
    ]
    for (band, lower, upper), codes in zip(bands, members.values(), strict=True):
        spans = [(start, end) for start, end, code in A1_CLASSES if code in codes]
        assert spans[0][0] == lower and spans[-1][1] == upper, band


@pytest.mark.parametrize(
    "bands",
    [
        (("a", 0, 10_500), ("b", 10_500, None)),
        (("a", 0, 99_999), ("b", 99_999, None)),
        # The upper is off an edge, yet the next band restarts on one.
        (("a", 0, 10_500), ("b", 10_000, None)),
    ],
)
def test_band_edges_off_class_edges_refuse(bands):
    with mock.patch.object(generator, "STANDIN_BANDS", bands):
        with pytest.raises(ValueError, match="class edge|partition"):
            generator._band_members(A1_CLASSES)


def test_ageing_factors_are_g8r2_diagnostics_whose_periods_match_declarations():
    context = _document("reference_aging.json")
    feed = chronicle_feed_identity(_document("chronicle_feed.json"))
    assert context["feed_identity"] == feed
    selected = validate_factor_context(context)
    assert set(selected) == set(generator.RSZ_FACTORS)
    for factor_id, entry in selected.items():
        declaration, receipt = entry["declaration"], entry["receipt"]
        assert declaration["purpose"] == "diagnostic"
        assert declaration["feed_identity"] == feed
        source_year, expected = generator.RSZ_FACTORS[factor_id]
        assert declaration["source_period"] == source_year
        assert declaration["target_period"] == 2025
        for schedule, period in (
            ("source_schedule", declaration["source_period"]),
            ("target_schedule", declaration["target_period"]),
        ):
            (slot,) = declaration[schedule]
            assert slot["selector"]["period_value"] == period
            assert slot["validation"]["period_coverage"] == {
                "start_date": f"{period}-01-01",
                "end_date": f"{period}-12-31",
            }
        assert receipt["source_period"] == declaration["source_period"]
        assert receipt["target_period"] == declaration["target_period"]
        assert entry["operational_factor"] == {
            "numerator": expected,
            "denominator": 1_000_000,
        }


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    index=st.integers(min_value=0, max_value=40),
    shift=st.integers(min_value=-3, max_value=3).filter(bool),
    where=st.sampled_from(["source_period", "target_period", "period"]),
)
def test_each_rolled_period_matches_its_declaration(index, shift, where):
    rolled = _rolled_rows()
    row = json.loads(rolled[index % len(rolled)])
    if where == "period":
        row["period"] += shift
    else:
        row["metadata"][where] = str(int(row["metadata"][where]) + shift)
    assert "numerator_period" in _pair_violations(row) or (
        "denominator_period" in _pair_violations(row)
    )


def test_every_money_factor_lives_in_the_reference_profile_context():
    assert _money_context_violations(_SPEC) == []
    assert (
        _document("reference_profiles.json")["factor_context_resource"]
        == "reference_aging.json"
    )


_MONEY_FACTOR = {"factors": [{"declaration": {"purpose": "money_normalization_input"}}]}


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        # A money node on the reference context passes, in either resource.
        (
            {
                "reference_aging.json": _MONEY_FACTOR,
                "transport_graph.json": {
                    "nodes": [
                        {
                            "kernel": "money.normalize@1",
                            "params": {
                                "factor_context": {"resource": "reference_aging.json"}
                            },
                        }
                    ]
                },
            },
            [],
        ),
        (
            {
                "entitlement_graph.json": {
                    "steps": [
                        {
                            "kernel": "money.normalize@1",
                            "params": {
                                "factor_context": {"resource": "reference_aging"}
                            },
                        }
                    ]
                }
            },
            [],
        ),
        # A second context, with or without a node that selects it.
        ({"money_factor_context.json": _MONEY_FACTOR}, ["stray_factor"]),
        (
            {
                "money_factor_context.json": _MONEY_FACTOR,
                "transport_graph.json": {
                    "nodes": [
                        {
                            "kernel": "money.normalize@1",
                            "params": {
                                "factor_context": {
                                    "resource": "money_factor_context.json"
                                }
                            },
                        }
                    ]
                },
            },
            ["stray_factor", "money_context"],
        ),
        (
            {
                "entitlement_graph.json": {
                    "steps": [
                        {
                            "kernel": "money.normalize@1",
                            "params": {"factor_context": {"resource": "other.json"}},
                        }
                    ]
                }
            },
            ["money_context"],
        ),
        (
            {"transport_graph.json": {"nodes": [{"kernel": "money.normalize@1"}]}},
            ["money_context"],
        ),
    ],
)
def test_money_context_mutants(tmp_path, files, expected):
    (tmp_path / "reference_profiles.json").write_text(
        (_SPEC / "reference_profiles.json").read_text()
    )
    (tmp_path / "reference_aging.json").write_text(
        (_SPEC / "reference_aging.json").read_text()
    )
    for name, document in files.items():
        (tmp_path / name).write_text(json.dumps(document))
    assert sorted(kind for kind, _ in _money_context_violations(tmp_path)) == sorted(
        expected
    )


def test_profiles_name_the_factor_context_and_keep_ageing_deferred():
    profiles = _document("reference_profiles.json")
    assert profiles["factor_context_resource"] == "reference_aging.json"
    assert profiles["interval_resource"] == "reference_intervals.json"
    raw = _document("target_references.json")
    effective, intervals, receipt = resolve_reference_profiles(
        raw,
        profiles,
        {},
        _document("reference_intervals.json"),
        factor_context=_document("reference_aging.json"),
    )
    assert receipt["deferred_dimensions"] == ["aging_index"]
    assert receipt["transformed_names"] == []
    assert intervals == _document("reference_intervals.json")
    assert set(effective) == set(raw) & REFERENCE_DOCUMENT_KEYS
    assert effective["target_references"] == raw["target_references"]
    assert (
        validate_reference_resolution(
            effective,
            receipt,
            references_sha256=reference_document_sha256(effective),
            interval_document=intervals,
        )
        == receipt
    )
    references = parse_reference_document(effective, country="be")
    a1 = [
        reference
        for reference in references
        if reference.metadata.get("interval_table_id") == generator.STANDIN_TABLE
    ]
    assert len(a1) == 16
    for reference in a1:
        assert json.loads(reference.metadata["reference_prerequisites"]) == [
            "reference_profile:aging_index",
            f"measurement:{generator.STANDIN_MEASUREMENT}",
        ]


def test_only_demography_and_commune_support_are_active():
    active = {
        reference_set: {
            row["metadata"]["contract_target_id"]
            for row in _rows(reference_set)
            if row["metadata"]["activation_status"] == "active"
        }
        for reference_set in SETS
    }
    assert active == {
        "calibration": {"statbel_population_by_region_sex_age"},
        "precal": {"statbel_population_by_commune"},
        "holdout": set(),
    }


def test_every_parked_row_refuses_direct_compilation():
    for name in SETS.values():
        document = _document(name)
        for reference in _parsed(document["target_references"], document["hierarchy"]):
            if reference.metadata["activation_status"] != "active":
                with pytest.raises(
                    ValueError,
                    match="non-executable placeholder|requires reference profile",
                ):
                    compile_ledger_target_references([], [reference], country="be")


def test_belgium_still_refuses_to_activate():
    spec = load_transport_spec(_SPEC)
    with pytest.raises(ValueError, match="not activated") as error:
        validate_transport_activation(spec)
    for missing in ("benefit_unit_rule.json", "axiom_rules_bindings.json"):
        assert missing in str(error.value)
    assert not (_SPEC / "transport_graph.json").exists()
    assert not (_SPEC / "receipt_contract.json").exists()
    assert not (_SPEC / "currency_bridge.json").exists()


def test_b1_stays_inactive_with_named_failed_checks():
    checks = _document("reference_inventory.json")["b1_checks"]
    assert {
        key: checks[key]["result"]
        for key in ("tax_definition", "income_ranking", "policy_year")
    } == {"tax_definition": "fail", "income_ranking": "fail", "policy_year": "fail"}
    for key in ("tax_definition", "income_ranking", "policy_year"):
        assert checks[key]["missing"]
    # Every "passed" statement is backed by an assertion the generator makes.
    assert checks["asserted_evidence"] and all(checks["asserted_evidence"].values())
    assert checks["tax_definition"]["evidence"]["b1_deciles_sum_minus_total_eur"] == (
        pytest.approx(0, abs=0.1)
    )
    assert checks["tax_definition"]["evidence"][
        "top_percentiles_sum_minus_decile_10_eur"
    ] == pytest.approx(0, abs=0.1)
    b1 = [row for row in _rows() if row["family"] == "income_tax"]
    assert len(b1) == 19
    assert {row["metadata"]["activation_status"] for row in b1} == {
        "inactive_pending_b1_checks"
    }
    assert all(row.get("value_operation", "identity") == "identity" for row in b1)
    assert all(row["period"] == 2023 and row["signed"] is True for row in b1)
    evidence = checks["tax_definition"]["evidence"]
    assert abs(evidence["difference_eur"]) < 0.1
    assert evidence["negative_deciles"] == ["decile_01", "decile_02"]


def test_standin_gaps_publish_the_card_shares():
    gaps = _document("reference_inventory.json")["standin_gaps"]
    shares = gaps["national"]["component_shares_of_taxable_income"]
    assert {key: round(value * 100, 2) for key, value in shares.items()} == {
        "professional_income": 98.05,
        "immovable_property_income": 1.49,
        "capital_and_movable_property_income": 0.55,
        "miscellaneous_income": 0.12,
        "deductible_expenditures": 0.21,
    }
    assert round(gaps["national"]["omitted_share_gross"] * 100, 2) == 2.37
    assert gaps["declarations_with_taxable_but_no_professional_income"] == 42007
    assert gaps["zero_income_declarations_outside_a1"] == 523770
    assert [band["band"] for band in gaps["bands"]] == [
        band for band, _, _ in generator.STANDIN_BANDS
    ]
    for band in [gaps["national"], *gaps["bands"]]:
        assert abs(band["identity_residual_eur"]) < 0.05
    assert math.isclose(
        sum(band["declarations_iy2023"] for band in gaps["bands"]),
        gaps["national"]["declarations_iy2023"],
    )


def test_a1_band_counts_and_b4_types_share_one_ageing_pair():
    counts = {
        row["metadata"]["ageing_pair"]
        for row in _rows()
        if row["metadata"]["contract_target_id"]
        in {
            "statbel_a1_standin_declarations_by_band",
            "statbel_b4_declarations_by_type",
        }
    }
    assert counts == {"fpb_t07_total_population"}
    identity = _document("reference_inventory.json")["b4_identity"]
    assert identity["closes_exactly"] is True
    assert identity["joint"] + identity["individual"] == identity["a1_declarations"]


def test_jrc_is_deferred_and_august_gaps_are_named():
    for reference_set in SETS:
        for row in _rows(reference_set):
            for selector in _selectors(row):
                assert "jrc" not in selector["source_name"]
    exclusions = {
        row["id"] for row in _document("reference_inventory.json")["exclusions"]
    }
    assert {"jrc_euromod_be_statistics", "ris_amount_and_recipients"} <= exclusions
    onem = next(
        row for row in _rows() if row["name"] == "onem_complete_unemployment_recipients"
    )
    assert onem["metadata"]["activation_status"].startswith("requires_ageing_ruling")


def test_unemployment_choice_names_the_larger_line():
    choice = _document("reference_inventory.json")["unemployment_series_choice"]
    lines = {key: row["eur"] for key, row in choice["lines_2025"].items()}
    assert choice["lines_2025"]["chosen"]["label"] == "a. Chômage"
    assert choice["lines_2025"]["alternative"]["label"] == "4. Chômage"
    assert lines["alternative"] == pytest.approx(
        lines["chosen"] + lines["company_supplement"] + lines["career_breaks"],
        abs=2e6,
    )


def test_inventory_feed_identity_is_the_pin():
    assert _document("reference_inventory.json")["feed_identity"] == _document(
        "chronicle_feed.json"
    )


@pytest.fixture(scope="module")
def artifact():
    path = os.environ.get("BE_CHRONICLE_ARTIFACT")
    if not path:
        pytest.skip("BE_CHRONICLE_ARTIFACT is unset; Belgian feed audit unavailable")
    return load_ledger_consumer_artifact(
        Path(path), feed_pin=_document("chronicle_feed.json"), lexical_floats=True
    )


@pytest.fixture(scope="module")
def by_record_set(artifact):
    index = {}
    for fact in artifact.facts:
        index.setdefault(fact.get("layout", {}).get("record_set_id"), []).append(fact)
    return index


def _matched(selector, by_record_set):
    pool = by_record_set.get(selector.get("record_set_id"), [])
    return [fact for fact in pool if fact_matches_selector(fact, selector)]


def _executable(row):
    row = copy.deepcopy(row)
    row["metadata"]["activation_status"] = "active"
    for key in ("required_reference_profile", "reference_prerequisites"):
        row["metadata"].pop(key, None)
    return row


def test_generation_reproduces_committed_resources_with_exact_feed_identity(artifact):
    generated = generator.build_reference_documents(artifact)
    assert set(generated) == set(GENERATED)
    for filename, document in generated.items():
        assert generator._encode(document) == (_SPEC / filename).read_text(), filename
    assert artifact.fact_row_count == _document("chronicle_feed.json")["fact_row_count"]


def test_check_mode_passes_and_refuses_a_stale_copy(artifact, tmp_path, monkeypatch):
    monkeypatch.setattr(
        generator, "load_ledger_consumer_artifact", lambda *args, **kwargs: artifact
    )
    monkeypatch.setattr(
        "sys.argv", ["generate", "--artifact", "unused", "--check", "--out", str(_SPEC)]
    )
    generator.main()
    for name in ("chronicle_feed.json", *GENERATED):
        (tmp_path / name).write_text((_SPEC / name).read_text())
    stale = _document("holdout_references.json")
    stale["description"] += " (edited)"
    (tmp_path / "holdout_references.json").write_text(generator._encode(stale))
    monkeypatch.setattr(
        "sys.argv",
        ["generate", "--artifact", "unused", "--check", "--out", str(tmp_path)],
    )
    with pytest.raises(SystemExit, match="Stale generated resource"):
        generator.main()


def test_every_selector_resolves_to_one_fact_or_its_declared_set(
    artifact, by_record_set
):
    for name in SETS.values():
        document = _document(name)
        for row in document["target_references"]:
            base = _matched(row["ledger_selector"], by_record_set)
            pins = row["ledger_selector"].get("dimension_values", {})
            rosters = [value for value in pins.values() if isinstance(value, list)]
            expected = len(rosters[0]) if rosters else 1
            assert len(base) == expected, row["name"]
            for operand in row.get("value_operands", [])[1:]:
                selector = {k: v for k, v in operand.items() if k != "role"}
                assert len(_matched(selector, by_record_set)) == 1, row["name"]
            assert all(generator._belgian(fact) for fact in base)
            (reference,) = _parsed([_executable(row)], document["hierarchy"])
            needed = {}
            for selector in reference_fact_selectors(reference):
                for fact in _matched(selector, by_record_set):
                    needed[fact["aggregate_fact_key"]] = fact
            (spec,) = compile_ledger_target_references(
                needed.values(), [reference], country="be"
            ).specs
            if row.get("value_operation") == "rolled_forward_by_ratio":
                assert spec.metadata["ledger_value_base_cells"] == str(expected)
                numerator = _matched(
                    {k: v for k, v in row["value_operands"][1].items() if k != "role"},
                    by_record_set,
                )[0]
                denominator = _matched(
                    {k: v for k, v in row["value_operands"][2].items() if k != "role"},
                    by_record_set,
                )[0]
                total = math.fsum(float(fact["value"]) for fact in base)
                assert spec.value == pytest.approx(
                    total * float(numerator["value"]) / float(denominator["value"]),
                    rel=1e-12,
                )
            else:
                assert spec.value == float(base[0]["value"])


def test_german_and_french_rows_are_refused_and_never_selected(artifact, by_record_set):
    foreign = [
        fact
        for fact in artifact.facts
        if fact["geography"]["level"] == "country"
        and fact["geography"]["id"] in {"DE", "FR"}
    ]
    assert len(foreign) == 72
    for fact in foreign:
        with pytest.raises(ValueError, match="Belgian selectors read Belgian facts"):
            generator._selector(fact)
    keys = {fact["aggregate_fact_key"] for fact in foreign}
    for reference_set in SETS:
        for row in _rows(reference_set):
            for selector in _selectors(row):
                assert (
                    not {
                        fact["aggregate_fact_key"]
                        for fact in _matched(selector, by_record_set)
                    }
                    & keys
                )


def test_resolved_sets_share_no_fact_and_holdouts_read_none(artifact, by_record_set):
    reads = {}
    for reference_set in SETS:
        for row in _rows(reference_set):
            keys = {
                fact["aggregate_fact_key"]
                for selector in _selectors(row)
                for fact in _matched(selector, by_record_set)
            }
            for key in keys:
                reads.setdefault(key, set()).add(reference_set)
    assert not [
        key for key, sets in reads.items() if "holdout" in sets and len(sets) > 1
    ]
    bases = {}
    for reference_set in SETS:
        for row in _rows(reference_set):
            for fact in _matched(row["ledger_selector"], by_record_set):
                bases.setdefault(fact["aggregate_fact_key"], set()).add(reference_set)
    assert all(len(sets) == 1 for sets in bases.values())


def test_resolved_reads_list_what_each_selector_resolves_to(artifact, by_record_set):
    listed = {entry["name"]: entry for entry in _reads()}
    for reference_set in SETS:
        for row in _rows(reference_set):
            entry = listed[row["name"]]
            selectors = list(_selectors(row))
            base = _matched(selectors[0], by_record_set)
            operands = [
                fact
                for selector in selectors[1:]
                for fact in _matched(selector, by_record_set)
            ]
            assert entry["base_fact_keys"] == sorted(
                {fact["aggregate_fact_key"] for fact in base}
            )
            assert entry["operand_fact_keys"] == sorted(
                {fact["aggregate_fact_key"] for fact in operands}
            )
            assert entry["semantic_fact_keys"] == sorted(
                {fact["semantic_fact_key"] for fact in [*base, *operands]}
            )


def test_a_holdout_on_b1s_income_total_collides_with_a1s_bands(artifact):
    """B.1 and A.1 print one semantic total, which the bands add up to."""

    inventory = generator.Inventory(artifact.facts)
    total = inventory.one(
        spec=generator.B1_VALUES,
        measure="taxable_income_eur",
        geography="BE",
        dimensions={"tax_return.rank_group": "all"},
    )
    documents = copy.deepcopy(_documents())
    row = documents["holdout"]["target_references"][0]
    row["ledger_selector"] = generator._selector(total)
    del row["value_operands"]
    named = {name: documents[key] for key, name in SETS.items()}
    context = _document("reference_aging.json")
    reads = generator.resolved_reads(
        inventory, {**named, "reference_aging.json": context}
    )
    assert ("holdout_semantic", total["semantic_fact_key"]) in _read_violations(
        documents, context, reads
    )


def test_projection_flags_match_the_feed(artifact, by_record_set):
    for reference_set in SETS:
        for row in _rows(reference_set):
            if row.get("value_operation") != "rolled_forward_by_ratio":
                continue
            selectors = list(_selectors(row))
            base = _matched(selectors[0], by_record_set)
            (denominator,) = _matched(selectors[2], by_record_set)
            equal = len(base) == 1 and generator._value(base[0]) == generator._value(
                denominator
            )
            assert ("ageing_target_equals" in row["metadata"]) == equal, row["name"]


def test_a_changed_b1_fact_refuses_generation(artifact):
    facts = list(artifact.facts)
    index = next(
        position
        for position, fact in enumerate(facts)
        if fact["layout"].get("record_set_spec_id")
        == "spf_finances.pit.tax_before_withholding.country.v1"
    )
    facts[index] = {**facts[index], "value": 1.0}
    with pytest.raises(ValueError, match="Reviewed B.1 evidence changed"):
        generator._b1_checks(generator.Inventory(facts))


def test_shared_series_check_accepts_a_mismatched_fpb_row_that_ours_refuses(
    artifact, by_record_set
):
    """Documents a shared limitation: every T07 row shares one series key."""

    row = copy.deepcopy(
        next(row for row in _rows() if row["name"] == "statbel_b4_declarations__joint")
    )
    working_age = next(
        fact
        for fact in by_record_set[
            f"{generator.FPB}cy2023.labour_market.levels.working_age_population"
        ]
    )
    row["value_operands"][2] = {
        "role": "denominator",
        **generator._selector(working_age),
        "assertion": "observation",
    }
    assert "row" in _pair_violations(row)
    (reference,) = _parsed(
        [_executable(row)], _document("target_references.json")["hierarchy"]
    )
    needed = {
        fact["aggregate_fact_key"]: fact
        for selector in reference_fact_selectors(reference)
        for fact in _matched(selector, by_record_set)
    }
    compile_ledger_target_references(needed.values(), [reference], country="be")


def test_factor_receipts_replay_against_the_feed(artifact):
    context = _document("reference_aging.json")
    for entry in context["factors"]:
        replay_reference_factor(
            entry["declaration"],
            artifact.facts,
            context["feed_identity"],
            entry["operational_factor"],
            entry["receipt"],
        )


@pytest.mark.parametrize(
    ("field", "altered"),
    [
        ("facts_sha256", "0" * 64),
        ("manifest_sha256", "0" * 64),
        ("fact_row_count", 87111),
        ("consumer_fact_schema_versions", ["wrong"]),
        ("consumer_fact_schema_sha256", "0" * 64),
        ("artifact_schema_version", "wrong"),
    ],
)
def test_changed_feed_identity_refuses_authoring(artifact, field, altered, tmp_path):
    pin = _document("chronicle_feed.json")
    pin[field] = altered
    (tmp_path / "chronicle_feed.json").write_text(json.dumps(pin))
    with pytest.raises(ChronicleFeedPinMismatchError):
        generator.build_reference_documents(artifact, package_dir=tmp_path)


@pytest.mark.parametrize(
    ("table", "key", "period"),
    [
        ("AGEING_PAIRS", "fpb_t11_total_resources", 2026),
        ("AGEING_PAIRS", "fpb_t07_total_population", 2023),
        ("RSZ_OPERANDS", None, 2025),
    ],
)
def test_a_changed_reviewed_operand_refuses_generation(artifact, table, key, period):
    if table == "AGEING_PAIRS":
        pairs = copy.deepcopy(generator.AGEING_PAIRS)
        pairs[key]["operands"][period] = str(
            Fraction(pairs[key]["operands"][period]) + 1
        )
        patch = mock.patch.object(generator, "AGEING_PAIRS", pairs)
        message = "Reviewed FPB operand"
    else:
        operands = {**generator.RSZ_OPERANDS, period: "1"}
        patch = mock.patch.object(generator, "RSZ_OPERANDS", operands)
        message = "Reviewed RSZ operand"
    with patch, pytest.raises(ValueError, match=message):
        generator.build_reference_documents(artifact)
