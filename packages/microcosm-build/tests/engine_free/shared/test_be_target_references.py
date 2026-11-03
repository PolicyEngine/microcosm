"""Belgian reference documents: separation, ageing, bands, surfaces and a feed audit.

The static tests read the committed documents. The feed audit needs the
Chronicle ``be`` consumer artifact, which CI does not hold: set
``BE_CHRONICLE_ARTIFACT`` to its directory to run it, as for NZ.
"""

from __future__ import annotations

import copy
import functools
import itertools
import json
import math
import os
import re
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
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
from microcosm.build.transport.population_kernels import _surface_bands
from microcosm.build.transport.target_kernels import (
    decode_target_surface,
    encode_target_surface,
    parse_reference_document,
)
from microcosm.frame.transport import quantile_map
from test_support.paths import paths_for
from tools import generate_be_target_references as generator

_SPEC = paths_for("microcosm-build").package / "src/microcosm/build/be"
SETS = {
    "calibration": "target_references.json",
    "precal": "precal_references.json",
    "holdout": "holdout_references.json",
}
# The committed selectors' fields. The shared vocabulary admits more, including
# other fact keys that neither the pin check nor the cell check reads.
_SELECTOR_FIELDS = frozenset(
    {
        "aggregate_fact_key",
        "assertion",
        "dimension_values",
        "dimensions",
        "entity_name",
        "geography_id",
        "geography_level",
        "geography_vintage",
        "period_type",
        "period_value",
        "record_set_id",
        "source_concept",
        "source_measure_id",
        "source_name",
    }
)
SURFACES = generator.SURFACE_RESOURCE
GENERATED = (
    *SETS.values(),
    "reference_aging.json",
    "reference_intervals.json",
    "reference_inventory.json",
    "reference_profiles.json",
    SURFACES,
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


def _listed(value):
    return value if isinstance(value, list) else [value]


def _row_cells(row):
    """One identity per cell a row reads, whether pinned by key or by roster.

    A pinned selector also names its record set, measure, geography and
    dimension values, so pinned and roster cells share one identity and a
    holdout pinning a fact that calibration reads through a roster collides.
    A selector with several rosters or a record-set list reads every
    combination of them, so the cells are their product. The vocabulary can't
    pair a record set with a dimension value, so for a wage class the product
    holds its 24 cells and 24 that no fact fills: a superset, never a miss.
    """

    cells = set()
    for selector in _selectors(row):
        pins = selector.get("dimension_values", {})
        names = sorted(pins)
        for record_set in _listed(selector["record_set_id"]):
            for area in _listed(selector["geography_id"]):
                for members in itertools.product(*(_listed(pins[n]) for n in names)):
                    cells.add(
                        (
                            record_set,
                            selector["source_measure_id"],
                            area,
                            json.dumps(
                                dict(zip(names, members, strict=True)),
                                sort_keys=True,
                            ),
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
            measures.setdefault(row["measure"], set()).add(reference_set)
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


def _keys(facts):
    return [key for fact in facts for key in fact.values()]


def _distinct_texts(values):
    return (
        bool(values)
        and all(isinstance(value, str) and value for value in values)
        and len(set(values)) == len(values)
    )


def _unpinned(row):
    """Whether a row's selectors read through a spelling no check here can see.

    The pin check reads aggregate keys and the cell check reads measures and
    dimension values literally, but the resolver skips an empty field, admits
    other fact keys, checks only the pins a selector names and matches a list
    by membership. So each selector keeps to the committed fields, leaves none
    empty, pins every dimension it declares, and pins an aggregate key or, for
    a base selector, a roster. Only a base selector may hold several rosters
    or a list of distinct record sets, and several rosters need a declared
    member count; geography_id may be a list too, and no other field. The feed
    audit stays the proof: these shapes catch hand edits, and a roster's
    members bind to no listed key without the artifact.
    """

    for index, selector in enumerate(_selectors(row)):
        pins = selector.get("dimension_values") or {}
        rosters = [value for value in pins.values() if isinstance(value, list)]
        members = [member for value in pins.values() for member in _listed(value)]
        lists = {"geography_id", "record_set_id"} if index == 0 else {"geography_id"}
        record_sets = selector.get("record_set_id")
        if (
            set(selector) - _SELECTOR_FIELDS
            or any(value is None or value == "" for value in selector.values())
            or not isinstance(selector.get("dimensions"), list)
            or set(pins) != set(selector["dimensions"])
            or not all(rosters)
            or not all(isinstance(item, str | int) and item != "" for item in members)
            or any(
                isinstance(value, list | dict) and key not in lists
                for key, value in selector.items()
                if key not in {"dimensions", "dimension_values"}
            )
            or (isinstance(record_sets, list) and not _distinct_texts(record_sets))
            or (len(rosters) > 1 and (index > 0 or _declared_members(row) is None))
            or not (selector.get("aggregate_fact_key") or (index == 0 and rosters))
        ):
            return True
    return False


def _declared_members(row):
    """The member count a row declares for its base selector, if any."""

    for operand in row.get("value_operands", []):
        if operand.get("role") == "base":
            return operand.get("expected_member_count")
    return row.get("expected_member_count")


def _miscounted(row, entry):
    """Whether a roster base disagrees with its own member count or its entry.

    A roster selector reads the product of its rosters, which must equal the
    member count it declares and the number of keys its entry lists. The
    resolver refuses a short read at compile time; this refuses the
    declaration, or an entry that dropped or gained a member, in CI.
    """

    pins = row["ledger_selector"].get("dimension_values") or {}
    rosters = [value for value in pins.values() if isinstance(value, list)]
    if not rosters:
        return False
    count = math.prod(len(roster) for roster in rosters)
    return _declared_members(row) != count or (
        entry is not None and len(set(_keys(entry["base_facts"]))) != count
    )


def _pin_violations(documents, factors, listed):
    """Each selector must pin what its resolved-read entry lists, under its set.

    A digest ties an entry to its selectors but not to the keys it lists, so a
    selector rewritten with a recomputed digest would keep its old keys. Roster
    selectors name no key: their entries must list as many keys as the rosters
    name cells, and only the feed audit proves which facts they are.
    """

    violations = []
    for reference_set, document in documents.items():
        for row in document["target_references"]:
            entry = listed.get(row["name"])
            if _unpinned(row):
                violations.append(("unpinned", row["name"]))
            if _miscounted(row, entry):
                violations.append(("member_count", row["name"]))
            if entry is None:
                continue
            if entry["reference_set"] != reference_set:
                violations.append(("relabelled", row["name"]))
            base = row["ledger_selector"].get("aggregate_fact_key")
            operands = {
                key
                for operand in row.get("value_operands", [])
                if "aggregate_fact_key" in operand
                for key in _listed(operand["aggregate_fact_key"])
            }
            if (
                base is not None and _keys(entry["base_facts"]) != [base]
            ) or not operands <= set(_keys(entry["operand_facts"])):
                violations.append(("pin", row["name"]))
    for factor in factors["factors"]:
        declaration = factor["declaration"]
        name = f"factor:{declaration['id']}"
        pins = {
            slot["selector"]["aggregate_fact_key"]
            for schedule in ("source_schedule", "target_schedule")
            for slot in declaration[schedule]
        }
        if name not in listed:
            continue
        if pins != set(_keys(listed[name]["operand_facts"])):
            violations.append(("pin", name))
        if listed[name]["reference_set"] != "factor_context":
            violations.append(("relabelled", name))
    return violations


def _read_violations(documents, factors, reads):
    """Disjointness on the listed feed keys, for every selector as committed.

    A digest ties each listed entry to its row's exact selectors, so a selector
    respelled to dodge the cell check fails here. The pin check ties each pinned
    key, each roster's member count and each row's set to its entry and refuses
    a spelling these checks can't read, and the feed audit proves every listed
    key. Partition totals count as read, because a holdout on a band set's total
    shares its information.
    """

    listed = {entry["name"]: entry for entry in reads}
    violations = _pin_violations(documents, factors, listed)
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
        for key in _keys(entry["base_facts"]):
            bases.setdefault(key, set()).add(reference_set)
            facts.setdefault(key, set()).add(reference_set)
        for key in _keys(entry["operand_facts"]):
            facts.setdefault(key, set()).add(reference_set)
        total = entry.get("partition_total")
        for key in _keys([*entry["semantic_facts"], *([total] if total else [])]):
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
        "precal_references.json": 565 + 5 + 26,
        "holdout_references.json": 7,
    }
    assert inventory["precal_references.json"]["by_family"] == {
        "geography_support": 565,
        "wage_distribution": 5,
        "pension_distribution": 26,
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
    assert active == {"calibration": 18, "precal": 565 + 5, "holdout": 0}


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
        ("calibration_reuses_holdout_column_on_another_entity", "holdout_measure"),
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
    elif mutant == "calibration_reuses_holdout_column_on_another_entity":
        donor = next(
            row for row in calibration if row["entity"] != holdout[0]["entity"]
        )
        donor["measure"] = holdout[0]["measure"]
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
        ("recomputed_digest_on_a_rewritten_base", "pin"),
        ("recomputed_digest_on_a_rewritten_operand", "pin"),
        ("factor_entry_keeps_an_old_key", "pin"),
        ("relabelled_entry", "relabelled"),
        ("relabelled_factor_entry", "relabelled"),
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
    row = documents["holdout"]["target_references"][0]
    if mutant.startswith(("respelled", "recomputed")):
        # A band row's selector with the class named only by its aggregate key:
        # the cell check sees a new cell, yet the selector still reads the class.
        donor = next(
            row
            for row in documents["calibration"]["target_references"]
            if row["name"] == band["name"]
        )
        selector = copy.deepcopy(donor["ledger_selector"])
        del selector["dimension_values"]
        selector["aggregate_fact_key"] = _keys(band["base_facts"])[2]
        if mutant == "recomputed_digest_on_a_rewritten_operand":
            operand = row["value_operands"][1]
            row["value_operands"][1] = {"role": operand["role"], **selector}
        else:
            row["ledger_selector"] = selector
        assert _separation_violations(documents) == []
        if mutant != "respelled_selector":
            entries[row["name"]]["selector_sha256"] = generator.selector_digest(row)
    elif mutant == "factor_entry_keeps_an_old_key":
        factor = next(
            entry for entry in reads if entry["reference_set"] == "factor_context"
        )
        factor["operand_facts"][0] = {"aggregate_fact_key": "stale"}
    elif mutant == "relabelled_entry":
        entries[row["name"]]["reference_set"] = "calibration"
    elif mutant == "relabelled_factor_entry":
        factor = next(
            entry for entry in reads if entry["reference_set"] == "factor_context"
        )
        factor["reference_set"] = "calibration"
    elif mutant == "dropped_entry":
        reads.remove(holdout)
    elif mutant == "holdout_reads_a_band_class":
        holdout["base_facts"].append(band["base_facts"][0])
    elif mutant == "holdout_reads_a_factor_operand":
        factor = next(
            entry for entry in reads if entry["reference_set"] == "factor_context"
        )
        holdout["operand_facts"].append(factor["operand_facts"][0])
    elif mutant == "holdout_on_a_partition_total":
        holdout["semantic_facts"].append(band["partition_total"])
    else:
        precal = next(entry for entry in reads if entry["reference_set"] == "precal")
        precal["base_facts"].append(band["base_facts"][0])
    assert kind in {
        violation[0]
        for violation in _read_violations(
            documents, _document("reference_aging.json"), reads
        )
    }


@pytest.mark.parametrize("target", ["base", "operand"])
@pytest.mark.parametrize(
    "spelling",
    [
        {"semantic_fact_key": "key"},
        {"legacy_fact_key": "key"},
        {"source_record_id": "key"},
        {"layout_groupby_value_id": "key"},
        {"dimensions": {generator.A1_CLASS: "class_12_13"}},
        {"aggregate_fact_key": None, "source_measure_id": None},
        {"aggregate_fact_key": ""},
        {
            "dimensions": [generator.A1_CLASS],
            "dimension_values": {generator.A1_CLASS: ["class_12_13"]},
            "source_measure_id": None,
        },
        {
            "dimensions": [generator.A1_CLASS],
            "dimension_values": {generator.A1_CLASS: ["class_12_13"]},
            "source_measure_id": "",
        },
        {
            "dimensions": [generator.A1_CLASS],
            "dimension_values": {generator.A1_CLASS: ["class_12_13"]},
            "semantic_fact_key": "key",
        },
        {},
        {"dimensions": {"a": "x"}, "dimension_values": {"a": ["x"]}},
        {"dimensions": ["a", "b"], "dimension_values": {"a": ["x"]}},
        {"dimensions": ["a", "b"], "dimension_values": {"a": ["x"], "b": ["y"]}},
        {"dimensions": ["a"], "dimension_values": {"a": []}},
        {"dimensions": ["a"], "dimension_values": {"a": [None]}},
        {"dimensions": ["a"], "dimension_values": {"a": [""]}},
        {"dimensions": ["a"], "dimension_values": {"a": [[]]}},
        {"dimensions": ["a"], "dimension_values": {"a": ["x", None]}},
        {"dimensions": ["a", "b"], "dimension_values": {"a": ["x"], "b": None}},
        {"dimensions": ["a", "b"], "dimension_values": {"a": ["x"], "b": ""}},
        {"dimensions": ["a", "b"], "dimension_values": {"a": ["x"], "b": []}},
    ],
)
def test_spellings_the_checks_cannot_read_refuse(spelling, target):
    """A selector can name a fact in ways neither the key nor cell check sees."""

    documents, reads = copy.deepcopy(_documents()), copy.deepcopy(_reads())
    row = documents["holdout"]["target_references"][0]
    selector = row["ledger_selector"]
    if target == "operand":
        selector = row["value_operands"][1]
    del selector["aggregate_fact_key"]
    selector.update(spelling)
    next(entry for entry in reads if entry["name"] == row["name"])[
        "selector_sha256"
    ] = generator.selector_digest(row)
    violations = _read_violations(documents, _document("reference_aging.json"), reads)
    assert ("unpinned", row["name"]) in violations
    assert ("digest", row["name"]) not in violations


def test_an_operand_needs_a_key_even_with_a_roster():
    """Only a base selector may read a roster; an operand reads one fact."""

    row = copy.deepcopy(_rows("holdout")[0])
    operand = row["value_operands"][1]
    del operand["aggregate_fact_key"]
    operand["dimensions"] = [generator.A1_CLASS]
    operand["dimension_values"] = {generator.A1_CLASS: ["class_12_13"]}
    assert _unpinned(row)
    assert not _unpinned(_rows("holdout")[0])


def test_only_a_base_selector_may_hold_several_rosters():
    """A declared member count admits several rosters on the base, never an operand."""

    row = copy.deepcopy(_rows("holdout")[0])
    row["value_operands"][0]["expected_member_count"] = 2
    operand = row["value_operands"][1]
    operand["dimensions"] = ["a", "b"]
    operand["dimension_values"] = {"a": ["x"], "b": ["y", "z"]}
    assert _unpinned(row)
    operand["dimension_values"]["a"] = "x"
    assert not _unpinned(row)


WAGE = f"{generator.WAGE_CONTRACT}__lt100"
# One real T17 cell of the <100 class, for mutants: men under 20. Record set
# 9ae5b76f27368269 holds the class's Man-Homme cells. A pin, not a value.
WAGE_MEMBER = {
    "record_set_id": "rsz.t17.2026Q1.9ae5b76f27368269.employed_workers",
    "dimension_values": {
        "rsz.age_class": "-20",
        generator.WAGE_CLASS: "<100",
        "rsz.sex": "Man-Homme",
    },
}


def _named(documents, name):
    return next(
        row
        for document in documents.values()
        for row in document["target_references"]
        if row["name"] == name
    )


def _entry(reads, name):
    return next(entry for entry in reads if entry["name"] == name)


@pytest.mark.parametrize(
    ("mutant", "kind"),
    [
        ("partial_roster", "unpinned"),
        ("partial_roster_without_dimensions", "unpinned"),
        ("roster_beside_an_unpinned_dimension", "unpinned"),
        ("no_key_and_no_roster", "unpinned"),
        ("rosters_without_a_member_count", "unpinned"),
        ("repeated_record_set", "unpinned"),
        ("record_set_list_on_an_operand", "unpinned"),
        ("key_list_on_the_base", "unpinned"),
        ("list_valued_measure", "unpinned"),
        ("declared_count_disagrees", "member_count"),
        ("roster_gains_a_member", "member_count"),
        ("entry_loses_a_member", "member_count"),
        ("entry_gains_a_member", "member_count"),
        ("entry_swaps_a_member_for_a_holdout_fact", "holdout_fact"),
    ],
)
def test_summed_row_spellings_and_mutants_are_caught(mutant, kind):
    """A summed wage row reads its rosters' product, and its entry lists as many."""

    documents, reads = copy.deepcopy(_documents()), copy.deepcopy(_reads())
    row = _named(documents, WAGE)
    selector = row["ledger_selector"]
    pins = selector["dimension_values"]
    holdout = documents["holdout"]["target_references"][0]
    entry = _entry(reads, row["name"])
    subject = row["name"]
    if mutant == "partial_roster":
        del pins["rsz.sex"]
    elif mutant == "partial_roster_without_dimensions":
        del pins["rsz.sex"], selector["dimensions"]
    elif mutant == "roster_beside_an_unpinned_dimension":
        selector["dimensions"].append("rsz.region")
    elif mutant == "no_key_and_no_roster":
        selector.update(copy.deepcopy(WAGE_MEMBER))
    elif mutant == "rosters_without_a_member_count":
        del row["expected_member_count"]
    elif mutant == "repeated_record_set":
        selector["record_set_id"][1] = selector["record_set_id"][0]
    elif mutant == "record_set_list_on_an_operand":
        row = holdout
        operand = row["value_operands"][1]
        operand["record_set_id"] = [operand["record_set_id"], "other"]
        subject = row["name"]
    elif mutant == "key_list_on_the_base":
        selector["aggregate_fact_key"] = _keys(entry["base_facts"])
    elif mutant == "list_valued_measure":
        selector["source_measure_id"] = [generator.WAGE_MEASURE, "jobs"]
    elif mutant == "declared_count_disagrees":
        row["expected_member_count"] = 23
    elif mutant == "roster_gains_a_member":
        pins["rsz.age_class"].append("75+")
    elif mutant == "entry_loses_a_member":
        entry["base_facts"].pop(0)
    elif mutant == "entry_gains_a_member":
        entry["base_facts"].append({"aggregate_fact_key": "ledger.aggregate_fact.v2:x"})
    else:
        # A regenerated entry lists a holdout's fact in place of a member.
        subject = holdout["ledger_selector"]["aggregate_fact_key"]
        entry["base_facts"][0] = {"aggregate_fact_key": subject}
    _entry(reads, row["name"])["selector_sha256"] = generator.selector_digest(row)
    violations = _read_violations(documents, _document("reference_aging.json"), reads)
    assert (kind, subject) in violations
    assert ("digest", row["name"]) not in violations


def test_a_holdout_pinning_one_wage_member_collides():
    """_row_cells gives the exact cell a holdout pinning one member builds."""

    documents = copy.deepcopy(_documents())
    selector = {
        **copy.deepcopy(_named(documents, WAGE)["ledger_selector"]),
        **copy.deepcopy(WAGE_MEMBER),
        "aggregate_fact_key": "ledger.aggregate_fact.v2:member",
    }
    documents["holdout"]["target_references"][0]["ledger_selector"] = selector
    cell = (
        WAGE_MEMBER["record_set_id"],
        generator.WAGE_MEASURE,
        "BE",
        json.dumps(WAGE_MEMBER["dimension_values"], sort_keys=True),
    )
    violations = _separation_violations(documents)
    assert ("holdout_read", cell) in violations
    assert ("base", cell) in violations


@functools.cache
def _all_rows():
    return tuple(
        json.dumps(row) for reference_set in SETS for row in _rows(reference_set)
    )


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(index=st.integers(min_value=0, max_value=10**4), data=st.data())
def test_row_cells_hold_every_member_a_selector_can_name(index, data):
    """Property: one record set, area and member of each roster is one cell."""

    rows = _all_rows()
    selector = json.loads(rows[index % len(rows)])["ledger_selector"]
    single = {
        **selector,
        **{
            key: data.draw(st.sampled_from(selector[key]))
            for key in ("record_set_id", "geography_id")
            if isinstance(selector[key], list)
        },
    }
    if "dimension_values" in selector:
        single["dimension_values"] = {
            name: data.draw(st.sampled_from(value))
            if isinstance(value, list)
            else value
            for name, value in selector["dimension_values"].items()
        }
    (cell,) = _row_cells({"ledger_selector": single})
    assert cell in _row_cells({"ledger_selector": selector})


def test_row_cells_expand_the_full_roster_product():
    rows = {row["name"]: row for row in _rows("precal") + _rows()}
    for row in _surface_rows(_documents())["wages"]:
        # 2 record sets x 12 ages x 2 sexes: 24 cells hold facts, 24 can't.
        assert len(_row_cells(row)) == 48
    for row in _surface_rows(_documents())["legal_pensions"]:
        assert len(_row_cells(row)) == 1
    band = rows["statbel_a1_standin_declarations__10k_20k"]
    assert len(_row_cells({"ledger_selector": band["ledger_selector"]})) == 10


def _surface_rows(documents):
    """Each surface's precal rows, in document order."""

    return {
        surface: [
            row
            for row in documents["precal"]["target_references"]
            if row["metadata"].get("precal_use") == f"quantile_surface:{surface}"
        ]
        for surface in ("wages", "legal_pensions")
    }


def _finite(edge):
    return (
        isinstance(edge, int | float)
        and not isinstance(edge, bool)
        and math.isfinite(edge)
    )


def _band_violations(document, references):
    """Whether a band document is a bands parameter transport.quantile_map@1 takes.

    The kernel takes the document verbatim, so it holds exactly a nonempty
    bands list of lower, upper and reference. Edges strictly increase and stay
    contiguous, the first lower bound is finite and nonnegative, only the last
    upper bound may be null, and each reference names a distinct precal row.
    """

    if (
        not isinstance(document, dict)
        or set(document) != {"bands"}
        or not isinstance(document["bands"], list)
        or not document["bands"]
        or any(
            not isinstance(band, dict) or set(band) != {"lower", "upper", "reference"}
            for band in document["bands"]
        )
    ):
        return ["shape"]
    bands, problems = document["bands"], set()
    if not _finite(bands[0]["lower"]) or bands[0]["lower"] < 0:
        problems.add("first_lower")
    for band, following in zip(bands, [*bands[1:], None], strict=True):
        lower, upper = band["lower"], band["upper"]
        if upper is None:
            if following is not None:
                problems.add("open_interior")
        elif not (_finite(lower) and _finite(upper) and lower < upper):
            problems.add("order")
        elif following is not None and following["lower"] != upper:
            problems.add("gap")
    names = [band["reference"] for band in bands]
    if not all(isinstance(name, str) for name in names):
        problems.add("reference")
    else:
        if len(set(names)) != len(names):
            problems.add("duplicate_reference")
        if not set(names) <= set(references):
            problems.add("reference")
    return sorted(problems)


def _surface_violations(documents, surfaces):
    """The surfaces must read their declared facts through their own rows.

    _surface_bands checks only that each reference is an unfiltered person
    count, which T17's jobs measure and SFPD's Totaal row also are. So each
    wage row sums T17's 2026-Q1 employed_workers cells and stays active, each
    pension row pins one SFPD group 1 recipient band and stays parked, and
    each band document maps its surface's rows in order, with the label each
    row pins and the edges its provenance records.
    """

    violations = []
    rows = _surface_rows(documents)
    precal = {row["name"] for row in documents["precal"]["target_references"]}
    quarter = generator.WAGE_PERIOD.replace("-", "")
    for row in rows["wages"]:
        selector = row["ledger_selector"]
        if (
            selector.get("source_measure_id") != generator.WAGE_MEASURE
            or selector.get("source_concept") != f"rsz.{generator.WAGE_MEASURE}"
            or selector.get("period_value") != generator.WAGE_PERIOD
            or not all(
                isinstance(name, str)
                and name.startswith(f"rsz.t17.{quarter}.")
                and name.endswith(f".{generator.WAGE_MEASURE}")
                for name in _listed(selector.get("record_set_id"))
            )
        ):
            violations.append(("wage_cells", row["name"]))
        if (
            row.get("value_operation") != "sum"
            or row.get("period_match_policy") != "exact"
            or row["metadata"]["activation_status"] != "active"
        ):
            violations.append(("wage_row", row["name"]))
    for row in rows["legal_pensions"]:
        selector = row["ledger_selector"]
        pins = dict(selector.get("dimension_values", {}))
        band = pins.pop(generator.PENSION_BAND, None)
        if (
            selector.get("record_set_id") != generator.PENSION_RECORD_SET
            or selector.get("source_measure_id") != "recipients"
            or pins != generator.PENSION_COHORT
            or not isinstance(band, str)
            or band == generator.PENSION_TOTAL_LABEL
        ):
            violations.append(("pension_band", row["name"]))
        if row["metadata"]["activation_status"] != generator.PENSION_STATUS:
            violations.append(("pension_active", row["name"]))
    for surface, pin in (
        ("wages", generator.WAGE_CLASS),
        ("legal_pensions", generator.PENSION_BAND),
    ):
        entry = surfaces[surface]
        if entry.get("status") != generator.METHOD_CARD_STATUS:
            violations.append(("status", surface))
        problems = _band_violations(entry["band_document"], precal)
        violations += [(f"band_{problem}", surface) for problem in problems]
        if problems == ["shape"]:
            continue
        bands = entry["band_document"]["bands"]
        if [band["reference"] for band in bands] != [
            row["name"] for row in rows[surface]
        ]:
            violations.append(("band_rows", surface))
        labels = {
            row["name"]: row["ledger_selector"].get("dimension_values", {}).get(pin)
            for row in rows[surface]
        }
        if [
            (
                source["reference"],
                source["printed_label"],
                source["lower"],
                source["upper"],
            )
            for source in entry["band_provenance"]
        ] != [
            (
                band["reference"],
                labels.get(band["reference"]),
                band["lower"],
                band["upper"],
            )
            for band in bands
        ]:
            violations.append(("provenance", surface))
    return violations


def test_quantile_surfaces_read_their_declared_rows():
    documents, surfaces = _documents(), _document(SURFACES)
    assert _surface_violations(documents, surfaces) == []
    rows = _surface_rows(documents)
    assert [len(rows["wages"]), len(rows["legal_pensions"])] == [5, 26]
    wages, pensions = surfaces["wages"], surfaces["legal_pensions"]
    assert (wages["unit"], pensions["unit"]) == ("eur_per_day", "eur_per_month")
    assert (wages["source_period"], pensions["source_period"]) == ("2026-Q1", "2025-01")
    assert (wages["donor_column"], pensions["donor_column"]) == (
        "employment_income",
        None,
    )
    assert [
        (band["lower"], band["upper"]) for band in wages["band_document"]["bands"]
    ] == [
        (0, 100),
        (100, 150),
        (150, 200),
        (200, 300),
        (300, None),
    ]
    assert [
        (band["lower"], band["upper"]) for band in pensions["band_document"]["bands"]
    ] == [(0.01, 125), *((125 * k, 125 * (k + 1)) for k in range(1, 25)), (3125, None)]
    entries = {entry["name"]: entry for entry in _reads()}
    for row in rows["wages"]:
        assert row["expected_member_count"] == 24
        assert "aggregate_fact_key" not in row["ledger_selector"]
        assert len(entries[row["name"]]["base_facts"]) == 24
    # The unknown-pay class has no row and no band.
    assert generator.WAGE_UNKNOWN_CLASS not in {
        row["ledger_selector"]["dimension_values"][generator.WAGE_CLASS]
        for row in rows["wages"]
    }
    evidence = _document("reference_inventory.json")["quantile_surfaces"]
    known = evidence["wages"]
    assert [row["reference"] for row in known["classes"]] == [
        row["name"] for row in rows["wages"]
    ]
    assert (
        math.fsum(row["workers"] for row in known["classes"])
        == (known["known_pay_workers"])
    )
    assert (
        known["known_pay_workers"] + known["unknown_pay"]["workers"]
        == (known["table_workers"])
    )
    assert known["known_plus_unknown_equals_table"] is True
    assert len(known["unknown_pay"]["facts"]) == 24
    band_counts = evidence["legal_pensions"]
    assert (
        math.fsum(row["recipients"] for row in band_counts["bands"])
        == (band_counts["total_fact"]["recipients"])
    )
    assert band_counts["bands_sum_to_total"] is True


@pytest.mark.parametrize(
    ("mutant", "kind"),
    [
        ("wage_row_counts_jobs", "wage_cells"),
        ("wage_row_reads_jobs_record_sets", "wage_cells"),
        ("wage_row_reads_another_quarter", "wage_cells"),
        ("wage_row_parked", "wage_row"),
        ("pension_row_reads_the_total", "pension_band"),
        ("pension_row_reads_expenditure", "pension_band"),
        ("pension_row_reads_another_group", "pension_band"),
        ("pension_row_activated", "pension_active"),
        ("band_gap", "band_gap"),
        ("band_open_interior", "band_open_interior"),
        ("band_negative_floor", "band_first_lower"),
        ("band_reference_outside_precal", "band_reference"),
        ("band_reference_repeated", "band_duplicate_reference"),
        ("band_document_carries_metadata", "band_shape"),
        ("bands_out_of_row_order", "band_rows"),
        ("provenance_label_swapped", "provenance"),
        ("surface_status_dropped", "status"),
    ],
)
def test_surface_mutants_are_caught(mutant, kind):
    documents, surfaces = copy.deepcopy(_documents()), _document(SURFACES)
    rows = _surface_rows(documents)
    wage, pension = rows["wages"][0]["ledger_selector"], rows["legal_pensions"][0]
    bands = surfaces["wages"]["band_document"]["bands"]
    if mutant == "wage_row_counts_jobs":
        wage["source_measure_id"], wage["source_concept"] = "jobs", "rsz.jobs"
        wage["record_set_id"] = [
            name.replace(".employed_workers", ".jobs") for name in wage["record_set_id"]
        ]
    elif mutant == "wage_row_reads_jobs_record_sets":
        wage["record_set_id"][0] = wage["record_set_id"][0].replace(
            ".employed_workers", ".jobs"
        )
    elif mutant == "wage_row_reads_another_quarter":
        wage["period_value"] = "2025-Q4"
        wage["record_set_id"] = [
            name.replace("2026Q1", "2025Q4") for name in wage["record_set_id"]
        ]
    elif mutant == "wage_row_parked":
        rows["wages"][0]["metadata"]["activation_status"] = "requires_d1902"
    elif mutant == "pension_row_reads_the_total":
        pension["ledger_selector"]["dimension_values"][generator.PENSION_BAND] = (
            generator.PENSION_TOTAL_LABEL
        )
    elif mutant == "pension_row_reads_expenditure":
        pension["ledger_selector"]["source_measure_id"] = "monthly_expenditure_eur"
        pension["ledger_selector"]["record_set_id"] = (
            "sfpd.amount_bands.jan2025.group1.monthly_expenditure_eur"
        )
    elif mutant == "pension_row_reads_another_group":
        pension["ledger_selector"]["record_set_id"] = (
            "sfpd.amount_bands.jan2025.group2.recipients"
        )
        pension["ledger_selector"]["dimension_values"]["sfpd.pension_category"] = (
            "Gehuwd: Rust gezinsbedrag"
        )
    elif mutant == "pension_row_activated":
        pension["metadata"]["activation_status"] = "active"
    elif mutant == "band_gap":
        bands[1]["lower"] = 101
    elif mutant == "band_open_interior":
        bands[1]["upper"] = None
    elif mutant == "band_negative_floor":
        bands[0]["lower"] = -1
    elif mutant == "band_reference_outside_precal":
        bands[0]["reference"] = "statbel_b4_declarations__joint"
    elif mutant == "band_reference_repeated":
        bands[1]["reference"] = bands[0]["reference"]
    elif mutant == "band_document_carries_metadata":
        surfaces["wages"]["band_document"]["unit"] = "eur_per_day"
    elif mutant == "bands_out_of_row_order":
        bands[0]["reference"], bands[1]["reference"] = (
            bands[1]["reference"],
            bands[0]["reference"],
        )
    elif mutant == "provenance_label_swapped":
        surfaces["wages"]["band_provenance"][0]["printed_label"] = "100 - 149"
    else:
        surfaces["legal_pensions"]["status"] = "active"
    assert kind in {
        violation[0] for violation in _surface_violations(documents, surfaces)
    }


@functools.cache
def _precal_names():
    return tuple(row["name"] for row in _rows("precal"))


@functools.cache
def _other_names():
    return tuple(
        row["name"]
        for reference_set in ("calibration", "holdout")
        for row in _rows(reference_set)
    )


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    edges=st.lists(
        st.integers(min_value=0, max_value=10**6), min_size=3, max_size=30, unique=True
    ).map(sorted),
    open_top=st.booleans(),
    rupture=st.sampled_from(
        [
            "overlap",
            "hole",
            "swap",
            "zero_width",
            "open_interior",
            "negative_floor",
            "foreign_reference",
            "repeated_reference",
            "metadata",
        ]
    ),
    data=st.data(),
)
def test_band_documents_hold_their_properties_and_each_break_refuses(
    edges, open_top, rupture, data
):
    """Property: contiguous increasing edges over precal rows pass; one break refuses.

    The kernel ranks onto these edges. A gap, an overlap, an open interior
    band, a negative floor, or a reference outside precal or used twice would
    change the shares it divides or the population it reads.
    """

    names = _precal_names()
    bounds = list(
        zip(edges, [*edges[1:], None] if open_top else edges[1:], strict=False)
    )
    references = data.draw(
        st.lists(
            st.sampled_from(names),
            min_size=len(bounds),
            max_size=len(bounds),
            unique=True,
        )
    )
    document = {
        "bands": [
            {"lower": lower, "upper": upper, "reference": name}
            for (lower, upper), name in zip(bounds, references, strict=True)
        ]
    }
    assert _band_violations(document, names) == []
    bands = document["bands"]
    index = data.draw(st.integers(min_value=0, max_value=len(bands) - 2))
    if rupture == "overlap":
        bands[index]["upper"] += data.draw(st.integers(min_value=1, max_value=10))
        expected = "gap"
    elif rupture == "hole":
        bands[index + 1]["lower"] += data.draw(st.integers(min_value=1, max_value=10))
        expected = "gap"
    elif rupture == "swap":
        # A swapped open top band becomes an open interior band.
        bands[index], bands[index + 1] = bands[index + 1], bands[index]
        expected = "open_interior" if bands[index]["upper"] is None else "gap"
    elif rupture == "zero_width":
        bands[index]["upper"] = bands[index]["lower"]
        expected = "order"
    elif rupture == "open_interior":
        bands[index]["upper"] = None
        expected = "open_interior"
    elif rupture == "negative_floor":
        bands[0]["lower"] = -data.draw(st.integers(min_value=1, max_value=10**6))
        expected = "first_lower"
    elif rupture == "foreign_reference":
        bands[index]["reference"] = data.draw(st.sampled_from(_other_names()))
        expected = "reference"
    elif rupture == "repeated_reference":
        bands[index + 1]["reference"] = bands[index]["reference"]
        expected = "duplicate_reference"
    else:
        document["unit"] = "eur_per_day"
        expected = "shape"
    assert expected in _band_violations(document, names)


def _member_facts(row, values):
    """Synthetic facts filling a row's members: resolver probes, not Belgian values."""

    selector = row["ledger_selector"]
    if "aggregate_fact_key" in selector:
        return [_synthetic_fact(selector, values[0])]
    pins = selector["dimension_values"]
    names = sorted(pins)
    record_sets = selector["record_set_id"]
    members = itertools.product(*(_listed(pins[name]) for name in names))
    return [
        _synthetic_fact(
            {
                **selector,
                "aggregate_fact_key": f"synthetic:{row['name']}:{index}",
                "record_set_id": record_sets[index % len(record_sets)],
                "dimension_values": dict(zip(names, member, strict=True)),
            },
            values[index],
        )
        for index, member in enumerate(members)
    ]


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(surface=st.sampled_from(["wages", "legal_pensions"]), data=st.data())
def test_band_shares_do_not_depend_on_compilation_order(surface, data):
    """Property: the shares follow the band document, whatever order compiles.

    Differential: each share equals its row's count over the total, computed
    exactly. Pension rows compile through _executable, as B3's parked rows do.
    """

    documents = _documents()
    rows = [_executable(row) for row in _surface_rows(documents)[surface]]
    document = _document(SURFACES)[surface]["band_document"]
    facts, counts = [], []
    for row in rows:
        size = _declared_members(row) or 1
        values = data.draw(
            st.lists(
                st.integers(min_value=1, max_value=10**9), min_size=size, max_size=size
            )
        )
        facts += _member_facts(row, values)
        counts.append(sum(values))
    shares = []
    for _ in range(2):
        references = _parsed(
            data.draw(st.permutations(rows)), documents["precal"]["hierarchy"]
        )
        registry = compile_ledger_target_references(
            data.draw(st.permutations(facts)), references, country="be"
        )
        shares.append(
            _surface_bands(document, SimpleNamespace(registry=registry), unit="person")
        )
    total = sum(counts)
    assert shares[0] == shares[1]
    assert [band["share"] for band in shares[0]] == [
        float(Fraction(count, total)) for count in counts
    ]


def _printed_cents(cents):
    euros, rest = divmod(cents, 100)
    return f"{euros:,}".replace(",", ".") + f",{rest:02d}"


def _pension_fact(lower, last_cent, label=None):
    """A synthetic SFPD band fact with Chronicle's constraint shape (cents)."""

    constraints = [
        {
            "operator": ">=",
            "unit": "eur",
            "value": lower / 100,
            "variable": generator.PENSION_AMOUNT,
        }
    ]
    if last_cent is not None:
        constraints.append(
            {
                "operator": "<=",
                "unit": "eur",
                "value": last_cent / 100,
                "variable": generator.PENSION_AMOUNT,
            }
        )
    if label is None:
        label = _printed_cents(lower) + (
            " en >" if last_cent is None else f" - {_printed_cents(last_cent)}"
        )
    return {
        "dimensions": {generator.PENSION_BAND: label},
        "universe_constraints": {"constraints": constraints},
    }


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    cuts=st.lists(st.integers(min_value=2, max_value=10**7), max_size=30, unique=True),
    order=st.randoms(use_true_random=False),
)
def test_pension_bands_tile_any_cent_ladder_from_one_cent(cuts, order):
    """Property: bands bounded to the cent below the next band tile half-open."""

    lowers = [1, *sorted(cuts)]
    facts = [
        _pension_fact(lower, None if following is None else following - 1)
        for lower, following in zip(lowers, [*lowers[1:], None], strict=True)
    ]
    order.shuffle(facts)
    tiled = generator._pension_bands(facts)
    assert [band[0] for band in tiled] == [Fraction(lower, 100) for lower in lowers]
    assert [band[1] for band in tiled] == [
        *(Fraction(lower, 100) for lower in lowers[1:]),
        None,
    ]


@pytest.mark.parametrize(
    "facts",
    [
        # A cent missing between bands, and two bands sharing one.
        [_pension_fact(1, 12498), _pension_fact(12500, None)],
        [_pension_fact(1, 12500), _pension_fact(12500, None)],
        # An open band below the top, a closed top, a floor above one cent.
        [_pension_fact(1, None), _pension_fact(12500, None)],
        [_pension_fact(1, 12499), _pension_fact(12500, 24999)],
        [_pension_fact(100, 12499), _pension_fact(12500, None)],
        # A label that disagrees with its constraints.
        [_pension_fact(1, 12499, "0,01 - 125,99"), _pension_fact(12500, None)],
        # The review's mutant: a band that ends below its floor, with the next
        # band starting one cent above that end, so adjacency alone holds.
        [
            _pension_fact(1, 24999),
            _pension_fact(25000, 24999),
            _pension_fact(25000, None),
        ],
        # The same mutant on the first band, which ends at cent 0.
        [_pension_fact(1, 0), _pension_fact(1, None)],
    ],
)
def test_pension_band_gaps_and_label_drift_refuse(facts):
    with pytest.raises(ValueError, match="SFPD"):
        generator._pension_bands(facts)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    cuts=st.lists(
        st.integers(min_value=2, max_value=10**7), min_size=2, max_size=30, unique=True
    ),
    data=st.data(),
)
def test_pension_bands_accept_only_strictly_rising_floors(cuts, data):
    """Property: every ladder the reader accepts has non-empty bands.

    Start from a valid ladder. Sometimes end one closed band some cents below
    its floor, and start the next band one cent above that end, so that pair
    still looks adjacent. The reader must either refuse, or return bands whose
    floors rise strictly, each below its upper edge.
    """

    lowers = [1, *sorted(cuts)]
    lasts = [following - 1 for following in lowers[1:]] + [None]
    if data.draw(st.booleans()):
        index = data.draw(st.integers(min_value=0, max_value=len(lowers) - 2))
        drop = data.draw(st.integers(min_value=0, max_value=lowers[index] - 1))
        lasts[index] = lowers[index] - 1 - drop
        lowers[index + 1] = lasts[index] + 1
    facts = [
        _pension_fact(lower, last) for lower, last in zip(lowers, lasts, strict=True)
    ]
    try:
        tiled = generator._pension_bands(facts)
    except ValueError as error:
        assert "SFPD" in str(error)
        return
    floors = [band[0] for band in tiled]
    assert floors == sorted(set(floors))
    assert all(band[1] is None or band[0] < band[1] for band in tiled)


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
        "assertion": selector.get("assertion", "observation"),
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


def test_only_demography_commune_support_and_wage_classes_are_active():
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
        "precal": {"statbel_population_by_commune", generator.WAGE_CONTRACT},
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
    # The pension surface rows are parked, so each is a named gap.
    for row in _rows("precal"):
        parked = row["metadata"]["activation_status"] != "active"
        assert (f"precal_references.{row['name']}" in str(error.value)) == parked
    assert not (_SPEC / "transport_graph.json").exists()
    assert not (_SPEC / "receipt_contract.json").exists()
    assert not (_SPEC / "currency_bridge.json").exists()
    # Only the surface rows' metadata and the inventory name the band resource.
    readers = {
        path.name
        for path in _SPEC.glob("*.json")
        if path.name != SURFACES and SURFACES in path.read_text()
    }
    assert readers == {
        "country_package.json",
        "precal_references.json",
        "reference_inventory.json",
    }


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
    pool = [
        fact
        for record_set in _listed(selector.get("record_set_id"))
        for fact in by_record_set.get(record_set, [])
    ]
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
            expected = math.prod(len(roster) for roster in rosters)
            assert len(base) == expected, row["name"]
            assert len({fact["aggregate_fact_key"] for fact in base}) == expected
            assert _declared_members(row) in {None, expected}, row["name"]
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
            elif row.get("value_operation") == "sum":
                assert spec.metadata["ledger_member_fact_count"] == str(expected)
                assert spec.value == math.fsum(float(fact["value"]) for fact in base)
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
            assert _keys(entry["base_facts"]) == sorted(
                {fact["aggregate_fact_key"] for fact in base}
            )
            assert _keys(entry["operand_facts"]) == sorted(
                {fact["aggregate_fact_key"] for fact in operands}
            )
            assert _keys(entry["semantic_facts"]) == sorted(
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


WAGE_WORKERS = [100_457, 1_218_251, 1_182_699, 1_066_243, 369_203]
PENSION_TOTAL_KEY = "ledger.aggregate_fact.v2:f9dc8ab69a3b6cbcc0db3f68"
# A test fixture for the open top band, never a Belgian method value: the
# tail shape is d1902 choice 24 (wages), and no choice yet names the pension tail.
PARETO_FIXTURE = {"method": "pareto", "alpha": 2.0}


def _quarter_cells(by_record_set, spec):
    return [
        fact
        for facts in by_record_set.values()
        for fact in facts
        if fact["layout"].get("record_set_spec_id") == spec
        and fact["period"]["value"] == generator.WAGE_PERIOD
        and fact["observed_measure"]["source_measure_id"] == generator.WAGE_MEASURE
    ]


def _by_class(facts):
    totals = {}
    for fact in facts:
        label = fact["dimensions"][generator.WAGE_CLASS]
        totals[label] = totals.get(label, 0) + fact["value"]
    return totals


def test_wage_classes_resolve_to_24_disjoint_members(by_record_set):
    rows = _surface_rows(_documents())["wages"]
    seen, workers = set(), []
    for row in rows:
        facts = _matched(row["ledger_selector"], by_record_set)
        keys = {fact["aggregate_fact_key"] for fact in facts}
        assert len(facts) == len(keys) == 24, row["name"]
        assert keys == set(_keys(_entry(_reads(), row["name"])["base_facts"]))
        assert not keys & seen
        seen |= keys
        assert (
            len(
                {
                    (f["dimensions"]["rsz.age_class"], f["dimensions"]["rsz.sex"])
                    for f in facts
                }
            )
            == 24
        )
        assert {
            (
                fact["layout"]["record_set_spec_id"],
                fact["observed_measure"]["source_measure_id"],
                fact["entity"]["name"],
                fact["observed_measure"]["unit"],
                fact["period"]["value"],
            )
            for fact in facts
        } == {(generator.WAGE_SPEC, "employed_workers", "person", "count", "2026-Q1")}
        workers.append(sum(fact["value"] for fact in facts))
    assert workers == WAGE_WORKERS
    assert sum(workers) == 3_936_853
    table = _quarter_cells(by_record_set, generator.WAGE_SPEC)
    unknown = [
        fact
        for fact in table
        if fact["dimensions"][generator.WAGE_CLASS] == generator.WAGE_UNKNOWN_CLASS
    ]
    assert len(unknown) == 24 and not {f["aggregate_fact_key"] for f in unknown} & seen
    assert sum(fact["value"] for fact in unknown) == 191_700
    assert sum(fact["value"] for fact in table) == 4_128_553 == sum(workers) + 191_700
    evidence = _document("reference_inventory.json")["quantile_surfaces"]["wages"]
    assert [row["workers"] for row in evidence["classes"]] == WAGE_WORKERS
    assert _keys(evidence["unknown_pay"]["facts"]) == sorted(
        fact["aggregate_fact_key"] for fact in unknown
    )
    assert (evidence["known_pay_workers"], evidence["table_workers"]) == (
        3_936_853,
        4_128_553,
    )
    # Diagnostic: T7 crosses activity and status, and agrees class by class.
    t7 = _by_class(_quarter_cells(by_record_set, generator.WAGE_CROSS_CHECK_SPEC))
    assert t7 == _by_class(table)
    assert evidence["t7_cross_check"]["workers_by_class"] == {
        label: float(value) for label, value in sorted(t7.items())
    }
    assert evidence["t7_cross_check"]["equals_t17_by_class"] is True


def test_each_wage_member_cell_is_one_of_its_row_cells(by_record_set):
    """The cell a holdout pinning a member builds is the row's; the rest hold no fact."""

    for row in _surface_rows(_documents())["wages"]:
        record_sets = row["ledger_selector"]["record_set_id"]
        # A cell names its record set, so only that record set's facts can fill it.
        filled = {
            cell
            for record_set in record_sets
            for fact in by_record_set[record_set]
            for cell in _row_cells({"ledger_selector": generator._selector(fact)})
        }
        own = {
            cell
            for fact in _matched(row["ledger_selector"], by_record_set)
            for cell in _row_cells({"ledger_selector": generator._selector(fact)})
        }
        assert len(own) == 24 and own <= _row_cells(row)
        assert not (_row_cells(row) - own) & filled


def test_pension_bands_sum_to_their_published_total(by_record_set):
    rows = _surface_rows(_documents())["legal_pensions"]
    resolved = [_matched(row["ledger_selector"], by_record_set) for row in rows]
    assert all(len(facts) == 1 for facts in resolved)
    bands = [facts[0] for facts in resolved]
    group = by_record_set[generator.PENSION_RECORD_SET]
    (total,) = [
        fact for fact in group if fact["aggregate_fact_key"] == PENSION_TOTAL_KEY
    ]
    assert total["dimensions"][generator.PENSION_BAND] == generator.PENSION_TOTAL_LABEL
    assert sum(fact["value"] for fact in bands) == total["value"] == 2_449_233
    # Every recipient band of group 1 is read once; the total never is.
    assert {fact["aggregate_fact_key"] for fact in bands} == {
        fact["aggregate_fact_key"] for fact in group if fact is not total
    }
    expenditure = by_record_set[
        "sfpd.amount_bands.jan2025.group1.monthly_expenditure_eur"
    ]
    assert expenditure and not {fact["aggregate_fact_key"] for fact in expenditure} & {
        fact["aggregate_fact_key"] for fact in bands
    }
    # The band document's edges are Chronicle's bounds, tiled half-open.
    tiled = generator._pension_bands(bands)
    surface = _document(SURFACES)["legal_pensions"]["band_document"]["bands"]
    assert [
        (
            generator._number(lower),
            None if upper is None else generator._number(upper),
        )
        for lower, upper, *_ in tiled
    ] == [(band["lower"], band["upper"]) for band in surface]
    evidence = _document("reference_inventory.json")["quantile_surfaces"]
    assert evidence["legal_pensions"]["total_fact"]["aggregate_fact_key"] == (
        PENSION_TOTAL_KEY
    )
    assert [row["recipients"] for row in evidence["legal_pensions"]["bands"]] == [
        float(fact["value"]) for fact in bands
    ]
    entries = {entry["name"]: entry for entry in _reads()}
    assert {
        entries[row["name"]]["partition_total"]["semantic_fact_key"] for row in rows
    } == {total["semantic_fact_key"]}


def _compiled_surface(artifact, by_record_set, rows):
    """The decoded target surface a quantile-map node would receive for rows."""

    precal = _document(SETS["precal"])
    document = {key: precal[key] for key in REFERENCE_DOCUMENT_KEYS if key in precal}
    document["target_references"] = rows
    references = parse_reference_document(document, country="be")
    needed = {
        fact["aggregate_fact_key"]: fact
        for reference in references
        for selector in reference_fact_selectors(reference)
        for fact in _matched(selector, by_record_set)
    }
    registry = compile_ledger_target_references(
        needed.values(), references, country="be"
    )
    payload = encode_target_surface(
        registry,
        references,
        references_sha256=reference_document_sha256(document),
        facts={
            "facts_sha256": artifact.facts_sha256,
            "manifest_sha256": artifact.manifest_sha256,
            "n_facts": len(artifact.facts),
        },
    )
    return decode_target_surface(payload)


@pytest.mark.parametrize(
    ("surface", "counts"),
    [("wages", WAGE_WORKERS), ("legal_pensions", None)],
)
def test_surfaces_compile_into_shares_the_quantile_map_takes(
    artifact, by_record_set, surface, counts
):
    rows = _surface_rows(_documents())[surface]
    if surface == "legal_pensions":
        rows = [_executable(row) for row in rows]  # parked until rank_only lands
    compiled = _compiled_surface(artifact, by_record_set, rows)
    entry = _document(SURFACES)[surface]
    bands = _surface_bands(entry["band_document"], compiled, unit="person")
    values = [spec.value for spec in compiled.registry.specs]
    if counts is not None:
        assert values == counts
    shares = [band["share"] for band in bands]
    assert all(share > 0 for share in shares)
    assert math.isclose(math.fsum(shares), 1, rel_tol=0, abs_tol=1e-12)
    assert shares == [value / math.fsum(values) for value in values]
    assert [(band["lower"], band["upper"]) for band in bands] == [
        (band["lower"], band["upper"]) for band in entry["band_document"]["bands"]
    ]
    donors = np.geomspace(1, 10**6, 400)
    mapped = quantile_map(
        donors, np.ones_like(donors), bands, interpolation=PARETO_FIXTURE
    )
    assert np.all(np.diff(mapped) > 0) and mapped[0] >= bands[0]["lower"]
    # Documented limit: uniform interpolation refuses the open top band.
    with pytest.raises(ValueError, match="Uniform interpolation requires finite"):
        quantile_map(donors, np.ones_like(donors), bands, interpolation="uniform")


def test_jobs_counts_pass_the_kernel_check_but_not_the_surface_check(
    artifact, by_record_set
):
    """_surface_bands can't tell T17's jobs from its employed workers; this test can."""

    documents = copy.deepcopy(_documents())
    rows = _surface_rows(documents)["wages"]
    for row in rows:
        selector = row["ledger_selector"]
        selector["source_measure_id"], selector["source_concept"] = "jobs", "rsz.jobs"
        selector["record_set_id"] = [
            name.replace(".employed_workers", ".jobs")
            for name in selector["record_set_id"]
        ]
        assert len(_matched(selector, by_record_set)) == 24
    compiled = _compiled_surface(artifact, by_record_set, rows)
    bands = _surface_bands(
        _document(SURFACES)["wages"]["band_document"], compiled, unit="person"
    )
    assert [spec.value for spec in compiled.registry.specs] != WAGE_WORKERS
    assert len(bands) == 5
    assert {
        kind for kind, _ in _surface_violations(documents, _document(SURFACES))
    } == {"wage_cells"}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("relabel_a_wage_cell", "daily-wage classes changed"),
        ("drop_a_wage_cell", "not one cell per age and sex"),
        ("move_a_wage_cell_to_the_other_sex_set", "not one cell per age and sex"),
        ("drop_a_pension_band", "leaves a gap or overlap"),
        ("change_a_pension_band_count", "sum to their published total"),
        ("change_a_pension_band_bound", "disagrees with its constraints"),
    ],
)
def test_a_changed_surface_fact_refuses_generation(artifact, change, message):
    facts = list(artifact.facts)

    def first(record_set, label=None):
        return next(
            index
            for index, fact in enumerate(facts)
            if fact["layout"].get("record_set_id") == record_set
            and (label is None or label in fact["dimensions"].values())
        )

    wage = first(WAGE_MEMBER["record_set_id"], "-20")
    band = first(generator.PENSION_RECORD_SET, "125,00 - 249,99")
    if change == "relabel_a_wage_cell":
        facts[wage] = copy.deepcopy(facts[wage])
        facts[wage]["dimensions"][generator.WAGE_CLASS] = "< 100"
    elif change == "drop_a_wage_cell":
        del facts[wage]
    elif change == "move_a_wage_cell_to_the_other_sex_set":
        facts[wage] = copy.deepcopy(facts[wage])
        facts[wage]["layout"]["record_set_id"] = (
            "rsz.t17.2026Q1.1188cc8134270812.employed_workers"
        )
    elif change == "drop_a_pension_band":
        del facts[band]
    elif change == "change_a_pension_band_count":
        facts[band] = {**facts[band], "value": facts[band]["value"] + 1}
    else:
        facts[band] = copy.deepcopy(facts[band])
        for row in facts[band]["universe_constraints"]["constraints"]:
            if row["variable"] == generator.PENSION_AMOUNT and row["operator"] == "<=":
                row["value"] = 249.98
    inventory = generator.Inventory(facts)
    with pytest.raises(ValueError, match=message):
        generator.wage_surface_references(inventory)
        generator.pension_surface_references(inventory)
