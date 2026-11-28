"""The NZ golden-08 differential case: generated, valid against its oracle, traced.

The engine-free tests check that the committed case is the generator's output;
that it declares the five comparisons and tolerances written out here; that the
oracle extract rebuilds each oracle request to the hash the oracle recorded
(written out here), points at each value's place in the oracle files, and
agrees with the committed golden-08 fixture and the packaged pins; that the
packaged pipeline reproduces every input the oracle states, with one known
conflict; that every cell whose provenance names the fixture or the extract
equals the value there; that the unit rule, rules bindings, call order and call
params are the packaged ones; and that every table and concept-frame cell names
its source. The engine-gated tests run the differential, check that perturbing
the case's placeholder and secondary cells (some through the generator's own
pipeline) moves no compared output, and check that a rent change does move one.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from copy import deepcopy
from decimal import Decimal
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.transport import compose, entitlement
from microcosm.build.transport.registry import transport_entity_schema
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.unit_construction import BenefitUnitRule
from microcosm.graph import Node, Owned, Slice
from tools import extract_nz_golden08_oracle as extractor
from tools import generate_nz_golden08_case as generator
from tools import transport_bridge_differential as driver

CASE_TEXT = generator.CASE.read_text(encoding="utf-8")
CASE = json.loads(CASE_TEXT)
EXTRACT = json.loads(generator.EXTRACT.read_text(encoding="utf-8"))
FIXTURE = json.loads(generator.FIXTURE.read_text(encoding="utf-8"))
FIXTURE_FILE = (
    "packages/microcosm-build/tests/fixtures/nz/"
    "axiom_oracle_golden_08_accommodation_supplement.json"
)
REGENERATE = "uv run --no-sync python tools/generate_nz_golden08_case.py"
#: Where a case table cell comes from.
TABLE_SOURCES = {
    "structural",
    "constructed",
    "concept",
    "closure",
    "encoded",
    "unit_attributes",
    "masks",
}
#: Where a concept-frame cell comes from.
FRAME_SOURCES = {
    "structural",
    "constructed",
    "fixture",
    "oracle",
    "oracle_implied",
    "unstated",
}
#: The placeholders that remain: every concept-frame cell and weight no source
#: states.
UNSTATED = {
    "CASE_DATA_UNSTATED_PARENT_MAIN_BENEFIT_FLAG",
    "CASE_DATA_UNSTATED_OTHER_RECEIPT_FLAGS",
    "CASE_DATA_UNSTATED_LIQUID_FINANCIAL_ASSETS",
    "CASE_DATA_UNSTATED_PROPERTY_TAX",
    "CASE_DATA_UNSTATED_USUAL_WEEKLY_HOURS",
    "CASE_DATA_UNSTATED_ENROLLED_FULL_TIME",
    "CASE_DATA_UNSTATED_HAS_DISABILITY",
    "CASE_DATA_UNSTATED_SEX",
    "CASE_DATA_UNSTATED_LEGAL_MARITAL_STATUS",
    "CASE_DATA_UNSTATED_TAKE_UP_SEED",
    "CASE_DATA_UNSTATED_SOURCE_HOUSEHOLD_ID",
    "CASE_DATA_UNSTATED_DONOR_SUPPORT_STRATUM",
    "CASE_DATA_UNSTATED_AS_AREA_ALT",
    "CASE_DATA_UNSTATED_HOUSEHOLD_WEIGHT_KIND",
    "CASE_DATA_UNSTATED_HOUSEHOLD_WEIGHT",
}
SOCIAL_HOUSING = (
    "accommodation_supplement_rent_paid_to_registered_community_housing_provider"
    "_social_housing"
)

MASK_COLUMNS = {
    "accommodation_supplement_bridge_has_dependent_child",
    "accommodation_supplement_bridge_single_without_dependent_child",
}

BASE_RATE = "accommodation_supplement_base_rate_weekly_amount"
CUTOUT = "accommodation_supplement_non_beneficiary_income_cutout_weekly_amount"
QUALIFYING = "accommodation_supplement_weekly_qualifying_accommodation_costs"
BEFORE_ROUNDING = "accommodation_supplement_weekly_amount_before_rounding"
ROUNDED = "accommodation_supplement_rounded_weekly_payment"
#: The five comparisons as (call, column, oracle_pointer, tolerance). The
#: tolerances follow the ruling at
#: https://github.com/PolicyEngine/microcosm/issues/1183#issuecomment-6856239769:
#: 1e-9 for an unsolved weekly amount, the cutout solve's 1e-9 plus 5e-10, and
#: the rounded payment exactly.
COMPARISONS = (
    (
        "nz.scn.S0.bridge.base_rate",
        BASE_RATE,
        f"/golden_08/request/inputs/{BASE_RATE}/value",
        1e-9,
    ),
    (
        "nz.as.bridge.cutout",
        CUTOUT,
        f"/golden_08/request/inputs/{CUTOUT}/value",
        1.5e-9,
    ),
    (
        "nz.scn.S0.as",
        QUALIFYING,
        f"/golden_08/response/outputs/{QUALIFYING}/value",
        1e-9,
    ),
    (
        "nz.scn.S0.as",
        BEFORE_ROUNDING,
        f"/golden_08/response/outputs/{BEFORE_ROUNDING}/value",
        1e-9,
    ),
    (
        "nz.scn.S0.as",
        ROUNDED,
        f"/golden_08/response/outputs/{ROUNDED}/value",
        0,
    ),
)
CALLS = [
    "nz.as.bridge.sole_parent_gross_rate",
    "nz.scn.S0.bridge.base_rate",
    "nz.as.bridge.cutout",
    "nz.scn.S0.as",
]
#: The one input where the pipeline differs from the oracle, as (request,
#: entity, column, oracle value, pipeline value): the closure's default.
KNOWN_CONFLICT = (
    "golden-07",
    "person",
    "jobseeker_support_benefit_commenced_on_or_after_1998_07_01",
    False,
    True,
)
#: Where each oracle request sits in the extract and in the requests file.
REQUESTS = {
    "golden-08": ("/golden_08/request", 8),
    "golden-00": ("/household_requests/golden-00", 0),
    "golden-07": ("/household_requests/golden-07", 7),
}
#: The hash the oracle recorded for each request, written out so that an edit
#: to an extract input that also recomputes the extract's hash fails here.
REQUEST_SHA256 = {
    "golden-08": "50636d0d291ed5cc33d26dc5d419cd87c252095aaae1d0e5b19397c1f5a5c2c1",
    "golden-00": "1ac639dd25ef790c315c1622ca064c20fad7d5f76e7ddbf13f9c5269dc6c5b08",
    "golden-07": "074a15620602b8de42f2bde008e8767b1db112d691bf072a8c7a723c9a4829b2",
}


@cache
def _spec():
    return compose.load_transport_spec("nz")


def _canonical(value):
    """JSON text that tells True from 1 and 25 from 25.0."""
    return json.dumps(value, sort_keys=True)


def _same(left, right):
    return _canonical(left) == _canonical(right)


def _sha256(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _equals_oracle(cell, entry):
    """Whether a table cell is an oracle input's value, exactly, by its kind.

    A bool input needs a bool, an integer input an int (not a bool), and a
    decimal input a number (not a bool) whose exact binary value is the
    decimal's. The closure's defaults for decimal inputs are ints (0), as in
    the graph, so a decimal accepts an int.
    """
    kind, value = entry["kind"], entry["value"]
    if kind == "bool":
        assert type(value) is bool, entry
        return type(cell) is bool and cell is value
    if kind == "integer":
        assert type(value) is int, entry
        return type(cell) is int and cell == value
    assert kind == "decimal" and type(value) is str, entry
    return type(cell) in (int, float) and Decimal(cell) == Decimal(value)


def _entry(extract_pointer):
    """The extract input record a ``.../inputs/<name>/value`` pointer names."""
    head, _, last = extract_pointer.rpartition("/")
    assert last == "value", extract_pointer
    return driver._pointer(EXTRACT, head)


def test_committed_case_is_the_generators_output():
    assert generator.render(generator.build_case()) == CASE_TEXT, (
        "The golden-08 differential case is stale. If the change is intended, "
        f"run `{REGENERATE}`, review the diff and commit it."
    )


def test_case_declares_the_ruled_comparisons_and_passes_the_tools_validation():
    assert CASE["comparisons"] == [
        {
            "call": call,
            "entity": "family",
            "column": column,
            "oracle_pointer": pointer,
            "tolerance": tolerance,
        }
        for call, column, pointer, tolerance in COMPARISONS
    ]
    assert [
        row["column"]
        for row in CASE["comparisons"]
        if row["oracle_pointer"].startswith("/golden_08/request/")
    ] == EXTRACT["golden_08"]["bridge_inputs"]
    assert {
        row["column"]
        for row in CASE["comparisons"]
        if row["oracle_pointer"].startswith("/golden_08/response/")
    } == set(extractor.OUTPUTS)
    expected = driver._validate_case(CASE, EXTRACT)
    for row, values in zip(CASE["comparisons"], expected, strict=True):
        assert values.tolist() == [
            float(driver._pointer(EXTRACT, row["oracle_pointer"]))
        ]


def _rebuilt_request(request):
    """The oracle request an extract record describes, as the requests file
    holds it."""
    period = request["period"]["value"]
    entity_id = request["entity_id"]["value"]
    interval = {"start": period["start"], "end": period["end"]}
    return {
        "dataset": {
            "inputs": [
                {
                    "entity": request["entity"],
                    "entity_id": entity_id,
                    "interval": interval,
                    "name": row["name"],
                    "value": {"kind": row["kind"], "value": row["value"]},
                }
                for row in request["inputs"].values()
            ],
            "relations": [],
        },
        "mode": request["mode"]["value"],
        "queries": [
            {
                "entity_id": entity_id,
                "outputs": request["queried_outputs"]["value"],
                "period": period,
            }
        ],
    }


def test_each_oracle_request_rebuilds_from_the_extract_to_its_recorded_hash():
    golden = EXTRACT["golden_08"]
    households = EXTRACT["household_requests"]
    assert set(households) == {"golden-00", "golden-07"}
    requests = {"golden-08": golden["request"], **households}
    recorded = {
        "golden-08": golden["response"]["request_sha256"],
        **{name: request["request_sha256"] for name, request in households.items()},
    }
    for name, request in requests.items():
        _, index = REQUESTS[name]
        assert request["id"]["value"] == name
        assert recorded[name]["source"] == "full_responses"
        assert recorded[name]["pointer"] == f"/requests/{index}/request_sha256"
        assert recorded[name]["value"] == REQUEST_SHA256[name]
        assert _sha256(_rebuilt_request(request)) == REQUEST_SHA256[name], name


def _token(text):
    """``text`` as one RFC 6901 pointer token."""
    return text.replace("~", "~0").replace("/", "~1")


def test_every_extract_pointer_names_its_place_in_the_oracle_files():
    golden = EXTRACT["golden_08"]
    requests = {"golden-08": golden["request"], **EXTRACT["household_requests"]}
    for name, request in requests.items():
        _, index = REQUESTS[name]
        inputs = request["inputs"]
        # The requests file lists inputs in name order, so this ties each
        # input's pointer index below to its name.
        assert list(inputs) == sorted(inputs), name
        assert [(row["source"], row["pointer"]) for row in inputs.values()] == [
            ("requests", f"/requests/{index}/request/dataset/inputs/{i}/value/value")
            for i in range(len(inputs))
        ], name
    outputs = golden["response"]["outputs"]
    assert {name: (row["source"], row["pointer"]) for name, row in outputs.items()} == {
        name: (
            "full_responses",
            f"/requests/8/response/results/0/outputs/{_token(row['id'])}/value/value",
        )
        for name, row in outputs.items()
    }


def test_extract_is_bound_to_the_packaged_pins():
    assert EXTRACT["golden_08"]["bridge_inputs"] == [BASE_RATE, CUTOUT]
    assert EXTRACT["sources"] == extractor.SOURCES
    bridge = compose.transport_resource(_spec(), "as_rate_bridge")
    source = bridge["differential_expectations"]["source"]
    assert EXTRACT["sources"]["requests"]["sha256"] == source["sha256"]
    assert EXTRACT["oracle"]["commit"] == source["commit"]


def test_extract_agrees_with_the_committed_golden_08_fixture():
    request = EXTRACT["golden_08"]["request"]
    inputs = request["inputs"]
    assert _canonical(
        {
            name: {"kind": row["kind"], "value": row["value"]}
            for name, row in inputs.items()
        }
    ) == _canonical(FIXTURE["inputs"])
    assert request["entity"] == FIXTURE["entity"]
    module = "nz:" + FIXTURE["module"].removeprefix("nz/").removesuffix(".yaml")
    for name, row in inputs.items():
        assert row["name"] == f"{module}#input.{name}"
    outputs = EXTRACT["golden_08"]["response"]["outputs"]
    assert list(outputs) == list(extractor.OUTPUTS)
    assert _canonical(
        {
            name: {"kind": row["kind"], "value": row["value"]}
            for name, row in outputs.items()
        }
    ) == _canonical(FIXTURE["outputs"])
    for name, row in outputs.items():
        assert row["id"] == f"{module}#{name}"
    assert EXTRACT["oracle"]["commit"] == FIXTURE["source"]["commit"]
    assert request["entity_id"]["value"] == FIXTURE["entity_id"]
    assert _same(request["period"]["value"], FIXTURE["period"])


def _cells(record):
    return record if isinstance(record, list) else [record]


def test_every_cell_names_a_source_and_only_known_cells_are_unstated():
    provenance = CASE["provenance"]
    frame = provenance["concept_frame"]
    for entity, rows in CASE["tables"].items():
        assert list(provenance["tables"][entity]) == list(rows[0])
        for record in provenance["tables"][entity].values():
            assert record["from"] in TABLE_SOURCES
            if record["from"] == "concept":
                owner, _, column = record["column"].partition(".")
                assert record["cells"] == frame[owner][column]
            for read in record.get("reads", ()):
                owner, _, column = read.partition(".")
                assert column in frame[owner], read
    unstated = set()
    for columns in frame.values():
        for record in columns.values():
            for cell in _cells(record):
                assert cell["from"] in FRAME_SOURCES
                if cell["from"] == "unstated":
                    unstated.add(cell["constant"])
    for record in provenance["weights"].values():
        if isinstance(record, dict) and record.get("from") == "unstated":
            unstated.add(record["constant"])
    assert unstated == UNSTATED == set(generator.UNSTATED)
    assert set(provenance["unstated"]) == UNSTATED
    for name in UNSTATED:
        assert hasattr(generator, name)
    writers = provenance["graph_start"]["writers"]
    assert {entity: list(columns) for entity, columns in writers.items()} == {
        entity: list(columns) for entity, columns in frame.items()
    }
    assert {
        column: node
        for column, node in writers["person"].items()
        if column.startswith("receives_")
    } == {
        "receives_nz_super": "nz.takeup.assign",
        "receives_supported_living_payment": "nz.takeup.assign",
        "receives_sole_parent_support": "nz.takeup.assign",
        "receives_jobseeker_support": "nz.takeup.assign",
    }


def test_the_parents_age_is_golden_07s_applicant_age():
    pointer = "/household_requests/golden-07/inputs/jobseeker_support_applicant_age"
    cell = CASE["provenance"]["concept_frame"]["person"]["age"][0]
    assert cell["from"] == "oracle"
    assert cell["extract_pointer"] == f"{pointer}/value"
    assert "uses 30" in cell["note"]
    assert FIXTURE["household"]["members"][0]["age"] is None
    stated = driver._pointer(EXTRACT, f"{pointer}/value")
    parent = CASE["tables"]["person"][0]
    tables = CASE["provenance"]["tables"]["person"]
    assert tables["age"]["cells"][0] == cell
    assert parent["age"] == stated
    for column in (
        "jobseeker_support_applicant_age",
        "supported_living_payment_applicant_age",
    ):
        assert tables[column]["from"] == "encoded"
        assert "person.age" in tables[column]["reads"]
        assert parent[column] == stated


def test_the_pipeline_reproduces_every_input_the_oracle_states():
    records = CASE["provenance"]["oracle_cross_check"]
    golden = EXTRACT["golden_08"]
    requests = {"golden-08": golden["request"], **EXTRACT["household_requests"]}
    stated = [
        (name, column)
        for name, request in requests.items()
        for column in request["inputs"]
        if name != "golden-08" or column not in (BASE_RATE, CUTOUT)
    ]
    assert [(row["request"], row["column"]) for row in records] == stated
    conflicts = []
    for row in records:
        prefix, _ = REQUESTS[row["request"]]
        assert row["extract_pointer"] == f"{prefix}/inputs/{row['column']}/value"
        assert row["entity"] == requests[row["request"]]["entity"].lower()
        entry = _entry(row["extract_pointer"])
        assert _same(row["oracle"], entry["value"])
        if row["status"] == "equal":
            assert _equals_oracle(row["pipeline"], entry), row
            assert row["exact"] is True
        else:
            assert row["status"] == "known_closure_conflict", row
            conflicts.append(
                (
                    row["request"],
                    row["entity"],
                    row["column"],
                    row["oracle"],
                    row["pipeline"],
                )
            )
    assert _canonical(conflicts) == _canonical([KNOWN_CONFLICT])
    parent = {
        "person": CASE["tables"]["person"][0],
        "family": CASE["tables"]["family"][0],
    }
    for row in records:
        table = CASE["provenance"]["tables"][row["entity"]]
        if row["column"] in parent[row["entity"]]:
            assert _same(parent[row["entity"]][row["column"]], row["pipeline"])
            assert table[row["column"]]["checked_against"] == row["extract_pointer"]


def test_cells_equal_the_fixture_and_extract_values_their_provenance_names():
    conflict = KNOWN_CONFLICT[:3]
    checked = 0
    for entity, rows in CASE["tables"].items():
        for column, record in CASE["provenance"]["tables"][entity].items():
            if record["from"] == "concept":
                cells = record["cells"]
                cells = cells if isinstance(cells, list) else [cells] * len(rows)
                for row, cell in zip(rows, cells, strict=True):
                    if cell["from"] == "fixture":
                        assert cell["file"] == FIXTURE_FILE
                        value = driver._pointer(FIXTURE, cell["pointer"])
                        assert _same(row[column], value), (entity, column, cell)
                    elif cell["from"] == "oracle":
                        entry = _entry(cell["extract_pointer"])
                        assert _equals_oracle(row[column], entry), (column, cell)
                    else:
                        continue
                    checked += 1
            pointer = record.get("checked_against")
            if pointer is not None:
                request = next(
                    name
                    for name, (prefix, _) in REQUESTS.items()
                    if pointer.startswith(f"{prefix}/")
                )
                if (request, entity, column) == conflict:
                    assert _same(rows[0][column], KNOWN_CONFLICT[4])
                else:
                    entry = _entry(pointer)
                    assert _equals_oracle(rows[0][column], entry), (column, pointer)
                checked += 1
    # Four ages (the parent's from golden-07, three from the fixture) and the
    # 39 oracle-checked inputs: golden-08's 28 less the two bridges,
    # golden-07's 11 and golden-00's 2.
    assert checked == 43


def test_the_family_is_the_fixtures_sole_parent_with_three_children():
    members = FIXTURE["household"]["members"]
    assert [member["role"] for member in members] == [
        "sole parent",
        "child",
        "child",
        "child",
    ]
    assert _same([member["age"] for member in members], [None, 0, 1, 10])
    assert _same([row["age"] for row in CASE["tables"]["person"][1:]], [0, 1, 10])
    family = CASE["tables"]["family"]
    expected = {
        "is_sole_parent": True,
        "is_single": True,
        "is_partnered": False,
        "n_dependent_children": 3,
        "youngest_dependent_child_age": 0,
        "accommodation_supplement_bridge_has_dependent_child": True,
        "accommodation_supplement_bridge_single_without_dependent_child": False,
        # Encoded from the fixture's tenure and checked against no oracle input.
        "accommodation_supplement_has_accommodation_costs": True,
        "accommodation_supplement_rent_paid_to_kainga_ora": False,
        SOCIAL_HOUSING: False,
    }
    assert len(family) == 1
    assert _same({column: family[0][column] for column in expected}, expected)
    # The household's tenure, which those three cells read, is the fixture's.
    assert FIXTURE["household"]["tenure"] == "rented_private"
    assert CASE["provenance"]["concept_frame"]["household"]["tenure"] == {
        "from": "fixture",
        "file": FIXTURE_FILE,
        "pointer": "/household/tenure",
    }
    assert {
        column
        for column, record in CASE["provenance"]["tables"]["family"].items()
        if "household.tenure" in record.get("reads", ())
        and "checked_against" not in record
    } == {
        "accommodation_supplement_has_accommodation_costs",
        "accommodation_supplement_rent_paid_to_kainga_ora",
        SOCIAL_HOUSING,
    }


def test_the_unit_rule_and_call_order_are_the_packaged_ones():
    packaged = compose.transport_resource(_spec(), "benefit_unit_rule")
    rule = BenefitUnitRule.from_spec(packaged)
    assert _same(CASE["unit_rule"], rule.to_dict())
    # The driver loads the case's unit rule this way.
    assert BenefitUnitRule.from_dict(CASE["unit_rule"]) == rule
    assert [call["id"] for call in CASE["calls"]] == CALLS


def _packaged_params(spec, bindings):
    """Each call's node params, resolved from the packaged entitlement graph as
    the entitlement factory resolves them, with each binding's id as its
    engine ref."""
    config = SimpleNamespace(
        engine_refs={row["id"]: row["id"] for row in bindings["bindings"]}
    )
    rows, _ = entitlement._rows(spec)
    params = {}
    for row, scenario in rows:
        if row.get("id") not in CALLS:
            continue
        assert row["id"] not in params, row["id"]
        node = compose._node(
            spec,
            row,
            config,
            {},
            allowed_kernels=entitlement._KERNELS,
            resolved_params={
                name: entitlement._parameter(spec, config, value, scenario)
                for name, value in row.get("params", {}).items()
            },
            profile_selection=None if scenario is None else scenario.get("knobs", {}),
        )
        params[row["id"]] = dict(node.params)
    return params


def test_the_rules_bindings_and_call_params_are_the_packaged_ones():
    spec = _spec()
    bindings = compose.transport_resource(spec, "axiom_rules_bindings")
    # The driver builds its engines from the case's bindings.
    assert _same(CASE["rules_bindings"], bindings)
    packaged = _packaged_params(spec, bindings)
    assert sorted(packaged) == sorted(CALLS)
    for call in CASE["calls"]:
        assert _same(call["params"], packaged[call["id"]]), call["id"]
    # The cutout comparison's 1.5e-9 is this solve tolerance plus 5e-10; the
    # other three calls solve nothing.
    assert {
        call["id"]: call["params"]["tolerance"]
        for call in CASE["calls"]
        if "tolerance" in call["params"]
    } == {"nz.as.bridge.cutout": 1e-9}


def test_masks_are_family_data_and_the_tool_can_build_every_call():
    assert MASK_COLUMNS <= set(CASE["tables"]["family"][0])
    assert all(call["kernel"] != "calibration.prepare@1" for call in CASE["calls"])
    rule = BenefitUnitRule.from_dict(CASE["unit_rule"])
    schema, _ = transport_entity_schema(rule)
    tables = {name: pd.DataFrame(rows) for name, rows in CASE["tables"].items()}
    weights = {
        name: Weights(
            np.asarray(row["values"], dtype=np.float64), WeightKind(row["kind"])
        )
        for name, row in CASE["weights"].items()
    }
    Frame(tables, schema, weights)
    # The ``first`` aggregation reads the lowest person id: it must be the parent.
    person_ids = [row[schema.person_id_column] for row in CASE["tables"]["person"]]
    assert person_ids == sorted(person_ids)
    assert (
        CASE["tables"]["person"][0]["jobseeker_support_dependent_children_count"] == 3
    )
    refs = {
        row["id"]: f"prepared:{row['id']}" for row in CASE["rules_bindings"]["bindings"]
    }
    available = {name: set(table.columns) for name, table in tables.items()}
    for call in CASE["calls"]:
        Node(
            call["id"],
            call["kernel"],
            inputs=tuple(Slice(e, tuple(c)) for e, c in call["inputs"].items()),
            outputs=tuple(Owned(**output) for output in call["outputs"]),
            params=driver._resolve_params(call["params"], refs),
        )
        for entity, columns in call["inputs"].items():
            assert set(columns) <= available[entity]
        for output in call["outputs"]:
            assert output["column"] not in available[output["entity"]]
            available[output["entity"]].add(output["column"])


_ENGINE_INSTALLED = importlib.util.find_spec("axiom_rules_engine") is not None
if _ENGINE_INSTALLED:
    from axiom_rules_engine.dense import NativeCompiledDenseProgram

    _DENSE_AVAILABLE = NativeCompiledDenseProgram is not None
else:
    _DENSE_AVAILABLE = False
RULESPEC_NZ = os.environ.get("POPULACE_RULESPEC_NZ")
needs_engine = pytest.mark.skipif(
    not _DENSE_AVAILABLE or not RULESPEC_NZ,
    reason=(
        "needs axiom_rules_engine with its dense extension and a rulespec-nz "
        "checkout at POPULACE_RULESPEC_NZ"
    ),
)


@needs_engine
def test_the_engine_reproduces_golden_08():
    receipt = driver.differential(deepcopy(CASE), EXTRACT, Path(RULESPEC_NZ))
    assert receipt["status"] == "passed", receipt["comparisons"]


# Concept-frame perturbations: each edits the person and household concept
# tables and the generator's own pipeline rebuilds the case from them, so the
# encoded and unit-attribute cells that read an edited column move with it.
def _parent_age(person, household):
    # golden-07 states 25; the concept frame in test_nz_axiom_input_closure.py
    # uses 30.
    person.loc[person["person_id"] == 1, "age"] = 30


def _child_ages(person, household):
    # The youngest stays under 14, the Jobseeker Support rates' age threshold.
    person.loc[person["person_id"] > 1, "age"] = [3, 6, 10]


def _rent(person, household):
    # 500 a week instead of 600; the rent concept is annual.
    household["rent"] = 500.0 * 52


def _through_the_pipeline(edit, moved):
    def perturb(case):
        case = generator.build_case(concept_edit=edit)
        for entity, row, column, value in moved:
            assert _same(case["tables"][entity][row][column], value), column
        return case

    return perturb


PARENT_AGE_MOVES = [
    ("person", 0, "age", 30),
    ("person", 0, "jobseeker_support_applicant_age", 30),
    ("person", 0, "supported_living_payment_applicant_age", 30),
]
CHILD_AGES_MOVE = [
    ("person", 1, "age", 3),
    ("person", 2, "age", 6),
    ("family", 0, "youngest_dependent_child_age", 3),
    ("family", 0, "n_dependent_children", 3),
]
RENT_MOVES = [("family", 0, "accommodation_supplement_weekly_rent_paid", 500.0)]


def _family_id(case):
    case["tables"]["family"][0]["family_id"] = 7
    for row in case["tables"]["person"]:
        row["person_family_id"] = 7
    return case


def _weight(case):
    case["weights"]["household"]["values"] = [2.0]
    return case


# The case takes the household's tenure from the fixture on the expectation
# that these two perturbations move no compared output; cash assets are the
# placeholder CASE_DATA_UNSTATED_LIQUID_FINANCIAL_ASSETS. At the pinned
# rulespec-nz 8dc2507, accommodation_supplement/core.yaml reads these four
# inputs only through accommodation_supplement_non_rate_entitlement_requirements
# and the two judgments it combines; no compared amount reads it. If the engine
# run shows a compared output moving, the fixture's tenure is no longer a
# sufficient source and the case must state these cells.
def _cash_assets(case):
    case["tables"]["family"][0]["accommodation_supplement_cash_assets"] = 1000000.0
    return case


def _tenure(case):
    family = case["tables"]["family"][0]
    family["accommodation_supplement_has_accommodation_costs"] = False
    family["accommodation_supplement_rent_paid_to_kainga_ora"] = True
    family[SOCIAL_HOUSING] = True
    return case


@needs_engine
@pytest.mark.parametrize(
    "perturb",
    [
        _through_the_pipeline(_parent_age, PARENT_AGE_MOVES),
        _through_the_pipeline(_child_ages, CHILD_AGES_MOVE),
        _family_id,
        _weight,
        _cash_assets,
        _tenure,
    ],
    ids=[
        "parent-age",
        "child-ages",
        "family-id",
        "household-weight",
        "cash-assets",
        "tenure",
    ],
)
def test_placeholder_and_secondary_cells_move_no_compared_output(perturb):
    root = Path(RULESPEC_NZ)
    base = driver.differential(deepcopy(CASE), EXTRACT, root)
    assert base["status"] == "passed", base["comparisons"]
    changed = driver.differential(perturb(deepcopy(CASE)), EXTRACT, root)
    assert [row["actual"] for row in changed["comparisons"]] == [
        row["actual"] for row in base["comparisons"]
    ]


@needs_engine
def test_a_rent_change_through_the_pipeline_moves_the_qualifying_costs():
    # The positive control: the comparison detects a perturbation that must
    # move a compared amount (core.yaml's qualifying costs are the weekly rent
    # paid less accommodation income, which is 0 here).
    root = Path(RULESPEC_NZ)
    base = driver.differential(deepcopy(CASE), EXTRACT, root)
    assert base["status"] == "passed", base["comparisons"]
    case = _through_the_pipeline(_rent, RENT_MOVES)(deepcopy(CASE))
    changed = driver.differential(case, EXTRACT, root)
    moved = {
        row["column"]: (row["actual"], before["actual"])
        for row, before in zip(changed["comparisons"], base["comparisons"], strict=True)
        if row["actual"] != before["actual"]
    }
    assert QUALIFYING in moved, changed["comparisons"]
    assert changed["status"] == "failed"
