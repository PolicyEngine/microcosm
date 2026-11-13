"""The NZ golden-08 differential case: generated, valid against its oracle, traced.

The engine-free tests check that the committed case is the generator's output,
that the differential tool's own validation accepts it against the committed
oracle extract, that the extract agrees with the committed golden-08 fixture,
that the packaged pipeline reproduces every input the oracle states, and that
every table and concept-frame cell names its source. The engine-gated tests run
the differential and check that perturbing the case's placeholder and
secondary cells moves no compared output.
"""

from __future__ import annotations

import importlib.util
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def test_committed_case_is_the_generators_output():
    assert generator.render(generator.build_case()) == CASE_TEXT


def test_case_passes_the_tools_validation_against_the_extract():
    expected = driver._validate_case(CASE, EXTRACT)
    declared = [
        (row["call"], row["column"], row["tolerance"]) for row in CASE["comparisons"]
    ]
    assert declared == [
        (call, column, tolerance)
        for call, _, column, tolerance in generator.COMPARISONS
    ]
    assert {
        column
        for _, section, column, _ in generator.COMPARISONS
        if section == "response"
    } == set(extractor.OUTPUTS)
    for row, values in zip(CASE["comparisons"], expected, strict=True):
        assert values.tolist() == [
            float(driver._pointer(EXTRACT, row["oracle_pointer"]))
        ]


def test_extract_agrees_with_the_committed_golden_08_fixture():
    request = EXTRACT["golden_08"]["request"]
    assert {
        name: {"kind": row["kind"], "value": row["value"]}
        for name, row in request["inputs"].items()
    } == FIXTURE["inputs"]
    outputs = EXTRACT["golden_08"]["response"]["outputs"]
    assert list(outputs) == list(extractor.OUTPUTS)
    assert set(outputs) == set(FIXTURE["outputs"])
    for name, row in outputs.items():
        assert {"kind": row["kind"], "value": row["value"]} == FIXTURE["outputs"][name]
    assert EXTRACT["oracle"]["commit"] == FIXTURE["source"]["commit"]
    assert request["entity_id"]["value"] == FIXTURE["entity_id"]
    assert request["period"]["value"] == FIXTURE["period"]
    positions = [row["pointer"] for row in request["inputs"].values()]
    assert positions == [
        f"/requests/8/request/dataset/inputs/{index}/value/value"
        for index in range(len(positions))
    ]


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
    requests = [
        ("golden-08", golden["request"]),
        *EXTRACT["household_requests"].items(),
    ]
    stated = [
        (name, column)
        for name, request in requests
        for column in request["inputs"]
        if name != "golden-08" or column not in golden["bridge_inputs"]
    ]
    assert [(row["request"], row["column"]) for row in records] == stated
    differ = {
        (row["entity"], row["column"]) for row in records if row["status"] != "equal"
    }
    assert differ == set(generator.EXPECTED_CLOSURE_CONFLICTS)
    assert all(row["exact"] for row in records if row["status"] == "equal")
    parent = {
        "person": CASE["tables"]["person"][0],
        "family": CASE["tables"]["family"][0],
    }
    for row in records:
        assert driver._pointer(EXTRACT, row["extract_pointer"]) == row["oracle"]
        table = CASE["provenance"]["tables"][row["entity"]]
        if row["column"] in parent[row["entity"]]:
            assert parent[row["entity"]][row["column"]] == row["pipeline"]
            assert table[row["column"]]["checked_against"] == row["extract_pointer"]


def test_masks_are_family_data_and_the_tool_can_build_every_call():
    assert MASK_COLUMNS <= set(CASE["tables"]["family"][0])
    assert all(call["kernel"] != "calibration.prepare@1" for call in CASE["calls"])
    assert [call["id"] for call in CASE["calls"]] == list(generator.CALLS)
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


def _set_person(case, row, value, *columns):
    for column in columns:
        case["tables"]["person"][row][column] = value


def _parent_age(case):
    # The concept frame's parent age; the case uses golden-07's 25.
    _set_person(case, 0, 30, "age", "supported_living_payment_applicant_age")


def _child_ages(case):
    # Test perturbations within the dependent-child age limit, not case data.
    for row, age in ((2, 5), (3, 16)):
        _set_person(
            case,
            row,
            age,
            "age",
            "jobseeker_support_applicant_age",
            "supported_living_payment_applicant_age",
        )


def _family_id(case):
    case["tables"]["family"][0]["family_id"] = 7
    for row in case["tables"]["person"]:
        row["person_family_id"] = 7


def _weight(case):
    case["weights"]["household"]["values"] = [2.0]


# The NZ hub accepted the fixture's tenure as a source (ruling 6, 2026-11-13)
# on the expectation that these two perturbations move no compared output;
# cash assets are now the placeholder CASE_DATA_UNSTATED_LIQUID_FINANCIAL_ASSETS.
# At the pinned rulespec-nz 8dc2507, accommodation_supplement/core.yaml reads
# these four inputs only through
# accommodation_supplement_non_rate_entitlement_requirements and the two
# judgments it combines (lines 403-516); no compared amount reads it. If the
# engine run shows a compared output moving, the hub revisits that ruling.
def _cash_assets(case):
    case["tables"]["family"][0]["accommodation_supplement_cash_assets"] = 1000000.0


def _tenure(case):
    family = case["tables"]["family"][0]
    family["accommodation_supplement_has_accommodation_costs"] = False
    family["accommodation_supplement_rent_paid_to_kainga_ora"] = True
    family[SOCIAL_HOUSING] = True


@needs_engine
@pytest.mark.parametrize(
    "perturb",
    [_parent_age, _child_ages, _family_id, _weight, _cash_assets, _tenure],
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
    case = deepcopy(CASE)
    perturb(case)
    changed = driver.differential(case, EXTRACT, root)
    assert [row["actual"] for row in changed["comparisons"]] == [
        row["actual"] for row in base["comparisons"]
    ]
