#!/usr/bin/env python3
"""Generate the NZ golden-08 bridge/AS differential case from the packaged spec.

The case is the ``--case`` input of tools/transport_bridge_differential.py; the
golden-08 oracle extract (tools/extract_nz_golden08_oracle.py) is its
``--oracle``. Its calls are the packaged entitlement nodes, resolved as the
entitlement factory resolves them: ``entitlement._parameter`` for each param,
then ``compose._node``, with each rules binding's id as its own engine
reference. The tool swaps those ids for prepared references. No param is
copied by hand.

Calls, in order: nz.as.bridge.sole_parent_gross_rate,
nz.scn.S0.bridge.base_rate, nz.as.bridge.cutout and nz.scn.S0.as. The masks
node nz.as.bridge.masks is not a call: ``calibration.prepare@1`` refuses a node
that lacks its typed ``preparation`` output (``graph_inputs.require_outputs``),
and the tool's nodes declare no typed outputs. Its two columns are family table
data instead, computed by ``prepare_calibration_columns`` (the function that
kernel calls) from the masks node's own ``bindings``.

The tables come from the packaged pipeline, run on golden-08's household:

1. A concept frame (person and household tables) with exactly the columns the
   packaged nodes nz.units.attributes, nz.encode.person and nz.as.inputs read.
   Each cell takes a stated source: the in-repo golden-08 fixture's household
   record, the oracle extract (golden-07's applicant age, and the mortgage
   concepts golden-08's owner payments imply), or a named
   ``CASE_DATA_UNSTATED_*`` constant. A column without a rule stops the
   generator.
2. Units from CREATE's own unit construction (``create_unit_declaration`` and
   ``build_declared_units``) under the packaged benefit unit rule.
3. The family structure columns, the person inputs and the Accommodation
   Supplement family inputs from those three packaged nodes' own kernels
   (UnitAttributesKernel, ConceptsEncodeKernel, ConceptsEncodeGroupsKernel),
   each run on its packaged node with the packaged mapping and closure. The
   kernels apply the closure's defaults themselves.
4. A cross-check: every input value the oracle extract states (golden-08's,
   less the two bridge amounts, and golden-00's and golden-07's on the parent)
   must equal the encoded or defaulted value. Only the one known closure
   conflict may differ; any other difference stops the generator.

The case's ``provenance`` names every cell's source; the tool ignores that key.
``--check`` fails if the committed case differs from the generated one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from microcosm.build.calibration_columns import prepare_calibration_columns
from microcosm.build.transport import compose, entitlement
from microcosm.build.transport.column_kernels import (
    ConceptsEncodeGroupsKernel,
    ConceptsEncodeKernel,
    UnitAttributesKernel,
)
from microcosm.build.transport.registry import transport_entity_schema
from microcosm.build.transport.unit_declarations import (
    CREATE_V1_REF,
    create_unit_declaration,
)
from microcosm.frame.concepts import (
    CONCEPTS,
    MARITAL_STATUS_DOMAIN,
    SEX_DOMAIN,
    TENURE_DOMAIN,
)
from microcosm.frame.unit_construction import BenefitUnitRule, build_declared_units

ROOT = Path(__file__).resolve().parents[1]
BUILD_TESTS = ROOT / "packages/microcosm-build/tests"
CASE = BUILD_TESTS / "golden/nz_golden08_differential_case.json"
EXTRACT = BUILD_TESTS / "golden/nz_golden08_oracle_extract.json"
FIXTURE = (
    BUILD_TESTS / "fixtures/nz/axiom_oracle_golden_08_accommodation_supplement.json"
)
FIXTURE_NAME = (
    "packages/microcosm-build/tests/fixtures/nz/"
    "axiom_oracle_golden_08_accommodation_supplement.json"
)
FORMAT = "microcosm.nz.golden08_differential_case@1"

MASKS = "nz.as.bridge.masks"
CALLS = (
    "nz.as.bridge.sole_parent_gross_rate",
    "nz.scn.S0.bridge.base_rate",
    "nz.as.bridge.cutout",
    "nz.scn.S0.as",
)
#: The packaged CREATE row, and the nodes that produce the calls' table
#: inputs with the kernel class that serves each one's kernel ref.
CREATE = "nz.create"
UNITS = "nz.units.attributes"
PERSON_ENCODE = "nz.encode.person"
GROUP_ENCODE = "nz.as.inputs"
KERNELS = {
    UNITS: UnitAttributesKernel,
    PERSON_ENCODE: ConceptsEncodeKernel,
    GROUP_ENCODE: ConceptsEncodeGroupsKernel,
}

#: Tolerances follow the NZ hub's 2026-11-03 ruling: every weekly amount that
#: no solve produces within 1e-9, a solved one within its solve's declared
#: tolerance plus 5e-10, and the rounded payment exactly.
WEEKLY_TOLERANCE = 1e-9
SOLVE_MARGIN = 5e-10
CUTOUT_TOLERANCE = 1.5e-9
#: (call, extract section, column, tolerance). Request inputs hold the two
#: bridge amounts; response outputs hold the Accommodation Supplement results.
COMPARISONS = (
    (
        "nz.scn.S0.bridge.base_rate",
        "request",
        "accommodation_supplement_base_rate_weekly_amount",
        WEEKLY_TOLERANCE,
    ),
    (
        "nz.as.bridge.cutout",
        "request",
        "accommodation_supplement_non_beneficiary_income_cutout_weekly_amount",
        CUTOUT_TOLERANCE,
    ),
    (
        "nz.scn.S0.as",
        "response",
        "accommodation_supplement_weekly_qualifying_accommodation_costs",
        WEEKLY_TOLERANCE,
    ),
    (
        "nz.scn.S0.as",
        "response",
        "accommodation_supplement_weekly_amount_before_rounding",
        WEEKLY_TOLERANCE,
    ),
    (
        "nz.scn.S0.as",
        "response",
        "accommodation_supplement_rounded_weekly_payment",
        0,
    ),
)
#: The cross-check compares oracle decimals with encoded floats within the
#: weekly tolerance; booleans and integers compare exactly. Each record says
#: whether its match was exact.
CROSS_CHECK_TOLERANCE = WEEKLY_TOLERANCE
#: The one input where the closure's default differs from an oracle value:
#: golden-07 sends False, the closure defaults True. base_rate and cutout
#: override it to False, as the oracle sends it.
EXPECTED_CLOSURE_CONFLICTS = frozenset(
    {("person", "jobseeker_support_benefit_commenced_on_or_after_1998_07_01")}
)

#: The parent's age: golden-07's applicant age, a request on golden-08's
#: scenario stem (the extract checks the stem).
PARENT_AGE_INPUT = ("golden-07", "jobseeker_support_applicant_age")
PARENT_AGE_NOTE = (
    "The in-repo golden-08 fixture states no parent age "
    "(/household/members/0/age is null); the golden-08 concept frame in "
    "test_nz_axiom_input_closure.py uses 30."
)
#: Binding 39 sums the two mortgage concepts into this golden-08 input, which
#: golden-08 sends as 0.
OWNER_PAYMENTS = (
    "golden-08",
    "accommodation_supplement_owner_weekly_required_payments",
)
OWNER_CONCEPTS = [
    "fact:household.mortgage_interest",
    "fact:household.mortgage_principal",
]
#: Rent and the income concepts are annual (concepts.py declares their period
#: "year"); the fixture states weekly amounts.
WEEKS_PER_YEAR = 52

#: Structural labels. The ``first`` aggregation reads the lowest person id, so
#: the parent holds it.
PERSON_IDS = (1, 2, 3, 4)
HOUSEHOLD_ID = 1

#: Stated placeholders: no source states these cells. OPEN-QUESTIONS.md lists
#: each with what reads it.
#: Which main benefit the parent receives. The fixture states that the parent
#: receives one (/household/receives_main_benefit) and golden-08 sends
#: non_beneficiary false; neither names the benefit.
CASE_DATA_UNSTATED_PARENT_MAIN_BENEFIT_FLAG = "receives_sole_parent_support"
#: Every other receipt flag, on the parent and the children.
CASE_DATA_UNSTATED_OTHER_RECEIPT_FLAGS = False
CASE_DATA_UNSTATED_LIQUID_FINANCIAL_ASSETS = 0.0
CASE_DATA_UNSTATED_PROPERTY_TAX = 0.0
CASE_DATA_UNSTATED_USUAL_WEEKLY_HOURS = 0.0
CASE_DATA_UNSTATED_ENROLLED_FULL_TIME = False
CASE_DATA_UNSTATED_HAS_DISABILITY = False
CASE_DATA_UNSTATED_SEX = "female"
CASE_DATA_UNSTATED_LEGAL_MARITAL_STATUS = "never_married"
CASE_DATA_UNSTATED_TAKE_UP_SEED = 0.5
CASE_DATA_UNSTATED_SOURCE_HOUSEHOLD_ID = "golden-08"
CASE_DATA_UNSTATED_DONOR_SUPPORT_STRATUM = "golden-08"
CASE_DATA_UNSTATED_AS_AREA_ALT = "1"
#: Stated placeholders (NZ hub ruling, 2026-11-13): no source weights a
#: one-household case. The NZ graph weights households.
CASE_DATA_UNSTATED_HOUSEHOLD_WEIGHT_KIND = "design"
CASE_DATA_UNSTATED_HOUSEHOLD_WEIGHT = 1.0
UNSTATED = (
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
)

#: The NZ CREATE row's unit_rule selector. ``skeleton_create_declaration``
#: resolves it with ``_param``, which hands a resource selection to
#: ``_resource_param``; a whole-resource json selection is the canonical JSON
#: of ``transport_resource(spec, "benefit_unit_rule")``. While the row keeps
#: this selector, the CREATE's unit rule is this resource.
CREATE_UNIT_RULE_SELECTOR = {
    "resource": "benefit_unit_rule",
    "path": [],
    "encoding": "json",
}
ROLES_NOTE = "one sole parent (member 0) and that parent's children (members 1-3)"
#: Household concept-frame columns that only a placeholder fills.
HOUSEHOLD_PLACEHOLDERS = {
    "property_tax": "CASE_DATA_UNSTATED_PROPERTY_TAX",
    "as_area_alt": "CASE_DATA_UNSTATED_AS_AREA_ALT",
    "source_household_id": "CASE_DATA_UNSTATED_SOURCE_HOUSEHOLD_ID",
    "donor_support_stratum": "CASE_DATA_UNSTATED_DONOR_SUPPORT_STRATUM",
}


def _token(value):
    return str(value).replace("~", "~0").replace("/", "~1")


def _pointer(*tokens):
    return "".join("/" + _token(token) for token in tokens)


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if value is pd.NA:
        return None
    if hasattr(value, "item") and not isinstance(value, str):
        return value.item()
    return value


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _oracle_cell(entry):
    """A table value for one oracle input, refusing any inexact conversion."""
    kind, value = entry["kind"], entry["value"]
    if kind == "bool" and isinstance(value, bool):
        return value
    if kind == "integer" and isinstance(value, int) and not isinstance(value, bool):
        return value
    if kind == "decimal" and isinstance(value, str):
        number = float(value)
        if Decimal(repr(number)) != Decimal(value):
            raise ValueError(f"Oracle decimal {value!r} is not exact in float64.")
        return number
    raise ValueError(f"Unsupported oracle input {entry!r}.")


def _unstated(name, note=None):
    record = {"from": "unstated", "constant": name}
    return record if note is None else {**record, "note": note}


def _fixture(*tokens, note=None):
    record = {"from": "fixture", "file": FIXTURE_NAME, "pointer": _pointer(*tokens)}
    return record if note is None else {**record, "note": note}


class _Oracle:
    """Oracle inputs by request, read from the extract with their pointers."""

    def __init__(self, extract):
        self.extract = extract
        self.requests = {"golden-08": ("golden_08", "request")}
        self.requests.update(
            {
                name: ("household_requests", name)
                for name in extract["household_requests"]
            }
        )

    def request(self, name):
        record = self.extract
        for token in self.requests[name]:
            record = record[token]
        return record

    def entry(self, request, name):
        entry = self.request(request)["inputs"].get(name)
        if entry is None:
            raise ValueError(f"{request} states no {name}.")
        path = self.requests[request]
        return entry, {
            "from": "oracle",
            "request": request,
            "extract_pointer": _pointer(*path, "inputs", name, "value"),
            "oracle_file": entry["source"],
            "oracle_pointer": entry["pointer"],
        }


def _rows(spec, config):
    """The five packaged nodes, built exactly as the entitlement factory builds them."""
    rows, _ = entitlement._rows(spec)
    nodes = {}
    for row, scenario in rows:
        if row.get("id") not in (MASKS, *CALLS):
            continue
        if row["id"] in nodes or row.get("artifact_inputs"):
            raise ValueError(f"Node {row['id']!r} is repeated or reads artifacts.")
        params = {
            name: entitlement._parameter(spec, config, value, scenario)
            for name, value in row.get("params", {}).items()
        }
        node = compose._node(
            spec,
            row,
            config,
            {},
            allowed_kernels=entitlement._KERNELS,
            resolved_params=params,
            profile_selection=None if scenario is None else scenario.get("knobs", {}),
        )
        added = sorted(set(node.params) - set(params))
        expected = ["engine_ref", "period"] if "rules_binding" in row else []
        if added != expected or any(node.params[key] != params[key] for key in params):
            raise ValueError(
                f"Node {row['id']!r} params changed unexpectedly: {added}."
            )
        nodes[row["id"]] = (node, added)
    missing = sorted({MASKS, *CALLS} - set(nodes))
    if missing:
        raise ValueError(f"The packaged entitlement graph lacks {missing}.")
    return nodes


def _skeleton(spec, config):
    """The CREATE row and the three producer nodes, built as compose builds them."""
    rows = [dict(row) for row in compose.transport_graph_document(spec)["nodes"]]
    types = {
        (row["id"], output.name): output.type
        for row in rows
        for output in compose._artifact_outputs(row)
    }
    by_id = {row["id"]: row for row in rows}
    create = by_id[CREATE]
    if (
        create.get("structural") != "create"
        or create["kernel"] != CREATE_V1_REF
        or create.get("params", {}).get("unit_rule") != CREATE_UNIT_RULE_SELECTOR
    ):
        raise ValueError("The CREATE no longer takes the whole benefit_unit_rule.")
    nodes = {}
    for name, kernel in KERNELS.items():
        node = compose._node(spec, by_id[name], config, types)
        if node.kernel != kernel.ref or node.artifact_inputs or node.sources:
            raise ValueError(f"{name} is not a plain {kernel.ref} node.")
        nodes[name] = node
    return create, nodes


def _run(name, node, tables):
    """One packaged node through its own kernel's ``run``.

    The context holds what these kernels read from the executor's context:
    the node, its params, a view of each entity it reads or owns (the id, the
    person memberships and the declared columns) and no artifacts.
    """
    views = {}
    for entity in {item.entity for item in (*node.inputs, *node.outputs)}:
        table = tables[entity]
        keep = [f"{entity}_id"]
        if entity == "person":
            keep += [
                column
                for column in table.columns
                if column.startswith("person_")
                and column.endswith("_id")
                and column != "person_id"
            ]
        for item in node.inputs:
            if item.entity == entity:
                keep += [column for column in item.columns if column not in keep]
        views[entity] = table.loc[:, keep].copy()
    context = SimpleNamespace(
        node=node, params=dict(node.params), tables=views, artifacts={}
    )
    result = KERNELS[name]().run(context)
    out = {}
    for (entity, column), values in result.columns.items():
        out.setdefault(entity, {})[column] = values
    return {entity: pd.DataFrame(columns) for entity, columns in out.items()}


def _frame_columns(nodes):
    """Concept-frame columns per entity: what the producer nodes read."""
    columns = {}
    for node in nodes.values():
        for item in node.inputs:
            listed = columns.setdefault(item.entity, [])
            listed += [column for column in item.columns if column not in listed]
    return columns


def _concepts():
    return {(item.entity, item.name): item for item in CONCEPTS}


def _typed(values, column, concept):
    """A column with the dtype its concept, or the receipt layer, declares."""
    if concept is not None and str(concept.unit) == "person_id":
        return pd.array(values, dtype="Int64")
    if concept is not None and concept.dtype == "int":
        return np.asarray(values, dtype=np.int64)
    if concept is not None and concept.dtype == "float":
        return np.asarray(values, dtype=np.float64)
    if all(isinstance(value, bool) for value in values):
        return np.asarray(values, dtype=bool)
    if column in {"as_area", "as_area_alt", "donor_support_stratum"}:
        return pd.array(values, dtype="string")
    if all(isinstance(value, str) for value in values):
        return np.asarray(values, dtype=object)
    raise ValueError(f"No dtype for concept-frame column {column}.")


def _person_frame(columns, fixture, oracle, receipt_flags):
    """The person concept table and each cell's source, parent first."""
    household = fixture["household"]
    members = household["members"]
    roles = [member["role"] for member in members]
    if roles != ["sole parent", *["child"] * (len(members) - 1)]:
        raise ValueError(f"The fixture household is not one sole parent: {roles}.")
    if len(members) != len(PERSON_IDS):
        raise ValueError("The fixture household changed size.")
    if members[0]["age"] is not None:
        raise ValueError("The fixture now states the parent's age; restate it.")
    if (
        household["weekly_income"] != 0
        or household["receives_main_benefit"] is not True
    ):
        raise ValueError("The fixture household's income or receipt changed.")
    if CASE_DATA_UNSTATED_PARENT_MAIN_BENEFIT_FLAG not in receipt_flags:
        raise ValueError("The parent's main-benefit flag is not a receipt flag.")
    concepts = _concepts()
    n = len(PERSON_IDS)
    table = pd.DataFrame(
        {
            "person_id": np.asarray(PERSON_IDS, dtype=np.int64),
            "person_household_id": np.full(n, HOUSEHOLD_ID, dtype=np.int64),
        }
    )
    sources = {
        "person_id": {"from": "structural"},
        "person_household_id": {"from": "structural"},
    }

    def role(index, fact):
        return _fixture(
            "household", "members", index, "role", note=f"{fact}: {ROLES_NOTE}"
        )

    def constant(name):
        return [globals()[name]] * n, [_unstated(name)] * n

    for column in columns:
        concept = concepts.get(("person", column))
        if column == "age":
            entry, record = oracle.entry(*PARENT_AGE_INPUT)
            values = [_oracle_cell(entry), *(member["age"] for member in members[1:])]
            cells = [{**record, "reads": PARENT_AGE_INPUT[1], "note": PARENT_AGE_NOTE}]
            cells += [
                _fixture("household", "members", index, "age") for index in range(1, n)
            ]
        elif column == "partner_person_id":
            values = [None] * n
            cells = [role(index, "no co-resident partner") for index in range(n)]
        elif column == "parent_1_person_id":
            values = [None, *[PERSON_IDS[0]] * (n - 1)]
            cells = [role(0, "no co-resident parent")]
            cells += [role(index, "a child of member 0") for index in range(1, n)]
        elif column == "parent_2_person_id":
            values = [None] * n
            cells = [
                role(index, "one co-resident parent at most") for index in range(n)
            ]
        elif column in receipt_flags:
            other = _unstated("CASE_DATA_UNSTATED_OTHER_RECEIPT_FLAGS")
            values = [CASE_DATA_UNSTATED_OTHER_RECEIPT_FLAGS] * n
            cells = [other] * n
            if column == CASE_DATA_UNSTATED_PARENT_MAIN_BENEFIT_FLAG:
                values[0] = True
                cells[0] = _unstated(
                    "CASE_DATA_UNSTATED_PARENT_MAIN_BENEFIT_FLAG",
                    note=(
                        "The fixture states the parent receives a main benefit "
                        "(/household/receives_main_benefit); golden-08 sends "
                        "non_beneficiary false. Neither names the benefit."
                    ),
                )
        elif column == "liquid_financial_assets":
            values, cells = constant("CASE_DATA_UNSTATED_LIQUID_FINANCIAL_ASSETS")
        elif (
            concept is not None
            and concept.monetary is not None
            and str(concept.period) == "year"
        ):
            values = [0.0] * n
            cells = [
                _fixture(
                    "household",
                    "weekly_income",
                    note="the household's income is 0, so each income concept is 0",
                )
            ] * n
        elif column == "usual_weekly_hours":
            values, cells = constant("CASE_DATA_UNSTATED_USUAL_WEEKLY_HOURS")
        elif column == "enrolled_full_time":
            values, cells = constant("CASE_DATA_UNSTATED_ENROLLED_FULL_TIME")
        elif column == "has_disability":
            values, cells = constant("CASE_DATA_UNSTATED_HAS_DISABILITY")
        elif column == "sex" and CASE_DATA_UNSTATED_SEX in SEX_DOMAIN:
            values, cells = constant("CASE_DATA_UNSTATED_SEX")
        elif (
            column == "legal_marital_status"
            and CASE_DATA_UNSTATED_LEGAL_MARITAL_STATUS in MARITAL_STATUS_DOMAIN
        ):
            values, cells = constant("CASE_DATA_UNSTATED_LEGAL_MARITAL_STATUS")
        elif column == "take_up_seed":
            values, cells = constant("CASE_DATA_UNSTATED_TAKE_UP_SEED")
        else:
            raise ValueError(
                f"No source rule for concept-frame column person.{column}."
            )
        table[column] = _typed(values, column, concept)
        sources[column] = cells
    return table, sources


def _household_frame(columns, fixture, oracle, mapping):
    """The household concept table (one row) and each cell's source."""
    household = fixture["household"]
    if household["tenure"] not in TENURE_DOMAIN:
        raise ValueError(f"Fixture tenure {household['tenure']!r} is not declared.")
    concepts = _concepts()
    owners = [
        index
        for index, row in enumerate(mapping["bindings"])
        if row["engine_input"] == OWNER_PAYMENTS[1]
    ]
    owner = mapping["bindings"][owners[0]] if len(owners) == 1 else None
    if (
        owner is None
        or owner["transform"]["kind"] != "scaled_sum"
        or owner["concepts"] != OWNER_CONCEPTS
        or any(
            concepts[("household", item.partition(".")[2])].monetary.signed
            for item in OWNER_CONCEPTS
        )
    ):
        raise ValueError("Owner payments are no longer an unsigned two-concept sum.")
    owner_entry, owner_record = oracle.entry(*OWNER_PAYMENTS)
    if _oracle_cell(owner_entry) != 0:
        raise ValueError("golden-08's owner payments are no longer 0.")
    table = pd.DataFrame({"household_id": np.asarray([HOUSEHOLD_ID], dtype=np.int64)})
    sources = {"household_id": {"from": "structural"}}
    for column in columns:
        concept = concepts.get(("household", column))
        if column == "reference_person_id":
            value = PERSON_IDS[0]
            record = _fixture(
                "household", "members", 0, "role", note="the household's one adult"
            )
        elif column == "tenure":
            value, record = household["tenure"], _fixture("household", "tenure")
        elif column == "rent":
            value = float(household["weekly_rent"] * WEEKS_PER_YEAR)
            record = _fixture(
                "household",
                "weekly_rent",
                note=f"weekly; the rent concept is annual, so times {WEEKS_PER_YEAR}",
            )
        elif column in ("mortgage_interest", "mortgage_principal"):
            value = 0.0
            record = {
                **owner_record,
                "from": "oracle_implied",
                "reads": OWNER_PAYMENTS[1],
                "binding": _pointer("bindings", owners[0]),
                "note": (
                    "The binding sums the two unsigned mortgage concepts with no "
                    "tenure test, so golden-08's 0 makes each 0."
                ),
            }
        elif column == "as_area":
            value, record = str(household["area"]), _fixture("household", "area")
        elif column in HOUSEHOLD_PLACEHOLDERS:
            name = HOUSEHOLD_PLACEHOLDERS[column]
            value, record = globals()[name], _unstated(name)
        else:
            raise ValueError(
                f"No source rule for concept-frame column household.{column}."
            )
        table[column] = _typed([value], column, concept)
        sources[column] = record
    return table, sources


def _call(node):
    entities = [item.entity for item in node.inputs]
    if len(set(entities)) != len(entities):
        raise ValueError(f"Node {node.id!r} repeats an input entity.")
    if node.artifact_inputs or node.artifact_outputs:
        raise ValueError(f"Node {node.id!r} has typed artifacts the tool can't run.")
    outputs = []
    for output in node.outputs:
        ownership = getattr(output.ownership, "value", output.ownership)
        if output.rows != "all" or ownership != "produced" or output.rewrite:
            raise ValueError(f"Node {node.id!r} output {output.column!r} isn't plain.")
        outputs.append(
            {"entity": output.entity, "column": output.column, "dtype": output.dtype}
        )
    return {
        "id": node.id,
        "kernel": node.kernel,
        "inputs": {item.entity: list(item.columns) for item in node.inputs},
        "outputs": outputs,
        "params": _jsonable(dict(node.params)),
    }


def _ordered_columns(calls, entity):
    columns = []
    for call in calls:
        for column in call["inputs"].get(entity, ()):
            if column not in columns:
                columns.append(column)
    return columns


def _masks(node, family):
    """Option A: the masks node's columns, by the function its kernel calls."""
    bindings = json.loads(node.params["bindings"])
    required = bindings["required_inputs"]
    if set(required) != {"family"}:
        raise ValueError("The masks node now reads more than the family.")
    table = pd.DataFrame([{name: family[name] for name in required["family"]}])
    prepared = prepare_calibration_columns({"family": table}, bindings)["family"]
    names = [column["name"] for column in bindings["columns"]]
    if {output.column for output in node.outputs} != set(names):
        raise ValueError("The masks node's outputs differ from its bindings.")
    source = {
        "from": "masks",
        "node": MASKS,
        "function": "microcosm.build.calibration_columns.prepare_calibration_columns",
        "param": "bindings",
        "inputs": list(required["family"]),
    }
    return {name: (bool(prepared[name].iloc[0]), source) for name in names}


def _matches(actual, expected):
    """(equal, exact) for a pipeline cell against an oracle value."""
    flag = isinstance(actual, bool | np.bool_)
    if isinstance(expected, bool):
        equal = flag and bool(actual) is expected
        return equal, equal
    if isinstance(expected, int):
        equal = not flag and isinstance(actual, int | np.integer)
        equal = equal and int(actual) == expected
        return equal, equal
    if flag:
        return False, False
    number = float(actual)
    return abs(number - expected) <= CROSS_CHECK_TOLERANCE, number == expected


def _cross_check(oracle, bridge, encoded, ids):
    """Every oracle-stated input against the pipeline's value for it."""
    records, failures = [], []
    for request in oracle.requests:
        section = oracle.request(request)
        entity = section["entity"].lower()
        table, row = encoded[entity], ids[entity]
        for name, entry in section["inputs"].items():
            if request == "golden-08" and name in bridge:
                continue
            _, source = oracle.entry(request, name)
            if name not in table.columns:
                failures.append(f"{request} states {name}; nothing supplies it.")
                continue
            cell = table.at[row, name]
            expected = _oracle_cell(entry)
            equal, exact = _matches(cell, expected)
            known = (entity, name) in EXPECTED_CLOSURE_CONFLICTS
            if not equal and not known:
                failures.append(
                    f"{request} {name}: oracle {expected!r}, pipeline {cell!r}."
                )
            records.append(
                {
                    "request": request,
                    "entity": entity,
                    "column": name,
                    "extract_pointer": source["extract_pointer"],
                    "oracle": entry["value"],
                    "pipeline": _jsonable(cell),
                    "status": "equal" if equal else "known_closure_conflict",
                    "exact": exact,
                }
            )
    if failures:
        raise ValueError(
            "The pipeline disagrees with the oracle:\n" + "\n".join(failures)
        )
    conflicts = {
        (row["entity"], row["column"])
        for row in records
        if row["status"] == "known_closure_conflict"
    }
    if conflicts != EXPECTED_CLOSURE_CONFLICTS:
        raise ValueError(f"Closure/oracle conflicts changed: {sorted(conflicts)}.")
    return records


def _encoded_source(node_id, node, column, engine_entity, closure, mapping):
    """How the closure says a producer's column is fed, with its pointers."""
    paths = node.params.get("rulespec_paths", ())
    entries = [
        (index, entry)
        for index, entry in enumerate(closure["entries"])
        if entry["input"] == column
        and entry["entity"] == engine_entity
        and entry["rulespec_path"] in paths
    ]
    if not entries:
        raise ValueError(f"{node_id} column {column} has no closure entry.")
    classes = {(entry["class"], entry.get("encoded_by")) for _, entry in entries}
    if len(classes) != 1:
        raise ValueError(f"{column} is fed differently by its modules: {classes}.")
    entry_class, encoded_by = classes.pop()
    pointers = [_pointer("entries", index) for index, _ in entries]
    if entry_class == "defaulted":
        values = {json.dumps(entry["value"]) for _, entry in entries}
        if len(values) != 1:
            raise ValueError(f"{column} has conflicting closure defaults.")
        return {
            "from": "closure",
            "node": node_id,
            "pointers": pointers,
            "value": entries[0][1]["value"],
        }
    if entry_class != "encoded":
        raise ValueError(f"{column} is {entry_class}; the case can't tabulate it.")
    record = {
        "from": "encoded",
        "node": node_id,
        "encoded_by": encoded_by,
        "closure_entries": pointers,
    }
    if encoded_by == "state_binding":
        record["reads"] = sorted(
            {name for _, entry in entries for name in entry["state_binding"]["columns"]}
        )
        return record
    modules = {entry["rulespec_path"] for _, entry in entries}
    found = [
        (index, row)
        for index, row in enumerate(mapping["bindings"])
        if row["engine_input"] == column
        and row["engine_entity"] == engine_entity
        and row["module"] in modules
    ]
    if not found:
        raise ValueError(f"{column} is concept-encoded but no binding names it.")
    record["bindings"] = [_pointer("bindings", index) for index, _ in found]
    record["reads"] = sorted(
        {
            concept.removeprefix("fact:")
            for _, row in found
            for concept in row["concepts"]
        }
    )
    return record


def build_case(extract_path=EXTRACT, fixture_path=FIXTURE):
    """The golden-08 case document."""
    spec = compose.load_transport_spec("nz")
    resource = compose.transport_resource
    bindings = resource(spec, "axiom_rules_bindings")
    config = SimpleNamespace(
        engine_refs={row["id"]: row["id"] for row in bindings["bindings"]}
    )
    nodes = _rows(spec, config)
    calls = [_call(nodes[name][0]) for name in CALLS]
    produced = {
        (output["entity"], output["column"])
        for call in calls
        for output in call["outputs"]
    }
    create, producers = _skeleton(spec, config)

    unit_document = resource(spec, "benefit_unit_rule")
    rule = BenefitUnitRule.from_spec(unit_document)
    unit_rule = _jsonable(rule.to_dict())
    if BenefitUnitRule.from_dict(json.loads(json.dumps(unit_rule))) != rule:
        raise ValueError("The unit rule does not survive a JSON round trip.")
    schema, _ = transport_entity_schema(rule)
    if set(schema.entities) != {"person", "household", "family"}:
        raise ValueError(f"Unexpected NZ transport entities {schema.entities}.")

    weight_entities = {
        node.get("params", {}).get("weight_entity")
        for node in resource(spec, "transport_graph")["nodes"]
    } - {None}
    if weight_entities != {"household"}:
        raise ValueError(f"NZ weights other entities: {sorted(weight_entities)}.")

    extract = json.loads(Path(extract_path).read_text(encoding="utf-8"))
    fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    oracle = _Oracle(extract)
    closure = resource(spec, "axiom_input_closure")
    mapping = resource(spec, "concept_mapping")
    receipt_flags = [
        column.partition(".")[2] for column in closure["receipt_flags"]["columns"]
    ]

    # 1. The concept frame: what the three producer nodes read.
    frame_columns = _frame_columns(producers)
    if set(frame_columns) != {"person", "household", "family"}:
        raise ValueError(f"Producer nodes read {sorted(frame_columns)}.")
    person, person_sources = _person_frame(
        frame_columns["person"], fixture, oracle, receipt_flags
    )
    household, household_sources = _household_frame(
        frame_columns["household"], fixture, oracle, mapping
    )

    # 2. Units, as CREATE builds them (population_kernels.py:234-244).
    declaration = create_unit_declaration(create["kernel"], unit_document)
    partitions = build_declared_units(person, household, declaration)
    tables = {"person": person, "household": household}
    for unit in declaration.units:
        membership = partitions[unit.rule.entity]
        tables[unit.rule.entity] = membership.units
        person[unit.rule.membership_column] = membership.person_unit
    family_units = tables["family"]
    missing = set(frame_columns["family"]) - set(family_units.columns)
    if len(family_units) != 1 or missing:
        raise ValueError(f"CREATE's units are not one family with {sorted(missing)}.")
    constructed = {
        "from": "constructed",
        "function": "microcosm.frame.unit_construction.build_declared_units",
    }
    family_sources = {column: constructed for column in family_units.columns}
    person_sources[rule.membership_column] = constructed

    # 3. The three producer nodes, through their own kernels.
    outputs = {name: _run(name, node, tables) for name, node in producers.items()}
    attributes = outputs[UNITS]["family"]
    person_inputs = outputs[PERSON_ENCODE]["person"]
    family_inputs = outputs[GROUP_ENCODE]["family"]
    family_id = _jsonable(family_units["family_id"].iloc[0])

    # 4. Every oracle-stated input against the pipeline.
    cross_check = _cross_check(
        oracle,
        set(extract["golden_08"]["bridge_inputs"]),
        {"person": person_inputs, "family": family_inputs},
        {"person": PERSON_IDS[0], "family": family_id},
    )
    checked = {
        (row["entity"], row["column"]): row["extract_pointer"] for row in cross_check
    }

    def with_check(entity, column, record):
        pointer = checked.get((entity, column))
        return record if pointer is None else {**record, "checked_against": pointer}

    # The case tables: ids, then exactly the columns the calls read.
    names = bindings["entity_names"]
    provenance = {"person": {}, "household": {}, "family": {}}
    id_columns = ["person_id", "person_household_id", rule.membership_column]
    person_rows = person.loc[:, id_columns].copy()
    for column in id_columns:
        provenance["person"][column] = person_sources[column]
    for column in _ordered_columns(calls, "person"):
        if ("person", column) in produced:
            raise ValueError(f"A call produces person.{column}; refuse to tabulate it.")
        found = [
            source
            for source, present in (
                ("frame", column in frame_columns["person"]),
                ("encoded", column in person_inputs.columns),
            )
            if present
        ]
        if len(found) != 1:
            raise ValueError(f"person.{column} has sources {found}.")
        if found == ["frame"]:
            person_rows[column] = person[column].to_numpy()
            provenance["person"][column] = {
                "from": "concept",
                "column": f"person.{column}",
                "cells": person_sources[column],
            }
        else:
            ordered = person_inputs[column].reindex(person["person_id"].to_numpy())
            person_rows[column] = ordered.to_numpy()
            provenance["person"][column] = with_check(
                "person",
                column,
                _encoded_source(
                    PERSON_ENCODE,
                    producers[PERSON_ENCODE],
                    column,
                    names["person"],
                    closure,
                    mapping,
                ),
            )

    household_rows = household.loc[:, ["household_id"]].copy()
    provenance["household"]["household_id"] = household_sources["household_id"]

    family = {"family_id": family_id}
    provenance["family"]["family_id"] = constructed
    unit_reads = [
        f"{item.entity}.{column}"
        for item in producers[UNITS].inputs
        for column in item.columns
    ]
    structure = {column: attributes.at[family_id, column] for column in attributes}
    masks = _masks(nodes[MASKS][0], structure)
    for column in _ordered_columns(calls, "family"):
        if ("family", column) in produced:
            continue
        found = [
            source
            for source, present in (
                ("attributes", column in attributes.columns),
                ("masks", column in masks),
                ("encoded", column in family_inputs.columns),
            )
            if present
        ]
        if len(found) != 1:
            raise ValueError(f"family.{column} has sources {found}.")
        if found == ["attributes"]:
            family[column] = structure[column]
            provenance["family"][column] = {
                "from": "unit_attributes",
                "node": UNITS,
                "reads": unit_reads,
            }
        elif found == ["masks"]:
            family[column], provenance["family"][column] = masks[column]
        else:
            family[column] = family_inputs.at[family_id, column]
            provenance["family"][column] = with_check(
                "family",
                column,
                _encoded_source(
                    GROUP_ENCODE,
                    producers[GROUP_ENCODE],
                    column,
                    producers[GROUP_ENCODE].params["engine_entity"],
                    closure,
                    mapping,
                ),
            )

    cutout = next(call for call in calls if call["id"] == "nz.as.bridge.cutout")
    if not math.isclose(
        cutout["params"]["tolerance"] + SOLVE_MARGIN, CUTOUT_TOLERANCE, abs_tol=1e-24
    ):
        raise ValueError(
            "The cutout solve's tolerance changed; restate its comparison."
        )
    comparisons = []
    for call_id, section, column, tolerance in COMPARISONS:
        if ("family", column) not in {
            (output["entity"], output["column"])
            for call in calls
            if call["id"] == call_id
            for output in call["outputs"]
        }:
            raise ValueError(f"{call_id} does not produce family.{column}.")
        group = "inputs" if section == "request" else "outputs"
        record = extract["golden_08"][section][group][column]
        if record["source"] not in extract["sources"]:
            raise ValueError(f"Extract record for {column} names no source.")
        comparisons.append(
            {
                "call": call_id,
                "entity": "family",
                "column": column,
                "oracle_pointer": _pointer(
                    "golden_08", section, group, column, "value"
                ),
                "tolerance": tolerance,
            }
        )

    return {
        "format": FORMAT,
        "description": (
            "NZ golden-08 differential: the packaged reg 17 base rate and reg 18 "
            "cutout bridges and the S0 Accommodation Supplement node, run on the "
            "golden-08 sole-parent family as the packaged unit construction, "
            "encoders and closure build it, and compared with the axiom-oracles "
            "extract. Generated by tools/generate_nz_golden08_case.py."
        ),
        "unit_rule": unit_rule,
        "rules_bindings": _jsonable(bindings),
        "tables": {
            "person": _jsonable(person_rows.to_dict(orient="records")),
            "household": _jsonable(household_rows.to_dict(orient="records")),
            "family": [_jsonable(family)],
        },
        "weights": {
            "household": {
                "kind": CASE_DATA_UNSTATED_HOUSEHOLD_WEIGHT_KIND,
                "values": [CASE_DATA_UNSTATED_HOUSEHOLD_WEIGHT],
            }
        },
        "calls": calls,
        "comparisons": comparisons,
        "provenance": {
            "unit_rule": (
                "BenefitUnitRule.from_spec(benefit_unit_rule).to_dict(); the "
                f"CREATE row {create['id']} selects that resource whole"
            ),
            "rules_bindings": "axiom_rules_bindings, verbatim",
            "calls": {
                name: {
                    "graph": "entitlement_graph",
                    "params_added_by_node": nodes[name][1],
                }
                for name in CALLS
            },
            "omitted_calls": {
                MASKS: (
                    "calibration.prepare@1 needs its typed preparation output, "
                    "which the tool's nodes do not declare; its columns are "
                    "family table data."
                )
            },
            "pipeline": {
                "units": (
                    "create_unit_declaration + build_declared_units, as "
                    f"{create['id']} ({create['kernel']}) builds them"
                ),
                "kernels": {name: node.kernel for name, node in producers.items()},
                "context": (
                    "each kernel's own run() on its packaged node, with the "
                    "node's params, a view of each entity it reads or owns, "
                    "and no artifacts"
                ),
            },
            "weights": {
                "entity": "household",
                "kind": _unstated("CASE_DATA_UNSTATED_HOUSEHOLD_WEIGHT_KIND"),
                "values": _unstated("CASE_DATA_UNSTATED_HOUSEHOLD_WEIGHT"),
            },
            "comparisons": {
                "weekly": WEEKLY_TOLERANCE,
                "solved": f"solve tolerance + {SOLVE_MARGIN}",
                "rounded": 0,
            },
            "oracle_cross_check": cross_check,
            "oracle_extract_sha256": _sha256(extract_path),
            "fixture_sha256": _sha256(fixture_path),
            "unstated": list(UNSTATED),
            "concept_frame": {
                "person": person_sources,
                "household": household_sources,
                "family": family_sources,
            },
            "tables": provenance,
        },
    }


def render(document):
    return json.dumps(document, indent=2) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path, default=EXTRACT)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--out", type=Path, default=CASE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render(build_case(args.extract, args.fixture))
    if args.check:
        if not args.out.exists() or args.out.read_text(encoding="utf-8") != text:
            raise SystemExit(f"Golden-08 differential case is stale: {args.out}")
        print("Golden-08 differential case matches the generator.")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out}.")


if __name__ == "__main__":
    main()
