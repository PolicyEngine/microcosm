#!/usr/bin/env python3
"""Generate NZ transport declarations without reading donor records.

The graph maps P1 wages with ``transport.quantile_map@2`` on scaled band
edges and names the two remaining money quantile producers as pending. It declares the executable wiring the shared kernels support.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from fractions import Fraction
from functools import cache
from pathlib import Path

from microcosm.build.transport.codecs import (
    DONOR_SOURCE_CODEC,
    FACTS_SOURCE_CODEC,
    RULESPEC_SOURCE_CODEC,
)
from microcosm.build.transport.terminal_kernels import EXPORT_SOURCE_CODEC
from microcosm.frame import EntitySchema
from microcosm.frame.concepts import (
    CONCEPT_FRAME_SCHEMA,
    SOURCE_HOUSEHOLD_ID_COLUMN,
    TransportRule,
    concept,
    concept_schema_sha256,
)
from microcosm.frame.unit_construction import BenefitUnitRule

ROOT = Path(__file__).resolve().parents[1]
NZ = ROOT / "packages/microcosm-build/src/microcosm/build/nz"
ENTITIES = ("person", "household", "family")
ENTITLED_PAYMENT = "accommodation_supplement_entitled_weekly_payment"


def read(name):
    return json.loads((NZ / f"{name}.json").read_text())


def select(resource, encoding="json", *path):
    return {"resource": resource, "path": list(path), "encoding": encoding}


def document(name, resource):
    return {
        name: select(resource),
        f"{name}_sha256": select(resource, "sha256"),
    }


@cache
def unit_schema():
    rule = BenefitUnitRule.from_spec(read("benefit_unit_rule"))
    return rule, EntitySchema(
        group_entities=(*CONCEPT_FRAME_SCHEMA.group_entities, rule.entity)
    )


@cache
def automatic_columns():
    """Read the same auto-carried identity fields as CLI preparation."""
    _, schema = unit_schema()
    return {
        schema.person_entity: {
            schema.person_id_column,
            *(schema.membership_column(group) for group in schema.group_entities),
        },
        **{group: {schema.id_column(group)} for group in schema.group_entities},
    }


def slice_(entity, columns):
    rule, schema = unit_schema()
    names = list(
        dict.fromkeys(
            column for column in columns if column not in automatic_columns()[entity]
        )
    )
    if not names:
        if entity == schema.person_entity:
            names = [concept("fact:person.age").name]
        elif entity == rule.entity:
            names = [rule.head_column]
        else:
            names = [SOURCE_HOUSEHOLD_ID_COLUMN]
    return {"entity": entity, "columns": names}


def owned(entity, column, dtype=None, **kwargs):
    return {
        "entity": entity,
        "column": column,
        **({"dtype": dtype} if dtype else {}),
        **kwargs,
    }


def edge(name, producer, artifact=None):
    return {"name": name, "producer": producer, "artifact": artifact or name}


def node(id, kernel, **kwargs):
    return {"id": id, "kernel": kernel, **kwargs}


def mapping_document():
    return json.loads(
        (
            ROOT
            / "packages/microcosm-frame/src/microcosm/frame/adapters/axiom_concept_mappings/nz.json"
        ).read_text()
    )


def raw_slices():
    """Read maintained concepts and the unit builder's structural roster."""
    columns = {
        "person": [
            "person_id",
            "person_household_id",
            "person_family_id",
            "age",
            "sex",
            "partner_person_id",
            "parent_1_person_id",
            "parent_2_person_id",
            "usual_weekly_hours",
            "take_up_seed",
        ],
        "household": [
            "household_id",
            "reference_person_id",
            "source_household_id",
            "donor_support_stratum",
            "tenure",
            "rent",
            "mortgage_interest",
            "mortgage_principal",
            "property_tax",
        ],
        "family": [
            "family_id",
            "family_household_id",
            "family_head_person_id",
            "family_source_household_id",
        ],
    }
    paths = set(read("axiom_input_closure")["rulespec_paths"])
    for binding in mapping_document()["bindings"]:
        if binding["module"] in paths:
            for id in binding["concepts"]:
                item = concept(id)
                if item.transport is not TransportRule.DROP:
                    columns[item.entity].append(item.name)
    return [slice_(entity, names) for entity, names in columns.items()]


def closure_inputs(binding, entity):
    bindings = {row["id"]: row for row in read("axiom_rules_bindings")["bindings"]}
    path = bindings[binding]["rulespec_path"]
    return [
        entry["input"]
        for entry in read("axiom_input_closure")["entries"]
        if entry["rulespec_path"] == path and entry["entity"] == entity
    ]


def closure_entry_dtype(entry, mapping):
    """Derive the graph dtype from the encoder's declared operation.

    Defaults follow apply_defaults; mapping and state values follow _apply,
    _group_values and _state_values in frame/concept_mapping.py. Refuse an
    unreviewed transform rather than inventing a narrower output type.
    """
    if entry["class"] == "defaulted":
        dtype = {bool: "bool", int: "int64", float: "float64"}.get(type(entry["value"]))
        if dtype is None:
            raise ValueError(f"Unsupported default dtype for {entry['input']!r}.")
        return dtype
    if entry.get("encoded_by") == "state_binding":
        dtype = "bool"
        group_rule = entry["state_binding"]["group_rule"]
    elif entry.get("encoded_by") == "concept_binding":
        matches = [
            binding
            for binding in mapping["bindings"]
            if binding["module"] == entry["rulespec_path"]
            and binding["engine_entity"] == entry["entity"]
            and binding["engine_input"] == entry["input"]
        ]
        if len(matches) != 1:
            raise ValueError(f"Input {entry['input']!r} needs one concept binding.")
        binding = matches[0]
        transform = binding["transform"]["kind"]
        if transform == "identity":
            dtype = {
                "bool": "bool",
                "int": "int64",
                "float": "float64",
                "str": "string",
            }[concept(binding["concepts"][0]).dtype]
        elif transform in {
            "predicate",
            "relationship_role",
            "positive",
            "take_up_threshold",
            "unit_composition",
        }:
            dtype = "bool"
        elif transform == "coresident_child_count":
            dtype = "int64"
        elif transform in {
            "share",
            "sum",
            "scaled_sum",
            "signed_part",
            "product",
            "allocate_to_reference_person",
            "scale",
            "fraction",
        }:
            dtype = "float64"
        else:
            raise ValueError(
                f"Input {entry['input']!r} has unsupported transform {transform!r}."
            )
        group_rule = binding.get("group_rule")
    else:
        raise ValueError(f"Input {entry['input']!r} has no encoder-owned dtype.")
    if group_rule == "any_member":
        return "bool"
    if group_rule == "sum_over_members":
        if dtype in {"bool", "int64"}:
            return "int64"
        if dtype != "float64":
            raise ValueError(f"Input {entry['input']!r} cannot sum {dtype} values.")
    elif group_rule == "allocate_to_reference_unit":
        return "float64"
    elif group_rule not in {None, "reference_member", "household_value"}:
        raise ValueError(
            f"Input {entry['input']!r} has unsupported group rule {group_rule!r}."
        )
    return dtype


def encoder_outputs(paths, entity):
    """Declare every closure-owned input with its exact portable output dtype."""
    mapping = mapping_document()
    dtypes = {}
    for entry in read("axiom_input_closure")["entries"]:
        if (
            entry["rulespec_path"] in paths
            and entry["entity"] == entity
            and entry.get("encoded_by") != "graph_node"
        ):
            name = entry["input"]
            dtype = closure_entry_dtype(entry, mapping)
            if name in dtypes and dtypes[name] != dtype:
                raise ValueError(f"Input {name!r} has conflicting encoder dtypes.")
            dtypes[name] = dtype
    frame_entity = {"Person": "person", "Family": "family"}[entity]
    return [owned(frame_entity, name, dtype) for name, dtype in sorted(dtypes.items())]


WAGE_TABLE = "ird_wage_salary_national_ty2026"
WAGE_REFERENCE = "ird_wage_salary_people_by_band__{}"
P1_FACTOR = "p1_ty2026_to_ty2027_{}"
AGING_PROFILES = ("low", "central", "high")
#: R2C-9: central anchors on the ten $50,000 bands from $500,000 plus the
#: open top (rows 231-241); VPT re-anchors at $300,000 (rows 227-241).
TAIL_ANCHORS = {"central": 500_000, "vpt": 300_000}
TAIL_BRACKET = ["1.00000095367431640625", "64"]
TAIL_ITERATIONS = 64
EARNINGS_CENTS = {
    "columns": [
        {
            "output_column": "employment_income_cents",
            "quantum": "0.01",
            "rounding": "half_even",
            "source_column": "employment_income",
        }
    ],
    "entity": "person",
}


def _sha256_json(document):
    """SHA-256 of canonical JSON (sorted keys, no whitespace)."""
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def scaled_edge(cut_cents, factor):
    """cut_cents × factor / 100 dollars, exact in Fraction, rounded once."""
    return float(Fraction(cut_cents) * factor / 100)


def earnings_bands_document(intervals=None, aging=None):
    """Scale each wage cut by each profile's P1 factor (design 80, section 2).

    Reads the wage interval table and the P1 factors; writes, for each
    ``aging_index`` profile, the bare band list ``transport.quantile_map@2``
    reads, plus the two declared open-top tails.
    """
    intervals = read("reference_intervals") if intervals is None else intervals
    aging = read("reference_aging") if aging is None else aging
    (table,) = [row for row in intervals["tables"] if row["id"] == WAGE_TABLE]
    factors = {entry["receipt"]["factor_id"]: entry for entry in aging["factors"]}
    cuts = [table["first_lower_cut_cents"]] + [
        band["upper_cent"] for band in table["bands"]
    ]
    profiles = {}
    for profile in AGING_PROFILES:
        entry = factors[P1_FACTOR.format(profile)]
        operational = entry["operational_factor"]
        factor = Fraction(operational["numerator"], operational["denominator"])
        edges = [None if cut is None else scaled_edge(cut, factor) for cut in cuts]
        profiles[profile] = {
            "factor_id": P1_FACTOR.format(profile),
            "operational_factor": operational,
            "factor_receipt_sha256": entry["receipt"]["receipt_sha256"],
            "map_bands": {
                "bands": [
                    {
                        "lower": edges[index],
                        "upper": edges[index + 1],
                        "reference": WAGE_REFERENCE.format(band["id"]),
                    }
                    for index, band in enumerate(table["bands"])
                ]
            },
        }
    tails = {}
    for name, anchor in TAIL_ANCHORS.items():
        start = next(
            index
            for index, cut in enumerate(cuts)
            if cut is not None and cut == anchor * 100
        )
        tails[name] = {
            "method": "pareto",
            "estimator": {
                "id": "grouped_pareto_mle@1",
                "anchor_references": [
                    WAGE_REFERENCE.format(band["id"]) for band in table["bands"][start:]
                ],
                "bracket": list(TAIL_BRACKET),
                "iterations": TAIL_ITERATIONS,
            },
        }
    return {
        "schema_version": 1,
        "country": "nz",
        "generator": "tools/build_nz_transport_graph.py",
        "description": (
            "Effective TY2027 wage band edges in dollars for transport.quantile_map@2. "
            "Each edge is the wage table's TY2026 cut in cents times the selected "
            "profile's P1 factor over 100, computed exactly and rounded once to "
            "binary64. Band counts stay the TY2026 IRD facts the references pin: "
            "scaling the edges, not the mapped values, keeps one producer of "
            "person.employment_income (design 80, section 2). The generator's "
            "--check recomputes every edge from reference_intervals.json and "
            "reference_aging.json."
        ),
        "interval_table_id": WAGE_TABLE,
        "interval_table_sha256": _sha256_json(table),
        "profiles": profiles,
        "tails": tails,
        "tail_disclosure": (
            "R2C-9: the central tail fits grouped_pareto_mle@1 to the ten $50,000 "
            "bands from $500,000 plus the open top; VPT re-anchors at $300,000 over "
            "all fourteen $50,000 bands plus the top. The two-point closed form is a "
            "G8-x diagnostic and is not declared here."
        ),
    }


def validate_earnings_bands(document, intervals=None, aging=None):
    """Refuse any edge, factor or table identity the sources don't reproduce.

    Recomputes every profile's edges as cut_cents × P1 factor / 100 from the
    wage interval table and reference_aging.json, and compares the factor
    receipts and the table digest. Returns nothing; raises on the first
    difference, naming it.
    """
    expected = earnings_bands_document(intervals, aging)
    if document.get("interval_table_sha256") != expected["interval_table_sha256"]:
        raise ValueError("earnings_bands.json names a stale wage interval table")
    for profile, want in expected["profiles"].items():
        have = document.get("profiles", {}).get(profile)
        if have is None:
            raise ValueError(f"earnings_bands.json lacks profile {profile!r}")
        for field in ("factor_id", "operational_factor", "factor_receipt_sha256"):
            if have.get(field) != want[field]:
                raise ValueError(
                    f"earnings_bands.json profile {profile!r} has a stale {field}"
                )
        if have.get("map_bands") != want["map_bands"]:
            raise ValueError(
                f"earnings_bands.json profile {profile!r} edges differ from "
                "cut × factor"
            )
    if document.get("tails") != expected["tails"]:
        raise ValueError("earnings_bands.json tails differ from the declared anchors")


def earnings_nodes(population, prefix, profile="central", tail="central"):
    """The P1 wage map and its cents node, upstream of the person encoder."""
    return [
        node(
            f"{prefix}.earnings.map",
            "transport.quantile_map@2",
            population=population,
            inputs=[slice_("person", ["employment_income"])],
            outputs=[owned("person", "employment_income", "float64", rewrite=True)],
            params={
                "entity": "person",
                "column": "employment_income",
                "bands": select(
                    "earnings_bands", "json", "profiles", profile, "map_bands"
                ),
                "bands_sha256": select("earnings_bands", "sha256"),
                "interpolation": select("earnings_bands", "json", "tails", tail),
            },
            artifact_inputs=[edge("surface", "nz.facts.precal")],
        ),
        node(
            f"{prefix}.earnings.cents",
            "money.minor_units@1",
            population=population,
            inputs=[slice_("person", ["employment_income"])],
            outputs=[owned("person", "employment_income_cents", "int64")],
            params={
                "declaration": json.dumps(
                    EARNINGS_CENTS, sort_keys=True, separators=(",", ":")
                )
            },
        ),
    ]


MEASUREMENT_ID = "ird_taxable_income_ty2027_v1"
MEASURE_OUTPUTS = {
    "measure": "ird_taxable_income_proxy_cents",
    "unit": "ird_taxable_income_proxy_nzd",
    "universe_status": "ird_taxable_income_proxy_universe_status",
}
#: Design 80, section 3: the stand-in's components, all int64 cents columns.
MARKET_COMPONENTS = (
    "employment_income_cents",
    "nonfarm_self_employment_income_cents",
    "farm_self_employment_income_cents",
    "interest_income_cents",
    "dividend_income_cents",
    "rental_income_cents",
)
TRANSFER_COMPONENTS = (
    "jobseeker_support_annual_gross_transfer_cents",
    "sole_parent_support_annual_gross_transfer_cents",
    "supported_living_payment_annual_gross_transfer_cents",
    "nz_super_annual_gross_transfer_cents",
)
#: R2C-7: VEX's declared annual-transfer exposure, as exact decimal text.
VEX_EXPOSURE = "0.75"


def measure_declaration(components):
    """C's measure.combine@1 declaration: R2C-10's floor 0 and scale 100."""
    return {
        "entity": "person",
        "components": [{"column": column, "sign": 1} for column in components],
        "add_backs": ["rental_income_cents"],
        "floor": 0,
        "scale": 100,
        "outputs": dict(MEASURE_OUTPUTS),
    }


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def income_measurements_document():
    """The stand-in measure, its VSI sibling and the published omission register."""
    measures = {}
    for name, components, description in (
        (
            "central",
            (*MARKET_COMPONENTS, *TRANSFER_COMPONENTS),
            "Design 80 section 3: P1-mapped wages, signed self-employment, "
            "interest, dividends, rental profit with losses added back "
            "(max(0, r)), and the annual gross JS, SPS, SLP and NZS transfers.",
        ),
        (
            "vsi",
            MARKET_COMPONENTS,
            "VSI stress test: the central measure without the four transfer "
            "addends, with the same flags, payments and entitlement rules.",
        ),
    ):
        declaration = measure_declaration(components)
        measures[name] = {
            "description": description,
            "declaration": declaration,
            "declaration_sha256": hashlib.sha256(
                _canonical(declaration).encode()
            ).hexdigest(),
        }
    unknown = "unknown"
    return {
        "schema_version": 1,
        "country": "nz",
        "generator": "tools/build_nz_transport_graph.py",
        "measurement_id": MEASUREMENT_ID,
        "description": (
            "The d1952 stand-in income measure: a person-level, annual, TY2027 "
            "NZD statistical measure computed by measure.combine@1 in integer "
            "cents. It is not IRD taxable income. Universe: included at x > 0, "
            "excluded at x <= 0, unresolved on any null component (R2C-14). The "
            "unit column is max(0, x)/100 (R2C-10)."
        ),
        "measures": measures,
        "omission_register": {
            "description": (
                "Items IRD taxable income contains that the stand-in omits. A size "
                "appears only where a repo resource or a cited public source gives "
                "one; design 80's percentages are the proposal's claims and carry "
                "no source here, so every size reads unknown. Unknown is not zero."
            ),
            "items": [
                {"item": item, "size": unknown, "reason": reason}
                for item, reason in (
                    (
                        "private_pension_income",
                        "The taxable share of private pensions is unknown.",
                    ),
                    ("realized_capital_gains", "NZ taxes few realised gains."),
                    (
                        "veterans_pension",
                        "Receipt is unsupported under H4 (receipt_contract.json); "
                        "the VP rows stay parked.",
                    ),
                    ("youth_and_young_parent_payment", "No program binding."),
                    ("acc_weekly_compensation", "No ACC producer."),
                    ("student_allowances", "No student-allowance producer."),
                    ("paid_parental_leave", "No PPL producer."),
                    ("trust_income", "No trust producer; may overlap other items."),
                    ("overseas_income", "No overseas-income producer."),
                    ("look_through_company_income", "No LTC producer."),
                    (
                        "partnership_and_maori_authority_allocations",
                        "No allocation producer.",
                    ),
                    ("taxable_property_disposals", "No disposal producer."),
                    (
                        "deductions_and_losses_beyond_those_netted",
                        "Only self-employment losses and the rental add-back are "
                        "netted.",
                    ),
                )
            ],
        },
        "review_triggers": {
            "description": (
                "R2C-3: VSI escalation triggers are review triggers, not gates. When "
                "one fires the hub writes a ruling before any release claim. No gate "
                "kind computes them yet, so G8-x reports them."
            ),
            "scenario": "VSI",
            "triggers": [
                {
                    "measure": "accommodation_supplement_recipients",
                    "relative_change_above": "0.10",
                },
                {
                    "measure": "accommodation_supplement_expenditure",
                    "relative_change_above": "0.10",
                },
                {
                    "measure": "central_weighted_positive_population_crossing_floor",
                    "share_above": "0.05",
                },
            ],
        },
        "diagnostics": {
            "wage_earner_count": {
                "ruling": "R2C-8",
                "status": "diagnostic only; G8-x reports it",
                "comparison": (
                    "the sum of the 235 IRD P1 wage counts against the model's "
                    "weighted persons with positive employment_income, before and "
                    "after the fit"
                ),
            }
        },
    }


def standin_node(population, prefix, measure="central"):
    """nz.income.standin: C's measure.combine@1, the measurement's only producer."""
    components = [
        row["column"]
        for row in income_measurements_document()["measures"][measure]["declaration"][
            "components"
        ]
    ]
    return node(
        f"{prefix}.income.standin",
        "measure.combine@1",
        population=population,
        inputs=[slice_("person", components)],
        outputs=[
            owned("person", MEASURE_OUTPUTS["measure"], "Int64"),
            owned("person", MEASURE_OUTPUTS["unit"], "float64"),
            owned("person", MEASURE_OUTPUTS["universe_status"], "string"),
        ],
        params={
            "measure": select(
                "income_measurements", "json", "measures", measure, "declaration"
            ),
            "measure_sha256": select(
                "income_measurements",
                "value",
                "measures",
                measure,
                "declaration_sha256",
            ),
            "measurement_id": MEASUREMENT_ID,
        },
    )


def calibration_nodes(
    population="nz.central",
    prefix="nz",
    money_resource="money_normalization",
    earnings_profile="central",
    earnings_tail="central",
    measure="central",
    exposure=None,
):
    """Build one complete money/engine/receipt/calibration branch."""
    resources = read("money_normalization")
    normalized = resources["columns"]
    central = read("scenarios")["central_knobs"]
    bindings = {row["id"]: row for row in read("axiom_rules_bindings")["bindings"]}
    paths = [
        bindings[name]["rulespec_path"]
        for name in ("main_benefits_entitlement", "main_benefits_rates", "nz_super")
    ]
    rows = [
        node(
            f"{prefix}.money.normalize",
            "money.normalize@1",
            population=population,
            sources=["nz_chronicle_facts"],
            inputs=[slice_(entity, names) for entity, names in normalized.items()],
            outputs=[
                owned(entity, column, "float64", rewrite=True)
                for entity, names in normalized.items()
                for column in names
            ]
            + [
                owned(entity, f"{column}{resources['cents_suffix']}", "int64")
                for entity, names in normalized.items()
                for column in names
            ],
            params={
                **document("declaration", money_resource),
                **document("factor_context", "reference_aging"),
                "profile_selection": json.dumps(
                    central, sort_keys=True, separators=(",", ":")
                ),
            },
        ),
        *earnings_nodes(population, prefix, earnings_profile, earnings_tail),
        node(
            f"{prefix}.encode.person",
            "concepts.encode@1",
            population=population,
            inputs=raw_slices()[:2],
            outputs=encoder_outputs(paths, "Person"),
            params={
                **document("mapping", "concept_mapping"),
                **document("closure", "axiom_input_closure"),
                "rulespec_paths": paths,
            },
        ),
    ]
    for short, binding in (
        ("entitlement", "main_benefits_entitlement"),
        ("super", "nz_super"),
    ):
        variables = bindings[binding]["variables"]
        rows.append(
            node(
                f"{prefix}.benefits.{short}",
                "simulate.rules_by_ref@1",
                population=population,
                sources=["rulespec_nz"],
                rules_binding=binding,
                inputs=[slice_("person", closure_inputs(binding, "Person"))],
                outputs=[
                    owned(
                        "person",
                        column,
                        "int64" if column.startswith("entitled_") else "float64",
                    )
                    for column in variables
                ],
                params={"variables": variables},
            )
        )
    gross = copy.deepcopy(read("benefit_gross_up")["nodes"])
    for row in gross:
        row["id"] = row["id"].replace("nz.", f"{prefix}.", 1)
        row["population"] = population
    rows.extend(gross[:-1])
    receipt = read("receipt_contract")["receipts"]
    receipt_inputs = [receipt["seed_column"], "age"]
    for program in receipt["programs"]:
        receipt_inputs.extend(program["judgment_columns"])
        receipt_inputs.append(program["payment_column"])
    rows.append(
        node(
            f"{prefix}.takeup.assign",
            "takeup.assign@1",
            population=population,
            inputs=[
                slice_(
                    entity, [*names, *(receipt_inputs if entity == "person" else [])]
                )
                for entity, names in read("calibration_bindings")[
                    "required_inputs"
                ].items()
            ],
            outputs=[
                owned("person", program["output"], "bool")
                for program in receipt["programs"]
            ],
            params={
                "receipt_contract": select("receipt_contract", "json", "receipts"),
                "receipt_contract_sha256": select("receipt_contract", "sha256"),
            },
            artifact_inputs=[edge("surface", "nz.facts.precal")],
        )
    )
    annual = gross[-1]
    if exposure is not None:
        declaration = json.loads(annual["params"]["declaration"])
        declaration["exposure"]["central"] = exposure
        annual["params"]["declaration"] = _canonical(declaration)
    rows.append(annual)
    rows.append(standin_node(population, prefix, measure))
    prep = read("calibration_bindings")
    required = {
        entity: list(names) for entity, names in prep["required_inputs"].items()
    }
    for program in receipt["programs"]:
        required["person"].append(program["output"])
    outputs = [
        owned(
            entry["entity"],
            entry["name"],
            "int64"
            if "constant" in entry or entry in prep.get("roster_counts", ())
            else "bool",
        )
        for entry in [*prep["columns"], *prep.get("roster_counts", ())]
    ]
    # Interval outputs read their table's income and universe columns.
    interval_inputs = [
        MEASURE_OUTPUTS["measure"],
        MEASURE_OUTPUTS["universe_status"],
    ]
    if any("interval" in entry for entry in prep["columns"]):
        required["person"].extend(interval_inputs)
    rows.append(
        node(
            f"{prefix}.targets.prepare",
            "calibration.prepare@1",
            population=population,
            inputs=[slice_(entity, names) for entity, names in required.items()],
            outputs=outputs,
            params={
                "bindings": select("calibration_bindings"),
                **document("references", "target_references"),
            },
        )
    )
    rows.append(
        node(
            f"{prefix}.targets.compile",
            "targets.compile@1",
            population=population,
            sources=["nz_chronicle_facts"],
            params={"country": "nz", **document("references", "target_references")},
        )
    )
    problem_inputs = {entity: [] for entity in ENTITIES}
    for output in outputs:
        problem_inputs[output["entity"]].append(output["column"])
    problem_inputs["person"].extend(
        program["output"] for program in receipt["programs"]
    )
    if any("interval" in entry for entry in prep["columns"]):
        # The interval diagnostics read the table's columns on the problem
        # frame, and per-band support groups by source household (hunk D).
        problem_inputs["person"].extend(interval_inputs)
        problem_inputs["household"].append(SOURCE_HOUSEHOLD_ID_COLUMN)
    rows.append(
        node(
            f"{prefix}.targets.problem",
            "targets.problem@1",
            population=population,
            inputs=[slice_(entity, names) for entity, names in problem_inputs.items()],
            params={"entities": list(ENTITIES), "weight_entity": "household"},
            artifact_inputs=[
                edge("surface", f"{prefix}.targets.compile"),
                edge("preparation", f"{prefix}.targets.prepare"),
            ],
        )
    )
    rows.append(
        node(
            f"{prefix}.calibrate",
            "calibrate.ordered_adam@1",
            structural="reweight",
            base=population,
            mass="free",
            inputs=[slice_("household", ["household_id"])],
            weights={"entity": "household", "to_kind": "calibrated", "mass": "free"},
            params={
                name: select("calibration_settings", "value", name)
                for name in (
                    "epochs",
                    "learning_rate",
                    "mass",
                    "max_weight_ratio",
                    "weight_anchor",
                )
            },
            artifact_inputs=[edge("problem", f"{prefix}.targets.problem")],
        )
    )
    return rows


def as_encoder(population="nz.as", prefix="nz.as"):
    bindings = {row["id"]: row for row in read("axiom_rules_bindings")["bindings"]}
    path = bindings["accommodation_supplement"]["rulespec_path"]
    outputs = encoder_outputs([path], "Family")
    inputs = raw_slices()
    inputs[0]["columns"].extend(
        program["output"]
        for program in read("receipt_contract")["receipts"]["programs"]
    )
    inputs[1]["columns"].extend(["as_area", "as_area_alt"])
    return node(
        f"{prefix}.inputs",
        "concepts.encode_groups@1",
        population=population,
        inputs=inputs,
        outputs=outputs,
        params={
            **document("mapping", "concept_mapping"),
            **document("closure", "axiom_input_closure"),
            **document("unit_rule", "benefit_unit_rule"),
            "engine_entity": "Family",
            "rulespec_paths": [path],
            "knobs": select("as_encoding"),
            "scenario_sha256": select("as_encoding", "sha256"),
        },
    )


def documents():
    mapping = mapping_document()
    normal = read("money_normalization")
    hold = copy.deepcopy(normal)
    hold["description"] = (
        "The hold-fixed sensitivity converts the same donor columns to NZD and holds all four index families fixed at factor one. It uses the central reference-aging profile."
    )
    hold["status"] = (
        "The VHF scenario selects this sibling declaration in its own normalization node."
    )
    hold["index_family_factors"] = {
        name: {
            "identity": True,
            "reason": "The declared hold-fixed sensitivity holds this donor index family at factor one.",
        }
        for name in normal["index_family_factors"]
    }
    support_definition = copy.deepcopy(read("atomic_geography_support"))
    modal = {
        row["ta_code"]: str(row["modal_area"])
        for row in read("as_area_crosswalk")["rows"]
    }
    support_definition["columns"]["as_area_alt"] = {
        "kind": "code",
        "source": "as_area_crosswalk.json modal_area",
        "vintage": "ta_2025",
        "relation": "best_fit",
    }
    for row in support_definition["rows"]:
        row["codes"]["as_area_alt"] = modal[row["codes"]["ta_code"]]
    assignment = {
        "version": 1,
        "identity": ["household_id"],
        "stream": ["sha256-u53-v1", "nz-joint-geography", 0, 21],
        "outputs": {
            "area": "nz_atomic_area",
            "system": "nz_area_system",
            "basis": "nz_area_basis",
        },
        "systems": [
            {
                "id": "nz-joint",
                "level": "ta_region_as_area",
                "code_system": "stats_nz_joint_ta_regc_as",
                "vintage": "ta2026_regc2025_au2017",
                "source": "nz_chronicle_facts",
                "selector": {},
                "constraints": [],
                "observed_area": None,
                "stages": [{"level": "area", "weight": "population"}],
                "layers": [
                    {
                        "input": input_,
                        "output": output,
                        "vintage": "ta2026_regc2025_au2017",
                        "relation": "exact",
                        "source": "atomic_geography_support.json",
                    }
                    for input_, output in (
                        ("ta_code", "ta_code"),
                        ("region_code", "household_region_code"),
                        ("as_area", "as_area"),
                        ("as_area_alt", "as_area_alt"),
                    )
                ],
            }
        ],
    }
    rule = document("unit_rule", "benefit_unit_rule")
    rows = [
        node(
            "nz.create",
            "transport.create@1",
            structural="create",
            sources=["populace_us_donor", "nz_chronicle_facts"],
            params={
                "donor_source": "populace_us_donor",
                "facts_source": "nz_chronicle_facts",
                "donor_sha256": select("transport_inputs", "value", "donor", "sha256"),
                "donor_size": select(
                    "transport_inputs", "value", "donor", "size_bytes"
                ),
                "donor_pin_sha256": select("source_stages", "sha256"),
                **rule,
                **document("mass_reference", "mass_references"),
                "country": "nz",
                "concept_schema_sha256": concept_schema_sha256(),
                "seed_stream": "nz:takeup:v0",
            },
        ),
        node(
            "nz.open",
            "transport.boundary@1",
            structural="filter",
            base="nz.create",
            inputs=[slice_("person", ["age"])],
        ),
        node(
            "nz.facts.precal",
            "targets.compile@1",
            population="nz.open",
            sources=["nz_chronicle_facts"],
            params={"country": "nz", **document("references", "precal_references")},
        ),
        node(
            "nz.geo.support",
            "geography.support_from_facts@1",
            population="nz.open",
            params={
                "definition": select("transport_atomic_geography_support"),
                "definition_sha256": select(
                    "transport_atomic_geography_support", "sha256"
                ),
                "system": "nz-joint",
            },
            artifact_inputs=[edge("surface", "nz.facts.precal")],
        ),
    ]
    geo_params = {
        "definition": select("geography_assignment"),
        "stream": select("geography_assignment", "value", "stream"),
    }
    support = [edge("nz-joint", "nz.geo.support", "support")]
    assigned = list(assignment["outputs"].values())
    derived = [layer["output"] for layer in assignment["systems"][0]["layers"]]
    rows.extend(
        [
            node(
                "nz.geo.assign",
                "geography.assign_atomic@1",
                population="nz.open",
                inputs=[slice_("household", ["household_id"])],
                outputs=[owned("household", column, "string") for column in assigned],
                params=geo_params,
                artifact_inputs=support,
            ),
            node(
                "nz.geo.derive",
                "geography.derive@1",
                population="nz.open",
                inputs=[slice_("household", assigned)],
                outputs=[owned("household", column, "string") for column in derived],
                params=geo_params,
                artifact_inputs=support,
            ),
            node(
                "nz.geo.person",
                "geography.person_location@1",
                population="nz.open",
                inputs=[
                    slice_("household", ["household_region_code"]),
                    slice_("person", ["person_id", "person_household_id"]),
                ],
                outputs=[owned("person", "region_code", "string")],
                params={
                    "columns": json.dumps(
                        {"household_region_code": "region_code"},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                },
            ),
            node(
                "nz.geo.gate",
                "geography.gate@1",
                population="nz.open",
                inputs=[slice_("household", [*assigned, *derived])],
                params=geo_params,
                artifact_inputs=support,
            ),
            node(
                "nz.units.attributes",
                "transport.unit_attributes@1",
                population="nz.open",
                inputs=[
                    slice_(
                        "person",
                        [
                            "person_id",
                            "person_household_id",
                            "person_family_id",
                            "age",
                            "partner_person_id",
                            "parent_1_person_id",
                            "parent_2_person_id",
                            "usual_weekly_hours",
                        ],
                    ),
                    slice_("household", ["household_id", "reference_person_id"]),
                    slice_(
                        "family",
                        ["family_id", "family_household_id", "family_head_person_id"],
                    ),
                ],
                outputs=[
                    owned("family", "family_type", "string"),
                    owned("family", "n_dependent_children", "int64"),
                    owned("family", "is_sole_parent", "bool"),
                    owned("family", "is_partnered", "bool"),
                    owned("family", "is_single", "bool"),
                    owned("family", "n_adults", "int64"),
                    owned("family", "youngest_dependent_child_age", "Int64"),
                    owned("household", "n_family_units", "int64"),
                ],
                params=rule,
            ),
            node(
                "nz.central",
                "transport.boundary@1",
                structural="filter",
                base="nz.open",
                inputs=[slice_("person", ["age"])],
            ),
        ]
    )
    rows.extend(calibration_nodes())
    rows.extend(
        [
            node(
                "nz.as",
                "transport.boundary@1",
                structural="filter",
                base="nz.calibrate",
                inputs=[slice_("person", ["age"])],
            ),
            as_encoder(),
            node(
                "nz.terminal",
                "transport.boundary@1",
                structural="filter",
                base="nz.as",
                inputs=[slice_("person", ["age"])],
            ),
        ]
    )
    export = read("export_contract")
    roster = {entity: [] for entity in ENTITIES}
    for entity, names in export["_schema"]["structural_columns"].items():
        roster[entity].extend(names)
    roster["person"].extend(["age", "sex"])
    roster["household"].extend(
        [
            "source_household_id",
            "donor_support_stratum",
            "household_region_code",
            "tenure",
            "rent",
            "mortgage_interest",
            "mortgage_principal",
            "property_tax",
        ]
    )
    roster["family"].append("family_source_household_id")
    for row in rows:
        if row["kernel"] in {
            "money.normalize@1",
            "transport.quantile_map@2",
            "concepts.encode@1",
            "concepts.encode_groups@1",
            "takeup.assign@1",
        }:
            # R2C-6: the wage map's rewrite joins the roster beside the
            # normalized donor money, so the support gate range-checks it.
            for output in row.get("outputs", ()):
                if row["kernel"] not in {
                    "money.normalize@1",
                    "transport.quantile_map@2",
                } or output.get("rewrite"):
                    roster[output["entity"]].append(output["column"])
    roster = {
        entity: sorted(set(names) - automatic_columns()[entity])
        for entity, names in roster.items()
    }
    export["optional"] = sorted(
        set(export["optional"])
        | {name for names in roster.values() for name in names}
        - set(export["required"])
    )
    # The entitlement graph's gated payment is a formula output like the
    # engine's own; the closed export never carries it.
    export["formula_owned_excluded"] = sorted(
        set(export["formula_owned_excluded"]) | {ENTITLED_PAYMENT}
    )
    export["entity_slices"] = {"export": roster, "gates": roster}
    export["_description"] = (
        "The graph enforces this closed entity-qualified roster at composition. The export and terminal gate slices retain replay inputs and assigned receipt flags and exclude formula outputs, scratch cents and reporting state. Frame projection supplies structural identifiers and memberships automatically."
    )
    export["_slice_requirement"] = (
        "The graph selects export_contract for opt-in composition validation and declares its exact export and terminal gate slices."
    )
    population = [slice_(entity, names) for entity, names in roster.items()]
    rows.extend(
        [
            node(
                "nz.calibration.diagnostics",
                "diagnostics.calibration@1",
                population="nz.terminal",
                inputs=[slice_("household", ["household_id"])],
                params={"weight_entity": "household"},
                artifact_inputs=[
                    edge("problem", "nz.targets.problem"),
                    edge("solution", "nz.calibrate"),
                    edge("result", "nz.calibrate"),
                    edge("surface", "nz.targets.compile"),
                ],
            ),
            node(
                "nz.gates.terminal",
                "gates.battery@1",
                population="nz.terminal",
                inputs=population,
                params={
                    "country": "nz",
                    **document("gates", "gates"),
                    "phase": "terminal",
                    "release_candidate": False,
                    "entities": list(ENTITIES),
                    "weight_entity": "household",
                },
                artifact_inputs=[
                    edge("problem", "nz.targets.problem"),
                    edge("solution", "nz.calibrate"),
                    edge("diagnostics", "nz.calibration.diagnostics"),
                    edge("surface", "nz.targets.compile"),
                ],
            ),
            node(
                "nz.export.prepare",
                "export.prepare@1",
                population="nz.terminal",
                inputs=population,
                params={
                    "entities": list(ENTITIES),
                    "weight_entity": "household",
                    "time_period": select(
                        "export_contract", "value", "_period", "dataset_time_period"
                    ),
                    "bindings": select("export_replay_bindings"),
                },
                artifact_inputs=[edge("gate_report", "nz.gates.terminal")],
            ),
            node(
                "nz.export.readback",
                "export.readback@1",
                population="nz.terminal",
                sources=["nz_exported_dataset"],
                artifact_inputs=[edge("export_descriptor", "nz.export.prepare")],
            ),
            node(
                "nz.package",
                "transport.package@1",
                population="nz.terminal",
                params={"bindings": {"prepared": "engine_refs"}},
                artifact_inputs=[
                    edge("export_readback", "nz.export.readback"),
                    edge("gate_report", "nz.gates.terminal"),
                    edge("diagnostics", "nz.calibration.diagnostics"),
                    edge("surface", "nz.targets.compile"),
                ],
            ),
        ]
    )
    graph = {
        "schema_version": 1,
        "country": "nz",
        "description": "The NZ transport declaration retains named pending money producers. The graph makes no estimate or publication claim.",
        "export_contract": "export_contract",
        "sources": [
            {"name": name, "codec": codec}
            for name, codec in (
                ("populace_us_donor", DONOR_SOURCE_CODEC),
                ("nz_chronicle_facts", FACTS_SOURCE_CODEC),
                ("nz_holdout_facts", FACTS_SOURCE_CODEC),
                ("rulespec_nz", RULESPEC_SOURCE_CODEC),
                ("nz_exported_dataset", EXPORT_SOURCE_CODEC),
            )
        ],
        "nodes": rows,
        "endpoints": {
            "geography": "nz.geo.gate",
            "calibration": "nz.calibrate",
            "terminal": ["nz.export.prepare", "nz.package"],
        },
        "pending_outputs": [
            "G8-x income reconciliation evidence: the native-band report, omission-register and identity-receipt artifacts and the VSI, VEX and VPT run evidence that measurement_prefit requires (R2C-18), and the NZS sharing binding (R2C-5)",
            "Census rent and liquid-assets quantile producers",
        ],
    }
    add_evidence(graph)
    return {
        "export_replay_bindings.json": {
            "export_contract_sha256": hashlib.sha256(
                (json.dumps(export, indent=2, ensure_ascii=False) + "\n").encode()
            ).hexdigest(),
            "axiom_input_closure_sha256": hashlib.sha256(
                (NZ / "axiom_input_closure.json").read_bytes()
            ).hexdigest(),
        },
        "as_encoding.json": {
            "rent_allocation": "reference_unit",
            "asset_test": True,
            "area_column": "as_area",
            "rent_factor": 1,
        },
        "transport_inputs.json": {
            "donor": read("source_stages")["stages"][0]["artifacts"][0]
        },
        "transport_atomic_geography_support.json": support_definition,
        "concept_mapping.json": mapping,
        "geography_assignment.json": assignment,
        "money_normalization_hold_fixed.json": hold,
        "export_contract.json": export,
        "transport_graph.json": graph,
        "earnings_bands.json": earnings_bands_document(),
        "income_measurements.json": income_measurements_document(),
    }


def closure_slices():
    """The frame columns the closure producer reads.

    The closure's concept, state and default entries feed their entities'
    columns, and every declared relation input's physical column and role
    column feed the person slice, where the producer reads them (NZ
    declares no relation today).
    """
    from microcosm.build.transport.evidence_contract import binding_relation_inputs

    bindings = read("axiom_rules_bindings")
    names = bindings["entity_names"]
    to_frame = {engine: frame for frame, engine in names.items()}
    columns = {}
    for entry in read("axiom_input_closure")["entries"]:
        if entry.get("encoded_by") == "graph_node":
            continue
        columns.setdefault(to_frame[entry["entity"]], []).append(entry["input"])
    for row in bindings["bindings"]:
        for item in binding_relation_inputs(row):
            for column in (item.column, item.role_column):
                if column is not None:
                    columns.setdefault("person", []).append(column)
    return [
        slice_(entity, sorted(set(names))) for entity, names in sorted(columns.items())
    ]


def add_evidence(graph):
    """Declare the five release-evidence producers and the battery's edges.

    The support producers slice the columns the composed roster names, read
    from this graph with the packaged resources.
    """
    from microcosm.build.transport import compose

    spec = copy.deepcopy(compose.load_transport_spec("nz"))
    spec["resources"]["transport_graph"] = copy.deepcopy(graph)
    ranged = compose.support_roster_slices(spec, {"copied", "normalized", "currency"})
    bounded = compose.support_roster_slices(
        spec, {"copied", "normalized", "currency", "quantile_mapped"}
    )
    rows = graph["nodes"]
    money = next(
        row
        for row in rows
        if row["kernel"] == "money.normalize@1"
        and row.get("population") == "nz.central"
    )
    roster = {"prepared": "support_roster"}
    evidence = [
        node(
            "nz.evidence.donor_receipt",
            "transport.donor_receipt@1",
            population="nz.open",
            sources=["populace_us_donor"],
            params={
                "donor_source": "populace_us_donor",
                "donor_sha256": select("transport_inputs", "value", "donor", "sha256"),
                "donor_size": select(
                    "transport_inputs", "value", "donor", "size_bytes"
                ),
                "donor_locator": select(
                    "transport_inputs", "value", "donor", "locator"
                ),
                "donor_revision": select(
                    "transport_inputs", "value", "donor", "revision"
                ),
                "source_stages": select("source_stages"),
                "source_stages_sha256": select("source_stages", "sha256"),
            },
        ),
        node(
            "nz.evidence.donor_ranges",
            "transport.donor_ranges@1",
            population="nz.open",
            inputs=[slice_(item["entity"], item["columns"]) for item in ranged],
            params={"support_roster": roster},
        ),
        node(
            "nz.evidence.input_closure",
            "transport.input_closure@1",
            population="nz.terminal",
            inputs=closure_slices(),
            params={
                **document("closure", "axiom_input_closure"),
                **document("mapping", "concept_mapping"),
                **document("bindings", "axiom_rules_bindings"),
                "engine_refs": {"prepared": "engine_refs"},
            },
        ),
        node(
            "nz.evidence.weights",
            "transport.weights_audit@1",
            population="nz.terminal",
            inputs=[
                slice_("person", ["age"]),
                slice_("household", [SOURCE_HOUSEHOLD_ID_COLUMN]),
                slice_(unit_schema()[0].entity, []),
            ],
            params={
                "weight_entity": "household",
                "unit_entities": {"prepared": "unit_entities"},
            },
            artifact_inputs=[
                edge("problem", "nz.targets.problem"),
                edge("solution", "nz.calibrate"),
                edge("result", "nz.calibrate"),
            ],
        ),
        node(
            "nz.evidence.support",
            "transport.support_bounds@1",
            population="nz.terminal",
            sources=list(money["sources"]),
            inputs=[slice_(item["entity"], item["columns"]) for item in bounded],
            params={
                "support_roster": roster,
                **{
                    name: money["params"][name]
                    for name in (
                        "declaration",
                        "declaration_sha256",
                        "factor_context",
                        "factor_context_sha256",
                        "profile_selection",
                    )
                },
            },
            artifact_inputs=[edge("donor_ranges", "nz.evidence.donor_ranges")],
        ),
    ]
    battery = next(row for row in rows if row["id"] == "nz.gates.terminal")
    rows[rows.index(battery) : rows.index(battery)] = evidence
    battery["artifact_inputs"] += [
        edge("result", "nz.calibrate"),
        edge("donor_artifact_receipt", "nz.evidence.donor_receipt"),
        edge("axiom_input_closure", "nz.evidence.input_closure"),
        edge("donor_ranges", "nz.evidence.donor_ranges"),
        edge("donor_support_bounds", "nz.evidence.support"),
        edge("fit_weight_records", "nz.evidence.weights"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        validate_earnings_bands(read("earnings_bands"))
    for name, value in documents().items():
        target = NZ / name
        rendered = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
        if args.check:
            if not target.exists() or target.read_text() != rendered:
                raise SystemExit(f"Generated resource is stale: {target}")
        else:
            target.write_text(rendered)
    print("NZ transport graph resources match the generator.")


if __name__ == "__main__":
    main()
