"""Synthetic end-to-end NZ-shaped graph; no output estimates New Zealand.

The forty invented donor households, facts, tax, benefit amounts, exchange
rate, factors and AS rules serve software tests only. The graph exercises the
real transport kernels with pure Python toy engine adapters, never real data
or a country engine. Toy tax applies one flat invented rate and carries no
claim about NZ law, withholding or policy accuracy.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256

import pandas as pd

from microcosm.build.ledger_artifact import (
    chronicle_feed_identity,
    load_ledger_consumer_artifact,
)
from microcosm.build.reference_factors import resolve_reference_factor
from microcosm.build.transport import compose as compose_module
from microcosm.build.transport.codecs import FACTS_SOURCE_CODEC
from microcosm.build.transport.compose import TransportGraphConfig
from microcosm.build.transport.evidence_contract import binding_relation_inputs
from microcosm.build.transport.registry import build_transport_registry
from microcosm.build.transport.takeup_kernels import reporting_cell_declarations
from microcosm.frame import Frame, VariableMetadata, WeightKind, Weights
from microcosm.frame.adapters.axiom import NZ_SCHEMA
from microcosm.frame.concepts import concept_for_column, split_for_transport
from microcosm.frame.transport import derive_transport_seed, read_populace_us_donor
from microcosm.frame.unit_construction import build_benefit_units
from microcosm.graph import Owned
from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import dtype_for_token
from test_support.microcosm_build.chronicle_feed import write_pinned_artifact
from test_support.microcosm_build.money_normalization import (
    DIMENSION,
    graph_declaration,
    money_factor_declaration,
    money_feed_facts,
)
from test_support.microcosm_build.transport_composed import (
    PERIOD,
    ToyRulesEngine,
    _calibration_totals,
    _document,
    _edge,
    _fact,
    _node,
    _owned,
    _references,
    _select,
    _slice,
)
from test_support.microcosm_build.transport_entitlement import (
    TOY_PROGRAMS,
    EntitlementFixture,
    EntitlementToyEngine,
    _as_and_gap,
    _boundary,
    _bridge,
    _seed,
    make_entitlement_fixture,
)
from test_support.microcosm_build.transport_graph import reference_row, write_facts
from test_support.microcosm_build.transport_population import (
    UNIT_RULE as BASE_UNIT_RULE,
)
from test_support.microcosm_build.transport_population import (
    donor_pin,
    write_population_sources,
)

UNIT_RULE = replace(
    BASE_UNIT_RULE,
    dependent_child=replace(
        BASE_UNIT_RULE.dependent_child, parent_independence_min_age=16
    ),
    split_parents="first_parent_unit",
)

SCENARIOS = ("S0", "S1", "V1", "V3", "VHF")
SHARED_BRIDGE = "nz.as.bridge"
FLAGS = ("receives_main", "receives_other", "receives_pension")
GROSS = ("toy_main_annual_gross", "toy_other_annual_gross")
DEMOGRAPHY = tuple(
    (f"toy_demography_{region}_{band}", region, lower, upper)
    for region in ("R1", "R2")
    for band, lower, upper in (("younger", 0, 40), ("older", 40, 100))
)


def synthetic_households(tables):
    """Replace the donor tables with forty invented households and clone lineage."""
    template = tables["person"].iloc[:3].to_dict("records")
    persons, households = [], []
    for h in range(40):
        household = 1000 + h
        source = 1000 if h == 1 else household
        ages = (18 + h % 23, 40 + h % 25, 15 + h % 3)
        if h >= 30:
            ages = (65 + (h - 30), 70 + (h - 30), 16)
        for member, raw in enumerate(template):
            row = dict(raw)
            identity = 10000 + h * 3 + member
            row.update(
                person_id=identity,
                person_household_id=household,
                source_household_id=source,
                source_person_id=f"synthetic-{identity}",
                person_source_id=f"synthetic-{identity}",
                age=ages[member],
                A_LINENO=member + 1,
                A_SPOUSE=(2 if member == 0 else 1) if member < 2 else 0,
                PEPAR1=1 if member == 2 else -1,
                PEPAR2=2 if member == 2 else -1,
                A_EXPRRP=1 if member == 0 else (4 if member == 1 else 5),
                is_household_head=member == 0,
                employment_income_before_lsr=float(10 + h * 3 + member),
                taxable_interest_income=float(h + member + 1),
                tax_exempt_interest_income=0.0,
                rental_income=-120.5 if h == 0 and member == 0 else float(h),
                farm_operations_income=float(member),
                pre_subsidy_rent=float(100 + h) if member == 0 else 0.0,
                bank_account_assets=float(h * 2 + member),
            )
            persons.append(row)
        households.append(
            {
                "household_id": household,
                "household_weight": 0.5 if h < 2 else 1.0,
                "household_support_channel": "asec",
            }
        )
    # Teen parents retain co-resident parent pointers; a parentless child and
    # non-partnered coparents exercise distinct unit-construction inputs.
    for h, age in ((2, 16), (3, 15)):
        first, parent, child = persons[3 * h : 3 * h + 3]
        first.update(A_SPOUSE=0)
        parent.update(age=age, A_SPOUSE=0, PEPAR1=1, A_EXPRRP=5)
        child.update(age=1, PEPAR1=2, PEPAR2=-1)
    persons[12].update(A_SPOUSE=0)
    persons[13].update(A_SPOUSE=0)
    persons[14].update(PEPAR2=2)
    persons[17].update(PEPAR1=-1, PEPAR2=-1)
    tables["person"] = pd.DataFrame(persons)
    tables["household"] = pd.DataFrame(households)


class NzToyRulesEngine(ToyRulesEngine):
    """Invented tax and region-age program adapter, carrying no country estimate."""

    def variables(self):
        if self.program().get("kind") == "tax":
            return ("taxable_income",)
        return super().variables()

    def variable_metadata(self, name):
        if self.program().get("kind") == "tax":
            return VariableMetadata(name, "person", "float", "year")
        return super().variable_metadata(name)

    def materialize(self, bundle, variables, period):
        if self.program().get("kind") != "tax":
            program = self.program()
            simple = [
                name
                for name in variables
                if program["outputs"][name]["operation"] != "region_age"
            ]
            result = super().materialize(bundle, simple, period)
            person = bundle.table("person")
            for name in set(variables) - set(simple):
                row = program["outputs"][name]
                regions = person["person_household_id"].map(
                    bundle.table("household").set_index("household_id")["region"]
                )
                age = person[row["input"]]
                result[name] = (
                    (regions == row["region"])
                    & (age >= row["lower"])
                    & (age < row["upper"])
                ).to_numpy(dtype="int64")
            return result
        self.calls.append((tuple(variables), period))
        tax = bundle.table("person")["taxable_income"].to_numpy() * 0.2
        return {name: tax for name in variables}


class NzEntitlementToyEngine(EntitlementToyEngine):
    """Feed assigned member receipts into the synthetic AS input bridge."""

    def cache_identity(self):
        return {"formula": "synthetic-assigned-receipt-as-v2"}

    def materialize(self, bundle, variables, period):
        result = super().materialize(bundle, variables, period)
        if self.program()["kind"] == "as":
            person = bundle.table("person")
            units = bundle.table("family")["family_id"]
            for name, columns in (
                ("seed_as_super", ["receives_pension"]),
                ("seed_as_beneficiary", ["receives_main", "receives_other"]),
            ):
                if name in variables and set(columns) <= set(person):
                    flags = (
                        person[columns]
                        .any(axis=1)
                        .groupby(person["person_family_id"])
                        .any()
                    )
                    result[name] = units.map(flags).fillna(False).to_numpy(dtype=float)
        return result


def _reprepare(fixture, spec=None):
    spec = fixture.spec if spec is None else spec
    bindings = spec["resources"]["axiom_rules_bindings"]
    tree = fixture.sources["rulespec_nz"]
    engines = {
        row["id"]: (
            NzEntitlementToyEngine if row["id"] in TOY_PROGRAMS else NzToyRulesEngine
        )(tree, row)
        for row in bindings["bindings"]
    }
    registry = build_transport_registry(
        bindings, tree, unit_rule=UNIT_RULE, engines_by_binding=engines
    )
    return replace(
        fixture,
        spec=spec,
        registry=registry,
        engines_by_binding=engines,
        config=replace(fixture.config, engine_refs=registry.engine_refs, extensions=()),
    )


class NzToyFixture(EntitlementFixture):
    def reprepare(self, spec=None):
        return _reprepare(self, spec)


def _normalizer(population, prefix, *, scenario=False, hold_fixed=False):
    declaration = "money_hold_fixed" if hold_fixed else "money_normalization"
    columns = graph_declaration()["columns"]
    return _node(
        f"{prefix}.normalize",
        "money.normalize@1",
        population=population,
        sources=["money_facts"],
        inputs=[_slice(entity, names) for entity, names in columns.items()],
        outputs=[
            output
            for entity, names in columns.items()
            for name in names
            for output in (
                _owned(entity, name, "float64", rewrite=True),
                _owned(entity, name + "_cents", "int64"),
            )
        ],
        params={
            "declaration": _select(declaration),
            "declaration_sha256": _select(declaration, "sha256"),
            "factor_context": _select("money_factors"),
            "factor_context_sha256": _select("money_factors", "sha256"),
            "profile_selection": {"scenario": "knobs", "encoding": "json"}
            if scenario
            else _select("central_knobs"),
            "chronicle_feed_pin": _select("money_feed_pin"),
        },
    )


def _money_bridges(population, prefix):
    nodes = []
    for short, source in (("main", "toy_main_payment"), ("other", "toy_other_payment")):
        output = f"toy_{short}_annual_gross"
        nodes.append(
            _node(
                f"{prefix}.gross_up.{short}",
                "simulate.solve_zero@1",
                population=population,
                sources=["rulespec_nz"],
                inputs=[_slice("person", [source])],
                outputs=[_owned("person", output, "float64")],
                params={
                    "engine_ref": {"binding": "toy_tax", "field": "engine_ref"},
                    "period": {"binding": "toy_tax", "field": "period"},
                    "variables": ["toy_tax"],
                    "input_overrides": "[]",
                    "output_coefficients": canonical_json(
                        [
                            {
                                "source_entity": "person",
                                "source_column": source,
                                "coefficient": 52,
                                "entity": "person",
                                "column": output,
                            },
                            {
                                "variable": "toy_tax",
                                "coefficient": 1,
                                "entity": "person",
                                "column": output,
                            },
                            {
                                "trial": True,
                                "coefficient": -1,
                                "entity": "person",
                                "column": output,
                            },
                        ]
                    ).decode("utf-8"),
                    "solve_input": canonical_json(
                        {"entity": "person", "column": "taxable_income"}
                    ).decode("utf-8"),
                    "bracket": [0, 1000000],
                    "tolerance": 0.00000001,
                    "iterations": 48,
                    "quantum": "0.01",
                    "tie_rule": "half_even",
                    "quantized_residual_bound": "0.0050001",
                    "certificate_margin": "0.000001",
                    "extra_evaluation_budget": 9,
                },
            )
        )
    minor = {
        "entity": "person",
        "columns": [
            {
                "source_column": source,
                "output_column": source + "_cents",
                "quantum": "0.01",
                "rounding": "exact",
            }
            for source in GROSS
        ]
        + [
            {
                "source_column": "toy_pension_payment",
                "output_column": "toy_pension_weekly_cents",
                "quantum": "0.01",
                "rounding": "half_even",
                "source_tax_basis": "before_tax",
            }
        ],
    }
    nodes.append(
        _node(
            f"{prefix}.gross_up.cents",
            "money.minor_units@1",
            population=population,
            inputs=[_slice("person", [*GROSS, "toy_pension_payment"])],
            outputs=[
                _owned("person", row["output_column"], "int64")
                for row in minor["columns"]
            ],
            params={
                "declaration": canonical_json(minor).decode("utf-8"),
            },
        )
    )
    return nodes


def _annual_transfers(population, prefix):
    transfers = [
        {
            "flag_column": flag,
            "source_column": source,
            "source_basis": basis,
            "output_column": f"toy_{short}_transfer_cents",
            "source_tax_basis": "before_tax",
        }
        for short, flag, source, basis in (
            ("main", FLAGS[0], GROSS[0] + "_cents", "annual_gross_cents"),
            ("other", FLAGS[1], GROSS[1] + "_cents", "annual_gross_cents"),
            (
                "pension",
                FLAGS[2],
                "toy_pension_weekly_cents",
                "weekly_before_tax_cents",
            ),
        )
    ]
    declaration = {
        "entity": "person",
        "weeks_per_year": 52,
        "exposure": {"central": 1, "convention": "full_year_stock_equivalent"},
        "transfers": transfers,
    }
    return _node(
        f"{prefix}.annual_transfers",
        "money.annual_transfers@1",
        population=population,
        inputs=[
            _slice("person", [*FLAGS, *(row["source_column"] for row in transfers)])
        ],
        outputs=[_owned("person", row["output_column"], "int64") for row in transfers],
        params={
            "declaration": canonical_json(declaration).decode("utf-8"),
        },
    )


def _classifiers():
    income = {
        "entity": "family",
        "person_entity": "person",
        "unit_id_column": "family_id",
        "membership_column": "person_family_id",
        "weight_entity": "household",
        "expected_weight_kind": "calibrated",
        "tie_key": "family_id",
        "boundary_scenario": "S0",
        "n_bins": 10,
        "rank_method": "weighted_midpoint",
        "components": [
            {
                "entity": "person",
                "column": name,
                "kind": "market",
                "tax_basis": "before_tax",
                "annualisation_factor": 1,
            }
            for name in ("employment_income", "interest_income", "rental_income")
        ]
        + [
            {
                "entity": "person",
                "column": "toy_pension_payment",
                "kind": "benefit",
                "tax_basis": "before_tax",
                "annualisation_factor": 52,
                "receipt_column": "receives_pension",
            }
        ],
        "excluded_columns": [{"entity": "family", "column": "toy_as_weekly"}],
    }
    receipt = {
        "entity": "family",
        "person_entity": "person",
        "unit_id_column": "family_id",
        "membership_column": "person_family_id",
        "predicates": {
            "superannuitant": {
                "entity": "person",
                "columns": ["receives_pension"],
                "reduction": "any",
            },
            "beneficiary": {
                "entity": "person",
                "columns": ["receives_main", "receives_other"],
                "reduction": "any",
            },
        },
        "priority": ["superannuitant", "beneficiary"],
        "default": "other",
    }
    income["components"] += [
        {
            "entity": "person",
            "column": column,
            "kind": "benefit",
            "tax_basis": "before_tax",
            "annualisation_factor": 1,
            "receipt_column": flag,
        }
        for column, flag in zip(GROSS, FLAGS[:2], strict=True)
    ]
    return {"income": income, "receipt": receipt}


def _reporting(population, prefix, scenario):
    baseline = scenario == "S0"
    income = _node(
        f"{prefix}.reporting.income_decile",
        "reporting.income_decile@1" if baseline else "reporting.carry_income_decile@1",
        population=population,
        inputs=[
            _slice(
                "person",
                [
                    "employment_income",
                    "interest_income",
                    "rental_income",
                    "toy_pension_payment",
                    *GROSS,
                    *FLAGS,
                ],
            ),
            _slice("household", ["source_household_id"]),
        ],
        outputs=[_owned("family", "income_decile", "int64")],
        params={
            "classifier": _select("toy_reporting", "json", "income"),
            "classifier_sha256": _select("toy_reporting", "sha256"),
            "scenario": scenario,
            "entities": ["person", "household", "family"],
        },
    )
    if not baseline:
        income["artifact_inputs"] = [
            _edge("baseline_cells", "nz.scn.S0.reporting.income_decile")
        ]
    receipt = _node(
        f"{prefix}.reporting.receipt_group",
        "reporting.receipt_group@1",
        population=population,
        inputs=[_slice("person", FLAGS)],
        outputs=[_owned("family", "receipt_group", "string")],
        params={
            "classifier": _select("toy_reporting", "json", "receipt"),
            "classifier_sha256": _select("toy_reporting", "sha256"),
            "scenario": scenario,
        },
    )
    return [income, receipt]


#: Invented P1-style factor for the toy's scaled wage edges (G8-r2c E).
TOY_EARNINGS_FACTOR = Fraction(5, 4)
#: The toy wage cuts before scaling, in currency units; the last band is open.
TOY_EARNINGS_CUTS = (100, 200, 400, None)
#: Exact anchor counts: (200, 400] holds 3 and the open top 1, so the grouped
#: Pareto MLE solves 2^-alpha = 1/4, alpha = 2.
TOY_TOP_BAND_COUNT = 1.0
TOY_EARNINGS_TOP = "toy_person_top_band"


def toy_earnings_bands():
    """The toy counterpart of earnings_bands.json: cut × factor, one open top."""
    edges = [
        None if cut is None else float(Fraction(cut) * TOY_EARNINGS_FACTOR)
        for cut in TOY_EARNINGS_CUTS
    ]
    references = ("toy_person_lower_band", "toy_person_upper_band", TOY_EARNINGS_TOP)
    return {
        "map_bands": {
            "bands": [
                {"lower": edges[i], "upper": edges[i + 1], "reference": reference}
                for i, reference in enumerate(references)
            ]
        },
        "tail": {
            "method": "pareto",
            "estimator": {
                "id": "grouped_pareto_mle@1",
                "anchor_references": list(references[1:]),
                "bracket": ["1.00000095367431640625", "64"],
                "iterations": 64,
            },
        },
    }


def _earnings_map_v2(resources, rows, facts):
    """Map toy wages with ``transport.quantile_map@2``, as NZ's nz.earnings.map does."""
    resources["toy_earnings_bands"] = toy_earnings_bands()
    resources["precal_references"]["target_references"].append(
        reference_row(TOY_EARNINGS_TOP, entity="person", measure="toy_person_count")
    )
    facts.append(_fact(TOY_EARNINGS_TOP, TOY_TOP_BAND_COUNT))
    mapper = next(
        row for row in rows if row["id"] == "nz.transport.qmap.employment_income"
    )
    mapper["kernel"] = "transport.quantile_map@2"
    mapper["params"].update(
        bands=_select("toy_earnings_bands", "json", "map_bands"),
        bands_sha256=_select("toy_earnings_bands", "sha256"),
        interpolation=_select("toy_earnings_bands", "json", "tail"),
    )


# ---------------------------------------------------------------------------
# The toy income stand-in (G8-r2c G at toy scale)
# ---------------------------------------------------------------------------

#: The toy's counterpart of NZ's ird_taxable_income_ty2027_v1.
TOY_MEASUREMENT = "toy_income_standin_v1"
TOY_STANDIN_NODE = "nz.income.standin"
TOY_STANDIN_TABLE = "toy_income_standin"
TOY_STANDIN_OUTPUTS = {
    "measure": "toy_income_proxy_cents",
    "unit": "toy_income_proxy_nzd",
    "universe_status": "toy_income_proxy_universe_status",
}
#: Signed components, as N/income_measurements.json lists NZ's: wages, signed
#: farm income, interest, rental (the one add-back) and the three annual
#: transfers the toy's annual-transfers node writes.
TOY_STANDIN_COMPONENTS = (
    "employment_income_cents",
    "farm_self_employment_income_cents",
    "interest_income_cents",
    "rental_income_cents",
    "toy_main_transfer_cents",
    "toy_other_transfer_cents",
    "toy_pension_transfer_cents",
)
#: The coarse table's source-period cuts in cents; the last band is open.
#: There is no nil band, and the first cut sits above zero, as NZ's does.
TOY_STANDIN_FIRST_CUT = 15_000
TOY_STANDIN_CUTS = (40_000, 60_000)
#: Native member labels of each coarse band, in band order. A coarse row
#: sums its members' facts, as F's 24 rows sum IRD's native rows.
TOY_STANDIN_MEMBERS = (
    ("n01", "n02", "n03"),
    ("n04", "n05"),
    ("n06", "n07", "n08", "n09"),
)
#: Invented native facts: (count, amount in currency units) per member,
#: chosen near the toy's pre-fit weighted totals so R2C-3's ratio bounds hold.
TOY_STANDIN_FACTS = {
    "n01": (17.0, 5_000.0),
    "n02": (19.0, 5_000.0),
    "n03": (18.0, 5_100.0),
    "n04": (20.0, 9_300.0),
    "n05": (19.0, 9_200.0),
    "n06": (6.0, 9_000.0),
    "n07": (6.0, 8_000.0),
    "n08": (5.0, 7_000.0),
    "n09": (5.0, 6_700.0),
}
#: The toy's own ageing dimension; its factors come from the shared fixture.
TOY_AGING_DIMENSION = "toy_aging_index"
#: Support thresholds suited to forty households; R2C-3's ratio bounds stand.
TOY_PREFIT_SUPPORT = {"minimum_source_households": 3, "minimum_kish_ess": 2}


def toy_standin_declaration():
    """The toy counterpart of income_measurements.json's ``measures.central``."""
    return {
        "entity": "person",
        "components": [
            {"column": column, "sign": 1} for column in TOY_STANDIN_COMPONENTS
        ],
        "add_backs": ["rental_income_cents"],
        "floor": 0,
        "scale": 100,
        "outputs": dict(TOY_STANDIN_OUTPUTS),
    }


def toy_standin_table():
    """The toy counterpart of ird_taxable_income_standin_ty2024."""
    from test_support.microcosm_build.reference_intervals import (
        interval_table_fixture,
    )

    table = interval_table_fixture(
        table_id=TOY_STANDIN_TABLE,
        cuts=TOY_STANDIN_CUTS,
        nil=False,
        first=TOY_STANDIN_FIRST_CUT,
        companions=True,
        indicator_prefix="toy_income_standin_band",
    )
    table.update(
        income_column=TOY_STANDIN_OUTPUTS["measure"],
        universe_column=TOY_STANDIN_OUTPUTS["universe_status"],
    )
    table["diagnostics"]["support_household_entity"] = "household"
    for band in table["bands"]:
        band["indicator"] += "_indicator"
        band.update(
            count_reference=f"toy_income_standin_count__{band['id']}",
            amount_reference=f"toy_income_standin_amount__{band['id']}",
        )
    return table


def toy_standin_member_facts():
    """One count and one amount fact per native member, NZ-geography."""
    from test_support.microcosm_build.reference_intervals import band_fact

    rows = []
    for label, (count, amount) in TOY_STANDIN_FACTS.items():
        for kind, value in (("count", count), ("money", amount)):
            fact = band_fact(kind, label, value, record_set="toy.standin.bands")
            fact["geography"].update(id="NZ", name="Synthetic destination")
            rows.append(fact)
    return rows


def toy_standin_references(table):
    """F's multi-member rows at toy scale: ``sum`` counts, ``identity`` amounts."""
    from test_support.microcosm_build.reference_intervals import summed_row

    rows = []
    for band, members in zip(table["bands"], TOY_STANDIN_MEMBERS, strict=True):
        for kind, name in (
            ("count", band["count_reference"]),
            ("money", band["amount_reference"]),
        ):
            row = summed_row(kind, name, members, indicator=band["indicator"])
            row.update(
                family="income",
                measure="toy_people"
                if kind == "count"
                else TOY_STANDIN_OUTPUTS["unit"],
            )
            row["metadata"].update(
                required_reference_profile=TOY_AGING_DIMENSION,
                required_measurement_binding=TOY_MEASUREMENT,
                reference_prerequisites=json.dumps(
                    [
                        f"reference_profile:{TOY_AGING_DIMENSION}",
                        f"measurement:{TOY_MEASUREMENT}",
                    ]
                ),
                interval_table_id=table["id"],
            )
            rows.append(row)
    return rows


def _income_standin(resources, rows, facts, problem):
    """Wire the toy stand-in as G wires nz.income.standin, prepare and problem."""
    from test_support.microcosm_build.reference_intervals import prefit_gate_entry

    declaration = toy_standin_declaration()
    resources["toy_income_measurements"] = {
        "schema_version": 1,
        "measures": {
            "central": {
                "declaration": declaration,
                "declaration_sha256": sha256(canonical_json(declaration)).hexdigest(),
            }
        },
    }
    # E's nz.earnings.cents: the mapped float wage becomes integer cents.
    wages = {
        "entity": "person",
        "columns": [
            {
                "source_column": "employment_income",
                "output_column": "employment_income_cents",
                "quantum": "0.01",
                "rounding": "half_even",
            }
        ],
    }
    rows.append(
        _node(
            "nz.earnings.cents",
            "money.minor_units@1",
            population="nz.central",
            inputs=[_slice("person", ["employment_income"])],
            outputs=[_owned("person", "employment_income_cents", "int64")],
            params={"declaration": canonical_json(wages).decode("utf-8")},
        )
    )
    rows.append(
        _node(
            TOY_STANDIN_NODE,
            "measure.combine@1",
            population="nz.central",
            inputs=[_slice("person", list(TOY_STANDIN_COMPONENTS))],
            outputs=[
                _owned("person", TOY_STANDIN_OUTPUTS["measure"], "Int64"),
                _owned("person", TOY_STANDIN_OUTPUTS["unit"], "float64"),
                _owned("person", TOY_STANDIN_OUTPUTS["universe_status"], "string"),
            ],
            params={
                "measure": _select(
                    "toy_income_measurements",
                    "json",
                    "measures",
                    "central",
                    "declaration",
                ),
                "measure_sha256": _select(
                    "toy_income_measurements",
                    "value",
                    "measures",
                    "central",
                    "declaration_sha256",
                ),
                "measurement_id": TOY_MEASUREMENT,
            },
        )
    )
    table = toy_standin_table()
    resources["toy_reference_intervals"] = {"schema_version": 1, "tables": [table]}
    references = toy_standin_references(table)
    document = resources["target_references"]
    document["target_references"] += references
    document["hierarchy"].setdefault("target_labels", {}).update(
        {
            row["metadata"]["contract_target_id"]: f"Synthetic {row['name']}"
            for row in references
        }
    )
    facts += toy_standin_member_facts()
    # The compiler replays each selected factor against the facts it loads,
    # under the feed pin (_pin_calibration_facts writes both).
    for row in rows:
        if row["kernel"] == "targets.compile@1":
            row["params"]["chronicle_feed_pin"] = _select("toy_calibration_feed_pin")
    transformation = {
        "operation": "scale_money_and_intervals",
        "reference_names": [band["amount_reference"] for band in table["bands"]],
        "count_reference_names": [band["count_reference"] for band in table["bands"]],
        "interval_table_ids": [table["id"]],
        "required_measurement_binding": TOY_MEASUREMENT,
    }
    resources["reference_profiles"] = {
        "schema_version": 1,
        "reference_resource": "target_references",
        "factor_context_resource": "toy_reference_aging",
        "interval_resource": "toy_reference_intervals",
        "dimensions": {
            TOY_AGING_DIMENSION: {
                "default": "central",
                # One profile: the toy feed carries one factor's anchor facts.
                "profiles": {
                    "central": {
                        "transformations": [
                            {**transformation, "factor_id": "toy_central"}
                        ]
                    }
                },
            }
        },
    }
    bindings = resources["calibration_bindings"]
    measured = [TOY_STANDIN_OUTPUTS["measure"], TOY_STANDIN_OUTPUTS["universe_status"]]
    added = [{"entity": "person", "name": "toy_people", "constant": 1}] + [
        {
            "entity": "person",
            "name": band["indicator"],
            "interval": {"table": table["id"], "band": band["id"]},
        }
        for band in table["bands"]
    ]
    bindings["columns"] += added
    prepare = next(row for row in rows if row["id"] == "nz.calibration.prepare")
    prepare["inputs"].append(_slice("person", measured))
    prepare["outputs"] += [_owned("person", row["name"], "bool") for row in added]
    prepare["outputs"][-len(added)]["dtype"] = "int64"
    # R2C-25: the problem slices the unit column, or _compiled_problem
    # (T/target_kernels.py:541-551) refuses every amount row.
    problem["inputs"] += [
        _slice(
            "person",
            [row["name"] for row in added] + [*measured, TOY_STANDIN_OUTPUTS["unit"]],
        ),
        _slice("household", ["source_household_id"]),
    ]
    resources["gates"]["gates"] += [
        {
            "id": "toy_target_profile_coverage",
            "gate": "target_profile_coverage",
            "phase": "terminal",
            "criticality": "release_blocking",
            "evidence_absent_blocks": True,
            "parameters": {"required_families": ["income", "toy"]},
        },
        {
            **prefit_gate_entry(
                tables=[table["id"]],
                evidence=list(TOY_PREFIT_EVIDENCE),
                **TOY_PREFIT_SUPPORT,
            ),
            "id": "toy_measurement_prefit",
            # R2C-18 leaves every alias unwired, so the gate fails every run;
            # as a diagnostic it reports that without blocking the toy export.
            "criticality": "diagnostic",
            "evidence_absent_blocks": True,
        },
    ]


def _pin_calibration_facts(resources, facts, root):
    """Write the calibration facts and the ageing factors' facts as one feed.

    NZ's IRD ageing factors resolve against the pinned Chronicle feed the
    compiler loads; the toy's ageing factors do the same against its own.
    """
    from test_support.microcosm_build.reference_factors import factor_fixture

    _, factor_facts, _ = factor_fixture()
    plain = json.loads(json.dumps(factor_facts, default=float))
    path, pin = write_pinned_artifact(root, [*facts, *plain])
    pin["source_commit"] = "c" * 40
    identity = chronicle_feed_identity(pin)
    artifact = load_ledger_consumer_artifact(path, feed_pin=pin, lexical_floats=True)
    declaration, _, _ = factor_fixture(feed_identity=identity)
    declaration["id"] = "toy_central"
    factor, receipt = resolve_reference_factor(declaration, artifact.facts, identity)
    entries = [
        {"declaration": declaration, "operational_factor": factor, "receipt": receipt}
    ]
    resources["toy_calibration_feed_pin"] = pin
    resources["toy_reference_aging"] = {
        "schema_version": 1,
        "feed_identity": identity,
        "factors": entries,
    }
    return path


#: The packaged gate's required_evidence list (N/gates.json measurement_prefit).
TOY_PREFIT_EVIDENCE = (
    "native_band_report",
    "omission_register",
    "scenario_vsi",
    "scenario_vex",
    "scenario_vpt",
    "identity_receipts",
)


def make_nz_toy_fixture(root):
    """Prepare forty synthetic households and the complete transport declaration."""
    base = make_entitlement_fixture(root, variants=False)
    spec = copy.deepcopy(base.spec)
    r = spec["resources"]
    population = write_population_sources(
        root / "forty-households",
        mass=120,
        edit_tables=synthetic_households,
        time_period=2024,
    )
    frame = _synthetic_frame(population)
    structural = {
        "person": {"person_id", "person_household_id", "person_family_id"},
        "household": {"household_id"},
        "family": {"family_id"},
    }
    create_outputs = tuple(
        Owned(entity, column, str(frame.table(entity)[column].dtype))
        for entity in ("person", "household", "family")
        for column in frame.table(entity)
        if column not in structural[entity]
    )
    r["benefit_unit_rule"] = UNIT_RULE.to_dict()
    sources = dict(base.sources)
    sources["donor_pool"] = population.donor
    r["donor_pin"] = donor_pin(population.donor)
    facts = [
        json.loads(row)
        for row in sources["nz_calibration_facts"].read_text().splitlines()
    ]
    totals = {name: value for name, _, value in _calibration_totals(frame)}
    for fact in facts:
        name = fact["semantic_fact_key"].rsplit(":", 1)[-1]
        if name in totals:
            fact["value"] = totals[name]
    write_facts(sources["nz_calibration_facts"], facts)
    path, pin, context = _quarterly_money_feed(root / "money-feed")
    sources["money_facts"] = path
    r.update(
        money_normalization=graph_declaration(),
        money_feed_pin=pin,
        money_factors=context,
        central_knobs={DIMENSION: "central", "receipt_draw": "central"},
        toy_reporting=_classifiers(),
    )
    r["money_hold_fixed"] = copy.deepcopy(r["money_normalization"])
    r["money_hold_fixed"]["index_family_factors"] = {
        family: {"identity": True, "reason": "Synthetic hold-fixed sensitivity."}
        for family in r["money_normalization"]["index_family_factors"]
    }
    # Extend the same content-pinned toy rule module, never a statutory module.
    bindings = r["axiom_rules_bindings"]["bindings"]
    for binding in bindings:
        tree_path = sources["rulespec_nz"] / binding["rulespec_path"]
        program = json.loads(tree_path.read_text())
        if binding["id"] == "main_benefit":
            program["outputs"]["toy_main_payment"].update(
                input="input_age", coefficient=0.04
            )
            program["outputs"]["toy_other_eligible"] = {
                "input": "input_age",
                "operation": "at_most",
                "threshold": 64,
                "dtype": "int64",
            }
            program["outputs"]["toy_other_adult"] = {
                "input": "input_age",
                "operation": "at_least",
                "threshold": 18,
                "dtype": "int64",
            }
            program["outputs"]["toy_other_payment"] = {
                "input": "input_age",
                "operation": "linear",
                "coefficient": 0.02,
                "offset": 3,
                "dtype": "float64",
            }
            for name, region, lower, upper in DEMOGRAPHY:
                program["outputs"][name] = {
                    "input": "input_age",
                    "operation": "region_age",
                    "region": region,
                    "lower": lower,
                    "upper": upper,
                    "dtype": "int64",
                }
        if binding["id"] == "pension":
            program["outputs"]["toy_pension_eligible"]["threshold"] = 65
        variables = list(program["outputs"])
        if binding["id"] == "main_benefit":
            # No request reads it, but the module keeps reading the toy
            # mapping's bound income input, as its closure entry says.
            program["outputs"]["toy_income_probe"] = {
                "input": "input_income",
                "operation": "linear",
                "coefficient": 0,
                "offset": 0,
                "dtype": "float64",
            }
        payload = canonical_json(program)
        tree_path.write_bytes(payload)
        binding.update(sha256=sha256(payload).hexdigest(), variables=variables)
    r["target_references"] = _references(
        [
            reference_row(name, entity="person", measure=name + "_indicator")
            for name, _, _, _ in DEMOGRAPHY
        ]
    )
    for name, region, lower, upper in DEMOGRAPHY:
        fact = _fact(name, 30)
        fact["label"] = f"Synthetic {region} population ages {lower} to {upper - 1}"
        fact["geography"].update(level="region", id=region, name=f"Synthetic {region}")
        age_band = f"{lower}-{upper - 1}"
        fact["dimensions"] = {"age_band": age_band}
        fact["dimension_labels"] = {"age_band": "Synthetic age band"}
        fact["dimension_value_labels"] = {
            "age_band": {age_band: f"Synthetic ages {lower} to {upper - 1}"}
        }
        facts.append(fact)
    tax = {
        "entity": "person",
        "kind": "tax",
        "relations": [],
        "outputs": {"toy_tax": {"dtype": "float64"}},
    }
    tax_path = sources["rulespec_nz"] / "toy/tax.yaml"
    tax_path.write_bytes(canonical_json(tax))
    bindings.append(
        {
            "id": "toy_tax",
            "rulespec_path": "toy/tax.yaml",
            "sha256": sha256(tax_path.read_bytes()).hexdigest(),
            "entity": "person",
            "engine_entity": "Person",
            "period": PERIOD,
            "variables": ["toy_tax"],
        }
    )
    skeleton = r["transport_graph"]
    rows = skeleton["nodes"]
    rows[:] = [row for row in rows if row["id"] != "nz.transport.currency"]
    rows.insert(
        next(
            i for i, row in enumerate(rows) if row["id"].startswith("nz.transport.qmap")
        ),
        _boundary("nz.central", "nz.open"),
    )
    _earnings_map_v2(r, rows, facts)
    central_members = {
        "nz.encode.person",
        "nz.rules.main_benefits",
        "nz.rules.pension",
        "nz.takeup.assign",
        "nz.targets.compile",
        "nz.targets.problem",
    }
    for row in rows:
        if row["id"] in central_members or row["id"].startswith("nz.transport.qmap"):
            row["population"] = "nz.central"
        if row["id"] == "nz.calibrate":
            row["base"] = "nz.central"
        if row["id"] == "nz.rules.main_benefits":
            row["inputs"].append(_slice("household", ["region"]))
            row["outputs"] += [
                _owned("person", name, "int64") for name, _, _, _ in DEMOGRAPHY
            ]
            row["outputs"] += [
                _owned("person", name, dtype)
                for name, dtype in (
                    ("toy_other_eligible", "int64"),
                    ("toy_other_adult", "int64"),
                    ("toy_other_payment", "float64"),
                )
            ]
    index = next(i for i, row in enumerate(rows) if row["id"] == "nz.central") + 1
    rows.insert(index, _normalizer("nz.central", "nz.central"))
    # The compiler orders these stages through their declared inputs.
    rows.extend(_money_bridges("nz.central", "nz.central"))
    rows.append(_annual_transfers("nz.central", "nz.central"))
    _receipts(r, facts, rows)
    sources["nz_calibration_facts"] = write_facts(
        sources["nz_calibration_facts"], facts
    )
    skeleton["sources"].append({"name": "money_facts", "codec": FACTS_SOURCE_CODEC})
    by_id = {row["id"]: row for row in rows}
    by_id["nz.units.attributes"]["inputs"] = [
        _slice(
            "person",
            ["age", "partner_person_id", "parent_1_person_id", "parent_2_person_id"],
        ),
        _slice("household", ["reference_person_id"]),
        _slice(
            "family",
            [
                "family_household_id",
                "family_head_person_id",
                "family_source_household_id",
            ],
        ),
    ]
    problem = by_id["nz.targets.problem"]
    problem["inputs"].append(_slice("person", ["receives_other"]))
    _prepare(r, rows, problem)
    _income_standin(r, rows, facts, problem)
    sources["nz_calibration_facts"] = _pin_calibration_facts(
        r, facts, root / "calibration-feed"
    )
    # A closed replay slice excludes every scratch cent, formula and reporting output.
    export_inputs = [
        _slice(
            "person",
            [
                "age",
                "employment_income",
                "interest_income",
                "rental_income",
                "farm_self_employment_income",
                "liquid_financial_assets",
                *FLAGS,
                "input_age",
                "input_income",
                "input_pension_age",
            ],
        ),
        _slice("household", ["rent", "source_household_id", "ta", "region"]),
        _slice("family", ["family_source_household_id", "family_household_id"]),
    ]
    by_id["nz.export.prepare"]["inputs"] = export_inputs
    by_id["nz.gates.terminal"]["inputs"] = export_inputs
    r["calibration"]["epochs"] = 4
    scenarios = [
        {"id": "S0", "tier": "entitlement", "knobs": r["central_knobs"]},
        {
            "id": "S1",
            "tier": "entitlement",
            "knobs": {
                "input_overrides": [
                    {"entity": "family", "column": "as_assets", "value": 0}
                ]
            },
        },
        {
            "id": "V1",
            "tier": "calibration",
            "knobs": {DIMENSION: "low", "receipt_draw": "central"},
        },
        {
            "id": "V3",
            "tier": "calibration",
            "knobs": {DIMENSION: "central", "receipt_draw": "variant"},
        },
        {
            "id": "VHF",
            "tier": "calibration",
            "knobs": {
                DIMENSION: "central",
                "receipt_draw": "central",
                "hold_fixed": True,
            },
        },
    ]
    entitlement = r["entitlement_graph"]
    entitlement["scenario_nodes"] = []
    # As in the package, the shared bridge runs on its own FILTER branch.
    shared = [
        row for row in entitlement["nodes"] if row["id"].startswith(SHARED_BRIDGE + ".")
    ]
    for row in shared:
        row["population"] = SHARED_BRIDGE
    entitlement["nodes"].insert(
        entitlement["nodes"].index(shared[0]), _boundary(SHARED_BRIDGE, "nz.as")
    )
    for scenario in scenarios:
        name = scenario["id"]
        if name in ("S0", "S1"):
            pop = f"nz.scn.{name}"
            nodes = [_boundary(pop, SHARED_BRIDGE)]
            if name == "S1":
                nodes.append(
                    _seed(pop, f"{pop}.override", scenario=True, fields=("as_assets",))
                )
            nodes += _as_and_gap(pop, pop, name)
        else:
            nodes, pop = _variant(r, rows, name)
        gap = next(row for row in nodes if row["kernel"] == "takeup.gap@1")
        # Reporting edges govern the cell diagnostics' columns.
        gap["inputs"].append(_slice("family", ["income_decile", "receipt_group"]))
        nodes += _reporting(
            pop, f"nz.scn.{name}" if name.startswith("S") else f"nz.v.{name}", name
        )
        scenario["nodes"] = nodes
    r["scenarios"] = {"scenarios": scenarios, "central_knobs": r["central_knobs"]}
    # Seed bridge nodes explicitly read assigned receipt flags after calibration.
    for row in [
        *entitlement["nodes"],
        *(node for scenario in scenarios for node in scenario["nodes"]),
    ]:
        if any(
            name in row.get("params", {}).get("variables", ())
            for name in ("seed_as_super", "seed_as_beneficiary")
        ):
            row["inputs"][0]["columns"] += list(FLAGS)
    _cell_gates(r)
    _exemptions(r)
    for row in r["entitlement_gap"]["target_references"]:
        r["entitlement_annualisation"][row["name"]] = (
            52 if row["measure"] == "toy_as_weekly" else 1
        )
    _evidence(spec)
    fixture = NzToyFixture(
        root,
        spec,
        TransportGraphConfig(
            engine_refs=base.config.engine_refs, create_outputs=create_outputs
        ),
        base.registry,
        sources,
        base.engines_by_binding,
    ).reprepare()
    _close_toy_inputs(fixture)
    return fixture.reprepare(fixture.spec)


#: The toy's identity and assignment columns (the NZ list's counterpart).
TOY_STRUCTURAL_COLUMNS = (
    ("family", "family_head_person_id", "CREATE synthesizes the family head pointer."),
    (
        "family",
        "family_household_id",
        "CREATE synthesizes the family's household pointer.",
    ),
    (
        "family",
        "family_source_household_id",
        "CREATE synthesizes the family's donor household lineage.",
    ),
    ("household", "donor_support_stratum", "CREATE synthesizes the support stratum."),
    ("person", "take_up_seed", "CREATE derives the take-up seed from the person id."),
    (
        "household",
        "source_household_id",
        "The donor supplies the household identifier.",
    ),
    ("household", "ta", "geography.derive@1 assigns the territorial authority."),
    ("household", "region", "geography.derive@1 assigns the region."),
)
#: The toy's declared limit on the calibrated person-mass change.
TOY_MAXIMUM_RELATIVE_MASS_CHANGE = 0.05


def closure_slices(resources):
    """The closure producer's input slices for the toy's resources.

    The closure's concept, state and default entries feed their entities'
    columns, and the producer reads each declared relation's column and role
    column on the person entity; a slice delivers only the columns it names.
    """
    names = resources["axiom_rules_bindings"]["entity_names"]
    to_frame = {engine: frame for frame, engine in names.items()}
    closure = {}
    for entry in resources["axiom_input_closure"]["entries"]:
        if entry.get("encoded_by") != "graph_node":
            closure.setdefault(to_frame[entry["entity"]], []).append(entry["input"])
    for binding in resources["axiom_rules_bindings"]["bindings"]:
        for item in binding_relation_inputs(binding):
            for column in (item.column, item.role_column):
                if column is not None:
                    closure.setdefault("person", []).append(column)
    return [_slice(e, sorted(set(c))) for e, c in sorted(closure.items())]


def _evidence(spec):
    """Wire the five release-evidence producers and their three gates.

    The toy closure's surface engine commit becomes the toy bindings' engine
    commit, as ruling H36 requires; the donor pin also carries the staged
    artifact the donor receipt authenticates.
    """
    r = spec["resources"]
    commit = r["axiom_rules_bindings"]["engine"]["commit"]
    r["axiom_input_closure"]["input_surface"]["engine"]["commit"] = commit
    pin = r["donor_pin"]
    pin.update(
        locator="synthetic_donor.h5",
        revision="synthetic-forty-households",
        stages=[
            {
                "stage": "load_synthetic_donor",
                "artifacts": [
                    {
                        "locator": "synthetic_donor.h5",
                        "revision": "synthetic-forty-households",
                        "sha256": pin["sha256"],
                        "size_bytes": pin["size"],
                    }
                ],
            }
        ],
    )
    r["gates"]["gates"] += [
        {
            "id": "toy_release_input_coverage",
            "gate": "release_input_coverage",
            "phase": "terminal",
            "criticality": "release_blocking",
            "evidence_absent_blocks": True,
            "parameters": {"allow_engine_optional": False},
        },
        {
            "id": "toy_donor_support_bounds",
            "gate": "support",
            "phase": "terminal",
            "criticality": "release_blocking",
            "evidence_absent_blocks": True,
            "parameters": {
                "structural_columns": [
                    {"entity": entity, "column": column, "reason": reason}
                    for entity, column, reason in TOY_STRUCTURAL_COLUMNS
                ]
            },
        },
        {
            "id": "toy_calibration_weights_audit",
            "gate": "weights_audit",
            "phase": "terminal",
            "criticality": "release_blocking",
            "evidence_absent_blocks": True,
            "parameters": {
                "maximum_relative_mass_change": TOY_MAXIMUM_RELATIVE_MASS_CHANGE
            },
        },
    ]
    rows = r["transport_graph"]["nodes"]
    ranged = compose_module.support_roster_slices(
        spec, {"copied", "normalized", "currency"}
    )
    bounded = compose_module.support_roster_slices(
        spec, {"copied", "normalized", "currency", "quantile_mapped"}
    )
    money = next(row for row in rows if row["id"] == "nz.central.normalize")
    roster = {"prepared": "support_roster"}
    evidence = [
        _node(
            "nz.evidence.donor_receipt",
            "transport.donor_receipt@1",
            population="nz.open",
            sources=["donor_pool"],
            params={
                "donor_source": "donor_pool",
                "donor_sha256": _select("donor_pin", "value", "sha256"),
                "donor_size": _select("donor_pin", "value", "size"),
                "donor_locator": _select("donor_pin", "value", "locator"),
                "donor_revision": _select("donor_pin", "value", "revision"),
                "source_stages": _select("donor_pin", "json"),
                "source_stages_sha256": _select("donor_pin", "sha256"),
            },
        ),
        _node(
            "nz.evidence.donor_ranges",
            "transport.donor_ranges@1",
            population="nz.open",
            inputs=ranged,
            params={"support_roster": roster},
        ),
        # The toy's terminal branches from calibration beside nz.as, so the
        # family module's encoded inputs exist on nz.as only.
        _node(
            "nz.evidence.input_closure",
            "transport.input_closure@1",
            population="nz.as",
            inputs=closure_slices(r),
            params={
                **_document("closure", "axiom_input_closure"),
                **_document("mapping", "concept_mapping"),
                **_document("bindings", "axiom_rules_bindings"),
                "engine_refs": {"prepared": "engine_refs"},
            },
        ),
        _node(
            "nz.evidence.weights",
            "transport.weights_audit@1",
            population="nz.terminal",
            inputs=[
                _slice("person", ["age"]),
                _slice("household", ["source_household_id"]),
                _slice("family", ["family_household_id"]),
            ],
            params={
                "weight_entity": "household",
                "unit_entities": {"prepared": "unit_entities"},
            },
            artifact_inputs=[
                _edge("problem", "nz.targets.problem"),
                _edge("solution", "nz.calibrate"),
                _edge("result", "nz.calibrate"),
            ],
        ),
        _node(
            "nz.evidence.support",
            "transport.support_bounds@1",
            population="nz.terminal",
            sources=list(money["sources"]),
            inputs=bounded,
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
                        "chronicle_feed_pin",
                    )
                    if name in money["params"]
                },
            },
            artifact_inputs=[_edge("donor_ranges", "nz.evidence.donor_ranges")],
        ),
    ]
    battery = next(row for row in rows if row["id"] == "nz.gates.terminal")
    rows[rows.index(battery) : rows.index(battery)] = evidence
    battery["artifact_inputs"] = [
        *battery.get("artifact_inputs", ()),
        _edge("result", "nz.calibrate"),
        _edge("donor_artifact_receipt", "nz.evidence.donor_receipt"),
        _edge("axiom_input_closure", "nz.evidence.input_closure"),
        _edge("donor_ranges", "nz.evidence.donor_ranges"),
        _edge("donor_support_bounds", "nz.evidence.support"),
        _edge("fit_weight_records", "nz.evidence.weights"),
    ]


def _close_toy_inputs(fixture):
    """Close every bound toy module's live roots, as a country closure does.

    The NZ toy rewrites its rule programs and adds the entitlement and tax
    modules, so the base toy closure no longer matches them. Each live root
    keeps its entry, or becomes a concept binding where the toy mapping binds
    it, or a graph_node entry naming the toy node that owns, overrides or
    solves it.
    """
    from microcosm.build.transport.evidence_kernels import engine_root_inputs
    from microcosm.frame.concept_mapping import ConceptMapping

    r = fixture.spec["resources"]
    closure = r["axiom_input_closure"]
    bindings = r["axiom_rules_bindings"]
    names = bindings["entity_names"]
    to_frame = {engine: frame for frame, engine in names.items()}
    bound = {
        binding.ref
        for binding in ConceptMapping.from_dict(r["concept_mapping"]).bindings
    }
    rows = compose_module._declared_rows(fixture.spec)
    owners = {}
    for row in rows:
        if row.get("structural", "none") != "none":
            continue
        for output in row.get("outputs", ()):
            owners.setdefault((output["entity"], output["column"]), row["id"])
        params = row.get("params") or {}
        solve = params.get("solve_input")
        if isinstance(solve, str):
            solve = json.loads(solve)
        if isinstance(solve, dict):
            owners.setdefault((solve["entity"], solve["column"]), row["id"])
        requests = [params]
        components = params.get("components") or []
        if isinstance(components, str):
            components = json.loads(components)
        requests += [item for item in components if isinstance(item, dict)]
        for request in requests:
            overrides = request.get("input_overrides") or []
            if isinstance(overrides, str):
                overrides = json.loads(overrides)
            for item in overrides if isinstance(overrides, list) else ():
                if isinstance(item, dict) and {"entity", "column"} <= set(item):
                    owners.setdefault((item["entity"], item["column"]), row["id"])
    entries = {
        (entry["rulespec_path"], entry["entity"], entry["input"]): entry
        for entry in closure["entries"]
    }
    closed = []
    for row in bindings["bindings"]:
        engine = fixture.registry.engines[fixture.registry.engine_refs[row["id"]]]
        path = row["rulespec_path"]
        closure["rulespec_paths"][path] = row["sha256"]
        for entity, roots in sorted(engine_root_inputs(engine, row, names).items()):
            for name in roots:
                key = (path, entity, name)
                if key in entries:
                    closed.append(entries[key])
                    continue
                if (name, entity, path) in bound:
                    closed.append(
                        {
                            "rulespec_path": path,
                            "entity": entity,
                            "input": name,
                            "class": "encoded",
                            "encoded_by": "concept_binding",
                        }
                    )
                    continue
                owner = owners.get((to_frame[entity], name))
                if owner is None:
                    raise ValueError(f"No toy node supplies {entity}.{name} ({path}).")
                closed.append(
                    {
                        "rulespec_path": path,
                        "entity": entity,
                        "input": name,
                        "class": "encoded",
                        "encoded_by": "graph_node",
                        "node": owner,
                        "reason": f"The synthetic toy node {owner} writes or solves it.",
                    }
                )
    closure["entries"] = closed


def _receipts(resources, facts, rows):
    bindings = {
        "schema_version": 1,
        "required_inputs": {"person": ["age"]},
        "input_domains": {"person": {"age": {"kind": "nonnegative_integer"}}},
        "columns": [
            {
                "entity": "person",
                "name": name,
                "predicates": [
                    {"column": "age", "operator": "ge", "value": lower},
                    {"column": "age", "operator": "lt", "value": upper},
                ],
            }
            for name, lower, upper in (("toy_young", 18, 30), ("toy_adult", 30, 65))
        ],
    }
    resources["calibration_bindings"] = bindings
    programs = []
    for short, flag, judgments, payment in (
        ("main", FLAGS[0], ["toy_main_eligible"], "toy_main_payment"),
        (
            "other",
            FLAGS[1],
            ["toy_other_eligible", "toy_other_adult"],
            "toy_other_payment",
        ),
    ):
        strata = []
        for filter_ in ("toy_young", "toy_adult"):
            name = f"toy_receipt_{short}_{filter_}"
            strata.append({"target": name, "filter": filter_})
            resources["precal_references"]["target_references"].append(
                reference_row(name, entity="person", measure=flag, filter_=filter_)
            )
            facts.append(_fact(name, 0.25 if short == "other" else 1.0))
        programs.append(
            {
                "program": f"toy.{short}",
                "output": flag,
                "judgment_columns": judgments,
                "judgment_reduction": "all",
                "payment_column": payment,
                "strata": strata,
                "outside_strata": "not_assigned",
            }
        )
    programs.append(
        {
            "program": "toy.pension",
            "output": FLAGS[2],
            "judgment_columns": ["toy_pension_eligible"],
            "judgment_reduction": "all",
            "payment_column": "toy_pension_payment",
            "target": "toy_receipt_pension",
        }
    )
    resources["precal_references"]["target_references"].append(
        reference_row("toy_receipt_pension", entity="person", measure=FLAGS[2])
    )
    facts.append(_fact("toy_receipt_pension", 10))
    resources["receipt_contract"] = {
        "receipts": {
            "contract_version": 2,
            "seed_column": "take_up_seed",
            "default_assignment_profile": "central",
            "assignment_knob": "receipt_draw",
            "assignment_profiles": {
                "central": {"draw_key_suffix": ""},
                "variant": {"draw_key_suffix": "variant"},
            },
            "programs": programs,
            "exclusion_groups": [[FLAGS[2], FLAGS[0], FLAGS[1]]],
        }
    }
    receipt = next(row for row in rows if row["id"] == "nz.takeup.assign")
    receipt["outputs"] = [_owned("person", flag, "bool") for flag in FLAGS]
    receipt["inputs"] = [
        _slice(
            "person",
            [
                "age",
                "take_up_seed",
                "toy_main_eligible",
                "toy_main_payment",
                "toy_other_eligible",
                "toy_other_adult",
                "toy_other_payment",
                "toy_pension_eligible",
                "toy_pension_payment",
                *(name for name, _, _, _ in DEMOGRAPHY),
            ],
        )
    ]
    receipt["artifact_inputs"] = [_edge("surface", "nz.facts.precal")]
    receipt["params"].update(
        **_document("calibration_bindings", "calibration_bindings"),
        profile_selection=_select("central_knobs"),
    )


def _prepare(resources, rows, problem):
    bindings = resources["calibration_bindings"]
    for name, _, _, _ in DEMOGRAPHY:
        bindings["required_inputs"]["person"].append(name)
        bindings["input_domains"]["person"][name] = {"kind": "nonnegative_integer"}
        bindings["columns"].append(
            {
                "entity": "person",
                "name": name + "_indicator",
                "predicates": [{"column": name, "operator": "eq", "value": 1}],
            }
        )
    prepare = _node(
        "nz.calibration.prepare",
        "calibration.prepare@1",
        population="nz.central",
        inputs=[_slice("person", bindings["required_inputs"]["person"])],
        outputs=[_owned("person", row["name"], "bool") for row in bindings["columns"]],
        params={
            "bindings": _select("calibration_bindings"),
            **_document("references", "target_references"),
        },
    )
    rows.append(prepare)
    problem["artifact_inputs"].append(_edge("preparation", prepare["id"]))
    problem["inputs"].append(
        _slice("person", [row["name"] for row in bindings["columns"]])
    )


def _variant(resources, rows, name):
    prefix = f"nz.v.{name}"
    own = [_boundary(prefix, "nz.open")]
    selected = []
    for row in rows:
        if row.get("population") == "nz.central":
            clone = copy.deepcopy(row)
            clone["id"] = prefix + "." + row["id"].removeprefix("nz.")
            clone["population"] = prefix
            selected.append(clone)
    names = {
        old["id"]: new["id"]
        for old, new in zip(
            [row for row in rows if row.get("population") == "nz.central"],
            selected,
            strict=True,
        )
    }
    for row in selected:
        if row["kernel"] == "money.normalize@1":
            row["params"]["profile_selection"] = {
                "scenario": "knobs",
                "encoding": "json",
            }
            if name == "VHF":
                row["params"]["declaration"] = _select("money_hold_fixed")
                row["params"]["declaration_sha256"] = _select(
                    "money_hold_fixed", "sha256"
                )
        if row["kernel"] == "takeup.assign@1":
            row["params"]["profile_selection"] = {
                "scenario": "knobs",
                "encoding": "json",
            }
        for edge in row.get("artifact_inputs", []):
            edge["producer"] = names.get(edge["producer"], edge["producer"])
    own += selected
    calibration = copy.deepcopy(
        next(row for row in rows if row["id"] == "nz.calibrate")
    )
    calibration.update(id=f"{prefix}.calibrate", base=prefix)
    calibration["artifact_inputs"][0]["producer"] = names["nz.targets.problem"]
    population = prefix + ".housing"
    group = copy.deepcopy(next(row for row in rows if row["id"] == "nz.as.inputs"))
    group.update(id=prefix + ".inputs", population=population)
    own += [
        calibration,
        _boundary(population, calibration["id"]),
        group,
        _seed(population, prefix + ".seed"),
        *_bridge(population, prefix),
        *_as_and_gap(population, prefix, name),
    ]
    return own, population


def _cell_gates(resources):
    grid = {
        "entity": "family",
        "measure": "toy_as_weekly",
        "period": PERIOD,
        "annualisation_factor": 52,
        "tables": [
            {
                "name": "toy_income",
                "axes": [{"column": "income_decile", "values": list(range(1, 11))}],
            },
            {
                "name": "toy_receipt",
                "axes": [
                    {
                        "column": "receipt_group",
                        "values": ["beneficiary", "superannuitant", "other"],
                    }
                ],
            },
        ],
    }
    resources["scenarios"]["gap_conventions"] = {"reporting_cells": grid}
    resources["gates"]["gates"] += [
        {
            "id": "toy_weight_ess",
            "gate": "weight_ess",
            "phase": "terminal",
            "criticality": "release_blocking",
            "parameters": {
                "minimum_ess_fraction": 0.1,
                "aggregation_column": "source_household_id",
            },
        },
        {
            "id": "toy_cell_support",
            "gate": "cell_support",
            "phase": "terminal",
            "criticality": "release_blocking",
            "evidence_absent_blocks": True,
            "parameters": {
                "required_scenarios": list(SCENARIOS),
                "required_cells": [
                    row["name"] for row in reporting_cell_declarations(grid)
                ],
                "aggregation_column": "source_household_id",
                "required_weight_entity": "household",
                "required_weight_kind": "calibrated",
                "minimum_cell_ess": 0.1,
                "maximum_contribution_cv": 100,
                "maximum_single_donor_dollar_share": 1,
                "target_minimum_cell_ess": 0.1,
                "target_maximum_single_donor_dollar_share": 1,
            },
        },
    ]


def _exemptions(resources):
    selections = {}
    for resource, name, entity, measure in (
        ("target_references", "toy_parked_calibration", "person", "age"),
        ("holdout_references", "toy_parked_tripwire", "person", "receives_main"),
    ):
        row = reference_row(
            name, entity=entity, measure=measure, activation_status="pending"
        )
        resources[resource]["target_references"].append(row)
        selections[resource] = {
            "exempt": [
                {
                    "name": name,
                    "activation_status": "pending",
                    "reason": "The synthetic negative fixture awaits its reviewed producer.",
                }
            ]
        }
    resources["reference_selection"] = {"schema_version": 1, "resources": selections}


def toy_donor_factor(profile):
    """An independent exact oracle for the toy's four-quarter pay factor."""
    q = {
        "low": Fraction(101, 100),
        "central": Fraction(102, 100),
        "high": Fraction(103, 100),
    }[profile]
    ratio = Fraction(110) * sum(q**i for i in range(4)) / 4 / 101
    return Fraction(round(ratio * 10**6), 10**6)


def _quarterly_money_feed(root):
    facts = money_feed_facts()
    quarters = [
        (f"2024-Q{i + 1}", f"2024-{3 * i + 1:02d}-01", end)
        for i, end in enumerate(
            ("2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31")
        )
    ]
    target = [
        (f"2027-Q{i + 1}", f"2027-{3 * i + 1:02d}-01", end)
        for i, end in enumerate(
            ("2027-03-31", "2027-06-30", "2027-09-30", "2027-12-31")
        )
    ]
    source = []
    for i, (period, start, end) in enumerate(quarters):
        fact = copy.deepcopy(facts[i // 2])
        fact.update(
            aggregate_fact_key=f"source-{i}",
            period={"type": "quarter", "value": period},
            period_coverage={"start_date": start, "end_date": end},
        )
        source.append(fact)
    anchor = copy.deepcopy(facts[2])
    anchor.update(
        period={"type": "quarter", "value": target[0][0]},
        period_coverage={"start_date": target[0][1], "end_date": target[0][2]},
    )
    growth = copy.deepcopy(facts[3])
    growth["value"] = 8.243216
    path, pin = write_pinned_artifact(root, [*source, anchor, growth])
    pin["source_commit"] = "c" * 40
    artifact = load_ledger_consumer_artifact(path, feed_pin=pin, lexical_floats=True)
    identity = chronicle_feed_identity(pin)
    entries = []
    for profile, delta in (
        ("low", "-4.182815"),
        ("central", "0"),
        ("high", "4.307665"),
    ):
        declaration = money_factor_declaration(profile, identity)

        def slots(template, schedule, *, projected):
            result = []
            for i, (period, start, end) in enumerate(schedule):
                row = copy.deepcopy(template)
                row["selector"].update(period_type="quarter", period_value=period)
                row["validation"]["period_coverage"] = {
                    "start_date": start,
                    "end_date": end,
                }
                row["sequence"] = i
                if projected:
                    row["projection_offset"] = i
                result.append(row)
            return result

        declaration["source_schedule"] = slots(
            declaration["source_schedule"][0], quarters, projected=False
        )
        declaration["target_schedule"] = slots(
            declaration["target_schedule"][0], target, projected=True
        )
        declaration["subperiods_per_year"] = 4
        declaration["growth"]["scenario_delta"] = delta
        declaration["expected_operational_factor"] = {
            "numerator": int(toy_donor_factor(profile) * 10**6),
            "denominator": 10**6,
        }
        factor, receipt = resolve_reference_factor(
            declaration, artifact.facts, identity
        )
        entries.append(
            {
                "declaration": declaration,
                "operational_factor": factor,
                "receipt": receipt,
            }
        )
    return (
        path,
        pin,
        {"schema_version": 1, "feed_identity": identity, "factors": entries},
    )


def _synthetic_frame(sources):
    donor = read_populace_us_donor(sources.donor, **donor_pin(sources.donor))
    tables, _ = split_for_transport(donor.tables)
    for entity, table in tables.items():
        for name in table:
            concept = concept_for_column(entity, name)
            if concept is not None and concept.dtype == "str":
                table[name] = pd.array(table[name], dtype=dtype_for_token("string"))
    person, household = tables["person"], tables["household"]
    family, membership = build_benefit_units(person, household, UNIT_RULE)
    person[UNIT_RULE.membership_column] = membership
    person["take_up_seed"] = derive_transport_seed(
        donor.source_person_ids, "synthetic:composed:takeup"
    )
    household["donor_support_stratum"] = pd.array(
        donor.support_strata, dtype=dtype_for_token("string")
    )
    tables["family"] = family
    weights = pd.Series(donor.weights, index=household["household_id"])
    scale = 120 / person["person_household_id"].map(weights).sum()
    return Frame(
        tables,
        NZ_SCHEMA,
        {"household": Weights(donor.weights * scale, WeightKind.DESIGN)},
    )
