"""Reusable real-engine comparison; policy data lives in test fixtures.

``compare_msd_benefit_rates(engine, engine_ref=...)`` accepts the pinned real
Axiom adapter as well as the test double. It executes the declared affine
solver, certified quantization, integer-cent bridge and annual transfers.
It never constructs an engine or reads a donor or Chronicle artifact.
"""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pandas as pd

from microcosm.build.transport.compose import _param, load_transport_spec
from microcosm.build.transport.money_kernels import (
    MONEY_ANNUAL_TRANSFERS,
    MONEY_MINOR_UNITS,
    weekly_gross_cents_for_reporting,
)
from microcosm.frame import ExportContract, Frame, VariableMetadata, WeightKind, Weights
from microcosm.frame.adapters.axiom import NZ_SCHEMA
from microcosm.frame.rules_kernels import SimulateSolveZeroKernel
from microcosm.graph import Node, Owned, Slice
from test_support.microcosm_frame.kernels import _context
from test_support.paths import paths_for


def schedule_fixture():
    path = (
        paths_for("microcosm-build").tests / "fixtures/nz/schedule_1_test_fixture.json"
    )
    return json.loads(path.read_text())["brackets"]


def exact_tax(gross, brackets):
    """Evaluate test-fixture piecewise data using Fraction only."""
    total = Fraction(0)
    for index, bracket in enumerate(brackets):
        lower = Fraction(bracket["lower"])
        upper = (
            Fraction(brackets[index + 1]["lower"])
            if index + 1 < len(brackets)
            else gross
        )
        total += max(Fraction(0), min(gross, upper) - lower) * Fraction(bracket["rate"])
    return total


def exact_inverse(net_annual, brackets):
    """Independent exact inverse used only as a differential test oracle."""
    tax_at_lower = Fraction(0)
    for index, bracket in enumerate(brackets):
        lower, rate = Fraction(bracket["lower"]), Fraction(bracket["rate"])
        candidate = (net_annual + tax_at_lower - rate * lower) / (1 - rate)
        if index + 1 == len(brackets) or candidate <= Fraction(
            brackets[index + 1]["lower"]
        ):
            return candidate
        tax_at_lower += (Fraction(brackets[index + 1]["lower"]) - lower) * rate
    raise AssertionError("Unreachable fixture inverse")


class ScheduleFixtureEngine:
    """Engine protocol test double; refuses metadata for its solver input."""

    def __init__(self, brackets=None):
        self.brackets = schedule_fixture() if brackets is None else brackets
        self.calls = 0

    def entity_schema(self):
        return NZ_SCHEMA

    def variables(self):
        return ("taxable_income",)

    def variable_metadata(self, name):
        if name != "individual_income_tax_before_credits":
            raise ValueError("Input variables have no engine output metadata")
        return VariableMetadata(
            name=name, entity="person", dtype="float", period="year"
        )

    def export_contract(self):
        return ExportContract.empty()

    def write_dataset(self, bundle, path, period):
        return None

    def materialize(self, bundle, variables, period):
        assert tuple(variables) == ("individual_income_tax_before_credits",)
        assert period == "2026-27"
        self.calls += 1
        return {
            variables[0]: np.asarray(
                [
                    float(exact_tax(Fraction(str(value)), self.brackets))
                    for value in bundle.table("person")["taxable_income"]
                ]
            )
        }


def _node(row, spec, engine_ref):
    return Node(
        row["id"],
        row["kernel"],
        inputs=tuple(
            Slice(item["entity"], tuple(item["columns"])) for item in row["inputs"]
        ),
        outputs=tuple(
            Owned(item["entity"], item["column"], item["dtype"])
            for item in row["outputs"]
        ),
        params={
            key: _param(
                spec,
                value,
                {
                    "income_tax_schedule_1": engine_ref,
                    "main_benefits_rates": engine_ref,
                },
            )
            for key, value in row["params"].items()
        },
    )


def compare_msd_benefit_rates(
    engine, *, engine_ref="msd-schedule-fixture", declaration=None
):
    """Return the 14 published/model comparisons through the real A–C path.

    The real-engine check supplies its pinned Schedule 1 adapter and reference.
    This helper returns evidence; callers assert the required discrepancy bound.
    """
    spec = load_transport_spec("nz")
    resource = spec["resources"]["msd_benefit_rates_20260401"]
    cases = [
        row
        for row in resource["weekly_main_benefits"]
        if row["individual_validation_point"]
    ]
    declaration = (
        spec["resources"]["benefit_gross_up"] if declaration is None else declaration
    )
    programmes = (
        "jobseeker_support",
        "sole_parent_support",
        "supported_living_payment",
    )
    ids = np.arange(1, len(cases) + 1, dtype=np.int64)
    person = pd.DataFrame(
        {"person_id": ids, "person_household_id": ids, "person_family_id": ids}
    )
    for programme in programmes:
        person[f"{programme}_net_weekly_payment"] = [
            float(row["after_tax_at_m"]) if row["programme"] == programme else 0
            for row in cases
        ]
        person[f"receives_{programme}"] = [
            row["programme"] == programme for row in cases
        ]
    person["receives_nz_super"] = False
    person["nz_super_weekly_amount_before_tax"] = 0.0
    frame = Frame(
        {
            "person": person,
            "household": pd.DataFrame({"household_id": ids}),
            "family": pd.DataFrame({"family_id": ids}),
        },
        NZ_SCHEMA,
        {"household": Weights(np.ones(len(ids)), WeightKind.DESIGN)},
    )
    solver = SimulateSolveZeroKernel({engine_ref: engine})
    kernels = {
        solver.ref: solver,
        MONEY_MINOR_UNITS.ref: MONEY_MINOR_UNITS,
        MONEY_ANNUAL_TRANSFERS.ref: MONEY_ANNUAL_TRANSFERS,
    }
    receipts = {}
    for row in declaration["nodes"]:
        if row["kernel"] not in kernels:
            continue  # Published nets stand in for the rates engine in this check.
        node = _node(row, spec, engine_ref)
        result = kernels[node.kernel].run(
            _context(frame, node, weighted_entities=("household",))
        )
        for (entity, column), values in result.columns.items():
            frame.table(entity)[column] = values.to_numpy()
        receipts[node.id] = result.receipt
    comparison = []
    for index, case in enumerate(cases):
        programme = case["programme"]
        annual = int(
            frame.table("person")[f"{programme}_annual_gross_transfer_cents"].iloc[
                index
            ]
        )
        weekly = weekly_gross_cents_for_reporting(annual, 52)
        model = f"{weekly // 100}.{weekly % 100:02d}"
        receipt = receipts[f"nz.gross_up.{programme}"]
        comparison.append(
            {
                **case,
                "annual_gross_cents": annual,
                "model_gross": model,
                "difference": str(Fraction(model) - Fraction(case["before_tax"])),
                "receipt": receipt,
            }
        )
    return comparison
