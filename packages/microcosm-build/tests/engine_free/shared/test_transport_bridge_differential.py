"""Refusals for declared differentials, using invented engine outputs only."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from microcosm.frame import EntitySchema, ExportContract, VariableMetadata
from microcosm.frame.rules_kernels import SimulateCounterfactualKernel
from microcosm.graph import KernelRegistry
from tools import transport_bridge_differential as driver


class InventedRatesEngine:
    """A tiny protocol engine; these rates are invented test data."""

    def __init__(self):
        self.calls = 0

    def variable_metadata(self, name):
        return VariableMetadata(name, "person", "float", "year")

    def variables(self):
        return ("income", "rate")

    def entity_schema(self):
        return EntitySchema(group_entities=("family",))

    def materialize(self, bundle, variables, period):
        self.calls += 1
        person = bundle.table("person")
        values = np.maximum(person["rate"] - person["income"], 0).to_numpy()
        return {variable: values for variable in variables}

    def export_contract(self):
        return ExportContract.empty()

    def write_dataset(self, bundle, path, period):
        return None


@pytest.fixture
def differential_case(monkeypatch):
    engine = InventedRatesEngine()
    kernel = SimulateCounterfactualKernel({"prepared-rates": engine})
    registry = KernelRegistry()
    registry.register(kernel)
    prepared = SimpleNamespace(
        schema=engine.entity_schema(),
        kernels=registry,
        engine_refs={"rates": "prepared-rates"},
    )
    preparation = []

    def prepare(*args, **kwargs):
        preparation.append("registry")
        return prepared

    def unit_rule(data):
        preparation.append("unit_rule")
        return object()

    monkeypatch.setattr(driver, "build_transport_registry", prepare)
    monkeypatch.setattr(driver.BenefitUnitRule, "from_dict", unit_rule)
    case = {
        "unit_rule": {},
        "rules_bindings": {},
        "tables": {
            "person": [
                {
                    "person_id": 1,
                    "person_family_id": 100,
                    "income": 10,
                    "rate": 50,
                },
                {
                    "person_id": 2,
                    "person_family_id": 100,
                    "income": 11,
                    "rate": 51,
                },
            ],
            "family": [{"family_id": 100}],
        },
        "weights": {"family": {"kind": "design", "values": [1]}},
        "calls": [
            {
                "id": "bridge",
                "kernel": kernel.ref,
                "inputs": {"person": ["income", "rate"]},
                "outputs": [
                    {"entity": "family", "column": "bridge", "dtype": "float64"}
                ],
                "params": {
                    "engine_ref": "rates",
                    "period": "2026-27",
                    "variables": ["weekly_benefit"],
                    "input_overrides": [
                        {"entity": "person", "column": "income", "value": 0}
                    ],
                    "output_coefficients": [
                        {
                            "variable": "weekly_benefit",
                            "coefficient": 1,
                            "entity": "family",
                            "column": "bridge",
                            "aggregation": "sum",
                        }
                    ],
                    "resource_sha256": "a" * 64,
                },
            }
        ],
        "comparisons": [
            {
                "call": "bridge",
                "entity": "family",
                "column": "bridge",
                "oracle_pointer": "/weekly",
                "tolerance": 0,
            }
        ],
    }
    return case, engine, preparation


@pytest.mark.parametrize("call_id", ["", None], ids=["empty", "non-string"])
def test_differential_refuses_empty_call_id_before_preparation(
    differential_case, call_id
):
    case, engine, preparation = differential_case
    case["calls"][0]["id"] = call_id
    with pytest.raises(ValueError, match="non-empty call id"):
        driver.differential(case, {"weekly": 101}, Path("unused"))
    assert preparation == []
    assert engine.calls == 0


def test_differential_refuses_duplicate_bridge_id_with_different_coefficients(
    differential_case,
):
    case, engine, preparation = differential_case
    replacement = deepcopy(case["calls"][0])
    replacement["params"]["output_coefficients"][0]["coefficient"] = 2
    case["calls"].append(replacement)
    # The old driver overwrote the coefficient-1 result and falsely passed 202.
    with pytest.raises(ValueError, match="duplicate call id.*bridge"):
        driver.differential(case, {"weekly": 202}, Path("unused"))
    assert preparation == []
    assert engine.calls == 0


@pytest.mark.parametrize("within_call", [False, True], ids=["two-calls", "one-call"])
def test_differential_refuses_multiple_output_owners_before_preparation(
    differential_case, within_call
):
    case, engine, preparation = differential_case
    if within_call:
        case["calls"][0]["outputs"] *= 2
    else:
        replacement = deepcopy(case["calls"][0])
        replacement["id"] = "other-bridge"
        case["calls"].append(replacement)
    with pytest.raises(ValueError, match="output coordinate.*family.*bridge"):
        driver.differential(case, {"weekly": 101}, Path("unused"))
    assert preparation == []
    assert engine.calls == 0


@pytest.mark.parametrize(
    "reference,match",
    [
        ({"call": "absent"}, "comparison names missing call"),
        ({"column": "absent"}, "comparison names missing output"),
        ({"entity": "person"}, "comparison names missing output"),
    ],
    ids=["missing-call", "missing-column", "wrong-entity"],
)
def test_differential_refuses_missing_comparison_references_before_preparation(
    differential_case, reference, match
):
    case, engine, preparation = differential_case
    case["comparisons"][0].update(reference)
    with pytest.raises(ValueError, match=match):
        driver.differential(case, {"weekly": 101}, Path("unused"))
    assert preparation == []
    assert engine.calls == 0


def test_differential_refuses_comparison_output_owned_by_another_call(
    differential_case,
):
    case, engine, preparation = differential_case
    other = deepcopy(case["calls"][0])
    other["id"] = "other-bridge"
    other["outputs"][0]["column"] = "other-output"
    other["params"]["output_coefficients"][0]["column"] = "other-output"
    case["calls"].append(other)
    case["comparisons"][0]["column"] = "other-output"
    with pytest.raises(ValueError, match="comparison names missing output"):
        driver.differential(case, {"weekly": 101}, Path("unused"))
    assert preparation == []
    assert engine.calls == 0


def test_differential_requires_comparisons_before_preparation(differential_case):
    case, engine, preparation = differential_case
    case["comparisons"] = []
    with pytest.raises(ValueError, match="must declare comparisons"):
        driver.differential(case, {}, Path("unused"))
    assert preparation == []
    assert engine.calls == 0


@pytest.mark.parametrize("token", ["-1", "01", "+1", " 1", "1 ", "-", "", "\u0661"])
def test_oracle_pointer_refuses_noncanonical_array_indices(token):
    with pytest.raises(ValueError, match="canonical non-negative array index"):
        driver._pointer({"items": [10, 3]}, f"/items/{token}")


@pytest.mark.parametrize("token", ["2", "123456789012345678901234567890"])
def test_oracle_pointer_refuses_out_of_bounds_array_indices(token):
    with pytest.raises(ValueError, match="array index.*out of bounds"):
        driver._pointer({"items": [10, 3]}, f"/items/{token}")


@pytest.mark.parametrize("token", ["~", "~2", "name~x", "~01~"])
def test_oracle_pointer_refuses_invalid_escape_sequences(token):
    with pytest.raises(ValueError, match="invalid JSON pointer escape"):
        driver._pointer({token: 7}, f"/{token}")


@pytest.mark.parametrize(
    "pointer,expected",
    [
        ("", {"items": [10, 3], "a/b": {"~value": [7]}, "~1": 9, "01": 8, "": 5}),
        ("/", 5),
        ("/items/0", 10),
        ("/items/1", 3),
        ("/a~1b/~0value/0", 7),
        ("/~01", 9),
        ("/01", 8),
    ],
    ids=[
        "document",
        "empty-key",
        "zero",
        "one",
        "escaped",
        "decode-once",
        "object-key",
    ],
)
def test_oracle_pointer_accepts_canonical_indices_and_rfc6901_escapes(
    pointer, expected
):
    document = {"items": [10, 3], "a/b": {"~value": [7]}, "~1": 9, "01": 8, "": 5}
    assert driver._pointer(document, pointer) == expected


@pytest.mark.parametrize(
    "document,pointer", [({}, "/missing"), ({"item": 7}, "/item/x")]
)
def test_oracle_pointer_refuses_missing_or_scalar_targets(document, pointer):
    with pytest.raises(ValueError, match="oracle_pointer does not resolve"):
        driver._pointer(document, pointer)


@pytest.mark.parametrize(
    "comparison,oracle,match",
    [
        ({"oracle_pointer": "/items/-1"}, {"items": [101]}, "canonical"),
        ({"oracle_pointer": "/absent"}, {"weekly": 101}, "does not resolve"),
        ({"tolerance": -1}, {"weekly": 101}, "finite nonnegative tolerance"),
        ({}, {"weekly": float("nan")}, "values must be finite"),
    ],
    ids=["invalid-index", "missing-pointer", "negative-tolerance", "nonfinite-oracle"],
)
def test_differential_refuses_invalid_oracle_comparison_before_preparation(
    differential_case, comparison, oracle, match
):
    case, engine, preparation = differential_case
    case["comparisons"][0].update(comparison)
    with pytest.raises(ValueError, match=match):
        driver.differential(case, oracle, Path("unused"))
    assert preparation == []
    assert engine.calls == 0
