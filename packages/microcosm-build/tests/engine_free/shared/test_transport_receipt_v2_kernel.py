"""Receipt surface validation and the reserved declared-unit dispatch seam."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.transport.artifact_types import TARGET_SURFACE_TYPE
from microcosm.build.transport.column_kernels import TAKEUP_ASSIGN
from microcosm.build.transport.receipt_contract import receipt_dispatch_tables
from microcosm.build.transport.target_kernels import compile_target_surface
from microcosm.graph import ArtifactValue, KernelContext, Node, NumericScope, Owned
from test_support.microcosm_build.transport_graph import (
    canonical_text,
    reference_document,
    reference_row,
    toy_fact,
    write_facts,
)
from test_support.microcosm_build.transport_population import digest


def _context(
    tmp_path,
    *,
    filter_name="adult",
    entity="person",
    measure="receives_alpha",
    output="receives_alpha",
    unit="count",
):
    contract = {
        "contract_version": 2,
        "seed_column": "seed",
        "default_assignment_profile": "central",
        "assignment_knob": "assignment",
        "assignment_profiles": {"central": {"draw_key_suffix": ""}},
        "programs": [
            {
                "program": "alpha",
                "output": "receives_alpha",
                "judgment_columns": ["eligible"],
                "judgment_reduction": "any",
                "payment_column": "payment",
                "strata": [{"target": "alpha_target", "filter": "adult"}],
                "outside_strata": "not_assigned",
            }
        ],
        "exclusion_groups": [],
        "unmodelled_programs": ["unsupported"],
    }
    bindings = {
        "schema_version": 1,
        "required_inputs": {"person": ["age"], "household": ["size"]},
        "input_domains": {
            "person": {"age": {"kind": "nonnegative_integer"}},
            "household": {"size": {"kind": "nonnegative_integer"}},
        },
        "columns": [
            {
                "entity": "person",
                "name": "adult",
                "predicates": [{"column": "age", "operator": "ge", "value": 18}],
            }
        ],
    }
    document = reference_document(
        [
            reference_row(
                "alpha_target", entity=entity, measure=measure, filter_=filter_name
            )
        ]
    )
    fact = toy_fact("alpha_target", 1, entity=entity)
    fact["observed_measure"]["unit"] = unit
    facts = write_facts(tmp_path / "facts.jsonl", [fact])
    surface, _ = compile_target_surface(
        facts, document, country="xx", references_sha256=digest(document)
    )
    params = {
        "receipt_contract": canonical_text(contract),
        "receipt_contract_sha256": digest(contract),
        "calibration_bindings": canonical_text(bindings),
        "calibration_bindings_sha256": digest(bindings),
        "profile_selection": canonical_text({"assignment": "central"}),
    }
    node = Node(
        "receipt",
        TAKEUP_ASSIGN.ref,
        population="open",
        params=params,
        outputs=(Owned("person", output, "bool"),),
    )
    return KernelContext(
        node=node,
        tables={
            "person": pd.DataFrame(
                {
                    "person_id": [1],
                    "age": [20],
                    "seed": [0.2],
                    "eligible": [1],
                    "payment": [1],
                }
            ),
            "household": pd.DataFrame({"household_id": [1], "size": [1]}),
        },
        weights={"person": SimpleNamespace(values=np.ones(1))},
        strata=pd.DataFrame(),
        params=node.params,
        rng=np.random.default_rng(1),
        artifacts={
            "surface": ArtifactValue(
                payload=surface,
                type=TARGET_SURFACE_TYPE,
                key="0" * 64,
                producer_key="0" * 64,
                numerics=NumericScope(),
            )
        },
    )


def test_v2_kernel_checks_declared_stratum_filter(tmp_path):
    with pytest.raises(ValueError, match="declared stratum filter"):
        TAKEUP_ASSIGN.run(_context(tmp_path, filter_name="different"))


@pytest.mark.parametrize(
    "field,value", [("entity", "household"), ("measure", "other"), ("unit", "currency")]
)
def test_v2_kernel_requires_declared_count_surface(tmp_path, field, value):
    with pytest.raises(ValueError, match="person count"):
        TAKEUP_ASSIGN.run(_context(tmp_path, **{field: value}))


@pytest.mark.parametrize("output", ["unsupported", "receives_unknown"])
def test_unmodelled_or_unknown_output_never_defaults_false(tmp_path, output):
    with pytest.raises(ValueError, match="unmodelled or unknown receipt outputs"):
        TAKEUP_ASSIGN.run(_context(tmp_path, output=output))


def test_v2_kernel_validates_full_bindings_before_preparation(tmp_path):
    context = _context(tmp_path)
    result = TAKEUP_ASSIGN.run(context)
    assert result.columns["person", "receives_alpha"].tolist() == [True]
    assert result.receipt["programs"]["receives_alpha"]["strata"][0]["numerical_check"][
        "passed"
    ]
    context.tables["household"].loc[0, "size"] = -1
    with pytest.raises(ValueError, match="Calibration input household.size"):
        TAKEUP_ASSIGN.run(context)


@pytest.mark.parametrize(
    "field", ["entitlement_entity", "payee", "draw_unit", "projection"]
)
def test_dispatch_reads_each_reserved_field_with_registered_fake_entity(field):
    person, fake = object(), object()
    row = dict.fromkeys(
        ["entitlement_entity", "payee", "draw_unit", "projection"], "person"
    )
    row[field] = "registered_fake"
    result = receipt_dispatch_tables({"person": person, "registered_fake": fake}, row)
    assert result[field] is fake
    assert all(value is person for name, value in result.items() if name != field)


def test_v2_kernel_resolves_profile_from_declared_knob(tmp_path):
    context = _context(tmp_path)
    params = copy.deepcopy(dict(context.params))
    params["profile_selection"] = json.dumps(
        {"unrelated": "central"}, sort_keys=True, separators=(",", ":")
    )
    context = KernelContext(
        node=context.node,
        tables=context.tables,
        weights=context.weights,
        strata=context.strata,
        params=params,
        rng=context.rng,
        artifacts=context.artifacts,
    )
    with pytest.raises(ValueError, match="assignment_knob"):
        TAKEUP_ASSIGN.run(context)
