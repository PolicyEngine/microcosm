"""Gap and band invariants over synthetic populations and published toy facts.

The weighted-gap path is compared with both a plain family-level sum and the
existing target compiler/matrix estimator. Hypothesis checks exact subtraction,
determinism, ordered labelled bounds, and deterministic tie breaking. Values
and annualisation factors in this file are invented test inputs.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, Phase, example, given, settings
from hypothesis import strategies as st

from microcosm.build.transport.takeup_kernels import (
    BANDS_TYPE,
    GAP_TYPE,
    TAKEUP_BANDS,
    TAKEUP_GAP,
    decode_bands,
    decode_gap,
    register_takeup_kernels,
)
from microcosm.build.transport.target_kernels import compile_target_surface
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.frame import Frame
from microcosm.graph import (
    ArtifactOutput,
    ArtifactType,
    ArtifactValue,
    Kernel,
    KernelContext,
    KernelRegistry,
    Node,
    NumericScope,
    Slice,
)
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.transport_graph import (
    COUNTRY,
    ENTITIES,
    canonical_text,
    reference_document,
    reference_row,
    toy_fact,
    toy_frame,
    write_facts,
)

_SHA = "3" * 64
_KEY = "a" * 64
_FACT_NAMES = ("annual_support", "recipient_units")
_FACTOR = 52
_PHASES = tuple(phase for phase in Phase if phase != Phase.explain)


def _frame(values=None, weights=None) -> Frame:
    original = toy_frame(weights=weights)
    tables = {entity: original.table(entity).copy(deep=True) for entity in ENTITIES}
    if values is not None:
        tables["family"]["family_support"] = np.asarray(values, dtype=np.float64)
    tables["family"]["support_recipient"] = tables["family"]["family_support"] > 0
    return Frame(
        tables,
        original.schema,
        {"household": original.weights_for("household")},
        original.strata.copy(),
        metadata=dict(original.metadata),
    )


def _references() -> dict:
    amount = reference_row(
        _FACT_NAMES[0],
        entity="family",
        measure="family_support",
        filter_="support_recipient",
    )
    amount["metadata"].update(
        {
            "decile": "toy_decile",
            "family_type": "toy_type",
            "receipt_status": "toy_receipt",
        }
    )
    count = reference_row(_FACT_NAMES[1], entity="family", measure="support_recipient")
    return reference_document([amount, count])


def _facts(tmp_path, values=(600, 5)):
    return write_facts(
        tmp_path / "facts.jsonl",
        [
            toy_fact(name, value, entity="family")
            for name, value in zip(_FACT_NAMES, values, strict=True)
        ],
    )


def _gap_node(*, scenario="S0", factors=None, references=None, **params) -> Node:
    return Node(
        "gap",
        TAKEUP_GAP.ref,
        population="open",
        inputs=(
            Slice("person", ("age",)),
            Slice("household", ("rent",)),
            Slice("family", ("family_support", "support_recipient")),
        ),
        sources=("facts",),
        params={
            "country": COUNTRY,
            "references": canonical_text(
                _references() if references is None else references
            ),
            "references_sha256": _SHA,
            "entities": ENTITIES,
            "weight_entity": "family",
            "annualisation_factors": canonical_text(
                {_FACT_NAMES[0]: _FACTOR, _FACT_NAMES[1]: 1}
                if factors is None
                else factors
            ),
            "annualisation_sha256": _SHA,
            "scenario": scenario,
            **params,
        },
        artifact_outputs=(ArtifactOutput("gap", GAP_TYPE),),
    )


def _context(node, *, frame=None, facts=None, artifacts=None) -> KernelContext:
    return KernelContext(
        node=node,
        tables={}
        if frame is None
        else {entity: frame.table(entity).copy() for entity in ENTITIES},
        weights={} if frame is None else {"family": frame.resolve_weights("family")},
        strata=pd.Series([], dtype=object, name="stratum")
        if frame is None
        else frame.strata.copy(),
        params=node.params,
        rng=np.random.default_rng(0),
        sources={} if facts is None else {"facts": facts},
        artifacts={} if artifacts is None else artifacts,
    )


def _gap(
    tmp_path, *, values=None, weights=None, actual=(600, 5), scenario="S0"
) -> bytes:
    return TAKEUP_GAP.run(
        _context(
            _gap_node(scenario=scenario),
            frame=_frame(values, weights),
            facts=_facts(tmp_path, actual),
        )
    ).artifacts["gap"]


def _value(payload, kind=GAP_TYPE) -> ArtifactValue:
    return ArtifactValue(
        payload=payload, type=kind, key=_KEY, producer_key=_KEY, numerics=NumericScope()
    )


def _bands_node(labels, *, baseline="S0", **params) -> Node:
    return Node(
        "bands",
        TAKEUP_BANDS.ref,
        population="open",
        params={
            "baseline_scenario": baseline,
            "artifact_scenarios": canonical_text(
                {f"gap_{index}": label for index, label in enumerate(labels)}
            ),
            **params,
        },
        artifact_outputs=(ArtifactOutput("bands", BANDS_TYPE),),
    )


def _bands(payloads, *, baseline="S0") -> bytes:
    labels = [decode_gap(payload)["scenario"] for payload in payloads]
    node = _bands_node(labels, baseline=baseline)
    return TAKEUP_BANDS.run(
        _context(
            node,
            artifacts={
                f"gap_{index}": _value(payload)
                for index, payload in enumerate(payloads)
            },
        )
    ).artifacts["bands"]


def test_neutral_kernel_contracts_and_idempotent_registration() -> None:
    registry = KernelRegistry()
    for _ in range(2):
        register_takeup_kernels(registry)
    for kernel in (TAKEUP_GAP, TAKEUP_BANDS):
        assert isinstance(kernel, Kernel)
        assert "nz" not in kernel.ref
        assert kernel.implementation_hash() == kernel.implementation_hash()
        assert kernel.capabilities.consumes_se is False


@settings(
    max_examples=24,
    deadline=None,
    phases=_PHASES,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    values=st.lists(
        st.floats(min_value=0, max_value=2000, allow_nan=False, allow_infinity=False),
        min_size=8,
        max_size=8,
    ),
    weights=st.lists(
        st.floats(min_value=1, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=6,
        max_size=6,
    ),
    actual=st.floats(min_value=0, max_value=1e8, allow_nan=False, allow_infinity=False),
)
@example(values=[0] * 8, weights=[1] * 6, actual=0)
@example(values=[1] * 8, weights=[1] * 6, actual=1)
def test_weighted_total_is_sum_family_weight_times_annualised_support_and_gap_exact(
    tmp_path, values, weights, actual
) -> None:
    frame = _frame(values, weights)
    payload = _gap(tmp_path, values=values, weights=weights, actual=(actual, 5))
    rows = {row["name"]: row for row in decode_gap(payload)["gaps"]}
    expected = float(
        np.sum(frame.resolve_weights("family").values * (np.asarray(values) * _FACTOR))
    )
    assert rows["annual_support"]["E"] == expected
    assert rows["annual_support"]["A"] == actual
    assert rows["annual_support"]["gap"] == expected - actual
    count = float(
        np.sum(frame.resolve_weights("family").values * (np.asarray(values) > 0))
    )
    assert rows["recipient_units"]["E"] == count
    assert all(row["gap"] == row["E"] - row["A"] for row in rows.values())


@settings(
    max_examples=12,
    deadline=None,
    phases=_PHASES,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    values=st.lists(st.integers(min_value=0, max_value=2000), min_size=8, max_size=8)
)
def test_gap_differential_matches_existing_target_compiler_and_matrix(
    tmp_path, values
) -> None:
    frame = _frame(values)
    facts = _facts(tmp_path)
    _, registry = compile_target_surface(
        facts, _references(), country=COUNTRY, references_sha256=_SHA
    )
    problem = build_constraint_matrix(frame, registry.to_target_set(), "family")
    estimates = problem.estimates(frame.resolve_weights("family").values)
    rows = decode_gap(_gap(tmp_path, values=values))["gaps"]
    for row, spec, estimate in zip(rows, registry.specs, estimates, strict=True):
        assert row["A"] == spec.value
        assert row["E"] == pytest.approx(
            estimate * row["annualisation_factor"], rel=1e-12
        )
        assert row["reference"] == spec.name
        assert row["ledger_fact_key"] == spec.metadata["ledger_fact_key"]


@settings(
    max_examples=12,
    deadline=None,
    phases=_PHASES,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    values=st.lists(st.integers(min_value=0, max_value=2000), min_size=8, max_size=8)
)
def test_gap_and_bands_deterministic_and_path_independent(tmp_path, values) -> None:
    payload = _gap(tmp_path, values=values)
    assert payload == _gap(tmp_path, values=values)
    assert _bands([payload]) == _bands([payload])
    node = _gap_node()
    moved = tmp_path / "moved.jsonl"
    moved.write_bytes(_facts(tmp_path).read_bytes())
    result = TAKEUP_GAP.run(_context(node, frame=_frame(values), facts=moved))
    assert result.artifacts["gap"] == payload
    assert str(tmp_path) not in payload.decode()


def test_reference_metadata_carries_declared_group_labels_and_annualisation(
    tmp_path,
) -> None:
    document = decode_gap(_gap(tmp_path))
    assert document["scenario"] == "S0"
    assert document["weight_entity"] == "family"
    row = document["gaps"][0]
    assert row["annualisation_factor"] == _FACTOR
    assert row["metadata"] == {
        "activation_status": "active",
        "decile": "toy_decile",
        "family_type": "toy_type",
        "receipt_status": "toy_receipt",
    }


@settings(
    max_examples=12,
    deadline=None,
    phases=_PHASES,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    central=st.integers(min_value=0, max_value=2000),
    alternate=st.integers(min_value=0, max_value=2000),
    other=st.integers(min_value=0, max_value=2000),
)
def test_bands_ordered_with_each_bound_naming_its_scenario(
    tmp_path, central, alternate, other
) -> None:
    payloads = [
        _gap(tmp_path, values=[value] * 8, scenario=label)
        for label, value in (("S0", central), ("S1", alternate), ("V1", other))
    ]
    bands = decode_bands(_bands(payloads))
    source_rows = {
        decode_gap(payload)["scenario"]: {
            (row["name"], row["period"]): row for row in decode_gap(payload)["gaps"]
        }
        for payload in payloads
    }
    for row in bands["bands"]:
        assert row["low"]["value"] <= row["baseline"]["value"] <= row["high"]["value"]
        assert row["baseline"]["scenario"] == "S0"
        for bound in ("low", "baseline", "high"):
            selected = row[bound]
            assert (
                selected["value"]
                == source_rows[selected["scenario"]][row["name"], row["period"]][
                    row["metric"]
                ]
            )
        values = [
            source_rows[label][row["name"], row["period"]][row["metric"]]
            for label in source_rows
        ]
        assert row["low"]["value"] == min(values)
        assert row["high"]["value"] == max(values)
    assert _bands(payloads) == _bands(list(reversed(payloads)))


@settings(
    max_examples=8,
    deadline=None,
    phases=_PHASES,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(value=st.integers(min_value=0, max_value=2000))
def test_equal_bounds_choose_first_scenario_label_deterministically(
    tmp_path, value
) -> None:
    payloads = [
        _gap(tmp_path, values=[value] * 8, scenario=label) for label in ("Z", "S0", "A")
    ]
    for row in decode_bands(_bands(payloads))["bands"]:
        assert row["low"]["scenario"] == "A"
        assert row["high"]["scenario"] == "A"


@pytest.mark.parametrize(
    "factors",
    [
        {},
        {"annual_support": 52},
        {"annual_support": 52, "recipient_units": 1, "extra": 1},
    ],
)
def test_gap_requires_factor_for_exactly_every_reference(tmp_path, factors) -> None:
    with pytest.raises(ValueError, match="cover exactly"):
        TAKEUP_GAP.run(
            _context(_gap_node(factors=factors), frame=_frame(), facts=_facts(tmp_path))
        )


@pytest.mark.parametrize("factor", [0, -1, True, "52"])
def test_gap_rejects_invalid_annualisation(tmp_path, factor) -> None:
    with pytest.raises((ValueError, TypeError), match="number|positive"):
        TAKEUP_GAP.run(
            _context(
                _gap_node(factors={"annual_support": factor, "recipient_units": 1}),
                frame=_frame(),
                facts=_facts(tmp_path),
            )
        )


def test_gap_refuses_uncompiled_placeholder_or_missing_column(tmp_path) -> None:
    references = _references()
    references["target_references"][0]["metadata"]["activation_status"] = (
        "awaiting_evidence"
    )
    with pytest.raises(ValueError, match="placeholder"):
        TAKEUP_GAP.run(
            _context(
                _gap_node(references=references), frame=_frame(), facts=_facts(tmp_path)
            )
        )
    references = _references()
    references["target_references"][0]["measure"] = "missing_support"
    with pytest.raises(ValueError, match="uncompilable|No targets"):
        TAKEUP_GAP.run(
            _context(
                _gap_node(references=references), frame=_frame(), facts=_facts(tmp_path)
            )
        )


def test_gap_requires_honest_params_typed_outputs_and_canonical_documents(
    tmp_path,
) -> None:
    node = _gap_node(extra=True)
    with pytest.raises(ValueError, match="unknown parameter"):
        TAKEUP_GAP.run(_context(node, frame=_frame(), facts=_facts(tmp_path)))
    node = replace(_gap_node(), artifact_outputs=(ArtifactOutput("gap", BANDS_TYPE),))
    with pytest.raises(ValueError, match="typed artifact output"):
        TAKEUP_GAP.run(_context(node, frame=_frame(), facts=_facts(tmp_path)))
    node = _gap_node()
    node = replace(
        node, params={**node.params, "references": json.dumps(_references())}
    )
    with pytest.raises(ValueError, match="canonical JSON"):
        TAKEUP_GAP.run(_context(node, frame=_frame(), facts=_facts(tmp_path)))


def test_gap_decoder_refuses_wrong_subtraction_and_noncanonical_bytes(tmp_path) -> None:
    payload = _gap(tmp_path)
    document = json.loads(payload)
    document["gaps"][0]["gap"] += 1
    with pytest.raises(ValueError, match="minus A exactly"):
        decode_gap(canonical_json(document))
    with pytest.raises(ValueError, match="noncanonical"):
        decode_gap(json.dumps(json.loads(payload)).encode())


def test_bands_refuses_missing_central_scenario_duplicates_and_wrong_artifact_type(
    tmp_path,
) -> None:
    payload = _gap(tmp_path)
    node = _bands_node(["S1"])
    with pytest.raises(ValueError, match="central scenario"):
        TAKEUP_BANDS.run(_context(node, artifacts={"gap_0": _value(payload)}))
    node = _bands_node(["S0", "S0"])
    with pytest.raises(ValueError, match="distinct"):
        TAKEUP_BANDS.run(
            _context(
                node, artifacts={"gap_0": _value(payload), "gap_1": _value(payload)}
            )
        )
    node = _bands_node(["S0"])
    with pytest.raises(ValueError, match="needs gap artifacts"):
        TAKEUP_BANDS.run(
            _context(
                node, artifacts={"gap_0": _value(payload, ArtifactType("wrong", 1))}
            )
        )


def test_bands_refuses_mismatched_scenario_country_cells_and_semantics(
    tmp_path,
) -> None:
    central = _gap(tmp_path)
    alternate = decode_gap(_gap(tmp_path, scenario="S1"))
    node = _bands_node(["S0", "S1"])
    for field, value, message in (
        ("scenario", "S2", "declared scenario"),
        ("country", "yy", "same country"),
        ("gaps", alternate["gaps"][:1], "table cells"),
    ):
        changed = {**alternate, field: value}
        with pytest.raises(ValueError, match=message):
            TAKEUP_BANDS.run(
                _context(
                    node,
                    artifacts={
                        "gap_0": _value(central),
                        "gap_1": _value(canonical_json(changed)),
                    },
                )
            )
    changed = copy.deepcopy(alternate)
    changed["gaps"][0]["metadata"]["decile"] = "different"
    with pytest.raises(ValueError, match="semantics"):
        _bands([central, canonical_json(changed)])


def test_bands_decoder_refuses_unlabelled_reversed_or_unmatched_bounds(
    tmp_path,
) -> None:
    payload = _bands([_gap(tmp_path)])
    for mutation, message in (
        (lambda row: row["low"].pop("scenario"), "name its scenario"),
        (lambda row: row["low"].update(value=row["high"]["value"] + 1), "ordered"),
        (lambda row: row["high"].update(scenario="absent"), "undeclared scenario"),
    ):
        changed = json.loads(payload)
        mutation(changed["bands"][0])
        with pytest.raises(ValueError, match=message):
            decode_bands(canonical_json(changed))


def test_pinned_facts_hash_is_checked(tmp_path) -> None:
    with pytest.raises(ValueError, match="pinned hash"):
        TAKEUP_GAP.run(
            _context(
                _gap_node(facts_sha256="0" * 64), frame=_frame(), facts=_facts(tmp_path)
            )
        )


def test_annualisation_resource_digest_must_be_sha256(tmp_path) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        TAKEUP_GAP.run(
            _context(
                _gap_node(annualisation_sha256="unbound"),
                frame=_frame(),
                facts=_facts(tmp_path),
            )
        )
