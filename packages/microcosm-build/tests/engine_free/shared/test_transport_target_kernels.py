"""``targets.compile@1``, ``targets.problem@1`` and ``takeup.compare@1``.

Invariants (Hypothesis property tests):

- every compiled target traces to exactly one executable reference, and the
  trace is a bijection between references and targets;
- no placeholder reference compiles: a document carrying any reference whose
  ``activation_status`` is not empty or ``"active"`` refuses the node;
- the ordered problem's rows, measured at its starting weights, equal the
  direct weighted totals of the same columns under the same filters (a
  differential between the constraint-matrix path and a plain ``numpy`` dot
  product), for person-, household- and family-level targets;
- the comparison reports exactly the fact value as the comparator and the
  current-weight total as the model value.

All facts and populations are synthetic (``test_support``); nothing here is
any country's data.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.transport.artifact_types import (
    COMPARISON_TYPE,
    TARGET_SURFACE_TYPE,
)
from microcosm.build.transport.target_kernels import (
    TAKEUP_COMPARE,
    TARGETS_COMPILE,
    TARGETS_PROBLEM,
    decode_target_surface,
    parse_reference_document,
)
from microcosm.calibrate.artifacts import PROBLEM_TYPE, decode_problem
from microcosm.frame import WeightKind
from microcosm.graph import (
    ArtifactOutput,
    ArtifactValue,
    Kernel,
    KernelContext,
    Node,
    NumericScope,
    Slice,
)
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.transport_graph import (
    CALIBRATION_TARGETS,
    COUNTRY,
    ENTITIES,
    HOLDOUT_TARGETS,
    canonical_text,
    reference_document,
    reference_row,
    references_for,
    toy_facts,
    toy_frame,
    weighted_total,
    write_facts,
)

SHA = "3" * 64
_KEY = "a" * 64

ALL_TARGETS = CALIBRATION_TARGETS + HOLDOUT_TARGETS


def _compile_node(document, **params) -> Node:
    return Node(
        "compile",
        "targets.compile@1",
        population="open",
        sources=("facts",),
        params={
            "country": COUNTRY,
            "references": canonical_text(document),
            "references_sha256": SHA,
            **params,
        },
        artifact_outputs=(ArtifactOutput("surface", TARGET_SURFACE_TYPE),),
    )


def _context(node: Node, *, facts=None, frame=None, artifacts=None) -> KernelContext:
    tables = {}
    weights = {}
    strata = pd.Series([], dtype=object, name="stratum")
    if frame is not None:
        tables = {entity: frame.table(entity).copy() for entity in ENTITIES}
        weights = {"household": frame.weights_for("household")}
        strata = frame.strata.copy()
    return KernelContext(
        node=node,
        tables=tables,
        weights=weights,
        strata=strata,
        params=node.params,
        rng=np.random.default_rng(0),
        sources={} if facts is None else {"facts": facts},
        artifacts={} if artifacts is None else artifacts,
    )


def _artifact(payload: bytes, kind) -> ArtifactValue:
    return ArtifactValue(
        payload=payload, type=kind, key=_KEY, producer_key=_KEY, numerics=NumericScope()
    )


def _surface(tmp_path, targets=CALIBRATION_TARGETS, frame=None) -> bytes:
    facts = write_facts(tmp_path / "facts.jsonl", toy_facts(targets, frame))
    result = TARGETS_COMPILE.run(
        _context(_compile_node(references_for(targets)), facts=facts)
    )
    return result.artifacts["surface"]


def _problem_node() -> Node:
    return Node(
        "problem",
        "targets.problem@1",
        population="open",
        params={"entities": ENTITIES, "weight_entity": "household"},
        artifact_outputs=(ArtifactOutput("problem", PROBLEM_TYPE),),
    )


def test_kernels_declare_honest_country_neutral_contracts() -> None:
    for kernel in (TARGETS_COMPILE, TARGETS_PROBLEM, TAKEUP_COMPARE):
        assert isinstance(kernel, Kernel)
        assert "nz" not in kernel.ref
        assert kernel.capabilities.consumes_se is False
        assert kernel.implementation_hash() == kernel.implementation_hash()
    assert (
        len(
            {
                TARGETS_COMPILE.implementation_hash(),
                TARGETS_PROBLEM.implementation_hash(),
                TAKEUP_COMPARE.implementation_hash(),
            }
        )
        == 3
    )


def test_compile_emits_a_traced_surface_bound_to_facts_not_paths(tmp_path) -> None:
    payload = _surface(tmp_path)
    surface = decode_target_surface(payload)

    assert [spec.name for spec in surface.registry.specs] == [
        name for name, *_ in CALIBRATION_TARGETS
    ]
    assert [row["reference"] for row in surface.trace] == [
        name for name, *_ in CALIBRATION_TARGETS
    ]
    assert {row["activation_status"] for row in surface.trace} == {"active"}
    assert all(row["ledger_fact_key"] for row in surface.trace)
    assert all(spec.hierarchy is not None for spec in surface.registry.specs)
    assert surface.references_sha256 == SHA
    assert str(tmp_path) not in payload.decode("utf-8")
    # The same facts under another file name compile to the same bytes.
    moved = tmp_path / "elsewhere" / "renamed.jsonl"
    moved.parent.mkdir()
    moved.write_bytes((tmp_path / "facts.jsonl").read_bytes())
    again = TARGETS_COMPILE.run(
        _context(_compile_node(references_for(CALIBRATION_TARGETS)), facts=moved)
    )
    assert again.artifacts["surface"] == payload


def test_compile_values_are_the_fact_values(tmp_path) -> None:
    surface = decode_target_surface(_surface(tmp_path))
    facts = {fact["label"]: fact["value"] for fact in toy_facts(CALIBRATION_TARGETS)}
    for spec in surface.registry.specs:
        assert spec.value == facts[f"Toy fact {spec.name}"]


def test_compile_refuses_unmatched_duplicate_and_noncanonical_input(tmp_path) -> None:
    facts = write_facts(tmp_path / "facts.jsonl", toy_facts(CALIBRATION_TARGETS[:1]))
    with pytest.raises(ValueError, match="did not match a Ledger fact"):
        TARGETS_COMPILE.run(
            _context(_compile_node(references_for(CALIBRATION_TARGETS)), facts=facts)
        )

    row = reference_row("toy_adults", entity="person", measure="is_adult")
    with pytest.raises(ValueError, match="repeat the name"):
        TARGETS_COMPILE.run(
            _context(_compile_node(reference_document([row, row])), facts=facts)
        )

    node = _compile_node(references_for(CALIBRATION_TARGETS[:1]))
    spaced = Node(
        node.id,
        node.kernel,
        population=node.population,
        sources=node.sources,
        params={
            **node.params,
            "references": json.dumps(references_for(CALIBRATION_TARGETS[:1])),
        },
        artifact_outputs=node.artifact_outputs,
    )
    with pytest.raises(ValueError, match="canonical JSON"):
        TARGETS_COMPILE.run(_context(spaced, facts=facts))

    with pytest.raises(ValueError, match="pinned hash"):
        TARGETS_COMPILE.run(
            _context(
                _compile_node(
                    references_for(CALIBRATION_TARGETS[:1]), facts_sha256="0" * 64
                ),
                facts=facts,
            )
        )

    with pytest.raises(ValueError, match="SHA-256"):
        TARGETS_COMPILE.run(
            _context(
                _compile_node(
                    references_for(CALIBRATION_TARGETS[:1]), references_sha256="x"
                ),
                facts=facts,
            )
        )


def test_compile_refuses_a_node_with_misdeclared_outputs(tmp_path) -> None:
    facts = write_facts(tmp_path / "facts.jsonl", toy_facts(CALIBRATION_TARGETS))
    node = _compile_node(references_for(CALIBRATION_TARGETS))
    wrong = Node(
        node.id,
        node.kernel,
        population=node.population,
        sources=node.sources,
        params=node.params,
        artifact_outputs=(ArtifactOutput("surface", COMPARISON_TYPE),),
    )
    with pytest.raises(ValueError, match="typed outputs"):
        TARGETS_COMPILE.run(_context(wrong, facts=facts))


def test_surface_decode_refuses_a_tampered_trace_or_registry(tmp_path) -> None:
    document = json.loads(_surface(tmp_path))

    for status in ("requires_harvested_fact_reference", "placeholder"):
        placeholder = json.loads(json.dumps(document))
        placeholder["trace"][0]["activation_status"] = status
        with pytest.raises(ValueError, match="executable reference"):
            decode_target_surface(canonical_json(placeholder))

    doubled = json.loads(json.dumps(document))
    doubled["trace"][1]["reference"] = doubled["trace"][0]["reference"]
    with pytest.raises(ValueError, match="two targets to one reference"):
        decode_target_surface(canonical_json(doubled))

    drifted = json.loads(json.dumps(document))
    drifted["registry"]["specs"][0]["value"] += 1.0
    with pytest.raises(ValueError, match="registry version"):
        decode_target_surface(canonical_json(drifted))

    with pytest.raises(ValueError, match="canonical JSON"):
        decode_target_surface(json.dumps(document).encode())


_STATUSES = (
    "",
    "active",
    "requires_harvested_cell_references",
    "requires_harvested_fact_reference",
    "draft",
    "placeholder",
)


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)
@given(
    chosen=st.lists(
        st.tuples(st.integers(0, len(ALL_TARGETS) - 1), st.sampled_from(_STATUSES)),
        min_size=1,
        max_size=len(ALL_TARGETS),
        unique_by=lambda item: item[0],
    )
)
def test_property_trace_is_a_bijection_and_no_placeholder_compiles(
    tmp_path, chosen
) -> None:
    facts = write_facts(tmp_path / "all_facts.jsonl", toy_facts(ALL_TARGETS))
    rows = []
    for index, status in chosen:
        name, entity, measure, filter_, _factor = ALL_TARGETS[index]
        row = reference_row(name, entity=entity, measure=measure, filter_=filter_)
        row["metadata"] = {"activation_status": status} if status else {}
        rows.append(row)
    node = _compile_node(reference_document(rows))
    executable = all(status in {"", "active"} for _, status in chosen)
    if not executable:
        with pytest.raises(ValueError, match="non-executable placeholder"):
            TARGETS_COMPILE.run(_context(node, facts=facts))
        return
    surface = decode_target_surface(
        TARGETS_COMPILE.run(_context(node, facts=facts)).artifacts["surface"]
    )
    references = [row["name"] for row in rows]
    traced = [row["reference"] for row in surface.trace]
    targets = [spec.name for spec in surface.registry.specs]
    assert traced == references == targets
    assert len(set(traced)) == len(traced)
    # Each trace row carries its own reference's status and its own fact.
    for row, (_, status), spec in zip(
        surface.trace, chosen, surface.registry.specs, strict=True
    ):
        assert row["activation_status"] == status
        assert row["ledger_fact_key"] == f"toy.aggregate_fact.v1:{row['reference']}"
        assert spec.metadata["ledger_fact_key"] == row["ledger_fact_key"]


def test_reference_document_parses_like_the_spec_loader() -> None:
    document = references_for(CALIBRATION_TARGETS)
    references = parse_reference_document(document, country=COUNTRY)
    assert [reference.name for reference in references] == [
        name for name, *_ in CALIBRATION_TARGETS
    ]
    assert all(reference.hierarchy is not None for reference in references)
    with pytest.raises(ValueError, match="declares country"):
        parse_reference_document(document, country="yy")
    carried = reference_document(
        [{**reference_row("v", entity="person", measure="age"), "value": 3}]
    )
    with pytest.raises(ValueError, match="observed value"):
        parse_reference_document(carried, country=COUNTRY)
    with pytest.raises(ValueError, match="carries only"):
        parse_reference_document(
            {**document, "target_profile": {"schema_version": 2}}, country=COUNTRY
        )


def test_problem_encodes_the_surface_on_the_population_axis(tmp_path) -> None:
    frame = toy_frame()
    payload = _surface(tmp_path)
    result = TARGETS_PROBLEM.run(
        _context(
            _problem_node(),
            frame=frame,
            artifacts={"surface": _artifact(payload, TARGET_SURFACE_TYPE)},
        )
    )
    ordered = decode_problem(result.artifacts["problem"])
    surface = decode_target_surface(payload)

    assert ordered.entity_ids == tuple(range(frame.n("household")))
    assert ordered.problem.weight_entity == "household"
    assert ordered.problem.skipped == ()
    assert ordered.bindings["surface_sha256"] == surface.sha256
    assert ordered.bindings["registry_version"] == surface.registry.version
    assert [row["reference"] for row in ordered.target_metadata] == [
        spec.name for spec in surface.registry.specs
    ]
    assert result.receipt["n_targets"] == len(CALIBRATION_TARGETS)
    assert result.receipt["weight_kind"] == "design"


def test_problem_refuses_targets_on_columns_the_node_does_not_slice(tmp_path) -> None:
    frame = toy_frame()
    payload = _surface(tmp_path)
    context = _context(
        _problem_node(),
        frame=frame,
        artifacts={"surface": _artifact(payload, TARGET_SURFACE_TYPE)},
    )
    sliced = dict(context.tables)
    sliced["person"] = sliced["person"].drop(columns=["employment_income"])
    narrowed = KernelContext(
        node=context.node,
        tables=sliced,
        weights=context.weights,
        strata=context.strata,
        params=context.params,
        rng=context.rng,
        artifacts=context.artifacts,
    )
    with pytest.raises(ValueError, match="uncompilable target"):
        TARGETS_PROBLEM.run(narrowed)

    missing_family = {k: v for k, v in context.tables.items() if k != "family"}
    with pytest.raises(ValueError, match="family"):
        TARGETS_PROBLEM.run(
            KernelContext(
                node=context.node,
                tables=missing_family,
                weights=context.weights,
                strata=context.strata,
                params=context.params,
                rng=context.rng,
                artifacts=context.artifacts,
            )
        )

    with pytest.raises(ValueError, match="alias 'surface'"):
        TARGETS_PROBLEM.run(_context(_problem_node(), frame=frame))

    masked = Node(
        "problem",
        "targets.problem@1",
        population="open",
        inputs=(Slice("person", ("employment_income", "is_adult"), rows="is_adult"),),
        params={"entities": ENTITIES, "weight_entity": "household"},
        artifact_outputs=(ArtifactOutput("problem", PROBLEM_TYPE),),
    )
    with pytest.raises(ValueError, match="row-masked"):
        TARGETS_PROBLEM.run(
            _context(
                masked,
                frame=frame,
                artifacts={"surface": _artifact(payload, TARGET_SURFACE_TYPE)},
            )
        )


_weight = st.floats(min_value=0.5, max_value=500.0, allow_nan=False)


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)
@given(weights=st.lists(_weight, min_size=6, max_size=6))
def test_property_problem_rows_equal_direct_weighted_totals(tmp_path, weights) -> None:
    reweighted = toy_frame(weights=weights)
    payload = _surface(tmp_path, frame=reweighted)
    result = TARGETS_PROBLEM.run(
        _context(
            _problem_node(),
            frame=reweighted,
            artifacts={"surface": _artifact(payload, TARGET_SURFACE_TYPE)},
        )
    )
    ordered = decode_problem(result.artifacts["problem"])
    estimates = ordered.problem.estimates(ordered.problem.initial_weights.values)
    direct = [
        weighted_total(reweighted, entity, measure, filter_)
        for _name, entity, measure, filter_, _factor in CALIBRATION_TARGETS
    ]
    assert any(filter_ for *_, filter_, _ in CALIBRATION_TARGETS)
    np.testing.assert_allclose(estimates, direct, rtol=1e-12, atol=1e-9)
    assert np.array_equal(ordered.problem.initial_weights.values, np.asarray(weights))


def _compare_node() -> Node:
    return Node(
        "compare",
        "takeup.compare@1",
        population="terminal",
        sources=("facts",),
        params={
            "country": COUNTRY,
            "references": canonical_text(references_for(HOLDOUT_TARGETS)),
            "references_sha256": SHA,
            "entities": ENTITIES,
            "weight_entity": "household",
        },
        artifact_outputs=(ArtifactOutput("comparison", COMPARISON_TYPE),),
    )


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)
@given(weights=st.lists(_weight, min_size=6, max_size=6))
def test_property_comparison_reports_the_fact_and_the_weighted_model(
    tmp_path, weights
) -> None:
    facts = write_facts(tmp_path / "holdout.jsonl", toy_facts(HOLDOUT_TARGETS))
    frame = toy_frame(weights=weights, kind=WeightKind.CALIBRATED)
    result = TAKEUP_COMPARE.run(_context(_compare_node(), facts=facts, frame=frame))
    document = json.loads(result.artifacts["comparison"])
    (row,) = document["comparisons"]
    (fact,) = toy_facts(HOLDOUT_TARGETS)

    assert row["comparator"] == fact["value"]
    assert row["model"] == pytest.approx(
        weighted_total(frame, "person", "receives_support"), rel=1e-12
    )
    assert row["difference"] == pytest.approx(row["model"] - row["comparator"])
    assert row["ratio"] == pytest.approx(row["model"] / row["comparator"])
    assert row["reference"] == row["name"] == HOLDOUT_TARGETS[0][0]
    assert document["weight_kind"] == "calibrated"
    assert result.receipt["n_comparisons"] == 1
