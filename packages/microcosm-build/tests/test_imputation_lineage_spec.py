"""The dashboard and migration gates consume the one US imputation bundle."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping

import pytest

from microcosm.build.spec_engine import CompiledSpecIR, compile_spec, load_bundle
from microcosm.build.spec_engine.model import thaw_json
from microcosm.build.us_runtime.acs_transfer import (
    ACS_GROUP_TRANSFER_PREDICTORS,
    ACS_OPTIONAL_PERSON_TRANSFER_PREDICTORS,
    ACS_PERSON_TRANSFER_PREDICTORS,
    DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
)
from microcosm.build.us_runtime.puf_support import PUF_TAX_DETAIL_DEFAULT_PREDICTORS
from microcosm.build.us_runtime.stacked_spine import stacked_gap_fill_plan
from microcosm.build.us_runtime.us_late_producer_registry import (
    CANONICAL_US_LATE_PRODUCER_REGISTRY,
    CANONICAL_US_LATE_TRANSFER_GROUPS,
)
from microcosm.fit.qrf import DEFAULT_N_ESTIMATORS, DEFAULT_ZERO_ATOL
from tools.emit_lineage_dashboard import emit

pytest.importorskip(
    "policyengine_us",
    reason="live-engine oracle: the wheels gate's venv installs no engine",
    exc_type=ModuleNotFoundError,
)


@pytest.fixture(scope="module")
def compiled_us() -> CompiledSpecIR:
    return compile_spec(load_bundle("us"))


@pytest.fixture(scope="module")
def imputation(compiled_us: CompiledSpecIR) -> Mapping[str, object]:
    return compiled_us.resource("imputation")


def test_bundle_declares_every_legacy_transfer_family_and_target(
    imputation: Mapping[str, object],
) -> None:
    declared = {family["id"]: family for family in imputation["families"]}
    live: dict[str, list[str]] = {}
    for direction in stacked_gap_fill_plan():
        for entity, families in dict(direction.target_families).items():
            for family, targets in families.items():
                live[f"early/{direction.name}/{entity}/{family}"] = list(targets)
    for group in CANONICAL_US_LATE_TRANSFER_GROUPS:
        live[f"late/{group.entity}/{group.family}"] = list(group.targets)

    assert set(live) <= set(declared)
    for family_id, targets in live.items():
        assert [row["name"] for row in declared[family_id]["targets"]] == targets


def test_every_imputed_family_names_declared_predictors_and_model(
    imputation: Mapping[str, object],
) -> None:
    blocks = imputation["predictor_blocks"]
    models = imputation["models"]
    for family in imputation["families"]:
        assert family["model"] in models, family["id"]
        assert family["predictors"], family["id"]
        assert set(family["predictors"]) <= set(blocks), family["id"]


def test_predictor_blocks_match_constants_era_oracle(
    imputation: Mapping[str, object],
) -> None:
    blocks = imputation["predictor_blocks"]
    assert blocks["acs_person_required"]["columns"] == list(
        ACS_PERSON_TRANSFER_PREDICTORS
    )
    assert blocks["acs_person_optional_income"]["columns"] + blocks[
        "acs_person_optional_structure"
    ]["columns"] == list(ACS_OPTIONAL_PERSON_TRANSFER_PREDICTORS)
    assert blocks["acs_group_required"]["columns"] == list(
        ACS_GROUP_TRANSFER_PREDICTORS
    )
    assert blocks["puf_tax_detail"]["columns"] == list(
        PUF_TAX_DETAIL_DEFAULT_PREDICTORS
    )


def test_model_and_split_attributes_match_constants_era_oracle(
    imputation: Mapping[str, object],
) -> None:
    params = imputation["models"]["regime_gated_qrf"]["params"]
    assert params["n_estimators"] == DEFAULT_N_ESTIMATORS
    assert params["zero_atol"] == DEFAULT_ZERO_ATOL
    for family in imputation["families"]:
        if family["stage"] == "gap_fill_stacked_spine":
            assert (
                family["max_targets_per_fit"]
                == DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
            ), family["id"]


def test_non_transfer_producers_match_constants_era_registry(
    compiled_us: CompiledSpecIR,
) -> None:
    compiled_nodes = {
        node.name: node
        for node in compiled_us.producer_graph.nodes
        if node.kind != "late_transfer"
    }
    live = {
        name: producer
        for name, producer in CANONICAL_US_LATE_PRODUCER_REGISTRY.items()
        if str(producer.kind) != "late_transfer"
    }
    assert set(compiled_nodes) == set(live)
    for name, producer in live.items():
        assert [thaw_json(row)["column"] for row in compiled_nodes[name].outputs] == [
            getattr(output, "column", None)
            or getattr(output, "name", None)
            or str(output)
            for output in producer.outputs
        ], name


def test_complete_bundle_family_inventory(
    imputation: Mapping[str, object],
) -> None:
    counts = Counter(family["stage"] for family in imputation["families"])
    targets = Counter(
        family["stage"]
        for family in imputation["families"]
        for _target in family["targets"]
    )
    assert counts == {
        "gap_fill_stacked_spine": 13,
        "primary_puf_qrf": 1,
        "late_producer_dag": 19,
    }
    assert targets == {
        "gap_fill_stacked_spine": 48,
        "primary_puf_qrf": 65,
        "late_producer_dag": 70,
    }
    for family in imputation["families"]:
        keys = [(target["entity"], target["name"]) for target in family["targets"]]
        assert len(keys) == len(set(keys)), family["id"]


def test_dashboard_is_an_exact_compiled_ir_projection(
    compiled_us: CompiledSpecIR,
) -> None:
    payload = emit()
    encoded = json.dumps(payload, indent=1) + "\n"
    assert json.loads(encoded) == payload
    assert payload["spec_binding"]["spec_sha256"] == payload["spec_sha256"]
    assert payload["compiler_ir_abi"] == compiled_us.compiler_ir_abi.to_wire()
    assert payload["counts"] == {
        "imputed_variables": 183,
        "families": 33,
        "computed_producers": 38,
        "producer_outputs": 227,
        "graph_output_columns": 152,
        "typed_graph_output_columns": 134,
        "column_lineage_closure": 173,
        "typed_columns": 173,
        "write_event_segments": 241,
        "graph_final_owner_segments": 170,
        "graph_final_cell_atoms": 763,
        "graph_authority_columns": 134,
        "graph_authority_segments": 152,
        "family_authority_columns": 48,
        "family_authority_segments": 48,
        "take_up_authority_columns": 13,
        "take_up_ownership_segments": 14,
        "lineage_authority_segments": 214,
        "take_up_programs": 13,
        "typed_artifacts": 84,
        "typed_scopes": 7,
        "boolean": 45,
        "amount": 132,
        "categorical": 5,
        "count": 1,
        "value_kinds": {"amount": 132, "category": 5, "count": 1, "flag": 45},
    }
    assert len(payload["known_gaps"]) == 1
    expected_producers = [
        {
            "id": node.id,
            "name": node.name,
            "kind": node.kind,
            "kernel": node.kernel,
            "outputs": [thaw_json(row) for row in node.outputs],
            "write_scopes": [thaw_json(row) for row in node.write_scopes],
        }
        for node in compiled_us.producer_graph.nodes
    ]
    assert payload["computed_producers"] == expected_producers

    expected_write_events = [
        {
            "producer": node.id,
            "stage": node.id,
            "entity": scope["entity"],
            "column": scope["column"],
            "row_scope": scope["row_scope"],
            "mode": scope["mode"],
            **segment,
        }
        for node in compiled_us.producer_graph.nodes
        for scope_value in node.write_scopes
        for scope in [thaw_json(scope_value)]
        for segment in scope["cell_segments"]
    ]
    assert payload["write_event_segments"] == expected_write_events

    graph = compiled_us.producer_graph
    registry = thaw_json(graph.scope_registry)
    scope_atoms = {row["id"]: tuple(row["atoms"]) for row in registry["scopes"]}

    def segment_atoms(segment):
        if segment["predicate"] == "origin_clone":
            return (f"origin:{segment['origin']}/clone:{segment['clone_index']}",)
        return scope_atoms[segment["coverage_scope"]]

    raw_candidates = {}
    for node in graph.nodes:
        for scope_value in node.write_scopes:
            scope = thaw_json(scope_value)
            for segment in scope["cell_segments"]:
                for atom in segment_atoms(segment):
                    raw_candidates.setdefault(
                        (scope["entity"], scope["column"], atom), []
                    ).append((node.id, segment["write_policy"]))

    reachable = {node.id: set() for node in graph.nodes}
    for producer, consumer in graph.edges:
        if producer in reachable and consumer in reachable:
            reachable[producer].add(consumer)
    changed = True
    while changed:
        changed = False
        for node_id, descendants in reachable.items():
            expanded = set(descendants)
            for child in tuple(descendants):
                expanded.update(reachable[child])
            if expanded != descendants:
                reachable[node_id] = expanded
                changed = True

    matrix_cells = {}
    for matrix_row in graph.ownership_matrix:
        final_action = next(
            action for action in matrix_row["producer_actions"] if action["owns_final"]
        )
        matrix_cells[
            (
                matrix_row["entity"],
                matrix_row["target"],
                f"origin:{matrix_row['origin']}/clone:{matrix_row['clone_index']}",
            )
        ] = (matrix_row["final_owner"], final_action["action"])

    final_cells = {}
    for row in payload["graph_final_owner_segments"]:
        assert row["authority_surface"] == "producer_graph"
        assert row["origin_class"] == row["producer_kind"]
        for atom in row["row_scopes"]:
            cell = (row["entity"], row["column"], atom)
            assert cell not in final_cells
            final_cells[cell] = (row["owner"], row["write_policy"])
    assert set(final_cells) == set(raw_candidates)
    assert len(final_cells) == 763
    assert (
        sum("finalization" in row for row in payload["graph_final_owner_segments"])
        == 18
    )
    for cell, candidates in raw_candidates.items():
        if cell in matrix_cells:
            assert final_cells[cell] == matrix_cells[cell]
            continue
        expected = [
            candidate
            for candidate in candidates
            if not any(
                other[0] in reachable[candidate[0]]
                for other in candidates
                if other != candidate
            )
        ]
        assert len(expected) == 1, cell
        assert final_cells[cell] == expected[0]

    contracts_by_key = {
        row["key"]: row for row in compiled_us.typed_inventory["columns"]
    }
    closure = payload["column_lineage_closure"]
    assert [row["key"] for row in closure] == sorted(contracts_by_key)
    assert all(row["lineage_segments"] for row in closure)
    assert all(
        segment.get("origin_class")
        for row in closure
        for segment in row["lineage_segments"]
    )
    surfaces_by_key = {
        row["key"]: {
            segment["authority_surface"] for segment in row["lineage_segments"]
        }
        for row in closure
    }

    take_up = compiled_us.resource("take_up")
    take_up_registry = take_up["scope_registry"]
    take_up_scope_atoms = {
        row["id"]: list(row["atoms"]) for row in take_up_registry["scopes"]
    }
    take_up_universe = list(take_up_registry["universe"])
    engine_programs = thaw_json(compiled_us.generated_authorities["engine_abi_lock"])[
        "programs"
    ]
    expected_take_up_segments = []
    for program in take_up["programs"]:
        abi = engine_programs[program["id"]]
        segments = program.get("segments", []) or [
            {
                "row_scopes": take_up_universe,
                "ownership": program["ownership"],
                "pipeline": program["pipeline"],
                "final_owner_stage": program["final_owner_stage"],
            }
        ]
        for segment in segments:
            stage = segment["final_owner_stage"]
            ownership = segment["ownership"]
            expected_take_up_segments.append(
                {
                    "authority_surface": "take_up",
                    "predicate_space": take_up_registry["predicate_space"],
                    "program": program["id"],
                    "column_key": f"{abi['entity']}.{abi['variable']}",
                    "entity": abi["entity"],
                    "column": abi["variable"],
                    "row_scopes": list(
                        segment.get(
                            "row_scopes",
                            take_up_scope_atoms.get(segment.get("row_scope")),
                        )
                    ),
                    "owner": stage,
                    "stage": stage,
                    "origin_class": ownership,
                    "ownership": ownership,
                    "pipeline": list(segment["pipeline"]),
                }
            )
    assert payload["take_up_ownership_segments"] == expected_take_up_segments

    imputation = compiled_us.resource("imputation")
    expected_family_segments = []
    for family in imputation["families"]:
        if family["stage"] != "gap_fill_stacked_spine":
            continue
        recipient_channel = family["recipient"]["channel"]
        scope_id = f"{recipient_channel}_source"
        for target in family["targets"]:
            expected_family_segments.append(
                {
                    "authority_surface": "imputation_family",
                    "predicate_space": registry["predicate_space"],
                    "family_id": family["id"],
                    "column_key": f"{target['entity']}.{target['name']}",
                    "entity": target["entity"],
                    "column": target["name"],
                    "row_scopes": list(scope_atoms[scope_id]),
                    "source_row_scope": scope_id,
                    "producer": family["execution_contract"],
                    "owner": family["id"],
                    "stage": family["stage"],
                    "origin_class": "modeled",
                    "direction": family.get("direction"),
                    "recipient_channel": recipient_channel,
                    "producer_binding": dict(target["producer_binding"]),
                }
            )
    expected_family_segments.sort(
        key=lambda row: (
            row["column_key"],
            row["family_id"],
            tuple(row["row_scopes"]),
        )
    )
    assert payload["family_authority_segments"] == expected_family_segments

    graph_atoms_by_key = {}
    for segment in payload["graph_authority_segments"]:
        graph_atoms_by_key.setdefault(segment["column_key"], set()).update(
            segment["row_scopes"]
        )
    family_atoms_by_key = {
        segment["column_key"]: set(segment["row_scopes"])
        for segment in expected_family_segments
    }
    shared_graph_family = set(graph_atoms_by_key) & set(family_atoms_by_key)
    assert len(shared_graph_family) == 20
    assert all(
        graph_atoms_by_key[key].isdisjoint(family_atoms_by_key[key])
        for key in shared_graph_family
    )

    lineage_segments = [
        segment for row in closure for segment in row["lineage_segments"]
    ]
    assert Counter(segment["authority_surface"] for segment in lineage_segments) == {
        "producer_graph": 152,
        "imputation_family": 48,
        "take_up": 14,
    }
    assert Counter(frozenset(surfaces) for surfaces in surfaces_by_key.values()) == {
        frozenset({"producer_graph"}): 113,
        frozenset({"imputation_family"}): 28,
        frozenset({"take_up"}): 11,
        frozenset({"producer_graph", "imputation_family"}): 19,
        frozenset({"producer_graph", "take_up"}): 1,
        frozenset({"producer_graph", "imputation_family", "take_up"}): 1,
    }
    assert surfaces_by_key["person.taxable_interest_income"] == {
        "producer_graph",
        "imputation_family",
    }
    assert surfaces_by_key["person.takes_up_medicare_if_eligible"] == {
        "producer_graph",
        "take_up",
    }
    assert surfaces_by_key["spm_unit.takes_up_housing_assistance_if_eligible"] == {
        "producer_graph",
        "imputation_family",
        "take_up",
    }

    cells = [
        (segment["predicate_space"], segment["column_key"], atom)
        for segment in lineage_segments
        for atom in segment["row_scopes"]
    ]
    assert len(cells) == len(set(cells))
    assert (
        sum(
            payload["counts"][name]
            for name in ("boolean", "amount", "categorical", "count")
        )
        == payload["counts"]["imputed_variables"]
    )

    closure_by_key = {row["key"]: row for row in closure}
    assert all(
        variable["column_lineage_segments"]
        == closure_by_key[variable["column_key"]]["lineage_segments"]
        for variable in payload["variables"]
    )
