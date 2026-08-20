"""Emit dashboard lineage JSON from the compiled US country bundle.

The tool compiles the packaged generation-1 bundle, then projects its typed
column contracts, compiler-expanded output closure, raw write events, and
cell-exact owner segments in each compiler predicate space. Normalized family
metadata is read only through the resulting compiler IR; there is deliberately
no dashboard-only lineage spec to edit in parallel.

The column closure covers every typed contract across the graph, early-family,
and take-up compiler surfaces. A column can have multiple lineage surfaces when
their exact cells are disjoint or their compiler predicate spaces are not
comparable; the dashboard never invents cross-space supersession. It does not
claim every incidental source column present in a historical H5 artifact.
Artifact-presence certification is owned by the plan-derived selector contract
and its current build receipts.

    uv run python tools/emit_lineage_dashboard.py --out /path/to/lineage.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from microcosm.build.spec_engine import CompiledSpecIR, compile_spec, load_bundle
from microcosm.build.spec_engine.model import thaw_json

ROOT = Path(__file__).resolve().parents[1]
US_BUNDLE = ROOT / "packages" / "microcosm-build" / "src" / "microcosm" / "build" / "us"


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location}: expected object")
    return value


def _array(value: object, location: str) -> Sequence[object]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"{location}: expected array")
    return value


def _family_donor(family: Mapping[str, Any]) -> str | None:
    donor = family.get("donor")
    recipient = family.get("recipient")
    if isinstance(donor, Mapping) and isinstance(recipient, Mapping):
        donor_channel = donor.get("channel")
        recipient_channel = recipient.get("channel")
        if isinstance(donor_channel, str) and isinstance(recipient_channel, str):
            return f"{donor_channel} → {recipient_channel}"
    if isinstance(donor, Mapping):
        channel = donor.get("channel")
        if isinstance(channel, str):
            return channel
    return None


def _predictors(
    family: Mapping[str, Any],
    blocks: Mapping[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    block_ids = [
        str(value)
        for value in _array(family.get("predictors", []), "family/predictors")
    ]
    required: list[str] = []
    optional: list[str] = []
    for block_id in block_ids:
        block = _mapping(blocks.get(block_id), f"predictor_blocks/{block_id}")
        columns = [
            str(value)
            for value in _array(
                block.get("columns", []), f"predictor_blocks/{block_id}/columns"
            )
        ]
        destination = optional if block.get("availability") == "observed" else required
        destination.extend(columns)
    return block_ids, required, optional


def _compiled_mapping(value: object, location: str) -> dict[str, Any]:
    thawed = thaw_json(value)
    if not isinstance(thawed, dict):
        raise TypeError(f"{location}: expected compiled object")
    return thawed


def _write_event_segments(compiled: CompiledSpecIR) -> list[dict[str, object]]:
    """Flatten every compiler-owned write event without changing its predicate."""

    rows: list[dict[str, object]] = []
    for node in compiled.producer_graph.nodes:
        for scope_index, scope_value in enumerate(node.write_scopes):
            scope = _compiled_mapping(
                scope_value,
                f"producer_graph/{node.id}/write_scopes/{scope_index}",
            )
            for segment_index, segment_value in enumerate(
                _array(
                    scope.get("cell_segments", []),
                    (
                        f"producer_graph/{node.id}/write_scopes/{scope_index}/"
                        "cell_segments"
                    ),
                )
            ):
                segment = _mapping(
                    segment_value,
                    (
                        f"producer_graph/{node.id}/write_scopes/{scope_index}/"
                        f"cell_segments/{segment_index}"
                    ),
                )
                rows.append(
                    {
                        "producer": node.id,
                        "stage": node.id,
                        "entity": scope["entity"],
                        "column": scope["column"],
                        "row_scope": scope["row_scope"],
                        "mode": scope["mode"],
                        **dict(segment),
                    }
                )
    return rows


def _producer_rows(compiled: CompiledSpecIR) -> list[dict[str, object]]:
    """Return compiler-expanded outputs and their exact write authorities."""

    rows: list[dict[str, object]] = []
    for node in compiled.producer_graph.nodes:
        outputs = [
            _compiled_mapping(value, f"producer_graph/{node.id}/outputs/{index}")
            for index, value in enumerate(node.outputs)
        ]
        write_scopes = [
            _compiled_mapping(
                value,
                f"producer_graph/{node.id}/write_scopes/{index}",
            )
            for index, value in enumerate(node.write_scopes)
        ]
        output_keys = [(row["entity"], row["column"]) for row in outputs]
        scope_keys = [(row["entity"], row["column"]) for row in write_scopes]
        if output_keys != scope_keys:
            raise ValueError(
                f"producer_graph/{node.id}: compiled output/write-scope order differs"
            )
        rows.append(
            {
                "id": node.id,
                "name": node.name,
                "kind": node.kind,
                "kernel": node.kernel,
                "outputs": outputs,
                "write_scopes": write_scopes,
            }
        )
    return rows


def _column_contracts(compiled: CompiledSpecIR) -> list[dict[str, object]]:
    inventory = _compiled_mapping(compiled.typed_inventory, "typed_inventory")
    columns = [
        dict(_mapping(value, f"typed_inventory/columns/{index}"))
        for index, value in enumerate(
            _array(inventory.get("columns", []), "typed_inventory/columns")
        )
    ]
    keys = [str(row["key"]) for row in columns]
    if keys != sorted(set(keys)):
        raise ValueError("typed_inventory/columns: keys must be sorted and unique")
    return columns


def _take_up_rows(
    compiled: CompiledSpecIR,
    *,
    contracts_by_key: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    take_up = _mapping(compiled.resource("take_up"), "take_up")
    generated = _compiled_mapping(
        compiled.generated_authorities, "authorities/generated"
    )
    engine_lock = _mapping(
        generated.get("engine_abi_lock"),
        "authorities/generated/engine_abi_lock",
    )
    engine_programs = _mapping(
        engine_lock.get("programs"),
        "authorities/generated/engine_abi_lock/programs",
    )

    rows: list[dict[str, object]] = []
    for index, value in enumerate(
        _array(take_up.get("programs", []), "take_up/programs")
    ):
        program = _mapping(value, f"take_up/programs/{index}")
        program_id = str(program["id"])
        engine_program = _mapping(
            engine_programs.get(program_id),
            f"authorities/generated/engine_abi_lock/programs/{program_id}",
        )
        variable = str(program["variable"])
        if engine_program.get("variable") != variable:
            raise ValueError(
                f"take_up/programs/{index}/variable: compiled engine ABI disagrees"
            )
        key = f"{engine_program['entity']}.{variable}"
        if key not in contracts_by_key:
            raise ValueError(
                f"take_up/programs/{index}: ABI-bound column {key!r} has no contract"
            )
        segments = [dict(value) for value in program.get("segments", [])]
        if "final_owner_stage" in program:
            final_owner_stages = [str(program["final_owner_stage"])]
        else:
            final_owner_stages = list(
                dict.fromkeys(str(segment["final_owner_stage"]) for segment in segments)
            )
        rows.append(
            {
                "id": program_id,
                "variable": variable,
                "column_key": key,
                "ownership": program["ownership"],
                "pipeline": list(program.get("pipeline", [])),
                "segments": segments,
                "final_owner_stages": final_owner_stages,
                "column_contract": dict(contracts_by_key[key]),
            }
        )
    return rows


def _scope_algebra(
    value: object,
    location: str,
) -> tuple[str, tuple[str, ...], dict[str, tuple[str, ...]]]:
    registry = _compiled_mapping(value, location)
    predicate_space = str(registry["predicate_space"])
    universe = tuple(
        str(atom)
        for atom in _array(registry.get("universe", []), f"{location}/universe")
    )
    if not universe or len(universe) != len(set(universe)):
        raise ValueError(f"{location}/universe: unique non-empty atoms required")
    scopes: dict[str, tuple[str, ...]] = {}
    for index, scope_value in enumerate(
        _array(registry.get("scopes", []), f"{location}/scopes")
    ):
        scope = _mapping(scope_value, f"{location}/scopes/{index}")
        scope_id = str(scope["id"])
        atoms = tuple(
            str(atom)
            for atom in _array(
                scope.get("atoms", []), f"{location}/scopes/{index}/atoms"
            )
        )
        if scope_id in scopes or not atoms or len(atoms) != len(set(atoms)):
            raise ValueError(f"{location}/scopes/{index}: invalid scope declaration")
        if not set(atoms) <= set(universe):
            raise ValueError(f"{location}/scopes/{index}: atom outside universe")
        scopes[scope_id] = atoms
    return predicate_space, universe, scopes


def _producer_segment_atoms(
    segment: Mapping[str, Any],
    *,
    scopes: Mapping[str, tuple[str, ...]],
    universe: Sequence[str],
    location: str,
) -> tuple[str, ...]:
    if segment.get("predicate") == "origin_clone":
        atoms = (f"origin:{segment['origin']}/clone:{int(segment['clone_index'])}",)
    elif segment.get("predicate") == "coverage_scope":
        scope_id = str(segment.get("coverage_scope"))
        if scope_id not in scopes:
            raise ValueError(f"{location}: unknown coverage scope {scope_id!r}")
        atoms = scopes[scope_id]
    else:
        raise ValueError(f"{location}: unsupported compiler segment predicate")
    if not set(atoms) <= set(universe):
        raise ValueError(f"{location}: segment atom outside compiler universe")
    return atoms


def _take_up_ownership_segments(
    compiled: CompiledSpecIR,
    take_up_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Totalize take-up ownership over the compiler's closed scope algebra."""

    take_up = _mapping(compiled.resource("take_up"), "take_up")
    predicate_space, universe, scopes = _scope_algebra(
        take_up.get("scope_registry"),
        "take_up/scope_registry",
    )
    rows: list[dict[str, object]] = []
    for program in take_up_rows:
        contract = _mapping(program["column_contract"], "take_up/column_contract")
        raw_segments = _array(program["segments"], f"take_up/{program['id']}/segments")
        segments = (
            list(raw_segments)
            if raw_segments
            else [
                {
                    "row_scopes": universe,
                    "ownership": program["ownership"],
                    "pipeline": program["pipeline"],
                    "final_owner_stage": program["final_owner_stages"][0],
                }
            ]
        )
        covered_atoms: list[str] = []
        for segment_index, segment_value in enumerate(segments):
            segment = _mapping(
                segment_value,
                f"take_up/{program['id']}/segments/{segment_index}",
            )
            row_scopes = (
                [str(value) for value in segment["row_scopes"]]
                if "row_scopes" in segment
                else list(scopes[str(segment["row_scope"])])
            )
            covered_atoms.extend(row_scopes)
            stage = str(segment["final_owner_stage"])
            ownership = str(segment["ownership"])
            rows.append(
                {
                    "authority_surface": "take_up",
                    "predicate_space": predicate_space,
                    "program": program["id"],
                    "column_key": program["column_key"],
                    "entity": contract["entity"],
                    "column": program["variable"],
                    "row_scopes": row_scopes,
                    "owner": stage,
                    "stage": stage,
                    "origin_class": ownership,
                    "ownership": ownership,
                    "pipeline": list(segment["pipeline"]),
                }
            )
        if len(covered_atoms) != len(set(covered_atoms)) or set(covered_atoms) != set(
            universe
        ):
            raise ValueError(
                f"take_up/{program['id']}: ownership segments must partition "
                f"{list(universe)!r}, got {covered_atoms!r}"
            )
    return rows


def _graph_final_owner_segments(
    compiled: CompiledSpecIR,
) -> tuple[list[dict[str, object]], int]:
    """Select the unique DAG-maximal writer for every exact graph cell atom."""

    graph = compiled.producer_graph
    if graph.scope_registry is None:
        raise ValueError("producer_graph/scope_registry: compiled registry required")
    predicate_space, universe, scopes = _scope_algebra(
        graph.scope_registry,
        "producer_graph/scope_registry",
    )
    nodes_by_id = {node.id: node for node in graph.nodes}
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

    matrix_rows = [
        dict(_mapping(value, f"producer_graph/ownership_matrix/{index}"))
        for index, value in enumerate(graph.ownership_matrix)
    ]
    matrix_keys = {(str(row["entity"]), str(row["target"])) for row in matrix_rows}
    scope_by_owner_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    raw_cells: set[tuple[str, str, str]] = set()
    for node in graph.nodes:
        for scope_index, scope_value in enumerate(node.write_scopes):
            scope = _compiled_mapping(
                scope_value,
                f"producer_graph/{node.id}/write_scopes/{scope_index}",
            )
            scope_key = (node.id, str(scope["entity"]), str(scope["column"]))
            if scope_key in scope_by_owner_key:
                raise ValueError(
                    f"producer_graph/{node.id}: duplicate write scope for "
                    f"{scope_key[1:]!r}"
                )
            scope_by_owner_key[scope_key] = scope
            for segment_index, segment_value in enumerate(
                _array(
                    scope.get("cell_segments", []),
                    f"producer_graph/{node.id}/write_scopes/{scope_index}/cell_segments",
                )
            ):
                location = (
                    f"producer_graph/{node.id}/write_scopes/{scope_index}/"
                    f"cell_segments/{segment_index}"
                )
                segment = dict(_mapping(segment_value, location))
                atoms = _producer_segment_atoms(
                    segment,
                    scopes=scopes,
                    universe=universe,
                    location=location,
                )
                key = (str(scope["entity"]), str(scope["column"]))
                raw_cells.update((key[0], key[1], atom) for atom in atoms)
                if key not in matrix_keys:
                    candidates.append(
                        {
                            "node_id": node.id,
                            "scope": scope,
                            "segment": segment,
                            "atoms": atoms,
                        }
                    )

    candidates_by_cell: dict[tuple[str, str, str], list[int]] = {}
    for candidate_index, candidate in enumerate(candidates):
        scope = candidate["scope"]
        for atom in candidate["atoms"]:
            cell = (str(scope["entity"]), str(scope["column"]), str(atom))
            candidates_by_cell.setdefault(cell, []).append(candidate_index)
    selected_atoms: dict[int, set[str]] = {}
    for cell, indices in candidates_by_cell.items():
        if len({candidates[index]["node_id"] for index in indices}) != len(indices):
            raise ValueError(f"producer_graph: duplicate writer segment for {cell!r}")
        for left_index, left in enumerate(indices):
            for right in indices[left_index + 1 :]:
                left_id = candidates[left]["node_id"]
                right_id = candidates[right]["node_id"]
                if (
                    right_id not in reachable[left_id]
                    and left_id not in reachable[right_id]
                ):
                    raise ValueError(
                        f"producer_graph: incomparable writers for exact cell {cell!r}"
                    )
        maximal = [
            index
            for index in indices
            if not any(
                candidates[other]["node_id"] in reachable[candidates[index]["node_id"]]
                for other in indices
                if other != index
            )
        ]
        if len(maximal) != 1:
            raise ValueError(
                f"producer_graph: expected one DAG-maximal writer for {cell!r}"
            )
        selected_atoms.setdefault(maximal[0], set()).add(cell[2])

    def report_row(
        *,
        owner_id: str,
        scope: Mapping[str, Any],
        segment: Mapping[str, Any],
        atoms: Sequence[str],
        finalization: object | None = None,
    ) -> dict[str, object]:
        node = nodes_by_id.get(owner_id)
        if node is None:
            raise ValueError(f"final owner {owner_id!r} is not a compiled producer")
        row: dict[str, object] = {
            "authority_surface": "producer_graph",
            "predicate_space": predicate_space,
            "producer": node.id,
            "producer_name": node.name,
            "producer_kind": node.kind,
            "origin_class": node.kind,
            "owner": node.id,
            "stage": node.id,
            "column_key": f"{scope['entity']}.{scope['column']}",
            "entity": scope["entity"],
            "column": scope["column"],
            "row_scopes": list(atoms),
            "source_row_scope": scope["row_scope"],
            "write_scope_mode": scope["mode"],
            "write_policy": segment["write_policy"],
            "source_predicate": dict(segment),
        }
        if finalization is not None:
            row["finalization"] = finalization
        return row

    rows: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(candidates):
        atoms = selected_atoms.get(candidate_index, set())
        if not atoms:
            continue
        ordered_atoms = [atom for atom in universe if atom in atoms]
        if atoms == set(candidate["atoms"]):
            rows.append(
                report_row(
                    owner_id=candidate["node_id"],
                    scope=candidate["scope"],
                    segment=candidate["segment"],
                    atoms=ordered_atoms,
                )
            )
        else:
            for atom in ordered_atoms:
                rows.append(
                    report_row(
                        owner_id=candidate["node_id"],
                        scope=candidate["scope"],
                        segment=candidate["segment"],
                        atoms=[atom],
                    )
                )

    for matrix_index, matrix_row in enumerate(matrix_rows):
        owner_id = str(matrix_row["final_owner"])
        actions = [
            _mapping(
                value,
                f"producer_graph/ownership_matrix/{matrix_index}/producer_actions/{index}",
            )
            for index, value in enumerate(
                _array(
                    matrix_row.get("producer_actions", []),
                    f"producer_graph/ownership_matrix/{matrix_index}/producer_actions",
                )
            )
        ]
        final_actions = [
            action for action in actions if action.get("owns_final") is True
        ]
        if len(final_actions) != 1 or final_actions[0].get("producer") != owner_id:
            raise ValueError(
                f"producer_graph/ownership_matrix/{matrix_index}: expected exactly "
                f"one final action for {owner_id!r}"
            )
        scope_key = (owner_id, str(matrix_row["entity"]), str(matrix_row["target"]))
        scope = scope_by_owner_key.get(scope_key)
        if scope is None:
            raise ValueError(
                f"producer_graph/ownership_matrix/{matrix_index}: final owner "
                f"{owner_id!r} has no compiled write scope"
            )
        final_segment = {
            "predicate": "origin_clone",
            "origin": matrix_row["origin"],
            "clone_index": matrix_row["clone_index"],
            "write_policy": final_actions[0]["action"],
        }
        matching = [
            segment
            for value in _array(
                scope.get("cell_segments", []),
                f"producer_graph/{owner_id}/write_scopes/cell_segments",
            )
            for segment in [_mapping(value, "compiled final-owner segment")]
            if all(segment.get(key) == value for key, value in final_segment.items())
        ]
        if len(matching) != 1:
            raise ValueError(
                f"producer_graph/ownership_matrix/{matrix_index}: final owner "
                "action does not match one compiled write segment"
            )
        atom = f"origin:{matrix_row['origin']}/clone:{matrix_row['clone_index']}"
        rows.append(
            report_row(
                owner_id=owner_id,
                scope=scope,
                segment=final_segment,
                atoms=[atom],
                finalization=matrix_row["finalization"],
            )
        )

    final_cells = [
        (str(row["entity"]), str(row["column"]), str(atom))
        for row in rows
        for atom in _array(row["row_scopes"], "graph final row scopes")
    ]
    if len(final_cells) != len(set(final_cells)):
        raise ValueError("producer_graph: final cell atoms are not uniquely owned")
    if set(final_cells) != raw_cells:
        raise ValueError(
            "producer_graph: final cell atoms do not close raw write coverage"
        )
    return rows, len(final_cells)


def _family_authority_segments(
    compiled: CompiledSpecIR,
    *,
    families: Sequence[object],
    contracts: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Project every compiler-normalized early-family target cell."""

    if compiled.producer_graph.scope_registry is None:
        raise ValueError("producer_graph/scope_registry: compiled registry required")
    predicate_space, _universe, scopes = _scope_algebra(
        compiled.producer_graph.scope_registry,
        "producer_graph/scope_registry",
    )
    contract_keys = {str(row["key"]) for row in contracts}
    claims: list[dict[str, object]] = []
    claimed_cells: dict[tuple[str, str, str], str] = {}
    for family_index, family_value in enumerate(families):
        family = _mapping(family_value, f"imputation/families/{family_index}")
        if family.get("stage") != "gap_fill_stacked_spine":
            continue
        for target_index, target_value in enumerate(
            _array(
                family.get("targets", []), f"imputation/families/{family_index}/targets"
            )
        ):
            target = _mapping(
                target_value,
                f"imputation/families/{family_index}/targets/{target_index}",
            )
            key = f"{target['entity']}.{target['name']}"
            if key not in contract_keys:
                raise ValueError(
                    f"compiled early-family output {key!r} has no typed contract"
                )
            recipient = _mapping(
                family.get("recipient"),
                f"imputation/families/{family_index}/recipient",
            )
            recipient_channel = str(recipient["channel"])
            scope_id = f"{recipient_channel}_source"
            if scope_id not in scopes:
                raise ValueError(
                    f"imputation/families/{family_index}: no compiler scope for "
                    f"recipient channel {recipient_channel!r}"
                )
            family_atoms = list(scopes[scope_id])
            for atom in family_atoms:
                cell = (predicate_space, key, atom)
                previous = claimed_cells.get(cell)
                if previous is not None:
                    raise ValueError(
                        f"compiled early-family cell {cell!r} has peer owners "
                        f"{previous!r} and {family['id']!r}"
                    )
                claimed_cells[cell] = str(family["id"])
            binding = _mapping(
                target.get("producer_binding"),
                f"imputation/families/{family_index}/targets/{target_index}/producer_binding",
            )
            claims.append(
                {
                    "authority_surface": "imputation_family",
                    "predicate_space": predicate_space,
                    "family_id": family["id"],
                    "column_key": key,
                    "entity": target["entity"],
                    "column": target["name"],
                    "row_scopes": family_atoms,
                    "source_row_scope": scope_id,
                    "producer": family["execution_contract"],
                    "owner": family["id"],
                    "stage": family["stage"],
                    "origin_class": "modeled",
                    "direction": family.get("direction"),
                    "recipient_channel": recipient_channel,
                    "producer_binding": dict(binding),
                }
            )
    return sorted(
        claims,
        key=lambda row: (
            str(row["column_key"]),
            str(row["family_id"]),
            tuple(str(value) for value in row["row_scopes"]),
        ),
    )


def _column_lineage_closure(
    *,
    contracts: Sequence[Mapping[str, object]],
    graph_segments: Sequence[Mapping[str, object]],
    family_segments: Sequence[Mapping[str, object]],
    take_up_segments: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Close every typed column over exact compiler-derived lineage cells."""

    contracts_by_key = {str(row["key"]): dict(row) for row in contracts}
    lineage_segments = [
        *(dict(row) for row in graph_segments),
        *(dict(row) for row in family_segments),
        *(dict(row) for row in take_up_segments),
    ]
    segments_by_key: dict[str, list[dict[str, object]]] = {}
    owners_by_cell: dict[tuple[str, str, str], str] = {}
    for segment in lineage_segments:
        key = str(segment["column_key"])
        if key not in contracts_by_key:
            raise ValueError(f"lineage authority {key!r} has no typed contract")
        location = f"lineage/{segment['authority_surface']}/{key}"
        atoms = [
            str(value)
            for value in _array(segment.get("row_scopes", []), f"{location}/row_scopes")
        ]
        if not atoms or len(atoms) != len(set(atoms)):
            raise ValueError(f"{location}: unique non-empty row scopes required")
        for atom in atoms:
            cell = (str(segment["predicate_space"]), key, atom)
            previous = owners_by_cell.get(cell)
            if previous is not None:
                raise ValueError(
                    f"lineage cell {cell!r} has peer final authorities "
                    f"{previous!r} and {segment['authority_surface']!r}"
                )
            owners_by_cell[cell] = str(segment["authority_surface"])
        segments_by_key.setdefault(key, []).append(segment)
    if set(segments_by_key) != set(contracts_by_key):
        raise ValueError(
            "typed column closure mismatch: "
            f"missing={sorted(set(contracts_by_key) - set(segments_by_key))!r}, "
            f"unknown={sorted(set(segments_by_key) - set(contracts_by_key))!r}"
        )
    closure = [
        {
            **contracts_by_key[key],
            "lineage_segments": segments_by_key[key],
        }
        for key in sorted(contracts_by_key)
    ]
    return closure, [dict(row) for row in graph_segments]


def emit(bundle_root: Path = US_BUNDLE) -> dict[str, object]:
    """Compile ``bundle_root`` and return its derived dashboard projection."""

    compiled = compile_spec(load_bundle(bundle_root))
    imputation = compiled.resource("imputation")
    blocks = _mapping(imputation.get("predictor_blocks"), "predictor_blocks")
    models = _mapping(imputation.get("models"), "models")
    contracts = _column_contracts(compiled)
    contracts_by_key = {str(row["key"]): row for row in contracts}
    write_event_segments = _write_event_segments(compiled)
    graph_final_owner_segments, graph_final_cell_atoms = _graph_final_owner_segments(
        compiled
    )
    take_up = _take_up_rows(compiled, contracts_by_key=contracts_by_key)
    take_up_ownership_segments = _take_up_ownership_segments(compiled, take_up)
    variables: list[dict[str, object]] = []
    families = _array(imputation.get("families"), "families")
    all_graph_keys = {str(row["column_key"]) for row in graph_final_owner_segments}
    take_up_keys = {str(row["column_key"]) for row in take_up_ownership_segments}
    graph_authority_segments = [
        dict(row)
        for row in graph_final_owner_segments
        if str(row["column_key"]) in contracts_by_key
    ]
    graph_keys = {str(row["column_key"]) for row in graph_authority_segments}
    family_authority_segments = _family_authority_segments(
        compiled,
        families=families,
        contracts=contracts,
    )
    closure, graph_authority_segments = _column_lineage_closure(
        contracts=contracts,
        graph_segments=graph_authority_segments,
        family_segments=family_authority_segments,
        take_up_segments=take_up_ownership_segments,
    )
    closure_by_key = {str(row["key"]): row for row in closure}
    for family_index, family_value in enumerate(families):
        family = _mapping(family_value, f"families/{family_index}")
        block_ids, required, optional = _predictors(family, blocks)
        for target_index, target_value in enumerate(
            _array(family.get("targets"), f"families/{family_index}/targets")
        ):
            target = _mapping(
                target_value, f"families/{family_index}/targets/{target_index}"
            )
            column_key = f"{target['entity']}.{target['name']}"
            contract = contracts_by_key.get(column_key)
            if contract is None:
                raise ValueError(
                    f"families/{family_index}/targets/{target_index}: target "
                    f"{column_key!r} has no compiler column contract"
                )
            variables.append(
                {
                    "variable": target["name"],
                    "entity": target["entity"],
                    "column_key": column_key,
                    "family_id": family["id"],
                    "stage": family["stage"],
                    "direction": family.get("direction"),
                    "draw": target["value_kind"],
                    "dtype": target["dtype"],
                    "model": family["model"],
                    "predictor_blocks": block_ids,
                    "predictors_required": required,
                    "predictors_optional": optional,
                    "donor": _family_donor(family),
                    "requires_concepts": list(target.get("requires_concepts", [])),
                    "waiver": target.get("waiver"),
                    "column_contract": dict(contract),
                }
            )

    producers = _producer_rows(compiled)
    for variable in variables:
        column_key = str(variable["column_key"])
        variable["column_lineage_segments"] = list(
            closure_by_key[column_key]["lineage_segments"]
        )

    typed_inventory = _compiled_mapping(compiled.typed_inventory, "typed_inventory")
    waiver_records = list(
        _array(imputation.get("waiver_records", []), "waiver_records")
    )
    value_kind_counts = Counter(str(row["draw"]) for row in variables)
    return {
        "schema_version": compiled.spec_binding.schema_version,
        "spec_binding": compiled.spec_binding.to_wire(),
        "spec_sha256": compiled.spec_binding.spec_sha256,
        "compiler_ir_abi": compiled.compiler_ir_abi.to_wire(),
        "models": dict(models),
        "predictor_blocks": dict(blocks),
        "variables": variables,
        "computed_producers": producers,
        "column_lineage_closure": closure,
        "write_event_segments": write_event_segments,
        "graph_final_owner_segments": graph_final_owner_segments,
        "graph_authority_segments": graph_authority_segments,
        "family_authority_segments": family_authority_segments,
        "take_up_ownership_segments": take_up_ownership_segments,
        "take_up": take_up,
        "typed_columns": contracts,
        "typed_artifacts": list(typed_inventory["artifacts"]),
        "typed_scopes": list(typed_inventory["scopes"]),
        "known_gaps": waiver_records,
        "counts": {
            "imputed_variables": len(variables),
            "families": len(families),
            "computed_producers": len(producers),
            "producer_outputs": sum(len(producer["outputs"]) for producer in producers),
            "graph_output_columns": len(all_graph_keys),
            "typed_graph_output_columns": len(graph_keys),
            "column_lineage_closure": len(closure),
            "typed_columns": len(contracts),
            "write_event_segments": len(write_event_segments),
            "graph_final_owner_segments": len(graph_final_owner_segments),
            "graph_final_cell_atoms": graph_final_cell_atoms,
            "graph_authority_columns": len(
                {str(row["column_key"]) for row in graph_authority_segments}
            ),
            "graph_authority_segments": len(graph_authority_segments),
            "family_authority_columns": len(
                {str(row["column_key"]) for row in family_authority_segments}
            ),
            "family_authority_segments": len(family_authority_segments),
            "take_up_authority_columns": len(take_up_keys),
            "take_up_ownership_segments": len(take_up_ownership_segments),
            "lineage_authority_segments": (
                len(graph_authority_segments)
                + len(family_authority_segments)
                + len(take_up_ownership_segments)
            ),
            "take_up_programs": len(take_up),
            "typed_artifacts": len(typed_inventory["artifacts"]),
            "typed_scopes": len(typed_inventory["scopes"]),
            "boolean": value_kind_counts["flag"],
            "amount": value_kind_counts["amount"],
            "categorical": value_kind_counts["category"],
            "count": value_kind_counts["count"],
            "value_kinds": dict(sorted(value_kind_counts.items())),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=US_BUNDLE)
    args = parser.parse_args(argv)
    payload = emit(args.bundle)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    counts = _mapping(payload["counts"], "counts")
    print(
        f"wrote {args.out}: {counts['imputed_variables']} variables, "
        f"{counts['families']} families, spec {str(payload['spec_sha256'])[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
