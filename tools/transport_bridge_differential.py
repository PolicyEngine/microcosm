"""Run a small declared bridge/AS differential and write a local JSON receipt.

No country engine is installed by this command. If ``axiom_rules_engine`` is
absent it writes a skipped receipt and exits successfully. A hub case JSON
declares ``unit_rule`` (BenefitUnitRule data), ``rules_bindings`` (the transport
registry contract), ``tables`` (entity -> records), ``weights`` (entity ->
{values, kind}), and ordered ``calls``. Each call has ``id``, ``kernel``,
``inputs`` (entity -> data-column names), ``outputs`` (Owned records), and
``params`` (the kernel contract). ``engine_ref`` values name binding ids and
are replaced with prepared references, including embedded component/term refs.

``comparisons`` contains records with ``call``, ``entity``, ``column``,
``oracle_pointer`` (RFC 6901 pointer into the oracle JSON) and explicit
``tolerance``. Call ids must be nonempty and distinct, each entity/column must
have one output owner, and comparisons must name an output of their call.
These declarations and oracle comparisons are checked before registry
preparation or engine execution. All numbers come from case data or the
oracle; the script has no policy constants.

The golden-08 executable case is unresolved pending an approved declaration.
``build/nz/as_rate_bridge.json`` declares three sole-parent cutout evaluations:
``gross_rate``, ``reduction_at_anchor`` and ``reduction_after_step``, followed by
slope, threshold and cutout composition. Its ``why_not_solve_zero`` explains
why a zero solve on the unit's own Jobseeker schedule cannot reproduce that
harness pairing. This driver has no arithmetic-composition contract for those
evaluations and cannot yet run that recipe. Approved sampled-unit cases may
declare additional small entity-table rows and corresponding oracle arrays.
This command does not read donor HDF5.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.transport.registry import build_transport_registry
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.unit_construction import BenefitUnitRule
from microcosm.graph import KernelContext, Node, Owned, Slice


def _pointer(document, pointer):
    if pointer == "":
        return document
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("oracle_pointer must be an RFC 6901 JSON pointer.")
    value = document
    for token in pointer[1:].split("/"):
        if re.search(r"~(?:[^01]|$)", token):
            raise ValueError("oracle_pointer contains an invalid JSON pointer escape.")
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if re.fullmatch(r"0|[1-9][0-9]*", token) is None:
                raise ValueError(
                    "oracle_pointer requires a canonical non-negative array index."
                )
            index = int(token)
            if index >= len(value):
                raise ValueError("oracle_pointer array index is out of bounds.")
            value = value[index]
        elif isinstance(value, dict) and token in value:
            value = value[token]
        else:
            raise ValueError("oracle_pointer does not resolve to an oracle value.")
    return value


def _validate_case(case, oracle):
    """Refuse ambiguous calls and comparisons before preparing any engine."""
    declared_outputs = {}
    for call in case["calls"]:
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("Every differential call needs a non-empty call id.")
        if call_id in declared_outputs:
            raise ValueError(
                f"The differential case has duplicate call id {call_id!r}."
            )
        declared_outputs[call_id] = set()
    owners = {}
    for call in case["calls"]:
        call_id = call["id"]
        for output in call["outputs"]:
            coordinate = (output["entity"], output["column"])
            if coordinate in owners:
                raise ValueError(
                    f"The differential output coordinate {coordinate!r} has "
                    f"multiple owners: {owners[coordinate]!r} and {call_id!r}."
                )
            owners[coordinate] = call_id
            declared_outputs[call_id].add(coordinate)
    if not case["comparisons"]:
        raise ValueError("The differential case must declare comparisons.")
    expected_values = []
    for row in case["comparisons"]:
        call_id = row["call"]
        if call_id not in declared_outputs:
            raise ValueError(f"A comparison names missing call {call_id!r}.")
        coordinate = (row["entity"], row["column"])
        if coordinate not in declared_outputs[call_id]:
            raise ValueError(
                f"A comparison names missing output {coordinate!r} "
                f"for call {call_id!r}."
            )
        tolerance = row["tolerance"]
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, int | float)
            or not np.isfinite(tolerance)
            or tolerance < 0
        ):
            raise ValueError(
                "Every comparison needs an explicit finite nonnegative tolerance."
            )
        expected = np.atleast_1d(
            np.asarray(_pointer(oracle, row["oracle_pointer"]), dtype=np.float64)
        )
        if not np.all(np.isfinite(expected)):
            raise ValueError("Oracle comparison values must be finite.")
        expected_values.append(expected)
    return expected_values


def _resolve_params(params, refs):
    result = dict(params)
    result["engine_ref"] = refs[result["engine_ref"]]
    if "variables" in result:
        result["variables"] = tuple(result["variables"])
    if "bracket" in result:
        result["bracket"] = tuple(result["bracket"])
    for name in ("input_overrides", "output_coefficients", "components", "solve_input"):
        if name not in result:
            continue
        data = result[name]
        if isinstance(data, str):
            data = json.loads(data)
        else:
            data = json.loads(json.dumps(data))
        records = data if isinstance(data, list) else [data]
        for record in records:
            if "engine_ref" in record:
                record["engine_ref"] = refs[record["engine_ref"]]
        result[name] = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return result


def _call_tables(frame, node):
    """Expose only declared columns and structural ids, as graph execution does."""
    columns = {output.entity: [] for output in node.outputs}
    for item in node.inputs:
        columns.setdefault(item.entity, []).extend(item.columns)
    tables = {}
    for entity, declared in columns.items():
        structural = [frame.schema.entity_id_column(entity)]
        if entity == frame.schema.person_entity:
            structural.extend(
                frame.schema.membership_column(group)
                for group in frame.schema.group_entities
            )
        tables[entity] = frame.table(entity).loc[
            :, list(dict.fromkeys(structural + declared))
        ]
    return tables


def differential(case, oracle, rulespec_root):
    """Validate declarations, execute unique calls, and compare oracle data."""
    expected_values = _validate_case(case, oracle)
    rule = BenefitUnitRule.from_dict(case["unit_rule"])
    prepared = build_transport_registry(
        case["rules_bindings"], rulespec_root, unit_rule=rule
    )
    tables = {
        entity: pd.DataFrame(records) for entity, records in case["tables"].items()
    }
    weights = {
        entity: Weights(
            np.asarray(row["values"], dtype=np.float64), WeightKind(row["kind"])
        )
        for entity, row in case["weights"].items()
    }
    frame = Frame(tables, prepared.schema, weights)
    outputs = {}
    receipts = []
    for call in case["calls"]:
        node = Node(
            call["id"],
            call["kernel"],
            inputs=tuple(
                Slice(entity, tuple(columns))
                for entity, columns in call["inputs"].items()
            ),
            outputs=tuple(Owned(**output) for output in call["outputs"]),
            params=_resolve_params(call["params"], prepared.engine_refs),
        )
        result = prepared.kernels.get(node.kernel).run(
            KernelContext(
                node=node,
                tables=_call_tables(frame, node),
                weights=weights,
                strata=frame.strata,
                params=node.params,
                rng=np.random.default_rng(0),
            )
        )
        receipts.append({"call": node.id, "receipt": dict(result.receipt)})
        for (entity, column), values in result.columns.items():
            outputs[(node.id, entity, column)] = values.to_numpy()
            ids = frame.schema.entity_id_column(entity)
            tables[entity][column] = tables[entity][ids].map(values).to_numpy()
        frame = Frame(tables, prepared.schema, weights)
    comparisons = []
    for row, expected in zip(case["comparisons"], expected_values, strict=True):
        actual = outputs[(row["call"], row["entity"], row["column"])]
        tolerance = row["tolerance"]
        same_shape = actual.shape == expected.shape
        differences = np.abs(actual - expected) if same_shape else None
        comparisons.append(
            {
                **row,
                "actual": actual.tolist(),
                "expected": expected.tolist(),
                "maximum_absolute_difference": float(differences.max())
                if same_shape and len(actual)
                else None,
                "passed": bool(
                    same_shape and len(actual) and np.all(differences <= tolerance)
                ),
            }
        )
    return {
        "status": "passed" if all(row["passed"] for row in comparisons) else "failed",
        "comparisons": comparisons,
        "calls": receipts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--rulespec-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if importlib.util.find_spec("axiom_rules_engine") is None:
        receipt = {"status": "skipped", "reason": "axiom_rules_engine is unavailable"}
    else:
        case_bytes, oracle_bytes = args.case.read_bytes(), args.oracle.read_bytes()
        receipt = differential(
            json.loads(case_bytes), json.loads(oracle_bytes), args.rulespec_root
        )
        receipt.update(
            case_sha256=hashlib.sha256(case_bytes).hexdigest(),
            oracle_sha256=hashlib.sha256(oracle_bytes).hexdigest(),
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": receipt["status"], "receipt": str(args.out)}))
    return int(receipt["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
