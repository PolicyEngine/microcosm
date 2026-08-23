"""Emit dashboard lineage JSON from the resolved US country bundle.

The packaged generation-1 bundle is the same authority the compiler reads.
This tool derives a presentation view from its normalized imputation domain;
there is deliberately no dashboard-only lineage spec to edit in parallel.

    uv run python tools/emit_lineage_dashboard.py --out /path/to/lineage.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from microcosm.build.spec_engine import load_bundle

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


def _producer_rows(graph: Mapping[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, value in enumerate(
        _array(graph.get("nodes", []), "producer_graph/nodes")
    ):
        node = _mapping(value, f"producer_graph/nodes/{index}")
        outputs = [
            {
                key: output[key]
                for key in ("entity", "column", "coverage_scope", "write_policy")
                if key in output
            }
            for output_index, output_value in enumerate(
                _array(node.get("outputs", []), f"producer_graph/nodes/{index}/outputs")
            )
            for output in [
                _mapping(
                    output_value,
                    f"producer_graph/nodes/{index}/outputs/{output_index}",
                )
            ]
        ]
        rows.append(
            {
                "id": node["id"],
                "name": node["name"],
                "kind": node["kind"],
                "kernel": node["kernel"],
                "outputs": outputs,
            }
        )
    return rows


def emit(bundle_root: Path = US_BUNDLE) -> dict[str, object]:
    """Compile ``bundle_root`` and return its derived dashboard projection."""

    resolved = load_bundle(bundle_root)
    imputation = resolved.domain("imputation").to_wire()
    blocks = _mapping(imputation.get("predictor_blocks"), "predictor_blocks")
    models = _mapping(imputation.get("models"), "models")
    variables: list[dict[str, object]] = []
    families = _array(imputation.get("families"), "families")
    for family_index, family_value in enumerate(families):
        family = _mapping(family_value, f"families/{family_index}")
        block_ids, required, optional = _predictors(family, blocks)
        for target_index, target_value in enumerate(
            _array(family.get("targets"), f"families/{family_index}/targets")
        ):
            target = _mapping(
                target_value, f"families/{family_index}/targets/{target_index}"
            )
            variables.append(
                {
                    "variable": target["name"],
                    "entity": target["entity"],
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
                }
            )

    graph = _mapping(imputation.get("producer_graph"), "producer_graph")
    producers = _producer_rows(graph)
    waiver_records = list(
        _array(imputation.get("waiver_records", []), "waiver_records")
    )
    value_kind_counts = Counter(str(row["draw"]) for row in variables)
    return {
        "schema_version": resolved.schema_version,
        "spec_binding": resolved.spec_binding.to_wire(),
        "spec_sha256": resolved.spec_sha256,
        "models": dict(models),
        "predictor_blocks": dict(blocks),
        "variables": variables,
        "computed_producers": producers,
        "known_gaps": waiver_records,
        "counts": {
            "imputed_variables": len(variables),
            "families": len(families),
            "computed_producers": len(producers),
            "boolean": value_kind_counts["flag"],
            "amount": value_kind_counts["amount"],
            "categorical": value_kind_counts["category"],
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
