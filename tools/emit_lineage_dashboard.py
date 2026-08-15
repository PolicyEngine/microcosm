"""Emit the dashboard's lineage JSON from the imputation lineage spec.

The spec (``specs/us_imputation_lineage.yaml``) is the source of truth; this
tool flattens it to one row per imputed variable — its stage, family,
predictor set (resolved to columns), model, and donor path — plus the model
attributes and computed producers, for ``microcosm.institute`` to render.
Nothing on the dashboard page is hand-typed.

    uv run python tools/emit_lineage_dashboard.py --out /path/to/lineage.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "specs" / "us_imputation_lineage.yaml"

_BOOLEAN_PREFIXES = ("has_", "is_", "receives_")


def _draw_kind(target: str, family: str) -> str:
    if target.startswith(_BOOLEAN_PREFIXES) or family.endswith("boolean"):
        return "boolean"
    return "amount"


def emit(spec_path: Path = SPEC) -> dict:
    text = spec_path.read_text(encoding="utf-8")
    spec = yaml.safe_load(text)
    sets = spec["predictor_sets"]
    variables = []
    for family in spec["imputed_families"]:
        pset = sets[family["predictor_set"]]
        for target in family["targets"]:
            variables.append(
                {
                    "variable": target,
                    "entity": family["entity"],
                    "family": family["family"],
                    "family_id": family["id"],
                    "stage": family["stage"],
                    "draw": _draw_kind(target, family["family"]),
                    "model": family["model"],
                    "predictor_set": family["predictor_set"],
                    "predictors_required": list(pset.get("required", [])),
                    "predictors_optional": list(pset.get("optional", [])),
                    "donor": family.get("donor")
                    or f"{family.get('donor_channel')} → {family.get('recipient_channel')}",
                }
            )
    return {
        "schema_version": spec["schema_version"],
        "spec_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "provenance": spec["provenance"],
        "models": spec["models"],
        "predictor_sets": sets,
        "variables": variables,
        "computed_producers": spec["computed_producers"],
        "known_gaps": spec.get("known_gaps", []),
        "counts": {
            "imputed_variables": len(variables),
            "families": len(spec["imputed_families"]),
            "computed_producers": len(spec["computed_producers"]),
            "boolean": sum(v["draw"] == "boolean" for v in variables),
            "amount": sum(v["draw"] == "amount" for v in variables),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=SPEC)
    args = parser.parse_args(argv)
    payload = emit(args.spec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out}: {payload['counts']['imputed_variables']} variables, "
        f"{payload['counts']['families']} families, spec {payload['spec_sha256'][:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
