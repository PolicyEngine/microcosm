#!/usr/bin/env python3
"""Print a real Axiom input-coverage diagnostic without reading any microdata."""

import argparse
import json
from pathlib import Path

from microcosm.build.concept_coverage import build_concept_coverage
from microcosm.frame.adapters.axiom import AxiomEngine
from microcosm.frame.schema import EntitySchema


def main() -> None:
    """Compile the explicit module/root/entity surface and emit closed JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--rulespec-root", type=Path, action="append", required=True)
    parser.add_argument("--person-entity", default="person")
    parser.add_argument("--group-entity", action="append", required=True)
    args = parser.parse_args()
    schema = EntitySchema(
        person_entity=args.person_entity, group_entities=tuple(args.group_entity)
    )
    engine = AxiomEngine(args.module, schema, rulespec_roots=args.rulespec_root)
    print(json.dumps(build_concept_coverage(engine), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
