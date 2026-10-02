#!/usr/bin/env python3
"""Pass a public Microcosm export through the pinned Orrery parser."""

from __future__ import annotations

import subprocess
from pathlib import Path

from microcosm.graph import (
    Graph,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    orrery_json,
)

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "tools" / "orrery-contract" / "verify.mjs"


def contract_graph() -> Graph:
    """Use omitted population to cover the compiler-resolved API contract."""

    return Graph(
        "contract",
        sources=(SourceRef("fixture", "contract@1"),),
        nodes=(
            Node(
                "source",
                "contract.create@1",
                structural=StructuralDelta.CREATE,
                sources=("fixture",),
                outputs=(Owned("person", "amount", "float64"),),
            ),
            Node(
                "consumer",
                "contract.consume@1",
                inputs=(Slice("person", ("amount",)),),
                outputs=(Owned("person", "result", "float64"),),
            ),
        ),
    )


def main() -> int:
    subprocess.run(
        ["node", str(VERIFY)],
        input=orrery_json(contract_graph()),
        text=True,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
