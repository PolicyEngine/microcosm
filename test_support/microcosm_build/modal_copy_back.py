"""A run directory as ``modal volume get`` copies it back from a Modal run.

The Modal runner (``tools/modal_us_stage.py``) mirrors a stage's state
directory to ``runs/<run_id>/state/`` on its runs volume and writes the
stage's receipt to ``runs/<run_id>/receipts/``. ``modal volume get
microcosm-us-stage-runs runs/<run_id> ./modal-runs/`` copies both back, as
``modal-runs/<run_id>/state/`` and ``modal-runs/<run_id>/receipts/``. The
receipts written here list every state file with its SHA-256 and size, as the
runner's do (``push_state``), and carry the fields the re-stage command reads.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from test_support.microcosm_build.us_modal_stage_plan_tool import plan_lib

MODAL_RUN_ID = "uk-rowwise-modal-20261104"
MODAL_STAGE = "build"


def copy_back(
    run_directory: Path,
    modal_runs: Path,
    *,
    run_id: str = MODAL_RUN_ID,
    name: str = "out",
) -> tuple[Path, Path]:
    """Move a finished run directory into a copied-back Modal run.

    The directory becomes ``<modal_runs>/<run_id>/state/<name>``, as a tool
    run on Modal leaves its output directory in the stage's state, and the
    stage's completed receipt describes the state. Returns the copied-back
    directory and the receipt's path.
    """
    run_root = modal_runs / run_id
    state = run_root / "state"
    state.mkdir(parents=True)
    copied = state / name
    shutil.move(str(run_directory), str(copied))
    return copied, write_receipt(run_root, run_id=run_id)


def write_receipt(
    run_root: Path,
    *,
    run_id: str = MODAL_RUN_ID,
    status: str = "COMPLETED",
    stopped_at_budget: bool = False,
) -> Path:
    """Write the receipt of a stage whose state is ``<run_root>/state``."""
    state = run_root / "state"
    outputs = []
    for relative in plan_lib.tree_listing(state):
        sha256, size = plan_lib.sha256_file(state / relative)
        outputs.append({"path": relative, "bytes": size, "sha256": sha256})
    receipt = {
        "schema": plan_lib.RECEIPT_SCHEMA,
        "status": status,
        "stage": MODAL_STAGE,
        "run_id": run_id,
        "returncode": 0 if status == "COMPLETED" else 1,
        "stopped_at_budget": stopped_at_budget,
        "finished_at": "2026-11-04T12:00:00Z",
        "outputs": outputs,
        "outputs_not_mirrored": [],
    }
    receipts = run_root / "receipts"
    receipts.mkdir(exist_ok=True)
    path = receipts / f"{MODAL_STAGE}-2026-11-04T110000Z.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return path


def rewrite_receipt(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    """Apply ``change`` to the receipt at ``path`` and write it back."""
    receipt = json.loads(path.read_text(encoding="utf-8"))
    change(receipt)
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
