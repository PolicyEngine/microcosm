"""Check that the current shared weighting reproduces every weighted run's weights.

The ``w_`` runs used microcosm#1104's ``target_loss_weights.py`` before it
merged. They stay valid for main as long as main's module gives each run
the same weights. The module's bytes can change without that: review fixes
that touch only other surfaces leave these weights alone. So this checks the
weights, not the file.

For every ``results/runs/w_*.json``, it rebuilds the run's training weights
the way ``sweep.py`` did (the run's training rows and family multipliers) and
its full-surface yardstick, with whichever module ``sweep.py`` loads now:
the package import, or the file named by ``MICROCOSM_TARGET_LOSS_WEIGHTS_FILE``.
It compares both loss-vector digests with the run's receipt. It writes
``results/weights_check.json`` and exits non-zero on any mismatch.

Run: ``uv run python experiments/us-acs-local-l2-basis-20260928/check_weights.py``
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sweep import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    SHARED_WEIGHTS_FILE_ENV,
    RunSpec,
    load_inputs,
    loss_vector_sha256,
    loss_weights_for,
    row_names,
    sha256,
    shared_weight_module,
    training_rows,
)

RUNS = HERE / "results" / "runs"
OUT = HERE / "results" / "weights_check.json"


def main() -> int:
    inputs = load_inputs(DEFAULT_CHECKPOINT, verify=True, need_registry=True)
    module = shared_weight_module()
    all_rows = list(range(inputs.matrix.shape[0]))
    full_names = row_names(inputs, all_rows)
    cache: dict[tuple, tuple[str, str]] = {}
    checked, mismatches = [], []
    for path in sorted(RUNS.glob("w_*.json")):
        payload = json.loads(path.read_text())
        spec = RunSpec.from_mapping(payload["spec"])
        recorded = payload["target_loss_weights"]
        key = (spec.holdout_fold, spec.family_loss_multipliers)
        if key not in cache:
            train = training_rows(inputs, spec.holdout_fold)
            weights = loss_weights_for(spec, inputs, train)
            cache[key] = (
                loss_vector_sha256(row_names(inputs, train), weights.train),
                loss_vector_sha256(full_names, weights.full),
            )
        train_digest, full_digest = cache[key]
        row = {
            "run_id": spec.run_id,
            "train_matches": train_digest == recorded["train"]["loss_vector_sha256"],
            "full_matches": full_digest
            == recorded["full_surface"]["loss_vector_sha256"],
            "recorded_module_sha256": recorded["module_sha256"],
        }
        checked.append(row)
        if not (row["train_matches"] and row["full_matches"]):
            mismatches.append(
                {
                    **row,
                    "train_digest_now": train_digest,
                    "train_digest_recorded": recorded["train"]["loss_vector_sha256"],
                }
            )
    result = {
        "ok": not mismatches and bool(checked),
        "module_file": module.__file__,
        "module_sha256": sha256(Path(module.__file__)),
        "module_loaded_from_file": bool(os.environ.get(SHARED_WEIGHTS_FILE_ENV)),
        "recorded_module_sha256": sorted(
            {row["recorded_module_sha256"] for row in checked}
        ),
        "registry_sha256": inputs.registry_receipt["verified_registry_sha256"],
        "n_runs": len(checked),
        "n_distinct_weightings": len(cache),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "mismatches"}, indent=1))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
