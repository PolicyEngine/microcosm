"""Re-extract the PE-native surface, saving the UNSCALED (raw) matrix + targets
so calibration uses the eCPS +1 loss on raw magnitudes (not the comparison's
normalized ~0.01 scale, where the +1 regularizer breaks)."""
import os
import sys

import numpy as np

# The scoring-harness checkout that provides the extraction entrypoint; set
# SCORING_HARNESS_SRC to its src/ directory to re-run this snapshot.
sys.path.insert(0, os.environ["SCORING_HARNESS_SRC"])
from microplex_us.pipelines.ecps_replacement_comparison import _extract_pe_native_loss_inputs
from pathlib import Path

cand = "/Users/maxghenis/.claude-worktrees/microplex-spec-build/artifacts/spec_candidate_full_2024_v5_beats/candidate_timeperiod.h5"
repo = Path("/Users/maxghenis/.claude-worktrees/usdata-f7458313")
print("extracting (raw)...", flush=True)
inp = _extract_pe_native_loss_inputs(
    input_dataset_path=cand, period=2024,
    policyengine_us_data_repo=repo, policyengine_us_data_python=repo/".venv/bin/python",
    skip_tax_expenditure_targets=False,
)
A_s = np.asarray(inp["scaled_matrix"], np.float64)      # (n_hh, n_targets) scaled
b_s = np.asarray(inp["scaled_target"], np.float64)
w0  = np.asarray(inp["initial_weights"], np.float64)
b_raw = inp.get("unscaled_target")
scaling = inp.get("scaling")
names = [str(x) for x in inp["metadata"]["target_names"]]
print("has unscaled_target:", b_raw is not None, "| has scaling:", scaling is not None, flush=True)

if b_raw is not None and scaling is not None:
    b_raw = np.asarray(b_raw, np.float64); scaling = np.asarray(scaling, np.float64)
    # scaled = raw * scaling  =>  raw_A = scaled_A / scaling  (per target/column)
    A_raw = A_s / scaling[None, :]
    # verify: raw estimate at w0 ~ raw target
    est = A_raw.T @ w0
    print("raw b: median %.4g max %.4g | est/b median %.3f" % (np.median(b_raw), b_raw.max(), np.median((est+1)/(b_raw+1))), flush=True)
else:
    raise SystemExit("no unscaled arrays returned; need a different extraction path")

out = "/Users/maxghenis/.claude-worktrees/microplex-spec-build/artifacts/v5_target_surface_raw.npz"
np.savez_compressed(out, A=A_raw, b=b_raw, w0=w0, names=np.array(names, dtype=object))
loss = float(np.mean(((A_raw.T@w0 - b_raw + 1.0)/(b_raw + 1.0))**2))
print("SAVED", out, "| raw initial loss %.4f (should be small, ~eCPS-comparable)" % loss, flush=True)
