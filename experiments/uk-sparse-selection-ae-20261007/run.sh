#!/bin/bash
# microcosm#1124 steps 0, 1a and 1b. Run from the repository root of a tree whose
# code matches the baseline run's (see the README); one solve on the machine at a time.
set -euo pipefail
RUN=${RUN:?set RUN to the finished size build, e.g. runs/uk-local-k25-675d2a3fd}
CACHE=${CACHE:?set CACHE to a directory outside the repository and the run}
OUT=${OUT:?set OUT to a directory outside the repository and the run}
HERE=experiments/uk-sparse-selection-ae-20261007
TOOL=tools/run_uk_size_experiment.py
PY=.venv/bin/python
C=(--run-dir "$RUN" --cache-dir "$CACHE")
X=(--out "$OUT" --confirm-exclusive)

# Step 0: cache, the control (must reproduce the stored refit), the census.
$PY $TOOL build-cache "${C[@]}" --confirm-exclusive
$PY $TOOL control "${C[@]}" "${X[@]}"
$PY $TOOL census "${C[@]}" "${X[@]}"
# Step 1a: the pre-registered grid.
$PY $TOOL run "${C[@]}" "${X[@]}" --experiments $HERE/experiments.json --max-rss-gib 18
$PY $TOOL score "${C[@]}" --out "$OUT"
# Step 1b, read off 1a by the pre-registered rules: the A×E points, then the holdouts.
$PY $TOOL plan-step1b --out "$OUT" --stage ae --to "$OUT/step1b_ae.json"
$PY $TOOL run "${C[@]}" "${X[@]}" --experiments "$OUT/step1b_ae.json" --max-rss-gib 18
$PY $TOOL score "${C[@]}" --out "$OUT"
$PY $TOOL plan-step1b --out "$OUT" --stage holdout --to "$OUT/step1b_holdout.json"
$PY $TOOL run "${C[@]}" "${X[@]}" --experiments "$OUT/step1b_holdout.json" --max-rss-gib 18
# Publish the disclosure-controlled aggregates and write the README tables.
$PY $TOOL publish --out "$OUT" --to $HERE/results
$PY $HERE/analyze.py
