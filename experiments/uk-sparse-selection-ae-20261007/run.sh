#!/bin/bash
# microcosm#1124 step 0 and step 1a. Run from the repository root of a tree whose
# code matches the baseline run's (see the README); one solve on the machine at a time.
set -euo pipefail
RUN=${RUN:?set RUN to the finished size build, e.g. runs/uk-local-k25-675d2a3fd}
CACHE=${CACHE:?set CACHE to a directory outside the repository and the run}
OUT=${OUT:?set OUT to a directory outside the repository and the run}
HERE=experiments/uk-sparse-selection-ae-20261007
TOOL=tools/run_uk_size_experiment.py

.venv/bin/python $TOOL build-cache --run-dir "$RUN" --cache-dir "$CACHE" --confirm-exclusive
.venv/bin/python $TOOL control --run-dir "$RUN" --cache-dir "$CACHE" --out "$OUT" --confirm-exclusive
.venv/bin/python $TOOL census --run-dir "$RUN" --cache-dir "$CACHE" --out "$OUT" --confirm-exclusive
.venv/bin/python $TOOL run --run-dir "$RUN" --cache-dir "$CACHE" --out "$OUT" \
  --experiments $HERE/experiments.json --confirm-exclusive --max-rss-gib 18
.venv/bin/python $TOOL score --run-dir "$RUN" --cache-dir "$CACHE" --out "$OUT"
.venv/bin/python $TOOL publish --out "$OUT" --to $HERE/results
.venv/bin/python $HERE/analyze.py
