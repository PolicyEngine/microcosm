#!/bin/bash
# Run time_slices.py against a checkout at the scorer pin. Set ROOT (the
# checkout, which the scorer's `git rev-parse HEAD` needs as the working
# directory), PYTHON (a venv with the pin's locked dependencies), INCUMBENT,
# CANDIDATE, LEDGER and OUT.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
exec "$PYTHON" -I -B "$HERE/time_slices.py" "$ROOT" "$INCUMBENT" "$CANDIDATE" "$LEDGER" "$OUT"
