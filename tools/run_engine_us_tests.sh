#!/usr/bin/env bash
set -euo pipefail

/usr/bin/time -v uv run --no-sync python \
  -m tools.run_engine_test_categories \
  --json-report engine-us-timings.json \
  --markdown-report engine-us-timings.md
