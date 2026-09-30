#!/usr/bin/env bash
set -euo pipefail

if [[ -f engine-us-timings.md ]]; then
  : "${GITHUB_STEP_SUMMARY:?GITHUB_STEP_SUMMARY must be set}"
  cat engine-us-timings.md >> "$GITHUB_STEP_SUMMARY"
fi
