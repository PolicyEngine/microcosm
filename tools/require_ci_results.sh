#!/usr/bin/env bash
set -euo pipefail

fail=0

require_success() {
  if [[ "$2" != "success" ]]; then
    echo "::error::$1 = $2 (expected success)"
    fail=1
  fi
}

require_conditional() {
  expected=$2
  result=$3
  if [[ "$expected" == "true" && "$result" != "success" ]]; then
    echo "::error::$1 = $result (expected success)"
    fail=1
  elif [[ "$expected" != "true" && "$result" != "skipped" ]]; then
    echo "::error::$1 = $result (expected skipped)"
    fail=1
  fi
}

require_success changes "$CHANGES_RESULT"
require_success lint "$LINT_RESULT"
require_success engine-free "$ENGINE_FREE_RESULT"
require_success integration-uk "$INTEGRATION_UK_RESULT"
require_success wheels "$WHEELS_RESULT"
require_conditional engine-us "$EXPECT_US" "$ENGINE_US_RESULT"
require_conditional engine-uk "$EXPECT_UK" "$ENGINE_UK_RESULT"

exit "$fail"
