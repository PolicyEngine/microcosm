#!/usr/bin/env bash
set -euo pipefail

# Install microcosm-emitter at the minimum release builds accept
# (MINIMUM_EMITTER_VERSION) and at the newest release pinned in
# tools/installed-emitter/requirements.txt, each into its own uv tool
# directory, and run the installed-emitter tests once for each. The tests send
# the emitter's Hugging Face Hub requests to a loopback stand-in, so they need
# no credential. The optional argument is the directory that receives the tool
# directories; by default it is a new temporary directory.

tool_root="${1:-$(mktemp -d)}"
minimum=$(
  uv run --no-sync python -c \
    'from microcosm.build.telemetry_emitter import MINIMUM_EMITTER_VERSION as v; print(v)'
)
newest=$(sed -n 's/^microcosm-emitter==//p' tools/installed-emitter/requirements.txt)
if [[ -z "$newest" ]]; then
  echo "::error::tools/installed-emitter/requirements.txt pins no microcosm-emitter release"
  exit 1
fi
if [[ "$(printf '%s\n' "$minimum" "$newest" | sort -V | head -n 1)" != "$minimum" ]]; then
  echo "::error::The pinned microcosm-emitter $newest is older than MINIMUM_EMITTER_VERSION $minimum; raise it in tools/installed-emitter/requirements.txt"
  exit 1
fi
versions=("$minimum")
if [[ "$newest" != "$minimum" ]]; then
  versions+=("$newest")
fi

files=()
while IFS= read -r file; do files+=("$file"); done < <(
  uv run --no-sync python tools/ci_test_plan.py list-job installed-emitter
)

for version in "${versions[@]}"; do
  tool_directory="$tool_root/microcosm-emitter-$version"
  UV_TOOL_DIR="$tool_directory/tools" UV_TOOL_BIN_DIR="$tool_directory/bin" \
    uv tool install "microcosm-emitter==$version" \
    --constraints tools/installed-emitter/constraints.txt
  echo "Running the installed-emitter tests with microcosm-emitter $version"
  MICROCOSM_EMITTER_COMMAND="$tool_directory/bin/microcosm-emitter" \
    INSTALLED_EMITTER_VERSION="$version" \
    uv run --no-sync pytest "${files[@]}" --run-integration \
    -v --tb=short --maxfail=1 --durations=25 -p no:cacheprovider
done
