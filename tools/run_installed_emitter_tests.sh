#!/usr/bin/env bash
set -euo pipefail

# Install microcosm-emitter at each release tools/installed_emitter_releases.py
# names (the minimum release builds accept, then the newest release pinned in
# tools/installed-emitter/requirements.txt), each into its own uv tool
# directory, and run the installed-emitter tests once for each. Every release
# is tested even when an earlier one fails; the script fails if any did. The
# tests send the emitter's Hugging Face Hub requests to a loopback stand-in, so
# they need no credential. The optional argument is the directory that
# receives the tool directories; by default it is a new temporary directory.

tool_root="${1:-$(mktemp -d)}"

releases=()
while IFS= read -r release; do
  if [[ -n "$release" ]]; then releases+=("$release"); fi
done < <(uv run --no-sync python tools/installed_emitter_releases.py)
if ((${#releases[@]} == 0)); then
  echo "::error::tools/installed_emitter_releases.py named no microcosm-emitter release"
  exit 1
fi

files=()
while IFS= read -r file; do
  if [[ -n "$file" ]]; then files+=("$file"); fi
done < <(uv run --no-sync python tools/ci_test_plan.py list-job installed-emitter)
if ((${#files[@]} == 0)); then
  echo "::error::tools/ci_test_plan.py listed no test files for the installed-emitter job"
  exit 1
fi

# Install one release and run the tests with it; return nonzero if either fails.
test_release() {
  local release=$1
  local tool_directory="$tool_root/microcosm-emitter-$release"
  UV_TOOL_DIR="$tool_directory/tools" UV_TOOL_BIN_DIR="$tool_directory/bin" \
    uv tool install "microcosm-emitter==$release" \
    --constraints tools/installed-emitter/constraints.txt || return 1
  echo "Running the installed-emitter tests with microcosm-emitter $release"
  MICROCOSM_EMITTER_COMMAND="$tool_directory/bin/microcosm-emitter" \
    INSTALLED_EMITTER_VERSION="$release" \
    uv run --no-sync pytest "${files[@]}" --run-integration \
    -v --tb=short --maxfail=1 --durations=25 -p no:cacheprovider || return 1
}

failed=()
for release in "${releases[@]}"; do
  if ! test_release "$release"; then failed+=("$release"); fi
done
if ((${#failed[@]} != 0)); then
  echo "::error::The installed-emitter tests failed with microcosm-emitter ${failed[*]}"
  exit 1
fi
