#!/usr/bin/env bash

set -euo pipefail

readonly packages_dir="${1:-packages}"
readonly wheel_dir="${2:-dist}"

shopt -s nullglob
shards=("${packages_dir}"/*/)
if ((${#shards[@]} == 0)); then
  echo "::error::No package shards found under ${packages_dir}/"
  exit 1
fi

mkdir -p "${wheel_dir}"
existing_wheels=("${wheel_dir}"/*.whl)
if ((${#existing_wheels[@]} != 0)); then
  echo "::error::Wheel output directory is not empty: ${wheel_dir}/"
  exit 1
fi

for shard in "${shards[@]}"; do
  uv build --wheel "${shard}" --out-dir "${wheel_dir}"
done

for shard in "${shards[@]}"; do
  distribution=$(basename "${shard%/}")
  normalized_distribution=${distribution//-/_}
  wheels=("${wheel_dir}/${normalized_distribution}"-*.whl)
  if ((${#wheels[@]} != 1)); then
    echo "::error::Expected one wheel for ${distribution}; found ${#wheels[@]}"
    exit 1
  fi

  uvx --from check-wheel-contents==0.6.3 check-wheel-contents \
    --no-config \
    --src-dir "${shard}src" \
    "${wheels[0]}"
done

built_wheels=("${wheel_dir}"/*.whl)
if ((${#built_wheels[@]} != ${#shards[@]})); then
  echo "::error::Expected ${#shards[@]} wheels; found ${#built_wheels[@]}"
  exit 1
fi

printf 'Built and inspected %d wheels:\n' "${#built_wheels[@]}"
printf '  %s\n' "${built_wheels[@]}"
