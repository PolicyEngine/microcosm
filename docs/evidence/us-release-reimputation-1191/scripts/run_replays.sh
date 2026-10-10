#!/bin/zsh
# Run the three replay groups one after another (each holds ~12 GB).
#
# Usage: run_replays.sh BUILDER_ROOT BASE_H5 RELEASE_H5 RELEASE_DIAGNOSTICS ASEC_2023_PPPUB23
#   BUILDER_ROOT  checkout of the commit that built the release (its .venv is used)
#   RELEASE_DIAGNOSTICS  the release's calibration_diagnostics.json
#   ASEC_2023_PPPUB23  the cached Census ASEC 2023 person file (pppub23.csv)
# On a shared host, set MAX_COMPRESSOR_GB to wait between groups until the macOS
# memory compressor holds less than that many GB.
set -eu
builder_root=$1 base_h5=$2 release_h5=$3 diagnostics=$4 asec=$5
here=${0:A:h}
cd "$here/.."
python="$builder_root/.venv/bin/python"
common=(--builder-root "$builder_root" --base-h5 "$base_h5" --release-h5 "$release_h5"
        --release-diagnostics "$diagnostics" --asec-2023-source "$asec")
export PYTHONHASHSEED=0

wait_for_memory() {
  [[ -z ${MAX_COMPRESSOR_GB:-} ]] && return 0
  while true; do
    gb=$(vm_stat | awk '/occupied by compressor/ {gsub("\\.","",$NF); print int($NF*16384/1e9)}')
    (( gb < MAX_COMPRESSOR_GB )) && return 0
    echo "compressor ${gb} GB >= ${MAX_COMPRESSOR_GB} GB; waiting" >&2
    sleep 60
  done
}

for group in release_order isolated attribution; do
  wait_for_memory
  case $group in
    attribution) variants=(--variant mass_only --variant ss_only --variant isolated_t4) ;;
    *) variants=(--variant $group) ;;
  esac
  "$python" -B scripts/replay_release_reimputation.py "${common[@]}" "${variants[@]}" \
    --out receipts/replay.$group.json > receipts/replay.$group.log 2>&1
done
