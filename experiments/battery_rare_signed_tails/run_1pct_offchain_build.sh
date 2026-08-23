#!/bin/bash
# Off-chain 1% verification build for the rare signed-tail lane.
#
# Usage: run_1pct_offchain_build.sh <worktree> <out-dir>
#
# Lane rules enforced here:
#   - at most a 1% sample (--sample-fraction 0.01, fixed);
#   - OFF-CHAIN: no --logbook-prev-row-digest, POPULACE_LOGBOOK_PREV_ROW_DIGEST
#     unset, and logbook-pending-chain.txt is never read or written (the
#     genesis-null logbook row stays in this run's own spool directory);
#   - serial queue: refuses to start while any other build_us_multispine_pool
#     process is alive;
#   - RSS ceiling: samples the build's RSS every 20 s to rss_samples.csv and
#     kills the build if it exceeds 15 GiB.
set -u

WORKTREE="$1"
OUT_DIR="$2"
LOG="${OUT_DIR}/build.log"
MEMCSV="${OUT_DIR}/rss_samples.csv"
RSS_LIMIT_KIB=$((15 * 1024 * 1024))

if pgrep -f "build_us_multispine_pool.py" > /dev/null 2>&1; then
  echo "REFUSE: a build_us_multispine_pool process is already running" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
cd "${WORKTREE}"

export ASEC_H5=/Users/maxghenis/PolicyEngine/_buildo-runtime/out/591-pawtyp-pool/asec-producer-checkpoints/asec_raw_stage.checkpoint.h5
export ASEC_SHA=51e9fafcd6f16140018fa90c7afbeb6d79008bfc8c122e437d23a399b30553fe
export ACS_H_ZIP=/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0/csv_hus.zip
export ACS_H_SHA=8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0
export ACS_P_ZIP=/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894/csv_pus.zip
export ACS_P_SHA=afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894
export RENT_H5=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/acs_2022.h5
export RENT_SHA=0b319b496f19a6913066f9c5ea572edfda3d78a187be6f375846617d0b441bd4
export PUF_H5=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2024.h5
export PUF_SHA=7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df
export PUF_CSV=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2015.csv
export PUF_CSV_SHA=0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df

echo "1pct off-chain build: worktree=${WORKTREE} HEAD=$(git rev-parse --short HEAD) at $(date -u +%FT%TZ)" | tee "${LOG}"
echo "epoch_s,rss_kib" > "${MEMCSV}"

env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST \
  "${WORKTREE}/.venv/bin/python" tools/build_us_multispine_pool.py \
  --sample-fraction 0.01 \
  --sample-seed 578 \
  --clone-attachment-fraction 1.0 \
  --clone-attachment-seed 578 \
  --asec-raw-stage-h5 "${ASEC_H5}" \
  --asec-raw-stage-h5-sha256 "${ASEC_SHA}" \
  --acs-household-zip "${ACS_H_ZIP}" \
  --acs-household-zip-sha256 "${ACS_H_SHA}" \
  --acs-person-zip "${ACS_P_ZIP}" \
  --acs-person-zip-sha256 "${ACS_P_SHA}" \
  --acs-rent-h5 "${RENT_H5}" \
  --acs-rent-h5-sha256 "${RENT_SHA}" \
  --puf-h5 "${PUF_H5}" \
  --puf-h5-sha256 "${PUF_SHA}" \
  --puf-source-year-csv "${PUF_CSV}" \
  --puf-source-year-csv-sha256 "${PUF_CSV_SHA}" \
  --out "${OUT_DIR}/populace_us_2024_stacked_pool_1pct.h5" \
  >> "${LOG}" 2>&1 &
BUILD_PID=$!

while kill -0 "${BUILD_PID}" 2>/dev/null; do
  RSS_KIB=$(ps -o rss= -p "${BUILD_PID}" | tr -d ' ')
  if [ -n "${RSS_KIB}" ]; then
    echo "$(date +%s),${RSS_KIB}" >> "${MEMCSV}"
    if [ "${RSS_KIB}" -gt "${RSS_LIMIT_KIB}" ]; then
      echo "RSS ${RSS_KIB} KiB exceeded 15 GiB ceiling; killing build" | tee -a "${LOG}"
      kill "${BUILD_PID}"
      wait "${BUILD_PID}"
      exit 3
    fi
  fi
  sleep 20
done
wait "${BUILD_PID}"
RC=$?
echo "build exit ${RC} at $(date -u +%FT%TZ); peak RSS KiB: $(cut -d, -f2 "${MEMCSV}" | sort -n | tail -1)" | tee -a "${LOG}"
exit "${RC}"
