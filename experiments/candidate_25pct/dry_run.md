# Candidate 25% dry-run receipt

Date: 2026-08-23  
Status: **NOT RUN — stopped at a missing required stage-2 input**

The requested dry-run did not run because current exact-k stage 2 requires a
manifest-pinned Ledger consumer-artifact directory, while this host contains
only the bare v9.4 JSONL feed. Exact-k mandates both the facts and manifest
SHA-256 pins (`tools/build_us_fiscal_refresh_release.py:1563-1566`), and the
loader rejects a manifest pin for a bare facts file
(`packages/microcosm-build/src/microcosm/build/ledger_artifact.py:148-158`).

Observed facts path:

`/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl`

Observed facts SHA-256:

`b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080`

Missing required input: the matching
`policyengine_ledger.consumer_artifact.v1/manifest.json` and therefore its exact
local path and measured SHA-256.

Validation disposition:

- `run-candidate.sh`: not written; a runnable command would require a fabricated
  manifest pin.
- `bash -n`: not run because no honest script exists.
- Script `--dry-run`: not run for the same reason; no partial command was
  represented as executable.
- Release-builder native validation: unavailable; the builder offers no
  dry-run/validate-arguments mode.
- `tools/preflight_us_release_gates.py`: not applicable to exact-k because it
  requires `--base-h5` plus `--selection-source-manifest`
  (`tools/preflight_us_release_gates.py:42-117`), inputs exact-k rejects
  (`tools/build_us_fiscal_refresh_release.py:1500-1519,1573-1579`).

The complete verified-input receipt and the independently discovered remaining
blockers are in `experiments/candidate_25pct/input_audit.md`.
