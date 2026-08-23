# Candidate 25% current-main input audit

Date: 2026-08-23

Code: `d69131a3` (`origin/main` at lane start)

Result: **STOPPED — a required stage-2 Ledger artifact is absent**

This is a read-only host audit. No pool or release builder was run, no release
command was launched, no publication or promotion was attempted, and no
logbook chain state was touched.

## Current two-stage contract

Stage 1 is `tools/build_us_multispine_pool.py`. It accepts only local,
SHA-pinned raw inputs and writes a nullable input-only H5 plus manifest and gate
diagnostics; calibration is explicitly downstream
(`tools/build_us_multispine_pool.py:2-16`). The current exact-k stage-2 path is
`tools/build_us_fiscal_refresh_release.py`: `N` resolves to the authenticated
pool household count (`tools/build_us_fiscal_refresh_release.py:8227-8249`) and
uses a full-pool refit, while `57240` uses exact-k Sampford selection and refit
(`tools/build_us_fiscal_refresh_release.py:10126-10140,10207-10285`). Thus the
owner's dense and sparse artifacts require two separately pinned stage-2 runs
against the same authenticated stage-1 manifest.

## Stage 1 inputs verified on this host

The six required file-and-SHA pairs are declared at
`tools/build_us_multispine_pool.py:441-520`. They were copied exactly from the
current host queue at
`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/build-queue.sh:37-53`
and independently measured on this host.

| Required input | Exact local path | Measured SHA-256 | Requiring code |
|---|---|---|---|
| ASEC raw-stage H5 | `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/591-pawtyp-pool/asec-producer-checkpoints/asec_raw_stage.checkpoint.h5` | `51e9fafcd6f16140018fa90c7afbeb6d79008bfc8c122e437d23a399b30553fe` | `tools/build_us_multispine_pool.py:443-454` |
| ACS 2024 household ZIP | `/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0/csv_hus.zip` | `8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0` | `tools/build_us_multispine_pool.py:455-466` |
| ACS 2024 person ZIP | `/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894/csv_pus.zip` | `afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894` | `tools/build_us_multispine_pool.py:467-478` |
| ACS 2022 rent-donor H5 | `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/acs_2022.h5` | `0b319b496f19a6913066f9c5ea572edfda3d78a187be6f375846617d0b441bd4` | `tools/build_us_multispine_pool.py:479-490` |
| Processed PUF H5 | `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2024.h5` | `7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df` | `tools/build_us_multispine_pool.py:491-502` |
| Restricted source-year PUF CSV | `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2015.csv` | `0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df` | `tools/build_us_multispine_pool.py:503-514` |

The required 25% controls are accepted explicitly at
`tools/build_us_multispine_pool.py:531-552`. A future host command must use
`--sample-fraction 0.25 --sample-seed 578 --clone-attachment-fraction 1.0
--clone-attachment-seed 578`. The pool manifest SHA and authenticated
`publication_run_id` are stage-1 outputs, not static inputs available to this
no-build lane; stage 2 authenticates both at
`tools/build_us_fiscal_refresh_release.py:8227-8249`.

The off-chain requirement is material: the pool parser falls back from the
omitted `--logbook-prev-row-digest` to
`POPULACE_LOGBOOK_PREV_ROW_DIGEST`
(`tools/build_us_multispine_pool.py:554-560,4250-4262`). Any eventual host
invocation therefore must be prefixed with
`env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST` and must not pass the CLI flag.

## Stage 2: first missing required host input

Exact-k makes both Ledger hashes mandatory
(`tools/build_us_fiscal_refresh_release.py:1563-1566`) and passes them to the
artifact loader before target compilation
(`tools/build_us_fiscal_refresh_release.py:8358-8371`). The reviewed v9.4 facts
bytes do exist:

- Path:
  `/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl`
- Size: `131852600` bytes
- Measured SHA-256:
  `b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080`

However, they are a bare JSONL file. Neither of these required-manifest
candidates exists:

- `/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/manifest.json`
- `/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4/manifest.json`

Targeted read-only searches of the host build-runtime inputs, Ledger checkout,
and Hugging Face cache found no matching
`policyengine_ledger.consumer_artifact.v1` directory. A valid directory must
contain both `manifest.json` and `consumer_facts.jsonl`, and the loader verifies
the schema plus the facts hash (`packages/microcosm-build/src/microcosm/build/ledger_artifact.py:88-142`).
It explicitly rejects a manifest hash supplied with a bare feed
(`packages/microcosm-build/src/microcosm/build/ledger_artifact.py:148-158`).

There is therefore no exact local path or measured SHA-256 for the required
v9.4 Ledger `manifest.json`. Per the owner's binding instruction, the input
inventory stops here. No path or pin has been fabricated.

## Independent blockers already established in parallel

These do not relax or move the first-input stop above.

1. **No ratified stage-2 seed or `pi_hi`.** Exact-k requires an explicit
   nonnegative seed and a finite `pi_hi`
   (`tools/build_us_fiscal_refresh_release.py:1500-1528`). Current-main's
   runtime-contract authority says exact-k `k`, `pi_hi`, and `seed` have no
   default and must not be minted
   (`tools/us_bundle_generation/contracts.py:1-12`). The owner supplied the
   stage-1 sampling/attachment seeds, not these stage-2 run-request values.
   Copying values from a test or docstring would be tuning/fabrication.

2. **No builder-compatible incumbent diagnostics on the current target
   surface.** The only located release diagnostics are
   `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/buildo-run/sparse/releases/populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z/calibration_diagnostics.json`
   (19512724 bytes; SHA-256
   `870449b44e86b13b25bcea1a57f0e7af37f4d4db18be815eea3acdf9fe6eb40e`).
   They encode 5659 targets and target-surface SHA-256
   `49bb0fe3dfd4c399e7b3f900b0e5ba29d9d72413d9170dfc155a9fa5e91c6f6f`,
   and omit `build.target_loss_basis`. Current committed incumbent evidence
   has 32842 target rows, but
   `experiments/replacement_scorecard/incumbent_48b9d479.json` is a scorecard,
   not a `calibration_diagnostics.json` (31962593 bytes; SHA-256
   `b2ad1a07f9668bc5d796cc9de99ef12da781b1ee8163ea65781871a20da441c8`).
   The builder requires top-level diagnostic target rows, exact target-surface
   equality, and the same loss basis
   (`tools/build_us_fiscal_refresh_release.py:2676-2736,5862-5885,6666-6803`).

3. **The required full SCF extract is absent.** Stage 2 always resolves the
   full SCF donor for auto-loan inputs
   (`tools/build_us_fiscal_refresh_release.py:9572-9587`). The default local
   path `/Users/maxghenis/.cache/microcosm/scf/p22i6.dta` does not exist, and
   current-main has no member or archive SHA pin for a provisioning download
   (`packages/microcosm-build/src/microcosm/build/us_runtime/scf_auto_loans.py:67-77,162-216`).
   No path or SHA can be recorded for this missing file.

## Validation boundary

The release builder's complete parser exposes no `--dry-run`, `--validate-only`,
or `--validate-args` mode
(`tools/build_us_fiscal_refresh_release.py:829-1471`).
`tools/preflight_us_release_gates.py` is not an exact-k argument validator: it
requires both a bare `--base-h5` and a
`--selection-source-manifest`
(`tools/preflight_us_release_gates.py:42-117`), while exact-k requires a pool
manifest, rejects `--base-h5`, and forbids selection-source inputs
(`tools/build_us_fiscal_refresh_release.py:1500-1519,1573-1579`).

Because the mandatory Ledger manifest is absent, no complete stage-2 command
exists to print or validate. Consequently no `run-candidate.sh` was written,
and running `bash -n` or its requested dry-run would certify a fictional input
set rather than this host.
