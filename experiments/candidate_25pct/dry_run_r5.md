# Candidate 25% owner-ruling-A dry-run receipt, round 5

Date: 2026-08-24

Launcher commit tested: `52a2bcfbd98d55444ec55abaffce41cbd773a184`

Launcher SHA-256 tested:
`484874a22d63e8a0faf16f3eb504eed152a0a2d5993d9b6db5d5e18aeee69838`

Outcome: **PASS (exit 0)**. The dry-run resolved and SHA-checked every
immutable pool, release, and scoring input, including the sparse schema-3 SSI
prior basis; verified the full SCF Stata header; checked every used parser flag;
proved the sparse command obeys owner ruling A and the builder's literal L0
default; and printed the complete pool, dense, sparse, and scorer commands. It
did not invoke a pool/release builder or scorer and did not mutate publication,
promotion, staging, launchd, or candidate output state.

The external candidate root already carries code pin `8fa966d9`, while this
receipt tests `52a2bcfb`. Dry-run reported
`execute_mode_conflict=true dry_run_mutation=false` and continued read-only;
execute mode deliberately remains fail-closed until the owner authorizes
cleanup or a fresh output root. The candidate pool directory remained empty,
the sparse root remained absent, and the candidate code pin, main log, external
launcher, and dense release-ID file retained their pre-run sizes and mtimes.

A pre-existing, independently managed `candidate-25/smoke/pool.log` changed
during the earlier review window. The committed launcher never reads or writes
the `smoke/` subtree; this concurrent external activity is not attributed to
the dry-run. No pool or release build was launched by this task.

## Invocation

```text
$ ./experiments/candidate_25pct/run-candidate.sh --dry-run
exit 0
```

## Full stdout

```text
[2026-08-24T12:48:46Z] RUN START mode=dry-run label=one-surface+pkg3,legacy-release-arm,not-exact-k-certified
[2026-08-24T12:48:46Z] LAUNCHER OK sha256=484874a22d63e8a0faf16f3eb504eed152a0a2d5993d9b6db5d5e18aeee69838 invoked_path=./experiments/candidate_25pct/run-candidate.sh canonical_path=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/experiments/candidate_25pct/run-candidate.sh
[2026-08-24T12:48:46Z] CODE OK commit=52a2bcfbd98d55444ec55abaffce41cbd773a184 worktree=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=pool flag=--asec-raw-stage-h5 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=pool flag=--asec-raw-stage-h5-sha256 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=pool flag=--acs-household-zip source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=pool flag=--acs-household-zip-sha256 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=pool flag=--acs-person-zip source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=pool flag=--acs-person-zip-sha256 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=pool flag=--acs-rent-h5 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=pool flag=--acs-rent-h5-sha256 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=pool flag=--puf-h5 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=pool flag=--puf-h5-sha256 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=pool flag=--puf-source-year-csv source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=pool flag=--puf-source-year-csv-sha256 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=pool flag=--sample-fraction source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=pool flag=--sample-seed source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=pool flag=--clone-attachment-fraction source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=pool flag=--clone-attachment-seed source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=pool flag=--checkpoint-root source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=pool flag=--out source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--base-h5 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--dense-default-dataset source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--ledger-facts source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--ledger-facts-sha256 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--export-input-mass-reference-h5 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--asec-2023-weeks-unemployed-source source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--scf-summary-extract source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--scf-full-extract source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--sipp-tip-donor source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--sipp-vehicle-donor source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--org-wages-donor source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--ssi-take-up-prior-weight-basis source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--ssi-take-up-prior-weight-basis-sha256 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--checkpoint-root source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--release-id source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--seed source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--epochs source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--skip-reform-validation source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--no-staging source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=dense flag=--out source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=sparse flag=--l0-refit-lambda-share source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=scorer flag=--incumbent source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/score_us_release_head_to_head.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=scorer flag=--candidate source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/score_us_release_head_to_head.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=scorer flag=--ledger-facts source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/score_us_release_head_to_head.py
[2026-08-24T12:48:46Z] PARSER FLAG OK surface=scorer flag=--out-prefix source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/score_us_release_head_to_head.py
[2026-08-24T12:48:46Z] CODE PIN OBSERVED recorded_commit=8fa966d9398efc3a445845051501082295a244c9 planned_commit=52a2bcfbd98d55444ec55abaffce41cbd773a184 path=/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/code.commit execute_mode_conflict=true dry_run_mutation=false
[2026-08-24T12:48:48Z] INPUT OK role=asec-raw-stage-h5 size=653410492 sha256=51e9fafcd6f16140018fa90c7afbeb6d79008bfc8c122e437d23a399b30553fe path=/Users/maxghenis/PolicyEngine/_buildo-runtime/out/591-pawtyp-pool/asec-producer-checkpoints/asec_raw_stage.checkpoint.h5
[2026-08-24T12:48:49Z] INPUT OK role=acs-household-zip size=251500587 sha256=8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0 path=/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0/csv_hus.zip
[2026-08-24T12:48:50Z] INPUT OK role=acs-person-zip size=602847146 sha256=afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894 path=/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894/csv_pus.zip
[2026-08-24T12:48:52Z] INPUT OK role=acs-rent-h5 size=472220686 sha256=0b319b496f19a6913066f9c5ea572edfda3d78a187be6f375846617d0b441bd4 path=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/acs_2022.h5
[2026-08-24T12:48:53Z] INPUT OK role=puf-h5 size=316939164 sha256=7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df path=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2024.h5
[2026-08-24T12:48:53Z] INPUT OK role=puf-source-year-csv size=126034649 sha256=0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df path=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2015.csv
[2026-08-24T12:48:53Z] INPUT OK role=ledger-v9.4 size=131852600 sha256=b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080 path=/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl
[2026-08-24T12:48:55Z] INPUT OK role=export-input-mass-reference size=354374956 sha256=c2065b642ab00da74746afdfd9f06890e5f32f9b10bd6610ff236452d40f39c5 path=/Users/maxghenis/PolicyEngine/_buildg-runtime/forensics/populace_us_2024.h5
[2026-08-24T12:48:55Z] INPUT OK role=scf-summary-extract size=24904185 sha256=6b8dd2d935a76ed225ddebc80fb2db22a467f0c80d9a1acaa67b4584aa4bafd1 path=/Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/scf_cache/rscfp2022.dta
[2026-08-24T12:48:55Z] INPUT OK role=scf-full-extract size=236952250 sha256=61e2fceb1594e4009eb996d6e25d38a5d8e4874930fc2bfce3c87ffa6946ad0a path=/Users/maxghenis/.cache/microcosm/scf/p22i6.dta
[2026-08-24T12:48:55Z] INPUT HEADER OK role=scf-full-extract format=stata-dta release=118 byteorder=LSF path=/Users/maxghenis/.cache/microcosm/scf/p22i6.dta
[2026-08-24T12:48:56Z] INPUT OK role=asec-2023-weeks-source size=150165063 sha256=d2e000250782adfbdd7f29c82b66d866591a30f0d330496698ec19f9c784ce11 path=/Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/asec_education/asecpub23csv.zip
[2026-08-24T12:49:07Z] INPUT OK role=sipp-full-donor size=3726010471 sha256=5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2 path=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/pu2023.csv
[2026-08-24T12:49:07Z] INPUT OK role=sipp-tips-donor size=65649523 sha256=1f0bcb8e045ef1118e8eba4b4a2997bdaaf947bd0dd09d41fa7c7d5657a3d7d5 path=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/pu2023_slim.csv
[2026-08-24T12:49:07Z] INPUT OK role=cps-org-wages-compressed size=1677834 sha256=66fa5b6aa4087413b691038767b51f603281ff55411b58259922f78e67460372 path=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/census_cps_org_2024_wages.csv.gz
[2026-08-24T12:49:07Z] INPUT OK role=packaged-cd-crosswalk size=77935 sha256=c7cb040b1f57ca2ea2adcbfe60cc2b250ca23acbc4b640cd421e766fa54c1aec path=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/packages/microcosm-build/src/microcosm/build/us_runtime/data/congressional_district_vintage_crosswalk.csv
[2026-08-24T12:49:07Z] INPUT OK role=dense-ssi-prior-weight-basis size=3957 sha256=56118bde095b8ef2559a26a3478ff5f8b61939eca402dffcec61189e7de631e3 path=/Users/maxghenis/PolicyEngine/_buildo-runtime/out/buildo-run/dense-p2/releases/populace-us-2024-buildp-dense-cae8640-20260728T050443Z/us_ssi_take_up.json
[2026-08-24T12:49:07Z] INPUT OK role=sparse-ssi-prior-weight-basis size=4782 sha256=25fe8af50a99d717f3408b2de7f0849d2307d4f05b1a7d55d2703999002fff0a path=/Users/maxghenis/PolicyEngine/_buildo-runtime/inputs/attempt6_basis_schema3_seed.json
[2026-08-24T12:49:07Z] INPUT OK role=incumbent-evidence size=31962593 sha256=b2ad1a07f9668bc5d796cc9de99ef12da781b1ee8163ea65781871a20da441c8 path=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/experiments/replacement_scorecard/incumbent_48b9d479.json
[2026-08-24T12:49:09Z] INPUT OK role=incumbent-h5 size=462915783 sha256=48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e path=/Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/26dcad66867687f15735dc4926523e3741920836/populace_us_2024.h5
[2026-08-24T12:49:09Z] DENSE RELEASE ID value=populace-us-2024-onesurface-pkg3-legacy-dense-52a2bcfb-20260824T124846Z state_file=/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/release-dense/release_id.txt
[2026-08-24T12:49:09Z] SPARSE RELEASE ID value=populace-us-2024-onesurface-pkg3-legacy-sparse-52a2bcfb-20260824T124846Z state_file=/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/release-sparse/release_id.txt
[2026-08-24T12:49:09Z] OWNER RULING A ACTIVE sparse_path=legacy-cold-l0 default_lambda_share=0.8 realized_count=non-exact selection_source=none exact_k=none pi_hi=none keogh_mass_protection=omitted zero_operator_waivers=true
[2026-08-24T12:49:09Z] SPARSE LEGACY CONTRACT OK l0_lambda_share=0.8 source=none exact_k=none pi_hi=none selection_mass_protection=none operator_waivers=none seed=0 epochs=6000 no_staging=true
[2026-08-24T12:49:09Z] PRECONDITION PLAN stage=pool poll_seconds=300 need_reclaimable_gib=85 checks=no-pool-or-release-builder,AC-power,go-marker:/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/.max-go
COMMAND stage=pool /usr/bin/time -l /usr/bin/env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST /Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/.venv/bin/python /Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py --sample-fraction 0.25 --sample-seed 578 --clone-attachment-fraction 1.0 --clone-attachment-seed 578 --asec-raw-stage-h5 /Users/maxghenis/PolicyEngine/_buildo-runtime/out/591-pawtyp-pool/asec-producer-checkpoints/asec_raw_stage.checkpoint.h5 --asec-raw-stage-h5-sha256 51e9fafcd6f16140018fa90c7afbeb6d79008bfc8c122e437d23a399b30553fe --acs-household-zip /Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0/csv_hus.zip --acs-household-zip-sha256 8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0 --acs-person-zip /Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894/csv_pus.zip --acs-person-zip-sha256 afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894 --acs-rent-h5 /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/acs_2022.h5 --acs-rent-h5-sha256 0b319b496f19a6913066f9c5ea572edfda3d78a187be6f375846617d0b441bd4 --puf-h5 /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2024.h5 --puf-h5-sha256 7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df --puf-source-year-csv /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2015.csv --puf-source-year-csv-sha256 0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df --checkpoint-root /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/pool/checkpoints --out /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/pool/pool.h5
[2026-08-24T12:49:09Z] POOL MANIFEST PIN DEFERRED dynamic_output=/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/pool/pool.manifest.json release-wrapper-will-record-and-consume-full-sha256
[2026-08-24T12:49:09Z] PRECONDITION PLAN stage=release-dense poll_seconds=300 need_reclaimable_gib=85 checks=no-pool-or-release-builder,AC-power,go-marker:/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/.max-go
COMMAND stage=release-dense /usr/bin/time -l /usr/bin/env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST /Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/.venv/bin/python /Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py --base-h5 /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/pool/pool.h5 --dense-default-dataset --ledger-facts /Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl --ledger-facts-sha256 b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080 --export-input-mass-reference-h5 /Users/maxghenis/PolicyEngine/_buildg-runtime/forensics/populace_us_2024.h5 --asec-2023-weeks-unemployed-source /Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/asec_education/asecpub23csv.zip --scf-summary-extract /Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/scf_cache/rscfp2022.dta --scf-full-extract /Users/maxghenis/.cache/microcosm/scf/p22i6.dta --sipp-tip-donor /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/pu2023_slim.csv --sipp-vehicle-donor /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/pu2023.csv --org-wages-donor /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/census_cps_org_2024_wages.csv.gz --ssi-take-up-prior-weight-basis /Users/maxghenis/PolicyEngine/_buildo-runtime/out/buildo-run/dense-p2/releases/populace-us-2024-buildp-dense-cae8640-20260728T050443Z/us_ssi_take_up.json --ssi-take-up-prior-weight-basis-sha256 56118bde095b8ef2559a26a3478ff5f8b61939eca402dffcec61189e7de631e3 --seed 0 --epochs 3000 --checkpoint-root /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/release-dense/checkpoints --release-id populace-us-2024-onesurface-pkg3-legacy-dense-52a2bcfb-20260824T124846Z --out /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/release-dense --skip-reform-validation --no-staging
[2026-08-24T12:49:09Z] PRECONDITION PLAN stage=release-sparse poll_seconds=300 need_reclaimable_gib=85 checks=no-pool-or-release-builder,AC-power,go-marker:/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/.max-go
COMMAND stage=release-sparse /usr/bin/time -l /usr/bin/env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST /Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/.venv/bin/python /Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py --base-h5 /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/pool/pool.h5 --ledger-facts /Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl --ledger-facts-sha256 b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080 --export-input-mass-reference-h5 /Users/maxghenis/PolicyEngine/_buildg-runtime/forensics/populace_us_2024.h5 --asec-2023-weeks-unemployed-source /Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/asec_education/asecpub23csv.zip --scf-summary-extract /Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/scf_cache/rscfp2022.dta --scf-full-extract /Users/maxghenis/.cache/microcosm/scf/p22i6.dta --sipp-tip-donor /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/pu2023_slim.csv --sipp-vehicle-donor /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/pu2023.csv --org-wages-donor /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/census_cps_org_2024_wages.csv.gz --ssi-take-up-prior-weight-basis /Users/maxghenis/PolicyEngine/_buildo-runtime/inputs/attempt6_basis_schema3_seed.json --ssi-take-up-prior-weight-basis-sha256 25fe8af50a99d717f3408b2de7f0849d2307d4f05b1a7d55d2703999002fff0a --seed 0 --epochs 6000 --checkpoint-root /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/release-sparse/checkpoints --release-id populace-us-2024-onesurface-pkg3-legacy-sparse-52a2bcfb-20260824T124846Z --out /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/release-sparse --skip-reform-validation --no-staging
[2026-08-24T12:49:09Z] INCUMBENT EVIDENCE path=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/experiments/replacement_scorecard/incumbent_48b9d479.json sha256=b2ad1a07f9668bc5d796cc9de99ef12da781b1ee8163ea65781871a20da441c8 (evidence only; scorer consumes the H5)
[2026-08-24T12:49:09Z] DRY-RUN COMPLETE no pool/release builder, scorer, publication, promotion, staging, or launchd mutation ran
[2026-08-24T12:49:09Z] DENSE ARTIFACT planned_path=/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/release-dense/artifacts/populace_us_2024.h5 sha256=pending-stage-2a
[2026-08-24T12:49:09Z] SPARSE ARTIFACT planned_path=/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/release-sparse/artifacts/populace_us_2024.h5 sha256=pending-stage-2b
SCORER COMMAND dense /usr/bin/env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST /Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/.venv/bin/python /Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/score_us_release_head_to_head.py --incumbent /Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/26dcad66867687f15735dc4926523e3741920836/populace_us_2024.h5 --candidate /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/release-dense/artifacts/populace_us_2024.h5 --ledger-facts /Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl --out-prefix /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/scores/dense-head-to-head
SCORER COMMAND sparse /usr/bin/env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST /Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/.venv/bin/python /Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/score_us_release_head_to_head.py --incumbent /Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/26dcad66867687f15735dc4926523e3741920836/populace_us_2024.h5 --candidate /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/release-sparse/artifacts/populace_us_2024.h5 --ledger-facts /Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl --out-prefix /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/scores/sparse-head-to-head
```

## Syntax and side-effect checks

```text
$ bash -n experiments/candidate_25pct/run-candidate.sh
exit 0

$ shellcheck experiments/candidate_25pct/run-candidate.sh
exit 0

$ test -e /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/release-sparse
exit 1
```

- No pool, release, or scorer command was invoked by the dry-run.
- No candidate pool file, sparse directory, checkpoint, state pin, log, RSS
  CSV, score, or launchd label was created or changed by the dry-run.
- The pre-existing code pin remained
  `8fa966d9398efc3a445845051501082295a244c9`.
- No publication, promotion, push, tuning, or staging occurred.
- `POPULACE_LOGBOOK_PREV_ROW_DIGEST` remained unset and no pending-chain state
  was read or written.
