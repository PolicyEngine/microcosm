# Candidate 25% legacy-arm dry-run receipt, round 4

Date: 2026-08-23

Launcher commit tested: `6327ec0208803a811f59783723a2cd4df5824ad2`

Launcher SHA-256 tested:
`94f113bf1d3d7fc58c3973549b66a8aef9f2a3d55931c5f0ea60560e65f16a1a`

Outcome: **PASS (exit 0)**. The dry-run resolved and SHA-checked every
immutable pool, release, and scoring input; verified the full SCF Stata header;
checked every used current-parser flag; and printed the complete pool, dense,
and dense-scorer commands. It did not invoke a builder, selection tool, release,
scorer, publication, promotion, staging, or launchd mutation.

The requested external installation path was not writable in the managed
session: attempting to create
`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25` returned
`Operation not permitted`. The real dry-run therefore invoked the committed
canonical executable directly. The external candidate root was still absent
afterward. No external launcher is claimed to exist.

## Invocation

```text
$ ./experiments/candidate_25pct/run-candidate.sh --dry-run
exit 0
```

## Full stdout

```text
[2026-08-23T13:25:13Z] RUN START mode=dry-run label=one-surface+pkg3,legacy-release-arm,not-exact-k-certified
[2026-08-23T13:25:13Z] LAUNCHER OK sha256=94f113bf1d3d7fc58c3973549b66a8aef9f2a3d55931c5f0ea60560e65f16a1a invoked_path=./experiments/candidate_25pct/run-candidate.sh canonical_path=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/experiments/candidate_25pct/run-candidate.sh
[2026-08-23T13:25:13Z] CODE OK commit=6327ec0208803a811f59783723a2cd4df5824ad2 worktree=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook
[2026-08-23T13:25:13Z] PARSER FLAG OK surface=pool flag=--asec-raw-stage-h5 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-23T13:25:13Z] PARSER FLAG OK surface=pool flag=--asec-raw-stage-h5-sha256 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-23T13:25:13Z] PARSER FLAG OK surface=pool flag=--acs-household-zip source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-23T13:25:13Z] PARSER FLAG OK surface=pool flag=--acs-household-zip-sha256 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-23T13:25:13Z] PARSER FLAG OK surface=pool flag=--acs-person-zip source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-23T13:25:13Z] PARSER FLAG OK surface=pool flag=--acs-person-zip-sha256 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-23T13:25:13Z] PARSER FLAG OK surface=pool flag=--acs-rent-h5 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-23T13:25:13Z] PARSER FLAG OK surface=pool flag=--acs-rent-h5-sha256 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-23T13:25:13Z] PARSER FLAG OK surface=pool flag=--puf-h5 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-23T13:25:13Z] PARSER FLAG OK surface=pool flag=--puf-h5-sha256 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-23T13:25:13Z] PARSER FLAG OK surface=pool flag=--puf-source-year-csv source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-23T13:25:13Z] PARSER FLAG OK surface=pool flag=--puf-source-year-csv-sha256 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-23T13:25:13Z] PARSER FLAG OK surface=pool flag=--sample-fraction source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-23T13:25:13Z] PARSER FLAG OK surface=pool flag=--sample-seed source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-23T13:25:13Z] PARSER FLAG OK surface=pool flag=--clone-attachment-fraction source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-23T13:25:13Z] PARSER FLAG OK surface=pool flag=--clone-attachment-seed source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=pool flag=--checkpoint-root source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=pool flag=--out source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--base-h5 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--dense-default-dataset source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--ledger-facts source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--ledger-facts-sha256 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--export-input-mass-reference-h5 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--asec-2023-weeks-unemployed-source source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--scf-summary-extract source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--scf-full-extract source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--sipp-tip-donor source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--sipp-vehicle-donor source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--org-wages-donor source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--ssi-take-up-prior-weight-basis source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--ssi-take-up-prior-weight-basis-sha256 source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--checkpoint-root source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--release-id source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--seed source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--epochs source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--skip-reform-validation source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--no-staging source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=dense flag=--out source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=selection flag=--selection-source-manifest source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=selection flag=--selection-mass-protection source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=scorer flag=--incumbent source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/score_us_release_head_to_head.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=scorer flag=--candidate source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/score_us_release_head_to_head.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=scorer flag=--ledger-facts source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/score_us_release_head_to_head.py
[2026-08-23T13:25:14Z] PARSER FLAG OK surface=scorer flag=--out-prefix source=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/score_us_release_head_to_head.py
[2026-08-23T13:25:14Z] CODE PIN PLANNED commit=6327ec0208803a811f59783723a2cd4df5824ad2 path=/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/code.commit
[2026-08-23T13:25:16Z] INPUT OK role=asec-raw-stage-h5 size=653410492 sha256=51e9fafcd6f16140018fa90c7afbeb6d79008bfc8c122e437d23a399b30553fe path=/Users/maxghenis/PolicyEngine/_buildo-runtime/out/591-pawtyp-pool/asec-producer-checkpoints/asec_raw_stage.checkpoint.h5
[2026-08-23T13:25:17Z] INPUT OK role=acs-household-zip size=251500587 sha256=8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0 path=/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0/csv_hus.zip
[2026-08-23T13:25:19Z] INPUT OK role=acs-person-zip size=602847146 sha256=afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894 path=/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894/csv_pus.zip
[2026-08-23T13:25:21Z] INPUT OK role=acs-rent-h5 size=472220686 sha256=0b319b496f19a6913066f9c5ea572edfda3d78a187be6f375846617d0b441bd4 path=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/acs_2022.h5
[2026-08-23T13:25:22Z] INPUT OK role=puf-h5 size=316939164 sha256=7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df path=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2024.h5
[2026-08-23T13:25:23Z] INPUT OK role=puf-source-year-csv size=126034649 sha256=0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df path=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2015.csv
[2026-08-23T13:25:23Z] INPUT OK role=ledger-v9.4 size=131852600 sha256=b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080 path=/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl
[2026-08-23T13:25:24Z] INPUT OK role=export-input-mass-reference size=354374956 sha256=c2065b642ab00da74746afdfd9f06890e5f32f9b10bd6610ff236452d40f39c5 path=/Users/maxghenis/PolicyEngine/_buildg-runtime/forensics/populace_us_2024.h5
[2026-08-23T13:25:25Z] INPUT OK role=scf-summary-extract size=24904185 sha256=6b8dd2d935a76ed225ddebc80fb2db22a467f0c80d9a1acaa67b4584aa4bafd1 path=/Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/scf_cache/rscfp2022.dta
[2026-08-23T13:25:25Z] INPUT OK role=scf-full-extract size=236952250 sha256=61e2fceb1594e4009eb996d6e25d38a5d8e4874930fc2bfce3c87ffa6946ad0a path=/Users/maxghenis/.cache/microcosm/scf/p22i6.dta
[2026-08-23T13:25:25Z] INPUT HEADER OK role=scf-full-extract format=stata-dta release=118 byteorder=LSF path=/Users/maxghenis/.cache/microcosm/scf/p22i6.dta
[2026-08-23T13:25:25Z] INPUT OK role=ssi-prior-weight-basis size=3957 sha256=56118bde095b8ef2559a26a3478ff5f8b61939eca402dffcec61189e7de631e3 path=/Users/maxghenis/PolicyEngine/_buildo-runtime/out/buildo-run/dense-p2/releases/populace-us-2024-buildp-dense-cae8640-20260728T050443Z/us_ssi_take_up.json
[2026-08-23T13:25:26Z] INPUT OK role=asec-2023-weeks-source size=150165063 sha256=d2e000250782adfbdd7f29c82b66d866591a30f0d330496698ec19f9c784ce11 path=/Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/asec_education/asecpub23csv.zip
[2026-08-23T13:25:40Z] INPUT OK role=sipp-full-donor size=3726010471 sha256=5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2 path=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/pu2023.csv
[2026-08-23T13:25:40Z] INPUT OK role=sipp-tips-donor size=65649523 sha256=1f0bcb8e045ef1118e8eba4b4a2997bdaaf947bd0dd09d41fa7c7d5657a3d7d5 path=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/pu2023_slim.csv
[2026-08-23T13:25:40Z] INPUT OK role=cps-org-wages-compressed size=1677834 sha256=66fa5b6aa4087413b691038767b51f603281ff55411b58259922f78e67460372 path=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/census_cps_org_2024_wages.csv.gz
[2026-08-23T13:25:40Z] INPUT OK role=packaged-cd-crosswalk size=77935 sha256=c7cb040b1f57ca2ea2adcbfe60cc2b250ca23acbc4b640cd421e766fa54c1aec path=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/packages/microcosm-build/src/microcosm/build/us_runtime/data/congressional_district_vintage_crosswalk.csv
[2026-08-23T13:25:40Z] INPUT OK role=incumbent-evidence size=31962593 sha256=b2ad1a07f9668bc5d796cc9de99ef12da781b1ee8163ea65781871a20da441c8 path=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/experiments/replacement_scorecard/incumbent_48b9d479.json
[2026-08-23T13:25:42Z] INPUT OK role=incumbent-h5 size=462915783 sha256=48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e path=/Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/26dcad66867687f15735dc4926523e3741920836/populace_us_2024.h5
[2026-08-23T13:25:42Z] DENSE RELEASE ID value=populace-us-2024-onesurface-pkg3-legacy-dense-6327ec02-20260823T132513Z state_file=/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/release-dense/release_id.txt
[2026-08-23T13:25:42Z] PRECONDITION PLAN stage=pool poll_seconds=300 need_reclaimable_gib=90 checks=no-pool-or-release-builder,AC-power,go-marker:/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/.max-go
COMMAND stage=pool /usr/bin/time -l /usr/bin/env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST /Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/.venv/bin/python /Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_multispine_pool.py --sample-fraction 0.25 --sample-seed 578 --clone-attachment-fraction 1.0 --clone-attachment-seed 578 --asec-raw-stage-h5 /Users/maxghenis/PolicyEngine/_buildo-runtime/out/591-pawtyp-pool/asec-producer-checkpoints/asec_raw_stage.checkpoint.h5 --asec-raw-stage-h5-sha256 51e9fafcd6f16140018fa90c7afbeb6d79008bfc8c122e437d23a399b30553fe --acs-household-zip /Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0/csv_hus.zip --acs-household-zip-sha256 8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0 --acs-person-zip /Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894/csv_pus.zip --acs-person-zip-sha256 afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894 --acs-rent-h5 /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/acs_2022.h5 --acs-rent-h5-sha256 0b319b496f19a6913066f9c5ea572edfda3d78a187be6f375846617d0b441bd4 --puf-h5 /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2024.h5 --puf-h5-sha256 7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df --puf-source-year-csv /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2015.csv --puf-source-year-csv-sha256 0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df --checkpoint-root /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/pool/checkpoints --out /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/pool/pool.h5
[2026-08-23T13:25:42Z] POOL MANIFEST PIN DEFERRED dynamic_output=/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/pool/pool.manifest.json release-wrapper-will-record-and-consume-full-sha256
[2026-08-23T13:25:42Z] PRECONDITION PLAN stage=release-dense poll_seconds=300 need_reclaimable_gib=110 checks=no-pool-or-release-builder,AC-power,go-marker:/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/.max-go
COMMAND stage=release-dense /usr/bin/time -l /usr/bin/env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST /Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/.venv/bin/python /Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/build_us_fiscal_refresh_release.py --base-h5 /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/pool/pool.h5 --dense-default-dataset --ledger-facts /Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl --ledger-facts-sha256 b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080 --export-input-mass-reference-h5 /Users/maxghenis/PolicyEngine/_buildg-runtime/forensics/populace_us_2024.h5 --asec-2023-weeks-unemployed-source /Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/asec_education/asecpub23csv.zip --scf-summary-extract /Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/scf_cache/rscfp2022.dta --scf-full-extract /Users/maxghenis/.cache/microcosm/scf/p22i6.dta --sipp-tip-donor /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/pu2023_slim.csv --sipp-vehicle-donor /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/pu2023.csv --org-wages-donor /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/census_cps_org_2024_wages.csv.gz --ssi-take-up-prior-weight-basis /Users/maxghenis/PolicyEngine/_buildo-runtime/out/buildo-run/dense-p2/releases/populace-us-2024-buildp-dense-cae8640-20260728T050443Z/us_ssi_take_up.json --ssi-take-up-prior-weight-basis-sha256 56118bde095b8ef2559a26a3478ff5f8b61939eca402dffcec61189e7de631e3 --seed 0 --epochs 3000 --checkpoint-root /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/release-dense/checkpoints --release-id populace-us-2024-onesurface-pkg3-legacy-dense-6327ec02-20260823T132513Z --out /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/release-dense --skip-reform-validation --no-staging
[2026-08-23T13:25:42Z] STAGE 2b SPARSE STOP: no selection derivation or sparse release command will run
[2026-08-23T13:25:42Z] OWNER QUESTION: which rule may choose the candidate pool support: (A) current legacy fixed-penalty L0 at default 0.8, accepting its non-exact realized count, or (B) a newly ratified exact-57,240 rule, including algorithm, seed, and Keogh-carrier inclusion policy? If B uses current exact-k, supply pi_hi and its artifact pins. Current main has no legacy CLI that derives an exact 57,240 selection manifest from a pool.
[2026-08-23T13:25:42Z] SPARSE PENDING owner ruling on candidate-pool selection doctrine
[2026-08-23T13:25:42Z] DENSE ARTIFACT planned_path=/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/release-dense/artifacts/populace_us_2024.h5 sha256=pending-stage-2a
[2026-08-23T13:25:42Z] SPARSE ARTIFACT not-produced; sparse is pending the owner ruling above
[2026-08-23T13:25:42Z] INCUMBENT EVIDENCE path=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/experiments/replacement_scorecard/incumbent_48b9d479.json sha256=b2ad1a07f9668bc5d796cc9de99ef12da781b1ee8163ea65781871a20da441c8 (evidence only; scorer consumes the H5)
SCORER COMMAND dense /usr/bin/env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST /Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/.venv/bin/python /Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/tools/score_us_release_head_to_head.py --incumbent /Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/26dcad66867687f15735dc4926523e3741920836/populace_us_2024.h5 --candidate /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/release-dense/artifacts/populace_us_2024.h5 --ledger-facts /Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl --out-prefix /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/scores/dense-head-to-head
[2026-08-23T13:25:42Z] DRY-RUN COMPLETE no builder, selection tool, release, scorer, publication, promotion, staging, or launchd mutation ran
```

## Syntax and side-effect checks

```text
$ bash -n experiments/candidate_25pct/run-candidate.sh
exit 0

$ test -e /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25
exit 1
```

- No pool or release command was invoked.
- No output directory, checkpoint, state pin, log, RSS CSV, or launchd label was
  created or changed.
- No publication, promotion, push, tuning, or scorer run occurred.
- `POPULACE_LOGBOOK_PREV_ROW_DIGEST` remained unset and no pending-chain state
  was read or written.
