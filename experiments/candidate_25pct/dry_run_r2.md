# Candidate 25% dry-run receipt, round 2

Date: 2026-08-23

Status: **NOT RUN — ordered recovery stopped at input 1**

Chronicle at the provenance-pinned v9.4 commit
`0575510d93ec972f4348e0631315431d1e0ffb9b` rejects the reviewed 37,405-row
feed before it can write a consumer artifact: row 1 lacks the required
`assertion` field. The exact exporter validates rows before writing its facts
and manifest (`policyengine_ledger/consumer.py:642-699` at that commit), and
its packaged schema requires `assertion`, `provenance_class`, the Ledger schema
version, and Ledger identity prefixes
(`policyengine_ledger/schemas/consumer_fact.v1.schema.json:7-29,57-91` at that
commit).

The earliest historical artifact writer that can read the mixed legacy feed
rewrites it from the reviewed SHA-256
`b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080`
to `f455145f07a3047a325effc957f0d5dc8d4e317e96fec594a5625ef30e20cff6`.
It inserts default assertions and reserializes JSON
(`policyengine_ledger/consumer.py:603-658,714-728` at commit
`e462f285bb99cd81c04a01a68368e52f3c67a7c5`). That output was rejected and
remained under `/private/tmp`; it was not installed or pinned.

Exact-k requires both the facts and manifest SHA-256 pins
(`tools/build_us_fiscal_refresh_release.py:1563-1566`). Because no
Chronicle-produced manifest can truthfully pin the required facts bytes, there
is no complete stage-2 command to print or validate. The required
`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/run-candidate.sh`
was not written, so `bash -n`, the script `--dry-run`, and a release-builder
parse-time no-op were not run.

The mode ruling also lacks a non-fabricated `pi_hi`: current authority declares
it required with `default: null`
(`packages/microcosm-build/src/microcosm/build/us/spec/selection.yaml:19-31`),
while `0.95` is only an example and the launcher requires an explicit value
(`tools/build_us_exact_k_ladder_release.py:16-27,151-200`). The July incumbent
used `--dense-default-dataset --seed 0` and no `pi_hi`
(`/Users/maxghenis/PolicyEngine/_buildo-runtime/scripts/buildp_sparse9.sh:123-143`).

Full evidence and the ordered-stop disposition are in
`experiments/candidate_25pct/input_audit_r2.md`.
