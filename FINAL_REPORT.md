# Final report: 25% candidate runbook, round 2

Date: 2026-08-23

Branch: `candidate-25pct-runbook`

Outcome: **STOPPED at ordered input 1; no artifact or run script was
fabricated**

## Outcome

Round 2 could not recover a Chronicle consumer-artifact directory containing
the pinned v9.4 facts bytes. The host provenance identifies Ledger commit
`0575510d93ec972f4348e0631315431d1e0ffb9b`, 37,405 rows, and SHA-256
`b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080`
(`/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9.PROVENANCE.md:55-60`).

The supported `build-consumer-artifact` command from an archived snapshot of
that exact commit rejects row 1 before producing any output. Its builder
schema-validates facts before writing the artifact
(`policyengine_ledger/consumer.py:642-699` at commit `0575510d93ec`), while the
packaged schema requires `assertion`, `provenance_class`,
`ledger.consumer_fact.v1`, and Ledger-prefixed identity keys
(`policyengine_ledger/schemas/consumer_fact.v1.schema.json:7-29,57-91` at the
same commit). Measured feed composition is:

```text
37006 rows: no assertion, no provenance_class, arch.consumer_fact.v1
  388 rows: assertion, no provenance_class, ledger.consumer_fact.v1
   11 rows: assertion and provenance_class, ledger.consumer_fact.v1
```

The earliest historical artifact writer capable of reading the mixed legacy
rows corroborates the byte incompatibility. It inserts default assertions and
reserializes JSON (`policyengine_ledger/consumer.py:603-658,714-728` at commit
`e462f285bb99cd81c04a01a68368e52f3c67a7c5`), producing:

```text
reviewed input:  131852600 bytes  b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080
rejected output: 137948301 bytes  f455145f07a3047a325effc957f0d5dc8d4e317e96fec594a5625ef30e20cff6
first byte difference: offset 23
```

That output stayed in `/private/tmp` and was not substituted. Copying the
reviewed bytes into the generated directory and changing its manifest would
not be producer output. Microcosm requires `manifest.json` plus
`consumer_facts.jsonl` and verifies `manifest.facts_sha256` against the facts
bytes (`packages/microcosm-build/src/microcosm/build/ledger_artifact.py:88-142`);
exact-k requires both Ledger pins
(`tools/build_us_fiscal_refresh_release.py:1563-1566`). Consequently no valid
artifact path or manifest SHA-256 was recorded, and the requested
`.../candidate-25/inputs/ledger-v9_4/` directory was not created.

## Exact-k mode ruling

The owner-stated `--seed 0` is defensible for comparison and remains pending
Max's ratification. The July incumbent passed `--seed 0`
(`/Users/maxghenis/PolicyEngine/_buildo-runtime/scripts/buildp_sparse9.sh:123-143`),
and current code preserves `0` as the pre-exact-k legacy default
(`tools/build_us_fiscal_refresh_release.py:1581-1583`).

There is no defensible default or July-incumbent value for `pi_hi`:

- current selection authority says `required: true` and `default: null`
  (`packages/microcosm-build/src/microcosm/build/us/spec/selection.yaml:19-31`);
- its generator explicitly forbids minting exact-k run values
  (`tools/us_bundle_generation/contracts.py:1-12,1508-1544`);
- the builder requires an explicit `--exact-k-pi-hi`
  (`tools/build_us_fiscal_refresh_release.py:857-869,1500-1528`);
- the July incumbent used `--dense-default-dataset`, not exact-k, and supplied
  no `pi_hi`
  (`/Users/maxghenis/PolicyEngine/_buildo-runtime/scripts/buildp_sparse9.sh:123-143`);
- `0.95` is only a launcher example; the actual config schema requires the
  caller to provide the value
  (`tools/build_us_exact_k_ladder_release.py:16-27,151-200`).

The builder correctly forbids combining exact-k with the legacy dense arm
(`tools/build_us_fiscal_refresh_release.py:1568-1579`). Per the owner's
explicit ruling, `0.95` was not promoted from an example into a run pin. This
is a second independent stop even if the Ledger artifact is later resolved.

## Ordered disposition

Recovery stopped at input 1 as required. Therefore this round did not:

- download or import-check SCF `p22i6.dta`;
- generate current-surface incumbent diagnostics;
- compute a frozen target-surface digest;
- write `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/run-candidate.sh`;
- run `bash -n`, a script `--dry-run`, or a builder parse-time no-op;
- run a pool or release build;
- publish, promote, push, or touch pending logbook-chain state.

No later missing input was represented as recovered. The complete evidence is
in `experiments/candidate_25pct/input_audit_r2.md`; the committed non-run
receipt is `experiments/candidate_25pct/dry_run_r2.md`. `PROGRESS.md` carries
the current state/done/next handoff.

## Validation and commits

- Exact-commit Chronicle export: failed closed on row 1 before artifact output,
  as required by its schema-validation path.
- Historical compatibility export: completed only under `/private/tmp`; SHA
  mismatch and byte difference measured; output rejected.
- `git diff --check`: passed for both the evidence commit and this final report.
- Python tests and linters: not run; this round changed Markdown journals only,
  and the ordered stop forbade progressing into builder validation.

Round-2 commits before this report:

- `4948d9eb` — start round-two candidate input recovery.
- `5916c91a` — record the unreproducible Ledger artifact stop and dry-run
  non-receipt.

No push was made.

## Required handoff

The owner must ratify both:

1. a reviewed Chronicle-native feed/artifact with a new facts pin, or a
   producer-owned compatibility exporter that preserves the mixed legacy
   `b3c08356…` bytes and emits a valid manifest; and
2. an exact-k `pi_hi` run-request value.

After both are resolved, restart ordered recovery at the Ledger artifact. Do
not resume at SCF and do not synthesize a partial command from the missing
pins.
